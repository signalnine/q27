// DFlash2 drafter runtime -- see src/dflash2.h. The forward mirrors z-lab
// dflash/model.py operation for operation (parity gates: tools/dflash2_smoke
// vs bench/dflash2/p1_qtap_al.py, and the --dflash2 byte-identity matrix).
// Phase 2: batched ingest (fc read once per round), on-device top-16 +
// selector walk (no per-round D2H), width runtime-selectable to D2_WMAX.
#include "dflash2.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

#include "blocks.cuh"  // rmsnorm_heads, rope_neox_partial
#include "kernels.cuh" // gemv_f16, rmsnorm, P3/CP3, gemv_f16_3, rmsnorm3, add3, silu_mul3
#include "spec3.cuh"   // IP3, rope3, batched decls

#define D2CHECK(x)                                                                             \
    do {                                                                                       \
        cudaError_t err__ = (x);                                                               \
        if (err__ != cudaSuccess) {                                                            \
            fprintf(stderr, "dflash2 CUDA error %s at %s:%d\n", cudaGetErrorString(err__),     \
                    __FILE__, __LINE__);                                                       \
            abort();                                                                           \
        }                                                                                      \
    } while (0)

namespace q27d2 {

using q27k::CP3;
using q27k::IP3;
using q27k::P3;

static P3 mkP3(float* base, size_t stride, int n) {
    P3 p{};
    for (int i = 0; i < 16; i++) p.p[i] = base + stride * (size_t)(i < n ? i : 0);
    return p;
}
static CP3 mkCP3(const float* base, size_t stride, int n) {
    CP3 p{};
    for (int i = 0; i < 16; i++) p.p[i] = base + stride * (size_t)(i < n ? i : 0);
    return p;
}

// ---- kernels ------------------------------------------------------------

// fp16 matrix row -> fp32 vector (embedding lookup with the row known host-side)
__global__ void k_d2_rowcast(const __half* __restrict__ src, float* __restrict__ dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        dst[i] = __half2float(src[i]);
}

// Grouped dynamic causal conv (z-lab _grouped_dynamic_convolve), R rows of H.
// out[r][c] = sum_{o<K} (base[o][c] + dyn[r][s][o][c/16]) * in[r-o][c], in[<0]=0.
// dyn layout per row: [2][K][G] (kernel_projection output view); s selects the
// prepare (0) / finish (1) slice; base points at base_kernel[s] ([K][H] fp32).
__global__ void k_d2_dconv(const float* __restrict__ in, const float* __restrict__ dyn,
                           const float* __restrict__ base, float* __restrict__ out, int rows,
                           int s) {
    int r = blockIdx.y;
    if (r >= rows) return;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < D2_H; c += gridDim.x * blockDim.x) {
        int g = c / 16;
        float acc = 0.f;
        for (int o = 0; o < D2_CONVK; o++) {
            if (r - o < 0) break;
            float kv = base[o * D2_H + c] +
                       dyn[(size_t)r * (2 * D2_CONVK * D2_CONVG) + s * (D2_CONVK * D2_CONVG) +
                           o * D2_CONVG + g];
            acc += kv * in[(size_t)(r - o) * D2_H + c];
        }
        out[(size_t)r * D2_H + c] = acc;
    }
}

// Bidirectional sliding-window attention: nrows noise queries vs [ctx ring +
// nrows noise] keys. Block = (query row, q head); two-pass softmax in smem.
// GQA: q head h reads kv head h/4. ctx_n is read through a device pointer so
// a future graph capture stays shape-stable. Bring-up kernel -- clarity over
// speed.
__global__ void k_d2_attn(const float* __restrict__ q, const float* __restrict__ ringK,
                          const float* __restrict__ ringV, const int* __restrict__ ring_pos,
                          const int* __restrict__ d_ctx_n, const float* __restrict__ nk,
                          const float* __restrict__ nv, const int* __restrict__ npos,
                          float* __restrict__ out, int nrows) {
    extern __shared__ float sc[]; // scores [ctx_n + nrows]
    const int r = blockIdx.x, h = blockIdx.y, kh = h / (D2_NH / D2_NKV);
    if (r >= nrows) return;
    const int ctx_n = *d_ctx_n;
    const float scale = rsqrtf((float)D2_HD);
    const float* qv = q + (size_t)r * D2_QD + (size_t)h * D2_HD;
    const int qpos = npos[r], total = ctx_n + nrows;
    for (int j = threadIdx.x; j < total; j += blockDim.x) {
        const float* kv;
        int kpos;
        if (j < ctx_n) {
            kv = ringK + (size_t)j * D2_KVD + (size_t)kh * D2_HD;
            kpos = ring_pos[j];
        } else {
            kv = nk + (size_t)(j - ctx_n) * D2_KVD + (size_t)kh * D2_HD;
            kpos = npos[j - ctx_n];
        }
        bool vis = (qpos - kpos) < D2_WINDOW && (kpos - qpos) < D2_WINDOW;
        float d = 0.f;
        for (int c = 0; c < D2_HD; c++) d += qv[c] * kv[c];
        sc[j] = vis ? d * scale : -INFINITY;
    }
    __syncthreads();
    __shared__ float s_max, s_sum;
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int j = 0; j < total; j++) m = fmaxf(m, sc[j]);
        float sum = 0.f;
        for (int j = 0; j < total; j++) sum += expf(sc[j] - m);
        s_max = m;
        s_sum = sum;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < total; j += blockDim.x) sc[j] = expf(sc[j] - s_max) / s_sum;
    __syncthreads();
    for (int c = threadIdx.x; c < D2_HD; c += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < total; j++) {
            const float* vv = j < ctx_n
                                  ? ringV + (size_t)j * D2_KVD + (size_t)kh * D2_HD
                                  : nv + (size_t)(j - ctx_n) * D2_KVD + (size_t)kh * D2_HD;
            acc += sc[j] * vv[c];
        }
        out[(size_t)r * D2_QD + (size_t)h * D2_HD + c] = acc;
    }
}

// Per-row top-16 over the vocab, two stages so the whole grid participates
// (the one-block iterative version measured 4 ms/round). Tie semantics = the
// host scan's (equal values: lowest vocab id ranks first) -- stage 2 orders
// by (value desc, id asc), and any global top-16 element is necessarily in
// its slice's local top-16, so the result is exact.
constexpr int D2_T16B = 512; // stage-1 blocks per row

__global__ void k_d2_top16a(const float* __restrict__ logits, float* __restrict__ c1v,
                            int* __restrict__ c1i, int rows) {
    const int r = blockIdx.x, b = blockIdx.y;
    if (r >= rows) return;
    const int slice = (D2_V + D2_T16B - 1) / D2_T16B;
    const int v0 = b * slice, v1 = min(v0 + slice, D2_V);
    const float* lg = logits + (size_t)r * D2_V;
    __shared__ float sv[128];
    __shared__ int si[128];
    __shared__ int picked[D2_TOPK];
    for (int it = 0; it < D2_TOPK; it++) {
        float best = -INFINITY;
        int besti = -1;
        for (int v = v0 + threadIdx.x; v < v1; v += blockDim.x) {
            bool skip = false;
            for (int p = 0; p < it; p++) skip |= (picked[p] == v);
            if (skip) continue;
            float x = lg[v];
            if (x > best || (x == best && v < besti)) { best = x; besti = v; }
        }
        sv[threadIdx.x] = best;
        si[threadIdx.x] = besti;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                float xo = sv[threadIdx.x + s];
                int io = si[threadIdx.x + s];
                if (xo > sv[threadIdx.x] ||
                    (xo == sv[threadIdx.x] && io != -1 &&
                     (si[threadIdx.x] == -1 || io < si[threadIdx.x]))) {
                    sv[threadIdx.x] = xo;
                    si[threadIdx.x] = io;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            picked[it] = si[0];
            c1v[((size_t)r * D2_T16B + b) * D2_TOPK + it] = sv[0];
            c1i[((size_t)r * D2_T16B + b) * D2_TOPK + it] = si[0];
        }
        __syncthreads();
    }
}

__global__ void k_d2_top16b(const float* __restrict__ c1v, const int* __restrict__ c1i,
                            int* __restrict__ cand, float* __restrict__ cval, int rows) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int N = D2_T16B * D2_TOPK;
    const float* cv = c1v + (size_t)r * N;
    const int* ci = c1i + (size_t)r * N;
    __shared__ float sv[256];
    __shared__ int si[256];
    __shared__ int picked[D2_TOPK]; // candidate-array slots already taken
    for (int it = 0; it < D2_TOPK; it++) {
        float best = -INFINITY;
        int besti = -1; // slot in the candidate array
        for (int j = threadIdx.x; j < N; j += blockDim.x) {
            bool skip = false;
            for (int p = 0; p < it; p++) skip |= (picked[p] == j);
            if (skip) continue;
            float x = cv[j];
            if (x > best ||
                (x == best && besti != -1 && ci[j] < ci[besti]) ||
                (x == best && besti == -1)) {
                best = x;
                besti = j;
            }
        }
        sv[threadIdx.x] = best;
        si[threadIdx.x] = besti;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                float xo = sv[threadIdx.x + s];
                int io = si[threadIdx.x + s];
                bool take = false;
                if (io != -1) {
                    if (si[threadIdx.x] == -1) take = xo > -INFINITY || true;
                    else if (xo > sv[threadIdx.x]) take = true;
                    else if (xo == sv[threadIdx.x] && ci[io] < ci[si[threadIdx.x]]) take = true;
                }
                if (take) {
                    sv[threadIdx.x] = xo;
                    si[threadIdx.x] = io;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            picked[it] = si[0];
            cand[(size_t)r * D2_TOPK + it] = ci[si[0]];
            cval[(size_t)r * D2_TOPK + it] = sv[0];
        }
        __syncthreads();
    }
}

// Selector path walk (z-lab CandidateSelector.select, greedy): one block,
// K sequential positions. score(c) = unary logit + dot(pred_row(prev) *
// hp_row, succ_row(c)); argmax with ties to the lower candidate slot (the
// host walk's strict-> semantics). prev chains; out[pos] = the pick.
// sampled (2026-09-07, ninfer draw_rank semantics): the pick is DRAWN from
// q = softmax(inv_temp * (score - max)) over the 16 candidates with one Philox
// uniform keyed (seed, posW[pos], KIND_D2_PROPOSAL); q is retained in qrow so
// the verify tail can accept with min(1, p/q) and correct from max(p - q, 0).
// Expected lane accept becomes sum_v min(p, q) instead of p(argmax) -- the
// sampled-serving acceptance gap vs ninfer (0.645 vs 0.754 lane-1). The
// greedy branch is the unchanged code path (bitwise).
__global__ void k_d2_walk(const int* __restrict__ cand, const float* __restrict__ cval,
                          const float* __restrict__ hp, const __half* __restrict__ pred,
                          const __half* __restrict__ succ, const int* __restrict__ d_anchor,
                          int K, int* __restrict__ out, int sampled,
                          const q27k::SampleParams* __restrict__ sp,
                          const int* __restrict__ posW, float* __restrict__ qrow) {
    __shared__ float ph[D2_RANK];
    __shared__ float sc[D2_TOPK][256 / 32]; // per-warp partials per candidate
    __shared__ int s_prev;
    if (threadIdx.x == 0) s_prev = *d_anchor; // device-read: graph-stable anchor
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int pos = 0; pos < K; pos++) {
        const int prev = s_prev;
        for (int i = threadIdx.x; i < D2_RANK; i += blockDim.x)
            ph[i] = __half2float(pred[(size_t)prev * D2_RANK + i]) *
                    hp[(size_t)pos * D2_RANK + i];
        __syncthreads();
        // 8 warps x 16 candidates: warp w handles candidates w, w+8
        for (int i = warp; i < D2_TOPK; i += blockDim.x / 32) {
            const __half* sr = succ + (size_t)cand[(size_t)pos * D2_TOPK + i] * D2_RANK;
            float acc = 0.f;
            for (int j = lane; j < D2_RANK; j += 32) acc += ph[j] * __half2float(sr[j]);
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffff, acc, off);
            if (lane == 0) sc[i][0] = acc;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float best = -INFINITY;
            int besti = 0;
            float s[D2_TOPK];
            for (int i = 0; i < D2_TOPK; i++) {
                s[i] = cval[(size_t)pos * D2_TOPK + i] + sc[i][0];
                if (s[i] > best) { best = s[i]; besti = i; }
            }
            if (sampled) {
                const float invT = sp->inv_temp;
                float w[D2_TOPK], sum = 0.f;
                for (int i = 0; i < D2_TOPK; i++) {
                    w[i] = expf(invT * (s[i] - best));
                    sum += w[i];
                }
                const float inv = 1.f / sum;
                const float u = q27k::philox_uniform(sp->seed, (unsigned)posW[pos],
                                                     q27k::KIND_D2_PROPOSAL, 0u);
                float q[D2_TOPK], cdf = 0.f;
                int pick = -1, last_pos = 0;
                for (int i = 0; i < D2_TOPK; i++) {
                    q[i] = w[i] * inv;
                    cdf += q[i];
                    if (q[i] > 0.f) last_pos = i;
                    if (pick < 0 && u < cdf) pick = i;
                }
                // fp32 cdf can end below u (rounding): fall back to the LAST
                // candidate with q > 0, never a zero-probability slot (the
                // tail would otherwise accept an unproposable token with
                // p >= q == 0; gpt-6-astra design review). The stored law
                // must equal the realized draw law, so that slot's q absorbs
                // the cdf tail P(u >= cdf) = 1 - cdf it actually wins.
                if (cdf < 1.f) q[last_pos] += 1.f - cdf;
                for (int i = 0; i < D2_TOPK; i++) qrow[(size_t)pos * D2_TOPK + i] = q[i];
                besti = pick < 0 ? last_pos : pick;
            }
            s_prev = cand[(size_t)pos * D2_TOPK + besti];
            out[pos] = s_prev;
        }
        __syncthreads();
    }
}

void d2_walk_launch(const int* d_cand, const float* d_cval, const float* d_hp,
                    const __half* d_pred, const __half* d_succ, const int* d_anchor, int K,
                    int* d_out, bool sampled, const q27k::SampleParams* d_sp, const int* d_posW,
                    float* d_qrow, cudaStream_t st) {
    k_d2_walk<<<1, 256, 0, st>>>(d_cand, d_cval, d_hp, d_pred, d_succ, d_anchor, K, d_out,
                                 sampled ? 1 : 0, d_sp, d_posW, d_qrow);
    D2CHECK(cudaGetLastError());
}

// ---- pack loader --------------------------------------------------------

void Dflash2::load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "dflash2: cannot open %s\n", path); abort(); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    pack.resize(sz);
    if (fread(pack.data(), 1, sz, f) != (size_t)sz) abort();
    fclose(f);
    const char* p = pack.data();
    assert(!memcmp(p, "D2W2", 4));
    int n;
    memcpy(&n, p + 4, 4);
    size_t off = 8;
    for (int i = 0; i < n; i++) {
        int nlen;
        memcpy(&nlen, p + off, 4);
        off += 4;
        std::string name(p + off, nlen);
        off += nlen;
        D2Tensor t;
        int ndim;
        memcpy(&t.dtype, p + off, 4);
        memcpy(&ndim, p + off + 4, 4);
        off += 8;
        int64_t dims[4] = {1, 1, 1, 1};
        memcpy(dims, p + off, 8 * ndim);
        off += 8 * ndim;
        int64_t nbytes;
        memcpy(&nbytes, p + off, 8);
        off += 8;
        off += (64 - (off % 64)) % 64;
        D2CHECK(cudaMalloc(&t.dev, nbytes));
        D2CHECK(cudaMemcpy(t.dev, p + off, nbytes, cudaMemcpyHostToDevice));
        off += nbytes;
        t.rows = dims[0];
        t.cols = ndim > 1 ? dims[1] : 1;
        for (int d = 2; d < ndim; d++) t.cols *= dims[d];
        t.n = t.rows * t.cols;
        if (t.dtype == 2 || t.dtype == 3) { // q4_g64 / q8_g128: scales blob follows
            int64_t sbytes;
            memcpy(&sbytes, p + off, 8);
            off += 8;
            off += (64 - (off % 64)) % 64;
            D2CHECK(cudaMalloc(&t.dscales, sbytes));
            D2CHECK(cudaMemcpy(t.dscales, p + off, sbytes, cudaMemcpyHostToDevice));
            off += sbytes;
        } else {
            t.host = p + off - nbytes; // codebooks read host-side by the walk
        }
        w[name] = t;
    }
    assert(T("fc.weight").rows == D2_H && T("fc.weight").cols == D2_TAPD);
    assert(T("candidate_selector.predecessor_codebook").rows == D2_V);
}

// Quantized matmul mirroring the engine's qx5+mm5: quantize the W-column
// activation once (g32, eo/isum), then gemv_q4_n/q8_n reads each weight ONCE
// and shares it across all W columns. act is [W][cols] contiguous.
void Dflash2::mmq(const std::string& name, const float* act, float* out, int W,
                  cudaStream_t st) {
    const D2Tensor& t = T(name);
    const int cols = (int)t.cols, rows = (int)t.rows;
    q27k::XQ3 xq{};
    for (int i = 0; i < 16; i++) xq.q[i] = dxq[i < W ? i : 0];
    q27k::quantize3(mkCP3(act, cols, W), cols, xq, st, W);
    if (W == 1) { // batched kernels need nbatch >= 2; single column (n=1 ingest)
        if (t.dtype == 2)
            q27k::gemv_q4((const uint8_t*)t.dev, (const __half*)t.dscales, dxq[0], out, rows,
                          cols, st);
        else
            q27k::gemv_q8((const int8_t*)t.dev, (const __half*)t.dscales, dxq[0], out, rows,
                          cols, st);
        return;
    }
    q27k::XQuant qs[16];
    float* ysa[16];
    for (int i = 0; i < 16; i++) {
        qs[i] = dxq[i < W ? i : 0];
        ysa[i] = out + (size_t)(i < W ? i : 0) * rows;
    }
    if (t.dtype == 2)
        q27k::gemv_q4_n((const uint8_t*)t.dev, (const __half*)t.dscales, qs, W, ysa, rows,
                        cols, st);
    else
        q27k::gemv_q8_n((const int8_t*)t.dev, (const __half*)t.dscales, qs, W, ysa, rows, cols,
                        st);
}

void Dflash2::quant_act(const float* act, int cols, int W, cudaStream_t st) {
    q27k::XQ3 xq{};
    for (int i = 0; i < 16; i++) xq.q[i] = dxq[i < W ? i : 0];
    q27k::quantize3(mkCP3(act, cols, W), cols, xq, st, W);
}

// gemv using the activation already quantized into dxq by quant_act. W >= 2
// (the shared-activation groups in draft_compute are always width K+1 >= 2).
void Dflash2::mmq_pre(const std::string& name, float* out, int W, cudaStream_t st) {
    const D2Tensor& t = T(name);
    const int cols = (int)t.cols, rows = (int)t.rows;
    q27k::XQuant qs[16];
    float* ysa[16];
    for (int i = 0; i < 16; i++) {
        qs[i] = dxq[i < W ? i : 0];
        ysa[i] = out + (size_t)(i < W ? i : 0) * rows;
    }
    if (t.dtype == 2)
        q27k::gemv_q4_n((const uint8_t*)t.dev, (const __half*)t.dscales, qs, W, ysa, rows,
                        cols, st);
    else
        q27k::gemv_q8_n((const int8_t*)t.dev, (const __half*)t.dscales, qs, W, ysa, rows, cols,
                        st);
}

void Dflash2::alloc(int cap) {
    ctx_cap = cap;
    for (int l = 0; l < D2_LAYERS; l++) {
        D2CHECK(cudaMalloc(&ringK[l], (size_t)cap * D2_KVD * 4));
        D2CHECK(cudaMalloc(&ringV[l], (size_t)cap * D2_KVD * 4));
    }
    D2CHECK(cudaMalloc(&d_ring_pos, cap * 4));
    D2CHECK(cudaMalloc(&s_fc, (size_t)D2_WMAX * D2_H * 4));
    D2CHECK(cudaMalloc(&s_ct, (size_t)D2_WMAX * D2_H * 4));
    D2CHECK(cudaMalloc(&nx, (size_t)D2_WMAX * D2_H * 4));
    D2CHECK(cudaMalloc(&nh1, (size_t)D2_WMAX * D2_H * 4));
    D2CHECK(cudaMalloc(&ny0, (size_t)D2_WMAX * D2_H * 4));
    D2CHECK(cudaMalloc(&ndyn, (size_t)D2_WMAX * 2 * D2_CONVK * D2_CONVG * 4));
    D2CHECK(cudaMalloc(&nq, (size_t)D2_WMAX * D2_QD * 4));
    D2CHECK(cudaMalloc(&nk, (size_t)D2_WMAX * D2_KVD * 4));
    D2CHECK(cudaMalloc(&nv, (size_t)D2_WMAX * D2_KVD * 4));
    D2CHECK(cudaMalloc(&natt, (size_t)D2_WMAX * D2_QD * 4));
    D2CHECK(cudaMalloc(&no, (size_t)D2_WMAX * D2_H * 4));
    D2CHECK(cudaMalloc(&ngate, (size_t)D2_WMAX * D2_I * 4));
    D2CHECK(cudaMalloc(&nup, (size_t)D2_WMAX * D2_I * 4));
    D2CHECK(cudaMalloc(&nhf, (size_t)(D2_WMAX - 1) * D2_H * 4));
    D2CHECK(cudaMalloc(&nlogits, (size_t)(D2_WMAX - 1) * D2_V * 4));
    D2CHECK(cudaMalloc(&nhp, (size_t)(D2_WMAX - 1) * D2_RANK * 4));
    D2CHECK(cudaMalloc(&maskrow, D2_H * 4));
    D2CHECK(cudaMalloc(&d_posW, D2_WMAX * 4));
    D2CHECK(cudaMalloc(&d_ing_pos, 4096 * 4));
    D2CHECK(cudaMalloc(&d_cand, (size_t)(D2_WMAX - 1) * D2_TOPK * 4));
    D2CHECK(cudaMemset(d_cand, 0, (size_t)(D2_WMAX - 1) * D2_TOPK * 4)); // valid ids pre-warm
    D2CHECK(cudaMalloc(&d_cval, (size_t)(D2_WMAX - 1) * D2_TOPK * 4));
    D2CHECK(cudaMalloc(&d_qrow, (size_t)(D2_WMAX - 1) * D2_TOPK * 4));
    D2CHECK(cudaMemset(d_qrow, 0, (size_t)(D2_WMAX - 1) * D2_TOPK * 4));
    D2CHECK(cudaMalloc(&d_prop, (size_t)(D2_WMAX - 1) * 4));
    D2CHECK(cudaMalloc(&d_c1v, (size_t)(D2_WMAX - 1) * D2_T16B * D2_TOPK * 4));
    D2CHECK(cudaMalloc(&d_c1i, (size_t)(D2_WMAX - 1) * D2_T16B * D2_TOPK * 4));
    // device scalar mirror of ctx_n for the attention kernel (graph-stable);
    // zeroed so capture_draft's warm run (before any ingest/draft uploads
    // the live count) attends over an EMPTY ring, not garbage.
    D2CHECK(cudaMalloc(&d_ctx_n, 4));
    D2CHECK(cudaMemset(d_ctx_n, 0, 4));
    for (int i = 0; i < D2_WMAX - 1; i++) hxq[i] = q27k::xquant_alloc(D2_H);
    // drafter activation-quant scratch, sized to the widest activation (the
    // fc input = concatenated taps, TAPD=25600)
    for (int i = 0; i < D2_WMAX; i++) dxq[i] = q27k::xquant_alloc(D2_TAPD);
    // anchor-token device buffer (always: the selector walk reads it on
    // device so the drafter forward is graph-capturable)
    D2CHECK(cudaMalloc(&d_anchor_tok, 4));
    // cache the mask-token embedding once (engine Q8 embed if set, else the
    // packed fp16 target.embed)
    if (eembed_data) {
        D2CHECK(cudaMalloc(&d_mask_tok, 4));
        const int mtok = D2_MASK;
        D2CHECK(cudaMemcpy(d_mask_tok, &mtok, 4, cudaMemcpyHostToDevice));
        q27k::embed_row_q8(eembed_data, eembed_scales, d_mask_tok, D2_H, maskrow, 0);
    } else {
        k_d2_rowcast<<<40, 256>>>(f16("target.embed.weight") + (size_t)D2_MASK * D2_H, maskrow,
                                  D2_H);
    }
    D2CHECK(cudaDeviceSynchronize());
}

// ---- context ingest (batched: fc weight read once for all T rows) -------

void Dflash2::ingest(const float* d_taps, const int* h_pos, int T, cudaStream_t st) {
    assert(T <= 4096);
    if (T <= 0) return;
    // Coverage bookkeeping (rollback_to relies on contiguous positions). A
    // chunk that does not continue at ctx_end is a prefill seed window that
    // starts past the retained rows (base < NP - D2_SEED_WINDOW): every
    // retained row is then older than the window's start, i.e. > D2_WINDOW
    // behind every position the drafter will ever query from here, so
    // dropping them is lossless and keeps the ring contiguous (a latched
    // "not contiguous" flag would reset a valid ring on the NEXT turn;
    // gpt-6-astra ring review). A gap INSIDE the chunk never happens (all
    // callers pass consecutive positions) and is treated the same way.
    if (ctx_n > 0 && (h_pos[0] != ctx_end || h_pos[T - 1] - h_pos[0] != T - 1)) {
        static const bool dbg = getenv("Q27_D2_DEBUG") != nullptr;
        if (dbg)
            fprintf(stderr, "[d2] ring ingest NOT contiguous: pos %d..%d (T=%d) after ctx_end %d "
                            "(ctx_n %d) -> dropping retained rows\n", h_pos[0], h_pos[T - 1], T,
                    ctx_end, ctx_n);
        ctx_n = 0;
    }
    if (h_pos[T - 1] - h_pos[0] != T - 1) ctx_contig = false; // internal gap: never expected
    // Slide the ring: keep only the rows that can still fall inside the
    // drafter's window once this chunk lands -- D2_WINDOW - T of them (the
    // attention masks anything older than D2_WINDOW behind the query, so
    // dropping the rest is lossless). With ctx_cap == 2 * D2_WINDOW that
    // bound also makes the compaction copy overlap-free: the slide fires at
    // ctx_n > ctx_cap - T, so src0 = ctx_n - keep > D2_WINDOW >= keep (the
    // old keep = D2_WINDOW could copy [2047,4095) onto [0,2048) -- an
    // overlapping cudaMemcpyAsync, undefined; gpt-6-astra ring review).
    if (ctx_n + T > ctx_cap) {
        const int keep = std::min(ctx_n, std::max(0, D2_WINDOW - T));
        const int src0 = ctx_n - keep;
        assert(src0 >= keep); // no overlap between the copied tail and its destination
        if (src0 > 0 && keep > 0) {
            for (int l = 0; l < D2_LAYERS; l++) {
                D2CHECK(cudaMemcpyAsync(ringK[l], ringK[l] + (size_t)src0 * D2_KVD,
                                        (size_t)keep * D2_KVD * 4, cudaMemcpyDeviceToDevice,
                                        st));
                D2CHECK(cudaMemcpyAsync(ringV[l], ringV[l] + (size_t)src0 * D2_KVD,
                                        (size_t)keep * D2_KVD * 4, cudaMemcpyDeviceToDevice,
                                        st));
            }
            D2CHECK(cudaMemcpyAsync(d_ring_pos, d_ring_pos + src0, keep * 4,
                                    cudaMemcpyDeviceToDevice, st));
        }
        ctx_n = keep;
    }
    D2CHECK(cudaMemcpyAsync(d_ing_pos, h_pos, T * 4, cudaMemcpyHostToDevice, st));
    D2CHECK(cudaMemcpyAsync(d_ring_pos + ctx_n, h_pos, T * 4, cudaMemcpyHostToDevice, st));
    for (int c0 = 0; c0 < T; c0 += D2_WMAX) {
        const int n = std::min(D2_WMAX, T - c0);
        mmq("fc.weight", d_taps + (size_t)c0 * D2_TAPD, s_fc, n, st);
        q27k::rmsnorm3(mkCP3(s_fc, D2_H, n), f32("hidden_norm.weight"), mkP3(s_ct, D2_H, n),
                       D2_H, D2_EPS, st, n);
        // all 10 K/V projections read the SAME s_ct -- quantize it once (n>=2)
        if (n >= 2) quant_act(s_ct, D2_H, n, st);
        for (int l = 0; l < D2_LAYERS; l++) {
            char nm[64];
            // ring rows for this chunk are contiguous [n][KVD] from base+ctx_n+c0
            float* kr0 = ringK[l] + (size_t)(ctx_n + c0) * D2_KVD;
            float* vr0 = ringV[l] + (size_t)(ctx_n + c0) * D2_KVD;
            P3 kr{};
            for (int i = 0; i < 16; i++) kr.p[i] = kr0 + (size_t)(i < n ? i : 0) * D2_KVD;
            snprintf(nm, sizeof nm, "layers.%d.self_attn.k_proj.weight", l);
            if (n >= 2) mmq_pre(nm, kr0, n, st); else mmq(nm, s_ct, kr0, n, st);
            snprintf(nm, sizeof nm, "layers.%d.self_attn.k_norm.weight", l);
            q27k::rmsnorm_heads(kr0, f32(nm), kr0, n * D2_NKV, D2_HD, D2_HD, D2_EPS, st);
            {
                IP3 ip{};
                for (int i = 0; i < 16; i++) ip.p[i] = d_ing_pos + c0 + (i < n ? i : 0);
                q27k::rope3(kr, D2_NKV, D2_HD, D2_HD, D2_HD, ip, D2_THETA, st, n);
            }
            snprintf(nm, sizeof nm, "layers.%d.self_attn.v_proj.weight", l);
            if (n >= 2) mmq_pre(nm, vr0, n, st); else mmq(nm, s_ct, vr0, n, st);
        }
    }
    ctx_n += T;
    ctx_end = h_pos[T - 1] + 1;
    D2CHECK(cudaMemcpyAsync(d_ctx_n, &ctx_n, 4, cudaMemcpyHostToDevice, st));
}

// ---- draft block --------------------------------------------------------

void Dflash2::draft(int anchor_token, int anchor_pos, int K, cudaStream_t st, int* out_host,
                    bool sampling) {
    assert(K >= 1 && K < D2_WMAX);
    const int W = K + 1;
    // per-round H2D (NEVER inside the captured graph): anchor token + positions
    // + the ring-count mirror. The count was only uploaded by ingest(), so a
    // host-side rollback() (round truncation) left the NEXT draft attending
    // discarded rows until the following ingest (gpt-6-astra completion
    // review 2026-09-07, P2). Refreshing here makes every draft see the
    // current host count unconditionally -- 4 bytes/round.
    D2CHECK(cudaMemcpyAsync(d_ctx_n, &ctx_n, 4, cudaMemcpyHostToDevice, st));
    D2CHECK(cudaMemcpyAsync(d_anchor_tok, &anchor_token, 4, cudaMemcpyHostToDevice, st));
    int hpos[D2_WMAX];
    for (int i = 0; i < W; i++) hpos[i] = anchor_pos + i;
    D2CHECK(cudaMemcpyAsync(d_posW, hpos, W * 4, cudaMemcpyHostToDevice, st));
    if (!eembed_data) // eager fallback (CLI): anchor row from the fp16 embed
        k_d2_rowcast<<<40, 256, 0, st>>>(
            f16("target.embed.weight") + (size_t)anchor_token * D2_H, nx, D2_H);
    const bool samp = sampling && d_sp; // sampled walk needs the engine sampler params
    if (draft_exec && draft_exec_k == K && (!samp || draft_exec_s))
        CUDA_CHECK(cudaGraphLaunch(samp ? draft_exec_s : draft_exec, st));
    else
        draft_compute(K, st, samp);
    if (out_host) {
        D2CHECK(cudaMemcpyAsync(out_host, d_prop, K * 4, cudaMemcpyDeviceToHost, st));
        D2CHECK(cudaStreamSynchronize(st));
    }
}

// Capture the drafter compute once and reuse it (replaces ~50 eager launches
// per round with one graph launch). Only when the engine Q8 embed is set (the
// anchor lookup + walk read d_anchor_tok on device); the CLI path stays eager.
void Dflash2::capture_draft(int K, cudaStream_t st) {
    if (!eembed_data) return;
    // seed the per-round device inputs so the warm run + capture are valid
    // (anchor, positions, AND the ring count the attention kernel reads)
    const int a0 = 0, p0[D2_WMAX] = {0};
    D2CHECK(cudaMemcpyAsync(d_ctx_n, &ctx_n, 4, cudaMemcpyHostToDevice, st));
    D2CHECK(cudaMemcpyAsync(d_anchor_tok, &a0, 4, cudaMemcpyHostToDevice, st));
    D2CHECK(cudaMemcpyAsync(d_posW, p0, (K + 1) * 4, cudaMemcpyHostToDevice, st));
    draft_compute(K, st); // warm (init lazy state)
    D2CHECK(cudaStreamSynchronize(st));
    cudaGraph_t g;
    D2CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    draft_compute(K, st);
    D2CHECK(cudaStreamEndCapture(st, &g));
    D2CHECK(cudaGraphInstantiate(&draft_exec, g, nullptr, nullptr, 0));
    D2CHECK(cudaGraphDestroy(g));
    draft_exec_k = K;
    if (d_sp) { // sampled-walk twin: identical forward, walk kernel in sampled mode
        draft_compute(K, st, true); // warm
        D2CHECK(cudaStreamSynchronize(st));
        cudaGraph_t gs;
        D2CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        draft_compute(K, st, true);
        D2CHECK(cudaStreamEndCapture(st, &gs));
        D2CHECK(cudaGraphInstantiate(&draft_exec_s, gs, nullptr, nullptr, 0));
        D2CHECK(cudaGraphDestroy(gs));
    }
}

void Dflash2::draft_compute(int K, cudaStream_t st, bool sampling) {
    const int W = K + 1;
    char nm[64];
    // anchor row (engine Q8 embed, device anchor) + K mask rows
    if (eembed_data)
        q27k::embed_row_q8(eembed_data, eembed_scales, d_anchor_tok, D2_H, nx, st);
    for (int r = 1; r < W; r++)
        D2CHECK(cudaMemcpyAsync(nx + (size_t)r * D2_H, maskrow, D2_H * 4,
                                cudaMemcpyDeviceToDevice, st));
    IP3 posW{};
    for (int i = 0; i < 16; i++) posW.p[i] = d_posW + (i < W ? i : 0);

    P3 xP = mkP3(nx, D2_H, W);
    CP3 xC = mkCP3(nx, D2_H, W);
    P3 h1P = mkP3(nh1, D2_H, W);
    CP3 y0C = mkCP3(ny0, D2_H, W);
    dim3 cgrid(20, W);

    for (int l = 0; l < D2_LAYERS; l++) {
        // -- attention half --
        snprintf(nm, sizeof nm, "layers.%d.input_layernorm.weight", l);
        q27k::rmsnorm3(xC, f32(nm), h1P, D2_H, D2_EPS, st, W);
        snprintf(nm, sizeof nm, "layers.%d.attention_conv.kernel_projection.weight", l);
        mmq(nm, nh1, ndyn, W, st);
        snprintf(nm, sizeof nm, "layers.%d.attention_conv.base_kernel", l);
        const float* baseA = f32(nm);
        k_d2_dconv<<<cgrid, 256, 0, st>>>(nh1, ndyn, baseA, ny0, W, 0);
        // q/k/v all read ny0 (post attention-conv) -- quantize it ONCE
        quant_act(ny0, D2_H, W, st);
        snprintf(nm, sizeof nm, "layers.%d.self_attn.q_proj.weight", l);
        mmq_pre(nm, nq, W, st);
        snprintf(nm, sizeof nm, "layers.%d.self_attn.k_proj.weight", l);
        mmq_pre(nm, nk, W, st);
        snprintf(nm, sizeof nm, "layers.%d.self_attn.v_proj.weight", l);
        mmq_pre(nm, nv, W, st);
        // nq/nk are [W][*D] contiguous and each head is exactly head_dim with
        // stride head_dim, so all W rows' heads norm in ONE launch (W*NH and
        // W*NKV heads) -- bit-identical to the per-row loop, W-1 fewer launches.
        snprintf(nm, sizeof nm, "layers.%d.self_attn.q_norm.weight", l);
        q27k::rmsnorm_heads(nq, f32(nm), nq, W * D2_NH, D2_HD, D2_HD, D2_EPS, st);
        snprintf(nm, sizeof nm, "layers.%d.self_attn.k_norm.weight", l);
        q27k::rmsnorm_heads(nk, f32(nm), nk, W * D2_NKV, D2_HD, D2_HD, D2_EPS, st);
        q27k::rope3(mkP3(nq, D2_QD, W), D2_NH, D2_HD, D2_HD, D2_HD, posW, D2_THETA, st, W);
        q27k::rope3(mkP3(nk, D2_KVD, W), D2_NKV, D2_HD, D2_HD, D2_HD, posW, D2_THETA, st, W);
        {
            dim3 grid(W, D2_NH);
            // fixed max smem (ring cap + block) so the launch config is
            // graph-stable across rounds; the kernel reads d_ctx_n for the
            // live count and uses only that many score slots.
            size_t smem = (size_t)(ctx_cap + D2_WMAX) * 4;
            k_d2_attn<<<grid, 128, smem, st>>>(nq, ringK[l], ringV[l], d_ring_pos, d_ctx_n, nk,
                                               nv, d_posW, natt, W);
        }
        snprintf(nm, sizeof nm, "layers.%d.self_attn.o_proj.weight", l);
        mmq(nm, natt, no, W, st);
        k_d2_dconv<<<cgrid, 256, 0, st>>>(no, ndyn, baseA + D2_CONVK * D2_H, ny0, W, 1);
        q27k::add3(xP, y0C, D2_H, st, W);
        // -- mlp half --
        snprintf(nm, sizeof nm, "layers.%d.post_attention_layernorm.weight", l);
        q27k::rmsnorm3(xC, f32(nm), h1P, D2_H, D2_EPS, st, W);
        snprintf(nm, sizeof nm, "layers.%d.mlp_conv.kernel_projection.weight", l);
        mmq(nm, nh1, ndyn, W, st);
        snprintf(nm, sizeof nm, "layers.%d.mlp_conv.base_kernel", l);
        const float* baseM = f32(nm);
        k_d2_dconv<<<cgrid, 256, 0, st>>>(nh1, ndyn, baseM, ny0, W, 0);
        // gate/up both read ny0 (post mlp-conv) -- quantize it ONCE
        quant_act(ny0, D2_H, W, st);
        snprintf(nm, sizeof nm, "layers.%d.mlp.gate_proj.weight", l);
        mmq_pre(nm, ngate, W, st);
        snprintf(nm, sizeof nm, "layers.%d.mlp.up_proj.weight", l);
        mmq_pre(nm, nup, W, st);
        q27k::silu_mul3(mkP3(ngate, D2_I, W), mkCP3(nup, D2_I, W), D2_I, st, W);
        snprintf(nm, sizeof nm, "layers.%d.mlp.down_proj.weight", l);
        mmq(nm, ngate, no, W, st);
        k_d2_dconv<<<cgrid, 256, 0, st>>>(no, ndyn, baseM + D2_CONVK * D2_H, ny0, W, 1);
        q27k::add3(xP, y0C, D2_H, st, W);
    }
    // final norm on the K mask rows, head + selector projection
    q27k::rmsnorm3(mkCP3(nx + D2_H, D2_H, K), f32("norm.weight"), mkP3(nhf, D2_H, K), D2_H,
                   D2_EPS, st, K);
    if (ehead_data) {
        q27k::XQ3 xq{};
        for (int i = 0; i < 16; i++) xq.q[i] = hxq[i < K ? i : 0];
        q27k::quantize3(mkCP3(nhf, D2_H, K), D2_H, xq, st, K);
        float* ys[16];
        for (int i = 0; i < 16; i++) ys[i] = nlogits + (size_t)(i < K ? i : 0) * D2_V;
        q27k::XQuant qs[16];
        for (int i = 0; i < 16; i++) qs[i] = hxq[i < K ? i : 0];
        if (ehead_q4)
            q27k::gemv_q4_n((const uint8_t*)ehead_data, ehead_scales, qs, K, ys, D2_V, D2_H,
                            st);
        else
            q27k::gemv_q8_n((const int8_t*)ehead_data, ehead_scales, qs, K, ys, D2_V, D2_H,
                            st);
    } else {
        q27k::gemv_f16_3(f16("target.head.weight"), mkCP3(nhf, D2_H, K),
                         mkP3(nlogits, D2_V, K), D2_V, D2_H, st, K);
    }
    mmq("candidate_selector.hidden_projection.weight", nhf, nhp, K, st);
    // on-device top-16 (two-stage) + selector walk; proposals land in d_prop
    {
        dim3 g1(K, D2_T16B);
        k_d2_top16a<<<g1, 128, 0, st>>>(nlogits, d_c1v, d_c1i, K);
        k_d2_top16b<<<K, 256, 0, st>>>(d_c1v, d_c1i, d_cand, d_cval, K);
    }
    k_d2_walk<<<1, 256, 0, st>>>(d_cand, d_cval, nhp,
                                 (const __half*)T("candidate_selector.predecessor_codebook").dev,
                                 (const __half*)T("candidate_selector.successor_codebook").dev,
                                 d_anchor_tok, K, d_prop, (sampling && d_sp) ? 1 : 0, d_sp,
                                 d_posW, d_qrow);
    D2CHECK(cudaGetLastError());
}

} // namespace q27d2
