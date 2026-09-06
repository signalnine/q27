// DFlash2 drafter runtime -- see src/dflash2.h. Eager bring-up: the forward
// mirrors z-lab dflash/model.py operation for operation (the parity gate is
// tools/dflash2_smoke.cu vs bench/dflash2/p1_qtap_al.py on the same dump).
#include "dflash2.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

#include "blocks.cuh"  // rmsnorm_heads, rope_neox_partial
#include "kernels.cuh" // gemv_f16, rmsnorm, P3/CP3, gemv_f16_3, rmsnorm3, add3, silu_mul3
#include "spec3.cuh"   // IP3, rope3, gemv_f16_3 batched decls

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

// Bidirectional sliding-window attention: 8 noise queries vs [ctx ring + 8
// noise] keys. Block = (query row, q head); two-pass softmax in smem. GQA:
// q head h reads kv head h/4. Bring-up kernel -- clarity over speed.
__global__ void k_d2_attn(const float* __restrict__ q, const float* __restrict__ ringK,
                          const float* __restrict__ ringV, const int* __restrict__ ring_pos,
                          int ctx_n, const float* __restrict__ nk, const float* __restrict__ nv,
                          const int* __restrict__ npos, float* __restrict__ out, int nrows) {
    extern __shared__ float sc[]; // scores [ctx_n + nrows]
    const int r = blockIdx.x, h = blockIdx.y, kh = h / (D2_NH / D2_NKV);
    if (r >= nrows) return;
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
    // softmax (thread 0 reduction: total <= a few thousand, bring-up only)
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
    assert(!memcmp(p, "D2W1", 4));
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
        t.host = p + off;
        t.rows = dims[0];
        t.cols = ndim > 1 ? dims[1] : 1;
        for (int d = 2; d < ndim; d++) t.cols *= dims[d];
        t.n = t.rows * t.cols;
        D2CHECK(cudaMalloc(&t.dev, nbytes));
        D2CHECK(cudaMemcpy(t.dev, t.host, nbytes, cudaMemcpyHostToDevice));
        off += nbytes;
        w[name] = t;
    }
    assert(T("fc.weight").rows == D2_H && T("fc.weight").cols == D2_TAPD);
    assert(T("candidate_selector.predecessor_codebook").rows == D2_V);
}

void Dflash2::alloc(int cap) {
    ctx_cap = cap;
    for (int l = 0; l < D2_LAYERS; l++) {
        D2CHECK(cudaMalloc(&ringK[l], (size_t)cap * D2_KVD * 4));
        D2CHECK(cudaMalloc(&ringV[l], (size_t)cap * D2_KVD * 4));
    }
    D2CHECK(cudaMalloc(&d_ring_pos, cap * 4));
    D2CHECK(cudaMalloc(&s_fc, D2_H * 4));
    D2CHECK(cudaMalloc(&s_ct, D2_H * 4));
    D2CHECK(cudaMalloc(&nx, (size_t)D2_W * D2_H * 4));
    D2CHECK(cudaMalloc(&nh1, (size_t)D2_W * D2_H * 4));
    D2CHECK(cudaMalloc(&ny0, (size_t)D2_W * D2_H * 4));
    D2CHECK(cudaMalloc(&ndyn, (size_t)D2_W * 2 * D2_CONVK * D2_CONVG * 4));
    D2CHECK(cudaMalloc(&nq, (size_t)D2_W * D2_QD * 4));
    D2CHECK(cudaMalloc(&nk, (size_t)D2_W * D2_KVD * 4));
    D2CHECK(cudaMalloc(&nv, (size_t)D2_W * D2_KVD * 4));
    D2CHECK(cudaMalloc(&natt, (size_t)D2_W * D2_QD * 4));
    D2CHECK(cudaMalloc(&no, (size_t)D2_W * D2_H * 4));
    D2CHECK(cudaMalloc(&ngate, (size_t)D2_W * D2_I * 4));
    D2CHECK(cudaMalloc(&nup, (size_t)D2_W * D2_I * 4));
    D2CHECK(cudaMalloc(&nhf, (size_t)D2_K * D2_H * 4));
    D2CHECK(cudaMalloc(&nlogits, (size_t)D2_K * D2_V * 4));
    D2CHECK(cudaMalloc(&nhp, (size_t)D2_K * D2_RANK * 4));
    D2CHECK(cudaMalloc(&maskrow, D2_H * 4));
    D2CHECK(cudaMalloc(&d_pos8, D2_W * 4));
    D2CHECK(cudaMalloc(&d_ing_pos, 4096 * 4));
    // cache the mask-token embedding once
    k_d2_rowcast<<<40, 256>>>(f16("target.embed.weight") + (size_t)D2_MASK * D2_H, maskrow,
                              D2_H);
    D2CHECK(cudaDeviceSynchronize());
}

// ---- context ingest -----------------------------------------------------

void Dflash2::ingest(const float* d_taps, const int* h_pos, int T, cudaStream_t st) {
    assert(ctx_n + T <= ctx_cap && T <= 4096);
    D2CHECK(cudaMemcpyAsync(d_ing_pos, h_pos, T * 4, cudaMemcpyHostToDevice, st));
    D2CHECK(cudaMemcpyAsync(d_ring_pos + ctx_n, h_pos, T * 4, cudaMemcpyHostToDevice, st));
    for (int t = 0; t < T; t++) {
        q27k::gemv_f16(f16("fc.weight"), d_taps + (size_t)t * D2_TAPD, s_fc, D2_H, D2_TAPD, st);
        q27k::rmsnorm(s_fc, f32("hidden_norm.weight"), s_ct, D2_H, D2_EPS, st);
        for (int l = 0; l < D2_LAYERS; l++) {
            char nm[64];
            float* krow = ringK[l] + (size_t)(ctx_n + t) * D2_KVD;
            float* vrow = ringV[l] + (size_t)(ctx_n + t) * D2_KVD;
            snprintf(nm, sizeof nm, "layers.%d.self_attn.k_proj.weight", l);
            q27k::gemv_f16(f16(nm), s_ct, krow, D2_KVD, D2_H, st);
            snprintf(nm, sizeof nm, "layers.%d.self_attn.k_norm.weight", l);
            q27k::rmsnorm_heads(krow, f32(nm), krow, D2_NKV, D2_HD, D2_HD, D2_EPS, st);
            q27k::rope_neox_partial(krow, D2_NKV, D2_HD, D2_HD, D2_HD, d_ing_pos + t, D2_THETA,
                                    st);
            snprintf(nm, sizeof nm, "layers.%d.self_attn.v_proj.weight", l);
            q27k::gemv_f16(f16(nm), s_ct, vrow, D2_KVD, D2_H, st);
        }
    }
    ctx_n += T;
}

// ---- draft block --------------------------------------------------------

void Dflash2::draft(int anchor_token, int anchor_pos, int* out, cudaStream_t st) {
    char nm[64];
    // noise rows: anchor embedding + 7 mask embeddings
    k_d2_rowcast<<<40, 256, 0, st>>>(
        f16("target.embed.weight") + (size_t)anchor_token * D2_H, nx, D2_H);
    for (int r = 1; r < D2_W; r++)
        D2CHECK(cudaMemcpyAsync(nx + (size_t)r * D2_H, maskrow, D2_H * 4,
                                cudaMemcpyDeviceToDevice, st));
    int hpos[D2_W];
    for (int i = 0; i < D2_W; i++) hpos[i] = anchor_pos + i;
    D2CHECK(cudaMemcpyAsync(d_pos8, hpos, D2_W * 4, cudaMemcpyHostToDevice, st));
    IP3 pos8{};
    for (int i = 0; i < 16; i++) pos8.p[i] = d_pos8 + (i < D2_W ? i : 0);

    P3 xP = mkP3(nx, D2_H, D2_W);
    CP3 xC = mkCP3(nx, D2_H, D2_W);
    P3 h1P = mkP3(nh1, D2_H, D2_W);
    CP3 h1C = mkCP3(nh1, D2_H, D2_W);
    P3 y0P = mkP3(ny0, D2_H, D2_W);
    CP3 y0C = mkCP3(ny0, D2_H, D2_W);
    P3 dynP = mkP3(ndyn, 2 * D2_CONVK * D2_CONVG, D2_W);
    P3 oP = mkP3(no, D2_H, D2_W);
    CP3 oC = mkCP3(no, D2_H, D2_W);
    dim3 cgrid(20, D2_W);

    for (int l = 0; l < D2_LAYERS; l++) {
        // -- attention half --
        snprintf(nm, sizeof nm, "layers.%d.input_layernorm.weight", l);
        q27k::rmsnorm3(xC, f32(nm), h1P, D2_H, D2_EPS, st, D2_W);
        snprintf(nm, sizeof nm, "layers.%d.attention_conv.kernel_projection.weight", l);
        q27k::gemv_f16_3(f16(nm), h1C, dynP, 2 * D2_CONVK * D2_CONVG, D2_H, st, D2_W);
        snprintf(nm, sizeof nm, "layers.%d.attention_conv.base_kernel", l);
        const float* baseA = f32(nm);
        k_d2_dconv<<<cgrid, 256, 0, st>>>(nh1, ndyn, baseA, ny0, D2_W, 0);
        snprintf(nm, sizeof nm, "layers.%d.self_attn.q_proj.weight", l);
        q27k::gemv_f16_3(f16(nm), y0C, mkP3(nq, D2_QD, D2_W), D2_QD, D2_H, st, D2_W);
        snprintf(nm, sizeof nm, "layers.%d.self_attn.k_proj.weight", l);
        q27k::gemv_f16_3(f16(nm), y0C, mkP3(nk, D2_KVD, D2_W), D2_KVD, D2_H, st, D2_W);
        snprintf(nm, sizeof nm, "layers.%d.self_attn.v_proj.weight", l);
        q27k::gemv_f16_3(f16(nm), y0C, mkP3(nv, D2_KVD, D2_W), D2_KVD, D2_H, st, D2_W);
        snprintf(nm, sizeof nm, "layers.%d.self_attn.q_norm.weight", l);
        for (int r = 0; r < D2_W; r++)
            q27k::rmsnorm_heads(nq + (size_t)r * D2_QD, f32(nm), nq + (size_t)r * D2_QD, D2_NH,
                                D2_HD, D2_HD, D2_EPS, st);
        snprintf(nm, sizeof nm, "layers.%d.self_attn.k_norm.weight", l);
        for (int r = 0; r < D2_W; r++)
            q27k::rmsnorm_heads(nk + (size_t)r * D2_KVD, f32(nm), nk + (size_t)r * D2_KVD,
                                D2_NKV, D2_HD, D2_HD, D2_EPS, st);
        q27k::rope3(mkP3(nq, D2_QD, D2_W), D2_NH, D2_HD, D2_HD, D2_HD, pos8, D2_THETA, st,
                    D2_W);
        q27k::rope3(mkP3(nk, D2_KVD, D2_W), D2_NKV, D2_HD, D2_HD, D2_HD, pos8, D2_THETA, st,
                    D2_W);
        {
            dim3 grid(D2_W, D2_NH);
            size_t smem = (size_t)(ctx_n + D2_W) * 4;
            k_d2_attn<<<grid, 128, smem, st>>>(nq, ringK[l], ringV[l], d_ring_pos, ctx_n, nk,
                                               nv, d_pos8, natt, D2_W);
        }
        snprintf(nm, sizeof nm, "layers.%d.self_attn.o_proj.weight", l);
        q27k::gemv_f16_3(f16(nm), mkCP3(natt, D2_QD, D2_W), oP, D2_H, D2_QD, st, D2_W);
        k_d2_dconv<<<cgrid, 256, 0, st>>>(no, ndyn, baseA + D2_CONVK * D2_H, ny0, D2_W, 1);
        q27k::add3(xP, y0C, D2_H, st, D2_W);
        // -- mlp half --
        snprintf(nm, sizeof nm, "layers.%d.post_attention_layernorm.weight", l);
        q27k::rmsnorm3(xC, f32(nm), h1P, D2_H, D2_EPS, st, D2_W);
        snprintf(nm, sizeof nm, "layers.%d.mlp_conv.kernel_projection.weight", l);
        q27k::gemv_f16_3(f16(nm), h1C, dynP, 2 * D2_CONVK * D2_CONVG, D2_H, st, D2_W);
        snprintf(nm, sizeof nm, "layers.%d.mlp_conv.base_kernel", l);
        const float* baseM = f32(nm);
        k_d2_dconv<<<cgrid, 256, 0, st>>>(nh1, ndyn, baseM, ny0, D2_W, 0);
        snprintf(nm, sizeof nm, "layers.%d.mlp.gate_proj.weight", l);
        q27k::gemv_f16_3(f16(nm), y0C, mkP3(ngate, D2_I, D2_W), D2_I, D2_H, st, D2_W);
        snprintf(nm, sizeof nm, "layers.%d.mlp.up_proj.weight", l);
        q27k::gemv_f16_3(f16(nm), y0C, mkP3(nup, D2_I, D2_W), D2_I, D2_H, st, D2_W);
        q27k::silu_mul3(mkP3(ngate, D2_I, D2_W), mkCP3(nup, D2_I, D2_W), D2_I, st, D2_W);
        snprintf(nm, sizeof nm, "layers.%d.mlp.down_proj.weight", l);
        q27k::gemv_f16_3(f16(nm), mkCP3(ngate, D2_I, D2_W), oP, D2_H, D2_I, st, D2_W);
        k_d2_dconv<<<cgrid, 256, 0, st>>>(no, ndyn, baseM + D2_CONVK * D2_H, ny0, D2_W, 1);
        q27k::add3(xP, y0C, D2_H, st, D2_W);
    }
    // final norm on the 7 mask rows, head + selector projection
    q27k::rmsnorm3(mkCP3(nx + D2_H, D2_H, D2_K), f32("norm.weight"), mkP3(nhf, D2_H, D2_K),
                   D2_H, D2_EPS, st, D2_K);
    q27k::gemv_f16_3(f16("target.head.weight"), mkCP3(nhf, D2_H, D2_K),
                     mkP3(nlogits, D2_V, D2_K), D2_V, D2_H, st, D2_K);
    q27k::gemv_f16_3(f16("candidate_selector.hidden_projection.weight"),
                     mkCP3(nhf, D2_H, D2_K), mkP3(nhp, D2_RANK, D2_K), D2_RANK, D2_H, st,
                     D2_K);
    // host: top-16 per row + selector walk
    static std::vector<float> hlog, hhp;
    hlog.resize((size_t)D2_K * D2_V);
    hhp.resize((size_t)D2_K * D2_RANK);
    D2CHECK(cudaMemcpyAsync(hlog.data(), nlogits, hlog.size() * 4, cudaMemcpyDeviceToHost, st));
    D2CHECK(cudaMemcpyAsync(hhp.data(), nhp, hhp.size() * 4, cudaMemcpyDeviceToHost, st));
    D2CHECK(cudaStreamSynchronize(st));
    const __half* pred = (const __half*)T("candidate_selector.predecessor_codebook").host;
    const __half* succ = (const __half*)T("candidate_selector.successor_codebook").host;
    int prev = anchor_token;
    for (int posn = 0; posn < D2_K; posn++) {
        const float* lg = hlog.data() + (size_t)posn * D2_V;
        int cand[D2_TOPK];
        float cval[D2_TOPK];
        for (int i = 0; i < D2_TOPK; i++) { cand[i] = -1; cval[i] = -INFINITY; }
        for (int v = 0; v < D2_V; v++) {
            float x = lg[v];
            if (x <= cval[D2_TOPK - 1]) continue;
            int i = D2_TOPK - 1;
            while (i > 0 && cval[i - 1] < x) { cval[i] = cval[i - 1]; cand[i] = cand[i - 1]; i--; }
            cval[i] = x;
            cand[i] = v;
        }
        const float* hp = hhp.data() + (size_t)posn * D2_RANK;
        float ph[D2_RANK];
        for (int r = 0; r < D2_RANK; r++)
            ph[r] = __half2float(pred[(size_t)prev * D2_RANK + r]) * hp[r];
        float best = -INFINITY;
        int besti = 0;
        for (int i = 0; i < D2_TOPK; i++) {
            const __half* sr = succ + (size_t)cand[i] * D2_RANK;
            float s = cval[i];
            for (int r = 0; r < D2_RANK; r++) s += ph[r] * __half2float(sr[r]);
            if (s > best) { best = s; besti = i; }
        }
        prev = cand[besti];
        out[posn] = prev;
    }
}

} // namespace q27d2
