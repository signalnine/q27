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

// Bidirectional sliding-window attention, flash-decoding layout (2026-09-07
// rewrite; the bring-up kernel -- one block per (query row, q head), a
// per-thread key-row walk and a single-thread softmax -- measured 375 us per
// launch at a 2.3K-row ring in serving, 1.87 ms of the 3.95 ms draft graph).
// Grid = (D2_ASPLIT key splits) x (kv heads); a block serves ALL query
// vectors of its kv head (nrows rows x 4 GQA heads, <= 48) against its key
// range in 32-key tiles staged through smem with float4 loads (rows padded
// to 132 floats so 8 consecutive rows land on disjoint bank quads). Online
// softmax per query across tiles; each block writes a (max, sum, acc[128])
// partial and k_d2_attn_combine merges the splits. Keys = the last
// D2_WINDOW ring rows (older ones are outside every query's window) + the
// nrows noise rows; the position mask stays authoritative. Numerics: fp32
// throughout, order differs from the serial kernel (test_d2_attn vs CPU).
constexpr int D2_ASPLIT = 32;              // key splits per kv head (graph-stable grid)
constexpr int D2_ATILE = 32;               // keys per smem tile
constexpr int D2_ATP = D2_HD + 4;          // padded row stride (floats)
constexpr int D2_AQMAX = D2_WMAX * (D2_NH / D2_NKV); // query vectors per kv head at max width
constexpr int D2_APART = D2_HD + 2;        // partial record: m, l, acc[D2_HD]
constexpr size_t D2_ASMEM = (size_t)(32 * D2_ATP + D2_ATILE * D2_ATP + 32 * 33) * 4;

__global__ void __launch_bounds__(256) k_d2_attn_split(
    const float* __restrict__ q, const float* __restrict__ ringK, const float* __restrict__ ringV,
    const int* __restrict__ ring_pos, const int* __restrict__ d_ctx_n, const float* __restrict__ nk,
    const float* __restrict__ nv, const int* __restrict__ npos, float* __restrict__ part,
    int nrows) {
    extern __shared__ __align__(16) float smem[];
    float* qs = smem;                   // [32][ATP] this query group
    float* kt = qs + 32 * D2_ATP;       // [ATILE][ATP] K tile, then V tile
    float* sc = kt + D2_ATILE * D2_ATP; // [32][33] probabilities
    __shared__ float s_m[32], s_l[32];
    const int split = blockIdx.x, kh = blockIdx.y;
    const int ctx_n = *d_ctx_n;
    const int j0 = ctx_n > D2_WINDOW ? ctx_n - D2_WINDOW : 0;
    const int nctx = ctx_n - j0, total = nctx + nrows;
    const int per = (total + D2_ASPLIT - 1) / D2_ASPLIT;
    const int kb = split * per, ke = min(total, kb + per);
    const int NQ = nrows * (D2_NH / D2_NKV);
    const float scale = rsqrtf((float)D2_HD);
    const int t = threadIdx.x, ql = t >> 3, kk = t & 7, dd = kk * 16;
    for (int g0 = 0; g0 < NQ; g0 += 32) {
        const int gn = min(32, NQ - g0);
        for (int i = t; i < gn * (D2_HD / 4); i += 256) {
            const int l = i / (D2_HD / 4), c4 = i % (D2_HD / 4), qi = g0 + l;
            const int r = qi >> 2, h = kh * (D2_NH / D2_NKV) + (qi & 3);
            *reinterpret_cast<float4*>(qs + l * D2_ATP + c4 * 4) =
                *reinterpret_cast<const float4*>(q + (size_t)r * D2_QD + (size_t)h * D2_HD + c4 * 4);
        }
        if (t < 32) { s_m[t] = -INFINITY; s_l[t] = 0.f; }
        __syncthreads();
        float acc[16];
#pragma unroll
        for (int i = 0; i < 16; i++) acc[i] = 0.f;
        const bool live = ql < gn;
        const int qpos = live ? npos[(g0 + ql) >> 2] : 0;
        for (int tb = kb; tb < ke; tb += D2_ATILE) {
            const int tn = min(D2_ATILE, ke - tb);
            for (int i = t; i < tn * (D2_HD / 4); i += 256) {
                const int row = i / (D2_HD / 4), c4 = i % (D2_HD / 4), j = tb + row;
                const float* kv = j < nctx ? ringK + (size_t)(j0 + j) * D2_KVD + (size_t)kh * D2_HD
                                           : nk + (size_t)(j - nctx) * D2_KVD + (size_t)kh * D2_HD;
                *reinterpret_cast<float4*>(kt + row * D2_ATP + c4 * 4) =
                    *reinterpret_cast<const float4*>(kv + c4 * 4);
            }
            __syncthreads();
            float s4[4];
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const int row = kk + 8 * m;
                float d = -INFINITY;
                if (live && row < tn) {
                    const float* kr = kt + row * D2_ATP;
                    const float* qr = qs + ql * D2_ATP;
                    float acc_d = 0.f;
#pragma unroll 8
                    for (int c = 0; c < D2_HD; c += 4) {
                        const float4 a = *reinterpret_cast<const float4*>(qr + c);
                        const float4 b = *reinterpret_cast<const float4*>(kr + c);
                        acc_d += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                    }
                    const int j = tb + row;
                    const int kpos = j < nctx ? ring_pos[j0 + j] : npos[j - nctx];
                    const bool vis = (qpos - kpos) < D2_WINDOW && (kpos - qpos) < D2_WINDOW;
                    d = vis ? acc_d * scale : -INFINITY;
                }
                s4[m] = d;
            }
            float tmax = fmaxf(fmaxf(s4[0], s4[1]), fmaxf(s4[2], s4[3]));
            for (int o = 1; o < 8; o <<= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
            const float m_old = s_m[ql];
            const float m_new = fmaxf(m_old, tmax);
            float psum = 0.f;
#pragma unroll
            for (int m = 0; m < 4; m++) {
                const float p = (s4[m] == -INFINITY) ? 0.f : expf(s4[m] - m_new);
                sc[ql * 33 + kk + 8 * m] = p;
                psum += p;
            }
            for (int o = 1; o < 8; o <<= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
            const float alpha = (m_old == -INFINITY) ? 0.f : expf(m_old - m_new);
            __syncthreads(); // every lane has read s_m/s_l and finished with the K tile
            if (kk == 0) { s_m[ql] = m_new; s_l[ql] = s_l[ql] * alpha + psum; }
#pragma unroll
            for (int i = 0; i < 16; i++) acc[i] *= alpha;
            for (int i = t; i < tn * (D2_HD / 4); i += 256) {
                const int row = i / (D2_HD / 4), c4 = i % (D2_HD / 4), j = tb + row;
                const float* vv = j < nctx ? ringV + (size_t)(j0 + j) * D2_KVD + (size_t)kh * D2_HD
                                           : nv + (size_t)(j - nctx) * D2_KVD + (size_t)kh * D2_HD;
                *reinterpret_cast<float4*>(kt + row * D2_ATP + c4 * 4) =
                    *reinterpret_cast<const float4*>(vv + c4 * 4);
            }
            __syncthreads();
            if (live) {
                for (int row = 0; row < tn; row++) {
                    const float p = sc[ql * 33 + row];
                    const float* vr = kt + row * D2_ATP + dd;
#pragma unroll
                    for (int c4 = 0; c4 < 4; c4++) {
                        const float4 v = *reinterpret_cast<const float4*>(vr + c4 * 4);
                        acc[c4 * 4 + 0] += p * v.x;
                        acc[c4 * 4 + 1] += p * v.y;
                        acc[c4 * 4 + 2] += p * v.z;
                        acc[c4 * 4 + 3] += p * v.w;
                    }
                }
            }
            __syncthreads(); // before the next tile overwrites kt / sc / s_m
        }
        if (live) {
            float* pr = part + (((size_t)split * D2_NKV + kh) * D2_AQMAX + (g0 + ql)) * D2_APART;
            if (kk == 0) { pr[0] = s_m[ql]; pr[1] = s_l[ql]; }
#pragma unroll
            for (int i = 0; i < 16; i++) pr[2 + dd + i] = acc[i];
        }
        __syncthreads();
    }
}

// Merge the D2_ASPLIT partials of one (query, kv head): block = (qi, kh),
// thread = head dim. Neutral partials (m = -inf) contribute nothing.
__global__ void k_d2_attn_combine(const float* __restrict__ part, float* __restrict__ out,
                                  int nrows) {
    const int qi = blockIdx.x, kh = blockIdx.y, d = threadIdx.x;
    const int r = qi >> 2, h = kh * (D2_NH / D2_NKV) + (qi & 3);
    if (r >= nrows) return;
    const float* base = part + ((size_t)kh * D2_AQMAX + qi) * D2_APART;
    const size_t stride = (size_t)D2_NKV * D2_AQMAX * D2_APART;
    float M = -INFINITY;
    for (int s = 0; s < D2_ASPLIT; s++) M = fmaxf(M, base[(size_t)s * stride]);
    float L = 0.f, A = 0.f;
    for (int s = 0; s < D2_ASPLIT; s++) {
        const float* pr = base + (size_t)s * stride;
        const float m = pr[0];
        if (m == -INFINITY) continue;
        const float w = expf(m - M);
        L += pr[1] * w;
        A += pr[2 + d] * w;
    }
    out[(size_t)r * D2_QD + (size_t)h * D2_HD + d] = A / L;
}
static void d2_attn_launches(const float* q, const float* ringK, const float* ringV,
                             const int* ring_pos, const int* d_ctx_n, const float* nk,
                             const float* nv, const int* npos, float* part, float* out, int nrows,
                             cudaStream_t st) {
    k_d2_attn_split<<<dim3(D2_ASPLIT, D2_NKV), 256, D2_ASMEM, st>>>(q, ringK, ringV, ring_pos,
                                                                    d_ctx_n, nk, nv, npos, part,
                                                                    nrows);
    k_d2_attn_combine<<<dim3(nrows * (D2_NH / D2_NKV), D2_NKV), D2_HD, 0, st>>>(part, out, nrows);
}
// Bare launcher for test_kernels (exactness vs a CPU reference).
void d2_attn_launch(const float* q, const float* ringK, const float* ringV, const int* ring_pos,
                    const int* d_ctx_n, const float* nk, const float* nv, const int* npos,
                    float* part, float* out, int nrows, cudaStream_t st) {
    d2_attn_launches(q, ringK, ringV, ring_pos, d_ctx_n, nk, nv, npos, part, out, nrows, st);
    D2CHECK(cudaGetLastError());
}

// Per-row top-16 over the vocab (2026-09-07 rewrite: two in-smem bitonic
// sorts replace two 16-iteration argmax loops that cost 171 us/round, ~40%
// of the drafter's non-gemv time). Stage 1: 256 blocks per row each sort
// their 970-element slice (padded to 1024) and emit their top 16; stage 2:
// one block per row sorts the 4096 survivors. Keys pack (value, id) so a
// descending sort orders by value desc then id asc -- the host scan's tie
// rule -- and any global top-16 element is necessarily in its slice's top
// 16, so the result is exact (test_kernels test_d2_top16 vs a CPU sort).
constexpr int D2_T16B = 256;    // stage-1 blocks per row
constexpr int D2_T16A_N = 1024; // stage-1 sorted tile (slice 970 padded)
constexpr int D2_T16B_N = D2_T16B * D2_TOPK; // 4096 stage-2 candidates
static_assert(D2_V % D2_T16B == 0 && D2_V / D2_T16B <= D2_T16A_N, "top-16 slice geometry");

// (value, id) -> orderable u64: monotonic float map in the high word (same
// transform as blocks.cu am_pack), ~id in the low word so equal values order
// by ascending id under a DESCENDING sort. Padding key 0 sorts last (a real
// key's high word is never 0).
__device__ __forceinline__ unsigned long long d2_key(float v, int id) {
    unsigned u = __float_as_uint(v);
    if ((u & 0x7fffffffu) == 0) u = 0;
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long)u << 32) | (unsigned)(~id);
}
__device__ __forceinline__ float d2_key_val(unsigned long long k) {
    unsigned u = (unsigned)(k >> 32);
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u);
}
__device__ __forceinline__ int d2_key_id(unsigned long long k) {
    return (int)~(unsigned)(k & 0xffffffffull);
}
// In-smem bitonic sort, descending, N a power of two, any blockDim.
template <int N>
__device__ __forceinline__ void d2_bitonic_desc(unsigned long long* s) {
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const unsigned long long a = s[i], b = s[ixj];
                    const bool desc = (i & k) == 0;
                    if (desc ? (a < b) : (a > b)) { s[i] = b; s[ixj] = a; }
                }
            }
            __syncthreads();
        }
    }
}

__global__ void k_d2_top16a(const float* __restrict__ logits, unsigned long long* __restrict__ c1,
                            int rows, int vocab, const int* __restrict__ ids) {
    const int r = blockIdx.x, b = blockIdx.y;
    if (r >= rows) return;
    const int slice = (vocab + D2_T16B - 1) / D2_T16B;
    const int v0 = b * slice;
    const float* lg = logits + (size_t)r * D2_V;
    __shared__ unsigned long long s[D2_T16A_N];
    for (int i = threadIdx.x; i < D2_T16A_N; i += blockDim.x)
        s[i] = i < slice && v0 + i < vocab
                   ? d2_key(lg[v0 + i], ids ? ids[v0 + i] : v0 + i) : 0ull;
    __syncthreads();
    d2_bitonic_desc<D2_T16A_N>(s);
    if (threadIdx.x < D2_TOPK)
        c1[((size_t)r * D2_T16B + b) * D2_TOPK + threadIdx.x] = s[threadIdx.x];
}

__global__ void k_d2_top16b(const unsigned long long* __restrict__ c1, int* __restrict__ cand,
                            float* __restrict__ cval, int rows) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    extern __shared__ unsigned long long sb[]; // [D2_T16B_N]
    const unsigned long long* src = c1 + (size_t)r * D2_T16B_N;
    for (int i = threadIdx.x; i < D2_T16B_N; i += blockDim.x) sb[i] = src[i];
    __syncthreads();
    d2_bitonic_desc<D2_T16B_N>(sb);
    if (threadIdx.x < D2_TOPK) {
        cand[(size_t)r * D2_TOPK + threadIdx.x] = d2_key_id(sb[threadIdx.x]);
        cval[(size_t)r * D2_TOPK + threadIdx.x] = d2_key_val(sb[threadIdx.x]);
    }
}
static void d2_top16_launches(const float* logits, unsigned long long* c1, int* cand, float* cval,
                              int K, cudaStream_t st, int vocab = D2_V, const int* ids = nullptr) {
    dim3 g1(K, D2_T16B);
    k_d2_top16a<<<g1, 256, 0, st>>>(logits, c1, K, vocab, ids);
    k_d2_top16b<<<K, 1024, (size_t)D2_T16B_N * 8, st>>>(c1, cand, cval, K);
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
                          const int* __restrict__ posW, float* __restrict__ qrow,
                          float proposal_inv_temp) {
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
                const float invT = proposal_inv_temp > 0.f ? proposal_inv_temp : sp->inv_temp;
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

void d2_top16_launch(const float* d_logits, unsigned long long* d_c1, int* d_cand, float* d_cval,
                     int K, cudaStream_t st, int vocab, const int* d_ids) {
    d2_top16_launches(d_logits, d_c1, d_cand, d_cval, K, st, vocab, d_ids);
    D2CHECK(cudaGetLastError());
}

void d2_walk_launch(const int* d_cand, const float* d_cval, const float* d_hp,
                    const __half* d_pred, const __half* d_succ, const int* d_anchor, int K,
                    int* d_out, bool sampled, const q27k::SampleParams* d_sp, const int* d_posW,
                    float* d_qrow, cudaStream_t st, float proposal_inv_temp) {
    k_d2_walk<<<1, 256, 0, st>>>(d_cand, d_cval, d_hp, d_pred, d_succ, d_anchor, K, d_out,
                                 sampled ? 1 : 0, d_sp, d_posW, d_qrow, proposal_inv_temp);
    D2CHECK(cudaGetLastError());
}

// ---- pack loader --------------------------------------------------------

// Gather already-quantized rows without requantizing: the proposal head's
// only numerical change is its vocabulary subset. Original IDs also key the
// top-16 tie break, embedding lookup, codebooks, and sparse-q rejection.
__global__ void k_d2_gather_head(const unsigned* src, const __half* scales,
                                unsigned* dst, __half* dst_scales, const int* ids,
                                int words, int groups) {
    const int row = blockIdx.x, id = ids[row];
    for (int i = threadIdx.x; i < words; i += blockDim.x)
        dst[(size_t)row * words + i] = src[(size_t)id * words + i];
    for (int i = threadIdx.x; i < groups; i += blockDim.x)
        dst_scales[(size_t)row * groups + i] = scales[(size_t)id * groups + i];
}

void d2_gather_head_launch(const void* src, const __half* scales, void* dst, __half* dst_scales,
                           const int* ids, int rows, bool q4, cudaStream_t st) {
    k_d2_gather_head<<<rows, 256, 0, st>>>((const unsigned*)src, scales, (unsigned*)dst,
                                         dst_scales, ids, D2_H / (q4 ? 8 : 4),
                                         D2_H / (q4 ? 64 : 128));
    D2CHECK(cudaGetLastError());
}

void Dflash2::load_engine_shortlist(const char* path) {
    auto fail = []() {
        fprintf(stderr, "dflash2 shortlist: expected 128..248320 unique int32 token IDs, "
                        "row count a multiple of 128, quantized engine head set before capture\n");
        exit(1);
    };
    if (!ehead_data || !ehead_scales || d_head_ids || draft_exec || draft_exec_s) fail();
    FILE* f = fopen(path, "rb");
    if (!f) { perror("dflash2 shortlist open"); exit(1); }
    if (fseek(f, 0, SEEK_END)) { fclose(f); fail(); }
    const long bytes = ftell(f);
    if (bytes < 128 * 4 || bytes > D2_V * 4 || bytes % (128 * 4)) { fclose(f); fail(); }
    rewind(f);
    const int rows = (int)(bytes / 4);
    std::vector<int> ids(rows);
    const size_t got = fread(ids.data(), 4, rows, f);
    fclose(f);
    if (got != (size_t)rows) fail();
    std::vector<unsigned char> seen(D2_V, 0);
    for (int id : ids) {
        if (id < 0 || id >= D2_V || seen[id]) fail();
        seen[id] = 1;
    }
    void* head = nullptr;
    __half* scales = nullptr;
    D2CHECK(cudaMalloc(&d_head_ids, (size_t)rows * 4));
    D2CHECK(cudaMemcpy(d_head_ids, ids.data(), (size_t)rows * 4, cudaMemcpyHostToDevice));
    D2CHECK(cudaMalloc(&head, (size_t)rows * D2_H / (ehead_q4 ? 2 : 1)));
    D2CHECK(cudaMalloc(&scales, (size_t)rows * (D2_H / (ehead_q4 ? 64 : 128)) * 2));
    d2_gather_head_launch(ehead_data, ehead_scales, head, scales, d_head_ids, rows, ehead_q4, 0);
    D2CHECK(cudaDeviceSynchronize());
    ehead_data = head;
    ehead_scales = scales;
    head_vocab = rows;
    fprintf(stderr, "dflash2 proposal head shortlist: %d/%d rows\n", rows, D2_V);
}

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
    D2CHECK(cudaMalloc(&d_c1, (size_t)(D2_WMAX - 1) * D2_T16B_N * 8));
    // device scalar mirror of ctx_n for the attention kernel (graph-stable);
    // zeroed so capture_draft's warm run (before any ingest/draft uploads
    // the live count) attends over an EMPTY ring, not garbage.
    D2CHECK(cudaMalloc(&d_ctx_n, 4));
    D2CHECK(cudaMemset(d_ctx_n, 0, 4));
    D2CHECK(cudaMalloc(&d_attn_part, (size_t)D2_ASPLIT * D2_NKV * D2_AQMAX * D2_APART * 4));
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
        d2_attn_launches(nq, ringK[l], ringV[l], d_ring_pos, d_ctx_n, nk, nv, d_posW, d_attn_part,
                         natt, W, st);
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
            q27k::gemv_q4_n((const uint8_t*)ehead_data, ehead_scales, qs, K, ys, head_vocab, D2_H,
                            st);
        else
            q27k::gemv_q8_n((const int8_t*)ehead_data, ehead_scales, qs, K, ys, head_vocab, D2_H,
                            st);
    } else {
        q27k::gemv_f16_3(f16("target.head.weight"), mkCP3(nhf, D2_H, K),
                         mkP3(nlogits, D2_V, K), D2_V, D2_H, st, K);
    }
    mmq("candidate_selector.hidden_projection.weight", nhf, nhp, K, st);
    // on-device top-16 (two-stage) + selector walk; proposals land in d_prop
    {
        d2_top16_launches(nlogits, d_c1, d_cand, d_cval, K, st, head_vocab, d_head_ids);
    }
    k_d2_walk<<<1, 256, 0, st>>>(d_cand, d_cval, nhp,
                                 (const __half*)T("candidate_selector.predecessor_codebook").dev,
                                 (const __half*)T("candidate_selector.successor_codebook").dev,
                                 d_anchor_tok, K, d_prop, (sampling && d_sp) ? 1 : 0, d_sp,
                                 d_posW, d_qrow, proposal_inv_temp);
    D2CHECK(cudaGetLastError());
}

} // namespace q27d2
