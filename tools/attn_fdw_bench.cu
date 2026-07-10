// attn_fdw_bench -- day-1 gate for the shared-KV W-query verify attention
// kernel (ceiling-8+ plan, re-attribution 2026-07-10: verify width marginal
// = ctx-dependent attention 0.58ms/lane @26K -> 1.36 @61K + ctx-independent
// 0.82). fd2 re-reads the KV chunk per lane (L2-cached at 26K, thrashing at
// 61K+: 4 kvh x 61K x 512B = 125MB > 96MB L2). fdw stages each KV tile
// through smem ONCE per (split, kvh) block and lets W warps -- one per lane,
// fd2's exact per-lane register math -- consume it.
//
// Legs per (ctx, W): fd2 fork (kernel+combine, verbatim) vs fdw prototype
// (+ same combine). Gate: max rel err vs fd2 < 1e-4 (fp order differs:
// sequential rows per lane vs warp-striped+merge) AND fdw >= 2x at 61K W=8.
// Usage: attn_fdw_bench   (synthetic tensors; no model)
#include <cuda_fp8.h>
#include <float.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "../src/fdmma.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err__ = (x);                                               \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static constexpr int N_KV = 4, GQA = 6, HD = 256, NQH = N_KV * GQA;
static constexpr int FD2_NS = 128, FD_ST = 258;
struct CP3 { const float* p[8]; };
struct P3 { float* p[8]; };
struct IP3 { const int* p[8]; };

// ---- fd2 + combine, forked verbatim from src/spec3.cu ----
template <typename CT>
__device__ __forceinline__ void fd2_ld8(const CT* __restrict__ row, int lane, float* o) {
    if constexpr (sizeof(CT) == 1) {
        uint32_t a = *reinterpret_cast<const uint32_t*>(
            reinterpret_cast<const uint8_t*>(row) + 4 * lane);
        uint32_t b = *reinterpret_cast<const uint32_t*>(
            reinterpret_cast<const uint8_t*>(row) + 128 + 4 * lane);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_fp8_e4m3 e;
            e.__x = (a >> (8 * i)) & 0xFF;
            o[i] = float(e);
            e.__x = (b >> (8 * i)) & 0xFF;
            o[4 + i] = float(e);
        }
    } else {
        uint2 a = *reinterpret_cast<const uint2*>(row + 4 * lane);
        uint2 b = *reinterpret_cast<const uint2*>(row + 128 + 4 * lane);
        const __half* ha = reinterpret_cast<const __half*>(&a);
        const __half* hb = reinterpret_cast<const __half*>(&b);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            o[i] = __half2float(ha[i]);
            o[4 + i] = __half2float(hb[i]);
        }
    }
}

template <typename CT, int NW>
__global__ void k_attn_fd2(CP3 qp, int q_stride, const CT* __restrict__ kc,
                           const CT* __restrict__ vc, float* __restrict__ part, IP3 pos,
                           int n_kv_heads, int gqa, int head_dim, float scale) {
    const int t = blockIdx.x, sp = blockIdx.y, kvh = blockIdx.z;
    const int seq = *pos.p[t] + 1;
    if (sp * ((seq + FD2_NS - 1) / FD2_NS) >= seq) return;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_mrg = smem + 6 * 256;
    __shared__ float s_ml[NW][6][2];
    for (int idx = threadIdx.x; idx < gqa * head_dim; idx += blockDim.x)
        s_q[idx] = qp.p[t][(size_t)(kvh * gqa + idx / head_dim) * q_stride + idx % head_dim];
    for (int idx = threadIdx.x; idx < 6 * 256; idx += blockDim.x) s_mrg[idx] = 0.f;
    __syncthreads();
    const int chunk = (seq + FD2_NS - 1) / FD2_NS;
    const int p_lo = sp * chunk, p_hi = min(seq, p_lo + chunk);
    float m[6], l[6], acc[6][8];
#pragma unroll
    for (int j = 0; j < 6; j++) {
        m[j] = -FLT_MAX;
        l[j] = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) acc[j][i] = 0.f;
    }
    for (int p = p_lo + warp; p < p_hi; p += NW) {
        const CT* kp = kc + ((size_t)p * n_kv_heads + kvh) * head_dim;
        const CT* vp = vc + ((size_t)p * n_kv_heads + kvh) * head_dim;
        float kv[8], vv[8];
        fd2_ld8(kp, lane, kv);
        fd2_ld8(vp, lane, vv);
#pragma unroll
        for (int j = 0; j < 6; j++) {
            const float4 qa = reinterpret_cast<const float4*>(s_q + j * 256)[lane];
            const float4 qb = reinterpret_cast<const float4*>(s_q + j * 256 + 128)[lane];
            float d = qa.x * kv[0] + qa.y * kv[1] + qa.z * kv[2] + qa.w * kv[3] +
                      qb.x * kv[4] + qb.y * kv[5] + qb.z * kv[6] + qb.w * kv[7];
            for (int off = 16; off > 0; off >>= 1) d += __shfl_down_sync(0xffffffff, d, off);
            d = __shfl_sync(0xffffffff, d, 0) * scale;
            float mn = fmaxf(m[j], d);
            float so = expf(m[j] - mn), w = expf(d - mn);
            l[j] = l[j] * so + w;
            m[j] = mn;
#pragma unroll
            for (int i = 0; i < 8; i++) acc[j][i] = acc[j][i] * so + w * vv[i];
        }
    }
#pragma unroll
    for (int j = 0; j < 6; j++)
        if (lane == 0) { s_ml[warp][j][0] = m[j]; s_ml[warp][j][1] = l[j]; }
    __syncthreads();
    for (int w = 0; w < NW; w++) {
        if (warp == w) {
#pragma unroll
            for (int j = 0; j < 6; j++) {
                float mb = -FLT_MAX;
#pragma unroll
                for (int u = 0; u < NW; u++) mb = fmaxf(mb, s_ml[u][j][0]);
                float scw = m[j] == -FLT_MAX ? 0.f : expf(m[j] - mb);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    s_mrg[j * 256 + 4 * lane + i] += acc[j][i] * scw;
                    s_mrg[j * 256 + 128 + 4 * lane + i] += acc[j][4 + i] * scw;
                }
            }
        }
        __syncthreads();
    }
    for (int j = warp; j < 6; j += NW) {
        float mb = -FLT_MAX;
        for (int u = 0; u < NW; u++) mb = fmaxf(mb, s_ml[u][j][0]);
        float lb = 0.f;
        for (int u = 0; u < NW; u++)
            lb += s_ml[u][j][1] * (s_ml[u][j][0] == -FLT_MAX ? 0.f : expf(s_ml[u][j][0] - mb));
        size_t pair = (size_t)t * (n_kv_heads * gqa) + kvh * gqa + j;
        float* dst = part + (pair * FD2_NS + sp) * FD_ST;
        if (lane == 0) { dst[0] = mb; dst[1] = lb; }
        for (int d = lane; d < head_dim; d += 32) dst[2 + d] = s_mrg[j * 256 + d];
    }
}

__global__ void k_attn_fd_combine(const float* __restrict__ part, P3 outp, int n_heads,
                                  int head_dim, int ns, IP3 pos) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int seq = *pos.p[t] + 1;
    const int chunk = (seq + ns - 1) / ns;
    const int used = (seq + chunk - 1) / chunk;
    size_t pair = (size_t)t * n_heads + h;
    const float* pp = part + pair * ns * FD_ST;
    __shared__ float s_m, s_l;
    if (threadIdx.x == 0) {
        float mg = -FLT_MAX;
        for (int sp = 0; sp < used; sp++) mg = fmaxf(mg, pp[sp * FD_ST]);
        float lg = 0.f;
        for (int sp = 0; sp < used; sp++)
            if (pp[sp * FD_ST] != -FLT_MAX)
                lg += pp[sp * FD_ST + 1] * expf(pp[sp * FD_ST] - mg);
        s_m = mg;
        s_l = lg;
    }
    __syncthreads();
    const float mg = s_m, inv = 1.0f / s_l;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float a = 0.f;
        for (int sp = 0; sp < used; sp++) {
            float ms = pp[sp * FD_ST];
            if (ms != -FLT_MAX) a += pp[sp * FD_ST + 2 + d] * expf(ms - mg);
        }
        outp.p[t][(size_t)h * head_dim + d] = a * inv;
    }
}

// ---- fdw: shared-KV W-query prototype ----
// Grid (FD2_NS, n_kv_heads); block = W warps, warp w = lane w (fd2's exact
// per-lane register math: acc[6][8], m/l[6], warp-shuffle dot). KV rows
// staged through smem in R-row tiles by all threads cooperatively; every
// warp consumes every tile row for its own lane. Rows are processed
// SEQUENTIALLY per lane (no cross-warp stripe, no merge phase). Lanes have
// different visible lengths (pos[t]); a warp simply stops scoring rows past
// its own seq. Partial layout identical to fd2 -> same combine kernel.
template <typename CT, int W, int RT>
__global__ void __launch_bounds__(W * 32)
    k_attn_fdw(CP3 qp, int q_stride, const CT* __restrict__ kc, const CT* __restrict__ vc,
               float* __restrict__ part, IP3 pos, int n_kv_heads, int gqa, int head_dim,
               float scale) {
    const int sp = blockIdx.x, kvh = blockIdx.y;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    // deepest lane bounds the split geometry: fdw iterates rows with the
    // deepest lane's chunking; a lane contributes only rows < its own seq.
    // Bench geometry keeps per-lane ceil-div chunks equal (verify lanes
    // differ by < FD2_NS rows); the ENGINE integration must handle unequal
    // chunks (flagged in the plan).
    const int seq_max = *pos.p[W - 1] + 1;
    const int chunk = (seq_max + FD2_NS - 1) / FD2_NS;
    const int p_lo = sp * chunk, p_hi = min(seq_max, p_lo + chunk);
    if (p_lo >= seq_max) return;
    const int my_seq = *pos.p[warp] + 1;

    // q lives in REGISTERS: each warp needs only its own lane's 6 heads
    // (8 floats x 6 = 48 regs/thread). smem holds only the raw KV tile.
    float qr[6][8];
#pragma unroll
    for (int j = 0; j < 6; j++) {
        const float* qrow = qp.p[warp] + (size_t)(kvh * gqa + j) * q_stride;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            qr[j][i] = qrow[4 * lane + i];
            qr[j][4 + i] = qrow[128 + 4 * lane + i];
        }
    }

    extern __shared__ unsigned char smem_raw[];
    CT* s_kv = (CT*)smem_raw; // [RT][2][256] raw CT bytes, single buffer

    float m[6], l[6], acc[6][8];
#pragma unroll
    for (int j = 0; j < 6; j++) {
        m[j] = -FLT_MAX;
        l[j] = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) acc[j][i] = 0.f;
    }

    const int nthreads = W * 32;
    for (int p0 = p_lo; p0 < p_hi; p0 += RT) {
        const int rn = min(RT, p_hi - p0);
        __syncthreads();
        // stage raw rows: RT x 512B (fp8) or RT x 1KB (half), coalesced u32/u2
        {
            const int words = rn * 2 * head_dim * (int)sizeof(CT) / 4;
            for (int idx = threadIdx.x; idx < words; idx += nthreads) {
                int bytes_per_row = 2 * head_dim * (int)sizeof(CT); // K row + V row
                int r = idx * 4 / bytes_per_row;
                int off = idx * 4 - r * bytes_per_row;
                int p = p0 + r;
                bool isv = off >= head_dim * (int)sizeof(CT);
                int ro = isv ? off - head_dim * (int)sizeof(CT) : off;
                const CT* src = (isv ? vc : kc) + ((size_t)p * n_kv_heads + kvh) * head_dim;
                *(uint32_t*)((unsigned char*)s_kv + r * bytes_per_row + off) =
                    *(const uint32_t*)((const unsigned char*)src + ro);
            }
        }
        __syncthreads();
        for (int r = 0; r < rn; r++) {
            const int p = p0 + r;
            if (p >= my_seq) break; // causal: this lane sees rows < my_seq
            const CT* kr = s_kv + (size_t)r * 2 * head_dim;
            const CT* vr = kr + head_dim;
            float kv[8], vv[8];
            fd2_ld8(kr, lane, kv);
            fd2_ld8(vr, lane, vv);
#pragma unroll
            for (int j = 0; j < 6; j++) {
                float d = qr[j][0] * kv[0] + qr[j][1] * kv[1] + qr[j][2] * kv[2] +
                          qr[j][3] * kv[3] + qr[j][4] * kv[4] + qr[j][5] * kv[5] +
                          qr[j][6] * kv[6] + qr[j][7] * kv[7];
                for (int off = 16; off > 0; off >>= 1)
                    d += __shfl_down_sync(0xffffffff, d, off);
                d = __shfl_sync(0xffffffff, d, 0) * scale;
                float mn = fmaxf(m[j], d);
                float so = expf(m[j] - mn), w = expf(d - mn);
                l[j] = l[j] * so + w;
                m[j] = mn;
#pragma unroll
                for (int i = 0; i < 8; i++) acc[j][i] = acc[j][i] * so + w * vv[i];
            }
        }
    }

    // write this lane's partial for this split -- combine derives used-splits
    // from THIS lane's seq with fd2's per-lane chunking; fdw iterated with the
    // deepest lane's chunking, so remap: this partial covers absolute rows
    // [p_lo, p_hi) intersect [0, my_seq). Write into the slot the combine
    // will read for that range under the lane's own chunk size. Prototype
    // simplification: my_chunk == chunk only when all lanes share seq; for
    // verify lanes seq differs by <8 of 61K -- chunk differs by at most 1 row
    // per split and slots stay aligned in practice ONLY when chunk equal.
    // The bench uses equal-chunk geometry (see host: per-lane pos differ by
    // <FD2_NS so ceil-div chunks match); the ENGINE integration must handle
    // the general case (own-chunk tail splits) -- flagged in the plan.
    if (p_lo >= my_seq) return;
#pragma unroll
    for (int j = 0; j < 6; j++) {
        size_t pair = (size_t)warp * (n_kv_heads * gqa) + kvh * gqa + j;
        float* dst = part + (pair * FD2_NS + sp) * FD_ST;
        if (lane == 0) { dst[0] = m[j]; dst[1] = l[j]; }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            dst[2 + 4 * lane + i] = acc[j][i];
            dst[2 + 128 + 4 * lane + i] = acc[j][4 + i];
        }
    }
}

// ---- fdw v2: warp = (lane, head-pair) ----
// Block = W lanes x 3 head-pairs = 3W warps; every warp iterates every tile
// row but owns only 2 of the 6 gqa heads -> 2 sequential dot+reduce chains
// per row instead of 6 (v1's serialization problem). q in registers
// (2 heads x 8 = 16), acc[2][8], m/l[2] -> ~60 regs. Tile staged once per
// block as raw CT bytes, all 3W warps consume it. Partial layout unchanged.
template <typename CT, int W, int RT, int NS>
__global__ void __launch_bounds__(W * 3 * 32)
    k_attn_fdw2(CP3 qp, int q_stride, const CT* __restrict__ kc, const CT* __restrict__ vc,
                float* __restrict__ part, IP3 pos, int n_kv_heads, int gqa, int head_dim,
                float scale) {
    const int sp = blockIdx.x, kvh = blockIdx.y;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int t = warp / 3, hp = warp % 3; // lane t, heads {2hp, 2hp+1}
    const int seq_max = *pos.p[W - 1] + 1;
    const int chunk = (seq_max + NS - 1) / NS;
    const int p_lo = sp * chunk, p_hi = min(seq_max, p_lo + chunk);
    if (p_lo >= seq_max) return;
    const int my_seq = *pos.p[t] + 1;

    float qr[2][8];
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const float* qrow = qp.p[t] + (size_t)(kvh * gqa + 2 * hp + j) * q_stride;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            qr[j][i] = qrow[4 * lane + i];
            qr[j][4 + i] = qrow[128 + 4 * lane + i];
        }
    }

    extern __shared__ unsigned char smem_raw[];
    CT* s_kv = (CT*)smem_raw; // [RT][2][256] raw, single buffer

    float m[2] = {-FLT_MAX, -FLT_MAX}, l[2] = {0.f, 0.f}, acc[2][8];
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
        for (int i = 0; i < 8; i++) acc[j][i] = 0.f;

    const int nthreads = W * 3 * 32;
    for (int p0 = p_lo; p0 < p_hi; p0 += RT) {
        const int rn = min(RT, p_hi - p0);
        __syncthreads();
        {
            const int words = rn * 2 * head_dim * (int)sizeof(CT) / 4;
            for (int idx = threadIdx.x; idx < words; idx += nthreads) {
                int bytes_per_row = 2 * head_dim * (int)sizeof(CT);
                int r = idx * 4 / bytes_per_row;
                int off = idx * 4 - r * bytes_per_row;
                int p = p0 + r;
                bool isv = off >= head_dim * (int)sizeof(CT);
                int ro = isv ? off - head_dim * (int)sizeof(CT) : off;
                const CT* src = (isv ? vc : kc) + ((size_t)p * n_kv_heads + kvh) * head_dim;
                *(uint32_t*)((unsigned char*)s_kv + r * bytes_per_row + off) =
                    *(const uint32_t*)((const unsigned char*)src + ro);
            }
        }
        __syncthreads();
        const int rme = min(rn, my_seq - p0); // causal bound for this lane
        for (int r = 0; r < rme; r++) {
            const CT* kr = s_kv + (size_t)r * 2 * head_dim;
            const CT* vr = kr + head_dim;
            float kv[8], vv[8];
            fd2_ld8(kr, lane, kv);
            fd2_ld8(vr, lane, vv);
#pragma unroll
            for (int j = 0; j < 2; j++) {
                float d = qr[j][0] * kv[0] + qr[j][1] * kv[1] + qr[j][2] * kv[2] +
                          qr[j][3] * kv[3] + qr[j][4] * kv[4] + qr[j][5] * kv[5] +
                          qr[j][6] * kv[6] + qr[j][7] * kv[7];
                for (int off = 16; off > 0; off >>= 1)
                    d += __shfl_down_sync(0xffffffff, d, off);
                d = __shfl_sync(0xffffffff, d, 0) * scale;
                float mn = fmaxf(m[j], d);
                float so = expf(m[j] - mn), w = expf(d - mn);
                l[j] = l[j] * so + w;
                m[j] = mn;
#pragma unroll
                for (int i = 0; i < 8; i++) acc[j][i] = acc[j][i] * so + w * vv[i];
            }
        }
    }
    if (p_lo >= my_seq) return;
#pragma unroll
    for (int j = 0; j < 2; j++) {
        size_t pair = (size_t)t * (n_kv_heads * gqa) + kvh * gqa + 2 * hp + j;
        float* dst = part + (pair * NS + sp) * FD_ST;
        if (lane == 0) { dst[0] = m[j]; dst[1] = l[j]; }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            dst[2 + 4 * lane + i] = acc[j][i];
            dst[2 + 128 + 4 * lane + i] = acc[j][4 + i];
        }
    }
}

// ---- fdw v3: thread-per-row scoring + block-online-softmax ----
// v1/v2 lose to fd2 on the per-row shuffle-reduce latency chain (ncu: DRAM
// 14%, SM 40%, 8.1 stall cycles/instr). v3 breaks the chain: within a warp
// each THREAD scores a different KV row (sequential 256-FMA dot from smem,
// no shuffles, 32 rows in flight), softmax runs per 32-row batch (one
// warp-max + one warp-sum per query per batch), and the o-update broadcasts
// each row's weight with a single shuffle. Numerics: BLOCK-online-softmax
// (batch rescale) -- tolerance-class vs fd2's per-row rescale, same family
// as the prefill MMA kernel's block softmax. Warp = (lane, head-pair) as v2.
template <typename CT, int W, int RT, int NS>
__global__ void __launch_bounds__(W * 3 * 32)
    k_attn_fdw3(CP3 qp, int q_stride, const CT* __restrict__ kc, const CT* __restrict__ vc,
                float* __restrict__ part, IP3 pos, int n_kv_heads, int gqa, int head_dim,
                float scale) {
    const int sp = blockIdx.x, kvh = blockIdx.y;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int t = warp / 3, hp = warp % 3;
    const int seq_max = *pos.p[W - 1] + 1;
    const int chunk = (seq_max + NS - 1) / NS;
    const int p_lo = sp * chunk, p_hi = min(seq_max, p_lo + chunk);
    if (p_lo >= seq_max) return;
    const int my_seq = *pos.p[t] + 1;

    extern __shared__ unsigned char smem_raw[];
    CT* s_kv = (CT*)smem_raw;                                  // [RT][2][256] raw
    float* s_q = (float*)(smem_raw + (size_t)RT * 2 * head_dim * sizeof(CT)); // [W][6][256]
    for (int idx = threadIdx.x; idx < W * gqa * head_dim; idx += W * 3 * 32) {
        int tt = idx / (gqa * head_dim), r = idx % (gqa * head_dim);
        s_q[idx] = qp.p[tt][(size_t)(kvh * gqa + r / head_dim) * q_stride + r % head_dim];
    }

    float m[2] = {-FLT_MAX, -FLT_MAX}, l[2] = {0.f, 0.f}, acc[2][8];
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
        for (int i = 0; i < 8; i++) acc[j][i] = 0.f;

    const int nthreads = W * 3 * 32;
    for (int p0 = p_lo; p0 < p_hi; p0 += RT) {
        const int rn = min(RT, p_hi - p0);
        __syncthreads();
        {
            const int words = rn * 2 * head_dim * (int)sizeof(CT) / 4;
            for (int idx = threadIdx.x; idx < words; idx += nthreads) {
                int bytes_per_row = 2 * head_dim * (int)sizeof(CT);
                int r = idx * 4 / bytes_per_row;
                int off = idx * 4 - r * bytes_per_row;
                int p = p0 + r;
                bool isv = off >= head_dim * (int)sizeof(CT);
                int ro = isv ? off - head_dim * (int)sizeof(CT) : off;
                const CT* src = (isv ? vc : kc) + ((size_t)p * n_kv_heads + kvh) * head_dim;
                *(uint32_t*)((unsigned char*)s_kv + r * bytes_per_row + off) =
                    *(const uint32_t*)((const unsigned char*)src + ro);
            }
        }
        __syncthreads();
        const int rme = min(rn, my_seq - p0); // this lane's causal row count
        for (int rb = 0; rb < rme; rb += 32) {
            const int rcnt = min(32, rme - rb);
            // thread `lane` scores row rb+lane for both heads: sequential
            // 256-FMA dots from smem (fp32 x dequant-on-read)
            float s0 = -FLT_MAX, s1 = -FLT_MAX;
            if (lane < rcnt) {
                const CT* kr = s_kv + (size_t)(rb + lane) * 2 * head_dim;
                const float* q0 = s_q + ((size_t)t * 6 + 2 * hp) * 256;
                const float* q1 = q0 + 256;
                float d0 = 0.f, d1 = 0.f;
#pragma unroll 8
                for (int i = 0; i < 256; i++) {
                    float kx;
                    if constexpr (sizeof(CT) == 1) {
                        __nv_fp8_e4m3 e;
                        e.__x = reinterpret_cast<const uint8_t*>(kr)[i];
                        kx = float(e);
                    } else {
                        kx = __half2float(kr[i]);
                    }
                    d0 += q0[i] * kx;
                    d1 += q1[i] * kx;
                }
                s0 = d0 * scale;
                s1 = d1 * scale;
            }
            // block-online-softmax: batch max/sum via warp reduce
#pragma unroll
            for (int j = 0; j < 2; j++) {
                float sj = j ? s1 : s0;
                float bm = sj;
                for (int off = 16; off > 0; off >>= 1)
                    bm = fmaxf(bm, __shfl_down_sync(0xffffffff, bm, off));
                bm = __shfl_sync(0xffffffff, bm, 0);
                float mn = fmaxf(m[j], bm);
                float so = expf(m[j] - mn);
                float w = (lane < rcnt) ? expf(sj - mn) : 0.f;
                float ls = w;
                for (int off = 16; off > 0; off >>= 1)
                    ls += __shfl_down_sync(0xffffffff, ls, off);
                ls = __shfl_sync(0xffffffff, ls, 0);
                l[j] = l[j] * so + ls;
                m[j] = mn;
#pragma unroll
                for (int i = 0; i < 8; i++) acc[j][i] *= so;
                // o-update: broadcast each row's weight, all lanes FMA their
                // dim slice of that row's V from smem
                for (int r = 0; r < rcnt; r++) {
                    float wr = __shfl_sync(0xffffffff, w, r);
                    if (wr == 0.f) continue;
                    const CT* vr = s_kv + (size_t)(rb + r) * 2 * head_dim + head_dim;
                    float vv[8];
                    fd2_ld8(vr, lane, vv);
#pragma unroll
                    for (int i = 0; i < 8; i++) acc[j][i] += wr * vv[i];
                }
            }
        }
    }
    if (p_lo >= my_seq) return;
#pragma unroll
    for (int j = 0; j < 2; j++) {
        size_t pair = (size_t)t * (n_kv_heads * gqa) + kvh * gqa + 2 * hp + j;
        float* dst = part + (pair * NS + sp) * FD_ST;
        if (lane == 0) { dst[0] = m[j]; dst[1] = l[j]; }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            dst[2 + 4 * lane + i] = acc[j][i];
            dst[2 + 128 + 4 * lane + i] = acc[j][4 + i];
        }
    }
}

template <typename F> static double timeit(F&& fn, int reps) {
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (int r = 0; r < reps; r++) fn();
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return (double)ms / reps;
}

int main() {
    for (int CTX : {26000, 61000}) {
        // synthetic fp8 KV + fp32 q; W lanes at positions CTX-1 .. CTX+W-2
        // (consecutive verify rows; equal ceil-div chunks at these seqs)
        const int MAXW = 8;
        const size_t kvn = (size_t)(CTX + MAXW) * N_KV * HD;
        __nv_fp8_e4m3 *kc, *vc;
        CUDA_CHECK(cudaMalloc(&kc, kvn));
        CUDA_CHECK(cudaMalloc(&vc, kvn));
        {
            // e4m3 codes with exp capped (b &= 0x3F): values in [-1.875, 1.875],
            // no NaN (0x7F/0xFF unreachable). Independent draws for K and V.
            std::vector<uint8_t> h(kvn), h2(kvn);
            unsigned s = 42;
            for (size_t i = 0; i < kvn; i++) {
                s = s * 1664525u + 1013904223u;
                h[i] = (s >> 13) & 0xBF; // clear exp bit 6: |v| <= 1.875, no NaN
                s = s * 1664525u + 1013904223u;
                h2[i] = (s >> 13) & 0xBF;
            }
            CUDA_CHECK(cudaMemcpy(kc, h.data(), kvn, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(vc, h2.data(), kvn, cudaMemcpyHostToDevice));
        }
        float* q[8];
        float* o2[8];
        float* ow[8];
        int* posd[8];
        for (int t = 0; t < MAXW; t++) {
            CUDA_CHECK(cudaMalloc(&q[t], NQH * HD * 4));
            std::vector<float> hq(NQH * HD);
            unsigned s = 7 + t;
            for (auto& v : hq) {
                s = s * 1664525u + 1013904223u;
                v = ((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
            }
            CUDA_CHECK(cudaMemcpy(q[t], hq.data(), NQH * HD * 4, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&o2[t], NQH * HD * 4));
            CUDA_CHECK(cudaMalloc(&ow[t], NQH * HD * 4));
            int hp = CTX - 1 + t;
            CUDA_CHECK(cudaMalloc(&posd[t], 4));
            CUDA_CHECK(cudaMemcpy(posd[t], &hp, 4, cudaMemcpyHostToDevice));
        }
        float* part;
        CUDA_CHECK(cudaMalloc(&part, (size_t)MAXW * NQH * FD2_NS * FD_ST * 4));
        printf("== ctx %d (KV %zu MB all-heads)\n", CTX, 2 * kvn / 1000000);
        for (int W : {2, 4, 8}) {
            CP3 qp{};
            P3 o2p{}, owp{};
            IP3 pp{};
            for (int t = 0; t < W; t++) {
                qp.p[t] = q[t];
                o2p.p[t] = o2[t];
                owp.p[t] = ow[t];
                pp.p[t] = posd[t];
            }
            const float scale = 1.0f / sqrtf((float)HD);
            size_t sm2 = (size_t)(2 * 6) * 256 * 4;
            auto fd2 = [&] {
                dim3 g1(W, FD2_NS, N_KV);
                k_attn_fd2<__nv_fp8_e4m3, 4><<<g1, 128, sm2>>>(qp, HD, kc, vc, part, pp, N_KV,
                                                               GQA, HD, scale);
                dim3 g2(NQH, W);
                k_attn_fd_combine<<<g2, 256>>>(part, o2p, NQH, HD, FD2_NS, pp);
            };
            constexpr int RT = 64;
            auto launch_fdw = [&](auto WC) {
                constexpr int WW = decltype(WC)::value;
                size_t sw = (size_t)RT * 2 * HD * sizeof(__nv_fp8_e4m3); // raw tile
                dim3 g1(FD2_NS, N_KV);
                k_attn_fdw<__nv_fp8_e4m3, WW, RT><<<g1, WW * 32, sw>>>(qp, HD, kc, vc, part,
                                                                       pp, N_KV, GQA, HD,
                                                                       scale);
                dim3 g2(NQH, WW);
                k_attn_fd_combine<<<g2, 256>>>(part, owp, NQH, HD, FD2_NS, pp);
            };
            auto fdw = [&] {
                if (W == 2) launch_fdw(std::integral_constant<int, 2>{});
                else if (W == 4) launch_fdw(std::integral_constant<int, 4>{});
                else launch_fdw(std::integral_constant<int, 8>{});
            };
            auto launch_fdw2 = [&](auto WC) {
                constexpr int WW = decltype(WC)::value;
                constexpr int RT2 = 64;
                size_t sw = (size_t)RT2 * 2 * HD * sizeof(__nv_fp8_e4m3);
                dim3 g1(FD2_NS, N_KV);
                k_attn_fdw2<__nv_fp8_e4m3, WW, RT2, FD2_NS><<<g1, WW * 3 * 32, sw>>>(
                    qp, HD, kc, vc, part, pp, N_KV, GQA, HD, scale);
                dim3 g2(NQH, WW);
                k_attn_fd_combine<<<g2, 256>>>(part, owp, NQH, HD, FD2_NS, pp);
            };
            auto fdw2 = [&] {
                if (W == 2) launch_fdw2(std::integral_constant<int, 2>{});
                else if (W == 4) launch_fdw2(std::integral_constant<int, 4>{});
                else launch_fdw2(std::integral_constant<int, 8>{});
            };
            auto launch_fdw3 = [&](auto WC) {
                constexpr int WW = decltype(WC)::value;
                constexpr int RT3 = 64;
                size_t sw = (size_t)RT3 * 2 * HD * sizeof(__nv_fp8_e4m3) +
                            (size_t)WW * 6 * 256 * 4;
                static bool attr3[9] = {};
                if (!attr3[WW]) {
                    CUDA_CHECK(cudaFuncSetAttribute(
                        k_attn_fdw3<__nv_fp8_e4m3, WW, RT3, FD2_NS>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, sw));
                    attr3[WW] = true;
                }
                dim3 g1(FD2_NS, N_KV);
                k_attn_fdw3<__nv_fp8_e4m3, WW, RT3, FD2_NS><<<g1, WW * 3 * 32, sw>>>(
                    qp, HD, kc, vc, part, pp, N_KV, GQA, HD, scale);
                dim3 g2(NQH, WW);
                k_attn_fd_combine<<<g2, 256>>>(part, owp, NQH, HD, FD2_NS, pp);
            };
            auto fdw3 = [&] {
                if (W == 2) launch_fdw3(std::integral_constant<int, 2>{});
                else if (W == 4) launch_fdw3(std::integral_constant<int, 4>{});
                else launch_fdw3(std::integral_constant<int, 8>{});
            };
            // fdmma: the tensor-core shared-KV kernel (W>=4; W=2 stays fd2)
            fdmma::FCP3 mqp{};
            fdmma::FIP3 mpp{};
            for (int t = 0; t < W; t++) { mqp.p[t] = qp.p[t]; mpp.p[t] = pp.p[t]; }
            auto fdmma_leg = [&] {
                fdmma::launch_fdmma(mqp, HD, kc, vc, part, mpp, N_KV, GQA, HD, scale,
                                    FD2_NS, W, 0);
                dim3 g2(NQH, W);
                k_attn_fd_combine<<<g2, 256>>>(part, owp, NQH, HD, FD2_NS, pp);
            };
            // correctness: fdw vs fd2 outputs (fp order differs -> tolerance)
            CUDA_CHECK(cudaMemset(part, 0, (size_t)MAXW * NQH * FD2_NS * FD_ST * 4));
            fd2();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemset(part, 0, (size_t)MAXW * NQH * FD2_NS * FD_ST * 4));
            fdw();
            CUDA_CHECK(cudaDeviceSynchronize());
            double mre = 0;
            int nan_a = 0, nan_b = 0;
            float a0 = 0, b0 = 0;
            for (int t = 0; t < W; t++) {
                std::vector<float> a(NQH * HD), b(NQH * HD);
                CUDA_CHECK(cudaMemcpy(a.data(), o2[t], NQH * HD * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(b.data(), ow[t], NQH * HD * 4, cudaMemcpyDeviceToHost));
                if (t == 0) { a0 = a[0]; b0 = b[0]; }
                double rms = 0;
                for (auto v : a) rms += (double)v * v;
                rms = sqrt(rms / a.size()) + 1e-12;
                for (size_t i = 0; i < a.size(); i++) {
                    if (std::isnan(a[i])) nan_a++;
                    if (std::isnan(b[i])) nan_b++;
                    double d = fabs((double)a[i] - b[i]) / rms;
                    if (d > mre) mre = d;
                }
            }
            if (nan_a || nan_b) printf("  W=%d NAN a=%d b=%d\n", W, nan_a, nan_b);
            // v2 correctness against fd2 as well
            CUDA_CHECK(cudaMemset(part, 0, (size_t)MAXW * NQH * FD2_NS * FD_ST * 4));
            fdw2();
            CUDA_CHECK(cudaDeviceSynchronize());
            double mre2 = 0;
            for (int t = 0; t < W; t++) {
                std::vector<float> a(NQH * HD), b(NQH * HD);
                CUDA_CHECK(cudaMemcpy(a.data(), o2[t], NQH * HD * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(b.data(), ow[t], NQH * HD * 4, cudaMemcpyDeviceToHost));
                double rms = 0;
                for (auto v : a) rms += (double)v * v;
                rms = sqrt(rms / a.size()) + 1e-12;
                for (size_t i = 0; i < a.size(); i++) {
                    double d = fabs((double)a[i] - b[i]) / rms;
                    if (d > mre2) mre2 = d;
                }
            }
            // v3 correctness
            CUDA_CHECK(cudaMemset(part, 0, (size_t)MAXW * NQH * FD2_NS * FD_ST * 4));
            fdw3();
            CUDA_CHECK(cudaDeviceSynchronize());
            double mre3 = 0;
            for (int t = 0; t < W; t++) {
                std::vector<float> a(NQH * HD), b(NQH * HD);
                CUDA_CHECK(cudaMemcpy(a.data(), o2[t], NQH * HD * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(b.data(), ow[t], NQH * HD * 4, cudaMemcpyDeviceToHost));
                double rms = 0;
                for (auto v : a) rms += (double)v * v;
                rms = sqrt(rms / a.size()) + 1e-12;
                for (size_t i = 0; i < a.size(); i++) {
                    double d = fabs((double)a[i] - b[i]) / rms;
                    if (d > mre3) mre3 = d;
                }
            }
            // fdmma correctness vs fd2 (both fp8 attention; fdmma adds Q/P
            // e4m3 -- expect physics-class rel, not fp-reorder-class)
            double mrem = -1;
            if (W >= 4) {
                CUDA_CHECK(cudaMemset(part, 0, (size_t)MAXW * NQH * FD2_NS * FD_ST * 4));
                fdmma_leg();
                CUDA_CHECK(cudaDeviceSynchronize());
                mrem = 0;
                for (int t = 0; t < W; t++) {
                    std::vector<float> a(NQH * HD), b(NQH * HD);
                    CUDA_CHECK(cudaMemcpy(a.data(), o2[t], NQH * HD * 4, cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(b.data(), ow[t], NQH * HD * 4, cudaMemcpyDeviceToHost));
                    double rms = 0;
                    for (auto v : a) rms += (double)v * v;
                    rms = sqrt(rms / a.size()) + 1e-12;
                    for (size_t i = 0; i < a.size(); i++) {
                        double d = fabs((double)a[i] - b[i]) / rms;
                        if (d > mrem) mrem = d;
                    }
                }
            }
            double ms2 = timeit(fd2, 100), msw = timeit(fdw, 100), msw2 = timeit(fdw2, 100);
            double msw3 = timeit(fdw3, 100);
            double msm = W >= 4 ? timeit(fdmma_leg, 100) : 0;
            printf("  W=%d fd2 %7.1f | v1 %7.1f %.2fx | v2 %7.1f %.2fx | v3 %7.1f %.2fx "
                   "rel %.1e",
                   W, ms2 * 1e3, msw * 1e3, ms2 / msw, msw2 * 1e3, ms2 / msw2, msw3 * 1e3,
                   ms2 / msw3, mre3);
            if (W >= 4)
                printf(" | FDMMA %7.1f %.2fx rel %.1e\n", msm * 1e3, ms2 / msm, mrem);
            else
                printf("\n");
        }
        for (int t = 0; t < MAXW; t++) {
            CUDA_CHECK(cudaFree(q[t]));
            CUDA_CHECK(cudaFree(o2[t]));
            CUDA_CHECK(cudaFree(ow[t]));
            CUDA_CHECK(cudaFree(posd[t]));
        }
        CUDA_CHECK(cudaFree(kc));
        CUDA_CHECK(cudaFree(vc));
        CUDA_CHECK(cudaFree(part));
    }
    printf("GO bar: fdw >= 2x fd2 at 61K W=8, maxrel < 1e-4.\n");
    return 0;
}
