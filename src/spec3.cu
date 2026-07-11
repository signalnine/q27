#include <cfloat>
#include <cstdlib>
#include <cstring>

#include "cuda_common.h"
#include "fdmma.cuh"
#include "spec3.cuh"
#include "turbo3.cuh"

namespace q27k {

__device__ __forceinline__ float wred(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__global__ void k_l2norm3(__grid_constant__ const P3 xp, int head_dim, float eps) {
    float* xh = xp.p[blockIdx.y] + (size_t)blockIdx.x * head_dim;
    __shared__ float sh[128];
    float acc = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) acc += xh[i] * xh[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(fmaxf(sh[0], eps * eps));
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) xh[i] *= inv;
}
void l2norm3(P3 x, int n_heads, int head_dim, float eps, cudaStream_t st, int ntok) {
    dim3 g(n_heads, ntok);
    k_l2norm3<<<g, 128, 0, st>>>(x, head_dim, eps);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_gemv_f16_3(const __half* __restrict__ W, __grid_constant__ const CP3 xp,
                             __grid_constant__ const P3 yp, int64_t cols) {
    const float* x = xp.p[blockIdx.y];
    const __half* wr = W + (size_t)blockIdx.x * cols;
    float acc = 0.f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
        acc += __half2float(wr[c]) * x[c];
    __shared__ float sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) yp.p[blockIdx.y][blockIdx.x] = sh[0];
}
void gemv_f16_3(const __half* W, CP3 x, P3 y, int64_t rows, int64_t cols, cudaStream_t st,
                int ntok) {
    dim3 g((unsigned)rows, ntok);
    k_gemv_f16_3<<<g, 256, 0, st>>>(W, x, y, cols);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_gdn_gates3(__grid_constant__ const CP3 ar, __grid_constant__ const CP3 br,
                             const float* __restrict__ a, const float* __restrict__ dt,
                             __grid_constant__ const P3 g, __grid_constant__ const P3 b,
                             int n) {
    int h = threadIdx.x;
    if (h >= n) return;
    int t = blockIdx.x;
    float x = ar.p[t][h] + dt[h];
    float sp = x > 20.f ? x : log1pf(expf(x));
    g.p[t][h] = a[h] * sp;
    b.p[t][h] = 1.0f / (1.0f + expf(-br.p[t][h]));
}
void gdn_gates3(CP3 ar, CP3 br, const float* a, const float* dt, P3 g, P3 b, int n,
                cudaStream_t st, int ntok) {
    k_gdn_gates3<<<ntok, 64, 0, st>>>(ar, br, a, dt, g, b, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_gated_norm3(__grid_constant__ const CP3 op, const float* __restrict__ w,
                              __grid_constant__ const CP3 zp, __grid_constant__ const P3 outp,
                              int head_dim, float eps) {
    const int t = blockIdx.y;
    const float* oh = op.p[t] + (size_t)blockIdx.x * head_dim;
    const float* zh = zp.p[t] + (size_t)blockIdx.x * head_dim;
    float* yh = outp.p[t] + (size_t)blockIdx.x * head_dim;
    __shared__ float sh[128];
    float acc = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) acc += oh[i] * oh[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float zi = zh[i];
        yh[i] = oh[i] * inv * w[i] * (zi / (1.0f + expf(-zi)));
    }
}
void gated_norm3(CP3 o, const float* w, CP3 z, P3 out, int n_heads, int head_dim, float eps,
                 cudaStream_t st, int ntok) {
    dim3 g(n_heads, ntok);
    k_gated_norm3<<<g, 128, 0, st>>>(o, w, z, out, head_dim, eps);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_sigmoid_gate3(__grid_constant__ const P3 outp,
                                __grid_constant__ const CP3 qgp, int head_dim, int n) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    int t = blockIdx.y;
    int h = e / head_dim, d = e % head_dim;
    float gv = qgp.p[t][(size_t)h * 2 * head_dim + head_dim + d];
    outp.p[t][e] *= 1.0f / (1.0f + expf(-gv));
}
void sigmoid_gate3(P3 out, CP3 qg, int n_heads, int head_dim, cudaStream_t st, int ntok) {
    int n = n_heads * head_dim;
    dim3 g((n + 255) / 256, ntok);
    k_sigmoid_gate3<<<g, 256, 0, st>>>(out, qg, head_dim, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_rope3(__grid_constant__ const P3 xp, int head_dim, int n_rot, int stride,
                        __grid_constant__ const IP3 pos, float freq_base) {
    const int t = blockIdx.y;
    float* xh = xp.p[t] + (size_t)blockIdx.x * stride;
    int d = threadIdx.x;
    if (d >= n_rot / 2) return;
    float theta = (float)(*pos.p[t]) * powf(freq_base, -2.0f * d / n_rot);
    float c = cosf(theta), s = sinf(theta);
    float x0 = xh[d], x1 = xh[d + n_rot / 2];
    xh[d] = x0 * c - x1 * s;
    xh[d + n_rot / 2] = x0 * s + x1 * c;
}
void rope3(P3 x, int n_heads, int head_dim, int n_rot, int stride, IP3 pos, float freq_base,
           cudaStream_t st, int ntok) {
    dim3 g(n_heads, ntok);
    k_rope3<<<g, 32, 0, st>>>(x, head_dim, n_rot, stride, pos, freq_base);
    CUDA_CHECK(cudaGetLastError());
}

template <typename CT>
__global__ void k_kv_store3(__grid_constant__ const CP3 kp, __grid_constant__ const CP3 vp,
                            CT* __restrict__ kc, CT* __restrict__ vc,
                            __grid_constant__ const IP3 pos, int rowlen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowlen) return;
    int t = blockIdx.y;
    size_t off = (size_t)(*pos.p[t]) * rowlen + i;
    kv_set(kc[off], kp.p[t][i]);
    kv_set(vc[off], vp.p[t][i]);
}
void kv_store3(CP3 k, CP3 v, void* kc, void* vc, IP3 pos, int rowlen, cudaStream_t st,
               int ntok, bool fp8) {
    dim3 g((rowlen + 255) / 256, ntok);
    if (fp8)
        k_kv_store3<<<g, 256, 0, st>>>(k, v, (__nv_fp8_e4m3*)kc, (__nv_fp8_e4m3*)vc, pos,
                                       rowlen);
    else
        k_kv_store3<<<g, 256, 0, st>>>(k, v, (__half*)kc, (__half*)vc, pos, rowlen);
    CUDA_CHECK(cudaGetLastError());
}

// ---- turbo3 KV (Q27_KV=turbo3|turbo3v; format src/turbo3.cuh; port spec
// docs/plans/2026-07-11-turbo3-kv-port-spec.md).
//
// Store: one CUDA block == one 128-group; 128 threads cooperatively
// L2-normalize (pre-rotation), forward-WHT, and 3-bit-quantize one K or V
// group into its 50-byte block, norm corrected by grp_norm/recon_norm. All
// reductions and the butterfly are fixed-order => run-to-run deterministic.
// The CPU oracle sums norms serially, so tree-order skew can flip elements
// sitting exactly on a centroid midpoint (rare; the block stays
// self-consistent because idx and corrected norm derive from the same pass).
// k_plain (turbo3v): the K side bypasses quantization and stores plain fp16
// rows -- the GQA=6 escape if turbo3-K craters (port-spec risk section).
// Assumes head_dim == 256 (two groups/head), like the whole fd2 family.
__global__ void k_kv_store_t3(__grid_constant__ const CP3 kp, __grid_constant__ const CP3 vp,
                              void* __restrict__ kc, void* __restrict__ vc,
                              __grid_constant__ const IP3 pos, int n_kv_heads, int head_dim,
                              int k_plain) {
    const int t = blockIdx.z, j = threadIdx.x;
    const bool is_v = blockIdx.y == 1;
    const int h = blockIdx.x >> 1, g = blockIdx.x & 1;
    const int p = *pos.p[t];
    const float* src = (is_v ? vp.p[t] : kp.p[t]) + (size_t)h * head_dim + g * 128;
    if (!is_v && k_plain) {
        ((__half*)kc)[(size_t)p * n_kv_heads * head_dim + (size_t)h * head_dim + g * 128 + j] =
            __float2half_rn(src[j]);
        return;
    }
    q27turbo::block_turbo3* dst = (q27turbo::block_turbo3*)(is_v ? vc : kc) +
                                  (size_t)p * n_kv_heads * 2 + h * 2 + g;
    __shared__ float xs[128], red[128];
    float x = src[j];
    xs[j] = x;
    red[j] = x * x;
    __syncthreads();
#pragma unroll
    for (int s = 64; s > 0; s >>= 1) {
        if (j < s) red[j] += red[j + s];
        __syncthreads();
    }
    float gn = sqrtf(red[0]);
    float inv = gn > 1e-10f ? 1.f / gn : 0.f;
    xs[j] = xs[j] * inv * q27turbo::TURBO_S1[j];
    q27turbo::turbo3_butterfly128(xs, j);
    float w = xs[j] * (q27turbo::TURBO_INV_SQRT_128 * q27turbo::TURBO_S2[j]);
    int idx = q27turbo::turbo3_nearest(w);
    red[j] = q27turbo::TURBO_CENTROIDS_3BIT[idx] * q27turbo::TURBO_CENTROIDS_3BIT[idx];
    __syncthreads();
#pragma unroll
    for (int s = 64; s > 0; s >>= 1) {
        if (j < s) red[j] += red[j + s];
        __syncthreads();
    }
    float rn = sqrtf(red[0]);
    float corr = rn > 1e-10f ? gn / rn : gn;
    // warp-cooperative pack: 4 lanes share a qs byte (shfl), ballot -> signs
    int low2 = idx & 3;
    unsigned bal = __ballot_sync(0xffffffffu, idx & 4);
    int l1 = __shfl_down_sync(0xffffffffu, low2, 1);
    int l2 = __shfl_down_sync(0xffffffffu, low2, 2);
    int l3 = __shfl_down_sync(0xffffffffu, low2, 3);
    if ((j & 3) == 0)
        dst->qs[j >> 2] = (uint8_t)(low2 | (l1 << 2) | (l2 << 4) | (l3 << 6));
    if ((j & 7) == 0) dst->signs[j >> 3] = (uint8_t)((bal >> (j & 31)) & 0xFF);
    if (j == 0) dst->norm = __float2half(corr);
}

void kv_store_t3(CP3 k, CP3 v, void* kc, void* vc, IP3 pos, int n_kv_heads, int head_dim,
                 cudaStream_t st, int ntok, bool k_plain) {
    dim3 g(n_kv_heads * (head_dim >> 7), 2, ntok); // x: (head, group); y: K,V
    k_kv_store_t3<<<g, 128, 0, st>>>(k, v, kc, vc, pos, n_kv_heads, head_dim, k_plain);
    CUDA_CHECK(cudaGetLastError());
}

// Per-128-group WHT rotate: forward = s1 -> butterfly -> inv_sqrt*s2 (the
// store kernel's rotation, applied to Q so <WHT q, WHT K> == <q, K>);
// inverse = s2 -> butterfly -> inv_sqrt*s1 (un-rotates the pooled attention
// output; V's inverse distributes over the softmax-weighted sum). Operand
// order matches the CPU fwht exactly => device output is bitwise CPU-equal.
template <bool INV>
__global__ void k_wht3(__grid_constant__ const P3 xp, int gph, int stride) {
    const int t = blockIdx.y, j = threadIdx.x;
    const int hh = blockIdx.x / gph, g = blockIdx.x % gph;
    float* xh = xp.p[t] + (size_t)hh * stride + g * 128;
    __shared__ float xs[128];
    xs[j] = xh[j] * (INV ? q27turbo::TURBO_S2[j] : q27turbo::TURBO_S1[j]);
    q27turbo::turbo3_butterfly128(xs, j);
    xh[j] = xs[j] * (q27turbo::TURBO_INV_SQRT_128 *
                     (INV ? q27turbo::TURBO_S1[j] : q27turbo::TURBO_S2[j]));
}

void wht3(P3 x, int n_heads, int head_dim, int stride, bool inv, cudaStream_t st, int ntok) {
    dim3 g(n_heads * (head_dim >> 7), ntok);
    if (inv)
        k_wht3<true><<<g, 128, 0, st>>>(x, head_dim >> 7, stride);
    else
        k_wht3<false><<<g, 128, 0, st>>>(x, head_dim >> 7, stride);
    CUDA_CHECK(cudaGetLastError());
}

static inline P3 out2p(float* o) { return P3{{o, o, o, o}}; }

// Flash-decode: grid (kv_head, split, token). Each block covers one position
// range for ALL 6 GQA q-heads of its kv head (K/V read once, not 6x), online
// softmax per warp, block-merged partials {m, l, acc[256]} to scratch; a
// combine kernel merges splits. Works for 1..4 tokens via gridDim.z.
template <typename CT>
__global__ void k_attn_fd(__grid_constant__ const CP3 qp, int q_stride, const CT* __restrict__ kc,
                          const CT* __restrict__ vc, float* __restrict__ part, IP3 pos,
                          int n_kv_heads, int gqa, int head_dim, float scale) {
    const int kvh = blockIdx.x, sp = blockIdx.y, t = blockIdx.z;
    const int seq = *pos.p[t] + 1;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    constexpr int NW = 8;

    extern __shared__ float smem[];
    float* s_q = smem;                  // [6][256]
    float* s_acc = smem + 6 * 256;      // [NW][6][256]
    __shared__ float s_ml[NW][6][2];

    for (int idx = threadIdx.x; idx < gqa * head_dim; idx += blockDim.x)
        s_q[idx] = qp.p[t][(size_t)(kvh * gqa + idx / head_dim) * q_stride + idx % head_dim];
    for (int idx = threadIdx.x; idx < NW * 6 * 256; idx += blockDim.x) s_acc[idx] = 0.f;
    __syncthreads();

    const int chunk = (seq + FD_NS - 1) / FD_NS;
    const int p_lo = sp * chunk, p_hi = min(seq, p_lo + chunk);

    float m[6], l[6];
#pragma unroll
    for (int j = 0; j < 6; j++) { m[j] = -FLT_MAX; l[j] = 0.f; }
    float* accw = s_acc + warp * 6 * 256;

    for (int p = p_lo + warp; p < p_hi; p += NW) {
        const CT* kp = kc + ((size_t)p * n_kv_heads + kvh) * head_dim;
        const CT* vp = vc + ((size_t)p * n_kv_heads + kvh) * head_dim;
        float kv[8], vv[8];
#pragma unroll
        for (int u = 0; u < 8; u++) {
            kv[u] = kv2f(kp[lane + 32 * u]);
            vv[u] = kv2f(vp[lane + 32 * u]);
        }
#pragma unroll
        for (int j = 0; j < 6; j++) {
            float d = 0.f;
#pragma unroll
            for (int u = 0; u < 8; u++) d += s_q[j * 256 + lane + 32 * u] * kv[u];
            for (int off = 16; off > 0; off >>= 1) d += __shfl_down_sync(0xffffffff, d, off);
            d = __shfl_sync(0xffffffff, d, 0) * scale;
            float mn = fmaxf(m[j], d);
            float so = expf(m[j] - mn), w = expf(d - mn);
            l[j] = l[j] * so + w;
            m[j] = mn;
#pragma unroll
            for (int u = 0; u < 8; u++) {
                float* a = accw + j * 256 + lane + 32 * u;
                *a = *a * so + w * vv[u];
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 6; j++) {
        if (lane == 0) { s_ml[warp][j][0] = m[j]; s_ml[warp][j][1] = l[j]; }
    }
    __syncthreads();

    // merge the 8 warps' partials -> one {m, l, acc} per head for this split
    for (int j = warp; j < 6; j += NW) { // warps 0..5 each own a head
        float mb = -FLT_MAX;
        for (int w = 0; w < NW; w++) mb = fmaxf(mb, s_ml[w][j][0]);
        float lb = 0.f;
        float sc[NW];
        for (int w = 0; w < NW; w++) {
            sc[w] = s_ml[w][j][0] == -FLT_MAX ? 0.f : expf(s_ml[w][j][0] - mb);
            lb += s_ml[w][j][1] * sc[w];
        }
        size_t pair = (size_t)t * (n_kv_heads * gqa) + kvh * gqa + j;
        float* dst = part + (pair * FD_NS + sp) * FD_ST;
        if (lane == 0) { dst[0] = mb; dst[1] = lb; }
        for (int d = lane; d < head_dim; d += 32) {
            float a = 0.f;
            for (int w = 0; w < NW; w++) a += s_acc[(w * 6 + j) * 256 + d] * sc[w];
            dst[2 + d] = a;
        }
    }
}

__global__ void k_attn_fd_combine(const float* __restrict__ part,
                                  __grid_constant__ const P3 outp, int n_heads,
                                  int head_dim, int ns, IP3 pos) {
    const int h = blockIdx.x, t = blockIdx.y;
    // splits at sp*chunk >= seq are empty; fd2 never writes them, and for
    // v1 they hold {-FLT_MAX, 0, ...} which contribute exactly zero -- so
    // skipping them is bitwise-identical for v1 and REQUIRED for fd2
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

// ---- attn-fd2 (docs/attn-fd2-design.md): register-accumulator flash-decode.
// Same grid and partial layout as v1; the per-warp accumulator moves from a
// 55KB smem array (which capped the SM at 1 block / 8 resident warps -- the
// measured latency-hiding ceiling, BUILDLOG 2026-07-04 night) into 48
// registers per lane. Lane owns dims D(l) = {4l..4l+3, 128+4l..128+4l+3},
// chosen so K/V rows load as two 4-byte words per lane instead of 16 single
// bytes. smem shrinks to s_q + a 6KB cross-warp merge buffer (~12.3KB).
// The merge is barrier-SERIALIZED in warp order -- smem atomics would
// reorder fp adds run-to-run and break the bitwise repeat-run determinism
// the transient-detection methodology depends on.

template <typename CT>
__device__ __forceinline__ void fd2_ld8(const CT* __restrict__ row, int lane, float* o) {
    if constexpr (sizeof(CT) == 1) {
        // fp8 row = 256B, 4B-aligned at 4*lane
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
        // half row = 512B, 8B-aligned at 8*lane
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

// fd2 lane load from a pair of turbo3 blocks: lane l owns dims {4l..4l+3}
// (block b2[0]) and {128+4l..128+4l+3} (b2[1]) -- the fd2 lane layout maps
// 1:1 onto the two 128-groups. One qs byte (qs[l]) covers the lane's 4 dims
// in a block; sign byte signs[l>>1], bits (l&1)*4+i; centroid LUT scaled by
// the block's hoisted fp16 norm.
__device__ __forceinline__ void fd2_ld8_t3(const q27turbo::block_turbo3* __restrict__ b2,
                                           int lane, float* o) {
#pragma unroll
    for (int g = 0; g < 2; g++) {
        const q27turbo::block_turbo3* b = b2 + g;
        float norm = __half2float(b->norm);
        uint8_t q = b->qs[lane];
        uint8_t s = b->signs[lane >> 1];
        int sh = (lane & 1) * 4;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = ((q >> (2 * i)) & 3) | (((s >> (sh + i)) & 1) << 2);
            o[4 * g + i] = q27turbo::TURBO_CENTROIDS_3BIT[idx] * norm;
        }
    }
}

template <typename CT, int NW>
__global__ void k_attn_fd2(__grid_constant__ const CP3 qp, int q_stride,
                           const CT* __restrict__ kc,
                           const CT* __restrict__ vc, float* __restrict__ part, IP3 pos,
                           int n_kv_heads, int gqa, int head_dim, float scale) {
    // P14: lane (token) is the FASTEST-varying grid axis so the vw same-split
    // blocks for a given (head, split) co-schedule and share the ~1MB KV chunk
    // in L2 instead of each lane re-streaming it from DRAM (Task 1 R~=4.25).
    // PURE INDEX REMAP: per-block work, per-lane fp accumulation order, and the
    // scratch-cell addressing per (head, split, lane) are byte-for-byte
    // unchanged; only the block enumeration order differs. Launch grid is
    // dim3(ntok, FD2_NS, n_kv_heads) to match (see attn_decode3_fd2).
    const int t = blockIdx.x, sp = blockIdx.y, kvh = blockIdx.z;
    const int seq = *pos.p[t] + 1;
    // empty split: no partial is written; the combine kernel derives the
    // used-split count from pos and never reads these slots. Keeps high
    // FD2_NS free at short ctx (measured +2.4%/round at 2K without this).
    if (sp * ((seq + FD2_NS - 1) / FD2_NS) >= seq) return;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;

    extern __shared__ float smem[];
    float* s_q = smem;              // [6][256]
    float* s_mrg = smem + 6 * 256;  // [6][256] cross-warp accumulator merge
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

    // serialized rescale-add of each warp's register accumulator into s_mrg
    // (fixed warp order 0..NW-1 => bitwise-deterministic across runs)
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

    // per-head block {m, l} + merged acc -> partial (layout identical to v1;
    // the combine kernel is untouched)
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

// k_attn_fd2 with turbo3 K/V loads (KT3=false: plain fp16 K rows, the
// turbo3v split). Row addressing is in 50-byte blocks (pos*n_kv_heads*2 +
// kvh*2), so this is a separate kernel rather than a CT overload (port spec
// item 3); the softmax / online-max / serialized warp merge / partial layout
// are copied byte-identical from k_attn_fd2 -- only the loads differ. The
// fp8/fp16 k_attn_fd2 instantiations are textually untouched (bitwise gate).
// Q arrives already WHT-rotated when KT3 (engine rotates after rope);
// VKQ accumulates in the rotated V basis -- the engine's single post-combine
// inverse-WHT un-rotates the pooled output (linearity).
template <bool KT3, int NW>
__global__ void k_attn_fd2_t3(__grid_constant__ const CP3 qp, int q_stride,
                              const void* __restrict__ kc, const void* __restrict__ vc,
                              float* __restrict__ part, IP3 pos, int n_kv_heads, int gqa,
                              int head_dim, float scale) {
    const int t = blockIdx.x, sp = blockIdx.y, kvh = blockIdx.z;
    const int seq = *pos.p[t] + 1;
    if (sp * ((seq + FD2_NS - 1) / FD2_NS) >= seq) return;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;

    extern __shared__ float smem[];
    float* s_q = smem;              // [6][256]
    float* s_mrg = smem + 6 * 256;  // [6][256] cross-warp accumulator merge
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
        float kv[8], vv[8];
        if constexpr (KT3)
            fd2_ld8_t3((const q27turbo::block_turbo3*)kc + ((size_t)p * n_kv_heads + kvh) * 2,
                       lane, kv);
        else
            fd2_ld8((const __half*)kc + ((size_t)p * n_kv_heads + kvh) * head_dim, lane, kv);
        fd2_ld8_t3((const q27turbo::block_turbo3*)vc + ((size_t)p * n_kv_heads + kvh) * 2,
                   lane, vv);
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

    // serialized rescale-add (fixed warp order => bitwise-deterministic)
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

// per-instantiation one-shot smem-attribute raise for k_attn_fd<CT>
template <typename CT>
static void fd_setattr(size_t sm) {
    static bool attr = false;
    if (!attr) {
        CUDA_CHECK(cudaFuncSetAttribute(k_attn_fd<CT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
        attr = true;
    }
}

template <typename CT>
static void fd_launch(CP3 q, int q_stride, const void* kc, const void* vc, float* scratch,
                      IP3 pos, int n_kv_heads, int gqa, int head_dim, float scale, size_t sm,
                      int ntok, cudaStream_t st) {
    fd_setattr<CT>(sm);
    dim3 g1(n_kv_heads, FD_NS, ntok);
    k_attn_fd<CT><<<g1, 256, sm, st>>>(q, q_stride, (const CT*)kc, (const CT*)vc, scratch, pos,
                                       n_kv_heads, gqa, head_dim, scale);
}

void attn_decode3_fd2(CP3 q, int q_stride, const void* kc, const void* vc, P3 out,
                      float* scratch, IP3 pos, int max_ctx, int n_q_heads, int n_kv_heads,
                      int head_dim, float scale, cudaStream_t st, int ntok, int kvk) {
    (void)max_ctx;
    int gqa = n_q_heads / n_kv_heads;
    // NW=4 (128 threads): probe-favored -- more blocks/SM for latency hiding.
    // smem 12.3KB, under the 48KB default: no cudaFuncSetAttribute needed.
    constexpr int NW2 = 4;
    size_t sm = (size_t)(2 * 6) * 256 * sizeof(float);
    dim3 g1(ntok, FD2_NS, n_kv_heads);  // P14: lane (x) fastest -> cross-lane KV L2 reuse
    if (kvk == KV_T3)
        k_attn_fd2_t3<true, NW2><<<g1, NW2 * 32, sm, st>>>(q, q_stride, kc, vc, scratch, pos,
                                                           n_kv_heads, gqa, head_dim, scale);
    else if (kvk == KV_T3V)
        k_attn_fd2_t3<false, NW2><<<g1, NW2 * 32, sm, st>>>(q, q_stride, kc, vc, scratch, pos,
                                                            n_kv_heads, gqa, head_dim, scale);
    else if (kvk == KV_FP8)
        k_attn_fd2<__nv_fp8_e4m3, NW2><<<g1, NW2 * 32, sm, st>>>(
            q, q_stride, (const __nv_fp8_e4m3*)kc, (const __nv_fp8_e4m3*)vc, scratch, pos,
            n_kv_heads, gqa, head_dim, scale);
    else
        k_attn_fd2<__half, NW2><<<g1, NW2 * 32, sm, st>>>(q, q_stride, (const __half*)kc,
                                                          (const __half*)vc, scratch, pos,
                                                          n_kv_heads, gqa, head_dim, scale);
    dim3 g2(n_q_heads, ntok);
    k_attn_fd_combine<<<g2, 256, 0, st>>>(scratch, out, n_q_heads, head_dim, FD2_NS, pos);
    CUDA_CHECK(cudaGetLastError());
}

void attn_decode3(CP3 q, int q_stride, const void* kc, const void* vc, P3 out, float* scratch,
                  IP3 pos, int max_ctx, int n_q_heads, int n_kv_heads, int head_dim, float scale,
                  cudaStream_t st, int ntok, int kvk) {
    // turbo3 has exactly one read path (fd2); Q27_FD=v1/mma fall through here
    // silently -- neither kernel family has a block-cache leg.
    if (kvk >= KV_T3) {
        attn_decode3_fd2(q, q_stride, kc, vc, out, scratch, pos, max_ctx, n_q_heads,
                         n_kv_heads, head_dim, scale, st, ntok, kvk);
        return;
    }
    const bool fp8 = kvk == KV_FP8;
    // fd2 is the default; Q27_FD=v1 keeps the original kernel (bit-for-bit
    // old behavior, incl. the retired bitwise canonical). Read at launch
    // time: graph capture bakes the choice for the process lifetime.
    const char* fd = getenv("Q27_FD");
    // Q27_FD=mma: fp8-MMA shared-KV verify attention (src/fdmma.cuh; plan
    // docs/plans/2026-07-10-fdmma-verify-attn.md). One KV pass scores all
    // lanes -- measured 3.65x over fd2 at 61K W=8, near-flat in W. Engages
    // only for fp8 KV, width 4..8, sm_89+ (the e4m3 MMA is a no-op stub
    // below); everything else falls through to fd2, so W=2..3 rounds and
    // the plain path are untouched. Numerics are tolerance-class (fp8 Q/P)
    // -- OPT-IN until the acceptance A/B replay gate clears a default flip.
    if (fd && strcmp(fd, "mma") == 0 && fp8 && ntok >= 4 && ntok <= 12) {
        static int arch = -1;
        if (arch < 0) {
            int dev, mj, mn;
            CUDA_CHECK(cudaGetDevice(&dev));
            CUDA_CHECK(cudaDeviceGetAttribute(&mj, cudaDevAttrComputeCapabilityMajor, dev));
            CUDA_CHECK(cudaDeviceGetAttribute(&mn, cudaDevAttrComputeCapabilityMinor, dev));
            arch = mj * 10 + mn;
        }
        if (arch >= 89) {
            fdmma::FCP3 mq;
            fdmma::FIP3 mp;
            for (int t = 0; t < 16; t++) { mq.p[t] = q.p[t]; mp.p[t] = pos.p[t]; }
            // width-12 P2 (review): honor the dispatch's return -- an
            // uninstantiated width must FALL THROUGH to fd2, not silently
            // combine unwritten partials.
            // tuning 2026-07-10: stages=1 (single-buffered K/V, 2 CTAs/SM,
            // 168 regs) is the DEFAULT -- bitwise-identical to the shipped
            // 2-stage kernel (shared fdmma_tile_compute) and +5..26% at
            // every measured (ctx, W); Q27_FDMMA_STAGES=2 restores the old
            // staging for A/B. Read once; graphs bake it at capture.
            static const int fdmma_stages = [] {
                const char* e = getenv("Q27_FDMMA_STAGES");
                return e && atoi(e) == 2 ? 2 : 1;
            }();
            // split-count retune (tuning 2026-07-10): fdmma's grid is
            // (ns, kv_heads) with 2 CTAs/SM resident -- ns = SMs*2/kv_heads
            // fills EXACTLY one wave (85 on the 5090; 128 left a half-empty
            // second wave, +29-40% at 61K). fdmma-only: fd2 keeps FD2_NS
            // (its splits were swept for its own shape). Capped by
            // FD_MAXNS (scratch rows). NOT bitwise across ns values (split
            // boundaries move -> combine fp order) -- rebuild-class
            // tie-lottery, the regime the mma basin matrix cleared;
            // Q27_FDMMA_NS pins it for A/B.
            static const int fdmma_ns = [n_kv_heads] {
                if (const char* e = getenv("Q27_FDMMA_NS")) {
                    int v = atoi(e);
                    if (v >= 1 && v <= FD_MAXNS) return v;
                }
                int dev2, smc;
                CUDA_CHECK(cudaGetDevice(&dev2));
                CUDA_CHECK(cudaDeviceGetAttribute(&smc, cudaDevAttrMultiProcessorCount, dev2));
                int v = (smc * 2) / n_kv_heads;
                return v < 16 ? 16 : v > FD_MAXNS ? FD_MAXNS : v;
            }();
            if (fdmma::launch_fdmma(mq, q_stride, kc, vc, scratch, mp, n_kv_heads,
                                    n_q_heads / n_kv_heads, head_dim, scale, fdmma_ns, ntok, st,
                                    fdmma_stages)) {
                dim3 g2(n_q_heads, ntok);
                k_attn_fd_combine<<<g2, 256, 0, st>>>(scratch, out, n_q_heads, head_dim,
                                                      fdmma_ns, pos);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
        }
    }
    if (!fd || strcmp(fd, "v1") != 0) {
        attn_decode3_fd2(q, q_stride, kc, vc, out, scratch, pos, max_ctx, n_q_heads,
                         n_kv_heads, head_dim, scale, st, ntok, kvk);
        return;
    }
    (void)max_ctx;
    int gqa = n_q_heads / n_kv_heads;
    size_t sm = (size_t)(6 + 8 * 6) * 256 * sizeof(float);
    if (fp8)
        fd_launch<__nv_fp8_e4m3>(q, q_stride, kc, vc, scratch, pos, n_kv_heads, gqa, head_dim,
                                 scale, sm, ntok, st);
    else
        fd_launch<__half>(q, q_stride, kc, vc, scratch, pos, n_kv_heads, gqa, head_dim, scale,
                          sm, ntok, st);
    dim3 g2(n_q_heads, ntok);
    k_attn_fd_combine<<<g2, 256, 0, st>>>(scratch, out, n_q_heads, head_dim, FD_NS, pos);
    CUDA_CHECK(cudaGetLastError());
}

// single-token plain-path attention through the same flash-decode kernels
void attn_decode(const float* q, int q_stride, const void* kcache, const void* vcache,
                 float* out, float* scratch, const int* d_pos, int max_ctx, int n_q_heads,
                 int n_kv_heads, int head_dim, float scale, cudaStream_t st, int kvk) {
    CP3 qp{{q, q, q}};
    IP3 pp{{d_pos, d_pos, d_pos}};
    attn_decode3(qp, q_stride, kcache, vcache, out2p(out), scratch, pp, max_ctx, n_q_heads,
                 n_kv_heads, head_dim, scale, st, 1, kvk);
}

__global__ void k_embed3(const int8_t* __restrict__ W, const __half* __restrict__ S,
                         __grid_constant__ const IP3 tok, int64_t cols,
                         __grid_constant__ const P3 outp) {
    const int t = blockIdx.y;
    int64_t row = *tok.p[t];
    const int8_t* wr = W + row * cols;
    const __half* sr = S + row * (cols / 128);
    for (int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; c < cols;
         c += (int64_t)gridDim.x * blockDim.x)
        outp.p[t][c] = (float)wr[c] * __half2float(sr[c / 128]);
}
void embed3(const int8_t* W, const __half* S, IP3 tok, int64_t cols, P3 out, cudaStream_t st,
            int ntok) {
    dim3 g(8, ntok);
    k_embed3<<<g, 256, 0, st>>>(W, S, tok, cols, out);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_prep_round(const int* __restrict__ dP, const int* __restrict__ dtok,
                             __grid_constant__ const WIP3 pos_v,
                             __grid_constant__ const WIP3 pos_m, int nv, int nm, int* outcome) {
    int P = *dP;
    // verify lanes a.. (width-12: up to 12) and MTP draft positions (ladder
    // ceiling 7 -- policy-decoupled from verify width, plan 2026-07-10)
    for (int t = 0; t < nv; t++) *pos_v.p[t] = P + 1 + t;
    for (int k = 0; k < nm; k++) *pos_m.p[k] = P + 1 + k;
    outcome[1] = *dtok; // t1 snapshot (pre-round)
}
void prep_round(const int* d_P, const int* d_token, WIP3 pos_v, WIP3 pos_m, int nv, int nm,
                int* outcome, cudaStream_t st) {
    k_prep_round<<<1, 1, 0, st>>>(d_P, d_token, pos_v, pos_m, nv, nm, outcome);
    CUDA_CHECK(cudaGetLastError());
}

// width-12: lanes ride IP3/CP3 structs (the flat list capped at 8). The
// acceptance walk is the same leading-run chain as the old a1..a7 bools:
// draft k accepts iff k <= max_draft, all earlier drafts accepted, and
// lane k's argmax equals draft k. All 11 draft / 12 verdict slots are
// dereferenced unconditionally (engine allocates every lane; the old
// kernel read all 7/8 the same way) -- only slots < max_draft / n matter.
__global__ void k_finish_round(int* __restrict__ dP, int* __restrict__ dtok,
                               __grid_constant__ const IP3 drafts,
                               __grid_constant__ const IP3 verdicts,
                               __grid_constant__ const CP3 x1s, float* __restrict__ h_next,
                               int* __restrict__ outcome, int n_embd,
                               const int* __restrict__ cap, int max_draft) {
    int dr[11], v[12];
#pragma unroll
    for (int k = 0; k < 11; k++) dr[k] = *drafts.p[k];
#pragma unroll
    for (int t = 0; t < 12; t++) v[t] = *verdicts.p[t];
    // P12/P12b: max_draft gates depth to the verified columns (narrow-verify
    // graph). P7 (*cap): in-grammar rounds accept only the pending token;
    // drafts are unconstrained and must not commit past the constrained lane.
    int n = 1;
    if (!*cap) {
#pragma unroll
        for (int k = 1; k <= 11; k++)
            if (k <= max_draft && n == k && v[k - 1] == dr[k - 1]) n = k + 1;
    }
    const float* src = x1s.p[n - 1];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_embd; i += gridDim.x * blockDim.x)
        h_next[i] = src[i];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int nt = v[n - 1];
        *dtok = nt;
        *dP += n;
        outcome[0] = n;
        // width-12 outcome layout: [0]=n, [1]=t1(prep), [2..12]=dr1..dr11 (up
        // to 12 emitted tokens live in [1..n]), [13]=new pending.
#pragma unroll
        for (int k = 0; k < 11; k++) outcome[2 + k] = dr[k];
        outcome[13] = nt; // new pending token (P7: host grammar needs it pre-round)
    }
}
void finish_round(int* d_P, int* d_token, IP3 drafts, IP3 verdicts, CP3 x1s, float* h_next,
                  int* outcome, int n_embd, const int* cap, int max_draft, cudaStream_t st) {
    k_finish_round<<<4, 256, 0, st>>>(d_P, d_token, drafts, verdicts, x1s, h_next, outcome,
                                      n_embd, cap, max_draft);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace q27k
