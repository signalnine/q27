#include <cfloat>
#include <cstdlib>
#include <cstring>

#include "blocks.cuh" // Q27_CONV_OLDEST_FIRST / Q27_GDN_HEAD_TILE (M1 chunk twins)
#include "cuda_common.h"
#include "fdmma.cuh"
#include "spec3.cuh"
#include "turbo3.cuh"
#include "turbo5.cuh"
#include "i8g64.cuh"

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

// ---- M1 record+fold GDN verify chunks (batched-decode spec Appendix A) ----
// MIRROR WARNING (M1): the tap arithmetic below is a body twin of k_conv_step
// (blocks.cu) -- any arithmetic change there MUST be mirrored here and re-gated
// with ninv's CHUNK leg (bitwise, both arches). Lane L = blockIdx.y + 1; taps
// are the raw inputs L-3..L. Absolute input a >= 1 comes from lane a's qkv
// buffer; a <= 0 comes from the committed ring AFTER lane 0's in-place shift
// (slot a+2: ring = [x(-2), x(-1), x(0)] oldest-first). No ring writes.
__global__ void k_gdn_conv_chunk3(const float* __restrict__ ring,
                                  __grid_constant__ const CP3 qkv,
                                  const float* __restrict__ w,
                                  __grid_constant__ const P3 out, int channels) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int L = blockIdx.y + 1;
    const float* wc = w + (size_t)c * 4; // [channels][4], taps contiguous
    auto tap = [&](int a) {
        return a >= 1 ? qkv.p[a][c] : ring[(size_t)(a + 2) * channels + c];
    };
    float r0 = tap(L - 3), r1 = tap(L - 2), r2 = tap(L - 1), x = tap(L);
#if Q27_CONV_OLDEST_FIRST
    float acc = r0 * wc[0] + r1 * wc[1] + r2 * wc[2] + x * wc[3];
#else
    float acc = r0 * wc[3] + r1 * wc[2] + r2 * wc[1] + x * wc[0];
#endif
    out.p[L][c] = acc / (1.0f + expf(-acc)); // silu
}

void gdn_conv_chunk3(const float* ring, CP3 qkv, const float* convw, P3 out, int channels,
                     int nsp, cudaStream_t st) {
    dim3 g((channels + 255) / 256, nsp);
    k_gdn_conv_chunk3<<<g, 256, 0, st>>>(ring, qkv, convw, out, channels);
    CUDA_CHECK(cudaGetLastError());
}

// MIRROR WARNING (M1): the per-step body below is a smem twin of k_delta_step
// (blocks.cu) and k_delta_scan_T (prefill.cu) -- same 48x512 launch, same
// i-tile/j mapping, same part[0..3] reduction order, same decay/pred/d/update
// sequence. Any arithmetic change there MUST be mirrored here and re-gated
// with ninv's CHUNK leg. Committed S (post lane-0) is read once into 64KB
// dynamic smem; each speculative lane's o is written; S is NEVER written --
// the commit Fold (delta_scan_seq over the recorded rows) advances it.
// 2026-09-07 register-resident state: the [128][128] state chain lives in
// registers (float sreg[32] per thread, thread owns column j, rows i0..i0+31),
// exactly as k_delta_step (blocks.cu) already does for a single token -- only
// the cross-thread scratch (sq/sk/part/dj) stays in smem. Removing the 64KB
// dynamic S[] frees occupancy and drops the per-lane smem state traffic. The
// arithmetic is byte-for-byte the smem version's: fp32 in a register == fp32 in
// smem, same per-element expressions, same part[0..3] reduction order (gated by
// build/gdn_fuse_eq). sreg[k] stands in for S[(i0+k)*SK + j].
__global__ void k_gdn_delta_chunk3(const float* __restrict__ S0,
                                   __grid_constant__ const CP3 conv3,
                                   __grid_constant__ const CP3 g3,
                                   __grid_constant__ const CP3 beta3,
                                   __grid_constant__ const P3 o3, int nsp) {
    constexpr int SK = 128;
    __shared__ float sq[SK], sk[SK], part[4][SK], dj[SK];
    const int h = blockIdx.x;
    const int j = threadIdx.x & (SK - 1);
    const int it = threadIdx.x >> 7;
    const int i0 = it * 32;
#if Q27_GDN_HEAD_TILE
    const int qk = h % 16;
#else
    const int qk = h / 3;
#endif
    const float scale = rsqrtf((float)SK);
    const float* Sgh = S0 + (size_t)h * SK * SK;
    float sreg[32];
#pragma unroll
    for (int k = 0; k < 32; k++) sreg[k] = Sgh[(size_t)(i0 + k) * SK + j];
    for (int t = 0; t < nsp; t++) {
        const float* conv = conv3.p[t + 1];
        if (it == 0) {
            sq[j] = conv[qk * SK + j] * scale;
            sk[j] = conv[2048 + qk * SK + j];
        }
        __syncthreads();
        const float decay = expf(g3.p[t + 1][h]);
        float pred = 0.f;
#pragma unroll 8
        for (int k = 0; k < 32; k++) {
            float s = sreg[k] * decay;
            sreg[k] = s;
            pred += sk[i0 + k] * s;
        }
        part[it][j] = pred;
        __syncthreads();
        if (it == 0) {
            float p = part[0][j] + part[1][j] + part[2][j] + part[3][j];
            float vj = conv[4096 + h * SK + j];
            dj[j] = beta3.p[t + 1][h] * (vj - p);
        }
        __syncthreads();
        float d = dj[j];
        float acc = 0.f;
#pragma unroll 8
        for (int k = 0; k < 32; k++) {
            float s = sreg[k] + sk[i0 + k] * d;
            sreg[k] = s;
            acc += sq[i0 + k] * s;
        }
        part[it][j] = acc;
        __syncthreads();
        if (it == 0)
            o3.p[t + 1][h * SK + j] = part[0][j] + part[1][j] + part[2][j] + part[3][j];
        __syncthreads();
    }
}

void gdn_delta_chunk3(const float* S0, CP3 conv, CP3 g, CP3 beta, P3 o, int nsp,
                      cudaStream_t st) {
    // register-resident state: only the static sq/sk/part/dj smem now.
    k_gdn_delta_chunk3<<<48, 512, 0, st>>>(S0, conv, g, beta, o, nsp);
    CUDA_CHECK(cudaGetLastError());
}

// ---- 2026-08-18 GDN mix fusion (round-budget follow-on) -------------------
// nsys attribution: the decode mix ran 6 kernels x 48 layers x members per
// round, chain wall 38.8 us of which 13.0 us was inter-kernel gap, and the
// delta pair moved S three times (delta_step read+write, chunk3 re-read).
// These two kernels collapse the chain to 3 launches (conv_step + convnorm3 +
// delta_all) and move S twice (read once, write once).
//
// MIRROR WARNING (extends the M1 warning above): k_gdn_delta_all's lane-0
// phase is the FOURTH copy of the delta-step arithmetic (k_delta_step,
// k_delta_scan_T, k_gdn_delta_chunk3, here) -- same tile split, same
// part[0..3] reduction order, same decay/pred/d/update sequence. Any
// arithmetic change in one MUST land in all four, re-gated with
// build/gdn_fuse_eq (bitwise) + ninv's CHUNK+FOLD leg.

// k_gdn_delta_all: lane 0's committed delta step PLUS lanes 1..nsp's chunk
// outputs, one launch. S is staged raw into smem, lane 0 updates it in smem
// (writing the committed rows to global exactly as k_delta_step does), then
// the chunk loop runs from the smem copy -- which holds bit-for-bit the bytes
// chunk3 would have re-read from global, because fp32 stored to smem and fp32
// stored to global are the same value. Lane 0's arithmetic is k_delta_step's
// with S[i*SK+j] standing in for the sreg[] registers: same per-element
// expressions, same accumulation order (the pragma-unroll difference does not
// reassociate the single-accumulator chains).
// 2026-09-07 register-resident state (see k_gdn_delta_chunk3 note): sreg[32]
// per thread replaces the 64KB dynamic S[]. Lane 0 chains + writes committed
// rows to Soh; lanes 1..nsp chain in registers. sreg[k] == S[(i0+k)*SK + j].
__global__ void k_gdn_delta_all(const float* __restrict__ Ssrc, float* __restrict__ Sdst,
                                __grid_constant__ const CP3 conv3,
                                __grid_constant__ const CP3 g3,
                                __grid_constant__ const CP3 beta3,
                                __grid_constant__ const P3 o3, int nsp) {
    constexpr int SK = 128;
    __shared__ float sq[SK], sk[SK], part[4][32], dj[32];
    const int h = blockIdx.x;
    // 2026-09-08 column-tile split (the k_delta_scan_T pattern): a block owns
    // one head's 32-column slice over all 128 rows (4 row-tiles x 32 cols =
    // 128 threads), grid (48 heads, 4 tiles) = 192 CTAs instead of 48. Every
    // per-(row, column) expression and the part[0..3] sum order are unchanged
    // -> bitwise (gdn_fuse_eq + ninv CHUNK+FOLD). sq/sk are loaded in full by
    // the 128 threads (was: the 128 it==0 threads).
    const int jj = threadIdx.x & 31;
    const int j = blockIdx.y * 32 + jj;
    const int it = threadIdx.x >> 5;
    const int i0 = it * 32;
#if Q27_GDN_HEAD_TILE
    const int qk = h % 16;
#else
    const int qk = h / 3;
#endif
    const float scale = rsqrtf((float)SK);
    const float* Sgh = Ssrc + (size_t)h * SK * SK;
    float* Soh = Sdst + (size_t)h * SK * SK;
    float sreg[32];
#pragma unroll
    for (int k = 0; k < 32; k++) sreg[k] = Sgh[(size_t)(i0 + k) * SK + j];
    // ---- lane 0: k_delta_step, register-resident, committed rows written out ----
    {
        const float* conv = conv3.p[0];
        sq[threadIdx.x] = conv[qk * SK + threadIdx.x] * scale;
        sk[threadIdx.x] = conv[2048 + qk * SK + threadIdx.x];
        __syncthreads();
        const float decay = expf(g3.p[0][h]);
        float pred = 0.f;
#pragma unroll 8
        for (int k = 0; k < 32; k++) {
            float s = sreg[k] * decay;
            sreg[k] = s;
            pred += sk[i0 + k] * s;
        }
        part[it][jj] = pred;
        __syncthreads();
        if (it == 0) {
            float p = part[0][jj] + part[1][jj] + part[2][jj] + part[3][jj];
            float vj = conv[4096 + h * SK + j];
            dj[jj] = beta3.p[0][h] * (vj - p);
        }
        __syncthreads();
        float d = dj[jj];
        float acc = 0.f;
#pragma unroll 8
        for (int k = 0; k < 32; k++) {
            float s = sreg[k] + sk[i0 + k] * d;
            sreg[k] = s;
            Soh[(size_t)(i0 + k) * SK + j] = s;
            acc += sq[i0 + k] * s;
        }
        part[it][jj] = acc;
        __syncthreads();
        if (it == 0)
            o3.p[0][h * SK + j] = part[0][jj] + part[1][jj] + part[2][jj] + part[3][jj];
        __syncthreads();
    }
    // ---- lanes 1..nsp: k_gdn_delta_chunk3's loop, verbatim (register form) ----
    for (int t = 0; t < nsp; t++) {
        const float* conv = conv3.p[t + 1];
        sq[threadIdx.x] = conv[qk * SK + threadIdx.x] * scale;
        sk[threadIdx.x] = conv[2048 + qk * SK + threadIdx.x];
        __syncthreads();
        const float decay = expf(g3.p[t + 1][h]);
        float pred = 0.f;
#pragma unroll 8
        for (int k = 0; k < 32; k++) {
            float s = sreg[k] * decay;
            sreg[k] = s;
            pred += sk[i0 + k] * s;
        }
        part[it][jj] = pred;
        __syncthreads();
        if (it == 0) {
            float p = part[0][jj] + part[1][jj] + part[2][jj] + part[3][jj];
            float vj = conv[4096 + h * SK + j];
            dj[jj] = beta3.p[t + 1][h] * (vj - p);
        }
        __syncthreads();
        float d = dj[jj];
        float acc = 0.f;
#pragma unroll 8
        for (int k = 0; k < 32; k++) {
            float s = sreg[k] + sk[i0 + k] * d;
            sreg[k] = s;
            acc += sq[i0 + k] * s;
        }
        part[it][jj] = acc;
        __syncthreads();
        if (it == 0)
            o3.p[t + 1][h * SK + j] = part[0][jj] + part[1][jj] + part[2][jj] + part[3][jj];
        __syncthreads();
    }
}

void gdn_delta_all(const float* Ssrc, float* Sdst, CP3 conv, CP3 g, CP3 beta, P3 o, int nsp,
                   cudaStream_t st) {
    // register-resident state: only the static sq/sk/part/dj smem now.
    k_gdn_delta_all<<<dim3(48, 4), 128, 0, st>>>(Ssrc, Sdst, conv, g, beta, o, nsp);
    CUDA_CHECK(cudaGetLastError());
}

// k_gdn_convnorm3: lanes 1..vw-1's conv (k_gdn_conv_chunk3's tap arithmetic,
// verbatim) + ALL lanes' q||k l2norm (k_l2norm3's 128-thread tree, verbatim
// pairing order) + the record copies (k_gdn_record3), one launch. Lane 0's
// conv values come from k_conv_step (stream-ordered, unchanged). The norm
// blocks write the post-norm value once instead of chunk3's write + l2norm3's
// read-modify-write -- same final bytes.
//
// Grid, 128 threads/block:
//   blocks [0, 32*vw)          role A: lane = bx/32, head = bx%32, channels
//                              head*128 + tid (q||k, 0..4095). conv (lane>=1)
//                              -> norm (all lanes) -> record (lane>=1).
//   blocks [32*vw, +48*(vw-1)) role B: v channels 4096..10239 for lanes
//                              1..vw-1, conv + record only. First block of a
//                              lane's segment also records g/beta (tid < 48).
__global__ void k_gdn_convnorm3(const float* __restrict__ ring,
                                __grid_constant__ const CP3 qkv,
                                const float* __restrict__ w,
                                __grid_constant__ const P3 out,
                                __grid_constant__ const CP3 g3,
                                __grid_constant__ const CP3 beta3,
                                float* __restrict__ rq, float* __restrict__ rc,
                                float* __restrict__ rg, float* __restrict__ rb,
                                int channels, int n_heads, float eps, int vw) {
    constexpr int HD = 128; // norm head_dim == the l2norm3 call's GDN_DIM
    const int bx = blockIdx.x, tid = threadIdx.x;
    const int na = 32 * vw;
    auto conv_at = [&](int L, int c) {
        const float* wc = w + (size_t)c * 4;
        auto tap = [&](int a) {
            return a >= 1 ? qkv.p[a][c] : ring[(size_t)(a + 2) * channels + c];
        };
        float r0 = tap(L - 3), r1 = tap(L - 2), r2 = tap(L - 1), x = tap(L);
#if Q27_CONV_OLDEST_FIRST
        float acc = r0 * wc[0] + r1 * wc[1] + r2 * wc[2] + x * wc[3];
#else
        float acc = r0 * wc[3] + r1 * wc[2] + r2 * wc[1] + x * wc[0];
#endif
        return acc / (1.0f + expf(-acc)); // silu
    };
    if (bx < na) { // role A: q||k conv + norm + record
        const int lane = bx / 32, head = bx - lane * 32;
        const int c = head * HD + tid;
        float val = lane == 0 ? out.p[0][c] : conv_at(lane, c);
        __shared__ float sh[128];
        sh[tid] = val * val;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) { // k_l2norm3's tree, blockDim 128
            if (tid < s) sh[tid] += sh[tid + s];
            __syncthreads();
        }
        const float inv = rsqrtf(fmaxf(sh[0], eps * eps));
        const float nv = val * inv;
        out.p[lane][c] = nv;
        if (lane >= 1) {
            const int t = lane - 1;
            rc[(size_t)t * channels + c] = nv;
            rq[(size_t)t * channels + c] = qkv.p[lane][c];
        }
    } else { // role B: v-region conv + record, lanes 1..vw-1
        const int b2 = bx - na;
        const int vblk = (channels - 32 * HD) / 128; // 48
        const int lane = b2 / vblk + 1, cb = (b2 - (lane - 1) * vblk) * 128 + 32 * HD;
        const int c = cb + tid;
        const float val = conv_at(lane, c);
        out.p[lane][c] = val;
        const int t = lane - 1;
        rc[(size_t)t * channels + c] = val;
        rq[(size_t)t * channels + c] = qkv.p[lane][c];
        if (cb == 32 * HD && tid < n_heads) {
            rg[(size_t)t * n_heads + tid] = g3.p[lane][tid];
            rb[(size_t)t * n_heads + tid] = beta3.p[lane][tid];
        }
    }
}

void gdn_convnorm3(const float* ring, CP3 qkv, const float* convw, P3 out, CP3 g, CP3 beta,
                   float* rec_qkv, float* rec_conv, float* rec_g, float* rec_beta, int channels,
                   int n_heads, float eps, int vw, cudaStream_t st) {
    const int nblk = 32 * vw + ((channels - 4096) / 128) * (vw - 1);
    k_gdn_convnorm3<<<nblk, 128, 0, st>>>(ring, qkv, convw, out, g, beta, rec_qkv, rec_conv,
                                          rec_g, rec_beta, channels, n_heads, eps, vw);
    CUDA_CHECK(cudaGetLastError());
}

// Record lanes 1..nsp into the per-layer arena (row L-1): raw qkv + post-norm
// conv rows (contiguous [row][channels], so the Fold's conv_ring_update /
// delta_scan_seq consume them with prefill's row stride), g/beta scalars
// ([row][n_heads]). Pure copies -- launch after l2norm3, before the lane
// buffers are reused by the next layer.
__global__ void k_gdn_record3(__grid_constant__ const CP3 qkv,
                              __grid_constant__ const CP3 conv,
                              __grid_constant__ const CP3 g,
                              __grid_constant__ const CP3 beta,
                              float* __restrict__ rq, float* __restrict__ rc,
                              float* __restrict__ rg, float* __restrict__ rb,
                              int channels, int n_heads) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int t = blockIdx.y; // arena row t <- verify lane t+1
    rq[(size_t)t * channels + c] = qkv.p[t + 1][c];
    rc[(size_t)t * channels + c] = conv.p[t + 1][c];
    if (c < n_heads) {
        rg[(size_t)t * n_heads + c] = g.p[t + 1][c];
        rb[(size_t)t * n_heads + c] = beta.p[t + 1][c];
    }
}

void gdn_record3(CP3 qkv, CP3 conv, CP3 g, CP3 beta, float* rec_qkv, float* rec_conv,
                 float* rec_g, float* rec_beta, int channels, int n_heads, int nsp,
                 cudaStream_t st) {
    dim3 grid((channels + 255) / 256, nsp);
    k_gdn_record3<<<grid, 256, 0, st>>>(qkv, conv, g, beta, rec_qkv, rec_conv, rec_g,
                                        rec_beta, channels, n_heads);
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
// Two weights, one launch (2026-09-08): blockIdx.z picks (W, y); the body is
// k_gemv_f16_3's verbatim, so every output is bitwise the same as two
// launches. The GDN alpha/beta gate projections share the activation x and
// were 96 of the width-8 round's graph nodes at ~4 us each.
__global__ void k_gemv_f16_3x2(const __half* __restrict__ Wa, const __half* __restrict__ Wb,
                               __grid_constant__ const CP3 xp, __grid_constant__ const P3 ya,
                               __grid_constant__ const P3 yb, int64_t cols) {
    const __half* W = blockIdx.z ? Wb : Wa;
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
    if (threadIdx.x == 0) (blockIdx.z ? yb : ya).p[blockIdx.y][blockIdx.x] = sh[0];
}
void gemv_f16_3x2(const __half* Wa, const __half* Wb, CP3 x, P3 ya, P3 yb, int64_t rows,
                  int64_t cols, cudaStream_t st, int ntok) {
    dim3 g((unsigned)rows, ntok, 2);
    k_gemv_f16_3x2<<<g, 256, 0, st>>>(Wa, Wb, x, ya, yb, cols);
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
                            void* const* __restrict__ ktab, void* const* __restrict__ vtab,
                            __grid_constant__ const IP3 pos, int rowlen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowlen) return;
    int t = blockIdx.y;
    // M2a: row via the pair's block table (addressing-only; same bytes).
    const int p = *pos.p[t];
    kv_set(kv_row<CT>(ktab, p, rowlen)[i], kp.p[t][i]);
    kv_set(kv_row<CT>(vtab, p, rowlen)[i], vp.p[t][i]);
}
void kv_store3(CP3 k, CP3 v, void* const* ktab, void* const* vtab, IP3 pos, int rowlen,
               cudaStream_t st, int ntok, bool fp8) {
    dim3 g((rowlen + 255) / 256, ntok);
    if (fp8)
        k_kv_store3<__nv_fp8_e4m3><<<g, 256, 0, st>>>(k, v, ktab, vtab, pos, rowlen);
    else
        k_kv_store3<__half><<<g, 256, 0, st>>>(k, v, ktab, vtab, pos, rowlen);
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
// V is turbo3 in every turbo kind; only K varies, so the kernel takes the
// kv_kind itself rather than a widening pile of bools:
//   KV_T3   -- K turbo3 (3-bit blocks)
//   KV_T3V  -- K bypasses quantization entirely and stores plain fp16 rows,
//              the GQA=6 escape if turbo3-K craters (port-spec risk section)
//   KV_T5K  -- K turbo5 (5-bit blocks, src/turbo5.cuh)
// Assumes head_dim == 256 (two groups/head), like the whole fd2 family.
__global__ void k_kv_store_t3(__grid_constant__ const CP3 kp, __grid_constant__ const CP3 vp,
                              void* const* __restrict__ ktab, void* const* __restrict__ vtab,
                              __grid_constant__ const IP3 pos, int n_kv_heads, int head_dim,
                              int kvk) {
    const int t = blockIdx.z, j = threadIdx.x;
    const bool is_v = blockIdx.y == 1;
    const int h = blockIdx.x >> 1, g = blockIdx.x & 1;
    const int p = *pos.p[t];
    const float* src = (is_v ? vp.p[t] : kp.p[t]) + (size_t)h * head_dim + g * 128;
    // M2a: in-page block/row index via the pair's table (addressing-only).
    const int bpr = n_kv_heads * 2; // blocks per row (head_dim 256 = 2 groups/head)
    const size_t blk = (size_t)h * 2 + g;
    __shared__ float xs[128], red[128];
    if (!is_v && kvk == KV_T3V) {
        kv_row<__half>(ktab, p, n_kv_heads * head_dim)[(size_t)h * head_dim + g * 128 + j] =
            __float2half_rn(src[j]);
        return;
    }
    if (!is_v && kvk == KV_T5K) {
        q27turbo::turbo5_quant_group(src[j], kv_blk<q27turbo::block_turbo5>(ktab, p, bpr) + blk,
                                     j, xs, red);
        return;
    }
    if (!is_v && kvk == KV_I8G64) {
        q27turbo::i8g64_quant_group(src[j], kv_blk<q27turbo::block_i8g64>(ktab, p, bpr) + blk,
                                    j, red);
        return;
    }
    q27turbo::turbo3_quant_group(
        src[j], kv_blk<q27turbo::block_turbo3>(is_v ? vtab : ktab, p, bpr) + blk, j, xs, red);
}

void kv_store_t3(CP3 k, CP3 v, void* const* ktab, void* const* vtab, IP3 pos, int n_kv_heads,
                 int head_dim, cudaStream_t st, int ntok, int kvk) {
    dim3 g(n_kv_heads * (head_dim >> 7), 2, ntok); // x: (head, group); y: K,V
    k_kv_store_t3<<<g, 128, 0, st>>>(k, v, ktab, vtab, pos, n_kv_heads, head_dim, kvk);
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
__global__ void k_attn_fd(__grid_constant__ const CP3 qp, int q_stride,
                          const void* const* __restrict__ ktab,
                          const void* const* __restrict__ vtab, float* __restrict__ part,
                          IP3 pos, int n_kv_heads, int gqa, int head_dim, float scale) {
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
        // M2a: token-row via the pair's block table, then the head offset
        // (addressing-only; the streamed bytes and fp order are unchanged).
        const CT* kp = kv_row_c<CT>(ktab, p, n_kv_heads * head_dim) + (size_t)kvh * head_dim;
        const CT* vp = kv_row_c<CT>(vtab, p, n_kv_heads * head_dim) + (size_t)kvh * head_dim;
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
                           const void* const* __restrict__ ktab,
                           const void* const* __restrict__ vtab, float* __restrict__ part,
                           IP3 pos, int n_kv_heads, int gqa, int head_dim, float scale) {
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

    // M2a: page-hoisted streaming -- each warp walks its p-sequence unchanged
    // (same positions, same fp order = bitwise), but the table load happens
    // once per 64-row page instead of per iteration (the naive per-iteration
    // form measured +5-6% on the fd2 wall at 61K; fd2 is DRAM-saturated and
    // the dependent load sat in the chain).
    for (int p = p_lo + warp; p < p_hi;) {
        const int pg = p >> KV_PAGE_SHIFT;
        const int pend = min(p_hi, (pg + 1) << KV_PAGE_SHIFT);
        // linear induction within the page (the original loop's strength
        // reduction, which the naive (p & 63) recompute was defeating)
        const size_t rowe = (size_t)n_kv_heads * head_dim;
        const CT* kp = (const CT*)ktab[pg] + (size_t)(p & KV_PAGE_MASK) * rowe +
                       (size_t)kvh * head_dim;
        const CT* vp = (const CT*)vtab[pg] + (size_t)(p & KV_PAGE_MASK) * rowe +
                       (size_t)kvh * head_dim;
        for (; p < pend; p += NW, kp += NW * rowe, vp += NW * rowe) {
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

// k_attn_fd2 for the turbo (block-cache) family. V is always turbo3; KVK
// selects the K leg -- KV_T3 turbo3 blocks, KV_T3V plain fp16 rows, KV_T5K
// turbo5 5-bit blocks. Row addressing is in blocks (pos*n_kv_heads*2 +
// kvh*2), so this is a separate kernel rather than a CT overload (port spec
// item 3); the softmax / online-max / serialized warp merge / partial layout
// are copied byte-identical from k_attn_fd2 -- only the loads differ. The
// fp8/fp16 k_attn_fd2 instantiations are textually untouched (bitwise gate),
// and widening the old `bool KT3` to KVK leaves the turbo3/turbo3v legs
// compiling to the same branch they always took.
// Q arrives already WHT-rotated whenever K is (engine rotates after rope);
// VKQ accumulates in the rotated V basis -- the engine's single post-combine
// inverse-WHT un-rotates the pooled output (linearity).
template <int KVK, int NW>
__global__ void k_attn_fd2_t3(__grid_constant__ const CP3 qp, int q_stride,
                              const void* const* __restrict__ ktab,
                              const void* const* __restrict__ vtab,
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

    // M2a: page-hoisted streaming (see the scalar fd2 leg's rationale).
    for (int p = p_lo + warp; p < p_hi;) {
        const int pg = p >> KV_PAGE_SHIFT;
        const void* kpage = ktab[pg];
        const void* vpage = vtab[pg];
        const int pend = min(p_hi, (pg + 1) << KV_PAGE_SHIFT);
        const int bpr = n_kv_heads * 2; // blocks per token-row
        // linear induction within the page (block index and fp16-row offset)
        size_t ib = (size_t)(p & KV_PAGE_MASK) * bpr + (size_t)kvh * 2;
        size_t rowoff = ((size_t)(p & KV_PAGE_MASK) * n_kv_heads + kvh) * (size_t)head_dim;
        for (; p < pend;
             p += NW, ib += (size_t)NW * bpr, rowoff += (size_t)NW * n_kv_heads * head_dim) {
        float kv[8], vv[8];
        if constexpr (KVK == KV_T3)
            fd2_ld8_t3((const q27turbo::block_turbo3*)kpage + ib, lane, kv);
        else if constexpr (KVK == KV_T5K)
            q27turbo::turbo5_ld8_lane((const q27turbo::block_turbo5*)kpage + ib, lane, kv);
        else if constexpr (KVK == KV_I8G64)
            q27turbo::i8g64_ld8_lane((const q27turbo::block_i8g64*)kpage + ib, lane, kv);
        else
            fd2_ld8((const __half*)kpage + rowoff, lane, kv);
        fd2_ld8_t3((const q27turbo::block_turbo3*)vpage + ib, lane, vv);
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
static void fd_launch(CP3 q, int q_stride, const void* const* ktab, const void* const* vtab,
                      float* scratch, IP3 pos, int n_kv_heads, int gqa, int head_dim,
                      float scale, size_t sm, int ntok, cudaStream_t st) {
    fd_setattr<CT>(sm);
    dim3 g1(n_kv_heads, FD_NS, ntok);
    k_attn_fd<CT><<<g1, 256, sm, st>>>(q, q_stride, ktab, vtab, scratch, pos, n_kv_heads, gqa,
                                       head_dim, scale);
}

void attn_decode3_fd2(CP3 q, int q_stride, const void* const* ktab, const void* const* vtab,
                      P3 out, float* scratch, IP3 pos, int max_ctx, int n_q_heads,
                      int n_kv_heads, int head_dim, float scale, cudaStream_t st, int ntok,
                      int kvk) {
    (void)max_ctx;
    int gqa = n_q_heads / n_kv_heads;
    // NW=4 (128 threads): probe-favored -- more blocks/SM for latency hiding.
    // smem 12.3KB, under the 48KB default: no cudaFuncSetAttribute needed.
    constexpr int NW2 = 4;
    size_t sm = (size_t)(2 * 6) * 256 * sizeof(float);
    dim3 g1(ntok, FD2_NS, n_kv_heads);  // P14: lane (x) fastest -> cross-lane KV L2 reuse
    if (kvk == KV_T3)
        k_attn_fd2_t3<KV_T3, NW2><<<g1, NW2 * 32, sm, st>>>(q, q_stride, ktab, vtab, scratch, pos,
                                                            n_kv_heads, gqa, head_dim, scale);
    else if (kvk == KV_T3V)
        k_attn_fd2_t3<KV_T3V, NW2><<<g1, NW2 * 32, sm, st>>>(q, q_stride, ktab, vtab, scratch, pos,
                                                             n_kv_heads, gqa, head_dim, scale);
    else if (kvk == KV_T5K)
        k_attn_fd2_t3<KV_T5K, NW2><<<g1, NW2 * 32, sm, st>>>(q, q_stride, ktab, vtab, scratch, pos,
                                                             n_kv_heads, gqa, head_dim, scale);
    else if (kvk == KV_I8G64)
        k_attn_fd2_t3<KV_I8G64, NW2><<<g1, NW2 * 32, sm, st>>>(q, q_stride, ktab, vtab,
                                                               scratch, pos, n_kv_heads, gqa,
                                                               head_dim, scale);
    else if (kvk == KV_FP8)
        k_attn_fd2<__nv_fp8_e4m3, NW2><<<g1, NW2 * 32, sm, st>>>(
            q, q_stride, ktab, vtab, scratch, pos, n_kv_heads, gqa, head_dim, scale);
    else
        k_attn_fd2<__half, NW2><<<g1, NW2 * 32, sm, st>>>(q, q_stride, ktab, vtab, scratch,
                                                          pos, n_kv_heads, gqa, head_dim,
                                                          scale);
    dim3 g2(n_q_heads, ntok);
    k_attn_fd_combine<<<g2, 256, 0, st>>>(scratch, out, n_q_heads, head_dim, FD2_NS, pos);
    CUDA_CHECK(cudaGetLastError());
}

void attn_decode3(CP3 q, int q_stride, const void* const* ktab, const void* const* vtab,
                  P3 out, float* scratch, IP3 pos, int max_ctx, int n_q_heads, int n_kv_heads,
                  int head_dim, float scale, cudaStream_t st, int ntok, int kvk) {
    // turbo3v (diagnostic) has exactly one read path (fd2). turbo3 and
    // turbo5k both have fdmma + H16 legs; Q27_FD=v1 still falls to fd2 for
    // every turbo kind.
    //
    // THE RETURN IS THE GUARD, not a shortcut. Both branches below select a K
    // format from a closed list -- the mma leg engages on an explicit kind
    // list, its H16 sibling maps anything unrecognized to fmt 0 (fp16 rows),
    // and the v1 tail templates on __half/__nv_fp8_e4m3. A kind that reached
    // any of them without its own leg would be read at the WRONG format with
    // no error, so every such kind must return here. turbo5k LEFT this list on
    // 08-01 (c) when it gained both mma legs -- which is the maintenance rule:
    // add the leg, then take the kind out, rather than widening a condition
    // downstream and leaving the guard stale.
    if (kvk == KV_T3V || kvk == KV_I8G64) {
        attn_decode3_fd2(q, q_stride, ktab, vtab, out, scratch, pos, max_ctx, n_q_heads,
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
    if (fd && strcmp(fd, "mma") == 0 && kvk != KV_T3V && ntok >= 4 && ntok <= W_PLUMB) {
        static int arch = -1;
        if (arch < 0) {
            int dev, mj, mn;
            CUDA_CHECK(cudaGetDevice(&dev));
            CUDA_CHECK(cudaDeviceGetAttribute(&mj, cudaDevAttrComputeCapabilityMajor, dev));
            CUDA_CHECK(cudaDeviceGetAttribute(&mn, cudaDevAttrComputeCapabilityMinor, dev));
            arch = mj * 10 + mn;
        }
        // turbo5k is NOT in this list, deliberately -- a turbo5 e4m3 leg was
        // built, measured and DELETED on 2026-08-01 (d). The e4m3 tiles carry
        // 3 mantissa bits (~3.1% max relative error), which against each
        // format's OWN quantization error is +17.6% for turbo3 (noise on an
        // 18.6%-RMS format) but +81% for turbo5, whose own error is 5.0% --
        // inflating it 1.29x in quadrature and throwing away roughly a third
        // of what the 5th bit was bought for. Trading precision for speed is
        // exactly backwards in a format that exists to fix a TAIL. turbo5k
        // falls through to the H16 branch below, which stages fp16 (11
        // mantissa bits, no meaningful loss) and runs on sm_80+ -- so it gets
        // an mma leg on BOTH arches without the precision cost. Above H16's
        // ntok<=8 smem cap it lands on fd2, which is correct, not a gap.
        if (arch >= 89 && (fp8 || kvk == KV_T3)) {
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
            //
            // W16 CORRECTION: stages=1 is only better BECAUSE two CTAs fit.
            // That stops being true at W>=14. s_q holds fdmma_qrows(W) = 16-
            // rounded 6W rows: 80 rows through W=13, but 96 from W=14 (6*14=84),
            // which pushes smem(W,1) from 48.0KB to 52.1KB. Two CTAs then need
            // 104.2KB > the 100KB sharedMemPerMultiprocessor on sm_120/sm_86, so
            // occupancy drops to ONE CTA -- and a single-CTA kernel is strictly
            // better double-buffered. So: keep stages=1 while 2 CTAs actually
            // co-reside, else fall back to the 2-stage kernel. Both variants are
            // bitwise-equal (shared fdmma_tile_compute), so this is a pure
            // scheduling choice; Q27_FDMMA_STAGES pins it either way for A/B.
            static const int stages_pin = [] {
                const char* e = getenv("Q27_FDMMA_STAGES");
                return e ? atoi(e) : 0; // 0 = auto
            }();
            static const int smem_per_sm = [] {
                int dev3, v;
                CUDA_CHECK(cudaGetDevice(&dev3));
                CUDA_CHECK(cudaDeviceGetAttribute(
                    &v, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev3));
                return v;
            }();
            const bool two_cta =
                (size_t)2 * fdmma::fdmma_smem_bytes(ntok, 1) <= (size_t)smem_per_sm;
            const int fdmma_stages =
                (stages_pin == 1 || stages_pin == 2) ? stages_pin : (two_cta ? 1 : 2);
            // split-count retune (tuning 2026-07-10): fdmma's grid is
            // (ns, kv_heads) with 2 CTAs/SM resident -- ns = SMs*2/kv_heads
            // fills EXACTLY one wave (85 on the 5090; 128 left a half-empty
            // second wave, +29-40% at 61K). fdmma-only: fd2 keeps FD2_NS
            // (its splits were swept for its own shape). Capped by
            // FD_MAXNS (scratch rows). NOT bitwise across ns values (split
            // boundaries move -> combine fp order) -- rebuild-class
            // tie-lottery, the regime the mma basin matrix cleared;
            // Q27_FDMMA_NS pins it for A/B.
            // W16: the "2" was the resident-CTA count, baked in when every live
            // width ran 2 CTAs/SM. It is now `two_cta` -- at W>=14 one CTA is
            // resident, so one wave is smc (not 2*smc) CTAs and ns must halve
            // or the grid overflows into a half-empty second wave (the exact
            // failure the 128->85 retune fixed). W<=13 still computes 85 on the
            // 5090, bit-for-bit the shipped value.
            // The device query and the env read stay in one-shot statics (this
            // runs per attention layer per round -- a getenv on that path would
            // be a real cost). Only the width-dependent arithmetic is per-call.
            static const int ns_pin = [] {
                const char* e = getenv("Q27_FDMMA_NS");
                if (!e) return 0;
                int v = atoi(e);
                return (v >= 1 && v <= FD_MAXNS) ? v : 0; // 0 = auto
            }();
            static const int smc = [] {
                int dev2, v;
                CUDA_CHECK(cudaGetDevice(&dev2));
                CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev2));
                return v;
            }();
            int fdmma_ns = (smc * (two_cta ? 2 : 1)) / n_kv_heads;
            fdmma_ns = fdmma_ns < 16 ? 16 : fdmma_ns > FD_MAXNS ? FD_MAXNS : fdmma_ns;
            if (ns_pin) fdmma_ns = ns_pin;
            if (fdmma::launch_fdmma(mq, q_stride, ktab, vtab, scratch, mp, n_kv_heads,
                                    n_q_heads / n_kv_heads, head_dim, scale, fdmma_ns, ntok, st,
                                    fdmma_stages, /*t3=*/kvk == KV_T3)) {
                dim3 g2(n_q_heads, ntok);
                k_attn_fd_combine<<<g2, 256, 0, st>>>(scratch, out, n_q_heads, head_dim,
                                                      fdmma_ns, pos);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
        } else if (arch >= 80 && ntok <= 8) {
            // H16 (fp16-MMA) verify: sm_80..88, all KV formats (fp16 gets
            // its first mma leg); W caps at 8 (smem). Tolerance-class like
            // e4m3 fdmma -- engages only under Q27_FD=mma.
            // One 1-CTA wave: ns = SMs/kv_heads (H16 is 1 CTA/SM).
            static const int h16_ns = [n_kv_heads] {
                if (const char* e = getenv("Q27_FDMMA_NS")) {
                    int v = atoi(e);
                    if (v >= 1 && v <= FD_MAXNS) return v;
                }
                int dev2, smc;
                CUDA_CHECK(cudaGetDevice(&dev2));
                CUDA_CHECK(cudaDeviceGetAttribute(&smc, cudaDevAttrMultiProcessorCount, dev2));
                int v = smc / n_kv_heads;
                return v < 8 ? 8 : v > FD_MAXNS ? FD_MAXNS : v;
            }();
            fdmma::FCP3 mq;
            fdmma::FIP3 mp;
            for (int t = 0; t < 16; t++) { mq.p[t] = q.p[t]; mp.p[t] = pos.p[t]; }
            const int fmt = kvk == KV_T3    ? 2
                            : kvk == KV_T5K ? 3
                            : kvk == KV_FP8 ? 1
                                            : 0;
            if (fdmma::launch_fdmma_h16(mq, q_stride, ktab, vtab, scratch, mp, n_kv_heads,
                                        n_q_heads / n_kv_heads, head_dim, scale, h16_ns,
                                        ntok, fmt, st)) {
                dim3 g2(n_q_heads, ntok);
                k_attn_fd_combine<<<g2, 256, 0, st>>>(scratch, out, n_q_heads, head_dim,
                                                      h16_ns, pos);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
        }
    }
    if (!fd || strcmp(fd, "v1") != 0 || kvk >= KV_T3) {
        // no turbo kind ever runs the v1 kernel (no block leg there)
        attn_decode3_fd2(q, q_stride, ktab, vtab, out, scratch, pos, max_ctx, n_q_heads,
                         n_kv_heads, head_dim, scale, st, ntok, kvk);
        return;
    }
    (void)max_ctx;
    int gqa = n_q_heads / n_kv_heads;
    size_t sm = (size_t)(6 + 8 * 6) * 256 * sizeof(float);
    if (fp8)
        fd_launch<__nv_fp8_e4m3>(q, q_stride, ktab, vtab, scratch, pos, n_kv_heads, gqa,
                                 head_dim, scale, sm, ntok, st);
    else
        fd_launch<__half>(q, q_stride, ktab, vtab, scratch, pos, n_kv_heads, gqa, head_dim,
                          scale, sm, ntok, st);
    dim3 g2(n_q_heads, ntok);
    k_attn_fd_combine<<<g2, 256, 0, st>>>(scratch, out, n_q_heads, head_dim, FD_NS, pos);
    CUDA_CHECK(cudaGetLastError());
}

// single-token plain-path attention through the same flash-decode kernels
void attn_decode(const float* q, int q_stride, const void* const* ktab,
                 const void* const* vtab, float* out, float* scratch, const int* d_pos,
                 int max_ctx, int n_q_heads, int n_kv_heads, int head_dim, float scale,
                 cudaStream_t st, int kvk) {
    CP3 qp{{q, q, q}};
    IP3 pp{{d_pos, d_pos, d_pos}};
    attn_decode3(qp, q_stride, ktab, vtab, out2p(out), scratch, pp, max_ctx, n_q_heads,
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
// lane k's argmax equals draft k. All W_PLUMB-1 draft / W_PLUMB verdict slots are
// dereferenced unconditionally (engine allocates every lane; the old
// kernel read all 7/8 the same way) -- only slots < max_draft / n matter.
__global__ void k_finish_round(int* __restrict__ dP, int* __restrict__ dtok,
                               __grid_constant__ const IP3 drafts,
                               __grid_constant__ const IP3 verdicts,
                               __grid_constant__ const CP3 x1s, float* __restrict__ h_next,
                               int* __restrict__ outcome, int n_embd,
                               const int* __restrict__ cap, int max_draft) {
    // W16: these were 11/12 literals. They are the lane plumbing, so they now
    // derive from W_PLUMB (cuda_common.h) -- a width change is one constant,
    // not a grep. NDRAFT = W_PLUMB-1 drafts feeding W_PLUMB verify columns.
    constexpr int NDRAFT = W_PLUMB - 1;
    int dr[NDRAFT], v[W_PLUMB];
#pragma unroll
    for (int k = 0; k < NDRAFT; k++) dr[k] = *drafts.p[k];
#pragma unroll
    for (int t = 0; t < W_PLUMB; t++) v[t] = *verdicts.p[t];
    // P12/P12b: max_draft gates depth to the verified columns (narrow-verify
    // graph). P7 (*cap): in-grammar rounds accept only the pending token;
    // drafts are unconstrained and must not commit past the constrained lane.
    int n = 1;
    if (!*cap) {
#pragma unroll
        for (int k = 1; k <= NDRAFT; k++)
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
        // outcome layout (OUTCOME_INTS = W_PLUMB+2): [0]=n, [1]=t1(prep),
        // [2 .. W_PLUMB]=dr1..dr(W_PLUMB-1) (the emitted tokens live in
        // [1..n]), [W_PLUMB+1]=new pending.
#pragma unroll
        for (int k = 0; k < NDRAFT; k++) outcome[2 + k] = dr[k];
        outcome[OUTCOME_INTS - 1] = nt; // new pending (P7: host grammar needs it pre-round)
    }
}
void finish_round(int* d_P, int* d_token, IP3 drafts, IP3 verdicts, CP3 x1s, float* h_next,
                  int* outcome, int n_embd, const int* cap, int max_draft, cudaStream_t st) {
    k_finish_round<<<4, 256, 0, st>>>(d_P, d_token, drafts, verdicts, x1s, h_next, outcome,
                                      n_embd, cap, max_draft);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace q27k
