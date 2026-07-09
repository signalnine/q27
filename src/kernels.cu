#include <cstdio>
#include <cstdlib>
#include "cuda_common.h"
#include "kernels.cuh"

namespace q27k {

// ---------------- dequant ----------------

__global__ void k_dequant_q4(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                             float* __restrict__ out, int64_t rows, int64_t cols) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * cols;
    if (idx >= n) return;
    int64_t r = idx / cols, c = idx % cols;
    uint8_t b = W[r * (cols / 2) + c / 2];
    int nib = (c & 1) ? (b >> 4) : (b & 0xF);
    float s = __half2float(S[r * (cols / 64) + c / 64]);
    out[idx] = (nib - 8) * s;
}

__global__ void k_dequant_q8(const int8_t* __restrict__ W, const __half* __restrict__ S,
                             float* __restrict__ out, int64_t rows, int64_t cols) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * cols;
    if (idx >= n) return;
    int64_t r = idx / cols, c = idx % cols;
    float s = __half2float(S[r * (cols / 128) + c / 128]);
    out[idx] = (float)W[r * cols + c] * s;
}

void dequant_q4(const uint8_t* W, const __half* S, float* out, int64_t rows, int64_t cols,
                cudaStream_t st) {
    int64_t n = rows * cols;
    k_dequant_q4<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(W, S, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}
void dequant_q8(const int8_t* W, const __half* S, float* out, int64_t rows, int64_t cols,
                cudaStream_t st) {
    int64_t n = rows * cols;
    k_dequant_q8<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(W, S, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- GEMV (reference) ----------------
// One block per output row, 256 threads grid-stride the reduction axis.

template <int BLOCK>
__device__ __forceinline__ float block_reduce(float v) {
    __shared__ float sh[BLOCK];
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    return sh[0];
}

__device__ __forceinline__ float warp_reduce(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// ---------------- activation quantization ----------------
// One warp per 32-element block: reduce max|x|, quantize, emit both byte orders.

__global__ void k_quantize_x(const float* __restrict__ x, int8_t* __restrict__ nat,
                             uint2* __restrict__ eo, float* __restrict__ scale,
                             int* __restrict__ isum, int nblocks) {
    int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (b >= nblocks) return;
    int lane = threadIdx.x & 31;
    float v = x[b * 32 + lane];
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    float s = amax / 127.f;
    float inv = s > 0.f ? 1.f / s : 0.f;
    int q = __float2int_rn(v * inv);
    q = max(-127, min(127, q));
    nat[b * 32 + lane] = (int8_t)q;
    int bsum = q;
    for (int off = 16; off > 0; off >>= 1) bsum += __shfl_xor_sync(0xffffffff, bsum, off);
    if (lane == 0) { scale[b] = s; isum[b] = bsum; }
    // even/odd packing: group u (0..3) covers elements u*8..u*8+7 of this block.
    // ALL lanes must execute the shuffles (divergent shfl_sync is UB); lanes >=4
    // compute redundant values and discard.
    int base = (lane & 3) * 8;
    uint32_t e = 0, o = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        int qe = __shfl_sync(0xffffffff, q, base + 2 * k);
        int qo = __shfl_sync(0xffffffff, q, base + 2 * k + 1);
        e |= (uint32_t)(uint8_t)(int8_t)qe << (8 * k);
        o |= (uint32_t)(uint8_t)(int8_t)qo << (8 * k);
    }
    if (lane < 4) eo[b * 4 + lane] = make_uint2(e, o);
}

// Group-64 variant: warp per 64-element group, lane owns elements g*64+lane
// and g*64+32+lane (both loads and stores coalesced). Emits nat64 + s64 only
// (no eo/isum -- the MMA path needs neither).
__global__ void k_quantize_x_g64(const float* __restrict__ x, int8_t* __restrict__ nat64,
                                 float* __restrict__ s64, int ngroups) {
    int g = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (g >= ngroups) return;
    int lane = threadIdx.x & 31;
    float v0 = x[g * 64 + lane];
    float v1 = x[g * 64 + 32 + lane];
    float amax = fmaxf(fabsf(v0), fabsf(v1));
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    float s = amax / 127.f;
    float inv = s > 0.f ? 1.f / s : 0.f;
    int q0 = max(-127, min(127, __float2int_rn(v0 * inv)));
    int q1 = max(-127, min(127, __float2int_rn(v1 * inv)));
    nat64[g * 64 + lane] = (int8_t)q0;
    nat64[g * 64 + 32 + lane] = (int8_t)q1;
    if (lane == 0) s64[g] = s;
}

XQuant xquant_alloc(int64_t max_cols, bool g64) {
    XQuant xq;
    CUDA_CHECK(cudaMalloc((void**)&xq.nat, max_cols));
    CUDA_CHECK(cudaMalloc((void**)&xq.eo, max_cols / 8 * sizeof(uint2)));
    CUDA_CHECK(cudaMalloc((void**)&xq.scale, max_cols / 32 * 4));
    CUDA_CHECK(cudaMalloc((void**)&xq.isum, max_cols / 32 * 4));
    if (g64) {
        CUDA_CHECK(cudaMalloc((void**)&xq.nat64, max_cols));
        CUDA_CHECK(cudaMalloc((void**)&xq.s64, max_cols / 64 * 4));
    }
    return xq;
}

void quantize_x_g64(const float* x, int64_t cols, const XQuant& xq, cudaStream_t st) {
    int ngroups = (int)(cols / 64);
    int warps = 8;
    k_quantize_x_g64<<<(ngroups + warps - 1) / warps, warps * 32, 0, st>>>(x, xq.nat64, xq.s64,
                                                                           ngroups);
    CUDA_CHECK(cudaGetLastError());
}

void quantize_x(const float* x, int64_t cols, const XQuant& xq, cudaStream_t st) {
    int nblocks = (int)(cols / 32);
    int warps = 8;
    k_quantize_x<<<(nblocks + warps - 1) / warps, warps * 32, 0, st>>>(x, xq.nat, xq.eo, xq.scale,
                                                                       xq.isum, nblocks);
    CUDA_CHECK(cudaGetLastError());
}

// Warp per row, dp4a against int8-quantized activations (mmvq-style).
// A 16-byte weight chunk covers 32 weights = one x-block: one w-scale, one x-scale.
__global__ void k_gemv_q4(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                          const int8_t* __restrict__ xnat, const uint2* __restrict__ xeo,
                          const float* __restrict__ xs, const int* __restrict__ xisum,
                          float* __restrict__ y, int64_t rows, int64_t cols) {
    (void)xnat;
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint4* wr = (const uint4*)(W + row * (cols / 2));
    const __half* sr = S + row * (cols / 64);
    const int n_chunks = (int)(cols / 32);

    float acc = 0.f;
    for (int ch = lane; ch < n_chunks; ch += 32) {
        uint4 w = __ldg(wr + ch);
        float s = __half2float(__ldg(sr + (ch >> 1))) * __ldg(xs + ch);
        // Task 2 (verify-gemv): 2x uint4 activation reads, same bytes/order as
        // the old 4x uint2 (bitwise; see k_gemv_q4_n for the Phase-0 rationale).
        const uint4* xp = (const uint4*)(xeo + (size_t)ch * 4);
        const uint4 xv0 = __ldg(xp), xv1 = __ldg(xp + 1);
        const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
        int di = 0;
        di = __dp4a((int)(ws[0] & 0x0F0F0F0Fu), (int)xv0.x, di);
        di = __dp4a((int)((ws[0] >> 4) & 0x0F0F0F0Fu), (int)xv0.y, di);
        di = __dp4a((int)(ws[1] & 0x0F0F0F0Fu), (int)xv0.z, di);
        di = __dp4a((int)((ws[1] >> 4) & 0x0F0F0F0Fu), (int)xv0.w, di);
        di = __dp4a((int)(ws[2] & 0x0F0F0F0Fu), (int)xv1.x, di);
        di = __dp4a((int)((ws[2] >> 4) & 0x0F0F0F0Fu), (int)xv1.y, di);
        di = __dp4a((int)(ws[3] & 0x0F0F0F0Fu), (int)xv1.z, di);
        di = __dp4a((int)((ws[3] >> 4) & 0x0F0F0F0Fu), (int)xv1.w, di);
        acc += s * (float)(di - 8 * __ldg(xisum + ch));
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// Warp per row; chunk = 32 int8 weights (two uint4 loads) = one x-block.
__global__ void k_gemv_q8(const int8_t* __restrict__ W, const __half* __restrict__ S,
                          const int8_t* __restrict__ xnat, const float* __restrict__ xs,
                          float* __restrict__ y, int64_t rows, int64_t cols) {
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint4* wr = (const uint4*)(W + row * cols);
    const uint4* xr = (const uint4*)xnat;
    const __half* sr = S + row * (cols / 128);
    const int n_chunks = (int)(cols / 32);

    float acc = 0.f;
    for (int ch = lane; ch < n_chunks; ch += 32) {
        uint4 w0 = __ldg(wr + 2 * ch), w1 = __ldg(wr + 2 * ch + 1);
        uint4 x0 = __ldg(xr + 2 * ch), x1 = __ldg(xr + 2 * ch + 1);
        float s = __half2float(__ldg(sr + (ch >> 2))) * __ldg(xs + ch);
        int di = 0;
        di = __dp4a((int)w0.x, (int)x0.x, di);
        di = __dp4a((int)w0.y, (int)x0.y, di);
        di = __dp4a((int)w0.z, (int)x0.z, di);
        di = __dp4a((int)w0.w, (int)x0.w, di);
        di = __dp4a((int)w1.x, (int)x1.x, di);
        di = __dp4a((int)w1.y, (int)x1.y, di);
        di = __dp4a((int)w1.z, (int)x1.z, di);
        di = __dp4a((int)w1.w, (int)x1.w, di);
        acc += s * (float)di;
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ---------------- batched GEMV (speculative verify) ----------------
// Same warp-per-row walk as the single-column kernels, but each weight chunk is
// dp4a'd against N activation columns. Weight bytes amortize N ways.

// P10-A0: lane args as by-value arrays so one template covers 2..10 lanes
// (fused 2-slot verify = 10). Same math order as the old 5-wide param lists.
struct Q4Lanes {
    const uint2* eo[10];
    const float* xs[10];
    const int* is[10];
    float* y[10];
};
struct Q8Lanes {
    const int8_t* nat[10];
    const float* xs[10];
    float* y[10];
};

template <int N>
__global__ void k_gemv_q4_n(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                            const Q4Lanes L, int64_t rows, int64_t cols) {
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint4* wr = (const uint4*)(W + row * (cols / 2));
    const __half* sr = S + row * (cols / 64);
    const int n_chunks = (int)(cols / 32);

    const uint2* const* eos = L.eo;
    const float* const* xss = L.xs;
    const int* const* iss = L.is;
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; n++) acc[n] = 0.f;

    for (int ch = lane; ch < n_chunks; ch += 32) {
        uint4 w = __ldg(wr + ch);
        const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
        float wsc = __half2float(__ldg(sr + (ch >> 1)));
#pragma unroll
        for (int n = 0; n < N; n++) {
            // Task 2 (verify-gemv): activation reads as 2x uint4 instead of
            // 4x uint2 -- same 32 bytes, same component order into the same
            // dp4a sequence (integer-exact, fp acc order untouched, bitwise),
            // but half the L1TEX wavefronts. Phase 0 measured these 8B loads
            // at 32B lane stride as THE stall (long_scoreboard 90%, 10/32
            // bytes/sector); 16B loads double the per-instruction utilization.
            const uint4* xp = (const uint4*)(eos[n] + (size_t)ch * 4);
            const uint4 xv0 = __ldg(xp), xv1 = __ldg(xp + 1);
            int di = 0;
            di = __dp4a((int)(ws[0] & 0x0F0F0F0Fu), (int)xv0.x, di);
            di = __dp4a((int)((ws[0] >> 4) & 0x0F0F0F0Fu), (int)xv0.y, di);
            di = __dp4a((int)(ws[1] & 0x0F0F0F0Fu), (int)xv0.z, di);
            di = __dp4a((int)((ws[1] >> 4) & 0x0F0F0F0Fu), (int)xv0.w, di);
            di = __dp4a((int)(ws[2] & 0x0F0F0F0Fu), (int)xv1.x, di);
            di = __dp4a((int)((ws[2] >> 4) & 0x0F0F0F0Fu), (int)xv1.y, di);
            di = __dp4a((int)(ws[3] & 0x0F0F0F0Fu), (int)xv1.z, di);
            di = __dp4a((int)((ws[3] >> 4) & 0x0F0F0F0Fu), (int)xv1.w, di);
            acc[n] += wsc * __ldg(xss[n] + ch) * (float)(di - 8 * __ldg(iss[n] + ch));
        }
    }
    float* const* yy = L.y;
#pragma unroll
    for (int n = 0; n < N; n++) {
        float v = warp_reduce(acc[n]);
        if (lane == 0) yy[n][row] = v;
    }
}

template <int N>
__global__ void k_gemv_q8_n(const int8_t* __restrict__ W, const __half* __restrict__ S,
                            const Q8Lanes L, int64_t rows, int64_t cols) {
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint4* wr = (const uint4*)(W + row * cols);
    const __half* sr = S + row * (cols / 128);
    const int n_chunks = (int)(cols / 32);

    const int8_t* const* nats = L.nat;
    const float* const* xss = L.xs;
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; n++) acc[n] = 0.f;

    for (int ch = lane; ch < n_chunks; ch += 32) {
        uint4 w0 = __ldg(wr + 2 * ch), w1 = __ldg(wr + 2 * ch + 1);
        float wsc = __half2float(__ldg(sr + (ch >> 2)));
#pragma unroll
        for (int n = 0; n < N; n++) {
            const uint4* xr = (const uint4*)nats[n];
            uint4 x0 = __ldg(xr + 2 * ch), x1 = __ldg(xr + 2 * ch + 1);
            int di = 0;
            di = __dp4a((int)w0.x, (int)x0.x, di);
            di = __dp4a((int)w0.y, (int)x0.y, di);
            di = __dp4a((int)w0.z, (int)x0.z, di);
            di = __dp4a((int)w0.w, (int)x0.w, di);
            di = __dp4a((int)w1.x, (int)x1.x, di);
            di = __dp4a((int)w1.y, (int)x1.y, di);
            di = __dp4a((int)w1.z, (int)x1.z, di);
            di = __dp4a((int)w1.w, (int)x1.w, di);
            acc[n] += wsc * __ldg(xss[n] + ch) * (float)di;
        }
    }
    float* const* yy = L.y;
#pragma unroll
    for (int n = 0; n < N; n++) {
        float v = warp_reduce(acc[n]);
        if (lane == 0) yy[n][row] = v;
    }
}

void gemv_q4_n(const uint8_t* W, const __half* S, const XQuant* q, int nb, float* const* ys,
               int64_t rows, int64_t cols, cudaStream_t st) {
    unsigned blocks = (unsigned)((rows + 7) / 8);
    Q4Lanes L;
    for (int i = 0; i < 10; i++) {
        const XQuant& qq = q[i < nb ? i : 0];
        L.eo[i] = qq.eo; L.xs[i] = qq.scale; L.is[i] = qq.isum;
        L.y[i] = ys[i < nb ? i : 0];
    }
    switch (nb) {
        case 2: k_gemv_q4_n<2><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 3: k_gemv_q4_n<3><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 4: k_gemv_q4_n<4><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 5: k_gemv_q4_n<5><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 6: k_gemv_q4_n<6><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 7: k_gemv_q4_n<7><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // maxd6 width-7
        case 8: k_gemv_q4_n<8><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // maxd7 width-8
        case 10: k_gemv_q4_n<10><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        default: fprintf(stderr, "gemv_q4_n: bad nbatch %d\n", nb); exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}

void gemv_q8_n(const int8_t* W, const __half* S, const XQuant* q, int nb, float* const* ys,
               int64_t rows, int64_t cols, cudaStream_t st) {
    unsigned blocks = (unsigned)((rows + 7) / 8);
    Q8Lanes L;
    for (int i = 0; i < 10; i++) {
        const XQuant& qq = q[i < nb ? i : 0];
        L.nat[i] = qq.nat; L.xs[i] = qq.scale;
        L.y[i] = ys[i < nb ? i : 0];
    }
    switch (nb) {
        case 2: k_gemv_q8_n<2><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 3: k_gemv_q8_n<3><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 4: k_gemv_q8_n<4><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 5: k_gemv_q8_n<5><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 6: k_gemv_q8_n<6><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 7: k_gemv_q8_n<7><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // maxd6 width-7
        case 8: k_gemv_q8_n<8><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // maxd7 width-8
        case 10: k_gemv_q8_n<10><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        default: fprintf(stderr, "gemv_q8_n: bad nbatch %d\n", nb); exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_gemv_f16(const __half* __restrict__ W, const float* __restrict__ x,
                           float* __restrict__ y, int64_t cols) {
    int64_t r = blockIdx.x;
    const __half* wr = W + r * cols;
    float acc = 0.f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
        acc += __half2float(wr[c]) * x[c];
    float sum = block_reduce<256>(acc);
    if (threadIdx.x == 0) y[r] = sum;
}

void gemv_q4(const uint8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st) {
    unsigned blocks = (unsigned)((rows + 7) / 8);
    k_gemv_q4<<<blocks, 256, 0, st>>>(W, S, xq.nat, xq.eo, xq.scale, xq.isum, y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}
void gemv_q8(const int8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st) {
    unsigned blocks = (unsigned)((rows + 7) / 8);
    k_gemv_q8<<<blocks, 256, 0, st>>>(W, S, xq.nat, xq.scale, y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}
void gemv_f16(const __half* W, const float* x, float* y, int64_t rows, int64_t cols,
              cudaStream_t st) {
    k_gemv_f16<<<(unsigned)rows, 256, 0, st>>>(W, x, y, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- grid-merged 3-token variants ----------------

__global__ void k_rmsnorm3(CP3 xp, const float* __restrict__ w, P3 yp, int n, float eps) {
    const float* x = xp.p[blockIdx.x];
    float* y = yp.p[blockIdx.x];
    __shared__ float sh[32];
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += x[i] * x[i];
    acc = warp_reduce(acc);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
        v = warp_reduce(v);
        if (threadIdx.x == 0) sh[0] = v;
    }
    __syncthreads();
    float inv = rsqrtf(sh[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}
void rmsnorm3(CP3 x, const float* w, P3 y, int n, float eps, cudaStream_t st, int ntok) {
    k_rmsnorm3<<<ntok, 1024, 0, st>>>(x, w, y, n, eps);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_add3(P3 xp, CP3 yp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) xp.p[blockIdx.y][i] += yp.p[blockIdx.y][i];
}
void add3(P3 x, CP3 y, int n, cudaStream_t st, int ntok) {
    dim3 g((n + 255) / 256, ntok);
    k_add3<<<g, 256, 0, st>>>(x, y, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_silu_mul3(P3 gp, CP3 up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = gp.p[blockIdx.y][i];
    gp.p[blockIdx.y][i] = (v / (1.f + expf(-v))) * up.p[blockIdx.y][i];
}
void silu_mul3(P3 g, CP3 u, int n, cudaStream_t st, int ntok) {
    dim3 gr((n + 255) / 256, ntok);
    k_silu_mul3<<<gr, 256, 0, st>>>(g, u, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_quantize_x3(CP3 xp, int8_t* n0, int8_t* n1, int8_t* n2, int8_t* n3, int8_t* n4,
                              int8_t* n5, int8_t* n6, int8_t* n7, uint2* e0, uint2* e1, uint2* e2,
                              uint2* e3, uint2* e4, uint2* e5, uint2* e6, uint2* e7, float* s0,
                              float* s1, float* s2, float* s3, float* s4, float* s5, float* s6,
                              float* s7, int* i0, int* i1, int* i2, int* i3, int* i4, int* i5,
                              int* i6, int* i7, int nblocks) {
    int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (b >= nblocks) return;
    const int t = blockIdx.y;
    const float* x = xp.p[t];
    // P12b lesson (lane-count landmine): every lane must select its OWN
    // buffers -- a fall-through overwrote lane 4's activation at ntok=6 and
    // corrupted the depth-5 verify (memcheck-blind). maxd6: 7th lane (n6).
    int8_t* nat = t == 0 ? n0 : t == 1 ? n1 : t == 2 ? n2 : t == 3 ? n3 : t == 4 ? n4
                                                : t == 5 ? n5 : t == 6 ? n6 : n7;
    uint2* eo = t == 0 ? e0 : t == 1 ? e1 : t == 2 ? e2 : t == 3 ? e3 : t == 4 ? e4
                                              : t == 5 ? e5 : t == 6 ? e6 : e7;
    float* scale = t == 0 ? s0 : t == 1 ? s1 : t == 2 ? s2 : t == 3 ? s3 : t == 4 ? s4
                                                 : t == 5 ? s5 : t == 6 ? s6 : s7;
    int* isum = t == 0 ? i0 : t == 1 ? i1 : t == 2 ? i2 : t == 3 ? i3 : t == 4 ? i4
                                              : t == 5 ? i5 : t == 6 ? i6 : i7;
    int lane = threadIdx.x & 31;
    float v = x[b * 32 + lane];
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    float s = amax / 127.f;
    float inv = s > 0.f ? 1.f / s : 0.f;
    int q = __float2int_rn(v * inv);
    q = max(-127, min(127, q));
    nat[b * 32 + lane] = (int8_t)q;
    int bsum = q;
    for (int off = 16; off > 0; off >>= 1) bsum += __shfl_xor_sync(0xffffffff, bsum, off);
    if (lane == 0) { scale[b] = s; isum[b] = bsum; }
    int base = (lane & 3) * 8;
    uint32_t e = 0, o = 0;
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        int qe = __shfl_sync(0xffffffff, q, base + 2 * kk);
        int qo = __shfl_sync(0xffffffff, q, base + 2 * kk + 1);
        e |= (uint32_t)(uint8_t)(int8_t)qe << (8 * kk);
        o |= (uint32_t)(uint8_t)(int8_t)qo << (8 * kk);
    }
    if (lane < 4) eo[b * 4 + lane] = make_uint2(e, o);
}
void quantize3(CP3 x, int64_t cols, const XQ3& xq, cudaStream_t st, int ntok) {
    int nblocks = (int)(cols / 32);
    dim3 g((nblocks + 7) / 8, ntok);
    k_quantize_x3<<<g, 256, 0, st>>>(
        x, xq.q[0].nat, xq.q[1].nat, xq.q[2].nat, xq.q[3].nat, xq.q[4].nat, xq.q[5].nat,
        xq.q[6].nat, xq.q[7].nat, xq.q[0].eo, xq.q[1].eo, xq.q[2].eo, xq.q[3].eo, xq.q[4].eo,
        xq.q[5].eo, xq.q[6].eo, xq.q[7].eo, xq.q[0].scale, xq.q[1].scale, xq.q[2].scale,
        xq.q[3].scale, xq.q[4].scale, xq.q[5].scale, xq.q[6].scale, xq.q[7].scale,
        xq.q[0].isum, xq.q[1].isum, xq.q[2].isum, xq.q[3].isum, xq.q[4].isum, xq.q[5].isum,
        xq.q[6].isum, xq.q[7].isum, nblocks);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- elementwise ----------------

__global__ void k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w,
                          float* __restrict__ y, int n, float eps) {
    __shared__ float sh[32];
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += x[i] * x[i];
    acc = warp_reduce(acc);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
        v = warp_reduce(v);
        if (threadIdx.x == 0) sh[0] = v;
    }
    __syncthreads();
    float inv = rsqrtf(sh[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}

__global__ void k_silu_mul(const float* __restrict__ g, const float* __restrict__ u,
                           float* __restrict__ o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = g[i];
    o[i] = (v / (1.f + expf(-v))) * u[i];
}

__global__ void k_embed_row_q8(const int8_t* __restrict__ W, const __half* __restrict__ S,
                               const int* __restrict__ d_token, int64_t cols,
                               float* __restrict__ out) {
    int64_t row = *d_token;
    const int8_t* wr = W + row * cols;
    const __half* sr = S + row * (cols / 128);
    for (int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; c < cols;
         c += (int64_t)gridDim.x * blockDim.x)
        out[c] = (float)wr[c] * __half2float(sr[c / 128]);
}

void rmsnorm(const float* x, const float* w, float* y, int n, float eps, cudaStream_t st) {
    k_rmsnorm<<<1, 1024, 0, st>>>(x, w, y, n, eps);
    CUDA_CHECK(cudaGetLastError());
}
void silu_mul(const float* g, const float* u, float* o, int n, cudaStream_t st) {
    k_silu_mul<<<(n + 255) / 256, 256, 0, st>>>(g, u, o, n);
    CUDA_CHECK(cudaGetLastError());
}
void embed_row_q8(const int8_t* W, const __half* S, const int* d_token, int64_t cols, float* out,
                  cudaStream_t st) {
    k_embed_row_q8<<<8, 256, 0, st>>>(W, S, d_token, cols, out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace q27k
