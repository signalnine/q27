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

XQuant xquant_alloc(int64_t max_cols) {
    XQuant xq;
    CUDA_CHECK(cudaMalloc((void**)&xq.nat, max_cols));
    CUDA_CHECK(cudaMalloc((void**)&xq.eo, max_cols / 8 * sizeof(uint2)));
    CUDA_CHECK(cudaMalloc((void**)&xq.scale, max_cols / 32 * 4));
    CUDA_CHECK(cudaMalloc((void**)&xq.isum, max_cols / 32 * 4));
    return xq;
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
        const uint2* xp = xeo + (size_t)ch * 4;
        const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
        int di = 0;
#pragma unroll
        for (int u = 0; u < 4; u++) {
            uint2 xv = __ldg(xp + u);
            di = __dp4a((int)(ws[u] & 0x0F0F0F0Fu), (int)xv.x, di);
            di = __dp4a((int)((ws[u] >> 4) & 0x0F0F0F0Fu), (int)xv.y, di);
        }
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

template <int N>
__global__ void k_gemv_q4_n(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                            const uint2* __restrict__ eo0, const uint2* __restrict__ eo1,
                            const uint2* __restrict__ eo2, const uint2* __restrict__ eo3,
                            const float* __restrict__ xs0, const float* __restrict__ xs1,
                            const float* __restrict__ xs2, const float* __restrict__ xs3,
                            const int* __restrict__ is0, const int* __restrict__ is1,
                            const int* __restrict__ is2, const int* __restrict__ is3,
                            float* __restrict__ y, int64_t rows, int64_t cols) {
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint4* wr = (const uint4*)(W + row * (cols / 2));
    const __half* sr = S + row * (cols / 64);
    const int n_chunks = (int)(cols / 32);

    const uint2* eos[4] = {eo0, eo1, eo2, eo3};
    const float* xss[4] = {xs0, xs1, xs2, xs3};
    const int* iss[4] = {is0, is1, is2, is3};
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; n++) acc[n] = 0.f;

    for (int ch = lane; ch < n_chunks; ch += 32) {
        uint4 w = __ldg(wr + ch);
        const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
        float wsc = __half2float(__ldg(sr + (ch >> 1)));
#pragma unroll
        for (int n = 0; n < N; n++) {
            const uint2* xp = eos[n] + (size_t)ch * 4;
            int di = 0;
#pragma unroll
            for (int u = 0; u < 4; u++) {
                uint2 xv = __ldg(xp + u);
                di = __dp4a((int)(ws[u] & 0x0F0F0F0Fu), (int)xv.x, di);
                di = __dp4a((int)((ws[u] >> 4) & 0x0F0F0F0Fu), (int)xv.y, di);
            }
            acc[n] += wsc * __ldg(xss[n] + ch) * (float)(di - 8 * __ldg(iss[n] + ch));
        }
    }
#pragma unroll
    for (int n = 0; n < N; n++) {
        float v = warp_reduce(acc[n]);
        if (lane == 0) y[(size_t)n * rows + row] = v;
    }
}

template <int N>
__global__ void k_gemv_q8_n(const int8_t* __restrict__ W, const __half* __restrict__ S,
                            const int8_t* __restrict__ n0, const int8_t* __restrict__ n1,
                            const int8_t* __restrict__ n2, const int8_t* __restrict__ n3,
                            const float* __restrict__ xs0, const float* __restrict__ xs1,
                            const float* __restrict__ xs2, const float* __restrict__ xs3,
                            float* __restrict__ y, int64_t rows, int64_t cols) {
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint4* wr = (const uint4*)(W + row * cols);
    const __half* sr = S + row * (cols / 128);
    const int n_chunks = (int)(cols / 32);

    const int8_t* nats[4] = {n0, n1, n2, n3};
    const float* xss[4] = {xs0, xs1, xs2, xs3};
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
#pragma unroll
    for (int n = 0; n < N; n++) {
        float v = warp_reduce(acc[n]);
        if (lane == 0) y[(size_t)n * rows + row] = v;
    }
}

void gemv_q4_n(const uint8_t* W, const __half* S, const XQuant* q, int nb, float* y,
               int64_t rows, int64_t cols, cudaStream_t st) {
    unsigned blocks = (unsigned)((rows + 7) / 8);
    const XQuant &a = q[0], &b = q[nb > 1 ? 1 : 0], &c = q[nb > 2 ? 2 : 0], &d = q[nb > 3 ? 3 : 0];
    switch (nb) {
        case 2: k_gemv_q4_n<2><<<blocks, 256, 0, st>>>(W, S, a.eo, b.eo, c.eo, d.eo, a.scale,
                b.scale, c.scale, d.scale, a.isum, b.isum, c.isum, d.isum, y, rows, cols); break;
        case 3: k_gemv_q4_n<3><<<blocks, 256, 0, st>>>(W, S, a.eo, b.eo, c.eo, d.eo, a.scale,
                b.scale, c.scale, d.scale, a.isum, b.isum, c.isum, d.isum, y, rows, cols); break;
        case 4: k_gemv_q4_n<4><<<blocks, 256, 0, st>>>(W, S, a.eo, b.eo, c.eo, d.eo, a.scale,
                b.scale, c.scale, d.scale, a.isum, b.isum, c.isum, d.isum, y, rows, cols); break;
        default: fprintf(stderr, "gemv_q4_n: bad nbatch %d\n", nb); exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}

void gemv_q8_n(const int8_t* W, const __half* S, const XQuant* q, int nb, float* y, int64_t rows,
               int64_t cols, cudaStream_t st) {
    unsigned blocks = (unsigned)((rows + 7) / 8);
    const XQuant &a = q[0], &b = q[nb > 1 ? 1 : 0], &c = q[nb > 2 ? 2 : 0], &d = q[nb > 3 ? 3 : 0];
    switch (nb) {
        case 2: k_gemv_q8_n<2><<<blocks, 256, 0, st>>>(W, S, a.nat, b.nat, c.nat, d.nat, a.scale,
                b.scale, c.scale, d.scale, y, rows, cols); break;
        case 3: k_gemv_q8_n<3><<<blocks, 256, 0, st>>>(W, S, a.nat, b.nat, c.nat, d.nat, a.scale,
                b.scale, c.scale, d.scale, y, rows, cols); break;
        case 4: k_gemv_q8_n<4><<<blocks, 256, 0, st>>>(W, S, a.nat, b.nat, c.nat, d.nat, a.scale,
                b.scale, c.scale, d.scale, y, rows, cols); break;
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
