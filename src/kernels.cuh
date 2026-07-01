// q27 reference kernels: correct first, fast later (M2 replaces the GEMVs).
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace q27k {

// Dequantize an entire tensor to f32 (validation / small tensors only).
void dequant_q4(const uint8_t* W, const __half* S, float* out, int64_t rows, int64_t cols,
                cudaStream_t st = 0);
void dequant_q8(const int8_t* W, const __half* S, float* out, int64_t rows, int64_t cols,
                cudaStream_t st = 0);

// Quantized activation vector (mmvq-style): per 32-element block, int8 values with
// f32 scale and integer block sum. Two byte orders are kept: natural (for Q8 weights)
// and even/odd split per 8 (matching Q4 nibble unpack for dp4a).
struct XQuant {
    int8_t* nat = nullptr;   // [cols]
    uint2* eo = nullptr;     // [cols/8]: .x = bytes {x0,x2,x4,x6}, .y = {x1,x3,x5,x7}
    float* scale = nullptr;  // [cols/32]
    int* isum = nullptr;     // [cols/32] sum of quantized values per block
};
XQuant xquant_alloc(int64_t max_cols);
void quantize_x(const float* x, int64_t cols, const XQuant& xq, cudaStream_t st = 0);

// y[r] = sum_c W[r,c] * x[c].  W quantized row-major, reduction along contiguous axis.
// Q4/Q8 use dp4a against the pre-quantized activation vector.
void gemv_q4(const uint8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st = 0);
void gemv_q8(const int8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st = 0);

// Batched: one weight pass, N quantized activation columns -> y[n][rows]
// (y column-major by batch: y + n*rows). N in 2..4. The speculative-verify core.
void gemv_q4_n(const uint8_t* W, const __half* S, const XQuant* xqs, int nbatch, float* y,
               int64_t rows, int64_t cols, cudaStream_t st = 0);
void gemv_q8_n(const int8_t* W, const __half* S, const XQuant* xqs, int nbatch, float* y,
               int64_t rows, int64_t cols, cudaStream_t st = 0);
void gemv_f16(const __half* W, const float* x, float* y, int64_t rows, int64_t cols,
              cudaStream_t st = 0);

// y = x * rsqrt(mean(x^2) + eps) * w      (single vector, n elements)
void rmsnorm(const float* x, const float* w, float* y, int n, float eps, cudaStream_t st = 0);

// out[i] = silu(gate[i]) * up[i]
void silu_mul(const float* gate, const float* up, float* out, int n, cudaStream_t st = 0);

// out[0..cols) = dequantized row *d_token of a Q8_G128 matrix (embedding lookup)
void embed_row_q8(const int8_t* W, const __half* S, const int* d_token, int64_t cols, float* out,
                  cudaStream_t st = 0);

} // namespace q27k
