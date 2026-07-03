// Batched prefill kernels: process a chunk of T prompt tokens per pass so the
// weight stream amortizes across tokens (GEMM with a 32-token register tile)
// instead of re-reading 16GB per token. GDN state scans sequentially inside a
// single kernel with S resident in shared memory; attention runs the proven
// two-pass softmax in 32-token sub-batches.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels.cuh"

namespace q27k {

// Batched GEMM: y[t*rows + row]. Activations quantized token-major with
// quantize_x(xT, T*cols, xq) — token t's chunk data at offset t*(cols/32).
void gemm_q4_T(const uint8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
               int64_t cols, int T, cudaStream_t st);
void gemm_q8_T(const int8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
               int64_t cols, int T, cudaStream_t st);
void gemm_f16_T(const __half* W, const float* xT, float* y, int64_t rows, int64_t cols, int T,
                cudaStream_t st);

// Batched small ops. Layout everywhere: [T][dim] token-major contiguous.
void embed_rows_q8_T(const int8_t* emb, const __half* scales, const int* toks, int cols, int T,
                     float* out, cudaStream_t st);
void rmsnorm_T(const float* x, const float* w, float* y, int n, int T, float eps,
               cudaStream_t st, int in_row = 0, int out_row = 0);
void rmsnorm_heads_T(const float* x, const float* w, float* y, int n_heads, int head_dim,
                     int stride, int row_elems, int T, float eps, cudaStream_t st);
void l2norm_heads_T(float* x, int n_heads, int head_dim, int row_elems, int T, float eps,
                    cudaStream_t st);
void rope_neox_T(float* x, int n_heads, int head_dim, int n_rot, int stride, int row_elems,
                 int base_pos, int T, float freq_base, cudaStream_t st);
void gdn_gates_T(const float* ar, const float* br, const float* a, const float* dt, float* g,
                 float* b, int n_heads, int T, cudaStream_t st);
void sigmoid_gate_mul_T(float* out, const float* qg, int n_heads, int head_dim, int T,
                        cudaStream_t st);
void gated_norm_gdn_T(const float* o, const float* w, const float* z, float* out, int n_heads,
                      int head_dim, int T, float eps, cudaStream_t st);
void conv_prefill_T(float* ring, const float* qkvT, const float* w, float* outT, int channels,
                    int T, cudaStream_t st);
void kv_store_T(const float* kT, const float* vT, void* kc, void* vc, int base_pos,
                int rowlen, int T, cudaStream_t st, bool fp8 = false);
// Flash-attention prefill for a sub-batch of SB tokens starting at (base_pos+t0);
// online softmax, no position scratch. Caches fp16, or fp8 E4M3 when fp8 (P2).
void attn_prefill_T(const float* qT, int q_stride, int q_row, const void* kc, const void* vc,
                    float* outT, int out_row, int base_pos, int t0, int SB, int n_q_heads,
                    int n_kv_heads, int head_dim, float scale, cudaStream_t st,
                    bool fp8 = false);
// Sequential gated delta rule over T tokens, S resident in shared memory.
void delta_scan_T(float* S_global, const float* convT, const float* gT, const float* betaT,
                  float* oT, int T, cudaStream_t st);

} // namespace q27k
