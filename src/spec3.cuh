// 3-token grid-merged variants for the speculative round. Same per-token work
// distribution as the single-token kernels; tokens ride an extra grid dim.
#pragma once
#include <cuda_runtime.h>

#include "kernels.cuh" // P3/CP3

namespace q27k {

struct IP3 { const int* p[3]; };

// L2 norm over contiguous heads, 3 tokens. (q||k are contiguous: pass 32 heads.)
void l2norm3(P3 x, int n_heads, int head_dim, float eps, cudaStream_t st = 0);

// f16 GEMV, one weight, 3 activation columns.
void gemv_f16_3(const __half* W, CP3 x, P3 y, int64_t rows, int64_t cols, cudaStream_t st = 0);

// gdn gate math for 3 tokens.
void gdn_gates3(CP3 ar, CP3 br, const float* a, const float* dt, P3 g, P3 b, int n,
                cudaStream_t st = 0);

// gated RMS norm (DeltaNet output), 3 tokens.
void gated_norm3(CP3 o, const float* w, CP3 z, P3 out, int n_heads, int head_dim, float eps,
                 cudaStream_t st = 0);

// attention output sigmoid gate, 3 tokens.
void sigmoid_gate3(P3 out, CP3 qg, int n_heads, int head_dim, cudaStream_t st = 0);

// neox partial rope, 3 tokens with per-token device positions.
void rope3(P3 x, int n_heads, int head_dim, int n_rot, int stride, IP3 pos, float freq_base,
           cudaStream_t st = 0);

// KV store for 3 tokens (disjoint slots).
void kv_store3(CP3 k, CP3 v, __half* kc, __half* vc, IP3 pos, int rowlen, cudaStream_t st = 0);

// Flash-decode split-K partial layout: FD_NS position splits per (token, head)
// pair, each partial = {m, l, acc[256]} = FD_ST floats. Every split writes its
// full partial (even when its position range is empty), so scratch must hold
// ntok * n_q_heads * FD_NS * FD_ST floats regardless of context length.
static constexpr int FD_NS = 16;   // splits over positions
static constexpr int FD_ST = 258;  // per-partial stride: m, l, acc[256]

// causal decode attention for 3 tokens; token t attends cache[0 .. *pos.p[t]].
// scratch: [3][n_q_heads][FD_NS][FD_ST] floats (see above).
void attn_decode3(CP3 q, int q_stride, const __half* kc, const __half* vc, P3 out, float* scratch,
                  IP3 pos, int max_ctx, int n_q_heads, int n_kv_heads, int head_dim, float scale,
                  cudaStream_t st = 0);

// embedding row lookup for 3 device tokens.
void embed3(const int8_t* W, const __half* S, IP3 tok, int64_t cols, P3 out, cudaStream_t st = 0);

// Device-side round bookkeeping: prep derives all positions from *d_P and
// snapshots t1; finish decides acceptance, selects next token + h_next, bumps
// *d_P, and writes outcome = {n, t1, dr1, dr2} for a single 16B readback.
void prep_round(const int* d_P, const int* d_token, int* pos_a, int* pos_b, int* pos_c,
                int* pos_m, int* pos_m2, int* outcome, cudaStream_t st = 0);
void finish_round(int* d_P, int* d_token, const int* d_draft, const int* d_draft2,
                  const int* va, const int* vb, const int* vc, const float* x1a,
                  const float* x1b, const float* x1c, float* h_next, int* outcome, int n_embd,
                  cudaStream_t st = 0);

} // namespace q27k
