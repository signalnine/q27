// Grid-merged multi-token variants for the speculative round. Same per-token
// work distribution as the single-token kernels; tokens ride an extra grid dim
// (ntok in 1..4 live lanes; struct slots beyond ntok are never read).
#pragma once
#include <cuda_runtime.h>

#include "kernels.cuh" // P3/CP3

namespace q27k {

struct IP3 { const int* p[5]; };

// L2 norm over contiguous heads, ntok tokens. (q||k are contiguous: pass 32 heads.)
void l2norm3(P3 x, int n_heads, int head_dim, float eps, cudaStream_t st = 0, int ntok = 3);

// f16 GEMV, one weight, ntok activation columns.
void gemv_f16_3(const __half* W, CP3 x, P3 y, int64_t rows, int64_t cols, cudaStream_t st = 0,
                int ntok = 3);

// gdn gate math for ntok tokens.
void gdn_gates3(CP3 ar, CP3 br, const float* a, const float* dt, P3 g, P3 b, int n,
                cudaStream_t st = 0, int ntok = 3);

// gated RMS norm (DeltaNet output), ntok tokens.
void gated_norm3(CP3 o, const float* w, CP3 z, P3 out, int n_heads, int head_dim, float eps,
                 cudaStream_t st = 0, int ntok = 3);

// attention output sigmoid gate, ntok tokens.
void sigmoid_gate3(P3 out, CP3 qg, int n_heads, int head_dim, cudaStream_t st = 0, int ntok = 3);

// neox partial rope, ntok tokens with per-token device positions.
void rope3(P3 x, int n_heads, int head_dim, int n_rot, int stride, IP3 pos, float freq_base,
           cudaStream_t st = 0, int ntok = 3);

// KV store for ntok tokens (disjoint slots). fp8: E4M3 cache elements (P2).
void kv_store3(CP3 k, CP3 v, void* kc, void* vc, IP3 pos, int rowlen, cudaStream_t st = 0,
               int ntok = 3, bool fp8 = false);

// Flash-decode split-K partial layout: NS position splits per (token, head)
// pair, each partial = {m, l, acc[256]} = FD_ST floats. Every split writes its
// full partial (even when its position range is empty), so scratch must hold
// ntok * n_q_heads * FD_MAXNS * FD_ST floats regardless of context length.
// FD_NS stays 16 so Q27_FD=v1 reproduces the historical kernel bit-for-bit;
// fd2 uses its own FD2_NS -- with register accumulators the block is cheap,
// and the grid needs ~4-5 blocks per SM resident for latency hiding
// (4 kv-heads x FD2_NS x ntok blocks; see docs/attn-fd2-design.md).
static constexpr int FD_NS = 16;    // v1 splits over positions (frozen)
static constexpr int FD2_NS = 128;  // fd2 splits (perf-swept, BUILDLOG)
static constexpr int FD_MAXNS = FD2_NS > FD_NS ? FD2_NS : FD_NS;
static constexpr int FD_ST = 258;   // per-partial stride: m, l, acc[256]

// causal decode attention for ntok tokens; token t attends cache[0 .. *pos.p[t]].
// scratch: [ntok][n_q_heads][FD_NS][FD_ST] floats (see above).
// Default path = fd2 (register-accumulator kernel, docs/attn-fd2-design.md);
// Q27_FD=v1 selects the original kernel. The env is read at LAUNCH time, so
// graph capture bakes the choice per process.
void attn_decode3(CP3 q, int q_stride, const void* kc, const void* vc, P3 out, float* scratch,
                  IP3 pos, int max_ctx, int n_q_heads, int n_kv_heads, int head_dim, float scale,
                  cudaStream_t st = 0, int ntok = 3, bool fp8 = false);
// explicit fd2 entry point (unit gate compares this against Q27_FD=v1)
void attn_decode3_fd2(CP3 q, int q_stride, const void* kc, const void* vc, P3 out,
                      float* scratch, IP3 pos, int max_ctx, int n_q_heads, int n_kv_heads,
                      int head_dim, float scale, cudaStream_t st = 0, int ntok = 3,
                      bool fp8 = false);

// embedding row lookup for ntok device tokens.
void embed3(const int8_t* W, const __half* S, IP3 tok, int64_t cols, P3 out, cudaStream_t st = 0,
            int ntok = 3);

// Device-side round bookkeeping: prep derives all positions from *d_P and
// snapshots t1; finish decides acceptance over the depth-4 draft chain,
// selects next token + h_next, bumps *d_P, and writes
// outcome = {n, t1, dr1, dr2, dr3, dr4} for a single small readback.
void prep_round(const int* d_P, const int* d_token, int* pos_a, int* pos_b, int* pos_c,
                int* pos_d, int* pos_e, int* pos_m, int* pos_m2, int* pos_m3, int* pos_m4,
                int* outcome, cudaStream_t st = 0);
void finish_round(int* d_P, int* d_token, const int* d_draft, const int* d_draft2,
                  const int* d_draft3, const int* d_draft4, const int* va, const int* vb,
                  const int* vc, const int* vd, const int* ve, const float* x1a,
                  const float* x1b, const float* x1c, const float* x1d, const float* x1e,
                  float* h_next, int* outcome, int n_embd, const int* cap,
                  cudaStream_t st = 0);

} // namespace q27k
