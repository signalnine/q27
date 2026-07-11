// Grid-merged multi-token variants for the speculative round. Same per-token
// work distribution as the single-token kernels; tokens ride an extra grid dim
// (ntok in 1..4 live lanes; struct slots beyond ntok are never read).
#pragma once
#include <cuda_runtime.h>

#include "cuda_common.h" // KvKind

#include "kernels.cuh" // P3/CP3

namespace q27k {

struct IP3 { const int* p[16]; }; // width-12: 12 live lanes max, 16 slots (see P3/CP3)
// Writable-int lane bundle (positions, verdict slots): prep/finish hit the
// 17/25-param wall at width 8, so wide-verify pointer args ride these
// by-value structs instead of growing the signatures (width-12 P0).
struct WIP3 { int* p[16]; };

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

// turbo3 KV (Q27_KV=turbo3|turbo3v; format src/turbo3.cuh, port spec
// docs/plans/2026-07-11-turbo3-kv-port-spec.md). Cooperative per-128-group
// store: L2-normalize -> forward WHT -> 3-bit nearest-centroid pack with
// corrected fp16 norm, written block-addressed (row = n_kv_heads *
// head_dim/128 blocks of 50 B). k_plain: the K side is stored as plain fp16
// rows instead (turbo3v = fp16 K + turbo3 V). K must already be rope'd.
void kv_store_t3(CP3 k, CP3 v, void* kc, void* vc, IP3 pos, int n_kv_heads, int head_dim,
                 cudaStream_t st = 0, int ntok = 3, bool k_plain = false);

// Per-128-group Walsh-Hadamard rotate, in place, ntok tokens: forward on Q
// after rope (inv=false), inverse on attention output after the combine
// (inv=true). stride = floats between consecutive heads (2*head_dim for the
// q||gate layout, head_dim for attnout); only the first head_dim floats of
// each head are touched, so the gate half of qg is preserved.
void wht3(P3 x, int n_heads, int head_dim, int stride, bool inv, cudaStream_t st = 0,
          int ntok = 3);

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
// kvk (KvKind, cuda_common.h): widened from `bool fp8` -- 0/1 keep the old
// meaning bit-for-bit; KV_T3/KV_T3V route to the fd2 turbo3 kernel (v1 and
// mma have no turbo3 path and fall through to fd2 silently).
void attn_decode3(CP3 q, int q_stride, const void* kc, const void* vc, P3 out, float* scratch,
                  IP3 pos, int max_ctx, int n_q_heads, int n_kv_heads, int head_dim, float scale,
                  cudaStream_t st = 0, int ntok = 3, int kvk = KV_F16);
// explicit fd2 entry point (unit gate compares this against Q27_FD=v1)
void attn_decode3_fd2(CP3 q, int q_stride, const void* kc, const void* vc, P3 out,
                      float* scratch, IP3 pos, int max_ctx, int n_q_heads, int n_kv_heads,
                      int head_dim, float scale, cudaStream_t st = 0, int ntok = 3,
                      int kvk = KV_F16);

// embedding row lookup for ntok device tokens.
void embed3(const int8_t* W, const __half* S, IP3 tok, int64_t cols, P3 out, cudaStream_t st = 0,
            int ntok = 3);

// Device-side round bookkeeping (width-12 P0: pointer-struct signatures --
// the old flat lists sat at 17/25 params and could not widen). prep derives
// nv verify positions (pos_v.p[t] = P+1+t) and nm MTP positions from *d_P
// and snapshots t1; finish decides acceptance over the draft chain, selects
// next token + h_next, bumps *d_P, and writes
// outcome = {n, t1, dr1..dr11, pending} (14 ints) for a single small readback.
void prep_round(const int* d_P, const int* d_token, WIP3 pos_v, WIP3 pos_m, int nv, int nm,
                int* outcome, cudaStream_t st = 0);
// max_draft (P12 gated depth): the widest verify column this graph computed
// (W-1 for a width-W verify). Drafts beyond it are forced rejected so a
// narrow-verify graph never commits an uncomputed lane. max_draft=4 = the full
// depth-4 round (bit-identical to the pre-P12 path). drafts = 11 slots
// (lanes b..l), verdicts/x1s = 12 lanes; slots past max_draft/width are
// dereferenced but never influence the outcome (same class as the old flat
// args, which also read all 7/8 unconditionally).
void finish_round(int* d_P, int* d_token, IP3 drafts, IP3 verdicts, CP3 x1s, float* h_next,
                  int* outcome, int n_embd, const int* cap, int max_draft = 4,
                  cudaStream_t st = 0);

} // namespace q27k
