// Grid-merged multi-token variants for the speculative round. Same per-token
// work distribution as the single-token kernels; tokens ride an extra grid dim
// (ntok in 1..4 live lanes; struct slots beyond ntok are never read).
#pragma once
#include <cuda_runtime.h>

#include "cuda_common.h" // KvKind

#include "kernels.cuh" // P3/CP3

namespace q27k {

// IP3 (the const-int lane pack) moved to kernels.cuh next to P3/CP3 -- the
// sampled tail in blocks.cuh takes it too.
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

// M1 record+fold GDN verify chunks (batched-decode spec Appendix A). The
// speculative lanes 1..nsp of a verify round read COMMITTED state only and
// write per-lane outputs only -- no role-state writes. Lane 0 (the pending
// token, always accepted) still commits in place via conv_step/delta_step;
// the accepted speculative lanes are folded into committed state afterwards
// (Engine::flush_fold) from the rows gdn_record3 retains.
//
// gdn_conv_chunk3: per-lane 4-tap conv + silu for lanes 1..nsp. Taps are the
// raw qkv rows of absolute inputs L-3..L, read from the committed ring for
// inputs <= 0 (POST lane-0 shift: launch after lane 0's conv_step) and from
// the lane qkv buffers for inputs >= 1. Arithmetic mirrors k_conv_step
// (blocks.cu) -- same tap order, same silu; ONLY the ring writes are gone.
void gdn_conv_chunk3(const float* ring, CP3 qkv, const float* convw, P3 out, int channels,
                     int nsp, cudaStream_t st = 0);
// gdn_delta_chunk3: gated delta rule for lanes 1..nsp with the committed S
// (POST lane-0: launch after lane 0's delta_step) resident in 64KB dynamic
// smem; writes each lane's o only. Per-step arithmetic mirrors k_delta_step /
// k_delta_scan_T (same tile split, same part[0..3] reduction order) -- the
// smem round-trip is value-preserving fp32, proven bitwise by the seq-prefill
// gate and tools/gdn_chunk_bench.
void gdn_delta_chunk3(const float* S0, CP3 conv, CP3 g, CP3 beta, P3 o, int nsp,
                      cudaStream_t st = 0);
// gdn_record3: retain lane L=1..nsp's raw qkv row, POST-l2norm conv row (launch
// after l2norm3) and g/beta scalars into the per-layer record arena (row L-1),
// so the commit Fold can replay exactly the bytes the verify consumed after
// the lane buffers are reused by the next layer.
// 2026-08-18 GDN mix fusion: the vw>1 decode mix chain collapsed 6 -> 3
// launches. gdn_convnorm3 = conv_chunk3 + l2norm3 + record3 (one launch, all
// lanes); gdn_delta_all = delta_step (lane 0, commits S) + delta_chunk3
// (lanes 1..nsp) with S staged once in smem. Both are bitwise twins of the
// sequences they replace -- gated by build/gdn_fuse_eq; MIRROR WARNINGS in
// spec3.cu apply.
void gdn_convnorm3(const float* ring, CP3 qkv, const float* convw, P3 out, CP3 g, CP3 beta,
                   float* rec_qkv, float* rec_conv, float* rec_g, float* rec_beta, int channels,
                   int n_heads, float eps, int vw, cudaStream_t st);
void gdn_delta_all(const float* Ssrc, float* Sdst, CP3 conv, CP3 g, CP3 beta, P3 o, int nsp,
                   cudaStream_t st);
void gdn_record3(CP3 qkv, CP3 conv, CP3 g, CP3 beta, float* rec_qkv, float* rec_conv,
                 float* rec_g, float* rec_beta, int channels, int n_heads, int nsp,
                 cudaStream_t st = 0);

// attention output sigmoid gate, ntok tokens.
void sigmoid_gate3(P3 out, CP3 qg, int n_heads, int head_dim, cudaStream_t st = 0, int ntok = 3);

// neox partial rope, ntok tokens with per-token device positions.
void rope3(P3 x, int n_heads, int head_dim, int n_rot, int stride, IP3 pos, float freq_base,
           cudaStream_t st = 0, int ntok = 3);

// KV store for ntok tokens (disjoint slots). fp8: E4M3 cache elements (P2).
// M2a: caches are reached through per-pair 64-row block tables (ktab/vtab =
// the pair's table slice, entries = page bases; identity-mapped over the
// per-layer allocs until the M2b pool). Addressing-only vs the raw-pointer
// form -- same bytes, same fp order.
void kv_store3(CP3 k, CP3 v, void* const* ktab, void* const* vtab, IP3 pos, int rowlen,
               cudaStream_t st = 0, int ntok = 3, bool fp8 = false);

// turbo KV store (Q27_KV=turbo3|turbo3v|turbo5k; formats src/turbo3.cuh and
// src/turbo5.cuh, port spec docs/plans/2026-07-11-turbo3-kv-port-spec.md,
// 5-bit plan docs/plans/2026-08-01-5bit-k.md). Cooperative per-128-group
// store: L2-normalize -> forward WHT -> nearest-centroid pack with corrected
// fp16 norm, written block-addressed (row = n_kv_heads * head_dim/128 blocks).
// V is turbo3 (50 B/block) for every kind; kvk picks the K leg -- KV_T3
// turbo3, KV_T3V plain fp16 rows, KV_T5K turbo5 (82 B/block). K must already
// be rope'd.
void kv_store_t3(CP3 k, CP3 v, void* const* ktab, void* const* vtab, IP3 pos, int n_kv_heads,
                 int head_dim, cudaStream_t st = 0, int ntok = 3, int kvk = KV_T3);

// Per-128-group Walsh-Hadamard rotate, in place, ntok tokens: forward on Q
// after rope (inv=false), inverse on attention output after the combine
// (inv=true). stride = floats between consecutive heads (2*head_dim for the
// q||gate layout, head_dim for attnout); only the first head_dim floats of
// each head are touched, so the gate half of qg is preserved.
void wht3(P3 x, int n_heads, int head_dim, int stride, bool inv, cudaStream_t st = 0,
          int ntok = 3);

// Flash-decode split-K partial layout: NS position splits per (token, head)
// pair, each partial = {m, l, acc[256]} = FD_ST floats. scratch must hold
// ntok * n_q_heads * FD_MAXNS * FD_ST floats regardless of context length.
// EMPTY SPLITS ARE NOT WRITTEN by fd2 (the live default path): it returns
// early for sp*chunk >= seq, and k_attn_fd_combine re-derives the same
// used-split count from the same *pos and never reads past it. Those cells of
// scratch therefore hold undefined memory -- a reader that sums a fixed ns
// instead of `used` would pull garbage into the online-softmax rescale. This
// header used to claim every split wrote its partial, which was true only of
// the frozen v1 kernel; the writer/reader agreement is the real contract, and
// any new consumer of `part` must re-derive `used` exactly as the combine does.
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
// meaning bit-for-bit; KV_T3/KV_T3V/KV_T5K route to the fd2 turbo kernel.
// Only KV_T3 has an mma leg; KV_T3V and KV_T5K are returned to fd2 by an
// explicit guard at the top of attn_decode3, because the mma/H16/v1 tails
// would otherwise read them at the WRONG format rather than erroring.
void attn_decode3(CP3 q, int q_stride, const void* const* ktab, const void* const* vtab,
                  P3 out, float* scratch, IP3 pos, int max_ctx, int n_q_heads, int n_kv_heads,
                  int head_dim, float scale, cudaStream_t st = 0, int ntok = 3,
                  int kvk = KV_F16);
// explicit fd2 entry point (unit gate compares this against Q27_FD=v1)
void attn_decode3_fd2(CP3 q, int q_stride, const void* const* ktab, const void* const* vtab,
                      P3 out, float* scratch, IP3 pos, int max_ctx, int n_q_heads,
                      int n_kv_heads, int head_dim, float scale, cudaStream_t st = 0,
                      int ntok = 3, int kvk = KV_F16);

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
