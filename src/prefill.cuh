// Batched prefill kernels: process a chunk of T prompt tokens per pass so the
// weight stream amortizes across tokens (GEMM with a 32-token register tile)
// instead of re-reading 16GB per token. GDN state scans sequentially inside a
// single kernel with S resident in shared memory; attention runs the proven
// two-pass softmax in 32-token sub-batches.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_common.h" // KvKind
#include "kernels.cuh"

namespace q27k {

// Caller-owned split-K partial buffer for the prefill weight GEMM. Passing a
// reserved SplitKScratch* to gemm_q4_T/gemm_q8_T lets the launcher auto-split
// the K axis across gridDim.z when the output grid underfills the SMs (short-
// prompt / suffix-round prefill), filling idle SMs at 1.8-2.3x. Split output
// is tolerance-gated (non-bitwise vs the single-CTA path -- the group-scaled
// float K-sum is regrouped), same class as the attention P4 split, so only
// the g64 serving path splits; sk==nullptr (the default) disables splitting
// entirely, keeping the canonical --pf leg and the kernel tests bitwise.
// Pinned to one stream for its lifetime (each engine owns its own).
struct SplitKScratch {
    float* buf = nullptr;
    size_t cap = 0; // floats
};
void splitk_scratch_reserve(SplitKScratch* sk);

// Batched GEMM: y[t*rows + row]. Activations quantized token-major with
// quantize_x(xT, T*cols, xq) — token t's chunk data at offset t*(cols/32).
void gemm_q4_T(const uint8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
               int64_t cols, int T, cudaStream_t st, SplitKScratch* sk = nullptr);
void gemm_q8_T(const int8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
               int64_t cols, int T, cudaStream_t st, SplitKScratch* sk = nullptr);
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
// turbo3 prefill store: cooperative per-128-group quantize of T tokens'
// rope'd K + raw V into block caches at rows base_pos..base_pos+T-1
// (same pipeline as the decode store -- shared turbo3_quant_group).
// k_plain: K side stays plain fp16 rows (turbo3v).
void kv_store_T_t3(const float* kT, const float* vT, void* kc, void* vc, int base_pos,
                   int n_kv_heads, int head_dim, int T, cudaStream_t st, bool k_plain = false);
// Per-128-group WHT rotate over a flat [T][row_stride] buffer (prefill Q
// forward after rope; attention output inverse before the sigmoid gate).
// head_stride = floats between heads within a row (2*head_dim for qgT,
// head_dim for attnT).
void wht_T(float* x, int n_heads, int head_dim, int head_stride, int row_stride, int T,
           bool inv, cudaStream_t st);
// Flash-attention prefill for a sub-batch of SB tokens starting at (base_pos+t0);
// online softmax. Caches fp16, or fp8 E4M3 when fp8 (P2). At deep base_pos the
// MMA path splits positions across gridDim.z blocks (P4, SM starvation fix)
// and merges {m,l,O} partials in a combine kernel; `part` must then hold
// n_q_heads * SB_rounded_to_16 * PF_SPLIT_MAX * 258 floats. part == nullptr
// disables splitting (exact pre-split path).
constexpr int PF_SPLIT_MAX = 8;
// kvk (KvKind): widened from `bool fp8` (0/1 keep the old meaning). KV_T3
// runs the f16-MMA path with turbo3 tile dequant (lite via Q27_ATTN_PF=lite);
// KV_T3V (fp16 K + turbo3 V, diagnostic) always uses the lite path.
void attn_prefill_T(const float* qT, int q_stride, int q_row, const void* kc, const void* vc,
                    float* outT, int out_row, float* part, int base_pos, int t0, int SB,
                    int n_q_heads, int n_kv_heads, int head_dim, float scale, cudaStream_t st,
                    int kvk = KV_F16);
// Sequential gated delta rule over T tokens, S resident in shared memory.
// P6: the 128 S-columns per head are independent, so the launcher slices them
// across delta_scan_nsplit() blocks per head (48 blocks alone starve 170 SMs).
// Q27_DS_SPLIT forces the count (1/2/4/8; 1 = exact legacy kernel, split
// paths reorder the row reductions -> tolerance-gated like attention splits).
int delta_scan_nsplit(int T);
// Caller-owned scratch for the WY scan's KKt/QKt chunk panels: written by
// k_delta_wy_kk and read by k_delta_wy on the caller's stream with no
// cross-stream ordering, so each engine must pass its own instance (R1b
// prerequisite -- a shared set races once two engines' prefills are in
// flight, and regrow would free panels another stream still reads). A
// WyScratch is also pinned to ONE stream for its lifetime: regrow drains
// only the caller's stream before freeing. The seq path leaves it untouched.
struct WyScratch {
    float* kkt = nullptr;
    float* qkt = nullptr;
    int cap_nch = 0;
};
// Pre-size the panels for prompts chunked at <= T_max (engine: PF_T) so the
// serving path never regrows mid-flight -- a regrow there is a stream drain
// plus cudaFree/cudaMalloc under the global allocator lock while a sibling
// engine may be decoding. delta_scan_T keeps the lazy grow as a fallback for
// callers that skip this.
void wy_scratch_reserve(WyScratch* wy, int T_max);
void delta_scan_T(float* S_global, const float* convT, const float* gT, const float* betaT,
                  float* oT, int T, cudaStream_t st, WyScratch* wy);

} // namespace q27k
