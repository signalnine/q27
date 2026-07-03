// qwen35 block kernels (reference implementations; see docs/SPEC.md).
// VERIFY-flag assumptions are compile-time constants below — flip when the
// source-verification workflow reports.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <cuda_runtime.h>

// VERIFY-1/3/4/5 all resolved against ggml source (workflow wf_b19a6dde, high confidence):
// head expansion = tile/modulo (converter pre-permutes V heads to tiled order),
// imrope == neox for text, conv oldest-first, l2norm eps floors the norm.
#define Q27_GDN_HEAD_TILE 1
#define Q27_CONV_OLDEST_FIRST 1

namespace q27k {

// Per-head RMS norm with row stride (handles q interleaved in the qg buffer).
// x/y: base + h*stride; in-place OK (y==x).
void rmsnorm_heads(const float* x, const float* w, float* y, int n_heads, int head_dim,
                   int stride, float eps, cudaStream_t st = 0);

// Per-head L2 norm, in place, contiguous heads.
void l2norm_heads(float* x, int n_heads, int head_dim, float eps, cudaStream_t st = 0);

// Neox-style partial rope over first n_rot dims of each head (pairs d, d+n_rot/2),
// theta_d = pos * freq_base^(-2d/n_rot). In place; stride between heads.
// pos read from device (*d_pos) so the launch is CUDA-graph-stable.
void rope_neox_partial(float* x, int n_heads, int head_dim, int n_rot, int stride,
                       const int* d_pos, float freq_base, cudaStream_t st = 0);

// Causal decode attention for one new token; seq_len = *d_pos + 1 read on device.
// Q strided (interleaved qg), caches contiguous [pos][n_kv][head_dim], fp16 or
// fp8 E4M3 elements (fp8 = true selects fp8; P2, opt-in via Q27_KV=fp8).
// scratch: [n_q_heads][FD_NS][FD_ST] floats (flash-decode partials, spec3.cuh).
void attn_decode(const float* q, int q_stride, const void* kcache, const void* vcache,
                 float* out, float* scratch, const int* d_pos, int max_ctx, int n_q_heads,
                 int n_kv_heads, int head_dim, float scale, cudaStream_t st = 0,
                 bool fp8 = false);

// Store this token's K/V rows into the caches at position *d_pos.
void kv_store(const float* kbuf, const float* vbuf, void* kcache, void* vcache,
              const int* d_pos, int rowlen, cudaStream_t st = 0, bool fp8 = false);

// End-of-token bookkeeping (device-chained decode): d_gen[*d_step] = *d_token;
// (*d_step)++; (*d_pos)++.
void advance(int* d_pos, int* d_step, int* d_gen, const int* d_token, cudaStream_t st = 0);

// out[h*head_dim+d] *= sigmoid(qg[h*(2*head_dim) + head_dim + d])
void sigmoid_gate_mul(float* out, const float* qg, int n_heads, int head_dim,
                      cudaStream_t st = 0);

// g[h] = ssm_a[h] * softplus(alpha[h] + dt[h]);  beta_out[h] = sigmoid(beta_raw[h])
void gdn_gates(const float* alpha_raw, const float* beta_raw, const float* ssm_a,
               const float* ssm_dt, float* g, float* beta_out, int n_heads,
               cudaStream_t st = 0);

// One decode conv step over all channels. Reads ring_src, writes rotated ring to
// ring_dst (== ring_src for in-place / committed tokens; spare buffer for
// speculative tokens -> accept = pointer swap, reject = free rollback).
void conv_step(const float* ring_src, float* ring_dst, const float* qkv, const float* convw,
               float* out, int channels, cudaStream_t st = 0);

// Gated delta rule, one token. conv_out layout [q 16x128 | k 16x128 | v 48x128]
// (L2 norm already applied to q,k in place). Reads S_src, writes S_dst (same
// pointer for committed tokens, spare for speculative).
void delta_step(const float* S_src, float* S_dst, const float* conv_out, const float* g,
                const float* beta, float* o, cudaStream_t st = 0);

// out[h*128+d] = rms_norm(o_h)[d] * w[d] * silu(z[h*128+d])   (DeltaNet gated norm)
void gated_norm_gdn(const float* o, const float* w, const float* z, float* out, int n_heads,
                    int head_dim, float eps, cudaStream_t st = 0);

// x += y
void add_inplace(float* x, const float* y, int n, cudaStream_t st = 0);

// index of max element (greedy sampling). d_out: single int on device.
// Teacher-forced NLL over a [nrows, vocab] row-major logit matrix:
// nll[r] = logsumexp(logits[r,:]) - logits[r, tgt[r]].
void nll_rows(const float* logits, const int* tgt, float* nll, int nrows, int64_t vocab,
              cudaStream_t st = 0);

// d_scratch: one u64 on device (caller-allocated; no allocation during graph capture).
// P7: constrained variant -- argmax over tokens whose bit is set in
// pool[mask_ids[slot]]; id -1 or null pool = plain argmax (bitwise).
void argmax_masked(const float* x, int n, const unsigned* pool, int words, const int* mask_ids,
                   int slot, int* d_out, unsigned long long* d_scratch, cudaStream_t st = 0);
void argmax(const float* x, int n, int* d_out, unsigned long long* d_scratch,
            cudaStream_t st = 0);

} // namespace q27k
