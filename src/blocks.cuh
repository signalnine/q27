// qwen35 block kernels (reference implementations; see docs/SPEC.md).
// VERIFY-flag assumptions are compile-time constants below — flip when the
// source-verification workflow reports.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// [VERIFY-1] DeltaNet 16->48 head expansion: 1 = tile/modulo (h%16), 0 = interleave (h/3)
#define Q27_GDN_HEAD_TILE 1
// [VERIFY-4] conv window orientation: 1 = tap j multiplies oldest+j (oldest first)
#define Q27_CONV_OLDEST_FIRST 1
// [VERIFY-5] l2norm eps: 1 = rsqrt(sum+eps), 0 = rsqrt(max(sum,eps))
#define Q27_L2NORM_EPS_ADD 1

namespace q27k {

// Per-head RMS norm with row stride (handles q interleaved in the qg buffer).
// x/y: base + h*stride; in-place OK (y==x).
void rmsnorm_heads(const float* x, const float* w, float* y, int n_heads, int head_dim,
                   int stride, float eps, cudaStream_t st = 0);

// Per-head L2 norm, in place, contiguous heads.
void l2norm_heads(float* x, int n_heads, int head_dim, float eps, cudaStream_t st = 0);

// Neox-style partial rope over first n_rot dims of each head (pairs d, d+n_rot/2),
// theta_d = pos * freq_base^(-2d/n_rot). In place; stride between heads.
void rope_neox_partial(float* x, int n_heads, int head_dim, int n_rot, int stride, int pos,
                       float freq_base, cudaStream_t st = 0);

// Causal decode attention for one new token. Q strided (interleaved qg), caches
// contiguous [pos][n_kv][head_dim] f32. scratch: [n_q_heads * (seq_len)] floats.
void attn_decode(const float* q, int q_stride, const float* kcache, const float* vcache,
                 float* out, float* scratch, int seq_len, int n_q_heads, int n_kv_heads,
                 int head_dim, float scale, cudaStream_t st = 0);

// out[h*head_dim+d] *= sigmoid(qg[h*(2*head_dim) + head_dim + d])
void sigmoid_gate_mul(float* out, const float* qg, int n_heads, int head_dim,
                      cudaStream_t st = 0);

// g[h] = ssm_a[h] * softplus(alpha[h] + dt[h]);  beta_out[h] = sigmoid(beta_raw[h])
void gdn_gates(const float* alpha_raw, const float* beta_raw, const float* ssm_a,
               const float* ssm_dt, float* g, float* beta_out, int n_heads,
               cudaStream_t st = 0);

// One decode conv step over all channels; ring [3][channels] rotates in place.
// convw: [channels][4] f32 (taps contiguous). out = silu(conv). qkv = current column.
void conv_step(float* ring, const float* qkv, const float* convw, float* out, int channels,
               cudaStream_t st = 0);

// Gated delta rule, one token. conv_out layout [q 16x128 | k 16x128 | v 48x128]
// (L2 norm already applied to q,k in place). S: [48][128*128] (i fast). o: [48*128].
void delta_step(float* S, const float* conv_out, const float* g, const float* beta, float* o,
                cudaStream_t st = 0);

// out[h*128+d] = rms_norm(o_h)[d] * w[d] * silu(z[h*128+d])   (DeltaNet gated norm)
void gated_norm_gdn(const float* o, const float* w, const float* z, float* out, int n_heads,
                    int head_dim, float eps, cudaStream_t st = 0);

// x += y
void add_inplace(float* x, const float* y, int n, cudaStream_t st = 0);

// index of max element (greedy sampling). d_out: single int on device.
void argmax(const float* x, int n, int* d_out, cudaStream_t st = 0);

} // namespace q27k
