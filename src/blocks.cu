#include "blocks.cuh"
#include "cuda_common.h"

#include <cfloat>

namespace q27k {

// ---------------- norms ----------------

__global__ void k_rmsnorm_heads(const float* __restrict__ x, const float* __restrict__ w,
                                float* __restrict__ y, int head_dim, int stride, float eps) {
    const float* xh = x + (size_t)blockIdx.x * stride;
    float* yh = y + (size_t)blockIdx.x * stride;
    __shared__ float sh[256];
    float acc = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) acc += xh[i] * xh[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) yh[i] = xh[i] * inv * w[i];
}

void rmsnorm_heads(const float* x, const float* w, float* y, int n_heads, int head_dim,
                   int stride, float eps, cudaStream_t st) {
    k_rmsnorm_heads<<<n_heads, 256, 0, st>>>(x, w, y, head_dim, stride, eps);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_l2norm_heads(float* __restrict__ x, int head_dim, float eps) {
    float* xh = x + (size_t)blockIdx.x * head_dim;
    __shared__ float sh[128];
    float acc = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) acc += xh[i] * xh[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    // ggml semantics: y = x / max(sqrt(sum), eps)  == x * rsqrt(max(sum, eps^2))
    float inv = rsqrtf(fmaxf(sh[0], eps * eps));
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) xh[i] *= inv;
}

void l2norm_heads(float* x, int n_heads, int head_dim, float eps, cudaStream_t st) {
    k_l2norm_heads<<<n_heads, 128, 0, st>>>(x, head_dim, eps);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- rope ----------------

__global__ void k_rope_neox(float* __restrict__ x, int head_dim, int n_rot, int stride,
                            const int* __restrict__ d_pos, float freq_base) {
    float* xh = x + (size_t)blockIdx.x * stride;
    int d = threadIdx.x; // pair index 0..n_rot/2-1
    if (d >= n_rot / 2) return;
    float theta = (float)(*d_pos) * powf(freq_base, -2.0f * d / n_rot);
    float c = cosf(theta), s = sinf(theta);
    float x0 = xh[d], x1 = xh[d + n_rot / 2];
    xh[d] = x0 * c - x1 * s;
    xh[d + n_rot / 2] = x0 * s + x1 * c;
}

void rope_neox_partial(float* x, int n_heads, int head_dim, int n_rot, int stride,
                       const int* d_pos, float freq_base, cudaStream_t st) {
    k_rope_neox<<<n_heads, 32, 0, st>>>(x, head_dim, n_rot, stride, d_pos, freq_base);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- attention (decode, reference) ----------------

// attention decode implemented in spec3.cu (flash-decode)

template <typename CT>
__global__ void k_kv_store(const float* __restrict__ kbuf, const float* __restrict__ vbuf,
                           CT* __restrict__ kc, CT* __restrict__ vc,
                           const int* __restrict__ d_pos, int rowlen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowlen) return;
    size_t off = (size_t)(*d_pos) * rowlen + i;
    kv_set(kc[off], kbuf[i]);
    kv_set(vc[off], vbuf[i]);
}

void kv_store(const float* kbuf, const float* vbuf, void* kcache, void* vcache,
              const int* d_pos, int rowlen, cudaStream_t st, bool fp8) {
    dim3 g((rowlen + 255) / 256);
    if (fp8)
        k_kv_store<<<g, 256, 0, st>>>(kbuf, vbuf, (__nv_fp8_e4m3*)kcache,
                                      (__nv_fp8_e4m3*)vcache, d_pos, rowlen);
    else
        k_kv_store<<<g, 256, 0, st>>>(kbuf, vbuf, (__half*)kcache, (__half*)vcache, d_pos,
                                      rowlen);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_advance(int* d_pos, int* d_step, int* d_gen, const int* d_token) {
    d_gen[*d_step] = *d_token;
    (*d_step)++;
    (*d_pos)++;
}

void advance(int* d_pos, int* d_step, int* d_gen, const int* d_token, cudaStream_t st) {
    k_advance<<<1, 1, 0, st>>>(d_pos, d_step, d_gen, d_token);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_sigmoid_gate_mul(float* __restrict__ out, const float* __restrict__ qg,
                                   int head_dim) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int h = e / head_dim, d = e % head_dim;
    float gv = qg[(size_t)h * 2 * head_dim + head_dim + d];
    out[e] *= 1.0f / (1.0f + expf(-gv));
}

void sigmoid_gate_mul(float* out, const float* qg, int n_heads, int head_dim, cudaStream_t st) {
    int n = n_heads * head_dim;
    k_sigmoid_gate_mul<<<(n + 255) / 256, 256, 0, st>>>(out, qg, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- DeltaNet ----------------

__global__ void k_gdn_gates(const float* __restrict__ ar, const float* __restrict__ br,
                            const float* __restrict__ a, const float* __restrict__ dt,
                            float* __restrict__ g, float* __restrict__ b, int n) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n) return;
    float x = ar[h] + dt[h];
    float sp = x > 20.f ? x : log1pf(expf(x)); // softplus
    g[h] = a[h] * sp;
    b[h] = 1.0f / (1.0f + expf(-br[h]));
}

void gdn_gates(const float* alpha_raw, const float* beta_raw, const float* ssm_a,
               const float* ssm_dt, float* g, float* beta_out, int n_heads, cudaStream_t st) {
    k_gdn_gates<<<1, 64, 0, st>>>(alpha_raw, beta_raw, ssm_a, ssm_dt, g, beta_out, n_heads);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_conv_step(const float* __restrict__ rin, float* __restrict__ rout,
                            const float* __restrict__ qkv, const float* __restrict__ w,
                            float* __restrict__ out, int channels) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float* wc = w + (size_t)c * 4; // [channels][4], taps contiguous
    float r0 = rin[c], r1 = rin[channels + c], r2 = rin[2 * channels + c], x = qkv[c];
#if Q27_CONV_OLDEST_FIRST
    float acc = r0 * wc[0] + r1 * wc[1] + r2 * wc[2] + x * wc[3];
#else
    float acc = r0 * wc[3] + r1 * wc[2] + r2 * wc[1] + x * wc[0];
#endif
    out[c] = acc / (1.0f + expf(-acc)); // silu
    rout[c] = r1;
    rout[channels + c] = r2;
    rout[2 * channels + c] = x;
}

void conv_step(const float* ring_src, float* ring_dst, const float* qkv, const float* convw,
               float* out, int channels, cudaStream_t st) {
    k_conv_step<<<(channels + 255) / 256, 256, 0, st>>>(ring_src, ring_dst, qkv, convw, out,
                                                        channels);
    CUDA_CHECK(cudaGetLastError());
}

// Gated delta rule, one token. Block per v-head, 512 threads = (i-tile: 4) x (j: 128).
// The i-reductions (pred = k^T S, o = q^T S) parallelize across 4 tiles of 32 i each;
// consecutive threads hit consecutive j -> coalesced on S[i*SK + j].
__global__ void k_delta_step(const float* __restrict__ Ssrc, float* __restrict__ Sdst,
                             const float* __restrict__ conv,
                             const float* __restrict__ g, const float* __restrict__ beta,
                             float* __restrict__ o) {
    constexpr int SK = 128;
    const int h = blockIdx.x;
    const int j = threadIdx.x & (SK - 1);   // 0..127
    const int it = threadIdx.x >> 7;        // 0..3
    const int i0 = it * 32;
#if Q27_GDN_HEAD_TILE
    const int qk = h % 16;
#else
    const int qk = h / 3;
#endif
    __shared__ float sq[SK], sk[SK], part[4][SK], dj[SK];
    const float scale = rsqrtf((float)SK);
    if (it == 0) {
        sq[j] = conv[qk * SK + j] * scale;
        sk[j] = conv[2048 + qk * SK + j];
    }
    __syncthreads();

    const float decay = expf(g[h]);
    const float* Si = Ssrc + (size_t)h * SK * SK;
    float* So = Sdst + (size_t)h * SK * SK;

    float pred = 0.f;
#pragma unroll 8
    for (int i = i0; i < i0 + 32; i++) {
        float s = Si[i * SK + j] * decay;
        So[i * SK + j] = s;
        pred += sk[i] * s;
    }
    part[it][j] = pred;
    __syncthreads();
    if (it == 0) {
        float p = part[0][j] + part[1][j] + part[2][j] + part[3][j];
        float vj = conv[4096 + h * SK + j];
        dj[j] = beta[h] * (vj - p);
    }
    __syncthreads();

    float d = dj[j];
    float acc = 0.f;
#pragma unroll 8
    for (int i = i0; i < i0 + 32; i++) {
        float s = So[i * SK + j] + sk[i] * d;
        So[i * SK + j] = s;
        acc += sq[i] * s;
    }
    part[it][j] = acc;
    __syncthreads();
    if (it == 0)
        o[h * SK + j] = part[0][j] + part[1][j] + part[2][j] + part[3][j];
}

void delta_step(const float* S_src, float* S_dst, const float* conv_out, const float* g,
                const float* beta, float* o, cudaStream_t st) {
    k_delta_step<<<48, 512, 0, st>>>(S_src, S_dst, conv_out, g, beta, o);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_gated_norm_gdn(const float* __restrict__ o, const float* __restrict__ w,
                                 const float* __restrict__ z, float* __restrict__ out,
                                 int head_dim, float eps) {
    const float* oh = o + (size_t)blockIdx.x * head_dim;
    const float* zh = z + (size_t)blockIdx.x * head_dim;
    float* yh = out + (size_t)blockIdx.x * head_dim;
    __shared__ float sh[128];
    float acc = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) acc += oh[i] * oh[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float zi = zh[i];
        yh[i] = oh[i] * inv * w[i] * (zi / (1.0f + expf(-zi)));
    }
}

void gated_norm_gdn(const float* o, const float* w, const float* z, float* out, int n_heads,
                    int head_dim, float eps, cudaStream_t st) {
    k_gated_norm_gdn<<<n_heads, 128, 0, st>>>(o, w, z, out, head_dim, eps);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- misc ----------------

__global__ void k_add(float* __restrict__ x, const float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
void add_inplace(float* x, const float* y, int n, cudaStream_t st) {
    k_add<<<(n + 255) / 256, 256, 0, st>>>(x, y, n);
    CUDA_CHECK(cudaGetLastError());
}

// Multi-block argmax: pack (orderable float bits, index) into u64, atomicMax.
__device__ __forceinline__ unsigned long long am_pack(float v, int idx) {
    unsigned u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u); // monotonic float->uint map
    return ((unsigned long long)u << 32) | (unsigned)idx;
}

// Exact inverse of am_pack's monotonic map (recovers the packed float value).
// P14: the fused draft top-2 kernel unpacks the top1 pack to feed k_margin's
// pairwise top-2 merge on the same values -- so the fused margin is bit-identical
// to k_margin's.
__device__ __forceinline__ float am_unpack_val(unsigned long long p) {
    unsigned u = (unsigned)(p >> 32);
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u);
}

__global__ void k_argmax_reset(unsigned long long* best) { *best = 0; }

__global__ void k_argmax(const float* __restrict__ x, int n,
                         unsigned long long* __restrict__ best) {
    float bv = -FLT_MAX;
    int bi = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (x[i] > bv) { bv = x[i]; bi = i; }
    unsigned long long p = am_pack(bv, bi);
    __shared__ unsigned long long sh[256];
    sh[threadIdx.x] = p;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] = max(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(best, sh[0]);
}

__global__ void k_argmax_extract(const unsigned long long* best, int* out) {
    *out = (int)(unsigned)(*best & 0xffffffffull);
}

void argmax(const float* x, int n, int* d_out, unsigned long long* d_scratch, cudaStream_t st) {
    k_argmax_reset<<<1, 1, 0, st>>>(d_scratch);
    k_argmax<<<128, 256, 0, st>>>(x, n, d_scratch);
    k_argmax_extract<<<1, 1, 0, st>>>(d_scratch, d_out);
    CUDA_CHECK(cudaGetLastError());
}

// P14: fused top-2 for the draft path -- one full-vocab pass produces the argmax
// token (bit-identical tie semantics to k_argmax: per-thread strict >, packed-u64
// max across threads/blocks) AND the top1-top2 margin (P12 gate signal). Replaces
// k_argmax + k_margin on the draft path; both stay in-tree (verify path / tests).
// Launch grid matches k_argmax (<<<128,256>>>) so the per-thread index partition --
// hence every tie-break -- is identical. Margin is pure selection (no fp arithmetic
// besides the final subtract) so it equals k_margin's value order-independently.
__global__ void k_argmax_top2(const float* __restrict__ x, int n,
                              unsigned long long* __restrict__ blk1,
                              float* __restrict__ blk2) {
    float v1 = -FLT_MAX, v2 = -FLT_MAX;
    int i1 = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float xi = x[i];
        if (xi > v1) { v2 = v1; v1 = xi; i1 = i; }
        else if (xi > v2) { v2 = xi; }
    }
    __shared__ unsigned long long s1[256];
    __shared__ float s2[256];
    s1[threadIdx.x] = am_pack(v1, i1);
    s2[threadIdx.x] = v2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            unsigned long long a1 = s1[threadIdx.x], b1 = s1[threadIdx.x + s];
            float a2 = s2[threadIdx.x], b2 = s2[threadIdx.x + s];
            // top1: packed max, identical lattice to k_argmax's reduction.
            // top2: k_margin's pairwise merge (below) on the unpacked top1 values.
            float a1v = am_unpack_val(a1), b1v = am_unpack_val(b1);
            s1[threadIdx.x] = max(a1, b1);
            s2[threadIdx.x] = (a1v >= b1v) ? fmaxf(a2, b1v) : fmaxf(b2, a1v);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { blk1[blockIdx.x] = s1[0]; blk2[blockIdx.x] = s2[0]; }
}

// Single block (<=256 threads): the same pairwise (top1,top2) merge over the nblk
// per-block partials; writes tok = low 32 bits of the global max pack and
// margin = top1 - top2. Mirrors k_top2's cross-block combine as a tree (not an
// atomic) since we also need the paired top2, but the top1 winner is identical to
// k_argmax's atomicMax result (both are order-independent packed max).
__global__ void k_top2_finalize(const unsigned long long* __restrict__ blk1,
                                const float* __restrict__ blk2, int nblk,
                                int* __restrict__ tok, float* __restrict__ margin_out) {
    __shared__ unsigned long long s1[256];
    __shared__ float s2[256];
    int t = threadIdx.x;
    if (t < nblk) { s1[t] = blk1[t]; s2[t] = blk2[t]; }
    else { s1[t] = am_pack(-FLT_MAX, 0); s2[t] = -FLT_MAX; }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            unsigned long long a1 = s1[t], b1 = s1[t + s];
            float a2 = s2[t], b2 = s2[t + s];
            float a1v = am_unpack_val(a1), b1v = am_unpack_val(b1);
            s1[t] = max(a1, b1);
            s2[t] = (a1v >= b1v) ? fmaxf(a2, b1v) : fmaxf(b2, a1v);
        }
        __syncthreads();
    }
    if (t == 0) {
        tok[0] = (int)(unsigned)(s1[0] & 0xffffffffull);
        margin_out[0] = am_unpack_val(s1[0]) - s2[0];
    }
}

void argmax_margin(const float* x, int n, int* d_tok, float* d_margin,
                   unsigned long long* d_blk1, float* d_blk2, cudaStream_t st) {
    k_argmax_top2<<<128, 256, 0, st>>>(x, n, d_blk1, d_blk2);
    k_top2_finalize<<<1, 128, 0, st>>>(d_blk1, d_blk2, 128, d_tok, d_margin);
    CUDA_CHECK(cudaGetLastError());
}

// Top1-top2 logit margin: the drafter's own confidence, the p_min-gated-depth
// signal (P12). Single block, grid-stride local top-2 -> shared-mem pairwise
// merge -> *out = m1 - m2. Runs on the SAME logits the drafter argmaxed; a
// SEPARATE pass that never touches k_argmax, so the canonical greedy gate is
// unaffected. Graph-capture safe (fixed launch, device out).
__global__ void k_margin(const float* __restrict__ x, int n, float* __restrict__ out) {
    float v1 = -FLT_MAX, v2 = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float xi = x[i];
        if (xi > v1) { v2 = v1; v1 = xi; }
        else if (xi > v2) { v2 = xi; }
    }
    __shared__ float s1[256], s2[256];
    s1[threadIdx.x] = v1; s2[threadIdx.x] = v2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            float a1 = s1[threadIdx.x], a2 = s2[threadIdx.x];
            float b1 = s1[threadIdx.x + s], b2 = s2[threadIdx.x + s];
            s1[threadIdx.x] = fmaxf(a1, b1);
            s2[threadIdx.x] = (a1 >= b1) ? fmaxf(a2, b1) : fmaxf(b2, a1);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = s1[0] - s2[0];
}
void margin(const float* x, int n, float* d_out, cudaStream_t st) {
    k_margin<<<1, 256, 0, st>>>(x, n, d_out);
    CUDA_CHECK(cudaGetLastError());
}

// P7: argmax over grammar-legal tokens only. mask_ids[slot] indexes a
// device-resident bitmask pool (-1 = unconstrained); the ids buffer is
// rewritten by the host between CUDA-graph launches while the pool pointer
// stays fixed (graph-capture safe). Null pool or id -1 walks the exact same
// comparison sequence as k_argmax -- bitwise-identical result, which the
// canonical gate relies on.
__global__ void k_argmax_masked(const float* __restrict__ x, int n,
                                const unsigned* __restrict__ pool, int words,
                                const int* __restrict__ mask_ids, int slot,
                                unsigned long long* __restrict__ best) {
    const unsigned* m = nullptr;
    if (pool != nullptr && mask_ids != nullptr) {
        int id = mask_ids[slot];
        if (id >= 0) m = pool + (size_t)id * words;
    }
    float bv = -FLT_MAX;
    int bi = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if (m && !((m[i >> 5] >> (i & 31)) & 1u)) continue;
        if (x[i] > bv) { bv = x[i]; bi = i; }
    }
    unsigned long long p = am_pack(bv, bi);
    __shared__ unsigned long long sh[256];
    sh[threadIdx.x] = p;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] = max(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(best, sh[0]);
}

void argmax_masked(const float* x, int n, const unsigned* pool, int words, const int* mask_ids,
                   int slot, int* d_out, unsigned long long* d_scratch, cudaStream_t st) {
    k_argmax_reset<<<1, 1, 0, st>>>(d_scratch);
    k_argmax_masked<<<128, 256, 0, st>>>(x, n, pool, words, mask_ids, slot, d_scratch);
    k_argmax_extract<<<1, 1, 0, st>>>(d_scratch, d_out);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- sampling (roadmap #2, temp>0) ----------------
// Greedy is NOT here -- it stays on k_argmax/k_argmax_masked (bitwise). These
// kernels run only when a request sets temperature>0.

// Philox4x32-10 (Salmon et al. 2011): counter-based, stateless. One uniform
// in (0,1) from counter word 0. key = 64-bit seed; counter = (c0,c1,c2,0).
__device__ __forceinline__ unsigned am_mulhi(unsigned a, unsigned b, unsigned& lo) {
    unsigned long long p = (unsigned long long)a * b;
    lo = (unsigned)p;
    return (unsigned)(p >> 32);
}
__device__ __forceinline__ float philox_uniform(unsigned long long seed, unsigned c0,
                                                 unsigned c1, unsigned c2) {
    unsigned k0 = (unsigned)seed, k1 = (unsigned)(seed >> 32);
    unsigned x0 = c0, x1 = c1, x2 = c2, x3 = 0u;
    const unsigned M0 = 0xD2511F53u, M1 = 0xCD9E8D57u, W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
#pragma unroll
    for (int r = 0; r < 10; r++) {
        unsigned lo0, lo1;
        unsigned hi0 = am_mulhi(M0, x0, lo0);
        unsigned hi1 = am_mulhi(M1, x2, lo1);
        unsigned n0 = hi1 ^ x1 ^ k0, n1 = lo1, n2 = hi0 ^ x3 ^ k1, n3 = lo0;
        x0 = n0; x1 = n1; x2 = n2; x3 = n3;
        k0 += W0; k1 += W1;
    }
    // map to (0,1): the top ~128 x0 values round to 2^32 in fp32, giving u==1 ->
    // -log(-log(u)) = +inf (CUDA-review #2). Clamp to the largest float below 1.
    float u = ((float)x0 + 0.5f) * (1.0f / 4294967296.0f);
    return fminf(u, 0x1.fffffep-1f);
}

// Single block: max M, logsumexp logZ at inv_temp, and the top-p logit
// threshold via a fixed 12-iteration bisection on a prob cutoff (no sort, no
// atomics -> deterministic, and graph-capturable at fixed geometry). Reads
// inv_temp/top_p from the device param block (graph-fixed pointer). Writes
// out[0]=logit_thresh, out[1]=M, out[2]=logZ. thresh clamped <= M so the argmax
// token is always in the nucleus (guards degenerate/tiny top_p).
__global__ void k_nucleus_d(const float* __restrict__ x, int n, const SampleParams* __restrict__ sp,
                            float* __restrict__ out) {
    const float inv_temp = sp->inv_temp, top_p = sp->top_p;
    __shared__ float sh[1024];
    __shared__ float s_M, s_logZ, s_lo, s_hi, s_thresh;
    const int t = threadIdx.x, B = blockDim.x;
    float v = -FLT_MAX;
    for (int i = t; i < n; i += B) v = fmaxf(v, x[i]);
    sh[t] = v; __syncthreads();
    for (int s = B / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] = fmaxf(sh[t], sh[t + s]);
        __syncthreads();
    }
    if (t == 0) s_M = sh[0];
    __syncthreads();
    const float M = s_M;
    float se = 0.f;
    for (int i = t; i < n; i += B) se += expf(inv_temp * (x[i] - M));
    sh[t] = se; __syncthreads();
    for (int s = B / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    // Bisect on the LOGIT threshold directly (unbounded below), not a prob cutoff
    // in [0,1]: the old 12-step tau bisection could not resolve cutoffs < 2^-12, so
    // a diffuse distribution fell through to thresh=-FLT_MAX = full vocab (CUDA
    // review #3). The window below M must cover all non-negligible SCALED mass:
    // weights go as exp(inv_temp * (x - M)), so 40 raw logits only span 40*T_inv
    // scaled nats -- at T=10 a fixed 40-raw window kept ~8% of the nucleus (review
    // 2026-07-09 P1 #4). Scale the window to 40 scaled nats (relative weight
    // e^-40 ~ 4e-18: negligible at any vocab size); T <= 1 keeps the original 40
    // exactly, so low-temp behavior is bitwise-unchanged. Bisection resolution in
    // scaled space stays constant: 40 / 2^16 nats.
    if (t == 0) {
        const float win = 40.0f * fmaxf(1.0f, 1.0f / inv_temp);
        s_logZ = logf(sh[0]); s_lo = M - win; s_hi = M;
    }
    __syncthreads();
    const float logZ = s_logZ;
    if (top_p < 1.0f) {
        for (int it = 0; it < 16; it++) {
            const float thr = 0.5f * (s_lo + s_hi); // s_lo/s_hi settled by prior __syncthreads
            float mass = 0.f;
            for (int i = t; i < n; i += B)
                if (x[i] >= thr) mass += expf(inv_temp * (x[i] - M) - logZ);
            sh[t] = mass; __syncthreads();
            for (int s = B / 2; s > 0; s >>= 1) {
                if (t < s) sh[t] += sh[t + s];
                __syncthreads();
            }
            if (t == 0) { if (sh[0] >= top_p) s_lo = thr; else s_hi = thr; } // mass decreasing in thr
            __syncthreads();
        }
    }
    if (t == 0) {
        float thresh = (top_p >= 1.0f) ? -FLT_MAX : s_lo; // s_lo is now the logit threshold
        s_thresh = fminf(thresh, M); // argmax token always in nucleus
    }
    __syncthreads();
    // out[3] = nucleus mass = sum_{x_i >= thresh} softmax_full(i) (Phase 2 accept
    // test needs the RENORMALIZED served prob p(dr)=softmax_full(dr)/mass, not
    // softmax_full(dr); mass==1 when top_p>=1 so the plain path is unaffected).
    // One extra grid-stride pass over the just-fixed threshold (argmax cost class).
    const float thr = s_thresh;
    float ms = 0.f;
    for (int i = t; i < n; i += B)
        if (x[i] >= thr) ms += expf(inv_temp * (x[i] - M) - logZ);
    sh[t] = ms; __syncthreads();
    for (int s = B / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    if (t == 0) {
        out[0] = thr;
        out[1] = M;
        out[2] = logZ;
        out[3] = sh[0]; // nucleus mass
    }
}

// Gumbel-max over the nucleus: argmax_i (inv_temp*x_i + G_i), G_i from Philox,
// restricted to x_i >= nuc[0]. Reads inv_temp/seed from the param block and the
// key position from *d_pos (device). Same am_pack+atomicMax reduction as
// k_argmax (order-independent -> deterministic token).
__global__ void k_gumbel_d(const float* __restrict__ x, int n, const SampleParams* __restrict__ sp,
                           const float* __restrict__ nuc, const int* __restrict__ d_pos,
                           unsigned kind, unsigned long long* __restrict__ best) {
    const float inv_temp = sp->inv_temp;
    const unsigned long long seed = sp->seed;
    const unsigned pos = (unsigned)*d_pos;
    const float thresh = nuc[0];
    float bv = -FLT_MAX;
    int bi = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if (x[i] < thresh) continue;
        float u = philox_uniform(seed, pos, kind, (unsigned)i);
        float g = -logf(-logf(u));
        float key = inv_temp * x[i] + g;
        if (key > bv) { bv = key; bi = i; }
    }
    unsigned long long p = am_pack(bv, bi);
    __shared__ unsigned long long sh[256];
    sh[threadIdx.x] = p;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] = max(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(best, sh[0]);
}

void sample_g(const float* logits, int n, const SampleParams* d_sp, float* d_nuc,
              const int* d_pos, unsigned draw_kind, int* d_out, unsigned long long* d_scratch,
              cudaStream_t st) {
    k_nucleus_d<<<1, 1024, 0, st>>>(logits, n, d_sp, d_nuc);
    k_argmax_reset<<<1, 1, 0, st>>>(d_scratch);
    k_gumbel_d<<<128, 256, 0, st>>>(logits, n, d_sp, d_nuc, d_pos, draw_kind, d_scratch);
    k_argmax_extract<<<1, 1, 0, st>>>(d_scratch, d_out);
    CUDA_CHECK(cudaGetLastError());
}

// nucleus stats for one lane (Phase 2 spec path calls this 5x, one per verify
// lane, into d_nuc + lane*4). Same kernel the plain path uses -> spec and plain
// sample the identical served distribution (the spec==non-spec gate relies on it).
void nucleus(const float* logits, int n, const SampleParams* d_sp, float* d_nuc,
             cudaStream_t st) {
    k_nucleus_d<<<1, 1024, 0, st>>>(logits, n, d_sp, d_nuc);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- spec rejection sampling (roadmap #2 Phase 2, temp>0) --------
// Greedy spec (k_finish_round equality chain) is untouched; these run only in
// the sampled 2nd graph set. Philox draw kinds (counter word c1) are disjoint
// so no two draw sites ever share a counter: 0/1 = plain eager/graph (Phase 1),
// 2 = spec accept uniforms, 3 = spec stop Gumbel. All key on *d_P (the committed
// position at round start; strictly increasing -> no cross-round collision).
static constexpr unsigned KIND_SPEC_ACCEPT = 2u;
static constexpr unsigned KIND_SPEC_STOP = 3u;

// Serial accept walk (1 thread, like k_prep_round). For lane k (0..3), the draft
// dr_{k+1} is accepted with prob p_served(dr) = softmax_full(dr)/mass restricted
// to the nucleus (0 if the draft fell outside it). q is a delta at the greedy
// draft, so min(1,p/q) = p (rejection sampling, Leviathan/Chen 2023). First
// reject stops the chain; all-accept leaves stop_lane=4 for the free bonus draw.
// out[3] = {n, stop_lane, exclude_token}. *cap forces n=1 (in-grammar; Phase-3
// hook, never set on the sampled path today since tools are off under sampling).
__global__ void k_spec_accept(const float* __restrict__ logits2,
                              const float* __restrict__ nuc5, const int* __restrict__ dr1p,
                              const int* __restrict__ dr2p, const int* __restrict__ dr3p,
                              const int* __restrict__ dr4p, const SampleParams* __restrict__ sp,
                              const int* __restrict__ dP, const int* __restrict__ cap,
                              int max_draft, int vocab, int* __restrict__ out) {
    const float inv_temp = sp->inv_temp;
    const unsigned long long seed = sp->seed;
    const unsigned pos = (unsigned)*dP;
    const int dr[4] = {*dr1p, *dr2p, *dr3p, *dr4p};
    // P14 gate: width-W verify walks max_draft = W-1 drafts. stop_lane inits to
    // max_draft so all-accept commits max_draft drafts + the bonus lane (n=max_draft+1).
    int stop_lane = max_draft, exclude = -1;
    if (*cap) {
        stop_lane = 0; // n=1: commit only the pending, resample lane 0 fresh
    } else {
        for (int k = 0; k < max_draft; k++) {
            const float* nl = nuc5 + (size_t)k * 4;
            const float thr = nl[0], M = nl[1], logZ = nl[2], mass = nl[3];
            const int d = dr[k];
            const float xd = logits2[(size_t)k * vocab + d];
            const float p = (xd >= thr) ? expf(inv_temp * (xd - M) - logZ) / mass : 0.f;
            const float u = philox_uniform(seed, pos, KIND_SPEC_ACCEPT, (unsigned)k);
            if (u < p) continue;        // accept draft k, extend the chain
            stop_lane = k; exclude = d; // first reject: resample this lane sans d
            break;
        }
    }
    out[0] = stop_lane + 1; // n committed (pending + accepted drafts)
    out[1] = stop_lane;
    out[2] = exclude;
}
void spec_accept(const float* logits2, const float* nuc5, const int* dr1, const int* dr2,
                 const int* dr3, const int* dr4, const SampleParams* d_sp, const int* d_P,
                 const int* cap, int max_draft, int vocab, int* d_spec, cudaStream_t st) {
    k_spec_accept<<<1, 1, 0, st>>>(logits2, nuc5, dr1, dr2, dr3, dr4, d_sp, d_P, cap, max_draft,
                                   vocab, d_spec);
    CUDA_CHECK(cudaGetLastError());
}

// Gumbel-max over the STOP lane's nucleus, excluding the rejected draft: the
// residual resample norm(max(0,p-q)). Clone of k_gumbel_d with the logit base
// (logits2 + stop*vocab) and nucleus (nuc5 + stop*4) selected on-device from
// d_spec -- fixed base pointers, graph-safe. stop_lane==4 => exclude==-1, a
// plain nucleus draw (the bonus). Same order-independent am_pack+atomicMax as
// k_argmax -> deterministic token.
__global__ void k_sample_stop(const float* __restrict__ logits2, const float* __restrict__ nuc5,
                              const int* __restrict__ spec, const SampleParams* __restrict__ sp,
                              const int* __restrict__ dP, int vocab,
                              unsigned long long* __restrict__ best) {
    const int stop_lane = spec[1];
    const int exclude = spec[2];
    const float inv_temp = sp->inv_temp;
    const unsigned long long seed = sp->seed;
    const unsigned pos = (unsigned)*dP;
    const float* x = logits2 + (size_t)stop_lane * vocab;
    const float thresh = nuc5[(size_t)stop_lane * 4];
    float bv = -FLT_MAX;
    int bi = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab; i += gridDim.x * blockDim.x) {
        if (x[i] < thresh || i == exclude) continue;
        float u = philox_uniform(seed, pos, KIND_SPEC_STOP, (unsigned)i);
        float g = -logf(-logf(u));
        float key = inv_temp * x[i] + g;
        if (key > bv) { bv = key; bi = i; }
    }
    unsigned long long p = am_pack(bv, bi);
    __shared__ unsigned long long sh[256];
    sh[threadIdx.x] = p;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] = max(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(best, sh[0]);
}
// draws the new pending token into d_out (= d_token), mirroring sample_g's tail.
void sample_stop(const float* logits2, const float* nuc5, const int* d_spec,
                 const SampleParams* d_sp, const int* d_P, int vocab, int* d_out,
                 unsigned long long* d_scratch, cudaStream_t st) {
    k_argmax_reset<<<1, 1, 0, st>>>(d_scratch);
    k_sample_stop<<<128, 256, 0, st>>>(logits2, nuc5, d_spec, d_sp, d_P, vocab, d_scratch);
    k_argmax_extract<<<1, 1, 0, st>>>(d_scratch, d_out);
    CUDA_CHECK(cudaGetLastError());
}

// Finish bookkeeping keyed on n from k_spec_accept (mirror of k_finish_round,
// which keys on the equality chain). h_next = the stop lane's hidden x1[n-1]
// (the position that predicted the new pending nt); *dP += n; outcome carries
// n, the drafts, and nt (already in dtok from k_sample_stop). outcome[1] (the
// pre-round pending t1) was snapshotted by k_prep_round.
__global__ void k_finish_sampled(int* __restrict__ dP, const int* __restrict__ dtok,
                                 const int* __restrict__ spec, const int* __restrict__ dr1p,
                                 const int* __restrict__ dr2p, const int* __restrict__ dr3p,
                                 const int* __restrict__ dr4p, const float* __restrict__ x1a,
                                 const float* __restrict__ x1b, const float* __restrict__ x1c,
                                 const float* __restrict__ x1d, const float* __restrict__ x1e,
                                 float* __restrict__ h_next, int* __restrict__ outcome,
                                 int n_embd) {
    int n = spec[0];
    const float* src = n == 5 ? x1e : n == 4 ? x1d : n == 3 ? x1c : n == 2 ? x1b : x1a;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_embd; i += gridDim.x * blockDim.x)
        h_next[i] = src[i];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *dP += n;
        outcome[0] = n;
        outcome[6] = *dtok; // new pending token nt
        outcome[2] = *dr1p;
        outcome[3] = *dr2p;
        outcome[4] = *dr3p;
        outcome[5] = *dr4p;
    }
}
void finish_sampled(int* d_P, const int* d_token, const int* d_spec, const int* dr1,
                    const int* dr2, const int* dr3, const int* dr4, const float* x1a,
                    const float* x1b, const float* x1c, const float* x1d, const float* x1e,
                    float* h_next, int* outcome, int n_embd, cudaStream_t st) {
    k_finish_sampled<<<4, 256, 0, st>>>(d_P, d_token, d_spec, dr1, dr2, dr3, dr4, x1a, x1b, x1c,
                                        x1d, x1e, h_next, outcome, n_embd);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- teacher-forced NLL (P0 quality gate) ----------------

// nll[r] = logsumexp(logits[r, :]) - logits[r, tgt[r]] over a [nrows, vocab]
// row-major logit matrix. Block per row, two passes (max, then sum-exp).
__global__ void k_nll_rows(const float* __restrict__ logits, const int* __restrict__ tgt,
                           float* __restrict__ nll, int64_t vocab) {
    const float* row = logits + (size_t)blockIdx.x * vocab;
    __shared__ float sh[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    float mx = -FLT_MAX;
    for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) mx = fmaxf(mx, row[i]);
    for (int off = 16; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
    if (lane == 0) sh[wid] = mx;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (unsigned)nw ? sh[threadIdx.x] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
        if (threadIdx.x == 0) sh[0] = v;
    }
    __syncthreads();
    mx = sh[0];
    __syncthreads();
    float se = 0.f;
    for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) se += expf(row[i] - mx);
    for (int off = 16; off > 0; off >>= 1) se += __shfl_xor_sync(0xffffffff, se, off);
    if (lane == 0) sh[wid] = se;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int w = 0; w < nw; w++) s += sh[w];
        nll[blockIdx.x] = logf(s) + mx - row[tgt[blockIdx.x]];
    }
}

void nll_rows(const float* logits, const int* tgt, float* nll, int nrows, int64_t vocab,
              cudaStream_t st) {
    k_nll_rows<<<nrows, 1024, 0, st>>>(logits, tgt, nll, vocab);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace q27k
