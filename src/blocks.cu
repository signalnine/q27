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

__global__ void k_kv_store(const float* __restrict__ kbuf, const float* __restrict__ vbuf,
                           __half* __restrict__ kc, __half* __restrict__ vc,
                           const int* __restrict__ d_pos, int rowlen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowlen) return;
    size_t off = (size_t)(*d_pos) * rowlen + i;
    kc[off] = __float2half_rn(kbuf[i]);
    vc[off] = __float2half_rn(vbuf[i]);
}

void kv_store(const float* kbuf, const float* vbuf, __half* kcache, __half* vcache,
              const int* d_pos, int rowlen, cudaStream_t st) {
    k_kv_store<<<(rowlen + 255) / 256, 256, 0, st>>>(kbuf, vbuf, kcache, vcache, d_pos, rowlen);
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

} // namespace q27k
