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
#if Q27_L2NORM_EPS_ADD
    float inv = rsqrtf(sh[0] + eps);
#else
    float inv = rsqrtf(fmaxf(sh[0], eps));
#endif
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) xh[i] *= inv;
}

void l2norm_heads(float* x, int n_heads, int head_dim, float eps, cudaStream_t st) {
    k_l2norm_heads<<<n_heads, 128, 0, st>>>(x, head_dim, eps);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- rope ----------------

__global__ void k_rope_neox(float* __restrict__ x, int head_dim, int n_rot, int stride, int pos,
                            float freq_base) {
    float* xh = x + (size_t)blockIdx.x * stride;
    int d = threadIdx.x; // pair index 0..n_rot/2-1
    if (d >= n_rot / 2) return;
    float theta = pos * powf(freq_base, -2.0f * d / n_rot);
    float c = cosf(theta), s = sinf(theta);
    float x0 = xh[d], x1 = xh[d + n_rot / 2];
    xh[d] = x0 * c - x1 * s;
    xh[d + n_rot / 2] = x0 * s + x1 * c;
}

void rope_neox_partial(float* x, int n_heads, int head_dim, int n_rot, int stride, int pos,
                       float freq_base, cudaStream_t st) {
    k_rope_neox<<<n_heads, 32, 0, st>>>(x, head_dim, n_rot, stride, pos, freq_base);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- attention (decode, reference) ----------------

__global__ void k_attn_decode(const float* __restrict__ q, int q_stride,
                              const float* __restrict__ kc, const float* __restrict__ vc,
                              float* __restrict__ out, float* __restrict__ scratch, int seq_len,
                              int n_kv_heads, int head_dim, float scale) {
    const int h = blockIdx.x;
    const int kvh = h / (gridDim.x / n_kv_heads);
    const float* qh = q + (size_t)h * q_stride;
    float* sc = scratch + (size_t)h * seq_len;
    __shared__ float sh[256];
    __shared__ float s_max, s_sum;

    // scores
    float lmax = -FLT_MAX;
    for (int p = threadIdx.x; p < seq_len; p += blockDim.x) {
        const float* kp = kc + ((size_t)p * n_kv_heads + kvh) * head_dim;
        float dot = 0.f;
        for (int i = 0; i < head_dim; i++) dot += qh[i] * kp[i];
        float sv = dot * scale;
        sc[p] = sv;
        lmax = fmaxf(lmax, sv);
    }
    sh[threadIdx.x] = lmax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) s_max = sh[0];
    __syncthreads();

    // softmax weights
    float lsum = 0.f;
    for (int p = threadIdx.x; p < seq_len; p += blockDim.x) {
        float e = expf(sc[p] - s_max);
        sc[p] = e;
        lsum += e;
    }
    sh[threadIdx.x] = lsum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) s_sum = sh[0];
    __syncthreads();

    // weighted V accumulation: thread d walks all positions
    float inv = 1.0f / s_sum;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < seq_len; p++)
            acc += sc[p] * vc[((size_t)p * n_kv_heads + kvh) * head_dim + d];
        out[(size_t)h * head_dim + d] = acc * inv;
    }
}

void attn_decode(const float* q, int q_stride, const float* kcache, const float* vcache,
                 float* out, float* scratch, int seq_len, int n_q_heads, int n_kv_heads,
                 int head_dim, float scale, cudaStream_t st) {
    k_attn_decode<<<n_q_heads, 256, 0, st>>>(q, q_stride, kcache, vcache, out, scratch, seq_len,
                                             n_kv_heads, head_dim, scale);
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

__global__ void k_conv_step(float* __restrict__ ring, const float* __restrict__ qkv,
                            const float* __restrict__ w, float* __restrict__ out, int channels) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float* wc = w + (size_t)c * 4; // [channels][4], taps contiguous
    float r0 = ring[c], r1 = ring[channels + c], r2 = ring[2 * channels + c], x = qkv[c];
#if Q27_CONV_OLDEST_FIRST
    float acc = r0 * wc[0] + r1 * wc[1] + r2 * wc[2] + x * wc[3];
#else
    float acc = r0 * wc[3] + r1 * wc[2] + r2 * wc[1] + x * wc[0];
#endif
    out[c] = acc / (1.0f + expf(-acc)); // silu
    ring[c] = r1;
    ring[channels + c] = r2;
    ring[2 * channels + c] = x;
}

void conv_step(float* ring, const float* qkv, const float* convw, float* out, int channels,
               cudaStream_t st) {
    k_conv_step<<<(channels + 255) / 256, 256, 0, st>>>(ring, qkv, convw, out, channels);
    CUDA_CHECK(cudaGetLastError());
}

// Gated delta rule, one token, one block per v-head, thread = v-index j.
//   S = S*exp(g); pred_j = sum_i k_i S_ij; d_j = beta*(v_j - pred_j);
//   S_ij += k_i d_j; o_j = sum_i q_i S_ij      (q pre-scaled by 1/sqrt(128))
__global__ void k_delta_step(float* __restrict__ S, const float* __restrict__ conv,
                             const float* __restrict__ g, const float* __restrict__ beta,
                             float* __restrict__ o) {
    constexpr int SK = 128;
    const int h = blockIdx.x;        // v-head 0..47
    const int j = threadIdx.x;       // v-index 0..127
#if Q27_GDN_HEAD_TILE
    const int qk = h % 16;
#else
    const int qk = h / 3;
#endif
    __shared__ float sq[SK], sk[SK];
    const float scale = rsqrtf((float)SK);
    if (j < SK) {
        sq[j] = conv[qk * SK + j] * scale;         // q block at offset 0
        sk[j] = conv[2048 + qk * SK + j];          // k block at offset 16*128
    }
    __syncthreads();

    const float decay = expf(g[h]);
    const float b = beta[h];
    const float vj = conv[4096 + h * SK + j];      // v block at offset 2*16*128
    float* Sh = S + (size_t)h * SK * SK;           // S[i,j] at Sh[j*SK + i]

    float pred = 0.f;
    float* col = Sh + (size_t)j * SK;
    for (int i = 0; i < SK; i++) {
        float s = col[i] * decay;
        col[i] = s;
        pred += sk[i] * s;
    }
    float d = b * (vj - pred);
    float acc = 0.f;
    for (int i = 0; i < SK; i++) {
        float s = col[i] + sk[i] * d;
        col[i] = s;
        acc += sq[i] * s;
    }
    o[h * SK + j] = acc;
}

void delta_step(float* S, const float* conv_out, const float* g, const float* beta, float* o,
                cudaStream_t st) {
    k_delta_step<<<48, 128, 0, st>>>(S, conv_out, g, beta, o);
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

__global__ void k_argmax(const float* __restrict__ x, int n, int* __restrict__ out) {
    __shared__ float sv[256];
    __shared__ int si[256];
    float best = -FLT_MAX;
    int bi = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (x[i] > best) { best = x[i]; bi = i; }
    sv[threadIdx.x] = best;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s && sv[threadIdx.x + s] > sv[threadIdx.x]) {
            sv[threadIdx.x] = sv[threadIdx.x + s];
            si[threadIdx.x] = si[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // single-block final reduce via atomics on (value, index) packed compare
        // simple approach: gridDim.x == 1 required
        *out = si[0];
    }
}
void argmax(const float* x, int n, int* d_out, cudaStream_t st) {
    k_argmax<<<1, 256, 0, st>>>(x, n, d_out); // single block: 248320/256 ~ 970 iters/thread, fine
    CUDA_CHECK(cudaGetLastError());
}

} // namespace q27k
