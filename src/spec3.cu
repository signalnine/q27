#include <cfloat>

#include "cuda_common.h"
#include "spec3.cuh"

namespace q27k {

__device__ __forceinline__ float wred(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__global__ void k_l2norm3(P3 xp, int head_dim, float eps) {
    float* xh = xp.p[blockIdx.y] + (size_t)blockIdx.x * head_dim;
    __shared__ float sh[128];
    float acc = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) acc += xh[i] * xh[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(fmaxf(sh[0], eps * eps));
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) xh[i] *= inv;
}
void l2norm3(P3 x, int n_heads, int head_dim, float eps, cudaStream_t st) {
    dim3 g(n_heads, 3);
    k_l2norm3<<<g, 128, 0, st>>>(x, head_dim, eps);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_gemv_f16_3(const __half* __restrict__ W, CP3 xp, P3 yp, int64_t cols) {
    const float* x = xp.p[blockIdx.y];
    const __half* wr = W + (size_t)blockIdx.x * cols;
    float acc = 0.f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
        acc += __half2float(wr[c]) * x[c];
    __shared__ float sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) yp.p[blockIdx.y][blockIdx.x] = sh[0];
}
void gemv_f16_3(const __half* W, CP3 x, P3 y, int64_t rows, int64_t cols, cudaStream_t st) {
    dim3 g((unsigned)rows, 3);
    k_gemv_f16_3<<<g, 256, 0, st>>>(W, x, y, cols);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_gdn_gates3(CP3 ar, CP3 br, const float* __restrict__ a,
                             const float* __restrict__ dt, P3 g, P3 b, int n) {
    int h = threadIdx.x;
    if (h >= n) return;
    int t = blockIdx.x;
    float x = ar.p[t][h] + dt[h];
    float sp = x > 20.f ? x : log1pf(expf(x));
    g.p[t][h] = a[h] * sp;
    b.p[t][h] = 1.0f / (1.0f + expf(-br.p[t][h]));
}
void gdn_gates3(CP3 ar, CP3 br, const float* a, const float* dt, P3 g, P3 b, int n,
                cudaStream_t st) {
    k_gdn_gates3<<<3, 64, 0, st>>>(ar, br, a, dt, g, b, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_gated_norm3(CP3 op, const float* __restrict__ w, CP3 zp, P3 outp, int head_dim,
                              float eps) {
    const int t = blockIdx.y;
    const float* oh = op.p[t] + (size_t)blockIdx.x * head_dim;
    const float* zh = zp.p[t] + (size_t)blockIdx.x * head_dim;
    float* yh = outp.p[t] + (size_t)blockIdx.x * head_dim;
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
void gated_norm3(CP3 o, const float* w, CP3 z, P3 out, int n_heads, int head_dim, float eps,
                 cudaStream_t st) {
    dim3 g(n_heads, 3);
    k_gated_norm3<<<g, 128, 0, st>>>(o, w, z, out, head_dim, eps);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_sigmoid_gate3(P3 outp, CP3 qgp, int head_dim, int n) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    int t = blockIdx.y;
    int h = e / head_dim, d = e % head_dim;
    float gv = qgp.p[t][(size_t)h * 2 * head_dim + head_dim + d];
    outp.p[t][e] *= 1.0f / (1.0f + expf(-gv));
}
void sigmoid_gate3(P3 out, CP3 qg, int n_heads, int head_dim, cudaStream_t st) {
    int n = n_heads * head_dim;
    dim3 g((n + 255) / 256, 3);
    k_sigmoid_gate3<<<g, 256, 0, st>>>(out, qg, head_dim, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_rope3(P3 xp, int head_dim, int n_rot, int stride, IP3 pos, float freq_base) {
    const int t = blockIdx.y;
    float* xh = xp.p[t] + (size_t)blockIdx.x * stride;
    int d = threadIdx.x;
    if (d >= n_rot / 2) return;
    float theta = (float)(*pos.p[t]) * powf(freq_base, -2.0f * d / n_rot);
    float c = cosf(theta), s = sinf(theta);
    float x0 = xh[d], x1 = xh[d + n_rot / 2];
    xh[d] = x0 * c - x1 * s;
    xh[d + n_rot / 2] = x0 * s + x1 * c;
}
void rope3(P3 x, int n_heads, int head_dim, int n_rot, int stride, IP3 pos, float freq_base,
           cudaStream_t st) {
    dim3 g(n_heads, 3);
    k_rope3<<<g, 32, 0, st>>>(x, head_dim, n_rot, stride, pos, freq_base);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_kv_store3(CP3 kp, CP3 vp, float* __restrict__ kc, float* __restrict__ vc,
                            IP3 pos, int rowlen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowlen) return;
    int t = blockIdx.y;
    size_t off = (size_t)(*pos.p[t]) * rowlen + i;
    kc[off] = kp.p[t][i];
    vc[off] = vp.p[t][i];
}
void kv_store3(CP3 k, CP3 v, float* kc, float* vc, IP3 pos, int rowlen, cudaStream_t st) {
    dim3 g((rowlen + 255) / 256, 3);
    k_kv_store3<<<g, 256, 0, st>>>(k, v, kc, vc, pos, rowlen);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_attn_decode3(CP3 qp, int q_stride, const float* __restrict__ kc,
                               const float* __restrict__ vc, P3 outp,
                               float* __restrict__ scratch, IP3 pos, int max_ctx, int n_kv_heads,
                               int head_dim, float scale) {
    const int t = blockIdx.y;
    const int seq_len = *pos.p[t] + 1;
    const int h = blockIdx.x;
    const int kvh = h / (gridDim.x / n_kv_heads);
    const float* qh = qp.p[t] + (size_t)h * q_stride;
    float* sc = scratch + ((size_t)t * gridDim.x + h) * max_ctx;
    __shared__ float sh[256];
    __shared__ float s_max, s_sum;

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

    float inv = 1.0f / s_sum;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int p = 0; p < seq_len; p++)
            acc += sc[p] * vc[((size_t)p * n_kv_heads + kvh) * head_dim + d];
        outp.p[t][(size_t)h * head_dim + d] = acc * inv;
    }
}
void attn_decode3(CP3 q, int q_stride, const float* kc, const float* vc, P3 out, float* scratch,
                  IP3 pos, int max_ctx, int n_q_heads, int n_kv_heads, int head_dim, float scale,
                  cudaStream_t st) {
    dim3 g(n_q_heads, 3);
    k_attn_decode3<<<g, 256, 0, st>>>(q, q_stride, kc, vc, out, scratch, pos, max_ctx, n_kv_heads,
                                      head_dim, scale);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_embed3(const int8_t* __restrict__ W, const __half* __restrict__ S, IP3 tok,
                         int64_t cols, P3 outp) {
    const int t = blockIdx.y;
    int64_t row = *tok.p[t];
    const int8_t* wr = W + row * cols;
    const __half* sr = S + row * (cols / 128);
    for (int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; c < cols;
         c += (int64_t)gridDim.x * blockDim.x)
        outp.p[t][c] = (float)wr[c] * __half2float(sr[c / 128]);
}
void embed3(const int8_t* W, const __half* S, IP3 tok, int64_t cols, P3 out, cudaStream_t st) {
    dim3 g(8, 3);
    k_embed3<<<g, 256, 0, st>>>(W, S, tok, cols, out);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_prep_round(const int* __restrict__ dP, const int* __restrict__ dtok,
                             int* pa, int* pb, int* pc, int* pm, int* pm2, int* outcome) {
    int P = *dP;
    *pa = P + 1;
    *pb = P + 2;
    *pc = P + 3;
    *pm = P + 1;
    *pm2 = P + 2;
    outcome[1] = *dtok; // t1 snapshot (pre-round)
}
void prep_round(const int* d_P, const int* d_token, int* pos_a, int* pos_b, int* pos_c,
                int* pos_m, int* pos_m2, int* outcome, cudaStream_t st) {
    k_prep_round<<<1, 1, 0, st>>>(d_P, d_token, pos_a, pos_b, pos_c, pos_m, pos_m2, outcome);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_finish_round(int* __restrict__ dP, int* __restrict__ dtok,
                               const int* __restrict__ dr1p, const int* __restrict__ dr2p,
                               const int* __restrict__ vap, const int* __restrict__ vbp,
                               const int* __restrict__ vcp, const float* __restrict__ x1a,
                               const float* __restrict__ x1b, const float* __restrict__ x1c,
                               float* __restrict__ h_next, int* __restrict__ outcome,
                               int n_embd) {
    int dr1 = *dr1p, dr2 = *dr2p, va = *vap, vb = *vbp, vc = *vcp;
    int n = 1 + (va == dr1 ? 1 : 0) + ((va == dr1 && vb == dr2) ? 1 : 0);
    const float* src = n == 3 ? x1c : n == 2 ? x1b : x1a;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_embd; i += gridDim.x * blockDim.x)
        h_next[i] = src[i];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *dtok = n == 3 ? vc : n == 2 ? vb : va;
        *dP += n;
        outcome[0] = n;
        outcome[2] = dr1;
        outcome[3] = dr2;
    }
}
void finish_round(int* d_P, int* d_token, const int* d_draft, const int* d_draft2, const int* va,
                  const int* vb, const int* vc, const float* x1a, const float* x1b,
                  const float* x1c, float* h_next, int* outcome, int n_embd, cudaStream_t st) {
    k_finish_round<<<4, 256, 0, st>>>(d_P, d_token, d_draft, d_draft2, va, vb, vc, x1a, x1b, x1c,
                                      h_next, outcome, n_embd);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace q27k
