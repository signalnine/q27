// gdn_chunk_bench -- P0 day-1 gate for the GDN chunked-scan plan
// (docs/plans/2026-07-10-gdn-chunk.md). Serial W-launch chains
// (k_conv_step / k_delta_step forked verbatim from src/blocks.cu) vs
// one-launch chunk kernels that loop W tokens in-kernel with IDENTICAL
// per-step arithmetic. Gates: bitwise identity of all W role states +
// outputs, and wall-clock ratio (GO >= 2.5x on delta at W=8).
// Usage: gdn_chunk_bench   (no model needed; synthetic tensors)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err__ = (x);                                               \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static constexpr int GDN_CH = 10240, GDN_HEADS = 48, SK = 128;

// ---- serial kernels, forked verbatim from src/blocks.cu ----
__global__ void k_conv_step(const float* __restrict__ rin, float* __restrict__ rout,
                            const float* __restrict__ qkv, const float* __restrict__ w,
                            float* __restrict__ out, int channels) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float* wc = w + (size_t)c * 4;
    float r0 = rin[c], r1 = rin[channels + c], r2 = rin[2 * channels + c], x = qkv[c];
    float acc = r0 * wc[3] + r1 * wc[2] + r2 * wc[1] + x * wc[0];
    out[c] = acc / (1.0f + expf(-acc));
    rout[c] = r1;
    rout[channels + c] = r2;
    rout[2 * channels + c] = x;
}

__global__ void k_delta_step(const float* __restrict__ Ssrc, float* __restrict__ Sdst,
                             const float* __restrict__ conv,
                             const float* __restrict__ g, const float* __restrict__ beta,
                             float* __restrict__ o) {
    const int h = blockIdx.x;
    const int j = threadIdx.x & (SK - 1);
    const int it = threadIdx.x >> 7;
    const int i0 = it * 32;
    const int qk = h / 3;
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

// ---- chunk kernels ----
// conv: fully parallel over (token, channel); ring role t = inputs [t-2..t]
__global__ void k_conv_chunk(const float* __restrict__ rin, float* __restrict__ routs,
                             const float* __restrict__ qkvs, const float* __restrict__ w,
                             float* __restrict__ outs, int channels, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= channels * W) return;
    int t = idx / channels, c = idx % channels;
    const float* wc = w + (size_t)c * 4;
    auto in = [&](int tt) { // tt in [-3, W): history from ring, else qkv[tt]
        if (tt >= 0) return qkvs[(size_t)tt * channels + c];
        return rin[(tt + 3) * channels + c]; // rin[0]=oldest(t-3), [1]=t-2, [2]=t-1
    };
    float a = in(t - 3), b = in(t - 2), cc2 = in(t - 1), x = in(t);
    float acc = a * wc[3] + b * wc[2] + cc2 * wc[1] + x * wc[0];
    outs[(size_t)t * channels + c] = acc / (1.0f + expf(-acc));
    float* rout = routs + (size_t)t * 3 * channels;
    rout[c] = in(t - 2);
    rout[channels + c] = in(t - 1);
    rout[2 * channels + c] = in(t);
}

// delta: one CTA per head; S resident in dynamic smem; loop W tokens with
// per-step arithmetic identical to k_delta_step (same tile split, same
// part[0..3] reduction order); stream each token's post-state to its role
// buffer and its o to the lane output.
__global__ void k_delta_chunk(const float* __restrict__ Ssrc, float* const* __restrict__ Sdsts,
                              const float* const* __restrict__ convs,
                              const float* const* __restrict__ gs,
                              const float* const* __restrict__ betas,
                              float* const* __restrict__ os, int W) {
    extern __shared__ float s_S[]; // [SK*SK] = 64KB
    const int h = blockIdx.x;
    const int j = threadIdx.x & (SK - 1);
    const int it = threadIdx.x >> 7;
    const int i0 = it * 32;
    const int qk = h / 3;
    __shared__ float sq[SK], sk[SK], part[4][SK], dj[SK];
    const float scale = rsqrtf((float)SK);
    const float* Si = Ssrc + (size_t)h * SK * SK;
#pragma unroll 8
    for (int i = i0; i < i0 + 32; i++) s_S[i * SK + j] = Si[i * SK + j];
    for (int t = 0; t < W; t++) {
        const float* conv = convs[t];
        __syncthreads();
        if (it == 0) {
            sq[j] = conv[qk * SK + j] * scale;
            sk[j] = conv[2048 + qk * SK + j];
        }
        __syncthreads();
        const float decay = expf(gs[t][h]);
        float* So = Sdsts[t] + (size_t)h * SK * SK;
        float pred = 0.f;
#pragma unroll 8
        for (int i = i0; i < i0 + 32; i++) {
            float s = s_S[i * SK + j] * decay;
            s_S[i * SK + j] = s;
            pred += sk[i] * s;
        }
        part[it][j] = pred;
        __syncthreads();
        if (it == 0) {
            float p = part[0][j] + part[1][j] + part[2][j] + part[3][j];
            float vj = conv[4096 + h * SK + j];
            dj[j] = betas[t][h] * (vj - p);
        }
        __syncthreads();
        float d = dj[j];
        float acc = 0.f;
#pragma unroll 8
        for (int i = i0; i < i0 + 32; i++) {
            float s = s_S[i * SK + j] + sk[i] * d;
            s_S[i * SK + j] = s;
            So[i * SK + j] = s; // stream role-k snapshot
            acc += sq[i] * s;
        }
        part[it][j] = acc;
        __syncthreads();
        if (it == 0)
            os[t][h * SK + j] = part[0][j] + part[1][j] + part[2][j] + part[3][j];
    }
}

template <typename F> static double timeit(F&& fn, int reps) {
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (int r = 0; r < reps; r++) fn();
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return (double)ms / reps;
}

static void fill_rand(float* d, size_t n, unsigned seed, float scale) {
    std::vector<float> v(n);
    unsigned s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        v[i] = (((s >> 8) & 0xFFFF) / 65536.0f - 0.5f) * scale;
    }
    CUDA_CHECK(cudaMemcpy(d, v.data(), n * 4, cudaMemcpyHostToDevice));
}

static bool bitwise_eq(const float* a, const float* b, size_t n, const char* tag) {
    std::vector<float> va(n), vb(n);
    CUDA_CHECK(cudaMemcpy(va.data(), a, n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vb.data(), b, n * 4, cudaMemcpyDeviceToHost));
    if (memcmp(va.data(), vb.data(), n * 4)) {
        size_t k = 0;
        while (k < n && va[k] == vb[k]) k++;
        printf("  BITWISE FAIL %s at %zu: %a vs %a\n", tag, k, va[k], vb[k]);
        return false;
    }
    return true;
}

int main() {
    const size_t SBYTES = (size_t)GDN_HEADS * SK * SK;
    const int MAXW = 16;
    // ---- delta ----
    float *S0, *g, *beta;
    CUDA_CHECK(cudaMalloc(&S0, SBYTES * 4));
    CUDA_CHECK(cudaMalloc(&g, MAXW * GDN_HEADS * 4));
    CUDA_CHECK(cudaMalloc(&beta, MAXW * GDN_HEADS * 4));
    fill_rand(S0, SBYTES, 7, 0.5f);
    fill_rand(g, MAXW * GDN_HEADS, 11, 2.0f); // decay = exp(g), g in (-1,1)
    fill_rand(beta, MAXW * GDN_HEADS, 13, 1.0f);
    float *convs[MAXW], *Sser[MAXW], *Schk[MAXW], *oser[MAXW], *ochk[MAXW];
    for (int t = 0; t < MAXW; t++) {
        CUDA_CHECK(cudaMalloc(&convs[t], GDN_CH * 4));
        fill_rand(convs[t], GDN_CH, 100 + t, 1.0f);
        CUDA_CHECK(cudaMalloc(&Sser[t], SBYTES * 4));
        CUDA_CHECK(cudaMalloc(&Schk[t], SBYTES * 4));
        CUDA_CHECK(cudaMalloc(&oser[t], GDN_HEADS * SK * 4));
        CUDA_CHECK(cudaMalloc(&ochk[t], GDN_HEADS * SK * 4));
    }
    // device pointer arrays for the chunk kernel
    float **d_Sdsts, **d_convs, **d_gs, **d_betas, **d_os;
    CUDA_CHECK(cudaMalloc(&d_Sdsts, MAXW * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_convs, MAXW * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_gs, MAXW * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_betas, MAXW * sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_os, MAXW * sizeof(float*)));
    {
        float* hg[MAXW];
        float* hb[MAXW];
        for (int t = 0; t < MAXW; t++) { hg[t] = g + t * GDN_HEADS; hb[t] = beta + t * GDN_HEADS; }
        CUDA_CHECK(cudaMemcpy(d_Sdsts, Schk, MAXW * sizeof(float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_convs, convs, MAXW * sizeof(float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gs, hg, MAXW * sizeof(float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_betas, hb, MAXW * sizeof(float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_os, ochk, MAXW * sizeof(float*), cudaMemcpyHostToDevice));
    }
    const size_t CHUNK_SM = (size_t)SK * SK * 4;
    CUDA_CHECK(cudaFuncSetAttribute(k_delta_chunk, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    CHUNK_SM));
    printf("== delta_step: serial W-launch chain vs one-launch smem-resident chunk\n");
    for (int W : {4, 8, 16}) {
        auto serial = [&] {
            for (int t = 0; t < W; t++)
                k_delta_step<<<GDN_HEADS, 512>>>(t == 0 ? S0 : Sser[t - 1], Sser[t], convs[t],
                                                 g + t * GDN_HEADS, beta + t * GDN_HEADS,
                                                 oser[t]);
        };
        auto chunk = [&] {
            k_delta_chunk<<<GDN_HEADS, 512, CHUNK_SM>>>(S0, d_Sdsts, d_convs, d_gs, d_betas,
                                                        d_os, W);
        };
        serial();
        chunk();
        CUDA_CHECK(cudaDeviceSynchronize());
        bool ok = true;
        for (int t = 0; t < W; t++) {
            ok &= bitwise_eq(Sser[t], Schk[t], SBYTES, "S");
            ok &= bitwise_eq(oser[t], ochk[t], GDN_HEADS * SK, "o");
        }
        double ms_s = timeit(serial, 200), ms_c = timeit(chunk, 200);
        printf("  W=%2d serial %8.1f us  chunk %8.1f us  %.2fx  bitwise %s\n", W, ms_s * 1e3,
               ms_c * 1e3, ms_s / ms_c, ok ? "OK" : "FAIL");
    }

    // ---- conv ----
    printf("== conv_step: serial W-launch chain vs one-launch parallel chunk\n");
    float *ring0, *convw, *qkvs;
    CUDA_CHECK(cudaMalloc(&ring0, 3 * GDN_CH * 4));
    CUDA_CHECK(cudaMalloc(&convw, GDN_CH * 4 * 4));
    CUDA_CHECK(cudaMalloc(&qkvs, (size_t)MAXW * GDN_CH * 4));
    fill_rand(ring0, 3 * GDN_CH, 21, 1.0f);
    fill_rand(convw, GDN_CH * 4, 23, 0.5f);
    fill_rand(qkvs, (size_t)MAXW * GDN_CH, 25, 1.0f);
    float *rser[MAXW], *rchk, *cser[MAXW], *cchk;
    for (int t = 0; t < MAXW; t++) {
        CUDA_CHECK(cudaMalloc(&rser[t], 3 * GDN_CH * 4));
        CUDA_CHECK(cudaMalloc(&cser[t], GDN_CH * 4));
    }
    CUDA_CHECK(cudaMalloc(&rchk, (size_t)MAXW * 3 * GDN_CH * 4));
    CUDA_CHECK(cudaMalloc(&cchk, (size_t)MAXW * GDN_CH * 4));
    for (int W : {4, 8, 16}) {
        auto serial = [&] {
            for (int t = 0; t < W; t++)
                k_conv_step<<<(GDN_CH + 255) / 256, 256>>>(t == 0 ? ring0 : rser[t - 1],
                                                           rser[t], qkvs + (size_t)t * GDN_CH,
                                                           convw, cser[t], GDN_CH);
        };
        auto chunk = [&] {
            k_conv_chunk<<<(GDN_CH * W + 255) / 256, 256>>>(ring0, rchk, qkvs, convw, cchk,
                                                            GDN_CH, W);
        };
        serial();
        chunk();
        CUDA_CHECK(cudaDeviceSynchronize());
        bool ok = true;
        for (int t = 0; t < W; t++) {
            ok &= bitwise_eq(rser[t], rchk + (size_t)t * 3 * GDN_CH, 3 * GDN_CH, "ring");
            ok &= bitwise_eq(cser[t], cchk + (size_t)t * GDN_CH, GDN_CH, "conv");
        }
        double ms_s = timeit(serial, 200), ms_c = timeit(chunk, 200);
        printf("  W=%2d serial %8.1f us  chunk %8.1f us  %.2fx  bitwise %s\n", W, ms_s * 1e3,
               ms_c * 1e3, ms_s / ms_c, ok ? "OK" : "FAIL");
    }
    printf("GO bar: delta chunk >= 2.5x at W=8 with bitwise OK (plan P0).\n");
    return 0;
}
