// maxd7 width-8 cost attribution (BUILDLOG 2026-07-09): the width-8 verify
// round measured +3.0 ms over width-7 (~2x the decayed-increment
// extrapolation). This sweeps the two candidate kernels in isolation:
//   (a) batched GEMV gemv_q4_n / gemv_q8_n at nb = 5..8 on real weight shapes
//       (register/occupancy suspect: per-lane accumulators grow with NB)
//   (b) fd2 decode attention at ntok = 5..8, fp8 KV, verify-shaped positions,
//       at cctx-like (28672) and 61K depths
// Usage: width_bench model.q27
#include <cuda_fp8.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/device_model.h"
#include "../src/kernels.cuh"
#include "../src/loader.h"
#include "../src/spec3.cuh"

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err__ = (x);                                               \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static std::vector<float> rand_vec(size_t n, unsigned seed) {
    std::vector<float> v(n);
    unsigned s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        v[i] = ((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
    }
    return v;
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

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s model.q27\n", argv[0]); return 1; }
    q27::Model m = q27::Model::open(argv[1]);
    q27::DeviceModel dm(m);

    // ---- (a) batched GEMV width sweep on real weights ----
    const int64_t cols = 5120;
    q27k::XQuant qs[16]; // W16: lanes for the nb=9..16 sweep
    float* ys[16];
    std::vector<float> x = rand_vec(cols, 91);
    float* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, cols * 4));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), cols * 4, cudaMemcpyHostToDevice));
    for (int i = 0; i < 16; i++) {
        qs[i] = q27k::xquant_alloc(cols);
        q27k::quantize_x(d_x, cols, qs[i]);
        CUDA_CHECK(cudaMalloc(&ys[i], 248320 * 4));
    }
    // Q4 ffn_gate, rotated across 4 layers so L2 never holds the weights
    const char* names[4] = {"blk.0.ffn_gate.weight", "blk.1.ffn_gate.weight",
                            "blk.2.ffn_gate.weight", "blk.4.ffn_gate.weight"};
    const q27::DevTensor* fd[4];
    for (int i = 0; i < 4; i++) fd[i] = &dm.upload(names[i]);
    int64_t frows = m.get(names[0]).rows();
    printf("== gemv width sweep (ms/call, L2-rotated Q4 ffn 17408x5120 | resident Q8 head 248320x5120) ==\n");
    const q27::DevTensor& hd = dm.upload("output.weight");
    int64_t hrows = m.get("output.weight").rows();
    double prev4 = 0, prev8 = 0;
    for (int nb = 5; nb <= 16; nb++) { // W16: sweep through the new N=13..16 instantiations
        int rot = 0;
        double q4 = timeit(
            [&] {
                const q27::DevTensor* t = fd[rot++ & 3];
                q27k::gemv_q4_n((const uint8_t*)t->data, (const __half*)t->scales, qs, nb,
                                ys, frows, cols, 0);
            },
            40);
        double q8 = timeit(
            [&] {
                q27k::gemv_q8_n((const int8_t*)hd.data, (const __half*)hd.scales, qs, nb,
                                ys, hrows, cols, 0);
            },
            40);
        printf("  nb=%d: q4_ffn %.4f ms (%+.1f%%)   q8_head %.4f ms (%+.1f%%)\n", nb, q4,
               prev4 ? (q4 / prev4 - 1) * 100 : 0.0, q8,
               prev8 ? (q8 / prev8 - 1) * 100 : 0.0);
        prev4 = q4; prev8 = q8;
    }

    // ---- (b) fd2 attention width sweep, fp8 KV, verify-shaped positions ----
    const int NKV = 4, GQA = 6, HD = 256, NH = NKV * GQA;
    const int SEQMAX = 61440, ROW = NKV * HD;
    std::vector<float> kf = rand_vec((size_t)SEQMAX * ROW, 7);
    std::vector<__nv_fp8_e4m3> k8((size_t)SEQMAX * ROW), v8((size_t)SEQMAX * ROW);
    for (size_t i = 0; i < kf.size(); i++) {
        k8[i] = __nv_fp8_e4m3(kf[i]);
        v8[i] = __nv_fp8_e4m3(kf[(i * 7919) % kf.size()]);
    }
    void *d_k8, *d_v8;
    CUDA_CHECK(cudaMalloc(&d_k8, k8.size()));
    CUDA_CHECK(cudaMalloc(&d_v8, v8.size()));
    CUDA_CHECK(cudaMemcpy(d_k8, k8.data(), k8.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v8, v8.data(), v8.size(), cudaMemcpyHostToDevice));
    const int QROW = NH * 2 * HD;
    float* d_q[8];
    float* d_o[8];
    int* d_pos[8];
    for (int t = 0; t < 8; t++) {
        std::vector<float> qv = rand_vec(QROW, 100 + t);
        CUDA_CHECK(cudaMalloc(&d_q[t], (size_t)QROW * 4));
        CUDA_CHECK(cudaMemcpy(d_q[t], qv.data(), (size_t)QROW * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_o[t], (size_t)NH * HD * 4));
        CUDA_CHECK(cudaMalloc(&d_pos[t], 4));
    }
    float* d_scr;
    CUDA_CHECK(cudaMalloc(&d_scr, (size_t)8 * NH * q27k::FD_MAXNS * q27k::FD_ST * 4));
    const float scale = 1.0f / sqrtf((float)HD);
    printf("== fd2 fp8 width sweep (ms/call; x16 layers = ms/round share) ==\n");
    for (int seq : {28672, 61440}) {
        double prev = 0;
        for (int ntok = 5; ntok <= 8; ntok++) {
            q27k::CP3 q{{d_q[0], d_q[1], d_q[2], d_q[3], d_q[4], d_q[5], d_q[6], d_q[7]}};
            q27k::P3 o{{d_o[0], d_o[1], d_o[2], d_o[3], d_o[4], d_o[5], d_o[6], d_o[7]}};
            q27k::IP3 P{{d_pos[0], d_pos[1], d_pos[2], d_pos[3], d_pos[4], d_pos[5], d_pos[6],
                         d_pos[7]}};
            for (int t = 0; t < ntok; t++) {
                int p = seq - 1 - (ntok - 1 - t);
                CUDA_CHECK(cudaMemcpy(d_pos[t], &p, 4, cudaMemcpyHostToDevice));
            }
            double ms = timeit(
                [&] {
                    q27k::attn_decode3_fd2(q, 2 * HD, d_k8, d_v8, o, d_scr, P, SEQMAX, NH,
                                           NKV, HD, scale, 0, ntok, true);
                },
                50);
            printf("  seq=%d ntok=%d: %.4f ms (%+.1f%% vs prev; x16 = %.2f ms/round)\n", seq,
                   ntok, ms, prev ? (ms / prev - 1) * 100 : 0.0, ms * 16);
            prev = ms;
        }
    }
    return 0;
}
