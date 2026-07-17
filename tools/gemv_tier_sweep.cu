// sm_86 GEMV occupancy tier sweep (2026-07-17, Ampere pass item #2).
// Focused fork of width_bench.cu part (a): times gemv_q4_n / gemv_q8_n on
// real weight shapes over N=2..8 (the w8-build ladder range). Built THREE
// times with the launch_bounds tier forced (4CTA / 3CTA / 2CTA) so each N's
// per-tier ms/call can be compared -- the tier boundary is a compile-time
// launch_bounds min-CTA pin, so a runtime knob cannot sweep it.
// Vanilla model only (standing benchmark rule): real Q4 ffn + real Q8 head.
// Usage: gemv_tier_sweep model.q27
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/device_model.h"
#include "../src/kernels.cuh"
#include "../src/loader.h"

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err__ = (x);                                               \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#ifndef TIER_TAG
#define TIER_TAG "?"
#endif

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

    const int64_t cols = 5120;
    q27k::XQuant qs[16];
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
    // Q4 ffn_gate rotated over 4 layers so L2 never holds the weights
    // (the 3090's 6 MB L2 is exactly the variable this sweep exists for).
    const char* names[4] = {"blk.0.ffn_gate.weight", "blk.1.ffn_gate.weight",
                            "blk.2.ffn_gate.weight", "blk.4.ffn_gate.weight"};
    const q27::DevTensor* fd[4];
    for (int i = 0; i < 4; i++) fd[i] = &dm.upload(names[i]);
    int64_t frows = m.get(names[0]).rows();
    const q27::DevTensor& hd = dm.upload("output.weight");
    int64_t hrows = m.get("output.weight").rows();

    printf("== TIER %s : gemv ms/call (L2-rotated Q4 ffn 17408x5120 | Q8 head %ldx5120) ==\n",
           TIER_TAG, (long)hrows);
    for (int nb = 2; nb <= 8; nb++) {
        int rot = 0;
        double q4 = timeit(
            [&] {
                const q27::DevTensor* t = fd[rot++ & 3];
                q27k::gemv_q4_n((const uint8_t*)t->data, (const __half*)t->scales, qs, nb,
                                ys, frows, cols, 0);
            },
            100);
        double q8 = timeit(
            [&] {
                q27k::gemv_q8_n((const int8_t*)hd.data, (const __half*)hd.scales, qs, nb,
                                ys, hrows, cols, 0);
            },
            100);
        printf("TIER %s nb=%d q4 %.4f q8 %.4f\n", TIER_TAG, nb, q4, q8);
    }
    return 0;
}
