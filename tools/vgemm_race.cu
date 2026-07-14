// vgemm_race -- racecheck driver for k_vgemm's ONLY cross-warp communication:
// the intra-CTA fixed-order smem reduce (which aliases s_w) and k_reduce_z.
// racecheck instruments every shared-memory access, so it cannot finish on a
// real 47MB weight; this drives the identical code path on a tiny synthetic one.
// Usage: compute-sanitizer --tool=racecheck build/vgemm_race
#include <cstdio>
#include <vector>
#include "../src/device_model.h"
#include "../src/loader.h"
#include "../src/kernels.cuh"
#include "../src/vgemm.cuh"

int main() {
    // small but structurally identical: cols multiple of KB=256, rows > MR,
    // and z forced > 1 so BOTH reduces run.
    const int64_t rows = 128, cols = 1024;   // 4 row-CTAs, 8 stages
    std::vector<uint8_t> hw(rows * cols / 2, 0x53);
    std::vector<__half> hs(rows * (cols / 64), __float2half(0.01f));
    q27::DevTensor w{};
    w.dtype = q27::DType::Q4_G64;
    w.rows = rows; w.cols = cols;
    CUDA_CHECK(cudaMalloc(&w.data, hw.size()));
    CUDA_CHECK(cudaMalloc(&w.scales, hs.size() * 2));
    CUDA_CHECK(cudaMemcpy(w.data, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(w.scales, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice));

    float* d_x; CUDA_CHECK(cudaMalloc(&d_x, cols * 4));
    std::vector<float> hx(cols);
    for (int i = 0; i < cols; i++) hx[i] = 0.001f * (float)((i * 37) % 251) - 0.12f;
    CUDA_CHECK(cudaMemcpy(d_x, hx.data(), cols * 4, cudaMemcpyHostToDevice));

    q27k::XLanes X{}; q27k::YLanes Y{};
    q27k::XQuant xq[W_PLUMB];
    for (int t = 0; t < W_PLUMB; t++) {
        xq[t] = q27k::xquant_alloc(cols);
        q27k::quantize_x(d_x, cols, xq[t]);
        X.nat[t] = xq[t].nat; X.xs[t] = xq[t].scale;
        CUDA_CHECK(cudaMalloc(&Y.y[t], rows * 4));
    }
    const q27::DevTensor* wl[1] = {&w};
    size_t wsb = q27k::vgemm_ws_bytes(wl, 1);
    float* ws = nullptr;
    if (wsb) CUDA_CHECK(cudaMalloc(&ws, wsb));
    printf("rows=%ld cols=%ld  z=%d  ws=%zu B\n", (long)rows, (long)cols,
           q27k::vgemm_z(rows, cols), wsb);
    for (int T : {2, 8, 16}) {
        bool ok = q27k::vgemm_verify(w, X, Y, ws, T, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("  T=%2d launched=%d\n", T, (int)ok);
    }
    printf("vgemm_race: done\n");
    return 0;
}
