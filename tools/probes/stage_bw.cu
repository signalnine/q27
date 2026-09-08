// tools/probes/stage_bw.cu -- see README.md. Stage-fill pipeline and L2 bandwidth.
// (1) staging-only: the spike's cp.async tile pipeline (W 128x64 B + X 128x128 B
// + scales per stage, s2, one barrier per stage) with NO compute -- how fast can
// the tiles stream? (2) plain L2 bandwidth: 16-B loads over an L2-resident buffer.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("err %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(void* dst, const void* src) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(dst)), "l"(src)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

template <int STAGES, int TILE_BYTES>
__global__ void __launch_bounds__(256, 1) k_stage(const unsigned char* __restrict__ W, const unsigned char* __restrict__ X, int wrow, int xrow, int k_tiles, unsigned* out) {
    extern __shared__ __align__(128) unsigned char smem[];
    const int tid = threadIdx.x;
    const int r0 = blockIdx.y * 128, t0 = blockIdx.x * 128;
    auto stage = [&](int slot, int kt) {
        unsigned char* sw = smem + slot * TILE_BYTES; unsigned char* sx = sw + 8192;
        for (int i = tid; i < 512; i += 256) { int r = i / 4, c = i % 4; cp16(sw + r * 64 + c * 16, W + (size_t)(r0 + r) * wrow + kt * 64 + c * 16); }
        for (int i = tid; i < 1024; i += 256) { int r = i / 8, c = i % 8; cp16(sx + r * 128 + c * 16, X + (size_t)(t0 + r) * xrow + kt * 128 + c * 16); }
        cp_commit();
    };
    for (int s = 0; s < STAGES - 1; s++) { if (s < k_tiles) stage(s, s); else cp_commit(); }
    unsigned acc = 0;
    for (int kt = 0; kt < k_tiles; kt++) {
        cp_wait<STAGES - 2>(); __syncthreads();
        const int nxt = kt + STAGES - 1;
        if (nxt < k_tiles) stage(nxt % STAGES, nxt); else cp_commit();
        acc += ((const unsigned*)(smem + (kt % STAGES) * TILE_BYTES))[tid];
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
}
__global__ void k_l2(const uint4* __restrict__ p, size_t n16, int reps, unsigned* out) {
    unsigned acc = 0;
    for (int r = 0; r < reps; r++)
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) { uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0xdeadbeef) out[threadIdx.x] = acc;
}
int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    unsigned* out; CK(cudaMalloc(&out, 4096));
    // attn_out shape: rows 5120 (W row = 4096 B packed), cols 8192 (X row = 8192 B), T=1024
    const int rows = 5120, cols = 8192, T = 1024, k_tiles = cols / 128;
    unsigned char *W, *X; CK(cudaMalloc(&W, (size_t)rows * cols / 2)); CK(cudaMalloc(&X, (size_t)T * cols));
    CK(cudaMemset(W, 1, (size_t)rows * cols / 2)); CK(cudaMemset(X, 2, (size_t)T * cols));
    dim3 grid(T / 128, rows / 128);
    const size_t bytes = (size_t)grid.x * grid.y * k_tiles * (8192 + 16384);
    auto time = [&](auto fn, int reps) { fn(); CK(cudaDeviceSynchronize()); cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventRecord(e0)); for (int i = 0; i < reps; i++) fn(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return ms / reps; };
    CK(cudaFuncSetAttribute(k_stage<2, 24576>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * 24576));
    CK(cudaFuncSetAttribute(k_stage<3, 24576>, cudaFuncAttributeMaxDynamicSharedMemorySize, 3 * 24576));
    CK(cudaFuncSetAttribute(k_stage<4, 24576>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 24576));
    double t2 = time([&] { k_stage<2, 24576><<<grid, 256, 2 * 24576>>>(W, X, cols / 2, cols, k_tiles, out); }, 20);
    double t3 = time([&] { k_stage<3, 24576><<<grid, 256, 3 * 24576>>>(W, X, cols / 2, cols, k_tiles, out); }, 20);
    double t4 = time([&] { k_stage<4, 24576><<<grid, 256, 4 * 24576>>>(W, X, cols / 2, cols, k_tiles, out); }, 20);
    printf("staging-only (attn_out T=1024 grid %dx%d, %zu MB per pass): s2 %.3f ms (%.2f TB/s)  s3 %.3f ms (%.2f TB/s)  s4 %.3f ms (%.2f TB/s)\n",
           grid.x, grid.y, bytes >> 20, t2, bytes / (t2 * 1e-3) / 1e12, t3, bytes / (t3 * 1e-3) / 1e12, t4, bytes / (t4 * 1e-3) / 1e12);
    printf("  full spike kernel on this shape: 0.201 ms -> staging alone is %.0f%% of it (s2)\n", 100 * t2 / 0.201);
    for (size_t mb : {16, 32, 64, 256}) {
        const size_t n16 = mb * 1024 * 1024 / 16;
        double tl = time([&] { k_l2<<<p.multiProcessorCount * 8, 256>>>((const uint4*)X, n16 < (size_t)T * cols / 16 ? n16 : (size_t)T * cols / 16, 4, out); }, 10);
        size_t eff = (n16 < (size_t)T * cols / 16 ? n16 : (size_t)T * cols / 16) * 16 * 4;
        printf("plain 16-B loads over %zu MB x4: %.3f ms -> %.2f TB/s\n", eff / 4 >> 20, tl, eff / (tl * 1e-3) / 1e12);
    }
    return 0;
}
