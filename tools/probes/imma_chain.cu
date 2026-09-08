// tools/probes/imma_chain.cu -- see README.md. Dependent-chain IMMA throughput = the honest pipe peak.
// time-scaling check: does IMMA-only time scale with iters? plus a dependent-chain variant
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("err %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)
__device__ __forceinline__ void mma_a(int& d0, int& d1, int& d2, int& d3, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// NCH independent accumulator chains per warp, each chain: iters dependent IMMAs
template <int NCH>
__global__ void __launch_bounds__(256, 1) k_chain(int iters, int* out, unsigned seed) {
    const int lane = threadIdx.x & 31;
    unsigned a0 = seed ^ lane, a1 = seed * 3 ^ lane, a2 = seed * 5 ^ lane, a3 = seed * 7 ^ lane, b0 = seed * 11 ^ lane, b1 = seed * 13 ^ lane;
    int d[NCH][4];
#pragma unroll
    for (int c = 0; c < NCH; c++) for (int e = 0; e < 4; e++) d[c][e] = c + e;
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int c = 0; c < NCH; c++) mma_a(d[c][0], d[c][1], d[c][2], d[c][3], a0, a1, a2, a3, b0, b1);
    }
    int s = 0;
#pragma unroll
    for (int c = 0; c < NCH; c++) for (int e = 0; e < 4; e++) s += d[c][e];
    if (s == 123456789) out[threadIdx.x] = s;
}
template <int NCH> double run(int blocks, int iters, int* out) {
    k_chain<NCH><<<blocks, 256>>>(iters / 4, out, 1); CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0)); k_chain<NCH><<<blocks, 256>>>(iters, out, 2); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    const double macs = (double)blocks * 8 * iters * NCH * 4096.0;
    printf("  NCH=%d iters=%d: %.3f ms -> %.0f TOPS (%.1f ns per IMMA per warp)\n", NCH, iters, ms, 2 * macs / (ms * 1e-3) / 1e12, ms * 1e6 / (iters * NCH));
    return 2 * macs / (ms * 1e-3) / 1e12;
}
int main() {
    int* out; CK(cudaMalloc(&out, 4096));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); const int nsm = p.multiProcessorCount;
    printf("%s %d SMs, 1 block (8 warps)/SM\n", p.name, nsm);
    run<1>(nsm, 20000, out); run<1>(nsm, 40000, out);
    run<4>(nsm, 10000, out); run<8>(nsm, 10000, out); run<16>(nsm, 5000, out); run<16>(nsm, 10000, out);
    run<32>(nsm, 5000, out);
    return 0;
}
