// tools/probes/imma_fold.cu -- see README.md. Register-only IMMA + fold instruction-mix ceilings.
// Register-only: the real per-tile-group instruction mix (mma_z, mma_a, fold)
// with NCH independent tiles per warp, no loads. Fold variants: 0 none (xor
// sink), 1 = full W4A8 fold (IADD, FFMA, FMUL, FFMA per output), 2 = fold
// without the FMUL (scale product hoisted: not bitwise, just a cost probe),
// 3 = I2F + FMUL + FFMA (3 ops, cvt-based).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("err %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)
__device__ __forceinline__ void mma_z(int& d0, int& d1, int& d2, int& d3, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
}
__device__ __forceinline__ void mma_a(int& d0, int& d1, int& d2, int& d3, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float vmul(float a, float b) { float r; asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float vfma(float a, float b, float c) { float r; asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float vi2f16(int d) { int m; asm volatile("add.s32 %0, %1, 0x4B400000;" : "=r"(m) : "r"(d)); return vfma(__int_as_float(m), 0.0625f, -786432.0f); }
__device__ __forceinline__ float vcvt(int d) { float r; asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(r) : "r"(d)); return r; }

template <int NCH, int FOLD>
__global__ void __launch_bounds__(256, 1) k_mix(int iters, float* out, unsigned seed) {
    const int lane = threadIdx.x & 31;
    unsigned a[2][4], b[NCH][4];
#pragma unroll
    for (int h = 0; h < 2; h++) for (int e = 0; e < 4; e++) a[h][e] = seed * (h * 4 + e + 1) ^ lane;
#pragma unroll
    for (int c = 0; c < NCH; c++) for (int e = 0; e < 4; e++) b[c][e] = seed * (c * 4 + e + 3) ^ (lane * 7);
    float acc[NCH][4] = {}; float w0 = 0.01f * seed, w1 = 0.02f * seed, xs0[NCH], xs1[NCH];
#pragma unroll
    for (int c = 0; c < NCH; c++) { xs0[c] = 0.1f * (c + 1); xs1[c] = 0.2f * (c + 1); }
    for (int it = 0; it < iters; it++) {
        a[0][0] += 0x01010101u; a[1][2] ^= (unsigned)it;  // loop-variant inputs: ptxas must not hoist
#pragma unroll
        for (int c = 0; c < NCH; c++) {
            int d0, d1, d2, d3;
            mma_z(d0, d1, d2, d3, a[0][0], a[0][1], a[0][2], a[0][3], b[c][0], b[c][1]);
            mma_a(d0, d1, d2, d3, a[1][0], a[1][1], a[1][2], a[1][3], b[c][2], b[c][3]);
            if (FOLD == 1) {
                acc[c][0] = vfma(vmul(w0, xs0[c]), vi2f16(d0), acc[c][0]);
                acc[c][1] = vfma(vmul(w0, xs1[c]), vi2f16(d1), acc[c][1]);
                acc[c][2] = vfma(vmul(w1, xs0[c]), vi2f16(d2), acc[c][2]);
                acc[c][3] = vfma(vmul(w1, xs1[c]), vi2f16(d3), acc[c][3]);
            } else if (FOLD == 2) {
                acc[c][0] = vfma(xs0[c], vi2f16(d0), acc[c][0]);
                acc[c][1] = vfma(xs1[c], vi2f16(d1), acc[c][1]);
                acc[c][2] = vfma(xs0[c], vi2f16(d2), acc[c][2]);
                acc[c][3] = vfma(xs1[c], vi2f16(d3), acc[c][3]);
            } else if (FOLD == 3) {
                acc[c][0] = vfma(vmul(w0, xs0[c]), vcvt(d0), acc[c][0]);
                acc[c][1] = vfma(vmul(w0, xs1[c]), vcvt(d1), acc[c][1]);
                acc[c][2] = vfma(vmul(w1, xs0[c]), vcvt(d2), acc[c][2]);
                acc[c][3] = vfma(vmul(w1, xs1[c]), vcvt(d3), acc[c][3]);
            } else {
                acc[c][0] += __int_as_float(d0 ^ d1 ^ d2 ^ d3);
            }
        }
    }
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < NCH; c++) for (int e = 0; e < 4; e++) s += acc[c][e];
    if (s == 12345.678f) out[threadIdx.x] = s;
}
template <int NCH, int FOLD> double run(int blocks, int iters, float* out) {
    k_mix<NCH, FOLD><<<blocks, 256>>>(iters / 4, out, 1); CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0)); k_mix<NCH, FOLD><<<blocks, 256>>>(iters, out, 2); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    return 2.0 * blocks * 8.0 * iters * NCH * 2 * 4096.0 / (ms * 1e-3) / 1e12;
}
int main() {
    float* out; CK(cudaMalloc(&out, 4096));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); const int nsm = p.multiProcessorCount;
    printf("%s: register-only IMMA pair + fold, 8 warps/SM, NCH tiles per warp (TOPS)\n", p.name);
    printf("%4s %8s %8s %8s %8s\n", "NCH", "no fold", "fold4op", "fold3op", "cvt3op");
    printf("%4d %8.0f %8.0f %8.0f %8.0f\n", 8, run<8,0>(nsm, 8000, out), run<8,1>(nsm, 8000, out), run<8,2>(nsm, 8000, out), run<8,3>(nsm, 8000, out));
    printf("%4d %8.0f %8.0f %8.0f %8.0f\n", 16, run<16,0>(nsm, 4000, out), run<16,1>(nsm, 4000, out), run<16,2>(nsm, 4000, out), run<16,3>(nsm, 4000, out));
    printf("%4d %8.0f %8.0f %8.0f %8.0f\n", 32, run<32,0>(nsm, 2000, out), run<32,1>(nsm, 2000, out), run<32,2>(nsm, 2000, out), run<32,3>(nsm, 2000, out));
    return 0;
}
