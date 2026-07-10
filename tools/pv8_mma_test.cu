// fp8-PV MMA fragment-layout correctness microtest (Phase B pre-build gate).
// The whole risk of the real kernel: P leaves QK^T in ACCUMULATOR layout
// (thread(gid,tg) holds keys {n*8+2tg, +1} for n=0..3) but the PV MMA A
// operand wants A[token][key] in m16n8k32.e4m3 layout (thread holds 4
// consecutive keys tg*4..+3). This test builds ONE warp doing O[16,256] =
// P[16,32] @ V[32,256] with: (a) accumulator-layout P written to smem then
// re-read in A-frag layout (per-warp relayout, __syncwarp only), (b) V read
// STRIDED from a [key][dim] fp8 buffer as the B operand, (c) mma_e4m3 ->
// O accumulator in the SAME layout the f16 PV produces. Checks O vs a CPU
// reference (fp8-rounded P and V). GO = max abs err < 1e-2 (fp8 granularity).
//
// Build: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=sm_120 tools/pv8_mma_test.cu -o build/pv8_mma_test
#include <cstdint>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CK(x)                                                              \
    do {                                                                   \
        cudaError_t e_ = (x);                                              \
        if (e_ != cudaSuccess) {                                           \
            fprintf(stderr, "%s @%d\n", cudaGetErrorString(e_), __LINE__); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

constexpr int PP = 32, HD = 256, TT = 16, LDK = 272;

__device__ __forceinline__ void mma_e4m3(float& d0, float& d1, float& d2, float& d3, uint32_t a0,
                                         uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1, float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2),
          "f"(c3));
}

// P in the QK^T ACCUMULATOR layout: sacc[n][e] for n=0..3, e=0..3, where
// s[n][0]=P[gid][n*8+2tg], s[n][1]=P[gid][n*8+2tg+1], s[n][2]=P[gid+8][...],
// s[n][3]=P[gid+8][...+1]. V as [key][dim] fp8. Output O[16][256] via the
// f16-PV accumulator convention (o[n=0..31][0..3]).
__global__ void k_pv8(const float* Pin, const __nv_fp8_e4m3* Vin, float* Oout) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, tg = lane & 3;
    __shared__ __nv_fp8_e4m3 sP[TT * PP]; // [token][key], fp8

    // seed accumulator-layout P registers from Pin[token*PP+key]
    float s[4][4];
#pragma unroll
    for (int n = 0; n < 4; n++) {
        s[n][0] = Pin[gid * PP + (n * 8 + 2 * tg)];
        s[n][1] = Pin[gid * PP + (n * 8 + 2 * tg + 1)];
        s[n][2] = Pin[(gid + 8) * PP + (n * 8 + 2 * tg)];
        s[n][3] = Pin[(gid + 8) * PP + (n * 8 + 2 * tg + 1)];
    }

    // (a) relayout P: write accumulator-held values to sP[token][key] (fp8),
    // __syncwarp, read back in A-frag layout. Per-warp: warpsync suffices.
#pragma unroll
    for (int n = 0; n < 4; n++) {
        sP[gid * PP + (n * 8 + 2 * tg)] = __nv_fp8_e4m3(s[n][0]);
        sP[gid * PP + (n * 8 + 2 * tg + 1)] = __nv_fp8_e4m3(s[n][1]);
        sP[(gid + 8) * PP + (n * 8 + 2 * tg)] = __nv_fp8_e4m3(s[n][2]);
        sP[(gid + 8) * PP + (n * 8 + 2 * tg + 1)] = __nv_fp8_e4m3(s[n][3]);
    }
    __syncwarp();
    // A operand: thread(gid,tg) -> a0=P[gid][tg*4..+3], a1=P[gid+8][tg*4..+3],
    // a2=P[gid][tg*4+16..+19], a3=P[gid+8][tg*4+16..+19]
    uint32_t a0 = *(const uint32_t*)(sP + gid * PP + tg * 4);
    uint32_t a1 = *(const uint32_t*)(sP + (gid + 8) * PP + tg * 4);
    uint32_t a2 = *(const uint32_t*)(sP + gid * PP + tg * 4 + 16);
    uint32_t a3 = *(const uint32_t*)(sP + (gid + 8) * PP + tg * 4 + 16);

    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[i][e] = 0.f;

    // (b) B operand = V[key][dim] as n8(dim) x k32(key): thread(gid,tg) needs
    // b0 = V[key tg*4..+3][dim = ntile*8+gid], b1 = V[key tg*4+16..+19][dim].
    // 4 STRIDED bytes (stride HD) gathered + packed.
#pragma unroll
    for (int nt = 0; nt < 32; nt++) {
        const int dim = nt * 8 + gid;
        uint32_t b0 = 0, b1 = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned char lo = *(const unsigned char*)&Vin[(tg * 4 + j) * HD + dim];
            unsigned char hi = *(const unsigned char*)&Vin[(tg * 4 + j + 16) * HD + dim];
            b0 |= (uint32_t)lo << (8 * j);
            b1 |= (uint32_t)hi << (8 * j);
        }
        mma_e4m3(o[nt][0], o[nt][1], o[nt][2], o[nt][3], a0, a1, a2, a3, b0, b1, o[nt][0],
                 o[nt][1], o[nt][2], o[nt][3]);
    }

    // write O in the f16-PV accumulator convention: o[nt][*] holds
    // O[gid][dim=nt*8+2tg], O[gid][+1], O[gid+8][...], O[gid+8][+1]
#pragma unroll
    for (int nt = 0; nt < 32; nt++) {
        const int d0 = nt * 8 + tg * 2;
        Oout[gid * HD + d0] = o[nt][0];
        Oout[gid * HD + d0 + 1] = o[nt][1];
        Oout[(gid + 8) * HD + d0] = o[nt][2];
        Oout[(gid + 8) * HD + d0 + 1] = o[nt][3];
    }
}

int main() {
    float hP[TT * PP];
    float hV[PP * HD];
    for (int i = 0; i < TT * PP; i++) hP[i] = (float)((i * 37) % 13) / 40.0f; // ~[0,0.3]
    for (int i = 0; i < PP * HD; i++) hV[i] = ((float)((i * 53) % 29) / 29.0f - 0.5f) * 0.8f;

    // CPU reference: fp8-round P and V (both operands are e4m3 in the MMA)
    float ref[TT * HD];
    for (int t = 0; t < TT; t++)
        for (int d = 0; d < HD; d++) {
            float acc = 0.f;
            for (int k = 0; k < PP; k++) {
                float p = (float)__nv_fp8_e4m3(hP[t * PP + k]);
                float v = (float)__nv_fp8_e4m3(hV[k * HD + d]);
                acc += p * v;
            }
            ref[t * HD + d] = acc;
        }

    float *dP, *dO;
    __nv_fp8_e4m3* dV;
    CK(cudaMalloc(&dP, sizeof hP));
    CK(cudaMalloc(&dV, PP * HD));
    CK(cudaMalloc(&dO, sizeof ref));
    __nv_fp8_e4m3 hV8[PP * HD];
    for (int i = 0; i < PP * HD; i++) hV8[i] = __nv_fp8_e4m3(hV[i]);
    CK(cudaMemcpy(dP, hP, sizeof hP, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hV8, PP * HD, cudaMemcpyHostToDevice));
    k_pv8<<<1, 32>>>(dP, dV, dO);
    CK(cudaDeviceSynchronize());
    float hO[TT * HD];
    CK(cudaMemcpy(hO, dO, sizeof hO, cudaMemcpyDeviceToHost));

    float maxerr = 0.f;
    int bad = 0;
    for (int i = 0; i < TT * HD; i++) {
        float e = fabsf(hO[i] - ref[i]);
        if (e > maxerr) maxerr = e;
        if (e > 1e-2f) bad++;
    }
    printf("max abs err %.5f, elems > 1e-2: %d/%d -> %s\n", maxerr, bad, TT * HD,
           maxerr < 1e-2f ? "GO (layout correct)" : "FAIL (fragment layout wrong)");
    return maxerr < 1e-2f ? 0 : 1;
}
