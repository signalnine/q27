// W4A8 prefill GEMM spike -- docs/plans/2026-09-08-prefill-attack.md phase 2.
//
// Standalone: synthetic Q4_G64 weights + fp16 scales, real quantize_x_g64
// activations, the incumbent q27k::gemm_q4_T (live dispatch: k_gemm_mma_ntx at
// T >= 96 saturated, k_gemm_mma_T<MR=64> otherwise) as the BITWISE reference,
// and the new kernel k_w4a8 in several tile/pipeline variants, all timed
// in-process on the same inputs.
//
// Why the incumbent sits at 36-48% of the vendor int8 ceiling (recon 09-08):
// not the MMA. Per 64-K group and 16x8 output tile the fold costs four
// int->float conversions (I2F: 16/clk/SM, one eighth of the FMA rate), four
// FMUL and four FFMA (measured 09-08 on the register-only harness: I2F is NOT
// slow on sm_120 -- cvt+2 ops reach 896 TOPS vs 827 for the 4-op exact fold). Plus scalar LDS.32 A-fragment loads (LSU 50%, ncu 07-19) and
// a single-buffered stage with a register-staged next stage.
//
// What k_w4a8 changes, all BITWISE against the incumbent:
//   * weights stay nibble-PACKED in smem (cp.async 16 B, STAGES-deep ring);
//     one ldmatrix.x4 per 16-row tile per 64-group hands each lane the packed
//     bytes for 8 consecutive K of its row (both k32 halves, both row halves)
//   * unpack = one SHL+LOP3 (even K) / one LOP3 (odd K) per register: the
//     nibble goes into the HIGH half of the byte XOR 0x80, i.e. the s8 value
//     16*(u-8). The MMA then yields exactly 16*d (|16d| <= 1,040,384)
//   * K inside each 32-block is consumed in a permuted order (even K's in the
//     mma's first 16 K-slots, odd K's in the last 16); activations are stored
//     in that order per 32-block (Xp, produced here by k_permute_x; the port
//     folds it into quantize_x_g64), so the standard ldmatrix B fragment
//     matches. The dot product is order-invariant in int32, so d is exact
//   * conversion: f = fma(int_as_float(16d + 0x4B400000), 0.0625f, -786432.f)
//     == (float)d exactly (both steps are exact integers below 2^22), one
//     IADD + one FFMA instead of an I2F
//   * the fold itself is the incumbent's expression, acc += wsc * xs * f, in
//     the incumbent's group order (increasing K), so the fp32 sequence per
//     output is identical -> byte-identical y.
//
// Build (from the repo root):
//   nvcc -O2 -std=c++17 -gencode arch=compute_120a,code=sm_120a -I src \
//     tools/gemm_w4a8_spike.cu src/prefill.cu src/kernels.cu -o build/gemm_w4a8_spike
// Run: build/gemm_w4a8_spike [--time] [--shape NAME] [--only VARIANT-SUBSTR] [--notest]
// exits 1 on any mismatch.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "../src/kernels.cuh"
#include "../src/prefill.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

namespace spike {

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
// cp.async with zero-fill: src_size 0 reads nothing and writes zeros (the
// token / row tails). src must still be a valid global address.
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
    const int sz = ok ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(sz));
}
__device__ __forceinline__ void cp8(void* dst, const void* src, bool ok) {
    const int sz = ok ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(sz));
}
__device__ __forceinline__ void cp4(void* dst, const void* src, bool ok) {
    const int sz = ok ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_u32(dst)), "l"(src),
                 "r"(sz));
}
template <int BYTES>
__device__ __forceinline__ void cpn(void* dst, const void* src, bool ok) {
    if constexpr (BYTES == 16) cp16(dst, src, ok);
    else if constexpr (BYTES == 8) cp8(dst, src, ok);
    else cp4(dst, src, ok);
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                        unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}
__device__ __forceinline__ void mma_s8_zero(int& d0, int& d1, int& d2, int& d3, uint32_t a0,
                                            uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                            uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%10,%11,%12,%13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
}
__device__ __forceinline__ void mma_s8_acc(int& d0, int& d1, int& d2, int& d3, uint32_t a0,
                                           uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                           uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 16-B chunk swizzle so an 8-row ldmatrix phase hits 8 distinct bank groups.
template <int CPR>
__device__ __forceinline__ int swz(int r, int c) {
    if constexpr (CPR >= 8) return c ^ (r & 7);
    else if constexpr (CPR == 4) return c ^ ((r >> 1) & 3);
    else return c ^ ((r >> 2) & 1);
}

// nibble u (0..15) -> s8 16*(u-8): high half of the byte, top bit flipped
__device__ __forceinline__ uint32_t unpack_even(uint32_t p) {
    return ((p << 4) & 0xF0F0F0F0u) ^ 0x80808080u;
}
__device__ __forceinline__ uint32_t unpack_odd(uint32_t p) {
    return (p & 0xF0F0F0F0u) ^ 0x80808080u;
}
// (float)d from 16*d, exact: 12582912 + 16d is an exact float for
// |16d| < 2^22; the fma scales by 1/16 and subtracts 786432, both exact.
template <int FOLD>
__device__ __forceinline__ float i2f16(int d16) {
    if constexpr (FOLD == 1) return fmaf(__int_as_float(d16 + 0x4B400000), 0.0625f, -786432.0f);
    else if constexpr (FOLD == 2) return (float)d16;  // 16d: the 1/16 rides on wsc (3-op fold)
    else return (float)d16 * 0.0625f;  // I2F reference leg (also exact)
}


// Ordered fp32 primitives: asm volatile keeps program order among themselves
// and the (volatile) IMMAs, which is what the interleaved fold needs.
__device__ __forceinline__ float vmul(float a, float b) {
    float r; asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float vfma(float a, float b, float c) {
    float r; asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r;
}
__device__ __forceinline__ float vi2f16(int d16) {  // (float)d exact, see i2f16
    int m; asm volatile("add.s32 %0, %1, 0x4B400000;" : "=r"(m) : "r"(d16));
    return vfma(__int_as_float(m), 0.0625f, -786432.0f);
}

// Activation permute: nat64 [T][cols] -> Xp [T][cols] with each 32-K block
// stored as [16 even K][16 odd K]. Layout only; values unchanged.
__global__ void k_permute_x(const int8_t* __restrict__ nat, int8_t* __restrict__ xp, size_t n32) {
    const size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n32) return;
    const uint4 v = *(const uint4*)(nat + b * 32 + 0);
    const uint4 w = *(const uint4*)(nat + b * 32 + 16);
    const uint32_t in[8] = {v.x, v.y, v.z, v.w, w.x, w.y, w.z, w.w};
    uint32_t ev[4], od[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        // in[2i], in[2i+1] hold K 8i..8i+7; even bytes -> ev[i], odd -> od[i]
        const uint32_t lo = in[2 * i], hi = in[2 * i + 1];
        ev[i] = __byte_perm(lo, hi, 0x6420);
        od[i] = __byte_perm(lo, hi, 0x7531);
    }
    *(uint4*)(xp + b * 32 + 0) = make_uint4(ev[0], ev[1], ev[2], ev[3]);
    *(uint4*)(xp + b * 32 + 16) = make_uint4(od[0], od[1], od[2], od[3]);
}

// BN weight rows x BM tokens per block, BK K per stage, STAGES-deep cp.async
// ring, WARPS_M x WARPS_N warps (warp tile BN/WARPS_M rows x BM/WARPS_N
// tokens). y[t * rows + r] fp32, the incumbent's layout.
template <int BN, int BM, int BK, int STAGES, int WARPS_M, int WARPS_N, int FOLD, int MINB, int ABL = 0>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32, MINB)
k_w4a8(const uint8_t* __restrict__ W, const __half* __restrict__ S, const int8_t* __restrict__ Xp,
       const float* __restrict__ xs, float* __restrict__ y, int rows, int cols, int T) {
    constexpr int NTHR = WARPS_M * WARPS_N * 32;
    constexpr int WM = BN / WARPS_M, WN = BM / WARPS_N;
    constexpr int MT = WM / 16, NTL = WN / 8;
    constexpr int G = BK / 64;
    constexpr int WRB = BK / 2, XRB = BK;
    constexpr int WCPR = WRB / 16, XCPR = XRB / 16;
    constexpr int W_BYTES = BN * WRB, X_BYTES = BM * XRB;
    constexpr int WS_BYTES = BN * (G == 1 ? 2 : G) * 2, XS_BYTES = BM * G * 4;
    constexpr int STAGE_BYTES = W_BYTES + X_BYTES + WS_BYTES + XS_BYTES;
    static_assert(WM % 16 == 0 && WN % 8 == 0, "warp tile");
    static_assert(STAGE_BYTES % 128 == 0, "stage alignment");
    extern __shared__ __align__(128) unsigned char smem[];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WARPS_M, wn = warp / WARPS_M;
    const int gid = lane >> 2, tg = lane & 3;
    const int r0 = blockIdx.y * BN, t0 = blockIdx.x * BM;
    const int k_tiles = cols / BK;
    const int wrow = cols / 2;   // packed bytes per weight row
    const int ngrp = cols / 64;  // scale groups per row / token

    auto stage_tile = [&](int slot, int kt) {
        unsigned char* sw = smem + slot * STAGE_BYTES;
        unsigned char* sx = sw + W_BYTES;
        unsigned char* sws = sx + X_BYTES;
        unsigned char* sxs = sws + WS_BYTES;
        const int k0 = kt * BK;
        for (int i = tid; i < BN * WCPR; i += NTHR) {
            const int r = i / WCPR, c = i % WCPR, gr = r0 + r;
            const bool ok = gr < rows;
            const uint8_t* src = ok ? W + (size_t)gr * wrow + k0 / 2 + c * 16 : W;
            cp16(sw + r * WRB + swz<WCPR>(r, c) * 16, src, ok);
        }
        for (int i = tid; i < BM * XCPR; i += NTHR) {
            const int r = i / XCPR, c = i % XCPR, gt = t0 + r;
            const bool ok = gt < T;
            const int8_t* src = ok ? Xp + (size_t)gt * cols + k0 + c * 16 : Xp;
            cp16(sx + r * XRB + swz<XCPR>(r, c) * 16, src, ok);
        }
        for (int i = tid; i < BN; i += NTHR) {
            const int gr = r0 + i;
            const bool ok = gr < rows;
            if constexpr (G == 1) {
                // one fp16 scale per row per stage is below cp.async's 4-byte
                // minimum: copy the aligned PAIR (ngrp is even, so the row
                // base is) and pick the half by tile parity at compute time
                const __half* src = ok ? S + (size_t)gr * ngrp + (k0 / 64 & ~1) : S;
                cp4(sws + i * 4, src, ok);
            } else {
                const __half* src = ok ? S + (size_t)gr * ngrp + k0 / 64 : S;
                cpn<G * 2>(sws + i * G * 2, src, ok);
            }
        }
        // token scales land GROUP-MAJOR ([g][BM]) so the compute side reads a
        // token pair as one 8-B load: the [token][g] layout made the compiler
        // emit LDS.128 (4 wavefronts each) for 16 B of which 8 were needed --
        // 32 wavefronts per warp-stage, 40% of the ldmatrix traffic (ncu 09-08)
        for (int i = tid; i < BM * G; i += NTHR) {
            const int gg = i / BM, t = i % BM, gt = t0 + t;
            const bool ok = gt < T;
            const float* src = ok ? xs + (size_t)gt * ngrp + k0 / 64 + gg : xs;
            cp4(sxs + (gg * BM + t) * 4, src, ok);
        }
        cp_commit();
    };

    float acc[MT][NTL][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NTL; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    // prologue: STAGES-1 tiles in flight (always commit STAGES-1 groups)
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < k_tiles) stage_tile(s, s);
        else cp_commit();
    }

    // ldmatrix lane roles (constant per lane)
    const int a_row = lane & 15, a_chunk_hi = lane >> 4;  // A: rows 0-15, k32 half
    const int b_row = lane & 7, b_chunk = lane >> 3;      // B: tokens 0-7, 4 chunks/group

    if constexpr (ABL & 2) { cp_wait<0>(); __syncthreads(); }
    // kt loop unrolled by STAGES so every smem slot offset is a compile-time
    // constant (kt % 3 as a runtime slot index cost 6% -- dynamic addressing
    // on every ldmatrix); the inner body is one k-tile.
#pragma unroll 1
    for (int kt0 = 0; kt0 < k_tiles; kt0 += STAGES) {
#pragma unroll
    for (int s_ = 0; s_ < STAGES; s_++) {
        const int kt = kt0 + s_;
        if (kt >= k_tiles) break;
        if constexpr (!(ABL & 2)) cp_wait<STAGES - 2>();
        if constexpr (!(ABL & 4)) __syncthreads();
        if constexpr (!(ABL & 2)) {
            const int nxt = kt + STAGES - 1;
            if (nxt < k_tiles) stage_tile((s_ + STAGES - 1) % STAGES, nxt);
            else cp_commit();
        }
        const int slot = s_;
        const unsigned char* sw = smem + slot * STAGE_BYTES;
        const unsigned char* sx = sw + W_BYTES;
        const __half* sws = (const __half*)(sx + X_BYTES);
        const float* sxs = (const float*)(sx + X_BYTES + WS_BYTES);

#pragma unroll
        for (int g = 0; g < G; g++) {
            uint32_t A[MT][2][4];
            float wsc[MT][2];
#pragma unroll
            for (int i = 0; i < MT; i++) {
                const int row = wm * WM + i * 16 + a_row;
                const int chunk = g * 2 + a_chunk_hi;
                uint32_t p0, p1, p2, p3;
                if constexpr (ABL & 8) { p0 = (uint32_t)kt * 0x01010101u + lane; p1 = p0 ^ 0x5a5a5a5au; p2 = p0 + 7u; p3 = p1 + 3u; }
                else ldsm_x4(p0, p1, p2, p3, smem_u32(sw + row * WRB + swz<WCPR>(row, chunk) * 16));
                if constexpr (ABL & 32) {  // no unpack (wrong numerics, timing only)
                    A[i][0][0] = p0; A[i][0][1] = p1; A[i][0][2] = p0 ^ 0x0f0f0f0fu; A[i][0][3] = p1 ^ 0x0f0f0f0fu;
                    A[i][1][0] = p2; A[i][1][1] = p3; A[i][1][2] = p2 ^ 0x0f0f0f0fu; A[i][1][3] = p3 ^ 0x0f0f0f0fu;
                } else {
                A[i][0][0] = unpack_even(p0);
                A[i][0][1] = unpack_even(p1);
                A[i][0][2] = unpack_odd(p0);
                A[i][0][3] = unpack_odd(p1);
                A[i][1][0] = unpack_even(p2);
                A[i][1][1] = unpack_even(p3);
                A[i][1][2] = unpack_odd(p2);
                A[i][1][3] = unpack_odd(p3);
                }
                const int rr = wm * WM + i * 16 + gid;
                constexpr int GS = G == 1 ? 2 : G;  // smem halves per row (see stage_tile)
                const int gi = G == 1 ? (kt & 1) : g;
                if constexpr (ABL & 16) { wsc[i][0] = 0.01f * (kt + 1); wsc[i][1] = 0.02f * (kt + 1); }
                else {
                wsc[i][0] = __half2float(sws[rr * GS + gi]);
                wsc[i][1] = __half2float(sws[(rr + 8) * GS + gi]);
                }
                if constexpr (FOLD == 2) { wsc[i][0] *= 0.0625f; wsc[i][1] *= 0.0625f; }  // exact
            }
#pragma unroll
            for (int j = 0; j < NTL; j++) {
                const int trow = wn * WN + j * 8 + b_row;
                const int chunk = g * 4 + b_chunk;
                uint32_t b0, b1, b2, b3;
                if constexpr (ABL & 8) { b0 = (uint32_t)kt * 0x03030303u + lane * 5u + j; b1 = b0 ^ 0xa5a5a5a5u; b2 = b0 + 11u; b3 = b1 + 13u; }
                else ldsm_x4(b0, b1, b2, b3, smem_u32(sx + trow * XRB + swz<XCPR>(trow, chunk) * 16));
                const int tt = wn * WN + j * 8 + 2 * tg;
                float xs0, xs1;
                if constexpr (ABL & 16) { xs0 = 0.1f * (j + 1) + kt; xs1 = 0.2f * (j + 1) + kt; }
                else { const float2 v = *(const float2*)(sxs + g * BM + tt); xs0 = v.x; xs1 = v.y; }
#pragma unroll
                for (int i = 0; i < MT; i++) {
                    int d0, d1, d2, d3;
                    mma_s8_zero(d0, d1, d2, d3, A[i][0][0], A[i][0][1], A[i][0][2], A[i][0][3], b0,
                                b1);
                    mma_s8_acc(d0, d1, d2, d3, A[i][1][0], A[i][1][1], A[i][1][2], A[i][1][3], b2,
                               b3);
                    if constexpr (ABL & 1) {
                        acc[i][j][0] += __int_as_float(d0 ^ d1 ^ d2 ^ d3);
                    } else {
                    const float f0 = i2f16<FOLD>(d0), f1 = i2f16<FOLD>(d1);
                    const float f2 = i2f16<FOLD>(d2), f3 = i2f16<FOLD>(d3);
                    acc[i][j][0] += wsc[i][0] * xs0 * f0;
                    acc[i][j][1] += wsc[i][0] * xs1 * f1;
                    acc[i][j][2] += wsc[i][1] * xs0 * f2;
                    acc[i][j][3] += wsc[i][1] * xs1 * f3;
                    }
                }
            }
        }
    }
    }
    cp_wait<0>();

#pragma unroll
    for (int i = 0; i < MT; i++) {
        const int row0 = r0 + wm * WM + i * 16 + gid;
#pragma unroll
        for (int j = 0; j < NTL; j++) {
            const int tok0 = t0 + wn * WN + j * 8 + 2 * tg;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = row0 + (e >= 2 ? 8 : 0), tok = tok0 + (e & 1);
                if (row < rows && tok < T) y[(size_t)tok * rows + row] = acc[i][j][e];
            }
        }
    }
}

template <int BN, int BM, int BK, int STAGES, int WARPS_M, int WARPS_N, int FOLD, int MINB = 1, int ABL = 0>
void launch(const uint8_t* W, const __half* S, const int8_t* Xp, const float* xs, float* y,
            int rows, int cols, int T, cudaStream_t st) {
    constexpr int G = BK / 64;
    constexpr size_t SMEM =
        (size_t)STAGES * (BN * (BK / 2) + BM * BK + BN * (G == 1 ? 2 : G) * 2 + BM * G * 4);
    auto* kfn = k_w4a8<BN, BM, BK, STAGES, WARPS_M, WARPS_N, FOLD, MINB, ABL>;
    static bool attr = false;
    if (!attr) {
        CK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
        attr = true;
    }
    if (cols % BK) { fprintf(stderr, "cols %d %% BK %d\n", cols, BK); exit(1); }
    dim3 grid((T + BM - 1) / BM, (rows + BN - 1) / BN);
    kfn<<<grid, WARPS_M * WARPS_N * 32, SMEM, st>>>(W, S, Xp, xs, y, rows, cols, T);
    CK(cudaGetLastError());
}

// Lagged-fold variant: the fp32 fold of one 16-row tile step is deferred by
// one step (its raw int32 results and scales stay in registers), so the FFMAs
// of step n interleave with the IMMAs of step n+1 instead of running after
// the stage's last IMMA has drained -- with every warp at the barrier in
// lockstep, that exposed fold idled the tensor pipe once per stage. The lag
// crosses the stage barrier (the queued step is always m-tile MT-1 there).
// Numerics unchanged: each output's fold sequence is the same, only later.
// The very first fold runs on zeroed state (0*0*0 added to 0.0f), exact.
template <int BN, int BM, int BK, int STAGES, int WARPS_M, int WARPS_N, int MINB>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32, MINB)
k_w4a8_lag(const uint8_t* __restrict__ W, const __half* __restrict__ S,
           const int8_t* __restrict__ Xp, const float* __restrict__ xs, float* __restrict__ y,
           int rows, int cols, int T) {
    constexpr int NTHR = WARPS_M * WARPS_N * 32;
    constexpr int WM = BN / WARPS_M, WN = BM / WARPS_N;
    constexpr int MT = WM / 16, NTL = WN / 8;
    constexpr int G = BK / 64;
    constexpr int WRB = BK / 2, XRB = BK;
    constexpr int WCPR = WRB / 16, XCPR = XRB / 16;
    constexpr int W_BYTES = BN * WRB, X_BYTES = BM * XRB;
    constexpr int GS = G == 1 ? 2 : G;
    constexpr int WS_BYTES = BN * GS * 2, XS_BYTES = BM * G * 4;
    constexpr int STAGE_BYTES = W_BYTES + X_BYTES + WS_BYTES + XS_BYTES;
    static_assert(WM % 16 == 0 && WN % 8 == 0, "warp tile");
    static_assert(STAGE_BYTES % 128 == 0, "stage alignment");
    extern __shared__ __align__(128) unsigned char smem[];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WARPS_M, wn = warp / WARPS_M;
    const int gid = lane >> 2, tg = lane & 3;
    const int r0 = blockIdx.y * BN, t0 = blockIdx.x * BM;
    const int k_tiles = cols / BK;
    const int wrow = cols / 2;
    const int ngrp = cols / 64;

    auto stage_tile = [&](int slot, int kt) {
        unsigned char* sw = smem + slot * STAGE_BYTES;
        unsigned char* sx = sw + W_BYTES;
        unsigned char* sws = sx + X_BYTES;
        unsigned char* sxs = sws + WS_BYTES;
        const int k0 = kt * BK;
        for (int i = tid; i < BN * WCPR; i += NTHR) {
            const int r = i / WCPR, c = i % WCPR, gr = r0 + r;
            const bool ok = gr < rows;
            const uint8_t* src = ok ? W + (size_t)gr * wrow + k0 / 2 + c * 16 : W;
            cp16(sw + r * WRB + swz<WCPR>(r, c) * 16, src, ok);
        }
        for (int i = tid; i < BM * XCPR; i += NTHR) {
            const int r = i / XCPR, c = i % XCPR, gt = t0 + r;
            const bool ok = gt < T;
            const int8_t* src = ok ? Xp + (size_t)gt * cols + k0 + c * 16 : Xp;
            cp16(sx + r * XRB + swz<XCPR>(r, c) * 16, src, ok);
        }
        for (int i = tid; i < BN; i += NTHR) {
            const int gr = r0 + i;
            const bool ok = gr < rows;
            if constexpr (G == 1) {
                const __half* src = ok ? S + (size_t)gr * ngrp + (k0 / 64 & ~1) : S;
                cp4(sws + i * 4, src, ok);
            } else {
                const __half* src = ok ? S + (size_t)gr * ngrp + k0 / 64 : S;
                cpn<G * 2>(sws + i * G * 2, src, ok);
            }
        }
        for (int i = tid; i < BM; i += NTHR) {
            const int gt = t0 + i;
            const bool ok = gt < T;
            const float* src = ok ? xs + (size_t)gt * ngrp + k0 / 64 : xs;
            cpn<G * 4>(sxs + i * G * 4, src, ok);
        }
        cp_commit();
    };

    float acc[MT][NTL][4];
    int dq[NTL][4];       // queued step: raw 16*d
    float wq[2];          // queued step: row scales
    float xq[NTL][2];     // queued step: token scales
#pragma unroll
    for (int j = 0; j < NTL; j++) {
        xq[j][0] = xq[j][1] = 0.f;
#pragma unroll
        for (int e = 0; e < 4; e++) { dq[j][e] = 0; }
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
    }
    wq[0] = wq[1] = 0.f;

    auto fold = [&](int ip) {  // ip is a compile-time constant at every call site
#pragma unroll
        for (int j = 0; j < NTL; j++) {
            const float f0 = i2f16<true>(dq[j][0]), f1 = i2f16<true>(dq[j][1]);
            const float f2 = i2f16<true>(dq[j][2]), f3 = i2f16<true>(dq[j][3]);
            acc[ip][j][0] += wq[0] * xq[j][0] * f0;
            acc[ip][j][1] += wq[0] * xq[j][1] * f1;
            acc[ip][j][2] += wq[1] * xq[j][0] * f2;
            acc[ip][j][3] += wq[1] * xq[j][1] * f3;
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < k_tiles) stage_tile(s, s);
        else cp_commit();
    }
    const int a_row = lane & 15, a_chunk_hi = lane >> 4;
    const int b_row = lane & 7, b_chunk = lane >> 3;

    for (int kt = 0; kt < k_tiles; kt++) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nxt = kt + STAGES - 1;
            if (nxt < k_tiles) stage_tile(nxt % STAGES, nxt);
            else cp_commit();
        }
        const int slot = kt % STAGES;
        const unsigned char* sw = smem + slot * STAGE_BYTES;
        const unsigned char* sx = sw + W_BYTES;
        const __half* sws = (const __half*)(sx + X_BYTES);
        const float* sxs = (const float*)(sx + X_BYTES + WS_BYTES);

#pragma unroll
        for (int g = 0; g < G; g++) {
            uint32_t B[NTL][4];
            float xsv[NTL][2];
#pragma unroll
            for (int j = 0; j < NTL; j++) {
                const int trow = wn * WN + j * 8 + b_row;
                const int chunk = g * 4 + b_chunk;
                ldsm_x4(B[j][0], B[j][1], B[j][2], B[j][3],
                        smem_u32(sx + trow * XRB + swz<XCPR>(trow, chunk) * 16));
                const int tt = wn * WN + j * 8 + 2 * tg;
                xsv[j][0] = sxs[tt * G + g];
                xsv[j][1] = sxs[(tt + 1) * G + g];
            }
            const int gi = G == 1 ? (kt & 1) : g;
#pragma unroll
            for (int i = 0; i < MT; i++) {
                const int row = wm * WM + i * 16 + a_row;
                const int chunk = g * 2 + a_chunk_hi;
                uint32_t p0, p1, p2, p3;
                ldsm_x4(p0, p1, p2, p3, smem_u32(sw + row * WRB + swz<WCPR>(row, chunk) * 16));
                const uint32_t a00 = unpack_even(p0), a01 = unpack_even(p1);
                const uint32_t a02 = unpack_odd(p0), a03 = unpack_odd(p1);
                const uint32_t a10 = unpack_even(p2), a11 = unpack_even(p3);
                const uint32_t a12 = unpack_odd(p2), a13 = unpack_odd(p3);
                const int rr = wm * WM + i * 16 + gid;
                const float w0 = __half2float(sws[rr * GS + gi]);
                const float w1 = __half2float(sws[(rr + 8) * GS + gi]);
                int dn[NTL][4];
#pragma unroll
                for (int j = 0; j < NTL; j++) {
                    mma_s8_zero(dn[j][0], dn[j][1], dn[j][2], dn[j][3], a00, a01, a02, a03,
                                B[j][0], B[j][1]);
                    mma_s8_acc(dn[j][0], dn[j][1], dn[j][2], dn[j][3], a10, a11, a12, a13,
                               B[j][2], B[j][3]);
                }
                fold((i + MT - 1) % MT);  // previous step, while these IMMAs run
#pragma unroll
                for (int j = 0; j < NTL; j++) {
                    xq[j][0] = xsv[j][0]; xq[j][1] = xsv[j][1];
#pragma unroll
                    for (int e = 0; e < 4; e++) dq[j][e] = dn[j][e];
                }
                wq[0] = w0; wq[1] = w1;
            }
        }
    }
    cp_wait<0>();
    fold(MT - 1);  // the last queued step

#pragma unroll
    for (int i = 0; i < MT; i++) {
        const int row0 = r0 + wm * WM + i * 16 + gid;
#pragma unroll
        for (int j = 0; j < NTL; j++) {
            const int tok0 = t0 + wn * WN + j * 8 + 2 * tg;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = row0 + (e >= 2 ? 8 : 0), tok = tok0 + (e & 1);
                if (row < rows && tok < T) y[(size_t)tok * rows + row] = acc[i][j][e];
            }
        }
    }
}

// Lagged-fold variant 2 (ORDERED): same lag, but the fold of the queued
// step's tile j is issued right after tile j's two IMMAs, in asm order, so
// the compiler cannot hoist the IMMA bursts ahead of the folds.
// Original comment: the fp32 fold of one 16-row tile step is deferred by
// one step (its raw int32 results and scales stay in registers), so the FFMAs
// of step n interleave with the IMMAs of step n+1 instead of running after
// the stage's last IMMA has drained -- with every warp at the barrier in
// lockstep, that exposed fold idled the tensor pipe once per stage. The lag
// crosses the stage barrier (the queued step is always m-tile MT-1 there).
// Numerics unchanged: each output's fold sequence is the same, only later.
// The very first fold runs on zeroed state (0*0*0 added to 0.0f), exact.
template <int BN, int BM, int BK, int STAGES, int WARPS_M, int WARPS_N, int MINB>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32, MINB)
k_w4a8_lag2(const uint8_t* __restrict__ W, const __half* __restrict__ S,
           const int8_t* __restrict__ Xp, const float* __restrict__ xs, float* __restrict__ y,
           int rows, int cols, int T) {
    constexpr int NTHR = WARPS_M * WARPS_N * 32;
    constexpr int WM = BN / WARPS_M, WN = BM / WARPS_N;
    constexpr int MT = WM / 16, NTL = WN / 8;
    constexpr int G = BK / 64;
    constexpr int WRB = BK / 2, XRB = BK;
    constexpr int WCPR = WRB / 16, XCPR = XRB / 16;
    constexpr int W_BYTES = BN * WRB, X_BYTES = BM * XRB;
    constexpr int GS = G == 1 ? 2 : G;
    constexpr int WS_BYTES = BN * GS * 2, XS_BYTES = BM * G * 4;
    constexpr int STAGE_BYTES = W_BYTES + X_BYTES + WS_BYTES + XS_BYTES;
    static_assert(WM % 16 == 0 && WN % 8 == 0, "warp tile");
    static_assert(STAGE_BYTES % 128 == 0, "stage alignment");
    extern __shared__ __align__(128) unsigned char smem[];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WARPS_M, wn = warp / WARPS_M;
    const int gid = lane >> 2, tg = lane & 3;
    const int r0 = blockIdx.y * BN, t0 = blockIdx.x * BM;
    const int k_tiles = cols / BK;
    const int wrow = cols / 2;
    const int ngrp = cols / 64;

    auto stage_tile = [&](int slot, int kt) {
        unsigned char* sw = smem + slot * STAGE_BYTES;
        unsigned char* sx = sw + W_BYTES;
        unsigned char* sws = sx + X_BYTES;
        unsigned char* sxs = sws + WS_BYTES;
        const int k0 = kt * BK;
        for (int i = tid; i < BN * WCPR; i += NTHR) {
            const int r = i / WCPR, c = i % WCPR, gr = r0 + r;
            const bool ok = gr < rows;
            const uint8_t* src = ok ? W + (size_t)gr * wrow + k0 / 2 + c * 16 : W;
            cp16(sw + r * WRB + swz<WCPR>(r, c) * 16, src, ok);
        }
        for (int i = tid; i < BM * XCPR; i += NTHR) {
            const int r = i / XCPR, c = i % XCPR, gt = t0 + r;
            const bool ok = gt < T;
            const int8_t* src = ok ? Xp + (size_t)gt * cols + k0 + c * 16 : Xp;
            cp16(sx + r * XRB + swz<XCPR>(r, c) * 16, src, ok);
        }
        for (int i = tid; i < BN; i += NTHR) {
            const int gr = r0 + i;
            const bool ok = gr < rows;
            if constexpr (G == 1) {
                const __half* src = ok ? S + (size_t)gr * ngrp + (k0 / 64 & ~1) : S;
                cp4(sws + i * 4, src, ok);
            } else {
                const __half* src = ok ? S + (size_t)gr * ngrp + k0 / 64 : S;
                cpn<G * 2>(sws + i * G * 2, src, ok);
            }
        }
        for (int i = tid; i < BM; i += NTHR) {
            const int gt = t0 + i;
            const bool ok = gt < T;
            const float* src = ok ? xs + (size_t)gt * ngrp + k0 / 64 : xs;
            cpn<G * 4>(sxs + i * G * 4, src, ok);
        }
        cp_commit();
    };

    float acc[MT][NTL][4];
    int dq[NTL][4];       // queued step: raw 16*d
    float wq[2];          // queued step: row scales
    float xq[NTL][2];     // queued step: token scales
#pragma unroll
    for (int j = 0; j < NTL; j++) {
        xq[j][0] = xq[j][1] = 0.f;
#pragma unroll
        for (int e = 0; e < 4; e++) { dq[j][e] = 0; }
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
    }
    wq[0] = wq[1] = 0.f;

    auto fold = [&](int ip) {  // ip is a compile-time constant at every call site
#pragma unroll
        for (int j = 0; j < NTL; j++) {
            const float f0 = i2f16<true>(dq[j][0]), f1 = i2f16<true>(dq[j][1]);
            const float f2 = i2f16<true>(dq[j][2]), f3 = i2f16<true>(dq[j][3]);
            acc[ip][j][0] += wq[0] * xq[j][0] * f0;
            acc[ip][j][1] += wq[0] * xq[j][1] * f1;
            acc[ip][j][2] += wq[1] * xq[j][0] * f2;
            acc[ip][j][3] += wq[1] * xq[j][1] * f3;
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < k_tiles) stage_tile(s, s);
        else cp_commit();
    }
    const int a_row = lane & 15, a_chunk_hi = lane >> 4;
    const int b_row = lane & 7, b_chunk = lane >> 3;

    for (int kt = 0; kt < k_tiles; kt++) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nxt = kt + STAGES - 1;
            if (nxt < k_tiles) stage_tile(nxt % STAGES, nxt);
            else cp_commit();
        }
        const int slot = kt % STAGES;
        const unsigned char* sw = smem + slot * STAGE_BYTES;
        const unsigned char* sx = sw + W_BYTES;
        const __half* sws = (const __half*)(sx + X_BYTES);
        const float* sxs = (const float*)(sx + X_BYTES + WS_BYTES);

#pragma unroll
        for (int g = 0; g < G; g++) {
            uint32_t B[NTL][4];
            float xsv[NTL][2];
#pragma unroll
            for (int j = 0; j < NTL; j++) {
                const int trow = wn * WN + j * 8 + b_row;
                const int chunk = g * 4 + b_chunk;
                ldsm_x4(B[j][0], B[j][1], B[j][2], B[j][3],
                        smem_u32(sx + trow * XRB + swz<XCPR>(trow, chunk) * 16));
                const int tt = wn * WN + j * 8 + 2 * tg;
                xsv[j][0] = sxs[tt * G + g];
                xsv[j][1] = sxs[(tt + 1) * G + g];
            }
            const int gi = G == 1 ? (kt & 1) : g;
#pragma unroll
            for (int i = 0; i < MT; i++) {
                const int row = wm * WM + i * 16 + a_row;
                const int chunk = g * 2 + a_chunk_hi;
                uint32_t p0, p1, p2, p3;
                ldsm_x4(p0, p1, p2, p3, smem_u32(sw + row * WRB + swz<WCPR>(row, chunk) * 16));
                const uint32_t a00 = unpack_even(p0), a01 = unpack_even(p1);
                const uint32_t a02 = unpack_odd(p0), a03 = unpack_odd(p1);
                const uint32_t a10 = unpack_even(p2), a11 = unpack_even(p3);
                const uint32_t a12 = unpack_odd(p2), a13 = unpack_odd(p3);
                const int rr = wm * WM + i * 16 + gid;
                const float w0 = __half2float(sws[rr * GS + gi]);
                const float w1 = __half2float(sws[(rr + 8) * GS + gi]);
                int dn[NTL][4];
                constexpr int ip_dummy = 0; (void)ip_dummy;
#pragma unroll
                for (int j = 0; j < NTL; j++) {
                    mma_s8_zero(dn[j][0], dn[j][1], dn[j][2], dn[j][3], a00, a01, a02, a03,
                                B[j][0], B[j][1]);
                    mma_s8_acc(dn[j][0], dn[j][1], dn[j][2], dn[j][3], a10, a11, a12, a13,
                               B[j][2], B[j][3]);
                    // fold the queued step's tile j (its IMMAs were a full step ago)
                    const int ip = (i + MT - 1) % MT;
                    const float f0 = vi2f16(dq[j][0]), f1 = vi2f16(dq[j][1]);
                    const float f2 = vi2f16(dq[j][2]), f3 = vi2f16(dq[j][3]);
                    const float t00 = vmul(wq[0], xq[j][0]), t01 = vmul(wq[0], xq[j][1]);
                    const float t10 = vmul(wq[1], xq[j][0]), t11 = vmul(wq[1], xq[j][1]);
                    acc[ip][j][0] = vfma(t00, f0, acc[ip][j][0]);
                    acc[ip][j][1] = vfma(t01, f1, acc[ip][j][1]);
                    acc[ip][j][2] = vfma(t10, f2, acc[ip][j][2]);
                    acc[ip][j][3] = vfma(t11, f3, acc[ip][j][3]);
                }
#pragma unroll
                for (int j = 0; j < NTL; j++) {
                    xq[j][0] = xsv[j][0]; xq[j][1] = xsv[j][1];
#pragma unroll
                    for (int e = 0; e < 4; e++) dq[j][e] = dn[j][e];
                }
                wq[0] = w0; wq[1] = w1;
            }
        }
    }
    cp_wait<0>();
    fold(MT - 1);  // the last queued step

#pragma unroll
    for (int i = 0; i < MT; i++) {
        const int row0 = r0 + wm * WM + i * 16 + gid;
#pragma unroll
        for (int j = 0; j < NTL; j++) {
            const int tok0 = t0 + wn * WN + j * 8 + 2 * tg;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = row0 + (e >= 2 ? 8 : 0), tok = tok0 + (e & 1);
                if (row < rows && tok < T) y[(size_t)tok * rows + row] = acc[i][j][e];
            }
        }
    }
}

template <int BN, int BM, int BK, int STAGES, int WARPS_M, int WARPS_N, int MINB = 1>
void launch_lag2(const uint8_t* W, const __half* S, const int8_t* Xp, const float* xs, float* y,
                 int rows, int cols, int T, cudaStream_t st) {
    constexpr int G = BK / 64;
    constexpr size_t SMEM =
        (size_t)STAGES * (BN * (BK / 2) + BM * BK + BN * (G == 1 ? 2 : G) * 2 + BM * G * 4);
    auto* kfn = k_w4a8_lag2<BN, BM, BK, STAGES, WARPS_M, WARPS_N, MINB>;
    static bool attr = false;
    if (!attr) {
        CK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
        attr = true;
    }
    dim3 grid((T + BM - 1) / BM, (rows + BN - 1) / BN);
    kfn<<<grid, WARPS_M * WARPS_N * 32, SMEM, st>>>(W, S, Xp, xs, y, rows, cols, T);
    CK(cudaGetLastError());
}

template <int BN, int BM, int BK, int STAGES, int WARPS_M, int WARPS_N, int MINB = 1>
void launch_lag(const uint8_t* W, const __half* S, const int8_t* Xp, const float* xs, float* y,
                int rows, int cols, int T, cudaStream_t st) {
    constexpr int G = BK / 64;
    constexpr size_t SMEM =
        (size_t)STAGES * (BN * (BK / 2) + BM * BK + BN * (G == 1 ? 2 : G) * 2 + BM * G * 4);
    auto* kfn = k_w4a8_lag<BN, BM, BK, STAGES, WARPS_M, WARPS_N, MINB>;
    static bool attr = false;
    if (!attr) {
        CK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
        attr = true;
    }
    if (cols % BK) { fprintf(stderr, "cols %d %% BK %d\n", cols, BK); exit(1); }
    dim3 grid((T + BM - 1) / BM, (rows + BN - 1) / BN);
    kfn<<<grid, WARPS_M * WARPS_N * 32, SMEM, st>>>(W, S, Xp, xs, y, rows, cols, T);
    CK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Warp-specialized variant: one PRODUCER warp streams the stage ring with
// cp.async and signals a per-slot "full" mbarrier (cp.async.mbarrier.arrive:
// the arrive fires when that lane's copies have landed); WARPS_M x WARPS_N
// CONSUMER warps wait on "full", compute, and arrive on the slot's "empty"
// mbarrier, which the producer waits on before refilling. No __syncthreads in
// the loop: consumers never wait for each other, only for data. Measured
// motivation (09-08): with the fill removed the 16-warp kernel computes at
// 609 TOPS, with it 428 -- the block-wide barrier + everyone-copies structure
// serialised fill and math.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_cp_arrive(uint64_t* bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned parity) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}

template <int BN, int BM, int BK, int STAGES, int WARPS_M, int WARPS_N, int FOLD>
__global__ void __launch_bounds__((WARPS_M* WARPS_N + 1) * 32, 1)
k_w4a8_ws(const uint8_t* __restrict__ W, const __half* __restrict__ S,
          const int8_t* __restrict__ Xp, const float* __restrict__ xs, float* __restrict__ y,
          int rows, int cols, int T) {
    constexpr int NCW = WARPS_M * WARPS_N;  // consumer warps
    constexpr int WM = BN / WARPS_M, WN = BM / WARPS_N;
    constexpr int MT = WM / 16, NTL = WN / 8;
    constexpr int G = BK / 64;
    static_assert(G >= 2, "ws variant: BK >= 128");
    constexpr int WRB = BK / 2, XRB = BK;
    constexpr int WCPR = WRB / 16, XCPR = XRB / 16;
    constexpr int W_BYTES = BN * WRB, X_BYTES = BM * XRB;
    constexpr int WS_BYTES = BN * G * 2, XS_BYTES = BM * G * 4;
    constexpr int STAGE_BYTES = W_BYTES + X_BYTES + WS_BYTES + XS_BYTES;
    static_assert(STAGE_BYTES % 128 == 0, "stage alignment");
    extern __shared__ __align__(128) unsigned char smem[];
    uint64_t* full = (uint64_t*)(smem + STAGES * STAGE_BYTES);
    uint64_t* empty = full + STAGES;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r0 = blockIdx.y * BN, t0 = blockIdx.x * BM;
    const int k_tiles = cols / BK;
    const int wrow = cols / 2, ngrp = cols / 64;

    if (tid == 0) {
#pragma unroll
        for (int s_ = 0; s_ < STAGES; s_++) { mbar_init(&full[s_], 32); mbar_init(&empty[s_], NCW); }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (warp == NCW) {
        // ---------------- producer warp ----------------
        for (int f = 0; f < k_tiles; f++) {
            const int slot = f % STAGES, round = f / STAGES;
            if (f >= STAGES) mbar_wait(&empty[slot], (round - 1) & 1);
            unsigned char* sw = smem + slot * STAGE_BYTES;
            unsigned char* sx = sw + W_BYTES;
            unsigned char* sws = sx + X_BYTES;
            unsigned char* sxs = sws + WS_BYTES;
            const int k0 = f * BK;
            for (int i = lane; i < BN * WCPR; i += 32) {
                const int r = i / WCPR, c = i % WCPR, gr = r0 + r;
                const bool ok = gr < rows;
                const uint8_t* src = ok ? W + (size_t)gr * wrow + k0 / 2 + c * 16 : W;
                cp16(sw + r * WRB + swz<WCPR>(r, c) * 16, src, ok);
            }
            for (int i = lane; i < BM * XCPR; i += 32) {
                const int r = i / XCPR, c = i % XCPR, gt = t0 + r;
                const bool ok = gt < T;
                const int8_t* src = ok ? Xp + (size_t)gt * cols + k0 + c * 16 : Xp;
                cp16(sx + r * XRB + swz<XCPR>(r, c) * 16, src, ok);
            }
            for (int i = lane; i < BN; i += 32) {
                const int gr = r0 + i;
                const bool ok = gr < rows;
                const __half* src = ok ? S + (size_t)gr * ngrp + k0 / 64 : S;
                cpn<G * 2>(sws + i * G * 2, src, ok);
            }
            for (int i = lane; i < BM; i += 32) {
                const int gt = t0 + i;
                const bool ok = gt < T;
                const float* src = ok ? xs + (size_t)gt * ngrp + k0 / 64 : xs;
                cpn<G * 4>(sxs + i * G * 4, src, ok);
            }
            mbar_cp_arrive(&full[slot]);
        }
        return;
    }

    // ---------------- consumer warps ----------------
    const int wm = warp % WARPS_M, wn = warp / WARPS_M;
    const int gid = lane >> 2, tg = lane & 3;
    const int a_row = lane & 15, a_chunk_hi = lane >> 4;
    const int b_row = lane & 7, b_chunk = lane >> 3;
    float acc[MT][NTL][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NTL; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

#pragma unroll 1
    for (int kt0 = 0; kt0 < k_tiles; kt0 += STAGES) {
#pragma unroll
    for (int s_ = 0; s_ < STAGES; s_++) {
        const int kt = kt0 + s_;
        if (kt >= k_tiles) break;
        mbar_wait(&full[s_], (kt / STAGES) & 1);
        const unsigned char* sw = smem + s_ * STAGE_BYTES;
        const unsigned char* sx = sw + W_BYTES;
        const __half* sws = (const __half*)(sx + X_BYTES);
        const float* sxs = (const float*)(sx + X_BYTES + WS_BYTES);
#pragma unroll
        for (int g = 0; g < G; g++) {
            uint32_t A[MT][2][4];
            float wsc[MT][2];
#pragma unroll
            for (int i = 0; i < MT; i++) {
                const int row = wm * WM + i * 16 + a_row;
                const int chunk = g * 2 + a_chunk_hi;
                uint32_t p0, p1, p2, p3;
                ldsm_x4(p0, p1, p2, p3, smem_u32(sw + row * WRB + swz<WCPR>(row, chunk) * 16));
                A[i][0][0] = unpack_even(p0); A[i][0][1] = unpack_even(p1);
                A[i][0][2] = unpack_odd(p0);  A[i][0][3] = unpack_odd(p1);
                A[i][1][0] = unpack_even(p2); A[i][1][1] = unpack_even(p3);
                A[i][1][2] = unpack_odd(p2);  A[i][1][3] = unpack_odd(p3);
                const int rr = wm * WM + i * 16 + gid;
                wsc[i][0] = __half2float(sws[rr * G + g]);
                wsc[i][1] = __half2float(sws[(rr + 8) * G + g]);
                if constexpr (FOLD == 2) { wsc[i][0] *= 0.0625f; wsc[i][1] *= 0.0625f; }
            }
#pragma unroll
            for (int j = 0; j < NTL; j++) {
                const int trow = wn * WN + j * 8 + b_row;
                const int chunk = g * 4 + b_chunk;
                uint32_t b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, smem_u32(sx + trow * XRB + swz<XCPR>(trow, chunk) * 16));
                const int tt = wn * WN + j * 8 + 2 * tg;
                const float xs0 = sxs[tt * G + g], xs1 = sxs[(tt + 1) * G + g];
#pragma unroll
                for (int i = 0; i < MT; i++) {
                    int d0, d1, d2, d3;
                    mma_s8_zero(d0, d1, d2, d3, A[i][0][0], A[i][0][1], A[i][0][2], A[i][0][3], b0, b1);
                    mma_s8_acc(d0, d1, d2, d3, A[i][1][0], A[i][1][1], A[i][1][2], A[i][1][3], b2, b3);
                    const float f0 = i2f16<FOLD>(d0), f1 = i2f16<FOLD>(d1);
                    const float f2 = i2f16<FOLD>(d2), f3 = i2f16<FOLD>(d3);
                    acc[i][j][0] += wsc[i][0] * xs0 * f0;
                    acc[i][j][1] += wsc[i][0] * xs1 * f1;
                    acc[i][j][2] += wsc[i][1] * xs0 * f2;
                    acc[i][j][3] += wsc[i][1] * xs1 * f3;
                }
            }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[s_]);
    }
    }

#pragma unroll
    for (int i = 0; i < MT; i++) {
        const int row0 = r0 + wm * WM + i * 16 + gid;
#pragma unroll
        for (int j = 0; j < NTL; j++) {
            const int tok0 = t0 + wn * WN + j * 8 + 2 * tg;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = row0 + (e >= 2 ? 8 : 0), tok = tok0 + (e & 1);
                if (row < rows && tok < T) y[(size_t)tok * rows + row] = acc[i][j][e];
            }
        }
    }
}

template <int BN, int BM, int BK, int STAGES, int WARPS_M, int WARPS_N, int FOLD>
void launch_ws(const uint8_t* W, const __half* S, const int8_t* Xp, const float* xs, float* y,
               int rows, int cols, int T, cudaStream_t st) {
    constexpr int G = BK / 64;
    constexpr size_t SMEM = (size_t)STAGES * (BN * (BK / 2) + BM * BK + BN * G * 2 + BM * G * 4) + 2 * STAGES * 8;
    auto* kfn = k_w4a8_ws<BN, BM, BK, STAGES, WARPS_M, WARPS_N, FOLD>;
    static bool attr = false;
    if (!attr) {
        CK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM));
        attr = true;
    }
    dim3 grid((T + BM - 1) / BM, (rows + BN - 1) / BN);
    kfn<<<grid, (WARPS_M * WARPS_N + 1) * 32, SMEM, st>>>(W, S, Xp, xs, y, rows, cols, T);
    CK(cudaGetLastError());
}

}  // namespace spike

// ---------------------------------------------------------------------------
struct Variant {
    const char* name;
    void (*fn)(const uint8_t*, const __half*, const int8_t*, const float*, float*, int, int, int,
               cudaStream_t);
};
#define V(name, BN, BM, BK, ST, WM, WN, MG) \
    Variant{name, spike::launch<BN, BM, BK, ST, WM, WN, MG, 1>}
#define VF(name, BN, BM, BK, ST, WM, WN, FOLD) \
    Variant{name, spike::launch<BN, BM, BK, ST, WM, WN, FOLD, 1>}
#define V2(name, BN, BM, BK, ST, WM, WN, MG) \
    Variant{name, spike::launch<BN, BM, BK, ST, WM, WN, MG, 2>}
#define VL(name, BN, BM, BK, ST, WM, WN) \
    Variant{name, spike::launch_lag<BN, BM, BK, ST, WM, WN, 1>}
#define VL2(name, BN, BM, BK, ST, WM, WN) \
    Variant{name, spike::launch_lag2<BN, BM, BK, ST, WM, WN, 1>}
#define VA(name, ABL) Variant{name, spike::launch<128, 128, 128, 2, 4, 2, 1, 1, ABL>}
#define VW(name, BN, BM, BK, ST, WM, WN, FOLD) \
    Variant{name, spike::launch_ws<BN, BM, BK, ST, WM, WN, FOLD>}
#define VA256(name, ABL) Variant{name, spike::launch<256, 128, 128, 2, 4, 2, 1, 1, ABL>}
#define VA44(name, ABL) Variant{name, spike::launch<128, 128, 128, 2, 4, 4, 1, 1, ABL>}

static const Variant variants[] = {
    VF("128x128x128 s2 4x2 magic", 128, 128, 128, 2, 4, 2, 1),
    VF("128x128x128 s2 4x4 magic", 128, 128, 128, 2, 4, 4, 1),
    VW("ws 128x128x128 s2 4x2 magic", 128, 128, 128, 2, 4, 2, 1),
    VW("ws 128x128x128 s3 4x2 magic", 128, 128, 128, 3, 4, 2, 1),
    VW("ws 128x128x128 s3 4x2 cvt16", 128, 128, 128, 3, 4, 2, 2),
    VW("ws 128x128x128 s2 4x4 magic", 128, 128, 128, 2, 4, 4, 1),
    VW("ws 128x128x128 s3 4x4 magic", 128, 128, 128, 3, 4, 4, 1),
    VW("ws 256x128x128 s2 4x2 magic", 256, 128, 128, 2, 4, 2, 1),
};

struct Shape { const char* name; int rows, cols; };
static const Shape shapes[] = {
    {"ffn_gate", 17408, 5120}, {"attn_out", 5120, 8192}, {"ffn_down", 5120, 17408},
    {"attn_qkvg", 16384, 5120},
};

static double time_ms(const std::function<void()>& f, int reps) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < 3; i++) f();
    CK(cudaEventRecord(e0));
    for (int i = 0; i < reps; i++) f();
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    return ms / reps;
}

int main(int argc, char** argv) {
    bool do_time = false, do_test = true; std::string only, onlyv;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--time")) do_time = true;
        else if (!strcmp(argv[i], "--shape") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) onlyv = argv[++i];
        else if (!strcmp(argv[i], "--notest")) do_test = false;
    }
    int dev = 0; CK(cudaGetDevice(&dev));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    printf("device %s sm_%d%d %d SMs smem/block %zu\n", p.name, p.major, p.minor,
           p.multiProcessorCount, p.sharedMemPerBlockOptin);

    const int Tmax = 4096;
    int fails = 0;
    for (const Shape& sh : shapes) {
        if (!only.empty() && only != sh.name) continue;
        const int rows = sh.rows, cols = sh.cols;
        // synthetic Q4_G64 weights + fp16 scales (positive, ~1e-2 spread)
        std::vector<uint8_t> hw((size_t)rows * cols / 2);
        for (auto& v : hw) v = (uint8_t)(rand() & 0xff);
        std::vector<__half> hs((size_t)rows * cols / 64);
        for (auto& v : hs) v = __float2half(0.002f + 0.03f * (rand() / (float)RAND_MAX));
        uint8_t* W; __half* S;
        CK(cudaMalloc(&W, hw.size())); CK(cudaMalloc(&S, hs.size() * 2));
        CK(cudaMemcpy(W, hw.data(), hw.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(S, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice));
        // activations: fp32 -> real quantize_x_g64 -> permute
        std::vector<float> hx((size_t)Tmax * cols);
        for (auto& v : hx) v = (rand() / (float)RAND_MAX - 0.5f) * 4.f;
        float* x; CK(cudaMalloc(&x, hx.size() * 4));
        CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        q27k::XQuant xq = q27k::xquant_alloc((int64_t)Tmax * cols, true);
        q27k::quantize_x_g64(x, (int64_t)Tmax * cols, xq, 0);
        int8_t* xp; CK(cudaMalloc(&xp, (size_t)Tmax * cols));
        {
            const size_t n32 = (size_t)Tmax * cols / 32;
            spike::k_permute_x<<<(unsigned)((n32 + 255) / 256), 256>>>(xq.nat64, xp, n32);
            CK(cudaGetLastError());
        }
        float *yref, *ynew;
        CK(cudaMalloc(&yref, (size_t)Tmax * rows * 4));
        CK(cudaMalloc(&ynew, (size_t)Tmax * rows * 4));
        std::vector<float> href((size_t)Tmax * rows), hnew((size_t)Tmax * rows);

        // correctness: T values covering full tiles, token tails, tiny T
        const int Ts[] = {1024, 1000, 512, 257, 96, 37, 1, 4096};
        for (int T : Ts) {
            if (!do_test) break;
            CK(cudaMemset(yref, 0xff, (size_t)T * rows * 4));
            q27k::gemm_q4_T(W, S, xq, yref, rows, cols, T, 0, nullptr);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(href.data(), yref, (size_t)T * rows * 4, cudaMemcpyDeviceToHost));
            for (const Variant& v : variants) {
                if (!onlyv.empty() && !strstr(v.name, onlyv.c_str())) continue;
                if (!strncmp(v.name, "abl", 3)) continue;  // timing-only
                CK(cudaMemset(ynew, 0xff, (size_t)T * rows * 4));
                v.fn(W, S, xp, xq.s64, ynew, rows, cols, T, 0);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(hnew.data(), ynew, (size_t)T * rows * 4, cudaMemcpyDeviceToHost));
                size_t nd = 0, first = (size_t)-1;
                for (size_t i = 0; i < (size_t)T * rows; i++)
                    if (memcmp(&href[i], &hnew[i], 4)) { nd++; if (first == (size_t)-1) first = i; }
                if (nd) {
                    fails++;
                    printf("MISMATCH %-9s T=%-5d %-28s %zu/%zu differ; first at tok %zu row %zu: ref %.9g new %.9g\n",
                           sh.name, T, v.name, nd, (size_t)T * rows, first / rows, first % rows,
                           href[first], hnew[first]);
                }
            }
        }
        printf("%-9s (%d x %d): bitwise vs gemm_q4_T over T in {1024,1000,512,257,96,37,1,4096}: %s\n",
               sh.name, rows, cols, fails ? "SEE ABOVE" : "ALL VARIANTS IDENTICAL");

        if (do_time) {
            for (int T : {1024, 4096}) {
                const double fl = 2.0 * T * rows * cols;
                const int reps = T == 4096 ? 10 : 20;
                const double t_ref = time_ms([&] {
                    q27k::gemm_q4_T(W, S, xq, yref, rows, cols, T, 0, nullptr); }, reps);
                printf("  T=%-5d %-28s %8.3f ms %7.1f TOPS\n", T, "incumbent (live dispatch)", t_ref,
                       fl / (t_ref * 1e9));
                for (const Variant& v : variants) {
                    if (!onlyv.empty() && !strstr(v.name, onlyv.c_str())) continue;
                    const double t = time_ms([&] { v.fn(W, S, xp, xq.s64, ynew, rows, cols, T, 0); },
                                             reps);
                    printf("  T=%-5d %-28s %8.3f ms %7.1f TOPS  %5.2fx\n", T, v.name, t,
                           fl / (t * 1e9), t_ref / t);
                }
            }
        }
        CK(cudaFree(W)); CK(cudaFree(S)); CK(cudaFree(x)); CK(cudaFree(xp));
        CK(cudaFree(yref)); CK(cudaFree(ynew));
        CK(cudaFree(xq.nat)); CK(cudaFree(xq.eo)); CK(cudaFree(xq.scale)); CK(cudaFree(xq.isum));
        CK(cudaFree(xq.nat64)); CK(cudaFree(xq.s64));
    }
    if (fails) { printf("%d MISMATCH(ES)\n", fails); return 1; }
    printf("all bitwise\n");
    return 0;
}
