#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "blocks.cuh"
#include "cuda_common.h"
#include "prefill.cuh"

namespace q27k {

static __device__ __forceinline__ float warp_reduce_f(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// ---------------- batched GEMM ----------------
// Warp per row; token tile of TB accumulators so each 16-byte weight chunk is
// dp4a'd against TB tokens' activations (weight DRAM traffic /TB; activation
// tile stays L2-resident).

// reports the __CUDA_ARCH__ of the image actually loaded on this device --
// the honest capability signal for arch-conditional instruction paths (an
// sm_89 device running the sm_86 image must NOT enable sm_89-only code)
__global__ void k_arch_probe(int* out) {
#if defined(__CUDA_ARCH__)
    *out = __CUDA_ARCH__; // 860, 890, 1200, ...
#else
    *out = 0;
#endif
}

template <int TB, int CS>
__global__ void k_gemm_q4_T(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                            const uint2* __restrict__ eo, const float* __restrict__ xs,
                            const int* __restrict__ xisum, float* __restrict__ y,
                            int64_t rows, int64_t cols, int T, int t0) {
    // RB warps = RB rows per block; a CS-chunk x TB-token activation tile is
    // staged in dynamic shared memory and reused by all rows. Per-lane chunk
    // order (c0+lane, c0+lane+32, ...) matches the serial k_gemv_q4 walk, so
    // sums are bitwise-identical to the single-token path.
    constexpr int RB = 16;
    constexpr int EOP = TB * 4 + 1, XSP = TB + 1; // padded rows (bank conflicts)
    extern __shared__ unsigned char smem_raw[];
    uint2* s_eo = (uint2*)smem_raw;
    float* s_xs = (float*)(s_eo + CS * EOP);
    int* s_is = (int*)(s_xs + CS * XSP);

    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    int64_t row = (int64_t)blockIdx.x * RB + warp;
    const int nt = min(TB, T - t0);
    const int n_chunks = (int)(cols / 32);
    const size_t ept = (size_t)n_chunks * 4;
    const uint4* wr = row < rows ? (const uint4*)(W + row * (cols / 2)) : nullptr;
    const __half* sr = row < rows ? S + row * (cols / 64) : nullptr;

    float acc[TB];
#pragma unroll
    for (int i = 0; i < TB; i++) acc[i] = 0.f;

    for (int c0 = 0; c0 < n_chunks; c0 += CS) {
        __syncthreads();
        for (int idx = threadIdx.x; idx < CS * TB * 4; idx += blockDim.x) {
            int u = idx & 3, r = idx >> 2, tt = r % TB, cc = r / TB;
            s_eo[cc * EOP + tt * 4 + u] =
                (c0 + cc < n_chunks && tt < nt)  // tt<nt guards the OOB read past T (CUDA-review #5)
                    ? __ldg(eo + (size_t)(t0 + tt) * ept + (size_t)(c0 + cc) * 4 + u)
                    : make_uint2(0, 0);
        }
        for (int idx = threadIdx.x; idx < CS * TB; idx += blockDim.x) {
            int tt = idx % TB, cc = idx / TB;
            bool ok = c0 + cc < n_chunks && tt < nt;
            s_xs[cc * XSP + tt] =
                ok ? __ldg(xs + (size_t)(t0 + tt) * n_chunks + c0 + cc) : 0.f;
            s_is[cc * XSP + tt] =
                ok ? __ldg(xisum + (size_t)(t0 + tt) * n_chunks + c0 + cc) : 0;
        }
        __syncthreads();
        if (!wr) continue;
#pragma unroll
        for (int cc = lane; cc < CS; cc += 32) {
            const int ch = c0 + cc;
            if (ch >= n_chunks) break;
            uint4 w = __ldg(wr + ch);
            float ws = __half2float(__ldg(sr + (ch >> 1)));
            const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int tt = 0; tt < TB; tt++) {
                if (tt >= nt) break;
                int di = 0;
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    uint2 xv = s_eo[cc * EOP + tt * 4 + u];
                    di = __dp4a((int)(wv[u] & 0x0F0F0F0Fu), (int)xv.x, di);
                    di = __dp4a((int)((wv[u] >> 4) & 0x0F0F0F0Fu), (int)xv.y, di);
                }
                acc[tt] += ws * s_xs[cc * XSP + tt] * (float)(di - 8 * s_is[cc * XSP + tt]);
            }
        }
    }
    if (!wr) return;
#pragma unroll
    for (int i = 0; i < TB; i++) {
        float v = warp_reduce_f(acc[i]);
        if (lane == 0 && i < nt) y[(size_t)(t0 + i) * rows + row] = v;
    }
}

template <int TB, int CS>
__global__ void k_gemm_q8_T(const int8_t* __restrict__ W, const __half* __restrict__ S,
                            const int8_t* __restrict__ nat, const float* __restrict__ xs,
                            float* __restrict__ y, int64_t rows, int64_t cols, int T, int t0) {
    constexpr int RB = 16;
    constexpr int XP = TB * 2 + 1, XSP = TB + 1;
    extern __shared__ unsigned char smem_raw[];
    uint4* s_x = (uint4*)smem_raw;
    float* s_xs = (float*)(s_x + CS * XP);

    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    int64_t row = (int64_t)blockIdx.x * RB + warp;
    const int nt = min(TB, T - t0);
    const int n_chunks = (int)(cols / 32);
    const uint4* wr = row < rows ? (const uint4*)(W + row * cols) : nullptr;
    const __half* sr = row < rows ? S + row * (cols / 128) : nullptr;

    float acc[TB];
#pragma unroll
    for (int i = 0; i < TB; i++) acc[i] = 0.f;

    for (int c0 = 0; c0 < n_chunks; c0 += CS) {
        __syncthreads();
        for (int idx = threadIdx.x; idx < CS * TB * 2; idx += blockDim.x) {
            int u = idx & 1, r = idx >> 1, tt = r % TB, cc = r / TB;
            s_x[cc * XP + tt * 2 + u] =
                (c0 + cc < n_chunks && tt < nt)  // tt<nt guards the OOB read past T (CUDA-review #5)
                    ? __ldg((const uint4*)(nat + (size_t)(t0 + tt) * cols) + 2 * (c0 + cc) + u)
                    : make_uint4(0, 0, 0, 0);
        }
        for (int idx = threadIdx.x; idx < CS * TB; idx += blockDim.x) {
            int tt = idx % TB, cc = idx / TB;
            s_xs[cc * XSP + tt] =
                (c0 + cc < n_chunks && tt < nt) ? __ldg(xs + (size_t)(t0 + tt) * n_chunks + c0 + cc) : 0.f;
        }
        __syncthreads();
        if (!wr) continue;
#pragma unroll
        for (int cc = lane; cc < CS; cc += 32) {
            const int ch = c0 + cc;
            if (ch >= n_chunks) break;
            uint4 w0 = __ldg(wr + 2 * ch), w1 = __ldg(wr + 2 * ch + 1);
            float ws = __half2float(__ldg(sr + (ch >> 2)));
#pragma unroll
            for (int tt = 0; tt < TB; tt++) {
                if (tt >= nt) break;
                uint4 x0 = s_x[cc * XP + tt * 2], x1 = s_x[cc * XP + tt * 2 + 1];
                int di = 0;
                di = __dp4a((int)w0.x, (int)x0.x, di);
                di = __dp4a((int)w0.y, (int)x0.y, di);
                di = __dp4a((int)w0.z, (int)x0.z, di);
                di = __dp4a((int)w0.w, (int)x0.w, di);
                di = __dp4a((int)w1.x, (int)x1.x, di);
                di = __dp4a((int)w1.y, (int)x1.y, di);
                di = __dp4a((int)w1.z, (int)x1.z, di);
                di = __dp4a((int)w1.w, (int)x1.w, di);
                acc[tt] += ws * s_xs[cc * XSP + tt] * (float)di;
            }
        }
    }
    if (!wr) return;
#pragma unroll
    for (int i = 0; i < TB; i++) {
        float v = warp_reduce_f(acc[i]);
        if (lane == 0 && i < nt) y[(size_t)(t0 + i) * rows + row] = v;
    }
}

// ---------------- int8 tensor-core GEMM (P1) ----------------
// mma.sync m16n8k32: each MMA covers exactly one 32-element activation quant
// block, so the int32 dot per (row, token, chunk) equals the dp4a path's
// integer exactly (Q4 nibbles are unpacked to s8 with the -8 offset folded in
// during staging). The fp scale-and-add per chunk has a different ACCUMULATION
// ORDER than the serial GEMV's lane-partial tree, so outputs differ from the
// dp4a path by fp rounding only (~1e-6 rel) -- gated by unit test + --pfdbg +
// full-corpus --nll, with Q27_PREFILL=dp4a keeping the exact path available.
//
// Tile: block = 256 threads = 8 warps as 4(M) x 2(N); block tile 64 rows x
// 32 tokens; K staged 128 elements at a time in smem.

static __device__ __forceinline__ void mma_s8(int& d0, int& d1, int& d2, int& d3, uint32_t a0,
                                              uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                              uint32_t b1) {
    const int z = 0;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(z), "r"(z), "r"(z), "r"(z));
}

// Accumulating form (c = d): lets two K=32 steps of a 64-group chain in int32
// so the fp dequant step runs once per 64 (the g64 activation-regroup path).
static __device__ __forceinline__ void mma_s8_acc(int& d0, int& d1, int& d2, int& d3, uint32_t a0,
                                                  uint32_t a1, uint32_t a2, uint32_t a3,
                                                  uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Q4IN: weights arrive nibble-packed (Q4_G64, scale per 64) and are unpacked
// to s8 in staging; otherwise s8 rows (Q8_G128, scale per 128).
// XG64: activations quantized per-64 (nat64/s64 -- aligned with the Q4 weight
// group), so two K=32 mmas chain in int32 before ONE fp dequant step. NOT
// serial-vs-batched identical (per-64 amax changes the int8 values vs the
// decode path's per-32) -- gated by tolerance + PPL + canonical instead of
// the pf-identity gate (policy sign-off 2026-07-04). XG64=false is the exact
// legacy path, bit-identical to the pre-regroup kernel.
template <bool Q4IN, bool XG64>
__global__ void k_gemm_mma_T(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                             const int8_t* __restrict__ nat, const float* __restrict__ xs,
                             float* __restrict__ y, int64_t rows, int64_t cols, int T) {
    constexpr int MR = 64, NT = 128, KS = 128;  // block tile: rows, tokens, staged K
    constexpr int XGS = XG64 ? 64 : 32;         // activation quant group
    constexpr int XSC = KS / XGS;               // x-scales per token per stage
    // NT=128 (was 64): doubles A-fragment reuse per staged weight byte at
    // T=1024 chunks -- measured 242 -> ? TOPS on the 16K pf bench. Same
    // per-output FP order (bitwise vs NT=64).
    constexpr int TS = NT / 16;                 // token subtiles per warp (x2 warps)
    constexpr int LDW = KS + 16, LDX = KS + 16; // padded smem strides (bytes)
    // (double-buffered stages and __launch_bounds__ occupancy forcing both
    // measured SLOWER than this single-buffer + register-pipeline shape --
    // 225/254 vs 204 us on the ffn_gate micro; local optimum, do not retry)
    extern __shared__ unsigned char smem_raw[];
    int8_t* s_w = (int8_t*)smem_raw;                  // [MR][LDW]
    int8_t* s_x = (int8_t*)(s_w + MR * LDW);          // [NT][LDX]
    float* s_ws = (float*)(s_x + NT * LDX);           // [MR][KS/64 or KS/128]
    float* s_xs = (float*)(s_ws + MR * (Q4IN ? 2 : 1)); // [NT][XSC]

    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int wm = warp % 4, wn = warp / 4;      // warp tile: rows wm*16, tokens wn*16
    const int gid = lane >> 2, tg = lane & 3;    // mma fragment coords
    // token blocks ride blockIdx.x (fastest-scheduled) so the T/NT blocks
    // sharing one row group's weights run near-concurrently and hit L2;
    // row-major scheduling re-read the weights from DRAM once PER TOKEN
    // BLOCK (8x traffic at T=256 -- measured, this was the whole gap)
    const int64_t r0 = (int64_t)blockIdx.y * MR; // block row base
    const int t0 = blockIdx.x * NT;              // block token base
    const int n_stages = (int)(cols / KS);

    float acc[TS][4];
#pragma unroll
    for (int s = 0; s < TS; s++)
#pragma unroll
        for (int e = 0; e < 4; e++) acc[s][e] = 0.f;

    // Register-buffered staging pipeline: stage st+1's global loads are
    // issued right after stage st's smem stores, so their DRAM latency hides
    // behind stage st's mma work (the old serial stage->compute structure
    // was latency-bound, not BW-bound). Per-thread slice: 4 (Q4) / 8 (Q8)
    // weight u32 + 4 activation u32 + 2 predicated scale floats. The Q4
    // nibble unpack happens at the reg->smem store, off the load path.
    constexpr int WLD = Q4IN ? MR * (KS / 2) / 4 / 256 : MR * KS / 4 / 256;
    constexpr int XLD = NT * KS / 4 / 256;
    constexpr int XSL = (NT * XSC + 255) / 256; // x-scale slices (can exceed 256 threads)
    const int tid = threadIdx.x;
    const int nws = Q4IN ? MR * 2 : MR;
    uint32_t rw[WLD], rx[XLD];
    float rws = 0.f, rxs[XSL];

    auto load_stage = [&](int st) {
        const int64_t k0 = (int64_t)st * KS;
        if (Q4IN) {
#pragma unroll
            for (int i = 0; i < WLD; i++) {
                int idx = i * 256 + tid;
                int rr = idx / 16, pb4 = idx % 16; // 16 u32 of packed bytes per row
                rw[i] = r0 + rr < rows
                            ? __ldg((const uint32_t*)(W + (r0 + rr) * (cols / 2) + k0 / 2) + pb4)
                            : 0x88888888u; // unpacks to 0 after -8
            }
        } else {
#pragma unroll
            for (int i = 0; i < WLD; i++) {
                int idx = i * 256 + tid;
                int rr = idx / (KS / 4), u = idx % (KS / 4);
                rw[i] = r0 + rr < rows
                            ? __ldg((const uint32_t*)(W + (r0 + rr) * cols + k0) + u)
                            : 0u;
            }
        }
#pragma unroll
        for (int i = 0; i < XLD; i++) {
            int idx = i * 256 + tid;
            int tt = idx / (KS / 4), u = idx % (KS / 4);
            rx[i] = t0 + tt < T
                        ? __ldg((const uint32_t*)(nat + (size_t)(t0 + tt) * cols + k0) + u)
                        : 0u;
        }
        if (tid < nws) {
            if (Q4IN) {
                int rr = tid / 2, g = tid % 2;
                rws = r0 + rr < rows
                          ? __half2float(__ldg(S + (r0 + rr) * (cols / 64) + k0 / 64 + g))
                          : 0.f;
            } else {
                rws = r0 + tid < rows
                          ? __half2float(__ldg(S + (r0 + tid) * (cols / 128) + k0 / 128))
                          : 0.f;
            }
        }
#pragma unroll
        for (int i = 0; i < XSL; i++) {
            int idx = i * 256 + tid;
            int tt = idx / XSC, cc = idx % XSC;
            rxs[i] = (idx < NT * XSC && t0 + tt < T)
                         ? __ldg(xs + (size_t)(t0 + tt) * (cols / XGS) + k0 / XGS + cc)
                         : 0.f;
        }
    };
    auto store_stage = [&]() {
        if (Q4IN) {
#pragma unroll
            for (int i = 0; i < WLD; i++) {
                int idx = i * 256 + tid;
                int rr = idx / 16, pb4 = idx % 16;
                int8_t* dst = s_w + rr * LDW + pb4 * 8;
                // vector unpack: interleave lo/hi nibbles bytewise, then
                // per-byte -8 (modular = two's complement, same s8 values as
                // the old byte-store loop) -- 2 u32 stores instead of 8
                // byte stores
                const uint32_t p = rw[i];
                const uint32_t lo = p & 0x0F0F0F0Fu, hi = (p >> 4) & 0x0F0F0F0Fu;
                *(uint32_t*)dst = __vsub4(__byte_perm(lo, hi, 0x5140), 0x08080808u);
                *(uint32_t*)(dst + 4) = __vsub4(__byte_perm(lo, hi, 0x7362), 0x08080808u);
            }
        } else {
#pragma unroll
            for (int i = 0; i < WLD; i++) {
                int idx = i * 256 + tid;
                int rr = idx / (KS / 4), u = idx % (KS / 4);
                *(uint32_t*)(s_w + rr * LDW + u * 4) = rw[i];
            }
        }
#pragma unroll
        for (int i = 0; i < XLD; i++) {
            int idx = i * 256 + tid;
            int tt = idx / (KS / 4), u = idx % (KS / 4);
            *(uint32_t*)(s_x + tt * LDX + u * 4) = rx[i];
        }
        if (tid < nws) s_ws[tid] = rws;
#pragma unroll
        for (int i = 0; i < XSL; i++) {
            int idx = i * 256 + tid;
            if (idx < NT * XSC) s_xs[idx] = rxs[i];
        }
    };

    load_stage(0);
    for (int st = 0; st < n_stages; st++) {
        __syncthreads();
        store_stage();
        if (st + 1 < n_stages) load_stage(st + 1);
        __syncthreads();
        if constexpr (!XG64) {
            // 4 chunks of 32 per stage; per chunk, TS mma pairs (token subtiles of 8)
#pragma unroll
            for (int cc = 0; cc < 4; cc++) {
                const int kb = cc * 32;
                const int8_t* wrow0 = s_w + (wm * 16 + gid) * LDW + kb;
                uint32_t a0 = *(const uint32_t*)(wrow0 + tg * 4);
                uint32_t a1 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4);
                uint32_t a2 = *(const uint32_t*)(wrow0 + tg * 4 + 16);
                uint32_t a3 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4 + 16);
                const float wsc0 = s_ws[(wm * 16 + gid) * (Q4IN ? 2 : 1) + (Q4IN ? kb / 64 : 0)];
                const float wsc1 =
                    s_ws[(wm * 16 + gid + 8) * (Q4IN ? 2 : 1) + (Q4IN ? kb / 64 : 0)];
#pragma unroll
                for (int s = 0; s < TS; s++) {
                    const int tb = wn * (NT / 2) + s * 8; // token subtile base
                    const int8_t* xcol = s_x + (tb + gid) * LDX + kb;
                    uint32_t b0 = *(const uint32_t*)(xcol + tg * 4);
                    uint32_t b1 = *(const uint32_t*)(xcol + tg * 4 + 16);
                    int d0, d1, d2, d3;
                    mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                    const float xs0 = s_xs[(tb + tg * 2) * 4 + cc];
                    const float xs1 = s_xs[(tb + tg * 2 + 1) * 4 + cc];
                    acc[s][0] += wsc0 * xs0 * (float)d0;
                    acc[s][1] += wsc0 * xs1 * (float)d1;
                    acc[s][2] += wsc1 * xs0 * (float)d2;
                    acc[s][3] += wsc1 * xs1 * (float)d3;
                }
            }
        } else {
            // 2 groups of 64 per stage; per group, two chained int32 mmas
            // then ONE fp dequant step (w-scale group == x-scale group for
            // Q4; Q8's per-128 w-scale is constant across the stage)
#pragma unroll
            for (int gg = 0; gg < 2; gg++) {
                const int kb = gg * 64;
                const int8_t* wrow0 = s_w + (wm * 16 + gid) * LDW + kb;
                uint32_t a0 = *(const uint32_t*)(wrow0 + tg * 4);
                uint32_t a1 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4);
                uint32_t a2 = *(const uint32_t*)(wrow0 + tg * 4 + 16);
                uint32_t a3 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4 + 16);
                uint32_t a4 = *(const uint32_t*)(wrow0 + tg * 4 + 32);
                uint32_t a5 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4 + 32);
                uint32_t a6 = *(const uint32_t*)(wrow0 + tg * 4 + 48);
                uint32_t a7 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4 + 48);
                const float wsc0 = s_ws[(wm * 16 + gid) * (Q4IN ? 2 : 1) + (Q4IN ? gg : 0)];
                const float wsc1 =
                    s_ws[(wm * 16 + gid + 8) * (Q4IN ? 2 : 1) + (Q4IN ? gg : 0)];
#pragma unroll
                for (int s = 0; s < TS; s++) {
                    const int tb = wn * (NT / 2) + s * 8; // token subtile base
                    const int8_t* xcol = s_x + (tb + gid) * LDX + kb;
                    uint32_t b0 = *(const uint32_t*)(xcol + tg * 4);
                    uint32_t b1 = *(const uint32_t*)(xcol + tg * 4 + 16);
                    uint32_t b2 = *(const uint32_t*)(xcol + tg * 4 + 32);
                    uint32_t b3 = *(const uint32_t*)(xcol + tg * 4 + 48);
                    int d0, d1, d2, d3;
                    mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                    mma_s8_acc(d0, d1, d2, d3, a4, a5, a6, a7, b2, b3);
                    const float xs0 = s_xs[(tb + tg * 2) * 2 + gg];
                    const float xs1 = s_xs[(tb + tg * 2 + 1) * 2 + gg];
                    acc[s][0] += wsc0 * xs0 * (float)d0;
                    acc[s][1] += wsc0 * xs1 * (float)d1;
                    acc[s][2] += wsc1 * xs0 * (float)d2;
                    acc[s][3] += wsc1 * xs1 * (float)d3;
                }
            }
        }
    }

    const int64_t row0 = r0 + wm * 16 + gid;
#pragma unroll
    for (int s = 0; s < TS; s++) {
        const int tok0 = t0 + wn * (NT / 2) + s * 8 + tg * 2;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            int64_t row = row0 + (e >= 2 ? 8 : 0);
            int tok = tok0 + (e & 1);
            if (row < rows && tok < T) y[(size_t)tok * rows + row] = acc[s][e];
        }
    }
}

// Re-read each launch (a getenv is noise next to a kernel launch) so tests
// can flip paths in-process via setenv.
static bool prefill_use_mma() {
    const char* e = getenv("Q27_PREFILL");
    return !(e && !strcmp(e, "dp4a"));
}

// Activation-regroup dispatch: Q27_PF_XG=32 selects the exact legacy path
// (serial-vs-batched identity holds there -- the --pf gate's leg); default
// is the g64 path (tolerance-gated). Re-read per launch so tests can flip
// in-process, same policy as prefill_use_mma.
static bool prefill_xg64() {
    const char* e = getenv("Q27_PF_XG");
    return !(e && !strcmp(e, "32"));
}

template <bool Q4IN, bool XG64>
static void launch_gemm_mma_x(const uint8_t* W, const __half* S, const XQuant& xq, float* y,
                              int64_t rows, int64_t cols, int T, cudaStream_t st) {
    constexpr int MR = 64, NT = 128, KS = 128, LDW = KS + 16, LDX = KS + 16;
    constexpr int XSC = XG64 ? 2 : 4;
    const size_t SM = (size_t)MR * LDW + (size_t)NT * LDX + (MR * (Q4IN ? 2 : 1) + NT * XSC) * 4;
    if (cols % KS) {
        fprintf(stderr, "gemm_mma: cols %ld not a multiple of %d\n", (long)cols, KS);
        exit(1);
    }
    static bool attr = false;
    if (!attr) {
        CUDA_CHECK(cudaFuncSetAttribute(k_gemm_mma_T<Q4IN, XG64>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
        attr = true;
    }
    dim3 grid((unsigned)((T + NT - 1) / NT), (unsigned)((rows + MR - 1) / MR));
    k_gemm_mma_T<Q4IN, XG64><<<grid, 256, SM, st>>>(W, S, XG64 ? xq.nat64 : xq.nat,
                                                    XG64 ? xq.s64 : xq.scale, y, rows, cols, T);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Q4IN>
static void launch_gemm_mma(const uint8_t* W, const __half* S, const XQuant& xq, float* y,
                            int64_t rows, int64_t cols, int T, cudaStream_t st) {
    if (prefill_xg64() && xq.nat64)
        launch_gemm_mma_x<Q4IN, true>(W, S, xq, y, rows, cols, T, st);
    else
        launch_gemm_mma_x<Q4IN, false>(W, S, xq, y, rows, cols, T, st);
}

void gemm_q4_T(const uint8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
               int64_t cols, int T, cudaStream_t st) {
    if (prefill_use_mma()) {
        launch_gemm_mma<true>(W, S, xq, y, rows, cols, T, st);
        return;
    }
    constexpr int TB = 32, CS = 32, RB = 16;
    constexpr size_t SM = (size_t)CS * (TB * 4 + 1) * sizeof(uint2) +
                          (size_t)CS * (TB + 1) * (sizeof(float) + sizeof(int));
    static bool attr = false;
    if (!attr) {
        CUDA_CHECK(cudaFuncSetAttribute(k_gemm_q4_T<TB, CS>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
        attr = true;
    }
    dim3 grid((unsigned)((rows + RB - 1) / RB));
    for (int t0 = 0; t0 < T; t0 += TB)
        k_gemm_q4_T<TB, CS><<<grid, RB * 32, SM, st>>>(W, S, xq.eo, xq.scale, xq.isum, y, rows,
                                                       cols, T, t0);
    CUDA_CHECK(cudaGetLastError());
}

void gemm_q8_T(const int8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
               int64_t cols, int T, cudaStream_t st) {
    if (prefill_use_mma()) {
        launch_gemm_mma<false>((const uint8_t*)W, S, xq, y, rows, cols, T, st);
        return;
    }
    constexpr int TB = 32, CS = 32, RB = 16;
    constexpr size_t SM = (size_t)CS * (TB * 2 + 1) * sizeof(uint4) +
                          (size_t)CS * (TB + 1) * sizeof(float);
    static bool attr = false;
    if (!attr) {
        CUDA_CHECK(cudaFuncSetAttribute(k_gemm_q8_T<TB, CS>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
        attr = true;
    }
    dim3 grid((unsigned)((rows + RB - 1) / RB));
    for (int t0 = 0; t0 < T; t0 += TB)
        k_gemm_q8_T<TB, CS><<<grid, RB * 32, SM, st>>>(W, S, xq.nat, xq.scale, y, rows, cols, T,
                                                       t0);
    CUDA_CHECK(cudaGetLastError());
}

// F16 weights (ssm alpha/beta: 48x5120). Block per (row, token) with the same
// 256-thread strided walk + shared-memory tree as the serial k_gemv_f16, so
// reductions are bitwise-identical to the single-token path.
__global__ void k_gemm_f16_T(const __half* __restrict__ W, const float* __restrict__ xT,
                             float* __restrict__ y, int64_t rows, int64_t cols) {
    int64_t r = blockIdx.x;
    int t = blockIdx.y;
    const __half* wr = W + r * cols;
    const float* x = xT + (size_t)t * cols;
    float acc = 0.f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
        acc += __half2float(wr[c]) * x[c];
    __shared__ float sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)t * rows + r] = sh[0];
}

void gemm_f16_T(const __half* W, const float* xT, float* y, int64_t rows, int64_t cols, int T,
                cudaStream_t st) {
    dim3 grid((unsigned)rows, (unsigned)T);
    k_gemm_f16_T<<<grid, 256, 0, st>>>(W, xT, y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- batched small ops ----------------

__global__ void k_embed_rows_q8_T(const int8_t* __restrict__ emb, const __half* __restrict__ sc,
                                  const int* __restrict__ toks, int cols, float* __restrict__ out) {
    int t = blockIdx.y;
    int tok = toks[t];
    const int8_t* row = emb + (size_t)tok * cols;
    const __half* sr = sc + (size_t)tok * (cols / 128);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cols; i += gridDim.x * blockDim.x)
        out[(size_t)t * cols + i] = (float)row[i] * __half2float(sr[i / 128]);
}

void embed_rows_q8_T(const int8_t* emb, const __half* scales, const int* toks, int cols, int T,
                     float* out, cudaStream_t st) {
    dim3 grid(20, T);
    k_embed_rows_q8_T<<<grid, 256, 0, st>>>(emb, scales, toks, cols, out);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_rmsnorm_T(const float* __restrict__ x, const float* __restrict__ w,
                            float* __restrict__ y, int n, int in_row, int out_row, float eps) {
    const float* xr = x + (size_t)blockIdx.x * in_row;
    float* yr = y + (size_t)blockIdx.x * out_row;
    __shared__ float sh[32];
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += xr[i] * xr[i];
    acc = warp_reduce_f(acc);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
        v = warp_reduce_f(v);
        if (threadIdx.x == 0) sh[0] = v;
    }
    __syncthreads();
    float inv = rsqrtf(sh[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * inv * w[i];
}

void rmsnorm_T(const float* x, const float* w, float* y, int n, int T, float eps,
               cudaStream_t st, int in_row, int out_row) {
    if (in_row == 0) in_row = n;
    if (out_row == 0) out_row = n;
    k_rmsnorm_T<<<T, 1024, 0, st>>>(x, w, y, n, in_row, out_row, eps);
    CUDA_CHECK(cudaGetLastError());
}

// heads variant: blockIdx.x = head, blockIdx.y = token; row_elems = full row
// length per token in the buffer (e.g. 12288 for qg).
__global__ void k_rmsnorm_heads_T(const float* __restrict__ x, const float* __restrict__ w,
                                  float* __restrict__ y, int head_dim, int stride,
                                  int row_elems, float eps) {
    const float* xh = x + (size_t)blockIdx.y * row_elems + (size_t)blockIdx.x * stride;
    float* yh = y + (size_t)blockIdx.y * row_elems + (size_t)blockIdx.x * stride;
    __shared__ float sh[256];
    float acc = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) acc += xh[i] * xh[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) yh[i] = xh[i] * inv * w[i];
}

void rmsnorm_heads_T(const float* x, const float* w, float* y, int n_heads, int head_dim,
                     int stride, int row_elems, int T, float eps, cudaStream_t st) {
    dim3 grid(n_heads, T);
    k_rmsnorm_heads_T<<<grid, 256, 0, st>>>(x, w, y, head_dim, stride, row_elems, eps);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_l2norm_heads_T(float* __restrict__ x, int head_dim, int row_elems,
                                 float eps) {
    float* xh = x + (size_t)blockIdx.y * row_elems + (size_t)blockIdx.x * head_dim;
    __shared__ float sh[128];
    float acc = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) acc += xh[i] * xh[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    // ggml semantics: y = x / max(sqrt(sum), eps) == x * rsqrt(max(sum, eps^2)).
    // Was max(sum, eps) here -- diverged from the decode/spec paths (blocks.cu:44,
    // spec3.cu:26) by up to 1000x on near-zero heads (CUDA-review #6).
    float inv = rsqrtf(fmaxf(sh[0], eps * eps));
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) xh[i] *= inv;
}

void l2norm_heads_T(float* x, int n_heads, int head_dim, int row_elems, int T, float eps,
                    cudaStream_t st) {
    dim3 grid(n_heads, T);
    k_l2norm_heads_T<<<grid, 128, 0, st>>>(x, head_dim, row_elems, eps);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_rope_neox_T(float* __restrict__ x, int n_rot, int stride, int row_elems,
                              int base_pos, float freq_base) {
    float* xh = x + (size_t)blockIdx.y * row_elems + (size_t)blockIdx.x * stride;
    int d = threadIdx.x;
    if (d >= n_rot / 2) return;
    float theta = (float)(base_pos + blockIdx.y) * powf(freq_base, -2.0f * d / n_rot);
    float c = cosf(theta), s = sinf(theta);
    float x0 = xh[d], x1 = xh[d + n_rot / 2];
    xh[d] = x0 * c - x1 * s;
    xh[d + n_rot / 2] = x0 * s + x1 * c;
}

void rope_neox_T(float* x, int n_heads, int head_dim, int n_rot, int stride, int row_elems,
                 int base_pos, int T, float freq_base, cudaStream_t st) {
    dim3 grid(n_heads, T);
    k_rope_neox_T<<<grid, 32, 0, st>>>(x, n_rot, stride, row_elems, base_pos, freq_base);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_gdn_gates_T(const float* __restrict__ ar, const float* __restrict__ br,
                              const float* __restrict__ a, const float* __restrict__ dt,
                              float* __restrict__ g, float* __restrict__ b, int n) {
    int h = threadIdx.x;
    if (h >= n) return;
    int t = blockIdx.x;
    float x = ar[(size_t)t * n + h] + dt[h];
    float sp = x > 20.f ? x : log1pf(expf(x));
    g[(size_t)t * n + h] = a[h] * sp;
    b[(size_t)t * n + h] = 1.0f / (1.0f + expf(-br[(size_t)t * n + h]));
}

void gdn_gates_T(const float* ar, const float* br, const float* a, const float* dt, float* g,
                 float* b, int n_heads, int T, cudaStream_t st) {
    k_gdn_gates_T<<<T, 64, 0, st>>>(ar, br, a, dt, g, b, n_heads);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_sigmoid_gate_mul_T(float* __restrict__ out, const float* __restrict__ qg,
                                     int head_dim, int n, int qg_row) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    int t = blockIdx.y;
    int h = e / head_dim, d = e % head_dim;
    float gv = qg[(size_t)t * qg_row + (size_t)h * 2 * head_dim + head_dim + d];
    out[(size_t)t * n + e] *= 1.0f / (1.0f + expf(-gv));
}

void sigmoid_gate_mul_T(float* out, const float* qg, int n_heads, int head_dim, int T,
                        cudaStream_t st) {
    int n = n_heads * head_dim;
    dim3 grid((n + 255) / 256, T);
    k_sigmoid_gate_mul_T<<<grid, 256, 0, st>>>(out, qg, head_dim, n, n_heads * 2 * head_dim);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_gated_norm_gdn_T(const float* __restrict__ o, const float* __restrict__ w,
                                   const float* __restrict__ z, float* __restrict__ out,
                                   int head_dim, int n_heads, float eps) {
    size_t row = (size_t)blockIdx.y * n_heads * head_dim;
    const float* oh = o + row + (size_t)blockIdx.x * head_dim;
    const float* zh = z + row + (size_t)blockIdx.x * head_dim;
    float* yh = out + row + (size_t)blockIdx.x * head_dim;
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

void gated_norm_gdn_T(const float* o, const float* w, const float* z, float* out, int n_heads,
                      int head_dim, int T, float eps, cudaStream_t st) {
    dim3 grid(n_heads, T);
    k_gated_norm_gdn_T<<<grid, 128, 0, st>>>(o, w, z, out, head_dim, n_heads, eps);
    CUDA_CHECK(cudaGetLastError());
}

// Depthwise conv k=4 over the chunk. Fully parallel: token t taps come from
// qkvT[t-3..t] with negative indices reading the incoming ring (oldest-first).
// Ring is updated to the last 3 tokens at the end (t >= T-3 writers).
__global__ void k_conv_prefill_T(float* __restrict__ ring, const float* __restrict__ qkvT,
                                 const float* __restrict__ w, float* __restrict__ outT,
                                 int channels, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    int t = blockIdx.y;
    const float* wc = w + (size_t)c * 4;
    auto tap = [&](int ti) {
        return ti >= 0 ? qkvT[(size_t)ti * channels + c] : ring[(size_t)(3 + ti) * channels + c];
    };
    float x0 = tap(t - 3), x1 = tap(t - 2), x2 = tap(t - 1), x3 = tap(t);
#if Q27_CONV_OLDEST_FIRST
    float acc = x0 * wc[0] + x1 * wc[1] + x2 * wc[2] + x3 * wc[3];
#else
    float acc = x0 * wc[3] + x1 * wc[2] + x2 * wc[1] + x3 * wc[0];
#endif
    outT[(size_t)t * channels + c] = acc / (1.0f + expf(-acc));
}

__global__ void k_conv_ring_update_T(float* __restrict__ ring, const float* __restrict__ qkvT,
                                     int channels, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    // new ring = raw qkv of tokens T-3, T-2, T-1 (oldest first); for T < 3 shift in
    float r[3];
    for (int i = 0; i < 3; i++) {
        int ti = T - 3 + i;
        r[i] = ti >= 0 ? qkvT[(size_t)ti * channels + c] : ring[(size_t)(3 + ti) * channels + c];
    }
    for (int i = 0; i < 3; i++) ring[(size_t)i * channels + c] = r[i];
}

void conv_prefill_T(float* ring, const float* qkvT, const float* w, float* outT, int channels,
                    int T, cudaStream_t st) {
    dim3 grid((channels + 255) / 256, T);
    k_conv_prefill_T<<<grid, 256, 0, st>>>(ring, qkvT, w, outT, channels, T);
    k_conv_ring_update_T<<<(channels + 255) / 256, 256, 0, st>>>(ring, qkvT, channels, T);
    CUDA_CHECK(cudaGetLastError());
}

template <typename CT>
__global__ void k_kv_store_T(const float* __restrict__ kT, const float* __restrict__ vT,
                             CT* __restrict__ kc, CT* __restrict__ vc, int base_pos,
                             int rowlen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowlen) return;
    int t = blockIdx.y;
    size_t off = (size_t)(base_pos + t) * rowlen + i;
    kv_set(kc[off], kT[(size_t)t * rowlen + i]);
    kv_set(vc[off], vT[(size_t)t * rowlen + i]);
}

void kv_store_T(const float* kT, const float* vT, void* kc, void* vc, int base_pos,
                int rowlen, int T, cudaStream_t st, bool fp8) {
    dim3 grid((rowlen + 255) / 256, T);
    if (fp8)
        k_kv_store_T<<<grid, 256, 0, st>>>(kT, vT, (__nv_fp8_e4m3*)kc, (__nv_fp8_e4m3*)vc,
                                           base_pos, rowlen);
    else
        k_kv_store_T<<<grid, 256, 0, st>>>(kT, vT, (__half*)kc, (__half*)vc, base_pos, rowlen);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- fp16 tensor-core flash-attention prefill (P1.5) ----------
// Block per (kv head, 16-token tile); 6 warps, warp j owns GQA q-head j of
// this kv head with an m16 tile of 16 tokens. K/V slabs of 32 positions are
// staged in smem once and shared by all 6 warps. QK^T and P*V run on
// mma.sync.m16n8k16 (f16 in, f32 accumulate); softmax stats and O stay fp32.
// Q is rounded to fp16 (K/V already are, in cache) -- numerics differ from
// the fp32-dot FA-lite kernel at rounding level; tolerance-gated like P1.
// Q27_ATTN_PF=lite selects the old kernel.

static __device__ __forceinline__ void mma_f16(float& d0, float& d1, float& d2, float& d3,
                                               uint32_t a0, uint32_t a1, uint32_t a2,
                                               uint32_t a3, uint32_t b0, uint32_t b1, float c0,
                                               float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2),
          "f"(c3));
}

// fp8 e4m3 QK^T: mma.sync.m16n8k32 (4 fp8/reg, f32 accumulate). sm_89+ only
// (Blackwell sm_120 has it); sm_86 gets an accumulate-identity no-op so the
// build links -- the fp8-MMA prefill path never engages on the 3090 (no fp8 HW,
// and attn_prefill_launch only routes here under Q27_PF_FP8MMA + fp8 KV).
static __device__ __forceinline__ void mma_e4m3(float& d0, float& d1, float& d2, float& d3,
                                                uint32_t a0, uint32_t a1, uint32_t a2,
                                                uint32_t a3, uint32_t b0, uint32_t b1, float c0,
                                                float c1, float c2, float c3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2),
          "f"(c3));
#else
    d0 = c0; d1 = c1; d2 = c2; d3 = c3;
#endif
}

static __device__ __forceinline__ uint32_t h2u(__half2 h) {
    return *reinterpret_cast<uint32_t*>(&h);
}

// cp.async 16-byte global->shared copy (sm_80+; compiles on sm_86 and sm_120).
// Bytes beyond src_bytes are zero-filled -- used to zero tail positions past
// p_hi without a branch. Phase 1: hides the K/V load latency (long_scoreboard,
// 30% of the deep-context stall per the 2026-07-07 ncu attribution) by
// prefetching the next PP-tile's raw fp8 while the current tile's MMAs run.
static __device__ __forceinline__ void cpasync16(void* smem, const void* gmem, int src_bytes) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem),
                 "r"(src_bytes));
}
static __device__ __forceinline__ void cpasync_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
static __device__ __forceinline__ void cpasync_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

// transposed 8x8 b16 fragment loads for the PV phase: one x2 per (n, h)
// replaces 4 scalar smem loads + 2 packs, same bits into the mma.
static __device__ __forceinline__ void ldm_x2_trans(uint32_t& r0, uint32_t& r1, const void* p) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(a));
}

// Split-position partials (P4): with gridDim.z = S > 1, split s covers a
// PP-aligned slice of the tile's causal range and writes an UNNORMALIZED
// partial {m, l, O[256]} per (q-head, token row) to `part`; a combine kernel
// merges the S slices. S == 1 writes normalized output directly (the
// pre-split path, bit-identical). Splits exist because grid (4 kv heads,
// T/16 tiles) is only 64 blocks per 256-token chunk on a 170-SM part -- at
// long context the attention wall is SM starvation, not bandwidth.
constexpr int PF_PART_STRIDE = 258; // m, l, O[256]

template <typename CT>
__global__ void __launch_bounds__(192, 1)
k_attn_prefill_mma(const float* __restrict__ qT, int q_stride, int q_row,
                   const CT* __restrict__ kc, const CT* __restrict__ vc,
                   float* __restrict__ outT, int out_row, float* __restrict__ part,
                   int base_pos, int tile_t0, int T, int n_kv_heads, int head_dim,
                   float scale, int cp_async) {
    constexpr int TT = 16, PP = 32, HD = 256, LDH = HD + 8; // padded smem rows (halfs)
    const int kvh = blockIdx.x;
    const int t0 = tile_t0 + blockIdx.y * TT;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tg = lane & 3;
    const int nsp = gridDim.z, sp = blockIdx.z;
    const int trows = gridDim.y * TT;
    const int qh = kvh * 6 + warp;

    const int tile_last = min(t0 + TT, T) - 1;            // last live token in tile
    const int tile_max = base_pos + tile_last + 1;        // positions the tile needs
    if (tile_last < t0) return;                           // whole tile past T

    // this split's PP-aligned position slice
    const int chunk = nsp > 1 ? ((tile_max + nsp - 1) / nsp + PP - 1) / PP * PP : tile_max;
    const int p_lo = sp * chunk, p_hi = min(tile_max, p_lo + chunk);
    if (nsp > 1 && p_lo >= tile_max) {
        // empty split (per-tile ranges are shorter than the grid allows for):
        // stamp sentinel partials so the combine skips this slice
        if (tg == 0) {
            size_t b0 = ((size_t)(qh * trows + t0 + gid) * nsp + sp) * PF_PART_STRIDE;
            size_t b1 = ((size_t)(qh * trows + t0 + gid + 8) * nsp + sp) * PF_PART_STRIDE;
            part[b0] = -FLT_MAX; part[b0 + 1] = 0.f;
            part[b1] = -FLT_MAX; part[b1 + 1] = 0.f;
        }
        return;
    }

    extern __shared__ __half s_h[];
    __half* s_q = s_h;                    // [6][TT][LDH]
    __half* s_k = s_q + 6 * TT * LDH;     // [PP][LDH]
    __half* s_v = s_k + PP * LDH;         // [PP][LDH]
    // cp.async raw prefetch buffers: one PP-tile of K and V, contiguous (no LDH
    // pad, 16B-aligned). fp8 only -- the fp16 path lacks smem room under the
    // 99 KB sm_120 cap, so it stays on the blocking staging (CPA=false).
    constexpr bool CPA = (sizeof(CT) == 1);
    CT* s_kraw = (CT*)(s_v + PP * LDH);   // [PP*HD] (valid only when CPA)
    CT* s_vraw = s_kraw + PP * HD;        // [PP*HD]
    const bool cpa = CPA && cp_async;

    // stage Q (fp32 -> fp16) once: warp w stages its own head's 16 tokens
    {
        for (int idx = lane; idx < TT * HD; idx += 32) {
            int tt = idx / HD, d = idx % HD;
            int t = t0 + tt;
            float v = t < T ? qT[(size_t)t * q_row + (size_t)qh * q_stride + d] : 0.f;
            s_q[(warp * TT + tt) * LDH + d] = __float2half_rn(v);
        }
    }

    // per-warp state: O accumulators (16 tokens x 256 dims across the warp),
    // row stats for the 2 fragment rows this thread touches (gid, gid+8)
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[i][e] = 0.f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.f, l1 = 0.f;

    // cp.async prologue: issue the first tile's raw K/V prefetch. src_bytes=0
    // zero-fills tail positions past p_hi (fp8 0x00 -> kv2h -> 0.0, matching the
    // blocking path's explicit zero).
    if (cpa) {
        for (int idx = threadIdx.x; idx < PP * (HD / 16); idx += blockDim.x) {
            int pp = idx / (HD / 16), d16 = (idx % (HD / 16)) * 16;
            int gpos = p_lo + pp;
            size_t off = ((size_t)gpos * n_kv_heads + kvh) * head_dim + d16;
            cpasync16(s_kraw + pp * HD + d16, &kc[off], gpos < p_hi ? 16 : 0);
            cpasync16(s_vraw + pp * HD + d16, &vc[off], gpos < p_hi ? 16 : 0);
        }
        cpasync_commit();
    }

    for (int p0 = p_lo; p0 < p_hi; p0 += PP) {
        const int np = min(PP, p_hi - p0);
        if (cpa) {
            cpasync_wait_all();  // this tile's raw K/V is resident
            __syncthreads();
            // convert raw fp8 -> half in s_k/s_v: identical bytes through kv2h
            // as the blocking path, so the MMA inputs are bit-for-bit unchanged
            for (int idx = threadIdx.x; idx < PP * (HD / 8); idx += blockDim.x) {
                int pp = idx / (HD / 8), d8 = (idx % (HD / 8)) * 8;
                __half2* kd = (__half2*)(s_k + pp * LDH + d8);
                __half2* vd = (__half2*)(s_v + pp * LDH + d8);
                const CT* kr = s_kraw + pp * HD + d8;
                const CT* vr = s_vraw + pp * HD + d8;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    kd[j] = __halves2half2(kv2h(kr[2 * j]), kv2h(kr[2 * j + 1]));
                    vd[j] = __halves2half2(kv2h(vr[2 * j]), kv2h(vr[2 * j + 1]));
                }
            }
            __syncthreads();  // half ready; raw buffer free to overwrite
            // prefetch the NEXT tile -- overlaps the QK^T/PV MMAs below (hides
            // the K/V load latency that dominates the deep-context stall)
            if (p0 + PP < p_hi) {
                for (int idx = threadIdx.x; idx < PP * (HD / 16); idx += blockDim.x) {
                    int pp = idx / (HD / 16), d16 = (idx % (HD / 16)) * 16;
                    int gpos = p0 + PP + pp;
                    size_t off = ((size_t)gpos * n_kv_heads + kvh) * head_dim + d16;
                    cpasync16(s_kraw + pp * HD + d16, &kc[off], gpos < p_hi ? 16 : 0);
                    cpasync16(s_vraw + pp * HD + d16, &vc[off], gpos < p_hi ? 16 : 0);
                }
                cpasync_commit();
            }
        } else {
            __syncthreads();
            // vectorized staging: 8 KV elements per load (uint2 for fp8, uint4
            // for fp16), same converted bits as the old elementwise path
            for (int idx = threadIdx.x; idx < PP * (HD / 8); idx += blockDim.x) {
                int pp = idx / (HD / 8), d8 = (idx % (HD / 8)) * 8;
                bool ok = pp < np;
                size_t off = ((size_t)(p0 + pp) * n_kv_heads + kvh) * head_dim + d8;
                CT kraw[8], vraw[8];
                __half2* kd = (__half2*)(s_k + pp * LDH + d8);
                __half2* vd = (__half2*)(s_v + pp * LDH + d8);
                if (ok) {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) { kraw[j] = kc[off + j]; vraw[j] = vc[off + j]; }
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        kd[j] = __halves2half2(kv2h(kraw[2 * j]), kv2h(kraw[2 * j + 1]));
                        vd[j] = __halves2half2(kv2h(vraw[2 * j]), kv2h(vraw[2 * j + 1]));
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        kd[j] = __halves2half2(__float2half_rn(0.f), __float2half_rn(0.f));
                        vd[j] = __halves2half2(__float2half_rn(0.f), __float2half_rn(0.f));
                    }
                }
            }
            __syncthreads();
        }

        // S = Q K^T for this warp's 16 tokens x 32 positions (4 n8 subtiles)
        float s[4][4];
#pragma unroll
        for (int n = 0; n < 4; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[n][e] = 0.f;
        const __half* qb = s_q + warp * TT * LDH;
#pragma unroll
        for (int kk = 0; kk < HD / 16; kk++) {
            const int kb = kk * 16;
            uint32_t a0 = h2u(*(const __half2*)(qb + gid * LDH + kb + tg * 2));
            uint32_t a1 = h2u(*(const __half2*)(qb + (gid + 8) * LDH + kb + tg * 2));
            uint32_t a2 = h2u(*(const __half2*)(qb + gid * LDH + kb + tg * 2 + 8));
            uint32_t a3 = h2u(*(const __half2*)(qb + (gid + 8) * LDH + kb + tg * 2 + 8));
#pragma unroll
            for (int n = 0; n < 4; n++) {
                uint32_t b0 = h2u(*(const __half2*)(s_k + (n * 8 + gid) * LDH + kb + tg * 2));
                uint32_t b1 =
                    h2u(*(const __half2*)(s_k + (n * 8 + gid) * LDH + kb + tg * 2 + 8));
                // B is col-major k x n: b regs must hold {K[n][k], K[n][k+1]}
                // pairs along k -- but s_k rows are contiguous in k, so the
                // half2 load above is exactly {k=tg*2, k=tg*2+1} of column
                // n*8+gid. Correct by construction.
                mma_f16(s[n][0], s[n][1], s[n][2], s[n][3], a0, a1, a2, a3, b0, b1, s[n][0],
                        s[n][1], s[n][2], s[n][3]);
            }
        }

        // mask + online softmax. Thread's elements: rows gid (e0,e1), gid+8
        // (e2,e3); cols n*8 + tg*2, +1. Causal bound per row; np bound per col.
        const int r0g = t0 + gid, r1g = t0 + gid + 8; // global token index per row
        const int b0r = base_pos + r0g + 1 - p0;      // valid cols for row gid
        const int b1r = base_pos + r1g + 1 - p0;
        float rmax0 = -FLT_MAX, rmax1 = -FLT_MAX;
#pragma unroll
        for (int n = 0; n < 4; n++) {
            const int c0 = n * 8 + tg * 2, c1 = c0 + 1;
#pragma unroll
            for (int e = 0; e < 4; e++) s[n][e] *= scale;
            if (c0 >= b0r || c0 >= np) s[n][0] = -FLT_MAX;
            if (c1 >= b0r || c1 >= np) s[n][1] = -FLT_MAX;
            if (c0 >= b1r || c0 >= np) s[n][2] = -FLT_MAX;
            if (c1 >= b1r || c1 >= np) s[n][3] = -FLT_MAX;
            rmax0 = fmaxf(rmax0, fmaxf(s[n][0], s[n][1]));
            rmax1 = fmaxf(rmax1, fmaxf(s[n][2], s[n][3]));
        }
        // reduce row max across the 4 threads of each row quad (same gid)
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffff, rmax0, off));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffff, rmax1, off));
        }
        const float mn0 = fmaxf(m0, rmax0), mn1 = fmaxf(m1, rmax1);
        const float sc0 = expf(m0 - mn0), sc1 = expf(m1 - mn1);
        float rl0 = 0.f, rl1 = 0.f;
#pragma unroll
        for (int n = 0; n < 4; n++) {
            s[n][0] = s[n][0] == -FLT_MAX ? 0.f : expf(s[n][0] - mn0);
            s[n][1] = s[n][1] == -FLT_MAX ? 0.f : expf(s[n][1] - mn0);
            s[n][2] = s[n][2] == -FLT_MAX ? 0.f : expf(s[n][2] - mn1);
            s[n][3] = s[n][3] == -FLT_MAX ? 0.f : expf(s[n][3] - mn1);
            rl0 += s[n][0] + s[n][1];
            rl1 += s[n][2] + s[n][3];
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            rl0 += __shfl_xor_sync(0xffffffff, rl0, off);
            rl1 += __shfl_xor_sync(0xffffffff, rl1, off);
        }
        l0 = l0 * sc0 + rl0;
        l1 = l1 * sc1 + rl1;
        m0 = mn0;
        m1 = mn1;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            o[i][0] *= sc0;
            o[i][1] *= sc0;
            o[i][2] *= sc1;
            o[i][3] *= sc1;
        }

        // P (fp16 A-frags, k = 32 positions = 2 k16 steps) x V -> O
        // A-frag identity: subtile pair (2n, 2n+1) of S supplies a0..a3.
        uint32_t pa[2][4];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            pa[h][0] = h2u(__floats2half2_rn(s[2 * h][0], s[2 * h][1]));
            pa[h][1] = h2u(__floats2half2_rn(s[2 * h][2], s[2 * h][3]));
            pa[h][2] = h2u(__floats2half2_rn(s[2 * h + 1][0], s[2 * h + 1][1]));
            pa[h][3] = h2u(__floats2half2_rn(s[2 * h + 1][2], s[2 * h + 1][3]));
        }
#pragma unroll
        for (int n = 0; n < 32; n++) {
            const int d0 = n * 8;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int kp0 = h * 16;
                uint32_t b0, b1;
                ldm_x2_trans(b0, b1, s_v + (kp0 + (lane & 15)) * LDH + d0);
                mma_f16(o[n][0], o[n][1], o[n][2], o[n][3], pa[h][0], pa[h][1], pa[h][2],
                        pa[h][3], b0, b1, o[n][0], o[n][1], o[n][2], o[n][3]);
            }
        }
    }

    const int tr0 = t0 + gid, tr1 = t0 + gid + 8;
    if (nsp > 1) {
        // write UNNORMALIZED partials {m, l, O} for the combine pass; row
        // stats are quad-uniform after the shfl reductions, so tg==0 writes
        size_t b0 = ((size_t)(qh * trows + tr0) * nsp + sp) * PF_PART_STRIDE;
        size_t b1 = ((size_t)(qh * trows + tr1) * nsp + sp) * PF_PART_STRIDE;
        if (tg == 0) {
            part[b0] = m0; part[b0 + 1] = l0;
            part[b1] = m1; part[b1 + 1] = l1;
        }
#pragma unroll
        for (int n = 0; n < 32; n++) {
            const int d0 = n * 8 + tg * 2;
            part[b0 + 2 + d0] = o[n][0];
            part[b0 + 2 + d0 + 1] = o[n][1];
            part[b1 + 2 + d0] = o[n][2];
            part[b1 + 2 + d0 + 1] = o[n][3];
        }
        return;
    }
    // normalize + write: thread's O elements are (rows gid/gid+8, cols
    // n*8 + tg*2, +1) of the 16x256 tile
    const float inv0 = l0 > 0.f ? 1.0f / l0 : 0.f, inv1 = l1 > 0.f ? 1.0f / l1 : 0.f;
#pragma unroll
    for (int n = 0; n < 32; n++) {
        const int d0 = n * 8 + tg * 2;
        if (tr0 < T) {
            outT[(size_t)tr0 * out_row + (size_t)qh * head_dim + d0] = o[n][0] * inv0;
            outT[(size_t)tr0 * out_row + (size_t)qh * head_dim + d0 + 1] = o[n][1] * inv0;
        }
        if (tr1 < T) {
            outT[(size_t)tr1 * out_row + (size_t)qh * head_dim + d0] = o[n][2] * inv1;
            outT[(size_t)tr1 * out_row + (size_t)qh * head_dim + d0 + 1] = o[n][3] * inv1;
        }
    }
}

// fp8 QK^T variant of k_attn_prefill_mma (Phase 2, Q27_PF_FP8MMA). Q staged as
// e4m3 (s_q 25.3KB vs 50.7), K read straight from the fp8 cache in the MMA
// (mma.sync.m16n8k32.e4m3) so s_k is dropped and s_kraw is double-buffered for
// cp.async -- the smem relayout the ec1a54c revert prescribed. PV stays fp16 (V
// still converted to s_v). Softmax/PV/output are byte-identical to the f16
// kernel (m16n8 f32 accumulator layout is K-independent); only QK^T numerics
// change (Q-cast + fp8 accumulate), tolerance-gated. fp8 KV only. ~66KB smem.
__global__ void __launch_bounds__(192, 1)
k_attn_prefill_mma_fp8q(const float* __restrict__ qT, int q_stride, int q_row,
                        const __nv_fp8_e4m3* __restrict__ kc,
                        const __nv_fp8_e4m3* __restrict__ vc, float* __restrict__ outT,
                        int out_row, float* __restrict__ part, int base_pos, int tile_t0, int T,
                        int n_kv_heads, int head_dim, float scale, int cp_async) {
    // LDQ/LDK pad s_q/s_kraw so the fp8 QK^T uint32 reads don't 8-way bank-conflict
    // (word stride coprime-ish to 32; the f16 path gets this free via LDH).
    constexpr int TT = 16, PP = 32, HD = 256, LDH = HD + 8, LDQ = HD + 4, LDK = HD + 16;
    // The a/b fragment loads read `*(uint32_t*)(base + row*LD? + kb + tg*4[+16])`.
    // Row strides and every byte offset are multiples of 4, so the uint32 loads
    // are 4B-aligned -- these static_asserts fail the build if a stride change
    // breaks that invariant (review finding #3). The reads also never touch the
    // pad tail [HD, LD?): max offset is (n*8+gid)*LD + 7*32 + 3*4 + 16 + 3 <
    // row_base + HD, so the uninitialized pad bytes are never consumed (#4).
    static_assert(LDQ % 4 == 0 && LDK % 4 == 0, "fp8 QK^T uint32 loads must stay 4B-aligned");
    const int kvh = blockIdx.x;
    const int t0 = tile_t0 + blockIdx.y * TT;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tg = lane & 3;
    const int nsp = gridDim.z, sp = blockIdx.z;
    const int trows = gridDim.y * TT;
    const int qh = kvh * 6 + warp;

    const int tile_last = min(t0 + TT, T) - 1;
    const int tile_max = base_pos + tile_last + 1;
    if (tile_last < t0) return;

    const int chunk = nsp > 1 ? ((tile_max + nsp - 1) / nsp + PP - 1) / PP * PP : tile_max;
    const int p_lo = sp * chunk, p_hi = min(tile_max, p_lo + chunk);
    if (nsp > 1 && p_lo >= tile_max) {
        if (tg == 0) {
            size_t b0 = ((size_t)(qh * trows + t0 + gid) * nsp + sp) * PF_PART_STRIDE;
            size_t b1 = ((size_t)(qh * trows + t0 + gid + 8) * nsp + sp) * PF_PART_STRIDE;
            part[b0] = -FLT_MAX; part[b0 + 1] = 0.f;
            part[b1] = -FLT_MAX; part[b1 + 1] = 0.f;
        }
        return;
    }

    // smem: s_q fp8 [6*TT*LDQ] | s_v half [PP*LDH] | s_kraw fp8 x2 [2*PP*LDK] |
    // s_vraw fp8 [PP*HD] ~= 65.9KB. No s_k (K consumed as fp8 in the MMA).
    extern __shared__ unsigned char smem_raw8[];
    __nv_fp8_e4m3* s_q = (__nv_fp8_e4m3*)smem_raw8;           // [6][TT][LDQ]
    __half* s_v = (__half*)(s_q + 6 * TT * LDQ);              // [PP][LDH]
    __nv_fp8_e4m3* s_kraw = (__nv_fp8_e4m3*)(s_v + PP * LDH); // [2][PP*LDK] ping-pong
    __nv_fp8_e4m3* s_vraw = s_kraw + 2 * PP * LDK;            // [PP*HD] (V converted, no pad)
    const bool cpa = cp_async != 0;

    // stage Q (fp32 -> e4m3) once: warp w stages its own head's 16 tokens
    for (int idx = lane; idx < TT * HD; idx += 32) {
        int tt = idx / HD, d = idx % HD;
        int t = t0 + tt;
        float v = t < T ? qT[(size_t)t * q_row + (size_t)qh * q_stride + d] : 0.f;
        s_q[(warp * TT + tt) * LDQ + d] = __nv_fp8_e4m3(v);
    }

    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[i][e] = 0.f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.f, l1 = 0.f;

    int cur = 0;
    if (cpa) {
        // prologue: prefetch tile p_lo K into s_kraw[0], V into s_vraw
        for (int idx = threadIdx.x; idx < PP * (HD / 16); idx += blockDim.x) {
            int pp = idx / (HD / 16), d16 = (idx % (HD / 16)) * 16;
            int gpos = p_lo + pp;
            size_t off = ((size_t)gpos * n_kv_heads + kvh) * head_dim + d16;
            cpasync16(s_kraw + pp * LDK + d16, &kc[off], gpos < p_hi ? 16 : 0);
            cpasync16(s_vraw + pp * HD + d16, &vc[off], gpos < p_hi ? 16 : 0);
        }
        cpasync_commit();
    }

    for (int p0 = p_lo; p0 < p_hi; p0 += PP) {
        const int np = min(PP, p_hi - p0);
        __nv_fp8_e4m3* kbuf = s_kraw + (cpa ? cur : 0) * PP * LDK;
        if (cpa) {
            cpasync_wait_all();  // this tile's raw K in kbuf, V in s_vraw
            __syncthreads();
            // convert only V (raw fp8 -> half in s_v); K stays raw in kbuf for
            // the fp8 MMA. Same kv2h bytes as the blocking path.
            for (int idx = threadIdx.x; idx < PP * (HD / 8); idx += blockDim.x) {
                int pp = idx / (HD / 8), d8 = (idx % (HD / 8)) * 8;
                __half2* vd = (__half2*)(s_v + pp * LDH + d8);
                const __nv_fp8_e4m3* vr = s_vraw + pp * HD + d8;
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    vd[j] = __halves2half2(kv2h(vr[2 * j]), kv2h(vr[2 * j + 1]));
            }
            __syncthreads();  // s_v ready; s_vraw free; kbuf still holds this tile
            // prefetch NEXT tile K into the OTHER buffer + V into s_vraw --
            // overlaps the QK^T/PV MMAs; kbuf (s_kraw[cur]) is untouched
            if (p0 + PP < p_hi) {
                __nv_fp8_e4m3* knext = s_kraw + (1 - cur) * PP * LDK;
                for (int idx = threadIdx.x; idx < PP * (HD / 16); idx += blockDim.x) {
                    int pp = idx / (HD / 16), d16 = (idx % (HD / 16)) * 16;
                    int gpos = p0 + PP + pp;
                    size_t off = ((size_t)gpos * n_kv_heads + kvh) * head_dim + d16;
                    cpasync16(knext + pp * LDK + d16, &kc[off], gpos < p_hi ? 16 : 0);
                    cpasync16(s_vraw + pp * HD + d16, &vc[off], gpos < p_hi ? 16 : 0);
                }
                cpasync_commit();
            }
        } else {
            __syncthreads();
            // blocking: load K raw into kbuf (=s_kraw[0]); convert V -> s_v
            for (int idx = threadIdx.x; idx < PP * (HD / 8); idx += blockDim.x) {
                int pp = idx / (HD / 8), d8 = (idx % (HD / 8)) * 8;
                bool ok = pp < np;
                size_t off = ((size_t)(p0 + pp) * n_kv_heads + kvh) * head_dim + d8;
                __half2* vd = (__half2*)(s_v + pp * LDH + d8);
                __nv_fp8_e4m3* kd = kbuf + pp * LDK + d8;
                if (ok) {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) kd[j] = kc[off + j];
                    #pragma unroll
                    for (int j = 0; j < 4; j++)
                        vd[j] = __halves2half2(kv2h(vc[off + 2 * j]), kv2h(vc[off + 2 * j + 1]));
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) kd[j] = __nv_fp8_e4m3(0.f);
                    #pragma unroll
                    for (int j = 0; j < 4; j++)
                        vd[j] = __halves2half2(__float2half_rn(0.f), __float2half_rn(0.f));
                }
            }
            __syncthreads();
        }

        // S = Q K^T, fp8 m16n8k32 (8 k32 steps). a/b regs = uint32 of 4
        // consecutive e4m3 at k = tg*4 + {0..3} (low) and +16 (high); rows
        // gid / gid+8. Accumulator s[n][0..3] identical layout to the f16 path.
        float s[4][4];
#pragma unroll
        for (int n = 0; n < 4; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[n][e] = 0.f;
        const __nv_fp8_e4m3* qb = s_q + warp * TT * LDQ;
#pragma unroll
        for (int kk = 0; kk < HD / 32; kk++) {
            const int kb = kk * 32;
            uint32_t a0 = *(const uint32_t*)(qb + gid * LDQ + kb + tg * 4);
            uint32_t a1 = *(const uint32_t*)(qb + (gid + 8) * LDQ + kb + tg * 4);
            uint32_t a2 = *(const uint32_t*)(qb + gid * LDQ + kb + tg * 4 + 16);
            uint32_t a3 = *(const uint32_t*)(qb + (gid + 8) * LDQ + kb + tg * 4 + 16);
#pragma unroll
            for (int n = 0; n < 4; n++) {
                uint32_t b0 = *(const uint32_t*)(kbuf + (n * 8 + gid) * LDK + kb + tg * 4);
                uint32_t b1 = *(const uint32_t*)(kbuf + (n * 8 + gid) * LDK + kb + tg * 4 + 16);
                mma_e4m3(s[n][0], s[n][1], s[n][2], s[n][3], a0, a1, a2, a3, b0, b1, s[n][0],
                         s[n][1], s[n][2], s[n][3]);
            }
        }

        // mask + online softmax (verbatim from k_attn_prefill_mma)
        const int r0g = t0 + gid, r1g = t0 + gid + 8;
        const int b0r = base_pos + r0g + 1 - p0;
        const int b1r = base_pos + r1g + 1 - p0;
        float rmax0 = -FLT_MAX, rmax1 = -FLT_MAX;
#pragma unroll
        for (int n = 0; n < 4; n++) {
            const int c0 = n * 8 + tg * 2, c1 = c0 + 1;
#pragma unroll
            for (int e = 0; e < 4; e++) s[n][e] *= scale;
            if (c0 >= b0r || c0 >= np) s[n][0] = -FLT_MAX;
            if (c1 >= b0r || c1 >= np) s[n][1] = -FLT_MAX;
            if (c0 >= b1r || c0 >= np) s[n][2] = -FLT_MAX;
            if (c1 >= b1r || c1 >= np) s[n][3] = -FLT_MAX;
            rmax0 = fmaxf(rmax0, fmaxf(s[n][0], s[n][1]));
            rmax1 = fmaxf(rmax1, fmaxf(s[n][2], s[n][3]));
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffff, rmax0, off));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffff, rmax1, off));
        }
        const float mn0 = fmaxf(m0, rmax0), mn1 = fmaxf(m1, rmax1);
        const float sc0 = expf(m0 - mn0), sc1 = expf(m1 - mn1);
        float rl0 = 0.f, rl1 = 0.f;
#pragma unroll
        for (int n = 0; n < 4; n++) {
            s[n][0] = s[n][0] == -FLT_MAX ? 0.f : expf(s[n][0] - mn0);
            s[n][1] = s[n][1] == -FLT_MAX ? 0.f : expf(s[n][1] - mn0);
            s[n][2] = s[n][2] == -FLT_MAX ? 0.f : expf(s[n][2] - mn1);
            s[n][3] = s[n][3] == -FLT_MAX ? 0.f : expf(s[n][3] - mn1);
            rl0 += s[n][0] + s[n][1];
            rl1 += s[n][2] + s[n][3];
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            rl0 += __shfl_xor_sync(0xffffffff, rl0, off);
            rl1 += __shfl_xor_sync(0xffffffff, rl1, off);
        }
        l0 = l0 * sc0 + rl0;
        l1 = l1 * sc1 + rl1;
        m0 = mn0;
        m1 = mn1;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            o[i][0] *= sc0;
            o[i][1] *= sc0;
            o[i][2] *= sc1;
            o[i][3] *= sc1;
        }

        // P x V -> O (fp16, verbatim from k_attn_prefill_mma)
        uint32_t pa[2][4];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            pa[h][0] = h2u(__floats2half2_rn(s[2 * h][0], s[2 * h][1]));
            pa[h][1] = h2u(__floats2half2_rn(s[2 * h][2], s[2 * h][3]));
            pa[h][2] = h2u(__floats2half2_rn(s[2 * h + 1][0], s[2 * h + 1][1]));
            pa[h][3] = h2u(__floats2half2_rn(s[2 * h + 1][2], s[2 * h + 1][3]));
        }
#pragma unroll
        for (int n = 0; n < 32; n++) {
            const int d0 = n * 8;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int kp0 = h * 16;
                uint32_t b0, b1;
                ldm_x2_trans(b0, b1, s_v + (kp0 + (lane & 15)) * LDH + d0);
                mma_f16(o[n][0], o[n][1], o[n][2], o[n][3], pa[h][0], pa[h][1], pa[h][2],
                        pa[h][3], b0, b1, o[n][0], o[n][1], o[n][2], o[n][3]);
            }
        }

        if (cpa) cur = 1 - cur;
    }

    // normalize + write (verbatim from k_attn_prefill_mma). The split path (nsp>1)
    // writes partials at ABSOLUTE row tr0=t0+gid while k_attn_pf_combine reads
    // RELATIVE rows [0,trows) -- correct ONLY when t0==0 (CUDA-review #4 / rule 6a).
    // attn_prefill_launch enforces this: nsplit>1 only when `part && t0==0`, so this
    // kernel is never called with t0>0 and a split simultaneously. Preserve that
    // gate if this kernel gains a t0>0 sub-batching caller.
    const int tr0 = t0 + gid, tr1 = t0 + gid + 8;
    if (nsp > 1) {
        size_t b0 = ((size_t)(qh * trows + tr0) * nsp + sp) * PF_PART_STRIDE;
        size_t b1 = ((size_t)(qh * trows + tr1) * nsp + sp) * PF_PART_STRIDE;
        if (tg == 0) {
            part[b0] = m0; part[b0 + 1] = l0;
            part[b1] = m1; part[b1 + 1] = l1;
        }
#pragma unroll
        for (int n = 0; n < 32; n++) {
            const int d0 = n * 8 + tg * 2;
            part[b0 + 2 + d0] = o[n][0];
            part[b0 + 2 + d0 + 1] = o[n][1];
            part[b1 + 2 + d0] = o[n][2];
            part[b1 + 2 + d0 + 1] = o[n][3];
        }
        return;
    }
    const float inv0 = l0 > 0.f ? 1.0f / l0 : 0.f, inv1 = l1 > 0.f ? 1.0f / l1 : 0.f;
#pragma unroll
    for (int n = 0; n < 32; n++) {
        const int d0 = n * 8 + tg * 2;
        if (tr0 < T) {
            outT[(size_t)tr0 * out_row + (size_t)qh * head_dim + d0] = o[n][0] * inv0;
            outT[(size_t)tr0 * out_row + (size_t)qh * head_dim + d0 + 1] = o[n][1] * inv0;
        }
        if (tr1 < T) {
            outT[(size_t)tr1 * out_row + (size_t)qh * head_dim + d0] = o[n][2] * inv1;
            outT[(size_t)tr1 * out_row + (size_t)qh * head_dim + d0 + 1] = o[n][3] * inv1;
        }
    }
}

// merge S split partials for one (token, q-head): m* = max m_s,
// O = sum O_s * exp(m_s - m*), out = O / (sum l_s * exp(m_s - m*))
__global__ void k_attn_pf_combine(const float* __restrict__ part, float* __restrict__ outT,
                                  int out_row, int head_dim, int trows, int nsp) {
    const int t = blockIdx.x, qh = blockIdx.y;
    const float* pp = part + (size_t)(qh * trows + t) * nsp * PF_PART_STRIDE;
    __shared__ float s_m, s_l;
    if (threadIdx.x == 0) {
        float mg = -FLT_MAX;
        for (int s = 0; s < nsp; s++) mg = fmaxf(mg, pp[(size_t)s * PF_PART_STRIDE]);
        float lg = 0.f;
        for (int s = 0; s < nsp; s++) {
            float ms = pp[(size_t)s * PF_PART_STRIDE];
            if (ms != -FLT_MAX) lg += pp[(size_t)s * PF_PART_STRIDE + 1] * expf(ms - mg);
        }
        s_m = mg;
        s_l = lg;
    }
    __syncthreads();
    const float mg = s_m, inv = s_l > 0.f ? 1.0f / s_l : 0.f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float a = 0.f;
        for (int s = 0; s < nsp; s++) {
            float ms = pp[(size_t)s * PF_PART_STRIDE];
            if (ms != -FLT_MAX) a += pp[(size_t)s * PF_PART_STRIDE + 2 + d] * expf(ms - mg);
        }
        outT[(size_t)t * out_row + (size_t)qh * head_dim + d] = a * inv;
    }
}

// FA-lite attention prefill: block per (kv head, 8-token tile). Warp w owns
// token (tile_t0 + w) and all 6 GQA q-heads of this kv head; K/V rows are
// staged in shared memory once per 32-position tile and shared by all 48
// (head, token) pairs. Online softmax (no position scratch). Note: fp
// summation order differs from the serial decode kernel; gated empirically
// on identical continuations.
template <typename CT>
__global__ void k_attn_prefill_T(const float* __restrict__ qT, int q_stride, int q_row,
                                 const CT* __restrict__ kc, const CT* __restrict__ vc,
                                 float* __restrict__ outT, int out_row, int base_pos,
                                 int tile_t0, int T, int n_kv_heads, int gqa, int head_dim,
                                 float scale) {
    constexpr int TT = 8, PP = 32, HD = 256;
    const int kvh = blockIdx.x;
    const int t0 = tile_t0 + blockIdx.y * TT;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int t = t0 + warp;            // this warp's token (local chunk index)
    const bool live = t < T;
    const int bound = base_pos + t + 1; // causal bound for this token
    const int tile_max = base_pos + min(t0 + TT, T); // positions needed by tile

    extern __shared__ float s_kv[];     // [PP][2][HD]: K then V per position
    float q[6][8], acc[6][8], m[6], l[6];
#pragma unroll
    for (int j = 0; j < 6; j++) {
        m[j] = -FLT_MAX;
        l[j] = 0.f;
#pragma unroll
        for (int d = 0; d < 8; d++) {
            acc[j][d] = 0.f;
            q[j][d] = live ? qT[(size_t)t * q_row + (size_t)(kvh * gqa + j) * q_stride +
                                lane + 32 * d]
                           : 0.f;
        }
    }

    for (int p0 = 0; p0 < tile_max; p0 += PP) {
        __syncthreads();
        const int np = min(PP, tile_max - p0);
        for (int idx = threadIdx.x; idx < np * HD; idx += blockDim.x) {
            int pp = idx / HD, d = idx % HD;
            size_t off = ((size_t)(p0 + pp) * n_kv_heads + kvh) * head_dim + d;
            s_kv[(pp * 2) * HD + d] = kv2f(kc[off]);
            s_kv[(pp * 2 + 1) * HD + d] = kv2f(vc[off]);
        }
        __syncthreads();
        if (!live) continue;
        const int lim = min(np, bound - p0);
        for (int pp = 0; pp < lim; pp++) {
            const float* kp = s_kv + (pp * 2) * HD;
            const float* vp = s_kv + (pp * 2 + 1) * HD;
#pragma unroll
            for (int j = 0; j < 6; j++) {
                float d = 0.f;
#pragma unroll
                for (int u = 0; u < 8; u++) d += q[j][u] * kp[lane + 32 * u];
                for (int off = 16; off > 0; off >>= 1)
                    d += __shfl_down_sync(0xffffffff, d, off);
                d = __shfl_sync(0xffffffff, d, 0) * scale;
                float mn = fmaxf(m[j], d);
                float sc_old = expf(m[j] - mn), w = expf(d - mn);
                l[j] = l[j] * sc_old + w;
                m[j] = mn;
#pragma unroll
                for (int u = 0; u < 8; u++)
                    acc[j][u] = acc[j][u] * sc_old + w * vp[lane + 32 * u];
            }
        }
    }
    if (!live) return;
#pragma unroll
    for (int j = 0; j < 6; j++) {
        float inv = 1.0f / l[j];
#pragma unroll
        for (int u = 0; u < 8; u++)
            outT[(size_t)t * out_row + (size_t)(kvh * gqa + j) * head_dim + lane + 32 * u] =
                acc[j][u] * inv;
    }
}

template <typename CT>
static void attn_prefill_launch(const float* qT, int q_stride, int q_row, const void* kc,
                                const void* vc, float* outT, int out_row, float* part,
                                int base_pos, int t0, int SB, int n_q_heads, int n_kv_heads,
                                int head_dim, float scale, cudaStream_t st) {
    const char* e = getenv("Q27_ATTN_PF");
    const bool use_mma =
        !(e && !strcmp(e, "lite")) && head_dim == 256 && n_q_heads == 6 * n_kv_heads;
    if (use_mma) {
        constexpr int TT = 16, PP = 32, LDH = 256 + 8;
        // fp8 adds one PP-tile of raw K+V (16 KB) for the cp.async prefetch;
        // fp16 has no room under the 99 KB cap and keeps the blocking path.
        const size_t SM = (size_t)(6 * TT + 2 * PP) * LDH * sizeof(__half) +
                          (sizeof(CT) == 1 ? (size_t)2 * PP * 256 * sizeof(CT) : 0);
        static int cpa_env = -1;
        if (cpa_env < 0) {
            const char* ce = getenv("Q27_PF_CPASYNC");
            cpa_env = ce ? atoi(ce) : 1;  // default ON for the fp8 path
        }
        // Phase 2: fp8 QK^T MMA -- DEFAULT ON for the fp8 KV path (fully gated:
        // +11.8% @128K, greedy-identical, logit A/B cosine 0.99998 argmax-match
        // @131K, needle 6/6 @301K). Own smem relayout (s_q e4m3 + double-buffered
        // s_kraw, no s_k) in a separate kernel; the fp16 KV path below and the
        // fp16 canonical are untouched. Set Q27_PF_FP8MMA=0 to force the f16-MMA
        // fp8 path (bisection / <sm_89 auto-fallback via the guard below).
        // fp8 m16n8k32 MMA is sm_89+ (Blackwell sm_120 ok); the mma_e4m3
        // stub NO-OPs on <sm_89 and would silently emit garbage. Gate on the
        // LOADED IMAGE, not the physical device (review follow-up 2026-07-09
        // #4): the build carries sm_86 + sm_120 SASS only, so an sm_89 device
        // (Ada) runs the sm_86 image -- where mma_e4m3 IS the no-op stub even
        // though the device attribute says CC 8.9. k_arch_probe reports the
        // __CUDA_ARCH__ the running image was compiled for. Cached (image
        // doesn't change); the env is RE-READ per launch -- test_kernels
        // exercises both the f16-staging bitwise path (=0) and the fp8q
        // tolerance path (=1) in one process, and a getenv at launch
        // frequency (once per chunk x layer) is noise.
        static int fp8mma_arch = -1;
        if (fp8mma_arch < 0) {
            int* d_arch = nullptr;
            int h_arch = 0;
            if (cudaMalloc(&d_arch, 4) == cudaSuccess) {
                k_arch_probe<<<1, 1>>>(d_arch);
                if (cudaMemcpy(&h_arch, d_arch, 4, cudaMemcpyDeviceToHost) != cudaSuccess)
                    h_arch = 0;
                cudaFree(d_arch);
            }
            fp8mma_arch = h_arch >= 890 ? 1 : 0;
            if (!fp8mma_arch)
                fprintf(stderr,
                        "[pfattn] loaded image sm_%d < sm_89: fp8-MMA prefill unavailable, "
                        "using f16-MMA\n",
                        h_arch / 10);
        }
        const char* fe = getenv("Q27_PF_FP8MMA");
        const int fp8mma_env = (fe ? atoi(fe) : 1) && fp8mma_arch;  // default ON
        if constexpr (sizeof(CT) == 1) {
            if (fp8mma_env) {
                constexpr int LDQ = 260, LDK = 272;  // match k_attn_prefill_mma_fp8q pads
                const size_t SM8 = (size_t)6 * TT * LDQ * sizeof(__nv_fp8_e4m3) +
                                   (size_t)PP * LDH * sizeof(__half) +
                                   (size_t)(2 * PP * LDK + PP * 256) * sizeof(__nv_fp8_e4m3);
                static bool attr8 = false;
                if (!attr8) {
                    CUDA_CHECK(cudaFuncSetAttribute(
                        k_attn_prefill_mma_fp8q,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, SM8));
                    attr8 = true;
                }
                int nsplit = 1;
                if (part && t0 == 0) {
                    const char* se = getenv("Q27_PF_SPLIT");
                    nsplit = se ? atoi(se) : (base_pos + t0 + SB) / 4096;
                    nsplit = nsplit < 1 ? 1 : nsplit > PF_SPLIT_MAX ? PF_SPLIT_MAX : nsplit;
                }
                const int tiles = (SB + TT - 1) / TT;
                dim3 grid(n_kv_heads, tiles, nsplit);
                k_attn_prefill_mma_fp8q<<<grid, 192, SM8, st>>>(
                    qT, q_stride, q_row, (const __nv_fp8_e4m3*)kc, (const __nv_fp8_e4m3*)vc, outT,
                    out_row, part, base_pos, t0, t0 + SB, n_kv_heads, head_dim, scale, cpa_env);
                CUDA_CHECK(cudaGetLastError());
                if (nsplit > 1) {
                    dim3 g2(SB, n_q_heads);
                    k_attn_pf_combine<<<g2, 256, 0, st>>>(part, outT, out_row, head_dim,
                                                          tiles * TT, nsplit);
                    CUDA_CHECK(cudaGetLastError());
                }
                return;
            }
        }
        static bool attr2 = false;
        if (!attr2) {
            CUDA_CHECK(cudaFuncSetAttribute(k_attn_prefill_mma<CT>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
            attr2 = true;
        }
        // position splits (P4): grid (4 kvh, <=16 tiles) alone starves the
        // SMs; add splits once the context is deep enough to amortize the
        // combine. Q27_PF_SPLIT=N forces N (1 = pre-split behavior).
        int nsplit = 1;
        // split path only when t0==0: its partial writes use absolute t0+row
        // while the combine reads relative rows, so t0>0 corrupts (CUDA-review
        // #4). The non-split path handles any t0 correctly.
        if (part && t0 == 0) {
            const char* se = getenv("Q27_PF_SPLIT");
            nsplit = se ? atoi(se) : (base_pos + t0 + SB) / 4096;
            nsplit = nsplit < 1 ? 1 : nsplit > PF_SPLIT_MAX ? PF_SPLIT_MAX : nsplit;
        }
        const int tiles = (SB + TT - 1) / TT;
        dim3 grid(n_kv_heads, tiles, nsplit);
        k_attn_prefill_mma<CT><<<grid, 192, SM, st>>>(qT, q_stride, q_row, (const CT*)kc,
                                                      (const CT*)vc, outT, out_row, part,
                                                      base_pos, t0, t0 + SB, n_kv_heads,
                                                      head_dim, scale, cpa_env);
        CUDA_CHECK(cudaGetLastError());
        if (nsplit > 1) {
            dim3 g2(SB, n_q_heads);
            k_attn_pf_combine<<<g2, 256, 0, st>>>(part, outT, out_row, head_dim, tiles * TT,
                                                  nsplit);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }
    constexpr int TT = 8, PP = 32;
    const size_t SM = (size_t)PP * 2 * 256 * sizeof(float);
    static bool attr = false;
    if (!attr) {
        CUDA_CHECK(cudaFuncSetAttribute(k_attn_prefill_T<CT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
        attr = true;
    }
    int gqa = n_q_heads / n_kv_heads;
    dim3 grid(n_kv_heads, (SB + TT - 1) / TT);
    k_attn_prefill_T<CT><<<grid, 256, SM, st>>>(qT, q_stride, q_row, (const CT*)kc,
                                                (const CT*)vc, outT, out_row, base_pos, t0,
                                                t0 + SB, n_kv_heads, gqa, head_dim, scale);
    CUDA_CHECK(cudaGetLastError());
}

void attn_prefill_T(const float* qT, int q_stride, int q_row, const void* kc, const void* vc,
                    float* outT, int out_row, float* part, int base_pos, int t0, int SB,
                    int n_q_heads, int n_kv_heads, int head_dim, float scale, cudaStream_t st,
                    bool fp8) {
    if (fp8)
        attn_prefill_launch<__nv_fp8_e4m3>(qT, q_stride, q_row, kc, vc, outT, out_row, part,
                                           base_pos, t0, SB, n_q_heads, n_kv_heads, head_dim,
                                           scale, st);
    else
        attn_prefill_launch<__half>(qT, q_stride, q_row, kc, vc, outT, out_row, part, base_pos,
                                    t0, SB, n_q_heads, n_kv_heads, head_dim, scale, st);
}

// Sequential delta rule over the chunk: S tile lives in dynamic shared memory
// (128x128 floats = 64KB), loaded once and written back once. Same intra-token
// structure as k_delta_step: 512 threads = 4 i-tiles x 128 j.
__global__ void k_delta_scan_T(float* __restrict__ Sg, const float* __restrict__ convT,
                               const float* __restrict__ gT, const float* __restrict__ betaT,
                               float* __restrict__ oT, int T) {
    constexpr int SK = 128;
    constexpr int GDN_CH = 10240;
    constexpr int NH = 48;
    extern __shared__ float smem[];
    float* S = smem;                       // [128][128]
    __shared__ float sq[SK], sk[SK], part[4][SK], dj[SK];
    const int h = blockIdx.x;
    const int j = threadIdx.x & (SK - 1);
    const int it = threadIdx.x >> 7;
    const int i0 = it * 32;
    const int qk = h % 16;
    const float scale = rsqrtf((float)SK);

    float* Sgh = Sg + (size_t)h * SK * SK;
    for (int i = i0; i < i0 + 32; i++) S[i * SK + j] = Sgh[i * SK + j];
    __syncthreads();

    for (int t = 0; t < T; t++) {
        const float* conv = convT + (size_t)t * GDN_CH;
        if (it == 0) {
            sq[j] = conv[qk * SK + j] * scale;
            sk[j] = conv[2048 + qk * SK + j];
        }
        __syncthreads();
        const float decay = expf(gT[(size_t)t * NH + h]);
        float pred = 0.f;
#pragma unroll 8
        for (int i = i0; i < i0 + 32; i++) {
            float s = S[i * SK + j] * decay;
            S[i * SK + j] = s;
            pred += sk[i] * s;
        }
        part[it][j] = pred;
        __syncthreads();
        if (it == 0) {
            float p = part[0][j] + part[1][j] + part[2][j] + part[3][j];
            float vj = conv[4096 + h * SK + j];
            dj[j] = betaT[(size_t)t * NH + h] * (vj - p);
        }
        __syncthreads();
        float d = dj[j];
        float acc = 0.f;
#pragma unroll 8
        for (int i = i0; i < i0 + 32; i++) {
            float s = S[i * SK + j] + sk[i] * d;
            S[i * SK + j] = s;
            acc += sq[i] * s;
        }
        part[it][j] = acc;
        __syncthreads();
        if (it == 0)
            oT[(size_t)t * (NH * SK) + h * SK + j] =
                part[0][j] + part[1][j] + part[2][j] + part[3][j];
        __syncthreads();
    }

    for (int i = i0; i < i0 + 32; i++) Sgh[i * SK + j] = S[i * SK + j];
}

// P6: column-split scan. S columns are independent (pred_j, dj and o_j read
// only column j), so slice the 128 v-columns of each head across gridDim.x
// blocks and deepen row-parallelism per column: NTILE row tiles instead of 4,
// shrinking the two serial 32-iteration row loops to RPT = NCOL/4. Same
// 4-barriers-per-token structure (the legacy kernel's trailing barrier is
// covered by the next token's sq/sk barrier: the out-reduce reads part/conv
// only, never sq/sk). Row reductions change fp grouping vs k_delta_scan_T ->
// tolerance-gated in test_delta_split; Q27_DS_SPLIT=1 is the exact path.
// Slice index on blockIdx.x: blocks sharing a head reuse the same sq/sk rows
// (L2), and x is scheduled fastest.
template <int NCOL>
__global__ void k_delta_scan_split(float* __restrict__ Sg, const float* __restrict__ convT,
                                   const float* __restrict__ gT,
                                   const float* __restrict__ betaT, float* __restrict__ oT,
                                   int T) {
    constexpr int SK = 128;
    constexpr int GDN_CH = 10240;
    constexpr int NH = 48;
    constexpr int NTILE = 512 / NCOL;  // row tiles per column
    constexpr int RPT = SK / NTILE;    // rows per thread
    extern __shared__ float smem[];
    float* S = smem;  // [SK][NCOL]
    __shared__ float sq[SK], sk[SK], part[NTILE][NCOL + 1], dj[NCOL];
    const int h = blockIdx.y;
    const int c0 = blockIdx.x * NCOL;
    const int tid = threadIdx.x;
    const int j = tid % NCOL;
    const int it = tid / NCOL;
    const int i0 = it * RPT;
    const int qk = h % 16;
    const float scale = rsqrtf((float)SK);

    float* Sgh = Sg + (size_t)h * SK * SK;
    for (int i = i0; i < i0 + RPT; i++) S[i * NCOL + j] = Sgh[i * SK + c0 + j];

    for (int t = 0; t < T; t++) {
        const float* conv = convT + (size_t)t * GDN_CH;
        if (tid < SK) {
            sq[tid] = conv[qk * SK + tid] * scale;
            sk[tid] = conv[2048 + qk * SK + tid];
        }
        __syncthreads();
        const float decay = expf(gT[(size_t)t * NH + h]);
        float pred = 0.f;
#pragma unroll
        for (int i = i0; i < i0 + RPT; i++) {
            float s = S[i * NCOL + j] * decay;
            S[i * NCOL + j] = s;
            pred += sk[i] * s;
        }
        part[it][j] = pred;
        __syncthreads();
        if (tid < NCOL) {
            float p = 0.f;
#pragma unroll
            for (int r = 0; r < NTILE; r++) p += part[r][tid];
            float vj = conv[4096 + h * SK + c0 + tid];
            dj[tid] = betaT[(size_t)t * NH + h] * (vj - p);
        }
        __syncthreads();
        const float d = dj[j];
        float acc = 0.f;
#pragma unroll
        for (int i = i0; i < i0 + RPT; i++) {
            float s = S[i * NCOL + j] + sk[i] * d;
            S[i * NCOL + j] = s;
            acc += sq[i] * s;
        }
        part[it][j] = acc;
        __syncthreads();
        if (tid < NCOL) {
            float p = 0.f;
#pragma unroll
            for (int r = 0; r < NTILE; r++) p += part[r][tid];
            oT[(size_t)t * (NH * SK) + h * SK + c0 + tid] = p;
        }
    }

    for (int i = i0; i < i0 + RPT; i++) Sgh[i * SK + c0 + j] = S[i * NCOL + j];
}

int delta_scan_nsplit(int T) {
    (void)T;
    // auto default 8: 384 blocks / 170 SMs. Measured T=256 launch: 748us
    // exact, 428/456/413us at 2/4/8 (4 is reproducibly worst -- 192 blocks
    // is the awkward wave count); 26K real prefill 15.02s -> 13.48s.
    const char* e = getenv("Q27_DS_SPLIT");
    int n = e ? atoi(e) : 8;
    return n <= 1 ? 1 : n < 4 ? 2 : n < 8 ? 4 : 8;
}

// ---------------- chunked-WY delta scan (Q27_DS_MODE=wy) ----------------
// Reformulates the per-token rank-1 recurrence as per-64-token-chunk GEMMs +
// a forward substitution, sequential only ACROSS chunks (T/64 steps instead
// of T). Derivation validated to 1e-15 in f64 (scratchpad delta_wy.py):
//   Lam_t = prod_{s<=t} d_s (ratios Lam_t/Lam_s <= 1 only -- numerically safe)
//   (I + diag(beta) tril(ratio .* K K^T, -1)) rhat = diag(beta)(V - Lam K S0)
//   O   = Lam (Q S0) + tril(ratio .* Q K^T, 0) rhat
//   S_C = Lam_C S0 + K^T diag(Lam_C/Lam_s) rhat
// The solve is independent per v-column, so the state stays column-split.
// NOT bitwise vs the sequential scan (different reduction order) -- gated by
// the wy-vs-seq tolerance test in test_kernels.

constexpr int WY_C = 64; // chunk length

// Kernel A: per (qk-head, chunk): KKt[t][s] = k_t.k_s, QKt[t][s] = q_t.k_s
// (q pre-scaled by rsqrt(128) exactly as the sequential path does).
__global__ void k_delta_wy_kk(const float* __restrict__ convT, float* __restrict__ KKt,
                              float* __restrict__ QKt, int T, int nch) {
    constexpr int SK = 128, GDN_CH = 10240;
    const int qk = blockIdx.y;
    const int c = blockIdx.x;
    const int t0 = c * WY_C;
    const int rows = min(WY_C, T - t0);
    const float scale = rsqrtf((float)SK);
    extern __shared__ float akq[];
    float* skm = akq;              // [WY_C][SK]
    float* sqm = skm + WY_C * SK;  // [WY_C][SK]
    const int tid = threadIdx.x;   // 256
    for (int i = tid; i < rows * SK; i += blockDim.x) {
        int t = i / SK, d = i % SK;
        skm[i] = convT[(size_t)(t0 + t) * GDN_CH + 2048 + qk * SK + d];
        sqm[i] = convT[(size_t)(t0 + t) * GDN_CH + qk * SK + d];
    }
    __syncthreads();
    // lower-triangle pairs (incl diag) of KKt; full rows of QKt (s <= t used)
    for (int e = tid; e < WY_C * WY_C; e += blockDim.x) {
        int tt = e / WY_C, ss = e % WY_C;
        if (tt >= rows || ss > tt) continue;
        const float* kt = skm + tt * SK;
        const float* ks = skm + ss * SK;
        const float* qt = sqm + tt * SK;
        float dk = 0.f, dq = 0.f;
#pragma unroll 4
        for (int i = 0; i < SK; i++) {
            float kv = ks[i];
            dk += kt[i] * kv;
            dq += qt[i] * kv;
        }
        size_t base = ((size_t)qk * nch + c) * WY_C * WY_C;
        KKt[base + e] = dk;
        QKt[base + e] = dq * scale;
    }
}

// Kernel B: per (v-head, column-block of NCOLW): sequential over chunks.
// State S[:, j0:j0+NCOLW] lives in smem. Phases are warp-tiled register
// GEMMs (the one-thread-one-output scalar dots ran latency-bound at ~1
// instr/cycle): warp w owns token rows 8w..8w+7 with lane = column, K.S and
// Q.S accumulate together over ONE pass of S (Q.S rides in registers across
// the substitution, so the output phase never re-reads S -- which also lets
// the state update share its barrier region). K/Q rows come in as float4
// warp-broadcast __ldg (the chunk is L1/L2-hot; staging it in smem would
// break the 2-blocks/SM occupancy that covers the warp-0-only substitution).
constexpr int NCOLW = 32;
__global__ void __launch_bounds__(256, 2)
k_delta_wy(float* __restrict__ Sg, const float* __restrict__ convT,
           const float* __restrict__ gT, const float* __restrict__ betaT,
           float* __restrict__ oT, const float* __restrict__ KKt,
           const float* __restrict__ QKt, int T, int nch) {
    constexpr int SK = 128, GDN_CH = 10240, NH = 48;
    const int h = blockIdx.y;
    const int qk = h % 16;
    const int j0 = blockIdx.x * NCOLW;
    const int tid = threadIdx.x; // 256
    const int warp = tid >> 5, lane = tid & 31;
    const float qscale = rsqrtf((float)SK);

    extern __shared__ float wsm[];
    // sk (K chunk) and kk (KKt tile) stay in L2: sk is shared by the 4
    // col-blocks of the head and kk by 3 v-heads -- keeping them out of smem
    // halves the block footprint (82 -> 41KB) so 2 blocks/SM overlap.
    float* S = wsm;                    // [SK][NCOLW]   16KB
    float* rhat = S + SK * NCOLW;      // [WY_C][NCOLW]  8KB
    float* R = rhat + WY_C * NCOLW;    // [WY_C][WY_C]  16KB ratio expf(ll_t-ll_s)
    float* ll = R + WY_C * WY_C;       // [WY_C] log-lambda
    float* bm = ll + WY_C;             // [WY_C]
    float* lamv = bm + WY_C;           // [WY_C] expf(ll_t)
    float* wv = lamv + WY_C;           // [WY_C] expf(ll_C - ll_t)
    float* A = wv + WY_C;              // packed strict-lower bm.R.KKt   8KB

    float* Sgh = Sg + (size_t)h * SK * SK;
    for (int i = tid; i < SK * NCOLW; i += blockDim.x)
        S[i] = Sgh[(i / NCOLW) * SK + j0 + i % NCOLW];
    // ragged-tail rows are read through R == 0 terms in the tri product;
    // keep them finite (0 * NaN would poison live rows)
    for (int i = tid; i < WY_C * NCOLW; i += blockDim.x) rhat[i] = 0.f;

    for (int c = 0; c < nch; c++) {
        const int t0 = c * WY_C;
        const int rows = min(WY_C, T - t0);
        __syncthreads();
        if (tid < 32) {
            // pair-per-lane inclusive shuffle scan replaces the 64-step
            // serial prefix (reassociation only -- tolerance-gated)
            float g0 = 2 * lane < rows ? gT[(size_t)(t0 + 2 * lane) * NH + h] : 0.f;
            float g1 = 2 * lane + 1 < rows ? gT[(size_t)(t0 + 2 * lane + 1) * NH + h] : 0.f;
            float ps = g0 + g1;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                float v = __shfl_up_sync(0xffffffffu, ps, off);
                if (lane >= off) ps += v;
            }
            float base = ps - (g0 + g1);
            if (2 * lane < rows) ll[2 * lane] = base + g0;
            if (2 * lane + 1 < rows) ll[2 * lane + 1] = base + g0 + g1;
        }
        for (int t = tid; t < rows; t += blockDim.x)
            bm[t] = betaT[(size_t)(t0 + t) * NH + h];
        __syncthreads();
        // ratio matrix: R[t][s] = expf(ll_t - ll_s) for s <= t (log-space:
        // immune to lambda underflow; exp of a <=0 argument, underflow-to-0
        // is the correct fully-decayed limit)
        for (int e = tid; e < rows * WY_C; e += blockDim.x) {
            int tt = e / WY_C, ss = e % WY_C;
            R[e] = ss <= tt ? expf(ll[tt] - ll[ss]) : 0.f;
        }
        for (int t = tid; t < rows; t += blockDim.x) {
            lamv[t] = expf(ll[t]);
            wv[t] = expf(ll[rows - 1] - ll[t]);
        }
        __syncthreads();

        // fused K.S / Q.S (each 64x32 over k = 128): lane accumulates its
        // column for the warp's 8 rows; ragged rows clamp to row rows-1
        // (loads stay in-bounds, writes are guarded)
        const int tw = warp * 8;
        float accK[8], accQ[8];
#pragma unroll
        for (int r = 0; r < 8; r++) {
            accK[r] = 0.f;
            accQ[r] = 0.f;
        }
        if (tw + 8 <= rows) {
            // full band: one base pointer, row strides fold into the LDG
            // immediate (a per-row pointer array spilled at 128 regs)
            const float4* kp0 =
                (const float4*)(convT + (size_t)(t0 + tw) * GDN_CH + 2048 + qk * SK);
#pragma unroll 2
            for (int k4 = 0; k4 < SK / 4; k4++) {
                const float s0 = S[(4 * k4) * NCOLW + lane];
                const float s1 = S[(4 * k4 + 1) * NCOLW + lane];
                const float s2 = S[(4 * k4 + 2) * NCOLW + lane];
                const float s3 = S[(4 * k4 + 3) * NCOLW + lane];
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    float4 kv = __ldg(kp0 + r * (GDN_CH / 4) + k4);
                    accK[r] = fmaf(kv.x, s0, accK[r]);
                    accK[r] = fmaf(kv.y, s1, accK[r]);
                    accK[r] = fmaf(kv.z, s2, accK[r]);
                    accK[r] = fmaf(kv.w, s3, accK[r]);
                    // q row = k row - 2048 floats
                    float4 qv = __ldg(kp0 + r * (GDN_CH / 4) - 512 + k4);
                    accQ[r] = fmaf(qv.x, s0, accQ[r]);
                    accQ[r] = fmaf(qv.y, s1, accQ[r]);
                    accQ[r] = fmaf(qv.z, s2, accQ[r]);
                    accQ[r] = fmaf(qv.w, s3, accQ[r]);
                }
            }
        } else {
            // ragged band (cold: T tail only): clamp to row rows-1 with
            // inline address math (loads in-bounds, writes guarded)
            for (int k4 = 0; k4 < SK / 4; k4++) {
                const float s0 = S[(4 * k4) * NCOLW + lane];
                const float s1 = S[(4 * k4 + 1) * NCOLW + lane];
                const float s2 = S[(4 * k4 + 2) * NCOLW + lane];
                const float s3 = S[(4 * k4 + 3) * NCOLW + lane];
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    const float4* kp =
                        (const float4*)(convT + (size_t)(t0 + min(tw + r, rows - 1)) * GDN_CH +
                                        2048 + qk * SK);
                    float4 kv = __ldg(kp + k4);
                    accK[r] = fmaf(kv.x, s0, accK[r]);
                    accK[r] = fmaf(kv.y, s1, accK[r]);
                    accK[r] = fmaf(kv.z, s2, accK[r]);
                    accK[r] = fmaf(kv.w, s3, accK[r]);
                    float4 qv = __ldg(kp - 512 + k4);
                    accQ[r] = fmaf(qv.x, s0, accQ[r]);
                    accQ[r] = fmaf(qv.y, s1, accQ[r]);
                    accQ[r] = fmaf(qv.z, s2, accQ[r]);
                    accQ[r] = fmaf(qv.w, s3, accQ[r]);
                }
            }
        }
        // rhat[t][j] = beta_t (v_t[j] - lam_t * (k_t . S[:,j]))
#pragma unroll
        for (int r = 0; r < 8; r++) {
            const int t = tw + r;
            if (t < rows) {
                float vt = convT[(size_t)(t0 + t) * GDN_CH + 4096 + h * SK + j0 + lane];
                rhat[t * NCOLW + lane] = bm[t] * (vt - lamv[t] * accK[r]);
            }
        }
        // fold bm.R.KKt into a packed strict-lower triangle so the serial
        // substitution runs on smem only (its per-step L2 KKt loads were the
        // kernel's dominant stall), and QKt into R for the output pass (R is
        // read-only below; s > t entries stay 0); both read only R/bm and
        // globals, so they ride the pre-substitution barrier
        {
            const float* kkg = KKt + ((size_t)qk * nch + c) * WY_C * WY_C;
            for (int e = tid; e < rows * WY_C; e += blockDim.x) {
                int tt = e / WY_C, ss = e % WY_C;
                if (ss < tt) A[tt * (tt - 1) / 2 + ss] = bm[tt] * R[e] * __ldg(kkg + e);
            }
            const float* qkg = QKt + ((size_t)qk * nch + c) * WY_C * WY_C;
            for (int e = tid; e < rows * WY_C; e += blockDim.x)
                R[e] *= __ldg(qkg + e);
        }
        __syncthreads();

        // blocked forward substitution: 8-row diagonal blocks. Warp 0
        // solves the block serially (columns = lanes; each lane touches only
        // its own column, so no intra-warp sync), then all 8 warps apply the
        // rank-8 rhs update to the remaining rows -- the O(t) serial sums of
        // the flat solve were the kernel's dominant stall
        for (int tb = 0; tb + 1 < rows; tb += 8) {
            if (tid < 32) {
                // register triangular solve: solved rows stay in rc[] so the
                // chain never round-trips smem (full 8-row band unrolled)
                if (tb + 8 <= rows) {
                    float rc[8];
                    rc[0] = rhat[tb * NCOLW + tid];
#pragma unroll
                    for (int r = 1; r < 8; r++) {
                        const float* Ar = A + (tb + r) * (tb + r - 1) / 2 + tb;
                        float acc = 0.f;
#pragma unroll
                        for (int i = 0; i < r; i++) acc = fmaf(Ar[i], rc[i], acc);
                        rc[r] = rhat[(tb + r) * NCOLW + tid] - acc;
                        rhat[(tb + r) * NCOLW + tid] = rc[r];
                    }
                } else {
                    for (int t = tb + 1; t < rows; t++) {
                        const float* Ar = A + t * (t - 1) / 2;
                        float acc = 0.f;
                        for (int s = tb; s < t; s++)
                            acc = fmaf(Ar[s], rhat[s * NCOLW + tid], acc);
                        rhat[t * NCOLW + tid] -= acc;
                    }
                }
            }
            __syncthreads();
            // rank-8 rhs update of the remaining rows: the solved band is
            // loop-invariant, so it rides in registers; A row segments are
            // warp-broadcast loads feeding 4 independent chains
            float rr[8];
#pragma unroll
            for (int s = 0; s < 8; s++) rr[s] = rhat[(tb + s) * NCOLW + lane];
            for (int t = tb + 8 + warp; t < rows; t += 8) {
                const float* Ar = A + t * (t - 1) / 2 + tb;
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                a0 = fmaf(Ar[0], rr[0], a0);
                a1 = fmaf(Ar[1], rr[1], a1);
                a2 = fmaf(Ar[2], rr[2], a2);
                a3 = fmaf(Ar[3], rr[3], a3);
                a0 = fmaf(Ar[4], rr[4], a0);
                a1 = fmaf(Ar[5], rr[5], a1);
                a2 = fmaf(Ar[6], rr[6], a2);
                a3 = fmaf(Ar[7], rr[7], a3);
                rhat[t * NCOLW + lane] -= (a0 + a1) + (a2 + a3);
            }
            __syncthreads();
        }

        // outputs: lam_t qscale (Q.S, still in registers) + tril(R.QKt) rhat;
        // the s-loop runs the warp's full 8-row band -- s > t terms hit
        // R == 0 and contribute exact zeros
        {
            float accT[8];
#pragma unroll
            for (int r = 0; r < 8; r++) accT[r] = lamv[tw + r] * accQ[r] * qscale;
            for (int s4 = 0; s4 < (tw + 8) / 4; s4++) {
                const float r0 = rhat[(4 * s4) * NCOLW + lane];
                const float r1 = rhat[(4 * s4 + 1) * NCOLW + lane];
                const float r2 = rhat[(4 * s4 + 2) * NCOLW + lane];
                const float r3 = rhat[(4 * s4 + 3) * NCOLW + lane];
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    float4 a = *(const float4*)(R + (tw + r) * WY_C + 4 * s4);
                    accT[r] = fmaf(a.x, r0, accT[r]);
                    accT[r] = fmaf(a.y, r1, accT[r]);
                    accT[r] = fmaf(a.z, r2, accT[r]);
                    accT[r] = fmaf(a.w, r3, accT[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const int t = tw + r;
                if (t < rows)
                    oT[(size_t)(t0 + t) * (NH * SK) + h * SK + j0 + lane] = accT[r];
            }
        }

        // state: S = lam_C S + K^T (wv rhat); warp w owns state rows
        // 16w..16w+15. Same barrier region as the outputs: nothing above
        // reads S anymore, both only read post-substitution rhat
        {
            const float lC = wv[0] * lamv[0]; // = expf(ll_{rows-1})
            const int i0 = warp * 16;
            float acc[16];
#pragma unroll
            for (int r = 0; r < 16; r++) acc[r] = S[(i0 + r) * NCOLW + lane] * lC;
            const float* kr = convT + (size_t)t0 * GDN_CH + 2048 + qk * SK;
            for (int t = 0; t < rows; t++, kr += GDN_CH) {
                const float wr = wv[t] * rhat[t * NCOLW + lane];
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    float4 kv = __ldg((const float4*)kr + (i0 >> 2) + q);
                    acc[4 * q] = fmaf(kv.x, wr, acc[4 * q]);
                    acc[4 * q + 1] = fmaf(kv.y, wr, acc[4 * q + 1]);
                    acc[4 * q + 2] = fmaf(kv.z, wr, acc[4 * q + 2]);
                    acc[4 * q + 3] = fmaf(kv.w, wr, acc[4 * q + 3]);
                }
            }
#pragma unroll
            for (int r = 0; r < 16; r++) S[(i0 + r) * NCOLW + lane] = acc[r];
        }
    }
    __syncthreads();
    for (int i = tid; i < SK * NCOLW; i += blockDim.x)
        Sgh[(i / NCOLW) * SK + j0 + i % NCOLW] = S[i];
}

static void wy_grow(WyScratch* wy, int nch, cudaStream_t st) {
    // Regrow syncs the owning stream before freeing: earlier chunks of this
    // engine's own prefill may still be reading the old panels. A WyScratch
    // is pinned to one stream, so draining st is sufficient.
    if (nch <= wy->cap_nch) return;
    if (wy->kkt) {
        CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaFree(wy->kkt));
        CUDA_CHECK(cudaFree(wy->qkt));
    }
    const size_t bytes = (size_t)16 * nch * WY_C * WY_C * 4;
    CUDA_CHECK(cudaMalloc(&wy->kkt, bytes));
    CUDA_CHECK(cudaMalloc(&wy->qkt, bytes));
    // zero once per allocation: k_delta_wy's QKt fold reads the strict-upper
    // triangle the producer never writes, neutralized only by R == 0 there --
    // and 0 * NaN/Inf from recycled pages is NaN, poisoning live oT rows.
    // Zeros persist (no kernel stores those entries), so the reads are
    // defined and 0 * 0 keeps the exact-zero semantics bitwise. KKt reads
    // are all producer-guarded; zeroed for symmetry.
    CUDA_CHECK(cudaMemsetAsync(wy->kkt, 0, bytes, st));
    CUDA_CHECK(cudaMemsetAsync(wy->qkt, 0, bytes, st));
    wy->cap_nch = nch;
}

void wy_scratch_reserve(WyScratch* wy, int T_max) {
    wy_grow(wy, (T_max + WY_C - 1) / WY_C, 0);
}

static void delta_scan_wy(float* S_global, const float* convT, const float* gT,
                          const float* betaT, float* oT, int T, cudaStream_t st,
                          WyScratch* wy) {
    const int nch = (T + WY_C - 1) / WY_C;
    wy_grow(wy, nch, st); // no-op for engines: wy_scratch_reserve pre-sized
    {
        dim3 g(nch, 16);
        const size_t sma = (size_t)2 * WY_C * 128 * 4;
        static bool attra = false;
        if (!attra) {
            CUDA_CHECK(cudaFuncSetAttribute(k_delta_wy_kk,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, sma));
            attra = true;
        }
        k_delta_wy_kk<<<g, 256, sma, st>>>(convT, wy->kkt, wy->qkt, T, nch);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        dim3 g(128 / NCOLW, 48);
        const size_t sm = ((size_t)128 * NCOLW + (size_t)WY_C * NCOLW +
                           (size_t)WY_C * WY_C + 4 * WY_C +
                           (size_t)WY_C * (WY_C - 1) / 2) * 4;
        static bool attr = false;
        if (!attr) {
            CUDA_CHECK(cudaFuncSetAttribute(k_delta_wy,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
            attr = true;
        }
        k_delta_wy<<<g, 256, sm, st>>>(S_global, convT, gT, betaT, oT, wy->kkt, wy->qkt, T,
                                       nch);
        CUDA_CHECK(cudaGetLastError());
    }
}

void delta_scan_T(float* S_global, const float* convT, const float* gT, const float* betaT,
                  float* oT, int T, cudaStream_t st, WyScratch* wy) {
    // re-read per call (getenv is noise next to a launch) so tests can flip
    // paths in-process via setenv, same policy as prefill_use_mma.
    // wy DEFAULT since 2026-07-04 (2913 vs 2560 t/s @16K post-tiling; own
    // tolerance suite + canonical/pf gates). Q27_DS_MODE=seq restores the
    // sequential scan -- the full exact/identity configuration is
    // Q27_DS_MODE=seq Q27_PF_XG=32.
    const char* mode = getenv("Q27_DS_MODE");
    if (!(mode && !strcmp(mode, "seq"))) {
        delta_scan_wy(S_global, convT, gT, betaT, oT, T, st, wy);
        return;
    }
    const int cs = delta_scan_nsplit(T);
    if (cs == 1) {
        static bool attr_set = false;
        if (!attr_set) {
            CUDA_CHECK(cudaFuncSetAttribute(k_delta_scan_T,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            128 * 128 * 4));
            attr_set = true;
        }
        k_delta_scan_T<<<48, 512, 128 * 128 * 4, st>>>(S_global, convT, gT, betaT, oT, T);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    dim3 grid(cs, 48);
    const size_t sm = (size_t)128 * (128 / cs) * 4;
    switch (cs) {
        case 2:
            k_delta_scan_split<64><<<grid, 512, sm, st>>>(S_global, convT, gT, betaT, oT, T);
            break;
        case 4:
            k_delta_scan_split<32><<<grid, 512, sm, st>>>(S_global, convT, gT, betaT, oT, T);
            break;
        case 8:
            k_delta_scan_split<16><<<grid, 512, sm, st>>>(S_global, convT, gT, betaT, oT, T);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace q27k
