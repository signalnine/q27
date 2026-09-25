#include <cstdio>
#include <cstdlib>
#include "cuda_common.h"
#include "kernels.cuh"

// W16 retier (measured, tools/width_bench.cu on the 5090): the batched verify
// GEMV's 3-CTA/256-thread pin gives ~80 registers, and past N=9 the per-lane
// accumulators no longer fit -- ptxas spills, and the spill grows with N.
// Dropping those widths to a 2-CTA pin (128 regs) buys the accumulators back:
//
//   q4 ffn 17408x5120, ms/call     N=10    N=11    N=12    N=13    N=16
//     3-CTA (old, spilling)       0.0536  0.0638  0.0702  0.0800  0.180
//     2-CTA (this)                0.0476  0.0509  0.0554  0.0638  0.113
//
// N=9 is the crossover and stays 3-CTA (0.0411 vs 0.0448 for 2-CTA). The
// threshold therefore sits at 10 -- which leaves EVERY ladder width untouched:
// gated rounds only ever verify 2..gate_maxd+1 (<= 8), so the widths this moves
// are exactly the suffix rounds' (sfx_width() == W_MAX). Register allocation
// only; values are bit-identical (canonical-gated).
#ifndef Q27_GEMV_2CTA_MIN
#define Q27_GEMV_2CTA_MIN 10
#endif
// LADDER RETIER (measured 2026-07-13, same sweep): the 4-CTA/64-reg tier was
// never occupancy-swept -- it dates from the depth-4 era and spills at the
// widths the ladder actually verifies (ptxas: N=4 48B, N=5 36B, N=8 24B spill
// stores). 3 CTAs / 80 regs is faster at EVERY ladder width:
//
//   q4 ffn 17408x5120, ms/call    N=5     N=6     N=7     N=8
//     4-CTA (old)                0.0361  0.0338  0.0368  0.0438
//     3-CTA (this)               0.0332  0.0329  0.0341  0.0391
//
// N<=3 stays 4-CTA (the narrow gated graphs; untouched). Unlike the 2-CTA
// retier above -- which only helps suffix rounds -- this one hits EVERY gated
// round: canonical 139.8 -> 142.2 t/s and shortbench suite 171.9 -> 174.9
// (+1.7%), both bitwise EXACT. Smaller at depth (+0.7% @26K) where attention,
// not the weight GEMV, owns the round.
#ifndef Q27_GEMV_3CTA_MIN_Q4
#define Q27_GEMV_3CTA_MIN_Q4 4
#endif
#ifndef Q27_GEMV_3CTA_MIN_Q8
#define Q27_GEMV_3CTA_MIN_Q8 6
#endif

namespace q27k {

// ---------------- dequant ----------------

__global__ void k_dequant_q4(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                             float* __restrict__ out, int64_t rows, int64_t cols) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * cols;
    if (idx >= n) return;
    int64_t r = idx / cols, c = idx % cols;
    uint8_t b = W[r * (cols / 2) + c / 2];
    int nib = (c & 1) ? (b >> 4) : (b & 0xF);
    float s = __half2float(S[r * (cols / 64) + c / 64]);
    out[idx] = (nib - 8) * s;
}

__global__ void k_dequant_q8(const int8_t* __restrict__ W, const __half* __restrict__ S,
                             float* __restrict__ out, int64_t rows, int64_t cols) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows * cols;
    if (idx >= n) return;
    int64_t r = idx / cols, c = idx % cols;
    float s = __half2float(S[r * (cols / 128) + c / 128]);
    out[idx] = (float)W[r * cols + c] * s;
}

void dequant_q4(const uint8_t* W, const __half* S, float* out, int64_t rows, int64_t cols,
                cudaStream_t st) {
    int64_t n = rows * cols;
    k_dequant_q4<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(W, S, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}
void dequant_q8(const int8_t* W, const __half* S, float* out, int64_t rows, int64_t cols,
                cudaStream_t st) {
    int64_t n = rows * cols;
    k_dequant_q8<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(W, S, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- GEMV (reference) ----------------
// One block per output row, 256 threads grid-stride the reduction axis.

template <int BLOCK>
__device__ __forceinline__ float block_reduce(float v) {
    __shared__ float sh[BLOCK];
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    return sh[0];
}

__device__ __forceinline__ float warp_reduce(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// ---------------- activation quantization ----------------
// One warp per 32-element block: reduce max|x|, quantize, emit both byte orders.

__global__ void k_quantize_x(const float* __restrict__ x, int8_t* __restrict__ nat,
                             uint2* __restrict__ eo, float* __restrict__ scale,
                             int* __restrict__ isum, int nblocks) {
    int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (b >= nblocks) return;
    int lane = threadIdx.x & 31;
    float v = x[b * 32 + lane];
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    float s = amax / 127.f;
    float inv = s > 0.f ? 1.f / s : 0.f;
    int q = __float2int_rn(v * inv);
    q = max(-127, min(127, q));
    nat[b * 32 + lane] = (int8_t)q;
    int bsum = q;
    for (int off = 16; off > 0; off >>= 1) bsum += __shfl_xor_sync(0xffffffff, bsum, off);
    if (lane == 0) { scale[b] = s; isum[b] = bsum; }
    // even/odd packing: group u (0..3) covers elements u*8..u*8+7 of this block.
    // ALL lanes must execute the shuffles (divergent shfl_sync is UB); lanes >=4
    // compute redundant values and discard.
    int base = (lane & 3) * 8;
    uint32_t e = 0, o = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        int qe = __shfl_sync(0xffffffff, q, base + 2 * k);
        int qo = __shfl_sync(0xffffffff, q, base + 2 * k + 1);
        e |= (uint32_t)(uint8_t)(int8_t)qe << (8 * k);
        o |= (uint32_t)(uint8_t)(int8_t)qo << (8 * k);
    }
    if (lane < 4) eo[b * 4 + lane] = make_uint2(e, o);
}

// Group-64 variant: warp per 64-element group, lane owns elements g*64+lane
// and g*64+32+lane (both loads and stores coalesced). Emits nat64 + s64 only
// (no eo/isum -- the MMA path needs neither).
__global__ void k_quantize_x_g64(const float* __restrict__ x, int8_t* __restrict__ nat64,
                                 float* __restrict__ s64, int ngroups) {
    int g = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (g >= ngroups) return;
    int lane = threadIdx.x & 31;
    float v0 = x[g * 64 + lane];
    float v1 = x[g * 64 + 32 + lane];
    float amax = fmaxf(fabsf(v0), fabsf(v1));
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    float s = amax / 127.f;
    float inv = s > 0.f ? 1.f / s : 0.f;
    int q0 = max(-127, min(127, __float2int_rn(v0 * inv)));
    int q1 = max(-127, min(127, __float2int_rn(v1 * inv)));
    nat64[g * 64 + lane] = (int8_t)q0;
    nat64[g * 64 + 32 + lane] = (int8_t)q1;
    if (lane == 0) s64[g] = s;
}

XQuant xquant_alloc(int64_t max_cols, bool g64) {
    XQuant xq;
    CUDA_CHECK(cudaMalloc((void**)&xq.nat, max_cols));
    CUDA_CHECK(cudaMalloc((void**)&xq.eo, max_cols / 8 * sizeof(uint2)));
    CUDA_CHECK(cudaMalloc((void**)&xq.scale, max_cols / 32 * 4));
    CUDA_CHECK(cudaMalloc((void**)&xq.isum, max_cols / 32 * 4));
    if (g64) {
        CUDA_CHECK(cudaMalloc((void**)&xq.nat64, max_cols));
        CUDA_CHECK(cudaMalloc((void**)&xq.s64, max_cols / 64 * 4));
    }
    return xq;
}

void quantize_x_g64(const float* x, int64_t cols, const XQuant& xq, cudaStream_t st) {
    int ngroups = (int)(cols / 64);
    int warps = 8;
    k_quantize_x_g64<<<(ngroups + warps - 1) / warps, warps * 32, 0, st>>>(x, xq.nat64, xq.s64,
                                                                           ngroups);
    CUDA_CHECK(cudaGetLastError());
}

void quantize_x(const float* x, int64_t cols, const XQuant& xq, cudaStream_t st) {
    int nblocks = (int)(cols / 32);
    int warps = 8;
    k_quantize_x<<<(nblocks + warps - 1) / warps, warps * 32, 0, st>>>(x, xq.nat, xq.eo, xq.scale,
                                                                       xq.isum, nblocks);
    CUDA_CHECK(cudaGetLastError());
}

// Warp per row, dp4a against int8-quantized activations (mmvq-style).
// A 16-byte weight chunk covers 32 weights = one x-block: one w-scale, one x-scale.
__global__ void k_gemv_q4(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                          const int8_t* __restrict__ xnat, const uint2* __restrict__ xeo,
                          const float* __restrict__ xs, const int* __restrict__ xisum,
                          float* __restrict__ y, int64_t rows, int64_t cols) {
    (void)xnat;
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint4* wr = (const uint4*)(W + row * (cols / 2));
    const __half* sr = S + row * (cols / 64);
    const int n_chunks = (int)(cols / 32);

    float acc = 0.f;
    for (int ch = lane; ch < n_chunks; ch += 32) {
        uint4 w = __ldg(wr + ch);
        float s = __half2float(__ldg(sr + (ch >> 1))) * __ldg(xs + ch);
        // Task 2 (verify-gemv): 2x uint4 activation reads, same bytes/order as
        // the old 4x uint2 (bitwise; see k_gemv_q4_n for the Phase-0 rationale).
        const uint4* xp = (const uint4*)(xeo + (size_t)ch * 4);
        const uint4 xv0 = __ldg(xp), xv1 = __ldg(xp + 1);
        const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
        int di = 0;
        di = __dp4a((int)(ws[0] & 0x0F0F0F0Fu), (int)xv0.x, di);
        di = __dp4a((int)((ws[0] >> 4) & 0x0F0F0F0Fu), (int)xv0.y, di);
        di = __dp4a((int)(ws[1] & 0x0F0F0F0Fu), (int)xv0.z, di);
        di = __dp4a((int)((ws[1] >> 4) & 0x0F0F0F0Fu), (int)xv0.w, di);
        di = __dp4a((int)(ws[2] & 0x0F0F0F0Fu), (int)xv1.x, di);
        di = __dp4a((int)((ws[2] >> 4) & 0x0F0F0F0Fu), (int)xv1.y, di);
        di = __dp4a((int)(ws[3] & 0x0F0F0F0Fu), (int)xv1.z, di);
        di = __dp4a((int)((ws[3] >> 4) & 0x0F0F0F0Fu), (int)xv1.w, di);
        acc += s * (float)(di - 8 * __ldg(xisum + ch));
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// Warp per row; chunk = 32 int8 weights (two uint4 loads) = one x-block.
__global__ void k_gemv_q8(const int8_t* __restrict__ W, const __half* __restrict__ S,
                          const int8_t* __restrict__ xnat, const float* __restrict__ xs,
                          float* __restrict__ y, int64_t rows, int64_t cols) {
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint4* wr = (const uint4*)(W + row * cols);
    const uint4* xr = (const uint4*)xnat;
    const __half* sr = S + row * (cols / 128);
    const int n_chunks = (int)(cols / 32);

    float acc = 0.f;
    for (int ch = lane; ch < n_chunks; ch += 32) {
        uint4 w0 = __ldg(wr + 2 * ch), w1 = __ldg(wr + 2 * ch + 1);
        uint4 x0 = __ldg(xr + 2 * ch), x1 = __ldg(xr + 2 * ch + 1);
        float s = __half2float(__ldg(sr + (ch >> 2))) * __ldg(xs + ch);
        int di = 0;
        di = __dp4a((int)w0.x, (int)x0.x, di);
        di = __dp4a((int)w0.y, (int)x0.y, di);
        di = __dp4a((int)w0.z, (int)x0.z, di);
        di = __dp4a((int)w0.w, (int)x0.w, di);
        di = __dp4a((int)w1.x, (int)x1.x, di);
        di = __dp4a((int)w1.y, (int)x1.y, di);
        di = __dp4a((int)w1.z, (int)x1.z, di);
        di = __dp4a((int)w1.w, (int)x1.w, di);
        acc += s * (float)di;
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ---------------- batched GEMV (speculative verify) ----------------
// Same warp-per-row walk as the single-column kernels, but each weight chunk is
// dp4a'd against N activation columns. Weight bytes amortize N ways.

// P10-A0: lane args as by-value arrays so one template covers 2..10 lanes
// (fused 2-slot verify = 10). Same math order as the old 5-wide param lists.
// width-12 2026-07-10: slots widened to 16 (struct plumbing only; N=9,11,12
// stay uninstantiated until the P1 register-cliff measurement).
struct Q4Lanes {
    const uint2* eo[16];
    const float* xs[16];
    const int* is[16];
    float* y[16];
};
struct Q8Lanes {
    const int8_t* nat[16];
    const float* xs[16];
    float* y[16];
};

// maxd7 width-8 attribution (BUILDLOG 2026-07-09): N=8 naturally compiles to
// 68 regs -> 3 CTAs/SM (vs 64 regs / 4 CTAs at N<=7), and this latency-bound
// kernel (long_scoreboard-dominated, verify-gemv Phase 0) loses 25% of its
// latency-hiding warps: +16.5%/call. Pin 4 CTAs/SM through N=8 (ptxas fits
// 64 regs); N=10 keeps its natural 3. Register allocation only -- values are
// bit-identical (canonical-gated).
template <int N>
__global__ void __launch_bounds__(256, N < Q27_GEMV_3CTA_MIN_Q4 ? 4 : N < Q27_GEMV_2CTA_MIN ? 3 : 2)
    k_gemv_q4_n(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                __grid_constant__ const Q4Lanes L, int64_t rows, int64_t cols) {
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint4* wr = (const uint4*)(W + row * (cols / 2));
    const __half* sr = S + row * (cols / 64);
    const int n_chunks = (int)(cols / 32);

    const uint2* const* eos = L.eo;
    const float* const* xss = L.xs;
    const int* const* iss = L.is;
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; n++) acc[n] = 0.f;

    for (int ch = lane; ch < n_chunks; ch += 32) {
        uint4 w = __ldg(wr + ch);
        const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
        float wsc = __half2float(__ldg(sr + (ch >> 1)));
#pragma unroll
        for (int n = 0; n < N; n++) {
            // Task 2 (verify-gemv): activation reads as 2x uint4 instead of
            // 4x uint2 -- same 32 bytes, same component order into the same
            // dp4a sequence (integer-exact, fp acc order untouched, bitwise),
            // but half the L1TEX wavefronts. Phase 0 measured these 8B loads
            // at 32B lane stride as THE stall (long_scoreboard 90%, 10/32
            // bytes/sector); 16B loads double the per-instruction utilization.
            const uint4* xp = (const uint4*)(eos[n] + (size_t)ch * 4);
            const uint4 xv0 = __ldg(xp), xv1 = __ldg(xp + 1);
            int di = 0;
            di = __dp4a((int)(ws[0] & 0x0F0F0F0Fu), (int)xv0.x, di);
            di = __dp4a((int)((ws[0] >> 4) & 0x0F0F0F0Fu), (int)xv0.y, di);
            di = __dp4a((int)(ws[1] & 0x0F0F0F0Fu), (int)xv0.z, di);
            di = __dp4a((int)((ws[1] >> 4) & 0x0F0F0F0Fu), (int)xv0.w, di);
            di = __dp4a((int)(ws[2] & 0x0F0F0F0Fu), (int)xv1.x, di);
            di = __dp4a((int)((ws[2] >> 4) & 0x0F0F0F0Fu), (int)xv1.y, di);
            di = __dp4a((int)(ws[3] & 0x0F0F0F0Fu), (int)xv1.z, di);
            di = __dp4a((int)((ws[3] >> 4) & 0x0F0F0F0Fu), (int)xv1.w, di);
            acc[n] += wsc * __ldg(xss[n] + ch) * (float)(di - 8 * __ldg(iss[n] + ch));
        }
    }
    float* const* yy = L.y;
#pragma unroll
    for (int n = 0; n < N; n++) {
        float v = warp_reduce(acc[n]);
        if (lane == 0) yy[n][row] = v;
    }
}

// Same treatment: q8_n at N=8 needs 94 regs -> 2 CTAs/SM (N=6/7: 80 -> 3;
// N<=5: 64 -> 4). The v1.4 Q8 residual writers (ssm_out/attn_output) are
// mid-size and latency-sensitive like the Q4 mats. Pin 3 CTAs through N=8
// (85-reg budget, small spill beats the 2-CTA cliff); N<=5 keeps 4.
template <int N>
__global__ void __launch_bounds__(256, N < Q27_GEMV_3CTA_MIN_Q8 ? 4 : N < Q27_GEMV_2CTA_MIN ? 3 : 2)
    k_gemv_q8_n(const int8_t* __restrict__ W, const __half* __restrict__ S,
                __grid_constant__ const Q8Lanes L, int64_t rows, int64_t cols) {
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint4* wr = (const uint4*)(W + row * cols);
    const __half* sr = S + row * (cols / 128);
    const int n_chunks = (int)(cols / 32);

    const int8_t* const* nats = L.nat;
    const float* const* xss = L.xs;
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; n++) acc[n] = 0.f;

    for (int ch = lane; ch < n_chunks; ch += 32) {
        uint4 w0 = __ldg(wr + 2 * ch), w1 = __ldg(wr + 2 * ch + 1);
        float wsc = __half2float(__ldg(sr + (ch >> 2)));
#pragma unroll
        for (int n = 0; n < N; n++) {
            const uint4* xr = (const uint4*)nats[n];
            uint4 x0 = __ldg(xr + 2 * ch), x1 = __ldg(xr + 2 * ch + 1);
            int di = 0;
            di = __dp4a((int)w0.x, (int)x0.x, di);
            di = __dp4a((int)w0.y, (int)x0.y, di);
            di = __dp4a((int)w0.z, (int)x0.z, di);
            di = __dp4a((int)w0.w, (int)x0.w, di);
            di = __dp4a((int)w1.x, (int)x1.x, di);
            di = __dp4a((int)w1.y, (int)x1.y, di);
            di = __dp4a((int)w1.z, (int)x1.z, di);
            di = __dp4a((int)w1.w, (int)x1.w, di);
            acc[n] += wsc * __ldg(xss[n] + ch) * (float)di;
        }
    }
    float* const* yy = L.y;
#pragma unroll
    for (int n = 0; n < N; n++) {
        float v = warp_reduce(acc[n]);
        if (lane == 0) yy[n][row] = v;
    }
}

void gemv_q4_n(const uint8_t* W, const __half* S, const XQuant* q, int nb, float* const* ys,
               int64_t rows, int64_t cols, cudaStream_t st) {
    unsigned blocks = (unsigned)((rows + 7) / 8);
    Q4Lanes L;
    for (int i = 0; i < 16; i++) {
        const XQuant& qq = q[i < nb ? i : 0];
        L.eo[i] = qq.eo; L.xs[i] = qq.scale; L.is[i] = qq.isum;
        L.y[i] = ys[i < nb ? i : 0];
    }
    switch (nb) {
        case 2: k_gemv_q4_n<2><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 3: k_gemv_q4_n<3><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 4: k_gemv_q4_n<4><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 5: k_gemv_q4_n<5><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 6: k_gemv_q4_n<6><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 7: k_gemv_q4_n<7><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // maxd6 width-7
        case 8: k_gemv_q4_n<8><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // maxd7 width-8
        case 9: k_gemv_q4_n<9><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;  // width-12:
        case 10: k_gemv_q4_n<10><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 11: k_gemv_q4_n<11><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // suffix
        case 12: k_gemv_q4_n<12><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // widths 9..12
        case 13: k_gemv_q4_n<13><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // W16:
        case 14: k_gemv_q4_n<14><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 15: k_gemv_q4_n<15><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // suffix
        case 16: k_gemv_q4_n<16><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // widths 13..16
        default: fprintf(stderr, "gemv_q4_n: bad nbatch %d\n", nb); exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}

void gemv_q8_n(const int8_t* W, const __half* S, const XQuant* q, int nb, float* const* ys,
               int64_t rows, int64_t cols, cudaStream_t st) {
    unsigned blocks = (unsigned)((rows + 7) / 8);
    Q8Lanes L;
    for (int i = 0; i < 16; i++) {
        const XQuant& qq = q[i < nb ? i : 0];
        L.nat[i] = qq.nat; L.xs[i] = qq.scale;
        L.y[i] = ys[i < nb ? i : 0];
    }
    switch (nb) {
        case 2: k_gemv_q8_n<2><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 3: k_gemv_q8_n<3><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 4: k_gemv_q8_n<4><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 5: k_gemv_q8_n<5><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 6: k_gemv_q8_n<6><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 7: k_gemv_q8_n<7><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // maxd6 width-7
        case 8: k_gemv_q8_n<8><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // maxd7 width-8
        case 9: k_gemv_q8_n<9><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;  // width-12:
        case 10: k_gemv_q8_n<10><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 11: k_gemv_q8_n<11><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // suffix
        case 12: k_gemv_q8_n<12><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // widths 9..12
        case 13: k_gemv_q8_n<13><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // W16:
        case 14: k_gemv_q8_n<14><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 15: k_gemv_q8_n<15><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // suffix
        case 16: k_gemv_q8_n<16><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break; // widths 13..16
        default: fprintf(stderr, "gemv_q8_n: bad nbatch %d\n", nb); exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_gemv_f16(const __half* __restrict__ W, const float* __restrict__ x,
                           float* __restrict__ y, int64_t cols) {
    int64_t r = blockIdx.x;
    const __half* wr = W + r * cols;
    float acc = 0.f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
        acc += __half2float(wr[c]) * x[c];
    float sum = block_reduce<256>(acc);
    if (threadIdx.x == 0) y[r] = sum;
}

void gemv_q4(const uint8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st) {
    unsigned blocks = (unsigned)((rows + 7) / 8);
    k_gemv_q4<<<blocks, 256, 0, st>>>(W, S, xq.nat, xq.eo, xq.scale, xq.isum, y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}
void gemv_q8(const int8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st) {
    unsigned blocks = (unsigned)((rows + 7) / 8);
    k_gemv_q8<<<blocks, 256, 0, st>>>(W, S, xq.nat, xq.scale, y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}
void gemv_f16(const __half* W, const float* x, float* y, int64_t rows, int64_t cols,
              cudaStream_t st) {
    k_gemv_f16<<<(unsigned)rows, 256, 0, st>>>(W, x, y, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- grid-merged 3-token variants ----------------

__global__ void k_rmsnorm3(__grid_constant__ const CP3 xp, const float* __restrict__ w,
                           __grid_constant__ const P3 yp, int n, float eps) {
    const float* x = xp.p[blockIdx.x];
    float* y = yp.p[blockIdx.x];
    __shared__ float sh[32];
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += x[i] * x[i];
    acc = warp_reduce(acc);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
        v = warp_reduce(v);
        if (threadIdx.x == 0) sh[0] = v;
    }
    __syncthreads();
    float inv = rsqrtf(sh[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}
void rmsnorm3(CP3 x, const float* w, P3 y, int n, float eps, cudaStream_t st, int ntok) {
    k_rmsnorm3<<<ntok, 1024, 0, st>>>(x, w, y, n, eps);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_add3(__grid_constant__ const P3 xp, __grid_constant__ const CP3 yp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) xp.p[blockIdx.y][i] += yp.p[blockIdx.y][i];
}
void add3(P3 x, CP3 y, int n, cudaStream_t st, int ntok) {
    dim3 g((n + 255) / 256, ntok);
    k_add3<<<g, 256, 0, st>>>(x, y, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_silu_mul3(__grid_constant__ const P3 gp, __grid_constant__ const CP3 up,
                            int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = gp.p[blockIdx.y][i];
    gp.p[blockIdx.y][i] = (v / (1.f + expf(-v))) * up.p[blockIdx.y][i];
}
void silu_mul3(P3 g, CP3 u, int n, cudaStream_t st, int ntok) {
    dim3 gr((n + 255) / 256, ntok);
    k_silu_mul3<<<gr, 256, 0, st>>>(g, u, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void k_quantize_x3(__grid_constant__ const CP3 xp,
                              __grid_constant__ const XQ3 xq, int nblocks) {
    int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (b >= nblocks) return;
    const int t = blockIdx.y;
    const float* x = xp.p[t];
    // P12b lesson (lane-count landmine): every lane must select its OWN
    // buffers -- a fall-through overwrote lane 4's activation at ntok=6 and
    // corrupted the depth-5 verify (memcheck-blind). width-12 review: the
    // flat 8-pointer arg list was that landmine's descendant (its terminal
    // fall-through would have aliased lanes 8..11 onto lane 7), so lanes now
    // ride the XQ3 struct and index their own slot by construction.
    int8_t* nat = xq.q[t].nat;
    uint2* eo = xq.q[t].eo;
    float* scale = xq.q[t].scale;
    int* isum = xq.q[t].isum;
    int lane = threadIdx.x & 31;
    float v = x[b * 32 + lane];
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    float s = amax / 127.f;
    float inv = s > 0.f ? 1.f / s : 0.f;
    int q = __float2int_rn(v * inv);
    q = max(-127, min(127, q));
    nat[b * 32 + lane] = (int8_t)q;
    int bsum = q;
    for (int off = 16; off > 0; off >>= 1) bsum += __shfl_xor_sync(0xffffffff, bsum, off);
    if (lane == 0) { scale[b] = s; isum[b] = bsum; }
    int base = (lane & 3) * 8;
    uint32_t e = 0, o = 0;
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        int qe = __shfl_sync(0xffffffff, q, base + 2 * kk);
        int qo = __shfl_sync(0xffffffff, q, base + 2 * kk + 1);
        e |= (uint32_t)(uint8_t)(int8_t)qe << (8 * kk);
        o |= (uint32_t)(uint8_t)(int8_t)qo << (8 * kk);
    }
    if (lane < 4) eo[b * 4 + lane] = make_uint2(e, o);
}
void quantize3(CP3 x, int64_t cols, const XQ3& xq, cudaStream_t st, int ntok) {
    int nblocks = (int)(cols / 32);
    dim3 g((nblocks + 7) / 8, ntok);
    k_quantize_x3<<<g, 256, 0, st>>>(x, xq, nblocks);
    CUDA_CHECK(cudaGetLastError());
}

// Fused rmsnorm3 + quantize_x3 (2026-09-08): one block per lane runs
// k_rmsnorm3's body verbatim (y = x * inv * w), then, after a block barrier,
// k_quantize_x3's per-32-group body over the y it just wrote (warp w takes
// groups w, w+32, ...). Every value is computed by the same expressions on
// the same inputs, so y / nat / eo / scale / isum are bitwise those of the
// two-launch sequence (test_rmsnorm3q). Every rmsnorm3 in the verify forward
// is followed by a quantize of the same buffer -- 129 node pairs at width 8.
__global__ void k_rmsnorm3q(__grid_constant__ const CP3 xp, const float* __restrict__ w,
                            __grid_constant__ const P3 yp, __grid_constant__ const XQ3 xq, int n,
                            float eps) {
    const int t = blockIdx.x;
    const float* x = xp.p[t];
    float* y = yp.p[t];
    __shared__ float sh[32];
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += x[i] * x[i];
    acc = warp_reduce(acc);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
        v = warp_reduce(v);
        if (threadIdx.x == 0) sh[0] = v;
    }
    __syncthreads();
    float inv = rsqrtf(sh[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
    __syncthreads(); // y complete and visible block-wide before the quantize reads it
    // ---- k_quantize_x3 body, group b, lane = threadIdx.x & 31 ----
    int8_t* nat = xq.q[t].nat;
    uint2* eo = xq.q[t].eo;
    float* scale = xq.q[t].scale;
    int* isum = xq.q[t].isum;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    const int nblocks = n / 32;
    for (int b = warp; b < nblocks; b += nwarp) {
        float v = y[b * 32 + lane];
        float amax = fabsf(v);
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
        float s = amax / 127.f;
        float inv_s = s > 0.f ? 1.f / s : 0.f;
        int q = __float2int_rn(v * inv_s);
        q = max(-127, min(127, q));
        nat[b * 32 + lane] = (int8_t)q;
        int bsum = q;
        for (int off = 16; off > 0; off >>= 1) bsum += __shfl_xor_sync(0xffffffff, bsum, off);
        if (lane == 0) { scale[b] = s; isum[b] = bsum; }
        int base = (lane & 3) * 8;
        uint32_t e = 0, o = 0;
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            int qe = __shfl_sync(0xffffffff, q, base + 2 * kk);
            int qo = __shfl_sync(0xffffffff, q, base + 2 * kk + 1);
            e |= (uint32_t)(uint8_t)(int8_t)qe << (8 * kk);
            o |= (uint32_t)(uint8_t)(int8_t)qo << (8 * kk);
        }
        if (lane < 4) eo[b * 4 + lane] = make_uint2(e, o);
    }
}
void rmsnorm3q(CP3 x, const float* w, P3 y, const XQ3& xq, int n, float eps, cudaStream_t st,
               int ntok) {
    k_rmsnorm3q<<<ntok, 1024, 0, st>>>(x, w, y, xq, n, eps);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- elementwise ----------------

__global__ void k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w,
                          float* __restrict__ y, int n, float eps) {
    __shared__ float sh[32];
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += x[i] * x[i];
    acc = warp_reduce(acc);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
        v = warp_reduce(v);
        if (threadIdx.x == 0) sh[0] = v;
    }
    __syncthreads();
    float inv = rsqrtf(sh[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}

__global__ void k_silu_mul(const float* __restrict__ g, const float* __restrict__ u,
                           float* __restrict__ o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = g[i];
    o[i] = (v / (1.f + expf(-v))) * u[i];
}

__global__ void k_embed_row_q8(const int8_t* __restrict__ W, const __half* __restrict__ S,
                               const int* __restrict__ d_token, int64_t cols,
                               float* __restrict__ out) {
    int64_t row = *d_token;
    const int8_t* wr = W + row * cols;
    const __half* sr = S + row * (cols / 128);
    for (int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; c < cols;
         c += (int64_t)gridDim.x * blockDim.x)
        out[c] = (float)wr[c] * __half2float(sr[c / 128]);
}

void rmsnorm(const float* x, const float* w, float* y, int n, float eps, cudaStream_t st) {
    k_rmsnorm<<<1, 1024, 0, st>>>(x, w, y, n, eps);
    CUDA_CHECK(cudaGetLastError());
}
void silu_mul(const float* g, const float* u, float* o, int n, cudaStream_t st) {
    k_silu_mul<<<(n + 255) / 256, 256, 0, st>>>(g, u, o, n);
    CUDA_CHECK(cudaGetLastError());
}

// T2_G128 embedding rows (Bonsai 2 slim packs, 2026-09-19): element e of a
// row lives in the device-interleaved 16-code word e/16 (t2_interleave_device
// order), value = (code - 1) * scale[e/128]. Same float per element as the
// exact-Q8 row (int8 = trit), so lookups are bitwise the Q8 path's.
__device__ __forceinline__ float t2_row_elem(const uint8_t* __restrict__ row, const __half* __restrict__ sr,
                                             int64_t e) {
    const uint32_t w = ((const uint32_t*)row)[e >> 4];
    const int j = (int)(e & 15);
    const int f = j < 8 ? 4 * (j >> 1) + (j & 1) : 4 * ((j - 8) >> 1) + 2 + ((j - 8) & 1);
    const int code = (int)((w >> (2 * f)) & 3u);
    return (float)(code - 1) * __half2float(sr[e >> 7]);
}
__global__ void k_embed_row_t2(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                               const int* __restrict__ d_token, int64_t cols,
                               float* __restrict__ out) {
    int64_t row = *d_token;
    const uint8_t* wr = W + row * (cols / 4);
    const __half* sr = S + row * (cols / 128);
    for (int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; c < cols;
         c += (int64_t)gridDim.x * blockDim.x)
        out[c] = t2_row_elem(wr, sr, c);
}
void embed_row_t2(const uint8_t* W, const __half* S, const int* d_token, int64_t cols, float* out,
                  cudaStream_t st) {
    k_embed_row_t2<<<8, 256, 0, st>>>(W, S, d_token, cols, out);
    CUDA_CHECK(cudaGetLastError());
}
void embed_row_q8(const int8_t* W, const __half* S, const int* d_token, int64_t cols, float* out,
                  cudaStream_t st) {
    k_embed_row_q8<<<8, 256, 0, st>>>(W, S, d_token, cols, out);
    CUDA_CHECK(cudaGetLastError());
}

// ---- T2_G128 ternary GEMV (kernels.cuh for the layout contract) ----
// Warp per row like k_gemv_q4; a 32-element chunk is 8 weight bytes (two
// interleaved words), dp4a'd against the SAME even/odd activation words the
// Q4 kernel reads. value = code - 1, so sum((c-1)*x) = dp4a(c, x) - isum.
__device__ __forceinline__ int t2_dot32(uint2 w, const uint4 xv0, const uint4 xv1) {
    const uint32_t M = 0x03030303u;
    int di = 0;
    di = __dp4a((int)(w.x & M), (int)xv0.x, di);
    di = __dp4a((int)((w.x >> 2) & M), (int)xv0.y, di);
    di = __dp4a((int)((w.x >> 4) & M), (int)xv0.z, di);
    di = __dp4a((int)((w.x >> 6) & M), (int)xv0.w, di);
    di = __dp4a((int)(w.y & M), (int)xv1.x, di);
    di = __dp4a((int)((w.y >> 2) & M), (int)xv1.y, di);
    di = __dp4a((int)((w.y >> 4) & M), (int)xv1.z, di);
    di = __dp4a((int)((w.y >> 6) & M), (int)xv1.w, di);
    return di;
}
__global__ void k_gemv_t2(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                          const uint2* __restrict__ xeo, const float* __restrict__ xs,
                          const int* __restrict__ xisum, float* __restrict__ y, int64_t rows,
                          int64_t cols) {
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint2* wr = (const uint2*)(W + row * (cols / 4));
    const __half* sr = S + row * (cols / 128);
    const int n_chunks = (int)(cols / 32);
    float acc = 0.f;
    for (int ch = lane; ch < n_chunks; ch += 32) {
        const uint2 w = __ldg(wr + ch);
        const uint4* xp = (const uint4*)(xeo + (size_t)ch * 4);
        const uint4 xv0 = __ldg(xp), xv1 = __ldg(xp + 1);
        const float s = __half2float(__ldg(sr + (ch >> 2))) * __ldg(xs + ch);
        acc += s * (float)(t2_dot32(w, xv0, xv1) - __ldg(xisum + ch));
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}
struct T2Lanes {
    const uint2* eo[16];
    const float* xs[16];
    const int* is[16];
    float* y[16];
};
template <int N>
__global__ void __launch_bounds__(256, N < Q27_GEMV_3CTA_MIN_Q4 ? 4 : N < Q27_GEMV_2CTA_MIN ? 3 : 2)
    k_gemv_t2_n(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                __grid_constant__ const T2Lanes L, int64_t rows, int64_t cols) {
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint2* wr = (const uint2*)(W + row * (cols / 4));
    const __half* sr = S + row * (cols / 128);
    const int n_chunks = (int)(cols / 32);
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; n++) acc[n] = 0.f;
    for (int ch = lane; ch < n_chunks; ch += 32) {
        const uint2 w = __ldg(wr + ch);
        const float wsc = __half2float(__ldg(sr + (ch >> 2)));
#pragma unroll
        for (int n = 0; n < N; n++) {
            const uint4* xp = (const uint4*)(L.eo[n] + (size_t)ch * 4);
            const uint4 xv0 = __ldg(xp), xv1 = __ldg(xp + 1);
            acc[n] += wsc * __ldg(L.xs[n] + ch) *
                      (float)(t2_dot32(w, xv0, xv1) - __ldg(L.is[n] + ch));
        }
    }
#pragma unroll
    for (int n = 0; n < N; n++) {
        float v = warp_reduce(acc[n]);
        if (lane == 0) L.y[n][row] = v;
    }
}
void gemv_t2(const uint8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st) {
    unsigned blocks = (unsigned)((rows + 7) / 8);
    k_gemv_t2<<<blocks, 256, 0, st>>>(W, S, xq.eo, xq.scale, xq.isum, y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}
void gemv_t2_n(const uint8_t* W, const __half* S, const XQuant* q, int nb, float* const* ys,
               int64_t rows, int64_t cols, cudaStream_t st) {
    unsigned blocks = (unsigned)((rows + 7) / 8);
    T2Lanes L;
    for (int i = 0; i < 16; i++) {
        const XQuant& qq = q[i < nb ? i : 0];
        L.eo[i] = qq.eo; L.xs[i] = qq.scale; L.is[i] = qq.isum;
        L.y[i] = ys[i < nb ? i : 0];
    }
    switch (nb) {
        case 1: k_gemv_t2_n<1><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 2: k_gemv_t2_n<2><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 3: k_gemv_t2_n<3><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 4: k_gemv_t2_n<4><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 5: k_gemv_t2_n<5><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 6: k_gemv_t2_n<6><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 7: k_gemv_t2_n<7><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 8: k_gemv_t2_n<8><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 9: k_gemv_t2_n<9><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 10: k_gemv_t2_n<10><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 11: k_gemv_t2_n<11><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 12: k_gemv_t2_n<12><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 13: k_gemv_t2_n<13><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 14: k_gemv_t2_n<14><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 15: k_gemv_t2_n<15><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        case 16: k_gemv_t2_n<16><<<blocks, 256, 0, st>>>(W, S, L, rows, cols); break;
        default: fprintf(stderr, "gemv_t2_n: bad nbatch %d\n", nb); exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}
// Sequential 2-bit fields (element f at field f) -> the kernel order above.
__global__ void k_t2_interleave(uint32_t* __restrict__ w, uint64_t nwords) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nwords) return;
    const uint32_t v = w[i];
    uint32_t o = 0;
#pragma unroll
    for (int b = 0; b < 4; b++) {
        const uint32_t e0 = (v >> (2 * (2 * b))) & 3u;      // e[2b]
        const uint32_t e1 = (v >> (2 * (2 * b + 1))) & 3u;  // e[2b+1]
        const uint32_t e2 = (v >> (2 * (8 + 2 * b))) & 3u;  // e[8+2b]
        const uint32_t e3 = (v >> (2 * (9 + 2 * b))) & 3u;  // e[9+2b]
        o |= e0 << (2 * (4 * b + 0));
        o |= e1 << (2 * (4 * b + 1));
        o |= e2 << (2 * (4 * b + 2));
        o |= e3 << (2 * (4 * b + 3));
    }
    w[i] = o;
}
void t2_interleave_device(uint8_t* W, uint64_t bytes, cudaStream_t st) {
    if (bytes % 4) { fprintf(stderr, "t2_interleave: %llu bytes not word-aligned\n",
                             (unsigned long long)bytes); exit(1); }
    const uint64_t n = bytes / 4;
    k_t2_interleave<<<(unsigned)((n + 255) / 256), 256, 0, st>>>((uint32_t*)W, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---- T3_G128 ternary GEMV (kernels.cuh for the device layout contract) ----
// Geometry helpers shared by the relayout, the GEMVs and the T2 conversion.
__host__ __device__ __forceinline__ int t3_tail_u32(int ni) { // u32 per lane in a tail window
    int u = (8 * ni + 4) / 5;
    return (u + 1) & ~1;
}
__host__ __device__ __forceinline__ uint64_t t3_row_bytes_hd(uint64_t cols) {
    const uint64_t nch = cols / 32, nfull = nch / 160, tail = nch % 160;
    uint64_t b = nfull * 1024;
    if (tail) b += 32ull * 4ull * (uint64_t)t3_tail_u32((int)((tail + 31) / 32));
    return b;
}
uint64_t t3_row_bytes(uint64_t cols) { return t3_row_bytes_hd(cols); }
uint64_t t3_device_bytes(uint64_t rows, uint64_t cols) { return rows * t3_row_bytes_hd(cols); }

// One u32 = four scaled base-3 bytes; next() pops the next (most significant
// remaining) code of every byte into the four byte lanes of a dp4a word. The
// two 16-bit lanes keep the *3 carry inside a lane (3 * 255 < 65536).
// Per round: two IMADs (FMA pipe) + one PRMT + two ANDs on the INT pipe. The
// popped code of a 16-bit lane sits alone in that lane's high byte (3*255 <
// 1024), so one __byte_perm gathers the four codes into the four byte lanes
// (bytes e3.1, o3.1, e3.3, o3.3); the ANDs drop them for the next round. The
// 3090 measures this GEMV INT-issue-bound at parity with its DRAM time, so
// every op here is a measurable fraction of the T3 decode step.
// 2026-09-25: the pop no longer chains through a masked remainder. With q the
// scaled byte, T_r = floor(3^r * q / 256) and digit_r = T_{r+1} - 3*T_r
// (exact; brute-forced over all 243 values). Each T comes straight from the
// ORIGINAL lanes (e * 3^(r+1) < 65536, so floor(/256) is the lane's high
// byte, gathered by one PRMT), and because every byte of the difference is
// 0..2, the four-lane subtraction is one plain 32-bit IMAD with no borrow
// between bytes. Per u32: 21 ops for its five words (was 27), and the five
// rounds no longer depend on each other. k is constant-folded once the
// callers' loops unroll.
struct T3Dec {
    uint32_t e, o, t, k;
    __device__ __forceinline__ void init(uint32_t w) {
        e = w & 0x00FF00FFu;
        o = __byte_perm(w, 0u, 0x4341); // bytes {w.1, 0, w.3, 0}
        t = 0u;                          // T_0 = floor(q/256) = 0 (q <= 255)
        k = 3u;
    }
    __device__ __forceinline__ uint32_t next() {
        const uint32_t T = __byte_perm(e * k, o * k, 0x7351); // {E.1, O.1, E.3, O.3}
        const uint32_t d = T - 3u * t;
        t = T;
        k *= 3u;
        return d;
    }
};
// Words 8I..8I+7 of a unit (chunk I's eight dp4a words) from its u32s; the
// decoders are stepped in word order across the unrolled chunk sequence, so
// dec[] must be carried from chunk I to I+1 by the caller (static indices).
template <int I>
__device__ __forceinline__ void t3_chunk_words(const uint32_t* __restrict__ w, T3Dec* dec,
                                               uint32_t (&cw)[8]) {
#pragma unroll
    for (int q = 0; q < 8; q++) {
        const int qu = 8 * I + q, k = qu / 5, r = qu % 5;
        if (r == 0) dec[k].init(w[k]);
        cw[q] = dec[k].next();
    }
}
__device__ __forceinline__ int t3_dot32(const uint32_t (&cw)[8], const uint4 xv0, const uint4 xv1) {
    int di = 0;
    di = __dp4a((int)cw[0], (int)xv0.x, di);
    di = __dp4a((int)cw[1], (int)xv0.y, di);
    di = __dp4a((int)cw[2], (int)xv0.z, di);
    di = __dp4a((int)cw[3], (int)xv0.w, di);
    di = __dp4a((int)cw[4], (int)xv1.x, di);
    di = __dp4a((int)cw[5], (int)xv1.y, di);
    di = __dp4a((int)cw[6], (int)xv1.z, di);
    di = __dp4a((int)cw[7], (int)xv1.w, di);
    return di;
}
// Same float chain as k_gemv_t2 per chunk (s = wscale * xs; acc += s * (dot -
// isum)), same lane -> chunk map and order: the two kernels are bitwise on the
// same ternary matrix.
// Pointers arrive pre-offset to the unit's first chunk (ch0); chunk I of the
// unit is ch0 + 32*I, so every per-chunk offset is a compile-time constant
// the loads fold into their immediates ((ch0 + 32I) >> 2 == (ch0 >> 2) + 8I).
// CK: bounds-check the chunk (tail windows only; a full window never runs past
// n_chunks).
template <int I, bool CK>
__device__ __forceinline__ void t3_acc_chunk(float& acc, const uint32_t (&cw)[8], int ch0,
                                             int n_chunks, const __half* __restrict__ sr0,
                                             const uint4* __restrict__ xp0,
                                             const float* __restrict__ xs0,
                                             const int* __restrict__ is0) {
    if (CK && ch0 + 32 * I >= n_chunks) return;
    const uint4 xv0 = __ldg(xp0 + 64 * I), xv1 = __ldg(xp0 + 64 * I + 1);
    const float s = __half2float(__ldg(sr0 + 8 * I)) * __ldg(xs0 + 32 * I);
    acc += s * (float)(t3_dot32(cw, xv0, xv1) - __ldg(is0 + 32 * I));
}
template <int NI, bool CK>
__device__ __forceinline__ void t3_unit_acc(float& acc, const uint32_t* __restrict__ w, int ch0,
                                            int n_chunks, const __half* __restrict__ sr,
                                            const uint2* __restrict__ xeo,
                                            const float* __restrict__ xs,
                                            const int* __restrict__ xisum) {
    const __half* sr0 = sr + (ch0 >> 2);
    const uint4* xp0 = (const uint4*)(xeo + (size_t)ch0 * 4);
    const float* xs0 = xs + ch0;
    const int* is0 = xisum + ch0;
    T3Dec dec[8];
    uint32_t cw[8];
    if (NI > 0) { t3_chunk_words<0>(w, dec, cw); t3_acc_chunk<0, CK>(acc, cw, ch0, n_chunks, sr0, xp0, xs0, is0); }
    if (NI > 1) { t3_chunk_words<1>(w, dec, cw); t3_acc_chunk<1, CK>(acc, cw, ch0, n_chunks, sr0, xp0, xs0, is0); }
    if (NI > 2) { t3_chunk_words<2>(w, dec, cw); t3_acc_chunk<2, CK>(acc, cw, ch0, n_chunks, sr0, xp0, xs0, is0); }
    if (NI > 3) { t3_chunk_words<3>(w, dec, cw); t3_acc_chunk<3, CK>(acc, cw, ch0, n_chunks, sr0, xp0, xs0, is0); }
    if (NI > 4) { t3_chunk_words<4>(w, dec, cw); t3_acc_chunk<4, CK>(acc, cw, ch0, n_chunks, sr0, xp0, xs0, is0); }
}
__global__ void k_gemv_t3(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                          const uint2* __restrict__ xeo, const float* __restrict__ xs,
                          const int* __restrict__ xisum, float* __restrict__ y, int64_t rows,
                          int64_t cols, uint64_t row_bytes) {
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint8_t* wr = W + row * row_bytes;
    const __half* sr = S + row * (cols / 128);
    const int n_chunks = (int)(cols / 32), n_full = n_chunks / 160, tail = n_chunks % 160;
    float acc = 0.f;
    // half-split window: u32 0..3 of every lane's unit sit lane-major in the
    // first 512 B, u32 4..7 in the second, so each LDG.128 is 512 contiguous B.
    // (Prefetching window m+1 before decoding m measured no gain single-lane
    // and -20% at width 2 from register pressure, 2026-09-25; not done.)
    for (int m = 0; m < n_full; m++) {
        const uint4 a = __ldg((const uint4*)(wr + (size_t)m * 1024 + lane * 16));
        const uint4 b = __ldg((const uint4*)(wr + (size_t)m * 1024 + 512 + lane * 16));
        const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        t3_unit_acc<5, false>(acc, w, 160 * m + lane, n_chunks, sr, xeo, xs, xisum);
    }
    if (tail) {
        const int ni = (tail + 31) / 32, nu = t3_tail_u32(ni), ch0 = 160 * n_full + lane;
        const uint32_t* up = (const uint32_t*)(wr + (size_t)n_full * 1024 + lane * 4 * nu);
        uint32_t w[8] = {0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u,
                         0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u};
        if (nu >= 2) { const uint2 v = __ldg((const uint2*)up); w[0] = v.x; w[1] = v.y; }
        if (nu >= 4) { const uint2 v = __ldg((const uint2*)up + 1); w[2] = v.x; w[3] = v.y; }
        if (nu >= 6) { const uint2 v = __ldg((const uint2*)up + 2); w[4] = v.x; w[5] = v.y; }
        if (nu >= 8) { const uint2 v = __ldg((const uint2*)up + 3); w[6] = v.x; w[7] = v.y; }
        switch (ni) { // warp-uniform
            case 1: t3_unit_acc<1, true>(acc, w, ch0, n_chunks, sr, xeo, xs, xisum); break;
            case 2: t3_unit_acc<2, true>(acc, w, ch0, n_chunks, sr, xeo, xs, xisum); break;
            case 3: t3_unit_acc<3, true>(acc, w, ch0, n_chunks, sr, xeo, xs, xisum); break;
            default: t3_unit_acc<4, true>(acc, w, ch0, n_chunks, sr, xeo, xs, xisum); break;
        }
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}
// Multi-lane twin (mirrors k_gemv_t2_n's chain: acc += wsc * xs * (dot - isum)).
template <int N, int I, bool CK>
__device__ __forceinline__ void t3_acc_chunk_n(float* acc, const uint32_t (&cw)[8], int ch0,
                                               int n_chunks, const __half* __restrict__ sr0,
                                               const T2Lanes& L) {
    if (CK && ch0 + 32 * I >= n_chunks) return;
    const int ch = ch0 + 32 * I;
    const float wsc = __half2float(__ldg(sr0 + 8 * I));
#pragma unroll
    for (int n = 0; n < N; n++) {
        const uint4* xp = (const uint4*)(L.eo[n] + (size_t)ch * 4);
        const uint4 xv0 = __ldg(xp), xv1 = __ldg(xp + 1);
        acc[n] += wsc * __ldg(L.xs[n] + ch) * (float)(t3_dot32(cw, xv0, xv1) - __ldg(L.is[n] + ch));
    }
}
template <int N, int NI, bool CK>
__device__ __forceinline__ void t3_unit_acc_n(float* acc, const uint32_t* __restrict__ w, int ch0,
                                              int n_chunks, const __half* __restrict__ sr,
                                              const T2Lanes& L) {
    const __half* sr0 = sr + (ch0 >> 2);
    T3Dec dec[8];
    uint32_t cw[8];
    if (NI > 0) { t3_chunk_words<0>(w, dec, cw); t3_acc_chunk_n<N, 0, CK>(acc, cw, ch0, n_chunks, sr0, L); }
    if (NI > 1) { t3_chunk_words<1>(w, dec, cw); t3_acc_chunk_n<N, 1, CK>(acc, cw, ch0, n_chunks, sr0, L); }
    if (NI > 2) { t3_chunk_words<2>(w, dec, cw); t3_acc_chunk_n<N, 2, CK>(acc, cw, ch0, n_chunks, sr0, L); }
    if (NI > 3) { t3_chunk_words<3>(w, dec, cw); t3_acc_chunk_n<N, 3, CK>(acc, cw, ch0, n_chunks, sr0, L); }
    if (NI > 4) { t3_chunk_words<4>(w, dec, cw); t3_acc_chunk_n<N, 4, CK>(acc, cw, ch0, n_chunks, sr0, L); }
}
// widths 1-2 fit T2's 4-CTA tier at 64 regs (ptxas: 52/64); 3 spills there
template <int N>
__global__ void __launch_bounds__(256, N < 3 ? 4 : N < Q27_GEMV_3CTA_MIN_Q4 ? 3 : 2)
    k_gemv_t3_n(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                __grid_constant__ const T2Lanes L, int64_t rows, int64_t cols, uint64_t row_bytes) {
    int64_t row = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint8_t* wr = W + row * row_bytes;
    const __half* sr = S + row * (cols / 128);
    const int n_chunks = (int)(cols / 32), n_full = n_chunks / 160, tail = n_chunks % 160;
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; n++) acc[n] = 0.f;
    for (int m = 0; m < n_full; m++) {
        const uint4 a = __ldg((const uint4*)(wr + (size_t)m * 1024 + lane * 16));
        const uint4 b = __ldg((const uint4*)(wr + (size_t)m * 1024 + 512 + lane * 16));
        const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        t3_unit_acc_n<N, 5, false>(acc, w, 160 * m + lane, n_chunks, sr, L);
    }
    if (tail) {
        const int ni = (tail + 31) / 32, nu = t3_tail_u32(ni), ch0 = 160 * n_full + lane;
        const uint32_t* up = (const uint32_t*)(wr + (size_t)n_full * 1024 + lane * 4 * nu);
        uint32_t w[8] = {0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u,
                         0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u};
        if (nu >= 2) { const uint2 v = __ldg((const uint2*)up); w[0] = v.x; w[1] = v.y; }
        if (nu >= 4) { const uint2 v = __ldg((const uint2*)up + 1); w[2] = v.x; w[3] = v.y; }
        if (nu >= 6) { const uint2 v = __ldg((const uint2*)up + 2); w[4] = v.x; w[5] = v.y; }
        if (nu >= 8) { const uint2 v = __ldg((const uint2*)up + 3); w[6] = v.x; w[7] = v.y; }
        switch (ni) {
            case 1: t3_unit_acc_n<N, 1, true>(acc, w, ch0, n_chunks, sr, L); break;
            case 2: t3_unit_acc_n<N, 2, true>(acc, w, ch0, n_chunks, sr, L); break;
            case 3: t3_unit_acc_n<N, 3, true>(acc, w, ch0, n_chunks, sr, L); break;
            default: t3_unit_acc_n<N, 4, true>(acc, w, ch0, n_chunks, sr, L); break;
        }
    }
#pragma unroll
    for (int n = 0; n < N; n++) {
        float v = warp_reduce(acc[n]);
        if (lane == 0) L.y[n][row] = v;
    }
}
static void t3_check_cols(int64_t cols, const char* who) {
    if (cols % 128) { fprintf(stderr, "%s: cols %lld not a multiple of 128\n", who, (long long)cols); exit(1); }
}
void gemv_t3(const uint8_t* W, const __half* S, const XQuant& xq, float* y, int64_t rows,
             int64_t cols, cudaStream_t st) {
    t3_check_cols(cols, "gemv_t3");
    unsigned blocks = (unsigned)((rows + 7) / 8);
    k_gemv_t3<<<blocks, 256, 0, st>>>(W, S, xq.eo, xq.scale, xq.isum, y, rows, cols,
                                      t3_row_bytes_hd((uint64_t)cols));
    CUDA_CHECK(cudaGetLastError());
}
void gemv_t3_n(const uint8_t* W, const __half* S, const XQuant* q, int nb, float* const* ys,
               int64_t rows, int64_t cols, cudaStream_t st) {
    t3_check_cols(cols, "gemv_t3_n");
    unsigned blocks = (unsigned)((rows + 7) / 8);
    const uint64_t rb = t3_row_bytes_hd((uint64_t)cols);
    T2Lanes L;
    for (int i = 0; i < 16; i++) {
        const XQuant& qq = q[i < nb ? i : 0];
        L.eo[i] = qq.eo; L.xs[i] = qq.scale; L.is[i] = qq.isum;
        L.y[i] = ys[i < nb ? i : 0];
    }
    switch (nb) {
        case 1: k_gemv_t3_n<1><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 2: k_gemv_t3_n<2><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 3: k_gemv_t3_n<3><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 4: k_gemv_t3_n<4><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 5: k_gemv_t3_n<5><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 6: k_gemv_t3_n<6><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 7: k_gemv_t3_n<7><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 8: k_gemv_t3_n<8><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 9: k_gemv_t3_n<9><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 10: k_gemv_t3_n<10><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 11: k_gemv_t3_n<11><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 12: k_gemv_t3_n<12><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 13: k_gemv_t3_n<13><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 14: k_gemv_t3_n<14><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 15: k_gemv_t3_n<15><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        case 16: k_gemv_t3_n<16><<<blocks, 256, 0, st>>>(W, S, L, rows, cols, rb); break;
        default: fprintf(stderr, "gemv_t3_n: bad nbatch %d\n", nb); exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}

// FORMAT.md bytes (26 per 128-group, c0 least significant) -> the device
// layout above. Thread per output u32; load-time only, so the per-element
// base-3 division is fine. Every pad slot (tail-window words past the row's
// chunks, unused rounds) is code 1 = byte 128.
__device__ __forceinline__ int t3_host_code(const uint8_t* __restrict__ row26, int e) {
    const int g = e >> 7, j = e & 127;
    int v = row26[g * 26 + (j / 5)];
    const int d = j % 5;
    if (d >= 1) v /= 3;
    if (d >= 2) v /= 3;
    if (d >= 3) v /= 3;
    if (d >= 4) v /= 3;
    return v % 3;
}
__global__ void k_t3_relayout(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int64_t rows,
                              int64_t cols, uint64_t row_bytes, uint64_t total_u32) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_u32) return;
    const uint64_t row_u32 = row_bytes / 4;
    const int64_t row = (int64_t)(idx / row_u32);
    const int off = (int)(idx % row_u32) * 4;
    const int n_chunks = (int)(cols / 32), n_full = n_chunks / 160;
    int m, L, k, ni;
    if (off < n_full * 1024) {
        const int wo = off % 1024, half = wo / 512;
        m = off / 1024; L = (wo % 512) / 16; k = half * 4 + (wo % 16) / 4; ni = 5;
    } else {
        const int tail = n_chunks - n_full * 160;
        ni = (tail + 31) / 32;
        const int nu = t3_tail_u32(ni), toff = off - n_full * 1024;
        m = n_full; L = toff / (4 * nu); k = (toff % (4 * nu)) / 4;
    }
    const uint8_t* row26 = src + (uint64_t)row * (uint64_t)(cols / 128) * 26ull;
    uint32_t out = 0;
#pragma unroll
    for (int b = 0; b < 4; b++) {
        int V = 0;
#pragma unroll
        for (int r = 0; r < 5; r++) {
            const int qu = 5 * k + r, i = qu / 8, q = qu % 8;
            int code = 1;
            if (i < ni) {
                const int c = 160 * m + 32 * i + L;
                if (c < n_chunks) code = t3_host_code(row26, 32 * c + 8 * (q / 2) + 2 * b + (q % 2));
            }
            V = V * 3 + code;
        }
        out |= (uint32_t)((V * 256 + 242) / 243) << (8 * b);
    }
    ((uint32_t*)dst)[idx] = out;
}
void t3_relayout_device(const uint8_t* src26, uint8_t* dst, int64_t rows, int64_t cols,
                        cudaStream_t st) {
    t3_check_cols(cols, "t3_relayout_device");
    const uint64_t rb = t3_row_bytes_hd((uint64_t)cols), n = (uint64_t)rows * rb / 4;
    k_t3_relayout<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(src26, dst, rows, cols, rb, n);
    CUDA_CHECK(cudaGetLastError());
}

// T3 device layout -> T2 device-interleaved words (kernels.cuh order: field
// 4b+s of the 16-code word = eo word s of that 16-run, lane b), i.e. T2 word
// = W0 | W1<<2 | W2<<4 | W3<<6 over the four dp4a words 4h..4h+3 of chunk c
// (16-run h). Block = one (row, window); warp = one (chunk-in-unit i, half h)
// over the 32 lanes' units, so the first-word residue R0 = (8i+4h)%5 is
// warp-uniform and every decode is static per instantiation.
// lo/hi: the unit's u32 0..3 and 4..7 (a full window keeps them 512 B apart,
// a tail unit is contiguous so hi = lo + 4)
template <int QU0>
__device__ __forceinline__ uint32_t t3_t2_word(const uint32_t* __restrict__ lo,
                                               const uint32_t* __restrict__ hi) {
    constexpr int K0 = QU0 / 5, R0 = QU0 % 5, K1 = K0 + 1;
    uint32_t Wd[4] = {0u, 0u, 0u, 0u};
    T3Dec d;
    d.init(__ldg(K0 < 4 ? lo + K0 : hi + (K0 - 4)));
#pragma unroll
    for (int r = 0; r < 5; r++) {
        const uint32_t x = d.next();
        if (r >= R0 && r - R0 < 4) Wd[r - R0] = x;
    }
    if constexpr (R0 > 1) {
        T3Dec d2;
        d2.init(__ldg(K1 < 4 ? lo + K1 : hi + (K1 - 4)));
#pragma unroll
        for (int r = 0; r < R0 - 1; r++) Wd[5 - R0 + r] = d2.next();
    }
    return Wd[0] | (Wd[1] << 2) | (Wd[2] << 4) | (Wd[3] << 6);
}
__global__ void __launch_bounds__(320) k_t3_to_t2(const uint8_t* __restrict__ W3, uint8_t* __restrict__ W2,
                                                 int64_t rows, int64_t cols, uint64_t row_bytes) {
    const int n_chunks = (int)(cols / 32), n_full = n_chunks / 160, tail = n_chunks % 160;
    const int n_win = n_full + (tail ? 1 : 0);
    const int64_t row = blockIdx.x / n_win;
    const int m = blockIdx.x % n_win;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int i = warp / 2, h = warp % 2;
    const int ni = m < n_full ? 5 : (tail + 31) / 32;
    if (i >= ni) return; // tail windows use the first 2*ni warps
    const int c = 160 * m + 32 * i + lane;
    if (c >= n_chunks) return;
    const uint8_t* wr = W3 + (uint64_t)row * row_bytes;
    const uint32_t* lo;
    const uint32_t* hi;
    if (m < n_full) {
        lo = (const uint32_t*)(wr + (size_t)m * 1024 + lane * 16);
        hi = (const uint32_t*)(wr + (size_t)m * 1024 + 512 + lane * 16);
    } else {
        lo = (const uint32_t*)(wr + (size_t)n_full * 1024 + lane * 4 * t3_tail_u32(ni));
        hi = lo + 4;
    }
    uint32_t o;
    switch (warp) { // warp-uniform
        case 0: o = t3_t2_word<0>(lo, hi); break;
        case 1: o = t3_t2_word<4>(lo, hi); break;
        case 2: o = t3_t2_word<8>(lo, hi); break;
        case 3: o = t3_t2_word<12>(lo, hi); break;
        case 4: o = t3_t2_word<16>(lo, hi); break;
        case 5: o = t3_t2_word<20>(lo, hi); break;
        case 6: o = t3_t2_word<24>(lo, hi); break;
        case 7: o = t3_t2_word<28>(lo, hi); break;
        case 8: o = t3_t2_word<32>(lo, hi); break;
        default: o = t3_t2_word<36>(lo, hi); break;
    }
    ((uint32_t*)(W2 + (uint64_t)row * (cols / 4)))[2 * c + h] = o;
}
void t3_to_t2_device(const uint8_t* W3, uint8_t* W2, int64_t rows, int64_t cols, cudaStream_t st) {
    t3_check_cols(cols, "t3_to_t2_device");
    const int n_chunks = (int)(cols / 32), n_win = (n_chunks + 159) / 160;
    const unsigned blocks = (unsigned)(rows * n_win);
    k_t3_to_t2<<<blocks, 320, 0, st>>>(W3, W2, rows, cols, t3_row_bytes_hd((uint64_t)cols));
    CUDA_CHECK(cudaGetLastError());
}

// ---- Bonsai 2 activation rotation (kernels.cuh for the contract) ----
// One 256-thread block per 1024-element chunk; the butterfly pairs (j, j+h)
// with the low element taking a+b and the high one a-b, each element written
// by exactly one thread per stage -- the same operand order as the reference
// runtime's shared-memory FWHT, so the two are bit-equal on the same input.
// Scale 1/32 is exact in fp32, so applying it last costs nothing in bits.
template <bool INV>
__device__ __forceinline__ void hadamard1024_chunk(float* __restrict__ xb,
                                                   const float* __restrict__ sb, float* s) {
    const int j0 = threadIdx.x;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int j = j0 + 256 * i;
        s[j] = INV ? xb[j] : xb[j] * sb[j];
    }
    __syncthreads();
    for (int h = 1; h < 1024; h <<= 1) {
#pragma unroll
        for (int k = 0; k < 2; k++) {
            const int idx = j0 + 256 * k;                   // 512 pairs per stage
            const int j = ((idx / h) * 2 * h) + (idx % h);
            const float a = s[j], b = s[j + h];
            s[j] = a + b;
            s[j + h] = a - b;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int j = j0 + 256 * i;
        xb[j] = INV ? (s[j] * 0.03125f) * sb[j] : s[j] * 0.03125f;
    }
}
template <bool INV>
__global__ void __launch_bounds__(256) k_hadamard1024_rows(float* __restrict__ x,
                                                           const float* __restrict__ signs,
                                                           long row_stride) {
    __shared__ float s[1024];
    hadamard1024_chunk<INV>(x + (long)blockIdx.y * row_stride + (long)blockIdx.x * 1024,
                            signs + (long)blockIdx.x * 1024, s);
}
template <bool INV>
__global__ void __launch_bounds__(256) k_hadamard1024_lanes(P3 x, const float* __restrict__ signs) {
    __shared__ float s[1024];
    hadamard1024_chunk<INV>(x.p[blockIdx.y] + (long)blockIdx.x * 1024,
                            signs + (long)blockIdx.x * 1024, s);
}
static void hadamard_check(int width, int n) {
    if (width <= 0 || width % 1024 != 0 || n <= 0 || n > 65535) {
        fprintf(stderr, "hadamard1024: bad geometry width=%d n=%d\n", width, n);
        exit(1);
    }
}
void hadamard1024_rows(float* x, const float* signs, int width, int rows, long row_stride,
                       bool inv, cudaStream_t st) {
    hadamard_check(width, rows);
    dim3 g(width / 1024, rows);
    if (inv) k_hadamard1024_rows<true><<<g, 256, 0, st>>>(x, signs, row_stride);
    else     k_hadamard1024_rows<false><<<g, 256, 0, st>>>(x, signs, row_stride);
    CUDA_CHECK(cudaGetLastError());
}
void hadamard1024(float* x, const float* signs, int width, bool inv, cudaStream_t st) {
    hadamard1024_rows(x, signs, width, 1, width, inv, st);
}
void hadamard1024_lanes(P3 x, const float* signs, int width, int nlanes, bool inv,
                        cudaStream_t st) {
    hadamard_check(width, nlanes);
    if (nlanes > 16) { fprintf(stderr, "hadamard1024_lanes: nlanes %d > 16\n", nlanes); exit(1); }
    dim3 g(width / 1024, nlanes);
    if (inv) k_hadamard1024_lanes<true><<<g, 256, 0, st>>>(x, signs);
    else     k_hadamard1024_lanes<false><<<g, 256, 0, st>>>(x, signs);
    CUDA_CHECK(cudaGetLastError());
}
// ---- fused rotate + quantize (kernels.cuh for the contract) ----
__device__ __forceinline__ int gdn_perm_src(int i, int hd, int nk, int rep) {
    const int h = i % hd, r = (i / hd) % rep, k = i / (hd * rep); // i = k*rep*hd + r*hd + h
    return r * nk * hd + k * hd + h;
}
// k_quantize_x's per-group body on a value v held by this lane, group b.
__device__ __forceinline__ void quant_group32(float v, int b, int lane, int8_t* __restrict__ nat,
                                              uint2* __restrict__ eo, float* __restrict__ scale,
                                              int* __restrict__ isum) {
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    float s = amax / 127.f;
    float inv = s > 0.f ? 1.f / s : 0.f;
    int q = __float2int_rn(v * inv);
    q = max(-127, min(127, q));
    nat[b * 32 + lane] = (int8_t)q;
    int bsum = q;
    for (int off = 16; off > 0; off >>= 1) bsum += __shfl_xor_sync(0xffffffff, bsum, off);
    if (lane == 0) { scale[b] = s; isum[b] = bsum; }
    int base = (lane & 3) * 8;
    uint32_t e = 0, o = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        int qe = __shfl_sync(0xffffffff, q, base + 2 * k);
        int qo = __shfl_sync(0xffffffff, q, base + 2 * k + 1);
        e |= (uint32_t)(uint8_t)(int8_t)qe << (8 * k);
        o |= (uint32_t)(uint8_t)(int8_t)qo << (8 * k);
    }
    if (lane < 4) eo[b * 4 + lane] = make_uint2(e, o);
}
// 256 threads: sign+load one 1024-chunk into s, butterfly, then warp w
// quantizes groups w*4..w*4+3 of the chunk (v = s * 1/32 exactly as the
// standalone rotation stores it).
template <bool PERM>
__device__ __forceinline__ void rotq_chunk(const float* __restrict__ x,
                                           const float* __restrict__ signs, int chunk, float* s,
                                           int8_t* __restrict__ nat, uint2* __restrict__ eo,
                                           float* __restrict__ scale, int* __restrict__ isum,
                                           int hd, int nk, int rep) {
    const int j0 = threadIdx.x, base = chunk * 1024;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int j = j0 + 256 * i, idx = base + j;
        const int src = PERM ? gdn_perm_src(idx, hd, nk, rep) : idx;
        s[j] = x[src] * signs[idx];
    }
    __syncthreads();
    for (int h = 1; h < 1024; h <<= 1) {
#pragma unroll
        for (int k = 0; k < 2; k++) {
            const int idx = j0 + 256 * k;
            const int j = ((idx / h) * 2 * h) + (idx % h);
            const float a = s[j], b = s[j + h];
            s[j] = a + b;
            s[j + h] = a - b;
        }
        __syncthreads();
    }
    const int lane = j0 & 31, warp = j0 >> 5;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int g = warp * 4 + k;
        quant_group32(s[g * 32 + lane] * 0.03125f, chunk * 32 + g, lane, nat, eo, scale, isum);
    }
}
template <bool PERM>
__global__ void __launch_bounds__(256) k_rotq(const float* __restrict__ x,
                                              const float* __restrict__ signs,
                                              int8_t* __restrict__ nat, uint2* __restrict__ eo,
                                              float* __restrict__ scale, int* __restrict__ isum,
                                              int hd, int nk, int rep) {
    __shared__ float s[1024];
    rotq_chunk<PERM>(x, signs, blockIdx.x, s, nat, eo, scale, isum, hd, nk, rep);
}
template <bool PERM>
__global__ void __launch_bounds__(256) k_rotq3(__grid_constant__ const CP3 xp,
                                               const float* __restrict__ signs,
                                               __grid_constant__ const XQ3 xq, int hd, int nk,
                                               int rep) {
    __shared__ float s[1024];
    const int t = blockIdx.y;
    rotq_chunk<PERM>(xp.p[t], signs, blockIdx.x, s, xq.q[t].nat, xq.q[t].eo, xq.q[t].scale,
                     xq.q[t].isum, hd, nk, rep);
}
void rotq(const float* x, const float* signs, int width, const XQuant& xq, cudaStream_t st,
          bool perm, int hd, int nk, int rep) {
    if (width % 1024) { fprintf(stderr, "rotq: width %d not a multiple of 1024\n", width); exit(1); }
    if (perm) k_rotq<true><<<width / 1024, 256, 0, st>>>(x, signs, xq.nat, xq.eo, xq.scale, xq.isum, hd, nk, rep);
    else      k_rotq<false><<<width / 1024, 256, 0, st>>>(x, signs, xq.nat, xq.eo, xq.scale, xq.isum, 0, 0, 0);
    CUDA_CHECK(cudaGetLastError());
}
void rotq3(CP3 x, const float* signs, int width, const XQ3& xq, int ntok, cudaStream_t st,
           bool perm, int hd, int nk, int rep) {
    if (width % 1024) { fprintf(stderr, "rotq3: width %d not a multiple of 1024\n", width); exit(1); }
    dim3 g(width / 1024, ntok);
    if (perm) k_rotq3<true><<<g, 256, 0, st>>>(x, signs, xq, hd, nk, rep);
    else      k_rotq3<false><<<g, 256, 0, st>>>(x, signs, xq, 0, 0, 0);
    CUDA_CHECK(cudaGetLastError());
}
// k_rmsnorm3q's norm (1024 threads, bitwise rmsnorm3) + per-chunk rotate +
// quantize. The 1024-thread butterfly runs the same 512 pair ops per stage
// as the 256-thread chunk function (threads >= 512 idle), so the rotated
// values are bitwise those of hadamard1024.
__global__ void k_rmsnorm3_rotq(__grid_constant__ const CP3 xp, const float* __restrict__ w,
                                __grid_constant__ const P3 yp, const float* __restrict__ signs,
                                __grid_constant__ const XQ3 xq, int n, float eps) {
    const int t = blockIdx.x;
    const float* x = xp.p[t];
    float* y = yp.p[t];
    __shared__ float sh[32];
    __shared__ float s[1024];
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += x[i] * x[i];
    acc = warp_reduce(acc);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
        v = warp_reduce(v);
        if (threadIdx.x == 0) sh[0] = v;
    }
    __syncthreads();
    float inv = rsqrtf(sh[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
    __syncthreads(); // y complete block-wide before the chunks read it back
    int8_t* nat = xq.q[t].nat;
    uint2* eo = xq.q[t].eo;
    float* scale = xq.q[t].scale;
    int* isum = xq.q[t].isum;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int c = 0; c < n / 1024; c++) {
        const int base = c * 1024;
        s[threadIdx.x] = y[base + threadIdx.x] * signs[base + threadIdx.x];
        __syncthreads();
        for (int h = 1; h < 1024; h <<= 1) {
            if (threadIdx.x < 512) {
                const int idx = threadIdx.x;
                const int j = ((idx / h) * 2 * h) + (idx % h);
                const float a = s[j], b = s[j + h];
                s[j] = a + b;
                s[j + h] = a - b;
            }
            __syncthreads();
        }
        quant_group32(s[warp * 32 + lane] * 0.03125f, c * 32 + warp, lane, nat, eo, scale, isum);
        __syncthreads(); // s is reloaded by the next chunk
    }
}
void rmsnorm3_rotq(CP3 x, const float* w, P3 y, const float* signs, const XQ3& xq, int n,
                   float eps, cudaStream_t st, int ntok) {
    if (n % 1024) { fprintf(stderr, "rmsnorm3_rotq: n %d not a multiple of 1024\n", n); exit(1); }
    k_rmsnorm3_rotq<<<ntok, 1024, 0, st>>>(x, w, y, signs, xq, n, eps);
    CUDA_CHECK(cudaGetLastError());
}

// out[k*rep*hd + r*hd + h] = in[r*nk*hd + k*hd + h]; one block per row/lane.
__global__ void k_gdn_v_perm_rows(const float* __restrict__ in, float* __restrict__ out, int hd,
                                  int nk, int rep, long stride) {
    const int n = hd * nk * rep;
    const float* ir = in + (long)blockIdx.x * stride;
    float* orow = out + (long)blockIdx.x * stride;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int h = i % hd, r = (i / hd) % rep, k = i / (hd * rep);  // i = k*rep*hd + r*hd + h
        orow[i] = ir[r * nk * hd + k * hd + h];
    }
}
__global__ void k_gdn_v_perm_lanes(CP3 in, P3 out, int hd, int nk, int rep) {
    const int n = hd * nk * rep;
    const float* ir = in.p[blockIdx.x];
    float* orow = out.p[blockIdx.x];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int h = i % hd, r = (i / hd) % rep, k = i / (hd * rep);
        orow[i] = ir[r * nk * hd + k * hd + h];
    }
}
void gdn_v_tiled_to_grouped_rows(const float* in, float* out, int hd, int nk, int rep, int rows,
                                 long stride, cudaStream_t st) {
    if (in == out) { fprintf(stderr, "gdn_v_tiled_to_grouped: in-place not supported\n"); exit(1); }
    k_gdn_v_perm_rows<<<rows, 256, 0, st>>>(in, out, hd, nk, rep, stride);
    CUDA_CHECK(cudaGetLastError());
}
void gdn_v_tiled_to_grouped(const float* in, float* out, int hd, int nk, int rep, cudaStream_t st) {
    gdn_v_tiled_to_grouped_rows(in, out, hd, nk, rep, 1, (long)hd * nk * rep, st);
}
void gdn_v_tiled_to_grouped_lanes(CP3 in, P3 out, int hd, int nk, int rep, int nlanes,
                                  cudaStream_t st) {
    k_gdn_v_perm_lanes<<<nlanes, 256, 0, st>>>(in, out, hd, nk, rep);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace q27k
