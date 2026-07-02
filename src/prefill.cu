#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "blocks.cuh"
#include "prefill.cuh"

namespace q27k {

#define CUDA_CHECK(x)                                                                     \
    do {                                                                                  \
        cudaError_t err__ = (x);                                                          \
        if (err__ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),        \
                    __FILE__, __LINE__);                                                  \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

static __device__ __forceinline__ float warp_reduce_f(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// ---------------- batched GEMM ----------------
// Warp per row; token tile of TB accumulators so each 16-byte weight chunk is
// dp4a'd against TB tokens' activations (weight DRAM traffic /TB; activation
// tile stays L2-resident).

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
                c0 + cc < n_chunks
                    ? __ldg(eo + (size_t)(t0 + tt) * ept + (size_t)(c0 + cc) * 4 + u)
                    : make_uint2(0, 0);
        }
        for (int idx = threadIdx.x; idx < CS * TB; idx += blockDim.x) {
            int tt = idx % TB, cc = idx / TB;
            bool ok = c0 + cc < n_chunks;
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
                c0 + cc < n_chunks
                    ? __ldg((const uint4*)(nat + (size_t)(t0 + tt) * cols) + 2 * (c0 + cc) + u)
                    : make_uint4(0, 0, 0, 0);
        }
        for (int idx = threadIdx.x; idx < CS * TB; idx += blockDim.x) {
            int tt = idx % TB, cc = idx / TB;
            s_xs[cc * XSP + tt] =
                c0 + cc < n_chunks ? __ldg(xs + (size_t)(t0 + tt) * n_chunks + c0 + cc) : 0.f;
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

// Q4IN: weights arrive nibble-packed (Q4_G64, scale per 64) and are unpacked
// to s8 in staging; otherwise s8 rows (Q8_G128, scale per 128).
template <bool Q4IN>
__global__ void k_gemm_mma_T(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                             const int8_t* __restrict__ nat, const float* __restrict__ xs,
                             float* __restrict__ y, int64_t rows, int64_t cols, int T) {
    constexpr int MR = 64, NT = 32, KS = 128;   // block tile: rows, tokens, staged K
    constexpr int LDW = KS + 16, LDX = KS + 16; // padded smem strides (bytes)
    extern __shared__ unsigned char smem_raw[];
    int8_t* s_w = (int8_t*)smem_raw;                  // [MR][LDW]
    int8_t* s_x = (int8_t*)(s_w + MR * LDW);          // [NT][LDX]
    float* s_ws = (float*)(s_x + NT * LDX);           // [MR][KS/64 or KS/128]
    float* s_xs = (float*)(s_ws + MR * (Q4IN ? 2 : 1)); // [NT][4]

    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int wm = warp % 4, wn = warp / 4;      // warp tile: rows wm*16, tokens wn*16
    const int gid = lane >> 2, tg = lane & 3;    // mma fragment coords
    const int64_t r0 = (int64_t)blockIdx.x * MR; // block row base
    const int t0 = blockIdx.y * NT;              // block token base
    const int n_stages = (int)(cols / KS);

    float acc[2][4];
#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
        for (int e = 0; e < 4; e++) acc[s][e] = 0.f;

    for (int st = 0; st < n_stages; st++) {
        const int64_t k0 = (int64_t)st * KS;
        __syncthreads();
        // stage weights: MR rows x KS elements (unpack nibbles if Q4)
        if (Q4IN) {
            // KS/2 = 64 packed bytes per row; thread handles 4 packed -> 8 s8
            for (int idx = threadIdx.x; idx < MR * (KS / 2) / 4; idx += blockDim.x) {
                int rr = idx / 16, pb4 = idx % 16; // 16 u32 of packed bytes per row
                uint32_t p = r0 + rr < rows
                                 ? __ldg((const uint32_t*)(W + (r0 + rr) * (cols / 2) + k0 / 2) +
                                         pb4)
                                 : 0x88888888u; // unpacks to 0 after -8
                int8_t* dst = s_w + rr * LDW + pb4 * 8;
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    uint32_t byte = (p >> (8 * b)) & 0xFF;
                    dst[2 * b] = (int8_t)((int)(byte & 0xF) - 8);
                    dst[2 * b + 1] = (int8_t)((int)(byte >> 4) - 8);
                }
            }
        } else {
            for (int idx = threadIdx.x; idx < MR * KS / 4; idx += blockDim.x) {
                int rr = idx / (KS / 4), u = idx % (KS / 4);
                *(uint32_t*)(s_w + rr * LDW + u * 4) =
                    r0 + rr < rows ? __ldg((const uint32_t*)(W + (r0 + rr) * cols + k0) + u) : 0u;
            }
        }
        // stage activations: NT tokens x KS bytes (zero-fill past T)
        for (int idx = threadIdx.x; idx < NT * KS / 4; idx += blockDim.x) {
            int tt = idx / (KS / 4), u = idx % (KS / 4);
            *(uint32_t*)(s_x + tt * LDX + u * 4) =
                t0 + tt < T ? __ldg((const uint32_t*)(nat + (size_t)(t0 + tt) * cols + k0) + u)
                            : 0u;
        }
        // stage scales: weight scale per group, x scale per 32-chunk
        if (Q4IN) {
            for (int idx = threadIdx.x; idx < MR * 2; idx += blockDim.x) {
                int rr = idx / 2, g = idx % 2;
                s_ws[rr * 2 + g] =
                    r0 + rr < rows
                        ? __half2float(__ldg(S + (r0 + rr) * (cols / 64) + k0 / 64 + g))
                        : 0.f;
            }
        } else {
            for (int idx = threadIdx.x; idx < MR; idx += blockDim.x)
                s_ws[idx] = r0 + idx < rows
                                ? __half2float(__ldg(S + (r0 + idx) * (cols / 128) + k0 / 128))
                                : 0.f;
        }
        for (int idx = threadIdx.x; idx < NT * 4; idx += blockDim.x) {
            int tt = idx / 4, cc = idx % 4;
            s_xs[tt * 4 + cc] =
                t0 + tt < T
                    ? __ldg(xs + (size_t)(t0 + tt) * (cols / 32) + k0 / 32 + cc)
                    : 0.f;
        }
        __syncthreads();

        // 4 chunks of 32 per stage; per chunk, 2 mma (token subtiles of 8)
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
            for (int s = 0; s < 2; s++) {
                const int tb = wn * 16 + s * 8; // token subtile base
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
    }

    const int64_t row0 = r0 + wm * 16 + gid;
#pragma unroll
    for (int s = 0; s < 2; s++) {
        const int tok0 = t0 + wn * 16 + s * 8 + tg * 2;
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

template <bool Q4IN>
static void launch_gemm_mma(const uint8_t* W, const __half* S, const XQuant& xq, float* y,
                            int64_t rows, int64_t cols, int T, cudaStream_t st) {
    constexpr int MR = 64, NT = 32, KS = 128, LDW = KS + 16, LDX = KS + 16;
    const size_t SM = (size_t)MR * LDW + (size_t)NT * LDX + (MR * (Q4IN ? 2 : 1) + NT * 4) * 4;
    if (cols % KS) {
        fprintf(stderr, "gemm_mma: cols %ld not a multiple of %d\n", (long)cols, KS);
        exit(1);
    }
    static bool attr = false;
    if (!attr) {
        CUDA_CHECK(cudaFuncSetAttribute(k_gemm_mma_T<Q4IN>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
        attr = true;
    }
    dim3 grid((unsigned)((rows + MR - 1) / MR), (unsigned)((T + NT - 1) / NT));
    k_gemm_mma_T<Q4IN><<<grid, 256, SM, st>>>(W, S, xq.nat, xq.scale, y, rows, cols, T);
    CUDA_CHECK(cudaGetLastError());
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
    float inv = rsqrtf(fmaxf(sh[0], eps));
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

__global__ void k_kv_store_T(const float* __restrict__ kT, const float* __restrict__ vT,
                             __half* __restrict__ kc, __half* __restrict__ vc, int base_pos,
                             int rowlen) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rowlen) return;
    int t = blockIdx.y;
    size_t off = (size_t)(base_pos + t) * rowlen + i;
    kc[off] = __float2half_rn(kT[(size_t)t * rowlen + i]);
    vc[off] = __float2half_rn(vT[(size_t)t * rowlen + i]);
}

void kv_store_T(const float* kT, const float* vT, __half* kc, __half* vc, int base_pos,
                int rowlen, int T, cudaStream_t st) {
    dim3 grid((rowlen + 255) / 256, T);
    k_kv_store_T<<<grid, 256, 0, st>>>(kT, vT, kc, vc, base_pos, rowlen);
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

static __device__ __forceinline__ uint32_t h2u(__half2 h) {
    return *reinterpret_cast<uint32_t*>(&h);
}

__global__ void __launch_bounds__(192, 1)
k_attn_prefill_mma(const float* __restrict__ qT, int q_stride, int q_row,
                   const __half* __restrict__ kc, const __half* __restrict__ vc,
                   float* __restrict__ outT, int out_row, int base_pos, int tile_t0, int T,
                   int n_kv_heads, int head_dim, float scale) {
    constexpr int TT = 16, PP = 32, HD = 256, LDH = HD + 8; // padded smem rows (halfs)
    const int kvh = blockIdx.x;
    const int t0 = tile_t0 + blockIdx.y * TT;
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tg = lane & 3;

    extern __shared__ __half s_h[];
    __half* s_q = s_h;                    // [6][TT][LDH]
    __half* s_k = s_q + 6 * TT * LDH;     // [PP][LDH]
    __half* s_v = s_k + PP * LDH;         // [PP][LDH]

    // stage Q (fp32 -> fp16) once: warp w stages its own head's 16 tokens
    {
        const int qh = kvh * 6 + warp;
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

    const int tile_last = min(t0 + TT, T) - 1;            // last live token in tile
    const int tile_max = base_pos + tile_last + 1;        // positions the tile needs
    if (tile_last < t0) return;                           // whole tile past T

    for (int p0 = 0; p0 < tile_max; p0 += PP) {
        const int np = min(PP, tile_max - p0);
        __syncthreads();
        for (int idx = threadIdx.x; idx < PP * HD; idx += blockDim.x) {
            int pp = idx / HD, d = idx % HD;
            size_t off = ((size_t)(p0 + pp) * n_kv_heads + kvh) * head_dim + d;
            bool ok = pp < np;
            s_k[pp * LDH + d] = ok ? kc[off] : __float2half_rn(0.f);
            s_v[pp * LDH + d] = ok ? vc[off] : __float2half_rn(0.f);
        }
        __syncthreads();

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
            const int d0 = n * 8 + gid;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int kp = h * 16 + tg * 2;
                uint32_t b0 =
                    h2u(__halves2half2(s_v[kp * LDH + d0], s_v[(kp + 1) * LDH + d0]));
                uint32_t b1 =
                    h2u(__halves2half2(s_v[(kp + 8) * LDH + d0], s_v[(kp + 9) * LDH + d0]));
                mma_f16(o[n][0], o[n][1], o[n][2], o[n][3], pa[h][0], pa[h][1], pa[h][2],
                        pa[h][3], b0, b1, o[n][0], o[n][1], o[n][2], o[n][3]);
            }
        }
    }

    // normalize + write: thread's O elements are (rows gid/gid+8, cols
    // n*8 + tg*2, +1) of the 16x256 tile
    const int qh = kvh * 6 + warp;
    const float inv0 = l0 > 0.f ? 1.0f / l0 : 0.f, inv1 = l1 > 0.f ? 1.0f / l1 : 0.f;
    const int tr0 = t0 + gid, tr1 = t0 + gid + 8;
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

// FA-lite attention prefill: block per (kv head, 8-token tile). Warp w owns
// token (tile_t0 + w) and all 6 GQA q-heads of this kv head; K/V rows are
// staged in shared memory once per 32-position tile and shared by all 48
// (head, token) pairs. Online softmax (no position scratch). Note: fp
// summation order differs from the serial decode kernel; gated empirically
// on identical continuations.
__global__ void k_attn_prefill_T(const float* __restrict__ qT, int q_stride, int q_row,
                                 const __half* __restrict__ kc, const __half* __restrict__ vc,
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
            s_kv[(pp * 2) * HD + d] = __half2float(kc[off]);
            s_kv[(pp * 2 + 1) * HD + d] = __half2float(vc[off]);
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

void attn_prefill_T(const float* qT, int q_stride, int q_row, const __half* kc, const __half* vc,
                    float* outT, int out_row, float* scratch, int base_pos, int t0, int SB,
                    int max_ctx, int n_q_heads, int n_kv_heads, int head_dim, float scale,
                    cudaStream_t st) {
    (void)scratch; (void)max_ctx;
    const char* e = getenv("Q27_ATTN_PF");
    const bool use_mma =
        !(e && !strcmp(e, "lite")) && head_dim == 256 && n_q_heads == 6 * n_kv_heads;
    if (use_mma) {
        constexpr int TT = 16, PP = 32, LDH = 256 + 8;
        const size_t SM = (size_t)(6 * TT + 2 * PP) * LDH * sizeof(__half);
        static bool attr2 = false;
        if (!attr2) {
            CUDA_CHECK(cudaFuncSetAttribute(k_attn_prefill_mma,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
            attr2 = true;
        }
        dim3 grid(n_kv_heads, (SB + TT - 1) / TT);
        k_attn_prefill_mma<<<grid, 192, SM, st>>>(qT, q_stride, q_row, kc, vc, outT, out_row,
                                                  base_pos, t0, t0 + SB, n_kv_heads, head_dim,
                                                  scale);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    constexpr int TT = 8, PP = 32;
    const size_t SM = (size_t)PP * 2 * 256 * sizeof(float);
    static bool attr = false;
    if (!attr) {
        CUDA_CHECK(cudaFuncSetAttribute(k_attn_prefill_T,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
        attr = true;
    }
    int gqa = n_q_heads / n_kv_heads;
    dim3 grid(n_kv_heads, (SB + TT - 1) / TT);
    k_attn_prefill_T<<<grid, 256, SM, st>>>(qT, q_stride, q_row, kc, vc, outT, out_row,
                                            base_pos, t0, t0 + SB, n_kv_heads, gqa, head_dim,
                                            scale);
    CUDA_CHECK(cudaGetLastError());
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

void delta_scan_T(float* S_global, const float* convT, const float* gT, const float* betaT,
                  float* oT, int T, cudaStream_t st) {
    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(k_delta_scan_T,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        128 * 128 * 4));
        attr_set = true;
    }
    k_delta_scan_T<<<48, 512, 128 * 128 * 4, st>>>(S_global, convT, gT, betaT, oT, T);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace q27k
