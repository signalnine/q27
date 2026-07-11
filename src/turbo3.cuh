// turbo3 KV quantization -- ported from TurboQuant (Gabe's llama.cpp fork
// llama-cpp-turboquant @ c3e6dbb13, ggml-turbo-quant.c). 3-bit PolarQuant
// with a baked Walsh-Hadamard rotation over 128-element groups: K/V are
// stored WHT-rotated + quantized to 8 Lloyd-Max centroids scaled by a
// per-block L2 norm. Correctness contract: dequant(K) == WHT(K), so the
// attention dot uses WHT(q).dequant(K) == q.K (WHT orthonormal), and the
// pooled output is WHT(true output) -> one inverse-WHT un-rotates it.
//
// q27 mapping (head_dim=256 = TWO 128-groups, TURBO_D=128, no padding):
//   block_turbo3_0 = 50 bytes covering 128 dims; 2 blocks per (kv_head).
// This header is the SHARED device+host definition; the microtest
// (tools/turbo3_test.cu) validates it bit-for-bit against the CPU ref.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace q27turbo {

static constexpr int QK_TURBO3 = 128;          // elements per block == WHT group
static constexpr float TURBO_INV_SQRT_128 = 0.08838834764831845f;

// 50-byte block: fp16 norm + 2-bit indices (qs) + 1 high bit (signs).
struct block_turbo3 {
    __half  norm;                 // corrected L2 norm (grp_norm / recon_norm)
    uint8_t qs[QK_TURBO3 / 4];    // 32 B: low 2 bits, 4 per byte
    uint8_t signs[QK_TURBO3 / 8]; // 16 B: high bit, 8 per byte
};
static_assert(sizeof(block_turbo3) == sizeof(__half) + QK_TURBO3/4 + QK_TURBO3/8,
              "turbo3 block must be 50 bytes, no padding");

// 8 Lloyd-Max centroids for a unit-norm coordinate (verbatim). The _LIST
// macros keep the device __constant__ arrays and the host oracle mirrors
// literally the same token sequence -- no re-typed copies (phase 1's
// "4096 mismatches" was a hand-transcription bug in exactly such a copy).
#define Q27_TURBO_CENTROIDS_LIST { \
    -0.190207f, -0.118786f, -0.066822f, -0.021663f, \
     0.021663f,  0.066822f,  0.118786f,  0.190207f }
// WHT sign diagonals (seed 42) -- MUST byte-match the fork's s1/s2.
#define Q27_TURBO_S1_LIST { -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1 }
#define Q27_TURBO_S2_LIST { 1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1 }

#ifdef __CUDACC__
__device__ __constant__ static float TURBO_CENTROIDS_3BIT[8] = Q27_TURBO_CENTROIDS_LIST;
__device__ __constant__ static float TURBO_S1[128] = Q27_TURBO_S1_LIST;
__device__ __constant__ static float TURBO_S2[128] = Q27_TURBO_S2_LIST;
#endif
// host mirrors for CPU oracles (validated transitively by tools/turbo3_test,
// which checks this header's device copies against its own fork-verbatim set)
[[maybe_unused]] static const float TURBO_CENTROIDS_HOST[8] = Q27_TURBO_CENTROIDS_LIST;
[[maybe_unused]] static const float TURBO_S1_HOST[128] = Q27_TURBO_S1_LIST;
[[maybe_unused]] static const float TURBO_S2_HOST[128] = Q27_TURBO_S2_LIST;

// nearest-centroid index via the 7 midpoints (matches nearest_centroid_3bit).
__host__ __device__ __forceinline__ int turbo3_nearest(float v) {
    if (v < -0.154496f) return 0;
    if (v < -0.092804f) return 1;
    if (v < -0.044243f) return 2;
    if (v <  0.000000f) return 3;
    if (v <  0.044243f) return 4;
    if (v <  0.092804f) return 5;
    if (v <  0.154496f) return 6;
    return 7;
}

// One dequanted element from a block (idx = low2 | hi1<<2) * norm.
__device__ __forceinline__ float turbo3_dequant(const block_turbo3* b, int j, float norm) {
    uint8_t low2 = (b->qs[j >> 2] >> ((j & 3) * 2)) & 0x3;
    uint8_t hi1  = (b->signs[j >> 3] >> (j & 7)) & 0x1;
    return TURBO_CENTROIDS_3BIT[low2 | (hi1 << 2)] * norm;
}

#ifdef __CUDACC__
// In-smem 128-point Hadamard butterfly, one thread per element (j = tid,
// blockDim.x == 128). Operand order matches the CPU ref exactly (lo+hi /
// lo-hi), so device results are bitwise CPU-equal; fixed order => run-to-run
// deterministic. Caller applies the s1/s2/inv_sqrt diagonals around it.
__device__ __forceinline__ void turbo3_butterfly128(float* xs, int j) {
#pragma unroll
    for (int h = 1; h < 128; h <<= 1) {
        __syncthreads();
        float a = xs[j], b = xs[j ^ h];
        __syncthreads();
        xs[j] = (j & h) ? (b - a) : (a + b);
    }
    __syncthreads();
}

// Cooperative 128-group quantize: one CUDA block of 128 threads = one group,
// thread j owns element x = src[j]. Fixed-order L2 reduce -> s1/butterfly/s2
// forward WHT -> nearest-centroid pack (shfl + ballot) -> corrected fp16
// norm. xs/red are caller-provided smem[128]. Shared by the decode store
// (spec3.cu k_kv_store_t3) and the prefill store (prefill.cu) so the two
// writers cannot drift.
__device__ __forceinline__ void turbo3_quant_group(float x, block_turbo3* dst, int j,
                                                   float* xs, float* red) {
    xs[j] = x;
    red[j] = x * x;
    __syncthreads();
#pragma unroll
    for (int s = 64; s > 0; s >>= 1) {
        if (j < s) red[j] += red[j + s];
        __syncthreads();
    }
    float gn = sqrtf(red[0]);
    float inv = gn > 1e-10f ? 1.f / gn : 0.f;
    xs[j] = xs[j] * inv * TURBO_S1[j];
    turbo3_butterfly128(xs, j);
    float w = xs[j] * (TURBO_INV_SQRT_128 * TURBO_S2[j]);
    int idx = turbo3_nearest(w);
    red[j] = TURBO_CENTROIDS_3BIT[idx] * TURBO_CENTROIDS_3BIT[idx];
    __syncthreads();
#pragma unroll
    for (int s = 64; s > 0; s >>= 1) {
        if (j < s) red[j] += red[j + s];
        __syncthreads();
    }
    float rn = sqrtf(red[0]);
    float corr = rn > 1e-10f ? gn / rn : gn;
    int low2 = idx & 3;
    unsigned bal = __ballot_sync(0xffffffffu, idx & 4);
    int l1 = __shfl_down_sync(0xffffffffu, low2, 1);
    int l2 = __shfl_down_sync(0xffffffffu, low2, 2);
    int l3 = __shfl_down_sync(0xffffffffu, low2, 3);
    if ((j & 3) == 0)
        dst->qs[j >> 2] = (uint8_t)(low2 | (l1 << 2) | (l2 << 4) | (l3 << 6));
    if ((j & 7) == 0) dst->signs[j >> 3] = (uint8_t)((bal >> (j & 31)) & 0xFF);
    if (j == 0) dst->norm = __float2half(corr);
}

// One dequanted element d (0..255) from a 2-block row (head_dim 256).
__device__ __forceinline__ float turbo3_deq_elem(const block_turbo3* row2, int d) {
    const block_turbo3* b = row2 + (d >> 7);
    return turbo3_dequant(b, d & 127, __half2float(b->norm));
}

// Stage 8 consecutive dims (d8 8-aligned, 0..248) from a 2-block row into
// 4 half2 (the mma prefill smem-tile layout): 2 qs bytes + 1 signs byte +
// 1 norm per call -- the amortized-byte read pattern the fork's V loop uses.
__device__ __forceinline__ void turbo3_stage8_h2(const block_turbo3* row2, int d8,
                                                 __half2* dst) {
    const block_turbo3* b = row2 + (d8 >> 7);
    int j0 = d8 & 127;
    float norm = __half2float(b->norm);
    uint8_t q0 = b->qs[j0 >> 2], q1 = b->qs[(j0 >> 2) + 1];
    uint8_t s8 = b->signs[j0 >> 3];
    float v[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int i0 = ((q0 >> (2 * i)) & 3) | (((s8 >> i) & 1) << 2);
        int i1 = ((q1 >> (2 * i)) & 3) | (((s8 >> (4 + i)) & 1) << 2);
        v[i] = TURBO_CENTROIDS_3BIT[i0] * norm;
        v[4 + i] = TURBO_CENTROIDS_3BIT[i1] * norm;
    }
#pragma unroll
    for (int j = 0; j < 4; j++)
        dst[j] = __halves2half2(__float2half_rn(v[2 * j]), __float2half_rn(v[2 * j + 1]));
}
#endif

} // namespace q27turbo
