#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x)                                                                  \
    do {                                                                               \
        cudaError_t err_ = (x);                                                        \
        if (err_ != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error: %s\n  at %s:%d: %s\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__, #x);                                           \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

// LANE PLUMBING WIDTH -- the FIXED number of lane slots the round kernels
// address. Q27_W_MAX (engine.cuh) only caps how many go LIVE and sizes the
// per-width memory (GDN role sets, captured-graph zoo); W_PLUMB sizes the
// structs, the outcome layout, and every "list every lane" array. It is 16
// because the lane-pointer structs (P3/CP3/IP3/XQ3/WIP3) carry p[16] and the
// fdmma verify kernel asserts 6*W <= 96 rows -- 16 is the hard ceiling of the
// current kernel family, so there is no reason to plumb anything narrower.
//
// It lives HERE, not in engine.cuh, because spec3.cu (k_prep_round /
// k_finish_round -- the kernels that walk every lane slot) does not include
// engine.cuh. Before W16 those kernels carried hardcoded 11/12 literals that
// no compiler could tie back to the plumbing width; a widening had to find
// them by grep. Now they derive from this constant.
#ifndef Q27_W_PLUMB
#define Q27_W_PLUMB 16
#endif
static constexpr int W_PLUMB = Q27_W_PLUMB;
static_assert(W_PLUMB <= 16, "lane-pointer structs are p[16]; k_attn_fdmma asserts 6*W <= 96");
// Round outcome layout: {n, t1, dr1..dr(W_PLUMB-1), pending}.
static constexpr int OUTCOME_INTS = W_PLUMB + 2;

// KV-cache format kind (Q27_KV): scalar fp16 (default) / fp8 E4M3 ("fp8") /
// turbo3 3-bit blocks ("turbo3", src/turbo3.cuh) / turbo3 V with plain fp16 K
// ("turbo3v" -- the GQA=6 escape hatch if turbo3-K craters, port spec risk
// section). Values 0/1 keep the old `bool fp8` call sites meaning-compatible
// (false->KV_F16, true->KV_FP8) where the parameter widened to int.
enum KvKind : int { KV_F16 = 0, KV_FP8 = 1, KV_T3 = 2, KV_T3V = 3 };

#ifdef __CUDACC__
#include <cuda_fp8.h>
// KV-cache element conversions (P2): fp16 default, fp8 E4M3 opt-in (Q27_KV=fp8).
// E4M3 store saturates to +-448; kvstats probe (2026-07-02, 8K wikitext tokens)
// measured K amax <= 21.8, V amax <= 118.6 across all 17 attention layers, so
// scale-free E4M3 has >=3.8x headroom and per-row scales buy nothing.
__device__ __forceinline__ float kv2f(__half x) { return __half2float(x); }
__device__ __forceinline__ float kv2f(__nv_fp8_e4m3 x) { return float(x); }
__device__ __forceinline__ __half kv2h(__half x) { return x; }
__device__ __forceinline__ __half kv2h(__nv_fp8_e4m3 x) {
    return __half(__nv_cvt_fp8_to_halfraw(x.__x, __NV_E4M3));
}
__device__ __forceinline__ void kv_set(__half& d, float x) { d = __float2half_rn(x); }
__device__ __forceinline__ void kv_set(__nv_fp8_e4m3& d, float x) { d = __nv_fp8_e4m3(x); }
#endif
