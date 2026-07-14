// vgemm -- the flat-in-W weight path for the verify round.
// Plan: docs/plans/2026-07-13-gemm-verify.md (P1).
//
// WHY. The batched verify GEMV (gemv_q4_n/gemv_q8_n) is not bandwidth-bound at
// width, it is REGISTER-bound: over the round's real 401-weight sequence it does
// 1230 GB/s at W=5 but collapses to 444 GB/s at W=16, against a 1453 GB/s SOL.
// Measured (nsys, one width-12 suffix round): the weight GEMV is 15.7 ms of a
// 21.4 ms GPU-busy round -- 73%. An m16n8k32 s8 MMA over the SAME sequence is
// FLAT: 11.4 ms at W=5, 12.5 ms at W=16. Per-lane marginal W14->W16: GEMV
// 4.34 ms/lane, this kernel 0.155 -- a factor of 28.
//
// The point is not the speedup. The point is that a weight path FLAT in W is the
// only thing that makes a verify-width raise pay: today a 12->16 cap needs +62%
// more accepted tokens to break even (and gets +37%, hence the 07-13 W16 NO-GO);
// with this kernel it needs +2.5%.
//
// DETERMINISM IS A HARD REQUIREMENT, not a preference. q27's entire measurement
// discipline -- the canonical md5, the byte-identity gates, "n=1 per prompt is
// exact" -- rests on greedy decode being run-to-run bitwise. A grid.z atomicAdd
// epilogue is NOT (measured: 11-20K of 61K floats differ run-to-run on ffn_down
// at z=8). So there are NO atomics here. Two K-splits, both fixed-order:
//   - INTRA-CTA (KG warp-groups): partials summed through smem in warp order.
//   - CROSS-CTA (grid.z): partials to a workspace, then k_reduce_z sums i=0..z-1
//     in index order. Costs 0.4-0.7 ms/round; that is the price of determinism
//     and it is worth paying.
// MR=32 is what makes the deterministic split affordable: it doubles the row-CTA
// supply vs MR=64, so the cross-CTA z needed is halved and the partials stay in
// L2. It is also the ONLY tile that is spill-free on all four instantiations
// (Q4/Q8 x MODE 0/1) -- see the gate in tools/vgemm_test.cu.
//
// NUMERICS vs gemv_q4_n/gemv_q8_n: the int8 activations are IDENTICAL (same
// k_quantize_x3 nat/scale, group-32); int8xint8 products are bit-exact in int32
// (|acc| <= 128*127*32 fits); the Q4 "-8" nibble bias just moves from the GEMV's
// isum term to a __vsub4 at the smem unpack -- algebraically the same. ONLY the
// fp32 accumulation ORDER differs. Measured rel: Q4 1.05e-6, Q8 2.80e-6.
// => tolerance-class, run-to-run deterministic. Suffix-round output may re-roll
// an argmax near-tie; the ladder never reaches this kernel (see gemm_min).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "cuda_common.h" // W_PLUMB

namespace q27 { struct DevTensor; }

namespace q27k {

// Per-lane activation / output pointers. Same __grid_constant__ by-value idiom as
// P3/CP3/Q4Lanes: the verify lanes live in separate buffers and are never packed.
struct XLanes {
    const int8_t* nat[W_PLUMB]; // group-32 int8 activations (k_quantize_x3 already writes these)
    const float* xs[W_PLUMB];   // group-32 scales
};
struct YLanes {
    float* y[W_PLUMB];
};

// The tile. NT is fixed at W_PLUMB (a verify is at most W_PLUMB wide, so grid.x
// is always 1); MR is the row tile.
static constexpr int VG_MR = 32;
static constexpr int VG_KS = 128; // K per mma stage
static constexpr int VG_WM = VG_MR / 16;
static constexpr int VG_KG = 8 / (VG_WM * 2); // warp-groups splitting K inside the CTA
static constexpr int VG_KB = VG_KG * VG_KS;   // K staged per super-step
static_assert(VG_KG >= 1 && VG_KG * VG_WM * 2 == 8, "256 threads = 8 warps = WM x 2 x KG");

// Every decode `cols` (5120, 6144, 17408) is a multiple of KB=256. A weight that
// is not would silently drop its K tail, so the launcher aborts instead.
static constexpr int VG_KB_MULT = VG_KB;

constexpr size_t vgemm_smem_bytes(bool q4in) {
    constexpr int LDW = VG_KB + 16, LDX = VG_KB + 16, XSC = VG_KS / 32;
    const int nws = q4in ? 2 : 1;
    size_t s = (size_t)VG_MR * LDW + (size_t)W_PLUMB * LDX +
               (VG_MR * VG_KG * nws + W_PLUMB * VG_KG * XSC) * 4;
    size_t red = (VG_KG > 1) ? 8u * 32u * 4u * 4u : 0u; // the intra-CTA reduce aliases s_w
    return s > red ? s : red;
}

// Cross-CTA K-split for one weight: enough CTAs to fill the machine, never more
// than the K axis can feed. Pure function of the shape -> a captured graph bakes
// a stable z, and the same weight always gets the same z (determinism).
int vgemm_z(int64_t rows, int64_t cols);

// Workspace: max over the round's weights of z*W_PLUMB*rows floats. Walk the real
// weight list -- do NOT hardcode; a z-policy change silently overruns it.
size_t vgemm_ws_bytes(const q27::DevTensor* const* ws, int n);

// The dispatch. Returns false if the shape is not eligible (caller MUST fall back
// to the GEMV and honor the false, exactly like launch_fdmma).
bool vgemm_verify(const q27::DevTensor& w, const XLanes& X, const YLanes& Y, float* ws, int T,
                  cudaStream_t st);

// Occupancy introspection for the CI gate (tools/vgemm_test.cu gate 4). The
// budget has ZERO slack: 64 regs x 256 threads x 4 CTAs = 65536 = the whole SM
// register file. One extra live value silently costs a CTA tier and ~20%.
struct VgemmAttrs {
    int regs;
    size_t local_bytes; // stack frame -- MUST be 0 (spill)
    size_t smem;
    int cta_per_sm;
};
VgemmAttrs vgemm_attrs(bool q4in, int mode);

} // namespace q27k
