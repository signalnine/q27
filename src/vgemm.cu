// vgemm -- flat-in-W verify weight path. See vgemm.cuh for the why.
// The kernel is the design spike's, promoted: the arithmetic is NOT re-derived
// here (it was validated against gemv_q4_n/gemv_q8_n at rel 1e-6), only wrapped.
#include <algorithm>

#include "device_model.h"
#include "loader.h"
#include "vgemm.cuh"

namespace q27k {

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

// MODE 0: store straight to the per-lane outputs (z == 1 -- no workspace, no
//         reduce node, deterministic by construction; this is the vocab head).
// MODE 1: store this z-slice's partial; k_reduce_z sums them in index order.
template <int MR, bool Q4IN, int MODE>
__global__ __launch_bounds__(256, 4) void k_vgemm(const uint8_t* __restrict__ W,
                                                  const __half* __restrict__ S,
                                                  __grid_constant__ const XLanes X,
                                                  __grid_constant__ const YLanes Y,
                                                  float* __restrict__ ws, int64_t rows,
                                                  int64_t cols, int T, int stages_per_z) {
    constexpr int NT = W_PLUMB;
    constexpr int KS = VG_KS;
    constexpr int WM = MR / 16;
    constexpr int KG = 8 / (WM * 2);
    constexpr int KB = KG * KS;
    constexpr int LDW = KB + 16, LDX = KB + 16;
    constexpr int XSC = KS / 32;      // x-scales per row per stage (group-32 activations)
    constexpr int NWS = Q4IN ? 2 : 1; // w-scales per row per stage (g64 vs g128)
    static_assert(KG >= 1 && KG * WM * 2 == 8, "warp split");

    extern __shared__ unsigned char smem_raw[];
    int8_t* s_w = (int8_t*)smem_raw;
    int8_t* s_x = (int8_t*)(s_w + MR * LDW);
    float* s_ws = (float*)(s_x + NT * LDX);
    float* s_xs = (float*)(s_ws + MR * KG * NWS);

    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid & 31;
    const int wm = warp % WM;
    const int wn = (warp / WM) % 2;
    const int kg = warp / (WM * 2);
    const int gid = lane >> 2, tg = lane & 3;

    const int64_t r0 = (int64_t)blockIdx.y * MR;
    const int n_stages = (int)(cols / KS);
    const int s_begin = blockIdx.z * stages_per_z;
    const int s_end = min(n_stages, s_begin + stages_per_z);
    if (s_begin >= s_end) return;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    constexpr int WBY = Q4IN ? (MR * KB / 2) : (MR * KB);
    constexpr int WLD = WBY / 4 / 256;
    constexpr int XLD = (NT * KB / 4) / 256;
    constexpr int SLD = (MR * KG * NWS + 255) / 256;
    constexpr int XSL = (NT * KG * XSC + 255) / 256;
    constexpr int WPR = Q4IN ? (KB / 8) : (KB / 4); // u32 per row per super-step

    uint32_t rw[WLD], rx[XLD];
    float rws[SLD], rxs[XSL];

    auto load_stage = [&](int sst) {
        const int64_t k0 = (int64_t)sst * KS;
#pragma unroll
        for (int i = 0; i < WLD; i++) {
            int idx = i * 256 + tid;
            int rr = idx / WPR, pb = idx % WPR;
            bool ok = (r0 + rr < rows);
            const uint8_t* base =
                Q4IN ? (W + (r0 + rr) * (cols / 2) + k0 / 2) : (W + (r0 + rr) * cols + k0);
            // 0x88 = the q4 zero point pre-bias, so padded rows contribute 0.
            rw[i] = ok ? __ldg((const uint32_t*)base + pb) : (Q4IN ? 0x88888888u : 0u);
        }
#pragma unroll
        for (int i = 0; i < XLD; i++) {
            int idx = i * 256 + tid;
            int tt = idx / (KB / 4), u = idx % (KB / 4);
            rx[i] = tt < T ? __ldg((const uint32_t*)(X.nat[tt] + k0) + u) : 0u;
        }
#pragma unroll
        for (int i = 0; i < SLD; i++) {
            int idx = i * 256 + tid;
            int rr = idx / (KG * NWS), g = idx % (KG * NWS);
            rws[i] = (idx < MR * KG * NWS && r0 + rr < rows)
                         ? __half2float(__ldg(S + (r0 + rr) * (cols / (Q4IN ? 64 : 128)) +
                                              k0 / (Q4IN ? 64 : 128) + g))
                         : 0.f;
        }
#pragma unroll
        for (int i = 0; i < XSL; i++) {
            int idx = i * 256 + tid;
            int tt = idx / (KG * XSC), cc = idx % (KG * XSC);
            rxs[i] = (idx < NT * KG * XSC && tt < T) ? __ldg(X.xs[tt] + k0 / 32 + cc) : 0.f;
        }
    };
    auto store_stage = [&]() {
#pragma unroll
        for (int i = 0; i < WLD; i++) {
            int idx = i * 256 + tid;
            int rr = idx / WPR, pb = idx % WPR;
            const uint32_t p = rw[i];
            if constexpr (Q4IN) {
                int8_t* dst = s_w + rr * LDW + pb * 8;
                const uint32_t lo = p & 0x0F0F0F0Fu, hi = (p >> 4) & 0x0F0F0F0Fu;
                // the GEMV's isum "-8" bias, applied at unpack instead
                *(uint32_t*)dst = __vsub4(__byte_perm(lo, hi, 0x5140), 0x08080808u);
                *(uint32_t*)(dst + 4) = __vsub4(__byte_perm(lo, hi, 0x7362), 0x08080808u);
            } else {
                *(uint32_t*)(s_w + rr * LDW + pb * 4) = p;
            }
        }
#pragma unroll
        for (int i = 0; i < XLD; i++) {
            int idx = i * 256 + tid;
            int tt = idx / (KB / 4), u = idx % (KB / 4);
            *(uint32_t*)(s_x + tt * LDX + u * 4) = rx[i];
        }
#pragma unroll
        for (int i = 0; i < SLD; i++) {
            int idx = i * 256 + tid;
            if (idx < MR * KG * NWS) s_ws[idx] = rws[i];
        }
#pragma unroll
        for (int i = 0; i < XSL; i++) {
            int idx = i * 256 + tid;
            if (idx < NT * KG * XSC) s_xs[idx] = rxs[i];
        }
    };

    load_stage(s_begin);
    for (int sst = s_begin; sst < s_end; sst += KG) {
        __syncthreads();
        store_stage();
        if (sst + KG < s_end) load_stage(sst + KG);
        __syncthreads();
        if (sst + kg < s_end) {
            const int kbase = kg * KS; // this warp-group's slice of the staged K
#pragma unroll
            for (int cc = 0; cc < 4; cc++) {
                const int kb = kbase + cc * 32;
                const int8_t* wrow0 = s_w + (wm * 16 + gid) * LDW + kb;
                uint32_t a0 = *(const uint32_t*)(wrow0 + tg * 4);
                uint32_t a1 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4);
                uint32_t a2 = *(const uint32_t*)(wrow0 + tg * 4 + 16);
                uint32_t a3 = *(const uint32_t*)(wrow0 + 8 * LDW + tg * 4 + 16);
                float wsc0, wsc1;
                if constexpr (Q4IN) {
                    wsc0 = s_ws[(wm * 16 + gid) * (KG * 2) + kb / 64];
                    wsc1 = s_ws[(wm * 16 + gid + 8) * (KG * 2) + kb / 64];
                } else {
                    wsc0 = s_ws[(wm * 16 + gid) * KG + kb / 128];
                    wsc1 = s_ws[(wm * 16 + gid + 8) * KG + kb / 128];
                }
                const int tb = wn * 8;
                const int8_t* xcol = s_x + (tb + gid) * LDX + kb;
                uint32_t b0 = *(const uint32_t*)(xcol + tg * 4);
                uint32_t b1 = *(const uint32_t*)(xcol + tg * 4 + 16);
                int d0, d1, d2, d3;
                mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                const float xs0 = s_xs[(tb + tg * 2) * (KG * XSC) + kb / 32];
                const float xs1 = s_xs[(tb + tg * 2 + 1) * (KG * XSC) + kb / 32];
                acc[0] += wsc0 * xs0 * (float)d0;
                acc[1] += wsc0 * xs1 * (float)d1;
                acc[2] += wsc1 * xs0 * (float)d2;
                acc[3] += wsc1 * xs1 * (float)d3;
            }
        }
    }

    // intra-CTA K-split reduce, FIXED WARP ORDER (no atomics, no shuffle races).
    // Aliases s_w -- dead by here, and 4KB <= MR*LDW. Nothing syncs after the
    // early return, so the divergent exit is legal.
    if constexpr (KG > 1) {
        float* red = (float*)smem_raw;
        __syncthreads();
        for (int e = 0; e < 4; e++) red[(warp * 32 + lane) * 4 + e] = acc[e];
        __syncthreads();
        if (kg != 0) return;
        const int b = wm + WM * wn;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            float s = 0.f;
            for (int g = 0; g < KG; g++) s += red[((b + g * WM * 2) * 32 + lane) * 4 + e];
            acc[e] = s;
        }
    }

    const int64_t row0 = r0 + wm * 16 + gid;
    const int tok0 = wn * 8 + tg * 2;
#pragma unroll
    for (int e = 0; e < 4; e++) {
        int64_t row = row0 + (e >= 2 ? 8 : 0);
        int tok = tok0 + (e & 1);
        if (row < rows && tok < T) {
            if constexpr (MODE == 1)
                ws[((size_t)blockIdx.z * T + tok) * rows + row] = acc[e];
            else
                Y.y[tok][row] = acc[e];
        }
    }
}

// Deterministic fixed-order reduce of the grid.z partials. i ascending, always.
__global__ void k_reduce_z(const float* __restrict__ ws, __grid_constant__ const YLanes Y,
                           int64_t rows, int T, int z) {
    int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (r >= rows || t >= T) return;
    float s = 0.f;
    for (int i = 0; i < z; i++) s += ws[((size_t)i * T + t) * rows + r];
    Y.y[t][r] = s;
}

// ---- host ----

// One wave is 170 SMs x 4 CTAs = 680 co-resident CTAs; aim at ~2 waves so the
// tail is cheap. Bounded by 8 (the deterministic partials must stay small) and by
// the K axis (never fewer than 4 stages per slice, or the prologue dominates).
static constexpr int VG_CTA_TARGET = 1400;
static constexpr int VG_ZMAX = 8;

int vgemm_z(int64_t rows, int64_t cols) {
    const int row_ctas = (int)((rows + VG_MR - 1) / VG_MR);
    const int n_stages = (int)(cols / VG_KS);
    int z = (VG_CTA_TARGET + row_ctas - 1) / row_ctas;
    if (z < 1) z = 1;
    if (z > VG_ZMAX) z = VG_ZMAX;
    const int zk = n_stages / 4;
    if (z > zk) z = zk > 0 ? zk : 1;
    // Mirror the launcher's KB-align + trim so vgemm_ws_bytes() sizes the SAME z
    // the launch will actually use (an under-sized workspace is a heap overrun).
    int spz = (n_stages + z - 1) / z;
    spz = (spz + VG_KG - 1) / VG_KG * VG_KG;
    z = (n_stages + spz - 1) / spz;
    return z < 1 ? 1 : z;
}

size_t vgemm_ws_bytes(const q27::DevTensor* const* wl, int n) {
    size_t mx = 0;
    for (int i = 0; i < n; i++) {
        const q27::DevTensor& w = *wl[i];
        int z = vgemm_z(w.rows, w.cols);
        if (z <= 1) continue; // MODE 0 needs no workspace
        size_t b = (size_t)z * W_PLUMB * (size_t)w.rows * 4;
        if (b > mx) mx = b;
    }
    return mx;
}

template <bool Q4IN, int MODE>
static void set_attr_once() {
    static bool done = false;
    if (done) return;
    // 13.75 KB is under the 48 KB static default, so this is a no-op today. Keep
    // it (and the assert) so a future tile bump cannot silently reintroduce the
    // lazy-cudaFuncSetAttribute-during-graph-capture hazard.
    static_assert(vgemm_smem_bytes(Q4IN) < 48 * 1024,
                  "smem over the 48KB default -- setattr must move out of capture");
    CUDA_CHECK(cudaFuncSetAttribute(k_vgemm<VG_MR, Q4IN, MODE>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)vgemm_smem_bytes(Q4IN)));
    done = true;
}

template <bool Q4IN>
static void launch(const q27::DevTensor& w, const XLanes& X, const YLanes& Y, float* ws, int T,
                   cudaStream_t st) {
    const int n_stages = (int)(w.cols / VG_KS);
    int z = vgemm_z(w.rows, w.cols);
    int spz = (n_stages + z - 1) / z;
    spz = (spz + VG_KG - 1) / VG_KG * VG_KG; // KB-align: a slice can never straddle n_stages
    z = (n_stages + spz - 1) / spz;          // trim empty slices (k_reduce_z would sum garbage)
    dim3 grid(1, (unsigned)((w.rows + VG_MR - 1) / VG_MR), (unsigned)z);
    const size_t sm = vgemm_smem_bytes(Q4IN);
    if (z == 1) {
        set_attr_once<Q4IN, 0>();
        k_vgemm<VG_MR, Q4IN, 0><<<grid, 256, sm, st>>>((const uint8_t*)w.data,
                                                       (const __half*)w.scales, X, Y, nullptr,
                                                       w.rows, w.cols, T, spz);
    } else {
        set_attr_once<Q4IN, 1>();
        k_vgemm<VG_MR, Q4IN, 1><<<grid, 256, sm, st>>>((const uint8_t*)w.data,
                                                       (const __half*)w.scales, X, Y, ws, w.rows,
                                                       w.cols, T, spz);
        dim3 g2((unsigned)((w.rows + 255) / 256), (unsigned)T);
        k_reduce_z<<<g2, 256, 0, st>>>(ws, Y, w.rows, T, z);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <bool Q4IN, int MODE>
static VgemmAttrs attrs_of() {
    cudaFuncAttributes a{};
    CUDA_CHECK(cudaFuncGetAttributes(&a, k_vgemm<VG_MR, Q4IN, MODE>));
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k_vgemm<VG_MR, Q4IN, MODE>,
                                                             256, vgemm_smem_bytes(Q4IN)));
    return VgemmAttrs{a.numRegs, (size_t)a.localSizeBytes, vgemm_smem_bytes(Q4IN), blocks};
}

VgemmAttrs vgemm_attrs(bool q4in, int mode) {
    if (q4in) return mode ? attrs_of<true, 1>() : attrs_of<true, 0>();
    return mode ? attrs_of<false, 1>() : attrs_of<false, 0>();
}

bool vgemm_verify(const q27::DevTensor& w, const XLanes& X, const YLanes& Y, float* ws, int T,
                  cudaStream_t st) {
    if (T < 2 || T > W_PLUMB) return false;
    // A cols that is not a multiple of KB would silently drop its K tail. Every
    // decode shape (5120/6144/17408) complies; abort rather than corrupt.
    if (w.cols % VG_KB_MULT != 0) {
        fprintf(stderr, "vgemm: cols %ld not a multiple of %d\n", (long)w.cols, VG_KB_MULT);
        abort();
    }
    if (w.dtype == q27::DType::Q4_G64) launch<true>(w, X, Y, ws, T, st);
    else if (w.dtype == q27::DType::Q8_G128) launch<false>(w, X, Y, ws, T, st);
    else return false;
    return true;
}

} // namespace q27k
