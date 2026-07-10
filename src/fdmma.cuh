// k_attn_fdmma<W>: fp8-MMA split-KV W-query verify attention.
// Design + verdict chain: docs/plans/2026-07-10-fdmma-verify-attn.md.
// Spike home: lives in tools/ (fdmma_test.cu microtests, attn_fdw_bench.cu
// perf leg); promoted to src/spec3.cu at engine integration.
//
// One block per (split sp, kv-head): streams the union of the live lanes'
// KV windows ONCE through smem, scoring all W lanes x 6 gqa heads with
// m16n8k32 e4m3 MMA. Dense (head, lane) row packing r = j*W + t: M = 6W
// live rows of a 96-row padded Q matrix; warp w owns rows 16w..16w+15 and
// the full 16x256 O in registers (o[32][4], the pv8 accumulator); live
// warps = ceil(6W/16), dead warps stage + barrier only. Partials in the
// FD_ST=258 {m, l, acc[256]} layout -- k_attn_fd_combine consumes them
// unchanged (per-lane used-split derivation preserved by construction:
// write iff sp*chunk_t < seq_t).
//
// Donor idioms (verbatim from src/prefill.cu k_attn_prefill_mma_pv8):
// QK/PV fragment address math + LDQ/LDK pads, mask + online softmax block
// (bound test widened to two-sided), s_P byte-store relayout + __syncwarp,
// s_vt transpose, cp.async double-buffer with src_bytes=0 tail zero-fill.
#pragma once
#include <cuda_fp8.h>
#include <float.h>

#include <cstdint>

namespace fdmma {

// width-12 2026-07-10: p[16] plumbing (matches q27k::CP3/IP3); the kernel
// stays W<=8 until s_geo re-stride + dispatch cases 9..16 land (plan P2).
struct FCP3 { const float* p[16]; };
struct FIP3 { const int* p[16]; };

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
    d0 = c0; d1 = c1; d2 = c2; d3 = c3; // sm<89 stub: caller must arch-gate
#endif
}
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

// smem: s_q [96][LDQ] | s_kraw x2 [PP][LDK] | s_vraw x2 [PP][HD] |
//       s_vt [HD][PP] | s_P [6][TT][PP] | s_geo (per-lane lo/hi + flags)
// = 70,016B + 48B geo. LDQ/LDK pads kill 8-way conflicts on the u32
// fragment loads; every fragment offset is 4B-aligned and never reads the
// pad tail (prefill review findings #3/#4).
constexpr int FDMMA_TT = 16, FDMMA_PP = 32, FDMMA_HD = 256;
constexpr int FDMMA_LDQ = FDMMA_HD + 4, FDMMA_LDK = FDMMA_HD + 16;
constexpr int FDMMA_ST = 258; // == FD_ST
static_assert(FDMMA_LDQ % 4 == 0 && FDMMA_LDK % 4 == 0, "u32 fragment loads need 4B alignment");

constexpr size_t fdmma_smem_bytes() {
    return (size_t)96 * FDMMA_LDQ                 // s_q
           + 2 * FDMMA_PP * FDMMA_LDK             // s_kraw ping-pong
           + 2 * FDMMA_PP * FDMMA_HD              // s_vraw ping-pong
           + (size_t)FDMMA_HD * FDMMA_PP          // s_vt
           + 6 * FDMMA_TT * FDMMA_PP              // s_P
           + 3 * 16 * sizeof(int) + 4 * sizeof(int); // s_geo: lo/hi/seq[16] + {any,beg,end,pad}
}

template <int W>
__global__ void __launch_bounds__(192, 1)
    k_attn_fdmma(FCP3 qp, int q_stride, const __nv_fp8_e4m3* __restrict__ kc,
                 const __nv_fp8_e4m3* __restrict__ vc, float* __restrict__ part, FIP3 pos,
                 int n_kv_heads, int gqa, int head_dim, float scale) {
    constexpr int TT = FDMMA_TT, PP = FDMMA_PP, HD = FDMMA_HD, LDQ = FDMMA_LDQ, LDK = FDMMA_LDK;
    constexpr int M = 6 * W;                      // live rows (dense r = j*W + t)
    constexpr int LIVE_WARPS = (M + TT - 1) / TT; // W=4 -> 2 .. W=12 -> 5
    // width-12 P2: geometry is 16-ready (M = 6W <= 96 s_q rows, LIVE_WARPS
    // <= 6 = the 192-thread launch); s_geo strides at 16 lanes.
    static_assert(W >= 2 && W <= 16, "s_q is 96 rows = 6*16 lanes max");
    const int sp = blockIdx.x, kvh = blockIdx.y;
    const int nsp = gridDim.x; // == ns fed to k_attn_fd_combine (single source, no split-brain)
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int gid = lane >> 2, tg = lane & 3;

    extern __shared__ unsigned char smem_raw8[];
    __nv_fp8_e4m3* s_q = (__nv_fp8_e4m3*)smem_raw8;                 // [96][LDQ]
    __nv_fp8_e4m3* s_kraw = s_q + (size_t)96 * LDQ;                 // [2][PP][LDK]
    __nv_fp8_e4m3* s_vraw = s_kraw + 2 * PP * LDK;                  // [2][PP][HD]
    __nv_fp8_e4m3* s_vt = s_vraw + 2 * PP * HD;                     // [HD][PP]
    __nv_fp8_e4m3* s_P = s_vt + (size_t)HD * PP;                    // [6][TT][PP]
    int* s_geo = (int*)(s_P + 6 * TT * PP); // lo[8] hi[8] seq[8] any beg end

    // ---- per-lane split geometry (combine's own formula, spec3.cu:248-250).
    // Only t < W is ever dereferenced: pos.p/qp.p slots beyond ntok hold
    // garbage by contract. Dead rows (r >= M) get lo=hi=0 (mask-all, no UB).
    if (threadIdx.x < W) {
        const int t = threadIdx.x;
        const int seq = *pos.p[t] + 1;
        const int ch = (seq + nsp - 1) / nsp;
        const int lo = sp * ch, hi = min(seq, lo + ch);
        s_geo[t] = lo;
        s_geo[16 + t] = hi;
        s_geo[32 + t] = seq;
    }
    if (threadIdx.x >= W && threadIdx.x < 16) {
        s_geo[threadIdx.x] = 0; s_geo[16 + threadIdx.x] = 0; s_geo[32 + threadIdx.x] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int beg = INT_MAX, end = 0, any = 0;
        for (int t = 0; t < W; t++)
            if (s_geo[t] < s_geo[32 + t]) { // live: lo < seq (fd2's rule)
                any = 1;
                beg = min(beg, s_geo[t]);
                end = max(end, s_geo[16 + t]);
            }
        s_geo[48] = any;
        s_geo[49] = any ? (beg & ~(PP - 1)) : 0;
        s_geo[50] = end;
    }
    __syncthreads();
    // uniform early-out BEFORE any cp.async is issued (pending-async at
    // kernel exit is UB); combine never reads the unwritten slots.
    if (!s_geo[48]) return;
    const int p_beg = s_geo[49], p_end = s_geo[50];

    // ---- stage Q once: rows [0, LIVE_WARPS*TT). Rows < M from lane t = r%W,
    // head j = r/W (bare e4m3 cast, no scale -- prefill precedent); rows >= M
    // ZERO-FILLED (the A fragments of the last live warp read them; unstaged
    // smem would be an initcheck fail and one mask-reorder from corruption).
    for (int idx = threadIdx.x; idx < LIVE_WARPS * TT * HD; idx += 192) {
        const int r = idx / HD, d = idx % HD;
        float v = 0.f;
        if (r < M) {
            const int t = r % W, j = r / W;
            v = qp.p[t][(size_t)(kvh * gqa + j) * q_stride + d];
        }
        s_q[(size_t)r * LDQ + d] = __nv_fp8_e4m3(v);
    }

    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[i][e] = 0.f;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.f, l1 = 0.f;

    // this thread's two rows and their causal windows (smem-read once)
    const int R0 = warp * TT + gid, R1 = R0 + 8;
    const int lo0 = R0 < M ? s_geo[R0 % W] : 0, hi0 = R0 < M ? s_geo[16 + R0 % W] : 0;
    const int lo1 = R1 < M ? s_geo[R1 % W] : 0, hi1 = R1 < M ? s_geo[16 + R1 % W] : 0;

    // ---- prologue prefetch (tile p_beg) into buffer 0
    int cur = 0;
    for (int idx = threadIdx.x; idx < PP * (HD / 16); idx += 192) {
        const int pp = idx / (HD / 16), d16 = (idx % (HD / 16)) * 16;
        const int gpos = p_beg + pp;
        const size_t off = ((size_t)gpos * n_kv_heads + kvh) * head_dim + d16;
        cpasync16(s_kraw + pp * LDK + d16, &kc[off], gpos < p_end ? 16 : 0);
        cpasync16(s_vraw + pp * HD + d16, &vc[off], gpos < p_end ? 16 : 0);
    }
    cpasync_commit();

    for (int p0 = p_beg; p0 < p_end; p0 += PP) {
        __nv_fp8_e4m3* kbuf = s_kraw + cur * PP * LDK;
        __nv_fp8_e4m3* vbuf = s_vraw + cur * PP * HD;
        cpasync_wait_all();
        __syncthreads();
        // transpose V [key][dim] -> s_vt[dim][key] (all 6 warps)
        for (int idx = threadIdx.x; idx < PP * HD; idx += 192)
            s_vt[(idx % HD) * PP + (idx / HD)] = vbuf[idx];
        __syncthreads();
        if (p0 + PP < p_end) {
            __nv_fp8_e4m3* knext = s_kraw + (1 - cur) * PP * LDK;
            __nv_fp8_e4m3* vnext = s_vraw + (1 - cur) * PP * HD;
            for (int idx = threadIdx.x; idx < PP * (HD / 16); idx += 192) {
                const int pp = idx / (HD / 16), d16 = (idx % (HD / 16)) * 16;
                const int gpos = p0 + PP + pp;
                const size_t off = ((size_t)gpos * n_kv_heads + kvh) * head_dim + d16;
                cpasync16(knext + pp * LDK + d16, &kc[off], gpos < p_end ? 16 : 0);
                cpasync16(vnext + pp * HD + d16, &vc[off], gpos < p_end ? 16 : 0);
            }
            cpasync_commit();
        }
        cur = 1 - cur;
        // dead warps: staging + barriers only. BOTH __syncthreads of this
        // iteration are ABOVE this guard -- adding any barrier below it
        // deadlocks (checked-fragile, keep it that way).
        if (warp >= LIVE_WARPS) continue;

        // ---- QK^T: A = s_q rows R0/R1, B = K natural [key][LDK] col-major
        float s[4][4];
#pragma unroll
        for (int n = 0; n < 4; n++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[n][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < HD / 32; kk++) {
            const int kb = kk * 32;
            uint32_t a0 = *(const uint32_t*)(s_q + (size_t)R0 * LDQ + kb + tg * 4);
            uint32_t a1 = *(const uint32_t*)(s_q + (size_t)R1 * LDQ + kb + tg * 4);
            uint32_t a2 = *(const uint32_t*)(s_q + (size_t)R0 * LDQ + kb + tg * 4 + 16);
            uint32_t a3 = *(const uint32_t*)(s_q + (size_t)R1 * LDQ + kb + tg * 4 + 16);
#pragma unroll
            for (int n = 0; n < 4; n++) {
                uint32_t b0 = *(const uint32_t*)(kbuf + (n * 8 + gid) * LDK + kb + tg * 4);
                uint32_t b1 = *(const uint32_t*)(kbuf + (n * 8 + gid) * LDK + kb + tg * 4 + 16);
                mma_e4m3(s[n][0], s[n][1], s[n][2], s[n][3], a0, a1, a2, a3, b0, b1, s[n][0],
                         s[n][1], s[n][2], s[n][3]);
            }
        }

        // ---- mask + online softmax (prefill idiom; bound test TWO-SIDED:
        // global pos p0+c masked iff outside this row's [lo, hi) window --
        // the lo side carves per-lane windows out of the shared union stream)
        const int a0r = lo0 - p0, b0r = hi0 - p0;
        const int a1r = lo1 - p0, b1r = hi1 - p0;
        float rmax0 = -FLT_MAX, rmax1 = -FLT_MAX;
#pragma unroll
        for (int n = 0; n < 4; n++) {
            const int c0 = n * 8 + tg * 2, c1 = c0 + 1;
#pragma unroll
            for (int e = 0; e < 4; e++) s[n][e] *= scale;
            if (c0 < a0r || c0 >= b0r) s[n][0] = -FLT_MAX;
            if (c1 < a0r || c1 >= b0r) s[n][1] = -FLT_MAX;
            if (c0 < a1r || c0 >= b1r) s[n][2] = -FLT_MAX;
            if (c1 < a1r || c1 >= b1r) s[n][3] = -FLT_MAX;
            rmax0 = fmaxf(rmax0, fmaxf(s[n][0], s[n][1]));
            rmax1 = fmaxf(rmax1, fmaxf(s[n][2], s[n][3]));
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffff, rmax0, off));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffff, rmax1, off));
        }
        const float mn0 = fmaxf(m0, rmax0), mn1 = fmaxf(m1, rmax1);
        const float sc0 = m0 == -FLT_MAX ? 0.f : expf(m0 - mn0);
        const float sc1 = m1 == -FLT_MAX ? 0.f : expf(m1 - mn1);
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

        // ---- P relayout D-frag -> A-frag through s_P: BYTE stores (u32
        // stores are impossible from the D fragment -- cols {2tg, 2tg+1}
        // live in different threads than keys {tg*4..+3}), __syncwarp
        // (same-warp ownership), u32 reads. Donor prefill.cu:1700-1712.
        __nv_fp8_e4m3* pw = s_P + warp * TT * PP;
#pragma unroll
        for (int n = 0; n < 4; n++) {
            pw[gid * PP + (n * 8 + 2 * tg)] = __nv_fp8_e4m3(s[n][0]);
            pw[gid * PP + (n * 8 + 2 * tg + 1)] = __nv_fp8_e4m3(s[n][1]);
            pw[(gid + 8) * PP + (n * 8 + 2 * tg)] = __nv_fp8_e4m3(s[n][2]);
            pw[(gid + 8) * PP + (n * 8 + 2 * tg + 1)] = __nv_fp8_e4m3(s[n][3]);
        }
        __syncwarp();
        uint32_t a0 = *(const uint32_t*)(pw + gid * PP + tg * 4);
        uint32_t a1 = *(const uint32_t*)(pw + (gid + 8) * PP + tg * 4);
        uint32_t a2 = *(const uint32_t*)(pw + gid * PP + tg * 4 + 16);
        uint32_t a3 = *(const uint32_t*)(pw + (gid + 8) * PP + tg * 4 + 16);
        // ---- PV: B = s_vt[dim][key], 4 consecutive keys = 1 aligned u32
#pragma unroll
        for (int nt = 0; nt < 32; nt++) {
            const int dim = nt * 8 + gid;
            uint32_t b0 = *(const uint32_t*)(s_vt + dim * PP + tg * 4);
            uint32_t b1 = *(const uint32_t*)(s_vt + dim * PP + tg * 4 + 16);
            mma_e4m3(o[nt][0], o[nt][1], o[nt][2], o[nt][3], a0, a1, a2, a3, b0, b1, o[nt][0],
                     o[nt][1], o[nt][2], o[nt][3]);
        }
    }

    // ---- epilogue: unnormalized partials, fd2's exact layout + write rule.
    // Write iff row live AND lane live at this split (sp*chunk_t < seq_t):
    // combine's max loop reads EVERY slot sp < used_t unguarded over
    // never-zeroed scratch -- the written-slot set must match exactly.
    if (warp >= LIVE_WARPS) return;
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        const int R = rr ? R1 : R0;
        if (R >= M) continue;
        const int t = R % W, j = R / W;
        if (s_geo[t] >= s_geo[32 + t]) continue; // lane empty at this split
        const float m = rr ? m1 : m0, l = rr ? l1 : l0;
        const size_t pair = (size_t)t * (n_kv_heads * gqa) + kvh * gqa + j;
        float* dst = part + (pair * nsp + sp) * FDMMA_ST;
        if (tg == 0) { dst[0] = m; dst[1] = l; }
#pragma unroll
        for (int n = 0; n < 32; n++) {
            const int c = n * 8 + tg * 2;
            dst[2 + c] = o[n][rr ? 2 : 0];
            dst[2 + c + 1] = o[n][rr ? 3 : 1];
        }
    }
}

// launcher: grid (ns, n_kv_heads); ns MUST equal the ns passed to
// k_attn_fd_combine (128 = FD2_NS in the engine). One-shot smem attr raise.
template <int W>
inline void launch_fdmma_w(FCP3 qp, int q_stride, const void* kc, const void* vc, float* part,
                           FIP3 pos, int n_kv_heads, int gqa, int head_dim, float scale, int ns,
                           cudaStream_t st) {
    static bool attr = false;
    const size_t sm = fdmma_smem_bytes();
    if (!attr) {
        cudaFuncSetAttribute(k_attn_fdmma<W>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
        attr = true;
    }
    dim3 g((unsigned)ns, (unsigned)n_kv_heads);
    k_attn_fdmma<W><<<g, 192, sm, st>>>(qp, q_stride, (const __nv_fp8_e4m3*)kc,
                                        (const __nv_fp8_e4m3*)vc, part, pos, n_kv_heads, gqa,
                                        head_dim, scale);
}

inline bool launch_fdmma(FCP3 qp, int q_stride, const void* kc, const void* vc, float* part,
                         FIP3 pos, int n_kv_heads, int gqa, int head_dim, float scale, int ns,
                         int ntok, cudaStream_t st) {
    switch (ntok) {
        case 4: launch_fdmma_w<4>(qp, q_stride, kc, vc, part, pos, n_kv_heads, gqa, head_dim, scale, ns, st); return true;
        case 5: launch_fdmma_w<5>(qp, q_stride, kc, vc, part, pos, n_kv_heads, gqa, head_dim, scale, ns, st); return true;
        case 6: launch_fdmma_w<6>(qp, q_stride, kc, vc, part, pos, n_kv_heads, gqa, head_dim, scale, ns, st); return true;
        case 7: launch_fdmma_w<7>(qp, q_stride, kc, vc, part, pos, n_kv_heads, gqa, head_dim, scale, ns, st); return true;
        case 8: launch_fdmma_w<8>(qp, q_stride, kc, vc, part, pos, n_kv_heads, gqa, head_dim, scale, ns, st); return true;
        // width-12 P2: the suffix drafter's wide verify (Q27_SUFFIX_W).
        // 13..16 compile (kernel is 16-ready) but stay uninstantiated until
        // something launches them; caller MUST honor the false return.
        case 9: launch_fdmma_w<9>(qp, q_stride, kc, vc, part, pos, n_kv_heads, gqa, head_dim, scale, ns, st); return true;
        case 10: launch_fdmma_w<10>(qp, q_stride, kc, vc, part, pos, n_kv_heads, gqa, head_dim, scale, ns, st); return true;
        case 11: launch_fdmma_w<11>(qp, q_stride, kc, vc, part, pos, n_kv_heads, gqa, head_dim, scale, ns, st); return true;
        case 12: launch_fdmma_w<12>(qp, q_stride, kc, vc, part, pos, n_kv_heads, gqa, head_dim, scale, ns, st); return true;
        default: return false; // W<4 (and 13..16) stay on fd2
    }
}

} // namespace fdmma
