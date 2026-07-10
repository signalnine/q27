// fdmma_test -- microtests for k_attn_fdmma (design doc
// docs/plans/2026-07-10-fdmma-verify-attn.md, "microtest plan" section).
// Structural surfaces are asserted EXACTLY (written-slot set both directions
// on NaN-poisoned scratch, poisoned t>=W pointers, bitwise repeat-run);
// numerics are HARD-gated vs a MODELED fp64 reference that replicates the
// kernel's exact pipeline (per-sp partial chains, 32-key block softmax,
// e4m3 Q and P casts, fp64 combine) -- residual is fp32/MMA rounding only
// (measured 4-10e-6). A second EXACT-softmax fp64 ref is reported unGated:
// it measures the quantization physics (cos 0.9987-0.9999, rel 5-19% on
// synthetics), the quantity the engine acceptance A/B will judge.
// Usage: fdmma_test   (synthetic tensors; no model)
#include <cuda_fp8.h>
#include <float.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <algorithm>
#include <vector>

#include "../src/fdmma.cuh"

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err__ = (x);                                               \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static constexpr int N_KV = 4, GQA = 6, HD = 256, NQH = N_KV * GQA;
static constexpr int NS = 128, ST = 258;
static int fails = 0;
#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) { printf("FAIL "); printf(__VA_ARGS__); printf("\n"); fails++; } \
    } while (0)

// combine fork (verbatim src/spec3.cu k_attn_fd_combine)
struct P3 { float* p[8]; };
__global__ void k_attn_fd_combine(const float* __restrict__ part, P3 outp, int n_heads,
                                  int head_dim, int ns, fdmma::FIP3 pos) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int seq = *pos.p[t] + 1;
    const int chunk = (seq + ns - 1) / ns;
    const int used = (seq + chunk - 1) / chunk;
    size_t pair = (size_t)t * n_heads + h;
    const float* pp = part + pair * ns * ST;
    __shared__ float s_m, s_l;
    if (threadIdx.x == 0) {
        float mg = -FLT_MAX;
        for (int sp = 0; sp < used; sp++) mg = fmaxf(mg, pp[sp * ST]);
        float lg = 0.f;
        for (int sp = 0; sp < used; sp++)
            if (pp[sp * ST] != -FLT_MAX) lg += pp[sp * ST + 1] * expf(pp[sp * ST] - mg);
        s_m = mg;
        s_l = lg;
    }
    __syncthreads();
    const float mg = s_m, inv = 1.0f / s_l;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float a = 0.f;
        for (int sp = 0; sp < used; sp++) {
            float ms = pp[sp * ST];
            if (ms != -FLT_MAX) a += pp[sp * ST + 2 + d] * expf(ms - mg);
        }
        outp.p[t][(size_t)h * head_dim + d] = a * inv;
    }
}

static float e4m3_rt(float v) { // host round-trip through e4m3
    __nv_fp8_e4m3 e(v);
    return float(e);
}

int main() {
    const int MAXSEQ = 1024 + 32;
    const size_t kvn = (size_t)MAXSEQ * N_KV * HD;
    std::vector<uint8_t> hk(kvn), hv(kvn);
    unsigned s = 42;
    for (size_t i = 0; i < kvn; i++) {
        s = s * 1664525u + 1013904223u;
        hk[i] = (s >> 13) & 0xBF; // |v| <= 1.875, no NaN codes
        s = s * 1664525u + 1013904223u;
        hv[i] = (s >> 13) & 0xBF;
    }
    __nv_fp8_e4m3 *kc, *vc;
    CUDA_CHECK(cudaMalloc(&kc, kvn));
    CUDA_CHECK(cudaMalloc(&vc, kvn));
    CUDA_CHECK(cudaMemcpy(kc, hk.data(), kvn, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vc, hv.data(), kvn, cudaMemcpyHostToDevice));

    const int QSTRIDE = 2 * HD; // engine q_stride
    std::vector<float> hq(8 * (size_t)NQH * QSTRIDE);
    for (auto& v : hq) {
        s = s * 1664525u + 1013904223u;
        v = ((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
    }
    float* d_q[8];
    int* d_pos[8];
    float* d_out[8];
    for (int t = 0; t < 8; t++) {
        CUDA_CHECK(cudaMalloc(&d_q[t], (size_t)NQH * QSTRIDE * 4));
        CUDA_CHECK(cudaMemcpy(d_q[t], hq.data() + t * (size_t)NQH * QSTRIDE,
                              (size_t)NQH * QSTRIDE * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_pos[t], 4));
        CUDA_CHECK(cudaMalloc(&d_out[t], (size_t)NQH * HD * 4));
    }
    float* part;
    const size_t partn = (size_t)8 * NQH * NS * ST;
    CUDA_CHECK(cudaMalloc(&part, partn * 4));

    const float scale = 1.0f / sqrtf((float)HD);
    const float qnan = nanf("");

    // seq bases: straddle k*NS so per-lane chunk_t DISAGREE within a block
    // (design risk #1) plus a short-seq case with many empty splits.
    for (int W : {4, 5, 6, 8}) {
        for (int base : {NS * 7 - 2 /*894: straddles 896=7*128*/, 800, 130}) {
            // lane t position = base - 1 + t  (seq_t = base + t)
            fdmma::FCP3 qp{};
            fdmma::FIP3 pp{};
            P3 op{};
            for (int t = 0; t < W; t++) {
                int hp = base - 1 + t;
                CUDA_CHECK(cudaMemcpy(d_pos[t], &hp, 4, cudaMemcpyHostToDevice));
                qp.p[t] = d_q[t];
                pp.p[t] = d_pos[t];
                op.p[t] = d_out[t];
            }
            // poison t >= W slots: kernel must never dereference them
            for (int t = W; t < 8; t++) {
                qp.p[t] = (const float*)0xdeadbeef00ull;
                pp.p[t] = (const int*)0xdeadbeef00ull;
            }
            // T1: NaN-poison scratch, run, assert exact written-slot set
            {
                std::vector<float> poison(partn, qnan);
                CUDA_CHECK(cudaMemcpy(part, poison.data(), partn * 4, cudaMemcpyHostToDevice));
            }
            bool ok = fdmma::launch_fdmma(qp, QSTRIDE, kc, vc, part, pp, N_KV, GQA, HD, scale,
                                          NS, W, 0);
            CHECK(ok, "launch W=%d", W);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> hpart(partn);
            CUDA_CHECK(cudaMemcpy(hpart.data(), part, partn * 4, cudaMemcpyDeviceToHost));
            for (int t = 0; t < W; t++) {
                const int seq = base + t;
                const int chunk = (seq + NS - 1) / NS;
                const int used = (seq + chunk - 1) / chunk;
                for (int h = 0; h < NQH; h++) {
                    for (int sp = 0; sp < NS; sp++) {
                        const size_t idx = (((size_t)t * NQH + h) * NS + sp) * ST;
                        const bool wrote = !std::isnan(hpart[idx]);
                        const bool expect = sp < used;
                        if (wrote != expect) {
                            CHECK(false, "slot W=%d base=%d t=%d h=%d sp=%d wrote=%d expect=%d",
                                  W, base, t, h, sp, wrote, expect);
                            sp = NS; h = NQH; // stop flood
                        }
                    }
                }
            }
            // T3: bitwise repeat-run
            std::vector<float> hpart2(partn);
            fdmma::launch_fdmma(qp, QSTRIDE, kc, vc, part, pp, N_KV, GQA, HD, scale, NS, W, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hpart2.data(), part, partn * 4, cudaMemcpyDeviceToHost));
            // compare only written slots (unwritten hold stale NaN pattern both runs)
            CHECK(memcmp(hpart.data(), hpart2.data(), partn * 4) == 0,
                  "bitwise repeat W=%d base=%d", W, base);

            // T2: combine -> outputs vs fp64 CPU reference
            dim3 g2(NQH, W);
            k_attn_fd_combine<<<g2, 256>>>(part, op, NQH, HD, NS, pp);
            CUDA_CHECK(cudaDeviceSynchronize());
            // Two references, one purpose each:
            // (M) MODELED ref -- replicates the kernel's numerics pipeline in
            //     fp64: per-sp partial chains, per-32-tile block softmax with
            //     running max, e4m3 cast of P (o accumulates quantized w; l
            //     accumulates unquantized w -- the kernel's l comes from the
            //     fp32 reduce BEFORE the cast), fp64 combine. Residual vs the
            //     kernel = fp32/MMA rounding only -> HARD GATE (rel < 1e-2,
            //     cosine > 0.9999): any masking/tiling/fragment bug explodes.
            // (X) EXACT fp64 softmax ref -- reported, no hard gate: measures
            //     the quantization PHYSICS (feeds the acceptance-risk call
            //     and the fp8q-fallback decision).
            double wcosM = 1.0, wrelM = 0.0, wcosX = 1.0, wrelX = 0.0;
            for (int t = 0; t < W; t++) {
                std::vector<float> got(NQH * HD);
                CUDA_CHECK(cudaMemcpy(got.data(), d_out[t], (size_t)NQH * HD * 4,
                                      cudaMemcpyDeviceToHost));
                const int seq = base + t;
                const int chunk = (seq + NS - 1) / NS;
                const int used = (seq + chunk - 1) / chunk;
                for (int h = 0; h < NQH; h++) {
                    const int kvh = h / GQA;
                    std::vector<double> qh(HD);
                    for (int d = 0; d < HD; d++)
                        qh[d] = e4m3_rt(hq[t * (size_t)NQH * QSTRIDE + (size_t)h * QSTRIDE + d]);
                    auto score = [&](int p) {
                        double d0 = 0;
                        const size_t off = ((size_t)p * N_KV + kvh) * HD;
                        for (int d = 0; d < HD; d++) {
                            __nv_fp8_e4m3 e;
                            e.__x = hk[off + d];
                            d0 += qh[d] * (double)float(e);
                        }
                        return d0 * (double)scale;
                    };
                    // (M) modeled: per-sp partials over the block's tile grid
                    std::vector<double> pm(used), pl(used);
                    std::vector<std::vector<double>> po(used, std::vector<double>(HD, 0.0));
                    for (int sp = 0; sp < used; sp++) {
                        const int lo = sp * chunk, hi = std::min(seq, lo + chunk);
                        // block p_beg = min over LIVE lanes' lo at this sp, &~31
                        int beg = INT_MAX, end = 0;
                        for (int u = 0; u < W; u++) {
                            const int sq = base + u, cu = (sq + NS - 1) / NS;
                            const int lu = sp * cu, hu = std::min(sq, lu + cu);
                            if (lu < sq) { beg = std::min(beg, lu); end = std::max(end, hu); }
                        }
                        beg &= ~31;
                        double m = -std::numeric_limits<double>::infinity(), l = 0;
                        for (int p0 = beg; p0 < end; p0 += 32) {
                            double tmax = -std::numeric_limits<double>::infinity();
                            double w[32];
                            bool msk[32];
                            for (int c = 0; c < 32; c++) {
                                const int p = p0 + c;
                                msk[c] = p < lo || p >= hi;
                                if (msk[c]) { w[c] = 0; continue; }
                                w[c] = score(p);
                                tmax = std::max(tmax, w[c]);
                            }
                            const double mn = std::max(m, tmax);
                            const double scl = std::isinf(m) ? 0.0 : exp(m - mn);
                            double rl = 0;
                            for (int d = 0; d < HD; d++) po[sp][d] *= scl;
                            for (int c = 0; c < 32; c++) {
                                if (msk[c]) continue;
                                const double ww = exp(w[c] - mn);
                                rl += ww;
                                const double wq = e4m3_rt((float)ww); // P cast
                                const size_t off = ((size_t)(p0 + c) * N_KV + kvh) * HD;
                                for (int d = 0; d < HD; d++) {
                                    __nv_fp8_e4m3 e;
                                    e.__x = hv[off + d];
                                    po[sp][d] += wq * (double)float(e);
                                }
                            }
                            l = l * scl + rl;
                            m = mn;
                        }
                        pm[sp] = m;
                        pl[sp] = l;
                    }
                    // fp64 combine
                    double mg = -std::numeric_limits<double>::infinity();
                    for (int sp = 0; sp < used; sp++) mg = std::max(mg, pm[sp]);
                    double lg = 0;
                    for (int sp = 0; sp < used; sp++)
                        if (!std::isinf(pm[sp])) lg += pl[sp] * exp(pm[sp] - mg);
                    std::vector<double> orefM(HD, 0.0);
                    for (int sp = 0; sp < used; sp++) {
                        if (std::isinf(pm[sp])) continue;
                        const double f = exp(pm[sp] - mg);
                        for (int d = 0; d < HD; d++) orefM[d] += po[sp][d] * f;
                    }
                    for (int d = 0; d < HD; d++) orefM[d] /= lg;
                    // (X) exact softmax
                    std::vector<double> sc_(seq);
                    double mx = -1e300;
                    for (int p = 0; p < seq; p++) { sc_[p] = score(p); mx = std::max(mx, sc_[p]); }
                    double lsum = 0;
                    for (int p = 0; p < seq; p++) { sc_[p] = exp(sc_[p] - mx); lsum += sc_[p]; }
                    std::vector<double> orefX(HD, 0.0);
                    for (int p = 0; p < seq; p++) {
                        const size_t off = ((size_t)p * N_KV + kvh) * HD;
                        const double w = sc_[p] / lsum;
                        for (int d = 0; d < HD; d++) {
                            __nv_fp8_e4m3 e;
                            e.__x = hv[off + d];
                            orefX[d] += w * (double)float(e);
                        }
                    }
                    auto cmp = [&](const std::vector<double>& ref, double& wcos, double& wrel) {
                        double dot = 0, na = 0, nb = 0, mrel = 0, rms = 0;
                        for (int d = 0; d < HD; d++) rms += ref[d] * ref[d];
                        rms = sqrt(rms / HD) + 1e-12;
                        for (int d = 0; d < HD; d++) {
                            const double a = ref[d], b = got[(size_t)h * HD + d];
                            dot += a * b; na += a * a; nb += b * b;
                            mrel = std::max(mrel, fabs(a - b) / rms);
                        }
                        wcos = std::min(wcos, dot / (sqrt(na * nb) + 1e-300));
                        wrel = std::max(wrel, mrel);
                    };
                    cmp(orefM, wcosM, wrelM);
                    cmp(orefX, wcosX, wrelX);
                }
            }
            CHECK(wcosM > 0.9999, "modeled cosine W=%d base=%d worst %.7f", W, base, wcosM);
            CHECK(wrelM < 1e-2, "modeled rel W=%d base=%d worst %.3e", W, base, wrelM);
            printf("W=%d base=%4d: slots OK, bitwise OK | modeled cos %.7f rel %.2e | "
                   "physics cos %.6f rel %.2e\n",
                   W, base, wcosM, wrelM, wcosX, wrelX);
        }
    }
    printf(fails ? "fdmma_test: %d FAILURES\n" : "fdmma_test: all tests PASS\n", fails);
    return fails ? 1 : 0;
}
