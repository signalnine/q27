// Self-consistency tests: GPU kernels vs CPU reference over the same q27 data.
// No external ground truth needed; validates layout, nibble order, scales, GEMV.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "blocks.cuh"
#include "cuda_common.h"
#include "device_model.h"
#include "kernels.cuh"
#include "loader.h"
#include "prefill.cuh"
#include "spec3.cuh"

using q27::DType;

static int g_fail = 0;

static void check(const char* name, double err, double tol) {
    bool ok = err < tol;
    printf("  %-44s %s  (err %.3e, tol %.0e)\n", name, ok ? "PASS" : "FAIL", err, tol);
    if (!ok) g_fail++;
}

// CPU dequant of row r, element c
static float cpu_deq(const q27::Tensor& t, int64_t r, int64_t c) {
    if (t.dtype == DType::Q4_G64) {
        uint8_t b = t.data[r * (t.cols() / 2) + c / 2];
        int nib = (c & 1) ? (b >> 4) : (b & 0xF);
        __half s = ((const __half*)t.scales)[r * (t.cols() / 64) + c / 64];
        return (nib - 8) * __half2float(s);
    }
    if (t.dtype == DType::Q8_G128) {
        int8_t v = ((const int8_t*)t.data)[r * t.cols() + c];
        __half s = ((const __half*)t.scales)[r * (t.cols() / 128) + c / 128];
        return (float)v * __half2float(s);
    }
    if (t.dtype == DType::F16) return __half2float(((const __half*)t.data)[r * t.cols() + c]);
    return ((const float*)t.data)[r * t.cols() + c];
}

static std::vector<float> rand_vec(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

static void test_dequant(q27::DeviceModel& dm, const q27::Model& m, const char* name) {
    const q27::Tensor& t = m.get(name);
    const q27::DevTensor& d = dm.upload(name);
    int64_t rows = std::min<int64_t>(t.rows(), 32), cols = t.cols();

    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, t.rows() * cols * 4));
    if (t.dtype == DType::Q4_G64)
        q27k::dequant_q4((const uint8_t*)d.data, (const __half*)d.scales, d_out, t.rows(), cols);
    else
        q27k::dequant_q8((const int8_t*)d.data, (const __half*)d.scales, d_out, t.rows(), cols);
    std::vector<float> got(rows * cols);
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, rows * cols * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    double maxd = 0;
    for (int64_t r = 0; r < rows; r++)
        for (int64_t c = 0; c < cols; c++)
            maxd = std::max(maxd, (double)std::fabs(got[r * cols + c] - cpu_deq(t, r, c)));
    char label[128];
    snprintf(label, sizeof label, "dequant %s (%s)", name, q27::dtype_name(t.dtype));
    check(label, maxd, 1e-6);
}

static void test_gemv(q27::DeviceModel& dm, const q27::Model& m, const char* name) {
    const q27::Tensor& t = m.get(name);
    const q27::DevTensor& d = dm.upload(name);
    int64_t rows = t.rows(), cols = t.cols();
    int64_t check_rows = std::min<int64_t>(rows, 128);

    std::vector<float> x = rand_vec(cols, 42);
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, cols * 4));
    CUDA_CHECK(cudaMalloc(&d_y, rows * 4));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), cols * 4, cudaMemcpyHostToDevice));
    q27k::XQuant xq = q27k::xquant_alloc(cols);
    q27k::quantize_x(d_x, cols, xq);

    switch (t.dtype) {
        case DType::Q4_G64:
            q27k::gemv_q4((const uint8_t*)d.data, (const __half*)d.scales, xq, d_y, rows, cols);
            break;
        case DType::Q8_G128:
            q27k::gemv_q8((const int8_t*)d.data, (const __half*)d.scales, xq, d_y, rows, cols);
            break;
        case DType::F16:
            q27k::gemv_f16((const __half*)d.data, d_x, d_y, rows, cols);
            break;
        default:
            printf("  gemv %s: unsupported dtype, skip\n", name);
            return;
    }
    std::vector<float> got(check_rows);
    CUDA_CHECK(cudaMemcpy(got.data(), d_y, check_rows * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    // normalize worst abs error by RMS of the reference dots: per-row relative
    // error is meaningless when a row's true dot is near zero (int8-x noise
    // dominates the tiny denominator).
    double max_abs = 0, ss = 0;
    for (int64_t r = 0; r < check_rows; r++) {
        double ref = 0;
        for (int64_t c = 0; c < cols; c++) ref += (double)cpu_deq(t, r, c) * x[c];
        ss += ref * ref;
        max_abs = std::max(max_abs, std::fabs(got[r] - ref));
    }
    double max_rel = max_abs / (std::sqrt(ss / check_rows) + 1e-9);
    char label[128];
    snprintf(label, sizeof label, "gemv %s (%s)", name, q27::dtype_name(t.dtype));
    check(label, max_rel, 2e-2); // includes int8 activation-quant error
}

static void test_rmsnorm(const q27::Model& m) {
    const q27::Tensor& w = m.get("output_norm.weight");
    int n = (int)w.cols();
    std::vector<float> x = rand_vec(n, 7);
    const float* wf = (const float*)w.data;

    float *d_x, *d_w, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, n * 4));
    CUDA_CHECK(cudaMalloc(&d_w, n * 4));
    CUDA_CHECK(cudaMalloc(&d_y, n * 4));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, wf, n * 4, cudaMemcpyHostToDevice));
    q27k::rmsnorm(d_x, d_w, d_y, n, 1e-6f);
    std::vector<float> got(n);
    CUDA_CHECK(cudaMemcpy(got.data(), d_y, n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_y));

    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    double inv = 1.0 / std::sqrt(ss / n + 1e-6);
    double maxd = 0;
    for (int i = 0; i < n; i++)
        maxd = std::max(maxd, std::fabs((double)got[i] - x[i] * inv * wf[i]));
    check("rmsnorm(5120) vs CPU", maxd, 1e-4);
}

static void test_silu_mul() {
    int n = 17408;
    std::vector<float> g = rand_vec(n, 1), u = rand_vec(n, 2);
    float *d_g, *d_u, *d_o;
    CUDA_CHECK(cudaMalloc(&d_g, n * 4)); CUDA_CHECK(cudaMalloc(&d_u, n * 4));
    CUDA_CHECK(cudaMalloc(&d_o, n * 4));
    CUDA_CHECK(cudaMemcpy(d_g, g.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u, u.data(), n * 4, cudaMemcpyHostToDevice));
    q27k::silu_mul(d_g, d_u, d_o, n);
    std::vector<float> got(n);
    CUDA_CHECK(cudaMemcpy(got.data(), d_o, n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_g)); CUDA_CHECK(cudaFree(d_u)); CUDA_CHECK(cudaFree(d_o));
    double maxd = 0;
    for (int i = 0; i < n; i++) {
        double ref = ((double)g[i] / (1.0 + std::exp(-(double)g[i]))) * u[i];
        maxd = std::max(maxd, std::fabs((double)got[i] - ref));
    }
    check("silu_mul(17408) vs CPU", maxd, 1e-5);
}

static void test_embed(q27::DeviceModel& dm, const q27::Model& m) {
    const q27::Tensor& t = m.get("token_embd.weight");
    const q27::DevTensor& d = dm.upload("token_embd.weight");
    int64_t row = 1234, cols = t.cols();
    float* d_out;
    int* d_tok;
    int row_i = (int)row;
    CUDA_CHECK(cudaMalloc(&d_out, cols * 4));
    CUDA_CHECK(cudaMalloc(&d_tok, 4));
    CUDA_CHECK(cudaMemcpy(d_tok, &row_i, 4, cudaMemcpyHostToDevice));
    q27k::embed_row_q8((const int8_t*)d.data, (const __half*)d.scales, d_tok, cols, d_out);
    std::vector<float> got(cols);
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, cols * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
    double maxd = 0;
    for (int64_t c = 0; c < cols; c++)
        maxd = std::max(maxd, (double)std::fabs(got[c] - cpu_deq(t, row, c)));
    check("embed_row_q8(token 1234)", maxd, 1e-6);
}

static void test_gemv_batch(q27::DeviceModel& dm, const q27::Model& m, const char* name) {
    const q27::Tensor& t = m.get(name);
    const q27::DevTensor& d = dm.upload(name);
    int64_t rows = t.rows(), cols = t.cols();
    const int NB = 3;

    q27k::XQuant xqs[4];
    float *d_x, *d_yb, *d_y1;
    CUDA_CHECK(cudaMalloc(&d_x, cols * 4));
    CUDA_CHECK(cudaMalloc(&d_yb, (size_t)NB * rows * 4));
    CUDA_CHECK(cudaMalloc(&d_y1, rows * 4));
    for (int n = 0; n < NB; n++) {
        std::vector<float> x = rand_vec(cols, 100 + n);
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), cols * 4, cudaMemcpyHostToDevice));
        xqs[n] = q27k::xquant_alloc(cols);
        q27k::quantize_x(d_x, cols, xqs[n]);
    }
    double maxd = 0;
    if (t.dtype == DType::Q4_G64) {
        float* const ysb[3] = {d_yb, d_yb + rows, d_yb + 2 * rows};
        q27k::gemv_q4_n((const uint8_t*)d.data, (const __half*)d.scales, xqs, NB, ysb, rows, cols);
        for (int n = 0; n < NB; n++) {
            q27k::gemv_q4((const uint8_t*)d.data, (const __half*)d.scales, xqs[n], d_y1, rows, cols);
            std::vector<float> yb(rows), y1(rows);
            CUDA_CHECK(cudaMemcpy(yb.data(), d_yb + (size_t)n * rows, rows * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(y1.data(), d_y1, rows * 4, cudaMemcpyDeviceToHost));
            for (int64_t r = 0; r < rows; r++)
                maxd = std::max(maxd, (double)std::fabs(yb[r] - y1[r]));
        }
    } else {
        float* const ysb[3] = {d_yb, d_yb + rows, d_yb + 2 * rows};
        q27k::gemv_q8_n((const int8_t*)d.data, (const __half*)d.scales, xqs, NB, ysb, rows, cols);
        for (int n = 0; n < NB; n++) {
            q27k::gemv_q8((const int8_t*)d.data, (const __half*)d.scales, xqs[n], d_y1, rows, cols);
            std::vector<float> yb(rows), y1(rows);
            CUDA_CHECK(cudaMemcpy(yb.data(), d_yb + (size_t)n * rows, rows * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(y1.data(), d_y1, rows * 4, cudaMemcpyDeviceToHost));
            for (int64_t r = 0; r < rows; r++)
                maxd = std::max(maxd, (double)std::fabs(yb[r] - y1[r]));
        }
    }
    char label[128];
    snprintf(label, sizeof label, "gemv_n(3) vs 3x gemv %s", name);
    check(label, maxd, 1e-5); // same math, same order -> near bit-identical
}

// P1: MMA prefill GEMM vs dp4a on real tensors. Same integer chunk dots by
// construction; fp accumulation order differs, so compare at rounding-noise
// tolerance. T=33 exercises the token-tail path. A real indexing/unpack bug
// shows as O(1)+ relative error -- clean separation from the 1e-6 noise floor.
static void test_gemm_mma(q27::DeviceModel& dm, const q27::Model& m, const char* name) {
    const q27::Tensor& t = m.get(name);
    const q27::DevTensor& d = dm.upload(name);
    int64_t rows = t.rows(), cols = t.cols();
    const int T = 33;

    std::vector<float> x = rand_vec((size_t)T * cols, 7);
    float *d_x, *d_ya, *d_yb;
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)T * cols * 4));
    CUDA_CHECK(cudaMalloc(&d_ya, (size_t)T * rows * 4));
    CUDA_CHECK(cudaMalloc(&d_yb, (size_t)T * rows * 4));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), (size_t)T * cols * 4, cudaMemcpyHostToDevice));
    // dp4a GEMM staging over-reads activations to the full 32-token tile (the
    // discard is via nt guards downstream), so the XQuant buffer must be
    // token-tile padded exactly like the engine's xqT (PF_T tokens).
    const int Tpad = (T + 31) & ~31;
    q27k::XQuant xq = q27k::xquant_alloc((size_t)Tpad * cols);
    q27k::quantize_x(d_x, (size_t)T * cols, xq);

    auto run = [&](const char* mode, float* y) {
        setenv("Q27_PREFILL", mode, 1);
        setenv("Q27_PF_XG", "32", 1); // exact-path leg (dp4a is g32; also the
                                      // null-nat64 XQuant would force it)
        if (t.dtype == DType::Q4_G64)
            q27k::gemm_q4_T((const uint8_t*)d.data, (const __half*)d.scales, xq, y, rows, cols,
                            T, 0);
        else
            q27k::gemm_q8_T((const int8_t*)d.data, (const __half*)d.scales, xq, y, rows, cols,
                            T, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    run("dp4a", d_ya);
    run("mma", d_yb);
    unsetenv("Q27_PREFILL");
    unsetenv("Q27_PF_XG");
    std::vector<float> ya((size_t)T * rows), yb((size_t)T * rows);
    CUDA_CHECK(cudaMemcpy(ya.data(), d_ya, ya.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(yb.data(), d_yb, yb.size() * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0;
    for (size_t i = 0; i < ya.size(); i++)
        maxrel = std::max(maxrel, (double)std::fabs(ya[i] - yb[i]) / (1.0 + std::fabs(ya[i])));
    char label[128];
    snprintf(label, sizeof label, "gemm MMA vs dp4a %s", name);
    check(label, maxrel, 1e-4);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_ya));
    CUDA_CHECK(cudaFree(d_yb));
}

// Regroup gate: MMA g64 path vs the dp4a exact path fed the SAME g64
// quantization expanded to g32 form (nat = nat64, both 32-halves of each
// 64-group share its s64 scale, isum/eo rebuilt from nat64). Integer dots
// identical by construction; only fp accumulation grouping differs (per-64
// with int32 chaining vs per-32), so the bound is rounding noise -- an
// indexing/fragment bug in the g64 kernel shows as O(1). This test + PPL +
// canonical REPLACE the serial-vs-batched identity gate for the g64 default
// (policy sign-off 2026-07-04).
static void test_gemm_mma_g64(q27::DeviceModel& dm, const q27::Model& m, const char* name) {
    const q27::Tensor& t = m.get(name);
    const q27::DevTensor& d = dm.upload(name);
    int64_t rows = t.rows(), cols = t.cols();
    const int T = 33;

    std::vector<float> x = rand_vec((size_t)T * cols, 13);
    float *d_x, *d_ya, *d_yb;
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)T * cols * 4));
    CUDA_CHECK(cudaMalloc(&d_ya, (size_t)T * rows * 4));
    CUDA_CHECK(cudaMalloc(&d_yb, (size_t)T * rows * 4));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), (size_t)T * cols * 4, cudaMemcpyHostToDevice));
    const int Tpad = (T + 31) & ~31;
    q27k::XQuant xq = q27k::xquant_alloc((size_t)Tpad * cols, /*g64=*/true);
    q27k::quantize_x_g64(d_x, (size_t)T * cols, xq);
    CUDA_CHECK(cudaDeviceSynchronize());

    // pull (nat64, s64) to host and expand into the reference XQuant's g32 form
    const size_t n = (size_t)T * cols, nb32 = n / 32, ng64 = n / 64;
    std::vector<int8_t> h_nat(n);
    std::vector<float> h_s64(ng64);
    CUDA_CHECK(cudaMemcpy(h_nat.data(), xq.nat64, n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_s64.data(), xq.s64, ng64 * 4, cudaMemcpyDeviceToHost));
    std::vector<float> h_scale(nb32);
    std::vector<int> h_isum(nb32);
    std::vector<uint2> h_eo(n / 8);
    for (size_t b = 0; b < nb32; b++) {
        h_scale[b] = h_s64[b / 2];
        int s = 0;
        for (int i = 0; i < 32; i++) s += h_nat[b * 32 + i];
        h_isum[b] = s;
        for (int u = 0; u < 4; u++) { // elements u*8..u*8+7: even bytes in .x, odd in .y
            const int8_t* p = &h_nat[b * 32 + u * 8];
            uint32_t e = 0, o = 0;
            for (int k = 0; k < 4; k++) {
                e |= (uint32_t)(uint8_t)p[2 * k] << (8 * k);
                o |= (uint32_t)(uint8_t)p[2 * k + 1] << (8 * k);
            }
            h_eo[b * 4 + u] = make_uint2(e, o);
        }
    }
    q27k::XQuant xqr = q27k::xquant_alloc((size_t)Tpad * cols);
    CUDA_CHECK(cudaMemcpy(xqr.nat, h_nat.data(), n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(xqr.scale, h_scale.data(), nb32 * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(xqr.isum, h_isum.data(), nb32 * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(xqr.eo, h_eo.data(), n / 8 * sizeof(uint2), cudaMemcpyHostToDevice));

    auto run = [&](const char* mode, const char* xg, const q27k::XQuant& q, float* y) {
        setenv("Q27_PREFILL", mode, 1);
        setenv("Q27_PF_XG", xg, 1);
        if (t.dtype == DType::Q4_G64)
            q27k::gemm_q4_T((const uint8_t*)d.data, (const __half*)d.scales, q, y, rows, cols,
                            T, 0);
        else
            q27k::gemm_q8_T((const int8_t*)d.data, (const __half*)d.scales, q, y, rows, cols,
                            T, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    run("mma", "64", xq, d_ya);
    run("dp4a", "32", xqr, d_yb);
    unsetenv("Q27_PREFILL");
    unsetenv("Q27_PF_XG");
    std::vector<float> ya((size_t)T * rows), yb((size_t)T * rows);
    CUDA_CHECK(cudaMemcpy(ya.data(), d_ya, ya.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(yb.data(), d_yb, yb.size() * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0;
    for (size_t i = 0; i < ya.size(); i++)
        maxrel = std::max(maxrel, (double)std::fabs(ya[i] - yb[i]) / (1.0 + std::fabs(ya[i])));
    char label[128];
    snprintf(label, sizeof label, "gemm MMA g64 vs exact %s", name);
    check(label, maxrel, 1e-4);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_ya));
    CUDA_CHECK(cudaFree(d_yb));
}

// P1.5: MMA flash-attention prefill vs the FA-lite reference on random data.
// Edge shapes on purpose: T=23 (partial 16-token tile), base_pos=37 (slab
// boundary misalignment). Differences = fp16 Q/P rounding + reorder (~1e-3);
// a masking or fragment bug shows as O(1).
static void test_attn_mma() {
    const int NKV = 4, GQA = 6, HD = 256, T = 23, BASE = 37;
    const int QROW = NKV * GQA * 2 * HD, OROW = NKV * GQA * HD, SEQ = BASE + T;
    std::vector<float> qh = rand_vec((size_t)T * QROW, 11);
    std::vector<float> kvh = rand_vec((size_t)SEQ * NKV * HD * 2, 12);
    std::vector<__half> kvhalf(kvh.size());
    for (size_t i = 0; i < kvh.size(); i++) kvhalf[i] = __float2half_rn(kvh[i]);

    float *d_q, *d_oa, *d_ob;
    __half *d_k, *d_v;
    CUDA_CHECK(cudaMalloc(&d_q, (size_t)T * QROW * 4));
    CUDA_CHECK(cudaMalloc(&d_oa, (size_t)T * OROW * 4));
    CUDA_CHECK(cudaMalloc(&d_ob, (size_t)T * OROW * 4));
    CUDA_CHECK(cudaMalloc(&d_k, (size_t)SEQ * NKV * HD * 2));
    CUDA_CHECK(cudaMalloc(&d_v, (size_t)SEQ * NKV * HD * 2));
    CUDA_CHECK(cudaMemcpy(d_q, qh.data(), (size_t)T * QROW * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, kvhalf.data(), (size_t)SEQ * NKV * HD * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, kvhalf.data() + (size_t)SEQ * NKV * HD,
                          (size_t)SEQ * NKV * HD * 2, cudaMemcpyHostToDevice));

    auto run = [&](const char* mode, float* out) {
        setenv("Q27_ATTN_PF", mode, 1);
        q27k::attn_prefill_T(d_q, 2 * HD, QROW, d_k, d_v, out, OROW, nullptr, BASE, 0, T,
                             NKV * GQA, NKV, HD, 1.0f / sqrtf((float)HD), 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    run("lite", d_oa);
    run("mma", d_ob);
    unsetenv("Q27_ATTN_PF");
    std::vector<float> oa((size_t)T * OROW), ob((size_t)T * OROW);
    CUDA_CHECK(cudaMemcpy(oa.data(), d_oa, oa.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ob.data(), d_ob, ob.size() * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0;
    for (size_t i = 0; i < oa.size(); i++)
        maxrel = std::max(maxrel, (double)std::fabs(oa[i] - ob[i]) / (1.0 + std::fabs(oa[i])));
    check("attn prefill MMA vs lite (T=23, base=37)", maxrel, 5e-3);
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_oa));
    CUDA_CHECK(cudaFree(d_ob));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
}

// P2: fp8 E4M3 KV-cache store kernels vs the host-side saturating conversion,
// bitwise. Seeds the input with range edges: saturation (+-500 -> +-448),
// sub-denormal (1e-4 -> 0), zero, and an exactly representable value.
static void test_kv_fp8_store() {
    const int ROW = 4 * 256, T = 8;
    std::vector<float> kf = rand_vec((size_t)T * ROW, 21), vf = rand_vec((size_t)T * ROW, 22);
    for (auto* f : {&kf, &vf}) {
        (*f)[0] = 500.f; (*f)[1] = -500.f; (*f)[2] = 1e-4f;
        (*f)[3] = 0.f;   (*f)[4] = 0.4375f; (*f)[5] = 448.f;
    }
    float *d_k, *d_v;
    uint8_t *d_kc, *d_vc;
    CUDA_CHECK(cudaMalloc(&d_k, (size_t)T * ROW * 4));
    CUDA_CHECK(cudaMalloc(&d_v, (size_t)T * ROW * 4));
    CUDA_CHECK(cudaMalloc(&d_kc, (size_t)T * ROW));
    CUDA_CHECK(cudaMalloc(&d_vc, (size_t)T * ROW));
    CUDA_CHECK(cudaMemcpy(d_k, kf.data(), (size_t)T * ROW * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, vf.data(), (size_t)T * ROW * 4, cudaMemcpyHostToDevice));
    q27k::kv_store_T(d_k, d_v, d_kc, d_vc, 0, ROW, T, 0, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> kc((size_t)T * ROW), vc((size_t)T * ROW);
    CUDA_CHECK(cudaMemcpy(kc.data(), d_kc, kc.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vc.data(), d_vc, vc.size(), cudaMemcpyDeviceToHost));
    long bad = 0;
    for (size_t i = 0; i < kc.size(); i++) {
        if (kc[i] != __nv_fp8_e4m3(kf[i]).__x) bad++;
        if (vc[i] != __nv_fp8_e4m3(vf[i]).__x) bad++;
    }
    check("fp8 kv_store_T vs host cvt (bitwise)", (double)bad, 1);
    // single-row store at a device-side position (decode path)
    int pos = 3;
    int* d_pos;
    CUDA_CHECK(cudaMalloc(&d_pos, 4));
    CUDA_CHECK(cudaMemcpy(d_pos, &pos, 4, cudaMemcpyHostToDevice));
    q27k::kv_store(d_k, d_v, d_kc, d_vc, d_pos, ROW, 0, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(kc.data(), d_kc, kc.size(), cudaMemcpyDeviceToHost));
    bad = 0;
    for (int i = 0; i < ROW; i++)
        if (kc[(size_t)pos * ROW + i] != __nv_fp8_e4m3(kf[i]).__x) bad++;
    check("fp8 kv_store row vs host cvt (bitwise)", (double)bad, 1);
    CUDA_CHECK(cudaFree(d_k)); CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_kc)); CUDA_CHECK(cudaFree(d_vc));
    CUDA_CHECK(cudaFree(d_pos));
}

// P10-A0 go/no-go: does the n-lane GEMV scale to 10 lanes, or do registers/
// BW break it? Compare t(N=10) vs 2x t(N=5) on (a) the Q8 verify head
// (248320x5120, 1.27GB -- streams past L2) and (b) Q4 ffn_gate ROTATING over
// 4 distinct layers (a single 45MB tensor sits hot in the 96MB L2 and lies
// about weight streaming -- paid-for lesson). Ratio <= ~1.3 => fusion pays.
static void test_gemv10_scaling(q27::DeviceModel& dm, const q27::Model& m) {
    const int REPS = 30;
    q27k::XQuant qs[10];
    float* ys[10];
    const int64_t cols = 5120;
    std::vector<float> x = rand_vec(cols, 91);
    float* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, cols * 4));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), cols * 4, cudaMemcpyHostToDevice));
    for (int i = 0; i < 10; i++) {
        qs[i] = q27k::xquant_alloc(cols);
        q27k::quantize_x(d_x, cols, qs[i]);
        CUDA_CHECK(cudaMalloc(&ys[i], 248320 * 4));
    }
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    auto timeit = [&](auto&& fn) {
        fn(); // warm
        CUDA_CHECK(cudaEventRecord(e0));
        for (int r = 0; r < REPS; r++) fn();
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        return (double)ms / REPS;
    };
    // (a) Q8 verify head
    const q27::DevTensor& hd = dm.upload("output.weight");
    const q27::Tensor& ht = m.get("output.weight");
    double h5 = timeit([&] {
        q27k::gemv_q8_n((const int8_t*)hd.data, (const __half*)hd.scales, qs, 5, ys,
                        ht.rows(), cols, 0);
        q27k::gemv_q8_n((const int8_t*)hd.data, (const __half*)hd.scales, qs + 5, 5, ys + 5,
                        ht.rows(), cols, 0);
    });
    double h10 = timeit([&] {
        q27k::gemv_q8_n((const int8_t*)hd.data, (const __half*)hd.scales, qs, 10, ys,
                        ht.rows(), cols, 0);
    });
    printf("  gemv10 head Q8: 2x5=%.3fms 1x10=%.3fms ratio(10 vs 2x5)=%.2f\n", h5, h10,
           h10 / h5);
    // (b) Q4 ffn_gate rotating 4 layers
    const char* names[4] = {"blk.0.ffn_gate.weight", "blk.1.ffn_gate.weight",
                            "blk.2.ffn_gate.weight", "blk.4.ffn_gate.weight"};
    const q27::DevTensor* fd[4];
    for (int i = 0; i < 4; i++) fd[i] = &dm.upload(names[i]);
    const q27::Tensor& ft = m.get(names[0]);
    int rot5 = 0, rot10 = 0;
    double f5 = timeit([&] {
        const q27::DevTensor* t = fd[rot5++ & 3];
        q27k::gemv_q4_n((const uint8_t*)t->data, (const __half*)t->scales, qs, 5, ys,
                        ft.rows(), cols, 0);
        q27k::gemv_q4_n((const uint8_t*)t->data, (const __half*)t->scales, qs + 5, 5, ys + 5,
                        ft.rows(), cols, 0);
    });
    double f10 = timeit([&] {
        const q27::DevTensor* t = fd[rot10++ & 3];
        q27k::gemv_q4_n((const uint8_t*)t->data, (const __half*)t->scales, qs, 10, ys,
                        ft.rows(), cols, 0);
    });
    printf("  gemv10 ffn Q4 (L2-rotated): 2x5=%.3fms 1x10=%.3fms ratio=%.2f\n", f5, f10,
           f10 / f5);
    // correctness: lane 7 of a fresh 10-lane HEAD run == a plain 1-lane gemv
    // (must re-run the head here -- the ffn bench above overwrote ys[])
    q27k::gemv_q8_n((const int8_t*)hd.data, (const __half*)hd.scales, qs, 10, ys, ht.rows(),
                    cols, 0);
    std::vector<float> got(128), ref(128);
    CUDA_CHECK(cudaMemcpy(got.data(), ys[7], 128 * 4, cudaMemcpyDeviceToHost));
    q27k::gemv_q8((const int8_t*)hd.data, (const __half*)hd.scales, qs[7], ys[0], ht.rows(),
                  cols, 0);
    CUDA_CHECK(cudaMemcpy(ref.data(), ys[0], 128 * 4, cudaMemcpyDeviceToHost));
    double maxd = 0;
    for (int i = 0; i < 128; i++) maxd = std::max(maxd, (double)std::fabs(got[i] - ref[i]));
    check("gemv10 lane7 == single-lane gemv (bitwise)", maxd, 1e-30);
    for (int i = 0; i < 10; i++) CUDA_CHECK(cudaFree(ys[i]));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
}

// P7: masked argmax -- argmax restricted to grammar-legal tokens via a
// resident bitmask pool + per-slot mask ids. mask id -1 (or null pool) must
// match plain argmax BITWISE (canonical gate depends on it).
static void test_masked_argmax() {
    const int N = 248320, WORDS = (N + 31) / 32;
    std::vector<float> logits = rand_vec(N, 71);
    std::vector<uint32_t> m0(WORDS, 0), m1(WORDS, 0);
    std::mt19937 rng(72);
    for (int i = 0; i < N; i++)
        if (rng() % 100 == 0) m0[i >> 5] |= 1u << (i & 31);  // ~1% legal
    int only = 123457;
    m1[only >> 5] |= 1u << (only & 31);  // exactly one legal token
    // CPU references
    auto cpu_am = [&](const std::vector<uint32_t>* m) {
        int bi = 0; float bv = -1e30f;
        for (int i = 0; i < N; i++) {
            if (m && !(((*m)[i >> 5] >> (i & 31)) & 1u)) continue;
            if (logits[i] > bv) { bv = logits[i]; bi = i; }
        }
        return bi;
    };
    float* d_x; uint32_t* d_pool; int* d_ids; int* d_out;
    unsigned long long* d_scr;
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)N * 4));
    CUDA_CHECK(cudaMalloc(&d_pool, (size_t)2 * WORDS * 4));
    CUDA_CHECK(cudaMalloc(&d_ids, 8 * 4));
    CUDA_CHECK(cudaMalloc(&d_out, 4));
    CUDA_CHECK(cudaMalloc(&d_scr, 8));
    CUDA_CHECK(cudaMemcpy(d_x, logits.data(), (size_t)N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pool, m0.data(), (size_t)WORDS * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pool + WORDS, m1.data(), (size_t)WORDS * 4,
                          cudaMemcpyHostToDevice));
    int ids[8] = {0, 1, -1, 0, 0, 0, 0, 0};
    CUDA_CHECK(cudaMemcpy(d_ids, ids, 32, cudaMemcpyHostToDevice));
    auto run = [&](int slot, const uint32_t* pool) {
        q27k::argmax_masked(d_x, N, pool, WORDS, d_ids, slot, d_out, d_scr, 0);
        int out;
        CUDA_CHECK(cudaMemcpy(&out, d_out, 4, cudaMemcpyDeviceToHost));
        return out;
    };
    int plain;
    q27k::argmax(d_x, N, d_out, d_scr, 0);
    CUDA_CHECK(cudaMemcpy(&plain, d_out, 4, cudaMemcpyDeviceToHost));
    check("masked argmax (~1% legal) vs CPU", std::fabs((double)(run(0, d_pool) - cpu_am(&m0))),
          0.5);
    check("masked argmax single-legal", std::fabs((double)(run(1, d_pool) - only)), 0.5);
    check("masked argmax id=-1 == plain", std::fabs((double)(run(2, d_pool) - plain)), 0.5);
    check("masked argmax null pool == plain", std::fabs((double)(run(0, nullptr) - plain)),
          0.5);
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_pool)); CUDA_CHECK(cudaFree(d_ids));
    CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_scr));
}

// P2: every E4M3 value is exactly representable in fp16, so the fp8 kernels
// reading an fp8 cache must match the fp16 kernels reading the host-dequantized
// cache BITWISE -- this isolates the load-conversion plumbing with no tolerance.
static void test_attn_fp8() {
    const int NKV = 4, GQA = 6, HD = 256, T = 23, BASE = 37, SEQ = BASE + T;
    const int QROW = NKV * GQA * 2 * HD, OROW = NKV * GQA * HD, ROW = NKV * HD;
    std::vector<float> qh = rand_vec((size_t)T * QROW, 31);
    std::vector<float> kf = rand_vec((size_t)SEQ * ROW, 32), vf = rand_vec((size_t)SEQ * ROW, 33);
    std::vector<uint8_t> k8(kf.size()), v8(vf.size());
    std::vector<__half> kh(kf.size()), vh(vf.size());
    for (size_t i = 0; i < kf.size(); i++) {
        __nv_fp8_e4m3 a(kf[i]), b(vf[i]);
        k8[i] = a.__x; v8[i] = b.__x;
        kh[i] = __float2half_rn(float(a)); vh[i] = __float2half_rn(float(b));
    }
    float *d_q, *d_oa, *d_ob;
    uint8_t *d_k8, *d_v8;
    __half *d_kh, *d_vh;
    CUDA_CHECK(cudaMalloc(&d_q, (size_t)T * QROW * 4));
    CUDA_CHECK(cudaMalloc(&d_oa, (size_t)T * OROW * 4));
    CUDA_CHECK(cudaMalloc(&d_ob, (size_t)T * OROW * 4));
    CUDA_CHECK(cudaMalloc(&d_k8, k8.size()));
    CUDA_CHECK(cudaMalloc(&d_v8, v8.size()));
    CUDA_CHECK(cudaMalloc(&d_kh, kh.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_vh, vh.size() * 2));
    CUDA_CHECK(cudaMemcpy(d_q, qh.data(), (size_t)T * QROW * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k8, k8.data(), k8.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v8, v8.data(), v8.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kh, kh.data(), kh.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vh, vh.data(), vh.size() * 2, cudaMemcpyHostToDevice));

    std::vector<float> oa((size_t)T * OROW), ob((size_t)T * OROW);
    auto maxd = [&]() {
        CUDA_CHECK(cudaMemcpy(oa.data(), d_oa, oa.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ob.data(), d_ob, ob.size() * 4, cudaMemcpyDeviceToHost));
        double m = 0;
        for (size_t i = 0; i < oa.size(); i++)
            m = std::max(m, (double)std::fabs(oa[i] - ob[i]));
        return m;
    };
    const float scale = 1.0f / sqrtf((float)HD);
    for (const char* mode : {"lite", "mma"}) {
        setenv("Q27_ATTN_PF", mode, 1);
        q27k::attn_prefill_T(d_q, 2 * HD, QROW, d_k8, d_v8, d_oa, OROW, nullptr, BASE, 0, T,
                             NKV * GQA, NKV, HD, scale, 0, true);
        q27k::attn_prefill_T(d_q, 2 * HD, QROW, d_kh, d_vh, d_ob, OROW, nullptr, BASE, 0, T,
                             NKV * GQA, NKV, HD, scale, 0, false);
        CUDA_CHECK(cudaDeviceSynchronize());
        char label[80];
        snprintf(label, sizeof label, "fp8 attn prefill %s == fp16(deq) (bitwise)", mode);
        check(label, maxd(), 1e-30);
    }
    unsetenv("Q27_ATTN_PF");
    // flash-decode: one token at position SEQ-1 (reads rows [0, SEQ))
    int pos = SEQ - 1;
    int* d_pos;
    float* d_scr;
    CUDA_CHECK(cudaMalloc(&d_pos, 4));
    CUDA_CHECK(cudaMemcpy(d_pos, &pos, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_scr, (size_t)NKV * GQA * q27k::FD_NS * q27k::FD_ST * 4));
    q27k::attn_decode(d_q, 2 * HD, d_k8, d_v8, d_oa, d_scr, d_pos, SEQ, NKV * GQA, NKV, HD,
                      scale, 0, true);
    q27k::attn_decode(d_q, 2 * HD, d_kh, d_vh, d_ob, d_scr, d_pos, SEQ, NKV * GQA, NKV, HD,
                      scale, 0, false);
    CUDA_CHECK(cudaDeviceSynchronize());
    oa.resize(OROW); ob.resize(OROW);
    check("fp8 attn decode == fp16(deq) (bitwise)", maxd(), 1e-30);
    CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_oa)); CUDA_CHECK(cudaFree(d_ob));
    CUDA_CHECK(cudaFree(d_k8)); CUDA_CHECK(cudaFree(d_v8));
    CUDA_CHECK(cudaFree(d_kh)); CUDA_CHECK(cudaFree(d_vh));
    CUDA_CHECK(cudaFree(d_pos)); CUDA_CHECK(cudaFree(d_scr));
}

// P4: split-position MMA prefill vs the exact single-split path at a deep
// base_pos (splits change fp summation grouping -> tolerance like MMA-vs-lite),
// plus the fp8 == fp16(dequantized-cache) bitwise identity under forced splits
// (both sides split identically, so exactness must survive).
static void test_attn_split() {
    const int NKV = 4, GQA = 6, HD = 256, T = 37, BASE = 9000, SEQ = BASE + T;
    const int QROW = NKV * GQA * 2 * HD, OROW = NKV * GQA * HD, ROW = NKV * HD;
    std::vector<float> qh = rand_vec((size_t)T * QROW, 41);
    std::vector<float> kvh = rand_vec((size_t)SEQ * ROW * 2, 42);
    std::vector<__half> kvhalf(kvh.size());
    for (size_t i = 0; i < kvh.size(); i++) kvhalf[i] = __float2half_rn(kvh[i]);
    float *d_q, *d_oa, *d_ob, *d_part;
    __half *d_k, *d_v;
    const int TROWS = (T + 15) / 16 * 16;
    CUDA_CHECK(cudaMalloc(&d_q, (size_t)T * QROW * 4));
    CUDA_CHECK(cudaMalloc(&d_oa, (size_t)T * OROW * 4));
    CUDA_CHECK(cudaMalloc(&d_ob, (size_t)T * OROW * 4));
    CUDA_CHECK(cudaMalloc(&d_part,
                          (size_t)NKV * GQA * TROWS * q27k::PF_SPLIT_MAX * 258 * 4));
    CUDA_CHECK(cudaMalloc(&d_k, (size_t)SEQ * ROW * 2));
    CUDA_CHECK(cudaMalloc(&d_v, (size_t)SEQ * ROW * 2));
    CUDA_CHECK(cudaMemcpy(d_q, qh.data(), (size_t)T * QROW * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, kvhalf.data(), (size_t)SEQ * ROW * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, kvhalf.data() + (size_t)SEQ * ROW, (size_t)SEQ * ROW * 2,
                          cudaMemcpyHostToDevice));
    const float scale = 1.0f / sqrtf((float)HD);
    // forced 5-way split (odd count exercises empty tail slices) vs no split
    setenv("Q27_PF_SPLIT", "5", 1);
    q27k::attn_prefill_T(d_q, 2 * HD, QROW, d_k, d_v, d_oa, OROW, d_part, BASE, 0, T,
                         NKV * GQA, NKV, HD, scale, 0);
    setenv("Q27_PF_SPLIT", "1", 1);
    q27k::attn_prefill_T(d_q, 2 * HD, QROW, d_k, d_v, d_ob, OROW, d_part, BASE, 0, T,
                         NKV * GQA, NKV, HD, scale, 0);
    unsetenv("Q27_PF_SPLIT");
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> oa((size_t)T * OROW), ob((size_t)T * OROW);
    CUDA_CHECK(cudaMemcpy(oa.data(), d_oa, oa.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ob.data(), d_ob, ob.size() * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0;
    for (size_t i = 0; i < oa.size(); i++)
        maxrel = std::max(maxrel, (double)std::fabs(oa[i] - ob[i]) / (1.0 + std::fabs(oa[i])));
    check("attn prefill split=5 vs 1 (T=37, base=9000)", maxrel, 5e-3);
    CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_oa)); CUDA_CHECK(cudaFree(d_ob));
    CUDA_CHECK(cudaFree(d_part)); CUDA_CHECK(cudaFree(d_k)); CUDA_CHECK(cudaFree(d_v));
}

// q/k blocks of conv l2-normalized per head, matching the engine's input
// contract (l2norm_heads_T runs before delta_scan_T; with ||k|| ~ 11 instead
// of 1 the delta update is expansive and chaotically amplifies
// reduction-reorder noise into false FAILs). Shared by all delta tests.
static void l2norm_qk_host(std::vector<float>& conv, int T, int CH, int SK) {
    for (int t = 0; t < T; t++)
        for (int hh = 0; hh < 32; hh++) {
            float* p = conv.data() + (size_t)t * CH + hh * SK;
            double n2 = 0;
            for (int i = 0; i < SK; i++) n2 += (double)p[i] * p[i];
            float inv = 1.f / std::sqrt((float)n2 + 1e-6f);
            for (int i = 0; i < SK; i++) p[i] *= inv;
        }
}

// P6: column-split delta scan vs the exact one-block-per-head path. S columns
// are independent (pred_j, dj, the rank-1 update and o_j touch only column j),
// so the split only reorders the row reductions (4x32-serial -> NTILE x RPT)
// -> tolerance gate like P4's attention split. Q27_DS_SPLIT=1 must route to
// the untouched legacy kernel.
static void test_delta_wy() {
    const int NH = 48, SK = 128, CH = 10240, T = 256;
    const size_t SN = (size_t)NH * SK * SK, ON = (size_t)T * NH * SK;
    std::vector<float> S0 = rand_vec(SN, 61), conv = rand_vec((size_t)T * CH, 62);
    std::vector<float> g = rand_vec((size_t)T * NH, 63), beta = rand_vec((size_t)T * NH, 64);
    for (auto& x : beta) x = 1.f / (1.f + std::exp(-x));
    l2norm_qk_host(conv, T, CH, SK);
    float *d_S, *d_conv, *d_g, *d_beta, *d_o;
    CUDA_CHECK(cudaMalloc(&d_S, SN * 4));
    CUDA_CHECK(cudaMalloc(&d_conv, conv.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_g, g.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_beta, beta.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_o, ON * 4));
    CUDA_CHECK(cudaMemcpy(d_conv, conv.data(), conv.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice));

    std::vector<float> Sa(SN), oa(ON), Sb(SN), ob(ON);
    q27k::WyScratch ws; // wy-leg panels, freed below
    // mild decay (engine-typical) and strong decay (lambda underflows f32 over
    // a 64-chunk -- exercises the log-space ratio path; absolute-tol check
    // because outputs themselves shrink toward 0 there)
    for (float dscale : {0.1f, 2.5f}) {
        std::vector<float> gs = g;
        for (auto& x : gs) x = -std::fabs(x) * dscale;
        CUDA_CHECK(cudaMemcpy(d_g, gs.data(), gs.size() * 4, cudaMemcpyHostToDevice));
        // T=1 (no recurrence: index bugs undamped), 64 (single chunk),
        // 200 (ragged tail), 256 (full multi-chunk)
        for (int Tn : {1, 64, 200, 256}) {
            setenv("Q27_DS_MODE", "seq", 1); // wy is default since 2026-07-04
            setenv("Q27_DS_SPLIT", "1", 1);
            CUDA_CHECK(cudaMemcpy(d_S, S0.data(), SN * 4, cudaMemcpyHostToDevice));
            q27k::delta_scan_T(d_S, d_conv, d_g, d_beta, d_o, Tn, 0, &ws);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(Sa.data(), d_S, SN * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(oa.data(), d_o, (size_t)Tn * NH * SK * 4,
                                  cudaMemcpyDeviceToHost));
            setenv("Q27_DS_MODE", "wy", 1);
            CUDA_CHECK(cudaMemcpy(d_S, S0.data(), SN * 4, cudaMemcpyHostToDevice));
            q27k::delta_scan_T(d_S, d_conv, d_g, d_beta, d_o, Tn, 0, &ws);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(Sb.data(), d_S, SN * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ob.data(), d_o, (size_t)Tn * NH * SK * 4,
                                  cudaMemcpyDeviceToHost));
            unsetenv("Q27_DS_MODE");
            double mo = 0, ms = 0;
            for (size_t i = 0; i < (size_t)Tn * NH * SK; i++)
                mo = std::max(mo, (double)std::fabs(oa[i] - ob[i]) / (1.0 + std::fabs(oa[i])));
            for (size_t i = 0; i < SN; i++)
                ms = std::max(ms, (double)std::fabs(Sa[i] - Sb[i]) / (1.0 + std::fabs(Sa[i])));
            char label[64];
            double tol = Tn == 1 ? 1e-5 : 2e-3;
            snprintf(label, sizeof label, "delta wy vs seq (o, T=%d, d=%.1f)", Tn, dscale);
            check(label, mo, tol);
            snprintf(label, sizeof label, "delta wy vs seq (S, T=%d, d=%.1f)", Tn, dscale);
            check(label, ms, tol);
        }
    }
    cudaFree(ws.kkt); cudaFree(ws.qkt); // cudaFree(nullptr) is a no-op
    cudaFree(d_S); cudaFree(d_conv); cudaFree(d_g); cudaFree(d_beta); cudaFree(d_o);
}

static void test_delta_split() {
    const int NH = 48, SK = 128, CH = 10240, T = 64;
    const size_t SN = (size_t)NH * SK * SK, ON = (size_t)T * NH * SK;
    std::vector<float> S0 = rand_vec(SN, 51), conv = rand_vec((size_t)T * CH, 52);
    std::vector<float> g = rand_vec((size_t)T * NH, 53), beta = rand_vec((size_t)T * NH, 54);
    // keep the recurrence in its real regime, matching the engine's input
    // contract: decay = exp(g) <= 1, beta in (0,1), q/k l2-normalized
    // (rationale at l2norm_qk_host)
    for (auto& x : g) x = -std::fabs(x) * 0.1f;
    for (auto& x : beta) x = 1.f / (1.f + std::exp(-x));
    l2norm_qk_host(conv, T, CH, SK);
    float *d_S, *d_conv, *d_g, *d_beta, *d_o;
    CUDA_CHECK(cudaMalloc(&d_S, SN * 4));
    CUDA_CHECK(cudaMalloc(&d_conv, conv.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_g, g.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_beta, beta.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_o, ON * 4));
    CUDA_CHECK(cudaMemcpy(d_conv, conv.data(), conv.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice));

    // this whole test exercises the sequential/split path; wy is the
    // delta_scan_T default since 2026-07-04, so pin seq for the duration
    setenv("Q27_DS_MODE", "seq", 1);
    setenv("Q27_DS_SPLIT", "1", 1);
    check("ds nsplit env=1", std::fabs((double)q27k::delta_scan_nsplit(T) - 1), 0.5);
    setenv("Q27_DS_SPLIT", "4", 1);
    check("ds nsplit env=4", std::fabs((double)q27k::delta_scan_nsplit(T) - 4), 0.5);
    unsetenv("Q27_DS_SPLIT");
    int cs_auto = q27k::delta_scan_nsplit(T);
    check("ds nsplit auto valid", (cs_auto == 1 || cs_auto == 2 || cs_auto == 4 || cs_auto == 8)
                                      ? 0.0 : 1.0, 0.5);

    std::vector<float> Sa(SN), oa(ON), Sb(SN), ob(ON);
    q27k::WyScratch ws; // unused on the seq-pinned path, signature requires it
    // Tn=1 has no recurrence: any indexing/race bug shows undamped at ~0
    // tolerance; Tn=64 then bounds the compounded reorder noise.
    for (int Tn : {1, T}) {
        setenv("Q27_DS_SPLIT", "1", 1);
        CUDA_CHECK(cudaMemcpy(d_S, S0.data(), SN * 4, cudaMemcpyHostToDevice));
        q27k::delta_scan_T(d_S, d_conv, d_g, d_beta, d_o, Tn, 0, &ws);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(Sa.data(), d_S, SN * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(oa.data(), d_o, (size_t)Tn * NH * SK * 4, cudaMemcpyDeviceToHost));
        for (int cs : {2, 4, 8}) {
            char v[4], label[64];
            snprintf(v, sizeof v, "%d", cs);
            setenv("Q27_DS_SPLIT", v, 1);
            CUDA_CHECK(cudaMemcpy(d_S, S0.data(), SN * 4, cudaMemcpyHostToDevice));
            q27k::delta_scan_T(d_S, d_conv, d_g, d_beta, d_o, Tn, 0, &ws);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(Sb.data(), d_S, SN * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ob.data(), d_o, (size_t)Tn * NH * SK * 4,
                                  cudaMemcpyDeviceToHost));
            double mo = 0, ms = 0;
            for (size_t i = 0; i < (size_t)Tn * NH * SK; i++)
                mo = std::max(mo, (double)std::fabs(oa[i] - ob[i]) / (1.0 + std::fabs(oa[i])));
            for (size_t i = 0; i < SN; i++)
                ms = std::max(ms, (double)std::fabs(Sa[i] - Sb[i]) / (1.0 + std::fabs(Sa[i])));
            double tol = Tn == 1 ? 1e-5 : 1e-3;
            snprintf(label, sizeof label, "delta scan split=%d vs 1 (o, T=%d)", cs, Tn);
            check(label, mo, tol);
            snprintf(label, sizeof label, "delta scan split=%d vs 1 (S final, T=%d)", cs, Tn);
            check(label, ms, tol);
        }
    }
    // timing at the engine's chunk size (PF_T=256), informational only
    {
        const int TB = 256, REP = 50;
        std::vector<float> cB = rand_vec((size_t)TB * CH, 55), gB = rand_vec((size_t)TB * NH, 56);
        std::vector<float> bB = rand_vec((size_t)TB * NH, 57);
        for (auto& x : gB) x = -std::fabs(x) * 0.1f;
        for (auto& x : bB) x = 1.f / (1.f + std::exp(-x));
        float *d_cB, *d_gB, *d_bB, *d_oB;
        CUDA_CHECK(cudaMalloc(&d_cB, cB.size() * 4));
        CUDA_CHECK(cudaMalloc(&d_gB, gB.size() * 4));
        CUDA_CHECK(cudaMalloc(&d_bB, bB.size() * 4));
        CUDA_CHECK(cudaMalloc(&d_oB, (size_t)TB * NH * SK * 4));
        CUDA_CHECK(cudaMemcpy(d_cB, cB.data(), cB.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gB, gB.data(), gB.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bB, bB.data(), bB.size() * 4, cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        for (int cs : {1, 2, 4, 8}) {
            char v[4];
            snprintf(v, sizeof v, "%d", cs);
            setenv("Q27_DS_SPLIT", v, 1);
            CUDA_CHECK(cudaMemcpy(d_S, S0.data(), SN * 4, cudaMemcpyHostToDevice));
            for (int w = 0; w < 3; w++)
                q27k::delta_scan_T(d_S, d_cB, d_gB, d_bB, d_oB, TB, 0, &ws);
            CUDA_CHECK(cudaEventRecord(e0));
            for (int r = 0; r < REP; r++)
                q27k::delta_scan_T(d_S, d_cB, d_gB, d_bB, d_oB, TB, 0, &ws);
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            printf("  delta scan T=256 split=%d: %.0f us/launch\n", cs, ms * 1000.f / REP);
        }
        CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaFree(d_cB)); CUDA_CHECK(cudaFree(d_gB));
        CUDA_CHECK(cudaFree(d_bB)); CUDA_CHECK(cudaFree(d_oB));
    }
    unsetenv("Q27_DS_SPLIT");
    unsetenv("Q27_DS_MODE");
    CUDA_CHECK(cudaFree(d_S)); CUDA_CHECK(cudaFree(d_conv)); CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_beta)); CUDA_CHECK(cudaFree(d_o));
}

// R1b prerequisite: multi-slot engines run batched prefill on their own
// streams. The wy KKt/QKt panels are written by k_delta_wy_kk and read by
// k_delta_wy with no cross-stream ordering, so they must be per-engine
// state: a shared set races once two engines' chunks are in flight, and the
// lazy regrow cudaFrees panels the other stream's queued kernels still
// reference. Contract: two contexts, chained scans interleaved across two
// streams with no host sync, match their isolated serial runs BITWISE (same
// kernels, same data, same launch config -- only scratch sharing can differ).
static void test_wy_stream_isolation() {
    const int NH = 48, SK = 128, CH = 10240, ITERS = 48;
    const size_t SN = (size_t)NH * SK * SK;
    setenv("Q27_DS_MODE", "wy", 1); // the scratch under test is wy-only
    struct Ctx {
        int T;
        float *d_S, *d_conv, *d_g, *d_beta, *d_o;
        std::vector<float> S0, S_ref, o_ref;
        q27k::WyScratch ws; // per-context, the per-engine model
    };
    Ctx ctx[2] = {{512}, {1024}}; // aggregate: pointers null, ws default
    for (int c = 0; c < 2; c++) {
        Ctx& x = ctx[c];
        const size_t ON = (size_t)x.T * NH * SK;
        x.S0 = rand_vec(SN, 71 + 10 * c);
        std::vector<float> conv = rand_vec((size_t)x.T * CH, 72 + 10 * c);
        std::vector<float> g = rand_vec((size_t)x.T * NH, 73 + 10 * c);
        std::vector<float> beta = rand_vec((size_t)x.T * NH, 74 + 10 * c);
        // input contract: decay <= 1, beta in (0,1), q/k l2-normalized
        for (auto& v : g) v = -std::fabs(v) * 0.1f;
        for (auto& v : beta) v = 1.f / (1.f + std::exp(-v));
        l2norm_qk_host(conv, x.T, CH, SK);
        CUDA_CHECK(cudaMalloc(&x.d_S, SN * 4));
        CUDA_CHECK(cudaMalloc(&x.d_conv, conv.size() * 4));
        CUDA_CHECK(cudaMalloc(&x.d_g, g.size() * 4));
        CUDA_CHECK(cudaMalloc(&x.d_beta, beta.size() * 4));
        CUDA_CHECK(cudaMalloc(&x.d_o, ON * 4));
        CUDA_CHECK(cudaMemcpy(x.d_conv, conv.data(), conv.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(x.d_g, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(x.d_beta, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice));
        // isolated serial reference: ITERS chained scans, fully synced
        CUDA_CHECK(cudaMemcpy(x.d_S, x.S0.data(), SN * 4, cudaMemcpyHostToDevice));
        for (int i = 0; i < ITERS; i++)
            q27k::delta_scan_T(x.d_S, x.d_conv, x.d_g, x.d_beta, x.d_o, x.T, 0, &x.ws);
        CUDA_CHECK(cudaDeviceSynchronize());
        x.S_ref.resize(SN);
        x.o_ref.resize(ON);
        CUDA_CHECK(cudaMemcpy(x.S_ref.data(), x.d_S, SN * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(x.o_ref.data(), x.d_o, ON * 4, cudaMemcpyDeviceToHost));
    }
    // interleaved: both contexts chained on their own streams, no host sync
    cudaStream_t st[2];
    CUDA_CHECK(cudaStreamCreate(&st[0]));
    CUDA_CHECK(cudaStreamCreate(&st[1]));
    for (int c = 0; c < 2; c++)
        CUDA_CHECK(cudaMemcpyAsync(ctx[c].d_S, ctx[c].S0.data(), SN * 4,
                                   cudaMemcpyHostToDevice, st[c]));
    for (int i = 0; i < ITERS; i++)
        for (int c = 0; c < 2; c++)
            q27k::delta_scan_T(ctx[c].d_S, ctx[c].d_conv, ctx[c].d_g, ctx[c].d_beta,
                               ctx[c].d_o, ctx[c].T, st[c], &ctx[c].ws);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int c = 0; c < 2; c++) {
        Ctx& x = ctx[c];
        const size_t ON = (size_t)x.T * NH * SK;
        std::vector<float> S(SN), o(ON);
        CUDA_CHECK(cudaMemcpy(S.data(), x.d_S, SN * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(o.data(), x.d_o, ON * 4, cudaMemcpyDeviceToHost));
        const int bad = (memcmp(S.data(), x.S_ref.data(), SN * 4) != 0) +
                        (memcmp(o.data(), x.o_ref.data(), ON * 4) != 0);
        char label[64];
        snprintf(label, sizeof label, "wy stream isolation (ctx %c, T=%d)", 'A' + c, x.T);
        check(label, (double)bad, 0.5);
        CUDA_CHECK(cudaFree(x.d_S)); CUDA_CHECK(cudaFree(x.d_conv));
        CUDA_CHECK(cudaFree(x.d_g)); CUDA_CHECK(cudaFree(x.d_beta));
        CUDA_CHECK(cudaFree(x.d_o));
        CUDA_CHECK(cudaFree(x.ws.kkt)); CUDA_CHECK(cudaFree(x.ws.qkt));
    }
    CUDA_CHECK(cudaStreamDestroy(st[0]));
    CUDA_CHECK(cudaStreamDestroy(st[1]));
    unsetenv("Q27_DS_MODE");
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27";
    q27::Model m = q27::Model::open(path);
    q27::DeviceModel dm(m);

    printf("q27 kernel self-consistency tests (%s)\n", path);
    test_dequant(dm, m, "blk.0.ffn_gate.weight");   // Q4
    test_dequant(dm, m, "blk.3.attn_k.weight");     // Q8 (v1.1 policy)
    test_gemv(dm, m, "blk.0.ffn_gate.weight");      // Q4 GEMV
    test_gemv(dm, m, "blk.3.attn_k.weight");        // Q8 GEMV
    test_gemv(dm, m, "blk.0.ssm_alpha.weight");     // F16 GEMV
    test_gemv_batch(dm, m, "blk.0.ffn_gate.weight");
    test_gemv_batch(dm, m, "blk.3.attn_k.weight");
    test_gemm_mma(dm, m, "blk.0.attn_qkv.weight");  // Q4 10240x5120
    test_gemm_mma(dm, m, "blk.0.ffn_down.weight");  // Q4 5120x17408 (K tail shapes)
    test_gemm_mma(dm, m, "blk.3.attn_k.weight");    // Q8 1024x5120
    test_gemm_mma(dm, m, "output.weight");          // Q8 248320x5120 (big-rows)
    test_gemm_mma_g64(dm, m, "blk.0.attn_qkv.weight");
    test_gemm_mma_g64(dm, m, "blk.0.ffn_down.weight");
    test_gemm_mma_g64(dm, m, "blk.3.attn_k.weight");
    test_gemm_mma_g64(dm, m, "output.weight");
    test_attn_mma();
    test_attn_split();
    test_delta_split();
    test_delta_wy();
    test_wy_stream_isolation();
    test_masked_argmax();
    test_gemv10_scaling(dm, m);
    test_kv_fp8_store();
    test_attn_fp8();
    test_rmsnorm(m);
    test_silu_mul();
    test_embed(dm, m);
    printf("resident: %.2f GB\n%s\n", dm.bytes_resident() / 1e9, g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
