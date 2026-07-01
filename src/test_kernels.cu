// Self-consistency tests: GPU kernels vs CPU reference over the same q27 data.
// No external ground truth needed; validates layout, nibble order, scales, GEMV.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cuda_common.h"
#include "device_model.h"
#include "kernels.cuh"
#include "loader.h"

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
        q27k::gemv_q4_n((const uint8_t*)d.data, (const __half*)d.scales, xqs, NB, d_yb, rows, cols);
        for (int n = 0; n < NB; n++) {
            q27k::gemv_q4((const uint8_t*)d.data, (const __half*)d.scales, xqs[n], d_y1, rows, cols);
            std::vector<float> yb(rows), y1(rows);
            CUDA_CHECK(cudaMemcpy(yb.data(), d_yb + (size_t)n * rows, rows * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(y1.data(), d_y1, rows * 4, cudaMemcpyDeviceToHost));
            for (int64_t r = 0; r < rows; r++)
                maxd = std::max(maxd, (double)std::fabs(yb[r] - y1[r]));
        }
    } else {
        q27k::gemv_q8_n((const int8_t*)d.data, (const __half*)d.scales, xqs, NB, d_yb, rows, cols);
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
    test_rmsnorm(m);
    test_silu_mul();
    test_embed(dm, m);
    printf("resident: %.2f GB\n%s\n", dm.bytes_resident() / 1e9, g_fail ? "FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
