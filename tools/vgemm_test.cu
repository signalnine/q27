// vgemm_test -- P1 gates 3 and 4 for the flat-in-W verify weight path.
//
// GATE 3 (correctness): every decode weight shape x {Q4_G64, Q8_G128} x width
// 2..16, checked on ALL W_PLUMB lanes -- the old mma16_bench only ever validated
// lane 0, which is exactly how a lane-aliasing bug would hide (the 8->12 widening
// lost k_quantize_x3 that way). Reference is the shipped gemv_q4_n/gemv_q8_n.
// Also asserts: lanes >= T are NEVER written (NaN-poisoned), and the kernel is
// BITWISE-STABLE across repeats (the determinism claim -- the whole reason we
// refused an atomicAdd epilogue).
//
// GATE 4 (occupancy): ptxas/driver-reported regs, spill and CTA/SM for all four
// instantiations. FAILS LOUD. There are zero spare registers at 4 CTAs/SM
// (64 regs x 256 thr x 4 = 65536 = the whole SM budget); one extra live value
// costs a CTA tier and ~20%. This project has now shipped a false occupancy
// assumption twice -- this gate is the answer to that.
//
// Usage: vgemm_test model.q27
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/device_model.h"
#include "../src/kernels.cuh"
#include "../src/loader.h"
#include "../src/vgemm.cuh"

static int fails = 0;
#define CHECK(cond, ...)                                                                 \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            printf("  FAIL: ");                                                          \
            printf(__VA_ARGS__);                                                         \
            printf("\n");                                                                \
            fails++;                                                                     \
        }                                                                                \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.q27\n", argv[0]); return 1; }
    q27::Model m = q27::Model::open(argv[1]);
    q27::DeviceModel dm(m);

    // ---- GATE 4: occupancy, first, because it is the cheapest to fail ----
    printf("== gate 4: register / spill / CTA-per-SM budget (MR=%d) ==\n", q27k::VG_MR);
    {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, dev);
        printf("  device: %s sm_%d%d  %d SMs  %d regs/SM  %zu B smem/SM\n", p.name, p.major,
               p.minor, p.multiProcessorCount, p.regsPerMultiprocessor,
               p.sharedMemPerMultiprocessor);
        // The four instantiations the round actually launches.
        struct { const char* tag; bool q4; int mode; } inst[4] = {
            {"Q4 MODE=0 (z==1, head)", true, 0},
            {"Q4 MODE=1 (z>1, partials)", true, 1},
            {"Q8 MODE=0", false, 0},
            {"Q8 MODE=1", false, 1},
        };
        for (auto& I : inst) {
            q27k::VgemmAttrs a = q27k::vgemm_attrs(I.q4, I.mode);
            printf("  %-28s regs=%3d  stack=%zu  smem=%5zu  CTA/SM=%d\n", I.tag, a.regs,
                   (size_t)a.local_bytes, a.smem, a.cta_per_sm);
            CHECK(a.regs <= 64, "%s: %d regs > 64 (would drop below 4 CTA/SM)", I.tag, a.regs);
            CHECK(a.local_bytes == 0, "%s: %zu B of local/spill (must be 0)", I.tag,
                  (size_t)a.local_bytes);
            CHECK(a.cta_per_sm >= 4, "%s: only %d CTA/SM (need 4)", I.tag, a.cta_per_sm);
        }
    }

    // ---- GATE 3: numerics vs the shipped GEMV, all lanes, all widths ----
    // Discover the round's DISTINCT weight shapes from the model itself -- layer
    // 0 is GDN and layer 3 is attention on this checkpoint, but do not hardcode
    // that: find the first layer carrying each tensor. (Hardcoding blk.0.attn_q
    // threw "missing tensor" on the first run.)
    std::vector<std::string> shapes;
    auto first_with = [&](const char* leaf) {
        for (int il = 0; il < 80; il++) {
            char b[96];
            snprintf(b, sizeof b, "blk.%d.%s", il, leaf);
            if (m.find(b)) { shapes.push_back(b); return; }
        }
        printf("  (no layer carries %s -- skipped)\n", leaf);
    };
    for (const char* leaf : {"ffn_gate.weight", "ffn_down.weight", "ffn_up.weight",
                             "attn_qkv.weight", "attn_gate.weight", "ssm_out.weight",
                             "attn_q.weight", "attn_output.weight", "attn_k.weight"})
        first_with(leaf);
    if (m.find("output_q4.weight")) shapes.push_back("output_q4.weight");
    // Q27_VGEMM_FAST: racecheck/initcheck are ~50x slower than memcheck, so the
    // sanitizer gate runs a reduced but REPRESENTATIVE set -- both z paths (z=1
    // MODE 0 head, z>1 MODE 1 partials + reduce) and both dtypes.
    if (getenv("Q27_VGEMM_FAST")) {
        std::vector<std::string> keep;
        for (auto& n : shapes)
            if (n.find("ffn_down") != std::string::npos ||   // Q4, z=8, MODE 1
                n.find("ssm_out") != std::string::npos ||    // Q8, z=8, MODE 1
                n.find("output_q4") != std::string::npos)    // Q4, z=1, MODE 0
                keep.push_back(n);
        shapes = keep;
    }
    const int NS = (int)shapes.size();
    const int TSTEP = getenv("Q27_VGEMM_FAST") ? 5 : 1;

    // lanes: distinct activations per lane, so a lane mix-up cannot pass
    const int64_t MAXC = 17408;
    q27k::XQuant xq[W_PLUMB];
    const int8_t* h_nat[W_PLUMB];
    float* d_y_g[W_PLUMB];  // gemv reference outputs
    float* d_y_v[W_PLUMB];  // vgemm outputs
    const int64_t MAXR = 248320;
    std::vector<float> hx(MAXC);
    float* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, MAXC * 4));
    for (int t = 0; t < W_PLUMB; t++) {
        xq[t] = q27k::xquant_alloc(MAXC);
        for (int64_t i = 0; i < MAXC; i++)
            hx[i] = sinf(0.017f * (float)i + 1.31f * (float)t) * (0.5f + 0.1f * (float)t);
        CUDA_CHECK(cudaMemcpy(d_x, hx.data(), MAXC * 4, cudaMemcpyHostToDevice));
        q27k::quantize_x(d_x, MAXC, xq[t]);
        CUDA_CHECK(cudaMalloc(&d_y_g[t], MAXR * 4));
        CUDA_CHECK(cudaMalloc(&d_y_v[t], MAXR * 4));
        (void)h_nat;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // workspace, sized exactly as the engine will size it
    std::vector<const q27::DevTensor*> wl;
    for (int s = 0; s < NS; s++) wl.push_back(&dm.upload(shapes[s]));
    size_t wsb = q27k::vgemm_ws_bytes(wl.data(), (int)wl.size());
    float* d_ws;
    CUDA_CHECK(cudaMalloc(&d_ws, wsb));
    printf("\n== gate 3: numerics vs gemv, ALL %d lanes, widths 2..%d ==\n", W_PLUMB, W_PLUMB);
    printf("  workspace %.2f MB (max over the shape list)\n", wsb / 1e6);

    std::vector<float> hg(MAXR), hv(MAXR), hv2(MAXR);
    for (int s = 0; s < NS; s++) {
        const q27::DevTensor& w = *wl[s];
        const bool q4 = (w.dtype == q27::DType::Q4_G64);
        int64_t rows = w.rows, cols = w.cols;
        double worst = 0;
        int zused = q27k::vgemm_z(rows, cols);
        for (int T = 2; T <= W_PLUMB; T += TSTEP) {
            // reference
            q27k::XQuant qs[W_PLUMB];
            float* ys[W_PLUMB];
            for (int t = 0; t < W_PLUMB; t++) { qs[t] = xq[t]; ys[t] = d_y_g[t]; }
            if (q4)
                q27k::gemv_q4_n((const uint8_t*)w.data, (const __half*)w.scales, qs, T, ys, rows,
                                cols, 0);
            else
                q27k::gemv_q8_n((const int8_t*)w.data, (const __half*)w.scales, qs, T, ys, rows,
                                cols, 0);
            // vgemm; poison every lane output first so an unwritten slot is caught
            q27k::XLanes X{};
            q27k::YLanes Y{};
            for (int t = 0; t < W_PLUMB; t++) {
                X.nat[t] = xq[t].nat;
                X.xs[t] = xq[t].scale;
                Y.y[t] = d_y_v[t];
                float qn = nanf("");
                std::vector<float> poison(rows, qn);
                CUDA_CHECK(cudaMemcpy(d_y_v[t], poison.data(), rows * 4, cudaMemcpyHostToDevice));
            }
            bool ok = q27k::vgemm_verify(w, X, Y, d_ws, T, 0);
            CHECK(ok, "%s T=%d: vgemm_verify refused an eligible shape", shapes[s].c_str(), T);
            CUDA_CHECK(cudaDeviceSynchronize());
            for (int t = 0; t < W_PLUMB; t++) {
                CUDA_CHECK(cudaMemcpy(hv.data(), d_y_v[t], rows * 4, cudaMemcpyDeviceToHost));
                if (t >= T) { // lanes past the live width must be UNTOUCHED (still NaN)
                    int written = 0;
                    for (int64_t r = 0; r < rows; r++)
                        if (!isnan(hv[r])) written++;
                    CHECK(written == 0, "%s T=%d: lane %d (>= T) had %d values written",
                          shapes[s].c_str(), T, t, written);
                    continue;
                }
                CUDA_CHECK(cudaMemcpy(hg.data(), d_y_g[t], rows * 4, cudaMemcpyDeviceToHost));
                double num = 0, den = 0;
                for (int64_t r = 0; r < rows; r++) {
                    double d = (double)hv[r] - (double)hg[r];
                    num += d * d;
                    den += (double)hg[r] * (double)hg[r];
                }
                double rel = den > 0 ? sqrt(num / den) : 0;
                if (rel > worst) worst = rel;
                CHECK(rel < 1e-5, "%s T=%d lane=%d: rel %.3e vs gemv", shapes[s].c_str(), T, t, rel);
            }
        }
        // determinism: same input, same binary, 8 repeats -> bitwise identical
        int diff = 0;
        q27k::XLanes X{};
        q27k::YLanes Y{};
        for (int t = 0; t < W_PLUMB; t++) {
            X.nat[t] = xq[t].nat; X.xs[t] = xq[t].scale; Y.y[t] = d_y_v[t];
        }
        for (int rep = 0; rep < 8; rep++) {
            q27k::vgemm_verify(w, X, Y, d_ws, W_PLUMB, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy((rep ? hv2 : hv).data(), d_y_v[3], rows * 4,
                                  cudaMemcpyDeviceToHost));
            if (rep)
                for (int64_t r = 0; r < rows; r++)
                    if (hv[r] != hv2[r]) diff++;
        }
        CHECK(diff == 0, "%s: NON-DETERMINISTIC -- %d floats differ across 8 repeats",
              shapes[s].c_str(), diff);
        printf("  %-26s %-3s rows=%6ld cols=%6ld z=%d  worst rel %.2e  %s\n", shapes[s].c_str(),
               q4 ? "Q4" : "Q8", (long)rows, (long)cols, zused, worst,
               diff == 0 ? "bitwise-stable" : "NONDET");
    }

    printf("\n%s\n", fails == 0 ? "vgemm_test: ALL PASS" : "vgemm_test: FAILURES");
    return fails ? 1 : 0;
}
