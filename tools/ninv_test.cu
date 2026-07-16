// ninv_test -- N-invariance gate for the continuous-batching determinism
// contract (docs/plans/2026-07-14-continuous-batching.md, P1 Task 5).
//
// CLAIM UNDER TEST ("bitwise-when-untrimmed"): a lane's output from the
// multi-lane verify weight kernels is bitwise independent of (a) how many
// OTHER lanes run in the same launch (T) and (b) WHICH slot the lane occupies.
// If that holds, a fused cross-engine round at union width T=w1+w2 produces,
// for each engine's lanes, exactly the bytes its solo round would have -- the
// whole Task 10 solo-equivalence gate rests on this.
//
// Method: 3 payload lanes get fixed host-generated pseudo-random activations,
// quantized ONCE via the engine's own quantize3 (k_quantize_x3) so both runs
// share bit-identical XQuant buffers -- the weight kernel is the only variable.
// Run A: T=N1, payload in the prefix slots {0,1,2}. Run B: T=N2>N1, the SAME
// payload buffers mapped to scattered slots (e.g. {1,4,7}); every other live
// slot carries JUNK that differs per run (junk proves isolation -- zeros would
// vacuously pass a lane-bleed bug whose contribution is x*0). Payload y
// buffers are pre-poisoned with a DIFFERENT byte pattern per run, so "kernel
// wrote nothing" can never compare equal. Outputs compared bitwise (word
// compare over rows floats).
//
// Families x shapes (discovered from the model like vgemm_test, not hardcoded):
//   vgemm_verify Q4  -- ffn_down (z=8, MODE 1 + k_reduce_z), output_q4 (z=1, MODE 0)
//   vgemm_verify Q8  -- ssm_out  (z=8, MODE 1)   [no engine Q8 shape has z=1]
//   gemv_q4_n        -- ffn_down (wide-K), output_q4 (tall head)
//   gemv_q8_n        -- ssm_out
//   gemv_f16_3       -- ssm_alpha (48 x 5120, the exact engine use)
// (N1,N2) in {(2,5),(3,9),(5,12),(9,16)}. All kernels accept T in 2..W_PLUMB;
// the ENGINE only reaches vgemm at vw >= gemm_min (9), but the fused round may
// grant any union width, so the low-T legs are tested too.
//
// A FAILURE HERE IS A FINDING, NOT A BUG TO FIX (plan addendum A1): the
// determinism contract downgrades for that family and the design doc gets the
// measured diff. Do not touch the kernels.
//
// SEAM LEG (P2 exit review, 2026-07-16): the tables above prove the
// MULTI-lane family invariant against itself (T/slot). But the SOLO draft
// path runs the SINGLE-lane family (k_gemv_q4/q8 via mm()/mtp_mm1,
// k_quantize_x via qx, k_rmsnorm, k_add via add_inplace, k_embed_row_q8,
// k_silu_mul) while the fused draft step runs the multi-lane twins -- so the
// P2c "fused margins bitwise vs solo" claim ALSO rests on single==multi for
// the payload lane, a seam no gate covered (test_kernels compares the gemv
// pair at 1e-5 tolerance only). The gemv twins carry different
// __launch_bounds__ tiers and differently-shaped accumulate expressions, so
// bitwise equality across the seam is a MEASUREMENT, not a property of "same
// math". The leg: identical payload input -> single-lane kernel vs multi-lane
// twin (payload in one live lane, junk elsewhere, T in {2,4}), BITWISE
// memcmp. EITHER verdict is a valid result -- a DIFFER downgrades the
// documented claim to tolerance-class (A1 flavor); do not touch the kernels.
//
// Usage: ninv_test [model.q27]   (default: the canonical qwen36-27b-mtp)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/blocks.cuh" // add_inplace (the solo residual add; seam leg)
#include "../src/device_model.h"
#include "../src/kernels.cuh"
#include "../src/loader.h"
#include "../src/spec3.cuh"
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

// xorshift32: explicit, seed-stable host PRNG (rand() would tie the gate to
// the libc). Maps to [-scale, scale).
static inline uint32_t xs32(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}
static void fill_rand(float* h, int64_t n, uint32_t seed, float scale) {
    uint32_t s = seed ? seed : 1u;
    for (int64_t i = 0; i < n; i++)
        h[i] = ((int32_t)xs32(s) >> 8) * (scale / 8388608.0f);
}

static const int NPAY = 3;                 // payload lanes under test
static const int64_t MAXC = 17408;         // widest decode cols (ffn_down)
static const int64_t MAXR = 248320;        // tallest rows (vocab head)

struct Case {
    int n1, n2;
    int slotsA[NPAY]; // -1 = unused (cases with N1 < 3 carry only N1 payloads)
    int slotsB[NPAY];
};
static const Case CASES[] = {
    {2, 5, {0, 1, -1}, {1, 4, -1}},
    {3, 9, {0, 1, 2}, {1, 4, 7}},
    {5, 12, {0, 1, 2}, {2, 9, 11}},
    {9, 16, {0, 1, 2}, {5, 10, 15}},
};
static const int NCASES = (int)(sizeof(CASES) / sizeof(CASES[0]));

enum Fam { F_VGEMM, F_GEMV, F_F16 };

// Persistent payload state: float activations + XQuant, quantized once.
static float* d_pay_x[NPAY];
static q27k::XQuant pay_xq[NPAY];
// Junk pool: one buffer set per slot, refilled with fresh junk before every run.
static float* d_junk_x[W_PLUMB];
static q27k::XQuant junk_xq[W_PLUMB];
static float* d_junk_y[W_PLUMB];
// Per-run payload outputs (A and B kept separate for the host compare).
static float* d_y_run[2][NPAY];

static std::vector<float> h_scratch; // MAXC staging for uploads

static void regen_junk(uint32_t seed) {
    for (int s = 0; s < W_PLUMB; s++) {
        fill_rand(h_scratch.data(), MAXC, seed ^ (0x9E3779B9u * (uint32_t)(s + 1)),
                  0.7f + 0.05f * (float)s);
        CUDA_CHECK(cudaMemcpy(d_junk_x[s], h_scratch.data(), MAXC * 4, cudaMemcpyHostToDevice));
        q27k::quantize_x(d_junk_x[s], MAXC, junk_xq[s]);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// One launch of `fam` on weight `w` at width T with payload lanes at `slots`.
// run selects the output buffer set + poison pattern.
static void launch_run(Fam fam, const q27::DevTensor& w, float* d_ws, int T, const int* slots,
                       int npay, int run) {
    // slot -> payload lane (-1 = junk)
    int lane_of[W_PLUMB];
    for (int s = 0; s < W_PLUMB; s++) lane_of[s] = -1;
    for (int i = 0; i < npay; i++) lane_of[slots[i]] = i;
    // poison payload outputs with a per-run pattern: an unwritten row can
    // never compare equal across runs.
    for (int i = 0; i < npay; i++)
        CUDA_CHECK(cudaMemset(d_y_run[run][i], run ? 0xBB : 0xAA, MAXR * 4));

    if (fam == F_VGEMM) {
        q27k::XLanes X{};
        q27k::YLanes Y{};
        for (int s = 0; s < W_PLUMB; s++) { // ALL slots get valid pointers (vgemm_test idiom)
            int l = lane_of[s];
            X.nat[s] = l >= 0 ? pay_xq[l].nat : junk_xq[s].nat;
            X.xs[s] = l >= 0 ? pay_xq[l].scale : junk_xq[s].scale;
            Y.y[s] = l >= 0 ? d_y_run[run][l] : d_junk_y[s];
        }
        bool ok = q27k::vgemm_verify(w, X, Y, d_ws, T, 0);
        CHECK(ok, "vgemm_verify refused rows=%ld cols=%ld T=%d", (long)w.rows, (long)w.cols, T);
    } else if (fam == F_GEMV) {
        q27k::XQuant qs[W_PLUMB];
        float* ys[W_PLUMB];
        for (int s = 0; s < W_PLUMB; s++) {
            int l = lane_of[s];
            qs[s] = l >= 0 ? pay_xq[l] : junk_xq[s];
            ys[s] = l >= 0 ? d_y_run[run][l] : d_junk_y[s];
        }
        if (w.dtype == q27::DType::Q4_G64)
            q27k::gemv_q4_n((const uint8_t*)w.data, (const __half*)w.scales, qs, T, ys, w.rows,
                            w.cols, 0);
        else
            q27k::gemv_q8_n((const int8_t*)w.data, (const __half*)w.scales, qs, T, ys, w.rows,
                            w.cols, 0);
    } else { // F_F16
        q27k::CP3 x{};
        q27k::P3 y{};
        for (int s = 0; s < W_PLUMB; s++) {
            int l = lane_of[s];
            x.p[s] = l >= 0 ? d_pay_x[l] : d_junk_x[s];
            y.p[s] = l >= 0 ? d_y_run[run][l] : d_junk_y[s];
        }
        q27k::gemv_f16_3((const __half*)w.data, x, y, w.rows, w.cols, 0, T);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Runs the 4 cases for one (family, weight); returns this table's fail count.
static int run_table(const char* tag, Fam fam, const q27::DevTensor& w, float* d_ws,
                     uint32_t junk_salt) {
    const int before = fails;
    std::vector<float> ha(w.rows), hb(w.rows);
    for (int c = 0; c < NCASES; c++) {
        const Case& K = CASES[c];
        const int npay = K.n1 < NPAY ? K.n1 : NPAY;
        // fresh junk PER RUN: if a payload output ever depends on junk-lane
        // contents, the two runs cannot agree.
        regen_junk(junk_salt + 2u * (uint32_t)c);
        launch_run(fam, w, d_ws, K.n1, K.slotsA, npay, 0);
        regen_junk(junk_salt + 2u * (uint32_t)c + 1u);
        launch_run(fam, w, d_ws, K.n2, K.slotsB, npay, 1);

        long diffs[NPAY] = {0, 0, 0};
        float maxad = 0.f;
        long total = 0;
        for (int i = 0; i < npay; i++) {
            CUDA_CHECK(cudaMemcpy(ha.data(), d_y_run[0][i], w.rows * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hb.data(), d_y_run[1][i], w.rows * 4, cudaMemcpyDeviceToHost));
            if (memcmp(ha.data(), hb.data(), w.rows * 4) != 0)
                for (uint64_t r = 0; r < w.rows; r++)
                    if (memcmp(&ha[r], &hb[r], 4) != 0) { // bitwise, NaN-safe
                        diffs[i]++;
                        maxad = fmaxf(maxad, fabsf(ha[r] - hb[r]));
                    }
            total += diffs[i];
        }
        printf("  %-22s T=%2d slots{%d,%d,%d} vs T=%2d slots{%d,%d,%d}  diffs=%ld/%ld/%ld  %s",
               tag, K.n1, K.slotsA[0], K.slotsA[1], K.slotsA[2], K.n2, K.slotsB[0], K.slotsB[1],
               K.slotsB[2], diffs[0], diffs[1], diffs[2], total == 0 ? "PASS" : "FAIL");
        if (total) printf("  (max |a-b| %.3e of %llu rows)", maxad, (unsigned long long)w.rows);
        printf("\n");
        CHECK(total == 0, "%s (N1=%d,N2=%d): %ld payload floats differ across runs", tag, K.n1,
              K.n2, total);
    }
    return fails - before;
}

// ---------------------------------------------------------------------------
// SEAM LEG (file-header comment: single-lane kernels vs their multi-lane
// twins across the solo/fused draft seam). Reuses the payload/junk state
// above; junk lanes carry per-run junk so lane bleed cannot vacuously pass.

static int seam_pairs_differ = 0; // pairs with >= 1 differing config

// float-bit lexicographic map -> ulp distance between bitwise-unequal floats
// (NaN/Inf just map to large deltas; we only ever print this on a DIFF).
static int64_t ulp_delta(float a, float b) {
    int32_t ia, ib;
    memcpy(&ia, &a, 4);
    memcpy(&ib, &b, 4);
    int64_t la = ia >= 0 ? (int64_t)ia : (int64_t)0x80000000LL - ia;
    int64_t lb = ib >= 0 ? (int64_t)ib : (int64_t)0x80000000LL - ib;
    int64_t d = la - lb;
    return d < 0 ? -d : d;
}

// bitwise word compare of two device f32 buffers; tracks max ulp over diffs
static long seam_cmp(const float* d_a, const float* d_b, int64_t n, int64_t* max_ulp) {
    std::vector<float> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), d_a, n * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), d_b, n * 4, cudaMemcpyDeviceToHost));
    if (memcmp(ha.data(), hb.data(), n * 4) == 0) return 0;
    long diffs = 0;
    for (int64_t i = 0; i < n; i++)
        if (memcmp(&ha[i], &hb[i], 4) != 0) {
            diffs++;
            int64_t u = ulp_delta(ha[i], hb[i]);
            if (u > *max_ulp) *max_ulp = u;
        }
    return diffs;
}

// raw byte compare (the quantize pair's int8/eo/isum sub-buffers)
static long seam_cmp_bytes(const void* d_a, const void* d_b, int64_t bytes) {
    std::vector<uint8_t> ha(bytes), hb(bytes);
    CUDA_CHECK(cudaMemcpy(ha.data(), d_a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), d_b, bytes, cudaMemcpyDeviceToHost));
    long diffs = 0;
    for (int64_t i = 0; i < bytes; i++)
        if (ha[i] != hb[i]) diffs++;
    return diffs;
}

static void seam_row(const char* tag, int T, int slot, long diffs, int64_t max_ulp, int64_t n,
                     bool* pair_diff) {
    printf("  %-24s T=%d slot=%d  diffs=%ld/%lld  %s", tag, T, slot, diffs,
           (long long)n, diffs == 0 ? "PASS" : "DIFF");
    if (diffs) printf("  (max %lld ulp)", (long long)max_ulp);
    printf("\n");
    if (diffs) *pair_diff = true;
}

static void seam_leg(q27::DeviceModel& dm, const q27::DevTensor& w_down,
                     const q27::DevTensor& w_head, const q27::DevTensor& w_sout) {
    const q27::DevTensor& emb = dm.upload("token_embd.weight"); // Q8_G128, cols 5120
    const int64_t NE = 5120; // rmsnorm/add/embed width (N_EMBD)
    const int64_t NF = MAXC; // silu/quantize width (N_FFN)
    const struct { int T, slot; } CFG[] = {{2, 0}, {4, 2}};

    // leg-owned buffers: single/multi payload outputs + rmsnorm weight +
    // fresh XQuant pair (pay_xq must stay untouched -- the gemv rows read it
    // as the SHARED input of both runs) + token slots for the embed pair.
    float *d_ys, *d_ym, *d_wrms;
    CUDA_CHECK(cudaMalloc(&d_ys, NF * 4));
    CUDA_CHECK(cudaMalloc(&d_ym, NF * 4));
    CUDA_CHECK(cudaMalloc(&d_wrms, NE * 4));
    fill_rand(h_scratch.data(), NE, 0xBEEFu, 0.9f);
    CUDA_CHECK(cudaMemcpy(d_wrms, h_scratch.data(), NE * 4, cudaMemcpyHostToDevice));
    q27k::XQuant xq_s = q27k::xquant_alloc(NF), xq_m = q27k::xquant_alloc(NF);
    int* d_tok; // slot 0 = payload token 1234; slots 1.. = junk row ids
    CUDA_CHECK(cudaMalloc(&d_tok, W_PLUMB * 4));
    {
        int htok[W_PLUMB];
        for (int s = 0; s < W_PLUMB; s++) htok[s] = 100 + 37 * s;
        htok[0] = 1234;
        CUDA_CHECK(cudaMemcpy(d_tok, htok, sizeof htok, cudaMemcpyHostToDevice));
    }

    printf("\n== seam: single-lane kernels vs multi-lane twins (payload lane, bitwise) ==\n");
    bool pd_gemv4 = false, pd_gemv8 = false, pd_q = false, pd_rms = false, pd_add = false,
         pd_emb = false, pd_silu = false;

    for (const auto& c : CFG) {
        const int T = c.T, slot = c.slot;
        int64_t mu;
        long d;

        // gemv_q4 vs gemv_q4_n (both engine Q4 shapes), gemv_q8 vs gemv_q8_n.
        // Both runs read the SAME pay_xq[0] device bytes: the kernel pair is
        // the only variable.
        const q27::DevTensor* ws[3] = {&w_down, &w_head, &w_sout};
        const char* wtag[3] = {"gemv_q4[ffn_down]", "gemv_q4[head]", "gemv_q8[ssm_out]"};
        for (int wi = 0; wi < 3; wi++) {
            const q27::DevTensor& w = *ws[wi];
            regen_junk(0x900u + 0x10u * (uint32_t)wi + (uint32_t)T);
            CUDA_CHECK(cudaMemset(d_y_run[0][0], 0xAA, w.rows * 4));
            CUDA_CHECK(cudaMemset(d_y_run[1][0], 0xBB, w.rows * 4));
            if (w.dtype == q27::DType::Q4_G64)
                q27k::gemv_q4((const uint8_t*)w.data, (const __half*)w.scales, pay_xq[0],
                              d_y_run[0][0], w.rows, w.cols);
            else
                q27k::gemv_q8((const int8_t*)w.data, (const __half*)w.scales, pay_xq[0],
                              d_y_run[0][0], w.rows, w.cols);
            q27k::XQuant qs[W_PLUMB];
            float* ys[W_PLUMB];
            for (int s = 0; s < W_PLUMB; s++) { qs[s] = junk_xq[s]; ys[s] = d_junk_y[s]; }
            qs[slot] = pay_xq[0];
            ys[slot] = d_y_run[1][0];
            if (w.dtype == q27::DType::Q4_G64)
                q27k::gemv_q4_n((const uint8_t*)w.data, (const __half*)w.scales, qs, T, ys,
                                w.rows, w.cols);
            else
                q27k::gemv_q8_n((const int8_t*)w.data, (const __half*)w.scales, qs, T, ys,
                                w.rows, w.cols);
            CUDA_CHECK(cudaDeviceSynchronize());
            mu = 0;
            d = seam_cmp(d_y_run[0][0], d_y_run[1][0], w.rows, &mu);
            seam_row(wtag[wi], T, slot, d, mu, w.rows, wi == 2 ? &pd_gemv8 : &pd_gemv4);
        }

        // quantize_x vs quantize3: all four output sub-buffers, bitwise.
        {
            regen_junk(0x950u + (uint32_t)T);
            CUDA_CHECK(cudaMemset(xq_s.nat, 0xAA, NF));
            CUDA_CHECK(cudaMemset(xq_s.eo, 0xAA, NF / 8 * sizeof(uint2)));
            CUDA_CHECK(cudaMemset(xq_s.scale, 0xAA, NF / 32 * 4));
            CUDA_CHECK(cudaMemset(xq_s.isum, 0xAA, NF / 32 * 4));
            CUDA_CHECK(cudaMemset(xq_m.nat, 0xBB, NF));
            CUDA_CHECK(cudaMemset(xq_m.eo, 0xBB, NF / 8 * sizeof(uint2)));
            CUDA_CHECK(cudaMemset(xq_m.scale, 0xBB, NF / 32 * 4));
            CUDA_CHECK(cudaMemset(xq_m.isum, 0xBB, NF / 32 * 4));
            q27k::quantize_x(d_pay_x[0], NF, xq_s);
            q27k::CP3 xs{};
            q27k::XQ3 x3{};
            for (int s = 0; s < W_PLUMB; s++) { xs.p[s] = d_junk_x[s]; x3.q[s] = junk_xq[s]; }
            xs.p[slot] = d_pay_x[0];
            x3.q[slot] = xq_m;
            q27k::quantize3(xs, NF, x3, 0, T);
            CUDA_CHECK(cudaDeviceSynchronize());
            mu = 0;
            d = seam_cmp_bytes(xq_s.nat, xq_m.nat, NF);
            d += seam_cmp_bytes(xq_s.eo, xq_m.eo, NF / 8 * sizeof(uint2));
            d += seam_cmp(xq_s.scale, xq_m.scale, NF / 32, &mu);
            d += seam_cmp_bytes(xq_s.isum, xq_m.isum, NF / 32 * 4);
            seam_row("quantize_x", T, slot, d, mu, NF + NF / 8 + 2 * (NF / 32), &pd_q);
        }

        // rmsnorm vs rmsnorm3
        {
            regen_junk(0x960u + (uint32_t)T);
            CUDA_CHECK(cudaMemset(d_ys, 0xAA, NE * 4));
            CUDA_CHECK(cudaMemset(d_ym, 0xBB, NE * 4));
            q27k::rmsnorm(d_pay_x[0], d_wrms, d_ys, (int)NE, 1e-6f);
            q27k::CP3 x{};
            q27k::P3 y{};
            for (int s = 0; s < W_PLUMB; s++) { x.p[s] = d_junk_x[s]; y.p[s] = d_junk_y[s]; }
            x.p[slot] = d_pay_x[0];
            y.p[slot] = d_ym;
            q27k::rmsnorm3(x, d_wrms, y, (int)NE, 1e-6f, 0, T);
            CUDA_CHECK(cudaDeviceSynchronize());
            mu = 0;
            d = seam_cmp(d_ys, d_ym, NE, &mu);
            seam_row("rmsnorm", T, slot, d, mu, NE, &pd_rms);
        }

        // add_inplace vs add3 (in place: both sides seeded with the payload;
        // junk lanes get d_junk_y as their mutable accumulator)
        {
            regen_junk(0x970u + (uint32_t)T);
            CUDA_CHECK(cudaMemcpy(d_ys, d_pay_x[0], NE * 4, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_ym, d_pay_x[0], NE * 4, cudaMemcpyDeviceToDevice));
            q27k::add_inplace(d_ys, d_pay_x[1], (int)NE);
            q27k::P3 x{};
            q27k::CP3 yy{};
            for (int s = 0; s < W_PLUMB; s++) { x.p[s] = d_junk_y[s]; yy.p[s] = d_junk_x[s]; }
            x.p[slot] = d_ym;
            yy.p[slot] = d_pay_x[1];
            q27k::add3(x, yy, (int)NE, 0, T);
            CUDA_CHECK(cudaDeviceSynchronize());
            mu = 0;
            d = seam_cmp(d_ys, d_ym, NE, &mu);
            seam_row("add", T, slot, d, mu, NE, &pd_add);
        }

        // embed_row_q8 vs embed3 (junk lanes read junk token rows; read-only
        // weight, so a junk lane sharing the payload row would be harmless)
        {
            regen_junk(0x980u + (uint32_t)T);
            CUDA_CHECK(cudaMemset(d_ys, 0xAA, NE * 4));
            CUDA_CHECK(cudaMemset(d_ym, 0xBB, NE * 4));
            q27k::embed_row_q8((const int8_t*)emb.data, (const __half*)emb.scales, d_tok, NE,
                               d_ys);
            q27k::IP3 tk{};
            q27k::P3 out{};
            for (int s = 0; s < W_PLUMB; s++) { tk.p[s] = d_tok + s; out.p[s] = d_junk_y[s]; }
            tk.p[slot] = d_tok;
            out.p[slot] = d_ym;
            q27k::embed3((const int8_t*)emb.data, (const __half*)emb.scales, tk, NE, out, 0, T);
            CUDA_CHECK(cudaDeviceSynchronize());
            mu = 0;
            d = seam_cmp(d_ys, d_ym, NE, &mu);
            seam_row("embed_row_q8", T, slot, d, mu, NE, &pd_emb);
        }

        // silu_mul vs silu_mul3 (solo path runs out == gate in place; the
        // multi twin is in place on g by construction)
        {
            regen_junk(0x990u + (uint32_t)T);
            CUDA_CHECK(cudaMemcpy(d_ys, d_pay_x[0], NF * 4, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_ym, d_pay_x[0], NF * 4, cudaMemcpyDeviceToDevice));
            q27k::silu_mul(d_ys, d_pay_x[1], d_ys, (int)NF);
            q27k::P3 g{};
            q27k::CP3 u{};
            for (int s = 0; s < W_PLUMB; s++) { g.p[s] = d_junk_y[s]; u.p[s] = d_junk_x[s]; }
            g.p[slot] = d_ym;
            u.p[slot] = d_pay_x[1];
            q27k::silu_mul3(g, u, (int)NF, 0, T);
            CUDA_CHECK(cudaDeviceSynchronize());
            mu = 0;
            d = seam_cmp(d_ys, d_ym, NF, &mu);
            seam_row("silu_mul", T, slot, d, mu, NF, &pd_silu);
        }
    }

    seam_pairs_differ =
        (int)pd_gemv4 + pd_gemv8 + pd_q + pd_rms + pd_add + pd_emb + pd_silu;
    if (seam_pairs_differ == 0) printf("SEAM BITWISE: ALL PASS\n");
    else printf("SEAM BITWISE: %d pairs DIFFER (tolerance-class)\n", seam_pairs_differ);

    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_ym));
    CUDA_CHECK(cudaFree(d_wrms));
    CUDA_CHECK(cudaFree(d_tok));
}

int main(int argc, char** argv) {
    const char* path =
        argc > 1 ? argv[1] : "/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27";
    q27::Model m = q27::Model::open(path);
    q27::DeviceModel dm(m);

    // shape discovery, vgemm_test style -- never hardcode a layer index
    auto first_with = [&](const char* leaf) -> std::string {
        for (int il = 0; il < 80; il++) {
            char b[96];
            snprintf(b, sizeof b, "blk.%d.%s", il, leaf);
            if (m.find(b)) return b;
        }
        fprintf(stderr, "ninv_test: no layer carries %s\n", leaf);
        exit(1);
    };
    const std::string n_ffn_down = first_with("ffn_down.weight");   // Q4, 5120 x 17408
    const std::string n_ssm_out = first_with("ssm_out.weight");     // Q8, 5120 x 6144
    const std::string n_ssm_alpha = first_with("ssm_alpha.weight"); // F16, 48 x 5120
    if (!m.find("output_q4.weight")) { fprintf(stderr, "ninv_test: no output_q4\n"); return 1; }
    const q27::DevTensor& w_down = dm.upload(n_ffn_down);
    const q27::DevTensor& w_head = dm.upload("output_q4.weight");   // Q4, 248320 x 5120
    const q27::DevTensor& w_sout = dm.upload(n_ssm_out);
    const q27::DevTensor& w_alpha = dm.upload(n_ssm_alpha);

    // buffers
    h_scratch.resize(MAXC);
    for (int t = 0; t < NPAY; t++) {
        CUDA_CHECK(cudaMalloc(&d_pay_x[t], MAXC * 4));
        pay_xq[t] = q27k::xquant_alloc(MAXC);
        CUDA_CHECK(cudaMalloc(&d_y_run[0][t], MAXR * 4));
        CUDA_CHECK(cudaMalloc(&d_y_run[1][t], MAXR * 4));
    }
    for (int s = 0; s < W_PLUMB; s++) {
        CUDA_CHECK(cudaMalloc(&d_junk_x[s], MAXC * 4));
        junk_xq[s] = q27k::xquant_alloc(MAXC);
        CUDA_CHECK(cudaMalloc(&d_junk_y[s], MAXR * 4));
    }
    // payload activations: fixed seed, then quantized ONCE via the engine's own
    // quantize3 -- both runs of every case share these exact device bytes.
    {
        q27k::CP3 xs{};
        q27k::XQ3 xq{};
        for (int t = 0; t < NPAY; t++) {
            fill_rand(h_scratch.data(), MAXC, 0x51A1u + 977u * (uint32_t)t,
                      0.5f + 0.1f * (float)t);
            CUDA_CHECK(cudaMemcpy(d_pay_x[t], h_scratch.data(), MAXC * 4,
                                  cudaMemcpyHostToDevice));
            xs.p[t] = d_pay_x[t];
            xq.q[t] = pay_xq[t];
        }
        q27k::quantize3(xs, MAXC, xq, 0, NPAY);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // vgemm workspace: sized exactly as the engine sizes it (max over shapes)
    const q27::DevTensor* wl[3] = {&w_down, &w_head, &w_sout};
    size_t wsb = q27k::vgemm_ws_bytes(wl, 3);
    float* d_ws;
    CUDA_CHECK(cudaMalloc(&d_ws, wsb));

    printf("== ninv: lane output invariance vs T and slot (W_PLUMB=%d, ws %.2f MB) ==\n",
           W_PLUMB, wsb / 1e6);
    struct Row { const char* tag; Fam fam; const q27::DevTensor* w; uint32_t salt; };
    const Row rows[] = {
        {"vgemm_q4[ffn_down]", F_VGEMM, &w_down, 0x10u},  // MODE 1 (z>1) + reduce
        {"vgemm_q4[head]", F_VGEMM, &w_head, 0x20u},      // MODE 0 (z=1)
        {"vgemm_q8[ssm_out]", F_VGEMM, &w_sout, 0x30u},   // MODE 1 (no Q8 z=1 shape exists)
        {"gemv_q4_n[ffn_down]", F_GEMV, &w_down, 0x40u},
        {"gemv_q4_n[head]", F_GEMV, &w_head, 0x50u},
        {"gemv_q8_n[ssm_out]", F_GEMV, &w_sout, 0x60u},
        {"gemv_f16_3[ssm_alpha]", F_F16, &w_alpha, 0x70u},
    };
    // family verdicts aggregate the per-weight tables
    int fam_fail[3] = {0, 0, 0};
    for (const Row& r : rows) fam_fail[r.fam] += run_table(r.tag, r.fam, *r.w, d_ws, r.salt);

    printf("\nfamily verdicts: vgemm_verify %s | gemv_q4_n/gemv_q8_n %s | gemv_f16_3 %s\n",
           fam_fail[F_VGEMM] ? "FAIL" : "PASS", fam_fail[F_GEMV] ? "FAIL" : "PASS",
           fam_fail[F_F16] ? "FAIL" : "PASS");

    // P2 exit review: the single/multi-lane seam (file-header SEAM LEG note).
    seam_leg(dm, w_down, w_head, w_sout);
    // Measured 2026-07-16 on BOTH arches (5090/sm_120 and 3090/sm_86 -- ptxas
    // contraction is per-arch): ALL PASS, every pair, T in {2,4}. The seam IS
    // bitwise, so it joins the gated contract: a future divergence (compiler
    // flag, launch_bounds retier, kernel edit) must fail this binary, not
    // silently downgrade the fused-vs-solo margin claim.
    fails += seam_pairs_differ;

    if (fails == 0) printf("NINV ALL PASS\n");
    else printf("ninv_test: %d FAILURES -- finding, not a bug: contract downgrades per A1\n",
                fails);
    return fails ? 1 : 0;
}
