#pragma once
// P13 adaptive draft-depth ceiling (Q27_MAXD=auto), extracted from engine.cuh
// for CPU testability (accept-gate plan Task 5; tests: tools/test_depthctl.cpp).
//
// Floats the per-stream ceiling between 4 and 5 from REALIZED acceptance:
// promote 4->5 when depth-4 rounds saturate the ceiling often enough
// (sat_ema >= hi), demote 5->4 when the 5th lane stops paying
// (yield_ema < lo). On promote, yield_ema is seeded just above the demote
// line so depth-5 gets a bounded grace window (~1/ema_a rounds) to prove
// itself. The ceiling changes round grouping / draft depth / verify width
// only -- never the emitted sequence (greedy is width-invariant).
//
// Host-side control logic only: no CUDA types, no engine state. The engine
// guards calls with (maxd_auto && md_used >= 0) semantics via update()'s
// own md_used check.
struct DepthCtl {
    int cur = 4;              // live ceiling (starts shallow)
    float sat_ema = 0.f;      // depth-4: EMA of (n reached ceiling)
    float yield_ema = 1.f;    // depth-5: EMA of (5th lane accepted)
    float ema_a = 1.f / 16.f; // EMA weight (~11-round half-life)
    float hi = 0.50f;         // promote 4->5 when sat_ema >= hi (Q27_MAXD_HI)
    // demote 5->4 when yield_ema < lo (Q27_MAXD_LO). 0.35 = the measured d5
    // win/loss crossover in CONDITIONAL yield units, post-verify-gemv
    // (BUILDLOG 2026-07-08 accept-gate Phase 0: -1.7% at y5 .282 @61K,
    // +0.2% at .355, +2.7% at .45+ @26K). The pre-Phase-1 0.10 was both
    // 3.5x below the crossover and in unconditional units (~y5*fired).
    float lo = 0.35f;
    long rounds4 = 0, rounds5 = 0;  // gated rounds run at each ceiling
    long promotes = 0, demotes = 0; // 4->5 / 5->4 transitions

    // Fold one gated greedy round into the ceiling. md_used = ceiling the
    // round drafted under (<0 = not a gated round: no-op), gate_cap = this
    // round's margin-run depth, n = tokens committed (1..6).
    void update(int md_used, int gate_cap, int n) {
        if (md_used < 0) return;
        if (md_used < 5) {
            rounds4++;
            float hit = (n >= md_used + 1) ? 1.f : 0.f;
            sat_ema += ema_a * (hit - sat_ema);
            // promote: seed yield just above the demote line (clamped -- a
            // 2*lo seed past 1.0 would stretch the grace window arbitrarily)
            // so depth-5 gets ~ln(2)/ema_a all-miss rounds to prove itself.
            if (sat_ema >= hi) {
                cur = 5;
                yield_ema = 2.f * lo > 1.f ? 1.f : 2.f * lo;
                promotes++;
            }
        } else {
            rounds5++;
            // Phase 1: yield evidence is CONDITIONAL on the 5th lane firing
            // (cap reached the ceiling). Unfired rounds say nothing about the
            // deep lane -- and under early-exit they barely pay for it. The
            // unconditional p(n=6) ~ y5*fired sat ABOVE the old lo=0.10 on
            // traffic where fixed depth-5 measured -1.7% (docs @61K).
            if (gate_cap >= md_used) {
                float hit = (n >= 6) ? 1.f : 0.f;
                yield_ema += ema_a * (hit - yield_ema);
                if (yield_ema < lo) { cur = 4; sat_ema = 0.f; demotes++; }
            }
        }
    }
};
