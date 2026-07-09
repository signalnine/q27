#pragma once
// Adaptive draft-depth ceiling (Q27_MAXD=auto): P13's 4<->5 controller,
// extended 2026-07-08 to a 4..6 ladder per the maxd6 GO verdict
// (docs/maxd6-decision.md). CPU-tested: tools/test_depthctl.cpp.
//
// Per-level EMAs from REALIZED acceptance: at the live level k,
//   sat[k] = EMA(n reached k+1), unconditional  -- promote k->k+1 at >= hi
//   yld[k] = EMA(n reached k+1 | top lane FIRED, cap >= k)
//                                              -- demote k->k-1 at < lo
// lo = 0.35 is the measured d5 win/loss crossover in conditional-yield units
// (BUILDLOG 2026-07-08 accept-gate Phase 0); the level-6 breakeven has the
// same structure (maxd6-decision.md rerun). hi = 0.50 revalidated 3x.
//
// Entering a level (either direction) resets its sat and seeds its yield just
// above the demote line: every stint gets a bounded grace window
// (~ln(2)/ema_a all-miss fired rounds) and must re-earn the next promote --
// prevents stale-EMA thrash cascades (6->5->4). At k_max=5 this is
// behavior-identical to the shipped P13+Phase-1 controller (sat[5]/yld[4]
// were never read there).
//
// The ceiling changes round grouping / draft depth / verify width only --
// never the emitted sequence (greedy is width-invariant).
struct DepthCtl {
    int k_min = 4, k_max = 5; // ladder range; engine raises k_max to gate_maxd
                              // under Q27_MAXD=auto with dexit on (up to 7)
    int cur = 4;              // live ceiling (starts shallow)
    float sat[8] = {};        // [level]; only the live level's entries update
    float yld[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    float ema_a = 1.f / 16.f; // EMA weight (~11-round half-life)
    float hi = 0.50f;         // 4->5 promote bar on sat (Q27_MAXD_HI)
    // 5->6 promote bar (Q27_MAXD_HI6), deliberately ABOVE hi: bursty ~50%-sat
    // level-5 traffic (docs flavor, sat5 ~.46) pays a level-6 drafting tax the
    // conditional-yield demote bar cannot see (lane 6 fires too rarely to
    // amortize the 6th draft step on cap>=5 rounds; measured -1.9%), while
    // sustained deep saturation (cctx .71) wins +4.1%. 0.60 separates them.
    float hi6 = 0.60f;
    float hi7 = 0.60f;        // 6->7 promote bar (Q27_MAXD_HI7); authorized by
                              // live sat6 .64 on CC traffic (BUILDLOG 07-08)
    float lo = 0.35f;         // demote bar on conditional yield (Q27_MAXD_LO)
    // Level-6 fired-rate bar (Q27_MAXD_FLO6): the 6th draft step runs on every
    // cap>=5 round, but lane 6 only pays when the margin run REACHES 6. Traffic
    // that promotes yet rarely fires 6-deep (docs: fired6 ~.3 at y6 ~.70,
    // measured -2%) is invisible to the conditional-yield bar; cctx fires .6+.
    // Applies at level 6 only: at level 5 low-fired/high-yield flavors WIN
    // (testgen: fired .30, +3.9%), so a fired bar there would be wrong.
    float flo6 = 0.45f;
    float flo7 = 0.45f;       // level-7 fired bar (Q27_MAXD_FLO7)
    float fired_ema = 0.f;    // deep-stint (level>=6) EMA of (cap reached md)
    long rounds[8] = {};      // gated rounds run at each level
    long promotes = 0, demotes = 0;

    void enter(int k) {
        sat[k] = 0.f;
        yld[k] = 2.f * lo > 1.f ? 1.f : 2.f * lo;
        if (k >= 6) {
            float f = k == 6 ? flo6 : flo7;
            fired_ema = 2.f * f > 1.f ? 1.f : 2.f * f;
        }
    }

    // Per-request isolation (review 2026-07-09): the controller lives on the
    // engine, which serves unrelated requests back-to-back -- without a reset
    // each request inherits the previous tenant's ceiling and EMAs (depth
    // trajectory becomes traffic-history-dependent, and a saturating tenant
    // hands the next one a deep ceiling it hasn't earned). Called at
    // generate() entry; tunables (bars, ema_a, k_max) survive, learned state
    // does not. Cost: each request re-earns depth over ~2/ema_a rounds.
    void reset() {
        cur = k_min;
        for (int i = 0; i < 8; i++) { sat[i] = 0.f; yld[i] = 1.f; rounds[i] = 0; }
        fired_ema = 0.f;
        promotes = demotes = 0;
        enter(cur);
    }

    // Fold one gated greedy round into the ceiling. md = ceiling the round
    // drafted under (<0 = not a gated round: no-op), cap = this round's
    // margin-run depth, n = tokens committed (1..md+1).
    void update(int md, int cap, int n) {
        if (md < 0) return;
        if (md > 7) md = 7;
        rounds[md]++;
        float hit = (n >= md + 1) ? 1.f : 0.f;
        sat[md] += ema_a * (hit - sat[md]);
        if (md < k_max && sat[md] >= (md >= 6 ? hi7 : md >= 5 ? hi6 : hi)) {
            cur = md + 1;
            enter(cur);
            promotes++;
            return; // this round's evidence went into the promote
        }
        // Deep-level fired-rate bar (levels >= 6): every round updates it
        // (fired = margin run reached the ceiling); demote when deep
        // confidence is too rare to amortize the extra draft step.
        if (md >= 6) {
            fired_ema += ema_a * ((cap >= md ? 1.f : 0.f) - fired_ema);
            if (fired_ema < (md == 6 ? flo6 : flo7)) {
                cur = md - 1;
                enter(cur);
                demotes++;
                return;
            }
        }
        // yield evidence is CONDITIONAL on the top lane firing (accept-gate
        // Phase 1): unfired rounds say nothing about the deep lane and, under
        // early-exit, barely pay for it.
        if (md > k_min && cap >= md) {
            yld[md] += ema_a * (hit - yld[md]);
            if (yld[md] < lo) {
                cur = md - 1;
                enter(cur);
                demotes++;
            }
        }
    }
};
