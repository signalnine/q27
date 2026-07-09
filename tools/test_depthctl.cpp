// CPU unit tests for src/depthctl.h -- the P13 adaptive draft-depth ceiling
// extracted from engine.cuh (accept-gate plan Task 5). No CUDA; pins the
// controller semantics so the Phase-1/2 changes are reviewable in isolation.
//
//   g++ -std=c++17 -O1 -o build/test_depthctl tools/test_depthctl.cpp && build/test_depthctl
//
// Semantics pinned here (P13, BUILDLOG:1286 + engine.cuh):
//   - starts at ceiling 4; depth-4 rounds update sat_ema only (hit = n reached
//     md_used+1), depth-5 rounds update yield_ema only (hit = n >= 6)
//   - promote 4->5 at sat_ema >= hi (0.50), seeding yield_ema = 2*lo
//   - demote 5->4 at yield_ema < lo (0.10), zeroing sat_ema
//   - a = 1/16 => a hit=1 run from 0 crosses 0.5 on update #11
#include "../src/depthctl.h"

#include <cmath>
#include <cstdio>

static int fails = 0;
#define CHECK(cond, name)                                          \
    do {                                                           \
        bool ok = (cond);                                          \
        printf("  %-58s %s\n", name, ok ? "PASS" : "FAIL");        \
        if (!ok) fails++;                                          \
    } while (0)

int main() {
    { // starts shallow, disabled-md guard
        DepthCtl c;
        CHECK(c.cur == 4 && c.sat[4] == 0.f && c.yld[5] == 1.f, "init: cur=4, EMAs at rest");
        c.update(-1, -1, 5);
        CHECK(c.cur == 4 && c.rounds[4] == 0, "md_used<0 is a no-op");
    }
    { // promote timing: saturated depth-4 stream crosses hi=0.5 on round 11
        DepthCtl c;
        int promoted_at = -1;
        for (int i = 1; i <= 16 && promoted_at < 0; i++) {
            c.update(4, 4, 5); // n == md_used+1: ceiling saturated
            if (c.cur == 5) promoted_at = i;
        }
        CHECK(promoted_at == 11, "promote on saturated round 11 (1-(15/16)^11 >= .5)");
        CHECK(c.promotes == 1 && c.rounds[4] == 11, "promote/rounds4 counters");
        CHECK(std::fabs(c.yld[5] - 2.f * c.lo) < 1e-6f, "promote seeds yield_ema = 2*lo");
    }
    { // depth-4 misses decay sat_ema; no promote
        DepthCtl c;
        for (int i = 0; i < 8; i++) c.update(4, 4, 5);
        float peak = c.sat[4];
        for (int i = 0; i < 8; i++) c.update(4, 0, 1); // n=1: far from ceiling
        CHECK(c.sat[4] < peak && c.cur == 4, "unsaturated rounds decay sat_ema, no promote");
    }
    { // demote timing: from the 2*lo seed, an all-miss depth-5 stream demotes on round 11
        DepthCtl c;
        for (int i = 0; i < 11; i++) c.update(4, 4, 5); // drive the promote
        CHECK(c.cur == 5, "(pre) promoted");
        int demoted_at = -1;
        for (int i = 1; i <= 16 && demoted_at < 0; i++) {
            c.update(5, 5, 5); // depth-5 round, 5th lane NOT accepted (n=5 < 6)
            if (c.cur == 4) demoted_at = i;
        }
        CHECK(demoted_at == 11, "demote on miss round 11 (2lo*(15/16)^11 < lo)");
        CHECK(c.demotes == 1 && c.sat[4] == 0.f, "demote zeroes sat_ema");
    }
    { // level exclusivity: depth-4 rounds never touch yield_ema, depth-5 never sat_ema
        DepthCtl c;
        float y0 = c.yld[5];
        for (int i = 0; i < 5; i++) c.update(4, 4, 5);
        CHECK(c.yld[5] == y0, "depth-4 rounds leave yield_ema untouched");
        float s0 = c.sat[4];
        c.update(5, 5, 6);
        CHECK(c.sat[4] == s0 && c.rounds[5] == 1, "depth-5 rounds leave sat_ema untouched");
    }
    { // depth-5 hits hold the ceiling: yield_ema recovers toward 1
        DepthCtl c;
        for (int i = 0; i < 11; i++) c.update(4, 4, 5);
        for (int i = 0; i < 64; i++) c.update(5, 5, 6); // 5th lane accepted
        CHECK(c.cur == 5 && c.yld[5] > 0.9f, "sustained accepts keep depth-5");
    }
    // ---- accept-gate Phase 1 (BUILDLOG 2026-07-08 Phase 0 measurement) ----
    { // yield evidence is CONDITIONAL on the 5th lane firing (cap == ceiling);
      // unfired rounds say nothing about the deep lane and barely pay for it
      // under early-exit. Unconditional p(n=6) ~ y5*fired sat ABOVE lo on
      // traffic where depth-5 measured -1.7% (docs61k).
        DepthCtl c;
        for (int i = 0; i < 11; i++) c.update(4, 4, 5);
        float y0 = c.yld[5];
        for (int i = 0; i < 32; i++) c.update(5, 3, 4); // depth-5, lane 5 unfired
        CHECK(c.yld[5] == y0 && c.cur == 5, "unfired lane 5: no yield evidence, no demote");
        c.update(5, 5, 5); // fired, missed
        CHECK(c.yld[5] < y0, "fired+missed decays yield_ema");
    }
    { // promote seed clamps at 1.0 once lo passes 0.5
        DepthCtl c;
        c.lo = 0.6f;
        for (int i = 0; i < 11; i++) c.update(4, 4, 5);
        CHECK(c.cur == 5 && c.yld[5] <= 1.f, "promote seed = min(1, 2*lo)");
    }
    { // production bar: a fired stream at ~33% yield (codegen/docs61k flavor)
      // demotes; the crossover measured y5~0.35 (Phase 0 run 3 + 61K)
        DepthCtl c;
        c.lo = 0.35f;
        for (int i = 0; i < 11; i++) c.update(4, 4, 5);
        int demoted_at = -1;
        for (int i = 1; i <= 96 && demoted_at < 0; i++) {
            c.update(5, 5, (i % 3 == 0) ? 6 : 5); // fired every round, hit 1-in-3
            if (c.cur == 4) demoted_at = i;
        }
        CHECK(demoted_at >= 15 && demoted_at <= 64, "33%-yield fired stream demotes (15..64 rounds)");
    }
    { // and a 50%-yield fired stream HOLDS at bar 0.35 (winning regime, echo/testgen)
        DepthCtl c;
        c.lo = 0.35f;
        for (int i = 0; i < 11; i++) c.update(4, 4, 5);
        for (int i = 1; i <= 128; i++) c.update(5, 5, (i % 2 == 0) ? 6 : 5);
        CHECK(c.cur == 5 && c.demotes == 0, "50%-yield fired stream holds depth-5 at bar .35");
    }
    // ---- maxd6 ladder (2026-07-08 GO verdict): levels 4..6 ----
    { // ladder disabled by default: k_max=5 keeps 4<->5 behavior, never reaches 6
        DepthCtl c;
        for (int i = 0; i < 64; i++) c.update(c.cur, c.cur, c.cur + 1); // always saturated
        CHECK(c.cur == 5, "k_max=5 (default): saturated stream tops out at 5");
    }
    { // full promote path 4->5->6: entering a level resets its sat, so level 6
      // needs a fresh saturated stint AT level 5; the 5->6 bar is hi6=0.60
      // (15 rounds: 1-(15/16)^15 >= .6), not hi=0.50 (11 rounds).
        DepthCtl c;
        c.k_max = 6;
        int at5 = -1, at6 = -1, i = 0;
        while (i++ < 64 && at6 < 0) {
            c.update(c.cur, c.cur, c.cur + 1); // every round saturates its ceiling
            if (c.cur == 5 && at5 < 0) at5 = i;
            if (c.cur == 6 && at6 < 0) at6 = i;
        }
        CHECK(at5 == 11 && at6 == 26, "promote 4->5 on round 11, 5->6 on round 26 (hi6)");
        CHECK(c.promotes == 2, "two promotes counted");
    }
    { // demote 6->5 on dead lane-6 yield; fresh level-5 seed prevents a cascade
        DepthCtl c;
        c.k_max = 6;
        for (int i = 0; i < 26; i++) c.update(c.cur, c.cur, c.cur + 1); // reach 6 (11 + hi6's 15)
        CHECK(c.cur == 6, "(pre) at 6");
        int demoted_at = -1;
        for (int i = 1; i <= 16 && demoted_at < 0; i++) {
            c.update(6, 6, 6); // fired, 6th lane never accepted (n=6 < 7)
            if (c.cur == 5) demoted_at = i;
        }
        CHECK(demoted_at == 11, "demote 6->5 on miss round 11");
        CHECK(c.cur == 5 && c.demotes == 1, "single demote, no instant cascade to 4");
        c.update(5, 5, 6); // one fired hit at level 5
        CHECK(c.cur == 5, "level-5 stint continues on its own fresh evidence");
    }
    { // level exclusivity: level-5 rounds never touch level-6 EMAs
        DepthCtl c;
        c.k_max = 6;
        for (int i = 0; i < 11; i++) c.update(4, 4, 5); // reach 5
        float y6 = c.yld[6], s6 = c.sat[6];
        for (int i = 0; i < 8; i++) c.update(5, 5, 5);
        CHECK(c.yld[6] == y6 && c.sat[6] == s6, "level-5 rounds leave level-6 EMAs untouched");
    }
    { // 5->6 promote bar is HIGHER (hi6=0.60): bursty ~50%-sat level-5 traffic
      // (docs flavor) stays at 5; only sustained deep saturation promotes.
      // Measured: docs sat5 ~.46 bursty lost -1.9% at level 6 (drafting tax the
      // conditional yield bar cannot see); cctx sat5 .71 wins +4.1%.
        DepthCtl c;
        c.k_max = 6;
        for (int i = 0; i < 11; i++) c.update(4, 4, 5); // reach 5
        for (int i = 1; i <= 128; i++) c.update(5, 5, (i % 2 == 0) ? 6 : 5); // 50% sat
        CHECK(c.cur == 5 && c.promotes == 1, "50%-sat level-5 stream never promotes to 6");
        DepthCtl d;
        d.k_max = 6;
        for (int i = 0; i < 11; i++) d.update(4, 4, 5);
        int at6 = -1;
        for (int i = 1; i <= 32 && at6 < 0; i++) {
            d.update(5, 5, 6); // fully saturated level-5 stream (cctx flavor)
            if (d.cur == 6) at6 = i;
        }
        CHECK(at6 == 15, "sustained saturation promotes 5->6 on round 15 (1-(15/16)^15 >= .6)");
    }
    { // level-6 fired-rate bar (flo6): docs-flavor traffic whose margins rarely
      // run 6-deep (fired ~0.3) pays the 6th draft step without landing lane-6
      // tokens -- conditional yield can't see it (y6 ~0.70 when it DOES fire,
      // measured). cctx fires 0.6+. flo6=0.45 separates them.
        DepthCtl c;
        c.k_max = 6;
        for (int i = 0; i < 11; i++) c.update(4, 4, 5);
        for (int i = 0; i < 15; i++) c.update(5, 5, 6); // sustained sat -> promote to 6
        CHECK(c.cur == 6, "(pre) at 6 via sustained sat");
        int demoted_at = -1;
        for (int i = 1; i <= 64 && demoted_at < 0; i++) {
            // fired 1-in-3 (cap=6 every 3rd round); accepted whenever fired
            bool fired = (i % 3 == 0);
            c.update(6, fired ? 6 : 5, fired ? 7 : 6);
            if (c.cur == 5) demoted_at = i;
        }
        CHECK(demoted_at > 0 && demoted_at <= 40, "low-fired level-6 stream demotes via flo6");
        DepthCtl d;
        d.k_max = 6;
        for (int i = 0; i < 11; i++) d.update(4, 4, 5);
        for (int i = 0; i < 15; i++) d.update(5, 5, 6);
        for (int i = 1; i <= 128; i++) {
            bool fired = (i % 3 != 0); // fired 2-in-3, accepted whenever fired (cctx flavor)
            d.update(6, fired ? 6 : 5, fired ? 7 : 6);
        }
        CHECK(d.cur == 6 && d.demotes == 0, "cctx-like 0.67-fired level-6 stream holds");
    }
    // ---- maxd7 ladder (2026-07-09): level 7 ----
    { // k_max=6 regression: saturated stream tops out at 6 (no accidental 7)
        DepthCtl c;
        c.k_max = 6;
        for (int i = 0; i < 64; i++) c.update(c.cur, c.cur, c.cur + 1);
        CHECK(c.cur == 6, "k_max=6: saturated stream tops out at 6");
    }
    { // full promote path 4->5->6->7: 11 (hi .5) + 15 (hi6 .6) + 15 (hi7 .6)
        DepthCtl c;
        c.k_max = 7;
        int at7 = -1, i = 0;
        while (i++ < 96 && at7 < 0) {
            c.update(c.cur, c.cur, c.cur + 1);
            if (c.cur == 7 && at7 < 0) at7 = i;
        }
        CHECK(at7 == 41 && c.promotes == 3, "promote 4->5@11, 5->6@26, 6->7@41");
    }
    { // demote 7->6 on dead lane-7 yield; level-6 EMAs untouched by level-7 rounds
        DepthCtl c;
        c.k_max = 7;
        for (int i = 0; i < 41; i++) c.update(c.cur, c.cur, c.cur + 1); // reach 7
        CHECK(c.cur == 7, "(pre) at 7");
        float y6 = c.yld[6];
        int demoted_at = -1;
        for (int i = 1; i <= 16 && demoted_at < 0; i++) {
            c.update(7, 7, 7); // fired, 7th lane never accepted (n=7 < 8)
            if (c.cur == 6) demoted_at = i;
        }
        CHECK(demoted_at == 11 && c.cur == 6, "demote 7->6 on miss round 11");
        CHECK(c.yld[6] == y6 || c.yld[6] == 2.f * c.lo, "level-7 stint left level-6 yld to its fresh seed");
    }
    { // level-7 fired-rate bar (flo7): margins rarely running 7-deep demotes
        DepthCtl c;
        c.k_max = 7;
        for (int i = 0; i < 41; i++) c.update(c.cur, c.cur, c.cur + 1);
        int demoted_at = -1;
        for (int i = 1; i <= 64 && demoted_at < 0; i++) {
            bool fired = (i % 3 == 0); // fired 1-in-3, accepted whenever fired
            c.update(7, fired ? 7 : 6, fired ? 8 : 7);
            if (c.cur == 6) demoted_at = i;
        }
        CHECK(demoted_at > 0 && demoted_at <= 40, "low-fired level-7 stream demotes via flo7");
    }
    { // reset(): per-request isolation (review 2026-07-09) -- learned state
      // cleared, tunables survive, next request re-earns depth from k_min
        DepthCtl c;
        c.k_max = 7;
        c.hi = 0.42f; // tunable must survive reset
        for (int i = 0; i < 41; i++) c.update(c.cur, c.cur, c.cur + 1); // deep + warm
        CHECK(c.cur > c.k_min && c.promotes > 0, "(pre) warmed to a deep ceiling");
        c.reset();
        CHECK(c.cur == c.k_min, "reset returns to k_min");
        CHECK(c.promotes == 0 && c.demotes == 0, "reset clears counters");
        bool rz = true;
        for (int i = 0; i < 8; i++) rz = rz && c.rounds[i] == 0 && c.sat[i] == 0.f;
        CHECK(rz, "reset clears per-level rounds and sat");
        CHECK(c.hi == 0.42f && c.k_max == 7, "tunables survive reset");
        for (int i = 0; i < 41; i++) c.update(c.cur, c.cur, c.cur + 1);
        CHECK(c.cur == 7, "post-reset request re-earns depth normally");
    }
    printf(fails ? "%d FAILED\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
