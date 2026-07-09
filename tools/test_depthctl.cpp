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
        CHECK(c.cur == 4 && c.sat_ema == 0.f && c.yield_ema == 1.f, "init: cur=4, EMAs at rest");
        c.update(-1, -1, 5);
        CHECK(c.cur == 4 && c.rounds4 == 0, "md_used<0 is a no-op");
    }
    { // promote timing: saturated depth-4 stream crosses hi=0.5 on round 11
        DepthCtl c;
        int promoted_at = -1;
        for (int i = 1; i <= 16 && promoted_at < 0; i++) {
            c.update(4, 4, 5); // n == md_used+1: ceiling saturated
            if (c.cur == 5) promoted_at = i;
        }
        CHECK(promoted_at == 11, "promote on saturated round 11 (1-(15/16)^11 >= .5)");
        CHECK(c.promotes == 1 && c.rounds4 == 11, "promote/rounds4 counters");
        CHECK(std::fabs(c.yield_ema - 2.f * c.lo) < 1e-6f, "promote seeds yield_ema = 2*lo");
    }
    { // depth-4 misses decay sat_ema; no promote
        DepthCtl c;
        for (int i = 0; i < 8; i++) c.update(4, 4, 5);
        float peak = c.sat_ema;
        for (int i = 0; i < 8; i++) c.update(4, 0, 1); // n=1: far from ceiling
        CHECK(c.sat_ema < peak && c.cur == 4, "unsaturated rounds decay sat_ema, no promote");
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
        CHECK(demoted_at == 11, "demote on miss round 11 (.2*(15/16)^11 < .1)");
        CHECK(c.demotes == 1 && c.sat_ema == 0.f, "demote zeroes sat_ema");
    }
    { // level exclusivity: depth-4 rounds never touch yield_ema, depth-5 never sat_ema
        DepthCtl c;
        float y0 = c.yield_ema;
        for (int i = 0; i < 5; i++) c.update(4, 4, 5);
        CHECK(c.yield_ema == y0, "depth-4 rounds leave yield_ema untouched");
        float s0 = c.sat_ema;
        c.update(5, 5, 6);
        CHECK(c.sat_ema == s0 && c.rounds5 == 1, "depth-5 rounds leave sat_ema untouched");
    }
    { // depth-5 hits hold the ceiling: yield_ema recovers toward 1
        DepthCtl c;
        for (int i = 0; i < 11; i++) c.update(4, 4, 5);
        for (int i = 0; i < 64; i++) c.update(5, 5, 6); // 5th lane accepted
        CHECK(c.cur == 5 && c.yield_ema > 0.9f, "sustained accepts keep depth-5");
    }
    // ---- accept-gate Phase 1 (BUILDLOG 2026-07-08 Phase 0 measurement) ----
    { // yield evidence is CONDITIONAL on the 5th lane firing (cap == ceiling);
      // unfired rounds say nothing about the deep lane and barely pay for it
      // under early-exit. Unconditional p(n=6) ~ y5*fired sat ABOVE lo on
      // traffic where depth-5 measured -1.7% (docs61k).
        DepthCtl c;
        for (int i = 0; i < 11; i++) c.update(4, 4, 5);
        float y0 = c.yield_ema;
        for (int i = 0; i < 32; i++) c.update(5, 3, 4); // depth-5, lane 5 unfired
        CHECK(c.yield_ema == y0 && c.cur == 5, "unfired lane 5: no yield evidence, no demote");
        c.update(5, 5, 5); // fired, missed
        CHECK(c.yield_ema < y0, "fired+missed decays yield_ema");
    }
    { // promote seed clamps at 1.0 once lo passes 0.5
        DepthCtl c;
        c.lo = 0.6f;
        for (int i = 0; i < 11; i++) c.update(4, 4, 5);
        CHECK(c.cur == 5 && c.yield_ema <= 1.f, "promote seed = min(1, 2*lo)");
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
    printf(fails ? "%d FAILED\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
