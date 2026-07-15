// CPU unit tests for src/conductor.h -- continuous-batching P1 trim policy
// (plan 2026-07-14 Task 6) + the ConductorCore scheduling skeleton (Task 9).
// No CUDA; pins q27::trim_widths() and the membership/round-boundary logic
// so the fused round's width arbitration and the conductor's scheduling are
// reviewable in isolation from the GPU pieces.
//
//   g++ -O2 -std=c++17 -Wall -Wextra -I src tools/test_conductor.cpp
//     -o build/test_conductor && build/test_conductor   (one line)
//
// trim_widths semantics pinned (design doc Decisions item 3 + Task 6 DEFINE):
//   - repeatedly decrement the CURRENT WIDEST lane-count until sum <= cap
//   - ties: suffix lanes first, then higher slot index (one deterministic
//     victim per step -- same inputs always trim identically)
//   - floor 2: a lane is never decremented below 2 (no width-1 gemv); if all
//     trimmable lanes sit at/below floor the cap is unsatisfiable and the
//     function returns with sum > cap (caller's problem, not an infinite loop)
//   - k <= 1 returns immediately (solo bypasses fusion; is_suffix may be null)
#include "../src/conductor.h"

#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(cond, name)                                          \
    do {                                                           \
        bool ok = (cond);                                          \
        printf("  %-58s %s\n", name, ok ? "PASS" : "FAIL");        \
        if (!ok) fails++;                                          \
    } while (0)

static bool eq(const int* a, const int* b, int k) {
    for (int i = 0; i < k; i++)
        if (a[i] != b[i]) return false;
    return true;
}

int main() {
    { // fits: under cap, untouched
        int w[] = {4, 5};
        const bool s[] = {false, false};
        q27::trim_widths(w, s, 2, 12);
        int e[] = {4, 5};
        CHECK(eq(w, e, 2), "fits: {4,5} cap 12 unchanged");
    }
    { // exactly at cap: no trim (boundary of the sum<=cap loop condition)
        int w[] = {6, 6};
        const bool s[] = {false, false};
        q27::trim_widths(w, s, 2, 12);
        int e[] = {6, 6};
        CHECK(eq(w, e, 2), "boundary: {6,6} cap 12 unchanged (sum==cap)");
    }
    { // overflow trims widest first; the {8,7}->{7,7} tie then goes to the
      // higher slot index (neither suffix), landing {6,6} not {5,7}
        int w[] = {8, 7};
        const bool s[] = {false, false};
        q27::trim_widths(w, s, 2, 12);
        int e[] = {6, 6};
        CHECK(eq(w, e, 2), "widest-first: {8,7} cap 12 -> {6,6}");
    }
    { // suffix absorbs all trim before the gated lane loses any (plan case:
      // the suffix lane is the widest at every step until they meet)
        int w[] = {12, 6};
        const bool s[] = {true, false};
        q27::trim_widths(w, s, 2, 12);
        int e[] = {6, 6};
        CHECK(eq(w, e, 2), "suffix-first: {12(sfx),6} cap 12 -> {6,6}");
    }
    { // equal-width tie between a gated and a suffix lane: suffix loses first
      // even though it has the LOWER slot index (suffix rank beats index rank)
        int w[] = {8, 8};
        const bool s[] = {true, false};
        q27::trim_widths(w, s, 2, 15);
        int e[] = {7, 8};
        CHECK(eq(w, e, 2), "tie rule: sfx before gated: {8(sfx),8} cap 15 -> {7,8}");
    }
    { // floor 2: k=4 all-suffix round-robins down from the highest index and
      // stops exactly at cap, well above the floor
        int w[] = {12, 12, 12, 12};
        const bool s[] = {true, true, true, true};
        q27::trim_widths(w, s, 4, 16);
        int e[] = {4, 4, 4, 4};
        CHECK(eq(w, e, 4), "floor path: all-sfx {12,12,12,12} cap 16 -> {4,4,4,4}");
    }
    { // k=1 never trims (solo bypasses fusion anyway); is_suffix legal as null
      // because the function must return before reading it
        int w[] = {16};
        q27::trim_widths(w, nullptr, 1, 12);
        CHECK(w[0] == 16, "k=1: {16} cap 12 untouched (nullptr is_suffix ok)");
    }
    // ---- extra edges (load-bearing for the Task 9 round loop) ----
    { // k=0: no lanes, no reads, no crash
        q27::trim_widths(nullptr, nullptr, 0, 12);
        CHECK(true, "k=0: no-op, no deref");
    }
    { // cap already violated by floors: nothing trimmable -> must TERMINATE
      // and leave the floors intact (sum>cap is the caller's problem)
        int w[] = {2, 2, 2};
        const bool s[] = {false, false, false};
        q27::trim_widths(w, s, 3, 4);
        int e[] = {2, 2, 2};
        CHECK(eq(w, e, 3), "unsatisfiable: {2,2,2} cap 4 terminates unchanged");
    }
    { // partial floor: the one trimmable lane drops to 2, then we stop even
      // though sum (6) still exceeds cap (4) -- floor beats cap
        int w[] = {2, 5, 2};
        const bool s[] = {false, false, false};
        q27::trim_widths(w, s, 3, 4);
        int e[] = {2, 2, 2};
        CHECK(eq(w, e, 3), "floor beats cap: {2,5,2} cap 4 -> {2,2,2}");
    }
    { // sub-floor lane is never a victim and never raised: floor means "do not
      // decrement below 2", not "clamp up to 2"
        int w[] = {1, 9};
        const bool s[] = {false, false};
        q27::trim_widths(w, s, 2, 6);
        int e[] = {1, 5};
        CHECK(eq(w, e, 2), "sub-floor lane untouched: {1,9} cap 6 -> {1,5}");
    }
    { // all-equal, no suffix: deterministic round-robin from the highest index
        int w[] = {5, 5, 5, 5};
        const bool s[] = {false, false, false, false};
        q27::trim_widths(w, s, 4, 18);
        int e[] = {5, 5, 4, 4};
        CHECK(eq(w, e, 4), "all-equal tie determinism: {5,5,5,5} cap 18 -> {5,5,4,4}");
    }
    { // all-equal, mixed classes: BOTH suffix lanes lose before any gated lane
      // does, highest-index suffix first
        int w[] = {5, 5, 5, 5};
        const bool s[] = {false, true, false, true};
        q27::trim_widths(w, s, 4, 18);
        int e[] = {5, 4, 5, 4};
        CHECK(eq(w, e, 4), "mixed tie: sfx lanes {1,3} absorb {5,5,5,5} cap 18 -> {5,4,5,4}");
    }
    { // widest-first is the PRIMARY key: a wide gated lane trims before a
      // narrower suffix lane (suffix rank only breaks ties)
        int w[] = {6, 10};
        const bool s[] = {true, false};
        q27::trim_widths(w, s, 2, 14);
        int e[] = {6, 8};
        CHECK(eq(w, e, 2), "width beats class: {6(sfx),10} cap 14 -> {6,8}");
    }

    // ---- ConductorCore scheduling skeleton (Task 9) ----
    // Fake member implementing the duck-typed MemberT surface. `budget` is
    // rounds until a natural finish (post_round's done, modeled in the round
    // hooks); `cancel` mirrors DecodeTask::cancel (A3: observed at the
    // pre-check boundary only, finish path runs there).
    struct FakeMember {
        int id = 0;
        int budget = 0;
        bool cancel = false;
        int want = 4;
        bool suffix = false;
        int granted = -1;
        int rounds = 0;      // rounds this member actually ran (solo or fused)
        bool finished = false; // finish path ran (pre_round or a round's done)
        bool left = false;     // on_leave ran
        bool pre_round() {
            if (cancel || budget <= 0) {
                finished = true; // models finish_decode("cancelled"/"n_max")
                return false;
            }
            return true;
        }
        int want_width() { return want; }
        bool round_is_suffix() const { return suffix; }
        void set_granted(int w) { granted = w; }
    };
    {
        q27::ConductorCore<FakeMember> core(12);
        FakeMember A{}, B{}, C{}, D{};
        A.id = 1; A.budget = 6; A.want = 4;
        B.id = 2; B.budget = 2; B.want = 5;
        C.id = 3; C.budget = 5; C.want = 6;
        D.id = 4; D.budget = 2; D.want = 2;
        int solo_calls = 0, fused_calls = 0;
        std::vector<std::vector<int>> round_ids;     // member ids per round
        std::vector<std::vector<int>> round_granted; // granted widths per fused round
        core.solo_round = [&](FakeMember& mm) {
            solo_calls++;
            round_ids.push_back({mm.id});
            mm.rounds++;
            mm.budget--;
            if (mm.budget <= 0) {
                mm.finished = true;
                return true; // done (post_round would return false)
            }
            return false;
        };
        core.fused_round = [&](FakeMember** ms, const int* granted, const bool* sfx,
                               int k, bool* done) {
            (void)sfx;
            fused_calls++;
            std::vector<int> ids, gr;
            for (int i = 0; i < k; i++) {
                ids.push_back(ms[i]->id);
                gr.push_back(granted[i]);
                ms[i]->rounds++;
                ms[i]->budget--;
                done[i] = ms[i]->budget <= 0;
                if (done[i]) ms[i]->finished = true;
            }
            round_ids.push_back(ids);
            round_granted.push_back(gr);
            // MID-ROUND JOIN (boundary test): D registers while round 2 is in
            // flight; it must be absent from THIS round and join the next.
            if (fused_calls == 1) core.join(&D);
        };
        core.on_leave = [&](FakeMember& mm) { mm.left = true; };

        // round 1: A alone -> solo hook, not fused
        core.join(&A);
        int live = core.round();
        CHECK(live == 1 && solo_calls == 1 && fused_calls == 0 &&
                  round_ids.back() == std::vector<int>({1}),
              "core: k==1 takes the solo hook");
        // round 2: B and C joined at the boundary; want {4,5,6} = 15 > 12
        // overflows -> trim to {4,4,4} (widest-first, higher-index ties);
        // D joins MID-ROUND from inside the hook.
        core.join(&B);
        core.join(&C);
        live = core.round();
        CHECK(fused_calls == 1 && round_ids.back() == std::vector<int>({1, 2, 3}),
              "core: joins land at the boundary; k==3 takes the fused hook");
        CHECK(round_granted.back() == std::vector<int>({4, 4, 4}),
              "core: overflow round trimmed {4,5,6} cap 12 -> {4,4,4}");
        CHECK(live == 3, "core: mid-round join deferred (D absent from round 2)");
        // round 3: D is a member now; wants {3,3,3,2} = 11 <= 12 -> granted
        // == want (trim untouched under the cap). B's budget hits 0 -> done.
        A.want = B.want = C.want = 3;
        live = core.round();
        CHECK(round_ids.back() == std::vector<int>({1, 2, 3, 4}),
              "core: mid-round join lands the NEXT round");
        CHECK(round_granted.back() == std::vector<int>({3, 3, 3, 2}),
              "core: no-overflow round leaves want untouched (no trim)");
        CHECK(live == 3 && B.finished && B.left,
              "core: natural finish leaves at the round's end (B done)");
        // round 4: C cancelled mid-flight -> pre-check boundary removes it
        // (finish path + on_leave), no further rounds for C; A+D run fused.
        C.cancel = true;
        int c_rounds = C.rounds;
        live = core.round();
        CHECK(C.finished && C.left && C.rounds == c_rounds,
              "core: cancelled member gets the done-path, no further rounds");
        CHECK(round_ids.back() == std::vector<int>({1, 4}) && live == 1,
              "core: round 4 = {A,D} fused; D finishes (budget 2)");
        // rounds 5-6: A alone again -> solo; finishes at budget 0
        live = core.round();
        CHECK(round_ids.back() == std::vector<int>({1}) && live == 1,
              "core: back to solo at k==1 (round 5)");
        live = core.round();
        CHECK(live == 0 && A.finished && A.left && D.finished && D.left,
              "core: A finishes (round 6); all members left, core idle");
        // idle round: no members, no hooks
        int sc = solo_calls, fc = fused_calls;
        live = core.round();
        CHECK(live == 0 && solo_calls == sc && fused_calls == fc,
              "core: idle round runs no hooks");
        // membership-only-at-boundaries, summarized: the per-round member
        // lists above are exactly {1},{1,2,3},{1,2,3,4},{1,4},{1},{1}
        const std::vector<std::vector<int>> expect = {
            {1}, {1, 2, 3}, {1, 2, 3, 4}, {1, 4}, {1}, {1}};
        CHECK(round_ids == expect, "core: full round-membership trace matches");
    }
    { // solo member cancelled at the boundary: no solo hook, done-path only
        q27::ConductorCore<FakeMember> core(12);
        FakeMember A{};
        A.id = 1; A.budget = 3; A.cancel = true;
        bool left = false;
        int solo_calls = 0;
        core.solo_round = [&](FakeMember&) { solo_calls++; return false; };
        core.fused_round = [&](FakeMember**, const int*, const bool*, int, bool*) {};
        core.on_leave = [&](FakeMember& mm) { left = mm.left = true; };
        core.join(&A);
        int live = core.round();
        CHECK(live == 0 && solo_calls == 0 && A.finished && left,
              "core: pre-check cancel beats the solo hook");
    }
    printf(fails ? "test_conductor: %d FAILED\n" : "test_conductor: ALL PASS\n", fails);
    return fails ? 1 : 0;
}
