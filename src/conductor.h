#pragma once
// Continuous-batching P1 (design doc docs/plans/2026-07-14-continuous-batching
// -design.md): host-side conductor pieces. This file starts with the LANE
// POLICY only (design Decisions item 3): per-engine gates + cap -- each engine
// keeps its own P12-gated width and suffix width; when the union of requested
// widths exceeds the fused round's cap (W_MAX), trim the WIDEST requesters
// first, suffix lanes before gated lanes. Rationale: the widest lane paid the
// least per marginal slot (deep speculative lanes are the cheapest to lose),
// and suffix lanes are opportunistic re-emission bets while gated lanes carry
// depthctl-earned evidence -- so at equal width the bet yields before the
// earned width does.
//
// The Conductor (Task 9: registry, round loop, token queues) calls a THIN
// ENGINE-OWNED surface only: the entrypoints (solo_view()/pre/mix/post/
// ffn_pair/qx5/mm5/tails/T()/set_round_width/draft_and_gate/suffix_propose/
// the P2a step-granular draft pieces (draft_sample_bootstrap/draft_md_used/
// draft_step_launch/draft_margin/draft_floor_topup)/
// the P2c fused-draft pieces (mtp_step_view/draft_step_prep/mtp_pre/
// mtp_attn/mtp_post/mtp_tail/draft_margin_d2h)/
// commit_outcome/pre_round/post_round/decode_step/finish_decode -- the last
// only from the A2 catch epilogue, fail_member below) plus the named
// ACCESSORS engine.cuh
// declares for the conductor (shared_dm/is_attn_layer/fast_head_on/
// vgemm_ws/round_width/stream/gate_theta/outcome_dev/end_reason, each with a why-
// comment at its declaration). No friends; every raw-member need is met by
// adding an accessor in engine.cuh, never by reaching into the engine from
// this header (consensus addendum A4). The trim policy, TokenQueue and the
// ConductorCore scheduling skeleton below are pure host code (deterministic,
// CPU-tested in tools/test_conductor.cpp); the fused round + the real
// Conductor further down are CUDA-only and compile away under plain g++
// (the __CUDACC__ guard), so the CPU unit test never sees them.

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace q27 {

// Trim requested verify widths want[0..k) in place until sum(want) <= cap.
// One deterministic victim per step: the current widest lane; ties broken
// suffix-first, then higher slot index. Floor 2 -- never decrement a lane
// below 2 (engine floor: no width-1 gemv); a lane already under the floor is
// never a victim and never raised. If every lane sits at/below the floor and
// the sum still exceeds cap, return with sum > cap rather than loop -- an
// unsatisfiable cap is the caller's admission-control problem. k <= 1 returns
// before touching either array (solo bypasses fusion; is_suffix may be null).
inline void trim_widths(int* want, const bool* is_suffix, int k, int cap) {
    if (k <= 1) return;
    int sum = 0;
    for (int i = 0; i < k; i++) sum += want[i];
    while (sum > cap) {
        int pick = -1; // widest trimmable; ascending scan => later equal
                       // candidates of the same class win (higher-index rule)
        for (int i = 0; i < k; i++) {
            if (want[i] <= 2) continue; // at/under floor: never a victim
            // width tie: i wins unless it would demote a suffix pick to a
            // gated one (sfx beats gated; same class -> later slot, i > pick)
            if (pick < 0 || want[i] > want[pick] ||
                (want[i] == want[pick] && (is_suffix[i] || !is_suffix[pick])))
                pick = i;
        }
        if (pick < 0) return; // all floored: cap unsatisfiable, stop
        want[pick]--;
        sum--;
    }
}

// ---------------------------------------------------------------------------
// P1 Task 9: per-request token queue. Producer = the conductor thread (one
// push per emitted token, from post_round's sink, at round boundaries);
// consumer = the request thread (pop, then SSE-write OUTSIDE any lock).
// push/close are NON-BLOCKING (unbounded buffer, O(1) amortized + a notify):
// gate-ownership invariant A7 -- the conductor must never block on a request
// thread while holding the GPU gate. Error slot: A2 keeps CUDA failures
// process-fatal, so `error` only ever carries a HOST-side failure the request
// thread should surface instead of a normal finish. Setters: the conductor's
// A2 unwind (fail_member -- member bookkeeping threw) and the M4 registration
// refusal; the server's batch_generate reads it back and surfaces it
// (gs.end="error" in the [req] line, 500 when nothing was emitted, an SSE
// `error` event on the Anthropic stream). It rides the same close() wakeup.
struct TokenQueue {
    struct Token {
        int id = -1;
        bool forced = false;
    };
    void push(int id, bool forced) {
        {
            std::lock_guard<std::mutex> lk(m);
            buf.push_back({id, forced});
        }
        cv.notify_one();
    }
    // Close with the finish reason (Engine gs.end: "eos"/"n_max"/...). The
    // reason pointer must be a string literal / static (never freed).
    // LIFETIME RULE (review H1): the consumer may destroy this queue the
    // MOMENT it observes `closed` (pop() returning false unblocks the
    // request thread, whose stack owns the queue) -- so once the mutex
    // unlocks, NO queue member, cv included, may be touched. The notify
    // therefore runs INSIDE the locked region: a waiter cannot re-check the
    // predicate until the lock drops, and a consumer entering pop() blocks
    // on the mutex, so the notify always executes on a live cv. (A notify
    // after the unlock could race the consumer observing `closed` via an
    // earlier wakeup and freeing the queue: use-after-free.)
    void close(const char* why) {
        std::lock_guard<std::mutex> lk(m);
        closed = true;
        reason = why;
        cv.notify_one(); // under the lock -- see LIFETIME RULE above
    }
    // Close with a host-side error instead of a normal finish (A2 note
    // above). `what` is COPIED into the queue (the A2 unwind passes
    // e.what(), which dies with the exception object), unlike close()'s
    // static-only reason.
    void fail(const char* what) {
        std::lock_guard<std::mutex> lk(m);
        error_owned = what;
        error = error_owned.c_str();
        closed = true;
        reason = "error";
        cv.notify_one(); // under the lock -- see close()'s LIFETIME RULE
    }
    // Drain everything available, APPENDING to out; blocks until at least one
    // token arrives or the queue closes. Returns false only when the queue is
    // closed AND nothing was delivered -- `while (q.pop(out)) {}` drains a
    // stream to completion.
    bool pop(std::vector<Token>& out) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !buf.empty() || closed; });
        bool got = !buf.empty();
        out.insert(out.end(), buf.begin(), buf.end());
        buf.clear();
        return got || !closed;
    }
    bool pop(std::vector<int>& out) {
        std::vector<Token> tokens;
        const bool more = pop(tokens);
        for (const auto& token : tokens) out.push_back(token.id);
        return more;
    }
    const char* finish_reason() {
        std::lock_guard<std::mutex> lk(m);
        return reason;
    }
    const char* error_or_null() {
        std::lock_guard<std::mutex> lk(m);
        return error;
    }

private:
    std::mutex m;
    std::condition_variable cv;
    std::vector<Token> buf;
    bool closed = false;
    const char* reason = "";
    const char* error = nullptr;
    std::string error_owned; // backing store for `error` (fail() copies)
};

// ---------------------------------------------------------------------------
// P1 Task 9: scheduling core -- membership, round boundaries and width
// arbitration ONLY. Pure host logic, templated on the member type so the CPU
// unit test (tools/test_conductor.cpp) drives it with fake engines; the real
// Conductor (CUDA section below) instantiates it with its Member. Duck-typed
// MemberT surface:
//   bool pre_round();        false => member is DONE (cancel/budget/ctx);
//                            its finish path already ran inside
//   bool parked() const;     true after a pre_round() that returned true but
//                            may not write this round (incremental KV could
//                            not grow): skipped this round, kept as a member
//   int  want_width();       draft phase (real: suffix_propose() or
//                            draft_and_gate() on the member engine's OWN
//                            stream); returns the want width, >= 2.
//                            P2a: only the FALLBACK when the draft_widths
//                            hook below is unset -- the real conductor
//                            installs the hook so gated members' draft
//                            steps interleave across engines
//   bool round_is_suffix();  this round's proposal class (trim + GEMM policy)
//   void set_granted(int w); install the post-trim width (set_round_width)
// Hooks (owned by the wrapper):
//   solo_round(m) -> bool    k==1: today's decode_step path; true = done
//   fused_round(ms, granted, sfx, k, done)  k>=2: fused verify + commit +
//                            post-round per member; fills done[0..k)
//   on_leave(m)              member left (finish path already ran): close its
//                            queue, release per-member resources
// Membership INVARIANT (design "Scheduler"): the member set mutates only
// inside round() -- joins drain at the top, leavers (pre-check failures and
// members whose round reported done) are erased before round() returns.
// A join() during a round (from another thread in the real conductor, or
// from inside a hook in the test) lands at the NEXT round. Explicit
// NON-GOALS: no fairness machinery, no dynamic priorities -- FIFO joins +
// the trim policy are the whole scheduler.
template <class MemberT>
struct ConductorCore {
    // MAX_K sizes round()'s stack arrays. The LEGAL member ceiling is
    // MAX_K/2 = 8, not 16: floor-2 lanes mean k members need >= 2k union
    // lane slots, and the lane plumbing has W_PLUMB = 16 slots (review L1 --
    // the old comment claimed 16 members were a legal union). MAX_K
    // deliberately EQUALS W_PLUMB (static_assert at the CUDA Conductor
    // below; this pure-host half compiles without cuda_common.h) so the
    // hard k <= MAX_K/2 check in round() is exactly k <= W_PLUMB/2.
    enum { MAX_K = 16 };
    int cap;             // union width cap (the real conductor passes W_MAX)
    explicit ConductorCore(int cap_) : cap(cap_) {}
    std::vector<MemberT*> members; // live set; mutated at round boundaries only
    std::vector<MemberT*> joins;   // staged joins; drained at boundaries
    std::function<bool(MemberT&)> solo_round;
    // Some rounds are intentionally ineligible for fusion (currently the
    // host-forced token queue used for reasoning/tool close transitions).
    // Run one such member through the exact solo decode_step path before
    // drafting any other member; otherwise fused drafting overwrites the
    // staged forced id and the transition never reaches decoder state.
    std::function<bool(MemberT&)> needs_solo_round;

    std::function<void(MemberT**, const int*, const bool*, int, bool*)> fused_round;
    std::function<void(MemberT&)> on_leave;
    // P2a (optional): batch draft hook -- fills want[] and sfx[] for all k
    // members at once so the real conductor can interleave the gated
    // members' draft steps across engines (concurrent draft graphs). When
    // unset (the CPU unit test, hookless embedders), round() falls back to
    // the per-member want_width() calls -- the serial P1 behavior.
    std::function<void(MemberT**, int*, bool*, int)> draft_widths;

    void join(MemberT* m) { joins.push_back(m); }

    // One round. Returns the live member count after the round (0 = idle).
    int round() {
        // boundary: joins land here and only here
        for (MemberT* m : joins) members.push_back(m);
        joins.clear();
        last_idle = false;
        // pre-checks (decode_step's top, A3): a done/cancelled member leaves
        // NOW, before any draft work -- its finish path ran inside
        // pre_round(), it gets its on_leave, and it is absent from this and
        // every later round.
        for (size_t i = 0; i < members.size();) {
            if (!members[i]->pre_round()) {
                MemberT* gone = members[i];
                members.erase(members.begin() + i);
                on_leave(*gone);
            } else {
                i++;
            }
        }
        // Parked members (incremental KV, issue #42 step 2) passed their
        // pre-checks but may not write this round -- their lineage could not
        // grow safely. They stay members, sit this round out, and are
        // re-checked at the next boundary; the round runs over the rest.
        std::vector<MemberT*> act;
        act.reserve(members.size());
        for (MemberT* m : members)
            if (!m->parked()) act.push_back(m);
        auto drop = [&](MemberT* gone) {
            members.erase(std::find(members.begin(), members.end(), gone));
            on_leave(*gone);
        };
        if (needs_solo_round) {
            for (MemberT* mm : act) {
                if (!needs_solo_round(*mm)) continue;
                if (solo_round(*mm)) drop(mm);
                return (int)members.size();
            }
        }
        const int k = (int)act.size();
        if (k == 0) {
            // every live member parked: nothing ran. The wrapper waits for a
            // join / release instead of spinning on the gate.
            last_idle = !members.empty();
            return (int)members.size();
        }
        if (k == 1) {
            // solo fallthrough: byte-for-byte today's path (captured
            // graphs); fusion only engages at >= 2 (design "Scheduler").
            if (solo_round(*act[0])) drop(act[0]);
            return (int)members.size();
        }
        // Batch-formation HARD check (review L1), not an assert: k members at
        // the floor-2 trim need 2k union lanes, so k > MAX_K/2 (== W_PLUMB/2
        // = 8) can never trim under the plumbing and the union view would
        // overflow engine lane arrays -- memory corruption, not a policy
        // miss. Unreachable in the server (its slot count is clamped <= 4);
        // this is for EMBEDDERS driving the class directly, who get a real
        // refuse-to-run (the guardrail posture) even in an NDEBUG build.
        if (k > MAX_K / 2) {
            fprintf(stderr,
                    "conductor: FATAL -- %d members exceed the legal union ceiling %d "
                    "(floor-2 lanes over %d plumbed lane slots). Admission control is "
                    "the caller's job; refusing to form the batch.\n",
                    k, MAX_K / 2, (int)MAX_K);
            abort();
        }
        MemberT* ms[MAX_K];
        int want[MAX_K];
        bool sfx[MAX_K], done[MAX_K];
        for (int i = 0; i < k; i++) ms[i] = act[i];
        // Draft phase (P2a): the batch hook interleaves gated members' draft
        // steps so the engines' chains run concurrently on their own
        // streams; without it, the per-member calls are the serial path.
        // Either way this runs strictly AFTER every pre_round() above and
        // strictly BEFORE trim/set_granted/fused_round below -- the P1
        // ordering, unchanged.
        if (draft_widths) {
            draft_widths(ms, want, sfx, k);
        } else {
            for (int i = 0; i < k; i++) {
                want[i] = ms[i]->want_width();
                sfx[i] = ms[i]->round_is_suffix();
            }
        }
        // trim mutates only on overflow (sum > cap): under the cap, granted
        // == want and untrimmed lanes keep the bitwise contract.
        // Q27_BATCH_DBG=1 (Task 10, A5 trim-active gate): one stderr line per
        // fused round, want->granted per lane (s=suffix, g=gated) -- the
        // trim-fired evidence the gate reads. Single buffered write so lines
        // stay whole against concurrent request-thread stderr.
        static const bool dbg = [] {
            const char* e = getenv("Q27_BATCH_DBG");
            return e && atoi(e) != 0;
        }();
        int want0[MAX_K];
        if (dbg)
            for (int i = 0; i < k; i++) want0[i] = want[i];
        trim_widths(want, sfx, k, cap);
        if (dbg) {
            char line[256];
            int off = snprintf(line, sizeof line, "[bat] k=%d cap=%d", k, cap);
            for (int i = 0; i < k && off < (int)sizeof line - 16; i++)
                off += snprintf(line + off, sizeof line - off, " %d->%d%c", want0[i],
                                want[i], sfx[i] ? 's' : 'g');
            fprintf(stderr, "%s\n", line);
        }
        for (int i = 0; i < k; i++) ms[i]->set_granted(want[i]);
        fused_round(ms, want, sfx, k, done);
        // leave-on-done at the boundary (EOS/budget/client-stop this round)
        for (int i = k - 1; i >= 0; i--)
            if (done[i]) drop(ms[i]);
        return (int)members.size();
    }
    // true when the last round() ran no member because every live one was
    // parked (incremental KV); false after any round that did work
    bool last_idle = false;
};

} // namespace q27

// ---------------------------------------------------------------------------
// P1 Task 8: the fused verify round (design doc "Fused round anatomy" 3-4).
// CUDA TUs only -- tools/test_conductor.cpp (g++, CPU) compiles just the trim
// policy above. Engine is header-only, so pulling it in here keeps include
// order a non-issue for future users (server.cu, fused_smoke.cu).
#ifdef __CUDACC__
#include <chrono> // P3 T3: capture/instantiate walls ([gcache] telemetry)
#include <memory>
#include <thread>

#include "api_common.h" // GpuGate (the R1b FIFO ticket lock, unchanged)
#include "engine.cuh"

namespace q27 {

// Union slot u -> (engines[map[u].eng], lane map[u].lane). Every LaneView
// field of slot u is a POINTER INTO THE OWNING ENGINE'S OWN LANE BUFFERS, so
// the union weight sweep writes each engine's activations in place and the
// per-engine mixers/tails then read their members untouched. Slots >= view.vw
// keep engines[0]'s solo padding (same "never dereferenced" contract as the
// solo view's own high slots).
struct UnionView {
    Engine::LaneView view;
    struct { int eng; int lane; } map[W_PLUMB];
    int k = 0; // engines in the union
};

// Build the union view over engines es[0..k) at granted widths w[0..k).
// view.vw = sum(w); view.stm = the conductor stream (cstm); view.vgemm_ws =
// es[0]'s workspace. A9 sizing: d_vgemm_ws is allocated from
// vgemm_ws_bytes_model() (engine.cuh init), which sizes z * W_PLUMB * rows
// floats -- i.e. W_PLUMB (16) LANES, not W_MAX -- so any engine's workspace
// covers any legal union; the structural bound asserted here is therefore
// sum(w) <= W_PLUMB (the lane plumbing / vgemm NT tile). The conductor's
// POLICY cap stays W_MAX via trim_widths.
//
// GEMM-family policy (P1 Task 9, from the Task 8 finding): mm5 keys
// vgemm-vs-GEMV on view.vw >= view.gemm_min, so a union crossing the member
// threshold (9) would fork the numeric family away from what each lane's
// SOLO round took (vgemm==gemv was deliberately never claimed bitwise).
// is_suffix[m] says whether member m fires a suffix round THIS round; the
// union's threshold is set so every lane stays on its solo family:
//   all gated  -> 99: force the GEMV family. Gated solo widths (<= 8) always
//                 took the GEMV, and the GEMV lanes are N-invariant
//                 (ninv_test), so untrimmed gated lanes stay BITWISE.
//   all suffix -> 2: force vgemm. Solo suffix rounds verify at sfx_width()
//                 (the serving config runs 12 >= gemm_min) on k_vgemm, and
//                 vgemm lanes are N-invariant, so untrimmed suffix lanes
//                 stay bitwise. (A legacy-narrow suffix config, sfx_width()
//                 < 9, would have taken the GEMV solo -- that fork is the
//                 documented tolerance class; the serving target is wide.)
//   mixed      -> 99: gated lanes carry the bitwise contract, suffix lanes
//                 numeric-fork to the GEMV family (documented tolerance
//                 class). trim_widths evicts suffix lanes first, so mixed
//                 unions are rare and the fork is bounded to opportunistic
//                 re-emission bets, never depthctl-earned gated lanes.
// k >= 3 DEFAULT OVERRIDE (2026-08-16 C-sweep, BUILDLOG (g)): at three or
// more members the all-gated/mixed union goes vgemm (threshold 2) anyway.
// The bitwise-preserving GEMV family register-cliffs in union width (q4
// ffn/call 0.055 ms @N=12 -> 0.113 @N=16) and cost the whole C=7 rolloff;
// on vgemm the ladder is monotone (C=3..7: +16..37%, peak 309.8 -> 400.0
// t/s). Price: gated fused lanes at k >= 3 surrender bitwise-vs-solo for
// the documented rel~1e-6 vgemm class. Determinism, stated precisely: each
// kernel is run-to-run deterministic, but the family now flips with k at
// the 2/3 boundary and k changes mid-request as members join/leave -- so at
// C >= 3 a request's emitted tokens depend on the per-round k SEQUENCE,
// which is arrival-timing-dependent. Solo replay of a fused greedy request
// is not expected to match, and multi-slot token-identity A/Bs at 3+ slots
// require Q27_BATCH_GEMM=0. k <= 2 keeps the GEMV and the full bitwise
// contract -- also the regime where the GEMV measured faster (07-15 k=2
// A/B: vgemm -1.5%). The policy keys on member count k, not union width:
// k=2 can reach vw=12 (trim cap) on the GEMV, which the k=2 A/B covered;
// keying on vw would surrender the measured-winning bitwise regime.
// GRAPH SAFETY: gemm_min is part of the view identity (union_view_eq
// compares it), so a per-round family change can never replay a stale exec;
// it captures a fresh variant instead.
// Q27_BATCH_GEMM overrides the default: =1 forces always-vgemm (threshold
// 2) at ANY k; =0 restores the pre-C-sweep default (gated/mixed on GEMV at
// any k -- all-suffix unions STILL take vgemm, their solo family; this is
// not a vgemm kill switch); unset or empty = the k >= 3 auto above.
inline UnionView build_union_view(Engine** es, const int* w, int k, cudaStream_t cstm,
                                  const bool* is_suffix) {
    static const int batch_gemm = [] { // -1 = auto (vgemm at k >= 3), 0 = never, 1 = always
        const char* e = getenv("Q27_BATCH_GEMM");
        if (!e || !*e) return -1;
        char* end = nullptr;
        long v = strtol(e, &end, 10);
        if (end == e) { // non-numeric ("auto", "on", ...): loud, then auto --
            fprintf(stderr, // silently pinning 0 here would re-enter the C=7 rolloff regime
                    "Q27_BATCH_GEMM='%s' not numeric; using auto (vgemm at k >= 3)\n", e);
            return -1;
        }
        return v != 0 ? 1 : 0;
    }();
    assert(k >= 1 && k <= W_PLUMB);
    // CRITICAL union-sweep precondition: every engine's lanes 0..w[m]-1 are
    // engine-owned and distinct -- an engine listed twice would hand two
    // union slots the SAME buffers and the sweep would double-write them.
    for (int a = 0; a < k; a++)
        for (int b = a + 1; b < k; b++) assert(es[a] != es[b]);
    // One weight set serves every lane: the fused sweep reads weights (and
    // attn_layer geometry) through es[0], so all members must share the
    // DeviceModel (the server's shared_model/shared_dm construction).
    for (int m = 1; m < k; m++) assert(&es[m]->shared_dm() == &es[0]->shared_dm());
    UnionView uv;
    uv.k = k;
    // start from es[0]'s solo view: sane padding for slots >= union width
    uv.view = es[0]->solo_view();
    int u = 0;
    for (int m = 0; m < k; m++) {
        assert(w[m] >= 2 && w[m] <= W_MAX); // per-engine: logits2/roles are W_MAX-lane
        // granted width must already be installed (mix/tail read member vw)
        assert(es[m]->round_width() == w[m]);
        const Engine::LaneView sv = es[m]->solo_view();
        for (int j = 0; j < w[m]; j++, u++) {
            assert(u < W_PLUMB);
            uv.map[u] = {m, j};
            uv.view.x1[u] = sv.x1[j];       uv.view.qkv[u] = sv.qkv[j];
            uv.view.z[u] = sv.z[j];         uv.view.alpha[u] = sv.alpha[j];
            uv.view.betar[u] = sv.betar[j]; uv.view.g[u] = sv.g[j];
            uv.view.beta[u] = sv.beta[j];   uv.view.o[u] = sv.o[j];
            uv.view.og[u] = sv.og[j];       uv.view.y[u] = sv.y[j];
            uv.view.qg[u] = sv.qg[j];       uv.view.kbuf[u] = sv.kbuf[j];
            uv.view.vbuf[u] = sv.vbuf[j];   uv.view.attnout[u] = sv.attnout[j];
            uv.view.ffn_g[u] = sv.ffn_g[j]; uv.view.ffn_u[u] = sv.ffn_u[j];
            uv.view.h[u] = sv.h[j];         uv.view.lg[u] = sv.lg[j];
            uv.view.xq[u] = sv.xq[j];
            uv.view.dv[u] = sv.dv[j];
            uv.view.vtok.p[u] = sv.vtok.p[j];
            uv.view.pos.p[u] = sv.pos.p[j];
        }
    }
    assert(u <= W_PLUMB); // A9: es[0]'s vgemm_ws is sized for W_PLUMB lanes
    assert(es[0]->vgemm_ws() != nullptr);
    uv.view.vw = u;
    uv.view.stm = cstm;
    uv.view.vgemm_ws = es[0]->vgemm_ws();
    // union-class GEMM family (policy block in the header comment above):
    // all-suffix -> vgemm (2); all-gated OR mixed -> GEMV (99, gated lanes
    // keep the bitwise contract; mixed unions' suffix lanes tolerance-fork)
    // at k <= 2, vgemm (2) at k >= 3 per the C-sweep default override.
    bool all_sfx = true; // k >= 1 asserted above, so this is never vacuous
    for (int m = 0; m < k; m++) all_sfx &= is_suffix[m];
    const bool want_vgemm = all_sfx || batch_gemm == 1 ||
                            (batch_gemm == -1 && k >= 3);
    uv.view.gemm_min = want_vgemm ? 2 : 99;
    return uv;
}

// P3 T3 hit-guard mirror (B8 discipline, plan 2026-07-16-batch-p3-capture.md):
// field-wise equality over everything build_union_view derives. Before ANY
// cached-exec replay, the conductor re-derives the union view from live host
// state and compares it against the capture-stored copy -- a mismatch means
// the exec's baked pointer program no longer describes reality, and it must
// never launch. Field-wise rather than one struct memcmp because LaneView/
// UnionView carry alignment padding (indeterminate bytes) and map[] slots at
// and beyond vw are never written; comparing those would fault good hits.
// MIRROR WARNING (the fused_verify_round skeleton discipline, review M3): a
// new pointer field added to LaneView or UnionView MUST be compared here too,
// or the guard goes blind to it.
inline bool union_view_eq(const UnionView& a, const UnionView& b) {
    const Engine::LaneView &x = a.view, &y = b.view;
    auto lanes_eq = [](const std::array<float*, W_PLUMB>& p,
                       const std::array<float*, W_PLUMB>& q) {
        return std::memcmp(p.data(), q.data(), sizeof(float*) * W_PLUMB) == 0;
    };
#define Q27_UVEQ(F) if (!lanes_eq(x.F, y.F)) return false;
    Q27_UVEQ(x1) Q27_UVEQ(qkv) Q27_UVEQ(z) Q27_UVEQ(alpha) Q27_UVEQ(betar)
    Q27_UVEQ(g) Q27_UVEQ(beta) Q27_UVEQ(o) Q27_UVEQ(og) Q27_UVEQ(y)
    Q27_UVEQ(qg) Q27_UVEQ(kbuf) Q27_UVEQ(vbuf) Q27_UVEQ(attnout)
    Q27_UVEQ(ffn_g) Q27_UVEQ(ffn_u) Q27_UVEQ(h) Q27_UVEQ(lg)
#undef Q27_UVEQ
    for (int t = 0; t < W_PLUMB; t++) {
        // XQuant is 6 pointers (nat/scale/eo/nat64/s64/isum -- kernels.cuh:17),
        // no padding: memcmp over sizeof covers ALL of them; do NOT "simplify"
        // to per-field compares (a new field would silently blind the guard)
        if (std::memcmp(&x.xq[t], &y.xq[t], sizeof(q27k::XQuant))) return false;
        if (x.vtok.p[t] != y.vtok.p[t] || x.pos.p[t] != y.pos.p[t] ||
            x.dv[t] != y.dv[t])
            return false;
    }
    if (x.vgemm_ws != y.vgemm_ws || x.vw != y.vw || x.stm != y.stm ||
        x.gemm_min != y.gemm_min)
        return false;
    if (a.k != b.k) return false;
    for (int u = 0; u < x.vw; u++) // slots >= vw: never written, never read
        if (a.map[u].eng != b.map[u].eng || a.map[u].lane != b.map[u].lane)
            return false;
    return true;
}

// One fused GREEDY verify round over engines es[0..k) at granted widths
// granted[0..k) on the conductor stream cstm. Launch-only body: eager when
// called directly, and the T3 capture target when the conductor wraps it in
// stream capture (Q27_BATCH_GRAPH=1, graph_round -- which then passes
// draft_done/ev_draft_end as nullptr, both hoisted outside the capture).
// Caller contract (the conductor / smoke driver):
//   - each engine's draft phase already ran on ITS OWN stm (draft_and_gate),
//     widths were trimmed, and set_round_width(granted[m]) installed member
//     vw = granted[m] (mixers + tails read it);
//   - draft_done[m] was recorded on es[m]->stm AFTER its last draft launch;
//   - after this returns, the caller D2Hs each engine's d_outcome on cstm,
//     syncs cstm ONCE, then commit_outcome() per engine. Any work the caller
//     later puts on an engine stm must be ordered after that sync (or after
//     an event recorded on cstm).
// Anatomy (mirrors spec_verify_forward, union-vs-per-engine split per the
// design): union embed3; per layer { union rmsnorm3; pre(union weight sweep);
// per-engine mix (SEQUENCE state: KV/GDN roles, serial on cstm at P1); post
// (union); union add3/rmsnorm3; ffn_pair(union); union add3 }; union output
// norm + head into per-slot lg = each engine's own logits2 lanes; per-engine
// greedy tails on cstm.
// SKELETON MIRROR WARNING (review M3): the loop below is a hand-maintained
// COPY of Engine::spec_verify_forward's skeleton (engine.cuh) with the
// union-vs-per-engine split applied. Structural changes THERE (layer-loop
// shape, kernel order, head selection) MUST be mirrored HERE and re-gated
// with fused_smoke (build line in tools/fused_smoke.cu's header).
// pre/post/ffn_pair are Engine methods but read ONLY
// the view + shared weights (dm/T(il)/EPS), so calling them on es[0] with the
// union view sweeps every engine's lanes in one pass -- that is the whole
// point: one weight read serves all slots.
// NOTE mm5 dispatch: mm5 compares the union view's vw against
// view.gemm_min, which build_union_view sets per union class AND member
// count (policy block above build_union_view): k <= 2 gated/mixed keep the
// solo family (GEMV, bitwise); all-suffix, or k >= 3, or Q27_BATCH_GEMM=1
// go vgemm (tolerance class -- the 2026-08-16 C-sweep default);
// Q27_BATCH_GEMM=0 restores always-GEMV for gated/mixed. The once-"can no
// longer occur" hazard (union taking k_vgemm where solo took the dp4a GEMV)
// is now the DELIBERATE default at k >= 3. Q27_GEMM_MIN=99 pins in older
// gates pin the SOLO threshold only, not the union family.
// sampled[m] (Task 9, nullable = all-greedy) picks member m's verify tail:
// spec_verify_tail_sampled (nucleus + rejection accept + finish_sampled)
// instead of the greedy argmax tail. The FORWARD is shared by design (both
// solo graph sets capture the same spec_verify_forward), so sampled and
// greedy members coexist in one union sweep; only the per-engine tail forks.
// ev_draft_end (P2 Task 1, nullable -- the smoke driver passes none): a
// TIMING event recorded on cstm immediately after the draft_done waits, i.e.
// at the draft->verify phase boundary. Recording an event on an in-order
// stream adds no ordering and no synchronization; it only timestamps the
// point where cstm was released to begin verify work.

// P2b (plan 2026-07-15-batch-p2-overlap.md Task 3): mixer fork/join plumbing.
// The caller (the Conductor) owns k side streams + a fork event + k mix
// events; fused_verify_round, per mixer layer, records `fork` on cstm after
// the union pre, launches engine m's mix on side[m] (fenced behind `fork`),
// records mix[m] on side[m], and makes cstm wait every mix[m] before the
// union post. nullptr = the P1 serial path (fused_smoke leg B keeps it as
// the serial reference; legs C/D run the fork through the real Conductor).
//
// WHY P2b OVERLAPS WHERE P2a REALIZED ~0%: Task 2 measured the draft-overlap
// gain at essentially zero because draft steps are WEIGHT-BW-BOUND -- two
// engines stepping concurrently read the SAME MTP weights and just share one
// DDR/L2 bandwidth stream, so concurrency buys nothing. The mixers rest on
// DIFFERENT physics: fdmma verify attention is OCCUPANCY-bound (12.5% occ,
// documented in the fdmma plan), and the GDN conv/delta chains are small
// LATENCY-bound kernels; each engine's mix reads and writes ONLY its own
// sequence state (its KV cache, conv rings, S roles, lane activations -- the
// B2 audit below), i.e. DIFFERENT memory per engine. Co-resident streams
// therefore add real parallelism instead of splitting one shared read
// stream. P2a's lesson ("fusion-or-nothing") applies to weight sweeps;
// fork/join applies to state chains.
//
// B2 ISOLATION AUDIT (2026-07-15 at a01c110, blocking precondition -- every
// device buffer a mix touches is Engine-owned; nothing DeviceModel-shared is
// written):
//   gdn_mix(il, st):  conv_ring[il] (lane 0 RW in place; chunk R),
//     S[il] (lane 0 RW in place; chunk R), qkv/qkv_L (R), convout/convout_L
//     (W then R, l2norm3 in place), g_L/beta_L (R), o_L (W), rec_qkv/rec_conv/
//     rec_g/rec_beta[il] (W, the M1 record arena) -- all Engine members. The
//     ONE shared read is cw = T(il,"ssm_conv1d.weight").data: a DeviceModel
//     WEIGHT, read-only at round time (concurrent reads are safe).
//   attn_mix(il, st): qg_L (RW: wht3 rotates in place, attn reads), kbuf_L/
//     vbuf_L (R), KV rows via kv_ktab(ci)/kv_vtab(ci) (M2a: the store/decode
//     kernels READ d_kv_tab entries, then W disjoint rows / R through them),
//     scratch (RW, the fd/fdmma partials buffer), attnout_L (W, wht3 inverse
//     in place), d_pos_L (R) -- all Engine members; NO weight reads at all.
//     M2b WARNING: the audit's soundness now also rests on d_kv_tab entries
//     being 'mutated only at init/commit boundaries, never inside a round' --
//     admission-time remaps must stay stream-ordered against in-flight rounds.
//   Host-side reads: vw/kv_kind/kv_fp8/max_ctx/attn_cache_idx --
//     per-engine members, mutated only at init/commit boundaries, never
//     inside a round.
//   (a) NO cudaGraphExec launches inside either mix -- plain kernel wrappers
//       only (conv_step/l2norm3/delta_step/wht3/kv_store3/kv_store_t3/
//       attn_decode3). The engines' graph execs are per-engine members and
//       are never launched from the fused round.
//   (b) host-side statics in the launch path: launch_fdmma_w's per-
//       instantiation `static bool attr` (fdmma.cuh:415), fd_setattr<CT>'s
//       (spec3.cu:627), and attn_decode3's one-shot arch/stages_pin/
//       smem_per_sm/ns_pin/smc statics -- all one-shot cudaFuncSetAttribute
//       / device-query latches, set on the FIRST launch (engine warm-up /
//       graph capture, long before any fused round). Every mixer launch
//       still issues from the SINGLE conductor thread -- the fork is
//       streams, not threads -- so no host state is ever raced.
//   (c) NO cudaMalloc/cudaMallocAsync in the round path (grep-verified:
//       engine allocations live in init/prefill/ckpt_save only; the fused
//       round allocates nothing).
struct MixerFork {
    const cudaStream_t* side; // [k] conductor-owned side streams
    cudaEvent_t fork;         // recorded on cstm after each union pre
    const cudaEvent_t* mix;   // [k] recorded on side[m] after engine m's mix
};
inline void fused_verify_round(Engine** es, const int* granted, int k, cudaStream_t cstm,
                               const cudaEvent_t* draft_done, const bool* is_suffix,
                               const bool* sampled = nullptr,
                               cudaEvent_t ev_draft_end = nullptr,
                               const MixerFork* mf = nullptr) {
    // P3 T3: draft_done may be nullptr ONLY on the conductor's graph path
    // (Q27_BATCH_GRAPH=1): Conductor::graph_round issues these waits itself
    // on cstm BEFORE cudaStreamBeginCapture (a cudaStreamWaitEvent on an
    // event recorded outside the capture is capture-illegal), and its eager
    // guard-trip fallback runs after those same hoisted waits. Every other
    // caller passes real events and takes the unchanged line below.
    if (draft_done)
        for (int m = 0; m < k; m++) CUDA_CHECK(cudaStreamWaitEvent(cstm, draft_done[m], 0));
    if (ev_draft_end) CUDA_CHECK(cudaEventRecord(ev_draft_end, cstm));
    UnionView uv = build_union_view(es, granted, k, cstm, is_suffix);
    const Engine::LaneView& v = uv.view;
    Engine& e0 = *es[0];
    const DevTensor& emb = e0.shared_dm().get("token_embd.weight");
    q27k::embed3((const int8_t*)emb.data, (const __half*)emb.scales, v.vtok, N_EMBD,
                 LANESV(v, h), v.stm, v.vw);
    q27k::CP3 Hc LANESV(v, h), Yc LANESV(v, y);
    q27k::P3 Hm LANESV(v, h), X1m LANESV(v, x1);
    // P2b: one fork/join per mixer layer. DEVICE-side ordering only (B6):
    // cudaEventRecord + cudaStreamWaitEvent, never a host sync -- the round's
    // one host sync stays in the caller. The stream argument is the ONLY
    // delta vs the serial path: same kernels, same launch params, same
    // per-engine buffers, so per-lane bytes must be identical (B1 gate).
    // M1 record+fold: the fused GDN mix runs the SAME launch sequence as the
    // solo path (the P3 T2 table twins retired with the rotation) -- lane 0
    // in place on committed state, chunk kernels for the speculative lanes,
    // record arena writes. Every pointer is init-fixed, so captured execs are
    // perm-invariant with no per-round device staging at all. CALLER
    // CONTRACT: every member's flush_fold(cstm) must have been enqueued on
    // cstm before this round (Conductor::fused_round does; fused_smoke leg B
    // does) so the verify reads FOLDED committed state; the flush stays
    // OUTSIDE any captured graph (its launch count varies with the accepted
    // n).
    auto mix_all = [&](int il, bool attn) {
        if (!mf) { // serial P1 path (smoke leg B / embedders without a pool)
            for (int m = 0; m < k; m++) {
                if (attn) es[m]->attn_mix(il, cstm);
                else      es[m]->gdn_mix(il, cstm);
            }
            return;
        }
        CUDA_CHECK(cudaEventRecord(mf->fork, cstm));
        for (int m = 0; m < k; m++) {
            CUDA_CHECK(cudaStreamWaitEvent(mf->side[m], mf->fork, 0));
            if (attn) es[m]->attn_mix(il, mf->side[m]);
            else      es[m]->gdn_mix(il, mf->side[m]);
            CUDA_CHECK(cudaEventRecord(mf->mix[m], mf->side[m]));
        }
        for (int m = 0; m < k; m++)
            CUDA_CHECK(cudaStreamWaitEvent(cstm, mf->mix[m], 0));
    };
    for (int il = 0; il < N_LAYER; il++) {
        const float* an = (const float*)e0.T(il, "attn_norm.weight").data;
        q27k::rmsnorm3(Hc, an, X1m, N_EMBD, EPS, v.stm, v.vw);
        if (e0.is_attn_layer(il)) {
            e0.attn_pre(il, v);
            mix_all(il, true);
            e0.attn_post(il, v);
        } else {
            e0.gdn_pre(il, v);
            mix_all(il, false);
            e0.gdn_post(il, v);
        }
        q27k::add3(Hm, Yc, N_EMBD, v.stm, v.vw);
        const float* pn = (const float*)e0.T(il, "post_attention_norm.weight").data;
        q27k::rmsnorm3(Hc, pn, X1m, N_EMBD, EPS, v.stm, v.vw);
        e0.ffn_pair(il, v);
        q27k::add3(Hm, Yc, N_EMBD, v.stm, v.vw);
    }
    const float* on = (const float*)e0.shared_dm().get("output_norm.weight").data;
    q27k::rmsnorm3(Hc, on, X1m, N_EMBD, EPS, v.stm, v.vw);
    e0.qx5(v, v.x1, N_EMBD);
    const char* vhead = (e0.fast_head_on() && e0.shared_dm().model_has("output_q4.weight"))
                            ? "output_q4.weight"
                            : "output.weight";
    e0.mm5(v, e0.shared_dm().get(vhead), v.lg);
    // per-engine tails: own lane pointers (solo view), granted width,
    // conductor stream -- argmax/accept + finish land in each engine's own
    // d_v/d_outcome/h_next, and the fold commit semantics are untouched.
    for (int m = 0; m < k; m++) {
        Engine::LaneView tv = es[m]->solo_view(); // vw already = granted[m]
        tv.stm = cstm;
        if (sampled && sampled[m]) es[m]->spec_verify_tail_sampled(tv);
        else es[m]->spec_verify_tail(tv);
    }
}

// ---------------------------------------------------------------------------
// P2c (docs/plans/2026-07-16-batch-p2c-draft-fusion.md Task 2): fused draft
// steps -- per step of the interleaved loop, ONE union MTP weight sweep
// (eh_proj mm + the MTP ffn + the dominant 248320-row head mm) serves every
// still-active gated member. This is the P0/P1 union pattern applied to
// mtp_forward: drafts were measured WEIGHT-BW-BOUND (P2a realized ~0 from
// pure overlap -- concurrent engines just split one DDR/L2 read stream), so
// the win is one weight read instead of k.
//
// Union slot m = engine m's mtp_step_view(step) lane 0 -- the per-step
// h_src/tok/pos/draft_dst/margin_dst chain pointers (the table lives in
// Engine::mtp_step_view, SHARED with the solo capture path so the two can
// never drift). Engines advance in LOCKSTEP through the interleaved loop --
// every active member enters at step 0 and advances by 1 per iteration --
// so `step` is shared (the caller asserts launched[m] == step per member).
// Slots >= k keep es[0]'s solo padding (never read; the gemv_*_n
// `i < nb ? i : 0` convention).
inline Engine::MtpLaneView build_mtp_union_view(Engine** es, int k, int step,
                                                cudaStream_t cstm) {
    assert(k >= 1 && k <= W_PLUMB); // == ConductorCore MAX_K (static_assert
                                    // at the Conductor below); MtpLaneView
                                    // lane arrays are W_PLUMB-slotted
    // no duplicate engines: a dup would hand two union slots the SAME chain
    // buffers and the sweep would double-write them
    for (int a = 0; a < k; a++)
        for (int b = a + 1; b < k; b++) assert(es[a] != es[b]);
    // one weight set serves every lane: all members must share the
    // DeviceModel (the server's shared_model construction)
    for (int m = 1; m < k; m++) assert(&es[m]->shared_dm() == &es[0]->shared_dm());
    Engine::MtpLaneView v = es[0]->mtp_step_view(step);
    for (int m = 1; m < k; m++) {
        const Engine::MtpLaneView sv = es[m]->mtp_step_view(step);
        v.e_hn[m] = sv.e_hn[0];       v.x_mtp[m] = sv.x_mtp[0];
        v.x1[m] = sv.x1[0];           v.y[m] = sv.y[0];
        v.lg[m] = sv.lg[0];           v.ffn_g[m] = sv.ffn_g[0];
        v.ffn_u[m] = sv.ffn_u[0];     v.h_src[m] = sv.h_src[0];
        v.tok[m] = sv.tok[0];         v.pos[m] = sv.pos[0];
        v.draft_dst[m] = sv.draft_dst[0];
        v.margin_dst[m] = sv.margin_dst[0];
        v.xq[m] = sv.xq[0];           v.am_blk1[m] = sv.am_blk1[0];
        v.am_blk2[m] = sv.am_blk2[0]; v.amax[m] = sv.amax[0];
    }
    v.vw = k;
    v.stm = cstm;
    v.gemm_min = 99; // A1/Task-9 policy, MTP flavor: solo drafts run the
                     // dp4a GEMV family, so the union must NEVER take vgemm
                     // (mtp_mm has no vgemm path and asserts this). Margins
                     // are BITWISE vs solo, and BOTH halves of that claim are
                     // ninv-gated (tools/ninv_test.cu): the family tables pin
                     // multi-lane N-invariance (T/slot), and the SEAM LEG
                     // (P2 exit review 2026-07-16) pins the other half this
                     // comment used to assume -- the solo path runs the
                     // SINGLE-lane kernels (mtp_mm1/qx/rmsnorm/add/embed/
                     // silu_mul) while the fused step runs the multi-lane
                     // twins, and the leg measured every pair bitwise-equal
                     // on both arches (T in {2,4}, payload lane vs junk).
                     // A regression on either half now fails ninv_test.
    return v;
}

// One fused draft step at chain position `step` across engines es[0..k):
// per-engine prep (step 0: prep_round bookkeeping; step>0: the x1 -> hs[step]
// chain D2D -- same relative order as solo, the preamble precedes the step's
// first kernel) -> union mtp_pre (embed/norms + ONE eh_proj sweep) ->
// per-engine MTP attention -> union mtp_post (ONE ffn + head sweep) ->
// per-lane argmax_margin tail. Eager, all on cstm -- the solo path keeps its
// captured per-engine draft graphs untouched.
// The MTP attentions run SERIAL on cstm, deliberately WITHOUT the P2b
// side-stream fork/join: each is ONE token's attention against the tiny MTP
// KV (vs the verify mixers' W lanes), so the per-engine record+wait
// choreography would cost a comparable wall to what it hides, and the weight
// sweeps on either side are the actual round wall.
// The tail runs as the union-view loop: mtp_tail is per-lane already, and
// the union view carries each engine's OWN argmax scratch
// (lg/draft_dst/margin_dst/am_blk1/am_blk2 per lane), so lane t launches
// exactly the solo argmax_margin call with engine t's buffers.
inline void fused_draft_step(Engine** es, int k, int step, cudaStream_t cstm) {
    assert(k >= 2); // k==1 stays on the captured solo step graphs (the
                    // multi-lane kernel family has no nbatch=1 -- Task 1)
    for (int m = 0; m < k; m++) es[m]->draft_step_prep(step, cstm);
    Engine::MtpLaneView v = build_mtp_union_view(es, k, step, cstm);
    es[0]->mtp_pre(v);
    for (int m = 0; m < k; m++) es[m]->mtp_attn(v.pos[m], cstm);
    es[0]->mtp_post(v);
    es[0]->mtp_tail(v);
}

// ---------------------------------------------------------------------------
// P1 Task 9: the Conductor -- ONE dedicated thread owning every decode round
// in batch mode; request threads own everything else (prefill, SSE, slots).
// Server-agnostic: Task 10 wires request threads to register_member() and
// TokenQueue draining; nothing here knows about HTTP or slots. Lifecycle:
// the constructor spawns the thread; the destructor (or request_stop() +
// destructor) cancels any remaining members, closes their queues, and joins.
class Conductor {
public:
    // One registered decode: the (engine, task, queue) triple plus this
    // round's proposal class and the draft-completion event. Implements
    // ConductorCore's duck-typed MemberT surface over the thin Engine
    // entrypoints (A4).
    struct Member {
        Engine* e = nullptr;
        Engine::DecodeTask* t = nullptr;
        TokenQueue* q = nullptr;
        bool sampled = false;             // tail choice (fused_verify_round)
        bool sfx_round = false;           // set per round by want_width()
        int gate_cap = -1, md_used = -1;  // draft_and_gate outs -> commit_outcome
        cudaEvent_t draft_done = nullptr; // recorded on e->stm after drafting
        bool pre_round() { return e->pre_round(*t); }
        bool parked() const { return t->parked; }
        // P2a: the serial draft path. The real conductor installs the
        // core.draft_widths hook (Conductor::draft_widths below), which
        // supersedes this per-member call with the interleaved equivalent;
        // this stays as the core's hookless fallback and the reference
        // semantics the interleave must (and does -- B8) reproduce.
        int want_width() {
            gate_cap = md_used = -1;
            // mirror spec_round's branch order: the suffix drafter fires
            // before the MTP chain (greedy only -- spec_sample_round has no
            // suffix branch), on this engine's OWN stream.
            if (!sampled) {
                int sw = e->suffix_propose();
                if (sw > 0) {
                    sfx_round = true;
                    return sw;
                }
            }
            sfx_round = false;
            return e->draft_and_gate(sampled, &gate_cap, &md_used);
        }
        bool round_is_suffix() const { return sfx_round; }
        void set_granted(int w) { e->set_round_width(w); }
    };
    // Review L1: ConductorCore::round()'s hard k <= MAX_K/2 check stands in
    // for k <= W_PLUMB/2 (the pure-host half cannot see cuda_common.h);
    // this pin keeps the two ceilings the same number.
    static_assert((int)ConductorCore<Member>::MAX_K == W_PLUMB,
                  "MAX_K must equal W_PLUMB or round()'s union ceiling check drifts");

    // gate: the server's GpuGate -- prefill chunks time-slice against decode
    // rounds through it, unchanged (design "Scheduler"). cap: the union
    // width cap fed to trim_widths (W_MAX; the W16 build raises it).
    explicit Conductor(GpuGate& gate_, int cap_ = W_MAX) : gate(gate_), core(cap_) {
        CUDA_CHECK(cudaStreamCreate(&cstm)); // created ONCE; all fused rounds
        // P2 Task 1: fused-round phase-wall pool -- 3 TIMING events (default
        // flags, NOT cudaEventDisableTiming), created once and reused every
        // round. Reuse is sound under the B3 invariant enforced by the
        // round_active bracket (opened in draft_widths, closed in
        // fused_round): exactly ONE fused round is in flight per Conductor
        // (single conductor thread, synchronous round loop -- each round
        // records, syncs cstm, and reads elapsed before returning), so a
        // record can never overwrite a timestamp that is still to be read.
        CUDA_CHECK(cudaEventCreate(&ev_round_start));
        CUDA_CHECK(cudaEventCreate(&ev_draft_end));
        CUDA_CHECK(cudaEventCreate(&ev_verify_end));
        // P2b: conductor-owned mixer side streams + the fork/join event pool
        // (MixerFork rationale + B2 audit at fused_verify_round). Side
        // streams are NOT the engines' stms -- the draft_done/stm ordering
        // contract stays untouched -- and are NonBlocking so they never
        // implicitly serialize against the legacy default stream.
        //
        // EVENT-POOL SIZING (the plan's "justify or size 2x" call): ONE fork
        // event + MAX_K mix events, REUSED for every mixer layer of every
        // round. This is legal under documented CUDA semantics WITHOUT any
        // consumption argument: cudaStreamWaitEvent snapshots the work
        // captured by the most recent cudaEventRecord AT THE TIME OF THE
        // WAIT CALL, so a later re-record cannot retarget an already-issued
        // wait. All records and waits are issued by this single conductor
        // thread in program order -- layer il's waits are issued before
        // layer il+1 re-records -- and the B3 round_active guard forbids a
        // second in-flight round whose records could interleave. Chosen over
        // a 2x alternating pool because snapshot semantics make "was the
        // prior wait consumed?" irrelevant, which is the simpler-to-justify
        // (and assert-backed) invariant.
        // issue #25: this loop is the FIRST allocation after the server reports
        // "at ready", so a config that sized KV + prefix-cache staging to the
        // brim dies here -- as a bare "CUDA error: out of memory at
        // cudaStreamCreateWithFlags", with nothing on screen connecting it to
        // --ctx or --prefix-cache, and it crash-loops because a larger cache
        // makes the next boot worse. The streams are structural (MAX_K is the
        // union width), so unlike the exec cache below there is nothing to
        // shrink; the honest fix is to say WHY before CUDA_CHECK aborts.
        {
            size_t freeb = 0, totalb = 0;
            if (cudaMemGetInfo(&freeb, &totalb) == cudaSuccess &&
                freeb < (64ull << 20)) {
                fprintf(stderr,
                        "[conductor] only %.0f MB free before creating %d side "
                        "streams -- if this OOMs, the cause is the VRAM budget, "
                        "not the conductor. An explicit --ctx skips the auto-ctx "
                        "safety margin, and --prefix-cache pins staging per slot "
                        "on top of it; drop --ctx or lower "
                        "--prefix-cache-max-tokens.\n",
                        freeb / 1e6, (int)ConductorCore<Member>::MAX_K);
            }
        }
        for (int i = 0; i < ConductorCore<Member>::MAX_K; i++) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&side_[i], cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_mix_[i], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming));
        // P3 T3 (plan 2026-07-16-batch-p3-capture.md): fused-verify exec
        // cache config. Per-INSTANCE env read, deliberately NOT a process
        // static: fused_smoke A/Bs an eager conductor against a graph
        // conductor in one process (leg E sets the env between legs).
        // Default OFF -- the Q27_BATCH precedent: eager is the reference
        // path, capture is the opt-in lever.
        {
            const char* e = getenv("Q27_BATCH_GRAPH");
            graphs_on_ = e && atoi(e) != 0;
        }
        if (graphs_on_) {
            const char* c = getenv("Q27_BATCH_GRAPH_CAP");
            if (c && atoi(c) > 0) gc_cap_ = atoi(c);
            // Startup headroom check (T0 finding: ~7.2 MB device memory per
            // instantiated exec; budgeted 8 MB, so cap 32 ~ 230 MB). Runs
            // here because both the server and fused_smoke construct the
            // Conductor AFTER the engines, so free VRAM already reflects
            // weights + KV + the solo graph zoo. SHRINK, never abort: a
            // smaller cache only costs LRU recapture (~2.4 ms first-sight,
            // T1) where a hard exit would kill a server that runs fine.
            size_t freeb = 0, totalb = 0;
            CUDA_CHECK(cudaMemGetInfo(&freeb, &totalb));
            const size_t per_exec = 8ull << 20;
            // Reserve a margin (E2.1): the old budget was ALL free VRAM, which
            // never bound at cap 64 (512 MB) but at cap 512 would let the exec
            // cache claim every spare byte -- and allocations still happen after
            // this point (prefix-cache pages, arena growth, driver scratch). A
            // cache that starves them trades a 20-28 ms capture stall for an
            // OOM, which is a strictly worse deal.
            const size_t margin = 768ull << 20;
            const size_t budget = freeb > margin ? freeb - margin : 0;
            if ((size_t)gc_cap_ * per_exec > budget) {
                int shrunk = (int)(budget / per_exec);
                if (shrunk < 1) shrunk = 1;
                fprintf(stderr,
                        "[gcache] headroom: %.0f MB free (%.0f MB budget after margin) < "
                        "cap %d x 8 MB/exec -- shrinking exec-cache cap to %d "
                        "(LRU recapture covers the rest)\n",
                        freeb / 1e6, budget / 1e6, gc_cap_, shrunk);
                gc_cap_ = shrunk;
            }
            // (no gcache_.reserve here: a bad_alloc before the try block
            // below would leak the handles just created; growth to <= 32
            // entries at round time is noise and shares the posture of the
            // other conductor-thread vector ops -- owned/members push_back)
            fprintf(stderr,
                    "[gcache] fused-verify graph cache ON (Q27_BATCH_GRAPH=1, cap %d)\n",
                    gc_cap_);
        }
        // Exception guard (P2 exit review): a std::function assignment or
        // std::thread construction throw below (bad_alloc / system_error)
        // would leak every handle just created -- the dtor of an object
        // whose ctor throws never runs. Tear down + rethrow.
        try {
            core.solo_round = [this](Member& mm) { return this->solo_round(mm); };
            core.needs_solo_round = [](Member& mm) { return mm.t->round_forced; };
            core.fused_round = [this](Member** ms, const int* granted, const bool* sfx,
                                      int k, bool* done) {
                this->fused_round(ms, granted, sfx, k, done);
            };
            core.draft_widths = [this](Member** ms, int* want, bool* sfx, int k) {
                this->draft_widths(ms, want, sfx, k);
            };
            core.on_leave = [this](Member& mm) { this->leave(mm); };
            th = std::thread([this] { run(); });
        } catch (...) {
            destroy_handles();
            throw;
        }
    }
    ~Conductor() {
        request_stop();
        th.join();
        destroy_handles();
    }
    Conductor(const Conductor&) = delete;
    Conductor& operator=(const Conductor&) = delete;

    // Registration API (Task 10 calls this from a request thread AFTER its
    // prefill completes and the DecodeTask is built). The conductor owns
    // token delivery from here: it installs the owning queue sink over
    // t->on_token (see the DecodeTask field comment) -- post_round() then
    // pushes each emitted token; pushes are non-blocking (A7). Client-stop
    // is expressed via t->cancel (A3), never via the sink's return value.
    // The join lands at the next round boundary.
    // on_emit (Task 10, nullable): runs on the CONDUCTOR thread inside
    // post_round, per token, BEFORE the queue push -- i.e. between the
    // on_round scan and on_pending, the exact slot the solo on_token body
    // occupies. The server routes tool-constrain grammar feeding (tc.on_id)
    // here so its ordering against scan_round/on_pending is unchanged AND
    // its engine mutations (set_tool_constraint's async copies) keep running
    // under the GPU gate, now held by the conductor's round lease.
    void register_member(Engine* e, Engine::DecodeTask* t, TokenQueue* q,
                         std::function<void(int)> on_emit = nullptr) {
        assert(!t->force_plain_sample); // plain-sample rounds have no fused
                                        // path; Q27_SAMPLE_PLAIN is an A/B
                                        // lever, unsupported in batch mode
        auto mm = std::unique_ptr<Member>(new Member());
        mm->e = e;
        mm->t = t;
        mm->q = q;
        mm->sampled = t->sampling;
        // The owning queue sink is BUILT here but installed over t->on_token
        // only on the accepted path below (review pass 2): on the M4 refusal
        // the caller unblocks off fail() and may destroy q immediately, so a
        // sink installed before the stop-check would leave t->on_token
        // dangling over the dead queue.
        std::function<bool(int)> sink;
        if (on_emit)
            sink = [q, t, oe = std::move(on_emit)](int id) {
                oe(id);
                q->push(id, t->callback_forced);
                return true;
            };
        else
            sink = [q, t](int id) {
                q->push(id, t->callback_forced);
                return true;
            };
        CUDA_CHECK(cudaEventCreateWithFlags(&mm->draft_done, cudaEventDisableTiming));
        {
            std::lock_guard<std::mutex> lk(m);
            // Review M4: registration racing shutdown. If the stop flag is
            // already up, run()'s shutdown drain may have ALREADY adopted its
            // last-instant joins -- a member pushed now would never be
            // cancelled and its caller would block on the queue forever.
            // Refuse under the same lock the drain takes: fail the queue so
            // the caller unblocks with a surfaced error instead of a hang.
            // Unreachable in the current server (it destroys the Conductor
            // only after the HTTP listener stops taking requests -- stop
            // semantics of the vendored httplib verified, see the conductor
            // construction comment in server.cu), but this class is
            // server-agnostic and embedders deserve the check.
            if (stop) {
                CUDA_CHECK(cudaEventDestroy(mm->draft_done));
                // Same close-edge discipline as fail_member: finish_decode
                // BEFORE the queue edge, so this exit stamps gs.end/dec/rounds
                // like every other one and the [req] line cannot report a
                // previous request's telemetry. make_decode_task() has already
                // run (batch_generate fills the task under the prefill lease,
                // then registers), so g0/n_max/Ph are populated and the task is
                // safe to finish. The sink is still not installed here, which
                // is the separate M4 rule above.
                e->finish_decode(*t, "error");
                q->fail("conductor stopping: registration refused");
                return;
            }
            t->on_token = std::move(sink);
            // only a conductor member can sit a round out (incremental KV);
            // a solo generate() task keeps can_park false and stops instead
            t->can_park = true;
            join_q.push_back(std::move(mm));
        }
        cv.notify_one();
    }
    // Incremental KV (issue #42 step 2): the server calls this when KV pages
    // may have been released (a slot freed), so a round loop whose members
    // are all parked re-checks them now instead of at the idle timeout.
    void poke() {
        {
            std::lock_guard<std::mutex> lk(m);
            poked = true;
        }
        cv.notify_one();
    }
    // Shutdown: the round loop exits at the next boundary; remaining members
    // are cancelled (A3 semantics -- finish_decode runs, queues close) so no
    // request thread is left blocking on a queue.
    void request_stop() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv.notify_one();
    }

private:
    // Handle teardown shared by the dtor and the ctor's exception guard
    // (cstm + 3 phase events + MAX_K side streams + MAX_K mix events +
    // fork). Order mirrors the old dtor body exactly.
    void destroy_handles() {
        // P3 T3: exec-cache teardown FIRST -- every cached exec + graph
        // (T0: ~7.2 MB device memory per exec; leaving them leaks ~cap x
        // 8 MB). Safe: the dtor joins the conductor thread before calling
        // this, and every round ends on a cstm sync, so no exec is in
        // flight; the ctor's exception path sees an empty cache. The final
        // [gcache] line is the run's telemetry summary (T3 gate e).
        if (graphs_on_)
            fprintf(stderr,
                    "[gcache] final: rounds=%ld hits=%ld misses=%ld evictions=%ld "
                    "guard_trips=%ld entries=%zu cap=%d\n",
                    gc_rounds_, gc_hits_, gc_misses_, gc_evictions_, gc_guard_trips_,
                    gcache_.size(), gc_cap_);
        for (auto& e : gcache_) gc_destroy(e);
        gcache_.clear();
        CUDA_CHECK(cudaEventDestroy(ev_round_start));
        CUDA_CHECK(cudaEventDestroy(ev_draft_end));
        CUDA_CHECK(cudaEventDestroy(ev_verify_end));
        for (int i = 0; i < ConductorCore<Member>::MAX_K; i++) {
            CUDA_CHECK(cudaStreamDestroy(side_[i]));
            CUDA_CHECK(cudaEventDestroy(ev_mix_[i]));
        }
        CUDA_CHECK(cudaEventDestroy(ev_fork_));
        CUDA_CHECK(cudaStreamDestroy(cstm));
    }

    // Conductor thread body. GATE-OWNERSHIP INVARIANT (A7): in batch mode
    // the gate holders are EXACTLY this thread (one Lease per decode round)
    // and request threads (prefill chunks). The conductor never blocks on a
    // request thread while holding the gate -- TokenQueue::push is
    // non-blocking -- and request threads never hold the gate while waiting
    // on a TokenQueue (Task 10 drains AFTER the prefill Lease is released).
    // Releasing the Lease at every round boundary IS the maybe_yield
    // equivalent: the FIFO gate hands queued prefill chunks the GPU between
    // rounds exactly as today's round_gap() interleave does.
    // NOTE for Task 10: engines under the conductor must keep on_round_gap
    // UNSET -- post_round() would otherwise yield the conductor's own Lease
    // from inside a round, defeating the one-Lease-per-round structure.
    void run() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m);
                // idle = no members and no joins: block on the cv (no spin)
                cv.wait(lk, [&] {
                    return stop || !join_q.empty() || !core.members.empty();
                });
                if (stop) break;
                for (auto& j : join_q) {
                    core.join(j.get());
                    owned.push_back(std::move(j));
                }
                join_q.clear();
            }
            bool idle;
            {
                GpuGate::Lease lease(gate); // released per round (A7 above)
                core.round();
                idle = core.last_idle;
            }
            // Every member parked (incremental KV): holding the gate and
            // re-running would starve the prefill that the banker's head of
            // line is waiting on. Wait off the gate for a join, a poke (a
            // slot freed pages), or a short timeout, then re-check.
            if (idle) {
                std::unique_lock<std::mutex> lk(m);
                cv.wait_for(lk, std::chrono::milliseconds(2),
                            [&] { return stop || poked || !join_q.empty(); });
                poked = false;
            }
        }
        // shutdown drain: adopt any last-instant joins, then cancel + finish
        // every remaining member exactly like a client cancel (A3).
        {
            std::lock_guard<std::mutex> lk(m);
            for (auto& j : join_q) {
                core.join(j.get());
                owned.push_back(std::move(j));
            }
            join_q.clear();
        }
        // Review pass 2 (sibling window of the M4 refusal): core.join() only
        // STAGES into core.joins -- the boundary drain in round() is what
        // promotes them to members, and no further round will run. Merge the
        // staged joins into the member set BEFORE the cancel pass or it
        // would skip them (it walks core.members only), leaving their
        // request threads blocked forever on queues nobody closes.
        // request_stop()'s contract: NO request thread is left blocking,
        // whether its member was live, staged pre-stop, or refused post-stop.
        for (Member* mm : core.joins) core.members.push_back(mm);
        core.joins.clear();
        for (Member* mm : core.members) {
            mm->t->cancel.store(true);
            mm->pre_round(); // runs finish_decode("cancelled")
            leave(*mm);      // queue close + event destroy + free
        }
        core.members.clear();
    }

    // A2 catch epilogue, shared by solo_round/fused_round below. A throwing
    // round cannot have run finish_decode (every finish_decode call inside
    // pre_round/post_round/decode_step is immediately followed by a
    // non-throwing return), so run it here with "error": it closes the
    // Q27_PROF_DECODE profiler bracket if open, finalizes GenStats and
    // stamps gs.end = "error" -- the request thread's [req] line and the
    // server's error surfacing key off that stamp. finish_decode's body is
    // non-throwing host bookkeeping (chrono + fprintf; its CUDA_CHECK exits,
    // never throws -- the A2 fatal posture).
    // CLOSE-EDGE RULE (review pass 2, same LIFETIME class as TokenQueue's
    // H1 rule): fail() closes the queue, and the close is the
    // synchronization edge after which the request thread may destroy BOTH
    // the queue and the frame-local DecodeTask (engine.cuh, the
    // DecodeTask::bat_members comment). So fail() must be the LAST access
    // to q AND t on every catch path: the finish_decode epilogue and all
    // counter updates run strictly before it, and mm.q is nulled first so
    // leave() -- which runs after the member's removal -- cannot touch the
    // dead queue either.
    void fail_member(Member& mm, const char* what) {
        mm.e->finish_decode(*mm.t, "error");
        TokenQueue* q = mm.q;
        mm.q = nullptr;
        q->fail(what); // copies what(); LAST q/t access -- rule above
    }

    // Solo fallthrough (k==1): decode_step IS today's path -- captured round
    // graphs, spec_round's own bookkeeping/telemetry, tokens through the
    // queue sink. true = done. bat telemetry: a solo round contributes k=1
    // per round that actually RAN (pre_round exits inside decode_step do not
    // advance t->rounds, so the delta is the ran-count); the update runs in
    // EVERY arm before any queue op (fail_member's close-edge rule).
    // A2 unwind (addendum A2, review M1): CUDA failures stay process-fatal
    // (CUDA_CHECK exits, never throws -- deliberately not wrapped); a HOST
    // exception from the bookkeeping half (post_round -> on_emit/queue sink,
    // grammar scan, ...) must kill only THIS member, not the conductor
    // thread. fail_member runs the error epilogue + queue fail; returning
    // true reports done so core.round() erases the member.
    bool solo_round(Member& mm) {
        long r0 = mm.t->rounds;
        try {
            bool done = !mm.e->decode_step(*mm.t);
            mm.t->bat_members += mm.t->rounds - r0;
            return done;
        } catch (const std::exception& ex) {
            mm.t->bat_members += mm.t->rounds - r0;
            fail_member(mm, ex.what());
        } catch (...) {
            mm.t->bat_members += mm.t->rounds - r0;
            fail_member(mm, "unknown host exception in solo round bookkeeping");
        }
        return true; // failed member leaves this round
    }

    // P2a: the batch draft phase -- Member::want_width() over all k members,
    // with the GATED members' margin loops INTERLEAVED so their per-step
    // draft graphs run concurrently on the engines' own streams (the
    // sequential host loop was the only serializer; the graphs already
    // lived on per-engine stms). Suffix members keep suffix_propose as-is
    // (one-shot host test + prep/H2D staging, no margin loop). Scheduling
    // only: each engine's stm sees the IDENTICAL call sequence
    // draft_and_gate would have enqueued, so per-member values and bytes
    // must match the serial path -- and B1 makes that the gate (the Task 0
    // refs), not an assumption.
    //
    // P2c ON TOP: when >= 2 gated members are still active at a step, the
    // per-engine graph launches are replaced by ONE eager fused_draft_step
    // on cstm (union MTP weight sweep -- drafts are weight-BW-bound, so
    // P2a's overlap realized ~0 and fusion is the lever). The margins are
    // computed by the same dp4a GEMV family at union width, ninv-proven
    // bitwise per lane, so the loop arithmetic below is UNCHANGED and B8
    // still re-derives {cap, W, launched} against draft_and_gate's
    // semantics. A single remaining active member falls back to its solo
    // step graphs (no nbatch=1 multi-lane kernels), and top-ups stay solo
    // by the same argument.
    //
    // EQUIVALENCE to draft_and_gate's loop, side by side. draft_and_gate:
    //   for (k = 0; k < md_used; k++) {
    //       launch step k; launched++;
    //       sync stm; if (margin[k] < theta) break;
    //       cap++;
    //   }
    //   W = max(2, cap+1); top-up launches [launched, min(W, md_used));
    // Interleaved, per gated member i (all active members share the step
    // counter -- every member enters at step 0 and advances by 1 per
    // iteration, so `step` IS member i's next k):
    //   - each iteration launches exactly step `step` on i's stm and
    //     increments launched[i]  == launch-k + launched++ above;
    //   - i's stm is synced past step `step`'s D2H before margin read
    //     == the per-step sync above (extra syncs of OTHER members' stms
    //     order nothing on i's stm);
    //   - margin[step] < theta  -> i exits with cap[i] unchanged. cap[i]
    //     was incremented once per PASSED step 0..step-1, so cap[i] ==
    //     step == draft_and_gate's cap at its break  (sub-theta break);
    //   - margin[step] >= theta -> cap[i]++ (== step+1), and i exits iff
    //     cap[i] == mdu[i]  == the loop bound k+1 < md_used failing after
    //     cap++  (full run: cap == md_used, launched == md_used);
    //   - on exit: W = max(2, cap+1) and the top-up range
    //     [launched, min(W, mdu)) fire with the SAME values -- so the same
    //     graph launches land on i's stm (margin steps 0..launched-1, then
    //     top-up steps; the range is empty except at cap==0, where
    //     launched==1 < W==2 <= mdu).
    // The margins themselves are computed by the same per-engine graphs on
    // the same per-engine state, so the break step is identical, hence
    // {cap, launched, W, md_used} are identical. B8 below re-derives them
    // from the recorded margins every round and asserts equality.
    void draft_widths(Member** ms, int* want, bool* sfx, int k) {
        // B3 bracket OPENS here, not in fused_round() (P2 exit review): the
        // P2c draft phase below already runs fused work on cstm and
        // re-records the members' draft_done events, so the one-round-in-
        // flight invariant the event pools rest on must hold from the FIRST
        // fused-phase record. core.round() always pairs this hook with
        // fused_round() (same thread, program order), which asserts the
        // bracket is open and CLOSES it.
        assert(!round_active && "B3: fused rounds must not overlap per Conductor");
        round_active = true;
        // Round-budget T2: host wall around the whole draft phase. Exact
        // because this phase host-syncs every step; see GenStats::fdraft_ms.
        const auto fd_t0 = std::chrono::steady_clock::now();
        enum { MAX_K = ConductorCore<Member>::MAX_K };
        int act[MAX_K];                              // gated members still in the loop
        int cap[MAX_K], launched[MAX_K], mdu[MAX_K]; // per-member loop state
        int na = 0;
        // T4 (docs/plans/2026-08-18-batch-round-budget.md): draft no deeper than
        // the trim can possibly GRANT. Measured 2026-08-18: at C=8 every lane is
        // granted width 2 while ~2.86 draft steps per member launch, so ~65% of
        // drafted tokens are computed and discarded, and the phase is 25.5% of
        // the round.
        //
        // The ceiling is DERIVED, not tuned. trim_widths never trims a lane
        // already at floor 2 (see its `want[i] <= 2` guard), so member i's grant
        // is maximised when every OTHER member sits at that floor:
        //     sum = want_i + 2*(k-1) <= cap   =>   want_i <= cap - 2*(k-1)
        // and when cap - 2*(k-1) < 2 the cap is unsatisfiable, trim gives up with
        // EVERY lane at 2. So max_grant(k) = max(2, cap - 2*(k-1)), exactly.
        // A verify of width W checks W-1 DRAFTED positions, hence the -1.
        //
        // This can never drop a step whose token could have been verified, which
        // is why accepted-tokens-per-round is unchanged BY CONSTRUCTION rather
        // than by measurement. Suffix members do not weaken it: they skip the MTP
        // chain entirely (the `continue` above) and trim floors them at 2 like
        // anyone else, so "every other member >= 2" still holds.
        //
        // At cap=W_MAX=12 this is 9/7/5/3/1/1/1 steps for k=2..8 -- non-binding
        // until k=5 (gate_maxd is 7) and pinned at ONE step from k=6, which is
        // where the whole win is. Same monotone shape SGLang's batch-size table
        // uses (bs 1-7 -> up to 7, 8-31 -> up to 3, 32+ -> 0-1), derived instead
        // of tuned: q27 knows k and cap exactly at round start, so it needs no
        // lagged acceptance EMA to infer what the scheduler already decided.
        static const bool ceil_on = [] { // Q27_DRAFT_CEIL=0 restores pre-T4 depth
            const char* e = getenv("Q27_DRAFT_CEIL");
            return !(e && atoi(e) == 0);
        }();
        // Ceiling is max_grant, NOT max_grant-1. A width-W verify walks W-1
        // drafts, but draft_and_gate's floor top-up maintains the stronger
        // invariant "W draft ROWS must exist" (engine.cuh, the cap==0 case), so
        // cutting to W-1 would break that invariant to save one step. Holding at
        // max_grant preserves it exactly and still cuts C=8 from ~3.35 launched
        // steps per member to 2. Revisiting the extra step is a separate change
        // that has to re-derive the row invariant first.
        const int max_grant = core.cap - 2 * (k - 1) < 2 ? 2 : core.cap - 2 * (k - 1);
        // E3.1 (Q27_DRAFT_CEIL1=1): drop the floor from 2 steps to 1.
        //
        // ROW INVARIANT, RE-DERIVED (the T4 comment above asked for exactly
        // this before touching the extra step). k_finish_round's accept walk is
        //     for (k = 1; k <= NDRAFT; k++)
        //         if (k <= max_draft && n == k && v[k-1] == dr[k-1]) n = k+1;
        // with max_draft = vw-1, so it reads dr[0 .. vw-2] -- **vw-1 rows, not
        // vw**. At the floor vw=2 that is dr[0] ALONE, and dr[0] is written by
        // draft step 0. Step 1's output dr[1] is loaded into a register and
        // copied to outcome[3], which the host reads only for n > 2 (outcome
        // layout: emitted tokens live in [1..n], and n <= vw = 2 here). So the
        // second step cannot change n, cannot change the emitted token, and
        // cannot change h_next (src = x1s.p[n-1], n <= 2).
        //
        // The "W draft ROWS must exist" invariant is therefore STRONGER than
        // the kernel needs. Its stated purpose (engine.cuh, spec_round) is
        // "byte/round identity with the monolithic path" -- an A/B-equivalence
        // property against Q27_DEXIT=0, not a correctness property of the
        // verify. Under this flag that identity is knowingly given up at the
        // floor; the EMITTED tokens are unaffected, which is what the canonical
        // digest and tok/round measure.
        //
        // At C>=6 the trim floors every lane to 2, so today every member runs 2
        // draft steps per round and uses exactly ONE of them, forever. This
        // makes the second step pure waste at high concurrency.
        static const bool ceil1 = [] {
            const char* e = getenv("Q27_DRAFT_CEIL1");
            return e && atoi(e) != 0;
        }();
        const int step_ceiling = (ceil1 && max_grant > 1) ? max_grant - 1 : max_grant;
        for (int i = 0; i < k; i++) {
            Member& mm = *ms[i];
            mm.gate_cap = mm.md_used = -1;
            // mirror spec_round's branch order (== want_width): the suffix
            // drafter fires before the MTP chain, greedy only, on this
            // engine's OWN stream. Suffix decisions are host one-shots over
            // per-engine state, so member i+1's decision landing before
            // member i's MTP steps reorders nothing observable.
            if (!mm.sampled) {
                int sw = mm.e->suffix_propose();
                if (sw > 0) {
                    mm.sfx_round = true;
                    sfx[i] = true;
                    want[i] = sw;
                    continue;
                }
            }
            mm.sfx_round = false;
            sfx[i] = false;
            // draft_and_gate's preamble, hoisted per member: sampled
            // bootstrap once before step 0, then the drafting ceiling
            // (the same dctl/gate_maxd read).
            if (mm.sampled) mm.e->draft_sample_bootstrap();
            mdu[i] = mm.e->draft_md_used(mm.sampled);
            if (ceil_on && mdu[i] > step_ceiling) mdu[i] = step_ceiling;
            cap[i] = launched[i] = 0;
            act[na++] = i;
        }
        // P2c ORDER FENCE: fused steps run on cstm, but each member's prior
        // GPU work lives on its OWN stm (the sampled bootstrap enqueued just
        // above; prefill / solo rounds before the member's first fused
        // round). One event per member orders cstm behind it. draft_done
        // doubles as the fence event: this wait is issued before
        // fused_round() re-records it (single conductor thread, program
        // order), and cudaStreamWaitEvent snapshots the record at call time
        // -- the ctor event-pool argument. Skipped at na < 2: the loop below
        // then never touches cstm (pure P2a solo path on the member's stm).
        if (na >= 2) {
            for (int j = 0; j < na; j++) {
                Member& mm = *ms[act[j]];
                CUDA_CHECK(cudaEventRecord(mm.draft_done, mm.e->stream()));
                CUDA_CHECK(cudaStreamWaitEvent(cstm, mm.draft_done, 0));
            }
        }
        // draft_done ORDERING NOTE (P2c): fused_round() records each
        // member's draft_done on the member's OWN stm after this returns,
        // and that stays correct -- everything a member contributes OFF cstm
        // (suffix prep/H2D staging, sampled bootstrap, floor top-ups, the
        // k==1 fallback steps below) is on its stm and thus captured, while
        // the fused steps here run ON cstm, the same in-order stream the
        // verify runs on (and are host-synced per step besides), so the
        // verify needs no event to see them.
        for (int step = 0; na > 0; step++) {
            if (na >= 2) {
                // P2c: ONE union MTP weight sweep serves every still-active
                // member at this chain position. When the active set shrinks
                // the next iteration simply fuses at the smaller na -- ninv
                // (slot/width invariance of the GEMV lanes) keeps every
                // remaining member's margins bitwise across the width step.
                Engine* aes[MAX_K];
                for (int j = 0; j < na; j++) {
                    // LOCKSTEP invariant build_mtp_union_view relies on:
                    // every active member is about to run exactly `step`
                    assert(launched[act[j]] == step &&
                           "P2c: active members must be in draft-step lockstep");
                    aes[j] = ms[act[j]]->e;
                }
                fused_draft_step(aes, na, step, cstm);
                for (int j = 0; j < na; j++) aes[j]->draft_margin_d2h(step, cstm);
                // ONE sync for all active members' margins (replaces P2a's
                // per-member stm syncs; B7's argument applies unchanged).
                CUDA_CHECK(cudaStreamSynchronize(cstm));
            } else {
                // Active set is down to ONE member: fall back to its
                // captured solo step graphs on its OWN stm, exactly the P2a
                // path -- the multi-lane kernel family has no nbatch=1
                // instantiation (gemv_*_n starts at 2; Task 1 DECIDE), so a
                // width-1 "union" cannot run the fused kernels. Ordering vs
                // the fused steps this member's chain already ran on cstm is
                // by HOST program order: the per-step cstm sync above
                // completed them before this launch is issued.
                ms[act[0]]->e->draft_step_launch(step);
                CUDA_CHECK(cudaStreamSynchronize(ms[act[0]]->e->stream()));
            }
            int keep = 0;
            for (int j = 0; j < na; j++) {
                const int i = act[j];
                Member& mm = *ms[i];
                launched[i]++; // member i launched step `step` above
                bool out;
                if (mm.e->draft_margin(step) < mm.e->gate_theta()) {
                    out = true; // sub-theta break: cap[i] stays == step
                } else {
                    cap[i]++;                   // == step+1
                    out = cap[i] == mdu[i];     // draft_and_gate's loop bound
                }
                if (!out) {
                    act[keep++] = i; // stable order: syncs stay member-order
                    continue;
                }
                // draft_and_gate's epilogue for this member, at its exit
                // step: floor W, width-floor top-up on ITS stm, out-params.
                // P2c top-up fencing: the top-up graphs (rare, cap==0 only)
                // stay per-engine solo launches on the member's stm; they
                // read chain state the fused steps wrote on cstm, and that
                // is safe by HOST program order -- this exit decision runs
                // strictly after the per-step cstm sync completed those
                // writes. fused_round() then records draft_done on this stm,
                // so the verify is fenced behind the top-up as before.
                int W = cap[i] + 1 < 2 ? 2 : cap[i] + 1;
                mm.e->draft_floor_topup(launched[i], W, mdu[i]);
                mm.gate_cap = cap[i];
                mm.md_used = mdu[i];
                want[i] = W;
            }
            na = keep;
        }
        // B8 (always-on; no build defines NDEBUG, so assert is live):
        // re-derive {cap, W, launched} for every gated member by running
        // draft_and_gate's arithmetic over the SAME recorded margins
        // (h_draft_margin persists on the engine; the re-run reads exactly
        // the prefix this round refreshed, because it breaks at the same
        // first sub-theta). Any mismatch is an interleave logic bug --
        // caught here, before it can reach the byte gate.
        for (int i = 0; i < k; i++) {
            if (sfx[i]) continue;
            const Engine& e = *ms[i]->e;
            int rcap = 0, rlaunched = 0;
            for (int s = 0; s < mdu[i]; s++) {
                rlaunched++;
                if (e.draft_margin(s) < e.gate_theta()) break;
                rcap++;
            }
            int rW = rcap + 1 < 2 ? 2 : rcap + 1;
            assert(rcap == ms[i]->gate_cap && "B8: interleaved cap != draft_and_gate cap");
            assert(rW == want[i] && "B8: interleaved W != draft_and_gate W");
            assert(rlaunched == launched[i] && "B8: interleaved launch count diverged");
            (void)rcap; (void)rW; (void)rlaunched;
        }
        fdraft_ms_ = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - fd_t0)
                         .count();
    }

    // One fused round over k >= 2 members (under the caller's Lease).
    // Sequence per the plan: drafts already ran inside draft_widths() above
    // (P2a: gated members' steps interleaved across engines) on each
    // engine's OWN stm; record each draft_done event; fused verify on cstm
    // (which waits on the events); per-engine outcome D2H on cstm + ONE
    // sync; per-member commit_outcome (spec_round's post-outcome mirror,
    // incl. dctl/histograms) + post_round (tokens -> queue via the sink,
    // EOS/budget/client-stop -> done).
    // Q27_PHASE_STATS in fused rounds (design call resolved, P2 Task 1):
    // gs.draft_ms/verify_ms/draft_steps ARE stamped, from coarse cstm event
    // brackets, with SHARED-WALL semantics -- the one fused wall is
    // attributed IN FULL to EACH member (see the accumulation loop below).
    // Still deliberately skipped: the per-width verify buckets vw_ms/vw_n
    // (a fused verify runs ONE union width; binning it per member width
    // would misprice the curve), sfx_ms/sfx_rounds, and [sfxdbg]'s propose
    // trace lines. Everything else spec_round mutates (last_pending,
    // sfx_valid/sfx.append, fold_pending, dctl, gate_cap/n/lane hists, sfx_fired/
    // sfx_tok, gs.dec/rounds/cb_ms/end) is mirrored via commit_outcome +
    // post_round.
    void fused_round(Member** ms, const int* granted, const bool* sfx, int k,
                     bool* done) {
        // B3 invariant, enforced not commented: exactly ONE fused round in
        // flight per Conductor (single conductor thread, synchronous round
        // loop). The 3-event phase pool is reused every round on the
        // strength of this -- each round records, syncs, and reads elapsed
        // before returning -- so a future pipelining change must trip here
        // loudly instead of silently corrupting timestamps. The bracket
        // OPENS in draft_widths() (P2c: the draft phase already records on
        // cstm/draft_done) and closes at the bottom of this function;
        // core.round() always runs the two back to back on this thread.
        assert(round_active && "B3: round bracket must be open (draft_widths runs first)");
        Engine* es[ConductorCore<Member>::MAX_K] = {};
        bool sampled[ConductorCore<Member>::MAX_K] = {};
        cudaEvent_t evs[ConductorCore<Member>::MAX_K] = {};
        for (int i = 0; i < k; i++) {
            es[i] = ms[i]->e;
            sampled[i] = ms[i]->sampled;
            evs[i] = ms[i]->draft_done;
            CUDA_CHECK(cudaEventRecord(evs[i], es[i]->stream()));
        }
        // M1: fold each member's previous-round commit into committed GDN
        // state before this round's verify reads it. Replaces P3 T2's
        // stage_perm_async at the same program point -- stream-ordered on
        // cstm before the verify body, OUTSIDE any captured graph (the fold's
        // launch count varies with the accepted n; everything the graph bakes
        // is init-fixed). Normally a no-op: post_round already folded on the
        // member stream; this covers hand-driven callers (fused_smoke leg B).
        for (int i = 0; i < k; i++) es[i]->flush_fold(cstm);
        // P2 Task 1: coarse per-round phase walls, bracketed by timing
        // events on cstm (records on an in-order stream do not reorder or
        // synchronize any work -- they only timestamp):
        //   ev_round_start .. ev_draft_end = the cstm-visible DRAFT wait
        //     (ev_draft_end is recorded by fused_verify_round right after
        //     its draft_done waits). P2a interleaves the margin loops but
        //     still host-syncs every step (draft_widths above), so this
        //     span is only the unsynced draft tail (floor top-up launches /
        //     suffix prep+H2D); the concurrent margin-loop wall lives
        //     host-side in draft_widths, before ev_round_start exists.
        //   ev_draft_end .. ev_verify_end = fused VERIFY: union sweep +
        //     per-engine mixers/tails + the outcome D2H enqueue.
        // P3 T3 GRAPH-MODE NOTE (Q27_BATCH_GRAPH=1): the phd/phv brackets
        // keep their meaning under graph replay -- BOTH records stay
        // OUTSIDE the capture at the same program points (graph_round
        // hoists the draft_done waits + the ev_draft_end record before
        // BeginCapture), so a replayed round never re-records them and no
        // stale timestamp node is frozen into a graph. ONE semantics
        // delta: on a MISS round the phv span absorbs the one-time
        // capture+instantiate host stall (measured 2026-08-18 at k=4:
        // 20-28 ms, mean ~23 -- the earlier "~2.4 ms median, T1" was 10x
        // stale; see BUILDLOG (j)) while the
        // GPU idles between ev_draft_end and the graph launch; HIT rounds
        // show the true replay wall. GenStats phd/phv/phs keep their
        // shared-wall per-member attribution unchanged.
        CUDA_CHECK(cudaEventRecord(ev_round_start, cstm));
        // P2b: hand the side-stream pool to the verify round so each
        // engine's mixers fork off cstm per layer (rationale + audit at
        // MixerFork / fused_verify_round).
        MixerFork mfork{side_, ev_fork_, ev_mix_};
        if (!graphs_on_) {
            fused_verify_round(es, granted, k, cstm, evs, sfx, sampled, ev_draft_end,
                               &mfork);
        } else {
            // P3 T3: shape-keyed exec cache -- hit: guard + replay; miss:
            // capture-without-execute + instantiate + launch THIS round.
            graph_round(es, granted, k, evs, sfx, sampled, &mfork);
        }
        int oc[ConductorCore<Member>::MAX_K][OUTCOME_INTS];
        for (int i = 0; i < k; i++)
            CUDA_CHECK(cudaMemcpyAsync(oc[i], es[i]->outcome_dev(), OUTCOME_INTS * 4,
                                       cudaMemcpyDeviceToHost, cstm));
        CUDA_CHECK(cudaEventRecord(ev_verify_end, cstm));
        CUDA_CHECK(cudaStreamSynchronize(cstm)); // ONE sync for the batch
        float ph_d = 0.f, ph_v = 0.f; // this round's phase walls (ms)
        CUDA_CHECK(cudaEventElapsedTime(&ph_d, ev_round_start, ev_draft_end));
        CUDA_CHECK(cudaEventElapsedTime(&ph_v, ev_draft_end, ev_verify_end));
        for (int i = 0; i < k; i++) {
            // Q27_PHASE_STATS (P2 Task 1), SHARED-WALL semantics: a fused
            // round has ONE wall, attributed IN FULL to EACH member's gs --
            // phd/phv answer "how long did THIS request's rounds spend in
            // each phase" (matching the [req] per-request parse). Summing
            // phd/phv ACROSS concurrently-batched requests double-counts
            // the wall. phs stays honest per-member (steps THIS member
            // launched). Runs BEFORE the try block below for the same
            // close-edge reason as the bat counters: gs must be final
            // before any queue op can let the request thread proceed.
            if (es[i]->phase_stats_on()) {
                // launched = min(cap+1, md): exact identity with
                // draft_and_gate's margin loop (a sub-theta break at
                // step k has counted that step, cap+1; a full run is
                // md). Excludes floor top-up launches, mirroring the
                // solo dexit accounting (engine.cuh, gs.draft_steps +=
                // launched). gate_cap < 0 = suffix round, no MTP steps.
                long ph_s = 0;
                if (ms[i]->gate_cap >= 0) {
                    ph_s = ms[i]->gate_cap + 1;
                    if (ph_s > ms[i]->md_used) ph_s = ms[i]->md_used;
                }
                es[i]->phase_stats_add(ph_d, ph_v, ph_s); // A4 accessor
                // draft_widths() ran immediately before this round on this
                // thread (core.round() pairs them), so fdraft_ms_ is THIS
                // round's draft wall. Shared-wall, same as phd/phv.
                es[i]->phase_stats_add_fdraft(fdraft_ms_);
            }
            // Task 10 [req] bat= telemetry FIRST: this member's round ran
            // k-wide (k >= 2 by the core's dispatch; on the catch path the
            // GPU work also already ran -- the throw is host bookkeeping).
            // The counters must be updated BEFORE any queue op: fail_member
            // in the catch arms closes the queue, the synchronization edge
            // after which the request thread may destroy the frame-local
            // DecodeTask (fail_member's close-edge rule) -- a post-catch
            // increment was a write-after-close UAF.
            ms[i]->t->bat_members += k;
            ms[i]->t->bat_r2++;
            // A2 unwind (addendum A2, review M1): per-member host bookkeeping
            // (commit_outcome + post_round, whose sink runs on_emit + the
            // queue push). A throw here fails THIS member's queue via
            // fail_member (error epilogue, then fail() as the LAST q/t
            // access), removes the member via done[i], and lets the loop
            // continue to members i+1..k-1 -- the conductor thread stays
            // alive for everyone else. CUDA errors are NOT wrapped:
            // CUDA_CHECK exits the process (A2 keeps them fatal).
            try {
                int em[W_MAX];
                int n = es[i]->commit_outcome(oc[i], em, ms[i]->sampled, sfx[i],
                                              ms[i]->gate_cap, ms[i]->md_used);
                done[i] = !es[i]->post_round(*ms[i]->t, em, n);
            } catch (const std::exception& ex) {
                fail_member(*ms[i], ex.what());
                done[i] = true;
            } catch (...) {
                fail_member(*ms[i], "unknown host exception in fused round bookkeeping");
                done[i] = true;
            }
        }
        round_active = false; // B3: always reached -- the catch arms above
                              // swallow host exceptions per member and
                              // CUDA_CHECK exits the process, never throws
    }

    // ------------------------------------------------------------------
    // P3 T3 (plan 2026-07-16-batch-p3-capture.md): shape-keyed fused-verify
    // exec cache, behind Q27_BATCH_GRAPH=1 (ctor latch graphs_on_).
    //
    // WHY A SHAPE KEY IS SUFFICIENT (the T1-proven invariance): for a fixed
    // key the fused verify launch sequence is ROUND-INVARIANT -- every
    // pointer build_union_view bakes is an init-fixed engine member, and
    // every round-varying input (tokens, positions, d_P, KV rows, margins,
    // sampled Philox state, tool masks d_mask_ids/d_accept_cap) is DEVICE-
    // resident and staged outside the capture: the same invariance the solo
    // graph zoo has always replayed on. The ONE host-resolved per-round
    // value was the GDN role-pointer set -- gone entirely under M1
    // record+fold (committed/arena pointers are init-fixed); the pre-round
    // flush_fold is the round's mutable input and stays OUTSIDE the graph.
    //
    // KEY COMPONENTS (== the T1 census key: 28 keys/KV live alphabet,
    // top-32 = 100%, so the cap-32 LRU holds the whole zoo):
    //   eng[k]  ordered engine tuple -- capture bakes each engine's lane
    //           buffers into specific union slots and its mix/tail launches
    //           in slot order; a different tuple is a different pointer
    //           program, not a variant of the same one.
    //   gw[k]   exact granted-width vector -- per-lane grids, the union vw,
    //           each member's mix lane walk and tail width are baked from
    //           it (the solo analog: verify_graph_w is per-width).
    //   sfx[k]  suffix class per member -- with k and the Q27_BATCH_GEMM
    //           latch it determines the union's gemm_min family (gemm_min =
    //           f(sfx vector, k, latch): all-suffix / k >= 3 / latch=1 ->
    //           vgemm(2), else GEMV(99)), i.e. WHICH mm5 branch the capture
    //           recorded. sfx and k are BOTH key components; the latch is
    //           process-constant -- family is fully determined by the key.
    //           Any future policy input must itself be a keyed component
    //           (vw, e.g., is NOT keyed -- it is derived from gw[]), or
    //           union_view_eq becomes the only line of defense.
    //   smp[k]  sampled mask -- per-member tail kernel choice
    //           (spec_verify_tail vs _sampled), baked at capture.
    //   kvk[k]  per-engine kv_kind -- attn_mix's kv_store/attn_decode
    //           kernel-family branch. Init-fixed per engine, keyed anyway
    //           (the plan's key spec): an embedder recycling an Engine*
    //           address for a differently-configured engine must MISS.
    // No hash: the alphabet is <= 32 entries, a linear key_eq scan per
    // round is ~free next to a 20+ ms round.
    enum { GK_MAX = ConductorCore<Member>::MAX_K };
    struct GraphKey {
        int k = 0;
        Engine* eng[GK_MAX] = {};
        int gw[GK_MAX] = {};
        unsigned char sfx[GK_MAX] = {};
        unsigned char smp[GK_MAX] = {};
        int kvk[GK_MAX] = {};
    };
    // field-wise, never memcmp: the structs carry alignment padding and
    // only [0..k) of each array is live
    static bool key_eq(const GraphKey& a, const GraphKey& b) {
        if (a.k != b.k) return false;
        for (int i = 0; i < a.k; i++)
            if (a.eng[i] != b.eng[i] || a.gw[i] != b.gw[i] || a.sfx[i] != b.sfx[i] ||
                a.smp[i] != b.smp[i] || a.kvk[i] != b.kvk[i])
                return false;
        return true;
    }
    struct GCacheEnt {
        GraphKey key;
        cudaGraph_t graph = nullptr; // kept alive so teardown owns both halves
        cudaGraphExec_t exec = nullptr;
        // ALWAYS-ON hit-guard reference (B8 discipline): the capture-time
        // union-view pointer table. A hit re-derives it from live host state
        // and compares; any mismatch means the exec would replay STALE baked
        // pointers and must never launch. (M1: the per-engine role-table/
        // perm-staging snapshot is gone with the rotation -- GDN launches
        // carry only init-fixed pointers, all covered by the union view.)
        UnionView uv;
        long tick = 0;    // LRU stamp
        size_t nodes = 0; // telemetry ([gcache] miss lines)
    };
    int gc_find(const GraphKey& key) const {
        for (size_t i = 0; i < gcache_.size(); i++)
            if (key_eq(gcache_[i].key, key)) return (int)i;
        return -1;
    }
    void gc_destroy(GCacheEnt& e) {
        CUDA_CHECK(cudaGraphExecDestroy(e.exec));
        CUDA_CHECK(cudaGraphDestroy(e.graph));
    }
    // The ALWAYS-ON hit guard (B8: re-derive, never trust): rebuild the
    // union view from live host state (build_union_view is launch-free
    // pure host work); memcmp-equal against the capture-stored copy or the
    // hit is refused. M1: the role-table/perm checks are gone with the
    // rotation -- every GDN pointer a captured round consumes (committed
    // state, record arena, lane buffers) is an init-fixed engine member,
    // and the fold-staging expectation is enforced host-side (fused_round
    // enqueues flush_fold(cstm) before this guard runs, which zeroes
    // fold_pending).
    bool gc_guard_ok(const GCacheEnt& e, Engine** es, const int* granted, int k,
                     const bool* sfx) const {
        UnionView now = build_union_view(es, granted, k, cstm, sfx);
        return union_view_eq(e.uv, now);
    }
    static void gc_gwstr(char* buf, size_t n, const int* granted, const bool* sfx,
                         const bool* sampled, int k) {
        int off = 0;
        buf[0] = 0;
        for (int i = 0; i < k && off < (int)n - 8; i++)
            off += snprintf(buf + off, n - off, "%s%d%c%s", i ? "," : "", granted[i],
                            sfx[i] ? 's' : 'g', (sampled && sampled[i]) ? "*" : "");
    }
    // Graph-mode verify body. CAPTURE BOUNDARY (plan + T1 spike, verbatim):
    //   OUTSIDE, BEFORE: the draft phase (its per-step host syncs are a
    //     hard boundary -- the 07-14 GPU-side-depth NO-GO stands); the
    //     draft_done waits (a cudaStreamWaitEvent on an event recorded
    //     outside the capture is capture-illegal -- hoisted here);
    //     flush_fold (already enqueued by fused_round: the round's mutable
    //     input -- its launch count varies with the accepted n, so it can
    //     never live inside a capture); the ev_round_start/ev_draft_end
    //     timing records (replay must never re-record them; fused_round's
    //     GenStats note).
    //   INSIDE: the whole fused_verify_round body, incl. the P2b mixer
    //     fork/join side-stream choreography (T0/T1-proven topology) and
    //     the M1 chunk-kernel GDN launches (init-fixed pointers only),
    //     under Relaxed capture mode (the T0-proven mode).
    //   OUTSIDE, AFTER: outcome D2Hs + ev_verify_end + the round's ONE
    //     host sync (the pageable-gating caveat: host gating rides on
    //     those blocking D2H semantics -- never capture or move them).
    // Miss = capture-without-execute, instantiate, then LAUNCH the fresh
    // exec for the SAME round (T1: bitwise on first sight), insert LRU.
    // Hit = guard, then replay. Guard trip = fprintf + evict + eager
    // fallback for this round; a stale exec is never launched.
    void graph_round(Engine** es, const int* granted, int k, const cudaEvent_t* evs,
                     const bool* sfx, const bool* sampled, const MixerFork* mf) {
        static const bool dbg = [] { // the [bat]/Q27_BATCH_DBG latch pattern
            const char* e = getenv("Q27_BATCH_DBG");
            return e && atoi(e) != 0;
        }();
        gc_rounds_++;
        for (int i = 0; i < k; i++) CUDA_CHECK(cudaStreamWaitEvent(cstm, evs[i], 0));
        CUDA_CHECK(cudaEventRecord(ev_draft_end, cstm));
        GraphKey key;
        key.k = k;
        for (int i = 0; i < k; i++) {
            key.eng[i] = es[i];
            key.gw[i] = granted[i];
            key.sfx[i] = sfx[i] ? 1 : 0;
            key.smp[i] = (sampled && sampled[i]) ? 1 : 0;
            key.kvk[i] = es[i]->kv_cache_kind();
        }
        char gws[96];
        gc_gwstr(gws, sizeof gws, granted, sfx, sampled, k);
        const int hit = gc_find(key);
        if (hit >= 0) {
            if (gc_guard_ok(gcache_[hit], es, granted, k, sfx)) {
                gc_hits_++;
                gcache_[hit].tick = ++gc_tick_;
                CUDA_CHECK(cudaGraphLaunch(gcache_[hit].exec, cstm));
                if (dbg)
                    fprintf(stderr,
                            "[gcache] r=%ld hit  k=%d gw=%s h=%ld m=%ld ev=%ld gt=%ld\n",
                            gc_rounds_, k, gws, gc_hits_, gc_misses_, gc_evictions_,
                            gc_guard_trips_);
                return;
            }
            // guard trip: NEVER launch a stale exec. Loud regardless of dbg.
            gc_guard_trips_++;
            fprintf(stderr,
                    "[gcache] GUARD TRIP r=%ld k=%d gw=%s -- capture-stored union-view "
                    "pointers != re-derived host state; evicting key, eager "
                    "fallback this round (h=%ld m=%ld ev=%ld gt=%ld)\n",
                    gc_rounds_, k, gws, gc_hits_, gc_misses_, gc_evictions_,
                    gc_guard_trips_);
            gc_destroy(gcache_[hit]);
            gcache_.erase(gcache_.begin() + hit);
            // Defensive re-flush before the eager fallback (idempotent:
            // fused_round already flushed; a broken caller path that skipped
            // it would otherwise "recover" into a verify over unfolded
            // state).
            for (int i = 0; i < k; i++) es[i]->flush_fold(cstm);
            fused_verify_round(es, granted, k, cstm, /*draft_done=*/nullptr, sfx,
                               sampled, /*ev_draft_end=*/nullptr, mf);
            return;
        }
        // MISS. No warmup pre-capture pass exists anywhere (deliberate
        // deviation from the plan's "optional startup pre-capture" line):
        // first-sight capture: 2026-08-18 [gcache] telemetry measures 20-28 ms
        // at k=4 (the "~2.4 ms median (T1)" figure was 10x stale) -- warmup-class
        // already -- and a pre-capture pass would need synthetic round
        // state for shapes that may never arrive. YAGNI.
        gc_misses_++;
        const auto t0 = std::chrono::steady_clock::now();
        // Evict-at-cap BEFORE capture/instantiate (2026-07-17 4-lane crash):
        // the old evict-after order held cap+1 execs at the boundary, so a
        // headroom-shrunk cap still overflowed by one exec's allocation on
        // the instantiate -- the ctor guard shrank to 20 and the server
        // died instantiating entry 21.
        auto evict_lru = [&] {
            size_t victim = 0;
            for (size_t i = 1; i < gcache_.size(); i++)
                if (gcache_[i].tick < gcache_[victim].tick) victim = i;
            gc_evictions_++;
            if (dbg)
                fprintf(stderr, "[gcache] r=%ld evict (cap %d, LRU tick %ld)\n",
                        gc_rounds_, gc_cap_, gcache_[victim].tick);
            gc_destroy(gcache_[victim]);
            gcache_.erase(gcache_.begin() + victim);
        };
        cudaGraph_t g = nullptr;
        cudaGraphExec_t x = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(cstm, cudaStreamCaptureModeRelaxed));
        fused_verify_round(es, granted, k, cstm, /*draft_done=*/nullptr, sfx, sampled,
                           /*ev_draft_end=*/nullptr, mf);
        CUDA_CHECK(cudaStreamEndCapture(cstm, &g));
        const auto t_cap = std::chrono::steady_clock::now();
        // E2.2 EXEC RECYCLING. Measured 2026-08-19: a miss is ~3.1 ms capture +
        // ~6.1 ms INSTANTIATE, while cudaGraphExecUpdate is 0.94 ms -- the
        // expensive half is exactly the half an update skips. Graphs whose
        // per-lane widths are a PERMUTATION of each other have identical
        // topology (measured: 234/234 cross-permutation updates succeed), and
        // the C=4 shape space is 89 exact keys over just 30 topology classes.
        // So rather than evicting an exec and instantiating a new one, re-point
        // the LRU same-class exec at this round's pointers and re-key it.
        // Exact-key hits stay free; only the miss path changes.
        //
        // Safety: cudaGraphExecUpdate must not touch an exec pending execution.
        // Every fused round ends with the outcome D2H + host sync on cstm
        // before the next round captures, and round_active asserts rounds never
        // overlap -- so no cached exec is in flight here.
        static const bool recycle_on = [] { // Q27_GC_RECYCLE=0 -> always instantiate
            const char* e = getenv("Q27_GC_RECYCLE");
            return !(e && atoi(e) == 0);
        }();
        auto same_class = [&](const GraphKey& a, const GraphKey& b) {
            if (a.k != b.k) return false;
            int va[GK_MAX], vb[GK_MAX];
            unsigned ta[GK_MAX], tb[GK_MAX];
            for (int i = 0; i < a.k; i++) {
                va[i] = a.gw[i];
                vb[i] = b.gw[i];
                // sfx/smp/kvk select WHICH kernels run, not just their extents,
                // so they must match as a multiset too.
                ta[i] = ((unsigned)a.sfx[i] << 24) | ((unsigned)a.smp[i] << 16) | (unsigned)a.kvk[i];
                tb[i] = ((unsigned)b.sfx[i] << 24) | ((unsigned)b.smp[i] << 16) | (unsigned)b.kvk[i];
            }
            std::sort(va, va + a.k); std::sort(vb, vb + b.k);
            std::sort(ta, ta + a.k); std::sort(tb, tb + b.k);
            for (int i = 0; i < a.k; i++)
                if (va[i] != vb[i] || ta[i] != tb[i]) return false;
            return true;
        };
        // ONLY recycle an entry the cache was about to destroy anyway.
        //
        // v1 of this recycled the LRU same-class entry on EVERY miss, which
        // measured 4.6% SLOWER (round 28.28 vs 27.04 at C=4): re-keying an
        // entry steals a shape that would later have been a FREE exact-key hit.
        // The control leg showed 138 of 191 rounds were hits, and v1 turned
        // them into 139 recycles at ~6.5 ms each -- strictly more work than 53
        // misses at ~14 ms. Caching a recurring shape beats re-pointing it as
        // soon as it recurs ~3 times, and here shapes recur ~6 times.
        //
        // So gate on the cache being FULL. At capacity every miss already pays
        // evict + instantiate, i.e. an exec is destroyed regardless; recycling
        // the LRU same-class victim replaces ~6 ms of instantiate with 0.94 ms
        // of update and costs no hit that was not already being given up.
        int recycle = -1;
        if (recycle_on && (int)gcache_.size() >= gc_cap_)
            for (size_t i = 0; i < gcache_.size(); i++)
                if (same_class(gcache_[i].key, key) &&
                    (recycle < 0 || gcache_[i].tick < gcache_[recycle].tick))
                    recycle = (int)i;
        if (recycle >= 0) {
            cudaGraphExecUpdateResultInfo info{};
            cudaError_t uerr = cudaGraphExecUpdate(gcache_[recycle].exec, g, &info);
            if (uerr == cudaSuccess) {
                GCacheEnt& ent = gcache_[recycle];
                CUDA_CHECK(cudaGraphDestroy(ent.graph)); // the shape it used to hold
                ent.graph = g;
                ent.key = key;
                ent.uv = build_union_view(es, granted, k, cstm, sfx); // guard reference
                ent.tick = ++gc_tick_;
                CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &ent.nodes));
                gc_recycles_++;
                const auto t_r = std::chrono::steady_clock::now();
                CUDA_CHECK(cudaGraphLaunch(ent.exec, cstm)); // THIS round runs via it
                if (dbg)
                    fprintf(stderr,
                            "[gcache] r=%ld recycle k=%d gw=%s nodes=%zu cap+upd=%.2fms "
                            "(cap=%.2f upd=%.2f) h=%ld m=%ld rc=%ld rcf=%ld\n",
                            gc_rounds_, k, gws, ent.nodes,
                            std::chrono::duration<double, std::milli>(t_r - t0).count(),
                            std::chrono::duration<double, std::milli>(t_cap - t0).count(),
                            std::chrono::duration<double, std::milli>(t_r - t_cap).count(),
                            gc_hits_, gc_misses_, gc_recycles_, gc_recycle_fails_);
                return;
            }
            (void)cudaGetLastError(); // refused: fall through to a real instantiate
            gc_recycle_fails_++;
        }
        while ((int)gcache_.size() >= gc_cap_) evict_lru();
        // Instantiate is the one device allocation on the round path that can
        // OOM on a tight boot (this line killed a 4x49152 server mid-traffic,
        // BUILDLOG 2026-07-17). Shrink-never-abort, same recovery shape as
        // the guard trip above: clear the sticky error, evict LRU and retry;
        // if the cache is EMPTY and it still OOMs, graphs cannot work at this
        // headroom -- disable the cache for the rest of the run and serve
        // this (and every later) round eagerly. Non-OOM errors keep the
        // loud-abort contract.
        cudaError_t ierr = cudaGraphInstantiate(&x, g, 0);
        while (ierr == cudaErrorMemoryAllocation && !gcache_.empty()) {
            (void)cudaGetLastError();
            fprintf(stderr,
                    "[gcache] r=%ld instantiate OOM -- evicting LRU and retrying "
                    "(%zu entries)\n",
                    gc_rounds_, gcache_.size());
            evict_lru();
            ierr = cudaGraphInstantiate(&x, g, 0);
        }
        if (ierr != cudaSuccess) {
            (void)cudaGetLastError();
            if (ierr != cudaErrorMemoryAllocation) CUDA_CHECK(ierr);
            (void)cudaGraphDestroy(g);
            fprintf(stderr,
                    "[gcache] instantiate OOM with cache empty -- graph cache OFF for "
                    "the rest of this run, eager fused rounds (h=%ld m=%ld ev=%ld)\n",
                    gc_hits_, gc_misses_, gc_evictions_);
            graphs_on_ = false;
            // Defensive re-flush before the eager round (idempotent; the
            // capture above recorded launches without executing them).
            for (int i = 0; i < k; i++) es[i]->flush_fold(cstm);
            fused_verify_round(es, granted, k, cstm, /*draft_done=*/nullptr, sfx,
                               sampled, /*ev_draft_end=*/nullptr, mf);
            return;
        }
        const auto t1 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaGraphLaunch(x, cstm)); // THIS round runs via the exec
        GCacheEnt ent;
        ent.key = key;
        ent.graph = g;
        ent.exec = x;
        ent.uv = build_union_view(es, granted, k, cstm, sfx); // guard reference
        ent.tick = ++gc_tick_;
        CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &ent.nodes));
        gcache_.push_back(ent);
        if (dbg)
            fprintf(stderr,
                    "[gcache] r=%ld miss k=%d gw=%s nodes=%zu cap+inst=%.2fms "
                    "(cap=%.2f inst=%.2f) h=%ld "
                    "m=%ld ev=%ld gt=%ld\n",
                    gc_rounds_, k, gws, ent.nodes,
                    std::chrono::duration<double, std::milli>(t1 - t0).count(),
                    std::chrono::duration<double, std::milli>(t_cap - t0).count(),
                    std::chrono::duration<double, std::milli>(t1 - t_cap).count(), gc_hits_,
                    gc_misses_, gc_evictions_, gc_guard_trips_);
    }

    // Done-path epilogue (finish_decode already ran inside pre_round/
    // decode_step/post_round and stamped gs.end): hand the stop reason to
    // the request thread via the queue close, release the event, free the
    // Member. Runs on the conductor thread only; `owned` needs no lock.
    // q == nullptr: the A2 unwind already failed (closed) the queue -- it
    // must not be touched again (H1 lifetime rule: the request thread may
    // have destroyed it the moment it observed closed).
    void leave(Member& mm) {
        if (mm.q) mm.q->close(mm.e->end_reason());
        CUDA_CHECK(cudaEventDestroy(mm.draft_done));
        for (size_t i = 0; i < owned.size(); i++) {
            if (owned[i].get() == &mm) {
                owned.erase(owned.begin() + i);
                break;
            }
        }
    }

    GpuGate& gate;
    ConductorCore<Member> core;             // conductor-thread-only state
    std::vector<std::unique_ptr<Member>> owned; // conductor-thread-only
    cudaStream_t cstm = nullptr;
    // P2 Task 1 phase-wall pool (TIMING events; ctor/dtor comments) + the
    // B3 one-round-in-flight invariant flag (conductor-thread-only).
    cudaEvent_t ev_round_start = nullptr, ev_draft_end = nullptr,
                ev_verify_end = nullptr;
    // P2b mixer fork/join pool (ctor comment: sizing justification). Sized
    // MAX_K, used [0..k) per round; conductor-thread-only like cstm.
    cudaStream_t side_[ConductorCore<Member>::MAX_K] = {};
    cudaEvent_t ev_fork_ = nullptr, ev_mix_[ConductorCore<Member>::MAX_K] = {};
    bool round_active = false;
    // This round's draft-phase host wall (round-budget T2). Written at the end
    // of draft_widths(), read in fused_round(); core.round() pairs the two on
    // one thread, which is the same invariant round_active already asserts.
    double fdraft_ms_ = 0;
    // P3 T3 exec-cache state (conductor-thread-only after the ctor, like
    // cstm; ctor comment for the graphs_on_/gc_cap_ latches). Counters feed
    // the per-round dbg lines + the teardown summary (gate e telemetry).
    bool graphs_on_ = false;
    int gc_cap_ = 32; // LRU capacity (Q27_BATCH_GRAPH_CAP; headroom-shrunk)
    long gc_tick_ = 0, gc_rounds_ = 0;
    long gc_hits_ = 0, gc_misses_ = 0, gc_evictions_ = 0, gc_guard_trips_ = 0;
    long gc_recycles_ = 0, gc_recycle_fails_ = 0;
    std::vector<GCacheEnt> gcache_;
    std::thread th;
    std::mutex m; // guards join_q + stop + poked (the cross-thread handoff surface)
    std::condition_variable cv;
    std::vector<std::unique_ptr<Member>> join_q;
    bool stop = false;
    bool poked = false; // incremental KV: pages may have been released
};

} // namespace q27
#endif // __CUDACC__
