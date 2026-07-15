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
// commit_outcome/pre_round/post_round/decode_step/finish_decode -- the last
// only from the A2 catch epilogue, fail_member below) plus the named
// ACCESSORS engine.cuh
// declares for the conductor (shared_dm/is_attn_layer/fast_head_on/
// vgemm_ws/round_width/stream/outcome_dev/end_reason, each with a why-
// comment at its declaration). No friends; every raw-member need is met by
// adding an accessor in engine.cuh, never by reaching into the engine from
// this header (consensus addendum A4). The trim policy, TokenQueue and the
// ConductorCore scheduling skeleton below are pure host code (deterministic,
// CPU-tested in tools/test_conductor.cpp); the fused round + the real
// Conductor further down are CUDA-only and compile away under plain g++
// (the __CUDACC__ guard), so the CPU unit test never sees them.

#include <cassert>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
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
    void push(const int* ids, int n) {
        if (n <= 0) return;
        {
            std::lock_guard<std::mutex> lk(m);
            buf.insert(buf.end(), ids, ids + n);
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
    bool pop(std::vector<int>& out) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !buf.empty() || closed; });
        bool got = !buf.empty();
        out.insert(out.end(), buf.begin(), buf.end());
        buf.clear();
        return got || !closed;
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
    std::vector<int> buf;
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
//   int  want_width();       draft phase (real: suffix_propose() or
//                            draft_and_gate() on the member engine's OWN
//                            stream); returns the want width, >= 2
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
    std::function<void(MemberT**, const int*, const bool*, int, bool*)> fused_round;
    std::function<void(MemberT&)> on_leave;

    void join(MemberT* m) { joins.push_back(m); }

    // One round. Returns the live member count after the round (0 = idle).
    int round() {
        // boundary: joins land here and only here
        for (MemberT* m : joins) members.push_back(m);
        joins.clear();
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
        const int k = (int)members.size();
        if (k == 0) return 0;
        if (k == 1) {
            // solo fallthrough: byte-for-byte today's path (captured
            // graphs); fusion only engages at >= 2 (design "Scheduler").
            if (solo_round(*members[0])) {
                MemberT* gone = members[0];
                members.clear();
                on_leave(*gone);
            }
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
        for (int i = 0; i < k; i++) {
            ms[i] = members[i];
            want[i] = ms[i]->want_width();
            sfx[i] = ms[i]->round_is_suffix();
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
        for (int i = k - 1; i >= 0; i--) {
            if (done[i]) {
                MemberT* gone = members[i];
                members.erase(members.begin() + i);
                on_leave(*gone);
            }
        }
        return (int)members.size();
    }
};

} // namespace q27

// ---------------------------------------------------------------------------
// P1 Task 8: the fused verify round (design doc "Fused round anatomy" 3-4).
// CUDA TUs only -- tools/test_conductor.cpp (g++, CPU) compiles just the trim
// policy above. Engine is header-only, so pulling it in here keeps include
// order a non-issue for future users (server.cu, fused_smoke.cu).
#ifdef __CUDACC__
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
// Q27_BATCH_GEMM=1 overrides to always-vgemm (threshold 2) -- the
// tolerance-class PERF leg for the Task 11 A/B (vgemm is ~flat in width, so
// wide unions want it even where solo took the GEMV).
inline UnionView build_union_view(Engine** es, const int* w, int k, cudaStream_t cstm,
                                  const bool* is_suffix) {
    static const bool force_vgemm = [] {
        const char* e = getenv("Q27_BATCH_GEMM");
        return e && atoi(e) != 0;
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
    // keep the bitwise contract; mixed unions' suffix lanes tolerance-fork).
    bool all_sfx = true; // k >= 1 asserted above, so this is never vacuous
    for (int m = 0; m < k; m++) all_sfx &= is_suffix[m];
    uv.view.gemm_min = (force_vgemm || all_sfx) ? 2 : 99;
    return uv;
}

// One fused GREEDY verify round over engines es[0..k) at granted widths
// granted[0..k), eager (NO graph capture -- P3 territory), on the conductor
// stream cstm. Caller contract (the conductor / smoke driver):
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
// NOTE mm5 dispatch: RESOLVED by the union GEMM-family policy (Task 9) --
// mm5 now compares the union view's vw against view.gemm_min, which
// build_union_view sets per union class (is_suffix[0..k)) so every lane
// stays on the numeric family its solo round took; Q27_BATCH_GEMM=1 is the
// always-vgemm tolerance-class perf override. The old hazard (union >= 9
// silently taking k_vgemm where solo widths <= 8 took the dp4a GEMV) can no
// longer occur; Q27_GEMM_MIN=99 pins in older gates are redundant-but-
// harmless.
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
inline void fused_verify_round(Engine** es, const int* granted, int k, cudaStream_t cstm,
                               const cudaEvent_t* draft_done, const bool* is_suffix,
                               const bool* sampled = nullptr,
                               cudaEvent_t ev_draft_end = nullptr) {
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
    for (int il = 0; il < N_LAYER; il++) {
        const float* an = (const float*)e0.T(il, "attn_norm.weight").data;
        q27k::rmsnorm3(Hc, an, X1m, N_EMBD, EPS, v.stm, v.vw);
        if (e0.is_attn_layer(il)) {
            e0.attn_pre(il, v);
            for (int m = 0; m < k; m++) es[m]->attn_mix(il, cstm);
            e0.attn_post(il, v);
        } else {
            e0.gdn_pre(il, v);
            for (int m = 0; m < k; m++) es[m]->gdn_mix(il, cstm);
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
    // d_v/d_outcome/h_next, and perm-role commit semantics are untouched.
    for (int m = 0; m < k; m++) {
        Engine::LaneView tv = es[m]->solo_view(); // vw already = granted[m]
        tv.stm = cstm;
        if (sampled && sampled[m]) es[m]->spec_verify_tail_sampled(tv);
        else es[m]->spec_verify_tail(tv);
    }
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
        // round. Reuse is sound under the B3 invariant enforced by
        // round_active in fused_round(): exactly ONE fused round is in
        // flight per Conductor (single conductor thread, synchronous round
        // loop -- each round records, syncs cstm, and reads elapsed before
        // returning), so a record can never overwrite a timestamp that is
        // still to be read.
        CUDA_CHECK(cudaEventCreate(&ev_round_start));
        CUDA_CHECK(cudaEventCreate(&ev_draft_end));
        CUDA_CHECK(cudaEventCreate(&ev_verify_end));
        core.solo_round = [this](Member& mm) { return this->solo_round(mm); };
        core.fused_round = [this](Member** ms, const int* granted, const bool* sfx,
                                  int k, bool* done) {
            this->fused_round(ms, granted, sfx, k, done);
        };
        core.on_leave = [this](Member& mm) { this->leave(mm); };
        th = std::thread([this] { run(); });
    }
    ~Conductor() {
        request_stop();
        th.join();
        CUDA_CHECK(cudaEventDestroy(ev_round_start));
        CUDA_CHECK(cudaEventDestroy(ev_draft_end));
        CUDA_CHECK(cudaEventDestroy(ev_verify_end));
        CUDA_CHECK(cudaStreamDestroy(cstm));
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
            sink = [q, oe = std::move(on_emit)](int id) {
                oe(id);
                q->push(&id, 1);
                return true;
            };
        else
            sink = [q](int id) {
                q->push(&id, 1);
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
                q->fail("conductor stopping: registration refused");
                return;
            }
            t->on_token = std::move(sink);
            join_q.push_back(std::move(mm));
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
            GpuGate::Lease lease(gate); // released per round (A7 above)
            core.round();
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

    // One fused round over k >= 2 members (under the caller's Lease).
    // Sequence per the plan: drafts already ran inside want_width() on each
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
    // sfx_valid/sfx.append, perm, dctl, gate_cap/n/lane hists, sfx_fired/
    // sfx_tok, gs.dec/rounds/cb_ms/end) is mirrored via commit_outcome +
    // post_round.
    void fused_round(Member** ms, const int* granted, const bool* sfx, int k,
                     bool* done) {
        // B3 invariant, enforced not commented: exactly ONE fused round in
        // flight per Conductor (single conductor thread, synchronous round
        // loop). The 3-event phase pool is reused every round on the
        // strength of this -- each round records, syncs, and reads elapsed
        // before returning -- so a future pipelining change must trip here
        // loudly instead of silently corrupting timestamps.
        assert(!round_active && "B3: fused rounds must not overlap per Conductor");
        round_active = true;
        Engine* es[ConductorCore<Member>::MAX_K] = {};
        bool sampled[ConductorCore<Member>::MAX_K] = {};
        cudaEvent_t evs[ConductorCore<Member>::MAX_K] = {};
        for (int i = 0; i < k; i++) {
            es[i] = ms[i]->e;
            sampled[i] = ms[i]->sampled;
            evs[i] = ms[i]->draft_done;
            CUDA_CHECK(cudaEventRecord(evs[i], es[i]->stream()));
        }
        // P2 Task 1: coarse per-round phase walls, bracketed by timing
        // events on cstm (records on an in-order stream do not reorder or
        // synchronize any work -- they only timestamp):
        //   ev_round_start .. ev_draft_end = the cstm-visible DRAFT wait
        //     (ev_draft_end is recorded by fused_verify_round right after
        //     its draft_done waits). Under the current SERIAL host drafts,
        //     draft_and_gate syncs every margin step on the member's own
        //     stm before we get here, so this span is only the unsynced
        //     draft tail (floor top-up launches / suffix prep+H2D) -- i.e.
        //     the whole draft phase tail as cstm sees it. Post-P2a overlap
        //     this same bracket becomes the real concurrent-draft wall.
        //   ev_draft_end .. ev_verify_end = fused VERIFY: union sweep +
        //     per-engine mixers/tails + the outcome D2H enqueue.
        CUDA_CHECK(cudaEventRecord(ev_round_start, cstm));
        fused_verify_round(es, granted, k, cstm, evs, sfx, sampled, ev_draft_end);
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
            if (es[i]->phase_stats) {
                es[i]->gs.draft_ms += ph_d;
                es[i]->gs.verify_ms += ph_v;
                if (ms[i]->gate_cap >= 0) {
                    // launched = min(cap+1, md): exact identity with
                    // draft_and_gate's margin loop (a sub-theta break at
                    // step k has counted that step, cap+1; a full run is
                    // md). Excludes floor top-up launches, mirroring the
                    // solo dexit accounting (engine.cuh, gs.draft_steps +=
                    // launched). gate_cap < 0 = suffix round, no MTP steps.
                    int ph_s = ms[i]->gate_cap + 1;
                    if (ph_s > ms[i]->md_used) ph_s = ms[i]->md_used;
                    es[i]->gs.draft_steps += ph_s;
                }
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
    bool round_active = false;
    std::thread th;
    std::mutex m; // guards join_q + stop (the cross-thread handoff surface)
    std::condition_variable cv;
    std::vector<std::unique_ptr<Member>> join_q;
    bool stop = false;
};

} // namespace q27
#endif // __CUDACC__
