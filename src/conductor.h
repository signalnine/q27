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
// ENGINE-OWNED surface only -- solo_view()/pre/mix/post/tails/
// set_round_width/draft_and_gate/suffix_propose/commit_outcome/pre_round/
// post_round/decode_step -- no friend access, no raw member reaches from
// this header (consensus addendum A4). The trim policy, TokenQueue and the
// ConductorCore scheduling skeleton below are pure host code (deterministic,
// CPU-tested in tools/test_conductor.cpp); the fused round + the real
// Conductor further down are CUDA-only and compile away under plain g++
// (the __CUDACC__ guard), so the CPU unit test never sees them.

#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
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
// thread should surface instead of a normal finish (nothing sets it at Task
// 9; the server wiring may -- it rides the same close() wakeup).
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
    void close(const char* why) {
        {
            std::lock_guard<std::mutex> lk(m);
            closed = true;
            reason = why;
        }
        cv.notify_one();
    }
    // Close with a host-side error instead of a normal finish (A2 note above).
    void fail(const char* what) {
        {
            std::lock_guard<std::mutex> lk(m);
            error = what;
            closed = true;
            reason = "error";
        }
        cv.notify_one();
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
    enum { MAX_K = 16 }; // >= any legal union (floor-2 lanes under a
                         // W_MAX/W_PLUMB cap); asserted per round
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
        assert(k <= MAX_K);
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
        trim_widths(want, sfx, k, cap);
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
    for (int m = 1; m < k; m++) assert(&es[m]->dm == &es[0]->dm);
    UnionView uv;
    uv.k = k;
    // start from es[0]'s solo view: sane padding for slots >= union width
    uv.view = es[0]->solo_view();
    int u = 0;
    for (int m = 0; m < k; m++) {
        assert(w[m] >= 2 && w[m] <= W_MAX); // per-engine: logits2/roles are W_MAX-lane
        // granted width must already be installed (mix/tail read member vw)
        assert(es[m]->vw == w[m]);
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
    assert(es[0]->d_vgemm_ws != nullptr);
    uv.view.vw = u;
    uv.view.stm = cstm;
    uv.view.vgemm_ws = es[0]->d_vgemm_ws;
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
// greedy tails on cstm. pre/post/ffn_pair are Engine methods but read ONLY
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
inline void fused_verify_round(Engine** es, const int* granted, int k, cudaStream_t cstm,
                               const cudaEvent_t* draft_done, const bool* is_suffix,
                               const bool* sampled = nullptr) {
    for (int m = 0; m < k; m++) CUDA_CHECK(cudaStreamWaitEvent(cstm, draft_done[m], 0));
    UnionView uv = build_union_view(es, granted, k, cstm, is_suffix);
    const Engine::LaneView& v = uv.view;
    Engine& e0 = *es[0];
    const DevTensor& emb = e0.dm.get("token_embd.weight");
    q27k::embed3((const int8_t*)emb.data, (const __half*)emb.scales, v.vtok, N_EMBD,
                 LANESV(v, h), v.stm, v.vw);
    q27k::CP3 Hc LANESV(v, h), Yc LANESV(v, y);
    q27k::P3 Hm LANESV(v, h), X1m LANESV(v, x1);
    for (int il = 0; il < N_LAYER; il++) {
        const float* an = (const float*)e0.T(il, "attn_norm.weight").data;
        q27k::rmsnorm3(Hc, an, X1m, N_EMBD, EPS, v.stm, v.vw);
        if (e0.attn_layer[il]) {
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
    const float* on = (const float*)e0.dm.get("output_norm.weight").data;
    q27k::rmsnorm3(Hc, on, X1m, N_EMBD, EPS, v.stm, v.vw);
    e0.qx5(v, v.x1, N_EMBD);
    const char* vhead = (e0.fast_head && e0.dm.model_has("output_q4.weight"))
                            ? "output_q4.weight"
                            : "output.weight";
    e0.mm5(v, e0.dm.get(vhead), v.lg);
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

    // gate: the server's GpuGate -- prefill chunks time-slice against decode
    // rounds through it, unchanged (design "Scheduler"). cap: the union
    // width cap fed to trim_widths (W_MAX; the W16 build raises it).
    explicit Conductor(GpuGate& gate_, int cap_ = W_MAX) : gate(gate_), core(cap_) {
        CUDA_CHECK(cudaStreamCreate(&cstm)); // created ONCE; all fused rounds
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
    void register_member(Engine* e, Engine::DecodeTask* t, TokenQueue* q) {
        assert(!t->force_plain_sample); // plain-sample rounds have no fused
                                        // path; Q27_SAMPLE_PLAIN is an A/B
                                        // lever, unsupported in batch mode
        auto mm = std::unique_ptr<Member>(new Member());
        mm->e = e;
        mm->t = t;
        mm->q = q;
        mm->sampled = t->sampling;
        t->on_token = [q](int id) {
            q->push(&id, 1);
            return true;
        };
        CUDA_CHECK(cudaEventCreateWithFlags(&mm->draft_done, cudaEventDisableTiming));
        {
            std::lock_guard<std::mutex> lk(m);
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
        for (Member* mm : core.members) {
            mm->t->cancel.store(true);
            mm->pre_round(); // runs finish_decode("cancelled")
            leave(*mm);      // queue close + event destroy + free
        }
        core.members.clear();
    }

    // Solo fallthrough (k==1): decode_step IS today's path -- captured round
    // graphs, spec_round's own bookkeeping/telemetry, tokens through the
    // queue sink. true = done.
    bool solo_round(Member& mm) { return !mm.e->decode_step(*mm.t); }

    // One fused round over k >= 2 members (under the caller's Lease).
    // Sequence per the plan: drafts already ran inside want_width() on each
    // engine's OWN stm; record each draft_done event; fused verify on cstm
    // (which waits on the events); per-engine outcome D2H on cstm + ONE
    // sync; per-member commit_outcome (spec_round's post-outcome mirror,
    // incl. dctl/histograms) + post_round (tokens -> queue via the sink,
    // EOS/budget/client-stop -> done).
    // TODO(Task 10) telemetry skipped in fused rounds, deliberately:
    // Q27_PHASE_STATS buckets (gs.draft_ms/draft_steps/verify_ms/vw_ms/vw_n/
    // sfx_ms/sfx_rounds) -- a fused round's wall is SHARED across members,
    // so per-engine attribution needs a design call, and [sfxdbg]'s propose
    // trace lines. Everything else spec_round mutates (last_pending,
    // sfx_valid/sfx.append, perm, dctl, gate_cap/n/lane hists, sfx_fired/
    // sfx_tok, gs.dec/rounds/cb_ms/end) is mirrored via commit_outcome +
    // post_round.
    void fused_round(Member** ms, const int* granted, const bool* sfx, int k,
                     bool* done) {
        Engine* es[ConductorCore<Member>::MAX_K] = {};
        bool sampled[ConductorCore<Member>::MAX_K] = {};
        cudaEvent_t evs[ConductorCore<Member>::MAX_K] = {};
        for (int i = 0; i < k; i++) {
            es[i] = ms[i]->e;
            sampled[i] = ms[i]->sampled;
            evs[i] = ms[i]->draft_done;
            CUDA_CHECK(cudaEventRecord(evs[i], es[i]->stm));
        }
        fused_verify_round(es, granted, k, cstm, evs, sfx, sampled);
        int oc[ConductorCore<Member>::MAX_K][OUTCOME_INTS];
        for (int i = 0; i < k; i++)
            CUDA_CHECK(cudaMemcpyAsync(oc[i], es[i]->d_outcome, OUTCOME_INTS * 4,
                                       cudaMemcpyDeviceToHost, cstm));
        CUDA_CHECK(cudaStreamSynchronize(cstm)); // ONE sync for the batch
        for (int i = 0; i < k; i++) {
            int em[W_MAX];
            int n = es[i]->commit_outcome(oc[i], em, ms[i]->sampled, sfx[i],
                                          ms[i]->gate_cap, ms[i]->md_used);
            done[i] = !es[i]->post_round(*ms[i]->t, em, n);
        }
    }

    // Done-path epilogue (finish_decode already ran inside pre_round/
    // decode_step/post_round and stamped gs.end): hand the stop reason to
    // the request thread via the queue close, release the event, free the
    // Member. Runs on the conductor thread only; `owned` needs no lock.
    void leave(Member& mm) {
        mm.q->close(mm.e->gs.end);
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
    std::thread th;
    std::mutex m; // guards join_q + stop (the cross-thread handoff surface)
    std::condition_variable cv;
    std::vector<std::unique_ptr<Member>> join_q;
    bool stop = false;
};

} // namespace q27
#endif // __CUDACC__
