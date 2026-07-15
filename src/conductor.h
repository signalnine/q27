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
// The Conductor itself (registry, round loop, token queues) lands in a later
// task and calls a THIN ENGINE-OWNED surface only -- solo_view()/pre/mix/post/
// tails/set_round_width -- no friend access, no raw member reaches from this
// header (consensus addendum A4). The trim policy below is pure host code
// (deterministic, no allocation, CPU-tested in tools/test_conductor.cpp); the
// fused round further down is CUDA-only and compiles away under plain g++
// (the __CUDACC__ guard), so the CPU unit test never sees it.

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

} // namespace q27

// ---------------------------------------------------------------------------
// P1 Task 8: the fused verify round (design doc "Fused round anatomy" 3-4).
// CUDA TUs only -- tools/test_conductor.cpp (g++, CPU) compiles just the trim
// policy above. Engine is header-only, so pulling it in here keeps include
// order a non-issue for future users (server.cu, fused_smoke.cu).
#ifdef __CUDACC__
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
inline void fused_verify_round(Engine** es, const int* granted, int k, cudaStream_t cstm,
                               const cudaEvent_t* draft_done, const bool* is_suffix) {
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
    // per-engine greedy tails: own lane pointers (solo view), granted width,
    // conductor stream -- argmax + finish_round land in each engine's own
    // d_v/d_outcome/h_next, and perm-role commit semantics are untouched.
    for (int m = 0; m < k; m++) {
        Engine::LaneView tv = es[m]->solo_view(); // vw already = granted[m]
        tv.stm = cstm;
        es[m]->spec_verify_tail(tv);
    }
}

} // namespace q27
#endif // __CUDACC__
