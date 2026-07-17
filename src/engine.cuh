// q27 Engine: qwen35 hybrid forward + MTP speculative decode. Header-only
// (all methods inline) so both the CLI and the server can embed it.
#pragma once
#include <algorithm>
#include <atomic>
#include <memory>
#include <chrono>
#include <functional>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_profiler_api.h>

#include "blocks.cuh"
#include "spec3.cuh"
#include "prefill.cuh"
#include "cuda_common.h"
#include "depthctl.h"
#include "suffixdraft.h"
#include "device_model.h"
#include "kernels.cuh"
#include "loader.h"
#include "turbo3.cuh"
#include "vgemm.cuh"

using q27::DevTensor;
using q27::DType;

static constexpr int N_LAYER = 64;
static constexpr int N_EMBD = 5120;
static constexpr int N_FFN = 17408;
static constexpr int N_HEAD = 24, N_KV = 4, HEAD_DIM = 256;
static constexpr int N_ROT = 64;
static constexpr float FREQ_BASE = 1e7f;
static constexpr float EPS = 1e-6f;
static constexpr int GDN_CH = 10240, GDN_V = 6144, GDN_HEADS = 48, GDN_DIM = 128;
static constexpr int VOCAB = 248320;
static constexpr int MAX_GEN_TRACK = 65536;
// width-12 (plan 2026-07-10): max verify width = lane count = GDN role count
// = perm modulus. The MTP draft ladder stays policy-capped at 4..7
// (D_MAX_MTP); widths 9..12 are reserved for the suffix drafter (P1).
// Q27_W_MAX build knob (2026-07-11): narrow builds for smaller cards.
// Each width costs one GDN role set (~157MB/engine) AND one perm's worth
// of the graph zoo (~1.5x captures from 8->12), so W_MAX=8 reclaims
// ~1.5-2GB -- the difference between fitting and OOMing a 24GB 3090.
// Floor 8 keeps the full maxd7 (4..7) ladder; cap 12 = the lane plumbing.
// Struct arrays stay p[16]; the unused high slots are never dereferenced.
#ifndef Q27_W_MAX
#define Q27_W_MAX 12
#endif
static constexpr int W_MAX = Q27_W_MAX;
static constexpr int D_MAX_MTP = 7;
static_assert(W_MAX >= 8 && W_MAX <= W_PLUMB,
              "Q27_W_MAX in [8, W_PLUMB] (>=8 for the maxd7 ladder, <= the lane plumbing)");
// LANE PLUMBING is FIXED at W_PLUMB (16, cuda_common.h), independent of W_MAX.
// The per-lane verify buffers, the p[16] kernel structs, and the finish-kernel
// outcome layout ({n, t1, dr1..dr15, pending} = 18 ints) are always W_PLUMB
// wide; W_MAX only caps how many lanes go LIVE (verify width) + the role/perm/
// graph count that scales memory. Arrays that LIST all lane pointers or read
// the fixed outcome use W_PLUMB; arrays sized by live width use W_MAX.
// W_PLUMB-wide lane-pointer aggregate from one array member (audit refactor):
// expands to the exact brace list the q27k::*3 kernel wrappers take.
static_assert(W_PLUMB == 16, "LANESW lists 16 slots -- keep it in step with W_PLUMB");
#define LANESW(F)                                                              \
    {{F##_L[0], F##_L[1], F##_L[2], F##_L[3], F##_L[4], F##_L[5], F##_L[6],    \
      F##_L[7], F##_L[8], F##_L[9], F##_L[10], F##_L[11], F##_L[12],           \
      F##_L[13], F##_L[14], F##_L[15]}}
// View-side twin of LANESW (P0 batching, design 2026-07-14): the identical
// 16-slot brace list read from a LaneView array field V.F instead of the
// member arrays, so the weight-sweep halves (pre/post) can run over a union
// view. Solo views copy the member pointers, so the expansion is
// pointer-identical to LANESW there.
#define LANESV(V, F)                                                           \
    {{(V).F[0], (V).F[1], (V).F[2], (V).F[3], (V).F[4], (V).F[5], (V).F[6],    \
      (V).F[7], (V).F[8], (V).F[9], (V).F[10], (V).F[11], (V).F[12],           \
      (V).F[13], (V).F[14], (V).F[15]}}

struct Engine {
    // P10-A1: weights (Model + DeviceModel) are shared read-only across slots.
    // The owning ctor keeps them in owned_*; the borrowing ctor binds refs to
    // a caller-owned pair so two Engines share one 17.7 GB weight set. All
    // per-slot mutable state below stays per-Engine.
    std::unique_ptr<q27::Model> owned_model;
    std::unique_ptr<q27::DeviceModel> owned_dm;
    q27::Model& model;
    q27::DeviceModel& dm;
    int max_ctx;
    bool attn_layer[N_LAYER + 1] = {false};
    cudaStream_t stm;
    cudaGraphExec_t graph_exec = nullptr;
    cudaGraphExec_t sample_graph = nullptr; // plain forward + sample (temp>0)
    q27k::WyScratch wy_scratch; // per-engine WY prefill panels (R1b prereq)

    // activations (device)
    float *h, *x1, *y, *qg, *kbuf, *vbuf, *attnout, *scratch;
    float *qkv, *convout, *z, *alpha, *betar, *g, *beta, *o, *og;
    float *ffn_g, *ffn_u, *logits;
    // device decode state
    int *d_pos, *d_token, *d_step, *d_gen;
    unsigned long long* d_amax;
    // Sampling (roadmap #2). samp is the per-request config the server sets
    // before generate(); inv_temp<=0 => greedy (spec path, bitwise). d_samp is
    // the device param block read by the captured sample_graph; d_nuc holds the
    // per-step {thresh,M,logZ}. All idle for greedy requests.
    q27k::SampleParams samp{0.f, 1.f, 0ull};
    q27k::SampleParams* d_samp = nullptr;
    float* d_nuc = nullptr;  // [5][4]: {thresh,M,logZ,mass} per verify lane
    int* d_spec = nullptr;   // [3]: {n, stop_lane, exclude_token} (Phase-2 verdict)
    bool samp_first = false; // first sampled token comes from the retained prefill logits
    // MTP draft head state (stage 1: host-driven acceptance measurement)
    float *h_next, *e_hn, *x_mtp, *mtp_logits;
    void *mtp_k, *mtp_v;
    int *d_pos_m, *d_draft;
    // speculative decode (depth-1): b-token buffers, spare GDN state, batch quant
    // audit 2026-07-12: per-lane buffers live in 12-wide arrays now
    // ([0] aliases the primary lane's pointer, set after allocation).
    // Was ~200 named members + 12-wide brace lists at every call site.
    std::array<float*, W_PLUMB> h_L, x1_L, y_L, qg_L, kbuf_L, vbuf_L,
        attnout_L, qkv_L, convout_L, z_L, alpha_L, betar_L, g_L, beta_L,
        o_L, og_L, ffn_g_L, ffn_u_L;
    std::array<int*, W_PLUMB> d_pos_L, d_v_L;
    std::array<q27k::XQuant, W_PLUMB> xq_L; // [0],[1] alias xq2[0],xq2[1]
    float *logits2, *y2big;
    // P2 (docs/plans/2026-07-13-gemm-verify.md): the flat-in-W verify weight path.
    // The batched GEMV is register-bound at width -- 15.7 ms of a 21.4 ms width-12
    // round (nsys), collapsing to 444 GB/s at W=16 against a 1453 GB/s SOL. k_vgemm
    // is FLAT (11.8 -> 12.5 ms over widths 5..16). gemm_min is the width at which
    // mm5 switches: 9, i.e. STRICTLY ABOVE the ladder's max verify width
    // (gate_maxd+1 <= 8), so every gated/draft/sampled round keeps the GEMV and the
    // canonical bitwise gate holds BY CONSTRUCTION rather than by hope. Only suffix
    // rounds (which verify at sfx_width() == W_MAX) reach the GEMM.
    // gemm_min_rows keeps the tiny attn_k/attn_v (1024 rows) on the GEMV -- they
    // cannot fill a K-split grid, and they are only 170 MB/round.
    float* d_vgemm_ws = nullptr;
    int gemm_min = 9;
    int64_t gemm_min_rows = 4096;
    // GDN role state, spare sets 1..11 (role 0 = S/conv_ring). Roles 8..11
    // (indices 7..10) exist only when Q27_W_MAX admits them -- nullptr else.
    // Was 11 pairs of named members + ternary chains (audit 2026-07-12).
    float *S_sp[W_PLUMB - 1][N_LAYER], *ring_sp[W_PLUMB - 1][N_LAYER];
    float *h_next2;
    int *d_pos_m2, *d_draft2;
    // depth-3 lane (d): 4th verify column + pass-3 draft chain
    float *h_next3;
    int *d_pos_m3, *d_draft3;
    // depth-4 lane (e): 5th verify column + pass-4 draft chain (P3)
    float *h_next4;
    int *d_pos_m4, *d_draft4;
    // depth-5 lane (f): 6th verify column + pass-5 draft chain (P12b)
    float *h_next5;
    int *d_pos_m5, *d_draft5;
    // depth-6 lane (g): 7th verify column + pass-6 draft chain (maxd6 ladder)
    float *h_next6;
    int *d_pos_m6, *d_draft6;
    // depth-7 lane (h): 8th verify column + pass-7 draft chain (maxd7 ladder)
    float *h_next7;
    int *d_pos_m7, *d_draft7;
    // wide lanes (verify columns 9..W_PLUMB). VERIFY-ONLY -- no h_nextN (the
    // finish h_next select reads x1_*; the MTP chain never extends past depth
    // 7) and no d_pos_mN. These draft slots exist so the suffix drafter can
    // stage proposals 8..W_PLUMB-1; the MTP chain never writes them.
    // W16: the named d_draft8..11 members became this array -- k_finish_round
    // dereferences EVERY slot < W_PLUMB-1 unconditionally, so the widening has
    // to fill all of them, and a by-name brace list is exactly the landmine
    // the 8->12 widening tripped over. d_draft_L[k] is draft k+1 (d_draft_L[0]
    // == d_draft); the MTP chain still writes d_draft..d_draft7 by name.
    std::array<int*, W_PLUMB - 1> d_draft_L;
    int *d_P, *d_outcome;
    q27k::XQuant xq2[2];
    // P7 constrained tool decoding: resident mask pool + per-slot ids +
    // acceptance cap. All -1/0 when inactive -> bitwise-identical decode.
    unsigned* d_mask_pool = nullptr;
    int* d_mask_ids = nullptr;
    int* d_accept_cap = nullptr;
    int mask_words = 0, mask_pool_used = 0;
    int h_mask_id0 = -1, h_cap0 = 0; // async-copy sources (must outlive copy)
    static constexpr int MASK_POOL_CAP = 512;
    // GDN state as W_MAX=12 physical buffers with a cyclic role permutation
    // (history: 6 at P12b, 8 at maxd7, 12 at width-12). role r (0=primary,
    // 1..11 = post-b..post-l) -> physical (r+perm)%12. accept n tokens ->
    // perm += n-1 (mod 12). One captured graph per perm. Shallower ceilings
    // use only a role prefix (a bitwise-identical subset -- the modulus only
    // relabels WHICH physical buffer holds a role; every access goes through
    // SBuf/RBuf, so values and emitted tokens are modulus-invariant as long
    // as modulus >= max commit n). Invariant: role 0 = the last-committed
    // state.
    bool fast_head = false; // opt-in: Q4 head for verify too (output may differ)
    bool batched_prefill = true;
    // Minimum prompt tokens for the chunked prefill path (Q27_PF_BATCH_MIN).
    // Below it, prefill walks the prompt serially: two ungraphed forwards +
    // two stream syncs PER TOKEN (~22ms/tok on sm_86, ~11 on sm_120) AND it
    // clears the slot's snapshot + checkpoint ring -- a tiny prompt routed to
    // a slot destroys its conversation cache. The chunked path handles small
    // T already (every long prompt ends in an arbitrary tail chunk). Floor 2:
    // NP=1 would snap_save an empty prefix (have_snap on nothing).
    int pf_batch_min = 32;

    // ---- batched prefill (M6) ----
    // Prefill chunk size. 256 left GEMM launches at ~320 blocks on 170 SMs
    // (27% of int8 peak) and re-read all 17.7GB of weights T/256 times; 1024
    // fills the machine and cuts weight re-reads 4x. Costs ~0.8GB scratch.
    // -D-overridable since 2026-07-17 (sm_86 sweep: 82 SMs fill at smaller T
    // and the scratch is turbo3 ctx budget); defaults unchanged.
#ifndef Q27_PF_T
#define Q27_PF_T 1024
#endif
#ifndef Q27_PF_SB
#define Q27_PF_SB 32
#endif
    static constexpr int PF_T = Q27_PF_T;
    static constexpr int PF_SB = Q27_PF_SB; // attention sub-batch (scratch rows)
    int* d_prompt = nullptr;          // whole prompt on device
    int d_prompt_cap = 0;
    float *hT, *x1T, *yT, *qkvT, *convT, *zT, *oT, *ogT, *qgT, *kT, *vT, *attnT;
    float *alphaT, *betarT, *gT, *betaT, *ffnGT, *ffnUT, *embT, *ehnT, *xmtpT;
    float* pf_part; // P4 split-attention partials: [24 heads][PF_T][SPLIT_MAX][258]
    q27k::XQuant xqT;

    // ---- prefix cache (M6.5): snapshot of GDN state + conv rings taken right
    // after prefill (perm==0), keyed by the prompt tokens it covers. Attention
    // and MTP KV rows are append-only during generation, so prefix rows stay
    // valid; only the recurrent state needs snapshot/restore.
    float* S_snap[N_LAYER] = {};
    float* ring_snap[N_LAYER] = {};
    std::vector<int> snap_toks;
    bool have_snap = false;
    int perm = 0;
    // ---- P3 T2 (capture plan 2026-07-16): device-resolved GDN role tables.
    // d_gdn_tab = ONE flat init-time upload of [2][N_LAYER][W_MAX] float*:
    // ring half first, S half second; entry [il][ph] = the physical buffer
    // RBuf/SBuf return when (role+perm)%W_MAX == ph (attn layers stay
    // nullptr, never indexed). The conv_step_t/delta_step_t twins index
    // (table + il*W_MAX) with *d_perm_scalar, so a captured fused round no
    // longer bakes host-resolved role pointers -- the T3 enabler for
    // cross-round graph exec reuse. Tables are read ONLY when a caller asks
    // gdn_mix for use_tables: the fused path does from T2 on (mix_all in
    // conductor.h, eager first per the plan gate; T3 captures the same
    // launches); the solo path never passes the flag, so its kernels/
    // pointers are byte-for-byte untouched. h_perm_pin is the PINNED
    // per-engine staging int the caller cudaMemcpyAsyncs to d_perm_scalar
    // on cstm before each fused round / graph launch (pinned: no new
    // pageable-blocking semantics; at most one DISTINCT VALUE in flight per
    // round -- the guard-trip fallback may enqueue a second byte-identical
    // copy (perm is constant within a round); the round sync fences the
    // next rewrite).
    float** d_gdn_tab = nullptr;   // base == ring half
    float** d_gdn_S_tab = nullptr; // = d_gdn_tab + N_LAYER*W_MAX
    int* d_perm_scalar = nullptr;
    int* h_perm_pin = nullptr;
    // Stage the CURRENT host perm to the device scalar on st (T3 calls this
    // per capture-mode round, before the graph launch on the same stream).
    void stage_perm_async(cudaStream_t st) {
        *h_perm_pin = perm;
        CUDA_CHECK(cudaMemcpyAsync(d_perm_scalar, h_perm_pin, sizeof(int),
                                   cudaMemcpyHostToDevice, st));
    }
    // ---- GRAPH ZOO (read before any width/depth change: miss one and a decode
    // path silently runs a stale graph). perm is mod-W_MAX=12 (12 GDN state
    // buffers), so every spec/gated set below is [..][perm=0..11]. Two NON-spec
    // single-token graphs live at the top of the struct: `graph_exec` (plain
    // greedy forward, step_free fallback) and `sample_graph` (plain temp>0
    // forward+sample, the non-spec sampled loop). The perm-indexed spec/gated
    // sets and their callers:
    //
    //   spec_graph[12]             monolithic UNGATED GREEDY round (draft to
    //                              gate_maxd + width-5 verify, one graph).
    //                              -> spec_round, ungated branch (Q27_PMIN unset,
    //                                 unconstrained). The default greedy path.
    //   spec_sample_graph[12]      monolithic UNGATED SAMPLED round.
    //                              -> spec_sample_round, ungated branch. Default
    //                                 sampled path.
    //   verify_graph_w[13][12]     per-width GREEDY verify, [W=1..12][perm].
    //                              -> gated greedy round (spec_round), both
    //                                 Q27_DEXIT on and off. Widths 9..12 are
    //                                 suffix-only (captured in P1).
    //   verify_sample_graph_w[6][12] per-width SAMPLED verify, [W=2..5][perm].
    //                              -> gated sampled round (spec_sample_round),
    //                                 both Q27_DEXIT on and off.
    //   draft_step_graph[7][12]    per-draft-STEP graphs, [step=0..gate_maxd-1].
    //                              -> the early-exit loop in BOTH gated rounds
    //                                 (default when Q27_DEXIT on). Launched one
    //                                 step at a time; concatenated back-to-back
    //                                 they reproduce the monolithic draft exactly.
    //   draft_graph[12]            monolithic depth-gate_maxd draft (P11 split).
    //                              -> P11 constrained-tool path; AND the
    //                                 Q27_DEXIT=0 monolithic-draft A/B fallback
    //                                 (greedy + sampled).
    //   draft_graph_lo[12]         monolithic DEPTH-4 draft; captured only when
    //                              gate_maxd==5 (auto or fixed Q27_MAXD=5).
    //                              -> constrained-tool path under auto; the
    //                                 Q27_DEXIT=0 depth-4 fallback (greedy auto
    //                                 md_used==4; sampled gate_maxd==5).
    //   verify_graph[12]           monolithic WIDTH-5 verify (P11 split).
    //                              -> ONLY the P11 constrained-tool path.
    //
    // REDUNDANT-BUT-KEPT after P14: with Q27_DEXIT default-ON, the gated
    // early-exit path drives draft_step_graph, so the monolithic draft_graph /
    // draft_graph_lo are redundant FOR THE GATED PATH -- but both stay live for
    // (a) the P11 constrained-tool path and (b) the Q27_DEXIT=0 A/B baseline, so
    // NEITHER is removable. draft_graph_lo is the closest to dead (its only
    // unique callers are constrained-tool-under-auto + the DEXIT=0 auto/sampled
    // fallback); flagged a removable-candidate in the P14 capstone BUILDLOG
    // entry, deliberately NOT removed here.
    cudaGraphExec_t spec_graph[W_MAX] = {}; // width-12: perm is mod-12 (12 GDN state buffers)
    // P11: split draft/verify graphs for the constrained tool path
    cudaGraphExec_t draft_graph[W_MAX] = {};
    cudaGraphExec_t verify_graph[W_MAX] = {};
    // P12 confidence-gated depth: one verify graph per width W (index [W][perm],
    // W in 1..5). spec_round drafts width-5, reads the 4 draft margins, computes
    // cap = leading run of margin >= pmin_theta, launches verify_graph_w[cap+1].
    // Greedy tokens are width-invariant (lanes are independent grid indices), so
    // this changes only round count + verify width, never the emitted sequence.
    cudaGraphExec_t verify_graph_w[W_MAX + 1][W_MAX] = {}; // [W=1..12][perm=0..11]
    float* d_draft_margin = nullptr; // [7] drafter top1-top2 margins (device)
    float h_draft_margin[7] = {};
    // P14: block-partial scratch for the fused draft argmax+margin (k_argmax_top2
    // -> k_top2_finalize). 128 blocks each emit one (packed top1, top2) pair.
    unsigned long long* d_am_blk1 = nullptr; // [128] packed (top1,idx) per block
    float* d_am_blk2 = nullptr;              // [128] top2 per block
    float pmin_theta = 0.f; // Q27_PMIN; <=0 => gating off (always full width 5)
    // gate_maxd = deepest draft the gate may reach (Q27_MAXD, 4 or 5). Default 4:
    // the robust win on all traffic (depth-4 gate = +10.8% @60K). maxd=5 draws 5
    // MTP passes/round and only pays off on high-acceptance (agentic) traffic
    // (+2.6% agentic, -8% docs), so it is opt-in.
    int gate_maxd = 4;
    // exact bytes of resident GDN recurrent state (all S buffers + conv
    // rings, every set); accumulated at alloc, read by server slot admission
    size_t gdn_state_bytes = 0;
    // Single source for every context guard/clamp (generate() loop, CLI spec
    // loop, server n_max clamps). A width-W verify round launched at P writes
    // attention-KV rows P+1..P+W and MTP-KV rows P+1..P+gate_maxd, so a round
    // may only launch while P + ctx_round_reserve() <= max_ctx. Review
    // 2026-07-09 P0 #1: the depth-5-era hardcoded 7 (and the servers' -6)
    // overran the caches by up to 2 rows when gate_maxd reached 6/7.
    // width-12 P0: the reserve is keyed on the widest LAUNCHABLE verify
    // (verify_w_max), not the draft ceiling -- a suffix round wider than
    // gate_maxd+1 (P1) would otherwise overrun the caches, the exact
    // depth-5-era bug class again.
    // widest LAUNCHABLE verify: the gated width, or the suffix width when
    // the drafter is armed wider (Q27_SUFFIX_W; sfx_width() is declared
    // further down -- in-class bodies see the complete class).
    int verify_w_max() const { return suffix_on ? sfx_width() : gate_maxd + 1; }
    int ctx_round_reserve() const { return std::max(gate_maxd, verify_w_max() - 1) + 2; }
    // P13 adaptive maxd (Q27_MAXD=auto): float the draft-depth ceiling per stream
    // between 4 and 5 from realized acceptance, so agentic streaks get depth-5
    // (+2.6%) while prose stays depth-4 (no -8% draft tax) -- automatically, per
    // stream, with no env retune. Sits on the Q27_PMIN gate (no-op without it).
    // Start shallow; promote 4->5 when depth-4 rounds saturate the ceiling often
    // enough (sat_ema), demote 5->4 when the 5th lane stops paying (yield_ema).
    // The ceiling changes round grouping / draft depth / verify width only -- never
    // the emitted sequence (greedy is width-invariant), so decode stays bitwise.
    bool maxd_auto = false;
    cudaGraphExec_t draft_graph_lo[W_MAX] = {}; // depth-4 draft (auto mode only)
    DepthCtl dctl; // ceiling + EMAs + counters, extracted to depthctl.h for
                   // CPU tests (tools/test_depthctl.cpp). Lifetime =
                   // conversation lineage (review 2026-07-09 + follow-up):
                   // warm state carries across same-conversation turns (a
                   // measured +1.6% on short requests vs per-request
                   // re-earning), and the server's claim_slot resets it when
                   // a non-prefix-restoring request takes the slot over (new
                   // lineage must not inherit the previous tenant's state).
                   // Q27_MAXD_RESET=1 is the stricter every-request reset at
                   // generate() entry.
    bool maxd_reset = false; // Q27_MAXD_RESET=1: reset dctl per request
    // maxd6 GO-IF telemetry (host-side counters only; decode/graphs untouched):
    // per-round margin-run depth (cap, 0..gate_maxd) and accepted length
    // (n, 1..gate_maxd+1) over gated greedy rounds. At a fixed depth-5 ceiling:
    // fired fraction = cap_hist[5]/sum(cap_hist), depth-5 saturation =
    // n_hist[6]/sum(n_hist), p(5th lane accepted | fired) = n_hist[6]/cap_hist[5].
    // width-12 P3 (review pre-req): sized for ANY future ladder ceiling
    // <= W_MAX-1 -- the old [8]/[9] were safe only under the 4..7 policy.
    long gate_cap_hist[W_MAX] = {};    // [cap 0..W_MAX-1]
    long gate_n_hist[W_MAX + 1] = {};  // [n 1..W_MAX]; index 0 unused
    // acceptance-gate Phase 0: per-draft-lane conditional acceptance on gated
    // rounds. Lane j (1..gate_maxd) FIRED iff cap >= j; ACCEPTED iff n >= j+1.
    // Gives the live yields p(acc_j | fired_j) that the two marginals above
    // cannot reconstruct (docs/acceptance-gate-design.md).
    long gate_lane_fired[W_MAX] = {}, gate_lane_acc[W_MAX] = {}; // [j 1..W_MAX-1]; 0 unused
    // Phase 2 (sampling): 2nd fused perm set -- identical draft half, sampled
    // (rejection) verify tail. Captured only when the sampler kernels are warm.
    cudaGraphExec_t spec_sample_graph[W_MAX] = {};
    // P14: per-width sampled verify graphs (sampled analog of verify_graph_w).
    // [W=2..5][perm=0..5]; the sampled+gated round drafts depth-4, reads the 4
    // draft margins, caps the accept walk at W-1, and launches this at width W.
    cudaGraphExec_t verify_sample_graph_w[6][W_MAX] = {}; // [W<=5][perm 0..11] (sampled ceiling stays 4)
    // P14 draft early-exit: one graph per draft STEP (k=0..gate_maxd-1), so the
    // gated rounds can stop drafting at the first sub-theta margin (llama's
    // p_min stops DRAFTING; the P12 gate only narrowed verify). Steps 0..k
    // launched back-to-back on stm reproduce the monolithic draft graph's
    // kernel sequence exactly (see spec_draft_step_launches). Q27_DEXIT=0
    // restores the monolithic draft (A/B lever); default ON when gated.
    cudaGraphExec_t draft_step_graph[D_MAX_MTP][W_MAX] = {}; // [step 0..6][perm 0..11]
    bool dexit_on = true; // Q27_DEXIT (only reached when pmin_theta > 0)
    // Q27_PHASE_STATS=1: per-round draft/verify wall split (Saguaro draft-
    // fraction measurement, survey 2026-07-09). Host steady_clock stamps at
    // the gated round's existing sync boundaries -- no device work, no new
    // syncs, tokens unaffected. Draft bucket = prep + MTP chain + margin
    // reads (the critical-path cost off-path drafting would hide); verify
    // bucket = width-floor top-up (cap==0 rounds only, slight draft
    // undercount) + verify graph + finish + outcome read. Suffix rounds are
    // deliberately NOT phase-stamped (own sfx counters; they'd pollute the
    // per-width verify curve).
    bool phase_stats = false;
    // Q27_SUFFIX=1: zero-model echo drafter (suffixdraft.h). When the
    // committed stream's suffix (incl the pending token) recurs earlier with
    // match >= sfx_L, the earlier continuation fills the draft lanes and the
    // round runs prep + verify only -- NO MTP passes. Correctness-invariant
    // by construction (greedy verify decides every emitted token); trades
    // round count only. Phase-0 (tools/suffix_sim.py): cctx fire ~35% @ AL
    // 11.5 with L=12, docs fire 0% -- silent on neutral traffic. Known
    // Phase-1 gap, measure-first: suffix-committed positions get no MTP KV
    // rows, so the FIRST MTP round after a suffix burst drafts against
    // stale rows (acceptance dip, not correctness); batched MTP re-warm
    // from resident lane hiddens is the Phase-2 fix if replay A/B says so.
    bool suffix_on = false;
    int sfx_L = 12;              // Q27_SUFFIX_L: min match length to fire
    // Q27_SUFFIX_W (width-12 P1): decouple the SUFFIX verify width from the
    // MTP gated width. 0/unset or <= gate_maxd+1 = legacy (suffix rides the
    // gated width); 9..12 = wide suffix rounds (one extra per-perm graph at
    // exactly that width -- suffix rounds always launch full width). The
    // MTP ladder stays 4..gate_maxd regardless.
    int sfx_w = 0;
    int sfx_width() const { return sfx_w > gate_maxd + 1 ? sfx_w : gate_maxd + 1; }
    q27::SuffixDraft sfx;
    bool sfx_valid = false;      // last_pending valid (>=1 round this request)
    long sfx_fired = 0, sfx_tok = 0; // engine-cumulative, like glf/gla
    int h_sfx_prop[W_MAX];       // host staging for the H2D draft copies (W_MAX-1 max)
    bool sfx_dbg = false;        // Q27_SUFFIX_DBG: per-round propose trace
    bool tool_split_active = false; // set by set_tool_constraint when constraining
    float* SBuf(int il, int role) {
        int ph = (role + perm) % W_MAX;
        return ph == 0 ? S[il] : S_sp[ph - 1][il];
    }
    float* RBuf(int il, int role) {
        int ph = (role + perm) % W_MAX;
        return ph == 0 ? conv_ring[il] : ring_sp[ph - 1][il];
    }
    q27k::XQuant xq;
    // layer state
    float* conv_ring[N_LAYER];
    float* S[N_LAYER];
    // P2: attention + MTP KV caches, fp16 by default; fp8 E4M3 when Q27_KV=fp8
    // (34 vs 68 KB/token). Same [pos][kv_head][head_dim] element layout, only
    // the element size changes. NOT lossless -- opt-in, tolerance-gated.
    // turbo3 (Q27_KV=turbo3, phase 1 2026-07-11): 3-bit WHT-rotated blocks
    // (src/turbo3.cuh), row = N_KV*(HEAD_DIM/128) 50-B blocks = 400 B (~13.4
    // vs 68 KB/token); turbo3v = fp16 K + turbo3 V (the GQA=6 K-risk escape,
    // port spec docs/plans/2026-07-11-turbo3-kv-port-spec.md). Phase 1 is
    // DECODE-ONLY: batched prefill has no turbo3 leg yet (guard below), so
    // serving is gated off; quality triage runs over --nll-serial / -n.
    bool kv_fp8 = false;
    int kv_kind = KV_F16;
    size_t kv_esz() const { return kv_fp8 ? 1 : 2; }
    // bytes per K/V cache buffer (one attn layer, one stream). K and V sized
    // separately: turbo3v keeps K fp16 while V goes turbo3.
    size_t kv_bytes(bool is_v) const {
        size_t t3 = (size_t)max_ctx * N_KV * (HEAD_DIM / 128) * sizeof(q27turbo::block_turbo3);
        if (kv_kind == KV_T3 || (kv_kind == KV_T3V && is_v)) return t3;
        return (size_t)max_ctx * N_KV * HEAD_DIM * kv_esz();
    }
    std::vector<void*> kcache, vcache;
    std::vector<int> attn_cache_idx;

    // Owning: self-loads weights (CLI, single-slot).
    Engine(const std::string& path, int ctx)
        : owned_model(std::make_unique<q27::Model>(q27::Model::open(path))),
          owned_dm(std::make_unique<q27::DeviceModel>(*owned_model)),
          model(*owned_model), dm(*owned_dm), max_ctx(ctx < 32 ? 32 : ctx) {  // floor: spec-graph warmup touches ~gate_maxd+2 positions
        init(ctx, /*own_weights=*/true);
    }
    // Borrowing: shares a caller-owned Model+DeviceModel (weights already
    // uploaded by the caller). Multi-slot serving builds N of these.
    Engine(q27::Model& m, q27::DeviceModel& d, int ctx)
        : model(m), dm(d), max_ctx(ctx < 32 ? 32 : ctx) {  // floor: spec-graph warmup touches ~gate_maxd+2 positions
        init(ctx, /*own_weights=*/false);
    }

  private:
    void init(int ctx, bool own_weights) {
        CUDA_CHECK(cudaStreamCreate(&stm));
        const char* kve = getenv("Q27_KV");
        kv_fp8 = kve && !strcmp(kve, "fp8");
        kv_kind = kv_fp8                          ? KV_FP8
                  : kve && !strcmp(kve, "turbo3") ? KV_T3
                  : kve && !strcmp(kve, "turbo3v") ? KV_T3V
                                                   : KV_F16;
        if (kv_fp8) fprintf(stderr, "KV cache: fp8 E4M3 (opt-in, 34 KB/token)\n");
        else if (kv_kind == KV_T3)
            fprintf(stderr, "KV cache: turbo3 3-bit K+V (opt-in, ~13.4 KB/token)\n");
        else if (kv_kind == KV_T3V)
            fprintf(stderr, "KV cache: turbo3 V + fp16 K (opt-in, diagnostic)\n");
        const std::string& mj = model.meta_json;
        size_t p = mj.find("\"attn_layers\": [");
        if (p == std::string::npos) { fprintf(stderr, "no attn_layers in meta\n"); exit(1); }
        p += strlen("\"attn_layers\": [");
        while (p < mj.size() && mj[p] != ']') {
            int v = atoi(mj.c_str() + p);
            if (v <= N_LAYER) attn_layer[v] = true;
            p = mj.find_first_of(",]", p);
            if (mj[p] == ',') p++;
        }

        auto A = [](void** pp, size_t n) { CUDA_CHECK(cudaMalloc(pp, n)); };
        A((void**)&h, N_EMBD * 4); A((void**)&x1, N_EMBD * 4); A((void**)&y, N_EMBD * 4);
        A((void**)&qg, 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf, N_KV * HEAD_DIM * 4); A((void**)&vbuf, N_KV * HEAD_DIM * 4);
        A((void**)&attnout, N_HEAD * HEAD_DIM * 4);
        // flash-decode split-K partials: ntok * heads * FD_NS * FD_ST floats,
        // independent of ctx (sized for 5 lanes; was 3*N_HEAD*max_ctx, which
        // under-allocates whenever max_ctx < FD_NS*FD_ST = 4128)
        A((void**)&scratch, W_MAX * (size_t)N_HEAD * q27k::FD_MAXNS * q27k::FD_ST * 4); // width-12: 12 lanes
        A((void**)&qkv, GDN_CH * 4); A((void**)&convout, GDN_CH * 4); A((void**)&z, GDN_V * 4);
        A((void**)&alpha, GDN_HEADS * 4); A((void**)&betar, GDN_HEADS * 4);
        A((void**)&g, GDN_HEADS * 4); A((void**)&beta, GDN_HEADS * 4);
        A((void**)&o, GDN_V * 4); A((void**)&og, GDN_V * 4);
        A((void**)&ffn_g, N_FFN * 4); A((void**)&ffn_u, N_FFN * 4);
        A((void**)&logits, VOCAB * 4);
        A((void**)&d_pos, 4); A((void**)&d_token, 4); A((void**)&d_step, 4);
        // sized to max_ctx (was fixed MAX_GEN_TRACK=65536): batched prefill's
        // final step_with writes d_gen[NP-1], and NP can reach max_ctx, so any
        // prompt > 65536 with --ctx > 65536 wrote OOB (CUDA-review #1). NP is
        // already bounded <= max_ctx by the generate() guard, so this is exact.
        A((void**)&d_gen, (size_t)max_ctx * 4);
        A((void**)&d_amax, 8);
        A((void**)&d_samp, sizeof(q27k::SampleParams));
        // d_nuc: 5 lanes x {thresh,M,logZ,mass}. Plain path uses lane 0; the
        // sampled spec round (Phase 2) fills all 5 verify lanes. d_spec holds the
        // rejection-sampling verdict {n, stop_lane, exclude_token}.
        A((void**)&d_nuc, 5 * 4 * 4);
        A((void**)&d_spec, 3 * 4);
        A((void**)&d_draft_margin, 7 * 4); // maxd7: up to 7 draft margins
        A((void**)&d_am_blk1, 128 * 8);    // P14: fused draft argmax+margin scratch
        A((void**)&d_am_blk2, 128 * 4);
        A((void**)&h_next, N_EMBD * 4); A((void**)&e_hn, 2 * N_EMBD * 4);
        A((void**)&x_mtp, N_EMBD * 4); A((void**)&mtp_logits, VOCAB * 4);
        A(&mtp_k, kv_bytes(false));
        A(&mtp_v, kv_bytes(true));
        A((void**)&d_pos_m, 4); A((void**)&d_draft, 4);
        A((void**)&h_L[1], N_EMBD * 4); A((void**)&x1_L[1], N_EMBD * 4); A((void**)&y_L[1], N_EMBD * 4);
        A((void**)&qg_L[1], 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_L[1], N_KV * HEAD_DIM * 4); A((void**)&vbuf_L[1], N_KV * HEAD_DIM * 4);
        A((void**)&attnout_L[1], N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_L[1], GDN_CH * 4); A((void**)&convout_L[1], GDN_CH * 4);
        A((void**)&z_L[1], GDN_V * 4);
        A((void**)&alpha_L[1], GDN_HEADS * 4); A((void**)&betar_L[1], GDN_HEADS * 4);
        A((void**)&g_L[1], GDN_HEADS * 4); A((void**)&beta_L[1], GDN_HEADS * 4);
        A((void**)&o_L[1], GDN_V * 4); A((void**)&og_L[1], GDN_V * 4);
        A((void**)&ffn_g_L[1], N_FFN * 4); A((void**)&ffn_u_L[1], N_FFN * 4);
        A((void**)&logits2, W_MAX * (size_t)VOCAB * 4); // width-12: 12 verify lanes
        // P2: k_vgemm's deterministic cross-CTA K-split partials. Sized by WALKING
        // THE WEIGHT LIST (never hardcoded -- the z policy decides how big this is,
        // and a policy change would silently overrun a fixed buffer). Sized off the
        // Model, not the DeviceModel: upload_all() has not run yet at this point.
        // ~3.3 MB on this checkpoint (ffn_gate at z=3 x 16 lanes x 17408 rows).
        {
            size_t wsb = q27k::vgemm_ws_bytes_model(dm.model(), gemm_min_rows);
            if (wsb) A((void**)&d_vgemm_ws, wsb);
        }
        mask_words = (VOCAB + 31) / 32;
        A((void**)&d_mask_pool, (size_t)MASK_POOL_CAP * mask_words * 4);
        A((void**)&d_mask_ids, W_MAX * 4);
        if (const char* ce = getenv("Q27_CKPT_INTERVAL")) ckpt_interval = atoi(ce);
        if (const char* cs = getenv("Q27_CKPT_SLOTS")) ckpt_slots = std::max(1, atoi(cs));
        A((void**)&d_accept_cap, 4);
        CUDA_CHECK(cudaMemset(d_mask_ids, 0xFF, W_MAX * 4)); // all -1 = unconstrained
        CUDA_CHECK(cudaMemset(d_accept_cap, 0, 4));
        A((void**)&y2big, 2 * (size_t)N_FFN * 4);
        xq2[0] = q27k::xquant_alloc(N_FFN);
        xq2[1] = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_L[0], 4); A((void**)&d_pos_L[1], 4);
        A((void**)&d_v_L[0], 4); A((void**)&d_v_L[1], 4);
        A((void**)&h_L[2], N_EMBD * 4); A((void**)&x1_L[2], N_EMBD * 4); A((void**)&y_L[2], N_EMBD * 4);
        A((void**)&qg_L[2], 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_L[2], N_KV * HEAD_DIM * 4); A((void**)&vbuf_L[2], N_KV * HEAD_DIM * 4);
        A((void**)&attnout_L[2], N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_L[2], GDN_CH * 4); A((void**)&convout_L[2], GDN_CH * 4);
        A((void**)&z_L[2], GDN_V * 4);
        A((void**)&alpha_L[2], GDN_HEADS * 4); A((void**)&betar_L[2], GDN_HEADS * 4);
        A((void**)&g_L[2], GDN_HEADS * 4); A((void**)&beta_L[2], GDN_HEADS * 4);
        A((void**)&o_L[2], GDN_V * 4); A((void**)&og_L[2], GDN_V * 4);
        A((void**)&ffn_g_L[2], N_FFN * 4); A((void**)&ffn_u_L[2], N_FFN * 4);
        A((void**)&h_next2, N_EMBD * 4);
        xq_L[2] = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_L[2], 4); A((void**)&d_pos_m2, 4); A((void**)&d_draft2, 4);
        A((void**)&d_v_L[2], 4);
        A((void**)&h_L[3], N_EMBD * 4); A((void**)&x1_L[3], N_EMBD * 4); A((void**)&y_L[3], N_EMBD * 4);
        A((void**)&qg_L[3], 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_L[3], N_KV * HEAD_DIM * 4); A((void**)&vbuf_L[3], N_KV * HEAD_DIM * 4);
        A((void**)&attnout_L[3], N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_L[3], GDN_CH * 4); A((void**)&convout_L[3], GDN_CH * 4);
        A((void**)&z_L[3], GDN_V * 4);
        A((void**)&alpha_L[3], GDN_HEADS * 4); A((void**)&betar_L[3], GDN_HEADS * 4);
        A((void**)&g_L[3], GDN_HEADS * 4); A((void**)&beta_L[3], GDN_HEADS * 4);
        A((void**)&o_L[3], GDN_V * 4); A((void**)&og_L[3], GDN_V * 4);
        A((void**)&ffn_g_L[3], N_FFN * 4); A((void**)&ffn_u_L[3], N_FFN * 4);
        A((void**)&h_next3, N_EMBD * 4);
        xq_L[3] = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_L[3], 4); A((void**)&d_pos_m3, 4); A((void**)&d_draft3, 4);
        A((void**)&d_v_L[3], 4);
        A((void**)&h_L[4], N_EMBD * 4); A((void**)&x1_L[4], N_EMBD * 4); A((void**)&y_L[4], N_EMBD * 4);
        A((void**)&qg_L[4], 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_L[4], N_KV * HEAD_DIM * 4); A((void**)&vbuf_L[4], N_KV * HEAD_DIM * 4);
        A((void**)&attnout_L[4], N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_L[4], GDN_CH * 4); A((void**)&convout_L[4], GDN_CH * 4);
        A((void**)&z_L[4], GDN_V * 4);
        A((void**)&alpha_L[4], GDN_HEADS * 4); A((void**)&betar_L[4], GDN_HEADS * 4);
        A((void**)&g_L[4], GDN_HEADS * 4); A((void**)&beta_L[4], GDN_HEADS * 4);
        A((void**)&o_L[4], GDN_V * 4); A((void**)&og_L[4], GDN_V * 4);
        A((void**)&ffn_g_L[4], N_FFN * 4); A((void**)&ffn_u_L[4], N_FFN * 4);
        A((void**)&h_next4, N_EMBD * 4);
        xq_L[4] = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_L[4], 4); A((void**)&d_pos_m4, 4); A((void**)&d_draft4, 4);
        A((void**)&d_v_L[4], 4);
        // depth-5 lane (f), P12b
        A((void**)&h_L[5], N_EMBD * 4); A((void**)&x1_L[5], N_EMBD * 4); A((void**)&y_L[5], N_EMBD * 4);
        A((void**)&qg_L[5], 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_L[5], N_KV * HEAD_DIM * 4); A((void**)&vbuf_L[5], N_KV * HEAD_DIM * 4);
        A((void**)&attnout_L[5], N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_L[5], GDN_CH * 4); A((void**)&convout_L[5], GDN_CH * 4);
        A((void**)&z_L[5], GDN_V * 4);
        A((void**)&alpha_L[5], GDN_HEADS * 4); A((void**)&betar_L[5], GDN_HEADS * 4);
        A((void**)&g_L[5], GDN_HEADS * 4); A((void**)&beta_L[5], GDN_HEADS * 4);
        A((void**)&o_L[5], GDN_V * 4); A((void**)&og_L[5], GDN_V * 4);
        A((void**)&ffn_g_L[5], N_FFN * 4); A((void**)&ffn_u_L[5], N_FFN * 4);
        A((void**)&h_next5, N_EMBD * 4);
        xq_L[5] = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_L[5], 4); A((void**)&d_pos_m5, 4); A((void**)&d_draft5, 4);
        A((void**)&d_v_L[5], 4);
        // depth-6 lane (g), maxd6
        A((void**)&h_L[6], N_EMBD * 4); A((void**)&x1_L[6], N_EMBD * 4); A((void**)&y_L[6], N_EMBD * 4);
        A((void**)&qg_L[6], 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_L[6], N_KV * HEAD_DIM * 4); A((void**)&vbuf_L[6], N_KV * HEAD_DIM * 4);
        A((void**)&attnout_L[6], N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_L[6], GDN_CH * 4); A((void**)&convout_L[6], GDN_CH * 4);
        A((void**)&z_L[6], GDN_V * 4);
        A((void**)&alpha_L[6], GDN_HEADS * 4); A((void**)&betar_L[6], GDN_HEADS * 4);
        A((void**)&g_L[6], GDN_HEADS * 4); A((void**)&beta_L[6], GDN_HEADS * 4);
        A((void**)&o_L[6], GDN_V * 4); A((void**)&og_L[6], GDN_V * 4);
        A((void**)&ffn_g_L[6], N_FFN * 4); A((void**)&ffn_u_L[6], N_FFN * 4);
        A((void**)&h_next6, N_EMBD * 4);
        xq_L[6] = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_L[6], 4); A((void**)&d_pos_m6, 4); A((void**)&d_draft6, 4);
        A((void**)&d_v_L[6], 4);
        // depth-7 lane (h), maxd7
        A((void**)&h_L[7], N_EMBD * 4); A((void**)&x1_L[7], N_EMBD * 4); A((void**)&y_L[7], N_EMBD * 4);
        A((void**)&qg_L[7], 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_L[7], N_KV * HEAD_DIM * 4); A((void**)&vbuf_L[7], N_KV * HEAD_DIM * 4);
        A((void**)&attnout_L[7], N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_L[7], GDN_CH * 4); A((void**)&convout_L[7], GDN_CH * 4);
        A((void**)&z_L[7], GDN_V * 4);
        A((void**)&alpha_L[7], GDN_HEADS * 4); A((void**)&betar_L[7], GDN_HEADS * 4);
        A((void**)&g_L[7], GDN_HEADS * 4); A((void**)&beta_L[7], GDN_HEADS * 4);
        A((void**)&o_L[7], GDN_V * 4); A((void**)&og_L[7], GDN_V * 4);
        A((void**)&ffn_g_L[7], N_FFN * 4); A((void**)&ffn_u_L[7], N_FFN * 4);
        A((void**)&h_next7, N_EMBD * 4);
        xq_L[7] = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_L[7], 4); A((void**)&d_pos_m7, 4); A((void**)&d_draft7, 4);
        A((void**)&d_v_L[7], 4);
        // WIDE lanes 8..W_PLUMB-1: verify columns 9..W_PLUMB, no h_next/pos_m
        // (the MTP chain stays <= depth 7; these lanes are suffix-fed).
        // Allocated for every build regardless of W_MAX -- "plumbing is fixed"
        // means k_finish_round may dereference any slot < W_PLUMB, so a narrow
        // build still needs the pointers to be real and zeroed. The per-lane
        // alloc is uniform up here, so W16 loops it instead of unrolling four
        // more copy-pasted blocks (the 8->12 widening's by-hand list is exactly
        // where the k_quantize_x3 lane-aliasing bug hid).
        for (int L = 8; L < W_PLUMB; L++) {
            A((void**)&h_L[L], N_EMBD * 4); A((void**)&x1_L[L], N_EMBD * 4);
            A((void**)&y_L[L], N_EMBD * 4);
            A((void**)&qg_L[L], 2 * N_HEAD * HEAD_DIM * 4);
            A((void**)&kbuf_L[L], N_KV * HEAD_DIM * 4);
            A((void**)&vbuf_L[L], N_KV * HEAD_DIM * 4);
            A((void**)&attnout_L[L], N_HEAD * HEAD_DIM * 4);
            A((void**)&qkv_L[L], GDN_CH * 4); A((void**)&convout_L[L], GDN_CH * 4);
            A((void**)&z_L[L], GDN_V * 4);
            A((void**)&alpha_L[L], GDN_HEADS * 4); A((void**)&betar_L[L], GDN_HEADS * 4);
            A((void**)&g_L[L], GDN_HEADS * 4); A((void**)&beta_L[L], GDN_HEADS * 4);
            A((void**)&o_L[L], GDN_V * 4); A((void**)&og_L[L], GDN_V * 4);
            A((void**)&ffn_g_L[L], N_FFN * 4); A((void**)&ffn_u_L[L], N_FFN * 4);
            xq_L[L] = q27k::xquant_alloc(N_FFN);
            A((void**)&d_pos_L[L], 4); A((void**)&d_v_L[L], 4);
            A((void**)&d_draft_L[L - 1], 4); // lane L verifies draft L
            // nothing writes a wide slot until a suffix round stages proposals
            // into it, but finish reads every slot every round -> zero them.
            CUDA_CHECK(cudaMemset(d_draft_L[L - 1], 0, 4));
            CUDA_CHECK(cudaMemset(d_v_L[L], 0, 4));
            CUDA_CHECK(cudaMemset(d_pos_L[L], 0, 4));
        }
        // lane index 0 aliases the primary lane's buffers (audit refactor):
        // every W_PLUMB-wide call site indexes one array instead of N names.
        h_L[0] = h; x1_L[0] = x1; y_L[0] = y; qg_L[0] = qg; kbuf_L[0] = kbuf;
        vbuf_L[0] = vbuf; attnout_L[0] = attnout; qkv_L[0] = qkv;
        convout_L[0] = convout; z_L[0] = z; alpha_L[0] = alpha;
        betar_L[0] = betar; g_L[0] = g; beta_L[0] = beta; o_L[0] = o;
        og_L[0] = og; ffn_g_L[0] = ffn_g; ffn_u_L[0] = ffn_u;
        xq_L[0] = xq2[0]; xq_L[1] = xq2[1];
        // draft slots 1..7 are the MTP chain's, still written by name; alias
        // them into the array so every "list every draft" site is one loop.
        d_draft_L[0] = d_draft; d_draft_L[1] = d_draft2; d_draft_L[2] = d_draft3;
        d_draft_L[3] = d_draft4; d_draft_L[4] = d_draft5; d_draft_L[5] = d_draft6;
        d_draft_L[6] = d_draft7;
        A((void**)&d_P, 4);
        A((void**)&d_outcome, OUTCOME_INTS * 4); // {n, t1, dr1..dr(W_PLUMB-1), pending}
        CUDA_CHECK(cudaMemset(mtp_k, 0, kv_bytes(false)));
        CUDA_CHECK(cudaMemset(mtp_v, 0, kv_bytes(true)));
        CUDA_CHECK(cudaMemset(d_pos, 0, 4));
        CUDA_CHECK(cudaMemset(d_step, 0, 4));
        xq = q27k::xquant_alloc(N_FFN);
        // batched prefill buffers (~130MB) + attention scratch
        auto fal = [](size_t n) { float* p; CUDA_CHECK(cudaMalloc((void**)&p, n * 4)); return p; };
        hT = fal((size_t)PF_T * N_EMBD); x1T = fal((size_t)PF_T * N_EMBD);
        yT = fal((size_t)PF_T * N_EMBD); qkvT = fal((size_t)PF_T * GDN_CH);
        convT = fal((size_t)PF_T * GDN_CH); zT = fal((size_t)PF_T * GDN_V);
        oT = fal((size_t)PF_T * GDN_V); ogT = fal((size_t)PF_T * GDN_V);
        qgT = fal((size_t)PF_T * N_HEAD * 2 * HEAD_DIM);
        kT = fal((size_t)PF_T * N_KV * HEAD_DIM); vT = fal((size_t)PF_T * N_KV * HEAD_DIM);
        attnT = fal((size_t)PF_T * N_HEAD * HEAD_DIM);
        alphaT = fal((size_t)PF_T * GDN_HEADS); betarT = fal((size_t)PF_T * GDN_HEADS);
        gT = fal((size_t)PF_T * GDN_HEADS); betaT = fal((size_t)PF_T * GDN_HEADS);
        ffnGT = fal((size_t)PF_T * N_FFN); ffnUT = fal((size_t)PF_T * N_FFN);
        embT = fal((size_t)PF_T * N_EMBD); ehnT = fal((size_t)PF_T * 2 * N_EMBD);
        xmtpT = fal((size_t)PF_T * N_EMBD);
        pf_part = fal((size_t)N_HEAD * PF_T * q27k::PF_SPLIT_MAX * 258);
        xqT = q27k::xquant_alloc((size_t)PF_T * N_FFN, /*g64=*/true);
        q27k::wy_scratch_reserve(&wy_scratch, PF_T); // fixed cap: no mid-serving regrow
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                CUDA_CHECK(cudaMalloc((void**)&S_snap[il],
                                      (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4));
                CUDA_CHECK(cudaMalloc((void**)&ring_snap[il], 3 * GDN_CH * 4));
            }

        int cache_slot = 0;
        for (int il = 0; il < N_LAYER; il++) {
            if (attn_layer[il]) {
                void *k, *v;
                A(&k, kv_bytes(false));
                A(&v, kv_bytes(true));
                kcache.push_back(k); vcache.push_back(v);
                attn_cache_idx.push_back(cache_slot++);
                conv_ring[il] = nullptr; S[il] = nullptr;
            } else {
                A((void**)&conv_ring[il], 3 * GDN_CH * 4);
                CUDA_CHECK(cudaMemset(conv_ring[il], 0, 3 * GDN_CH * 4));
                A((void**)&S[il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4);
                CUDA_CHECK(cudaMemset(S[il], 0, (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4));
                // spare sets 1..7 are allocated at EVERY gate_maxd (review
                // 2026-07-09, accepted tradeoff ~157MB each): the perm
                // rotation is uniformly mod-12 ((role+perm)%12), so all 12
                // sets enter rotation even at shallow ceilings. Roles 8..11
                // (indices 7..10) exist only when W_MAX admits them (Q27_W_MAX
                // knob); skipped sets stay nullptr and are never addressed
                // (SBuf/RBuf index (role+perm)%W_MAX < W_MAX). This is the
                // narrow-build memory win: each skipped role = ~157MB/engine.
                for (int r = 0; r < W_PLUMB - 1; r++) {
                    if (r >= 7 && W_MAX <= r + 1) {
                        S_sp[r][il] = ring_sp[r][il] = nullptr;
                        continue;
                    }
                    A((void**)&S_sp[r][il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4);
                    A((void**)&ring_sp[r][il], 3 * GDN_CH * 4);
                }
                // (W_MAX + 1) S-buffers (main + W_MAX-1 spares + snap) + rings
                // per GDN layer -- server slot admission sizes its floor from
                // this (review 2026-07-09: the old hardcoded "5 sets" predated
                // maxd6/7; the (W_MAX+1) form tracks the Q27_W_MAX knob).
                gdn_state_bytes +=
                    (size_t)(W_MAX + 1) * ((size_t)GDN_HEADS * GDN_DIM * GDN_DIM + 3 * GDN_CH) * 4;
                attn_cache_idx.push_back(-1);
            }
        }
        // P3 T2: upload the device role-pointer tables (ring half + S half,
        // one flat alloc) and the perm scalar, ONCE, now that every role
        // buffer above has its final address. Inert unless the conv/delta
        // table twins run (gdn_mix use_tables; see the member block).
        {
            const size_t half = (size_t)N_LAYER * W_MAX;
            std::vector<float*> tab(2 * half, nullptr);
            for (int il = 0; il < N_LAYER; il++) {
                if (attn_layer[il]) continue;
                for (int ph = 0; ph < W_MAX; ph++) {
                    tab[(size_t)il * W_MAX + ph] = ph == 0 ? conv_ring[il] : ring_sp[ph - 1][il];
                    tab[half + (size_t)il * W_MAX + ph] = ph == 0 ? S[il] : S_sp[ph - 1][il];
                }
            }
            CUDA_CHECK(cudaMalloc((void**)&d_gdn_tab, 2 * half * sizeof(float*)));
            CUDA_CHECK(cudaMemcpy(d_gdn_tab, tab.data(), 2 * half * sizeof(float*),
                                  cudaMemcpyHostToDevice));
            d_gdn_S_tab = d_gdn_tab + half;
            CUDA_CHECK(cudaMalloc((void**)&d_perm_scalar, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_perm_scalar, 0, sizeof(int)));
            CUDA_CHECK(cudaMallocHost((void**)&h_perm_pin, sizeof(int)));
            *h_perm_pin = 0;
        }
        if (own_weights) {
            fprintf(stderr, "uploading weights...\n");
            dm.upload_all();
            dm.checksum_baseline();
            fprintf(stderr, "resident: %.2f GB (checksummed)\n", dm.bytes_resident() / 1e9);
        }
    }

  public:
    const DevTensor& T(int il, const char* leaf) {
        char buf[96];
        snprintf(buf, sizeof buf, "blk.%d.%s", il, leaf);
        return dm.get(buf);
    }
    const DevTensor& T2(int il, const char* leaf) { return T(il, leaf); }

    void qx(const float* x, int cols) { qx(x, cols, stm); }
    // P2c: explicit-stream twin -- attn_block's stream form (mtp_attn runs
    // the MTP attention on the conductor stream in the fused draft step)
    // quantizes member state on the caller's stream; solo passes stm.
    void qx(const float* x, int cols, cudaStream_t st) { q27k::quantize_x(x, cols, xq, st); }

    void mm(const DevTensor& w, const float* x, float* out) { mm(w, x, out, stm); }
    // P2c: explicit-stream twin, same contract as qx above.
    void mm(const DevTensor& w, const float* x, float* out, cudaStream_t st) {
        switch (w.dtype) {
            case DType::Q4_G64:
                q27k::gemv_q4((const uint8_t*)w.data, (const __half*)w.scales, xq, out, w.rows,
                              w.cols, st);
                break;
            case DType::Q8_G128:
                q27k::gemv_q8((const int8_t*)w.data, (const __half*)w.scales, xq, out, w.rows,
                              w.cols, st);
                break;
            case DType::F16:
                q27k::gemv_f16((const __half*)w.data, x, out, w.rows, w.cols, st);
                break;
            default:
                fprintf(stderr, "mm: unsupported dtype\n");
                exit(1);
        }
    }

    void gdn_block(int il, const float* xin, float* yout) {
        qx(xin, N_EMBD);
        mm(T(il, "attn_qkv.weight"), xin, qkv);
        mm(T(il, "attn_gate.weight"), xin, z);
        mm(T(il, "ssm_alpha.weight"), xin, alpha);
        mm(T(il, "ssm_beta.weight"), xin, betar);
        q27k::gdn_gates(alpha, betar, (const float*)T(il, "ssm_a").data,
                        (const float*)T(il, "ssm_dt.bias").data, g, beta, GDN_HEADS, stm);
        q27k::conv_step(conv_ring[il], conv_ring[il], qkv,
                        (const float*)T(il, "ssm_conv1d.weight").data, convout, GDN_CH, stm);
        q27k::l2norm_heads(convout, 16, GDN_DIM, EPS, stm);
        q27k::l2norm_heads(convout + 2048, 16, GDN_DIM, EPS, stm);
        q27k::delta_step(S[il], S[il], convout, g, beta, o, stm);
        q27k::gated_norm_gdn(o, (const float*)T(il, "ssm_norm.weight").data, z, og, GDN_HEADS,
                             GDN_DIM, EPS, stm);
        qx(og, GDN_V);
        mm(T(il, "ssm_out.weight"), og, yout);
    }

    void attn_block(int il, const float* xin, float* yout, void* kc = nullptr,
                    void* vc = nullptr, const int* pos_src = nullptr) {
        attn_block(il, xin, yout, kc, vc, pos_src, stm);
    }
    // P2c: explicit-stream form, same contract as gdn_mix/attn_mix's stream
    // param -- mtp_attn runs the MTP attention on the conductor stream in the
    // fused draft step; every other caller comes through the defaulted form
    // above with member stm (same value, same launch sequence).
    void attn_block(int il, const float* xin, float* yout, void* kc, void* vc,
                    const int* pos_src, cudaStream_t st) {
        if (!kc) {
            int ci = attn_cache_idx[il];
            kc = kcache[ci];
            vc = vcache[ci];
        }
        if (!pos_src) pos_src = d_pos;
        qx(xin, N_EMBD, st);
        mm(T(il, "attn_q.weight"), xin, qg, st);
        q27k::rmsnorm_heads(qg, (const float*)T(il, "attn_q_norm.weight").data, qg, N_HEAD,
                            HEAD_DIM, 2 * HEAD_DIM, EPS, st);
        mm(T(il, "attn_k.weight"), xin, kbuf, st);
        q27k::rmsnorm_heads(kbuf, (const float*)T(il, "attn_k_norm.weight").data, kbuf, N_KV,
                            HEAD_DIM, HEAD_DIM, EPS, st);
        mm(T(il, "attn_v.weight"), xin, vbuf, st);
        q27k::rope_neox_partial(qg, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, pos_src, FREQ_BASE, st);
        q27k::rope_neox_partial(kbuf, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, pos_src, FREQ_BASE, st);
        // turbo3: Q forward-WHT after rope (K's rotation is folded into the
        // store; <WHT q, WHT K> == <q,K>); turbo3v keeps K fp16 => Q raw.
        // Host branches on kv_kind only -- fixed at init, graph-capture-safe.
        if (kv_kind == KV_T3) {
            q27k::P3 qw{{qg}};
            q27k::wht3(qw, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, false, st, 1);
        }
        if (kv_kind >= KV_T3) {
            q27k::CP3 kw{{kbuf}};
            q27k::CP3 vw3{{vbuf}};
            q27k::IP3 pw{{pos_src}};
            q27k::kv_store_t3(kw, vw3, kc, vc, pw, N_KV, HEAD_DIM, st, 1,
                              /*k_plain=*/kv_kind == KV_T3V);
        } else {
            q27k::kv_store(kbuf, vbuf, kc, vc, pos_src, N_KV * HEAD_DIM, st, kv_fp8);
        }
        q27k::attn_decode(qg, 2 * HEAD_DIM, kc, vc, attnout, scratch, pos_src,
                          max_ctx, N_HEAD, N_KV, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM), st,
                          kv_kind);
        // turbo3 V accumulates in the rotated basis: one inverse-WHT on the
        // pooled output BEFORE the sigmoid gate (elementwise gate does not
        // commute with the rotation).
        if (kv_kind >= KV_T3) {
            q27k::P3 ow{{attnout}};
            q27k::wht3(ow, N_HEAD, HEAD_DIM, HEAD_DIM, true, st, 1);
        }
        q27k::sigmoid_gate_mul(attnout, qg, N_HEAD, HEAD_DIM, st);
        qx(attnout, N_HEAD * HEAD_DIM, st);
        mm(T(il, "attn_output.weight"), attnout, yout, st);
    }

    void ffn(int il, const float* xin, float* yout) {
        qx(xin, N_EMBD);
        mm(T(il, "ffn_gate.weight"), xin, ffn_g);
        mm(T(il, "ffn_up.weight"), xin, ffn_u);
        q27k::silu_mul(ffn_g, ffn_u, ffn_g, N_FFN, stm);
        qx(ffn_g, N_FFN);
        mm(T(il, "ffn_down.weight"), ffn_g, yout);
    }

    // enqueue one full token onto stm (no syncs, no allocations: graph-safe)
    // taps: DFlash Phase-0 rig plumbing (docs/dflash-block-verify-design.md
    // P0a) -- when non-null, the residual stream h is copied after each
    // DFLASH_TAP layer. Host-side branch only: build_graph captures with
    // taps == nullptr, so the graphed paths are byte-identical.
    static constexpr int DFLASH_TAPS[5] = {1, 16, 31, 46, 61}; // z-lab target_layer_ids; P0a measured the +-1 convention equal (AL 2.10 vs 2.18)
    void token_launches(float* taps = nullptr) {
        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed_row_q8((const int8_t*)emb.data, (const __half*)emb.scales, d_token, N_EMBD, h,
                           stm);
        int tap_k = 0;
        for (int il = 0; il < N_LAYER; il++) {
            q27k::rmsnorm(h, (const float*)T(il, "attn_norm.weight").data, x1, N_EMBD, EPS, stm);
            if (attn_layer[il]) attn_block(il, x1, y);
            else gdn_block(il, x1, y);
            q27k::add_inplace(h, y, N_EMBD, stm);
            q27k::rmsnorm(h, (const float*)T(il, "post_attention_norm.weight").data, x1, N_EMBD,
                          EPS, stm);
            ffn(il, x1, y);
            q27k::add_inplace(h, y, N_EMBD, stm);
            if (taps && tap_k < 5 && il == DFLASH_TAPS[tap_k]) {
                CUDA_CHECK(cudaMemcpyAsync(taps + (size_t)tap_k * N_EMBD, h, N_EMBD * 4,
                                           cudaMemcpyDeviceToDevice, stm));
                tap_k++;
            }
        }
        q27k::rmsnorm(h, (const float*)dm.get("output_norm.weight").data, x1, N_EMBD, EPS, stm);
        qx(x1, N_EMBD);
        mm(dm.get("output.weight"), x1, logits);
        q27k::argmax(logits, VOCAB, d_token, d_amax, stm); // d_token becomes NEXT token
        q27k::advance(d_pos, d_step, d_gen, d_token, stm); // record + pos++
    }

    // Sampled plain step (temp>0): identical forward to token_launches, but the
    // tail samples from the served distribution instead of argmax. Reads params
    // from d_samp and the key position from *d_pos (both device -> one captured
    // graph serves every request). draw_kind=1 (graph draws); the eager first
    // token uses kind 0 so the two never share a Philox counter. Greedy path is
    // untouched -- this is a SEPARATE graph, never on the canonical-gated path.
    void token_launches_sampled() {
        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed_row_q8((const int8_t*)emb.data, (const __half*)emb.scales, d_token, N_EMBD, h,
                           stm);
        for (int il = 0; il < N_LAYER; il++) {
            q27k::rmsnorm(h, (const float*)T(il, "attn_norm.weight").data, x1, N_EMBD, EPS, stm);
            if (attn_layer[il]) attn_block(il, x1, y);
            else gdn_block(il, x1, y);
            q27k::add_inplace(h, y, N_EMBD, stm);
            q27k::rmsnorm(h, (const float*)T(il, "post_attention_norm.weight").data, x1, N_EMBD,
                          EPS, stm);
            ffn(il, x1, y);
            q27k::add_inplace(h, y, N_EMBD, stm);
        }
        q27k::rmsnorm(h, (const float*)dm.get("output_norm.weight").data, x1, N_EMBD, EPS, stm);
        qx(x1, N_EMBD);
        mm(dm.get("output.weight"), x1, logits);
        q27k::sample_g(logits, VOCAB, d_samp, d_nuc, d_pos, 1, d_token, d_amax, stm);
        q27k::advance(d_pos, d_step, d_gen, d_token, stm);
    }

    // MTP draft head (blk.64, SPEC.md): draft the token AFTER d_token, given
    // h_next = post-output_norm hidden of the current position (*d_pos_m).
    // margin_dst != null (draft path only): fuse the trailing argmax with the P12
    // top1-top2 margin into ONE full-vocab pass -- draft_dst gets the SAME token as
    // the plain argmax (bit-identical tie semantics), margin_dst gets margin()'s
    // value. margin_dst == null keeps the plain argmax for every other caller.
    // P2c (docs/plans/2026-07-16-batch-p2c-draft-fusion.md): the P0 pattern
    // applied to the MTP step -- everything one draft step reads or writes
    // that is PER-LANE sits behind MtpLaneView, so the fused cross-engine
    // step can point union slot k at any engine's chain state. Solo:
    // mtp_solo_view() fills lane 0 with this engine's members
    // (pointer-identical) and pads slots >= vw with lane 0 (never read --
    // the gemv_*_n `i < nb ? i : 0` convention). Deliberately NOT here: the
    // MTP attention state (mtp_k/mtp_v, qg/kbuf/vbuf/attnout, scratch) --
    // step 4 stays member-based (mtp_attn), exactly like the verify mixers.
    struct MtpLaneView {
        // per-lane chain state (slot L = one engine's current draft step):
        // e_hn = embed||hidden concat; x_mtp = residual; x1/y = layer io;
        // lg = mtp_logits; ffn_g/ffn_u = MLP gate/up scratch
        std::array<float*, W_PLUMB> e_hn, x_mtp, x1, y, lg, ffn_g, ffn_u;
        std::array<const float*, W_PLUMB> h_src;  // h_next chain / hs[k]
        std::array<const int*, W_PLUMB> tok, pos; // tok_src chain, d_pos_m..m7
        std::array<int*, W_PLUMB> draft_dst;      // d_draft_L slot
        std::array<float*, W_PLUMB> margin_dst;   // d_draft_margin + k (or null)
        std::array<q27k::XQuant, W_PLUMB> xq;     // per-engine quant slot
        std::array<unsigned long long*, W_PLUMB> am_blk1; // argmax_margin scratch
        std::array<float*, W_PLUMB> am_blk2;
        std::array<unsigned long long*, W_PLUMB> amax; // plain-argmax scratch
        int vw;           // live lanes (union k; solo 1)
        cudaStream_t stm; // stream the step runs on
        // A1/Task-9 policy, MTP flavor: solo drafts run the dp4a GEMV family,
        // so the union head mm must NEVER take vgemm. 99 keeps every legal
        // union width (k <= MAX_K/2 = 8) below the branch; mtp_mm has no
        // vgemm path at all and asserts the policy (defensive).
        int gemm_min;
    };
    MtpLaneView mtp_solo_view(const float* h_src, const int* tok_src, int* draft_dst,
                              const int* pos_src, float* margin_dst) {
        MtpLaneView v{};
        for (int t = 0; t < W_PLUMB; t++) {
            v.e_hn[t] = e_hn; v.x_mtp[t] = x_mtp; v.x1[t] = x1; v.y[t] = y;
            v.lg[t] = mtp_logits; v.ffn_g[t] = ffn_g; v.ffn_u[t] = ffn_u;
            v.h_src[t] = h_src; v.tok[t] = tok_src; v.pos[t] = pos_src;
            v.draft_dst[t] = draft_dst; v.margin_dst[t] = margin_dst;
            v.xq[t] = xq; v.am_blk1[t] = d_am_blk1; v.am_blk2[t] = d_am_blk2;
            v.amax[t] = d_amax;
        }
        v.vw = 1;
        v.stm = stm;
        v.gemm_min = 99; // defensive -- see MtpLaneView
        return v;
    }
    // qx5/mm5 twins over the MTP view (A4 thin surface: MtpLaneView shares no
    // fields with LaneView, so no adapter/friend layer -- two small helpers).
    void mtp_qx(const MtpLaneView& v, const std::array<float*, W_PLUMB>& x, int cols) {
        q27k::XQ3 q{};
        q27k::CP3 xs{};
        for (int i = 0; i < W_PLUMB; i++) {
            q.q[i] = v.xq[i];
            xs.p[i] = x[i];
        }
        q27k::quantize3(xs, cols, q, v.stm, v.vw);
    }
    void mtp_mm(const MtpLaneView& v, const DevTensor& w, const std::array<float*, W_PLUMB>& ys_a) {
        // the MTP union must stay on the gemv family solo drafts use (view
        // comment): no vgemm branch BY CONSTRUCTION, assert the policy.
        assert(v.vw < v.gemm_min && "mtp_mm: union width crossed the vgemm threshold");
        q27k::XQuant qs[W_PLUMB];
        float* ys[W_PLUMB];
        for (int i = 0; i < W_PLUMB; i++) {
            qs[i] = v.xq[i];
            ys[i] = ys_a[i];
        }
        switch (w.dtype) {
            case DType::Q4_G64:
                q27k::gemv_q4_n((const uint8_t*)w.data, (const __half*)w.scales, qs, v.vw, ys,
                                w.rows, w.cols, v.stm);
                break;
            case DType::Q8_G128:
                q27k::gemv_q8_n((const int8_t*)w.data, (const __half*)w.scales, qs, v.vw, ys,
                                w.rows, w.cols, v.stm);
                break;
            default:
                // mirror mm()/mtp_mm1: fail loud on a dtype with no multi-lane
                // twin (the old bare else silently fed any non-Q4 weight to
                // the Q8 kernel; every MTP weight is Q4/Q8, so this is a
                // guard, not a live-path change).
                fprintf(stderr, "mtp_mm: unsupported dtype\n");
                exit(1);
        }
    }
    // single-lane mm twin reading the view's lane 0 instead of members. The
    // P2c DECIDE gate: gemv_*_n has NO nbatch=1 kernel (its switch starts at
    // 2, and k_gemv_q4 != k_gemv_q4_n<1>), so the solo composition KEEPS the
    // single-lane kernels -- zero numeric risk, and the captured draft
    // graphs record the pre-refactor launch sequence verbatim (B-A8).
    // Multi-lane enters only via the fused union view (vw >= 2, Task 2).
    void mtp_mm1(const MtpLaneView& v, const DevTensor& w, float* out) {
        switch (w.dtype) {
            case DType::Q4_G64:
                q27k::gemv_q4((const uint8_t*)w.data, (const __half*)w.scales, v.xq[0], out,
                              w.rows, w.cols, v.stm);
                break;
            case DType::Q8_G128:
                q27k::gemv_q8((const int8_t*)w.data, (const __half*)w.scales, v.xq[0], out,
                              w.rows, w.cols, v.stm);
                break;
            default:
                fprintf(stderr, "mtp_mm1: unsupported dtype\n");
                exit(1);
        }
    }
    // P2c seams, mirroring the verify split: pre/post are the WEIGHT-SWEEP
    // halves (anatomy steps 1-3 / 5-6), reading everything through the view;
    // attn is step 4 (member-based: own MTP KV + attention scratch, explicit
    // stream like gdn_mix); tail is step 7 (per-lane loop, own buffers via
    // the view). vw == 1 keeps the single-lane kernels (the mtp_mm1 DECIDE
    // branch); vw >= 2 exists only for the fused cross-engine step -- the
    // solo path NEVER takes it. Both are host branches baked per graph
    // capture, same class as mm5's width branch.
    void mtp_pre(const MtpLaneView& v) {
        const int il = 64;
        const DevTensor& emb = dm.get("token_embd.weight");
        const float* en = (const float*)T(il, "nextn.enorm.weight").data;
        const float* hn = (const float*)T(il, "nextn.hnorm.weight").data;
        if (v.vw == 1) {
            q27k::embed_row_q8((const int8_t*)emb.data, (const __half*)emb.scales, v.tok[0],
                               N_EMBD, v.e_hn[0], v.stm);
            q27k::rmsnorm(v.e_hn[0], en, v.e_hn[0], N_EMBD, EPS, v.stm);
            q27k::rmsnorm(v.h_src[0], hn, v.e_hn[0] + N_EMBD, N_EMBD, EPS, v.stm);
            q27k::quantize_x(v.e_hn[0], 2 * N_EMBD, v.xq[0], v.stm);
            mtp_mm1(v, T(il, "nextn.eh_proj.weight"), v.x_mtp[0]);
            return;
        }
        q27k::IP3 tk LANESV(v, tok);
        q27k::embed3((const int8_t*)emb.data, (const __half*)emb.scales, tk, N_EMBD,
                     LANESV(v, e_hn), v.stm, v.vw);
        q27k::CP3 Ec LANESV(v, e_hn);
        q27k::P3 Em LANESV(v, e_hn);
        q27k::rmsnorm3(Ec, en, Em, N_EMBD, EPS, v.stm, v.vw);
        q27k::P3 E2m{}; // per-lane e_hn + N_EMBD (the hidden half of the concat)
        for (int i = 0; i < W_PLUMB; i++) E2m.p[i] = v.e_hn[i] + N_EMBD;
        q27k::rmsnorm3(LANESV(v, h_src), hn, E2m, N_EMBD, EPS, v.stm, v.vw);
        mtp_qx(v, v.e_hn, 2 * N_EMBD);
        mtp_mm(v, T(il, "nextn.eh_proj.weight"), v.x_mtp);
    }
    // step 4 -- member-based (own MTP KV + attention scratch), explicit
    // stream: the fused step runs each engine's MTP attention on the
    // conductor stream; solo passes member stm (same value, same sequence).
    void mtp_attn(const int* pos_src, cudaStream_t st) {
        const int il = 64;
        q27k::rmsnorm(x_mtp, (const float*)T(il, "attn_norm.weight").data, x1, N_EMBD, EPS, st);
        attn_block(il, x1, y, mtp_k, mtp_v, pos_src, st);
    }
    void mtp_post(const MtpLaneView& v) {
        const int il = 64;
        const float* pn = (const float*)T(il, "post_attention_norm.weight").data;
        const float* sn = (const float*)T(il, "nextn.shared_head_norm.weight").data;
        // drafts use the Q4 head copy when present (verify keeps the Q8 head,
        // so output remains exactly the faithful model's greedy text)
        const DevTensor* head = dm.model_has("output_q4.weight")
                                    ? &dm.get("output_q4.weight")
                                    : &dm.get("output.weight");
        if (v.vw == 1) {
            q27k::add_inplace(v.x_mtp[0], v.y[0], N_EMBD, v.stm);
            q27k::rmsnorm(v.x_mtp[0], pn, v.x1[0], N_EMBD, EPS, v.stm);
            // ffn(il, x1, y) unrolled onto the view's lane 0 + stream
            q27k::quantize_x(v.x1[0], N_EMBD, v.xq[0], v.stm);
            mtp_mm1(v, T(il, "ffn_gate.weight"), v.ffn_g[0]);
            mtp_mm1(v, T(il, "ffn_up.weight"), v.ffn_u[0]);
            q27k::silu_mul(v.ffn_g[0], v.ffn_u[0], v.ffn_g[0], N_FFN, v.stm);
            q27k::quantize_x(v.ffn_g[0], N_FFN, v.xq[0], v.stm);
            mtp_mm1(v, T(il, "ffn_down.weight"), v.y[0]);
            q27k::add_inplace(v.x_mtp[0], v.y[0], N_EMBD, v.stm);
            q27k::rmsnorm(v.x_mtp[0], sn, v.x1[0], N_EMBD, EPS, v.stm);
            q27k::quantize_x(v.x1[0], N_EMBD, v.xq[0], v.stm);
            mtp_mm1(v, *head, v.lg[0]);
            return;
        }
        q27k::P3 Xm LANESV(v, x_mtp);
        q27k::CP3 Xc LANESV(v, x_mtp);
        q27k::CP3 Yc LANESV(v, y);
        q27k::P3 X1m LANESV(v, x1);
        q27k::add3(Xm, Yc, N_EMBD, v.stm, v.vw);
        q27k::rmsnorm3(Xc, pn, X1m, N_EMBD, EPS, v.stm, v.vw);
        mtp_qx(v, v.x1, N_EMBD);
        mtp_mm(v, T(il, "ffn_gate.weight"), v.ffn_g);
        mtp_mm(v, T(il, "ffn_up.weight"), v.ffn_u);
        q27k::silu_mul3(LANESV(v, ffn_g), LANESV(v, ffn_u), N_FFN, v.stm, v.vw);
        mtp_qx(v, v.ffn_g, N_FFN);
        mtp_mm(v, T(il, "ffn_down.weight"), v.y);
        q27k::add3(Xm, Yc, N_EMBD, v.stm, v.vw);
        q27k::rmsnorm3(Xc, sn, X1m, N_EMBD, EPS, v.stm, v.vw);
        mtp_qx(v, v.x1, N_EMBD);
        mtp_mm(v, *head, v.lg);
    }
    void mtp_tail(const MtpLaneView& v) {
        for (int t = 0; t < v.vw; t++) {
            if (v.margin_dst[t])
                q27k::argmax_margin(v.lg[t], VOCAB, v.draft_dst[t], v.margin_dst[t],
                                    v.am_blk1[t], v.am_blk2[t], v.stm);
            else
                q27k::argmax(v.lg[t], VOCAB, v.draft_dst[t], v.amax[t], v.stm);
        }
    }
    // solo mtp_forward = composition over mtp_solo_view() -- byte-identical
    // by construction (the vw == 1 branches launch the pre-refactor kernels
    // with the pre-refactor arguments in the pre-refactor order).
    void mtp_forward(const float* h_src = nullptr, const int* tok_src = nullptr,
                     int* draft_dst = nullptr, const int* pos_src = nullptr,
                     float* margin_dst = nullptr) {
        if (!h_src) h_src = h_next;
        if (!tok_src) tok_src = d_token;
        if (!draft_dst) draft_dst = d_draft;
        if (!pos_src) pos_src = d_pos_m;
        MtpLaneView v = mtp_solo_view(h_src, tok_src, draft_dst, pos_src, margin_dst);
        mtp_pre(v);
        mtp_attn(pos_src, stm);
        mtp_post(v);
        mtp_tail(v);
    }
    // P2c Task 2 (fused draft steps): the per-step CHAIN-POINTER TABLE, the
    // single source of truth for both the solo capture path
    // (spec_draft_step_launches below is its composition) and the fused
    // cross-engine union builder (build_mtp_union_view in conductor.h reads
    // slot m from engine m's mtp_step_view(step)). Step 0's entries are the
    // old k==0 mtp_forward args (h_next, d_token, d_draft, d_pos_m,
    // margin+0); step k>0 chains MTP's own post-head-norm hidden through
    // hs[k] into draft k+1 at pos_m{k+1}. One table so the two paths can
    // never drift -- the 8->12 widening's by-name brace-list landmine.
    // The table itself lives in mtp_step_chain (P2 exit review dedup):
    // BOTH readers -- mtp_step_view and draft_step_prep's D2D target --
    // index the same brace lists, so a chain edit lands in one place.
    struct MtpStepChain {
        float* h;       // hs[step]: this step's h_src; step+1's D2D target
        const int* tok; // ts[step]: token the step embeds
        int* dst;       // ds[step]: draft slot the step's argmax writes
        const int* pos; // ps[step]: the step's MTP position
    };
    MtpStepChain mtp_step_chain(int step) {
        assert(step >= 0 && step < D_MAX_MTP);
        float* hs[D_MAX_MTP] = {h_next, h_next2, h_next3, h_next4, h_next5, h_next6, h_next7};
        const int* ts[D_MAX_MTP] = {d_token, d_draft, d_draft2, d_draft3,
                                    d_draft4, d_draft5, d_draft6};
        int* ds[D_MAX_MTP] = {d_draft, d_draft2, d_draft3, d_draft4,
                              d_draft5, d_draft6, d_draft7};
        const int* ps[D_MAX_MTP] = {d_pos_m, d_pos_m2, d_pos_m3, d_pos_m4,
                                    d_pos_m5, d_pos_m6, d_pos_m7};
        return {hs[step], ts[step], ds[step], ps[step]};
    }
    MtpLaneView mtp_step_view(int step) {
        const MtpStepChain c = mtp_step_chain(step);
        return mtp_solo_view(c.h, c.tok, c.dst, c.pos, d_draft_margin + step);
    }
    // The step's per-engine PREAMBLE, stream-parameterized for the fused
    // path (solo passes stm): step 0 = prep_round (round bookkeeping --
    // d_P/d_outcome reset + lane/MTP position staging); step k>0 = the
    // x1 -> hs[k] chain D2D (MTP's post-head-norm hidden becomes the next
    // step's h_src). Same relative order as solo: the preamble precedes the
    // step's first kernel.
    void draft_step_prep(int step, cudaStream_t st) {
        if (step == 0) {
            q27k::prep_round(d_P, d_token, lane_pos(), mtp_pos(), W_MAX, D_MAX_MTP, d_outcome,
                             st);
            return;
        }
        CUDA_CHECK(cudaMemcpyAsync(mtp_step_chain(step).h, x1, N_EMBD * 4,
                                   cudaMemcpyDeviceToDevice, st));
    }

    // vw = verify batch width (# lanes: pending + drafts), read at GRAPH-CAPTURE
    // time only. P12 gated depth captures one verify graph per width in 1..5;
    // the struct slots beyond vw are never read by the ntok=vw kernels. vw=5 is
    // the full depth-4 round (bit-identical to the pre-P12 verify).
    int vw = 5;
    // dmax = # MTP drafts the draft graph produces (4 default; 5 for the gated
    // depth-5 draft graph). Capture-time only, like vw.
    int dmax = 4;
    // Continuous-batching P0 (docs/plans/2026-07-14-continuous-batching.md):
    // everything the verify forward's WEIGHT-SWEEP half reads or writes that
    // is PER-LANE, gathered behind one view so the fused cross-engine round
    // (P1) can point union slot k at any engine's lane buffers. Solo path:
    // solo_view() returns this engine's own lanes -- pointer-identical to the
    // member arrays, so routing the existing round through it changes nothing
    // by construction. Deliberately NOT here: mixer state (RBuf/SBuf roles,
    // kcache/vcache, attention scratch, convout -- all touched only inside
    // the member-based mix sections) and the tails' d_mask_*/d_amax/
    // finish_round state. Mixers and tails stay per-engine (design 07-14).
    struct LaneView {
        // sweep activations, written IN PLACE in the owning engine's buffers
        // (its mix/tail then reads them back as members, untouched):
        // x1/y/h = layer io; qkv/z/alpha/betar/g/beta + o/og = GDN pre/post;
        // qg/kbuf/vbuf/attnout = attn pre/post; ffn_g/ffn_u = MLP gate/up;
        // lg[t] = lane t's logits (logits2 + t*VOCAB; t >= W_MAX aliases
        // lane 0, never read -- same rule as the head mm5 today).
        std::array<float*, W_PLUMB> x1, qkv, z, alpha, betar, g, beta, o, og,
            y, qg, kbuf, vbuf, attnout, ffn_g, ffn_u, h, lg;
        std::array<q27k::XQuant, W_PLUMB> xq; // quantize3 dst / mm5 act src
        q27k::IP3 vtok;               // embed3 sources (pending + drafts)
        q27k::WIP3 pos;               // per-lane position ptrs (rope3)
        std::array<int*, W_PLUMB> dv; // per-lane argmax outs (greedy tail)
        float* vgemm_ws;              // k_vgemm workspace (>= union width)
        int vw;                       // live width THIS round
        cudaStream_t stm;             // stream the round runs on
        // P1 Task 9 GEMM-family policy: mm5 keys vgemm-vs-GEMV on the VIEW's
        // width, so a fused union crossing the member threshold (9) would
        // silently compute logits on a different numeric family than each
        // lane's solo round did (Task 8 finding; vgemm==gemv was never
        // claimed bitwise). The view carries its own threshold: solo_view()
        // copies the member (zero change); build_union_view() sets it per
        // union class -- all-gated 99 (GEMV, bitwise vs solo), all-suffix 2
        // (vgemm, the family solo suffix rounds took), Q27_BATCH_GEMM=1
        // forces 2 (the tolerance-class perf leg).
        int gemm_min;
    };
    LaneView solo_view() {
        LaneView v{};
        v.x1 = x1_L; v.qkv = qkv_L; v.z = z_L; v.alpha = alpha_L;
        v.betar = betar_L; v.g = g_L; v.beta = beta_L; v.o = o_L;
        v.og = og_L; v.y = y_L; v.qg = qg_L; v.kbuf = kbuf_L;
        v.vbuf = vbuf_L; v.attnout = attnout_L; v.ffn_g = ffn_g_L;
        v.ffn_u = ffn_u_L; v.h = h_L;
        for (int t = 0; t < W_PLUMB; t++)
            v.lg[t] = logits2 + (size_t)(t < W_MAX ? t : 0) * VOCAB;
        v.xq = xq_L;
        v.dv = d_v_L;
        v.vtok = verify_tokens();
        v.pos = lane_pos();
        v.vgemm_ws = d_vgemm_ws;
        v.vw = vw;
        v.stm = stm;
        v.gemm_min = gemm_min;
        return v;
    }
    // W16: the flat 12-pointer overloads are gone. They existed so a call site
    // could name its lanes, but at W_PLUMB=16 mm5's flat form would take 17
    // params -- the exact signature wall that pushed prep/finish onto by-value
    // structs. Every caller already had its lanes in a W_PLUMB array (or can
    // build one, as the vocab head does), so the array form is the only form.
    // P0 batching: qx5/mm5 read per-lane state through the view (solo:
    // pointer-identical to the members), so the fused round can hand them a
    // union view without touching the weight-sweep code again.
    void qx5(const LaneView& v, const std::array<float*, W_PLUMB>& x, int cols) {
        q27k::XQ3 q{};
        q27k::CP3 xs{};
        for (int i = 0; i < W_PLUMB; i++) {
            q.q[i] = v.xq[i]; // solo: xq_L (xq_L[0]/[1] alias xq2[0]/[1])
            xs.p[i] = x[i];
        }
        q27k::quantize3(xs, cols, q, v.stm, v.vw);
    }
    void mm5(const LaneView& v, const DevTensor& w, const std::array<float*, W_PLUMB>& ys_a) {
        // P2: WIDE rounds take the flat-in-W MMA GEMM; the ladder keeps the GEMV.
        // Both `vw` and `gemm_min` are host ints read at CUDA-GRAPH CAPTURE, so the
        // branch is baked per graph -- no per-call work, no divergence at replay.
        // vw <= 8 (every gated, draft and sampled round) can never take this branch:
        // gemm_min is 9 and build_spec_graphs aborts if gate_maxd+1 ever reaches it.
        // That is what makes the canonical bitwise gate structural.
        // k_vgemm reuses the group-32 int8 activations quantize3 ALREADY writes
        // (xq_L[i].nat/.scale are dead stores on the dp4a GEMV path today), so this
        // adds no quantize pass, no buffer and no graph node on the activation side.
        // P1 Task 9: the threshold comes from the VIEW (solo: a copy of the
        // member, identical branch; fused: the union-class policy set by
        // build_union_view) so a union crossing the member's 9 cannot fork
        // the numeric family away from what each lane's solo round took.
        if (v.vw >= v.gemm_min && (int64_t)w.rows >= gemm_min_rows) {
            q27k::XLanes X{};
            q27k::YLanes Y{};
            for (int i = 0; i < W_PLUMB; i++) {
                X.nat[i] = v.xq[i].nat;
                X.xs[i] = v.xq[i].scale;
                Y.y[i] = ys_a[i];
            }
            // Honor the false: an ineligible shape MUST fall through to the GEMV
            // rather than silently produce nothing (the launch_fdmma contract).
            if (q27k::vgemm_verify(w, X, Y, v.vgemm_ws, v.vw, v.stm)) return;
        }
        q27k::XQuant qs[W_PLUMB];
        float* ys[W_PLUMB];
        for (int i = 0; i < W_PLUMB; i++) {
            qs[i] = v.xq[i];
            ys[i] = ys_a[i];
        }
        if (w.dtype == DType::Q4_G64)
            q27k::gemv_q4_n((const uint8_t*)w.data, (const __half*)w.scales, qs, v.vw, ys, w.rows,
                            w.cols, v.stm);
        else
            q27k::gemv_q8_n((const int8_t*)w.data, (const __half*)w.scales, qs, v.vw, ys, w.rows,
                            w.cols, v.stm);
    }

    // P0 batching split (design 2026-07-14, docs/plans/2026-07-14-continuous-
    // batching-design.md): each pair is pre(view) -> mix(member) -> post(view).
    // pre/post are the WEIGHT-SWEEP halves -- pure per-lane weight/elementwise
    // ops reading everything through the LaneView -- so the P1 fused round can
    // run them ONCE over a union view spanning engines. mix is everything
    // touching this engine's SEQUENCE STATE (GDN role buffers RBuf/SBuf +
    // convout, KV cache, attention scratch, the kv_kind-branched turbo3
    // rotates) and stays member-based: member vw/stm, which in solo equal
    // v.vw/v.stm, so the composed pair emits the bit-identical launch
    // sequence -- graph capture records the same nodes in the same order
    // (addendum A8). The fused driver never calls the composed pairs, only
    // pre/mix/post individually (P1 Task 8: mix takes an explicit stream --
    // the conductor's -- while width stays member vw, the granted width).
    void gdn_pre(int il, const LaneView& v) {
        qx5(v, v.x1, N_EMBD);
        mm5(v, T(il, "attn_qkv.weight"), v.qkv);
        mm5(v, T(il, "attn_gate.weight"), v.z);
        q27k::gemv_f16_3((const __half*)T(il, "ssm_alpha.weight").data,
                         LANESV(v, x1),
                         LANESV(v, alpha), GDN_HEADS,
                         N_EMBD, v.stm, v.vw);
        q27k::gemv_f16_3((const __half*)T(il, "ssm_beta.weight").data,
                         LANESV(v, x1),
                         LANESV(v, betar), GDN_HEADS,
                         N_EMBD, v.stm, v.vw);
        const float* sa = (const float*)T(il, "ssm_a").data;
        const float* sdt = (const float*)T(il, "ssm_dt.bias").data;
        q27k::gdn_gates3(LANESV(v, alpha),
                         LANESV(v, betar), sa, sdt,
                         LANESV(v, g),
                         LANESV(v, beta), GDN_HEADS, v.stm, v.vw);
    }
    // -- split point: sequence state (RBuf/SBuf recurrent roles) begins here;
    //    weight sweep above reads only the view (design 2026-07-14) --
    // P1 batching: mix takes an explicit STREAM (the fused round runs every
    // engine's mix on the conductor stream; solo passes member stm -- same
    // value, same launch sequence). Width stays MEMBER vw: each engine's mix
    // walks its OWN granted lanes 0..vw-1, never the union width (the
    // conductor sets vw per round via set_round_width before the fused round).
    // P3 T2: use_tables=false (every existing call site) keeps the shipped
    // conv_step/delta_step launches with host-resolved RBuf/SBuf pointers --
    // the solo path is untouched by construction. use_tables=true swaps in
    // the TABLE TWINS (identical math, device-resolved role pointers via
    // d_gdn_tab + *d_perm_scalar); the FUSED path passes it (mix_all,
    // conductor.h -- eager from T2 on, captured in T3), after the caller's
    // stage_perm_async has landed the round's perm in the scalar.
    void gdn_mix(int il, cudaStream_t st, bool use_tables = false) {
        const float eps = EPS;
        const float* cw = (const float*)T(il, "ssm_conv1d.weight").data;
        // P12: per-lane recurrent chain -- role k reads role k-1 (written fresh
        // earlier this round) and writes role k. Only lanes < vw are live; a
        // width-vw graph skips the rest, leaving their (never-read) role buffers
        // untouched. Lane a (role 0, the pending token) always runs.
        if (use_tables) {
            float* const* rt = d_gdn_tab + (size_t)il * W_MAX;
            float* const* stab = d_gdn_S_tab + (size_t)il * W_MAX;
            q27k::conv_step_t(rt, d_perm_scalar, 0, 0, W_MAX, qkv, cw, convout, GDN_CH, st);
            for (int L = 1; L < vw; L++)
                q27k::conv_step_t(rt, d_perm_scalar, L - 1, L, W_MAX, qkv_L[L], cw,
                                  convout_L[L], GDN_CH, st);
            q27k::l2norm3(LANESW(convout), 32,
                          GDN_DIM, eps, st, vw);
            q27k::delta_step_t(stab, d_perm_scalar, 0, 0, W_MAX, convout, g, beta, o, st);
            for (int L = 1; L < vw; L++)
                q27k::delta_step_t(stab, d_perm_scalar, L - 1, L, W_MAX, convout_L[L], g_L[L],
                                   beta_L[L], o_L[L], st);
            return;
        }
        q27k::conv_step(RBuf(il, 0), RBuf(il, 0), qkv, cw, convout, GDN_CH, st); // lane 0
        for (int L = 1; L < vw; L++)
            q27k::conv_step(RBuf(il, L - 1), RBuf(il, L), qkv_L[L], cw, convout_L[L], GDN_CH, st);
        // q||k are contiguous (offsets 0 and 2048): 32 heads in one merged call
        q27k::l2norm3(LANESW(convout), 32,
                      GDN_DIM, eps, st, vw);
        q27k::delta_step(SBuf(il, 0), SBuf(il, 0), convout, g, beta, o, st); // lane 0
        for (int L = 1; L < vw; L++)
            q27k::delta_step(SBuf(il, L - 1), SBuf(il, L), convout_L[L], g_L[L], beta_L[L], o_L[L], st);
    }
    // -- split point: back to the weight sweep (per-lane elementwise + o-proj
    //    on the view); mix wrote o_L in place, the view aliases it --
    void gdn_post(int il, const LaneView& v) {
        const float* nw = (const float*)T(il, "ssm_norm.weight").data;
        q27k::gated_norm3(LANESV(v, o), nw,
                          LANESV(v, z),
                          LANESV(v, og), GDN_HEADS, GDN_DIM, EPS, v.stm, v.vw);
        qx5(v, v.og, GDN_V);
        mm5(v, T(il, "ssm_out.weight"), v.y);
    }
    void gdn_pair(int il, const LaneView& v) {
        gdn_pre(il, v);
        gdn_mix(il, stm);
        gdn_post(il, v);
    }

    void attn_pre(int il, const LaneView& v) {
        qx5(v, v.x1, N_EMBD);
        mm5(v, T(il, "attn_q.weight"), v.qg);
        const float* qn = (const float*)T(il, "attn_q_norm.weight").data;
        const float* kn = (const float*)T(il, "attn_k_norm.weight").data;
        for (int L = 0; L < v.vw; L++)
            q27k::rmsnorm_heads(v.qg[L], qn, v.qg[L], N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS, v.stm);
        mm5(v, T(il, "attn_k.weight"), v.kbuf);
        for (int L = 0; L < v.vw; L++)
            q27k::rmsnorm_heads(v.kbuf[L], kn, v.kbuf[L], N_KV, HEAD_DIM, HEAD_DIM, EPS, v.stm);
        mm5(v, T(il, "attn_v.weight"), v.vbuf);
        // rope reads the view's per-lane positions (WIP3 -> IP3: same
        // pointers, const-qualified for the kernel wrapper)
        q27k::IP3 P{};
        for (int i = 0; i < W_PLUMB; i++) P.p[i] = v.pos.p[i];
        q27k::rope3(LANESV(v, qg),
                    N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, P,
                    FREQ_BASE, v.stm, v.vw);
        q27k::rope3(LANESV(v, kbuf), N_KV, HEAD_DIM, N_ROT,
                    HEAD_DIM, P, FREQ_BASE, v.stm, v.vw);
    }
    // -- split point: sequence state (KV cache + attention scratch + kv_kind-
    //    branched turbo3 rotates) begins here (design 2026-07-14) --
    // P1 batching: explicit stream, member width -- same contract as gdn_mix.
    void attn_mix(int il, cudaStream_t st) {
        int ci = attn_cache_idx[il];
        q27k::IP3 P LANESW(d_pos);
        float kq = 1.0f / sqrtf((float)HEAD_DIM);
        // turbo3: rotate all vw Q lanes post-rope (see attn_block); host
        // branch on kv_kind only (init-fixed, graph-capture-safe)
        if (kv_kind == KV_T3)
            q27k::wht3(LANESW(qg), N_HEAD, HEAD_DIM, 2 * HEAD_DIM, false, st, vw);
        // store vw lanes (disjoint slots); each token's attention only reads
        // cache[0 .. its own pos], so later tokens' entries are invisible to earlier ones
        if (kv_kind >= KV_T3)
            q27k::kv_store_t3(LANESW(kbuf),
                              LANESW(vbuf), kcache[ci], vcache[ci],
                              P, N_KV, HEAD_DIM, st, vw, /*k_plain=*/kv_kind == KV_T3V);
        else
            q27k::kv_store3(LANESW(kbuf),
                            LANESW(vbuf), kcache[ci], vcache[ci],
                            P, N_KV * HEAD_DIM, st, vw, kv_fp8);
        q27k::attn_decode3(LANESW(qg), 2 * HEAD_DIM, kcache[ci],
                           vcache[ci],
                           LANESW(attnout),
                           scratch, P, max_ctx, N_HEAD, N_KV, HEAD_DIM, kq, st, vw, kv_kind);
        // inverse-WHT on all vw pooled outputs BEFORE the sigmoid gate
        if (kv_kind >= KV_T3)
            q27k::wht3(LANESW(attnout),
                       N_HEAD, HEAD_DIM, HEAD_DIM, true, st, vw);
    }
    // -- split point: back to the weight sweep (sigmoid gate is pure per-lane
    //    elementwise; mix wrote attnout_L in place, the view aliases it) --
    void attn_post(int il, const LaneView& v) {
        q27k::sigmoid_gate3(LANESV(v, attnout),
                            LANESV(v, qg), N_HEAD, HEAD_DIM, v.stm, v.vw);
        qx5(v, v.attnout, N_HEAD * HEAD_DIM);
        mm5(v, T(il, "attn_output.weight"), v.y);
    }
    void attn_pair(int il, const LaneView& v) {
        attn_pre(il, v);
        attn_mix(il, stm);
        attn_post(il, v);
    }

    // ffn_pair is all-"pre" (design 2026-07-14): every op is a per-lane
    // weight/elementwise sweep on the view, no sequence state, so it needs no
    // mix seam -- the P1 fused round calls it whole on the union view.
    void ffn_pair(int il, const LaneView& v) {
        qx5(v, v.x1, N_EMBD);
        mm5(v, T(il, "ffn_gate.weight"), v.ffn_g);
        mm5(v, T(il, "ffn_up.weight"), v.ffn_u);
        q27k::silu_mul3(LANESV(v, ffn_g),
                        LANESV(v, ffn_u), N_FFN, v.stm, v.vw);
        qx5(v, v.ffn_g, N_FFN);
        mm5(v, T(il, "ffn_down.weight"), v.y);
    }

    // launch sequence for one speculative round (graph-capturable: all state
    // through device memory, pointers fixed for a given parity)
    // P11: draft half -- prep + the 4 sequential MTP passes producing
    // d_draft..d_draft4. Split out so the constrained path can read the
    // drafts back and stage per-lane grammar masks before the verify half.
    // P14 early-exit: one draft step. k==0 is prep_round + draft 1 (h_next,
    // embed(t1)) at pos_m -> d_draft; k>0 chains MTP's own post-head-norm hidden
    // (x1) via the h_next{k+1} D2D into draft k+1 at pos_m{k+1} -> d_draft{k+1}
    // (each pass also fills its MTP KV row).
    // P12: each draft's top1-top2 margin (the drafter's confidence) lands in
    // d_draft_margin[k]. P14: FUSED into the draft's argmax -- one full-vocab
    // pass per draft (was argmax + a separate k_margin scan). Value-identical
    // to the old pair (same token, same margin), so every existing graph stays
    // token-identical and the margin slot mapping (0..4 for drafts 1..5) is
    // unchanged. Margins are write-only scratch, read only by the gated round.
    // Concatenating steps 0..dmax-1 reproduces the pre-refactor monolithic
    // kernel sequence byte-for-byte (the D2D that used to trail step k now
    // leads step k+1 -- same stream order), so every existing graph capture is
    // unchanged; the gated rounds additionally capture each step alone
    // (draft_step_graph) to stop drafting at the first sub-theta margin.
    // the W_PLUMB verify positions + 7 MTP positions ride WIP3 structs
    // (prep_round hit the 17-param wall). The wide positions are written every
    // round but only read by graphs captured at vw > 8 (suffix widths, P1).
    q27k::WIP3 lane_pos() {
        return LANESW(d_pos);
    }
    q27k::WIP3 mtp_pos() {
        return {{d_pos_m, d_pos_m2, d_pos_m3, d_pos_m4, d_pos_m5, d_pos_m6, d_pos_m7}};
    }
    void spec_draft_step_launches(int k) {
        // P2c: prep + chain selection live in draft_step_prep/mtp_step_view
        // (ONE pointer table shared with the fused union builder).
        // Composition is byte-identical to the pre-P2c body: step 0 =
        // prep_round + the (h_next, d_token, d_draft, d_pos_m, margin+0)
        // MTP pass; step k>0 = the x1 -> hs[k] D2D + the (hs[k], ts[k],
        // ds[k], ps[k], margin+k) pass -- same kernels, same args, same
        // order, so every existing graph capture records the same nodes.
        draft_step_prep(k, stm);
        MtpLaneView v = mtp_step_view(k);
        mtp_pre(v);
        mtp_attn(v.pos[0], stm);
        mtp_post(v);
        mtp_tail(v);
    }

    void spec_draft_launches() {
        for (int k = 0; k < dmax; k++) spec_draft_step_launches(k);
    }

    // P11: verify half -- batch-5 forward of {pending, d1..d4}, masked argmax
    // per lane, finish_round. Reads d_draft..d_draft4 (set by the draft half)
    // and d_mask_ids/d_accept_cap (staged by the host between halves).
    // batch-5 forward of {pending, d1..d4} -> logits2[5*VOCAB]. Shared verbatim
    // by the greedy verify tail and the Phase-2 sampled tail (so the two never
    // drift; the sampled path samples the identical logits the greedy path argmaxes).
    // Lane t of the verify batch reads token: t=0 the pending token, t>=1 the
    // t'th draft. One loop over W_PLUMB so a widening never has to grow a
    // hand-written brace list again.
    q27k::IP3 verify_tokens() const {
        q27k::IP3 t{};
        t.p[0] = d_token;
        for (int k = 0; k + 1 < W_PLUMB; k++) t.p[k + 1] = d_draft_L[k];
        return t;
    }
    void spec_verify_forward(const LaneView& v) {
        // P0 batching: the CALLER builds the view (solo: solo_view() -- a
        // vw/stm snapshot taken exactly when the members were read before),
        // so the P1 fused round can hand this same forward a union view.
        // SKELETON MIRROR WARNING (review M3): fused_verify_round in
        // src/conductor.h mirrors this loop skeleton (embed3 -> per-layer
        // rmsnorm3/pair/add3/rmsnorm3/ffn_pair/add3 -> output norm/qx5/head
        // mm5) with per-engine mix sub-launches. Structural changes HERE
        // MUST be mirrored there and re-gated with fused_smoke (build line
        // in tools/fused_smoke.cu's header).
        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed3((const int8_t*)emb.data, (const __half*)emb.scales, v.vtok,
                     N_EMBD, LANESV(v, h), v.stm,
                     v.vw);
        q27k::CP3 Hc LANESV(v, h),
            Yc LANESV(v, y);
        q27k::P3 Hm LANESV(v, h),
            X1m LANESV(v, x1);
        for (int il = 0; il < N_LAYER; il++) {
            const float* an = (const float*)T(il, "attn_norm.weight").data;
            q27k::rmsnorm3(Hc, an, X1m, N_EMBD, EPS, v.stm, v.vw);
            if (attn_layer[il]) attn_pair(il, v);
            else gdn_pair(il, v);
            q27k::add3(Hm, Yc, N_EMBD, v.stm, v.vw);
            const float* pn = (const float*)T(il, "post_attention_norm.weight").data;
            q27k::rmsnorm3(Hc, pn, X1m, N_EMBD, EPS, v.stm, v.vw);
            ffn_pair(il, v);
            q27k::add3(Hm, Yc, N_EMBD, v.stm, v.vw);
        }
        const float* on = (const float*)dm.get("output_norm.weight").data;
        q27k::rmsnorm3(Hc, on, X1m, N_EMBD, EPS, v.stm, v.vw);
        qx5(v, v.x1, N_EMBD);
        const char* vhead = (fast_head && dm.model_has("output_q4.weight")) ? "output_q4.weight"
                                                                             : "output.weight";
        // lane t's logits live at v.lg[t] (solo: logits2 + t*VOCAB, alloc is
        // W_MAX*VOCAB; only lanes < vw are computed, and only those are read).
        mm5(v, dm.get(vhead), v.lg);
    }

    // P11: verify half -- batch-5 forward, masked argmax per lane, finish_round.
    // P0 batching: forward + per-lane argmax read the view; the mask pool
    // (d_mask_pool/d_mask_ids/d_amax) and finish_round stay MEMBER-based --
    // the tail is per-engine forever (design 2026-07-14).
    // P1 batching split: the greedy TAIL (per-lane argmax + finish_round)
    // factored out of spec_verify_launches so the fused round can run ONE
    // union forward and then each engine's own tail. The tail reads width and
    // stream from the view (fused: granted width + the conductor stream;
    // solo: solo_view() where v.vw == vw and v.stm == stm, so the composed
    // function emits the bit-identical launch sequence -- addendum A8). The
    // mask pool (d_mask_pool/d_mask_ids/d_amax) and finish_round's commit
    // state stay MEMBER-based -- the tail is per-engine forever.
    void spec_verify_tail(const LaneView& v) {
        // P7: slot 0 (the post-pending lane) is the constrained one; the rest
        // keep id -1 (v1 caps acceptance in-grammar instead of chasing
        // draft-dependent states the host cannot know pre-launch).
        // W16: was an unrolled `if (vw > k)` chain per lane. The loop emits the
        // identical launch sequence in the identical order for any vw, so the
        // captured graphs -- and the tokens they produce -- are unchanged at
        // every width the chain covered.
        for (int t = 0; t < v.vw; t++)
            q27k::argmax_masked(v.lg[t], VOCAB, d_mask_pool, mask_words,
                                d_mask_ids, t, v.dv[t], d_amax, v.stm);
        // P12: a width-vw verify computed columns 0..vw-1; cap acceptance at vw-1
        // drafts so finish never commits an uncomputed lane. vw=5 => max_draft=4.
        q27k::IP3 drafts{};
        for (int k = 0; k + 1 < W_PLUMB; k++) drafts.p[k] = d_draft_L[k];
        q27k::finish_round(d_P, d_token, drafts,
                           LANESW(d_v),
                           LANESW(x1),
                           h_next, d_outcome, N_EMBD, d_accept_cap, v.vw - 1, v.stm);
    }
    void spec_verify_launches(const LaneView& v) {
        spec_verify_forward(v);
        spec_verify_tail(v);
    }

    // Phase 2: sampled verify tail. Same forward; replace the 5 argmax lanes +
    // equality-chain finish with per-lane nucleus stats, rejection-sampling
    // acceptance (k_spec_accept), a resample of the new pending from the stop
    // lane (k_sample_stop -> d_token), and finish keyed on the accepted count n.
    // Draws key Philox on *d_P; greedy graphs stay bitwise (separate graph set).
    // P0 batching: forward + per-lane nucleus read the view; spec_accept/
    // sample_stop/finish_sampled stay MEMBER-based (they take logits2 as a
    // flat base pointer -- per-engine tail state, per-engine forever).
    // P1 batching split: sampled TAIL, same seam as spec_verify_tail. Width
    // and stream come from the view (solo: == members, bit-identical); the
    // flat logits2 base and the accept/finish commit state stay MEMBER-based.
    void spec_verify_tail_sampled(const LaneView& v) {
        // P14: width-vw sampled verify -- nucleus stats + accept walk over the
        // first vw lanes only (vw=5 monolithic; vw=cap+1 under the gate). The
        // accept walk caps at vw-1 drafts so finish never commits an uncomputed
        // lane. vw=5 => max_draft=4 (the pre-P14 behavior). k_finish_sampled is
        // unchanged: it keys on n<=vw and its src select covers n in 1..5.
        for (int k = 0; k < v.vw; k++)
            q27k::nucleus(v.lg[k], VOCAB, d_samp, d_nuc + k * 4, v.stm);
        q27k::spec_accept(logits2, d_nuc, d_draft, d_draft2, d_draft3, d_draft4, d_samp, d_P,
                          d_accept_cap, v.vw - 1, VOCAB, d_spec, v.stm);
        q27k::sample_stop(logits2, d_nuc, d_spec, d_samp, d_P, VOCAB, d_token, d_amax, v.stm);
        q27k::finish_sampled(d_P, d_token, d_spec, d_draft, d_draft2, d_draft3, d_draft4, x1,
                             x1_L[1], x1_L[2], x1_L[3], x1_L[4], h_next, d_outcome, N_EMBD,
                             v.stm);
    }
    void spec_verify_launches_sampled(const LaneView& v) {
        spec_verify_forward(v);
        spec_verify_tail_sampled(v);
    }

    void spec_round_launches() {
        spec_draft_launches();
        // P0 batching: build the solo view ONCE per round, verify half only --
        // the draft half never touches mm5/qx5 (Task 2 grep), so it stays
        // member-based and view-free.
        const LaneView sv = solo_view();
        spec_verify_launches(sv);
    }

    void spec_sample_round_launches() {
        spec_draft_launches();
        const LaneView sv = solo_view();
        spec_verify_launches_sampled(sv);
    }

    void build_spec_graphs() {
        // one warm (executing) round to initialize lazy CUDA state, then reset.
        // seed + reset are factored so the Phase-2 sampled graph set warms the
        // same way (its verify tail launches new kernels that also need warming).
        // P12b: Q27_MAXD (4 or 5) picks the deepest gated draft. Read here (not
        // with Q27_PMIN below) because it shapes capture: draft depth, warm
        // width, and how many per-width verify graphs to build.
        // P14 draft early-exit: default ON when gated; Q27_DEXIT=0 restores the
        // monolithic draft (the A/B lever). Read BEFORE Q27_MAXD: the auto
        // ladder ceiling depends on it (below).
        if (const char* de = getenv("Q27_DEXIT")) dexit_on = atoi(de) != 0;
        if (const char* md = getenv("Q27_MAXD")) {
            // maxd6: auto = the 4..6 ladder (per-step drafting serves every
            // ceiling). Under Q27_DEXIT=0 the monolithic gated draft has no
            // depth-5 graph beneath a depth-6 ceiling, so auto clamps to the
            // shipped 4..5 ladder there (dexit is default-on; A/B knob only).
            // auto = the 4..6 ladder (maxd7 A/B: depth-7 LOSES ~6% vs d6 even at
            // y7 .77/fired .73 -- the width-8 round costs +3.0 ms, ~2x the
            // extrapolation; cost attribution owed before any default). auto7
            // opts into the 4..7 ladder for that future retune.
            if (!strcmp(md, "auto")) { maxd_auto = true; gate_maxd = dexit_on ? 6 : 5; }
            else if (!strcmp(md, "auto7")) { maxd_auto = true; gate_maxd = dexit_on ? 7 : 5; }
            else gate_maxd = atoi(md);
        }
        if (gate_maxd < 4) gate_maxd = 4;
        if (gate_maxd > 7) gate_maxd = 7;
        if (maxd_auto) dctl.k_max = gate_maxd;
        // P2: the width at which mm5 switches from the GEMV to k_vgemm.
        // Q27_GEMM_MIN=99 disables the GEMM entirely (the in-binary A/B control:
        // same binary, GEMM off, must reproduce the old round AND byte-identical
        // output -- gate 5).
        if (const char* e = getenv("Q27_GEMM_MIN")) gemm_min = atoi(e);
        if (const char* e = getenv("Q27_PF_BATCH_MIN")) pf_batch_min = std::max(2, atoi(e));
        // Canonical coupling (same failure class as the gemm_min guardrail):
        // the canonical bitwise prompt is 5 tokens, i.e. SERIAL-path under the
        // default 32. The chunked path rounds differently (measured: greedy
        // text diverges), so a sub-default setting here silently re-paths the
        // canonical gate. The server profile sets 2 deliberately (serving has
        // its own refs); anything else gets a banner, not a refusal.
        if (pf_batch_min < 32)
            fprintf(stderr,
                    "q27: Q27_PF_BATCH_MIN=%d < 32 -- tiny prompts take the CHUNKED "
                    "prefill path; the 5-token canonical md5 does NOT hold here\n",
                    pf_batch_min);
        // THE GUARDRAIL. The canonical bitwise gate is structural only while the
        // ladder's widest verify (gate_maxd+1) stays strictly below gemm_min. If a
        // future ceiling or a careless env ever crosses that line, the ladder would
        // silently start computing logits on a different numeric path and the
        // canonical md5 would drift -- a class of bug this project has paid for
        // twice today. Refuse to run instead.
        if (gate_maxd + 1 >= gemm_min) {
            fprintf(stderr,
                    "q27: FATAL -- ladder verify width %d reaches the GEMM path "
                    "(Q27_GEMM_MIN=%d). The canonical bitwise gate no longer holds "
                    "by construction. Raise Q27_GEMM_MIN above %d or lower Q27_MAXD.\n",
                    gate_maxd + 1, gemm_min, gate_maxd + 1);
            abort();
        }
        // P13 adaptive-maxd tunables (bench-tunable; defaults from the design)
        if (const char* e = getenv("Q27_MAXD_RESET")) maxd_reset = atoi(e) != 0;
        if (const char* e = getenv("Q27_MAXD_EMA")) dctl.ema_a = (float)atof(e);
        if (const char* e = getenv("Q27_MAXD_HI")) dctl.hi = (float)atof(e);
        if (const char* e = getenv("Q27_MAXD_HI6")) dctl.hi6 = (float)atof(e);
        if (const char* e = getenv("Q27_MAXD_HI7")) dctl.hi7 = (float)atof(e);
        if (const char* e = getenv("Q27_MAXD_FLO7")) dctl.flo7 = (float)atof(e);
        if (const char* e = getenv("Q27_MAXD_FLO6")) dctl.flo6 = (float)atof(e);
        if (const char* e = getenv("Q27_MAXD_LO")) dctl.lo = (float)atof(e);
        // width-12 P1: suffix envs parsed BEFORE the warm/capture section --
        // Q27_SUFFIX_W shapes the warm width and adds one per-perm verify
        // graph at exactly that width. <= gate_maxd+1 (or unset) = legacy.
        // value-aware since the CC-defaults flip: Q27_SUFFIX=0 disables
        // (was presence-only -- =0 used to ENABLE).
        suffix_on = getenv("Q27_SUFFIX") && atoi(getenv("Q27_SUFFIX")) != 0;
        sfx_dbg = getenv("Q27_SUFFIX_DBG") != nullptr;
        if (const char* sl = getenv("Q27_SUFFIX_L")) sfx_L = atoi(sl);
        if (const char* sw = getenv("Q27_SUFFIX_W")) {
            sfx_w = atoi(sw);
            if (sfx_w < gate_maxd + 1) sfx_w = 0;      // narrower than gated = legacy
            if (sfx_w > W_MAX) sfx_w = W_MAX;
        }
        // W16: lane seeds were a by-name z0..z11 list that stopped at 12 -- the
        // same hand-written-lane-list shape that left refinish_round's slots
        // nullptr. Warm-up positions only (prep_round rewrites every lane each
        // round), but it must cover W_PLUMB or a wider build seeds garbage.
        int zs[W_PLUMB];
        for (int i = 0; i < W_PLUMB; i++) zs[i] = i;
        int z0 = 0, z1 = 1, z2 = 2, z3 = 3, z4 = 4, z5 = 5, z6 = 6;
        auto seed_positions = [&]() {
            for (int i = 0; i < W_PLUMB; i++)
                CUDA_CHECK(
                    cudaMemcpyAsync(d_pos_L[i], &zs[i], 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemcpyAsync(d_pos_m, &z0, 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemcpyAsync(d_pos_m2, &z1, 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemcpyAsync(d_pos_m3, &z2, 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemcpyAsync(d_pos_m4, &z3, 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemcpyAsync(d_pos_m5, &z4, 4, cudaMemcpyHostToDevice, stm)); // P12b
            CUDA_CHECK(cudaMemcpyAsync(d_pos_m6, &z5, 4, cudaMemcpyHostToDevice, stm)); // maxd6
            CUDA_CHECK(cudaMemcpyAsync(d_pos_m7, &z6, 4, cudaMemcpyHostToDevice, stm)); // maxd7
            CUDA_CHECK(cudaMemcpyAsync(d_token, &z0, 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemset(d_P, 0, 4));
        };
        auto reset_gdn_mtp = [&]() {
            for (int il = 0; il < N_LAYER; il++)
                if (!attn_layer[il]) {
                    size_t sb = (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4;
                    CUDA_CHECK(cudaMemset(S[il], 0, sb));
                    CUDA_CHECK(cudaMemset(conv_ring[il], 0, 3 * GDN_CH * 4));
                    for (int r = 0; r < W_PLUMB - 1; r++)
                        if (S_sp[r][il]) {
                            CUDA_CHECK(cudaMemset(S_sp[r][il], 0, sb));
                            CUDA_CHECK(cudaMemset(ring_sp[r][il], 0, 3 * GDN_CH * 4));
                        }
                }
            CUDA_CHECK(cudaMemset(mtp_k, 0, kv_bytes(false)));
            CUDA_CHECK(cudaMemset(mtp_v, 0, kv_bytes(true)));
        };
        // P12b: warm the WIDEST kernels (distinct gemv<N>/ntok instantiations)
        // so graph capture never triggers a lazy module load. Output is
        // discarded and reset below. width-12 P1: the suffix width (when
        // wider than the gated width) is the widest thing captured.
        perm = 0; dmax = gate_maxd; vw = sfx_width();
        seed_positions();
        spec_round_launches();
        CUDA_CHECK(cudaStreamSynchronize(stm));
        reset_gdn_mtp();
        // capture all 12 cyclic permutations (capture records; does not execute)
        for (int p = 0; p < W_MAX; p++) {
            perm = p;
            // monolithic ungated round: depth-4 draft + width-5 verify.
            dmax = 4; vw = 5;
            cudaGraph_t gr;
            CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
            spec_round_launches();
            CUDA_CHECK(cudaStreamEndCapture(stm, &gr));
            CUDA_CHECK(cudaGraphInstantiate(&spec_graph[p], gr, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphDestroy(gr));
            // P12b: the gated draft graph produces gate_maxd drafts + margins.
            dmax = gate_maxd;
            // P11: same round, split at the draft/verify boundary. The two
            // halves reference the identical buffers, so launching D then V
            // back-to-back on stm equals the monolithic graph; the host reads
            // drafts + stages masks between them in the constrained path.
            cudaGraph_t gd, gv;
            CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
            spec_draft_launches();
            CUDA_CHECK(cudaStreamEndCapture(stm, &gd));
            CUDA_CHECK(cudaGraphInstantiate(&draft_graph[p], gd, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphDestroy(gd));
            // P13 adaptive maxd: also capture the depth-4 draft (draft_graph_lo)
            // so spec_round can pick draft depth per round. gate_maxd is forced to
            // 5 under auto, so draft_graph[p] above is the depth-5 (hi) graph.
            // P14: capture it whenever gate_maxd==5 (auto OR fixed Q27_MAXD=5), not
            // just auto -- the sampled+gated path is always a depth-4 draft and
            // needs the depth-4 graph even under fixed depth-5. One extra graph per
            // perm, no new buffers; greedy still selects draft_graph_lo only under
            // maxd_auto (spec_round unchanged), so its behavior is untouched.
            if (gate_maxd >= 5) {
                dmax = 4;
                cudaGraph_t gdl;
                CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
                spec_draft_launches();
                CUDA_CHECK(cudaStreamEndCapture(stm, &gdl));
                CUDA_CHECK(cudaGraphInstantiate(&draft_graph_lo[p], gdl, nullptr, nullptr, 0));
                CUDA_CHECK(cudaGraphDestroy(gdl));
                dmax = gate_maxd;
            }
            // P14 early-exit: capture each draft step alone. Steps don't read
            // dmax; steps 0..3 are depth-independent and step 4 exists only
            // when gate_maxd==5. draft_graph/draft_graph_lo stay captured for
            // the constrained path and the Q27_DEXIT=0 monolithic fallback.
            for (int k = 0; k < gate_maxd; k++) {
                cudaGraph_t gs;
                CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
                spec_draft_step_launches(k);
                CUDA_CHECK(cudaStreamEndCapture(stm, &gs));
                CUDA_CHECK(cudaGraphInstantiate(&draft_step_graph[k][p], gs, nullptr, nullptr, 0));
                CUDA_CHECK(cudaGraphDestroy(gs));
            }
            CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
            spec_verify_launches(solo_view());
            CUDA_CHECK(cudaStreamEndCapture(stm, &gv));
            CUDA_CHECK(cudaGraphInstantiate(&verify_graph[p], gv, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphDestroy(gv));
            // P12/P12b: per-width verify graphs (W = cap+1 lanes, 2..6). Same
            // buffers as the widest verify; only ntok/nbatch shrink + finish caps
            // at W-1, so committed state and emitted tokens are width-invariant.
            // W>=2: the batched gemv (gemv_q?_n) has no nbatch=1 kernel, so the
            // shallowest gated round verifies {pending, d1} (width 2), not 1.
            for (int W = 2; W <= gate_maxd + 1; W++) {
                vw = W;
                cudaGraph_t gw;
                CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
                spec_verify_launches(solo_view());
                CUDA_CHECK(cudaStreamEndCapture(stm, &gw));
                CUDA_CHECK(cudaGraphInstantiate(&verify_graph_w[W][p], gw, nullptr, nullptr, 0));
                CUDA_CHECK(cudaGraphDestroy(gw));
            }
            // width-12 P1: the suffix drafter's wide verify. Suffix rounds
            // always launch full width, so only sfx_w itself is captured
            // (not every width 9..sfx_w) -- 12 extra graphs total.
            if (sfx_w > gate_maxd + 1) {
                vw = sfx_w;
                cudaGraph_t gs_;
                CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
                spec_verify_launches(solo_view());
                CUDA_CHECK(cudaStreamEndCapture(stm, &gs_));
                CUDA_CHECK(
                    cudaGraphInstantiate(&verify_graph_w[sfx_w][p], gs_, nullptr, nullptr, 0));
                CUDA_CHECK(cudaGraphDestroy(gs_));
            }
            vw = 5;
        }
        // Phase 2: sampled graph set -- identical draft half, rejection-sampling
        // verify tail. Warm with dummy params (the greedy graphs above are
        // already instantiated and independent of the device state churned here).
        q27k::SampleParams warm{1.f, 1.f, 0ull};
        CUDA_CHECK(cudaMemcpyAsync(d_samp, &warm, sizeof warm, cudaMemcpyHostToDevice, stm));
        dmax = 4; vw = 5; // sampling stays depth-4 (5-lane) in this phase
        seed_positions();
        spec_sample_round_launches();
        CUDA_CHECK(cudaStreamSynchronize(stm));
        reset_gdn_mtp();
        for (int p = 0; p < W_MAX; p++) {
            perm = p;
            cudaGraph_t gr;
            CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
            spec_sample_round_launches();
            CUDA_CHECK(cudaStreamEndCapture(stm, &gr));
            CUDA_CHECK(cudaGraphInstantiate(&spec_sample_graph[p], gr, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphDestroy(gr));
            // P14: per-width sampled verify graphs (W=2..5), mirroring the greedy
            // verify_graph_w loop. The sampled tail is always depth-4, so the
            // widest sampled verify is width-5 (W<=5) regardless of gate_maxd.
            // Same buffers as the monolithic sampled verify; only vw shrinks.
            for (int W = 2; W <= 5; W++) {
                vw = W;
                cudaGraph_t gw;
                CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
                spec_verify_launches_sampled(solo_view());
                CUDA_CHECK(cudaStreamEndCapture(stm, &gw));
                CUDA_CHECK(
                    cudaGraphInstantiate(&verify_sample_graph_w[W][p], gw, nullptr, nullptr, 0));
                CUDA_CHECK(cudaGraphDestroy(gw));
            }
            vw = 5;
        }
        perm = 0;
        // P12: confidence-gated depth. Q27_PMIN=theta engages the gate (drafter
        // top1-top2 margin >= theta extends the verify one lane deeper). <=0 or
        // unset = off (always full width 5 = the canonical depth-4 round).
        const char* pm = getenv("Q27_PMIN");
        if (pm) pmin_theta = (float)atof(pm);
        phase_stats = getenv("Q27_PHASE_STATS") && atoi(getenv("Q27_PHASE_STATS")) != 0;
        // (suffix envs parsed above, pre-capture -- width-12 P1)
        if (suffix_on)
            fprintf(stderr, "suffix drafter ON: L>=%d, width %d (greedy gated rounds only)\n",
                    sfx_L, sfx_width());
        fprintf(stderr,
                "spec graphs captured (%d perms, depth-4; +split D/V; +per-width verify "
                "2..%d%s; +P14 sampled per-width verify 2..5 + per-step draft 0..%d); "
                "Q27_PMIN=%.3f (%s), gate_maxd=%d%s, dexit=%d\n",
                W_MAX, // was the literal "12" -- the capture loop is over W_MAX
                gate_maxd + 1, maxd_auto ? "; +P13 depth-4 draft" : "", gate_maxd - 1,
                pmin_theta, pmin_theta > 0 ? "gated" : "off", gate_maxd,
                maxd_auto ? (gate_maxd >= 6 ? (gate_maxd == 7 ? " (auto: ladder 4..7)"
                                                                 : " (auto: ladder 4..6)")
                                              : " (auto: floats 4..5)")
                          : "",
                dexit_on ? 1 : 0);
    }

    // one speculative round; returns tokens emitted (1..gate_maxd+1).
    // All position math + acceptance runs on device; host reads 36 bytes.
    int spec_round(int* emit) {
        int md_used = -1;  // P13: draft-depth ceiling actually used this round
        int gate_cap = -1; // this round's margin-run depth (gated branches only)
        bool sfx_round = false; // suffix-drafted round (own stats class)
        std::chrono::steady_clock::time_point ph_t0, ph_t1;
        bool ph_timed = false; // set once the draft half is stamped (gated branches)
        int ph_W = 0;          // this round's verify width (gated branches)
        if (tool_split_active && on_drafts) {
            // P11 constrained path: run drafts, read them back, let the host
            // stage per-lane grammar masks (uncapped), then verify. Full spec
            // acceptance survives inside tool-call bodies (vs the cap=1 hack).
            // P13: under auto, draft_graph[perm] is depth-5; the constrained
            // verify is width-5 (reads 4 drafts), so draft the depth-4 graph.
            CUDA_CHECK(
                cudaGraphLaunch(maxd_auto ? draft_graph_lo[perm] : draft_graph[perm], stm));
            int dr[4];
            CUDA_CHECK(cudaMemcpyAsync(&dr[0], d_draft, 4, cudaMemcpyDeviceToHost, stm));
            CUDA_CHECK(cudaMemcpyAsync(&dr[1], d_draft2, 4, cudaMemcpyDeviceToHost, stm));
            CUDA_CHECK(cudaMemcpyAsync(&dr[2], d_draft3, 4, cudaMemcpyDeviceToHost, stm));
            CUDA_CHECK(cudaMemcpyAsync(&dr[3], d_draft4, 4, cudaMemcpyDeviceToHost, stm));
            CUDA_CHECK(cudaStreamSynchronize(stm));
            on_drafts(dr); // stages d_mask_ids[0..4], sets d_accept_cap=0 (async on stm)
            CUDA_CHECK(cudaGraphLaunch(verify_graph[perm], stm));
        } else if (suffix_on && sfx_valid && pmin_theta > 0.f && h_mask_id0 < 0 &&
                   sfx.propose_with(last_pending, sfx_width() - 1, h_sfx_prop) >= sfx_L) {
            // Suffix round: the committed stream's suffix recurs -- fill the
            // draft lanes from the earlier continuation and skip the MTP
            // chain entirely (zero draft cost). prep_round is the same
            // launch that opens draft step 0; the verify graph, finish walk
            // and outcome path are the standard gated-round machinery, so
            // emitted tokens stay greedy-identical regardless of proposal
            // quality. h_next for the next round comes from the verify's
            // resident role buffers, unaffected by how drafts were made.
            sfx_round = true;
            const int sw = sfx_width(); // wide when Q27_SUFFIX_W > gate_maxd+1 (P1)
            if (phase_stats) ph_t0 = std::chrono::steady_clock::now();
            q27k::prep_round(d_P, d_token, lane_pos(), mtp_pos(), W_MAX, D_MAX_MTP, d_outcome,
                             stm);
            for (int k = 0; k < sw - 1; k++)
                CUDA_CHECK(cudaMemcpyAsync(d_draft_L[k], &h_sfx_prop[k], 4,
                                           cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaGraphLaunch(verify_graph_w[sw][perm], stm));
        } else if (pmin_theta > 0.f && h_mask_id0 < 0) {
            // P12/P12b confidence-gated depth (unconstrained decode only): draft to
            // the ceiling + per-draft margins, cap depth at the leading run with
            // margin >= theta, and verify only cap+1 lanes -- skipping the deep-KV
            // verify when the drafter is unconfident. Emitted tokens are
            // width-invariant vs spec_graph (only round count changes).
            // Adaptive maxd: the ceiling floats over the depthctl ladder
            // (4..gate_maxd) per stream. Under dexit the per-STEP draft graphs
            // serve every ceiling; the monolithic fallback below only knows
            // depth-4 (draft_graph_lo) and depth-gate_maxd (draft_graph).
            md_used = maxd_auto ? dctl.cur : gate_maxd;
            if (phase_stats) ph_t0 = std::chrono::steady_clock::now();
            if (dexit_on) {
                // P14 draft early-exit: launch draft steps one at a time and
                // stop at the first sub-theta margin (that step still ran --
                // its margin is what stopped the loop, so its MTP KV row and
                // d_draft slot are written, same as monolithic). cap semantics
                // are unchanged: leading run of margin >= theta.
                int cap = 0, launched = 0;
                for (int k = 0; k < md_used; k++) {
                    CUDA_CHECK(cudaGraphLaunch(draft_step_graph[k][perm], stm));
                    launched++;
                    CUDA_CHECK(cudaMemcpyAsync(h_draft_margin + k, d_draft_margin + k, 4,
                                               cudaMemcpyDeviceToHost, stm));
                    CUDA_CHECK(cudaStreamSynchronize(stm));
                    if (h_draft_margin[k] < pmin_theta) break;
                    cap++;
                }
                if (phase_stats) {
                    ph_t1 = std::chrono::steady_clock::now();
                    gs.draft_ms +=
                        std::chrono::duration<double, std::milli>(ph_t1 - ph_t0).count();
                    gs.draft_steps += launched;
                    ph_timed = true;
                }
                int W = cap + 1 < 2 ? 2 : cap + 1; // no width-1 gemv; floor at 2
                // Width-floor top-up: a width-W verify can commit up to n=W
                // tokens (finish walks max_draft=W-1 drafts), so W draft rows
                // must exist for byte/round identity with the monolithic path.
                // Only fires at cap==0 (W=2 > launched=1): run draft step 1
                // too. Its inputs (d_draft, x1 after step 0) are final at this
                // point, so it writes exactly what monolithic step 1 writes.
                for (int k = launched; k < W && k < md_used; k++)
                    CUDA_CHECK(cudaGraphLaunch(draft_step_graph[k][perm], stm));
                CUDA_CHECK(cudaGraphLaunch(verify_graph_w[W][perm], stm));
                ph_W = W;
                gate_cap = cap;
            } else {
                // Q27_DEXIT=0: monolithic gated draft (the pre-P14 A/B baseline).
                cudaGraphExec_t dg =
                    (maxd_auto && md_used == 4) ? draft_graph_lo[perm] : draft_graph[perm];
                CUDA_CHECK(cudaGraphLaunch(dg, stm));
                CUDA_CHECK(cudaMemcpyAsync(h_draft_margin, d_draft_margin, 7 * 4,
                                           cudaMemcpyDeviceToHost, stm));
                CUDA_CHECK(cudaStreamSynchronize(stm));
                if (phase_stats) {
                    ph_t1 = std::chrono::steady_clock::now();
                    gs.draft_ms +=
                        std::chrono::duration<double, std::milli>(ph_t1 - ph_t0).count();
                    gs.draft_steps += md_used;
                    ph_timed = true;
                }
                int cap = 0;
                while (cap < md_used && h_draft_margin[cap] >= pmin_theta) cap++;
                int W = cap + 1 < 2 ? 2 : cap + 1; // no width-1 gemv; floor at 2
                CUDA_CHECK(cudaGraphLaunch(verify_graph_w[W][perm], stm));
                ph_W = W;
                gate_cap = cap;
            }
        } else {
            CUDA_CHECK(cudaGraphLaunch(spec_graph[perm], stm));
        }
        // outcome: [0]=n, [1..W_PLUMB]=the emitted tokens, [OUTCOME_INTS-1]=new pending.
        int oc[OUTCOME_INTS];
        CUDA_CHECK(cudaMemcpyAsync(oc, d_outcome, OUTCOME_INTS * 4, cudaMemcpyDeviceToHost, stm));
        CUDA_CHECK(cudaStreamSynchronize(stm));
        if (ph_timed) {
            double v = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - ph_t1)
                           .count();
            gs.verify_ms += v;
            if (ph_W >= 2 && ph_W <= W_MAX) { gs.vw_ms[ph_W] += v; gs.vw_n[ph_W]++; }
        } else if (sfx_round && phase_stats) {
            // width-12 P1: suffix rounds are pure verify (prep + H2D + wide
            // graph); their wall gets its OWN bucket instead of polluting
            // the gated per-width curve -- every suffix round runs one
            // width (sfx_width()), so sfx_ms/sfx_rounds IS the wide-width
            // cost point the P2 curve needs.
            gs.sfx_ms += std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - ph_t0)
                             .count();
            gs.sfx_rounds++;
        }
        int n = oc[0];
        if (sfx_round) {
            sfx_fired++;
            sfx_tok += n;
        }
        if (gate_cap >= 0) {
            gate_cap_hist[gate_cap]++; gate_n_hist[n]++;
            for (int j = 1; j <= gate_cap; j++) {
                gate_lane_fired[j]++;
                if (n >= j + 1) gate_lane_acc[j]++;
            }
        }
        // P13 adaptive maxd: fold this round's realized accept into the ceiling
        // (extracted to depthctl.h; semantics + comments live there).
        if (maxd_auto) {
            if (sfx_round) {
                // Suffix rounds consume exactly the saturating echo stretches
                // that drive ladder promotion (measured on repro: suffix-on
                // starved the ladder, md5 113 -> 0, and the non-echo rounds
                // lost the depth-5 edge = the whole win). A suffix round
                // committing past the current ceiling is ceiling-saturation
                // evidence -- the MTP chain saturates on those same echo
                // tokens (baseline's n=6 rounds). A non-saturating suffix
                // round says nothing about the MTP drafter (it wasn't
                // consulted): skip, don't push demotion.
                if (n >= dctl.cur + 1) dctl.update(dctl.cur, dctl.cur, n);
            } else {
                dctl.update(md_used, gate_cap, n);
            }
        }
        for (int k = 0; k < n; k++) emit[k] = oc[1 + k];
        last_pending = oc[OUTCOME_INTS - 1];
        sfx_valid = true; // pending now known on host; suffix may fire next round
        if (sfx_dbg) {
            fprintf(stderr, "[sfxdbg-oc] n=%d em=", n);
            for (int k = 0; k < n; k++) fprintf(stderr, "%d,", oc[1 + k]);
            fprintf(stderr, " pend=%d sfx_round=%d\n", oc[OUTCOME_INTS - 1], sfx_round ? 1 : 0);
        }
        perm = (perm + (n - 1)) % W_MAX;
        return n;
    }

    // ---- P1 continuous-batching conductor surface (addendum A4) ----
    // The conductor drives a fused round through THESE entrypoints plus the
    // pre/mix/post trio, ffn_pair, solo_view() and the split verify tails --
    // no friend access, no raw member reaches from conductor.h. Everything
    // below is host bookkeeping the solo path already runs inside spec_round;
    // spec_round itself is left untouched (it owns telemetry/adaptive-depth
    // extras the P1 fixed-ladder conductor does not use).
    //
    // Thin named accessors (review M3): every raw member conductor.h reads,
    // behind a name with a one-line why, so the reach surface stays
    // greppable and deliberate (A4: a new member need adds an accessor here
    // with a comment saying why -- never a raw reach from conductor.h).
    // why: fused members must share ONE weight set -- the union sweep reads
    // weights through es[0]; build_union_view identity-compares this, and
    // the fused forward reads embed/norm/head tensors through it.
    const q27::DeviceModel& shared_dm() const { return dm; }
    // why: the fused layer loop forks the attn vs GDN pre/mix/post trio per
    // layer, exactly as spec_verify_forward does.
    bool is_attn_layer(int il) const { return attn_layer[il]; }
    // why: the fused head must pick the same output tensor (Q4 fast head vs
    // fp16) the solo verify picks, or fused logits fork numerically.
    bool fast_head_on() const { return fast_head; }
    // why: the union round borrows es[0]'s k_vgemm workspace (A9: sized from
    // vgemm_ws_bytes_model for W_PLUMB lanes, so any engine's covers any
    // legal union).
    float* vgemm_ws() const { return d_vgemm_ws; }
    // why: build_union_view asserts the granted width was installed
    // (set_round_width) before the member's mix/tails read it.
    int round_width() const { return vw; }
    // why: the conductor records each member's draft_done event on the
    // stream its draft phase ran on, then makes the fused stream wait on it.
    cudaStream_t stream() const { return stm; }
    // why: the conductor's interleaved draft loop (P2a) gates each member's
    // margin reads against ITS OWN theta (draft_and_gate's pmin_theta
    // compare), per-member because Q27_PMIN is per-engine config.
    float gate_theta() const { return pmin_theta; }
    // why: the conductor D2Hs every member's round outcome itself -- one
    // sync for the whole batch -- then hands the host copy to
    // commit_outcome().
    const int* outcome_dev() const { return d_outcome; }
    // why: leave() hands the finish reason (stamped by finish_decode) to
    // the request thread via TokenQueue::close.
    const char* end_reason() const { return gs.end; }
    // why: fused rounds have no spec_round to stamp the Q27_PHASE_STATS
    // walls, so the conductor stamps them from its cstm event brackets
    // (SHARED-WALL semantics; fused_round's accumulation comment) -- gated
    // behind the SAME env latch the solo stamps use.
    bool phase_stats_on() const { return phase_stats; }
    // why: the phase fields stay engine-owned (GenStats); the conductor adds
    // its per-round walls + this member's launched steps through one named
    // mutator instead of reaching into gs (A4).
    void phase_stats_add(double draft_ms, double verify_ms, long steps) {
        gs.draft_ms += draft_ms;
        gs.verify_ms += verify_ms;
        gs.draft_steps += steps;
    }
    // ---- P3 T3 exec-cache accessors (A4; plan 2026-07-16-batch-p3-capture) --
    // why: the conductor's graph-cache shape key includes each member's
    // KV-cache kind -- attn_mix's kv_store/attn_decode kernel family
    // branches on it at capture time. Init-fixed per engine, keyed anyway:
    // a recycled Engine* address must never hit a differently-configured
    // engine's cached exec.
    int kv_cache_kind() const { return kv_kind; }
    // why: the T3 ALWAYS-ON hit guard re-derives the device state a cached
    // exec's table twins consume -- the GDN role-table base, the perm
    // scalar the twins dereference, and the pinned staging int
    // stage_perm_async copies from -- and compares each against the
    // capture-stored snapshot (B8 discipline) before any replay.
    float* const* gdn_role_tab() const { return d_gdn_tab; }
    const int* perm_scalar_dev() const { return d_perm_scalar; }
    const int* perm_pin_host() const { return h_perm_pin; }
    // why: the guard's staging-expectation check -- stage_perm_async(cstm)
    // must already have run this round, so the pinned int must carry the
    // CURRENT perm; a stale value means the copy the twins depend on was
    // never staged and a replay would consume last round's rotation.
    int cur_perm() const { return perm; }
    //
    // Set the granted verify width for the NEXT (eager, fused) round. vw is
    // capture-time state for the graph zoo, so this must only be called on
    // the conductor path, never between graph replays.
    void set_round_width(int w) {
        assert(w >= 2 && w <= W_MAX);
        vw = w;
    }
    // P2a step-granular draft entrypoints: draft_and_gate's margin loop,
    // exploded so the conductor can interleave step k across all gated
    // members (each engine's chain stays on its OWN stm; the conductor syncs
    // and reads margins between steps). draft_and_gate below is EXACTLY these
    // pieces reassembled -- the solo/smoke paths run the same code, so the
    // canonical/sampled-seed gates also gate the extraction.
    //
    // why: the conductor bootstraps a sampled member's first token once
    // before its step 0 (draft_and_gate's samp_first block, verbatim).
    void draft_sample_bootstrap() {
        if (!samp_first) return;
        samp_first = false;
        q27k::sample_g(logits, VOCAB, d_samp, d_nuc, d_pos, 0, d_token, d_amax, stm);
    }
    // why: the conductor hoists each member's drafting ceiling (the
    // dctl/gate_maxd read at draft_and_gate's top) before its interleaved
    // loop; also carries draft_and_gate's gated-config precondition.
    int draft_md_used(bool sampled) const {
        assert(pmin_theta > 0.f && dexit_on && !tool_split_active);
        return sampled ? 4 : (maxd_auto ? dctl.cur : gate_maxd);
    }
    // why: the conductor launches step k on EVERY active member's stm before
    // syncing any of them -- graph launch + margin D2H only, deliberately NO
    // sync (that is the whole overlap).
    void draft_step_launch(int k) {
        CUDA_CHECK(cudaGraphLaunch(draft_step_graph[k][perm], stm));
        CUDA_CHECK(cudaMemcpyAsync(h_draft_margin + k, d_draft_margin + k, 4,
                                   cudaMemcpyDeviceToHost, stm));
    }
    // why: the conductor reads step k's margin after ITS stream sync (the
    // host value is garbage until the caller synced stm past the D2H above).
    float draft_margin(int k) const { return h_draft_margin[k]; }
    // P2c: the margin-D2H half of draft_step_launch, stream-parameterized --
    // fused draft steps run on the conductor stream, so their margins land
    // via cstm (the caller syncs cstm before reading draft_margin(k)).
    void draft_margin_d2h(int k, cudaStream_t st) {
        CUDA_CHECK(cudaMemcpyAsync(h_draft_margin + k, d_draft_margin + k, 4,
                                   cudaMemcpyDeviceToHost, st));
    }
    // why: the conductor fires draft_and_gate's width-floor top-up when a
    // member leaves the interleaved loop. A width-W verify walks W-1 drafts,
    // so W draft rows must exist; the range is empty except at cap==0
    // (launched == cap+1 >= W otherwise -- see draft_and_gate).
    void draft_floor_topup(int launched, int W, int md_used) {
        for (int k = launched; k < W && k < md_used; k++)
            CUDA_CHECK(cudaGraphLaunch(draft_step_graph[k][perm], stm));
    }
    // Draft phase of one GATED round on THIS engine's stm: the P14 dexit
    // margin loop of spec_round verbatim (per-step draft graphs, D2H margin,
    // stop at first sub-theta), including the width-floor top-up. Returns the
    // WANT width (cap+1, floored 2) -- the conductor trims the union, calls
    // set_round_width(granted), then the fused verify. P1 scope: requires the
    // gated dexit config (pmin_theta > 0, dexit_on, no tool split).
    // sampled=true mirrors spec_sample_round's gated dexit branch instead:
    // first-token bootstrap from the retained prefill logits (samp_first) and
    // the FIXED sampled ceiling 4 (the sampled tail is 4-draft this phase),
    // so a sampled member's fused round consumes the identical drafts +
    // Philox keys its solo round would.
    // out_cap/out_md (Task 9): this round's margin-run depth and drafting
    // ceiling, for commit_outcome's telemetry/depthctl mirror -- W alone is
    // ambiguous at the floor (cap 0 and cap 1 both return W=2).
    int draft_and_gate(bool sampled = false, int* out_cap = nullptr, int* out_md = nullptr) {
        if (sampled) draft_sample_bootstrap();
        const int md_used = draft_md_used(sampled); // asserts the gated config
        int cap = 0, launched = 0;
        for (int k = 0; k < md_used; k++) {
            draft_step_launch(k);
            launched++;
            CUDA_CHECK(cudaStreamSynchronize(stm));
            if (draft_margin(k) < pmin_theta) break;
            cap++;
        }
        int W = cap + 1 < 2 ? 2 : cap + 1; // no width-1 gemv; floor at 2
        // Width-floor top-up (see spec_round): a width-W verify walks W-1
        // drafts, so W draft rows must exist. Only fires at cap==0.
        draft_floor_topup(launched, W, md_used);
        if (out_cap) *out_cap = cap;
        if (out_md) *out_md = md_used;
        return W;
    }
    // Suffix branch of one GREEDY round on THIS engine's stm: the fire test +
    // prep_round + draft-lane H2D staging of spec_round's suffix branch,
    // verbatim, WITHOUT the verify graph launch (the fused verify replaces
    // it). Returns the suffix verify width (sfx_width()) when the committed
    // stream's suffix recurs (match >= sfx_L), else 0 -- the conductor then
    // falls back to draft_and_gate(). Greedy-only, like the solo branch
    // (spec_sample_round has no suffix path). Trim may grant less than the
    // returned width: the verify then walks granted-1 of the staged drafts,
    // and the extra staged lanes are never read (same contract as the gated
    // rounds' unread draft rows). Suffix rounds skip the MTP chain, so the
    // stale-MTP-KV note on suffix_on applies to fused rounds identically.
    int suffix_propose() {
        if (!(!tool_split_active && suffix_on && sfx_valid && pmin_theta > 0.f &&
              h_mask_id0 < 0 &&
              sfx.propose_with(last_pending, sfx_width() - 1, h_sfx_prop) >= sfx_L))
            return 0;
        q27k::prep_round(d_P, d_token, lane_pos(), mtp_pos(), W_MAX, D_MAX_MTP, d_outcome,
                         stm);
        const int sw = sfx_width();
        for (int k = 0; k < sw - 1; k++)
            CUDA_CHECK(cudaMemcpyAsync(d_draft_L[k], &h_sfx_prop[k], 4,
                                       cudaMemcpyHostToDevice, stm));
        return sw;
    }
    // Post-verify host commit for one fused round: EXACTLY what the solo
    // round does after its outcome sync. Greedy (spec_round): em[]
    // extraction, last_pending from oc[OUTCOME_INTS-1], suffix arming, perm
    // advance, PLUS the telemetry/adaptive-depth block -- suffix counters,
    // gate_cap/gate_n/lane histograms, and the dctl ladder update (Task 9:
    // Q27_MAXD=auto members MUST feed dctl exactly like spec_round or the
    // adaptive ceiling drifts between solo and fused serving). Sampled
    // (spec_sample_round): the sampled outcome layout differs -- pending at
    // oc[6], no suffix arming, and NO dctl/histogram updates (the sampled
    // ceiling is fixed at 4; spec_sample_round updates nothing either).
    // oc = this engine's d_outcome, already on host (the conductor does one
    // D2H + sync per round for the whole batch). gate_cap/md_used come from
    // draft_and_gate's out-params (-1 = suffix/none, skips the gated block).
    // TRIM CLAMP (fused-only divergence, documented): under a trimmed grant,
    // lanes past vw-1 were never verified, so cap is clamped to vw-1 for the
    // histograms and for dctl -- `cap >= md` (the fired test) then stays
    // false and the round contributes no spurious demote evidence for a lane
    // that never ran. Untrimmed rounds (granted == want) have cap <= vw-1
    // already, so solo-equivalent traffic is bitwise-identical bookkeeping.
    int commit_outcome(const int* oc, int* emit, bool sampled = false,
                       bool sfx_round = false, int gate_cap = -1, int md_used = -1) {
        int n = oc[0];
        if (!sampled) {
            if (sfx_round) {
                sfx_fired++;
                sfx_tok += n;
                // suffix-round ceiling-saturation evidence (see spec_round's
                // maxd_auto suffix arm for the full rationale)
                if (maxd_auto && n >= dctl.cur + 1) dctl.update(dctl.cur, dctl.cur, n);
            } else if (gate_cap >= 0) {
                int cap = gate_cap < vw - 1 ? gate_cap : vw - 1; // trim clamp
                gate_cap_hist[cap]++;
                gate_n_hist[n]++;
                for (int j = 1; j <= cap; j++) {
                    gate_lane_fired[j]++;
                    if (n >= j + 1) gate_lane_acc[j]++;
                }
                if (maxd_auto) dctl.update(md_used, cap, n);
            }
        }
        for (int k = 0; k < n; k++) emit[k] = oc[1 + k];
        if (sampled) {
            last_pending = oc[6]; // sampled outcome: {n, t1..t5, pending}
        } else {
            last_pending = oc[OUTCOME_INTS - 1];
            sfx_valid = true;
            if (sfx_dbg) {
                fprintf(stderr, "[sfxdbg-oc] n=%d em=", n);
                for (int k = 0; k < n; k++) fprintf(stderr, "%d,", oc[1 + k]);
                fprintf(stderr, " pend=%d sfx_round=%d\n", oc[OUTCOME_INTS - 1],
                        sfx_round ? 1 : 0);
            }
        }
        perm = (perm + (n - 1)) % W_MAX;
        return n;
    }

    // Sampled decode round (temp>0): produces exactly one token, so it plugs
    // into generate()'s decode loop in place of spec_round (n=1), reusing its
    // ctx-guard / eos / on_token / round-gap logic. The first call samples from
    // the retained prefill logits (kind 0, no forward -- the last prompt token's
    // GDN update already ran; re-forwarding would double-apply it); later calls
    // replay sample_graph, which forwards the just-emitted token and samples the
    // next. No MTP, no spec: correctness-first per the design (Phase 2 adds spec
    // rejection sampling for speed).
    int sample_round(int* emit) {
        if (samp_first) {
            samp_first = false;
            q27k::sample_g(logits, VOCAB, d_samp, d_nuc, d_pos, 0, d_token, d_amax, stm);
        } else {
            CUDA_CHECK(cudaGraphLaunch(sample_graph, stm));
        }
        int tok;
        CUDA_CHECK(cudaMemcpyAsync(&tok, d_token, 4, cudaMemcpyDeviceToHost, stm));
        CUDA_CHECK(cudaStreamSynchronize(stm));
        emit[0] = tok;
        return 1;
    }

    // Phase 2 sampled spec round (temp>0): the sampled 2nd graph set restores
    // spec speed under sampling. Like spec_round, but the pending token is
    // resampled (not argmax'd) and the accept chain is rejection sampling. The
    // first call samples the pending from the retained prefill logits (kind 0,
    // no forward) -- symmetric with the greedy bootstrap (step_with's argmax);
    // h_next is the prefill hidden. Tools are off under sampling, so no split path.
    int spec_sample_round(int* emit) {
        // First token from the retained prefill logits (kind 0, no forward) --
        // BEFORE any spec round, on both the gated and ungated branches, so the
        // gated/ungated first token is identical at a given seed.
        if (samp_first) {
            samp_first = false;
            q27k::sample_g(logits, VOCAB, d_samp, d_nuc, d_pos, 0, d_token, d_amax, stm);
        }
        if (pmin_theta > 0.f) {
            // P14 confidence-gated sampled round -- mirror spec_round's gated
            // branch. The sampled tail is 4-draft this phase, so ALWAYS draft
            // depth-4: draft_graph[perm] is depth-4 when gate_maxd==4; under
            // gate_maxd==5 (auto or fixed) draft_graph_lo[perm] is the depth-4
            // draft (captured whenever gate_maxd==5). Cap the accept walk at 4.
            // Tools are off under sampling, so no split path.
            const int md_used = 4; // sampled ceiling is 4; cap <= 4 by construction
            if (dexit_on) {
                // P14 draft early-exit, sampled flavor: per-step draft graphs
                // are depth-independent (steps 0..3 here), margins and caps are
                // value-identical to the monolithic depth-4 draft, and the
                // accept walk consumes the identical drafts + Philox keys -- so
                // emitted bytes and round counts match Q27_DEXIT=0 exactly.
                int cap = 0, launched = 0;
                for (int k = 0; k < md_used; k++) {
                    CUDA_CHECK(cudaGraphLaunch(draft_step_graph[k][perm], stm));
                    launched++;
                    CUDA_CHECK(cudaMemcpyAsync(h_draft_margin + k, d_draft_margin + k, 4,
                                               cudaMemcpyDeviceToHost, stm));
                    CUDA_CHECK(cudaStreamSynchronize(stm));
                    if (h_draft_margin[k] < pmin_theta) break;
                    cap++;
                }
                assert(cap <= 4);
                int W = cap + 1 < 2 ? 2 : cap + 1; // no width-1 gemv; floor at 2
                // Width-floor top-up (see spec_round): a width-W sampled verify
                // walks max_draft=W-1 drafts, so W draft rows must exist. Only
                // fires at cap==0.
                for (int k = launched; k < W && k < md_used; k++)
                    CUDA_CHECK(cudaGraphLaunch(draft_step_graph[k][perm], stm));
                CUDA_CHECK(cudaGraphLaunch(verify_sample_graph_w[W][perm], stm));
            } else {
                // Q27_DEXIT=0: monolithic depth-4 gated draft (A/B baseline).
                cudaGraphExec_t dg = (gate_maxd >= 5) ? draft_graph_lo[perm] : draft_graph[perm];
                CUDA_CHECK(cudaGraphLaunch(dg, stm));
                CUDA_CHECK(cudaMemcpyAsync(h_draft_margin, d_draft_margin, md_used * 4,
                                           cudaMemcpyDeviceToHost, stm));
                CUDA_CHECK(cudaStreamSynchronize(stm));
                int cap = 0;
                while (cap < md_used && h_draft_margin[cap] >= pmin_theta) cap++;
                assert(cap <= 4);
                int W = cap + 1 < 2 ? 2 : cap + 1; // no width-1 gemv; floor at 2
                CUDA_CHECK(cudaGraphLaunch(verify_sample_graph_w[W][perm], stm));
            }
            // P13 EMA (sat/yield) is NOT updated from sampled rounds this phase
            // (sampled ceiling is fixed at 4); adaptive-maxd applies to greedy only.
        } else {
            CUDA_CHECK(cudaGraphLaunch(spec_sample_graph[perm], stm));
        }
        int oc[7];
        CUDA_CHECK(cudaMemcpyAsync(oc, d_outcome, 28, cudaMemcpyDeviceToHost, stm));
        CUDA_CHECK(cudaStreamSynchronize(stm));
        int n = oc[0];
        for (int k = 0; k < n; k++) emit[k] = oc[1 + k];
        last_pending = oc[6];
        perm = (perm + (n - 1)) % W_MAX;
        return n;
    }
    int last_pending = -1;
    // P11: called mid-round in the constrained path with the 4 draft tokens;
    // the host advances a grammar copy over [pending, d1..d4] and stages the
    // 5 lane masks + cap=0. Null -> capped path (or unconstrained).
    std::function<void(const int*)> on_drafts;
    // P7: called after each spec round with the NEW pending token so the
    // host grammar can stage next round's slot-0 mask (state must include
    // the pending token -- masks lag one token otherwise).
    std::function<void(int)> on_pending;
    // P15: called after each greedy spec round with the round's emitted
    // tokens BEFORE anything is emitted or committed host-side. Returns -1
    // for no action, or m in [1..n] when the <tool_call> marker completed at
    // em[m-1]: generate() truncates the round to m tokens and refinishes so
    // the first post-marker decision onward is grammar-masked (the engage-lag
    // fix). Server-only, greedy-only; CLI never sets it.
    std::function<int(const int*, int)> on_round;
    // R1b: optional preemption hook, called between decode rounds and
    // between prefill chunks -- the two boundaries where this engine's
    // device state is coherent. Returns true if the GPU was actually
    // handed to another request. The server wires it to
    // GpuGate::maybe_yield behind a drain of THIS engine's stream; the
    // CLI never sets it, so the canonical paths execute no new code.
    std::function<bool()> on_round_gap;

    // ---- P7 constrained tool decoding API ----
    // Upload a vocab bitmask into the resident pool; returns its stable
    // index (-1 if the pool is full -- caller falls back to unconstrained).
    int mask_pool_add(const void* bits) {
        if (mask_pool_used >= MASK_POOL_CAP) return -1;
        CUDA_CHECK(cudaMemcpyAsync(d_mask_pool + (size_t)mask_pool_used * mask_words, bits,
                                   (size_t)mask_words * 4, cudaMemcpyHostToDevice, stm));
        return mask_pool_used++;
    }
    // Constrain slot 0 to pool[mask_id] and cap acceptance at 1/round
    // (mask_id -1 = deactivate). Takes effect from the next spec round.
    // With on_drafts wired (P11) the round runs split and per-lane masks are
    // staged mid-round, so activate the split path instead of capping.
    void set_tool_constraint(int mask_id) {
        h_mask_id0 = mask_id;
        // P11 split path is OPT-IN (Q27_TOOL_SPLIT=1): it 4.2x's in-call decode
        // (49 -> 204 t/s) but has a FLAKY memory-corruption/race bug that faults
        // (CUDA illegal access -> exit 1) after accumulated multi-request state
        // at large ctx. Debug findings (2026-07-04): memcheck-CLEAN (so it's a
        // race / host-side / alloc-layout-dependent, NOT a plain OOB); does NOT
        // reproduce with 1-2 requests -- needs the state a real CRUSH session
        // builds (mask-pool growth + checkpoint saves + prefix-cache hits over
        // many turns). The split ALGORITHM is proven correct (split==capped
        // token-identical gate passed); this is an orchestration hazard in the
        // mid-round host readback + async mask staging, not the masking logic.
        // To resume: reproduce under CRUSH load with compute-sanitizer --tool
        // racecheck AND initcheck (memcheck won't catch it); suspect the async
        // d_mask_ids/d_mask_pool staging ordering vs the verify graph launch.
        // Default = the safe capped path until fixed.
        static const bool split_ok = getenv("Q27_TOOL_SPLIT") != nullptr;
        tool_split_active = (mask_id >= 0) && (bool)on_drafts && split_ok;
        // cap only when constraining WITHOUT the split path (P7 v1 fallback)
        h_cap0 = (mask_id >= 0 && !tool_split_active) ? 1 : 0;
        CUDA_CHECK(cudaMemcpyAsync(d_mask_ids, &h_mask_id0, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemcpyAsync(d_accept_cap, &h_cap0, 4, cudaMemcpyHostToDevice, stm));
    }
    // P11: stage all 5 lane mask ids at once (split path). cap stays 0.
    // W16: was a 12-entry {-1,...} brace list -- slots 12..15 would value-init
    // to 0, which is a VALID mask-pool id, not "unconstrained". Harmless today
    // (only 5 ints are ever copied to the device, and clear_tool_constraint
    // rewrites all W_PLUMB), but it is the wrong default to leave lying around.
    int h_mask_ids5[W_PLUMB] = {-1, -1, -1, -1, -1, -1, -1, -1,
                                -1, -1, -1, -1, -1, -1, -1, -1};
    void set_tool_masks5(const int ids[5]) {
        for (int i = 0; i < 5; i++) h_mask_ids5[i] = ids[i];
        h_cap0 = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_mask_ids, h_mask_ids5, 5 * 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemcpyAsync(d_accept_cap, &h_cap0, 4, cudaMemcpyHostToDevice, stm));
    }
    // Full constraint reset: every lane mask id back to -1 + cap 0 (covers
    // stale split-path lanes that set_tool_constraint(-1) alone would miss).
    void clear_tool_constraint() {
        h_mask_id0 = -1;
        h_cap0 = 0;
        tool_split_active = false;
        for (int i = 0; i < W_PLUMB; i++) h_mask_ids5[i] = -1;
        CUDA_CHECK(cudaMemcpyAsync(d_mask_ids, h_mask_ids5, W_MAX * 4, cudaMemcpyHostToDevice,
                                   stm));
        CUDA_CHECK(cudaMemcpyAsync(d_accept_cap, &h_cap0, 4, cudaMemcpyHostToDevice, stm));
    }
    // P15 engage-lag fix: rewind the JUST-FINISHED greedy spec round from n
    // accepted tokens to m (1 <= m <= n) and re-decide the pending token under
    // the freshly staged slot-0 mask. Everything needed is still resident:
    // per-lane GDN states / conv rings sit in the rotating role buffers (the
    // round "commits" by advancing perm, never by copying -- state-after-lane-
    // (m-1) is old role m-1), lane hiddens in x1..x1_L[5], lane logits in
    // logits2. KV/MTP rows past the kept position are rewritten by the next
    // round. Must be called BETWEEN rounds (server on_round hook), with the
    // engage mask already staged on this stream so the re-argmax orders after
    // it. Returns the new pending token. The CLI/canonical path never sets
    // on_round, so this code is unreachable there.
    int refinish_round(int m, int n, int P_target) {
        perm = (perm + (m - n) + W_MAX) % W_MAX;
        CUDA_CHECK(cudaMemcpyAsync(d_P, &P_target, 4, cudaMemcpyHostToDevice, stm));
        // W16: this was a hand-written 12-entry brace list into a W_PLUMB array
        // -- at W_PLUMB=16 it left lanes[12..15] = nullptr, and a wide suffix
        // round that commits n >= 13 with a tool marker completing there would
        // memcpy from nullptr and take the server down. x1_L[0] aliases x1, so
        // the whole list is just x1_L.
        CUDA_CHECK(cudaMemcpyAsync(h_next, x1_L[m - 1], (size_t)N_EMBD * 4,
                                   cudaMemcpyDeviceToDevice, stm));
        q27k::argmax_masked(logits2 + (size_t)(m - 1) * VOCAB, VOCAB, d_mask_pool, mask_words,
                            d_mask_ids, 0, d_token, d_amax, stm);
        int new_pending = -1;
        CUDA_CHECK(cudaMemcpyAsync(&new_pending, d_token, 4, cudaMemcpyDeviceToHost, stm));
        CUDA_CHECK(cudaStreamSynchronize(stm));
        return new_pending;
    }

    // ---- batched prefill (M6): T-token chunk versions of the blocks ----
    void qxT(const float* x, int cols, int T) {
        q27k::quantize_x(x, (int64_t)T * cols, xqT, stm);
        // g64 requant for the MMA GEMM (unconditional: the kernel is noise
        // next to the GEMMs and keeping nat64 always-fresh means every
        // dispatch choice downstream is safe)
        q27k::quantize_x_g64(x, (int64_t)T * cols, xqT, stm);
    }
    void mmT(const DevTensor& w, const float* xT, float* yout, int T) {
        switch (w.dtype) {
            case DType::Q4_G64:
                q27k::gemm_q4_T((const uint8_t*)w.data, (const __half*)w.scales, xqT, yout,
                                w.rows, w.cols, T, stm);
                break;
            case DType::Q8_G128:
                q27k::gemm_q8_T((const int8_t*)w.data, (const __half*)w.scales, xqT, yout,
                                w.rows, w.cols, T, stm);
                break;
            case DType::F16:
                q27k::gemm_f16_T((const __half*)w.data, xT, yout, w.rows, w.cols, T, stm);
                break;
            default:
                fprintf(stderr, "mmT: unsupported dtype\n");
                exit(1);
        }
    }

    void gdn_block_T(int il, int T) {
        qxT(x1T, N_EMBD, T);
        mmT(T2(il, "attn_qkv.weight"), x1T, qkvT, T);
        mmT(T2(il, "attn_gate.weight"), x1T, zT, T);
        mmT(T2(il, "ssm_alpha.weight"), x1T, alphaT, T);
        mmT(T2(il, "ssm_beta.weight"), x1T, betarT, T);
        q27k::gdn_gates_T(alphaT, betarT, (const float*)T2(il, "ssm_a").data,
                          (const float*)T2(il, "ssm_dt.bias").data, gT, betaT, GDN_HEADS, T, stm);
        q27k::conv_prefill_T(conv_ring[il], qkvT,
                             (const float*)T2(il, "ssm_conv1d.weight").data, convT, GDN_CH, T,
                             stm);
        q27k::l2norm_heads_T(convT, 16, GDN_DIM, GDN_CH, T, EPS, stm);
        q27k::l2norm_heads_T(convT + 2048, 16, GDN_DIM, GDN_CH, T, EPS, stm);
        q27k::delta_scan_T(S[il], convT, gT, betaT, oT, T, stm, &wy_scratch);
        q27k::gated_norm_gdn_T(oT, (const float*)T2(il, "ssm_norm.weight").data, zT, ogT,
                               GDN_HEADS, GDN_DIM, T, EPS, stm);
        qxT(ogT, GDN_V, T);
        mmT(T2(il, "ssm_out.weight"), ogT, yT, T);
    }

    void attn_block_T(int il, int base, int T, void* kc, void* vc) {
        const int QROW = N_HEAD * 2 * HEAD_DIM, KVROW = N_KV * HEAD_DIM;
        qxT(x1T, N_EMBD, T);
        mmT(T2(il, "attn_q.weight"), x1T, qgT, T);
        q27k::rmsnorm_heads_T(qgT, (const float*)T2(il, "attn_q_norm.weight").data, qgT, N_HEAD,
                              HEAD_DIM, 2 * HEAD_DIM, QROW, T, EPS, stm);
        mmT(T2(il, "attn_k.weight"), x1T, kT, T);
        q27k::rmsnorm_heads_T(kT, (const float*)T2(il, "attn_k_norm.weight").data, kT, N_KV,
                              HEAD_DIM, HEAD_DIM, KVROW, T, EPS, stm);
        mmT(T2(il, "attn_v.weight"), x1T, vT, T);
        q27k::rope_neox_T(qgT, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, QROW, base, T, FREQ_BASE,
                          stm);
        q27k::rope_neox_T(kT, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, KVROW, base, T, FREQ_BASE, stm);
        // turbo3 (phase 2): same rotation contract as decode -- forward-WHT
        // on Q post-rope (K's rotation folds into the store), inverse-WHT on
        // the pooled output BEFORE the sigmoid gate. turbo3v: K plain fp16,
        // Q unrotated.
        if (kv_kind == KV_T3)
            q27k::wht_T(qgT, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, QROW, T, false, stm);
        if (kv_kind >= KV_T3)
            q27k::kv_store_T_t3(kT, vT, kc, vc, base, N_KV, HEAD_DIM, T, stm,
                                /*k_plain=*/kv_kind == KV_T3V);
        else
            q27k::kv_store_T(kT, vT, kc, vc, base, KVROW, T, stm, kv_fp8);
        q27k::attn_prefill_T(qgT, 2 * HEAD_DIM, QROW, kc, vc, attnT, N_HEAD * HEAD_DIM, pf_part,
                             base, 0, T, N_HEAD, N_KV, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM),
                             stm, kv_kind);
        if (kv_kind >= KV_T3)
            q27k::wht_T(attnT, N_HEAD, HEAD_DIM, HEAD_DIM, N_HEAD * HEAD_DIM, T, true, stm);
        q27k::sigmoid_gate_mul_T(attnT, qgT, N_HEAD, HEAD_DIM, T, stm);
        qxT(attnT, N_HEAD * HEAD_DIM, T);
        mmT(T2(il, "attn_output.weight"), attnT, yT, T);
    }

    void ffn_T(int il, int T) {
        qxT(x1T, N_EMBD, T);
        mmT(T2(il, "ffn_gate.weight"), x1T, ffnGT, T);
        mmT(T2(il, "ffn_up.weight"), x1T, ffnUT, T);
        q27k::silu_mul(ffnGT, ffnUT, ffnGT, (int64_t)T * N_FFN, stm);
        qxT(ffnGT, N_FFN, T);
        mmT(T2(il, "ffn_down.weight"), ffnGT, yT, T);
    }

    // Forward a chunk of T prompt tokens starting at absolute position `base`.
    // Leaves hT = final residual for each token. Updates conv rings, GDN state,
    // attention KV caches in place.
    void prefill_chunk(const int* d_toks, int base, int T) {
        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed_rows_q8_T((const int8_t*)emb.data, (const __half*)emb.scales, d_toks,
                              N_EMBD, T, hT, stm);
        for (int il = 0; il < N_LAYER; il++) {
            q27k::rmsnorm_T(hT, (const float*)T2(il, "attn_norm.weight").data, x1T, N_EMBD, T,
                            EPS, stm);
            if (attn_layer[il]) {
                int ci = attn_cache_idx[il];
                attn_block_T(il, base, T, kcache[ci], vcache[ci]);
            } else {
                gdn_block_T(il, T);
            }
            q27k::add_inplace(hT, yT, (int64_t)T * N_EMBD, stm);
            q27k::rmsnorm_T(hT, (const float*)T2(il, "post_attention_norm.weight").data, x1T,
                            N_EMBD, T, EPS, stm);
            ffn_T(il, T);
            q27k::add_inplace(hT, yT, (int64_t)T * N_EMBD, stm);
        }
    }

    // Warm the MTP KV cache for pairs (h(base+t), token[base+t+1]) -> stored at
    // position base+t+1. Only the K/V projections matter for warming; the MTP
    // attention/FFN outputs were always discarded here, so they are skipped.
    void mtp_warm_T(const int* d_toks_next, int base, int T) {
        const int il = 64;
        const DevTensor& emb = dm.get("token_embd.weight");
        const int KVROW = N_KV * HEAD_DIM;
        // x1T currently holds output_norm(hT) (set by caller)
        q27k::embed_rows_q8_T((const int8_t*)emb.data, (const __half*)emb.scales, d_toks_next,
                              N_EMBD, T, embT, stm);
        q27k::rmsnorm_T(embT, (const float*)T2(il, "nextn.enorm.weight").data, ehnT, N_EMBD, T,
                        EPS, stm, N_EMBD, 2 * N_EMBD);
        q27k::rmsnorm_T(x1T, (const float*)T2(il, "nextn.hnorm.weight").data, ehnT + N_EMBD,
                        N_EMBD, T, EPS, stm, N_EMBD, 2 * N_EMBD);
        qxT(ehnT, 2 * N_EMBD, T);
        mmT(T2(il, "nextn.eh_proj.weight"), ehnT, xmtpT, T);
        q27k::rmsnorm_T(xmtpT, (const float*)T2(il, "attn_norm.weight").data, x1T, N_EMBD, T,
                        EPS, stm);
        qxT(x1T, N_EMBD, T);
        mmT(T2(il, "attn_k.weight"), x1T, kT, T);
        q27k::rmsnorm_heads_T(kT, (const float*)T2(il, "attn_k_norm.weight").data, kT, N_KV,
                              HEAD_DIM, HEAD_DIM, KVROW, T, EPS, stm);
        mmT(T2(il, "attn_v.weight"), x1T, vT, T);
        q27k::rope_neox_T(kT, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, KVROW, base + 1, T, FREQ_BASE,
                          stm);
        if (kv_kind >= KV_T3)
            q27k::kv_store_T_t3(kT, vT, mtp_k, mtp_v, base + 1, N_KV, HEAD_DIM, T, stm,
                                /*k_plain=*/kv_kind == KV_T3V);
        else
            q27k::kv_store_T(kT, vT, mtp_k, mtp_v, base + 1, KVROW, T, stm, kv_fp8);
    }

    // ---- P9: same-session GDN checkpoint ring (host pinned RAM) ----
    // Dropped every ckpt_interval tokens at prefill chunk boundaries; on a
    // stable-snapshot miss, generation restores the nearest checkpoint <=
    // the first divergence point instead of a full cold prefill. GDN-only:
    // divergence at m means tokens [0,m) are unchanged, so the append-only
    // attention/MTP KV rows below m stay valid in place. Cleared on reset()
    // (new session -> old KV rows no longer describe this conversation).
    struct Ckpt {
        std::vector<int> toks;
        float* buf = nullptr; // pinned: per GDN layer, S then conv ring
    };
    std::vector<Ckpt> ckpts;
    int ckpt_interval = 4096, ckpt_slots = 16, ckpt_next = 0;

    size_t ckpt_layer_floats() const {
        return (size_t)GDN_HEADS * GDN_DIM * GDN_DIM + 3 * GDN_CH;
    }
    void ckpt_clear() {
        for (auto& c : ckpts) c.toks.clear();
        ckpt_next = 0;
    }
    void ckpt_save(const std::vector<int>& prompt, int len) {
        if (ckpt_interval <= 0) return;
        if ((int)ckpts.size() < ckpt_slots) ckpts.resize(ckpt_slots);
        Ckpt& c = ckpts[ckpt_next];
        ckpt_next = (ckpt_next + 1) % ckpt_slots;
        size_t lf = ckpt_layer_floats();
        if (!c.buf) {
            size_t total = 0;
            for (int il = 0; il < N_LAYER; il++)
                if (!attn_layer[il]) total += lf;
            CUDA_CHECK(cudaMallocHost((void**)&c.buf, total * 4));
        }
        float* h = c.buf;
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                CUDA_CHECK(cudaMemcpyAsync(h, S[il],
                                           (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4,
                                           cudaMemcpyDeviceToHost, stm));
                h += (size_t)GDN_HEADS * GDN_DIM * GDN_DIM;
                CUDA_CHECK(cudaMemcpyAsync(h, conv_ring[il], 3 * GDN_CH * 4,
                                           cudaMemcpyDeviceToHost, stm));
                h += 3 * GDN_CH;
            }
        c.toks.assign(prompt.begin(), prompt.begin() + len);
    }
    // largest checkpoint whose covered prefix matches the new prompt
    int ckpt_best(const std::vector<int>& prompt) const {
        int best = -1;
        size_t best_len = 0;
        for (size_t k = 0; k < ckpts.size(); k++) {
            const auto& c = ckpts[k];
            // c.toks.size()+1 > prompt.size(), NOT c.toks.size() > prompt.size()-1:
            // the latter underflows size_t on an empty prompt (0-1 = SIZE_MAX), so
            // nothing is skipped and std::equal below reads past the empty vector's
            // begin() (nullptr) -> crash. reuse_len() runs this at slot-selection,
            // BEFORE the engine-entry NP>=1 guard, so the empty prompt reaches here.
            if (!c.buf || c.toks.empty() || c.toks.size() + 1 > prompt.size()) continue;
            if (c.toks.size() <= best_len) continue;
            if (std::equal(c.toks.begin(), c.toks.end(), prompt.begin())) {
                best = (int)k;
                best_len = c.toks.size();
            }
        }
        return best;
    }
    void ckpt_restore(int k) {
        CUDA_CHECK(cudaStreamSynchronize(stm)); // pending D2H saves must land
        const float* h = ckpts[k].buf;
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                CUDA_CHECK(cudaMemcpyAsync(S[il], h,
                                           (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4,
                                           cudaMemcpyHostToDevice, stm));
                h += (size_t)GDN_HEADS * GDN_DIM * GDN_DIM;
                CUDA_CHECK(cudaMemcpyAsync(conv_ring[il], h, 3 * GDN_CH * 4,
                                           cudaMemcpyHostToDevice, stm));
                h += 3 * GDN_CH;
            }
        perm = 0;
        CUDA_CHECK(cudaMemset(d_P, 0, 4));
    }

    // Router-facing reuse probe: length of the prompt prefix this engine can
    // restore without re-prefill -- the stable snapshot when the prompt
    // STRICTLY extends it (same predicate as generate(): have_snap and
    // snap_toks.size() <= NP-1), else the best P9 checkpoint whose covered
    // prefix matches. 0 = nothing reusable. Must stay in lockstep with
    // generate()'s hit logic; the multi-slot server routes on this.
    int reuse_len(const std::vector<int>& prompt) const {
        int NP = (int)prompt.size();
        if (have_snap && (int)snap_toks.size() <= NP - 1 &&
            std::equal(snap_toks.begin(), snap_toks.end(), prompt.begin()))
            return (int)snap_toks.size();
        int ck = ckpt_best(prompt);
        return ck >= 0 ? (int)ckpts[ck].toks.size() : 0;
    }
    // True when this engine holds no conversation state worth preserving
    // (fresh slot, or last request took the serial path which clears both).
    bool cache_empty() const {
        if (have_snap || !snap_toks.empty()) return false;
        for (auto& c : ckpts)
            if (!c.toks.empty()) return false;
        return true;
    }

    void snap_save(const std::vector<int>& prompt, int upto = -1) {
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                CUDA_CHECK(cudaMemcpyAsync(S_snap[il], S[il],
                                           (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4,
                                           cudaMemcpyDeviceToDevice, stm));
                CUDA_CHECK(cudaMemcpyAsync(ring_snap[il], conv_ring[il], 3 * GDN_CH * 4,
                                           cudaMemcpyDeviceToDevice, stm));
            }
        if (upto < 0) upto = (int)prompt.size() - 1;
        snap_toks.assign(prompt.begin(), prompt.begin() + upto);
        have_snap = true;
    }

    void snap_restore() {
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                CUDA_CHECK(cudaMemcpyAsync(S[il], S_snap[il],
                                           (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4,
                                           cudaMemcpyDeviceToDevice, stm));
                CUDA_CHECK(cudaMemcpyAsync(conv_ring[il], ring_snap[il], 3 * GDN_CH * 4,
                                           cudaMemcpyDeviceToDevice, stm));
            }
        perm = 0;
        CUDA_CHECK(cudaMemset(d_P, 0, 4));
    }

    // Reset all decode state for a fresh request (positions, GDN recurrent state,
    // conv rings, MTP KV). Weight buffers and captured graphs are unaffected.
    void reset() {
        CUDA_CHECK(cudaMemset(d_pos, 0, 4));
        CUDA_CHECK(cudaMemset(d_step, 0, 4));
        CUDA_CHECK(cudaMemset(d_P, 0, 4));
        perm = 0;
        // width-12 fix-in-passing: this memset used to stop at spare 5 (a
        // stale P12b-era list -- benign only because every role is written
        // before it is read within a round). All 11 spare sets now reset,
        // matching build_spec_graphs' reset_gdn_mtp.
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                size_t sb = (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4;
                CUDA_CHECK(cudaMemset(S[il], 0, sb));
                CUDA_CHECK(cudaMemset(conv_ring[il], 0, 3 * GDN_CH * 4));
                for (int r = 0; r < W_PLUMB - 1; r++)
                    if (S_sp[r][il]) {
                        CUDA_CHECK(cudaMemset(S_sp[r][il], 0, sb));
                        CUDA_CHECK(cudaMemset(ring_sp[r][il], 0, 3 * GDN_CH * 4));
                    }
            }
        CUDA_CHECK(cudaMemset(mtp_k, 0, kv_bytes(false)));
        CUDA_CHECK(cudaMemset(mtp_v, 0, kv_bytes(true)));
        CUDA_CHECK(cudaStreamSynchronize(stm));
    }

    // R0 telemetry: filled by every generate() call, read by the server's
    // [req] log line. Host-side bookkeeping only -- no device work, no syncs.
    struct GenStats {
        int prompt = 0, hit = 0, ckpt = -1, pf = 0; // tokens
        double pf_ms = 0, dec_ms = 0, cb_ms = 0;    // cb = time inside on_token
        // (review L3) under Q27_BATCH=1 the sink is the conductor's queue
        // push: cb_ms then times on_emit + TokenQueue::push on the CONDUCTOR
        // thread, not the client SSE write (which runs on the request thread
        // draining the queue and is not timed anywhere).
        // R1b: time parked in on_round_gap calls that actually handed the
        // GPU over (includes the pre-yield drain of our own in-flight
        // chunks), and how many handovers. pf_ms/dec_ms stay wall-inclusive
        // of yields; analyzers subtract gw_ms for GPU-busy accounting.
        double gw_ms = 0;
        // Q27_PHASE_STATS: summed gated-round draft/verify wall (ms) and MTP
        // draft steps launched, this request. dec_ms - draft_ms - verify_ms =
        // host round-gap + any unattributed (constrained/ungated) rounds.
        // FUSED rounds (P2 Task 1): the conductor stamps draft_ms/verify_ms
        // from coarse cstm event brackets with SHARED-WALL semantics -- the
        // one fused-round wall is attributed IN FULL to EACH member, so
        // summing phd/phv across concurrently-batched requests DOUBLE-COUNTS
        // the wall (per-request phd/phv stays the honest "time my rounds
        // spent in phase X" read). Fused draft_ms is only the cstm-visible
        // draft TAIL: the draft phase (P2a interleave / P2c fused steps)
        // runs host-synced per step inside Conductor::draft_widths, BEFORE
        // the round's ev_round_start is recorded -- so phd brackets just the
        // unsynced launches between round start and the draft_done waits
        // (floor top-ups, suffix prep/H2D), never the draft wall itself.
        // draft_steps is NOT shared: steps THIS member launched.
        double draft_ms = 0, verify_ms = 0;
        long draft_steps = 0;
        // verify wall bucketed by verify width W=cap+1 (floored 2, <=W_MAX)
        // -- the marginal-lane cost curve that prices deep ladders with
        // draft off the critical path (Saguaro follow-up). Index by W; 0..1
        // unused. NOTE (width-12 review): buckets 9..12 stay ZERO until
        // suffix rounds get their own phase stamping (P1) -- gated rounds
        // cap at W=8 (ladder <=7) and the suffix branch is deliberately not
        // stamped; server [req] prints phwn/phwm for 2..8 only.
        double vw_ms[W_MAX + 1] = {};
        long vw_n[W_MAX + 1] = {};
        // width-12 P1: suffix-round wall + count (per request). All suffix
        // rounds run one width (sfx_width()), so sfx_ms/sfx_rounds is the
        // per-round cost at that width; server appends sfxm/sfxn.
        double sfx_ms = 0;
        long sfx_rounds = 0;
        int dec = 0, rounds = 0, yields = 0;
        const char* end = "";
    };
    GenStats gs;

    // P1 continuous batching: everything generate()'s decode loop carries
    // ACROSS rounds, gathered so the conductor can drive one round at a time
    // (decode_step) for several engines. Host bookkeeping only -- no device
    // state lives here; iteration-locals (em[], n, ...) stay in decode_step.
    struct DecodeTask {
        int n_max = 0;   // emit budget (generate()'s n_max)
        int eos = -1;    // stop token id (-1 = never fires; ids are >= 0)
        // Per-token sink; false = client-stop. In generate() this holds a
        // std::ref to the caller's callable (the generate() frame outlives
        // the loop, and the existing call sites all pass [&] lambdas whose
        // copies would alias anyway); conductor mode (Task 10) installs an
        // owning sink instead.
        std::function<bool(int)> on_token;
        int emitted = 0; // tokens delivered to on_token so far (return value)
        int rounds = 0;  // decode rounds run -> gs.rounds at finish
        int Ph = 0;      // host mirror of the last written position (ctx guard)
        std::chrono::steady_clock::time_point g0; // decode wall start (dec_ms)
        // decode-phase park baseline: dec_ms covers decode only, so the
        // contended-print suffix must use decode-phase gw deltas, not
        // request totals (prefill parks belong to pf_ms).
        double gw_pf = 0;  // gs.gw_ms at decode start
        int yields_pf = 0; // gs.yields at decode start
        bool sampling = false;           // temp>0: sampled rounds, no tc hooks
        bool force_plain_sample = false; // Q27_SAMPLE_PLAIN A/B lever
        bool prof_decode = false; // Q27_PROF_DECODE bracket open; finish closes
        // Cancellation (consensus addendum A3): the request thread sets this
        // on SSE write failure / client disconnect / shutdown. Checked at
        // ROUND BOUNDARIES ONLY (top of decode_step): a cancelled task does
        // no further GPU work but still runs finish_decode() so GenStats and
        // teardown land exactly like a natural exit. Nothing sets it yet --
        // it lands with the struct; consumers arrive with the conductor.
        std::atomic<bool> cancel{false};
        // P1 Task 10 [req] telemetry, CONDUCTOR-filled (solo generate()
        // leaves both 0): bat_members sums the round's member count k over
        // every decode round this task ran GPU work in (mean batch width =
        // bat_members / rounds); bat_r2 counts the rounds with k >= 2, i.e.
        // actually fused. Plain longs on purpose: only the conductor thread
        // writes them, and the request thread reads them after the queue
        // closes (the close is the synchronization edge).
        long bat_members = 0, bat_r2 = 0;
    };

    // R1b preemption point (no-op when the hook is unset or nobody waits).
    // Shared by the prefill chunk loops and decode_step; park time goes to
    // gs.gw_ms/gs.yields wherever it fires.
    void round_gap() {
        if (!on_round_gap) return;
        auto y0 = std::chrono::steady_clock::now();
        if (on_round_gap()) {
            gs.gw_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - y0)
                            .count();
            gs.yields++;
        }
    }

    // Pre-loop decode initialization (the code that sat between generate()'s
    // prefill epilogue and its round loop). Fills t in place: DecodeTask owns
    // a std::atomic, so it is neither copyable nor movable -- callers
    // default-construct one and hand it in. P = prompt.size()-1, the host
    // position mirror the ctx guard advances.
    template <typename F>
    void make_decode_task(DecodeTask& t, int n_max, int eos, F& on_token, int P) {
        // Sampling (temp>0): upload request params once; the sampled path
        // replaces the greedy spec_round in the loop. Default = spec_sample_round
        // (Phase 2: sampled depth-4 speculation, fast). Greedy (inv_temp<=0)
        // leaves d_samp untouched and runs the spec path bitwise. d_pos is NP
        // here (prefill's last advance), so the first eager draw keys the token
        // at position NP with kind 0.
        const bool sampling = samp.inv_temp > 0.f;
        // Q27_SAMPLE_PLAIN forces the Phase-1 plain sampler (one token/round, no
        // spec) even under sampling -- the A/B lever for the spec==non-spec
        // distribution gate (docs/sampling-design.md sec 4).
        static const bool force_plain_sample = getenv("Q27_SAMPLE_PLAIN") != nullptr;
        if (sampling) {
            CUDA_CHECK(cudaMemcpyAsync(d_samp, &samp, sizeof samp, cudaMemcpyHostToDevice, stm));
            samp_first = true;
        }
        t.n_max = n_max;
        t.eos = eos;
        t.on_token = std::ref(on_token);
        t.Ph = P;
        t.g0 = std::chrono::steady_clock::now();
        t.gw_pf = gs.gw_ms;
        t.yields_pf = gs.yields;
        t.sampling = sampling;
        t.force_plain_sample = force_plain_sample;
        // Q27_PROF_DECODE=1: bracket the decode loop with a cudaProfiler
        // range so `nsys --capture-range=cudaProfilerApi` records ONLY the
        // decode slice -- prefill otherwise floods the trace (the CLI
        // --tokens path walks the prompt serially; see BUILDLOG nsys notes).
        // No-op without a profiler attached; finish_decode() below is the
        // single exit funnel, so every decode exit path closes the range.
        t.prof_decode = getenv("Q27_PROF_DECODE") != nullptr;
        if (t.prof_decode) {
            CUDA_CHECK(cudaStreamSynchronize(stm));
            (void)cudaProfilerStart();
        }
    }

    // Decode epilogue (generate()'s old `done` lambda): closes the optional
    // profiler bracket, finalizes GenStats, prints [gen-done]. Runs EXACTLY
    // ONCE per task: from the decode_step call that returns false, or --
    // when host bookkeeping THREW mid-round under the conductor -- from the
    // A2 catch epilogue (Conductor::fail_member, conductor.h) with
    // why="error"; the two are exclusive (every finish_decode call inside
    // pre_round/post_round/decode_step is followed by a non-throwing
    // return, so a throwing round cannot have already finished).
    void finish_decode(DecodeTask& t, const char* why) {
        if (t.prof_decode) {
            CUDA_CHECK(cudaStreamSynchronize(stm));
            (void)cudaProfilerStop();
        }
        double dt =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t.g0).count();
        gs.dec = t.emitted;
        gs.dec_ms = dt * 1000.0;
        gs.rounds = t.rounds;
        gs.end = why;
        // wall-inclusive of yield parks; the suffix keeps a contended
        // print from reading as a decode regression
        if (gs.yields > t.yields_pf)
            fprintf(stderr,
                    "[gen-done] %s: %d tokens in %.1fs (%.1f t/s; parked %.0fms/%d "
                    "yields in decode), n_max=%d\n",
                    why, t.emitted, dt, t.emitted / (dt > 0 ? dt : 1), gs.gw_ms - t.gw_pf,
                    gs.yields - t.yields_pf, t.n_max);
        else
            fprintf(stderr, "[gen-done] %s: %d tokens in %.1fs (%.1f t/s), n_max=%d\n", why,
                    t.emitted, dt, t.emitted / (dt > 0 ? dt : 1), t.n_max);
        // Phase 2 acceptance-vs-temp telemetry (sampled path only; keeps the
        // greedy [gen-done] line shortbench_suite parses untouched). tokens/
        // round is the sampled spec acceptance -- it sags as temperature rises.
        if (t.sampling && !t.force_plain_sample)
            fprintf(stderr,
                    "[sample-stats] T=%.3f top_p=%.3f: %.3f tokens/round (%d tok/%d rounds)\n",
                    samp.inv_temp > 0.f ? 1.0f / samp.inv_temp : 0.f, samp.top_p,
                    t.rounds > 0 ? (double)t.emitted / t.rounds : 0.0, t.emitted, t.rounds);
    }

    // Round-boundary pre-checks (the top of generate()'s old loop body).
    // Returns false when generation is done BEFORE any GPU work -- cancel,
    // budget, ctx guard -- after running finish_decode(). Factored out of
    // decode_step (P1 Task 9) so the conductor's fused round loop runs the
    // IDENTICAL checks per member at each round boundary; decode_step calls
    // it unchanged, so the solo path is byte-identical.
    bool pre_round(DecodeTask& t) {
        // A3: cancellation is a round-boundary event ONLY -- checked here at
        // the top, before any GPU work, on the same footing (finish_decode +
        // false) as the natural exits.
        if (t.cancel.load()) {
            finish_decode(t, "cancelled");
            return false;
        }
        if (t.emitted >= t.n_max) {
            finish_decode(t, "n_max");
            return false;
        }
        // ctx guard: a round writes attention-KV rows P+1..P+gate_maxd+1 (and
        // MTP rows P+1..P+gate_maxd); launching past the reserve would write
        // beyond the caches and corrupt adjacent allocations (which the prefix
        // cache would then reuse). Stop instead -- a max-length response ends a
        // few tokens short of the absolute ceiling rather than corrupting state.
        if (t.Ph + ctx_round_reserve() > max_ctx) {
            finish_decode(t, "ctx-guard");
            return false;
        }
        return true;
    }

    // Post-round host bookkeeping (the tail of generate()'s old loop body):
    // P15 grammar scan/truncate, suffix-index append, Ph advance, per-token
    // emission (EOS / client-stop / budget), on_pending, round_gap. Returns
    // false when generation is done, after running finish_decode(). Factored
    // out of decode_step (P1 Task 9): the conductor's fused round loop runs
    // this same bookkeeping per member after commit_outcome(), so tokens,
    // stop reasons and GenStats land exactly as the solo loop produces them.
    bool post_round(DecodeTask& t, const int* em, int n) {
        t.rounds++;
        // P15 engage-lag fix: let the host grammar scan the whole round
        // pre-emission; on a mid-round <tool_call> completion, truncate to
        // the marker token and re-decide the pending under the staged mask.
        if (!t.sampling && on_round) {
            int m = on_round(em, n);
            if (m >= 1 && m <= n) {
                last_pending = refinish_round(m, n, t.Ph + m);
                n = m;
            }
        }
        // suffix index tracks the committed stream (post-truncation n);
        // the pending token rides along virtually in propose_with.
        if (suffix_on)
            for (int k = 0; k < n; k++) sfx.append(em[k]);
        t.Ph += n;
        for (int k = 0; k < n && t.emitted < t.n_max; k++) {
            if (em[k] == t.eos) {
                finish_decode(t, "eos");
                return false;
            }
            auto c0 = std::chrono::steady_clock::now();
            bool cont = t.on_token(em[k]);
            gs.cb_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - c0)
                            .count();
            if (!cont) {
                finish_decode(t, "client-stop");
                return false;
            }
            t.emitted++;
        }
        if (!t.sampling && on_pending) on_pending(last_pending);
        round_gap();
        return true;
    }

    // One decode round + its host bookkeeping (generate()'s old loop body).
    // Returns false when generation is done -- budget, ctx guard, EOS,
    // client-stop, or t.cancel -- after running finish_decode(), so GenStats
    // and [gen-done] land exactly as the inline loop produced them.
    // P1 Task 9: pre-checks and post-round bookkeeping are factored into
    // pre_round()/post_round() above -- the moved code is verbatim, so this
    // composition is byte-identical -- because the conductor's FUSED round
    // loop calls those two around fused_verify_round()+commit_outcome() in
    // place of the solo round below.
    bool decode_step(DecodeTask& t) {
        if (!pre_round(t)) return false;
        int em[W_MAX]; // width-12: spec_round can emit up to 12 tokens
        int n = t.sampling
                    ? (t.force_plain_sample ? sample_round(em) : spec_sample_round(em))
                    : spec_round(em);
        return post_round(t, em, n);
    }

    // P1 Task 10 (continuous batching): the prefill/setup half of generate()
    // -- everything from its entry through the d_P epilogue, moved VERBATIM
    // so the server's batch mode (Q27_BATCH=1) can run prefill on the request
    // thread under its own scoped gate lease and hand ONLY the decode loop to
    // the conductor. Returns false on the refused path (gs.end = "refused",
    // nothing prefilled); on success *P_out = prompt.size()-1, the host
    // position mirror make_decode_task() takes. generate() below composes
    // generate_prefill + make_decode_task + decode_step unchanged, so the
    // solo path (and the graphs it replays, A8) is byte-identical by
    // construction -- gated by the canonical/sampled/replay gates as always.
    bool generate_prefill(const std::vector<int>& prompt, int stable_len, int* P_out) {
        int NP = (int)prompt.size();
        gs = GenStats{};
        gs.prompt = NP;
        // Depth-controller lifetime (review 2026-07-09 + follow-up): the
        // ladder carries across same-lineage turns (re-earning depth from
        // k_min costs a measured -1.6% on 256-token requests, BUILDLOG
        // 2026-07-09); the server's claim_slot resets it when a new lineage
        // takes the slot over. Q27_MAXD_RESET=1 is the stricter
        // every-request reset.
        if (maxd_auto && maxd_reset) dctl.reset();
        // Suffix drafter: rebuild the match index over this request's full
        // prompt (multi-turn re-renders arrive whole, so reset covers
        // history). ~2-4ms host at 25K tokens, once per request, off the
        // decode path. sfx_valid arms after round 1 (host learns pending).
        if (suffix_on) {
            sfx.reset(prompt);
            sfx_valid = false;
        }
        auto t_in = std::chrono::steady_clock::now();
        // prefill writes KV rows [0, NP); nothing downstream bounds NP against
        // the cache allocations (found by kernel review) -- refuse cleanly
        // instead of corrupting adjacent buffers
        if (NP < 1 || NP > max_ctx) {
            // NP==0 would decode from stale d_token/recurrent state and echo the
            // prior request's token (Security #2); NP>max_ctx overruns the cache.
            fprintf(stderr, "[gen] prompt %d out of range (1..%d) -- refusing\n", NP, max_ctx);
            gs.end = "refused";
            return false;
        }
        // 07-05 audit (c), clear-at-claim: a non-CUDA throw between a prior
        // generate() and its tc.end() can leave a stale lane-0 mask +
        // accept-cap-1 (or stale split-path lane masks) on this engine, which
        // would silently constrain THIS request. Guarded on the host mirrors,
        // so paths that never constrained (CLI, canonical gates) issue no new
        // device work and stay bitwise.
        {
            bool stale = h_mask_id0 >= 0 || h_cap0 != 0;
            for (int i = 0; i < 5 && !stale; i++) stale = h_mask_ids5[i] >= 0;
            if (stale) {
                fprintf(stderr, "[toolgram] stale device constraint cleared at claim\n");
                clear_tool_constraint();
            }
        }
        if (batched_prefill && NP >= pf_batch_min) {
            // prefix-cache hit: prompt extends the snapshotted prefix -> restore
            // recurrent state and prefill only the new suffix
            int base = 0;
            if (have_snap && (int)snap_toks.size() <= NP - 1) {
                size_t L = 0;
                while (L < snap_toks.size() && snap_toks[L] == prompt[L]) L++;
                if (L == snap_toks.size()) base = (int)L;
            }
            int ck = -1;
            if (base == 0) {
                ck = ckpt_best(prompt); // P9: mid-history divergence fallback
                if (ck >= 0) base = (int)ckpts[ck].toks.size();
            }
            fprintf(stderr, "[gen] prompt=%d prefix_hit=%d snap=%zu ckpt=%d\n", NP, base,
                    snap_toks.size(), ck);
            gs.hit = base;
            gs.ckpt = ck;
            gs.pf = NP - base;
            // P9 alias fix (audit, 2026-07-12): re-prefilling [base..NP)
            // overwrites those KV rows with THIS conversation. Any cached
            // state whose coverage extends past base and does not match the
            // new prompt would later restore GDN state over foreign KV rows
            // (measured: a diverged branch left ring entries that restored
            // 4468 tokens of state over 1300+ foreign rows). Entries that
            // ARE a prefix of the new prompt stay valid: deterministic
            // prefill rewrites their rows with identical values.
            auto covers_prefix = [&](const std::vector<int>& t) {
                return t.size() <= prompt.size() &&
                       std::equal(t.begin(), t.end(), prompt.begin());
            };
            for (auto& c : ckpts)
                if (c.buf && (int)c.toks.size() > base && !covers_prefix(c.toks))
                    c.toks.clear();
            if (have_snap && (int)snap_toks.size() > base && !covers_prefix(snap_toks)) {
                have_snap = false;
                snap_toks.clear();
            }
            if (ck >= 0) ckpt_restore(ck);
            else if (base > 0) snap_restore();
            else { reset(); ckpt_clear(); }
            if (d_prompt_cap < NP) {
                if (d_prompt) CUDA_CHECK(cudaFree(d_prompt));
                CUDA_CHECK(cudaMalloc((void**)&d_prompt, (size_t)NP * 4));
                d_prompt_cap = NP;
            }
            CUDA_CHECK(cudaMemcpyAsync(d_prompt, prompt.data(), (size_t)NP * 4,
                                       cudaMemcpyHostToDevice, stm));
            const DevTensor& onw = dm.get("output_norm.weight");
            const int snap_upto =
                (stable_len > base && stable_len < NP) ? stable_len : NP - 1;
            int last_ck = base;
            for (int c0 = base; c0 < snap_upto; c0 += PF_T) {
                int Tc = std::min((int)PF_T, snap_upto - c0);
                prefill_chunk(d_prompt + c0, c0, Tc);
                q27k::rmsnorm_T(hT, (const float*)onw.data, x1T, N_EMBD, Tc, EPS, stm);
                mtp_warm_T(d_prompt + c0 + 1, c0, Tc);
                if (ckpt_interval > 0 && (c0 + Tc) - last_ck >= ckpt_interval) {
                    ckpt_save(prompt, c0 + Tc);
                    last_ck = c0 + Tc;
                }
                round_gap();
            }
            snap_save(prompt, snap_upto);
            for (int c0 = snap_upto; c0 < NP - 1; c0 += PF_T) {
                int Tc = std::min((int)PF_T, (NP - 1) - c0);
                prefill_chunk(d_prompt + c0, c0, Tc);
                q27k::rmsnorm_T(hT, (const float*)onw.data, x1T, N_EMBD, Tc, EPS, stm);
                mtp_warm_T(d_prompt + c0 + 1, c0, Tc);
                if (ckpt_interval > 0 && (c0 + Tc) - last_ck >= ckpt_interval) {
                    ckpt_save(prompt, c0 + Tc);
                    last_ck = c0 + Tc;
                }
                round_gap();
            }
            int pos_last = NP - 1;
            CUDA_CHECK(cudaMemcpyAsync(d_pos, &pos_last, 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemcpyAsync(d_step, &pos_last, 4, cudaMemcpyHostToDevice, stm));
            step_with(prompt[NP - 1]);
        } else {
            reset();
            // Serial path leaves no reusable cache: clear the snapshot AND the
            // checkpoint ring (Ckpt contract: the KV rows under a checkpoint
            // must still describe its conversation -- this prefill overwrites
            // them). Stale snap_toks/ckpts here let multi-slot routing send an
            // old conversation back to a dead slot and ckpt_restore over
            // foreign KV rows (R1 review finding).
            have_snap = false;
            snap_toks.clear();
            ckpt_clear();
            gs.pf = NP;
            for (size_t i = 0; i < prompt.size(); i++) {
                step_with(prompt[i]);
                if (i + 1 < prompt.size()) {
                    CUDA_CHECK(cudaStreamSynchronize(stm));
                    CUDA_CHECK(cudaMemcpyAsync(h_next, x1, N_EMBD * 4, cudaMemcpyDeviceToDevice,
                                               stm));
                    int nt = prompt[i + 1], mp = (int)i + 1;
                    CUDA_CHECK(cudaMemcpyAsync(d_token, &nt, 4, cudaMemcpyHostToDevice, stm));
                    CUDA_CHECK(cudaMemcpyAsync(d_pos_m, &mp, 4, cudaMemcpyHostToDevice, stm));
                    mtp_forward();
                    CUDA_CHECK(cudaStreamSynchronize(stm));
                }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stm));
        gs.pf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                             t_in)
                       .count();
        CUDA_CHECK(cudaMemcpyAsync(h_next, x1, N_EMBD * 4, cudaMemcpyDeviceToDevice, stm));
        int P = (int)prompt.size() - 1;
        CUDA_CHECK(cudaMemcpyAsync(d_P, &P, 4, cudaMemcpyHostToDevice, stm));
        *P_out = P;
        return true;
    }

    // Prompt + speculative generation. Calls on_token(id) for each generated
    // token; stop when on_token returns false, n_max hit, or eos. Uses the spec
    // path (requires build_spec_graphs()). MTP KV warmed during prompt.
    // stable_len (P8): token index of the stable-prefix boundary (end of the
    // last input message). The GDN snapshot is taken THERE instead of at the
    // prompt tail, so the next turn's re-rendered history prefix-matches and
    // only the per-turn suffix re-prefills. -1 = legacy tail snapshot.
    template <typename F>
    int generate(const std::vector<int>& prompt, int n_max, int eos, F&& on_token,
                 int stable_len = -1) {
        int P = 0;
        if (!generate_prefill(prompt, stable_len, &P)) return 0;
        // Decode: pre-loop state lives in DecodeTask, one round per
        // decode_step (P1: the conductor drives the same step function for
        // several engines). finish_decode() runs inside the step that
        // returns false, so every exit path lands its GenStats/[gen-done].
        DecodeTask t;
        make_decode_task(t, n_max, eos, on_token, P);
        while (decode_step(t)) {}
        return t.emitted;
    }

    void build_graph() {
        // warm run (outside capture) so lazy CUDA state is initialized
        int zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_token, &zero, 4, cudaMemcpyHostToDevice, stm));
        token_launches();
        CUDA_CHECK(cudaStreamSynchronize(stm));
        // reset state mutated by the warm run
        CUDA_CHECK(cudaMemset(d_pos, 0, 4));
        CUDA_CHECK(cudaMemset(d_step, 0, 4));
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                CUDA_CHECK(cudaMemset(conv_ring[il], 0, 3 * GDN_CH * 4));
                CUDA_CHECK(cudaMemset(S[il], 0, (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4));
            }

        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
        token_launches();
        CUDA_CHECK(cudaStreamEndCapture(stm, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        fprintf(stderr, "token graph captured\n");

        // Sampled sibling graph (roadmap #2): same forward, sampled tail. Warm
        // once (init lazy state for the new sampler kernels), reset, capture.
        // Never used by greedy requests -> greedy graph + canonical untouched.
        q27k::SampleParams warm{1.f, 1.f, 0ull};
        CUDA_CHECK(cudaMemcpyAsync(d_samp, &warm, sizeof warm, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemcpyAsync(d_token, &zero, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemset(d_pos, 0, 4));
        CUDA_CHECK(cudaMemset(d_step, 0, 4));
        token_launches_sampled();
        CUDA_CHECK(cudaStreamSynchronize(stm));
        CUDA_CHECK(cudaMemset(d_pos, 0, 4));
        CUDA_CHECK(cudaMemset(d_step, 0, 4));
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                CUDA_CHECK(cudaMemset(conv_ring[il], 0, 3 * GDN_CH * 4));
                CUDA_CHECK(cudaMemset(S[il], 0, (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4));
            }
        cudaGraph_t sgraph;
        CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
        token_launches_sampled();
        CUDA_CHECK(cudaStreamEndCapture(stm, &sgraph));
        CUDA_CHECK(cudaGraphInstantiate(&sample_graph, sgraph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(sgraph));
        fprintf(stderr, "sample graph captured\n");
    }

    // feed one known token (prompt phase): set d_token, replay graph
    void step_with(int token) {
        CUDA_CHECK(cudaMemcpyAsync(d_token, &token, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stm));
    }
    // generation step: d_token already holds the model's own prediction
    void step_free() { CUDA_CHECK(cudaGraphLaunch(graph_exec, stm)); }

    // DFlash Phase-0 tap capture: EAGER forward (no graph) writing the
    // residual stream after DFLASH_TAPS layers into d_taps. Debug rig only.
    float* d_taps = nullptr; // [5][N_EMBD]
    void step_taps(int token) {
        if (!d_taps) CUDA_CHECK(cudaMalloc((void**)&d_taps, 5 * N_EMBD * 4));
        CUDA_CHECK(cudaMemcpyAsync(d_token, &token, 4, cudaMemcpyHostToDevice, stm));
        token_launches(d_taps);
    }
    void step_taps_free() {
        if (!d_taps) CUDA_CHECK(cudaMalloc((void**)&d_taps, 5 * N_EMBD * 4));
        token_launches(d_taps);
    }
};
