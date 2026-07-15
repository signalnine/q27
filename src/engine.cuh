// q27 Engine: qwen35 hybrid forward + MTP speculative decode. Header-only
// (all methods inline) so both the CLI and the server can embed it.
#pragma once
#include <algorithm>
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

    // ---- batched prefill (M6) ----
    // Prefill chunk size. 256 left GEMM launches at ~320 blocks on 170 SMs
    // (27% of int8 peak) and re-read all 17.7GB of weights T/256 times; 1024
    // fills the machine and cuts weight re-reads 4x. Costs ~0.8GB scratch.
    static constexpr int PF_T = 1024;
    static constexpr int PF_SB = 32;  // attention sub-batch (scratch rows)
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

    void qx(const float* x, int cols) { q27k::quantize_x(x, cols, xq, stm); }

    void mm(const DevTensor& w, const float* x, float* out) {
        switch (w.dtype) {
            case DType::Q4_G64:
                q27k::gemv_q4((const uint8_t*)w.data, (const __half*)w.scales, xq, out, w.rows,
                              w.cols, stm);
                break;
            case DType::Q8_G128:
                q27k::gemv_q8((const int8_t*)w.data, (const __half*)w.scales, xq, out, w.rows,
                              w.cols, stm);
                break;
            case DType::F16:
                q27k::gemv_f16((const __half*)w.data, x, out, w.rows, w.cols, stm);
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
        if (!kc) {
            int ci = attn_cache_idx[il];
            kc = kcache[ci];
            vc = vcache[ci];
        }
        if (!pos_src) pos_src = d_pos;
        qx(xin, N_EMBD);
        mm(T(il, "attn_q.weight"), xin, qg);
        q27k::rmsnorm_heads(qg, (const float*)T(il, "attn_q_norm.weight").data, qg, N_HEAD,
                            HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        mm(T(il, "attn_k.weight"), xin, kbuf);
        q27k::rmsnorm_heads(kbuf, (const float*)T(il, "attn_k_norm.weight").data, kbuf, N_KV,
                            HEAD_DIM, HEAD_DIM, EPS, stm);
        mm(T(il, "attn_v.weight"), xin, vbuf);
        q27k::rope_neox_partial(qg, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, pos_src, FREQ_BASE, stm);
        q27k::rope_neox_partial(kbuf, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, pos_src, FREQ_BASE, stm);
        // turbo3: Q forward-WHT after rope (K's rotation is folded into the
        // store; <WHT q, WHT K> == <q,K>); turbo3v keeps K fp16 => Q raw.
        // Host branches on kv_kind only -- fixed at init, graph-capture-safe.
        if (kv_kind == KV_T3) {
            q27k::P3 qw{{qg}};
            q27k::wht3(qw, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, false, stm, 1);
        }
        if (kv_kind >= KV_T3) {
            q27k::CP3 kw{{kbuf}};
            q27k::CP3 vw3{{vbuf}};
            q27k::IP3 pw{{pos_src}};
            q27k::kv_store_t3(kw, vw3, kc, vc, pw, N_KV, HEAD_DIM, stm, 1,
                              /*k_plain=*/kv_kind == KV_T3V);
        } else {
            q27k::kv_store(kbuf, vbuf, kc, vc, pos_src, N_KV * HEAD_DIM, stm, kv_fp8);
        }
        q27k::attn_decode(qg, 2 * HEAD_DIM, kc, vc, attnout, scratch, pos_src,
                          max_ctx, N_HEAD, N_KV, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM), stm,
                          kv_kind);
        // turbo3 V accumulates in the rotated basis: one inverse-WHT on the
        // pooled output BEFORE the sigmoid gate (elementwise gate does not
        // commute with the rotation).
        if (kv_kind >= KV_T3) {
            q27k::P3 ow{{attnout}};
            q27k::wht3(ow, N_HEAD, HEAD_DIM, HEAD_DIM, true, stm, 1);
        }
        q27k::sigmoid_gate_mul(attnout, qg, N_HEAD, HEAD_DIM, stm);
        qx(attnout, N_HEAD * HEAD_DIM);
        mm(T(il, "attn_output.weight"), attnout, yout);
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
    void mtp_forward(const float* h_src = nullptr, const int* tok_src = nullptr,
                     int* draft_dst = nullptr, const int* pos_src = nullptr,
                     float* margin_dst = nullptr) {
        if (!h_src) h_src = h_next;
        if (!tok_src) tok_src = d_token;
        if (!draft_dst) draft_dst = d_draft;
        if (!pos_src) pos_src = d_pos_m;
        const int il = 64;
        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed_row_q8((const int8_t*)emb.data, (const __half*)emb.scales, tok_src, N_EMBD,
                           e_hn, stm);
        q27k::rmsnorm(e_hn, (const float*)T(il, "nextn.enorm.weight").data, e_hn, N_EMBD, EPS,
                      stm);
        q27k::rmsnorm(h_src, (const float*)T(il, "nextn.hnorm.weight").data, e_hn + N_EMBD,
                      N_EMBD, EPS, stm);
        qx(e_hn, 2 * N_EMBD);
        mm(T(il, "nextn.eh_proj.weight"), e_hn, x_mtp);

        q27k::rmsnorm(x_mtp, (const float*)T(il, "attn_norm.weight").data, x1, N_EMBD, EPS, stm);
        attn_block(il, x1, y, mtp_k, mtp_v, pos_src);
        q27k::add_inplace(x_mtp, y, N_EMBD, stm);
        q27k::rmsnorm(x_mtp, (const float*)T(il, "post_attention_norm.weight").data, x1, N_EMBD,
                      EPS, stm);
        ffn(il, x1, y);
        q27k::add_inplace(x_mtp, y, N_EMBD, stm);
        q27k::rmsnorm(x_mtp, (const float*)T(il, "nextn.shared_head_norm.weight").data, x1,
                      N_EMBD, EPS, stm);
        qx(x1, N_EMBD);
        // drafts use the Q4 head copy when present (verify keeps the Q8 head,
        // so output remains exactly the faithful model's greedy text)
        const DevTensor* head = dm.model_has("output_q4.weight")
                                    ? &dm.get("output_q4.weight")
                                    : &dm.get("output.weight");
        mm(*head, x1, mtp_logits);
        if (margin_dst)
            q27k::argmax_margin(mtp_logits, VOCAB, draft_dst, margin_dst, d_am_blk1, d_am_blk2, stm);
        else
            q27k::argmax(mtp_logits, VOCAB, draft_dst, d_amax, stm);
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
        return v;
    }
    // W16: the flat 12-pointer overloads are gone. They existed so a call site
    // could name its lanes, but at W_PLUMB=16 mm5's flat form would take 17
    // params -- the exact signature wall that pushed prep/finish onto by-value
    // structs. Every caller already had its lanes in a W_PLUMB array (or can
    // build one, as the vocab head does), so the array form is the only form.
    void qx5(const std::array<float*, W_PLUMB>& x, int cols) {
        q27k::XQ3 q{};
        q27k::CP3 xs{};
        for (int i = 0; i < W_PLUMB; i++) {
            q.q[i] = xq_L[i]; // xq_L[0]/[1] alias xq2[0]/[1]
            xs.p[i] = x[i];
        }
        q27k::quantize3(xs, cols, q, stm, vw);
    }
    void mm5(const DevTensor& w, const std::array<float*, W_PLUMB>& ys_a) {
        // P2: WIDE rounds take the flat-in-W MMA GEMM; the ladder keeps the GEMV.
        // Both `vw` and `gemm_min` are host ints read at CUDA-GRAPH CAPTURE, so the
        // branch is baked per graph -- no per-call work, no divergence at replay.
        // vw <= 8 (every gated, draft and sampled round) can never take this branch:
        // gemm_min is 9 and build_spec_graphs aborts if gate_maxd+1 ever reaches it.
        // That is what makes the canonical bitwise gate structural.
        // k_vgemm reuses the group-32 int8 activations quantize3 ALREADY writes
        // (xq_L[i].nat/.scale are dead stores on the dp4a GEMV path today), so this
        // adds no quantize pass, no buffer and no graph node on the activation side.
        if (vw >= gemm_min && (int64_t)w.rows >= gemm_min_rows) {
            q27k::XLanes X{};
            q27k::YLanes Y{};
            for (int i = 0; i < W_PLUMB; i++) {
                X.nat[i] = xq_L[i].nat;
                X.xs[i] = xq_L[i].scale;
                Y.y[i] = ys_a[i];
            }
            // Honor the false: an ineligible shape MUST fall through to the GEMV
            // rather than silently produce nothing (the launch_fdmma contract).
            if (q27k::vgemm_verify(w, X, Y, d_vgemm_ws, vw, stm)) return;
        }
        q27k::XQuant qs[W_PLUMB];
        float* ys[W_PLUMB];
        for (int i = 0; i < W_PLUMB; i++) {
            qs[i] = xq_L[i];
            ys[i] = ys_a[i];
        }
        if (w.dtype == DType::Q4_G64)
            q27k::gemv_q4_n((const uint8_t*)w.data, (const __half*)w.scales, qs, vw, ys, w.rows,
                            w.cols, stm);
        else
            q27k::gemv_q8_n((const int8_t*)w.data, (const __half*)w.scales, qs, vw, ys, w.rows,
                            w.cols, stm);
    }

    void gdn_pair(int il) {
        const float eps = EPS;
        qx5(x1_L, N_EMBD);
        mm5(T(il, "attn_qkv.weight"), qkv_L);
        mm5(T(il, "attn_gate.weight"), z_L);
        q27k::gemv_f16_3((const __half*)T(il, "ssm_alpha.weight").data,
                         LANESW(x1),
                         LANESW(alpha), GDN_HEADS,
                         N_EMBD, stm, vw);
        q27k::gemv_f16_3((const __half*)T(il, "ssm_beta.weight").data,
                         LANESW(x1),
                         LANESW(betar), GDN_HEADS,
                         N_EMBD, stm, vw);
        const float* sa = (const float*)T(il, "ssm_a").data;
        const float* sdt = (const float*)T(il, "ssm_dt.bias").data;
        q27k::gdn_gates3(LANESW(alpha),
                         LANESW(betar), sa, sdt,
                         LANESW(g),
                         LANESW(beta), GDN_HEADS, stm, vw);
        const float* cw = (const float*)T(il, "ssm_conv1d.weight").data;
        // P12: per-lane recurrent chain -- role k reads role k-1 (written fresh
        // earlier this round) and writes role k. Only lanes < vw are live; a
        // width-vw graph skips the rest, leaving their (never-read) role buffers
        // untouched. Lane a (role 0, the pending token) always runs.
        q27k::conv_step(RBuf(il, 0), RBuf(il, 0), qkv, cw, convout, GDN_CH, stm); // lane 0
        for (int L = 1; L < vw; L++)
            q27k::conv_step(RBuf(il, L - 1), RBuf(il, L), qkv_L[L], cw, convout_L[L], GDN_CH, stm);
        // q||k are contiguous (offsets 0 and 2048): 32 heads in one merged call
        q27k::l2norm3(LANESW(convout), 32,
                      GDN_DIM, eps, stm, vw);
        q27k::delta_step(SBuf(il, 0), SBuf(il, 0), convout, g, beta, o, stm); // lane 0
        for (int L = 1; L < vw; L++)
            q27k::delta_step(SBuf(il, L - 1), SBuf(il, L), convout_L[L], g_L[L], beta_L[L], o_L[L], stm);
        const float* nw = (const float*)T(il, "ssm_norm.weight").data;
        q27k::gated_norm3(LANESW(o), nw,
                          LANESW(z),
                          LANESW(og), GDN_HEADS, GDN_DIM, eps, stm, vw);
        qx5(og_L, GDN_V);
        mm5(T(il, "ssm_out.weight"), y_L);
    }

    void attn_pair(int il) {
        int ci = attn_cache_idx[il];
        qx5(x1_L, N_EMBD);
        mm5(T(il, "attn_q.weight"), qg_L);
        const float* qn = (const float*)T(il, "attn_q_norm.weight").data;
        const float* kn = (const float*)T(il, "attn_k_norm.weight").data;
        for (int L = 0; L < vw; L++)
            q27k::rmsnorm_heads(qg_L[L], qn, qg_L[L], N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        mm5(T(il, "attn_k.weight"), kbuf_L);
        for (int L = 0; L < vw; L++)
            q27k::rmsnorm_heads(kbuf_L[L], kn, kbuf_L[L], N_KV, HEAD_DIM, HEAD_DIM, EPS, stm);
        mm5(T(il, "attn_v.weight"), vbuf_L);
        q27k::IP3 P LANESW(d_pos);
        q27k::rope3(LANESW(qg),
                    N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, P,
                    FREQ_BASE, stm, vw);
        q27k::rope3(LANESW(kbuf), N_KV, HEAD_DIM, N_ROT,
                    HEAD_DIM, P, FREQ_BASE, stm, vw);
        float kq = 1.0f / sqrtf((float)HEAD_DIM);
        // turbo3: rotate all vw Q lanes post-rope (see attn_block); host
        // branch on kv_kind only (init-fixed, graph-capture-safe)
        if (kv_kind == KV_T3)
            q27k::wht3(LANESW(qg), N_HEAD, HEAD_DIM, 2 * HEAD_DIM, false, stm, vw);
        // store vw lanes (disjoint slots); each token's attention only reads
        // cache[0 .. its own pos], so later tokens' entries are invisible to earlier ones
        if (kv_kind >= KV_T3)
            q27k::kv_store_t3(LANESW(kbuf),
                              LANESW(vbuf), kcache[ci], vcache[ci],
                              P, N_KV, HEAD_DIM, stm, vw, /*k_plain=*/kv_kind == KV_T3V);
        else
            q27k::kv_store3(LANESW(kbuf),
                            LANESW(vbuf), kcache[ci], vcache[ci],
                            P, N_KV * HEAD_DIM, stm, vw, kv_fp8);
        q27k::attn_decode3(LANESW(qg), 2 * HEAD_DIM, kcache[ci],
                           vcache[ci],
                           LANESW(attnout),
                           scratch, P, max_ctx, N_HEAD, N_KV, HEAD_DIM, kq, stm, vw, kv_kind);
        // inverse-WHT on all vw pooled outputs BEFORE the sigmoid gate
        if (kv_kind >= KV_T3)
            q27k::wht3(LANESW(attnout),
                       N_HEAD, HEAD_DIM, HEAD_DIM, true, stm, vw);
        q27k::sigmoid_gate3(LANESW(attnout),
                            LANESW(qg), N_HEAD, HEAD_DIM, stm, vw);
        qx5(attnout_L, N_HEAD * HEAD_DIM);
        mm5(T(il, "attn_output.weight"), y_L);
    }

    void ffn_pair(int il) {
        qx5(x1_L, N_EMBD);
        mm5(T(il, "ffn_gate.weight"), ffn_g_L);
        mm5(T(il, "ffn_up.weight"), ffn_u_L);
        q27k::silu_mul3(LANESW(ffn_g),
                        LANESW(ffn_u), N_FFN, stm, vw);
        qx5(ffn_g_L, N_FFN);
        mm5(T(il, "ffn_down.weight"), y_L);
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
        if (k == 0) {
            q27k::prep_round(d_P, d_token, lane_pos(), mtp_pos(), W_MAX, D_MAX_MTP, d_outcome,
                             stm);
            mtp_forward(h_next, d_token, d_draft, d_pos_m, d_draft_margin + 0);
            return;
        }
        float* hs[7] = {h_next, h_next2, h_next3, h_next4, h_next5, h_next6, h_next7};
        const int* ts[7] = {d_token, d_draft, d_draft2, d_draft3, d_draft4, d_draft5, d_draft6};
        int* ds[7] = {d_draft, d_draft2, d_draft3, d_draft4, d_draft5, d_draft6, d_draft7};
        const int* ps[7] = {d_pos_m, d_pos_m2, d_pos_m3, d_pos_m4, d_pos_m5, d_pos_m6, d_pos_m7};
        CUDA_CHECK(cudaMemcpyAsync(hs[k], x1, N_EMBD * 4, cudaMemcpyDeviceToDevice, stm));
        mtp_forward(hs[k], ts[k], ds[k], ps[k], d_draft_margin + k);
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
    void spec_verify_forward() {
        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed3((const int8_t*)emb.data, (const __half*)emb.scales, verify_tokens(),
                     N_EMBD, LANESW(h), stm,
                     vw);
        q27k::CP3 Hc LANESW(h),
            Yc LANESW(y);
        q27k::P3 Hm LANESW(h),
            X1m LANESW(x1);
        for (int il = 0; il < N_LAYER; il++) {
            const float* an = (const float*)T(il, "attn_norm.weight").data;
            q27k::rmsnorm3(Hc, an, X1m, N_EMBD, EPS, stm, vw);
            if (attn_layer[il]) attn_pair(il);
            else gdn_pair(il);
            q27k::add3(Hm, Yc, N_EMBD, stm, vw);
            const float* pn = (const float*)T(il, "post_attention_norm.weight").data;
            q27k::rmsnorm3(Hc, pn, X1m, N_EMBD, EPS, stm, vw);
            ffn_pair(il);
            q27k::add3(Hm, Yc, N_EMBD, stm, vw);
        }
        const float* on = (const float*)dm.get("output_norm.weight").data;
        q27k::rmsnorm3(Hc, on, X1m, N_EMBD, EPS, stm, vw);
        qx5(x1_L, N_EMBD);
        const char* vhead = (fast_head && dm.model_has("output_q4.weight")) ? "output_q4.weight"
                                                                             : "output.weight";
        // lane t's logits live at logits2 + t*VOCAB (the alloc is W_MAX*VOCAB;
        // only lanes < vw are computed, and only those are ever read).
        std::array<float*, W_PLUMB> lg{};
        for (int t = 0; t < W_PLUMB; t++)
            lg[t] = logits2 + (size_t)(t < W_MAX ? t : 0) * VOCAB;
        mm5(dm.get(vhead), lg);
    }

    // P11: verify half -- batch-5 forward, masked argmax per lane, finish_round.
    void spec_verify_launches() {
        spec_verify_forward();
        // P7: slot 0 (the post-pending lane) is the constrained one; the rest
        // keep id -1 (v1 caps acceptance in-grammar instead of chasing
        // draft-dependent states the host cannot know pre-launch).
        // W16: was an unrolled `if (vw > k)` chain per lane. The loop emits the
        // identical launch sequence in the identical order for any vw, so the
        // captured graphs -- and the tokens they produce -- are unchanged at
        // every width the chain covered.
        for (int t = 0; t < vw; t++)
            q27k::argmax_masked(logits2 + (size_t)t * VOCAB, VOCAB, d_mask_pool, mask_words,
                                d_mask_ids, t, d_v_L[t], d_amax, stm);
        // P12: a width-vw verify computed columns 0..vw-1; cap acceptance at vw-1
        // drafts so finish never commits an uncomputed lane. vw=5 => max_draft=4.
        q27k::IP3 drafts{};
        for (int k = 0; k + 1 < W_PLUMB; k++) drafts.p[k] = d_draft_L[k];
        q27k::finish_round(d_P, d_token, drafts,
                           LANESW(d_v),
                           LANESW(x1),
                           h_next, d_outcome, N_EMBD, d_accept_cap, vw - 1, stm);
    }

    // Phase 2: sampled verify tail. Same forward; replace the 5 argmax lanes +
    // equality-chain finish with per-lane nucleus stats, rejection-sampling
    // acceptance (k_spec_accept), a resample of the new pending from the stop
    // lane (k_sample_stop -> d_token), and finish keyed on the accepted count n.
    // Draws key Philox on *d_P; greedy graphs stay bitwise (separate graph set).
    void spec_verify_launches_sampled() {
        spec_verify_forward();
        // P14: width-vw sampled verify -- nucleus stats + accept walk over the
        // first vw lanes only (vw=5 monolithic; vw=cap+1 under the gate). The
        // accept walk caps at vw-1 drafts so finish never commits an uncomputed
        // lane. vw=5 => max_draft=4 (the pre-P14 behavior). k_finish_sampled is
        // unchanged: it keys on n<=vw and its src select covers n in 1..5.
        for (int k = 0; k < vw; k++)
            q27k::nucleus(logits2 + (size_t)k * VOCAB, VOCAB, d_samp, d_nuc + k * 4, stm);
        q27k::spec_accept(logits2, d_nuc, d_draft, d_draft2, d_draft3, d_draft4, d_samp, d_P,
                          d_accept_cap, vw - 1, VOCAB, d_spec, stm);
        q27k::sample_stop(logits2, d_nuc, d_spec, d_samp, d_P, VOCAB, d_token, d_amax, stm);
        q27k::finish_sampled(d_P, d_token, d_spec, d_draft, d_draft2, d_draft3, d_draft4, x1,
                             x1_L[1], x1_L[2], x1_L[3], x1_L[4], h_next, d_outcome, N_EMBD, stm);
    }

    void spec_round_launches() {
        spec_draft_launches();
        spec_verify_launches();
    }

    void spec_sample_round_launches() {
        spec_draft_launches();
        spec_verify_launches_sampled();
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
            spec_verify_launches();
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
                spec_verify_launches();
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
                spec_verify_launches();
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
                spec_verify_launches_sampled();
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
        // R1b: time parked in on_round_gap calls that actually handed the
        // GPU over (includes the pre-yield drain of our own in-flight
        // chunks), and how many handovers. pf_ms/dec_ms stay wall-inclusive
        // of yields; analyzers subtract gw_ms for GPU-busy accounting.
        double gw_ms = 0;
        // Q27_PHASE_STATS: summed gated-round draft/verify wall (ms) and MTP
        // draft steps launched, this request. dec_ms - draft_ms - verify_ms =
        // host round-gap + any unattributed (constrained/ungated) rounds.
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
            return 0;
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
        // R1b preemption point (no-op when the hook is unset or nobody waits)
        auto round_gap = [&] {
            if (!on_round_gap) return;
            auto y0 = std::chrono::steady_clock::now();
            if (on_round_gap()) {
                gs.gw_ms += std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - y0)
                                .count();
                gs.yields++;
            }
        };
        if (batched_prefill && NP >= 32) {
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
        int emitted = 0, rounds = 0;
        auto g0 = std::chrono::steady_clock::now();
        // decode-phase park baseline: dt below covers decode only, so the
        // contended-print suffix must use decode-phase gw, not request-total
        // (prefill parks belong to pf_ms; mixing them over-corrects t/s)
        const double gw_pf = gs.gw_ms;
        const int yields_pf = gs.yields;
        // Q27_PROF_DECODE=1: bracket the decode loop with a cudaProfiler
        // range so `nsys --capture-range=cudaProfilerApi` records ONLY the
        // decode slice -- prefill otherwise floods the trace (the CLI
        // --tokens path walks the prompt serially; see BUILDLOG nsys notes).
        // No-op without a profiler attached; done() below is the single
        // exit funnel, so every decode exit path closes the range.
        const bool prof_decode = getenv("Q27_PROF_DECODE") != nullptr;
        if (prof_decode) {
            CUDA_CHECK(cudaStreamSynchronize(stm));
            (void)cudaProfilerStart();
        }
        auto done = [&](const char* why) {
            if (prof_decode) {
                CUDA_CHECK(cudaStreamSynchronize(stm));
                (void)cudaProfilerStop();
            }
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - g0)
                            .count();
            gs.dec = emitted;
            gs.dec_ms = dt * 1000.0;
            gs.rounds = rounds;
            gs.end = why;
            // wall-inclusive of yield parks; the suffix keeps a contended
            // print from reading as a decode regression
            if (gs.yields > yields_pf)
                fprintf(stderr,
                        "[gen-done] %s: %d tokens in %.1fs (%.1f t/s; parked %.0fms/%d "
                        "yields in decode), n_max=%d\n",
                        why, emitted, dt, emitted / (dt > 0 ? dt : 1), gs.gw_ms - gw_pf,
                        gs.yields - yields_pf, n_max);
            else
                fprintf(stderr, "[gen-done] %s: %d tokens in %.1fs (%.1f t/s), n_max=%d\n",
                        why, emitted, dt, emitted / (dt > 0 ? dt : 1), n_max);
            // Phase 2 acceptance-vs-temp telemetry (sampled path only; keeps the
            // greedy [gen-done] line shortbench_suite parses untouched). tokens/
            // round is the sampled spec acceptance -- it sags as temperature rises.
            if (sampling && !force_plain_sample)
                fprintf(stderr,
                        "[sample-stats] T=%.3f top_p=%.3f: %.3f tokens/round (%d tok/%d rounds)\n",
                        samp.inv_temp > 0.f ? 1.0f / samp.inv_temp : 0.f, samp.top_p,
                        rounds > 0 ? (double)emitted / rounds : 0.0, emitted, rounds);
        };
        // ctx guard: a round writes attention-KV rows P+1..P+gate_maxd+1 (and
        // MTP rows P+1..P+gate_maxd); launching past the reserve would write
        // beyond the caches and corrupt adjacent allocations (which the prefix
        // cache would then reuse). Stop instead -- a max-length response ends a
        // few tokens short of the absolute ceiling rather than corrupting state.
        int Ph = P;
        while (emitted < n_max) {
            if (Ph + ctx_round_reserve() > max_ctx) { done("ctx-guard"); return emitted; }
            int em[W_MAX]; // width-12: spec_round can emit up to 12 tokens
            int n = sampling ? (force_plain_sample ? sample_round(em) : spec_sample_round(em))
                             : spec_round(em);
            rounds++;
            // P15 engage-lag fix: let the host grammar scan the whole round
            // pre-emission; on a mid-round <tool_call> completion, truncate to
            // the marker token and re-decide the pending under the staged mask.
            if (!sampling && on_round) {
                int m = on_round(em, n);
                if (m >= 1 && m <= n) {
                    last_pending = refinish_round(m, n, Ph + m);
                    n = m;
                }
            }
            // suffix index tracks the committed stream (post-truncation n);
            // the pending token rides along virtually in propose_with.
            if (suffix_on)
                for (int k = 0; k < n; k++) sfx.append(em[k]);
            Ph += n;
            for (int k = 0; k < n && emitted < n_max; k++) {
                if (em[k] == eos) { done("eos"); return emitted; }
                auto c0 = std::chrono::steady_clock::now();
                bool cont = on_token(em[k]);
                gs.cb_ms += std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - c0)
                                .count();
                if (!cont) { done("client-stop"); return emitted; }
                emitted++;
            }
            if (!sampling && on_pending) on_pending(last_pending);
            round_gap();
        }
        done("n_max");
        return emitted;
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
