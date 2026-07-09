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
#include "device_model.h"
#include "kernels.cuh"
#include "loader.h"

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
    float *h_b, *x1_b, *y_b, *qg_b, *kbuf_b, *vbuf_b, *attnout_b;
    float *qkv_b, *convout_b, *z_b, *alpha_b, *betar_b, *g_b, *beta_b, *o_b, *og_b;
    float *ffn_g_b, *ffn_u_b, *logits2, *y2big;
    float *S_spare[N_LAYER], *ring_spare[N_LAYER];
    float *S_spare2[N_LAYER], *ring_spare2[N_LAYER];
    float *S_spare3[N_LAYER], *ring_spare3[N_LAYER];
    float *S_spare4[N_LAYER], *ring_spare4[N_LAYER];
    float *S_spare5[N_LAYER], *ring_spare5[N_LAYER]; // P12b: 6th role (maxd=5)
    float *S_spare6[N_LAYER], *ring_spare6[N_LAYER]; // maxd6: 7th role (maxd=6)
    float *S_spare7[N_LAYER], *ring_spare7[N_LAYER]; // maxd7: 8th role (maxd=7)
    float *h_c, *x1_c, *y_c, *qg_c, *kbuf_c, *vbuf_c, *attnout_c;
    float *qkv_c, *convout_c, *z_c, *alpha_c, *betar_c, *g_c, *beta_c, *o_c, *og_c;
    float *ffn_g_c, *ffn_u_c;
    float *h_next2;
    q27k::XQuant xqC;
    int *d_pos_c, *d_pos_m2, *d_draft2, *d_vc;
    // depth-3 lane (d): 4th verify column + pass-3 draft chain
    float *h_d, *x1_d, *y_d, *qg_d, *kbuf_d, *vbuf_d, *attnout_d;
    float *qkv_d, *convout_d, *z_d, *alpha_d, *betar_d, *g_d, *beta_d, *o_d, *og_d;
    float *ffn_g_d, *ffn_u_d;
    float *h_next3;
    q27k::XQuant xqD;
    int *d_pos_d, *d_pos_m3, *d_draft3, *d_vd;
    // depth-4 lane (e): 5th verify column + pass-4 draft chain (P3)
    float *h_e, *x1_e, *y_e, *qg_e, *kbuf_e, *vbuf_e, *attnout_e;
    float *qkv_e, *convout_e, *z_e, *alpha_e, *betar_e, *g_e, *beta_e, *o_e, *og_e;
    float *ffn_g_e, *ffn_u_e;
    float *h_next4;
    q27k::XQuant xqE;
    int *d_pos_e, *d_pos_m4, *d_draft4, *d_ve;
    // depth-5 lane (f): 6th verify column + pass-5 draft chain (P12b)
    float *h_f, *x1_f, *y_f, *qg_f, *kbuf_f, *vbuf_f, *attnout_f;
    float *qkv_f, *convout_f, *z_f, *alpha_f, *betar_f, *g_f, *beta_f, *o_f, *og_f;
    float *ffn_g_f, *ffn_u_f;
    float *h_next5;
    q27k::XQuant xqF;
    int *d_pos_f, *d_pos_m5, *d_draft5, *d_vf;
    // depth-6 lane (g): 7th verify column + pass-6 draft chain (maxd6 ladder)
    float *h_g, *x1_g, *y_g, *qg_g, *kbuf_g, *vbuf_g, *attnout_g;
    float *qkv_g, *convout_g, *z_g, *alpha_g, *betar_g, *g_g, *beta_g, *o_g, *og_g;
    float *ffn_g_g, *ffn_u_g;
    float *h_next6;
    q27k::XQuant xqG;
    int *d_pos_g, *d_pos_m6, *d_draft6, *d_vg;
    // depth-7 lane (h): 8th verify column + pass-7 draft chain (maxd7 ladder)
    float *h_h, *x1_h, *y_h, *qg_h, *kbuf_h, *vbuf_h, *attnout_h;
    float *qkv_h, *convout_h, *z_h, *alpha_h, *betar_h, *g_h, *beta_h, *o_h, *og_h;
    float *ffn_g_h, *ffn_u_h;
    float *h_next7;
    q27k::XQuant xqH;
    int *d_pos_h, *d_pos_m7, *d_draft7, *d_vh;
    int *d_P, *d_outcome;
    q27k::XQuant xq2[2];
    int *d_pos_a, *d_pos_b, *d_va, *d_vb;
    // P7 constrained tool decoding: resident mask pool + per-slot ids +
    // acceptance cap. All -1/0 when inactive -> bitwise-identical decode.
    unsigned* d_mask_pool = nullptr;
    int* d_mask_ids = nullptr;
    int* d_accept_cap = nullptr;
    int mask_words = 0, mask_pool_used = 0;
    int h_mask_id0 = -1, h_cap0 = 0; // async-copy sources (must outlive copy)
    static constexpr int MASK_POOL_CAP = 512;
    // GDN state as 7 physical buffers with a cyclic role permutation (P12b: 6
    // for maxd=5; maxd6: 7 for the ladder ceiling 6). role r (0=primary,
    // 1..6 = post-b..post-g) -> physical (r+perm)%7. accept n tokens ->
    // perm += n-1 (mod 7). One captured graph per perm. Shallower ceilings use
    // only a role prefix (a bitwise-identical subset). Invariant: role 0 = the
    // last-committed state.
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
    // path silently runs a stale graph). perm is mod-6 (6 GDN state buffers), so
    // every spec/gated set below is [.. ][perm=0..5]. Two NON-spec single-token
    // graphs live at the top of the struct: `graph_exec` (plain greedy forward,
    // step_free fallback) and `sample_graph` (plain temp>0 forward+sample, the
    // non-spec sampled loop). The perm-indexed spec/gated sets and their callers:
    //
    //   spec_graph[6]              monolithic UNGATED GREEDY round (draft to
    //                              gate_maxd + width-5 verify, one graph).
    //                              -> spec_round, ungated branch (Q27_PMIN unset,
    //                                 unconstrained). The default greedy path.
    //   spec_sample_graph[6]       monolithic UNGATED SAMPLED round.
    //                              -> spec_sample_round, ungated branch. Default
    //                                 sampled path.
    //   verify_graph_w[7][6]       per-width GREEDY verify, [W=1..6][perm].
    //                              -> gated greedy round (spec_round), both
    //                                 Q27_DEXIT on and off.
    //   verify_sample_graph_w[6][6] per-width SAMPLED verify, [W=2..5][perm].
    //                              -> gated sampled round (spec_sample_round),
    //                                 both Q27_DEXIT on and off.
    //   draft_step_graph[5][6]     per-draft-STEP graphs, [step=0..gate_maxd-1].
    //                              -> the early-exit loop in BOTH gated rounds
    //                                 (default when Q27_DEXIT on). Launched one
    //                                 step at a time; concatenated back-to-back
    //                                 they reproduce the monolithic draft exactly.
    //   draft_graph[6]             monolithic depth-gate_maxd draft (P11 split).
    //                              -> P11 constrained-tool path; AND the
    //                                 Q27_DEXIT=0 monolithic-draft A/B fallback
    //                                 (greedy + sampled).
    //   draft_graph_lo[6]          monolithic DEPTH-4 draft; captured only when
    //                              gate_maxd==5 (auto or fixed Q27_MAXD=5).
    //                              -> constrained-tool path under auto; the
    //                                 Q27_DEXIT=0 depth-4 fallback (greedy auto
    //                                 md_used==4; sampled gate_maxd==5).
    //   verify_graph[6]            monolithic WIDTH-5 verify (P11 split).
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
    cudaGraphExec_t spec_graph[8] = {}; // maxd7: perm is mod-8 (8 GDN state buffers)
    // P11: split draft/verify graphs for the constrained tool path
    cudaGraphExec_t draft_graph[8] = {};
    cudaGraphExec_t verify_graph[8] = {};
    // P12 confidence-gated depth: one verify graph per width W (index [W][perm],
    // W in 1..5). spec_round drafts width-5, reads the 4 draft margins, computes
    // cap = leading run of margin >= pmin_theta, launches verify_graph_w[cap+1].
    // Greedy tokens are width-invariant (lanes are independent grid indices), so
    // this changes only round count + verify width, never the emitted sequence.
    cudaGraphExec_t verify_graph_w[9][8] = {}; // [W=1..8][perm=0..7]
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
    // loop, server n_max clamps). A full-width verify round launched at P
    // writes attention-KV rows P+1..P+gate_maxd+1 and MTP-KV rows
    // P+1..P+gate_maxd, so a round may only launch while
    // P + ctx_round_reserve() <= max_ctx. Review 2026-07-09 P0 #1: the
    // depth-5-era hardcoded 7 (and the servers' -6) overran the caches by up
    // to 2 rows when gate_maxd reached 6/7.
    int ctx_round_reserve() const { return gate_maxd + 2; }
    // P13 adaptive maxd (Q27_MAXD=auto): float the draft-depth ceiling per stream
    // between 4 and 5 from realized acceptance, so agentic streaks get depth-5
    // (+2.6%) while prose stays depth-4 (no -8% draft tax) -- automatically, per
    // stream, with no env retune. Sits on the Q27_PMIN gate (no-op without it).
    // Start shallow; promote 4->5 when depth-4 rounds saturate the ceiling often
    // enough (sat_ema), demote 5->4 when the 5th lane stops paying (yield_ema).
    // The ceiling changes round grouping / draft depth / verify width only -- never
    // the emitted sequence (greedy is width-invariant), so decode stays bitwise.
    bool maxd_auto = false;
    cudaGraphExec_t draft_graph_lo[8] = {}; // depth-4 draft (auto mode only)
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
    long gate_cap_hist[8] = {}; // [cap 0..7]
    long gate_n_hist[9] = {};   // [n 1..8]; index 0 unused
    // acceptance-gate Phase 0: per-draft-lane conditional acceptance on gated
    // rounds. Lane j (1..gate_maxd) FIRED iff cap >= j; ACCEPTED iff n >= j+1.
    // Gives the live yields p(acc_j | fired_j) that the two marginals above
    // cannot reconstruct (docs/acceptance-gate-design.md).
    long gate_lane_fired[8] = {}, gate_lane_acc[8] = {}; // [j 1..7]; index 0 unused
    // Phase 2 (sampling): 2nd fused perm set -- identical draft half, sampled
    // (rejection) verify tail. Captured only when the sampler kernels are warm.
    cudaGraphExec_t spec_sample_graph[8] = {};
    // P14: per-width sampled verify graphs (sampled analog of verify_graph_w).
    // [W=2..5][perm=0..5]; the sampled+gated round drafts depth-4, reads the 4
    // draft margins, caps the accept walk at W-1, and launches this at width W.
    cudaGraphExec_t verify_sample_graph_w[6][8] = {}; // [W<=5][perm 0..7] (sampled ceiling stays 4)
    // P14 draft early-exit: one graph per draft STEP (k=0..gate_maxd-1), so the
    // gated rounds can stop drafting at the first sub-theta margin (llama's
    // p_min stops DRAFTING; the P12 gate only narrowed verify). Steps 0..k
    // launched back-to-back on stm reproduce the monolithic draft graph's
    // kernel sequence exactly (see spec_draft_step_launches). Q27_DEXIT=0
    // restores the monolithic draft (A/B lever); default ON when gated.
    cudaGraphExec_t draft_step_graph[7][8] = {}; // [step 0..6][perm 0..7]
    bool dexit_on = true; // Q27_DEXIT (only reached when pmin_theta > 0)
    bool tool_split_active = false; // set by set_tool_constraint when constraining
    float* SBuf(int il, int role) {
        int ph = (role + perm) % 8;
        return ph == 0 ? S[il]
               : ph == 1 ? S_spare[il]
               : ph == 2 ? S_spare2[il]
               : ph == 3 ? S_spare3[il]
               : ph == 4 ? S_spare4[il]
               : ph == 5 ? S_spare5[il]
               : ph == 6 ? S_spare6[il]
                         : S_spare7[il];
    }
    float* RBuf(int il, int role) {
        int ph = (role + perm) % 8;
        return ph == 0 ? conv_ring[il]
               : ph == 1 ? ring_spare[il]
               : ph == 2 ? ring_spare2[il]
               : ph == 3 ? ring_spare3[il]
               : ph == 4 ? ring_spare4[il]
               : ph == 5 ? ring_spare5[il]
               : ph == 6 ? ring_spare6[il]
                         : ring_spare7[il];
    }
    q27k::XQuant xq;
    // layer state
    float* conv_ring[N_LAYER];
    float* S[N_LAYER];
    // P2: attention + MTP KV caches, fp16 by default; fp8 E4M3 when Q27_KV=fp8
    // (34 vs 68 KB/token). Same [pos][kv_head][head_dim] element layout, only
    // the element size changes. NOT lossless -- opt-in, tolerance-gated.
    bool kv_fp8 = false;
    size_t kv_esz() const { return kv_fp8 ? 1 : 2; }
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
        if (kv_fp8) fprintf(stderr, "KV cache: fp8 E4M3 (opt-in, 34 KB/token)\n");
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
        A((void**)&scratch, 8 * (size_t)N_HEAD * q27k::FD_MAXNS * q27k::FD_ST * 4); // maxd7: 8 lanes
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
        A(&mtp_k, (size_t)max_ctx * N_KV * HEAD_DIM * kv_esz());
        A(&mtp_v, (size_t)max_ctx * N_KV * HEAD_DIM * kv_esz());
        A((void**)&d_pos_m, 4); A((void**)&d_draft, 4);
        A((void**)&h_b, N_EMBD * 4); A((void**)&x1_b, N_EMBD * 4); A((void**)&y_b, N_EMBD * 4);
        A((void**)&qg_b, 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_b, N_KV * HEAD_DIM * 4); A((void**)&vbuf_b, N_KV * HEAD_DIM * 4);
        A((void**)&attnout_b, N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_b, GDN_CH * 4); A((void**)&convout_b, GDN_CH * 4);
        A((void**)&z_b, GDN_V * 4);
        A((void**)&alpha_b, GDN_HEADS * 4); A((void**)&betar_b, GDN_HEADS * 4);
        A((void**)&g_b, GDN_HEADS * 4); A((void**)&beta_b, GDN_HEADS * 4);
        A((void**)&o_b, GDN_V * 4); A((void**)&og_b, GDN_V * 4);
        A((void**)&ffn_g_b, N_FFN * 4); A((void**)&ffn_u_b, N_FFN * 4);
        A((void**)&logits2, 8 * (size_t)VOCAB * 4); // maxd7: 8 verify lanes
        mask_words = (VOCAB + 31) / 32;
        A((void**)&d_mask_pool, (size_t)MASK_POOL_CAP * mask_words * 4);
        A((void**)&d_mask_ids, 8 * 4);
        if (const char* ce = getenv("Q27_CKPT_INTERVAL")) ckpt_interval = atoi(ce);
        if (const char* cs = getenv("Q27_CKPT_SLOTS")) ckpt_slots = std::max(1, atoi(cs));
        A((void**)&d_accept_cap, 4);
        CUDA_CHECK(cudaMemset(d_mask_ids, 0xFF, 8 * 4)); // all -1 = unconstrained
        CUDA_CHECK(cudaMemset(d_accept_cap, 0, 4));
        A((void**)&y2big, 2 * (size_t)N_FFN * 4);
        xq2[0] = q27k::xquant_alloc(N_FFN);
        xq2[1] = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_a, 4); A((void**)&d_pos_b, 4);
        A((void**)&d_va, 4); A((void**)&d_vb, 4);
        A((void**)&h_c, N_EMBD * 4); A((void**)&x1_c, N_EMBD * 4); A((void**)&y_c, N_EMBD * 4);
        A((void**)&qg_c, 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_c, N_KV * HEAD_DIM * 4); A((void**)&vbuf_c, N_KV * HEAD_DIM * 4);
        A((void**)&attnout_c, N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_c, GDN_CH * 4); A((void**)&convout_c, GDN_CH * 4);
        A((void**)&z_c, GDN_V * 4);
        A((void**)&alpha_c, GDN_HEADS * 4); A((void**)&betar_c, GDN_HEADS * 4);
        A((void**)&g_c, GDN_HEADS * 4); A((void**)&beta_c, GDN_HEADS * 4);
        A((void**)&o_c, GDN_V * 4); A((void**)&og_c, GDN_V * 4);
        A((void**)&ffn_g_c, N_FFN * 4); A((void**)&ffn_u_c, N_FFN * 4);
        A((void**)&h_next2, N_EMBD * 4);
        xqC = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_c, 4); A((void**)&d_pos_m2, 4); A((void**)&d_draft2, 4);
        A((void**)&d_vc, 4);
        A((void**)&h_d, N_EMBD * 4); A((void**)&x1_d, N_EMBD * 4); A((void**)&y_d, N_EMBD * 4);
        A((void**)&qg_d, 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_d, N_KV * HEAD_DIM * 4); A((void**)&vbuf_d, N_KV * HEAD_DIM * 4);
        A((void**)&attnout_d, N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_d, GDN_CH * 4); A((void**)&convout_d, GDN_CH * 4);
        A((void**)&z_d, GDN_V * 4);
        A((void**)&alpha_d, GDN_HEADS * 4); A((void**)&betar_d, GDN_HEADS * 4);
        A((void**)&g_d, GDN_HEADS * 4); A((void**)&beta_d, GDN_HEADS * 4);
        A((void**)&o_d, GDN_V * 4); A((void**)&og_d, GDN_V * 4);
        A((void**)&ffn_g_d, N_FFN * 4); A((void**)&ffn_u_d, N_FFN * 4);
        A((void**)&h_next3, N_EMBD * 4);
        xqD = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_d, 4); A((void**)&d_pos_m3, 4); A((void**)&d_draft3, 4);
        A((void**)&d_vd, 4);
        A((void**)&h_e, N_EMBD * 4); A((void**)&x1_e, N_EMBD * 4); A((void**)&y_e, N_EMBD * 4);
        A((void**)&qg_e, 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_e, N_KV * HEAD_DIM * 4); A((void**)&vbuf_e, N_KV * HEAD_DIM * 4);
        A((void**)&attnout_e, N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_e, GDN_CH * 4); A((void**)&convout_e, GDN_CH * 4);
        A((void**)&z_e, GDN_V * 4);
        A((void**)&alpha_e, GDN_HEADS * 4); A((void**)&betar_e, GDN_HEADS * 4);
        A((void**)&g_e, GDN_HEADS * 4); A((void**)&beta_e, GDN_HEADS * 4);
        A((void**)&o_e, GDN_V * 4); A((void**)&og_e, GDN_V * 4);
        A((void**)&ffn_g_e, N_FFN * 4); A((void**)&ffn_u_e, N_FFN * 4);
        A((void**)&h_next4, N_EMBD * 4);
        xqE = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_e, 4); A((void**)&d_pos_m4, 4); A((void**)&d_draft4, 4);
        A((void**)&d_ve, 4);
        // depth-5 lane (f), P12b
        A((void**)&h_f, N_EMBD * 4); A((void**)&x1_f, N_EMBD * 4); A((void**)&y_f, N_EMBD * 4);
        A((void**)&qg_f, 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_f, N_KV * HEAD_DIM * 4); A((void**)&vbuf_f, N_KV * HEAD_DIM * 4);
        A((void**)&attnout_f, N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_f, GDN_CH * 4); A((void**)&convout_f, GDN_CH * 4);
        A((void**)&z_f, GDN_V * 4);
        A((void**)&alpha_f, GDN_HEADS * 4); A((void**)&betar_f, GDN_HEADS * 4);
        A((void**)&g_f, GDN_HEADS * 4); A((void**)&beta_f, GDN_HEADS * 4);
        A((void**)&o_f, GDN_V * 4); A((void**)&og_f, GDN_V * 4);
        A((void**)&ffn_g_f, N_FFN * 4); A((void**)&ffn_u_f, N_FFN * 4);
        A((void**)&h_next5, N_EMBD * 4);
        xqF = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_f, 4); A((void**)&d_pos_m5, 4); A((void**)&d_draft5, 4);
        A((void**)&d_vf, 4);
        // depth-6 lane (g), maxd6
        A((void**)&h_g, N_EMBD * 4); A((void**)&x1_g, N_EMBD * 4); A((void**)&y_g, N_EMBD * 4);
        A((void**)&qg_g, 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_g, N_KV * HEAD_DIM * 4); A((void**)&vbuf_g, N_KV * HEAD_DIM * 4);
        A((void**)&attnout_g, N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_g, GDN_CH * 4); A((void**)&convout_g, GDN_CH * 4);
        A((void**)&z_g, GDN_V * 4);
        A((void**)&alpha_g, GDN_HEADS * 4); A((void**)&betar_g, GDN_HEADS * 4);
        A((void**)&g_g, GDN_HEADS * 4); A((void**)&beta_g, GDN_HEADS * 4);
        A((void**)&o_g, GDN_V * 4); A((void**)&og_g, GDN_V * 4);
        A((void**)&ffn_g_g, N_FFN * 4); A((void**)&ffn_u_g, N_FFN * 4);
        A((void**)&h_next6, N_EMBD * 4);
        xqG = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_g, 4); A((void**)&d_pos_m6, 4); A((void**)&d_draft6, 4);
        A((void**)&d_vg, 4);
        // depth-7 lane (h), maxd7
        A((void**)&h_h, N_EMBD * 4); A((void**)&x1_h, N_EMBD * 4); A((void**)&y_h, N_EMBD * 4);
        A((void**)&qg_h, 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_h, N_KV * HEAD_DIM * 4); A((void**)&vbuf_h, N_KV * HEAD_DIM * 4);
        A((void**)&attnout_h, N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_h, GDN_CH * 4); A((void**)&convout_h, GDN_CH * 4);
        A((void**)&z_h, GDN_V * 4);
        A((void**)&alpha_h, GDN_HEADS * 4); A((void**)&betar_h, GDN_HEADS * 4);
        A((void**)&g_h, GDN_HEADS * 4); A((void**)&beta_h, GDN_HEADS * 4);
        A((void**)&o_h, GDN_V * 4); A((void**)&og_h, GDN_V * 4);
        A((void**)&ffn_g_h, N_FFN * 4); A((void**)&ffn_u_h, N_FFN * 4);
        A((void**)&h_next7, N_EMBD * 4);
        xqH = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_h, 4); A((void**)&d_pos_m7, 4); A((void**)&d_draft7, 4);
        A((void**)&d_vh, 4);
        A((void**)&d_P, 4); A((void**)&d_outcome, 40); // maxd7: {n, t1, dr1..dr7, pending}
        CUDA_CHECK(cudaMemset(mtp_k, 0, (size_t)max_ctx * N_KV * HEAD_DIM * kv_esz()));
        CUDA_CHECK(cudaMemset(mtp_v, 0, (size_t)max_ctx * N_KV * HEAD_DIM * kv_esz()));
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
                A(&k, (size_t)max_ctx * N_KV * HEAD_DIM * kv_esz());
                A(&v, (size_t)max_ctx * N_KV * HEAD_DIM * kv_esz());
                kcache.push_back(k); vcache.push_back(v);
                attn_cache_idx.push_back(cache_slot++);
                conv_ring[il] = nullptr; S[il] = nullptr;
            } else {
                A((void**)&conv_ring[il], 3 * GDN_CH * 4);
                CUDA_CHECK(cudaMemset(conv_ring[il], 0, 3 * GDN_CH * 4));
                A((void**)&S[il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4);
                CUDA_CHECK(cudaMemset(S[il], 0, (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4));
                A((void**)&S_spare[il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4);
                A((void**)&ring_spare[il], 3 * GDN_CH * 4);
                A((void**)&S_spare2[il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4);
                A((void**)&ring_spare2[il], 3 * GDN_CH * 4);
                A((void**)&S_spare3[il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4);
                A((void**)&ring_spare3[il], 3 * GDN_CH * 4);
                A((void**)&S_spare4[il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4);
                A((void**)&ring_spare4[il], 3 * GDN_CH * 4);
                A((void**)&S_spare5[il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4); // P12b
                A((void**)&ring_spare5[il], 3 * GDN_CH * 4);
                // spare sets 6/7 are allocated at EVERY gate_maxd (review
                // 2026-07-09, accepted tradeoff ~157MB each): the perm
                // rotation is uniformly mod-8 ((role+perm)%8), so all 8 sets
                // enter rotation even at shallow ceilings. A width-dependent
                // modulus would save the VRAM but complicate the
                // refinish_round rewind (the P15-hardened state machinery).
                A((void**)&S_spare6[il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4); // maxd6
                A((void**)&ring_spare6[il], 3 * GDN_CH * 4);
                A((void**)&S_spare7[il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4); // maxd7
                A((void**)&ring_spare7[il], 3 * GDN_CH * 4);
                // 9 S-buffers (main + 7 spares + snap) + 9 rings per GDN layer
                // -- keep in sync when adding sets; server slot admission
                // sizes its floor from this (review 2026-07-09: the old
                // hardcoded "5 sets ~3GB" predated maxd6/7)
                gdn_state_bytes += 9ull * ((size_t)GDN_HEADS * GDN_DIM * GDN_DIM + 3 * GDN_CH) * 4;
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
        q27k::kv_store(kbuf, vbuf, kc, vc, pos_src, N_KV * HEAD_DIM, stm, kv_fp8);
        q27k::attn_decode(qg, 2 * HEAD_DIM, kc, vc, attnout, scratch, pos_src,
                          max_ctx, N_HEAD, N_KV, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM), stm,
                          kv_fp8);
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
    void token_launches() {
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
    void qx5(const float* xa, const float* xb, const float* xc, const float* xd, const float* xe,
             const float* xf, const float* xg, const float* xh, int cols) {
        q27k::XQ3 q{{xq2[0], xq2[1], xqC, xqD, xqE, xqF, xqG, xqH}};
        q27k::quantize3({{xa, xb, xc, xd, xe, xf, xg, xh}}, cols, q, stm, vw);
    }
    void mm5(const DevTensor& w, float* out_a, float* out_b, float* out_c, float* out_d,
             float* out_e, float* out_f, float* out_g, float* out_h) {
        q27k::XQuant qs[8] = {xq2[0], xq2[1], xqC, xqD, xqE, xqF, xqG, xqH};
        float* const ys[8] = {out_a, out_b, out_c, out_d, out_e, out_f, out_g, out_h};
        if (w.dtype == DType::Q4_G64)
            q27k::gemv_q4_n((const uint8_t*)w.data, (const __half*)w.scales, qs, vw, ys, w.rows,
                            w.cols, stm);
        else
            q27k::gemv_q8_n((const int8_t*)w.data, (const __half*)w.scales, qs, vw, ys, w.rows,
                            w.cols, stm);
    }

    void gdn_pair(int il) {
        const float eps = EPS;
        qx5(x1, x1_b, x1_c, x1_d, x1_e, x1_f, x1_g, x1_h, N_EMBD);
        mm5(T(il, "attn_qkv.weight"), qkv, qkv_b, qkv_c, qkv_d, qkv_e, qkv_f, qkv_g, qkv_h);
        mm5(T(il, "attn_gate.weight"), z, z_b, z_c, z_d, z_e, z_f, z_g, z_h);
        q27k::gemv_f16_3((const __half*)T(il, "ssm_alpha.weight").data,
                         {{x1, x1_b, x1_c, x1_d, x1_e, x1_f, x1_g, x1_h}},
                         {{alpha, alpha_b, alpha_c, alpha_d, alpha_e, alpha_f, alpha_g, alpha_h}}, GDN_HEADS,
                         N_EMBD, stm, vw);
        q27k::gemv_f16_3((const __half*)T(il, "ssm_beta.weight").data,
                         {{x1, x1_b, x1_c, x1_d, x1_e, x1_f, x1_g, x1_h}},
                         {{betar, betar_b, betar_c, betar_d, betar_e, betar_f, betar_g, betar_h}}, GDN_HEADS,
                         N_EMBD, stm, vw);
        const float* sa = (const float*)T(il, "ssm_a").data;
        const float* sdt = (const float*)T(il, "ssm_dt.bias").data;
        q27k::gdn_gates3({{alpha, alpha_b, alpha_c, alpha_d, alpha_e, alpha_f, alpha_g, alpha_h}},
                         {{betar, betar_b, betar_c, betar_d, betar_e, betar_f, betar_g, betar_h}}, sa, sdt,
                         {{g, g_b, g_c, g_d, g_e, g_f, g_g, g_h}},
                         {{beta, beta_b, beta_c, beta_d, beta_e, beta_f, beta_g, beta_h}}, GDN_HEADS, stm, vw);
        const float* cw = (const float*)T(il, "ssm_conv1d.weight").data;
        // P12: per-lane recurrent chain -- role k reads role k-1 (written fresh
        // earlier this round) and writes role k. Only lanes < vw are live; a
        // width-vw graph skips the rest, leaving their (never-read) role buffers
        // untouched. Lane a (role 0, the pending token) always runs.
        q27k::conv_step(RBuf(il, 0), RBuf(il, 0), qkv, cw, convout, GDN_CH, stm);   // a
        if (vw > 1) q27k::conv_step(RBuf(il, 0), RBuf(il, 1), qkv_b, cw, convout_b, GDN_CH, stm);
        if (vw > 2) q27k::conv_step(RBuf(il, 1), RBuf(il, 2), qkv_c, cw, convout_c, GDN_CH, stm);
        if (vw > 3) q27k::conv_step(RBuf(il, 2), RBuf(il, 3), qkv_d, cw, convout_d, GDN_CH, stm);
        if (vw > 4) q27k::conv_step(RBuf(il, 3), RBuf(il, 4), qkv_e, cw, convout_e, GDN_CH, stm);
        if (vw > 5) q27k::conv_step(RBuf(il, 4), RBuf(il, 5), qkv_f, cw, convout_f, GDN_CH, stm);
        if (vw > 6) q27k::conv_step(RBuf(il, 5), RBuf(il, 6), qkv_g, cw, convout_g, GDN_CH, stm);
        if (vw > 7) q27k::conv_step(RBuf(il, 6), RBuf(il, 7), qkv_h, cw, convout_h, GDN_CH, stm);
        // q||k are contiguous (offsets 0 and 2048): 32 heads in one merged call
        q27k::l2norm3({{convout, convout_b, convout_c, convout_d, convout_e, convout_f, convout_g, convout_h}}, 32,
                      GDN_DIM, eps, stm, vw);
        q27k::delta_step(SBuf(il, 0), SBuf(il, 0), convout, g, beta, o, stm);          // a
        if (vw > 1) q27k::delta_step(SBuf(il, 0), SBuf(il, 1), convout_b, g_b, beta_b, o_b, stm);
        if (vw > 2) q27k::delta_step(SBuf(il, 1), SBuf(il, 2), convout_c, g_c, beta_c, o_c, stm);
        if (vw > 3) q27k::delta_step(SBuf(il, 2), SBuf(il, 3), convout_d, g_d, beta_d, o_d, stm);
        if (vw > 4) q27k::delta_step(SBuf(il, 3), SBuf(il, 4), convout_e, g_e, beta_e, o_e, stm);
        if (vw > 5) q27k::delta_step(SBuf(il, 4), SBuf(il, 5), convout_f, g_f, beta_f, o_f, stm);
        if (vw > 6) q27k::delta_step(SBuf(il, 5), SBuf(il, 6), convout_g, g_g, beta_g, o_g, stm);
        if (vw > 7) q27k::delta_step(SBuf(il, 6), SBuf(il, 7), convout_h, g_h, beta_h, o_h, stm);
        const float* nw = (const float*)T(il, "ssm_norm.weight").data;
        q27k::gated_norm3({{o, o_b, o_c, o_d, o_e, o_f, o_g, o_h}}, nw, {{z, z_b, z_c, z_d, z_e, z_f, z_g, z_h}},
                          {{og, og_b, og_c, og_d, og_e, og_f, og_g, og_h}}, GDN_HEADS, GDN_DIM, eps, stm, vw);
        qx5(og, og_b, og_c, og_d, og_e, og_f, og_g, og_h, GDN_V);
        mm5(T(il, "ssm_out.weight"), y, y_b, y_c, y_d, y_e, y_f, y_g, y_h);
    }

    void attn_pair(int il) {
        int ci = attn_cache_idx[il];
        qx5(x1, x1_b, x1_c, x1_d, x1_e, x1_f, x1_g, x1_h, N_EMBD);
        mm5(T(il, "attn_q.weight"), qg, qg_b, qg_c, qg_d, qg_e, qg_f, qg_g, qg_h);
        const float* qn = (const float*)T(il, "attn_q_norm.weight").data;
        const float* kn = (const float*)T(il, "attn_k_norm.weight").data;
        q27k::rmsnorm_heads(qg, qn, qg, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        if (vw > 1) q27k::rmsnorm_heads(qg_b, qn, qg_b, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        if (vw > 2) q27k::rmsnorm_heads(qg_c, qn, qg_c, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        if (vw > 3) q27k::rmsnorm_heads(qg_d, qn, qg_d, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        if (vw > 4) q27k::rmsnorm_heads(qg_e, qn, qg_e, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        if (vw > 5) q27k::rmsnorm_heads(qg_f, qn, qg_f, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        if (vw > 6) q27k::rmsnorm_heads(qg_g, qn, qg_g, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        if (vw > 7) q27k::rmsnorm_heads(qg_h, qn, qg_h, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        mm5(T(il, "attn_k.weight"), kbuf, kbuf_b, kbuf_c, kbuf_d, kbuf_e, kbuf_f, kbuf_g, kbuf_h);
        q27k::rmsnorm_heads(kbuf, kn, kbuf, N_KV, HEAD_DIM, HEAD_DIM, EPS, stm);
        if (vw > 1) q27k::rmsnorm_heads(kbuf_b, kn, kbuf_b, N_KV, HEAD_DIM, HEAD_DIM, EPS, stm);
        if (vw > 2) q27k::rmsnorm_heads(kbuf_c, kn, kbuf_c, N_KV, HEAD_DIM, HEAD_DIM, EPS, stm);
        if (vw > 3) q27k::rmsnorm_heads(kbuf_d, kn, kbuf_d, N_KV, HEAD_DIM, HEAD_DIM, EPS, stm);
        if (vw > 4) q27k::rmsnorm_heads(kbuf_e, kn, kbuf_e, N_KV, HEAD_DIM, HEAD_DIM, EPS, stm);
        if (vw > 5) q27k::rmsnorm_heads(kbuf_f, kn, kbuf_f, N_KV, HEAD_DIM, HEAD_DIM, EPS, stm);
        if (vw > 6) q27k::rmsnorm_heads(kbuf_g, kn, kbuf_g, N_KV, HEAD_DIM, HEAD_DIM, EPS, stm);
        if (vw > 7) q27k::rmsnorm_heads(kbuf_h, kn, kbuf_h, N_KV, HEAD_DIM, HEAD_DIM, EPS, stm);
        mm5(T(il, "attn_v.weight"), vbuf, vbuf_b, vbuf_c, vbuf_d, vbuf_e, vbuf_f, vbuf_g, vbuf_h);
        q27k::IP3 P{{d_pos_a, d_pos_b, d_pos_c, d_pos_d, d_pos_e, d_pos_f, d_pos_g, d_pos_h}};
        q27k::rope3({{qg, qg_b, qg_c, qg_d, qg_e, qg_f, qg_g, qg_h}}, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, P,
                    FREQ_BASE, stm, vw);
        q27k::rope3({{kbuf, kbuf_b, kbuf_c, kbuf_d, kbuf_e, kbuf_f, kbuf_g, kbuf_h}}, N_KV, HEAD_DIM, N_ROT,
                    HEAD_DIM, P, FREQ_BASE, stm, vw);
        float kq = 1.0f / sqrtf((float)HEAD_DIM);
        // store vw lanes (disjoint slots); each token's attention only reads
        // cache[0 .. its own pos], so later tokens' entries are invisible to earlier ones
        q27k::kv_store3({{kbuf, kbuf_b, kbuf_c, kbuf_d, kbuf_e, kbuf_f, kbuf_g, kbuf_h}},
                        {{vbuf, vbuf_b, vbuf_c, vbuf_d, vbuf_e, vbuf_f, vbuf_g, vbuf_h}}, kcache[ci], vcache[ci],
                        P, N_KV * HEAD_DIM, stm, vw, kv_fp8);
        q27k::attn_decode3({{qg, qg_b, qg_c, qg_d, qg_e, qg_f, qg_g, qg_h}}, 2 * HEAD_DIM, kcache[ci],
                           vcache[ci],
                           {{attnout, attnout_b, attnout_c, attnout_d, attnout_e, attnout_f, attnout_g, attnout_h}},
                           scratch, P, max_ctx, N_HEAD, N_KV, HEAD_DIM, kq, stm, vw, kv_fp8);
        q27k::sigmoid_gate3({{attnout, attnout_b, attnout_c, attnout_d, attnout_e, attnout_f, attnout_g, attnout_h}},
                            {{qg, qg_b, qg_c, qg_d, qg_e, qg_f, qg_g, qg_h}}, N_HEAD, HEAD_DIM, stm, vw);
        qx5(attnout, attnout_b, attnout_c, attnout_d, attnout_e, attnout_f, attnout_g, attnout_h, N_HEAD * HEAD_DIM);
        mm5(T(il, "attn_output.weight"), y, y_b, y_c, y_d, y_e, y_f, y_g, y_h);
    }

    void ffn_pair(int il) {
        qx5(x1, x1_b, x1_c, x1_d, x1_e, x1_f, x1_g, x1_h, N_EMBD);
        mm5(T(il, "ffn_gate.weight"), ffn_g, ffn_g_b, ffn_g_c, ffn_g_d, ffn_g_e, ffn_g_f, ffn_g_g, ffn_g_h);
        mm5(T(il, "ffn_up.weight"), ffn_u, ffn_u_b, ffn_u_c, ffn_u_d, ffn_u_e, ffn_u_f, ffn_u_g, ffn_u_h);
        q27k::silu_mul3({{ffn_g, ffn_g_b, ffn_g_c, ffn_g_d, ffn_g_e, ffn_g_f, ffn_g_g, ffn_g_h}},
                        {{ffn_u, ffn_u_b, ffn_u_c, ffn_u_d, ffn_u_e, ffn_u_f, ffn_u_g, ffn_u_h}}, N_FFN, stm, vw);
        qx5(ffn_g, ffn_g_b, ffn_g_c, ffn_g_d, ffn_g_e, ffn_g_f, ffn_g_g, ffn_g_h, N_FFN);
        mm5(T(il, "ffn_down.weight"), y, y_b, y_c, y_d, y_e, y_f, y_g, y_h);
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
    void spec_draft_step_launches(int k) {
        if (k == 0) {
            q27k::prep_round(d_P, d_token, d_pos_a, d_pos_b, d_pos_c, d_pos_d, d_pos_e, d_pos_f,
                             d_pos_g, d_pos_h, d_pos_m, d_pos_m2, d_pos_m3, d_pos_m4, d_pos_m5,
                             d_pos_m6, d_pos_m7, d_outcome, stm);
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
    void spec_verify_forward() {
        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed3((const int8_t*)emb.data, (const __half*)emb.scales,
                     {{d_token, d_draft, d_draft2, d_draft3, d_draft4, d_draft5, d_draft6,
                       d_draft7}},
                     N_EMBD, {{h, h_b, h_c, h_d, h_e, h_f, h_g, h_h}}, stm, vw);
        q27k::CP3 Hc{{h, h_b, h_c, h_d, h_e, h_f, h_g, h_h}},
            Yc{{y, y_b, y_c, y_d, y_e, y_f, y_g, y_h}};
        q27k::P3 Hm{{h, h_b, h_c, h_d, h_e, h_f, h_g, h_h}},
            X1m{{x1, x1_b, x1_c, x1_d, x1_e, x1_f, x1_g, x1_h}};
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
        qx5(x1, x1_b, x1_c, x1_d, x1_e, x1_f, x1_g, x1_h, N_EMBD);
        const char* vhead = (fast_head && dm.model_has("output_q4.weight")) ? "output_q4.weight"
                                                                             : "output.weight";
        mm5(dm.get(vhead), logits2, logits2 + VOCAB, logits2 + 2 * (size_t)VOCAB,
            logits2 + 3 * (size_t)VOCAB, logits2 + 4 * (size_t)VOCAB, logits2 + 5 * (size_t)VOCAB,
            logits2 + 6 * (size_t)VOCAB, logits2 + 7 * (size_t)VOCAB);
    }

    // P11: verify half -- batch-5 forward, masked argmax per lane, finish_round.
    void spec_verify_launches() {
        spec_verify_forward();
        // P7: slot 0 (the post-pending lane) is the constrained one; slots
        // 1-4 keep id -1 (v1 caps acceptance in-grammar instead of chasing
        // draft-dependent states the host cannot know pre-launch)
        q27k::argmax_masked(logits2, VOCAB, d_mask_pool, mask_words, d_mask_ids, 0, d_va,
                            d_amax, stm);
        if (vw > 1)
            q27k::argmax_masked(logits2 + VOCAB, VOCAB, d_mask_pool, mask_words, d_mask_ids, 1,
                                d_vb, d_amax, stm);
        if (vw > 2)
            q27k::argmax_masked(logits2 + 2 * (size_t)VOCAB, VOCAB, d_mask_pool, mask_words,
                                d_mask_ids, 2, d_vc, d_amax, stm);
        if (vw > 3)
            q27k::argmax_masked(logits2 + 3 * (size_t)VOCAB, VOCAB, d_mask_pool, mask_words,
                                d_mask_ids, 3, d_vd, d_amax, stm);
        if (vw > 4)
            q27k::argmax_masked(logits2 + 4 * (size_t)VOCAB, VOCAB, d_mask_pool, mask_words,
                                d_mask_ids, 4, d_ve, d_amax, stm);
        if (vw > 5)
            q27k::argmax_masked(logits2 + 5 * (size_t)VOCAB, VOCAB, d_mask_pool, mask_words,
                                d_mask_ids, 5, d_vf, d_amax, stm);
        if (vw > 6)
            q27k::argmax_masked(logits2 + 6 * (size_t)VOCAB, VOCAB, d_mask_pool, mask_words,
                                d_mask_ids, 6, d_vg, d_amax, stm);
        if (vw > 7)
            q27k::argmax_masked(logits2 + 7 * (size_t)VOCAB, VOCAB, d_mask_pool, mask_words,
                                d_mask_ids, 7, d_vh, d_amax, stm);
        // P12: a width-vw verify computed columns 0..vw-1; cap acceptance at vw-1
        // drafts so finish never commits an uncomputed lane. vw=5 => max_draft=4.
        q27k::finish_round(d_P, d_token, d_draft, d_draft2, d_draft3, d_draft4, d_draft5,
                           d_draft6, d_draft7, d_va, d_vb, d_vc, d_vd, d_ve, d_vf, d_vg, d_vh,
                           x1, x1_b, x1_c, x1_d, x1_e, x1_f, x1_g, x1_h, h_next, d_outcome,
                           N_EMBD, d_accept_cap, vw - 1, stm);
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
                             x1_b, x1_c, x1_d, x1_e, h_next, d_outcome, N_EMBD, stm);
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
        // P13 adaptive-maxd tunables (bench-tunable; defaults from the design)
        if (const char* e = getenv("Q27_MAXD_RESET")) maxd_reset = atoi(e) != 0;
        if (const char* e = getenv("Q27_MAXD_EMA")) dctl.ema_a = (float)atof(e);
        if (const char* e = getenv("Q27_MAXD_HI")) dctl.hi = (float)atof(e);
        if (const char* e = getenv("Q27_MAXD_HI6")) dctl.hi6 = (float)atof(e);
        if (const char* e = getenv("Q27_MAXD_HI7")) dctl.hi7 = (float)atof(e);
        if (const char* e = getenv("Q27_MAXD_FLO7")) dctl.flo7 = (float)atof(e);
        if (const char* e = getenv("Q27_MAXD_FLO6")) dctl.flo6 = (float)atof(e);
        if (const char* e = getenv("Q27_MAXD_LO")) dctl.lo = (float)atof(e);
        int z0 = 0, z1 = 1, z2 = 2, z3 = 3, z4 = 4, z5 = 5, z6 = 6, z7 = 7;
        auto seed_positions = [&]() {
            CUDA_CHECK(cudaMemcpyAsync(d_pos_a, &z0, 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemcpyAsync(d_pos_b, &z1, 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemcpyAsync(d_pos_c, &z2, 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemcpyAsync(d_pos_d, &z3, 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemcpyAsync(d_pos_e, &z4, 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemcpyAsync(d_pos_f, &z5, 4, cudaMemcpyHostToDevice, stm)); // P12b
            CUDA_CHECK(cudaMemcpyAsync(d_pos_g, &z6, 4, cudaMemcpyHostToDevice, stm)); // maxd6
            CUDA_CHECK(cudaMemcpyAsync(d_pos_h, &z7, 4, cudaMemcpyHostToDevice, stm)); // maxd7
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
                    CUDA_CHECK(cudaMemset(S_spare[il], 0, sb));
                    CUDA_CHECK(cudaMemset(conv_ring[il], 0, 3 * GDN_CH * 4));
                    CUDA_CHECK(cudaMemset(ring_spare[il], 0, 3 * GDN_CH * 4));
                    CUDA_CHECK(cudaMemset(S_spare2[il], 0, sb));
                    CUDA_CHECK(cudaMemset(ring_spare2[il], 0, 3 * GDN_CH * 4));
                    CUDA_CHECK(cudaMemset(S_spare3[il], 0, sb));
                    CUDA_CHECK(cudaMemset(ring_spare3[il], 0, 3 * GDN_CH * 4));
                    CUDA_CHECK(cudaMemset(S_spare4[il], 0, sb));
                    CUDA_CHECK(cudaMemset(ring_spare4[il], 0, 3 * GDN_CH * 4));
                    CUDA_CHECK(cudaMemset(S_spare5[il], 0, sb));           // P12b
                    CUDA_CHECK(cudaMemset(ring_spare5[il], 0, 3 * GDN_CH * 4));
                    CUDA_CHECK(cudaMemset(S_spare6[il], 0, sb));           // maxd6
                    CUDA_CHECK(cudaMemset(ring_spare6[il], 0, 3 * GDN_CH * 4));
                    CUDA_CHECK(cudaMemset(S_spare7[il], 0, sb));           // maxd7
                    CUDA_CHECK(cudaMemset(ring_spare7[il], 0, 3 * GDN_CH * 4));
                }
            CUDA_CHECK(cudaMemset(mtp_k, 0, (size_t)max_ctx * N_KV * HEAD_DIM * kv_esz()));
            CUDA_CHECK(cudaMemset(mtp_v, 0, (size_t)max_ctx * N_KV * HEAD_DIM * kv_esz()));
        };
        // P12b: warm the WIDEST kernels (6-lane verify + 5th draft = distinct
        // gemv<6>/ntok=6 instantiations) so graph capture never triggers a lazy
        // module load. Output is discarded and reset below.
        perm = 0; dmax = gate_maxd; vw = gate_maxd + 1;
        seed_positions();
        spec_round_launches();
        CUDA_CHECK(cudaStreamSynchronize(stm));
        reset_gdn_mtp();
        // capture all 8 cyclic permutations (capture records; does not execute)
        for (int p = 0; p < 8; p++) {
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
        for (int p = 0; p < 8; p++) {
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
        fprintf(stderr,
                "spec graphs captured (8 perms, depth-4; +split D/V; +per-width verify "
                "2..%d%s; +P14 sampled per-width verify 2..5 + per-step draft 0..%d); "
                "Q27_PMIN=%.3f (%s), gate_maxd=%d%s, dexit=%d\n",
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
                gate_cap = cap;
            } else {
                // Q27_DEXIT=0: monolithic gated draft (the pre-P14 A/B baseline).
                cudaGraphExec_t dg =
                    (maxd_auto && md_used == 4) ? draft_graph_lo[perm] : draft_graph[perm];
                CUDA_CHECK(cudaGraphLaunch(dg, stm));
                CUDA_CHECK(cudaMemcpyAsync(h_draft_margin, d_draft_margin, 7 * 4,
                                           cudaMemcpyDeviceToHost, stm));
                CUDA_CHECK(cudaStreamSynchronize(stm));
                int cap = 0;
                while (cap < md_used && h_draft_margin[cap] >= pmin_theta) cap++;
                int W = cap + 1 < 2 ? 2 : cap + 1; // no width-1 gemv; floor at 2
                CUDA_CHECK(cudaGraphLaunch(verify_graph_w[W][perm], stm));
                gate_cap = cap;
            }
        } else {
            CUDA_CHECK(cudaGraphLaunch(spec_graph[perm], stm));
        }
        // maxd7 outcome: [0]=n, [1..8]=up to 8 emitted tokens, [9]=new pending.
        int oc[10];
        CUDA_CHECK(cudaMemcpyAsync(oc, d_outcome, 40, cudaMemcpyDeviceToHost, stm));
        CUDA_CHECK(cudaStreamSynchronize(stm));
        int n = oc[0];
        if (gate_cap >= 0) {
            gate_cap_hist[gate_cap]++; gate_n_hist[n]++;
            for (int j = 1; j <= gate_cap; j++) {
                gate_lane_fired[j]++;
                if (n >= j + 1) gate_lane_acc[j]++;
            }
        }
        // P13 adaptive maxd: fold this round's realized accept into the ceiling
        // (extracted to depthctl.h; semantics + comments live there).
        if (maxd_auto) dctl.update(md_used, gate_cap, n);
        for (int k = 0; k < n; k++) emit[k] = oc[1 + k];
        last_pending = oc[9];
        perm = (perm + (n - 1)) % 8;
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
        perm = (perm + (n - 1)) % 8;
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
    int h_mask_ids5[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
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
        for (int i = 0; i < 8; i++) h_mask_ids5[i] = -1;
        CUDA_CHECK(cudaMemcpyAsync(d_mask_ids, h_mask_ids5, 8 * 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemcpyAsync(d_accept_cap, &h_cap0, 4, cudaMemcpyHostToDevice, stm));
    }
    // P15 engage-lag fix: rewind the JUST-FINISHED greedy spec round from n
    // accepted tokens to m (1 <= m <= n) and re-decide the pending token under
    // the freshly staged slot-0 mask. Everything needed is still resident:
    // per-lane GDN states / conv rings sit in the rotating role buffers (the
    // round "commits" by advancing perm, never by copying -- state-after-lane-
    // (m-1) is old role m-1), lane hiddens in x1..x1_f, lane logits in
    // logits2. KV/MTP rows past the kept position are rewritten by the next
    // round. Must be called BETWEEN rounds (server on_round hook), with the
    // engage mask already staged on this stream so the re-argmax orders after
    // it. Returns the new pending token. The CLI/canonical path never sets
    // on_round, so this code is unreachable there.
    int refinish_round(int m, int n, int P_target) {
        perm = (perm + (m - n) + 8) % 8;
        CUDA_CHECK(cudaMemcpyAsync(d_P, &P_target, 4, cudaMemcpyHostToDevice, stm));
        const float* lanes[8] = {x1, x1_b, x1_c, x1_d, x1_e, x1_f, x1_g, x1_h};
        CUDA_CHECK(cudaMemcpyAsync(h_next, lanes[m - 1], (size_t)N_EMBD * 4,
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
        q27k::kv_store_T(kT, vT, kc, vc, base, KVROW, T, stm, kv_fp8);
        q27k::attn_prefill_T(qgT, 2 * HEAD_DIM, QROW, kc, vc, attnT, N_HEAD * HEAD_DIM, pf_part,
                             base, 0, T, N_HEAD, N_KV, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM),
                             stm, kv_fp8);
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
            if (!c.buf || c.toks.empty() || c.toks.size() > prompt.size() - 1) continue;
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
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                size_t sb = (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4;
                CUDA_CHECK(cudaMemset(S[il], 0, sb));
                CUDA_CHECK(cudaMemset(S_spare[il], 0, sb));
                CUDA_CHECK(cudaMemset(S_spare2[il], 0, sb));
                CUDA_CHECK(cudaMemset(S_spare3[il], 0, sb));
                CUDA_CHECK(cudaMemset(S_spare4[il], 0, sb));
                CUDA_CHECK(cudaMemset(S_spare5[il], 0, sb));              // P12b
                CUDA_CHECK(cudaMemset(conv_ring[il], 0, 3 * GDN_CH * 4));
                CUDA_CHECK(cudaMemset(ring_spare[il], 0, 3 * GDN_CH * 4));
                CUDA_CHECK(cudaMemset(ring_spare2[il], 0, 3 * GDN_CH * 4));
                CUDA_CHECK(cudaMemset(ring_spare3[il], 0, 3 * GDN_CH * 4));
                CUDA_CHECK(cudaMemset(ring_spare4[il], 0, 3 * GDN_CH * 4));
                CUDA_CHECK(cudaMemset(ring_spare5[il], 0, 3 * GDN_CH * 4)); // P12b
            }
        CUDA_CHECK(cudaMemset(mtp_k, 0, (size_t)max_ctx * N_KV * HEAD_DIM * kv_esz()));
        CUDA_CHECK(cudaMemset(mtp_v, 0, (size_t)max_ctx * N_KV * HEAD_DIM * kv_esz()));
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
            int em[8]; // maxd7: spec_round can emit up to 8 tokens (depth-7)
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
};
