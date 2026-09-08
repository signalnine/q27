// DFlash2 drafter runtime (Phase 1 bring-up, eager, numerics-first).
// Design: docs/plans/2026-09-06-dflash2-integration.md. Loads the flat pack
// written by tools/dflash2_pack.py; reuses the engine's gemv/rmsnorm/rope
// primitives; adds the two genuinely new kernels (grouped dynamic causal
// conv, bidirectional sliding-window attention). Top-16 + selector walk run
// on the HOST for bring-up -- the whole path is eager and unoptimized by
// intent (correctness gates first; graphs and kernels are Phase 2).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "kernels.cuh" // XQuant (engine-head reuse)

namespace q27k { struct SampleParams; } // blocks.cuh (sampled selector walk)

namespace q27d2 {

// geometry (z-lab/Qwen3.8-27B-DFlash2 checkpoint; asserts at load)
constexpr int D2_H = 5120, D2_TAPS = 5, D2_TAPD = 25600;
constexpr int D2_NH = 32, D2_NKV = 8, D2_HD = 128;
constexpr int D2_QD = D2_NH * D2_HD, D2_KVD = D2_NKV * D2_HD;
constexpr int D2_I = 17408, D2_V = 248320, D2_LAYERS = 5;
// K is a runtime choice 1..D2_WMAX-1 (checkpoint masks reach 15; the engine
// verify caps width at W_MAX=12). D2_W/D2_K are the recommended defaults.
constexpr int D2_WMAX = 12;
constexpr int D2_W = 8, D2_K = 7, D2_MASK = 248070;
constexpr int D2_WINDOW = 2048, D2_CONVK = 2, D2_CONVG = 320; // groups = H/16
constexpr int D2_RANK = 256, D2_TOPK = 16;
constexpr float D2_EPS = 1e-6f, D2_THETA = 1e7f;

struct D2Tensor {
    void* dev = nullptr;        // device copy (packed data)
    void* dscales = nullptr;    // device group scales (Q4/Q8; nullptr otherwise)
    const void* host = nullptr; // pointer into the pack's host buffer
    int dtype = 0;              // 0 = f16, 1 = f32, 2 = q4_g64, 3 = q8_g128
    int64_t rows = 0, cols = 0, n = 0;
};

struct Dflash2 {
    // ---- weights ----
    std::map<std::string, D2Tensor> w;
    std::vector<char> pack; // host backing (codebooks are walked host-side)
    void load(const char* path);
    const D2Tensor& T(const std::string& name) const { return w.at(name); }
    const __half* f16(const std::string& n) const { return (const __half*)T(n).dev; }
    const float* f32(const std::string& n) const { return (const float*)T(n).dev; }
    // Quantized matmul: quantize the W-column activation once, then the
    // engine's weight-shared int4/int8 gemv (weight read ONCE across columns,
    // vs gemv_f16_3's per-column re-read). act is [W][cols] contiguous, out is
    // [W][rows] contiguous (cols/rows from the tensor shape).
    void mmq(const std::string& name, const float* act, float* out, int W, cudaStream_t st);
    // Split of mmq for shared activations: quantize the W-col activation into
    // dxq once, then mmq_pre reuses it for each weight that reads the SAME
    // activation (q/k/v share one input, gate/up share one). Byte-identical to
    // separate mmq calls; removes redundant quantize launches.
    void quant_act(const float* act, int cols, int W, cudaStream_t st);
    void mmq_pre(const std::string& name, float* out, int W, cudaStream_t st);
    q27k::XQuant dxq[D2_WMAX] = {}; // activation-quant scratch (sized to TAPD)

    // ---- context ring (bring-up: append-only, capacity-capped, no wrap) ----
    int ctx_cap = 0, ctx_n = 0;
    float *ringK[D2_LAYERS] = {}, *ringV[D2_LAYERS] = {}; // [cap][KVD] fp32
    int* d_ring_pos = nullptr;                            // [cap]
    // scratch (sized for the widest block, D2_WMAX rows)
    float *s_fc = nullptr, *s_ct = nullptr;              // [WMAX][H] ingest rows
    float *nx = nullptr, *nh1 = nullptr, *ny0 = nullptr; // [WMAX][H]
    float *ndyn = nullptr;                               // [WMAX][2*CONVK*CONVG]
    float *nq = nullptr, *nk = nullptr, *nv = nullptr;   // [WMAX][QD/KVD]
    float *natt = nullptr, *no = nullptr;                // [WMAX][QD], [WMAX][H]
    float *ngate = nullptr, *nup = nullptr;              // [WMAX][I]
    float *nhf = nullptr;                                // [WMAX-1][H] final-normed mask rows
    float *nlogits = nullptr;                            // [WMAX-1][V]
    float *nhp = nullptr;                                // [WMAX-1][RANK]
    float* maskrow = nullptr;                            // cached mask-token embedding
    int* d_posW = nullptr;                               // noise positions [WMAX]
    int* d_ing_pos = nullptr;                            // ingest positions (chunk)
    // Phase 2 on-device selector: per-position top-16 (values+ids) and the
    // walked path. Proposals stay on device -- the CLI stages them into the
    // engine's d_draft_L slots with D2D copies, no host sync in the round.
    int* d_cand = nullptr;   // [WMAX-1][TOPK]
    float* d_cval = nullptr; // [WMAX-1][TOPK]
    int* d_prop = nullptr;   // [WMAX-1] the walked draft tokens
    // Sampled selector walk (2026-09-07): per position the 16-way proposal
    // distribution q = softmax(E/T) the walk actually drew from, over the
    // d_cand ids of that position. The engine's sparse-q rejection tail reads
    // it (accept min(1,p/q), correct from max(p-q,0)). Written ONLY by the
    // sampled walk; the greedy walk (and graph) never touch it.
    float* d_qrow = nullptr; // [WMAX-1][TOPK]
    const q27k::SampleParams* d_sp = nullptr; // engine's device sampler params (fixed ptr)
    void set_sampler(const q27k::SampleParams* sp) { d_sp = sp; }
    int* d_ctx_n = nullptr;  // device mirror of ctx_n (graph-stable attn)
    float* d_c1v = nullptr;  // top-16 stage-1 candidates [WMAX-1][512*16]
    int* d_c1i = nullptr;
    // Phase 2: engine quantized-head reuse for drafter logits (replaces the
    // 2.5 GB fp16 target.head gemv, ~10.4 -> ~1 ms/round). Numerics shift
    // (Q4/Q8 head vs fp16) -- acceptance impact measured E2E; output tokens
    // are verify-decided and unaffected. Cleared = fp16 pack head.
    const void* ehead_data = nullptr;
    const __half* ehead_scales = nullptr;
    bool ehead_q4 = false;
    q27k::XQuant hxq[D2_WMAX - 1] = {};
    void set_engine_head(const void* data, const __half* scales, bool q4) {
        ehead_data = data;
        ehead_scales = scales;
        ehead_q4 = q4;
    }
    // Engine Q8 embedding reuse (serving): the drafter's anchor/mask embed
    // rows come from the engine's own token_embd (Q8_G128) instead of a
    // packed fp16 target.embed -- saves 2.5 GB of VRAM (= more KV/ctx).
    const int8_t* eembed_data = nullptr;
    const __half* eembed_scales = nullptr;
    int* d_anchor_tok = nullptr;
    int* d_mask_tok = nullptr;
    void set_engine_embed(const int8_t* data, const __half* scales) {
        eembed_data = data;
        eembed_scales = scales;
    }
    void alloc(int cap);

    // append T committed-token context rows. taps: device [T][TAPD] fp32,
    // positions: host absolute positions (uploaded internally). Batched: the
    // fc weight (131 MB fp16) is read ONCE for all T rows. Serving: the ring
    // slides -- when it would overflow, the oldest rows past the sliding
    // window are dropped (lossless: the drafter attends only D2_WINDOW back).
    void ingest(const float* d_taps, const int* h_pos, int T, cudaStream_t st);
    // Host coverage bookkeeping: rows hold positions [ctx_end - ctx_n, ctx_end)
    // whenever every ingest continued at ctx_end (ctx_contig). A gap makes
    // the by-count rollback below meaningless, so rollback_to resets instead.
    int ctx_end = 0;
    bool ctx_contig = true;
    void reset_ctx() { ctx_n = 0; ctx_end = 0; ctx_contig = true; } // cold ring
    // Drop the last `rows` ingested rows (round truncation: post_round's
    // on_round can shrink a committed round AFTER dflash2_round already
    // ingested its accepted lanes -- the ring is append-only, so phantom
    // rows would otherwise coexist with the re-committed positions and
    // pollute drafter attention). Host counter only; the device tail is
    // dead until the next ingest overwrites it.
    void rollback(int rows) {
        const int r = rows < ctx_n ? rows : ctx_n;
        ctx_n -= r;
        ctx_end -= r;
    }
    // Turn alignment (2026-09-07): keep the rows for positions < pos, drop
    // the rest. A prefix-cache warm turn re-prefills only [base, NP), so the
    // rows below the (verified) common prefix stay valid -- the ring no
    // longer cold-starts every turn. Non-contiguous coverage => reset.
    void rollback_to(int pos) {
        if (!ctx_contig || pos <= ctx_end - ctx_n) { reset_ctx(); return; }
        if (pos < ctx_end) rollback(ctx_end - pos);
    }

    // one draft block of K proposals (width K+1): anchor (pending) token at
    // anchor_pos. Fully device-side; proposals land in d_prop[0..K-1]. No
    // stream sync. out_host (optional): also D2H the proposals (syncs).
    // sampling: the selector walk SAMPLES its path from softmax(E/T) (needs
    // set_sampler; ignored otherwise) and retains q in d_qrow. Greedy walk
    // (default) is the bitwise-unchanged argmax path.
    void draft(int anchor_token, int anchor_pos, int K, cudaStream_t st,
               int* out_host = nullptr, bool sampling = false);
    // The device-side compute of one draft (everything after the per-round
    // H2D of anchor/positions) -- captured once by capture_draft and replayed,
    // dropping the ~50 eager launches per round. Requires the engine Q8 embed
    // (device anchor lookup); falls back to eager when unavailable.
    void draft_compute(int K, cudaStream_t st, bool sampling = false);
    // Captures the greedy graph, plus the sampled-walk twin when a sampler is
    // set (identical forward, only the walk kernel's mode differs).
    void capture_draft(int K, cudaStream_t st);
    cudaGraphExec_t draft_exec = nullptr, draft_exec_s = nullptr;
    int draft_exec_k = 0;
};

// Bare launcher for the selector walk kernel (test_kernels: integrated
// device-walk + sparse-q verify gate on synthetic candidates/codebooks).
void d2_walk_launch(const int* d_cand, const float* d_cval, const float* d_hp,
                    const __half* d_pred, const __half* d_succ, const int* d_anchor, int K,
                    int* d_out, bool sampled, const q27k::SampleParams* d_sp, const int* d_posW,
                    float* d_qrow, cudaStream_t st);

} // namespace q27d2
