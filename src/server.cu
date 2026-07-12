// q27 HTTP server. Multi-slot (--slots N), R1b round-granularity GPU
// time-slicing across slots. Greedy by default (spec decode); temperature>0
// routes to the sampled plain path (roadmap #2, Phase 1).
// Endpoints:
//   GET  /health, /v1/models
//   POST /v1/chat/completions, /v1/completions        (OpenAI)
//   POST /v1/messages                                 (Anthropic, Claude Code-grade:
//        thinking blocks, tool_use/tool_result, input_json_delta streaming)
//   POST /v1/responses                                (OpenAI Responses, Codex CLI)
//
// usage: q27-server model.q27 model.tok [--port 8080] [--host 127.0.0.1]
//                   [--ctx 8192] [--fast-head] [--slots N] [--slot1-ctx M]
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <set>
#include <tuple>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>

#include "engine.cuh"
#include "tokenizer.h"
#include "api_common.h"
#include "toolgram.h"
#include "toolconstrain.h"
#include "../third_party/httplib.h"
#include "../third_party/json.hpp"

using json = nlohmann::json;
using q27::Msg;
using q27::StreamSplitter;

// Serialize with invalid-UTF-8 replacement: json::dump's strict default
// throws type_error.316, and an uncaught throw in a handler or streaming
// lambda is std::terminate. The Utf8Gate on every generation pipeline keeps
// split characters intact; this is the backstop for everything else.
// File-scope on purpose -- helpers with explicit capture lists use it too.
static std::string jdump(const json& j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

// Sampling params (roadmap #2) shared across all 3 API shapes. temperature<=0
// or absent -> greedy (inv_temp 0 routes generate() to the bitwise spec path).
// top_p defaults to 1 (full vocab). seed is honored for reproducible A/B; the
// OpenAI shapes carry it natively, and it is read harmlessly on the others.
// Q27_FORCE_TEMP / Q27_FORCE_TOP_P let the server apply sampling to clients that
// send NO temperature (CC/CRUSH) -- the exit-gate "default on" config
// (docs/sampling-exit-gate.md). An explicit request temperature still wins: the
// body.value default only fires when the key is absent. Env unset -> force_temp 0
// -> byte-identical to the old greedy path (canonical 4c4120c7 is CLI-generated and
// untouched regardless). A forced request with no client seed draws a distinct
// atomic-counter seed, LOGGED, so each trial is an independent sample yet reproducible
// by replaying that seed as an explicit request field.
static q27k::SampleParams parse_sample(const json& body) {
    static const double force_temp = []{ const char* e = getenv("Q27_FORCE_TEMP"); return e ? atof(e) : 0.0; }();
    static const double force_tp   = []{ const char* e = getenv("Q27_FORCE_TOP_P"); return e ? atof(e) : 1.0; }();
    static std::atomic<unsigned long long> force_seed_ctr{0};

    q27k::SampleParams s{0.f, 1.f, 0ull};
    double temp = body.value("temperature", force_temp);
    if (temp > 0.0) {
        s.inv_temp = (float)(1.0 / temp);
        double tp = body.value("top_p", force_tp);
        s.top_p = (float)((tp > 0.0 && tp <= 1.0) ? tp : 1.0);
        if (body.contains("seed") && body["seed"].is_number())
            s.seed = (unsigned long long)body["seed"].get<long long>();
        else if (force_temp > 0.0) {
            s.seed = ++force_seed_ctr;   // distinct independent draw per forced request
            fprintf(stderr, "[force-sample] temp=%.3f top_p=%.3f seed=%llu\n",
                    temp, (double)s.top_p, s.seed);
        }
    }
    return s;
}

// P7/P15: per-request constrained tool decoding -- logic lives in
// toolconstrain.h (unit-tested CPU-side; see tools/test_toolconstrain.cpp).
// The engage-lag fix wires three hooks per generation: on_round (scan_round ->
// truncate+refinish), on_pending (next-round slot-0 mask), on_id (in-call feed).
using ToolConstrainer = q27::BasicToolConstrainer<Engine, q27::Tokenizer>;

// P15 review M1: the tc hooks capture the handler's stack-local constrainer by
// reference. A non-CUDA throw out of generate() (bad_alloc, json ops in the
// on_token callback) unwinds past the manual `eng.on_X = nullptr` lines --
// httplib catches at routing, the process survives, and the NEXT request on a
// hook-less path (OpenAI/Responses) would invoke dangling lambdas. This guard
// nulls them on any exit. Construct AFTER slot_guard so hooks clear BEFORE the
// slot is freed for reuse (reverse destruction order).
struct HookGuard {
    Engine& e;
    ~HookGuard() {
        e.on_pending = nullptr;
        e.on_drafts = nullptr;
        e.on_round = nullptr;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s model.q27 model.tok [--port N] [--host H] [--ctx C]\n"
                "  Defaults (2026-07-10) = the measured Claude-Code stack: fp8 KV +\n"
                "  Q27_PMIN=0.5 + Q27_MAXD=auto7 + Q27_SUFFIX_W=12 + Q27_FD=mma (sm_89+)\n"
                "  + fast-head + no-think + phase stats; --ctx auto-sizes to VRAM\n"
                "  (auto-ctx cap 262144 fp8/turbo3, 131072 fp16; single-slot). Escapes:\n"
                "  Q27_PROFILE=ref (conservative\n"
                "  reference: fp16/ungated/no-suffix/fd2), any individual Q27_* env,\n"
                "  --kv-fp16 --no-fast-head --think. The CLI binary keeps reference\n"
                "  defaults (bitwise canonical).\n",
                argv[0]);
        return 1;
    }
    // loopback by default: this server has NO auth -- exposing it beyond
    // the local host is an explicit operator decision (--host 0.0.0.0).
    std::string model = argv[1], tokpath = argv[2], host = "127.0.0.1";
    // served model id = model file stem (e.g. qwen36-27b-mtp-q6k), not a
    // hardcoded name that goes stale the moment a second model exists
    std::string served_name = model.substr(model.find_last_of('/') + 1);
    if (served_name.size() > 4 && served_name.substr(served_name.size() - 4) == ".q27")
        served_name.resize(served_name.size() - 4);
    int port = 8080, ctx = -1; // -1 = auto-size to VRAM (single-slot)
    int n_slots = 1, slot1_ctx = 32768;
    int fast_flag = -1;        // tri-state: explicit flag wins over profile
    int think_flag = -1;
    bool kv_fp16 = false;
    bool constrain_tools = false;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--slots") && i + 1 < argc) n_slots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--slot1-ctx") && i + 1 < argc) slot1_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fast-head")) fast_flag = 1;
        else if (!strcmp(argv[i], "--no-fast-head")) fast_flag = 0;
        else if (!strcmp(argv[i], "--no-think")) think_flag = 0;
        else if (!strcmp(argv[i], "--think")) think_flag = 1;
        else if (!strcmp(argv[i], "--constrain-tools")) constrain_tools = true;
        else if (!strcmp(argv[i], "--kv-fp16")) kv_fp16 = true;
    }
    // CC-SERVING DEFAULTS (width-12 + tuning day, 2026-07-10): a bare
    // `q27-server model tok` serves the full measured stack -- the exact
    // config every live trial and record number was earned on. Mechanism:
    // setenv(overwrite=0), so any user-set Q27_* env wins untouched;
    // Q27_PROFILE=ref restores the conservative reference behavior (fp16
    // KV, ungated, no suffix, fd2). The CLI binary keeps reference
    // defaults so the bitwise canonical gates are untouched.
    // fp8 KV + fdmma need sm_89+; older parts fall back to fp16 + fd2.
    const char* prof = getenv("Q27_PROFILE");
    const bool ref_profile = prof && !strcmp(prof, "ref");
    int cc_arch = 0;
    {
        int dev = 0, mj = 0, mn = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&mj, cudaDevAttrComputeCapabilityMajor, dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&mn, cudaDevAttrComputeCapabilityMinor, dev));
        cc_arch = mj * 10 + mn;
    }
    // flag defaults follow the profile: CC = fast-head + no-think (every
    // live trial); ref = the conservative pre-flip behavior. Explicit
    // flags win in both.
    const bool fast = fast_flag >= 0 ? fast_flag != 0 : !ref_profile;
    const bool no_think_srv = think_flag >= 0 ? think_flag == 0 : !ref_profile;
    if (no_think_srv) fprintf(stderr, "no-think: empty-think prefill on all chat paths\n");
    if (kv_fp16) setenv("Q27_KV", "fp16", 1);
    if (!ref_profile) {
        if (cc_arch >= 89) {
            setenv("Q27_KV", "fp8", 0);
            setenv("Q27_FD", "mma", 0);
        } else if (cc_arch >= 80) {
            // H16 (fp16-MMA) verify: Ampere gets the mma path too
            // (2026-07-12, docs/plans/2026-07-12-fdmma-f16.md); KV default
            // stays fp16 there (no fp8 HW) -- turbo3 opt-in recommended.
            setenv("Q27_FD", "mma", 0);
        }
        setenv("Q27_PMIN", "0.5", 0);
        setenv("Q27_MAXD", "auto7", 0);
        setenv("Q27_SUFFIX", "1", 0);
        setenv("Q27_SUFFIX_W", "12", 0);
        setenv("Q27_PHASE_STATS", "1", 0);
    }
    fprintf(stderr,
            "profile: %s (sm_%d) | kv=%s fd=%s pmin=%s maxd=%s suffix=%s/w%s fast-head=%d "
            "think=%d\n",
            ref_profile ? "ref" : "cc", cc_arch, getenv("Q27_KV") ? getenv("Q27_KV") : "fp16",
            getenv("Q27_FD") ? getenv("Q27_FD") : "fd2",
            getenv("Q27_PMIN") ? getenv("Q27_PMIN") : "off",
            getenv("Q27_MAXD") ? getenv("Q27_MAXD") : "4",
            getenv("Q27_SUFFIX") ? getenv("Q27_SUFFIX") : "0",
            getenv("Q27_SUFFIX_W") ? getenv("Q27_SUFFIX_W") : "-", fast ? 1 : 0,
            no_think_srv ? 0 : 1);

    // --ctx auto (single-slot): size the KV budget to free VRAM. The fixed
    // cost (weights + GDN role sets + graph zoo + buffers) SCALES WITH
    // Q27_W_MAX -- each width adds one role set (~157MB) and ~one perm's
    // worth of captured graphs (~130MB), so a narrow build frees budget.
    // Anchor: 131072 fp8 W_MAX=12 measured ~27.0GB total on the 17.73GB
    // v1.4 artifact => non-weight base ~1.27GB + weights (stat'd from the
    // model file, so heavier tiers like q6-v1 at 20.49GB size correctly) +
    // (W_MAX+1)*0.157 roles + W_MAX*0.13 graphs.
    // per-token = 34KB fp8 / 68KB fp16 (attn + MTP KV). NOTE the anchor was
    // calibrated on the 5090 fp8/mma path; the sm_86 fp16/fd2 fallback runs
    // heavier, so on a 24GB card the fit can still miss -- hence no forced
    // floor: clamp to what actually fits and warn rather than OOM.
    if (ctx < 0) {
        if (n_slots > 1) {
            ctx = 8192; // legacy default; multi-slot should pass --ctx
            fprintf(stderr, "--ctx not set with --slots %d: using %d (pass --ctx)\n", n_slots,
                    ctx);
        } else {
            size_t free_b = 0, total_b = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            const char* kvv = getenv("Q27_KV");
            const bool fp8 = kvv && !strcmp(kvv, "fp8");
            const bool t3 = kvv && !strcmp(kvv, "turbo3");
            const bool t3v = kvv && !strcmp(kvv, "turbo3v");
            struct stat wst {};
            const double wbytes = stat(model.c_str(), &wst) == 0 ? (double)wst.st_size : 17.73e9;
            const double fixed = wbytes + 1.27e9 + (Q27_W_MAX + 1) * 0.157e9 + Q27_W_MAX * 0.13e9;
            // per-token KV bytes across the 17 attn+MTP cache pairs: turbo3
            // 2*400 B, turbo3v 2048+400 B, fp8 2*1024, fp16 2*2048
            const double slack = 1.0e9,
                         per_tok = t3 ? 13.6e3 : t3v ? 41.6e3 : fp8 ? 34e3 : 68e3;
            long budget = (long)((double)free_b - fixed - slack);
            long c = budget > 0 ? (long)(budget / per_tok) : 0;
            // cap: native window (262144) for the compact KV formats
            // (2026-07-11, Gabe sign-off: fp8 measured to 294912 on the
            // 5090, turbo3 to 655360 with needle 6/6 @361K -- the cap is a
            // TTFT/estimate-margin guard, not a VRAM fact); fp16 keeps the
            // historical 131072 (it barely clears it anyway).
            const long cap = (fp8 || t3 || t3v) ? 262144 : 131072;
            if (c > cap) c = cap;
            ctx = (int)(c / 4096 * 4096);
            if (ctx < 4096) {
                fprintf(stderr,
                        "--ctx auto: only %d fits (free %.1fGB, %s KV, W_MAX=%d) -- likely to "
                        "OOM; pass a smaller --ctx or rebuild with a lower Q27_W_MAX\n",
                        ctx, free_b / 1e9, t3 ? "turbo3" : t3v ? "turbo3v" : fp8 ? "fp8" : "fp16", Q27_W_MAX);
                if (ctx < 2048) ctx = 2048; // give the ctor a floor to fail loudly at
            } else {
                fprintf(stderr, "--ctx auto: %d (free %.1fGB, %s KV, W_MAX=%d)\n", ctx,
                        free_b / 1e9, t3 ? "turbo3" : t3v ? "turbo3v" : fp8 ? "fp8" : "fp16", Q27_W_MAX);
                if (ctx < 16384)
                    fprintf(stderr, "  (tight -- a lower Q27_W_MAX build would free more)\n");
            }
        }
    }

    fprintf(stderr, "loading tokenizer...\n");
    q27::Tokenizer tok(tokpath);
    fprintf(stderr, "loading model...\n");
    // P10-A1: weights owned here and shared into the Engine(s) by reference.
    // Upload once; borrowing engines skip the 17.7 GB weight copy. (Multi-slot
    // will construct N engines from this same pair.)
    q27::Model shared_model = q27::Model::open(model);
    q27::DeviceModel shared_dm(shared_model);
    fprintf(stderr, "uploading weights...\n");
    shared_dm.upload_all();
    shared_dm.checksum_baseline();
    fprintf(stderr, "resident: %.2f GB (checksummed)\n", shared_dm.bytes_resident() / 1e9);
    // R1 multi-slot: N engines borrow the one uploaded weight set. Slot 0
    // gets --ctx; slots 1+ get --slot1-ctx (subagent/background conversations
    // measured 11-18K in R0). Per-slot GDN snapshot + ckpt ring + KV means an
    // interleaved second conversation no longer destroys the first one's
    // prefix cache (R0: that re-prefill class alone was 25% of a Claude Code
    // session). R1b: generations TIME-SLICE the GPU at round/chunk
    // granularity behind a FIFO gate instead of serializing whole requests
    // (R1's 24.4s residual queue wait); `busy` marks an engine claimed by an
    // in-flight (possibly yielded) generation.
    struct Slot {
        std::unique_ptr<Engine> eng;
        long last_used = 0;
        int id = 0;
        bool busy = false;                   // R1b: claimed by a generation
        bool stamp_on_free = false;          // LRU-stamp when freed (not refused)
        std::vector<int> tool_mask_host2dev; // per-engine mask-pool ids (P7)
    };
    n_slots = std::max(1, std::min(4, n_slots));
    std::vector<Slot> slots;
    for (int si = 0; si < n_slots; si++) {
        int sctx = si == 0 ? ctx : slot1_ctx;
        if (si > 0) {
            // coarse per-slot floor: GDN recurrent state (exact bytes from
            // slot 0's own allocation -- review 2026-07-09: the old "5 sets
            // ~3GB" constant predated the maxd6/7 widenings) + prefill/attn
            // scratch (~700 MB) + KV + MTP KV + slack; skip extra slots
            // rather than abort on cudaMalloc
            size_t freeb = 0, totalb = 0;
            cudaMemGetInfo(&freeb, &totalb);
            // KV bytes/token from the engine's own sizing (kv_bytes covers
            // fp16/fp8/turbo3 rows) -- slots[0] is always constructed first
            size_t row_b = (slots[0].eng->kv_bytes(false) + slots[0].eng->kv_bytes(true)) /
                           slots[0].eng->max_ctx;
            size_t kvb = (size_t)sctx * row_b * (slots[0].eng->kcache.size() + 1);
            size_t need = slots[0].eng->gdn_state_bytes + (700ull << 20) + kvb + (kvb >> 3) +
                          (512ull << 20);
            if (freeb < need) {
                fprintf(stderr, "slot %d SKIPPED: %.1f GB free < %.1f GB needed\n", si,
                        freeb / 1e9, need / 1e9);
                break;
            }
        }
        Slot s;
        s.id = si;
        s.eng = std::make_unique<Engine>(shared_model, shared_dm, sctx);
        s.eng->fast_head = fast;
        s.eng->build_graph();
        s.eng->build_spec_graphs();
        slots.push_back(std::move(s));
        fprintf(stderr, "slot %d ready: ctx=%d\n", si, sctx);
    }
    // Admission clamps below use the LARGEST slot; the routed slot re-clamps.
    int max_slot_ctx = 0;
    for (auto& s : slots) max_slot_ctx = std::max(max_slot_ctx, s.eng->max_ctx);
    // Largest prompt ANY route admits: the largest slot must retain a
    // positive decode budget past the spec-round reserve (review follow-up
    // 2026-07-09 #3: the old hardcoded -7 was the depth-5 reserve, so at
    // gate_maxd 6/7 a prompt inside the stale bound could clamp n_max to 0
    // and return an empty 200 instead of the context-limit 400). Every
    // engine parses the same env, so slot 0's reserve speaks for all; all
    // three routes preflight against this before claiming a slot.
    const int max_prompt = max_slot_ctx - slots[0].eng->ctx_round_reserve();
    const int EOS = tok.eos();
    // P7 shared mask cache (mutated only from generation callbacks, which
    // run while holding the GPU gate; pool ids are per-slot)
    std::vector<std::string> vocab_bytes_v = tok.vocab_bytes();
    q27::ToolMaskCache tool_mask_cache;
    tool_mask_cache.init(&vocab_bytes_v, tok.token_id("</tool_call>"));
    if (constrain_tools)
        fprintf(stderr, "constrain-tools: grammar-locked <tool_call> bodies (open=%d close=%d)\n",
                tok.token_id("<tool_call>"), tok.token_id("</tool_call>"));

    // R1b: FIFO ticket gate time-slices the GPU across concurrent
    // generations (q27::GpuGate). Engines are claimed via Slot::busy under
    // route_m before entering the gate, so routing only ever reads settled
    // engine state. Q27_NO_INTERLEAVE restores R1 whole-request
    // serialization (the yield hook is simply never installed) -- debug
    // lever for the rare mid-round host-interaction flake class.
    q27::GpuGate gpu_gate;
    std::mutex route_m;
    std::condition_variable route_cv;
    const bool no_interleave = getenv("Q27_NO_INTERLEAVE") != nullptr;
    fprintf(stderr, "interleave: %s\n",
            no_interleave ? "OFF (Q27_NO_INTERLEAVE)" : "round-granularity");
    std::atomic<long> req_counter{0};

    // R0 telemetry: one [req] stderr line per generation request, self-contained
    // so real-work anatomy (queue wait, tokenize, prefill reuse, decode, client
    // write time, conversation interleave) is measurable from the log alone.
    // conv = fnv1a64 over system text + first user text: stable across the turns
    // of one conversation, distinct across conversations (main thread, subagents,
    // and background utility calls differ in system and/or first user message).
    auto fnv1a = [](const std::string& s, unsigned long long h = 1469598103934665603ULL) {
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
        return h;
    };
    auto text_of = [](const json& v) -> std::string {
        if (v.is_string()) return v.get<std::string>();
        std::string out;
        if (v.is_array())
            for (auto& p : v) {
                std::string ty = p.value("type", "");
                if (ty == "text" || ty == "input_text" || ty == "output_text")
                    out += p.value("text", "");
            }
        return out;
    };
    auto conv_fp = [&](const json& body) -> unsigned long long {
        std::string sys, fu;
        if (body.contains("system")) sys = text_of(body["system"]);
        else if (body.contains("instructions") && body["instructions"].is_string())
            sys = body["instructions"].get<std::string>();
        const char* lk = body.contains("messages") && body["messages"].is_array() ? "messages"
                         : body.contains("input") && body["input"].is_array()    ? "input"
                                                                                 : nullptr;
        if (lk) {
            for (auto& m : body[lk]) {
                if (!m.is_object() || !m.contains("content")) continue;
                std::string role = m.value("role", "");
                if (sys.empty() && role == "system") sys = text_of(m["content"]);
                if (fu.empty() && role == "user") fu = text_of(m["content"]);
                if (!sys.empty() && !fu.empty()) break;
            }
        } else if (body.contains("input") && body["input"].is_string()) {
            fu = body["input"].get<std::string>();
        } else if (body.contains("prompt") && body["prompt"].is_string()) {
            fu = body["prompt"].get<std::string>();
        }
        q27::normalize_cc_billing_header(sys);  // hash what the engine prefills, not the raw stamp
        return fnv1a(fu, fnv1a(sys) ^ 0x9e3779b97f4a7c15ULL);
    };
    struct ReqTrace {
        long rid;
        const char* api;
        unsigned long long conv;
        std::chrono::steady_clock::time_point t0; // stamped after tokenize
        double tok_ms;                            // render + encode
    };
    auto ms_since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t)
            .count();
    };
    // t= is ms since server start (line printed when generate returns), so
    // inter-request GPU idle -- tool execution, client think time -- is
    // recoverable from the log alone: idle = t[n]-(qw+pf_ms+dec_ms)[n] - t[n-1].
    const auto srv_t0 = std::chrono::steady_clock::now();
    // gw/yields (R1b): time this request spent parked in GPU handovers and
    // how many happened. pf_ms/dec_ms are wall-inclusive of those parks, so
    // GPU-busy accounting is (pf_ms + dec_ms - gw); tps likewise understates
    // raw decode rate under contention. New fields sit after end= -- the
    // reqlog_gate parse regex stops there.
    // P15 constrain-tools telemetry: per-request grammar counters, appended
    // after end= like the P13/gate fields so reqlog_gate's parse is unaffected.
    auto tg_stats = [](const ToolConstrainer& tc) -> std::string {
        if (!tc.enabled) return "";
        char b[96];
        snprintf(b, sizeof b, " tg=%ld,%ld,%ld,%ld", tc.engaged, tc.disengaged, tc.pool_drops,
                 tc.rebinds);
        return std::string(b);
    };
    auto req_log = [&](const ReqTrace& rt, double qw_ms, const Engine& e, int slot_id,
                       const std::string& extra = std::string()) {
        const auto& g = e.gs;
        double tps = g.dec_ms > 0 ? g.dec * 1000.0 / g.dec_ms : 0.0;
        char p13buf[96], gatebuf[512], phbuf[352], sfxbuf[48];
        // Q27_SUFFIX: engine-cumulative suffix-round counters (fired, tokens
        // committed by suffix rounds), appended after end= like gch/glf.
        if (e.suffix_on)
            snprintf(sfxbuf, sizeof sfxbuf, " sfx=%ld,%ld", e.sfx_fired, e.sfx_tok);
        else
            sfxbuf[0] = '\0';
        // Q27_PHASE_STATS: per-request gated-round wall split, appended after
        // end= like the P13/gate fields (reqlog_gate's parse is unaffected).
        // phwn/phwm: per-verify-width round counts and summed verify ms, W=2..8.
        if (e.phase_stats)
            // sfxm/sfxn (width-12 P1): per-request suffix-round wall + count
            // -- the wide-width (sfx_width) cost point; suffix rounds are
            // deliberately NOT in phwn/phwm (own class, one width).
            snprintf(phbuf, sizeof phbuf,
                     " phd=%.1f phv=%.1f phs=%ld"
                     " phwn=%ld,%ld,%ld,%ld,%ld,%ld,%ld"
                     " phwm=%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f"
                     " sfxm=%.1f sfxn=%ld",
                     g.draft_ms, g.verify_ms, g.draft_steps, g.vw_n[2], g.vw_n[3],
                     g.vw_n[4], g.vw_n[5], g.vw_n[6], g.vw_n[7], g.vw_n[8], g.vw_ms[2],
                     g.vw_ms[3], g.vw_ms[4], g.vw_ms[5], g.vw_ms[6], g.vw_ms[7],
                     g.vw_ms[8], g.sfx_ms, g.sfx_rounds);
        else
            phbuf[0] = '\0';
        fprintf(stderr,
                "[req] rid=%ld api=%s conv=%08llx qw_ms=%.0f tok_ms=%.0f prompt=%d hit=%d "
                "ckpt=%d pf=%d pf_ms=%.0f dec=%d dec_ms=%.0f cb_ms=%.0f rounds=%d tps=%.1f "
                "end=%s gw=%.0f yields=%d slot=%d t=%.0f%s%s%s%s%s\n",
                rt.rid, rt.api, rt.conv, qw_ms, rt.tok_ms, g.prompt, g.hit, g.ckpt, g.pf,
                g.pf_ms, g.dec, g.dec_ms, g.cb_ms, g.rounds, tps,
                (g.end && g.end[0]) ? g.end : "?", g.gw_ms, g.yields, slot_id,
                ms_since(srv_t0),
                // P13: adaptive-maxd activity, cumulative on this engine
                // (per-request when Q27_MAXD_RESET=1 -- review 2026-07-09)
                e.maxd_auto ? (snprintf(p13buf, sizeof p13buf,
                                        " md4=%ld md5=%ld md6=%ld md7=%ld mprom=%ld"
                                        " mdem=%ld",
                                        e.dctl.rounds[4], e.dctl.rounds[5], e.dctl.rounds[6],
                                        e.dctl.rounds[7], e.dctl.promotes, e.dctl.demotes),
                               p13buf)
                            : "",
                // maxd6 GO-IF: cumulative gated-round histograms on this engine --
                // margin-run depth (gch, cap 0..5) and accepted length (gnh, n 1..6).
                // accept-gate Phase 0: per-lane fired/accepted (glf/gla, lanes 1..5)
                // -- the conditional yields the marginals cannot reconstruct.
                e.pmin_theta > 0.f
                    ? (snprintf(gatebuf, sizeof gatebuf,
                                " gch=%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld"
                                " gnh=%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld"
                                " glf=%ld,%ld,%ld,%ld,%ld,%ld,%ld"
                                " gla=%ld,%ld,%ld,%ld,%ld,%ld,%ld",
                                e.gate_cap_hist[0], e.gate_cap_hist[1], e.gate_cap_hist[2],
                                e.gate_cap_hist[3], e.gate_cap_hist[4], e.gate_cap_hist[5],
                                e.gate_cap_hist[6], e.gate_cap_hist[7], e.gate_n_hist[1],
                                e.gate_n_hist[2], e.gate_n_hist[3], e.gate_n_hist[4],
                                e.gate_n_hist[5], e.gate_n_hist[6], e.gate_n_hist[7],
                                e.gate_n_hist[8], e.gate_lane_fired[1], e.gate_lane_fired[2],
                                e.gate_lane_fired[3], e.gate_lane_fired[4],
                                e.gate_lane_fired[5], e.gate_lane_fired[6],
                                e.gate_lane_fired[7], e.gate_lane_acc[1], e.gate_lane_acc[2],
                                e.gate_lane_acc[3], e.gate_lane_acc[4], e.gate_lane_acc[5],
                                e.gate_lane_acc[6], e.gate_lane_acc[7]),
                       gatebuf)
                    : "",
                phbuf, sfxbuf, extra.c_str());
    };
    // R1b routing: claim a FREE engine (Slot::busy=false) that can take the
    // prompt, or block until one frees. Tiers among free engines unchanged
    // from R1: a slot that can actually restore a prefix of this prompt
    // (Engine::reuse_len -- snapshot extension or a P9 checkpoint, the same
    // predicate generate() honors) > an empty slot (never evict a live
    // conversation when a free one exists) > LRU eviction. Busy engines are
    // skipped outright -- reuse_len against a mid-generation engine reads
    // moving state. If the prompt fits NO slot even when all are free, the
    // largest free slot takes it and generate() refuses cleanly (no LRU
    // stamp, as before). Condvar wakeups barge (no ticket order here);
    // bounded by <=4 slots and self-limiting clients -- the GPU gate is the
    // fair one.
    long slot_use_counter = 0;
    auto claim_slot = [&](const std::vector<int>& prompt) -> Slot& {
        std::unique_lock<std::mutex> lk(route_m);
        // eligibility requires a POSITIVE decode budget on the slot, not
        // just prompt-fits (review follow-up 2026-07-09 #3): a slot whose
        // clamp would floor n_max to 0 must not take the request. The route
        // preflights guarantee at least the largest slot qualifies, so this
        // never deadlocks.
        const bool fits_any = (int)prompt.size() <= max_prompt;
        for (;;) {
            Slot* best = nullptr;
            int best_tier = -1, best_key = 0;
            for (auto& s : slots) {
                if (s.busy) continue;
                if (!fits_any) { // doomed prompt: largest free slot refuses it
                    if (!best || s.eng->max_ctx > best->eng->max_ctx) best = &s;
                    continue;
                }
                if ((int)prompt.size() + s.eng->ctx_round_reserve() > s.eng->max_ctx) continue;
                int rl = s.eng->reuse_len(prompt);
                int tier = rl > 0 ? 2 : s.eng->cache_empty() ? 1 : 0;
                bool better;
                if (!best) better = true;
                else if (tier != best_tier) better = tier > best_tier;
                else if (tier == 2) better = rl > best_key;
                else better = s.last_used < best->last_used;
                if (better) { best = &s; best_tier = tier; best_key = rl; }
            }
            if (best) {
                best->busy = true;
                // LRU is stamped at FREE, not here: eviction preference must
                // track completion recency (a slot claimed early but finishing
                // last is the likeliest to continue). Claim-time stamps are
                // invisible to routing anyway -- busy slots are never scanned.
                // Refused-class claims (fits nowhere) keep the old stamp.
                best->stamp_on_free = fits_any;
                // Lineage-aware DepthCtl reset (review follow-up 2026-07-09):
                // tier 2 = this slot restores a prefix of the prompt (same
                // conversation) -> the ladder carries its warm state; any
                // other claim is a new lineage taking the slot over, so it
                // must not inherit the previous tenant's ceiling/EMAs.
                // Q27_MAXD_RESET=1 remains the stricter every-request reset.
                if (fits_any && best_tier < 2 && best->eng->maxd_auto)
                    best->eng->dctl.reset();
                return *best;
            }
            route_cv.wait(lk);
        }
    };
    auto free_slot = [&](Slot& s) {
        {
            std::lock_guard<std::mutex> lk(route_m);
            s.busy = false;
            if (s.stamp_on_free) s.last_used = ++slot_use_counter;
        }
        route_cv.notify_all();
    };
    // scope guard so the claim is released on every exit path
    auto slot_guard = [&free_slot](Slot& s) {
        return std::shared_ptr<Slot>(&s, [&free_slot](Slot* p) { free_slot(*p); });
    };
    // Per-request yield hook: hand the GPU to a queued request at round /
    // chunk boundaries, draining OUR stream first so the handover is real.
    // Captures only stable objects (engine lives in `slots`, gate in main),
    // so a hook left installed on an engine is harmless, not dangling.
    auto make_yield = [&gpu_gate, no_interleave](Engine& e) -> std::function<bool()> {
        if (no_interleave) return nullptr;
        Engine* ep = &e;
        return [ep, &gpu_gate] {
            if (!gpu_gate.contended()) return false;
            CUDA_CHECK(cudaStreamSynchronize(ep->stm));
            return gpu_gate.maybe_yield();
        };
    };

    httplib::Server srv;
    srv.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        fprintf(stderr, "[http] %s %s -> %d\n", req.method.c_str(), req.path.c_str(),
                res.status);
    });

    srv.Get("/health", [&](const httplib::Request& req, httplib::Response& res) {
        // /health?verify=1 recomputes the resident-weight checksums (~20 ms;
        // read-only, so safe concurrently with generation -- but it launches
        // on the legacy default stream, which BARRIERS against in-flight
        // kernels on the blocking engine streams: expect a multi-ms stall
        // injected into whoever holds the GPU gate, not true overlap).
        if (req.has_param("verify")) {
            int bad = shared_dm.checksum_verify(true);
            res.set_content(std::string("{\"status\":\"") + (bad ? "corrupted" : "ok") +
                                "\",\"weight_mismatches\":" + std::to_string(bad) + "}",
                            "application/json");
            return;
        }
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json j = {{"object", "list"},
                  {"data", json::array({{{"id", served_name}, {"object", "model"},
                                         {"owned_by", "q27"}}})}};
        res.set_content(j.dump(), "application/json");
    });

    // ---------------- OpenAI chat/completions (text only) ----------------

    auto build_prompt = [&](const json& body) -> std::vector<int> {
        if (body.contains("messages")) {
            std::vector<std::pair<std::string, std::string>> msgs;
            for (auto& m : body["messages"]) {
                std::string role = m.value("role", "user");
                std::string content;
                // const operator[] on a missing key aborts (json.hpp assertion) --
                // a content-less message must not kill the server (Security #1;
                // mirrors the Anthropic-path guard in api_common.h).
                if (m.is_object() && m.contains("content")) {
                    if (m["content"].is_string()) content = m["content"];
                    else if (m["content"].is_array())
                        for (auto& part : m["content"])
                            if (part.value("type", "") == "text")
                                content += part.value("text", "");
                }
                msgs.push_back({role, content});
            }
            // enable_thinking=false: top-level (Qwen-style clients) or nested
            // chat_template_kwargs (llama.cpp/GLM-style) -> empty-think prefill
            bool think = body.value("enable_thinking", true);
            if (body.contains("chat_template_kwargs"))
                think = body["chat_template_kwargs"].value("enable_thinking", think);
            if (no_think_srv) think = false;
            return tok.apply_chat_template(msgs, think);
        }
        return tok.encode(body.value("prompt", std::string()));
    };

    auto handle = [&](const httplib::Request& req, httplib::Response& res, bool chat) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return; }
        int n_max = body.value("max_tokens", 256);
        bool stream = body.value("stream", false);
        auto tk0 = std::chrono::steady_clock::now();
        std::vector<int> prompt = build_prompt(body);
        ReqTrace rt{req_counter++, chat ? "oai" : "cmpl", conv_fp(body),
                    std::chrono::steady_clock::now(), ms_since(tk0)};
        // context-limit preflight BEFORE slot claim / SSE commit (review
        // follow-up 2026-07-09 #3): past this bound the routed slot's
        // n_max clamp floors at 0 -> empty 200
        if ((int)prompt.size() > max_prompt) {
            res.status = 400;
            res.set_content(json{{"error",
                                  {{"message", q27::ctx_limit_error_message(
                                                   (int)prompt.size(), max_prompt)},
                                   {"type", "invalid_request_error"},
                                   {"code", "context_length_exceeded"}}}}
                                .dump(),
                            "application/json");
            return;
        }
        if ((int)prompt.size() + n_max > max_slot_ctx)
            n_max = max_slot_ctx - (int)prompt.size();
        long created = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();

        const char* obj = chat ? "chat.completion" : "text_completion";
        const char* objd = chat ? "chat.completion.chunk" : "text_completion";

        if (!stream) {
            Slot& sl = claim_slot(prompt); // may wait for a free engine
            auto sl_lease = slot_guard(sl);
            Engine& eng = *sl.eng;
            eng.samp = parse_sample(body);
            q27::GpuGate::Lease lk(gpu_gate);
            double qw = ms_since(rt.t0);
            eng.on_round_gap = make_yield(eng);
            // re-clamp to the routed slot (rows P+1..P+gate_maxd+1 must stay
            // in ctx; reserve derived from the engine's active max depth)
            n_max = std::max(0, std::min(n_max, eng.max_ctx - (int)prompt.size() - (eng.ctx_round_reserve() - 1)));
            std::string text;
            q27::Utf8Gate ugate;
            int n = eng.generate(prompt, n_max, EOS, [&](int id) {
                text += ugate.feed(tok.decode_one(id));
                return true;
            });
            eng.on_round_gap = nullptr;
            text += ugate.flush();
            req_log(rt, qw, eng, sl.id);
            json choice;
            if (chat)
                choice = {{"index", 0}, {"finish_reason", n >= n_max ? "length" : "stop"},
                          {"message", {{"role", "assistant"}, {"content", text}}}};
            else
                choice = {{"index", 0}, {"finish_reason", n >= n_max ? "length" : "stop"},
                          {"text", text}};
            json out = {{"id", "q27-0"}, {"object", obj}, {"created", created},
                        {"model", served_name}, {"choices", json::array({choice})},
                        {"usage", {{"prompt_tokens", (int)prompt.size()},
                                   {"completion_tokens", n},
                                   {"total_tokens", (int)prompt.size() + n}}}};
            res.set_content(jdump(out), "application/json");
            return;
        }

        res.set_header("Content-Type", "text/event-stream");
        q27k::SampleParams samp = parse_sample(body);
        res.set_chunked_content_provider(
            "text/event-stream",
            [&, samp, prompt, n_max, created, chat, obj, objd, rt](size_t, httplib::DataSink& sink) {
                Slot& sl = claim_slot(prompt);
                auto sl_lease = slot_guard(sl);
                Engine& eng = *sl.eng;
                eng.samp = samp;
                q27::GpuGate::Lease lk(gpu_gate);
                double qw = ms_since(rt.t0);
                eng.on_round_gap = make_yield(eng);
                const int nm =
                    std::max(0, std::min(n_max, eng.max_ctx - (int)prompt.size() - (eng.ctx_round_reserve() - 1)));
                auto send = [&](const json& j) {
                    std::string s = "data: " + jdump(j) + "\n\n";
                    return sink.write(s.data(), s.size());
                };
                q27::Utf8Gate ugate;
                auto piece_chunk = [&](const std::string& piece) {
                    json delta = chat ? json{{"content", piece}} : json{};
                    json choice = chat
                        ? json{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}
                        : json{{"index", 0}, {"text", piece}, {"finish_reason", nullptr}};
                    return json{{"id", "q27-0"}, {"object", objd}, {"created", created},
                                {"model", served_name}, {"choices", json::array({choice})}};
                };
                eng.generate(prompt, nm, EOS, [&](int id) {
                    // empty pieces (control tokens, gate holdbacks) still probe
                    // the socket so a disconnected client stops generation
                    return send(piece_chunk(ugate.feed(tok.decode_one(id))));
                });
                eng.on_round_gap = nullptr;
                std::string tailp = ugate.flush();
                if (!tailp.empty()) send(piece_chunk(tailp));
                req_log(rt, qw, eng, sl.id);
                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
    };

    // ---------------- Anthropic /v1/messages ----------------
    // Request mapping (anthropic_msgs / anthropic_tools_json) lives in
    // api_common.h so count_tokens and the CPU self-tests share it.

    auto anthropic_400 = [](httplib::Response& res, const std::string& msg) {
        res.status = 400;
        res.set_content(q27::anthropic_error_json("invalid_request_error", msg),
                        "application/json");
    };

    // Claude Code calls this before compaction decisions; a 404 here means
    // CC estimates context blind and only discovers overflow by erroring.
    // Count = exactly what /v1/messages would prefill for the same body
    // (usage.input_tokens), whole-string encode (split-invariant at the P8
    // boundary, gated in test_tokenizer). CPU-only: no slot, no GPU gate.
    srv.Post("/v1/messages/count_tokens",
             [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { anthropic_400(res, "invalid JSON body"); return; }
        if (!body.contains("messages") || !body["messages"].is_array()) {
            anthropic_400(res, "messages: Field required");
            return;
        }
        std::string rendered = q27::chatml_prompt(
            q27::anthropic_msgs(body), q27::anthropic_tools_json(body), !no_think_srv);
        json out = {{"input_tokens", (long)tok.encode(rendered).size()}};
        res.set_content(jdump(out), "application/json");
    });

    srv.Post("/v1/messages", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { anthropic_400(res, "invalid JSON body"); return; }
        int n_max = body.value("max_tokens", 1024);
        bool stream = body.value("stream", false);
        json tools = q27::anthropic_tools_json(body);
        std::vector<std::string> tool_names_v;
        if (constrain_tools && tools.is_array())
            for (auto& t : tools)
                if (t.contains("function") && t["function"].contains("name"))
                    tool_names_v.push_back(t["function"]["name"].get<std::string>());
        auto tk0 = std::chrono::steady_clock::now();
        size_t stable_off = 0;
        std::string rendered =
            q27::chatml_prompt(q27::anthropic_msgs(body), tools, !no_think_srv, &stable_off);
        auto tk1 = std::chrono::steady_clock::now();
        // P8: split-encode at the stable boundary. Both turns encode the
        // shared history with the same split (the boundary always abuts the
        // <|im_start|> special, so tokenization is split-invariant there),
        // which is what makes the snapshot prefix-match next turn.
        std::vector<int> prompt = tok.encode(rendered.substr(0, stable_off));
        const int stable_len = (int)prompt.size();
        {
            std::vector<int> tailv = tok.encode(rendered.substr(stable_off));
            prompt.insert(prompt.end(), tailv.begin(), tailv.end());
        }
        auto tk2 = std::chrono::steady_clock::now();
        fprintf(stderr, "[timing] render %.1fms encode %.1fms (%zu chars -> %zu toks)\n",
                std::chrono::duration<double, std::milli>(tk1 - tk0).count(),
                std::chrono::duration<double, std::milli>(tk2 - tk1).count(),
                rendered.size(), prompt.size());
        // Anthropic-shaped context-limit refusal, BEFORE slot claim and the
        // SSE provider: the old path (engine end=refused inside a 200) reads
        // as retryable to Claude Code, which then loops the oversized prompt
        // instead of compacting. "prompt is too long" is CC's compact-now
        // signal. Ceiling is the shared max_prompt (largest slot minus the
        // depth-derived spec-round reserve, keeping n_max >= 1).
        if ((int)prompt.size() > max_prompt) {
            fprintf(stderr, "[ctx-limit] prompt=%zu max=%d -> 400\n", prompt.size(),
                    max_prompt);
            anthropic_400(res, q27::ctx_limit_error_message((int)prompt.size(), max_prompt));
            return;
        }
        if ((int)prompt.size() + n_max > max_slot_ctx)
            n_max = max_slot_ctx - (int)prompt.size();
        long rid = req_counter++;
        std::string mid = "msg_q27_" + std::to_string(rid);
        ReqTrace rt{rid, "anth", conv_fp(body), std::chrono::steady_clock::now(),
                    std::chrono::duration<double, std::milli>(tk2 - tk0).count()};

        if (!stream) {
            Slot& sl = claim_slot(prompt);
            auto sl_lease = slot_guard(sl);
            Engine& eng = *sl.eng;
            HookGuard hooks{eng}; // M1: clears tc hooks on unwind, pre slot-free
            eng.samp = parse_sample(body);
            q27::GpuGate::Lease lk(gpu_gate);
            double qw = ms_since(rt.t0);
            eng.on_round_gap = make_yield(eng);
            n_max = std::max(0, std::min(n_max, eng.max_ctx - (int)prompt.size() - (eng.ctx_round_reserve() - 1)));
            StreamSplitter sp;
            q27::Utf8Gate ugate;
            std::string think, text, tool_buf;
            std::vector<q27::ToolCall> calls;
            auto route = [&](StreamSplitter::Chan ch, const std::string& t) {
                if (ch == StreamSplitter::TOOL) { tool_buf += t; return; }
                if (!tool_buf.empty()) { // tool segment closed
                    calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));
                    tool_buf.clear();
                }
                (ch == StreamSplitter::THINK ? think : text) += t;
            };
            ToolConstrainer tc;
            tc.eng = &eng; tc.tok = &tok; tc.cache = &tool_mask_cache;
            tc.host2dev = &sl.tool_mask_host2dev;
            tc.enabled = constrain_tools && eng.samp.inv_temp <= 0.f; // constrained+sampled is Phase 3
            tc.begin(tool_names_v);
            eng.on_pending = [&](int id) { tc.on_pending(id); };
            eng.on_drafts = [&](const int* dr) { tc.on_drafts(dr); };
            if (tc.enabled)
                eng.on_round = [&](const int* em, int nr) { return tc.scan_round(em, nr); };
            int n = eng.generate(prompt, n_max, EOS, [&](int id) {
                tc.on_id(id);
                for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) route(ch, t);
                return true;
            }, stable_len);
            tc.end();
            eng.on_pending = nullptr;
            eng.on_drafts = nullptr;
            eng.on_round = nullptr;
            eng.on_round_gap = nullptr;
            req_log(rt, qw, eng, sl.id, tg_stats(tc));
            for (auto& [ch, t] : sp.feed(ugate.flush())) route(ch, t);
            for (auto& [ch, t] : sp.flush()) route(ch, t);
            if (!tool_buf.empty())
                calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));

            json content = json::array();
            std::string th = q27::strip_ws2(think), tx = q27::strip_ws2(text);
            if (!th.empty())
                content.push_back({{"type", "thinking"}, {"thinking", th},
                                   {"signature", "q27-local"}});
            bool any_call = false;
            int ci = 0;
            for (auto& c : calls) {
                if (!c.ok) { tx += (tx.empty() ? "" : "\n") + c.raw; continue; }
                any_call = true;
                (void)ci;
            }
            if (tools.is_array() && !tools.empty()) {
                // wrapper-less call recovery (see parse_bare_tool_calls)
                std::string pre;
                auto bcs = q27::parse_bare_tool_calls(tx, &pre, &tools);
                if (!bcs.empty()) {
                    fprintf(stderr, "[tool-fallback] %zu bare call(s) recovered (nonstream)\n",
                            bcs.size());
                    tx = pre;
                    for (auto& bc : bcs) calls.push_back(bc);
                    any_call = true;
                }
            }
            if (!tx.empty() || (!any_call && th.empty()))
                content.push_back({{"type", "text"}, {"text", tx}});
            for (auto& c : calls)
                if (c.ok)
                    content.push_back({{"type", "tool_use"},
                                       {"id", "toolu_q27_" + std::to_string(rid) + "_" +
                                                  std::to_string(ci++)},
                                       {"name", c.name}, {"input", c.arguments}});
            const char* sr = any_call ? "tool_use" : (n >= n_max ? "max_tokens" : "end_turn");
            json out = {{"id", mid}, {"type", "message"}, {"role", "assistant"},
                        {"model", served_name}, {"content", content},
                        {"stop_reason", sr}, {"stop_sequence", nullptr},
                        {"usage", {{"input_tokens", (int)prompt.size()},
                                   {"output_tokens", n}}}};
            res.set_content(jdump(out), "application/json");
            return;
        }

        res.set_header("Content-Type", "text/event-stream");
        const bool has_tools = tools.is_array() && !tools.empty();
        q27k::SampleParams samp = parse_sample(body);
        res.set_chunked_content_provider(
            "text/event-stream",
            [&, samp, prompt, n_max, mid, rid, has_tools, tool_names_v, tools, stable_len, rt](
                size_t, httplib::DataSink& sink) {
                Slot& sl = claim_slot(prompt);
                auto sl_lease = slot_guard(sl);
                Engine& eng = *sl.eng;
                HookGuard hooks{eng}; // M1: clears tc hooks on unwind, pre slot-free
                eng.samp = samp;
                q27::GpuGate::Lease lk(gpu_gate);
                double qw = ms_since(rt.t0);
                eng.on_round_gap = make_yield(eng);
                const int nm =
                    std::max(0, std::min(n_max, eng.max_ctx - (int)prompt.size() - (eng.ctx_round_reserve() - 1)));
                ToolConstrainer tc;
                tc.eng = &eng; tc.tok = &tok; tc.cache = &tool_mask_cache;
                tc.host2dev = &sl.tool_mask_host2dev;
                tc.enabled = constrain_tools && eng.samp.inv_temp <= 0.f; // constrained+sampled is Phase 3
                tc.begin(tool_names_v);
                int block_counter = 0, tool_counter = 0;
                bool any_call = false;
                auto ev = [&](const char* name, const json& j) {
                    std::string s = std::string("event: ") + name + "\ndata: " + jdump(j) + "\n\n";
                    return sink.write(s.data(), s.size());
                };
                json msg = {{"id", mid}, {"type", "message"}, {"role", "assistant"},
                            {"model", served_name}, {"content", json::array()},
                            {"stop_reason", nullptr}, {"stop_sequence", nullptr},
                            {"usage", {{"input_tokens", (int)prompt.size()}, {"output_tokens", 0}}}};
                ev("message_start", {{"type", "message_start"}, {"message", msg}});

                StreamSplitter sp;
                std::string tool_buf, text_accum;
                q27::Utf8Gate ugate;
                int idx = -1;       // open think/text block index, -1 = none
                int chan_open = -1; // 0 text, 1 think
                bool any = false;
                auto close_block = [&]() {
                    if (idx < 0) return;
                    if (chan_open == 1)
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "signature_delta"}, {"signature", "q27-local"}}}});
                    ev("content_block_stop", {{"type", "content_block_stop"}, {"index", idx}});
                    idx = -1;
                };
                auto open_block = [&](int chan) {
                    if (idx >= 0 && chan_open != chan) close_block();
                    if (idx < 0) {
                        idx = block_counter++;
                        json cb = chan == 1 ? json{{"type", "thinking"}, {"thinking", ""}}
                                            : json{{"type", "text"}, {"text", ""}};
                        ev("content_block_start", {{"type", "content_block_start"},
                                                   {"index", idx}, {"content_block", cb}});
                        chan_open = chan;
                        any = true;
                    }
                };
                auto emit_tool = [&]() {
                    auto c = q27::parse_tool_call(q27::strip_ws2(tool_buf));
                    tool_buf.clear();
                    if (!c.ok) { // malformed: surface as text so nothing is lost
                        open_block(0);
                        text_accum += c.raw;
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "text_delta"}, {"text", c.raw}}}});
                        return;
                    }
                    any_call = true;
                    close_block();
                    int ti = block_counter++;
                    std::string tid = "toolu_q27_" + std::to_string(rid) + "_" +
                                      std::to_string(tool_counter++);
                    ev("content_block_start",
                       {{"type", "content_block_start"}, {"index", ti},
                        {"content_block", {{"type", "tool_use"}, {"id", tid}, {"name", c.name},
                                           {"input", json::object()}}}});
                    ev("content_block_delta",
                       {{"type", "content_block_delta"}, {"index", ti},
                        {"delta", {{"type", "input_json_delta"},
                                   {"partial_json", jdump(c.arguments)}}}});
                    ev("content_block_stop", {{"type", "content_block_stop"}, {"index", ti}});
                };
                auto emit_seg = [&](StreamSplitter::Chan ch, const std::string& t) {
                    if (ch == StreamSplitter::TOOL) { tool_buf += t; return; }
                    if (!tool_buf.empty()) emit_tool();
                    if (t.empty()) return;
                    int chan = ch == StreamSplitter::THINK ? 1 : 0;
                    // suppress pure-whitespace text before the first block or between blocks
                    if (chan == 0 && idx < 0 && q27::strip_ws2(t).empty()) return;
                    open_block(chan);
                    if (chan == 0) text_accum += t;
                    if (chan == 1)
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "thinking_delta"}, {"thinking", t}}}});
                    else
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "text_delta"}, {"text", t}}}});
                };
                eng.on_pending = [&](int id) { tc.on_pending(id); };
                eng.on_drafts = [&](const int* dr) { tc.on_drafts(dr); };
                if (tc.enabled)
                    eng.on_round = [&](const int* em, int nr) { return tc.scan_round(em, nr); };
                int produced = eng.generate(prompt, nm, EOS, [&](int id) {
                    tc.on_id(id);
                    for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) emit_seg(ch, t);
                    return true;
                }, stable_len);
                tc.end();
                eng.on_pending = nullptr;
                eng.on_drafts = nullptr;
                eng.on_round = nullptr;
                eng.on_round_gap = nullptr;
                req_log(rt, qw, eng, sl.id, tg_stats(tc));
                for (auto& [ch, t] : sp.feed(ugate.flush())) emit_seg(ch, t);
                for (auto& [ch, t] : sp.flush()) emit_seg(ch, t);
                if (!tool_buf.empty()) emit_tool();
                if (has_tools) {
                    // wrapper-less call recovery: text already streamed as
                    // text_delta (cosmetic); the tool_use blocks still fire
                    std::string pre;
                    auto bcs = q27::parse_bare_tool_calls(text_accum, &pre, &tools);
                    if (!bcs.empty()) {
                        fprintf(stderr, "[tool-fallback] %zu bare call(s) recovered (stream)\n",
                                bcs.size());
                        any_call = true;
                        any = true;
                        close_block();
                        for (auto& bc : bcs) {
                            int ti = block_counter++;
                            std::string tid = "toolu_q27_" + std::to_string(rid) + "_" +
                                              std::to_string(tool_counter++);
                            ev("content_block_start",
                               {{"type", "content_block_start"}, {"index", ti},
                                {"content_block", {{"type", "tool_use"}, {"id", tid},
                                                   {"name", bc.name},
                                                   {"input", json::object()}}}});
                            ev("content_block_delta",
                               {{"type", "content_block_delta"}, {"index", ti},
                                {"delta", {{"type", "input_json_delta"},
                                           {"partial_json", jdump(bc.arguments)}}}});
                            ev("content_block_stop",
                               {{"type", "content_block_stop"}, {"index", ti}});
                        }
                    }
                }
                if (idx < 0 && !any) { // nothing at all: empty text block for validity
                    idx = block_counter++;
                    chan_open = 0;
                    ev("content_block_start", {{"type", "content_block_start"}, {"index", idx},
                                               {"content_block", {{"type", "text"}, {"text", ""}}}});
                }
                close_block();
                const char* sr = any_call ? "tool_use"
                                          : (produced >= nm ? "max_tokens" : "end_turn");
                ev("message_delta", {{"type", "message_delta"},
                                     {"delta", {{"stop_reason", sr}, {"stop_sequence", nullptr}}},
                                     {"usage", {{"output_tokens", produced}}}});
                ev("message_stop", {{"type", "message_stop"}});
                sink.done();
                return true;
            });
    });

    // ---------------- OpenAI Responses API (Codex CLI) ----------------
    // Wire facts from codex-rs (v0.143): client keys off the JSON `type` field
    // in SSE data (event: lines ignored); the agent loop consumes only
    // response.output_item.done items; response.completed{response:{id}} is the
    // required terminator; function_call.arguments is a JSON-encoded STRING;
    // tool results arrive as function_call_output with a bare-string output.
    // 400 is fatal to Codex, 500 retries -- so tolerate quirks, 500 on bugs.

    srv.Post("/v1/responses", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return; }

        long rid = req_counter++;
        std::string resp_id = "resp_q27_" + std::to_string(rid);

        // tools: flat function entries pass through; `custom` freeform tools
        // (apply_patch) are bridged to a one-string-param function; hosted tool
        // types (web_search etc.) are skipped, never rejected.
        json tools = json::array();
        std::set<std::string> custom_names;
        if (body.contains("tools"))
            for (auto& t : body["tools"]) {
                std::string ty = t.value("type", "");
                if (ty == "function") {
                    tools.push_back({{"type", "function"},
                                     {"function", {{"name", t.value("name", "")},
                                                   {"description", t.value("description", "")},
                                                   {"parameters", t.contains("parameters")
                                                                      ? t["parameters"]
                                                                      : json::object()}}}});
                } else if (ty == "custom") {
                    std::string cn = t.value("name", "");
                    custom_names.insert(cn);
                    json params = {{"type", "object"},
                                   {"properties",
                                    {{"input", {{"type", "string"},
                                                {"description", "The complete raw input text "
                                                                "for this tool."}}}}},
                                   {"required", json::array({"input"})}};
                    tools.push_back({{"type", "function"},
                                     {"function", {{"name", cn},
                                                   {"description", t.value("description", "")},
                                                   {"parameters", params}}}});
                }
            }

        // input -> messages. instructions is the system prompt.
        std::vector<Msg> msgs;
        if (body.contains("instructions") && body["instructions"].is_string())
            msgs.push_back({"system", body["instructions"]});
        // content flattening shared with conv_fp (text_of above)
        if (body.contains("input")) {
            if (body["input"].is_string()) {
                msgs.push_back({"user", body["input"]});
            } else if (body["input"].is_array()) {
                for (auto& it : body["input"]) {
                    std::string ty = it.value("type", "message");
                    if (ty == "message") {
                        std::string role = it.value("role", "user");
                        if (role == "developer") role = "system";
                        msgs.push_back({role, text_of(it["content"])});
                    } else if (ty == "function_call" || ty == "custom_tool_call") {
                        json args;
                        if (ty == "function_call") {
                            try { args = json::parse(it.value("arguments", "{}")); }
                            catch (...) { args = it.value("arguments", ""); }
                        } else {
                            args = {{"input", it.value("input", "")}};
                        }
                        msgs.push_back({"assistant",
                                        q27::tool_call_text(it.value("name", ""), args)});
                    } else if (ty == "function_call_output" || ty == "custom_tool_call_output") {
                        std::string out;
                        if (it.contains("output")) {
                            if (it["output"].is_string()) out = it["output"];
                            else out = text_of(it["output"]);
                        }
                        msgs.push_back({"user", q27::tool_response_text(out)});
                    }
                    // reasoning items in history are dropped: prior-turn thinking
                    // is not re-fed (matches the chat template's behavior)
                }
            }
        }
        // merge consecutive same-role messages (tool call + output sequences)
        std::vector<Msg> merged;
        for (auto& m : msgs) {
            if (!merged.empty() && merged.back().role == m.role)
                merged.back().content += "\n" + m.content;
            else
                merged.push_back(m);
        }

        int n_max = body.value("max_output_tokens", 4096);
        auto tk0 = std::chrono::steady_clock::now();
        std::vector<int> prompt =
            tok.encode(q27::chatml_prompt(merged, tools, !no_think_srv));
        ReqTrace rt{rid, "resp", conv_fp(body), std::chrono::steady_clock::now(),
                    ms_since(tk0)};
        // review follow-up 2026-07-09 #3: the bound includes the spec-round
        // reserve (max_prompt), so a prompt that passes can never have its
        // n_max floored to 0 by the routed slot's clamp (empty 200/stream)
        if ((int)prompt.size() > max_prompt) {
            res.status = 400; // context_length_exceeded is fatal-class for codex, correctly
            res.set_content("{\"error\":{\"code\":\"context_length_exceeded\"}}",
                            "application/json");
            return;
        }
        if ((int)prompt.size() + n_max > max_slot_ctx)
            n_max = max_slot_ctx - (int)prompt.size();
        bool stream = body.value("stream", false);

        // shared generation -> output items
        struct GenOut { json items = json::array(); int produced = 0; };
        auto make_item_cbs = [&](json& items, int& tool_counter,
                                 std::function<bool(const std::string&)> on_text_delta) {
            // returns the segment router; caller finishes with finish()
            struct Ctx {
                std::string think, text, tool_buf;
            };
            auto ctx = std::make_shared<Ctx>();
            auto flush_think = [&items, ctx, rid]() {
                std::string th = q27::strip_ws2(ctx->think);
                ctx->think.clear();
                if (th.empty()) return json();
                json r = {{"type", "reasoning"}, {"id", "rs_q27_" + std::to_string(rid)},
                          {"summary", json::array({{{"type", "summary_text"}, {"text", th}}})},
                          {"encrypted_content", nullptr}};
                items.push_back(r);
                return r;
            };
            auto flush_text = [&items, ctx, rid]() {
                std::string tx = q27::strip_ws2(ctx->text);
                ctx->text.clear();
                if (tx.empty()) return json();
                json m = {{"type", "message"}, {"id", "msg_q27_" + std::to_string(rid)},
                          {"role", "assistant"}, {"status", "completed"},
                          {"content",
                           json::array({{{"type", "output_text"}, {"text", tx},
                                         {"annotations", json::array()}}})}};
                items.push_back(m);
                return m;
            };
            auto flush_tool = [&items, ctx, rid, &tool_counter, &custom_names]() {
                auto c = q27::parse_tool_call(q27::strip_ws2(ctx->tool_buf));
                ctx->tool_buf.clear();
                std::string cid = "call_q27_" + std::to_string(rid) + "_" +
                                  std::to_string(tool_counter++);
                json it;
                if (!c.ok) { // malformed call: surface as text so codex shows it
                    it = {{"type", "message"}, {"role", "assistant"}, {"status", "completed"},
                          {"content", json::array({{{"type", "output_text"}, {"text", c.raw},
                                                    {"annotations", json::array()}}})}};
                } else if (custom_names.count(c.name)) {
                    std::string input = c.arguments.is_object() && c.arguments.contains("input") &&
                                                c.arguments["input"].is_string()
                                            ? c.arguments["input"].get<std::string>()
                                            : jdump(c.arguments);
                    it = {{"type", "custom_tool_call"}, {"call_id", cid}, {"name", c.name},
                          {"input", input}};
                } else {
                    it = {{"type", "function_call"}, {"call_id", cid}, {"name", c.name},
                          {"arguments", jdump(c.arguments)}};
                }
                items.push_back(it);
                return it;
            };
            return std::make_tuple(ctx, flush_think, flush_text, flush_tool);
        };

        if (!stream) {
            Slot& sl = claim_slot(prompt);
            auto sl_lease = slot_guard(sl);
            Engine& eng = *sl.eng;
            eng.samp = parse_sample(body);
            q27::GpuGate::Lease lk(gpu_gate);
            double qw = ms_since(rt.t0);
            eng.on_round_gap = make_yield(eng);
            n_max = std::max(0, std::min(n_max, eng.max_ctx - (int)prompt.size() - (eng.ctx_round_reserve() - 1)));
            json items = json::array();
            int tool_counter = 0;
            auto [ctx, flush_think, flush_text, flush_tool] =
                make_item_cbs(items, tool_counter, nullptr);
            StreamSplitter sp;
            auto route = [&](StreamSplitter::Chan ch, const std::string& t) {
                if (ch == StreamSplitter::TOOL) {
                    if (!ctx->think.empty()) flush_think();
                    if (!ctx->text.empty()) flush_text();
                    ctx->tool_buf += t;
                    return;
                }
                if (!ctx->tool_buf.empty()) flush_tool();
                if (ch == StreamSplitter::THINK) ctx->think += t;
                else {
                    if (!ctx->think.empty()) flush_think();
                    ctx->text += t;
                }
            };
            q27::Utf8Gate ugate;
            int produced = eng.generate(prompt, n_max, EOS, [&](int id) {
                for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) route(ch, t);
                return true;
            });
            eng.on_round_gap = nullptr;
            req_log(rt, qw, eng, sl.id);
            for (auto& [ch, t] : sp.feed(ugate.flush())) route(ch, t);
            for (auto& [ch, t] : sp.flush()) route(ch, t);
            if (!ctx->tool_buf.empty()) flush_tool();
            flush_think();
            flush_text();
            json out = {{"id", resp_id}, {"object", "response"}, {"status", "completed"},
                        {"model", served_name}, {"output", items},
                        {"usage", {{"input_tokens", (int)prompt.size()},
                                   {"output_tokens", produced},
                                   {"total_tokens", (int)prompt.size() + produced}}}};
            res.set_content(jdump(out), "application/json");
            return;
        }

        res.set_header("Content-Type", "text/event-stream");
        q27k::SampleParams samp = parse_sample(body);
        res.set_chunked_content_provider(
            "text/event-stream",
            [&, samp, prompt, n_max, resp_id, rid, custom_names, tools, rt](size_t, httplib::DataSink& sink) {
                Slot& sl = claim_slot(prompt);
                auto sl_lease = slot_guard(sl);
                Engine& eng = *sl.eng;
                eng.samp = samp;
                q27::GpuGate::Lease lk(gpu_gate);
                double qw = ms_since(rt.t0);
                eng.on_round_gap = make_yield(eng);
                const int nm =
                    std::max(0, std::min(n_max, eng.max_ctx - (int)prompt.size() - (eng.ctx_round_reserve() - 1)));
                auto ev = [&](const json& j) {
                    // codex keys off data.type; the event: line is decorative
                    std::string s = "event: " + j.value("type", std::string("x")) +
                                    "\ndata: " + jdump(j) + "\n\n";
                    return sink.write(s.data(), s.size());
                };
                ev({{"type", "response.created"},
                    {"response", {{"id", resp_id}, {"object", "response"},
                                  {"status", "in_progress"}}}});

                json items = json::array();
                int tool_counter = 0;
                std::set<std::string> cn = custom_names;
                std::string think, text, tool_buf, text_accum;
                int out_index = 0;
                auto item_done = [&](const json& it) {
                    ev({{"type", "response.output_item.done"}, {"output_index", out_index++},
                        {"item", it}});
                    items.push_back(it);
                };
                auto flush_think = [&]() {
                    std::string th = q27::strip_ws2(think);
                    think.clear();
                    if (th.empty()) return;
                    item_done({{"type", "reasoning"}, {"id", "rs_q27_" + std::to_string(rid)},
                               {"summary",
                                json::array({{{"type", "summary_text"}, {"text", th}}})},
                               {"encrypted_content", nullptr}});
                };
                // codex 0.143 enforces the item lifecycle: an output_text.delta
                // needs an already-OPEN item (else "OutputTextDelta without
                // active item" and codex aborts the turn -- the T5/T8 failure).
                // Open the message item + content part before the first delta;
                // flush closes the full added->delta->done sequence. msg_index
                // reserves the current out_index while streaming (only one item
                // is ever open -- route flushes think/text before a tool).
                const std::string msg_id = "msg_q27_" + std::to_string(rid);
                int msg_index = -1;
                auto open_text = [&]() {
                    if (msg_index >= 0) return;
                    msg_index = out_index;
                    ev({{"type", "response.output_item.added"}, {"output_index", msg_index},
                        {"item", {{"type", "message"}, {"id", msg_id}, {"role", "assistant"},
                                  {"status", "in_progress"}, {"content", json::array()}}}});
                    ev({{"type", "response.content_part.added"}, {"item_id", msg_id},
                        {"output_index", msg_index}, {"content_index", 0},
                        {"part", {{"type", "output_text"}, {"text", ""},
                                  {"annotations", json::array()}}}});
                };
                auto flush_text = [&]() {
                    if (msg_index < 0) { text.clear(); return; }
                    std::string tx = q27::strip_ws2(text);
                    text.clear();
                    ev({{"type", "response.output_text.done"}, {"item_id", msg_id},
                        {"output_index", msg_index}, {"content_index", 0}, {"text", tx}});
                    ev({{"type", "response.content_part.done"}, {"item_id", msg_id},
                        {"output_index", msg_index}, {"content_index", 0},
                        {"part", {{"type", "output_text"}, {"text", tx},
                                  {"annotations", json::array()}}}});
                    json it = {{"type", "message"}, {"id", msg_id}, {"role", "assistant"},
                               {"status", "completed"},
                               {"content", json::array({{{"type", "output_text"}, {"text", tx},
                                                         {"annotations", json::array()}}})}};
                    ev({{"type", "response.output_item.done"}, {"output_index", msg_index},
                        {"item", it}});
                    items.push_back(it);
                    out_index = msg_index + 1;
                    msg_index = -1;
                };
                auto flush_tool = [&]() {
                    auto c = q27::parse_tool_call(q27::strip_ws2(tool_buf));
                    tool_buf.clear();
                    std::string cid = "call_q27_" + std::to_string(rid) + "_" +
                                      std::to_string(tool_counter++);
                    if (!c.ok) {
                        item_done({{"type", "message"}, {"role", "assistant"},
                                   {"status", "completed"},
                                   {"content",
                                    json::array({{{"type", "output_text"}, {"text", c.raw},
                                                  {"annotations", json::array()}}})}});
                    } else if (cn.count(c.name)) {
                        std::string input =
                            c.arguments.is_object() && c.arguments.contains("input") &&
                                    c.arguments["input"].is_string()
                                ? c.arguments["input"].get<std::string>()
                                : jdump(c.arguments);
                        item_done({{"type", "custom_tool_call"}, {"call_id", cid},
                                   {"name", c.name}, {"input", input}});
                    } else {
                        item_done({{"type", "function_call"}, {"call_id", cid}, {"name", c.name},
                                   {"arguments", jdump(c.arguments)}});
                    }
                };
                auto route = [&](StreamSplitter::Chan ch, const std::string& t) {
                    if (ch == StreamSplitter::TOOL) {
                        if (!think.empty()) flush_think();
                        if (!text.empty()) flush_text();
                        tool_buf += t;
                        return;
                    }
                    if (!tool_buf.empty()) flush_tool();
                    if (ch == StreamSplitter::THINK) { think += t; return; }
                    if (!think.empty()) flush_think();
                    if (text.empty() && q27::strip_ws2(t).empty()) return;
                    open_text();
                    text += t;
                    text_accum += t; // survives flush_text for bare-call recovery
                    ev({{"type", "response.output_text.delta"}, {"item_id", msg_id},
                        {"output_index", msg_index}, {"content_index", 0}, {"delta", t}});
                };
                StreamSplitter sp;
                q27::Utf8Gate ugate;
                int produced = eng.generate(prompt, nm, EOS, [&](int id) {
                    for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) route(ch, t);
                    return true;
                });
                eng.on_round_gap = nullptr;
                req_log(rt, qw, eng, sl.id);
                for (auto& [ch, t] : sp.feed(ugate.flush())) route(ch, t);
                for (auto& [ch, t] : sp.flush()) route(ch, t);
                if (!tool_buf.empty()) flush_tool();
                flush_think();
                flush_text();
                // wrapper-less call recovery (parity with the Anthropic path):
                // the model sometimes emits a bare {"name":...,"arguments":...}
                // as text, no <tool_call> wrapper -- it already streamed as an
                // output_text item (cosmetic), but codex needs it as a
                // function_call to execute. Emit the recovered calls as items
                // after the text (T5 task-queue failed exactly here). UNLIKE
                // the Anthropic path, recovery runs even with empty `tools`:
                // codex registers its shell tool as a hosted type this handler
                // skips (so `tools` is empty) yet the model still emits bare
                // calls for it -- the parser only recovers well-shaped
                // name+arguments JSON, which codex validates against its own
                // tool set, so a spurious recovery is harmless.
                {
                    std::string pre;
                    auto bcs = q27::parse_bare_tool_calls(text_accum, &pre,
                                                          tools.empty() ? nullptr : &tools);
                    if (!bcs.empty())
                        fprintf(stderr, "[tool-fallback] %zu bare call(s) recovered (resp)\n",
                                bcs.size());
                    for (auto& bc : bcs) {
                        std::string cid = "call_q27_" + std::to_string(rid) + "_" +
                                          std::to_string(tool_counter++);
                        if (cn.count(bc.name)) {
                            std::string input =
                                bc.arguments.is_object() && bc.arguments.contains("input") &&
                                        bc.arguments["input"].is_string()
                                    ? bc.arguments["input"].get<std::string>()
                                    : jdump(bc.arguments);
                            item_done({{"type", "custom_tool_call"}, {"call_id", cid},
                                       {"name", bc.name}, {"input", input}});
                        } else {
                            item_done({{"type", "function_call"}, {"call_id", cid},
                                       {"name", bc.name}, {"arguments", jdump(bc.arguments)}});
                        }
                    }
                }
                ev({{"type", "response.completed"},
                    {"response", {{"id", resp_id}, {"object", "response"},
                                  {"status", "completed"}, {"output", items},
                                  {"usage", {{"input_tokens", (int)prompt.size()},
                                             {"input_tokens_details", {{"cached_tokens", 0}}},
                                             {"output_tokens", produced},
                                             {"output_tokens_details", {{"reasoning_tokens", 0}}},
                                             {"total_tokens", (int)prompt.size() + produced}}}}}});
                sink.done();
                return true;
            });
    });

    srv.Post("/v1/chat/completions",
             [&](const httplib::Request& r, httplib::Response& s) { handle(r, s, true); });
    srv.Post("/v1/completions",
             [&](const httplib::Request& r, httplib::Response& s) { handle(r, s, false); });

    fprintf(stderr, "q27-server listening on http://%s:%d (ctx %d, %s head)\n", host.c_str(),
            port, ctx, fast ? "fast" : "faithful");
    srv.listen(host.c_str(), port);
    return 0;
}
