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
#include <optional>
#include <set>
#include <tuple>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>

#include "engine.cuh"
#include "tokenizer.h"
#include "api_common.h"
#include "conductor.h"
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
    return q27::json_dump_replace(j);
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
    // jnum/jint/jbool (api_common.h), not value(): a present-but-null field is
    // how many OpenAI-compatible clients spell "unset", and value() throws on
    // it -> httplib 500. Read null/wrong-typed as absent.
    double temp = q27::jnum(body, "temperature", force_temp);
    if (temp > 0.0) {
        s.inv_temp = (float)(1.0 / temp);
        double tp = q27::jnum(body, "top_p", force_tp);
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

// Reasoning budgets are observed through Engine::on_round before host emission.
// The whole accepted round is parsed first, so a natural close later in that
// round wins. When an overshooting round would consume reserved close/answer
// capacity, only the prefix through the trip is retained and re-finished. On
// sampled paths that refinish immediately precedes the forced close, so the
// provisional argmax pending is replaced before any next-token decision.
struct ReasoningBudgetObserver {
    q27::Tokenizer& tok;
    q27::ThinkBudgetState& state;
    Engine& eng;
    Engine::DecodeTask& task;
    const std::vector<int>& close_ids;
    StreamSplitter split;
    q27::Utf8Gate gate;

    ReasoningBudgetObserver(q27::Tokenizer& tok_, q27::ThinkBudgetState& state_,
                            Engine& eng_, Engine::DecodeTask& task_,
                            const std::vector<int>& close_ids_,
                            StreamSplitter::Chan initial)
        : tok(tok_), state(state_), eng(eng_), task(task_), close_ids(close_ids_) {
        split.chan = initial;
    }

    void apply(q27::ThinkBudgetAction action, int current_public_tokens = 0,
               int current_context_tokens = -1) {
        if (action == q27::ThinkBudgetAction::NONE ||
            eng.reasoning_transition_active(task))
            return;
        eng.force_reasoning_close(
            task, close_ids, action == q27::ThinkBudgetAction::FORCE_PUBLIC,
            current_public_tokens, current_context_tokens);
    }

    void start() { apply(state.start(split.chan)); }

    bool observe_token(q27::ThinkBudgetState& target_state,
                       StreamSplitter& target_split, q27::Utf8Gate& target_gate,
                       int id, bool forced) {
        const auto before = target_split.chan;
        const auto segments = target_split.feed(target_gate.feed(tok.decode_one(id)));
        target_state.observe(before, target_split.chan, forced);
        for (const auto& [ch, text] : segments)
            if (ch != StreamSplitter::THINK &&
                text.find_first_not_of(" \t\r\n") != std::string::npos)
                return true;
        return false;
    }

    bool observe_token(int id, bool forced) {
        return observe_token(state, split, gate, id, forced);
    }

    int visible_prefix(const int* ids, int n, bool forced, bool* hits_eos) const {
        *hits_eos = false;
        if (forced) return n;
        int visible = std::min(n, std::max(0, task.n_max - task.emitted));
        for (int i = 0; i < visible; i++)
            if (ids[i] == task.eos) {
                *hits_eos = true;
                return i;
            }
        return visible;
    }

    // Pure planning pass. Tool scanning runs only over the prefix this returns,
    // so a discarded suffix cannot engage or poison the stateful grammar hook.
    int preview_round(const int* ids, int n) {
        const bool forced = task.round_forced;
        bool hits_eos = false;
        const int visible = visible_prefix(ids, n, forced, &hits_eos);
        q27::ThinkBudgetState trial_state = state;
        StreamSplitter trial_split = split;
        q27::Utf8Gate trial_gate = gate;
        int trip_prefix = -1;
        bool public_answer_after_trip = false;
        for (int i = 0; i < visible; i++) {
            const bool emitted_public =
                observe_token(trial_state, trial_split, trial_gate, ids[i], forced);
            if (trip_prefix > 0 && emitted_public) public_answer_after_trip = true;
            if (!forced && trip_prefix < 0 && trial_state.limit >= 0 &&
                !trial_state.transition_pending &&
                trial_split.chan == StreamSplitter::THINK &&
                trial_state.used >= trial_state.limit)
                trip_prefix = i + 1;
        }
        if (hits_eos) return -1;

        const auto action = trial_state.finish_round(trial_split.chan);
        const bool natural_close_after_trip =
            trip_prefix > 0 && action == q27::ThinkBudgetAction::NONE &&
            trial_split.chan != StreamSplitter::THINK;
        const bool buffered_public_answer =
            trial_split.chan != StreamSplitter::THINK &&
            (!trial_gate.pend.empty() ||
             trial_split.hold.find_first_not_of(" \t\r\n") != std::string::npos);
        const bool natural_close_keeps_answer = public_answer_after_trip ||
            buffered_public_answer || task.n_max - task.emitted - visible >= 1;
        const bool force_full_round_fits =
            action != q27::ThinkBudgetAction::NONE &&
            eng.reasoning_close_fits(
                task, visible, visible, (int)close_ids.size(),
                action == q27::ThinkBudgetAction::FORCE_PUBLIC);
        if ((action == q27::ThinkBudgetAction::NONE &&
             (!natural_close_after_trip || natural_close_keeps_answer)) ||
            force_full_round_fits)
            return -1;
        return trip_prefix > 0 && trip_prefix < n ? trip_prefix : -1;
    }

    void commit_round(const int* ids, int n) {
        const bool forced = task.round_forced;
        bool hits_eos = false;
        const int visible = visible_prefix(ids, n, forced, &hits_eos);
        for (int i = 0; i < visible; i++) observe_token(ids[i], forced);
        // EOS wins immediately. Count only tokens before it, but never queue a
        // close that post_round cannot commit after taking the EOS exit.
        if (hits_eos) return;
        apply(state.finish_round(split.chan), visible, visible);
    }

    int observe_round(const int* ids, int n) {
        const int m = preview_round(ids, n);
        const int kept = (m >= 1 && m < n) ? m : n;
        commit_round(ids, kept);
        return m;
    }

};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s model.q27 model.tok [--port N] [--host H] [--ctx C]\n"
                "  [--api-key KEY] [--api-key-file PATH]\n"
                "  Defaults (2026-07-10) = the measured Claude-Code stack: fp8 KV +\n"
                "  Q27_PMIN=0.5 + Q27_MAXD=auto7 + Q27_SUFFIX_W=<W_MAX> + Q27_FD=mma (sm_89+)\n"
                "  + fast-head + no-think + phase stats; --ctx auto-sizes to VRAM\n"
                "  (auto-ctx cap 262144 fp8/turbo3, 131072 fp16; single-slot). Escapes:\n"
                "  Q27_PROFILE=ref (conservative\n"
                "  reference: fp16/ungated/no-suffix/fd2), any individual Q27_* env,\n"
                "  --kv-fp16 --no-fast-head --think --request-think (honor per-\n"
                "  request enable_thinking; off by default). The CLI keeps reference\n"
                "  defaults (bitwise canonical).\n"
                "  Reasoning budget: prompt-seeded <think> blocks default to half\n"
                "  the request's max_tokens and are force-closed on trip so the\n"
                "  answer still gets written. A no-think request keeps ordinary\n"
                "  speculative decode; --think-budget N (N>0) or an explicit\n"
                "  request budget also arms a later model-generated block.\n"
                "  --think-budget 0 disables. Request conventions: thinking.\n"
                "  budget_tokens (Anthropic), thinking_token_budget (OpenAI/Qwen),\n"
                "  or chat_template_kwargs.thinking_budget -- honored only under\n"
                "  --request-think. A trip is reported in reasoning usage.\n"
                "  Auth: no API key is required by default (loopback-only is the\n"
                "  safety net). --api-key KEY may repeat; --api-key-file PATH loads\n"
                "  one key per line (# comments ignored); Q27_API_KEY adds one more.\n"
                "  Any configured key is accepted via 'Authorization: Bearer <key>'\n"
                "  (OpenAI/llama.cpp convention) or 'x-api-key: <key>' (Anthropic\n"
                "  convention, what Claude Code sends) on every endpoint except\n"
                "  /health.\n"
                "  Prefix cache (P16, opt-in, writes to disk): --prefix-cache DIR\n"
                "  persists the P8 stable prefix so a RESTART or a fresh\n"
                "  conversation restores instead of re-prefilling. Tuning:\n"
                "  --prefix-cache-max-gb 20 --prefix-cache-min 4096\n"
                "  --prefix-cache-max-tokens 32768 --prefix-cache-step 8192.\n"
                "  max-tokens also sizes the staging buffers: TWO per slot\n"
                "  (restore and persist, so a restore never waits on a write),\n"
                "  both pinned at boot rather than on the first request.\n"
                "  --prefix-cache-ram-gb 0 adds a pinned host-RAM tier above the\n"
                "  disk; off by default (the boot prefetch already makes reads\n"
                "  36-39 ms, so it saves little -- see the plan doc).\n",
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
    bool req_think = false; // --request-think: honor per-request thinking fields (else ignored)
    // --think-budget: cap on tokens generated inside a <think> block. <0
    // (default) = THINK_BUDGET_FRAC for prompt-seeded thinking only, preserving
    // speculative width on ordinary no-think requests; 0 = unbounded; >0 = an
    // explicit absolute cap that also arms a later model-generated block.
    int think_budget_flag = -1;
    std::vector<std::string> api_keys;
    q27::PrefixCacheCfg pfx_cfg; // P16: root empty = off
    double pfx_ram_gb = 0;       // P16c: host-RAM tier budget (0 = off)
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            ++i; // "auto" -> -1 (VRAM auto-size). atoi("auto") is 0, which would
                 // SILENTLY start a ctx-0 server (issue #6); the docs advertise
                 // --ctx auto, so honor it. (Omitting --ctx also auto-sizes.)
            ctx = strcmp(argv[i], "auto") == 0 ? -1 : atoi(argv[i]);
        }
        else if (!strcmp(argv[i], "--slots") && i + 1 < argc) n_slots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--slot1-ctx") && i + 1 < argc) slot1_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fast-head")) fast_flag = 1;
        else if (!strcmp(argv[i], "--no-fast-head")) fast_flag = 0;
        else if (!strcmp(argv[i], "--no-think")) think_flag = 0;
        else if (!strcmp(argv[i], "--think")) think_flag = 1;
        else if (!strcmp(argv[i], "--request-think")) req_think = true;
        else if (!strcmp(argv[i], "--think-budget") && i + 1 < argc)
            think_budget_flag = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--constrain-tools")) constrain_tools = true;
        else if (!strcmp(argv[i], "--kv-fp16")) kv_fp16 = true;
        // P16 persistent prefix cache (opt-in: it writes to the user's disk)
        else if (!strcmp(argv[i], "--prefix-cache") && i + 1 < argc) pfx_cfg.root = argv[++i];
        else if (!strcmp(argv[i], "--prefix-cache-max-gb") && i + 1 < argc)
            pfx_cfg.max_bytes = (size_t)(atof(argv[++i]) * 1e9);
        else if (!strcmp(argv[i], "--prefix-cache-min") && i + 1 < argc)
            pfx_cfg.min_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefix-cache-max-tokens") && i + 1 < argc)
            pfx_cfg.max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefix-cache-step") && i + 1 < argc)
            pfx_cfg.step_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefix-cache-ram-gb") && i + 1 < argc)
            pfx_ram_gb = atof(argv[++i]);
        // Auth config is fail-LOUD in both directions: an empty key can never
        // authenticate anything (api_key_valid rejects an empty `provided`
        // before comparing), so accepting one would stand up a server that
        // 401s every request; and a key FILE that opens but yields nothing
        // usable would stand up a server with auth silently OFF, which is the
        // opposite of what the operator asked for. Refuse both at boot.
        else if (!strcmp(argv[i], "--api-key") && i + 1 < argc) {
            if (!argv[++i][0]) {
                fprintf(stderr, "error: --api-key: empty key (an empty key can never match)\n");
                return 1;
            }
            api_keys.push_back(argv[i]);
        }
        else if (!strcmp(argv[i], "--api-key-file") && i + 1 < argc) {
            const size_t before = api_keys.size();
            if (!q27::load_api_key_file(argv[++i], &api_keys)) {
                fprintf(stderr, "error: --api-key-file %s: could not open\n", argv[i]);
                return 1;
            }
            if (api_keys.size() == before) {
                fprintf(stderr,
                        "error: --api-key-file %s: no keys found (every line blank or a "
                        "#comment) -- refusing to start with auth silently disabled\n",
                        argv[i]);
                return 1;
            }
        }
    }
    // Q27_API_KEY: a second, additive source (not exclusive with the CLI
    // flags above -- all configured keys are valid simultaneously, matching
    // --api-key-file's multi-key semantics). Preferred for containerized
    // deployments where CLI args are visible via `ps` but env vars set
    // through the orchestrator's secret store are not.
    if (const char* envkey = getenv("Q27_API_KEY"))
        if (envkey[0]) api_keys.push_back(envkey);
    // The existing loopback-by-default posture (see the comment above host's
    // declaration) was this server's ONLY safety net before auth existed.
    // Now that --api-key/--api-key-file/Q27_API_KEY exist, warn loudly
    // (not refuse -- some deployments intentionally run without auth behind
    // their OWN reverse-proxy auth layer, and silently breaking that on
    // upgrade would be worse) when binding non-loopback with none configured.
    if (api_keys.empty() && host != "127.0.0.1" && host != "localhost" && host != "::1")
        fprintf(stderr,
                "WARNING: binding %s with NO API key configured (--api-key / "
                "--api-key-file / Q27_API_KEY) -- this server will accept "
                "unauthenticated requests from anyone who can reach it.\n",
                host.c_str());
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
    // 2026-07-16 BATCH DEFAULTS FLIP: record whether Q27_BATCH came from the
    // USER'S environment BEFORE the profile block below can default it on.
    // The M2 guard at the conductor site keys its two-tier semantics on this
    // bool: user-explicit + incompatible env = refuse (exit 1, unchanged);
    // profile-default + incompatible env = auto-disable and serve.
    const bool batch_env_user = getenv("Q27_BATCH") != nullptr;
    if (!ref_profile) {
        if (cc_arch >= 89) {
            setenv("Q27_KV", "fp8", 0);
            setenv("Q27_FD", "mma", 0);
        } else if (cc_arch >= 80) {
            // H16 (fp16-MMA) verify: Ampere gets the mma path too
            // (2026-07-12, docs/plans/2026-07-12-fdmma-f16.md).
            //
            // KV defaulted to turbo3 here from 2026-07-17 (Gabe sign-off):
            // no fp8 HW on Ampere, and the 5090's turbo3 decode tax INVERTS
            // on a bandwidth-starved part -- 3090 q4s decode narr +1.1% /
            // code +9.8% over fp16 at 4.27x the context (262144 on 24 GB,
            // needle 6/6 @233K; BUILDLOG 2026-07-17).
            //
            // 2026-08-01 (e), Gabe sign-off: turbo5k REPLACES turbo3 as the
            // Ampere default. It carries 43% fewer catastrophic positions
            // (65 vs 114 against an fp16 reference on the 64K agentic
            // corpus, BUILDLOG 2026-08-01 (b)) at 1.14x turbo3's per-round
            // decode cost once the H16 mma leg lands (BUILDLOG (d)) -- and
            // Ampere is precisely where the tail mattered most, because fp8
            // is unavailable and turbo3 was the only route to a deep window.
            //
            // THE TRADE IS CONTEXT, and it is real: turbo5k is 19008 B/token
            // against turbo3's 14400, so auto-ctx shrinks to 75.8% of the
            // turbo3 window -- on the q4s tier a bare 24 GB boot sizes to
            // ~159744 instead of ~208896. If a deployment needs the deeper
            // window more than the tail, Q27_KV=turbo3 restores it exactly
            // (setenv(overwrite=0): any user Q27_KV wins, and
            // Q27_PROFILE=ref still keeps fp16).
            //
            // UNMEASURED, stated: the 1.14x decode ratio is from sm_120. On
            // Ampere BOTH formats run the same H16 kernel (no e4m3 leg
            // exists here), so the comparison should transfer at least as
            // well -- but it has not been run on a 3090. One accept_kv_ab
            // run there closes it.
            setenv("Q27_KV", "turbo5k", 0);
            setenv("Q27_FD", "mma", 0);
        }
        setenv("Q27_PMIN", "0.5", 0);
        setenv("Q27_MAXD", "auto7", 0);
        setenv("Q27_SUFFIX", "1", 0);
        // Tiny prompts through the CHUNKED prefill path (2026-07-17, Ampere
        // tuning pass): the serial walk below Q27_PF_BATCH_MIN costs ~22ms
        // per token on sm_86 (TTFT 567ms on a 23-token prompt) and clears
        // the slot's prefix cache. ref profile keeps the engine default 32.
        setenv("Q27_PF_BATCH_MIN", "2", 0);
        // W16: was the literal "12". The suffix wants the widest verify the
        // build actually has -- the engine clamps sfx_w to W_MAX anyway, so the
        // literal silently meant "W_MAX" on the w8 build and "12" everywhere
        // else. Naming W_MAX makes a wider build use its width by default.
        setenv("Q27_SUFFIX_W", std::to_string(W_MAX).c_str(), 0);
        setenv("Q27_PHASE_STATS", "1", 0);
        // Continuous batching (P1..P3) DEFAULTS ON since 2026-07-16 (BUILDLOG
        // "P3 T4: THE BAR PASSES" + "P3 LIVE CC A/B"): 2-slot aggregate 1.41x
        // fp8 / 1.40x turbo3 (bar 1.38x), live CC fused rounds -17..-19%
        // phv/round at matched depth, solo cost 0.00% -- a single-slot server
        // pays nothing (k=1 falls through to the proven solo path, byte-
        // identical). GRAPH_CAP=64: live CC traffic drew a 44+ graph-key
        // alphabet vs the LRU-32 default (86% hits, benign eviction churn);
        // 64 swallows the observed alphabet at ~460 MB worst case, and the
        // conductor's ctor headroom check SHRINKS-never-aborts, so tight
        // configs self-protect. Kill switches: Q27_BATCH=0, Q27_BATCH_GRAPH=0,
        // Q27_PROFILE=ref (ref skips this whole block, so it stays off there).
        setenv("Q27_BATCH", "1", 0);
        setenv("Q27_BATCH_GRAPH", "1", 0);
        setenv("Q27_BATCH_GRAPH_CAP", "64", 0);
    }
    // Q27_SAMPLED=0 (issue #1, small-VRAM greedy boots): the engine skips the
    // sampled graph set; this server refuses temperature>0 requests with a
    // 400 up front. Contradiction guard (two-tier precedent): forcing
    // sampling onto a boot that cannot sample is a config error -- refuse
    // loudly at boot, not per-request.
    const bool sampled_on = [] {
        const char* e = getenv("Q27_SAMPLED");
        return !e || atoi(e) != 0;
    }();
    if (!sampled_on) {
        const char* ft = getenv("Q27_FORCE_TEMP");
        if (ft && atof(ft) > 0.0) {
            fprintf(stderr,
                    "q27-server: FATAL -- Q27_SAMPLED=0 with Q27_FORCE_TEMP=%s: every "
                    "request would be forced onto the disabled sampled path\n",
                    ft);
            exit(1);
        }
        fprintf(stderr,
                "sampled graphs OFF (Q27_SAMPLED=0): temperature>0 requests get 400\n");
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

    // Per-engine non-KV reserve, the single source of truth shared by
    // auto-ctx (below) and the multi-slot skip loop. roles + graph zoo +
    // scratch, arch/width/gate scaled. Calibrated to the known-good 2x48K
    // fp8 anchor on a 32GB 5090 (~5.8 GB/slot on W12). See the auto-ctx
    // comment for the term-by-term rationale.
    const double kEngGw8 = Q27_W_MAX < 8 ? Q27_W_MAX : 8;
    const double kEngGwx = cc_arch >= 89 ? 0.13e9 : 0.43e9;
    const double kEngGraphs = kEngGw8 * 0.13e9 + (Q27_W_MAX - kEngGw8) * kEngGwx;
    const double kEngMonoSave = cc_arch >= 89 ? 0.08e9 : 0.15e9;
    const double kEngSampSave = cc_arch >= 89 ? 0.34e9 : 0.60e9;
    const double kEngBase = cc_arch >= 120 ? 0.89e9 : cc_arch >= 89 ? 2.13e9 : 1.77e9;
    // per-slot non-KV = single-engine stack + co-residency scratch (the
    // multi-slot `per_slot` in the auto-ctx block); the skip loop reserves
    // this + KV so it agrees with what auto-ctx sized for.
    const size_t ENG_FIXED_BYTES =
        (size_t)(kEngBase + kEngGraphs + (Q27_W_MAX + 1) * 0.157e9 + 2.2e9 -
                 (constrain_tools ? 0.0 : kEngMonoSave) - (sampled_on ? 0.0 : kEngSampSave));

    // --ctx auto: sizing moved to AFTER the weight upload (2026-07-17), and
    // multi-slot-aware since 2026-07-18: each borrowing engine carries its
    // own role sets + graph zoo + KV, so N slots divide the budget. Sized in
    // the post-upload block below; no legacy 8192 fallback.
    fprintf(stderr, "loading tokenizer...\n");
    q27::Tokenizer tok(tokpath);
    const std::vector<int> think_close_ids = tok.encode("</think>\n\n");
    fprintf(stderr, "loading model...\n");
    // P10-A1: weights owned here and shared into the Engine(s) by reference.
    // Upload once; borrowing engines skip the 17.7 GB weight copy. (Multi-slot
    // will construct N engines from this same pair.)
    q27::Model shared_model = q27::Model::open(model);
    validate_arch(shared_model); // before the upload, not in Engine::init after it
    q27::DeviceModel shared_dm(shared_model);
    fprintf(stderr, "uploading weights...\n");
    shared_dm.upload_all();
    shared_dm.checksum_baseline();
    fprintf(stderr, "resident: %.2f GB (checksummed)\n", shared_dm.bytes_resident() / 1e9);

    // --ctx auto (single-slot): size the KV budget to MEASURED free VRAM
    // with the weights already resident, so tier size (q4s/v1.4/q6),
    // upload alignment overhead, and any co-tenant process fall out of the
    // measurement instead of a model. What remains estimated is the non-KV
    // stack allocated after this point (GDN role sets + spec/sample graph
    // zoo + workspaces): each width adds one role set (~157MB) and ~one
    // perm's worth of graphs (~130MB), plus an arch base -- the sm_86
    // fd2/h16 path carries heavier workspaces than sm_120, and the old
    // 1.27GB anchor predates the P1-P3 graph growth (calibrated 2026-07-17
    // from in-process free deltas on both cards; see BUILDLOG).
    // per-token KV bytes are EXACT: 18 K/V pairs (17 attn + 1 MTP),
    // per-buffer/token = N_KV*HEAD_DIM*esz (fp16 2048 B, fp8 1024 B) or
    // N_KV*(HEAD_DIM/128)*50 B = 400 B turbo3 / *82 B = 656 B turbo5 K.
    // MIRROR WARNING: matches Engine::kv_bytes -- update together.
    {
        // clamp slot count BEFORE auto-ctx divides the budget by it. Ceiling
        // 8 = the conductor's hard MAX_K/2 fusion limit (W_PLUMB=16 lane
        // slots, floor-2 trim); slots past what VRAM fits are skipped in the
        // build loop below. Raised from 4 (2026-07-18): the 96GB PRO 6000
        // fit 4x262144 with 28.9 GB idle -- 4 was a VRAM guess, 8 is the
        // real plumbing ceiling.
        n_slots = std::max(1, std::min(8, n_slots));
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        fprintf(stderr, "vram: free %.2f GB post-weights\n", free_b / 1e9);
        if (ctx < 0) {
            const char* kvv = getenv("Q27_KV");
            const bool fp8 = kvv && !strcmp(kvv, "fp8");
            const bool t3 = kvv && !strcmp(kvv, "turbo3");
            const bool t3v = kvv && !strcmp(kvv, "turbo3v");
            const bool t5k = kvv && !strcmp(kvv, "turbo5k");
            // K+V bytes per token per layer-pair. turbo3 400+400, turbo3v
            // 2048(fp16 K)+400, turbo5k 656(82 B x 8 blocks)+400.
            const double pair =
                t3 ? 800.0 : t3v ? 2448.0 : t5k ? 1056.0 : fp8 ? 2048.0 : 4096.0;
            const double per_tok = 18.0 * pair;
            // calibrated 2026-07-17 (in-process free deltas, q4s tier):
            // sm_120 W12 fp8@131072: non-KV stack 4.49GB => base 0.89;
            // sm_86 w8 fp16@61440: 4.22GB => base 1.77 (turbo3 measured
            // +0.08 -- inside slack). Retro-check: predicts the hand-found
            // v1.4 3090 ceiling (24576) and the q4s 61440 knife-edge fit.
            // sm_89 calibrated 2026-07-18 (RunPod 4090 field test, two boots
            // agreeing within 75 MB): Ada's fixed stack runs ~1 GB fatter
            // than sm_120's -- the pre-calibration pick survived with 40 MB
            // to spare. >8-width graph slope on sm_89 is UNMEASURED; it
            // deliberately shares sm_86's fat slope below (under-pick beats
            // a dead boot).
            const double base = cc_arch >= 120 ? 0.89e9 : cc_arch >= 89 ? 2.13e9 : 1.77e9;
            // SINGLE-slot non-KV stack (base + roles + graph zoo - capture
            // saves), unchanged and calibrated: this is what N==1 uses, and
            // it must stay exact (drives the 262144/57344/... single-slot
            // picks). Reuses the hoisted width/arch-scaled terms.
            const double single_fixed = base + kEngGraphs + (Q27_W_MAX + 1) * 0.157e9 -
                                        (constrain_tools ? 0.0 : kEngMonoSave) -
                                        (sampled_on ? 0.0 : kEngSampSave);
            long budget, c;
            if (n_slots <= 1) {
                // sm_86/89 carry a fatter, ctx-scaled graph zoo + a larger
                // cudaGraphInstantiate transient than the fixed calibration
                // constant captures; the old 0.25 GB margin got eaten during
                // graph build and OOMed on 24GB cards that size below the 262144
                // cap (issue #6, NHClimber87: sized 184320 -> OOM). Give
                // Ampere/Ada ~1 GB of real headroom; sm_120 (32GB, cap usually
                // binds) keeps the tight margin.
                const double slack = cc_arch >= 120 ? 0.25e9 : 1.0e9;
                budget = (long)((double)free_b - single_fixed - slack);
                c = budget > 0 ? (long)(budget / per_tok) : 0;
            } else {
                // MULTI-slot: every slot is a full co-resident engine.
                // PER_SLOT = single_fixed + ~2.2 GB fragmentation/scratch that
                // co-residency adds (calibrated to the shipped 2x48K fp8
                // anchor on a 32GB 5090 == ~6.6 GB/slot on W12). Reduce the
                // slot count first so the picked ctx and the log are HONEST
                // (the build-loop skip is then a pure safety net), then split.
                const double per_slot = single_fixed + 2.2e9;
                int fit = (int)(free_b / (per_slot + per_tok * 4096.0));
                if (fit < 1) fit = 1;
                if (fit < n_slots) {
                    fprintf(stderr,
                            "--ctx auto: only %d of %d requested slots fit in %.1f GB "
                            "(~%.1f GB/slot fixed) -- sizing for %d\n",
                            fit, n_slots, free_b / 1e9, per_slot / 1e9, fit);
                    n_slots = fit;
                }
                const double slack = (cc_arch >= 120 ? 0.25e9 : 1.0e9) * n_slots; // arch margin (issue #6), per slot
                budget = (long)((double)free_b - n_slots * per_slot - slack);
                c = budget > 0 ? (long)(budget / (per_tok * n_slots)) : 0;
            }
            // cap: native window (262144) for the compact KV formats
            // (2026-07-11, Gabe sign-off: fp8 measured to 294912 on the
            // 5090, turbo3 to 655360 with needle 6/6 @361K -- the cap is a
            // TTFT/estimate-margin guard, not a VRAM fact); fp16 keeps the
            // historical 131072 (it barely clears it anyway).
            const long cap = (fp8 || t3 || t3v || t5k) ? 262144 : 131072;
            if (c > cap) c = cap;
            ctx = (int)(c / 4096 * 4096);
            if (ctx < 4096) {
                fprintf(stderr,
                        "--ctx auto: only %d fits (free %.1fGB post-weights, %s KV, W_MAX=%d) -- "
                        "likely to OOM; pass a smaller --ctx or rebuild with a lower Q27_W_MAX\n",
                        ctx, free_b / 1e9, t3 ? "turbo3" : t3v ? "turbo3v" : t5k ? "turbo5k" : fp8 ? "fp8" : "fp16",
                        Q27_W_MAX);
                if (ctx < 2048) ctx = 2048; // give the ctor a floor to fail loudly at
            } else {
                fprintf(stderr,
                        "--ctx auto: %d%s (free %.1fGB post-weights, %s KV, W_MAX=%d)\n", ctx,
                        n_slots > 1 ? " per slot" : "", free_b / 1e9,
                        t3 ? "turbo3" : t3v ? "turbo3v" : t5k ? "turbo5k" : fp8 ? "fp8" : "fp16", Q27_W_MAX);
                if (ctx < 16384)
                    fprintf(stderr, "  (tight -- a lower Q27_W_MAX build would free more)\n");
            }
            if (n_slots > 1) slot1_ctx = ctx; // auto: every slot gets the same window
        }
    }
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
    // P16 persistent prefix cache. Declared BEFORE `slots` so it outlives the
    // engines: an engine's destructor joins its writer thread, and that thread
    // calls into this object. Off unless --prefix-cache names a directory.
    q27::PrefixCache pfx_cache;
    q27::PrefixRam pfx_ram;
    if (!pfx_cfg.root.empty()) {
        const char* kve = getenv("Q27_KV");
        // MIRROR WARNING: this must agree with Engine::init's Q27_KV parse --
        // it only feeds the prefix-cache compat hash, so a kind missing here
        // would alias two formats onto one cache key.
        const int kvk = kve && !strcmp(kve, "fp8")       ? KV_FP8
                        : kve && !strcmp(kve, "turbo3")  ? KV_T3
                        : kve && !strcmp(kve, "turbo3v") ? KV_T3V
                        : kve && !strcmp(kve, "turbo5k") ? KV_T5K
                                                         : KV_F16;
        size_t model_bytes = 0;
        { struct stat st; if (::stat(model.c_str(), &st) == 0) model_bytes = (size_t)st.st_size; }
        const uint64_t compat =
            q27::pfx_compat_hash(model, model_bytes, N_LAYER, N_KV, HEAD_DIM, GDN_HEADS, GDN_DIM,
                                 GDN_CH, kvk, q27::PFX_VERSION);
        if (pfx_cache.init(pfx_cfg, compat))
            fprintf(stderr,
                    "prefix-cache: %s (%zu entr%s indexed, %.2f/%.2f GB, min %d, max %d, step %d)\n",
                    pfx_cfg.root.c_str(), pfx_cache.size(), pfx_cache.size() == 1 ? "y" : "ies",
                    pfx_cache.bytes() / 1e9, pfx_cfg.max_bytes / 1e9, pfx_cfg.min_tokens,
                    pfx_cfg.max_tokens, pfx_cfg.step_tokens);
    }
    // n_slots already clamped to [1,8] above (before auto-ctx divided the
    // budget by it); the conductor's MAX_K/2 fusion ceiling is 8.
    n_slots = std::max(1, std::min(8, n_slots));
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
            // Per-engine non-KV reserve = ENG_FIXED_BYTES (the same constant
            // auto-ctx sizes with, so the two estimators agree) + KV + margin.
            // The old need[] used a hardcoded ~1.2 GB that omitted the
            // graph zoo, so a slot could pass here and then OOM building its
            // own zoo (2026-07-18: aligned to ENG_FIXED_BYTES).
            size_t need = ENG_FIXED_BYTES + kvb + (kvb >> 3) + (256ull << 20);
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
        if (pfx_cache.enabled()) {
            s.eng->pcache = &pfx_cache;  // P16 disk tier
            s.eng->pram = &pfx_ram;      // P16c host-RAM tier (no-op when 0 slots)
        }
        // graph-zoo capture gate (issue #1): without --constrain-tools the
        // P11 monolithic draft/verify graphs are unreachable -- skip capture.
        s.eng->capture_constrained = constrain_tools;
        s.eng->build_graph();
        s.eng->build_spec_graphs();
        slots.push_back(std::move(s));
        fprintf(stderr, "slot %d ready: ctx=%d\n", si, sctx);
    }
    // P16c sizing needs an engine: pfx_bytes() depends on the KV format and
    // the layer geometry. Slots are built by now, so ask slot 0.
    if (pfx_cache.enabled() && pfx_ram_gb > 0 && !slots.empty())
        pfx_ram.init((size_t)(pfx_ram_gb * 1e9), slots[0].eng->pfx_bytes(pfx_cfg.max_tokens));
    // Pin every prefix-cache buffer HERE, before listen(): each one is sized by
    // --prefix-cache-max-tokens, which is known now, and a lazy cudaMallocHost
    // costs ~190 ms (staging) or ~135 ms (RAM slot) on whichever request first
    // needs it -- the post-restart request this feature exists to make fast.
    //
    // Synchronous on purpose. Boot has no cover left to hide it under (the
    // weight upload finished long before the slots were built), and a
    // background thread would race the graph zoo: engine.cuh captures with
    // cudaStreamCaptureModeGlobal, under which an allocation from ANY thread
    // is an illegal call, and P3 re-captures on new shapes during serving too.
    if (pfx_cache.enabled()) {
        pfx_ram.prealloc();
        for (auto& s : slots) s.eng->pfx_prealloc(pfx_cfg.max_tokens);
    }
    {
        // headroom line (also the auto-ctx calibration probe: this minus the
        // post-weights line minus exact KV = the non-KV fixed stack)
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        fprintf(stderr, "vram: free %.2f GB at ready\n", free_b / 1e9);
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
                if (!p.is_object()) continue; // bare string in a content array
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
        char p13buf[96], gatebuf[512], phbuf[352], sfxbuf[48], pfxbuf[32];
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
        if (getenv("Q27_NJOINT")) {
            fprintf(stderr, "[njoint] rid=%ld", rt.rid);
            for (int c = 0; c < Q27_W_MAX; c++)
                for (int n = 1; n <= Q27_W_MAX; n++)
                    if (e.gate_joint[c][n]) fprintf(stderr, " %d:%d=%ld", c, n, e.gate_joint[c][n]);
            fprintf(stderr, "\n");
        }
        fprintf(stderr,
                "[req] rid=%ld api=%s conv=%08llx qw_ms=%.0f tok_ms=%.0f prompt=%d hit=%d "
                "ckpt=%d pf=%d pf_ms=%.0f dec=%d dec_ms=%.0f cb_ms=%.0f rounds=%d tps=%.1f "
                "end=%s gw=%.0f yields=%d slot=%d t=%.0f%s%s%s%s%s%s\n",
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
                // P16: tokens restored from the disk prefix cache. Emitted only on
                // an actual disk hit, so the line stays byte-identical when the
                // feature is off (same rule as the md/gch/ph optionals above).
                g.pfx > 0 ? (snprintf(pfxbuf, sizeof pfxbuf, " pfx=%d", g.pfx), pfxbuf) : "",
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
    auto claim_slot = [&](const std::vector<int>& prompt, int requested,
                          bool thinking, const q27::ThinkCfg& cfg,
                          bool budget_aware) -> Slot& {
        std::unique_lock<std::mutex> lk(route_m);
        const q27::ThinkCfg raw_cfg{false, -1, true};
        const q27::ThinkCfg& limit_cfg = budget_aware ? cfg : raw_cfg;
        const int limit_close = budget_aware ? (int)think_close_ids.size() : 0;
        const int limit_flag = budget_aware ? think_budget_flag : 0;
        // Eligibility includes the request's forced-close reserve. This keeps
        // prefix reuse from routing a boundary request to a smaller slot that
        // cannot honor the bounded-thinking transition.
        auto can_fit = [&](Slot& s) {
            return q27::resolve_think_decode_limits(
                       requested, s.eng->max_ctx, (int)prompt.size(),
                       s.eng->ctx_round_reserve(), limit_close, thinking,
                       limit_cfg, limit_flag)
                .context_ok;
        };
        bool fits_any = false;
        for (auto& s : slots) fits_any = fits_any || can_fit(s);
        for (;;) {
            Slot* best = nullptr;
            int best_tier = -1, best_key = 0;
            for (auto& s : slots) {
                if (s.busy) continue;
                if (!fits_any) { // doomed prompt: largest free slot refuses it
                    if (!best || s.eng->max_ctx > best->eng->max_ctx) best = &s;
                    continue;
                }
                if (!can_fit(s)) continue;
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

    // ------------------------------------------------------------------
    // P1 continuous batching (Q27_BATCH; SERVING DEFAULT ON since 2026-07-16
    // via the CC profile block above -- Q27_BATCH=0 or Q27_PROFILE=ref turns
    // it off; the CLI binary never sets it). One Conductor owns
    // every decode round; request threads keep slot claim + tokenize +
    // prefill (under their own scoped gate lease) and drain a per-request
    // TokenQueue into the existing consumer lambdas. Q27_BATCH unset/0:
    // `conductor` stays null and every call site runs the pre-batch path
    // byte-for-byte (the batch branches below are additive). Batch mode
    // requires the gated dexit serving config (Q27_PMIN>0 + dexit, the CC
    // profile defaults above): the fused round's draft_and_gate() asserts
    // it. Constrained-CAPPED members (h_mask_id0 >= 0, --constrain-tools
    // engaged without Q27_TOOL_SPLIT) take the gated round SHAPE under
    // fusion where solo runs the full-width monolithic graph -- the mask +
    // accept-cap still apply in the per-engine tail, a documented P1
    // divergence class, not a gate target.
    // Lifecycle (review pass 2, VERIFIED against the vendored httplib
    // 0.18.3): conductor.reset() at main's end runs only after srv.listen()
    // returns, and listen_internal() calls task_queue->shutdown()
    // (third_party/httplib.h:6821), which joins EVERY ThreadPool worker --
    // in-flight handler threads included -- before returning (t.join(),
    // httplib.h:789); svr.stop() itself only closes the listen socket
    // (:6337-6346). So no request thread can reach register_member() after
    // the reset, and the conductor's M4 refusal stays an embedder guard,
    // not a server path.
    std::unique_ptr<q27::Conductor> conductor;
    {
        const char* e = getenv("Q27_BATCH");
        if (e && atoi(e) != 0) {
            // Batch-mode config validation (review M2), mirroring the
            // gemm_min guardrail's refuse-to-run posture (engine.cuh "THE
            // GUARDRAIL"): the conductor's draft_and_gate() asserts the
            // gated-dexit greedy-spec config PER ROUND -- i.e. deep into
            // serving, on a live request. Check the env strings HERE, before
            // the Conductor exists, with logic identical to the engine's own
            // parses (build_spec_graphs / make_decode_task /
            // set_tool_constraint -- the engine reads them later, so parse
            // the raw strings the same way it will).
            //
            // TWO-TIER SEMANTICS (2026-07-16 defaults flip): Q27_BATCH=1 can
            // now come from the CC profile default above, not only the user's
            // env. batch_env_user (captured BEFORE the profile setenv block)
            // picks the tier on an incompatible config:
            //   USER-EXPLICIT  -> FATAL exit(1), unchanged pre-flip behavior:
            //     the user asked for a combination that cannot run; refuse
            //     loudly rather than silently serve something else.
            //   PROFILE-DEFAULT -> one-line auto-disable notice + skip
            //     conductor construction; the server runs exactly as pre-P1.
            //     A DEFAULT must never turn a formerly-working invocation
            //     (e.g. a Q27_DEXIT=0 tuning run) into a dead server.
            // Q27_PROFILE=ref never reaches either tier: ref skips the whole
            // profile setenv block, so Q27_BATCH stays unset and batching is
            // simply OFF (ref = conservative reference, not a refusal).
            const char* why = nullptr; // non-null = config cannot batch
            char whybuf[96];
            const char* pm = getenv("Q27_PMIN");  // engine: atof; gate iff > 0
            const char* de = getenv("Q27_DEXIT"); // engine: atoi != 0, default on
            if (!pm || atof(pm) <= 0) {
                if (batch_env_user) {
                    fprintf(stderr, "q27-server: FATAL -- Q27_BATCH=1 requires the gated draft "
                                    "(Q27_PMIN > 0; it is %s). Set Q27_PMIN=0.5 or drop Q27_BATCH.\n",
                            pm ? pm : "unset");
                    exit(1);
                }
                snprintf(whybuf, sizeof whybuf, "Q27_PMIN=%s, batching needs the gated draft",
                         pm ? pm : "unset");
                why = whybuf;
            } else if (de && atoi(de) == 0) {
                if (batch_env_user) {
                    fprintf(stderr, "q27-server: FATAL -- Q27_BATCH=1 requires the dexit draft "
                                    "loop (Q27_DEXIT=%s disables it). Drop Q27_DEXIT or Q27_BATCH.\n",
                            de);
                    exit(1);
                }
                snprintf(whybuf, sizeof whybuf,
                         "Q27_DEXIT=%s, batching needs the dexit draft loop", de);
                why = whybuf;
            } else if (getenv("Q27_SAMPLE_PLAIN")) { // engine: presence-only
                if (batch_env_user) {
                    fprintf(stderr, "q27-server: FATAL -- Q27_SAMPLE_PLAIN (any value) forces the "
                                    "plain sampler, which has no fused path. Drop it or Q27_BATCH.\n");
                    exit(1);
                }
                why = "Q27_SAMPLE_PLAIN set, plain sampler has no fused path";
            } else if (getenv("Q27_TOOL_SPLIT")) { // engine: presence-only
                if (batch_env_user) {
                    fprintf(stderr, "q27-server: FATAL -- Q27_TOOL_SPLIT (any value) enables the "
                                    "split constrained rounds, which have no fused path. Drop it "
                                    "or Q27_BATCH.\n");
                    exit(1);
                }
                why = "Q27_TOOL_SPLIT set, split constrained rounds have no fused path";
            }
            if (why) {
                // profile-default tier: notice + fall through with conductor
                // null -- every call site runs the pre-batch path, exactly
                // the pre-P1 server.
                fprintf(stderr, "continuous batching: OFF (auto-disabled: %s)\n", why);
            } else {
                conductor = std::make_unique<q27::Conductor>(gpu_gate);
                fprintf(stderr, "continuous batching: ON (%s, union cap %d)\n",
                        batch_env_user ? "Q27_BATCH=1 explicit env"
                                       : "serving default since 2026-07-16",
                        W_MAX);
            }
        } else if (e) {
            fprintf(stderr, "continuous batching: OFF (Q27_BATCH=%s)\n", e);
        } else {
            // only reachable when the profile block didn't run (ref) or a
            // future path unsets it: report which.
            fprintf(stderr, "continuous batching: OFF (%s)\n",
                    ref_profile ? "Q27_PROFILE=ref" : "Q27_BATCH unset");
        }
    }
    // Batch-mode generation driver shared by every generate() call site.
    // Wiring contract (plan Task 10 + addenda A3/A7):
    //  - PREFILL runs on THIS request thread under the scoped lease below,
    //    with the caller's round_gap yield hook still installed -- cold
    //    prefills time-slice against conductor decode rounds at chunk
    //    granularity exactly as before;
    //  - the lease dies at the inner scope's end: from there the CONDUCTOR
    //    owns decode GPU arbitration (A7: a request thread must never hold
    //    the gate while blocked on a TokenQueue, or decode deadlocks);
    //  - on_round_gap is cleared BEFORE registration (Task 9 invariant: the
    //    conductor's per-round lease release IS the yield; a member yielding
    //    the conductor's own lease from inside post_round would break the
    //    one-Lease-per-round structure);
    //  - on_emit (nullable) runs on the CONDUCTOR thread inside post_round,
    //    between the on_round scan and on_pending -- the /v1/messages paths
    //    route tc.on_id there, so grammar feeding keeps its exact solo
    //    ordering AND the P7 mask-pool invariant survives: every mask-pool
    //    mutation site (tc.apply/on_drafts/on_pending -> mask_pool_add,
    //    set_tool_constraint/set_tool_masks5) still runs while the GPU gate
    //    is held -- now by the conductor's round lease (the "mutated only
    //    from generation callbacks, which run while holding the GPU gate"
    //    invariant at the tool_mask_cache declaration above);
    //  - THIS thread drains the queue into on_token (the unchanged consumer
    //    bodies). on_token returning false (client disconnect) sets t.cancel
    //    (A3) and the drain continues until the queue closes, so the member
    //    always finishes conductor-side teardown before this frame (tc, SSE
    //    sinks, hooks) unwinds. tc.end()'s constraint clear then runs on an
    //    engine the conductor has already left -- gate-less, but it is
    //    exactly the microsecond async-copy class GpuGate::Lease documents
    //    as exempt, and it is stream-ordered ahead of the slot's next work.
    // Returns t.emitted -- solo generate()'s return value for every natural
    // finish (on cancel it counts tokens delivered before the cut).
    // err_out (nullable): receives the queue's error slot when the member
    // FAILED (A2 host-exception unwind or the M4 registration refusal)
    // instead of finishing -- the caller's honest-surfacing hook (500 when
    // nothing was emitted / Anthropic SSE error event); gs.end="error" is
    // stamped either way so the [req] line never reports a failed
    // generation as a normal finish.
    auto batch_generate = [&](Engine& eng, const std::vector<int>& prompt, int nm,
                              std::function<bool(int, bool)> on_token,
                              std::function<void(int)> on_emit, int stable_len, double& qw,
                              const ReqTrace& rt, Engine::DecodeTask& t,
                              std::string* err_out) -> int {
        q27::TokenQueue q;
        {
            q27::GpuGate::Lease lk(gpu_gate);
            qw = ms_since(rt.t0);
            int P = 0;
            if (!eng.generate_prefill(prompt, stable_len, &P)) {
                eng.on_round_gap = nullptr;
                return 0; // refused; gs.end already stamped for req_log
            }
            eng.on_round_gap = nullptr; // Task 9 invariant (contract above)
            // register_member replaces this placeholder with the queue sink.
            auto placeholder = [](int) { return true; };
            eng.make_decode_task(t, nm, EOS, placeholder, P);
        } // lease released: decode arbitration belongs to the conductor
        conductor->register_member(&eng, &t, &q, std::move(on_emit));
        bool client_gone = false;
        std::vector<q27::TokenQueue::Token> ids;
        try {
            while (q.pop(ids)) {
                for (const auto& token : ids) {
                    if (!client_gone && !on_token(token.id, token.forced)) {
                        client_gone = true;
                        t.cancel.store(true); // A3: takes effect at a round boundary
                    }
                }
                ids.clear();
            }
        } catch (...) {
            // A2 mirror: never unwind past the drain while the conductor
            // still owns the member (its hooks reference this frame).
            // Cancel, drain to close, then let HookGuard & co. run.
            t.cancel.store(true);
            ids.clear();
            while (q.pop(ids)) ids.clear();
            throw;
        }
        // Error surfacing (review pass 2): a non-null error slot means the
        // A2 unwind failed the queue (host exception in this member's
        // bookkeeping) or registration was refused (M4) -- no normal finish
        // reason exists. Stamp gs.end for the [req] line (the A2 path
        // already stamped it conductor-side via finish_decode; the refusal
        // path never reached a finish), log the what() -- the queue copy is
        // the ONLY place it survives -- and hand it to the caller. Safe to
        // touch eng.gs here: the queue observed closed, the conductor's
        // fail() was its last access to this request's state (close-edge
        // rule, conductor.h fail_member).
        if (const char* err = q.error_or_null()) {
            eng.gs.end = "error";
            fprintf(stderr, "[req-error] rid=%ld %s\n", rt.rid, err);
            if (err_out) *err_out = err;
        }
        return t.emitted;
    };
    // Batch-mode [req] telemetry: mean members-per-round across this
    // request's decode rounds + how many of its rounds ran fused (k >= 2),
    // from the conductor-filled DecodeTask counters. Appended LAST (after
    // the sfx/ph/tg optionals; reqlog parsers stop at end=) and empty when
    // Q27_BATCH is off (Q27_BATCH=0 / ref profile / auto-disabled), where
    // the [req] line stays byte-identical to pre-P1. Since the 2026-07-16
    // defaults flip the default CC-profile line DOES carry bat=.
    auto bat_stats = [&conductor](const Engine::DecodeTask& t) -> std::string {
        if (!conductor) return std::string();
        char b[48];
        snprintf(b, sizeof b, " bat=%.1f,%ld",
                 t.rounds > 0 ? (double)t.bat_members / t.rounds : 0.0, t.bat_r2);
        return std::string(b);
    };

    httplib::Server srv;
    srv.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        fprintf(stderr, "[http] %s %s -> %d\n", req.method.c_str(), req.path.c_str(),
                res.status);
    });

    // Opt-in API key auth (no-op, zero overhead beyond one empty-vector
    // check per request, when api_keys is empty -- the loopback-only
    // default is unchanged). /health is intentionally exempt: infra health
    // checks (load balancers, container orchestrators) need it reachable
    // without distributing the secret to that infrastructure. Every other
    // endpoint requires a valid key. Runs before route dispatch, so an
    // invalid/missing key never reaches slot allocation, tokenization, or
    // any generation work.
    if (!api_keys.empty()) {
        srv.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
            if (req.path == "/health") return httplib::Server::HandlerResponse::Unhandled;
            std::string provided = q27::extract_api_key(req.get_header_value("Authorization"),
                                                         req.get_header_value("x-api-key"));
            if (q27::api_key_valid(provided, api_keys))
                return httplib::Server::HandlerResponse::Unhandled;
            // Anthropic-shaped error for the Anthropic-shaped endpoint
            // family (Claude Code's SDK reads error.message off this exact
            // shape -- see anthropic_error_json's own comment); OpenAI-
            // shaped for everything else, matching the 400-error split
            // already used elsewhere in this file.
            bool anthropic_shape = req.path.rfind("/v1/messages", 0) == 0;
            res.status = 401;
            if (!anthropic_shape) res.set_header("WWW-Authenticate", "Bearer");
            res.set_content(q27::auth_error_json(anthropic_shape), "application/json");
            return httplib::Server::HandlerResponse::Handled;
        });
        fprintf(stderr, "API key authentication enabled (%zu key%s configured)\n",
                api_keys.size(), api_keys.size() == 1 ? "" : "s");
    }

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

    // ---------------- OpenAI chat/completions ----------------
    // Tool calling: only /v1/chat/completions with a "messages" body gets the
    // think/tool-aware path below (routed_chat) -- /v1/completions and any
    // chat body that falls back to a raw "prompt" keep the ORIGINAL text-only
    // behavior byte-for-byte (build_prompt, unchanged). This mirrors the
    // /v1/messages tool pipeline exactly (chatml_prompt + tools_preamble +
    // ToolConstrainer + StreamSplitter + parse_tool_call/parse_bare_tool_calls);
    // only the request/response bridging (openai_msgs/openai_tools_json in
    // api_common.h) and the output envelope (OpenAI tool_calls[] shape,
    // finish_reason="tool_calls") are new.

    auto build_prompt = [&](const json& body) -> std::vector<int> {
        if (body.contains("messages")) {
            std::vector<std::pair<std::string, std::string>> msgs;
            for (auto& m : body["messages"]) {
                // is_object() BEFORE any value() call: value() on a non-object
                // element throws 306 -> httplib 500 (same ordering fix as
                // anthropic_msgs).
                if (!m.is_object()) continue;
                std::string role = m.value("role", "user");
                std::string content;
                // const operator[] on a missing key aborts (json.hpp assertion) --
                // a content-less message must not kill the server (Security #1;
                // mirrors the Anthropic-path guard in api_common.h).
                if (m.contains("content")) {
                    if (m["content"].is_string()) content = m["content"];
                    else if (m["content"].is_array())
                        for (auto& part : m["content"])
                            if (part.is_object() && part.value("type", "") == "text")
                                content += part.value("text", "");
                }
                msgs.push_back({role, content});
            }
            // Per-request thinking: the server profile is the default, an
            // explicit enable_thinking / chat_template_kwargs / Anthropic
            // thinking field overrides it either way (resolve_think in
            // api_common.h). Silent request on a no-think server -> no-think.
            bool think = q27::resolve_think(body, !no_think_srv, req_think);
            return tok.apply_chat_template(msgs, think);
        }
        return tok.encode(q27::jstr(body, "prompt"));
    };

    auto handle = [&](const httplib::Request& req, httplib::Response& res, bool chat) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return; }
        // Default when the client omits max_tokens: a generous floor, unified
        // across all three API shapes. Clamped to the context window below, so a
        // big default can't over-reserve; 8192 covers a real answer plus a short
        // think trace. A long *thinking* request should still set max_tokens
        // explicitly (a grad-level trace wants 16K+). jint/jbool: a
        // present-but-null field reads as absent (see parse_sample).
        int n_max = (int)q27::jint(body, "max_tokens", 8192);
        bool stream = q27::jbool(body, "stream", false);
        // stream_options.include_usage (OpenAI streaming spec, both API
        // shapes): when true, one extra SSE chunk -- empty choices + the
        // usage totals -- goes out after the finish_reason chunk, before
        // [DONE]. Tolerant parse: a non-object stream_options or non-bool
        // include_usage reads as false (a malformed option must not throw
        // out of the handler). Absent/false -> zero framing change.
        bool inc_usage = false;
        if (stream && body.contains("stream_options") && body["stream_options"].is_object()) {
            const auto& so = body["stream_options"];
            inc_usage = so.contains("include_usage") && so["include_usage"].is_boolean() &&
                        so["include_usage"].get<bool>();
        }
        // Tool-calling admission: gate strictly on chat==true AND an actual
        // "messages" array, so /v1/completions and the raw-"prompt" chat
        // fallback are provably untouched by anything below (they never
        // enter any of the branches this flag guards).
        const bool routed_chat = chat && body.contains("messages") && body["messages"].is_array();
        json tools = json::array();
        q27::ToolChoice tchoice;
        std::vector<std::string> tool_names_v;
        if (routed_chat) {
            try {
                tchoice = q27::parse_tool_choice(body);
                q27::apply_openai_parallel_tool_calls(body,tchoice);
                q27::OpenAIToolSelection selected=q27::select_openai_tools(body,tchoice);
                tools=std::move(selected.tools);
                tool_names_v=std::move(selected.names);
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error",{{"message",e.what()},
                                                 {"type","invalid_request_error"}}}}.dump(),
                                "application/json");
                return;
            }
        }
        const std::set<std::string> allowed_tool_names(tool_names_v.begin(),tool_names_v.end());
        long rid = req_counter++;
        auto tk0 = std::chrono::steady_clock::now();
        std::vector<int> prompt;
        int stable_len = -1; // -1 = legacy tail snapshot (build_prompt's fallback path)
        int sys_len = 0;     // P16b: system-block tokens (0 = none/feature off)
        bool thinking = false; // request wants a real <think> block; seeds the splitter below
        q27::ThinkCfg tcfg;    // .budget resolved against the final n_max at each loop
        if (routed_chat) {
            tcfg = q27::resolve_think_cfg(body, !no_think_srv, req_think, -1);
            thinking = tcfg.enabled;
            // thinking and a FORCED tool call are mutually exclusive: FORCED
            // injects <tool_call>\n into the tail (below), leaving no room for a
            // think block, so suppress the opener when a tool is forced.
            if (tchoice.mode == q27::ToolChoice::FORCED) thinking = false;
            size_t stable_off = 0, sys_off = 0;
            std::string rendered =
                q27::chatml_prompt(q27::openai_msgs(body), tools, thinking, &stable_off, &sys_off);
            // P16b: token length of the system+tools block. Measured with a
            // THIRD encode used only for its length -- the prompt itself is
            // still built from the same two pieces, so no request's bytes
            // change when the cache is on.
            if (pfx_cache.enabled() && sys_off > 0)
                sys_len = (int)tok.encode(rendered.substr(0, sys_off)).size();
            // FORCED tool_choice: inject the opener into the volatile tail
            // (past stable_off, alongside the assistant-open/think-prefill --
            // P8 prefix-cache reuse is unaffected). The stream router below
            // is pre-seeded straight into the TOOL channel since the marker
            // itself never appears in the GENERATED text this way.
            if (tchoice.mode == q27::ToolChoice::FORCED) rendered += "<tool_call>\n";
            prompt = tok.encode(rendered.substr(0, stable_off));
            stable_len = (int)prompt.size();
            std::vector<int> tailv = tok.encode(rendered.substr(stable_off));
            prompt.insert(prompt.end(), tailv.begin(), tailv.end());
        } else {
            prompt = build_prompt(body);
        }
        ReqTrace rt{rid, chat ? "oai" : "cmpl", conv_fp(body),
                    std::chrono::steady_clock::now(), ms_since(tk0)};
        // Reject an empty prompt before slot selection: reuse_len() would run
        // ckpt_best() over an empty vector, and (pre-fix) a zero-token prompt
        // decodes from stale recurrent state and echoes the prior request's
        // pending token. An empty /v1/completions prompt is nonsensical anyway;
        // chat/messages always tokenize non-empty (template structure).
        if (prompt.empty()) {
            res.status = 400;
            res.set_content(json{{"error", {{"message", "empty prompt"},
                                            {"type", "invalid_request_error"},
                                            {"code", "empty_prompt"}}}}
                                .dump(),
                            "application/json");
            return;
        }
        const bool think_aware=routed_chat && tchoice.mode!=q27::ToolChoice::FORCED;
        // context-limit preflight BEFORE slot claim / SSE commit (review
        // follow-up 2026-07-09 #3): past this bound the routed slot's
        // n_max clamp floors at 0 -> empty 200
        const auto admission = think_aware
            ? q27::resolve_think_decode_limits(
                  n_max, max_slot_ctx, (int)prompt.size(), slots[0].eng->ctx_round_reserve(),
                  (int)think_close_ids.size(), thinking, tcfg, think_budget_flag)
            : q27::resolve_think_decode_limits(
                  n_max, max_slot_ctx, (int)prompt.size(), slots[0].eng->ctx_round_reserve(),
                  0, false, q27::ThinkCfg{false, -1, true}, 0);
        const int request_max_prompt = !think_aware || admission.context_ok
            ? max_prompt
            : q27::max_prompt_for_think_decode(
                  n_max, max_slot_ctx, slots[0].eng->ctx_round_reserve(), max_prompt,
                  (int)prompt.size(), (int)think_close_ids.size(), thinking, tcfg,
                  think_budget_flag);
        if ((int)prompt.size() > max_prompt || !admission.context_ok) {
            res.status = 400;
            res.set_content(json{{"error",
                                  {{"message", q27::ctx_limit_error_message(
                                                   (int)prompt.size(), request_max_prompt)},
                                   {"type", "invalid_request_error"},
                                   {"code", "context_length_exceeded"}}}}
                                .dump(),
                            "application/json");
            return;
        }
        if ((int)prompt.size() + n_max > max_slot_ctx)
            n_max = max_slot_ctx - (int)prompt.size();
        // Q27_SAMPLED=0 preflight: the sampled graphs were never captured.
        // (Q27_FORCE_TEMP>0 is a boot-time FATAL on such boots, so the
        // request-absent default here is genuinely greedy.)
        if (!sampled_on && q27::jnum(body, "temperature", 0.0) > 0.0) {
            res.status = 400;
            res.set_content(json{{"error",
                                  {{"message", "sampling disabled: server booted with "
                                               "Q27_SAMPLED=0 (greedy-only)"},
                                   {"type", "invalid_request_error"},
                                   {"code", "sampling_disabled"}}}}
                                .dump(),
                            "application/json");
            return;
        }
        long created = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();

        const char* obj = chat ? "chat.completion" : "text_completion";
        const char* objd = chat ? "chat.completion.chunk" : "text_completion";

        if (!stream) {
            Slot& sl = claim_slot(prompt,n_max,thinking,tcfg,think_aware); // may wait for a free engine
            auto sl_lease = slot_guard(sl);
            Engine& eng = *sl.eng;
            HookGuard hooks{eng}; // safe even when routed_chat is false: hooks
                                  // are never set on that path, so the clear
                                  // on scope-exit is a no-op (P15 M1 pattern)
            eng.samp = parse_sample(body);
            eng.pfx_sys_len = sys_len; // P16b (0 on paths with no system boundary)
            // Q27_BATCH: solo keeps the whole-call lease; batch mode scopes
            // its prefill lease inside batch_generate (A7) and re-stamps qw.
            std::optional<q27::GpuGate::Lease> lk;
            if (!conductor) lk.emplace(gpu_gate);
            double qw = ms_since(rt.t0);
            eng.on_round_gap = make_yield(eng);
            // re-clamp to the routed slot (rows P+1..P+gate_maxd+1 must stay
            // in ctx; reserve derived from the engine's active max depth)
            auto limits = think_aware
                ? q27::resolve_think_decode_limits(
                      n_max, eng.max_ctx, (int)prompt.size(), eng.ctx_round_reserve(),
                      (int)think_close_ids.size(), thinking, tcfg, think_budget_flag)
                : q27::resolve_think_decode_limits(
                      n_max, eng.max_ctx, (int)prompt.size(), eng.ctx_round_reserve(),
                      0, false, q27::ThinkCfg{false, -1, true}, 0);
            n_max = limits.n_max;
            const int think_budget = limits.budget;

            if (!routed_chat) {
                // ORIGINAL text-only behavior, byte-for-byte unchanged.
                std::string text;
                q27::Utf8Gate ugate;
                auto on_tok = [&](int id) {
                    text += ugate.feed(tok.decode_one(id));
                    return true;
                };
                Engine::DecodeTask bt;
                std::string berr;
                int n = conductor
                            ? batch_generate(eng, prompt, n_max,
                                             [&](int id, bool) { return on_tok(id); },
                                             nullptr, -1, qw, rt, bt, &berr)
                            : eng.generate(prompt, n_max, EOS, on_tok);
                eng.on_round_gap = nullptr;
                text += ugate.flush();
                req_log(rt, qw, eng, sl.id, bat_stats(bt));
                // batch error surfacing (review pass 2): nothing emitted = an
                // honest 500 in the OpenAI error envelope; if tokens WERE
                // produced, keep the 200 with the partial text -- end=error is
                // already in the [req] line either way.
                if (!berr.empty() && n == 0) {
                    res.status = 500;
                    res.set_content(json{{"error", {{"message", berr},
                                                    {"type", "api_error"}}}}
                                        .dump(),
                                    "application/json");
                    return;
                }
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

            // routed_chat: think/tool-aware path, an exact mechanical twin of
            // the /v1/messages non-stream handler above, OpenAI-shaped output.
            StreamSplitter sp;
            q27::ThinkBudgetState tb{think_budget};
            Engine::DecodeTask bt;
            q27::Utf8Gate ugate;
            std::vector<std::pair<StreamSplitter::Chan,std::string>> segments;
            auto route = [&](StreamSplitter::Chan ch, const std::string& t) {
                if (ch == StreamSplitter::TOOL &&
                    tchoice.mode == q27::ToolChoice::NONE)
                    ch = StreamSplitter::TEXT;
                if (t.empty()) {
                    if (ch != StreamSplitter::TOOL && !segments.empty() &&
                        segments.back().first == StreamSplitter::TOOL)
                        segments.emplace_back(ch, std::string());
                    return;
                }
                if (!segments.empty() && segments.back().first == ch)
                    segments.back().second += t;
                else segments.emplace_back(ch, t);
            };
            ToolConstrainer tc;
            tc.eng = &eng; tc.tok = &tok; tc.cache = &tool_mask_cache;
            tc.host2dev = &sl.tool_mask_host2dev;
            // FORCED requests are prompt-injected past the <tool_call> marker
            // (above) -- scan_round's engage trigger scans GENERATED text for
            // that marker and will never fire, so grammar masking is skipped
            // for those (documented limitation, api_common.h ToolChoice
            // comment); AUTO (the default) and NONE are unaffected.
            tc.enabled = constrain_tools && tchoice.mode != q27::ToolChoice::FORCED &&
                        eng.samp.inv_temp <= 0.f; // constrained+sampled is Phase 3
            tc.begin(tool_names_v);
            // FORCED: the opener was injected into the PROMPT, not generated,
            // so the splitter must start already inside the TOOL channel or
            // the call body would be read back as ordinary text.
            if (tchoice.mode == q27::ToolChoice::FORCED) sp.chan = StreamSplitter::TOOL;
            // thinking: the <think> opener was prompt-injected (not generated), so
            // start the splitter INSIDE the THINK channel -- the model's first
            // generated token is already inside the block, and its </think> flips
            // back to text (same mechanism as the FORCED TOOL pre-seed above).
            else if (thinking) sp.chan = StreamSplitter::THINK;
            ReasoningBudgetObserver budget{tok, tb, eng, bt, think_close_ids, sp.chan};
            budget.start();
            eng.on_pending = [&](int id) { tc.on_pending(id); };
            eng.on_drafts = [&](const int* dr) { tc.on_drafts(dr); };
            eng.on_round = [&](const int* em, int nr) {
                int bm = budget.preview_round(em, nr);
                int budget_kept = (bm >= 1 && bm < nr) ? bm : nr;
                int m = tc.enabled ? tc.scan_round(em, budget_kept) : -1;
                int kept = (m >= 1 && m <= budget_kept) ? m : budget_kept;
                budget.commit_round(em, kept);
                if (kept < nr) return kept;
                return m;
            };
            auto on_tok = [&](int id, bool forced) {
                for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) {
                    if (forced && ch == StreamSplitter::TEXT && q27::strip_ws2(t).empty())
                        continue;
                    route(ch, t);
                }
                return true;
            };
            std::string berr;
            int n = conductor
                        ? batch_generate(eng, prompt, n_max,
                                         [&](int id, bool forced) { return on_tok(id, forced); },
                                         [&](int id) { tc.on_id(id); }, stable_len, qw,
                                         rt, bt, &berr)
                        : eng.generate(prompt, n_max, EOS, [&](int id) {
                              tc.on_id(id);
                              return on_tok(id, bt.callback_forced);
                          }, stable_len, &bt);
            tc.end();
            eng.on_pending = nullptr;
            eng.on_drafts = nullptr;
            eng.on_round = nullptr;
            eng.on_round_gap = nullptr;
            req_log(rt, qw, eng, sl.id, tg_stats(tc) + bat_stats(bt));
            if (!berr.empty() && n == 0) {
                res.status = 500;
                res.set_content(json{{"error", {{"message", berr}, {"type", "api_error"}}}}
                                    .dump(),
                                "application/json");
                return;
            }
            for (auto& [ch, t] : sp.feed(ugate.flush())) route(ch, t);
            const bool final_tool_incomplete=q27::unfinished_tool_wrapper(
                n,n_max,bt.budget_truncated,sp.chan);
            for (auto& [ch, t] : sp.flush()) route(ch, t);
            std::string unclosed_tool=q27::take_unclosed_final_tool_segment(
                segments,final_tool_incomplete);
            auto ordered = q27::resolve_ordered_tool_segments(
                segments, tools.is_array() && !tools.empty() ? &tools : nullptr,
                n < n_max && !bt.budget_truncated,
                [&](const std::string& name, size_t accepted) {
                    return q27::tool_choice_allows_call(
                        tchoice, allowed_tool_names, name, accepted);
                });
            ordered.text+=unclosed_tool;
            std::string tx = ordered.text;
            std::vector<q27::ToolCall> eligible_calls = std::move(ordered.calls);
            if (ordered.recovered)
                fprintf(stderr,
                        "[tool-fallback] %zu bare call(s) recovered (oai-nonstream)\n",
                        ordered.recovered);
            const bool any_call = !eligible_calls.empty();
            if (q27::forced_tool_choice_missing_is_error(
                    tchoice, any_call, n >= n_max || bt.budget_truncated)) {
                res.status = 500;
                res.set_content(json{{"error",{{"message","model produced no eligible tool call for forced tool_choice"},
                                                 {"type","api_error"}}}}.dump(),
                                "application/json");
                return;
            }
            json msg = q27::openai_chat_message_json(
                tx, eligible_calls, rid, ordered.reasoning);
            json choice = {{"index", 0},
                          {"finish_reason", q27::openai_tool_finish_reason(
                              any_call,final_tool_incomplete,n>=n_max || bt.budget_truncated)},
                          {"message", msg}};
            json out = {{"id", "chatcmpl-q27-" + std::to_string(rid)}, {"object", obj},
                        {"created", created}, {"model", served_name},
                        {"choices", json::array({choice})},
                        {"usage", {{"prompt_tokens", (int)prompt.size()},
                                   {"completion_tokens", n},
                                   {"total_tokens", (int)prompt.size() + n},
                                   // an accepted-and-inert budget field is worse
                                   // than none: say when it fired.
                                   {"reasoning_tokens", tb.used},
                                   {"reasoning_budget_exceeded", tb.tripped}}}};
            res.set_content(jdump(out), "application/json");
            return;
        }

        res.set_header("Content-Type", "text/event-stream");
        const bool has_tools = tools.is_array() && !tools.empty();
        q27k::SampleParams samp = parse_sample(body);
        res.set_chunked_content_provider(
            "text/event-stream",
            // EVERY handler local this lambda reads must be captured BY VALUE:
            // httplib runs the provider from write_response(), long after this
            // handler's frame is dead (routing() and write_response() are
            // sibling calls in Server::process_request). `thinking` shipped as
            // a by-reference read of that dead frame from 2026-07-20 until the
            // fix; the bug was benign only by stack-layout luck.
            [&, samp, prompt, n_max, created, chat, objd, rt, inc_usage, routed_chat,
             tools, tool_names_v, allowed_tool_names, tchoice, stable_len, has_tools, rid,
             thinking, tcfg, sys_len, think_aware](size_t, httplib::DataSink& sink) {
                Slot& sl = claim_slot(prompt,n_max,thinking,tcfg,think_aware);
                auto sl_lease = slot_guard(sl);
                Engine& eng = *sl.eng;
                HookGuard hooks{eng}; // see the non-stream twin
                eng.samp = samp;
                eng.pfx_sys_len = sys_len; // P16b
                std::optional<q27::GpuGate::Lease> lk; // see the non-stream twin
                if (!conductor) lk.emplace(gpu_gate);
                double qw = ms_since(rt.t0);
                eng.on_round_gap = make_yield(eng);
                auto limits = think_aware
                    ? q27::resolve_think_decode_limits(
                          n_max, eng.max_ctx, (int)prompt.size(), eng.ctx_round_reserve(),
                          (int)think_close_ids.size(), thinking, tcfg, think_budget_flag)
                    : q27::resolve_think_decode_limits(
                          n_max, eng.max_ctx, (int)prompt.size(), eng.ctx_round_reserve(),
                          0, false, q27::ThinkCfg{false, -1, true}, 0);
                const int nm = limits.n_max;
                const int think_budget = limits.budget;
                auto send = [&](const json& j) {
                    std::string s = "data: " + jdump(j) + "\n\n";
                    return sink.write(s.data(), s.size());
                };

                if (!routed_chat) {
                    // ORIGINAL text-only streaming behavior, byte-for-byte unchanged.
                    q27::Utf8Gate ugate;
                    auto piece_chunk = [&](const std::string& piece) {
                        json delta = chat ? json{{"content", piece}} : json{};
                        json choice = chat
                            ? json{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}
                            : json{{"index", 0}, {"text", piece}, {"finish_reason", nullptr}};
                        return json{{"id", "q27-0"}, {"object", objd}, {"created", created},
                                    {"model", served_name}, {"choices", json::array({choice})}};
                    };
                    auto on_tok = [&](int id) {
                        // empty pieces (control tokens, gate holdbacks) still probe
                        // the socket so a disconnected client stops generation
                        return send(piece_chunk(ugate.feed(tok.decode_one(id))));
                    };
                    Engine::DecodeTask bt;
                    // TODO(batch error surfacing): on a failed queue (A2) this
                    // stream just ends with a normal finish_reason -- the OpenAI
                    // SSE shape has no standard mid-stream error event, so none
                    // is invented; end=error lands in the [req] line and
                    // [req-error] carries the what().
                    int produced = conductor
                                       ? batch_generate(eng, prompt, nm,
                                                        [&](int id, bool) { return on_tok(id); },
                                                        nullptr, -1, qw, rt, bt, nullptr)
                                       : eng.generate(prompt, nm, EOS, on_tok);
                    eng.on_round_gap = nullptr;
                    std::string tailp = ugate.flush();
                    if (!tailp.empty()) send(piece_chunk(tailp));
                    // Terminal chunk with a real finish_reason (OpenAI streaming spec):
                    // clients otherwise never learn whether generation hit EOS or the
                    // token cap. produced >= nm == the length cap; else a stop.
                    {
                        const char* fr = produced >= nm ? "length" : "stop";
                        json fchoice = chat ? json{{"index", 0}, {"delta", json::object()},
                                                   {"finish_reason", fr}}
                                            : json{{"index", 0}, {"text", ""}, {"finish_reason", fr}};
                        send(json{{"id", "q27-0"}, {"object", objd}, {"created", created},
                                  {"model", served_name}, {"choices", json::array({fchoice})}});
                    }
                    // stream_options.include_usage: final usage chunk (empty
                    // choices) mirroring the non-stream usage body above.
                    if (inc_usage)
                        send(json{{"id", "q27-0"}, {"object", objd}, {"created", created},
                                  {"model", served_name}, {"choices", json::array()},
                                  {"usage", {{"prompt_tokens", (int)prompt.size()},
                                             {"completion_tokens", produced},
                                             {"total_tokens", (int)prompt.size() + produced}}}});
                    req_log(rt, qw, eng, sl.id, bat_stats(bt));
                    std::string done = "data: [DONE]\n\n";
                    sink.write(done.data(), done.size());
                    sink.done();
                    return true;
                }

                // routed_chat: think/tool-aware streaming path, an exact
                // mechanical twin of the /v1/messages SSE handler above.
                const std::string cid = "chatcmpl-q27-" + std::to_string(rid);
                ToolConstrainer tc;
                tc.eng = &eng; tc.tok = &tok; tc.cache = &tool_mask_cache;
                tc.host2dev = &sl.tool_mask_host2dev;
                tc.enabled = constrain_tools && tchoice.mode != q27::ToolChoice::FORCED &&
                            eng.samp.inv_temp <= 0.f; // constrained+sampled is Phase 3
                tc.begin(tool_names_v);
                StreamSplitter sp;
                q27::ThinkBudgetState tb{think_budget};
                Engine::DecodeTask bt;
                if (tchoice.mode == q27::ToolChoice::FORCED) sp.chan = StreamSplitter::TOOL;
                else if (thinking) sp.chan = StreamSplitter::THINK; // prompt-injected <think> opener -> start in THINK
                ReasoningBudgetObserver budget{tok, tb, eng, bt, think_close_ids, sp.chan};
                budget.start();
                q27::Utf8Gate ugate;
                bool alive = true; // cleared when a write fails (client disconnected)
                int tool_idx = 0;
                bool any_call = false;
                std::string tool_buf;
                q27::BareToolTextHoldback bare_text;
                bool forced_control_token = false;
                auto emit_text = [&](const std::string& t) {
                    if (t.empty()) return;
                    if (!send(q27::openai_stream_chunk(cid, objd, created, served_name,
                                                       json{{"content", t}})))
                        alive = false;
                };
                auto emit_call = [&](const q27::ToolCall& c) {
                    if (!c.ok || !q27::tool_choice_allows_call(
                        tchoice,allowed_tool_names,c.name,any_call?1u:0u))
                        return false;
                    any_call = true;
                    std::string tid =
                        "call_q27_" + std::to_string(rid) + "_" + std::to_string(tool_idx);
                    bool ok = send(q27::openai_stream_chunk(
                        cid, objd, created, served_name,
                        q27::openai_tool_call_delta(tool_idx, tid, c)));
                    tool_idx++;
                    if (!ok) alive = false;
                    return true;
                };
                auto classify_bare = [&](const std::string& source,bool allow_repair,
                                         auto&& visible) {
                    std::string pre,residual;
                    auto bcs = q27::parse_bare_tool_calls(
                        source,&pre,&tools,true,allow_repair,&residual);
                    if (bcs.empty()) return q27::BareToolCandidateResult{};
                    size_t cursor=0,recovered=0;
                    for (const auto& bc : bcs) {
                        visible(source.substr(cursor,bc.source_begin-cursor));
                        cursor=bc.source_end;
                        if (emit_call(bc)) recovered++;
                        else visible(source.substr(
                            bc.source_begin,bc.source_end-bc.source_begin));
                    }
                    visible(source.substr(cursor));
                    if (recovered)
                        fprintf(stderr,
                                "[tool-fallback] %zu bare call(s) recovered (oai-stream)\n",
                                recovered);
                    return q27::BareToolCandidateResult{true,recovered!=0};
                };
                auto emit_tool = [&]() {
                    auto c = q27::parse_tool_call(q27::strip_ws2(tool_buf));
                    tool_buf.clear();
                    if (!c.ok) {
                        if (!classify_bare(c.raw,true,emit_text)) emit_text(c.raw);
                    } else if (!emit_call(c)) emit_text(c.raw);
                };
                auto emit_seg = [&](StreamSplitter::Chan ch, const std::string& t) {
                    if (ch == StreamSplitter::TOOL) {
                        if (tool_buf.empty()) {
                            bare_text.finish(false,allowed_tool_names,
                                             emit_text,classify_bare);
                            bare_text.reset_context();
                        }
                        tool_buf += t;
                        return;
                    }
                    if (!tool_buf.empty()) emit_tool();
                    if (t.empty()) return;
                    // reasoning_content (no official OpenAI field for this;
                    // matches the vLLM/SGLang/llama.cpp convention -- see
                    // openai_reasoning_delta) rather than leaking raw <think>
                    // tags into `content` (the bug this whole path also
                    // happens to fix).
                    if (ch == StreamSplitter::THINK) {
                        bare_text.finish(false,allowed_tool_names,
                                         emit_text,classify_bare);
                        bare_text.reset_context();
                        if (!send(q27::openai_stream_chunk(cid, objd, created, served_name,
                                                            q27::openai_reasoning_delta(t))))
                            alive = false;
                        return;
                    }
                    // Only decoder-injected close whitespace is parser control.
                    // Preserve ordinary leading whitespace, including when
                    // thinking is disabled or the model closes naturally.
                    if (forced_control_token && q27::strip_ws2(t).empty()) return;
                    if (has_tools)
                        bare_text.route(t,allowed_tool_names,
                                        emit_text,classify_bare);
                    else emit_text(t);
                };
                eng.on_pending = [&](int id) { tc.on_pending(id); };
                eng.on_drafts = [&](const int* dr) { tc.on_drafts(dr); };
                eng.on_round = [&](const int* em, int nr) {
                    int bm = budget.preview_round(em, nr);
                    int budget_kept = (bm >= 1 && bm < nr) ? bm : nr;
                    int m = tc.enabled ? tc.scan_round(em, budget_kept) : -1;
                    int kept = (m >= 1 && m <= budget_kept) ? m : budget_kept;
                    budget.commit_round(em, kept);
                    if (kept < nr) return kept;
                    return m;
                };
                auto on_tok = [&](int id, bool forced) {
                    forced_control_token = forced;
                    for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) emit_seg(ch, t);
                    return alive; // stop generating once the client has disconnected
                };
                int produced = conductor
                                   ? batch_generate(eng, prompt, nm, on_tok,
                                                    [&](int id) { tc.on_id(id); },
                                                    stable_len, qw, rt, bt, nullptr)
                                   : eng.generate(prompt, nm, EOS, [&](int id) {
                                         tc.on_id(id);
                                         return on_tok(id, bt.callback_forced);
                                     }, stable_len, &bt);
                tc.end();
                eng.on_pending = nullptr;
                eng.on_drafts = nullptr;
                eng.on_round = nullptr;
                eng.on_round_gap = nullptr;
                req_log(rt, qw, eng, sl.id, tg_stats(tc) + bat_stats(bt));
                for (auto& [ch, t] : sp.feed(ugate.flush())) emit_seg(ch, t);
                const bool final_tool_incomplete=q27::unfinished_tool_wrapper(
                    produced,nm,bt.budget_truncated,sp.chan);
                for (auto& [ch, t] : sp.flush()) emit_seg(ch, t);
                if (!tool_buf.empty()) {
                    if (final_tool_incomplete) {
                        emit_text(tool_buf);
                        tool_buf.clear();
                    } else emit_tool();
                }
                if (!final_tool_incomplete)
                    bare_text.finish(produced < nm && !bt.budget_truncated,
                                     allowed_tool_names,emit_text,classify_bare);
                // TODO(batch error surfacing): no standard OpenAI mid-stream
                // error chunk exists (matches the plain-text leg's TODO
                // above); end=error lands in the [req] line, [req-error]
                // carries the what() (batch_generate logs it unconditionally
                // when err_out is null, same as that leg's nullptr err_out).
                if (q27::forced_tool_choice_missing_is_error(
                        tchoice, any_call, produced >= nm || bt.budget_truncated)) {
                    send(json{{"error",{{"message","model produced no eligible tool call for forced tool_choice"},
                                         {"type","api_error"}}}});
                    std::string done = "data: [DONE]\n\n";
                    sink.write(done.data(), done.size());
                    sink.done();
                    return true;
                }
                {
                    const char* fr=q27::openai_tool_finish_reason(
                        any_call,final_tool_incomplete,
                        produced>=nm || bt.budget_truncated);
                    send(q27::openai_stream_chunk(cid, objd, created, served_name,
                                                  json::object(), fr));
                }
                if (inc_usage)
                    send(json{{"id", cid}, {"object", objd}, {"created", created},
                              {"model", served_name}, {"choices", json::array()},
                              {"usage", {{"prompt_tokens", (int)prompt.size()},
                                         {"completion_tokens", produced},
                                         {"total_tokens", (int)prompt.size() + produced},
                                         {"reasoning_tokens", tb.used},
                                         {"reasoning_budget_exceeded", tb.tripped}}}});
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

    auto prepare_anthropic_prompt = [&](const json& body,
                                         q27::ToolChoice& tchoice,
                                         json& tools,
                                         std::vector<std::string>& tool_names,
                                         bool& thinking,
                                         q27::ThinkCfg& tcfg,
                                         size_t* stable_off,
                                         size_t* sys_off) {
        tchoice=q27::parse_anthropic_tool_choice(body);
        json all_tools=q27::anthropic_tools_json(body);
        json normalized={{"tools",all_tools}};
        q27::OpenAIToolSelection selected=q27::select_openai_tools(normalized,tchoice);
        const json unavailable=q27::unselected_openai_tools(all_tools,selected);
        // Only eligible schemas enter the callable interface; the rest remain
        // in a separately labelled accounting block.
        tools=std::move(selected.tools);
        tool_names=std::move(selected.names);
        tcfg=q27::resolve_think_cfg(body,!no_think_srv,req_think,-1);
        q27::validate_anthropic_tool_choice_thinking(tchoice,tcfg);
        thinking=tcfg.enabled;
        if(tchoice.mode==q27::ToolChoice::FORCED) {
            thinking=false;
            tcfg=q27::ThinkCfg{false,-1,false,true};
        }
        std::string rendered=q27::chatml_prompt(
            q27::anthropic_msgs(body),tools,thinking,stable_off,sys_off,
            q27::anthropic_tool_choice_instruction(tchoice),&unavailable);
        if(tchoice.mode==q27::ToolChoice::FORCED) rendered+="<tool_call>\n";
        return rendered;
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
        q27::ToolChoice tchoice;
        json tools;
        std::vector<std::string> tool_names;
        bool thinking=false;
        q27::ThinkCfg tcfg;
        std::string rendered;
        try {
            rendered=prepare_anthropic_prompt(
                body,tchoice,tools,tool_names,thinking,tcfg,nullptr,nullptr);
        } catch(const std::exception& e) {
            anthropic_400(res,e.what());
            return;
        }
        json out = {{"input_tokens", (long)tok.encode(rendered).size()}};
        res.set_content(jdump(out), "application/json");
    });

    srv.Post("/v1/messages", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { anthropic_400(res, "invalid JSON body"); return; }
        int n_max = (int)q27::jint(body, "max_tokens", 8192); // unified default (see /v1/chat/completions)
        bool stream = q27::jbool(body, "stream", false);
        auto tk0 = std::chrono::steady_clock::now();
        q27::ToolChoice tchoice;
        json tools;
        std::vector<std::string> tool_names_v;
        bool thinking=false;
        q27::ThinkCfg tcfg;
        size_t stable_off=0, sys_off=0;
        std::string rendered;
        try {
            rendered=prepare_anthropic_prompt(
                body,tchoice,tools,tool_names_v,thinking,tcfg,&stable_off,&sys_off);
        } catch(const std::exception& e) {
            anthropic_400(res,e.what());
            return;
        }
        auto tk1 = std::chrono::steady_clock::now();
        const std::set<std::string> allowed_tool_names(
            tool_names_v.begin(),tool_names_v.end());
        int sys_len = 0; // P16b: system-block tokens (0 = none/feature off)
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
        if (pfx_cache.enabled() && sys_off > 0)
            sys_len = (int)tok.encode(rendered.substr(0, sys_off)).size(); // P16b, see twin
        // Q27_SYSBLK=1: system-block geometry per request. The diagnostic for
        // "why did cross-session prefix reuse miss" -- a client whose system
        // block changes size between sessions cannot share a prefix at all.
        if (sys_len && getenv("Q27_SYSBLK"))
            fprintf(stderr, "[sysblk] sys_off=%zu chars sys_len=%d toks stable_off=%zu\n",
                    sys_off, sys_len, stable_off);
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
        const auto admission = q27::resolve_think_decode_limits(
            n_max, max_slot_ctx, (int)prompt.size(), slots[0].eng->ctx_round_reserve(),
            (int)think_close_ids.size(), thinking, tcfg, think_budget_flag);
        const int request_max_prompt = admission.context_ok
            ? max_prompt
            : q27::max_prompt_for_think_decode(
                  n_max, max_slot_ctx, slots[0].eng->ctx_round_reserve(), max_prompt,
                  (int)prompt.size(), (int)think_close_ids.size(), thinking, tcfg,
                  think_budget_flag);
        if ((int)prompt.size() > max_prompt || !admission.context_ok) {
            fprintf(stderr, "[ctx-limit] prompt=%zu max=%d -> 400\n", prompt.size(),
                    request_max_prompt);
            anthropic_400(
                res, q27::ctx_limit_error_message((int)prompt.size(), request_max_prompt));
            return;
        }
        // Q27_SAMPLED=0 preflight (see the OpenAI handler's twin)
        if (!sampled_on && q27::jnum(body, "temperature", 0.0) > 0.0) {
            anthropic_400(res,
                          "sampling disabled: server booted with Q27_SAMPLED=0 "
                          "(greedy-only)");
            return;
        }
        if ((int)prompt.size() + n_max > max_slot_ctx)
            n_max = max_slot_ctx - (int)prompt.size();
        long rid = req_counter++;
        std::string mid = "msg_q27_" + std::to_string(rid);
        ReqTrace rt{rid, "anth", conv_fp(body), std::chrono::steady_clock::now(),
                    std::chrono::duration<double, std::milli>(tk2 - tk0).count()};

        if (!stream) {
            Slot& sl = claim_slot(prompt, n_max, thinking, tcfg, true);
            auto sl_lease = slot_guard(sl);
            Engine& eng = *sl.eng;
            HookGuard hooks{eng}; // M1: clears tc hooks on unwind, pre slot-free
            eng.samp = parse_sample(body);
            eng.pfx_sys_len = sys_len; // P16b (0 on paths with no system boundary)
            std::optional<q27::GpuGate::Lease> lk; // solo whole-call hold; batch
            if (!conductor) lk.emplace(gpu_gate);  // leases inside batch_generate
            double qw = ms_since(rt.t0);
            eng.on_round_gap = make_yield(eng);
            auto limits = q27::resolve_think_decode_limits(
                n_max, eng.max_ctx, (int)prompt.size(), eng.ctx_round_reserve(),
                (int)think_close_ids.size(), thinking, tcfg, think_budget_flag);
            n_max = limits.n_max;
            const int think_budget = limits.budget;
            StreamSplitter sp;
            q27::ThinkBudgetState tb{think_budget};
            Engine::DecodeTask bt;
            if(tchoice.mode==q27::ToolChoice::FORCED) sp.chan=StreamSplitter::TOOL;
            else if (thinking) sp.chan = StreamSplitter::THINK; // prompt-injected opener
            ReasoningBudgetObserver budget{tok, tb, eng, bt, think_close_ids, sp.chan};
            budget.start();
            q27::Utf8Gate ugate;
            std::vector<std::pair<StreamSplitter::Chan,std::string>> segments;
            auto route = [&](StreamSplitter::Chan ch, const std::string& t) {
                if (t.empty()) {
                    if (ch != StreamSplitter::TOOL && !segments.empty() &&
                        segments.back().first == StreamSplitter::TOOL)
                        segments.emplace_back(ch, std::string());
                    return;
                }
                if (!segments.empty() && segments.back().first == ch)
                    segments.back().second += t;
                else segments.emplace_back(ch, t);
            };
            ToolConstrainer tc;
            tc.eng = &eng; tc.tok = &tok; tc.cache = &tool_mask_cache;
            tc.host2dev = &sl.tool_mask_host2dev;
            tc.enabled = constrain_tools && tchoice.mode!=q27::ToolChoice::FORCED &&
                         eng.samp.inv_temp <= 0.f; // constrained+sampled is Phase 3
            tc.begin(tool_names_v);
            eng.on_pending = [&](int id) { tc.on_pending(id); };
            eng.on_drafts = [&](const int* dr) { tc.on_drafts(dr); };
            eng.on_round = [&](const int* em, int nr) {
                int bm = budget.preview_round(em, nr);
                int budget_kept = (bm >= 1 && bm < nr) ? bm : nr;
                int m = tc.enabled ? tc.scan_round(em, budget_kept) : -1;
                int kept = (m >= 1 && m <= budget_kept) ? m : budget_kept;
                budget.commit_round(em, kept);
                if (kept < nr) return kept;
                return m;
            };
            auto on_tok = [&](int id, bool forced) {
                for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) {
                    if (forced && ch == StreamSplitter::TEXT && q27::strip_ws2(t).empty())
                        continue;
                    route(ch, t);
                }
                return true;
            };
            std::string berr;
            // batch: tc.on_id rides on_emit -- the CONDUCTOR thread, between
            // scan_round and on_pending, its exact solo slot (driver contract)
            int n = conductor
                        ? batch_generate(eng, prompt, n_max,
                                         [&](int id, bool forced) { return on_tok(id, forced); },
                                         [&](int id) { tc.on_id(id); }, stable_len, qw,
                                         rt, bt, &berr)
                        : eng.generate(prompt, n_max, EOS, [&](int id) {
                              tc.on_id(id);
                              return on_tok(id, bt.callback_forced);
                          }, stable_len, &bt);
            tc.end();
            eng.on_pending = nullptr;
            eng.on_drafts = nullptr;
            eng.on_round = nullptr;
            eng.on_round_gap = nullptr;
            req_log(rt, qw, eng, sl.id, tg_stats(tc) + bat_stats(bt));
            // batch error surfacing (review pass 2): nothing emitted = an
            // honest 500 in the Anthropic error envelope (api_error, NOT
            // invalid_request_error: 400s tell Claude Code to compact/give
            // up, 500s are retryable). Tokens produced = keep the 200 with
            // partial content; end=error is in the [req] line either way.
            if (!berr.empty() && n == 0) {
                res.status = 500;
                res.set_content(q27::anthropic_error_json("api_error", berr),
                                "application/json");
                return;
            }
            for (auto& [ch, t] : sp.feed(ugate.flush())) route(ch, t);
            const bool final_tool_incomplete=q27::unfinished_tool_wrapper(
                n,n_max,bt.budget_truncated,sp.chan);
            for (auto& [ch, t] : sp.flush()) route(ch, t);
            std::string unclosed_tool=q27::take_unclosed_final_tool_segment(
                segments,final_tool_incomplete);
            auto ordered = q27::resolve_ordered_tool_segments(
                segments, tools.is_array() && !tools.empty() ? &tools : nullptr,
                n < n_max && !bt.budget_truncated,
                [&](const std::string& name, size_t accepted) {
                    return q27::tool_choice_allows_call(
                        tchoice,allowed_tool_names,name,accepted);
                });
            ordered.append_visible_text(unclosed_tool);
            json content = json::array();
            std::string th = q27::strip_ws2(ordered.reasoning);
            if (!th.empty())
                content.push_back({{"type", "thinking"}, {"thinking", th},
                                   {"signature", "q27-local"}});
            const bool any_call = !ordered.calls.empty();
            if(q27::forced_tool_choice_missing_is_error(
                    tchoice,any_call,n>=n_max || bt.budget_truncated)) {
                res.status=500;
                res.set_content(q27::anthropic_error_json(
                    "api_error","model produced no eligible tool call for forced tool_choice"),
                    "application/json");
                return;
            }
            if (ordered.recovered)
                fprintf(stderr, "[tool-fallback] %zu bare call(s) recovered (nonstream)\n",
                        ordered.recovered);
            const size_t emitted=q27::append_anthropic_ordered_content(
                content,ordered,[&](const q27::ToolCall& call,size_t call_number) {
                    return json{{"type","tool_use"},
                                {"id","toolu_q27_"+std::to_string(rid)+"_"+
                                      std::to_string(call_number)},
                                {"name",call.name},{"input",call.arguments}};
                });
            if(emitted==0 && !any_call && th.empty())
                content.push_back({{"type","text"},{"text",""}});
            const bool generation_truncated=n>=n_max || bt.budget_truncated;
            const char* sr=q27::anthropic_tool_stop_reason(
                any_call,final_tool_incomplete,generation_truncated);
            json out = {{"id", mid}, {"type", "message"}, {"role", "assistant"},
                        {"model", served_name}, {"content", content},
                        {"stop_reason", sr}, {"stop_sequence", nullptr},
                        {"usage", {{"input_tokens", (int)prompt.size()},
                                   {"output_tokens", n},
                                   {"reasoning_tokens", tb.used},
                                   {"reasoning_budget_exceeded", tb.tripped}}}};
            res.set_content(jdump(out), "application/json");
            return;
        }

        res.set_header("Content-Type", "text/event-stream");
        const bool has_tools = tools.is_array() && !tools.empty();
        q27k::SampleParams samp = parse_sample(body);
        res.set_chunked_content_provider(
            "text/event-stream",
            // by-value or dangling: see the /v1/chat/completions twin
            [&, samp, prompt, n_max, mid, rid, has_tools, tool_names_v,
             allowed_tool_names, tchoice, tools, stable_len, rt, thinking, tcfg,
             sys_len](size_t, httplib::DataSink& sink) {
                Slot& sl = claim_slot(prompt, n_max, thinking, tcfg, true);
                auto sl_lease = slot_guard(sl);
                Engine& eng = *sl.eng;
                HookGuard hooks{eng}; // M1: clears tc hooks on unwind, pre slot-free
                eng.samp = samp;
                eng.pfx_sys_len = sys_len; // P16b
                std::optional<q27::GpuGate::Lease> lk; // see the non-stream twin
                if (!conductor) lk.emplace(gpu_gate);
                double qw = ms_since(rt.t0);
                eng.on_round_gap = make_yield(eng);
                auto limits = q27::resolve_think_decode_limits(
                    n_max, eng.max_ctx, (int)prompt.size(), eng.ctx_round_reserve(),
                    (int)think_close_ids.size(), thinking, tcfg, think_budget_flag);
                const int nm = limits.n_max;
                const int think_budget = limits.budget;
                ToolConstrainer tc;
                tc.eng = &eng; tc.tok = &tok; tc.cache = &tool_mask_cache;
                tc.host2dev = &sl.tool_mask_host2dev;
                tc.enabled = constrain_tools && tchoice.mode!=q27::ToolChoice::FORCED &&
                             eng.samp.inv_temp <= 0.f; // constrained+sampled is Phase 3
                tc.begin(tool_names_v);
                int block_counter = 0, tool_counter = 0;
                bool any_call = false;
                bool alive = true; // cleared when a write fails (client disconnected)
                auto ev = [&](const char* name, const json& j) {
                    std::string s = std::string("event: ") + name + "\ndata: " + jdump(j) + "\n\n";
                    bool ok = sink.write(s.data(), s.size());
                    if (!ok) alive = false;
                    return ok;
                };
                json msg = {{"id", mid}, {"type", "message"}, {"role", "assistant"},
                            {"model", served_name}, {"content", json::array()},
                            {"stop_reason", nullptr}, {"stop_sequence", nullptr},
                            {"usage", {{"input_tokens", (int)prompt.size()}, {"output_tokens", 0}}}};
                ev("message_start", {{"type", "message_start"}, {"message", msg}});

                StreamSplitter sp;
                q27::ThinkBudgetState tb{think_budget};
                Engine::DecodeTask bt;
                if(tchoice.mode==q27::ToolChoice::FORCED) sp.chan=StreamSplitter::TOOL;
                else if (thinking) sp.chan = StreamSplitter::THINK; // prompt-injected opener
                ReasoningBudgetObserver budget{tok, tb, eng, bt, think_close_ids, sp.chan};
                budget.start();
                std::string tool_buf;
                q27::BareToolTextHoldback bare_text;
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
                auto emit_text = [&](const std::string& t) {
                    if (t.empty()) return;
                    open_block(0);
                    ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                        {"delta", {{"type", "text_delta"}, {"text", t}}}});
                };
                auto emit_call = [&](const q27::ToolCall& c) {
                    if (!c.ok || !q27::tool_choice_allows_call(
                            tchoice,allowed_tool_names,c.name,any_call?1u:0u))
                        return false;
                    any_call = true;
                    any = true;
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
                    return true;
                };
                auto classify_bare = [&](const std::string& source,bool allow_repair,
                                         auto&& visible) {
                    std::string pre,residual;
                    auto calls=q27::parse_bare_tool_calls(
                        source,&pre,&tools,true,allow_repair,&residual);
                    if (calls.empty()) return q27::BareToolCandidateResult{};
                    size_t cursor=0,recovered=0;
                    for (const auto& call:calls) {
                        visible(source.substr(cursor,call.source_begin-cursor));
                        cursor=call.source_end;
                        if (emit_call(call)) recovered++;
                        else visible(source.substr(
                            call.source_begin,call.source_end-call.source_begin));
                    }
                    visible(source.substr(cursor));
                    if(recovered)
                        fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (stream)\n",
                                recovered);
                    return q27::BareToolCandidateResult{true,recovered!=0};
                };
                auto emit_tool = [&]() {
                    auto c = q27::parse_tool_call(q27::strip_ws2(tool_buf));
                    tool_buf.clear();
                    if (!c.ok) {
                        if (!classify_bare(c.raw,true,emit_text)) emit_text(c.raw);
                    } else if (!emit_call(c)) emit_text(c.raw);
                };
                bool forced_control_token = false;
                auto emit_seg = [&](StreamSplitter::Chan ch, const std::string& t) {
                    if (ch == StreamSplitter::TOOL) {
                        if(tool_buf.empty()) {
                            bare_text.finish(false,allowed_tool_names,
                                             emit_text,classify_bare);
                            bare_text.reset_context();
                        }
                        tool_buf += t;
                        return;
                    }
                    if (!tool_buf.empty()) emit_tool();
                    if (t.empty()) return;
                    const bool think = ch == StreamSplitter::THINK;
                    // Injected template whitespace is parser control; model
                    // whitespace, including after a natural close, is content.
                    if (!think && forced_control_token && q27::strip_ws2(t).empty()) return;
                    if (think) {
                        bare_text.finish(false,allowed_tool_names,
                                         emit_text,classify_bare);
                        bare_text.reset_context();
                        open_block(1);
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "thinking_delta"}, {"thinking", t}}}});
                    } else if (has_tools) {
                        bare_text.route(t,allowed_tool_names,
                                        emit_text,classify_bare);
                    } else emit_text(t);
                };
                eng.on_pending = [&](int id) { tc.on_pending(id); };
                eng.on_drafts = [&](const int* dr) { tc.on_drafts(dr); };
                eng.on_round = [&](const int* em, int nr) {
                    int bm = budget.preview_round(em, nr);
                    int budget_kept = (bm >= 1 && bm < nr) ? bm : nr;
                    int m = tc.enabled ? tc.scan_round(em, budget_kept) : -1;
                    int kept = (m >= 1 && m <= budget_kept) ? m : budget_kept;
                    budget.commit_round(em, kept);
                    if (kept < nr) return kept;
                    return m;
                };
                auto on_tok = [&](int id, bool forced) {
                    forced_control_token = forced;
                    for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) emit_seg(ch, t);
                    return alive; // stop generating once the client has disconnected
                };
                std::string berr;
                // batch: tc.on_id rides on_emit (conductor thread, solo slot);
                // a dead client flips `alive` -> the drain cancels (A3)
                int produced = conductor
                                   ? batch_generate(eng, prompt, nm, on_tok,
                                                    [&](int id) { tc.on_id(id); },
                                                    stable_len, qw, rt, bt, &berr)
                                   : eng.generate(prompt, nm, EOS, [&](int id) {
                                         tc.on_id(id);
                                         return on_tok(id, bt.callback_forced);
                                     }, stable_len, &bt);
                tc.end();
                eng.on_pending = nullptr;
                eng.on_drafts = nullptr;
                eng.on_round = nullptr;
                eng.on_round_gap = nullptr;
                req_log(rt, qw, eng, sl.id, tg_stats(tc) + bat_stats(bt));
                for (auto& [ch, t] : sp.feed(ugate.flush())) emit_seg(ch, t);
                const bool final_tool_incomplete=q27::unfinished_tool_wrapper(
                    produced,nm,bt.budget_truncated,sp.chan);
                for (auto& [ch, t] : sp.flush()) emit_seg(ch, t);
                if (!tool_buf.empty()) {
                    if (final_tool_incomplete) {
                        emit_text(tool_buf);
                        tool_buf.clear();
                    } else emit_tool();
                }
                if(!final_tool_incomplete)
                    bare_text.finish(produced<nm && !bt.budget_truncated,
                                     allowed_tool_names,emit_text,classify_bare);
                if (idx < 0 && !any) { // nothing at all: empty text block for validity
                    idx = block_counter++;
                    chan_open = 0;
                    ev("content_block_start", {{"type", "content_block_start"}, {"index", idx},
                                               {"content_block", {{"type", "text"}, {"text", ""}}}});
                }
                close_block();
                // batch error surfacing (review pass 2): the Anthropic SSE
                // shape has a first-class `error` event -- emit it through
                // the existing ev() writer (envelope = anthropic_error_json's
                // shape) so clients learn the generation FAILED instead of
                // reading a silent early end_turn. message_delta/message_stop
                // still follow: error-aware clients abort at the event,
                // naive ones still get a well-formed stream.
                const bool forced_tool_missing=q27::forced_tool_choice_missing_is_error(
                    tchoice,any_call,produced>=nm || bt.budget_truncated);
                if (!berr.empty() || forced_tool_missing)
                    ev("error", {{"type", "error"},
                                 {"error", {{"type", "api_error"},
                                    {"message", forced_tool_missing
                                        ? "model produced no eligible tool call for forced tool_choice"
                                        : berr}}}});
                const char* sr=q27::anthropic_tool_stop_reason(
                    any_call,final_tool_incomplete,
                    produced>=nm || bt.budget_truncated);
                ev("message_delta", {{"type", "message_delta"},
                                     {"delta", {{"stop_reason", sr}, {"stop_sequence", nullptr}}},
                                     {"usage", {{"output_tokens", produced},
                                                {"reasoning_tokens", tb.used},
                                                {"reasoning_budget_exceeded", tb.tripped}}}});
                ev("message_stop", {{"type", "message_stop"}});
                sink.done();
                return true;
            });
    });

    // ---------------- OpenAI Responses API (Codex CLI) ----------------
    // Wire facts from codex-rs (v0.143): client keys off the JSON `type` field
    // in SSE data (event: lines ignored); the agent loop consumes only
    // response.output_item.done items; response.completed / response.incomplete
    // terminate their matching lifecycle states; function_call.arguments is a
    // JSON-encoded STRING; tool results arrive as function_call_output with a
    // bare-string output.
    // 400 is fatal to Codex, 500 retries -- so tolerate quirks, 500 on bugs.

    srv.Post("/v1/responses", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return; }

        long rid = req_counter++;
        std::string resp_id = "resp_q27_" + std::to_string(rid);

        // tools: flat function entries pass through; `custom` freeform tools
        // (apply_patch) are bridged to a one-string-param function. The legacy
        // `shell` hosted form expands to the concrete client-executed functions;
        // other hosted types remain client-side but still govern output eligibility.
        json tools = json::array();
        std::set<std::string> custom_names;
        std::set<std::string> hosted_tool_types;
        std::set<std::string> declared_tool_names;
        std::set<std::string> function_names;
        bool duplicate_tool_name = false;
        try {
            q27::validate_responses_tool_fields(body);
        } catch(const std::exception& e) {
            res.status=400;
            res.set_content(json{{"error",{{"message",e.what()},
                                             {"type","invalid_request_error"}}}}.dump(),
                            "application/json");
            return;
        }
        if (body.contains("tools") && body["tools"].is_array())
            for (auto& t : body["tools"]) {
                if (!t.is_object()) continue; // value() on a non-object throws 306
                std::string ty = q27::jstr(t, "type");
                if (ty == "function") {
                    std::string fn = q27::jstr(t, "name");
                    if (!declared_tool_names.insert(fn).second) duplicate_tool_name = true;
                    function_names.insert(fn);
                    tools.push_back({{"type", "function"},
                                     {"function", {{"name", fn},
                                                   {"description", q27::jstr(t, "description")},
                                                   {"parameters", t.contains("parameters")
                                                                      ? t["parameters"]
                                                                      : json::object()}}}});
                } else if (ty == "custom") {
                    std::string cn = q27::jstr(t, "name");
                    custom_names.insert(cn);
                    if (!declared_tool_names.insert(cn).second) duplicate_tool_name = true;
                    json params = {{"type", "object"},
                                   {"properties",
                                    {{"input", {{"type", "string"},
                                                {"description", "The complete raw input text "
                                                                "for this tool."}}}}},
                                   {"required", json::array({"input"})}};
                    tools.push_back({{"type", "function"},
                                     {"function", {{"name", cn},
                                                   {"description", q27::jstr(t, "description")},
                                                   {"parameters", params}}}});
                } else if (!ty.empty()) {
                    hosted_tool_types.insert(ty);
                    if (ty == "shell")
                        for (auto& shell_tool : q27::responses_shell_prompt_tools()) {
                            const std::string name =
                                shell_tool["function"]["name"].get<std::string>();
                            if (!declared_tool_names.insert(name).second)
                                duplicate_tool_name = true;
                            tools.push_back(std::move(shell_tool));
                        }
                }
            }

        q27::ToolChoice tchoice;
        std::vector<std::string> tool_names_v;
        std::set<std::string> allowed_hosted_names;
        try {
            if (duplicate_tool_name)
                throw std::runtime_error("ambiguous duplicate Responses tool name");
            if (q27::responses_tool_names_ambiguous(
                    function_names, custom_names, hosted_tool_types))
                throw std::runtime_error("ambiguous duplicate Responses tool name");
            q27::validate_responses_tool_choice_declarations(
                body, function_names, custom_names, hosted_tool_types);
            json selection_body = body;
            selection_body["tools"] = tools;
            tchoice = q27::parse_responses_tool_choice(selection_body);
            q27::apply_openai_parallel_tool_calls(body, tchoice);
            if (tchoice.invalid)
                throw std::runtime_error("invalid tool_choice or parallel_tool_calls");

            q27::ToolChoice local_choice =
                q27::responses_registered_tool_choice(tchoice, hosted_tool_types);
            if (local_choice.mode == q27::ToolChoice::FORCED &&
                local_choice.forced_name.empty() && local_choice.allowed_names.empty() &&
                tools.empty() && !hosted_tool_types.empty())
                local_choice.mode = q27::ToolChoice::NONE;
            q27::OpenAIToolSelection selected =
                q27::select_openai_tools(selection_body, local_choice);
            tools = std::move(selected.tools);
            tool_names_v = std::move(selected.names);

            if (tchoice.mode != q27::ToolChoice::NONE) {
                auto allow_hosted = [&](const std::string& type) {
                    q27::add_responses_hosted_call_names(allowed_hosted_names, type);
                };
                if (!tchoice.forced_name.empty()) {
                    if (hosted_tool_types.count(tchoice.forced_name))
                        allow_hosted(tchoice.forced_name);
                } else if (!tchoice.allowed_names.empty()) {
                    for (const auto& name : tchoice.allowed_names)
                        if (hosted_tool_types.count(name)) allow_hosted(name);
                } else {
                    for (const auto& type : hosted_tool_types) allow_hosted(type);
                }
            }
            if (tchoice.mode == q27::ToolChoice::FORCED && tool_names_v.empty() &&
                allowed_hosted_names.empty())
                throw std::runtime_error("tool_choice requires at least one supported tool");
            const std::set<std::string> selected_names(tool_names_v.begin(), tool_names_v.end());
            for (auto it = custom_names.begin(); it != custom_names.end();)
                if (!selected_names.count(*it)) it = custom_names.erase(it);
                else ++it;
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error",{{"message",e.what()},
                                             {"type","invalid_request_error"}}}}.dump(),
                            "application/json");
            return;
        }
        std::set<std::string> eligible_call_names(tool_names_v.begin(), tool_names_v.end());
        eligible_call_names.insert(allowed_hosted_names.begin(), allowed_hosted_names.end());
        q27::ToolChoice response_choice =
            q27::responses_registered_tool_choice(tchoice, hosted_tool_types);

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
                    if (!it.is_object()) continue; // value() on a non-object throws 306
                    std::string ty = q27::jstr(it, "type", "message");
                    if (ty == "message") {
                        std::string role = q27::jstr(it, "role", "user");
                        if (role == "developer") role = "system";
                        msgs.push_back(
                            {role, it.contains("content") ? text_of(it["content"]) : std::string()});
                    } else if (ty == "function_call" || ty == "custom_tool_call") {
                        json args;
                        if (ty == "function_call") {
                            try { args = json::parse(q27::jstr(it, "arguments", "{}")); }
                            catch (...) { args = q27::jstr(it, "arguments"); }
                        } else {
                            args = {{"input", q27::jstr(it, "input")}};
                        }
                        msgs.push_back({"assistant",
                                        q27::tool_call_text(q27::jstr(it, "name"), args)});
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

        int n_max = (int)q27::jint(body, "max_output_tokens", 8192); // unified default (see /v1/chat/completions)
        // P16 does not apply to this shape (no stable_off / sys_off is computed
        // here), but the engine field is per-engine state -- set it so a
        // previous request's value cannot leak into this one.
        int sys_len = 0; // P16b: set below if this request carries a system block
        auto tk0 = std::chrono::steady_clock::now();
        q27::ThinkCfg tcfg = q27::resolve_think_cfg(body, !no_think_srv, req_think, -1);
        bool thinking = tcfg.enabled;
        if (tchoice.mode == q27::ToolChoice::FORCED) {
            thinking = false;
            tcfg = q27::ThinkCfg{false, -1, false, true};
        }
        size_t sys_off = 0;
        std::string rendered = q27::chatml_prompt(merged, tools, thinking, nullptr, &sys_off);
        if (tchoice.mode == q27::ToolChoice::FORCED) rendered += "<tool_call>\n";
        std::vector<int> prompt = tok.encode(rendered);
        // P16b applies here even though P16a does not: this shape computes no
        // stable_off (so it never persists a stable entry), but a system+tools
        // block is a system+tools block, and codex re-sends one every session.
        if (pfx_cache.enabled() && sys_off > 0)
            sys_len = (int)tok.encode(rendered.substr(0, sys_off)).size();
        ReqTrace rt{rid, "resp", conv_fp(body), std::chrono::steady_clock::now(),
                    ms_since(tk0)};
        // review follow-up 2026-07-09 #3: the bound includes the spec-round
        // reserve (max_prompt), so a prompt that passes can never have its
        // n_max floored to 0 by the routed slot's clamp (empty 200/stream)
        const auto admission = q27::resolve_think_decode_limits(
            n_max, max_slot_ctx, (int)prompt.size(), slots[0].eng->ctx_round_reserve(),
            (int)think_close_ids.size(), thinking, tcfg, think_budget_flag);
        if ((int)prompt.size() > max_prompt || !admission.context_ok) {
            res.status = 400; // context_length_exceeded is fatal-class for codex, correctly
            res.set_content("{\"error\":{\"code\":\"context_length_exceeded\"}}",
                            "application/json");
            return;
        }
        // Q27_SAMPLED=0 preflight (see the OpenAI handler's twin)
        if (!sampled_on && q27::jnum(body, "temperature", 0.0) > 0.0) {
            res.status = 400;
            res.set_content("{\"error\":{\"code\":\"sampling_disabled\",\"message\":"
                            "\"sampling disabled: server booted with Q27_SAMPLED=0\"}}",
                            "application/json");
            return;
        }
        if ((int)prompt.size() + n_max > max_slot_ctx)
            n_max = max_slot_ctx - (int)prompt.size();
        bool stream = q27::jbool(body, "stream", false);

        // shared generation -> output items
        struct GenOut { json items = json::array(); int produced = 0; };
        auto make_item_cbs = [&](json& items, int& tool_counter,
                                 std::function<bool(const std::string&)> on_text_delta) {
            (void)on_text_delta;
            struct Ctx {
                std::string think, text, tool_buf;
                int message_counter=0,reason_counter=0;
                bool tool_tail=false;
            };
            auto ctx=std::make_shared<Ctx>();
            auto flush_think=[&items,ctx,rid]() {
                std::string th=q27::strip_ws2(ctx->think);
                ctx->think.clear();
                if(th.empty()) return json();
                ctx->tool_tail=false;
                json item={{"type","reasoning"},
                           {"id","rs_q27_"+std::to_string(rid)+"_"+
                                 std::to_string(ctx->reason_counter++)},
                           {"status","completed"},
                           {"summary",json::array({{{"type","summary_text"},{"text",th}}})},
                           {"encrypted_content",nullptr}};
                items.push_back(item);
                return item;
            };
            auto push_text=[&items,rid,ctx](const std::string& tx,bool incomplete=false) {
                if(tx.empty()) return json();
                ctx->tool_tail=false;
                json item={{"type","message"},
                           {"id","msg_q27_"+std::to_string(rid)+"_"+
                                 std::to_string(ctx->message_counter++)},
                           {"role","assistant"},
                           {"status",incomplete?"incomplete":"completed"},
                           {"content",json::array({{{"type","output_text"},{"text",tx},
                                                    {"annotations",json::array()}}})}};
                items.push_back(item);
                return item;
            };
            auto flush_text=[ctx,push_text](bool incomplete=false) {
                return push_text(q27::take_responses_output_text(ctx->text),incomplete);
            };
            auto emit_tool=[&items,rid,&tool_counter,&custom_names,
                            &response_choice,&eligible_call_names,ctx,push_text](q27::ToolCall c) {
                if(!c.ok || !q27::tool_choice_allows_call(
                        response_choice,eligible_call_names,c.name,tool_counter))
                    return push_text(c.raw);
                const int call_index=tool_counter++;
                const std::string cid="call_q27_"+std::to_string(rid)+"_"+
                                      std::to_string(call_index);
                const std::string iid="fc_q27_"+std::to_string(rid)+"_"+
                                      std::to_string(call_index);
                json item;
                if(custom_names.count(c.name)) {
                    std::string input=c.arguments.is_object() && c.arguments.contains("input") &&
                                              c.arguments["input"].is_string()
                                          ?c.arguments["input"].get<std::string>()
                                          :jdump(c.arguments);
                    item={{"type","custom_tool_call"},{"id",iid},{"call_id",cid},
                          {"status","completed"},{"name",c.name},{"input",input}};
                } else {
                    item={{"type","function_call"},{"id",iid},{"call_id",cid},
                          {"status","completed"},{"name",c.name},
                          {"arguments",jdump(c.arguments)}};
                }
                ctx->tool_tail=true;
                items.push_back(item);
                return item;
            };
            auto flush_tool=[ctx,emit_tool]() {
                auto c=q27::parse_tool_call(q27::strip_ws2(ctx->tool_buf));
                ctx->tool_buf.clear();
                return emit_tool(std::move(c));
            };
            return std::make_tuple(ctx,flush_think,flush_text,flush_tool,
                                   emit_tool,push_text);
        };

        if (!stream) {
            Slot& sl = claim_slot(prompt, n_max, thinking, tcfg, true);
            auto sl_lease = slot_guard(sl);
            Engine& eng = *sl.eng;
            HookGuard hooks{eng};
            eng.samp = parse_sample(body);
            eng.pfx_sys_len = sys_len; // P16b (0 on paths with no system boundary)
            std::optional<q27::GpuGate::Lease> lk; // solo whole-call hold; batch
            if (!conductor) lk.emplace(gpu_gate);  // leases inside batch_generate
            double qw = ms_since(rt.t0);
            eng.on_round_gap = make_yield(eng);
            auto limits = q27::resolve_think_decode_limits(
                n_max, eng.max_ctx, (int)prompt.size(), eng.ctx_round_reserve(),
                (int)think_close_ids.size(), thinking, tcfg, think_budget_flag);
            n_max = limits.n_max;
            const int think_budget = limits.budget;
            json items = json::array();
            int tool_counter = 0;
            auto item_cbs = make_item_cbs(items, tool_counter, nullptr);
            auto& ctx = std::get<0>(item_cbs);
            auto& flush_think = std::get<1>(item_cbs);
            auto& flush_text = std::get<2>(item_cbs);
            auto& flush_tool = std::get<3>(item_cbs);
            auto& emit_tool = std::get<4>(item_cbs);
            auto& push_text = std::get<5>(item_cbs);
            auto flush_recoverable_text=[&](bool allow_repair,bool incomplete) {
                std::string source=q27::take_responses_output_text(ctx->text);
                if(source.empty()) return;
                std::string pre,residual;
                auto calls=q27::parse_bare_tool_calls(source, &pre, tools.empty()?nullptr:&tools, true, allow_repair, &residual);
                if(calls.empty()) {
                    push_text(source,incomplete);
                    return;
                }
                std::vector<bool> accepted(calls.size(),false);
                size_t accepted_calls=tool_counter;
                std::ptrdiff_t last_accepted=-1;
                for(size_t i=0;i<calls.size();i++) {
                    accepted[i]=calls[i].ok && q27::tool_choice_allows_call(
                        response_choice,eligible_call_names,calls[i].name,
                        accepted_calls);
                    if(accepted[i]) {
                        accepted_calls++;
                        last_accepted=(std::ptrdiff_t)i;
                    }
                }
                size_t cursor=0,recovered=0;
                for(size_t i=0;i<calls.size();i++) {
                    auto& call=calls[i];
                    const bool segment_incomplete=incomplete &&
                        last_accepted<(std::ptrdiff_t)i;
                    push_text(source.substr(cursor,call.source_begin-cursor),
                              segment_incomplete);
                    cursor=call.source_end;
                    if(accepted[i]) {
                        emit_tool(std::move(call));
                        recovered++;
                    } else push_text(source.substr(call.source_begin,
                                                   call.source_end-call.source_begin),
                                     segment_incomplete);
                }
                push_text(source.substr(cursor),incomplete);
                if(recovered)
                    fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (resp nonstream)\n",
                            recovered);
            };
            StreamSplitter sp;
            q27::ThinkBudgetState tb{think_budget};
            Engine::DecodeTask bt;
            if (tchoice.mode == q27::ToolChoice::FORCED) sp.chan = StreamSplitter::TOOL;
            else if (thinking) sp.chan = StreamSplitter::THINK; // prompt-injected opener
            ReasoningBudgetObserver budget{tok, tb, eng, bt, think_close_ids, sp.chan};
            budget.start();
            eng.on_round = [&](const int* em, int nr) {
                return budget.observe_round(em, nr);
            };
            auto route = [&](StreamSplitter::Chan ch, const std::string& t) {
                if (ch == StreamSplitter::TOOL) {
                    if (!ctx->think.empty()) flush_think();
                    if (!ctx->text.empty()) flush_recoverable_text(false,false);
                    ctx->tool_buf += t;
                    return;
                }
                if (!ctx->tool_buf.empty()) flush_tool();
                if (ch == StreamSplitter::THINK) {
                    if (!ctx->text.empty()) flush_recoverable_text(false,false);
                    ctx->think += t;
                } else {
                    if (!ctx->think.empty()) flush_think();
                    ctx->text += t;
                }
            };
            q27::Utf8Gate ugate;
            auto on_tok = [&](int id, bool forced) {
                for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) {
                    if (forced && ch == StreamSplitter::TEXT && q27::strip_ws2(t).empty())
                        continue;
                    route(ch, t);
                }
                return true;
            };
            std::string berr;
            int produced = conductor
                               ? batch_generate(eng, prompt, n_max,
                                                [&](int id, bool forced) { return on_tok(id, forced); },
                                                nullptr, -1,
                                                qw, rt, bt, &berr)
                               : eng.generate(prompt, n_max, EOS,
                                              [&](int id) { return on_tok(id, bt.callback_forced); },
                                              -1, &bt);
            eng.on_round = nullptr;
            eng.on_round_gap = nullptr;
            req_log(rt, qw, eng, sl.id, bat_stats(bt));
            // batch error surfacing (review pass 2): nothing emitted = 500
            // (retryable-class for codex; 400 is fatal, header comment).
            // Tokens produced = keep the 200 with partial items.
            if (!berr.empty() && produced == 0) {
                res.status = 500;
                res.set_content(json{{"error", {{"code", "internal_error"},
                                                {"message", berr}}}}
                                    .dump(),
                                "application/json");
                return;
            }
            const bool token_limit_reached=produced>=n_max || bt.budget_truncated;
            for (auto& [ch, t] : sp.feed(ugate.flush())) route(ch, t);
            const bool unfinished_tool_wrapper=q27::unfinished_tool_wrapper(
                produced,n_max,bt.budget_truncated,sp.chan);
            for (auto& [ch, t] : sp.flush()) route(ch, t);
            if (!ctx->tool_buf.empty()) {
                if (unfinished_tool_wrapper) {
                    push_text(ctx->tool_buf,true);
                    ctx->tool_buf.clear();
                    ctx->tool_tail=false;
                } else flush_tool();
            }
            flush_think();
            const std::string& buffered_text=ctx->text;
            std::string preview_prefix,preview_residual;
            const bool allow_repair=!token_limit_reached;
            const auto preview_calls=q27::parse_bare_tool_calls(buffered_text, &preview_prefix, tools.empty()?nullptr:&tools, true, allow_repair, &preview_residual);
            const bool tool_tail=q27::responses_tool_tail_after_bare_calls(
                ctx->tool_tail,buffered_text,preview_calls,response_choice,
                eligible_call_names,tool_counter);
            const bool limit_reached=token_limit_reached && !tool_tail;
            const auto terminal=q27::responses_terminal_state(limit_reached);
            flush_recoverable_text(allow_repair,terminal.incomplete);
            if (q27::forced_tool_choice_missing_is_error(
                    tchoice, tool_counter != 0, limit_reached)) {
                res.status = 500;
                res.set_content(json{{"error",{{"message","model produced no eligible tool call for forced tool_choice"},
                                                 {"type","api_error"}}}}.dump(),
                                "application/json");
                return;
            }
            const bool incomplete = terminal.incomplete;
            json out = {{"id", resp_id}, {"object", "response"},
                        {"status", incomplete ? "incomplete" : "completed"},
                        {"model", served_name}, {"output", items},
                        {"usage", {{"input_tokens", (int)prompt.size()},
                                   {"output_tokens", produced},
                                   {"output_tokens_details", {{"reasoning_tokens", tb.used},
                                                               {"reasoning_budget_exceeded", tb.tripped}}},
                                   {"total_tokens", (int)prompt.size() + produced}}}};
            if (incomplete)
                out["incomplete_details"] = {{"reason", "max_output_tokens"}};
            res.set_content(jdump(out), "application/json");
            return;
        }

        res.set_header("Content-Type", "text/event-stream");
        q27k::SampleParams samp = parse_sample(body);
        res.set_chunked_content_provider(
            "text/event-stream",
            // by-value or dangling: see the /v1/chat/completions twin
            [&, samp, prompt, n_max, resp_id, rid, custom_names, tools, rt,
             thinking, tcfg, sys_len, tchoice, response_choice,
             eligible_call_names](size_t, httplib::DataSink& sink) {
                Slot& sl = claim_slot(prompt, n_max, thinking, tcfg, true);
                auto sl_lease = slot_guard(sl);
                Engine& eng = *sl.eng;
                HookGuard hooks{eng};
                eng.samp = samp;
                eng.pfx_sys_len = sys_len; // P16b
                std::optional<q27::GpuGate::Lease> lk; // see the non-stream twin
                if (!conductor) lk.emplace(gpu_gate);
                double qw = ms_since(rt.t0);
                eng.on_round_gap = make_yield(eng);
                auto limits = q27::resolve_think_decode_limits(
                    n_max, eng.max_ctx, (int)prompt.size(), eng.ctx_round_reserve(),
                    (int)think_close_ids.size(), thinking, tcfg, think_budget_flag);
                const int nm = limits.n_max;
                const int think_budget = limits.budget;
                bool alive = true; // cleared when a write fails (client disconnected)
                auto ev = [&](const json& j) {
                    // codex keys off data.type; the event: line is decorative
                    std::string s = "event: " + j.value("type", std::string("x")) +
                                    "\ndata: " + jdump(j) + "\n\n";
                    bool ok = sink.write(s.data(), s.size());
                    if (!ok) alive = false;
                    return ok;
                };
                ev({{"type", "response.created"},
                    {"response", {{"id", resp_id}, {"object", "response"},
                                  {"status", "in_progress"}}}});

                json items=json::array();
                int tool_counter=0,out_index=0;
                int message_counter=0,reason_counter=0;
                bool output_tool_tail=false;
                std::set<std::string> cn=custom_names;
                std::string think,text,tool_buf,bare_pending,bare_probe,
                    bare_deferred,bare_deferred_trailing,active_msg_id;
                bool bare_holding=false,bare_mode10=false,bare_input_final=false;
                bool bare_deferred_mode10=false,bare_ordinary_call_seen=false;

                q27::IncrementalBareJsonEnd bare_scan;
                q27::JsonStringLexState bare_text_lex;
                q27::MarkdownFenceLexState bare_text_fence;
                auto item_done=[&](const json& item) {
                    ev({{"type","response.output_item.done"},{"output_index",out_index++},
                        {"item",item}});
                    items.push_back(item);
                };
                auto flush_think=[&]() {
                    std::string th=q27::strip_ws2(think);
                    think.clear();
                    if(th.empty()) return;
                    output_tool_tail=false;
                    item_done({{"type","reasoning"},
                               {"id","rs_q27_"+std::to_string(rid)+"_"+
                                     std::to_string(reason_counter++)},
                               {"status","completed"},
                               {"summary",json::array({{{"type","summary_text"},{"text",th}}})},
                               {"encrypted_content",nullptr}});
                };
                int msg_index=-1;
                auto open_text=[&]() {
                    if(msg_index>=0) return;
                    msg_index=out_index;
                    active_msg_id="msg_q27_"+std::to_string(rid)+"_"+
                                  std::to_string(message_counter++);
                    ev({{"type","response.output_item.added"},{"output_index",msg_index},
                        {"item",{{"type","message"},{"id",active_msg_id},
                                  {"role","assistant"},{"status","in_progress"},
                                  {"content",json::array()}}}});
                    ev({{"type","response.content_part.added"},{"item_id",active_msg_id},
                        {"output_index",msg_index},{"content_index",0},
                        {"part",{{"type","output_text"},{"text",""},
                                  {"annotations",json::array()}}}});
                };
                auto flush_text=[&](bool incomplete=false) {
                    if(msg_index<0) { text.clear(); return; }
                    std::string tx=q27::take_responses_output_text(text);
                    ev({{"type","response.output_text.done"},{"item_id",active_msg_id},
                        {"output_index",msg_index},{"content_index",0},{"text",tx}});
                    ev({{"type","response.content_part.done"},{"item_id",active_msg_id},
                        {"output_index",msg_index},{"content_index",0},
                        {"part",{{"type","output_text"},{"text",tx},
                                  {"annotations",json::array()}}}});
                    json item={{"type","message"},{"id",active_msg_id},
                               {"role","assistant"},
                               {"status",incomplete?"incomplete":"completed"},
                               {"content",json::array({{{"type","output_text"},{"text",tx},
                                                        {"annotations",json::array()}}})}};
                    ev({{"type","response.output_item.done"},{"output_index",msg_index},
                        {"item",item}});
                    items.push_back(item);
                    out_index=msg_index+1;
                    msg_index=-1;
                    active_msg_id.clear();
                };
                auto append_text=[&](const std::string& value) {
                    if(value.empty()) return;
                    q27::consume_bare_text_context(
                        bare_text_lex,bare_text_fence,value);
                    if(value.find_first_not_of(" \t\r\n")!=std::string::npos)
                        output_tool_tail=false;
                    open_text();
                    text+=value;
                    ev({{"type","response.output_text.delta"},{"item_id",active_msg_id},
                        {"output_index",msg_index},{"content_index",0},{"delta",value}});
                };
                auto emit_call=[&](q27::ToolCall call) {
                    if(!call.ok || !q27::tool_choice_allows_call(
                            response_choice,eligible_call_names,call.name,tool_counter))
                        return false;
                    const int call_index=tool_counter++;
                    const std::string cid="call_q27_"+std::to_string(rid)+"_"+
                                          std::to_string(call_index);
                    const std::string iid="fc_q27_"+std::to_string(rid)+"_"+
                                          std::to_string(call_index);
                    json item;
                    if(cn.count(call.name)) {
                        std::string input=call.arguments.is_object() &&
                                                  call.arguments.contains("input") &&
                                                  call.arguments["input"].is_string()
                                              ?call.arguments["input"].get<std::string>()
                                              :jdump(call.arguments);
                        item={{"type","custom_tool_call"},{"id",iid},{"call_id",cid},
                              {"status","completed"},{"name",call.name},{"input",input}};
                    } else {
                        item={{"type","function_call"},{"id",iid},{"call_id",cid},
                              {"status","completed"},{"name",call.name},
                              {"arguments",jdump(call.arguments)}};
                    }
                    json added=item;
                    added["status"]="in_progress";
                    if(added["type"]=="function_call") added["arguments"]="";
                    else added["input"]="";
                    ev({{"type","response.output_item.added"},{"output_index",out_index},
                        {"item",added}});
                    if(item["type"]=="function_call")
                        ev({{"type","response.function_call_arguments.done"},
                            {"item_id",iid},{"output_index",out_index},
                            {"name",call.name},{"arguments",item["arguments"]}});
                    item_done(item);
                    output_tool_tail=true;
                    return true;
                };
                auto flush_tool=[&]() {
                    auto call=q27::parse_tool_call(q27::strip_ws2(tool_buf));
                    tool_buf.clear();
                    if(!emit_call(call)) append_text(call.raw);
                };
                auto emit_recovered=[&](const std::string& source,
                                         std::vector<q27::ToolCall>& calls,
                                         bool incomplete_item) {
                    std::vector<bool> accepted(calls.size(),false);
                    size_t accepted_calls=tool_counter;
                    std::ptrdiff_t last_accepted=-1;
                    for(size_t i=0;i<calls.size();i++) {
                        accepted[i]=calls[i].ok && q27::tool_choice_allows_call(
                            response_choice,eligible_call_names,calls[i].name,
                            accepted_calls);
                        if(accepted[i]) {
                            accepted_calls++;
                            last_accepted=(std::ptrdiff_t)i;
                        }
                    }
                    size_t cursor=0,recovered=0;
                    for(size_t i=0;i<calls.size();i++) {
                        auto& call=calls[i];
                        append_text(source.substr(cursor,call.source_begin-cursor));
                        cursor=call.source_end;
                        const bool segment_incomplete=incomplete_item &&
                            last_accepted<(std::ptrdiff_t)i;
                        if(accepted[i]) {
                            if(msg_index>=0) flush_text(segment_incomplete);
                            if(emit_call(call)) {
                                recovered++;
                                bare_text_lex.reset();
                                bare_text_fence.reset();
                            } else append_text(source.substr(
                                call.source_begin,call.source_end-call.source_begin));
                        } else append_text(source.substr(call.source_begin,
                                                         call.source_end-call.source_begin));
                    }
                    append_text(source.substr(cursor));
                    return recovered;
                };
                auto flush_bare=[&](bool allow_repair,bool incomplete_item=false,
                                    bool defer_failure=false) {
                    if(!bare_holding) return false;
                    const bool candidate_mode10=bare_mode10;
                    std::string pre,residual;
                    auto calls=q27::parse_bare_tool_calls(bare_pending, &pre, tools.empty()?nullptr:&tools, true, allow_repair, &residual);
                    size_t recovered=0;
                    if(!calls.empty()) {
                        recovered=emit_recovered(bare_pending,calls,incomplete_item);
                        if(recovered)
                            fprintf(stderr,"[tool-fallback] %zu bare call(s) recovered (resp)\n",
                                    recovered);
                    } else if(defer_failure) {
                        bare_deferred=std::move(bare_pending);
                        bare_deferred_mode10=candidate_mode10;
                    } else append_text(bare_pending);
                    if(!candidate_mode10 && recovered!=0)
                        bare_ordinary_call_seen=true;
                    bare_pending.clear();
                    bare_holding=false;
                    bare_mode10=false;
                    // A balanced candidate is a JSON classification boundary.
                    // Preserve any Markdown fence opened by emitted text.
                    bare_text_lex.reset();
                    return calls.empty() && defer_failure;
                };
                auto route_bare_text=[&](const std::string& value) {
                    if(!bare_deferred.empty()) {
                        bare_deferred_trailing+=value;
                        return;
                    }

                    std::string remaining=std::move(bare_probe);
                    bare_probe.clear();
                    remaining+=value;
                    while(!remaining.empty()) {
                        if(bare_holding) {
                            bare_pending+=remaining;
                            remaining.clear();
                        } else {
                            const size_t object_pos=q27::bare_object_position(
                                remaining,bare_text_lex,bare_text_fence,
                                bare_input_final);
                            const size_t mode10_pos=bare_ordinary_call_seen
                                ?std::string::npos
                                :q27::bare_mode10_signature_position(
                                    remaining,eligible_call_names,bare_text_lex,
                                    bare_text_fence,bare_input_final);
                            const size_t opener=object_pos==std::string::npos?mode10_pos:
                                mode10_pos==std::string::npos?object_pos:
                                std::min(object_pos,mode10_pos);
                            if(opener==std::string::npos) {
                                size_t keep=bare_input_final || bare_ordinary_call_seen
                                    ?std::string::npos
                                    :q27::bare_mode10_probe_start(
                                        remaining,eligible_call_names,bare_text_lex,
                                        bare_text_fence,false);
                                if(!bare_input_final) {
                                    const size_t inline_keep=
                                        q27::bare_unresolved_inline_probe_start(
                                            remaining,bare_text_lex,bare_text_fence);
                                    if(inline_keep!=std::string::npos)
                                        keep=keep==std::string::npos?inline_keep:
                                             std::min(keep,inline_keep);
                                }
                                if(keep==std::string::npos) {
                                    append_text(remaining);
                                } else {
                                    append_text(remaining.substr(0,keep));
                                    bare_probe=remaining.substr(keep);
                                }
                                return;
                            }
                            if(opener) append_text(remaining.substr(0,opener));
                            bare_pending=remaining.substr(opener);
                            bare_holding=true;
                            bare_mode10=mode10_pos==opener;
                            bare_scan.begin(bare_mode10);
                            remaining.clear();
                        }
                        if(!bare_mode10 &&
                           !q27::plausible_bare_tool_prefix(bare_pending)) {
                            std::string retry=std::move(bare_pending);
                            bare_pending.clear();
                            bare_holding=false;
                            append_text(retry.substr(0,1));
                            remaining=retry.substr(1);
                            continue;
                        }
                        const size_t end=bare_scan.advance(bare_pending);
                        if(end==std::string::npos) return;
                        std::string trailing=bare_pending.substr(end);
                        bare_pending.resize(end);
                        const bool defer_failure=!bare_input_final &&
                            q27::bare_candidate_repair_eligible(
                                bare_pending,eligible_call_names,bare_mode10);
                        if(flush_bare(false,false,defer_failure)) {
                            bare_deferred_trailing=std::move(trailing);
                            return;
                        }
                        remaining=std::move(trailing);

                    }
                };
                auto finish_bare=[&](bool allow_repair,
                                     bool incomplete_item=false) {
                    if(!bare_probe.empty()) {
                        bare_input_final=true;
                        std::string final_probe=std::move(bare_probe);
                        bare_probe.clear();
                        route_bare_text(final_probe);
                        bare_input_final=false;
                    }
                    std::string trailing;
                    if(!bare_deferred.empty()) {
                        bare_pending=std::move(bare_deferred);
                        trailing=std::move(bare_deferred_trailing);
                        bare_holding=true;
                        bare_mode10=bare_deferred_mode10;
                        bare_deferred_mode10=false;
                        flush_bare(allow_repair,incomplete_item);
                    }
                    if(!trailing.empty()) {
                        bare_input_final=true;
                        route_bare_text(trailing);
                        bare_input_final=false;
                    }
                    flush_bare(allow_repair,incomplete_item);
                };

                bool forced_control_token=false;
                auto route=[&](StreamSplitter::Chan ch,const std::string& value) {
                    if(ch==StreamSplitter::TOOL) {
                        if(!think.empty()) flush_think();
                        // The wrapper boundary conclusively ends the preceding
                        // text segment. Classify complete bare calls without
                        // repairing an unfinished one ahead of this wrapper.
                        if(tool_buf.empty()) {
                            finish_bare(false,false);
                            if(msg_index>=0) flush_text(false);
                            bare_text_lex.reset();
                            bare_text_fence.reset();
                            bare_ordinary_call_seen=false;

                        }
                        tool_buf+=value;
                        return;
                    }
                    if(!tool_buf.empty()) flush_tool();
                    if(value.empty()) return;
                    if(ch==StreamSplitter::THINK) {
                        finish_bare(false,false);
                        if(msg_index>=0) flush_text(false);
                        bare_text_lex.reset();
                        bare_text_fence.reset();
                        bare_ordinary_call_seen=false;

                        think+=value;
                        return;
                    }
                    if(!think.empty()) flush_think();
                    if(forced_control_token && q27::strip_ws2(value).empty()) return;
                    route_bare_text(value);
                };
                StreamSplitter sp;
                q27::Utf8Gate ugate;
                q27::ThinkBudgetState tb{think_budget};
                Engine::DecodeTask bt;
                if (tchoice.mode == q27::ToolChoice::FORCED) sp.chan = StreamSplitter::TOOL;
                else if (thinking) sp.chan = StreamSplitter::THINK; // prompt-injected opener
                ReasoningBudgetObserver budget{tok, tb, eng, bt, think_close_ids, sp.chan};
                budget.start();
                eng.on_round = [&](const int* em, int nr) {
                    return budget.observe_round(em, nr);
                };
                auto on_tok = [&](int id, bool forced) {
                    forced_control_token = forced;
                    for (auto& [ch, t] : sp.feed(ugate.feed(tok.decode_one(id)))) route(ch, t);
                    return alive; // stop generating once the client has disconnected
                };
                // TODO(batch error surfacing): no mid-stream error event is
                // emitted here -- codex-rs (v0.143) keys only off the item /
                // completed types this handler already sends and defines no
                // error shape we could mirror without inventing protocol;
                // end=error lands in the [req] line and [req-error] carries
                // the what().
                int produced = conductor
                                   ? batch_generate(eng, prompt, nm, on_tok,
                                                    nullptr, -1,
                                                    qw, rt, bt, nullptr)
                                   : eng.generate(prompt, nm, EOS, [&](int id) {
                                         return on_tok(id, bt.callback_forced);
                                     }, -1, &bt);
                eng.on_round = nullptr;
                eng.on_round_gap = nullptr;
                req_log(rt, qw, eng, sl.id, bat_stats(bt));
                const bool token_limit_reached=produced>=nm || bt.budget_truncated;
                for (auto& [ch, t] : sp.feed(ugate.flush())) route(ch, t);
                const bool unfinished_tool_wrapper=q27::unfinished_tool_wrapper(
                    produced,nm,bt.budget_truncated,sp.chan);
                for (auto& [ch, t] : sp.flush()) route(ch, t);
                if (!tool_buf.empty()) {
                    if (unfinished_tool_wrapper) {
                        append_text(tool_buf);
                        tool_buf.clear();
                        output_tool_tail=false;
                    } else flush_tool();
                }
                flush_think();
                if(!bare_probe.empty()) {
                    bare_input_final=true;
                    std::string final_probe=std::move(bare_probe);
                    bare_probe.clear();
                    route_bare_text(final_probe);
                    bare_input_final=false;
                }
                std::string pending;
                if(bare_holding) pending=bare_pending;
                else if(!bare_deferred.empty())
                    pending=bare_deferred+bare_deferred_trailing;
                std::string preview_prefix,preview_residual;
                const bool allow_repair=!token_limit_reached;
                const auto preview_calls=q27::parse_bare_tool_calls(pending, &preview_prefix, tools.empty()?nullptr:&tools, true, allow_repair, &preview_residual);
                const bool tool_tail=q27::responses_tool_tail_after_bare_calls(
                    output_tool_tail,pending,preview_calls,response_choice,
                    eligible_call_names,tool_counter);

                const bool limit_reached=token_limit_reached && !tool_tail;
                const auto terminal=q27::responses_terminal_state(limit_reached);
                finish_bare(allow_repair,terminal.incomplete);
                flush_text(terminal.incomplete);
                if (q27::forced_tool_choice_missing_is_error(
                        tchoice, tool_counter != 0, limit_reached)) {
                    json failed_response = {{"id", resp_id}, {"object", "response"},
                                            {"status", "failed"}, {"output", items},
                                            {"error", {{"code", "tool_choice_no_match"},
                                                       {"message", "model produced no eligible tool call for forced tool_choice"}}}};
                    ev({{"type", "response.failed"}, {"response", failed_response}});
                    sink.done();
                    return true;
                }

                json final_response = {{"id", resp_id}, {"object", "response"},
                                       {"status", terminal.status},
                                       {"output", items},
                                       {"usage", {{"input_tokens", (int)prompt.size()},
                                                  {"input_tokens_details", {{"cached_tokens", 0}}},
                                                  {"output_tokens", produced},
                                                  {"output_tokens_details", {{"reasoning_tokens", tb.used},
                                                                               {"reasoning_budget_exceeded", tb.tripped}}},
                                                  {"total_tokens", (int)prompt.size() + produced}}}};
                if (terminal.incomplete)
                    final_response["incomplete_details"] = {{"reason", "max_output_tokens"}};
                ev({{"type", terminal.event}, {"response", final_response}});
                sink.done();
                return true;
            });
    });

    srv.Post("/v1/chat/completions",
             [&](const httplib::Request& r, httplib::Response& s) { handle(r, s, true); });
    srv.Post("/v1/completions",
             [&](const httplib::Request& r, httplib::Response& s) { handle(r, s, false); });

    // SO_REUSEADDR only (2026-07-22, thunderdome postmortem pt.2): httplib's
    // Linux default socket option is SO_REUSEPORT, which lets a SECOND
    // q27-server co-bind the same port -- the kernel then load-balances
    // incoming connections across both, silently splitting an eval's traffic
    // between two servers (potentially two different models). REUSEADDR keeps
    // the fast rebind-after-TIME_WAIT behavior without permitting live
    // co-binding, so a second bind fails and hits the FATAL below.
    srv.set_socket_options([](socket_t sock) {
        int opt = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const void*>(&opt),
                   sizeof(opt));
    });
    // Bind FIRST, print "listening" only on success (2026-07-22, thunderdome
    // postmortem): the old unconditional print + listen() meant a bind failure
    // (port squatter) logged "listening" and then CLEAN-EXITED 0 -- downstream
    // clients saw ConnectionRefused mid-task and the failure masqueraded as
    // harness infra. Fail loudly instead.
    if (!srv.bind_to_port(host.c_str(), port)) {
        fprintf(stderr,
                "FATAL: cannot bind %s:%d (port already in use? see `ss -tlnp | grep %d`)\n",
                host.c_str(), port, port);
        if (conductor) conductor->request_stop();
        conductor.reset();
        return 1;
    }
    fprintf(stderr, "q27-server listening on http://%s:%d (ctx %d, %s head)\n", host.c_str(),
            port, ctx, fast ? "fast" : "faithful");
    srv.listen_after_bind();
    // P1 Task 10 shutdown: stop the conductor (its thread cancels + closes
    // any remaining members) and join it BEFORE the engines it drives tear
    // down with `slots` at scope exit.
    if (conductor) conductor->request_stop();
    conductor.reset();
    return 0;
}
