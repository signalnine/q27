// Integration-level compile+run check for the new /v1/chat/completions
// tool-calling code in server.cu. server.cu needs nvcc (CUDA kernels) to
// build, unavailable in this CPU-only review environment -- so this harness
// fakes ONLY the CUDA-touching surface (Engine, Slot, conductor, httplib)
// with the exact interface the new code calls, and otherwise uses the REAL
// api_common.h, toolgram.h, toolconstrain.h, stream_split.h, and
// tokenizer.{h,cpp} unmodified. The build_prompt/handle block below this
// preamble is a byte-for-byte extraction of src/server.cu's new code
// (see tools/extract_check.sh) -- this is a genuine compile+run check of
// the shipped logic, not a reimplementation of it.
//
// Build+run:
//   bash tools/build_chat_completions_integration.sh
#include "api_common.h"
#include "toolconstrain.h"
#include "tokenizer.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

using json = nlohmann::json;
using q27::Msg;
using q27::StreamSplitter;

namespace q27k {
struct SampleParams { float inv_temp = 0.f; float top_p = 1.f; unsigned long long seed = 0; };
}

// ---- fake Tokenizer ---------------------------------------------------
// decode_one() replays a pre-scripted piece-table. encode() normally hands
// back one synthetic id per call; the decoder-control close is also installed
// in the table so forced ids traverse the real StreamSplitter path.
struct FakeTok {
    std::vector<std::string> pieces;
    int eos_id = 0;
    int next_encode_id = 1000;
    std::string decode_one(int id) const {
        return (id >= 0 && (size_t)id < pieces.size()) ? pieces[(size_t)id] : std::string();
    }
    std::vector<int> encode(const std::string& text) {
        const int id = next_encode_id++;
        if (text == "</think>\n\n") {
            if (pieces.size() <= (size_t)id) pieces.resize((size_t)id + 1);
            pieces[(size_t)id] = text;
        }
        return {id};
    }
    int eos() const { return eos_id; }
    int token_id(const std::string&) const { return -1; }
    std::vector<std::string> vocab_bytes() const { return pieces; }
    // Only used by the UNCHANGED build_prompt() fallback path (non-routed_chat
    // requests); real behavior is irrelevant here, tests for that path only
    // check the resulting prompt-size bookkeeping via encode()'s id count.
    std::vector<int> apply_chat_template(const std::vector<std::pair<std::string, std::string>>&,
                                         bool) {
        return {next_encode_id++};
    }
};

// ---- fake Engine --------------------------------------------------------
struct FakeEngine {
    struct DecodeTask {
        int rounds = 0, bat_members = 0;
        long bat_r2 = 0;
        int emitted = 0;
        int n_max = 100000;
        int eos = -1;
        int Ph = 0;
        std::atomic<bool> cancel{false};
        bool sampling = false;
        bool public_budget_reduced = false;
        bool budget_cancelled = false;
        bool budget_truncated = false;
        bool round_forced = false;
        bool callback_forced = false;
        std::vector<int> forced;
        size_t forced_pos = 0;
        bool has_forced() const { return forced_pos < forced.size(); }
        int pop_forced() { return forced[forced_pos++]; }
    };
    q27k::SampleParams samp;
    int max_ctx = 100000;
    int pfx_sys_len = 0;
    int ctx_round_reserve() const { return 8; }
    std::function<bool()> on_round_gap;
    std::function<void(int)> on_pending;
    std::function<void(const int*)> on_drafts;
    std::function<int(const int*, int)> on_round;

    int mask_pool_used = 0;
    int mask_pool_cap = 512;
    int constraint_sets = 0;
    int forced_commit_count = 0;
    int mask_pool_add(const void*) { return mask_pool_used >= mask_pool_cap ? -1 : mask_pool_used++; }
    void set_tool_constraint(int id) { if (id >= 0) constraint_sets++; }
    void set_tool_masks5(const int[5]) {}

    bool reasoning_transition_active(const DecodeTask& task) const {
        return task.round_forced || task.has_forced();
    }
    bool reasoning_close_fits(const DecodeTask& task, int current_public_tokens,
                              int current_context_tokens, int close_tokens,
                              bool from_public_budget) const {
        if (current_public_tokens < 0 || current_context_tokens < 0 || close_tokens <= 0)
            return false;
        const int public_needed = from_public_budget ? close_tokens + 1 : 1;
        if (task.n_max - task.emitted - current_public_tokens < public_needed) return false;
        return task.Ph + current_context_tokens + close_tokens - 1 +
                   ctx_round_reserve() <=
               max_ctx;
    }
    bool force_reasoning_close(DecodeTask& task, const std::vector<int>& ids,
                               bool from_public_budget = false,
                               int current_public_tokens = 0,
                               int current_context_tokens = -1) {
        if (ids.empty()) return false;
        if (reasoning_transition_active(task)) return true;
        const int cost = (int)ids.size();
        if (current_context_tokens < 0) current_context_tokens = current_public_tokens;
        if (!reasoning_close_fits(task, current_public_tokens, current_context_tokens,
                                  cost, from_public_budget)) {
            task.budget_cancelled = true;
            task.cancel.store(true);
            return false;
        }
        if (from_public_budget) {
            task.public_budget_reduced = true;
            task.n_max -= cost;
        }
        task.forced.insert(task.forced.end(), ids.begin(), ids.end());
        return true;
    }

    std::vector<int> script; // token ids the test wants generated
    int round_width = 1;
    int conditioned_token = -1;
    int stale_token = -1;
    bool forced_committed = false;
    int max_sampled_round = 0;

    int resolve_token(int id) const {
        return id == conditioned_token && !forced_committed ? stale_token : id;
    }

    template <typename F>
    int generate(const std::vector<int>& prompt, int n_max, int eos, F&& on_token,
                 int = -1, DecodeTask* external_task = nullptr) {
        DecodeTask local;
        DecodeTask& task = external_task ? *external_task : local;
        task.n_max = n_max;
        task.eos = eos;
        task.Ph = (int)prompt.size() - 1;
        task.emitted = 0;
        task.sampling = samp.inv_temp > 0.f;
        forced_committed = false;
        forced_commit_count = 0;
        max_sampled_round = 0;
        int n = 0;
        auto emit_forced = [&]() {
            while (task.has_forced()) {
                int id = task.pop_forced();
                task.round_forced = true;
                if (on_round) (void)on_round(&id, 1);
                task.callback_forced = true;
                const bool keep_going = on_token(id);
                task.callback_forced = false;
                task.round_forced = false;
                forced_committed = true;
                forced_commit_count++;
                task.Ph++;
                if (!keep_going) return false;
            }
            return true;
        };
        if (!emit_forced()) return 0;
        for (size_t pos = 0; pos < script.size() && n < task.n_max;) {
            const int width = round_width;
            const int available = (int)script.size() - (int)pos;
            const int proposed = std::min(width, available);
            std::vector<int> ids;
            ids.reserve((size_t)proposed);
            for (int i = 0; i < proposed; i++) ids.push_back(resolve_token(script[pos + i]));
            int committed = proposed;
            if (on_round) {
                int m = on_round(ids.data(), proposed);
                if (m >= 1 && m <= proposed) committed = m;
            }
            if (task.sampling) max_sampled_round = std::max(max_sampled_round, committed);
            task.Ph += committed;
            bool stopped = false;
            for (int i = 0; i < committed && n < task.n_max; i++) {
                if (ids[(size_t)i] == task.eos) {
                    stopped = true;
                    break;
                }
                task.callback_forced = false;
                if (!on_token(ids[(size_t)i])) {
                    stopped = true;
                    break;
                }
                n++;
                task.emitted = n;
            }
            pos += (size_t)(task.sampling ? committed : proposed);
            if (stopped) break;
            if (!emit_forced() || task.cancel.load()) break;
        }
        task.emitted = n;
        task.budget_truncated = task.budget_cancelled ||
                                (task.public_budget_reduced && n >= task.n_max);
        return n;
    }
};
using Engine = FakeEngine;

struct Slot {
    std::unique_ptr<Engine> eng;
    int id = 0;
    bool busy = false;
    long last_used = 0;
    std::vector<int> tool_mask_host2dev;
};

using ToolConstrainer = q27::BasicToolConstrainer<Engine, FakeTok>;

struct HookGuard {
    Engine& e;
    ~HookGuard() { e.on_pending = nullptr; e.on_drafts = nullptr; e.on_round = nullptr; }
};

struct ReasoningBudgetObserver {
    FakeTok& tok;
    q27::ThinkBudgetState& state;
    Engine& eng;
    Engine::DecodeTask& task;
    const std::vector<int>& close_ids;
    StreamSplitter split;
    q27::Utf8Gate gate;

    ReasoningBudgetObserver(FakeTok& tok_, q27::ThinkBudgetState& state_,
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

// ---- fake httplib -------------------------------------------------------
namespace httplib {
struct Request { std::string body; };
struct DataSink {
    std::string* out;
    bool write(const char* d, size_t n) { out->append(d, n); return true; }
    bool done() { return true; }
};
struct Response {
    int status = 200;
    std::string content, content_type;
    std::function<bool(size_t, DataSink&)> provider;
    void set_content(const std::string& s, const std::string& ct) { content = s; content_type = ct; }
    void set_header(const char*, const char*) {}
    void set_chunked_content_provider(const char*, std::function<bool(size_t, DataSink&)> p) {
        provider = std::move(p);
    }
};
} // namespace httplib

static std::string jdump(const json& j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

// ---- global test scaffolding (populated fresh per test in main()) ------
json g_last_response;
std::vector<json> g_sse_events; // parsed "data: {...}" payloads, [DONE] excluded

struct PreparedAnthropicPrompt {
    std::string rendered;
    q27::ToolChoice tchoice;
    json tools;
    std::vector<std::string> tool_names;
    bool thinking=false;
    q27::ThinkCfg tcfg;
};

static PreparedAnthropicPrompt prepare_anthropic_prompt_for_test(
        const json& body, bool no_think_srv) {
    const bool req_think=true;
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
    PreparedAnthropicPrompt result;
    result.rendered=prepare_anthropic_prompt(
        body,result.tchoice,result.tools,result.tool_names,result.thinking,
        result.tcfg,nullptr,nullptr);
    return result;
}

static void run_request(FakeTok& tok, std::string served_name, bool no_think_srv,
                        bool constrain_tools, bool sampled_on, int max_prompt,
                        int max_slot_ctx, std::atomic<long>& req_counter,
                        q27::ToolMaskCache<q27::ToolGrammar>& tool_mask_cache, std::vector<Slot>& slots,
                        const json& body, bool chat, bool batch = false,
                        int budget_flag = 0) {
    const int EOS = tok.eos();
    std::mutex route_m;
    void* conductor = batch ? reinterpret_cast<void*>(1) : nullptr;
    q27::GpuGate gpu_gate; // real type (api_common.h, CUDA-free) -- Lease is RAII-only here
    const bool req_think = true;
    const int think_budget_flag = budget_flag;
    const std::vector<int> think_close_ids = tok.encode("</think>\n\n");
    struct DisabledPrefixCache { bool enabled() const { return false; } } pfx_cache;

    auto ms_since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t)
            .count();
    };
    auto conv_fp = [&](const json&) -> unsigned long long { return 0; };
    struct ReqTrace {
        long rid; const char* api; unsigned long long conv;
        std::chrono::steady_clock::time_point t0; double tok_ms;
    };
    auto claim_slot = [&](const std::vector<int>&, int, bool, const q27::ThinkCfg&, bool) -> Slot& {
        slots[0].busy = true;
        return slots[0];
    };
    auto slot_guard = [&](Slot& s) {
        return std::shared_ptr<Slot>(&s, [](Slot* p) { p->busy = false; });
    };
    auto make_yield = [&](Engine&) -> std::function<bool()> { return nullptr; };
    auto tg_stats = [&](const ToolConstrainer&) -> std::string { return ""; };
    auto bat_stats = [&](const FakeEngine::DecodeTask&) -> std::string { return ""; };
    auto req_log = [&](const ReqTrace&, double, const Engine&, int,
                       const std::string& = std::string()) {};
    auto parse_sample = [&](const json& b) -> q27k::SampleParams {
        q27k::SampleParams s;
        double temp = b.value("temperature", 0.0);
        if (temp > 0.0) s.inv_temp = (float)(1.0 / temp);
        return s;
    };
    auto batch_generate = [&](Engine& eng, const std::vector<int>&, int nm,
                              std::function<bool(int, bool)> on_token,
                              std::function<void(int)> on_emit, int /*stable_len*/, double&,
                              const ReqTrace&, FakeEngine::DecodeTask& t,
                              std::string*) -> int {
        int n = 0;
        t.n_max = nm;
        t.eos = tok.eos();
        t.emitted = 0;
        t.sampling = eng.samp.inv_temp > 0.f;
        eng.forced_committed = false;
        eng.forced_commit_count = 0;
        eng.max_sampled_round = 0;
        auto emit_forced = [&]() {
            while (t.has_forced()) {
                int id = t.pop_forced();
                t.round_forced = true;
                if (eng.on_round) (void)eng.on_round(&id, 1);
                t.callback_forced = true;
                if (on_emit) on_emit(id);
                const bool keep_going = on_token(id, true);
                t.callback_forced = false;
                t.round_forced = false;
                eng.forced_committed = true;
                eng.forced_commit_count++;
                if (!keep_going) return false;
            }
            return true;
        };
        if (!emit_forced()) return 0;
        for (size_t pos = 0; pos < eng.script.size() && n < t.n_max;) {
            const int width = eng.round_width;
            const int available = (int)eng.script.size() - (int)pos;
            const int proposed = std::min(width, available);
            std::vector<int> ids;
            ids.reserve((size_t)proposed);
            for (int i = 0; i < proposed; i++)
                ids.push_back(eng.resolve_token(eng.script[pos + (size_t)i]));
            int committed = proposed;
            if (eng.on_round) {
                int m = eng.on_round(ids.data(), proposed);
                if (m >= 1 && m <= proposed) committed = m;
            }
            if (t.sampling)
                eng.max_sampled_round = std::max(eng.max_sampled_round, committed);
            for (int i = 0; i < committed && n < t.n_max; i++) {
                if (ids[(size_t)i] == t.eos) {
                    pos = eng.script.size();
                    break;
                }
                t.callback_forced = false;
                if (on_emit) on_emit(ids[(size_t)i]);
                if (!on_token(ids[(size_t)i], false)) {
                    pos = eng.script.size();
                    break;
                }
                n++;
                t.emitted = n;
            }
            pos += (size_t)(t.sampling ? committed : proposed);
            if (!emit_forced() || t.cancel.load()) break;
        }
        t.emitted = n;
        t.budget_truncated = t.budget_cancelled ||
                             (t.public_budget_reduced && n >= t.n_max);
        return n;
    };

    httplib::Request req;
    req.body = body.dump();
    httplib::Response res;

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
                        "[tool-fallback] %zu drifted call(s) recovered (oai-nonstream)\n",
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
                                "[tool-fallback] %zu drifted call(s) recovered (oai-stream)\n",
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

    handle(req, res, chat);

    g_last_response = json::object();
    g_sse_events.clear();
    if (!res.content.empty()) {
        g_last_response = json::parse(res.content);
        g_last_response["__status"] = res.status;
    }
    if (res.provider) {
        std::string buf;
        httplib::DataSink sink{&buf};
        res.provider(0, sink);
        size_t pos = 0;
        while (true) {
            size_t start = buf.find("data: ", pos);
            if (start == std::string::npos) break;
            size_t end = buf.find("\n\n", start);
            if (end == std::string::npos) break;
            std::string payload = buf.substr(start + 6, end - start - 6);
            if (payload != "[DONE]") g_sse_events.push_back(json::parse(payload));
            pos = end + 2;
        }
    }
}

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
} while (0)

static std::vector<Slot> fresh_slots() {
    std::vector<Slot> slots;
    Slot s; s.eng = std::make_unique<FakeEngine>(); s.id = 0;
    slots.push_back(std::move(s));
    return slots;
}

static q27::ToolMaskCache<q27::ToolGrammar> fresh_cache(std::vector<std::string>& vocab_bytes) {
    q27::ToolMaskCache<q27::ToolGrammar> c;
    c.init(&vocab_bytes, -1); // </tool_call> id unused (constrain_tools off in these tests)
    return c;
}

int main() {
    const auto responses_completed=q27::responses_terminal_state(1,2,false);
    CHECK(!responses_completed.incomplete &&
          std::string(responses_completed.status)=="completed" &&
          std::string(responses_completed.event)=="response.completed");
    const auto responses_exact_stop=q27::responses_terminal_state(false);
    CHECK(!responses_exact_stop.incomplete &&
          std::string(responses_exact_stop.status)=="completed" &&
          std::string(responses_exact_stop.event)=="response.completed");
    const auto responses_limited=q27::responses_terminal_state(2,2,false);
    CHECK(responses_limited.incomplete &&
          std::string(responses_limited.status)=="incomplete" &&
          std::string(responses_limited.event)=="response.incomplete");
    const auto responses_budget_limited=q27::responses_terminal_state(1,2,true);
    CHECK(responses_budget_limited.incomplete &&
          std::string(responses_budget_limited.status)=="incomplete" &&
          std::string(responses_budget_limited.event)=="response.incomplete");
    CHECK(q27::unfinished_tool_wrapper(
        4,4,false,q27::StreamSplitter::TOOL));
    CHECK(q27::unfinished_tool_wrapper(
        3,4,true,q27::StreamSplitter::TOOL));
    CHECK(!q27::unfinished_tool_wrapper(
        4,4,false,q27::StreamSplitter::TEXT));
    CHECK(!q27::unfinished_tool_wrapper(
        3,4,false,q27::StreamSplitter::TOOL));
    std::string responses_text="\nanswer ";
    const std::string responses_done=q27::take_responses_output_text(responses_text);
    CHECK(responses_done=="\nanswer " && responses_text.empty());

    // ---- Test 1: plain chat, no tools -> plain content, no tool_calls ----
    {
        FakeTok tok;
        tok.pieces = {"<eos>", "Hello", ", ", "world!"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3};
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role","user"},{"content","hi"}}})}};
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, true);
        CHECK(g_last_response["choices"][0]["message"]["content"] == "Hello, world!");
        CHECK(!g_last_response["choices"][0]["message"].contains("tool_calls"));
        CHECK(g_last_response["choices"][0]["finish_reason"] == "stop");
        CHECK(g_last_response["object"] == "chat.completion");
    }

    // ---- Test 2: tool call, non-stream ----
    // model emits: "Sure, checking." <tool_call>\n{"name": "get_weather",
    // "arguments": {"location": "Tokyo"}}\n</tool_call>
    {
        FakeTok tok;
        tok.pieces = {
            /*0*/ "<eos>",
            /*1*/ "Sure, checking.",
            /*2*/ "<tool_call>\n",
            /*3*/ "{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}\n",
            /*4*/ "</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3, 4};
        std::atomic<long> rc{0};
        json body = {
            {"tools", json::array({
                {{"type","function"},{"function",{{"name","get_weather"},
                    {"description","w"},{"parameters", json::object()}}}}
            })},
            {"messages", json::array({{{"role","user"},{"content","weather in tokyo?"}}})},
        };
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, true);
        auto& msg = g_last_response["choices"][0]["message"];
        CHECK(g_last_response["choices"][0]["finish_reason"] == "tool_calls");
        CHECK(msg["content"] == "Sure, checking.");
        CHECK(msg["tool_calls"].size() == 1);
        CHECK(msg["tool_calls"][0]["type"] == "function");
        CHECK(msg["tool_calls"][0]["function"]["name"] == "get_weather");
        json args = json::parse(msg["tool_calls"][0]["function"]["arguments"].get<std::string>());
        CHECK(args["location"] == "Tokyo");
    }

    // ---- Test 3: thinking + tool call, non-stream: <think> must never leak
    // into `content` verbatim (the original leak this path fixes), and must
    // not break tool-call parsing. Post-Patch-3, the think trace is
    // deliberately surfaced via `reasoning_content` (not silently dropped),
    // so it must land there specifically. ----
    {
        FakeTok tok;
        tok.pieces = {
            /*0*/ "<eos>",
            /*1*/ "<think>",
            /*2*/ "reasoning about tokyo weather",
            /*3*/ "</think>",
            /*4*/ "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}\n</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {2, 3, 4}; // prompt already contains <think>
        std::atomic<long> rc{0};
        json body = {
            {"tools", json::array({
                {{"type","function"},{"function",{{"name","get_weather"},{"description","w"},
                    {"parameters", json::object()}}}}
            })},
            {"messages", json::array({{{"role","user"},{"content","weather in tokyo?"}}})},
            {"enable_thinking", true},
        };
        run_request(tok, "q27-test", /*no_think_srv=*/false, false, true, 100000, 100000, rc,
                   cache, slots, body, true);
        auto& msg = g_last_response["choices"][0]["message"];
        CHECK(msg["content"].is_null()); // no leftover visible text, only the call
        CHECK(msg["tool_calls"].size() == 1);
        CHECK(msg["tool_calls"][0]["function"]["name"] == "get_weather");
        // the think trace must land in reasoning_content specifically...
        CHECK(msg["reasoning_content"] == "reasoning about tokyo weather");
        // ...and NOWHERE else (content, tool_calls) -- dump minus the
        // reasoning_content field's own value must not contain the phrase.
        json msg_no_reasoning = msg;
        msg_no_reasoning.erase("reasoning_content");
        CHECK(msg_no_reasoning.dump().find("reasoning about tokyo weather") == std::string::npos);
    }

    // ---- Test 4: multi-turn history round-trip -- a PRIOR assistant
    // tool_calls[] + a role:"tool" result message must both survive into the
    // model-visible prompt structure. We can't inspect the rendered prompt
    // string directly through this harness (FakeTok::encode discards it by
    // design), so this test instead exercises openai_msgs directly (already
    // covered end-to-end in test_openai_bridge.cpp) and confirms handle()
    // does not crash/error on a body containing that history shape end-to-end.
    {
        FakeTok tok;
        tok.pieces = {"<eos>", "72F and sunny tomorrow too."};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1};
        std::atomic<long> rc{0};
        json body = {
            {"messages", json::array({
                {{"role","user"},{"content","weather in Tokyo?"}},
                {{"role","assistant"},{"content", nullptr},
                 {"tool_calls", json::array({
                     {{"id","call_1"},{"type","function"},
                      {"function",{{"name","get_weather"},{"arguments","{\"location\":\"Tokyo\"}"}}}}
                 })}},
                {{"role","tool"},{"tool_call_id","call_1"},{"content","72F and sunny"}},
                {{"role","user"},{"content","and tomorrow?"}},
            })},
        };
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, true);
        CHECK(g_last_response["__status"] == 200);
        CHECK(g_last_response["choices"][0]["message"]["content"] == "72F and sunny tomorrow too.");
    }

    // ---- Test 5: tool_choice: "none" -> tools stripped, model can't call,
    // even if it emits <tool_call> markers they parse as a call but the
    // request must still render (tools array empty means chatml_prompt gets
    // no tools_preamble; we verify via the empty-tools bare-recovery gate:
    // parse_bare_tool_calls only runs when `tools` is non-empty in this path,
    // matching the /v1/messages precedent) ----
    {
        FakeTok tok;
        tok.pieces = {"<eos>", "plain answer, no tools mentioned"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1};
        std::atomic<long> rc{0};
        json body = {
            {"tool_choice", "none"},
            {"tools", json::array({
                {{"type","function"},{"function",{{"name","get_weather"},{"description","w"},
                    {"parameters", json::object()}}}}
            })},
            {"messages", json::array({{{"role","user"},{"content","weather in tokyo?"}}})},
        };
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, true);
        auto& msg = g_last_response["choices"][0]["message"];
        CHECK(msg["content"] == "plain answer, no tools mentioned");
        CHECK(!msg.contains("tool_calls"));
    }

    // ---- Test 6: tool_choice FORCED (named) -> the "<tool_call>\n" opener
    // is injected into the PROMPT (invisible to this harness's FakeTok, which
    // discards prompt text) and the stream splitter is pre-seeded into TOOL
    // channel, so generated text -- with NO literal opening marker this time
    // and a zero think budget -- must still parse as a tool call without an
    // injected </think>. ----
    {
        FakeTok tok;
        tok.pieces = {
            /*0*/ "<eos>",
            /*1*/ "{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n",
            /*2*/ "</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2}; // NOTE: no <tool_call> opener token at all
        std::atomic<long> rc{0};
        json body = {
            {"tool_choice", {{"type","function"},{"function",{{"name","get_weather"}}}}},
            {"thinking_token_budget", 0},
            {"tools", json::array({
                {{"type","function"},{"function",{{"name","get_weather"},{"description","w"},
                    {"parameters", json::object()}}}}
            })},
            {"messages", json::array({{{"role","user"},{"content","weather in paris?"}}})},
        };
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, true);
        auto& msg = g_last_response["choices"][0]["message"];
        CHECK(g_last_response["choices"][0]["finish_reason"] == "tool_calls");
        CHECK(msg["tool_calls"].size() == 1);
        CHECK(msg["tool_calls"][0]["function"]["name"] == "get_weather");
        json args = json::parse(msg["tool_calls"][0]["function"]["arguments"].get<std::string>());
        CHECK(args["location"] == "Paris");
    }

    // ---- Test 6b: forced-tool requests start in TOOL, so they must not lose
    // output capacity to a reasoning close that cannot occur. Exercise both
    // handler twins at the exact four-token boundary. ----
    for(bool stream:{false,true}) {
        FakeTok tok;
        tok.pieces = {"<eos>", "{\"name\":\"get_", "weather\",\"arguments\":",
                      "{}}", "</tool_call>"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->max_ctx = 13; // P=2, round reserve=8, four public tokens fit
        slots[0].eng->script = {1,2,3,4};
        std::atomic<long> rc{0};
        json body = {
            {"tool_choice", {{"type","function"},{"function",{{"name","get_weather"}}}}},
            {"tools", json::array({
                {{"type","function"},{"function",{{"name","get_weather"},
                    {"parameters",json::object()}}}}
            })},
            {"messages", json::array({{{"role","user"},{"content","weather?"}}})},
            {"max_tokens",4}, {"stream",stream},
        };
        run_request(tok,"q27-test",false,false,true,100,13,rc,cache,slots,
                    body,true,false,-1);
        if(stream) {
            bool saw_call=false;
            for(const auto& event:g_sse_events)
                if(event.contains("choices") && !event["choices"].empty() &&
                   event["choices"][0]["delta"].contains("tool_calls")) saw_call=true;
            CHECK(saw_call);
        } else {
            CHECK(g_last_response["choices"][0]["message"]["tool_calls"].size()==1);
        }
    }

    // ---- Test 8: /v1/completions (chat=false) with a raw "prompt" field
    // must take the ORIGINAL text-only path untouched (no routed_chat). ----
    {
        FakeTok tok;
        tok.pieces = {"<eos>", "plain completion text"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1};
        std::atomic<long> rc{0};
        json body = {{"prompt", "continue: "}};
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, /*chat=*/false);
        CHECK(g_last_response["object"] == "text_completion");
        CHECK(g_last_response["choices"][0]["text"] == "plain completion text");
        CHECK(!g_last_response["choices"][0].contains("message"));
    }

    // ---- Test 8: text-only completion admission does not reserve think-close
    // capacity even when the server default budget is enabled. ----
    {
        FakeTok tok;
        tok.pieces = {"<eos>", "x"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->max_ctx = 9; // P=1, round reserve=7, one output token fits
        slots[0].eng->script = {1};
        std::atomic<long> rc{0};
        json body = {{"prompt", "raw"}, {"max_tokens", 1}};
        run_request(tok, "q27-test", true, false, true, 100000, 9, rc, cache, slots,
                    body, false, false, -1);
        CHECK(g_last_response["__status"] == 200);
        CHECK(g_last_response["choices"][0]["text"] == "x");
    }

    // ---- Test 8: streaming, tool call -- verify SSE delta shapes ----
    {
        FakeTok tok;
        tok.pieces = {
            /*0*/ "<eos>",

            /*1*/ "Checking now.",
            /*2*/ "<tool_call>\n",
            /*3*/ "{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Rome\"}}\n",
            /*4*/ "</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3, 4};
        std::atomic<long> rc{0};
        json body = {
            {"stream", true},
            {"tools", json::array({
                {{"type","function"},{"function",{{"name","get_weather"},{"description","w"},
                    {"parameters", json::object()}}}}
            })},
            {"messages", json::array({{{"role","user"},{"content","weather in rome?"}}})},
        };
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, true);
        bool saw_content = false, saw_tool_call = false, saw_finish = false;
        for (auto& ev : g_sse_events) {
            auto& delta = ev["choices"][0]["delta"];
            if (delta.contains("content") && delta["content"] == "Checking now.") saw_content = true;
            if (delta.contains("tool_calls")) {
                saw_tool_call = true;
                CHECK(delta["tool_calls"][0]["index"] == 0);
                CHECK(delta["tool_calls"][0]["function"]["name"] == "get_weather");
                json args = json::parse(
                    delta["tool_calls"][0]["function"]["arguments"].get<std::string>());
                CHECK(args["location"] == "Rome");
            }
            if (ev["choices"][0]["finish_reason"] == "tool_calls") saw_finish = true;
        }
        CHECK(saw_content);
        CHECK(saw_tool_call);
        CHECK(saw_finish);
    }

    // ---- Test 8b: streaming with thinking enabled -- reasoning_content
    // deltas must appear, must carry the think text, and must never appear
    // under `content` instead. ----
    {
        FakeTok tok;
        tok.pieces = {
            /*0*/ "<eos>",
            /*1*/ "<think>",
            /*2*/ "pondering rome weather",
            /*3*/ "</think>",
            /*4*/ "Sunny in Rome.",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {2, 3, 4}; // prompt already contains <think>
        std::atomic<long> rc{0};
        json body = {
            {"stream", true},
            {"enable_thinking", true},
            {"messages", json::array({{{"role","user"},{"content","weather in rome?"}}})},
        };
        run_request(tok, "q27-test", /*no_think_srv=*/false, false, true, 100000, 100000, rc,
                   cache, slots, body, true);
        bool saw_reasoning = false, saw_content = false;
        for (auto& ev : g_sse_events) {
            auto& delta = ev["choices"][0]["delta"];
            if (delta.contains("reasoning_content")) {
                saw_reasoning = true;
                CHECK(delta["reasoning_content"] == "pondering rome weather");
                CHECK(!delta.contains("content"));
            }
            if (delta.contains("content")) {
                saw_content = true;
                CHECK(delta["content"] == "Sunny in Rome.");
                CHECK(!delta.contains("reasoning_content"));
            }
        }
        CHECK(saw_reasoning);
        CHECK(saw_content);
    }

    // ---- Test 8c: a request reasoning budget closes THINK at the token
    // boundary, routes subsequent output to content, and reports the trip. ----
    {
        FakeTok tok;
        tok.pieces = {
            /*0*/ "<eos>",
            /*1*/ "first thought",
            /*2*/ " then answer",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2}; // prompt already injected <think>
        std::atomic<long> rc{0};
        json body = {
            {"enable_thinking", true},
            {"thinking_token_budget", 1},
            {"max_tokens", 4},
            {"messages", json::array({{{"role","user"},{"content","answer"}}})},
        };
        run_request(tok, "q27-test", /*no_think_srv=*/false, false, true,
                    100000, 100000, rc, cache, slots, body, true);
        const auto& msg = g_last_response["choices"][0]["message"];
        CHECK(msg["reasoning_content"] == "first thought");
        CHECK(msg["content"] == " then answer");
        CHECK(g_last_response["usage"]["reasoning_tokens"] == 1);
        CHECK(g_last_response["usage"]["reasoning_budget_exceeded"] == true);
    }

    // ---- Test 9: malformed tool call body -> surfaced as content, not
    // silently dropped, finish_reason falls back to stop/length. ----
    {
        FakeTok tok;
        tok.pieces = {
            /*0*/ "<eos>",
            /*1*/ "<tool_call>\n",
            /*2*/ "not valid json at all",
            /*3*/ "</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3};
        std::atomic<long> rc{0};
        json body = {
            {"tools", json::array({
                {{"type","function"},{"function",{{"name","get_weather"},{"description","w"},
                    {"parameters", json::object()}}}}
            })},
            {"messages", json::array({{{"role","user"},{"content","weather?"}}})},
        };
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache, slots,
                   body, true);
        auto& msg = g_last_response["choices"][0]["message"];
        CHECK(!msg.contains("tool_calls"));
        CHECK(msg["content"].get<std::string>().find("not valid json at all") != std::string::npos);
        CHECK(g_last_response["choices"][0]["finish_reason"] == "stop");
    }

    // ---- Test 10: a cap trip is a decoder-state transition. The second
    // model token is deliberately conditioned on whether the forced close was
    // committed; host-only queueing would expose the stale reasoning successor. ----
    {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "still reasoning",
                      /*2*/ "Conditioned answer.", /*3*/ "STALE successor"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2};
        slots[0].eng->conditioned_token = 2;
        slots[0].eng->stale_token = 3;
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "finish"}}})},
                     {"enable_thinking", true}, {"thinking_token_budget", 1},
                     {"max_tokens", 2}};
        run_request(tok, "q27-test", false, false, true, 100000, 100000, rc, cache, slots,
                    body, true);
        const auto& msg = g_last_response["choices"][0]["message"];
        CHECK(msg["reasoning_content"] == "still reasoning");
        CHECK(msg["content"] == "Conditioned answer.");
        CHECK(msg["content"].get<std::string>().find("STALE") == std::string::npos);
        CHECK(g_last_response["usage"]["completion_tokens"] == 2);
        CHECK(g_last_response["usage"]["reasoning_tokens"] == 1);
        CHECK(g_last_response["usage"]["reasoning_budget_exceeded"] == true);
    }

    // ---- Test 11: zero budget closes before the first model token. ----
    {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "Immediate answer."};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1};
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "answer"}}})},
                     {"enable_thinking", true}, {"thinking_token_budget", 0},
                     {"max_tokens", 1}};
        run_request(tok, "q27-test", false, false, true, 100000, 100000, rc, cache, slots,
                    body, true);
        const auto& msg = g_last_response["choices"][0]["message"];
        CHECK(msg["content"] == "Immediate answer.");
        CHECK(!msg.contains("reasoning_content"));
        CHECK(g_last_response["usage"]["completion_tokens"] == 1);
        CHECK(g_last_response["usage"]["reasoning_tokens"] == 0);
        CHECK(g_last_response["usage"]["reasoning_budget_exceeded"] == true);
    }

    // ---- Test 12: a natural close later in the same fused round wins over a
    // crossed cap and is not followed by a duplicate injected close. ----
    {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "short thought", /*2*/ "</think>",
                      /*3*/ "Natural answer."};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3};
        slots[0].eng->round_width = 2;
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "answer"}}})},
                     {"enable_thinking", true}, {"thinking_token_budget", 1},
                     {"max_tokens", 3}};
        run_request(tok, "q27-test", false, false, true, 100000, 100000, rc, cache, slots,
                    body, true);
        const auto& msg = g_last_response["choices"][0]["message"];
        CHECK(msg["reasoning_content"] == "short thought");
        CHECK(msg["content"] == "Natural answer.");
        CHECK(g_last_response["usage"]["completion_tokens"] == 3);
        CHECK(g_last_response["usage"]["reasoning_tokens"] == 2);
        CHECK(g_last_response["usage"]["reasoning_budget_exceeded"] == false);
    }

    // ---- Test 12s: a sampled natural close plus answer may fill the accepted
    // round and public cap exactly. Preserve ordinary output and tails buffered
    // by the marker/UTF-8 gates instead of replacing them with a forced close. ----
    const std::vector<std::pair<std::string, std::string>> sampled_answers = {
        {"Natural sampled answer.", "Natural sampled answer."},
        {"<", "<"},
        {std::string("\xE2"), std::string("\xEF\xBF\xBD")},
    };
    for (bool batch : {false, true}) {
        for (const auto& [answer_piece, expected_answer] : sampled_answers) {
            FakeTok tok;
            tok.pieces = {/*0*/ "<eos>", /*1*/ "short thought",
                          /*2*/ "</think>\n\n", /*3*/ answer_piece};
            std::vector<std::string> vb = tok.pieces;
            auto cache = fresh_cache(vb);
            auto slots = fresh_slots();
            slots[0].eng->script = {1, 2, 3};
            slots[0].eng->round_width = 3;
            std::atomic<long> rc{0};
            json body = {{"messages", json::array({{{"role", "user"}, {"content", "answer"}}})},
                         {"enable_thinking", true}, {"thinking_token_budget", 1},
                         {"temperature", 0.7}, {"max_tokens", 3}};
            run_request(tok, "q27-test", false, false, true, 100000, 100000, rc,
                        cache, slots, body, true, batch);
            const auto& msg = g_last_response["choices"][0]["message"];
            CHECK(msg["reasoning_content"] == "short thought");
            CHECK(msg["content"] == "\n\n" + expected_answer);
            CHECK(slots[0].eng->forced_commit_count == 0);
            CHECK(slots[0].eng->max_sampled_round == 3);
            CHECK(g_last_response["usage"]["completion_tokens"] == 3);
            CHECK(g_last_response["usage"]["reasoning_tokens"] == 2);
            CHECK(g_last_response["usage"]["reasoning_budget_exceeded"] == false);
        }
    }

    // ---- Test 13: greedy fused decode may overshoot by the retained round,
    // then the conductor commits the forced close before the next proposal. ----
    {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "reason A", /*2*/ "reason B",
                      /*3*/ "Process complete."};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3};
        slots[0].eng->round_width = 2;
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "finish"}}})},
                     {"enable_thinking", true}, {"thinking_token_budget", 1},
                     {"max_tokens", 3}};
        run_request(tok, "q27-test", false, false, true, 100000, 100000, rc, cache, slots,
                    body, true, /*batch=*/true);
        const auto& msg = g_last_response["choices"][0]["message"];
        CHECK(msg["reasoning_content"] == "reason Areason B");
        CHECK(msg["content"] == "Process complete.");
        CHECK(msg["content"].get<std::string>().find("</think>") == std::string::npos);
        CHECK(g_last_response["usage"]["completion_tokens"] == 3);
        CHECK(g_last_response["usage"]["reasoning_tokens"] == 2);
        CHECK(g_last_response["usage"]["reasoning_budget_exceeded"] == true);
    }

    // ---- Test 13a: an overshooting greedy round is rewound only when keeping
    // it would consume the public answer reserved behind the forced close. ----
    {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "reason A", /*2*/ "discarded tail",
                      /*3*/ "Capacity-safe answer."};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3};
        slots[0].eng->round_width = 2;
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "finish"}}})},
                     {"enable_thinking", true}, {"thinking_token_budget", 1},
                     {"max_tokens", 2}};
        run_request(tok, "q27-test", false, false, true, 100000, 100000, rc, cache, slots,
                    body, true);
        const auto& msg = g_last_response["choices"][0]["message"];
        CHECK(msg["reasoning_content"] == "reason A");
        CHECK(msg["content"] == "Capacity-safe answer.");
        CHECK(g_last_response["usage"]["completion_tokens"] == 2);
        CHECK(g_last_response["usage"]["reasoning_tokens"] == 1);
        CHECK(g_last_response["usage"]["reasoning_budget_exceeded"] == true);
    }

    // ---- Test 13b: EOS wins over a same-round cap crossing. Tokens at and
    // after EOS are neither emitted nor observed as reasoning, in solo or the
    // widened sampled conductor fake, and no close is queued. ----
    for (bool batch : {false, true}) {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "final thought", /*2*/ "unreachable"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 0, 2};
        slots[0].eng->round_width = 3;
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "finish"}}})},
                     {"enable_thinking", true}, {"thinking_token_budget", 1},
                     {"temperature", 0.7}, {"max_tokens", 3}};
        run_request(tok, "q27-test", false, false, true, 100000, 100000, rc, cache, slots,
                    body, true, batch);
        CHECK(slots[0].eng->forced_commit_count == 0);
        CHECK(g_last_response["usage"]["completion_tokens"] == 1);
        CHECK(g_last_response["usage"]["reasoning_tokens"] == 1);
        CHECK(g_last_response["usage"]["reasoning_budget_exceeded"] == false);
    }

    // ---- Test 13c: a later think re-entry accounts for the current retained
    // round before borrowing close capacity from the public allowance. ----
    {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "<think>", /*2*/ "stale reason A",
                      /*3*/ "stale reason B", /*4*/ "Re-entry answer."};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3, 4};
        slots[0].eng->round_width = 3;
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "finish"}}})},
                     {"enable_thinking", true}, {"thinking_token_budget", 0},
                     {"max_tokens", 3}};
        run_request(tok, "q27-test", false, false, true, 100000, 100000, rc, cache, slots,
                    body, true);
        const auto& msg = g_last_response["choices"][0]["message"];
        CHECK(msg["content"] == "Re-entry answer.");
        CHECK(msg["content"].get<std::string>().find("stale") == std::string::npos);
        CHECK(g_last_response["usage"]["completion_tokens"] == 2);
        CHECK(g_last_response["usage"]["reasoning_tokens"] == 0);
        CHECK(g_last_response["usage"]["reasoning_budget_exceeded"] == true);
    }

    // ---- Test 13d: a budget rewind is planned before the stateful tool scan,
    // so a discarded same-round <tool_call> marker cannot stage a phantom mask. ----
    {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "reason A",
                      /*2*/ "</think><tool_call>", /*3*/ "Tool-safe answer."};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3};
        slots[0].eng->round_width = 2;
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "finish"}}})},
                     {"enable_thinking", true}, {"thinking_token_budget", 1},
                     {"max_tokens", 2},
                     {"tools", json::array({{{"type", "function"},
                         {"function", {{"name", "get_weather"}, {"description", "w"},
                                       {"parameters", json::object()}}}}})}};
        run_request(tok, "q27-test", false, false, true, 100000, 100000, rc, cache, slots,
                    body, true);
        const auto& msg = g_last_response["choices"][0]["message"];
        CHECK(msg["content"] == "Tool-safe answer.");
        CHECK(slots[0].eng->constraint_sets == 0);
    }

    // ---- Test 13e: sampled zero-budget close commits before any public
    // reasoning token. A later two-token re-entry round has no public capacity
    // for another close plus answer, so it retains only the opener and cancels
    // without exposing or consuming the discarded sampled pending. ----
    for (bool batch : {false, true}) {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "<think>", /*2*/ "unreachable answer"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2};
        slots[0].eng->round_width = 2;
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "finish"}}})},
                     {"enable_thinking", true}, {"thinking_token_budget", 0},
                     {"temperature", 0.7}, {"max_tokens", 2}};
        run_request(tok, "q27-test", false, false, true, 100000, 100000, rc, cache, slots,
                    body, true, batch);
        CHECK(slots[0].eng->forced_commit_count == 1);
        CHECK(slots[0].eng->max_sampled_round == 1);
        CHECK(g_last_response["usage"]["completion_tokens"] == 1);
        CHECK(g_last_response["usage"]["reasoning_budget_exceeded"] == true);
    }
    // ---- Test 13b: a sampled boundary-crossing round retains only the budget
    // prefix, then the forced close conditions the next draw in solo and batch. ----
    for (bool batch : {false, true}) {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "sampled reasoning",
                      /*2*/ "Sampled answer.", /*3*/ "STALE sampled successor"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2};
        slots[0].eng->round_width = 4;
        slots[0].eng->conditioned_token = 2;
        slots[0].eng->stale_token = 3;
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "finish"}}})},
                     {"enable_thinking", true}, {"thinking_token_budget", 1},
                     {"temperature", 0.7}, {"max_tokens", 2}};
        run_request(tok, "q27-test", false, false, true, 100000, 100000, rc, cache, slots,
                    body, true, batch);
        const auto& msg = g_last_response["choices"][0]["message"];
        CHECK(msg["reasoning_content"] == "sampled reasoning");
        CHECK(msg["content"] == "Sampled answer.");
        CHECK(msg["content"].get<std::string>().find("STALE") == std::string::npos);
        CHECK(g_last_response["usage"]["reasoning_tokens"] == 1);
    }

    // ---- Test 13c: bounded sampling keeps speculative width until the round
    // that crosses the budget, then truncates only that round's stale suffix. ----
    for (bool batch : {false, true}) {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "r1 ", /*2*/ "r2 ", /*3*/ "r3 ",
                      /*4*/ "r4 ", /*5*/ "Sampled answer.",
                      /*6*/ "STALE sampled successor"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3, 4, 5};
        slots[0].eng->round_width = 3;
        slots[0].eng->conditioned_token = 5;
        slots[0].eng->stale_token = 6;
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "finish"}}})},
                     {"enable_thinking", true}, {"thinking_token_budget", 4},
                     {"temperature", 0.7}, {"max_tokens", 5}};
        run_request(tok, "q27-test", false, false, true, 100000, 100000, rc, cache, slots,
                    body, true, batch);
        const auto& msg = g_last_response["choices"][0]["message"];
        CHECK(msg["reasoning_content"] == "r1 r2 r3 r4 ");
        CHECK(msg["content"] == "Sampled answer.");
        CHECK(msg["content"].get<std::string>().find("STALE") == std::string::npos);
        CHECK(g_last_response["usage"]["reasoning_tokens"] == 4);
        CHECK(g_last_response["usage"]["reasoning_budget_exceeded"] == true);
        CHECK(slots[0].eng->max_sampled_round == 3);
    }

    // ---- Test 14: injected template newlines never surface as an OpenAI
    // content delta; only the model-generated answer is public. ----
    {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "Immediate answer."};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1};
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "answer"}}})},
                     {"enable_thinking", true}, {"thinking_token_budget", 0},
                     {"max_tokens", 1}, {"stream", true}};
        run_request(tok, "q27-test", false, false, true, 100000, 100000, rc, cache, slots,
                    body, true);
        bool saw_answer = false, saw_control_ws = false;
        for (const auto& ev : g_sse_events) {
            if (!ev.contains("choices") || ev["choices"].empty()) continue;
            const auto& delta = ev["choices"][0]["delta"];
            if (!delta.contains("content") || !delta["content"].is_string()) continue;
            const std::string content = delta["content"].get<std::string>();
            if (!content.empty() && q27::strip_ws2(content).empty()) saw_control_ws = true;
            if (content == "Immediate answer.") saw_answer = true;
        }
        CHECK(saw_answer);
        CHECK(!saw_control_ws);
    }

    // ---- Test 15: bounded thinking near the context boundary is rejected
    // before generation instead of returning an empty successful response. ----
    {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "unused"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1};
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "answer"}}})},
                     {"enable_thinking", true}, {"thinking_token_budget", 0},
                     {"max_tokens", 8}};
        run_request(tok, "q27-test", false, false, true, 2, 10, rc, cache, slots,
                    body, true);
        CHECK(g_last_response["__status"] == 400);
        CHECK(g_last_response["error"]["code"] == "context_length_exceeded");
    }

    // ---- Test 16: ordinary overflow reports the real prompt ceiling; the
    // think-close reserve applies only to bounded-thinking rejection. ----
    {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "unused"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "answer"}}})},
                     {"enable_thinking", false}, {"max_tokens", 8}};
        run_request(tok, "q27-test", false, false, true, 1, 9, rc, cache, slots,
                    body, true);
        CHECK(g_last_response["__status"] == 400);
        CHECK(g_last_response["error"]["message"].get<std::string>().find(
                  "1 maximum") != std::string::npos);
    }

    // ---- Test 17: stream and non-stream both preserve model-generated leading
    // whitespace; only exact injected close-token whitespace is suppressed. ----
    for (bool stream : {false, true}) {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "\n", /*2*/ "answer"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2};
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "answer"}}})},
                     {"enable_thinking", false}, {"max_tokens", 2}, {"stream", stream}};
        run_request(tok, "q27-test", false, false, true, 100000, 100000, rc, cache, slots,
                    body, true);
        std::string content;
        if (stream) {
            for (const auto& ev : g_sse_events) {
                if (!ev.contains("choices") || ev["choices"].empty()) continue;
                const auto& delta = ev["choices"][0]["delta"];
                if (delta.contains("content") && delta["content"].is_string())
                    content += delta["content"].get<std::string>();
            }
        } else {
            content = g_last_response["choices"][0]["message"]["content"].get<std::string>();
        }
        CHECK(content == "\nanswer");
    }

    // ---- Test 18: a model-generated natural close keeps the same answer bytes
    // in stream and non-stream, even when it matches the template close. ----
    for (bool stream : {false, true}) {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ "</think>\n\n", /*2*/ "answer"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2};
        std::atomic<long> rc{0};
        json body = {{"messages", json::array({{{"role", "user"}, {"content", "answer"}}})},
                     {"enable_thinking", true}, {"thinking_token_budget", -1},
                     {"max_tokens", 2}, {"stream", stream}};
        run_request(tok, "q27-test", false, false, true, 100000, 100000, rc, cache, slots,
                    body, true);
        std::string content;
        if (stream) {
            for (const auto& ev : g_sse_events) {
                if (!ev.contains("choices") || ev["choices"].empty()) continue;
                const auto& delta = ev["choices"][0]["delta"];
                if (delta.contains("content") && delta["content"].is_string())
                    content += delta["content"].get<std::string>();
            }
        } else {
            content = g_last_response["choices"][0]["message"]["content"].get<std::string>();
        }
        CHECK(content == "\n\nanswer");
    }

    // ---- Test 19: a later exhausted think span suppresses its injected
    // whitespace and reports length only when the reduced cap is reached. ----
    for (bool batch : {false, true}) {
        for (bool stream : {false, true}) {
            for (int cap : {4, 8}) {
                FakeTok tok;
                tok.pieces = {/*0*/ "<eos>", /*1*/ "prefix", /*2*/ "<think>",
                              /*3*/ "suffix"};
                std::vector<std::string> vb = tok.pieces;
                auto cache = fresh_cache(vb);
                auto slots = fresh_slots();
                slots[0].eng->script = {1, 2, 3};
                std::atomic<long> rc{0};
                json body = {{"messages", json::array({{{"role", "user"}, {"content", "answer"}}})},
                             {"enable_thinking", true}, {"thinking_token_budget", 0},
                             {"max_tokens", cap}, {"stream", stream}};
                run_request(tok, "q27-test", false, false, true, 100000, 100000, rc,
                            cache, slots, body, true, batch);
                std::string content, finish;
                if (stream) {
                    for (const auto& ev : g_sse_events) {
                        if (!ev.contains("choices") || ev["choices"].empty()) continue;
                        const auto& choice = ev["choices"][0];
                        const auto& delta = choice["delta"];
                        if (delta.contains("content") && delta["content"].is_string())
                            content += delta["content"].get<std::string>();
                        if (choice.contains("finish_reason") && choice["finish_reason"].is_string())
                            finish = choice["finish_reason"].get<std::string>();
                    }
                } else {
                    const auto& choice = g_last_response["choices"][0];
                    content = choice["message"]["content"].get<std::string>();
                    finish = choice["finish_reason"].get<std::string>();
                }
                CHECK(content == "prefixsuffix");
                CHECK(finish == (cap == 4 ? "length" : "stop"));
            }
        }
    }

    // ---- Test 20: malformed tool text stays adjacent to preceding model
    // whitespace in both stream and non-stream responses. ----
    for (bool stream : {false, true}) {
        FakeTok tok;
        tok.pieces = {/*0*/ "<eos>", /*1*/ " before", /*2*/ "<tool_call>",
                      /*3*/ "not valid json", /*4*/ "</tool_call>", /*5*/ " after"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1, 2, 3, 4, 5};
        std::atomic<long> rc{0};
        json body = {
            {"tools", json::array({
                {{"type", "function"}, {"function", {{"name", "get_weather"},
                    {"description", "w"}, {"parameters", json::object()}}}}
            })},
            {"messages", json::array({{{"role", "user"}, {"content", "weather?"}}})},
            {"enable_thinking", false}, {"max_tokens", 5}, {"stream", stream},
        };
        run_request(tok, "q27-test", true, false, true, 100000, 100000, rc, cache,
                    slots, body, true);
        std::string content;
        if (stream) {
            for (const auto& ev : g_sse_events) {
                if (!ev.contains("choices") || ev["choices"].empty()) continue;
                const auto& delta = ev["choices"][0]["delta"];
                if (delta.contains("content") && delta["content"].is_string())
                    content += delta["content"].get<std::string>();
            }
        } else {
            content = g_last_response["choices"][0]["message"]["content"]
                          .get<std::string>();
        }
        CHECK(content == " beforenot valid json after");
    }
    // ---- Test 20: allowed_tools filters streamed wrapped calls. ----
    {
        FakeTok tok;
        tok.pieces = {
            /*0*/ "<eos>",
            /*1*/ "<tool_call>\n{\"name\":\"privileged\",\"arguments\":{}}\n</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1};
        std::atomic<long> rc{0};
        json declared=json::array({
            {{"type","function"},{"function",{{"name","safe"},{"parameters",json::object()}}}},
            {{"type","function"},{"function",{{"name","privileged"},{"parameters",json::object()}}}}
        });
        json allowed=json::array({
            {{"type","function"},{"function",{{"name","safe"}}}}
        });
        json body = {
            {"stream",true}, {"tools",declared},
            {"tool_choice",{{"type","allowed_tools"},
                {"allowed_tools",{{"mode","auto"},{"tools",allowed}}}}},
            {"messages",json::array({{{"role","user"},{"content","run safe"}}})},
        };
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        bool saw_tool=false, saw_raw=false, saw_stop=false;
        for (auto& ev : g_sse_events) {
            if (!ev.contains("choices") || ev["choices"].empty()) continue;
            auto& choice=ev["choices"][0];
            if (choice["delta"].contains("tool_calls")) saw_tool=true;
            if (choice["delta"].contains("content") &&
                choice["delta"]["content"].get<std::string>().find("privileged")!=std::string::npos)
                saw_raw=true;
            if (choice["finish_reason"] == "stop") saw_stop=true;
        }
        CHECK(!saw_tool);
        CHECK(saw_raw);
        CHECK(saw_stop);
    }

    // ---- Test 21: invalid named choice is a non-retryable client error. ----
    {
        FakeTok tok;
        tok.pieces = {"<eos>"};
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        std::atomic<long> rc{0};
        json body = {
            {"tools",json::array({
                {{"type","function"},{"function",{{"name","safe"},{"parameters",json::object()}}}}
            })},
            {"tool_choice",{{"type","function"},{"function",{{"name","missing"}}}}},
            {"messages",json::array({{{"role","user"},{"content","run"}}})},
        };
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        CHECK(g_last_response["__status"] == 400);
        CHECK(g_last_response["error"]["type"] == "invalid_request_error");
    }

    // ---- Test 12: parallel_tool_calls=false keeps only the first wrapped call. ----
    {
        FakeTok tok;
        tok.pieces = {
            "<eos>",
            "<tool_call>\n{\"name\":\"first\",\"arguments\":{}}\n</tool_call>",
            "<tool_call>\n{\"name\":\"second\",\"arguments\":{}}\n</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1,2};
        std::atomic<long> rc{0};
        json tools=json::array({
            {{"type","function"},{"function",{{"name","first"},{"parameters",json::object()}}}},
            {{"type","function"},{"function",{{"name","second"},{"parameters",json::object()}}}},
        });
        json body={{"tools",tools},{"parallel_tool_calls",false},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})}};
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        auto& msg=g_last_response["choices"][0]["message"];
        CHECK(msg["tool_calls"].size()==1);
        CHECK(msg["tool_calls"][0]["function"]["name"]=="first");
        CHECK(msg["content"].get<std::string>().find("second")!=std::string::npos);
    }

    // ---- Test 13: the same single-call contract holds for SSE output. ----
    {
        FakeTok tok;
        tok.pieces = {
            "<eos>",
            "<tool_call>\n{\"name\":\"first\",\"arguments\":{}}\n</tool_call>",
            "<tool_call>\n{\"name\":\"second\",\"arguments\":{}}\n</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1,2};
        std::atomic<long> rc{0};
        json tools=json::array({
            {{"type","function"},{"function",{{"name","first"},{"parameters",json::object()}}}},
            {{"type","function"},{"function",{{"name","second"},{"parameters",json::object()}}}},
        });
        json body={{"stream",true},{"tools",tools},{"parallel_tool_calls",false},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})}};
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        int call_count=0;
        bool saw_first=false,saw_second_raw=false;
        for(auto& ev:g_sse_events) {
            if(!ev.contains("choices") || ev["choices"].empty()) continue;
            const auto& delta=ev["choices"][0]["delta"];
            if(delta.contains("tool_calls")) {
                call_count++;
                saw_first=delta["tool_calls"][0]["function"]["name"]=="first";
            }
            if(delta.contains("content") &&
               delta["content"].get<std::string>().find("second")!=std::string::npos)
                saw_second_raw=true;
        }
        CHECK(call_count==1);
        CHECK(saw_first);
        CHECK(saw_second_raw);
    }

    // ---- Test 13b: an earlier bare call beats a later wrapped call. ----
    {
        FakeTok tok;
        tok.pieces = {
            "<eos>",
            "{\"name\":\"fi",
            "rst\",\"arguments\":{}}",
            "<tool_call>\n{\"name\":\"second\",\"arguments\":{}}\n</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1,2,3};
        std::atomic<long> rc{0};
        json tools=json::array({
            {{"type","function"},{"function",{{"name","first"},{"parameters",json::object()}}}},
            {{"type","function"},{"function",{{"name","second"},{"parameters",json::object()}}}},
        });
        json body={{"stream",true},{"tools",tools},{"parallel_tool_calls",false},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})}};
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        int call_count=0;
        bool saw_first=false,saw_first_raw=false,saw_second_raw=false;
        for(auto& ev:g_sse_events) {
            if(!ev.contains("choices") || ev["choices"].empty()) continue;
            const auto& delta=ev["choices"][0]["delta"];
            if(delta.contains("tool_calls")) {
                call_count++;
                saw_first=delta["tool_calls"][0]["function"]["name"]=="first";
            }
            if(delta.contains("content")) {
                const auto content=delta["content"].get<std::string>();
                if(content.find("\"name\":\"first\"")!=std::string::npos)
                    saw_first_raw=true;
                if(content.find("second")!=std::string::npos)
                    saw_second_raw=true;
            }
        }
        CHECK(call_count==1);
        CHECK(saw_first);
        CHECK(!saw_first_raw);
        CHECK(saw_second_raw);
    }

    // ---- Test 13c: a wrapper nested in an incomplete bare-call value stays text. ----
    {
        FakeTok tok;
        tok.pieces = {
            "<eos>",
            "{\"name\":\"first\",\"arguments\":{\"x\":",
            "<tool_call>\n{\"name\":\"second\",\"arguments\":{}}\n</tool_call>",
        };
        std::vector<std::string> vb = tok.pieces;
        auto cache = fresh_cache(vb);
        auto slots = fresh_slots();
        slots[0].eng->script = {1,2};
        std::atomic<long> rc{0};
        json tools=json::array({
            {{"type","function"},{"function",{{"name","first"},{"parameters",json::object()}}}},
            {{"type","function"},{"function",{{"name","second"},{"parameters",json::object()}}}},
        });
        json body={{"stream",true},{"tools",tools},{"parallel_tool_calls",false},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})}};
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        int call_count=0;
        std::string visible;
        for(auto& ev:g_sse_events) {
            if(!ev.contains("choices") || ev["choices"].empty()) continue;
            const auto& delta=ev["choices"][0]["delta"];
            if(delta.contains("tool_calls")) call_count++;
            if(delta.contains("content"))
                visible+=delta["content"].get<std::string>();
        }
        CHECK(call_count==0);
        CHECK(visible.find("\"name\":\"first\"")!=std::string::npos);
        CHECK(visible.find("<tool_call>")!=std::string::npos);
        CHECK(visible.find("\"name\":\"second\"")!=std::string::npos);
    }

    // ---- Test 13d: max-token truncation cannot expose an unclosed wrapper. ----
    {
        FakeTok tok;
        tok.pieces={"<eos>","<tool_call>{\"name\":\"first\",\"arguments\":{}}"};
        std::vector<std::string> vb=tok.pieces;
        auto cache=fresh_cache(vb);
        auto slots=fresh_slots();
        slots[0].eng->script={1};
        std::atomic<long> rc{0};
        json tools=json::array({
            {{"type","function"},{"function",{{"name","first"},{"parameters",json::object()}}}}
        });
        json body={{"max_tokens",1},{"tools",tools},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})}};
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        const auto& choice=g_last_response["choices"][0];
        CHECK(!choice["message"].contains("tool_calls"));
        CHECK(choice["message"]["content"].get<std::string>().find("first")!=std::string::npos);
        CHECK(choice["finish_reason"]=="length");
    }
    {
        FakeTok tok;
        tok.pieces={"<eos>","<tool_call>{\"name\":\"first\",\"arguments\":{}}"};
        std::vector<std::string> vb=tok.pieces;
        auto cache=fresh_cache(vb);
        auto slots=fresh_slots();
        slots[0].eng->script={1};
        std::atomic<long> rc{0};
        json tools=json::array({
            {{"type","function"},{"function",{{"name","first"},{"parameters",json::object()}}}}
        });
        json body={{"stream",true},{"max_tokens",1},{"tools",tools},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})}};
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        bool saw_tool=false,saw_raw=false,saw_length=false;
        for(const auto& ev:g_sse_events) {
            if(!ev.contains("choices") || ev["choices"].empty()) continue;
            const auto& choice=ev["choices"][0];
            const auto& delta=choice["delta"];
            if(delta.contains("tool_calls")) saw_tool=true;
            if(delta.contains("content") &&
               delta["content"].get<std::string>().find("first")!=std::string::npos)
                saw_raw=true;
            if(choice["finish_reason"]=="length") saw_length=true;
        }
        CHECK(!saw_tool);
        CHECK(saw_raw);
        CHECK(saw_length);
    }

    // ---- Test 13e: a closed call remains actionable at the exact token cap. ----
    {
        FakeTok tok;
        tok.pieces={"<eos>","<tool_call>{\"name\":\"first\",\"arguments\":{}}</tool_call>"};
        std::vector<std::string> vb=tok.pieces;
        auto cache=fresh_cache(vb);
        auto slots=fresh_slots();
        slots[0].eng->script={1};
        std::atomic<long> rc{0};
        json tools=json::array({
            {{"type","function"},{"function",{{"name","first"},{"parameters",json::object()}}}}
        });
        json body={{"max_tokens",1},{"tools",tools},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})}};
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        const auto& choice=g_last_response["choices"][0];
        CHECK(choice["message"]["tool_calls"].size()==1);
        CHECK(choice["message"]["tool_calls"][0]["function"]["name"]=="first");
        CHECK(choice["finish_reason"]=="tool_calls");
    }
    {
        FakeTok tok;
        tok.pieces={"<eos>","<tool_call>{\"name\":\"first\",\"arguments\":{}}</tool_call>"};
        std::vector<std::string> vb=tok.pieces;
        auto cache=fresh_cache(vb);
        auto slots=fresh_slots();
        slots[0].eng->script={1};
        std::atomic<long> rc{0};
        json tools=json::array({
            {{"type","function"},{"function",{{"name","first"},{"parameters",json::object()}}}}
        });
        json body={{"stream",true},{"max_tokens",1},{"tools",tools},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})}};
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        bool saw_tool=false,saw_tool_finish=false;
        for(const auto& ev:g_sse_events) {
            if(!ev.contains("choices") || ev["choices"].empty()) continue;
            const auto& choice=ev["choices"][0];
            if(choice["delta"].contains("tool_calls")) saw_tool=true;
            if(choice["finish_reason"]=="tool_calls") saw_tool_finish=true;
        }
        CHECK(saw_tool);
        CHECK(saw_tool_finish);
    }

    // ---- Test 13f: a malformed wrapper can contain multiple valid calls. ----
    {
        FakeTok tok;
        tok.pieces={"<eos>",
            "<tool_call>{\"name\":\"first\",\"arguments\":{}}"
            "{\"name\":\"second\",\"arguments\":{}}</tool_call>"};
        std::vector<std::string> vb=tok.pieces;
        auto cache=fresh_cache(vb);
        auto slots=fresh_slots();
        slots[0].eng->script={1};
        std::atomic<long> rc{0};
        json tools=json::array({
            {{"type","function"},{"function",{{"name","first"},{"parameters",json::object()}}}},
            {{"type","function"},{"function",{{"name","second"},{"parameters",json::object()}}}},
        });
        json body={{"stream",true},{"tools",tools},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})}};
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        std::vector<std::string> call_names;
        std::string visible;
        for(const auto& ev:g_sse_events) {
            if(!ev.contains("choices") || ev["choices"].empty()) continue;
            const auto& delta=ev["choices"][0]["delta"];
            if(delta.contains("tool_calls"))
                call_names.push_back(delta["tool_calls"][0]["function"]["name"]);
            if(delta.contains("content")) visible+=delta["content"].get<std::string>();
        }
        CHECK(call_names==std::vector<std::string>({"first","second"}));
        CHECK(visible.find("\"name\":\"first\"")==std::string::npos);
        CHECK(visible.find("\"name\":\"second\"")==std::string::npos);
    }

    // ---- Test 13g: balanced raw-value drift waits for tolerant final recovery. ----
    {
        FakeTok tok;
        tok.pieces={"<eos>",R"({"name":"Write","arguments":{"content":"const s = "x";"}})"," tail"};
        std::vector<std::string> vb=tok.pieces;
        auto cache=fresh_cache(vb);
        auto slots=fresh_slots();
        slots[0].eng->script={1,2};
        std::atomic<long> rc{0};
        json tools=json::array({
            {{"type","function"},{"function",{{"name","Write"},{"parameters",{
                {"type","object"},{"properties",{{"content",{{"type","string"}}}}}
            }}}}}
        });
        json body={{"stream",true},{"tools",tools},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})}};
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        int call_count=0;
        std::string visible;
        for(const auto& ev:g_sse_events) {
            if(!ev.contains("choices") || ev["choices"].empty()) continue;
            const auto& delta=ev["choices"][0]["delta"];
            if(delta.contains("tool_calls")) {
                call_count++;
                const auto& fn=delta["tool_calls"][0]["function"];
                CHECK(fn["name"]=="Write");
                CHECK(json::parse(fn["arguments"].get<std::string>())["content"]==
                      "const s = \"x\";");
            }
            if(delta.contains("content")) visible+=delta["content"].get<std::string>();
        }
        CHECK(call_count==1);
        CHECK(visible==" tail");
    }

    // ---- Test 14: malformed parallel_tool_calls is a client error. ----
    {
        FakeTok tok;
        tok.pieces={"<eos>"};
        std::vector<std::string> vb=tok.pieces;
        auto cache=fresh_cache(vb);
        auto slots=fresh_slots();
        std::atomic<long> rc{0};
        json body={{"parallel_tool_calls","no"},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})}};
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        CHECK(g_last_response["__status"]==400);
        CHECK(g_last_response["error"]["type"]=="invalid_request_error");
    }

    // ---- Test 15: forced non-stream truncation preserves partial output. ----
    {
        FakeTok tok;
        tok.pieces={"<eos>","not a tool call"};
        std::vector<std::string> vb=tok.pieces;
        auto cache=fresh_cache(vb);
        auto slots=fresh_slots();
        slots[0].eng->script={1};
        std::atomic<long> rc{0};
        json body={{"max_tokens",1},{"tool_choice","required"},
                   {"tools",json::array({{{"type","function"},{"function",{{"name","safe"},{"parameters",json::object()}}}}})},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})}};
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        CHECK(g_last_response["__status"]==200);
        CHECK(g_last_response["choices"][0]["finish_reason"]=="length");
        CHECK(g_last_response["choices"][0]["message"]["content"]=="not a tool call");
        CHECK(!g_last_response["choices"][0]["message"].contains("tool_calls"));
    }

    // ---- Test 16: forced SSE truncation terminates with length, not error. ----
    {
        FakeTok tok;
        tok.pieces={"<eos>","not a tool call"};
        std::vector<std::string> vb=tok.pieces;
        auto cache=fresh_cache(vb);
        auto slots=fresh_slots();
        slots[0].eng->script={1};
        std::atomic<long> rc{0};
        json body={{"stream",true},{"max_tokens",1},{"tool_choice","required"},
                   {"tools",json::array({{{"type","function"},{"function",{{"name","safe"},{"parameters",json::object()}}}}})},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})}};
        run_request(tok,"q27-test",true,false,true,100000,100000,rc,cache,slots,body,true);
        bool saw_error=false,saw_length=false,saw_partial=false;
        for(const auto& ev:g_sse_events) {
            if(ev.contains("error")) saw_error=true;
            if(!ev.contains("choices") || ev["choices"].empty()) continue;
            const auto& choice=ev["choices"][0];
            if(choice["finish_reason"]=="length") saw_length=true;
            if(choice.contains("delta") && choice["delta"].contains("content") &&
               choice["delta"]["content"]=="not a tool call") saw_partial=true;
        }
        CHECK(!saw_error);
        CHECK(saw_length);
        CHECK(saw_partial);
    }

    // ---- Test 16b: a forced request that re-enters THINK and exhausts its
    // reasoning budget is a truncated response, not a missing-tool error. ----
    for(bool stream:{false,true}) {
        FakeTok tok;
        tok.pieces={"<eos>","</tool_call><think>","unreachable reasoning"};
        std::vector<std::string> vb=tok.pieces;
        auto cache=fresh_cache(vb);
        auto slots=fresh_slots();
        slots[0].eng->script={1,2};
        std::atomic<long> rc{0};
        json body={{"stream",stream},{"max_tokens",2},
                   {"enable_thinking",true},{"thinking_token_budget",0},
                   {"tool_choice","required"},
                   {"tools",json::array({{{"type","function"},{"function",{
                       {"name","safe"},{"parameters",json::object()}}}}})},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})}};
        run_request(tok,"q27-test",false,false,true,100000,100000,rc,cache,slots,
                    body,true);
        if(!stream) {
            CHECK(g_last_response["__status"]==200);
            CHECK(g_last_response["choices"][0]["finish_reason"]=="length");
        } else {
            bool saw_error=false,saw_length=false;
            for(const auto& ev:g_sse_events) {
                if(ev.contains("error")) saw_error=true;
                if(!ev.contains("choices") || ev["choices"].empty()) continue;
                if(ev["choices"][0]["finish_reason"]=="length") saw_length=true;
            }
            CHECK(!saw_error);
            CHECK(saw_length);
        }
    }

    // ---- Test 17: count_tokens and live Anthropic requests share the exact
    // tool filtering, thinking, and forced-opener prompt construction. ----
    {
        json tools=json::array({
            {{"name","alpha"},{"description","first"},
             {"input_schema",{{"type","object"},{"properties",json::object()}}}},
            {{"name","beta"},{"description","second"},
             {"input_schema",{{"type","object"},{"properties",json::object()}}}},
        });
        json body={{"system","stable"},{"tools",tools},
                   {"messages",json::array({{{"role","user"},{"content","run"}}})},
                   {"tool_choice",{{"type","none"}}}};
        auto none=prepare_anthropic_prompt_for_test(body,false);
        CHECK(none.tools.empty());
        CHECK(none.tool_names.empty());
        CHECK(none.thinking);
        const json none_unavailable=q27::anthropic_tools_json(body);
        CHECK(none.rendered==q27::chatml_prompt(
            q27::anthropic_msgs(body),none.tools,none.thinking,nullptr,nullptr,
            q27::anthropic_tool_choice_instruction(none.tchoice),&none_unavailable));
        CHECK(none.rendered.find("# Tools")==std::string::npos);
        CHECK(none.rendered.find("<unavailable_tools>")!=std::string::npos);

        body["tool_choice"]={{"type","tool"},{"name","alpha"}};
        auto forced=prepare_anthropic_prompt_for_test(body,false);
        CHECK(forced.tchoice.mode==q27::ToolChoice::FORCED);
        CHECK(!forced.thinking);
        CHECK(forced.tool_names==std::vector<std::string>({"alpha"}));
        CHECK(forced.tools.size()==1);
        CHECK(forced.tools[0]["function"]["name"]=="alpha");
        const json all_forced_tools=q27::anthropic_tools_json(body);
        const json forced_unavailable=json::array({all_forced_tools[1]});
        CHECK(forced.rendered==q27::chatml_prompt(
            q27::anthropic_msgs(body),forced.tools,false,nullptr,nullptr,
            q27::anthropic_tool_choice_instruction(forced.tchoice),
            &forced_unavailable)+"<tool_call>\n");
        CHECK(forced.rendered.find("# Tools")!=std::string::npos);
        CHECK(forced.rendered.find("<unavailable_tools>")!=std::string::npos);
        CHECK(forced.rendered.find("stable")<forced.rendered.find("You must call only"));

        bool thinking_threw=false;
        body["thinking"]={{"type","enabled"},{"budget_tokens",1024}};
        try { (void)prepare_anthropic_prompt_for_test(body,false); }
        catch(const std::invalid_argument&) { thinking_threw=true; }
        CHECK(thinking_threw);
        body.erase("thinking");

        bool invalid_threw=false;
        body["tool_choice"]={{"type","tool"},{"name","missing"}};
        try { (void)prepare_anthropic_prompt_for_test(body,false); }
        catch(const std::runtime_error&) { invalid_threw=true; }
        CHECK(invalid_threw);
    }

    fprintf(stderr, failures ? "%d FAILURE(S)\n" : "all integration tests passed\n", failures);
    return failures ? 1 : 0;
}
