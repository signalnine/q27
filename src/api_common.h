// Shared prompt construction + tool-call parsing for the q27 API endpoints.
// Qwopus (qwen35) tool protocol, from the GGUF chat template:
//   system preamble lists tools as JSON inside <tools>...</tools>
//   model emits  <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
//   results go back as user content wrapped in <tool_response>...</tool_response>
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <mutex>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../third_party/json.hpp"
#include "markdown_lex.h"

#include "stream_split.h"

namespace q27 {
using json = nlohmann::json;

// ---- tolerant request-field readers ---------------------------------------
// json::value() THROWS type_error.302 when a key is PRESENT but null or
// wrong-typed, and httplib turns any throw out of a handler into a 500 with an
// EXCEPTION_WHAT header (the routing() try/catch in third_party/httplib.h).
// "Present and null" is how a large share of OpenAI-compatible clients spell
// "unset" -- {"max_tokens": null, "temperature": null, "stream": null} comes
// straight out of LangChain/LiteLLM-class request builders. Answering those
// with a 500 is wrong twice over: wrong actor (the request was fine), and
// wrong status class (500 is retryable, so the client loops the same body).
// These read null / wrong-typed exactly like ABSENT, which is what the field
// means on the wire. Numbers are read through double so an integer field sent
// as 8192.0 still works, and clamped to int range so a nonsense magnitude
// can't overflow the int the callers assign into.
inline double jnum(const json& b, const char* key, double dflt) {
    if (!b.is_object()) return dflt;
    const auto it = b.find(key);
    return (it != b.end() && it->is_number()) ? it->get<double>() : dflt;
}
inline long jint(const json& b, const char* key, long dflt) {
    if (!b.is_object()) return dflt;
    const auto it = b.find(key);
    if (it == b.end() || !it->is_number()) return dflt;
    const double v = it->get<double>();
    if (v >= 2147483647.0) return 2147483647L;
    if (v <= -2147483648.0) return -2147483648L;
    return (long)v;
}
inline bool jbool(const json& b, const char* key, bool dflt) {
    if (!b.is_object()) return dflt;
    const auto it = b.find(key);
    return (it != b.end() && it->is_boolean()) ? it->get<bool>() : dflt;
}
inline std::string jstr(const json& b, const char* key,
                        const std::string& dflt = std::string()) {
    if (!b.is_object()) return dflt;
    const auto it = b.find(key);
    return (it != b.end() && it->is_string()) ? it->get<std::string>() : dflt;
}
inline std::string json_dump_replace(const json& value) {
    return value.dump(-1,' ',false,json::error_handler_t::replace);
}

// Incremental UTF-8 boundary gate for streaming token pieces. BPE token
// boundaries can split a multi-byte character (em dash E2 80 94 is a Qwopus
// favorite), the raw piece is then invalid UTF-8, and nlohmann json::dump
// throws type_error.316 on it -- which took q27-server down mid-generation
// under Claude Code (R0, 2026-07-04). feed() returns the longest valid
// prefix, holding back an incomplete trailing sequence until its
// continuation bytes arrive; flush() ends the stream, turning a dangling
// partial into U+FFFD. Invalid leads/continuations pass through -- the
// dump-time replace error handler is the backstop for those.
struct Utf8Gate {
    std::string pend;
    static int seq_len(unsigned char b) {
        if (b < 0x80) return 1;
        if ((b & 0xE0) == 0xC0) return 2;
        if ((b & 0xF0) == 0xE0) return 3;
        if ((b & 0xF8) == 0xF0) return 4;
        return -1; // continuation or invalid lead byte
    }
    std::string feed(const std::string& piece) {
        pend += piece;
        size_t n = pend.size(), i = n;
        int back = 0;
        while (i > 0 && back < 4) {
            unsigned char b = (unsigned char)pend[i - 1];
            if ((b & 0xC0) != 0x80) { i--; break; }
            i--;
            back++;
        }
        size_t cut = n;
        if (i < n) {
            int L = seq_len((unsigned char)pend[i]);
            if (L > 0 && i + (size_t)L > n) cut = i; // incomplete tail: hold back
        }
        std::string out = pend.substr(0, cut);
        pend.erase(0, cut);
        return out;
    }
    std::string flush() {
        std::string out = pend.empty() ? std::string() : std::string("\xEF\xBF\xBD");
        pend.clear();
        return out;
    }
};

// R1b: FIFO ticket lock time-slicing the GPU across concurrent generations.
// Replaces the server's whole-generation mutex. The holder calls
// maybe_yield() at round/chunk boundaries: if anyone is queued it releases
// and re-acquires -- the fresh ticket lands at the TAIL, so contended
// requests round-robin at round granularity instead of head-of-line
// blocking for a whole generation. Solo path: one relaxed atomic load per
// call, no syscalls. contended() can miss a waiter arriving in the same
// instant (nwait is read unlocked); it is caught one round (~27ms) later.
struct GpuGate {
    void acquire() {
        std::unique_lock<std::mutex> lk(m);
        uint64_t t = next++;
        if (t != serving) {
            nwait.fetch_add(1, std::memory_order_relaxed);
            cv.wait(lk, [&] { return serving == t; });
            nwait.fetch_sub(1, std::memory_order_relaxed);
        }
    }
    void release() {
        { std::lock_guard<std::mutex> lk(m); serving++; }
        cv.notify_all();
    }
    int contended() const { return nwait.load(std::memory_order_relaxed); }
    // RAII whole-hold (the R1-equivalent region): exception-safe release,
    // same role the old lock_guard played at the server call sites.
    // Exemption to the drained-handover invariant: microsecond-scale async
    // copies queued AFTER the last yield point (tool-constraint clears,
    // n_max==0 tails) may still be in flight at ~Lease. All target
    // per-engine buffers and are stream-ordered ahead of that engine's next
    // work, so no cross-engine hazard exists; the GPU is "idle" at release
    // only up to those copies.
    struct Lease {
        explicit Lease(GpuGate& gg) : g(gg) { g.acquire(); }
        ~Lease() { g.release(); }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        GpuGate& g;
    };
    // Yield the GPU to queued waiters; true if a handover actually happened.
    // The new ticket is taken in the SAME critical section as the handover:
    // release();acquire() would let a descheduled yielder lose its queue
    // position to the next yielder (caught by the C1 self-test), breaking
    // strict rotation.
    bool maybe_yield() {
        if (!contended()) return false;
        std::unique_lock<std::mutex> lk(m);
        if (next - serving <= 1) return false; // raced: waiter already gone
        uint64_t t = next++;
        serving++;
        cv.notify_all();
        nwait.fetch_add(1, std::memory_order_relaxed);
        cv.wait(lk, [&] { return serving == t; });
        nwait.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }
private:
    std::mutex m;
    std::condition_variable cv;
    uint64_t next = 0, serving = 0;
    std::atomic<int> nwait{0};
};

struct Msg {
    std::string role;     // system | user | assistant
    std::string content;  // flattened text (think blocks already reconstructed)
};

// Tools preamble, verbatim structure from the chat template. `tools` entries
// must already be in {"type":"function","function":{...}} shape.
// Claude Code (<= 2.1.1xx era) prefixes its system prompt with
//   x-anthropic-billing-header: cc_version=...; cc_entrypoint=cli; cch=a5145;You are...
// The cch stamp is an integrity hint that CHANGES ON EVERY REQUEST, so the
// first bytes of the prompt mutate per turn -- which voids the P8 stable-prefix
// snapshot and P9 checkpoint routing for the entire conversation (measured
// under Claude Code: 126K-token full re-prefill, ~72s, on every turn). Pin the
// stamp to 'f's, mirroring llama.cpp's normalize_anthropic_billing_header
// (ggml-org/llama.cpp#21793), so both engines canonicalize to the same bytes.
// Only a header at the very start of the system text is touched, and the stamp
// is only looked for inside the short header segment.
//
// FORMAT DRIFT (measured 2026-07-24 against Claude Code 2.1.220): the `cch=`
// field is GONE, and the volatile stamp moved onto the version itself --
//   x-anthropic-billing-header: cc_version=2.1.220.473; cc_entrypoint=sdk-cli;
//                                                  ^^^ changes between conversations
// The old `cch=`-only normalizer returns early on that shape and does nothing,
// so the first ~15 tokens of every CC system prompt differ between sessions.
// That silently voids CROSS-SESSION reuse for all three tiers at once (P8
// snapshot, P9 ring, P16 disk) -- captured live: two sessions differing ONLY in
// their task text produced two distinct 21,504-token cache entries and never
// hit each other. Within one conversation the stamp is stable, which is why
// same-session warm turns kept working and this hid.
// Both stamp forms are now pinned: the legacy `cch=` value, and any 4th+
// dot-component of `cc_version=` (2.1.220 = the real version; .473 = the
// volatile tail). Pinning is safe -- this is prompt text an engine only needs
// to canonicalize, and llama.cpp does the same thing for the same reason.
inline void normalize_cc_billing_header(std::string& sys) {
    static const char* PFX = "x-anthropic-billing-header:";
    if (sys.rfind(PFX, 0) != 0) return;
    // legacy: cch=<stamp>;
    size_t cch = sys.find("cch=", 27);
    if (cch != std::string::npos && cch <= 160) {
        size_t v = cch + 4, end = sys.find(';', v);
        if (end != std::string::npos && end != v && end - v <= 16)
            for (size_t i = v; i < end; ++i) sys[i] = 'f';
    }
    // 2.1.220+: cc_version=<a.b.c>.<volatile>;  -- pin everything past the 3rd dot
    size_t cv = sys.find("cc_version=", 0);
    if (cv == std::string::npos || cv > 160) return;
    size_t v = cv + 11, end = sys.find(';', v);
    if (end == std::string::npos || end <= v || end - v > 64) return;
    int dots = 0;
    for (size_t i = v; i < end; ++i) {
        if (sys[i] == '.' && ++dots == 3) {
            for (size_t j = i + 1; j < end; ++j) sys[j] = 'f';
            return;
        }
    }
}

// Strip ChatML role delimiters from untrusted content/roles so they can't forge
// prompt structure (Security #7): the tokenizer matches <|im_start|>/<|im_end|>
// as control tokens anywhere, so a document or tool result containing them would
// otherwise become real role boundaries. Operator content that legitimately
// includes the literal markers loses them -- the safe tradeoff vs injection.
inline std::string strip_ctrl(std::string s) {
    for (const std::string& m : {std::string("<|im_start|>"), std::string("<|im_end|>")})
        for (size_t p; (p = s.find(m)) != std::string::npos;) s.erase(p, m.size());
    return s;
}

inline std::string tools_preamble(const json& tools) {
    std::string s = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    // tool declarations carry caller-controlled (and often third-party-
    // authored) description strings -- same forgery surface as message
    // content (review 2026-07-09 P1 #5)
    for (auto& t : tools) s += "\n" + strip_ctrl(t.dump());
    s += "\n</tools>\n\nFor each function call, return a JSON object with the function name "
         "and arguments inside <tool_call></tool_call> tags:\n<tool_call>\n{\"name\": "
         "<function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n<IMPORTANT>\n"
         "- Required parameters MUST be specified.\n- You may provide optional reasoning "
         "before the function call, but never after it.\n- If no function call is needed, "
         "answer normally and do not mention the tool interface.\n</IMPORTANT>";
    return s;
}

inline std::string unavailable_tools_preamble(const json& tools) {
    std::string s = "# Unavailable tools\n\nThe following tool declarations are included "
                    "for request accounting only. They are not callable in this response:\n\n"
                    "<unavailable_tools>";
    for (const auto& tool : tools) s += "\n" + strip_ctrl(tool.dump());
    s += "\n</unavailable_tools>";
    return s;
}

// Build the full ChatML prompt string. If tools are present they are merged
// into the (first) system message per the template's merged_system behavior.
// think=false appends the empty think block (enable_thinking=false
// convention); the tokenizer matches <think>/</think> as single added tokens.
// stable_off (P8): char offset where the trailing assistant-open begins.
// Everything before it re-renders identically next turn (snapshot-safe);
// everything after (assistant open + think prefill) is per-turn volatile.
// sys_off (P16b): char offset just past the system+tools block, or 0 when the
// request has neither. That block is the part MULTIPLE conversations share --
// Claude Code re-sends the same 20-25K-token system + tool definitions every
// session -- so it is the only boundary a cross-conversation cache entry can
// usefully be cut at. Like stable_off it abuts an <|im_start|>, so the same
// split-invariance argument applies.
inline std::string chatml_prompt(const std::vector<Msg>& msgs, const json& tools,
                                 bool think = true, size_t* stable_off = nullptr,
                                 size_t* sys_off = nullptr,
                                 const std::string& tool_instruction = {},
                                 const json* unavailable_tools = nullptr) {
    std::string p;
    size_t start = 0;
    std::string sys;
    if (!msgs.empty() && msgs[0].role == "system") { sys = strip_ctrl(msgs[0].content); start = 1; }
    // Over-refusal fix (2026-07-13, external review): under the no-think
    // serving default a bare request WITH NO SYSTEM PROMPT gives the model
    // zero context AND zero reasoning budget, so it falls to a defensive
    // refusal prior on borderline-legitimate requests (measured: a
    // signed-authorization pentest command). A minimal neutral default --
    // only when the client supplied none -- fully recovers compliance at
    // zero reasoning cost (real Claude Code always sends a system prompt, so
    // this never fires there). Q27_BARE=1 restores the no-default behavior.
    if (sys.empty() && !getenv("Q27_BARE")) sys = "You are a helpful assistant.";
    const bool has_tools=tools.is_array() && !tools.empty();
    const bool has_unavailable=unavailable_tools && unavailable_tools->is_array() &&
                               !unavailable_tools->empty();
    if (has_tools || has_unavailable || !sys.empty() || !tool_instruction.empty()) {
        p += "<|im_start|>system\n";
        bool need_separator=false;
        auto append_system_part=[&](const std::string& part) {
            if (part.empty()) return;
            if (need_separator) p += "\n\n";
            p += part;
            need_separator=true;
        };
        if (has_tools) append_system_part(tools_preamble(tools));
        if (has_unavailable)
            append_system_part(unavailable_tools_preamble(*unavailable_tools));
        append_system_part(sys);
        append_system_part(strip_ctrl(tool_instruction));
        p += "<|im_end|>\n";
    }
    if (sys_off) *sys_off = p.size();  // 0 when no system block was emitted
    for (size_t i = start; i < msgs.size(); i++)
        p += "<|im_start|>" + strip_ctrl(msgs[i].role) + "\n" + strip_ctrl(msgs[i].content) +
             "<|im_end|>\n";
    if (stable_off) *stable_off = p.size();
    p += "<|im_start|>assistant\n";
    // think=false: the empty CLOSED block signals "reasoning done" so the model
    // answers directly. think=true: prefill the OPEN <think> tag so the model
    // enters a real thinking block -- this checkpoint reasons inline and never
    // opens <think> on its own, but given the opener it fills a trace and closes
    // with </think> before answering. Both sit in the volatile tail (past
    // stable_off), so P8 prefix reuse is untouched. Generation paths pre-seed the
    // StreamSplitter to THINK when think=true so the model's first generated
    // token (already inside the block) routes to the think channel -- mirrors the
    // FORCED tool_choice TOOL pre-seed.
    if (!think) p += "<think>\n\n</think>\n\n";
    else p += "<think>\n";
    return p;
}

inline std::string tool_call_text(const std::string& name, const json& args) {
    return "<tool_call>\n{\"name\": \"" + name + "\", \"arguments\": " + args.dump() +
           "}\n</tool_call>";
}

inline std::string tool_response_text(const std::string& out) {
    return "<tool_response>\n" + out + "\n</tool_response>";
}

// Per-request thinking resolution, GATED behind the server's --request-think
// flag (`allow_request`). `server_default` is the server profile's stance
// (!no_think_srv): no-think serving passes false, --think / the ref profile
// pass true.
//
// Without --request-think (the default), the request's thinking fields are
// IGNORED and the server default stands -- so a benchmark or client that sends
// enable_thinking:True (many do) can't silently flip a no-think server into
// thinking mode. Thinking is then purely a boot decision (--think).
//
// With --request-think, an explicit request field OVERRIDES the default in
// either direction (a no-think server serves thinking on request; a --think
// server suppresses it). Three client conventions, all honored -- a given
// client sends exactly one:
//   OpenAI / Qwen   : top-level  "enable_thinking": <bool>
//                                "thinking_token_budget": <int>
//   llama.cpp / GLM : "chat_template_kwargs": {"enable_thinking": <bool>,
//                                              "thinking_budget": <int>}
//   Anthropic       : "thinking": {"type": "enabled"|"disabled",
//                                  "budget_tokens": <int>}
// Malformed/wrong-typed fields are ignored rather than thrown (Security #1).
// Later checks win over earlier ones; the conventions never co-occur in practice.
//
// BUDGET (2026-07-28). Every convention that can turn thinking ON can also
// bound it -- an accepted-and-inert field is worse than an absent one, because
// it reads as working. `budget` is a cap on tokens generated inside the
// <think> block; on trip the engine injects `</think>` and the answer proceeds.
// -1 means unbounded. Measured motivation: on an 11-pack / 240-scenario A/B
// (BUILDLOG 2026-07-28) the unbudgeted block arm truncated 28 times against the
// inline arm's 0, and every point it lost came from a run that never
// terminated -- where nothing truncated it was neutral or better.
struct ThinkCfg {
    bool enabled = false;
    int budget = -1;  // -1 = unbounded; >=0 = max tokens inside the block
    bool budget_set = false; // request supplied a budget, including negative opt-out
    bool enabled_set = false; // request explicitly enabled/disabled thinking
};

inline ThinkCfg resolve_think_cfg(const json& body, bool server_default, bool allow_request,
                                  int server_budget) {
    ThinkCfg c{server_default, server_budget};
    if (!allow_request) return c;
    auto take_budget = [&](const json& v) {
        // 0 is a legitimate "no thinking budget at all"; negative means
        // unbounded. nlohmann classifies unsigned values as integer too, so
        // validate the stored representation before narrowing to int.
        if (!v.is_number_integer()) return;
        if (v.is_number_unsigned()) {
            const uint64_t value = v.get<uint64_t>();
            if (value > (uint64_t)std::numeric_limits<int>::max()) return;
            c.budget = (int)value;
        } else {
            const int64_t value = v.get<int64_t>();
            if (value < (int64_t)std::numeric_limits<int>::min() ||
                value > (int64_t)std::numeric_limits<int>::max()) return;
            c.budget = (int)value;
        }
        c.budget_set = true;
    };
    if (body.contains("enable_thinking") && body["enable_thinking"].is_boolean()) {
        c.enabled = body["enable_thinking"].get<bool>();
        c.enabled_set = true;
    }
    if (body.contains("thinking_token_budget")) take_budget(body["thinking_token_budget"]);
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
        const auto& k = body["chat_template_kwargs"];
        if (k.contains("enable_thinking") && k["enable_thinking"].is_boolean()) {
            c.enabled = k["enable_thinking"].get<bool>();
            c.enabled_set = true;
        }
        if (k.contains("thinking_budget")) take_budget(k["thinking_budget"]);
    }
    if (body.contains("thinking") && body["thinking"].is_object()) {
        const auto& t = body["thinking"];
        if (t.contains("type") && t["type"].is_string()) {
            const std::string ty = t["type"].get<std::string>();
            if (ty == "enabled") { c.enabled = true; c.enabled_set = true; }
            else if (ty == "disabled") { c.enabled = false; c.enabled_set = true; }
        }
        if (t.contains("budget_tokens")) take_budget(t["budget_tokens"]);
    }
    return c;
}

// Thin compat wrapper: the boolean-only question, for call sites that do not
// run a generation loop (and for test_think_resolve's existing cases).
inline bool resolve_think(const json& body, bool server_default, bool allow_request) {
    return resolve_think_cfg(body, server_default, allow_request, -1).enabled;
}

// Server-side default budget as a FRACTION of the request's max_tokens. The
// fraction is the load-bearing part: whatever the cap, half of it remains for
// the answer after the block closes. An absolute default would starve small
// requests and barely bind large ones.
//
// 0.5 specifically is a judgement call, not a measurement. It lands on 8192 for
// a 16K request, which is the value club-3090 #765 reported clean terminations
// at -- but that thread compared two BUDGETS (8192 vs a 16K rerun), not a
// budget as a fraction of a cap, so read the agreement as a sanity check on
// magnitude rather than as their result reproducing here. If a measured
// fraction ever exists, it should replace this.
static constexpr double THINK_BUDGET_FRAC = 0.5;

// Resolve the server budget for a request whose public cap is `n_max`.
// `flag` is --think-budget: <0 = use the fractional default, 0 = unbounded,
// >0 = an explicit absolute cap. The fractional default applies only when the
// prompt already opened THINK: standard thinking-enabled serving starts there,
// while a later model-generated block is intentionally unbounded unless the
// client or an explicit positive server flag arms that guard.
inline int think_budget_default(int flag, int n_max) {
    if (flag == 0) return -1;
    if (flag > 0) return flag;
    return n_max > 0 ? (int)(THINK_BUDGET_FRAC * n_max) : -1;
}

inline int think_budget_for_request(bool prompt_starts_in_think, const ThinkCfg& cfg,
                                    int flag, int n_max) {
    if (cfg.enabled_set && !cfg.enabled) return -1;
    if (cfg.budget_set) return cfg.budget;
    if (!prompt_starts_in_think && flag < 0) return -1;
    return think_budget_default(flag, n_max);
}

struct ThinkDecodeLimits {
    int n_max = 0;
    int budget = -1;
    bool context_ok = true;
};

inline ThinkDecodeLimits resolve_think_decode_limits(
    int requested, int max_ctx, int prompt_tokens, int round_reserve,
    int close_tokens, bool active, const ThinkCfg& cfg, int flag) {
    const int raw_capacity = std::max(0, max_ctx - prompt_tokens - (round_reserve - 1));
    const int raw_n_max = std::max(0, std::min(requested, raw_capacity));
    const int raw_budget = think_budget_for_request(active, cfg, flag, raw_n_max);
    if (raw_capacity == 0) return {0, raw_budget, false};

    // Solve against the cap after reserving the close: fractional defaults
    // shrink with n_max, and an absolute cap at/above n_max is inert. The
    // candidate is valid only when the cap can fire before the public limit.
    // A prompt already in THINK needs one remaining answer token; a prompt in
    // TEXT also needs one public token for a model-generated <think> opener,
    // which is a delimiter rather than budgeted reasoning content.
    const int bounded_n_max = std::max(
        0, std::min(requested, std::max(0, raw_capacity - close_tokens)));
    const int bounded_budget =
        think_budget_for_request(active, cfg, flag, bounded_n_max);
    const int public_reserve = active ? 1 : 2;
    if (bounded_budget >= 0 && bounded_budget <= bounded_n_max - public_reserve)
        return {bounded_n_max, bounded_budget, true};

    // If a prompt already in THINK would fire under the unreserved cap, the
    // request cannot honor the close and still expose an answer token. A TEXT
    // prompt is different: thinking is only a future model choice. When its
    // public cap is too short for opener + budget + answer, preserve the
    // ordinary request and disable enforcement instead of reporting a false
    // context overflow; a spontaneous think then ends by the normal length cap.
    if (raw_budget >= 0 && raw_budget < raw_n_max) {
        if (!active) return {raw_n_max, -1, true};
        return {0, raw_budget, false};
    }

    // Unbounded or unattainable caps require neither one-token acceptance nor
    // close-token context. Report -1 to disable enforcement for this request.
    return {raw_n_max, -1, true};
}

// Nearest admissible prompt below a rejected request. Admission can be
// nonmonotonic near the boundary (an absolute cap can become inert), so a
// larger admissible prompt is not a truthful compaction target for the current
// rejection. Scan downward from the smaller of the ordinary ceiling and the
// token count immediately below the rejected prompt.
inline int max_prompt_for_think_decode(
    int requested, int max_ctx, int round_reserve, int prompt_ceiling,
    int rejected_prompt, int close_tokens, bool active, const ThinkCfg& cfg,
    int flag) {
    const int ceiling = std::max(0, std::min(prompt_ceiling, rejected_prompt - 1));
    for (int prompt = ceiling; prompt >= 0; prompt--)
        if (resolve_think_decode_limits(requested, max_ctx, prompt, round_reserve,
                                        close_tokens, active, cfg, flag).context_ok)
            return prompt;
    return 0;
}

// Request-local controller for bounded <think> spans. It observes semantic
// channel transitions and decides WHEN a close is needed; the engine owns HOW
// the tokenizer's close ids are committed through decoder state. Positive
// budgets are checked after a whole decode round, so a trip may overshoot by
// that round's accepted width. Request budget zero is the one pre-round case.
enum class ThinkBudgetAction { NONE, FORCE_RESERVED, FORCE_PUBLIC };

struct ThinkBudgetState {
    int limit = -1;
    int used = 0;
    bool tripped = false;
    bool transition_pending = false;
    bool reserved_close = true;

    explicit ThinkBudgetState(int cap = -1) : limit(cap) {}

    ThinkBudgetAction start(StreamSplitter::Chan initial = StreamSplitter::THINK) {
        if (limit == 0 && initial == StreamSplitter::THINK) {
            tripped = true;
            transition_pending = true;
            reserved_close = false;
            return ThinkBudgetAction::FORCE_RESERVED;
        }
        return ThinkBudgetAction::NONE;
    }

    // Feed every semantically committed token in round order. Forced control
    // ids change channel state but do not consume public reasoning usage.
    void observe(StreamSplitter::Chan before, StreamSplitter::Chan after,
                 bool forced = false) {
        if (!forced && before == StreamSplitter::THINK) used++;
        if (before == StreamSplitter::THINK && after != StreamSplitter::THINK)
            transition_pending = false;
    }


    // Call once after the complete retained round has been observed. A natural
    // close later in the same speculative round wins: no duplicate close is
    // queued. Once the reserved first close has tripped, any later re-entry
    // borrows its close ids from the remaining public token allowance.
    ThinkBudgetAction finish_round(StreamSplitter::Chan after) {
        if (limit < 0 || transition_pending || after != StreamSplitter::THINK ||
            used < limit)
            return ThinkBudgetAction::NONE;
        tripped = true;
        transition_pending = true;
        if (reserved_close) {
            reserved_close = false;
            return ThinkBudgetAction::FORCE_RESERVED;
        }
        return ThinkBudgetAction::FORCE_PUBLIC;
    }
};

// Anthropic error envelope, exactly the real API's shape: the SDK inside
// Claude Code reads error.message from it, and CC's compact-vs-retry
// decision substring-matches that message.
// Minimal JSON structural context for mode-10 quote repair. A colon can
// terminate a quoted OBJECT KEY, but inside a string VALUE it is ordinary
// content (for example the raw shell fragment `echo "key": value`).
struct JsonQuoteContext {
    struct Frame { char kind; bool expect_key; };
    std::vector<Frame> stack;
    bool opening_string_is_key() const {
        return !stack.empty() && stack.back().kind=='{' && stack.back().expect_key;
    }
    void structural(char c) {
        if(c=='{') stack.push_back({'{',true});
        else if(c=='[') stack.push_back({'[',false});
        else if(c=='}') { if(!stack.empty() && stack.back().kind=='{') stack.pop_back(); }
        else if(c==']') { if(!stack.empty() && stack.back().kind=='[') stack.pop_back(); }
        else if(c==':' && !stack.empty() && stack.back().kind=='{') stack.back().expect_key=false;
        else if(c==',' && !stack.empty() && stack.back().kind=='{') stack.back().expect_key=true;
    }
};

inline std::string escape_json_interior(const std::string& s);

inline void append_json_escaped_byte(std::string& out, unsigned char c) {
    switch(c) {
        case '"': out+="\\\""; return;
        case '\\': out+="\\\\"; return;
        case '\b': out+="\\b"; return;
        case '\f': out+="\\f"; return;
        case '\n': out+="\\n"; return;
        case '\r': out+="\\r"; return;
        case '\t': out+="\\t"; return;
        default:
            if(c<0x20) {
                static constexpr char hex[]="0123456789abcdef";
                out+="\\u00";
                out+=hex[c>>4];
                out+=hex[c&15];
            } else out+=(char)c;
    }
}

// Incremental tool-call argument streamer. Shares the
// mode-5/10 drift semantics of JsonSanitizerStepper but deliberately keeps
// its OWN loop: it is STREAMING (no full-call buffer for
// quote_terminates_string lookahead) and adds mode-3 <content>-tag capture,
// neither of which fits the stepper's buffer+index contract. Keep the
// escape/repair RULES in lockstep with the stepper; mechanics differ.
//
// Feeds the splitter's TOOL-channel bytes as they decode; once the call head
// parses ({"name": "X", "arguments": { — tolerating mode 9's missing
// opening quote on the arguments key), object fields stream out sanitized.
// A quoted `content` field is the one deliberate holdback: until its closing
// bytes arrive, valid JSON is indistinguishable from mode 3's raw
// `"content":"RAW</content>` drift. That value is released at finalization
// if valid, or escaped when the tag appears; other fields remain incremental.
// Heads that deviate (mode 6/7/8 shapes) or exceed the bound never stream:
// raw bytes stay exact for the buffered recovery path. A streamed call whose
// body ends unbalanced reaches the client unbalanced (production semantics —
// the model's bytes are the model's bytes); repair applies only where bytes
// have not already reached the wire.
struct ToolCallStreamer {
    enum State { HEAD, ARGS, DONE, INVALID_DONE, FALLBACK };
    State state = HEAD;
    std::string raw;      // entire body verbatim (fallback + logging)
    std::string name;     // valid once opened
    bool opened = false;  // head parsed; an opener chunk belongs on the wire

    bool active() const { return !raw.empty(); }
    bool invalid() const { return invalid_done_; }
    void reset() { *this = ToolCallStreamer(); }

    // Feed body bytes; returns the sanitized argument fragment to stream
    // (often empty). *opened_now fires on the feed that completes the head.
    std::string feed(const std::string& t, bool* opened_now) {
        if (opened_now) *opened_now = false;
        raw += t;
        if (state == DONE || state == INVALID_DONE) { add_trail(t); return ""; }
        if (state == FALLBACK) return "";
        std::string out;
        if (state == HEAD) {
            head_ += t;
            size_t consumed = 0;
            int m = match_head(head_, name, consumed);
            if (m == 0) {
                if (head_.size() > 512) state = FALLBACK;
                return "";
            }
            if (m < 0) { state = FALLBACK; return ""; }
            state = ARGS;
            opened = true;
            if (opened_now) *opened_now = true;
            std::string rest = head_.substr(consumed);
            head_.clear();
            scan(rest, out);
            sanitized_ += out;
            validate_done();
            return out;
        }
        scan(t, out);
        sanitized_ += out;
        validate_done();
        return out;
    }

    // Wrapper closed (or turn flushed). True = the args object completed
    // cleanly (DONE). Mid-ARGS: resolves a pending quote per the EOF rule
    // (terminator), appends the final bytes to *tail, returns false — the
    // handler closes the call as-is. HEAD/FALLBACK: false, nothing streamed.
    bool finalize(std::string* tail,bool allow_repair=true) {
        if (state == DONE) return true;
        if (!allow_repair) return false;
        if (state == ARGS) {
            if (defer_raw_tail_) {
                const std::string pending = raw_tail_;
                const bool array_value = !json_ctx_.stack.empty() &&
                    json_ctx_.stack.back().kind == '[';
                if (!raw_replay_work_) raw_replay_work_ = std::make_shared<size_t>(0);
                bool have_fallback = false;
                ToolCallStreamer fallback;
                std::string fallback_output;
                bool fallback_starts_call = false;
                bool have_incomplete = false;
                ToolCallStreamer incomplete;
                std::string incomplete_output;
                for (size_t terminal = pending.find("</content>");
                     terminal != std::string::npos;
                     terminal = pending.find("</content>", terminal + 1)) {
                    size_t after = terminal + 10;
                    while (after < pending.size() && is_ws(pending[after])) after++;
                    bool structural = false;
                    if (after == pending.size()) structural = true;
                    else if (pending[after] == '}' || pending[after] == ']') structural = true;
                    else if (pending[after] == ',') {
                        size_t next = after + 1;
                        while (next < pending.size() && is_ws(pending[next])) next++;
                        if (array_value) {
                            if (next < pending.size()) {
                                const char n = pending[next];
                                structural = n == '<' || n == '"' || n == '{' || n == '[' ||
                                    n == '-' || (n >= '0' && n <= '9') || n == 't' ||
                                    n == 'f' || n == 'n';
                            }
                        } else if (next < pending.size() && pending[next] == '"') {
                            bool key_escape = false;
                            size_t close = next + 1;
                            for (; close < pending.size(); close++) {
                                if (key_escape) { key_escape = false; continue; }
                                if (pending[close] == '\\') { key_escape = true; continue; }
                                if (pending[close] == '"') break;
                            }
                            size_t colon = close < pending.size() ? close + 1 : close;
                            while (colon < pending.size() && is_ws(pending[colon])) colon++;
                            structural = colon < pending.size() && pending[colon] == ':';
                        }
                    }
                    if (!structural) continue;
                    ToolCallStreamer replay = *this;
                    constexpr size_t max_raw_replay_work = 4u * 1024u * 1024u;
                    if (pending.size() > max_raw_replay_work -
                        std::min(*raw_replay_work_, max_raw_replay_work)) break;
                    *raw_replay_work_ += pending.size();
                    replay.raw.clear();
                    replay.defer_raw_tail_ = false;
                    replay.raw_tail_.clear();
                    replay.raw_tail_depth_ = raw_tail_depth_ + 1;
                    replay.defer_raw_tails_enabled_ = replay.raw_tail_depth_ < 8;
                    replay.tag_raw_ = false;
                    const std::string escaped = escape_json_interior(pending.substr(0, terminal));
                    replay.sanitized_ += escaped;
                    std::string scanned;
                    replay.scan("\"" + pending.substr(terminal + 10), scanned);
                    replay.sanitized_ += scanned;
                    replay.validate_done();
                    std::string replay_tail;
                    const bool complete = replay.finalize(&replay_tail);
                    const size_t trail_begin = replay.trail_.find_first_not_of(" \t\r\n");
                    bool clean_trail = trail_begin == std::string::npos;
                    if (!clean_trail) {
                        std::string remaining = replay.trail_.substr(trail_begin);
                        clean_trail = true;
                        int packed_count = 0;
                        while (!remaining.empty() && packed_count++ < 64) {
                            ToolCallStreamer packed;
                            packed.raw_replay_work_ = raw_replay_work_;
                            packed.raw_tail_depth_ = raw_tail_depth_ + 1;
                            bool packed_opened = false;
                            (void)packed.feed(remaining, &packed_opened);
                            std::string packed_tail;
                            if (!packed_opened || !packed.finalize(&packed_tail)) {
                                clean_trail = false;
                                break;
                            }
                            remaining = packed.trail_;
                            const size_t next = remaining.find_first_not_of(" \t\r\n");
                            if (next == std::string::npos) remaining.clear();
                            else if (next > 0) remaining.erase(0, next);
                        }
                        if (!remaining.empty()) clean_trail = false;
                    }
                    if (complete && clean_trail) {
                        *tail += escaped + scanned + replay_tail;
                        state = replay.state;
                        invalid_done_ = replay.invalid_done_;
                        trail_ = std::move(replay.trail_);
                        outer_seen_ = replay.outer_seen_;
                        defer_raw_tail_ = false;
                        raw_tail_.clear();
                        return true;
                    }
                    if (complete && !have_fallback) {
                        have_fallback = true;
                        fallback = std::move(replay);
                        fallback_output = escaped + scanned + replay_tail;
                        const size_t fallback_begin = fallback.trail_.find_first_not_of(" \t\r\n");
                        if (fallback_begin != std::string::npos) {
                            std::string packed_name;
                            size_t packed_args = 0;
                            fallback_starts_call =
                                match_head(fallback.trail_.substr(fallback_begin),
                                           packed_name, packed_args) == 1;
                        }
                    }
                    if (!complete && replay.state == ARGS && replay.trail_.empty()) {
                        have_incomplete = true;
                        incomplete = std::move(replay);
                        incomplete_output = escaped + scanned + replay_tail;
                    }
                }
                if (have_fallback && fallback_starts_call) {
                    *tail += fallback_output;
                    state = fallback.state;
                    invalid_done_ = fallback.invalid_done_;
                    trail_ = std::move(fallback.trail_);
                    outer_seen_ = fallback.outer_seen_;
                    defer_raw_tail_ = false;
                    raw_tail_.clear();
                    return true;
                }
                if (have_incomplete) {
                    *tail += incomplete_output;
                    state = incomplete.state;
                    invalid_done_ = incomplete.invalid_done_;
                    trail_ = std::move(incomplete.trail_);
                    outer_seen_ = incomplete.outer_seen_;
                    defer_raw_tail_ = false;
                    raw_tail_.clear();
                    return false;
                }
                if (have_fallback) {
                    *tail += fallback_output;
                    state = fallback.state;
                    invalid_done_ = fallback.invalid_done_;
                    trail_ = std::move(fallback.trail_);
                    outer_seen_ = fallback.outer_seen_;
                    defer_raw_tail_ = false;
                    raw_tail_.clear();
                    return true;
                }
                *tail += escape_json_interior(pending);
                *tail += '"';
                defer_raw_tail_ = false;
                raw_tail_.clear();
                tag_raw_ = false;
                return false;
            }
            if (defer_content_) {
                const std::string pending = deferred_;
                const std::string complete_text = sanitized_ + pending;
                auto balanced_end = [](const std::string& text) {
                    int depth = 0;
                    bool in_string = false, escaped = false, started = false;
                    for (size_t i = 0; i < text.size(); i++) {
                        const char c = text[i];
                        if (escaped) { escaped = false; continue; }
                        if (in_string) {
                            if (c == '\\') escaped = true;
                            else if (c == '"') in_string = false;
                            continue;
                        }
                        if (c == '"') in_string = true;
                        else if (c == '{') { depth++; started = true; }
                        else if (c == '}' && started && --depth == 0) return i;
                    }
                    return std::string::npos;
                };
                const size_t exact_end = balanced_end(complete_text);
                if (exact_end != std::string::npos && exact_end + 1 >= sanitized_.size()) {
                    try {
                        const json exact = json::parse(complete_text.substr(0, exact_end + 1));
                        if (exact.is_object()) {
                            const std::string emitted = complete_text.substr(
                                sanitized_.size(), exact_end + 1 - sanitized_.size());
                            *tail += emitted;
                            sanitized_ += emitted;
                            state = DONE;
                            defer_content_ = false;
                            deferred_.clear();
                            add_trail(complete_text.substr(exact_end + 1));
                            return true;
                        }
                    } catch (...) {}
                }

                auto prepare = [&](ToolCallStreamer& replay) {
                    replay.defer_content_ = false;
                    replay.deferred_.clear();
                    replay.defer_content_keys_ = false;
                };
                auto finish = [&](ToolCallStreamer& replay, const std::string& bytes,
                                  const std::string& prefix, std::string& emitted) {
                    emitted = prefix;
                    std::string scanned;
                    replay.scan(bytes, scanned);
                    replay.sanitized_ += scanned;
                    replay.validate_done();
                    emitted += scanned;
                    std::string replay_tail;
                    const bool complete = replay.finalize(&replay_tail);
                    emitted += replay_tail;
                    return complete;
                };
                auto adopt = [&](ToolCallStreamer& replay) {
                    state = replay.state;
                    invalid_done_ = replay.invalid_done_;
                    trail_ = std::move(replay.trail_);
                    outer_seen_ = replay.outer_seen_;
                    defer_content_ = false;
                    deferred_.clear();
                };

                // Find a provable closing quote for THIS content value. Raw
                // controls and lone backslashes are tolerated for ownership,
                // but unescaped inner quotes remain invalid; this distinguishes
                // `a\nb", "body":...` from raw `{"a":"b"}</content>`.
                const size_t any_tag = pending.find("</content>");
                size_t tag = std::string::npos;
                for (size_t search = any_tag; search != std::string::npos;
                     search = pending.find("</content>", search + 1)) {
                    size_t after = search + 10;
                    while (after < pending.size() && is_ws(pending[after])) after++;
                    if (after == pending.size() || pending[after] == ',' ||
                        pending[after] == '}' || pending[after] == ']') {
                        tag = search;
                        break;
                    }
                }
                size_t first_close_quote = std::string::npos;
                if (any_tag != std::string::npos) {
                    bool escaped = false;
                    size_t quote = std::string::npos;
                    char quote_delimiter = 0;
                    for (size_t i = 0; i < pending.size(); i++) {
                        if (escaped) { escaped = false; continue; }
                        if (pending[i] == '\\') { escaped = true; continue; }
                        if (pending[i] != '"') continue;
                        size_t next = i + 1;
                        while (next < pending.size() && is_ws(pending[next])) next++;
                        if (next < pending.size() &&
                            (pending[next] == ',' || pending[next] == '}' || pending[next] == ']')) {
                            quote = i;
                            quote_delimiter = pending[next];
                            break;
                        }
                    }
                    if (quote != std::string::npos) {
                        if (quote_delimiter == ',') {
                            ToolCallStreamer boundary = *this;
                            prepare(boundary);
                            boundary.recognize_tags_ = false;
                            std::string first_part;
                            boundary.scan(pending.substr(0, quote + 1), first_part);
                            boundary.sanitized_ += first_part;
                            boundary.validate_done();
                            boundary.recognize_tags_ = true;
                            std::string rest;
                            const bool boundary_complete = finish(
                                boundary, pending.substr(quote + 1), "", rest);
                            const size_t trail_start = boundary.trail_.find_first_not_of(" \t\r\n");
                            const bool packed = trail_start != std::string::npos &&
                                boundary.trail_[trail_start] == '{';
                            const bool stranded = boundary.trail_.find("</content>") !=
                                std::string::npos && !packed;
                            if (boundary_complete && !stranded) first_close_quote = quote;
                        } else {
                            std::vector<char> stack;
                            bool in_string = false, prefix_escape = false;
                            for (size_t i = 0; i + 1 < sanitized_.size(); i++) {
                                const char c = sanitized_[i];
                                if (prefix_escape) { prefix_escape = false; continue; }
                                if (in_string) {
                                    if (c == '\\') prefix_escape = true;
                                    else if (c == '"') in_string = false;
                                } else if (c == '"') in_string = true;
                                else if (c == '{' || c == '[') stack.push_back(c);
                                else if ((c == '}' || c == ']') && !stack.empty()) stack.pop_back();
                            }
                            std::string body;
                            body.reserve(quote + 8);
                            for (size_t i = 0; i < quote; i++) {
                                const unsigned char c = (unsigned char)pending[i];
                                if (c == '\\') {
                                    if (i + 1 < quote) {
                                        const char n = pending[i + 1];
                                        if (n == '"' || n == '\\' || n == '/' || n == 'b' || n == 'f' ||
                                            n == 'n' || n == 'r' || n == 't' || n == 'u') {
                                            body += '\\'; body += n; i++; continue;
                                        }
                                    }
                                    body += "\\\\";
                                } else if (c == '\n') body += "\\n";
                                else if (c == '\r') body += "\\r";
                                else if (c == '\t') body += "\\t";
                                else if (c < 0x20) {
                                    char hex[8];
                                    snprintf(hex, sizeof hex, "\\u%04x", c);
                                    body += hex;
                                } else body += (char)c;
                            }
                            std::string probe = sanitized_ + body + "\"";
                            for (auto it = stack.rbegin(); it != stack.rend(); ++it)
                                probe += *it == '{' ? '}' : ']';
                            try {
                                if (json::parse(probe).is_object()) first_close_quote = quote;
                            } catch (...) {}
                        }
                    }
                }
                if (tag != std::string::npos && first_close_quote == std::string::npos) {
                    ToolCallStreamer tagged = *this;
                    prepare(tagged);
                    const std::string escaped = escape_json_interior(pending.substr(0, tag));
                    tagged.sanitized_ += escaped;
                    std::string tagged_out;
                    if (finish(tagged, "\"" + pending.substr(tag + 10), escaped, tagged_out)) {
                        *tail += tagged_out;
                        adopt(tagged);
                        return true;
                    }
                }

                // No sentinel reconstruction completed. Preserve the prior
                // tolerant quote/control repair path without recursive
                // content deferral or duplicate raw buffering.
                ToolCallStreamer ordinary = *this;
                prepare(ordinary);
                std::string ordinary_out;
                bool complete = false;
                if (first_close_quote != std::string::npos) {
                    ordinary.recognize_tags_ = false;
                    std::string first_value;
                    ordinary.scan(pending.substr(0, first_close_quote + 1), first_value);
                    ordinary.sanitized_ += first_value;
                    ordinary.validate_done();
                    ordinary_out += first_value;
                    ordinary.recognize_tags_ = true;
                    std::string remainder;
                    complete = finish(ordinary, pending.substr(first_close_quote + 1), "", remainder);
                    ordinary_out += remainder;
                } else {
                    ordinary.recognize_tags_ = any_tag == std::string::npos || tag != std::string::npos;
                    complete = finish(ordinary, pending, "", ordinary_out);
                }
                *tail += ordinary_out;
                adopt(ordinary);
                return complete;
            }
            if (!tag_.empty()) { *tail += tag_; tag_.clear(); }
            if (pend_q_ || pend_tag_) {
                *tail += '"';
                *tail += pend_ws_;
                pend_q_ = pend_tag_ = false;
            }
        }
        return false;
    }

    // Bytes seen after the streamed call's arguments object closed. A
    // wrapper can pack more than one call; these are NOT framing — the
    // handler runs them through the bare-call recovery chain (review
    // 2026-07-17: the DONE-state byte drop silently lost every call after
    // the first, a regression vs the buffered path).
    const std::string& trail() const { return trail_; }

  private:
    std::string head_;
    std::string trail_;
    std::string sanitized_;
    bool invalid_done_ = false;
    int depth_ = 0;
    int raw_tail_depth_ = 0;
    std::shared_ptr<size_t> raw_replay_work_;
    bool in_str_ = false, esc_ = false, string_is_key_ = false;
    bool tag_raw_ = false;   // <content> opened a raw value; escape until its close tag
    bool defer_raw_tail_ = false;
    std::string raw_tail_;
    bool defer_raw_tails_enabled_ = true;
    bool content_value_ = false;
    std::string current_key_;
    bool defer_content_keys_ = true;
    bool next_content_value_ = false;
    // A quoted `content` value is ambiguous until finalization can prefer a
    // valid literal-tag interpretation over the malformed mode-3 sentinel.
    bool defer_content_ = false;
    std::string deferred_;
    bool recognize_tags_ = true;
    JsonQuoteContext json_ctx_;
    bool pend_q_ = false;   // in-string quote awaiting one-byte lookahead
    bool pend_tag_ = false; // full </content> matched, awaiting the same lookahead
    bool outer_seen_ = false; // the call object's OWN closing } consumed from the trail
    std::string pend_ws_;   // whitespace held behind the pending quote/tag
    std::string tag_;       // partial <content>/</content> capture (mode 3)

    // Post-DONE bytes: the head consumed the call object's opening { without
    // counting it, so exactly one closing } after the args object is the
    // call's own framing — swallow it once; everything else is trail.
    void add_trail(const std::string& s) {
        size_t k = 0;
        if (!outer_seen_) {
            while (k < s.size() && is_ws(s[k])) k++;
            if (k == s.size()) return;      // only ws so far: keep waiting
            outer_seen_ = true;
            if (s[k] == '}') k++;
        }
        trail_ += s.substr(k);
    }

    static bool is_content_key(const std::string& raw) {
        static constexpr char expected[] = "content";
        size_t out = 0;
        for (size_t i = 0; i < raw.size();) {
            unsigned value = static_cast<unsigned char>(raw[i++]);
            if (value == '\\') {
                if (i == raw.size()) return false;
                const char escape = raw[i++];
                if (escape == 'u') {
                    if (raw.size() - i < 4) return false;
                    value = 0;
                    for (int digit = 0; digit < 4; digit++) {
                        const char h = raw[i++];
                        unsigned nibble;
                        if (h >= '0' && h <= '9') nibble = static_cast<unsigned>(h - '0');
                        else if (h >= 'a' && h <= 'f') nibble = static_cast<unsigned>(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') nibble = static_cast<unsigned>(h - 'A' + 10);
                        else return false;
                        value = (value << 4) | nibble;
                    }
                } else {
                    switch (escape) {
                        case '"': value = '"'; break;
                        case '\\': value = '\\'; break;
                        case '/': value = '/'; break;
                        case 'b': value = '\b'; break;
                        case 'f': value = '\f'; break;
                        case 'n': value = '\n'; break;
                        case 'r': value = '\r'; break;
                        case 't': value = '\t'; break;
                        default: return false;
                    }
                }
            }
            if (out >= sizeof(expected) - 1 || value != static_cast<unsigned char>(expected[out]))
                return false;
            out++;
        }
        return out == sizeof(expected) - 1;
    }

    static bool is_ws(char c) { return c==' '||c=='\t'||c=='\r'||c=='\n'; }

    void validate_done() {
        if(state!=DONE) return;
        try {
            json parsed=json::parse(sanitized_);
            if(!parsed.is_object()) throw std::runtime_error("tool arguments are not an object");
        } catch(...) {
            invalid_done_=true;
            // The opener is already on wire, so this is not an ordinary
            // buffered fallback. Preserve all later feeds as recovery trail
            // for packed calls that start in a subsequent token.
            state=INVALID_DONE;
        }
    }

    void scan(const std::string& in, std::string& out) {
        for (size_t i = 0; i < in.size() && state == ARGS; ) {
            const char c = in[i];
            if (defer_raw_tail_) {
                raw_tail_ += c;
                i++;
                continue;
            }
            if (defer_content_) {
                deferred_ += c;
                i++;
                continue;
            }
            if (!tag_.empty()) {
                const char* want = in_str_ ? "</content>" : "<content>";
                const size_t wl = in_str_ ? 10 : 9;
                if (c == want[tag_.size()]) {
                    tag_ += c; i++;
                    if (tag_.size() == wl) {
                        if (in_str_ && tag_raw_ && defer_raw_tails_enabled_) {
                            defer_raw_tail_ = true;
                            raw_tail_ = tag_;
                            tag_.clear();
                        } else {
                            tag_.clear();
                            if (in_str_) pend_tag_ = true;
                            else {
                                out += '"';
                                in_str_ = true;
                                string_is_key_ = false;
                                tag_raw_ = true;
                                content_value_ = false;
                                next_content_value_ = false;
                            }
                        }
                    }
                    continue;
                }
                out += tag_;
                tag_.clear();
                continue;
            }
            if (pend_q_ || pend_tag_) {
                if (is_ws(c)) { pend_ws_ += c; i++; continue; }
                const bool terminates = string_is_key_ ? c == ':'
                    : (c == ',' || c == '}' || c == ']');
                if (terminates) {
                    out += '"'; out += pend_ws_;
                    in_str_ = false;
                    content_value_ = false;
                    tag_raw_ = false;
                    if (string_is_key_)
                        next_content_value_ = is_content_key(current_key_);
                } else {
                    out += pend_tag_ ? "</content>" : "\\\"";
                    for (char w : pend_ws_)
                        out += w=='\n' ? "\\n" : w=='\r' ? "\\r"
                             : w=='\t' ? "\\t" : std::string(1, w);
                }
                pend_ws_.clear();
                pend_q_ = pend_tag_ = false;
            }
            if (esc_) {
                esc_ = false;
                if (string_is_key_) current_key_ += c;
                out += c; i++; continue;
            }
            if (in_str_) {
                if (tag_raw_) {
                    if (recognize_tags_ && c == '<') { tag_ += c; i++; continue; }
                    append_json_escaped_byte(out,(unsigned char)c);
                    i++;
                    continue;
                }
                if (c == '\\') {
                    if (string_is_key_) current_key_ += c;
                    esc_ = true; out += c; i++; continue;
                }
                if (c == '"') { pend_q_ = true; i++; continue; }
                if (recognize_tags_ && content_value_ && c == '<') { tag_ += c; i++; continue; }
                if (string_is_key_) current_key_ += c;
                append_json_escaped_byte(out,(unsigned char)c);
                i++;
                continue;
            }
            if (next_content_value_ && !is_ws(c) && c != ':' && c != '"' && c != '<')
                next_content_value_ = false;
            if (recognize_tags_ && c == '<') { tag_ += c; i++; continue; }
            out += c; i++;
            if (c == '"') {
                string_is_key_ = json_ctx_.opening_string_is_key();
                in_str_ = true;
                if (string_is_key_) current_key_.clear();
                else {
                    content_value_ = next_content_value_;
                    defer_content_ = defer_content_keys_ && content_value_;
                    next_content_value_ = false;
                    deferred_.clear();
                }
            } else {
                json_ctx_.structural(c);
                if (c == '{') depth_++;
                else if (c == '}' && --depth_ == 0) {
                    state = DONE;
                    add_trail(in.substr(i));
                }
            }
        }
    }

    // 1 = matched (name_out set, consumed = index OF the args '{'),
    // 0 = undecided (need more bytes), -1 = not this shape (fallback).
    static int match_head(const std::string& b, std::string& name_out,
                          size_t& consumed) {
        size_t i = 0;
        auto skip = [&]() { while (i < b.size() && is_ws(b[i])) i++; return i < b.size(); };
        auto lit = [&](const char* s) -> int {
            for (size_t k = 0; s[k]; k++, i++) {
                if (i >= b.size()) return 0;
                if (b[i] != s[k]) return -1;
            }
            return 1;
        };
        int r;
        if (!skip()) return 0;
        if ((r = lit("{")) <= 0) return r;
        if (!skip()) return 0;
        if ((r = lit("\"name\"")) <= 0) return r;
        if (!skip()) return 0;
        if ((r = lit(":")) <= 0) return r;
        if (!skip()) return 0;
        if ((r = lit("\"")) <= 0) return r;
        std::string nm;
        for (;; i++) {
            if (i >= b.size()) return 0;
            if (b[i] == '\\') return -1;   // escaped names: buffered path
            if (b[i] == '"') { i++; break; }
            nm += b[i];
        }
        if (nm.empty()) return -1;
        if (!skip()) return 0;
        if ((r = lit(",")) <= 0) return r;
        if (!skip()) return 0;
        if (b[i] == '"') i++;              // mode-9: opening quote optional
        if ((r = lit("arguments\"")) <= 0) return r;
        if (!skip()) return 0;
        if ((r = lit(":")) <= 0) return r;
        if (!skip()) return 0;
        if (b[i] != '{') return -1;        // non-object args: buffered path
        name_out = nm;
        consumed = i;
        return 1;
    }
};

// Experimental harness-prefix prewarming accepts only an initial request:
// zero or more merged system messages followed by exactly one live user
// message. Return the closed static prompt and optionally the complete
// ordinary generation prompt. Token-level callers still verify that encoding
// the former is an exact prefix of encoding the latter before any side effect.
inline std::string initial_harness_prefix(const std::vector<Msg>& messages,
                                          const json& tools, bool think,
                                          std::string* full_prompt = nullptr,
                                          const std::string& tool_instruction = {},
                                          const json* unavailable_tools = nullptr) {
    if (messages.empty() || messages.back().role != "user")
        throw std::runtime_error(
            "prewarm requires an initial request ending in one user message");
    std::vector<Msg> prefix_messages(messages.begin(), messages.end() - 1);
    for (const auto& message : prefix_messages)
        if (message.role != "system")
            throw std::runtime_error(
                "prewarm accepts initial requests only (system plus final user)");
    if (full_prompt) *full_prompt=chatml_prompt(
        messages,tools,think,nullptr,nullptr,tool_instruction,unavailable_tools);
    size_t stable_bytes = 0;
    std::string prefix=chatml_prompt(
        prefix_messages,tools,think,&stable_bytes,nullptr,tool_instruction,
        unavailable_tools);
    prefix.resize(stable_bytes);
    return prefix;
}

inline std::string anthropic_error_json(const std::string& err_type,
                                        const std::string& message) {
    json e = {{"type", "error"},
              {"error", {{"type", err_type}, {"message", message}}}};
    return e.dump(-1, ' ', false, json::error_handler_t::replace);
}

// The real API's context-limit message, byte-for-byte format. CC (2.1.x)
// treats "prompt is too long" as compact-now; anything else (including our
// old end=refused empty 200) is retried verbatim and loops.
inline std::string ctx_limit_error_message(int n_prompt, int n_max_prompt) {
    return "prompt is too long: " + std::to_string(n_prompt) + " tokens > " +
           std::to_string(n_max_prompt) + " maximum";
}

// Anthropic tools -> qwen tools json for the system preamble (the
// /v1/messages request mapping; count_tokens must count the same bytes).
inline json anthropic_tools_json(const json& body) {
    json out = json::array();
    if (body.contains("tools") && body["tools"].is_array())
        for (auto& t : body["tools"]) {
            if (!t.is_object() || !t.contains("name") || !t["name"].is_string()) continue;
            const std::string& name=t["name"].get_ref<const std::string&>();
            if(name.empty()) continue;
            const std::string description=
                t.contains("description") && t["description"].is_string()
                    ? t["description"].get<std::string>() : "";
            const json parameters=
                t.contains("input_schema") && t["input_schema"].is_object()
                    ? t["input_schema"] : json::object();
            out.push_back({{"type", "function"},
                           {"function", {{"name", name},
                                         {"description", description},
                                         {"parameters", parameters}}}});
        }
    return out;
}

// Anthropic messages -> Msg list (thinking + tool_use reconstructed to
// model markers, tool_result wrapped in <tool_response>)
inline std::vector<Msg> anthropic_msgs(const json& body) {
    std::vector<Msg> msgs;
    if (body.contains("system")) {
        std::string sys;
        if (body["system"].is_string()) sys = body["system"];
        else if (body["system"].is_array())
            for (auto& b : body["system"])
                // is_object() FIRST: value() on a non-object throws 306, which
                // httplib reports as a 500 -- a bare string in the array is a
                // client-side shape error, not a server fault (same guard
                // openai_msgs has always had on its content parts).
                if (b.is_object() && b.value("type", "") == "text") sys += b.value("text", "");
        if (!sys.empty()) {
            normalize_cc_billing_header(sys);
            msgs.push_back({"system", sys});
        }
    }
    if (!body.contains("messages")) return msgs;
    for (auto& m : body["messages"]) {
        // is_object() BEFORE the first value() call: value() on a non-object
        // (messages:["hi"]) throws 306 -- which the old ordering did one line
        // ahead of the guard meant to prevent exactly that. A non-object
        // element is skipped, matching openai_msgs.
        if (!m.is_object()) continue;
        std::string role = m.value("role", "user"), think, content;
        // guard: const operator[] on a missing key is an abort (json.hpp
        // assertion) -- a content-less message must not kill the server
        if (!m.contains("content")) { msgs.push_back({role, content}); continue; }
        if (m["content"].is_string()) content = m["content"];
        else if (m["content"].is_array())
            for (auto& part : m["content"]) {
                if (!part.is_object()) continue; // bare string in a content array
                std::string ty = part.value("type", "");
                if (ty == "text") content += part.value("text", "");
                else if (ty == "thinking") think += part.value("thinking", "");
                else if (ty == "tool_use") {
                    if (!content.empty() && content.back() != '\n') content += "\n";
                    content += tool_call_text(part.value("name", ""),
                                              part.contains("input") ? part["input"]
                                                                     : json::object());
                } else if (ty == "tool_result") {
                    std::string rc;
                    if (part.contains("content")) {
                        if (part["content"].is_string()) rc = part["content"];
                        else if (part["content"].is_array())
                            for (auto& b : part["content"])
                                if (b.is_object() && b.value("type", "") == "text")
                                    rc += b.value("text", "");
                    }
                    if (!content.empty() && content.back() != '\n') content += "\n";
                    content += tool_response_text(rc);
                }
            }
        if (role == "assistant" && !think.empty())
            content = "<think>\n" + think + "\n</think>\n" + content;
        msgs.push_back({role, content});
    }
    return msgs;
}

// OpenAI chat/completions tools -> the same {"type":"function","function":{...}}
// shape tools_preamble/chatml_prompt expect. Unlike anthropic_tools_json this
// is nearly a pass-through (the wire shape already matches); entries missing
// "type":"function" or a function.name are dropped rather than failing the
// whole request, mirroring the Responses bridge's tolerance of hosted tool
// types it doesn't model (review parity: a malformed ONE tool must not take
// down an otherwise-valid request).
inline json openai_tools_json(const json& body) {
    json out = json::array();
    if (body.contains("tools") && body["tools"].is_array())
        for (const auto& t : body["tools"]) {
            if (!t.is_object() || !t.contains("type") || !t["type"].is_string() ||
                t["type"] != "function") continue;
            if (!t.contains("function") || !t["function"].is_object()) continue;
            const json& fn = t["function"];
            if (!fn.contains("name") || !fn["name"].is_string() ||
                fn["name"].get_ref<const std::string&>().empty()) continue;
            if (fn.contains("description") && !fn["description"].is_string()) continue;
            if (fn.contains("parameters") && !fn["parameters"].is_object()) continue;
            out.push_back({{"type", "function"},
                           {"function", {{"name", fn["name"]},
                                         {"description", fn.contains("description")
                                                             ? fn["description"] : json("")},
                                         {"parameters", fn.contains("parameters")
                                                            ? fn["parameters"]
                                                            : json::object()}}}});
        }
    return out;
}

// OpenAI chat/completions messages -> Msg list, the /v1/chat/completions twin
// of anthropic_msgs. Two bridges the flat "content is a string" reading
// misses entirely (silently dropping the model's own tool use from history,
// which breaks any multi-turn agentic loop after the first call):
//   - assistant.tool_calls[] (OpenAI shape: function.arguments is a JSON
//     STRING) -> reconstructed <tool_call> marker(s), appended after any
//     sibling content text (order matches anthropic_msgs' text-then-tool_use
//     handling).
//   - role:"tool" (tool_call_id + content) -> folded into a <tool_response>-
//     wrapped USER turn, same as anthropic_msgs' tool_result bridge. The
//     call_id is intentionally not echoed into the prompt text: the chat
//     template's <tool_response> carries no id, and the fine-tune associates
//     a result with the immediately preceding call by POSITION.
//   - role:"developer" (the newer OpenAI system-role alias) -> "system",
//     matching the /v1/responses bridge.
inline std::vector<Msg> openai_msgs(const json& body) {
    std::vector<Msg> msgs;
    if (!body.contains("messages") || !body["messages"].is_array()) return msgs;
    for (auto& m : body["messages"]) {
        if (!m.is_object()) continue;
        std::string role = m.value("role", "user");
        if (role == "developer") role = "system";
        std::string content;
        if (m.contains("content")) {
            if (m["content"].is_string()) content = m["content"];
            else if (m["content"].is_array())
                for (auto& part : m["content"])
                    if (part.is_object() && part.value("type", "") == "text")
                        content += part.value("text", "");
        }
        if (role == "tool") {
            msgs.push_back({"user", tool_response_text(content)});
            continue;
        }
        if (role == "assistant" && m.contains("tool_calls") && m["tool_calls"].is_array()) {
            for (auto& tc : m["tool_calls"]) {
                if (!tc.is_object() || !tc.contains("function") || !tc["function"].is_object())
                    continue;
                const json& fn = tc["function"];
                std::string name = fn.value("name", std::string());
                json args = json::object();
                if (fn.contains("arguments")) {
                    if (fn["arguments"].is_string()) {
                        // OpenAI wire shape: a JSON-encoded string. Keep the
                        // raw string (rather than dropping the call) if it
                        // fails to parse -- same "never lose a turn" stance
                        // as parse_tool_call's double-encode tolerance.
                        try { args = json::parse(fn["arguments"].get<std::string>()); }
                        catch (...) { args = fn["arguments"]; }
                    } else args = fn["arguments"];
                }
                if (!content.empty() && content.back() != '\n') content += "\n";
                content += tool_call_text(name, args);
            }
        }
        msgs.push_back({role, content});
    }
    return msgs;
}

// tool_choice (OpenAI shape): "auto"/absent -> AUTO; "none" -> NONE;
// "required", a named function, or allowed_tools mode:"required" -> FORCED.
// allowed_tools mode:"auto" keeps AUTO while narrowing the eligible registry.
struct ToolChoice {
    enum Mode { AUTO, NONE, FORCED } mode = AUTO;
    std::string forced_name; // non-empty only for a named function choice
    std::vector<std::string> allowed_names; // empty = every declared tool
    bool disable_parallel_tool_use = false;
    bool invalid = false;
};

inline bool forced_tool_choice_missing_is_error(const ToolChoice& choice,
                                                bool has_eligible_call,
                                                bool generation_truncated) {
    return choice.mode == ToolChoice::FORCED && !has_eligible_call &&
           !generation_truncated;
}

inline const char* openai_tool_finish_reason(bool has_eligible_call,
                                             bool final_tool_incomplete,
                                             bool generation_truncated) {
    if(has_eligible_call && !final_tool_incomplete) return "tool_calls";
    return generation_truncated ? "length" : "stop";
}

inline const char* anthropic_tool_stop_reason(bool has_eligible_call,
                                               bool final_tool_incomplete,
                                               bool generation_truncated) {
    if(has_eligible_call && !final_tool_incomplete) return "tool_use";
    return generation_truncated ? "max_tokens" : "end_turn";
}

// OpenAI's top-level parallel_tool_calls=false has the same output
// multiplicity contract as Anthropic's disable_parallel_tool_use=true.
// Keep it on ToolChoice so wrapped, recovered, streaming, and non-streaming
// calls all pass through one eligibility rule.
inline void apply_openai_parallel_tool_calls(const json& body,ToolChoice& choice) {
    if(!body.contains("parallel_tool_calls")) return;
    if(!body["parallel_tool_calls"].is_boolean()) {
        choice.invalid=true;
        return;
    }
    choice.disable_parallel_tool_use=!body["parallel_tool_calls"].get<bool>();
}

template<class NameSet>
inline bool tool_choice_allows_call(const ToolChoice& choice,const NameSet& allowed,
                                    const std::string& name,size_t accepted_calls) {
    return choice.mode != ToolChoice::NONE && allowed.count(name) &&
        (choice.forced_name.empty() || name == choice.forced_name) &&
        (!choice.disable_parallel_tool_use || accepted_calls == 0);
}
inline ToolChoice parse_tool_choice(const json& body) {
    ToolChoice tc;
    if (!body.contains("tool_choice")) return tc;
    const json& v = body["tool_choice"];
    if (v.is_string()) {
        if (v == "none") tc.mode = ToolChoice::NONE;
        else if (v == "required") tc.mode = ToolChoice::FORCED;
        // "auto" or any other/unknown string: default AUTO
        return tc;
    }
    if (v.is_null()) return tc;
    if (!v.is_object() || !v.contains("type") || !v["type"].is_string()) {
        tc.invalid = true;
        return tc;
    }
    if (v["type"] == "function") {
        if (!v.contains("function") || !v["function"].is_object() ||
            !v["function"].contains("name") || !v["function"]["name"].is_string() ||
            v["function"]["name"].get_ref<const std::string&>().empty()) {
            tc.invalid = true;
            return tc;
        }
        tc.mode = ToolChoice::FORCED;
        tc.forced_name = v["function"]["name"].get<std::string>();
        tc.allowed_names.push_back(tc.forced_name);
        return tc;
    }
    if (v["type"] == "allowed_tools") {
        if (!v.contains("allowed_tools") || !v["allowed_tools"].is_object()) {
            tc.invalid = true;
            return tc;
        }
        const json& allowed = v["allowed_tools"];
        if (!allowed.contains("mode") || !allowed["mode"].is_string() ||
            (allowed["mode"] != "auto" && allowed["mode"] != "required") ||
            !allowed.contains("tools") || !allowed["tools"].is_array()) {
            tc.invalid = true;
            return tc;
        }
        tc.mode = allowed["mode"] == "required" ? ToolChoice::FORCED : ToolChoice::AUTO;
        for (const auto& tool : allowed["tools"]) {
            if (!tool.is_object() || !tool.contains("type") || !tool["type"].is_string() ||
                tool["type"] != "function" || !tool.contains("function") ||
                !tool["function"].is_object() || !tool["function"].contains("name") ||
                !tool["function"]["name"].is_string() ||
                tool["function"]["name"].get_ref<const std::string&>().empty()) {
                tc.invalid = true;
                return tc;
            }
            const std::string name=tool["function"]["name"].get<std::string>();
            if (std::find(tc.allowed_names.begin(),tc.allowed_names.end(),name)==tc.allowed_names.end())
                tc.allowed_names.push_back(name);
        }
        if (tc.allowed_names.empty()) tc.invalid = true;
        return tc;
    }
    tc.invalid = true;
    return tc;
}

// Anthropic tool_choice: absent/auto -> AUTO, none -> NONE, any -> FORCED
// across the declared registry, and tool{name} -> FORCED for that one tool.
inline ToolChoice parse_anthropic_tool_choice(const json& body) {
    ToolChoice tc;
    if (!body.contains("tool_choice") || body["tool_choice"].is_null()) return tc;
    const json& v = body["tool_choice"];
    if (!v.is_object() || !v.contains("type") || !v["type"].is_string()) {
        tc.invalid = true;
        return tc;
    }
    if (v.contains("disable_parallel_tool_use")) {
        if (!v["disable_parallel_tool_use"].is_boolean()) {
            tc.invalid = true;
            return tc;
        }
        tc.disable_parallel_tool_use = v["disable_parallel_tool_use"].get<bool>();
    }
    const std::string type = v["type"].get<std::string>();
    if (type == "auto") return tc;
    if (type == "none") {
        tc.mode = ToolChoice::NONE;
        return tc;
    }
    if (type == "any") {
        tc.mode = ToolChoice::FORCED;
        return tc;
    }
    if (type == "tool") {
        if (!v.contains("name") || !v["name"].is_string() ||
            v["name"].get_ref<const std::string&>().empty()) {
            tc.invalid = true;
            return tc;
        }
        tc.mode = ToolChoice::FORCED;
        tc.forced_name = v["name"].get<std::string>();
        tc.allowed_names.push_back(tc.forced_name);
        return tc;
    }
    tc.invalid = true;
    return tc;
}

// Anthropic's incompatibility is request-scoped: an explicit thinking block
// cannot coexist with forced tool_choice. A server-side --think default is a
// local fallback, so forced choice may override it when the request is silent.
inline void validate_anthropic_tool_choice_thinking(
        const ToolChoice& choice, const ThinkCfg& thinking) {
    if (choice.mode == ToolChoice::FORCED && thinking.enabled && thinking.enabled_set)
        throw std::invalid_argument(
            "thinking.type: enabled is incompatible with forced tool_choice");
}

inline std::string anthropic_tool_choice_instruction(const ToolChoice& choice) {
    if (choice.mode == ToolChoice::NONE)
        return "Tool use is disabled for this response. Do not call any tool.";
    if (choice.mode != ToolChoice::FORCED) return {};
    if (choice.forced_name.empty())
        return "You must call at least one of the available tools before ending the response.";
    return "You must call only the tool named " + json(choice.forced_name).dump() +
           " before ending the response.";
}

inline void validate_responses_tool_fields(const json& body) {
    if(!body.contains("tools") || !body["tools"].is_array()) return;
    for(const auto& tool:body["tools"]) {
        if(!tool.is_object()) continue;
        auto require_string=[&](const char* field) {
            const auto it=tool.find(field);
            if(it!=tool.end() && !it->is_string())
                throw std::invalid_argument(
                    std::string("tools[].")+field+" must be a string");
        };
        require_string("type");
        const std::string type=jstr(tool,"type");
        if(type=="function" || type=="custom") {
            require_string("name");
            require_string("description");
        }
    }
}

// Legacy Codex clients can advertise their locally executed shell capability
// as the hosted Responses type `shell`, while still consuming ordinary
// function_call items. Translate that compatibility form into the two
// model-facing functions the client executes; true hosted execution remains
// the client's responsibility, not this inference server's.
inline json responses_shell_prompt_tools() {
    return json::array({
        {{"type","function"},{"function",{
            {"name","exec_command"},
            {"description","Runs a shell command and returns output or a session ID for ongoing interaction."},
            {"parameters",{{"type","object"},{"properties",{
                {"cmd",{{"type","string"},{"description","Shell command to execute."}}},
                {"workdir",{{"type","string"},{"description","Working directory for the command."}}},
                {"tty",{{"type","boolean"},{"description","Whether to allocate a PTY."}}},
                {"yield_time_ms",{{"type","number"},{"description","Wait before yielding output."}}},
                {"max_output_tokens",{{"type","number"},{"description","Output token budget."}}},
                {"shell",{{"type","string"},{"description","Shell binary to launch."}}}
            }},{"required",json::array({"cmd"})},{"additionalProperties",false}}}
        }}},
        {{"type","function"},{"function",{
            {"name","write_stdin"},
            {"description","Writes characters to an existing command session and returns recent output."},
            {"parameters",{{"type","object"},{"properties",{
                {"session_id",{{"type","number"},{"description","Identifier of the running command session."}}},
                {"chars",{{"type","string"},{"description","Bytes to write; empty polls without writing."}}},
                {"yield_time_ms",{{"type","number"},{"description","Wait before yielding output."}}},
                {"max_output_tokens",{{"type","number"},{"description","Output token budget."}}}
            }},{"required",json::array({"session_id"})},{"additionalProperties",false}}}
        }}}
    });
}

template<class NameSet>
inline void add_responses_hosted_call_names(NameSet& names,
                                            const std::string& hosted_type) {
    if (hosted_type != "shell") return;
    names.insert("exec_command");
    names.insert("write_stdin");
}

inline bool responses_tool_names_ambiguous(
    const std::set<std::string>& function_names,
    const std::set<std::string>& custom_names,
    const std::set<std::string>& hosted_tool_types) {
    std::set<std::string> hosted_call_names;
    for (const auto& type : hosted_tool_types)
        add_responses_hosted_call_names(hosted_call_names, type);
    for (const auto& name : function_names)
        if (custom_names.count(name) || hosted_tool_types.count(name) ||
            hosted_call_names.count(name))
            return true;
    for (const auto& name : custom_names)
        if (hosted_tool_types.count(name) || hosted_call_names.count(name))
            return true;
    return false;
}
inline void validate_responses_tool_choice_declarations(
    const json& body,
    const std::set<std::string>& function_names,
    const std::set<std::string>& custom_names,
    const std::set<std::string>& hosted_tool_types) {
    auto validate_tool = [&](const json& tool) {
        if (!tool.is_object()) return;
        const std::string type = jstr(tool, "type");
        if (type == "function" || type == "custom") {
            const std::string name = jstr(tool, "name");
            if (name.empty()) return;
            const bool declared = type == "function" ? function_names.count(name)
                                                       : custom_names.count(name);
            if (!declared)
                throw std::runtime_error("tool_choice kind/name not present in tools");
        } else if (!type.empty() && type != "allowed_tools" && type != "mcp" &&
                   !hosted_tool_types.count(type)) {
            throw std::runtime_error("tool_choice hosted type not present in tools");
        }
    };
    if (!body.contains("tool_choice") || !body["tool_choice"].is_object()) return;
    const json& source = body["tool_choice"];
    if (jstr(source, "type") != "allowed_tools") {
        validate_tool(source);
        return;
    }
    const json* allowed = &source;
    if (source.contains("allowed_tools") && source["allowed_tools"].is_object())
        allowed = &source["allowed_tools"];
    if (allowed->contains("tools") && (*allowed)["tools"].is_array())
        for (const auto& tool : (*allowed)["tools"]) validate_tool(tool);
}

inline ToolChoice responses_registered_tool_choice(
    const ToolChoice& choice, const std::set<std::string>& hosted_tool_types) {
    ToolChoice registered = choice;
    auto mapped_names = [](const std::string& type) {
        std::set<std::string> names;
        add_responses_hosted_call_names(names, type);
        return names;
    };
    if (!registered.forced_name.empty() &&
        hosted_tool_types.count(registered.forced_name)) {
        auto mapped = mapped_names(registered.forced_name);
        registered.forced_name.clear();
        registered.allowed_names.assign(mapped.begin(), mapped.end());
        if (registered.allowed_names.empty()) registered.mode = ToolChoice::NONE;
    } else if (!registered.allowed_names.empty()) {
        std::set<std::string> expanded;
        for (const auto& name : registered.allowed_names) {
            if (hosted_tool_types.count(name)) {
                auto mapped = mapped_names(name);
                expanded.insert(mapped.begin(), mapped.end());
            } else {
                expanded.insert(name);
            }
        }
        registered.allowed_names.assign(expanded.begin(), expanded.end());
        if (registered.allowed_names.empty()) registered.mode = ToolChoice::NONE;
    }
    return registered;
}


// Responses API names function/custom tools directly in object-form
// tool_choice, unlike Chat Completions' nested `function.name` shape.
// Normalize that wire form into ToolChoice so both APIs share selection and
// output-validation semantics.
inline ToolChoice parse_responses_tool_choice(const json& body) {
    if (body.contains("tool_choice") && body["tool_choice"].is_string() &&
        body["tool_choice"] != "auto" && body["tool_choice"] != "none" &&
        body["tool_choice"] != "required") {
        ToolChoice invalid;
        invalid.invalid = true;
        return invalid;
    }
    if (!body.contains("tool_choice") || !body["tool_choice"].is_object())
        return parse_tool_choice(body);
    json normalized = body;
    const json& source = body["tool_choice"];
    const std::string type = source.value("type", std::string());
    if ((type == "function" || type == "custom") && source.contains("name")) {
        normalized["tool_choice"] = {{"type","function"},
                                     {"function",{{"name",source["name"]}}}};
    } else if (type == "shell") {
        normalized["tool_choice"] = {{"type","function"},
                                     {"function",{{"name",type}}}};
    } else if (type == "allowed_tools") {
        json allowed;
        if (source.contains("allowed_tools") && source["allowed_tools"].is_object())
            allowed=source["allowed_tools"];
        else {
            if (source.contains("mode")) allowed["mode"]=source["mode"];
            if (source.contains("tools")) allowed["tools"]=source["tools"];
        }
        bool skipped_unavailable=false;
        if (allowed.contains("tools") && allowed["tools"].is_array()) {
            json tools = json::array();
            const bool optional=allowed.value("mode",std::string())=="auto";
            for (const auto& tool : allowed["tools"]) {
                const std::string tool_type = tool.is_object()
                    ? tool.value("type",std::string()) : std::string();
                if (tool.is_object() && tool.contains("name") &&
                    (tool_type == "function" || tool_type == "custom"))
                    tools.push_back({{"type","function"},
                                     {"function",{{"name",tool["name"]}}}});
                else if (tool_type == "shell")
                    tools.push_back({{"type","function"},
                                     {"function",{{"name",tool_type}}}});
                else if(optional && !tool_type.empty())
                    skipped_unavailable=true;
                else tools.push_back(tool);
            }
            if(skipped_unavailable && tools.empty()) {
                ToolChoice unavailable;
                unavailable.mode=ToolChoice::NONE;
                return unavailable;
            }
            allowed["tools"] = std::move(tools);
        }
        normalized["tool_choice"] = {{"type","allowed_tools"},
                                     {"allowed_tools",std::move(allowed)}};
    }
    return parse_tool_choice(normalized);
}

inline bool openai_stream_includes_usage(const json& body) {
    if (!jbool(body,"stream",false) || !body.contains("stream_options") ||
        !body["stream_options"].is_object()) return false;
    const json& options=body["stream_options"];
    return options.contains("include_usage") && options["include_usage"].is_boolean() &&
           options["include_usage"].get<bool>();
}

struct OpenAIToolSelection {
    json tools=json::array();
    std::vector<std::string> names;
};

// Normalize the declared registry once, then apply tool_choice's eligible
// subset. Both backends use this helper so prompt injection, grammar names,
// fallback parsing, and output validation all share the same registry.
inline OpenAIToolSelection select_openai_tools(const json& body,const ToolChoice& choice) {
    if (choice.invalid) throw std::runtime_error("invalid tool_choice or parallel_tool_calls");
    OpenAIToolSelection selected;
    if (choice.mode == ToolChoice::NONE) return selected;
    selected.tools=openai_tools_json(body);
    for (const auto& tool : selected.tools)
        selected.names.push_back(tool["function"]["name"].get<std::string>());
    if (!choice.allowed_names.empty()) {
        const std::set<std::string> declared(selected.names.begin(),selected.names.end());
        for (const auto& name : choice.allowed_names)
            if (!declared.count(name))
                throw std::runtime_error("tool_choice names a function not present in tools");
        const std::set<std::string> allowed(choice.allowed_names.begin(),choice.allowed_names.end());
        json filtered=json::array();
        for (auto& tool : selected.tools)
            if (allowed.count(tool["function"]["name"].get<std::string>()))
                filtered.push_back(std::move(tool));
        selected.tools=std::move(filtered);
        selected.names=choice.allowed_names;
    }
    if (choice.mode == ToolChoice::FORCED && selected.names.empty())
        throw std::runtime_error("tool_choice requires at least one valid function tool");
    return selected;
}

// Anthropic counts every declaration even when tool_choice narrows eligibility.
// The inactive block keeps that accounting while leaving only selected schemas
// in the callable interface. This choice-specific system prompt is intentional:
// Anthropic documents that changing tool_choice invalidates message-block cache
// entries (tool definitions are a separate hosted cache boundary).
inline json unselected_openai_tools(const json& all_tools,
                                    const OpenAIToolSelection& selected) {
    const std::set<std::string> active(selected.names.begin(),selected.names.end());
    json unavailable=json::array();
    for (const auto& tool : all_tools)
        if (!active.count(tool["function"]["name"].get<std::string>()))
            unavailable.push_back(tool);
    return unavailable;
}

// Parsed model tool call. `ok` false if the JSON was malformed (raw kept).
struct ToolCall {
    bool ok = false;
    std::string name;
    json arguments;
    std::string raw;
    size_t source_begin=std::string::npos;
    size_t source_end=std::string::npos;
    size_t rewritten_begin=std::string::npos;
    size_t rewritten_end=std::string::npos;
};

template<class NameSet>
inline bool tool_choice_allows_all_calls(const ToolChoice& choice,
                                         const NameSet& allowed,
                                         const std::vector<ToolCall>& calls,
                                         size_t accepted_calls=0) {
    if (calls.empty()) return false;
    for (const auto& call : calls) {
        if (!call.ok || !tool_choice_allows_call(
                choice, allowed, call.name, accepted_calls))
            return false;
        accepted_calls++;
    }
    return true;
}
template<class NameSet>
inline bool responses_tool_tail_after_bare_calls(
    bool current_tool_tail, const std::string& text,
    const std::vector<ToolCall>& calls, const ToolChoice& choice,
    const NameSet& allowed, size_t accepted_calls=0) {
    if (calls.empty())
        return current_tool_tail &&
               text.find_first_not_of(" \t\r\n") == std::string::npos;
    bool tool_tail = false;
    size_t tail_begin = 0;
    for (const auto& call : calls) {
        if (call.ok && tool_choice_allows_call(
                           choice, allowed, call.name, accepted_calls)) {
            tool_tail = true;
            accepted_calls++;
        } else {
            tool_tail = false;
        }
        if (call.source_end != std::string::npos && call.source_end <= text.size())
            tail_begin = call.source_end;
    }
    if (text.find_first_not_of(" \t\r\n", tail_begin) != std::string::npos)
        tool_tail = false;
    return tool_tail;
}

inline std::string escape_content_tags(const std::string& text);

// Q27_TOOL_STRICT=1: disable EVERY tolerant-parser rescue (the strict-parser
// A/B knob). Wrapped calls must be plain valid JSON (no <content>-tag rewrite,
// no double-encode unwrap); the wrapper-less bare-scan recovery is suppressed
// entirely. Suppressed rescues are logged ([q27-strict]) so a campaign can
// count what the tolerant chain WOULD have carried. Read once (server-lifetime
// knob, one leg per server run).
inline bool tool_strict() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("Q27_TOOL_STRICT"); v = e ? atoi(e) : 0; }
    return v == 1;
}

inline ToolCall parse_tool_call(const std::string& seg) {
    ToolCall tc;
    tc.raw = seg;
    if (tool_strict()) {
        // strict: the wrapped segment must parse as-is, with a JSON-object
        // arguments value. Anything else stays text (rescue suppressed).
        try {
            json j = json::parse(seg);
            tc.name = j.value("name", std::string());
            tc.arguments = j.contains("arguments") ? j["arguments"] : json::object();
            if (!tc.arguments.is_object()) {
                fprintf(stderr,
                        tc.arguments.is_string()
                            ? "[q27-strict] rejected double-encoded arguments (tool=%s)\n"
                            : "[q27-strict] rejected non-object arguments (tool=%s)\n",
                        tc.name.c_str());
                tc.ok = false;
                return tc;
            }
            tc.ok = !tc.name.empty();
        } catch (...) {
            tc.ok = false;
            if (seg.find("<content>") != std::string::npos ||
                seg.find("</content>") != std::string::npos)
                fprintf(stderr, "[q27-strict] rejected <content>-tagged call (mode 3)\n");
        }
        return tc;
    }
    try {
        json j = json::parse(escape_content_tags(seg));
        tc.name = j.value("name", std::string());
        tc.arguments = j.contains("arguments") ? j["arguments"] : json::object();
        if (tc.arguments.is_string()) // some models double-encode
            tc.arguments = json::parse(tc.arguments.get<std::string>());
        tc.ok = !tc.name.empty() && tc.arguments.is_object();
    } catch (...) { tc.ok = false; }
    return tc;
}

inline std::string strip_ws2(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Streaming Responses must withhold a possible wrapper-less call until it is
// classified, but arbitrary Markdown/JSON must keep flowing. Limit holdback
// to object prefixes whose first key can identify a supported call shape.
inline bool plausible_bare_tool_prefix_range(
    const std::string& text,size_t begin,size_t end) {
    end=std::min(end,text.size());
    if(begin>=end || text[begin]!='{') return false;
    size_t p=begin+1;
    while(p<end && (text[p]==' ' || text[p]=='\t' ||
                    text[p]=='\r' || text[p]=='\n')) p++;
    if(p==end) return true;
    static const char* keys[]={
        "\"name\"","\"arguments\"","\"function\"",
        "\"tool\"","\"tool_name\"","\"tool_call\""};
    const size_t available=end-p;
    for(const char* key:keys) {
        const size_t key_size=std::char_traits<char>::length(key);
        const size_t compared=std::min(available,key_size);
        if(text.compare(p,compared,key,compared)==0) return true;
    }
    return false;
}

inline bool plausible_bare_tool_prefix(const std::string& text) {
    return plausible_bare_tool_prefix_range(text,0,text.size());
}


inline void consume_bare_text_context(JsonStringLexState& string_state,
                                      MarkdownFenceLexState& fence_state,
                                      char ch) {
    consume_display_text_context(string_state,fence_state,ch);
}

inline void consume_bare_text_context(JsonStringLexState& string_state,
                                      MarkdownFenceLexState& fence_state,
                                      const std::string& text,
                                      size_t count=std::string::npos) {
    consume_display_text_context(string_state,fence_state,text,count);
}

inline bool bare_markdown_context_is_displayed(
    const std::string& text,size_t position,
    MarkdownFenceLexState& fence_state,bool end_is_final) {
    return markdown_context_is_displayed(
        text,position,fence_state,end_is_final);
}


inline bool bare_context_is_executable(
    const std::string& text,size_t position,
    JsonStringLexState& string_state,MarkdownFenceLexState& fence_state,
    bool end_is_final) {
    if(position<text.size()) string_state.settle_pending(text[position]);
    if(string_state.in_inert_container() ||
       !display_text_context_is_executable(
           text,position,string_state,fence_state,end_is_final)) return false;
    string_state.discard_pending_containers();
    return true;
}

inline bool bare_text_position_is_executable(
    const std::string& text,size_t position,
    JsonStringLexState string_state={},
    MarkdownFenceLexState fence_state={},bool end_is_final=true) {
    consume_bare_text_context(string_state,fence_state,text,position);
    return bare_context_is_executable(
        text,position,string_state,fence_state,end_is_final);
}

inline bool bare_position_is_displayed(const std::string& text,size_t position) {
    JsonStringLexState string_state;
    MarkdownFenceLexState fence_state;
    consume_bare_text_context(string_state,fence_state,text,position);
    return bare_markdown_context_is_displayed(
        text,position,fence_state,/*end_is_final=*/true);
}

inline size_t bare_object_position(const std::string& text,
                                   JsonStringLexState string_state={},
                                   MarkdownFenceLexState fence_state={},
                                   bool end_is_final=true) {
    for(size_t i=0;i<text.size();i++) {
        if(text[i]=='{' && bare_context_is_executable(
               text,i,string_state,fence_state,end_is_final)) return i;
        consume_bare_text_context(string_state,fence_state,text[i]);
    }
    return std::string::npos;
}

inline size_t bare_unresolved_inline_probe_start(
    const std::string& text,JsonStringLexState string_state={},
    MarkdownFenceLexState fence_state={}) {
    consume_bare_text_context(string_state,fence_state,text);
    fence_state.settle_before_non_backtick();
    return fence_state.fence_length==0 && fence_state.inline_ticks!=0
        ?0:std::string::npos;
}

// Return the byte after the first balanced top-level object, or npos while
// the candidate is incomplete. Braces inside JSON strings do not count.
struct IncrementalBareJsonEnd {
    size_t cursor=0;
    int depth=0;
    bool in_string=false;
    bool escaped=false;

    void begin(bool dropped_opener,size_t start=0) {
        cursor=start;
        depth=dropped_opener?1:0;
        // Mode 10 starts after a synthesized {"name":" prefix.
        in_string=dropped_opener;
        escaped=false;
    }

    size_t advance(const std::string& text) {
        while(cursor<text.size()) {
            const char ch=text[cursor++];
            if(escaped) { escaped=false; continue; }
            if(in_string) {
                if(ch=='\\') escaped=true;
                else if(ch=='"') in_string=false;
                continue;
            }
            if(ch=='"') in_string=true;
            else if(ch=='{') depth++;
            else if(ch=='}') {
                if(depth<=0) return std::string::npos;
                if(--depth==0) return cursor;
            }
        }
        return std::string::npos;
    }
};

inline size_t balanced_json_object_prefix_end(const std::string& text) {
    if(text.empty() || text[0]!='{') return std::string::npos;
    IncrementalBareJsonEnd scan;
    scan.begin(false);
    return scan.advance(text);
}

template<class NameSet>
inline size_t bare_mode10_signature_position(
    const std::string& text,const NameSet& names,
    JsonStringLexState string_state={},
    MarkdownFenceLexState fence_state={},bool end_is_final=true) {
    size_t first=std::string::npos;
    for(const auto& name:names) {
        if(name.empty()) continue;
        const std::string signature=name+"\", \"";
        for(size_t p=text.find(signature);p!=std::string::npos;
            p=text.find(signature,p+1)) {
            if(!bare_text_position_is_executable(
                   text,p,string_state,fence_state,end_is_final)) continue;
            first=std::min(first,p);
            break;
        }
    }
    return first;
}

// Start of the longest suffix that can still become a dropped-opener
// signature. Quoted and fenced suffixes are ordinary displayed content.
template<class NameSet>
inline size_t bare_mode10_probe_start(
    const std::string& text,const NameSet& names,
    JsonStringLexState string_state={},
    MarkdownFenceLexState fence_state={},bool end_is_final=true) {
    size_t best=std::string::npos;
    for(const auto& name:names) {
        if(name.empty()) continue;
        const std::string signature=name+"\", \"";
        const size_t cap=std::min(text.size(),signature.size()-1);
        for(size_t len=cap;len;--len) {
            const size_t start=text.size()-len;
            if(text.compare(start,len,signature,0,len)!=0) continue;
            if(!bare_text_position_is_executable(
                   text,start,string_state,fence_state,end_is_final)) break;
            best=best==std::string::npos?start:std::min(best,start);
            break;
        }
    }
    return best;
}

template<class NameSet>
inline bool bare_candidate_repair_eligible(
    const std::string& text,const NameSet& names,bool mode10) {
    // A mode-10 probe already contains a registered-name signature.  The
    // balanced-object path is deferred only for the exact mode-11 shape;
    // ordinary {"name":...} JSON must continue streaming immediately.
    if(mode10) return true;
    if(text.compare(0,7,"{\"name\"")!=0) return false;
    const size_t colon=text.find(':',7);
    if(colon==std::string::npos) return false;
    size_t q1=colon+1;
    while(q1<text.size() && (text[q1]==' ' || text[q1]=='\t' ||
                             text[q1]=='\r' || text[q1]=='\n')) q1++;
    if(q1==text.size() || text[q1]!='\"') return false;
    const size_t q2=text.find('\"',q1+1);
    if(q2==std::string::npos) return false;
    const std::string name=text.substr(q1+1,q2-q1-1);
    if(std::find(names.begin(),names.end(),name)==names.end()) return false;
    const size_t arguments=text.find("\"arguments\"",q2+1);
    if(arguments==std::string::npos) return false;
    const size_t arguments_colon=text.find(':',arguments+11);
    if(arguments_colon==std::string::npos) return false;
    size_t object=arguments_colon+1;
    while(object<text.size() && (text[object]==' ' || text[object]=='\t' ||
                                 text[object]=='\r' || text[object]=='\n')) object++;
    return object<text.size() && text[object]=='{';
}

// Streaming handlers cannot retract text after it reaches the wire. Hold only
// plausible wrapper-less call prefixes, classify complete strict candidates,
// and defer a balanced-but-malformed candidate until final tolerant recovery.
// EmitCandidate receives a callback that preserves Markdown/string context for
// visible bytes and reports parsing and policy acceptance separately.
struct BareToolCandidateResult {
    bool parsed=false;
    bool accepted=false;
    explicit operator bool() const { return parsed; }
};

struct BareToolTextHoldback {
    std::string pending;
    std::string probe;
    std::string deferred;
    std::string deferred_trailing;
    bool deferred_mode10=false;
    bool holding=false;
    bool mode10=false;
    bool input_final=false;
    bool input_allow_repair=false;
    bool ordinary_call_seen=false;
    IncrementalBareJsonEnd scan;
    JsonStringLexState string_state;
    MarkdownFenceLexState fence_state;

    template<class EmitText>
    void emit_visible(const std::string& text,EmitText&& emit_text) {
        consume_bare_text_context(string_state,fence_state,text);
        if(!text.empty()) emit_text(text);
    }

    template<class EmitText,class EmitCandidate>
    bool flush_candidate(bool allow_repair,EmitText&& emit_text,
                         EmitCandidate&& emit_candidate,
                         bool defer_failure=false) {
        if(!holding) return false;
        const bool candidate_mode10=mode10;
        auto visible=[&](const std::string& text) {
            emit_visible(text,emit_text);
        };
        const BareToolCandidateResult result=
            emit_candidate(pending,allow_repair,visible);
        if(!result.parsed) {
            if(defer_failure) {
                deferred=std::move(pending);
                deferred_mode10=candidate_mode10;
            } else visible(pending);
        }
        if(!candidate_mode10 && result.accepted) ordinary_call_seen=true;
        pending.clear();
        holding=false;
        mode10=false;
        string_state.reset();
        return !result.parsed && defer_failure;
    }

    template<class NameSet,class EmitText,class EmitCandidate>
    void route(const std::string& value,const NameSet& names,
               EmitText&& emit_text,EmitCandidate&& emit_candidate) {
        if(!deferred.empty()) {
            deferred_trailing+=value;
            return;
        }
        std::string remaining=std::move(probe);
        probe.clear();
        remaining+=value;
        while(!remaining.empty()) {
            if(holding) {
                pending+=remaining;
                remaining.clear();
            } else {
                const size_t object_pos=bare_object_position(
                    remaining,string_state,fence_state,input_final);
                const size_t mode10_pos=ordinary_call_seen?std::string::npos:
                    bare_mode10_signature_position(
                        remaining,names,string_state,fence_state,input_final);
                const size_t opener=object_pos==std::string::npos?mode10_pos:
                    mode10_pos==std::string::npos?object_pos:
                    std::min(object_pos,mode10_pos);
                if(opener==std::string::npos) {
                    size_t keep=input_final || ordinary_call_seen?std::string::npos:
                        bare_mode10_probe_start(
                            remaining,names,string_state,fence_state,false);
                    if(!input_final) {
                        const size_t inline_keep=bare_unresolved_inline_probe_start(
                            remaining,string_state,fence_state);
                        if(inline_keep!=std::string::npos)
                            keep=keep==std::string::npos?inline_keep:
                                 std::min(keep,inline_keep);
                    }
                    if(keep==std::string::npos) emit_visible(remaining,emit_text);
                    else {
                        emit_visible(remaining.substr(0,keep),emit_text);
                        probe=remaining.substr(keep);
                    }
                    return;
                }
                if(opener) emit_visible(remaining.substr(0,opener),emit_text);
                pending=remaining.substr(opener);
                holding=true;
                mode10=mode10_pos==opener;
                scan.begin(mode10);
                remaining.clear();
            }
            if(!mode10 && !plausible_bare_tool_prefix(pending)) {
                std::string retry=std::move(pending);
                pending.clear();
                holding=false;
                emit_visible(retry.substr(0,1),emit_text);
                remaining=retry.substr(1);
                continue;
            }
            const size_t end=scan.advance(pending);
            if(end==std::string::npos) return;
            std::string trailing=pending.substr(end);
            pending.resize(end);
            const bool defer_failure=!input_final &&
                bare_candidate_repair_eligible(pending,names,mode10);
            if(flush_candidate(input_final && input_allow_repair,
                               emit_text,emit_candidate,defer_failure)) {
                deferred_trailing=std::move(trailing);
                return;
            }
            remaining=std::move(trailing);
        }
    }

    template<class NameSet,class EmitText,class EmitCandidate>
    void finish(bool allow_repair,const NameSet& names,
                EmitText&& emit_text,EmitCandidate&& emit_candidate) {
        if(!probe.empty()) {
            input_final=true;
            input_allow_repair=allow_repair;
            std::string final_probe=std::move(probe);
            probe.clear();
            route(final_probe,names,emit_text,emit_candidate);
            input_final=false;
            input_allow_repair=false;
        }
        std::string trailing;
        if(!deferred.empty()) {
            pending=std::move(deferred);
            trailing=std::move(deferred_trailing);
            holding=true;
            mode10=deferred_mode10;
            deferred_mode10=false;
            flush_candidate(allow_repair,emit_text,emit_candidate);
        }
        if(!trailing.empty()) {
            input_final=true;
            input_allow_repair=allow_repair;
            route(trailing,names,emit_text,emit_candidate);
            input_final=false;
            input_allow_repair=false;
        }
        flush_candidate(allow_repair,emit_text,emit_candidate);
    }

    void reset_context() {
        string_state.reset();
        fence_state.reset();
        ordinary_call_seen=false;
    }
};

template<class EmitText,class EmitCall>
inline size_t route_bare_tool_sequence(
    const std::string& source,const std::vector<ToolCall>& calls,
    EmitText&& emit_text,EmitCall&& emit_call) {
    size_t cursor=0;
    for(const auto& call:calls) {
        if(call.source_begin<cursor || call.source_end<call.source_begin ||
           call.source_end>source.size()) {
            emit_text(source);
            return 0;
        }
        cursor=call.source_end;
    }
    cursor=0;
    size_t emitted=0;
    for(const auto& call:calls) {
        emit_text(source.substr(cursor,call.source_begin-cursor));
        cursor=call.source_end;
        if(emit_call(call)) emitted++;
        else emit_text(source.substr(
            call.source_begin,call.source_end-call.source_begin));
    }
    emit_text(source.substr(cursor));
    return emitted;
}

// Fallback for models that drop the <tool_call> wrapper and emit the call
// JSON as plain text (observed on long write calls under no-think greedy;
// llama.cpp's chat parser has the same class of tolerance). Scans for the
// first balanced {...} that parses as {"name":..., "arguments":...}. On
// success: prefix = text before the JSON, suffix = text after it (typically
// junk like "</file>" -- caller decides to drop it).
// Third observed drift mode: JSON framing with a raw-code value inside
// <content>...</content> tags (the fine-tune's SFT file format leaking into
// arguments). Rewrite `: <content>RAW</content>` spans into proper JSON
// strings so the call parses. Returns the input unchanged if no tag pair.
inline std::string escape_json_interior(const std::string& s) {
    std::string esc;
    for (unsigned char c : s) append_json_escaped_byte(esc,c);
    return esc;
}

// Rewrite a raw code/text value the fine-tune delimited with <content> tags into a
// proper JSON string so the call parses. Two observed shapes:
//   (1) "key": <content>RAW</content>   -- both angle tags in value position.
//   (2) "content": "RAW</content>       -- JSON-quote OPEN, tag CLOSE. The file-write
//       drift: RAW is a multi-line file body with unescaped quotes/newlines/braces
//       that break the JSON string; the model terminates it with a stray </content>
//       (SFT format leak) instead of a closing quote, then continues ", "file_path":...".
//       Verified 6/6 on the failing writes in the 2026-07-06 CRUSH A/B batch.
inline std::string escape_content_tags(const std::string& text) {
    size_t a = text.find("<content>");
    if (a != std::string::npos) {                    // shape 1
        size_t v = a + 9;
        size_t b = text.rfind("</content>");
        if (b == std::string::npos || b < v) return text;
        size_t k = text.find_last_not_of(" \t\r\n", a - 1);
        if (k == std::string::npos || text[k] != ':') return text;
        return text.substr(0, a) + "\"" + escape_json_interior(text.substr(v, b - v)) +
               "\"" + text.substr(b + 10);
    }
    // shape 2: "content": "RAW</content>  (no opening <content> tag)
    size_t close = text.find("</content>");
    if (close == std::string::npos) return text;
    size_t key = text.rfind("\"content\":", close); // nearest content key before the tag
    if (key == std::string::npos) return text;
    size_t q = text.find('"', key + 10);            // value-open quote after "content":
    if (q == std::string::npos || q > close) return text;
    return text.substr(0, q) + "\"" +
           escape_json_interior(text.substr(q + 1, close - q - 1)) + "\"" +
           text.substr(close + 10);
}

// Infer a tool name from an orphaned arguments object (drift mode 6: the model emits
// {"name": {ARGS}} with the name STRING value and the "arguments" key both absent from
// the bytes -- observed on the Claude Code tool schema). Match ARGS's keys against each
// tool's schema: an exact required-set match wins outright; else the highest
// (2*overlap - foreign) score, refusing on a tie (a wrong tool is worse than leaving it
// un-rescued). Needs the request's anthropic_tools_json; nullptr/no-match -> "".
inline std::string infer_tool_name(const json& tools, const json& args) {
    if (!tools.is_array() || !args.is_object()) return "";
    std::set<std::string> ak;
    for (auto it = args.begin(); it != args.end(); ++it) ak.insert(it.key());
    if (ak.empty()) return "";
    // Tie-break (2026-07-11, thunderdome T8 one-shot-quit root cause): the
    // modern CC registry has property-set near-twins (Bash and Monitor both
    // carry {command, description}), so orphaned Bash args scored a 4-4 tie
    // and the rescue refused. A tied candidate whose REQUIRED params are not
    // all present in ARGS could never validate as a call -- eliminate those;
    // a UNIQUE survivor wins. Both-satisfied ties still refuse (a wrong tool
    // remains worse than un-rescued).
    struct Cand { std::string name; bool req_ok; };
    std::vector<Cand> best;
    int best_score = 0;
    for (const auto& t : tools) {
        if (!t.contains("function")) continue;
        const json& fn = t["function"];
        std::string name = fn.value("name", std::string());
        if (name.empty()) continue;
        std::set<std::string> props, req;
        if (fn.contains("parameters") && fn["parameters"].is_object()) {
            const json& p = fn["parameters"];
            if (p.contains("properties") && p["properties"].is_object())
                for (auto it = p["properties"].begin(); it != p["properties"].end(); ++it) props.insert(it.key());
            if (p.contains("required") && p["required"].is_array())
                for (const auto& r : p["required"]) if (r.is_string()) req.insert(r.get<std::string>());
        }
        if (!req.empty() && req == ak) return name;   // exact required-set match: decisive
        int overlap = 0, foreign = 0;
        for (const auto& k : ak) (props.count(k) ? overlap : foreign)++;
        int score = 2 * overlap - foreign;
        if (overlap > 0 && score > 0) { // score-0 candidates never won before either
            bool rok = true;
            for (const auto& r : req)
                if (!ak.count(r)) { rok = false; break; }
            if (score > best_score) {
                best_score = score;
                best.clear();
                best.push_back({name, rok});
            } else if (score == best_score) {
                best.push_back({name, rok});
            }
        }
    }
    if (best.size() == 1) return best[0].name;
    std::string pick;
    for (const auto& c : best)
        if (c.req_ok) {
            if (!pick.empty()) return ""; // >1 required-satisfied: genuine ambiguity
            pick = c.name;
        }
    return pick;
}

// Drift mode 7 (2026-07-08, base Qwen3.6-27B-MTP on the CC schema, first turn,
// deterministic at greedy): the mode-6 orphaned args arrive nested one level
// deeper under a lone shell key -- {"name":\n{"function":{ARGS}}}. Inference on
// the raw object first (never disturbs a working mode-6 rescue); only when that
// fails, peel single-key object shells (bounded) and retry. On success the
// caller gets the INNER args -- the shell must not reach the tool.
// Drift mode 8 (2026-07-09, base Qwen3.6-27B-MTP, T10 ecommerce first turn,
// stable at greedy): the call object is well-formed but names the tool under
// a string-valued ALIAS key -- {"function": "Read", "arguments": {...}} --
// typically batched, behind a dangling {"name": prefix line, with a stray
// </tool_call> closer. Resolve the alias against the REGISTERED tool names
// only (a prose JSON example with an unregistered value must not become a
// call); args = the sibling arguments/parameters/input object, else the
// remainder minus the alias key.
inline bool resolve_aliased_call(const json& tools, const json& obj, std::string& name,
                                 json& args) {
    if (!tools.is_array() || !obj.is_object()) return false;
    static const char* aliases[] = {"function", "tool", "tool_name"};
    const char* akey = nullptr;
    std::string cand;
    for (const char* a : aliases)
        if (obj.contains(a) && obj[a].is_string()) { cand = obj[a]; akey = a; break; }
    if (!akey || cand.empty()) return false;
    bool registered = false;
    for (const auto& t : tools) {
        std::string nm;
        if (t.is_object() && t.contains("function") && t["function"].is_object())
            nm = t["function"].value("name", std::string());
        if (nm.empty() && t.is_object()) nm = t.value("name", std::string());
        if (nm == cand) { registered = true; break; }
    }
    if (!registered) return false;
    static const char* argkeys[] = {"arguments", "parameters", "input"};
    for (const char* k : argkeys)
        if (obj.contains(k) && obj[k].is_object()) {
            name = cand;
            args = obj[k];
            return true;
        }
    json rest = obj;
    rest.erase(akey);
    if (!rest.is_object()) return false;
    name = cand;
    args = std::move(rest);
    return true;
}

inline std::string infer_tool_name_unwrapped(const json& tools, json& args) {
    std::string nm = infer_tool_name(tools, args);
    if (!nm.empty()) return nm;
    { // mode 8: the extracted object itself is an alias-named call
        std::string an;
        json aa;
        if (resolve_aliased_call(tools, args, an, aa)) {
            args = std::move(aa);
            return an;
        }
    }
    static const char* shells[] = {"function", "arguments", "parameters", "input", "tool_call"};
    json u = args;
    for (int hop = 0; hop < 3 && u.is_object() && u.size() == 1; hop++) {
        bool peeled = false;
        for (const char* s : shells)
            if (u.contains(s) && u[s].is_object()) {
                json inner = u[s];
                u = std::move(inner);
                peeled = true;
                break;
            }
        if (!peeled) break;
        nm = infer_tool_name(tools, u);
        if (!nm.empty()) { args = std::move(u); return nm; }
    }
    return "";
}

// Recover a name-dropped mode-6 BATCH: {"name":<ws>{ARGS}[ {"name":<ws>{ARGS}]... where
// each outer {"name": never closes (net +1 depth per unit) so the main balanced scan
// misses the whole run (observed on CC greedy: six {"name":\n{"file_path":...} Read calls).
// For each unit, extract the balanced ARGS object (control-chars sanitized) and infer the
// tool from its key signature. Appends recovered calls; sets *first to the earliest hit.
inline void scan_namedropped(const std::string& text, const json* tools,
                             std::vector<ToolCall>& out, size_t* first) {
    if (!tools) return;
    size_t p = 0;
    JsonStringLexState string_state;
    MarkdownFenceLexState fence_state;
    size_t context_cursor=0;
    while ((p = text.find("{\"name\":", p)) != std::string::npos) {
        for(size_t cursor=context_cursor;cursor<p;cursor++)
            consume_bare_text_context(string_state,fence_state,text[cursor]);
        context_cursor=p;
        if(!bare_context_is_executable(
               text,p,string_state,fence_state,/*end_is_final=*/true)) {
            p+=8;
            continue;
        }
        size_t q = p + 8;
        while (q < text.size() && (text[q]==' '||text[q]=='\t'||text[q]=='\r'||text[q]=='\n')) q++;
        if (q >= text.size() || text[q] != '{') { p += 8; continue; }
        int depth = 0; bool in_str = false, esc = false; size_t e = std::string::npos;
        std::string san;
        for (size_t j = q; j < text.size(); j++) {
            char ch = text[j];
            if (esc) { esc = false; san += ch; continue; }
            if (in_str) {
                if (ch == '\\') { esc = true; san += ch; continue; }
                if (ch == '"') { in_str = false; san += ch; continue; }
                if (ch == '\n') { san += "\\n"; continue; }
                if (ch == '\r') { san += "\\r"; continue; }
                if (ch == '\t') { san += "\\t"; continue; }
                san += ch; continue;
            }
            san += ch;
            if (ch == '"') in_str = true;
            else if (ch == '{') depth++;
            else if (ch == '}' && --depth == 0) { e = j; break; }
        }
        if (e == std::string::npos) break;   // truncated final unit
        try {
            json args = json::parse(san);
            if (args.is_object()) {
                std::string nm = infer_tool_name_unwrapped(*tools, args);
                if (!nm.empty()) {
                    ToolCall tc; tc.ok = true; tc.name = nm; tc.arguments = std::move(args);
                    tc.raw=text.substr(p,e-p+1);
                    tc.rewritten_begin=p;
                    tc.rewritten_end=e+1;
                    if (*first == std::string::npos) *first = p;
                    out.push_back(std::move(tc));
                }
            }
        } catch (...) {}
        p = e + 1;
    }
}

// Scan for ALL recoverable bare calls. Balanced {"name":...,"arguments":...}
// objects anywhere in the text are collected (skipping unbalanced wrappers
// like the literal {"tool_call": opener, which nets +1 depth per blob and
// never closes); a trailing truncated {"name" candidate gets framing repair
// (close open string, strip junk tags, close braces). prefix = text before
// the first recovered call. `tools` (optional) enables mode-6 name inference.
// Drift mode 11 (2026-07-19, issue #4): a tool call whose big string argument
// is a raw code body with unescaped inner quotes / newlines / braces
// ({"name":"Write","arguments":{"content":"<raw source, to end>"}}). Normal
// JSON parsing dies on the first inner `"`, and no local escape heuristic is
// safe (code has `",` `"}` `[]string{"a","b"}` everywhere). Recover
// positionally: the raw value runs to the end of the object, so its terminator
// is the last `"` before the object's closing braces. Extract that span
// literally, then parse the object with the big value blanked to pick up name
// + any other scalar args. Registered-tool + shell-parses gating keeps prose
// out. Handles content-last and scalar-args-before-content; a scalar AFTER the
// big value (rare ordering) over-captures -- accepted vs the current total
// failure (the UN-RESCUED session death). Only runs when nothing else parsed.
// Escape ONLY what's actually unescaped in a nearly-valid JSON string body:
// a valid \-escape is kept verbatim, a bare `"` or raw control char is
// escaped, a lone `\` is doubled. Unlike json(s).dump() this does NOT double-
// escape content the model already escaped correctly (\n \t \" ...), which is
// the mostly-escaped-with-sparse-errors case (issue #4, 2026-07-20:
// {"content":"...\"fmt\"...\"strings"..."} -- most quotes escaped, one not).
inline std::string minimal_escape_body(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (size_t i = 0; i < s.size(); i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\\') {
            if (i + 1 < s.size()) {
                char n = s[i + 1];
                if (n == '"' || n == '\\' || n == '/' || n == 'b' || n == 'f' || n == 'n' ||
                    n == 'r' || n == 't' || n == 'u') {
                    out += '\\';
                    out += n;
                    i++;
                    continue;
                }
            }
            out += "\\\\";
            continue;
        }
        if (c == '"') {
            out += "\\\"";
            continue;
        }
        if (c < 0x20) {
            switch (c) {
                case '\n': out += "\\n"; break;
                case '\t': out += "\\t"; break;
                case '\r': out += "\\r"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                default: {
                    char buf[8];
                    snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                }
            }
            continue;
        }
        out += (char)c;
    }
    return out;
}

// Return the first top-level balanced {...} (JSON string/escape aware), so
// trailing junk after the call object -- </tool_call>, prose, a second blob --
// doesn't make json::parse reject an otherwise-valid reconstruction.
inline std::string first_balanced_object(const std::string& s) {
    size_t start = s.find('{');
    if (start == std::string::npos) return "";
    int depth = 0;
    bool in_str = false, esc = false;
    for (size_t i = start; i < s.size(); i++) {
        char ch = s[i];
        if (esc) { esc = false; continue; }
        if (in_str) {
            if (ch == '\\') esc = true;
            else if (ch == '"') in_str = false;
            continue;
        }
        if (ch == '"') in_str = true;
        else if (ch == '{') depth++;
        else if (ch == '}') {
            if (--depth == 0) return s.substr(start, i - start + 1);
        }
    }
    return "";
}


inline bool recover_raw_value_call(const std::string& text, const json& tools,
                                   std::vector<ToolCall>& out) {
    size_t mo = text.rfind("{\"name\"");
    if (mo == std::string::npos) return false;
    if (!bare_text_position_is_executable(text, mo)) return false;
    size_t colon = text.find(':', mo + 6);
    if (colon == std::string::npos) return false;
    size_t q1 = text.find('"', colon + 1);
    if (q1 == std::string::npos) return false;
    size_t q2 = text.find('"', q1 + 1);
    if (q2 == std::string::npos) return false;
    const std::string nm = text.substr(q1 + 1, q2 - q1 - 1);
    const json* fn = nullptr;
    for (const auto& t : tools)
        if (t.contains("function") && t["function"].value("name", std::string()) == nm) {
            fn = &t["function"];
            break;
        }
    if (!fn) return false;
    std::vector<std::string> strkeys;
    if (fn->contains("parameters") && (*fn)["parameters"].is_object()) {
        const json& pr = (*fn)["parameters"];
        if (pr.contains("properties") && pr["properties"].is_object())
            for (auto it = pr["properties"].begin(); it != pr["properties"].end(); ++it)
                if (it.value().is_object() &&
                    it.value().value("type", std::string()) == "string")
                    strkeys.push_back(it.key());
    }
    if (strkeys.empty()) return false;
    // For each string param, forward-scan candidate terminators of ITS value:
    // escape the span [opener+1, cand), keep the tail after cand literal, and
    // parse the reconstructed object. The FIRST candidate that parses is the
    // real terminator -- inner quotes leave the tail as un-parseable raw code,
    // and a scalar arg after the value forces the correct earlier terminator
    // (making that arg a valid sibling). Ordering-independent; map the
    // reconstructed suffix back to the source so following prose is preserved.
    for (const auto& k : strkeys) {
        size_t kp = text.find("\"" + k + "\"", mo);
        if (kp == std::string::npos) continue;
        size_t kc = text.find(':', kp + k.size() + 2);
        if (kc == std::string::npos) continue;
        size_t opener = text.find('"', kc + 1);
        if (opener == std::string::npos) continue;
        for (size_t cand = text.find('"', opener + 1); cand != std::string::npos;
             cand = text.find('"', cand + 1)) {
            // Minimal-escape the value body (don't double-escape already-valid
            // \-escapes) and trim trailing junk by taking the first balanced
            // object, so both fully-raw and mostly-escaped content recover.
            const std::string body = minimal_escape_body(text.substr(opener + 1, cand - opener - 1));
            const std::string recon = first_balanced_object(
                text.substr(mo, opener - mo) + "\"" + body + "\"" + text.substr(cand + 1));
            if (recon.empty()) continue;
            json obj;
            try { obj = json::parse(recon); } catch (...) { continue; }
            if (!obj.is_object() || obj.value("name", std::string()) != nm) continue;
            json args = obj.contains("arguments") && obj["arguments"].is_object()
                            ? obj["arguments"]
                            : json::object();
            const size_t suffix_offset=(opener-mo)+body.size()+2;
            if(recon.size()<suffix_offset) continue;
            const size_t source_end=cand+1+(recon.size()-suffix_offset);
            if(source_end>text.size()) continue;
            ToolCall tc;
            tc.ok = true;
            tc.name = nm;
            tc.rewritten_begin=mo;
            tc.rewritten_end=source_end;
            tc.arguments = std::move(args);
            tc.raw=text.substr(mo,source_end-mo);
            fprintf(stderr, "[drift] mode-11 raw-value rescue: %s.%s (%zu bytes)\n", nm.c_str(),
                    k.c_str(), cand - opener - 1);
            out.push_back(std::move(tc));
            return true;
        }
    }
    return false;
}

inline std::vector<ToolCall> parse_bare_tool_calls(const std::string& text_in,
                                                   std::string* prefix,
                                                   const json* tools = nullptr,
                                                   bool allow_o10 = true,
                                                   bool allow_eof_repair = true,
                                                   std::string* remaining_text = nullptr) {
    std::vector<ToolCall> out;
    if (tool_strict()) {
        // strict-parser A/B: the wrapper-less recovery chain (drift modes 1-6)
        // is OFF. Log when the text plausibly contained an intended call so the
        // campaign can count suppressed rescues against the tolerant leg.
        if (prefix) *prefix = "";
        if (remaining_text) *remaining_text = text_in;
        if (text_in.find("{\"name\"") != std::string::npos ||
            text_in.find("{\"tool_call\"") != std::string::npos ||
            text_in.find("</content>") != std::string::npos)
            fprintf(stderr, "[q27-strict] SUPPRESSED bare-call rescue: %.200s\n",
                    text_in.c_str());
        return out;
    }
    bool m2 = false, m5 = false, m6 = false, m8 = false; // drift-mode flags (exit-gate catalog)
    struct SourceMappedText {
        std::string value;
        std::vector<size_t> boundaries;

        explicit SourceMappedText(const std::string& source):value(source) {}

        void ensure_boundaries() {
            if(!boundaries.empty()) return;
            boundaries.reserve(value.size()+1);
            for(size_t i=0;i<=value.size();i++) boundaries.push_back(i);
        }

        size_t source_boundary(size_t position) const {
            return boundaries.empty()?position:boundaries.at(position);
        }

        void insert(size_t position,const std::string& added) {
            ensure_boundaries();
            const size_t source=boundaries.at(position);
            value.insert(position,added);
            boundaries.insert(boundaries.begin()+position+1,added.size(),source);
        }

        void replace_with(const std::string& replacement) {
            if(replacement==value) return;
            ensure_boundaries();
            size_t prefix=0;
            while(prefix<value.size() && prefix<replacement.size() &&
                  value[prefix]==replacement[prefix]) prefix++;
            size_t old_suffix=value.size(),new_suffix=replacement.size();
            while(old_suffix>prefix && new_suffix>prefix &&
                  value[old_suffix-1]==replacement[new_suffix-1]) {
                old_suffix--;
                new_suffix--;
            }
            const size_t source_begin=boundaries[prefix];
            const size_t source_end=boundaries[old_suffix];
            std::vector<size_t> next;
            next.reserve(replacement.size()+1);
            next.insert(next.end(),boundaries.begin(),boundaries.begin()+prefix+1);
            const size_t replacement_size=new_suffix-prefix;
            if(replacement_size==0) next.back()=source_end;
            else for(size_t i=0;i<replacement_size;i++)
                next.push_back(i+1==replacement_size?source_end:source_begin);
            next.insert(next.end(),boundaries.begin()+old_suffix+1,boundaries.end());
            value=replacement;
            boundaries=std::move(next);
        }
    };
    SourceMappedText rewritten(text_in);
    rewritten.replace_with(escape_content_tags(rewritten.value));
    // drift mode 9 (2026-07-11, codex-harnessed traffic): the model drops the
    // OPENING quote of the "arguments" key ({"name":"X",\narguments":{...}}).
    // "arguments" is the tool-call schema key, so quoting a bare `arguments":`
    // is unambiguous inside these segments. Applied before segmentation so
    // both the whole-object and truncated-tail parse paths see valid JSON.
    auto fix_arg_quote = [](SourceMappedText& text) {
        size_t p = 0;
        while ((p = text.value.find("arguments\"", p)) != std::string::npos) {
            if (p == 0 || text.value[p - 1] != '"') {
                text.insert(p,"\"");
                p += 11;
            } else p += 10;
        }
    };
    // drift mode 12 (2026-07-19, club-3090 cli-40 agent): the model drops the
    // QUOTES around the tool-NAME value -- {"name": bash, "arguments": {...}}.
    // Invalid JSON, so the whole call went UN-RESCUED and the agent's turn
    // stopped (turnsUsed=0). The arguments are valid JSON, so once the bare
    // name is quoted the object parses on the normal path. SAFE because we only
    // quote a bareword that EXACTLY matches a registered tool name -- prose,
    // non-tool JSON, null/numbers, and unknown names are left untouched.
    auto fix_unquoted_name = [](SourceMappedText& text, const json* tools) {
        if (!tools || !tools->is_array()) return;
        size_t p = 0;
        while ((p = text.value.find("\"name\":", p)) != std::string::npos) {
            size_t c = p + 7;
            while (c < text.value.size() &&
                   (text.value[c] == ' ' || text.value[c] == '\t' ||
                    text.value[c] == '\n' || text.value[c] == '\r')) c++;
            if (c < text.value.size() && text.value[c] != '"' &&
                (isalpha((unsigned char)text.value[c]) || text.value[c] == '_')) {
                size_t e = c;
                while (e < text.value.size() &&
                       (isalnum((unsigned char)text.value[e]) ||
                        text.value[e] == '_' || text.value[e] == '-' ||
                        text.value[e] == '.')) e++;
                const std::string word = text.value.substr(c, e - c);
                bool match = false;
                for (const auto& t : *tools)
                    if (t.contains("function") &&
                        t["function"].value("name", std::string()) == word) {
                        match = true;
                        break;
                    }
                if (match) {
                    // Add the closing quote only if one isn't already there:
                    // {"name": bash,   -> both quotes; {"name": read"  (dropped
                    // OPENING quote, stray close) -> opening only (thunderdome).
                    if (!(e < text.value.size() && text.value[e] == '"'))
                        text.insert(e,"\"");
                    text.insert(c,"\"");
                    p = e + 2;
                    continue;
                }
            }
            p = c;
        }
    };
    fix_arg_quote(rewritten);
    fix_unquoted_name(rewritten,tools);
    const std::string& text=rewritten.value;
    const bool m3 = (text != text_in);              // mode 3: <content>-tagged value rewritten
    const bool m4 = text.find("{\"tool_call\":") != std::string::npos; // mode 4: JSON-keyed opener
    size_t first = std::string::npos;
    size_t i = text.find('{');
    JsonStringLexState candidate_string_state;
    MarkdownFenceLexState candidate_fence_state;
    size_t candidate_context_cursor=0;
    while (i != std::string::npos) {
        // Displayed Markdown is an example or echoed input, not a call the
        // model is making.  Malformed earlier call text must not poison a
        // later executable candidate's independent recovery.
        for(size_t cursor=candidate_context_cursor;cursor<i;cursor++)
            consume_bare_text_context(
                candidate_string_state,candidate_fence_state,text[cursor]);
        candidate_context_cursor=i;
        if(!bare_context_is_executable(
               text,i,candidate_string_state,candidate_fence_state,
               /*end_is_final=*/true)) {
            i = text.find('{', i + 1);
            continue;
        }
        int depth = 0;
        bool in_str = false, esc = false;
        size_t end = std::string::npos;
        std::string san;  // segment with raw in-string control chars escaped
        for (size_t j = i; j < text.size(); j++) {
            char ch = text[j];
            if (esc) { esc = false; san += ch; continue; }
            if (in_str) {
                if (ch == '\\') { esc = true; san += ch; continue; }
                if (ch == '"') { in_str = false; san += ch; continue; }
                // fifth drift mode: literal newlines/tabs inside the string
                if (ch == '\n') { san += "\\n"; m5 = true; continue; }
                if (ch == '\r') { san += "\\r"; m5 = true; continue; }
                if (ch == '\t') { san += "\\t"; m5 = true; continue; }
                san += ch;
                continue;
            }
            san += ch;
            if (ch == '"') in_str = true;
            else if (ch == '{') depth++;
            else if (ch == '}' && --depth == 0) { end = j; break; }
        }
        if (end == std::string::npos) {
            if (!allow_eof_repair) break;
            // unbalanced to EOF: repair only a {"name" candidate (truncated
            // final call); otherwise keep scanning inner objects. `san` holds
            // the sanitized remainder (scan ran to EOF).
            if (san.rfind("{\"name\"", 0) != 0) { i = text.find('{', i + 1); continue; }
            std::string r = san;
            while (true) {
                size_t e2 = r.find_last_not_of(" \t\r\n");
                if (e2 == std::string::npos) break;
                r.resize(e2 + 1);
                if (r.size() >= 7 && r.compare(r.size() - 7, 7, "</file>") == 0)
                    r.resize(r.size() - 7);
                else break;
            }
            int d2 = 0;
            bool s2 = false, e2f = false;
            for (char ch : r) {
                if (e2f) { e2f = false; continue; }
                if (s2) {
                    if (ch == '\\') e2f = true;
                    else if (ch == '"') s2 = false;
                    continue;
                }
                if (ch == '"') s2 = true;
                else if (ch == '{') d2++;
                else if (ch == '}') d2--;
            }
            // Drift mode 13 (2026-07-24): the truncation can land INSIDE an
            // escape sequence. A trailing "\\" swallows the quote we are about
            // to append -- the string stays open, the object never parses, and
            // the whole call goes UN-RESCUED. Observed live: a wrapper-less
            // Write whose markdown `content` was cut mid-`\\"` while writing an
            // escaped JSON example. Same for a partial \\uXXXX. Trim back to the
            // last safe byte before closing.
            if (s2) {
                if (e2f) {
                    r.pop_back();  // dangling backslash
                } else {
                    const size_t u = r.find_last_of('\\');
                    // a COMPLETE \uXXXX is 6 bytes; anything shorter at the very
                    // end is a partial one. Leading-backslash check avoids
                    // trimming an escaped backslash that merely precedes a 'u'.
                    if (u != std::string::npos && u + 1 < r.size() && r[u + 1] == 'u' &&
                        r.size() - u < 6 && !(u > 0 && r[u - 1] == '\\'))
                        r.resize(u);
                }
                r += '"';
            }
            for (; d2 > 0; d2--) r += '}';
            bool shaped = false;
            try {
                json j = json::parse(r);
                shaped = j.is_object() && j.contains("name") && j.contains("arguments");
            } catch (...) {}
            bool recovered_here = false;
            if (shaped) {
                ToolCall tc = parse_tool_call(r);
                tc.raw=text.substr(i);
                tc.rewritten_begin=i;
                tc.rewritten_end=text.size();
                if (tc.ok) {
                    if (first == std::string::npos) first = i;
                    out.push_back(tc);
                    m2 = true;   // mode 2: truncated/unterminated JSON repaired
                    recovered_here = true;
                }
            }
            if (recovered_here) break;   // genuine truncated FINAL call consumed the rest
            // else: a dangling/failed {"name": opener -- e.g. the mode-6 HYBRID where the
            // model prepends a bare {"name": before a batch of VALID calls ("read all files
            // in parallel"). Don't discard the rest: advance past the opener and keep scanning
            // so the real calls after it recover normally.
            i = text.find('{', i + 1);
            continue;
        }
        const std::string raw_seg=text.substr(i,end-i+1);
        const std::string& seg = san;
        bool shaped = false, m6cand = false, m8cand = false;
        bool valid_json=false;
        json j6, j8;
        try {
            json j = json::parse(seg);
            valid_json=true;
            shaped = j.is_object() && j.contains("name") && j.contains("arguments");
            if (!shaped && j.is_object() && j.contains("name") &&
                j["name"].is_object() && !j.contains("arguments")) {
                m6cand = true; j6 = std::move(j);
            } else if (!shaped && j.is_object()) {
                m8cand = true; j8 = std::move(j);
            }
        } catch (...) {}
        if (shaped) {
            ToolCall tc = parse_tool_call(seg);
            tc.raw=raw_seg;
            tc.rewritten_begin=i;
            tc.rewritten_end=end+1;
            if (tc.ok) {
                if (first == std::string::npos) first = i;
                out.push_back(tc);
                i = text.find('{', end + 1);
                continue;
            }
        } else if (m6cand && tools) {
            // mode 6: {"name": {ARGS}} -- name string + "arguments" key both dropped.
            // Infer the tool from the orphaned args' key signature (mode 7:
            // unwrap a lone shell key first when the raw keys match nothing).
            json m6args = j6["name"];
            std::string nm = infer_tool_name_unwrapped(*tools, m6args);
            if (!nm.empty()) {
                ToolCall tc; tc.ok = true; tc.name = nm; tc.arguments = std::move(m6args);
                tc.raw=raw_seg;
                tc.rewritten_begin=i;
                tc.rewritten_end=end+1;
                if (first == std::string::npos) first = i;
                out.push_back(std::move(tc));
                m6 = true;
                i = text.find('{', end + 1);
                continue;
            }
        } else if (m8cand && tools) {
            // mode 10 tail (2026-07-18): flat name+args -- a STRING "name"
            // matching a registered tool, with the arguments as SIBLING keys
            // instead of nested under "arguments" ({"name":"Read","file_path":
            // ...}). This is what the mode-10 opener-splice produces, and
            // the model also emits it wrapper-less. Validate the name against
            // the registry so prose JSON with a "name" field can't match.
            if (j8.contains("name") && j8["name"].is_string()) {
                const std::string cand = j8["name"].get<std::string>();
                bool known = false;
                for (const auto& t : *tools)
                    if (t.contains("function") &&
                        t["function"].value("name", std::string()) == cand) { known = true; break; }
                if (known && !cand.empty()) {
                    json args = json::object();
                    for (auto it = j8.begin(); it != j8.end(); ++it)
                        if (it.key() != "name") args[it.key()] = it.value();
                    ToolCall tc; tc.ok = true; tc.name = cand; tc.arguments = std::move(args);
                    tc.raw=raw_seg;
                    tc.rewritten_begin=i;
                    tc.rewritten_end=end+1;
                    if (first == std::string::npos) first = i;
                    out.push_back(std::move(tc));
                    m8 = true;
                    i = text.find('{', end + 1);
                    continue;
                }
            }
            // mode 8: alias-named call object ({"function": "Read", ...});
            // registered-name validation inside keeps prose JSON out
            std::string an;
            json aa;
            if (resolve_aliased_call(*tools, j8, an, aa)) {
                ToolCall tc; tc.ok = true; tc.name = std::move(an); tc.arguments = std::move(aa);
                tc.raw=raw_seg;
                tc.rewritten_begin=i;
                tc.rewritten_end=end+1;
                if (first == std::string::npos) first = i;
                out.push_back(std::move(tc));
                m8 = true;
                i = text.find('{', end + 1);
                continue;
            }
        }
        i = text.find('{', valid_json ? end + 1 : i + 1);
    }
    // Fallback: name-dropped mode-6 BATCH (the main scan can't segment it). Only when the
    // standard scan found nothing, so normal calls are never double-counted.
    if (out.empty() && allow_eof_repair && tools &&
        text.find("{\"name\":") != std::string::npos) {
        scan_namedropped(text, tools, out, &first);
        if (!out.empty()) m6 = true;
    }
    if (prefix) {
        std::string p = first == std::string::npos ? "" : strip_ws2(text.substr(0, first));
        // drop a dangling {"tool_call": opener fragment (mode-4 wrapper junk)
        if (p.size() >= 13 && p.compare(p.size() - 13, 13, "{\"tool_call\":") == 0)
            p = strip_ws2(p.substr(0, p.size() - 13));
        // drop a dangling {"name": opener fragment (mode-6/8 hybrid prefix line)
        if (p.size() >= 8 && p.compare(p.size() - 8, 8, "{\"name\":") == 0)
            p = strip_ws2(p.substr(0, p.size() - 8));
        *prefix = p;
    }
    // Drift mode 10 (2026-07-18, SWE-bench flask-5014 first-tool-call
    // rescue miss): the model drops the ENTIRE `{"name": "` opener, emitting
    // `NAME", "key": val ...}` with no brace at all -- so the {-scanner above
    // finds no candidate. When nothing else rescued and a KNOWN tool name is
    // followed by the `", "` argument-separator signature (and is not already
    // properly quoted), splice the opener back and re-parse ONCE (allow_o10
    // false in the recursion so a still-broken splice cannot loop). This is
    // the deterministic early-quit the n=3 seal caught: a missed first call
    // ends the agent turn with a leaked-JSON text response.
    if (allow_o10 && tools && tools->is_array()) {
        struct O10Candidate { size_t source,end; std::string name; };
        std::vector<O10Candidate> candidates;
        for (const auto& t : *tools) {
            const std::string nm=t.contains("function")
                ?t["function"].value("name",std::string()):std::string();
            if(nm.empty()) continue;
            const std::string sig=nm+"\", \"";
            for(size_t p=text.find(sig);p!=std::string::npos;p=text.find(sig,p+1)) {
                // Quoted-string filtering runs after source-order sorting so
                // an accepted dropped-opener call can restore lexical state
                // before a later call in the same batch.
                // Once the ordinary scanner has found a call, a matching
                // signature after it is prose, not another executable item.
                // Mode 10 only repairs dropped openers that precede the first
                // already-classified call (or the whole turn when none exist).
                if(first!=std::string::npos && p>=first) continue;
                candidates.push_back({p,p+sig.size(),nm});
            }
        }
        std::sort(candidates.begin(),candidates.end(),
                  [](const O10Candidate& a,const O10Candidate& b) {
                      if(a.source!=b.source) return a.source<b.source;
                      return a.end>b.end; // same start: keep the longest name
                  });
        std::vector<O10Candidate> nonoverlapping;
        for(auto& candidate:candidates)
            if(nonoverlapping.empty() || candidate.source>=nonoverlapping.back().end)
                nonoverlapping.push_back(std::move(candidate));
        candidates=std::move(nonoverlapping);
        JsonStringLexState lexical;
        MarkdownFenceLexState fence;
        size_t lexical_cursor=0,repaired_until=0,invalid_until=0;
        const std::string mode10_opener="{\"name\": \"";
        struct InvalidO10Context { size_t source,end; };
        std::vector<InvalidO10Context> invalid_contexts;
        constexpr size_t max_invalid_contexts=8;
        auto add_invalid_context=[&](size_t source,size_t end) {
            if(invalid_contexts.size()>=max_invalid_contexts) return false;
            invalid_contexts.push_back({source,end});
            return true;
        };
        std::vector<O10Candidate> safe_candidates;
        auto consume_mode10_context=[&](size_t begin,size_t end) {
            end=std::min(end,text.size());
            for(size_t i=std::min(begin,end);i<end;i++)
                consume_bare_text_context(lexical,fence,text[i]);
        };
        for(auto& candidate:candidates) {
            if(candidate.source<repaired_until) continue;
            if(candidate.source<invalid_until) {
                const bool nested=std::any_of(
                    invalid_contexts.begin(),invalid_contexts.end(),
                    [&](const InvalidO10Context& context) {
                        return candidate.source>context.source &&
                               candidate.source<context.end;
                    });
                if(nested) continue;
                lexical.reset();
                fence.reset();
                lexical_cursor=candidate.source;
            } else if(lexical_cursor<invalid_until) {
                lexical.reset();
                fence.reset();
                lexical_cursor=invalid_until;
            }
            consume_mode10_context(lexical_cursor,candidate.source);
            lexical_cursor=candidate.source;
            if(!bare_context_is_executable(
                   text,candidate.source,lexical,fence,true)) continue;
            IncrementalBareJsonEnd completion;
            completion.begin(true,candidate.source);
            const size_t call_end=completion.advance(text);
            const size_t context_end=call_end==std::string::npos
                ?text.size():call_end;
            std::string probe=mode10_opener;
            probe.append(text,candidate.source,context_end-candidate.source);
            if(call_end==std::string::npos && allow_eof_repair) probe+='}';
            if(call_end==std::string::npos && !allow_eof_repair) {
                invalid_until=text.size();
                if(!add_invalid_context(candidate.source,context_end)) break;
                lexical.reset();
                fence.reset();
                lexical_cursor=candidate.end;
                continue;
            }
            std::string probe_prefix,probe_remaining;
            const auto parsed_candidate=parse_bare_tool_calls(
                probe,&probe_prefix,tools,/*allow_o10=*/false,
                allow_eof_repair,&probe_remaining);
            const bool validated=std::any_of(
                parsed_candidate.begin(),parsed_candidate.end(),
                [&](const ToolCall& call) {
                    return call.ok && call.name==candidate.name &&
                           call.source_begin==0;
                });
            if(!validated) {
                invalid_until=std::max(
                    invalid_until,
                    call_end==std::string::npos?text.size():call_end);
                if(!add_invalid_context(candidate.source,context_end)) break;
                lexical.reset();
                fence.reset();
                lexical_cursor=candidate.end;
                continue;
            }
            safe_candidates.push_back(candidate);
            repaired_until=call_end==std::string::npos?text.size():call_end;
            lexical_cursor=repaired_until;
            lexical.reset();
            fence.reset();
        }
        candidates=std::move(safe_candidates);
        if(!candidates.empty()) {
            struct O10Insertion { size_t begin,end,source; };
            // A validated candidate may follow malformed same-line call text
            // with an unmatched quote. The synthetic newline is removed by
            // source mapping and prevents that discarded prefix from poisoning
            // the independently validated repaired object.
            const std::string opener="\n{\"name\": \"";
            std::vector<O10Insertion> insertions;
            std::string synth;
            size_t cursor=0;
            for(const auto& candidate:candidates) {
                synth.append(text,cursor,candidate.source-cursor);
                const size_t begin=synth.size();
                synth+=opener;
                insertions.push_back({begin,synth.size(),candidate.source});
                cursor=candidate.source;
            }
            synth.append(text,cursor,std::string::npos);
            if(synth.find('}')==std::string::npos && allow_eof_repair) {
                const size_t begin=synth.size();
                synth+='}';
                insertions.push_back({begin,synth.size(),text.size()});
            }
            std::string pre2,rec_remaining;
            auto rec=parse_bare_tool_calls(synth,&pre2,tools,
                                           /*allow_o10=*/false,allow_eof_repair,
                                           &rec_remaining);
            if(!rec.empty()) {
                auto source_boundary=[&](size_t boundary) {
                    size_t removed=0;
                    for(const auto& insertion:insertions) {
                        if(boundary<=insertion.begin) break;
                        if(boundary<=insertion.end) return insertion.source;
                        removed+=insertion.end-insertion.begin;
                    }
                    return boundary-removed;
                };
                bool mapped=true;
                for(auto& call:rec) {
                    if(call.source_begin==std::string::npos ||
                       call.source_end==std::string::npos) {
                        mapped=false;
                        break;
                    }
                    const size_t begin=source_boundary(call.source_begin);
                    const size_t end=source_boundary(call.source_end);
                    if(end<begin || end>text.size()) {
                        mapped=false;
                        break;
                    }
                    call.source_begin=begin;
                    call.source_end=end;
                    call.rewritten_begin=begin;
                    call.rewritten_end=end;
                    call.raw=text.substr(begin,end-begin);
                }
                if(mapped) {
                    out=std::move(rec);
                    first=out.front().source_begin;
                    if(prefix) *prefix=strip_ws2(text.substr(0,first));
                    fprintf(stderr,"[drift] mode-10 dropped-opener rescue: %s (%zu calls)\n",
                            candidates.front().name.c_str(),out.size());
                }
            }
        }
    }
    // mode 11: raw code-body string value (unescaped inner quotes/newlines).
    // Last resort -- runs only when everything above failed. Sets the prefix
    // to the text before the call object so a preamble ("Let me write...")
    // still streams as text.
    if (out.empty() && allow_eof_repair && tools && tools->is_array() &&
        recover_raw_value_call(text, *tools, out)) {
        size_t mo = text.rfind("{\"name\"");
        const std::string before = mo != std::string::npos
            ? strip_ws2(text.substr(0, mo)) : std::string();
        if (prefix) *prefix = before;
        if (remaining_text) *remaining_text = before;
    }
    bool exact_source_spans=true;
    size_t source_cursor=0;
    for(auto& call:out) {
        if(call.rewritten_begin==std::string::npos ||
           call.rewritten_end==std::string::npos ||
           call.rewritten_begin>call.rewritten_end ||
           call.rewritten_end>rewritten.value.size()) {
            exact_source_spans=false;
            break;
        }
        const size_t begin=rewritten.source_boundary(call.rewritten_begin);
        const size_t end=rewritten.source_boundary(call.rewritten_end);
        if(begin<source_cursor || end<begin || end>text_in.size()) {
            exact_source_spans=false;
            break;
        }
        call.source_begin=begin;
        call.source_end=end;
        call.raw=text_in.substr(begin,end-begin);
        source_cursor=end;
    }
    if(!exact_source_spans) out.clear();
    if(remaining_text) {
        if(out.empty()) {
            *remaining_text=text_in;
        } else {
            std::string remainder;
            remainder.reserve(text_in.size());
            source_cursor=0;
            for(const auto& call:out) {
                remainder.append(text_in,source_cursor,
                                 call.source_begin-source_cursor);
                source_cursor=call.source_end;
            }
            remainder.append(text_in,source_cursor,std::string::npos);
            *remaining_text=std::move(remainder);
        }
    }
    // Drift catalog (exit gate, docs/sampling-exit-gate.md): tag which tool-format
    // drift mode(s) the fallback chain rescued, or flag an intended call it could
    // NOT recover. Log-only; the parse result is unchanged.
    if (!out.empty()) {
        char modes[8]; int mi = 0;
        modes[mi++] = '1';               // baseline: dropped-<tool_call>-wrapper recovery
        if (m2) modes[mi++] = '2';
        if (m3) modes[mi++] = '3';
        if (m4) modes[mi++] = '4';
        if (m5) modes[mi++] = '5';
        if (m6) modes[mi++] = '6';
        if (m8) modes[mi++] = '8';
        modes[mi] = 0;
        fprintf(stderr, "[drift] recovered=%zu modes=%s\n", out.size(), modes);
    } else if (text_in.find("{\"name\"") != std::string::npos ||
               text_in.find("{\"tool_call\"") != std::string::npos) {
        // ntools distinguishes plumbing (-1/0) from a schema/inference miss (>0 =
        // mode-6 args didn't confidently match any tool). Longer window so the call
        // (not just a long preamble) is visible for post-hoc arg-shape diagnosis.
        fprintf(stderr, "[drift] UN-RESCUED (ntools=%d) intended tool call: %.400s\n",
                tools ? (int)tools->size() : -1, text_in.c_str());
        // Corpus capture: Q27_DRIFT_CORPUS=<file> appends the FULL untruncated
        // miss (the stderr line caps at 400 chars, which is why issue #4's
        // payload wasn't visible). Each miss is a replayable fixture for
        // tools/test_tool_drift_corpus.cpp -- turns an unknown drift mode into
        // a permanent regression the next time it recurs. NUL-separated
        // records so embedded newlines don't confuse the reader.
        if (const char* cp = getenv("Q27_DRIFT_CORPUS")) {
            if (FILE* f = fopen(cp, "ab")) {
                fwrite(text_in.data(), 1, text_in.size(), f);
                fputc('\0', f);
                fclose(f);
            }
        }
    }
    return out;
}

struct OrderedToolPart {
    enum class Kind { Text, Call };
    Kind kind=Kind::Text;
    size_t text_begin=0;
    size_t text_end=0;
    size_t call_index=0;
};

struct OrderedToolOutput {
    std::string text;
    std::string reasoning;
    std::vector<ToolCall> calls;
    std::vector<OrderedToolPart> parts;
    size_t recovered = 0;

    void append_visible_text(const std::string& raw) {
        if(raw.empty()) return;
        const size_t begin=text.size();
        text+=raw;
        if(!parts.empty() && parts.back().kind==OrderedToolPart::Kind::Text &&
           parts.back().text_end==begin) {
            parts.back().text_end=text.size();
            return;
        }
        OrderedToolPart part;
        part.kind=OrderedToolPart::Kind::Text;
        part.text_begin=begin;
        part.text_end=text.size();
        parts.push_back(part);
    }

    void append_tool_call(ToolCall call) {
        OrderedToolPart part;
        part.kind=OrderedToolPart::Kind::Call;
        part.call_index=calls.size();
        calls.push_back(std::move(call));
        parts.push_back(part);
    }
};

inline std::string take_unclosed_final_tool_segment(
    std::vector<std::pair<StreamSplitter::Chan,std::string>>& segments,
    bool wrapper_incomplete) {
    if(!wrapper_incomplete || segments.empty() ||
       segments.back().first!=StreamSplitter::TOOL) return {};
    std::string raw=std::move(segments.back().second);
    segments.pop_back();
    return raw;
}

// Resolve generated text and wrapped calls in generation order. This matters
// when parallel calls are disabled: a bare call emitted before a later wrapped
// call must win, and rejected or malformed bytes must remain visible as text.
template<class Eligible>
inline OrderedToolOutput resolve_ordered_tool_segments(

    const std::vector<std::pair<StreamSplitter::Chan,std::string>>& segments,
    const json* tools,bool allow_eof_repair,Eligible eligible) {
    OrderedToolOutput out;
    auto append_text=[&](const std::string& raw,bool repair_eof) {
        if(!tools) { out.append_visible_text(raw); return; }
        if(raw.empty()) return;
        std::string pre,residual;
        auto calls=parse_bare_tool_calls(raw,&pre,tools,true,
                                         repair_eof,&residual);
        if(calls.empty()) { out.append_visible_text(raw); return; }
        size_t cursor=0;
        for(auto& call:calls) {
            out.append_visible_text(raw.substr(cursor,call.source_begin-cursor));
            cursor=call.source_end;
            if(eligible(call.name,out.calls.size())) {
                out.recovered++;
                out.append_tool_call(std::move(call));
            } else {
                out.append_visible_text(raw.substr(
                    call.source_begin,call.source_end-call.source_begin));
            }
        }
        out.append_visible_text(raw.substr(cursor));
    };
    std::string pending_text;
    auto flush_pending_text=[&](bool final_segment) {
        if(pending_text.empty()) return;
        append_text(pending_text,final_segment && allow_eof_repair);
        pending_text.clear();
    };
    for(const auto& segment:segments) {
        if(segment.first==StreamSplitter::TEXT) {
            pending_text+=segment.second;
            continue;
        }
        flush_pending_text(false);
        if(segment.first==StreamSplitter::THINK) {
            out.reasoning+=segment.second;
            continue;
        }
        ToolCall call=parse_tool_call(strip_ws2(segment.second));
        if(call.ok && eligible(call.name,out.calls.size()))
            out.append_tool_call(std::move(call));
        else out.append_visible_text(call.raw);
    }
    flush_pending_text(true);
    return out;
}

template<class ToolBlock>
inline size_t append_anthropic_ordered_content(
    json& content,const OrderedToolOutput& out,ToolBlock tool_block) {
    size_t emitted=0;
    size_t call_number=0;
    for(const auto& part:out.parts) {
        if(part.kind==OrderedToolPart::Kind::Call) {
            content.push_back(tool_block(out.calls.at(part.call_index),call_number++));
            emitted++;
            continue;
        }
        if(part.text_begin>=part.text_end) continue;
        content.push_back({{"type","text"},
                           {"text",out.text.substr(
                               part.text_begin,part.text_end-part.text_begin)}});
        emitted++;
    }
    return emitted;
}

// Single-call convenience wrapper (first recovered call). suffix retained for
// callers that trim trailing junk; multi-call callers use the vector form.
inline ToolCall parse_bare_tool_call(const std::string& text_in, std::string* prefix,
                                     std::string* suffix, const json* tools = nullptr) {
    auto v = parse_bare_tool_calls(text_in, prefix, tools);
    if (v.empty()) return ToolCall{};
    if (suffix) *suffix = "";
    return v.front();
}

// ---- OpenAI /v1/chat/completions response shaping -------------------------
// Pulled out of server.cu (same rationale as the Anthropic helpers above):
// pure JSON assembly, unit-tested without CUDA. server.cu's job is only to
// wire the engine callbacks that feed `calls`/`text` -- an exact mechanical
// twin of the already-shipped /v1/messages plumbing.

inline json openai_tool_call_json(const std::string& id, const ToolCall& c) {
    return {{"id", id}, {"type", "function"},
            {"function", {{"name", c.name}, {"arguments", json_dump_replace(c.arguments)}}}};
}

// Non-streaming choices[0].message. content is null ONLY when there is at
// least one tool call and no leftover text (matches real OpenAI's
// convention); otherwise content is always the string (possibly empty),
// never null, so a plain content-only turn never confuses a strict client.
// `calls` may include ok==false entries (malformed calls) -- the caller is
// expected to have already folded those into `text` (matching the
// /v1/messages precedent) before calling this. `reasoning` (optional):
// non-empty adds a `reasoning_content` field -- not part of the official
// OpenAI schema, but the de facto convention vLLM/SGLang/llama.cpp's server
// all converged on for surfacing a reasoning model's thinking trace over the
// chat/completions wire; unknown fields are inert to clients that don't
// look for it.
inline json openai_chat_message_json(const std::string& text, const std::vector<ToolCall>& calls,
                                     long rid, const std::string& reasoning = std::string()) {
    json msg = {{"role", "assistant"}};
    json tool_calls = json::array();
    int i = 0;
    for (auto& c : calls)
        if (c.ok)
            tool_calls.push_back(openai_tool_call_json(
                "call_q27_" + std::to_string(rid) + "_" + std::to_string(i++), c));
    bool any_call = !tool_calls.empty();
    msg["content"] = (any_call && text.empty()) ? json(nullptr) : json(text);
    if (any_call) msg["tool_calls"] = tool_calls;
    if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
    return msg;
}

// Streamed reasoning_content delta (see openai_chat_message_json's comment
// for the convention this matches).
inline json openai_reasoning_delta(const std::string& t) {
    return {{"reasoning_content", t}};
}

// One SSE chunk envelope (chat.completion.chunk), shared by every delta this
// endpoint emits (content, tool_calls, or the terminal empty-delta chunk).
inline json openai_stream_chunk(const std::string& id, const std::string& obj, long created,
                                const std::string& model, const json& delta,
                                const char* finish_reason = nullptr) {
    json choice = {{"index", 0}, {"delta", delta},
                   {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)}};
    return {{"id", id}, {"object", obj}, {"created", created}, {"model", model},
            {"choices", json::array({choice})}};
}

// Responses streaming has distinct terminal lifecycle events. One object owns
// every terminal field so the SSE event and nested response status cannot
// diverge when output reaches its public limit.
struct ResponsesTerminalState {
    bool incomplete;
    const char* status;
    const char* event;
};
inline ResponsesTerminalState responses_terminal_state(
    bool limit_reached) {
    return {limit_reached, limit_reached ? "incomplete" : "completed",
            limit_reached ? "response.incomplete" : "response.completed"};
}
inline ResponsesTerminalState responses_terminal_state(
    int produced, int limit, bool budget_truncated) {
    return responses_terminal_state(produced >= limit || budget_truncated);
}

inline bool unfinished_tool_wrapper(
    int produced,int limit,bool budget_truncated,
    StreamSplitter::Chan final_channel) {
    return (produced>=limit || budget_truncated) &&
           final_channel==StreamSplitter::TOOL;
}

inline std::string take_responses_output_text(std::string& text) {
    std::string output=std::move(text);
    text.clear();
    return output;
}

// One streamed tool_calls[] delta entry. Whole-shot (id+name+full arguments
// in a single chunk) rather than incremental-argument streaming -- matches
// the existing /v1/messages input_json_delta precedent (one full
// partial_json chunk per call, not char-by-char) and is spec-valid: a client
// that expects incremental fragments just accumulates a single fragment.
inline json openai_tool_call_delta(int index, const std::string& id, const ToolCall& c) {
    return {{"tool_calls", json::array({{{"index", index},
                                         {"id", id},
                                         {"type", "function"},
                                         {"function", {{"name", c.name},
                                                       {"arguments", json_dump_replace(c.arguments)}}}}})}};
}

// ---- API key authentication -----------------------------------------------
// q27 has no auth by default (loopback-only is the safety net) -- these are
// the pure, testable pieces of an opt-in bearer/x-api-key check, wired into
// a pre-routing handler in server.cu when --api-key/--api-key-file/
// Q27_API_KEY configure at least one key.

// Constant-time string comparison: prevents a timing side-channel where an
// early-exit compare leaks how many leading bytes of a guessed key matched
// via response-time variance. Deliberately does not early-exit on length
// mismatch either (still walks max(a,b) bytes) or on the first bit
// difference (accumulates via OR instead of returning).
inline bool secure_compare(const std::string& a, const std::string& b) {
    size_t n = std::max(a.size(), b.size());
    unsigned char diff = (unsigned char)(a.size() != b.size());
    for (size_t i = 0; i < n; i++) {
        unsigned char ca = i < a.size() ? (unsigned char)a[i] : 0;
        unsigned char cb = i < b.size() ? (unsigned char)b[i] : 0;
        diff |= (unsigned char)(ca ^ cb);
    }
    return diff == 0;
}

// q27 serves two API families with two different native auth header
// conventions -- support both, so neither client population needs special
// configuration:
//   Authorization: Bearer <key>   -- OpenAI / llama.cpp convention (Kilocode
//                                    and other OpenAI-compatible clients)
//   x-api-key: <key>              -- Anthropic convention (what Claude Code
//                                    actually sends to /v1/messages)
// x-api-key wins if a request somehow sends both (arbitrary but
// deterministic; no client sends both in practice). Returns "" if neither
// header is present/well-formed.
inline std::string extract_api_key(const std::string& authorization_header,
                                   const std::string& x_api_key_header) {
    if (!x_api_key_header.empty()) return x_api_key_header;
    static constexpr char bearer[]="bearer";
    constexpr size_t scheme_size=sizeof(bearer)-1;
    if(authorization_header.size()<=scheme_size) return "";
    for(size_t i=0;i<scheme_size;i++) {
        char ch=authorization_header[i];
        if(ch>='A' && ch<='Z') ch=(char)(ch-'A'+'a');
        if(ch!=bearer[i]) return "";
    }
    size_t token=scheme_size;
    if(authorization_header[token]!=' ' && authorization_header[token]!='\t')
        return "";
    while(token<authorization_header.size() &&
          (authorization_header[token]==' ' || authorization_header[token]=='\t'))
        token++;
    if(token<authorization_header.size()) return authorization_header.substr(token);
    return "";
}

// True if `provided` matches ANY configured key. Always scans every key
// (no early return on the first match) so total compare time depends only
// on key count/length, not on which key -- if any -- matched, or how far
// into it a wrong guess got. An empty `provided` is always rejected without
// comparing; an empty KEY is never configured either (load_api_key_file drops
// blank lines, and server.cu's arg parser refuses an empty --api-key /
// Q27_API_KEY at boot), so this is a fast path, not a security-relevant branch.
inline bool api_key_valid(const std::string& provided, const std::vector<std::string>& keys) {
    if (provided.empty()) return false;
    bool any = false;
    for (auto& k : keys) any |= secure_compare(provided, k);

    return any;
}

// 401 body shaped per API family, matching the existing anthropic_error_json
// / OpenAI-error-shape split already used elsewhere in this file for 400s --
// each client SDK gets the error shape it actually parses.
inline std::string auth_error_json(bool anthropic_shape) {
    if (anthropic_shape) return anthropic_error_json("authentication_error", "invalid x-api-key");
    json e = {{"error", {{"message", "Incorrect API key provided"},
                         {"type", "invalid_request_error"},
                         {"code", "invalid_api_key"}}}};
    return e.dump();
}

// Load newline-separated keys from a file (llama.cpp --api-key-file
// convention: one key per line, blank lines and lines starting with '#'
// ignored, surrounding whitespace trimmed). Returns false (and leaves `out`
// untouched) if the file can't be opened, so the caller can fail loudly
// with the actual path rather than silently starting with no auth.
// A file that OPENS but yields no keys (all blank/comment) also returns true
// with `out` unchanged -- the caller must compare sizes and refuse, or an
// operator who asked for auth gets a server with none (server.cu does).
inline bool load_api_key_file(const std::string& path, std::vector<std::string>* out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::vector<std::string> keys;
    std::string line;
    while (std::getline(f, line)) {
        size_t a = line.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        size_t b = line.find_last_not_of(" \t\r\n");
        line = line.substr(a, b - a + 1);
        if (line.empty() || line[0] == '#') continue;
        keys.push_back(line);
    }
    for (auto& k : keys) out->push_back(std::move(k));
    return true;
}

} // namespace q27
