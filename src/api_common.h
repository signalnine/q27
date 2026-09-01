// Shared prompt construction + tool-call parsing for the q27 API endpoints.
// Qwopus (qwen35) tool protocol, from the GGUF chat template:
//   system preamble lists tools as JSON inside <tools>...</tools>
//   model emits  <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
//   results go back as user content wrapped in <tool_response>...</tool_response>
#pragma once
#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
#include "drift_capture.h"

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

// Output-cap field with alias tolerance (issue #39). Modern OpenAI clients
// spell the cap max_completion_tokens; older ones max_tokens; Responses uses
// max_output_tokens. Reading exactly one spelling silently truncates every
// other client at the default. Precedence mirrors the Metal arm's
// max_tokens() helper (metal_server.cpp): aliases first, each endpoint's
// OFFICIAL field last so it wins when several are present. jint semantics
// (null/wrong-type reads as absent) are this arm's deliberate tolerance.
enum class CapApi { Chat, Messages, Responses };
inline long request_max_tokens(const json& b, long dflt, CapApi api) {
    long v = jint(b, "max_output_tokens", dflt);
    v = jint(b, "max_completion_tokens", v);
    v = jint(b, "max_tokens", v);
    if (api == CapApi::Chat)      v = jint(b, "max_completion_tokens", v);
    else if (api == CapApi::Responses) v = jint(b, "max_output_tokens", v);
    // Messages: max_tokens is the official field and was applied last already.
    return v;
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
    std::string role;      // system | user | assistant
    std::string content;   // flattened text (think blocks already reconstructed)
    std::string reasoning{}; // assistant <think> block from history (jinja-compatible);
                             // NSDMI so brace-init sites with 2 fields stay valid under
                             // -Wmissing-field-initializers (metal_server.cpp, -Werror)
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

// Per-model dialect default, set once at boot from the artifact's
// general.name. The 3.8 A/B (BUILDLOG 2026-08-14): instructed in its trained
// XML dialect the model needed ZERO drift rescues vs 4 under JSON, so 3.8-family
// checkpoints default to xml; 3.6 demonstrably complies with JSON and keeps it.
// Matching is normalized (lowercase, alnum-only) because conversion mangles the
// name -- the real 3.8 artifact says "Qwen38 27b Hf". Q27_TOOL_DIALECT=xml|json
// overrides in either direction.
inline bool& tool_dialect_xml_default() {
    static bool v = false;
    return v;
}
inline void set_tool_dialect_for_model(const std::string& meta_json) {
    std::string name;
    try { name = json::parse(meta_json).value("general.name", std::string()); }
    catch (...) {}
    std::string norm;
    for (char c : name)
        if (isalnum((unsigned char)c)) norm += (char)tolower((unsigned char)c);
    tool_dialect_xml_default() = norm.find("qwen38") != std::string::npos;
    fprintf(stderr, "tool dialect: %s (general.name \"%s\"%s)\n",
            tool_dialect_xml_default() ? "xml (trained-format default)" : "json",
            name.c_str(),
            getenv("Q27_TOOL_DIALECT") ? ", Q27_TOOL_DIALECT overrides" : "");
}
inline bool tool_dialect_xml() {
    const char* d = getenv("Q27_TOOL_DIALECT");
    if (d) return !strcmp(d, "xml");
    return tool_dialect_xml_default();
}
// Q27_TOOL_DIALECT=xml|json overrides either direction. The DEFAULT is now
// auto-selected from the artifact's general.name (see
// set_tool_dialect_for_model above): 3.8-family checkpoints boot as XML
// (their trained format -- the 08-14 A/B showed zero drift rescues under
// XML vs four under JSON), and 3.6-family boot as JSON (3.6 demonstrably
// complies with JSON). The knob exists for A/Bs and for fine-tunes whose
// general.name normalization misses the "qwen38" substring.

// Reasoning-effort instruction (Qwen3.8 chat_template.jinja): with thinking
// enabled the trained template injects this line at the HEAD of the system
// block and DEFAULTS to xhigh, so a thinking prompt rendered without it is
// out-of-distribution for the 3.8 checkpoint. Strings are byte-exact from the
// template. Keyed on the same general.name bit as the tool dialect: 3.6's
// template has no reasoning_effort, so 3.6-family renders stay untouched.
// Q27_REASONING_EFFORT: xhigh | low | medium (no line, per the template) |
// off (legacy pre-2026-08-15 rendering, for A/Bs) -- overrides either way.
// think-effort level: 0 = medium/off (no instruction line), 1 = low, 2 = xhigh.
inline int reasoning_effort_env_level() {
    // froggeric v22.3: the safe default is medium (zero injected tokens) to
    // avoid burning the reasoning budget; xhigh/low must be explicitly set.
    const char* e = getenv("Q27_REASONING_EFFORT");
    // NOTE (PR #37 review): the froggeric-template default of "medium" was
    // deliberately NOT inherited -- the effective default stays xhigh/off per
    // boot dialect until a measured benchmark arm justifies the change.
    // medium remains reachable explicitly (env, reasoning_effort field,
    // <|think_medium|> tag).
    const std::string v = e ? e : (tool_dialect_xml_default() ? "xhigh" : "off");
    if (v == "low" || v == "minimal") return 1;
    if (v == "xhigh" || v == "high" || v == "max" || v == "ultracode" || v == "extreme") return 2;
    return 0; // medium / off / none / unknown
}
inline std::string reasoning_effort_line_level(int level) {
    if (level == 2)
        return "Reasoning effort is set to xhigh. Please think carefully through "
               "the task, validate key assumptions, consider plausible alternatives, "
               "and prioritize correctness, consistency, and clarity in the final "
               "answer.";
    if (level == 1)
        return "Reasoning effort is set to low. Keep your thinking brief and "
               "focused, moving directly to the conclusion without unnecessary "
               "elaboration.";
    return ""; // medium (template emits nothing) / off (legacy) / unknown
}
inline std::string reasoning_effort_line() {
    return reasoning_effort_line_level(reasoning_effort_env_level());
}

// v22.3 request-driven template options (items 3/4/5).
// effort: -1 = env/boot default, else 0 medium, 1 low, 2 xhigh.
// auto_disable_thinking_with_tools: start thinking OFF when tools are present.
// tool_format: -1 = boot dialect, 0 json, 1 xml.
// Default for the two-tier tool-error escalation: OFF per PR #37 review --
// a direct probe of the detector over realistic benign outputs measured 3/12
// false positives (grep/diff exit code 1, short 'error: none' summaries), and
// grep returning 1 on no-match is one of the most common tool results in an
// agentic loop. Template parity stays one flag away:
// Q27_TOOL_ERROR_WARNINGS=1 re-enables at boot; a measured arm showing the
// escalation helps would justify flipping the default back.
inline bool tool_error_warnings_env_default() {
    const char* e = getenv("Q27_TOOL_ERROR_WARNINGS");
    if (!e || !*e) return false;
    std::string v = e;
    for (auto& c : v) c = (char)tolower((unsigned char)c);
    return v == "1" || v == "on" || v == "true" || v == "yes";
}
struct TemplateOpts {
    int effort = -1;
    bool auto_disable_thinking_with_tools = false;
    int tool_format = -1;
    bool force_disable_thinking = false; // reasoning_effort = none/off
    // P1b escalation warnings (froggeric Tool Error Recovery). OFF by default
    // per PR #37 review (measured 3/12 FP on benign outputs);
    // Q27_TOOL_ERROR_WARNINGS=1 flips the boot default on, and the
    // per-request tool_error_warnings field overrides either way.
    bool tool_error_warnings = tool_error_warnings_env_default();
    // Pre-rendered <tools> declaration lines in the CLIENT's key order with the
    // template's spacing (see anthropic_tools_decl). nlohmann::json sorts keys,
    // so without this the model sees {"function":{"description",...,"name"},
    // "type"} where its training data has {"type": "function", "function":
    // {"name", "description", "parameters"}}. Empty = legacy sorted dump.
    std::string tools_decl;
};
inline int effort_string_level(std::string v) {
    for (auto& c : v) c = (char)tolower((unsigned char)c);
    if (v == "minimal" || v == "low") return 1;
    if (v == "xhigh" || v == "high" || v == "max" || v == "ultracode" || v == "extreme") return 2;
    return 0; // none/off/medium/unknown
}
inline TemplateOpts template_opts_from_body(const json& body) {
    TemplateOpts o;
    auto read = [&](const json& b) {
        if (b.contains("reasoning_effort") && b["reasoning_effort"].is_string()) {
            const std::string ev = b["reasoning_effort"].get<std::string>();
            std::string el = ev;
            for (auto& c : el) c = (char)tolower((unsigned char)c);
            if (el == "none" || el == "off") o.force_disable_thinking = true; // item 2
            o.effort = effort_string_level(ev);
        }
        if (b.contains("auto_disable_thinking_with_tools") && b["auto_disable_thinking_with_tools"].is_boolean())
            o.auto_disable_thinking_with_tools = b["auto_disable_thinking_with_tools"].get<bool>();
        if (b.contains("tool_error_warnings") && b["tool_error_warnings"].is_boolean())
            o.tool_error_warnings = b["tool_error_warnings"].get<bool>(); // P1b off switch
        if (b.contains("tool_call_format") && b["tool_call_format"].is_string()) {
            const std::string f = b["tool_call_format"].get<std::string>();
            o.tool_format = (f == "json") ? 0 : (f == "xml" ? 1 : -1);
        }
    };
    read(body);
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object())
        read(body["chat_template_kwargs"]);
    // OpenAI Responses effort shapes: output_config / reasoning blocks.
    for (const char* key : {"reasoning", "output_config"}) {
        if (!body.contains(key) || !body[key].is_object()) continue;
        const auto& r = body[key];
        if (r.contains("effort") && r["effort"].is_string())
            o.effort = effort_string_level(r["effort"].get<std::string>());
        if (r.contains("output_effort") && r["output_effort"].is_string())
            o.effort = effort_string_level(r["output_effort"].get<std::string>());
    }
    return o;
}

// P1a (2026-08-22): the <|think_*|> toggle markers from the qwen3.8
// chat_template. Stripping: these are BPE-tokenisable control markers the
// model is trained to read in user/system content; they must not leak into
// the rendered prompt, and they MUTATE the thinking/effort state forward.
inline void strip_think_markers(std::string& s) {
    static const std::string mk[] = {"<|think_off|>", "<|think_on|>",
                                     "<|think_xhigh|>", "<|think_high|>",
                                     "<|think_ultracode|>", "<|think_extreme|>",
                                     "<|think_max|>", "<|think_medium|>",
                                     "<|think_low|>", "<|think_minimal|>"};
    for (const auto& m : mk)
        for (size_t p; (p = s.find(m)) != std::string::npos;) s.erase(p, m.size());
}

// P1a forward-scan of the (flattened) message list for <|think_*|> toggles;
// returns the final thinking/effort state. Mirrors the template: only
// system/developer/user roles are scanned (assistant and tool roles are not),
// and tool responses (user-role, <tool_response>-wrapped) are skipped.
struct ThinkToggles {
    bool thinking = true;
    int effort = 0; // 0 medium, 1 low, 2 xhigh
};
inline ThinkToggles scan_think_toggles(const std::vector<Msg>& msgs, bool base_think,
                                       int base_effort = -1) {
    ThinkToggles st{base_think, base_effort >= 0 ? base_effort : reasoning_effort_env_level()};
    for (const Msg& m : msgs) {
        if (m.role != "system" && m.role != "user") continue;
        if (m.role == "user" && m.content.rfind("<tool_response>", 0) == 0) continue;
        const std::string& c = m.content;
        if (c.find("<|think_off|>") != std::string::npos) st.thinking = false;
        else if (c.find("<|think_on|>") != std::string::npos) st.thinking = true;
        else if (c.find("<|think_xhigh|>") != std::string::npos ||
                 c.find("<|think_high|>") != std::string::npos ||
                 c.find("<|think_ultracode|>") != std::string::npos ||
                 c.find("<|think_extreme|>") != std::string::npos ||
                 c.find("<|think_max|>") != std::string::npos) {
            st.thinking = true; st.effort = 2;
        } else if (c.find("<|think_low|>") != std::string::npos ||
                   c.find("<|think_minimal|>") != std::string::npos) {
            st.thinking = true; st.effort = 1;
        } else if (c.find("<|think_medium|>") != std::string::npos) {
            st.thinking = true; st.effort = 0;
        }
    }
    return st;
}

// P1b: tool-response failure detection, mirrored from the qwen3.8
// chat_template (froggeric v22.3). Suppresses code/grep output and
// exit-code-0, distinguishes strong vs weak error signatures.
inline bool is_tool_response_failure(const std::string& content) {
    std::string full = content;
    for (auto& c : full) c = (char)tolower((unsigned char)c);
    const std::string head = full.substr(0, 120);
    const bool is_code_or_grep =
        full.find("throw new ") != std::string::npos ||
        full.find("throw error") != std::string::npos ||
        full.find("console.error") != std::string::npos ||
        full.find("logger.error") != std::string::npos ||
        full.find("logging.error") != std::string::npos ||
        head.find("import ") != std::string::npos ||
        head.find("def ") != std::string::npos ||
        head.find("function ") != std::string::npos;
    const bool exit_zero =
        head.find("exit code: 0") != std::string::npos ||
        head.find("process exited with code 0") != std::string::npos;
    const bool error_field_ok =
        head.find("\"error\": null") != std::string::npos ||
        head.find("\"error\":null") != std::string::npos ||
        head.find("\"error\": false") != std::string::npos ||
        head.find("\"error\":false") != std::string::npos ||
        head.find("\"error\": \"\"") != std::string::npos ||
        head.find("\"error\":\"\"") != std::string::npos;
    const bool strong_error =
        (head.find("\"error\":") != std::string::npos && !error_field_ok) ||
        head.find("\"status\": \"error\"") != std::string::npos ||
        head.find("\"status\":\"error\"") != std::string::npos ||
        head.find("traceback (most recent call last):") != std::string::npos ||
        head.find("command not found") != std::string::npos ||
        head.find("invalid syntax") != std::string::npos ||
        head.find("fatal:") != std::string::npos ||
        ((head.find("exit code: ") != std::string::npos ||
          head.find("process exited with code") != std::string::npos) && !exit_zero) ||
        head.find("exception:") == 0 ||
        head.find("failed to ") == 0;
    const bool weak_error =
        head.find("error:") != std::string::npos ||
        head.find("err!") != std::string::npos;
    const bool weak_suppressed =
        head.find("$ ") != std::string::npos ||
        head.find("took ") != std::string::npos ||
        content.size() >= 600;
    return !is_code_or_grep && (strong_error || (weak_error && !weak_suppressed));
}

// v22.3: strip a leading think block from content when explicit reasoning is
// present (avoids double-rendering a block the client also sent in text).
inline void strip_leading_think(std::string& content) {
    const char* lead_end = nullptr;
    if (content.rfind("<think>", 0) == 0 && content.find("</think>") != std::string::npos)
        lead_end = "</think>";
    else if (content.rfind("<thinking>", 0) == 0 && content.find("</thinking>") != std::string::npos)
        lead_end = "</thinking>";
    else if (content.rfind("</think>", 0) == 0) lead_end = "</think>";
    else if (content.rfind("</thinking>", 0) == 0) lead_end = "</thinking>";
    if (!lead_end) return;
    std::string close(lead_end);
    size_t p = content.find(close);
    if (p == std::string::npos) return;
    content = content.substr(p + close.size());
    size_t i = 0;
    while (i < content.size() && content[i] == '\n') i++;
    content = content.substr(i);
}
inline std::string tool_response_warning(int failures) {
    if (failures >= 2)
        return "\n\n\u26a0\ufe0f SYSTEM WARNING: " + std::to_string(failures) +
               " consecutive tool errors detected. Your previous approach is "
               "incorrect. You MUST use a fundamentally different approach or "
               "corrected arguments.";
    return "\n\n\u26a0\ufe0f SYSTEM WARNING: The previous tool call returned an "
           "error. Diagnose the failure and retry with completely corrected "
           "arguments.";
}
// Serialize in insertion order with the chat template's spacing: `": "` and
// `", "`, no newlines. This is what minja's `tojson` emits (captured from a
// running llama-server via /apply-template, 2026-08-22, and byte-identical to
// Python jinja2's). nlohmann's dump() has compact or indented, neither of
// which is this. Strings go through dump() so escaping matches.
inline std::string ordered_dump_spaced(const nlohmann::ordered_json& v) {
    if (v.is_object()) {
        std::string s = "{"; bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first) s += ", ";
            first = false;
            s += json(it.key()).dump() + ": " + ordered_dump_spaced(it.value());
        }
        return s + "}";
    }
    if (v.is_array()) {
        std::string s = "["; bool first = true;
        for (const auto& e : v) { if (!first) s += ", "; first = false; s += ordered_dump_spaced(e); }
        return s + "]";
    }
    return v.dump();
}

// The <tools> declaration lines for an Anthropic request, from the RAW body so
// the client's key order survives (the parsed `json` has already sorted it).
// Same acceptance rule as anthropic_tools_json: string, non-empty name; missing
// description -> ""; missing/non-object input_schema -> {}. `keep`, when given,
// restricts to those names (tool_choice selection) without reordering. Each
// line is strip_ctrl'd like the legacy dump: descriptions are caller-authored.
inline std::string anthropic_tools_decl(const std::string& raw_body,
                                        const std::vector<std::string>* keep = nullptr) {
    nlohmann::ordered_json body;
    try { body = nlohmann::ordered_json::parse(raw_body); } catch (...) { return ""; }
    if (!body.is_object() || !body.contains("tools") || !body["tools"].is_array()) return "";
    std::string out;
    for (const auto& t : body["tools"]) {
        if (!t.is_object() || !t.contains("name") || !t["name"].is_string()) continue;
        const std::string name = t["name"].get<std::string>();
        if (name.empty()) continue;
        if (keep && std::find(keep->begin(), keep->end(), name) == keep->end()) continue;
        nlohmann::ordered_json fn;
        fn["name"] = name;
        fn["description"] = (t.contains("description") && t["description"].is_string())
                                ? t["description"].get<std::string>() : std::string();
        fn["parameters"] = (t.contains("input_schema") && t["input_schema"].is_object())
                               ? t["input_schema"] : nlohmann::ordered_json::object();
        nlohmann::ordered_json entry;
        entry["type"] = "function";
        entry["function"] = fn;
        out += "\n" + strip_ctrl(ordered_dump_spaced(entry));
    }
    return out;
}

// The <tools> declaration lines for an OpenAI-shaped request. Entries are
// already in the trained shape, so each accepted one is passed through
// VERBATIM in the client's key order -- that is what minja's `tool | tojson`
// renders, extra keys included -- rather than rebuilt the way
// openai_tools_json() rebuilds them for tool_choice/grammar use. Acceptance is
// openai_tools_json's rule exactly, so the declaration and `selected.tools`
// list the same entries in the same order; `keep` applies the tool_choice
// subset without reordering.
inline std::string openai_tools_decl(const std::string& raw_body,
                                     const std::vector<std::string>* keep = nullptr) {
    nlohmann::ordered_json body;
    try { body = nlohmann::ordered_json::parse(raw_body); } catch (...) { return ""; }
    if (!body.is_object() || !body.contains("tools") || !body["tools"].is_array()) return "";
    std::string out;
    for (const auto& t : body["tools"]) {
        if (!t.is_object() || !t.contains("type") || !t["type"].is_string() ||
            t["type"] != "function") continue;
        if (!t.contains("function") || !t["function"].is_object()) continue;
        const auto& fn = t["function"];
        if (!fn.contains("name") || !fn["name"].is_string() ||
            fn["name"].get_ref<const std::string&>().empty()) continue;
        if (fn.contains("description") && !fn["description"].is_string()) continue;
        if (fn.contains("parameters") && !fn["parameters"].is_object()) continue;
        const std::string name = fn["name"].get<std::string>();
        if (keep && std::find(keep->begin(), keep->end(), name) == keep->end()) continue;
        out += "\n" + strip_ctrl(ordered_dump_spaced(t));
    }
    return out;
}

inline std::string tools_preamble(const json& tools, const std::string& decl = std::string()) {
    std::string s = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    // tool declarations carry caller-controlled (and often third-party-
    // authored) description strings -- same forgery surface as message
    // content (review 2026-07-09 P1 #5). `decl` is the client-ordered,
    // template-spaced rendering (already strip_ctrl'd); the sorted dump is
    // the fallback for callers that have no raw body (Metal, OpenAI path).
    if (!decl.empty()) s += decl;
    else for (auto& t : tools) s += "\n" + strip_ctrl(t.dump());
    s += "\n</tools>\n\n";
    if (tool_dialect_xml()) {
        // The template's text, verbatim (tools/golden/qwen38_tools_request.prompt).
        // The earlier paraphrase dropped the nesting reminder -- the one rule
        // every drift shape of 2026-08-20/21 violated -- and we parsed around
        // that downstream for two days without checking the prompt stated it.
        s += "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
             "<tool_call>\n<function=example_function_name>\n"
             "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
             "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
             "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
             "<IMPORTANT>\nReminder:\n"
             "- Function calls MUST follow the specified format: an inner <function=...></function> "
             "block must be nested within <tool_call></tool_call> XML tags\n"
             "- Required parameters MUST be specified\n"
             "- You may provide optional reasoning for your function call in natural language "
             "BEFORE the function call, but NOT after\n"
             "- If there is no function call available, answer the question like normal with your "
             "current knowledge and do not tell the user about function calls\n</IMPORTANT>";
    } else {
        s += "For each function call, return a JSON object with the function name "
             "and arguments inside <tool_call></tool_call> tags:\n<tool_call>\n{\"name\": "
             "<function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n<IMPORTANT>\n"
             "- Required parameters MUST be specified.\n- You may provide optional reasoning "
             "before the function call, but never after it.\n- If no function call is needed, "
             "answer normally and do not mention the tool interface.\n</IMPORTANT>";
    }
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
                                 const json* unavailable_tools = nullptr,
                                 const TemplateOpts* opts = nullptr) {
    std::string p;
    size_t start = 0;
    std::string sys;
    // v22.3: merge ALL consecutive leading system/developer messages into one
    // system block (joined with a blank line), mirroring the template's head
    // loop -- not just messages[0].
    while (start < msgs.size() &&
           (msgs[start].role == "system" || msgs[start].role == "developer")) {
        std::string part = strip_ctrl(msgs[start].content);
        if (!sys.empty()) sys += "\n\n";
        sys += part;
        start++;
    }
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
    // P1a: forward-scan the message list for <|think_*|> toggles. The final
    // thinking/effort state drives BOTH the reasoning_instructions line at the
    // system head and the generation prompt below (mirrors the template's
    // stateful ns_state). Template order: reasoning_instructions FIRST, then
    // tools, then the client system content. Items 3/4: a request-driven base
    // effort overrides the env/boot default, and auto_disable_thinking_with_tools
    // starts thinking OFF when tools are present.
    bool base_think = think;
    if (opts && opts->auto_disable_thinking_with_tools && tools.is_array() && !tools.empty())
        base_think = false;
    if (opts && opts->force_disable_thinking) base_think = false; // reasoning_effort none/off
    const ThinkToggles tt = scan_think_toggles(msgs, base_think, opts ? opts->effort : -1);
    const std::string effort = tt.thinking ? reasoning_effort_line_level(tt.effort)
                                           : std::string();
    strip_think_markers(sys);
    if (has_tools || has_unavailable || !sys.empty() || !tool_instruction.empty() ||
        !effort.empty()) {
        p += "<|im_start|>system\n";
        bool need_separator=false;
        auto append_system_part=[&](const std::string& part) {
            if (part.empty()) return;
            if (need_separator) p += "\n\n";
            p += part;
            need_separator=true;
        };
        append_system_part(effort);
        if (has_tools) append_system_part(tools_preamble(tools, opts ? opts->tools_decl : std::string()));
        if (has_unavailable)
            append_system_part(unavailable_tools_preamble(*unavailable_tools));
        append_system_part(sys);
        append_system_part(strip_ctrl(tool_instruction));
        p += "<|im_end|>\n";
    }
    if (sys_off) *sys_off = p.size();  // 0 when no system block was emitted
    int tool_failures = 0;
    // P1b switch: env/boot default unless the request carries an explicit
    // tool_error_warnings field (TemplateOpts default member is env-seeded,
    // so this covers both the nullptr and the populated-opts paths).
    const bool warn_on = opts ? opts->tool_error_warnings
                              : tool_error_warnings_env_default();
    for (size_t i = start; i < msgs.size(); i++) {
        const Msg& m = msgs[i];
        std::string content = m.content;
        // P1b: consecutive tool-error tracking. Tool responses reach this
        // renderer already <tool_response>-wrapped in a user turn.
        const bool toolresp = m.role == "user" &&
                              content.rfind("<tool_response>", 0) == 0;
        if (warn_on && toolresp)
            tool_failures = is_tool_response_failure(content)
                                          ? tool_failures + 1 : 0;
        else if (m.role == "user") tool_failures = 0; // plain user resets
        strip_think_markers(content);
        // P2: consecutive tool responses group under ONE user turn, each kept
        // as its own <tool_response> block (mirrors the template's prev_role
        // grouping -- no <|im_start|>/<|im_end|> between them).
        if (toolresp) {
            const bool prev_toolresp = i > start && msgs[i - 1].role == "user" &&
                                       msgs[i - 1].content.rfind("<tool_response>", 0) == 0;
            const bool next_toolresp = i + 1 < msgs.size() && msgs[i + 1].role == "user" &&
                                       msgs[i + 1].content.rfind("<tool_response>", 0) == 0;
            if (!prev_toolresp) p += "<|im_start|>user\n";
            if (tool_failures >= 1) {
                // warning goes INSIDE the response block, before the closer
                // (the template emits it between content and </tool_response>)
                std::string warn = tool_response_warning(tool_failures);
                size_t close = content.rfind("\n</tool_response>");
                if (close != std::string::npos) content.insert(close, warn);
                else content += warn;
            }
            p += strip_ctrl(content);
            p += next_toolresp ? "\n" : "<|im_end|>\n";
            continue;
        }
        // normal (non-tool) message
        p += "<|im_start|>" + strip_ctrl(m.role) + "\n";
        // v22.3 E: EVERY assistant history turn is wrapped in <think>...</think>
        // (even when reasoning is empty -> an empty think block). If the
        // client also sent the think block in text, strip it so the block is
        // not double-rendered.
        if (m.role == "assistant") {
            strip_leading_think(content);
            p += "<think>\n" + strip_ctrl(m.reasoning) + "\n</think>\n\n";
        }
        p += strip_ctrl(content) + "<|im_end|>\n";
    }
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
    // P1a: the generation prompt follows the final toggle state, not just the
    // caller's `think` flag. think=false -> empty CLOSED block; true -> OPEN.
    if (!tt.thinking) p += "<think>\n\n</think>\n\n";
    else p += "<think>\n";
    return p;
}

// P3 request-driven truncation: read max_tool_arg_chars / max_tool_response_chars
// from the request body (top-level or chat_template_kwargs). 0 when absent.
inline long body_num(const json& v) {
    if (v.is_number_float()) return (long)v.get<double>();
    if (v.is_number_integer()) return v.get<long>();
    if (v.is_number_unsigned()) return (long)v.get<uint64_t>();
    return 0;
}
inline long request_max_chars(const json& body, const char* key) {
    if (body.contains(key) && body[key].is_number()) return body_num(body[key]);
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
        const auto& k = body["chat_template_kwargs"];
        if (k.contains(key) && k[key].is_number()) return body_num(k[key]);
    }
    return -1; // absent -> text helpers fall back to the env vars
}

inline std::string tool_call_text(const std::string& name, const json& args,
                                  long max_arg_chars = -1) {
    // v22.3 (item 7): a raw string argument is echoed verbatim (not JSON-encoded),
    // matching the template's `_args = tc.arguments` when arguments is a string.
    std::string a = args.is_string() ? args.get<std::string>() : args.dump();
    if (max_arg_chars < 0) { // no explicit request value -> env fallback
        const char* e = getenv("Q27_MAX_TOOL_ARG_CHARS");
        max_arg_chars = e ? atol(e) : 0;
    }
    if (max_arg_chars > 0 && (long)a.size() > max_arg_chars)
        a = a.substr(0, (size_t)max_arg_chars) + "\n[TRUNCATED - original length " +
            std::to_string(a.size()) + " chars]";
    return "<tool_call>\n{\"name\": \"" + name + "\", \"arguments\": " + a +
           "}\n</tool_call>";
}

// v22.3 (item 6): render an assistant tool call in the PROMPT using the active
// tool dialect. For the XML dialect use the per-parameter form the template
// emits (<function=NAME><parameter=K>v</parameter>...</function>); otherwise
// fall back to the JSON tool_call_text.
inline std::string tool_call_text_dialect(const std::string& name, const json& args,
                                          long max_arg_chars = -1) {
    if (max_arg_chars < 0) { // no explicit request value -> env fallback
        const char* e = getenv("Q27_MAX_TOOL_ARG_CHARS");
        max_arg_chars = e ? atol(e) : 0;
    }
    if (!tool_dialect_xml()) return tool_call_text(name, args, max_arg_chars);
    auto trunc = [&](std::string v) -> std::string {
        if (max_arg_chars > 0 && (long)v.size() > max_arg_chars)
            return v.substr(0, (size_t)max_arg_chars) + "\n[TRUNCATED - original length " +
                   std::to_string(v.size()) + " chars]";
        return v;
    };
    std::string s = "<tool_call>\n<function=" + name + ">\n";
    if (args.is_object()) {
        for (auto it = args.begin(); it != args.end(); ++it)
            s += "<parameter=" + it.key() + ">\n" +
                 trunc(it.value().is_string() ? it.value().get<std::string>()
                                              : it.value().dump()) +
                 "\n</parameter>\n";
    } else if (args.is_string()) {
        s += trunc(args.get<std::string>()) + "\n";
    } else {
        s += trunc(args.dump()) + "\n";
    }
    s += "</function>\n</tool_call>";
    return s;
}

inline std::string tool_response_text(const std::string& out,
                                      long max_response_chars = -1) {
    std::string o = out;
    if (max_response_chars < 0) { // no explicit request value -> env fallback
        const char* e = getenv("Q27_MAX_TOOL_RESPONSE_CHARS");
        max_response_chars = e ? atol(e) : 0;
    }
    // v22.3 (item 8): truncation applies only when the dialect is NOT json.
    if (tool_dialect_xml() && max_response_chars > 0 && (long)o.size() > max_response_chars)
        o = o.substr(0, (size_t)max_response_chars) + "\n[TRUNCATED - original length " +
            std::to_string(o.size()) + " chars]";
    // The template runs every message's content through |trim. A tool result
    // ending in "\n" otherwise renders a blank line inside <tool_response>
    // that llama.cpp does not (golden test).
    const size_t a = o.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) o.clear();
    else o = o.substr(a, o.find_last_not_of(" \t\r\n") - a + 1);
    return "<tool_response>\n" + o + "\n</tool_response>";
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
    // OpenAI Responses: reasoning / output_config blocks. Effort is consumed by
    // TemplateOpts (prompt line); here we handle disabled/enabled + budget.
    for (const char* key : {"reasoning", "output_config"}) {
        if (!body.contains(key) || !body[key].is_object()) continue;
        const auto& r = body[key];
        if (r.contains("disabled") && r["disabled"].is_boolean()) {
            c.enabled = !r["disabled"].get<bool>();
            c.enabled_set = true;
        } else if (r.contains("enabled") && r["enabled"].is_boolean()) {
            c.enabled = r["enabled"].get<bool>();
            c.enabled_set = true;
        }
        if (r.contains("budget_tokens")) take_budget(r["budget_tokens"]);
        if (r.contains("output_budget_tokens")) take_budget(r["output_budget_tokens"]);
    }
    // some clients send a flat top-level budget_tokens
    if (body.contains("budget_tokens")) take_budget(body["budget_tokens"]);
    // item 2: reasoning_effort none/off disables thinking end-to-end (engine
    // think mode, not just the prompt render).
    auto effort_disable = [&](const json& b) {
        if (b.contains("reasoning_effort") && b["reasoning_effort"].is_string()) {
            std::string ev = b["reasoning_effort"].get<std::string>();
            for (auto& ch : ev) ch = (char)tolower((unsigned char)ch);
            if (ev == "none" || ev == "off") { c.enabled = false; c.enabled_set = true; }
        }
    };
    effort_disable(body);
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object())
        effort_disable(body["chat_template_kwargs"]);
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

// Q27_THINK_BUDGET_FRAC overrides it, read once (server-lifetime knob, one leg
// per run). Added 2026-08-20 to answer the comment above on its own terms: the
// fraction is a judgement call and the note asks for a measured one to replace
// it, which is impossible while it is a compile-time constant. Out-of-range or
// unparseable values fall back to the default rather than half-applying.
inline double think_budget_frac() {
    static const double v = [] {
        const char* e = getenv("Q27_THINK_BUDGET_FRAC");
        if (!e) return THINK_BUDGET_FRAC;
        const double d = atof(e);
        return (d > 0.0 && d < 1.0) ? d : THINK_BUDGET_FRAC;
    }();
    return v;
}

// Resolve the server budget for a request whose public cap is `n_max`.
// `flag` is --think-budget: <0 = use the fractional default, 0 = unbounded,
// >0 = an explicit absolute cap. The fractional default applies only when the
// prompt already opened THINK: standard thinking-enabled serving starts there,
// while a later model-generated block is intentionally unbounded unless the
// client or an explicit positive server flag arms that guard.
inline int think_budget_default(int flag, int n_max) {
    if (flag == 0) return -1;
    if (flag > 0) return flag;
    return n_max > 0 ? (int)(think_budget_frac() * n_max) : -1;
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
// Both tool-list shapes the server hands the parser (OpenAI function
// entries and bare Anthropic entries).
inline void declared_tool_names(const json* tools, DriftNames& names) {
    if (!tools || !tools->is_array()) return;
    for (const auto& t : *tools) {
        if (!t.is_object()) continue;
        if (t.contains("function") && t["function"].is_object() &&
            t["function"].contains("name") && t["function"]["name"].is_string())
            names.insert(t["function"]["name"].get<std::string>());
        else if (t.contains("name") && t["name"].is_string())
            names.insert(t["name"].get<std::string>());
    }
}

// Drift corpus: remember what this request declared (see drift_capture.h).
// No-op unless Q27_DRIFT_CORPUS is set.
inline void drift_register_tools(const json& tools) {
    if (!drift_corpus_path()) return;
    DriftNames names;
    declared_tool_names(&tools, names);
    drift_register_names(names);
}

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
    drift_register_tools(out);
    return out;
}

// Anthropic messages -> Msg list (thinking + tool_use reconstructed to
// model markers, tool_result wrapped in <tool_response>)
inline std::vector<Msg> anthropic_msgs(const json& body) {
    std::vector<Msg> msgs;
    // P3: request-driven truncation limits (top-level or chat_template_kwargs)
    const long max_arg = request_max_chars(body, "max_tool_arg_chars");
    const long max_resp = request_max_chars(body, "max_tool_response_chars");
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
            msgs.push_back({"system", sys, {}});
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
        if (!m.contains("content")) { msgs.push_back({role, content, {}}); continue; }
        if (m["content"].is_string()) content = m["content"];
        else if (m["content"].is_array())
            for (auto& part : m["content"]) {
                if (!part.is_object()) continue; // bare string in a content array
                std::string ty = part.value("type", "");
                if (ty == "text") content += part.value("text", "");
                else if (ty == "thinking") think += part.value("thinking", "");
                else if (ty == "tool_use") {
                    if (!content.empty() && content.back() != '\n') content += "\n";
                    content += tool_call_text_dialect(part.value("name", ""),
                                                      part.contains("input") ? part["input"]
                                                                             : json::object(),
                                                      max_arg);
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
                    content += tool_response_text(rc, max_resp);
                }
            }
        // reasoning block is emitted by the renderer (chatml_prompt), not
        // flattened here -- keeps the jinja's format and per-message logic
        // (preserve_reasoning / <|think_*|> toggles) in one place.
        msgs.push_back({role, content, think});
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
    drift_register_tools(out);
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
    // P3: request-driven truncation limits (top-level or chat_template_kwargs)
    const long max_arg = request_max_chars(body, "max_tool_arg_chars");
    const long max_resp = request_max_chars(body, "max_tool_response_chars");
    if (!body.contains("messages") || !body["messages"].is_array()) return msgs;
    for (auto& m : body["messages"]) {
        if (!m.is_object()) continue;
        std::string role = m.value("role", "user");
        if (role == "developer") role = "system";
        std::string content, reasoning;
        if (m.contains("content")) {
            if (m["content"].is_string()) content = m["content"];
            else if (m["content"].is_array())
                for (auto& part : m["content"])
                    if (part.is_object() && part.value("type", "") == "text")
                        content += part.value("text", "");
        }
        if (role == "tool") {
            msgs.push_back({"user", tool_response_text(content, max_resp), {}});
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
                content += tool_call_text_dialect(name, args, max_arg);
            }
        }
        if (role == "assistant" && m.contains("reasoning_content")) {
            const json& rc = m["reasoning_content"];
            if (rc.is_string()) reasoning = rc.get<std::string>();
            else if (rc.is_array())
                for (auto& part : rc)
                    if (part.is_object() && part.value("type", "") == "text")
                        reasoning += part.value("text", "");
        }
        msgs.push_back({role, content, reasoning});
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

// Per-tool parameter-key extraction, aligned with OpenAIToolSelection::names.
// Used by the XML-dialect toolgram (--constrain-tools on Qwen3.8) to build
// its schema-aware parameter-key allowlist: only declared property names are
// legal as <parameter=...> keys, which is what stops the model from
// emitting undeclared keys in its XML tool body. Returns one vector per tool,
// in the same order as `selected.names`; tools whose parameters block is
// missing/malformed get an empty list (the grammar then accepts any key for
// that tool -- a permissive fallback for malformed schemas rather than a
// silent rejection of the whole request).
inline std::vector<std::vector<std::string>>
tool_param_keys_per_name(const OpenAIToolSelection& selected) {
    std::vector<std::vector<std::string>> out;
    out.reserve(selected.tools.size());
    for (const auto& t : selected.tools) {
        std::vector<std::string> keys;
        if (t.contains("function") && t["function"].contains("parameters") &&
            t["function"]["parameters"].contains("properties") &&
            t["function"]["parameters"]["properties"].is_object()) {
            for (auto it = t["function"]["parameters"]["properties"].begin();
                 it != t["function"]["parameters"]["properties"].end(); ++it) {
                keys.push_back(it.key());
            }
        }
        out.push_back(std::move(keys));
    }
    return out;
}

// json-array overload (for callers that already moved `tools` out of the
// selection; the two backends' selected-tools arrays are both aligned with
// tool_names_v).
inline std::vector<std::vector<std::string>>
tool_param_keys_per_name(const json& tools) {
    std::vector<std::vector<std::string>> out;
    if (!tools.is_array()) return out;
    out.reserve(tools.size());
    for (const auto& t : tools) {
        std::vector<std::string> keys;
        if (t.contains("function") && t["function"].contains("parameters") &&
            t["function"]["parameters"].contains("properties") &&
            t["function"]["parameters"]["properties"].is_object()) {
            for (auto it = t["function"]["parameters"]["properties"].begin();
                 it != t["function"]["parameters"]["properties"].end(); ++it) {
                keys.push_back(it.key());
            }
        }
        out.push_back(std::move(keys));
    }
    return out;
}

// REQUIRED-argument lists, aligned with tool_names_v exactly like
// tool_param_keys_per_name above. Split out rather than folded into that
// function because the allowlist and the required set answer different
// questions: "may this key appear?" vs "must it have appeared before the call
// may close?". Only the first was ever threaded into the grammar, which is why
// --constrain-tools emitted schema-invalid calls with a well-formed shape
// (issue #2): a <function=write> that never produced <parameter=path> closed
// cleanly, the server returned 200, no drift was logged, and the client's
// schema validation was the only thing that noticed.
inline std::vector<std::vector<std::string>>
tool_required_keys_per_name(const json& tools) {
    std::vector<std::vector<std::string>> out;
    if (!tools.is_array()) return out;
    out.reserve(tools.size());
    for (const auto& t : tools) {
        std::vector<std::string> keys;
        if (t.contains("function") && t["function"].contains("parameters") &&
            t["function"]["parameters"].contains("required") &&
            t["function"]["parameters"]["required"].is_array()) {
            for (const auto& k : t["function"]["parameters"]["required"])
                if (k.is_string()) keys.push_back(k.get<std::string>());
        }
        out.push_back(std::move(keys));
    }
    return out;
}
inline std::vector<std::vector<std::string>>
tool_required_keys_per_name(const OpenAIToolSelection& selected) {
    json arr = json::array();
    for (const auto& t : selected.tools) arr.push_back(t);
    return tool_required_keys_per_name(arr);
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

// NATIVE XML TOOL DIALECT (2026-08-14). Qwen3.8's chat template trains
//   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>...\n</function>\n</tool_call>
// while tools_preamble historically instructed JSON-in-<tool_call>. The 3.8
// drift catalogue (modes 14-16) is the model reverting to this trained form
// under a JSON instruction; parsing the well-formed dialect first-class treats
// the cause rather than the symptoms. Value typing follows the template's own
// serialization: non-strings are written with tojson, so a value that parses
// as non-string JSON is used parsed, anything else stays a raw string.
// ONE RECOGNIZER FOR THE `<function...>` OPENER FAMILY (2026-08-20
// consolidation). Four spellings of the same opener were observed in the wild
// inside two days, and each one had arrived as its own branch or its own drift
// mode:
//
//   <function=Bash>          the trained form
//   <function="Bash">        trained, name quoted        (killed sessions at 50K)
//   <function name="Bash">   HTML-attribute syntax       (drift mode 19)
//   <function name=Bash>     attribute, unquoted         (not yet seen)
//
// Every one of them was a session-killer: the call lands in the text channel,
// Claude Code sees a final turn with no tool_use, and exits `completed` with
// the work unfinished. Adding a case per spelling was losing the race -- the
// model has many ways to write a name and only one of them was trained.
//
// So the opener is parsed by GRAMMAR rather than by literal: "<function", an
// optional `name` key, an `=`, an optionally quoted identifier, then `>`. That
// covers all four above, including the one nobody has reported yet.
//
// `<function>` with no name is deliberately NOT claimed: that is mode 16's
// JSON-wrapped shape and it has its own recovery.
inline bool parse_function_opener(const std::string& seg, size_t b, std::string& name,
                                  size_t& after) {
    if (seg.compare(b, 9, "<function") != 0) return false;
    size_t i = b + 9;
    const size_t gt = seg.find('>', i);
    if (gt == std::string::npos) return false;
    while (i < gt && isspace((unsigned char)seg[i])) i++;
    if (seg.compare(i, 4, "name") == 0) { // optional attribute key
        i += 4;
        while (i < gt && isspace((unsigned char)seg[i])) i++;
    }
    if (i >= gt || seg[i] != '=') return false; // `<function>` -> mode 16's
    i++;
    while (i < gt && isspace((unsigned char)seg[i])) i++;
    name = seg.substr(i, gt - i);
    after = gt + 1;
    return true;
}

// Both spellings of the PARAMETER tag, mirroring parse_function_opener above:
//   <parameter=KEY>         the trained form
//   <parameter name="KEY">  the attribute form, quoted or not
// The function opener absorbed both spellings in 0f46684; the parameter tag
// only ever accepted the first, so the same drift habit one level down dropped
// the call (2026-08-21, from the surviving misses of the parity sweep).
inline bool parse_parameter_opener(const std::string& seg, size_t b, std::string& key,
                                   size_t& after) {
    if (seg.compare(b, 10, "<parameter") != 0) return false;
    size_t i = b + 10;
    const size_t gt = seg.find('>', i);
    if (gt == std::string::npos) return false;
    while (i < gt && isspace((unsigned char)seg[i])) i++;
    if (seg.compare(i, 4, "name") == 0) { i += 4; while (i < gt && isspace((unsigned char)seg[i])) i++; }
    if (i >= gt || seg[i] != '=') return false;
    i++;
    while (i < gt && isspace((unsigned char)seg[i])) i++;
    key = seg.substr(i, gt - i);
    while (!key.empty() && isspace((unsigned char)key.back())) key.pop_back();
    if (key.size() >= 2 && (key.front() == '"' || key.front() == '\'') &&
        key.back() == key.front())
        key = key.substr(1, key.size() - 2);
    after = gt + 1;
    return true;
}

// Every spelling of a bare native opener the parser accepts -- see the
// table's users at bare_native_opener_position (the streaming holdback) and
// the batch scanner in parse_bare_tool_calls_impl. One table, three readers.
inline size_t bare_native_opener_len_at(const std::string& text,size_t i) {
    static const char* const k[]={"<function=","<name>","<parameter_name>"};
    for(const char* o:k) {
        const size_t n=std::char_traits<char>::length(o);
        if(text.compare(i,n,o)==0) return n;
    }
    return 0;
}

inline bool parse_native_xml_call(const std::string& seg, ToolCall& tc) {
    size_t b = seg.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return false;
    // Junk ahead of the opener (2026-08-25, pylint-6903): `<tool_use>\n<tool>\n
    // <parameter_name>\n<parameter_name>Read\n</parameter>\n<parameter=...` --
    // wrapper-ish tags the model invented, and a name tag opened twice, the
    // first one empty. Skip them; the opener that carries a name decides.
    for (;;) {
        bool skipped = false;
        for (const char* junk : {"<tool_use>", "<tool>", "<tool_calls>"}) {
            const size_t n = strlen(junk);
            if (seg.compare(b, n, junk) == 0) {
                b = seg.find_first_not_of(" \t\r\n", b + n);
                if (b == std::string::npos) return false;
                skipped = true;
            }
        }
        const size_t nl = bare_native_opener_len_at(seg, b);
        if (nl && seg.compare(b, 10, "<function=") != 0) {
            const size_t q = seg.find_first_not_of(" \t\r\n", b + nl);
            if (q != std::string::npos && seg[q] == '<' && bare_native_opener_len_at(seg, q)) {
                b = q;   // an empty name tag, then the real one
                skipped = true;
            }
        }
        if (!skipped) break;
    }
    size_t p;
    size_t name_tag = 0;
    if (parse_function_opener(seg, b, tc.name, p)) {
        // name normalization below strips the quoting; the grammar above
        // already absorbed the attribute-vs-equals spelling.
    } else if ((name_tag = bare_native_opener_len_at(seg, b)) != 0) {
        // XML drift mode 18 (issue #24): the model drops the `<function=`
        // opener and writes the bare trained `<name>` tag --
        //   <name>task\n</parameter>\n<parameter=description>\n...
        // Structurally the XML twin of JSON mode 10 (dropped `{"name":"`
        // opener, 601d7c3): the payload is intact, only the opener is gone,
        // and refusing it dumps a real tool call into the text channel where
        // the agent reads it as prose and stops. The name runs to end-of-line
        // rather than to a '>' -- the tag is already closed. Any stray
        // `</parameter>` before the first real `<parameter=` is ignored by the
        // scan below, which searches forward for the opener.
        size_t ns = b + name_tag;
        // Tolerate the name on the NEXT line after `<name>` -- a real
        // emission (and this harness's mangled tool transport) can open
        // `<name>` and drop the identifier on the following line. The old
        // code read "up to the first newline", yielding an EMPTY name and
        // refusing the whole call. Mirror parse_function_opener's leading-
        // whitespace skip so `<name>\nbash\n<parameter=...>` -> name="bash".
        while (ns < seg.size() && isspace((unsigned char)seg[ns])) ns++;
        size_t ne = seg.find('\n', ns);
        if (ne == std::string::npos) ne = seg.size();
        tc.name = seg.substr(ns, ne - ns);
        while (!tc.name.empty() &&
               (isspace((unsigned char)tc.name.back()) || tc.name.back() == '>'))
            tc.name.pop_back();
        // `<function=` is self-evidently a call, so it may carry zero
        // parameters. A bare `<name>` is weaker evidence -- ordinary prose can
        // open that way -- so require corroborating dialect structure before
        // claiming the whole segment as a tool call.
        if (seg.find("<parameter=", ne) == std::string::npos &&
            seg.find("</function>", ne) == std::string::npos)
            return false;
        p = ne;
    } else {
        return false;
    }
    // NAME NORMALIZATION (2026-08-20). Observed live, killing sessions at
    // ~50K tokens: `<function="Bash">` -- the trained opener with the name
    // QUOTED. It matches the `<function=` branch above, so the dialect is
    // recognised, but the name comes out as the six characters "Bash" WITH the
    // quotes, matches no declared tool, and the whole call is refused. The
    // structure was never the problem; two quote characters were.
    //
    // Applied to every branch rather than just that one: a name is a tool
    // identifier, and no declared tool has leading or trailing quotes or
    // whitespace in it, so stripping them cannot turn a valid name into a
    // different valid name.
    while (!tc.name.empty() && isspace((unsigned char)tc.name.front())) tc.name.erase(0, 1);
    while (!tc.name.empty() && isspace((unsigned char)tc.name.back())) tc.name.pop_back();
    if (tc.name.size() >= 2 && (tc.name.front() == '"' || tc.name.front() == '\'') &&
        tc.name.back() == tc.name.front()) {
        tc.name = tc.name.substr(1, tc.name.size() - 2);
    }
    if (tc.name.empty()) return false;
    tc.arguments = json::object();
    static const std::string PO = "<parameter", PC = "</parameter>";
    // Next position at or after `from` that opens a parameter in EITHER
    // spelling. Used for the scan and for the two later-parameter checks, so
    // all three agree on what counts as a parameter.
    auto next_param = [&seg](size_t from, std::string* k, size_t* aft) -> size_t {
        for (size_t c = seg.find(PO, from); c != std::string::npos; c = seg.find(PO, c + 10)) {
            std::string kk;
            size_t aa;
            if (parse_parameter_opener(seg, c, kk, aa)) {
                if (k) *k = kk;
                if (aft) *aft = aa;
                return c;
            }
        }
        return std::string::npos;
    };
    while (true) {
        std::string key;
        size_t after = 0;
        size_t ps = next_param(p, &key, &after);
        if (ps == std::string::npos) break;
        size_t vs = after;
        if (vs < seg.size() && seg[vs] == '\n') vs++;
        // The XML dialect has no escape mechanism, so a value may legitimately
        // contain `</parameter>` -- editing this parser's own tests, a doc
        // about tool calling, a prompt template. Taking the FIRST closer
        // truncated there and silently lost the rest of the value. Take the
        // LAST closer before this parameter's boundary instead: an earlier one
        // followed by more of the same value was content. Same collision #31
        // solved for `</tool_call>` a layer up, resolved the same way -- by
        // waiting for the evidence rather than guessing at first sight.
        //
        // The boundary is the next parameter opener or `</function>`, whichever
        // comes first, so this cannot reach across into a later parameter's
        // closer. No closer at all before that boundary still means an
        // unterminated mid-stream parameter, which the branch below refuses as
        // genuine mangling exactly as before -- that check is subsumed here
        // rather than dropped.
        size_t ve = std::string::npos;
        {
            // The bound is the next PARAMETER OPENER and nothing else. Using
            // `</function>` as well looked tighter and was wrong: a value
            // containing `</function>` (editing a template, a doc, this
            // parser's own tests) bounded the search before its real closer,
            // and the parameter read as unterminated mid-stream -- refused.
            // The opener is sufficient, because a later call's `</parameter>`
            // is always preceded by that call's own `<parameter=`.
            const size_t next_po = next_param(vs, nullptr, nullptr);
            const size_t bnd = next_po == std::string::npos ? seg.size() : next_po;
            // For the LAST parameter the bound is end-of-segment, and the
            // model does not always stop at its closer. The drift corpus
            // (2026-08-25, Qwen3.8 suite + SWE-bench captures) has three
            // shapes of what follows: a truncated second closer and a stray
            // parameter closer (`</parameter>\n</function>\n</functio\n
            // </parameter>`), a doubled closer then a stray one, and prose
            // then a stray closer pair (`</function>\nSome prose\n
            // </parameter>\n</function>`). Under the last-closer rule every
            // one of them swallowed the tail into the value -- harmless on a
            // description, dialect markup written into a file on `content`.
            //
            // So for the last parameter: the FIRST closer followed by
            // `</function>` ended it. A value whose own last line is
            // `</parameter>` (the #31 collision the last-closer rule exists
            // for) still reads whole, because its closer is the one
            // `</function>` follows. What is given up is a value that itself
            // contains `</parameter>\n</function>` and then MORE content: its
            // bytes are identical to the observed garble, the garble has been
            // seen and that value has not, and losing a tail is the safer
            // failure. Non-last parameters are unchanged: their bound is the
            // next opener.
            auto starts_with_fn = [&](size_t from, size_t to) {
                while (from < to && isspace((unsigned char)seg[from])) from++;
                return from + 11 <= to && seg.compare(from, 11, "</function>") == 0;
            };
            std::vector<size_t> cands;
            for (size_t c = seg.find(PC, vs); c != std::string::npos && c < bnd;
                 c = seg.find(PC, c + PC.size()))
                cands.push_back(c);
            if (!cands.empty()) {
                ve = cands.back();
                if (next_po == std::string::npos) {   // the last parameter
                    for (size_t i = 0; i < cands.size(); i++) {
                        const size_t to = i + 1 < cands.size() ? cands[i + 1] : bnd;
                        if (starts_with_fn(cands[i] + PC.size(), to)) { ve = cands[i]; break; }
                    }
                }
            }
        }
        bool unterminated = ve == std::string::npos;
        if (unterminated) {
            // Final-parameter leniency (2026-08-15, full-suite capture): the
            // model streams `<function=Bash>\n<parameter=arguments>\n{json}`
            // and stops -- no closers at all. Mode 14 already tolerated this
            // shape under <tool_name>; refusing it here was the crack the
            // first full suite fell through. Close at end-of-span, but only
            // for the LAST parameter: an unterminated one followed by another
            // <parameter= would be genuine mangling and still refuses.
            if (next_param(vs, nullptr, nullptr) != std::string::npos) return false;
            size_t fe = seg.find("</function>", vs);
            ve = fe == std::string::npos ? seg.size() : fe;
        }
        size_t vend = ve;
        // Strip EXACTLY the delimiter newline, mirroring the single leading
        // '\n' consumed above. This was a greedy trim over every trailing '\n'
        // and ' ', which silently rewrote the value -- invisible to Bash and
        // Read, fatal to Edit, whose oldString has to match the file byte for
        // byte. Reported live as "File Edit sometime still acting up": it broke
        // exactly when the edited region ended in whitespace. Everything
        // between the delimiters is content.
        if (vend > vs && seg[vend - 1] == '\n') {
            vend--;
            if (vend > vs && seg[vend - 1] == '\r') vend--;  // CRLF: both bytes
        }
        std::string val = seg.substr(vs, vend - vs);
        if (key.empty()) return false;
        json parsed;
        bool as_json = false;
        try { parsed = json::parse(val); as_json = !parsed.is_string(); } catch (...) {}
        // Tolerate mode-14's trailing unbalanced brace on a JSON payload.
        if (!as_json && !val.empty() && val.back() == '}') {
            std::string t2 = val.substr(0, val.size() - 1);
            while (!t2.empty() && isspace((unsigned char)t2.back())) t2.pop_back();
            try { parsed = json::parse(t2); as_json = !parsed.is_string(); } catch (...) {}
        }
        if (as_json && key == "arguments" && parsed.is_object()) {
            // arguments-as-one-parameter: merge, do not nest (mode-14 semantics)
            for (auto it = parsed.begin(); it != parsed.end(); ++it)
                tc.arguments[it.key()] = it.value();
        } else if (as_json) tc.arguments[key] = parsed;
        else tc.arguments[key] = val;
        if (unterminated) break;
        p = ve + PC.size();
        // The call closed: `</function>` right after this closer. Anything
        // after that is not this call's parameter -- observed 2026-08-25 as
        // `</function>\n<parameter=TaskUpdate>\n</function>`, an aborted
        // second call, which the scan below used to read as a parameter
        // named TaskUpdate and hand to the client as an argument. A value
        // that merely CONTAINS `</function>` is unaffected: the check is on
        // what follows a closer, not on the value.
        {
            size_t q = p;
            while (q < seg.size() && isspace((unsigned char)seg[q])) q++;
            if (seg.compare(q, 11, "</function>") == 0) break;
        }
    }
    tc.ok = true;   // zero-parameter calls are legal
    return true;
}

// ---- Drift-corpus capture (Q27_DRIFT_CORPUS=<file>) ------------------------
// One redacted JSONL record per dialect-bearing turn, tagged with what the
// chain did: recovered:<modes>, strict, unrescued, suppressed. The redaction
// and record format live in src/drift_capture.h; this is the plumbing that
// knows WHERE a turn is. Everything is behind the env var -- with it unset the
// parse path is byte-identical to before.
//
// depth: parse_bare_tool_calls re-enters itself (the mode-10 probe, the
// dialect retry) and parse_tool_call runs on segments inside it. Only the
// outermost call is a turn, so the bare chain captures at depth 1 and the
// strict parser only at depth 0.
inline thread_local int drift_parse_depth = 0;
inline thread_local DriftContext drift_ctx;
inline thread_local const json* drift_tools = nullptr;
// The streaming handlers hand a wrapped body that failed the strict parser to
// the bare chain as the same bytes (server.cu emit_tool: parse_tool_call on
// strip_ws2(tool_buf), then classify_bare on c.raw, which IS that string).
// Remembering the miss lets that record say `wrapped` without server.cu
// having to say so. One-shot: the next top-level bare parse on the thread
// consumes it whether or not the bytes match, so it cannot go stale.
inline thread_local std::string drift_last_strict_miss;
// Which recovery produced the result: a mode number, "native", "xmlclose",
// or the final block's mode flags. Set by the impl at each success site,
// read by the parse_bare_tool_calls wrapper that writes the record.
inline thread_local std::string drift_mode_hint;
// Records written on this thread; lets a caller tell whether the chain it
// drove produced one (the reasoning holdback may never invoke the parser).
inline thread_local size_t drift_records_written = 0;

struct DriftParseScope {
    DriftParseScope() { drift_parse_depth++; }
    ~DriftParseScope() { drift_parse_depth--; }
};
struct DriftCtxScope {
    DriftContext saved;
    const json* saved_tools;
    DriftCtxScope(bool wrapped, bool in_think, const json* tools)
        : saved(drift_ctx), saved_tools(drift_tools) {
        drift_ctx.wrapped = saved.wrapped || wrapped;
        drift_ctx.in_think = saved.in_think || in_think;
        if (tools) drift_tools = tools;
    }
    ~DriftCtxScope() { drift_ctx = saved; drift_tools = saved_tools; }
};

// Wider than looks_like_intended_tool_call on purpose. That predicate gates a
// WARNING, so it demands a closer the model never opened; the corpus wants
// every turn carrying dialect residue, because a dropped-opener turn that
// nothing recovered (`Read", "file_path": ...`, no {"name", no </function>)
// is exactly the miss this corpus exists to surface. Over-capture is cheap:
// the record is redacted and dedup collapses it.
inline bool drift_bearing(const std::string& text, const json* tools) {
    static const char* const kMarks[] = {"{\"name\"", "{\"tool_call\"", "\"arguments\"",
                                         "<tool_call>", "</tool_call>", "<function=", "</function>",
                                         "<parameter=", "</parameter>", "<tool_name>"};
    for (const char* m : kMarks)
        if (text.find(m) != std::string::npos) return true;
    // the mode-10 signature: a declared tool name at the start of a line with
    // the string's closing quote and the comma still attached
    // (`Read", "file_path": ...`). Anchored to the line start, so a name
    // quoted in prose (`I will Read" the file and Bash", then`) is not one.
    DriftNames names;
    declared_tool_names(tools, names);
    for (const auto& name : names) {
        const std::string needle = name + "\"";
        for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) {
            if (at != 0 && text[at - 1] != '\n') continue;
            size_t p = at + needle.size();
            while (p < text.size() && text[p] == ' ') p++;
            if (p < text.size() && text[p] == ',') return true;
        }
    }
    return false;
}

// A tool name survives redaction only when declared. The strict parser is
// called from the streaming handlers without the request's tool list; there
// the process-wide registry (fed by the request parsers above) stands in: a
// name is kept only if some request this process served declared it. Before
// the first such request, names are placeholders -- the safe direction.
inline void capture_drift(const std::string& text, const std::string& outcome,
                          const json* tools) {
    const char* path = drift_corpus_path();
    if (!path) return;
    if (!tools) tools = drift_tools;
    DriftNames names;
    if (tools) {
        declared_tool_names(tools, names);
        drift_register_names(names);
    } else {
        names = drift_registered_names();
    }
    write_drift_record(path, text, outcome, drift_ctx, &names);
    drift_records_written++;
}

inline ToolCall parse_tool_call_impl(const std::string& seg) {
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
    if (parse_native_xml_call(seg, tc)) return tc;   // trained XML dialect
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

// The strict parser only ever sees a wrapped body (the splitter's TOOL
// segment), from resolve_ordered_tool_segments and the four streaming
// handlers alike, so this is the one place a `strict` outcome is recorded.
// A miss is remembered so the bare-chain record for the same bytes can say
// `wrapped`. No declared-tool set reaches here from the streaming handlers;
// capture_drift falls back to the thread's context (set by the non-stream
// resolver) and otherwise keeps identifier-like tool names verbatim.
inline ToolCall parse_tool_call(const std::string& seg) {
    ToolCall tc = parse_tool_call_impl(seg);
    if (drift_parse_depth == 0 && drift_corpus_path()) {
        if (tc.ok) {
            DriftCtxScope wrapped(true, false, nullptr);
            capture_drift(seg, "strict", nullptr);
            drift_last_strict_miss.clear();
        } else {
            drift_last_strict_miss = seg;
        }
    }
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
    // Args-only object (issue #38 round 6): the model emits the ARGUMENTS
    // object bare -- {"command": ...} -- with no name key at all. Any quoted
    // identifier first key is plausible; classification decides whether a
    // call comes out (an ordinary JSON object is re-emitted untouched), so
    // the cost of holding is latency, not bytes.
    if(text[p]=='"') {
        size_t q=p+1;
        if(q==end) return true;
        if(!(isalpha((unsigned char)text[q]) || text[q]=='_')) return false;
        while(q<end && (isalnum((unsigned char)text[q]) || text[q]=='_' ||
                        text[q]=='-')) q++;
        if(q==end) return true;
        if(text[q]!='"') return false;
        q++;
        while(q<end && (text[q]==' '||text[q]=='\t')) q++;
        return q==end || text[q]==':';
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

// A bare native-dialect opener in streamed TEXT. The splitter routes a
// <tool_call>-wrapped call to TOOL, so any <function= that reaches the text
// holdback has no wrapper: the model dropped the opener (Qwen3-Coder does
// this; llama.cpp's grammar comments say so, and its grammar cannot emit the
// shape, which is why the 2026-08-22 leftover count was 0/293 there and
// 11/498 here). Same display-context rule as bare_object_position: a fenced
// or inline-code mention is prose.
// Every spelling of a bare native opener the parser accepts. ONE table, read
// by the streaming holdback's detector (what arms a candidate), its probe
// (what partial tail to hold across a chunk boundary) and the bare chain's
// batch scanner (where a span starts), so a spelling the parser learns is a
// spelling the stream sees. Mode 18's `<name>` was accepted by the parser
// for a month while the holdback knew only `<function=`, so it worked off
// stream only; `<parameter_name>` (2026-08-25, pylint-6903) went to a
// client as text for the same reason and ended the session.
inline size_t bare_native_opener_position(const std::string& text,
                                          JsonStringLexState string_state={},
                                          MarkdownFenceLexState fence_state={},
                                          bool end_is_final=true) {
    for(size_t i=0;i<text.size();i++) {
        if(text[i]=='<' && bare_native_opener_len_at(text,i) &&
           bare_context_is_executable(
               text,i,string_state,fence_state,end_is_final)) return i;
        consume_bare_text_context(string_state,fence_state,text[i]);
    }
    return std::string::npos;
}
// Start of the longest suffix that can still become an opener, so a chunk
// boundary inside one does not leak its head as visible text.
inline size_t bare_native_opener_probe_start(
    const std::string& text,JsonStringLexState string_state={},
    MarkdownFenceLexState fence_state={}) {
    static const char* const k[]={"<function=","<name>","<parameter_name>"};
    size_t best=std::string::npos;
    for(const char* o:k) {
        const size_t n=std::char_traits<char>::length(o);
        const size_t cap=std::min(text.size(),n-1);
        for(size_t len=cap;len;--len) {
            const size_t start=text.size()-len;
            if(text.compare(start,len,o,len)!=0) continue;
            if(best==std::string::npos || start<best) best=start;
            break;
        }
    }
    if(best==std::string::npos) return best;
    if(!bare_text_position_is_executable(text,best,string_state,fence_state,false))
        return std::string::npos;
    return best;
}

// A mode-22 opener in streamed bare TEXT: `<parameter=NAME>` where NAME is a
// declared tool, followed by another `<parameter=` (a real argument) or by
// `</function>` (a zero-argument call). The parameter-as-opener shape reaches
// the text holdback the same way `<function=` does -- the splitter only routes
// a `<tool_call>`-wrapped call to TOOL -- but the holdback never armed on it,
// so a bare mode-22 batch streamed out as text and never fired (issue #38,
// cosmicnag, 2026-08-26). The no-required zero-arg gate is applied later by the
// parser; arming is enough. `<parameter=key>` whose key is not a declared tool
// (an ordinary parameter) does not arm.
template<class NameSet>
inline size_t bare_mode22_opener_position(const std::string& text,const NameSet& names,
                                          JsonStringLexState string_state={},
                                          MarkdownFenceLexState fence_state={},
                                          bool end_is_final=true) {
    for(size_t i=0;i<text.size();i++) {
        if(text[i]=='<' && text.compare(i,10,"<parameter")==0 &&
           text.compare(i,11,"<parameter_")!=0) {
            std::string key; size_t after=0;
            if(parse_parameter_opener(text,i,key,after)) {
                if(names.find(key)!=names.end()) {
                    // parameter-as-opener (mode 22): NAME is a declared tool.
                    const size_t nb=text.find_first_not_of(" \t\r\n",after);
                    const bool param_next=nb!=std::string::npos && text.compare(nb,10,"<parameter")==0;
                    const bool zero_arg=nb!=std::string::npos && text.compare(nb,11,"</function>")==0;
                    if((param_next||zero_arg||(end_is_final&&nb==std::string::npos)) &&
                       bare_context_is_executable(text,i,string_state,fence_state,end_is_final))
                        return i;
                } else if(text.find("</parameter>",after)!=std::string::npos) {
                    // openerless parameter list (mode 21 class, issue #38
                    // 2026-08-29): an ordinary parameter key -- any spelling
                    // parse_parameter_opener accepts, including the HTML-attr
                    // form `<parameter name="command">` -- arms only once the
                    // visible text carries a `</parameter>` closer, the
                    // evidence a markdown mention of the tag never has. The
                    // parser's mode-21 chain (name inference + declared gate)
                    // decides whether a call comes out; a non-call candidate
                    // is re-emitted as text by the flush path.
                    if(bare_context_is_executable(text,i,string_state,fence_state,end_is_final))
                        return i;
                }
            }
        }
        consume_bare_text_context(string_state,fence_state,text[i]);
    }
    return std::string::npos;
}
// Start of a trailing `<parameter=...` that could STILL become a mode-22
// opener once more bytes arrive, so its head is held rather than leaked. Unlike
// `<function=` (an unambiguous 11-char prefix that arms immediately), a
// `<parameter=` is ambiguous with an ordinary parameter, so it is held only
// while unresolved: a partial `<parameter` prefix, a `<parameter=NAME` whose
// tag or declared-ness is not yet decidable, or a `<parameter=NAME>` (declared)
// whose boundary token has not arrived. A closed tag with an undeclared name,
// or a declared name followed by a real value, is not an opener and is not
// held.
template<class NameSet>
inline size_t bare_mode22_opener_probe_start(
    const std::string& text,const NameSet& names,
    JsonStringLexState string_state={},MarkdownFenceLexState fence_state={}) {
    static const std::string O="<parameter";
    size_t hold=std::string::npos;
    // (1) a trailing partial prefix of "<parameter"
    for(size_t len=std::min(text.size(),O.size()-1);len;--len) {
        const size_t start=text.size()-len;
        if(text.compare(start,len,O.c_str(),len)==0) { hold=start; break; }
    }
    // (2) the last complete "<parameter..." tag whose opener-ness is undecided
    //     (any spelling parse_parameter_opener accepts; NOT <parameter_name>,
    //     which belongs to the native opener table)
    size_t p=text.rfind(O);
    if(p!=std::string::npos && text.compare(p,11,"<parameter_")==0) p=std::string::npos;
    if(p!=std::string::npos) {
        std::string key; size_t after=0;
        if(!parse_parameter_opener(text,p,key,after)) {
            hold=(hold==std::string::npos?p:std::min(hold,p));   // tag not closed
        } else if(names.find(key)!=names.end()) {
            const size_t nb=text.find_first_not_of(" \t\r\n",after);
            bool decided=false;
            if(nb!=std::string::npos && text[nb]!='<') decided=true;  // a value -> ordinary param
            else if(nb!=std::string::npos) {
                // a '<' has started: decided only once it is long enough to be
                // classified as `<parameter` / `</function>` or ruled out
                const size_t avail=text.size()-nb;
                decided = avail>=11 ||
                          (text.compare(nb,std::min(avail,(size_t)10),"<parameter",std::min(avail,(size_t)10))!=0 &&
                           text.compare(nb,std::min(avail,(size_t)11),"</function>",std::min(avail,(size_t)11))!=0);
            }
            if(!decided) hold=(hold==std::string::npos?p:std::min(hold,p));
        } else if(text.find("</parameter>",after)==std::string::npos) {
            // ordinary key: a parameter LIST is still possible until its first
            // closer arrives; hold the tag rather than leak the head of a call
            // (a prose mention that never closes is re-emitted at finish).
            hold=(hold==std::string::npos?p:std::min(hold,p));
        }
    }
    if(hold==std::string::npos) return hold;
    if(!bare_text_position_is_executable(text,hold,string_state,fence_state,false))
        return std::string::npos;
    return hold;
}
// End of a bare native-dialect candidate: the first </function> that closes
// a parameter (or an empty call), plus one stray </tool_call> right after it.
// A </function> inside a value is preceded by value bytes, not </parameter>,
// so it does not end the call -- the same boundary the parser's value-closer
// search keeps. npos while the candidate is still open, including while the
// bytes after the closer could still become </tool_call>.
struct IncrementalBareNativeEnd {
    size_t cursor=0;
    void begin() { cursor=0; }
    static bool closes_call(const std::string& text,size_t closer) {
        size_t b=closer;
        while(b>0 && (text[b-1]==' ' || text[b-1]=='\t' ||
                      text[b-1]=='\r' || text[b-1]=='\n')) b--;
        static const std::string PC="</parameter>";
        if(b>=PC.size() && text.compare(b-PC.size(),PC.size(),PC)==0)
            return true;
        const size_t gt=text.find('>');
        return gt!=std::string::npos && gt+1==b &&
               text.find("<parameter=")==std::string::npos;
    }
    size_t advance(const std::string& text,bool final) {
        static const std::string FC="</function>",TC="</tool_call>";
        for(size_t p=text.find(FC,cursor);p!=std::string::npos;
            p=text.find(FC,p+1)) {
            if(!closes_call(text,p)) continue;
            size_t e=p+FC.size();
            size_t q=e;
            while(q<text.size() && (text[q]==' ' || text[q]=='\t' ||
                                    text[q]=='\r' || text[q]=='\n')) q++;
            const size_t avail=text.size()-q;
            if(avail>=TC.size()) {
                if(text.compare(q,TC.size(),TC)==0) e=q+TC.size();
            } else if(!final && text.compare(q,avail,TC,0,avail)==0) {
                return std::string::npos;
            }
            return e;
        }
        cursor=text.size()>=FC.size()?text.size()-FC.size()+1:0;
        return std::string::npos;
    }
};
// First </function> in [from, end) that closes a call (see
// IncrementalBareNativeEnd::closes_call); npos when none does. Callers fall
// back to the first literal closer so a call missing its </parameter> still
// bounds where it always did; only a closer INSIDE a value moves.
inline size_t native_function_closer(const std::string& text,size_t from=0) {
    static const std::string FC="</function>";
    for(size_t p=text.find(FC,from);p!=std::string::npos;p=text.find(FC,p+1))
        if(IncrementalBareNativeEnd::closes_call(text,p)) return p;
    return std::string::npos;
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
    // Args-only object (round 6): a balanced-but-malformed {"ident": ...}
    // defers to final tolerant recovery, where the trailing XML closers the
    // deferral captured decide whether mode 23 fires.
    if(!text.empty() && text[0]=='{' && text.compare(0,7,"{\"name\"")!=0) {
        size_t p=1;
        while(p<text.size() && isspace((unsigned char)text[p])) p++;
        if(p<text.size() && text[p]=='"' && p+1<text.size() &&
           (isalpha((unsigned char)text[p+1]) || text[p+1]=='_')) return true;
    }
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

// ---- Dialect residue -------------------------------------------------------
// The bytes a model leaves around a call that are neither the call nor text:
// a doubled <tool_call> opener, the closer/opener between two calls it ran
// together, a stray closer ahead of the wrapper, a truncated closer at the
// end. Shown to a client they read as garbage -- issue #38, third round:
// "</function>\n<tool_call>" in the visible content ahead of a call that
// then worked.

// Length of a residue token at s[i] within [i,end): a complete closer or
// wrapper token, or a truncated closer (`</functio`) that ends where a token
// would -- at whitespace, at the next tag, or at `end`. 0 if none.
inline size_t dialect_residue_token_at(const std::string& s, size_t i, size_t end,
                                      bool params_are_residue = true) {
    // a mode-22 opener with nothing inside it up to the bound --
    // `<parameter=TaskUpdate>\n</function>`, the model starting a call and
    // abandoning it -- is residue: mode 22 refuses a zero-parameter opener,
    // so there is no call to make and no value to show. NOT `<function=`:
    // `<function=TaskList>\n</function>` is a legal zero-parameter call.
    if (params_are_residue && s.compare(i, 11, "<parameter=") == 0) {
        const size_t gt = s.find('>', i);
        if (gt != std::string::npos && gt < end) {
            size_t q = gt + 1;
            while (q < end) {
                if (isspace((unsigned char)s[q])) { q++; continue; }
                size_t n = 0;
                for (const char* tk : {"</function>", "</parameter>", "</tool_call>"}) {
                    const size_t m = strlen(tk);
                    if (q + m <= end && s.compare(q, m, tk) == 0) { n = m; break; }
                }
                if (!n) break;
                q += n;
            }
            if (q == end) return gt + 1 - i;
        }
    }
    // an EMPTY name tag -- `<name>` or `<parameter_name>` with only whitespace
    // before the next tag or the bound -- is residue: the tag that carries the
    // name follows it (pylint-6903 opened the tag twice)
    for (const char* nt : {"<name>", "<parameter_name>"}) {
        const size_t n = strlen(nt);
        if (i + n <= end && s.compare(i, n, nt) == 0) {
            size_t q = i + n;
            while (q < end && isspace((unsigned char)s[q])) q++;
            if (q == end || s[q] == '<') return n;
        }
    }
    // the `{"tool_call":` JSON-keyed opener head, when it wraps an object: the
    // xmlclose retry blanks it to make the batch parse, and it is residue in
    // front of the first recovered call, not text (drift corpus ea9ead21)
    if (s.compare(i, 13, "{\"tool_call\":") == 0) {
        // the wrapped object can begin exactly at `end` (it IS the call being
        // absorbed), so check the full string, not [i,end)
        const size_t q = s.find_first_not_of(" \t\r\n", i + 13);
        if (q != std::string::npos && q < s.size() && s[q] == '{') return 13;
    }
    for (const char* tk : {"</function>", "</parameter>", "</tool_call>", "<tool_call>",
                           "<tool_use>", "</tool_use>", "<tool>", "</tool>", "<tool_calls>", "</tool_calls>"}) {
        const size_t n = strlen(tk);
        if (i + n <= end && s.compare(i, n, tk) == 0) return n;
        size_t l = 0;
        while (i + l < end && l < n - 1 && s[i + l] == tk[l]) l++;
        if (l >= 2 && (i + l == end || isspace((unsigned char)s[i + l]) || s[i + l] == '<')) return l;
    }
    return 0;
}

// The maximal suffix of `s` made of whitespace and residue tokens. `start` is
// where it begins (s.size() if none); `complete` says a whole token is in it;
// `partial` says it ends in a truncated token, which a later chunk may finish.
struct DialectResidueSuffix { size_t start; bool complete; bool partial; };
inline DialectResidueSuffix dialect_residue_suffix(const std::string& s,
                                                  bool params_are_residue = true) {
    DialectResidueSuffix r{0, false, false};
    size_t i = 0;
    while (i < s.size()) {
        if (isspace((unsigned char)s[i])) { i++; continue; }
        const size_t n = dialect_residue_token_at(s, i, s.size(), params_are_residue);
        if (!n) { i++; r.start = i; r.complete = false; r.partial = false; continue; }
        // a truncated token only counts at the very end
        const bool whole = i + n < s.size() || s.compare(i, n, "</function>") == 0 ||
                           s.compare(i, n, "</parameter>") == 0 || s.compare(i, n, "</tool_call>") == 0 ||
                           s.compare(i, n, "<tool_call>") == 0;
        if (whole) r.complete = true; else r.partial = true;
        i += n;
    }
    if (r.start == s.size()) { r.complete = false; r.partial = false; }
    return r;
}

// Widen each recovered call's span over the residue next to it -- only when
// at least one token is absorbed, so ordinary whitespace shape stays, and
// never across a neighbouring call. Text between content and residue stays
// text; the whitespace between residue and the call goes with the residue.
inline void absorb_dialect_residue(const std::string& text, std::vector<ToolCall>& calls) {
    for (size_t k = 0; k < calls.size(); k++) {
        ToolCall& c = calls[k];
        if (c.source_begin == std::string::npos || c.source_end == std::string::npos) continue;
        const size_t ub = k + 1 < calls.size() ? calls[k + 1].source_begin : text.size();
        size_t i = c.source_end, last_tok_end = c.source_end;
        bool tok = false;
        while (i < ub) {
            if (isspace((unsigned char)text[i])) { i++; continue; }
            const size_t n = dialect_residue_token_at(text, i, ub);
            if (!n) break;
            i += n; last_tok_end = i; tok = true;
        }
        if (tok) c.source_end = i == ub ? ub : last_tok_end;
        const size_t lb = k ? calls[k - 1].source_end : 0;
        size_t j = lb, content_end = lb, first_tok = std::string::npos;
        while (j < c.source_begin) {
            if (isspace((unsigned char)text[j])) { j++; continue; }
            const size_t n = dialect_residue_token_at(text, j, c.source_begin);
            if (!n) { j++; content_end = j; first_tok = std::string::npos; continue; }
            if (first_tok == std::string::npos) first_tok = j;
            j += n;
        }
        if (first_tok != std::string::npos) c.source_begin = content_end == lb ? lb : first_tok;
        c.raw = text.substr(c.source_begin, c.source_end - c.source_begin);
    }
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
    bool native=false; // holding a wrapper-less <function=...> candidate
    // Dialect residue that sat right before the candidate (a stray
    // `</function>`, a doubled opener). Not emitted when the candidate is
    // armed: it is classified together with the candidate, where the parser
    // folds it into an accepted call's span, and it goes back in front of a
    // candidate that is released as text. Issue #38, third round.
    std::string pre_residue;
    IncrementalBareJsonEnd scan;
    IncrementalBareNativeEnd native_scan;
    JsonStringLexState string_state;
    MarkdownFenceLexState fence_state;

    template<class EmitText>
    void emit_visible(const std::string& text,EmitText&& emit_text) {
        if(text.empty()) return;
        // held residue turned out to precede ordinary text: it was the
        // model's own text after all, so it goes out in front of it
        if(!pre_residue.empty()) {
            std::string held;
            held.swap(pre_residue);
            consume_bare_text_context(string_state,fence_state,held);
            emit_text(held);
        }
        consume_bare_text_context(string_state,fence_state,text);
        emit_text(text);
    }
    // Emit a head of non-candidate text, keeping its trailing dialect
    // residue back: what follows decides whether it was garbage ahead of a
    // call (folded into the call by the parser) or text (re-attached above).
    template<class EmitText>
    void emit_head(std::string head,EmitText&& emit_text) {
        const DialectResidueSuffix r=dialect_residue_suffix(head);
        std::string tail;
        if(r.complete || r.partial) { tail=head.substr(r.start); head.erase(r.start); }
        emit_visible(head,emit_text);
        pre_residue+=tail;
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
        std::string source=pre_residue;
        source+=pending;
        pre_residue.clear();
        const BareToolCandidateResult result=
            emit_candidate(source,allow_repair,visible);
        if(!result.parsed) {
            if(defer_failure) {
                deferred=std::move(source);
                deferred_mode10=candidate_mode10;
            } else visible(source);
        }
        if(!candidate_mode10 && result.accepted) ordinary_call_seen=true;
        pending.clear();
        holding=false;
        mode10=false;
        native=false;
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
                const size_t native_pos=bare_native_opener_position(
                    remaining,string_state,fence_state,input_final);
                const size_t mode22_pos=bare_mode22_opener_position(
                    remaining,names,string_state,fence_state,input_final);
                size_t opener=object_pos;
                if(mode10_pos<opener) opener=mode10_pos;
                if(native_pos<opener) opener=native_pos;
                if(mode22_pos<opener) opener=mode22_pos;
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
                        const size_t native_keep=bare_native_opener_probe_start(
                            remaining,string_state,fence_state);
                        if(native_keep!=std::string::npos)
                            keep=keep==std::string::npos?native_keep:
                                 std::min(keep,native_keep);
                        const size_t m22_keep=bare_mode22_opener_probe_start(
                            remaining,names,string_state,fence_state);
                        if(m22_keep!=std::string::npos)
                            keep=keep==std::string::npos?m22_keep:
                                 std::min(keep,m22_keep);
                    }
                    if(keep==std::string::npos) emit_head(remaining,emit_text);
                    else {
                        emit_head(remaining.substr(0,keep),emit_text);
                        probe=remaining.substr(keep);
                    }
                    return;
                }
                if(opener) emit_head(remaining.substr(0,opener),emit_text);
                pending=remaining.substr(opener);
                holding=true;
                mode10=mode10_pos==opener;
                native=!mode10 && object_pos!=opener &&
                       (native_pos==opener || mode22_pos==opener);
                if(native) native_scan.begin(); else scan.begin(mode10);
                remaining.clear();
            }
            if(!mode10 && !native && !plausible_bare_tool_prefix(pending)) {
                std::string retry=std::move(pending);
                pending.clear();
                holding=false;
                emit_visible(retry.substr(0,1),emit_text);
                remaining=retry.substr(1);
                continue;
            }
            const size_t end=native?native_scan.advance(pending,input_final)
                                   :scan.advance(pending);
            if(end==std::string::npos) return;
            std::string trailing=pending.substr(end);
            pending.resize(end);
            if(native) {
                // the scanner swallowed a stray </tool_call> so it would not
                // surface as prose; the candidate itself has no wrapper
                static const std::string TC="</tool_call>";
                size_t b=pending.size();
                while(b>0 && (pending[b-1]==' ' || pending[b-1]=='\t' ||
                              pending[b-1]=='\r' || pending[b-1]=='\n')) b--;
                if(b>=TC.size() && pending.compare(b-TC.size(),TC.size(),TC)==0)
                    pending.resize(b-TC.size());
            }
            const bool defer_failure=!input_final && !native &&
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
            // Round 6: XML closers that followed a deferred args-only object
            // are its intent evidence -- move them into the candidate so the
            // repair-time classify sees them (a bare object with no closers
            // stays an ordinary object and is re-emitted).
            size_t adv=0;
            for(;;) {
                size_t w=adv;
                while(w<trailing.size() && isspace((unsigned char)trailing[w])) w++;
                static const char* const closers[]={"</parameter>","</function>","</tool_call>"};
                bool moved=false;
                for(const char* c:closers) {
                    const size_t n=std::char_traits<char>::length(c);
                    if(trailing.compare(w,n,c)==0) { adv=w+n; moved=true; break; }
                }
                if(!moved) break;
            }
            if(adv>0) { pending+=trailing.substr(0,adv); trailing.erase(0,adv); }
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

    // The holdback has a call candidate in flight: armed (holding), speculatively
    // held in `probe`, mid-parse in `pending`, or a deferred-failure repair.
    // Used by the router so it will not strip a trailing `</function>` (a
    // candidate's own closer) into residue while a candidate is open.
    bool pending_candidate() const {
        return holding || !probe.empty() || !pending.empty() || !deferred.empty();
    }

    void reset_context() {
        string_state.reset();
        fence_state.reset();
        ordinary_call_seen=false;
        pre_residue.clear();
    }
};

// ---- Streaming tool routing shared by the SSE handlers ---------------------
// Issue #38, reopened (cosmicnag, 2026-08-25). The /v1/messages and
// /v1/chat/completions SSE handlers each carried a hand-copied twin of the
// per-chunk routing below. 152d3a2 gave one twin a reasoning holdback and
// missed the other, so on /v1/chat/completions a call emitted inside <think>
// still streamed out as reasoning_content and never fired; and even the fixed
// twin never blanked the <tool_call> wrapper before classifying reasoning, so
// only the bare shape recovered there. Twins drift. This is the one copy,
// tested through the real StreamSplitter (tools/test_openai_bridge.cpp).

// Inside reasoning a literal <tool_call> wrapper survives the splitter (it
// scans THINK for </think> only) and the display lexer treats it as an inert
// container, so the holdback never arms on the call inside it. The non-stream
// THINK branch blanks the two wrapper tokens (length-preserving) before
// classification; this does the same for a stream, where a token can straddle
// two chunks: the longest tail that is a proper prefix of either token is held
// until the next chunk decides what it was.
struct ThinkWrapperBlanker {
    std::string held;
    static const char* const* tokens() {
        static const char* const k[2] = {"<tool_call>", "</tool_call>"};
        return k;
    }
    std::string feed(const std::string& t) {
        held += t;
        for (int i = 0; i < 2; i++) {
            const char* tk = tokens()[i];
            const size_t n = std::char_traits<char>::length(tk);
            for (size_t at = held.find(tk); at != std::string::npos; at = held.find(tk, at + n))
                held.replace(at, n, std::string(n, ' '));
        }
        size_t keep = 0;
        for (int i = 0; i < 2; i++) {
            const char* tk = tokens()[i];
            const size_t n = std::char_traits<char>::length(tk);
            for (size_t k = std::min(n - 1, held.size()); k > keep; k--)
                if (held.compare(held.size() - k, k, tk, k) == 0) { keep = k; break; }
        }
        std::string out = held.substr(0, held.size() - keep);
        held.erase(0, held.size() - keep);
        return out;
    }
    std::string flush() { std::string out; out.swap(held); return out; }
};

// The per-request routing state a streaming handler keeps between splitter
// segments: the TEXT holdback, the reasoning holdback (never shared with TEXT:
// the display-context lexer state is per-channel, and a fence opened in prose
// must not silence a call in reasoning), the wrapper blanker for reasoning,
// and the wrapped segment being collected. The handler supplies what differs
// between endpoints -- how a text delta, a reasoning delta, a complete
// wrapped segment and a bare-chain classification are emitted.
struct StreamToolRouter {
    BareToolTextHoldback bare_text;
    BareToolTextHoldback think_text;
    ThinkWrapperBlanker think_blank;
    // A TEXT chunk's trailing dialect residue (a stray `</function>`, a
    // truncated closer), held until the next segment says what it was:
    // garbage ahead of a wrapper (dropped), or the model's own text (shown).
    std::string text_residue;
    std::string tool_buf;
    bool has_tools = false;
    // Drift corpus: the holdback decides whether the parser runs at all, so a
    // turn whose text carries dialect markup in a shape no opener matches
    // (`<tool_use>\n<tool>\n<parameter_name>Read...`, seen live 2026-08-25)
    // would never be recorded. Keep the turn's text, bounded, and record it
    // at finish when nothing else did. Only with Q27_DRIFT_CORPUS set.
    std::string text_seen;
    size_t records_at_start = drift_records_written;
    static constexpr size_t kTextSeenCap = 64 * 1024;

    template<class Names, class EmitText, class Classify>
    void release_text_residue(const Names& names, EmitText&& emit_text, Classify&& classify) {
        if (text_residue.empty()) return;
        std::string held;
        held.swap(text_residue);
        bare_text.route(held, names, emit_text, classify);
    }

    template<class Names, class EmitThink, class Classify>
    void settle_think(const Names& names, EmitThink&& emit_think, Classify&& classify,
                      bool allow_repair = false) {
        const std::string tail = think_blank.flush();
        if (!tail.empty()) think_text.route(tail, names, emit_think, classify);
        think_text.finish(allow_repair, names, emit_think, classify);
        think_text.reset_context();
    }
    template<class Names, class EmitText, class Classify>
    void settle_text(const Names& names, EmitText&& emit_text, Classify&& classify,
                     bool allow_repair = false) {
        bare_text.finish(allow_repair, names, emit_text, classify);
        bare_text.reset_context();
    }

    // One splitter segment. emit_tool consumes tool_buf (a complete wrapped
    // segment); classify is the bare-chain callback both holdbacks use.
    template<class Names, class EmitText, class EmitThink, class EmitTool, class Classify>
    void segment(StreamSplitter::Chan ch, const std::string& t, bool forced_control_token,
                 const Names& names, EmitText&& emit_text, EmitThink&& emit_think,
                 EmitTool&& emit_tool, Classify&& classify) {
        if (ch == StreamSplitter::TOOL) {
            if (tool_buf.empty()) {
                text_residue.clear();   // residue ahead of a wrapper is garbage
                settle_text(names, emit_text, classify);
                settle_think(names, emit_think, classify);
            }
            tool_buf += t;
            return;
        }
        if (!tool_buf.empty()) emit_tool();
        if (t.empty()) return;
        if (ch == StreamSplitter::THINK) {
            release_text_residue(names, emit_text, classify);
            settle_text(names, emit_text, classify);
            if (has_tools) think_text.route(think_blank.feed(t), names, emit_think, classify);
            else emit_think(t);
            return;
        }
        // leaving reasoning: settle any held candidate as thinking before
        // the channel changes
        settle_think(names, emit_think, classify);
        // Only decoder-injected close whitespace is parser control. Ordinary
        // leading whitespace, including after a natural close, is content.
        if (forced_control_token && strip_ws2(t).empty()) return;
        if (!has_tools) { emit_text(t); return; }
        if (drift_corpus_path() && text_seen.size() < kTextSeenCap) text_seen += t;
        std::string chunk;
        chunk.swap(text_residue);
        chunk += t;
        // Hold a chunk's trailing dialect residue back only when NO candidate
        // is open: once the holdback is holding one, a trailing `</function>`
        // is that call's own closer, not a stray to drop (issue #38 bare
        // mode-22). And a `<parameter=` opener is never held (the holdback's
        // probe holds a partial one), so a bare mode-22 opener reaches route().
        if (!bare_text.pending_candidate()) {
            const DialectResidueSuffix r =
                dialect_residue_suffix(chunk, /*params_are_residue=*/false);
            if (r.complete || r.partial) {
                text_residue = chunk.substr(r.start);
                chunk.erase(r.start);
            }
        }
        if (!chunk.empty()) bare_text.route(chunk, names, emit_text, classify);
    }

    // Generation end, after the splitter flush and the wrapped-tail handling.
    // A turn that ended still inside <think> can hold a complete call; without
    // this it is lost exactly as issue #38 reported.
    template<class Names, class EmitText, class EmitThink, class Classify>
    void finish(bool allow_repair, const Names& names, EmitText&& emit_text,
                EmitThink&& emit_think, Classify&& classify) {
        settle_think(names, emit_think, classify, allow_repair);
        // a complete closer with nothing after it is residue; a truncated one
        // could have been the model's own text, and is shown
        // ...but a trailing `</function>` while the holdback is still holding
        // is the candidate's own closer (bare mode-22, issue #38); keep it so
        // release_text_residue feeds it to the candidate.
        if (!bare_text.pending_candidate() &&
            dialect_residue_suffix(text_residue, /*params_are_residue=*/false).complete)
            text_residue.clear();
        release_text_residue(names, emit_text, classify);
        settle_text(names, emit_text, classify, allow_repair);
        if (drift_corpus_path() && drift_records_written == records_at_start &&
            drift_bearing(text_seen, nullptr))
            capture_drift(text_seen, "unrescued", nullptr);
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

// DRIFT MODE 23 (2026-09-01, issue #38 round 6, cosmicnag): the model emits
// the ARGUMENTS object bare -- {"command": "..."} with no name, no wrapper,
// no opener -- then closes with XML dialect closers (</parameter>
// </function>). The inverse chimera of mode 17, with mode-11-class escaping
// damage inside the value (mixed \" and raw ", literal newlines). The XML
// closers are the intent evidence: a bare JSON object in prose does NOT
// fire (nothing but whitespace/closers may follow the object). The value
// repair is the mode-11 terminator scan, tried per declared tool (its
// string params are the scan keys); the call fires only when exactly ONE
// tool yields a parse whose keys fit it.
inline bool recover_args_object_call(const std::string& text, const json& tools,
                                     std::vector<ToolCall>& out) {
    // The object start: the LAST bare '{' opening a quoted-identifier key.
    size_t mo=std::string::npos;
    for(size_t i=text.rfind('{');i!=std::string::npos;
        i=i?text.rfind('{',i-1):std::string::npos) {
        size_t p=i+1;
        while(p<text.size() && isspace((unsigned char)text[p])) p++;
        if(p<text.size() && text[p]=='"' && p+1<text.size() &&
           (isalpha((unsigned char)text[p+1])||text[p+1]=='_') &&
           text.compare(i,7,"{\"name\"")!=0 &&
           bare_text_position_is_executable(text,i)) { mo=i; break; }
        if(!i) break;
    }
    if(mo==std::string::npos) return false;
    // Closer evidence: the text after the object's last "} must be only
    // whitespace and XML closers, with at least one </function> or
    // </parameter> present.
    const size_t vend=text.rfind("\"}");
    if(vend==std::string::npos || vend<mo) return false;
    size_t q=vend+2; bool saw_closer=false;
    while(q<text.size()) {
        if(isspace((unsigned char)text[q])) { q++; continue; }
        bool moved=false;
        static const char* const closers[]={"</parameter>","</function>","</tool_call>"};
        for(const char* c:closers) {
            const size_t n=std::char_traits<char>::length(c);
            if(text.compare(q,n,c)==0) { q+=n; moved=true; saw_closer=true; break; }
        }
        if(!moved) return false;  // real content after the object: not a call
    }
    if(!saw_closer) return false;
    const std::string obj=text.substr(mo,vend+2-mo);
    // Per-tool trial of the mode-11 terminator-scan repair.
    std::string hit_name; json hit_args; int hits=0;
    for(const auto& t:tools) {
        if(!t.contains("function")) continue;
        const json& fn=t["function"];
        const std::string nm=fn.value("name",std::string());
        if(nm.empty()) continue;
        std::vector<std::string> strkeys; json props;
        if(fn.contains("parameters") && fn["parameters"].is_object() &&
           fn["parameters"].contains("properties") &&
           fn["parameters"]["properties"].is_object()) {
            props=fn["parameters"]["properties"];
            for(auto it=props.begin();it!=props.end();++it)
                if(it.value().is_object() &&
                   it.value().value("type",std::string())=="string")
                    strkeys.push_back(it.key());
        }
        json parsed;
        bool ok=false;
        try { parsed=json::parse(obj); ok=parsed.is_object(); } catch(...) {}
        if(!ok) {
            for(const auto& k:strkeys) {
                size_t kp=obj.find("\""+k+"\"");
                if(kp==std::string::npos) continue;
                size_t kc=obj.find(':',kp+k.size()+2);
                if(kc==std::string::npos) continue;
                size_t opener=obj.find('"',kc+1);
                if(opener==std::string::npos) continue;
                for(size_t cand=obj.find('"',opener+1);cand!=std::string::npos;
                    cand=obj.find('"',cand+1)) {
                    const std::string body=minimal_escape_body(
                        obj.substr(opener+1,cand-opener-1));
                    const std::string recon=first_balanced_object(
                        obj.substr(0,opener)+"\""+body+"\""+obj.substr(cand+1));
                    if(recon.empty()) continue;
                    try {
                        json o2=json::parse(recon);
                        if(o2.is_object()) { parsed=std::move(o2); ok=true; }
                    } catch(...) { continue; }
                    if(ok) break;
                }
                if(ok) break;
            }
        }
        if(!ok || !parsed.is_object() || parsed.empty()) continue;
        // Fit: every key known to this tool, every required key present.
        bool fits=true;
        for(auto it=parsed.begin();it!=parsed.end();++it)
            if(!props.is_object() || !props.contains(it.key())) { fits=false; break; }
        if(fits && fn.contains("parameters") &&
           fn["parameters"].contains("required") &&
           fn["parameters"]["required"].is_array())
            for(const auto& r:fn["parameters"]["required"])
                if(r.is_string() && !parsed.contains(r.get<std::string>())) { fits=false; break; }
        if(!fits) continue;
        hits++;
        if(hits>1) return false;  // ambiguous: refuse rather than guess
        hit_name=nm; hit_args=std::move(parsed);
    }
    if(hits!=1) return false;
    ToolCall tc;
    tc.ok=true;
    tc.name=hit_name;
    tc.arguments=std::move(hit_args);
    // Offsets are in the rewritten buffer; the span validator maps them back
    // to source and fills source_begin/source_end/raw (same as mode 11).
    tc.rewritten_begin=mo;
    tc.rewritten_end=q;
    fprintf(stderr,"[drift] mode 23: args-only object + XML closers -> %s\n",
            hit_name.c_str());
    out.push_back(std::move(tc));
    return true;
}

inline bool only_dialect_control_bytes(const std::string& s) {
    size_t i=0;
    bool saw_marker=false;
    while(i<s.size()) {
        if(isspace((unsigned char)s[i])) { i++; continue; }
        if(s.compare(i,11,"<tool_call>")==0) { i+=11; saw_marker=true; continue; }
        if(s.compare(i,12,"</tool_call>")==0) { i+=12; saw_marker=true; continue; }
        // A TRUNCATED marker at the very end is residue too: the model emitted
        // `<tool_call` and stopped, so there is no name, no arguments, nothing
        // to execute -- but it became the user's whole visible answer, which is
        // the failure #32 fixed for the complete marker. Only as the FINAL
        // token, and only long enough to identify (a bare "<t" could begin
        // anything), so a marker followed by prose is still shown.
        {
            const std::string rest = s.substr(i);
            for (const char* m : {"<tool_call>", "</tool_call>"}) {
                const size_t n = strlen(m);
                if (rest.size() >= 4 && rest.size() < n &&
                    strncmp(m, rest.c_str(), rest.size()) == 0)
                    return true;
            }
        }
        return false;
    }
    return saw_marker; // pure whitespace is not this function's business
}

// Would parse_tool_call() silently discard part of this wrapped body?
// parse_native_xml_call reads ONE call and stops, so a segment carrying several
// calls loses every one after the first, and trailing prose after the last
// </function> disappears with them. Measured 2026-08-21: a 14-call turn came
// back as 1 call. Where this returns true the segment goes through the same
// batch-aware chain that TEXT uses instead.
inline bool wrapped_body_exceeds_one_call(const std::string& body) {
    size_t openers = 0;
    for (size_t c = body.find("<function"); c != std::string::npos;
         c = body.find("<function", c + 9)) {
        std::string nm;
        size_t after;
        if (parse_function_opener(body, c, nm, after) && ++openers > 1) return true;
    }
    if (openers == 0) return false;
    // A mode-22 opener after a closed call -- `</function>\n<parameter=Bash>\n
    // <parameter=command>...` -- is a second call the strict parser would
    // silently drop (it reads one call and stops). An opener with nothing
    // inside (`<parameter=TaskUpdate>\n</function>`) is an aborted call and
    // stays here, where the strict parser now stops at `</function>` and the
    // tail is dropped as residue.
    for (size_t fc = body.find("</function>"); fc != std::string::npos; fc = body.find("</function>", fc + 11)) {
        size_t q = body.find_first_not_of(" \t\r\n", fc + 11);
        if (q == std::string::npos) break;
        std::string k;
        size_t after;
        if (!parse_parameter_opener(body, q, k, after)) continue;
        size_t r = body.find_first_not_of(" \t\r\n", after);
        std::string k2;
        size_t after2;
        if (r != std::string::npos && parse_parameter_opener(body, r, k2, after2)) return true;
    }
    const size_t last = body.rfind("</function>");
    if (last == std::string::npos) return false;
    return body.find_first_not_of(" \t\r\n", last + 11) != std::string::npos;
}

// First position at or after `from` where either spelling of a parameter tag
// opens. The drift modes below scan for parameters with their own literals, and
// a mode that only knows `<parameter=` cannot see `<parameter name="KEY">` --
// which would leave the attribute form working only for calls that ALSO carry a
// well-formed <function...> opener, i.e. the case least in need of rescue.
inline size_t find_parameter_opener(const std::string& s, size_t from,
                                    std::string* key = nullptr, size_t* after = nullptr) {
    for (size_t c = s.find("<parameter", from); c != std::string::npos;
         c = s.find("<parameter", c + 10)) {
        std::string k;
        size_t a;
        if (parse_parameter_opener(s, c, k, a)) {
            if (key) *key = k;
            if (after) *after = a;
            return c;
        }
    }
    return std::string::npos;
}

// Blank the XML dialect's markers where they sit OUTSIDE a JSON string.
//
// The model sometimes terminates a JSON call body with `</parameter></function>`
// instead of its final `}` -- the arguments are complete, the terminator is
// simply in the other dialect. The EOF brace repair only fires when the
// candidate is last in the text (right: bytes after a truncated call mean the
// model moved on, and repairing then invents content), so that residue blocked
// a call that had lost nothing.
//
// Outside a JSON string these tokens can never be valid JSON, so replacing them
// with spaces cannot change any parse that already succeeded. Spaces rather than
// deletion, so every source offset is preserved and the call spans stay valid --
// a shifted span is how a mode-20 batch used to throw std::out_of_range out of
// the request handler. Inside a string they are content and are left alone.
inline std::string blank_dialect_closers_outside_strings(const std::string& s) {
    static const char* const kClosers[] = {"</parameter>", "</function>", "</tool_call>"};
    std::string out = s;
    // Only bytes WE blanked may be overwritten by the brace repair below.
    // Writing a '}' over any convenient space corrupts real content -- it
    // turned `"content": "hello"` into `"content":}"hello"` the first time
    // this was written.
    std::vector<bool> blanked(s.size(), false);
    size_t blanked_end = std::string::npos;   // end of the last blanked marker
    bool in_str = false, esc = false;
    for (size_t i = 0; i < out.size(); i++) {
        const char c = out[i];
        if (esc) { esc = false; continue; }
        if (in_str) {
            if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        // A `{"tool_call":` JSON-keyed opener (drift mode 4) whose value is an
        // OBJECT, so `{"tool_call":\n{"name":...}}\n</tool_call>\n{"name":...}`
        // -- the model wraps the first call of a batch in the opener and never
        // closes its brace, which leaves the whole batch one malformed blob and
        // the main scan recovered only the last call (drift corpus ea9ead21).
        // Blanking the head (its brace included) makes the inner objects
        // top-level, and the JSON batch scan reads them all. Only here, in the
        // last-resort retry, so a well-formed single mode-4 call -- handled
        // upstream -- never reaches it.
        if (c == '{' && out.compare(i, 13, "{\"tool_call\":") == 0) {
            const size_t q = out.find_first_not_of(" \t\r\n", i + 13);
            if (q != std::string::npos && out[q] == '{') {
                for (size_t k = 0; k < 13; k++) { out[i + k] = ' '; blanked[i + k] = true; }
                i += 12;
                blanked_end = i + 1;
                continue;
            }
        }
        if (c != '<') continue;
        bool hit = false;
        for (const char* t : kClosers) {
            const size_t n = strlen(t);
            if (out.compare(i, n, t) == 0) {
                for (size_t k = 0; k < n; k++) { out[i + k] = ' '; blanked[i + k] = true; }
                i += n - 1;
                blanked_end = i + 1;
                hit = true;
                break;
            }
        }
        if (hit) continue;
        // The OPENER, but ONLY when it directly follows closer residue. A
        // batch reads `...}\n</parameter>\n</function>\n<tool_call>\n{...`, and
        // leaving the opener stopped the scan at the first call. Blanking it
        // unconditionally is NOT safe: an invalid call followed by a valid one
        // with a bare `<tool_call>` between them has an ambiguous boundary, and
        // test_chat_completions_integration refuses that case ON PURPOSE --
        // merging them would execute a call the model never framed. Requiring
        // closer residue immediately before keeps the two apart.
        if (out.compare(i, 11, "<tool_call>") == 0 && blanked_end != std::string::npos &&
            out.find_first_not_of(" \t\r\n", blanked_end) == i) {
            for (size_t k = 0; k < 11; k++) { out[i + k] = ' '; blanked[i + k] = true; }
            i += 10;
            blanked_end = i + 1;
        }
    }
    // Second pass: put the missing braces back, into the spaces the first pass
    // just made. Only the LAST unbalanced candidate is eligible for the EOF
    // brace repair, so a BATCH of these nested instead of separating -- object
    // two opened inside object one and the scanner saw a single malformed blob.
    // Writing `}` over a preceding space closes each object where the model
    // should have, and keeps the length identical so the spans stay valid.
    int depth = 0;
    in_str = false;
    esc = false;
    for (size_t i = 0; i < out.size(); i++) {
        const char c = out[i];
        if (esc) { esc = false; continue; }
        if (in_str) {
            if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '{') {
            // a new top-level call starting while the previous one is still
            // open is the batch case: close the previous one first
            if (depth > 0 && out.compare(i, 7, "{\"name\"") == 0) {
                // Walk back over whitespace, but only WRITE into bytes we
                // blanked: the newlines between the markers are the model's
                // own and must stay, while the blanked marker bytes are free.
                size_t sp = i;
                while (sp > 0 && depth > 0 &&
                       (blanked[sp - 1] || out[sp - 1] == ' ' || out[sp - 1] == '\n' ||
                        out[sp - 1] == '\r' || out[sp - 1] == '\t')) {
                    if (blanked[sp - 1] && out[sp - 1] == ' ') { out[sp - 1] = '}'; depth--; }
                    sp--;
                }
            }
            depth++;
        } else if (c == '}') {
            if (depth > 0) depth--;
        }
    }
    // and the final one, into its trailing spaces
    for (size_t i = out.size(); i > 0 && depth > 0; i--) {
        if (blanked[i - 1] && out[i - 1] == ' ') { out[i - 1] = '}'; depth--; continue; }
        if (out[i - 1] == ' ' || out[i - 1] == '\n' || out[i - 1] == '\r' ||
            out[i - 1] == '\t')
            continue;                       // model whitespace: skip, do not use
        break;                              // real content: stop, never overwrite it
    }

    // Third pass: the XML tag-closer landing inside a JSON KEY.
    //     {"name": "Write", "arguments>
    //     {"file_path": ..., "content": ...}
    // The model typed `>` where JSON needs `":`, leaving the key's string
    // unterminated so nothing downstream parses. `>` becomes the closing quote
    // and the newline after it becomes the colon: two single-character
    // substitutions, so no byte moves and the spans stay valid.
    //
    // Engages only on `"IDENT>` whose next non-space byte opens a value, which
    // is why a '>' inside a legitimate string value is untouched -- there the
    // key's quote already closed and we are not at a key boundary.
    for (size_t i = 0; i + 1 < out.size(); i++) {
        if (out[i] != '"') continue;
        size_t j = i + 1;
        while (j < out.size() && (isalnum((unsigned char)out[j]) || out[j] == '_')) j++;
        if (j == i + 1 || j >= out.size() || out[j] != '>') continue;
        const size_t nb = out.find_first_not_of(" \t\r\n", j + 1);
        if (nb == std::string::npos || (out[nb] != '{' && out[nb] != '[' && out[nb] != '"'))
            continue;
        size_t colon = std::string::npos;
        for (size_t k = j + 1; k < nb; k++)
            if (out[k] == '\n' || out[k] == ' ') { colon = k; break; }
        if (colon == std::string::npos) continue;   // nowhere to put the ':'
        out[j] = '"';
        out[colon] = ':';
        i = nb;
    }
    return out;
}

// A declared tool whose name differs from `nm` only in case. The model's own
// name for a tool ("task" for Task) is not a different tool, and refusing on
// case alone loses a call for nothing. UNIQUE match only: an ambiguous registry
// still refuses, because guessing between two real tools is worse than not
// calling one.
inline std::string canonical_declared_name(const json* tools, const std::string& nm) {
    if (!tools || !tools->is_array() || nm.empty()) return "";
    auto lower = [](std::string x) {
        for (auto& ch : x) ch = (char)tolower((unsigned char)ch);
        return x;
    };
    const std::string target = lower(nm);
    std::string hit;
    int n = 0;
    for (const auto& t : *tools) {
        if (!t.contains("function")) continue;
        const std::string dn = t["function"].value("name", std::string());
        if (dn.empty()) continue;
        if (dn == nm) return dn;              // exact wins outright
        if (lower(dn) == target) { hit = dn; n++; }
    }
    return n == 1 ? hit : std::string();
}

// Did the model INTEND a tool call that nothing recovered? Used only for the
// UN-RESCUED warning and the Q27_DRIFT_CORPUS capture, so a false negative here
// costs a silent drop rather than a wrong call -- which is exactly what it cost:
// this used to test for {"name" / {"tool_call" alone, so drift mode 22, made
// entirely of XML tags, produced no log line and no corpus entry for two days.
// A report of "no rescue logs" was accurate and told us nothing.
//
// The XML forms are gated on a `</function>` the model never opened, the same
// evidence mode 21 requires. Prose that merely quotes a tag (a markdown table
// of probe commands, say) has no closer and stays unflagged.
inline bool looks_like_intended_tool_call(const std::string& text_in) {
    if (text_in.find("{\"name\"") != std::string::npos ||
        text_in.find("{\"tool_call\"") != std::string::npos)
        return true;
    if (text_in.find("</function>") == std::string::npos) return false;
    return text_in.find("<parameter=") != std::string::npos ||
           text_in.find("<function") != std::string::npos ||
           text_in.find("<tool_name>") != std::string::npos;
}

// HALLUCINATED-RESULT RULE (2026-08-21, unified).
//
// Qwen3.8 does not only degrade its tool-call wrapper -- it also invents tool
// RESULTS wearing the very same tags:
//
//   <tool_calls>\n<result>\n<name>Read</name>\n<output>\nfile contents\n</output>
//   \n<tool_name>\n<parameter=file_path>\n/w/x\n</parameter>\n</function>
//
// Promoting that to a call feeds the model's own fiction back as though a tool
// had produced it, so every dialect recovery has to refuse it. The original
// guard was a WHOLE-STRING `find("<result>")` over the entire segment, which
// is both too weak and too strong:
//
//   too strong -- a legitimate call preceded by an unrelated, already-CLOSED
//   `<output>make: up to date</output>` (a quoted result from a prior step,
//   ordinary prose in an agent transcript) was thrown away wholesale. That is
//   the production leak behind "sometimes I still see <parameter= / </function>
//   in the output": the real call was sitting right there and the guard
//   vetoed the whole recovery chain.
//
//   too weak -- it says nothing about WHERE the tags sit, so it cannot
//   distinguish the two cases at all; it just refuses both.
//
// The discriminator is STRUCTURAL, not lexical: does the result element
// *contain* the candidate call?
//
//   (a) a <result>/<output> tag INSIDE the span  -> invented output is being
//       passed off as a parameter value; refuse.
//   (b) the span begins INSIDE an unclosed <result>/<output> element -> the
//       "call" is part of the invented result block; refuse.
//   otherwise -> every result block is closed before the call starts, so it is
//       prior context, not a wrapper around this call; ACCEPT.
//
// (b) is what catches the real hallucinated shapes above: the model opens
// <result> and never closes it, so the <tool_name>/<parameter= span it later
// emits is enclosed by it. (a) catches the inverse -- a well-formed-looking
// call whose parameter VALUE is a fabricated <result>.
//
// Nesting is handled by depth counting rather than a last-open scan, so a
// closed block followed by an open one still reads as enclosed.
inline bool hallucinated_result_around(const std::string& s, size_t begin, size_t end) {
    if (begin == std::string::npos) return false;
    if (end == std::string::npos || end > s.size()) end = s.size();
    static const char* const kTags[2][2] = {{"<result>", "</result>"},
                                            {"<output>", "</output>"}};
    for (const auto& tg : kTags) {
        const std::string open = tg[0], close = tg[1];
        // (a) either tag of the pair occurring within [begin, end)
        for (const std::string& t : {open, close}) {
            size_t p = s.find(t, begin);
            if (p != std::string::npos && p < end) return true;
        }
        // (b) unbalanced open before begin == the span is enclosed by it
        long depth = 0;
        size_t i = 0;
        while (i < begin) {
            size_t o = s.find(open, i), c = s.find(close, i);
            if (o >= begin && c >= begin) break; // (npos included)
            if (o < c) { depth++; i = o + open.size(); }
            else { if (depth > 0) depth--; i = c + close.size(); }
        }
        if (depth > 0) return true;
    }
    return false;
}

inline thread_local bool in_dialect_retry = false;
inline std::vector<ToolCall> parse_bare_tool_calls(const std::string& text_in,
                                                   std::string* prefix,
                                                   const json* tools = nullptr,
                                                   bool allow_o10 = true,
                                                   bool allow_eof_repair = true,
                                                   std::string* remaining_text = nullptr);
inline std::vector<ToolCall> parse_bare_tool_calls_impl(const std::string& text_in,
                                                        std::string* prefix,
                                                        const json* tools,
                                                        bool allow_o10,
                                                        bool allow_eof_repair,
                                                        std::string* remaining_text) {
    std::vector<ToolCall> out;
    if (tool_strict()) {
        // strict-parser A/B: the wrapper-less recovery chain (drift modes 1-6)
        // is OFF. Log when the text plausibly contained an intended call so the
        // campaign can count suppressed rescues against the tolerant leg.
        if (prefix) *prefix = "";
        if (remaining_text) *remaining_text = text_in;
        const bool plausible = text_in.find("{\"name\"") != std::string::npos ||
                               text_in.find("{\"tool_call\"") != std::string::npos ||
                               text_in.find("</content>") != std::string::npos;
        if (plausible)
            fprintf(stderr, "[q27-strict] SUPPRESSED bare-call rescue: %.200s\n",
                    text_in.c_str());
        return out;
    }
    // DRIFT MODE 14 (2026-08-14, Qwen3.8-27B): the model abandons the
    // <tool_call>{json}</tool_call> form it was instructed to use and emits its
    // chat-template's XML dialect instead, with no wrapper at all:
    //
    //   <tool_name>Read</tool_name>
    //   <parameter=file_path>/w/x.ts</parameter>
    //   <tool_name>Bash</tool_name>
    //   <parameter=arguments>\n{"command":"...","description":"..."}}
    //
    // Note the conventions disagree WITHIN one block: a scalar parameter with a
    // proper closer, then <parameter=arguments> carrying a whole JSON object,
    // unterminated, with a trailing unbalanced brace. Both must be recovered or
    // the turn is lost -- Claude Code sees plain text and the session ends.
    //
    // Engages ONLY when <tool_name> is present AND names a declared tool, so no
    // input that lacks this dialect can reach it.
    {
        static const std::string TN_OPEN = "<tool_name>", TN_CLOSE = "</tool_name>";
        static const std::string PM_OPEN = "<parameter=", PM_CLOSE = "</parameter>";
        size_t first = text_in.find(TN_OPEN);
        if (first != std::string::npos && tools && tools->is_array()) {
            auto declared = [&](const std::string& nm) {
                for (const auto& t : *tools)
                    if (t.contains("function") &&
                        t["function"].value("name", std::string()) == nm) return true;
                return false;
            };
            auto trim = [](std::string v) {
                size_t b = v.find_first_not_of(" \t\r\n");
                if (b == std::string::npos) return std::string();
                size_t e = v.find_last_not_of(" \t\r\n");
                return v.substr(b, e - b + 1);
            };
            // Tolerate a trailing unbalanced brace on the JSON payload.
            auto parse_relaxed = [&](std::string v) -> json {
                v = trim(std::move(v));
                for (int drop = 0; drop < 4 && !v.empty(); drop++) {
                    try { json j = json::parse(v); if (j.is_object()) return j; }
                    catch (...) {}
                    if (v.back() == '}' || isspace((unsigned char)v.back()))
                        { v.pop_back(); v = trim(std::move(v)); }
                    else break;
                }
                return json();
            };
            std::vector<ToolCall> xml_calls;
            size_t p = first;
            while ((p = text_in.find(TN_OPEN, p)) != std::string::npos) {
                size_t ns = p + TN_OPEN.size();
                size_t ne = text_in.find(TN_CLOSE, ns);
                if (ne == std::string::npos) break;
                const std::string nm = trim(text_in.substr(ns, ne - ns));
                size_t next_call = text_in.find(TN_OPEN, ne);
                size_t span_end = next_call == std::string::npos ? text_in.size() : next_call;
                if (!declared(nm)) { p = ne + TN_CLOSE.size(); continue; }
                ToolCall tc;
                tc.name = nm;
                tc.arguments = json::object();
                tc.source_begin = p;
                tc.source_end = span_end;
                size_t q = ne + TN_CLOSE.size();
                while (true) {
                    size_t ps = text_in.find(PM_OPEN, q);
                    if (ps == std::string::npos || ps >= span_end) break;
                    size_t ks = ps + PM_OPEN.size();
                    size_t kend = text_in.find('>', ks);
                    if (kend == std::string::npos || kend >= span_end) break;
                    const std::string key = trim(text_in.substr(ks, kend - ks));
                    size_t vs = kend + 1;
                    size_t ve = text_in.find(PM_CLOSE, vs);
                    if (ve == std::string::npos || ve > span_end) ve = span_end;  // missing closer
                    const std::string val = text_in.substr(vs, ve - vs);
                    if (key == "arguments") {
                        json j = parse_relaxed(val);
                        if (j.is_object())
                            for (auto it = j.begin(); it != j.end(); ++it)
                                tc.arguments[it.key()] = it.value();
                    } else if (!key.empty()) {
                        tc.arguments[key] = trim(val);
                    }
                    q = ve == span_end ? span_end : ve + PM_CLOSE.size();
                    if (q >= span_end) break;
                }
                if (!tc.arguments.empty()) { tc.ok = true; xml_calls.push_back(std::move(tc)); }
                p = span_end;
            }
            if (!xml_calls.empty()) {
                if (prefix) *prefix = text_in.substr(0, first);
                if (remaining_text) *remaining_text = "";
                fprintf(stderr, "[q27] drift mode 14: recovered %zu <tool_name> call(s)\n",
                        xml_calls.size());
                drift_mode_hint = "14";
                return xml_calls;
            }
        }
    }

    // DRIFT MODE 15 (2026-08-14, Qwen3.8-27B, found by re-running the same
    // thunderdome task after mode 14 landed). One drift deeper than 14: the
    // model drops the XML parameter form but still prefixes an unclosed pseudo
    // tag, then falls back toward JSON without the opening brace or quote:
    //
    //   <name>Bash, "arguments": {"command":"...","description":"..."}}
    //
    // This is mode 10 (dropped `{"name": "` opener) wearing an XML hat: mode 10
    // does not fire because the leading `<name>` prevents the bare-identifier
    // match, and mode 14 does not fire because there is no </tool_name> and no
    // <parameter=. Same trailing unbalanced brace as 14.
    //
    // Engages ONLY when the identifier after <name> is a declared tool.
    {
        static const std::string NM_OPEN = "<name>", NM_ATTR = "<name=";
        size_t p = text_in.find(NM_OPEN);
        size_t pa = text_in.find(NM_ATTR);
        if (pa != std::string::npos && (p == std::string::npos || pa < p)) p = pa;
        if (p != std::string::npos && tools && tools->is_array()) {
            size_t ns = p + (text_in.compare(p, NM_ATTR.size(), NM_ATTR) == 0
                                 ? NM_ATTR.size() : NM_OPEN.size());
            // Variant seen in the same session: <name="Read", "arguments": {...}
            // i.e. an attribute-style spelling instead of a closed tag.
            if (ns < text_in.size() && text_in[ns] == '"') ns++;
            while (ns < text_in.size() && isspace((unsigned char)text_in[ns])) ns++;
            size_t ne = ns;
            while (ne < text_in.size() &&
                   (isalnum((unsigned char)text_in[ne]) || text_in[ne] == '_' ||
                    text_in[ne] == '-' || text_in[ne] == '.')) ne++;
            const std::string nm = text_in.substr(ns, ne - ns);
            bool declared = false;
            for (const auto& t : *tools)
                if (t.contains("function") &&
                    t["function"].value("name", std::string()) == nm) declared = true;
            if (declared && !nm.empty()) {
                size_t ab = text_in.find("\"arguments\"", ne);
                json args = json::object();
                bool have = false;
                if (ab != std::string::npos) {
                    size_t ob = text_in.find('{', ab);
                    if (ob != std::string::npos) {
                        std::string body = text_in.substr(ob);
                        for (int drop = 0; drop < 4 && !body.empty(); drop++) {
                            try {
                                json j = json::parse(body);
                                if (j.is_object()) { args = j; have = true; break; }
                            } catch (...) {}
                            while (!body.empty() && isspace((unsigned char)body.back()))
                                body.pop_back();
                            if (!body.empty() && body.back() == '}') body.pop_back();
                            else break;
                        }
                    }
                }
                if (have) {
                    ToolCall tc;
                    tc.ok = true;
                    tc.name = nm;
                    tc.arguments = std::move(args);
                    tc.source_begin = p;
                    tc.source_end = text_in.size();
                    if (prefix) *prefix = text_in.substr(0, p);
                    if (remaining_text) *remaining_text = "";
                    fprintf(stderr, "[q27] drift mode 15: recovered <name>%s bare-args call\n",
                            nm.c_str());
                    drift_mode_hint = "15";
                    return std::vector<ToolCall>{std::move(tc)};
                }
            }
        }
    }

    // DRIFT MODE 16 (2026-08-14, Qwen3.8-27B): the call JSON is CORRECT and
    // complete, but wrapped in <function> instead of <tool_call>:
    //
    //   I'll start by exploring the workspace.
    //
    //   <function>
    //   {"name": "Bash", "arguments": {"command": "ls -la /workspace", ...}}
    //
    // Nothing is malformed except the wrapper, so this is the cheapest of the
    // 3.8 dialects to recover and the safest: the payload must still parse as a
    // complete object AND name a declared tool.
    //
    // DELIBERATELY NOT RESCUED, and pinned by a negative fixture: the model also
    // emits <tool_calls><result><name>X</name><output>...</output> blocks, which
    // are HALLUCINATED TOOL RESULTS, not calls. Treating those as calls would
    // feed invented command output back as if a tool had produced it.
    {
        static const std::string FN_OPEN = "<function>";
        size_t p = text_in.find(FN_OPEN);
        if (p != std::string::npos && tools && tools->is_array() &&
            !hallucinated_result_around(text_in, p, text_in.size())) {
            size_t ob = text_in.find('{', p + FN_OPEN.size());
            if (ob != std::string::npos) {
                std::string body = text_in.substr(ob);
                size_t close = body.rfind("</function>");
                if (close != std::string::npos) body = body.substr(0, close);
                for (int drop = 0; drop < 4 && !body.empty(); drop++) {
                    json j;
                    bool okp = false;
                    try { j = json::parse(body); okp = j.is_object(); } catch (...) {}
                    if (okp && j.contains("name")) {
                        const std::string nm = j.value("name", std::string());
                        bool declared = false;
                        for (const auto& t : *tools)
                            if (t.contains("function") &&
                                t["function"].value("name", std::string()) == nm) declared = true;
                        if (declared) {
                            ToolCall tc;
                            tc.ok = true;
                            tc.name = nm;
                            tc.arguments = j.contains("arguments") && j["arguments"].is_object()
                                               ? j["arguments"] : json::object();
                            tc.source_begin = p;
                            tc.source_end = text_in.size();
                            if (prefix) *prefix = text_in.substr(0, p);
                            if (remaining_text) *remaining_text = "";
                            fprintf(stderr, "[q27] drift mode 16: recovered <function>-wrapped %s\n",
                                    nm.c_str());
                            drift_mode_hint = "16";
                            return std::vector<ToolCall>{std::move(tc)};
                        }
                        break;
                    }
                    while (!body.empty() && isspace((unsigned char)body.back())) body.pop_back();
                    if (!body.empty() && body.back() == '}') body.pop_back(); else break;
                }
            }
        }
    }

    // DRIFT MODE 17 + BARE NATIVE DIALECT (2026-08-14, Qwen3.8 THINK mode).
    // Thinking-on emission degrades the wrapper two new ways, both captured
    // verbatim from archived thunderdome transcripts:
    //   (a) bare well-formed dialect, no <tool_call> wrapper:
    //         <function=Read>\n<parameter=file_path>\n/x\n</parameter>\n</function>
    //   (b) the CHIMERA that killed bench-time-tracker: JSON head, XML params:
    //         {"name": "Write",\n<parameter=file_path>\n/x\n</parameter>\n<parameter=content>\n...
    // (a) reuses parse_native_xml_call on the spanned substring. (b) extracts
    // the name from the JSON head and hands the rest to the XML parameter
    // walk. Both engage only on a declared name.
    //
    // Hallucinated-result protection (2026-08-21): the outer whole-string
    // find for <result>/<output> that used to gate this entire block is gone;
    // each mode below now applies hallucinated_result_around() to its OWN
    // candidate span. See that helper for why enclosure -- not mere presence
    // -- is the correct test.
    if (tools && tools->is_array()) {
        auto declared = [&](const std::string& nm) {
            for (const auto& t : *tools)
                if (t.contains("function") &&
                    t["function"].value("name", std::string()) == nm) return true;
            return false;
        };
        // A declared tool whose schema lists NO required parameters -- so a
        // zero-argument call is schema-valid. Used to tell a real zero-arg
        // mode-22 call (`<parameter=memex_recall>\n</function>`, issue #38)
        // from an aborted one (`<parameter=TaskUpdate>\n</function>`, where
        // taskId is required and a zero-arg call would be refused by the
        // client): the first fires, the second stays residue.
        auto no_required_params = [&](const std::string& nm) {
            for (const auto& t : *tools) {
                if (!t.contains("function") ||
                    t["function"].value("name", std::string()) != nm) continue;
                const json params = t["function"].value("parameters", json::object());
                const json req = params.value("required", json::array());
                return !req.is_array() || req.empty();
            }
            return false;
        };
        // (a) a bare opener with no <tool_call> wrapper, in ANY of its
        // spellings. One scan for "<function" and let parse_function_opener
        // decide -- before the consolidation this was two searches that between
        // them still missed <function="NAME">.
        // BATCHED, because a model that plans emits SEVERAL calls in one turn,
        // separated by stray <tool_call> markers. Recovering only the first and
        // returning turned the other thirteen into prose that the agent then
        // read back as its own answer -- measured 2026-08-21 on
        // bench-task-queue, where a 14-call turn yielded 1 call and 3480 bytes
        // of leaked dialect, and the session ended there. Modes 14, 20 and 22
        // were already batch-capable; this path was the odd one out.
        // Set when a `<function ...>` opener names a tool that is NOT declared.
        // Inference downstream must then refuse: mode 21 exists for emissions
        // with no usable name, not to override one that is present and wrong.
        // Substituting a tool whose keys happen to fit calls something the
        // model never asked for.
        bool named_undeclared = false;
        {
            std::vector<ToolCall> batch;
            size_t cur = 0, first_begin = std::string::npos;
            // A span starts at any opener spelling the parser accepts (the
            // table behind bare_native_opener_len_at). A name-tag spelling
            // (`<name>`, `<parameter_name>`) counts only when the name it
            // carries is declared: `<name>` is ordinary prose often enough
            // that an undeclared one must not start -- and break -- a batch.
            // An opener in the batch, in ANY of its spellings. mode22_name is
            // set (and mode22_params points at the first real parameter) when
            // the opener is the mode-22 `<parameter=NAME>` spelling: a call
            // whose name wears a parameter tag. Reporting it here rather than
            // in the separate mode-22 pass lets ONE batch mix the spellings --
            // `<function=X>` then `<parameter=Y>`, the shape a planning turn
            // emits when it drifts mid-batch (drift corpus 03a8a851, b645b48f).
            std::string mode22_name;
            size_t mode22_params = std::string::npos;
            auto next_opener = [&](size_t from, std::string* m22_name = nullptr,
                                   size_t* m22_params = nullptr) -> size_t {
                if (m22_name) m22_name->clear();
                if (m22_params) *m22_params = std::string::npos;
                for (size_t c = text_in.find('<', from); c != std::string::npos;
                     c = text_in.find('<', c + 1)) {
                    std::string nm_probe;
                    size_t after_probe;
                    if (parse_function_opener(text_in, c, nm_probe, after_probe)) return c;
                    // mode-22 opener: `<parameter=NAME>` (declared) whose next
                    // non-space byte opens the first REAL parameter -- mode 22's
                    // own gate, so an ordinary `<parameter=key>` value is not
                    // mistaken for a call opener.
                    {
                        std::string pn;
                        size_t pgt = 0;
                        if (find_parameter_opener(text_in, c, &pn, &pgt) == c) {
                            const size_t nxt = find_parameter_opener(text_in, pgt);
                            std::string canon = declared(pn) ? pn : canonical_declared_name(tools, pn);
                            const size_t nb = text_in.find_first_not_of(" \t\r\n", pgt);
                            if (!canon.empty() && nxt != std::string::npos && nb == nxt) {
                                if (m22_name) *m22_name = canon;
                                if (m22_params) *m22_params = nxt;   // a real parameter follows
                                return c;
                            }
                            // Zero-argument mode-22 call: the opener is followed
                            // by `</function>` (no parameter), and the tool takes
                            // no required args (issue #38, memex_recall). The
                            // aborted-call case -- a required-args tool opened and
                            // left empty -- has parameters missing and stays
                            // residue. m22_params points at the `</function>`, so
                            // the synthesized span is `<function=NAME>\n</function>`.
                            if (!canon.empty() && nb != std::string::npos &&
                                text_in.compare(nb, 11, "</function>") == 0 &&
                                no_required_params(canon)) {
                                if (m22_name) *m22_name = canon;
                                if (m22_params) *m22_params = nb;
                                return c;
                            }
                        }
                    }
                    const size_t nl = bare_native_opener_len_at(text_in, c);
                    if (!nl || text_in.compare(c, 10, "<function=") == 0) continue;
                    size_t q = text_in.find_first_not_of(" \t\r\n", c + nl);
                    if (q == std::string::npos) continue;
                    if (text_in[q] == '<' && bare_native_opener_len_at(text_in, q)) continue;  // empty; the next tag decides
                    size_t e = q;
                    while (e < text_in.size() && !isspace((unsigned char)text_in[e]) && text_in[e] != '<') e++;
                    std::string nm = text_in.substr(q, e - q);
                    while (!nm.empty() && (nm.back() == '>' || nm.back() == '"' || nm.back() == '\'')) nm.pop_back();
                    if (declared(nm) || !canonical_declared_name(tools, nm).empty()) return c;
                }
                return std::string::npos;
            };
            for (;;) {
                size_t fb = next_opener(cur, &mode22_name, &mode22_params);
                if (fb == std::string::npos) break;
                const bool synth = !mode22_name.empty();
                // A zero-argument mode-22 call (issue #38): mode22_params points
                // AT its `</function>`, so the call ends right there. Build the
                // span directly -- searching for the closer would run past this
                // call's `</function>` to the NEXT call's and sweep its
                // parameters in.
                const bool synth_zero_arg =
                    synth && text_in.compare(mode22_params, 11, "</function>") == 0;
                if (synth_zero_arg) {
                    ToolCall tc;
                    if (!parse_native_xml_call("<function=" + mode22_name + ">\n</function>", tc)) break;
                    tc.name = mode22_name;
                    tc.source_begin = fb;
                    tc.source_end = mode22_params + 11;
                    if (first_begin == std::string::npos) first_begin = fb;
                    cur = tc.source_end;
                    batch.push_back(std::move(tc));
                    if (cur >= text_in.size()) break;
                    continue;
                }
                // For a mode-22 opener the call body starts at the first real
                // parameter; for every other spelling the span is the opener
                // and what follows, exactly as before.
                std::string span = text_in.substr(synth ? mode22_params : fb);
                const size_t body_at = synth ? mode22_params : fb;
                size_t fe = native_function_closer(span);
                if (fe == std::string::npos) fe = span.find("</function>");
                size_t span_end = text_in.size();
                if (fe != std::string::npos) {
                    span = span.substr(0, fe + 11);
                    span_end = body_at + fe + 11;
                } else {
                    // No </function> at all. Running to end-of-text sweeps the
                    // NEXT call's parameters into this one, and
                    // parse_native_xml_call refuses an unterminated final
                    // parameter when another <parameter= follows -- correctly,
                    // because within ONE call that is mangling. In a batch the
                    // next <parameter= belongs to the next call. Bounding the
                    // span at the next call boundary keeps that refusal
                    // meaningful: the boundary is what distinguishes "the model
                    // started another call" from "the model mangled this one".
                    size_t nb = next_opener(std::max(fb, body_at) + 1);
                    // BOTH markers bound the call. `</tool_call>` is the one
                    // that matters: a batch separated by `</tool_call>\n<tool_call>`
                    // ends each call at the CLOSER, and searching only for the
                    // opener swept the closer into the value (find("<tool_call>")
                    // does not match inside "</tool_call>").
                    for (const char* m : {"</tool_call>", "<tool_call>"}) {
                        const size_t p = text_in.find(m, fb + 1);
                        if (p != std::string::npos && (nb == std::string::npos || p < nb)) nb = p;
                    }
                    if (nb != std::string::npos) {
                        span = text_in.substr(body_at, nb - body_at);
                        span_end = nb;
                    }
                }
                // A mode-22 span carries no `<function=NAME>` head -- it starts
                // at the first parameter. Synthesize the head so the one parser
                // reads it, exactly as the standalone mode-22 pass does; the
                // source span still points at the original `<parameter=NAME>`.
                if (synth) span = "<function=" + mode22_name + ">\n" + span;
                // Hallucinated-result rule (2026-08-21): a <result>/<output>
                // element enclosing this span -- or inside it -- means the
                // "call" is part of invented result output. Break the batch
                // rather than promoting fiction to a call; other modes apply
                // the same guard to their own spans. Uses the BOUNDED end, so
                // the guard inspects this call's span and not its successors'.
                if (hallucinated_result_around(text_in, fb, span_end)) break;
                ToolCall tc;
                // Stop at the first span that will not resolve rather than
                // skipping it: keep what was actually read, never invent the
                // rest. An unparseable FIRST span leaves the batch empty and
                // falls through to the modes below exactly as before.
                if (!parse_native_xml_call(span, tc)) break;
                if (synth) tc.name = mode22_name;   // declared by construction
                else if (!declared(tc.name)) {
                    // `<function name="NAME>` -- the quote opens and never
                    // closes before '>', so the name arrives as `"NAME`. Strip
                    // an unmatched quote ONLY when what remains names a
                    // DECLARED tool: that reads the name the caller offered
                    // rather than guessing one, the same rule mode 22 uses.
                    std::string alt = tc.name;
                    while (!alt.empty() && (alt.front() == '"' || alt.front() == '\''))
                        alt.erase(alt.begin());
                    while (!alt.empty() && (alt.back() == '"' || alt.back() == '\''))
                        alt.pop_back();
                    // `<function=name>` with the real name on the FOLLOWING
                    // line. The opener carries a placeholder where the name
                    // belongs -- the same habit mode 18 already tolerates for
                    // `<name>`, one tag over. Only the next line, only when it
                    // names a declared tool.
                    if (!declared(alt) && (alt == "name" || alt == "tool_name" ||
                                           alt == "function")) {
                        const size_t og = span.find('>');
                        if (og != std::string::npos) {
                            size_t ls = span.find_first_not_of(" \t\r\n", og + 1);
                            if (ls != std::string::npos) {
                                size_t le = span.find_first_of("\r\n<", ls);
                                std::string cand = span.substr(
                                    ls, (le == std::string::npos ? span.size() : le) - ls);
                                while (!cand.empty() && isspace((unsigned char)cand.back()))
                                    cand.pop_back();
                                if (declared(cand)) alt = cand;
                            }
                        }
                    }
                    // A name that differs only in CASE is the model's own name
                    // for the tool; refusing on case alone loses a call for
                    // nothing. Unique match only, so an ambiguous registry
                    // still refuses.
                    if (!declared(alt)) {
                        const std::string canon = canonical_declared_name(tools, alt);
                        if (!canon.empty()) alt = canon;
                    }
                    if (alt == tc.name || !declared(alt)) { named_undeclared = true; break; }
                    tc.name = alt;
                }
                tc.source_begin = fb;
                tc.source_end = span_end;
                if (first_begin == std::string::npos) first_begin = fb;
                cur = tc.source_end;
                batch.push_back(std::move(tc));
                if (cur >= text_in.size()) break;
            }
            if (!batch.empty()) {
                // Absorb a separator that carries no content into the preceding
                // call's span. append_text slices on these spans, so a gap left
                // between two calls is emitted as VISIBLE TEXT -- which would
                // put the raw <tool_call> marker back in front of the user, the
                // exact leak the batch exists to stop.
                for (size_t i = 0; i + 1 < batch.size(); i++) {
                    const size_t gb = batch[i].source_end, ge = batch[i + 1].source_begin;
                    if (ge <= gb) continue;
                    const std::string gap = text_in.substr(gb, ge - gb);
                    if (only_dialect_control_bytes(gap) ||
                        gap.find_first_not_of(" \t\r\n") == std::string::npos)
                        batch[i].source_end = ge;
                }
                if (prefix) *prefix = text_in.substr(0, first_begin);
                if (remaining_text) *remaining_text = "";
                fprintf(stderr, "[q27] bare native-dialect call(s) recovered: %zu\n",
                        batch.size());
                drift_mode_hint = "native";
                return batch;
            }
        }
        // (b) the chimera
        size_t jb = text_in.find("{\"name\"");
        if (jb != std::string::npos) {
            size_t pp = text_in.find("<parameter=", jb);
            if (pp != std::string::npos) {
                // name = first JSON string value after {"name"
                size_t q1 = text_in.find('"', text_in.find(':', jb) + 1);
                size_t q2 = q1 == std::string::npos ? q1 : text_in.find('"', q1 + 1);
                if (q2 != std::string::npos && q2 < pp) {
                    const std::string nm = text_in.substr(q1 + 1, q2 - q1 - 1);
                    if (declared(nm) &&
                        !hallucinated_result_around(text_in, jb, text_in.size())) {
                        // reuse the XML walk by synthesizing a well-formed span;
                        // a missing final </parameter> is tolerated by closing at EOF
                        std::string span = "<function=" + nm + ">\n" + text_in.substr(pp);
                        if (span.find("</parameter>", span.rfind("<parameter=")) == std::string::npos)
                            span += "\n</parameter>";
                        span += "\n</function>";
                        ToolCall tc;
                        if (parse_native_xml_call(span, tc)) {
                            tc.source_begin = jb;
                            tc.source_end = text_in.size();
                            if (prefix) *prefix = text_in.substr(0, jb);
                            if (remaining_text) *remaining_text = "";
                            fprintf(stderr, "[q27] drift mode 17: chimera json-head/xml-params %s\n",
                                    nm.c_str());
                            drift_mode_hint = "17";
                            return std::vector<ToolCall>{std::move(tc)};
                        }
                    }
                }
            }
        }
        // (c) DRIFT MODE 20 (2026-08-19, same A/B as mode 19): a BATCH of calls
        // in which the tool NAME is structurally absent --
        //   <tool_calls>\n<tool_name>\n<parameter=file_path>\n/x\n</parameter>\n</function>
        //   <tool>\n<tool_name>\n<parameter=file_path>\n/y\n</parameter>\n</function>
        // Plural `<tool_calls>` opener, an EMPTY `<tool_name>` where the name
        // belongs, `<tool>` as a separator, and closers that do not match their
        // openers. Mode 14 owns `<tool_name>NAME</tool_name>`; this fires only
        // when the tag is empty, so the two cannot collide.
        //
        // With no name in the bytes, the only way back is to infer it from the
        // parameter keys -- which is exactly what infer_tool_name() already does
        // for JSON mode 6, including its refuse-on-tie rule. A wrong tool is
        // worse than an un-rescued one, so inference failure leaves it as text.
        //
        // hallucinated_result_around() is what keeps this off the
        // HALLUCINATED-RESULT shape (<tool_calls><result><name>X</name>
        // <output>...<tool_name><parameter=...), which mode 16 pins with a
        // negative fixture: those are invented tool OUTPUT, and promoting them
        // to calls would feed the model's own fiction back as if a tool had
        // produced it. That shape leaves <result> UNCLOSED, so the per-call
        // span below reads as enclosed by it and is refused -- while a batch
        // that merely follows a closed <output> block still recovers.
        {
            static const std::string TN = "<tool_name>";
            std::vector<ToolCall> batch;
            size_t scan = 0;
            while (true) {
                size_t tn = text_in.find(TN, scan);
                if (tn == std::string::npos) break;
                size_t after = text_in.find_first_not_of(" \t\r\n", tn + TN.size());
                scan = tn + TN.size();
                // empty tag only: a named <tool_name> belongs to mode 14
                if (after == std::string::npos ||
                    text_in.compare(after, 11, "<parameter=") != 0)
                    continue;
                size_t end = native_function_closer(text_in, after);
                if (end == std::string::npos) end = text_in.find("</function>", after);
                size_t alt = text_in.find(TN, after);
                if (end == std::string::npos || (alt != std::string::npos && alt < end))
                    end = alt == std::string::npos ? text_in.size() : alt;
                else
                    end += 11;
                std::string span = "<function=q27_unnamed>\n" +
                                   text_in.substr(after, end - after);
                if (span.find("</function>") == std::string::npos) span += "\n</function>";
                ToolCall tc;
                if (hallucinated_result_around(text_in, tn, end)) {
                    fprintf(stderr, "[q27] drift mode 20: hallucinated <result>/<output> "
                                    "block encloses the call, refusing\n");
                    continue;
                }
                if (!parse_native_xml_call(span, tc) || tc.arguments.empty()) continue;
                const std::string nm = infer_tool_name(*tools, tc.arguments);
                if (nm.empty() || !declared(nm)) continue;
                tc.name = nm;
                tc.ok = true;
                // STAMP PER-CALL spans (review 2026-08-20): the prior code
                // only set batch.front().source_begin and batch.back().source_end,
                // leaving middle calls with source_begin == npos. The unclosed-tail
                // recovery (recover_unclosed_tool_tail) and the closed-wrapper
                // resolver (resolve_ordered_tool_segments) both compute
                // `raw.substr(cursor, c.source_begin - cursor)`; an npos middle
                // call underflows to SIZE_MAX and substr throws out_of_range
                // inside the request handler. Stamp every call so the gaps
                // between calls and the trailing text after the last call are
                // represented correctly.
                tc.source_begin = tn;
                tc.source_end = end;
                batch.push_back(std::move(tc));
                scan = end;
            }
            if (!batch.empty()) {
                if (prefix) *prefix = text_in.substr(0, batch.front().source_begin);
                // trailing text after the last call's region stays visible as
                // text (the prior override glued it to the last call's source_end)
                if (remaining_text) *remaining_text = text_in.substr(batch.back().source_end);
                fprintf(stderr, "[q27] drift mode 20: recovered %zu nameless <tool_name> call(s)\n",
                        batch.size());
                drift_mode_hint = "20";
                return batch;
            }
        }
        // (d) DRIFT MODE 21 (2026-08-20, issue #24 follow-up): the parameters
        // arrive with NO opener of any kind. Not `<function=`, not the mode-19
        // attribute spelling, not mode-18's bare `<name>`, not mode-20's empty
        // `<tool_name>` -- the emission simply starts at the first
        // `<parameter=KEY>` and ends at `</function>`:
        //
        //   <parameter=filePath>\n/code/x.go\n</parameter>
        //   <parameter=newString>\n...\n</parameter>\n</function>
        //
        // One degradation further than mode 20: there the tag was present and
        // empty, here there is no tag at all, so the name has to come entirely
        // from the parameter keys. Same infer_tool_name() with the same
        // refuse-on-tie rule, which also means a model that mangles the KEY
        // spelling (camelCase where the schema declares snake_case) is
        // correctly NOT rescued -- guessing a tool from keys that match nothing
        // is how you execute the wrong call.
        //
        // The `</function>` requirement is the guard against eating prose: a
        // paragraph can mention <parameter= in passing, but a closing tag it
        // never opened is dialect, not English.
        {
            size_t ps = find_parameter_opener(text_in, 0);
            size_t fe = ps == std::string::npos ? std::string::npos
                                                : native_function_closer(text_in, ps);
            if (ps != std::string::npos && fe == std::string::npos)
                fe = text_in.find("</function>", ps);
            if (fe != std::string::npos) {
                // Scoped hallucinated-result guard (2026-08-21). The span is
                // [ps, fe+11) -- the parameter list itself. A <result>/<output>
                // tag inside it, or an unclosed one enclosing it, is invented
                // output and refused; one that opened AND closed before ps is
                // prior context and must not veto this call. (The first attempt
                // at this scoped it only at the END, so a closed block earlier
                // in the tail still killed the call -- the very leak it meant
                // to fix.)
                const size_t span_end = fe + 11;
                if (hallucinated_result_around(text_in, ps, span_end)) {
                    fprintf(stderr,
                            "[q27] drift mode 21: hallucinated <result>/<output> "
                            "block in the parameter span, refusing\n");
                } else {
                std::string span =
                    "<function=q27_unnamed>\n" + text_in.substr(ps, fe + 11 - ps);
                ToolCall tc;
                if (!named_undeclared && parse_native_xml_call(span, tc) &&
                    !tc.arguments.empty()) {
                    // A parameter literally named `name` whose value names a
                    // DECLARED tool is the tool (2026-08-22, two live misses):
                    //   <parameter=name>\nBash</parameter>\n<parameter=command>...
                    // Read it and drop the key, rather than inferring from
                    // {name, command} -- which ties Bash against Monitor under
                    // the real registry and refuses. Openerless path only: an
                    // explicit <function=X> keeps `name` as an ordinary argument.
                    std::string nm;
                    auto nit = tc.arguments.find("name");
                    if (nit != tc.arguments.end() && nit->is_string()) {
                        std::string cand = nit->get<std::string>();
                        const size_t a = cand.find_first_not_of(" \t\r\n");
                        cand = a == std::string::npos ? "" :
                               cand.substr(a, cand.find_last_not_of(" \t\r\n") - a + 1);
                        const std::string canon = declared(cand) ? cand
                                                  : canonical_declared_name(tools, cand);
                        if (!canon.empty()) { nm = canon; tc.arguments.erase(nit); }
                    }
                    if (nm.empty()) nm = infer_tool_name(*tools, tc.arguments);
                    if (!nm.empty() && declared(nm)) {
                        tc.name = nm;
                        tc.ok = true;
                        tc.source_begin = ps;
                        tc.source_end = fe + 11;
                        if (prefix) *prefix = text_in.substr(0, ps);
                        if (remaining_text) *remaining_text = "";
                        fprintf(stderr,
                                "[q27] drift mode 21: openerless parameter list -> %s\n",
                                nm.c_str());
                        drift_mode_hint = "21";
                        return std::vector<ToolCall>{std::move(tc)};
                    }
                }
                }
            }
        }
        // (e) DRIFT MODE 22 (2026-08-20, chaudhryfaisal on issue #24,
        // corroborated the same day by the dialect survey on a different client
        // and schema): the tool NAME arrives as the first <parameter=...>,
        // unclosed, with the real parameters after it --
        //
        //   <parameter=bash>\n<parameter=command>\nls /code\n</parameter>\n</function>
        //
        // Sideways from mode 21 rather than one step further down: the name IS
        // in the bytes, wearing a parameter's tag. Because that leading tag
        // never closes, the parameter walk fails outright and mode 21 is left
        // with zero arguments to infer from -- which is why this died in
        // SILENCE, with no rescue line and no Q27_DRIFT_CORPUS entry.
        //
        // Engages only on a DECLARED tool name in a tag carrying NO value: the
        // next non-space byte must open the first real parameter. That reads a
        // name the caller offered instead of inferring one, so an undeclared
        // name falls through to mode 21 rather than being invented into a call.
        //
        // Runs AFTER mode 21 deliberately. A MIXED emission (survey capture
        // 001: a {"function=Read> call followed by a <parameter=Bash> one) is
        // rescued by mode 21 as the FIRST call, and generation order is worth
        // more than the second call -- running this first would return Bash and
        // drop the Read the model asked for first.
        {
            static const std::string PO = "<parameter=";
            static const std::string FC = "</function>";
            std::vector<ToolCall> batch;
            size_t first_begin = std::string::npos, cur = 0;
            for (;;) {
                std::string nm;
                size_t gt1 = 0;
                const size_t ps = find_parameter_opener(text_in, cur, &nm, &gt1);
                if (ps == std::string::npos) break;
                const size_t gt = gt1 - 1;   // the '>' itself
                const size_t nxt = find_parameter_opener(text_in, gt + 1);
                // `<parameter=task>` against a declared `Task`: same tool, the
                // model's own casing.
                if (!declared(nm)) {
                    const std::string canon = canonical_declared_name(tools, nm);
                    if (!canon.empty()) nm = canon;
                }
                if (nxt == std::string::npos || !declared(nm) ||
                    text_in.find_first_not_of(" \t\r\n", gt + 1) != nxt) {
                    cur = gt + 1;   // a parameter, not an opener
                    continue;
                }
                const size_t fe = text_in.find(FC, nxt);
                if (fe == std::string::npos) break;
                ToolCall tc;
                const std::string span =
                    "<function=" + nm + ">\n" + text_in.substr(nxt, fe + FC.size() - nxt);
                if (hallucinated_result_around(text_in, ps, fe + FC.size())) {
                    fprintf(stderr, "[q27] drift mode 22: hallucinated <result>/<output> "
                                    "block encloses the call, refusing\n");
                    cur = fe + FC.size();
                    continue;
                }
                if (parse_native_xml_call(span, tc) && !tc.arguments.empty()) {
                    tc.name = nm;
                    tc.ok = true;
                    tc.source_begin = ps;
                    tc.source_end = fe + FC.size();
                    if (first_begin == std::string::npos) first_begin = ps;
                    batch.push_back(std::move(tc));
                }
                cur = fe + FC.size();
            }
            if (!batch.empty()) {
                if (prefix) *prefix = text_in.substr(0, first_begin);
                if (remaining_text) *remaining_text = "";
                fprintf(stderr, "[q27] drift mode 22: parameter-as-opener -> %zu call(s)\n",
                        batch.size());
                drift_mode_hint = "22";
                return batch;
            }
        }
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
    // mode 23: args-only object + XML closers (round 6). Same last-resort
    // tier as mode 11; the closer-evidence gate keeps prose JSON as text.
    if (out.empty() && allow_eof_repair && tools && tools->is_array() &&
        recover_args_object_call(text, *tools, out)) {
        drift_mode_hint = "23";
        const std::string before = strip_ws2(text.substr(0, out.front().source_begin));
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
        drift_mode_hint = modes;
    }
    // LAST RESORT, once: the JSON dialect with its terminator written in XML.
    // Only reached when nothing else recovered, and only when blanking actually
    // changes the bytes, so it cannot alter a successful parse. The recursion
    // guard keeps it to a single retry.
    if (out.empty() && !in_dialect_retry) {
        const std::string blanked = blank_dialect_closers_outside_strings(text_in);
        if (blanked != text_in) {
            in_dialect_retry = true;
            auto again = parse_bare_tool_calls(blanked, prefix, tools, allow_o10,
                                               allow_eof_repair, remaining_text);
            in_dialect_retry = false;
            if (!again.empty()) {
                fprintf(stderr, "[q27] drift: JSON call terminated in the XML dialect -> %zu\n",
                        again.size());
                drift_mode_hint = "xmlclose";
                return again;
            }
        }
    }
    if (!out.empty()) {
    } else if (looks_like_intended_tool_call(text_in)) {
        // ntools distinguishes plumbing (-1/0) from a schema/inference miss (>0 =
        // mode-6 args didn't confidently match any tool). Longer window so the call
        // (not just a long preamble) is visible for post-hoc arg-shape diagnosis.
        fprintf(stderr, "[drift] UN-RESCUED (ntools=%d) intended tool call: %.400s\n",
                tools ? (int)tools->size() : -1, text_in.c_str());
        // The corpus record for this miss (Q27_DRIFT_CORPUS, full and
        // redacted, where the stderr line caps at 400 chars -- issue #4's
        // payload was invisible for that reason) is written by the
        // parse_bare_tool_calls wrapper below, alongside every success.
    }
    return out;
}

// The capture wrapper. Every recovery mode returns from the impl at its own
// site, so the record is made here, from the result: recovered:<mode> (the
// impl leaves the mode in drift_mode_hint), suppressed (the strict leg), or
// unrescued (dialect residue, nothing recovered -- drift_bearing is wider than
// the UN-RESCUED warning's bar on purpose; that bar is unchanged). Depth 1
// only: the mode-10 probe and the dialect retry re-enter through this same
// wrapper and are not turns; an inner call's hint is always overwritten by
// the outer call's own success site, so the hint names the outer mode.
// Nothing may call parse_bare_tool_calls_impl directly. With
// Q27_DRIFT_CORPUS unset this is the old function with one extra integer
// increment.
//
// Records are per top-level parse, which is per turn on the non-stream
// TEXT/TOOL path. The holdbacks (reasoning here, streaming in server.cu)
// classify per candidate, and re-classify a deferred candidate at finish
// with repair enabled, so one candidate can yield `unrescued` and then
// `recovered:*`. corpus_dedup.py folds those into one shape with both
// outcomes counted; `count` there is captures, not turns.
inline std::vector<ToolCall> parse_bare_tool_calls(const std::string& text_in,
                                                   std::string* prefix,
                                                   const json* tools,
                                                   bool allow_o10,
                                                   bool allow_eof_repair,
                                                   std::string* remaining_text) {
    DriftParseScope drift_scope;
    bool from_strict_miss = false;
    if (drift_parse_depth == 1) {
        drift_mode_hint.clear();
        from_strict_miss = !drift_last_strict_miss.empty() && drift_last_strict_miss == text_in;
        drift_last_strict_miss.clear();
    }
    DriftCtxScope drift_miss(from_strict_miss, false, nullptr);
    auto out = parse_bare_tool_calls_impl(text_in, prefix, tools, allow_o10,
                                          allow_eof_repair, remaining_text);
    // Every recovery mode returns from the impl at its own site, so the span
    // widening over adjacent dialect residue -- and the remaining-text that
    // depends on the spans -- happens here, once, for all of them.
    if (!out.empty()) {
        absorb_dialect_residue(text_in, out);
        if (remaining_text) {
            std::string remainder;
            remainder.reserve(text_in.size());
            size_t cursor = 0;
            for (const auto& call : out) {
                if (call.source_begin == std::string::npos || call.source_begin < cursor) { cursor = std::string::npos; break; }
                remainder.append(text_in, cursor, call.source_begin - cursor);
                cursor = call.source_end;
            }
            if (cursor != std::string::npos) {
                remainder.append(text_in, cursor, std::string::npos);
                *remaining_text = std::move(remainder);
            }
        }
    }
    if (drift_parse_depth == 1 && drift_corpus_path()) {
        if (!out.empty())
            capture_drift(text_in, "recovered:" + (drift_mode_hint.empty()
                                                       ? std::string("?") : drift_mode_hint), tools);
        else if (looks_like_intended_tool_call(text_in) || drift_bearing(text_in, tools))
            capture_drift(text_in, tool_strict() ? "suppressed" : "unrescued", tools);
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

// A TOOL body that recovered NOTHING and consists only of dialect control
// bytes carries no information for the user -- showing it just leaks protocol.
//
// The shape that made this necessary (2026-08-22, live): the model emitted the
// opener TWICE and then hit EOS --
//
//   <tool_call>          <- structural; the splitter erases it, enters TOOL
//   <tool_call>          <- inside TOOL only "</tool_call>" delimits, so this
//   (EOS)                   is BODY CONTENT, not a marker
//
// unfinished_tool_wrapper() reports false (it detects length/budget truncation,
// not EOS-inside-wrapper), so the unclosed-tail path never runs. emit_tool()
// then fails to parse, the drift chain finds nothing, and the fallback prints
// ToolCall::raw -- which IS the literal string "<tool_call>". The user's
// visible answer for the whole turn was that one marker.
//
// Deliberately NARROW: whitespace plus <tool_call>/</tool_call> only. Dialect
// tags that can carry a payload (<function=...>, <parameter=...>) are NOT
// stripped, because a body containing those is a malformed CALL whose bytes the
// user may need to see -- that is drift, and drift stays visible. This only
// suppresses residue that is provably content-free.
//
// The turn is lost either way; this decides what the user sees, and callers log
// so the loss is not silent.

inline std::string take_unclosed_final_tool_segment(
    std::vector<std::pair<StreamSplitter::Chan,std::string>>& segments,
    bool wrapper_incomplete) {
    if(!wrapper_incomplete || segments.empty() ||
       segments.back().first!=StreamSplitter::TOOL) return {};
    std::string raw=std::move(segments.back().second);
    segments.pop_back();
    return raw;
}

// Route an UNCLOSED-<tool_call> tail (wrapper stripped, generation truncated)
// through the drift chain. Complements bb60661, which routed a CLOSED-wrapped
// TOOL segment through parse_bare_tool_calls but deliberately left the unclosed
// one as raw text. Same safety bar that commit documented:
//
//   "no EOF repair, because a closed wrapper means drift rather than
//    truncation, and an unclosed one is removed upstream by
//    take_unclosed_final_tool_segment. Executing half a Write stays refused."
//
// allow_eof_repair=false inside means only bytes the parser accepts as a
// well-formed call execute: an inner call whose JSON/XML is COMPLETE (its
// </tool_call> merely never arrived because the generation was cut) recovers on
// mode 1 / the native dialect, while a value genuinely cut mid-parse (a
// partial Write) is rejected and shown as text. That is the same refusal
// boundary that protects the closed-wrapper path, just reached from the tail.
//
// emit_call returns true when the call was accepted (recovered++), false to
// leave its bytes as text (ineligible / refused -- never re-runs the chain, so
// tool_choice / disable_parallel_tool_use rejections can't be resurrected).
// Returns the number of accepted calls.
template<class EmitText, class EmitCall>
inline size_t recover_unclosed_tool_tail(const std::string& raw,
    const json* tools, EmitText emit_text, EmitCall emit_call) {
    if(raw.empty()) return 0;
    if(!tools || !tools->is_array() || tools->empty()){
        emit_text(raw); return 0;
    }
    std::string pre; // parse_bare_tool_calls writes here; we use cursor/source_begin instead (see comment above)
    DriftCtxScope drift_wrapped(true,false,tools);
    auto calls=parse_bare_tool_calls(raw,&pre,tools,true,false,nullptr);
    if(calls.empty()){ emit_text(raw); return 0; }
    size_t cursor=0,recovered=0;
    for(auto& c:calls){
        // defense-in-depth (review 2026-08-20): mode 20 now stamps per-call
        // spans, but if any future drift mode returns a batch with a missing
        // source_begin (npos) or a span that overlaps the cursor we've already
        // emitted, raw.substr(cursor, c.source_begin-cursor) would underflow
        // to SIZE_MAX and throw out_of_range inside the request handler.
        // Refuse to emit such a call -- dump the remaining tail as text and
        // stop -- rather than crash.
        if (c.source_begin == std::string::npos || c.source_begin < cursor) {
            emit_text(raw.substr(cursor));
            break;
        }
        emit_text(raw.substr(cursor,c.source_begin-cursor));
        cursor=c.source_end;
        if(emit_call(c)) recovered++;
        else emit_text(c.raw);
    }
    emit_text(raw.substr(cursor));
    return recovered;
}

// A finished reasoning block that may hold a complete tool call (issue #38).
// The whole-block consumers -- the non-stream resolver below and both
// /v1/responses legs, which accumulate reasoning and emit it once -- call
// this rather than each carrying the logic (the streaming SSE handlers use
// StreamToolRouter for the chunked case). The <tool_call> wrapper tokens are
// blanked, length-preserving, because the display lexer reads a literal one
// as an inert container and would never arm on the call inside it; fences
// and inline code are untouched, so a fenced example or an inline-code
// mention stays reasoning -- calling the parser directly here WOULD execute
// a fenced example (verified when 152d3a2 landed). accept(call) takes the
// call or refuses it; refused bytes stay reasoning. Returns the reasoning
// that remains when something was taken, else the input untouched.
template<class Accept>
inline std::string recover_calls_from_reasoning(const std::string& reasoning,
                                                const json* tools, size_t& taken,
                                                Accept&& accept) {
    taken=0;
    if(!tools || !tools->is_array() || tools->empty() ||
       !looks_like_intended_tool_call(reasoning)) return reasoning;
    DriftNames names;
    declared_tool_names(tools,names);
    std::string scan=reasoning;
    for(const char* marker : {"<tool_call>","</tool_call>"}) {
        const size_t n=std::char_traits<char>::length(marker);
        for(size_t at=scan.find(marker);at!=std::string::npos;at=scan.find(marker,at+1))
            scan.replace(at,n,std::string(n,' '));
    }
    DriftCtxScope drift_think(false,true,tools);
    const size_t drift_before=drift_records_written;
    BareToolTextHoldback hb;
    std::string kept;
    auto visible=[&](const std::string& text){ kept+=text; };
    auto classify=[&](const std::string& source,bool allow_repair,auto&& emit_visible) {
        std::string prefix,residual;
        auto calls=parse_bare_tool_calls(source,&prefix,tools,true,allow_repair,&residual);
        if(calls.empty()) return BareToolCandidateResult{};
        size_t cursor=0; bool any=false;
        for(auto& call:calls) {
            if(call.source_begin==std::string::npos || call.source_begin<cursor) break;
            const size_t begin=call.source_begin,end=call.source_end;
            emit_visible(source.substr(cursor,begin-cursor));
            cursor=end;
            if(call.ok && accept(call)) { taken++; any=true; }
            else emit_visible(source.substr(begin,end-begin));
        }
        emit_visible(source.substr(cursor));
        return BareToolCandidateResult{true,any};
    };
    hb.route(scan,names,visible,classify);
    hb.finish(false,names,visible,classify);
    // The holdback decides whether the parser runs at all; when it found no
    // candidate the corpus would otherwise never see a reasoning block that
    // passed the intent check.
    if(!taken && drift_corpus_path() && drift_records_written==drift_before)
        capture_drift(reasoning,"unrescued",tools);
    return taken?kept:reasoning;
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
            // defense-in-depth (review 2026-08-20): see recover_unclosed_tool_tail.
            // If any drift mode hands us a call with an unset source_begin
            // (npos) or an overlapping span, substr() would throw; dump the
            // remaining tail as visible text and stop.
            if (call.source_begin == std::string::npos || call.source_begin < cursor) {
                out.append_visible_text(raw.substr(cursor));
                break;
            }
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
    // trailing dialect residue ahead of a wrapped segment, or at the end of
    // the turn, is garbage rather than text (see absorb_dialect_residue)
    auto drop_trailing_residue=[&]() {
        if(!tools) return;
        const DialectResidueSuffix r=dialect_residue_suffix(pending_text);
        if(r.complete) pending_text.erase(r.start);
    };
    for(const auto& segment:segments) {
        if(segment.first==StreamSplitter::TEXT) {
            pending_text+=segment.second;
            continue;
        }
        if(segment.first==StreamSplitter::TOOL) drop_trailing_residue();
        flush_pending_text(false);
        if(segment.first==StreamSplitter::THINK) {
            // Issue #38: a complete tool call can arrive INSIDE reasoning, and
            // reasoning used to pass straight through to out.reasoning, so the
            // call was never parsed -- it reached the client as visible think
            // text and never fired. recover_calls_from_reasoning applies the
            // same display-context rules the TEXT branch uses; bytes that are
            // not a call are reasoning exactly as before.
            size_t taken=0;
            const std::string kept=recover_calls_from_reasoning(
                segment.second,tools,taken,[&](ToolCall& call) {
                    if(!eligible(call.name,out.calls.size())) return false;
                    out.append_tool_call(std::move(call));
                    return true;
                });
            out.reasoning+=kept;
            if(taken) {
                out.recovered+=taken;
                fprintf(stderr,"[tool-fallback] %zu call(s) recovered from "
                               "reasoning (nonstream)\n",taken);
            }
            continue;
        }
        const std::string body=strip_ws2(segment.second);
        DriftCtxScope drift_wrapped(true,false,tools);
        // A wrapped segment is not guaranteed to hold exactly one call. When it
        // holds more, or carries text after the last one, the strict parser
        // would take the first and drop the remainder silently -- which is how
        // a model's whole 14-call plan became prose it then read back as its
        // own answer. Hand those to the batch-aware chain instead.
        if(tools && wrapped_body_exceeds_one_call(body)) { append_text(body,false); continue; }
        ToolCall call=parse_tool_call(body);
        if(call.ok) {
            if(eligible(call.name,out.calls.size())) out.append_tool_call(std::move(call));
            // A parsed call the caller REJECTED stays text and is not offered a
            // second chance below: re-running the chain on it would resurrect
            // exactly what tool_choice/parallel-calls just refused.
            else out.append_visible_text(call.raw);
            continue;
        }
        // parse_tool_call refused. Everything it knows is the strict JSON form
        // and the trained XML dialect; the whole drift catalogue (modes 10, 11,
        // 13, 14, 15, 17, 20, 21) lives in parse_bare_tool_calls and used to be
        // reachable ONLY from the TEXT branch above. So a model that emitted
        // the <tool_call> wrapper correctly and then drifted INSIDE it was
        // unrecoverable no matter how many modes were catalogued -- three modes
        // shipped in one week with passing tests and could never fire here.
        // Same chain, same eligibility, same recovered counter as text.
        //
        // No EOF repair: the wrapper closed, so this is drift, not truncation.
        // A wrapper that did NOT close was already removed by
        // take_unclosed_final_tool_segment before this loop.
        if(only_dialect_control_bytes(body)) {
            fprintf(stderr,"[q27] tool body was dialect control bytes only "
                           "(%zu B, e.g. a repeated <tool_call> opener) -- "
                           "turn produced no call; suppressed from visible text\n",
                    body.size());
            capture_drift(body,"suppressed",tools);
            continue;
        }
        append_text(body,false);
    }
    drop_trailing_residue();
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
