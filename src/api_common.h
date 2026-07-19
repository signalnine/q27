// Shared prompt construction + tool-call parsing for the q27 API endpoints.
// Qwopus (qwen35) tool protocol, from the GGUF chat template:
//   system preamble lists tools as JSON inside <tools>...</tools>
//   model emits  <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
//   results go back as user content wrapped in <tool_response>...</tool_response>
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "../third_party/json.hpp"
#include "stream_split.h"

namespace q27 {
using json = nlohmann::json;

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
inline void normalize_cc_billing_header(std::string& sys) {
    static const char* PFX = "x-anthropic-billing-header:";
    if (sys.rfind(PFX, 0) != 0) return;
    size_t cch = sys.find("cch=", 27);
    if (cch == std::string::npos || cch > 160) return;  // header segment only
    size_t v = cch + 4, end = sys.find(';', v);
    if (end == std::string::npos || end == v || end - v > 16) return;
    for (size_t i = v; i < end; ++i) sys[i] = 'f';
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

// Build the full ChatML prompt string. If tools are present they are merged
// into the (first) system message per the template's merged_system behavior.
// think=false appends the empty think block (enable_thinking=false
// convention); the tokenizer matches <think>/</think> as single added tokens.
// stable_off (P8): char offset where the trailing assistant-open begins.
// Everything before it re-renders identically next turn (snapshot-safe);
// everything after (assistant open + think prefill) is per-turn volatile.
inline std::string chatml_prompt(const std::vector<Msg>& msgs, const json& tools,
                                 bool think = true, size_t* stable_off = nullptr) {
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
    if (tools.is_array() && !tools.empty()) {
        p += "<|im_start|>system\n" + tools_preamble(tools);
        if (!sys.empty()) p += "\n\n" + sys;
        p += "<|im_end|>\n";
    } else if (!sys.empty()) {
        p += "<|im_start|>system\n" + sys + "<|im_end|>\n";
    }
    for (size_t i = start; i < msgs.size(); i++)
        p += "<|im_start|>" + strip_ctrl(msgs[i].role) + "\n" + strip_ctrl(msgs[i].content) +
             "<|im_end|>\n";
    if (stable_off) *stable_off = p.size();
    p += "<|im_start|>assistant\n";
    if (!think) p += "<think>\n\n</think>\n\n";
    return p;
}

inline std::string tool_call_text(const std::string& name, const json& args) {
    return "<tool_call>\n{\"name\": \"" + name + "\", \"arguments\": " + args.dump() +
           "}\n</tool_call>";
}

inline std::string tool_response_text(const std::string& out) {
    return "<tool_response>\n" + out + "\n</tool_response>";
}

// Anthropic error envelope, exactly the real API's shape: the SDK inside
// Claude Code reads error.message from it, and CC's compact-vs-retry
// decision substring-matches that message.
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
    if (body.contains("tools"))
        for (auto& t : body["tools"]) {
            if (!t.contains("name")) continue;
            out.push_back({{"type", "function"},
                           {"function", {{"name", t["name"]},
                                         {"description", t.value("description", "")},
                                         {"parameters", t.contains("input_schema")
                                                            ? t["input_schema"]
                                                            : json::object()}}}});
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
                if (b.value("type", "") == "text") sys += b.value("text", "");
        if (!sys.empty()) {
            normalize_cc_billing_header(sys);
            msgs.push_back({"system", sys});
        }
    }
    if (!body.contains("messages")) return msgs;
    for (auto& m : body["messages"]) {
        std::string role = m.value("role", "user"), think, content;
        // guard: const operator[] on a missing key is an abort (json.hpp
        // assertion) -- a content-less message must not kill the server
        if (!m.is_object() || !m.contains("content")) { msgs.push_back({role, content}); continue; }
        if (m["content"].is_string()) content = m["content"];
        else if (m["content"].is_array())
            for (auto& part : m["content"]) {
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
                                if (b.value("type", "") == "text") rc += b.value("text", "");
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
        for (auto& t : body["tools"]) {
            if (!t.is_object() || t.value("type", "") != "function") continue;
            if (!t.contains("function") || !t["function"].is_object()) continue;
            const json& fn = t["function"];
            if (!fn.contains("name") || !fn["name"].is_string()) continue;
            out.push_back({{"type", "function"},
                           {"function", {{"name", fn["name"]},
                                         {"description", fn.value("description", "")},
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

// tool_choice (OpenAI shape): "auto"/absent -> AUTO (unchanged behavior);
// "none" -> NONE (tools stripped from the prompt entirely -- the model gets
// no tool definitions and cannot call anything this turn); "required" or a
// named {"type":"function","function":{"name":...}} -> FORCED. FORCED is a
// soft force (prompt-injected <tool_call> opener + pre-seeded stream router,
// see server.cu) -- it is NOT combined with --constrain-tools grammar
// masking (documented limitation: the grammar's engage trigger scans
// GENERATED text for the <tool_call> marker, which never appears in the
// output when it was injected into the PROMPT instead).
struct ToolChoice {
    enum Mode { AUTO, NONE, FORCED } mode = AUTO;
    std::string forced_name; // empty = any registered tool eligible
};
inline ToolChoice parse_tool_choice(const json& body) {
    ToolChoice tc;
    if (!body.contains("tool_choice")) return tc;
    const json& v = body["tool_choice"];
    if (v.is_string()) {
        if (v == "none") tc.mode = ToolChoice::NONE;
        else if (v == "required") tc.mode = ToolChoice::FORCED;
        // "auto" or any other/unknown string: default AUTO
    } else if (v.is_object() && v.value("type", "") == "function" && v.contains("function") &&
               v["function"].is_object()) {
        tc.mode = ToolChoice::FORCED;
        tc.forced_name = v["function"].value("name", std::string());
    }
    return tc;
}

// Parsed model tool call. `ok` false if the JSON was malformed (raw kept).
struct ToolCall {
    bool ok = false;
    std::string name;
    json arguments;
    std::string raw;
};

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
            if (tc.arguments.is_string()) {
                fprintf(stderr, "[q27-strict] rejected double-encoded arguments (tool=%s)\n",
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
        tc.ok = !tc.name.empty();
    } catch (...) { tc.ok = false; }
    return tc;
}

inline std::string strip_ws2(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
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
    for (char c : s) {
        switch (c) {
            case '"': esc += "\\\""; break;
            case '\\': esc += "\\\\"; break;
            case '\n': esc += "\\n"; break;
            case '\r': esc += "\\r"; break;
            case '\t': esc += "\\t"; break;
            default: esc += c;
        }
    }
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
    while ((p = text.find("{\"name\":", p)) != std::string::npos) {
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
inline bool recover_raw_value_call(const std::string& text, const json& tools,
                                   std::vector<ToolCall>& out) {
    size_t mo = text.rfind("{\"name\"");
    if (mo == std::string::npos) return false;
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
    // (making that arg a valid sibling). Ordering-independent. The call must
    // be at the end of the model output, which is the UN-RESCUED reality.
    for (const auto& k : strkeys) {
        size_t kp = text.find("\"" + k + "\"", mo);
        if (kp == std::string::npos) continue;
        size_t kc = text.find(':', kp + k.size() + 2);
        if (kc == std::string::npos) continue;
        size_t opener = text.find('"', kc + 1);
        if (opener == std::string::npos) continue;
        for (size_t cand = text.find('"', opener + 1); cand != std::string::npos;
             cand = text.find('"', cand + 1)) {
            // json(str).dump() yields a fully-escaped, quoted JSON string
            const std::string esc = json(text.substr(opener + 1, cand - opener - 1)).dump();
            const std::string recon = text.substr(mo, opener - mo) + esc + text.substr(cand + 1);
            json obj;
            try { obj = json::parse(recon); } catch (...) { continue; }
            if (!obj.is_object() || obj.value("name", std::string()) != nm) continue;
            json args = obj.contains("arguments") && obj["arguments"].is_object()
                            ? obj["arguments"]
                            : json::object();
            ToolCall tc;
            tc.ok = true;
            tc.name = nm;
            tc.arguments = std::move(args);
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
                                                   bool allow_o10 = true) {
    std::vector<ToolCall> out;
    if (tool_strict()) {
        // strict-parser A/B: the wrapper-less recovery chain (drift modes 1-6)
        // is OFF. Log when the text plausibly contained an intended call so the
        // campaign can count suppressed rescues against the tolerant leg.
        if (prefix) *prefix = "";
        if (text_in.find("{\"name\"") != std::string::npos ||
            text_in.find("{\"tool_call\"") != std::string::npos ||
            text_in.find("</content>") != std::string::npos)
            fprintf(stderr, "[q27-strict] SUPPRESSED bare-call rescue: %.200s\n",
                    text_in.c_str());
        return out;
    }
    bool m2 = false, m5 = false, m6 = false, m8 = false; // drift-mode flags (exit-gate catalog)
    // drift mode 9 (2026-07-11, codex-harnessed traffic): the model drops the
    // OPENING quote of the "arguments" key ({"name":"X",\narguments":{...}}).
    // "arguments" is the tool-call schema key, so quoting a bare `arguments":`
    // is unambiguous inside these segments. Applied before segmentation so
    // both the whole-object and truncated-tail parse paths see valid JSON.
    auto fix_arg_quote = [](std::string s) {
        size_t p = 0;
        while ((p = s.find("arguments\"", p)) != std::string::npos) {
            if (p == 0 || s[p - 1] != '"') { s.insert(p, "\""); p += 11; }
            else p += 10;
        }
        return s;
    };
    const std::string text = fix_arg_quote(escape_content_tags(text_in));
    const bool m3 = (text != text_in);              // mode 3: <content>-tagged value rewritten
    const bool m4 = text.find("{\"tool_call\":") != std::string::npos; // mode 4: JSON-keyed opener
    size_t first = std::string::npos;
    size_t i = text.find('{');
    while (i != std::string::npos) {
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
            if (s2) r += '"';
            for (; d2 > 0; d2--) r += '}';
            bool shaped = false;
            try {
                json j = json::parse(r);
                shaped = j.is_object() && j.contains("name") && j.contains("arguments");
            } catch (...) {}
            bool recovered_here = false;
            if (shaped) {
                ToolCall tc = parse_tool_call(r);
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
        const std::string& seg = san;
        bool shaped = false, m6cand = false, m8cand = false;
        json j6, j8;
        try {
            json j = json::parse(seg);
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
                if (first == std::string::npos) first = i;
                out.push_back(std::move(tc));
                m8 = true;
                i = text.find('{', end + 1);
                continue;
            }
        }
        i = text.find('{', i + 1);
    }
    // Fallback: name-dropped mode-6 BATCH (the main scan can't segment it). Only when the
    // standard scan found nothing, so normal calls are never double-counted.
    if (out.empty() && tools && text.find("{\"name\":") != std::string::npos) {
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
    if (out.empty() && allow_o10 && tools && tools->is_array()) {
        for (const auto& t : *tools) {
            std::string nm = t.contains("function")
                                 ? t["function"].value("name", std::string())
                                 : std::string();
            if (nm.empty()) continue;
            const std::string sig = nm + "\", \"";
            size_t p = text.find(sig);
            if (p == std::string::npos) continue;
            if (p > 0 && text[p - 1] == '"') continue; // already `"name": "NAME"`
            std::string synth = text.substr(0, p) + "{\"name\": \"" + text.substr(p);
            if (synth.find('}') == std::string::npos) synth += "}";
            std::string pre2;
            auto rec = parse_bare_tool_calls(synth, &pre2, tools, /*allow_o10=*/false);
            if (!rec.empty()) {
                out = std::move(rec);
                if (prefix) *prefix = pre2;
                fprintf(stderr, "[drift] mode-10 dropped-opener rescue: %s\n", nm.c_str());
                break;
            }
        }
    }
    // mode 11: raw code-body string value (unescaped inner quotes/newlines).
    // Last resort -- runs only when everything above failed. Sets the prefix
    // to the text before the call object so a preamble ("Let me write...")
    // still streams as text.
    if (out.empty() && tools && tools->is_array() &&
        recover_raw_value_call(text, *tools, out) && prefix) {
        size_t mo = text.rfind("{\"name\"");
        *prefix = mo != std::string::npos ? strip_ws2(text.substr(0, mo)) : std::string();
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
    }
    return out;
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
            {"function", {{"name", c.name}, {"arguments", c.arguments.dump()}}}};
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
                                                       {"arguments", c.arguments.dump()}}}}})}};
}

} // namespace q27
