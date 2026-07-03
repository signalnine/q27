// Shared prompt construction + tool-call parsing for the q27 API endpoints.
// Qwopus (qwen35) tool protocol, from the GGUF chat template:
//   system preamble lists tools as JSON inside <tools>...</tools>
//   model emits  <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
//   results go back as user content wrapped in <tool_response>...</tool_response>
#pragma once
#include <string>
#include <vector>

#include "../third_party/json.hpp"
#include "stream_split.h"

namespace q27 {
using json = nlohmann::json;

struct Msg {
    std::string role;     // system | user | assistant
    std::string content;  // flattened text (think blocks already reconstructed)
};

// Tools preamble, verbatim structure from the chat template. `tools` entries
// must already be in {"type":"function","function":{...}} shape.
inline std::string tools_preamble(const json& tools) {
    std::string s = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (auto& t : tools) s += "\n" + t.dump();
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
inline std::string chatml_prompt(const std::vector<Msg>& msgs, const json& tools,
                                 bool think = true) {
    std::string p;
    size_t start = 0;
    std::string sys;
    if (!msgs.empty() && msgs[0].role == "system") { sys = msgs[0].content; start = 1; }
    if (tools.is_array() && !tools.empty()) {
        p += "<|im_start|>system\n" + tools_preamble(tools);
        if (!sys.empty()) p += "\n\n" + sys;
        p += "<|im_end|>\n";
    } else if (!sys.empty()) {
        p += "<|im_start|>system\n" + sys + "<|im_end|>\n";
    }
    for (size_t i = start; i < msgs.size(); i++)
        p += "<|im_start|>" + msgs[i].role + "\n" + msgs[i].content + "<|im_end|>\n";
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

// Parsed model tool call. `ok` false if the JSON was malformed (raw kept).
struct ToolCall {
    bool ok = false;
    std::string name;
    json arguments;
    std::string raw;
};

inline ToolCall parse_tool_call(const std::string& seg) {
    ToolCall tc;
    tc.raw = seg;
    try {
        json j = json::parse(seg);
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
inline ToolCall parse_bare_tool_call(const std::string& text, std::string* prefix,
                                     std::string* suffix) {
    ToolCall none;
    for (size_t i = text.find('{'); i != std::string::npos; i = text.find('{', i + 1)) {
        int depth = 0;
        bool in_str = false, esc = false;
        size_t end = std::string::npos;
        for (size_t j = i; j < text.size(); j++) {
            char ch = text[j];
            if (esc) { esc = false; continue; }
            if (in_str) {
                if (ch == '\\') esc = true;
                else if (ch == '"') in_str = false;
                continue;
            }
            if (ch == '"') in_str = true;
            else if (ch == '{') depth++;
            else if (ch == '}' && --depth == 0) { end = j; break; }
        }
        if (end == std::string::npos) {
            // Unbalanced to EOF. Second observed mode: the model emits the
            // full call but never closes the JSON -- the argument string ends
            // in tag junk ("</file>") instead of "}} . Repair FRAMING only:
            // strip trailing junk tags, close the open string, close braces.
            // Payload truncation is not repaired (and should stay penalized).
            std::string cand = text.substr(i);
            if (cand.rfind("{\"name\"", 0) != 0 &&
                strip_ws2(cand).rfind("{\"name\"", 0) != 0)
                break;
            std::string r = cand;
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
            try {
                json j = json::parse(r);
                if (!j.is_object() || !j.contains("name") || !j.contains("arguments")) break;
            } catch (...) { break; }
            ToolCall tc = parse_tool_call(r);
            if (!tc.ok) break;
            if (prefix) *prefix = strip_ws2(text.substr(0, i));
            if (suffix) *suffix = "";
            return tc;
        }
        std::string seg = text.substr(i, end - i + 1);
        try {
            json j = json::parse(seg);
            if (!j.is_object() || !j.contains("name") || !j.contains("arguments")) continue;
        } catch (...) { continue; }
        ToolCall tc = parse_tool_call(seg);
        if (!tc.ok) continue;
        if (prefix) *prefix = strip_ws2(text.substr(0, i));
        if (suffix) *suffix = text.substr(end + 1);
        return tc;
    }
    return none;
}


} // namespace q27
