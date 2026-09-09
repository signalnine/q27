// stream_probe: replay a turn's raw text through the SAME streaming path the
// /v1/messages handler uses (StreamSplitter -> StreamToolRouter -> the bare
// chain), chunked like tokens, and print what the client would have seen.
//
// The batch replay (replay_missed_calls) feeds parse_bare_tool_calls the whole
// text; a live turn goes through the holdback, which decides whether the
// parser runs at all. A shape the parser recovers but the holdback never arms
// on streams out as text and ends the session -- and leaves no UN-RESCUED
// line, because nothing was ever parsed (2026-09-08, item 2: three of twelve
// first turns in the DFlash2 production arms died this way with a silent
// journal). This is the instrument that shows the difference.
//
//   ./build/stream_probe tools.json turn.txt [chunk_bytes=4]
//
// tools.json: an OpenAI-shape `tools` array (type/function/name/parameters)
// or an Anthropic one (name/input_schema); both are accepted. Prints one line
// per emitted block: TEXT <bytes> "<head>", TOOL_USE <name> <args>, plus the
// parser's own stderr. Exit 0 when at least one tool_use was emitted.
#include "api_common.h"
#include "stream_split.h"
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s tools.json turn.txt [chunk_bytes]\n", argv[0]); return 2; }
    std::ifstream tf(argv[1]); std::stringstream ts; ts << tf.rdbuf();
    json tools_in = json::parse(ts.str());
    if (tools_in.is_object() && tools_in.contains("tools")) tools_in = tools_in["tools"];
    json tools = json::array();
    for (const auto& t : tools_in) {
        if (t.contains("function")) { tools.push_back(t); continue; }
        json fn; fn["name"] = t.value("name", "");
        fn["parameters"] = t.contains("input_schema") ? t["input_schema"] : json::object();
        tools.push_back({{"type", "function"}, {"function", fn}});
    }
    std::set<std::string> names;
    for (const auto& t : tools) names.insert(t["function"].value("name", ""));
    std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf();
    const std::string text = ss.str();
    const size_t chunk = argc > 3 ? (size_t)atoi(argv[3]) : 4;

    int tool_uses = 0;
    std::string text_acc, think_acc;
    auto flush_text = [&]() {
        if (text_acc.empty()) return;
        std::string head = text_acc.substr(0, 60);
        for (auto& c : head) if (c == '\n') c = ' ';
        printf("TEXT %zu \"%s%s\"\n", text_acc.size(), head.c_str(), text_acc.size() > 60 ? "..." : "");
        text_acc.clear();
    };
    auto emit_text = [&](const std::string& t) { text_acc += t; };
    auto emit_think = [&](const std::string& t) { think_acc += t; };
    auto emit_call = [&](const q27::ToolCall& c) {
        if (!c.ok) return false;
        flush_text();
        printf("TOOL_USE %s %s\n", c.name.c_str(), c.arguments.dump().substr(0, 120).c_str());
        tool_uses++;
        return true;
    };
    auto classify_bare = [&](const std::string& source, bool allow_repair, auto&& visible) {
        std::string pre, residual;
        auto calls = q27::parse_bare_tool_calls(source, &pre, &tools, true, allow_repair, &residual);
        if (calls.empty()) return q27::BareToolCandidateResult{};
        size_t cursor = 0, recovered = 0;
        for (const auto& call : calls) {
            visible(source.substr(cursor, call.source_begin - cursor));
            cursor = call.source_end;
            if (emit_call(call)) recovered++;
            else visible(source.substr(call.source_begin, call.source_end - call.source_begin));
        }
        visible(source.substr(cursor));
        return q27::BareToolCandidateResult{true, recovered != 0};
    };
    q27::StreamToolRouter router;
    router.has_tools = !tools.empty();
    auto emit_tool = [&]() {
        auto c = q27::parse_tool_call(q27::strip_ws2(router.tool_buf));
        router.tool_buf.clear();
        if (!c.ok) { if (!classify_bare(c.raw, true, emit_text)) emit_text(c.raw); }
        else if (!emit_call(c)) emit_text(c.raw);
    };
    q27::StreamSplitter sp;
    auto emit_seg = [&](q27::StreamSplitter::Chan ch, const std::string& t) {
        router.segment(ch, t, false, names, emit_text, emit_think, emit_tool, classify_bare);
    };
    for (size_t i = 0; i < text.size(); i += chunk)
        for (auto& [ch, t] : sp.feed(text.substr(i, chunk))) emit_seg(ch, t);
    for (auto& [ch, t] : sp.flush()) emit_seg(ch, t);
    if (!router.tool_buf.empty()) emit_tool();
    router.finish(true, names, emit_text, emit_think, classify_bare);
    flush_text();
    if (!think_acc.empty()) printf("THINK %zu\n", think_acc.size());
    printf("== tool_use blocks: %d (%zu tools declared)\n", tool_uses, names.size());
    return tool_uses ? 0 : 1;
}
