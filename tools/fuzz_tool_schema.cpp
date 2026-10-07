// Fuzz the tool-call parser with the TOOL SCHEMA varying alongside the model
// output. tools/fuzz_tool_parser.cpp pins three well-formed tools, which is why
// it never reached the 2026-10-04 (BUILDLOG (ba)) crash: drift modes 11/22/23
// read the declared parameter types, and an MCP-style `"type":["string","null"]`
// or `"parameters":null` threw out of the streaming provider. Here the input
// is "<tools JSON>\n@@\n<model bytes>": the head goes through
// anthropic_tools_json (a throw there is the request's own 400 and is fine),
// and whatever it accepts drives every schema-reading entry point over the
// tail. Also walks the JSON tool grammar over the tail with the fuzzed names.
#include "api_common.h"
#include "toolgram.h"
#include <cstdint>
#include <string>
using json = nlohmann::json;

static const char* kSep = "\n@@\n";

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size > 64u << 10) return 0;
    const std::string in((const char*)data, size);
    const size_t cut = in.find(kSep);
    if (cut == std::string::npos) return 0;
    const std::string head = in.substr(0, cut), s = in.substr(cut + 4);
    json tools;
    try {
        json body = q27::parse_request_body(head.size() && head[0] == '{' ? head : "{\"tools\":" + head + "}");
        tools = q27::anthropic_tools_json(body);
    } catch (const std::exception&) { return 0; }  // the request path 400s here
    if (!tools.is_array()) return 0;
    std::set<std::string> names;
    std::vector<std::string> name_list;
    for (const auto& t : tools)
        if (t.is_object() && t.contains("name") && t["name"].is_string()) {
            names.insert(t["name"].get<std::string>());
            name_list.push_back(t["name"].get<std::string>());
        }
    std::string prefix, residual;
    for (bool repair : {false, true}) {
        auto calls = q27::parse_bare_tool_calls(s, &prefix, &tools, true, repair, &residual);
        for (const auto& c : calls)
            if (c.source_begin != std::string::npos &&
                (c.source_begin > s.size() || c.source_end > s.size() || c.source_end < c.source_begin))
                __builtin_trap();
    }
    {
        q27::StreamSplitter sp;
        std::vector<std::pair<q27::StreamSplitter::Chan, std::string>> segs;
        auto add = [&](auto& x) {
            if (!segs.empty() && segs.back().first == x.first) segs.back().second += x.second;
            else segs.push_back(x);
        };
        for (size_t i = 0; i < s.size(); i += 7) for (auto& x : sp.feed(s.substr(i, 7))) add(x);
        for (auto& x : sp.flush()) add(x);
        q27::resolve_ordered_tool_segments(segs, &tools, true, [](const std::string&, size_t) { return true; });
    }
    {
        q27::BareToolTextHoldback hb;
        std::string vis;
        auto emit = [&](const std::string& t) { vis += t; };
        auto classify = [&](const std::string& src, bool rep, auto&& visible) {
            std::string p, r;
            auto cs = q27::parse_bare_tool_calls(src, &p, &tools, true, rep, &r);
            if (cs.empty()) return q27::BareToolCandidateResult{};
            size_t cur = 0;
            for (auto& c : cs) {
                if (c.source_begin == std::string::npos || c.source_begin < cur || c.source_end > src.size()) break;
                visible(src.substr(cur, c.source_begin - cur));
                cur = c.source_end;
            }
            visible(src.substr(cur));
            return q27::BareToolCandidateResult{true, true};
        };
        for (size_t i = 0; i < s.size(); i += 5) hb.route(s.substr(i, 5), names, emit, classify);
        hb.finish(true, names, emit, classify);
    }
    q27::recover_unclosed_tool_tail(s, &tools, [](const std::string&) {}, [](const q27::ToolCall&) { return true; });
    {
        q27::ToolGrammar g;
        g.reset(name_list);
        for (char c : s) if (!g.advance(c)) break;
    }
    return 0;
}
