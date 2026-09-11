// Golden test: q27's renderer must produce, byte for byte, what llama.cpp's
// minja produces from the Qwen3.8 chat template for the same request.
//
// WHY. On 2026-08-22 the two renderers were shown byte-identical for a
// tool-free request, and every remaining difference in an agentic loop lived in
// the tools preamble: nlohmann::json sorts keys so the model saw
// {"function":{"description",...,"name",...},"type"} where the template (and
// the checkpoint's training data) has {"type": "function", "function":
// {"name", "description", "parameters"}}; the instruction text was a
// paraphrase that dropped the template's own "must be nested within
// <tool_call>" reminder -- the rule every drift shape of that week broke; and
// tool results kept a trailing newline the template trims.
//
// The golden is NOT a jinja2 render. It was captured from a running
// llama-server via /apply-template (commit log has the invocation), so it is
// what llama.cpp actually feeds the model. Regenerate it the same way if the
// template or the fixture changes; never edit it by hand.
#include "api_common.h"
#include <cstdio>
#include <fstream>
#include <sstream>

using json = nlohmann::json;

static std::string slurp(const char* p) {
    std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str();
}


static int fails = 0;
static void ok(bool c, const char* what) { printf("  %-66s %s\n", what, c ? "PASS" : "FAIL"); if (!c) fails++; }

// Boundaries the golden does not exercise.
static void boundaries() {
    using q27::anthropic_tools_decl;
    // spacing + escaping must match Python json.dumps(ensure_ascii=False): the
    // expected string below was produced by exactly that call.
    {
        nlohmann::ordered_json v = nlohmann::ordered_json::parse(R"JSON({"type": "function", "function": {"name": "Écrire", "description": "tab\there \"q\" / slash \\ back é 中", "parameters": {"type": "object", "properties": {"z": {"type": "string", "enum": ["a", "b"]}, "a": {"type": "integer"}}, "required": ["z"], "x": [1, 2.5, true, null, {}]}}})JSON");
        ok(q27::ordered_dump_spaced(v) == R"JSON({"type": "function", "function": {"name": "Écrire", "description": "tab\there \"q\" / slash \\ back é 中", "parameters": {"type": "object", "properties": {"z": {"type": "string", "enum": ["a", "b"]}, "a": {"type": "integer"}}, "required": ["z"], "x": [1, 2.5, true, null, {}]}}})JSON",
           "ordered_dump_spaced: unicode, escapes, nested, mixed scalars == json.dumps");
    }
    // malformed entries are skipped the way anthropic_tools_json skips them
    {
        const std::string raw = R"({"tools":[{"description":"nameless"},{"name":123},{"name":""},
            {"name":"ok","description":7,"input_schema":"bad"},{"name":"z2","input_schema":{"b":1,"a":2}}]})";
        const std::string d = anthropic_tools_decl(raw);
        ok(d.find("nameless") == std::string::npos && d.find("123") == std::string::npos,
           "decl: nameless / non-string / empty names are skipped");
        ok(d.find(R"({"type": "function", "function": {"name": "ok", "description": "", "parameters": {}}})") != std::string::npos,
           "decl: bad description -> \"\", bad input_schema -> {}");
        ok(d.find(R"("parameters": {"b": 1, "a": 2})") != std::string::npos,
           "decl: client key order preserved (b before a)");
    }
    // tool_choice subset: keep restricts without reordering
    {
        const std::string raw = R"({"tools":[{"name":"A"},{"name":"B"},{"name":"C"}]})";
        std::vector<std::string> keep = {"C", "A"};
        const std::string d = anthropic_tools_decl(raw, &keep);
        ok(d.find("\"B\"") == std::string::npos && d.find("\"A\"") < d.find("\"C\""),
           "decl: keep filters to the selection and keeps client order");
    }
    // control characters in a description are stripped like the legacy dump
    {
        const std::string raw = "{\"tools\":[{\"name\":\"n\",\"description\":\"a\\u0007b\"}]}";
        const std::string d = anthropic_tools_decl(raw);
        ok(d.find('\x07') == std::string::npos, "decl: control bytes stripped (strip_ctrl)");
    }
    // garbage in -> empty out, never a throw
    ok(anthropic_tools_decl("not json").empty() && anthropic_tools_decl("{}").empty() &&
           anthropic_tools_decl(R"({"tools":"x"})").empty(),
       "decl: unparseable / no tools / wrong type -> empty");
    // legacy fallback: no decl -> the sorted dump still renders the block
    {
        json tools = json::array({{{"type","function"},{"function",{{"name","Read"},{"description","d"},{"parameters",json::object()}}}}});
        const std::string pre = q27::tools_preamble(tools);
        ok(pre.find("\"name\":\"Read\"") != std::string::npos && pre.find("<tools>") != std::string::npos,
           "tools_preamble: empty decl falls back to the sorted dump");
    }
    // tool results are trimmed like the template's |trim
    ok(q27::tool_response_text("index.ts\n") == "<tool_response>\nindex.ts\n</tool_response>" &&
           q27::tool_response_text("  x  ") == "<tool_response>\nx\n</tool_response>" &&
           q27::tool_response_text("\n\n") == "<tool_response>\n\n</tool_response>",
       "tool_response_text: trailing/leading whitespace trimmed, empty stays empty");
}


// The OpenAI path, same conversation, tools already in the trained shape. The
// fixture deliberately gives Read's `parameters` as {type, required, properties}
// -- an order nlohmann::json would rewrite -- so the golden proves the client's
// order reaches the model.
static int openai_golden() {
    const std::string raw = slurp("tools/golden/qwen38_tools_request.openai.json");
    const std::string want = slurp("tools/golden/qwen38_tools_request.openai.prompt");
    if (raw.empty() || want.empty()) { fprintf(stderr, "openai fixture missing\n"); return 2; }
    json body = json::parse(raw);
    q27::tool_dialect_xml_default() = true;
    q27::ToolChoice tchoice = q27::parse_tool_choice(body);
    q27::OpenAIToolSelection selected = q27::select_openai_tools(body, tchoice);
    q27::TemplateOpts opts = q27::template_opts_from_body(body);
    opts.tools_decl = q27::openai_tools_decl(raw, &selected.names);
    const std::string got = q27::chatml_prompt(q27::openai_msgs(body), selected.tools, true,
                                               nullptr, nullptr, {}, nullptr, &opts);
    if (got == want) { printf("openai golden: PASS (%zu bytes)\n", got.size()); return 0; }
    size_t i = 0; while (i < got.size() && i < want.size() && got[i] == want[i]) i++;
    printf("openai golden: FAIL at byte %zu (got %zu, want %zu)\n", i, got.size(), want.size());
    return 1;
}

static void openai_boundaries() {
    using q27::openai_tools_decl;
    {   // pass-through: extra keys and client order survive, spacing is minja's
        const std::string raw = R"({"tools":[{"type":"function","function":{"name":"A","parameters":{"b":1,"a":2},"x-extra":true}}]})";
        ok(openai_tools_decl(raw) == "\n" R"({"type": "function", "function": {"name": "A", "parameters": {"b": 1, "a": 2}, "x-extra": true}})",
           "openai decl: entry passed through in client order with extra keys");
    }
    {   // malformed entries skipped the way select_openai_tools skips them
        const std::string raw = R"({"tools":["str",7,{"type":"function"},{"type":"function","function":{"name":""}},{"type":"function","function":{"name":"ok"}}]})";
        const std::string d = openai_tools_decl(raw);
        ok(d.find("\"ok\"") != std::string::npos && d.find("\"str\"") == std::string::npos &&
               d.find("\"name\": \"\"") == std::string::npos,
           "openai decl: non-objects, nameless and empty-name entries skipped");
    }
    {   // keep restricts without reordering
        const std::string raw = R"({"tools":[{"type":"function","function":{"name":"A"}},{"type":"function","function":{"name":"B"}},{"type":"function","function":{"name":"C"}}]})";
        std::vector<std::string> keep = {"C", "A"};
        const std::string d = openai_tools_decl(raw, &keep);
        ok(d.find("\"B\"") == std::string::npos && d.find("\"A\"") < d.find("\"C\""),
           "openai decl: keep filters to the selection, client order kept");
    }
    ok(openai_tools_decl("nope").empty() && openai_tools_decl(R"({"tools":{}})").empty(),
       "openai decl: garbage / wrong type -> empty");
}

// One Anthropic fixture against its llama.cpp capture; prints the first
// differing byte with context so a failure is diagnosable.
static int anthropic_golden(const char* tag, const char* fixture, const char* prompt) {
    const std::string raw = slurp(fixture);
    const std::string want = slurp(prompt);
    if (raw.empty() || want.empty()) { fprintf(stderr, "fixture missing (run from repo root)\n"); return 2; }
    json body = json::parse(raw);
    q27::tool_dialect_xml_default() = true;                 // a Qwen3.8 checkpoint boots XML
    q27::TemplateOpts opts = q27::template_opts_from_body(body);
    opts.tools_decl = q27::anthropic_tools_decl(raw);        // ordered, minja-spaced
    const json tools = q27::anthropic_tools_json(body);
    const std::string got = q27::chatml_prompt(q27::anthropic_msgs(body, &raw), tools, /*think=*/true,
                                               nullptr, nullptr, {}, nullptr, &opts);
    if (got == want) { printf("%s: PASS (%zu bytes)\n", tag, got.size()); return 0; }
    size_t i = 0; while (i < got.size() && i < want.size() && got[i] == want[i]) i++;
    printf("%s: FAIL at byte %zu (got %zu bytes, want %zu)\n", tag, i, got.size(), want.size());
    auto show = [&](const char* t, const std::string& s) {
        size_t a = i > 60 ? i - 60 : 0, b = std::min(s.size(), i + 80);
        printf("  %s: %s\n", t, json(s.substr(a, b - a)).dump().c_str());
    };
    show("got ", got); show("want", want);
    return 1;
}

// The 3.8 history rules (2026-09-10) on the pieces the first golden never
// had: thinking and text with the edges a q27 response really carries
// ("...\n" / "\n\n...\n\n"), text before a call, two calls in one turn, an
// Edit whose client key order is NOT alphabetical, non-string arguments
// (bool, list, object), a mid-conversation system message (Claude Code 2.1.x
// sends them; the capture template renders them inline, as ninfer does), a
// text-only assistant turn, and user text with a trailing newline.
static void history_boundaries() {
    q27::tool_dialect_xml_default() = true;
    ok(q27::trim_ws(" \n\tx y\n\n") == "x y" && q27::trim_ws("\n\n").empty() && q27::trim_ws("").empty(),
       "trim_ws: both edges, all-whitespace -> empty");
    // Python's str.strip set (jinja2 under transformers), not ASCII only
    ok(q27::trim_ws(" 　x y \x1c ") == "x y" &&
           q27::trim_ws("​x​") == "​x​",
       "trim_ws: NBSP/ideographic/U+2028/U+001C trimmed, interior kept, ZWSP is not space");
    ok(q27::tool_response_text(" out　") == "<tool_response>\nout\n</tool_response>",
       "tool_response_text: same strip set");
    ok(q27::assistant_content_38("\n\nI'll read.\n\n", {"<tool_call>A</tool_call>"}) ==
           "I'll read.\n\n<tool_call>A</tool_call>",
       "assistant_content_38: trimmed text, \\n\\n before the first call");
    ok(q27::assistant_content_38("  ", {"<tool_call>A</tool_call>", "<tool_call>B</tool_call>"}) ==
           "<tool_call>A</tool_call>\n<tool_call>B</tool_call>",
       "assistant_content_38: no text -> call first, \\n between calls");
    {   // OpenAI path: arguments string order and spacing reach the prompt
        json body = json::parse(R"({"messages":[{"role":"user","content":"u"},{"role":"assistant","content":"\n\nok\n","reasoning_content":"r\n","tool_calls":[{"type":"function","function":{"name":"Edit","arguments":"{\"old_string\":\"a\",\"new_string\":\"b\",\"n\":[1,2]}"}}]}]})");
        auto msgs = q27::openai_msgs(body);
        const std::string& c = msgs.back().content;
        ok(c.rfind("ok\n\n<tool_call>\n<function=Edit>\n<parameter=old_string>\na\n</parameter>\n<parameter=new_string>\nb\n</parameter>\n<parameter=n>\n[1, 2]\n</parameter>", 0) == 0,
           "openai_msgs: trimmed text, client arg order, spaced list");
        const std::string p = q27::chatml_prompt(msgs, json::array(), true);
        ok(p.find("<think>\nr\n</think>\n\nok\n\n<tool_call>") != std::string::npos,
           "chatml_prompt: reasoning trimmed, no doubled newlines");
    }
    {   // without the raw body the order falls back to sorted, spacing still template
        const std::string raw = R"({"messages":[{"role":"user","content":"u"},{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"E","input":{"z":1,"a":[1,2]}}]}]})";
        const std::string p1 = q27::chatml_prompt(q27::anthropic_msgs(json::parse(raw), &raw), json::array(), true);
        const std::string p0 = q27::chatml_prompt(q27::anthropic_msgs(json::parse(raw)), json::array(), true);
        ok(p1.find("<parameter=z>") < p1.find("<parameter=a>") && p1.find("[1, 2]") != std::string::npos,
           "anthropic_msgs(raw): client order z before a, spaced list");
        ok(p0.find("<parameter=a>") < p0.find("<parameter=z>") && p0.find("[1, 2]") != std::string::npos,
           "anthropic_msgs(no raw): sorted fallback, spacing unchanged");
    }
    {   // the JSON dialect (3.6 family) keeps its legacy flattening byte for byte
        q27::tool_dialect_xml_default() = false;
        const std::string raw = R"({"messages":[{"role":"user","content":"u\n"},{"role":"assistant","content":[{"type":"thinking","thinking":"r\n"},{"type":"text","text":"t"},{"type":"tool_use","id":"t","name":"E","input":{"a":1}}]}]})";
        const std::string p = q27::chatml_prompt(q27::anthropic_msgs(json::parse(raw), &raw), json::array(), true);
        ok(p.find("<think>\nr\n\n</think>\n\nt\n<tool_call>\n{\"name\": \"E\", \"arguments\": {\"a\":1}}") != std::string::npos &&
               p.find("u\n<|im_end|>") != std::string::npos,
           "json dialect: legacy render unchanged (no trim, one \\n, compact args)");
        q27::tool_dialect_xml_default() = true;
    }
}

int main() {
    boundaries();
    openai_boundaries();
    history_boundaries();
    if (fails) { printf("template golden: %d boundary FAILURE(S)\n", fails); return 1; }
    int rc = anthropic_golden("template golden", "tools/golden/qwen38_tools_request.anthropic.json",
                              "tools/golden/qwen38_tools_request.prompt");
    if (rc) return rc;
    rc = anthropic_golden("history golden", "tools/golden/qwen38_history_request.anthropic.json",
                          "tools/golden/qwen38_history_request.prompt");
    if (rc) return rc;
    return openai_golden();
}
