// CPU-only regression test for the bare-tool-call drift recoveries in
// api_common.h (parse_bare_tool_calls). Covers the modes that have bitten
// real Claude Code sessions:
//   mode 10 -- dropped `{"name": "` opener (issue: flask-5014 early quit)
//   mode 11 -- raw code-body string value, unescaped inner quotes (issue #4)
// plus the negatives (prose must not false-recover, well-formed calls take
// the normal path).
//
// Build + run (no CUDA needed):
//   g++ -std=c++17 -I src tools/test_tool_drift.cpp -o build/test_tool_drift && ./build/test_tool_drift
#include "api_common.h"
#include <cstdio>
#include <string>

using json = nlohmann::json;

static int failures = 0;
static void ok(bool cond, const char* name) {
    printf("  %s %s\n", cond ? "PASS" : "FAIL", name);
    if (!cond) failures++;
}

static json tool(const char* name, std::vector<std::pair<std::string, bool>> params) {
    // params: (key, is_required); all typed string for these tests
    json props = json::object(), req = json::array();
    for (auto& p : params) {
        props[p.first] = {{"type", "string"}};
        if (p.second) req.push_back(p.first);
    }
    return {{"type", "function"},
            {"function",
             {{"name", name},
              {"parameters", {{"type", "object"}, {"properties", props}, {"required", req}}}}}};
}

// Drift mode 13 (2026-07-24, found live by the grammar-engage probe): a
// wrapper-less call truncated INSIDE an escape sequence. The repair used to
// append the closing quote straight after a dangling backslash, which escaped
// it -- string still open, object never parsed, call UN-RESCUED. The real
// payload was a Write whose markdown content was cut while writing an escaped
// JSON example (`\\"role\\": \\"assistant\\",\\`).
// Drift mode 14 (2026-08-14, captured live from a thunderdome run on the
// Qwen3.8-27B q5f repack, bench-time-tracker trial-1). The model was told to
// emit `<tool_call>\n{"name":..., "arguments":{...}}\n</tool_call>` by
// tools_preamble and DID so correctly on turn 1, then drifted on turn 2 into
// its chat-template's XML dialect with no <tool_call> wrapper at all:
//
//   <tool_name>Read</tool_name>
//   <parameter=file_path>/workspace/tests/index.test.ts</parameter>
//   <tool_name>Bash</tool_name>
//   <parameter=arguments>
//   {"command":"...","description":"..."}}
//
// Three things make this its own mode rather than a variant of 10/11:
//   1. TWO calls are concatenated in one assistant block.
//   2. The conventions DISAGREE between them -- Read uses a scalar
//      <parameter=KEY>VALUE</parameter>, Bash uses <parameter=arguments>
//      wrapping a whole JSON object and never closes the tag.
//   3. The Bash payload carries one unbalanced trailing brace.
//
// Claude Code saw plain text, made no tool call, and the trial ended at
// num_turns=2 scoring 0.000 -- the same shape as the pre-fix 3.6 agentic
// ceiling, which was parser-bound rather than quality-bound.
//
// NOT YET RESCUED. This fixture pins the observed bytes and the CURRENT
// behaviour so a future parser change has a real target and a regression
// witness. Flip `expect_rescued` to true in the same commit that teaches
// parse_bare_tool_calls this dialect.
// Drift mode 15 (2026-08-14, Qwen3.8-27B): surfaced by re-running the same
// thunderdome task once mode 14 stopped the earlier stall. An unclosed <name>
// pseudo-tag, then a bare identifier and JSON args with no opening brace or
// quote, plus the same trailing unbalanced brace seen in mode 14.
static void test_mode15_name_tag_bare_args() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command"]}}}])");
    // Verbatim bytes from the captured transcript.
    std::string t =
        "<name>Bash, \"arguments\": {\"command\":\"ls /workspace/tests /workspace/src && cat "
        "/workspace/.eslintrc.cjs /workspace/vitest.config.ts\",\"description\":\"List tests "
        "and src, show eslint and vitest config\"}}";
    std::string pre;
    auto v = q27::parse_bare_tool_calls(t, &pre, &tools);
    ok(v.size() == 1 && v[0].ok && v[0].name == "Bash",
       "mode15: <name> pseudo-tag with bare args recovered");
    if (v.size() == 1)
        ok(v[0].arguments.value("description", std::string()) ==
               "List tests and src, show eslint and vitest config",
           "mode15: trailing unbalanced brace tolerated, args intact");
    // An undeclared name must NOT be rescued.
    std::string bad = "<name>NotATool, \"arguments\": {\"x\":1}}";
    auto v2 = q27::parse_bare_tool_calls(bad, &pre, &tools);
    ok(v2.empty(), "mode15: undeclared name is rejected");
}

// Drift mode 16 and the mode-15 attribute variant (2026-08-14), both harvested
// from the same batch of Qwen3.8 thunderdome runs. Also pins the one form that
// must NEVER be rescued.
static void test_mode16_and_15_variants() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command"]}}},{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}}])");
    std::string pre;

    // mode 15 variant: attribute spelling <name="Read", ...
    std::string attr = "<name=\"Read\", \"arguments\": {\"file_path\": \"/workspace/tests/tracker.test.ts\"}}";
    auto v1 = q27::parse_bare_tool_calls(attr, &pre, &tools);
    ok(v1.size() == 1 && v1[0].ok && v1[0].name == "Read" &&
           v1[0].arguments.value("file_path", std::string()) ==
               "/workspace/tests/tracker.test.ts",
       "mode15-attr: <name=\"X\" spelling recovered");

    // mode 16: correct JSON, wrong wrapper.
    std::string fn = "I'll start by exploring the workspace.\n\n<function>\n"
        "{\"name\": \"Bash\", \"arguments\": {\"command\": \"ls -la /workspace\", "
        "\"description\": \"List workspace\"}}";
    auto v2 = q27::parse_bare_tool_calls(fn, &pre, &tools);
    ok(v2.size() == 1 && v2[0].ok && v2[0].name == "Bash" &&
           v2[0].arguments.value("command", std::string()) == "ls -la /workspace",
       "mode16: <function>-wrapped JSON recovered");
    ok(pre.find("exploring the workspace") != std::string::npos,
       "mode16: prose before the wrapper is preserved as prefix");

    // MUST NOT RESCUE: a hallucinated tool RESULT, not a call. Rescuing this
    // would feed invented command output back as though a tool had run.
    std::string halluc =
        "I'll start by exploring the codebase.\n\n<tool_calls>\n<result>\n"
        "<name>Bash</name>\n<output>total 40\ndrwxr-xr-x 1 node node 4096 .\n</output>\n"
        "</result>\n</tool_calls>";
    auto v3 = q27::parse_bare_tool_calls(halluc, &pre, &tools);
    ok(v3.empty(), "hallucinated <result>/<output> block is NOT rescued as a call");

    // ...but the rule is ENCLOSURE, not mere presence (2026-08-21). A result
    // block that opened AND closed before the call is prior context -- a
    // quoted result from an earlier step -- and must not veto the real call
    // that follows it. The old whole-string guard threw this away.
    std::string after_closed_output =
        "The last command printed:\n<output>total 0</output>\n\nNow listing again.\n\n"
        "<function>\n{\"name\": \"Bash\", \"arguments\": {\"command\": \"ls -la\"}}";
    auto v4 = q27::parse_bare_tool_calls(after_closed_output, &pre, &tools);
    ok(v4.size() == 1 && v4[0].name == "Bash" &&
           v4[0].arguments.value("command", std::string()) == "ls -la",
       "mode16: a CLOSED <output> block before the wrapper does not kill the call");
    ok(pre.find("total 0") != std::string::npos,
       "mode16: the quoted result stays in the visible prefix");

    // and the converse: an UNCLOSED <result> enclosing the wrapper is an
    // invented result block, so the JSON inside it stays text.
    std::string enclosed =
        "<result>\n<function>\n{\"name\": \"Bash\", \"arguments\": {\"command\": \"ls\"}}";
    auto v5 = q27::parse_bare_tool_calls(enclosed, &pre, &tools);
    ok(v5.empty(), "mode16: an unclosed <result> enclosing the wrapper refuses");
}

// The enclosure rule itself (hallucinated_result_around), independent of any
// one drift mode: closed-before-span is context, unclosed-before-span or
// in-span is invented output.
static void test_hallucinated_result_enclosure_rule() {
    using q27::hallucinated_result_around;
    const std::string closed_before = "<output>x</output>\nCALL";
    ok(!hallucinated_result_around(closed_before, closed_before.find("CALL"),
                                   closed_before.size()),
       "enclosure: closed block before the span is prior context");

    const std::string open_before = "<result>\nCALL";
    ok(hallucinated_result_around(open_before, open_before.find("CALL"),
                                  open_before.size()),
       "enclosure: unclosed block before the span encloses it");

    const std::string in_span = "CALL <result>fake</result>";
    ok(hallucinated_result_around(in_span, 0, in_span.size()),
       "enclosure: a block inside the span is invented output");

    // closed, then reopened: still enclosed (depth counting, not last-open)
    const std::string reopened = "<output>a</output>\n<output>\nCALL";
    ok(hallucinated_result_around(reopened, reopened.rfind("CALL"), reopened.size()),
       "enclosure: closed-then-reopened still reads as enclosed");

    // nested same-tag pairs fully closed before the span
    const std::string nested_closed = "<result><result>a</result></result>CALL";
    ok(!hallucinated_result_around(nested_closed, nested_closed.find("CALL"),
                                   nested_closed.size()),
       "enclosure: fully-closed nested pairs before the span are context");

    // no tags at all, and the npos guard
    ok(!hallucinated_result_around("plain text", 0, 10),
       "enclosure: text with no result tags is clean");
    ok(!hallucinated_result_around("<result>x", std::string::npos, std::string::npos),
       "enclosure: npos span begin is not flagged");
}

// Native XML dialect (2026-08-14): the format Qwen3.8's chat template trains.
// Not a drift mode -- the first-class wrapped-body format, parsed by
// parse_tool_call via parse_native_xml_call. Round-trips the template's own
// example shape, typed values, multi-line values, zero-parameter calls, and
// refuses a truncated parameter rather than guessing.
// Per-model dialect default (keyed on general.name, normalized because
// conversion mangles it -- the real 3.8 artifact says "Qwen38 27b Hf").
// Think-mode drift (2026-08-14): bare native dialect without the wrapper, and
// the JSON-head/XML-params chimera that killed bench-time-tracker at turn 4.
// Full-suite capture (2026-08-15): <function=NAME> opener with mode-14-style
// <parameter=arguments> wrapping the whole JSON, streamed with NO closers.
// Fell between mode 14 (wrong opener) and the bare-native rescue (demanded
// closers) and zeroed the first four suite tasks.
// Replicates the server.cu response-path consumer walk (cursor ->
// source_begin/source_end -> substr). A call returned with an unset span
// (npos defaults) passes name/args checks but throws std::out_of_range in
// that walk -- the 2026-08-15 mid-suite server abort (drift mode 17, rid=61)
// -- so every rescue fixture must walk clean here.
static bool spans_walkable(const std::string& text,
                           const std::vector<q27::ToolCall>& calls) {
    size_t cursor = 0;
    for (const auto& c : calls) {
        if (c.source_begin == std::string::npos ||
            c.source_end == std::string::npos ||
            c.source_begin < cursor || c.source_end < c.source_begin ||
            c.source_end > text.size())
            return false;
        (void)text.substr(cursor, c.source_begin - cursor);
        cursor = c.source_end;
    }
    (void)text.substr(cursor);
    return true;
}

static void test_function_arguments_unterminated() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command"]}}}])");
    std::string pre;
    const std::string in_suite =
        "\n\n<function=Bash>\n<parameter=arguments>\n"
        "{\"command\":\"cat /workspace/schema.sql /workspace/TASK.md\","
        "\"description\":\"Show schema and task\"}";
    auto v = q27::parse_bare_tool_calls(in_suite, &pre, &tools);
    ok(v.size() == 1 && v[0].ok && v[0].name == "Bash" &&
           v[0].arguments.value("command", std::string()).rfind("cat /workspace", 0) == 0 &&
           v[0].arguments.value("description", std::string()) == "Show schema and task" &&
           !v[0].arguments.contains("arguments"),
       "suite-form: unterminated <parameter=arguments> merged, not nested");
    ok(v.size() == 1 && spans_walkable(in_suite, v),
       "suite-form: source span set and consumer-walkable");
    // an unterminated NON-final parameter (another <parameter= follows) still refuses
    q27::ToolCall t2;
    ok(!q27::parse_native_xml_call(
           "<function=Bash>\n<parameter=command>\nls\n<parameter=description>\nx\n</parameter>\n</function>", t2),
       "suite-form: unterminated mid-stream parameter still refused");
}

static void test_think_mode_drift() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Write","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}},{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}}])");
    std::string pre;
    // bare <function=NAME>, no <tool_call> wrapper (task-queue stray blocks)
    const std::string in_bare =
        "\n<function=Read>\n<parameter=file_path>\n/workspace/tests/phase-06.test.ts\n</parameter>\n</function>\n";
    auto v1 = q27::parse_bare_tool_calls(in_bare, &pre, &tools);
    ok(v1.size() == 1 && v1[0].ok && v1[0].name == "Read" &&
           v1[0].arguments.value("file_path", std::string()) ==
               "/workspace/tests/phase-06.test.ts",
       "bare native dialect without wrapper recovered");
    ok(v1.size() == 1 && spans_walkable(in_bare, v1) &&
           v1[0].source_end == in_bare.find("</function>") + 11,
       "bare native: source span ends at the closer");
    // the chimera, verbatim shape from the archived transcript
    const std::string in_chim =
        "The test suite is clear. Writing the implementation:\n\n"
        "{\"name\": \"Write\",\n<parameter=file_path>\n/workspace/src/index.ts\n</parameter>\n"
        "<parameter=content>\nimport { existsSync, mkdirSync } from 'fs';\nconst x = 1;\n</parameter>";
    auto v2 = q27::parse_bare_tool_calls(in_chim, &pre, &tools);
    ok(v2.size() == 1 && v2[0].ok && v2[0].name == "Write" &&
           v2[0].arguments.value("file_path", std::string()) == "/workspace/src/index.ts" &&
           v2[0].arguments.value("content", std::string()).rfind("import { existsSync", 0) == 0,
       "mode17: json-head/xml-params chimera recovered");
    ok(pre.find("Writing the implementation") != std::string::npos,
       "mode17: prose before the chimera preserved as prefix");
    ok(v2.size() == 1 && spans_walkable(in_chim, v2) &&
           v2[0].source_begin == in_chim.find("{\"name\""),
       "mode17: source span starts at the json head (rid=61 abort regression)");
    // an UNDECLARED chimera name stays text
    auto v3 = q27::parse_bare_tool_calls(
        "{\"name\": \"NotATool\",\n<parameter=x>\n1\n</parameter>", &pre, &tools);
    ok(v3.empty(), "mode17: undeclared chimera name is not rescued");
}

static void test_dialect_default_keying() {
    unsetenv("Q27_TOOL_DIALECT");
    auto meta = [](const char* n) { return std::string("{\"general.name\": \"") + n + "\"}"; };
    q27::set_tool_dialect_for_model(meta("Qwen38 27b Hf"));
    ok(q27::tool_dialect_xml(), "dialect: mangled 3.8 name selects xml");
    q27::set_tool_dialect_for_model(meta("Qwen3.8-27B"));
    ok(q27::tool_dialect_xml(), "dialect: clean 3.8 name selects xml");
    q27::set_tool_dialect_for_model(meta("Qwen3.6-27B"));
    ok(!q27::tool_dialect_xml(), "dialect: 3.6 stays json");
    q27::set_tool_dialect_for_model(meta("Qwopus3.6 27B v2"));
    ok(!q27::tool_dialect_xml(), "dialect: qwopus fine-tune stays json");
    // env overrides win in BOTH directions
    setenv("Q27_TOOL_DIALECT", "json", 1);
    q27::set_tool_dialect_for_model(meta("Qwen3.8-27B"));
    ok(!q27::tool_dialect_xml(), "dialect: env json overrides a 3.8 model");
    setenv("Q27_TOOL_DIALECT", "xml", 1);
    q27::set_tool_dialect_for_model(meta("Qwen3.6-27B"));
    ok(q27::tool_dialect_xml(), "dialect: env xml overrides a 3.6 model");
    unsetenv("Q27_TOOL_DIALECT");
    q27::tool_dialect_xml_default() = false;   // leave global state clean
}

static void test_preamble_swaps_with_dialect() {
    unsetenv("Q27_TOOL_DIALECT");
    q27::tool_dialect_xml_default() = true;
    json tools = json::array({tool("bash", {{"command", true}})});
    std::string pre = q27::tools_preamble(tools);
    ok(pre.find("<tool_call>") != std::string::npos &&
       pre.find("<function=") != std::string::npos,
       "preamble: xml dialect emits XML-format instructions");
    q27::tool_dialect_xml_default() = false;
    pre = q27::tools_preamble(tools);
    ok(pre.find("{\"name\":") != std::string::npos &&
       pre.find("<function=") == std::string::npos,
       "preamble: json dialect emits JSON-format instructions");
    q27::tool_dialect_xml_default() = false;   // clean up
    unsetenv("Q27_TOOL_DIALECT");
}

static void test_reasoning_effort_line() {
    // The trained 3.8 template injects the effort line at the HEAD of the
    // system block when thinking is on, defaulting to xhigh (2026-08-15).
    // NOTE (PR #37 review): the froggeric "default medium" change was dropped --
    // the effective default stays xhigh/off; medium is reachable explicitly.
    unsetenv("Q27_REASONING_EFFORT");
    unsetenv("Q27_TOOL_DIALECT");
    auto meta = [](const char* n) { return std::string("{\"general.name\": \"") + n + "\"}"; };
    std::vector<q27::Msg> msgs = {{"system", "Be terse.", {}}, {"user", "hi", {}}};
    json tools = json::parse(R"([{"type":"function","function":{"name":"Read","parameters":{"type":"object"}}}])");
    const std::string XH = "Reasoning effort is set to xhigh.";
    q27::set_tool_dialect_for_model(meta("Qwen38 27b Hf"));
    std::string p = q27::chatml_prompt(msgs, tools, /*think=*/true);
    size_t at = p.find(XH), tools_at = p.find("# Tools");
    ok(at != std::string::npos && tools_at != std::string::npos && at < tools_at,
       "effort: 3.8+think injects xhigh line before # Tools");
    ok(q27::chatml_prompt(msgs, tools, /*think=*/false).find(XH) == std::string::npos,
       "effort: no-think render carries no effort line");
    q27::set_tool_dialect_for_model(meta("Qwen3.6-27B"));
    ok(q27::chatml_prompt(msgs, tools, true).find("Reasoning effort") == std::string::npos,
       "effort: 3.6 family renders unchanged");
    q27::set_tool_dialect_for_model(meta("Qwen38 27b Hf"));
    setenv("Q27_REASONING_EFFORT", "off", 1);
    ok(q27::chatml_prompt(msgs, tools, true).find("Reasoning effort") == std::string::npos,
       "effort: off restores legacy rendering");
    setenv("Q27_REASONING_EFFORT", "low", 1);
    ok(q27::chatml_prompt(msgs, tools, true).find("Reasoning effort is set to low.") != std::string::npos,
       "effort: low selects the trained low string");
    setenv("Q27_REASONING_EFFORT", "medium", 1);
    ok(q27::chatml_prompt(msgs, tools, true).find("Reasoning effort") == std::string::npos,
       "effort: medium emits no line, matching the template");
    unsetenv("Q27_REASONING_EFFORT");
    q27::tool_dialect_xml_default() = false;   // leave global state clean
}

static void test_native_xml_dialect() {
    q27::ToolCall tc;
    ok(q27::parse_native_xml_call(
           "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
           "<parameter=units>\nmetric\n</parameter>\n</function>\n", tc) &&
           tc.ok && tc.name == "get_weather" &&
           tc.arguments.value("city", std::string()) == "Paris" &&
           tc.arguments.value("units", std::string()) == "metric",
       "native-xml: two scalar parameters round-trip");
    q27::ToolCall t2;
    ok(q27::parse_native_xml_call(
           "<function=Write>\n<parameter=content>\nline one\nline two\n</parameter>\n"
           "<parameter=count>\n3\n</parameter>\n<parameter=force>\ntrue\n</parameter>\n"
           "</function>", t2) &&
           t2.arguments.value("content", std::string()) == "line one\nline two" &&
           t2.arguments.value("count", 0) == 3 && t2.arguments.value("force", false) == true,
       "native-xml: multi-line string, tojson-typed number and bool");
    q27::ToolCall t3;
    ok(q27::parse_native_xml_call("<function=list_files>\n</function>", t3) &&
           t3.ok && t3.arguments.empty(),
       "native-xml: zero-parameter call is legal");
    q27::ToolCall t4;
    ok(q27::parse_native_xml_call(
           "<function=Write>\n<parameter=content>\ntruncated with no closer", t4) &&
           t4.arguments.value("content", std::string()) == "truncated with no closer",
       "native-xml: unterminated FINAL parameter closes at EOF (2026-08-15 leniency)");
    q27::ToolCall t5;
    ok(!q27::parse_native_xml_call("plain text, no dialect", t5),
       "native-xml: non-dialect text is not consumed");
}

// Drift mode 18 (2026-08-17, issue #24, reported against Qwen3.6 where the
// dialect default is JSON): the model reverts to its trained XML form but
// DROPS the `<function=` opener, writing the bare `<name>` tag and often a
// stray `</parameter>` before the first real one. Payload intact, opener gone
// -- the XML twin of JSON mode 10. Refusing it dumped a live tool call into
// the text channel, where the agent read it as prose and stopped at turn 1.
// Drift modes 19 and 20 (2026-08-19). Both were found by capturing the raw
// assistant text of every trial in the Qwen3.8 reasoning-effort A/B: 4 of 20
// trials emitted a tool call the parser refused, and every one of those died at
// ~50K tokens with hidden=0 -- Claude Code sees no tool_use block, decides the
// task is done, and exits `completed` in 11 seconds. The fixtures below are the
// exact bytes from those transcripts, not reconstructions.
static void test_mode19_attribute_opener() {
    // xhigh/bench-task-queue trial-1..3, byte-for-byte (the same call three
    // times over -- a deterministic basin, not trajectory noise).
    q27::ToolCall tc;
    ok(q27::parse_native_xml_call(
           "<function name=\"Bash\">\n"
           "<parameter=command>\n"
           "for f in /workspace/phases/phase-*.md; do echo \"=== $f ===\"; cat \"$f\"; echo; done\n"
           "</parameter>\n"
           "<parameter=description>\nRead all phase requirement files\n</parameter>\n"
           "</function>", tc) &&
           tc.ok && tc.name == "Bash" &&
           tc.arguments.value("description", std::string()) == "Read all phase requirement files" &&
           tc.arguments.value("command", std::string()).find("phase-*.md") != std::string::npos,
       "mode19: <function name=\"Bash\"> attribute opener recovers the call");
    // single quotes and extra attributes must not break the name extraction
    q27::ToolCall t2;
    ok(q27::parse_native_xml_call(
           "<function name='Read'>\n<parameter=file_path>\n/w/x.ts\n</parameter>\n</function>", t2) &&
           t2.ok && t2.name == "Read" &&
           t2.arguments.value("file_path", std::string()) == "/w/x.ts",
       "mode19: single-quoted attribute name");
    // the three openers this branch must NOT swallow
    q27::ToolCall t3;
    ok(q27::parse_native_xml_call("<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>", t3) &&
           t3.name == "Bash",
       "mode19: native <function= opener still takes the original path");
    q27::ToolCall t4;
    ok(!q27::parse_native_xml_call("<function> {\"name\": \"Bash\"} </function>", t4),
       "mode19: bare <function> (mode 16's shape) is not claimed here");
    q27::ToolCall t5;
    ok(!q27::parse_native_xml_call("<function of two variables>\nis not a call\n", t5),
       "mode19: prose after <function ...> without name= is not a call");
}

static void test_mode20_nameless_tool_name() {
    // xhigh/bench-time-tracker trial-1, byte-for-byte: plural opener, EMPTY
    // <tool_name>, <tool> separators, mismatched closers. The name exists
    // nowhere in the bytes, so it has to come from the parameter keys.
    const std::string raw =
        "Let me look at the remaining config files.\n\n"
        "<tool_calls>\n<tool_name>\n<parameter=file_path>\n/workspace/vitest.config.ts\n"
        "</parameter>\n</function>\n<tool>\n<tool_name>\n<parameter=file_path>\n"
        "/workspace/TASK.md\n</parameter>\n</function>\n</tool_call>";
    json tools = json::array({tool("Read", {{"file_path", true}}),
                              tool("Bash", {{"command", true}, {"description", false}})});
    std::string prefix, remaining;
    auto calls = q27::parse_bare_tool_calls(raw, &prefix, &tools, true, true, &remaining);
    ok(calls.size() == 2 && calls[0].name == "Read" && calls[1].name == "Read" &&
           calls[0].arguments.value("file_path", std::string()) == "/workspace/vitest.config.ts" &&
           calls[1].arguments.value("file_path", std::string()) == "/workspace/TASK.md",
       "mode20: nameless <tool_name> batch recovers both calls by inference");
    ok(prefix.find("remaining config files") != std::string::npos,
       "mode20: prose before the batch is preserved as prefix");

    // THE negative that matters: hallucinated tool RESULTS wear the same
    // <tool_calls> opener. Promoting those to calls would feed the model's own
    // invented output back as if a tool had produced it (mode 16's rule).
    const std::string hallucinated =
        "<tool_calls>\n<result>\n<name>Read</name>\n<output>\nfile contents\n</output>\n"
        "<tool_name>\n<parameter=file_path>\n/w/x\n</parameter>\n</function>";
    std::string p2;
    auto none = q27::parse_bare_tool_calls(hallucinated, &p2, &tools);
    ok(none.empty(), "mode20: hallucinated <result>/<output> block is NOT a call");

    // A named <tool_name> is mode 14's; mode 20 must not race it.
    const std::string named =
        "<tool_call>\n<tool_name>Read</tool_name>\n<parameter=file_path>\n/w/y\n</parameter>\n</function>";
    std::string p3;
    auto m14 = q27::parse_bare_tool_calls(named, &p3, &tools);
    ok(m14.size() == 1 && m14[0].name == "Read",
       "mode20: named <tool_name> still resolves to Read (mode 14 path intact)");

    // Inference must refuse rather than guess when the keys fit nothing.
    const std::string unknowable =
        "<tool_calls>\n<tool_name>\n<parameter=zzz_unknown>\nv\n</parameter>\n</function>";
    std::string p4;
    auto no = q27::parse_bare_tool_calls(unknowable, &p4, &tools);
    ok(no.empty(), "mode20: un-inferable parameter keys are left as text");
}

// Drift mode 21 (2026-08-20, issue #24 follow-up). One degradation past mode
// 20: there the name tag was present and empty, here there is no opener at all
// and the emission starts at the first <parameter=. Fixture is the reporter's
// bytes.
// Name normalization (2026-08-20). Found live in a budget-fraction A/B, not in
// a bug report: two plugin-marketplace trials died at exactly 50K tokens
// emitting `<function="Bash">`. The dialect is the TRAINED one and parses --
// the name just arrives wrapped in quote characters, matches no declared tool,
// and the call is dropped. Structure was never the issue.
static void test_quoted_name_normalization() {
    q27::ToolCall tc;
    ok(q27::parse_native_xml_call(
           "<function=\"Bash\">\n<parameter=command>\nls -la\n</parameter>\n</function>", tc) &&
           tc.ok && tc.name == "Bash" &&
           tc.arguments.value("command", std::string()) == "ls -la",
       "quoted name: <function=\"Bash\"> resolves to Bash");
    q27::ToolCall t2;
    ok(q27::parse_native_xml_call(
           "<function='Read'>\n<parameter=file_path>\n/w/x\n</parameter>\n</function>", t2) &&
           t2.name == "Read",
       "quoted name: single quotes stripped too");
    // an unbalanced quote is mangling, not a quoted name -- leave it alone
    q27::ToolCall t3;
    ok(q27::parse_native_xml_call(
           "<function=\"Bash>\n<parameter=command>\nls\n</parameter>\n</function>", t3) &&
           t3.name == "\"Bash",
       "quoted name: unbalanced quote is left verbatim (undeclared, so refused upstream)");
    // the ordinary unquoted form must be untouched
    q27::ToolCall t4;
    ok(q27::parse_native_xml_call(
           "<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>", t4) &&
           t4.name == "Bash",
       "quoted name: unquoted names unchanged");
}

static void test_mode21_openerless_params() {
    json tools = json::array({tool("FileAction", {{"filePath", true}, {"oldString", false},
                                                  {"newString", false}}),
                              tool("Bash", {{"command", true}, {"description", false}})});
    const std::string raw =
        "I'll update the imports.\n\n"
        "<parameter=filePath>\n/code/fileaction.go\n</parameter>\n"
        "<parameter=newString>\nimport (\n\t\"bytes\"\n)\n</parameter>\n"
        "<parameter=oldString>\nimport (\n\t\"fmt\"\n)\n</parameter>\n</function>";
    std::string pre;
    auto v = q27::parse_bare_tool_calls(raw, &pre, &tools);
    ok(v.size() == 1 && v[0].name == "FileAction" &&
           v[0].arguments.value("filePath", std::string()) == "/code/fileaction.go",
       "mode21: openerless parameter list recovers via key inference");
    ok(pre.find("update the imports") != std::string::npos,
       "mode21: prose before the parameters is preserved as prefix");

    // A closing tag it never opened is the evidence that this is dialect and
    // not English. Without it, prose that merely mentions <parameter= is text.
    const std::string prose =
        "the schema uses <parameter=filePath> for the target, which is unusual.";
    std::string p2;
    ok(q27::parse_bare_tool_calls(prose, &p2, &tools).empty(),
       "mode21: <parameter= in prose with no </function> is NOT a call");

    // Keys that match no declared tool must refuse rather than guess. This is
    // the camelCase-vs-snake_case case: a real risk when the model mangles the
    // spelling, and executing the wrong tool is worse than not executing one.
    const std::string unknown =
        "<parameter=zzz_nothing>\nv\n</parameter>\n</function>";
    std::string p3;
    ok(q27::parse_bare_tool_calls(unknown, &p3, &tools).empty(),
       "mode21: un-inferable keys refuse rather than guess a tool");

    // The better-specified openers must still win, not get shadowed by this.
    const std::string named =
        "<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>";
    std::string p4;
    auto nv = q27::parse_bare_tool_calls(named, &p4, &tools);
    ok(nv.size() == 1 && nv[0].name == "Bash",
       "mode21: an explicit <function= opener still resolves to Bash");
}

// Regression (2026-08-21): the <result>/<output> hallucinated-result guard is a
// WHOLE-STRING find, but it gates the entire mode 17/20/21 recovery block. So an
// openerless <parameter=...> + </function> call (mode 21) in a tail that ALSO
// contains an unrelated <output> or <result> anywhere -- a quoted command result,
// a prior hallucinated block in the same segment -- bails the whole chain and the
// real call leaks through as plain text. This is the "sometimes I still see
// <parameter= / </function>" report. The guard must be scoped to the parameter
// span, not the whole tail.
static void test_mode21_guard_scoped_to_span() {
    json tools = json::array({tool("FileAction", {{"filePath", true}, {"newString", false}}),
                              tool("Bash", {{"command", true}, {"description", false}})});

    // 1. The leak: a real openerless call, with an unrelated <output> block
    //    EARLIER in the same tail (e.g. a quoted result from a prior step).
    //    Today the whole-string guard sees <output> and skips mode 21 entirely.
    {
        const std::string raw =
            "Here is what the build printed:\n"
            "<output>make: all targets up to date</output>\n\n"
            "Now I'll fix the import.\n"
            "<parameter=filePath>\n/code/fileaction.go\n</parameter>\n"
            "<parameter=newString>\nimport \"bytes\"\n</parameter>\n</function>";
        std::string pre;
        auto v = q27::parse_bare_tool_calls(raw, &pre, &tools);
        ok(v.size() == 1 && v[0].name == "FileAction" &&
               v[0].arguments.value("filePath", std::string()) == "/code/fileaction.go",
           "mode21: an unrelated <output> earlier in the tail must not kill the call");
    }

    // 2. Same, with <result> as the contaminant.
    {
        const std::string raw =
            "<result>previous step output</result>\n"
            "<parameter=filePath>\n/code/x.go\n</parameter>\n</function>";
        std::string pre;
        auto v = q27::parse_bare_tool_calls(raw, &pre, &tools);
        ok(v.size() == 1 && v[0].name == "FileAction",
           "mode21: an unrelated <result> earlier in the tail must not kill the call");
    }

    // 3. The guard must STILL bite where it belongs: a hallucinated tool RESULT
    //    inside the span (<parameter> wrapping <result>/<output>) is invented
    //    output, not a call. This is the behavior the guard exists to protect.
    {
        const std::string raw =
            "<parameter=filePath>\n<result>fake</result>\n</parameter>\n</function>";
        std::string pre;
        auto v = q27::parse_bare_tool_calls(raw, &pre, &tools);
        ok(v.empty(),
           "mode21: <result> INSIDE the parameter span still refuses (hallucinated result)");
    }
}

static void test_mode18_bare_name_opener() {
    q27::ToolCall tc;
    ok(q27::parse_native_xml_call(
           "<name>task\n</parameter>\n"
           "<parameter=description>\nAdd timestamp comment to hello.py\n</parameter>\n"
           "<parameter=prompt>\nedit the file\n</parameter>\n"
           "<parameter=subagent_type>\ngeneral\n</parameter>\n</function>", tc) &&
           tc.ok && tc.name == "task" &&
           tc.arguments.value("description", std::string()) ==
               "Add timestamp comment to hello.py" &&
           tc.arguments.value("prompt", std::string()) == "edit the file" &&
           tc.arguments.value("subagent_type", std::string()) == "general",
       "mode18: bare <name> opener + stray </parameter> recovers the call");
    q27::ToolCall t2;
    ok(q27::parse_native_xml_call("<name>ListTools\n</function>", t2) &&
           t2.ok && t2.name == "ListTools" && t2.arguments.empty(),
       "mode18: bare <name> with </function> and no parameters");
    // The opener is weak evidence on its own, so corroborating dialect
    // structure is required -- otherwise ordinary prose gets eaten.
    q27::ToolCall t3;
    ok(!q27::parse_native_xml_call(
           "<name>foo is a placeholder in the docs, not a call.", t3),
       "mode18: prose opening with <name> and no dialect is NOT a call");
    q27::ToolCall t4;
    ok(q27::parse_native_xml_call("<name>Read\n<parameter=file_path>\n/x/y.py\n"
                                  "</parameter>\n</function>", t4) &&
           t4.name == "Read" &&
           t4.arguments.value("file_path", std::string()) == "/x/y.py",
       "mode18: bare <name> without the stray closer");
    // Mode-18 gap (found via pi's mangled tool transport): the name opens
    // `<name>` but lands on the NEXT line. Pre-fix this read an empty name
    // and refused the whole call. Corroborating <parameter= present, so this
    // must rescue -- and must NOT swallow an identical opener with no dialect.
    q27::ToolCall t5;
    ok(q27::parse_native_xml_call("<name>\nbash\n</parameter>\n"
                                  "<parameter=command>\nls /code\n</parameter>\n", t5) &&
           t5.ok && t5.name == "bash" &&
           t5.arguments.value("command", std::string()) == "ls /code",
       "mode18-FIX: name on the line after <name> is still recognized");
    q27::ToolCall t6;
    ok(!q27::parse_native_xml_call("<name>\njust prose, no dialect follows", t6),
       "mode18-FIX: name-on-next-line with no dialect is NOT a call");
}

static void test_mode14_tool_name_xml_dialect() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}},{"type":"function","function":{"name":"Bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command"]}}}])");
    // Verbatim bytes from the captured transcript.
    std::string t =
        "<tool_name>Read</tool_name>\n"
        "<parameter=file_path>/workspace/tests/index.test.ts</parameter>\n"
        "<tool_name>Bash</tool_name>\n"
        "<parameter=arguments>\n"
        "{\"command\":\"ls /workspace/src /workspace/tests && cat /workspace/.eslintrc.cjs "
        "/workspace/vitest.config.ts\",\"description\":\"List src/tests and show eslint and "
        "vitest configs\"}}";
    const bool expect_rescued = true;    // rescued as of the mode-14 parser path
    std::string pre;
    auto v = q27::parse_bare_tool_calls(t, &pre, &tools);
    ok((!v.empty()) == expect_rescued,
       expect_rescued ? "mode14: <tool_name>/<parameter=> dialect rescued"
                      : "mode14: <tool_name>/<parameter=> dialect UN-RESCUED (documented)");
    // Both calls must come back, with the scalar parameter AND the JSON-object
    // parameter each mapped correctly, and the trailing brace tolerated.
    ok(v.size() == 2, "mode14: both concatenated calls recovered");
    if (v.size() == 2) {
        ok(v[0].ok && v[0].name == "Read" &&
               v[0].arguments.value("file_path", std::string()) ==
                   "/workspace/tests/index.test.ts",
           "mode14: scalar <parameter=file_path> mapped");
        ok(v[1].ok && v[1].name == "Bash" &&
               v[1].arguments.value("command", std::string()).rfind("ls /workspace/src", 0) == 0 &&
               v[1].arguments.value("description", std::string()) ==
                   "List src/tests and show eslint and vitest configs",
           "mode14: <parameter=arguments> JSON object merged, trailing brace tolerated");
    }
}

static void test_mode13_truncated_mid_escape() {
    json tools = json::parse(R"([{"type":"function","function":{"name":"Write","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}}])");
    // trailing dangling backslash
    std::string t1 = "Let me write that.\n{\"name\": \"Write\", \"arguments\": {\"file_path\": \"g.md\", "
                     "\"content\": \"# Guide\\n\\n```json\\n{\\n  \\\"role\\\": \\\"assistant\\\",\\";
    std::string pre;
    auto v1 = q27::parse_bare_tool_calls(t1, &pre, &tools);
    ok(v1.size() == 1 && v1[0].ok && v1[0].name == "Write",
       "mode13: truncated at a dangling backslash");
    // partial \uXXXX at the cut
    std::string t2 = "{\"name\": \"Write\", \"arguments\": {\"file_path\": \"g.md\", \"content\": \"caf\\u00";
    auto v2 = q27::parse_bare_tool_calls(t2, &pre, &tools);
    ok(v2.size() == 1 && v2[0].ok && v2[0].name == "Write",
       "mode13: truncated inside a partial \\uXXXX");
    // a COMPLETE escape at the cut must still round-trip (no over-trim)
    std::string t3 = "{\"name\": \"Write\", \"arguments\": {\"file_path\": \"g.md\", \"content\": \"caf\\u00e9";
    auto v3 = q27::parse_bare_tool_calls(t3, &pre, &tools);
    ok(v3.size() == 1 && v3[0].ok &&
           v3[0].arguments.value("content", std::string()) == "caf\u00e9",
       "mode13: a COMPLETE escape at the cut is not over-trimmed");
}

// ---------------------------------------------------------------------------
// TOOL-CHANNEL REACHABILITY (2026-08-20). Every mode above is reached through
// parse_bare_tool_calls, and resolve_ordered_tool_segments used to call that
// only on TEXT segments. A segment the splitter claimed as TOOL -- meaning the
// model DID emit <tool_call>...</tool_call> and then drifted INSIDE it -- got
// parse_tool_call and nothing else, so modes 10/11/13/14/15/17/20/21 could not
// fire there at all. Adding a mode fixed the bare case and left the wrapped
// twin dead, which is why three modes shipped in one week with green tests and
// no measurable effect.
//
// These drive the real StreamSplitter and the real resolver rather than the
// parser in isolation, because the defect was never in a parser: it was in
// which parser the live path reaches.
static q27::OrderedToolOutput resolve_stream(const std::string& raw, const json* tools,
                                             size_t max_calls = 64) {
    q27::StreamSplitter sp;
    std::vector<std::pair<q27::StreamSplitter::Chan, std::string>> segments;
    // the server's own routing: coalesce like-channel runs, and keep the empty
    // boundary segment that separates back-to-back tool calls.
    auto route = [&](q27::StreamSplitter::Chan ch, const std::string& t) {
        if (t.empty()) {
            if (ch != q27::StreamSplitter::TOOL && !segments.empty() &&
                segments.back().first == q27::StreamSplitter::TOOL)
                segments.emplace_back(ch, std::string());
            return;
        }
        if (!segments.empty() && segments.back().first == ch) segments.back().second += t;
        else segments.emplace_back(ch, t);
    };
    for (auto& p : sp.feed(raw)) route(p.first, p.second);
    for (auto& p : sp.flush()) route(p.first, p.second);
    return q27::resolve_ordered_tool_segments(
        segments, tools, false,
        [max_calls](const std::string&, size_t accepted) { return accepted < max_calls; });
}

// Proves the premise rather than assuming it: these bytes really do land in the
// TOOL channel. If the splitter ever stopped claiming them the reachability
// tests below would pass for the wrong reason.
static bool claimed_as_tool(const std::string& raw) {
    q27::StreamSplitter sp;
    bool tool = false;
    for (auto& p : sp.feed(raw)) tool |= (p.first == q27::StreamSplitter::TOOL && !p.second.empty());
    for (auto& p : sp.flush()) tool |= (p.first == q27::StreamSplitter::TOOL && !p.second.empty());
    return tool;
}

static void test_tool_segment_reaches_drift_chain() {
    json tools = json::array({tool("Read", {{"file_path", true}}),
                              tool("Bash", {{"command", true}, {"description", false}})});

    // P1 + P6 + P7: mode 14 inside the wrapper, with prose ahead of it.
    const std::string m14 =
        "Let me check the test file.\n"
        "<tool_call>\n"
        "<tool_name>Read</tool_name>\n"
        "<parameter=file_path>/workspace/tests/index.test.ts</parameter>\n"
        "</tool_call>";
    ok(claimed_as_tool(m14), "wrapped: the payload really is routed to the TOOL channel");
    {
        auto out = resolve_stream(m14, &tools);
        ok(out.calls.size() == 1 && out.calls[0].name == "Read" &&
               out.calls[0].arguments.value("file_path", std::string()) ==
                   "/workspace/tests/index.test.ts",
           "wrapped mode14: <tool_name> payload recovered from a TOOL segment");
        ok(out.recovered >= 1, "wrapped mode14: counted as a recovery");
        ok(out.parts.size() >= 2 && out.parts[0].kind == q27::OrderedToolPart::Kind::Text &&
               out.text.substr(out.parts[0].text_begin,
                               out.parts[0].text_end - out.parts[0].text_begin)
                       .find("check the test file") != std::string::npos &&
               out.parts[1].kind == q27::OrderedToolPart::Kind::Call,
           "wrapped mode14: prose stays ordered before the recovered call");
    }

    // P2: the two-call mode 14 batch, verbatim bytes, inside one wrapper.
    {
        const std::string raw =
            "<tool_call>\n"
            "<tool_name>Read</tool_name>\n"
            "<parameter=file_path>/workspace/tests/index.test.ts</parameter>\n"
            "<tool_name>Bash</tool_name>\n"
            "<parameter=arguments>\n"
            "{\"command\":\"ls /workspace/src\",\"description\":\"List src\"}}\n"
            "</tool_call>";
        auto out = resolve_stream(raw, &tools);
        ok(out.calls.size() == 2 && out.calls[0].name == "Read" && out.calls[1].name == "Bash" &&
               out.calls[1].arguments.value("description", std::string()) == "List src",
           "wrapped mode14: both concatenated calls recovered from one TOOL segment");
    }

    // P3: mode 10, the dropped `{"name": "` opener, inside the wrapper.
    {
        const std::string raw = "<tool_call>\nRead\", \"file_path\": \"/x/y.py\"}\n</tool_call>";
        auto out = resolve_stream(raw, &tools);
        ok(out.calls.size() == 1 && out.calls[0].name == "Read" &&
               out.calls[0].arguments.value("file_path", std::string()) == "/x/y.py",
           "wrapped mode10: dropped-opener payload recovered from a TOOL segment");
    }

    // P4: mode 15, the <name> pseudo-tag with bare arguments.
    {
        const std::string raw =
            "<tool_call>\n<name>Bash, \"arguments\": {\"command\":\"ls\","
            "\"description\":\"list\"}}\n</tool_call>";
        auto out = resolve_stream(raw, &tools);
        ok(out.calls.size() == 1 && out.calls[0].name == "Bash" &&
               out.calls[0].arguments.value("command", std::string()) == "ls",
           "wrapped mode15: <name> pseudo-tag recovered from a TOOL segment");
    }

    // P5: mode 21, no opener at all, name inferred from the parameter keys.
    {
        json ft = json::array({tool("FileAction", {{"filePath", true}, {"newString", false}}),
                               tool("Bash", {{"command", true}})});
        const std::string raw =
            "<tool_call>\n"
            "<parameter=filePath>\n/code/fileaction.go\n</parameter>\n"
            "<parameter=newString>\nimport (\n\t\"bytes\"\n)\n</parameter>\n"
            "</function>\n</tool_call>";
        auto out = resolve_stream(raw, &ft);
        ok(out.calls.size() == 1 && out.calls[0].name == "FileAction" &&
               out.calls[0].arguments.value("filePath", std::string()) == "/code/fileaction.go",
           "wrapped mode21: openerless parameter list recovered from a TOOL segment");
    }

    // ---- negatives: the fallback must not widen what counts as a call ----

    // N1: the well-formed JSON form still takes the strict path, once.
    {
        const std::string raw =
            "<tool_call>\n{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/w/x\"}}\n</tool_call>";
        auto out = resolve_stream(raw, &tools);
        ok(out.calls.size() == 1 && out.calls[0].name == "Read" &&
               out.calls[0].arguments.value("file_path", std::string()) == "/w/x",
           "wrapped JSON: well-formed call still parses exactly once");
    }
    // N2: the trained native XML dialect, unchanged.
    {
        const std::string raw =
            "<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n"
            "</function>\n</tool_call>";
        auto out = resolve_stream(raw, &tools);
        ok(out.calls.size() == 1 && out.calls[0].name == "Bash" &&
               out.calls[0].arguments.value("command", std::string()) == "ls -la",
           "wrapped native XML: still parses exactly once");
    }
    // N3: a call that PARSED but is not eligible stays text. Re-running the
    // drift chain on it would resurrect a call the caller just rejected --
    // which is how parallel-calls-disabled would silently stop working.
    {
        const std::string raw =
            "<tool_call>\n{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/w/x\"}}\n</tool_call>";
        auto out = resolve_stream(raw, &tools, /*max_calls=*/0);
        ok(out.calls.empty() && out.text.find("\"file_path\"") != std::string::npos,
           "wrapped JSON: an ineligible call stays visible text, not re-rescued");
    }
    // N4: keys that fit no declared tool must refuse rather than guess.
    {
        const std::string raw =
            "<tool_call>\n<parameter=zzz_unknown>\nv\n</parameter>\n</function>\n</tool_call>";
        auto out = resolve_stream(raw, &tools);
        ok(out.calls.empty() && out.text.find("zzz_unknown") != std::string::npos,
           "wrapped: un-inferable parameter keys stay visible text");
    }
    // N5: a hallucinated tool RESULT wearing the dialect must not be promoted;
    // executing the model's own invented output is worse than losing a turn.
    {
        const std::string raw =
            "<tool_call>\n<tool_calls>\n<result>\n<name>Read</name>\n<output>\nfile contents\n"
            "</output>\n<tool_name>\n<parameter=file_path>\n/w/x\n</parameter>\n</function>\n"
            "</tool_call>";
        auto out = resolve_stream(raw, &tools);
        ok(out.calls.empty(), "wrapped: hallucinated <result>/<output> block is not a call");
    }
    // N6: no declared tools means no rescue, same as the TEXT branch.
    {
        auto out = resolve_stream(m14, nullptr);
        ok(out.calls.empty() && out.text.find("<tool_name>") != std::string::npos,
           "wrapped: with no tools declared the payload stays visible text");
    }
}


// The loop state the new branch shares with the TEXT branch: one eligibility
// callback, one running accepted-count, one ordered parts list. A fallback that
// got any of those wrong would still pass every test above, because each of
// those uses a single segment.
static void test_tool_segment_drift_edges() {
    json tools = json::array({tool("Read", {{"file_path", true}}),
                              tool("Bash", {{"command", true}, {"description", false}})});

    // Two wrappers, one strict and one drifted, in that order. The accepted
    // count has to advance across BOTH branches or the second call is indexed
    // against a stale total.
    const std::string mixed =
        "<tool_call>\n{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/w/a\"}}\n</tool_call>\n"
        "<tool_call>\n<tool_name>Bash</tool_name>\n<parameter=command>ls</parameter>\n</tool_call>";
    {
        auto out = resolve_stream(mixed, &tools);
        ok(out.calls.size() == 2 && out.calls[0].name == "Read" && out.calls[1].name == "Bash" &&
               out.calls[1].arguments.value("command", std::string()) == "ls",
           "mixed wrappers: strict call then drifted call, both recovered in order");
        // The newline BETWEEN the two wrappers is a legitimate text part, so
        // assert the call order and that nothing but whitespace separates them
        // rather than pinning a part count.
        std::vector<size_t> order;
        std::string between;
        for (const auto& part : out.parts) {
            if (part.kind == q27::OrderedToolPart::Kind::Call) order.push_back(part.call_index);
            else between += out.text.substr(part.text_begin, part.text_end - part.text_begin);
        }
        ok(order.size() == 2 && order[0] == 0 && order[1] == 1 &&
               between.find_first_not_of(" \t\r\n") == std::string::npos,
           "mixed wrappers: parts stay in generation order, no stray text");
    }
    // Same bytes with parallel calls disabled. The drifted second call must be
    // refused by the SAME policy that governs the strict one -- otherwise the
    // fallback quietly becomes a way around tool_choice.
    {
        auto out = resolve_stream(mixed, &tools, /*max_calls=*/1);
        ok(out.calls.size() == 1 && out.calls[0].name == "Read" &&
               out.text.find("<tool_name>Bash</tool_name>") != std::string::npos,
           "mixed wrappers: the call cap applies to the drifted one too");
    }

    // Prose sharing the wrapper with the call. parse_bare_tool_calls reports
    // spans, so the commentary has to survive as text rather than being eaten
    // with the call or duplicated alongside it.
    {
        const std::string raw =
            "<tool_call>\nLet me open it first.\n<tool_name>Read</tool_name>\n"
            "<parameter=file_path>/w/x.ts</parameter>\n</tool_call>";
        auto out = resolve_stream(raw, &tools);
        size_t first = out.text.find("Let me open it first.");
        ok(out.calls.size() == 1 && out.calls[0].name == "Read" && first != std::string::npos &&
               out.text.find("Let me open it first.", first + 1) == std::string::npos,
           "wrapped: prose inside the wrapper survives once, alongside the call");
    }

    // Reasoning must not be disturbed by the fallback.
    {
        const std::string raw =
            "<think>\nthe file first.\n</think>\nOpening it.\n"
            "<tool_call>\n<tool_name>Read</tool_name>\n"
            "<parameter=file_path>/w/x.ts</parameter>\n</tool_call>";
        auto out = resolve_stream(raw, &tools);
        ok(out.calls.size() == 1 && out.reasoning.find("the file first") != std::string::npos &&
               out.text.find("the file first") == std::string::npos,
           "wrapped: think block stays in reasoning while the drifted call is recovered");
    }

    // Degenerate wrappers must stay no-ops, not throw and not invent a call.
    {
        auto empty = resolve_stream("a<tool_call></tool_call>b", &tools);
        ok(empty.calls.empty() && empty.text.find('a') != std::string::npos &&
               empty.text.find('b') != std::string::npos,
           "wrapped: an empty wrapper is a no-op and keeps the surrounding text");
        auto ws = resolve_stream("<tool_call>\n   \n</tool_call>", &tools);
        ok(ws.calls.empty(), "wrapped: a whitespace-only wrapper is a no-op");
    }
}


// The four survey captures that were genuine routing casualties, verbatim
// bytes, wrapped as the TOOL segment the splitter handed the parser. These are
// the emissions the live server dropped on 2026-08-20 while the same bytes
// recovered offline -- the discrepancy that exposed the routing bug. Keeping
// them here means the next parser change is measured against what the model
// actually emitted, not against what we imagined it emits.
//
// (A fifth capture, unparsed.015, is deliberately absent: it was the model
// writing ABOUT the dialect inside markdown code spans, which the server was
// right to leave as text. It was the survey's false positive, not a miss.
// A sixth, unparsed.006, was a max_tokens truncation -- a different hole, and
// one the server refuses on purpose rather than execute half a Write.)
static void test_survey_captures_wrapped() {
    json tools = json::array({tool("Bash", {{"command", true}, {"description", false}}),
                              tool("Read", {{"file_path", true}}),
                              tool("Write", {{"file_path", true}, {"content", true}}),
                              tool("Grep", {{"pattern", true}, {"path", false}})});
    auto wrapped = [&](const std::string& b) {
        return resolve_stream("<tool_call>\n" + b + "</tool_call>", &tools);
    };

    // capture 003: a JSON brace fused onto the XML opener.
    {
        auto out = wrapped(
            "{\"function=Bash>\n"
            "<parameter=command>\n"
            "echo \"test-output\"; pwd; which git; git --version\n"
            "</parameter>\n"
            "</function>\n");
        ok(out.calls.size() == 1 && out.calls[0].name == "Bash" &&
               out.calls[0].arguments.value("command", std::string())
                       .find("which git") != std::string::npos,
           "capture 003: {\"function=Bash> chimera recovers from a TOOL segment");
    }
    // capture 008: same chimera, different turn -- it recurs, so it is a mode.
    {
        auto out = wrapped(
            "{\"function=Bash>\n"
            "<parameter=command>\n"
            "find /workspace -type f | head -50; echo \"---\"; ls -la /workspace/src/\n"
            "</parameter>\n"
            "</function>\n");
        ok(out.calls.size() == 1 && out.calls[0].name == "Bash",
           "capture 008: the same chimera on another turn recovers too");
    }
    // capture 009: <name> alone on its line, closed by a </parameter> it never
    // opened -- the tag soup is the evidence, and the name is on the next line.
    {
        auto out = wrapped(
            "<name>\n"
            "Bash\n"
            "</parameter>\n"
            "<parameter=command>\n"
            "find /workspace -type f 2>&1\n"
            "</parameter>\n"
            "</function>\n");
        ok(out.calls.size() == 1 && out.calls[0].name == "Bash" &&
               out.calls[0].arguments.value("command", std::string()) ==
                   "find /workspace -type f 2>&1",
           "capture 009: <name>-on-its-own-line recovers from a TOOL segment");
    }
    // capture 011: a proper JSON head, then XML parameters. Both halves are
    // well-formed and they belong to different dialects.
    {
        auto out = wrapped(
            "{\"name\": \"Bash\",\n"
            "<parameter=command>\n"
            "ls -la /workspace/ 2>&1; echo \"---\"\n"
            "</parameter>\n"
            "<parameter=description>\n"
            "List /workspace and check each target path\n"
            "</parameter>\n"
            "</function>\n");
        ok(out.calls.size() == 1 && out.calls[0].name == "Bash" &&
               out.calls[0].arguments.value("description", std::string()) ==
                   "List /workspace and check each target path",
           "capture 011: json-head/xml-params chimera recovers from a TOOL segment");
    }
}


// THE JSON-FUSED OPENER, and what is actually carrying it (2026-08-20, from
// the survey A/B). The model welds a JSON brace onto the trained XML opener:
//     {"function=Read>\n<parameter=file_path>\n/w/x\n</parameter>\n</function>
//     {"function="Read">\n<parameter=file_path>\n...
// Both recover -- but NOT by reading the name. No parser accepts `{"function=`;
// what rescues them is drift mode 21 inferring the tool from the parameter
// keys (the stderr line reads "openerless parameter list -> Read"). The name
// the model wrote is never consulted.
//
// That makes recovery a property of the CALLER'S SCHEMA rather than of the
// bytes: infer_tool_name() decides, and its first rule returns the first tool
// whose REQUIRED set exactly equals the argument keys, so genuine near-twins
// resolve by declaration order. Under the real Claude Code schema every live
// capture lands correctly, which is why this is written down and not changed.
// If someone later teaches the parser to read the fused name, these fixtures
// should keep passing and the mechanism note above becomes stale -- that is
// the intended direction, since reading a declared name is strictly less
// guessing than inferring one.
static void test_json_fused_opener_shapes() {
    // the survey's schema, required lists included -- they are what makes
    // inference work, so a test that drops them proves nothing. (An earlier
    // version of this session's replay harness declared no required keys and
    // made these captures look like parser gaps.)
    json real = json::array({tool("Bash", {{"command", true}, {"description", false}}),
                             tool("Read", {{"file_path", true}, {"limit", false}}),
                             tool("Write", {{"file_path", true}, {"content", true}}),
                             tool("Edit", {{"file_path", true}, {"old_string", true},
                                           {"new_string", true}}),
                             tool("Grep", {{"pattern", true}, {"path", false}})});
    auto call = [](const std::string& t, const json& tools) {
        std::string pre;
        return q27::parse_bare_tool_calls(t, &pre, &tools);
    };

    // verbatim from survey-ab/prefix/unparsed.003.t4
    {
        auto v = call("Let me try the file tools.\n\n{\"function=\"Read\">\n"
                      "<parameter=file_path>\n/workspace/.git/HEAD\n</parameter>\n"
                      "</function>tool>", real);
        ok(v.size() == 1 && v[0].name == "Read" &&
               v[0].arguments.value("file_path", std::string()) == "/workspace/.git/HEAD",
           "json-fused opener: {\"function=\"Read\"> recovers under the real schema");
    }
    // verbatim from survey-ab/prefix/unparsed.001.t2, first call
    {
        auto v = call("{\"function=Read>\n<parameter=file_path>\n/workspace/index.ts\n"
                      "</parameter>\n</function>", real);
        ok(v.size() == 1 && v[0].name == "Read",
           "json-fused opener: the unquoted spelling recovers too");
    }
    // the same emission with a parameter key no declared tool carries. The
    // name says Read and Read is declared, but nothing reads the name, so the
    // call is lost -- this is the fixture that pins the mechanism.
    {
        json only_read = json::array({tool("Read", {{"file_path", true}})});
        ok(call("{\"function=Read>\n<parameter=zzz_unknown>\nv\n</parameter>\n</function>",
                only_read).empty(),
           "json-fused opener: an un-inferable key loses it despite a declared name");
    }
    // the neighbouring dialects must be untouched
    {
        auto w = call("{\"name\": \"Write\",\n<parameter=file_path>\n/w/x\n</parameter>\n"
                      "<parameter=content>\nbody\n</parameter>\n</function>", real);
        ok(w.size() == 1 && w[0].name == "Write" &&
               w[0].arguments.value("content", std::string()) == "body",
           "json-fused opener: mode 17's {\"name\" chimera is unaffected");
        ok(call("the query string uses function=Read> as its selector, oddly.", real).empty(),
           "json-fused opener: bare `function=` in prose is not an opener");
        auto p = call("<function=Read>\n<parameter=file_path>\n/w/x\n</parameter>\n</function>",
                      real);
        ok(p.size() == 1 && p[0].name == "Read",
           "json-fused opener: the proper spelling still takes the named path");
    }
    // and it arrives wrapped, live -- which is the routing fix above.
    {
        auto out = resolve_stream(
            "<tool_call>\n{\"function=\"Read\">\n<parameter=file_path>\n/w/x\n"
            "</parameter>\n</function>\n</tool_call>", &real);
        ok(out.calls.size() == 1 && out.calls[0].name == "Read",
           "json-fused opener: recovered from inside a <tool_call> wrapper");
    }
}


// DRIFT MODE 22 (2026-08-20, issue #24 comment + survey capture 001): the tool
// NAME arrives as the first <parameter=...>, unclosed, with the real
// parameters after it:
//
//   <parameter=bash>
//   <parameter=command>
//   ls /code/docs; grep -rn "INIT_FILE" /code/docs
//   </parameter>
//   </function>
//
// Reported twice by chaudhryfaisal on two different commits, and independently
// captured by the dialect survey here on a different client and schema, so it
// is the dialect and not one setup.
//
// It died SILENTLY: the unclosed <parameter=bash> makes the parameter walk fail
// outright (measured: parsed=0, args={}), so mode 21 has no arguments to infer
// from and refuses. And the UN-RESCUED warning keys off {"name" / {"tool_call"
// / </content>, none of which appear here, so nothing was logged and nothing
// reached Q27_DRIFT_CORPUS -- "no rescue logs", exactly as reported.
//
// The rule reads the name rather than guessing it: a leading <parameter=X>
// counts as an opener only when X names a DECLARED tool and carries no value
// (the next non-whitespace must be another <parameter=). That is strictly
// safer than mode 21's key inference, which is why an undeclared name still
// falls through to it.
static void test_mode22_parameter_as_opener() {
    // Faisal's client: lowercase tool names, camelCase parameter keys.
    json fx = json::array({tool("bash", {{"command", true}, {"description", false}}),
                           tool("read", {{"filePath", true}}),
                           tool("edit", {{"filePath", true}, {"oldString", true},
                                         {"newString", true}})});
    auto call = [](const std::string& t, const json& tools) {
        std::string pre;
        return q27::parse_bare_tool_calls(t, &pre, &tools);
    };

    // verbatim, issue #24 comment 2026-08-21T01:12
    {
        auto v = call("<parameter=bash>\n<parameter=command>\n"
                      "ls /code/docs; grep -rn \"INIT_FILE\" /code/docs /code/README.md "
                      "2>/dev/null | head -30\n</parameter>\n</function>\n", fx);
        ok(v.size() == 1 && v[0].name == "bash" &&
               v[0].arguments.value("command", std::string()).rfind("ls /code/docs", 0) == 0,
           "mode22: parameter-as-opener recovers the single call");
    }
    // verbatim, issue #24 comment 2026-08-21T01:01 -- TWO calls, back to back
    {
        auto v = call("<parameter=bash>\n<parameter=command>\n"
                      "find /code -type f -name \"*.md\" 2>/dev/null | head -20; echo \"---\"; "
                      "ls -R /code/init/internal\n</parameter>\n</function>\n"
                      "<parameter=read>\n<parameter=filePath>\n"
                      "/code/init/internal/config/config.go\n</parameter>\n</function>\n", fx);
        ok(v.size() == 2 && v[0].name == "bash" && v[1].name == "read" &&
               v[1].arguments.value("filePath", std::string()) ==
                   "/code/init/internal/config/config.go",
           "mode22: a batch of two recovers both, in order");
    }
    // verbatim, the edit with three parameters
    {
        auto v = call("<parameter=edit>\n<parameter=filePath>\n/code/init/main.go\n</parameter>\n"
                      "<parameter=newString>\n//\tOLD\n</parameter>\n"
                      "<parameter=oldString>\n//\tNEW\n</parameter>\n</function>\n", fx);
        ok(v.size() == 1 && v[0].name == "edit" &&
               v[0].arguments.value("filePath", std::string()) == "/code/init/main.go" &&
               v[0].arguments.value("newString", std::string()) == "//\tOLD" &&
               v[0].arguments.value("oldString", std::string()) == "//\tNEW",
           "mode22: all three parameters mapped, none eaten by the opener");
    }
    // the same shape from the survey here, capitalized names, different schema
    {
        json survey = json::array({tool("Bash", {{"command", true}, {"description", false}}),
                                   tool("Read", {{"file_path", true}})});
        auto v = call("<parameter=Bash>\n<parameter=command>\n"
                      "find /workspace -type f | head -50\n</parameter>\n</function>\n", survey);
        ok(v.size() == 1 && v[0].name == "Bash",
           "mode22: survey capture 001's second call recovers too");
    }

    // ---- negatives ----

    // mode 21's own shape must still take mode 21's path: no declared tool name
    // in the first parameter, so the name comes from key inference.
    {
        auto v = call("<parameter=filePath>\n/code/x.go\n</parameter>\n"
                      "<parameter=newString>\nbody\n</parameter>\n"
                      "<parameter=oldString>\nold\n</parameter>\n</function>\n", fx);
        ok(v.size() == 1 && v[0].name == "edit" &&
               v[0].arguments.value("filePath", std::string()) == "/code/x.go",
           "mode22: mode 21's openerless list still resolves by inference");
    }
    // a declared name that carries a VALUE is a parameter, not an opener. Here
    // `read` is both a tool name and (pretend) a key with content after it.
    {
        json odd = json::array({tool("read", {{"read", true}, {"filePath", false}})});
        auto v = call("<parameter=read>\n/code/x.go\n</parameter>\n"
                      "<parameter=filePath>\n/code/y.go\n</parameter>\n</function>\n", odd);
        ok(v.size() == 1 && v[0].arguments.value("read", std::string()) == "/code/x.go",
           "mode22: a name-shaped parameter WITH a value stays a parameter");
    }
    // no </function> means prose, same guard mode 21 uses
    {
        ok(call("the docs say <parameter=bash> is the opener, which reads oddly.", fx).empty(),
           "mode22: <parameter=NAME> in prose with no </function> is not a call");
    }
    // an undeclared name in the opener slot must not be invented into a call
    {
        ok(call("<parameter=notatool>\n<parameter=zzz>\nv\n</parameter>\n</function>\n",
                fx).empty(),
           "mode22: an undeclared opener name is refused");
    }
    // hallucinated tool RESULTS wearing the same tags stay text
    {
        ok(call("<result>\n<parameter=bash>\n<parameter=command>\nls\n</parameter>\n"
                "</function>\n<output>\ntotal 0\n</output>\n", fx).empty(),
           "mode22: a <result>/<output> block is not promoted to a call");
    }
    // and it has to survive the wrapper, live
    {
        auto out = resolve_stream("<tool_call>\n<parameter=bash>\n<parameter=command>\n"
                                  "ls /code\n</parameter>\n</function>\n</tool_call>", &fx);
        ok(out.calls.size() == 1 && out.calls[0].name == "bash",
           "mode22: recovered from inside a <tool_call> wrapper");
    }
}


// The miss DETECTOR, which is a separate hole from the parser (2026-08-20).
// Mode 22 went unreported for two days partly because nothing recovered it and
// partly because nothing NOTICED: the UN-RESCUED warning and the
// Q27_DRIFT_CORPUS capture both keyed off {"name" / {"tool_call" / </content>,
// so an emission made entirely of XML tags produced no log line and no corpus
// entry. "No rescue logs" was a true and complete description of a call being
// dropped.
//
// The gate is the same evidence mode 21 uses: a </function> the model never
// opened is dialect, prose that merely quotes a tag is not.
static void test_intended_tool_call_detector() {
    ok(q27::looks_like_intended_tool_call(
           "<parameter=bash>\n<parameter=command>\nls /code\n</parameter>\n</function>"),
       "detector: mode 22's shape is now flagged as an intended call");
    ok(q27::looks_like_intended_tool_call(
           "<function=Read>\n<parameter=file_path>\n/w/x\n</parameter>\n</function>"),
       "detector: the native dialect is flagged");
    ok(q27::looks_like_intended_tool_call(
           "<tool_name>Read</tool_name>\n<parameter=file_path>\n/w/x\n</parameter>\n</function>"),
       "detector: mode 14's shape is flagged");
    ok(q27::looks_like_intended_tool_call("prose.\n\n{\"name\": \"Read\", \"file_path\": \"/x\"}"),
       "detector: the JSON forms it already knew still flag");

    // survey capture 015: the model writing ABOUT the dialect. Quoting a tag is
    // not opening one, and there is no </function> anywhere.
    ok(!q27::looks_like_intended_tool_call(
           "I verified with probes:\n\n| Grep | `^\\s*<function=[A-Za-z]+>` | Fixed listing |\n"
           "| Bash | `grep -rEn '<parameter=x>' /workspace` | Fixed listing |\n"),
       "detector: prose quoting the dialect is NOT flagged");
    ok(!q27::looks_like_intended_tool_call("the schema uses <parameter=filePath> for the target."),
       "detector: <parameter= with no </function> is NOT flagged");
    ok(!q27::looks_like_intended_tool_call("Here is the summary of what I changed."),
       "detector: ordinary prose is NOT flagged");
}


// THE NAMED-OPENER PATH RECOVERED ONLY THE FIRST CALL (2026-08-21, found by
// asking why q27's agentic sessions ended early against llama.cpp on the same
// weights). bench-task-queue trial-1 scored 0.000 on 111k tokens; its entire
// visible answer was the model's plan as prose:
//
//   <function name="TaskCreate>
//   <parameter=subject>
//   Phase 1: Basic FIFO queue
//   </parameter>
//   </function>
//   <tool_call>
//   <function name="TaskCreate>
//   ... x14
//
// The model batched fourteen calls into one turn and never closed the wrapper,
// so the whole generation arrived as ONE unclosed TOOL segment. Replayed on the
// tree before this change: 1 call recovered, 3480 of 3734 bytes leaked as text.
// The agent read its own plan back as an answer and stopped.
//
// Modes 14, 20 and 22 are all batch-capable. The `<function...>` opener path was
// not: it found the first parseable opener, spanned to the first `</function>`,
// and returned. Fourteen in, one out. That asymmetry is mine -- mode 22 was
// written batch-capable this morning and the older named path was left alone,
// and no test drove more than one call through it.
static void test_named_opener_batch() {
    json tools = json::array({tool("TaskCreate", {{"subject", true}, {"description", false}}),
                              tool("Bash", {{"command", true}, {"description", false}}),
                              tool("Read", {{"file_path", true}})});
    auto call = [](const std::string& t, const json& tl) {
        std::string pre;
        return q27::parse_bare_tool_calls(t, &pre, &tl);
    };
    auto named = [](const char* nm, const char* subj) {
        return std::string("<function name=\"") + nm + "\">\n<parameter=subject>\n" + subj +
               "\n</parameter>\n</function>";
    };
    // the live spelling: the quote opens and never closes before '>'
    auto unbalanced = [](const char* nm, const char* subj) {
        return std::string("<function name=\"") + nm + ">\n<parameter=subject>\n" + subj +
               "\n</parameter>\n</function>";
    };
    const std::string SEP = "\n<tool_call>\n";

    {
        auto v = call(named("TaskCreate", "A") + SEP + named("TaskCreate", "B"), tools);
        ok(v.size() == 2 &&
               v[0].arguments.value("subject", std::string()) == "A" &&
               v[1].arguments.value("subject", std::string()) == "B",
           "named-opener batch: two concatenated calls both recover, in order");
    }
    {
        auto v = call(named("TaskCreate", "A") + SEP + named("TaskCreate", "B") + SEP +
                          named("Bash", "C"),
                      tools);
        ok(v.size() == 3 && v[2].name == "Bash",
           "named-opener batch: three recover and the names are read per call");
    }
    {
        auto v = call(unbalanced("TaskCreate", "A") + SEP + unbalanced("TaskCreate", "B"), tools);
        ok(v.size() == 2, "named-opener batch: the live unbalanced-quote spelling batches too");
    }
    // fourteen, the size the live failure actually was
    {
        std::string big = named("TaskCreate", "P1");
        for (int i = 2; i <= 14; i++) big += SEP + named("TaskCreate", "Px");
        auto v = call(big, tools);
        ok(v.size() == 14, "named-opener batch: fourteen calls, the live count");
    }
    // THE SEPARATORS MUST NOT LEAK. Spans are what append_text slices on, so a
    // gap left between calls becomes visible text and puts the raw marker back
    // in front of the user -- the very thing the batch is meant to stop.
    {
        auto out = resolve_stream("<tool_call>\n" + named("TaskCreate", "A") + SEP +
                                      named("TaskCreate", "B"),
                                  &tools);
        ok(out.calls.size() == 2 && out.text.find("<tool_call>") == std::string::npos,
           "named-opener batch: inter-call markers are absorbed, not leaked as text");
    }

    // ---- negatives ----
    {
        auto v = call(named("TaskCreate", "only"), tools);
        ok(v.size() == 1 && v[0].arguments.value("subject", std::string()) == "only",
           "named-opener batch: a single call is unchanged");
    }
    {
        // The batch path refuses an undeclared name and falls through -- but
        // mode 21 downstream then INFERS a tool from the parameter keys and
        // recovers it anyway, so the model naming `NotATool` yields a
        // TaskCreate call. That predates this change (mode 21 cannot see that
        // an explicit name was present and undeclared) and is left alone here,
        // but it is a wart: pinned so it is visible rather than folklore.
        // The property that must hold is the safety one -- an undeclared name
        // is never itself emitted as a call.
        // The model NAMED a tool. If that name is not declared, inference must
        // not quietly substitute a different one: executing TaskCreate because
        // the keys happen to fit, when the model asked for NotATool, is calling
        // something nobody requested. Key inference is for emissions with no
        // usable name at all (mode 21's whole premise), not for overriding one
        // that is present and wrong.
        // 2026-09-08 (undeclared_passthrough): the call is EMITTED under the
        // name the model wrote, so the client can answer "no such tool" and the
        // model corrects itself (Claude Code 4.8 declares no Grep; 3 of 12
        // first turns died on it). Still never inferred past. The strict leg
        // keeps the refusal.
        auto v = call(named("NotATool", "x"), tools);
        ok(q27::undeclared_passthrough()
               ? (v.size() == 1 && v[0].name == "NotATool" &&
                  v[0].arguments.value("subject", std::string()) == "x")
               : v.empty(),
           "explicit undeclared name: passed through as written (refused in strict), not inferred past");
        // ...but an emission with NO opener still infers, which is mode 21
        // working as designed.
        auto w = call("<parameter=subject>\nx\n</parameter>\n</function>", tools);
        ok(w.size() == 1 && w[0].name == "TaskCreate",
           "explicit undeclared name: a nameless emission still infers");
    }
    {
        // a declared first call followed by an undeclared second: keep what was
        // read, do not invent the rest. With the pass-through the second call
        // rides along under its own name; strict keeps only the first.
        auto v = call(named("TaskCreate", "A") + SEP + named("NotATool", "B"), tools);
        ok(!v.empty() && v[0].name == "TaskCreate" &&
               v[0].arguments.value("subject", std::string()) == "A" &&
               (q27::undeclared_passthrough() ? (v.size() == 2 && v[1].name == "NotATool")
                                              : v.size() == 1),
           "named-opener batch: an undeclared later call does not poison the first");
    }
    {
        // prose after the last call is the user's, and must survive
        auto out = resolve_stream("<tool_call>\n" + named("TaskCreate", "A") +
                                      "\nthat is the plan.",
                                  &tools);
        ok(out.calls.size() == 1 && out.text.find("that is the plan.") != std::string::npos,
           "named-opener batch: trailing prose stays visible");
    }
    {
        // the invented-result guard still governs the whole block
        ok(call("<result>\n<output>fake</output>\n" + named("TaskCreate", "A"), tools).empty(),
           "named-opener batch: a hallucinated result block still refuses");
    }
}

// Unclosed-<tool_call> tail recovery (completes bb60661). When the generation
// is cut INSIDE <tool_call>...</tool_call>, the splitter stays on the TOOL
// channel; the server pulls that last segment off with
// take_unclosed_final_tool_segment and used to dump it to visible text raw, so
// a call whose inner JSON was COMPLETE (only the </tool_call> never arrived)
// was lost. recover_unclosed_tool_tail routes it through the drift chain with
// allow_eof_repair=false: COMPLETE inner calls recover, a value genuinely cut
// mid-parse (a partial Write) stays text.
static void test_unclosed_tool_tail_recovery() {
    json tools = json::array({tool("Read", {{"file_path", true}}),
                              tool("Write", {{"content", true}, {"file_path", true}})});

    // The server's own flow: split the raw bytes, and when the trailing
    // channel is TOOL (wrapper unclosed) pull the tail off and recover it.
    auto server_flow = [&](const std::string& raw, bool has_tools,
                           bool (*accept)(const q27::ToolCall&) = nullptr) {
        q27::StreamSplitter sp;
        std::vector<std::pair<q27::StreamSplitter::Chan, std::string>> segments;
        auto route = [&](q27::StreamSplitter::Chan ch, const std::string& t) {
            if (t.empty()) {
                if (ch != q27::StreamSplitter::TOOL && !segments.empty() &&
                    segments.back().first == q27::StreamSplitter::TOOL)
                    segments.emplace_back(ch, std::string());
                return;
            }
            if (!segments.empty() && segments.back().first == ch) segments.back().second += t;
            else segments.emplace_back(ch, t);
        };
        for (auto& p : sp.feed(raw)) route(p.first, p.second);
        for (auto& p : sp.flush()) route(p.first, p.second);
        const bool tail_unclosed =
            !segments.empty() && segments.back().first == q27::StreamSplitter::TOOL;
        std::string unclosed =
            q27::take_unclosed_final_tool_segment(segments, tail_unclosed);
        q27::OrderedToolOutput out;
        size_t rec = q27::recover_unclosed_tool_tail(
            unclosed, has_tools ? &tools : nullptr,
            [&](const std::string& t) { out.append_visible_text(t); },
            [&](q27::ToolCall c) {
                if (accept && !accept(c)) return false;
                out.append_tool_call(std::move(c));
                return true;
            });
        return std::make_pair(rec, out);
    };

    // 1. COMPLETE inner JSON, </tool_call> never arrived -> recovered (mode 1).
    {
        auto [rec, out] = server_flow(
            "<tool_call>\n{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/x\"}}\n",
            true);
        ok(rec == 1 && out.calls.size() == 1 && out.calls[0].name == "Read",
           "unclosed: complete inner JSON recovers from the unclosed tail");
    }

    // 2. Truncated mid-value (a partial Write) -> NOT recovered, stays text.
    //    mode 2/12 EOF repair is disabled here: this is the safety bar.
    {
        auto [rec, out] = server_flow(
            "<tool_call>\n{\"name\":\"Write\",\"arguments\":{\"file_path\":\"/x\",\"content\":\"trun",
            true);
        ok(rec == 0 && out.calls.empty() &&
               out.text.find("content\":\"trun") != std::string::npos,
           "unclosed: partial Write stays text (no half-execution)");
    }

    // 3. No tools configured -> stays text, nothing recovered.
    {
        auto [rec, out] = server_flow(
            "<tool_call>\n{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/x\"}}\n",
            false);
        ok(rec == 0 && out.calls.empty(), "unclosed: no tools -> stays text");
    }

    // 4. Rejected call (caller refuses it) stays text, not re-run.
    {
        auto [rec, out] = server_flow(
            "<tool_call>\n{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/x\"}}\n",
            true, [](const q27::ToolCall&) { return false; });
        ok(rec == 0 && out.calls.empty() &&
               out.text.find("\"name\":\"Read\"") != std::string::npos,
           "unclosed: a refused call stays text and is not resurrected");
    }

    // 5. Native-dialect inner (model omitted the final </parameter>, the exact
    //    shape that broke vLLM's regex parser) -> recovers even unclosed.
    {
        json t2 = json::array({tool("get_weather", {{"location", true}, {"unit", true}})});
        std::string raw = "<tool_call>\n<function=get_weather>\n<parameter=location>\nTokyo\n";
        raw += "</parameter>\n<parameter=unit>\ncelsius\n";
        q27::StreamSplitter sp;
        std::vector<std::pair<q27::StreamSplitter::Chan, std::string>> segments;
        for (auto& p : sp.feed(raw)) segments.push_back(p);
        for (auto& p : sp.flush()) segments.push_back(p);
        std::string unclosed = q27::take_unclosed_final_tool_segment(
            segments, !segments.empty() &&
                           segments.back().first == q27::StreamSplitter::TOOL);
        q27::OrderedToolOutput out;
        size_t rec = q27::recover_unclosed_tool_tail(
            unclosed, &t2,
            [&](const std::string& t) { out.append_visible_text(t); },
            [&](q27::ToolCall c) { out.append_tool_call(std::move(c)); return true; });
        ok(rec == 1 && out.calls.size() == 1 &&
               out.calls[0].arguments.value("unit", std::string()) == "celsius",
           "unclosed: native-dialect omitted-</parameter> recovers (vLLM shape)");
    }
}

// Regression for the mode-20 npos underflow (review 2026-08-20): mode 20
// (nameless <tool_name> batch) used to only stamp batch.front().source_begin
// and batch.back().source_end, so middle calls carried npos and the
// unclosed-tail / closed-wrapper resolvers' `raw.substr(cursor,
// c.source_begin-cursor)` underflowed to SIZE_MAX and threw out_of_range
// inside the request handler. After stamping per-call spans, every call has
// a usable source range and the recovery loop slices cleanly. This test
// would CRASH before the fix.
static void test_mode20_multicall_span_stamping() {
    json tools = json::array({tool("Read", {{"file_path", true}}),
                              tool("Write", {{"content", true}, {"file_path", true}})});
    // two nameless <tool_name> calls in one batch (mode 20 path)
    std::string raw =
        "<tool_calls>\n<tool_name>\n<parameter=file_path>/a</parameter>\n</function>"
        "\n<tool>\n<tool_name>\n<parameter=content>x</parameter>\n"
        "<parameter=file_path>/b</parameter>\n</function>";
    std::string pre, residual;
    auto calls = q27::parse_bare_tool_calls(raw, &pre, &tools);
    ok(calls.size() == 2,
       "mode20: two nameless <tool_name> calls parse into two ToolCalls");
    if (calls.size() == 2) {
        ok(calls[0].source_begin != std::string::npos &&
           calls[0].source_end != std::string::npos &&
           calls[1].source_begin != std::string::npos &&
           calls[1].source_end != std::string::npos,
           "mode20: per-call source spans are stamped (not npos middle)");
        ok(calls[0].source_end <= calls[1].source_begin,
           "mode20: call spans are ordered and non-overlapping");
    }
    // recover_unclosed_tool_tail must not throw on the middle call.
    bool threw = false;
    std::string threw_what;
    q27::OrderedToolOutput out;
    try {
        q27::recover_unclosed_tool_tail(
            raw, &tools,
            [&](const std::string& t) { out.append_visible_text(t); },
            [&](q27::ToolCall c) { out.append_tool_call(std::move(c)); return true; });
    } catch (const std::exception& e) {
        threw = true; threw_what = e.what();
    }
    ok(!threw,
       "mode20: recover_unclosed_tool_tail does not throw on the stamped spans");
    if (threw) printf("    threw: %s\n", threw_what.c_str());
    // Trailing text after the last call's region becomes visible as text
    // (the prior override glued it into back().source_end; the stamping fix
    // exposes it as remaining_text via append_visible_text).
    ok(out.text.find("/b") != std::string::npos ||
       out.calls.size() == 2,
       "mode20: trailing gap is either emitted as text or absorbed by the calls");
}


// PARAMETER VALUES MUST SURVIVE BYTE-FOR-BYTE (2026-08-21, reported live by
// chaudhryfaisal on #33: "File Edit sometime still acting up... lot better but
// still not 100%").
//
// Edit is the only tool that notices. Its oldString has to match the file
// exactly, so a value the parser trims fails to apply while the same trimming
// is invisible to Bash and Read. That is what makes it intermittent: it breaks
// exactly when the edited region happens to end in whitespace.
//
// The value extractor stripped ONE leading newline (correct -- the dialect puts
// the value on its own line) but then ran a GREEDY trailing trim over every
// '\n' and ' '. The format's contract is one delimiter newline at each end, so
// the trailing side now matches the leading side and everything between the
// delimiters is content.
static void test_parameter_value_fidelity() {
    json tools = json::array({tool("Edit", {{"filePath", true}, {"oldString", true},
                                            {"newString", true}}),
                              tool("Read", {{"file_path", true}})});
    auto round_trip = [](const std::string& val) {
        const std::string body =
            "<function=Edit>\n<parameter=filePath>\n/w/x.go\n</parameter>\n"
            "<parameter=oldString>\n" + val + "\n</parameter>\n"
            "<parameter=newString>\nREPLACED\n</parameter>\n</function>";
        q27::ToolCall tc;
        if (!q27::parse_native_xml_call(body, tc)) return std::string("<parse failed>");
        return tc.arguments.value("oldString", std::string());
    };

    ok(round_trip("hello world") == "hello world", "value fidelity: plain value unchanged");
    ok(round_trip("    if (x) {\n        return 1;\n    }") ==
           "    if (x) {\n        return 1;\n    }",
       "value fidelity: indented multi-line code is exact");
    // the three that broke Edit
    ok(round_trip("line1\n") == "line1\n",
       "value fidelity: a value ending in a blank line keeps it");
    ok(round_trip("code    ") == "code    ",
       "value fidelity: trailing spaces are content, not noise");
    ok(round_trip("   ") == "   ",
       "value fidelity: a whitespace-only value survives");
    // and the delimiters themselves are still consumed, exactly one each
    ok(round_trip("plain") == "plain", "value fidelity: delimiter newlines still stripped");
    {
        // CRLF line endings: the closer's delimiter is \r\n, and both bytes go
        const std::string body =
            "<function=Read>\n<parameter=file_path>\n/w/x.go\r\n</parameter>\n</function>";
        q27::ToolCall tc;
        ok(q27::parse_native_xml_call(body, tc) &&
               tc.arguments.value("file_path", std::string()) == "/w/x.go",
           "value fidelity: a CRLF delimiter is consumed whole");
    }
    {
        // a path must not pick up stray bytes -- the case the greedy trim was
        // protecting, kept honest
        const std::string body =
            "<function=Read>\n<parameter=file_path>\n/w/x.go\n</parameter>\n</function>";
        q27::ToolCall tc;
        ok(q27::parse_native_xml_call(body, tc) &&
               tc.arguments.value("file_path", std::string()) == "/w/x.go",
           "value fidelity: an ordinary path is unaffected");
    }
}


// A BATCH WITH NO </function> ANYWHERE (2026-08-21, from the UN-RESCUED lines of
// the batch-fix re-run -- the log found this, not a bug report):
//
//   <function=TaskCreate>
//   <parameter=arguments>
//   {"subject": "...", "description": "..."}}
//   </tool_call>
//   <tool_call>
//   <function=TaskCreate>
//   <parameter=arguments>
//   ...
//
// Each call carries its whole argument object in a single <parameter=arguments>
// and never closes it, and the batch is separated by </tool_call><tool_call>
// rather than by </function>. A SINGLE call of this shape has always recovered;
// the batch did not, and this predates the batch work (verified against 6412204,
// which returns 0 for the batch and 1 for the single, identically).
//
// The cause is a good rule misfiring: parse_native_xml_call refuses an
// unterminated final parameter when another <parameter= follows, because within
// ONE call that is genuine mangling. In a batch the next <parameter= belongs to
// the NEXT CALL. So a span with no </function> is bounded at the next call
// boundary instead of running to end-of-text, which keeps the mangling refusal
// intact for same-call cases -- the boundary is exactly what tells them apart.
static void test_unterminated_batch_no_function_closer() {
    json tools = json::array({tool("TaskCreate", {{"subject", true}, {"description", false},
                                                  {"activeForm", false}}),
                              tool("Bash", {{"command", true}, {"description", false}})});
    auto call = [](const std::string& t, const json& tl) {
        std::string pre;
        return q27::parse_bare_tool_calls(t, &pre, &tl);
    };
    auto argcall = [](const char* subj) {
        return std::string("<function=TaskCreate>\n<parameter=arguments>\n{\"subject\": \"") +
               subj + "\", \"description\": \"d\"}}\n";
    };
    const std::string SEP = "</tool_call>\n<tool_call>\n";

    {
        auto v = call(argcall("Implement TimeTracker") + SEP + argcall("Write tests"), tools);
        ok(v.size() == 2 &&
               v[0].arguments.value("subject", std::string()) == "Implement TimeTracker" &&
               v[1].arguments.value("subject", std::string()) == "Write tests",
           "unterminated batch: two <parameter=arguments> calls recover");
    }
    {
        auto v = call(argcall("A") + SEP + argcall("B") + SEP + argcall("C"), tools);
        ok(v.size() == 3, "unterminated batch: three recover");
    }
    {
        // the single case has always worked and must keep working
        auto v = call(argcall("only one"), tools);
        ok(v.size() == 1 && v[0].arguments.value("subject", std::string()) == "only one",
           "unterminated batch: a single unterminated call is unchanged");
    }
    {
        // mixed: a properly closed call, then an unterminated one
        auto v = call("<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n" +
                          SEP + argcall("after"),
                      tools);
        ok(v.size() == 2 && v[0].name == "Bash" &&
               v[1].arguments.value("subject", std::string()) == "after",
           "unterminated batch: a closed call followed by an unterminated one");
    }

    // ---- the refusal this must NOT weaken ----
    {
        // No call boundary here: the second <parameter= belongs to the SAME
        // call, so an unterminated first parameter is genuine mangling and
        // still refuses. This is the rule the fix threads around, not through.
        auto v = call("<function=Bash>\n<parameter=arguments>\n{\"command\":\"ls\"}\n"
                      "<parameter=command>\nls\n",
                      tools);
        ok(v.empty(), "unterminated batch: mid-call mangling still refuses (no boundary)");
    }
    {
        ok(call("<result>\n<output>x</output>\n" + argcall("A") + SEP + argcall("B"),
                tools).empty(),
           "unterminated batch: a hallucinated result block still refuses");
    }
}


// THE ATTRIBUTE SPELLING OF A PARAMETER (2026-08-21, from the surviving misses
// of the parity sweep):
//
//   <parameter name="command">ls</parameter>
//
// We already absorb both spellings of the FUNCTION opener -- `<function=NAME>`
// and `<function name="NAME">` were consolidated into one grammar in 0f46684 --
// but the parameter tag only ever accepted `<parameter=KEY>`. Same model, same
// turn, same drift habit applied one level down, and it dropped the call.
static void test_parameter_attribute_spelling() {
    json tools = json::array({tool("Bash", {{"command", true}, {"description", false}}),
                              tool("Read", {{"file_path", true}})});
    auto xml = [](const std::string& body) {
        q27::ToolCall tc;
        if (!q27::parse_native_xml_call(body, tc)) return q27::ToolCall{};
        return tc;
    };
    {
        auto tc = xml("<function=Bash>\n<parameter name=\"command\">\nls -la\n</parameter>\n"
                      "</function>");
        ok(tc.ok && tc.name == "Bash" &&
               tc.arguments.value("command", std::string()) == "ls -la",
           "parameter attribute: <parameter name=\"KEY\"> resolves");
    }
    {
        auto tc = xml("<function=Bash>\n<parameter name=command>\nls\n</parameter>\n</function>");
        ok(tc.ok && tc.arguments.value("command", std::string()) == "ls",
           "parameter attribute: the unquoted attribute form resolves");
    }
    {
        // mixed spellings inside ONE call, which is what drift actually looks like
        auto tc = xml("<function=Bash>\n<parameter=command>\nls\n</parameter>\n"
                      "<parameter name=\"description\">\nlist files\n</parameter>\n</function>");
        ok(tc.ok && tc.arguments.value("command", std::string()) == "ls" &&
               tc.arguments.value("description", std::string()) == "list files",
           "parameter attribute: mixed with the plain form in one call");
    }
    {
        // the ordinary spelling must be untouched
        auto tc = xml("<function=Read>\n<parameter=file_path>\n/w/x\n</parameter>\n</function>");
        ok(tc.ok && tc.arguments.value("file_path", std::string()) == "/w/x",
           "parameter attribute: the plain form is unchanged");
    }
    {
        // an empty attribute name is not a parameter
        auto tc = xml("<function=Bash>\n<parameter name=\"\">\nx\n</parameter>\n</function>");
        ok(!tc.ok || tc.arguments.empty() || !tc.arguments.contains(""),
           "parameter attribute: an empty key is refused");
    }
    {
        std::string pre;
        ok(q27::parse_bare_tool_calls(
               "the docs mention <parameter name=\"command\"> as the spelling.", &pre, &tools)
               .empty(),
           "parameter attribute: prose mentioning it is not a call");
    }
}


// The attribute spelling has to reach modes 21 and 22 as well. Both scan with
// their own local `<parameter=` literal, so a call whose parameters use the
// attribute form was invisible to them even after parse_native_xml_call learned
// it -- the fix would have covered only calls that also carried a well-formed
// <function...> opener, which is the case least likely to need rescuing.
static void test_parameter_attribute_reaches_drift_modes() {
    json tools = json::array({tool("FileAction", {{"filePath", true}, {"newString", false}}),
                              tool("bash", {{"command", true}, {"description", false}}),
                              tool("read", {{"filePath", true}})});
    auto call = [](const std::string& t, const json& tl) {
        std::string pre;
        return q27::parse_bare_tool_calls(t, &pre, &tl);
    };
    {
        // mode 21: no opener at all, parameters in the attribute spelling
        auto v = call("<parameter name=\"filePath\">\n/code/x.go\n</parameter>\n"
                      "<parameter name=\"newString\">\nbody\n</parameter>\n</function>",
                      tools);
        ok(v.size() == 1 && v[0].name == "FileAction" &&
               v[0].arguments.value("filePath", std::string()) == "/code/x.go",
           "attribute spelling: mode 21 infers from attribute-form keys");
    }
    {
        // mode 22: the tool name as the first parameter, attribute spelling
        auto v = call("<parameter name=\"bash\">\n<parameter name=\"command\">\n"
                      "ls /code\n</parameter>\n</function>",
                      tools);
        ok(v.size() == 1 && v[0].name == "bash" &&
               v[0].arguments.value("command", std::string()) == "ls /code",
           "attribute spelling: mode 22 reads an attribute-form opener");
    }
    {
        // and the plain spellings must be untouched in both
        auto v = call("<parameter=filePath>\n/code/y.go\n</parameter>\n</function>", tools);
        ok(v.size() == 1 && v[0].name == "FileAction",
           "attribute spelling: mode 21's plain form still works");
        auto w = call("<parameter=bash>\n<parameter=command>\nls\n</parameter>\n</function>",
                      tools);
        ok(w.size() == 1 && w[0].name == "bash",
           "attribute spelling: mode 22's plain form still works");
    }
    {
        ok(call("the schema uses <parameter name=\"filePath\"> for the target.", tools).empty(),
           "attribute spelling: prose with no </function> is still not a call");
    }
}


// A VALUE CONTAINING `</parameter>` (2026-08-21, found by the Edit fidelity
// sweep and flagged to the reporter as its own change).
//
// The XML dialect has no escape mechanism, so a tool call whose value
// legitimately contains the protocol's own closer collided with it: the scan
// took the FIRST `</parameter>` and truncated there. Editing any file that
// contains the dialect -- this parser's own tests, a doc about tool calling, a
// prompt template -- silently lost everything after the first occurrence.
//
// Same collision #31 solved for `</tool_call>` in the splitter, one layer down.
// The real closer is the LAST `</parameter>` before the next parameter opener
// or `</function>`: within one value, an earlier closer that is followed by
// more of the same value was content.
static void test_value_containing_closer() {
    auto val_of = [](const std::string& body, const char* key) {
        q27::ToolCall tc;
        if (!q27::parse_native_xml_call(body, tc)) return std::string("<parse failed>");
        return tc.arguments.value(key, std::string());
    };
    {
        const std::string v = "before\n</parameter>\nafter";
        ok(val_of("<function=Edit>\n<parameter=oldString>\n" + v +
                      "\n</parameter>\n</function>", "oldString") == v,
           "value with closer: survives when it is the only parameter");
    }
    {
        // two parameters, the FIRST one carrying the closer in its value
        const std::string v = "x\n</parameter>\ny";
        const std::string body = "<function=Edit>\n<parameter=oldString>\n" + v +
                                 "\n</parameter>\n<parameter=newString>\nNEW\n</parameter>\n"
                                 "</function>";
        ok(val_of(body, "oldString") == v && val_of(body, "newString") == "NEW",
           "value with closer: the following parameter still resolves");
    }
    {
        // two occurrences inside one value
        const std::string v = "a\n</parameter>\nb\n</parameter>\nc";
        ok(val_of("<function=Edit>\n<parameter=oldString>\n" + v +
                      "\n</parameter>\n</function>", "oldString") == v,
           "value with closer: more than one occurrence survives");
    }
    // ---- the ordinary shapes must be untouched ----
    {
        ok(val_of("<function=Edit>\n<parameter=oldString>\nplain\n</parameter>\n"
                  "<parameter=newString>\nNEW\n</parameter>\n</function>",
                  "oldString") == "plain",
           "value with closer: ordinary two-parameter call unchanged");
    }
    {
        // the final-parameter leniency (unterminated last value) still holds
        q27::ToolCall tc;
        ok(q27::parse_native_xml_call(
               "<function=Bash>\n<parameter=command>\nls -la\n", tc) &&
               tc.arguments.value("command", std::string()) == "ls -la",
           "value with closer: an unterminated final parameter still recovers");
    }
    {
        // REGRESSION GUARD. The first version of the fix bounded the closer
        // search at `</function>` as well as the next parameter opener, which
        // looked tighter and broke a value CONTAINING `</function>`: the bound
        // landed before the real closer, the parameter read as unterminated
        // mid-stream, and the whole call was refused. The next opener alone is
        // sufficient, because a later call's `</parameter>` is always preceded
        // by that call's own `<parameter=`.
        const std::string v = "x\n</function>\ny";
        const std::string body = "<function=Edit>\n<parameter=oldString>\n" + v +
                                 "\n</parameter>\n<parameter=newString>\nNEW\n</parameter>\n"
                                 "</function>";
        q27::ToolCall tc;
        ok(q27::parse_native_xml_call(body, tc) &&
               tc.arguments.value("oldString", std::string()) == v &&
               tc.arguments.value("newString", std::string()) == "NEW",
           "value with closer: a value containing </function> still parses");
    }
}


// THE GARBLED TAIL (2026-08-25, from the first Qwen3.8 drift-corpus capture,
// bench-task-queue): the model closes the call, then keeps going --
//
//   <parameter=description>\nInspect usage\n</parameter>\n</function>\n</functio\n</parameter>
//
// a truncated second closer and a stray `</parameter>`. The last-closer rule
// for the final parameter (its bound is end-of-segment) took that stray
// closer and the description came through as
// "Inspect usage\n</parameter>\n</function>\n</functio". Harmless on a
// description; on a `content` value it writes dialect markup into a file.
// A closer followed by `</function>` is the parameter's real end; a value
// that itself ends in a `</parameter>` line (the #31 collision this rule
// exists for) is still read whole because ITS closer is the one followed by
// the canonical ending.
static void test_last_parameter_garbled_tail() {
    auto val_of = [](const std::string& body, const char* key) {
        q27::ToolCall tc;
        if (!q27::parse_native_xml_call(body, tc)) return std::string("<parse failed>");
        return tc.arguments.value(key, std::string());
    };
    const std::string head = "<function=Bash>\n<parameter=command>\nls\n</parameter>\n"
                             "<parameter=description>\nInspect usage\n</parameter>\n</function>\n";
    ok(val_of(head + "</functio\n</parameter>", "description") == "Inspect usage",
       "garbled tail: truncated closer + stray </parameter> is not the value");
    ok(val_of(head + "</function>\n</parameter>\n</function>", "description") == "Inspect usage",
       "garbled tail: doubled closer + stray </parameter> is not the value");
    ok(val_of(head + "</function>", "description") == "Inspect usage",
       "garbled tail: a merely doubled closer is unchanged");
    // the collision the last-closer rule was written for still reads whole:
    // a value whose own last line is `</parameter>`
    const std::string v = "line one\n</parameter>";
    ok(val_of("<function=Edit>\n<parameter=old_string>\n" + v + "\n</parameter>\n</function>",
              "old_string") == v,
       "garbled tail: a value ending in a </parameter> line still survives");
    // prose between the closers (SWE-bench capture): the call closed, the
    // model talked, then dropped a stray closer -- the prose is not the value
    ok(val_of(head + "Some prose about the call.\n</parameter>", "description") == "Inspect usage",
       "garbled tail: prose after </function> then a stray </parameter> is not the value");
    ok(val_of(head + "Some prose.\n</parameter>\nmore\n</parameter>", "description") == "Inspect usage",
       "garbled tail: two stray closers after prose are not the value");
    // the trade the rule makes, recorded so it is not re-litigated: a LAST
    // value containing `</parameter>\n</function>` and then more content is
    // byte-identical to the observed prose-then-stray-closer garble, and the
    // observed shape wins -- the value reads up to its first closer that
    // `</function>` follows. As a non-last parameter it still reads whole.
    const std::string w = "x\n</parameter>\n</function>\ny";
    ok(val_of("<function=Edit>\n<parameter=old_string>\n" + w + "\n</parameter>\n</function>",
              "old_string") == "x",
       "garbled tail: a last value that looks like the garble reads as the garble");
    ok(val_of("<function=Edit>\n<parameter=old_string>\n" + w + "\n</parameter>\n"
              "<parameter=new_string>\nNEW\n</parameter>\n</function>", "old_string") == w,
       "garbled tail: the same value as a non-last parameter still reads whole");
}

// A NAME TAG NOBODY HAD SEEN (2026-08-25, SWE-bench pylint-6903, corpus
// 220c72b4): junk wrapper tags, a name tag opened twice with the first one
// empty, then a real parameter and EOS --
//
//   <tool_use>\n<tool>\n<parameter_name>\n<parameter_name>Read\n</parameter>\n<parameter=file_path>\n/w/x.py\n
//
// Mode 18's `<name>` cousin. It went to the client as text and ended the
// session because the streaming holdback only ever armed on `<function=`:
// the parser and the stream now read openers off ONE table.
static void test_parameter_name_opener() {
    json tools = json::array({tool("Read", {{"file_path", true}}), tool("Write", {{"file_path", true}, {"content", true}})});
    const std::string shape =
        "\n\n<tool_use>\n<tool>\n<parameter_name>\n<parameter_name>Read\n</parameter>\n<parameter=file_path>\n/w/x.py\n";
    {
        q27::ToolCall tc;
        ok(q27::parse_native_xml_call(shape, tc) && tc.name == "Read" &&
               tc.arguments.value("file_path", std::string()) == "/w/x.py",
           "parameter_name opener: the strict parser reads through the junk");
    }
    {
        std::string pre, remaining;
        auto v = q27::parse_bare_tool_calls(shape, &pre, &tools, true, true, &remaining);
        ok(v.size() == 1 && v[0].name == "Read" && v[0].arguments.value("file_path", std::string()) == "/w/x.py",
           "parameter_name opener: the bare chain recovers it");
        ok(v.size() == 1 && q27::strip_ws2(remaining).empty(),
           "parameter_name opener: the junk tags are residue, not remaining text");
    }
    {
        // the same bytes as TEXT through the resolver: one call, nothing shown
        auto out = resolve_stream(shape, &tools);
        ok(out.calls.size() == 1 && q27::strip_ws2(out.text).empty(),
           "parameter_name opener: resolver -> one call, no text");
    }
    {
        // safety: `<name>` in prose is still prose, and a <name> inside a value
        // does not split a real call into a batch
        std::string pre, remaining;
        auto none = q27::parse_bare_tool_calls("Use the <name> tag for the display name in your XML.",
                                               &pre, &tools, true, true, &remaining);
        ok(none.empty(), "parameter_name opener: <name> in prose is not a call");
        auto one = q27::parse_bare_tool_calls(
            "<function=Write>\n<parameter=file_path>\n/w/a.xml\n</parameter>\n<parameter=content>\n<name>foo</name>\n</parameter>\n</function>",
            &pre, &tools, true, true, &remaining);
        ok(one.size() == 1 && one[0].arguments.value("content", std::string()) == "<name>foo</name>",
           "parameter_name opener: a <name> inside a value stays in the value");
    }
}

// A BATCH THAT MIXES OPENER SPELLINGS (2026-08-25, drift corpus 03a8a851,
// b645b48f, a29175c3): a planning turn emits several calls and drifts
// mid-batch, so the first opens `<function=NAME>` and a later one opens
// `<parameter=NAME>` (mode 22's parameter-as-opener), or vice versa. The
// native batch scanner recovered only the run it started with and handed the
// rest to the client as prose. It now opens on either spelling, so one batch
// mixes them -- mode 22 folded into the same scan rather than a separate pass
// that only fires when the native one found nothing.
static void test_batch_mixed_opener_spellings() {
    json tools = json::array({tool("Read", {{"file_path", true}}),
                              tool("Bash", {{"command", true}, {"description", false}}),
                              tool("TaskUpdate", {{"status", false}, {"taskId", true}})});
    auto names = [](const std::vector<q27::ToolCall>& v) {
        std::string s;
        for (auto& c : v) { s += c.name; s += ' '; }
        return s;
    };
    std::string pre, remaining;
    // <function=> first, then two <parameter=NAME> continuations (03a8a851)
    {
        const std::string t =
            "<function=TaskUpdate>\n<parameter=status>\ndone\n</parameter>\n<parameter=taskId>\nt1\n</parameter>\n</function>\n"
            "<parameter=TaskUpdate>\n<parameter=status>\ndone\n</parameter>\n<parameter=taskId>\nt2\n</parameter>\n</function>\n"
            "<parameter=TaskUpdate>\n<parameter=status>\ndone\n</parameter>\n<parameter=taskId>\nt3\n</parameter>\n</function>";
        auto v = q27::parse_bare_tool_calls(t, &pre, &tools, true, true, &remaining);
        ok(v.size() == 3 && names(v) == "TaskUpdate TaskUpdate TaskUpdate " &&
               v[2].arguments.value("taskId", std::string()) == "t3",
           "mixed openers: <function=> then two <parameter=> continuations -> 3 calls");
        ok(q27::strip_ws2(remaining).empty(), "mixed openers: nothing left as text");
    }
    // two <function=> calls separated by <tool_call>, then a <parameter=> one (b645b48f)
    {
        const std::string t =
            "<function=Read>\n<parameter=file_path>\n/a\n</parameter>\n</function>\n<tool_call>\n"
            "<function=Read>\n<parameter=file_path>\n/b\n</parameter>\n</function>\n"
            "<parameter=Read>\n<parameter=file_path>\n/c\n</parameter>\n</function>";
        auto v = q27::parse_bare_tool_calls(t, &pre, &tools, true, true, &remaining);
        ok(v.size() == 3 && v[0].arguments.value("file_path", std::string()) == "/a" &&
               v[2].arguments.value("file_path", std::string()) == "/c",
           "mixed openers: <function=> x2 then <parameter=> -> 3 calls, generation order");
        ok(q27::strip_ws2(remaining).empty(), "mixed openers: b645b48f leaves no text");
    }
    // <parameter=> first, then a <function=> one -- the scanner must not need
    // to start on <function=
    {
        const std::string t =
            "<parameter=Read>\n<parameter=file_path>\n/a\n</parameter>\n</function>\n"
            "<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>";
        auto v = q27::parse_bare_tool_calls(t, &pre, &tools, true, true, &remaining);
        ok(v.size() == 2 && names(v) == "Read Bash ",
           "mixed openers: <parameter=> then <function=> -> both, in order");
    }
    // an UNDECLARED <parameter=NAME> is not an opener -- it is an ordinary
    // parameter, and must not start a spurious call or break the batch
    {
        const std::string t =
            "<function=Bash>\n<parameter=command>\nls\n</parameter>\n"
            "<parameter=unknownkey>\nvalue\n</parameter>\n</function>";
        auto v = q27::parse_bare_tool_calls(t, &pre, &tools, true, true, &remaining);
        ok(v.size() == 1 && v[0].name == "Bash" &&
               v[0].arguments.value("unknownkey", std::string()) == "value",
           "mixed openers: an undeclared <parameter=> stays an ordinary parameter");
    }
}

// A ZERO-ARGUMENT mode-22 call in a batch (2026-08-26, issue #38, cosmicnag):
// the model plans a parallel batch and its first call takes no arguments --
//   <parameter=memex_recall>\n</function>
//   <parameter=read>\n<parameter=path>\n/a\n</parameter>\n</function>\n</tool_call>
// mode 22 required a parameter after the opener, so the zero-arg call was
// dropped and the agent stalled. It fires now IFF the tool has no required
// params; a required-args tool opened empty stays residue (831ac625).
static void test_zero_arg_mode22_call() {
    json tools = json::array({
        tool("memex_recall", {}),
        tool("Read", {{"path", true}}),
        tool("TaskUpdate", {{"taskId", true}, {"status", false}}),
    });
    auto names = [](const std::vector<q27::ToolCall>& v) {
        std::string s; for (auto& c : v) { s += c.name; s += ' '; } return s;
    };
    std::string pre, remaining;
    {
        const std::string t =
            "<parameter=memex_recall>\n</function>\n"
            "<parameter=read>\n<parameter=path>\n/a/SKILL.md\n</parameter>\n</function>\n</tool_call>\n"
            "<parameter=read>\n<parameter=path>\n/b/CONTEXT.md\n</parameter>\n</function>\n</tool_call>";
        auto v = q27::parse_bare_tool_calls(t, &pre, &tools, true, true, &remaining);
        ok(v.size() == 3 && names(v) == "memex_recall Read Read " &&
               v[0].arguments.empty() &&
               v[1].arguments.value("path", std::string()) == "/a/SKILL.md",
           "zero-arg mode22: memex_recall + two reads recover in order");
        ok(q27::strip_ws2(remaining).empty(), "zero-arg mode22: nothing leaks as text");
    }
    {
        auto v = q27::parse_bare_tool_calls("<parameter=memex_recall>\n</function>",
                                            &pre, &tools, true, true, &remaining);
        ok(v.size() == 1 && v[0].name == "memex_recall" && v[0].arguments.empty(),
           "zero-arg mode22: an isolated zero-arg call fires");
    }
    {
        const std::string t =
            "<function=TaskUpdate>\n<parameter=taskId>\nt1\n</parameter>\n</function>\n"
            "<parameter=TaskUpdate>\n</function>";
        auto v = q27::parse_bare_tool_calls(t, &pre, &tools, true, true, &remaining);
        ok(v.size() == 1 && v[0].name == "TaskUpdate" &&
               v[0].arguments.value("taskId", std::string()) == "t1",
           "zero-arg mode22: an empty required-args opener stays residue (aborted call)");
    }
    {
        auto v = q27::parse_bare_tool_calls("<parameter=notatool>\n</function>",
                                            &pre, &tools, true, true, &remaining);
        ok(v.empty(), "zero-arg mode22: an undeclared empty opener is not a call");
    }
}

// AN ABORTED SECOND CALL (2026-08-25, drift corpus 831ac625): the model
// closes a call, opens another with the mode-22 spelling, and stops --
//
//   <function=TaskUpdate>...</parameter>\n</function>\n<parameter=TaskUpdate>\n</function>
//
// The strict parser kept scanning past `</function>` and handed the client a
// TaskUpdate call with a bogus `TaskUpdate` argument. Now: parameters stop
// where the call closed; an opener with nothing inside is residue, not text
// and not a call; and a real mode-22 second call after `</function>` (one
// that carries a parameter) goes to the batch chain rather than being
// silently dropped by a parser that reads one call.
static void test_aborted_second_call() {
    json tools = json::array({tool("TaskUpdate", {{"taskId", true}, {"status", false}}),
                              tool("Read", {{"file_path", true}}),
                              tool("Bash", {{"command", true}})});
    const std::string one =
        "<function=TaskUpdate>\n<parameter=status>\ndone\n</parameter>\n<parameter=taskId>\nt7\n</parameter>\n</function>";
    const std::string aborted = one + "\n<parameter=TaskUpdate>\n</function>";
    {
        q27::ToolCall tc;
        ok(q27::parse_native_xml_call(aborted, tc) && tc.arguments.size() == 2 &&
               tc.arguments.value("taskId", std::string()) == "t7" && !tc.arguments.contains("TaskUpdate"),
           "aborted second call: the strict parser stops at </function>");
    }
    {
        // wrapped: one call, and the aborted opener is neither text nor a call
        auto out = resolve_stream("<tool_call>\n" + aborted + "\n</tool_call>", &tools);
        ok(out.calls.size() == 1 && out.calls[0].arguments.size() == 2,
           "aborted second call: wrapped -> exactly one call with two arguments");
        ok(q27::strip_ws2(out.text).empty(), "aborted second call: wrapped -> nothing shown as text");
        // and a legal zero-parameter call is untouched by the residue rule
        const json list_tools = json::array({tool("TaskList", {})});
        auto zero = resolve_stream("<tool_call>\n<function=TaskList>\n</function>\n</tool_call>", &list_tools);
        ok(zero.calls.size() == 1 && zero.calls[0].name == "TaskList",
           "aborted second call: a zero-parameter <function=> call still fires");
    }
    {
        // bare, in text: same
        std::string pre, remaining;
        auto v = q27::parse_bare_tool_calls(aborted, &pre, &tools, true, true, &remaining);
        ok(v.size() == 1 && v[0].arguments.size() == 2, "aborted second call: bare -> exactly one call");
        ok(q27::strip_ws2(remaining).empty(), "aborted second call: bare -> the opener is residue, not remaining text");
    }
    {
        // a REAL mode-22 second call after </function> is not the aborted case
        const std::string real = "<function=Read>\n<parameter=file_path>\n/a\n</parameter>\n</function>\n"
                                 "<parameter=Bash>\n<parameter=command>\nls\n</parameter>\n</function>";
        ok(q27::wrapped_body_exceeds_one_call(real), "aborted second call: a second call with a parameter leaves the strict path");
        ok(!q27::wrapped_body_exceeds_one_call(aborted), "aborted second call: the empty opener stays on the strict path");
    }
}

// THE TERMINATOR WRITTEN IN THE WRONG DIALECT (2026-08-21, from the UN-RESCUED
// lines of the parity arm):
//
//   {"name": "TaskCreate", "arguments": {"subject": "Phase 1", "description": "..."}
//   </parameter>
//   </function>
//   <tool_call>
//   {"name": "TaskCreate", "arguments": {"subject": "Phase 2", ...
//
// The JSON body is complete except for its final `}` -- the model wrote the XML
// closers where the brace belonged. Nothing is missing; the terminator is in the
// other dialect.
//
// The EOF repair that would close those braces only fires when the candidate is
// LAST in the text, which is right: bytes after a truncated call mean the model
// moved on, and repairing then invents content. Measured: the same body with
// nothing after it recovers, with prose after it does not, and with dialect
// residue after it did not either -- which is the case that should.
//
// Outside a JSON string, `</parameter>`, `</function>` and `</tool_call>` can
// never be valid JSON, so they are blanked to spaces (in place, so every source
// offset is preserved and the spans stay valid) and the candidate becomes final.
// Prose is NOT blanked and still blocks the repair.
static void test_json_terminator_in_xml_dialect() {
    json tools = json::array({tool("TaskCreate", {{"subject", true}, {"description", false}}),
                              tool("Edit", {{"filePath", true}, {"oldString", false}})});
    auto call = [](const std::string& t, const json& tl) {
        std::string pre;
        return q27::parse_bare_tool_calls(t, &pre, &tl);
    };
    auto body = [](const char* subj) {
        return std::string("{\"name\": \"TaskCreate\", \"arguments\": {\"subject\": \"") + subj +
               "\", \"description\": \"d\"}";
    };

    {
        auto v = call(body("Phase 1") + "\n</parameter>\n</function>\n", tools);
        ok(v.size() == 1 && v[0].arguments.value("subject", std::string()) == "Phase 1",
           "xml terminator: unterminated JSON closed by XML residue recovers");
    }
    {
        auto v = call(body("Phase 1") + "\n</parameter>\n</function>\n<tool_call>\n" +
                          body("Phase 2") + "\n</parameter>\n</function>\n",
                      tools);
        ok(v.size() == 2 && v[0].arguments.value("subject", std::string()) == "Phase 1" &&
               v[1].arguments.value("subject", std::string()) == "Phase 2",
           "xml terminator: a batch of them recovers in order");
    }
    // ---- the refusals this must not weaken ----
    {
        ok(call(body("Phase 1") + "\nthat is the plan.", tools).empty(),
           "xml terminator: PROSE after a truncated call still blocks repair");
    }
    {
        // a value that legitimately contains the closer is inside a JSON
        // string, so it must be left alone
        auto v = call("{\"name\": \"Edit\", \"arguments\": {\"filePath\": \"/w/x\", "
                      "\"oldString\": \"a </parameter> b\"}}",
                      tools);
        ok(v.size() == 1 && v[0].arguments.value("oldString", std::string()) ==
                                "a </parameter> b",
           "xml terminator: a closer INSIDE a JSON string is untouched");
    }
    {
        // well-formed JSON is unaffected
        auto v = call("{\"name\": \"TaskCreate\", \"arguments\": {\"subject\": \"ok\"}}", tools);
        ok(v.size() == 1 && v[0].arguments.value("subject", std::string()) == "ok",
           "xml terminator: well-formed JSON unchanged");
    }
    {
        // REGRESSION GUARD, and a refusal this nearly broke. Blanking the
        // `<tool_call>` OPENER unconditionally recovers the batch above -- and
        // also merges an invalid call and a valid one that are separated by a
        // bare opener, which test_chat_completions_integration refuses on
        // purpose: that boundary is ambiguous and merging it executes a call
        // the model never framed. The opener is only blanked when closer
        // residue sits immediately before it, which is what tells a batch from
        // two unrelated emissions.
        auto v = call("{\"name\": \"first\", \"arguments\": "
                      "<tool_call>\n{\"name\": \"TaskCreate\", \"arguments\": "
                      "{\"subject\": \"x\"}}",
                      tools);
        ok(v.empty() || v.size() == 1,
           "xml terminator: a bare opener between two calls is not a batch boundary");
    }
}


// THE NAME ON THE FOLLOWING LINE, and a case-mismatched name (2026-08-21, from
// classifying every missed turn across 46 runs).
//
// Three spellings, one habit: the opener carries a PLACEHOLDER where the tool
// name belongs and the real name is the next line.
//
//   <function=name>      <name>        <parameter=name>
//   TaskCreate           Write         Bash
//   </parameter>         </parameter>  </parameter>
//
// Mode 18 already reads `<name>Write` when the name is on the SAME line. This
// is the next-line variant, and `name`/`tool_name`/`function` as the literal
// key is the same thing one tag over. Gated on the next line naming a DECLARED
// tool, so it reads what the model wrote instead of guessing -- the rule mode
// 22 uses.
//
// Separately: `<parameter=task>` where the declared tool is `Task`. A tool name
// that differs only in case is the model's own name for it, and refusing on
// case alone loses a call for nothing. Requires a UNIQUE case-insensitive
// match, so an ambiguous registry still refuses.
static void test_placeholder_name_on_next_line() {
    json tools = json::array({tool("TaskCreate", {{"subject", true}, {"description", false}}),
                              tool("Write", {{"file_path", true}, {"content", true}}),
                              tool("Bash", {{"command", true}, {"description", false}}),
                              tool("Task", {{"taskId", true}, {"status", false}})});
    auto call = [](const std::string& t, const json& tl) {
        std::string pre;
        return q27::parse_bare_tool_calls(t, &pre, &tl);
    };
    {
        auto v = call("<function=name>\nTaskCreate\n</parameter>\n"
                      "<parameter=subject>\nCore types\n</parameter>\n</function>", tools);
        ok(v.size() == 1 && v[0].name == "TaskCreate" &&
               v[0].arguments.value("subject", std::string()) == "Core types",
           "next-line name: <function=name> takes the following line");
    }
    {
        auto v = call("<name>\nWrite\n</parameter>\n<parameter=file_path>\n/w/x.ts\n"
                      "</parameter>\n<parameter=content>\nbody\n</parameter>\n</function>",
                      tools);
        ok(v.size() == 1 && v[0].name == "Write" &&
               v[0].arguments.value("file_path", std::string()) == "/w/x.ts",
           "next-line name: <name> with the name on the next line");
    }
    {
        auto v = call("<parameter=name>\nBash\n</parameter>\n<parameter=command>\n"
                      "node -e 1\n</parameter>\n</function>", tools);
        ok(v.size() == 1 && v[0].name == "Bash" &&
               v[0].arguments.value("command", std::string()) == "node -e 1",
           "next-line name: <parameter=name> carries the tool name as its value");
    }
    {
        auto v = call("<parameter=task>\n<parameter=taskId>\n2\n</parameter>\n"
                      "<parameter=status>\nin_progress\n</parameter>\n</function>", tools);
        // taskId parses as a JSON number here, so compare the dumped value --
        // .value(key, std::string()) THROWS on a non-string and took the whole
        // test binary down when this was written the obvious way.
        ok(v.size() == 1 && v[0].name == "Task" && v[0].arguments.contains("taskId") &&
               v[0].arguments["taskId"].dump() == "2",
           "next-line name: a case-mismatched tool name resolves to the declared one");
    }
    // ---- refusals ----
    {
        // the opener carries the placeholder `name`, the next line an
        // undeclared tool: the placeholder is never a tool (the pass-through
        // excludes it) and the line is not inferred past -- refused.
        ok(call("<function=name>\nNotATool\n</parameter>\n<parameter=subject>\nx\n"
                "</parameter>\n</function>", tools).empty(),
           "next-line name: an undeclared following line is refused");
    }
    // ---- 2026-09-08, item 2 of the (p) agenda: the first-turn shapes that
    // ended 4 of 36 SWE-bench sessions in the production arms (transcripts in
    // bench/crossengine/agentic-2026-09-08; replay with tools/stream_probe) ----
    {
        // `<parameter=function=Bash>`: the mode-22 opener with the name behind
        // a `function=` prefix (prodlad, xarray-4094)
        auto v = call("<parameter=function=Bash>\n<parameter=command>\ngrep -rn x /w\n"
                      "</parameter>\n<parameter=description>\nFind x\n</parameter>\n</function>",
                      tools);
        ok(v.size() == 1 && v[0].name == "Bash" &&
               v[0].arguments.value("command", std::string()) == "grep -rn x /w",
           "mode 22: <parameter=function=NAME> opener resolves to NAME");
    }
    {
        // the Anthropic-XML wrapper family around an UNNAMED call, closed by
        // </invoke> and then EOS (prodpfx/prodpfx2, requests-1142): mode 21
        // infers from the keys; the wrapper openers are absorbed into the span
        std::string pre, rest;
        auto v = q27::parse_bare_tool_calls(
            "Let me look.\n\n<function_calls>\n<invoke>\n<parameter=subject>\nx\n"
            "</parameter>\n</invoke>\n", &pre, &tools, true, true, &rest);
        ok(v.size() == 1 && v[0].name == "TaskCreate" &&
               v[0].arguments.value("subject", std::string()) == "x" &&
               pre == "Let me look.\n\n" && rest.find("<invoke>") == std::string::npos,
           "mode 21: unnamed <function_calls><invoke> closed by </invoke> infers, wrapper absorbed");
    }
    {
        // the trained form with a name the client did not declare (Grep under
        // Claude Code 4.8): emitted as written, never inferred to another tool
        auto v = call("<function=Grep>\n<parameter=pattern>\nfoo\n</parameter>\n"
                      "<parameter=-n>\ntrue\n</parameter>\n</function>", tools);
        ok(q27::undeclared_passthrough()
               ? (v.size() == 1 && v[0].name == "Grep" &&
                  v[0].arguments.value("pattern", std::string()) == "foo")
               : v.empty(),
           "undeclared Grep: passed through as written (refused in strict)");
        // ...but not a name that is not an identifier
        ok(call("<function=not a tool>\n<parameter=pattern>\nfoo\n</parameter>\n</function>",
                tools).empty(),
           "undeclared pass-through: a non-identifier name is still refused");
    }
    {
        // mode 18's same-line form must be untouched
        auto v = call("<name>Write\n</parameter>\n<parameter=file_path>\n/w/y\n</parameter>\n"
                      "<parameter=content>\nc\n</parameter>\n</function>", tools);
        ok(v.size() == 1 && v[0].name == "Write",
           "next-line name: mode 18's same-line spelling still works");
    }
    {
        // and an ordinary parameter literally named "name" on a declared tool
        // must not be mistaken for an opener when its value is not a tool
        ok(call("<parameter=name>\nnot a tool at all\n</parameter>\n"
                "<parameter=command>\nls\n</parameter>\n</function>", tools).size() == 1,
           "next-line name: a non-tool value falls through to key inference");
    }
}


// XML BLEEDING INTO THE JSON KEY (2026-08-22, live from the parity arm):
//
//   {"name": "Write", "arguments>
//   {"file_path": "/workspace/src/index.ts", "content": "..."}
//
// The model typed `>` where JSON needs `":` -- the XML dialect's tag-closer
// habit landing inside a JSON key. The string is left unterminated, so nothing
// downstream parses, and this is a Write with a whole file in it.
//
// Repaired without moving a single byte: `>` becomes the closing quote and the
// newline after it becomes the colon. Two single-character substitutions, so
// every source offset is preserved and the call spans stay valid -- the same
// constraint that governs the rest of the last-resort repair.
static void test_xml_bleed_into_json_key() {
    json tools = json::array({tool("Write", {{"file_path", true}, {"content", true}}),
                              tool("Bash", {{"command", true}})});
    auto call = [](const std::string& t, const json& tl) {
        std::string pre;
        return q27::parse_bare_tool_calls(t, &pre, &tl);
    };
    {
        auto v = call("{\"name\": \"Write\", \"arguments>\n"
                      "{\"file_path\": \"/w/index.ts\", \"content\": \"hello\"}",
                      tools);
        ok(v.size() == 1 && v[0].name == "Write" &&
               v[0].arguments.value("file_path", std::string()) == "/w/index.ts" &&
               v[0].arguments.value("content", std::string()) == "hello",
           "xml bleed: \"arguments> recovers as \"arguments\":");
    }
    {
        // well-formed JSON must be untouched
        auto v = call("{\"name\": \"Bash\", \"arguments\": {\"command\": \"ls\"}}", tools);
        ok(v.size() == 1 && v[0].arguments.value("command", std::string()) == "ls",
           "xml bleed: well-formed JSON unchanged");
    }
    {
        // a '>' inside a legitimate string VALUE must not be rewritten
        auto v = call("{\"name\": \"Bash\", \"arguments\": {\"command\": \"echo a > b\"}}", tools);
        ok(v.size() == 1 && v[0].arguments.value("command", std::string()) == "echo a > b",
           "xml bleed: a '>' inside a value is left alone");
    }
}


// A PARTIAL opener as the whole answer (2026-08-22, the last miss standing in
// the parity arm). The model emitted `<tool_call` -- no '>', no body -- and
// stopped. There is no call in that: no name, no arguments, nothing to execute.
//
// But it became the user's entire visible message, which is the same failure
// #32 fixed for the COMPLETE marker: a turn whose whole answer is a raw
// protocol fragment is worse than an empty one. only_dialect_control_bytes()
// knew `<tool_call>` and `</tool_call>` and not a truncated prefix of either.
//
// TO BE CLEAR ABOUT WHAT THIS IS: suppression, not recovery. The turn is still
// lost and the call still never happened. It stops the fragment reaching the
// user and stops it being counted as a parse failure it never was.
static void test_partial_opener_is_residue() {
    ok(q27::only_dialect_control_bytes("<tool_call"),
       "partial opener: a truncated <tool_call is content-free residue");
    ok(q27::only_dialect_control_bytes("\n<tool_call>\n<tool_ca"),
       "partial opener: a complete marker plus a truncated one");
    ok(q27::only_dialect_control_bytes("</tool_cal"),
       "partial opener: a truncated closer too");
    // ---- and what must still reach the user ----
    ok(!q27::only_dialect_control_bytes("<tool_call>\n<function=Bash>"),
       "partial opener: a body with a payload tag is NOT residue");
    ok(!q27::only_dialect_control_bytes("here is the answer"),
       "partial opener: prose is not residue");
    ok(!q27::only_dialect_control_bytes("<tool_call> and then some words"),
       "partial opener: a marker followed by prose is not residue");
    ok(!q27::only_dialect_control_bytes("   "),
       "partial opener: pure whitespace is still not this function's business");
    ok(!q27::only_dialect_control_bytes("<t"),
       "partial opener: a fragment too short to identify is not residue");
}


// THE TOOL NAME AS THE VALUE OF A PARAMETER CALLED `name` (2026-08-22, two
// live misses in the fp16 arm, both this shape):
//
//   <parameter=name>
//   Bash</parameter>
//   <parameter=command>
//   npx tsx scripts/perf-check.ts
//   </parameter>
//   </function>
//
// An earlier test for this spelling passed, and passed for the wrong reason:
// its toy schema had one tool, so key INFERENCE over {name, command} landed on
// Bash. Under Claude Code's real registry {command} fits Bash AND Monitor, the
// foreign `name` key counts equally against both, and infer_tool_name refuses
// the tie. The model wrote the tool's name; nothing read it.
//
// Rule: a parameter literally named `name` whose value names a DECLARED tool
// is the tool, and is removed from the arguments. Reading a name the model
// wrote is strictly less guessing than inferring one, and it is the same rule
// modes 18/22 already apply one tag over.
static void test_name_parameter_is_the_tool() {
    json twins = json::array({tool("Bash", {{"command", true}, {"description", false}, {"timeout", false}}),
                              tool("Monitor", {{"command", true}, {"timeout", false}}),
                              tool("Read", {{"file_path", true}, {"limit", false}})});
    auto call = [](const std::string& t, const json& tl) {
        std::string pre;
        return q27::parse_bare_tool_calls(t, &pre, &tl);
    };
    {   // live shape 1: closer on the same line as the name
        auto v = call("<parameter=name>\nBash</parameter>\n<parameter=command>\n"
                      "npx tsx scripts/perf-check.ts 2>&1 | tail -20\n</parameter>\n</function>", twins);
        ok(v.size() == 1 && v[0].name == "Bash" &&
               v[0].arguments.value("command", std::string()).rfind("npx tsx", 0) == 0 &&
               !v[0].arguments.contains("name"),
           "name param: Bash read from <parameter=name> despite the Monitor twin, `name` key removed");
    }
    {   // live shape 2: name on its own line
        auto v = call("<parameter=name>\nRead\n</parameter>\n<parameter=file_path>\n"
                      "/workspace/src/index.ts\n</parameter>\n</function>", twins);
        ok(v.size() == 1 && v[0].name == "Read" &&
               v[0].arguments.value("file_path", std::string()) == "/workspace/src/index.ts",
           "name param: Read read from <parameter=name>");
    }
    {   // inside a wrapper, the live delivery
        auto out = resolve_stream("<tool_call>\n<parameter=name>\nBash</parameter>\n<parameter=command>\nls\n"
                                  "</parameter>\n</function>\n</tool_call>", &twins);
        ok(out.calls.size() == 1 && out.calls[0].name == "Bash",
           "name param: recovered from a TOOL segment");
    }
    // ---- refusals ----
    {   // a `name` value that is not a declared tool stays an ordinary parameter
        // and the call falls back to inference as before
        json one = json::array({tool("Bash", {{"command", true}})});
        auto v = call("<parameter=name>\nnot a tool\n</parameter>\n<parameter=command>\nls\n</parameter>\n</function>", one);
        ok(v.size() == 1 && v[0].name == "Bash" && v[0].arguments.value("name", std::string()) == "not a tool",
           "name param: a non-tool value is left as a parameter");
    }
    {   // and with the twins present that same case still refuses the tie
        auto v = call("<parameter=name>\nnot a tool\n</parameter>\n<parameter=command>\nls\n</parameter>\n</function>", twins);
        ok(v.empty(), "name param: non-tool value + ambiguous keys still refuses");
    }
    {   // a declared tool that itself has a `name` parameter must not lose it
        json named = json::array({tool("Rename", {{"name", true}, {"path", true}}),
                                  tool("Bash", {{"command", true}})});
        auto v = call("<function=Rename>\n<parameter=name>\nBash\n</parameter>\n<parameter=path>\n/x\n</parameter>\n</function>", named);
        ok(v.size() == 1 && v[0].name == "Rename" && v[0].arguments.value("name", std::string()) == "Bash",
           "name param: an explicit <function=> opener keeps `name` as an argument");
    }
}

// N7. tool_strict() is a process-lifetime memo, so the strict leg cannot share
// a run with the tests above; main() dispatches on the env var and `make
// test-tools` invokes the binary a second time with Q27_TOOL_STRICT=1.
static void test_tool_segment_strict_leg() {
    json tools = json::array({tool("Read", {{"file_path", true}})});
    const std::string raw =
        "<tool_call>\n<tool_name>Read</tool_name>\n"
        "<parameter=file_path>/w/x.ts</parameter>\n</tool_call>";
    auto out = resolve_stream(raw, &tools);
    ok(out.calls.empty() && out.text.find("<tool_name>") != std::string::npos,
       "strict: a refused TOOL segment is NOT routed through the drift chain");
}

// Live failure 2026-08-22: after 8 clean tool calls the model emitted the
// OPENER TWICE and hit EOS. The splitter consumes the first (structural) and
// enters the TOOL channel; inside TOOL only "</tool_call>" delimits, so the
// second opener is BODY CONTENT. unfinished_tool_wrapper() reports false (it
// detects length/budget truncation, not EOS-inside-wrapper), so the tail path
// never runs, and the fallback printed ToolCall::raw -- the literal string
// "<tool_call>" became the user's entire visible answer for the turn.
static void test_repeated_opener_is_not_shown_as_text() {
    json tools = json::array({tool("bash", {{"command", true}})});

    for (const char* raw : {"<tool_call><tool_call>",
                            "<tool_call>\n<tool_call>",
                            "<tool_call>\n\n<tool_call>\n",
                            "<tool_call><tool_call><tool_call>"}) {
        auto out = resolve_stream(raw, &tools);
        ok(out.calls.empty() && out.text.find("<tool_call>") == std::string::npos,
           "repeated opener: no call, and the marker is not shown as text");
    }

    // The predicate itself. NARROW on purpose: dialect tags that can carry a
    // payload are NOT control-only, because a body containing them is drift
    // whose bytes the user may need to see.
    ok(q27::only_dialect_control_bytes("<tool_call>"), "control-only: bare opener");
    ok(q27::only_dialect_control_bytes(" \n</tool_call>\t"), "control-only: closer + ws");
    ok(q27::only_dialect_control_bytes("<tool_call>\n</tool_call>"),
       "control-only: opener + closer");
    ok(!q27::only_dialect_control_bytes("   \n\t "),
       "control-only: pure whitespace is NOT claimed");
    ok(!q27::only_dialect_control_bytes("<function=bash>"),
       "control-only: a payload-bearing dialect tag is drift, stays visible");
    ok(!q27::only_dialect_control_bytes("<tool_call>{\"name\":\"bash\"}"),
       "control-only: real content alongside a marker stays visible");

    // A malformed body that is NOT control-only must still reach the user.
    {
        auto out = resolve_stream("<tool_call>\n{\"nam\": \"bas\", oops\n</tool_call>", &tools);
        ok(out.calls.empty() && out.text.find("oops") != std::string::npos,
           "malformed body still visible (suppression did not overreach)");
    }
    // and a well-formed call is unaffected
    {
        auto out = resolve_stream(
            "<tool_call>\n{\"name\":\"bash\",\"arguments\":{\"command\":\"ls\"}}\n</tool_call>", &tools);
        ok(out.calls.size() == 1 && out.calls[0].name == "bash",
           "well-formed wrapped call unaffected by the suppression");
    }
}

int main() {
    if (getenv("Q27_TOOL_STRICT")) {
        test_tool_segment_strict_leg();
        printf(failures ? "\ntest_tool_drift(strict): %d FAILURE(S)\n"
                        : "\ntest_tool_drift(strict): ALL PASS\n", failures);
        return failures ? 1 : 0;
    }
    test_tool_segment_reaches_drift_chain();
    test_tool_segment_drift_edges();
    test_survey_captures_wrapped();
    test_json_fused_opener_shapes();
    test_mode22_parameter_as_opener();
    test_named_opener_batch();
    test_parameter_value_fidelity();
    test_unterminated_batch_no_function_closer();
    test_parameter_attribute_spelling();
    test_parameter_attribute_reaches_drift_modes();
    test_value_containing_closer();
    test_last_parameter_garbled_tail();
    test_aborted_second_call();
    test_batch_mixed_opener_spellings();
    test_zero_arg_mode22_call();
    test_parameter_name_opener();
    test_json_terminator_in_xml_dialect();
    test_placeholder_name_on_next_line();
    test_xml_bleed_into_json_key();
    test_partial_opener_is_residue();
    test_name_parameter_is_the_tool();
    test_intended_tool_call_detector();
    test_unclosed_tool_tail_recovery();
    test_mode20_multicall_span_stamping();
    test_mode13_truncated_mid_escape();
    test_function_arguments_unterminated();
    test_think_mode_drift();
    test_dialect_default_keying();
    test_preamble_swaps_with_dialect();
    test_reasoning_effort_line();
    test_native_xml_dialect();
    test_mode18_bare_name_opener();
    test_mode19_attribute_opener();
    test_mode20_nameless_tool_name();
    test_mode21_openerless_params();
    test_mode21_guard_scoped_to_span();
    test_quoted_name_normalization();
    test_mode14_tool_name_xml_dialect();
    test_mode15_name_tag_bare_args();
    test_mode16_and_15_variants();
    test_repeated_opener_is_not_shown_as_text();
    test_hallucinated_result_enclosure_rule();
    json tools = json::array();
    tools.push_back(tool("Write", {{"content", true}, {"file_path", true}}));
    tools.push_back(tool("Read", {{"file_path", true}}));

    auto call = [&](const std::string& txt) {
        std::string pre;
        return q27::parse_bare_tool_calls(txt, &pre, &tools);
    };

    // mode 10: dropped `{"name": "` opener
    {
        auto v = call("prose.\n\nRead\", \"file_path\": \"/x/y.py\"}");
        ok(v.size() == 1 && v[0].name == "Read" &&
               v[0].arguments.value("file_path", std::string()) == "/x/y.py",
           "mode10 dropped-opener");
    }
    // mode 11: raw code, unescaped inner quotes, content last
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"content\": \"package main\n"
                      "import \"fmt\"\nfunc main(){ fmt.Println(\"hi\") }\n\"}}");
        ok(v.size() == 1 && v[0].name == "Write" &&
               v[0].arguments.value("content", std::string()).find("fmt.Println") !=
                   std::string::npos,
           "mode11 raw-content-last");
    }
    // mode 11: inner []string{"a","b"} + a scalar AFTER content
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"content\": \"a := []string{\"x\", "
                      "\"y\"}\nfmt.Println(a)\n\", \"file_path\": \"m.go\"}}");
        ok(v.size() == 1 && v[0].name == "Write" &&
               v[0].arguments.value("file_path", std::string()) == "m.go" &&
               v[0].arguments.value("content", std::string()).find("[]string") !=
                   std::string::npos,
           "mode11 inner-braces + scalar-after");
    }
    // mode 11: scalar BEFORE content
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"file_path\": \"m.go\", \"content\": "
                      "\"func f(){ s := \"hi\" }\n\"}}");
        ok(v.size() == 1 && v[0].arguments.value("file_path", std::string()) == "m.go" &&
               !v[0].arguments.value("content", std::string()).empty(),
           "mode11 scalar-before-content");
    }
    // mode 11 refinement (issue #4, 2026-07-20, @chaudhryfaisal): content is
    // MOSTLY-escaped JSON (\n \" all escaped) with ONE sparse escape error
    // (\"x" -- bare closing quote) AND a trailing </tool_call>. json().dump()
    // would double-escape the already-escaped body, and the trailing tag broke
    // the reconstruction's parse -> UN-RESCUED. minimal-escape + first-balanced-
    // object recover it with the CORRECT content ("fmt" quotes preserved).
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"content\":\"package main\\n"
                      "import \\\"fmt\\\"\\nvar s = \\\"x\"\\n\",\"file_path\":\"m.go\"}}\n"
                      "</tool_call>");
        ok(v.size() == 1 && v[0].name == "Write" &&
               v[0].arguments.value("file_path", std::string()) == "m.go" &&
               v[0].arguments.value("content", std::string()).find("\"fmt\"") != std::string::npos,
           "mode11 mostly-escaped content + trailing tag");
    }
    // mode 12: unquoted tool-name value {"name": Read, "arguments": {...}}
    // (club-3090 cli-40: the model emitted {"name": bash, ...} and the whole
    // call went UN-RESCUED -> agent turn stopped at turnsUsed=0).
    {
        auto v = call("{\"name\": Read, \"arguments\": {\"file_path\": \"/x/y.py\"}}");
        ok(v.size() == 1 && v[0].name == "Read" &&
               v[0].arguments.value("file_path", std::string()) == "/x/y.py",
           "mode12 unquoted-name");
    }
    // mode 12b: dropped OPENING quote of the name value -> {"name": Read", ...}
    // (thunderdome 2026-07-20: model emitted {"name": read", ...} -- bareword +
    // stray closing quote; naive quoting would make "Read"" (invalid)).
    {
        auto v = call("{\"name\": Read\", \"arguments\": {\"file_path\": \"/x/y.py\"}}");
        ok(v.size() == 1 && v[0].name == "Read" &&
               v[0].arguments.value("file_path", std::string()) == "/x/y.py",
           "mode12b dropped-opening-quote");
    }
    // mode 12 negative: an unquoted name that is NOT a registered tool must be
    // left untouched (never quote arbitrary barewords).
    {
        auto v = call("{\"name\": notatool, \"arguments\": {\"file_path\": \"/x\"}}");
        ok(v.empty(), "mode12 unknown unquoted-name rejected");
    }
    // negative: well-formed call recovers via the normal path (not a drift mode)
    {
        auto v =
            call("{\"name\": \"Write\", \"arguments\": {\"file_path\": \"a.txt\", \"content\": "
                 "\"hello\"}}");
        ok(v.size() == 1 && v[0].arguments.value("content", std::string()) == "hello",
           "wellformed via normal path");
    }
    // negative: prose JSON with an unregistered "name" must NOT recover
    {
        auto v = call("config: {\"name\": \"my-app\", \"version\": \"1.0\"} shipped.");
        ok(v.empty(), "prose-unknown-name rejected");
    }
    // fence-skip (thunderdome 2026-07-20): a COMPLETE, well-formed call inside a
    // ```fenced``` block is a displayed example / echoed injection, NOT a call
    // the model is making -> must not recover (prose-to-execution guard).
    {
        auto v = call("Here is how it works:\n```json\n{\"name\": \"Read\", \"arguments\": "
                      "{\"file_path\": \"/etc/passwd\"}}\n```\nThat is the format.");
        ok(v.empty(), "fence-skip: fenced example not recovered");
    }
    // but a write whose CONTENT contains fences still recovers (its ``` are
    // after the call's opener, so the guard -- which looks only before -- ignores them)
    {
        auto v = call("{\"name\": \"Write\", \"arguments\": {\"content\": \"# doc\\n```go\\nx := 1\\n"
                      "```\\n\", \"file_path\": \"/x.md\"}}");
        ok(v.size() == 1 && v[0].name == "Write" &&
               v[0].arguments.value("file_path", std::string()) == "/x.md",
           "fence-skip: write w/ fenced content still recovers");
    }

    printf(failures ? "\nDRIFT TESTS: %d FAIL\n" : "\nDRIFT TESTS: all pass\n", failures);
    return failures ? 1 : 0;
}
