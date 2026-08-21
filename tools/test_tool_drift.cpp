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
    unsetenv("Q27_REASONING_EFFORT");
    unsetenv("Q27_TOOL_DIALECT");
    auto meta = [](const char* n) { return std::string("{\"general.name\": \"") + n + "\"}"; };
    std::vector<q27::Msg> msgs = {{"system", "Be terse."}, {"user", "hi"}};
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
