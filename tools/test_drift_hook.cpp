// The Q27_DRIFT_CORPUS hook sites in api_common.h: which resolver branch
// records which outcome, with which tags, exactly once per turn, and never a
// value byte. Complements tools/test_drift_capture.cpp (the redactor itself).
// Everything here toggles the env var at runtime, so the same binary also
// proves the default (unset) path writes nothing and parses identically.
//
// Build+run: g++ -std=c++17 -I src tools/test_drift_hook.cpp -o build/test_drift_hook && ./build/test_drift_hook
#include "api_common.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

using json = nlohmann::json;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("  FAIL %s:%d %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

static const char* kPath = "build/test_drift_hook.jsonl";
static bool verbose = false;   // -v: dump every record a case produced
using Seg = std::pair<q27::StreamSplitter::Chan, std::string>;

// The shape the server hands the parser (anthropic_tools_json / openai_tools_json
// both land here): OpenAI-style, with `required`, which key inference needs.
static json tools() {
    return json::parse(R"([
      {"type":"function","function":{"name":"Read","parameters":{"type":"object",
        "properties":{"file_path":{"type":"string"}},"required":["file_path"]}}},
      {"type":"function","function":{"name":"Bash","parameters":{"type":"object",
        "properties":{"command":{"type":"string"}},"required":["command"]}}}])");
}

static size_t resolve(const std::vector<Seg>& segs) {
    json t = tools();
    auto out = q27::resolve_ordered_tool_segments(segs, &t, true,
                                                  [](const std::string&, size_t) { return true; });
    return out.calls.size();
}

// Run one turn with capture on; return the records it produced.
static std::vector<json> capture(const std::vector<Seg>& segs, size_t* calls = nullptr) {
    remove(kPath);
    setenv("Q27_DRIFT_CORPUS", kPath, 1);
    const size_t n = resolve(segs);
    unsetenv("Q27_DRIFT_CORPUS");
    if (calls) *calls = n;
    std::vector<json> recs;
    std::ifstream f(kPath);
    for (std::string line; std::getline(f, line);)
        if (!line.empty()) recs.push_back(json::parse(line));
    remove(kPath);
    if (verbose) {
        printf("  turn %s -> calls=%zu records=%zu\n", segs.empty() ? "" : segs[0].second.substr(0, 40).c_str(), n, recs.size());
        for (const auto& r : recs)
            printf("    %-14s %-28s %s\n", r["outcome"].get<std::string>().c_str(),
                   r["tags"].dump().c_str(), r["redacted"].dump().c_str());
    }
    return recs;
}

static bool has_tag(const json& r, const char* t) {
    for (const auto& x : r["tags"]) if (x == t) return true;
    return false;
}
static bool no_secret(const std::vector<json>& recs) {
    for (const auto& r : recs)
        if (r["redacted"].get<std::string>().find("secret") != std::string::npos) return false;
    return true;
}

static void test_think_branch_tags_in_think() {
    auto recs = capture({{q27::StreamSplitter::THINK,
        "planning\n<tool_call>\n<function=Read>\n<parameter=file_path>\n/secret/a\n</parameter>\n</function>\n</tool_call>\n"}});
    CHECK(recs.size() == 1);
    if (recs.empty()) return;
    CHECK(has_tag(recs[0], "in_think"));
    CHECK(has_tag(recs[0], "xml"));
    CHECK(no_secret(recs));
}

static void test_wrapped_strict_call_is_strict_and_not_no_wrapper() {
    auto recs = capture({{q27::StreamSplitter::TOOL, R"({"name":"Read","arguments":{"file_path":"/secret/b"}})"}});
    CHECK(recs.size() == 1);
    if (recs.empty()) return;
    CHECK(recs[0]["outcome"] == "strict");
    CHECK(recs[0]["tags"] == json::array({"json"}));   // the splitter ate the wrapper; not no_wrapper
    CHECK(no_secret(recs));
}

static void test_wrapped_drift_goes_through_chain_as_wrapped() {
    // exceeds-one-call bodies bypass the strict parser and reach the bare
    // chain with the wrapper already stripped; the record must still say so
    auto recs = capture({{q27::StreamSplitter::TOOL,
        "{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/secret/c\"}}\n"
        "{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/secret/d\"}}"}});
    CHECK(recs.size() == 1);
    if (recs.empty()) return;
    CHECK(recs[0]["outcome"].get<std::string>().rfind("recovered:", 0) == 0);
    CHECK(!has_tag(recs[0], "no_wrapper"));
    CHECK(no_secret(recs));
}

static void test_early_return_mode_is_named() {
    // mode 21 (openerless parameter list) returns from the impl at its own
    // site, not through the final modes block; the hint must still name it
    auto recs = capture({{q27::StreamSplitter::TEXT,
        "I will now call <parameter=file_path>/secret/e</parameter></function> ok"}});
    CHECK(recs.size() == 1);
    if (recs.empty()) return;
    CHECK(recs[0]["outcome"] == "recovered:21");
    CHECK(has_tag(recs[0], "no_wrapper"));
    CHECK(no_secret(recs));
}

static void test_reasoning_without_a_candidate_is_still_recorded() {
    // passes looks_like_intended_tool_call, but the holdback finds nothing to
    // classify so the parser never runs; the segment must still be a record
    // (outcome deliberately unasserted: a parser fix may one day recover it)
    auto recs = capture({{q27::StreamSplitter::THINK,
        "<tool_call>{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/secret/j\"}}</tool_call>"}});
    CHECK(recs.size() == 1);
    if (recs.empty()) return;
    CHECK(has_tag(recs[0], "in_think"));
    CHECK(no_secret(recs));
}

static void test_control_bytes_are_suppressed() {
    auto recs = capture({{q27::StreamSplitter::TOOL, "<tool_call>\n<tool_call>"}});
    CHECK(recs.size() == 1);
    if (recs.empty()) return;
    CHECK(recs[0]["outcome"] == "suppressed");
}

static void test_undeclared_name_is_redacted() {
    auto recs = capture({{q27::StreamSplitter::TEXT, R"({"name":"Alice","arguments":{"file_path":"/secret/f"}})"}});
    CHECK(recs.size() == 1);
    if (recs.empty()) return;
    const std::string red = recs[0]["redacted"];
    CHECK(red.find("Alice") == std::string::npos);
    CHECK(red.find("NAME_1") != std::string::npos);
}

static void test_reentrant_modes_record_once() {
    // mode 10 re-enters parse_bare_tool_calls on a synthesised probe; the
    // depth guard must keep that to one record for the turn
    size_t calls = 0;
    auto recs = capture({{q27::StreamSplitter::TEXT, "prose.\n\nRead\", \"file_path\": \"/secret/g\"}"}}, &calls);
    CHECK(calls == 1);
    CHECK(recs.size() == 1);
    CHECK(no_secret(recs));
}

static void test_residue_below_the_warning_bar_is_still_captured() {
    // an arguments object with no name has no {"name" and no </function>, so
    // looks_like_intended_tool_call stays quiet (by design) and nothing can
    // recover it; the corpus must still see it
    size_t calls = 0;
    auto recs = capture({{q27::StreamSplitter::TEXT, "\"arguments\": {\"file_path\": \"/secret/h\"}"}}, &calls);
    CHECK(calls == 0);
    CHECK(recs.size() == 1);
    if (recs.empty()) return;
    CHECK(recs[0]["outcome"] == "unrescued");
    CHECK(no_secret(recs));
    // but plain prose is not a record, even quoting a tool name
    auto none = capture({{q27::StreamSplitter::TEXT, "Just a sentence about /secret/i and nothing else."}});
    CHECK(none.empty());
    auto quoted = capture({{q27::StreamSplitter::TEXT, "I will Read\" the file and Bash\", then stop."}});
    CHECK(quoted.empty());
}

static void test_strict_miss_hint_is_one_shot() {
    // a wrapped body that fails the strict parser tags the NEXT bare parse of
    // the same bytes as wrapped (the streaming handlers' order), and only that
    // one: an unrelated turn later must not inherit it
    remove(kPath);
    setenv("Q27_DRIFT_CORPUS", kPath, 1);
    // the wrapped mode-10 shape: strict JSON fails, the bare chain recovers
    const std::string body = "Read\", \"file_path\": \"/secret/k\"}";
    CHECK(!q27::parse_tool_call(body).ok);
    json t = tools();
    std::string pre;
    q27::parse_bare_tool_calls(body, &pre, &t);                 // the handler's next step
    q27::parse_bare_tool_calls(body, &pre, &t);                 // a later, unrelated turn
    unsetenv("Q27_DRIFT_CORPUS");
    std::vector<json> recs;
    std::ifstream f(kPath);
    for (std::string line; std::getline(f, line);) if (!line.empty()) recs.push_back(json::parse(line));
    remove(kPath);
    CHECK(recs.size() == 2);
    if (recs.size() != 2) return;
    CHECK(!has_tag(recs[0], "no_wrapper"));
    CHECK(has_tag(recs[1], "no_wrapper"));
    CHECK(no_secret(recs));
}

static void test_batch_records_once_per_turn() {
    auto recs = capture({{q27::StreamSplitter::TEXT,
        "<tool_call>\n<function=Read>\n<parameter=file_path>\n/secret/h\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=Read>\n<parameter=file_path>\n/secret/i\n</parameter>\n</function>\n</tool_call>"}});
    CHECK(recs.size() == 1);
    CHECK(no_secret(recs));
}

static void test_unset_writes_nothing_and_parses_the_same() {
    const std::vector<std::vector<Seg>> turns = {
        {{q27::StreamSplitter::TOOL, R"({"name":"Read","arguments":{"file_path":"/x"}})"}},
        {{q27::StreamSplitter::TEXT, "Read\", \"file_path\": \"/x\"}"}},
        {{q27::StreamSplitter::TEXT, "<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>"}},
        {{q27::StreamSplitter::THINK, "<tool_call>{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/x\"}}</tool_call>"}},
    };
    for (const auto& t : turns) {
        remove(kPath);
        unsetenv("Q27_DRIFT_CORPUS");
        const size_t off = resolve(t);
        CHECK(!std::ifstream(kPath).good());          // nothing written
        size_t on = 0;
        capture(t, &on);
        CHECK(on == off);                              // capture never changes a parse
    }
}

static void test_request_tool_list_feeds_the_registry() {
    // the streaming handlers reach parse_tool_call with no tool list; the
    // request parser must have registered the declared names by then, or a
    // well-formed streaming call loses its tool identity (seen live: Bash and
    // Edit as NAME_1 while a name from an earlier non-stream request survived)
    auto strict = [](const std::string& body) {
        remove(kPath);
        setenv("Q27_DRIFT_CORPUS", kPath, 1);
        q27::parse_tool_call(body);
        unsetenv("Q27_DRIFT_CORPUS");
        std::ifstream f(kPath);
        std::string line;
        std::getline(f, line);
        remove(kPath);
        return line.empty() ? json::object() : json::parse(line);
    };
    const std::string call = R"({"name":"mcp__ledger__open","arguments":{"name":"savings"}})";
    json before = strict(call);
    CHECK(before["outcome"] == "strict");
    CHECK(before["redacted"].get<std::string>().find("mcp__ledger__open") == std::string::npos);
    setenv("Q27_DRIFT_CORPUS", kPath, 1);
    q27::anthropic_tools_json(json::parse(R"({"tools":[{"name":"mcp__ledger__open",
        "input_schema":{"type":"object","properties":{"name":{"type":"string"}}}}]})"));
    unsetenv("Q27_DRIFT_CORPUS");
    json after = strict(call);
    CHECK(after["redacted"].get<std::string>().find("\"name\":\"mcp__ledger__open\"") != std::string::npos);
    CHECK(after["redacted"].get<std::string>().find("savings") == std::string::npos);  // the argument named `name` is a value
}

static void test_streaming_text_miss_without_a_candidate_is_recorded() {
    // seen live (pylint-6903, 2026-08-25): a dialect shape no opener matches,
    // so the holdback never invokes the parser and the client gets it as
    // text. The corpus exists to record exactly this; the router must not
    // depend on the parser having run.
    const std::string gen =
        "\n\n<tool_use>\n<tool>\n<parameter_name>\n<parameter_name>Read\n</parameter>\n"
        "<parameter=file_path>\n/secret/run.py\n";
    json t = tools();
    std::set<std::string> names{"Read", "Bash"};
    remove(kPath);
    setenv("Q27_DRIFT_CORPUS", kPath, 1);
    {
        q27::StreamSplitter sp;
        q27::StreamToolRouter r;
        r.has_tools = true;
        std::string text, think;
        std::vector<q27::ToolCall> calls;
        auto emit_text = [&](const std::string& s) { text += s; };
        auto emit_think = [&](const std::string& s) { think += s; };
        auto classify = [&](const std::string& src, bool rep, auto&& visible) {
            std::string pre, residual;
            auto cs = q27::parse_bare_tool_calls(src, &pre, &t, true, rep, &residual);
            if (cs.empty()) return q27::BareToolCandidateResult{};
            for (auto& c : cs) calls.push_back(c);
            visible(std::string());
            return q27::BareToolCandidateResult{true, true};
        };
        auto emit_tool = [&]() { r.tool_buf.clear(); };
        for (size_t i = 0; i < gen.size(); i += 7)
            for (auto& [ch, s] : sp.feed(gen.substr(i, 7)))
                r.segment(ch, s, false, names, emit_text, emit_think, emit_tool, classify);
        for (auto& [ch, s] : sp.flush()) r.segment(ch, s, false, names, emit_text, emit_think, emit_tool, classify);
        r.finish(true, names, emit_text, emit_think, classify);
        // Since 2026-09-08 the holdback arms on the `<tool_use>` wrapper, so
        // the stream recovers this shape (it used to leave it as text and
        // record it unrescued at finish). The record now comes from the parse
        // wrapper's success path; still exactly one, still redacted.
        CHECK(calls.size() == 1);
    }
    unsetenv("Q27_DRIFT_CORPUS");
    std::vector<json> recs;
    std::ifstream f(kPath);
    for (std::string line; std::getline(f, line);) if (!line.empty()) recs.push_back(json::parse(line));
    remove(kPath);
    CHECK(recs.size() == 1);
    if (recs.empty()) return;
    CHECK(recs[0]["outcome"].get<std::string>().rfind("recovered:", 0) == 0);
    CHECK(has_tag(recs[0], "xml"));
    CHECK(no_secret(recs));
    CHECK(recs[0]["redacted"].get<std::string>().find("<parameter_name>") != std::string::npos);  // the shape survives
}

int main(int argc, char** argv) {
    verbose = argc > 1 && std::string(argv[1]) == "-v";
    test_think_branch_tags_in_think();
    test_wrapped_strict_call_is_strict_and_not_no_wrapper();
    test_wrapped_drift_goes_through_chain_as_wrapped();
    test_early_return_mode_is_named();
    test_reasoning_without_a_candidate_is_still_recorded();
    test_control_bytes_are_suppressed();
    test_undeclared_name_is_redacted();
    test_reentrant_modes_record_once();
    test_residue_below_the_warning_bar_is_still_captured();
    test_strict_miss_hint_is_one_shot();
    test_request_tool_list_feeds_the_registry();
    test_streaming_text_miss_without_a_candidate_is_recorded();
    test_batch_records_once_per_turn();
    test_unset_writes_nothing_and_parses_the_same();
    if (fails) { printf("%d FAILURE(S)\n", fails); return 1; }
    printf("DRIFT HOOK: all pass\n");
    return 0;
}
