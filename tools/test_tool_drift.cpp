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

int main() {
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

    printf(failures ? "\nDRIFT TESTS: %d FAIL\n" : "\nDRIFT TESTS: all pass\n", failures);
    return failures ? 1 : 0;
}
