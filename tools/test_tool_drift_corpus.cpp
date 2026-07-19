// Mutation harness for bare-tool-call recovery. The point: cross REALISTIC
// argument values (source files, JSON, markdown, unicode) against the drift
// TRANSFORMS a model applies, and assert recovery. This is the test class we
// were missing -- prior drift tests used trivial arg values ("hello"), so
// mode 11 (raw code bodies with unescaped inner quotes) hid in the value
// content and only surfaced in production (issue #4). Run this against any
// parser change; it would have gone red on the pre-mode-11 code.
//
// Build + run (no CUDA):
//   g++ -std=c++17 -I src tools/test_tool_drift_corpus.cpp -o build/test_tool_drift_corpus \
//     && ./build/test_tool_drift_corpus
#include "api_common.h"
#include <cstdio>
#include <string>
#include <vector>

using json = nlohmann::json;
static int fails = 0;
static void check(bool cond, const std::string& what) {
    if (!cond) { printf("  FAIL %s\n", what.c_str()); fails++; }
}

// realistic content values a Write/Edit call carries. the ones flagged
// no_lookalike must round-trip EXACTLY through every recoverable transform;
// the lookalike one embeds a `", "key": "` sequence, so recovery may
// under-capture -- we only require it not crash and stay a prefix.
struct Fixture { const char* name; std::string val; bool no_lookalike; };

int main() {
    std::vector<Fixture> fx = {
        {"go", "package main\n\nimport (\n\t\"context\"\n\t\"fmt\"\n)\n\nfunc main() {\n"
               "\tfmt.Println(\"hello\")\n}\n", true},
        {"python", "def greet(name):\n    \"\"\"Say hi.\"\"\"\n    return f\"hello {name}\"\n", true},
        {"json_blob", "{\"a\": 1, \"b\": [\"x\", \"y\"], \"c\": {\"d\": true}}", true},
        {"markdown", "# Title\n\nSome text.\n\n```go\nx := []string{\"a\", \"b\"}\n```\n", true},
        {"braces_quotes", "if (ok) { m[\"key\"] = \"val\"; arr[] = {1, 2}; }\n", true},
        {"unicode", "note: \xe2\x9c\x93 done, \"quoted\" and \xe2\x80\x94 dash\n", true},
        {"single_line", "const X = 42;", true},
        {"empty", "", true},
        {"lookalike", "prefix code\", \"file_path\": \"decoy.txt\ntrailing", false},
    };

    json tools = json::array();
    tools.push_back({{"type", "function"},
                     {"function",
                      {{"name", "Write"},
                       {"parameters",
                        {{"type", "object"},
                         {"properties", {{"content", {{"type", "string"}}},
                                         {"file_path", {{"type", "string"}}}}},
                         {"required", {{"content"}, {"file_path"}}}}}}}});

    auto parse = [&](const std::string& txt) {
        std::string pre;
        return q27::parse_bare_tool_calls(txt, &pre, &tools);
    };
    auto raw_str = [](const std::string& s) { return s; }; // emitted unescaped (the drift)

    for (auto& f : fx) {
        const std::string P = "src/" + std::string(f.name) + ".txt";

        // T1: well-formed (properly escaped) -- must recover EXACTLY, always.
        {
            json obj = {{"name", "Write"},
                        {"arguments", {{"content", f.val}, {"file_path", P}}}};
            auto v = parse(obj.dump());
            check(v.size() == 1 && v[0].name == "Write" &&
                      v[0].arguments.value("content", std::string()) == f.val &&
                      v[0].arguments.value("file_path", std::string()) == P,
                  std::string(f.name) + " T1 wellformed");
        }

        // T2: raw unescaped content, content-last (mode 11).
        {
            std::string txt = "{\"name\": \"Write\", \"arguments\": {\"content\": \"" +
                              raw_str(f.val) + "\"}}";
            auto v = parse(txt);
            std::string got = v.empty() ? "" : v[0].arguments.value("content", std::string());
            if (f.no_lookalike)
                check(v.size() == 1 && got == f.val, std::string(f.name) + " T2 raw content-last");
            else  // under-capture allowed, but must be a prefix and not crash
                check(v.size() <= 1 && f.val.rfind(got, 0) == 0,
                      std::string(f.name) + " T2 raw (lookalike, prefix ok)");
        }

        // T3: raw content + file_path AFTER content (ordering that breaks
        // last-value assumptions).
        if (f.no_lookalike) {
            std::string txt = "{\"name\": \"Write\", \"arguments\": {\"content\": \"" + f.val +
                              "\", \"file_path\": \"" + P + "\"}}";
            auto v = parse(txt);
            check(v.size() == 1 && v[0].arguments.value("content", std::string()) == f.val &&
                      v[0].arguments.value("file_path", std::string()) == P,
                  std::string(f.name) + " T3 raw scalar-after");
        }

        // T4: prose preamble + raw content (the real CC shape).
        if (f.no_lookalike) {
            std::string txt = "Let me write the file.\n\n{\"name\": \"Write\", \"arguments\": "
                              "{\"content\": \"" + f.val + "\"}}";
            auto v = parse(txt);
            check(v.size() == 1 && v[0].arguments.value("content", std::string()) == f.val,
                  std::string(f.name) + " T4 preamble + raw");
        }
    }

    printf(fails ? "\nCORPUS: %d FAIL\n" : "\nCORPUS: all pass (%zu fixtures)\n",
           fails ? fails : (int)fx.size());
    return fails ? 1 : 0;
}
