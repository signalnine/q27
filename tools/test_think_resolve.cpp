// CPU unit test for resolve_think (api_common.h): per-request thinking opt-in.
// The server profile sets the DEFAULT (server_default = !no_think_srv); an
// explicit request field overrides it in either direction, across all three
// client conventions. Malformed fields must be ignored, never thrown.
//
// Build+run (no CUDA): g++ -std=c++17 -I src tools/test_think_resolve.cpp -o build/test_think_resolve && ./build/test_think_resolve
#include "api_common.h"
#include <cstdio>

using json = nlohmann::json;

static int failures = 0;
static void ok(bool c, const char* n) {
    printf("  %s %s\n", c ? "PASS" : "FAIL", n);
    if (!c) failures++;
}

int main() {
    using q27::resolve_think;

    // --- server default honored when the request says nothing ---
    ok(resolve_think(json::object(), false) == false, "no-think server + silent -> no-think");
    ok(resolve_think(json::object(), true) == true, "--think server + silent -> think");

    // --- OpenAI/Qwen top-level enable_thinking overrides in BOTH directions ---
    ok(resolve_think(json{{"enable_thinking", true}}, false) == true,
       "no-think server + enable_thinking:true -> think (THE opt-in)");
    ok(resolve_think(json{{"enable_thinking", false}}, true) == false,
       "--think server + enable_thinking:false -> no-think");

    // --- llama.cpp/GLM nested chat_template_kwargs.enable_thinking ---
    ok(resolve_think(json{{"chat_template_kwargs", {{"enable_thinking", true}}}}, false) == true,
       "nested kwargs true -> think");
    ok(resolve_think(json{{"chat_template_kwargs", {{"enable_thinking", false}}}}, true) == false,
       "nested kwargs false -> no-think");

    // --- Anthropic native thinking field (what Claude Code's toggle emits) ---
    ok(resolve_think(json{{"thinking", {{"type", "enabled"}}}}, false) == true,
       "anthropic thinking enabled -> think");
    ok(resolve_think(json{{"thinking", {{"type", "disabled"}}}}, true) == false,
       "anthropic thinking disabled -> no-think");
    ok(resolve_think(json{{"thinking", {{"type", "enabled"}, {"budget_tokens", 2000}}}}, false) == true,
       "anthropic thinking enabled + budget_tokens ignored -> think");

    // --- malformed / wrong-typed fields: never throw, leave the default in force ---
    ok(resolve_think(json{{"enable_thinking", "yes"}}, false) == false,
       "string enable_thinking ignored -> default");
    ok(resolve_think(json{{"enable_thinking", 1}}, true) == true,
       "int enable_thinking ignored -> default");
    ok(resolve_think(json{{"thinking", "enabled"}}, false) == false,
       "non-object thinking ignored -> default");
    ok(resolve_think(json{{"thinking", {{"type", 3}}}}, true) == true,
       "non-string thinking.type ignored -> default");
    ok(resolve_think(json{{"chat_template_kwargs", "x"}}, false) == false,
       "non-object chat_template_kwargs ignored -> default");
    ok(resolve_think(json{{"thinking", {{"type", "bogus"}}}}, false) == false,
       "unknown thinking.type leaves default (false)");
    ok(resolve_think(json{{"thinking", {{"type", "bogus"}}}}, true) == true,
       "unknown thinking.type leaves default (true)");

    printf(failures ? "\nTHINK-RESOLVE: %d FAIL\n" : "\nTHINK-RESOLVE: all pass\n", failures);
    return failures ? 1 : 0;
}
