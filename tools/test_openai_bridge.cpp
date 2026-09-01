// CPU unit tests for the new /v1/chat/completions bridges in api_common.h:
// openai_tools_json, openai_msgs, parse_tool_choice. Pure header logic, no
// CUDA/engine dependency -- same rationale as test_toolconstrain.cpp.
//
// Build+run: g++ -std=c++17 -I src tools/test_openai_bridge.cpp -o build/test_openai_bridge && ./build/test_openai_bridge
#include "api_common.h"

#include <cassert>
#include <cstdio>
#include <set>

using json = nlohmann::json;
using q27::Msg;

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

static void test_tools_passthrough() {
    json body = {{"tools", json::array({
        {{"type", "function"}, {"function", {{"name", "get_weather"},
            {"description", "get weather"},
            {"parameters", {{"type","object"},{"properties", {{"location", {{"type","string"}}}}}}}}}},
        // malformed entries must be dropped, not throw / take down the request
        {{"type", "web_search"}},                      // hosted type, not "function"
        {{"type", "function"}},                         // missing "function" key
        {{"type", "function"}, {"function", json::object()}}, // missing name
        {{"type", nullptr}, {"function", {{"name", "null_type"}}}},
        {{"type", "function"}, {"function", {{"name", "null_description"},
            {"description", nullptr}}}},
        {{"type", "function"}, {"function", {{"name", "bad_parameters"},
            {"parameters", json::array()}}}},
        {{"type", "function"}, {"function", {{"name", ""}}}},
    })}};
    json tools = q27::openai_tools_json(body);
    CHECK(tools.is_array());
    CHECK(tools.size() == 1);
    CHECK(tools[0]["function"]["name"] == "get_weather");
    CHECK(tools[0]["type"] == "function");
}

static void test_tools_absent() {
    json body = json::object();
    json tools = q27::openai_tools_json(body);
    CHECK(tools.is_array());
    CHECK(tools.empty());
}

static void test_responses_tool_fields_reject_wrong_types() {
    auto rejected=[](json tool) {
        try {
            q27::validate_responses_tool_fields(
                json{{"tools",json::array({std::move(tool)})}});
            return false;
        } catch(const std::invalid_argument&) {
            return true;
        }
    };
    CHECK(rejected(json{{"type",nullptr}}));
    CHECK(rejected(json{{"type","function"},{"name",nullptr}}));
    CHECK(rejected(json{{"type","custom"},{"name","apply_patch"},
                        {"description",nullptr}}));
    CHECK(!rejected(json{{"type","function"},{"name","safe"},
                         {"description","safe tool"}}));
}

static void test_msgs_plain_roundtrip() {
    json body = {{"messages", json::array({
        {{"role","system"},{"content","be terse"}},
        {{"role","user"},{"content","hi"}},
    })}};
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 2);
    CHECK(msgs[0].role == "system" && msgs[0].content == "be terse");
    CHECK(msgs[1].role == "user" && msgs[1].content == "hi");
}

static void test_msgs_content_parts_array() {
    json body = {{"messages", json::array({
        {{"role","user"},{"content", json::array({
            {{"type","text"},{"text","part one "}},
            {{"type","image_url"},{"image_url", {{"url","http://x"}}}}, // non-text part ignored, not crash
            {{"type","text"},{"text","part two"}},
        })}},
    })}};
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 1);
    CHECK(msgs[0].content == "part one part two");
}

static void test_msgs_developer_role_maps_to_system() {
    json body = {{"messages", json::array({
        {{"role","developer"},{"content","dev system prompt"}},
    })}};
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 1);
    CHECK(msgs[0].role == "system");
}

static void test_msgs_assistant_tool_calls_reconstructed() {
    // OpenAI wire shape: function.arguments is a JSON-ENCODED STRING.
    json body = {{"messages", json::array({
        {{"role","user"},{"content","what's the weather in Tokyo?"}},
        {{"role","assistant"},
         {"content", nullptr},
         {"tool_calls", json::array({
             {{"id","call_1"},{"type","function"},
              {"function", {{"name","get_weather"},{"arguments","{\"location\":\"Tokyo\"}"}}}}
         })}},
        {{"role","tool"},{"tool_call_id","call_1"},{"content","72F and sunny"}},
        {{"role","user"},{"content","thanks, and tomorrow?"}},
    })}};
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 4);
    CHECK(msgs[0].role == "user");
    CHECK(msgs[1].role == "assistant");
    CHECK(msgs[1].content.find("<tool_call>") != std::string::npos);
    CHECK(msgs[1].content.find("\"name\": \"get_weather\"") != std::string::npos);
    CHECK(msgs[1].content.find("\"location\":\"Tokyo\"") != std::string::npos);
    // role:"tool" folds into a USER turn wrapped in <tool_response>
    CHECK(msgs[2].role == "user");
    CHECK(msgs[2].content.find("<tool_response>") != std::string::npos);
    CHECK(msgs[2].content.find("72F and sunny") != std::string::npos);
    CHECK(msgs[3].role == "user");
    CHECK(msgs[3].content == "thanks, and tomorrow?");
}

static void test_msgs_assistant_content_plus_tool_calls() {
    // real-world shape: assistant text AND a tool call in the same turn
    json body = {{"messages", json::array({
        {{"role","assistant"}, {"content","Let me check that."},
         {"tool_calls", json::array({
             {{"id","call_9"},{"type","function"},
              {"function", {{"name","search"},{"arguments","{\"q\":\"x\"}"}}}}
         })}},
    })}};
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 1);
    CHECK(msgs[0].content.rfind("Let me check that.", 0) == 0);
    CHECK(msgs[0].content.find("<tool_call>") != std::string::npos);
}

static void test_msgs_malformed_arguments_string_kept_not_dropped() {
    json body = {{"messages", json::array({
        {{"role","assistant"}, {"content", nullptr},
         {"tool_calls", json::array({
             {{"id","call_x"},{"type","function"},
              {"function", {{"name","broken"},{"arguments","not-json{{"}}}}
         })}},
    })}};
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 1);
    // must not throw, and the call must still be present in some form
    CHECK(msgs[0].content.find("<tool_call>") != std::string::npos);
    CHECK(msgs[0].content.find("broken") != std::string::npos);
}

static void test_msgs_no_messages_key() {
    json body = json::object();
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.empty());
}

static void test_msgs_content_less_message_no_crash() {
    // a message with no "content" key at all must not abort (json.hpp assertion)
    json body = {{"messages", json::array({ {{"role","user"}} })}};
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 1);
    CHECK(msgs[0].content.empty());
}

// ---- tolerant request-field readers (jnum/jint/jbool/jstr) ----------------
// json::value() throws type_error.302 on a PRESENT-but-null key, and httplib
// turns a handler throw into a 500 -- but {"max_tokens": null} is how a large
// share of OpenAI-compatible clients spell "unset". null / wrong-typed must
// read exactly like absent.
static void test_jread_null_reads_as_absent() {
    json body = json::parse(R"({"max_tokens":null,"stream":null,"temperature":null,
                                "top_p":null,"prompt":null})");
    CHECK(q27::jint(body, "max_tokens", 8192) == 8192);
    CHECK(!q27::jbool(body, "stream", false));
    CHECK(q27::jbool(body, "stream", true));
    CHECK(q27::jnum(body, "temperature", 0.0) == 0.0);
    CHECK(q27::jnum(body, "top_p", 1.0) == 1.0);
    CHECK(q27::jstr(body, "prompt", "dflt") == "dflt");
}

static void test_jread_wrong_type_reads_as_absent() {
    json body = json::parse(R"({"max_tokens":"512","stream":"yes","temperature":{},"prompt":7})");
    CHECK(q27::jint(body, "max_tokens", 8192) == 8192);
    CHECK(!q27::jbool(body, "stream", false));
    CHECK(q27::jnum(body, "temperature", 0.0) == 0.0);
    CHECK(q27::jstr(body, "prompt", "dflt") == "dflt");
}

static void test_jread_present_values_win() {
    json body = json::parse(R"({"max_tokens":512,"stream":true,"temperature":0.7,
                                "top_p":0.95,"prompt":"hi"})");
    CHECK(q27::jint(body, "max_tokens", 8192) == 512);
    CHECK(q27::jbool(body, "stream", false));
    CHECK(q27::jnum(body, "temperature", 0.0) == 0.7);
    CHECK(q27::jnum(body, "top_p", 1.0) == 0.95);
    CHECK(q27::jstr(body, "prompt", "dflt") == "hi");
}

static void test_jint_float_and_absurd_magnitude() {
    // an integer field sent as a float still works; a nonsense magnitude
    // clamps instead of overflowing the int the callers assign into
    json body = json::parse(R"({"a":8192.0,"big":1e30,"neg":-1e30})");
    CHECK(q27::jint(body, "a", 0) == 8192);
    CHECK(q27::jint(body, "big", 0) == 2147483647L);
    CHECK(q27::jint(body, "neg", 0) == -2147483648L);
    CHECK(q27::jint(json::array(), "a", 5) == 5); // non-object body
}

static void test_parse_tool_call_requires_object_arguments() {
    for(const char* raw:{
            R"({"name":"first","arguments":null})",
            R"({"name":"first","arguments":[]})",
            R"({"name":"first","arguments":7})",
            R"({"name":"first","arguments":"null"})"})
        CHECK(!q27::parse_tool_call(raw).ok);
    auto missing=q27::parse_tool_call(R"({"name":"first"})");
    CHECK(missing.ok && missing.arguments.is_object());
    auto encoded=q27::parse_tool_call(
        R"({"name":"first","arguments":"{\"path\":\"safe\"}"})");
    CHECK(encoded.ok && encoded.arguments.is_object() &&
          encoded.arguments.value("path",std::string())=="safe");
}

// ---- malformed-shape robustness (must not throw -> no spurious 500) -------
static void test_anthropic_msgs_non_object_message_skipped() {
    json body = json::parse(R"({"messages":["hi",3,{"role":"user","content":"real"}]})");
    auto msgs = q27::anthropic_msgs(body);
    CHECK(msgs.size() == 1);
    if (msgs.size() == 1) CHECK(msgs[0].content == "real");
}

static void test_anthropic_msgs_bare_string_content_part_skipped() {
    json body = json::parse(
        R"({"messages":[{"role":"user","content":["bare",{"type":"text","text":"kept"}]}]})");
    auto msgs = q27::anthropic_msgs(body);
    CHECK(msgs.size() == 1);
    if (msgs.size() == 1) CHECK(msgs[0].content == "kept");
}

static void test_anthropic_msgs_system_array_of_strings() {
    json body = json::parse(R"({"system":["bare",{"type":"text","text":"sys"}],"messages":[]})");
    auto msgs = q27::anthropic_msgs(body);
    CHECK(msgs.size() == 1);
    if (msgs.size() == 1) {
        CHECK(msgs[0].role == "system");
        CHECK(msgs[0].content == "sys");
    }
}

static void test_anthropic_msgs_tool_result_bare_string_content() {
    json body = json::parse(R"({"messages":[{"role":"user","content":[
        {"type":"tool_result","content":["bare",{"type":"text","text":"out"}]}]}]})");
    auto msgs = q27::anthropic_msgs(body);
    CHECK(msgs.size() == 1);
    if (msgs.size() == 1) CHECK(msgs[0].content.find("out") != std::string::npos);
}

static void test_anthropic_tools_non_array_and_non_object_entries() {
    json body = json::parse(R"({"tools":["str",7,{"name":""},{"name":"Read","input_schema":{}}]})");
    json out = q27::anthropic_tools_json(body);
    CHECK(out.size() == 1);
    json body2 = json::parse(R"({"tools":{"not":"an array"}})");
    CHECK(q27::anthropic_tools_json(body2).empty());
}

// Claude Code's billing header carries a stamp that changes between
// conversations; if it is not pinned, the first ~15 tokens of every system
// prompt differ and NO prefix tier (P8 snapshot, P9 ring, P16 disk) can ever
// share state across sessions. CC 2.1.220 moved that stamp: the `cch=` field is
// gone and the volatile part rides a 4th component on cc_version. Captured live
// 2026-07-24 -- these two strings are verbatim from two real sessions.
static void test_billing_header_2_1_220_shape() {
    std::string a = "x-anthropic-billing-header: cc_version=2.1.220.473; cc_entrypoint=sdk-cli;You are";
    std::string b = "x-anthropic-billing-header: cc_version=2.1.220.c50; cc_entrypoint=sdk-cli;You are";
    q27::normalize_cc_billing_header(a);
    q27::normalize_cc_billing_header(b);
    CHECK(a == b);
    CHECK(a.find("cc_version=2.1.220.fff") != std::string::npos);
}
static void test_billing_header_legacy_cch_still_pinned() {
    std::string a = "x-anthropic-billing-header: cc_version=2.1.1; cc_entrypoint=cli; cch=a5145;X";
    std::string b = "x-anthropic-billing-header: cc_version=2.1.1; cc_entrypoint=cli; cch=b9e02;X";
    q27::normalize_cc_billing_header(a);
    q27::normalize_cc_billing_header(b);
    CHECK(a == b);
}
static void test_billing_header_leaves_other_prompts_alone() {
    std::string a = "You are a helpful assistant.", a0 = a;
    q27::normalize_cc_billing_header(a);
    CHECK(a == a0);
    std::string b = "x-anthropic-billing-header: cc_version=2.1.220; cc_entrypoint=sdk-cli;X", b0 = b;
    q27::normalize_cc_billing_header(b);
    CHECK(b == b0);  // no 4th component -> nothing to pin
}

static void test_tool_choice_absent_is_auto() {
    json body = json::object();
    auto tc = q27::parse_tool_choice(body);
    CHECK(tc.mode == q27::ToolChoice::AUTO);
}

static void test_tool_choice_none() {
    json body = {{"tool_choice", "none"}};
    auto tc = q27::parse_tool_choice(body);
    CHECK(tc.mode == q27::ToolChoice::NONE);
}

static void test_tool_choice_required() {
    json body = {{"tool_choice", "required"}};
    auto tc = q27::parse_tool_choice(body);
    CHECK(tc.mode == q27::ToolChoice::FORCED);
    CHECK(tc.forced_name.empty());
}

static void test_forced_tool_choice_truncation() {
    q27::ToolChoice forced;
    forced.mode=q27::ToolChoice::FORCED;
    CHECK(q27::forced_tool_choice_missing_is_error(forced,false,false));
    CHECK(!q27::forced_tool_choice_missing_is_error(forced,false,true));
    CHECK(!q27::forced_tool_choice_missing_is_error(forced,true,false));
    q27::ToolChoice automatic;
    CHECK(!q27::forced_tool_choice_missing_is_error(automatic,false,false));
}

static void test_anthropic_tool_stop_reason() {
    CHECK(std::string(q27::anthropic_tool_stop_reason(true,false,false))=="tool_use");
    CHECK(std::string(q27::anthropic_tool_stop_reason(true,true,true))=="max_tokens");
    CHECK(std::string(q27::anthropic_tool_stop_reason(false,false,true))=="max_tokens");
    CHECK(std::string(q27::anthropic_tool_stop_reason(false,false,false))=="end_turn");
    CHECK(std::string(q27::openai_tool_finish_reason(true,false,false))=="tool_calls");
    CHECK(std::string(q27::openai_tool_finish_reason(true,true,true))=="length");
    CHECK(std::string(q27::openai_tool_finish_reason(false,false,true))=="length");
    CHECK(std::string(q27::openai_tool_finish_reason(false,false,false))=="stop");
}

static void test_tool_choice_named_function() {
    json body = {{"tool_choice", {{"type","function"},{"function",{{"name","get_weather"}}}}}};
    auto tc = q27::parse_tool_choice(body);
    CHECK(tc.mode == q27::ToolChoice::FORCED);
    CHECK(tc.forced_name == "get_weather");
}

static void test_tool_choice_malformed_named_is_invalid() {
    for (const json& value : json::array({
             json{{"type","function"},{"function",json::object()}},
             json{{"type","function"},{"function",{{"name",""}}}},
             json{{"type",nullptr},{"function",{{"name","get_weather"}}}}
         })) {
        auto tc = q27::parse_tool_choice(json{{"tool_choice",value}});
        CHECK(tc.mode == q27::ToolChoice::AUTO);
        CHECK(tc.invalid);
    }
}

static void test_tool_choice_allowed_tools() {
    json tools=json::array({
        {{"type","function"},{"function",{{"name","get_weather"}}}},
        {{"type","function"},{"function",{{"name","get_time"}}}}
    });
    auto automatic=q27::parse_tool_choice({{"tool_choice",{{"type","allowed_tools"},
        {"allowed_tools",{{"mode","auto"},{"tools",tools}}}}}});
    CHECK(automatic.mode == q27::ToolChoice::AUTO);
    CHECK(!automatic.invalid);
    CHECK(automatic.allowed_names.size() == 2);
    auto required=q27::parse_tool_choice({{"tool_choice",{{"type","allowed_tools"},
        {"allowed_tools",{{"mode","required"},{"tools",tools}}}}}});
    CHECK(required.mode == q27::ToolChoice::FORCED);
    CHECK(!required.invalid);
    CHECK(required.allowed_names.size() == 2);
    json body={{"tools",json::array({
        {{"type","function"},{"function",{{"name","get_weather"}}}},
        {{"type","function"},{"function",{{"name","privileged"}}}},
        {{"type","function"},{"function",{{"name","get_time"}}}}
    })}};
    auto selected=q27::select_openai_tools(body,required);
    CHECK(selected.names.size() == 2);
    CHECK(selected.names[0] == "get_weather");
    CHECK(selected.names[1] == "get_time");
    CHECK(selected.tools.size() == 2);
    auto missing=q27::parse_tool_choice({{"tool_choice",{{"type","function"},
        {"function",{{"name","missing"}}}}}});
    bool threw=false;
    try { (void)q27::select_openai_tools(body,missing); }
    catch (const std::runtime_error&) { threw=true; }
    CHECK(threw);
}

static void test_openai_parallel_tool_calls() {
    const std::set<std::string> declared={"first","second"};
    q27::ToolChoice single=q27::parse_tool_choice(json::object());
    q27::apply_openai_parallel_tool_calls({{"parallel_tool_calls",false}},single);
    CHECK(single.disable_parallel_tool_use);
    CHECK(q27::tool_choice_allows_call(single,declared,"first",0));
    CHECK(!q27::tool_choice_allows_call(single,declared,"second",1));

    q27::ToolChoice parallel=q27::parse_responses_tool_choice(json::object());
    q27::apply_openai_parallel_tool_calls({{"parallel_tool_calls",true}},parallel);
    CHECK(!parallel.disable_parallel_tool_use);
    CHECK(q27::tool_choice_allows_call(parallel,declared,"second",1));

    q27::ToolChoice malformed=q27::parse_tool_choice(json::object());
    q27::apply_openai_parallel_tool_calls({{"parallel_tool_calls","no"}},malformed);
    CHECK(malformed.invalid);
}

static void test_recovered_call_batch_eligibility() {
    const std::set<std::string> declared={"first","second"};
    q27::ToolCall first; first.ok=true; first.name="first";
    q27::ToolCall second; second.ok=true; second.name="second";
    std::vector<q27::ToolCall> calls={first,second};

    q27::ToolChoice parallel=q27::parse_tool_choice(json::object());
    CHECK(q27::tool_choice_allows_all_calls(parallel,declared,calls));

    q27::ToolChoice single=q27::parse_tool_choice(json::object());
    q27::apply_openai_parallel_tool_calls({{"parallel_tool_calls",false}},single);
    CHECK(!q27::tool_choice_allows_all_calls(single,declared,calls));
    CHECK(q27::tool_choice_allows_all_calls(single,declared,{first}));
    CHECK(!q27::tool_choice_allows_all_calls(single,declared,{first},1));

    q27::ToolChoice named=q27::parse_tool_choice({{"tool_choice",{{"type","function"},
        {"function",{{"name","first"}}}}}});
    CHECK(!q27::tool_choice_allows_all_calls(named,declared,{second}));
}

static void test_anthropic_tool_choice_shapes() {
    auto absent=q27::parse_anthropic_tool_choice(json::object());
    CHECK(absent.mode == q27::ToolChoice::AUTO);
    CHECK(!absent.invalid);
    auto automatic=q27::parse_anthropic_tool_choice(
        {{"tool_choice",{{"type","auto"},{"disable_parallel_tool_use",true}}}});
    CHECK(automatic.mode == q27::ToolChoice::AUTO);
    CHECK(!automatic.invalid);
    CHECK(automatic.disable_parallel_tool_use);
    auto none=q27::parse_anthropic_tool_choice({{"tool_choice",{{"type","none"}}}});
    CHECK(none.mode == q27::ToolChoice::NONE);
    auto any=q27::parse_anthropic_tool_choice({{"tool_choice",{{"type","any"}}}});
    CHECK(any.mode == q27::ToolChoice::FORCED);
    CHECK(any.forced_name.empty());
    auto named=q27::parse_anthropic_tool_choice(
        {{"tool_choice",{{"type","tool"},{"name","get_weather"}}}});
    CHECK(named.mode == q27::ToolChoice::FORCED);
    CHECK(named.forced_name == "get_weather");
    CHECK(named.allowed_names.size() == 1);
    CHECK(q27::anthropic_tool_choice_instruction(automatic).empty());
    CHECK(q27::anthropic_tool_choice_instruction(none).find("disabled")!=std::string::npos);
    CHECK(q27::anthropic_tool_choice_instruction(any).find("at least one")!=std::string::npos);
    CHECK(q27::anthropic_tool_choice_instruction(named).find("get_weather")!=std::string::npos);
    const json thinking_body={
        {"thinking",{{"type","enabled"},{"budget_tokens",1024}}}};
    const q27::ThinkCfg requested_thinking=q27::resolve_think_cfg(
        thinking_body,false,true,-1);
    bool forced_thinking_threw=false;
    try {
        q27::validate_anthropic_tool_choice_thinking(named,requested_thinking);
    } catch(const std::invalid_argument&) { forced_thinking_threw=true; }
    CHECK(forced_thinking_threw);
    q27::validate_anthropic_tool_choice_thinking(
        named,q27::resolve_think_cfg(thinking_body,false,false,-1));
    q27::validate_anthropic_tool_choice_thinking(
        named,q27::resolve_think_cfg(json::object(),true,true,-1));
    q27::validate_anthropic_tool_choice_thinking(none,requested_thinking);
    json body={{"tools",json::array({
        {{"name","get_weather"},{"description","weather"},
         {"input_schema",{{"type","object"}}}},
        {{"name","get_time"},{"input_schema",{{"type","object"}}}}
    })}};
    json normalized={{"tools",q27::anthropic_tools_json(body)}};
    auto selected=q27::select_openai_tools(normalized,named);
    CHECK(selected.names.size() == 1);
    CHECK(selected.names[0] == "get_weather");
    const std::set<std::string> declared={"get_weather","get_time"};
    CHECK(q27::tool_choice_allows_call(named,declared,"get_weather",0));
    CHECK(!q27::tool_choice_allows_call(named,declared,"get_time",0));
    CHECK(!q27::tool_choice_allows_call(none,declared,"get_weather",0));
    CHECK(q27::tool_choice_allows_call(automatic,declared,"get_weather",0));
    CHECK(!q27::tool_choice_allows_call(automatic,declared,"get_time",1));
    CHECK(!q27::tool_choice_allows_call(automatic,declared,"undeclared",0));
    bool missing_threw=false;
    try {
        auto missing=q27::parse_anthropic_tool_choice(
            {{"tool_choice",{{"type","tool"},{"name","missing"}}}});
        (void)q27::select_openai_tools(normalized,missing);
    } catch(const std::runtime_error&) { missing_threw=true; }
    CHECK(missing_threw);
    for(const json& value:json::array({
            json("auto"), json::object(), json{{"type","tool"}},
            json{{"type","tool"},{"name",""}}, json{{"type","unknown"}}
        })) {
        auto malformed=q27::parse_anthropic_tool_choice({{"tool_choice",value}});
        CHECK(malformed.invalid);
    }
    auto bad_parallel=q27::parse_anthropic_tool_choice(
        {{"tool_choice",{{"type","auto"},{"disable_parallel_tool_use","yes"}}}});
    CHECK(bad_parallel.invalid);
}

static void test_responses_tool_choice_shapes() {
    auto named=q27::parse_responses_tool_choice({{"tool_choice",{{"type","function"},{"name","get_weather"}}}});
    CHECK(named.mode == q27::ToolChoice::FORCED);
    CHECK(named.forced_name == "get_weather");
    auto custom=q27::parse_responses_tool_choice({{"tool_choice",{{"type","custom"},{"name","shell"}}}});
    CHECK(custom.mode == q27::ToolChoice::FORCED);
    CHECK(custom.forced_name == "shell");
    auto allowed=q27::parse_responses_tool_choice({{"tool_choice",{{"type","allowed_tools"},
        {"mode","auto"},{"tools",json::array({
            {{"type","function"},{"name","get_weather"}},
            {{"type","custom"},{"name","shell"}}
        })}}}});
    CHECK(!allowed.invalid);
    CHECK(allowed.mode == q27::ToolChoice::AUTO);
    CHECK(allowed.allowed_names.size() == 2);
    CHECK(allowed.allowed_names[0] == "get_weather");
    CHECK(allowed.allowed_names[1] == "shell");
    auto hosted=q27::parse_responses_tool_choice({{"tool_choice",{{"type","shell"}}}});
    CHECK(!hosted.invalid);
    CHECK(hosted.mode == q27::ToolChoice::FORCED);
    CHECK(hosted.forced_name == "shell");
    const json shell_tools=q27::responses_shell_prompt_tools();
    CHECK(shell_tools.is_array() && shell_tools.size()==2);
    CHECK(shell_tools[0]["function"]["name"]=="exec_command");
    CHECK(shell_tools[0]["function"]["parameters"]["required"]==json::array({"cmd"}));
    CHECK(shell_tools[0]["function"]["parameters"]["properties"].contains("yield_time_ms"));
    CHECK(shell_tools[1]["function"]["name"]=="write_stdin");
    CHECK(shell_tools[1]["function"]["parameters"]["required"]==json::array({"session_id"}));
    auto hosted_allowed=q27::parse_responses_tool_choice({{"tool_choice",{{"type","allowed_tools"},
        {"mode","required"},{"tools",json::array({{{"type","shell"}}})}}}});
    CHECK(!hosted_allowed.invalid);
    CHECK(hosted_allowed.mode == q27::ToolChoice::FORCED);
    CHECK(hosted_allowed.allowed_names.size() == 1);
    CHECK(hosted_allowed.allowed_names[0] == "shell");
    const std::set<std::string> hosted_types={"shell"};
    auto registered_hosted=q27::responses_registered_tool_choice(hosted,hosted_types);
    CHECK(registered_hosted.mode == q27::ToolChoice::FORCED);
    CHECK(registered_hosted.forced_name.empty());
    CHECK(registered_hosted.allowed_names ==
          std::vector<std::string>({"exec_command","write_stdin"}));
    auto registered_allowed=q27::responses_registered_tool_choice(hosted_allowed,hosted_types);
    CHECK(registered_allowed.mode == q27::ToolChoice::FORCED);
    CHECK(registered_allowed.allowed_names ==
          std::vector<std::string>({"exec_command","write_stdin"}));
    auto registered_mixed=q27::responses_registered_tool_choice(allowed,hosted_types);
    CHECK(registered_mixed.mode == q27::ToolChoice::AUTO);
    CHECK(registered_mixed.allowed_names ==
          std::vector<std::string>({"exec_command","get_weather","write_stdin"}));
    CHECK(q27::responses_tool_names_ambiguous(
        {"exec_command"},{},hosted_types));
    CHECK(q27::responses_tool_names_ambiguous(
        {},{"write_stdin"},hosted_types));
    CHECK(q27::responses_tool_names_ambiguous(
        {"shell"},{},hosted_types));
    CHECK(q27::responses_tool_names_ambiguous(
        {"duplicate"},{"duplicate"},{}));
    CHECK(!q27::responses_tool_names_ambiguous(
        {"get_weather"},{"apply_patch"},hosted_types));
    q27::validate_responses_tool_choice_declarations(
        {{"tool_choice",{{"type","custom"},{"name","apply_patch"}}}},
        {"get_weather"},{"apply_patch"},hosted_types);
    bool mismatched_kind_threw=false;
    try {
        q27::validate_responses_tool_choice_declarations(
            {{"tool_choice",{{"type","function"},{"name","apply_patch"}}}},
            {"get_weather"},{"apply_patch"},hosted_types);
    } catch(const std::runtime_error&) { mismatched_kind_threw=true; }
    CHECK(mismatched_kind_threw);
    bool mismatched_allowed_kind_threw=false;
    try {
        q27::validate_responses_tool_choice_declarations(
            {{"tool_choice",{{"type","allowed_tools"},{"tools",json::array({
                {{"type","custom"},{"name","get_weather"}}
            })}}}},
            {"get_weather"},{"apply_patch"},hosted_types);
    } catch(const std::runtime_error&) { mismatched_allowed_kind_threw=true; }
    CHECK(mismatched_allowed_kind_threw);
    auto empty=q27::parse_responses_tool_choice({{"tool_choice",json::object()}});
    CHECK(empty.invalid);
    auto mcp=q27::parse_responses_tool_choice({{"tool_choice",{{"type","mcp"},
        {"server_label","filesystem"},{"name","read_file"}}}});
    CHECK(mcp.invalid);
    auto mcp_allowed=q27::parse_responses_tool_choice({{"tool_choice",{{"type","allowed_tools"},
        {"mode","auto"},{"tools",json::array({{{"type","mcp"},
            {"server_label","filesystem"}}})}}}});
    CHECK(!mcp_allowed.invalid && mcp_allowed.mode==q27::ToolChoice::NONE);
    auto web_search=q27::parse_responses_tool_choice(
        {{"tool_choice",{{"type","web_search_preview"}}}});
    CHECK(web_search.invalid);
    auto web_search_allowed=q27::parse_responses_tool_choice({{"tool_choice",{{"type","allowed_tools"},
        {"mode","auto"},{"tools",json::array({{{"type","web_search_preview"}}})}}}});
    CHECK(!web_search_allowed.invalid &&
          web_search_allowed.mode==q27::ToolChoice::NONE);
    auto mixed_hosted_allowed=q27::parse_responses_tool_choice({{"tool_choice",{{"type","allowed_tools"},
        {"mode","auto"},{"tools",json::array({
            {{"type","web_search_preview"}},
            {{"type","function"},{"name","get_weather"}}
        })}}}});
    CHECK(!mixed_hosted_allowed.invalid &&
          mixed_hosted_allowed.allowed_names==std::vector<std::string>({"get_weather"}));
    auto required_web_search=q27::parse_responses_tool_choice({{"tool_choice",{{"type","allowed_tools"},
        {"mode","required"},{"tools",json::array({{{"type","web_search_preview"}}})}}}});
    CHECK(required_web_search.invalid);
    auto unknown_string=q27::parse_responses_tool_choice({{"tool_choice","none "}});
    CHECK(unknown_string.invalid);
    auto malformed=q27::parse_responses_tool_choice({{"tool_choice",{{"type","function"},{"name",3}}}});
    CHECK(malformed.invalid);
    std::set<std::string> eligible={"get_weather"};
    q27::add_responses_hosted_call_names(eligible,"shell");
    auto none=q27::parse_responses_tool_choice({{"tool_choice","none"}});
    CHECK(!q27::tool_choice_allows_call(none,eligible,"get_weather",0));
    CHECK(!q27::tool_choice_allows_call(none,eligible,"exec_command",0));
    CHECK(q27::tool_choice_allows_call(named,eligible,"get_weather",0));
    CHECK(!q27::tool_choice_allows_call(named,eligible,"exec_command",0));
    auto hosted_output=hosted;
    hosted_output.forced_name.clear(); // hosted type maps to its concrete call names
    CHECK(q27::tool_choice_allows_call(hosted_output,eligible,"exec_command",0));
    q27::apply_openai_parallel_tool_calls({{"parallel_tool_calls",false}},hosted_output);
    CHECK(!q27::tool_choice_allows_call(hosted_output,eligible,"write_stdin",1));
    json response_tools=json::array({{{"type","function"},{"function",{
        {"name","get_weather"},{"parameters",json::object()}}}}});
    std::string bare_prefix,bare_remaining;
    auto bare_calls=q27::parse_bare_tool_calls(
        "answer {\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tokyo\"}}",
        &bare_prefix,&response_tools,true,true,&bare_remaining);
    CHECK(bare_calls.size()==1 && bare_calls[0].ok &&
          bare_calls[0].name=="get_weather" &&
          bare_calls[0].arguments["city"]=="Tokyo");
    CHECK(q27::responses_tool_tail_after_bare_calls(
        false,
        "answer {\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tokyo\"}}",
        bare_calls,named,eligible));
    std::string trailing_prefix,trailing_remaining;
    const std::string trailing_text =
        "{\"name\":\"get_weather\",\"arguments\":{}} ordinary text";
    auto trailing_calls=q27::parse_bare_tool_calls(
        trailing_text,&trailing_prefix,&response_tools,true,true,&trailing_remaining);
    CHECK(!q27::responses_tool_tail_after_bare_calls(
        false,trailing_text,trailing_calls,named,eligible));
    CHECK(!q27::responses_tool_tail_after_bare_calls(
        true,"ordinary text",{},named,eligible));
    CHECK(q27::responses_tool_tail_after_bare_calls(
        true," \n\t",{},named,eligible));
    auto required=q27::parse_responses_tool_choice({{"tool_choice","required"}});
    CHECK(q27::tool_choice_allows_call(required,eligible,bare_calls[0].name,0));
    CHECK(!q27::forced_tool_choice_missing_is_error(required,true,false));
}

static void test_stream_options_include_usage() {
    CHECK(q27::openai_stream_includes_usage({{"stream",true},
        {"stream_options",{{"include_usage",true}}}}));
    CHECK(!q27::openai_stream_includes_usage({{"stream",false},
        {"stream_options",{{"include_usage",true}}}}));
    CHECK(!q27::openai_stream_includes_usage({{"stream",true},
        {"stream_options",{{"include_usage","yes"}}}}));
    CHECK(!q27::openai_stream_includes_usage({{"stream",true},
        {"stream_options",json::array()}}));
}

static void test_tool_choice_unknown_string_is_auto() {
    json body = {{"tool_choice", "auto"}};
    auto tc = q27::parse_tool_choice(body);
    CHECK(tc.mode == q27::ToolChoice::AUTO);
}

// End-to-end sanity: openai_msgs + openai_tools_json feed correctly into the
// SAME chatml_prompt() the /v1/messages path already uses in production.
static void test_end_to_end_chatml_prompt() {
    json body = {
        {"tools", json::array({
            {{"type","function"},{"function",{{"name","get_weather"},{"description","w"},
                {"parameters", {{"type","object"},{"properties",{{"location",{{"type","string"}}}}}}}}}}
        })},
        {"messages", json::array({
            {{"role","system"},{"content","be terse"}},
            {{"role","user"},{"content","weather in tokyo?"}},
        })},
    };
    json tools = q27::openai_tools_json(body);
    auto msgs = q27::openai_msgs(body);
    size_t stable_off = 0;
    std::string rendered = q27::chatml_prompt(msgs, tools, /*think=*/false, &stable_off);
    CHECK(rendered.find("# Tools") != std::string::npos);
    CHECK(rendered.find("get_weather") != std::string::npos);
    CHECK(rendered.find("be terse") != std::string::npos);
    CHECK(rendered.find("weather in tokyo?") != std::string::npos);
    CHECK(rendered.rfind("<|im_start|>assistant\n<think>\n\n</think>\n\n") ==
          rendered.size() - std::string("<|im_start|>assistant\n<think>\n\n</think>\n\n").size());
    CHECK(stable_off > 0 && stable_off < rendered.size());
    // FORCED-mode prompt injection (server.cu appends this after chatml_prompt
    // returns): must land in the volatile tail, past stable_off.
    std::string forced = rendered + "<tool_call>\n";
    CHECK(forced.substr(0, stable_off) == rendered.substr(0, stable_off));
}

// P0 (2026-08-22): assistant reasoning must NOT be lost on the OpenAI path and
// must render as a jinja-style <think> block in chatml_prompt.
static void test_openai_reasoning_content_roundtrip() {
    json body = {
        {"messages", json::array({
            {{"role","user"},{"content","plan the build"}},
            {{"role","assistant"},{"content","Done."},{"reasoning_content","Let me check the steps."}},
            {{"role","user"},{"content","go"}},
        })}
    };
    auto msgs = q27::openai_msgs(body);
    CHECK(msgs.size() == 3);
    // assistant reasoning was captured, not dropped
    CHECK(msgs[1].role == "assistant");
    CHECK(msgs[1].content == "Done.");
    CHECK(msgs[1].reasoning == "Let me check the steps.");
    // no reasoning on the non-assistant turns
    CHECK(msgs[0].reasoning.empty());
    CHECK(msgs[2].reasoning.empty());
    // chatml_prompt renders it as a jinja-style <think> block
    std::string rendered = q27::chatml_prompt(msgs, {}, /*think=*/false);
    CHECK(rendered.find("<think>\nLet me check the steps.\n</think>\n\n") != std::string::npos);
    // order: think block comes before the assistant content
    CHECK(rendered.find("<think>\nLet me check the steps.\n</think>\n\nDone.") != std::string::npos);
    // absent reasoning_content -> no <think> block injected (regression guard)
    json body2 = {{"messages", json::array({
        {{"role","assistant"},{"content","hi"}},
    })}};
    auto msgs2 = q27::openai_msgs(body2);
    CHECK(msgs2[0].reasoning.empty());
    std::string r2 = q27::chatml_prompt(msgs2, {}, /*think=*/false);
    // the only <think> block must be the trailing EMPTY generation prompt
    // (no history reasoning block was injected)
    CHECK(r2.rfind("<think>\n\n</think>\n\n") ==
          r2.size() - std::string("<think>\n\n</think>\n\n").size());
}

// P1a: <|think_*|> toggles must be stripped and must drive the generation
// prompt + reasoning-instructions line (stateful, mirroring the template).
static void test_think_toggle_state() {
    unsetenv("Q27_REASONING_EFFORT");
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen38 27b Hf\"}");
    {   // <|think_off|> -> closed gen block even with think=true param
        json body = {{"messages", json::array({
            {{"role","user"},{"content","<|think_off|>Just answer."}},
        })}};
        auto msgs = q27::openai_msgs(body);
        std::string r = q27::chatml_prompt(msgs, {}, /*think=*/true);
        CHECK(r.find("<|think_off|>") == std::string::npos);          // stripped
        CHECK(r.find("Just answer.") != std::string::npos);
        CHECK(r.rfind("<think>\n\n</think>\n\n") ==                  // thinking off
              r.size() - std::string("<think>\n\n</think>\n\n").size());
    }
    {   // <|think_on|> with think=false -> OPEN gen block + effort line
        json body = {{"messages", json::array({
            {{"role","user"},{"content","<|think_on|>Think hard."}},
        })}};
        auto msgs = q27::openai_msgs(body);
        std::string r = q27::chatml_prompt(msgs, {}, /*think=*/false);
        CHECK(r.find("<|think_on|>") == std::string::npos);
        CHECK(r.rfind("<think>\n") == r.size() - std::string("<think>\n").size());
        // default effort stays xhigh (PR #37 review: froggeric "default medium"
        // dropped) -> the xhigh line is injected
        CHECK(r.find("Reasoning effort is set to xhigh.") != std::string::npos);
    }
    {   // <|think_xhigh|> arms xhigh effort on a no-think base
        json body = {{"messages", json::array({
            {{"role","user"},{"content","<|think_xhigh|>Be careful."}},
        })}};
        auto msgs = q27::openai_msgs(body);
        std::string r = q27::chatml_prompt(msgs, {}, /*think=*/false);
        CHECK(r.find("Reasoning effort is set to xhigh.") != std::string::npos);
    }
}

// P1b: consecutive tool-error responses accumulate system warnings.
static void test_tool_failure_warnings() {
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen38 27b Hf\"}");
    unsetenv("Q27_TOOL_ERROR_WARNINGS");
    auto tr = [&](const std::string& body) { return "<tool_response>\n" + body + "\n</tool_response>"; };
    json body = {{"messages", json::array({
        {{"role","user"},{"content","run the build"}},
        {{"role","user"},{"content",tr("error: command not found: make")}},
        {{"role","user"},{"content",tr("fatal: make: no targets")}},
    })}};
    // default is OFF per PR #37 review (measured 3/12 FP on benign outputs):
    // even two consecutive failures inject no warning when no opts are passed.
    auto msgs = q27::openai_msgs(body);
    std::string r = q27::chatml_prompt(msgs, {}, /*think=*/false);
    CHECK(r.find("SYSTEM WARNING") == std::string::npos);
    // env re-enables at boot (template parity is one flag away)
    setenv("Q27_TOOL_ERROR_WARNINGS", "1", 1);
    r = q27::chatml_prompt(msgs, {}, /*think=*/false);
    // 1st failure -> diagnose warning
    CHECK(r.find("SYSTEM WARNING: The previous tool call returned an error.") != std::string::npos);
    // 2nd consecutive failure -> change-approach warning with count 2
    CHECK(r.find("2 consecutive tool errors detected. Your previous approach is incorrect.") != std::string::npos);
    // a plain user turn after resets the counter (no "1 consecutive" survives)
    json body2 = {{"messages", json::array({
        {{"role","user"},{"content",tr("error: boom")}},
        {{"role","user"},{"content","the error is fixed now"}},
        {{"role","user"},{"content",tr("error: boom again")}},
    })}};
    auto msgs2 = q27::openai_msgs(body2);
    std::string r2 = q27::chatml_prompt(msgs2, {}, /*think=*/false);
    // only two separate single-failure warnings, no "2 consecutive"
    CHECK(r2.find("2 consecutive tool errors") == std::string::npos);
    CHECK(r2.find("The previous tool call returned an error.") != std::string::npos);
    unsetenv("Q27_TOOL_ERROR_WARNINGS");

    // per-request field (PR #37 review): explicit true wins over the OFF boot
    // default; explicit false wins over an ON env default.
    json body3 = {{"messages", body["messages"]}, {"tool_error_warnings", true}};
    auto opts = q27::template_opts_from_body(body3);
    CHECK(opts.tool_error_warnings);
    std::string r3 = q27::chatml_prompt(q27::openai_msgs(body3), {}, /*think=*/false, nullptr, nullptr, {}, {}, &opts);
    CHECK(r3.find("SYSTEM WARNING") != std::string::npos);   // request overrides boot default
    // chat_template_kwargs carries the field too
    json body4 = {{"messages", body["messages"]},
                  {"chat_template_kwargs",{{"tool_error_warnings",false}}}};
    CHECK(!q27::template_opts_from_body(body4).tool_error_warnings);
    // env ON + request false -> request wins
    setenv("Q27_TOOL_ERROR_WARNINGS", "1", 1);
    CHECK(q27::template_opts_from_body(body).tool_error_warnings);    // env on
    json body5 = {{"messages", body["messages"]}, {"tool_error_warnings", false}};
    CHECK(!q27::template_opts_from_body(body5).tool_error_warnings);  // request wins
    unsetenv("Q27_TOOL_ERROR_WARNINGS");
}

// P2: consecutive tool responses group under a single user turn.
static void test_tool_grouping() {
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen38 27b Hf\"}");
    auto tr = [&](const std::string& body) { return "<tool_response>\n" + body + "\n</tool_response>"; };
    json body = {{"messages", json::array({
        {{"role","user"},{"content","run both"}},
        {{"role","user"},{"content",tr("out A")}},
        {{"role","user"},{"content",tr("out B")}},
        {{"role","user"},{"content","all done"}},
    })}};
    auto msgs = q27::openai_msgs(body);
    std::string r = q27::chatml_prompt(msgs, {}, /*think=*/false);
    // exactly 3 user turns: "run both", the grouped pair, "all done"
    size_t n = 0;
    for (size_t p = 0; (p = r.find("<|im_start|>user\n", p)) != std::string::npos; ++n, ++p);
    CHECK(n == 3);
    // the pair shares one user turn: grouped marker present, no <|im_end|> between
    CHECK(r.find("</tool_response>\n<tool_response>") != std::string::npos);
    CHECK(r.find("out A\n</tool_response><|im_end|>") == std::string::npos);
}

// P3: Q27_MAX_TOOL_RESPONSE_CHARS / Q27_MAX_TOOL_ARG_CHARS truncate long output.
static void test_tool_truncation() {
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen38 27b Hf\"}");
    std::string big(100, 'x');
    {   // response truncation (role "tool" goes through tool_response_text)
        setenv("Q27_MAX_TOOL_RESPONSE_CHARS", "20", 1);
        json body = {{"messages", json::array({
            {{"role","user"},{"content","run it"}},
            {{"role","tool"},{"content",big}},
        })}};
        std::string r = q27::chatml_prompt(q27::openai_msgs(body), {}, false);
        unsetenv("Q27_MAX_TOOL_RESPONSE_CHARS");
        CHECK(r.find("TRUNCATED - original length") != std::string::npos);
        CHECK(r.find(big) == std::string::npos); // long tail gone
    }
    {   // arg truncation (assistant tool_calls go through tool_call_text)
        setenv("Q27_MAX_TOOL_ARG_CHARS", "30", 1);
        json body = {{"messages", json::array({
            {{"role","assistant"},{"content","calling"},{"tool_calls", json::array({
                {{"id","1"},{"type","function"},{"function",{{"name","ls"},
                    {"arguments", "{\"path\":\"/" + big + "\"}"}}}}
            })}},
        })}};
        std::string r = q27::chatml_prompt(q27::openai_msgs(body), {}, false);
        unsetenv("Q27_MAX_TOOL_ARG_CHARS");
        CHECK(r.find("TRUNCATED - original length") != std::string::npos);
    }
}

// P3 request-driven: limits come from chat_template_kwargs in the body, not env.
static void test_tool_truncation_request_body() {
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen38 27b Hf\"}");
    std::string big(100, 'x');
    json body = {{"messages", json::array({
        {{"role","user"},{"content","run it"}},
        {{"role","tool"},{"content",big}},
    })},
        {"chat_template_kwargs", {{"max_tool_response_chars", 20}}}};
    std::string r = q27::chatml_prompt(q27::openai_msgs(body), {}, false);
    CHECK(r.find("TRUNCATED - original length") != std::string::npos);
    // explicit 0 in the body disables truncation even with a huge output
    json body0 = {{"messages", json::array({
        {{"role","user"},{"content","run it"}},
        {{"role","tool"},{"content",big}},
    })},
        {"chat_template_kwargs", {{"max_tool_response_chars", 0}}}};
    std::string r0 = q27::chatml_prompt(q27::openai_msgs(body0), {}, false);
    CHECK(r0.find("TRUNCATED") == std::string::npos);
}

// v22.3: multiple leading system/developer messages merge into ONE system block.
static void test_v223_multi_system_merge() {
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen38 27b Hf\"}");
    json body = {{"messages", json::array({
        {{"role","system"},{"content","sys one"}},
        {{"role","system"},{"content","sys two"}},
        {{"role","user"},{"content","hi"}},
    })}};
    auto msgs = q27::openai_msgs(body);
    std::string r = q27::chatml_prompt(msgs, {}, false);
    size_t n = 0;
    for (size_t p = 0; (p = r.find("<|im_start|>system\n", p)) != std::string::npos; ++n, ++p);
    CHECK(n == 1);                       // both merged into one block
    CHECK(r.find("sys one\n\nsys two") != std::string::npos); // joined with blank line
}

// v22.3: new <|think_ultracode|>/extreme/max markers map to xhigh effort.
static void test_v223_new_think_markers() {
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen38 27b Hf\"}");
    json body = {{"messages", json::array({
        {{"role","user"},{"content","<|think_ultracode|>deep reasoning"}},
    })}};
    auto msgs = q27::openai_msgs(body);
    std::string r = q27::chatml_prompt(msgs, {}, false);
    CHECK(r.find("<|think_ultracode|>") == std::string::npos);  // stripped
    CHECK(r.find("Reasoning effort is set to xhigh.") != std::string::npos);
}

// v22.3: explicit reasoning + a think block already in content -> no double render.
static void test_v223_reasoning_dedup() {
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen38 27b Hf\"}");
    json body = {{"messages", json::array({
        {{"role","user"},{"content","q"}},
        {{"role","assistant"},{"content","<think>\ninline thinking\n</think>\nFinal answer."},
            {"reasoning_content","explicit reasoning"}},
        {{"role","user"},{"content","what did you reason"}},
    })}};
    auto msgs = q27::openai_msgs(body);
    std::string r = q27::chatml_prompt(msgs, {}, false);
    CHECK(r.find("<think>\nexplicit reasoning\n</think>\n\nFinal answer.") != std::string::npos);
    CHECK(r.find("inline thinking") == std::string::npos); // inline dup removed
}

// v22.3: failure detection suppresses code output and exit-code-0.
static void test_v223_failure_detection() {
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen38 27b Hf\"}");
    // detector-under-renderer checks need the escalation ON (default is OFF
    // per PR #37 review); the detector itself is covered directly by the
    // positive cases below firing at all.
    setenv("Q27_TOOL_ERROR_WARNINGS", "1", 1);
    {   // code output with console.error is suppressed (not a failure)
        json b = {{"messages", json::array({
            {{"role","user"},{"content","run"}},
            {{"role","tool"},{"content","import os\nconsole.error('boom')\nfunction x() {}"}},
        })}};
        std::string r = q27::chatml_prompt(q27::openai_msgs(b), {}, false);
        CHECK(r.find("SYSTEM WARNING") == std::string::npos);
    }
    {   // exit code 0 is not a failure
        json b = {{"messages", json::array({
            {{"role","tool"},{"content","exit code: 0\nall good"}},
        })}};
        std::string r = q27::chatml_prompt(q27::openai_msgs(b), {}, false);
        CHECK(r.find("SYSTEM WARNING") == std::string::npos);
    }
    {   // nonzero exit code IS a failure
        json b = {{"messages", json::array({
            {{"role","tool"},{"content","exit code: 2\nboom"}},
        })}};
        std::string r = q27::chatml_prompt(q27::openai_msgs(b), {}, false);
        CHECK(r.find("SYSTEM WARNING") != std::string::npos);
    }
    unsetenv("Q27_TOOL_ERROR_WARNINGS");
}

// v22.3 E: every assistant history turn is wrapped in <think>, even empty.
static void test_unconditional_think_wrap() {
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen38 27b Hf\"}");
    {   // assistant with no reasoning -> empty think block
        json b = {{"messages", json::array({
            {{"role","user"},{"content","q"}},
            {{"role","assistant"},{"content","just an answer"}},
        })}};
        std::string r = q27::chatml_prompt(q27::openai_msgs(b), {}, false);
        CHECK(r.find("<|im_start|>assistant\n<think>\n\n</think>\n\njust an answer<|im_end|>") != std::string::npos);
    }
    {   // assistant with reasoning -> its think block
        json b = {{"messages", json::array({
            {{"role","assistant"},{"content","answer"},{"reasoning_content","thought"}},
        })}};
        std::string r = q27::chatml_prompt(q27::openai_msgs(b), {}, false);
        CHECK(r.find("<|im_start|>assistant\n<think>\nthought\n</think>\n\nanswer<|im_end|>") != std::string::npos);
    }
}

// Items 3/4: request-driven reasoning_effort + auto_disable_thinking_with_tools.
static void test_request_effort_and_auto_disable() {
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen38 27b Hf\"}");
    json tools = json::parse(R"([{"type":"function","function":{"name":"R","parameters":{"type":"object"}}}])");
    json b = {{"messages", json::array({{{"role","user"},{"content","hi"}}})},
              {"reasoning_effort","low"},
              {"chat_template_kwargs",{{"auto_disable_thinking_with_tools",true}}}};
    auto opts = q27::template_opts_from_body(b);
    CHECK(opts.effort == 1);               // low -> level 1
    CHECK(opts.auto_disable_thinking_with_tools);
    auto msgs = q27::openai_msgs(b);
    // tools present + auto_disable -> thinking off -> no effort line
    std::string r = q27::chatml_prompt(msgs, tools, /*think=*/true, nullptr, nullptr, {}, {}, &opts);
    CHECK(r.find("Reasoning effort is set to low.") == std::string::npos);
    // no tools -> thinking on -> low line
    std::string r2 = q27::chatml_prompt(msgs, json::array(), /*think=*/true, nullptr, nullptr, {}, {}, &opts);
    CHECK(r2.find("Reasoning effort is set to low.") != std::string::npos);
}

// output_config / reasoning (OpenAI Responses) shapes for effort + budget + disabled.
static void test_output_config_fields() {
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen38 27b Hf\"}");
    json b = {{"messages", json::array({{{"role","user"},{"content","hi"}}})},
              {"reasoning", {{{"effort","low"}}}},
              {"output_config", {{"effort","high"},{"budget_tokens",1024},{"disabled",true}}}};
    auto opts = q27::template_opts_from_body(b);
    // output_config read after reasoning -> output_config.effort=high wins
    CHECK(opts.effort == 2);
    // resolve_think_cfg: disabled -> thinking off; budget 1024 from output_config
    auto tcfg = q27::resolve_think_cfg(b, /*server_default=*/true, /*allow_request=*/true, -1);
    CHECK(!tcfg.enabled);              // output_config.disabled
    CHECK(tcfg.enabled_set);
    CHECK(tcfg.budget_set);
    CHECK(tcfg.budget == 1024);        // output_config.budget_tokens
}

// item 2: reasoning_effort none/off disables thinking (prompt + engine).
// NOTE: the companion "default effort = medium" change was dropped per PR #37
// review -- the effective default stays xhigh/off; medium is reachable
// explicitly. Only the none/off disable behavior is covered here.
static void test_none_off_disables_thinking() {
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen38 27b Hf\"}");
    unsetenv("Q27_REASONING_EFFORT");
    {   // reasoning_effort none -> thinking OFF (closed gen prompt, no effort line)
        json b = {{"messages", json::array({{{"role","user"},{"content","hi"}}})},
                  {"reasoning_effort","none"}};
        auto opts = q27::template_opts_from_body(b);
        CHECK(opts.force_disable_thinking);
        // engine think mode must also be off (end-to-end)
        auto tcfg = q27::resolve_think_cfg(b, /*server_default=*/true, /*allow_request=*/true, -1);
        CHECK(!tcfg.enabled);
        std::string r = q27::chatml_prompt(q27::openai_msgs(b), json::array(), /*think=*/true, nullptr, nullptr, {}, {}, &opts);
        // closed generation block (thinking off) despite think=true
        CHECK(r.rfind("<think>\n\n</think>\n\n") == r.size() - std::string("<think>\n\n</think>\n\n").size());
    }
    {   // reasoning_effort off disables thinking too (via chat_template_kwargs)
        json b = {{"messages", json::array({{{"role","user"},{"content","hi"}}})},
                  {"chat_template_kwargs",{{"reasoning_effort","off"}}}};
        auto opts = q27::template_opts_from_body(b);
        CHECK(opts.force_disable_thinking);
    }
}

// item 6: XML-dialect assistant tool_calls render as <function=...><parameter=...>
static void test_xml_tool_call_rendering() {
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen38 27b Hf\"}");
    json body = {{"messages", json::array({
        {{"role","assistant"},{"content","calling"},{"tool_calls", json::array({
            {{"id","1"},{"type","function"},{"function",{{"name","edit"},
                {"arguments","{\"path\":\"/x\",\"newText\":\"hi\"}"}}}}
        })}},
    })}};
    auto msgs = q27::openai_msgs(body);
    std::string r = q27::chatml_prompt(msgs, {}, false);
    CHECK(r.find("<function=edit>") != std::string::npos);
    CHECK(r.find("<parameter=path>\n/x\n</parameter>") != std::string::npos);
    CHECK(r.find("<parameter=newText>\nhi\n</parameter>") != std::string::npos);
    // JSON dialect keeps the JSON tool_call_text
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen3.6-27B\"}");
    std::string r2 = q27::chatml_prompt(q27::openai_msgs(body), {}, false);
    CHECK(r2.find("<function=edit>") == std::string::npos);
    CHECK(r2.find("<tool_call>\n{\"name\": \"edit\"") != std::string::npos);
}

// item 8: tool-response truncation applies only in the XML (non-JSON) dialect.
static void test_truncation_json_dialect_skipped() {
    q27::set_tool_dialect_for_model("{\"general.name\": \"Qwen3.6-27B\"}");
    std::string big(200, 'x');
    json body = {{"messages", json::array({
        {{"role","tool"},{"content",big}},
    })},
        {"chat_template_kwargs", {{"max_tool_response_chars", 20}}}};
    std::string r = q27::chatml_prompt(q27::openai_msgs(body), {}, false);
    CHECK(r.find("TRUNCATED") == std::string::npos); // json dialect: no truncation
}

static void test_chat_message_plain_text_no_calls() {
    json msg = q27::openai_chat_message_json("hello there", {}, 42);
    CHECK(msg["role"] == "assistant");
    CHECK(msg["content"] == "hello there");
    CHECK(!msg.contains("tool_calls"));
}

static void test_chat_message_empty_text_no_calls_is_empty_string_not_null() {
    json msg = q27::openai_chat_message_json("", {}, 42);
    CHECK(msg["content"].is_string());
    CHECK(msg["content"] == "");
    CHECK(!msg.contains("tool_calls"));
}

static void test_chat_message_call_no_leftover_text_content_null() {
    q27::ToolCall c; c.ok = true; c.name = "get_weather"; c.arguments = {{"location","Tokyo"}};
    json msg = q27::openai_chat_message_json("", {c}, 7);
    CHECK(msg["content"].is_null());
    CHECK(msg["tool_calls"].size() == 1);
    CHECK(msg["tool_calls"][0]["id"] == "call_q27_7_0");
    CHECK(msg["tool_calls"][0]["type"] == "function");
    CHECK(msg["tool_calls"][0]["function"]["name"] == "get_weather");
    json args = json::parse(msg["tool_calls"][0]["function"]["arguments"].get<std::string>());
    CHECK(args["location"] == "Tokyo");
}

static void test_chat_message_call_plus_leftover_text() {
    q27::ToolCall c; c.ok = true; c.name = "search"; c.arguments = json::object();
    json msg = q27::openai_chat_message_json("Let me check that.", {c}, 1);
    CHECK(msg["content"] == "Let me check that.");
    CHECK(msg["tool_calls"].size() == 1);
}

static void test_chat_message_parallel_calls_indexed_and_ordered() {
    q27::ToolCall a; a.ok = true; a.name = "one"; a.arguments = json::object();
    q27::ToolCall bad; bad.ok = false; bad.raw = "garbage"; // must be skipped, not crash
    q27::ToolCall b; b.ok = true; b.name = "two"; b.arguments = json::object();
    json msg = q27::openai_chat_message_json("", {a, bad, b}, 3);
    CHECK(msg["tool_calls"].size() == 2);
    CHECK(msg["tool_calls"][0]["id"] == "call_q27_3_0");
    CHECK(msg["tool_calls"][0]["function"]["name"] == "one");
    CHECK(msg["tool_calls"][1]["id"] == "call_q27_3_1");
    CHECK(msg["tool_calls"][1]["function"]["name"] == "two");
}

static void test_chat_message_reasoning_content_included_when_present() {
    json msg = q27::openai_chat_message_json("the answer", {}, 1, "thinking it through");
    CHECK(msg["content"] == "the answer");
    CHECK(msg["reasoning_content"] == "thinking it through");
}

static void test_chat_message_reasoning_content_absent_when_empty() {
    json msg = q27::openai_chat_message_json("the answer", {}, 1, "");
    CHECK(!msg.contains("reasoning_content"));
}

static void test_chat_message_reasoning_content_with_tool_call() {
    q27::ToolCall c; c.ok = true; c.name = "get_weather"; c.arguments = json::object();
    json msg = q27::openai_chat_message_json("", {c}, 5, "deciding to check weather");
    CHECK(msg["content"].is_null());
    CHECK(msg["reasoning_content"] == "deciding to check weather");
    CHECK(msg["tool_calls"].size() == 1);
}

static void test_reasoning_delta_shape() {
    json d = q27::openai_reasoning_delta("partial thought");
    CHECK(d["reasoning_content"] == "partial thought");
}

static void test_stream_chunk_shape() {
    json j = q27::openai_stream_chunk("chatcmpl-1", "chat.completion.chunk", 123, "q27model",
                                      json{{"content", "hi"}});
    CHECK(j["id"] == "chatcmpl-1");
    CHECK(j["choices"][0]["delta"]["content"] == "hi");
    CHECK(j["choices"][0]["finish_reason"].is_null());
}

static void test_stream_chunk_finish_reason() {
    json j = q27::openai_stream_chunk("id", "obj", 0, "m", json::object(), "tool_calls");
    CHECK(j["choices"][0]["finish_reason"] == "tool_calls");
    CHECK(j["choices"][0]["delta"].empty());
}

static void test_tool_call_delta_shape() {
    q27::ToolCall c; c.ok = true; c.name = "get_weather"; c.arguments = {{"location","Paris"}};
    json d = q27::openai_tool_call_delta(0, "call_abc", c);
    CHECK(d["tool_calls"].size() == 1);
    CHECK(d["tool_calls"][0]["index"] == 0);
    CHECK(d["tool_calls"][0]["id"] == "call_abc");
    CHECK(d["tool_calls"][0]["function"]["name"] == "get_weather");
    json args = json::parse(d["tool_calls"][0]["function"]["arguments"].get<std::string>());
    CHECK(args["location"] == "Paris");
}

static void test_tool_call_invalid_utf8_replaced() {
    q27::ToolCall c;
    c.ok=true;
    c.name="get_weather";
    c.arguments={{"invalid",std::string(1,char(0xff))}};
    const std::string replacement="\xef\xbf\xbd";
    json invalid_delta=q27::openai_tool_call_delta(0,"call_invalid",c);
    json delta_args=json::parse(
        invalid_delta["tool_calls"][0]["function"]["arguments"].get<std::string>());
    CHECK(delta_args["invalid"]==replacement);
    json invalid_message=q27::openai_tool_call_json("call_invalid",c);
    json message_args=json::parse(
        invalid_message["function"]["arguments"].get<std::string>());
    CHECK(message_args["invalid"]==replacement);
}

static std::string feed_tool_bytewise(q27::ToolCallStreamer& streamer,
                                      const std::string& body,
                                      int& opened_count) {
    std::string out;
    for (char c : body) {
        bool opened = false;
        out += streamer.feed(std::string(1, c), &opened);
        if (opened) opened_count++;
    }
    return out;
}

static void test_tool_streamer_bytewise_production_shape() {
    q27::ToolCallStreamer streamer;
    int opened_count = 0;
    const std::string out = feed_tool_bytewise(streamer,
        "{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}",
        opened_count);
    std::string tail;
    CHECK(opened_count == 1);
    CHECK(streamer.opened);
    CHECK(streamer.name == "weather");
    CHECK(streamer.finalize(&tail));
    CHECK(tail.empty());
    CHECK(streamer.trail().empty());
    CHECK(json::parse(out)["city"] == "Paris");
}

static void test_tool_streamer_inline_repairs() {
    q27::ToolCallStreamer streamer;
    bool opened = false;
    const std::string raw_body = std::string(R"({"a":"b"} C:\tmp)") +
        "\nliteral </content>} {\"note\":\"function\"} junk before the real close";
    const std::string body =
        "{\"name\":\"write\",\"arguments\":{\"text\":\"a\"b\nc\",\"body\":<content>" +
        raw_body + "</content>}}";
    std::string out = streamer.feed(body, &opened);
    std::string tail;
    CHECK(opened);
    CHECK(!out.empty()); // raw bytes before the ambiguous close still stream
    CHECK(streamer.finalize(&tail));
    out += tail;
    CHECK(!streamer.invalid());
    const json args = json::parse(out);
    CHECK(args["text"] == "a\"b\nc");
    CHECK(args["body"] == raw_body);
}

static void test_tool_streamer_quoted_content_modes() {
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        std::string out = feed_tool_bytewise(streamer,
            "{\"name\":\"write\",\"arguments\":{\"content\":\"line\\nC:\\\\tmp\",\"path\":\"x\"}}",
            opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        CHECK(json::parse(out)["content"] == "line\nC:\\tmp");
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        const std::string raw = std::string(R"({"a":"b"} C:\tmp)") +
            "\nconst xs = [\"a\", \"b\"];\nline </content> literal";
        const std::string body =
            "{\"name\":\"write\",\"arguments\":{\"content\":\"" + raw +
            "</content>,\"path\":\"x\"}}";
        std::string out = feed_tool_bytewise(streamer, body, opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        CHECK(!streamer.invalid());
        const json args = json::parse(out);
        CHECK(args["content"] == raw);
        CHECK(args["path"] == "x");
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        const std::string raw = "say \"hi\"\nnext";
        const std::string body =
            "{\"name\":\"write\",\"arguments\":{\"content\":\"" + raw + "\"}}";
        std::string out = feed_tool_bytewise(streamer, body, opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        CHECK(!streamer.invalid());
        CHECK(json::parse(out)["content"] == raw);
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        std::string out = feed_tool_bytewise(streamer,
            "{\"name\":\"write\",\"arguments\":{\"content\":\"line\\n</content>\",\"path\":\"x\"}}",
            opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        const json args = json::parse(out);
        CHECK(args["content"] == "line\n</content>");
        CHECK(args["path"] == "x");
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        std::string out = feed_tool_bytewise(streamer,
            "{\"name\":\"write\",\"arguments\":{\"content\":\"a\",\"nested\":{\"content\":\"b\"}}}",
            opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        const json args = json::parse(out);
        CHECK(args["content"] == "a");
        CHECK(args["nested"]["content"] == "b");
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        const std::string raw_text = "a\nb";
        const std::string body =
            "{\"name\":\"write\",\"arguments\":{\"content\":\"keep </content>\",\"text\":\"" +
            raw_text + "\"}}";
        std::string out = feed_tool_bytewise(streamer, body, opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        const json args = json::parse(out);
        CHECK(args["content"] == "keep </content>");
        CHECK(args["text"] == raw_text);
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        const std::string raw_content = "say \"hi\"\n";
        const std::string body =
            "{\"name\":\"write\",\"arguments\":{\"content\":\"" + raw_content +
            "\",\"body\":<content>x</content>}}";
        std::string out = feed_tool_bytewise(streamer, body, opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        const json args = json::parse(out);
        CHECK(args["content"] == raw_content);
        CHECK(args["body"] == "x");
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        const std::string body =
            "{\"name\":\"write\",\"arguments\":{\"content\":\"raw</content>,"
            "\"path\":\"literal </content>\"}}";
        std::string out = feed_tool_bytewise(streamer, body, opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        const json args = json::parse(out);
        CHECK(args["content"] == "raw");
        CHECK(args["path"] == "literal </content>");
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        const std::string raw = "say \"hi\"</content>";
        const std::string body =
            "{\"name\":\"write\",\"arguments\":{\"content\":\"" + raw +
            "\",\"path\":\"x\"}}";
        std::string out = feed_tool_bytewise(streamer, body, opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        const json args = json::parse(out);
        CHECK(args["content"] == raw);
        CHECK(args["path"] == "x");
    }
    {
        q27::ToolCallStreamer streamer;
        std::string body = "{\"name\":\"write\",\"arguments\":{\"content\":\"a\nb\"";
        for (int i = 0; i < 512; i++) body += ",\"content\":\"v\"";
        body += "}}";
        bool opened = false;
        std::string out = streamer.feed(body, &opened);
        std::string tail;
        CHECK(opened);
        CHECK(streamer.finalize(&tail));
        out += tail;
        CHECK(json::parse(out)["content"] == "v");
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        std::string out = feed_tool_bytewise(streamer,
            "{\"name\":\"write\",\"arguments\":{\"body\":\"x</content>,y\",\"path\":\"p\"}}",
            opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        const json args = json::parse(out);
        CHECK(args["body"] == "x</content>,y");
        CHECK(args["path"] == "p");
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        std::string out = feed_tool_bytewise(streamer,
            "{\"name\":\"write\",\"arguments\":{\"old\":<content>a</content>,"
            "\"new\":<content>b</content>}}",
            opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        const json args = json::parse(out);
        CHECK(args["old"] == "a");
        CHECK(args["new"] == "b");
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        std::string out = feed_tool_bytewise(streamer,
            "{\"name\":\"write\",\"arguments\":{\"content\":\"ok\","
            "\"items\":[{\"content\":\"raw</content>}]}}",
            opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        const json args = json::parse(out);
        CHECK(args["content"] == "ok");
        CHECK(args["items"][0]["content"] == "raw");
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        std::string out = feed_tool_bytewise(streamer,
            "{\"name\":\"write\",\"arguments\":{\"old\":<content>a</content>,"
            "\"new\":<content>b</content>} literal</content>}}",
            opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        const json args = json::parse(out);
        CHECK(args["old"] == "a");
        CHECK(args["new"] == "b</content>} literal");
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        std::string out = feed_tool_bytewise(streamer,
            "{\"name\":\"write\",\"arguments\":{\"cont\\u0065nt\":\"raw</content>,\"path\":\"x\"}}",
            opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        const json args = json::parse(out);
        CHECK(args["content"] == "raw");
        CHECK(args["path"] == "x");
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        std::string out = feed_tool_bytewise(streamer,
            "{\"name\":\"w\",\"arguments\":{\"nested\":{\"body\":<content>x</content>}",
            opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(!streamer.finalize(&tail));
        out += tail;
        CHECK(out == "{\"nested\":{\"body\":\"x\"}");
    }
    {
        q27::ToolCallStreamer streamer;
        int opened_count = 0;
        const std::string second =
            "{\"name\":\"b\",\"arguments\":{\"new\":<content>y</content>}}";
        std::string out = feed_tool_bytewise(streamer,
            "{\"name\":\"a\",\"arguments\":{\"old\":<content>x</content>}}" + second,
            opened_count);
        std::string tail;
        CHECK(opened_count == 1);
        CHECK(streamer.finalize(&tail));
        out += tail;
        CHECK(json::parse(out)["old"] == "x");
        CHECK(streamer.trail() == second);
    }
}

static void test_tool_streamer_fallback_is_byte_exact() {
    q27::ToolCallStreamer streamer;
    const std::string body = "{ \"arguments\":{}, \"name\":\"wrong-order\" }";
    bool opened = false;
    CHECK(streamer.feed(body, &opened).empty());
    CHECK(!opened);
    CHECK(streamer.state == q27::ToolCallStreamer::FALLBACK);
    CHECK(streamer.raw == body);
}

static void test_tool_streamer_preserves_packed_call_trail() {
    q27::ToolCallStreamer streamer;
    bool opened = false;
    const std::string second = "{\"name\":\"b\",\"arguments\":{\"y\":2}}";
    std::string out = streamer.feed(
        "{\"name\":\"a\",\"arguments\":{\"content\":\"x\",\"x\":1}}" + second,
        &opened);
    std::string tail;
    CHECK(opened);
    CHECK(streamer.finalize(&tail));
    out += tail;
    CHECK(json::parse(out)["x"] == 1);
    CHECK(json::parse(out)["content"] == "x");
    CHECK(streamer.trail() == second);
}

static void test_tool_streamer_escapes_all_control_bytes() {
    q27::ToolCallStreamer streamer;
    std::string body="{\"name\":\"control\",\"arguments\":{\"input\":\"";
    std::string expected;
    for(unsigned char c:{(unsigned char)'\b',(unsigned char)'\f',(unsigned char)0,
                         (unsigned char)1,(unsigned char)'\n',(unsigned char)'\t'}) {
        body.push_back((char)c);
        expected.push_back((char)c);
    }
    body+="\"}}";
    bool opened=false;
    std::string out=streamer.feed(body,&opened);
    std::string tail;
    CHECK(opened);
    CHECK(streamer.finalize(&tail));
    out+=tail;
    CHECK(json::parse(out)["input"].get<std::string>()==expected);
    CHECK(q27::escape_json_interior(expected).find("\\u0000")!=std::string::npos);
}

static void test_tool_streamer_can_refuse_tail_repair() {
    q27::ToolCallStreamer streamer;
    bool opened=false;
    (void)streamer.feed("{\"name\":\"write\",\"arguments\":{\"path\":\"danger",&opened);
    std::string tail;
    CHECK(opened);
    CHECK(!streamer.finalize(&tail,false));
    CHECK(tail.empty());
    CHECK(streamer.state==q27::ToolCallStreamer::ARGS);
}

static void test_bare_tool_parser_can_refuse_eof_repair() {
    const std::string call="{\"name\":\"exec\",\"arguments\":{\"cmd\":\"danger";
    const std::string raw="before "+call;
    std::string prefix,remaining;
    CHECK(q27::parse_bare_tool_calls(raw,&prefix,nullptr,false,false).empty());
    const auto repaired=q27::parse_bare_tool_calls(raw,&prefix,nullptr,false,true,&remaining);
    CHECK(repaired.size()==1);
    CHECK(repaired[0].arguments["cmd"]=="danger");
    CHECK(repaired[0].source_begin==7 && repaired[0].source_end==raw.size());
    CHECK(repaired[0].raw==call);
    CHECK(remaining=="before ");
    const json tools=json::parse(R"([
      {"type":"function","function":{"name":"exec","parameters":{
        "type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}
    ])");
    const std::string nested="{\"name\":{\"cmd\":\"danger\"}";
    CHECK(q27::parse_bare_tool_calls(
        nested,&prefix,&tools,false,false,&remaining).empty());
    CHECK(remaining==nested);
    const std::string executable_call =
        R"({"name":"exec","arguments":{"cmd":"danger"}})";
    const std::string quoted_call = "He wrote \"example: " + executable_call + "\"";
    CHECK(q27::parse_bare_tool_calls(
        quoted_call,&prefix,&tools,false,true,&remaining).empty());
    CHECK(remaining==quoted_call);

    const std::string quoted_then_executable = quoted_call + "\n" + executable_call;
    const auto executable_calls = q27::parse_bare_tool_calls(
        quoted_then_executable,&prefix,&tools,false,true,&remaining);
    CHECK(executable_calls.size()==1);
    CHECK(executable_calls[0].source_begin==quoted_call.size()+1);
    CHECK(remaining==quoted_call+"\n");
    const std::string nested_json =
        R"({"payload":{"name":"exec","arguments":{"cmd":"danger"}}})";
    CHECK(q27::parse_bare_tool_calls(
        nested_json,&prefix,&tools,false,true,&remaining).empty());
    CHECK(remaining==nested_json);
    const std::string unterminated_nested_json =
        R"({"payload":{"name":"exec","arguments":{"cmd":"danger"}})";
    CHECK(q27::parse_bare_tool_calls(
        unterminated_nested_json,&prefix,&tools,false,true,&remaining).empty());
    CHECK(remaining==unterminated_nested_json);
    const std::string array_json =
        R"([{"name":"exec","arguments":{"cmd":"danger"}}])";
    CHECK(q27::parse_bare_tool_calls(
        array_json,&prefix,&tools,false,true,&remaining).empty());
    CHECK(remaining==array_json);

    const std::string quoted_namedrop =
        "He wrote \"example: {\"name\":{\"cmd\":\"danger\"}\"";
    std::vector<q27::ToolCall> namedropped_calls;
    size_t namedropped_first=std::string::npos;
    q27::scan_namedropped(
        quoted_namedrop,&tools,namedropped_calls,&namedropped_first);
    CHECK(namedropped_calls.empty());
    CHECK(namedropped_first==std::string::npos);

    const std::string quoted_raw_value =
        "He wrote \"example: {\"name\":\"exec\",\"arguments\":{\"cmd\":\"say \"hi\"\nnext\"}}\"";
    std::vector<q27::ToolCall> raw_value_calls;
    CHECK(!q27::recover_raw_value_call(quoted_raw_value,tools,raw_value_calls));
    CHECK(raw_value_calls.empty());
}

static void test_tool_streamer_marks_invalid_completed_args() {
    q27::ToolCallStreamer streamer;
    bool opened = false;
    (void)streamer.feed("{\"name\":\"bad\",\"arguments\":{\"x\":,}}", &opened);
    CHECK(opened);
    CHECK(streamer.invalid());
    CHECK(streamer.state == q27::ToolCallStreamer::INVALID_DONE);
}

static void test_bare_tool_stream_holdback_bounds() {
    CHECK(q27::plausible_bare_tool_prefix("{"));
    CHECK(q27::plausible_bare_tool_prefix("{  \n\"na"));
    CHECK(q27::plausible_bare_tool_prefix("{\"arguments\":{"));
    CHECK(q27::plausible_bare_tool_prefix("{\"tool_name\":\"read\"}"));
    // Round 6 (mode 23) inverted this: an ordinary-keyed object IS plausible
    // now -- it is held, classified, and re-emitted untouched when it is not
    // a call (see test_stream_router_bare_object_without_closers_stays_text).
    CHECK(q27::plausible_bare_tool_prefix("{\"title\":\"ordinary\"}"));
    CHECK(!q27::plausible_bare_tool_prefix("{code block"));
    const std::string complete="{\"name\":\"read\",\"arguments\":{\"text\":\"} kept\"}} tail";
    CHECK(q27::balanced_json_object_prefix_end(complete)==complete.size()-5);
    CHECK(q27::balanced_json_object_prefix_end("{\"name\":\"read\"")==
          std::string::npos);
    q27::IncrementalBareJsonEnd object_scan;
    object_scan.begin(false);
    std::string object_chunks="{\"name\":\"read\",\"arguments\":{\"text\":\"}";
    CHECK(object_scan.advance(object_chunks)==std::string::npos);
    const size_t scanned=object_scan.cursor;
    object_chunks+=" kept\"}} tail";
    CHECK(object_scan.advance(object_chunks)==object_chunks.size()-5);
    CHECK(scanned<object_scan.cursor);
    q27::IncrementalBareJsonEnd mode10_scan;
    mode10_scan.begin(true);
    std::string mode10_chunks="first\", \"payload\":{\"text\":\"}\"}";
    CHECK(mode10_scan.advance(mode10_chunks)==std::string::npos);
    mode10_chunks+="}";
    CHECK(mode10_scan.advance(mode10_chunks)==mode10_chunks.size());
    const std::set<std::string> names={"first","second"};
    CHECK(q27::bare_mode10_signature_position(
              "before first\", \"path\"",names)==7);
    CHECK(q27::bare_mode10_signature_position(
              "before \"first\", \"path\"",names)==std::string::npos);
    CHECK(q27::bare_mode10_probe_start("ordinary fi",names)==9);
    CHECK(q27::bare_mode10_probe_start("ordinary text",names)==std::string::npos);
    q27::JsonStringLexState quoted_chunk;
    quoted_chunk.consume("{\"note\":\"use ");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,quoted_chunk)==std::string::npos);
    CHECK(q27::bare_mode10_probe_start("fi",names,quoted_chunk)==
          std::string::npos);
    quoted_chunk.consume("\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,quoted_chunk)==
          std::string::npos);
    CHECK(q27::bare_object_position(
              "{\"name\":\"first\",\"arguments\":{}}",quoted_chunk)==
          std::string::npos);
    q27::MarkdownFenceLexState fenced_chunk;
    fenced_chunk.consume("```json\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},fenced_chunk)==
          std::string::npos);
    CHECK(q27::bare_mode10_probe_start("fi",names,{},fenced_chunk)==
          std::string::npos);
    CHECK(q27::bare_object_position(
              "{\"name\":\"first\",\"arguments\":{}}",{},fenced_chunk)==
          std::string::npos);
    fenced_chunk.consume("```\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},fenced_chunk)==0);
    q27::MarkdownFenceLexState long_fenced_chunk;
    long_fenced_chunk.consume("``````json\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},long_fenced_chunk)==
          std::string::npos);
    long_fenced_chunk.consume("```\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},long_fenced_chunk)==
          std::string::npos);
    long_fenced_chunk.consume(" ``````not-a-close\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},long_fenced_chunk)==
          std::string::npos);
    long_fenced_chunk.consume("  ``````   \n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},long_fenced_chunk)==0);
    q27::MarkdownFenceLexState tilde_fenced_chunk;
    tilde_fenced_chunk.consume("~~~json\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},tilde_fenced_chunk)==
          std::string::npos);
    tilde_fenced_chunk.consume("~~~not-a-close\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},tilde_fenced_chunk)==
          std::string::npos);
    tilde_fenced_chunk.consume("~~~\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},tilde_fenced_chunk)==0);
    q27::MarkdownFenceLexState quote_fenced_chunk;
    quote_fenced_chunk.consume("> ```json\n> ");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},quote_fenced_chunk)==
          std::string::npos);
    quote_fenced_chunk.consume("displayed\n> ```\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},quote_fenced_chunk)==0);
    q27::MarkdownFenceLexState list_fenced_chunk;
    list_fenced_chunk.consume("- ```json\n  ");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},list_fenced_chunk)==
          std::string::npos);
    list_fenced_chunk.consume("displayed\n      ```\n  ");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},list_fenced_chunk)==
          std::string::npos);
    list_fenced_chunk.consume("```\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},list_fenced_chunk)==0);
    q27::MarkdownFenceLexState nested_list_quote_fence;
    nested_list_quote_fence.consume("- > ~~~json\n  > ");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},nested_list_quote_fence)==
          std::string::npos);
    nested_list_quote_fence.consume("displayed\n  > ~~~\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},nested_list_quote_fence)==0);
    q27::MarkdownFenceLexState nested_list_fence;
    nested_list_fence.consume("- - ```json\n    ");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},nested_list_fence)==
          std::string::npos);
    nested_list_fence.consume("displayed\n    ```\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},nested_list_fence)==0);
    q27::MarkdownFenceLexState indented_fence;
    indented_fence.consume("    ```json\n    ");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},indented_fence)==
          std::string::npos);
    indented_fence.consume("displayed\n    ```\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},indented_fence)==0);
    q27::MarkdownFenceLexState indented_code;
    indented_code.consume("Example:\n\n    ");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"danger\"}",names,{},indented_code)==
          std::string::npos);
    CHECK(q27::bare_object_position(
              "{\"name\":\"first\",\"arguments\":{}}",{},indented_code)==
          std::string::npos);
    q27::MarkdownFenceLexState tab_indented_code;
    tab_indented_code.consume("Example:\n\n\t");
    CHECK(q27::bare_object_position(
              "{\"name\":\"first\",\"arguments\":{}}",{},tab_indented_code)==
          std::string::npos);
    q27::MarkdownFenceLexState blockquote_code;
    blockquote_code.consume("> quoted ");
    CHECK(q27::bare_object_position(
              "{\"name\":\"first\",\"arguments\":{}}",{},blockquote_code)==
          std::string::npos);
    CHECK(q27::bare_mode10_probe_start("fi",names,{},blockquote_code)==
          std::string::npos);
    q27::MarkdownFenceLexState list_code;
    list_code.consume("- example ");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"danger\"}",names,{},list_code)==
          std::string::npos);
    q27::MarkdownFenceLexState list_continuation;
    list_continuation.consume("- example\n  ");
    CHECK(q27::bare_object_position(
              "{\"name\":\"first\",\"arguments\":{}}",{},list_continuation)==
          std::string::npos);
    list_continuation.consume("displayed\nplain\n");
    CHECK(q27::bare_object_position(
              "{\"name\":\"first\",\"arguments\":{}}",{},list_continuation)==0);
    q27::MarkdownFenceLexState list_fence_deindent;
    list_fence_deindent.consume("- ~~~json\n  displayed\n");
    CHECK(q27::bare_mode10_signature_position(
              "  first\", \"path\":\"a\"}",names,{},list_fence_deindent)==
          std::string::npos);
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},list_fence_deindent)==0);
    q27::MarkdownFenceLexState quote_fence_exit;
    quote_fence_exit.consume("> ~~~json\n> displayed\n");
    CHECK(q27::bare_mode10_signature_position(
              "> first\", \"path\":\"a\"}",names,{},quote_fence_exit)==
          std::string::npos);
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},quote_fence_exit)==0);
    q27::MarkdownFenceLexState multiline_inline;
    multiline_inline.consume("Use `example:\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"danger\"}\n`",names,{},
              multiline_inline,false)==std::string::npos);
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"danger\"}",names,{},
              multiline_inline,false)==std::string::npos);
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"danger\"}",names,{},
              multiline_inline,true)==0);
    q27::MarkdownFenceLexState inline_markers;
    inline_markers.consume("Use ```foo``` here\n");
    CHECK(q27::bare_mode10_signature_position(
              "first\", \"path\":\"a\"}",names,{},inline_markers)==0);
    const std::string long_fenced_text=
        "``````json\n{\"name\":\"first\",\"arguments\":{}}\n``````\nafter";
    CHECK(q27::bare_position_is_displayed(
        long_fenced_text,long_fenced_text.find("{\"name\"")));
    CHECK(!q27::bare_position_is_displayed(
        long_fenced_text,long_fenced_text.find("after")));
    const std::string html_code_text=
        "<PRE class=\"example\"><code>{\"name\":\"first\",\"arguments\":{\"path\":\"danger\"}}</code></PRE>\n"
        "{\"name\":\"first\",\"arguments\":{\"path\":\"safe\"}}";
    const size_t displayed_html_call=html_code_text.find("{\"name\"");
    const size_t executable_html_tail=html_code_text.rfind("{\"name\"");
    CHECK(q27::bare_position_is_displayed(html_code_text,displayed_html_call));
    CHECK(!q27::bare_position_is_displayed(html_code_text,executable_html_tail));
    std::string html_prefix,html_residual;
    auto html_calls=q27::parse_bare_tool_calls(
        html_code_text,&html_prefix,nullptr,true,true,&html_residual);
    CHECK(html_calls.size()==1 && html_calls[0].name=="first" &&
          html_calls[0].arguments.value("path",std::string())=="safe");
    CHECK(html_prefix.find("danger")!=std::string::npos);
    const std::string html_comment_text=
        "<!-- {\"name\":\"first\",\"arguments\":{\"path\":\"danger\"}} -->\n"
        "{\"name\":\"first\",\"arguments\":{\"path\":\"safe\"}}";
    const size_t commented_call=html_comment_text.find("{\"name\"");
    const size_t post_comment_call=html_comment_text.rfind("{\"name\"");
    CHECK(q27::bare_position_is_displayed(html_comment_text,commented_call));
    CHECK(!q27::bare_position_is_displayed(html_comment_text,post_comment_call));
    std::string comment_prefix,comment_residual;
    auto comment_calls=q27::parse_bare_tool_calls(
        html_comment_text,&comment_prefix,nullptr,true,true,&comment_residual);
    CHECK(comment_calls.size()==1 && comment_calls[0].name=="first" &&
          comment_calls[0].arguments.value("path",std::string())=="safe");
    CHECK(comment_prefix.find("danger")!=std::string::npos);
    for(const char* tag:{"div","script","style","textarea","custom-panel"}) {
        const std::string raw_html=
            std::string("<")+tag+">\n{\"name\":\"first\",\"arguments\":{\"path\":\"danger\"}}\n</"+
            tag+">\n{\"name\":\"first\",\"arguments\":{\"path\":\"safe\"}}";
        const size_t displayed=raw_html.find("{\"name\"");
        const size_t executable=raw_html.rfind("{\"name\"");
        CHECK(q27::bare_position_is_displayed(raw_html,displayed));
        CHECK(!q27::bare_position_is_displayed(raw_html,executable));
        std::string prefix,residual;
        auto calls=q27::parse_bare_tool_calls(
            raw_html,&prefix,nullptr,true,true,&residual);
        CHECK(calls.size()==1 && calls[0].arguments.value(
              "path",std::string())=="safe");
        CHECK(prefix.find("danger")!=std::string::npos);
    }
    const std::string cdata_html=
        "<![CDATA[{\"name\":\"first\",\"arguments\":{\"path\":\"danger\"}}]]>\n"
        "{\"name\":\"first\",\"arguments\":{\"path\":\"safe\"}}";
    CHECK(q27::bare_position_is_displayed(
        cdata_html,cdata_html.find("{\"name\"")));
    CHECK(!q27::bare_position_is_displayed(
        cdata_html,cdata_html.rfind("{\"name\"")));
    const std::string void_html=
        "<hr>\n{\"name\":\"first\",\"arguments\":{}}";
    CHECK(!q27::bare_position_is_displayed(
        void_html,void_html.find("{\"name\"")));
    const std::string less_than_text=
        "I <3 this. {\"name\":\"first\",\"arguments\":{\"path\":\"safe\"}}";
    CHECK(!q27::bare_position_is_displayed(
        less_than_text,less_than_text.find("{\"name\"")));
    CHECK(q27::parse_bare_tool_calls(
        less_than_text,nullptr,nullptr,true,true,nullptr).size()==1);
    const std::string malformed_container_text=
        "{\"x\":] {\"name\":\"first\",\"arguments\":{\"path\":\"danger\"}}}\n"
        "{\"name\":\"first\",\"arguments\":{\"path\":\"safe\"}}";
    CHECK(!q27::bare_text_position_is_executable(
        malformed_container_text,malformed_container_text.find("{\"name\"")));
    CHECK(q27::bare_text_position_is_executable(
        malformed_container_text,malformed_container_text.rfind("{\"name\"")));
    auto malformed_container_calls=q27::parse_bare_tool_calls(
        malformed_container_text,nullptr,nullptr,true,true,nullptr);
    CHECK(malformed_container_calls.size()==1 &&
          malformed_container_calls[0].arguments.value(
              "path",std::string())=="safe");
    const std::string adjacent_html_text=
        "<<div>{\"name\":\"first\",\"arguments\":{\"path\":\"danger\"}}</div>\n"
        "{\"name\":\"first\",\"arguments\":{\"path\":\"safe\"}}";
    CHECK(q27::bare_position_is_displayed(
        adjacent_html_text,adjacent_html_text.find("{\"name\"")));
    CHECK(!q27::bare_position_is_displayed(
        adjacent_html_text,adjacent_html_text.rfind("{\"name\"")));
    const std::string html_attribute_text=
        "<img alt='>{\"name\":\"first\",\"arguments\":{\"path\":\"danger\"}}'>\n"
        "{\"name\":\"first\",\"arguments\":{\"path\":\"safe\"}}";
    CHECK(q27::bare_position_is_displayed(
        html_attribute_text,html_attribute_text.find("{\"name\"")));
    CHECK(!q27::bare_position_is_displayed(
        html_attribute_text,html_attribute_text.rfind("{\"name\"")));
    const std::string long_tag_prefix(32,'a');
    const std::string long_tag_text=
        "<"+long_tag_prefix+"x>{\"name\":\"first\",\"arguments\":{\"path\":\"danger\"}}"+
        "</"+long_tag_prefix+"y>{\"name\":\"first\",\"arguments\":{\"path\":\"still-danger\"}}"+
        "</"+long_tag_prefix+"x>\n"+
        "{\"name\":\"first\",\"arguments\":{\"path\":\"safe\"}}";
    const size_t long_first=long_tag_text.find("{\"name\"");
    const size_t long_second=long_tag_text.find("{\"name\"",long_first+1);
    CHECK(q27::bare_position_is_displayed(long_tag_text,long_first));
    CHECK(q27::bare_position_is_displayed(long_tag_text,long_second));
    auto long_calls=q27::parse_bare_tool_calls(
        long_tag_text,nullptr,nullptr,true,true,nullptr);
    CHECK(long_calls.size()==1 && long_calls[0].arguments.value(
          "path",std::string())=="safe");
    q27::MarkdownFenceLexState html_chunk;
    html_chunk.consume("<PrE><co");
    CHECK(q27::bare_object_position(
              "de>{\"name\":\"first\",\"arguments\":{}}",{},html_chunk)==
          std::string::npos);
    html_chunk.consume("de>shown</code></pre>\n");
    CHECK(q27::bare_object_position(
              "{\"name\":\"first\",\"arguments\":{}}",{},html_chunk)==0);
    const std::string markdown_link_text=
        "[example](https://example.invalid/{\"name\":\"first\",\"arguments\":{\"path\":\"danger\"}})\n"
        "{\"name\":\"first\",\"arguments\":{\"path\":\"safe\"}}";
    CHECK(q27::bare_position_is_displayed(
        markdown_link_text,markdown_link_text.find("{\"name\"")));
    CHECK(!q27::bare_position_is_displayed(
        markdown_link_text,markdown_link_text.rfind("{\"name\"")));
    auto markdown_link_calls=q27::parse_bare_tool_calls(
        markdown_link_text,nullptr,nullptr,true,true,nullptr);
    CHECK(markdown_link_calls.size()==1 && markdown_link_calls[0].arguments.value(
          "path",std::string())=="safe");
    const std::string markdown_reference_text=
        "[example]: https://example.invalid/{\"name\":\"first\",\"arguments\":{\"path\":\"danger\"}}\n"
        "{\"name\":\"first\",\"arguments\":{\"path\":\"safe\"}}";
    CHECK(q27::bare_position_is_displayed(
        markdown_reference_text,markdown_reference_text.find("{\"name\"")));
    auto markdown_reference_calls=q27::parse_bare_tool_calls(
        markdown_reference_text,nullptr,nullptr,true,true,nullptr);
    CHECK(markdown_reference_calls.size()==1 &&
          markdown_reference_calls[0].arguments.value(
              "path",std::string())=="safe");
    q27::MarkdownFenceLexState markdown_link_chunk;
    markdown_link_chunk.consume("[example](https://example.invalid/");
    CHECK(q27::bare_object_position(
              "{\"name\":\"first\",\"arguments\":{}}",{},markdown_link_chunk)==
          std::string::npos);
    markdown_link_chunk.consume(")\n");
    CHECK(q27::bare_object_position(
              "{\"name\":\"first\",\"arguments\":{}}",{},markdown_link_chunk)==0);
    const std::string multiline_markdown_link_text=
        "[example](https://example.invalid/\n \"{\"name\":\"first\",\"arguments\":{\"path\":\"danger\"}}\")\n"
        "{\"name\":\"first\",\"arguments\":{\"path\":\"safe\"}}";
    CHECK(q27::bare_position_is_displayed(multiline_markdown_link_text,
          multiline_markdown_link_text.find("{\"name\"")));
    auto multiline_markdown_link_calls=q27::parse_bare_tool_calls(
        multiline_markdown_link_text,nullptr,nullptr,true,true,nullptr);
    CHECK(multiline_markdown_link_calls.size()==1 &&
          multiline_markdown_link_calls[0].arguments.value(
              "path",std::string())=="safe");
    q27::BareToolTextHoldback holdback;
    std::string visible;
    std::vector<q27::ToolCall> held_calls;
    auto emit_visible=[&](const std::string& text) { visible+=text; };
    auto classify=[&](const std::string& source,bool allow_repair,auto&& emit_text) {
        std::string prefix,residual;
        auto calls=q27::parse_bare_tool_calls(
            source,&prefix,nullptr,true,allow_repair,&residual);
        if(calls.empty()) return q27::BareToolCandidateResult{};
        size_t cursor=0;
        for(auto& call:calls) {
            emit_text(source.substr(cursor,call.source_begin-cursor));
            cursor=call.source_end;
            held_calls.push_back(call);
        }
        emit_text(source.substr(cursor));
        return q27::BareToolCandidateResult{true,true};
    };
    const std::string nested_stream_json=
        R"({"payload":{"name":"first","arguments":{"path":"danger"}}})";
    for(char ch:nested_stream_json)
        holdback.route(std::string(1,ch),names,emit_visible,classify);
    holdback.finish(false,names,emit_visible,classify);
    CHECK(visible==nested_stream_json && held_calls.empty());
    q27::BareToolTextHoldback unterminated_holdback;
    std::string unterminated_visible;
    const std::string unterminated_stream_json=
        R"({"payload":{"name":"first","arguments":{"path":"danger"}})";
    for(char ch:unterminated_stream_json)
        unterminated_holdback.route(
            std::string(1,ch),names,
            [&](const std::string& text) { unterminated_visible+=text; },classify);
    unterminated_holdback.finish(
        true,names,
        [&](const std::string& text) { unterminated_visible+=text; },classify);
    CHECK(unterminated_visible==unterminated_stream_json && held_calls.empty());
    visible.clear();
    holdback.route("intro {\"name\":\"fi",names,emit_visible,classify);
    CHECK(visible=="intro " && held_calls.empty());
    holdback.route("rst\",\"arguments\":{}} tail",names,emit_visible,classify);
    CHECK(visible=="intro  tail");
    CHECK(held_calls.size()==1 && held_calls[0].name=="first");
    CHECK(visible.find("\"name\":\"first\"")==std::string::npos);

    q27::BareToolTextHoldback prose_brace_holdback;
    std::string prose_brace_visible;
    const size_t calls_before_prose_brace=held_calls.size();
    prose_brace_holdback.route("I can use {",names,
        [&](const std::string& text) { prose_brace_visible+=text; },classify);
    prose_brace_holdback.route(
        "{\"name\":\"first\",\"arguments\":{}}",names,
        [&](const std::string& text) { prose_brace_visible+=text; },classify);
    prose_brace_holdback.finish(false,names,
        [&](const std::string& text) { prose_brace_visible+=text; },classify);
    CHECK(prose_brace_visible=="I can use {");
    CHECK(held_calls.size()==calls_before_prose_brace+1 &&
          held_calls.back().name=="first");

    const json raw_tools=json::array({
        {{"type","function"},{"function",{{"name","Write"},{"parameters",{
            {"type","object"},{"properties",{{"content",{{"type","string"}}}}}
        }}}}}
    });
    const std::set<std::string> raw_names={"Write"};
    q27::BareToolTextHoldback raw_holdback;
    std::string raw_visible;
    std::vector<q27::ToolCall> raw_calls;
    auto classify_raw=[&](const std::string& source,bool allow_repair,auto&& emit_text) {
        std::string prefix,residual;
        auto calls=q27::parse_bare_tool_calls(
            source,&prefix,&raw_tools,true,allow_repair,&residual);
        if(calls.empty()) return q27::BareToolCandidateResult{};
        size_t cursor=0;
        for(auto& call:calls) {
            emit_text(source.substr(cursor,call.source_begin-cursor));
            cursor=call.source_end;
            raw_calls.push_back(call);
        }
        emit_text(source.substr(cursor));
        return q27::BareToolCandidateResult{true,true};
    };
    const std::string raw_source=R"({"name":"Write","arguments":{"content":"const s = "x";"}})";
    raw_holdback.route(raw_source,raw_names,
        [&](const std::string& text) { raw_visible+=text; },classify_raw);
    CHECK(raw_visible.empty() && raw_calls.empty());
    raw_holdback.route(" tail",raw_names,
        [&](const std::string& text) { raw_visible+=text; },classify_raw);
    CHECK(raw_visible.empty() && raw_calls.empty());
    raw_holdback.finish(true,raw_names,
        [&](const std::string& text) { raw_visible+=text; },classify_raw);
    CHECK(raw_visible==" tail");
    CHECK(raw_calls.size()==1 && raw_calls[0].name=="Write");
    CHECK(raw_calls[0].arguments["content"]=="const s = \"x\";");

    std::string raw_prefix,raw_remaining;
    auto raw_with_tail=q27::parse_bare_tool_calls(
        raw_source+" tail",&raw_prefix,&raw_tools,true,true,&raw_remaining);
    CHECK(raw_with_tail.size()==1);
    CHECK(raw_with_tail[0].source_begin==0);
    CHECK(raw_with_tail[0].source_end==raw_source.size());
    CHECK(raw_remaining==" tail");

    std::vector<std::string> routed;
    const size_t routed_calls=q27::route_bare_tool_sequence(
        raw_source+" tail",raw_with_tail,
        [&](const std::string& text) {
            if(!text.empty()) routed.push_back("text:"+text);
        },
        [&](const q27::ToolCall& call) {
            routed.push_back("call:"+call.name);
            return true;
        });
    CHECK(routed_calls==1);
    CHECK(routed==std::vector<std::string>({"call:Write","text: tail"}));

    q27::BareToolTextHoldback boundary_holdback;
    std::string boundary_visible;
    raw_calls.clear();
    boundary_holdback.route(raw_source,raw_names,
        [&](const std::string& text) { boundary_visible+=text; },classify_raw);
    boundary_holdback.finish(false,raw_names,
        [&](const std::string& text) { boundary_visible+=text; },classify_raw);
    CHECK(raw_calls.empty());
    CHECK(boundary_visible==raw_source);

    q27::BareToolTextHoldback rejected_holdback;
    std::string rejected_visible;
    std::vector<q27::ToolCall> accepted_after_rejection;
    const json filtered_tools=json::array({
        {{"type","function"},{"function",{{"name","first"},
            {"parameters",{{"type","object"},{"properties",json::object()}}}}}},
        {{"type","function"},{"function",{{"name","second"},
            {"parameters",{{"type","object"},{"properties",json::object()}}}}}}
    });
    auto reject_second=[&](const std::string& source,bool allow_repair,
                           auto&& emit_text) {
        std::string prefix,residual;
        auto calls=q27::parse_bare_tool_calls(
            source,&prefix,&filtered_tools,true,allow_repair,&residual);
        if(calls.empty()) return q27::BareToolCandidateResult{};
        size_t cursor=0,accepted=0;
        for(auto& call:calls) {
            emit_text(source.substr(cursor,call.source_begin-cursor));
            cursor=call.source_end;
            if(call.name=="first") {
                accepted_after_rejection.push_back(call);
                accepted++;
            } else emit_text(source.substr(
                call.source_begin,call.source_end-call.source_begin));
        }
        emit_text(source.substr(cursor));
        return q27::BareToolCandidateResult{true,accepted!=0};
    };
    const std::string rejected_source=
        "{\"name\":\"second\",\"arguments\":{}}";
    rejected_holdback.route(rejected_source,names,
        [&](const std::string& text) { rejected_visible+=text; },reject_second);
    CHECK(rejected_visible==rejected_source);
    rejected_holdback.route("first\", \"path\":\"safe\"}",names,
        [&](const std::string& text) { rejected_visible+=text; },reject_second);
    rejected_holdback.finish(true,names,
        [&](const std::string& text) { rejected_visible+=text; },reject_second);
    CHECK(accepted_after_rejection.size()==1 &&
          accepted_after_rejection[0].name=="first");
    CHECK(rejected_visible==rejected_source);

    q27::BareToolTextHoldback ordinary_holdback;
    std::string ordinary_visible;
    const std::string ordinary_json="{\"title\":\"ordinary\"}";
    ordinary_holdback.route(ordinary_json,names,
        [&](const std::string& text) { ordinary_visible+=text; },classify);
    ordinary_holdback.finish(false,names,
        [&](const std::string& text) { ordinary_visible+=text; },classify);
    CHECK(ordinary_visible==ordinary_json);

    q27::BareToolTextHoldback named_json_holdback;
    std::string named_json_visible;
    const std::string named_json="{\"name\":\"Alice\",\"age\":30}";
    named_json_holdback.route(named_json,names,
        [&](const std::string& text) { named_json_visible+=text; },classify);
    CHECK(named_json_visible==named_json);
    named_json_holdback.route(" tail",names,
        [&](const std::string& text) { named_json_visible+=text; },classify);
    CHECK(named_json_visible==named_json+" tail");
}

// 2026-08-22: the fp16-KV agentic arm lost 11 calls in 498 messages to ONE
// shape. The model omits the <tool_call> opener on the last call of a batch
// (or on a lone call after prose), writes a complete <function=...></function>
// and stops. The splitter never sees a wrapper, so the bytes reach
// BareToolTextHoldback as TEXT, and the holdback only knew JSON-object and
// mode-10 openers: the call streamed to the client verbatim, classify never
// ran, nothing was logged. The non-stream path recovers the same bytes via
// mode 17. llama.cpp is grammar-constrained and cannot emit the shape: 0/293
// messages on the same tasks. 9 of the 11 cost the last call of a batch; the
// other 2 were the only call in the turn, Claude Code took the text as the
// final answer, and the trial ended at turn 3.
// Issue #38 (cosmicnag, 2026-08-24): "I can see the tool call xml in the
// thinking section here and there ... and then the actual tool call never
// happens." The model emits a complete <tool_call> inside <think>; the
// splitter routes everything between <think> and </think> to THINK, and that
// channel never consulted the tool parser. Identical bytes in TEXT parsed
// fine, so this was an uncovered channel rather than a parse failure. The
// doubled </tool_call> in the report is a red herring -- it parses either way.
static void test_tool_call_inside_reasoning() {
    json body=json::parse(R"({"tools":[
        {"name":"read","input_schema":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"}}}}]})");
    const json tools=q27::anthropic_tools_json(body);
    const std::string inner=
        "<function=read>\n<parameter=path>\n/tmp/x.md\n</parameter>\n"
        "<parameter=limit>\n160\n</parameter>\n</function>\n";
    // mirror server.cu's route(): adjacent same-channel pieces are MERGED
    // before the resolver sees them. Feeding raw 7-byte fragments instead
    // tests a shape the handler never produces.
    auto resolve=[&](const std::string& gen) {
        q27::StreamSplitter sp;
        std::vector<std::pair<q27::StreamSplitter::Chan,std::string>> segs;
        auto route=[&](q27::StreamSplitter::Chan ch,const std::string& t) {
            if(t.empty()) return;
            if(!segs.empty() && segs.back().first==ch) segs.back().second+=t;
            else segs.emplace_back(ch,t);
        };
        for(size_t i=0;i<gen.size();i+=7)
            for(auto& x:sp.feed(gen.substr(i,7))) route(x.first,x.second);
        for(auto& x:sp.flush()) route(x.first,x.second);
        return q27::resolve_ordered_tool_segments(
            segs,&tools,false,[](const std::string&,size_t){return true;});
    };
    // the reported shape, doubled closer and all
    {
        auto o=resolve("<think>\nNext, read the temp file.\n<tool_call>\n"+inner+
                       "</tool_call>\n</tool_call>\n</think>\n");
        CHECK(o.calls.size()==1);
        CHECK(o.calls.size()==1 && o.calls[0].name=="read");
        CHECK(o.calls.size()==1 &&
              o.calls[0].arguments.value("path",std::string())=="/tmp/x.md");
        CHECK(o.reasoning.find("<function=read>")==std::string::npos);
    }
    // single closer, same result
    {
        auto o=resolve("<think>\nplanning\n<tool_call>\n"+inner+"</tool_call>\n</think>\n");
        CHECK(o.calls.size()==1 && o.calls[0].name=="read");
    }
    // bare, no wrapper, inside reasoning
    {
        auto o=resolve("<think>\nplanning\n"+inner+"</think>\n");
        CHECK(o.calls.size()==1 && o.calls[0].name=="read");
    }
    // SAFETY: a fenced example inside reasoning is the model writing ABOUT a
    // call. It must stay reasoning -- this is the in-band-signalling hazard,
    // and parse_bare_tool_calls alone would execute it.
    {
        const std::string fenced="<think>\nlike this:\n```\n<tool_call>\n"+inner+
                                 "</tool_call>\n```\n</think>\n";
        auto o=resolve(fenced);
        CHECK(o.calls.empty());
        CHECK(o.reasoning.find("<function=read>")!=std::string::npos);
    }
    // SAFETY: an inline-code mention stays reasoning
    {
        auto o=resolve("<think>\nI could call `<tool_call>` here\n</think>\n");
        CHECK(o.calls.empty());
    }
    // ordinary reasoning is untouched, and still reaches out.reasoning
    {
        auto o=resolve("<think>\njust thinking about the problem\n</think>\n");
        CHECK(o.calls.empty());
        CHECK(o.reasoning.find("just thinking")!=std::string::npos);
    }
    // a call in reasoning FOLLOWED by one in text yields both, in order
    {
        auto o=resolve("<think>\nplan\n"+inner+"</think>\nNow doing it.\n"+inner);
        CHECK(o.calls.size()==2);
    }
}

static void test_bare_tool_stream_holdback_native_xml() {
    json body=json::parse(R"({"tools":[
        {"name":"Read","input_schema":{"type":"object","properties":{"file_path":{"type":"string"}}}},
        {"name":"Bash","input_schema":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"}}}}]})");
    const json tools=q27::anthropic_tools_json(body);
    const std::set<std::string> names={"Read","Bash"};
    struct Run { std::string visible; std::vector<q27::ToolCall> calls; int classified=0; };
    auto drive=[&](const std::vector<std::string>& pieces,bool boundary_before) {
        Run r;
        q27::BareToolTextHoldback hb;
        auto emit=[&](const std::string& t) { r.visible+=t; };
        auto classify=[&](const std::string& source,bool allow_repair,auto&& visible) {
            r.classified++;
            std::string prefix,residual;
            auto calls=q27::parse_bare_tool_calls(
                source,&prefix,&tools,true,allow_repair,&residual);
            if(calls.empty()) return q27::BareToolCandidateResult{};
            size_t cursor=0;
            for(auto& c:calls) {
                visible(source.substr(cursor,c.source_begin-cursor));
                cursor=c.source_end;
                r.calls.push_back(c);
            }
            visible(source.substr(cursor));
            return q27::BareToolCandidateResult{true,true};
        };
        // the server calls finish(false)+reset_context() at every TOOL boundary
        if(boundary_before) { hb.finish(false,names,emit,classify); hb.reset_context(); }
        for(auto& p:pieces) hb.route(p,names,emit,classify);
        hb.finish(true,names,emit,classify);
        return r;
    };
    auto bytes=[](const std::string& s,size_t n) {
        std::vector<std::string> v;
        for(size_t i=0;i<s.size();i+=n) v.push_back(s.substr(i,n));
        return v;
    };
    const std::string fifth=
        "\n<function=Read>\n<parameter=file_path>\n/workspace/TASK.md\n</parameter>\n</function>\n";
    // (a) the live shape: last call of a batch, right after a wrapped call
    for(size_t n:{1,4,4096}) {
        auto r=drive(bytes(fifth,n),true);
        CHECK(r.calls.size()==1 && r.calls[0].name=="Read" &&
              r.calls[0].arguments.value("file_path",std::string())=="/workspace/TASK.md");
        CHECK(q27::strip_ws2(r.visible).empty());
    }
    // (b) prose, then a lone bare call: the prose streams, the call executes
    const std::string bash=
        "\n\nLet me look at the existing source.\n\n<function=Bash>\n<parameter=command>\n"
        "ls -la /workspace/src/ && echo \"exit: $?\"\n</parameter>\n<parameter=description>\n"
        "Check src\n</parameter>\n</function>\n";
    for(size_t n:{1,7,4096}) {
        auto r=drive(bytes(bash,n),false);
        CHECK(r.calls.size()==1 && r.calls[0].name=="Bash");
        CHECK(r.calls.size()==1 && r.calls[0].arguments.value("command",std::string())==
              "ls -la /workspace/src/ && echo \"exit: $?\"");
        CHECK(q27::strip_ws2(r.visible)=="Let me look at the existing source.");
    }
    // (c) a fenced example is displayed, never executed
    const std::string fenced=
        "Use it like this:\n```\n<function=Read>\n<parameter=file_path>\nx\n</parameter>\n</function>\n```\ndone\n";
    { auto r=drive(bytes(fenced,3),false); CHECK(r.calls.empty() && r.visible==fenced); }
    // (d) an inline-code mention is displayed
    const std::string inlined="call `<function=Read>` to read\n";
    { auto r=drive(bytes(inlined,2),false); CHECK(r.calls.empty() && r.visible==inlined); }
    // (e) </function> inside a value does not end the call early
    const std::string nested=
        "<function=Bash>\n<parameter=command>\necho '</function>' > f\n</parameter>\n</function>\n";
    {
        auto r=drive(bytes(nested,5),false);
        CHECK(r.calls.size()==1 &&
              r.calls[0].arguments.value("command",std::string())=="echo '</function>' > f");
        CHECK(q27::strip_ws2(r.visible).empty());
    }
    // (f) prose after the call keeps streaming
    {
        auto r=drive(bytes(fifth+"Reading the task next.\n",4),true);
        CHECK(r.calls.size()==1);
        CHECK(q27::strip_ws2(r.visible)=="Reading the task next.");
    }
    // (g) two bare calls back to back
    {
        auto r=drive(bytes(fifth+"<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n",4),true);
        CHECK(r.calls.size()==2 && r.calls[1].name=="Bash");
        CHECK(q27::strip_ws2(r.visible).empty());
    }
    // (h) a stray </tool_call> after a bare call is swallowed with it
    {
        auto r=drive(bytes(fifth+"</tool_call>\n",4),true);
        CHECK(r.calls.size()==1);
        CHECK(r.visible.find("</tool_call>")==std::string::npos);
    }
    // (i) a half-written opener at EOS is shown, not executed
    {
        auto r=drive(bytes("\n<function=Re",2),true);
        CHECK(r.calls.empty() && r.visible=="\n<function=Re");
    }
    // (j) JSON bare calls still work exactly as before
    {
        auto r=drive(bytes("\n{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/a\"}}\n",3),false);
        CHECK(r.calls.size()==1 && r.calls[0].name=="Read");
        CHECK(q27::strip_ws2(r.visible).empty());
    }
}

static void test_ordered_tool_segments_preserve_first_call_and_rejected_text() {
    json tools=json::parse(R"([
      {"type":"function","function":{"name":"first","parameters":{
        "type":"object","properties":{"path":{"type":"string"}},
        "required":["path"]}}},
      {"type":"function","function":{"name":"second","parameters":{"type":"object"}}}
    ])");
    const std::string incomplete_before_wrapper=
        "{\"name\":\"first\",\"arguments\":{\"path\":\"unfinished\"";
    const std::vector<std::pair<q27::StreamSplitter::Chan,std::string>>
        incomplete_then_wrapped={
            {q27::StreamSplitter::TEXT,incomplete_before_wrapper},
            {q27::StreamSplitter::TOOL,
             "{\"name\":\"second\",\"arguments\":{}}"}};
    auto boundary_order=q27::resolve_ordered_tool_segments(
        incomplete_then_wrapped,&tools,true,
        [](const std::string&,size_t accepted) { return accepted==0; });
    CHECK(boundary_order.calls.size()==1 &&
          boundary_order.calls[0].name=="second");
    CHECK(boundary_order.text==incomplete_before_wrapper);
    std::string mode10_prefix,mode10_remaining;
    CHECK(q27::parse_bare_tool_calls(
              "first\", \"path\":\"a\"}",&mode10_prefix,&tools,true,false,
              &mode10_remaining).size()==1);
    CHECK(q27::parse_bare_tool_calls(
              "first\", \"path\":\"a\"",&mode10_prefix,&tools,true,false,
              &mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "```json\nfirst\", \"path\":\"a\"}\n```",&mode10_prefix,
              &tools,true,false,&mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "``````json\nfirst\", \"path\":\"a\"}\n``````",&mode10_prefix,
              &tools,true,false,&mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "``````json\n ``````not-a-close\nfirst\", \"path\":\"a\"}\n"
              "  ``````   \n",&mode10_prefix,&tools,true,false,
              &mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "~~~json\nfirst\", \"path\":\"a\"}\n~~~\n",&mode10_prefix,
              &tools,true,false,&mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "> ```json\n> first\", \"path\":\"a\"}\n> ```\n",
              &mode10_prefix,&tools,true,false,&mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "- ```json\n  first\", \"path\":\"a\"}\n  ```\n",
              &mode10_prefix,&tools,true,false,&mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "- > ~~~json\n  > first\", \"path\":\"a\"}\n  > ~~~\n",
              &mode10_prefix,&tools,true,false,&mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "- - ```json\n    first\", \"path\":\"a\"}\n    ```\n",
              &mode10_prefix,&tools,true,false,&mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "- example\n\n    ```json\n    first\", \"path\":\"a\"}\n    ```\n",
              &mode10_prefix,&tools,true,false,&mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "Here is \"unfinished\n```json\nfirst\", \"path\":\"a\"}\n```\n",
              &mode10_prefix,&tools,true,false,&mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "Use ```first\", \"path\":\"a\"}``` as an example",
              &mode10_prefix,&tools,true,false,&mode10_remaining).empty());
    std::string inline_prefix,inline_remaining;
    auto inline_calls=q27::parse_bare_tool_calls(
        "Use ```foo``` here\nfirst\", \"path\":\"a\"}",
        &inline_prefix,&tools,true,false,&inline_remaining);
    CHECK(inline_calls.size()==1 && inline_calls[0].name=="first");
    std::string literal_prefix,literal_remaining;
    auto literal_tick_calls=q27::parse_bare_tool_calls(
        "Use ` as a literal\nfirst\", \"path\":\"a\"}",
        &literal_prefix,&tools,true,false,&literal_remaining);
    CHECK(literal_tick_calls.size()==1 && literal_tick_calls[0].name=="first");
    auto same_line_literal_calls=q27::parse_bare_tool_calls(
        "Use ` literally; first\", \"path\":\"a\"}",
        &literal_prefix,&tools,true,false,&literal_remaining);
    CHECK(same_line_literal_calls.size()==1 &&
          same_line_literal_calls[0].name=="first");
    auto same_line_object_calls=q27::parse_bare_tool_calls(
        "Use ` literally; {\"name\":\"first\",\"arguments\":{}}",
        &literal_prefix,&tools,true,false,&literal_remaining);
    CHECK(same_line_object_calls.size()==1 &&
          same_line_object_calls[0].name=="first");
    auto escaped_tick_calls=q27::parse_bare_tool_calls(
        "Use \\` as a literal\nfirst\", \"path\":\"a\"}",
        &literal_prefix,&tools,true,false,&literal_remaining);
    CHECK(escaped_tick_calls.size()==1 && escaped_tick_calls[0].name=="first");
    auto invalid_opener_calls=q27::parse_bare_tool_calls(
        " ```foo```\nfirst\", \"path\":\"a\"}",
        &literal_prefix,&tools,true,false,&literal_remaining);
    CHECK(invalid_opener_calls.size()==1 &&
          invalid_opener_calls[0].name=="first");
    const std::string executable_call=
        "{\"name\":\"first\",\"arguments\":{\"path\":\"a\"}}";
    CHECK(q27::parse_bare_tool_calls(
              "> "+executable_call,&mode10_prefix,&tools,true,true,
              &mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "- "+executable_call,&mode10_prefix,&tools,true,true,
              &mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "- example\n  "+executable_call,&mode10_prefix,&tools,true,true,
              &mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "Example:\n\n    "+executable_call,&mode10_prefix,&tools,true,true,
              &mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "Example:\n\n\t"+executable_call,&mode10_prefix,&tools,true,true,
              &mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              R"(> {"name":"first","arguments":{"path":"a "b""}})",
              &mode10_prefix,&tools,true,true,&mode10_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "> {\"name\":{\"path\":\"a\"}{\"name\":{\"path\":\"b\"}suffix",
              &mode10_prefix,&tools,true,true,&mode10_remaining).empty());
    auto executable_calls=q27::parse_bare_tool_calls(
        "plain\n"+executable_call,&mode10_prefix,&tools,true,true,
        &mode10_remaining);
    CHECK(executable_calls.size()==1 && executable_calls[0].name=="first");
    CHECK(q27::parse_bare_tool_calls(
              "{\"note\":\"use first\", \"path\":\"a\"}",
              &mode10_prefix,&tools,true,false,&mode10_remaining).empty());
    const std::string preserved="before arguments\": untouched\n";
    std::vector<std::pair<q27::StreamSplitter::Chan,std::string>> segments={
        {q27::StreamSplitter::TEXT,preserved+"{\"name\":\"first\",\"arguments\":{}}tail"},
        {q27::StreamSplitter::TOOL,"{\"name\":\"second\",\"arguments\":{}}"},
        {q27::StreamSplitter::TOOL,"{\"name\":"}
    };
    std::string repeated_invalid;
    for(int i=0;i<64;i++) repeated_invalid+="first\", \"path\":not-json\n";
    std::string repeated_prefix,repeated_remaining;
    CHECK(q27::parse_bare_tool_calls(
              repeated_invalid,&repeated_prefix,&tools,true,true,
              &repeated_remaining).empty());
    CHECK(q27::parse_bare_tool_calls(
              "first\", \"bad\": second\", \"path\":\"danger\"}",
              &repeated_prefix,&tools,true,true,
              &repeated_remaining).empty());
    auto out=q27::resolve_ordered_tool_segments(
        segments,&tools,true,
        [](const std::string&,size_t accepted){ return accepted==0; });
    CHECK(out.calls.size()==1);
    CHECK(out.calls[0].name=="first");
    CHECK(out.text.find(preserved)!=std::string::npos);
    CHECK(out.text.find("tail")!=std::string::npos);
    CHECK(out.text.find("second")!=std::string::npos);
    CHECK(out.text.find("{\"name\":")!=std::string::npos);
    CHECK(out.parts.size()==3);
    CHECK(out.parts[0].kind==q27::OrderedToolPart::Kind::Text);
    CHECK(out.text.substr(out.parts[0].text_begin,
                          out.parts[0].text_end-out.parts[0].text_begin)==preserved);
    CHECK(out.parts[1].kind==q27::OrderedToolPart::Kind::Call &&
          out.parts[1].call_index==0);
    CHECK(out.parts[2].kind==q27::OrderedToolPart::Kind::Text);
    json anthropic_content=json::array();
    CHECK(q27::append_anthropic_ordered_content(
              anthropic_content,out,
              [](const q27::ToolCall& call,size_t) {
                  return json{{"type","tool_use"},{"name",call.name}};
              })==3);
    CHECK(anthropic_content.size()==3);
    CHECK(anthropic_content[0]["type"]=="text" &&
          anthropic_content[0]["text"]==preserved);
    CHECK(anthropic_content[1]["type"]=="tool_use" &&
          anthropic_content[1]["name"]=="first");
    CHECK(anthropic_content[2]["type"]=="text" &&
          anthropic_content[2]["text"].get<std::string>().find("tail")==0);
    q27::OrderedToolOutput whitespace;
    whitespace.append_visible_text("\n  indented code  \n");
    json whitespace_content=json::array();
    CHECK(q27::append_anthropic_ordered_content(
              whitespace_content,whitespace,
              [](const q27::ToolCall&,size_t) { return json(); })==1);
    CHECK(whitespace_content[0]["text"]=="\n  indented code  \n");

    const std::string batch=
        "{\"name\":{\"path\":\"a\"}{\"name\":{\"path\":\"b\"}suffix";
    std::string prefix,remaining;
    auto recovered=q27::parse_bare_tool_calls(batch,&prefix,&tools,true,true,&remaining);
    CHECK(recovered.size()==1);
    CHECK(remaining.find("suffix")!=std::string::npos);
    CHECK(remaining.find("name")!=std::string::npos);

    const std::string dropped_batch=
        "before first\", \"path\":\"a\"} "
        "{\"name\":\"second\",\"arguments\":{}} after";
    std::string dropped_prefix,dropped_remaining;
    auto dropped_calls=q27::parse_bare_tool_calls(
        dropped_batch,&dropped_prefix,&tools,true,true,&dropped_remaining);
    CHECK(dropped_calls.size()==2);
    CHECK(dropped_calls.size()==2 && dropped_calls[0].name=="first");
    CHECK(dropped_calls.size()==2 && dropped_calls[1].name=="second");
    CHECK(dropped_calls.size()==2 &&
          dropped_calls[0].source_end<=dropped_calls[1].source_begin);
    CHECK(dropped_calls.size()==2 && dropped_calls[0].raw.find("second")==std::string::npos);
    CHECK(dropped_remaining.find("before")!=std::string::npos);
    CHECK(dropped_remaining.find("after")!=std::string::npos);
    CHECK(dropped_remaining.find("first")==std::string::npos);
    CHECK(dropped_remaining.find("second")==std::string::npos);
    const std::string invalid_then_valid_same_line=
        "first\", \"path\":not-json}; second\", \"path\":\"a\"}";
    std::string same_line_prefix,same_line_remaining;
    auto same_line_calls=q27::parse_bare_tool_calls(
        invalid_then_valid_same_line,&same_line_prefix,&tools,true,true,
        &same_line_remaining);
    CHECK(same_line_calls.empty());
    CHECK(same_line_remaining.find("second")!=std::string::npos);
    const std::string backticks_in_argument=
        "{\"name\":\"first\",\"arguments\":{\"path\":\"``````\"}} "
        "{\"name\":\"second\",\"arguments\":{}}";
    std::string backtick_prefix,backtick_remaining;
    auto backtick_calls=q27::parse_bare_tool_calls(
        backticks_in_argument,&backtick_prefix,&tools,true,true,
        &backtick_remaining);
    CHECK(backtick_calls.size()==2);
    CHECK(backtick_calls.size()==2 && backtick_calls[0].name=="first");
    CHECK(backtick_calls.size()==2 && backtick_calls[1].name=="second");
    const std::string invalid_then_valid=
        "first\", \"path\": not-json\nsecond\", \"path\":\"a\"}";
    std::string isolated_prefix,isolated_remaining;
    auto isolated_calls=q27::parse_bare_tool_calls(
        invalid_then_valid,&isolated_prefix,&tools,true,true,
        &isolated_remaining);
    CHECK(isolated_calls.empty());
    const std::string invalid_with_quoted_candidate=
        "first\", \"bad\":not-json, \"note\":\"first\", \"path\":\"danger\"}";
    std::string quoted_nested_prefix,quoted_nested_remaining;
    CHECK(q27::parse_bare_tool_calls(
              invalid_with_quoted_candidate,&quoted_nested_prefix,&tools,true,true,
              &quoted_nested_remaining).empty());
    CHECK(isolated_remaining.find("first")!=std::string::npos);
    const std::string registry_reverse=
        "second\", \"value\":1} first\", \"path\":\"a\"}";
    std::string reverse_prefix,reverse_remaining;
    auto reverse_calls=q27::parse_bare_tool_calls(
        registry_reverse,&reverse_prefix,&tools,true,true,&reverse_remaining);
    CHECK(reverse_calls.size()==2);
    CHECK(reverse_calls.size()==2 && reverse_calls[0].name=="second");
    CHECK(reverse_calls.size()==2 && reverse_calls[1].name=="first");
    CHECK(reverse_calls.size()==2 &&
          reverse_calls[0].source_end<=reverse_calls[1].source_begin);
    const json overlapping_tools=json::parse(R"([
      {"type":"function","function":{"name":"foo","parameters":{"type":"object"}}},
      {"type":"function","function":{"name":"oo","parameters":{"type":"object"}}}
    ])");
    std::string overlap_prefix,overlap_remaining;
    auto overlap_calls=q27::parse_bare_tool_calls(
        "foo\", \"x\":1}",&overlap_prefix,&overlapping_tools,true,true,
        &overlap_remaining);
    CHECK(overlap_calls.size()==1 && overlap_calls[0].name=="foo");
    const std::string post_call_prose=
        "{\"name\":\"first\",\"arguments\":{\"path\":\"a\"}} after "
        "second\", \"value\":1}";
    std::string post_prefix,post_remaining;
    auto post_calls=q27::parse_bare_tool_calls(
        post_call_prose,&post_prefix,&tools,true,true,&post_remaining);
    CHECK(post_calls.size()==1 && post_calls[0].name=="first");
    CHECK(post_remaining.find("second")!=std::string::npos);

    const std::string dropped_single="first\", \"path\":\"a\"} trailing prose";
    std::string single_prefix,single_remaining;
    auto single_call=q27::parse_bare_tool_calls(
        dropped_single,&single_prefix,&tools,true,true,&single_remaining);
    CHECK(single_call.size()==1 && single_call[0].name=="first");
    CHECK(single_call.size()==1 && single_call[0].raw.find("trailing prose")==std::string::npos);
    CHECK(single_remaining.find("trailing prose")!=std::string::npos);

    std::vector<std::pair<q27::StreamSplitter::Chan,std::string>> parallel_bare={
        {q27::StreamSplitter::TEXT,
         "before {\"name\":\"first\",\"arguments\":{\"path\":\"a\"}} middle "
         "{\"name\":\"second\",\"arguments\":{}} after"}
    };
    auto one=q27::resolve_ordered_tool_segments(
        parallel_bare,&tools,true,
        [](const std::string&,size_t accepted){ return accepted==0; });
    CHECK(one.calls.size()==1 && one.calls[0].name=="first");
    CHECK(one.text.find("second")!=std::string::npos);

    const std::string control_byte=
        "before {\"name\":\"first\",\"arguments\":{\"path\":\"a\nb\"}} after";
    auto sanitized=q27::resolve_ordered_tool_segments(
        {{q27::StreamSplitter::TEXT,control_byte}},&tools,true,
        [](const std::string&,size_t){ return true; });
    CHECK(sanitized.calls.size()==1 && sanitized.calls[0].name=="first");
    CHECK(sanitized.calls[0].arguments["path"]=="a\nb");
    CHECK(sanitized.text=="before  after");

    const std::string repaired_quote=
        "before {\"name\":\"first\",arguments\":{\"path\":\"a\"}} after";
    auto quote_call=q27::resolve_ordered_tool_segments(
        {{q27::StreamSplitter::TEXT,repaired_quote}},&tools,true,
        [](const std::string&,size_t){ return true; });
    CHECK(quote_call.calls.size()==1 && quote_call.calls[0].name=="first");
    CHECK(quote_call.calls[0].arguments["path"]=="a");
    CHECK(quote_call.text=="before  after");

    const std::string repeated_anchor(40,'x');
    const std::string ambiguous=repeated_anchor+
        "{\"name\":\"first\",arguments\":{\"path\":\"a\"}}"+repeated_anchor;
    auto ambiguous_call=q27::resolve_ordered_tool_segments(
        {{q27::StreamSplitter::TEXT,ambiguous}},&tools,true,
        [](const std::string&,size_t){ return true; });
    CHECK(ambiguous_call.calls.size()==1 &&
          ambiguous_call.calls[0].name=="first");
    CHECK(ambiguous_call.calls[0].raw==
          "{\"name\":\"first\",arguments\":{\"path\":\"a\"}}");
    CHECK(ambiguous_call.text==repeated_anchor+repeated_anchor);
    CHECK(one.text=="before  middle {\"name\":\"second\",\"arguments\":{}} after");
    const std::string truncated=
        "{\"name\":\"first\",\"arguments\":{\"path\":\"a\"";
    auto repaired_truncated=q27::resolve_ordered_tool_segments(
        {{q27::StreamSplitter::TEXT,truncated}},&tools,true,
        [](const std::string&,size_t){ return true; });
    CHECK(repaired_truncated.calls.size()==1 &&
          repaired_truncated.calls[0].name=="first" &&
          repaired_truncated.calls[0].arguments["path"]=="a");
    CHECK(repaired_truncated.text.empty());
    const std::string ordinary_json=
        "{\"name\":\"report\",\"arguments\":{}}";
    auto no_tools=q27::resolve_ordered_tool_segments(
        {{q27::StreamSplitter::TEXT,ordinary_json}},nullptr,true,
        [](const std::string&,size_t){ return true; });
    CHECK(no_tools.calls.empty());
    CHECK(no_tools.text==ordinary_json);
    std::vector<std::pair<q27::StreamSplitter::Chan,std::string>> unclosed={
        {q27::StreamSplitter::TOOL,ordinary_json}};
    std::string unclosed_raw=q27::take_unclosed_final_tool_segment(unclosed,true);
    auto truncated_wrapper=q27::resolve_ordered_tool_segments(
        unclosed,&tools,false,
        [](const std::string&,size_t){ return true; });
    truncated_wrapper.text+=unclosed_raw;
    CHECK(truncated_wrapper.calls.empty());
    CHECK(truncated_wrapper.text==ordinary_json);
    std::vector<std::pair<q27::StreamSplitter::Chan,std::string>> repaired_then_wrapped={
        {q27::StreamSplitter::TEXT,"first\", \"path\":\"a\"}"},
        {q27::StreamSplitter::TOOL,
         "{\"name\":\"second\",\"arguments\":{}}"}};
    auto repaired_first=q27::resolve_ordered_tool_segments(
        repaired_then_wrapped,&tools,true,
        [](const std::string&,size_t accepted){ return accepted==0; });
    CHECK(repaired_first.calls.size()==1 && repaired_first.calls[0].name=="first");
    CHECK(repaired_first.text.find("second")!=std::string::npos);
    const std::string displayed_wrapper=
        "```json\n<tool_call>{\"name\":\"first\",\"arguments\":{}}</tool_call>\n```";
    q27::StreamSplitter displayed_splitter;
    auto displayed_segments=displayed_splitter.feed(displayed_wrapper);
    auto displayed_tail=displayed_splitter.flush();
    displayed_segments.insert(displayed_segments.end(),
                              displayed_tail.begin(),displayed_tail.end());
    auto displayed=q27::resolve_ordered_tool_segments(
        displayed_segments,&tools,true,
        [](const std::string&,size_t){ return true; });
    CHECK(displayed.calls.empty());
    CHECK(displayed.text==displayed_wrapper);
}

// ---- Issue #38, reopened: the streaming router ------------------------------
// Drives q27::StreamToolRouter -- the one copy of the per-chunk routing both
// SSE handlers now call -- through the real StreamSplitter, with the exact
// call shape from the report inside a prompt-injected <think> block, in
// chunks small enough that the <tool_call> wrapper straddles them. The
// collectors stand in for the endpoint-specific emit_* lambdas; the classify
// callback is the handlers' classify_bare verbatim.
struct StreamRun {
    std::string text, think;
    std::vector<q27::ToolCall> calls;
};

static json stream_tools() {
    return json::parse(R"([
      {"type":"function","function":{"name":"read","parameters":{"type":"object",
        "properties":{"path":{"type":"string"},"limit":{"type":"integer"}},"required":["path"]}}},
      {"type":"function","function":{"name":"bash","parameters":{"type":"object",
        "properties":{"command":{"type":"string"}},"required":["command"]}}}])");
}

static StreamRun stream_turn(const std::string& generated, bool start_in_think,
                             size_t chunk, bool allow_repair_at_end = true,
                             bool with_tools = true) {
    const json tools = with_tools ? stream_tools() : json::array();
    std::set<std::string> names;
    for (const auto& t : tools) names.insert(t["function"]["name"].get<std::string>());
    q27::StreamSplitter sp;
    if (start_in_think) sp.chan = q27::StreamSplitter::THINK;
    q27::StreamToolRouter r;
    r.has_tools = with_tools;
    StreamRun out;
    auto emit_text = [&](const std::string& t) { out.text += t; };
    auto emit_think = [&](const std::string& t) { out.think += t; };
    auto emit_call = [&](const q27::ToolCall& c) { out.calls.push_back(c); return true; };
    auto classify = [&](const std::string& source, bool allow_repair, auto&& visible) {
        std::string pre, residual;
        auto bcs = q27::parse_bare_tool_calls(source, &pre, &tools, true, allow_repair, &residual);
        if (bcs.empty()) return q27::BareToolCandidateResult{};
        size_t cursor = 0, recovered = 0;
        for (const auto& bc : bcs) {
            visible(source.substr(cursor, bc.source_begin - cursor));
            cursor = bc.source_end;
            if (emit_call(bc)) recovered++;
            else visible(source.substr(bc.source_begin, bc.source_end - bc.source_begin));
        }
        visible(source.substr(cursor));
        return q27::BareToolCandidateResult{true, recovered != 0};
    };
    auto emit_tool = [&]() {
        auto c = q27::parse_tool_call(q27::strip_ws2(r.tool_buf));
        r.tool_buf.clear();
        if (!c.ok) {
            if (!classify(c.raw, true, emit_text) && !q27::only_dialect_control_bytes(c.raw))
                emit_text(c.raw);
        } else emit_call(c);
    };
    auto seg = [&](q27::StreamSplitter::Chan ch, const std::string& t) {
        r.segment(ch, t, false, names, emit_text, emit_think, emit_tool, classify);
    };
    for (size_t i = 0; i < generated.size(); i += chunk)
        for (auto& [ch, t] : sp.feed(generated.substr(i, chunk))) seg(ch, t);
    for (auto& [ch, t] : sp.flush()) seg(ch, t);
    if (!r.tool_buf.empty()) emit_tool();
    r.finish(allow_repair_at_end, names, emit_text, emit_think, classify);
    return out;
}

// the report's bytes: wrapped call in reasoning, doubled </tool_call>
static const char* kIssue38Think =
    "There's no agents/ directory. Next, read the temp file.\n"
    "<tool_call>\n<function=read>\n<parameter=path>\n/tmp/code-reviewer-diff.md\n"
    "</parameter>\n<parameter=limit>\n160\n</parameter>\n</function>\n</tool_call>\n</tool_call>\n";

static void test_stream_router_wrapped_call_in_reasoning() {
    for (size_t chunk : {size_t(1), size_t(3), size_t(7), size_t(64)}) {
        auto r = stream_turn(std::string(kIssue38Think) + "</think>\n\nReading it now.", true, chunk);
        CHECK(r.calls.size() == 1);
        if (r.calls.size() == 1) {
            CHECK(r.calls[0].name == "read");
            CHECK(r.calls[0].arguments.value("path", std::string()) == "/tmp/code-reviewer-diff.md");
        }
        CHECK(r.think.find("<function=") == std::string::npos);   // the call is not reasoning
        CHECK(r.think.find("<parameter=") == std::string::npos);
        CHECK(r.think.find("Next, read the temp file.") != std::string::npos);  // the prose is
        CHECK(q27::strip_ws2(r.text) == "Reading it now.");
    }
}

static void test_stream_router_bare_call_in_reasoning() {
    const std::string gen =
        "plan\n<function=bash>\n<parameter=command>\nls /tmp\n</parameter>\n</function>\n</think>\n\ndone";
    auto r = stream_turn(gen, true, 7);
    CHECK(r.calls.size() == 1);
    if (!r.calls.empty()) CHECK(r.calls[0].name == "bash");
    CHECK(r.think.find("<function=") == std::string::npos);
}

static void test_stream_router_four_calls_in_one_think_block() {
    // the reporter's host-only repro: read + bash, four calls, one block
    std::string gen = "gather everything first\n";
    for (int i = 0; i < 2; i++) {
        gen += "<tool_call>\n<function=read>\n<parameter=path>\n/w/f" + std::to_string(i) +
               ".ts\n</parameter>\n</function>\n</tool_call>\n";
        gen += "<tool_call>\n<function=bash>\n<parameter=command>\ncat /w/f" + std::to_string(i) +
               ".ts\n</parameter>\n</function>\n</tool_call>\n";
    }
    gen += "</think>\n\nok";
    auto r = stream_turn(gen, true, 7);
    CHECK(r.calls.size() == 4);
    if (r.calls.size() == 4) {
        CHECK(r.calls[0].name == "read");
        CHECK(r.calls[1].name == "bash");
        CHECK(r.calls[3].arguments.value("command", std::string()) == "cat /w/f1.ts");
    }
    CHECK(r.think.find("<function=") == std::string::npos);
}

static void test_stream_router_fenced_example_in_reasoning_stays_reasoning() {
    // the model writing ABOUT a call: a fenced example and an inline-code
    // mention must not fire -- the safety rule the non-stream fix recorded
    const std::string fenced =
        "The dialect looks like this:\n```\n<tool_call>\n<function=read>\n<parameter=path>\n/x\n"
        "</parameter>\n</function>\n</tool_call>\n```\nbut I will not call it.\n</think>\n\nno call";
    auto r = stream_turn(fenced, true, 7);
    CHECK(r.calls.empty());
    CHECK(r.think.find("<function=read>") != std::string::npos);   // kept as reasoning
    const std::string inline_mention =
        "I could use `<function=read>` here but the file is known.\n</think>\n\nanswer";
    auto r2 = stream_turn(inline_mention, true, 7);
    CHECK(r2.calls.empty());
}

static void test_stream_router_turn_ends_inside_think() {
    // no </think> ever arrives: the held call must still fire at finish
    auto r = stream_turn(kIssue38Think, true, 7, /*allow_repair_at_end=*/true);
    CHECK(r.calls.size() == 1);
    if (!r.calls.empty()) CHECK(r.calls[0].name == "read");
}

static void test_stream_router_reasoning_call_then_text_call_in_order() {
    const std::string gen = std::string(kIssue38Think) +
        "</think>\n\nNow run it.\n<tool_call>\n<function=bash>\n<parameter=command>\nmake\n"
        "</parameter>\n</function>\n</tool_call>";
    auto r = stream_turn(gen, true, 7);
    CHECK(r.calls.size() == 2);
    if (r.calls.size() == 2) {
        CHECK(r.calls[0].name == "read");
        CHECK(r.calls[1].name == "bash");
    }
    CHECK(q27::strip_ws2(r.text) == "Now run it.");
}

static void test_stream_router_without_tools_passes_reasoning_through() {
    auto r = stream_turn(std::string(kIssue38Think) + "</think>\n\nhi", true, 7, true, /*with_tools=*/false);
    CHECK(r.calls.empty());
    CHECK(r.think == std::string(kIssue38Think));   // byte-for-byte, nothing blanked
    CHECK(q27::strip_ws2(r.text) == "hi");
}

// Issue #38, third round (cosmicnag, 2026-08-25): calls fire, but
// "</function>\n<tool_call>" reached the visible content ahead of a call
// that then worked. Dialect residue next to a call -- a stray closer before
// the wrapper, a doubled <tool_call> opener whose inner copy was the "text
// before the call", the closer/opener between two calls the model ran
// together -- is not text.
static void test_dialect_residue_is_not_text() {
    const json tools = stream_tools();
    std::string pre, remaining;
    // the doubled-opener body a wrapped segment hands the bare chain
    auto one = q27::parse_bare_tool_calls(
        "\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n",
        &pre, &tools, true, true, &remaining);
    CHECK(one.size() == 1 && one[0].name == "bash");
    CHECK(q27::strip_ws2(remaining).empty());
    CHECK(one.size() == 1 && one[0].source_begin == 0);
    // two calls run together: </function>\n<tool_call>\n between them
    auto two = q27::parse_bare_tool_calls(
        "<function=read>\n<parameter=path>\n/a\n</parameter>\n</function>\n<tool_call>\n"
        "<function=read>\n<parameter=path>\n/b\n</parameter>\n</function>",
        &pre, &tools, true, true, &remaining);
    CHECK(two.size() == 2);
    CHECK(remaining.find("<tool_call>") == std::string::npos);
    CHECK(remaining.find("</function>") == std::string::npos);
    // a truncated closer after the last call
    auto trunc = q27::parse_bare_tool_calls(
        "<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</functio",
        &pre, &tools, true, true, &remaining);
    CHECK(trunc.size() == 1);
    CHECK(q27::strip_ws2(remaining).empty());
    // the streaming shape from the report: prose, stray closer, doubled opener
    const std::string reported =
        "I found that compat.js has its own registry.\n</function>\n<tool_call>\n<tool_call>\n"
        "<function=bash>\n<parameter=command>\nsed -n '100,215p' compat.js\n</parameter>\n"
        "</function>\n</tool_call>";
    for (size_t chunk : {size_t(1), size_t(5), size_t(64)}) {
        auto r = stream_turn(reported, false, chunk);
        CHECK(r.calls.size() == 1);
        if (!r.calls.empty()) CHECK(r.calls[0].name == "bash");
        CHECK(q27::strip_ws2(r.text) == "I found that compat.js has its own registry.");
        CHECK(r.text.find("</function>") == std::string::npos);
        CHECK(r.text.find("<tool_call>") == std::string::npos);
    }
    // a stray closer, then a BARE call in text
    auto bare = stream_turn("note\n</function>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>", false, 7);
    CHECK(bare.calls.size() == 1);
    CHECK(q27::strip_ws2(bare.text) == "note");
    // residue at the very end of a turn with no call is dropped...
    auto eos = stream_turn("done.\n</function>", false, 7);
    CHECK(eos.calls.empty());
    CHECK(q27::strip_ws2(eos.text) == "done.");
    // ...but a closer followed by more prose is the model's text, shown as before
    auto prose = stream_turn("a\n</function>\nb", false, 3);
    CHECK(prose.text == "a\n</function>\nb");
    // a wrapper INSIDE a JSON string value is content the splitter keeps in
    // TEXT and the chain refuses (no call); every byte must come out, in
    // order, whatever the chunking -- the residue hold only reorders when it
    // drops, and it must not drop here
    const std::string nested =
        "{\"name\":\"first\",\"arguments\":{\"x\":<tool_call>\n{\"name\":\"second\",\"arguments\":{}}\n</tool_call>\n          }}";
    for (size_t chunk : {size_t(1), size_t(7), size_t(41), size_t(200)}) {
        auto n = stream_turn(nested, false, chunk);
        CHECK(n.calls.empty());
        CHECK(n.text == nested);
    }
    // the non-stream resolver, same shape
    auto res = q27::resolve_ordered_tool_segments(
        {{q27::StreamSplitter::TEXT, "I found it.\n</function>\n"},
         {q27::StreamSplitter::TOOL, "\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n"}},
        &tools, true, [](const std::string&, size_t) { return true; });
    CHECK(res.calls.size() == 1);
    CHECK(q27::strip_ws2(res.text) == "I found it.");
    CHECK(res.text.find("<tool_call>") == std::string::npos);
}

// The `{"tool_call":`-wrapped JSON batch (drift corpus ea9ead21): the model
// opens the batch with the mode-4 head and never closes its brace, so the
// whole thing was one malformed blob and only the last call survived. The
// xmlclose retry now blanks the head, and both calls recover with nothing
// left as text.
static void test_tool_call_wrapped_json_batch() {
    const json tools = stream_tools();
    std::string pre, remaining;
    const std::string batch =
        "{\"tool_call\":\n{\"name\": \"read\", \"arguments\": {\"path\":\"/a\"}}\n</tool_call>\n"
        "{\"name\": \"read\", \"arguments\": {\"path\":\"/b\"}}\n</tool_call>";
    auto v = q27::parse_bare_tool_calls(batch, &pre, &tools, true, true, &remaining);
    CHECK(v.size() == 2);
    if (v.size() == 2) {
        CHECK(v[0].arguments.value("path", std::string()) == "/a");
        CHECK(v[1].arguments.value("path", std::string()) == "/b");
    }
    CHECK(q27::strip_ws2(remaining).empty());   // the head is residue, not text
    // a `{"tool_call":` whose value is a STRING is not this shape and must not
    // be mangled into a call
    auto none = q27::parse_bare_tool_calls("{\"tool_call\": \"see the docs\"}", &pre, &tools, true, true, &remaining);
    CHECK(none.empty());
}

// The streaming holdback arms on every bare opener spelling the parser
// accepts, not just `<function=` (pylint-6903 emitted `<parameter_name>`;
// mode 18 is `<name>`). Both were parser-recognised for weeks while the
// stream knew only `<function=`, so they worked off stream only.
// Streaming recovery of a bare mode-22 batch whose FIRST call takes no
// arguments (issue #38, cosmicnag). The model drops the <function=NAME> head
// and writes <parameter=NAME> instead; a zero-argument call is
// `<parameter=NAME>` immediately closed by `</function>`. This shape reaches
// the TEXT/THINK holdback (not the TOOL channel -- there is no <tool_call>
// wrapper), where the ambiguous `<parameter=` opener has to be held across
// chunk boundaries until it resolves. Uses its own tool set: memex_recall
// takes no arguments, read requires a path.
static json mode22_zero_arg_tools() {
    return json::parse(R"([
      {"type":"function","function":{"name":"memex_recall","parameters":{
        "type":"object","properties":{}}}},
      {"type":"function","function":{"name":"read","parameters":{"type":"object",
        "properties":{"path":{"type":"string"}},"required":["path"]}}}])");
}
static StreamRun stream_turn_tools(const json& tools, const std::string& generated,
                                   bool start_in_think, size_t chunk,
                                   bool allow_repair_at_end = true) {
    std::set<std::string> names;
    for (const auto& t : tools) names.insert(t["function"]["name"].get<std::string>());
    q27::StreamSplitter sp;
    if (start_in_think) sp.chan = q27::StreamSplitter::THINK;
    q27::StreamToolRouter r;
    r.has_tools = true;
    StreamRun out;
    auto emit_text = [&](const std::string& t) { out.text += t; };
    auto emit_think = [&](const std::string& t) { out.think += t; };
    auto emit_call = [&](const q27::ToolCall& c) { out.calls.push_back(c); return true; };
    auto classify = [&](const std::string& source, bool allow_repair, auto&& visible) {
        std::string pre, residual;
        auto bcs = q27::parse_bare_tool_calls(source, &pre, &tools, true, allow_repair, &residual);
        if (bcs.empty()) return q27::BareToolCandidateResult{};
        size_t cursor = 0, recovered = 0;
        for (const auto& bc : bcs) {
            visible(source.substr(cursor, bc.source_begin - cursor));
            cursor = bc.source_end;
            if (emit_call(bc)) recovered++;
            else visible(source.substr(bc.source_begin, bc.source_end - bc.source_begin));
        }
        visible(source.substr(cursor));
        return q27::BareToolCandidateResult{true, recovered != 0};
    };
    auto emit_tool = [&]() {
        auto c = q27::parse_tool_call(q27::strip_ws2(r.tool_buf));
        r.tool_buf.clear();
        if (!c.ok) {
            if (!classify(c.raw, true, emit_text) && !q27::only_dialect_control_bytes(c.raw))
                emit_text(c.raw);
        } else emit_call(c);
    };
    auto seg = [&](q27::StreamSplitter::Chan ch, const std::string& t) {
        r.segment(ch, t, false, names, emit_text, emit_think, emit_tool, classify);
    };
    for (size_t i = 0; i < generated.size(); i += chunk)
        for (auto& [ch, t] : sp.feed(generated.substr(i, chunk))) seg(ch, t);
    for (auto& [ch, t] : sp.flush()) seg(ch, t);
    if (!r.tool_buf.empty()) emit_tool();
    r.finish(allow_repair_at_end, names, emit_text, emit_think, classify);
    return out;
}

// cosmicnag's shape: a zero-arg memex_recall planned in parallel with reads,
// parameter-as-opener spelling, only stray </tool_call> closers.
static const char* kBareMode22Batch =
    "<parameter=memex_recall>\n</function>\n"
    "<parameter=read>\n<parameter=path>\n/a/SKILL.md\n</parameter>\n</function>\n"
    "<parameter=read>\n<parameter=path>\n/b/NOTES.md\n</parameter>\n</function>\n"
    "</tool_call>\n";

static void test_stream_router_bare_mode22_zero_arg_batch() {
    const json tools = mode22_zero_arg_tools();
    for (size_t chunk : {size_t(1), size_t(3), size_t(7), size_t(64)}) {
        auto r = stream_turn_tools(tools, kBareMode22Batch, false, chunk);
        CHECK(r.calls.size() == 3);
        if (r.calls.size() == 3) {
            CHECK(r.calls[0].name == "memex_recall");
            CHECK(r.calls[0].arguments.empty());
            CHECK(r.calls[1].name == "read");
            CHECK(r.calls[1].arguments.value("path", std::string()) == "/a/SKILL.md");
            CHECK(r.calls[2].name == "read");
            CHECK(r.calls[2].arguments.value("path", std::string()) == "/b/NOTES.md");
        }
        CHECK(r.text.find("<parameter=") == std::string::npos);   // nothing leaked as text
    }
}

// Same batch inside a <think> block: it is a call, not reasoning text.
static void test_stream_router_bare_mode22_in_reasoning() {
    const json tools = mode22_zero_arg_tools();
    const std::string gen = std::string("Load memory, then read.\n") + kBareMode22Batch +
                            "</think>\n\nDone.";
    for (size_t chunk : {size_t(1), size_t(7), size_t(64)}) {
        auto r = stream_turn_tools(tools, gen, true, chunk);
        CHECK(r.calls.size() == 3);
        if (!r.calls.empty()) CHECK(r.calls[0].name == "memex_recall");
        CHECK(r.think.find("<parameter=") == std::string::npos);
        CHECK(r.think.find("Load memory, then read.") != std::string::npos);
    }
}

// A lone zero-arg call at the very end of the stream (its </function> is the
// final token, so it lands in the router's trailing residue) still fires.
static void test_stream_router_bare_mode22_standalone_zero_arg() {
    const json tools = mode22_zero_arg_tools();
    for (size_t chunk : {size_t(1), size_t(3), size_t(7), size_t(64)}) {
        auto r = stream_turn_tools(tools, "<parameter=memex_recall>\n</function>\n", false, chunk);
        CHECK(r.calls.size() == 1);
        if (!r.calls.empty()) CHECK(r.calls[0].name == "memex_recall");
        CHECK(r.text.find("<parameter=") == std::string::npos);
    }
}

// The ambiguous `<parameter=` opener must never fabricate a call or swallow
// text when it is NOT a mode-22 opener: an undeclared name, a declared name
// followed by a real value (an ordinary parameter), a prose mention, a fenced
// example, or a partial tag at end of stream. Each must yield zero calls and
// surface its text verbatim.
// issue #38 round 5 (2026-08-29, cosmicnag): a nameless openerless call in
// the HTML-attribute spelling, streamed bare -- prose, a mangled opener
// remnant, `<parameter name="command">` (quoted attr) mixed with
// `<parameter=description>`, closed by a bare `</function>`. Non-stream
// recovered it via mode 21 (signature-inferred Bash); the streaming holdback
// never armed and the whole call leaked as text, halting the agent.
static json mode21_stream_tools() {
    return json::parse(R"([
      {"type":"function","function":{"name":"Bash","parameters":{"type":"object",
        "properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command"]}}},
      {"type":"function","function":{"name":"read","parameters":{"type":"object",
        "properties":{"path":{"type":"string"}},"required":["path"]}}}])");
}
// issue #38 round 6 (2026-09-01, cosmicnag): the ARGUMENTS object emitted
// bare -- {"command": "..."} with no name, no wrapper, no opener -- closed by
// XML dialect closers, with mode-11-class escaping damage inside the value
// (mixed \\" and raw ", literal newlines). The XML closers are the intent
// evidence; the repair is the mode-11 terminator scan tried per declared
// tool, firing only on a unique fit (mode 23).
static void test_stream_router_args_only_object_with_closers() {
    const json tools = mode21_stream_tools();
    // representative reduction of the report's bytes: escaped \\n, then a RAW
    // unescaped quote + literal newline mid-value, then escaped \\" later,
    // terminated "} + closers.
    const std::string gen =
        "This is the deciding evidence.\n Let me look.\n\n "
        "{\"command\":\"D=/tmp/x\\necho \"===\n inner raw quote region\\ngrep -i \\\"pat\\\" f | head -c 600\"}\n\n"
        " </parameter>\n </function>";
    for (size_t chunk : {size_t(1), size_t(7), size_t(64)}) {
        auto r = stream_turn_tools(tools, gen, false, chunk);
        CHECK(r.calls.size() == 1);
        if (!r.calls.empty()) {
            CHECK(r.calls[0].name == "Bash");
            const std::string cmd = r.calls[0].arguments.value("command", std::string());
            CHECK(cmd.find("echo \"===") != std::string::npos);      // raw quote survived
            CHECK(cmd.find("head -c 600") != std::string::npos);      // tail survived
        }
        CHECK(r.text.find("{\"command\"") == std::string::npos);      // no leak
        CHECK(r.text.find("deciding evidence.") != std::string::npos); // prose kept
    }
}

// The evidence gate both ways: a bare JSON object in prose WITHOUT closers is
// an ordinary object and must come back out as text; with closers but an
// AMBIGUOUS registry fit, refuse rather than guess.
static void test_stream_router_bare_object_without_closers_stays_text() {
    const json tools = mode21_stream_tools();
    const std::string gen = "the config is {\"command\":\"ls -la\"} as requested, done.";
    for (size_t chunk : {size_t(1), size_t(7), size_t(64)}) {
        auto r = stream_turn_tools(tools, gen, false, chunk);
        CHECK(r.calls.empty());
        CHECK(r.text.find("{\"command\":\"ls -la\"}") != std::string::npos);
        CHECK(r.text.find("done.") != std::string::npos);
    }
}

static void test_stream_router_openerless_html_attr_params() {
    const json tools = mode21_stream_tools();
    const std::string gen =
        "Let me redo both checks.\n\n>\n"
        "<parameter name=\"command\">hf models list --format json\n</parameter>\n"
        "<parameter=description>\nSum branch file sizes\n</parameter>\n</function>\n";
    for (size_t chunk : {size_t(1), size_t(3), size_t(7), size_t(64)}) {
        auto r = stream_turn_tools(tools, gen, false, chunk);
        CHECK(r.calls.size() == 1);
        if (!r.calls.empty()) {
            CHECK(r.calls[0].name == "Bash");
            CHECK(r.calls[0].arguments.value("command", std::string()).find("hf models list") == 0);
            CHECK(r.calls[0].arguments.value("description", std::string()) == "Sum branch file sizes");
        }
        CHECK(r.text.find("<parameter") == std::string::npos);   // nothing leaked
        CHECK(r.text.find("Let me redo both checks.") != std::string::npos);  // prose kept
    }
}

// The arming evidence gate the other way: an undeclared parameter pair QUOTED
// in prose (tag + closer present, but the keys infer no declared tool) must
// come back out as text, not a call and not a swallow.
static void test_stream_router_quoted_parameter_pair_stays_text() {
    const json tools = mode21_stream_tools();
    const std::string gen =
        "The syntax is <parameter=frobnicate>value</parameter> in that dialect.\nDone.";
    for (size_t chunk : {size_t(1), size_t(7), size_t(64)}) {
        auto r = stream_turn_tools(tools, gen, false, chunk);
        CHECK(r.calls.empty());
        CHECK(r.text.find("<parameter=frobnicate>value</parameter>") != std::string::npos);
        CHECK(r.text.find("Done.") != std::string::npos);
    }
}

static void test_stream_router_parameter_tag_that_is_not_a_call() {
    const json tools = mode22_zero_arg_tools();
    struct Case { const char* gen; const char* must_contain; } cases[] = {
        {"here is <parameter=foo>bar</parameter> in prose", "<parameter=foo>"},
        {"text <parameter=read>hello world</parameter> more", "<parameter=read>"},
        {"the <parameter=x> placeholder syntax is common", "<parameter=x>"},
        {"```xml\n<parameter=read>\n</function>\n```\ndone", "<parameter=read>"},
        {"trailing text <parameter=re", "trailing text"},
    };
    for (const auto& c : cases)
        for (size_t chunk : {size_t(1), size_t(7), size_t(64)}) {
            auto r = stream_turn_tools(tools, c.gen, false, chunk);
            CHECK(r.calls.empty());
            CHECK(r.text.find(c.must_contain) != std::string::npos);
        }
}

static void test_stream_router_arms_on_every_opener_spelling() {
    for (const std::string& opener : {std::string("<parameter_name>read"), std::string("<name>read")}) {
        const std::string shape = opener + "\n</parameter>\n<parameter=path>\n/w/x.py\n";
        for (size_t chunk : {size_t(1), size_t(7), size_t(64)}) {
            auto r = stream_turn(shape, false, chunk);
            CHECK(r.calls.size() == 1);
            if (!r.calls.empty()) CHECK(r.calls[0].name == "read");
            CHECK(q27::strip_ws2(r.text).empty());
        }
    }
    // The pylint-6903 shape opens with `<tool_use>`, which is a markdown HTML
    // block, so the display-context rule (a fenced or HTML example must not
    // fire) correctly refuses to arm mid-block: it streams as text. That is
    // not a silent drop -- the router records it to the corpus at finish
    // (test_drift_hook), and the OFFLINE chain, which has no display gate on
    // the native scan, recovers it (test_tool_drift). Pinned here so a future
    // change to displayed_html is a deliberate one.
    const std::string html_block =
        "\n\n<tool_use>\n<tool>\n<parameter_name>\n<parameter_name>read\n</parameter>\n<parameter=path>\n/w/x.py\n";
    auto h = stream_turn(html_block, false, 7);
    CHECK(h.calls.empty());
    // the safety cases: a fenced example and <name> as prose stay text
    const std::string fenced =
        "example:\n```\n<name>read\n</parameter>\n<parameter=path>\n/w\n</parameter>\n</function>\n```\ndone";
    auto f = stream_turn(fenced, false, 7);
    CHECK(f.calls.empty());
    CHECK(f.text == fenced);
    const std::string prose = "Use the <name> tag for the display name.";
    auto pr = stream_turn(prose, false, 7);
    CHECK(pr.calls.empty());
    CHECK(pr.text == prose);
}

// The whole-block helper both /v1/responses legs and the non-stream resolver
// call: a finished reasoning block with a call in it.
static void test_recover_calls_from_reasoning() {
    const json tools = stream_tools();
    std::vector<q27::ToolCall> got;
    auto take = [&](q27::ToolCall& c) { got.push_back(std::move(c)); return true; };
    size_t taken = 0;
    // wrapped, doubled closer: the call is taken, the prose stays, the wrapper does not
    std::string kept = q27::recover_calls_from_reasoning(kIssue38Think, &tools, taken, take);
    CHECK(taken == 1);
    CHECK(got.size() == 1 && got[0].name == "read");
    CHECK(kept.find("Next, read the temp file.") != std::string::npos);
    CHECK(kept.find("<function=") == std::string::npos);
    CHECK(kept.find("<tool_call>") == std::string::npos);
    // a fenced example is the model writing about a call: untouched, nothing taken
    got.clear();
    const std::string fenced = "like so:\n```\n<tool_call>\n<function=read>\n<parameter=path>\n/x\n</parameter>\n</function>\n</tool_call>\n```\nnot calling it";
    CHECK(q27::recover_calls_from_reasoning(fenced, &tools, taken, take) == fenced);
    CHECK(taken == 0 && got.empty());
    // refused by the acceptor (tool_choice says no): bytes stay reasoning, untouched
    auto refuse = [&](q27::ToolCall&) { return false; };
    CHECK(q27::recover_calls_from_reasoning(kIssue38Think, &tools, taken, refuse) == std::string(kIssue38Think));
    CHECK(taken == 0);
    // no tools declared: untouched
    CHECK(q27::recover_calls_from_reasoning(kIssue38Think, nullptr, taken, take) == std::string(kIssue38Think));
    CHECK(taken == 0);
}

static void test_think_wrapper_blanker_straddles_chunks() {
    q27::ThinkWrapperBlanker b;
    CHECK(b.feed("ab<tool_c") == "ab");                 // partial token held
    CHECK(b.feed("all>cd") == std::string(11, ' ') + "cd"); // completed: blanked, length kept
    CHECK(b.feed("<") == "");
    CHECK(b.feed("/tool_call>x") == std::string(12, ' ') + "x");
    CHECK(b.feed("<tool_x") == "<tool_x");               // not a prefix of either token
    CHECK(b.feed("y</tool_") == "y");
    CHECK(b.flush() == "</tool_");                        // the tail is content after all
    CHECK(b.feed("<tool_call></tool_call>") == std::string(23, ' '));
}

// issue #39: output-cap alias tolerance. The official field per endpoint
// wins over foreign/legacy aliases; absent/null reads as absent.
static void test_request_max_tokens_aliases() {
    using q27::request_max_tokens; using q27::CapApi;
    json b1 = {{"max_completion_tokens", 65536}};
    CHECK(request_max_tokens(b1, 8192, CapApi::Chat) == 65536);      // the report's repro
    CHECK(request_max_tokens(b1, 8192, CapApi::Messages) == 65536);  // alias accepted
    json b2 = {{"max_tokens", 1000}, {"max_completion_tokens", 2000}};
    CHECK(request_max_tokens(b2, 8192, CapApi::Chat) == 2000);       // official wins on chat
    CHECK(request_max_tokens(b2, 8192, CapApi::Messages) == 1000);   // official wins on messages
    json b3 = {{"max_tokens", 1000}, {"max_output_tokens", 3000}};
    CHECK(request_max_tokens(b3, 8192, CapApi::Responses) == 3000);  // official wins on responses
    CHECK(request_max_tokens(json::object(), 8192, CapApi::Chat) == 8192); // default
    json b4 = {{"max_completion_tokens", nullptr}};
    CHECK(request_max_tokens(b4, 8192, CapApi::Chat) == 8192);       // null = absent
}

int main() {
    test_request_max_tokens_aliases();
    test_stream_router_wrapped_call_in_reasoning();
    test_stream_router_bare_call_in_reasoning();
    test_stream_router_four_calls_in_one_think_block();
    test_stream_router_fenced_example_in_reasoning_stays_reasoning();
    test_stream_router_turn_ends_inside_think();
    test_stream_router_reasoning_call_then_text_call_in_order();
    test_stream_router_without_tools_passes_reasoning_through();
    test_think_wrapper_blanker_straddles_chunks();
    test_recover_calls_from_reasoning();
    test_tool_call_wrapped_json_batch();
    test_stream_router_arms_on_every_opener_spelling();
    test_stream_router_bare_mode22_zero_arg_batch();
    test_stream_router_bare_mode22_in_reasoning();
    test_stream_router_bare_mode22_standalone_zero_arg();
    test_stream_router_parameter_tag_that_is_not_a_call();
    test_stream_router_args_only_object_with_closers();
    test_stream_router_bare_object_without_closers_stays_text();
    test_stream_router_openerless_html_attr_params();
    test_stream_router_quoted_parameter_pair_stays_text();
    test_dialect_residue_is_not_text();
    test_tools_passthrough();
    test_tools_absent();
    test_responses_tool_fields_reject_wrong_types();
    test_msgs_plain_roundtrip();
    test_msgs_content_parts_array();
    test_msgs_developer_role_maps_to_system();
    test_msgs_assistant_tool_calls_reconstructed();
    test_msgs_assistant_content_plus_tool_calls();
    test_msgs_malformed_arguments_string_kept_not_dropped();
    test_msgs_no_messages_key();
    test_msgs_content_less_message_no_crash();
    test_billing_header_2_1_220_shape();
    test_billing_header_legacy_cch_still_pinned();
    test_billing_header_leaves_other_prompts_alone();
    test_jread_null_reads_as_absent();
    test_jread_wrong_type_reads_as_absent();
    test_jread_present_values_win();
    test_jint_float_and_absurd_magnitude();
    test_parse_tool_call_requires_object_arguments();
    test_anthropic_msgs_non_object_message_skipped();
    test_anthropic_msgs_bare_string_content_part_skipped();
    test_anthropic_msgs_system_array_of_strings();
    test_anthropic_msgs_tool_result_bare_string_content();
    test_anthropic_tools_non_array_and_non_object_entries();
    test_tool_choice_absent_is_auto();
    test_tool_choice_none();
    test_tool_choice_required();
    test_forced_tool_choice_truncation();
    test_anthropic_tool_stop_reason();
    test_tool_choice_named_function();
    test_tool_choice_unknown_string_is_auto();
    test_tool_choice_malformed_named_is_invalid();
    test_tool_choice_allowed_tools();
    test_openai_parallel_tool_calls();
    test_recovered_call_batch_eligibility();
    test_anthropic_tool_choice_shapes();
    test_responses_tool_choice_shapes();
    test_stream_options_include_usage();
    test_end_to_end_chatml_prompt();
    test_openai_reasoning_content_roundtrip();
    test_think_toggle_state();
    test_tool_failure_warnings();
    test_tool_grouping();
    test_tool_truncation();
    test_tool_truncation_request_body();
    test_v223_multi_system_merge();
    test_v223_new_think_markers();
    test_v223_reasoning_dedup();
    test_v223_failure_detection();
    test_unconditional_think_wrap();
    test_request_effort_and_auto_disable();
    test_output_config_fields();
    test_none_off_disables_thinking();
    test_xml_tool_call_rendering();
    test_truncation_json_dialect_skipped();
    test_chat_message_plain_text_no_calls();
    test_chat_message_empty_text_no_calls_is_empty_string_not_null();
    test_chat_message_call_no_leftover_text_content_null();
    test_chat_message_call_plus_leftover_text();
    test_chat_message_parallel_calls_indexed_and_ordered();
    test_chat_message_reasoning_content_included_when_present();
    test_chat_message_reasoning_content_absent_when_empty();
    test_chat_message_reasoning_content_with_tool_call();
    test_reasoning_delta_shape();
    test_stream_chunk_shape();
    test_stream_chunk_finish_reason();
    test_tool_call_delta_shape();
    test_tool_call_invalid_utf8_replaced();
    test_tool_streamer_bytewise_production_shape();
    test_tool_streamer_inline_repairs();
    test_tool_streamer_fallback_is_byte_exact();
    test_tool_streamer_preserves_packed_call_trail();
    test_tool_streamer_quoted_content_modes();
    test_tool_streamer_escapes_all_control_bytes();
    test_tool_streamer_can_refuse_tail_repair();
    test_bare_tool_parser_can_refuse_eof_repair();
    test_bare_tool_stream_holdback_bounds();
    test_bare_tool_stream_holdback_native_xml();
    test_tool_call_inside_reasoning();
    test_ordered_tool_segments_preserve_first_call_and_rejected_text();
    test_tool_streamer_marks_invalid_completed_args();
    if (failures) { fprintf(stderr, "%d FAILURE(S)\n", failures); return 1; }
    fprintf(stderr, "all tests passed\n");
    return 0;
}
