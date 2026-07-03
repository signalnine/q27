// q27 HTTP server. Single slot (MTP spec decode is 1-stream), greedy only.
// Endpoints:
//   GET  /health, /v1/models
//   POST /v1/chat/completions, /v1/completions        (OpenAI)
//   POST /v1/messages                                 (Anthropic, Claude Code-grade:
//        thinking blocks, tool_use/tool_result, input_json_delta streaming)
//   POST /v1/responses                                (OpenAI Responses, Codex CLI)
//
// usage: q27-server model.q27 model.tok [--port 8080] [--host 0.0.0.0]
//                   [--ctx 8192] [--fast-head]
#include <atomic>
#include <functional>
#include <memory>
#include <set>
#include <tuple>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>

#include "engine.cuh"
#include "tokenizer.h"
#include "api_common.h"
#include "../third_party/httplib.h"
#include "../third_party/json.hpp"

using json = nlohmann::json;
using q27::Msg;
using q27::StreamSplitter;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.q27 model.tok [--port N] [--host H] [--ctx C] "
                        "[--fast-head]\n", argv[0]);
        return 1;
    }
    std::string model = argv[1], tokpath = argv[2], host = "0.0.0.0";
    int port = 8080, ctx = 8192;
    bool fast = false;
    bool no_think_srv = false;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fast-head")) fast = true;
        else if (!strcmp(argv[i], "--no-think")) no_think_srv = true;
    }
    if (no_think_srv) fprintf(stderr, "no-think: empty-think prefill on all chat paths\n");

    fprintf(stderr, "loading tokenizer...\n");
    q27::Tokenizer tok(tokpath);
    fprintf(stderr, "loading model...\n");
    Engine eng(model, ctx);
    eng.fast_head = fast;
    eng.build_graph();
    eng.build_spec_graphs();
    const int EOS = tok.eos();

    std::mutex gpu; // single slot: serialize requests
    std::atomic<long> req_counter{0};

    httplib::Server srv;
    srv.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        fprintf(stderr, "[http] %s %s -> %d\n", req.method.c_str(), req.path.c_str(),
                res.status);
    });

    srv.Get("/health", [&](const httplib::Request& req, httplib::Response& res) {
        // /health?verify=1 recomputes the resident-weight checksums (~20 ms;
        // safe concurrently with generation -- read-only, separate stream).
        if (req.has_param("verify")) {
            int bad = eng.dm.checksum_verify(true);
            res.set_content(std::string("{\"status\":\"") + (bad ? "corrupted" : "ok") +
                                "\",\"weight_mismatches\":" + std::to_string(bad) + "}",
                            "application/json");
            return;
        }
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json j = {{"object", "list"},
                  {"data", json::array({{{"id", "q27-qwopus-27b"}, {"object", "model"},
                                         {"owned_by", "q27"}}})}};
        res.set_content(j.dump(), "application/json");
    });

    // ---------------- OpenAI chat/completions (text only) ----------------

    auto build_prompt = [&](const json& body) -> std::vector<int> {
        if (body.contains("messages")) {
            std::vector<std::pair<std::string, std::string>> msgs;
            for (auto& m : body["messages"]) {
                std::string role = m.value("role", "user");
                std::string content;
                if (m["content"].is_string()) content = m["content"];
                else if (m["content"].is_array())
                    for (auto& part : m["content"])
                        if (part.value("type", "") == "text") content += part.value("text", "");
                msgs.push_back({role, content});
            }
            // enable_thinking=false: top-level (Qwen-style clients) or nested
            // chat_template_kwargs (llama.cpp/GLM-style) -> empty-think prefill
            bool think = body.value("enable_thinking", true);
            if (body.contains("chat_template_kwargs"))
                think = body["chat_template_kwargs"].value("enable_thinking", think);
            if (no_think_srv) think = false;
            return tok.apply_chat_template(msgs, think);
        }
        return tok.encode(body.value("prompt", std::string()));
    };

    auto handle = [&](const httplib::Request& req, httplib::Response& res, bool chat) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return; }
        int n_max = body.value("max_tokens", 256);
        bool stream = body.value("stream", false);
        std::vector<int> prompt = build_prompt(body);
        if ((int)prompt.size() + n_max > ctx) n_max = ctx - (int)prompt.size();
        long created = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();

        const char* obj = chat ? "chat.completion" : "text_completion";
        const char* objd = chat ? "chat.completion.chunk" : "text_completion";

        if (!stream) {
            std::lock_guard<std::mutex> lk(gpu);
            std::string text;
            int n = eng.generate(prompt, n_max, EOS,
                                 [&](int id) { text += tok.decode_one(id); return true; });
            json choice;
            if (chat)
                choice = {{"index", 0}, {"finish_reason", n >= n_max ? "length" : "stop"},
                          {"message", {{"role", "assistant"}, {"content", text}}}};
            else
                choice = {{"index", 0}, {"finish_reason", n >= n_max ? "length" : "stop"},
                          {"text", text}};
            json out = {{"id", "q27-0"}, {"object", obj}, {"created", created},
                        {"model", "q27-qwopus-27b"}, {"choices", json::array({choice})},
                        {"usage", {{"prompt_tokens", (int)prompt.size()},
                                   {"completion_tokens", n},
                                   {"total_tokens", (int)prompt.size() + n}}}};
            res.set_content(out.dump(), "application/json");
            return;
        }

        res.set_header("Content-Type", "text/event-stream");
        res.set_chunked_content_provider(
            "text/event-stream",
            [&, prompt, n_max, created, chat, obj, objd](size_t, httplib::DataSink& sink) {
                std::lock_guard<std::mutex> lk(gpu);
                auto send = [&](const json& j) {
                    std::string s = "data: " + j.dump() + "\n\n";
                    return sink.write(s.data(), s.size());
                };
                eng.generate(prompt, n_max, EOS, [&](int id) {
                    std::string piece = tok.decode_one(id);
                    json delta = chat ? json{{"content", piece}} : json{};
                    json choice = chat
                        ? json{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}
                        : json{{"index", 0}, {"text", piece}, {"finish_reason", nullptr}};
                    json chunk = {{"id", "q27-0"}, {"object", objd}, {"created", created},
                                  {"model", "q27-qwopus-27b"}, {"choices", json::array({choice})}};
                    return send(chunk);
                });
                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
    };

    // ---------------- Anthropic /v1/messages ----------------

    // Anthropic tools -> qwen tools json for the system preamble
    auto anthropic_tools = [](const json& body) -> json {
        json out = json::array();
        if (body.contains("tools"))
            for (auto& t : body["tools"]) {
                if (!t.contains("name")) continue;
                out.push_back({{"type", "function"},
                               {"function", {{"name", t["name"]},
                                             {"description", t.value("description", "")},
                                             {"parameters", t.contains("input_schema")
                                                                ? t["input_schema"]
                                                                : json::object()}}}});
            }
        return out;
    };

    // Anthropic messages -> Msg list (thinking + tool_use reconstructed to
    // model markers, tool_result wrapped in <tool_response>)
    auto anthropic_msgs = [](const json& body) -> std::vector<Msg> {
        std::vector<Msg> msgs;
        if (body.contains("system")) {
            std::string sys;
            if (body["system"].is_string()) sys = body["system"];
            else if (body["system"].is_array())
                for (auto& b : body["system"])
                    if (b.value("type", "") == "text") sys += b.value("text", "");
            if (!sys.empty()) msgs.push_back({"system", sys});
        }
        for (auto& m : body["messages"]) {
            std::string role = m.value("role", "user"), think, content;
            if (m["content"].is_string()) content = m["content"];
            else if (m["content"].is_array())
                for (auto& part : m["content"]) {
                    std::string ty = part.value("type", "");
                    if (ty == "text") content += part.value("text", "");
                    else if (ty == "thinking") think += part.value("thinking", "");
                    else if (ty == "tool_use") {
                        if (!content.empty() && content.back() != '\n') content += "\n";
                        content += q27::tool_call_text(part.value("name", ""),
                                                       part.contains("input") ? part["input"]
                                                                              : json::object());
                    } else if (ty == "tool_result") {
                        std::string rc;
                        if (part.contains("content")) {
                            if (part["content"].is_string()) rc = part["content"];
                            else if (part["content"].is_array())
                                for (auto& b : part["content"])
                                    if (b.value("type", "") == "text") rc += b.value("text", "");
                        }
                        if (!content.empty() && content.back() != '\n') content += "\n";
                        content += q27::tool_response_text(rc);
                    }
                }
            if (role == "assistant" && !think.empty())
                content = "<think>\n" + think + "\n</think>\n" + content;
            msgs.push_back({role, content});
        }
        return msgs;
    };

    srv.Post("/v1/messages", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"type\":\"error\"}", "application/json"); return; }
        int n_max = body.value("max_tokens", 1024);
        bool stream = body.value("stream", false);
        json tools = anthropic_tools(body);
        auto tk0 = std::chrono::steady_clock::now();
        std::string rendered =
            q27::chatml_prompt(anthropic_msgs(body), tools, !no_think_srv);
        auto tk1 = std::chrono::steady_clock::now();
        std::vector<int> prompt = tok.encode(rendered);
        auto tk2 = std::chrono::steady_clock::now();
        fprintf(stderr, "[timing] render %.1fms encode %.1fms (%zu chars -> %zu toks)\n",
                std::chrono::duration<double, std::milli>(tk1 - tk0).count(),
                std::chrono::duration<double, std::milli>(tk2 - tk1).count(),
                rendered.size(), prompt.size());
        if ((int)prompt.size() + n_max > ctx) n_max = ctx - (int)prompt.size();
        long rid = req_counter++;
        std::string mid = "msg_q27_" + std::to_string(rid);

        if (!stream) {
            std::lock_guard<std::mutex> lk(gpu);
            StreamSplitter sp;
            std::string think, text, tool_buf;
            std::vector<q27::ToolCall> calls;
            auto route = [&](StreamSplitter::Chan ch, const std::string& t) {
                if (ch == StreamSplitter::TOOL) { tool_buf += t; return; }
                if (!tool_buf.empty()) { // tool segment closed
                    calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));
                    tool_buf.clear();
                }
                (ch == StreamSplitter::THINK ? think : text) += t;
            };
            int n = eng.generate(prompt, n_max, EOS, [&](int id) {
                for (auto& [ch, t] : sp.feed(tok.decode_one(id))) route(ch, t);
                return true;
            });
            for (auto& [ch, t] : sp.flush()) route(ch, t);
            if (!tool_buf.empty())
                calls.push_back(q27::parse_tool_call(q27::strip_ws2(tool_buf)));

            json content = json::array();
            std::string th = q27::strip_ws2(think), tx = q27::strip_ws2(text);
            if (!th.empty())
                content.push_back({{"type", "thinking"}, {"thinking", th},
                                   {"signature", "q27-local"}});
            bool any_call = false;
            int ci = 0;
            for (auto& c : calls) {
                if (!c.ok) { tx += (tx.empty() ? "" : "\n") + c.raw; continue; }
                any_call = true;
                (void)ci;
            }
            if (!any_call && tools.is_array() && !tools.empty()) {
                // wrapper-less call recovery (see parse_bare_tool_call)
                std::string pre, suf;
                auto bc = q27::parse_bare_tool_call(tx, &pre, &suf);
                if (bc.ok) {
                    fprintf(stderr, "[tool-fallback] bare '%s' recovered (nonstream), "
                            "%zu suffix bytes dropped\n", bc.name.c_str(), suf.size());
                    tx = pre;
                    calls.push_back(bc);
                    any_call = true;
                }
            }
            if (!tx.empty() || (!any_call && th.empty()))
                content.push_back({{"type", "text"}, {"text", tx}});
            for (auto& c : calls)
                if (c.ok)
                    content.push_back({{"type", "tool_use"},
                                       {"id", "toolu_q27_" + std::to_string(rid) + "_" +
                                                  std::to_string(ci++)},
                                       {"name", c.name}, {"input", c.arguments}});
            const char* sr = any_call ? "tool_use" : (n >= n_max ? "max_tokens" : "end_turn");
            json out = {{"id", mid}, {"type", "message"}, {"role", "assistant"},
                        {"model", "q27-qwopus-27b"}, {"content", content},
                        {"stop_reason", sr}, {"stop_sequence", nullptr},
                        {"usage", {{"input_tokens", (int)prompt.size()},
                                   {"output_tokens", n}}}};
            res.set_content(out.dump(), "application/json");
            return;
        }

        res.set_header("Content-Type", "text/event-stream");
        const bool has_tools = tools.is_array() && !tools.empty();
        res.set_chunked_content_provider(
            "text/event-stream",
            [&, prompt, n_max, mid, rid, has_tools](size_t, httplib::DataSink& sink) {
                std::lock_guard<std::mutex> lk(gpu);
                int block_counter = 0, tool_counter = 0;
                bool any_call = false;
                auto ev = [&](const char* name, const json& j) {
                    std::string s = std::string("event: ") + name + "\ndata: " + j.dump() + "\n\n";
                    return sink.write(s.data(), s.size());
                };
                json msg = {{"id", mid}, {"type", "message"}, {"role", "assistant"},
                            {"model", "q27-qwopus-27b"}, {"content", json::array()},
                            {"stop_reason", nullptr}, {"stop_sequence", nullptr},
                            {"usage", {{"input_tokens", (int)prompt.size()}, {"output_tokens", 0}}}};
                ev("message_start", {{"type", "message_start"}, {"message", msg}});

                StreamSplitter sp;
                std::string tool_buf, text_accum;
                int idx = -1;       // open think/text block index, -1 = none
                int chan_open = -1; // 0 text, 1 think
                bool any = false;
                auto close_block = [&]() {
                    if (idx < 0) return;
                    if (chan_open == 1)
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "signature_delta"}, {"signature", "q27-local"}}}});
                    ev("content_block_stop", {{"type", "content_block_stop"}, {"index", idx}});
                    idx = -1;
                };
                auto open_block = [&](int chan) {
                    if (idx >= 0 && chan_open != chan) close_block();
                    if (idx < 0) {
                        idx = block_counter++;
                        json cb = chan == 1 ? json{{"type", "thinking"}, {"thinking", ""}}
                                            : json{{"type", "text"}, {"text", ""}};
                        ev("content_block_start", {{"type", "content_block_start"},
                                                   {"index", idx}, {"content_block", cb}});
                        chan_open = chan;
                        any = true;
                    }
                };
                auto emit_tool = [&]() {
                    auto c = q27::parse_tool_call(q27::strip_ws2(tool_buf));
                    tool_buf.clear();
                    if (!c.ok) { // malformed: surface as text so nothing is lost
                        open_block(0);
                        text_accum += c.raw;
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "text_delta"}, {"text", c.raw}}}});
                        return;
                    }
                    any_call = true;
                    close_block();
                    int ti = block_counter++;
                    std::string tid = "toolu_q27_" + std::to_string(rid) + "_" +
                                      std::to_string(tool_counter++);
                    ev("content_block_start",
                       {{"type", "content_block_start"}, {"index", ti},
                        {"content_block", {{"type", "tool_use"}, {"id", tid}, {"name", c.name},
                                           {"input", json::object()}}}});
                    ev("content_block_delta",
                       {{"type", "content_block_delta"}, {"index", ti},
                        {"delta", {{"type", "input_json_delta"},
                                   {"partial_json", c.arguments.dump()}}}});
                    ev("content_block_stop", {{"type", "content_block_stop"}, {"index", ti}});
                };
                auto emit_seg = [&](StreamSplitter::Chan ch, const std::string& t) {
                    if (ch == StreamSplitter::TOOL) { tool_buf += t; return; }
                    if (!tool_buf.empty()) emit_tool();
                    if (t.empty()) return;
                    int chan = ch == StreamSplitter::THINK ? 1 : 0;
                    // suppress pure-whitespace text before the first block or between blocks
                    if (chan == 0 && idx < 0 && q27::strip_ws2(t).empty()) return;
                    open_block(chan);
                    if (chan == 0) text_accum += t;
                    if (chan == 1)
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "thinking_delta"}, {"thinking", t}}}});
                    else
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "text_delta"}, {"text", t}}}});
                };
                int produced = eng.generate(prompt, n_max, EOS, [&](int id) {
                    for (auto& [ch, t] : sp.feed(tok.decode_one(id))) emit_seg(ch, t);
                    return true;
                });
                for (auto& [ch, t] : sp.flush()) emit_seg(ch, t);
                if (!tool_buf.empty()) emit_tool();
                if (!any_call && has_tools) {
                    // wrapper-less call recovery: text already streamed as
                    // text_delta (cosmetic); the tool_use block still fires
                    std::string pre, suf;
                    auto bc = q27::parse_bare_tool_call(text_accum, &pre, &suf);
                    if (bc.ok) {
                        fprintf(stderr, "[tool-fallback] bare '%s' recovered (stream)\n",
                                bc.name.c_str());
                        any_call = true;
                        any = true;
                        close_block();
                        int ti = block_counter++;
                        std::string tid = "toolu_q27_" + std::to_string(rid) + "_" +
                                          std::to_string(tool_counter++);
                        ev("content_block_start",
                           {{"type", "content_block_start"}, {"index", ti},
                            {"content_block", {{"type", "tool_use"}, {"id", tid},
                                               {"name", bc.name}, {"input", json::object()}}}});
                        ev("content_block_delta",
                           {{"type", "content_block_delta"}, {"index", ti},
                            {"delta", {{"type", "input_json_delta"},
                                       {"partial_json", bc.arguments.dump()}}}});
                        ev("content_block_stop", {{"type", "content_block_stop"}, {"index", ti}});
                    }
                }
                if (idx < 0 && !any) { // nothing at all: empty text block for validity
                    idx = block_counter++;
                    chan_open = 0;
                    ev("content_block_start", {{"type", "content_block_start"}, {"index", idx},
                                               {"content_block", {{"type", "text"}, {"text", ""}}}});
                }
                close_block();
                const char* sr = any_call ? "tool_use"
                                          : (produced >= n_max ? "max_tokens" : "end_turn");
                ev("message_delta", {{"type", "message_delta"},
                                     {"delta", {{"stop_reason", sr}, {"stop_sequence", nullptr}}},
                                     {"usage", {{"output_tokens", produced}}}});
                ev("message_stop", {{"type", "message_stop"}});
                sink.done();
                return true;
            });
    });

    // ---------------- OpenAI Responses API (Codex CLI) ----------------
    // Wire facts from codex-rs (v0.143): client keys off the JSON `type` field
    // in SSE data (event: lines ignored); the agent loop consumes only
    // response.output_item.done items; response.completed{response:{id}} is the
    // required terminator; function_call.arguments is a JSON-encoded STRING;
    // tool results arrive as function_call_output with a bare-string output.
    // 400 is fatal to Codex, 500 retries -- so tolerate quirks, 500 on bugs.

    srv.Post("/v1/responses", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"error\":\"bad json\"}", "application/json"); return; }

        long rid = req_counter++;
        std::string resp_id = "resp_q27_" + std::to_string(rid);

        // tools: flat function entries pass through; `custom` freeform tools
        // (apply_patch) are bridged to a one-string-param function; hosted tool
        // types (web_search etc.) are skipped, never rejected.
        json tools = json::array();
        std::set<std::string> custom_names;
        if (body.contains("tools"))
            for (auto& t : body["tools"]) {
                std::string ty = t.value("type", "");
                if (ty == "function") {
                    tools.push_back({{"type", "function"},
                                     {"function", {{"name", t.value("name", "")},
                                                   {"description", t.value("description", "")},
                                                   {"parameters", t.contains("parameters")
                                                                      ? t["parameters"]
                                                                      : json::object()}}}});
                } else if (ty == "custom") {
                    std::string cn = t.value("name", "");
                    custom_names.insert(cn);
                    json params = {{"type", "object"},
                                   {"properties",
                                    {{"input", {{"type", "string"},
                                                {"description", "The complete raw input text "
                                                                "for this tool."}}}}},
                                   {"required", json::array({"input"})}};
                    tools.push_back({{"type", "function"},
                                     {"function", {{"name", cn},
                                                   {"description", t.value("description", "")},
                                                   {"parameters", params}}}});
                }
            }

        // input -> messages. instructions is the system prompt.
        std::vector<Msg> msgs;
        if (body.contains("instructions") && body["instructions"].is_string())
            msgs.push_back({"system", body["instructions"]});
        auto part_text = [](const json& content) {
            std::string out;
            if (content.is_string()) return content.get<std::string>();
            if (content.is_array())
                for (auto& p : content) {
                    std::string pt = p.value("type", "");
                    if (pt == "input_text" || pt == "output_text" || pt == "text")
                        out += p.value("text", "");
                }
            return out;
        };
        if (body.contains("input")) {
            if (body["input"].is_string()) {
                msgs.push_back({"user", body["input"]});
            } else if (body["input"].is_array()) {
                for (auto& it : body["input"]) {
                    std::string ty = it.value("type", "message");
                    if (ty == "message") {
                        std::string role = it.value("role", "user");
                        if (role == "developer") role = "system";
                        msgs.push_back({role, part_text(it["content"])});
                    } else if (ty == "function_call" || ty == "custom_tool_call") {
                        json args;
                        if (ty == "function_call") {
                            try { args = json::parse(it.value("arguments", "{}")); }
                            catch (...) { args = it.value("arguments", ""); }
                        } else {
                            args = {{"input", it.value("input", "")}};
                        }
                        msgs.push_back({"assistant",
                                        q27::tool_call_text(it.value("name", ""), args)});
                    } else if (ty == "function_call_output" || ty == "custom_tool_call_output") {
                        std::string out;
                        if (it.contains("output")) {
                            if (it["output"].is_string()) out = it["output"];
                            else out = part_text(it["output"]);
                        }
                        msgs.push_back({"user", q27::tool_response_text(out)});
                    }
                    // reasoning items in history are dropped: prior-turn thinking
                    // is not re-fed (matches the chat template's behavior)
                }
            }
        }
        // merge consecutive same-role messages (tool call + output sequences)
        std::vector<Msg> merged;
        for (auto& m : msgs) {
            if (!merged.empty() && merged.back().role == m.role)
                merged.back().content += "\n" + m.content;
            else
                merged.push_back(m);
        }

        int n_max = body.value("max_output_tokens", 4096);
        std::vector<int> prompt =
            tok.encode(q27::chatml_prompt(merged, tools, !no_think_srv));
        if ((int)prompt.size() + n_max > ctx) n_max = ctx - (int)prompt.size();
        if (n_max <= 0) {
            res.status = 400; // context_length_exceeded is fatal-class for codex, correctly
            res.set_content("{\"error\":{\"code\":\"context_length_exceeded\"}}",
                            "application/json");
            return;
        }
        bool stream = body.value("stream", false);

        // shared generation -> output items
        struct GenOut { json items = json::array(); int produced = 0; };
        auto make_item_cbs = [&](json& items, int& tool_counter,
                                 std::function<bool(const std::string&)> on_text_delta) {
            // returns the segment router; caller finishes with finish()
            struct Ctx {
                std::string think, text, tool_buf;
            };
            auto ctx = std::make_shared<Ctx>();
            auto flush_think = [&items, ctx, rid]() {
                std::string th = q27::strip_ws2(ctx->think);
                ctx->think.clear();
                if (th.empty()) return json();
                json r = {{"type", "reasoning"}, {"id", "rs_q27_" + std::to_string(rid)},
                          {"summary", json::array({{{"type", "summary_text"}, {"text", th}}})},
                          {"encrypted_content", nullptr}};
                items.push_back(r);
                return r;
            };
            auto flush_text = [&items, ctx, rid]() {
                std::string tx = q27::strip_ws2(ctx->text);
                ctx->text.clear();
                if (tx.empty()) return json();
                json m = {{"type", "message"}, {"id", "msg_q27_" + std::to_string(rid)},
                          {"role", "assistant"}, {"status", "completed"},
                          {"content",
                           json::array({{{"type", "output_text"}, {"text", tx},
                                         {"annotations", json::array()}}})}};
                items.push_back(m);
                return m;
            };
            auto flush_tool = [&items, ctx, rid, &tool_counter, &custom_names]() {
                auto c = q27::parse_tool_call(q27::strip_ws2(ctx->tool_buf));
                ctx->tool_buf.clear();
                std::string cid = "call_q27_" + std::to_string(rid) + "_" +
                                  std::to_string(tool_counter++);
                json it;
                if (!c.ok) { // malformed call: surface as text so codex shows it
                    it = {{"type", "message"}, {"role", "assistant"}, {"status", "completed"},
                          {"content", json::array({{{"type", "output_text"}, {"text", c.raw},
                                                    {"annotations", json::array()}}})}};
                } else if (custom_names.count(c.name)) {
                    std::string input = c.arguments.is_object() && c.arguments.contains("input") &&
                                                c.arguments["input"].is_string()
                                            ? c.arguments["input"].get<std::string>()
                                            : c.arguments.dump();
                    it = {{"type", "custom_tool_call"}, {"call_id", cid}, {"name", c.name},
                          {"input", input}};
                } else {
                    it = {{"type", "function_call"}, {"call_id", cid}, {"name", c.name},
                          {"arguments", c.arguments.dump()}};
                }
                items.push_back(it);
                return it;
            };
            return std::make_tuple(ctx, flush_think, flush_text, flush_tool);
        };

        if (!stream) {
            std::lock_guard<std::mutex> lk(gpu);
            json items = json::array();
            int tool_counter = 0;
            auto [ctx, flush_think, flush_text, flush_tool] =
                make_item_cbs(items, tool_counter, nullptr);
            StreamSplitter sp;
            auto route = [&](StreamSplitter::Chan ch, const std::string& t) {
                if (ch == StreamSplitter::TOOL) {
                    if (!ctx->think.empty()) flush_think();
                    if (!ctx->text.empty()) flush_text();
                    ctx->tool_buf += t;
                    return;
                }
                if (!ctx->tool_buf.empty()) flush_tool();
                if (ch == StreamSplitter::THINK) ctx->think += t;
                else {
                    if (!ctx->think.empty()) flush_think();
                    ctx->text += t;
                }
            };
            int produced = eng.generate(prompt, n_max, EOS, [&](int id) {
                for (auto& [ch, t] : sp.feed(tok.decode_one(id))) route(ch, t);
                return true;
            });
            for (auto& [ch, t] : sp.flush()) route(ch, t);
            if (!ctx->tool_buf.empty()) flush_tool();
            flush_think();
            flush_text();
            json out = {{"id", resp_id}, {"object", "response"}, {"status", "completed"},
                        {"model", "q27-qwopus-27b"}, {"output", items},
                        {"usage", {{"input_tokens", (int)prompt.size()},
                                   {"output_tokens", produced},
                                   {"total_tokens", (int)prompt.size() + produced}}}};
            res.set_content(out.dump(), "application/json");
            return;
        }

        res.set_header("Content-Type", "text/event-stream");
        res.set_chunked_content_provider(
            "text/event-stream",
            [&, prompt, n_max, resp_id, rid, custom_names](size_t, httplib::DataSink& sink) {
                std::lock_guard<std::mutex> lk(gpu);
                auto ev = [&](const json& j) {
                    // codex keys off data.type; the event: line is decorative
                    std::string s = "event: " + j.value("type", std::string("x")) +
                                    "\ndata: " + j.dump() + "\n\n";
                    return sink.write(s.data(), s.size());
                };
                ev({{"type", "response.created"},
                    {"response", {{"id", resp_id}, {"object", "response"},
                                  {"status", "in_progress"}}}});

                json items = json::array();
                int tool_counter = 0;
                std::set<std::string> cn = custom_names;
                std::string think, text, tool_buf;
                int out_index = 0;
                auto item_done = [&](const json& it) {
                    ev({{"type", "response.output_item.done"}, {"output_index", out_index++},
                        {"item", it}});
                    items.push_back(it);
                };
                auto flush_think = [&]() {
                    std::string th = q27::strip_ws2(think);
                    think.clear();
                    if (th.empty()) return;
                    item_done({{"type", "reasoning"}, {"id", "rs_q27_" + std::to_string(rid)},
                               {"summary",
                                json::array({{{"type", "summary_text"}, {"text", th}}})},
                               {"encrypted_content", nullptr}});
                };
                auto flush_text = [&]() {
                    std::string tx = q27::strip_ws2(text);
                    text.clear();
                    if (tx.empty()) return;
                    item_done({{"type", "message"}, {"id", "msg_q27_" + std::to_string(rid)},
                               {"role", "assistant"}, {"status", "completed"},
                               {"content",
                                json::array({{{"type", "output_text"}, {"text", tx},
                                              {"annotations", json::array()}}})}});
                };
                auto flush_tool = [&]() {
                    auto c = q27::parse_tool_call(q27::strip_ws2(tool_buf));
                    tool_buf.clear();
                    std::string cid = "call_q27_" + std::to_string(rid) + "_" +
                                      std::to_string(tool_counter++);
                    if (!c.ok) {
                        item_done({{"type", "message"}, {"role", "assistant"},
                                   {"status", "completed"},
                                   {"content",
                                    json::array({{{"type", "output_text"}, {"text", c.raw},
                                                  {"annotations", json::array()}}})}});
                    } else if (cn.count(c.name)) {
                        std::string input =
                            c.arguments.is_object() && c.arguments.contains("input") &&
                                    c.arguments["input"].is_string()
                                ? c.arguments["input"].get<std::string>()
                                : c.arguments.dump();
                        item_done({{"type", "custom_tool_call"}, {"call_id", cid},
                                   {"name", c.name}, {"input", input}});
                    } else {
                        item_done({{"type", "function_call"}, {"call_id", cid}, {"name", c.name},
                                   {"arguments", c.arguments.dump()}});
                    }
                };
                auto route = [&](StreamSplitter::Chan ch, const std::string& t) {
                    if (ch == StreamSplitter::TOOL) {
                        if (!think.empty()) flush_think();
                        if (!text.empty()) flush_text();
                        tool_buf += t;
                        return;
                    }
                    if (!tool_buf.empty()) flush_tool();
                    if (ch == StreamSplitter::THINK) { think += t; return; }
                    if (!think.empty()) flush_think();
                    if (text.empty() && q27::strip_ws2(t).empty()) return;
                    text += t;
                    ev({{"type", "response.output_text.delta"}, {"output_index", out_index},
                        {"content_index", 0}, {"delta", t}});
                };
                StreamSplitter sp;
                int produced = eng.generate(prompt, n_max, EOS, [&](int id) {
                    for (auto& [ch, t] : sp.feed(tok.decode_one(id))) route(ch, t);
                    return true;
                });
                for (auto& [ch, t] : sp.flush()) route(ch, t);
                if (!tool_buf.empty()) flush_tool();
                flush_think();
                flush_text();
                ev({{"type", "response.completed"},
                    {"response", {{"id", resp_id}, {"object", "response"},
                                  {"status", "completed"}, {"output", items},
                                  {"usage", {{"input_tokens", (int)prompt.size()},
                                             {"input_tokens_details", {{"cached_tokens", 0}}},
                                             {"output_tokens", produced},
                                             {"output_tokens_details", {{"reasoning_tokens", 0}}},
                                             {"total_tokens", (int)prompt.size() + produced}}}}}});
                sink.done();
                return true;
            });
    });

    srv.Post("/v1/chat/completions",
             [&](const httplib::Request& r, httplib::Response& s) { handle(r, s, true); });
    srv.Post("/v1/completions",
             [&](const httplib::Request& r, httplib::Response& s) { handle(r, s, false); });

    fprintf(stderr, "q27-server listening on http://%s:%d (ctx %d, %s head)\n", host.c_str(),
            port, ctx, fast ? "fast" : "faithful");
    srv.listen(host.c_str(), port);
    return 0;
}
