// q27 OpenAI-compatible HTTP server. Single slot (MTP spec decode is 1-stream),
// greedy only. Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions
// (stream + non-stream), POST /v1/completions (text prompt).
//
// usage: q27-server model.q27 model.tok [--port 8080] [--host 0.0.0.0]
//                   [--ctx 8192] [--fast-head]
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>

#include "engine.cuh"
#include "tokenizer.h"
#include "think_split.h"
#include "../third_party/httplib.h"
#include "../third_party/json.hpp"

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.q27 model.tok [--port N] [--host H] [--ctx C] "
                        "[--fast-head]\n", argv[0]);
        return 1;
    }
    std::string model = argv[1], tokpath = argv[2], host = "0.0.0.0";
    int port = 8080, ctx = 8192;
    bool fast = false;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--host") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fast-head")) fast = true;
    }

    fprintf(stderr, "loading tokenizer...\n");
    q27::Tokenizer tok(tokpath);
    fprintf(stderr, "loading model...\n");
    Engine eng(model, ctx);
    eng.fast_head = fast;
    eng.build_graph();
    eng.build_spec_graphs();
    const int EOS = tok.eos();

    std::mutex gpu; // single slot: serialize requests

    httplib::Server srv;

    srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json j = {{"object", "list"},
                  {"data", json::array({{{"id", "q27-qwopus-27b"}, {"object", "model"},
                                         {"owned_by", "q27"}}})}};
        res.set_content(j.dump(), "application/json");
    });

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
            return tok.apply_chat_template(msgs);
        }
        return tok.encode(body.value("prompt", std::string()));
    };

    // Anthropic: top-level `system` (string or blocks) + messages with string or
    // block content -> ChatML token ids.
    auto build_prompt_anthropic = [&](const json& body) -> std::vector<int> {
        std::vector<std::pair<std::string, std::string>> msgs;
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
                    // tool_use / tool_result handled by the responses path, not here
                }
            // reconstruct assistant reasoning so multi-turn context is faithful
            if (role == "assistant" && !think.empty())
                content = "<think>\n" + think + "\n</think>\n" + content;
            msgs.push_back({role, content});
        }
        return tok.apply_chat_template(msgs);
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

        // streaming SSE
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

    srv.Post("/v1/messages", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"type\":\"error\"}", "application/json"); return; }
        int n_max = body.value("max_tokens", 256);
        bool stream = body.value("stream", false);
        std::vector<int> prompt = build_prompt_anthropic(body);
        if ((int)prompt.size() + n_max > ctx) n_max = ctx - (int)prompt.size();
        std::string mid = "msg_q27";

        if (!stream) {
            std::lock_guard<std::mutex> lk(gpu);
            q27::ThinkSplitter sp;
            std::string think, text;
            int n = eng.generate(prompt, n_max, EOS, [&](int id) {
                for (auto& [ch, t] : sp.feed(tok.decode_one(id)))
                    (ch == q27::ThinkSplitter::THINK ? think : text) += t;
                return true;
            });
            for (auto& [ch, t] : sp.flush())
                (ch == q27::ThinkSplitter::THINK ? think : text) += t;
            json content = json::array();
            std::string th = q27::strip_ws(think);
            if (!th.empty())
                content.push_back({{"type", "thinking"}, {"thinking", th},
                                   {"signature", "q27-local"}});
            content.push_back({{"type", "text"}, {"text", q27::strip_ws(text)}});
            json out = {{"id", mid}, {"type", "message"}, {"role", "assistant"},
                        {"model", "q27-qwopus-27b"}, {"content", content},
                        {"stop_reason", n >= n_max ? "max_tokens" : "end_turn"},
                        {"stop_sequence", nullptr},
                        {"usage", {{"input_tokens", (int)prompt.size()},
                                   {"output_tokens", n}}}};
            res.set_content(out.dump(), "application/json");
            return;
        }

        res.set_header("Content-Type", "text/event-stream");
        res.set_chunked_content_provider(
            "text/event-stream",
            [&, prompt, n_max, mid](size_t, httplib::DataSink& sink) {
                std::lock_guard<std::mutex> lk(gpu);
                int block_counter = 0;
                auto ev = [&](const char* name, const json& j) {
                    std::string s = std::string("event: ") + name + "\ndata: " + j.dump() + "\n\n";
                    return sink.write(s.data(), s.size());
                };
                json msg = {{"id", mid}, {"type", "message"}, {"role", "assistant"},
                            {"model", "q27-qwopus-27b"}, {"content", json::array()},
                            {"stop_reason", nullptr}, {"stop_sequence", nullptr},
                            {"usage", {{"input_tokens", (int)prompt.size()}, {"output_tokens", 0}}}};
                ev("message_start", {{"type", "message_start"}, {"message", msg}});

                q27::ThinkSplitter sp;
                int idx = -1;            // current open content block index, -1 = none
                int chan_open = -1;      // which channel the open block is (0 text, 1 think)
                bool any = false;        // any block opened yet (to drop leading ws)
                auto open_block = [&](int chan) {
                    if (idx >= 0 && chan_open != chan) { // close the wrong-channel block
                        if (chan_open == 1)
                            ev("content_block_delta", {{"type", "content_block_delta"},
                                {"index", idx},
                                {"delta", {{"type", "signature_delta"}, {"signature", "q27-local"}}}});
                        ev("content_block_stop", {{"type", "content_block_stop"}, {"index", idx}});
                        idx = -1;
                    }
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
                auto emit_seg = [&](int chan, const std::string& t) {
                    if (t.empty()) return;
                    if (chan == 0 && !any && q27::strip_ws(t).empty()) return; // drop leading ws
                    open_block(chan);
                    if (chan == 1)
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "thinking_delta"}, {"thinking", t}}}});
                    else
                        ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                            {"delta", {{"type", "text_delta"}, {"text", t}}}});
                };
                int produced = eng.generate(prompt, n_max, EOS, [&](int id) {
                    for (auto& [ch, t] : sp.feed(tok.decode_one(id)))
                        emit_seg(ch == q27::ThinkSplitter::THINK ? 1 : 0, t);
                    return true;
                });
                for (auto& [ch, t] : sp.flush())
                    emit_seg(ch == q27::ThinkSplitter::THINK ? 1 : 0, t);
                if (idx < 0) { // nothing emitted: open an empty text block for protocol validity
                    idx = block_counter++;
                    ev("content_block_start", {{"type", "content_block_start"}, {"index", idx},
                                               {"content_block", {{"type", "text"}, {"text", ""}}}});
                }
                // thinking blocks need a signature_delta before close
                if (chan_open == 1)
                    ev("content_block_delta", {{"type", "content_block_delta"}, {"index", idx},
                        {"delta", {{"type", "signature_delta"}, {"signature", "q27-local"}}}});
                ev("content_block_stop", {{"type", "content_block_stop"}, {"index", idx}});
                ev("message_delta", {{"type", "message_delta"},
                                     {"delta", {{"stop_reason",
                                                 produced >= n_max ? "max_tokens" : "end_turn"},
                                                {"stop_sequence", nullptr}}},
                                     {"usage", {{"output_tokens", produced}}}});
                ev("message_stop", {{"type", "message_stop"}});
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
