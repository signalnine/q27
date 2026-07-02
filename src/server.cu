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

    srv.Post("/v1/chat/completions",
             [&](const httplib::Request& r, httplib::Response& s) { handle(r, s, true); });
    srv.Post("/v1/completions",
             [&](const httplib::Request& r, httplib::Response& s) { handle(r, s, false); });

    fprintf(stderr, "q27-server listening on http://%s:%d (ctx %d, %s head)\n", host.c_str(),
            port, ctx, fast ? "fast" : "faithful");
    srv.listen(host.c_str(), port);
    return 0;
}
