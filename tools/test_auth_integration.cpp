// Integration test for the API-key pre-routing handler, using REAL
// httplib::Server + httplib::Client (no CUDA needed -- httplib is
// header-only and the auth logic doesn't touch the engine at all). This
// exercises the exact same handler shape wired into server.cu's main(),
// verbatim, against real HTTP requests over a real loopback socket --
// not a mock of the HTTP layer.
//
// Build+run: g++ -std=c++17 -I src -I third_party -pthread tools/test_auth_integration.cpp -o build/test_auth_integration && ./build/test_auth_integration
#include "api_common.h"
#include "../third_party/httplib.h"

#include <cstdio>
#include <thread>
#include <chrono>

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

// Verbatim copy of the pre-routing handler installed in server.cu's main()
// -- same shape, same helper calls (q27::extract_api_key/api_key_valid/
// auth_error_json), just parameterized on api_keys instead of capturing it
// from main()'s local scope.
static void install_auth(httplib::Server& srv, const std::vector<std::string>& api_keys) {
    if (api_keys.empty()) return;
    srv.set_pre_routing_handler([&api_keys](const httplib::Request& req, httplib::Response& res) {
        if (req.path == "/health") return httplib::Server::HandlerResponse::Unhandled;
        std::string provided = q27::extract_api_key(req.get_header_value("Authorization"),
                                                     req.get_header_value("x-api-key"));
        if (q27::api_key_valid(provided, api_keys))
            return httplib::Server::HandlerResponse::Unhandled;
        bool anthropic_shape = req.path.rfind("/v1/messages", 0) == 0;
        res.status = 401;
        if (!anthropic_shape) res.set_header("WWW-Authenticate", "Bearer");
        res.set_content(q27::auth_error_json(anthropic_shape), "application/json");
        return httplib::Server::HandlerResponse::Handled;
    });
}

struct TestServer {
    httplib::Server srv;
    std::thread th;
    int port = 0;
    std::vector<std::string> api_keys; // stable storage -- the pre-routing
                                       // handler captures a reference to
                                       // THIS member, not to whatever
                                       // temporary was passed to start()

    void start(std::vector<std::string> keys) {
        api_keys = std::move(keys);
        install_auth(srv, api_keys);
        srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });
        srv.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"object\":\"list\",\"data\":[]}", "application/json");
        });
        srv.Post("/v1/messages", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"type\":\"message\"}", "application/json");
        });
        srv.Post("/v1/chat/completions", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"object\":\"chat.completion\"}", "application/json");
        });
        port = srv.bind_to_any_port("127.0.0.1");
        th = std::thread([this] { srv.listen_after_bind(); });
        // wait for the listener to actually be accepting connections
        for (int i = 0; i < 200 && !srv.is_running(); i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ~TestServer() {
        srv.stop();
        if (th.joinable()) th.join();
    }
};

using json = nlohmann::json;

// ---- No keys configured: everything works unauthenticated (default,
// backward-compatible behavior -- must be UNCHANGED from before this
// feature existed). ----
static void test_no_auth_configured_everything_open() {
    TestServer s;
    s.start({}); // no keys -- auth disabled entirely
    httplib::Client cli("127.0.0.1", s.port);
    auto r1 = cli.Get("/health");
    CHECK(r1 && r1->status == 200);
    auto r2 = cli.Get("/v1/models");
    CHECK(r2 && r2->status == 200);
    auto r3 = cli.Post("/v1/messages", "{}", "application/json");
    CHECK(r3 && r3->status == 200);
}

// ---- Keys configured: /health stays open, everything else requires auth. ----
static void test_health_exempt_others_require_auth() {
    TestServer s;
    s.start({"secret-key"});
    httplib::Client cli("127.0.0.1", s.port);
    auto health = cli.Get("/health");
    CHECK(health && health->status == 200); // no Authorization header sent at all

    auto models_noauth = cli.Get("/v1/models");
    CHECK(models_noauth && models_noauth->status == 401);

    auto msgs_noauth = cli.Post("/v1/messages", "{}", "application/json");
    CHECK(msgs_noauth && msgs_noauth->status == 401);
}

// ---- Correct Bearer token is accepted (OpenAI/llama.cpp convention). ----
static void test_bearer_token_accepted() {
    TestServer s;
    s.start({"secret-key"});
    httplib::Client cli("127.0.0.1", s.port);
    httplib::Headers h = {{"Authorization", "Bearer secret-key"}};
    auto r = cli.Get("/v1/models", h);
    CHECK(r && r->status == 200);
    auto r2 = cli.Post("/v1/chat/completions", h, "{}", "application/json");
    CHECK(r2 && r2->status == 200);
}

// ---- Correct x-api-key is accepted (Anthropic convention -- what Claude
// Code actually sends). ----
static void test_x_api_key_accepted() {
    TestServer s;
    s.start({"secret-key"});
    httplib::Client cli("127.0.0.1", s.port);
    httplib::Headers h = {{"x-api-key", "secret-key"}};
    auto r = cli.Post("/v1/messages", h, "{}", "application/json");
    CHECK(r && r->status == 200);
}

// ---- Wrong key is rejected with 401 and the correctly-shaped body per
// endpoint family. ----
static void test_wrong_key_rejected_with_shaped_body() {
    TestServer s;
    s.start({"secret-key"});
    httplib::Client cli("127.0.0.1", s.port);

    httplib::Headers wrong = {{"Authorization", "Bearer wrong-key"}};
    auto r_oa = cli.Post("/v1/chat/completions", wrong, "{}", "application/json");
    CHECK(r_oa && r_oa->status == 401);
    if (r_oa) {
        json body = json::parse(r_oa->body);
        CHECK(body["error"]["type"] == "invalid_request_error");
        CHECK(body["error"]["code"] == "invalid_api_key");
        CHECK(r_oa->has_header("WWW-Authenticate"));
    }

    httplib::Headers wrong_x = {{"x-api-key", "wrong-key"}};
    auto r_anth = cli.Post("/v1/messages", wrong_x, "{}", "application/json");
    CHECK(r_anth && r_anth->status == 401);
    if (r_anth) {
        json body = json::parse(r_anth->body);
        CHECK(body["type"] == "error");
        CHECK(body["error"]["type"] == "authentication_error");
    }
}

// ---- Multiple configured keys: any of them works (key-rotation scenario). ----
static void test_multiple_keys_any_accepted() {
    TestServer s;
    s.start({"key-old", "key-new"});
    httplib::Client cli("127.0.0.1", s.port);
    httplib::Headers h_old = {{"Authorization", "Bearer key-old"}};
    httplib::Headers h_new = {{"Authorization", "Bearer key-new"}};
    auto r1 = cli.Get("/v1/models", h_old);
    CHECK(r1 && r1->status == 200);
    auto r2 = cli.Get("/v1/models", h_new);
    CHECK(r2 && r2->status == 200);
}

// ---- Malformed/absent Authorization header is treated as no key, not a
// crash -- e.g. "Authorization: garbage" (no Bearer prefix at all). ----
static void test_malformed_header_no_crash_rejected() {
    TestServer s;
    s.start({"secret-key"});
    httplib::Client cli("127.0.0.1", s.port);
    httplib::Headers h = {{"Authorization", "garbage-not-bearer-shaped"}};
    auto r = cli.Get("/v1/models", h);
    CHECK(r && r->status == 401);
}

int main() {
    test_no_auth_configured_everything_open();
    test_health_exempt_others_require_auth();
    test_bearer_token_accepted();
    test_x_api_key_accepted();
    test_wrong_key_rejected_with_shaped_body();
    test_multiple_keys_any_accepted();
    test_malformed_header_no_crash_rejected();
    if (failures) { fprintf(stderr, "%d FAILURE(S)\n", failures); return 1; }
    fprintf(stderr, "all auth integration tests passed\n");
    return 0;
}
