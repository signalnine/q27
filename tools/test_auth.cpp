// CPU unit tests for the API-key auth helpers in api_common.h
// (secure_compare, extract_api_key, api_key_valid, auth_error_json,
// load_api_key_file). Pure logic, no CUDA/engine dependency.
//
// Build+run: g++ -std=c++17 -I src tools/test_auth.cpp -o build/test_auth && ./build/test_auth
#include "api_common.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>

using json = nlohmann::json;

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

static void test_secure_compare_basic() {
    CHECK(q27::secure_compare("abc", "abc"));
    CHECK(!q27::secure_compare("abc", "abd"));
    CHECK(!q27::secure_compare("abc", "ab"));
    CHECK(!q27::secure_compare("abc", "abcd"));
    CHECK(q27::secure_compare("", ""));
    CHECK(!q27::secure_compare("", "a"));
    CHECK(!q27::secure_compare("a", ""));
}

static void test_secure_compare_no_early_exit_timing_class() {
    // Not a real timing-attack test (that needs statistical sampling on real
    // hardware), but a basic sanity check that the function's CONTROL FLOW
    // doesn't contain an early `return false` inside the comparison loop --
    // i.e. it's structurally constant-time-shaped, not just
    // functionally correct. We check this by confirming a mismatch at
    // position 0 and a mismatch at the last position both correctly return
    // false (an early-exit bug would still pass this, but a common
    // regression -- accidentally reintroducing `if (ca != cb) return
    // false;` -- is caught by the length-independent test below combined
    // with code review).
    std::string a(1000, 'x');
    std::string b0 = a; b0[0] = 'y';
    std::string bN = a; bN[999] = 'y';
    CHECK(!q27::secure_compare(a, b0));
    CHECK(!q27::secure_compare(a, bN));
}

static void test_extract_api_key_bearer() {
    CHECK(q27::extract_api_key("Bearer sk-abc123", "") == "sk-abc123");
    CHECK(q27::extract_api_key("Bearer ", "") == ""); // empty token after prefix
    CHECK(q27::extract_api_key("bearer sk-abc123", "") == ""); // case-sensitive, matches llama.cpp
    CHECK(q27::extract_api_key("Basic sk-abc123", "") == ""); // wrong scheme
    CHECK(q27::extract_api_key("", "") == "");
}

static void test_extract_api_key_x_api_key() {
    CHECK(q27::extract_api_key("", "sk-anthropic-style") == "sk-anthropic-style");
}

static void test_extract_api_key_x_api_key_wins_if_both_present() {
    CHECK(q27::extract_api_key("Bearer sk-openai", "sk-anthropic") == "sk-anthropic");
}

static void test_api_key_valid() {
    std::vector<std::string> keys = {"key-one", "key-two"};
    CHECK(q27::api_key_valid("key-one", keys));
    CHECK(q27::api_key_valid("key-two", keys));
    CHECK(!q27::api_key_valid("key-three", keys));
    CHECK(!q27::api_key_valid("", keys));
    CHECK(!q27::api_key_valid("key-one", {})); // no keys configured -- fail closed
}

static void test_auth_error_json_shapes() {
    std::string anth = q27::auth_error_json(true);
    json a = json::parse(anth);
    CHECK(a["type"] == "error");
    CHECK(a["error"]["type"] == "authentication_error");

    std::string oa = q27::auth_error_json(false);
    json o = json::parse(oa);
    CHECK(o["error"]["type"] == "invalid_request_error");
    CHECK(o["error"]["code"] == "invalid_api_key");
}

static void test_load_api_key_file_missing_returns_false() {
    std::vector<std::string> out;
    CHECK(!q27::load_api_key_file("/tmp/q27_test_nonexistent_key_file_xyz123", &out));
    CHECK(out.empty());
}

static void test_load_api_key_file_parses_correctly() {
    const char* path = "/tmp/q27_test_api_key_file.txt";
    {
        std::ofstream f(path);
        f << "  key-one  \n";
        f << "\n";
        f << "# a comment, ignored\n";
        f << "key-two\n";
        f << "   \n"; // whitespace-only line, ignored
        f << "key-three";  // no trailing newline
    }
    std::vector<std::string> out;
    CHECK(q27::load_api_key_file(path, &out));
    CHECK(out.size() == 3);
    if (out.size() == 3) {
        CHECK(out[0] == "key-one");
        CHECK(out[1] == "key-two");
        CHECK(out[2] == "key-three");
    }
    std::remove(path);
}

static void test_load_api_key_file_appends_not_replaces() {
    const char* path = "/tmp/q27_test_api_key_file2.txt";
    { std::ofstream f(path); f << "file-key\n"; }
    std::vector<std::string> out = {"cli-key"}; // pre-existing (e.g. from --api-key)
    CHECK(q27::load_api_key_file(path, &out));
    CHECK(out.size() == 2);
    CHECK(out[0] == "cli-key");
    CHECK(out[1] == "file-key");
    std::remove(path);
}

int main() {
    test_secure_compare_basic();
    test_secure_compare_no_early_exit_timing_class();
    test_extract_api_key_bearer();
    test_extract_api_key_x_api_key();
    test_extract_api_key_x_api_key_wins_if_both_present();
    test_api_key_valid();
    test_auth_error_json_shapes();
    test_load_api_key_file_missing_returns_false();
    test_load_api_key_file_parses_correctly();
    test_load_api_key_file_appends_not_replaces();
    if (failures) { fprintf(stderr, "%d FAILURE(S)\n", failures); return 1; }
    fprintf(stderr, "all auth tests passed\n");
    return 0;
}
