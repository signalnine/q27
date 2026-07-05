// Gate the C++ tokenizer against a reference id-list produced by llama-tokenize.
// Usage: test_tokenizer q27.tok cases.txt
// cases.txt: alternating lines — text line, then space-separated reference ids.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <chrono>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "api_common.h"
#include "tokenizer.h"
#include "toolgram.h"

// Self-test for the streaming UTF-8 boundary gate (no data files needed).
// BPE token boundaries can split a multi-byte character; the raw piece is
// invalid UTF-8 and nlohmann json::dump throws type_error.316 on it (took
// q27-server down mid-generation under Claude Code, R0 2026-07-04).
static int utf8gate_selftest() {
    int fail = 0;
    auto expect = [&](const std::string& got, const std::string& want, const char* name) {
        if (got != want) {
            printf("utf8gate FAIL %s: got %zu bytes, want %zu\n", name, got.size(),
                   want.size());
            fail++;
        }
    };
    {   // ascii passthrough
        q27::Utf8Gate g;
        expect(g.feed("abc"), "abc", "ascii");
        expect(g.flush(), "", "ascii-flush");
    }
    {   // em dash E2 80 94 split 2+1
        q27::Utf8Gate g;
        expect(g.feed("\xE2\x80"), "", "emdash-2+1 hold");
        expect(g.feed("\x94"), "\xE2\x80\x94", "emdash-2+1 release");
        expect(g.flush(), "", "emdash-2+1 flush");
    }
    {   // em dash split 1+2
        q27::Utf8Gate g;
        expect(g.feed("\xE2"), "", "emdash-1+2 hold");
        expect(g.feed("\x80\x94"), "\xE2\x80\x94", "emdash-1+2 release");
    }
    {   // 4-byte emoji F0 9F 98 80 split 2+2
        q27::Utf8Gate g;
        expect(g.feed("\xF0\x9F"), "", "emoji-2+2 hold");
        expect(g.feed("\x98\x80"), "\xF0\x9F\x98\x80", "emoji-2+2 release");
    }
    {   // text then partial tail in one piece
        q27::Utf8Gate g;
        expect(g.feed("ok\xE2\x80"), "ok", "mixed hold");
        expect(g.feed("\x94x"), "\xE2\x80\x94x", "mixed release");
    }
    {   // complete char + new partial in one piece
        q27::Utf8Gate g;
        expect(g.feed("\xE2\x80\x94\xE2"), "\xE2\x80\x94", "chain hold");
        expect(g.feed("\x80\x94"), "\xE2\x80\x94", "chain release");
    }
    {   // dangling partial at end of generation -> U+FFFD
        q27::Utf8Gate g;
        expect(g.feed("\xF0\x9F"), "", "dangle hold");
        expect(g.flush(), "\xEF\xBF\xBD", "dangle flush");
    }
    {   // lone continuation byte: pass through (dump-replace is the backstop)
        q27::Utf8Gate g;
        expect(g.feed("\x80"), "\x80", "invalid passthrough");
    }
    {   // two complete chars in one piece
        q27::Utf8Gate g;
        expect(g.feed("\xE2\x80\x94\xE2\x80\x94"), "\xE2\x80\x94\xE2\x80\x94", "two chars");
    }
    printf("utf8gate self-test: %s\n", fail ? "FAIL" : "PASS");
    return fail;
}

// Self-test for the R1b GPU gate (host-only; no CUDA, no data files).
// The gate replaces the server's whole-generation mutex: FIFO tickets,
// maybe_yield() re-enqueues at the tail so contended requests round-robin
// at round granularity, and the solo path is one relaxed atomic load.
static int gpu_gate_selftest() {
    int fail = 0;
    auto expect = [&](bool ok, const char* name) {
        if (!ok) { printf("gpugate FAIL %s\n", name); fail++; }
    };
    {   // C2: solo fast path -- no waiters, maybe_yield declines every time
        q27::GpuGate g;
        g.acquire();
        bool any = false;
        for (int i = 0; i < 1000; i++) any |= g.maybe_yield();
        g.release();
        expect(!any, "solo maybe_yield stays false");
        expect(!g.contended(), "solo never contended");
    }
    {   // C4: contended() sees a queued waiter, clears when it drains
        q27::GpuGate g;
        g.acquire();
        std::thread w([&] { g.acquire(); g.release(); });
        while (!g.contended()) std::this_thread::yield();
        expect(g.contended(), "waiter visible");
        g.release();
        w.join();
        expect(!g.contended(), "drained");
    }
    {   // C1: strict FIFO round-robin under 3-way contention. Admission
        // order is forced by polling contended() between spawns, so the
        // ticket order is T0 < T1 < T2 and the record sequence must be
        // exactly 0,1,2 repeated K times (maybe_yield re-enqueues at tail).
        q27::GpuGate g;
        const int K = 3;
        std::vector<int> seq;
        g.acquire(); // hold so spawned threads queue up in order
        std::vector<std::thread> ts;
        for (int i = 0; i < 3; i++) {
            ts.emplace_back([&, i] {
                g.acquire();
                for (int k = 0; k < K; k++) {
                    seq.push_back(i); // guarded by gate ownership
                    g.maybe_yield();
                }
                g.release();
            });
            while (g.contended() < i + 1) std::this_thread::yield();
        }
        g.release();
        for (auto& t : ts) t.join();
        bool ok = seq.size() == 3 * K;
        for (size_t j = 0; ok && j < seq.size(); j++) ok = seq[j] == (int)(j % 3);
        if (!ok) {
            printf("gpugate C1 sequence:");
            for (int v : seq) printf(" %d", v);
            printf("\n");
        }
        expect(ok, "FIFO round-robin order");
    }
    {   // C3: stress -- mutual exclusion holds and nothing deadlocks
        q27::GpuGate g;
        std::atomic<int> overlap{0};
        int inside = 0; // guarded by the gate; >1 means exclusion broke
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> ts;
        for (int i = 0; i < 8; i++)
            ts.emplace_back([&] {
                g.acquire();
                for (int k = 0; k < 200; k++) {
                    if (++inside != 1) overlap++;
                    --inside;
                    g.maybe_yield();
                }
                g.release();
            });
        for (auto& t : ts) t.join();
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                       .count();
        expect(overlap.load() == 0, "mutual exclusion");
        expect(s < 30.0, "stress completes");
    }
    printf("gpugate self-test: %s\n", fail ? "FAIL" : "PASS");
    return fail;
}

// Anthropic API shapes (count_tokens + ctx-limit error). Claude Code
// substring-matches "prompt is too long" in error.message to trigger
// compact-now instead of retrying; the envelope must be the real API's
// {"type":"error","error":{"type":...,"message":...}} or the SDK surfaces
// an empty error. anthropic_msgs/anthropic_tools_json are the /v1/messages
// request mapping extracted from the server so count_tokens counts EXACTLY
// what a generation request would prefill.
static int anthropic_api_selftest() {
    using nlohmann::json;
    int fail = 0;
    auto expect = [&](bool ok, const char* name) {
        if (!ok) { printf("anthapi FAIL %s\n", name); fail++; }
    };
    {   // error envelope: exact real-API shape
        json e = json::parse(q27::anthropic_error_json("invalid_request_error", "boom"));
        expect(e.value("type", "") == "error", "envelope type");
        expect(e.contains("error") && e["error"].value("type", "") == "invalid_request_error",
               "inner type");
        expect(e["error"].value("message", "") == "boom", "inner message");
    }
    {   // ctx-limit message: the byte string CC pattern-matches
        expect(q27::ctx_limit_error_message(213538, 204698) ==
                   "prompt is too long: 213538 tokens > 204698 maximum",
               "ctx-limit message");
    }
    {   // request mapping parity with the served /v1/messages path
        json body = json::parse(R"({
            "system": [{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.119; cc_entrypoint=cli; cch=a5145;You are Claude Code."}],
            "messages": [
                {"role":"user","content":"hello"},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"hmm"},
                    {"type":"text","text":"I'll call a tool."},
                    {"type":"tool_use","name":"ls","input":{"path":"/w"}}]},
                {"role":"user","content":[{"type":"tool_result","content":[{"type":"text","text":"a.md"}]}]}
            ]})");
        auto msgs = q27::anthropic_msgs(body);
        expect(msgs.size() == 4, "msg count");
        expect(msgs.size() == 4 && msgs[0].role == "system" &&
                   msgs[0].content.find("cch=fffff;") != std::string::npos,
               "system + cch normalized");
        expect(msgs.size() == 4 && msgs[1].role == "user" && msgs[1].content == "hello",
               "plain user");
        expect(msgs.size() == 4 && msgs[2].role == "assistant" &&
                   msgs[2].content == "<think>\nhmm\n</think>\nI'll call a tool.\n"
                                      "<tool_call>\n{\"name\": \"ls\", \"arguments\": "
                                      "{\"path\":\"/w\"}}\n</tool_call>",
               "assistant think+text+tool_use");
        expect(msgs.size() == 4 && msgs[3].role == "user" &&
                   msgs[3].content == "<tool_response>\na.md\n</tool_response>",
               "tool_result wrap");
        // system as plain string
        json b2 = {{"system", "S"}, {"messages", json::array()}};
        auto m2 = q27::anthropic_msgs(b2);
        expect(m2.size() == 1 && m2[0].role == "system" && m2[0].content == "S",
               "system string form");
        // missing messages key: no UB, system only
        json b3 = {{"system", "S"}};
        expect(q27::anthropic_msgs(b3).size() == 1, "missing messages tolerated");
        // message without content: no UB, empty content kept (role preserved)
        json b4 = json::parse(R"({"messages":[{"role":"user"}]})");
        auto m4 = q27::anthropic_msgs(b4);
        expect(m4.size() == 1 && m4[0].role == "user" && m4[0].content.empty(),
               "missing content tolerated");
    }
    {   // tools mapping: anthropic input_schema -> qwen function shape
        json body = json::parse(R"({"tools":[
            {"name":"ls","description":"list","input_schema":{"type":"object"}},
            {"description":"nameless skipped"},
            {"name":"noschema"}]})");
        json t = q27::anthropic_tools_json(body);
        expect(t.is_array() && t.size() == 2, "nameless skipped");
        expect(t.size() == 2 && t[0]["type"] == "function" &&
                   t[0]["function"]["name"] == "ls" &&
                   t[0]["function"]["description"] == "list" &&
                   t[0]["function"]["parameters"]["type"] == "object",
               "function shape");
        expect(t.size() == 2 && t[1]["function"]["parameters"].is_object() &&
                   t[1]["function"]["parameters"].empty() &&
                   t[1]["function"]["description"] == "",
               "schema/description defaults");
        expect(q27::anthropic_tools_json(json::object()).is_array(), "no tools -> empty array");
    }
    printf("anthropic api shapes: %s\n", fail ? "FAIL" : "PASS");
    return fail;
}

int main(int argc, char** argv) {
    if (utf8gate_selftest()) return 2;
    if (gpu_gate_selftest()) return 3;
    if (anthropic_api_selftest()) return 4;
    if (argc != 3) { fprintf(stderr, "usage: %s q27.tok cases.txt\n", argv[0]); return 1; }
    q27::Tokenizer tok(argv[1]);
    std::ifstream in(argv[2]);
    std::string text, idline;
    int total = 0, pass = 0, tok_total = 0, tok_match = 0;
    while (std::getline(in, text) && std::getline(in, idline)) {
        std::vector<int> ref;
        std::istringstream ss(idline);
        int v;
        while (ss >> v) ref.push_back(v);
        auto got = tok.encode(text);
        total++;
        bool ok = got == ref;
        if (ok) pass++;
        // token-level agreement (prefix match length) for partial credit visibility
        size_t m = 0;
        while (m < got.size() && m < ref.size() && got[m] == ref[m]) m++;
        tok_total += (int)ref.size();
        tok_match += (int)m;
        if (!ok) {
            printf("MISMATCH: \"%.50s\"\n  ref(%zu): ", text.c_str(), ref.size());
            for (size_t i = 0; i < ref.size() && i < 20; i++) printf("%d ", ref[i]);
            printf("\n  got(%zu): ", got.size());
            for (size_t i = 0; i < got.size() && i < 20; i++) printf("%d ", got[i]);
            printf("\n");
        }
    }
    printf("\nexact: %d/%d cases   token-prefix agreement: %d/%d = %.2f%%\n", pass, total,
           tok_match, tok_total, 100.0 * tok_match / (tok_total ? tok_total : 1));

    // bare tool-call fallback: models sometimes drop the <tool_call> wrapper
    // and emit the JSON as plain text (observed: Qwopus v1.4 no-think greedy
    // on long write calls, with trailing junk like "</file>"). The server must
    // recover the call, keep the prefix text, and drop trailing junk.
    {
        std::string pre, suf;
        auto c1 = q27::parse_bare_tool_call(
            "Let me write it.\n\n{\"name\": \"write\", \"arguments\": {\"path\": \"a.ts\", "
            "\"content\": \"x{y}\\\"z\\\"\"}}\n</file>", &pre, &suf);
        bool ok1 = c1.ok && c1.name == "write" && c1.arguments.value("path", "") == "a.ts" &&
                   pre == "Let me write it.";
        auto c2 = q27::parse_bare_tool_call("no tools here, just {braces} in prose", &pre, &suf);
        auto c3 = q27::parse_bare_tool_call("{\"name\": \"x\"}", &pre, &suf); // no arguments
        // observed failure mode: model emits the full call but never closes
        // the JSON -- ends the string with tag junk instead of "}} . Repair
        // must close framing without touching the payload.
        auto c4 = q27::parse_bare_tool_call(
            "Writing now.\n\n{\"name\": \"write\", \"arguments\": {\"file_path\": \"/w/s.ts\", "
            "\"content\": \"export const x = {a: 1};\\nexport {y};\\n\n</file>", &pre, &suf);
        bool ok4 = c4.ok && c4.name == "write" &&
                   c4.arguments.value("content", "").find("export const x") == 0 &&
                   c4.arguments.value("content", "").find("</file>") == std::string::npos &&
                   pre == "Writing now.";
        // third observed mode (task-queue/plugin-marketplace trials): the
        // model emits JSON framing but the content VALUE as raw code inside
        // <content>...</content> tags -- invalid JSON. Recovery must
        // JSON-escape the tagged span. Both orders (content-first and
        // content-last) occur.
        auto c5 = q27::parse_bare_tool_call(
            "{\"name\": \"write\", \"arguments\": {\"content\": <content>const a = \"x\";\n"
            "if (a) { b(); }\n</content>, \"file_path\": \"/w/s.ts\"}}", &pre, &suf);
        bool ok5 = c5.ok && c5.name == "write" &&
                   c5.arguments.value("file_path", "") == "/w/s.ts" &&
                   c5.arguments.value("content", "").find("const a = \"x\";") == 0;
        // fourth observed mode (task-queue transcripts): literal {"tool_call":
        // as the opener with </tool_call> as the closer, MULTIPLE calls per
        // message. The outer object never closes (net +1 depth per blob), so
        // the scanner must skip it and recover every inner {"name":...}
        // object. This was missed by a scan bug: unbalanced non-{"name"
        // candidates aborted the whole scan instead of trying the next '{'.
        auto v6 = q27::parse_bare_tool_calls(
            "Let me look.\n\n{\"tool_call\":\n{\"name\": \"ls\", \"arguments\": "
            "{\"path\": \"/w\"}}\n</tool_call>\n{\"tool_call\":\n{\"name\": \"view\", "
            "\"arguments\": {\"file_path\": \"/w/a.md\"}}\n</tool_call>", &pre);
        bool ok6 = v6.size() == 2 && v6[0].name == "ls" && v6[1].name == "view" &&
                   v6[1].arguments.value("file_path", "") == "/w/a.md" &&
                   pre == "Let me look.";
        // fifth observed mode (the task-queue root cause): RAW newlines/tabs
        // inside the content string -- the model writes literal multi-line
        // code where JSON requires \n escapes. Strict json::parse rejects
        // control chars; recovery must escape them in-string.
        auto v7 = q27::parse_bare_tool_calls(
            "{\"tool_call\":\n{\"name\": \"write\", \"arguments\": {\"content\": \"interface A {\n"
            "  x: number;\n\ty: string;\n}\nexport {A};\n\", \"file_path\": \"/w/s.ts\"}}\n"
            "</tool_call>", &pre);
        bool ok7 = v7.size() == 1 && v7[0].name == "write" &&
                   v7[0].arguments.value("file_path", "") == "/w/s.ts" &&
                   v7[0].arguments.value("content", "").find("interface A {\n  x: number;") == 0;
        bool ok = ok1 && !c2.ok && !c3.ok && ok4 && ok5 && ok6 && ok7;
        printf("bare tool-call fallback: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    }

    // P7: ToolGrammar -- char-level pushdown machine enforcing the
    // <tool_call> body. Each observed drift mode must be UNSAMPLEABLE:
    // the machine rejects the first illegal char, and done() stays false
    // until the call is complete (EOS gets masked off that state upstream).
    {
        std::vector<std::string> names = {"write", "view", "ls"};
        auto fresh = [&]() { q27::ToolGrammar g; g.reset(names); return g; };

        // legal call accepted end-to-end, done() flips only at the end
        auto g1 = fresh();
        std::string legal =
            "{\"name\": \"write\", \"arguments\": {\"file_path\": \"/w/s.ts\", "
            "\"content\": \"const a = 1;\\nexport {a};\"}}";
        bool ok_legal = true;
        for (size_t k = 0; k < legal.size(); k++) {
            if (k + 1 < legal.size() && g1.done()) ok_legal = false;  // early done
            if (!g1.advance(legal[k])) { ok_legal = false; break; }
        }
        ok_legal = ok_legal && g1.done();

        // mode 5: RAW newline inside a string -- illegal at that char
        auto g2 = fresh();
        bool m5 = g2.advance_str("{\"name\": \"write\", \"arguments\": {\"content\": \"line1");
        bool m5_reject = m5 && !g2.advance('\n');

        // mode 3: <content> tag in value position -- '<' is no legal value start
        auto g3 = fresh();
        bool m3 = g3.advance_str("{\"name\": \"write\", \"arguments\": {\"content\": ");
        bool m3_reject = m3 && !g3.advance('<');

        // mode 4: {"tool_call": opener -- key must be "name"
        auto g4 = fresh();
        bool m4 = g4.advance_str("{\"");
        bool m4_reject = m4 && !g4.advance('t');

        // unknown tool name rejected at first divergent char
        auto g5 = fresh();
        bool m6 = g5.advance_str("{\"name\": \"wri");
        bool m6_cont = m6 && g5.advance('t') && g5.advance('e');
        bool m6_reject = m6_cont && !g5.advance('x');  // "writex" is no tool

        // mode 2: unterminated call -- done() false mid-string (EOS gate)
        auto g6 = fresh();
        bool m2 = g6.advance_str("{\"name\": \"write\", \"arguments\": {\"content\": \"abc");
        bool m2_notdone = m2 && !g6.done();

        // token_ok simulates without committing: same state gives same answer
        auto g7 = fresh();
        g7.advance_str("{\"name\": \"view\", \"arguments\": {\"file_path\": \"/a\"}");
        bool tok_ok = g7.token_ok("}") && !g7.token_ok("]") && g7.token_ok("}") &&
                      !g7.done();

        // edges: nested containers, \uXXXX, exponent numbers, empty args,
        // literals, ws-heavy formatting -- all legal JSON must stay sampleable
        auto g8 = fresh();
        bool edge1 = g8.advance_str(
            "{ \"name\" : \"ls\" ,\n  \"arguments\" : {\"a\": [1, -2.5e+3, true, null, "
            "{\"b\": [\"\\u00e9\", false]}], \"c\": \"\"} }") && g8.done();
        auto g9 = fresh();
        bool edge2 = g9.advance_str("{\"name\": \"view\", \"arguments\": {}}") && g9.done();
        auto g10 = fresh();
        bool edge3 = g10.advance_str("{\"name\": \"write\", \"arguments\": {\"x\": 1}}") &&
                     g10.done() && g10.advance('\n') && !g10.advance('}');

        // the closer arrives as TEXT (the model emits <tool_call> markers as
        // plain BPE pieces, not the added token): after done(), the grammar
        // must accept ws + the literal "</tool_call>" and then report
        // closed(); a wrong char mid-closer is illegal
        auto g11 = fresh();
        bool closer_ok = g11.advance_str("{\"name\": \"ls\", \"arguments\": {}}") &&
                         g11.done() && !g11.closed() &&
                         g11.advance_str("\n</tool_call>") && g11.closed() &&
                         g11.advance('x');  // post-close: anything goes
        auto g12 = fresh();
        bool closer_bad = g12.advance_str("{\"name\": \"ls\", \"arguments\": {}}\n</tool") &&
                          !g12.advance('x');  // "</toolx" is not the closer

        bool ok = ok_legal && m5_reject && m3_reject && m4_reject && m6_reject &&
                  m2_notdone && tok_ok && edge1 && edge2 && edge3 && closer_ok && closer_bad;
        printf("toolgram drift modes: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    }

    // P8: stable-prefix boundary. chatml_prompt reports the char offset
    // where the trailing assistant-open begins; everything before it is
    // deterministic re-rendered history (snapshot-safe), everything after is
    // per-turn volatile (assistant open + think prefill). The boundary sits
    // right after "<|im_end|>\n" and right before "<|im_start|>assistant\n",
    // and encoding the prefix substring must be deterministic.
    {
        std::vector<q27::Msg> msgs = {{"system", "S"}, {"user", "hi"},
                                      {"assistant", "yo"}, {"user", "go"}};
        size_t off = 0;
        std::string r = q27::chatml_prompt(msgs, nlohmann::json::array(), false, &off);
        bool ok = off > 0 && off < r.size() &&
                  r.compare(off, 22, "<|im_start|>assistant\n") == 0 &&
                  off >= 11 && r.compare(off - 11, 11, "<|im_end|>\n") == 0;
        auto a1 = tok.encode(r.substr(0, off));
        auto b1 = tok.encode(r.substr(off));
        auto a2 = tok.encode(r.substr(0, off));
        ok = ok && a1 == a2 && !b1.empty() && !a1.empty();
        printf("chatml stable boundary: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    }

    // Claude Code billing-header cch stamp: it mutates on every request, so it
    // must normalize to a constant or the P8 prefix cache never holds under
    // Claude Code (mirrors llama.cpp #21793).
    {
        std::string a = "x-anthropic-billing-header: cc_version=2.1.101.e51; "
                        "cc_entrypoint=cli; cch=a5145;You are Claude Code.";
        std::string b = "x-anthropic-billing-header: cc_version=2.1.101.e51; "
                        "cc_entrypoint=cli; cch=b9997;You are Claude Code.";
        q27::normalize_cc_billing_header(a);
        q27::normalize_cc_billing_header(b);
        bool ok = a == b && a.find("cch=fffff;") != std::string::npos;
        // 2.1.200-era header carries no cch -> untouched
        std::string c = "x-anthropic-billing-header: cc_version=2.1.200.77d; "
                        "cc_entrypoint=sdk-cli;You are a Claude agent.";
        std::string c0 = c;
        q27::normalize_cc_billing_header(c);
        ok = ok && c == c0;
        // non-CC prompt with a stray cch= in the body -> untouched
        std::string d = "You are a bot. Config: cch=zzzzz; end.";
        std::string d0 = d;
        q27::normalize_cc_billing_header(d);
        ok = ok && d == d0;
        // longer stamp still pinned, shape preserved
        std::string e = "x-anthropic-billing-header: cch=deadbeef01;You are Claude Code.";
        q27::normalize_cc_billing_header(e);
        ok = ok && e.find("cch=ffffffffff;") != std::string::npos;
        printf("billing-header cch normalize: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    }

    // count_tokens equivalence: the count must equal what /v1/messages
    // reports as usage.input_tokens for the same body. The served path
    // split-encodes at the P8 stable boundary; count_tokens encodes the
    // whole string. Equal only because the boundary abuts the <|im_start|>
    // added token (split-invariant tokenization there) -- gate it.
    {
        nlohmann::json body = nlohmann::json::parse(R"({
            "system": "You are terse.",
            "messages": [
                {"role":"user","content":"hi"},
                {"role":"assistant","content":"yo"},
                {"role":"user","content":"count something for me"}],
            "tools": [{"name":"ls","description":"list files","input_schema":{"type":"object"}}]})");
        size_t off = 0;
        std::string whole = q27::chatml_prompt(q27::anthropic_msgs(body),
                                               q27::anthropic_tools_json(body), false, &off);
        auto w = tok.encode(whole);
        auto a = tok.encode(whole.substr(0, off));
        auto b = tok.encode(whole.substr(off));
        bool ok = !w.empty() && w.size() == a.size() + b.size();
        printf("count_tokens == split-encode input_tokens: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    }

    // P7: ToolMaskCache -- exact lazy vocab-legality bitmasks keyed by
    // grammar-state signature. Synthetic vocab for logic; real vocab for a
    // sweep-cost measurement (design viability: a miss must be low-ms).
    {
        std::vector<std::string> vocab = {"{",  "\"",   "name", "\": \"", "write",
                                          "abc", "\n",  "}",    " ",      "</tool_call>",
                                          ""};
        const int CLOSER = 9, CTRL = 10;
        q27::ToolMaskCache mc;
        mc.init(&vocab, CLOSER);
        q27::ToolGrammar g;
        g.reset({"write", "view"});
        auto bit = [&](int mi, int id) {
            return (mc.mask(mi)[id >> 5] >> (id & 31)) & 1u;
        };
        int m0 = mc.get(g);
        bool fresh_ok = bit(m0, 0) && bit(m0, 8) && !bit(m0, 5) && !bit(m0, CLOSER) &&
                        !bit(m0, CTRL);
        int m0b = mc.get(g);
        bool cache_hit = (m0b == m0) && mc.size() == 1;
        g.advance_str("{\"name\": \"write\", \"arguments\": {\"content\": \"");
        int m1 = mc.get(g);
        // in-string: text legal, raw newline token illegal, '{' legal as
        // string CONTENT, closer still illegal
        bool instr_ok = bit(m1, 5) && !bit(m1, 6) && bit(m1, 0) && !bit(m1, CLOSER) &&
                        m1 != m0;
        g.advance_str("x\"}}");
        int m2 = mc.get(g);
        bool done_ok = g.done() && bit(m2, CLOSER) && bit(m2, 8) && !bit(m2, 7);
        bool ok = fresh_ok && cache_hit && instr_ok && done_ok;
        printf("toolmask cache: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    }
    // real-vocab sweep cost (informational)
    {
        auto vb = tok.vocab_bytes();
        int closer = tok.token_id("</tool_call>");
        q27::ToolMaskCache mc;
        mc.init(&vb, closer);
        q27::ToolGrammar g;
        g.reset({"write", "view", "ls", "bash", "grep", "glob", "edit"});
        auto t0 = std::chrono::steady_clock::now();
        int a = mc.get(g);
        g.advance_str("{\"name\": \"write\", \"arguments\": {\"content\": \"");
        int b = mc.get(g);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 2;
        long legal = 0;
        for (uint32_t w : mc.mask(b)) legal += __builtin_popcount(w);
        printf("  toolmask real vocab: %.1f ms/sweep, in-string legal tokens %ld/%zu "
               "(closer id %d, masks %zu)\n", ms, legal, vb.size(), closer, mc.size());
        (void)a;
    }

    // added-token matching: encode() must map literal <think>/</think> text to
    // their single vocab ids (HF added-token semantics; BPE merges cannot form
    // them). The server's Anthropic path string-renders prompts, so prefills
    // and prior-turn thinking reconstruction depend on this.
    {
        int t1 = tok.token_id("<think>"), t2 = tok.token_id("</think>");
        auto e1 = tok.encode("<think>"), e2 = tok.encode("</think>");
        bool ok = t1 >= 0 && t2 >= 0 && e1.size() == 1 && e1[0] == t1 &&
                  e2.size() == 1 && e2[0] == t2;
        printf("added-token encode <think>: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    }

    // chat-template no-think: think=false must append the empty think block
    // exactly like the Qwen3-family template with enable_thinking=false, and
    // it must use the SINGLE added-token ids (BPE renders "<think>" as 3
    // ordinary tokens, which the model does not treat as a think block)
    {
        std::vector<std::pair<std::string, std::string>> msgs = {{"user", "hi"}};
        auto ta = tok.apply_chat_template(msgs);
        auto tb = tok.apply_chat_template(msgs, false);
        std::string a = tok.decode(ta), b = tok.decode(tb);
        int t1 = tok.token_id("<think>"), t2 = tok.token_id("</think>");
        bool ids_ok = t1 >= 0 && t2 >= 0 && tb.size() > ta.size() &&
                      std::find(tb.begin(), tb.end(), t1) != tb.end() &&
                      std::find(tb.begin(), tb.end(), t2) != tb.end();
        bool ok = b == a + "<think>\n\n</think>\n\n" && ids_ok;
        printf("chat nothink suffix (single-token ids): %s\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    }
    return pass == total ? 0 : 1;
}
