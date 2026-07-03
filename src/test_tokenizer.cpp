// Gate the C++ tokenizer against a reference id-list produced by llama-tokenize.
// Usage: test_tokenizer q27.tok cases.txt
// cases.txt: alternating lines — text line, then space-separated reference ids.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>

#include "api_common.h"
#include "tokenizer.h"
#include "toolgram.h"

int main(int argc, char** argv) {
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
