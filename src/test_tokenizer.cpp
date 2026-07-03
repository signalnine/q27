// Gate the C++ tokenizer against a reference id-list produced by llama-tokenize.
// Usage: test_tokenizer q27.tok cases.txt
// cases.txt: alternating lines — text line, then space-separated reference ids.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "api_common.h"
#include "tokenizer.h"

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
                   pre == "Let me write it." && q27::strip_ws2(suf) == "</file>";
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
        bool ok = ok1 && !c2.ok && !c3.ok && ok4;
        printf("bare tool-call fallback: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;
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
