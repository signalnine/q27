// Gate the C++ tokenizer against a reference id-list produced by llama-tokenize.
// Usage: test_tokenizer q27.tok cases.txt
// cases.txt: alternating lines — text line, then space-separated reference ids.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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
