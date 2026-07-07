// E2E harness for the P15 engage-lag fix (docs/plans/2026-07-07-constrain-tools.md).
// Loads model + tokenizer, renders a ChatML tools prompt with the PRODUCTION
// preamble (api_common.h chatml_prompt), decodes greedily with the constrainer
// fully wired (scan_round -> on_round -> refinish_round; on_pending staging;
// on_id feeding), and emits machine-parsable counters + the raw generation.
// tools/constrain_gate.sh runs the assertions (C9/C10/C12/C13).
//
// Build (gate script): nvcc -O2 -std=c++17 -gencode arch=compute_120,code=sm_120
//   tools/constrain_e2e.cu src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu
//   src/device_model.cu src/loader.cpp src/tokenizer.cpp -o build/constrain_e2e
#include "../src/engine.cuh"
#include "../src/api_common.h"
#include "../src/tokenizer.h"
#include "../src/toolconstrain.h"

#include <string>
#include <vector>

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s model.q27 model.tok [--tools a,b] [--user TEXT] [-n N] [--ctx C]\n"
                "       [--fast-head] [--out FILE] [--no-constrain]\n",
                argv[0]);
        return 1;
    }
    std::string model = argv[1], tokpath = argv[2];
    std::string tools_csv = "getg_project,run_tests";
    std::string user = "Please list the files in the project.";
    std::string out_path;
    int n_gen = 250, ctx = 4096;
    bool fast = false, constrain = true, leak_test = false;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--tools") && i + 1 < argc) tools_csv = argv[++i];
        else if (!strcmp(argv[i], "--user") && i + 1 < argc) user = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fast-head")) fast = true;
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--no-constrain")) constrain = false;
        else if (!strcmp(argv[i], "--leak-test")) leak_test = true;
    }
    std::vector<std::string> names;
    for (size_t p = 0, q; p < tools_csv.size(); p = q + 1) {
        q = tools_csv.find(',', p);
        if (q == std::string::npos) q = tools_csv.size();
        if (q > p) names.push_back(tools_csv.substr(p, q - p));
    }

    q27::Tokenizer tok(tokpath);
    json tools = json::array();
    for (auto& n : names)
        tools.push_back({{"name", n},
                         {"description", "Tool " + n},
                         {"parameters",
                          {{"type", "object"},
                           {"properties", json::object()},
                           {"required", json::array()}}}});
    std::vector<q27::Msg> msgs{{"user", user}};
    std::string ptxt = q27::chatml_prompt(msgs, tools, /*think=*/false);
    std::vector<int> prompt = tok.encode(ptxt);
    fprintf(stderr, "[e2e] prompt tokens: %zu\n", prompt.size());

    Engine e(model, ctx);
    e.fast_head = fast;
    e.build_graph();
    e.build_spec_graphs();

    // C8 clear-at-claim gate: plant a stale RESTRICTIVE constraint (one legal
    // token + accept-cap-1, the exact leak shape from the 07-05 audit), then
    // run two plain generations and require byte-identity. Without the
    // generate()-entry clear, leg A's first decision is forced to the planted
    // token and the legs diverge.
    if (leak_test) {
        std::vector<uint32_t> only(((size_t)VOCAB + 31) / 32, 0);
        int forced = tok.encode("zebra")[0];
        only[forced >> 5] |= 1u << (forced & 31);
        int mid = e.mask_pool_add(only.data());
        std::vector<int> prompt2 = tok.encode(q27::chatml_prompt(msgs, json::array(), false));
        auto run = [&](const char* tag) {
            std::string t;
            e.generate(prompt2, 64, tok.eos(), [&](int id) {
                t += tok.decode_one(id);
                return true;
            });
            fprintf(stderr, "[leak-%s] %s\n", tag, t.substr(0, 60).c_str());
            return t;
        };
        e.set_tool_constraint(mid); // simulate the leak (no tc.end() ran)
        std::string a = run("stale");
        e.clear_tool_constraint(); // ground truth: an explicitly clean engine
        std::string b = run("clean");
        printf("LEAK_TEST=%s\n", a == b ? "PASS" : "FAIL");
        return a == b ? 0 : 1;
    }

    std::vector<std::string> vocab = tok.vocab_bytes();
    q27::ToolMaskCache cache;
    cache.init(&vocab, tok.token_id("</tool_call>"));
    std::vector<int> host2dev;
    q27::BasicToolConstrainer<Engine, q27::Tokenizer> tc;
    tc.eng = &e;
    tc.tok = &tok;
    tc.cache = &cache;
    tc.host2dev = &host2dev;
    tc.enabled = constrain;
    tc.begin(names);

    long refinish = 0, trunc = 0;
    e.on_pending = [&](int id) { tc.on_pending(id); };
    e.on_round = [&](const int* em, int n) {
        int m = tc.scan_round(em, n);
        if (m >= 1) {
            refinish++;
            if (m < n) trunc++;
        }
        return m;
    };
    std::string text;
    int n = e.generate(prompt, n_gen, tok.eos(), [&](int id) {
        tc.on_id(id);
        text += tok.decode_one(id);
        return true;
    });
    tc.end();
    e.on_round = nullptr;
    e.on_pending = nullptr;

    if (!out_path.empty()) {
        FILE* f = fopen(out_path.c_str(), "w");
        if (!f) { fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 1; }
        fwrite(text.data(), 1, text.size(), f);
        fclose(f);
    }
    printf("[e2e] emitted=%d engaged=%ld disengaged=%ld pool_drops=%ld rebinds=%ld "
           "refinish=%ld trunc=%ld masks=%zu\n",
           n, tc.engaged, tc.disengaged, tc.pool_drops, tc.rebinds, refinish, trunc,
           cache.size());
    return 0;
}
