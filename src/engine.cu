// q27 CLI: bench / greedy / spec-decode harness.
#include <chrono>
#include <tuple>
#include "engine.cuh"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.q27 --tokens \"1,2,3\" [-n N] [--ctx C] [--dump-logits f]\n",
                argv[0]);
        return 1;
    }
    std::string path = argv[1], dump;
    std::vector<int> toks;
    int n_gen = 8, ctx = 2048;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--tokens") && i + 1 < argc) {
            for (const char* p = argv[++i]; *p;) {
                toks.push_back(atoi(p));
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump = argv[++i];
    }
    bool mtp_stats = false, spec = false, fast = false;
    int pf_n = 0;
    bool pfcache = false;
    int stats_n = 0;
    int pfdbg_n = 0;
    std::string nll_path;
    int nll_chunk = 512, nll_long = 0, nll_max = 0, kvstats_n = 0;
    bool nll_serial = false, verify_weights = false;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--mtp")) mtp_stats = true;
        if (!strcmp(argv[i], "--spec")) { spec = true; mtp_stats = true; } // spec needs MTP warmup
        if (!strcmp(argv[i], "--fast-head")) fast = true;
        if (!strcmp(argv[i], "--pf") && i + 1 < argc) pf_n = atoi(argv[++i]);
        if (!strcmp(argv[i], "--pfcache")) pfcache = true;
        if (!strcmp(argv[i], "--stats") && i + 1 < argc) stats_n = atoi(argv[++i]);
        if (!strcmp(argv[i], "--pfdbg") && i + 1 < argc) pfdbg_n = atoi(argv[++i]);
        if (!strcmp(argv[i], "--nll") && i + 1 < argc) nll_path = argv[++i];
        if (!strcmp(argv[i], "--nll-chunk") && i + 1 < argc) nll_chunk = atoi(argv[++i]);
        if (!strcmp(argv[i], "--nll-long") && i + 1 < argc) nll_long = atoi(argv[++i]);
        if (!strcmp(argv[i], "--nll-max") && i + 1 < argc) nll_max = atoi(argv[++i]);
        if (!strcmp(argv[i], "--nll-serial")) nll_serial = true;
        if (!strcmp(argv[i], "--verify-weights")) verify_weights = true;
        if (!strcmp(argv[i], "--kvstats") && i + 1 < argc) kvstats_n = atoi(argv[++i]);
    }
    if (toks.empty() && nll_path.empty()) { fprintf(stderr, "need --tokens\n"); return 1; }

    Engine e(path, ctx);
    e.fast_head = fast;
    e.build_graph();
    if (spec) e.build_spec_graphs();

    if (stats_n > 0) {
        // E3 instrumentation: draft acceptance vs margin, rank-2 capture,
        // Q4-vs-Q8 draft-head agreement. Host-driven on the plain path.
        int N = stats_n;
        FILE* hid_dump = getenv("Q27_DUMP_HIDDENS") ? fopen(getenv("Q27_DUMP_HIDDENS"), "wb")
                                                    : nullptr;
        std::vector<float> l1(VOCAB), l2(VOCAB);
        float* d_l2;
        CUDA_CHECK(cudaMalloc((void**)&d_l2, (size_t)VOCAB * 4));
        auto top2 = [&](const std::vector<float>& l) {
            int b = 0; float bv = -1e30f, sv = -1e30f; int s = 0;
            for (int i = 0; i < VOCAB; i++) {
                if (l[i] > bv) { sv = bv; s = b; bv = l[i]; b = i; }
                else if (l[i] > sv) { sv = l[i]; s = i; }
            }
            return std::tuple<int, int, float>(b, s, bv - sv);
        };
        // generate ground truth + probe drafts as we go
        struct Pend { int pred = -1, pred_pos = -1; float margin = 0; int rank2 = -1; };
        std::vector<int> seq = toks;
        Pend p1, p2; // depth-1 and depth-2 pending predictions
        long n1 = 0, n1ok = 0, n2 = 0, n2ok = 0, r2cap = 0, r2tot = 0;
        long q48 = 0, q48ok = 0;
        // margin-binned depth-2 stats: bins <0.5, 0.5-1, 1-2, 2-4, >4
        long bn[5] = {0}, bok[5] = {0};
        auto bin = [](float m) { return m < 0.5f ? 0 : m < 1 ? 1 : m < 2 ? 2 : m < 4 ? 3 : 4; };
        std::vector<Pend> pend1(N + 8), pend2(N + 8);
        // E6 gate: depth-3 chain, binned by PASS-2 margin (the runtime gate
        // signal). d3 only matters when the d1,d2 prefix is accepted, so track
        // p(prefix ok | bin) and p(d3 ok | prefix ok, bin) separately.
        struct Pend3 { int pred = -1; float margin2 = 0; int d1 = -1, d2 = -1; };
        std::vector<Pend3> pend3(N + 8);
        long c3n[5] = {0}, c3pre[5] = {0}, c3ok[5] = {0};
        // P3 gate: depth-4 chain -- p(pass-4 draft | d1,d2,d3 all accepted).
        // Build depth-4 only if p(d4|prefix-3) holds ~>=60%.
        struct Pend4 { int pred = -1; float margin2 = 0; int d1 = -1, d2 = -1, d3 = -1; };
        std::vector<Pend4> pend4(N + 8);
        long c4n[5] = {0}, c4pre[5] = {0}, c4ok[5] = {0};
        // Depth-5 gate (roadmap #4, 2026-07-03): p(pass-5 draft | d1..d4 all
        // accepted). Chain barely decays through d4 (97.4%), so measure d5
        // before dismissing it -- projected +5-6%% net if the pattern holds.
        struct Pend5 { int pred = -1; float margin2 = 0; int d1 = -1, d2 = -1, d3 = -1, d4 = -1; };
        std::vector<Pend5> pend5(N + 8);
        long c5n[5] = {0}, c5pre[5] = {0}, c5ok[5] = {0};
        for (int step = 0; step < N + (int)toks.size() - 1; step++) {
            bool prompt_phase = step < (int)toks.size();
            int tok = prompt_phase ? toks[step] : -1;
            if (prompt_phase) e.step_with(tok);
            else e.step_free();
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            int next_tok;
            CUDA_CHECK(cudaMemcpy(&next_tok, e.d_token, 4, cudaMemcpyDeviceToHost));
            if (!prompt_phase) seq.push_back(next_tok);
            int pos = step; // hidden h(pos) just computed; next token = next_tok
            // score pending predictions targeting this newly known token
            int known_idx = (int)seq.size() - 1;
            for (auto* pd : {&pend1[0], &pend2[0]}) (void)pd;
            if (pend1[known_idx].pred >= 0) {
                n1++;
                bool ok = pend1[known_idx].pred == seq[known_idx];
                if (ok) n1ok++;
                else {
                    r2tot++;
                    if (pend1[known_idx].rank2 == seq[known_idx]) r2cap++;
                }
            }
            if (pend2[known_idx].pred >= 0) {
                n2++;
                bool ok = pend2[known_idx].pred == seq[known_idx];
                int b = bin(pend2[known_idx].margin);
                bn[b]++;
                if (ok) { n2ok++; bok[b]++; }
            }
            if (pend3[known_idx].pred >= 0 && known_idx >= 2) {
                const Pend3& p3 = pend3[known_idx];
                int b = bin(p3.margin2);
                c3n[b]++;
                if (p3.d1 == seq[known_idx - 2] && p3.d2 == seq[known_idx - 1]) {
                    c3pre[b]++;
                    if (p3.pred == seq[known_idx]) c3ok[b]++;
                }
            }
            if (pend4[known_idx].pred >= 0 && known_idx >= 3) {
                const Pend4& p4 = pend4[known_idx];
                int b = bin(p4.margin2);
                c4n[b]++;
                if (p4.d1 == seq[known_idx - 3] && p4.d2 == seq[known_idx - 2] &&
                    p4.d3 == seq[known_idx - 1]) {
                    c4pre[b]++;
                    if (p4.pred == seq[known_idx]) c4ok[b]++;
                }
            }
            if (pend5[known_idx].pred >= 0 && known_idx >= 4) {
                const Pend5& p5 = pend5[known_idx];
                int b = bin(p5.margin2);
                c5n[b]++;
                if (p5.d1 == seq[known_idx - 4] && p5.d2 == seq[known_idx - 3] &&
                    p5.d3 == seq[known_idx - 2] && p5.d4 == seq[known_idx - 1]) {
                    c5pre[b]++;
                    if (p5.pred == seq[known_idx]) c5ok[b]++;
                }
            }
            if (step >= N + (int)toks.size() - 2) break;
            // MTP pass 1: draft seq[known_idx+1] from (h(pos), next_tok)
            CUDA_CHECK(cudaMemcpyAsync(e.h_next, e.x1, N_EMBD * 4,
                                       cudaMemcpyDeviceToDevice, e.stm));
            int mp = pos + 1;
            CUDA_CHECK(cudaMemcpyAsync(e.d_pos_m, &mp, 4, cudaMemcpyHostToDevice, e.stm));
            e.mtp_forward();
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            CUDA_CHECK(cudaMemcpy(l1.data(), e.mtp_logits, (size_t)VOCAB * 4,
                                  cudaMemcpyDeviceToHost));
            auto [d1, d1b, m1] = top2(l1);
            if (hid_dump) { // E7 gate: dump the shared-head-norm hidden
                static std::vector<float> hbuf(N_EMBD);
                CUDA_CHECK(cudaMemcpy(hbuf.data(), e.x1, N_EMBD * 4, cudaMemcpyDeviceToHost));
                fwrite(hbuf.data(), 4, N_EMBD, hid_dump);
            }
            if (known_idx + 1 < (int)pend1.size()) {
                pend1[known_idx + 1] = {d1, pos + 2, m1, d1b};
            }
            // Q4 vs Q8 head agreement on the same hidden (xq still holds it)
            if (e.dm.model_has("output_q4.weight")) {
                e.mm(e.dm.get("output.weight"), e.x1, d_l2);
                CUDA_CHECK(cudaStreamSynchronize(e.stm));
                CUDA_CHECK(cudaMemcpy(l2.data(), d_l2, (size_t)VOCAB * 4,
                                      cudaMemcpyDeviceToHost));
                auto [q8b, q8s, q8m] = top2(l2);
                (void)q8s; (void)q8m;
                q48++;
                if (q8b == d1) q48ok++;
            }
            // MTP pass 2: chain from pass-1 hidden, draft seq[known_idx+2]
            int mp2 = pos + 2;
            CUDA_CHECK(cudaMemcpyAsync(e.d_token, &d1, 4, cudaMemcpyHostToDevice, e.stm));
            CUDA_CHECK(cudaMemcpyAsync(e.d_pos_m, &mp2, 4, cudaMemcpyHostToDevice, e.stm));
            e.mtp_forward(e.x1, nullptr, nullptr, nullptr);
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            CUDA_CHECK(cudaMemcpy(l1.data(), e.mtp_logits, (size_t)VOCAB * 4,
                                  cudaMemcpyDeviceToHost));
            auto [d2, d2b, m2] = top2(l1);
            (void)d2b;
            if (known_idx + 2 < (int)pend2.size()) pend2[known_idx + 2] = {d2, pos + 3, m2, -1};
            // MTP pass 3: chain from pass-2 hidden, draft seq[known_idx+3];
            // keyed by the PASS-2 margin (the E6 runtime gate signal)
            int mp3 = pos + 3;
            CUDA_CHECK(cudaMemcpyAsync(e.d_token, &d2, 4, cudaMemcpyHostToDevice, e.stm));
            CUDA_CHECK(cudaMemcpyAsync(e.d_pos_m, &mp3, 4, cudaMemcpyHostToDevice, e.stm));
            e.mtp_forward(e.x1, nullptr, nullptr, nullptr);
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            CUDA_CHECK(cudaMemcpy(l1.data(), e.mtp_logits, (size_t)VOCAB * 4,
                                  cudaMemcpyDeviceToHost));
            auto [d3, d3b, m3] = top2(l1);
            (void)d3b; (void)m3;
            if (known_idx + 3 < (int)pend3.size()) pend3[known_idx + 3] = {d3, m2, d1, d2};
            // MTP pass 4: chain from pass-3 hidden, draft seq[known_idx+4]
            // (P3 depth-4 gate measurement; binned by pass-2 margin like E6)
            int mp4 = pos + 4;
            CUDA_CHECK(cudaMemcpyAsync(e.d_token, &d3, 4, cudaMemcpyHostToDevice, e.stm));
            CUDA_CHECK(cudaMemcpyAsync(e.d_pos_m, &mp4, 4, cudaMemcpyHostToDevice, e.stm));
            e.mtp_forward(e.x1, nullptr, nullptr, nullptr);
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            CUDA_CHECK(cudaMemcpy(l1.data(), e.mtp_logits, (size_t)VOCAB * 4,
                                  cudaMemcpyDeviceToHost));
            auto [d4, d4b, m4] = top2(l1);
            (void)d4b; (void)m4;
            if (known_idx + 4 < (int)pend4.size()) pend4[known_idx + 4] = {d4, m2, d1, d2, d3};
            // MTP pass 5: chain from pass-4 hidden, draft seq[known_idx+5]
            int mp5 = pos + 5;
            CUDA_CHECK(cudaMemcpyAsync(e.d_token, &d4, 4, cudaMemcpyHostToDevice, e.stm));
            CUDA_CHECK(cudaMemcpyAsync(e.d_pos_m, &mp5, 4, cudaMemcpyHostToDevice, e.stm));
            e.mtp_forward(e.x1, nullptr, nullptr, nullptr);
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            CUDA_CHECK(cudaMemcpy(l1.data(), e.mtp_logits, (size_t)VOCAB * 4,
                                  cudaMemcpyDeviceToHost));
            auto [d5, d5b, m5] = top2(l1);
            (void)d5b; (void)m5;
            if (known_idx + 5 < (int)pend5.size()) pend5[known_idx + 5] = {d5, m2, d1, d2, d3, d4};
            // restore d_token for the next step_free
            CUDA_CHECK(cudaMemcpy(e.d_token, &next_tok, 4, cudaMemcpyHostToDevice));
        }
        if (hid_dump) fclose(hid_dump);
        printf("\nE3 stats over %ld draft-1 / %ld draft-2 evaluations:\n", n1, n2);
        printf("  p(draft1)          = %.1f%%\n", 100.0 * n1ok / (n1 ? n1 : 1));
        printf("  p(draft2 chained)  = %.1f%%\n", 100.0 * n2ok / (n2 ? n2 : 1));
        printf("  rank-2 capture     = %.1f%% of %ld rejections (tree gate: need >8pt)\n",
               100.0 * r2cap / (r2tot ? r2tot : 1), r2tot);
        printf("  Q4-vs-Q8 head agree= %.1f%% of %ld (E7 gate: need >=97%%)\n",
               100.0 * q48ok / (q48 ? q48 : 1), q48);
        printf("  draft-2 acceptance by pass-2 margin (E6 gate: high-margin bin >=70%%):\n");
        const char* bl[5] = {"<0.5", "0.5-1", "1-2", "2-4", ">4"};
        for (int b = 0; b < 5; b++)
            printf("    margin %-5s: %5.1f%% (n=%ld)\n", bl[b],
                   100.0 * bok[b] / (bn[b] ? bn[b] : 1), bn[b]);
        printf("  depth-3 chain by pass-2 margin (E6 gate):\n");
        for (int b = 0; b < 5; b++)
            printf("    margin %-5s: n=%4ld  p(prefix)=%5.1f%%  p(d3|prefix)=%5.1f%%\n", bl[b],
                   c3n[b], 100.0 * c3pre[b] / (c3n[b] ? c3n[b] : 1),
                   100.0 * c3ok[b] / (c3pre[b] ? c3pre[b] : 1));
        printf("  depth-4 chain by pass-2 margin (P3 gate: p(d4|prefix) >= ~60%%):\n");
        long t4n = 0, t4pre = 0, t4ok = 0;
        for (int b = 0; b < 5; b++) {
            t4n += c4n[b]; t4pre += c4pre[b]; t4ok += c4ok[b];
            printf("    margin %-5s: n=%4ld  p(prefix3)=%5.1f%%  p(d4|prefix3)=%5.1f%%\n",
                   bl[b], c4n[b], 100.0 * c4pre[b] / (c4n[b] ? c4n[b] : 1),
                   100.0 * c4ok[b] / (c4pre[b] ? c4pre[b] : 1));
        }
        printf("  depth-4 OVERALL: p(prefix3)=%.1f%%  p(d4|prefix3)=%.1f%%  "
               "extra t/round ~= +%.3f (ungated)\n",
               100.0 * t4pre / (t4n ? t4n : 1), 100.0 * t4ok / (t4pre ? t4pre : 1),
               (double)t4ok / (t4n ? t4n : 1));
        printf("  depth-5 chain by pass-2 margin (gate: p(d5|prefix4) decides depth-5):\n");
        long t5n = 0, t5pre = 0, t5ok = 0;
        for (int b = 0; b < 5; b++) {
            t5n += c5n[b]; t5pre += c5pre[b]; t5ok += c5ok[b];
            printf("    margin %-5s: n=%4ld  p(prefix4)=%5.1f%%  p(d5|prefix4)=%5.1f%%\n",
                   bl[b], c5n[b], 100.0 * c5pre[b] / (c5n[b] ? c5n[b] : 1),
                   100.0 * c5ok[b] / (c5pre[b] ? c5pre[b] : 1));
        }
        printf("  depth-5 OVERALL: p(prefix4)=%.1f%%  p(d5|prefix4)=%.1f%%  "
               "extra t/round ~= +%.3f (ungated; net = this minus ~7%%/depth round tax)\n",
               100.0 * t5pre / (t5n ? t5n : 1), 100.0 * t5ok / (t5pre ? t5pre : 1),
               (double)t5ok / (t5n ? t5n : 1));
        // cumulative gate margin>=theta; extra tokens/round ~= f * p(prefix|gate)
        // * p(d3|prefix,gate). Caveat: per-POSITION sampling; live rounds sample
        // accepted spans, so f here slightly understates the gated fraction.
        printf("  E6 projection (gate = pass-2 margin >= theta):\n");
        const char* tl[5] = {"0 (ungated)", "0.5", "1", "2", "4"};
        long t3n = 0;
        for (int b = 0; b < 5; b++) t3n += c3n[b];
        for (int t = 0; t < 5; t++) {
            long n = 0, pre = 0, ok = 0;
            for (int b = t; b < 5; b++) { n += c3n[b]; pre += c3pre[b]; ok += c3ok[b]; }
            double f = (double)n / (t3n ? t3n : 1);
            double ppre = (double)pre / (n ? n : 1);
            double pd3 = (double)ok / (pre ? pre : 1);
            printf("    theta %-11s: f=%4.1f%%  p(prefix|gate)=%5.1f%%  "
                   "p(d3|prefix,gate)=%5.1f%%  extra t/round=+%.3f\n",
                   tl[t], 100 * f, 100 * ppre, 100 * pd3, f * ppre * pd3);
        }
        return 0;
    }

    if (kvstats_n > 0) {
        // P2 design probe: prefill real text at fp16, then scan the attention
        // KV caches for value magnitudes. Decides scale-free E4M3 vs per-row
        // scales (E4M3: max 448, min denormal 2^-9).
        if (nll_path.empty()) { fprintf(stderr, "--kvstats needs --nll FILE for tokens\n"); return 1; }
        FILE* f = fopen(nll_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", nll_path.c_str()); return 1; }
        fseek(f, 0, SEEK_END);
        long fb = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<int> tk(fb / 4);
        if (fread(tk.data(), 4, tk.size(), f) != tk.size()) { fclose(f); return 1; }
        fclose(f);
        int N = std::min((int)tk.size(), std::min(kvstats_n, ctx));
        fprintf(stderr, "kvstats: prefilling %d tokens\n", N);
        int* d_toks;
        CUDA_CHECK(cudaMalloc((void**)&d_toks, (size_t)N * 4));
        CUDA_CHECK(cudaMemcpy(d_toks, tk.data(), (size_t)N * 4, cudaMemcpyHostToDevice));
        e.reset();
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        auto pf_t0 = std::chrono::steady_clock::now();
        const DevTensor& onw = e.dm.get("output_norm.weight");
        for (int c0 = 0; c0 < N; c0 += Engine::PF_T) {
            int T = std::min((int)Engine::PF_T, N - c0);
            e.prefill_chunk(d_toks + c0, c0, T);
            if (c0 + T < N) { // mtp_warm reads toks[c0+1 .. c0+T]
                q27k::rmsnorm_T(e.hT, (const float*)onw.data, e.x1T, N_EMBD, T, EPS, e.stm);
                e.mtp_warm_T(d_toks + c0 + 1, c0, T);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        double pf_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - pf_t0)
                          .count();
        fprintf(stderr, "kvstats prefill: %d tokens in %.2fs (%.1f t/s)\n", N, pf_s, N / pf_s);
        size_t n = (size_t)N * N_KV * HEAD_DIM;
        std::vector<__half> hb(n);
        auto scan = [&](const char* tag, int layer, const __half* dev) {
            CUDA_CHECK(cudaMemcpy(hb.data(), dev, n * 2, cudaMemcpyDeviceToHost));
            double amax = 0, asum = 0;
            long sat = 0, sub = 0; // |x| > 448 (E4M3 sat) / 0 < |x| < 2^-10 (flush to 0)
            for (size_t i = 0; i < n; i++) {
                double a = fabs((double)__half2float(hb[i]));
                asum += a;
                if (a > amax) amax = a;
                if (a > 448.0) sat++;
                if (a > 0 && a < 0.0009765625) sub++;
            }
            printf("%s L%-2d  amax %10.3f  mean|x| %8.4f  sat448 %ld (%.4f%%)  "
                   "sub2^-10 %ld (%.2f%%)\n", tag, layer, amax, asum / n, sat,
                   100.0 * sat / n, sub, 100.0 * sub / n);
        };
        if (e.kv_fp8) { fprintf(stderr, "--kvstats reads fp16 caches; unset Q27_KV\n"); return 1; }
        for (size_t s = 0; s < e.kcache.size(); s++) {
            int layer = -1;
            for (int il = 0; il < N_LAYER; il++)
                if (e.attn_cache_idx[il] == (int)s) layer = il;
            scan("K", layer, (const __half*)e.kcache[s]);
            scan("V", layer, (const __half*)e.vcache[s]);
        }
        scan("K", 64, (const __half*)e.mtp_k);
        scan("V", 64, (const __half*)e.mtp_v);
        return 0;
    }

    if (!nll_path.empty()) {
        // P0 quality gate: teacher-forced NLL. Chunked mode replicates the
        // llama-perplexity protocol exactly (independent chunks of C tokens,
        // logits rows [C/2, C-2] predict targets [C/2+1, C-1], leftover
        // tokens dropped, no BOS -- this model has add_bos=false), so the
        // resulting PPL is directly comparable to `llama-perplexity -c C` on
        // the same token stream. --nll-long: one pass, no resets, NLL
        // bucketed by position (long-context degradation gate).
        FILE* f = fopen(nll_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", nll_path.c_str()); return 1; }
        fseek(f, 0, SEEK_END);
        long fb = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<int> tk(fb / 4);
        if (fread(tk.data(), 4, tk.size(), f) != tk.size()) {
            fprintf(stderr, "short read on %s\n", nll_path.c_str());
            return 1;
        }
        fclose(f);
        const DevTensor& onw = e.dm.get("output_norm.weight");
        const DevTensor& head = e.dm.get("output.weight");
        const int PT = Engine::PF_T;

        if (nll_long > 0) {
            int N = std::min((int)tk.size(), std::min(nll_long, ctx));
            fprintf(stderr, "nll-long: %d tokens, single pass, no resets\n", N);
            int* d_toks;
            float *d_lg, *d_nll;
            CUDA_CHECK(cudaMalloc((void**)&d_toks, (size_t)N * 4));
            CUDA_CHECK(cudaMalloc((void**)&d_lg, (size_t)PT * VOCAB * 4));
            CUDA_CHECK(cudaMalloc((void**)&d_nll, PT * 4));
            CUDA_CHECK(cudaMemcpy(d_toks, tk.data(), (size_t)N * 4, cudaMemcpyHostToDevice));
            e.reset();
            const int NB = 14;
            const int bl[NB + 1] = {0,      2048,   8192,   16384,  32768,
                                    49152,  65536,  98304,  131072, 163840,
                                    196608, 229376, 262144, 327680, 1 << 30};
            const char* bn[NB] = {"0-2k",      "2k-8k",     "8k-16k",    "16k-32k",
                                  "32k-48k",   "48k-64k",   "64k-96k",   "96k-128k",
                                  "128k-160k", "160k-192k", "192k-224k", "224k-256k",
                                  "256k-320k", "320k+"};
            double bs[NB] = {0};
            long bc[NB] = {0};
            std::vector<float> h_nll(PT);
            for (int c0 = 0; c0 < N - 1; c0 += PT) {
                int T = std::min(PT, N - c0);
                e.prefill_chunk(d_toks + c0, c0, T);
                q27k::rmsnorm_T(e.hT, (const float*)onw.data, e.x1T, N_EMBD, T, EPS, e.stm);
                e.qxT(e.x1T, N_EMBD, T);
                e.mmT(head, e.x1T, d_lg, T);
                int nrows = std::min(T, N - 1 - c0);
                q27k::nll_rows(d_lg, d_toks + c0 + 1, d_nll, nrows, VOCAB, e.stm);
                CUDA_CHECK(cudaMemcpyAsync(h_nll.data(), d_nll, (size_t)nrows * 4,
                                           cudaMemcpyDeviceToHost, e.stm));
                CUDA_CHECK(cudaStreamSynchronize(e.stm));
                for (int r = 0; r < nrows; r++) {
                    int tpos = c0 + r + 1;
                    int b = 0;
                    while (tpos >= bl[b + 1]) b++;
                    bs[b] += h_nll[r];
                    bc[b]++;
                }
                if ((c0 / PT) % 32 == 0) fprintf(stderr, "  pos %d/%d\r", c0, N);
            }
            printf("\nlong-context NLL by target position (%d tokens, no resets):\n", N);
            for (int b = 0; b < NB; b++)
                if (bc[b])
                    printf("  %-8s: mean NLL %.4f  PPL %8.3f  (n=%ld)\n", bn[b], bs[b] / bc[b],
                           exp(bs[b] / bc[b]), bc[b]);
            return 0;
        }

        const int C = nll_chunk, first = C / 2;
        int nchunks = (int)tk.size() / C;
        if (nll_max > 0) nchunks = std::min(nchunks, nll_max);
        fprintf(stderr, "nll: %zu tokens, %d chunks of %d, count rows [%d, %d] (%s)\n",
                tk.size(), nchunks, C, first, C - 2, nll_serial ? "serial" : "batched");
        double sum = 0;
        long count = 0;
        if (nll_serial) {
            std::vector<float> lg(VOCAB);
            for (int ch = 0; ch < nchunks; ch++) {
                const int* ck = tk.data() + (size_t)ch * C;
                e.reset();
                for (int i = 0; i < C; i++) {
                    e.step_with(ck[i]);
                    if (i >= first && i <= C - 2) {
                        CUDA_CHECK(cudaStreamSynchronize(e.stm));
                        CUDA_CHECK(cudaMemcpy(lg.data(), e.logits, (size_t)VOCAB * 4,
                                              cudaMemcpyDeviceToHost));
                        double mx = -1e30;
                        for (int v = 0; v < VOCAB; v++) mx = std::max(mx, (double)lg[v]);
                        double se = 0;
                        for (int v = 0; v < VOCAB; v++) se += exp((double)lg[v] - mx);
                        sum += log(se) + mx - (double)lg[ck[i + 1]];
                        count++;
                    }
                }
                CUDA_CHECK(cudaStreamSynchronize(e.stm));
            }
        } else {
            int* d_toks;
            float *d_lg, *d_nll;
            CUDA_CHECK(cudaMalloc((void**)&d_toks, (size_t)C * 4));
            CUDA_CHECK(cudaMalloc((void**)&d_lg, (size_t)PT * VOCAB * 4));
            CUDA_CHECK(cudaMalloc((void**)&d_nll, PT * 4));
            std::vector<float> h_nll(PT);
            for (int ch = 0; ch < nchunks; ch++) {
                e.reset();
                CUDA_CHECK(cudaMemcpyAsync(d_toks, tk.data() + (size_t)ch * C, (size_t)C * 4,
                                           cudaMemcpyHostToDevice, e.stm));
                for (int c0 = 0; c0 < C; c0 += PT) {
                    int T = std::min(PT, C - c0);
                    e.prefill_chunk(d_toks + c0, c0, T);
                    q27k::rmsnorm_T(e.hT, (const float*)onw.data, e.x1T, N_EMBD, T, EPS, e.stm);
                    e.qxT(e.x1T, N_EMBD, T);
                    e.mmT(head, e.x1T, d_lg, T);
                    int nrows = std::min(T, C - 1 - c0);
                    if (nrows <= 0) continue;
                    q27k::nll_rows(d_lg, d_toks + c0 + 1, d_nll, nrows, VOCAB, e.stm);
                    CUDA_CHECK(cudaMemcpyAsync(h_nll.data(), d_nll, (size_t)nrows * 4,
                                               cudaMemcpyDeviceToHost, e.stm));
                    CUDA_CHECK(cudaStreamSynchronize(e.stm));
                    for (int r = 0; r < nrows; r++) {
                        int row = c0 + r;
                        if (row >= first && row <= C - 2) {
                            sum += h_nll[r];
                            count++;
                        }
                    }
                }
                if (ch % 32 == 0 && count)
                    fprintf(stderr, "  [%d/%d] running PPL %.4f\r", ch, nchunks,
                            exp(sum / count));
            }
        }
        printf("\nnll: %ld predictions over %d chunks: mean NLL %.6f, PPL %.4f\n", count,
               nchunks, sum / count, exp(sum / count));
        return 0;
    }

    if (pfcache) {
        // M6.5/P8 gate: turn 2 re-renders history, so the prompt TAIL
        // diverges (assistant-open/prefill replaced by rendered content).
        // The stable-prefix snapshot must still hit at the boundary and
        // produce continuations identical to a cold run. (The old gate
        // appended raw tokens -- a flow no re-rendering client takes -- and
        // hid a 100% cache-miss bug in real serving.)
        if (!spec) e.build_spec_graphs();
        e.ckpt_interval = 128; // dense ring so the mid-divergence leg has cover
        std::vector<int> A, B;
        for (int i = 0; i < 600; i++) A.push_back(toks[i % toks.size()]);
        const int SBL = 585; // turn-1 stable boundary
        B.assign(A.begin(), A.begin() + SBL);
        for (int i = 0; i < 25; i++) B.push_back(toks[(i * 7 + 5) % toks.size()]); // divergent tail
        for (int i = 0; i < 40; i++) B.push_back(toks[(i + 11) % toks.size()]);    // new content
        auto timed = [&](const std::vector<int>& p, std::vector<int>& out, int sbl) {
            auto t0 = std::chrono::steady_clock::now();
            double ttft = 0;
            bool first = true;
            e.generate(p, 16, -1, [&](int id) {
                if (first) {
                    ttft = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                               .count();
                    first = false;
                }
                out.push_back(id);
                return true;
            }, sbl);
            return ttft;
        };
        std::vector<int> o1, warm, cold;
        double t1 = timed(A, o1, SBL);                    // turn 1 (cold, snapshot at SBL)
        double tw = timed(B, warm, (int)B.size() - 8);    // turn 2 (tail-divergent resume)
        e.have_snap = false;
        e.ckpt_clear(); // truly cold: no checkpoint assistance either
        double tc = timed(B, cold, (int)B.size() - 8);    // turn 2 cold rerun
        printf("turn1 TTFT %.3fs | turn2 warm TTFT %.3fs | turn2 cold TTFT %.3fs "
               "(warm speedup %.1fx)\n", t1, tw, tc, tc / tw);
        printf("warm vs cold continuations: %s\n",
               warm == cold ? "IDENTICAL -- gate PASS" : "MISMATCH -- gate FAIL");
        if (warm != cold) {
            printf("cold: "); for (int t : cold) printf("%d ", t);
            printf("\nwarm: "); for (int t : warm) printf("%d ", t);
            printf("\n");
            return 1;
        }
        // P9 gate: MID-history divergence (compaction/edit/retry). Turn 3
        // keeps only the first 300 tokens of B, replaces the middle, and
        // extends. The stable snapshot (at B.size()-8) is useless; recovery
        // must come from a GDN checkpoint <= the divergence point, and
        // continuations must match a cold run. Requires checkpoints enabled
        // (Q27_CKPT_INTERVAL=128 set by the harness for a dense ring).
        std::vector<int> C(B.begin(), B.begin() + 300);
        for (int i = 0; i < 200; i++) C.push_back(toks[(i * 5 + 2) % toks.size()]);
        std::vector<int> warm3, cold3;
        double tw3 = timed(C, warm3, (int)C.size() - 8);  // checkpoint-assisted
        e.have_snap = false;
        e.ckpt_clear();
        double tc3 = timed(C, cold3, (int)C.size() - 8);  // cold rerun
        printf("turn3 (mid-divergence) warm TTFT %.3fs | cold TTFT %.3fs | ckpt base used: "
               "see [gen] log\n", tw3, tc3);
        printf("mid-divergence continuations: %s\n",
               warm3 == cold3 ? "IDENTICAL -- gate PASS" : "MISMATCH -- gate FAIL");
        if (warm3 != cold3) return 1;
        return 0;
    }

    if (pfdbg_n == 1) { // --pfdbg 1: stage diff, layer 0, token 0
        int tok0 = toks[0];
        auto grab = [&](const void* dev, size_t n) {
            std::vector<float> h_(n);
            CUDA_CHECK(cudaMemcpy(h_.data(), dev, n * 4, cudaMemcpyDeviceToHost));
            return h_;
        };
        auto md = [](const char* name, const std::vector<float>& a, const std::vector<float>& b) {
            float m = 0; size_t at = 0;
            for (size_t i = 0; i < a.size(); i++) {
                float d = fabsf(a[i] - b[i]);
                if (d > m) { m = d; at = i; }
            }
            printf("%-8s maxdiff %.6g at %zu (a=%.6g b=%.6g)\n", name, m, at,
                   m > 0 ? a[at] : 0.f, m > 0 ? b[at] : 0.f);
        };
        // serial layer 0
        e.reset();
        const DevTensor& emb = e.dm.get("token_embd.weight");
        CUDA_CHECK(cudaMemcpy(e.d_token, &tok0, 4, cudaMemcpyHostToDevice));
        q27k::embed_row_q8((const int8_t*)emb.data, (const __half*)emb.scales, e.d_token,
                           N_EMBD, e.h, e.stm);
        q27k::rmsnorm(e.h, (const float*)e.T(0, "attn_norm.weight").data, e.x1, N_EMBD, EPS,
                      e.stm);
        e.gdn_block(0, e.x1, e.y);
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        auto s_x1 = grab(e.x1, N_EMBD), s_qkv = grab(e.qkv, GDN_CH), s_z = grab(e.z, GDN_V);
        auto s_al = grab(e.alpha, GDN_HEADS), s_g = grab(e.g, GDN_HEADS),
             s_be = grab(e.beta, GDN_HEADS);
        auto s_co = grab(e.convout, GDN_CH), s_o = grab(e.o, GDN_V), s_og = grab(e.og, GDN_V);
        auto s_y = grab(e.y, N_EMBD), s_S = grab(e.S[0], 48 * 128 * 128);
        // batched layer 0, T=1
        e.reset();
        CUDA_CHECK(cudaMemcpy(e.d_token, &tok0, 4, cudaMemcpyHostToDevice));
        q27k::embed_rows_q8_T((const int8_t*)emb.data, (const __half*)emb.scales, e.d_token,
                              N_EMBD, 1, e.hT, e.stm);
        q27k::rmsnorm_T(e.hT, (const float*)e.T(0, "attn_norm.weight").data, e.x1T, N_EMBD, 1,
                        EPS, e.stm);
        e.gdn_block_T(0, 1);
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        md("x1", s_x1, grab(e.x1T, N_EMBD));
        md("qkv", s_qkv, grab(e.qkvT, GDN_CH));
        md("z", s_z, grab(e.zT, GDN_V));
        md("alpha", s_al, grab(e.alphaT, GDN_HEADS));
        md("g", s_g, grab(e.gT, GDN_HEADS));
        md("beta", s_be, grab(e.betaT, GDN_HEADS));
        md("convout", s_co, grab(e.convT, GDN_CH));
        md("o", s_o, grab(e.oT, GDN_V));
        md("og", s_og, grab(e.ogT, GDN_V));
        md("y", s_y, grab(e.yT, N_EMBD));
        md("S0", s_S, grab(e.S[0], 48 * 128 * 128));
        return 0;
    }

    if (pfdbg_n > 0) {
        // state diff: serial vs batched prefill of the SAME N-1 tokens
        int N = pfdbg_n;
        std::vector<int> prompt;
        for (int i = 0; i < N; i++) prompt.push_back(toks[i % toks.size()]);
        int T = N - 1;
        auto grab = [&](const void* dev, size_t bytes) {
            std::vector<float> h_(bytes / 4);
            CUDA_CHECK(cudaMemcpy(h_.data(), dev, bytes, cudaMemcpyDeviceToHost));
            return h_;
        };
        auto maxdiff = [](const std::vector<float>& a, const std::vector<float>& b) {
            float m = 0;
            size_t at = 0;
            for (size_t i = 0; i < a.size(); i++) {
                float d = fabsf(a[i] - b[i]);
                if (d > m) { m = d; at = i; }
            }
            printf(" maxdiff %.6g at %zu", m, at);
            return m;
        };
        // pass 1: serial
        e.reset();
        for (int i = 0; i < T; i++) {
            e.step_with(prompt[i]);
            if (i + 1 < T) {
                CUDA_CHECK(cudaStreamSynchronize(e.stm));
                CUDA_CHECK(cudaMemcpyAsync(e.h_next, e.x1, N_EMBD * 4,
                                           cudaMemcpyDeviceToDevice, e.stm));
                int nt = prompt[i + 1], mp = i + 1;
                CUDA_CHECK(cudaMemcpyAsync(e.d_token, &nt, 4, cudaMemcpyHostToDevice, e.stm));
                CUDA_CHECK(cudaMemcpyAsync(e.d_pos_m, &mp, 4, cudaMemcpyHostToDevice, e.stm));
                e.mtp_forward();
                CUDA_CHECK(cudaStreamSynchronize(e.stm));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        auto s_h = grab(e.h, N_EMBD * 4);
        auto s_S0 = grab(e.S[0], 48 * 128 * 128 * 4);
        auto s_S62 = grab(e.S[62], 48 * 128 * 128 * 4);
        auto s_ring0 = grab(e.conv_ring[0], 3 * GDN_CH * 4);
        auto grabh = [&](const __half* dev, size_t n) {
            std::vector<__half> tmp(n);
            CUDA_CHECK(cudaMemcpy(tmp.data(), dev, n * 2, cudaMemcpyDeviceToHost));
            std::vector<float> out(n);
            for (size_t i = 0; i < n; i++) out[i] = __half2float(tmp[i]);
            return out;
        };
        auto s_kc = grabh((const __half*)e.kcache[0], (size_t)T * N_KV * HEAD_DIM);
        auto s_mk = grabh((const __half*)e.mtp_k, (size_t)(T) * N_KV * HEAD_DIM);
        // pass 2: batched (chunked prefill only, no final serial token)
        e.reset();
        if (e.d_prompt_cap < N) {
            if (e.d_prompt) cudaFree(e.d_prompt);
            CUDA_CHECK(cudaMalloc((void**)&e.d_prompt, (size_t)N * 4));
            e.d_prompt_cap = N;
        }
        CUDA_CHECK(cudaMemcpy(e.d_prompt, prompt.data(), (size_t)N * 4, cudaMemcpyHostToDevice));
        for (int c0 = 0; c0 < T; c0 += Engine::PF_T) {
            int Tc = std::min((int)Engine::PF_T, T - c0);
            e.prefill_chunk(e.d_prompt + c0, c0, Tc);
            q27k::rmsnorm_T(e.hT, (const float*)e.dm.get("output_norm.weight").data, e.x1T,
                            N_EMBD, Tc, EPS, e.stm);
            e.mtp_warm_T(e.d_prompt + c0 + 1, c0, Tc);
        }
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        int lastrow = (T - 1) % Engine::PF_T;
        auto b_h = grab(e.hT + (size_t)lastrow * N_EMBD, N_EMBD * 4);
        auto b_S0 = grab(e.S[0], 48 * 128 * 128 * 4);
        auto b_S62 = grab(e.S[62], 48 * 128 * 128 * 4);
        auto b_ring0 = grab(e.conv_ring[0], 3 * GDN_CH * 4);
        auto b_kc = grabh((const __half*)e.kcache[0], (size_t)T * N_KV * HEAD_DIM);
        auto b_mk = grabh((const __half*)e.mtp_k, (size_t)(T) * N_KV * HEAD_DIM);
        printf("h(last):"); maxdiff(s_h, b_h); printf("\n");
        printf("S[0]   :"); maxdiff(s_S0, b_S0); printf("\n");
        printf("S[62]  :"); maxdiff(s_S62, b_S62); printf("\n");
        printf("ring[0]:"); maxdiff(s_ring0, b_ring0); printf("\n");
        printf("kcache :"); maxdiff(s_kc, b_kc); printf("\n");
        printf("mtp_k  :"); maxdiff(s_mk, b_mk); printf("\n");
        return 0;
    }

    if (pf_n > 0) {
        // M6 gate: identical generation after serial vs batched prefill + prefill t/s
        if (!spec) e.build_spec_graphs();
        std::vector<int> prompt;
        for (int i = 0; i < pf_n; i++) prompt.push_back(toks[i % toks.size()]);
        auto run = [&](bool batched, std::vector<int>& out) {
            e.batched_prefill = batched;
            auto t0 = std::chrono::steady_clock::now();
            double ttft = 0;
            bool first = true;
            e.generate(prompt, 32, -1, [&](int id) {
                if (first) {
                    ttft = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                               .count();
                    first = false;
                }
                out.push_back(id);
                return true;
            });
            return ttft;
        };
        std::vector<int> a, b;
        // Q27_PF_NOSERIAL=1 skips the serial leg (and the identity gate) for
        // fast batched-rate iteration; the gate still runs by default.
        const bool noserial = getenv("Q27_PF_NOSERIAL") != nullptr;
        double ts = noserial ? 0 : run(false, a);
        double tb = run(true, b);
        if (noserial) {
            printf("prefill %d tokens: batched TTFT %.3fs (%.1f t/s) [serial skipped]\n", pf_n,
                   tb, pf_n / tb);
            return 0;
        }
        printf("prefill %d tokens: serial TTFT %.2fs (%.1f t/s) | batched TTFT %.3fs "
               "(%.1f t/s) | speedup %.1fx\n",
               pf_n, ts, pf_n / ts, tb, pf_n / tb, ts / tb);
        printf("continuations %s (%zu vs %zu tokens)\n",
               a == b ? "IDENTICAL -- gate PASS" : "MISMATCH -- gate FAIL", a.size(), b.size());
        if (a != b) {
            printf("serial : ");
            for (size_t i = 0; i < a.size() && i < 16; i++) printf("%d ", a[i]);
            printf("\nbatched: ");
            for (size_t i = 0; i < b.size() && i < 16; i++) printf("%d ", b[i]);
            printf("\n");
            return 1;
        }
        return 0;
    }

    // prompt (with MTP KV warmup when measuring acceptance: pair h(i) with the
    // ACTUAL next prompt token at position i+1, mirroring llama.cpp prefill)
    for (size_t i = 0; i < toks.size(); i++) {
        e.step_with(toks[i]);
        if (mtp_stats && i + 1 < toks.size()) {
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            CUDA_CHECK(cudaMemcpyAsync(e.h_next, e.x1, N_EMBD * 4, cudaMemcpyDeviceToDevice,
                                       e.stm));
            int next_tok = toks[i + 1], mpos = (int)i + 1;
            CUDA_CHECK(cudaMemcpyAsync(e.d_token, &next_tok, 4, cudaMemcpyHostToDevice, e.stm));
            CUDA_CHECK(cudaMemcpyAsync(e.d_pos_m, &mpos, 4, cudaMemcpyHostToDevice, e.stm));
            e.mtp_forward();
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            // restore d_token for the next main step (it was clobbered)
            // (step_with overwrites d_token anyway; nothing to restore)
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(e.stm));

    if (!dump.empty()) {
        std::vector<float> lg(VOCAB);
        CUDA_CHECK(cudaMemcpy(lg.data(), e.logits, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost));
        FILE* f = fopen(dump.c_str(), "wb");
        fwrite(lg.data(), 4, VOCAB, f);
        fclose(f);
        fprintf(stderr, "logits -> %s\n", dump.c_str());
    }

    // generation: device-chained, zero host round-trips
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    int accepted = 0, drafted = 0;
    CUDA_CHECK(cudaEventRecord(t0, e.stm));
    if (spec) {
        CUDA_CHECK(cudaMemcpyAsync(e.h_next, e.x1, N_EMBD * 4, cudaMemcpyDeviceToDevice, e.stm));
        std::vector<int> out;
        int P = (int)toks.size() - 1;
        CUDA_CHECK(cudaMemcpyAsync(e.d_P, &P, 4, cudaMemcpyHostToDevice, e.stm));
        int total_emitted = 0, rounds = 0, hist[5] = {0, 0, 0, 0, 0};
        while ((int)out.size() < n_gen) {
            if (P + 6 > ctx) { fprintf(stderr, "ctx-guard: stopping at P=%d\n", P); break; }
            int em[5];
            int n = e.spec_round(em);
            for (int k = 0; k < n; k++) out.push_back(em[k]);
            rounds++;
            total_emitted += n;
            hist[n - 1]++;
            P += n;
        }
        fprintf(stderr, "round outcomes: 1-tok %d, 2-tok %d, 3-tok %d, 4-tok %d, 5-tok %d\n",
                hist[0], hist[1], hist[2], hist[3], hist[4]);
        drafted = rounds;
        accepted = total_emitted; // repurposed: tokens per round stats
        CUDA_CHECK(cudaEventRecord(t1, e.stm));
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        float msf = 0;
        CUDA_CHECK(cudaEventElapsedTime(&msf, t0, t1));
        printf("generated:");
        for (int i = 0; i < n_gen && i < (int)out.size(); i++) printf(" %d", out[i]);
        printf("\nspec decode: %d tokens in %.1f ms = %.2f t/s (%.2f tokens/round over %d rounds)\n",
               (int)out.size(), msf, out.size() * 1000.0f / msf,
               (double)accepted / drafted, drafted);
        if (verify_weights && e.dm.checksum_verify(true)) return 2;
        return 0;
    }
    if (mtp_stats) {
        // At loop entry: d_token = main's prediction for the next position,
        // x1 = h of the last processed position (= toks.size()-1 + i).
        int hpos = (int)toks.size() - 1;
        for (int i = 0; i < n_gen; i++) {
            CUDA_CHECK(cudaMemcpyAsync(e.h_next, e.x1, N_EMBD * 4, cudaMemcpyDeviceToDevice,
                                       e.stm));
            int mpos = hpos + 1; // position of the token being embedded (= d_token)
            CUDA_CHECK(cudaMemcpyAsync(e.d_pos_m, &mpos, 4, cudaMemcpyHostToDevice, e.stm));
            e.mtp_forward(); // drafts the token AFTER d_token
            int draft, main_next;
            CUDA_CHECK(cudaMemcpyAsync(&draft, e.d_draft, 4, cudaMemcpyDeviceToHost, e.stm));
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            e.step_free(); // main processes d_token -> new d_token = ground truth
            CUDA_CHECK(cudaMemcpyAsync(&main_next, e.d_token, 4, cudaMemcpyDeviceToHost, e.stm));
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            drafted++;
            if (draft == main_next) accepted++;
            hpos++;
        }
    } else {
        for (int i = 0; i < n_gen; i++) e.step_free();
    }
    CUDA_CHECK(cudaEventRecord(t1, e.stm));
    CUDA_CHECK(cudaStreamSynchronize(e.stm));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    // d_gen[i] = predicted-next after step i; generated tokens start at prompt_len-1
    std::vector<int> gen(toks.size() + n_gen);
    CUDA_CHECK(cudaMemcpy(gen.data(), e.d_gen, gen.size() * 4, cudaMemcpyDeviceToHost));
    printf("generated:");
    for (size_t i = toks.size() - 1; i < toks.size() - 1 + n_gen; i++) printf(" %d", gen[i]);
    printf("\ndecode: %d tokens in %.1f ms = %.2f t/s\n", n_gen, ms, n_gen * 1000.0f / ms);
    if (drafted)
        printf("mtp acceptance: %d/%d = %.1f%%\n", accepted, drafted,
               100.0 * accepted / drafted);
    return 0;
}

