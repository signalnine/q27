// q27 CLI: bench / greedy / spec-decode harness.
#include <chrono>
#include <tuple>
#include "engine.cuh"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s model.q27 --tokens \"1,2,3\" [-n N] [--ctx C] [--dump-logits f]\n"
                "       [--spec] [--temp T --top-p P --seed S]  (temp>0 = sampled spec decode)\n",
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
        } else if (!strcmp(argv[i], "--tokens-file") && i + 1 < argc) {
            // long prompts exceed the 128KB single-arg limit; read ids (comma /
            // space / newline separated) from a file instead.
            FILE* tf = fopen(argv[++i], "r");
            if (!tf) { fprintf(stderr, "cannot open tokens-file %s\n", argv[i]); return 1; }
            int v; char sep;
            while (fscanf(tf, "%d", &v) == 1) { toks.push_back(v); (void)fscanf(tf, "%c", &sep); }
            fclose(tf);
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump = argv[++i];
    }
    bool mtp_stats = false, spec = false, fast = false;
    int pf_n = 0;
    bool pfcache = false;
    int stats_n = 0, burst_n = 0;
    const char* burst_out = "burst_stats.csv";
    int pfdbg_n = 0;
    std::string nll_path;
    int nll_chunk = 512, nll_long = 0, nll_max = 0, kvstats_n = 0;
    bool nll_serial = false, verify_weights = false;
    std::string taps_path; // DFlash P0a tap capture (--dump-taps <file>)
    bool p0b = false;      // DFlash P0b S=16 verify-cost bench
    // Sampling (roadmap #2). temp>0 routes the --spec loop to the sampled spec
    // path (Phase 2); Q27_SAMPLE_PLAIN=1 forces the plain sampler for the A/B.
    double temp = 0.0, top_p = 1.0;
    unsigned long long seed = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--mtp")) mtp_stats = true;
        if (!strcmp(argv[i], "--spec")) { spec = true; mtp_stats = true; } // spec needs MTP warmup
        if (!strcmp(argv[i], "--fast-head")) fast = true;
        if (!strcmp(argv[i], "--pf") && i + 1 < argc) pf_n = atoi(argv[++i]);
        if (!strcmp(argv[i], "--pfcache")) pfcache = true;
        if (!strcmp(argv[i], "--stats") && i + 1 < argc) stats_n = atoi(argv[++i]);
        if (!strcmp(argv[i], "--burst-stats") && i + 1 < argc) burst_n = atoi(argv[++i]);
        if (!strcmp(argv[i], "--burst-out") && i + 1 < argc) burst_out = argv[++i];
        if (!strcmp(argv[i], "--pfdbg") && i + 1 < argc) pfdbg_n = atoi(argv[++i]);
        if (!strcmp(argv[i], "--nll") && i + 1 < argc) nll_path = argv[++i];
        if (!strcmp(argv[i], "--nll-chunk") && i + 1 < argc) nll_chunk = atoi(argv[++i]);
        if (!strcmp(argv[i], "--nll-long") && i + 1 < argc) nll_long = atoi(argv[++i]);
        if (!strcmp(argv[i], "--nll-max") && i + 1 < argc) nll_max = atoi(argv[++i]);
        if (!strcmp(argv[i], "--nll-serial")) nll_serial = true;
        if (!strcmp(argv[i], "--verify-weights")) verify_weights = true;
        if (!strcmp(argv[i], "--kvstats") && i + 1 < argc) kvstats_n = atoi(argv[++i]);
        if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = atof(argv[++i]);
        if (!strcmp(argv[i], "--top-p") && i + 1 < argc) top_p = atof(argv[++i]);
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
        if (!strcmp(argv[i], "--dump-taps") && i + 1 < argc) taps_path = argv[++i];
        if (!strcmp(argv[i], "--p0b")) p0b = true;
    }
    if (toks.empty() && nll_path.empty()) { fprintf(stderr, "need --tokens\n"); return 1; }
    // Review 2026-07-09 P0 #2: the direct CLI path fed prompt ingestion and
    // generation straight into the KV caches and d_gen (all sized --ctx) with
    // no bound -- an oversized tokens-file corrupted the caches, and the final
    // d_gen copy read past the allocation. Refuse/clamp up front instead.
    if ((int)toks.size() > ctx) {
        fprintf(stderr, "prompt %zu tokens > --ctx %d -- refusing (raise --ctx)\n", toks.size(),
                ctx);
        return 1;
    }
    if (!toks.empty() && (int)toks.size() + n_gen > ctx) {
        n_gen = ctx - (int)toks.size();
        fprintf(stderr, "-n clamped to %d (prompt %zu + n must fit --ctx %d)\n", n_gen,
                toks.size(), ctx);
    }

    Engine e(path, ctx);
    e.fast_head = fast;
    e.build_graph();
    if (spec) e.build_spec_graphs();

    if (!taps_path.empty()) {
        // DFlash P0a tap capture (docs/dflash-block-verify-design.md):
        // batched-prefill all but the last TAP_TAIL prompt tokens (the
        // drafter's feature window only needs the recent committed tokens),
        // then eager-forward the tail + generation dumping per-step taps.
        // File format per step: int32 committed token, 5*N_EMBD fp32 taps.
        const int TAP_TAIL = 160; // drafter window 128 + margin
        if ((int)toks.size() + n_gen + 8 > ctx) {
            fprintf(stderr, "--dump-taps: prompt %zu + n %d > --ctx %d -- refusing\n",
                    toks.size(), n_gen, ctx);
            return 1;
        }
        FILE* tf = fopen(taps_path.c_str(), "wb");
        if (!tf) { fprintf(stderr, "cannot open %s\n", taps_path.c_str()); return 1; }
        int NP = (int)toks.size();
        int base = NP > TAP_TAIL ? NP - TAP_TAIL : 0;
        if (base >= 32) {
            int* d_toks;
            CUDA_CHECK(cudaMalloc((void**)&d_toks, (size_t)NP * 4));
            CUDA_CHECK(cudaMemcpyAsync(d_toks, toks.data(), (size_t)NP * 4,
                                       cudaMemcpyHostToDevice, e.stm));
            const int PT = Engine::PF_T;
            for (int c0 = 0; c0 < base; c0 += PT) {
                int T = std::min(PT, base - c0);
                e.prefill_chunk(d_toks + c0, c0, T);
            }
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            CUDA_CHECK(cudaFree(d_toks));
        } else {
            base = 0;
        }
        CUDA_CHECK(cudaMemcpyAsync(e.d_pos, &base, 4, cudaMemcpyHostToDevice, e.stm));
        CUDA_CHECK(cudaMemcpyAsync(e.d_step, &base, 4, cudaMemcpyHostToDevice, e.stm));
        std::vector<float> htaps(5 * (size_t)N_EMBD);
        auto dump_step = [&](int committed) {
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            CUDA_CHECK(cudaMemcpy(htaps.data(), e.d_taps, htaps.size() * 4,
                                  cudaMemcpyDeviceToHost));
            fwrite(&committed, 4, 1, tf);
            fwrite(htaps.data(), 4, htaps.size(), tf);
        };
        for (int i = base; i < NP; i++) { e.step_taps(toks[i]); dump_step(toks[i]); }
        for (int g = 0; g < n_gen; g++) {
            int tok_in;
            CUDA_CHECK(cudaMemcpy(&tok_in, e.d_token, 4, cudaMemcpyDeviceToHost));
            e.step_taps_free();
            dump_step(tok_in);
        }
        fclose(tf);
        fprintf(stderr, "taps: %d steps (%d prompt-tail + %d gen) -> %s\n",
                NP - base + n_gen, NP - base, n_gen, taps_path.c_str());
        return 0;
    }

    if (p0b) {
        // DFlash P0b: batched-prefill the prompt to depth, then time the
        // S=16 verify anatomy -- one T=16 prefill_chunk + the 16-row head
        // (rmsnorm_T + qxT + mmT). Timing rig only: repeated chunks at the
        // same positions corrupt GDN state by design; state is discarded.
        // Q27_P0B_T: verify width sweep (spike 2026-07-09); default 16
        const char* te = getenv("Q27_P0B_T");
        const int TT = te ? atoi(te) : 16;
        int NP = (int)toks.size();
        if (NP < 64 || NP + TT > ctx || TT < 1 || TT > (int)Engine::PF_T) {
            fprintf(stderr, "--p0b: need 64 <= prompt, prompt+T <= ctx, 1 <= T <= PF_T\n");
            return 1;
        }
        int* d_toks;
        float* d_lg;
        CUDA_CHECK(cudaMalloc((void**)&d_toks, (size_t)NP * 4));
        CUDA_CHECK(cudaMalloc((void**)&d_lg, (size_t)TT * VOCAB * 4));
        CUDA_CHECK(cudaMemcpyAsync(d_toks, toks.data(), (size_t)NP * 4, cudaMemcpyHostToDevice,
                                   e.stm));
        const int PT = Engine::PF_T;
        const int depth = NP - TT;
        for (int c0 = 0; c0 < depth; c0 += PT) {
            int T = std::min(PT, depth - c0);
            e.prefill_chunk(d_toks + c0, c0, T);
        }
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        const DevTensor& onw = e.dm.get("output_norm.weight");
        const DevTensor& head = e.dm.get("output.weight");
        auto cycle = [&](bool with_head) {
            e.prefill_chunk(d_toks + depth, depth, TT);
            if (with_head) {
                q27k::rmsnorm_T(e.hT, (const float*)onw.data, e.x1T, N_EMBD, TT, EPS, e.stm);
                e.qxT(e.x1T, N_EMBD, TT);
                e.mmT(head, e.x1T, d_lg, TT);
            }
        };
        cycle(true); // warm
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        cudaEvent_t v0, v1;
        CUDA_CHECK(cudaEventCreate(&v0));
        CUDA_CHECK(cudaEventCreate(&v1));
        const int REPS = 50;
        for (int pass = 0; pass < 2; pass++) {
            bool with_head = pass == 1;
            CUDA_CHECK(cudaEventRecord(v0, e.stm));
            for (int r = 0; r < REPS; r++) cycle(with_head);
            CUDA_CHECK(cudaEventRecord(v1, e.stm));
            CUDA_CHECK(cudaEventSynchronize(v1));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, v0, v1));
            printf("p0b depth=%d T=%d %s: %.3f ms/cycle\n", depth, TT,
                   with_head ? "chunk+head" : "chunk-only", ms / REPS);
        }
        return 0;
    }

    if (burst_n > 0) {
        // ctx budget (review follow-up 2026-07-09 #1): the rig steps
        // prompt+burst_n positions AND chains SD MTP probes ahead of each --
        // all land in caches sized --ctx
        if ((int)toks.size() + burst_n + 10 > ctx) {
            fprintf(stderr, "--burst-stats: prompt %zu + N %d + probe depth 10 > --ctx %d -- "
                            "refusing (raise --ctx)\n", toks.size(), burst_n, ctx);
            return 1;
        }
        // Burst-depth gate rig (2026-07-04, path-1 decision): chain SD MTP
        // draft passes per FREE-region position on the plain serial path and
        // dump drafts + top1-top2 margins to CSV for offline gate analysis
        // (chain-length distribution, full-accept trigger, margin thresholds
        // -- decides whether gated deep bursts (d8-10) get built). Prompt
        // phase mirrors production MTP KV warmup: pair h(i) with the ACTUAL
        // next prompt token. Free-region drafting matches production exactly
        // (chain from the accepted token), so no stats-vs-live prompt skew;
        // the P3 per-position-vs-round discount still applies to projections.
        constexpr int SD = 10;
        std::vector<float> l1(VOCAB);
        auto top2 = [&](const std::vector<float>& l) {
            int b = 0; float bv = -1e30f, sv = -1e30f;
            for (int i = 0; i < VOCAB; i++) {
                if (l[i] > bv) { sv = bv; bv = l[i]; b = i; }
                else if (l[i] > sv) sv = l[i];
            }
            return std::pair<int, float>(b, bv - sv);
        };
        std::vector<int> seq = toks;
        FILE* out = fopen(burst_out, "w");
        if (!out) { fprintf(stderr, "burst-stats: cannot open %s\n", burst_out); return 1; }
        fprintf(out, "q");
        for (int k = 0; k < SD; k++) fprintf(out, ",d%d,m%d", k + 1, k + 1);
        fprintf(out, "\n");
        for (int step = 0; step < burst_n + (int)toks.size() - 1; step++) {
            bool prompt_phase = step < (int)toks.size();
            if (prompt_phase) e.step_with(toks[step]);
            else e.step_free();
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            int next_tok;
            CUDA_CHECK(cudaMemcpy(&next_tok, e.d_token, 4, cudaMemcpyDeviceToHost));
            if (!prompt_phase) seq.push_back(next_tok);
            int pos = step;
            if (prompt_phase) {
                // MTP KV warmup with the actual next prompt token
                if (step + 1 < (int)toks.size()) {
                    CUDA_CHECK(cudaMemcpyAsync(e.h_next, e.x1, N_EMBD * 4,
                                               cudaMemcpyDeviceToDevice, e.stm));
                    int wt = toks[step + 1], mp = pos + 1;
                    CUDA_CHECK(cudaMemcpyAsync(e.d_token, &wt, 4, cudaMemcpyHostToDevice,
                                               e.stm));
                    CUDA_CHECK(cudaMemcpyAsync(e.d_pos_m, &mp, 4, cudaMemcpyHostToDevice,
                                               e.stm));
                    e.mtp_forward();
                    CUDA_CHECK(cudaStreamSynchronize(e.stm));
                    CUDA_CHECK(cudaMemcpy(e.d_token, &next_tok, 4, cudaMemcpyHostToDevice));
                }
                continue;
            }
            if (step >= burst_n + (int)toks.size() - 2) break;
            // free region: chain SD draft passes from (h(pos), accepted tok)
            int q = (int)seq.size() - 1; // index of next_tok in seq
            int D[SD]; float M[SD];
            CUDA_CHECK(cudaMemcpyAsync(e.h_next, e.x1, N_EMBD * 4, cudaMemcpyDeviceToDevice,
                                       e.stm));
            for (int k = 0; k < SD; k++) {
                int mp = pos + 1 + k;
                if (k > 0)
                    CUDA_CHECK(cudaMemcpyAsync(e.d_token, &D[k - 1], 4, cudaMemcpyHostToDevice,
                                               e.stm));
                CUDA_CHECK(cudaMemcpyAsync(e.d_pos_m, &mp, 4, cudaMemcpyHostToDevice, e.stm));
                if (k == 0) e.mtp_forward();
                else e.mtp_forward(e.x1, nullptr, nullptr, nullptr);
                CUDA_CHECK(cudaStreamSynchronize(e.stm));
                CUDA_CHECK(cudaMemcpy(l1.data(), e.mtp_logits, (size_t)VOCAB * 4,
                                      cudaMemcpyDeviceToHost));
                auto [d, m] = top2(l1);
                D[k] = d; M[k] = m;
            }
            fprintf(out, "%d", q);
            for (int k = 0; k < SD; k++) fprintf(out, ",%d,%.4f", D[k], M[k]);
            fprintf(out, "\n");
            CUDA_CHECK(cudaMemcpy(e.d_token, &next_tok, 4, cudaMemcpyHostToDevice));
        }
        fclose(out);
        std::string seqp = std::string(burst_out) + ".seq";
        FILE* sf = fopen(seqp.c_str(), "w");
        for (size_t i = 0; i < seq.size(); i++)
            fprintf(sf, "%d%c", seq[i], i + 1 < seq.size() ? ' ' : '\n');
        fclose(sf);
        printf("burst-stats: %d free positions x %d-deep chains -> %s (+.seq)\n", burst_n, SD,
               burst_out);
        return 0;
    }

    if (stats_n > 0) {
        // ctx budget (review follow-up 2026-07-09 #1): steps prompt+N
        // positions with up-to-5-deep MTP probe chains (pend arrays carry a
        // +8 margin; use the same bound here)
        if ((int)toks.size() + stats_n + 8 > ctx) {
            fprintf(stderr, "--stats: prompt %zu + N %d + probe margin 8 > --ctx %d -- "
                            "refusing (raise --ctx)\n", toks.size(), stats_n, ctx);
            return 1;
        }
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
        struct Pend3 { int pred = -1; float margin2 = 0; int d1 = -1, d2 = -1; float margin3 = 0; };
        std::vector<Pend3> pend3(N + 8);
        long c3n[5] = {0}, c3pre[5] = {0}, c3ok[5] = {0};
        // accept-gate Phase 0b: acceptance binned by each pass's OWN margin
        // (the signal the live per-step gate actually thresholds), conditioned
        // on the prefix being accepted -- oKok/oKn = p(dK | prefix ok, mK bin).
        // The cK* bins above key on the pass-2 margin only.
        long o3n[5] = {0}, o3ok[5] = {0};
        // P3 gate: depth-4 chain -- p(pass-4 draft | d1,d2,d3 all accepted).
        // Build depth-4 only if p(d4|prefix-3) holds ~>=60%.
        struct Pend4 { int pred = -1; float margin2 = 0; int d1 = -1, d2 = -1, d3 = -1; float margin4 = 0; };
        std::vector<Pend4> pend4(N + 8);
        long c4n[5] = {0}, c4pre[5] = {0}, c4ok[5] = {0};
        long o4n[5] = {0}, o4ok[5] = {0};
        // Depth-5 gate (roadmap #4, 2026-07-03): p(pass-5 draft | d1..d4 all
        // accepted). Chain barely decays through d4 (97.4%), so measure d5
        // before dismissing it -- projected +5-6%% net if the pattern holds.
        struct Pend5 { int pred = -1; float margin2 = 0; int d1 = -1, d2 = -1, d3 = -1, d4 = -1; float margin5 = 0; };
        std::vector<Pend5> pend5(N + 8);
        long c5n[5] = {0}, c5pre[5] = {0}, c5ok[5] = {0};
        long o5n[5] = {0}, o5ok[5] = {0};
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
            // score pending predictions targeting this newly known token.
            // Guard: pends are only STORED for indices < N+8 (write sites are
            // range-checked); a prompt longer than N used to read past the
            // vectors here (heap garbage swamped every counter to 0).
            int known_idx = (int)seq.size() - 1;
            bool in_pend = known_idx < (int)pend1.size(); // all five are N+8
            if (in_pend && pend1[known_idx].pred >= 0) {
                n1++;
                bool ok = pend1[known_idx].pred == seq[known_idx];
                if (ok) n1ok++;
                else {
                    r2tot++;
                    if (pend1[known_idx].rank2 == seq[known_idx]) r2cap++;
                }
            }
            if (in_pend && pend2[known_idx].pred >= 0) {
                n2++;
                bool ok = pend2[known_idx].pred == seq[known_idx];
                int b = bin(pend2[known_idx].margin);
                bn[b]++;
                if (ok) { n2ok++; bok[b]++; }
            }
            if (in_pend && known_idx >= 2 && pend3[known_idx].pred >= 0) {
                const Pend3& p3 = pend3[known_idx];
                int b = bin(p3.margin2);
                c3n[b]++;
                if (p3.d1 == seq[known_idx - 2] && p3.d2 == seq[known_idx - 1]) {
                    c3pre[b]++;
                    int ob = bin(p3.margin3);
                    o3n[ob]++;
                    if (p3.pred == seq[known_idx]) { c3ok[b]++; o3ok[ob]++; }
                }
            }
            if (in_pend && known_idx >= 3 && pend4[known_idx].pred >= 0) {
                const Pend4& p4 = pend4[known_idx];
                int b = bin(p4.margin2);
                c4n[b]++;
                if (p4.d1 == seq[known_idx - 3] && p4.d2 == seq[known_idx - 2] &&
                    p4.d3 == seq[known_idx - 1]) {
                    c4pre[b]++;
                    int ob = bin(p4.margin4);
                    o4n[ob]++;
                    if (p4.pred == seq[known_idx]) { c4ok[b]++; o4ok[ob]++; }
                }
            }
            if (in_pend && known_idx >= 4 && pend5[known_idx].pred >= 0) {
                const Pend5& p5 = pend5[known_idx];
                int b = bin(p5.margin2);
                c5n[b]++;
                if (p5.d1 == seq[known_idx - 4] && p5.d2 == seq[known_idx - 3] &&
                    p5.d3 == seq[known_idx - 2] && p5.d4 == seq[known_idx - 1]) {
                    c5pre[b]++;
                    int ob = bin(p5.margin5);
                    o5n[ob]++;
                    if (p5.pred == seq[known_idx]) { c5ok[b]++; o5ok[ob]++; }
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
            (void)d3b;
            if (known_idx + 3 < (int)pend3.size()) pend3[known_idx + 3] = {d3, m2, d1, d2, m3};
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
            (void)d4b;
            if (known_idx + 4 < (int)pend4.size()) pend4[known_idx + 4] = {d4, m2, d1, d2, d3, m4};
            // MTP pass 5: chain from pass-4 hidden, draft seq[known_idx+5]
            int mp5 = pos + 5;
            CUDA_CHECK(cudaMemcpyAsync(e.d_token, &d4, 4, cudaMemcpyHostToDevice, e.stm));
            CUDA_CHECK(cudaMemcpyAsync(e.d_pos_m, &mp5, 4, cudaMemcpyHostToDevice, e.stm));
            e.mtp_forward(e.x1, nullptr, nullptr, nullptr);
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            CUDA_CHECK(cudaMemcpy(l1.data(), e.mtp_logits, (size_t)VOCAB * 4,
                                  cudaMemcpyDeviceToHost));
            auto [d5, d5b, m5] = top2(l1);
            (void)d5b;
            if (known_idx + 5 < (int)pend5.size()) pend5[known_idx + 5] = {d5, m2, d1, d2, d3, d4, m5};
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
        // accept-gate Phase 0b: same acceptance, binned by each pass's OWN
        // margin (the live gate's actual per-step signal), prefix-conditioned.
        // Flat rows across bins = margin does not predict acceptance at that
        // depth -> theta-schedule complement is dead, yield feedback is the
        // whole game (docs/acceptance-gate-design.md).
        printf("  acceptance by OWN-pass margin, prefix ok (accept-gate Phase 0b):\n");
        struct { const char* name; long *on, *ook; } own[3] = {
            {"d3", o3n, o3ok}, {"d4", o4n, o4ok}, {"d5", o5n, o5ok}};
        for (auto& d : own) {
            printf("    %s:", d.name);
            for (int b = 0; b < 5; b++)
                printf("  %s=%5.1f%%(n=%ld)", bl[b],
                       100.0 * d.ook[b] / (d.on[b] ? d.on[b] : 1), d.on[b]);
            printf("\n");
        }
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
        // format guard up front: the scan (and its prefill) is fp16-only
        if (e.kv_fp8 || e.kv_kind >= KV_T3) {
            fprintf(stderr, "--kvstats reads fp16 caches; unset Q27_KV\n");
            return 1;
        }
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

        if (nll_chunk > ctx) {
            // each chunk steps positions 0..C-1 into caches sized --ctx
            fprintf(stderr, "--nll-chunk %d > --ctx %d -- refusing (raise --ctx)\n", nll_chunk,
                    ctx);
            return 1;
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
        // ctx budget (review follow-up 2026-07-09 #1): synthesizes N tokens
        // into caches sized --ctx
        if (pfdbg_n > ctx) {
            fprintf(stderr, "--pfdbg %d > --ctx %d -- refusing (raise --ctx)\n", pfdbg_n, ctx);
            return 1;
        }
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
                    // --dump-logits on the BATCHED leg captures the post-prefill
                    // logits at position pf_n (still resident in e.logits before
                    // the first decode overwrites them) -- i.e. the batched
                    // prefill kernel's own output. This is the only route that
                    // exercises k_attn_prefill_mma[_fp8q]; --nll prefills per-token
                    // via step_with and never hits it. Enables an fp8q-vs-default
                    // logit A/B (cosine/maxdiff/KL) to quantify the fp8 QK^T delta
                    // at depth -- the default-on gate the greedy checks don't give.
                    if (batched && !dump.empty()) {
                        std::vector<float> lg(VOCAB);
                        CUDA_CHECK(cudaMemcpy(lg.data(), e.logits, (size_t)VOCAB * 4,
                                              cudaMemcpyDeviceToHost));
                        FILE* df = fopen(dump.c_str(), "wb");
                        if (df) {
                            fwrite(lg.data(), 4, VOCAB, df);
                            fclose(df);
                            fprintf(stderr, "pf logits[pos %d] -> %s\n", pf_n, dump.c_str());
                        }
                    }
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
        // Identity across serial-vs-batched only holds on the exact g32 path;
        // the default g64 activation regroup changes batched quantization BY
        // DESIGN (tolerance-gated instead -- policy 2026-07-04). Run with
        // Q27_PF_XG=32 to enforce the identity gate.
        const char* xg_env = getenv("Q27_PF_XG");
        const bool xg32 = xg_env && !strcmp(xg_env, "32");
        printf("continuations %s (%zu vs %zu tokens)\n",
               a == b ? "IDENTICAL -- gate PASS"
                      : (xg32 ? "MISMATCH -- gate FAIL"
                              : "MISMATCH -- expected under g64 regroup (set Q27_PF_XG=32 "
                                "for the identity gate)"),
               a.size(), b.size());
        if (a != b && !xg32) return 0;
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
        // Sampling (Phase 2): temp>0 resamples the pending token + rejection-
        // accepts drafts. Same bootstrap as Engine::generate -- e.logits holds
        // the last prompt token's logits and h_next/d_P/d_pos are set, so the
        // first eager draw keys at d_pos with kind 0. Q27_SAMPLE_PLAIN=1 forces
        // the plain sampler (no spec) for the spec==non-spec distribution A/B.
        const bool sampling = temp > 0.0;
        const bool plain_sample = getenv("Q27_SAMPLE_PLAIN") != nullptr;
        if (sampling) {
            e.samp = {(float)(1.0 / temp), (float)top_p, seed};
            CUDA_CHECK(cudaMemcpyAsync(e.d_samp, &e.samp, sizeof e.samp, cudaMemcpyHostToDevice,
                                       e.stm));
            e.samp_first = true;
            fprintf(stderr, "sampling: T=%.3f top_p=%.3f seed=%llu path=%s\n", temp, top_p, seed,
                    plain_sample ? "plain" : "spec");
        }
        // width-12 P1: arm the suffix drafter like Engine::generate does --
        // the CLI drove spec_round directly with an EMPTY suffix index, so
        // Q27_SUFFIX could never fire on --tokens replays (zero-fire by
        // construction, not by traffic). Round-grouping only; emitted
        // tokens stay greedy-identical.
        if (e.suffix_on) {
            e.sfx.reset(toks);
            e.sfx_valid = false;
        }
        int total_emitted = 0, rounds = 0, hist[W_MAX] = {0}; // width-12: up to 12-tok rounds
        while ((int)out.size() < n_gen) {
            if (P + e.ctx_round_reserve() > ctx) { fprintf(stderr, "ctx-guard: stopping at P=%d\n", P); break; }
            int em[W_MAX]; // width-12: a round emits up to 12 tokens
            int n = sampling ? (plain_sample ? e.sample_round(em) : e.spec_sample_round(em))
                             : e.spec_round(em);
            for (int k = 0; k < n; k++) out.push_back(em[k]);
            if (e.suffix_on)
                for (int k = 0; k < n; k++) e.sfx.append(em[k]);
            rounds++;
            total_emitted += n;
            hist[n - 1]++;
            P += n;
        }
        fprintf(stderr,
                "round outcomes: 1-tok %d, 2-tok %d, 3-tok %d, 4-tok %d, 5-tok %d, 6-tok %d, "
                "7-tok %d, 8-tok %d, 9-tok %d, 10-tok %d, 11-tok %d, 12-tok %d\n",
                hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7], hist[8],
                hist[9], hist[10], hist[11]);
        if (e.maxd_auto)
            fprintf(stderr,
                    "adaptive maxd: %ld rounds @depth-4, %ld @depth-5, %ld @depth-6, "
                    "%ld @depth-7 (%ld promotes, %ld demotes); final ceiling=%d\n",
                    e.dctl.rounds[4], e.dctl.rounds[5], e.dctl.rounds[6], e.dctl.rounds[7],
                    e.dctl.promotes, e.dctl.demotes, e.dctl.cur);
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

