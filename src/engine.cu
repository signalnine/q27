// q27 CLI: bench / greedy / spec-decode harness.
#include <chrono>
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
    int pfdbg_n = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--mtp")) mtp_stats = true;
        if (!strcmp(argv[i], "--spec")) { spec = true; mtp_stats = true; } // spec needs MTP warmup
        if (!strcmp(argv[i], "--fast-head")) fast = true;
        if (!strcmp(argv[i], "--pf") && i + 1 < argc) pf_n = atoi(argv[++i]);
        if (!strcmp(argv[i], "--pfcache")) pfcache = true;
        if (!strcmp(argv[i], "--pfdbg") && i + 1 < argc) pfdbg_n = atoi(argv[++i]);
    }
    if (toks.empty()) { fprintf(stderr, "need --tokens\n"); return 1; }

    Engine e(path, ctx);
    e.fast_head = fast;
    e.build_graph();
    if (spec) e.build_spec_graphs();

    if (pfcache) {
        // M6.5 gate: turn-2 resume must produce identical tokens to a cold run
        if (!spec) e.build_spec_graphs();
        std::vector<int> A, B;
        for (int i = 0; i < 600; i++) A.push_back(toks[i % toks.size()]);
        B = A;
        for (int i = 0; i < 64; i++) B.push_back(toks[(i + 3) % toks.size()]);
        auto timed = [&](const std::vector<int>& p, std::vector<int>& out) {
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
            });
            return ttft;
        };
        std::vector<int> o1, warm, cold;
        double t1 = timed(A, o1);                       // turn 1 (cold, saves snapshot)
        double tw = timed(B, warm);                     // turn 2 (resume from prefix)
        e.have_snap = false;
        double tc = timed(B, cold);                     // turn 2 cold rerun
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
        auto s_kc = grabh(e.kcache[0], (size_t)T * N_KV * HEAD_DIM);
        auto s_mk = grabh(e.mtp_k, (size_t)(T) * N_KV * HEAD_DIM);
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
        auto b_kc = grabh(e.kcache[0], (size_t)T * N_KV * HEAD_DIM);
        auto b_mk = grabh(e.mtp_k, (size_t)(T) * N_KV * HEAD_DIM);
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
        double ts = run(false, a);
        double tb = run(true, b);
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
        int total_emitted = 0, rounds = 0, hist1 = 0, hist2 = 0, hist3 = 0;
        while ((int)out.size() < n_gen) {
            int em[3];
            int n = e.spec_round(em);
            for (int k = 0; k < n; k++) out.push_back(em[k]);
            rounds++;
            total_emitted += n;
            if (n == 1) hist1++; else if (n == 2) hist2++; else hist3++;
            P += n;
        }
        fprintf(stderr, "round outcomes: 1-tok %d, 2-tok %d, 3-tok %d\n", hist1, hist2, hist3);
        drafted = rounds;
        accepted = total_emitted; // repurposed: tokens per round stats
        CUDA_CHECK(cudaEventRecord(t1, e.stm));
        CUDA_CHECK(cudaStreamSynchronize(e.stm));
        float msf = 0;
        CUDA_CHECK(cudaEventElapsedTime(&msf, t0, t1));
        printf("generated:");
        for (int i = 0; i < n_gen; i++) printf(" %d", out[i]);
        printf("\nspec decode: %d tokens in %.1f ms = %.2f t/s (%.2f tokens/round over %d rounds)\n",
               (int)out.size(), msf, out.size() * 1000.0f / msf,
               (double)accepted / drafted, drafted);
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

