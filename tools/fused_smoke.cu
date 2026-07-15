// fused_smoke -- 2-engine proof of the P1 fused verify round (continuous
// batching, docs/plans/2026-07-14-continuous-batching.md Task 8).
//
// CLAIM UNDER TEST: a manual conductor loop -- per-engine gated drafts on each
// engine's OWN stream, then ONE eager fused verify (union weight sweep +
// per-engine mixers/tails) on a dedicated conductor stream -- emits, for every
// engine, exactly the token ids its untouched solo generate() path emits.
// Widths here are 2 engines x <= 5 lanes = union <= 10 <= W_MAX (12), so the
// trim policy never fires and the "bitwise-when-untrimmed" contract applies
// in full (ninv_test proved the lane kernels N-invariant; this proves the
// PLUMBING: union view construction, in-place sweep, stream/event wiring,
// per-engine commit bookkeeping).
//
// Config is pinned via setenv BEFORE engine construction:
//   Q27_PMIN=0.5     -- gated rounds (the conductor path is gated-greedy)
//   Q27_MAXD=4       -- FIXED ladder: md_used constant, so no depthctl state
//                       can diverge between the solo and fused runs
//   Q27_GEMM_MIN=99  -- pin mm5 to the dp4a GEMV at every width. The union
//                       (up to 10) crosses the default gemm_min=9, where mm5
//                       would take k_vgemm while each lane's SOLO round
//                       (width <= 5) took the GEMV. vgemm==gemv was never
//                       claimed bitwise (gemm-verify plan declined exactly
//                       that), so an unpinned run would fork the numeric path
//                       and fail byte-identity for a reason that has nothing
//                       to do with the fused plumbing. Task 9 landed exactly
//                       that policy call in build_union_view (all-gated
//                       unions force the GEMV family), so this pin is now
//                       REDUNDANT-BUT-HARMLESS; kept so the smoke stays a
//                       plumbing test, independent of the policy code.
//
// Build (mirrors the Makefile q27 rule's file list, engine main() swapped
// for this driver; no Makefile edit per the plan's sensitive-file rule):
//   /usr/local/cuda/bin/nvcc -O2 -std=c++17 \
//     -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120 \
//     -Xcompiler -Wall tools/fused_smoke.cu src/blocks.cu src/prefill.cu \
//     src/kernels.cu src/spec3.cu src/vgemm.cu src/device_model.cu \
//     src/loader.cpp -o build/fused_smoke
//
// Run: build/fused_smoke [model.q27]   (default: the canonical vanilla qwen)
// Success line: "FUSED SMOKE PASS: streamA identical, streamB identical".

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/engine.cuh"
#include "../src/conductor.h"

static const char* DEF_MODEL = "/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27";
static constexpr int N_GEN = 32;

// Serial short-prompt prefill, mirroring generate()'s NP<32 branch verbatim
// (reset + step_with walk + MTP warm interleave + h_next/d_P epilogue). Leg B
// re-runs this between legs: reset() restores perm/GDN/MTP-KV to the same
// state a fresh engine has, and stale attention-KV rows past the prompt are
// rewritten before any read (the same argument the prefix cache rests on).
static void prefill_serial(Engine& e, const std::vector<int>& prompt) {
    e.reset();
    e.have_snap = false;
    e.snap_toks.clear();
    e.ckpt_clear();
    for (size_t i = 0; i < prompt.size(); i++) {
        e.step_with(prompt[i]);
        if (i + 1 < prompt.size()) {
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
            CUDA_CHECK(cudaMemcpyAsync(e.h_next, e.x1, N_EMBD * 4,
                                       cudaMemcpyDeviceToDevice, e.stm));
            int nt = prompt[i + 1], mp = (int)i + 1;
            CUDA_CHECK(cudaMemcpyAsync(e.d_token, &nt, 4, cudaMemcpyHostToDevice, e.stm));
            CUDA_CHECK(cudaMemcpyAsync(e.d_pos_m, &mp, 4, cudaMemcpyHostToDevice, e.stm));
            e.mtp_forward();
            CUDA_CHECK(cudaStreamSynchronize(e.stm));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(e.stm));
    CUDA_CHECK(cudaMemcpyAsync(e.h_next, e.x1, N_EMBD * 4, cudaMemcpyDeviceToDevice, e.stm));
    int P = (int)prompt.size() - 1;
    CUDA_CHECK(cudaMemcpyAsync(e.d_P, &P, 4, cudaMemcpyHostToDevice, e.stm));
    CUDA_CHECK(cudaStreamSynchronize(e.stm));
}

// A6 audit: build a union view at a fixed shape and verify, host-side, that
// every union slot's pointers are EXACTLY the owning engine's solo_view()
// entries at the mapped lane, and that no per-lane buffer aliases across
// engines (a duplicate would let the sweep double-write one buffer).
static int audit_union_view(Engine** es, int k, cudaStream_t cstm) {
    int fails = 0;
    int w[2] = {5, 5}; // widest gated shape; the mapping is width-independent
    const bool sfx[2] = {false, false}; // gated union -> GEMV family (policy)
    for (int m = 0; m < k; m++) es[m]->set_round_width(w[m]);
    q27::UnionView uv = q27::build_union_view(es, w, k, cstm, sfx);
    printf("[audit] union view: k=%d vw=%d (slot -> eng/lane, key ptrs)\n", k, uv.view.vw);
    for (int u = 0; u < uv.view.vw; u++) {
        const int m = uv.map[u].eng, j = uv.map[u].lane;
        const Engine::LaneView sv = es[m]->solo_view();
        printf("[audit]  slot %2d -> e%d lane %d  h=%p x1=%p lg=%p dv=%p tok=%p pos=%p\n",
               u, m, j, (void*)uv.view.h[u], (void*)uv.view.x1[u], (void*)uv.view.lg[u],
               (void*)uv.view.dv[u], (const void*)uv.view.vtok.p[u],
               (void*)uv.view.pos.p[u]);
#define FCHECK(F)                                                                        \
        do {                                                                             \
            if (uv.view.F[u] != sv.F[j]) {                                               \
                printf("[audit] FAIL slot %d field " #F " != e%d solo lane %d\n", u, m, j); \
                fails++;                                                                 \
            }                                                                            \
        } while (0)
        FCHECK(x1); FCHECK(qkv); FCHECK(z); FCHECK(alpha); FCHECK(betar);
        FCHECK(g); FCHECK(beta); FCHECK(o); FCHECK(og); FCHECK(y);
        FCHECK(qg); FCHECK(kbuf); FCHECK(vbuf); FCHECK(attnout);
        FCHECK(ffn_g); FCHECK(ffn_u); FCHECK(h); FCHECK(lg); FCHECK(dv);
#undef FCHECK
        if (uv.view.xq[u].nat != sv.xq[j].nat) { printf("[audit] FAIL slot %d xq.nat\n", u); fails++; }
        if (uv.view.vtok.p[u] != sv.vtok.p[j]) { printf("[audit] FAIL slot %d vtok\n", u); fails++; }
        if (uv.view.pos.p[u] != sv.pos.p[j]) { printf("[audit] FAIL slot %d pos\n", u); fails++; }
    }
    // no cross-slot aliasing on the in-place sweep targets (cudaMalloc'd
    // buffers cannot overlap, so pointer inequality == range disjointness)
    for (int a = 0; a < uv.view.vw; a++)
        for (int b = a + 1; b < uv.view.vw; b++) {
            if (uv.view.h[a] == uv.view.h[b] || uv.view.x1[a] == uv.view.x1[b] ||
                uv.view.y[a] == uv.view.y[b] || uv.view.lg[a] == uv.view.lg[b] ||
                uv.view.qkv[a] == uv.view.qkv[b] || uv.view.dv[a] == uv.view.dv[b] ||
                uv.view.xq[a].nat == uv.view.xq[b].nat) {
                printf("[audit] FAIL slots %d/%d alias\n", a, b);
                fails++;
            }
        }
    printf("[audit] %s\n", fails ? "FAIL" : "OK: slot pointers == owning solo views, no aliasing");
    return fails;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : DEF_MODEL;
    // Pin the gated-greedy config BEFORE any engine exists (rationale up top).
    setenv("Q27_PMIN", "0.5", 1);
    setenv("Q27_MAXD", "4", 1);
    setenv("Q27_GEMM_MIN", "99", 1);
    unsetenv("Q27_SUFFIX");   // suffix rounds are a different (non-gated) branch
    unsetenv("Q27_SUFFIX_W");
    unsetenv("Q27_DEXIT");    // default ON = the per-step draft loop we mirror
    unsetenv("Q27_KV");       // fp16 KV, the canonical-gate config
    unsetenv("Q27_TOOL_SPLIT");

    // canonical tokens (stream A) + shortbench "hash-table" prompt (stream B)
    const std::vector<int> pa = {760, 6511, 314, 9338, 369};
    const std::vector<int> pb = {814, 20139, 1204, 264, 5010, 1898, 13081,
                                 45776, 321, 948, 1754, 8024, 14387, 13};

    // one shared weight set, two engines -- the server's P10-A1 construction
    fprintf(stderr, "loading %s\n", path);
    q27::Model shared_model = q27::Model::open(path);
    q27::DeviceModel shared_dm(shared_model);
    shared_dm.upload_all();
    shared_dm.checksum_baseline();
    fprintf(stderr, "resident: %.2f GB\n", shared_dm.bytes_resident() / 1e9);
    Engine eA(shared_model, shared_dm, 2048);
    eA.build_graph();
    eA.build_spec_graphs();
    Engine eB(shared_model, shared_dm, 2048);
    eB.build_graph();
    eB.build_spec_graphs();

    // ---- Leg A: solo reference -- the untouched generate() path ----
    std::vector<int> refA, refB;
    std::vector<std::vector<int>> roundsA[2]; // per-round ids (divergence forensics)
    {
        Engine* egs[2] = {&eA, &eB};
        const std::vector<int>* ps[2] = {&pa, &pb};
        std::vector<int>* outs[2] = {&refA, &refB};
        for (int i = 0; i < 2; i++) {
            egs[i]->on_round = [&, i](const int* em, int n) {
                roundsA[i].emplace_back(em, em + n);
                return -1; // observe only
            };
            auto sink = [&](int id) { outs[i]->push_back(id); return true; };
            egs[i]->generate(*ps[i], N_GEN, /*eos=*/-1, sink);
            egs[i]->on_round = nullptr;
        }
    }
    fprintf(stderr, "[leg A] solo: streamA %zu tokens / %zu rounds, streamB %zu tokens / %zu rounds\n",
            refA.size(), roundsA[0].size(), refB.size(), roundsA[1].size());

    // ---- Leg B: fused conductor loop on re-prefilled engines ----
    prefill_serial(eA, pa);
    prefill_serial(eB, pb);
    cudaStream_t cstm;
    CUDA_CHECK(cudaStreamCreate(&cstm));
    cudaEvent_t ev[2];
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming));

    Engine* all[2] = {&eA, &eB};
    if (audit_union_view(all, 2, cstm)) {
        fprintf(stderr, "FUSED SMOKE FAIL: union view audit\n");
        return 1;
    }

    struct Stream {
        Engine* e;
        std::vector<int> out;
        std::vector<std::vector<int>> rounds;
        int Ph;
    } S[2] = {{&eA, {}, {}, (int)pa.size() - 1}, {&eB, {}, {}, (int)pb.size() - 1}};

    int round = 0;
    while ((int)S[0].out.size() < N_GEN || (int)S[1].out.size() < N_GEN) {
        Engine* es[2];
        int want[2], idx[2], k = 0;
        bool sfx[2] = {false, false};
        for (int i = 0; i < 2; i++) {
            Stream& s = S[i];
            if ((int)s.out.size() >= N_GEN) continue; // left at a round boundary
            if (s.Ph + s.e->ctx_round_reserve() > s.e->max_ctx) continue; // ctx guard mirror
            es[k] = s.e;
            want[k] = s.e->draft_and_gate(); // drafts on s.e->stm (own stream)
            idx[k] = i;
            k++;
        }
        if (k == 0) break; // both ctx-guarded (cannot happen at 2048/32)
        q27::trim_widths(want, sfx, k, W_MAX);
        int uw = 0;
        for (int m = 0; m < k; m++) uw += want[m];
        if (uw > W_MAX) { // 2 x <=5 <= 12: trim must never fire in this smoke
            fprintf(stderr, "FUSED SMOKE FAIL: unexpected trim (union %d)\n", uw);
            return 1;
        }
        for (int m = 0; m < k; m++) es[m]->set_round_width(want[m]);
        for (int m = 0; m < k; m++) CUDA_CHECK(cudaEventRecord(ev[m], es[m]->stm));
        q27::fused_verify_round(es, want, k, cstm, ev, sfx);
        int oc[2][OUTCOME_INTS];
        for (int m = 0; m < k; m++)
            CUDA_CHECK(cudaMemcpyAsync(oc[m], es[m]->d_outcome, OUTCOME_INTS * 4,
                                       cudaMemcpyDeviceToHost, cstm));
        CUDA_CHECK(cudaStreamSynchronize(cstm)); // one sync for the whole batch
        for (int m = 0; m < k; m++) {
            int em[W_MAX];
            int n = es[m]->commit_outcome(oc[m], em);
            Stream& s = S[idx[m]];
            s.Ph += n;
            s.rounds.emplace_back(em, em + n);
            for (int t = 0; t < n && (int)s.out.size() < N_GEN; t++) s.out.push_back(em[t]);
        }
        round++;
    }
    fprintf(stderr, "[leg B] fused: %d rounds; streamA %zu tokens / %zu rounds, streamB %zu tokens / %zu rounds\n",
            round, S[0].out.size(), S[0].rounds.size(), S[1].out.size(), S[1].rounds.size());

    // ---- compare ----
    int bad = 0;
    const std::vector<int>* refs[2] = {&refA, &refB};
    const char* names[2] = {"streamA", "streamB"};
    for (int i = 0; i < 2; i++) {
        const std::vector<int>& r = *refs[i];
        const std::vector<int>& f = S[i].out;
        size_t div = 0;
        while (div < r.size() && div < f.size() && r[div] == f[div]) div++;
        if (div == r.size() && r.size() == f.size()) {
            printf("%s: OK (%zu tokens identical)\n", names[i], r.size());
        } else {
            bad++;
            printf("%s: FAIL first divergence at index %zu (solo %s fused %s)\n", names[i],
                   div, div < r.size() ? std::to_string(r[div]).c_str() : "<end>",
                   div < f.size() ? std::to_string(f[div]).c_str() : "<end>");
            printf("  solo :");
            for (int t : r) printf(" %d", t);
            printf("\n  fused:");
            for (int t : f) printf(" %d", t);
            printf("\n  solo rounds:\n");
            for (size_t rr = 0; rr < roundsA[i].size(); rr++) {
                printf("   [%zu]", rr);
                for (int t : roundsA[i][rr]) printf(" %d", t);
                printf("\n");
            }
            printf("  fused rounds:\n");
            for (size_t rr = 0; rr < S[i].rounds.size(); rr++) {
                printf("   [%zu]", rr);
                for (int t : S[i].rounds[rr]) printf(" %d", t);
                printf("\n");
            }
        }
    }
    if (bad) {
        printf("FUSED SMOKE FAIL: %d stream(s) diverged\n", bad);
        return 1;
    }
    printf("FUSED SMOKE PASS: streamA identical, streamB identical\n");
    return 0;
}
