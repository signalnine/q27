// DFlash2 bring-up smoke: replay a `q27 --dump-taps` dump through the
// src/dflash2.cu runtime and report tokens/round -- the parity gate against
// bench/dflash2/p1_qtap_al.py mode q27 (same dump, torch reference).
//
// usage: dflash2_smoke <pack.d2w> <taps.bin> <prompt_len> [K] [--proposals]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#include "../src/dflash2.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s pack.d2w taps.bin prompt_len [K] [--proposals]\n", argv[0]);
        return 1;
    }
    int K = argc > 4 && argv[4][0] != '-' ? atoi(argv[4]) : q27d2::D2_K;
    bool show = !strcmp(argv[argc - 1], "--proposals");
    const int NT = q27d2::D2_TAPD;
    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    const long step = 4 + (long)NT * 4;
    int M = (int)(sz / step);
    if (sz % step) { fprintf(stderr, "bad dump size\n"); return 1; }
    std::vector<int> toks(M);
    std::vector<float> taps((size_t)M * NT);
    for (int i = 0; i < M; i++) {
        if (fread(&toks[i], 4, 1, f) != 1) return 1;
        if (fread(taps.data() + (size_t)i * NT, 4, NT, f) != (size_t)NT) return 1;
    }
    fclose(f);
    int P = atoi(argv[3]);
    fprintf(stderr, "dump: %d steps (%d prompt)\n", M, P);

    q27d2::Dflash2 d2;
    d2.load(argv[1]);
    d2.alloc(M + 16);
    float* d_taps;
    cudaMalloc(&d_taps, taps.size() * 4);
    cudaMemcpy(d_taps, taps.data(), taps.size() * 4, cudaMemcpyHostToDevice);

    std::vector<int> pos(M);
    for (int i = 0; i < M; i++) pos[i] = i;
    d2.ingest(d_taps, pos.data(), P, 0);

    int F = P, rounds = 0, produced_sum = 0;
    std::map<int, int> hist;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    while (F + 1 < M) {
        int prop[q27d2::D2_WMAX - 1];
        d2.draft(toks[F], F, K, 0, prop);
        if (show) {
            printf("round F=%d anchor=%d prop:", F, toks[F]);
            for (int j = 0; j < K; j++) printf(" %d", prop[j]);
            printf("\n");
        }
        int al = 0;
        for (int j = 0; j < K && F + 1 + j < M; j++) {
            if (prop[j] == toks[F + 1 + j]) al++;
            else break;
        }
        int produced = al + 1;
        if (produced > M - F - 1) produced = M - F - 1;
        d2.ingest(d_taps + (size_t)F * NT, pos.data() + F, produced, 0);
        F += produced;
        rounds++;
        produced_sum += produced;
        hist[produced]++;
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("[smoke] %d tok, %d rounds, %.2f tok/round, %.2f ms/round (drafter+ingest), hist={",
           produced_sum, rounds, (double)produced_sum / rounds, ms / rounds);
    bool first = true;
    for (auto& kv : hist) {
        printf("%s%d: %d", first ? "" : ", ", kv.first, kv.second);
        first = false;
    }
    printf("}\n");
    return 0;
}
