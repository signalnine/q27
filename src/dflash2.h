// DFlash2 drafter runtime (Phase 1 bring-up, eager, numerics-first).
// Design: docs/plans/2026-09-06-dflash2-integration.md. Loads the flat pack
// written by tools/dflash2_pack.py; reuses the engine's gemv/rmsnorm/rope
// primitives; adds the two genuinely new kernels (grouped dynamic causal
// conv, bidirectional sliding-window attention). Top-16 + selector walk run
// on the HOST for bring-up -- the whole path is eager and unoptimized by
// intent (correctness gates first; graphs and kernels are Phase 2).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace q27d2 {

// geometry (z-lab/Qwen3.8-27B-DFlash2 checkpoint; asserts at load)
constexpr int D2_H = 5120, D2_TAPS = 5, D2_TAPD = 25600;
constexpr int D2_NH = 32, D2_NKV = 8, D2_HD = 128;
constexpr int D2_QD = D2_NH * D2_HD, D2_KVD = D2_NKV * D2_HD;
constexpr int D2_I = 17408, D2_V = 248320, D2_LAYERS = 5;
constexpr int D2_W = 8, D2_K = 7, D2_MASK = 248070;
constexpr int D2_WINDOW = 2048, D2_CONVK = 2, D2_CONVG = 320; // groups = H/16
constexpr int D2_RANK = 256, D2_TOPK = 16;
constexpr float D2_EPS = 1e-6f, D2_THETA = 1e7f;

struct D2Tensor {
    void* dev = nullptr;     // device copy
    const void* host = nullptr; // pointer into the pack's host buffer
    int dtype = 0;           // 0 = f16, 1 = f32
    int64_t rows = 0, cols = 0, n = 0;
};

struct Dflash2 {
    // ---- weights ----
    std::map<std::string, D2Tensor> w;
    std::vector<char> pack; // host backing (codebooks are walked host-side)
    void load(const char* path);
    const D2Tensor& T(const std::string& name) const { return w.at(name); }
    const __half* f16(const std::string& n) const { return (const __half*)T(n).dev; }
    const float* f32(const std::string& n) const { return (const float*)T(n).dev; }

    // ---- context ring (bring-up: append-only, capacity-capped, no wrap) ----
    int ctx_cap = 0, ctx_n = 0;
    float *ringK[D2_LAYERS] = {}, *ringV[D2_LAYERS] = {}; // [cap][KVD] fp32
    int* d_ring_pos = nullptr;                            // [cap]
    // scratch
    float *s_fc = nullptr, *s_ct = nullptr;
    float *nx = nullptr, *nh1 = nullptr, *ny0 = nullptr; // [8][H]
    float *ndyn = nullptr;                               // [8][2*CONVK*CONVG]
    float *nq = nullptr, *nk = nullptr, *nv = nullptr;   // [8][QD/KVD]
    float *natt = nullptr, *no = nullptr;                // [8][QD], [8][H]
    float *ngate = nullptr, *nup = nullptr;              // [8][I]
    float *nhf = nullptr;                                // [7][H] final-normed mask rows
    float *nlogits = nullptr;                            // [7][V]
    float *nhp = nullptr;                                // [7][RANK]
    float* maskrow = nullptr;                            // cached mask-token embedding
    int* d_pos8 = nullptr;                               // noise positions
    int* d_ing_pos = nullptr;                            // ingest positions (chunk)
    void alloc(int cap);

    // append T committed-token context rows. taps: device [T][TAPD] fp32,
    // positions: host absolute positions (uploaded internally).
    void ingest(const float* d_taps, const int* h_pos, int T, cudaStream_t st);

    // one draft block: anchor (pending) token at anchor_pos; writes the K=7
    // proposed tokens to out. Eager; syncs the stream.
    void draft(int anchor_token, int anchor_pos, int* out, cudaStream_t st);
};

} // namespace q27d2
