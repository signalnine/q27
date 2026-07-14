// round_weight_cost -- how much of a verify round is the WEIGHT GEMV?
//
// The GEMM-verify pivot is priced entirely on this number: if the batched
// weight GEMV is most of a wide round, a weight path that is flat in W is
// the whole ballgame; if it is a third of it, the pivot is a footnote.
//
// Replays the round's EXACT mm5 call sequence (every layer, every weight, in
// order, at lane count N) and times only the GEMVs -- no GDN, no attention,
// no finish. Also reports the bytes moved, so the achieved GB/s and the
// counterfactual "same bytes at the MMA GEMM's measured rate" fall out.
//
// Usage: round_weight_cost model.q27 [N ...]
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/device_model.h"
#include "../src/kernels.cuh"
#include "../src/loader.h"
#include "../src/vgemm.cuh"

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t e = (x);                                                   \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static constexpr int N_LAYER = 64, N_EMBD = 5120, N_FFN = 17408;
static constexpr int GDN_CH = 10240, GDN_V = 6144;
static constexpr int N_HEAD = 24, N_KV = 4, HEAD_DIM = 256;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.q27 [N ...]\n", argv[0]); return 1; }
    q27::Model m = q27::Model::open(argv[1]);
    q27::DeviceModel dm(m);
    dm.upload_all();

    // which layers are attention (the rest are GDN) -- read it off the tensors
    std::vector<bool> is_attn(N_LAYER, false);
    for (int il = 0; il < N_LAYER; il++) {
        char b[96];
        snprintf(b, sizeof b, "blk.%d.attn_q.weight", il);
        is_attn[il] = m.find(b) != nullptr;
    }

    // the round's weight sequence, in engine order (engine.cuh gdn_pair/attn_pair/ffn_pair)
    std::vector<const q27::DevTensor*> seq;
    size_t bytes = 0;
    auto add = [&](const char* fmt, int il) {
        char b[96];
        snprintf(b, sizeof b, fmt, il);
        if (!m.find(b)) return;
        const q27::DevTensor& t = dm.get(b);
        seq.push_back(&t);
        const q27::Tensor& mt = m.get(b);
        int64_t r = mt.rows(), c = mt.cols();
        bytes += (t.dtype == q27::DType::Q4_G64) ? (size_t)r * c / 2 : (size_t)r * c;
    };
    for (int il = 0; il < N_LAYER; il++) {
        if (is_attn[il]) {
            add("blk.%d.attn_q.weight", il); add("blk.%d.attn_k.weight", il);
            add("blk.%d.attn_v.weight", il); add("blk.%d.attn_output.weight", il);
        } else {
            add("blk.%d.attn_qkv.weight", il); add("blk.%d.attn_gate.weight", il);
            add("blk.%d.ssm_out.weight", il);
        }
        add("blk.%d.ffn_gate.weight", il); add("blk.%d.ffn_up.weight", il);
        add("blk.%d.ffn_down.weight", il);
    }
    // vocab head: the serving default is --fast-head (Q4)
    const char* vh = m.find("output_q4.weight") ? "output_q4.weight" : "output.weight";
    {
        const q27::DevTensor& t = dm.get(vh);
        seq.push_back(&t);
        const q27::Tensor& mt = m.get(vh);
        bytes += (t.dtype == q27::DType::Q4_G64) ? (size_t)mt.rows() * mt.cols() / 2
                                                 : (size_t)mt.rows() * mt.cols();
    }
    printf("round weight sequence: %zu GEMV calls, %.2f GB of weights (head=%s)\n\n",
           seq.size(), bytes / 1e9, vh);

    // lanes: 16 quantized activation columns, widest row we ever feed (N_FFN)
    q27::DevTensor* biggest = nullptr;
    q27k::XQuant qs[16];
    float* ys[16];
    float* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)N_FFN * 4));
    std::vector<float> hx(N_FFN);
    for (int i = 0; i < N_FFN; i++) hx[i] = (float)((i * 2654435761u) % 1000) / 500.f - 1.f;
    CUDA_CHECK(cudaMemcpy(d_x, hx.data(), (size_t)N_FFN * 4, cudaMemcpyHostToDevice));
    for (int i = 0; i < 16; i++) {
        qs[i] = q27k::xquant_alloc(N_FFN);
        q27k::quantize_x(d_x, N_FFN, qs[i]);
        CUDA_CHECK(cudaMalloc(&ys[i], (size_t)248320 * 4));
    }
    (void)biggest;

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    printf("%3s %11s %11s %10s   %s\n", "N", "weight ms", "GB/s", "vs SOL", "round share (see note)");
    for (int a = 2; a < argc; a++) {
        int N = atoi(argv[a]);
        auto one = [&]() {
            for (const q27::DevTensor* t : seq) {
                if (t->dtype == q27::DType::Q4_G64)
                    q27k::gemv_q4_n((const uint8_t*)t->data, (const __half*)t->scales, qs, N, ys,
                                    t->rows, t->cols, 0);
                else
                    q27k::gemv_q8_n((const int8_t*)t->data, (const __half*)t->scales, qs, N, ys,
                                    t->rows, t->cols, 0);
            }
        };
        one();
        CUDA_CHECK(cudaDeviceSynchronize());
        const int REP = 5;
        CUDA_CHECK(cudaEventRecord(e0));
        for (int r = 0; r < REP; r++) one();
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        ms /= REP;
        double gbs = bytes / 1e9 / (ms / 1000.0);
        printf("%3d %11.2f %11.1f %9.0f%%\n", N, ms, gbs, 100.0 * gbs / 1453.0);
    }
    printf("\n(theoretical counterfactual: %.2f GB at the benched 1075 GB/s = %.2f ms)\n",
           bytes / 1e9, bytes / 1e9 / 1075.0 * 1000.0);

    // ---- the PROMOTED kernel (src/vgemm.cu), over the SAME 401-weight sequence.
    // This is the P1 perf gate: the spike measured a FORK, and it skipped the Q8
    // weights. src/vgemm has the Q8 leg, the deterministic two-pass reduce, and
    // the KB-align/trim fixes -- so re-measure, do not assume.
    q27k::XLanes X{};
    q27k::YLanes Y{};
    float* d_ybig[16];
    for (int t = 0; t < 16; t++) {
        X.nat[t] = qs[t].nat;
        X.xs[t] = qs[t].scale;
        CUDA_CHECK(cudaMalloc(&d_ybig[t], (size_t)248320 * 4));
        Y.y[t] = d_ybig[t];
    }
    size_t wsb = q27k::vgemm_ws_bytes(seq.data(), (int)seq.size());
    float* d_ws;
    CUDA_CHECK(cudaMalloc(&d_ws, wsb));
    printf("\n== src/vgemm (PROMOTED, Q4+Q8, deterministic reduce) over the same round ==\n");
    printf("   workspace %.2f MB\n", wsb / 1e6);
    printf("%3s %11s %11s %10s   %s\n", "T", "weight ms", "GB/s", "vs SOL", "vs GEMV");
    for (int a2 = 2; a2 < argc; a2++) {
        int T = atoi(argv[a2]);
        auto oneg = [&]() {
            for (const q27::DevTensor* t : seq)
                if (!q27k::vgemm_verify(*t, X, Y, d_ws, T, 0)) {
                    // ineligible -> the engine falls back to the GEMV; time that too
                    if (t->dtype == q27::DType::Q4_G64)
                        q27k::gemv_q4_n((const uint8_t*)t->data, (const __half*)t->scales, qs, T,
                                        ys, t->rows, t->cols, 0);
                    else
                        q27k::gemv_q8_n((const int8_t*)t->data, (const __half*)t->scales, qs, T,
                                        ys, t->rows, t->cols, 0);
                }
        };
        oneg();
        CUDA_CHECK(cudaDeviceSynchronize());
        const int R = 5;
        CUDA_CHECK(cudaEventRecord(e0));
        for (int r = 0; r < R; r++) oneg();
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        ms /= R;
        double gbs = bytes / 1e9 / (ms / 1000.0);
        printf("%3d %11.2f %11.1f %9.0f%%\n", T, ms, gbs, 100.0 * gbs / 1453.0);
    }
    return 0;
}
