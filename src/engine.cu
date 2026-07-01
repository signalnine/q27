// q27 engine: greedy decode reference path (M1 = correctness, not speed).
// Usage: q27 <model.q27> --tokens "1,2,3" -n 16 [--ctx 2048] [--dump-logits f.bin]
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "blocks.cuh"
#include "cuda_common.h"
#include "device_model.h"
#include "kernels.cuh"
#include "loader.h"

using q27::DevTensor;
using q27::DType;

// ---- model constants (qwen35 27B, see docs/SPEC.md) ----
static constexpr int N_LAYER = 64;         // main layers; 64 = MTP (not run in M1)
static constexpr int N_EMBD = 5120;
static constexpr int N_FFN = 17408;
static constexpr int N_HEAD = 24, N_KV = 4, HEAD_DIM = 256;
static constexpr int N_ROT = 64;
static constexpr float FREQ_BASE = 1e7f;
static constexpr float EPS = 1e-6f;
static constexpr int GDN_CH = 10240, GDN_V = 6144, GDN_HEADS = 48, GDN_DIM = 128;
static constexpr int VOCAB = 248320;

struct Engine {
    q27::Model model;
    q27::DeviceModel dm;
    int max_ctx;
    bool attn_layer[N_LAYER + 1] = {false};

    // activations (device)
    float *h, *x1, *y, *qg, *kbuf, *vbuf, *attnout, *scratch;
    float *qkv, *convout, *z, *alpha, *betar, *g, *beta, *o, *og;
    float *ffn_g, *ffn_u, *logits;
    int* d_argmax;
    // state
    float* conv_ring[N_LAYER];  // [3][10240] per ssm layer
    float* S[N_LAYER];          // [48][128*128] per ssm layer
    std::vector<float*> kcache, vcache; // per attn layer [max_ctx][4][256]
    std::vector<int> attn_cache_idx;    // layer -> cache slot (-1 if ssm)

    Engine(const std::string& path, int ctx) : model(q27::Model::open(path)), dm(model), max_ctx(ctx) {
        // layer map from metadata json ("attn_layers": [3, 7, ...])
        const std::string& mj = model.meta_json;
        size_t p = mj.find("\"attn_layers\": [");
        if (p == std::string::npos) { fprintf(stderr, "no attn_layers in meta\n"); exit(1); }
        p += strlen("\"attn_layers\": [");
        while (p < mj.size() && mj[p] != ']') {
            int v = atoi(mj.c_str() + p);
            if (v <= N_LAYER) attn_layer[v] = true;
            p = mj.find_first_of(",]", p);
            if (mj[p] == ',') p++;
        }

        auto A = [](void** p, size_t n) { CUDA_CHECK(cudaMalloc(p, n)); };
        A((void**)&h, N_EMBD * 4); A((void**)&x1, N_EMBD * 4); A((void**)&y, N_EMBD * 4);
        A((void**)&qg, 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf, N_KV * HEAD_DIM * 4); A((void**)&vbuf, N_KV * HEAD_DIM * 4);
        A((void**)&attnout, N_HEAD * HEAD_DIM * 4);
        A((void**)&scratch, (size_t)N_HEAD * max_ctx * 4);
        A((void**)&qkv, GDN_CH * 4); A((void**)&convout, GDN_CH * 4); A((void**)&z, GDN_V * 4);
        A((void**)&alpha, GDN_HEADS * 4); A((void**)&betar, GDN_HEADS * 4);
        A((void**)&g, GDN_HEADS * 4); A((void**)&beta, GDN_HEADS * 4);
        A((void**)&o, GDN_V * 4); A((void**)&og, GDN_V * 4);
        A((void**)&ffn_g, N_FFN * 4); A((void**)&ffn_u, N_FFN * 4);
        A((void**)&logits, VOCAB * 4); A((void**)&d_argmax, 4);

        int cache_slot = 0;
        for (int il = 0; il < N_LAYER; il++) {
            if (attn_layer[il]) {
                float *k, *v;
                A((void**)&k, (size_t)max_ctx * N_KV * HEAD_DIM * 4);
                A((void**)&v, (size_t)max_ctx * N_KV * HEAD_DIM * 4);
                kcache.push_back(k); vcache.push_back(v);
                attn_cache_idx.push_back(cache_slot++);
                conv_ring[il] = nullptr; S[il] = nullptr;
            } else {
                A((void**)&conv_ring[il], 3 * GDN_CH * 4);
                CUDA_CHECK(cudaMemset(conv_ring[il], 0, 3 * GDN_CH * 4));
                A((void**)&S[il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4);
                CUDA_CHECK(cudaMemset(S[il], 0, (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4));
                attn_cache_idx.push_back(-1);
            }
        }
        fprintf(stderr, "uploading weights...\n");
        dm.upload_all();
        fprintf(stderr, "resident: %.2f GB\n", dm.bytes_resident() / 1e9);
    }

    const DevTensor& T(int il, const char* leaf) {
        char buf[96];
        snprintf(buf, sizeof buf, "blk.%d.%s", il, leaf);
        return dm.get(buf);
    }

    // y = W x, dispatch on dtype
    void mm(const DevTensor& w, const float* x, float* out) {
        switch (w.dtype) {
            case DType::Q4_G64:
                q27k::gemv_q4((const uint8_t*)w.data, (const __half*)w.scales, x, out, w.rows, w.cols);
                break;
            case DType::Q8_G128:
                q27k::gemv_q8((const int8_t*)w.data, (const __half*)w.scales, x, out, w.rows, w.cols);
                break;
            case DType::F16:
                q27k::gemv_f16((const __half*)w.data, x, out, w.rows, w.cols);
                break;
            default:
                fprintf(stderr, "mm: unsupported dtype\n");
                exit(1);
        }
    }

    void gdn_block(int il, const float* xin, float* yout) {
        mm(T(il, "attn_qkv.weight"), xin, qkv);
        mm(T(il, "attn_gate.weight"), xin, z);
        mm(T(il, "ssm_alpha.weight"), xin, alpha);
        mm(T(il, "ssm_beta.weight"), xin, betar);
        q27k::gdn_gates(alpha, betar, (const float*)T(il, "ssm_a").data,
                        (const float*)T(il, "ssm_dt.bias").data, g, beta, GDN_HEADS);
        q27k::conv_step(conv_ring[il], qkv, (const float*)T(il, "ssm_conv1d.weight").data,
                        convout, GDN_CH);
        q27k::l2norm_heads(convout, 16, GDN_DIM, EPS);           // q heads
        q27k::l2norm_heads(convout + 2048, 16, GDN_DIM, EPS);    // k heads
        q27k::delta_step(S[il], convout, g, beta, o);
        q27k::gated_norm_gdn(o, (const float*)T(il, "ssm_norm.weight").data, z, og, GDN_HEADS,
                             GDN_DIM, EPS);
        mm(T(il, "ssm_out.weight"), og, yout);
    }

    void attn_block(int il, const float* xin, float* yout, int pos) {
        int ci = attn_cache_idx[il];
        mm(T(il, "attn_q.weight"), xin, qg);
        q27k::rmsnorm_heads(qg, (const float*)T(il, "attn_q_norm.weight").data, qg, N_HEAD,
                            HEAD_DIM, 2 * HEAD_DIM, EPS);
        mm(T(il, "attn_k.weight"), xin, kbuf);
        q27k::rmsnorm_heads(kbuf, (const float*)T(il, "attn_k_norm.weight").data, kbuf, N_KV,
                            HEAD_DIM, HEAD_DIM, EPS);
        mm(T(il, "attn_v.weight"), xin, vbuf);
        q27k::rope_neox_partial(qg, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, pos, FREQ_BASE);
        q27k::rope_neox_partial(kbuf, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, pos, FREQ_BASE);
        size_t row = (size_t)N_KV * HEAD_DIM * 4;
        CUDA_CHECK(cudaMemcpy((char*)kcache[ci] + pos * row, kbuf, row, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy((char*)vcache[ci] + pos * row, vbuf, row, cudaMemcpyDeviceToDevice));
        q27k::attn_decode(qg, 2 * HEAD_DIM, kcache[ci], vcache[ci], attnout, scratch, pos + 1,
                          N_HEAD, N_KV, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM));
        q27k::sigmoid_gate_mul(attnout, qg, N_HEAD, HEAD_DIM);
        mm(T(il, "attn_output.weight"), attnout, yout);
    }

    void ffn(int il, const float* xin, float* yout) {
        mm(T(il, "ffn_gate.weight"), xin, ffn_g);
        mm(T(il, "ffn_up.weight"), xin, ffn_u);
        q27k::silu_mul(ffn_g, ffn_u, ffn_g, N_FFN);
        mm(T(il, "ffn_down.weight"), ffn_g, yout);
    }

    // full forward for one token; returns argmax token id, logits stay on device
    int forward(int token, int pos) {
        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed_row_q8((const int8_t*)emb.data, (const __half*)emb.scales, token, N_EMBD, h);
        for (int il = 0; il < N_LAYER; il++) {
            q27k::rmsnorm(h, (const float*)T(il, "attn_norm.weight").data, x1, N_EMBD, EPS);
            if (attn_layer[il]) attn_block(il, x1, y, pos);
            else gdn_block(il, x1, y);
            q27k::add_inplace(h, y, N_EMBD);
            q27k::rmsnorm(h, (const float*)T(il, "post_attention_norm.weight").data, x1, N_EMBD, EPS);
            ffn(il, x1, y);
            q27k::add_inplace(h, y, N_EMBD);
        }
        q27k::rmsnorm(h, (const float*)dm.get("output_norm.weight").data, x1, N_EMBD, EPS);
        mm(dm.get("output.weight"), x1, logits);
        q27k::argmax(logits, VOCAB, d_argmax);
        int next;
        CUDA_CHECK(cudaMemcpy(&next, d_argmax, 4, cudaMemcpyDeviceToHost));
        return next;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.q27 --tokens \"1,2,3\" [-n N] [--ctx C] [--dump-logits f]\n", argv[0]); return 1; }
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
    if (toks.empty()) { fprintf(stderr, "need --tokens\n"); return 1; }

    Engine e(path, ctx);
    int next = 0;
    for (size_t i = 0; i < toks.size(); i++) next = e.forward(toks[i], (int)i);
    printf("prompt processed (%zu tokens). argmax after prompt: %d\n", toks.size(), next);

    if (!dump.empty()) {
        std::vector<float> lg(VOCAB);
        CUDA_CHECK(cudaMemcpy(lg.data(), e.logits, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost));
        FILE* f = fopen(dump.c_str(), "wb");
        fwrite(lg.data(), 4, VOCAB, f);
        fclose(f);
        fprintf(stderr, "logits -> %s\n", dump.c_str());
    }

    printf("generated:");
    int pos = (int)toks.size();
    for (int i = 0; i < n_gen; i++) {
        printf(" %d", next);
        fflush(stdout);
        next = e.forward(next, pos++);
    }
    printf("\n");
    return 0;
}
