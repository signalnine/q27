// q27 engine: CUDA-graph token decode. One captured graph replays per token;
// pos/token/step live on device so the graph is launch-stable, and greedy
// decode chains device-side (argmax writes d_token for the next replay).
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

static constexpr int N_LAYER = 64;
static constexpr int N_EMBD = 5120;
static constexpr int N_FFN = 17408;
static constexpr int N_HEAD = 24, N_KV = 4, HEAD_DIM = 256;
static constexpr int N_ROT = 64;
static constexpr float FREQ_BASE = 1e7f;
static constexpr float EPS = 1e-6f;
static constexpr int GDN_CH = 10240, GDN_V = 6144, GDN_HEADS = 48, GDN_DIM = 128;
static constexpr int VOCAB = 248320;
static constexpr int MAX_GEN_TRACK = 65536;

struct Engine {
    q27::Model model;
    q27::DeviceModel dm;
    int max_ctx;
    bool attn_layer[N_LAYER + 1] = {false};
    cudaStream_t stm;
    cudaGraphExec_t graph_exec = nullptr;

    // activations (device)
    float *h, *x1, *y, *qg, *kbuf, *vbuf, *attnout, *scratch;
    float *qkv, *convout, *z, *alpha, *betar, *g, *beta, *o, *og;
    float *ffn_g, *ffn_u, *logits;
    // device decode state
    int *d_pos, *d_token, *d_step, *d_gen;
    unsigned long long* d_amax;
    // MTP draft head state (stage 1: host-driven acceptance measurement)
    float *h_next, *e_hn, *x_mtp, *mtp_logits;
    float *mtp_k, *mtp_v;
    int *d_pos_m, *d_draft;
    // speculative decode (depth-1): b-token buffers, spare GDN state, batch quant
    float *h_b, *x1_b, *y_b, *qg_b, *kbuf_b, *vbuf_b, *attnout_b;
    float *qkv_b, *convout_b, *z_b, *alpha_b, *betar_b, *g_b, *beta_b, *o_b, *og_b;
    float *ffn_g_b, *ffn_u_b, *logits2, *y2big;
    float *S_spare[N_LAYER], *ring_spare[N_LAYER];
    float *S_spare2[N_LAYER], *ring_spare2[N_LAYER];
    float *h_c, *x1_c, *y_c, *qg_c, *kbuf_c, *vbuf_c, *attnout_c;
    float *qkv_c, *convout_c, *z_c, *alpha_c, *betar_c, *g_c, *beta_c, *o_c, *og_c;
    float *ffn_g_c, *ffn_u_c;
    float *h_next2;
    q27k::XQuant xqC;
    int *d_pos_c, *d_pos_m2, *d_draft2, *d_vc;
    q27k::XQuant xq2[2];
    int *d_pos_a, *d_pos_b, *d_va, *d_vb;
    // GDN state as 3 physical buffers with a cyclic role permutation:
    // role r (0=primary, 1=post-b, 2=post-c) -> physical (r + perm) % 3.
    // accept-1 token -> perm += 1; accept-2 -> perm += 2 (mod 3). 3 captured graphs.
    int perm = 0;
    cudaGraphExec_t spec_graph[3] = {nullptr, nullptr, nullptr};
    float* SBuf(int il, int role) {
        int ph = (role + perm) % 3;
        return ph == 0 ? S[il] : ph == 1 ? S_spare[il] : S_spare2[il];
    }
    float* RBuf(int il, int role) {
        int ph = (role + perm) % 3;
        return ph == 0 ? conv_ring[il] : ph == 1 ? ring_spare[il] : ring_spare2[il];
    }
    q27k::XQuant xq;
    // layer state
    float* conv_ring[N_LAYER];
    float* S[N_LAYER];
    std::vector<float*> kcache, vcache;
    std::vector<int> attn_cache_idx;

    Engine(const std::string& path, int ctx)
        : model(q27::Model::open(path)), dm(model), max_ctx(ctx) {
        CUDA_CHECK(cudaStreamCreate(&stm));
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

        auto A = [](void** pp, size_t n) { CUDA_CHECK(cudaMalloc(pp, n)); };
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
        A((void**)&logits, VOCAB * 4);
        A((void**)&d_pos, 4); A((void**)&d_token, 4); A((void**)&d_step, 4);
        A((void**)&d_gen, MAX_GEN_TRACK * 4);
        A((void**)&d_amax, 8);
        A((void**)&h_next, N_EMBD * 4); A((void**)&e_hn, 2 * N_EMBD * 4);
        A((void**)&x_mtp, N_EMBD * 4); A((void**)&mtp_logits, VOCAB * 4);
        A((void**)&mtp_k, (size_t)max_ctx * N_KV * HEAD_DIM * 4);
        A((void**)&mtp_v, (size_t)max_ctx * N_KV * HEAD_DIM * 4);
        A((void**)&d_pos_m, 4); A((void**)&d_draft, 4);
        A((void**)&h_b, N_EMBD * 4); A((void**)&x1_b, N_EMBD * 4); A((void**)&y_b, N_EMBD * 4);
        A((void**)&qg_b, 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_b, N_KV * HEAD_DIM * 4); A((void**)&vbuf_b, N_KV * HEAD_DIM * 4);
        A((void**)&attnout_b, N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_b, GDN_CH * 4); A((void**)&convout_b, GDN_CH * 4);
        A((void**)&z_b, GDN_V * 4);
        A((void**)&alpha_b, GDN_HEADS * 4); A((void**)&betar_b, GDN_HEADS * 4);
        A((void**)&g_b, GDN_HEADS * 4); A((void**)&beta_b, GDN_HEADS * 4);
        A((void**)&o_b, GDN_V * 4); A((void**)&og_b, GDN_V * 4);
        A((void**)&ffn_g_b, N_FFN * 4); A((void**)&ffn_u_b, N_FFN * 4);
        A((void**)&logits2, 3 * (size_t)VOCAB * 4);
        A((void**)&y2big, 2 * (size_t)N_FFN * 4);
        xq2[0] = q27k::xquant_alloc(N_FFN);
        xq2[1] = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_a, 4); A((void**)&d_pos_b, 4);
        A((void**)&d_va, 4); A((void**)&d_vb, 4);
        A((void**)&h_c, N_EMBD * 4); A((void**)&x1_c, N_EMBD * 4); A((void**)&y_c, N_EMBD * 4);
        A((void**)&qg_c, 2 * N_HEAD * HEAD_DIM * 4);
        A((void**)&kbuf_c, N_KV * HEAD_DIM * 4); A((void**)&vbuf_c, N_KV * HEAD_DIM * 4);
        A((void**)&attnout_c, N_HEAD * HEAD_DIM * 4);
        A((void**)&qkv_c, GDN_CH * 4); A((void**)&convout_c, GDN_CH * 4);
        A((void**)&z_c, GDN_V * 4);
        A((void**)&alpha_c, GDN_HEADS * 4); A((void**)&betar_c, GDN_HEADS * 4);
        A((void**)&g_c, GDN_HEADS * 4); A((void**)&beta_c, GDN_HEADS * 4);
        A((void**)&o_c, GDN_V * 4); A((void**)&og_c, GDN_V * 4);
        A((void**)&ffn_g_c, N_FFN * 4); A((void**)&ffn_u_c, N_FFN * 4);
        A((void**)&h_next2, N_EMBD * 4);
        xqC = q27k::xquant_alloc(N_FFN);
        A((void**)&d_pos_c, 4); A((void**)&d_pos_m2, 4); A((void**)&d_draft2, 4);
        A((void**)&d_vc, 4);
        CUDA_CHECK(cudaMemset(mtp_k, 0, (size_t)max_ctx * N_KV * HEAD_DIM * 4));
        CUDA_CHECK(cudaMemset(mtp_v, 0, (size_t)max_ctx * N_KV * HEAD_DIM * 4));
        CUDA_CHECK(cudaMemset(d_pos, 0, 4));
        CUDA_CHECK(cudaMemset(d_step, 0, 4));
        xq = q27k::xquant_alloc(N_FFN);

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
                A((void**)&S_spare[il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4);
                A((void**)&ring_spare[il], 3 * GDN_CH * 4);
                A((void**)&S_spare2[il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4);
                A((void**)&ring_spare2[il], 3 * GDN_CH * 4);
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

    void qx(const float* x, int cols) { q27k::quantize_x(x, cols, xq, stm); }

    void mm(const DevTensor& w, const float* x, float* out) {
        switch (w.dtype) {
            case DType::Q4_G64:
                q27k::gemv_q4((const uint8_t*)w.data, (const __half*)w.scales, xq, out, w.rows,
                              w.cols, stm);
                break;
            case DType::Q8_G128:
                q27k::gemv_q8((const int8_t*)w.data, (const __half*)w.scales, xq, out, w.rows,
                              w.cols, stm);
                break;
            case DType::F16:
                q27k::gemv_f16((const __half*)w.data, x, out, w.rows, w.cols, stm);
                break;
            default:
                fprintf(stderr, "mm: unsupported dtype\n");
                exit(1);
        }
    }

    void gdn_block(int il, const float* xin, float* yout) {
        qx(xin, N_EMBD);
        mm(T(il, "attn_qkv.weight"), xin, qkv);
        mm(T(il, "attn_gate.weight"), xin, z);
        mm(T(il, "ssm_alpha.weight"), xin, alpha);
        mm(T(il, "ssm_beta.weight"), xin, betar);
        q27k::gdn_gates(alpha, betar, (const float*)T(il, "ssm_a").data,
                        (const float*)T(il, "ssm_dt.bias").data, g, beta, GDN_HEADS, stm);
        q27k::conv_step(conv_ring[il], conv_ring[il], qkv,
                        (const float*)T(il, "ssm_conv1d.weight").data, convout, GDN_CH, stm);
        q27k::l2norm_heads(convout, 16, GDN_DIM, EPS, stm);
        q27k::l2norm_heads(convout + 2048, 16, GDN_DIM, EPS, stm);
        q27k::delta_step(S[il], S[il], convout, g, beta, o, stm);
        q27k::gated_norm_gdn(o, (const float*)T(il, "ssm_norm.weight").data, z, og, GDN_HEADS,
                             GDN_DIM, EPS, stm);
        qx(og, GDN_V);
        mm(T(il, "ssm_out.weight"), og, yout);
    }

    void attn_block(int il, const float* xin, float* yout, float* kc = nullptr,
                    float* vc = nullptr, const int* pos_src = nullptr) {
        if (!kc) {
            int ci = attn_cache_idx[il];
            kc = kcache[ci];
            vc = vcache[ci];
        }
        if (!pos_src) pos_src = d_pos;
        qx(xin, N_EMBD);
        mm(T(il, "attn_q.weight"), xin, qg);
        q27k::rmsnorm_heads(qg, (const float*)T(il, "attn_q_norm.weight").data, qg, N_HEAD,
                            HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        mm(T(il, "attn_k.weight"), xin, kbuf);
        q27k::rmsnorm_heads(kbuf, (const float*)T(il, "attn_k_norm.weight").data, kbuf, N_KV,
                            HEAD_DIM, HEAD_DIM, EPS, stm);
        mm(T(il, "attn_v.weight"), xin, vbuf);
        q27k::rope_neox_partial(qg, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, pos_src, FREQ_BASE, stm);
        q27k::rope_neox_partial(kbuf, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, pos_src, FREQ_BASE, stm);
        q27k::kv_store(kbuf, vbuf, kc, vc, pos_src, N_KV * HEAD_DIM, stm);
        q27k::attn_decode(qg, 2 * HEAD_DIM, kc, vc, attnout, scratch, pos_src,
                          max_ctx, N_HEAD, N_KV, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM), stm);
        q27k::sigmoid_gate_mul(attnout, qg, N_HEAD, HEAD_DIM, stm);
        qx(attnout, N_HEAD * HEAD_DIM);
        mm(T(il, "attn_output.weight"), attnout, yout);
    }

    void ffn(int il, const float* xin, float* yout) {
        qx(xin, N_EMBD);
        mm(T(il, "ffn_gate.weight"), xin, ffn_g);
        mm(T(il, "ffn_up.weight"), xin, ffn_u);
        q27k::silu_mul(ffn_g, ffn_u, ffn_g, N_FFN, stm);
        qx(ffn_g, N_FFN);
        mm(T(il, "ffn_down.weight"), ffn_g, yout);
    }

    // enqueue one full token onto stm (no syncs, no allocations: graph-safe)
    void token_launches() {
        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed_row_q8((const int8_t*)emb.data, (const __half*)emb.scales, d_token, N_EMBD, h,
                           stm);
        for (int il = 0; il < N_LAYER; il++) {
            q27k::rmsnorm(h, (const float*)T(il, "attn_norm.weight").data, x1, N_EMBD, EPS, stm);
            if (attn_layer[il]) attn_block(il, x1, y);
            else gdn_block(il, x1, y);
            q27k::add_inplace(h, y, N_EMBD, stm);
            q27k::rmsnorm(h, (const float*)T(il, "post_attention_norm.weight").data, x1, N_EMBD,
                          EPS, stm);
            ffn(il, x1, y);
            q27k::add_inplace(h, y, N_EMBD, stm);
        }
        q27k::rmsnorm(h, (const float*)dm.get("output_norm.weight").data, x1, N_EMBD, EPS, stm);
        qx(x1, N_EMBD);
        mm(dm.get("output.weight"), x1, logits);
        q27k::argmax(logits, VOCAB, d_token, d_amax, stm); // d_token becomes NEXT token
        q27k::advance(d_pos, d_step, d_gen, d_token, stm); // record + pos++
    }

    // MTP draft head (blk.64, SPEC.md): draft the token AFTER d_token, given
    // h_next = post-output_norm hidden of the current position (*d_pos_m).
    void mtp_forward(const float* h_src = nullptr, const int* tok_src = nullptr,
                     int* draft_dst = nullptr, const int* pos_src = nullptr) {
        if (!h_src) h_src = h_next;
        if (!tok_src) tok_src = d_token;
        if (!draft_dst) draft_dst = d_draft;
        if (!pos_src) pos_src = d_pos_m;
        const int il = 64;
        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed_row_q8((const int8_t*)emb.data, (const __half*)emb.scales, tok_src, N_EMBD,
                           e_hn, stm);
        q27k::rmsnorm(e_hn, (const float*)T(il, "nextn.enorm.weight").data, e_hn, N_EMBD, EPS,
                      stm);
        q27k::rmsnorm(h_src, (const float*)T(il, "nextn.hnorm.weight").data, e_hn + N_EMBD,
                      N_EMBD, EPS, stm);
        qx(e_hn, 2 * N_EMBD);
        mm(T(il, "nextn.eh_proj.weight"), e_hn, x_mtp);

        q27k::rmsnorm(x_mtp, (const float*)T(il, "attn_norm.weight").data, x1, N_EMBD, EPS, stm);
        attn_block(il, x1, y, mtp_k, mtp_v, pos_src);
        q27k::add_inplace(x_mtp, y, N_EMBD, stm);
        q27k::rmsnorm(x_mtp, (const float*)T(il, "post_attention_norm.weight").data, x1, N_EMBD,
                      EPS, stm);
        ffn(il, x1, y);
        q27k::add_inplace(x_mtp, y, N_EMBD, stm);
        q27k::rmsnorm(x_mtp, (const float*)T(il, "nextn.shared_head_norm.weight").data, x1,
                      N_EMBD, EPS, stm);
        qx(x1, N_EMBD);
        mm(dm.get("output.weight"), x1, mtp_logits);
        q27k::argmax(mtp_logits, VOCAB, draft_dst, d_amax, stm);
    }

    void qx3(const float* xa, const float* xb, const float* xc, int cols) {
        q27k::XQ3 q{{xq2[0], xq2[1], xqC}};
        q27k::quantize3({{xa, xb, xc}}, cols, q, stm);
    }
    void mm3(const DevTensor& w, float* out_a, float* out_b, float* out_c) {
        q27k::XQuant qs[3] = {xq2[0], xq2[1], xqC};
        float* const ys[3] = {out_a, out_b, out_c};
        if (w.dtype == DType::Q4_G64)
            q27k::gemv_q4_n((const uint8_t*)w.data, (const __half*)w.scales, qs, 3, ys, w.rows,
                            w.cols, stm);
        else
            q27k::gemv_q8_n((const int8_t*)w.data, (const __half*)w.scales, qs, 3, ys, w.rows,
                            w.cols, stm);
    }

    void gdn_pair(int il) {
        const float eps = EPS;
        qx3(x1, x1_b, x1_c, N_EMBD);
        mm3(T(il, "attn_qkv.weight"), qkv, qkv_b, qkv_c);
        mm3(T(il, "attn_gate.weight"), z, z_b, z_c);
        q27k::gemv_f16((const __half*)T(il, "ssm_alpha.weight").data, x1, alpha, GDN_HEADS,
                       N_EMBD, stm);
        q27k::gemv_f16((const __half*)T(il, "ssm_alpha.weight").data, x1_b, alpha_b, GDN_HEADS,
                       N_EMBD, stm);
        q27k::gemv_f16((const __half*)T(il, "ssm_beta.weight").data, x1, betar, GDN_HEADS,
                       N_EMBD, stm);
        q27k::gemv_f16((const __half*)T(il, "ssm_beta.weight").data, x1_b, betar_b, GDN_HEADS,
                       N_EMBD, stm);
        q27k::gemv_f16((const __half*)T(il, "ssm_alpha.weight").data, x1_c, alpha_c, GDN_HEADS,
                       N_EMBD, stm);
        q27k::gemv_f16((const __half*)T(il, "ssm_beta.weight").data, x1_c, betar_c, GDN_HEADS,
                       N_EMBD, stm);
        const float* sa = (const float*)T(il, "ssm_a").data;
        const float* sdt = (const float*)T(il, "ssm_dt.bias").data;
        q27k::gdn_gates(alpha, betar, sa, sdt, g, beta, GDN_HEADS, stm);
        q27k::gdn_gates(alpha_b, betar_b, sa, sdt, g_b, beta_b, GDN_HEADS, stm);
        q27k::gdn_gates(alpha_c, betar_c, sa, sdt, g_c, beta_c, GDN_HEADS, stm);
        const float* cw = (const float*)T(il, "ssm_conv1d.weight").data;
        q27k::conv_step(RBuf(il, 0), RBuf(il, 0), qkv, cw, convout, GDN_CH, stm);   // a
        q27k::conv_step(RBuf(il, 0), RBuf(il, 1), qkv_b, cw, convout_b, GDN_CH, stm); // b
        q27k::conv_step(RBuf(il, 1), RBuf(il, 2), qkv_c, cw, convout_c, GDN_CH, stm); // c
        q27k::l2norm_heads(convout, 16, GDN_DIM, eps, stm);
        q27k::l2norm_heads(convout + 2048, 16, GDN_DIM, eps, stm);
        q27k::l2norm_heads(convout_b, 16, GDN_DIM, eps, stm);
        q27k::l2norm_heads(convout_b + 2048, 16, GDN_DIM, eps, stm);
        q27k::l2norm_heads(convout_c, 16, GDN_DIM, eps, stm);
        q27k::l2norm_heads(convout_c + 2048, 16, GDN_DIM, eps, stm);
        q27k::delta_step(SBuf(il, 0), SBuf(il, 0), convout, g, beta, o, stm);          // a
        q27k::delta_step(SBuf(il, 0), SBuf(il, 1), convout_b, g_b, beta_b, o_b, stm);   // b
        q27k::delta_step(SBuf(il, 1), SBuf(il, 2), convout_c, g_c, beta_c, o_c, stm);   // c
        const float* nw = (const float*)T(il, "ssm_norm.weight").data;
        q27k::gated_norm_gdn(o, nw, z, og, GDN_HEADS, GDN_DIM, eps, stm);
        q27k::gated_norm_gdn(o_b, nw, z_b, og_b, GDN_HEADS, GDN_DIM, eps, stm);
        q27k::gated_norm_gdn(o_c, nw, z_c, og_c, GDN_HEADS, GDN_DIM, eps, stm);
        qx3(og, og_b, og_c, GDN_V);
        mm3(T(il, "ssm_out.weight"), y, y_b, y_c);
    }

    void attn_pair(int il) {
        int ci = attn_cache_idx[il];
        qx3(x1, x1_b, x1_c, N_EMBD);
        mm3(T(il, "attn_q.weight"), qg, qg_b, qg_c);
        const float* qn = (const float*)T(il, "attn_q_norm.weight").data;
        const float* kn = (const float*)T(il, "attn_k_norm.weight").data;
        q27k::rmsnorm_heads(qg, qn, qg, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        q27k::rmsnorm_heads(qg_b, qn, qg_b, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        q27k::rmsnorm_heads(qg_c, qn, qg_c, N_HEAD, HEAD_DIM, 2 * HEAD_DIM, EPS, stm);
        mm3(T(il, "attn_k.weight"), kbuf, kbuf_b, kbuf_c);
        q27k::rmsnorm_heads(kbuf, kn, kbuf, N_KV, HEAD_DIM, HEAD_DIM, EPS, stm);
        q27k::rmsnorm_heads(kbuf_b, kn, kbuf_b, N_KV, HEAD_DIM, HEAD_DIM, EPS, stm);
        q27k::rmsnorm_heads(kbuf_c, kn, kbuf_c, N_KV, HEAD_DIM, HEAD_DIM, EPS, stm);
        mm3(T(il, "attn_v.weight"), vbuf, vbuf_b, vbuf_c);
        q27k::rope_neox_partial(qg, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, d_pos_a, FREQ_BASE, stm);
        q27k::rope_neox_partial(kbuf, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, d_pos_a, FREQ_BASE, stm);
        q27k::rope_neox_partial(qg_b, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, d_pos_b, FREQ_BASE,
                                stm);
        q27k::rope_neox_partial(kbuf_b, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, d_pos_b, FREQ_BASE, stm);
        q27k::rope_neox_partial(qg_c, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, d_pos_c, FREQ_BASE,
                                stm);
        q27k::rope_neox_partial(kbuf_c, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, d_pos_c, FREQ_BASE, stm);
        float kq = 1.0f / sqrtf((float)HEAD_DIM);
        q27k::kv_store(kbuf, vbuf, kcache[ci], vcache[ci], d_pos_a, N_KV * HEAD_DIM, stm);
        q27k::attn_decode(qg, 2 * HEAD_DIM, kcache[ci], vcache[ci], attnout, scratch, d_pos_a,
                          max_ctx, N_HEAD, N_KV, HEAD_DIM, kq, stm);
        q27k::kv_store(kbuf_b, vbuf_b, kcache[ci], vcache[ci], d_pos_b, N_KV * HEAD_DIM, stm);
        q27k::attn_decode(qg_b, 2 * HEAD_DIM, kcache[ci], vcache[ci], attnout_b, scratch, d_pos_b,
                          max_ctx, N_HEAD, N_KV, HEAD_DIM, kq, stm);
        q27k::kv_store(kbuf_c, vbuf_c, kcache[ci], vcache[ci], d_pos_c, N_KV * HEAD_DIM, stm);
        q27k::attn_decode(qg_c, 2 * HEAD_DIM, kcache[ci], vcache[ci], attnout_c, scratch, d_pos_c,
                          max_ctx, N_HEAD, N_KV, HEAD_DIM, kq, stm);
        q27k::sigmoid_gate_mul(attnout, qg, N_HEAD, HEAD_DIM, stm);
        q27k::sigmoid_gate_mul(attnout_b, qg_b, N_HEAD, HEAD_DIM, stm);
        q27k::sigmoid_gate_mul(attnout_c, qg_c, N_HEAD, HEAD_DIM, stm);
        qx3(attnout, attnout_b, attnout_c, N_HEAD * HEAD_DIM);
        mm3(T(il, "attn_output.weight"), y, y_b, y_c);
    }

    void ffn_pair(int il) {
        qx3(x1, x1_b, x1_c, N_EMBD);
        mm3(T(il, "ffn_gate.weight"), ffn_g, ffn_g_b, ffn_g_c);
        mm3(T(il, "ffn_up.weight"), ffn_u, ffn_u_b, ffn_u_c);
        q27k::silu_mul3({{ffn_g, ffn_g_b, ffn_g_c}}, {{ffn_u, ffn_u_b, ffn_u_c}}, N_FFN, stm);
        qx3(ffn_g, ffn_g_b, ffn_g_c, N_FFN);
        mm3(T(il, "ffn_down.weight"), y, y_b, y_c);
    }

    // launch sequence for one speculative round (graph-capturable: all state
    // through device memory, pointers fixed for a given parity)
    void spec_round_launches() {
        // draft 1: (h_next, embed(t1)) at pos_m -> d_draft; MTP's own post-head-norm
        // hidden (x1) chains into draft 2 at pos_m2 -> d_draft2 (also fills MTP KV)
        mtp_forward(h_next, d_token, d_draft, d_pos_m);
        CUDA_CHECK(cudaMemcpyAsync(h_next2, x1, N_EMBD * 4, cudaMemcpyDeviceToDevice, stm));
        mtp_forward(h_next2, d_draft, d_draft2, d_pos_m2);

        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed_row_q8((const int8_t*)emb.data, (const __half*)emb.scales, d_token, N_EMBD, h,
                           stm);
        q27k::embed_row_q8((const int8_t*)emb.data, (const __half*)emb.scales, d_draft, N_EMBD,
                           h_b, stm);
        q27k::embed_row_q8((const int8_t*)emb.data, (const __half*)emb.scales, d_draft2, N_EMBD,
                           h_c, stm);
        q27k::CP3 Hc{{h, h_b, h_c}}, Yc{{y, y_b, y_c}};
        q27k::P3 Hm{{h, h_b, h_c}}, X1m{{x1, x1_b, x1_c}};
        for (int il = 0; il < N_LAYER; il++) {
            const float* an = (const float*)T(il, "attn_norm.weight").data;
            q27k::rmsnorm3(Hc, an, X1m, N_EMBD, EPS, stm);
            if (attn_layer[il]) attn_pair(il);
            else gdn_pair(il);
            q27k::add3(Hm, Yc, N_EMBD, stm);
            const float* pn = (const float*)T(il, "post_attention_norm.weight").data;
            q27k::rmsnorm3(Hc, pn, X1m, N_EMBD, EPS, stm);
            ffn_pair(il);
            q27k::add3(Hm, Yc, N_EMBD, stm);
        }
        const float* on = (const float*)dm.get("output_norm.weight").data;
        q27k::rmsnorm3(Hc, on, X1m, N_EMBD, EPS, stm);
        qx3(x1, x1_b, x1_c, N_EMBD);
        mm3(dm.get("output.weight"), logits2, logits2 + VOCAB, logits2 + 2 * (size_t)VOCAB);
        q27k::argmax(logits2, VOCAB, d_va, d_amax, stm);
        q27k::argmax(logits2 + VOCAB, VOCAB, d_vb, d_amax, stm);
        q27k::argmax(logits2 + 2 * (size_t)VOCAB, VOCAB, d_vc, d_amax, stm);
    }

    void build_spec_graphs() {
        // one warm (executing) round to initialize lazy CUDA state, then reset
        int z0 = 0, z1 = 1, z2 = 2;
        CUDA_CHECK(cudaMemcpyAsync(d_pos_a, &z0, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemcpyAsync(d_pos_b, &z1, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemcpyAsync(d_pos_c, &z2, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemcpyAsync(d_pos_m, &z0, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemcpyAsync(d_pos_m2, &z1, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemcpyAsync(d_token, &z0, 4, cudaMemcpyHostToDevice, stm));
        spec_round_launches();
        CUDA_CHECK(cudaStreamSynchronize(stm));
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                size_t sb = (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4;
                CUDA_CHECK(cudaMemset(S[il], 0, sb));
                CUDA_CHECK(cudaMemset(S_spare[il], 0, sb));
                CUDA_CHECK(cudaMemset(conv_ring[il], 0, 3 * GDN_CH * 4));
                CUDA_CHECK(cudaMemset(ring_spare[il], 0, 3 * GDN_CH * 4));
                CUDA_CHECK(cudaMemset(S_spare2[il], 0, sb));
                CUDA_CHECK(cudaMemset(ring_spare2[il], 0, 3 * GDN_CH * 4));
            }
        CUDA_CHECK(cudaMemset(mtp_k, 0, (size_t)max_ctx * N_KV * HEAD_DIM * 4));
        CUDA_CHECK(cudaMemset(mtp_v, 0, (size_t)max_ctx * N_KV * HEAD_DIM * 4));
        // capture all 3 cyclic permutations (capture records; does not execute)
        for (int p = 0; p < 3; p++) {
            perm = p;
            cudaGraph_t gr;
            CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
            spec_round_launches();
            CUDA_CHECK(cudaStreamEndCapture(stm, &gr));
            CUDA_CHECK(cudaGraphInstantiate(&spec_graph[p], gr, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphDestroy(gr));
        }
        perm = 0;
        fprintf(stderr, "spec graphs captured (3 perms, depth-2)\n");
    }

    // one speculative round (depth 2); returns tokens emitted (1..3)
    int spec_round(int P, int* emit) {
        int pa = P + 1, pb = P + 2, pc = P + 3;
        CUDA_CHECK(cudaMemcpyAsync(d_pos_a, &pa, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemcpyAsync(d_pos_b, &pb, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemcpyAsync(d_pos_c, &pc, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemcpyAsync(d_pos_m, &pa, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemcpyAsync(d_pos_m2, &pb, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaGraphLaunch(spec_graph[perm], stm));

        int va, vb, vc, dr1, dr2, t1;
        CUDA_CHECK(cudaMemcpyAsync(&va, d_va, 4, cudaMemcpyDeviceToHost, stm));
        CUDA_CHECK(cudaMemcpyAsync(&vb, d_vb, 4, cudaMemcpyDeviceToHost, stm));
        CUDA_CHECK(cudaMemcpyAsync(&vc, d_vc, 4, cudaMemcpyDeviceToHost, stm));
        CUDA_CHECK(cudaMemcpyAsync(&dr1, d_draft, 4, cudaMemcpyDeviceToHost, stm));
        CUDA_CHECK(cudaMemcpyAsync(&dr2, d_draft2, 4, cudaMemcpyDeviceToHost, stm));
        CUDA_CHECK(cudaMemcpyAsync(&t1, d_token, 4, cudaMemcpyDeviceToHost, stm));
        CUDA_CHECK(cudaStreamSynchronize(stm));

        if (va == dr1 && vb == dr2) { // both drafts verified
            perm = (perm + 2) % 3;
            emit[0] = t1; emit[1] = dr1; emit[2] = dr2;
            CUDA_CHECK(cudaMemcpyAsync(d_token, &vc, 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemcpyAsync(h_next, x1_c, N_EMBD * 4, cudaMemcpyDeviceToDevice, stm));
            return 3;
        }
        if (va == dr1) { // first draft verified
            perm = (perm + 1) % 3;
            emit[0] = t1; emit[1] = dr1;
            CUDA_CHECK(cudaMemcpyAsync(d_token, &vb, 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemcpyAsync(h_next, x1_b, N_EMBD * 4, cudaMemcpyDeviceToDevice, stm));
            return 2;
        }
        emit[0] = t1;
        CUDA_CHECK(cudaMemcpyAsync(d_token, &va, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaMemcpyAsync(h_next, x1, N_EMBD * 4, cudaMemcpyDeviceToDevice, stm));
        return 1;
    }

    void build_graph() {
        // warm run (outside capture) so lazy CUDA state is initialized
        int zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_token, &zero, 4, cudaMemcpyHostToDevice, stm));
        token_launches();
        CUDA_CHECK(cudaStreamSynchronize(stm));
        // reset state mutated by the warm run
        CUDA_CHECK(cudaMemset(d_pos, 0, 4));
        CUDA_CHECK(cudaMemset(d_step, 0, 4));
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                CUDA_CHECK(cudaMemset(conv_ring[il], 0, 3 * GDN_CH * 4));
                CUDA_CHECK(cudaMemset(S[il], 0, (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4));
            }

        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
        token_launches();
        CUDA_CHECK(cudaStreamEndCapture(stm, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
        fprintf(stderr, "token graph captured\n");
    }

    // feed one known token (prompt phase): set d_token, replay graph
    void step_with(int token) {
        CUDA_CHECK(cudaMemcpyAsync(d_token, &token, 4, cudaMemcpyHostToDevice, stm));
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stm));
    }
    // generation step: d_token already holds the model's own prediction
    void step_free() { CUDA_CHECK(cudaGraphLaunch(graph_exec, stm)); }
};

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
    bool mtp_stats = false, spec = false;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--mtp")) mtp_stats = true;
        if (!strcmp(argv[i], "--spec")) { spec = true; mtp_stats = true; } // spec needs MTP warmup
    }
    if (toks.empty()) { fprintf(stderr, "need --tokens\n"); return 1; }

    Engine e(path, ctx);
    e.build_graph();
    if (spec) e.build_spec_graphs();

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
        int total_emitted = 0, rounds = 0;
        while ((int)out.size() < n_gen) {
            int em[3];
            int n = e.spec_round(P, em);
            for (int k = 0; k < n; k++) out.push_back(em[k]);
            rounds++;
            total_emitted += n;
            P += n;
        }
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
