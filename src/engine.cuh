// q27 Engine: qwen35 hybrid forward + MTP speculative decode. Header-only
// (all methods inline) so both the CLI and the server can embed it.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "blocks.cuh"
#include "spec3.cuh"
#include "prefill.cuh"
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
    __half *mtp_k, *mtp_v;
    int *d_pos_m, *d_draft;
    // speculative decode (depth-1): b-token buffers, spare GDN state, batch quant
    float *h_b, *x1_b, *y_b, *qg_b, *kbuf_b, *vbuf_b, *attnout_b;
    float *qkv_b, *convout_b, *z_b, *alpha_b, *betar_b, *g_b, *beta_b, *o_b, *og_b;
    float *ffn_g_b, *ffn_u_b, *logits2, *y2big;
    float *S_spare[N_LAYER], *ring_spare[N_LAYER];
    float *S_spare2[N_LAYER], *ring_spare2[N_LAYER];
    float *S_spare3[N_LAYER], *ring_spare3[N_LAYER];
    float *h_c, *x1_c, *y_c, *qg_c, *kbuf_c, *vbuf_c, *attnout_c;
    float *qkv_c, *convout_c, *z_c, *alpha_c, *betar_c, *g_c, *beta_c, *o_c, *og_c;
    float *ffn_g_c, *ffn_u_c;
    float *h_next2;
    q27k::XQuant xqC;
    int *d_pos_c, *d_pos_m2, *d_draft2, *d_vc;
    int *d_P, *d_outcome;
    q27k::XQuant xq2[2];
    int *d_pos_a, *d_pos_b, *d_va, *d_vb;
    // GDN state as 4 physical buffers with a cyclic role permutation:
    // role r (0=primary, 1=post-b, 2=post-c, 3=post-d) -> physical (r+perm)%4.
    // accept n tokens -> perm += n-1 (mod 4). One captured graph per perm.
    // Invariant: role 0 always holds the last-committed state.
    bool fast_head = false; // opt-in: Q4 head for verify too (output may differ)
    bool batched_prefill = true;

    // ---- batched prefill (M6) ----
    static constexpr int PF_T = 256;  // chunk size
    static constexpr int PF_SB = 32;  // attention sub-batch (scratch rows)
    int* d_prompt = nullptr;          // whole prompt on device
    int d_prompt_cap = 0;
    float *hT, *x1T, *yT, *qkvT, *convT, *zT, *oT, *ogT, *qgT, *kT, *vT, *attnT;
    float *alphaT, *betarT, *gT, *betaT, *ffnGT, *ffnUT, *embT, *ehnT, *xmtpT;
    float* pf_scratch;
    q27k::XQuant xqT;

    // ---- prefix cache (M6.5): snapshot of GDN state + conv rings taken right
    // after prefill (perm==0), keyed by the prompt tokens it covers. Attention
    // and MTP KV rows are append-only during generation, so prefix rows stay
    // valid; only the recurrent state needs snapshot/restore.
    float* S_snap[N_LAYER] = {};
    float* ring_snap[N_LAYER] = {};
    std::vector<int> snap_toks;
    bool have_snap = false;
    int perm = 0;
    cudaGraphExec_t spec_graph[4] = {nullptr, nullptr, nullptr, nullptr};
    float* SBuf(int il, int role) {
        int ph = (role + perm) % 4;
        return ph == 0 ? S[il] : ph == 1 ? S_spare[il] : ph == 2 ? S_spare2[il]
                                                                 : S_spare3[il];
    }
    float* RBuf(int il, int role) {
        int ph = (role + perm) % 4;
        return ph == 0 ? conv_ring[il]
               : ph == 1 ? ring_spare[il]
               : ph == 2 ? ring_spare2[il]
                         : ring_spare3[il];
    }
    q27k::XQuant xq;
    // layer state
    float* conv_ring[N_LAYER];
    float* S[N_LAYER];
    std::vector<__half*> kcache, vcache;
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
        // flash-decode split-K partials: ntok * heads * FD_NS * FD_ST floats,
        // independent of ctx (sized for 4 lanes; was 3*N_HEAD*max_ctx, which
        // under-allocates whenever max_ctx < FD_NS*FD_ST = 4128)
        A((void**)&scratch, 4 * (size_t)N_HEAD * q27k::FD_NS * q27k::FD_ST * 4);
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
        A((void**)&mtp_k, (size_t)max_ctx * N_KV * HEAD_DIM * 2);
        A((void**)&mtp_v, (size_t)max_ctx * N_KV * HEAD_DIM * 2);
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
        A((void**)&d_P, 4); A((void**)&d_outcome, 16);
        CUDA_CHECK(cudaMemset(mtp_k, 0, (size_t)max_ctx * N_KV * HEAD_DIM * 2));
        CUDA_CHECK(cudaMemset(mtp_v, 0, (size_t)max_ctx * N_KV * HEAD_DIM * 2));
        CUDA_CHECK(cudaMemset(d_pos, 0, 4));
        CUDA_CHECK(cudaMemset(d_step, 0, 4));
        xq = q27k::xquant_alloc(N_FFN);
        // batched prefill buffers (~130MB) + attention scratch
        auto fal = [](size_t n) { float* p; CUDA_CHECK(cudaMalloc((void**)&p, n * 4)); return p; };
        hT = fal((size_t)PF_T * N_EMBD); x1T = fal((size_t)PF_T * N_EMBD);
        yT = fal((size_t)PF_T * N_EMBD); qkvT = fal((size_t)PF_T * GDN_CH);
        convT = fal((size_t)PF_T * GDN_CH); zT = fal((size_t)PF_T * GDN_V);
        oT = fal((size_t)PF_T * GDN_V); ogT = fal((size_t)PF_T * GDN_V);
        qgT = fal((size_t)PF_T * N_HEAD * 2 * HEAD_DIM);
        kT = fal((size_t)PF_T * N_KV * HEAD_DIM); vT = fal((size_t)PF_T * N_KV * HEAD_DIM);
        attnT = fal((size_t)PF_T * N_HEAD * HEAD_DIM);
        alphaT = fal((size_t)PF_T * GDN_HEADS); betarT = fal((size_t)PF_T * GDN_HEADS);
        gT = fal((size_t)PF_T * GDN_HEADS); betaT = fal((size_t)PF_T * GDN_HEADS);
        ffnGT = fal((size_t)PF_T * N_FFN); ffnUT = fal((size_t)PF_T * N_FFN);
        embT = fal((size_t)PF_T * N_EMBD); ehnT = fal((size_t)PF_T * 2 * N_EMBD);
        xmtpT = fal((size_t)PF_T * N_EMBD);
        pf_scratch = fal((size_t)PF_SB * N_HEAD * max_ctx);
        xqT = q27k::xquant_alloc((size_t)PF_T * N_FFN);
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                CUDA_CHECK(cudaMalloc((void**)&S_snap[il],
                                      (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4));
                CUDA_CHECK(cudaMalloc((void**)&ring_snap[il], 3 * GDN_CH * 4));
            }

        int cache_slot = 0;
        for (int il = 0; il < N_LAYER; il++) {
            if (attn_layer[il]) {
                __half *k, *v;
                A((void**)&k, (size_t)max_ctx * N_KV * HEAD_DIM * 2);
                A((void**)&v, (size_t)max_ctx * N_KV * HEAD_DIM * 2);
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
                A((void**)&S_spare3[il], (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4);
                A((void**)&ring_spare3[il], 3 * GDN_CH * 4);
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
    const DevTensor& T2(int il, const char* leaf) { return T(il, leaf); }

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

    void attn_block(int il, const float* xin, float* yout, __half* kc = nullptr,
                    __half* vc = nullptr, const int* pos_src = nullptr) {
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
        // drafts use the Q4 head copy when present (verify keeps the Q8 head,
        // so output remains exactly the faithful model's greedy text)
        const DevTensor* head = dm.model_has("output_q4.weight")
                                    ? &dm.get("output_q4.weight")
                                    : &dm.get("output.weight");
        mm(*head, x1, mtp_logits);
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
        q27k::gemv_f16_3((const __half*)T(il, "ssm_alpha.weight").data, {{x1, x1_b, x1_c}},
                         {{alpha, alpha_b, alpha_c}}, GDN_HEADS, N_EMBD, stm);
        q27k::gemv_f16_3((const __half*)T(il, "ssm_beta.weight").data, {{x1, x1_b, x1_c}},
                         {{betar, betar_b, betar_c}}, GDN_HEADS, N_EMBD, stm);
        const float* sa = (const float*)T(il, "ssm_a").data;
        const float* sdt = (const float*)T(il, "ssm_dt.bias").data;
        q27k::gdn_gates3({{alpha, alpha_b, alpha_c}}, {{betar, betar_b, betar_c}}, sa, sdt,
                         {{g, g_b, g_c}}, {{beta, beta_b, beta_c}}, GDN_HEADS, stm);
        const float* cw = (const float*)T(il, "ssm_conv1d.weight").data;
        q27k::conv_step(RBuf(il, 0), RBuf(il, 0), qkv, cw, convout, GDN_CH, stm);   // a
        q27k::conv_step(RBuf(il, 0), RBuf(il, 1), qkv_b, cw, convout_b, GDN_CH, stm); // b
        q27k::conv_step(RBuf(il, 1), RBuf(il, 2), qkv_c, cw, convout_c, GDN_CH, stm); // c
        // q||k are contiguous (offsets 0 and 2048): 32 heads in one merged call
        q27k::l2norm3({{convout, convout_b, convout_c}}, 32, GDN_DIM, eps, stm);
        q27k::delta_step(SBuf(il, 0), SBuf(il, 0), convout, g, beta, o, stm);          // a
        q27k::delta_step(SBuf(il, 0), SBuf(il, 1), convout_b, g_b, beta_b, o_b, stm);   // b
        q27k::delta_step(SBuf(il, 1), SBuf(il, 2), convout_c, g_c, beta_c, o_c, stm);   // c
        const float* nw = (const float*)T(il, "ssm_norm.weight").data;
        q27k::gated_norm3({{o, o_b, o_c}}, nw, {{z, z_b, z_c}}, {{og, og_b, og_c}}, GDN_HEADS,
                          GDN_DIM, eps, stm);
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
        q27k::IP3 P{{d_pos_a, d_pos_b, d_pos_c}};
        q27k::rope3({{qg, qg_b, qg_c}}, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, P, FREQ_BASE, stm);
        q27k::rope3({{kbuf, kbuf_b, kbuf_c}}, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, P, FREQ_BASE, stm);
        float kq = 1.0f / sqrtf((float)HEAD_DIM);
        // store all 3 first (disjoint slots); each token's attention only reads
        // cache[0 .. its own pos], so later tokens' entries are invisible to earlier ones
        q27k::kv_store3({{kbuf, kbuf_b, kbuf_c}}, {{vbuf, vbuf_b, vbuf_c}}, kcache[ci],
                        vcache[ci], P, N_KV * HEAD_DIM, stm);
        q27k::attn_decode3({{qg, qg_b, qg_c}}, 2 * HEAD_DIM, kcache[ci], vcache[ci],
                           {{attnout, attnout_b, attnout_c}}, scratch, P, max_ctx, N_HEAD, N_KV,
                           HEAD_DIM, kq, stm);
        q27k::sigmoid_gate3({{attnout, attnout_b, attnout_c}}, {{qg, qg_b, qg_c}}, N_HEAD,
                            HEAD_DIM, stm);
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
        q27k::prep_round(d_P, d_token, d_pos_a, d_pos_b, d_pos_c, d_pos_m, d_pos_m2, d_outcome,
                         stm);
        // draft 1: (h_next, embed(t1)) at pos_m -> d_draft; MTP's own post-head-norm
        // hidden (x1) chains into draft 2 at pos_m2 -> d_draft2 (also fills MTP KV)
        mtp_forward(h_next, d_token, d_draft, d_pos_m);
        CUDA_CHECK(cudaMemcpyAsync(h_next2, x1, N_EMBD * 4, cudaMemcpyDeviceToDevice, stm));
        mtp_forward(h_next2, d_draft, d_draft2, d_pos_m2);

        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed3((const int8_t*)emb.data, (const __half*)emb.scales,
                     {{d_token, d_draft, d_draft2}}, N_EMBD, {{h, h_b, h_c}}, stm);
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
        const char* vh = (fast_head && dm.model_has("output_q4.weight")) ? "output_q4.weight"
                                                                          : "output.weight";
        mm3(dm.get(vh), logits2, logits2 + VOCAB, logits2 + 2 * (size_t)VOCAB);
        q27k::argmax(logits2, VOCAB, d_va, d_amax, stm);
        q27k::argmax(logits2 + VOCAB, VOCAB, d_vb, d_amax, stm);
        q27k::argmax(logits2 + 2 * (size_t)VOCAB, VOCAB, d_vc, d_amax, stm);
        q27k::finish_round(d_P, d_token, d_draft, d_draft2, d_va, d_vb, d_vc, x1, x1_b, x1_c,
                           h_next, d_outcome, N_EMBD, stm);
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
        CUDA_CHECK(cudaMemset(d_P, 0, 4));
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
                CUDA_CHECK(cudaMemset(S_spare3[il], 0, sb));
                CUDA_CHECK(cudaMemset(ring_spare3[il], 0, 3 * GDN_CH * 4));
            }
        CUDA_CHECK(cudaMemset(mtp_k, 0, (size_t)max_ctx * N_KV * HEAD_DIM * 2));
        CUDA_CHECK(cudaMemset(mtp_v, 0, (size_t)max_ctx * N_KV * HEAD_DIM * 2));
        // capture all 4 cyclic permutations (capture records; does not execute)
        for (int p = 0; p < 4; p++) {
            perm = p;
            cudaGraph_t gr;
            CUDA_CHECK(cudaStreamBeginCapture(stm, cudaStreamCaptureModeGlobal));
            spec_round_launches();
            CUDA_CHECK(cudaStreamEndCapture(stm, &gr));
            CUDA_CHECK(cudaGraphInstantiate(&spec_graph[p], gr, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphDestroy(gr));
        }
        perm = 0;
        fprintf(stderr, "spec graphs captured (4 perms, depth-2)\n");
    }

    // one speculative round (depth 2); returns tokens emitted (1..3).
    // All position math + acceptance runs on device; host reads 16 bytes.
    int spec_round(int* emit) {
        CUDA_CHECK(cudaGraphLaunch(spec_graph[perm], stm));
        int oc[4];
        CUDA_CHECK(cudaMemcpyAsync(oc, d_outcome, 16, cudaMemcpyDeviceToHost, stm));
        CUDA_CHECK(cudaStreamSynchronize(stm));
        int n = oc[0];
        emit[0] = oc[1];
        if (n >= 2) emit[1] = oc[2];
        if (n == 3) emit[2] = oc[3];
        perm = (perm + (n - 1)) % 4;
        return n;
    }

    // ---- batched prefill (M6): T-token chunk versions of the blocks ----
    void qxT(const float* x, int cols, int T) { q27k::quantize_x(x, (int64_t)T * cols, xqT, stm); }
    void mmT(const DevTensor& w, const float* xT, float* yout, int T) {
        switch (w.dtype) {
            case DType::Q4_G64:
                q27k::gemm_q4_T((const uint8_t*)w.data, (const __half*)w.scales, xqT, yout,
                                w.rows, w.cols, T, stm);
                break;
            case DType::Q8_G128:
                q27k::gemm_q8_T((const int8_t*)w.data, (const __half*)w.scales, xqT, yout,
                                w.rows, w.cols, T, stm);
                break;
            case DType::F16:
                q27k::gemm_f16_T((const __half*)w.data, xT, yout, w.rows, w.cols, T, stm);
                break;
            default:
                fprintf(stderr, "mmT: unsupported dtype\n");
                exit(1);
        }
    }

    void gdn_block_T(int il, int T) {
        qxT(x1T, N_EMBD, T);
        mmT(T2(il, "attn_qkv.weight"), x1T, qkvT, T);
        mmT(T2(il, "attn_gate.weight"), x1T, zT, T);
        mmT(T2(il, "ssm_alpha.weight"), x1T, alphaT, T);
        mmT(T2(il, "ssm_beta.weight"), x1T, betarT, T);
        q27k::gdn_gates_T(alphaT, betarT, (const float*)T2(il, "ssm_a").data,
                          (const float*)T2(il, "ssm_dt.bias").data, gT, betaT, GDN_HEADS, T, stm);
        q27k::conv_prefill_T(conv_ring[il], qkvT,
                             (const float*)T2(il, "ssm_conv1d.weight").data, convT, GDN_CH, T,
                             stm);
        q27k::l2norm_heads_T(convT, 16, GDN_DIM, GDN_CH, T, EPS, stm);
        q27k::l2norm_heads_T(convT + 2048, 16, GDN_DIM, GDN_CH, T, EPS, stm);
        q27k::delta_scan_T(S[il], convT, gT, betaT, oT, T, stm);
        q27k::gated_norm_gdn_T(oT, (const float*)T2(il, "ssm_norm.weight").data, zT, ogT,
                               GDN_HEADS, GDN_DIM, T, EPS, stm);
        qxT(ogT, GDN_V, T);
        mmT(T2(il, "ssm_out.weight"), ogT, yT, T);
    }

    void attn_block_T(int il, int base, int T, __half* kc, __half* vc) {
        const int QROW = N_HEAD * 2 * HEAD_DIM, KVROW = N_KV * HEAD_DIM;
        qxT(x1T, N_EMBD, T);
        mmT(T2(il, "attn_q.weight"), x1T, qgT, T);
        q27k::rmsnorm_heads_T(qgT, (const float*)T2(il, "attn_q_norm.weight").data, qgT, N_HEAD,
                              HEAD_DIM, 2 * HEAD_DIM, QROW, T, EPS, stm);
        mmT(T2(il, "attn_k.weight"), x1T, kT, T);
        q27k::rmsnorm_heads_T(kT, (const float*)T2(il, "attn_k_norm.weight").data, kT, N_KV,
                              HEAD_DIM, HEAD_DIM, KVROW, T, EPS, stm);
        mmT(T2(il, "attn_v.weight"), x1T, vT, T);
        q27k::rope_neox_T(qgT, N_HEAD, HEAD_DIM, N_ROT, 2 * HEAD_DIM, QROW, base, T, FREQ_BASE,
                          stm);
        q27k::rope_neox_T(kT, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, KVROW, base, T, FREQ_BASE, stm);
        q27k::kv_store_T(kT, vT, kc, vc, base, KVROW, T, stm);
        q27k::attn_prefill_T(qgT, 2 * HEAD_DIM, QROW, kc, vc, attnT, N_HEAD * HEAD_DIM,
                             pf_scratch, base, 0, T, max_ctx, N_HEAD, N_KV, HEAD_DIM,
                             1.0f / sqrtf((float)HEAD_DIM), stm);
        q27k::sigmoid_gate_mul_T(attnT, qgT, N_HEAD, HEAD_DIM, T, stm);
        qxT(attnT, N_HEAD * HEAD_DIM, T);
        mmT(T2(il, "attn_output.weight"), attnT, yT, T);
    }

    void ffn_T(int il, int T) {
        qxT(x1T, N_EMBD, T);
        mmT(T2(il, "ffn_gate.weight"), x1T, ffnGT, T);
        mmT(T2(il, "ffn_up.weight"), x1T, ffnUT, T);
        q27k::silu_mul(ffnGT, ffnUT, ffnGT, (int64_t)T * N_FFN, stm);
        qxT(ffnGT, N_FFN, T);
        mmT(T2(il, "ffn_down.weight"), ffnGT, yT, T);
    }

    // Forward a chunk of T prompt tokens starting at absolute position `base`.
    // Leaves hT = final residual for each token. Updates conv rings, GDN state,
    // attention KV caches in place.
    void prefill_chunk(const int* d_toks, int base, int T) {
        const DevTensor& emb = dm.get("token_embd.weight");
        q27k::embed_rows_q8_T((const int8_t*)emb.data, (const __half*)emb.scales, d_toks,
                              N_EMBD, T, hT, stm);
        for (int il = 0; il < N_LAYER; il++) {
            q27k::rmsnorm_T(hT, (const float*)T2(il, "attn_norm.weight").data, x1T, N_EMBD, T,
                            EPS, stm);
            if (attn_layer[il]) {
                int ci = attn_cache_idx[il];
                attn_block_T(il, base, T, kcache[ci], vcache[ci]);
            } else {
                gdn_block_T(il, T);
            }
            q27k::add_inplace(hT, yT, (int64_t)T * N_EMBD, stm);
            q27k::rmsnorm_T(hT, (const float*)T2(il, "post_attention_norm.weight").data, x1T,
                            N_EMBD, T, EPS, stm);
            ffn_T(il, T);
            q27k::add_inplace(hT, yT, (int64_t)T * N_EMBD, stm);
        }
    }

    // Warm the MTP KV cache for pairs (h(base+t), token[base+t+1]) -> stored at
    // position base+t+1. Only the K/V projections matter for warming; the MTP
    // attention/FFN outputs were always discarded here, so they are skipped.
    void mtp_warm_T(const int* d_toks_next, int base, int T) {
        const int il = 64;
        const DevTensor& emb = dm.get("token_embd.weight");
        const int KVROW = N_KV * HEAD_DIM;
        // x1T currently holds output_norm(hT) (set by caller)
        q27k::embed_rows_q8_T((const int8_t*)emb.data, (const __half*)emb.scales, d_toks_next,
                              N_EMBD, T, embT, stm);
        q27k::rmsnorm_T(embT, (const float*)T2(il, "nextn.enorm.weight").data, ehnT, N_EMBD, T,
                        EPS, stm, N_EMBD, 2 * N_EMBD);
        q27k::rmsnorm_T(x1T, (const float*)T2(il, "nextn.hnorm.weight").data, ehnT + N_EMBD,
                        N_EMBD, T, EPS, stm, N_EMBD, 2 * N_EMBD);
        qxT(ehnT, 2 * N_EMBD, T);
        mmT(T2(il, "nextn.eh_proj.weight"), ehnT, xmtpT, T);
        q27k::rmsnorm_T(xmtpT, (const float*)T2(il, "attn_norm.weight").data, x1T, N_EMBD, T,
                        EPS, stm);
        qxT(x1T, N_EMBD, T);
        mmT(T2(il, "attn_k.weight"), x1T, kT, T);
        q27k::rmsnorm_heads_T(kT, (const float*)T2(il, "attn_k_norm.weight").data, kT, N_KV,
                              HEAD_DIM, HEAD_DIM, KVROW, T, EPS, stm);
        mmT(T2(il, "attn_v.weight"), x1T, vT, T);
        q27k::rope_neox_T(kT, N_KV, HEAD_DIM, N_ROT, HEAD_DIM, KVROW, base + 1, T, FREQ_BASE,
                          stm);
        q27k::kv_store_T(kT, vT, mtp_k, mtp_v, base + 1, KVROW, T, stm);
    }

    void snap_save(const std::vector<int>& prompt) {
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                CUDA_CHECK(cudaMemcpyAsync(S_snap[il], S[il],
                                           (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4,
                                           cudaMemcpyDeviceToDevice, stm));
                CUDA_CHECK(cudaMemcpyAsync(ring_snap[il], conv_ring[il], 3 * GDN_CH * 4,
                                           cudaMemcpyDeviceToDevice, stm));
            }
        snap_toks.assign(prompt.begin(), prompt.end() - 1);
        have_snap = true;
    }

    void snap_restore() {
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                CUDA_CHECK(cudaMemcpyAsync(S[il], S_snap[il],
                                           (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4,
                                           cudaMemcpyDeviceToDevice, stm));
                CUDA_CHECK(cudaMemcpyAsync(conv_ring[il], ring_snap[il], 3 * GDN_CH * 4,
                                           cudaMemcpyDeviceToDevice, stm));
            }
        perm = 0;
        CUDA_CHECK(cudaMemset(d_P, 0, 4));
    }

    // Reset all decode state for a fresh request (positions, GDN recurrent state,
    // conv rings, MTP KV). Weight buffers and captured graphs are unaffected.
    void reset() {
        CUDA_CHECK(cudaMemset(d_pos, 0, 4));
        CUDA_CHECK(cudaMemset(d_step, 0, 4));
        CUDA_CHECK(cudaMemset(d_P, 0, 4));
        perm = 0;
        for (int il = 0; il < N_LAYER; il++)
            if (!attn_layer[il]) {
                size_t sb = (size_t)GDN_HEADS * GDN_DIM * GDN_DIM * 4;
                CUDA_CHECK(cudaMemset(S[il], 0, sb));
                CUDA_CHECK(cudaMemset(S_spare[il], 0, sb));
                CUDA_CHECK(cudaMemset(S_spare2[il], 0, sb));
                CUDA_CHECK(cudaMemset(S_spare3[il], 0, sb));
                CUDA_CHECK(cudaMemset(conv_ring[il], 0, 3 * GDN_CH * 4));
                CUDA_CHECK(cudaMemset(ring_spare[il], 0, 3 * GDN_CH * 4));
                CUDA_CHECK(cudaMemset(ring_spare2[il], 0, 3 * GDN_CH * 4));
                CUDA_CHECK(cudaMemset(ring_spare3[il], 0, 3 * GDN_CH * 4));
            }
        CUDA_CHECK(cudaMemset(mtp_k, 0, (size_t)max_ctx * N_KV * HEAD_DIM * 2));
        CUDA_CHECK(cudaMemset(mtp_v, 0, (size_t)max_ctx * N_KV * HEAD_DIM * 2));
        CUDA_CHECK(cudaStreamSynchronize(stm));
    }

    // Prompt + speculative generation. Calls on_token(id) for each generated
    // token; stop when on_token returns false, n_max hit, or eos. Uses the spec
    // path (requires build_spec_graphs()). MTP KV warmed during prompt.
    template <typename F>
    int generate(const std::vector<int>& prompt, int n_max, int eos, F&& on_token) {
        int NP = (int)prompt.size();
        if (batched_prefill && NP >= 32) {
            // prefix-cache hit: prompt extends the snapshotted prefix -> restore
            // recurrent state and prefill only the new suffix
            int base = 0;
            if (have_snap && (int)snap_toks.size() <= NP - 1) {
                size_t L = 0;
                while (L < snap_toks.size() && snap_toks[L] == prompt[L]) L++;
                if (L == snap_toks.size()) base = (int)L;
            }
            fprintf(stderr, "[gen] prompt=%d prefix_hit=%d snap=%zu\n", NP, base,
                    snap_toks.size());
            if (base > 0) snap_restore();
            else reset();
            if (d_prompt_cap < NP) {
                if (d_prompt) CUDA_CHECK(cudaFree(d_prompt));
                CUDA_CHECK(cudaMalloc((void**)&d_prompt, (size_t)NP * 4));
                d_prompt_cap = NP;
            }
            CUDA_CHECK(cudaMemcpyAsync(d_prompt, prompt.data(), (size_t)NP * 4,
                                       cudaMemcpyHostToDevice, stm));
            const DevTensor& onw = dm.get("output_norm.weight");
            for (int c0 = base; c0 < NP - 1; c0 += PF_T) {
                int Tc = std::min((int)PF_T, (NP - 1) - c0);
                prefill_chunk(d_prompt + c0, c0, Tc);
                q27k::rmsnorm_T(hT, (const float*)onw.data, x1T, N_EMBD, Tc, EPS, stm);
                mtp_warm_T(d_prompt + c0 + 1, c0, Tc);
            }
            snap_save(prompt);
            int pos_last = NP - 1;
            CUDA_CHECK(cudaMemcpyAsync(d_pos, &pos_last, 4, cudaMemcpyHostToDevice, stm));
            CUDA_CHECK(cudaMemcpyAsync(d_step, &pos_last, 4, cudaMemcpyHostToDevice, stm));
            step_with(prompt[NP - 1]);
        } else {
            reset();
            have_snap = false; // serial path resets state without re-snapshotting
            for (size_t i = 0; i < prompt.size(); i++) {
                step_with(prompt[i]);
                if (i + 1 < prompt.size()) {
                    CUDA_CHECK(cudaStreamSynchronize(stm));
                    CUDA_CHECK(cudaMemcpyAsync(h_next, x1, N_EMBD * 4, cudaMemcpyDeviceToDevice,
                                               stm));
                    int nt = prompt[i + 1], mp = (int)i + 1;
                    CUDA_CHECK(cudaMemcpyAsync(d_token, &nt, 4, cudaMemcpyHostToDevice, stm));
                    CUDA_CHECK(cudaMemcpyAsync(d_pos_m, &mp, 4, cudaMemcpyHostToDevice, stm));
                    mtp_forward();
                    CUDA_CHECK(cudaStreamSynchronize(stm));
                }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stm));
        CUDA_CHECK(cudaMemcpyAsync(h_next, x1, N_EMBD * 4, cudaMemcpyDeviceToDevice, stm));
        int P = (int)prompt.size() - 1;
        CUDA_CHECK(cudaMemcpyAsync(d_P, &P, 4, cudaMemcpyHostToDevice, stm));
        int emitted = 0;
        auto g0 = std::chrono::steady_clock::now();
        auto done = [&](const char* why) {
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - g0)
                            .count();
            fprintf(stderr, "[gen-done] %s: %d tokens in %.1fs (%.1f t/s), n_max=%d\n", why,
                    emitted, dt, emitted / (dt > 0 ? dt : 1), n_max);
        };
        while (emitted < n_max) {
            int em[3];
            int n = spec_round(em);
            for (int k = 0; k < n && emitted < n_max; k++) {
                if (em[k] == eos) { done("eos"); return emitted; }
                if (!on_token(em[k])) { done("client-stop"); return emitted; }
                emitted++;
            }
        }
        done("n_max");
        return emitted;
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
