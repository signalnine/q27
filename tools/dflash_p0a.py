#!/usr/bin/env python3
"""DFlash Phase-0a: drafter acceptance-length on q27-captured hiddens.

Replays the z-lab/Qwen3.6-27B-DFlash drafter (torch bf16, exact algorithm
from FlashRT's _qwen36_rtx_dflash_forward.py reference) over a tap dump
produced by `q27 --dump-taps` and measures accepted length per cycle
against the committed greedy stream.

Dump format per step: int32 committed token, 5*5120 fp32 taps (residual
stream after target layers [1,16,31,46,61] for the step that CONSUMED
that token).

AL convention matches FlashRT: matched draft prefix + 1 (the verified
correction token always commits).

Usage: dflash_p0a.py <taps.bin> [--window 128] [--gen-only-from N]
"""
import argparse
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

DRAFTER = "/mnt/ai/models/qwen36-27b-dflash/model.safetensors"
GGUF = "/mnt/ai/models/qwen36-27b-mtp-gguf/Qwen3.6-27B-MTP-BF16.gguf"
HIDDEN, NTAP, BLOCK, MASK_ID = 5120, 5, 16, 248070
HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
THETA, EPS = 1e7, 1e-6
DEV = "cuda:0"


def rms(x, w):
    v = x.float()
    return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype) * w


def rope(x, pos):  # x: (T, H, D) NEOX-style full-dim rotation
    d2 = HEAD_DIM // 2
    inv = 1.0 / (THETA ** (torch.arange(0, d2, device=x.device).float() / d2))
    ang = pos.float()[:, None] * inv[None, :]  # (T, d2)
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = x[..., :d2].float(), x[..., d2:].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)


def load_drafter():
    w = {}
    with safe_open(DRAFTER, "pt") as f:
        for k in f.keys():
            w[k] = f.get_tensor(k).to(DEV)
    return w


def load_target_bits():
    import gguf
    r = gguf.GGUFReader(GGUF)
    out = {}
    for t in r.tensors:
        if t.name in ("token_embd.weight", "output.weight", "output_norm.weight"):
            a = np.array(t.data)
            if t.tensor_type == gguf.GGMLQuantizationType.BF16:
                a = a.view(np.uint16).astype(np.uint32) << 16
                a = a.view(np.float32)
            out[t.name] = torch.from_numpy(np.ascontiguousarray(a)).to(DEV, torch.bfloat16)
            out[t.name] = out[t.name].reshape(list(reversed(t.shape.tolist())))
    assert len(out) == 3, out.keys()
    return out["token_embd.weight"], out["output.weight"], out["output_norm.weight"]


def drafter_forward(w, embed_w, head_w, final_target_norm, prev_tok, taps, feat_ring):
    """One cycle. taps: (5, HIDDEN) fp32. feat_ring: (W, HIDDEN) bf16 past
    target_feat rows (per-token window). Returns 15 candidate tokens."""
    ids = torch.tensor([prev_tok] + [MASK_ID] * (BLOCK - 1), device=DEV)
    h = embed_w[ids].clone()  # (16, HIDDEN) bf16
    tf = rms(taps.to(DEV, torch.bfloat16).reshape(1, NTAP * HIDDEN) @ w["fc.weight"].T,
             w["hidden_norm.weight"])  # (1, HIDDEN)
    ctx = torch.cat([feat_ring, tf], 0) if feat_ring is not None else tf  # (C, HIDDEN)
    C = ctx.shape[0]
    pos_q = torch.arange(C, C + BLOCK, device=DEV)
    pos_k = torch.arange(0, C + BLOCK, device=DEV)
    for L in range(5):
        p = f"layers.{L}."
        hn = rms(h, w[p + "input_layernorm.weight"])
        # ctx rows (target_feat) enter K/V RAW per the FlashRT reference --
        # they are normalized ONCE by hidden_norm at creation, never by the
        # per-layer input layernorm
        q = (hn @ w[p + "self_attn.q_proj.weight"].T).view(BLOCK, HEADS, HEAD_DIM)
        kq = (hn @ w[p + "self_attn.k_proj.weight"].T).view(BLOCK, KV_HEADS, HEAD_DIM)
        vq = (hn @ w[p + "self_attn.v_proj.weight"].T).view(BLOCK, KV_HEADS, HEAD_DIM)
        kc = (ctx @ w[p + "self_attn.k_proj.weight"].T).view(C, KV_HEADS, HEAD_DIM)
        vc = (ctx @ w[p + "self_attn.v_proj.weight"].T).view(C, KV_HEADS, HEAD_DIM)
        q = rms(q, w[p + "self_attn.q_norm.weight"])
        k = rms(torch.cat([kc, kq], 0), w[p + "self_attn.k_norm.weight"])
        v = torch.cat([vc, vq], 0)
        q, k = rope(q, pos_q), rope(k, pos_k)
        q = q.transpose(0, 1)  # (H, 16, D)
        k = k.repeat_interleave(HEADS // KV_HEADS, 1).transpose(0, 1) if False else \
            k.transpose(0, 1).repeat_interleave(HEADS // KV_HEADS, 0)
        v = v.transpose(0, 1).repeat_interleave(HEADS // KV_HEADS, 0)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        h = h + a.transpose(0, 1).reshape(BLOCK, HEADS * HEAD_DIM) @ w[p + "self_attn.o_proj.weight"].T
        hf = rms(h, w[p + "post_attention_layernorm.weight"])
        h = h + (F.silu(hf @ w[p + "mlp.gate_proj.weight"].T) *
                 (hf @ w[p + "mlp.up_proj.weight"].T)) @ w[p + "mlp.down_proj.weight"].T
    hf = rms(h, w["norm.weight"])
    logits = hf @ head_w.T  # (16, VOCAB)
    return logits[1:].argmax(-1).tolist(), tf  # 15 candidates + this cycle's feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--skip", type=int, default=160,
                    help="steps to skip before measuring (prompt tail warms the ring)")
    args = ap.parse_args()

    rec = 4 + NTAP * HIDDEN * 4
    raw = open(args.dump, "rb").read()
    nstep = len(raw) // rec
    toks, taps = [], []
    for i in range(nstep):
        off = i * rec
        toks.append(struct.unpack_from("<i", raw, off)[0])
        taps.append(np.frombuffer(raw, np.float32, NTAP * HIDDEN, off + 4).reshape(NTAP, HIDDEN))
    print(f"{nstep} steps loaded", file=sys.stderr)

    w = load_drafter()
    embed_w, head_w, onw = load_target_bits()
    print("weights resident", file=sys.stderr)

    ring = []  # per-token feature window
    als, t = [], 1
    with torch.no_grad():
        while t < nstep - BLOCK:
            feat_ring = torch.cat(ring[-args.window:], 0) if ring else None
            taps_t = torch.from_numpy(taps[t - 1].copy())
            cands, tf = drafter_forward(w, embed_w, head_w, onw, toks[t - 1], taps_t, feat_ring)
            truth = toks[t:t + 15]
            m = 0
            while m < 15 and cands[m] == truth[m]:
                m += 1
            al = m + 1
            if t >= args.skip:
                als.append(al)
            # commit al tokens: append their features (per-token window needs
            # features of every committed token -- we only have the fc of the
            # CYCLE tap; per-token mode projects each committed step's taps)
            for j in range(al):
                idx = t - 1 + j
                tf_j = rms(torch.from_numpy(taps[idx].copy()).to(DEV, torch.bfloat16)
                           .reshape(1, NTAP * HIDDEN) @ w["fc.weight"].T,
                           w["hidden_norm.weight"])
                ring.append(tf_j)
            ring = ring[-args.window:]
            t += al
    a = np.array(als)
    print(f"cycles={len(a)} mean_AL={a.mean():.3f} median={np.median(a):.1f} "
          f"p25={np.percentile(a,25):.1f} p75={np.percentile(a,75):.1f} "
          f"full16={100*(a==16).mean():.1f}% one={100*(a==1).mean():.1f}%")


if __name__ == "__main__":
    main()
