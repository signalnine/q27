#!/usr/bin/env python3
"""Repack a BF16 GGUF into the q27 v1 format (see docs/FORMAT.md).

Usage:
  repack.py input.gguf output.q27 [--only REGEX] [--report N]

--only limits to tensors matching REGEX (smoke tests).
--report prints the N worst tensors by relative RMSE after quantization.
"""
import argparse
import json
import re
import struct
import sys
import time

import numpy as np
from gguf import GGUFReader

MAGIC = 0x46373251  # "Q27F" LE
VERSION = 1
ALIGN = 256

DTYPE_F32, DTYPE_F16, DTYPE_Q8, DTYPE_Q4 = 0, 1, 2, 3
DTYPE_NAMES = {DTYPE_F32: "F32", DTYPE_F16: "F16", DTYPE_Q8: "Q8_G128", DTYPE_Q4: "Q4_G64"}
GROUP_Q4, GROUP_Q8 = 64, 128


def policy(name: str) -> int:
    if (name.endswith("_norm.weight") or name.endswith("norm.weight")
            or name.endswith(".ssm_a") or name.endswith(".ssm_dt.bias")
            or "ssm_conv1d" in name):
        return DTYPE_F32
    if "ssm_alpha" in name or "ssm_beta" in name:
        return DTYPE_F16
    if name == "output.weight":
        return DTYPE_Q4  # v1.2: logits head to Q4 -- 3 full reads/spec-round, argmax-only consumer
    if name == "token_embd.weight" or name.startswith("blk.64."):
        return DTYPE_Q8
    if re.match(r"blk\.\d+\.attn_(k|v)\.weight$", name):
        return DTYPE_Q8  # KV projections: worst Q4 RMSE + errors persist in KV cache; ~84 MB total
    if name.endswith(".weight"):
        return DTYPE_Q4
    return DTYPE_F32  # biases and anything unrecognized stay f32


def to_f32(t) -> np.ndarray:
    """GGUF tensor -> f32 numpy array, row-major with contiguous axis last."""
    tt = t.tensor_type.name
    raw = np.asarray(t.data)
    if tt == "F32":
        arr = raw.view(np.float32)
    elif tt == "F16":
        arr = raw.view(np.float16).astype(np.float32)
    elif tt == "BF16":
        u16 = raw.view(np.uint16).astype(np.uint32)
        arr = (u16 << 16).view(np.float32)
    else:
        raise ValueError(f"{t.name}: unsupported source type {tt} (need BF16/F16/F32 input)")
    shape = tuple(reversed([int(d) for d in t.shape]))  # ne[0] is innermost
    return arr.reshape(shape)


def quant_q4(w: np.ndarray):
    rows, cols = (1, w.shape[0]) if w.ndim == 1 else (int(np.prod(w.shape[:-1])), w.shape[-1])
    assert cols % GROUP_Q4 == 0, f"cols {cols} not divisible by {GROUP_Q4}"
    g = w.reshape(rows, cols // GROUP_Q4, GROUP_Q4)
    scale = np.abs(g).max(axis=2) / 7.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.rint(g / scale[..., None]), -8, 7).astype(np.int8) + 8
    q = q.reshape(rows, cols).astype(np.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    deq = ((q.reshape(rows, cols // GROUP_Q4, GROUP_Q4).astype(np.float32) - 8)
           * scale[..., None]).reshape(rows, cols)
    return packed.tobytes(), scale.astype(np.float16).tobytes(), deq


def quant_q8(w: np.ndarray):
    rows, cols = (1, w.shape[0]) if w.ndim == 1 else (int(np.prod(w.shape[:-1])), w.shape[-1])
    assert cols % GROUP_Q8 == 0, f"cols {cols} not divisible by {GROUP_Q8}"
    g = w.reshape(rows, cols // GROUP_Q8, GROUP_Q8)
    scale = np.abs(g).max(axis=2) / 127.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.rint(g / scale[..., None]), -127, 127).astype(np.int8)
    deq = (q.astype(np.float32) * scale[..., None]).reshape(rows, cols)
    return q.tobytes(), scale.astype(np.float16).tobytes(), deq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--only", default=None)
    ap.add_argument("--report", type=int, default=15)
    args = ap.parse_args()

    t0 = time.time()
    r = GGUFReader(args.input)

    meta = {"q27_version": VERSION, "quant_policy": "v1.2",
            "group_q4": GROUP_Q4, "group_q8": GROUP_Q8, "nibble_order": "even=low"}
    for f in r.fields.values():
        if f.name.startswith(("qwen35.", "general.architecture", "general.name")):
            try:
                v = f.contents()
                if isinstance(v, bytes):
                    v = v.decode()
                meta[f.name] = v
            except Exception:
                pass
    # layer map
    attn_layers, ssm_layers = set(), set()
    for t in r.tensors:
        if t.name.startswith("blk."):
            n = int(t.name.split(".")[1])
            leaf = t.name.split(".", 2)[2]
            if leaf.startswith("attn_q."):
                attn_layers.add(n)
            if leaf.startswith("ssm_out"):
                ssm_layers.add(n)
    meta["attn_layers"] = sorted(attn_layers)
    meta["ssm_layers"] = sorted(ssm_layers)

    only = re.compile(args.only) if args.only else None
    entries, blobs = [], []
    errors = []
    offset = 0
    n_bytes_in = n_bytes_out = 0

    for t in r.tensors:
        if only and not only.search(t.name):
            continue
        w = to_f32(t)
        n_bytes_in += w.nbytes
        dt = policy(t.name)
        if dt == DTYPE_Q4 and w.shape[-1] % GROUP_Q4 != 0:
            dt = DTYPE_F16  # fallback, shouldn't happen on this model
        if dt == DTYPE_Q8 and w.shape[-1] % GROUP_Q8 != 0:
            dt = DTYPE_F16

        scales = b""
        if dt == DTYPE_F32:
            data = w.astype(np.float32).tobytes()
            deq = w
        elif dt == DTYPE_F16:
            data = w.astype(np.float16).tobytes()
            deq = w.astype(np.float16).astype(np.float32)
        elif dt == DTYPE_Q8:
            data, scales, deq = quant_q8(w)
        else:
            data, scales, deq = quant_q4(w)

        denom = float(np.sqrt(np.mean(w.astype(np.float64) ** 2))) or 1e-12
        rel_rmse = float(np.sqrt(np.mean((w - deq.reshape(w.shape)).astype(np.float64) ** 2))) / denom
        errors.append((rel_rmse, t.name, DTYPE_NAMES[dt]))

        data_off = offset
        offset += len(data)
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        scale_off = offset if scales else 0
        offset += len(scales)
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        n_bytes_out += len(data) + len(scales)

        entries.append((t.name, dt, w.shape, data_off, len(data), scale_off, len(scales)))
        blobs.append((data_off, data))
        if scales:
            blobs.append((scale_off, scales))
        del w, deq

    meta_b = json.dumps(meta).encode()
    with open(args.output, "wb") as f:
        f.write(struct.pack("<IIII", MAGIC, VERSION, len(entries), len(meta_b)))
        f.write(meta_b)
        for name, dt, shape, doff, dsize, soff, ssize in entries:
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<BB", dt, len(shape)))
            for d in shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<QQQQ", doff, dsize, soff, ssize))
        table_end = f.tell()
        pad = (table_end + ALIGN - 1) // ALIGN * ALIGN - table_end
        f.write(b"\0" * pad)
        base = f.tell()
        for off, blob in blobs:
            f.seek(base + off)
            f.write(blob)

    dt_s = time.time() - t0
    print(f"repacked {len(entries)} tensors: {n_bytes_in/1e9:.2f} GB f32-equiv -> "
          f"{n_bytes_out/1e9:.2f} GB in {dt_s:.0f}s -> {args.output}")
    errors.sort(reverse=True)
    print(f"\nworst {args.report} tensors by relative RMSE:")
    for rmse, name, dtn in errors[:args.report]:
        print(f"  {rmse:.4f}  {dtn:8s} {name}")


if __name__ == "__main__":
    main()
