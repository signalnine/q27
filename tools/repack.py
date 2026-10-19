#!/usr/bin/env python3
"""Repack BF16 GGUF weights into the q27 v1 format (see docs/FORMAT.md).

Usage:
  repack.py input.gguf output.q27 [--mtp mtp.gguf] [--only REGEX] [--report N]

--mtp joins the companion MTP GGUF emitted by current llama.cpp conversions;
the primary GGUF supplies blocks 0..63 and the companion supplies block 64.
--only limits to tensors matching REGEX (smoke tests).
--report prints the N worst tensors by relative RMSE after quantization.

Ternary source packs (PrismML fork "Q2_0", ggml type 42) are detected
automatically and repacked losslessly to T2_G128 (quant_policy bonsai-t2-v1);
see docs/FORMAT.md and docs/metal/plans/2026-07-14-ternary-tier.md for the encoding.
Binary source packs (fork "Q1_0", ggml type 41) likewise repack losslessly
to B1_G128 (quant_policy bonsai-b1-v1); see
docs/metal/plans/2026-07-15-binary-tier.md.
"""
import argparse
import json
import re
import struct
import os
import sys
import time

import numpy as np
import gguf.constants as _ggc
from gguf import GGUFReader

# The PrismML fork's types are absent from mainline gguf-py; forge the enum
# members so GGUFReader can parse their packs. (block_size, type_size) per
# ggml/src/ggml-common.h at tag prism-b9591-62061f9.
def _forge_fork_type(name, value, blck, tsize):
    if value in _ggc.GGMLQuantizationType._value2member_map_:
        return _ggc.GGMLQuantizationType(value)
    m = int.__new__(_ggc.GGMLQuantizationType, value)
    m._name_, m._value_ = name, value
    _ggc.GGMLQuantizationType._member_map_[name] = m
    _ggc.GGMLQuantizationType._value2member_map_[value] = m
    _ggc.GGML_QUANT_SIZES[m] = (blck, tsize)
    return m

_forge_fork_type("Q2_0", 42, 128, 34)
_forge_fork_type("Q1_0", 41, 128, 18)
# Bonsai 2 (2026-09): Prism-private ternary at group 128. PTQ1_0 = {qs[24]
# five trits/byte, qh[2] four trits/byte, fp16 d}; PQ2_0 = {fp16 d, qs[32]
# 2-bit slots} (the Q2_0 codec at group 128). ggml.h at fork 1a07bfa.
_forge_fork_type("PQ2_0", 142, 128, 34)
_forge_fork_type("PTQ1_0", 143, 128, 28)
BONSAI2_TYPES = ("PTQ1_0", "PQ2_0")

MAGIC = 0x46373251  # "Q27F" LE
VERSION = 1
ALIGN = 256

DTYPE_F32, DTYPE_F16, DTYPE_Q8, DTYPE_Q4, DTYPE_T2 = 0, 1, 2, 3, 4
DTYPE_T3 = 5   # five trits per byte (FORMAT.md T3_G128); Bonsai 2 8 GB packs (2026-09-20)
DTYPE_B1 = 6
DTYPE_FP4 = 7  # nvfp4 sidecars: e2m1 codes 2/byte + ue4m3 scale per 16 (--pf4)
DTYPE_NAMES = {DTYPE_F32: "F32", DTYPE_F16: "F16", DTYPE_Q8: "Q8_G128", DTYPE_Q4: "Q4_G64",
               DTYPE_T2: "T2_G128", DTYPE_T3: "T3_G128", DTYPE_B1: "B1_G128", DTYPE_FP4: "FP4_G16"}
GROUP_Q4, GROUP_Q8, GROUP_T2, GROUP_B1 = 64, 128, 128, 128
GROUP_FP4 = 16


Q8_EXTRA = None  # set from --q8 (v1.4 sensitivity experiments)
Q4_HEAD = False  # set from --q4-head (q4s tier: single Q4 lm_head)
PF4 = False      # set from --pf4 (fp4 prefill sidecars, ninfer-steals phase 2)
FP4_ROUND = False  # set from --fp4-round (T1 weight-grid attribution arm)

# The fp4 prefill include list, shared by --pf4 (which emits sidecars for these)
# and --fp4-round (which rounds these in place). Keeping one definition is the
# point: T1 only attributes the phase-2 W4A4 result if it moves the same
# tensors. blk.64 (MTP) is outside the prefill loop and the SSM path is
# deliberately excluded (ssm_out cancellation lesson, BUILDLOG 2026-08-14).
PF4_INCLUDE = re.compile(r"blk\.(\d+)\.(ffn_gate|ffn_up|ffn_down|attn_q|attn_output)\.weight$")


def pf4_included(name: str) -> bool:
    m = PF4_INCLUDE.match(name)
    return bool(m) and int(m.group(1)) < 64


def policy(name: str) -> int:
    if name.endswith(".pf4"):
        return DTYPE_FP4  # --pf4 sidecar aliases; never matched by real GGUF names
    if (name.endswith("_norm.weight") or name.endswith("norm.weight")
            or name.endswith(".ssm_a") or name.endswith(".ssm_dt.bias")
            or "ssm_conv1d" in name):
        return DTYPE_F32
    if "ssm_alpha" in name or "ssm_beta" in name:
        return DTYPE_F16
    if name == "output_q4.weight":
        return DTYPE_Q4  # v1.3: extra Q4 copy of the lm_head for MTP DRAFT passes only
    if name == "output.weight" and Q4_HEAD:
        return DTYPE_Q4  # q4s: the ONE lm_head, Q4 -- draft/verify/plain all read it
    if name in ("token_embd.weight", "output.weight") or name.startswith("blk.64."):
        return DTYPE_Q8
    if re.match(r"blk\.\d+\.attn_(k|v)\.weight$", name):
        return DTYPE_Q8  # KV projections: worst Q4 RMSE + errors persist in KV cache; ~84 MB total
    if Q8_EXTRA and name.endswith(".weight") and Q8_EXTRA.search(name):
        return DTYPE_Q8  # v1.4: PPL-sensitive tensors promoted per experiment
    if name.endswith(".weight"):
        return DTYPE_Q4
    return DTYPE_F32  # biases and anything unrecognized stay f32


def _check_tensor_layout(reader, path: str) -> None:
    """gguf-py hands out whatever byte window the tensor-info table points at
    and never checks that the windows are aligned, in order, disjoint and
    inside the file (llama.cpp's loader does). A corrupted offset therefore
    repacked silently: the tensor read a neighbour's bytes shifted by a few
    elements, quantized without complaint, and the RMSE report (computed
    against the same shifted bytes) looked normal (bug hunt 2026-10-19).
    Refuse such a file before anything is written."""
    align = 32
    f = reader.fields.get("general.alignment")
    if f is not None:
        align = int(f.parts[f.data[0]][0])
    size = os.path.getsize(path)
    data_start = None
    for t in reader.tensors:
        if data_start is None or t.data_offset < data_start:
            data_start = t.data_offset
    prev_end, prev_name = -1, None
    for t in sorted(reader.tensors, key=lambda t: t.data_offset):
        off, end = int(t.data_offset), int(t.data_offset) + int(t.n_bytes)
        if data_start is not None and (off - data_start) % align:
            sys.exit(f"{path}: tensor {t.name} data offset {off} is not {align}-byte aligned -- corrupt tensor table, refusing")
        if off < prev_end:
            sys.exit(f"{path}: tensor {t.name} at {off} overlaps {prev_name} (ends {prev_end}) -- corrupt tensor table, refusing")
        if end > size:
            sys.exit(f"{path}: tensor {t.name} runs past the end of the file ({end} > {size}) -- truncated or corrupt, refusing")
        prev_end, prev_name = end, t.name


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


# nvfp4 encode tables. e2m1 positive grid; ue4m3 = unsigned e4m3 (bias 7,
# subnormals, no infinity, 0x7f = NaN excluded from the encode range).
_E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
def _ue4m3_table():
    v = np.zeros(127, dtype=np.float32)  # byte values 0x00..0x7e
    for b in range(127):
        e, m = b >> 3, b & 7
        v[b] = (m / 8.0) * 2.0 ** -6 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return v
_UE4M3 = _ue4m3_table()

def _nearest_even(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Index of nearest grid value, ties to EVEN index (= even mantissa LSB,
    matching cvt.rn semantics on these formats). Saturates at the grid ends."""
    idx = np.searchsorted(grid, x)                      # first grid[i] >= x
    idx = np.clip(idx, 1, len(grid) - 1)
    lo, hi = grid[idx - 1], grid[idx]
    take_lo = (x - lo < hi - x) | ((x - lo == hi - x) & ((idx - 1) % 2 == 0))
    return np.where(take_lo, idx - 1, idx).astype(np.int32)

def quant_nvfp4(w: np.ndarray):
    """nvfp4: 16-elem groups along the contiguous axis, ue4m3 scale =
    rne(absmax/6), e2m1 codes from the ROUNDED scale's reciprocal (the
    device-codec convention: see src/i8g64.cuh's sibling note and the ninfer
    recon), packed 2/byte even=low like Q4."""
    rows, cols = (1, w.shape[0]) if w.ndim == 1 else (int(np.prod(w.shape[:-1])), w.shape[-1])
    assert cols % GROUP_FP4 == 0, f"cols {cols} not divisible by {GROUP_FP4}"
    g = w.reshape(rows, cols // GROUP_FP4, GROUP_FP4).astype(np.float32)
    amax = np.abs(g).max(axis=2)
    sb = _nearest_even(np.minimum(amax / 6.0, _UE4M3[-1]), _UE4M3).astype(np.uint8)
    sw = _UE4M3[sb]
    inv = np.where(sw > 0, 1.0 / np.where(sw > 0, sw, 1), 0.0).astype(np.float32)
    scaled = np.abs(g) * inv[..., None]
    ci = _nearest_even(np.minimum(scaled, 6.0), _E2M1_POS)
    codes = (ci | np.where((g < 0) & (ci > 0), 8, 0)).astype(np.uint8)
    q = codes.reshape(rows, cols)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    deq = (np.where(codes & 8, -1.0, 1.0) * _E2M1_POS[codes & 7]
           * sw[..., None]).reshape(rows, cols).astype(np.float32)
    return packed.tobytes(), sb.tobytes(), deq


def fp4_round(w: np.ndarray, codec: str) -> np.ndarray:
    """Round w onto the e2m1 grid and return the DEQUANTIZED values (no packing).

    Two codecs, because they are not the same measurement:

    `q27` reproduces quant_nvfp4 exactly -- one ue4m3 byte per 16 holding
    amax/6 in ABSOLUTE units. That is what --pf4 emits and what src/pf4.cu
    reads, so it is the right choice for reproducing the shipped sidecars.
    It is also miscalibrated for weights: measured on this checkpoint, ~100% of
    16-element blocks have amax/6 near 0.0033, which is BELOW ue4m3's smallest
    normal (2^-6 = 0.015625), so the scale lands in the subnormal region where
    the step is a flat 2^-9. The scale itself then carries 20-40% error before
    the value grid is applied, and 44% of the resulting error VARIANCE is scale
    quantization rather than e2m1.

    `nvfp4` is the canonical two-level scheme: a per-tensor fp32 global scale
    normalizes the block scales into e4m3's normal range, where precision is
    relative rather than absolute. Measured rel RMSE 0.0950 vs the q27 codec's
    0.1274 on blk.0.ffn_gate, against an oracle (exact fp32 block scale) of
    0.0941 -- so it recovers essentially all of the gap and what is left IS the
    value grid. This is the codec to use when the question is "is the fp4
    WEIGHT GRID survivable", which is what T1 asks.

    Storing the global scale needs a format change the sidecar payload cannot
    take (src/loader.cpp fixes the payload at codes + one scale byte per 16),
    so this path deliberately returns values rather than a pack: T1 puts them
    in an ordinary container tier and never serializes fp4.
    """
    rows, cols = (1, w.shape[0]) if w.ndim == 1 else (int(np.prod(w.shape[:-1])), w.shape[-1])
    assert cols % GROUP_FP4 == 0, f"cols {cols} not divisible by {GROUP_FP4}"
    g = w.reshape(rows, cols // GROUP_FP4, GROUP_FP4).astype(np.float32)
    amax = np.abs(g).max(axis=2)
    if codec == "nvfp4":
        # amax_tensor / (6 * 448): maps the largest block's scale to the top of
        # the ue4m3 range so no block is left in the subnormals.
        sg = float(np.abs(w).max()) / (6.0 * _UE4M3[-1])
        sg = sg if sg > 0 else 1.0
        sw = _UE4M3[_nearest_even(np.minimum(amax / 6.0 / sg, _UE4M3[-1]), _UE4M3)] * sg
    elif codec == "q27":
        sw = _UE4M3[_nearest_even(np.minimum(amax / 6.0, _UE4M3[-1]), _UE4M3)]
    else:
        raise ValueError(f"unknown fp4 codec {codec!r}")
    inv = np.where(sw > 0, 1.0 / np.where(sw > 0, sw, 1), 0.0).astype(np.float32)
    ci = _nearest_even(np.minimum(np.abs(g) * inv[..., None], 6.0), _E2M1_POS)
    deq = np.where(g < 0, -1.0, 1.0) * _E2M1_POS[ci] * sw[..., None]
    return deq.reshape(rows, cols).astype(np.float32)


def quant_q8(w: np.ndarray):
    rows, cols = (1, w.shape[0]) if w.ndim == 1 else (int(np.prod(w.shape[:-1])), w.shape[-1])
    assert cols % GROUP_Q8 == 0, f"cols {cols} not divisible by {GROUP_Q8}"
    g = w.reshape(rows, cols // GROUP_Q8, GROUP_Q8)
    scale = np.abs(g).max(axis=2) / 127.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.rint(g / scale[..., None]), -127, 127).astype(np.int8)
    deq = (q.astype(np.float32) * scale[..., None]).reshape(rows, cols)
    return q.tobytes(), scale.astype(np.float16).tobytes(), deq


def repack_t2(t):
    """Fork Q2_0 tensor -> (data, scales, zero_frac). Lossless byte-copy.

    Source blocks are {fp16 d; uint8 qs[32]} x (n/128); codes are sequential
    LSB-first 2-bit fields, code c decodes to (c-1)*d. T2_G128 keeps the code
    bytes verbatim and splits scales into the usual contiguous fp16 blob, so
    the round-trip is exact by construction — still verified below.
    Hard-fails on code 3 (+2): the Bonsai packs must be strictly ternary.
    """
    shape = tuple(reversed([int(d) for d in t.shape]))  # ne[0] innermost -> last
    rows, cols = int(np.prod(shape[:-1])), shape[-1]
    if cols % GROUP_T2 != 0:  # hard contract, must survive python -O (codex P2)
        raise ValueError(f"{t.name}: cols {cols} not divisible by {GROUP_T2}")
    nblocks = rows * cols // GROUP_T2
    blocks = np.asarray(t.data).reshape(nblocks, 34)
    scales = blocks[:, :2].copy()                       # fp16 LE bytes, [rows, cols/128]
    qs = np.ascontiguousarray(blocks[:, 2:])            # [nblocks, 32] code bytes

    n_zero = 0
    for shift in (0, 2, 4, 6):
        c = (qs >> shift) & 3
        if np.any(c == 3):
            raise ValueError(f"{t.name}: code 3 (+2) present — pack is not strictly ternary; "
                             f"T2_G128 cannot represent it losslessly as ternary")
        n_zero += int(np.count_nonzero(c == 1))
    zero_frac = n_zero / (rows * cols)

    # Round-trip gate: dequantize the GGUF blocks per the fork's reference
    # (dequantize_row_q2_0) and our (data, scales) blobs per FORMAT.md, compare
    # bit-exact, chunked by rows to bound memory on token_embd.
    d_f32 = scales.view(np.float16).astype(np.float32).reshape(rows, cols // GROUP_T2)
    qs_rows = qs.reshape(rows, cols // 4)
    step = max(1, (1 << 25) // cols)  # ~128 MB f32 per chunk
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        blk = blocks.reshape(rows, cols // GROUP_T2, 34)[r0:r1]
        codes_g = np.stack([(blk[..., 2:] >> s) & 3 for s in (0, 2, 4, 6)],
                           axis=-1).reshape(r1 - r0, cols)
        deq_gguf = ((codes_g.astype(np.float32) - 1.0)
                    * np.repeat(blk[..., :2].copy().view(np.float16).astype(np.float32)
                                .reshape(r1 - r0, cols // GROUP_T2), GROUP_T2, axis=1))
        q = qs_rows[r0:r1]
        codes_o = np.stack([(q >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(r1 - r0, cols)
        deq_ours = ((codes_o.astype(np.float32) - 1.0)
                    * np.repeat(d_f32[r0:r1], GROUP_T2, axis=1))
        if not np.array_equal(deq_gguf, deq_ours):
            raise ValueError(f"{t.name}: T2 round-trip mismatch in rows {r0}:{r1}")

    return qs.tobytes(), scales.tobytes(), zero_frac


def decode_bonsai2_codes(t):
    """Bonsai 2 PTQ1_0 / PQ2_0 tensor -> (codes int8 [rows, cols] in {-1,0,+1},
    d fp16 bytes [rows, cols/128]).

    PTQ1_0 element map (fork tests/test-ptq1_0-element-map.cpp, verified there
    against the CUDA accessor): element e of a 128-block lives in
      e <  80: byte qs[e & 15],        base-3 digit e >> 4   (16 bytes x 5 trits)
      e < 120: byte qs[16 + (t & 7)],  digit t >> 3, t = e-80 (8 bytes x 5 trits)
      else   : byte qh[t & 1],         digit t >> 1, t = e-120 (2 bytes x 4 trits)
    digit n of byte b: v = (b * 3^n) mod 256; trit = ((v*3) >> 8) - 1 -- the
    TQ1_0-style top-byte extraction (dequantize_row_ptq1_0 / vec_dot in
    ggml-cpu/quants.c). PQ2_0: sequential LSB-first 2-bit fields, code c -> c-1,
    code 3 forbidden (same as the v1 Q2_0 codec, repack_t2).
    """
    shape = tuple(reversed([int(d) for d in t.shape]))
    rows, cols = int(np.prod(shape[:-1])), shape[-1]
    if cols % 128 != 0:
        raise ValueError(f"{t.name}: cols {cols} not divisible by 128")
    nblocks = rows * cols // 128
    tt = t.tensor_type.name
    raw = np.asarray(t.data)
    if tt == "PTQ1_0":
        blocks = raw.reshape(nblocks, 28)
        qs, qh = blocks[:, :24], blocks[:, 24:26]
        d = blocks[:, 26:28].copy()
        codes = np.empty((nblocks, 128), dtype=np.int8)
        pow3 = np.array([1, 3, 9, 27, 81], dtype=np.uint8)  # uint8 wraps mod 256, as in the codec

        def trit(bytes_, n):
            v = (bytes_ * pow3[n]).astype(np.uint8)  # uint8 * uint8 -> wraps
            return (((v.astype(np.uint16) * 3) >> 8).astype(np.int8) - 1)
        for n in range(5):
            codes[:, 16 * n:16 * n + 16] = trit(qs[:, :16], n)
            codes[:, 80 + 8 * n:80 + 8 * n + 8] = trit(qs[:, 16:24], n)
        for n in range(4):
            codes[:, 120 + 2 * n:120 + 2 * n + 2] = trit(qh, n)
    elif tt == "PQ2_0":
        blocks = raw.reshape(nblocks, 34)
        d = blocks[:, :2].copy()
        q = blocks[:, 2:]
        c = np.stack([(q >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(nblocks, 128)
        if np.any(c == 3):
            raise ValueError(f"{t.name}: PQ2_0 code 3 present -- not strictly ternary")
        codes = (c.astype(np.int8) - 1)
    else:
        raise ValueError(f"{t.name}: not a Bonsai 2 type ({tt})")
    if np.any(codes < -1) or np.any(codes > 1):
        raise ValueError(f"{t.name}: decoded value outside {{-1,0,1}}")
    return codes.reshape(rows, cols), d.reshape(rows, cols // 128, 2)


def _bonsai2_reference_deq(codes, d, rows, cols, r0, r1):
    """Fork semantics: value = trit * d (fp16 -> f32), per 128-group."""
    dd = d[r0:r1].copy().view(np.float16).astype(np.float32).reshape(r1 - r0, cols // 128)
    return codes[r0:r1].astype(np.float32) * np.repeat(dd, 128, axis=1)


def repack_bonsai2_q4x(t):
    """Bonsai 2 ternary tensor -> EXACT Q4_G64 blobs (data, scales, zero_frac).

    Nibble = trit + 8 (7/8/9), Q4 scale per 64 = the 128-group's d duplicated,
    so FORMAT.md's (nibble-8)*scale reproduces trit*d bit-for-bit in f32. This
    is the Phase-1 container (docs/plans/2026-09-18-bonsai2-ternary.md): the
    model runs on the existing Q4 kernels while the rotation is the only new
    math. Round-trip gate below.
    """
    codes, d = decode_bonsai2_codes(t)
    rows, cols = codes.shape
    nib = (codes.astype(np.int16) + 8).astype(np.uint8).reshape(rows, cols // 2, 2)
    data = (nib[:, :, 0] | (nib[:, :, 1] << 4)).astype(np.uint8)  # even = low nibble
    scales = np.repeat(d, 2, axis=1)  # [rows, cols/64, 2] fp16 bytes
    zero_frac = float(np.count_nonzero(codes == 0)) / codes.size
    step = max(1, (1 << 25) // cols)
    s_f32 = scales.reshape(rows, cols // 64, 2).view(np.float16).astype(np.float32).reshape(rows, cols // 64)
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        ref = _bonsai2_reference_deq(codes, d, rows, cols, r0, r1)
        q = data[r0:r1]
        nibs = np.stack([q & 0x0F, q >> 4], axis=-1).reshape(r1 - r0, cols)
        ours = (nibs.astype(np.float32) - 8.0) * np.repeat(s_f32[r0:r1], 64, axis=1)
        if not np.array_equal(ref, ours):
            raise ValueError(f"{t.name}: Q4x round-trip mismatch in rows {r0}:{r1}")
    return data.tobytes(), scales.tobytes(), zero_frac


def repack_bonsai2_t2(t):
    """Bonsai 2 ternary tensor -> T2_G128 blobs (data, scales, zero_frac): the
    FORMAT.md layout (code = trit+1 in 2-bit field (i%4)*2 of byte i/4, fp16
    scale per 128). Lossless; the round-trip gate re-decodes the packed bytes."""
    codes, d = decode_bonsai2_codes(t)
    rows, cols = codes.shape
    c = (codes.astype(np.int16) + 1).astype(np.uint8).reshape(rows, cols // 4, 4)
    data = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    zero_frac = float(np.count_nonzero(codes == 0)) / codes.size
    step = max(1, (1 << 25) // cols)
    d_f32 = d.copy().view(np.float16).astype(np.float32).reshape(rows, cols // 128)
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        ref = _bonsai2_reference_deq(codes, d, rows, cols, r0, r1)
        q = data[r0:r1]
        cc = np.stack([(q >> sh) & 3 for sh in (0, 2, 4, 6)], axis=-1).reshape(r1 - r0, cols)
        ours = (cc.astype(np.float32) - 1.0) * np.repeat(d_f32[r0:r1], 128, axis=1)
        if not np.array_equal(ref, ours):
            raise ValueError(f"{t.name}: T2 round-trip mismatch in rows {r0}:{r1}")
    return data.tobytes(), d.tobytes(), zero_frac


_T3_POW = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


def repack_bonsai2_t3(t):
    """Bonsai 2 ternary tensor -> T3_G128 blobs (data, scales, zero_frac): the
    FORMAT.md layout, 26 bytes per 128-group, byte b = sum_k code(5b+k) * 3^k
    (code = trit + 1, c0 least significant), byte 25 = columns 125..127 with
    its two unused slots at code 1 (the canonical pad the loader checks).
    1.625 bits per weight plus the fp16 scale per 128; the CUDA engine relays
    it into its own window layout at upload (kernels.cuh). Lossless; the
    round-trip gate re-decodes the packed bytes."""
    codes, d = decode_bonsai2_codes(t)
    rows, cols = codes.shape
    c = (codes.astype(np.int16) + 1).astype(np.uint16).reshape(rows, cols // 128, 128)
    c = np.concatenate([c, np.ones((rows, cols // 128, 2), dtype=np.uint16)], axis=2)  # 130 slots
    data = (c.reshape(rows, cols // 128, 26, 5) * _T3_POW).sum(axis=3).astype(np.uint8)
    assert data.max() <= 242
    zero_frac = float(np.count_nonzero(codes == 0)) / codes.size
    step = max(1, (1 << 25) // cols)
    d_f32 = d.copy().view(np.float16).astype(np.float32).reshape(rows, cols // 128)
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        ref = _bonsai2_reference_deq(codes, d, rows, cols, r0, r1)
        b = data[r0:r1].astype(np.uint16)[..., None]                     # [r, g, 26, 1]
        cc = ((b // _T3_POW) % 3).reshape(r1 - r0, cols // 128, 130)[..., :128].reshape(r1 - r0, cols)
        ours = (cc.astype(np.float32) - 1.0) * np.repeat(d_f32[r0:r1], 128, axis=1)
        if not np.array_equal(ref, ours):
            raise ValueError(f"{t.name}: T3 round-trip mismatch in rows {r0}:{r1}")
    return data.tobytes(), d.tobytes(), zero_frac


def repack_bonsai2_q8x(t):
    """Bonsai 2 ternary tensor -> EXACT Q8_G128 blobs (data, scales, zero_frac):
    int8 = trit, scale = d. Used for token_embd and output in Phase 1 (the
    embedding lookup and the head are Q8-only paths in the CUDA engine)."""
    codes, d = decode_bonsai2_codes(t)
    rows, cols = codes.shape
    zero_frac = float(np.count_nonzero(codes == 0)) / codes.size
    step = max(1, (1 << 25) // cols)
    d_f32 = d.copy().view(np.float16).astype(np.float32).reshape(rows, cols // 128)
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        ref = _bonsai2_reference_deq(codes, d, rows, cols, r0, r1)
        ours = codes[r0:r1].astype(np.float32) * np.repeat(d_f32[r0:r1], 128, axis=1)
        if not np.array_equal(ref, ours):
            raise ValueError(f"{t.name}: Q8x round-trip mismatch in rows {r0}:{r1}")
    return codes.tobytes(), d.tobytes(), zero_frac


def bonsai2_hadamard_meta(reader):
    """Copy the prism.hadamard.* keys verbatim (the engine applies the
    activation-side transform from these; FORMAT.md 'hadamard')."""
    out = {}
    for f in reader.fields.values():
        if not f.name.startswith("prism.hadamard."):
            continue
        v = f.contents()
        if isinstance(v, bytes):
            v = v.decode()
        elif isinstance(v, list):
            v = [x.decode() if isinstance(x, bytes) else (int(x) if isinstance(x, (np.integer,)) else x) for x in v]
        elif isinstance(v, np.generic):
            v = v.item()
        out[f.name[len("prism.hadamard."):]] = v
    for k in ("version", "block_size", "transform", "axis", "sign_mode", "sign_widths",
              "sign_values", "weight_names"):
        if k not in out:
            raise ValueError(f"source GGUF is missing prism.hadamard.{k}")
    if out["version"] != 1 or out["transform"] != "normalized-sylvester-walsh-hadamard" \
            or out["axis"] != "input-last-dimension" or out["sign_mode"] != "explicit":
        raise ValueError(f"unsupported prism.hadamard variant: {out}")
    if sum(out["sign_widths"]) != len(out["sign_values"]) or any(w % out["block_size"] for w in out["sign_widths"]):
        raise ValueError("prism.hadamard sign table inconsistent")
    out.setdefault("inverse_weight_names", [])
    out.setdefault("gdn_v_grouped", False)
    return out


def repack_b1(t):
    """Fork Q1_0 tensor -> (data, scales). Lossless byte-copy (B1_G128, dtype 6).

    Source blocks are {fp16 d; uint8 qs[16]} x (n/128); bit j of a group lives
    at qs[j/8] bit (j%8) — sequential LSB-first like type 42 — and decodes to
    (2b-1)*d (dequantize_row_q1_0 at tag prism-b9591-62061f9). B1_G128 keeps
    the code bytes verbatim and splits scales into the contiguous fp16 blob,
    exactly the binary-tier plan's Phase-1 layout. Round-trip verified below.
    """
    shape = tuple(reversed([int(d) for d in t.shape]))  # ne[0] innermost -> last
    rows, cols = int(np.prod(shape[:-1])), shape[-1]
    if cols % GROUP_B1 != 0:  # hard contract, must survive python -O (codex P2)
        raise ValueError(f"{t.name}: cols {cols} not divisible by {GROUP_B1}")
    nblocks = rows * cols // GROUP_B1
    blocks = np.asarray(t.data).reshape(nblocks, 18)
    scales = blocks[:, :2].copy()                       # fp16 LE bytes
    qs = np.ascontiguousarray(blocks[:, 2:])            # [nblocks, 16] code bytes

    d_f32 = scales.view(np.float16).astype(np.float32).reshape(rows, cols // GROUP_B1)
    qs_rows = qs.reshape(rows, cols // 8)
    step = max(1, (1 << 25) // cols)  # ~128 MB f32 per chunk
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        blk = blocks.reshape(rows, cols // GROUP_B1, 18)[r0:r1]
        bits_g = np.unpackbits(blk[..., 2:], axis=-1,
                               bitorder="little").reshape(r1 - r0, cols)
        deq_gguf = ((bits_g.astype(np.float32) * 2.0 - 1.0)
                    * np.repeat(blk[..., :2].copy().view(np.float16).astype(np.float32)
                                .reshape(r1 - r0, cols // GROUP_B1), GROUP_B1, axis=1))
        bits_o = np.unpackbits(qs_rows[r0:r1], axis=-1,
                               bitorder="little").reshape(r1 - r0, cols)
        deq_ours = ((bits_o.astype(np.float32) * 2.0 - 1.0)
                    * np.repeat(d_f32[r0:r1], GROUP_B1, axis=1))
        if not np.array_equal(deq_gguf, deq_ours):
            raise ValueError(f"{t.name}: B1 round-trip mismatch in rows {r0}:{r1}")

    return qs.tobytes(), scales.tobytes()

def _field_value(reader, name):
    field = reader.fields.get(name)
    if field is None:
        return None
    value = field.contents()
    return value.decode() if isinstance(value, bytes) else value


def _tensor_signature(tensor):
    return tensor.tensor_type.name, tuple(int(d) for d in tensor.shape)


def _tensor_data_equal(left, right):
    left_bytes = np.asarray(left.data).view(np.uint8).reshape(-1)
    right_bytes = np.asarray(right.data).view(np.uint8).reshape(-1)
    if left_bytes.size != right_bytes.size:
        return False
    chunk = 64 * 1024 * 1024
    return all(np.array_equal(left_bytes[off:off + chunk], right_bytes[off:off + chunk])
               for off in range(0, left_bytes.size, chunk))


def _qwen35_base_tensor_specs():
    # Source GGUF shapes in ordinary row-major order; GGUFReader exposes them
    # reversed, so _require_exact_specs reverses before comparing. These are
    # the same frozen Qwen3.8 dimensions enforced by both runtimes.
    embd, ffn, vocab = 5120, 17408, 248320
    head, n_head, n_kv = 256, 24, 4
    specs = {
        "token_embd.weight": ("BF16", (vocab, embd)),
        "output_norm.weight": ("F32", (embd,)),
        "output.weight": ("BF16", (vocab, embd)),
    }
    for layer in range(64):
        prefix = f"blk.{layer}."
        specs.update({
            prefix + "attn_norm.weight": ("F32", (embd,)),
            prefix + "post_attention_norm.weight": ("F32", (embd,)),
            prefix + "ffn_gate.weight": ("BF16", (ffn, embd)),
            prefix + "ffn_up.weight": ("BF16", (ffn, embd)),
            prefix + "ffn_down.weight": ("BF16", (embd, ffn)),
        })
        if layer % 4 == 3:
            specs.update({
                prefix + "attn_q.weight": ("BF16", (2 * n_head * head, embd)),
                prefix + "attn_k.weight": ("BF16", (n_kv * head, embd)),
                prefix + "attn_v.weight": ("BF16", (n_kv * head, embd)),
                prefix + "attn_output.weight": ("BF16", (embd, n_head * head)),
                prefix + "attn_q_norm.weight": ("F32", (head,)),
                prefix + "attn_k_norm.weight": ("F32", (head,)),
            })
        else:
            specs.update({
                prefix + "attn_qkv.weight": ("BF16", (10240, embd)),
                prefix + "attn_gate.weight": ("BF16", (6144, embd)),
                prefix + "ssm_alpha.weight": ("BF16", (48, embd)),
                prefix + "ssm_beta.weight": ("BF16", (48, embd)),
                prefix + "ssm_a": ("F32", (48,)),
                prefix + "ssm_dt.bias": ("F32", (48,)),
                prefix + "ssm_conv1d.weight": ("F32", (10240, 4)),
                prefix + "ssm_norm.weight": ("F32", (128,)),
                prefix + "ssm_out.weight": ("BF16", (embd, 6144)),
            })
    # Raise, not assert: `python -O` strips asserts, and this count is the
    # contract every other base-tensor check is measured against.
    if len(specs) != 851:
        raise ValueError(f"base tensor spec table is malformed: {len(specs)} != 851")
    return specs


def _qwen35_mtp_tensor_specs():
    embd, ffn, head, n_head, n_kv = 5120, 17408, 256, 24, 4
    prefix = "blk.64."
    specs = {
        prefix + "nextn.enorm.weight": ("F32", (embd,)),
        prefix + "nextn.hnorm.weight": ("F32", (embd,)),
        prefix + "nextn.shared_head_norm.weight": ("F32", (embd,)),
        prefix + "nextn.eh_proj.weight": ("BF16", (embd, 2 * embd)),
        prefix + "attn_norm.weight": ("F32", (embd,)),
        prefix + "post_attention_norm.weight": ("F32", (embd,)),
        prefix + "attn_q_norm.weight": ("F32", (head,)),
        prefix + "attn_k_norm.weight": ("F32", (head,)),
        prefix + "attn_q.weight": ("BF16", (2 * n_head * head, embd)),
        prefix + "attn_k.weight": ("BF16", (n_kv * head, embd)),
        prefix + "attn_v.weight": ("BF16", (n_kv * head, embd)),
        prefix + "attn_output.weight": ("BF16", (embd, n_head * head)),
        prefix + "ffn_gate.weight": ("BF16", (ffn, embd)),
        prefix + "ffn_up.weight": ("BF16", (ffn, embd)),
        prefix + "ffn_down.weight": ("BF16", (embd, ffn)),
    }
    if len(specs) != 15:
        raise ValueError(f"MTP tensor spec table is malformed: {len(specs)} != 15")
    return specs


QWEN35_BASE_SPECS = _qwen35_base_tensor_specs()
QWEN35_MTP_SPECS = _qwen35_mtp_tensor_specs()
QWEN35_BASE_TENSORS = frozenset(QWEN35_BASE_SPECS)
QWEN35_MTP_TENSORS = frozenset(QWEN35_MTP_SPECS)
QWEN35_REQUIRED_METADATA = {
    "qwen35.embedding_length": 5120,
    "qwen35.feed_forward_length": 17408,
    "qwen35.attention.head_count": 24,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.ssm.state_size": 128,
    "qwen35.ssm.group_count": 16,
    "qwen35.ssm.inner_size": 6144,
    "qwen35.context_length": 262144,
    "qwen35.rope.dimension_count": 64,
    "qwen35.ssm.conv_kernel": 4,
    "qwen35.ssm.time_step_rank": 48,
    "qwen35.full_attention_interval": 4,
    "qwen35.rope.freq_base": 10000000.0,
    "qwen35.attention.layer_norm_rms_epsilon": 0.000001,
    "qwen35.rope.dimension_sections": [11, 11, 10, 0],
}


def _normalized_metadata(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return [_normalized_metadata(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_normalized_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _exact_uint(value, expected):
    value = _normalized_metadata(value)
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _require_qwen38_metadata(reader, label):
    for name, expected in QWEN35_REQUIRED_METADATA.items():
        actual = _field_value(reader, name)
        if actual is None:
            raise ValueError(f"--mtp {label} is missing architecture metadata: {name}")
        actual = _normalized_metadata(actual)
        if isinstance(expected, int):
            matches = _exact_uint(actual, expected)
        elif isinstance(expected, float):
            matches = (isinstance(actual, float) and
                       (abs(actual - expected) <= 1e-12
                        if name == "qwen35.attention.layer_norm_rms_epsilon"
                        else actual == expected))
        elif isinstance(expected, list):
            matches = (isinstance(actual, list) and
                       all(isinstance(item, int) and not isinstance(item, bool)
                           for item in actual) and actual == expected)
        else:
            matches = type(actual) is type(expected) and actual == expected
        if not matches:
            raise ValueError(
                f"--mtp {label} architecture metadata mismatch: "
                f"{name}: {actual!r} != {expected!r}")


def _check_recurrent_layers(primary, companion):
    """Newer llama.cpp GGUFs carry a per-layer qwen35.attention.recurrent_layers
    array (2026-10, e.g. bytkim/Qwen3.8-27B-pi-GGUF): 64 entries in the base
    view, 65 in the MTP view, whose extra trailing 0 is the MTP block (full
    attention, not recurrent). Accept exactly that shape, and only when it
    agrees with full_attention_interval; older GGUFs carry neither."""
    name = "qwen35.attention.recurrent_layers"
    pf, cf = primary.fields.get(name), companion.fields.get(name)
    if pf is None and cf is None:
        return
    if pf is None or cf is None:
        raise ValueError(f"--mtp architecture metadata mismatch: {name} present in one view only")
    p = _normalized_metadata(pf.contents())
    c = _normalized_metadata(cf.contents())
    iv = _normalized_metadata(primary.fields["qwen35.full_attention_interval"].contents())
    want = [0 if (i + 1) % iv == 0 else 1 for i in range(64)]
    if p != want or c != want + [0]:
        raise ValueError(f"--mtp architecture metadata mismatch: {name} "
                         f"(primary {len(p)} entries, companion {len(c)}; expected the "
                         f"full_attention_interval={iv} pattern plus a trailing MTP 0)")


def _check_split_architecture_metadata(primary, companion):
    allowed = {"qwen35.block_count", "qwen35.nextn_predict_layers",
               "qwen35.attention.recurrent_layers"}  # checked by _check_recurrent_layers
    _check_recurrent_layers(primary, companion)
    primary_fields = {name: _normalized_metadata(field.contents())
                      for name, field in primary.fields.items()
                      if name.startswith("qwen35.") and name not in allowed}
    companion_fields = {name: _normalized_metadata(field.contents())
                        for name, field in companion.fields.items()
                        if name.startswith("qwen35.") and name not in allowed}
    if primary_fields != companion_fields:
        differing = sorted(set(primary_fields) ^ set(companion_fields) |
                           {name for name in set(primary_fields) & set(companion_fields)
                            if primary_fields[name] != companion_fields[name]})
        raise ValueError("--mtp architecture metadata mismatch: " + ", ".join(differing))


def _require_exact_manifest(label, actual, expected):
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} ({', '.join(sorted(missing)[:4])})")
        if unexpected:
            parts.append(f"unexpected {len(unexpected)} ({', '.join(sorted(unexpected)[:4])})")
        raise ValueError(f"{label} tensor manifest mismatch: " + "; ".join(parts))


def _require_exact_specs(label, tensors_by_name, expected_specs):
    _require_exact_manifest(label, set(tensors_by_name), set(expected_specs))
    for name, (expected_type, expected_shape) in expected_specs.items():
        tensor = tensors_by_name[name]
        actual_type = tensor.tensor_type.name
        actual_shape = tuple(reversed(tuple(int(d) for d in tensor.shape)))
        if actual_type != expected_type or actual_shape != expected_shape:
            raise ValueError(
                f"{label} tensor spec mismatch: {name}: "
                f"{actual_type} {actual_shape} != {expected_type} {expected_shape}")


# HF MTP-head checkpoint (safetensors, torch layout [out, in]) -> the 15 GGUF-named
# blk.64 tensors the engine expects (QWEN35_MTP_SPECS). Qwen3.5/3.8 use neox rope,
# so llama.cpp's converter does not permute q/k: the HF rows are the GGUF rows.
MTP_SAFETENSORS_MAP = {
    "mtp.fc.weight": "nextn.eh_proj.weight",
    "mtp.pre_fc_norm_embedding.weight": "nextn.enorm.weight",
    "mtp.pre_fc_norm_hidden.weight": "nextn.hnorm.weight",
    "mtp.norm.weight": "nextn.shared_head_norm.weight",
    "mtp.layers.0.input_layernorm.weight": "attn_norm.weight",
    "mtp.layers.0.post_attention_layernorm.weight": "post_attention_norm.weight",
    "mtp.layers.0.self_attn.q_norm.weight": "attn_q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight": "attn_k_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight": "attn_q.weight",
    "mtp.layers.0.self_attn.k_proj.weight": "attn_k.weight",
    "mtp.layers.0.self_attn.v_proj.weight": "attn_v.weight",
    "mtp.layers.0.self_attn.o_proj.weight": "attn_output.weight",
    "mtp.layers.0.mlp.gate_proj.weight": "ffn_gate.weight",
    "mtp.layers.0.mlp.up_proj.weight": "ffn_up.weight",
    "mtp.layers.0.mlp.down_proj.weight": "ffn_down.weight",
}


class _F32Tensor:
    """A GGUF-tensor look-alike (name / tensor_type.name / data / shape in ne order)
    wrapping an f32 numpy array, so the emit loop's to_f32 + policy path applies."""
    class _TT:
        name = "F32"

    def __init__(self, name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self.name = name
        self.tensor_type = _F32Tensor._TT()
        self.data = arr.reshape(-1)
        self.shape = list(reversed(arr.shape))  # GGUF ne order (innermost first)


def load_mtp_safetensors(path):
    import torch
    from safetensors import safe_open
    out = []
    with safe_open(path, "pt") as f:
        keys = set(f.keys())
        missing = set(MTP_SAFETENSORS_MAP) - keys
        extra = keys - set(MTP_SAFETENSORS_MAP)
        if missing or extra:
            raise ValueError(f"--mtp-safetensors: missing {sorted(missing)}, unexpected {sorted(extra)}")
        for hf, leaf in MTP_SAFETENSORS_MAP.items():
            name = "blk.64." + leaf
            arr = f.get_tensor(hf).to(torch.float32).numpy()
            # Qwen3.5 RMSNorm is zero-centered (y = x * (1 + w)); llama.cpp's
            # converter stores the effective multiplier 1 + w, which is what
            # the engine's rmsnorm kernels multiply by. Verified against the
            # Qwen3.8 pack's blk.64 norms (offset exactly 1.0, corr 1.0).
            if name.endswith("norm.weight"):
                arr = arr + 1.0
            want_dtype, want_shape = QWEN35_MTP_SPECS[name]
            if tuple(arr.shape) != tuple(want_shape):
                raise ValueError(f"--mtp-safetensors: {hf} -> {name} shape {tuple(arr.shape)} != {want_shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"--mtp-safetensors: {hf} has non-finite values")
            out.append((name, _F32Tensor(name, arr)))
    return out


def merge_mtp_tensors(primary, companion):
    """Join ggml-org's Qwen3.8 base and MTP GGUF views fail-closed."""
    if _field_value(primary, "general.architecture") != "qwen35":
        raise ValueError("--mtp requires a qwen35 primary GGUF")
    if _field_value(companion, "general.architecture") != "qwen35":
        raise ValueError("--mtp companion is not a qwen35 GGUF")
    primary_name = _field_value(primary, "general.name")
    companion_name = _field_value(companion, "general.name")
    if not isinstance(primary_name, str) or not primary_name.strip():
        raise ValueError("--mtp primary is missing general.name")
    if not isinstance(companion_name, str) or not companion_name.strip():
        raise ValueError("--mtp companion is missing general.name")
    if primary_name != companion_name:
        raise ValueError(f"--mtp checkpoint mismatch: {primary_name!r} != {companion_name!r}")
    if not _exact_uint(_field_value(primary, "qwen35.block_count"), 64):
        raise ValueError("--mtp primary must contain the 64 base blocks")
    if (not _exact_uint(_field_value(companion, "qwen35.block_count"), 65)
            or not _exact_uint(_field_value(companion, "qwen35.nextn_predict_layers"), 1)):
        raise ValueError("--mtp companion must describe one MTP layer (block_count=65)")
    _require_qwen38_metadata(primary, "primary")
    _require_qwen38_metadata(companion, "companion")
    _check_split_architecture_metadata(primary, companion)

    primary_by_name = {t.name: t for t in primary.tensors}
    if len(primary_by_name) != len(primary.tensors):
        raise ValueError("primary GGUF contains duplicate tensor names")
    _require_exact_specs("primary", primary_by_name, QWEN35_BASE_SPECS)
    companion_by_name = {t.name: t for t in companion.tensors}
    if len(companion_by_name) != len(companion.tensors):
        raise ValueError("MTP GGUF contains duplicate tensor names")

    allowed_shared = {"token_embd.weight", "output_norm.weight", "output.weight"}
    shared = set(primary_by_name) & set(companion_by_name)
    unexpected = shared - allowed_shared
    if unexpected:
        raise ValueError("--mtp companion overlaps primary tensors: "
                         + ", ".join(sorted(unexpected)))
    if shared != allowed_shared:
        missing = allowed_shared - shared
        raise ValueError("--mtp companion is missing shared tensors: "
                         + ", ".join(sorted(missing)))
    for name in shared:
        if _tensor_signature(primary_by_name[name]) != _tensor_signature(companion_by_name[name]):
            raise ValueError(f"--mtp shared tensor shape/type mismatch: {name}")
        if not _tensor_data_equal(primary_by_name[name], companion_by_name[name]):
            raise ValueError(f"--mtp shared tensor contents differ: {name}")

    mtp_only = [t for t in companion.tensors if t.name not in primary_by_name]
    mtp_by_name = {t.name: t for t in mtp_only}
    _require_exact_specs("MTP", mtp_by_name, QWEN35_MTP_SPECS)
    _require_exact_manifest("companion", set(companion_by_name),
                            allowed_shared | QWEN35_MTP_TENSORS)
    return list(primary.tensors) + mtp_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--mtp", default=None,
                    help="companion BF16 MTP GGUF (llama.cpp --mtp output)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--bonsai2-container", choices=("q4x", "t2", "t2+q4x", "t3"), default="t2",
                    help="Bonsai 2 packs: t2 = every rotated matrix as T2_G128 (decode GEMV and "
                         "the Phase 3 prefill GEMM read it; ~9 GB); t2+q4x = the same plus a "
                         "'<name>.q4x' exact-Q4 shadow of every blk.* matrix (the Phase 2 layout, "
                         "Q27_T2_PF_SHADOW=1 A/B; +13 GB); q4x = every rotated matrix as EXACT "
                         "Q4_G64 (Phase 1); t3 = every blk.* matrix as T3_G128 (five trits per "
                         "byte, 1.75 bpw with scales; the 8 GB-card pack, 2026-09-20 -- pair it "
                         "with --slim). Embeddings/head stay exact Q8 (T2 under --slim).")
    ap.add_argument("--mtp-safetensors", default=None,
                    help="Bonsai 2 packs: an HF-named MTP head checkpoint (mtp.fc.weight, "
                         "mtp.layers.0.*, mtp.norm, mtp.pre_fc_norm_*; e.g. ProCreations/"
                         "Ternary-Bonsai-2-27B-MTP model_mtp.safetensors) appended as the "
                         "blk.64 MTP block (Q8 matmuls, F32 norms, UNROTATED -- the engine "
                         "skips the Hadamard rotation for that layer). Sets block_count 65.")
    ap.add_argument("--slim", action="store_true",
                    help="Bonsai 2 t2 packs: token_embd and output as T2_G128 too (exact; the "
                         "engine's T2 embedding lookups and head GEMV are bitwise the Q8 ones) and "
                         "no output_q4.weight copy -- 9.44 -> ~7.5 GB, the 12 GB-card pack (2026-09-19)")
    ap.add_argument("--name", default=None,
                    help="general.name to write (Bonsai 2 packs default to 'Bonsai2 Ternary Qwen38 27b'; "
                         "other packs keep the GGUF's unless given). Must contain qwen38 for the "
                         "engine's XML dialect and 3.8 template keying -- set it for 3.8 fine-tunes "
                         "whose GGUF name lacks it)")
    ap.add_argument("--report", type=int, default=15)
    ap.add_argument("--q8", default=None,
                    help="extra tensor-name regex forced to Q8_G128 (v1.4 policy experiments)")
    ap.add_argument("--tag", default=None,
                    help="quant_policy meta override (e.g. q6-v1 for the 6-bit tier)")
    ap.add_argument("--q4-head", action="store_true",
                    help="emit output.weight at Q4_G64 and skip the output_q4.weight copy "
                         "(q4s tier; engine falls back to output.weight for drafts)")
    ap.add_argument("--pf4", action="store_true",
                    help="also emit nvfp4 sidecar copies (<name>.pf4, dtype FP4_G16) of the "
                         "attn+FFN projection weights for the fp4 prefill path "
                         "(Q27_PREFILL=fp4; ninfer-steals phase 2). ~10.5 GB extra; readers "
                         "older than DTYPE 7 cannot open the resulting file")
    ap.add_argument("--fp4-round", action="store_true",
                    help="T1 attribution arm (docs/plans/2026-08-18-fp4-viability-tests.md): "
                         "round the fp4 include-list projections onto the e2m1 grid with the "
                         "ue4m3 per-16 scale BEFORE the container quantizer, then store the "
                         "rounded values in the normal tier. The engine runs its ordinary "
                         "full-precision-activation path, so a differential against the same "
                         "container unmodified isolates fp4 WEIGHT-GRID damage from the "
                         "activation damage the W4A4 arm confounded it with. Emits no "
                         "sidecars and needs no new kernel; pair with a container fine enough "
                         "to preserve the rounded values (Q8_G128 on the include list)")
    ap.add_argument("--fp4-round-exclude", default=None,
                    help="regex of include-list tensors to LEAVE at base precision. Pairs with "
                         "--fp4-round-source to isolate an outside tier's two variables: run "
                         "its weights with its carve-outs, then q27's own rounding with the "
                         "SAME carve-outs, and the difference is that tier's encoder alone")
    ap.add_argument("--fp4-round-source", default=None,
                    help="instead of rounding here, take the already-quantized fp4 values from "
                         "another engine's artifact and store those (scratchpad/ninfer_fp4.py "
                         "supplies the reader). Lets q27's per-position instrument score a "
                         "THIRD-PARTY fp4 tier -- including its mixed-precision carve-outs, "
                         "since a tensor that engine does not store as fp4 is left at base "
                         "precision. Valid only because that pipeline applies no AWQ/smoothing "
                         "compensation; if it ever does, its weights stop being droppable")
    ap.add_argument("--fp4-round-codec", choices=("nvfp4", "q27"), default="nvfp4",
                    help="which fp4 encoding --fp4-round applies. nvfp4 (default) is the "
                         "canonical two-level scheme -- a per-tensor fp32 global scale keeps "
                         "the ue4m3 block scales out of the subnormal region -- and is the "
                         "one that isolates the VALUE grid. q27 reproduces the single-level "
                         "encoding quant_nvfp4/--pf4 actually ship, which on this checkpoint "
                         "puts ~100%% of blocks in ue4m3 subnormals and charges 44%% of its "
                         "error variance to scale quantization; use it to price the shipped "
                         "sidecars, not to answer whether fp4 weights are viable")
    args = ap.parse_args()
    if args.slim and args.bonsai2_container == "q4x":
        ap.error("--slim stores token_embd/output as T2_G128; it needs a t2/t3 container "
                 "(q4x would emit Q4 embeddings the CUDA loader rejects)")
    global Q8_EXTRA, Q4_HEAD, PF4, FP4_ROUND
    if args.q8:
        Q8_EXTRA = re.compile(args.q8)
    Q4_HEAD = args.q4_head
    PF4 = args.pf4
    FP4_ROUND = args.fp4_round

    t0 = time.time()
    r = GGUFReader(args.input)
    _check_tensor_layout(r, args.input)
    mtp_reader = GGUFReader(args.mtp) if args.mtp else None
    if mtp_reader is not None:
        _check_tensor_layout(mtp_reader, args.mtp)
    source_tensors = merge_mtp_tensors(r, mtp_reader) if mtp_reader else list(r.tensors)
    arch = _field_value(r, "general.architecture")
    if not isinstance(arch, str) or not arch:
        raise ValueError("source GGUF is missing general.architecture")
    ternary = any(t.tensor_type.name == "Q2_0" for t in source_tensors)
    binary = any(t.tensor_type.name == "Q1_0" for t in source_tensors)
    bonsai2 = any(t.tensor_type.name in BONSAI2_TYPES for t in source_tensors)
    if bonsai2 and (ternary or binary or mtp_reader):
        raise ValueError("a Bonsai 2 pack cannot be mixed with v1 Bonsai types or --mtp")
    if mtp_reader:
        bad_types = sorted({t.tensor_type.name for t in source_tensors
                            if t.tensor_type.name not in ("BF16", "F32")})
        if bad_types:
            raise ValueError("--mtp requires the pinned BF16 GGUF layout "
                             "(BF16 matrices/F32 scalars); found "
                             + ", ".join(bad_types))
    if binary and ternary:
        raise ValueError("pack unexpectedly contains both binary (Q1_0) and ternary (Q2_0) tensors")
    if (not mtp_reader and not ternary and not binary and not bonsai2
            and _field_value(r, "general.architecture") == "qwen35"
            and _field_value(r, "qwen35.block_count") == 64):
        raise ValueError("base-only qwen35 GGUF: supply its companion with --mtp")

    meta = {"q27_version": VERSION,
            "quant_policy": args.tag or (("bonsai2-t3-v1" if args.bonsai2_container == "t3"
                                          else "bonsai2-t2-v1" if args.bonsai2_container != "q4x"
                                          else "bonsai2-q4x-v1") if bonsai2
                                         else "bonsai-t2-v1" if ternary
                                         else "bonsai-b1-v1" if binary
                                         else "v1.4" if args.q8 else "v1.3"),
            "group_q4": GROUP_Q4, "group_q8": GROUP_Q8, "nibble_order": "even=low"}
    if bonsai2:
        # Ternary values in exact Q4_G64 / Q8_G128 containers (trit*d bit-for-bit)
        # plus the rotation the engine must apply to activations. No MTP block.
        meta["bonsai2"] = True
        meta["bonsai2_container"] = args.bonsai2_container + ("-slim" if args.slim else "")
        meta["hadamard"] = bonsai2_hadamard_meta(r)
    if ternary or (bonsai2 and args.bonsai2_container != "q4x"):
        meta["group_t2"] = GROUP_T2
        meta["t2_codes"] = "0=-1,1=0,2=+1;3 forbidden"
        meta["t2_slot_order"] = "seq-lsb-first"
    if bonsai2 and args.bonsai2_container == "t3":
        meta["group_t3"] = 128
        meta["t3_codes"] = "base-3 five per byte, c0 least significant, 26 B per 128; code = trit+1"
    if binary:
        # Verbatim fork Q1_0 encoding; see the repack_b1 docstring and the
        # binary-tier plan.
        meta["group_b1"] = GROUP_B1
        meta["b1_codes"] = "1=+d,0=-d"
        meta["b1_bit_order"] = "seq-lsb-first"
    if args.q8:
        meta["q8_extra"] = args.q8
    if args.q4_head:
        meta["q4_head"] = True
    if args.pf4:
        meta["pf4_sidecars"] = True
        meta["group_fp4"] = GROUP_FP4
        meta["pf4_encoding"] = "e2m1 even=low, ue4m3 scale per 16"
    if args.fp4_round:
        # Self-describing: this artifact is NOT a shippable tier, it is one leg
        # of the T1 differential. Anything that reads quant_policy should see it.
        meta["fp4_round"] = True
        meta["group_fp4"] = GROUP_FP4
        meta["fp4_round_include"] = PF4_INCLUDE.pattern
        if args.fp4_round_exclude:
            meta["fp4_round_exclude"] = args.fp4_round_exclude
        meta["fp4_round_codec"] = ("borrowed:" + args.fp4_round_source
                                   if args.fp4_round_source else args.fp4_round_codec)
        meta["fp4_round_encoding"] = (
            "e2m1 + ue4m3 per 16 (two-level, per-tensor fp32 global scale), "
            "dequantized into the container tier" if args.fp4_round_codec == "nvfp4" else
            "e2m1 + ue4m3 per 16 (single-level, absolute block scale -- the shipped "
            "--pf4 encoding), dequantized into the container tier")
    # The primary owns every architectural field. Split validation proved the
    # companion matches; only its intentional 65-block/MTP declarations may
    # override the 64-block base view.
    for f in r.fields.values():
        if f.name.startswith(("qwen35.", "general.architecture", "general.name")):
            try:
                v = f.contents()
                if isinstance(v, bytes):
                    v = v.decode()
                meta[f.name] = v
            except Exception:
                pass
    if mtp_reader:
        meta["qwen35.block_count"] = 65
        meta["qwen35.nextn_predict_layers"] = 1
        rl = mtp_reader.fields.get("qwen35.attention.recurrent_layers")
        if rl is not None:  # the 65-entry view (validated in merge_mtp_tensors)
            meta["qwen35.attention.recurrent_layers"] = _normalized_metadata(rl.contents())
    if args.name and not bonsai2:
        # Fine-tunes ship arbitrary general.name values ("Source Step8 Derived
        # Mtp" for bytkim/Qwen3.8-27B-pi); the engine keys the XML tool
        # dialect and the 3.8 template rules on a "qwen38" substring.
        meta["general.name"] = args.name
        norm = "".join(c for c in args.name.lower() if c.isalnum())
        if "qwen38" not in norm and "qwen36" not in norm:
            print(f"WARNING: --name {args.name!r} contains neither qwen38 nor qwen36: the engine keys the "
                  "tool dialect and template rules on that substring and will boot this pack with "
                  "the JSON dialect + 3.6 rules", file=sys.stderr)
    if bonsai2:
        # The GGUF's general.name is "Hf"; the engine keys the tool dialect and
        # the 3.8 template rules on a "qwen38" substring (api_common.h
        # set_tool_dialect_for_model), so name the artifact for what it is.
        meta["general.name"] = args.name or "Bonsai2 Ternary Qwen38 27b"
        if not args.mtp_safetensors:
            meta["qwen35.block_count"] = 64
            meta.pop("qwen35.nextn_predict_layers", None)
    # layer map
    attn_layers, ssm_layers = set(), set()
    for t in source_tensors:
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

    extra = []
    for t in source_tensors:
        # MTP draft head copy: only for non-quantized-head packs and pack
        # types that carry MTP (no MTP in ternary/binary packs).
        if t.name == "output.weight" and not args.q4_head \
                and not ternary and not binary and not (bonsai2 and args.slim):
            extra.append(("output_q4.weight", t))
    if bonsai2 and args.mtp_safetensors:
        mtp_st = load_mtp_safetensors(args.mtp_safetensors)
        extra.extend(mtp_st)
        meta["qwen35.block_count"] = 65
        meta["qwen35.nextn_predict_layers"] = 1
        meta["bonsai2_mtp"] = {"source": __import__("os").path.basename(args.mtp_safetensors),
                               "layout": "hf-safetensors, unrotated, Q8 matmuls / F32 norms"}
        print(f"MTP block from {args.mtp_safetensors}: {len(mtp_st)} tensors (blk.64.*)")
    if bonsai2 and args.bonsai2_container == "t2+q4x":
        # Phase 2 layout: decode reads T2, prefill the exact-Q4 shadow (engine
        # TP() under Q27_T2_PF_SHADOW=1; the Phase 3 GEMM reads T2 natively).
        for t in r.tensors:
            if t.tensor_type.name in BONSAI2_TYPES and t.name.startswith("blk."):
                extra.append((t.name + ".q4x", t))
    if PF4:
        # fp4 prefill sidecars: attn+FFN projections only, per PF4_INCLUDE.
        for t in r.tensors:
            if pf4_included(t.name):
                extra.append((t.name + ".pf4", t))
    class _Alias:
        def __init__(self, name, t):
            self.name, self.tensor_type, self.data, self.shape = name, t.tensor_type, t.data, t.shape
    tensor_iter = source_tensors + [_Alias(n, t) for n, t in extra]
    zero_fracs = []
    fp4_rounded, fp4_containers, fp4_skipped = [], {}, []
    fp4_src = None
    fp4_exclude = re.compile(args.fp4_round_exclude) if args.fp4_round_exclude else None
    if FP4_ROUND and args.fp4_round_source:
        sys.path.insert(0, str(__import__("os").path.dirname(__import__("os").path.abspath(__file__))
                              + "/../scratchpad"))
        from ninfer_fp4 import NinferWeights
        fp4_src = NinferWeights(args.fp4_round_source)
    for t in tensor_iter:
        if only and not only.search(t.name):
            continue
        verbatim = None  # (dtype, repack_fn) for lossless byte-copy source types
        if t.tensor_type.name == "Q2_0":
            verbatim = (DTYPE_T2, repack_t2)
        elif binary and t.tensor_type.name == "Q1_0":
            verbatim = (DTYPE_B1, repack_b1)
        elif bonsai2 and t.tensor_type.name in BONSAI2_TYPES:
            # Phase 1 containers: exact Q8 for the two Q8-only engine paths
            # (embedding lookup, head), exact Q4 for every rotated matrix.
            if t.name in ("token_embd.weight", "output.weight") and not args.slim:
                verbatim = (DTYPE_Q8, repack_bonsai2_q8x)
            elif t.name.endswith(".q4x"):
                verbatim = (DTYPE_Q4, repack_bonsai2_q4x)   # prefill shadow of a T2 base
            elif args.bonsai2_container == "t3" and t.name.startswith("blk."):
                verbatim = (DTYPE_T3, repack_bonsai2_t3)    # 8 GB packs: the body only
            elif args.bonsai2_container != "q4x":
                verbatim = (DTYPE_T2, repack_bonsai2_t2)
            else:
                verbatim = (DTYPE_Q4, repack_bonsai2_q4x)
        if verbatim is not None:
            vdt, fn = verbatim
            shape = tuple(reversed([int(d) for d in t.shape]))
            out = fn(t)
            if vdt == DTYPE_T2 or (bonsai2 and len(out) == 3):
                data, scales, zero_frac = out
                zero_fracs.append((zero_frac, int(np.prod(shape)), t.name))
            else:
                data, scales = out
            n_bytes_in += int(np.prod(shape)) * 4
            n_bytes_out += len(data) + len(scales)
            errors.append((0.0, t.name, DTYPE_NAMES[vdt]))  # lossless, gate-verified
            data_off = offset
            offset = (offset + len(data) + ALIGN - 1) // ALIGN * ALIGN
            scale_off = offset
            offset = (offset + len(scales) + ALIGN - 1) // ALIGN * ALIGN
            entries.append((t.name, vdt, shape, data_off, len(data), scale_off, len(scales)))
            blobs.append((data_off, data))
            blobs.append((scale_off, scales))
            continue
        if ternary and t.tensor_type.name != "F32":
            raise ValueError(f"{t.name}: unexpected source type {t.tensor_type.name} in a "
                             f"ternary pack (expected Q2_0 or F32 only)")
        if binary and t.tensor_type.name != "F32":
            raise ValueError(f"{t.name}: unexpected source type {t.tensor_type.name} in a "
                             f"binary pack (expected Q1_0 or F32 only)")
        w = to_f32(t)
        n_bytes_in += w.nbytes
        w_ref = w  # error is always reported against the SOURCE weights
        if FP4_ROUND and pf4_included(t.name) and fp4_exclude and fp4_exclude.search(t.name):
            fp4_skipped.append(t.name)
        elif FP4_ROUND and pf4_included(t.name):
            # Round onto the e2m1 grid, then hand the rounded values to the
            # container quantizer. Groups run along the contiguous axis, same
            # as the sidecars. See fp4_round() for why the codec is a choice
            # and not a constant.
            if w.ndim < 2 or w.shape[-1] % GROUP_FP4 != 0:
                raise ValueError(f"{t.name}: --fp4-round needs a 2-D tensor whose contiguous "
                                 f"axis divides {GROUP_FP4}; got shape {w.shape}")
            if fp4_src is not None:
                borrowed = fp4_src.get(t.name)
                if borrowed is None:
                    fp4_skipped.append(t.name)  # their recipe keeps this one high-precision
                else:
                    if borrowed.shape != w.shape:
                        raise ValueError(f"{t.name}: --fp4-round-source gave {borrowed.shape}, "
                                         f"expected {w.shape}")
                    w = borrowed.astype(np.float32)
                    fp4_rounded.append(t.name)
            else:
                w = fp4_round(w, args.fp4_round_codec).reshape(w.shape)
                fp4_rounded.append(t.name)
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
        elif dt == DTYPE_FP4:
            data, scales, deq = quant_nvfp4(w)
        else:
            data, scales, deq = quant_q4(w)

        denom = float(np.sqrt(np.mean(w_ref.astype(np.float64) ** 2))) or 1e-12
        rel_rmse = float(np.sqrt(np.mean(
            (w_ref - deq.reshape(w_ref.shape)).astype(np.float64) ** 2))) / denom
        errors.append((rel_rmse, t.name, DTYPE_NAMES[dt]))
        if FP4_ROUND and w is not w_ref:
            fp4_containers.setdefault(DTYPE_NAMES[dt], 0)
            fp4_containers[DTYPE_NAMES[dt]] += 1

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
        del w, w_ref, deq

    if bonsai2:
        # The sign vectors also travel as F32 tensors (hadamard_signs.<width>),
        # so the engine loads them like any F32V tensor instead of parsing a
        # 28672-element JSON array with its hand-rolled meta scanner.
        hm = meta["hadamard"]
        off_s = 0
        for w in hm["sign_widths"]:
            vals = np.asarray(hm["sign_values"][off_s:off_s + w], dtype=np.float32)
            off_s += w
            data = vals.tobytes()
            data_off = offset
            offset = (offset + len(data) + ALIGN - 1) // ALIGN * ALIGN
            entries.append((f"hadamard_signs.{w}", DTYPE_F32, (w,), data_off, len(data), 0, 0))
            blobs.append((data_off, data))
            n_bytes_out += len(data)
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

    if FP4_ROUND:
        by_tier = ", ".join(f"{n}x {k}" for k, n in sorted(fp4_containers.items()))
        print(f"\nfp4-round: {len(fp4_rounded)} include-list tensors rounded to the e2m1/ue4m3 "
              f"grid before packing ({by_tier or 'none'})")
        coarse = {k: n for k, n in fp4_containers.items() if k != "Q8_G128"}
        if coarse:
            # A Q4_G64 container cannot represent the rounded values (16 levels
            # over 64 elements vs fp4's per-16 scale), so leg B stops being
            # "fp4 weights" and becomes "q4 of fp4" -- strictly worse, which
            # biases T1 toward a FALSE kill. Say so where it will be read.
            print(f"  WARNING: {sum(coarse.values())} landed in a container coarser than "
                  f"Q8_G128 ({', '.join(sorted(coarse))}). The differential against the same "
                  f"container is CONTAMINATED there -- re-run with "
                  f"--q8 '(attn_q|attn_output|ffn_gate|ffn_up|ffn_down)\\.'")
        if fp4_skipped:
            print(f"  {len(fp4_skipped)} left at base precision -- the source tier keeps them "
                  f"high-precision, and reproducing that carve-out is the point: "
                  f"{', '.join(fp4_skipped[:8])}{' ...' if len(fp4_skipped) > 8 else ''}")
        if len(fp4_rounded) + len(fp4_skipped) != 224:
            print(f"  NOTE: expected 224 include-list tensors (5 projections x 64/16 layers); "
                  f"got {len(fp4_rounded) + len(fp4_skipped)} -- check --only")

    if zero_fracs:
        total = sum(n for _, n, _ in zero_fracs)
        mean_zero = sum(z * n for z, n, _ in zero_fracs) / total
        zero_fracs.sort()
        print(f"\nT2 slot verification passed on {len(zero_fracs)} tensors "
              f"({total/1e9:.2f} B ternary weights, {mean_zero:.1%} zeros overall)")
        print(f"  least sparse: {zero_fracs[0][0]:.1%} {zero_fracs[0][2]}")
        print(f"  most sparse:  {zero_fracs[-1][0]:.1%} {zero_fracs[-1][2]}")


if __name__ == "__main__":
    main()
