#!/usr/bin/env python3
"""Pack the DFlash2 drafter checkpoint into a flat binary for src/dflash2.cu.

Format "D2W2" (little-endian): magic, int32 n_tensors, then per tensor:
int32 name_len, name, int32 dtype (0=f16, 1=f32, 2=q4_g64), int32 ndim,
int64 dims[ndim], int64 data_nbytes, pad to 64B, data; and for dtype==2
additionally int64 scales_nbytes, pad to 64B, fp16 scales (rows * cols/64).

Matmul weights go Q4_G64 (the engine's gemv_q4_n format: group 64, scale =
absmax/7, codes clip[-8,7]+8 packed 2/byte even=low, fp16 group scales --
identical to tools/repack.py quant_q4). Everything the drafter reads as a
matmul input rides the engine's weight-shared int4 path this way. Norm
weights and conv base kernels stay fp32; codebooks and the target embedding
stay fp16 (embedding lookups / the selector walk read them directly).
--q8 packs matmuls as Q8_G128 instead (gemv_q8_n; the numerics-safe
fallback if the Q4 acceptance gate fails). --with-target packs the target
embed + lm_head (the fp16 head is the fallback when the engine head is off).

Usage: dflash2_pack.py <drafter_dir> <out.d2w> [--with-target <hf_dir>] [--q8]
"""
import json
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open

F32_SUFFIXES = ('norm.weight', 'base_kernel')
# matmul weights -> quantized (the gemv path); everything else stays fp16/fp32
def is_matmul(name):
    return (name == 'fc.weight' or name.endswith('_proj.weight')
            or name.endswith('_projection.weight'))


def quant_q4(w):  # -> (packed uint8 bytes, fp16 scale bytes)
    rows, cols = w.shape
    assert cols % 64 == 0, f'{cols} not div 64'
    g = w.reshape(rows, cols // 64, 64)
    scale = np.abs(g).max(axis=2) / 7.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = (np.clip(np.rint(g / scale[..., None]), -8, 7).astype(np.int8) + 8)
    q = q.reshape(rows, cols).astype(np.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed.tobytes(), scale.astype(np.float16).tobytes()


def quant_q8(w):  # Q8_G128: scale = absmax/127, group 128, int8 + fp16 scale
    rows, cols = w.shape
    assert cols % 128 == 0, f'{cols} not div 128'
    g = w.reshape(rows, cols // 128, 128)
    scale = np.abs(g).max(axis=2) / 127.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.rint(g / scale[..., None]), -127, 127).astype(np.int8)
    return q.reshape(rows, cols).tobytes(), scale.astype(np.float16).tobytes()


def emit(f, name, t, q8):
    nb = name.encode()
    f.write(struct.pack('<i', len(nb)))
    f.write(nb)
    if is_matmul(name):
        w = t.float().numpy()
        if w.ndim != 2:
            raise SystemExit(f'{name}: matmul weight must be 2D')
        dtype = 3 if q8 else 2  # 3=q8_g128, 2=q4_g64
        data, scales = (quant_q8 if q8 else quant_q4)(w)
        f.write(struct.pack('<ii', dtype, w.ndim))
        f.write(struct.pack(f'<{w.ndim}q', *w.shape))
        f.write(struct.pack('<q', len(data)))
        f.write(b'\0' * ((-f.tell()) % 64))
        f.write(data)
        f.write(struct.pack('<q', len(scales)))
        f.write(b'\0' * ((-f.tell()) % 64))
        f.write(scales)
        return
    if any(name.endswith(s) for s in F32_SUFFIXES):
        a, dtype = t.float().numpy(), 1
    else:
        m = t.float().abs().max().item()
        assert m < 60000, f'{name}: max |w| {m} overflows fp16'
        a, dtype = t.to(torch.float16).numpy(), 0
    f.write(struct.pack('<ii', dtype, a.ndim))
    f.write(struct.pack(f'<{a.ndim}q', *a.shape))
    raw = a.tobytes()
    f.write(struct.pack('<q', len(raw)))
    f.write(b'\0' * ((-f.tell()) % 64))
    f.write(raw)


def main():
    drafter, out = sys.argv[1], sys.argv[2]
    hf = sys.argv[sys.argv.index('--with-target') + 1] if '--with-target' in sys.argv else None
    q8 = '--q8' in sys.argv
    names, tensors = [], {}
    with safe_open(f'{drafter}/model.safetensors', 'pt') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
            names.append(k)
    extra = []
    if hf:
        idx = json.load(open(f'{hf}/model.safetensors.index.json'))['weight_map']
        # The fp16 head is the engine-head fallback (Q27_D2_FP16HEAD); the fp16
        # embed is the drafter's anchor/mask lookup. Serving reuses the engine's
        # own Q8 head AND Q8 embed, so --no-head / --no-embed drop these 2.5 GB
        # each for the serving pack (~1.2 GB total).
        srcs = []
        if '--no-embed' not in sys.argv:
            srcs.append(('model.language_model.embed_tokens.weight', 'target.embed.weight'))
        if '--no-head' not in sys.argv:
            srcs.append(('lm_head.weight', 'target.head.weight'))
        for src, dst in srcs:
            with safe_open(f'{hf}/{idx[src]}', 'pt') as f:
                tensors[dst] = f.get_tensor(src)
            extra.append(dst)
    with open(out, 'wb') as f:
        f.write(b'D2W2')
        f.write(struct.pack('<i', len(names) + len(extra)))
        for k in names + extra:
            emit(f, k, tensors[k], q8)
    nq = sum(is_matmul(k) for k in names + extra)
    print(f'{out}: {len(names) + len(extra)} tensors, {nq} {"Q8" if q8 else "Q4"} matmuls')


if __name__ == '__main__':
    main()
