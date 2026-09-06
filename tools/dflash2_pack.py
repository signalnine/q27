#!/usr/bin/env python3
"""Pack the DFlash2 drafter checkpoint into a flat binary for src/dflash2.cu.

Format (little-endian): magic "D2W1", int32 n_tensors, then per tensor:
int32 name_len, name bytes, int32 dtype (0=f16, 1=f32), int32 ndim,
int64 dims[ndim], int64 nbytes, pad to 64-byte alignment, raw data.

Matrices go fp16 (bf16 -> fp16 is checked for overflow); norm weights and
conv base kernels go fp32 (the engine's rmsnorm/elementwise kernels take
float*). --with-target additionally packs the target's embedding and lm_head
rows (fp16, from the HF checkpoint) so the standalone smoke needs no .q27.

Usage: dflash2_pack.py <drafter_dir> <out.d2w> [--with-target <hf_dir>]
"""
import json
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open

F32_SUFFIXES = ('norm.weight', 'base_kernel')


def want_f32(name):
    return any(name.endswith(s) for s in F32_SUFFIXES)


def emit(f, name, t):
    if want_f32(name):
        a, dtype = t.float().numpy(), 1
    else:
        m = t.float().abs().max().item()
        assert m < 60000, f'{name}: max |w| {m} overflows fp16'
        a, dtype = t.to(torch.float16).numpy(), 0
    nb = name.encode()
    f.write(struct.pack('<i', len(nb)))
    f.write(nb)
    f.write(struct.pack('<ii', dtype, a.ndim))
    f.write(struct.pack(f'<{a.ndim}q', *a.shape))
    raw = a.tobytes()
    f.write(struct.pack('<q', len(raw)))
    pad = (-f.tell()) % 64
    f.write(b'\0' * pad)
    f.write(raw)


def main():
    drafter, out = sys.argv[1], sys.argv[2]
    hf = sys.argv[sys.argv.index('--with-target') + 1] if '--with-target' in sys.argv else None
    names, tensors = [], {}
    with safe_open(f'{drafter}/model.safetensors', 'pt') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
            names.append(k)
    extra = []
    if hf:
        idx = json.load(open(f'{hf}/model.safetensors.index.json'))['weight_map']
        for src, dst in (('model.language_model.embed_tokens.weight', 'target.embed.weight'),
                         ('lm_head.weight', 'target.head.weight')):
            with safe_open(f'{hf}/{idx[src]}', 'pt') as f:
                tensors[dst] = f.get_tensor(src)
            extra.append(dst)
    with open(out, 'wb') as f:
        f.write(b'D2W1')
        f.write(struct.pack('<i', len(names) + len(extra)))
        for k in names + extra:
            emit(f, k, tensors[k])
    print(f'{out}: {len(names) + len(extra)} tensors')


if __name__ == '__main__':
    main()
