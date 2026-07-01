# q27 weight format (version 1)

Offline-repacked weights for the q27 engine. Produced by `tools/repack.py` from the BF16 GGUF. Designed for mmap + single cudaMemcpy per tensor, and coalesced 128-byte warp reads in the fused-dequant GEMV.

## Container layout

```
[header]
  magic      u32   = 0x46373251  ("Q27F" little-endian)
  version    u32   = 1
  n_tensors  u32
  meta_len   u32
  meta       u8[meta_len]   JSON: arch config + layer map + quant policy
[tensor table]  n_tensors entries:
  name_len   u16
  name       u8[name_len]   GGUF tensor name, unchanged
  dtype      u8             0=F32  1=F16  2=Q8_G128  3=Q4_G64
  n_dims     u8
  shape      u64[n_dims]    numpy row-major shape (outer first; innermost/contiguous LAST)
  data_off   u64            relative to data section start, 256-byte aligned
  data_size  u64
  scale_off  u64            0 if dtype has no scales
  scale_size u64
[data section]  256-byte aligned blobs
```

## Quantized types

Both types quantize along the **contiguous (innermost) axis**, which is the GEMV
reduction axis for every matmul weight in this model.

### Q4_G64 (bulk weights)
- symmetric, group size 64: `scale = max(|wـgroup|) / 7`, `q = clip(round(w/scale), -8, 7) + 8` stored as unsigned nibble
- packing: element `i` of a row -> byte `i/2`; **even index = low nibble**, odd = high nibble
- scales: fp16, shape `[rows, cols/64]`, separate contiguous blob
- effective 4.25 bpw
- a warp reading 128 B gets 256 consecutive weights = exactly 4 groups

### Q8_G128 (quality-sensitive weights)
- symmetric, group size 128: `scale = max(|w_group|) / 127`, int8
- scales: fp16, `[rows, cols/128]`
- effective 8.125 bpw

## Quant policy (v1)

| tensors | dtype | why |
|---|---|---|
| all `*_norm.weight`, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, `output_norm` | F32 | tiny, numerically sensitive |
| `ssm_alpha.weight`, `ssm_beta.weight` | F16 | 48-wide heads, awkward group size, tiny anyway |
| `token_embd.weight`, `output.weight` | Q8_G128 | vocab quality; embed is row-lookup (not GEMV-read) |
| everything in `blk.64.*` (MTP layer) | Q8_G128 (matmuls) / F32 (norms) | draft/verify agreement must survive quantization or MTP acceptance craters |
| all other matmul weights (blk.0-63) | Q4_G64 | the ~14 GB bulk |

## Per-step read budget (decode)

Q4 bulk ~13.2 GB + Q8 lm_head ~1.3 GB + MTP layer ~0.4 GB + f16/f32 small tensors
=> ~14.8-15 GB per verify step. 5090 @ 1.79 TB/s => ~120 t/s ceiling before MTP amortization.
