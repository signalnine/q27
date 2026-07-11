# turbo3 KV port spec (recovered from wf_f94f54d8-2ab, 2026-07-11 session 13)

Verified against the actual q27 and TurboQuant sources by a 5-agent read/synthesize workflow. Phase-1 implementation follows this spec.

---

# q27 turbo3 KV PORT SPEC — Phase 1 (block-addressing + fd2 read path)

## Verified anchor facts (checked in-tree, not just reader-reported)

- q27 write: `k_kv_store` (`src/blocks.cu:79`) is **flat, one-thread-per-element**: `off = (*d_pos)*rowlen + i`, `rowlen = N_KV*HEAD_DIM = 1024`, `kv_set` casts f32->fp8/f16. No cross-lane cooperation.
- q27 read: `k_attn_fd2<CT,NW>` (`src/spec3.cu:328`), row base `kp = kc + ((size_t)p*n_kv_heads + kvh)*head_dim`; per-lane load `fd2_ld8<CT>` (`src/spec3.cu:297`).
- **Lane dim ownership** (`src/spec3.cu` comment 289-290 + code 300-323): lane `l` owns dims `{4l..4l+3}` and `{128+4l..128+4l+3}` -> `kv[0..3]` from the first 128, `kv[4..7]` from the second 128. This is already a clean **128/128 split of the 256-dim head**. This is the single most important fact: q27's fd2 lane layout maps 1:1 onto turbo3's two 128-groups with zero reshuffling.
- `engine.cuh:32` `N_HEAD=24, N_KV=4, HEAD_DIM=256`; **GQA ratio = 24/4 = 6.0**.
- block struct confirmed 50 B (`norm@0`(2) + `qs[32]@2` + `signs[16]@34`); the `//14 bytes` comment in `ggml-common.h` is stale (QK_TURBO3=32 era). CENTROIDS_3BIT, midpoints, `turbo_cpu_s1/s2[128]`, `inv_sqrt=0.08838834764831845f` all verified in `ggml-turbo-quant.c`.

---

## (1) turbo3 block-addressed KV layout for q27

head_dim=256, QK_TURBO3=128 -> **2 blocks per (kv_head)** (256 % 128 == 0, **no padding**).

- Blocks per token-row: `rowlen_blk = N_KV * (HEAD_DIM/128) = 4 * 2 = 8` blocks/row (K), same for V.
- Row stride in bytes: `nb_row = rowlen_blk * 50 = 400 B/row` (was fp8 1024 B, fp16 2048 B). -> **2.56x smaller than current fp8, 5.12x vs fp16.**
- Block index for (pos, kv_head `h`, group `g∈{0,1}`):
  ```
  blk = (block_turbo3_0*)cache + (size_t)pos*8 + h*2 + g
  ```
  group `g=0` -> head dims 0..127, `g=1` -> dims 128..255; each block carries its **own fp16 norm**.
- Total K cache bytes/engine/stream: `max_ctx * 8 * 50 = max_ctx * 400`. (Replaces `engine.cuh:711` `A(&k, max_ctx*N_KV*HEAD_DIM*kv_esz())`.) `kv_esz()` (element-size 1/2) no longer applies; add `kv_bpr()` = bytes-per-row = 400 for turbo3, else `N_KV*HEAD_DIM*kv_esz()`.

---

## (2) kv_store3 turbo3 write kernel — YES, q27 MUST add K/V rotation. TURBO_D = 128.

`k_kv_store`'s per-element independence is **incompatible** with turbo3 (needs L2-norm + WHT across each 128-group). Replace with a cooperative kernel mirroring `k_set_rows_turbo3` (`set-rows.cu:239`):

- Launch: `grid = dim3(N_KV*(HEAD_DIM/128) /*=8*/, 2 /*K,V*/)`, `128 threads`. One CUDA block == one 128-group; thread `j` owns group-local element `j`.
- Per-group pipeline (thread `j`):
  1. `x = buf[h*256 + g*128 + j]` into smem.
  2. `grp_norm = sqrt(Σ x_j²)` (smem reduction over 128), on the **pre-rotation** input.
  3. normalize: `x *= 1/grp_norm`.
  4. **forward RHT**: `x *= s1[j]` -> 7-stage in-place radix-2 Hadamard butterfly (h=1,2,4,…,64) -> `x *= 0.08838834764831845f * s2[j]`.
  5. `idx_j = nearest_centroid_3bit(x_j)` (7 midpoints below).
  6. `recon_norm = sqrt(Σ centroid[idx_j]²)`; **corrected norm = `grp_norm/recon_norm`** (mandatory — plain grp_norm mis-scales).
  7. pack warp-cooperatively: `qs[j/4] |= (idx&3)<<((j%4)*2)`; `if(idx&4) signs[j/8] |= 1<<(j%8)`; write `blk->norm = __float2half(corrected)` once.

**TURBO_D = 128, hard.** It is the WHT/block size, decoupled from head_dim. head_dim=256 -> two independent TURBO_D=128 groups. Reuse `turbo_cpu_s1/s2[128]` byte-for-byte (seed 42) — the writer's sign arrays MUST byte-match whatever Q-rotation the read path assumes.

**Rotation plumbing q27 must add around the store** (q27 has no ggml graph, so these are explicit kernel launches in `engine.cu` decode/prefill):
- Forward-WHT on **Q** (`qg`) per 128-group, inserted **after** rope (`engine.cuh:845`), **before** attn.
- Forward-WHT on K/V folded **inside** the store kernel (step 4 above) — do not rotate `kbuf`/`vbuf` separately; K is rope'd first (`engine.cuh:846`), then rotated+quantized.
- Inverse-WHT on `attnout` per 128-group **after** the fd2 combine, before the o-projection.

InnerQ: **skip entirely in phase 1** (pass NULL). It auto-disables for head_dim>128 anyway.

---

## (3) fd2 turbo3 read (dequant only; no in-kernel rotation)

Add a `k_attn_fd2_turbo3<NW>` (do NOT overload `CT` — row stride is in blocks, and norm must be hoisted). Softmax/online-max/merge code is **byte-identical** to `k_attn_fd2`; only the K/V load changes.

Row base per (pos `p`, kv_head `kvh`):
```
block_turbo3_0* kb = kbase + (size_t)p*8 + kvh*2;   // kb[0]=dims0..127, kb[1]=dims128..255
```
Per-lane load (replaces `fd2_ld8`). Lane `l` element `4l` maps to **`qs[l]`** (since `(4l)/4 == l`) — one qs byte serves all 4 of the lane's dims in a block; sign byte `signs[l/2]`, bits `(l&1)*4 + i`:
```
// group g in {0,1}; block gb = kb[g]; loads kv[4g .. 4g+3]
float norm = __half2float(gb->norm);          // hoisted: 1 fp16 load / block / lane
uint8_t q  = gb->qs[l];                        // 1 byte -> 4 dims
uint8_t s  = gb->signs[l>>1];
#pragma unroll
for (int i=0;i<4;i++){
    int idx = ((q>>(2*i))&3) | (((s>>((l&1)*4+i))&1)<<2);
    kv[4*g+i] = TURBO_CENTROIDS_3BIT[idx] * norm;
}
```
So each lane does **2 qs bytes + 2 signs bytes + 2 norms** for its full 8 dims (mirrors the reference's amortized byte-reuse). Do the same for V into `vv[8]`. Optionally precompute `sc[c]=centroid[c]*norm` (8 values/block) to hoist the norm-multiply, as the reference V loop does.

- **Q-rotate-once**: handled outside the kernel (item 2). The kernel receives already-rotated `s_q`; the `d = Σ qa·kv + qb·kv` dot is `<Q_rot,K_rot>` = true `<Q,K>` (WHT orthonormal, per-group).
- **V-inverse-rotate**: NOT in the kernel. VKQ accumulates in the rotated basis; the single post-combine inverse-WHT on `attnout` un-rotates the pooled output for all V at once (linearity). No per-element inverse.

---

## (4) Constants to embed (verbatim, verified)

```c
// Lloyd-Max centroids for N(0,1/128)
static const float TURBO_CENTROIDS_3BIT[8] = {
  -0.190207f,-0.118786f,-0.066822f,-0.021663f, 0.021663f,0.066822f,0.118786f,0.190207f };
// nearest-centroid midpoints (7)
static const float TURBO_MID_3BIT[7] = {
  -0.154496f,-0.092804f,-0.044243f, 0.0f, 0.044243f,0.092804f,0.154496f };
// WHT scale (group 128). (group 64 would be 0.125f — not used at head_dim=256)
#define TURBO_INV_SQRT_128  0.08838834764831845f
// turbo_cpu_s1[128], turbo_cpu_s2[128]  -> copy VERBATIM from ggml-turbo-quant.c:204-216
```
`nearest_centroid_3bit(v)`: return index via the 7 midpoints (idx 0..7). Sign bit = idx&4, magnitude = idx&3.

---

## (5) Microtest-first validation plan (CPU reference is the oracle)

The four exported CPU refs (`quantize_row_turbo3_0_ref`, `dequantize_row_turbo3_0`, `turbo_cpu_fwht_inverse`) are the authoritative round-trip. `tests/test-turbo-quant.c` is **not** wired into any build — replicate it in q27's `test_kernels.cu`.

1. **Struct/const gate**: `static_assert(sizeof(block_turbo3_0)==50)`; assert field offsets 0/2/34; assert centroids/midpoints/`s1`/`s2` bit-match the CPU ref (compile-time or a memcmp test).
2. **CPU round-trip oracle** (per 128-vector): `quantize_row_turbo3_0_ref(x,buf,128)` -> `dequantize_row_turbo3_0(buf,y,128)` -> **`turbo_cpu_fwht_inverse(y,128)`** -> compare `y` vs `x` (MSE/cosine). The inverse WHT is MANDATORY (dequant output is in the rotated domain).
3. **q27 store-kernel parity**: run q27's new store kernel on the same `x`, D2H the 50-byte block, `memcmp`/ULP-compare `qs`/`signs` against `quantize_row_turbo3_0_ref` output. Expect bit-identical indices (only fp16 norm may differ by rounding — assert ≤1 ULP).
4. **q27 read-kernel parity**: feed a CPU-`ref`-produced block to `fd2_ld8_turbo3`; assert dequant == `dequantize_row_turbo3_0` element-wise.
5. **End-to-end rotate invariance**: with q27's Q-forward-WHT + turbo3 K store + fd2 + output inverse-WHT wired, assert `<Q,K>` scores and attn output match the current fp16 path within tolerance on a synthetic head (this catches sign-array/group-split mistakes that unit tests miss).
6. **Quality gate (fresh — no in-repo oracle for head_dim=256)**: PPL within 5% of the fp16/q8-KV baseline on the q27 model; the only in-repo reference number is `scripts/turbo-quality-gate.sh` (q8_0 baseline 6.111, head_dim=128 only). q27 must establish its own bar.

---

## (6) Ordered Phase-1 change list

1. `blocks.cuh`/new `turbo3.cuh`: embed `block_turbo3_0` (50 B), centroids, midpoints, `s1/s2`, `inv_sqrt`; `nearest_centroid_3bit`, device WHT butterfly.
2. `engine.cuh`: add `bool kv_turbo3`; sizing path `kv_bpr()` (400 for turbo3) at the ~14 cache allocs (`503-504`, `678-679`, `711-713`, per-stream A(&k/&v)); `memset` sizes updated.
3. `blocks.cu`: add `k_kv_store3` cooperative store kernel (item 2) + `kv_store3` launcher (`grid=dim3(8,2),128`); keep `k_kv_store` for fp8/f16.
4. `engine.cu` decode + `prefill.cu`: insert **forward-WHT on Q** after rope; call `kv_store3` (folds K/V forward-WHT) when turbo3; insert **inverse-WHT on attnout** after combine. Add a small `wht_fwd`/`wht_inv` kernel (per-128-group, N_HEAD heads for Q, N_HEAD for output).
5. `spec3.cu`: add `k_attn_fd2_turbo3<NW>` + `fd2_ld8_turbo3` (item 3); dispatch in `attn_decode3_fd2` on `kv_turbo3`. **fp8/fp16 `fd2_ld8`/`k_attn_fd2` untouched -> canonical bitwise gate preserved** (turbo3 is a new third path).
6. `test_kernels.cu`: items 5.1-5.4 wired as unit tests; link the three CPU refs from `ggml-turbo-quant.c`.
7. Env gate `Q27_KV=turbo3` (opt-in, like `Q27_KV=fp8`), default off.

**Deferred to fdmma phase:**
- turbo3 in the fp8-MMA shared-KV verify attention (`fdmma.cuh`) — the LUT/tensor-core dequant of packed blocks and the tile-load even-pair extraction.
- InnerQ per-channel equalization (calibration; auto-off at head_dim>128 regardless).
- Any **q8_0-K / turbo3-V split** (asymmetric cache) — see risk.
- Prefill-attention turbo3 read (`prefill.cu` k_attn_prefill_mma) beyond the store side, if it uses a distinct read kernel.

---

## SINGLE BIGGEST RISK / UNKNOWN

**turbo3-K quality at q27's GQA ratio = exactly 6.0.** The reference's own auto-asymmetric rule (`llama-kv-cache.cpp:156`) upgrades K->q8_0 at **gqa_ratio ≥ 6** because low-KV-head models make turbo3-K catastrophic (Qwen2.5 7:1 -> PPL 2887 vs 7.4; Mistral 4:1 -> +4.4% fine). q27 sits **on the threshold**, and head_dim=256 turbo3 has **zero quality data anywhere in-tree** (only head_dim=128 models measured; CUDA turbo-KV correctness never validated per the merge doc). If turbo3-K craters at ratio 6, Phase 1's symmetric turbo3-K+V is unusable and the real prod config becomes **q8_0-K / turbo3-V**, which fd2 currently cannot express (single `CT` for both K and V) — that split is a larger refactor than Phase 1 scopes. **Mitigation: gate the very first bring-up on the item-5.6 PPL test with turbo3-V-only vs turbo3-K+V vs fp16 baseline before building the full fd2 turbo3-K read path; if K craters, pivot Phase 1 to turbo3-V-only + keep fp8/f16 K.**

Relevant files: `/mnt/ai/projects/q27/src/spec3.cu` (fd2 read, 297/328), `/mnt/ai/projects/q27/src/blocks.cu` (kv_store, 79/89), `/mnt/ai/projects/q27/src/engine.cuh` (32/435/711/845-847), `/mnt/ai/projects/llama-cpp-turboquant/ggml/src/ggml-turbo-quant.c` (CPU refs 42/170/204-260/276/340), `/mnt/ai/projects/llama-cpp-turboquant/ggml/src/ggml-cuda/set-rows.cu:239` (write kernel model), `/mnt/ai/projects/llama-cpp-turboquant/ggml/src/ggml-common.h:286` (struct).
