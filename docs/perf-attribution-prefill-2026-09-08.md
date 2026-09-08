# Prefill recon, 2026-09-08: where q27's prefill wall goes on Claude Code traffic, and where we stand vs ninfer

Status: RECON, no code. Inputs: the [req] journals of the four 12-instance
Claude Code runs of 09-08 (production DFlash2 and ladder at xhigh; campaign
DFlash2-Q8 and ladder at medium), ninfer's request logs from the 09-07
campaign (same 12 instances, medium), a cuBLASLt ceiling probe run today on
the 5090, and a read of the prefill code path. Supersedes the ROI note in
docs/plans/2026-08-17-prefill-performance.md section 6 ("measure the miss
rate on real traffic") -- this is that measurement.

## 1. What prefill costs on real traffic

Production DFlash2, xhigh, 299 requests over 12 SWE-bench instances:

| | prodd2 xhigh | prodlad xhigh | q27d2q8 medium | q27lad medium |
|---|--:|--:|--:|--:|
| requests | 299 | 271 | 209 | 218 |
| prompt tokens | 11.03 M | 9.31 M | 6.70 M | 6.86 M |
| cached (prefix hit) | 93.3% | 92.4% | 89.2% | 90.0% |
| computed prefill tokens | 734 K | 711 K | 725 K | 686 K |
| prefill wall | 255 s | 239 s | 234 s | 225 s |
| decode wall | 1640 s | 1292 s | 606 s | 1074 s |
| prefill share of prefill+decode | 13.5% | 15.6% | 27.9% | 17.3% |
| aggregate prefill tok/s | 2876 | 2980 | 3091 | 3051 |

Prefill is 13-16% of the engine's wall at production's xhigh effort and up
to 28% at medium (less thinking, same prefill). Per-request it is the whole
user-visible stall: a 7 s first turn, a 15.9 s mid-conversation re-prefill.

By computed-token size (prodd2; the other three runs match within a few %):

| pf tokens | n | tok/s | mean ms | share of prefill wall |
|---|--:|--:|--:|--:|
| 64-256 | 69 | 1376 | 137 | 3.7% |
| 256-1K | 167 | 2094 | 258 | 16.9% |
| 1K-4K | 37 | 2503 | 630 | 9.1% |
| 4K-16K | 8 | 2741 | 2488 | 7.8% |
| 16K-65K | 18 | 3248 | 8864 | 62.5% |

Two regimes: a ~120 ms fixed floor that dominates the 64-256 bucket
(64 layers of eager launches for a handful of tokens), and the big cold
prefills at ~3250 tok/s which are 62-73% of the prefill wall in every run.

## 2. The miss anatomy: two thirds of the prefill wall is cache policy

Classifying every request by conversation (the conv= fingerprint):

| prodd2 xhigh | requests | prefill wall | share |
|---|--:|--:|--:|
| first turn of a conversation (12 instances + 18 one-shot side requests) | 30 | 86 s | 33% |
| returning turn, previous request was the SAME conversation | 263 hits, 0 misses | 89 s | 35% |
| returning turn, previous request was ANOTHER conversation | 0 hits, 6 full misses | 80 s | 31% |

The same split holds in all four runs (eviction misses 31-36%, first turns
33-42%). The rule is exact: a returning turn hits if and only if the
immediately preceding request belonged to the same conversation. Every
time Claude Code interleaves one of its ~350-token side requests (a
different conversation: title generation, safety classifier, etc.), the
next turn of the main conversation re-prefills from zero -- 28 K tokens at
8.5 s, 48 K tokens at 15.7 s, four times in a row for one instance.

Why (src/engine.cuh:4869-4988): the two VRAM tiers, the P8 stable snapshot
and the P9 checkpoint ring (16 slots, 4096 interval), are per-slot and are
invalidated when a foreign prompt re-prefills the slot's rows [0, 350) --
the ring entries no longer cover a prefix of the new prompt and are cleared
(4960-4966). Every request in every run landed on slot 0 (sequential
traffic; Q27_BATCH=0 has one slot anyway), so multi-slot did not help. The
P16 disk tier and P16c host-RAM tier, which survive this, are opt-in
(`--prefix-cache <root>`, `--prefix-cache-ram-gb`) and are not in the
production command. The P16b system-block entry would also serve first
turns: measured 07-24 on real Claude Code, 25,750-token first turn ->
5,270 computed (restore 237 ms), and CC's system+tools block was ~21.4 K
tokens then.

ninfer on the same instances (09-07, medium): 95.9% reuse vs our 89-93%.
Its log shows `shared_stable_prefix` hits of 22,449 tokens on 9 of 14
early big requests -- the shared system block is stable enough across
these sessions to be reused -- and `private_endpoint` hits for returning
turns regardless of interleaving (its context_cache keeps 8 host state
slots / 8 GB host KV). That is the mechanism q27 has shipped but switched
off.

What is left after policy: ~89 s of the 255 s is genuine new content (tool
results, the assistant's re-rendered previous turn) at 2100-2800 tok/s,
plus whatever part of first turns is not shared. Kernel speed applies to
that and to every miss policy cannot avoid.

## 3. Kernel speed vs ninfer (nvfp4) on the same traffic class

ninfer's request log carries computed_prefill_tokens and prefill seconds,
so the campaign traffic gives a same-class comparison with no extra run:

| computed tokens | q27 (4 runs) tok/s | ninfer d2 tok/s | ninfer mtp tok/s | ratio |
|---|--:|--:|--:|--:|
| 64-256 | 1376-1456 (131-137 ms) | 1435 (96 ms) | 1190 (117 ms) | ~1.2x on the floor |
| 256-1K | 2094-2211 | 3536 | 3226 | 1.6x |
| 1K-4K | 2503-2843 | 5007 | 4654 | 1.8x |
| 4K-16K | 2741-3039 | 3364 (n=4) | 5032 | 1.2-1.7x |
| 16K-65K | 3248-3338 | 7153 (25-34 K) | 5981 (incl. a 63 K at 4100) | 2.2x |

At the 25 K cold first turn: ninfer 3.5 s, q27 7.2 s. The 08-27 cross-engine
run put vLLM (modelopt NVFP4) at 2.5x q27's cold TTFT at 26 K/51 K and
ninfer's INT8 artifact slightly below q27; the 08-15 club-3090 bench had
ninfer nvfp4 at 2.9x q27 at 10 K and ninfer int at 0.86x. So the gap to
ninfer/vLLM is the fp4 format on tensor cores plus GEMM kernel quality, not
engine structure; see section 4 for how much is kernel quality alone.

### 3b. How ninfer's prefill is built (read of ninfer-master, nothing run)

Published (docs/performance.md, Qwen3.8-27B nvfp4, int8-g64 KV, chunk 1024,
reuse off): 8,340 tok/s at 7.7 K, 5,298 at 64.5 K, 3,545 at 130 K, 2,203 at
260 K. Their groupwise-int tier: 3,275 at 7.7 K, 1,610 at 130 K. q27's
`--pf` numbers on the same model: 3,201 at 16 K, 1,834 at 128 K -- i.e.
q27 == ninfer-int, and ninfer-nvfp4 is 2.5x at short depth and 1.9x at
130 K (attention's share grows and no format touches it).

Structure, for what transfers and what does not:

- Same skeleton as ours: eager prefill (decode-only graphs), one 1024-token
  chunk per host round trip with a full stream sync after every chunk, a
  single prefill lane, no cross-request batching, no split-K, no
  fused chunked-prefill+decode. Nothing structural explains the gap.
- GEMM: block-scaled fp4 `mma.sync m16n8k64 mxf4nvf4` (no cuBLASLt/CUTLASS/
  tcgen05), W4A4 in prefill as in decode. At T >= 1024 (and T % 256 == 0)
  a warp-specialized TMA kernel: BlockM 256 x BlockN 128 x BlockK 128,
  3-stage mbarrier pipeline, 8 consumer warps + 128 producer threads.
  Smaller T falls to cp.async/ldmatrix schedules (M32-M128 x N64-N128,
  BlockK 256). Closed shape enum, no generic fallback.
- Fused epilogues remove whole passes: QKV+gate scatter, GDN qkv/z split,
  SwiGLU (gate/up in one kernel), residual add.
- Activations re-quantized to fp4 through HBM before every GEMM (4 passes
  per layer) -- a cost, not a saving; same class as our double `qxT`.
- Prompt attention: own kernel, Br = Bc = 64, grid (T/64, 24 heads), no KV
  split; int8-g64 KV does QK^T in int8 MMA with a Hadamard rotation, V to
  fp16. 12+4 warp-specialized producer/consumer variant for the nvfp4 KV.
- GDN: chunked (64) matmul formulation in bf16/fp32, three kernels
  (WY/UT prepare, sequential state passing, output), output sized to one
  resident wave.
- Prefix reuse is checkpoint-exact (StateImage + KV coverage + exact
  tokens), all-or-nothing per continuation; the context_cache keeps 8 host
  state slots / 8 GB host KV, which is why interleaved side requests do
  not evict a conversation (section 2).

Transferable: the TMA/warp-specialized GEMM shape (section 4 says int8 has
the headroom to use it), the fused epilogues, and host-resident state for
the cache. Not transferable without a format change: the fp4 operand.

## 4. Vendor ceiling: the int8 GEMM has 2x headroom without any format change

The 08-17 plan's P0 ("look up the dense int8 peak, compute percent of
peak") never ran. Measured today instead with cuBLASLt on the prefill
projection shapes (tools/cublaslt_peak.cu, device 0 = 5090, 170 SMs,
2407 MHz base, production serving idle):

| shape (N out x K in) | M | int8 TOPS | fp8 TFLOPS | fp16 TFLOPS |
|---|--:|--:|--:|--:|
| attn q+gate 16384 x 5120 | 1024 | 648 | 721 | 224 |
| attn q+gate | 4096 | 832 | 823 | 239 |
| attn out 5120 x 8192 | 1024 | 871 | 799 | 231 |
| ffn gate/up 17408 x 5120 | 1024 | 711 | 680 | 233 |
| ffn gate/up | 4096 | 896 | 798 | 239 |
| ffn down 5120 x 17408 | 1024 | 897 | 786 | 230 |

q27's `gemm_q4_T` (int8 MMA on smem-dequanted q4 weights, int8 g64
activations) measured 280-322 TFLOPS on these shapes, flat in M
(BUILDLOG 2026-08-17). That is 36-48% of what cuBLASLt reaches in the same
precision at M=1024 -- 2.0-2.8x headroom in the GEMM itself. The plan's
Lever 2 estimate ("~25% headroom") assumed a peak half this size and is
retracted. cuBLASLt's int8 at 650-900 TOPS sits where our own fp4
microbench kernel landed (785-868 TFLOPS); fp4 has a further 2x of tensor
peak above that, but a vendor-class int8 kernel on the EXISTING weights
reaches the level ninfer's fp4 GEMM runs at today. Caveat: cuBLASLt reads native int8
operands; a W4A8 kernel pays for the in-kernel dequant and the g64 scale
fold, so the realistic target is ~600-700 TOPS, i.e. ~2x on the GEMM share.

GEMM share of prefill (docs/perf-attribution-p14.md): 67% at 16 K, 35% at
128 K; at this traffic's 20-50 K cold prompts roughly 55-60%. A 2x GEMM is
therefore ~1.4x on the cold prefill wall at 25-50 K, ~1.25x at 128 K.

## 5. Code-path hot spots (src/prefill.cu, src/engine.cuh; nothing run)

- Eager, ~1600 launches per 1024-token chunk, no graph capture
  (prefill.cu:857). Sets the ~120 ms floor for the 64-256-token turns
  (20% of requests).
- `qxT` launches BOTH quantize_x (g32) and quantize_x_g64 on every
  projection group (engine.cuh:3775-3779); the g32 output is unused on the
  default g64 route. 2.3-4.3% of prefill.
- `mtp_warm_T` runs per chunk unconditionally (engine.cuh:5013, 5042) --
  eh_proj 5120x10240 + k/v GEMMs -- also when DFlash2 is the drafter and
  the MTP head is never consulted. ~1.5%.
- `k_gemm_mma_T` (prefill.cu:248): MR=64, KS=128, single-buffered, no
  cp.async, no ldmatrix/swizzle; the comment records that double-buffering
  and launch_bounds measured slower IN THIS SHAPE -- a local optimum, which
  is what section 4 says a different shape escapes. pf4.cu already carries
  the modern structure (128x128x256, 8 warps, 2-stage cp.async).
- `k_delta_wy` (prefill.cu:2890): 192 blocks on 170 SMs, 16 serialized
  WY_C=64 chunk steps per GDN layer per 1024-chunk, warp-0-only forward
  substitution inside each. 6-11% of prefill, length-flat.
- Prefill attention (`k_attn_prefill_mma_pv8` on the fp8-KV default):
  12.5% occupancy, but the FA2 relayout that reached 25% moved TTFT -1%
  (killed 07-09); the binding constraint is barrier serialization
  (No-Eligible 74%). The warp-specialized async rewrite
  (docs/plans/2026-07-09-prefill-async-rewrite.md, est. +20-27% at 128 K)
  was never built. 13.5% of prefill at 16 K, 54% at 128 K; ~25-30% at this
  traffic's depths.
- Tail-path host work per chunk: ckpt_save D2H, pfx_persist D2H (~50 ms/GB
  when it fires), DFlash2 tap copies (5 x T x 5120 floats D2D) on the last
  two chunks, one d2->ingest per chunk.

## 6. Ranked plan

0. **Switch the shipped cache tiers on (no code).** `/dev/shm` has 62 GB
   free, so the disk tier can be RAM-backed with zero SSD wear:
   `--prefix-cache /dev/shm/q27-pfx --prefix-cache-max-gb 40
   --prefix-cache-ram-gb 16 --prefix-cache-max-tokens 65536` (default
   max_tokens 32768 would refuse to persist the 48 K conversations; step
   stays 8192). Run the same 12 instances at xhigh against production.
   Bars: the 6 eviction misses become restores (<1 s + at most one step of
   re-prefill) and first turns after the first instance hit the
   system-block entry. Expected: prefill wall -40 to -50% (255 -> ~130 s),
   first-turn TTFT ~7 s -> ~2 s, the 15.7 s stalls gone. Risks to watch in
   the [pfx] log: pinned staging at max_tokens (pfx_bytes(64 K) ~2.2 GB per
   slot), persist D2H on the critical path, CC 2.1.170's billing header vs
   the normalizer, and the 07-24 finding that CC's system block varied
   2,144 tokens between sessions (ninfer's 22,449-token shared hits say it
   is stable inside this harness).
1. **Slot routing** (multi-slot configs only): keep a big state's slot for
   its conversation and send side requests elsewhere. Not applicable to
   the single-slot DFlash2 config, so (0) is the general fix.
2. **GEMM rewrite to the vendor shape** (P2 of the 08-17 plan, now with a
   measured 2x target): port pf4.cu's tile structure to a W4A8 int8 kernel.
   Bar: >= 1.6x on the M=1024 projection shapes, output bitwise-identical
   (int32 accumulation is exact; the g64 scale fold must apply at the same
   granularity so the fp32 epilogue does not reassociate). ~1.4x on cold
   prefill at 25-50 K.
3. **Trims:** drop the dead g32 quantize; guard mtp_warm_T on d2_on; ~4-6%
   of prefill, bitwise.
4. **Attention async rewrite** and **k_delta_wy** cross-chunk parallelism:
   the 128 K levers; smaller at this traffic's depths.
5. **Format (native fp4 tier):** still dead on the prior quality verdicts;
   (2) captures most of the fp4 GEMM advantage without it.
