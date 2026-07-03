# q27

A narrow inference engine for **Qwopus3.6-27B-v2-MTP** (Qwen3.6-27B hybrid + trained-in MTP heads) on a single RTX 5090. One model, one GPU, as fast as possible. In the spirit of [antirez/ds4](https://github.com/antirez/ds4).

## Why this model is a good target

- Dense-ish 27B that fits entirely in 32 GB VRAM at 4-bit -- no expert offload, no DRAM scatter, none of the DSV4 pain
- MTP draft head trained into the checkpoint: self-speculation without a separate draft model
- Hybrid Gated-DeltaNet architecture means near-O(1) memory per token for 48 of 65 layers. KV lives only in the 17 full-attention layers (16 + MTP, all **global**, no windowing): 68 KB/token at fp16 = ~4.3 GB @64K, ~8.5 GB @128K, ~17.8 GB @256K. A dense-attention 65-layer build would be ~68 GB @256K. At fp16 KV the practical allocation ceiling is ~180K; **fp8 E4M3 KV (P2, opt-in via `Q27_KV=fp8`) halves that to 34 KB/token and raises the ceiling to ~355K (was ~370K before P3's 5th GDN buffer set) -- the advertised 262K native fits** (allocates and runs; correctness validated to 361K, see risk 5)
- Measured baseline to beat: llama.cpp (MTP-TurboQuant fork) at 106-127 t/s single-stream

## Architecture facts (ground truth from GGUF metadata)

| | |
|---|---|
| arch | `qwen35` (Qwen3-Next-style hybrid) |
| layers | 65 total: 48 Gated DeltaNet + 16 full attention (every 4th: 3,7,...,63) + 1 MTP layer (64, full attention) |
| hidden | 5120 |
| FFN | SwiGLU, intermediate 17408 |
| full attention | GQA 24Q/4KV, head_dim 256, QK-norm, gated output (attn_q packs Q+gate: 12288 = 2x6144) |
| DeltaNet blocks | attn_qkv [5120,10240] + attn_gate [5120,6144] + conv1d(k=4) + a/dt/alpha/beta (48-dim heads) + ssm_norm + ssm_out [6144,5120] |
| RoPE | partial, dim 64 of 256, sections [11,11,10,0] (M-RoPE; degenerates to standard for text-only), freq_base 1e7 |
| vocab | 248320, embeddings + lm_head untied |
| MTP | 1 nextn layer: eh_proj [10240->5120] combines (embedding, hidden) -> full attn + FFN -> shared lm_head |
| context | 262144 native |

## Performance model

5090 GDDR7 ~1.79 TB/s. Single-stream decode is weight-read-bound.

| Stage | Per-step read | Ceiling | With MTP (~1.9x measured) |
|---|---|---|---|
| llama.cpp Q5_K_M (62% BW eff) | 18.2 GB | 61.6 t/s measured | 106-127 t/s measured |
| q27 Q5-class, 85-90% eff | ~18 GB | ~88 t/s | ~165 |
| q27 custom 4-bit at 85-90% eff | ~14.8 GB | ~103-109 t/s | ~200-225 |
| q27 4-bit **measured** (2026-07-02, +4000 OC) | ~15.5 GB/step | **91.0 t/s plain** (~75% eff) | **188.9** (depth-3 spec, 2.07x) |

The original "~120 t/s ceiling" row implied ~99% BW efficiency and is retired.
Plain decode sits ~15% under the honest 85-90% ceiling; that tail is GDN
recurrence + ~140 small-kernel launches/token, and three attempts on it
(E4 launch geometry, E5 fusions, cp.async) all came back negative.

## Design decisions

- **Weights**: custom 4-bit symmetric groupwise (group 64, fp16 scales), packed for coalesced 128B warp loads, dequant fused into GEMV. Embeddings, lm_head, MTP layer, norms at 8-bit/f32. Repacked offline from the BF16 GGUF.
- **KV cache**: fp16 for the 17 attention layers by default (f32 originally). FP8 E4M3 ships opt-in via `Q27_KV=fp8` (P2): scale-free saturating conversion (measured K amax <= 21.8, V amax <= 118.6 vs the 448 E4M3 max -- per-row scales buy nothing for a float format with that much headroom), same element-indexed layout, all store/load sites templated on the element type. Halves KV bytes (34 KB/token) and cuts long-ctx decode bandwidth (+11% decode @28.5K). NOT lossless -- default stays fp16 so decode canonicals hold bitwise; measured cost is noise-level (corpus PPL -0.05%, logit KL 3.4e-5). DeltaNet recurrent state is tiny and stays f32.
- **MTP**: first-class. Draft + verify in one pipeline under a single CUDA graph. No separate draft context, no re-prefill.
- **Stack**: plain CUDA C++. No CUTLASS, no deps beyond CUDA runtime. Offline repack tool is Python (runs once).
- **Serving**: OpenAI, Anthropic (Claude Code-grade), and OpenAI Responses (Codex-grade) shapes on one binary.

## Milestones

- **M0** DONE -- repack tool: BF16 GGUF -> q27 4-bit format (policy v1.2)
- **M1** DONE -- correctness: greedy decode, output verified vs llama.cpp
- **M2** DONE -- dp4a GEMVs + CUDA-graph decode: 80.1 t/s plain
- **M3** DONE -- MTP speculative pipeline, lossless (token-identical):
  depth-2 drafting, batched verify, 3-perm cyclic state graphs. **146.0 t/s**
  (llama.cpp MTP fork on same model/GPU: 101.5). Stretch target was 165;
  verify-GEMV bandwidth floor makes the remaining gap ~1-2%/iteration work.
- **M4** DONE -- dual lm_head (Q4 draft / Q8 verify), grid merges, device-side
  round bookkeeping. `--fast-head` opt-in: **156.5 t/s**
- **M5** DONE -- HTTP serving: OpenAI + Anthropic + OpenAI Responses, exact
  byte-level BPE tokenizer (gated 21/21 vs llama-tokenize), tool calling
- **E6** DONE -- ungated depth-3 speculation: measured p(d3 | d1,d2 correct)
  = 83.7% offline (docs/E6-design.md), so the round always drafts 3 and
  batch-4-verifies {pending, d1, d2, d3}. 4 GDN buffers under a mod-4 role
  permutation, 4 captured graphs. 3.12 tok/round, **188.9 t/s** @2k
  (204.8 long-gen); 8000-token output bit-identical to depth-2. Also fixed
  two latent bugs found en route: flash-decode scratch under-allocation at
  ctx<4128, and missing ctx guard letting spec rounds write KV rows past
  max_ctx (silent corruption the prefix cache could then reuse).

## Serving

```
make build/q27-server
./build/q27-server model.q27 model.tok --port 8080 --ctx 8192 [--fast-head]
```

Three API shapes on one server:
- **OpenAI**: `/v1/chat/completions`, `/v1/completions` (text)
- **Anthropic**: `/v1/messages` -- native Messages API with thinking blocks
  (Qwopus `<think>` mapped to thinking/signature blocks), tool_use/tool_result,
  input_json_delta streaming. Claude Code-compatible:
  `ANTHROPIC_BASE_URL=http://host:8080 claude`
- **OpenAI Responses**: `/v1/responses` -- Codex CLI-compatible: function
  tools, `custom` freeform tools (apply_patch bridged through a
  one-string-param function), function_call/function_call_output history,
  reasoning items; event set verified against the codex-rs client source.

Codex config (`~/.codex/config.toml`):
```toml
model_provider = "q27"
model = "gpt-5-codex"

[model_providers.q27]
name = "q27 local"
base_url = "http://localhost:8080/v1"
wire_api = "responses"
```

Model tool protocol: tools rendered as JSON in the system `<tools>` block per
the qwen35 chat template; `<tool_call>` output parsed by a streaming splitter
(src/stream_split.h) that also routes `<think>`. Single slot (spec decode is
single-stream), greedy sampling. `--fast-head` trades output exactness for
~7% more t/s.

## Progress log (tg t/s, greedy, token-identical output verified each step)

| change | t/s |
|---|---|
| reference kernels e2e | 43.4 |
| dp4a int8-activation GEMVs | 58.8 |
| coalesced delta state + wide norms + multiblock argmax | 66.5 |
| CUDA-graph token replay, device-chained decode | 75.9 |
| delta_step i-parallel v2 | 80.1 |
| + speculative decode depth-1 (host-driven) | 84.2 |
| + direct-write batched GEMV | 92.2 |
| + parity-pair captured graphs | 109.3 |
| + depth-2 drafting (2.13 tok/round) | 107.3 |
| + grid-merged 3-token small kernels | 115.1 |
| + dual lm_head: Q4 drafts, Q8 verify (v1.3 repack) | 121.1 |
| steady state (128-token bench, 2.39 tok/round) | **133.5** |
| `--fast-head` opt-in (Q4 verify; output differs, coherent) | 143.0 |
| + full grid merges (l2/f16/gates/rope/kv/attn/sigmoid/embed x3) | 145.8 lossless / 156.5 fast |
| + device-side round bookkeeping (1 sync + 16B readback/round) | 146.0 lossless / 156.5 fast |
| E1: display compositor off GPU 0 (cosmic-comp/Xwayland stole ~10%) | **157.4** lossless / **168.5** fast |
| warp-cooperative decode attention (coalesced K/V) | **168.6** lossless @2k; 65.8 @8k ctx (~2x long-ctx) |
| flash-decode (split-K, K/V shared across GQA heads) | **173.1** @2k / **159.6** @8k ctx lossless; 178.1 fast |
| fp16 KV cache (attn + MTP) | 169.7 @2k / 159.7 @8k; halves KV bytes, -2.1GB @32k ctx |
| E2: GDDR7 mem offset +4000 (tools/mem_oc.py, volatile) | **176.6** lossless / **185** fast-head; prefill ~+6% |
| E6: ungated depth-3 speculation (3.12 tok/round; batch-4 verify) | **188.9** @2k (128-tok) / **204.8** long-gen; 8000-token output bit-identical to depth-2 |
| P1: int8 tensor-core prefill GEMM (mma.sync m16n8k32) | prefill **1380 t/s** @600 / **1384** @4K (dp4a: 592/580, 2.35x); cold 28.1K TTFT **63.8s -> 35.7s**; PPL delta vs dp4a +0.04% (fp reorder only) |
| P1.5: fp16 tensor-core flash-attention prefill (m16n8k16) | cold 28.1K TTFT **35.7s -> 24.3s** (63.8s at day start, 2.63x total); prefill 1408 @600 / **1508** @4K; PPL 7.2139 (+0.006% vs exact); needle 3/3 @64K; kernel review: 0 confirmed bugs |
| v1.4 quant policy (ssm_out + attn_output -> Q8, +0.98 GB) | PPL **7.1928** (-0.29%); decode **+3.3%** on 2000-tok soak (acceptance 3.47 -> 3.67 t/round -- cleaner residual writers agree better with the MTP draft head); all gates re-derived |
| P2: fp8 E4M3 KV cache (opt-in, `Q27_KV=fp8`) | decode @28.5K ctx **105.7 -> 117.2 t/s** (+11%); 2K soak 208.3 vs 210.4 (-1%, acceptance 3.64 vs 3.67); ctx ceiling **~180K -> ~370K** (262K native fits); PPL 7.1889 (-0.05%), needle 3/3 @55K, logit KL 3.4e-5 |
| P3: depth-4 speculation (batch-5 verify, mod-5 perm) | 2K soak **210.4 -> 218.6 t/s** (4.36 t/round, 71% of rounds accept 5); 28.5K-depth fp8 **117.2 -> 126.6** (+8%; +19.8% vs pre-P2); canonical md5 unchanged (lossless); gate: p(d4\|prefix-3) measured 97.4% |
| P4: split-position FA prefill (SM-starvation fix) | attention kernel **1.93x** @26.6K; 128K prefill **~1.96x** (153 -> 78s); cold 28.5K TTFT **24.7 -> 21.4s**; cold **361.5K request 1324 -> 764s** (~12.6 min, needle exact); split-vs-exact 1.9e-5, combine cost 0.1% |

Headline numbers from E2 onward include the +4000 GDDR7 offset (~+4%; stock
depth-3 ~181 est. from the E2 ratio). Caveat: consumer GDDR7 has no ECC, and
weights load once -- a bit flipped by a marginal OC during a long session is a
persistent silent error the token-identity gates cannot catch (they compare
against the same resident state). **OBSERVED 2026-07-02:** after ~30 min of
sustained VRAM load (long-context benches), one run of the canonical
128-token gate came out WRONG (divergent text, acceptance down) on an
unchanged binary, then all subsequent runs -- ~2 min of idle later -- were
correct again; a cross-build check confirmed the binary was innocent. A
one-shot soft error from the heat-marginal +4000 offset is the only
explanation that fits. **Resolution: the daily offset is now +3000
(tools/mem_oc.py 3000, 2026-07-02) -- E2 measured gains flattening past
+3000, and depth-3 confirms: 188.2 t/s @2k at +3000 vs 188.9 at +4000 vs
183.1 stock. The marginal band above +3000 bought ~0.4% and produced the
soft error.** +4000 only for short supervised benches. A device-side
weight-checksum tool (verify after load, recheck on demand) is on the
roadmap as the detection mechanism. Offset is volatile -- reapply after
reboot.

## Prefill (M6)

Batched prefill: 256-token chunks, smem-staged dp4a GEMM (16 rows/block share
one activation tile; per-lane accumulation order matches the serial GEMV
exactly, so prefill is bitwise-identical to the serial path -- gated on
identical continuations). GDN state scans sequentially inside one kernel with
S resident in shared memory; attention runs two-pass softmax in 32-token
sub-batches; MTP warm skips attention/FFN (only the K/V stores matter).

| prompt | serial | batched | speedup |
|---|---|---|---|
| 512 | 76 t/s | 567 t/s | 7.5x |
| 4096 | 53 t/s | 453 t/s | 8.5x |

**Prefix cache (M6.5)**: GDN state + conv rings snapshotted after prefill
(attention/MTP KV rows are append-only, so prefix rows stay valid); next
request LCP-matches the snapshot and prefills only the suffix. Claude Code
turn 2 on a 26.7k-token context: **1.3s** (26,670/26,693 tokens reused).
Unconditionally correct: any mismatch falls back to full prefill; warm-vs-cold
continuations gated identical.

Real-world (Claude Code `claude -p`, 26.7k-token system prompt):
| | TTFT |
|---|---|
| pre-M6 (serial prefill) | 15-min timeout, 0 tokens |
| M6 (batched) | 139s |
| + coalesced attention prefill | 90s |
| + GEMM tuning + FA-lite attention | 61s |
| turn 2+ with prefix cache | **1.3s** |

## Roadmap (reordered 2026-07-02 after external review)

Context for the ordering: decode is in good shape (188.9 lossless = 1.86x the
fork; three consecutive micro-opt attempts on the remaining tail came back
negative). The user-visible gaps are cold prefill (~8x behind the fork's
tensor-core GEMM; 61s cold TTFT @26.7K, warm turns 1.3s via prefix cache) and
two advertised claims with no measurement behind them. Cheapest-blocking-first:

**P0 -- claims gates: DONE 2026-07-02.**
- Long-context: validated to 64K (risk 5: flat NLL-by-position, +2.0%
  cross-engine at [32K,64K), needle 3/3). `--nll` / `--nll-long` added to the
  CLI (batched teacher-forced NLL, protocol-identical to llama-perplexity;
  gated vs a serial reference path).
- PPL delta: **+3.35% vs Q5_K_M** (7.2135 vs 6.9797), marginally over the 3%
  bar -> NEW ITEM below.

**P0.5 -- DONE 2026-07-02: v1.4 = ssm_out + attn_output promoted to Q8.**
Sensitivity study (full-corpus --nll per candidate repack; baseline v1.3
PPL 7.2139, Q5_K_M 6.9797):

| candidate (Q4 -> Q8) | +GB | PPL | verdict |
|---|---|---|---|
| ffn_down first4+last4 | 0.35 | 7.2079 | dud |
| ffn_down first8+last8 | 0.69 | 7.2074 | dud |
| ffn_down ALL (ceiling probe) | 2.76 | 7.1466 | 29% of gap, unshippable ratio |
| attn_qkv (GDN in-proj) | 1.22 | 7.2396 | **WORSE than baseline** |
| residual writers, late-only | 0.12 | 7.2112 | dud (not concentrated) |
| **residual writers ALL = v1.4** | **0.98** | **7.1928** | shipped |

Findings: (1) ffn_down sensitivity is spread uniformly across layers -- no
cheap subset exists; (2) promoting the GDN in-projections HURTS: v1.3's Q4
errors there partially cancel against downstream quant errors, and breaking
the correlation costs +0.36% PPL; (3) v1.4's decode got FASTER (+3.3% on a
2000-token soak, 203.9 -> 210.6 t/s) because cleaner residual writers raise
MTP draft acceptance (3.47 -> 3.67 tokens/round) by more than the +0.98 GB
of reads cost -- quant policy and speculation acceptance are coupled.
Remaining gap to Q5_K_M: +3.05% (was +3.35%); closing more via uniform
promotion has terrible ROI (see ceiling probe) -- importance-weighted scales
(AWQ-style) are the real path if quality ever becomes the priority.
v1.3 archived at qwopus-27b-mtp-v13.q27 (old canonical sequences apply to
it); all canonical gates re-derived for v1.4 in this commit.

**P1 -- DONE 2026-07-02: prefill 2.35x, cold 28.1K TTFT 63.8s -> 35.7s.**
Kernel: k_gemm_mma_T (prefill.cu), one mma.sync per 32-element quant block,
Q4 nibbles unpacked to s8 (offset folded) in smem staging; unit-gated at
1e-6 vs dp4a on all four weight shapes; full-corpus PPL +0.04% (fp reorder);
needle 3/3 @64K; decode canonicals untouched; Q27_PREFILL=dp4a keeps the
exact-reference path. nsys at 28K now shows attention prefill dominant
(~16.4s of the remaining ~36s wall) with delta_scan next (~3.8s) -- the
long-context TTFT lever is attention prefill now, NOT GEMM. GEMM tile
tuning (ldmatrix, cp.async, wider N tile to cut the 8x weight re-read)
still pays at short ctx but is deferred. Original plan below for reference.

**P1 (original plan) -- prefill via int8 tensor-core MMA:**
mma.sync m16n8k32 s8s8s32 (sm_80+ PTX, works on sm_120) replaces dp4a in the
prefill GEMM. CORRECTION (2026-07-02, found during implementation): the
within-chunk int32 dots are exact under MMA, but the per-32-block activation
scales force one fp multiply-add per chunk, and dp4a's fp structure per
output (32 stride-32 lane-partials + shuffle tree, matching serial GEMV)
cannot be reproduced by an MMA accumulator without infeasible register cost.
So the ORIGINAL review point stands: the MMA path needs a tolerance gate,
not the bitwise gate. Divergence is pure fp-reorder noise (~1e-6 rel;
integer dots exact), so tolerance is tight. Gates: unit test (MMA chunk
dots == dp4a exactly), --pfdbg state maxdiff at fp-noise level, full-corpus
--nll PPL delta < 0.02% vs dp4a prefill, needle 3/3, Q27_PREFILL=dp4a env
keeps the exact path as reference/fallback, decode canonicals untouched.
Still no CUTLASS; "plain CUDA" stays true. dp4a prefill: ~590 t/s @512 /
~300 @26K; fork reference 2,300-2,400. Realistic target 2-4x -> cold 26.7K
TTFT from 61s toward ~15-25s.

**P2 -- DONE 2026-07-02: fp8 (E4M3) KV cache, opt-in via `Q27_KV=fp8`.**
Halves KV (68 -> 34 KB/token); measure-first probe (`--kvstats`, 8K wikitext
tokens through prefill) showed K amax <= 21.8 / V amax <= 118.6 vs the E4M3
448 saturation bound with negligible sub-denormal mass, so the design is
scale-free saturating conversion -- per-row scales only pay when a float
format is range-limited, and this one is not. All 3 store kernels + 3
attention consumers (flash-decode, MMA and lite FA prefill) templated on the
cache element type; fp16 stays the default and its canonicals hold bitwise
(md5 re-verified post-refactor). Gates: unit tests prove store == host
saturating cvt bitwise and fp8 kernels == fp16 kernels on the dequantized
cache bitwise (E4M3 is exact in fp16); corpus PPL 7.1889 vs 7.1928 (-0.05%,
noise); `--nll-long 65536` buckets flat and within 0.3% of fp16; needle 3/3
@55K through the server; logit A/B @512-tok prompt: cosine 0.9995, top-1
exact, KL 3.4e-5; pf/pfcache identical under fp8; acceptance 3.64 vs 3.67
t/round (-1% @2K soak). Wins: decode @28.5K ctx 105.7 -> 117.2 t/s (+11%),
alloc ceiling ~180K -> ~370K (262K native allocates and runs; cold TTFT
unchanged). Risk-6 tolerance-gate machinery now exists for future fp paths.
Correctness then validated to the new ceiling (see risk 5): fp16-vs-fp8 NLL
A/B flat to 160K (+-0.06% per bucket), fp8 NLL flat to 370K on a 783K-token
corpus, needle 6/6 on a 361.5K-token haystack including two placements
beyond the 262K native limit; 19.1 t/s decode at 361K depth.

**P1.5 -- DONE 2026-07-02: cold 28.1K TTFT 35.7s -> 24.3s.**
k_attn_prefill_mma: fp16 flash-attention prefill on mma.sync.m16n8k16 --
block per (kv head, 16-token tile), 6 warps = 6 GQA q-heads, K/V slabs
smem-shared, S->P register-identity reuse, fp32 softmax/O. Attention at 28K:
~16s -> ~4s. Gates: unit A/B vs FA-lite at 3.8e-4 (edge shapes T=23/base=37),
PPL 7.2139 (+0.006%), needle 3/3 @64K, canonicals untouched, 3-lens
adversarial kernel review with 0 confirmed findings; Q27_ATTN_PF=lite
fallback. Weight-checksum tool also landed: baseline at load, --verify-weights
(CLI) and /health?verify=1 (server), 867 tensors in ~20 ms. Review minors
fixed: prompt>ctx now refused in generate() (was a silent KV overrun,
pre-existing), dead pf_scratch removed (~100 MB @32K). Remaining minors
noted: both attention prefill kernels hardcode gqa=6/head_dim=256 (fine for
this model, silent wrongness if reused elsewhere); Q fp16 saturation above
65504 is theoretical for rmsnormed heads. Next long-ctx cost: delta_scan_T
(~3.8s @28K), then GEMM tile tuning at short ctx.

**P3 -- DONE 2026-07-02: depth-4 speculation (2K soak 210.4 -> 218.6 t/s,
28.5K-depth fp8 decode 117.2 -> 126.6 t/s).**
- Round-cost audit (nsys, --cuda-graph-trace=node): the argmax hypothesis was
  WRONG -- all 7 argmax chains total ~0.03 ms/round (0.2%); graphs already
  amortize launch overhead. Real budget: 4-lane batch GEMVs ~65%, the 3
  sequential MTP draft passes ~13% (of which ~1.1 ms = three 636 MB Q4
  draft-head reads, bandwidth-irreducible per pass), delta_step ~5%,
  small-kernel soup the rest. Argmax batching NOT built (measured pointless).
- Depth-4 gate: --stats pass-4 chain measured p(d4|prefix-3) = 97.4% overall
  (bar was ~60%), projecting +0.89 t/round ungated. BUILT: 5th lane (e),
  batch-5 verify, 4-pass draft chain, 5 GDN state buffers with mod-5 perm,
  5 captured graphs, ctx guard P+6. Soak: 4.36 t/round (was 3.67), 71% of
  rounds accept all 5; round cost +14% (4th MTP pass + 5-lane scaling) ->
  net +3.9% @2K, +8% @28.5K depth (rounds are weight-dominated there, so the
  extra lane's KV sweep doesn't bite). Canonical n=128 md5 UNCHANGED --
  greedy output stays bit-identical, as with E6. Cost: +610 MB (5th GDN
  buffer set) -> fp8 ctx ceiling ~370K -> ~355K; 262K native still fits.
- pf_scratch remnants removed (dead `scratch`/`max_ctx` params dropped from
  attn_prefill_T; the allocation itself died in P1.5).

**P4 -- DONE 2026-07-02: split-position FA prefill (long-ctx prefill ~2x;
cold 361.5K request 1324s -> 764s).** nsys on a pure 26.6K prefill showed
attention at 27% and climbing quadratically (~85% at 361K), and the cause was
SM starvation, not bandwidth: grid (4 kv heads, chunk/16 tiles) = 64 blocks
of 6 warps on a 170-SM part, ~20 TFLOPS sustained (~10% of fp16 MMA peak).
Fix: gridDim.z splits each tile's causal position range into PP-aligned
slices; each split emits an unnormalized {m, l, O[256]} partial per (q-head,
row) and k_attn_pf_combine merges (flash-decode's trick applied to prefill).
nsplit auto-scales with depth ((base+SB)/4096, capped 8, Q27_PF_SPLIT
overrides; 1 = bit-identical pre-split path, always used at short ctx).
Partials scratch 51 MB; combine cost 0.1%. Measured: attention kernel
6.01s -> 3.12s @26.6K (1.93x); 128K prefill wall ~153s -> ~78s (1.96x); cold
28.5K TTFT 24.7s -> 21.4s (attention is only ~27% there); cold 361.5K
1324s -> 764s (1.73x, ~12.6 min) with the deep needle still retrieving
exactly. Gates: unit split=5-vs-1 at 1.9e-5 (empty tail slices exercised),
fp8==fp16(deq) bitwise identities hold under splits, canonical md5 exact,
pf/pfcache IDENTICAL, nll-long 32K split-on/off equal to the 4th decimal,
needle 3/3 @55K with verbatim-identical answers. Remaining long-ctx costs:
delta_scan_T (~50s @361K), GEMM tile tuning at short ctx.

## Risk register

1. **Gated DeltaNet decode kernel** is the new risk center (was "simple dense" until we read the GGUF). llama.cpp's implementation is the semantic reference; validate per-layer.
2. 4-bit quality on a 27B: keep sensitive tensors high-bit, add importance-weighted scaling if PPL regresses > ~3% vs Q5_K_M. **STATUS: MEASURED + MITIGATED 2026-07-02** -- v1.3 measured +3.35% vs Q5_K_M (7.2135 vs 6.9797, identical tokens/protocol via `--nll`); v1.4 policy (residual writers to Q8, chosen by a 6-candidate sensitivity study -- see roadmap P0.5) lands at **7.1928 = +3.05%**, and decode got faster (+3.3%, acceptance-coupled). Still marginally over the 3% bar; the study shows uniform promotion cannot close the rest at acceptable cost (ffn_down ceiling probe: 29% of gap for +2.76 GB) -- importance-weighted scales are the documented path if ever needed. The t/s comparison remains bit-width-assisted (15.8 GB reads vs 18.2).
3. M-RoPE sections must match exactly or long-context quality silently degrades.
4. MTP acceptance rate must survive quantization (draft and verify disagreeing more = less speedup). STATUS: measured -- Q4 vs Q8 draft-head argmax agreement 98.1% (E3); depth-3 runtime acceptance 85.7%.
5. **Long-context correctness: VALIDATED to 361K (2026-07-02, fp8 KV).** Original 64K validation: (a) `--nll-long 65536` flat buckets; (b) cross-engine vs llama-perplexity +2.0% at [32K,64K) (smaller than the short-context delta, so no length-dependent divergence); (c) needle 3/3 @55K. Extended after P2 with a 783K-token corpus (War and Peace, tokenized with the model's own vocab; `--nll-long` buckets now reach 320K+): (d) fp16-vs-fp8 NLL A/B at 163840, bucket deltas within +-0.06% at ALL depths (fp8 cost does not grow with position); (e) fp8 `--nll-long 370000` single pass: buckets flat 7.2-7.6 to 256K, then a graceful +3% drift beyond the native 262K (7.89 at 256-320K, 7.69 at 320K+) -- no blowup even in RoPE-extrapolation territory; (f) needle retrieval **6/6 on a 361.5K-token haystack** (depths 35K/124K/213K/248K within native + 276K/337K BEYOND native, all exact, think traces naming surrounding chapters); (g) `--kvstats 131072`: K amax 21.7 / V amax 128.4, zero E4M3 saturation at depth. Decode at 361K depth: 19.1 t/s (fp8, spec). Caveat: each distinct long prompt is a full cold prefill (~22 min @361K) -- the GDN recurrent snapshot makes the prefix cache all-or-nothing, so mid-document divergence cannot reuse state. Risk 3 is covered to 64K by (a)+(b).
6. **fp-precision paths break the bitwise gate.** Batched prefill is currently bit-identical to serial because dp4a's int32 block sums are order-independent and the per-group fp scale-and-add matches serial order. RESOLVED for prefill: the roadmap's int8 mma.sync path keeps int32 accumulation, so the bitwise gate survives tensor-core prefill (P1). **RESOLVED for fp8 KV (P2, 2026-07-02):** the tolerance-gate machinery now exists and passed -- logit A/B vs the fp16 path (cosine 0.9995, top-1 exact, KL 3.4e-5 @512-tok prompt), corpus PPL delta -0.05%, needle 3/3 -- and fp8 ships opt-in (`Q27_KV=fp8`) with the fp16 default still bitwise-canonical. The same gate recipe applies to any future fp16/fp8 MMA decode path.
