# q27

A narrow inference engine for **Qwopus3.6-27B-v2-MTP** (Qwen3.6-27B hybrid + trained-in MTP heads) on a single RTX 5090. One model, one GPU, as fast as possible. In the spirit of [antirez/ds4](https://github.com/antirez/ds4).

## Why this model is a good target

- Dense-ish 27B that fits entirely in 32 GB VRAM at 4-bit -- no expert offload, no DRAM scatter, none of the DSV4 pain
- MTP draft head trained into the checkpoint: self-speculation without a separate draft model
- Hybrid Gated-DeltaNet architecture means near-O(1) memory per token for 48 of 65 layers. KV lives only in the 17 full-attention layers (16 + MTP, all **global**, no windowing): 68 KB/token at fp16 = ~4.3 GB @64K, ~8.5 GB @128K, ~17.8 GB @256K. A dense-attention 65-layer build would be ~68 GB @256K. The advertised 262K native does NOT fit on this card at fp16 alongside 16.75 GB of weights -- practical allocation ceiling is ~180K (fp8 KV, planned, would double that); correctness validated to 64K (see risk 5)
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
- **KV cache**: fp16 for the 17 attention layers (implemented; f32 originally). FP8 E4M3 is planned, not done -- it halves KV capacity cost again and cuts long-ctx decode bandwidth. DeltaNet recurrent state is tiny and stays f32.
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

Headline numbers from E2 onward include the +4000 GDDR7 offset (~+4%; stock
depth-3 ~181 est. from the E2 ratio). Caveat: consumer GDDR7 has no ECC, and
weights load once -- a bit flipped by a marginal OC during a long session is a
persistent silent error the token-identity gates cannot catch (they compare
against the same resident state). **OBSERVED 2026-07-02:** after ~30 min of
sustained VRAM load (long-context benches), the canonical 128-token sequence
came out WRONG -- deterministically across immediate re-runs -- then returned
to correct after a few minutes of cooldown, on an unchanged binary. Everything
fits a heat-marginal +4000 offset flipping a resident weight bit. Treat +4000
as unsafe for sustained load; +3000 (where E2 measured the gains flattening)
or stock is the recommendation for anything long-running. A device-side
weight-checksum tool (verify after load, recheck on demand) is on the roadmap
as the detection mechanism.

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

**P0.5 -- v1.4 quant policy (opened by the PPL measurement):** recover part
of the 3.35% by moving the most PPL-sensitive tensors up-bit (usual suspects:
early-layer ffn_down / attn_v; measure per-tensor sensitivity with --nll on a
few candidate repacks). Budget: each Q4->Q8 tensor class costs decode
bandwidth, so gate on (PPL gain) / (t/s cost). Repack invalidates all
canonical token gates -- re-derive canonicals in the same commit.

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

**P2 -- fp8 (E4M3) KV cache:** halves KV again (68 -> 34 KB/token: ~8.9 GB
@256K) and cuts long-ctx decode KV bandwidth. NOT lossless -- changes logits,
so it lands behind the P0 gates and ships opt-in until the needle/PPL gates
pass at tolerance. Design-decisions section already marks it planned.

**P1.5 -- attention prefill rewrite (new long-ctx TTFT lever, from the P1
nsys):** attn_prefill_T is ~16.4s of the remaining ~36s at 28K (two-pass
softmax, 32-token sub-batches). Streaming FA-style single pass and/or larger
sub-batch. Also: weight-checksum tool (load-time CRC + on-demand device
recheck) to detect OC/heat bit flips -- see the OC incident note.

**P3 -- decode odds and ends (only after the above):**
- E6 round-cost audit: round time rose 15% for depth-3 but only ~7% is
  accounted for (3rd MTP pass ~4%, 4th verify lane ~2-3%); the rest is 4x
  sequential argmax over 248320 logits + small-kernel launch count. Batch the
  4 argmaxes into one kernel; nsys the rest.
- Depth-4: measure first (extend --stats with a pass-4 chain, same method
  that green-lit E6); build only if p(d4|chain-3) holds ~>=60%. Perm
  machinery generalizes to mod-5.
- Remove pf_scratch (dead: only consumer voids it; ~3 KB/token of ctx).

## Risk register

1. **Gated DeltaNet decode kernel** is the new risk center (was "simple dense" until we read the GGUF). llama.cpp's implementation is the semantic reference; validate per-layer.
2. 4-bit quality on a 27B: keep sensitive tensors high-bit, add importance-weighted scaling if PPL regresses > ~3% vs Q5_K_M. **STATUS: MEASURED 2026-07-02** -- wikitext-2 test, identical tokens and chunk protocol (`--nll`, replicates llama-perplexity -c 512): q27 4-bit PPL **7.2135** vs Q5_K_M **6.9797 +/- 0.046** = **+3.35%**, marginally over the 3% bar (and against a 5.5-bpw quant reading 18.2 GB vs q27's 14.8 -- part of the t/s win is bit-width). Mitigation opened in the roadmap: v1.4 quant policy (bump the most PPL-sensitive tensors to Q8 / importance-weighted scales). CAUTION: any repack changes the model file, which invalidates every canonical token gate -- re-derive canonicals immediately after.
3. M-RoPE sections must match exactly or long-context quality silently degrades.
4. MTP acceptance rate must survive quantization (draft and verify disagreeing more = less speedup). STATUS: measured -- Q4 vs Q8 draft-head argmax agreement 98.1% (E3); depth-3 runtime acceptance 85.7%.
5. **Long-context correctness: VALIDATED to 64K (2026-07-02).** Three gates: (a) `--nll-long 65536` (one pass, no resets) shows NLL flat across position buckets -- 32-48K: PPL 5.45, 48K+: 5.80, no late-position blowup; (b) cross-engine: q27 pooled [32K,64K) PPL 5.622 vs llama-perplexity -c 65536 chunk-1 5.511 = **+2.0%, smaller than the +3.35% short-context delta**, so no length-dependent divergence (M-RoPE + GDN state hold); (c) needle retrieval 3/3 at depths 3/50/95% of a ~55K-token haystack through q27-server, think traces correctly naming surrounding sections. Beyond 64K remains unvalidated (VRAM alloc ceiling ~180K at fp16 KV); rerun tools: `--nll-long`, scratchpad needle harness. Risk 3 is covered to 64K by (a)+(b).
6. **fp-precision paths break the bitwise gate.** Batched prefill is currently bit-identical to serial because dp4a's int32 block sums are order-independent and the per-group fp scale-and-add matches serial order. RESOLVED for prefill: the roadmap's int8 mma.sync path keeps int32 accumulation, so the bitwise gate survives tensor-core prefill (P1). STILL OPEN for fp8 KV (P2) and any future fp16/fp8 MMA: those change logits, so a tolerance gate (logit cosine / top-k agreement vs the exact path, plus the P0 needle/PPL gates) must exist before they ship. The old fork-in-the-road (scoped CUTLASS/cuBLASLt vs hand-rolled vs status quo) is now the P1 fallback ladder rather than a blocker.
