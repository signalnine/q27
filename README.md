# q27

A narrow inference engine for **Qwopus3.6-27B-v2-MTP** (Qwen3.6-27B hybrid + trained-in MTP heads) on a single RTX 5090. One model, one GPU, as fast as possible. In the spirit of [antirez/ds4](https://github.com/antirez/ds4).

## State of the engine (2026-07-03)

- Decode: **177.5 t/s** stock short-bench / **213.2** stock 2K soak / **218.6**
  at +3000 OC soak (labels defined in "Decode methodology" -- not interchangeable)
- Prefill: cold 28.5K TTFT **~15.0s** after P1-P6 (was 63.8s at P1 start)
- Context: fp8 KV ceiling **~355K**, correctness validated to **361K** (risk 5)
- Quality: Thunderdome **0.786 vs 0.786** dead even against Q5_K_M (30 trials/leg);
  the +3.05% PPL gap does not appear in agentic coding
- Agentic wall time: **~3-4x** llama.cpp, down from 7.9x -- five-mode tool-drift
  catalog closed by P7 constrained decoding + tolerant parser recovery; warm
  turns via P8 stable-prefix + P9 same-session checkpoint caches
- Server defaults: fp8 KV (opt out `--kv-fp16`); `--constrain-tools` available
- Active: P10-A fused 2-slot serving, decided 2026-07-03
  (docs/P10-decision.md), staged behind an A0 go/no-go microbench

## Why this model is a good target

- Dense-ish 27B that fits entirely in 32 GB VRAM at 4-bit -- no expert offload, no DRAM scatter, none of the DSV4 pain
- MTP draft head trained into the checkpoint: self-speculation without a separate draft model
- Hybrid Gated-DeltaNet architecture means near-O(1) memory per token for 48 of 65 layers. KV lives only in the 17 full-attention layers (16 + MTP, all **global**, no windowing): 68 KB/token at fp16 = ~4.3 GB @64K, ~8.5 GB @128K, ~17.8 GB @256K. A dense-attention 65-layer build would be ~68 GB @256K. At fp16 KV the practical allocation ceiling is ~180K; **fp8 E4M3 KV (P2, opt-in via `Q27_KV=fp8`) halves that to 34 KB/token and raises the ceiling to ~355K (was ~370K before P3's 5th GDN buffer set) -- the advertised 262K native fits** (allocates and runs; correctness validated to 361K, see risk 5)
- The catch the per-token-memory napkin misses: attention KV is RESTORABLE state (any prefix row range replays for free) while GDN recurrent state is all-or-nothing per sequence -- you can only resume from a position you snapshotted. Hybrids make per-user context cheap but make context REUSE an engineering problem (prefix cache, mid-history divergence, multi-doc serving). That trade is where P8/checkpoint work lives; the measured cost of ignoring it was 7.9x wall-clock on agentic traffic (see roadmap)
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

Per-layer forward-pass semantics (extracted from llama.cpp source, not from
summaries): docs/SPEC.md.

## Performance model

5090 GDDR7 ~1.79 TB/s. Single-stream decode is weight-read-bound.

| Stage | Per-step read | Ceiling | With MTP (~1.9x measured) |
|---|---|---|---|
| llama.cpp Q5_K_M (62% BW eff) | 18.2 GB | 61.6 t/s measured | 106-127 t/s measured |
| q27 Q5-class, 85-90% eff | ~18 GB | ~88 t/s | ~165 |
| q27 custom 4-bit at 85-90% eff | ~14.8 GB | ~103-109 t/s | ~200-225 |
| q27 4-bit **measured** (2026-07-02, +4000 OC) | ~15.5 GB/step | **91.0 t/s plain** (~75% eff) | **188.9** (depth-3 spec, 2.07x) [superseded -- see Decode methodology] |

The original "~120 t/s ceiling" row implied ~99% BW efficiency and is retired.
Plain decode sits ~15% under the honest 85-90% ceiling; that tail is GDN
recurrence + ~140 small-kernel launches/token, and three attempts on it
(E4 launch geometry, E5 fusions, cp.async) all came back negative.

### Why self-speculation is the whole game at batch 1

Arithmetic-intensity framing (the same napkin datacenters use for the
opposite conclusion): a 5090 offers on the order of hundreds of int8 ops per
byte of DRAM bandwidth, and batch-1 decode with a KV/state cache uses ~2 ops
per byte -- >99% of the compute sits idle while weights stream. Datacenter
serving closes that gap by batching hundreds of USERS per weight read, which
is why API tokens are cheap and why a single-user GPU looks "wasted" in
cost-per-token terms. MTP self-speculation is the batch-1 counter-move: the
batch-5 verify amortizes one weight read across 5 candidate positions --
batching with yourself instead of with other users. That is exactly how
218.6 t/s clears the ~91 t/s plain-decode bandwidth ceiling (2.4x, at 4.36
accepted tokens/round): the idle ops-per-byte gets converted into
single-stream latency instead of multi-tenant throughput. Corollary: every
future decode win here is either (a) fewer bytes per step (quant policy,
fp8 KV) or (b) more accepted positions per weight read (deeper/gated
speculation) -- there is no third lever at batch 1.

### Decode methodology (canonical, 2026-07-02)

Two numbers are reported and they are NOT interchangeable (~16% gap):

- **Short bench** (SOTA-comparable): 128 tokens from the 5-token canonical
  prompt, `--spec`. **STOCK clocks: 177.5 t/s** (depth-4, v1.4, 3-run spread
  0.4); +3000 OC: 181.5. The community "160 on a 5090" numbers are stock
  short-bench -- the honest comparison is **177.5 vs 160 (+11%)**. Note
  depth-4 is a small LOSS on this bench vs the depth-3-era 183.1 stock
  (acceptance only reaches 3.56 t/round at 128 tokens vs 4.36 on the soak;
  the depth-4 round tax doesn't amortize) -- depth-4 was tuned on and pays on
  long generations.
- **2K soak** (long-generation number): 2000-token generation, **213.2 t/s
  STOCK / 218.6 at +3000 OC** (4.36 t/round both). Headline for agentic
  reply-length outputs.

OC policy: headline + SOTA comparisons are reported STOCK (community numbers
aren't OC'd; sidesteps the non-ECC tail-risk conversation). +3000 stays a
supervised-bench option (+2.3% short-bench measured); the weight-checksum
tool (`--verify-weights`, `/health?verify=1`) exists for OC sessions.

## Design decisions

- **Weights**: custom 4-bit symmetric groupwise (group 64, fp16 scales), packed for coalesced 128B warp loads, dequant fused into GEMV. Embeddings, lm_head, MTP layer, norms at 8-bit/f32. Repacked offline from the BF16 GGUF (container spec: docs/FORMAT.md).
- **KV cache**: fp16 for the 17 attention layers by default (f32 originally). FP8 E4M3 ships opt-in via `Q27_KV=fp8` (P2): scale-free saturating conversion (measured K amax <= 21.8, V amax <= 118.6 vs the 448 E4M3 max -- per-row scales buy nothing for a float format with that much headroom), same element-indexed layout, all store/load sites templated on the element type. Halves KV bytes (34 KB/token) and cuts long-ctx decode bandwidth (+11% decode @28.5K). NOT lossless -- default stays fp16 so decode canonicals hold bitwise; measured cost is noise-level (corpus PPL -0.05%, logit KL 3.4e-5). DeltaNet recurrent state is tiny and stays f32.
- **MTP**: first-class. Draft + verify in one pipeline under a single CUDA graph. No separate draft context, no re-prefill.
- **Stack**: plain CUDA C++. No CUTLASS, no deps beyond CUDA runtime. Offline tools are Python: tools/repack.py (runs once; docs/FORMAT.md) and tools/gguf_to_hf.py (certified GGUF -> HF inversion, 866/866 tensors byte-exact, for cross-engine reference runs).
- **Serving**: OpenAI, Anthropic (Claude Code-grade), and OpenAI Responses (Codex-grade) shapes on one binary. Since 2026-07-03 the SERVER defaults to fp8 KV (--kv-fp16 or Q27_KV=fp16 opts out); the CLI keeps fp16 so decode canonicals stay bitwise.

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
  (204.8 long-gen) [superseded -- P3 depth-4; see Decode methodology];
  8000-token output bit-identical to depth-2. Also fixed
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
single-stream; multi-slot is the active P10-A item, docs/P10-decision.md).
Greedy sampling only; sampled decode is designed but not built
(docs/sampling-design.md). `--fast-head` trades output exactness for
~7% more t/s.

## Progress log (tg t/s, greedy, token-identical output verified each step)

Chronological -- each row supersedes the previous. Current canonical numbers
live in "Decode methodology" above.

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
| P5: GEMM tile tuning (grid swap + reg pipeline + vector unpack + NT=64) | Q4 GEMM **-36%** / Q8 **-48%** @26.6K; prefill **1388 -> 1790 t/s** @600; cold 28.5K TTFT **21.4 -> 16.8s** [superseded -- P6: 15.0s]; 128K prefill ~78 -> ~57s [does not reconcile with P6's fp16-KV 117.6s -- see roadmap open verification]; arithmetic bitwise-unchanged (canonical + pf IDENTICAL) |
| P6: column-split delta scan (SM-starvation fix #2) | kernel **748 -> 413 us** @T=256 (1.81x, 48 -> 384 blocks); 26K prefill wall **15.0 -> 13.5s** (-10.3%); 28.5K **16.7 -> 15.0s**; 128K **125.5 -> 117.6s** (fp16-KV kvstats method); split-vs-exact 5e-8, PPL 7.1931 (+0.0003 = fp reorder), canonical md5 exact, pf IDENTICAL |

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
turn 2 on a 26.7k-token context: **1.3s** (26,670/26,693 tokens reused)
[superseded -- see P8: this gate replayed raw tokens, a flow no real client
takes; re-rendering clients missed the cache 100% of the time until the P8
stable-prefix snapshot]. Unconditionally correct: any mismatch falls back to
full prefill; warm-vs-cold continuations gated identical.

Real-world (Claude Code `claude -p`, 26.7k-token system prompt):
| | TTFT |
|---|---|
| pre-M6 (serial prefill) | 15-min timeout, 0 tokens |
| M6 (batched) | 139s |
| + coalesced attention prefill | 90s |
| + GEMM tuning + FA-lite attention | 61s |
| turn 2+ with prefix cache | **1.3s** |

[historical -- cold 28.5K TTFT is ~15.0s after P1-P6; the warm-turn number
required the P8 stable-prefix snapshot to hold on real re-rendering traffic]

## Roadmap (reordered 2026-07-02 after external review)

Context for the ordering: decode is in good shape (see "Decode methodology"
above for the canonical numbers -- 177.5 stock short-bench / 218.6 OC soak;
the 188.9 in this section's history is the depth-3-era +4000 short bench;
three consecutive micro-opt attempts on the remaining tail came back
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

**P5 -- DONE 2026-07-02: GEMM tile tuning (prefill GEMM -36%/-48%, cold
28.5K TTFT 21.4s -> 16.8s, short-prompt prefill 1388 -> 1790 t/s).**
Ladder, each step measured on an ffn_gate micro (17408x5120, T=256):
1. Grid swap (token blocks on blockIdx.x): the old row-major schedule
   re-read weights from DRAM once per 32-token block (8x traffic at T=256);
   swapped order gets 94% L2 hit rate (ncu). Alone it was NEUTRAL --
   the kernel was latency-bound, not BW-bound (2 blocks/SM, reg-limited,
   31% occupancy).
2. Register-pipelined staging: stage st+1's global loads issue right after
   stage st's smem stores, hiding DRAM latency behind mma work. 258 -> ~248.
3. Vectorized Q4 nibble unpack (byte_perm + vsub4, 2 u32 smem stores
   replacing 8 byte stores): 258 -> 217.
4. NT 32 -> 64 (halves token blocks, L2 traffic, and duplicated staging):
   217 -> 204 us.
Negatives (local optimum, do not retry): __launch_bounds__ 3 blocks/SM
(spills, 254), double-buffered smem at NT=64 (225) and NT=32 (237).
In-engine @26.6K: Q4 GEMM 9.78 -> 6.26s, Q8 1.44 -> 0.75s; whole prefill
GPU 22.2 -> 15.2s vs the pre-P4 morning baseline (1.46x today combined).
128K prefill wall ~78 -> ~57s (prefill-only). All arithmetic unchanged --
unit errors byte-identical, canonical md5 exact, pf/pfcache IDENTICAL,
soak 217.0 (decode untouched). Prefill cost order @26K is now
delta_scan_T 25% / attention 21% / Q4 GEMM 41%.

**P6 -- DONE 2026-07-02: column-split delta scan, kernel 1.81x, 26K prefill
wall 15.0 -> 13.5s.** k_delta_scan_T was the same SM-starvation disease P4
fixed for attention: 48 blocks (one per GDN head) on 170 SMs, 756us per
256-token launch, 25% of prefill @26K. The 128 S-columns per head are
independent (pred_j, dj and o_j read only column j), so
k_delta_scan_split<NCOL> slices them across gridDim.x blocks and deepens
row-parallelism per column (NTILE=512/NCOL row tiles instead of 4; the two
serial 32-iteration row loops shrink to NCOL/4). Same 4-barriers-per-token
structure -- the legacy kernel's 5th trailing barrier is provably covered by
the next token's sq/sk barrier. Q27_DS_SPLIT forces 1/2/4/8; 1 = untouched
legacy kernel; auto default 8 (measured 748 -> 428/456/413us for 2/4/8 --
4 is reproducibly WORST: 192 blocks is the awkward wave count on 170 SMs;
384 balances best). No combine kernel needed (unlike P4): the split is pure
parallelism, only sq/sk staging duplicates (CS x 1KB/token from L2).
Measured: 26K prefill 15.02 -> 13.48s (-10.3%), 28.5K 16.69 -> 14.96s, 128K
125.5 -> 117.6s (all fp16-KV kvstats-method A/B, same binary; --kvstats now
prints prefill wall time). Gates: split-vs-exact 5e-8 at T=1 AND T=64
(tolerance-gated like P4 -- row reductions reorder), full-corpus PPL 7.1931
vs 7.1928 (+0.004%), canonical md5 exact at split 1/4/8, --pf 200
continuations IDENTICAL, --pfdbg maxdiffs same order as the split=1
baseline. Test lesson worth keeping: the first unit-test run FAILED at 8e-2
with raw-normal conv data -- the delta update S += beta*k(v - k'S) is
chaotic when ||k||~11, amplifying legitimate 1e-7 reorder noise; the engine
l2-normalizes q/k per head before the scan (l2norm_heads_T), and with
in-contract data the split matches to 5e-8. Test data must honor kernel
input contracts before a tolerance FAIL means anything. Prefill cost order
@26K is now Q4 GEMM ~46% / attention ~23% / delta_scan ~15%.

**Next (post-P6, reordered 2026-07-02 after external review round 2):**
1. **Task-level quality A/B -- DONE 2026-07-03.** q27 v1.4 4-bit vs Q5_K_M
   (llama.cpp + MTP), Thunderdome standard suite T1-T10, CRUSH harness,
   no-think + greedy both legs, n=3 per task:
   **overall 0.786 vs 0.786 -- DEAD EVEN (30 trials/leg).**
   Per-task deltas: collab-server +0.103 (q27), fts +0.023, task-queue
   +0.022, plugin/ecommerce/monorepo/ssg within +-0.002, time-tracker -0.016,
   phantom-invoice -0.063, analytics -0.073 (bimodal 0.48/0.83 on BOTH legs
   -- task variance, not quant). Greedy determinism made n=3 near-zero
   variance on most tasks. **The +3.05% PPL does not appear in agentic
   coding.** What DID appear: five tool-format drift modes under no-think
   greedy (dropped <tool_call> wrapper; unterminated JSON w/ </file> junk;
   <content>-tagged raw values; {"tool_call": JSON-keyed opener; raw control
   chars inside JSON strings) -- structurally masked on the llama leg by
   grammar-constrained decoding, initially FATAL on q27 (task-queue 0.000,
   plugin 0.185 with zero writes executed), now fully recovered by the
   tolerant parser chain in api_common.h (17 recoveries in the final rerun,
   scores 0.782/0.899). Verdict: the quant is clean; tool-call discipline is
   a SERVING-LAYER property. **P7 constrained decoding SHIPPED 2026-07-03**
   (`--constrain-tools`): ToolGrammar + lazy mask cache + slot-0 masked
   verify + in-grammar acceptance cap + pending-token mask staging. E2E
   clean on time-tracker 0.84 / task-queue / collab 0.836 / plugin 0.903 --
   zero disengages, zero fallbacks needed for wrapped calls. (An early
   deterministic "0x65 rejected" disengage was root-caused to one-token-
   lagged masks before on_pending staging existed -- stale masks FORCE
   illegal tokens; gone since.) In-call throughput ~22 t/s (acceptance
   capped at 1/round in-grammar; drafts are generated inside the round
   graph and cannot be host-constrained -- split draft/verify graphs is the
   known optimization if tool-span speed ever matters).
2. **P8 -- DONE 2026-07-03: stable-prefix snapshot.** Root cause of the 7.9x
   eval wall-time: the snapshot included the volatile prompt tail
   (assistant-open + no-think prefill), which every re-rendering client
   replaces next turn -- divergence ~6 tokens before snapshot end, voided by
   the all-or-nothing check -> full re-prefill EVERY turn. The old --pfcache
   gate appended raw tokens (a flow no real client takes) and hid it since
   M6.5. Fix: chatml_prompt reports the boundary (end of last input message,
   always abutting <|im_start|> so split-encoding is tokenization-invariant),
   generate() prefills in two stages and snapshots at the boundary. Gate v2
   uses a tail-divergent turn 2: warm restore, continuations IDENTICAL.
   Measured: collab-server trial 2434s -> 536s (4.5x), score unchanged
   (0.836), prefix_hit=54-58K logged turn over turn -- the first real-traffic
   cache hits this server has ever had. Wall-time refresh on the full
   P7+P8+P9 stack (2026-07-03, worst two remaining tasks, n=3):
   analytics-dashboard 1954s -> 667s AND score 0.641 -> 0.820 (constraint
   fixed drift that was costing points, now beats the Q5 leg's 0.715);
   ecommerce-backend 1025s -> 199s (score 0.518 unchanged). The 13-19x
   pathological multipliers are now a uniform ~3-4x vs llama.cpp; the
   residual is per-turn suffix prefill + the in-call constraint cap. Remaining for TRUE mid-history edits
   (client compaction, edited files): periodic GDN checkpoints -- snapshot S
   + conv rings every N tokens, restart from nearest checkpoint <= divergence
   (llama.cpp PR #24785 / commit b9180 n_rs_seq is the reference design).
   State is ~28 MB/layer-set snapshot.
3. Sampling (temperature/top-p vs spec-verify acceptance).
4. Depth-5 MEASURED (pass-5 stats rig, 2026-07-03): **p(d5|prefix4) = 96.8%**,
   p(prefix4) = 89.0%, +0.862 t/round ungated per-position. Applying the
   stats-vs-live discount P3 exhibited (+0.890 projected -> +3.9% live),
   depth-5 nets **~+2-4% @2K** against ~+12-14% round cost (5th sequential
   MTP head pass + 6-lane verify) and a 6th GDN buffer set (-610MB fp8 ctx
   ceiling). Real but modest -- build only if decode t/s becomes the
   priority again. Margin-gating buys little on the soak (ungated chains
   already clean); the think-heavy/high-entropy acceptance measurement
   remains the open question before ANY depth change ships.
5. Known stale claim: "128K prefill ~57s" (P5 note) does not reconcile with
   the direct fp16-KV kvstats measurement (117.6s post-P6); the ~57s was a
   P5-era extrapolation on what were likely fp8-KV runs. Re-measure 128K
   under fp8 before using it in any cross-engine comparison.

## Risk register

1. **Gated DeltaNet decode kernel** is the new risk center (was "simple dense" until we read the GGUF). llama.cpp's implementation is the semantic reference; validate per-layer.
2. 4-bit quality on a 27B: keep sensitive tensors high-bit, add importance-weighted scaling if PPL regresses > ~3% vs Q5_K_M. **STATUS: MEASURED + MITIGATED 2026-07-02** -- v1.3 measured +3.35% vs Q5_K_M (7.2135 vs 6.9797, identical tokens/protocol via `--nll`); v1.4 policy (residual writers to Q8, chosen by a 6-candidate sensitivity study -- see roadmap P0.5) lands at **7.1928 = +3.05%**, and decode got faster (+3.3%, acceptance-coupled). Still marginally over the 3% bar; the study shows uniform promotion cannot close the rest at acceptable cost (ffn_down ceiling probe: 29% of gap for +2.76 GB) -- importance-weighted scales are the documented path if ever needed. The t/s comparison remains bit-width-assisted (15.8 GB reads vs 18.2).
3. M-RoPE sections must match exactly or long-context quality silently degrades.
4. MTP acceptance rate must survive quantization (draft and verify disagreeing more = less speedup). STATUS: measured -- Q4 vs Q8 draft-head argmax agreement 98.1% (E3); depth-3 runtime acceptance 85.7%.
5. **Long-context correctness: VALIDATED to 361K (2026-07-02, fp8 KV).** Original 64K validation: (a) `--nll-long 65536` flat buckets; (b) cross-engine vs llama-perplexity +2.0% at [32K,64K) (smaller than the short-context delta, so no length-dependent divergence); (c) needle 3/3 @55K. Extended after P2 with a 783K-token corpus (War and Peace, tokenized with the model's own vocab; `--nll-long` buckets now reach 320K+): (d) fp16-vs-fp8 NLL A/B at 163840, bucket deltas within +-0.06% at ALL depths (fp8 cost does not grow with position); (e) fp8 `--nll-long 370000` single pass: buckets flat 7.2-7.6 to 256K, then a graceful +3% drift beyond the native 262K (7.89 at 256-320K, 7.69 at 320K+) -- no blowup even in RoPE-extrapolation territory; (f) needle retrieval **6/6 on a 361.5K-token haystack** (depths 35K/124K/213K/248K within native + 276K/337K BEYOND native, all exact, think traces naming surrounding chapters); (g) `--kvstats 131072`: K amax 21.7 / V amax 128.4, zero E4M3 saturation at depth. Decode at 361K depth: 19.1 t/s (fp8, spec). Caveat: each distinct long prompt is a full cold prefill (~22 min @361K) -- the GDN recurrent snapshot makes the prefix cache all-or-nothing, so mid-document divergence cannot reuse state. Risk 3 is covered to 64K by (a)+(b).
6. **fp-precision paths break the bitwise gate.** Batched prefill is currently bit-identical to serial because dp4a's int32 block sums are order-independent and the per-group fp scale-and-add matches serial order. RESOLVED for prefill: the roadmap's int8 mma.sync path keeps int32 accumulation, so the bitwise gate survives tensor-core prefill (P1). **RESOLVED for fp8 KV (P2, 2026-07-02):** the tolerance-gate machinery now exists and passed -- logit A/B vs the fp16 path (cosine 0.9995, top-1 exact, KL 3.4e-5 @512-tok prompt), corpus PPL delta -0.05%, needle 3/3 -- and fp8 ships opt-in (`Q27_KV=fp8`) with the fp16 default still bitwise-canonical. The same gate recipe applies to any future fp16/fp8 MMA decode path.
