# q27

A narrow inference engine for **Qwopus3.6-27B-v2-MTP** (Qwen3.6-27B hybrid + trained-in MTP heads) on a single RTX 5090. One model, one GPU, as fast as possible. In the spirit of [antirez/ds4](https://github.com/antirez/ds4).

## State of the engine (2026-07-05)

- Decode at depth (the metric that matters for agentic work): **126.2 t/s at
  61K ctx** (was 78.0 pre-fd2), **156-164 t/s effective across real CRUSH
  trials to 74K ctx** (was 103-113). attn-fd2 register-accumulator
  flash-decode fixed the third SM-starvation/occupancy disease: attention
  was 99% of the depth cost at 5% DRAM BW, now 45%
- Decode short-ctx: **169.4 t/s** short-bench suite mean (5 fixed prompts,
  `tools/shortbench_suite.sh`; per-prompt spread 157-191 on trajectory
  alone) / **209.2** stock 2K soak (4.32 t/round). The old single-prompt
  short-bench number is retired as a benchmark -- it re-rolled 177.5 ->
  160.2 on one argmax tie while per-round cost moved +1.3%; it survives
  only as the bitwise gate (see Decode methodology)
- Prefill: cold 28.5K TTFT **~15.0s** after P1-P6 (was 63.8s at P1 start)
- Context: fp8 KV ceiling **~355K**, correctness validated to **361K** (risk 5)
- Quality: Thunderdome **0.786 vs 0.786** dead even against Q5_K_M (30
  trials/leg, 2026-07-03); same-day spot A/B 2026-07-05 (n=1/task):
  collab q27 0.847 vs llama 0.843, analytics q27 0.825 vs llama 0.478.
  Do NOT read the analytics delta as an engine win: analytics is bimodal on
  BOTH engines (this week's greedy draws -- q27 {0.49, 0.60, then 0.79-0.85
  x6}, llama {0.478-0.483 x3, 0.83 x2}; the 30-trial A/B scored the task a
  variance wash, delta -0.073 AGAINST q27). One draw per leg cannot
  separate the engines; llama sampled its low basin that day, q27 didn't
- Agentic wall time vs llama.cpp (same-day A/B 2026-07-05): collab q27 230s
  vs llama 120s -- q27 **1.92x slower** at equal score (0.847 vs 0.843);
  analytics q27 180s vs llama 190s, but the llama leg sat in its low-score
  basin, so that wall win is basin-confounded (a high-basin llama plausibly
  takes it). The engine-true claims are narrower: decode RATE at depth now
  beats llama's late-leg samples (161-164 vs 109-154 t/s), and the collab
  wall gap is OUTPUT VOLUME, not rate (q27's basin wrote 22K tokens vs
  llama's ~11K) -- a prompt/sampling lever, not an engine one. The llama
  leg was NOT handicapped: mainline b9857 with hybrid context checkpoints
  active -- its A/B server log shows LCP prefix reuse with f_keep ~0.99 at
  62-65K ctx (per-turn prompt evals of only ~0.2-1.3K tokens) and draft-mtp
  mean chain 4.8-7.0
- Serving: multi-slot (`--slots N`) with R1b round-granularity GPU
  time-slicing (FIFO gate + engine yield hooks; queue-wait class dead,
  outputs byte-identical solo vs interleaved); server defaults fp8 KV
  (opt out `--kv-fp16`). `--constrain-tools` exists but is OFF in eval
  serving: the capped grammar has a measured engage-lag hole (first
  post-engage token samples unmasked, so a hallucinated tool name
  greedy-loops to score 0 -- build log 2026-07-04) and in-grammar
  acceptance is capped 1/round (~22 t/s inside tool-call bodies). The
  0.786 tie was earned by the tolerant PARSER chain (17 recoveries in the
  final rerun), not the grammar; a strict-parser rerun with the grammar on
  (zero rescues, both legs) is an open gate, blocked on the engage-lag fix.
  Constraint is wired on the Anthropic `/v1/messages` path only
- P10-A status: A0 PASSED, A1 SHIPPED (R1 + R1b, 2026-07-04). Decode-at-depth
  attributed and fixed (fd2, 2026-07-05). Next: sampling
  (docs/sampling-design.md); A2 fusion / light utility slots only if
  telemetry shows engine-claim waits dominating

## Why this model is a good target

- Dense-ish 27B that fits entirely in 32 GB VRAM at 4-bit -- no expert offload, no DRAM scatter, none of the DSV4 pain
- MTP draft head trained into the checkpoint: self-speculation without a separate draft model
- Hybrid Gated-DeltaNet architecture means near-O(1) memory per token for 48 of 65 layers. KV lives only in the 17 full-attention layers (16 + MTP, all **global**, no windowing): 68 KB/token at fp16 = ~4.3 GB @64K, ~8.5 GB @128K, ~17.8 GB @256K. A dense-attention 65-layer build would be ~68 GB @256K. At fp16 KV the practical allocation ceiling is ~180K; **fp8 E4M3 KV (P2, opt-in via `Q27_KV=fp8`) halves that to 34 KB/token and raises the ceiling to ~355K (was ~370K before P3's 5th GDN buffer set) -- the advertised 262K native fits** (allocates and runs; correctness validated to 361K, see risk 5)
- The catch the per-token-memory napkin misses: attention KV is RESTORABLE state (any prefix row range replays for free) while GDN recurrent state is all-or-nothing per sequence -- you can only resume from a position you snapshotted. Hybrids make per-user context cheap but make context REUSE an engineering problem (prefix cache, mid-history divergence, multi-doc serving). That trade is where P8/checkpoint work lives; the measured cost of ignoring it was 7.9x wall-clock on agentic traffic (see build log P8/P9)
- Measured baseline to beat: llama.cpp mainline (b9857, `--spec-type
  draft-mtp`, Q5_K_M, greedy) at 106-127 t/s single-stream on this box;
  its late-leg A/B samples at 62-65K ctx reach 109-162 t/s. Community
  configs report higher (~140 t/s class on lighter quants, e.g. unsloth's
  UD-Q2_K_XL figure; a "mean 140.7 at Q6, patched" config is reported but
  not reproduced here) -- a strongest-opponent sweep (draft depth, p_min)
  on this box is an open item before any headline cross-engine claim

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

These numbers are NOT interchangeable -- each answers a different question:

- **Short-bench suite** (SOTA-comparable): 5 fixed genre-diverse short
  prompts x 128 tokens, `--spec`, STOCK clocks -- `tools/shortbench_suite.sh`.
  **fd2 era: 169.4 t/s mean** (157.2-190.8 per prompt, t/round 3.20-3.88).
  The per-prompt spread is trajectory/acceptance variance, which is exactly
  why no single short prompt may carry a cross-engine number.
- **Canonical prompt** (bitwise gate, NOT a benchmark): 128 tokens from the
  5-token canonical prompt. fd2 era 160.2 t/s / 3.25 t/round; `Q27_FD=v1`
  reproduces the pre-fd2 177.5/3.56 bit-for-bit. That 10% swing is one
  argmax tie re-rolling on a degenerate prompt (per-ROUND cost moved +1.3%)
  -- tie-lottery sensitivity is why it gates bitwise identity and nothing
  else. Depth-4 pays on long generations, not here.
- **2K soak** (long-generation number): 2000-token generation, **209.2 t/s
  STOCK fd2-era** (4.32 t/round; pre-fd2 213.2/4.36, the ~2% is the
  short-ctx split tax). Headline for agentic reply-length outputs.
- **Depth numbers** (fd2, 2026-07-05): **126.2 t/s @61K** single-request
  ground truth; **156-164 t/s effective** across real CRUSH trials to 74K.
  These, not the 2K numbers, predict agentic wall time.

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
| fd2: register-accumulator flash-decode (SM-starvation/occupancy fix #3, attn was 99% of depth cost at 5% DRAM BW) | 61K depth **78.0 -> 126.2 t/s** (+62%, 47.2 -> 29.2 ms/round); 16K **-18%/round**; instance 0.768 -> 0.156 ms @61K (45% DRAM BW); 2K +1.3%/round; acceptance parity exact; PPL in noise both KV modes; nll-long 160K bucket-identical; CANONICAL RE-DERIVED 4c4120c7 (old 58b6ae85 under Q27_FD=v1) |

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
roadmap as the detection mechanism [shipped in P1.5: `--verify-weights` /
`/health?verify=1`]. Offset is volatile -- reapply after reboot.

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

## Roadmap

**Active: P10-A -- fused 2-slot batch-10 serving.** Decided 2026-07-03; the
A/B/C analysis (fused vs vLLM-routing vs interleaved-only, all measured
comparisons) is docs/P10-decision.md. Staged, with a measured gate at each
step:
- A0 go/no-go microbench: does one weight stream feed 10 verify lanes at
  ~2x aggregate? Kill the plan if the 5 -> 10 lane scaling doesn't hold
  (the 4 -> 5 step cost +14% round tax, P3).
- A1 per-slot state + interleaved scheduling -- SHIPPED (R1 e8f71fd/c618c91
  + R1b 3568823/c615d8f/0449131): whole-generation queue waits are gone;
  the remaining head-of-line is engine-claim when conversations outnumber
  --slots (light utility slots are the lever if telemetry demands it).
- A2 fused batch-10 verify (graph indirection + 10-lane GEMVs + per-lane KV).
Prerequisite already shipped: server defaults to fp8 KV (2026-07-03).

**Next up: sampling** -- temperature/top-p with rejection-sampled spec
acceptance; the greedy path stays bitwise untouched. Design:
docs/sampling-design.md. Exit criterion (in the design doc): the quality
A/B and drift catalog re-run under the production sampling config --
every quality number in this README is greedy-no-think scoped.

**Open quality gates (red-team pass 2026-07-05):**
- strict-parser A/B rerun, both legs, tolerant-parser fallbacks disabled,
  zero rescues required -- the proof that the 0.786 tie is engine-true
  rather than harness-carried. Blocked on the engage-lag fix
- constraint-cost soak: one agentic soak with `--constrain-tools` on vs
  off (in-grammar acceptance cap 1/round is ~22 t/s inside call bodies;
  measure what that does to depth-heavy wall time before it defaults on)
- constrain-tools x serving-state gate before the flag ever defaults on
  under `--slots`: assert the global-mask-cache / per-slot host2dev /
  per-engine pool mapping stays coherent (the R1-deferred split-brain);
  define pool-full behavior (a full 512-mask pool today silently
  disengages constraint on that slot only); clear the device constraint
  at request claim (a non-CUDA throw before `tc.end()` leaks a stale
  lane-0 mask + accept-cap-1 into the next request on that slot); keep
  `Q27_TOOL_SPLIT` off under `--slots` (P11 race). Checkpoint-restore x
  grammar needs NO gate: audited 2026-07-05, restore touches only GDN
  state/conv rings/positions and grammar is per-request, engaging only on
  decoded output. Assistant-prefill continuations that end mid-tool-call
  decode unconstrained by design (parser recovery is the net)

**Measured and parked:**
- depth-5: nets ~+2-4% @2K for ~+12-14% round cost (measurement in the build
  log below); precondition for ANY depth change = think-heavy/high-entropy
  acceptance measurement
- chunked-WY delta scan
- cross-session checkpoint pool (P9 covers same-session)
- importance-weighted scales, AWQ-style (only path left on the +3.05% PPL
  gap -- see risk 2; Thunderdome says the gap doesn't bite on agentic coding)

**Open verification:** the P5-era "128K prefill ~57s" does not reconcile with
P6's direct fp16-KV measurement (117.6s); the ~57s was likely extrapolated
from fp8-KV runs. Re-measure 128K under fp8 before using it in any
cross-engine comparison.

## Build log

The full chronological build log (P0..P9, the quality A/B, every DONE
block with its numbers and negative results) lives in
[docs/BUILDLOG.md](docs/BUILDLOG.md).

## Risk register

1. **Gated DeltaNet decode kernel** is the new risk center (was "simple dense" until we read the GGUF). llama.cpp's implementation is the semantic reference; validate per-layer.
2. 4-bit quality on a 27B: keep sensitive tensors high-bit, add importance-weighted scaling if PPL regresses > ~3% vs Q5_K_M. **STATUS: MEASURED + MITIGATED 2026-07-02** -- v1.3 measured +3.35% vs Q5_K_M (7.2135 vs 6.9797, identical tokens/protocol via `--nll`); v1.4 policy (residual writers to Q8, chosen by a 6-candidate sensitivity study -- see build log P0.5) lands at **7.1928 = +3.05%**, and decode got faster (+3.3%, acceptance-coupled). Still marginally over the 3% bar; the study shows uniform promotion cannot close the rest at acceptable cost (ffn_down ceiling probe: 29% of gap for +2.76 GB) -- importance-weighted scales are the documented path if ever needed. The t/s comparison remains bit-width-assisted (15.8 GB reads vs 18.2).
3. M-RoPE sections must match exactly or long-context quality silently degrades.
4. MTP acceptance rate must survive quantization (draft and verify disagreeing more = less speedup). STATUS: measured -- Q4 vs Q8 draft-head argmax agreement 98.1% (E3); depth-3 runtime acceptance 85.7%.
5. **Long-context correctness: VALIDATED to 361K (2026-07-02, fp8 KV).** Original 64K validation: (a) `--nll-long 65536` flat buckets; (b) cross-engine vs llama-perplexity +2.0% at [32K,64K) (smaller than the short-context delta, so no length-dependent divergence); (c) needle 3/3 @55K. Extended after P2 with a 783K-token corpus (War and Peace, tokenized with the model's own vocab; `--nll-long` buckets now reach 320K+): (d) fp16-vs-fp8 NLL A/B at 163840, bucket deltas within +-0.06% at ALL depths (fp8 cost does not grow with position); (e) fp8 `--nll-long 370000` single pass: buckets flat 7.2-7.6 to 256K, then a graceful +3% drift beyond the native 262K (7.89 at 256-320K, 7.69 at 320K+) -- no blowup even in RoPE-extrapolation territory; (f) needle retrieval **6/6 on a 361.5K-token haystack** (depths 35K/124K/213K/248K within native + 276K/337K BEYOND native, all exact, think traces naming surrounding chapters); (g) `--kvstats 131072`: K amax 21.7 / V amax 128.4, zero E4M3 saturation at depth. Decode at 361K depth: 19.1 t/s (fp8, spec). Caveat: each distinct long prompt is a full cold prefill (~22 min @361K) -- the GDN recurrent snapshot makes the prefix cache all-or-nothing, so mid-document divergence cannot reuse state [mitigated same-session by P9's checkpoint ring -- restore from nearest checkpoint <= divergence; cross-session pool still parked]. Risk 3 is covered to 64K by (a)+(b).
6. **fp-precision paths break the bitwise gate.** Batched prefill is currently bit-identical to serial because dp4a's int32 block sums are order-independent and the per-group fp scale-and-add matches serial order. RESOLVED for prefill: the int8 mma.sync path keeps int32 accumulation, so the bitwise gate survives tensor-core prefill (P1). **RESOLVED for fp8 KV (P2, 2026-07-02):** the tolerance-gate machinery now exists and passed -- logit A/B vs the fp16 path (cosine 0.9995, top-1 exact, KL 3.4e-5 @512-tok prompt), corpus PPL delta -0.05%, needle 3/3 -- and fp8 ships opt-in (`Q27_KV=fp8`) with the fp16 default still bitwise-canonical. The same gate recipe applies to any future fp16/fp8 MMA decode path. **AMENDED for the g64 activation regroup (2026-07-04, policy sign-off):** batched-prefill activations now default to per-64 quantization (`Q27_PF_XG`, matching the Q4 weight group so two K=32 mmas chain in int32 before one fp dequant step). Per-64 amax changes the int8 values vs the decode path's per-32, so serial-vs-batched identity no longer holds BY DESIGN on the default path. Replacement gates: test_kernels g64-vs-exact (same quantized inputs through the dp4a exact path, rounding-noise bound), corpus PPL delta, canonical md5 (the canonical CLI run prefills serially and stays bitwise), thunderdome spot-check. `Q27_PF_XG=32` restores the exact path and the `--pf` identity gate enforces it there.
