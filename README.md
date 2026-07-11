# Quasar

A narrow inference engine for **Qwen3.6-27B-MTP** (hybrid GDN+attention, trained-in MTP heads) and its fine-tunes on a single RTX 5090. One model family, one GPU, as fast as possible. In the spirit of [antirez/ds4](https://github.com/antirez/ds4) -- ds4's DwarfStar is small and dense; Quasar is the same shelf pointed the other way: a compact source with outsized output.

Built under the codename `q27`, which the binaries (`q27`, `q27-server`), the `Q27_*` env vars, and the `q27k` namespace keep -- so every command below is unchanged.

**Baseline model (2026-07-09): vanilla Qwen3.6-27B-MTP** (`qwen36-27b-mtp`,
canonical md5 `a2982c51...`) -- the benchmark standard: bench rigs and gate
scripts default to it. Fine-tunes stay fully supported (`MODEL=`/`TOK=`/
`CANON_MD5=` env overrides; Qwopus3.6-27B-v2-MTP canonical `4c4120c7...`).
Historical numbers in this README were measured on Qwopus unless noted.

**Reference numbers** (2026-07-10, master, vanilla model, 5090; full
tables in BUILDLOG):

- Short-bench suite **172.2 t/s** (fp16 stock CLI, 5-prompt mean; canonical
  `a2982c51` EXACT). History: 161.8 (07-09) -> 149.5 (width-12 param-copy
  regression, caught by this suite) -> 172.2 (`__grid_constant__` fix --
  ABOVE the old baseline; the per-thread param copy predated width-12).
- Decode @26K (server replay, constructed cctx payload, fp8 basin):
  classic config 143.0 / **full default stack 176.3 t/s** (+23%).
- Echo (repetitive traffic, wide suffix): 2K CLI 317.9; **26K zero-config
  server 400.6 t/s** -- the degenerate-echo CEILING, quoted as a bound,
  never a headline.
- Live Claude-Code traffic (vanilla, n=3 x {T2,T5,T8} = 9 trials, 430
  requests): **231.3 t/s aggregate**, per-request median 225 / p75 277 /
  **peak 378 t/s**; suffix drafter AL 9.4 on 37% of decode.
- Prefill (fp8 batched TTFT): 8K 2.35s | 32K 10.4s | 128K 59.4s (~2200 t/s).
- Cross-engine (same model, GPU, harness, day -- 2026-07-10, executing
  the n>=3 protocol AS FILED 2026-07-05, five days before it could be
  passed: 3 trials x {T2,T5,T8} per engine, both legs strongest config,
  no-think greedy CC harness): **q27 +47% decode vs llama.cpp's best
  config** (231.3 vs 157.4 t/s aggregate over 430/197 requests; medians
  225 vs 155, peaks 378 vs 274; llama = draft-mtp10/p-min0.5/fa on
  Q5_K_M; the n=1 pilot read +40% -- the number STRENGTHENED under
  replication). The statistically solid claims: median convergence
  (T2 0.83 == 0.83, T5 0.78 vs 0.79 -- quality is the model's) and the
  within-leg decode gap (430 requests of telemetry). Score parity is a
  SYSTEM-level claim: engine + serving-layer parser, and the tolerant
  tool-call parser is load-bearing on this harness for any engine
  (strict parsing scores 0.000 on T8-class tasks). Reported
  descriptively, NOT as findings (n=9/leg cannot separate them --
  Fisher's exact p~=0.29): raw draws q27 8/9 in-band vs llama 5/9
  (llama's misses: one hard 0.00 at 443s, a 0.45, two bad T8 basins);
  T8 bimodal on BOTH engines, q27 2/3 vs llama 1/3 good -- one draw
  apart on a task documented as basin-lottery for every engine. Wall
  medians favor q27 3-4x on T2/T5 but wall is trajectory-confounded
  (llama generated ~2.3x tokens on its own trajectories), so within-leg
  decode telemetry is the rate currency. Decomposition: q27 reads ~15.8GB
  weights/step at 5.25bpw vs Q5_K_M's ~18.2GB -- ~15% of the gap is
  bit-width on a bandwidth-bound decode, ~+22% is mechanism (suffix +
  ladder + fdmma + prefix cache); the Thunderdome quality tie (0.786 ==
  0.786, 30 trials/leg) already cleared the quant side. llama's one
  winning cell: ngram speculation on a pure token loop (889 t/s --
  degenerate case; q27's suffix cap of 12 does not bind on real traffic).
  Historical note: the 07-09 shootout's llama 65K segv was a 24GB/3090
  ctx-create OOM path, not a model or engine defect -- today's 5090 legs
  ran -c 65536 clean.
- Fine-tune headroom: Qwopus +35% on acceptance alone (07-09 same-binary
  cctx replay, 219.0 vs 162.1 at auto). Its live triplet (pre-
  `__grid_constant__`): 197-222 t/s aggregate, peak 320.

## Quickstart

Requirements: an NVIDIA GPU with 24GB+ VRAM (built for the RTX 5090 /
sm_120; runs on sm_86+ with automatic fallbacks -- the fp8-KV + MMA fast
paths need sm_89+, and 24GB cards serve at reduced context), CUDA
toolkit 12.x at `/usr/local/cuda`, and gcc.

```bash
# 1. model + tokenizer from Hugging Face (~17GB; Apache-2.0)
huggingface-cli download signalnine/Qwen3.6-27B-MTP-q27 \
  --local-dir models/qwen36-27b-mtp
# fine-tune variant: signalnine/Qwopus3.6-27B-v2-MTP-q27
# verify: (cd models/qwen36-27b-mtp && md5sum -c CHECKSUMS.md5)

# 2. build (CLI + server + test suites)
git clone https://github.com/signalnine/q27 && cd q27
make

# 3. smoke test the CLI (should print 128 tokens; md5 of the output
#    line is the bitwise canonical a2982c51...)
./build/q27 ../models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
  --tokens "760,6511,314,9338,369" -n 128 --ctx 2048 --spec

# 4. serve -- zero config; defaults resolve the full measured stack
#    and --ctx auto-sizes to your VRAM (see Serving for escapes)
./build/q27-server ../models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
  ../models/qwen36-27b-mtp/qwen36-27b-mtp.tok --port 8080
```

Sanity-check the server (native Anthropic Messages API; OpenAI
`/v1/chat/completions` and `/v1/completions` also served):

```bash
curl -s localhost:8080/v1/messages -H 'content-type: application/json' \
  -d '{"model":"q27","max_tokens":32,"messages":[{"role":"user","content":"say hi"}]}'
```

Point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8080"
export ANTHROPIC_API_KEY="placeholder"
export ANTHROPIC_DEFAULT_OPUS_MODEL="q27"
export ANTHROPIC_DEFAULT_SONNET_MODEL="q27"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="q27"
claude
```

The server is single-model, so the model name in requests is accepted
as-is. Expect ~170-230 t/s decode on a 5090 depending on traffic shape
(see Reference numbers), warm multi-turn prefills served from the
prefix cache, and `count_tokens` + Anthropic-shaped context-limit
errors so Claude Code compacts correctly.

## State of the engine (2026-07-10)

One day, five shipped stages -- each gated (canonical EXACT / byte-identity
/ sanitizer / test suites) and pushed; details + negatives in BUILDLOG:

- **Width-12 verify architecture.** The lane fabric widened 8 -> 12 (p[16]
  struct family, 12 GDN role sets, perm mod 12, 12-perm graph zoo,
  prep/finish pointer-struct signatures). Policy-decoupled: the MTP ladder
  stays 4..7; widths 9..12 belong to the **suffix drafter** (`Q27_SUFFIX_W`,
  default 12), which fills draft lanes from recurring committed-stream
  suffixes at zero model cost. Live T8: **AL 10.61 on 61.6% of decode**
  (same fire rate as width-8, +63% tokens per fire -- the cap-release
  mechanism the plan predicted), good-basin score 0.84. Byte-identical to
  the pre-widen binary under fd2 by construction. Framing: stream-lookup
  self-speculation is an old family (llama's ngram spec is a sibling, and
  beats us on the degenerate pure loop); what is ours is the COMPOSITION
  -- per-round arbitration between the trained MTP ladder and the free
  suffix drafter, both verified through one shared-KV width-12 MMA
  kernel, with the suffix taking the echo cream so the ladder is not
  starved (auto rounds still promote to their ceiling on live traffic).
- **fdmma verify attention, tuned 1.75x in a day.** The fp8-MMA shared-KV
  verify kernel went 354.7 -> 282.4 (STAGES=1: single-buffered K/V at
  **2 CTAs/SM** -- ncu showed the 1-CTA kernel 89% no-eligible) -> **202.6us
  at 61K W=12 (5.6x over fd2)** via the split-count retune (ns =
  SMs*2/kv_heads, computed per GPU -- 128 splits left a half-empty wave).
  Three negatives attributed and recorded: prefetch reorder (wash),
  warp-pair PV and warp-specialized producer (both bitwise-correct, both
  lose -- for this kernel family CTA count dominates intra-CTA
  orchestration; reopening bar filed).
- **`__grid_constant__` on every struct-param kernel.** The vanilla
  short-bench suite caught an 8% engine-wide regression the depth gates
  missed: by-value lane structs indexed by blockIdx are copied to
  per-thread LOCAL memory (128B since P10-A0; width-12 doubled it).
  The fix is addressing-only (bitwise) and lands the suite at 172.2 --
  above the pre-width-12 baseline. Standing lesson: the short-ctx suite
  is the param/launch-overhead canary; run it after any plumbing change.
- **Zero-config Claude-Code serving.** A bare `q27-server model.q27
  model.tok` now IS the full measured stack (see Serving) with VRAM-sized
  auto `--ctx`; the CLI keeps reference defaults so the bitwise canonical
  world is untouched. The standing eval unit is a two-argument command.
- **Deep-MTP question closed (width-12 P3).** On the freshly measured
  wide-round curve, MTP ceilings 9/10 price NEGATIVE even on the hottest
  chains (acceptance saturates at 4.79 tok/rnd vs 2.7-3.2ms marginal
  lanes); ceiling 8 prices +2.7% MTP-wall on hot chains but the live cap
  mix (16% full-fire at 7, sat7 ~25%) plus suffix shadowing shrink it
  under 1% engine -- NO-BUILD, reopen conditions filed. The wide-lane
  marginal is **GEMV-N-bound**, not GDN-bound: the deferred-snapshot GDN
  chunk stays shelved; the honest lever for cheaper wide rounds is the
  mma16 batched-GEMM verify pivot (on file, not commissioned).

**Cross-engine (2026-07-10, the cleanest read yet):** both engines on
vanilla qwen, same 5090, same Claude Code harness, back to back -- q27
zero-config vs llama.cpp Q5_K_M at its best config (draft-mtp10,
p-min 0.5, fa): scores converge to the model (T2 0.84 == 0.84; T5
0.78/0.81; T8 llama drew the documented engine-independent bad basin),
**decode q27 +40%** (221 vs 157 t/s aggregate; peaks 362 vs 259), wall
1.8-3.5x. Arc for honesty: tuned llama was +31% at depth on 07-06, parity
on 07-07, q27 +40% today; llama's ngram spec still owns the degenerate
pure-loop echo (889 t/s vs 318 -- unbounded draft length vs q27's 12-lane
cap, which real traffic never reaches).

**Carried state (pre-2026-07-10, still load-bearing):**

- Adaptive draft ladder 4..7 (`Q27_MAXD=auto7`, now server default): 3-bar
  controller (`src/depthctl.h`), emitted text byte-identical at every
  ceiling. docs/maxd6-decision.md has the NO-GO -> GO-IF -> build trail.
- Confidence-gated depth + draft early-exit (`Q27_PMIN=0.5` +
  `Q27_DEXIT`, now server defaults): P12/P14; greedy width-invariant, sampled
  gate pays only with draft-side early-exit.
- fp8 KV end-to-end (now server default; needle 6/6 to 361K, fp8 QK^T MMA
  prefill +11.8% @128K, fp8-PV +2.4%); fp16 CLI canonical untouched.
- Sampling at spec speed (Phases 1-2, exit-gate PASSED, cleared to default
  T<=0.7/top-p 0.95; greedy stays bitwise).
- Agentic serving: the tolerant tool-call parser is load-bearing (drift
  modes 1-8 fixed; CC 0.00 -> 0.55 was a PARSER ceiling, not quant);
  `--constrain-tools` P15 engage-lag fixed, still opt-in (in-call cost).
- Multi-slot serving (`--slots N`) with R1b round-granularity GPU
  time-slicing; P9 same-session checkpoint ring; P8 stable-prefix snapshot
  (warm turns ~1.3s); `/v1/messages` native incl `count_tokens`.
- Long-context: validated to 361K needle 6/6 (fp8); ~355K fp8 KV ceiling.

## Why this model is a good target

- Dense-ish 27B that fits entirely in 32 GB VRAM at 4-bit -- no expert offload, no DRAM scatter, none of the DSV4 pain
- MTP draft head trained into the checkpoint: self-speculation without a separate draft model
- Hybrid Gated-DeltaNet architecture means near-O(1) memory per token for 48 of 65 layers. KV lives only in the 17 full-attention layers (16 + MTP, all **global**, no windowing): 68 KB/token at fp16 = ~4.3 GB @64K, ~8.5 GB @128K, ~17.8 GB @256K. A dense-attention 65-layer build would be ~68 GB @256K. At fp16 KV the practical allocation ceiling is ~180K; **fp8 E4M3 KV (P2; server default since 07-03, arch-gated sm_89+ with the `Q27_PROFILE=ref` escape 07-10; CLI opt-in via `Q27_KV=fp8`) halves that to 34 KB/token and raises the ceiling to ~285K est. post-width-12 (~355K before the role sets grew 5 -> 12; +627MB) -- the advertised 262K native still fits** (allocates and runs; correctness validated to 361K, see risk 5)
- The catch the per-token-memory napkin misses: attention KV is RESTORABLE state (any prefix row range replays for free) while GDN recurrent state is all-or-nothing per sequence -- you can only resume from a position you snapshotted. Hybrids make per-user context cheap but make context REUSE an engineering problem (prefix cache, mid-history divergence, multi-doc serving). That trade is where P8/checkpoint work lives; the measured cost of ignoring it was 7.9x wall-clock on agentic traffic (see build log P8/P9)
- Measured baseline to beat: llama.cpp mainline (b9857, `--spec-type
  draft-mtp`, Q5_K_M, greedy) at 106-127 t/s single-stream on this box;
  its late-leg A/B samples at 62-65K ctx reach 109-162 t/s. The strongest
  community reference (r/LocalLLM, 2026-07-03: same GDN arch on a 5090,
  llama.cpp with PR #24785 + b9180 n_rs_seq + the hybrid checkpoint-search
  fixes -- all present in our b9857 build) reports **140.7 t/s mean over
  20h of agentic use at Q6**, with draft=10 + p_min 0.5 beating draft=6 by
  15-20 t/s despite lower acceptance. **Sweep DONE 2026-07-06** (Q5_K_M @~2K,
  single-stream greedy): the +15% win is `p_min 0.5` (+14 t/s), NOT draft depth
  (6->10 buys ~0-1); draft=6/p_min0 (our old A/B config) 102.4 t/s vs
  draft=10/p_min0.5 117.9. So the honest strongest-llama decode baseline is
  **~117 t/s @2K**, and our earlier A/B UNDER-STATED llama by ~15%. Depth-matched
  confirm DONE 07-07 (parity); superseded by the 2026-07-10 same-model A/B
  (q27 +40% decode) -- see Reference numbers.

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
  **Current (2026-07-10, vanilla baseline): 172.2 t/s mean** (per-prompt
  143.9-179.9; vanilla series 161.8 on 07-09 -> 172.2 post
  `__grid_constant__`). Qwopus-era historical mean: 179.7 (07-08). The
  per-prompt spread is trajectory/acceptance variance, which is exactly why
  no single short prompt may carry a cross-engine number. It is also the
  param/launch-overhead CANARY: it caught the width-12 param-copy regression
  the depth gates missed (07-10).
- **Canonical prompt** (bitwise gate, NOT a benchmark): 128 tokens from the
  5-token canonical prompt -- vanilla baseline md5 `a2982c51...` (the
  standard; ~140 t/s on 07-10 trajectory), Qwopus `4c4120c7...` for
  fine-tune gating (~168-170 t/s fd2-era). Held bitwise through fd2,
  P12-P15, fp8q prefill, verify-gemv, the maxd6/7 widenings, the FULL
  width-12 widening (8 -> 12 lanes, perm mod 12), fdmma STAGES=1, and
  `__grid_constant__`; `Q27_FD=v1` reproduces the pre-fd2 text bit-for-bit.
  Tie-lottery sensitivity is why it gates bitwise identity and nothing
  else. It gates the CLI's reference defaults; the server's CC defaults are
  deliberately tolerance-class (fp8+mma) -- `Q27_PROFILE=ref` restores
  reference behavior there.
- **Tie-lottery methodology** (the project's most subtle measurement
  concept): tolerance-class numerics changes (fp8 paths, mma, split-count)
  re-roll greedy argmax ties -- **neutral in expectation, deterministic
  per build**. A quality flip on ONE benchmark basin (the T8 auth-gate)
  is therefore read via a basin MATRIX across tasks plus a re-roll on the
  next binary, not via a single retrial: mma flipped T8 bad on the 07-10
  morning binaries and re-rolled GOOD (0.84-0.85) on the width-12
  binaries with the identical kernel -- per-binary lottery, not
  kernel-class steering. Acceptance-sensitive decisions must name their
  basin; cross-BUILD text comparisons are invalid (same-binary legs
  only).
- **2K soak** (long-generation number): 2000-token generation, **209.2 t/s
  STOCK fd2-era** (4.32 t/round; pre-fd2 213.2/4.36, the ~2% is the
  short-ctx split tax). Headline for agentic reply-length outputs.
- **Depth numbers** (Qwopus 07-08-era history; CURRENT depth predictors are the 07-10 Reference numbers: cctx 26K 143.0/176.3 vanilla, live CC 213-227 t/s aggregate): **172.9 t/s ungated @61K** (verify-gemv, 2026-07-08
  fixtures -- NOT comparable to the P14-era 119-126 on the old corpus; the
  ms/round attribution bridges eras); **202-223 t/s @26K real-CC-transcript
  replay** (d4-gated -> auto-ladder-6); **156-164 t/s effective** across real
  CRUSH trials to 74K (fd2-era). These, not the 2K numbers, predict agentic
  wall time.

OC policy: headline + SOTA comparisons are reported STOCK (community numbers
aren't OC'd; sidesteps the non-ECC tail-risk conversation). +3000 stays a
supervised-bench option (+2.3% short-bench measured); the weight-checksum
tool (`--verify-weights`, `/health?verify=1`) exists for OC sessions.

## Design decisions

- **Weights**: custom 4-bit symmetric groupwise (group 64, fp16 scales), packed for coalesced 128B warp loads, dequant fused into GEMV. Embeddings, lm_head, MTP layer, norms at 8-bit/f32. Repacked offline from the BF16 GGUF (container spec: docs/FORMAT.md).
- **KV cache**: fp16 for the 17 attention layers by default (f32 originally). FP8 E4M3 ships as the SERVER default (since 07-03; arch-gated + profile-escaped 07-10; CLI opt-in via `Q27_KV=fp8`, P2): scale-free saturating conversion (measured K amax <= 21.8, V amax <= 118.6 vs the 448 E4M3 max -- per-row scales buy nothing for a float format with that much headroom), same element-indexed layout, all store/load sites templated on the element type. Halves KV bytes (34 KB/token) and cuts long-ctx decode bandwidth (+11% decode @28.5K). NOT lossless -- the CLI default stays fp16 so decode canonicals hold bitwise; measured cost is noise-level (corpus PPL -0.05%, logit KL 3.4e-5). DeltaNet recurrent state is tiny and stays f32. **turbo3 3-bit KV (phase 1, 2026-07-11; CLI opt-in `Q27_KV=turbo3`, or `turbo3v` for fp16-K + turbo3-V) stores K/V as WHT-rotated 50-byte blocks per 128 dims (TurboQuant port, src/turbo3.cuh): ~13.4 KB/token, 2.56x under fp8. Decode, verify, and batched prefill all have turbo3 legs (prefill dequants blocks in the f16-MMA smem staging; lite path is the oracle). Quality (wikitext PPL, qwopus): fp16 7.317 / fp8 7.327 / turbo3 7.381 (+0.87%; K adds only +0.17% -- the GQA=6 K-crater the fork saw did not reproduce).**
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
./build/q27-server model.q27 model.tok --port 8080
```

**Defaults (2026-07-10) = the measured Claude-Code stack.** A bare server
serves the exact config every live trial and record number was earned on:
fp8 KV + `Q27_FD=mma` (sm_89+; older parts fall back fp16 + fd2),
`Q27_PMIN=0.5`, `Q27_MAXD=auto7`, suffix drafter at width 12, fast-head,
no-think, phase stats; `--ctx` auto-sizes the KV budget to free VRAM
(cap 131072, single-slot). Every knob keeps its env/flag override
(user env always wins), `Q27_PROFILE=ref` restores the conservative
reference behavior (fp16, ungated, no suffix, fd2), and the **CLI binary
keeps reference defaults** so the bitwise canonical gates are untouched.
Escapes: `--kv-fp16 --no-fast-head --think`, any individual `Q27_*`.

Three API shapes on one server:
- **OpenAI**: `/v1/chat/completions`, `/v1/completions` (text)
- **Anthropic**: `/v1/messages` -- native Messages API with thinking blocks
  (Qwopus `<think>` mapped to thinking/signature blocks), tool_use/tool_result,
  input_json_delta streaming. Also `/v1/messages/count_tokens` (exact,
  == usage.input_tokens) and an anthropic-shaped context-limit error
  (`prompt is too long: N tokens > M maximum`, 400) so Claude Code compacts
  instead of retry-looping; plus cch billing-header normalization that keeps
  the prefix cache warm across CC turns. Claude Code-compatible:
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
(src/stream_split.h) that also routes `<think>`. Multi-slot (`--slots N`) with
R1b round-granularity GPU time-slicing (FIFO gate + yield hooks; byte-identical
solo vs interleaved). Greedy (spec decode) by default; `temperature>0` routes to
sampled SPEC decode -- top-p nucleus + Gumbel-max with rejection-sampled spec
acceptance (Phase 2, at spec speed), seeded and reproducible, greedy left
bitwise-unchanged (docs/sampling-design.md, Phases 1-2). The exit-gate A/B passed
(docs/sampling-exit-gate.md), so the server can default sampling on for clients
that send no temperature via `Q27_FORCE_TEMP`/`Q27_FORCE_TOP_P` (an explicit
request temperature still wins; a forced request gets a distinct logged seed).
The tolerant tool-call parser recovers six observed drift modes -- dropped
`<tool_call>` wrapper, truncated JSON, `<content>`-tagged and
quote-open/`</content>` bodies, `{"tool_call":` openers, in-string control chars,
and name-dropped `{"name":\n{args}}` calls (tool inferred from the arg-key
signature) -- logging each recovery for the drift catalog. `--fast-head` trades
output exactness for ~7% more t/s.

**Confidence-gated depth (P12 + P14) applies to BOTH greedy and sampled
decode.** `Q27_PMIN=theta` caps the verify width on the drafter's top1-top2
margin (skipping the deep-KV verify when the draft head is unconfident), and
`Q27_DEXIT` (P14, default-ON whenever `Q27_PMIN` is set) additionally stops
DRAFTING at the first sub-theta margin -- the llama p_min draft-stop that the
verify-only gate lacked. **These are the server DEFAULTS since 2026-07-10** (`Q27_PMIN=0.5
Q27_MAXD=auto7`; `Q27_PROFILE=ref` opts back out) -- the adaptive 4..7
depth ladder
(src/depthctl.h): promote when the current ceiling saturates (sat >= 0.50 for
4->5, >= 0.60 for 5->6), demote when the top lane's CONDITIONAL yield drops
below the measured breakeven (0.35) or, at level 6, when margin runs reach
6-deep too rarely to amortize the 6th draft step (fired < 0.45). Measured:
+2.7% geomean over d4-gated across the payload envelope; +4.2% over fixed-d5
on real-CC-transcript traffic (220.7 vs 211.9 t/s @25.8K, same-harness server
replay 2026-07-09); envelope flavors that don't
saturate never promote and stay within noise of the 4..5 ladder. Knobs:
`Q27_MAXD_HI/HI6/LO/FLO6/EMA`; greedy also tolerates theta=1.0 for +4.9%, but
sampled theta=1.0 nets -2.1%, so 0.5 is the cross-path default. The sampled
path keeps a fixed depth-4 ceiling. Greedy output stays bitwise-identical
under gating AND under any ceiling (only round count/segmentation + verify
width change -- canonical 4c4120c7 EXACT at d4/5/6/auto); sampled output stays
seeded-reproducible. Note: under `auto` the round SEGMENTATION varies with the
controller's EMA state (e.g. across identical replays on one server), so round
counts are not replay-deterministic mid-convergence -- tokens always are. The gate
is ON by default on the server (`Q27_PMIN=0.5` since 07-10); the CLI and
`Q27_PROFILE=ref` leave it unset, so reference/canonical traffic is untouched.

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
| P5: GEMM tile tuning (grid swap + reg pipeline + vector unpack + NT=64) | Q4 GEMM **-36%** / Q8 **-48%** @26.6K; prefill **1388 -> 1790 t/s** @600; cold 28.5K TTFT **21.4 -> 16.8s** [superseded -- P6: 15.0s]; 128K prefill ~78 -> ~57s [re-measured 2026-07-06: current 128K prefill is ~71-80s (fp8 g64 71.5 / fp16 exact 80.4); both this 57s and P6's 117.6s superseded]; arithmetic bitwise-unchanged (canonical + pf IDENTICAL) |
| P6: column-split delta scan (SM-starvation fix #2) | kernel **748 -> 413 us** @T=256 (1.81x, 48 -> 384 blocks); 26K prefill wall **15.0 -> 13.5s** (-10.3%); 28.5K **16.7 -> 15.0s**; 128K **125.5 -> 117.6s** (fp16-KV kvstats method) [superseded 2026-07-06: current 128K prefill ~71-80s after g64 regroup + delta-WY tiling]; split-vs-exact 5e-8, PPL 7.1931 (+0.0003 = fp reorder), canonical md5 exact, pf IDENTICAL |
| fd2: register-accumulator flash-decode (SM-starvation/occupancy fix #3, attn was 99% of depth cost at 5% DRAM BW) | 61K depth **78.0 -> 126.2 t/s** (+62%, 47.2 -> 29.2 ms/round); 16K **-18%/round**; instance 0.768 -> 0.156 ms @61K (45% DRAM BW); 2K +1.3%/round; acceptance parity exact; PPL in noise both KV modes; nll-long 160K bucket-identical; CANONICAL RE-DERIVED 4c4120c7 (old 58b6ae85 under Q27_FD=v1) |
| P12: confidence-gated depth (`p_min` equiv; `Q27_PMIN=theta`) -- gate verify width on the drafter's top1-top2 margin, skip the deep-KV verify when unconfident | decode **grows with ctx: 2K neutral / 16K +5.8% / 60K +10.8%** (theta 1.0; +7.0% theta 0.5); greedy output BITWISE-IDENTICAL (lanes are independent grid indices -> only round count + verify width change); higher theta wins at longer ctx (context-adaptive theta confirmed). P12b depth-5 (`Q27_MAXD=5`, opt-in): agentic +2.6% but docs -8% (always drafts to max, so the 5th MTP pass is pure cost at low acceptance) -> depth-4 stays default; adaptive maxd is the follow-on |
| P14 Task 2: fuse draft argmax+margin (`k_argmax_top2`) -- kills the dead ungated `k_margin` scan | -0.545 ms/round @61K (the removed scan); canonical 4c4120c7 EXACT (bitwise); test_kernels +3 fused assertions (token==argmax, margin==CPU top1-top2, all err 0) |
| P14 Task 3: P12 confidence gate ported to the sampled spec path (per-width sampled verify graphs, capped accept walk) | sampled verify-narrowing ALONE is a wash @61K docs (+0.0% theta0.5 -- a low-margin draft the sampler may accept gets skipped, tok/round drops, extra rounds offset the cheaper round); greedy cross-check healthy on the same binary (+6.6% theta1.0); substrate for Task 4; canonical 4c4120c7 EXACT (greedy untouched) |
| P14 Task 4: draft early-exit (`Q27_DEXIT`, margin-gated per-step draft graphs, `min(W,md_used)` width-floor top-up) | same-binary A/B @61K docs: greedy **+3.2%** (theta1.0), sampled **+5.4%**; emitted bytes + round counts bitwise-identical to the monolithic draft in all 8 identity cells; sampled gated+dexit now **+3.6% over ungated** (Task 3's sampled wash resolved); canonical 4c4120c7 EXACT |
| P14 Task 5: fd2 lane-innermost grid order (partial cross-lane KV L2 reuse; R~4.25 measured) | same-session pre/post A/B @61K ungated **116.1 -> 119.3 t/s (+2.7%, MARGINAL-KEPT)**; verify fd2 per-instance -10% toward the draft floor; 2K neutral (+0.0%); canonical 4c4120c7 EXACT (2-line index remap, bitwise on the full fd2 matrix) |
| prefill-attn Phase 1: cp.async K/V double-buffered prefetch (fp8 path) | fp8 128K prefill **72.1 -> 68.2s (+5.4%)**; bitwise (convert-on-consume of identical bytes); first "neutral" reading was an fp16-KV test artifact -- cp.async is dead code off the fp8 path |
| prefill-attn Phase 2: fp8 QK^T MMA (`mma.sync.e4m3`, Q staged fp8, bank-conflict padding) -- DEFAULT-ON on fp8 KV | 128K prefill **68.3 -> 59.6s (+11.8%**, ~2200 t/s); logit cosine 0.9999827 + argmax MATCH @131K; needle **6/6 to ~301K**; fp16 path + canonical untouched; `Q27_PF_FP8MMA=0` opts out |
| verify-gemv: activation reads 4x uint2 -> 2x uint4 in `k_gemv_q4_n` (+ single-col) | decode @61K **163.2 -> 172.9 t/s (+5.9%)** on 2026-07-08 fixtures; GEMV was LATENCY-bound (long_scoreboard 90%, 39-47% DRAM peak) -- weights were fine, the per-column activation loads hammered L1TEX; bitwise BY CONSTRUCTION (same bytes, same dp4a order); tensor-core verify NOT justified |
| accept-gate Phase 1: conditional lane-5 yield + `maxd_lo` 0.10 -> 0.35 (the measured d5 crossover) | `Q27_MAXD=auto` becomes the production rec: **+2.7% geomean over d4-gated** across the 5-payload envelope, beats BOTH fixed ceilings; the old unconditional yield EMA sat above the demote bar on traffic where fixed-d5 measured -1.7% |
| maxd6: adaptive ladder 4..6 (7-lane verify, perm mod-7, +157 MB; 3-bar depthctl hi/hi6/flo6) | real-CC-transcript @25.8K: d4 202.6 / d5 216.1 / d6 222.0 (7-tok rounds on 64%); **auto 220.7 vs d5 211.9 = +4.2% same-harness** (2026-07-09 review rerun; original +4.7% claim mixed harnesses); text byte-identical at every ceiling; canonical 4c4120c7 EXACT; non-saturating flavors never promote past 5 |

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

[historical -- cold 28.5K TTFT is ~15.0s after P1-P6 and cold 128K is 59.6s
after the 2026-07-07/08 prefill-attn pair; the warm-turn number required the
P8 stable-prefix snapshot to hold on real re-rendering traffic]

## Roadmap

**Recently shipped (2026-07-10):** width-12 verify + suffix width 12 (live
AL 10.6); fdmma STAGES=1 2-CTA + computed split count (5.6x fd2 at 61K
W12); `__grid_constant__` param fix (suite 172.2, above the old baseline);
zero-config CC serving defaults + auto `--ctx` + `Q27_PROFILE=ref`; P3
verdicts (MTP 9/10 NO-GO, 8 NO-BUILD, GDN chunk shelved -- wide marginal is
GEMV-N-bound; mma16 GEMM-verify pivot is the filed lever); vanilla + llama
cross-engine A/Bs. Open engine ideas, none commissioned: Saguaro-style
3090 off-path drafting; mma16 GEMM verify; W=16 revisit if live suffix AL
ever jams at 11.

**Previously shipped (2026-07-05 -> 08):** `/v1/messages/count_tokens` + the
Anthropic-shaped context-limit error; sampling Phases 1-2 + the exit-gate A/B
(passed); the tool-call parser drift fixes that unblocked agentic Claude Code;
P12 confidence-gated depth (`Q27_PMIN`); the P14 perf-levers bundle (merged);
P15 constrain-tools engage-lag fix + serving-state gates; the prefill-attn
pair (cp.async +5.4%, fp8 QK^T MMA default-on +11.8% @128K); verify-gemv
(+5.9% decode @61K); accept-gate Phases 0-1 (conditional yield, measured
crossover, `Q27_MAXD=auto` production rec); and the **maxd6 adaptive ladder
4..6** (see the State section).

**P10-A status**: A0 PASSED, A1 SHIPPED (R1 multi-slot + R1b round
interleaving; whole-generation queue waits gone; analysis in
docs/P10-decision.md). A2 fused batch-10 verify stays CONDITIONAL: build
only if engine-claim telemetry says conversations outnumber slots in real
use (light no-spec utility slots are the smaller lever first).

**Sampling -- DONE (2026-07-06).** temperature/top-p with rejection-sampled spec
acceptance (Phases 1-2, at spec speed); greedy stays bitwise. The exit-gate A/B
PASSED under the production sampling config (docs/sampling-exit-gate.md): sampled
T=0.7 >= greedy on both harnesses, no regression, no new drift mode -- cleared to
default at T<=0.7 / top_p 0.95. Reattribution: the "deterministic one-shot-quit
T2 basin" once filed here as sampling's case was ROOT-CAUSED to mode-6 tool-call
drift and fixed at the PARSER -- not a sampling problem. The 32K-token
mega-generation that blew CC's output cap remains a prompt/sampling lever, as is
the output-volume wall gap.

**Measurement debts -- CLOSED (2026-07-06):**
- Strongest-opponent llama sweep DONE: Q5_K_M + draft-mtp single-stream
  greedy @~2K ctx -- draft=6/p_min0 (our A/B config) **102.4 t/s**,
  draft=10/p_min0.5 (r/LocalLLM ref) **117.9**. The +15% win is `p_min 0.5`,
  NOT draft depth (p_min gives +14 t/s at either depth; draft 6->10 buys
  ~0-1). So our cross-engine A/B UNDER-STATED llama by ~15%; the honest
  strongest-llama decode baseline is **~117 t/s @2K** (q27 @2K 169-209 still
  wins clearly at short ctx).
- **Depth-matched cross-engine, multi-prompt confirm DONE 2026-07-07 (4 flavors
  @~75.4K matched tokens, greedy, 3 replays each, spread <=0.3%): post-P12/P14
  q27 (gated 0.5) is at PARITY with tuned llama (draft10/pmin0.5) on mixed
  flavors** -- transcript 123.3 vs 111.4 (q27 +10.7%), repro 153.0 vs 154.4
  (tie), code 93.1 vs 99.5 (llama +6.9%), geomean 120.6 vs 119.6. llama keeps
  a structural +45% on PURE-ECHO traffic (229.9 vs 158.0, 100% draft
  acceptance): depth-10 drafts vs q27's 4/5 ceiling, the regime the maxd6
  NO-GO already priced. The 07-06 n=1 result (ungated q27 145.6 vs 190.3,
  "~31% faster") was repro-flavored and pre-P12 -- superseded. P12 gate
  confirmed multi-prompt at depth: +4.3/+8.0/+11.1% (biggest at LOW acceptance).
- 128K prefill re-measured (`--kvstats 131072`, synthetic tokens -- prefill
  time is value-independent): **fp8 g64-default 71.5s, fp16 g64 76.5s, fp8
  exact(`Q27_PF_XG=32`) 75.5s, fp16 exact 80.4s** -- ~1700-1830 t/s. KV format
  ~6%, g64-vs-exact ~6%. **P6's 117.6s is STALE** (predates the g64 regroup
  +8.8% and delta-WY tiling); P5's ~57s was optimistic. Honest current
  128K prefill: **~71-80s**.

**Confidence-gated depth (q27's `p_min` equivalent) -- SHIPPED (P12, 2026-07-06).**
Empirically motivated by the depth-match (llama's p_min 0.5 is worth +36% at ~75K,
exactly why tuned llama beat q27 at depth). Mechanism: per-step drafter-margin gate
(the top1-top2 margin, the p_min analog) caps the verify width, skipping the
expensive deep-KV verify when the draft head is unconfident. `Q27_PMIN=theta`
engages it (unset = off = the canonical depth-4 round); greedy output stays
bitwise-identical (lanes are independent grid indices, so only round count + verify
width change). **Measured decode: 2K neutral, 16K +5.8%, 60K +10.8%** (theta 1.0),
growing with context, higher theta winning at longer ctx -- the context-adaptive
theta the offline margin bins predicted. The prior fixed-depth (P3), adaptive-depth,
and burst-depth negatives were all UNGATED or accept-count-gated and did not cover
this. **P12b depth-5** (`Q27_MAXD=5`, opt-in) works but is traffic-dependent
(+2.6% agentic vs -8% docs -- Path C always drafts to gate_maxd, so the 5th MTP
pass is pure cost at low acceptance), so depth-4 was the default at P12 time [the server default is `Q27_MAXD=auto7` (ladder 4..7) since 07-10]. Adaptive maxd
(P13, `Q27_MAXD=auto`) floats the ceiling 4..5 per stream from realized
acceptance. On branch `p12-confidence-gated-depth` (Phase-0/0b margin
measurement + implementation; see BUILDLOG).

**P14 continuation -- SHIPPED (2026-07-07, merged to master).**
The P12 gate now runs on the production SAMPLED path too, and draft early-exit
(`Q27_DEXIT`) closes the other half of llama's p_min -- the P12 gate only
narrowed VERIFY, while llama's p_min also stops DRAFTING. The Task 3 finding is
load-bearing: verify-narrowing ALONE is acceptance-NEGATIVE under sampling (a
low-margin draft the sampler might still accept gets skipped, so tok/round drops
and the cheaper round only breaks even, +0.0% @61K); it is the draft-side
early-exit that makes the sampled gate pay (+3.6% over ungated @61K). Fused
draft argmax+margin (`k_argmax_top2`) and the fd2 lane-innermost L2 fix
(+2.7% @61K, MARGINAL-KEPT) round out the bundle. Production config:
`Q27_PMIN=0.5` + `Q27_DEXIT` on both paths. Attribution:
docs/perf-attribution-p14.md.

**Open decode/prefill levers (post-maxd6):**
- **prefill-attn Phase 3 (occupancy; requires Gabe's explicit go on the
  smem-relayout).** [RETIRED 07-09: Phase 3a doubled occupancy for -1% TTFT -- the kernel is BARRIER-serialized, not occupancy-bound; the filed follow-up is the async producer/consumer + mbarrier rewrite (Eligible-Warps target).] The prefill-attention kernel was believed OCCUPANCY-bound (12.5%,
  dual register+smem limiter, DRAM 2%/tensor 33%); cp.async and fp8-MMA
  captured the latency-hiding wins available WITHIN 6 warps. A 2-CTA/SM play
  needs both a register cut (the o[32][4] accumulator = 128 regs) and smem
  halving -- a from-the-layout rewrite. Attention is still ~half of 128K
  prefill.
- **d7/d8 ladder extension, telemetry-gated.** cctx sat6 = 0.64 still
  saturates at depth 6; live `glf=`/`gla=` now report lane-6 fired/accepted on
  real serving traffic. Extend by the same recipe (S_spare7, hi7/flo7) IF
  sustained sat6 >= ~0.6 shows up live; the pointer-array lane refactor
  (maxd6-decision.md) becomes worth it at d7+. The P4-echo tail (llama
  depth-10 +45%) is the prize.
- **Task 6 fd2 lane-pair fusion (requires Gabe's explicit go).** Task 5
  captured only ~10% of the R~4.25 cross-lane KV headroom; the residual
  ~6 ms/round verify-attention is the lane-pair-fusion target -- helps every
  round of every traffic class, ~0 VRAM, and it lowers the ladder's per-lane
  breakeven (deeper levels get cheaper). The expensive kernel rewrite (fd3
  design doc + occupancy gate); DEFERRED pending explicit approval.
- **docs-class promote churn (~1%).** Boundary traffic (bursty sat5 ~0.46-0.5)
  keeps a ~1% auto-vs-fixed gap from promote exploration, bounded by flo6.
  Known shave: a demote-count promote-escalator. Not built (YAGNI at 1%,
  worst flavor only).

**Open quality gates (red-team pass 2026-07-05; P15 status 2026-07-07):**
- strict-parser A/B -- **DONE 2026-07-08, verdict: NOT engine-true; the mode-1
  (dropped-wrapper) rescue is load-bearing.** `Q27_TOOL_STRICT=1` severs every
  rescue (plain-JSON wrapped calls only, bare-scan off, [q27-strict] logs count
  suppressions). T8 CC greedy: tolerant **0.837** (12 mode-1 rescues incl. the
  OPENING turn) / strict **0.000** (first-turn wrapper-less call suppressed ->
  CC one-shot-quits) / strict + `--constrain-tools` **0.549** (grammar carries
  wrapped calls, session survives 491s, but one mid-session wrapper-less turn
  still bypasses the grammar -- the constrainer only engages at `<tool_call>`,
  so a bare-JSON turn is invisible to it). Honest framing: q27+CC scores
  measure engine+tolerant-parser as a SYSTEM; llama-server's chat parser has
  the same tolerance class, so cross-engine A/Bs stay apples-to-apples.
  Follow-up lever (not built): engage the constrain grammar on a bare
  `{"name"` opener too, closing the wrapper-less bypass -- that would make
  strict+constrain the zero-rescue configuration.
- constraint-cost soak -- **MEASURED 2026-07-07 (P15 session): in-call cap=1
  costs 3.1x at 75.7K depth** (33.0 vs 102.2 t/s inside call bodies, ~+4s per
  call-turn, byte-identical outputs) -> `--constrain-tools` stays OPT-IN for
  speed; safe (no score-0 basins) when robustness is worth the tax. The in-call
  speed fix remains the P11 split path (4.2x, blocked on the race below)
- constrain-tools x serving-state gates: SHIPPED with P15 -- split-brain
  ids are validated + rebound (rebinds counter), pool-full sticky-
  disengages per request (visible in `tg=`), and the device constraint is
  cleared at request claim (leak test in tools/constrain_gate.sh). Keep
  `Q27_TOOL_SPLIT` off under `--slots` (P11 race). Checkpoint-restore x
  grammar needs NO gate: audited 2026-07-05, restore touches only GDN
  state/conv rings/positions and grammar is per-request, engaging only on
  decoded output. Assistant-prefill continuations that end mid-tool-call
  decode unconstrained by design (parser recovery is the net)

**Measured and parked:**
- fixed depth-5 (+2-4% @2K for +12-14% round cost) and ungated burst depth
  (see the reopen candidate above for the gated variant that is NOT covered)
- chunked-WY delta scan
- cross-session checkpoint pool (P9 covers same-session)
- importance-weighted scales, AWQ-style (only path left on the +3.05% PPL
  gap -- see risk 2; Thunderdome says the gap doesn't bite on agentic coding)
- P11 split-path (`Q27_TOOL_SPLIT`): unexplained crash under accumulated
  multi-request state -- flake hunt required before any split/adaptive
  path ships; keep OFF under `--slots`

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
