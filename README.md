# Quasar

A narrow inference engine for **Qwen3.6-27B-MTP** (hybrid GDN+attention, trained-in MTP heads) and its fine-tunes on a single RTX 5090 (also supports 3090). One model family, one GPU, as fast as possible. In the spirit of [antirez/ds4](https://github.com/antirez/ds4)

## Why this is interesting

- **Fastest known way to run this model.** +47% decode over tuned
  llama.cpp on a 5090 (same model, GPU, harness, day; protocol filed
  before it could pass). On a 24GB 3090: +19% decode at +60% context
  over mainline llama.cpp. vLLM measured 4.7x slower wall
  on real Claude-Code traffic (its prefix cache gets 0% reuse on this
  hybrid-GDN architecture). sglang 0.5.15 cannot load the model at all
  (quantized-checkpoint loader gaps on the GDN layers; BUILDLOG
  2026-07-12).
- **turbo3 3-bit KV cache**, symmetric K+V: 13.4 KB/token, ~1% PPL,
  needle 6/6 at a 361K-token prompt, 655K context allocatable on a
  5090, two full 131K tenants at once, and a 24GB card promoted from a
  32K box to a 131K box. Ported from
  [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant);
  that fork refuses 3-bit K on this model class and caps 33% lower; I measured K costs +0.17%.
- **Native Anthropic Messages endpoint at Claude-Code grade**: thinking
  blocks, tool_use with input_json_delta streaming, exact
  `count_tokens`, anthropic-shaped context-limit errors, billing-header
  normalization so the prefix cache survives real CC turns, and a
  tolerant tool-call parser (nine cataloged drift modes) that is
  load-bearing for ANY engine on this harness. One env var points
  Claude Code at it; OpenAI and Codex (Responses) shapes ride the same
  binary.
- **Self-speculation as the whole design**: trained-in MTP ladder +
  free suffix drafter through one shared-KV MMA verify -- 5.3-5.8
  accepted tokens per weight read on live traffic (231-246 t/s
  aggregate on a 5090).
- **Receipts for everything**: bitwise canonical gates, negative
  results logged at the same rate as wins, and every number in this
  README traceable to a dated BUILDLOG entry. (The BUILDLOG is the
  ledger, not git history -- commits sometimes batch a day's entries.)

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
  server 400.6 t/s** -- the degenerate-echo CEILING.
- Live Claude-Code traffic (vanilla, 9 scored task trials across 3
  task types, 430 requests): **231.3 t/s aggregate**, per-request median 225 / p75 277 /
  **peak 378 t/s**; suffix drafter AL 9.4 on 37% of decode.
- Prefill (fp8 batched TTFT): 8K 2.35s | 32K 10.4s | 128K 59.4s (~2200 t/s).
- Cross-engine (2026-07-10, protocol filed 07-05: 3 scored task trials x 3 task
  types per engine, both legs strongest config, same day/GPU/harness):
  **q27 +47% decode vs llama.cpp's best** (231.3 vs 157.4 t/s aggregate
  over 430/197 requests; medians 225 vs 155, peaks 378 vs 274; llama =
  Q5_K_M draft-mtp10/p-min0.5/fa). Score medians converge per task
  (0.83 == 0.83; 0.78 vs 0.79) -- quality is the model's, and score parity is
  a system claim (the tolerant parser is load-bearing for BOTH engines;
  strict parsing scores 0.000 on the hardest task class). In-band draws q27 8/9
  vs llama 5/9 -- n=9 can't separate that (Fisher p~=0.29), reported
  descriptively. Wall favors q27 3-4x but is trajectory-confounded;
  decode telemetry is the rate currency. Decomposition: ~15 points of
  the 47 are bit-width (15.8 vs 18.2 GB/step), the rest is mechanism.
  Arc: tuned llama +31% on 07-06, parity 07-07, q27 +47% on 07-10.
  llama's ngram spec wins the repetition regime: on file re-emission
  its unbounded prompt-lookup drafts beat q27's fused MTP+suffix (an
  external fork-maintainer A/B measured 653 vs 377 t/s), because q27's
  suffix draft truncates at the 12-lane verify cap -- which DOES bind
  here (81% of suffix fires pin at 12; BUILDLOG 2026-07-13). Against
  llama's *deployed* config (draft-mtp) q27 still wins both regimes;
  the ngram win is to a mode llama does not run in production and that
  is mutually exclusive with draft-mtp and poor on novel prose.
- Fine-tune headroom, REVISED by the matched 21-task sweep (07-11):
  Qwopus is a SPEED fine-tune worth **+5.7% decode on real traffic**
  (246.5 vs 233.1 t/s aggregate, 5.65 vs 5.31 tok/rnd) at quality TIE
  with vanilla (median 0.836 vs 0.830, 13/21 tasks tied). The old +35%
  figure was the echo-heavy cctx replay BEST-CASE, not a traffic number.
- q6k weight tier (07-12, `qwen36-27b-mtp-q6k.q27`, 23.25 GB = 6.8
  bits/param): q6 + ffn_gate to Q8. Matched-protocol PPL **7.9127** --
  below unsloth Q5_K_M (7.9179) and Q6_K (7.9811), and inside
  single-run noise of their 26 GB UD-Q6_K_XL flagship (7.9584) at
  2.75 GB smaller. ffn_up is deliberately NOT promoted -- measured
  WORSE (+1.6% over q6; its Q4 noise cancels inside the SwiGLU
  product, the second such structure after the GDN in-projections).
  Decode 143.1 suite / 150.6 @26K; auto-ctx fp8 114688, turbo3 262144.
  5090-class only.
- q6 weight tier (07-12, `qwen36-27b-mtp-q6.q27`, 20.5 GB = 6.0
  bits/param): v1.4 + ffn_down promoted to Q8. Matched-protocol PPL
  8.0409 -> **7.9460** vs the Q5_K_M bar 7.9179 -- the gap to Q5_K_M
  shrinks from +1.55% to **+0.35%**. Price: suite decode 171.5 -> 152.1
  t/s (-11%), 26K replay 176.6 -> 169.2 (-4%). fp8 auto-ctx 196608 on a
  5090. Does NOT fit 24GB cards (fixed cost alone is 24.2 GB) -- the
  default 5.25 bpw artifact remains the 3090 answer.
- Weight-tier decode on live CC traffic (07-12, same-day scored task
  trials, 3 per tier, per-request median / p90, dec>=32 requests):

  | tier | bpw | live decode | vs default |
  |---|---|---|---|
  | default | 5.25 | **224.7** / 328.6 t/s | -- |
  | q6 | 6.0 | 212.4 / 308.7 | -5.5% |
  | q6k | 6.8 | 201.0 / 317.6 | -10.5% |

  Task scores do NOT separate across tiers (bimodal scoring basins at n=3; all 9
  trials completed) -- the quality tiers buy PPL margin, not task
  scores. The serving default stays 5.25 bpw.
- Weight tier x KV format -> max context on a 32GB 5090 (auto-ctx
  picks; * = boot-verified 07-12, ~ = formula from the same anchor):

  | KV format | default 5.25 bpw | q6 | q6k |
  |---|---|---|---|
  | fp8 (34 KB/tok) | 262144 (cap; 294912 fits) | 196608* | 114688* |
  | turbo3 (13.4 KB/tok) | 262144 (cap; 655360 fits) | 262144 (~495K fits) | 262144* (~292K fits) |
  | fp16 (68 KB/tok) | 131072 (cap) | ~98K | ~57K |

  turbo3 absorbs the tiers completely -- even q6k keeps the full native
  262144 window. fp8 is where the tiers bite (~29K tokens of window per
  GB of weights); q6k+fp8 at 114K can pinch a deep agentic session, so
  pair the quality tiers with `Q27_KV=turbo3`.
- 3090 (24GB, turbo3 + h16, 07-12): **102.2 t/s median** live CC decode
  at **131K ctx**, 3/3 scored task sessions -- vs vanilla mainline llama.cpp's
  85.6 t/s at 82K (2/3, one context-wall crash): **+19% decode, +60%
  ctx** on the strongest vanilla config. Raw prefill on the 3090 goes
  the other way: llama pp8192 1355 t/s vs q27 1065-1089 (-21%; parity
  on the 5090) -- serving-effective prefill still favors q27 via its
  prefix cache (2309-2569 vs 1720 t/s on real CC traffic).

## Quickstart

Requirements: an NVIDIA GPU with 24GB+ VRAM, CUDA toolkit 12.x at
`/usr/local/cuda`, and gcc. `make` builds ONE dual-arch binary
(sm_86 + sm_120: 3090 and 5090 class); arch dispatch is at runtime --
fp8-KV and the e4m3 MMA paths need sm_89+, Ampere (sm_86/80) runs the
fp16-MMA verify (h16) and fp16/turbo3 KV.

24GB cards (3090-class): build `make build/q27-server-w8` as well --
`Q27_W_MAX=8` shrinks the fixed VRAM stack so the server fits; the
default width-12 build OOMs at graph setup on 24GB. Serve it with
`Q27_KV=turbo3` for 131K context (fp16 KV caps ~32K there). The card
must be otherwise idle: ~2.7GB of other resident VRAM is the difference
between boot and OOM.

Pick a quant first. Three tiers, one repo
([signalnine/Qwen3.6-27B-MTP-q27](https://huggingface.co/signalnine/Qwen3.6-27B-MTP-q27))
-- all serve identically,
they trade decode speed for model quality:

| tier | file | GPU | pick it when |
|---|---|---|---|
| **default** (5.25 bpw) | `qwen36-27b-mtp.q27` | 24GB+ | you want max speed; the only tier that fits 24GB cards |
| q6 (6.0 bpw) | `qwen36-27b-mtp-q6.q27` | 32GB | +0.35% PPL off Q5_K_M for ~5% slower decode |
| q6k (6.8 bpw) | `qwen36-27b-mtp-q6k.q27` | 32GB | quality matching the best GGUFs of this model, ~10% slower decode |

Task scores measure the same across tiers (scored task trials, 07-12)
-- the quality tiers buy perplexity margin, not benchmark wins. When in
doubt take the default; with `Q27_KV=turbo3` every tier keeps the full
262144-token window on a 32GB card.

```bash
# 1. tokenizer + your chosen tier from Hugging Face (Apache-2.0);
#    swap --include for the tier file you picked above
huggingface-cli download signalnine/Qwen3.6-27B-MTP-q27 \
  --include qwen36-27b-mtp.q27 qwen36-27b-mtp.tok CHECKSUMS.md5 \
  --local-dir models/qwen36-27b-mtp
# fine-tune variant: signalnine/Qwopus3.6-27B-v2-MTP-q27
# verify: (cd models/qwen36-27b-mtp && md5sum -c CHECKSUMS.md5 --ignore-missing)

# 2. build (CLI + server + test suites) -- or skip the toolchain and
#    grab prebuilt linux x86_64 binaries (CUDA runtime statically
#    linked) from https://github.com/signalnine/q27/releases
git clone https://github.com/signalnine/q27 && cd q27
make

# 3. smoke test the CLI (should print 128 tokens; md5 of the output
#    line is the bitwise canonical a2982c51...)
./build/q27 ../models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
  --tokens "760,6511,314,9338,369" -n 128 --ctx 2048 --spec

# 4. serve -- zero config; defaults resolve the full measured stack
#    and --ctx auto-sizes to your VRAM (see Serving for escapes).
#    Binds 127.0.0.1 only: the server has NO auth. To reach it from
#    other machines or from containers (Claude Code in docker resolves
#    the host via the bridge, not loopback), opt in explicitly:
#      --host 0.0.0.0
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

Five stages shipped in one day, each gated; the full entries (and their
negatives) are the 2026-07-10 BUILDLOG block. Headlines:

- **Width-12 verify**: lanes 8 -> 12; widths 9..12 belong to the suffix
  drafter (live agentic-session AL 10.61 on 61.6% of decode -- the predicted cap
  release). Byte-identical at old widths by construction.
- **fdmma tuned 5.6x over fd2** at 61K W12 (2-CTA STAGES=1 + computed
  split count); three orchestration negatives filed with a do-not-retry
  bar (CTA count dominates intra-CTA choreography here).
- **`__grid_constant__`** on struct-param kernels: the short-bench suite
  caught an 8% engine-wide param-copy tax the depth gates missed; suite
  172.2, above the pre-regression baseline. The suite is the standing
  launch-overhead canary.
- **Zero-config serving**: bare `q27-server model tok` = the full
  measured stack; CLI keeps reference defaults (bitwise world untouched).
- **Deep-MTP closed by pricing**: ceilings 9/10 negative, 8 NO-BUILD;
  the wide-lane marginal is GEMV-N-bound (mma16 GEMM-verify is the filed
  pivot, not commissioned).

(2026-07-11/12 shipped the turbo3 KV arc and the h16 Ampere verify --
see the features list above and BUILDLOG.)

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
- Agentic serving: the tolerant tool-call parser is load-bearing -- nine
  drift modes plus the 07-11 inference tie-break; CC 0.00 -> 0.55 was a
  parser ceiling, not quant. `--constrain-tools` stays opt-in (in-call
  cost).
- Multi-slot serving (`--slots N`) with R1b round-granularity GPU
  time-slicing; P9 same-session checkpoint ring; P8 stable-prefix snapshot
  (warm turns ~1.3s); `/v1/messages` native incl `count_tokens`. turbo3
  fits TWO full 131K slots on the 5090 (capacity, not vLLM-style aggregate
  -- docs/multislot-throughput.md).
- Long-context: needle 6/6 at 361K on both fp8 and turbo3; measured
  allocation ceilings fp8 294,912 / turbo3 655,360 (W12 build, 5090).
- turbo3 3-bit KV (2026-07-11, `Q27_KV=turbo3`): full stack incl. the
  fp8-MMA verify leg; PPL +0.87% flat to 297K, acceptance TIES fp8 on
  basin-matched replay, needle 6/6 at a 361K prompt, allocates to 655K
  (2.5x native); promotes a 24GB 3090 from a 32K box to a 131K box.
- fdmma-h16 (2026-07-12): fp16-MMA verify on sm_80+, all KV formats,
  W<=8 -- Ampere's mma leg (3090 +32% replay decode; profile shows the
  round then sits at 81-90% of the DRAM roofline) and fp16-KV's first
  mma leg on every arch.

## Why this model is a good target

- Dense-ish 27B that fits entirely in 32 GB VRAM at 4-bit -- no expert offload, no DRAM scatter, none of the DSV4 pain
- MTP draft head trained into the checkpoint: self-speculation without a separate draft model
- Hybrid Gated-DeltaNet: near-O(1) memory per token for 48 of 65 layers.
  KV lives only in the 17 full-attention layers (16 + MTP, all global, no
  windowing): 68 KB/token fp16. A dense 65-layer build would need ~68 GB
  @256K.
- Measured 5090 KV ceilings: fp16 ~180K | fp8 (34 KB/tok, server default
  since 07-03) **294,912** measured | turbo3 3-bit (13.4 KB/tok)
  **655,360** -- 2.5x the 262K native window. Auto-ctx caps at 262144 for
  fp8/turbo3, 131072 for fp16; explicit `--ctx` overrides.
- turbo3 long-ctx quality: position-bucket NLL flat through 297K
  (tracks fp8 within +0.65-1.2% every bucket), needle 6/6 at a 361K
  prompt, two full 131K slots at once.
- The catch the per-token-memory napkin misses: attention KV is RESTORABLE state (any prefix row range replays for free) while GDN recurrent state is all-or-nothing per sequence -- you can only resume from a position you snapshotted. Hybrids make per-user context cheap but make context REUSE an engineering problem (prefix cache, mid-history divergence, multi-doc serving). That trade is where P8/checkpoint work lives; the measured cost of ignoring it was 7.9x wall-clock on agentic traffic (see build log P8/P9)
- The opponent, tuned honestly: llama.cpp's best measured config on this
  box is Q5_K_M + draft-mtp10 + p_min 0.5 (**~117 t/s @2K** single-stream;
  the win over stock is p_min, not draft depth -- swept 07-06). All
  cross-engine numbers use that config; see Reference numbers.

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

Single-stream decode is weight-read-bound. The model: t/s = BW x eff /
bytes-per-step x accepted-tokens-per-round. Every number below is
measured.

| Card | DRAM | GEMV eff | Plain ceiling | Live agentic (measured) |
|---|---|---|---|---|
| 5090 (GDDR7) | 1.79 TB/s | 85-90% assumed, consistent with live rates | ~103-109 t/s @15.8 GB/step | **231-246 t/s** aggregate (vanilla/Qwopus, CC harness, 07-10/11) |
| 3090 (sm_86) | 936 GB/s | **81-90% ncu-MEASURED** (big FFN GEMVs ~90% DRAM SOL) | ~52 t/s | **102.2 t/s** median (turbo3+h16, 07-12) |

The efficiency assumption stopped being an assumption on 07-12: ncu on
the 3090 clocks the GEMV family at 81-90% of DRAM speed-of-light, and
the GEMV weight stream is 68% of the round. Plain decode's residual
~15% tail is GDN recurrence + ~140 launches/token; three attempts on it
(E4/E5/cp.async) came back negative, and the 3090 profile re-confirmed
there is nothing else material left in the kernels.

### Why self-speculation is the whole game at batch 1

Arithmetic-intensity framing (the same napkin datacenters use for the
opposite conclusion): the GPU offers hundreds of int8 ops per byte of
DRAM bandwidth, and batch-1 decode uses ~2 -- >99% of compute idles
while weights stream. Datacenters close that gap by batching USERS per
weight read; q27 batches WITH ITSELF: the width-12 verify amortizes one
weight read across the MTP ladder's drafts plus the suffix drafter's
free lanes. Live traffic runs 5.3-5.8 accepted tokens per round (echo
stretches hit 9-10.6), which is how 231 t/s clears a ~105 t/s plain
ceiling on the 5090 and 102 clears ~52 on the 3090. Corollary, twice
proven now: every decode win is (a) fewer bytes per step (quant policy,
fp8 KV, turbo3 KV) or (b) more accepted positions per weight read
(ladder, suffix width) -- there is no third lever at batch 1, and
docs/multislot-throughput.md is what happens when you ask for one.

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
  per build**. A quality flip on ONE benchmark basin (an auth-gate scoring case)
  is therefore read via a basin MATRIX across tasks plus a re-roll on the
  next binary, not via a single retrial: mma flipped that task bad on the 07-10
  morning binaries and re-rolled GOOD (0.84-0.85) on the width-12
  binaries with the identical kernel -- per-binary lottery, not
  kernel-class steering. Acceptance-sensitive decisions must name their
  basin; cross-BUILD text comparisons are invalid (same-binary legs
  only).
- **2K soak** (long-generation number): 2000-token generation, **209.2 t/s
  STOCK fd2-era** (4.32 t/round; pre-fd2 213.2/4.36, the ~2% is the
  short-ctx split tax). Headline for agentic reply-length outputs.
- **Depth numbers**: the current predictors are the Reference-numbers
  block (cctx 26K 143.0/176.3, live CC 213-246 aggregate); the 07-08-era
  61K/74K series lives in BUILDLOG with its cross-era bridges. Depth
  numbers, not 2K numbers, predict agentic wall time.

OC policy: headline + SOTA comparisons are reported STOCK (community numbers
aren't OC'd; sidesteps the non-ECC tail-risk conversation). +3000 stays a
supervised-bench option (+2.3% short-bench measured); the weight-checksum
tool (`--verify-weights`, `/health?verify=1`) exists for OC sessions.

## Design decisions

- **Weights**: custom 4-bit symmetric groupwise (group 64, fp16 scales), packed for coalesced 128B warp loads, dequant fused into GEMV. Embeddings, lm_head, MTP layer, norms at 8-bit/f32. Repacked offline from the BF16 GGUF (container spec: docs/FORMAT.md).
- **KV cache**: fp16 by default; fp8 E4M3 is the server default since
  07-03 (`Q27_KV=fp8` on the CLI). Scale-free saturating conversion --
  measured amax sits 3.8x under the 448 E4M3 max, so per-row scales buy
  nothing. 34 KB/token, +11% decode @28.5K, PPL -0.05%, KL 3.4e-5. The
  CLI stays fp16 so the bitwise canonicals hold.
- **turbo3 3-bit KV** (`Q27_KV=turbo3`; `turbo3v` = fp16-K diagnostic):
  WHT-rotated 50-byte blocks per 128 dims (ported from
  [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant),
  src/turbo3.cuh), 13.4 KB/token. Full stack: decode, verify (fdmma
  dequant-to-e4m3 tiles), batched prefill. PPL fp16 7.317 / fp8 7.327 /
  turbo3 7.381; K costs +0.17% -- the GQA=6 K-crater the source fork
  guards against does not exist on this model, and keeping K at 3 bits
  is worth the 98K-vs-131K ctx gap against that same fork on a 3090.
  Acceptance ties fp8 exactly on basin-matched replay (5.818 tok/rnd);
  wall -4.4% @27K, +9.6% @61K.
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
fp8 KV + `Q27_FD=mma` (e4m3 on sm_89+, fp16-MMA h16 on sm_80..88; fp8 KV itself needs sm_89+, older parts default fp16 KV),
`Q27_PMIN=0.5`, `Q27_MAXD=auto7`, suffix drafter at width 12, fast-head,
no-think, phase stats; `--ctx` auto-sizes the KV budget to free VRAM
(capped at the 262144 native window for fp8/turbo3, 131072 fp16;
single-slot). Every knob keeps its env/flag override
(user env always wins), `Q27_PROFILE=ref` restores the conservative
reference behavior (fp16, ungated, no suffix, fd2), and the **CLI binary
keeps reference defaults** so the bitwise canonical gates are untouched.
Escapes: `--kv-fp16 --no-fast-head --think`, any individual `Q27_*`.

Behavior note (`--think`): the default serving profile is no-think for
speed -- it prefills an empty `<think></think>` block, which is what
carries the ~224 t/s headline. The cost is a reasoning model handed
zero reasoning budget, which over-refuses a narrow class of
borderline-but-legitimate requests (measured: a signed-authorization
pentest command it declines under no-think, it supplies under `--think`
after reasoning through the authorization; BUILDLOG 2026-07-13). If your
workload includes security/compliance-sensitive requests that should be
answered, run `--think` and trade the speed for the reasoning pass.

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
The tolerant tool-call parser recovers nine observed drift modes -- dropped
`<tool_call>` wrapper, truncated JSON, `<content>`-tagged and
quote-open/`</content>` bodies, `{"tool_call":` openers, in-string control chars,
and name-dropped `{"name":\n{args}}` calls (tool inferred from the arg-key
signature; on a score tie, candidates whose required params the args don't
cover are eliminated -- modern CC registries carry property-twins like
Bash/Monitor) -- logging each recovery for the drift catalog. `--fast-head` trades
output exactness for ~7% more t/s.

**Confidence-gated depth (P12 + P14), greedy and sampled.** `Q27_PMIN=theta`
caps verify width on the drafter's top1-top2 margin; `Q27_DEXIT` also stops
drafting at the first sub-theta margin (llama's p_min draft-stop). Server
defaults since 07-10: `Q27_PMIN=0.5 Q27_MAXD=auto7` -- the adaptive 4..7
ladder (src/depthctl.h; thresholds and knobs documented there). Measured:
+2.7% geomean over d4-gated across the envelope, +4.2% over fixed-d5 on
real-CC-transcript replay. Greedy output is bitwise-identical under gating
and at every ceiling (canonical EXACT at d4/5/6/auto); round segmentation
varies with controller state, tokens never do. The CLI and
`Q27_PROFILE=ref` leave it all unset.

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

Headline numbers from E2 onward include a GDDR7 offset. Consumer GDDR7
has no ECC and weights load once, so a marginal OC can plant a persistent
silent error the token-identity gates can't see. That happened on
2026-07-02 at +4000 (one wrong canonical run after 30 min of heat, then
clean again -- binary confirmed innocent). Daily offset is +3000 since:
the band above it bought ~0.4% and produced the soft error. +4000 only
for short supervised benches; `--verify-weights` / `/health?verify=1` is
the detector; offset is volatile across reboots.

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

**Recently shipped (2026-07-11/12):** the turbo3 KV arc (port -> prefill
-> fdmma leg -> ctx sweeps -> 2-slot -> 3090 promotion) and fdmma-h16;
BUILDLOG has the dozen entries.

**Shipped 2026-07-10:** width-12 verify + suffix width 12 (live
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

**Sampling -- DONE (07-06).** temperature/top-p with rejection-sampled
spec acceptance, at spec speed; greedy stays bitwise. Exit-gate A/B
passed (sampled T=0.7 >= greedy, both harnesses); cleared to default at
T<=0.7 / top_p 0.95. The one-shot-quit basin once blamed on sampling was
a parser bug.

**Measurement debts -- CLOSED (07-06/07):** strongest-llama swept (the
honest baseline is ~117 t/s @2K; p_min 0.5 is the win, not draft depth);
depth-matched multi-prompt confirm put gated q27 at parity with tuned
llama at 75K (geomean 120.6 vs 119.6; llama keeps +45% on pure-echo, the
regime the maxd6 NO-GO priced); 128K prefill re-measured honest at
71-80s. All superseded by the 07-10 protocol A/B; receipts in BUILDLOG.

**Confidence-gated depth -- SHIPPED (P12, 07-06).** Motivated by the
depth-match: llama's p_min was worth +36% at 75K. Measured 2K neutral /
16K +5.8% / 60K +10.8%, growing with context; greedy bitwise-identical.
Details in Serving above and BUILDLOG.

**P14 -- SHIPPED (07-07).** The gate runs on the sampled path too, and
`Q27_DEXIT` adds the draft-stop half of p_min. Load-bearing finding:
verify-narrowing alone is a wash under sampling (+0.0% @61K); the
draft-side early-exit is what pays (+3.6%). Attribution:
docs/perf-attribution-p14.md.

**Open decode/prefill levers (post-maxd6):**
- **prefill-attn Phase 3 (occupancy; requires Gabe's explicit go on the
  smem-relayout).** [RETIRED 07-09: Phase 3a doubled occupancy for -1% TTFT -- the kernel is BARRIER-serialized, not occupancy-bound; the filed follow-up is the async producer/consumer + mbarrier rewrite (Eligible-Warps target).] The prefill-attention kernel was believed OCCUPANCY-bound (12.5%,
  dual register+smem limiter, DRAM 2%/tensor 33%); cp.async and fp8-MMA
  captured the latency-hiding wins available WITHIN 6 warps. A 2-CTA/SM play
  needs both a register cut (the o[32][4] accumulator = 128 regs) and smem
  halving -- a from-the-layout rewrite. Attention is still ~half of 128K
  prefill.
- **Task 6 fd2 lane-pair fusion -- MEASURED NEGATIVE (07-07).** Bitwise
  but -4%: fd2 decode attention is LATENCY-bound, not BW-bound. Retry bar:
  <=~128 registers. (Superseded in practice by fdmma/h16 anyway.)
- **docs-class promote churn (~1%).** Boundary traffic (bursty sat5 ~0.46-0.5)
  keeps a ~1% auto-vs-fixed gap from promote exploration, bounded by flo6.
  Known shave: a demote-count promote-escalator. Not built (YAGNI at 1%,
  worst flavor only).

**Open quality gates (red-team pass 2026-07-05; P15 status 2026-07-07):**
- strict-parser A/B -- **DONE 2026-07-08, verdict: NOT engine-true; the mode-1
  (dropped-wrapper) rescue is load-bearing.** `Q27_TOOL_STRICT=1` severs every
  rescue (plain-JSON wrapped calls only, bare-scan off, [q27-strict] logs count
  suppressions). hardest-task CC greedy: tolerant **0.837** (12 mode-1 rescues incl. the
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
  gap -- see risk 2; scored task trials say the gap doesn't bite on agentic coding)
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
6. **fp-precision paths break the bitwise gate.** Batched prefill is currently bit-identical to serial because dp4a's int32 block sums are order-independent and the per-group fp scale-and-add matches serial order. RESOLVED for prefill: the int8 mma.sync path keeps int32 accumulation, so the bitwise gate survives tensor-core prefill (P1). **RESOLVED for fp8 KV (P2, 2026-07-02):** the tolerance-gate machinery now exists and passed -- logit A/B vs the fp16 path (cosine 0.9995, top-1 exact, KL 3.4e-5 @512-tok prompt), corpus PPL delta -0.05%, needle 3/3 -- and fp8 ships opt-in (`Q27_KV=fp8`) with the fp16 default still bitwise-canonical. The same gate recipe applies to any future fp16/fp8 MMA decode path. **AMENDED for the g64 activation regroup (2026-07-04, policy sign-off):** batched-prefill activations now default to per-64 quantization (`Q27_PF_XG`, matching the Q4 weight group so two K=32 mmas chain in int32 before one fp dequant step). Per-64 amax changes the int8 values vs the decode path's per-32, so serial-vs-batched identity no longer holds BY DESIGN on the default path. Replacement gates: test_kernels g64-vs-exact (same quantized inputs through the dp4a exact path, rounding-noise bound), corpus PPL delta, canonical md5 (the canonical CLI run prefills serially and stays bitwise), scored-task spot-check. `Q27_PF_XG=32` restores the exact path and the `--pf` identity gate enforces it there.
