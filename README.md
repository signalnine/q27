# Quasar

A narrow inference engine for **Qwen3.6-27B-MTP** (hybrid GDN+attention, trained-in MTP heads) and its fine-tunes on a single RTX 5090 (3090 and 4090/Ada also supported). One model family, one GPU, as fast as possible. In the spirit of [antirez/ds4](https://github.com/antirez/ds4)

## Why this is interesting

- **Fastest known way to run this model.** +47% decode over tuned
  llama.cpp on a 5090 (same model, GPU, harness, day; protocol filed
  before it could pass). On a 24GB 3090: +19% decode at 2x the
  context over mainline llama.cpp (262K turbo3 default vs their
  100-131K class). vLLM measured 4.7x slower wall on real
  Claude-Code traffic (its prefix cache gets 0% reuse on this
  hybrid-GDN architecture); sglang 0.5.15 cannot load the model at all
  (BUILDLOG 2026-07-12).
- **turbo3 3-bit KV cache**, symmetric K+V: 14.1 KB/token (14400 B,
  18-pair accounting), ~1% PPL, needle 6/6 at a 361K-token prompt,
  655K context allocatable on a 5090, two full 131K tenants at once,
  and a 24GB card promoted from a 32K box to a **262K box** (turbo3 is
  the Ampere serving default since v0.3.0; a bare w8 boot auto-sizes
  to the full native window). Ported from
  [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant);
  that fork refuses 3-bit K on this model class and caps 33% lower; I measured K costs +0.17%.
- **Native Anthropic Messages endpoint at Claude-Code grade**: thinking
  blocks, tool_use with input_json_delta streaming, exact
  `count_tokens`, anthropic-shaped context-limit errors, billing-header
  normalization so the prefix cache survives real CC turns, and a
  tolerant tool-call parser (nine cataloged drift modes) that ANY
  engine on this harness needs in order to score. One env var points
  Claude Code at it; OpenAI and Codex (Responses) shapes ride the same
  binary.
- **Self-speculation as the whole design**: trained-in MTP ladder +
  free suffix drafter through one shared-KV MMA verify -- 5.3-5.8
  accepted tokens per weight read on live traffic (231-246 t/s
  aggregate on a 5090).
- **Continuous batching on top of it** (serving default since
  2026-07-16): concurrent slots decode through ONE fused weight sweep,
  whole fused-verify rounds replayed as shape-keyed CUDA graphs --
  2-slot aggregate **1.41x** over round-interleaving, solo cost
  <=0.07%, byte-identity gated, zero config.
- **Receipts for everything**: bitwise canonical gates, negative
  results logged at the same rate as wins, and every number in this
  README traceable to a dated BUILDLOG entry.

**Baseline model (2026-07-09): vanilla Qwen3.6-27B-MTP** (`qwen36-27b-mtp`,
canonical md5 `a2982c51...`) -- the benchmark standard: bench rigs and gate
scripts default to it. Fine-tunes stay fully supported (`MODEL=`/`TOK=`/
`CANON_MD5=` env overrides; Qwopus3.6-27B-v2-MTP canonical `4c4120c7...`).
Pre-07-09 historical numbers (see BUILDLOG) were measured on Qwopus
unless noted.

## Quickstart

Requirements: an NVIDIA GPU with 24GB+ VRAM, CUDA toolkit 12.4+ at
`/usr/local/cuda` (12.4 is a hard floor: older ptxas rejects the sm_89
e4m3 MMA forms), and gcc. `make` builds ONE tri-arch binary (sm_86 +
sm_89 + sm_120: 3090, 4090/Ada, and 5090 class); arch dispatch is at
runtime -- fp8-KV and the e4m3 MMA paths need sm_89+, Ampere (sm_86/80)
runs the fp16-MMA verify (h16) and fp16/turbo3 KV. (v0.3.1+
release binaries include the sm_89 target; v0.3.0 did not.)

24GB cards (3090-class): build `make build/q27-server-w8` as well --
`Q27_W_MAX=8` shrinks the fixed VRAM stack so the server fits; the
default width-12 build OOMs at graph setup on 24GB. turbo3 is the
default on Ampere (no env needed); prefer the q4s tier: its
2.27GB-smaller weight file goes straight to KV budget (~74K tokens/GB at turbo3 rates).
Cards with less than a true 24 GiB (A10-class cloud parts, ECC
reserve, decimal-GB VRAM) can also add `Q27_MAXD=4` (trims the graph
zoo ~280MB) and `Q27_SAMPLED=0` for greedy-only serving (skips the
sampled graph set, ~600MB on sm_86; temperature>0 requests get a 400).
Field-measured on a 22.6 GiB cloud A10 (issue #1, a31108a): q4s +
both knobs boots the FULL 262,144 native window; default weights
reach 102,400. The card must be otherwise idle: ~2.7GB of other
resident VRAM is the difference between boot and OOM.

3090 decode is power-sensitive: a fully-powered card (350-420W) runs
**~130 t/s** on short code-gen turns (q4s/w8, measured 126-150); a
200W-capped card gives roughly half that (issue #6). And `--ctx auto`
(or just omitting `--ctx`) sizes to measured free VRAM with an
arch-scaled safety margin -- ~254K on a 24GB Ampere card (q4s/turbo3)
with ~0.9 GB headroom, rather than sizing to the brim and OOMing at
`cudaGraphInstantiate` on cards that land below the 262K cap (issue #6).

Pick a quant first. Four tiers, one repo
([signalnine/Qwen3.6-27B-MTP-q27](https://huggingface.co/signalnine/Qwen3.6-27B-MTP-q27))
-- all serve identically,
they trade decode speed for model quality:

| tier | file | GPU | pick it when |
|---|---|---|---|
| **default** (5.25 bpw) | `qwen36-27b-mtp.q27` | 24GB+ | the reference tier: bitwise canonical `a2982c51`, the most measured configuration |
| q4s (4.55 bpw) | `qwen36-27b-mtp-q4s.q27` | 24GB+ | max context on small cards; 2.27GB more KV budget, +5% decode, and wikitext PPL measures 0.26% BETTER than default (single Q4 lm_head + Q4 residual writers; error cancellation is real) |
| q6 (6.0 bpw) | `qwen36-27b-mtp-q6.q27` | 32GB | +0.35% PPL off Q5_K_M for ~5% slower decode |
| q6k (6.8 bpw) | `qwen36-27b-mtp-q6k.q27` | 32GB | quality matching the best GGUFs of this model, ~10% slower decode |
| q8 (8.1 bpw) | `qwen36-27b-mtp-q8.q27` | 48GB+ | the near-lossless reference for big cards; PPL 7.9942 (better than default; q6/q6k's tuned promotions still edge it on wikitext -- error cancellation is non-monotonic in bits), and the acceptance-recovery tier: decode runs +26% over pure byte-scaling. Does not fit 32GB cards |

Task scores measure the same across tiers (q4s: fully validated --
PPL, suite, agentic NLL, needle, 18-run task dome, no deficit) -- the
quality tiers buy perplexity margin, not benchmark wins. When in doubt take the default.

```bash
# 1. tokenizer + your chosen tier from Hugging Face (Apache-2.0);
#    swap --include for the tier file you picked above
huggingface-cli download signalnine/Qwen3.6-27B-MTP-q27 \
  --include qwen36-27b-mtp.q27 qwen36-27b-mtp.tok CHECKSUMS.md5 \
  --local-dir models/qwen36-27b-mtp
# fine-tune variant: signalnine/Qwopus3.6-27B-v2-MTP-q27
# verify: (cd models/qwen36-27b-mtp && md5sum -c CHECKSUMS.md5 --ignore-missing)

# 2. build (CLI + server + test suites) -- or skip the toolchain and
#    grab prebuilt linux x86_64 binaries (sm_86/89/120 fatbin, CUDA
#    runtime statically linked; NEEDS NVIDIA driver r580+ -- on older
#    drivers build from source with your driver's toolkit, 12.4+ for
#    sm_89, 12.8+ for sm_120) from https://github.com/signalnine/q27/releases
git clone https://github.com/signalnine/q27 && cd q27
make

# 3. smoke test the CLI (should print 128 tokens; md5 of the output
#    line is the bitwise canonical a2982c51...)
./build/q27 ../models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
  --tokens "760,6511,314,9338,369" -n 128 --ctx 2048 --spec

# 4. serve -- zero config; defaults resolve the full measured stack
#    and --ctx auto-sizes to your VRAM (see Serving for escapes).
#    Binds 127.0.0.1 only. To reach it from other machines or from
#    containers (Claude Code in docker resolves the host via the
#    bridge, not loopback), opt in explicitly -- and set an API key
#    at the same time, since --host 0.0.0.0 with no key accepts
#    unauthenticated requests from anyone who can reach the port:
#      --host 0.0.0.0 --api-key <your-key>
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
errors so Claude Code compacts correctly. Concurrent sessions
(`--slots N`) batch through one weight sweep by default -- 234-239 t/s
aggregate at 2 slots, zero config (see Serving).

## State of the engine (2026-07-19)

One binary serves Claude Code, Codex, and OpenAI clients at 231-246 t/s
aggregate live decode on a 5090 (90-116 t/s at 262K context on a 3090, turbo3 default),
with continuous batching ON by default: two concurrent slots decode
through one fused weight sweep plus shape-keyed CUDA-graph round replay
at **1.41x** aggregate (fp8 237.7 / turbo3 224.2 t/s), solo cost
<=0.07%, byte-identity gated. The 07-14..16 batching campaign got there
in three solo-neutral phases (FIFO 1.00x -> fused verify 1.21x -> fused
draft steps 1.31x -> graph replay 1.41x) and CLOSED on a measured P4
NO-GO; its physics triad: weight-BW-bound work wants FUSION,
state-latency-bound work wants OVERLAP, saturated work wants neither.
Single-stream, the 07-13 pass (k_vgemm flat-in-W verify GEMM, GEMV
occupancy retiers, GDN delta-step fusion) holds the short-bench suite
at 177.4 t/s. Full chronology, per-phase receipts, and every negative:
[docs/BUILDLOG.md](docs/BUILDLOG.md) (the 2026-07-13..16 entries).

Short cold prompts (<=128 tokens, the stateless single-shot / first-turn
case) get a prefill split-K that fills the SMs the small-output GEMMs
leave idle: ~8% faster prefill (server-measured, 62.4->57.4ms at 70
tokens), on by default since v0.3.3. The agentic NLL gate cleared it
(+0.018% on the full 154K CC corpus, worst segment +0.063%, vs a >+2%
bar); it auto-disables once the grid saturates, and every canonical
stays bitwise (the CLI eager-forwards, only the server path splits).
`Q27_GEMM_SPLITK=0` opts out. Warm agentic turns don't see it -- the
checkpoint already skips their prefill. Deep cold prefill (the saturated
large-T grid) gets a complementary lever since v0.3.4: an ntx M-minitile
GEMM that shares one activation `ldmatrix` load across two row-tiles
(+3.4% GEMM / ~6% cold-prefill wall, bitwise, sm_120; `Q27_PF_NTX=0`
opts out). fp4/`tcgen05` is a hardware dead end here -- consumer/
workstation Blackwell has no fp4 MMA in PTX, so int8 is the ceiling.

### Reference numbers (v0.2.0, 2026-07-16, vanilla model, 5090)

Measured at `c0c5c5e` unless dated otherwise; full tables and history
in [docs/BUILDLOG.md](docs/BUILDLOG.md) and
[docs/BENCHMARKING.md](docs/BENCHMARKING.md).

- Short-bench suite **177.4 t/s** (fp16 stock CLI, 5-prompt mean;
  canonical `a2982c51` EXACT, stock clocks).
- 2-slot continuous-batching aggregate (`tools/batch_ab.sh`): fp8
  168.9 -> **237.7 t/s (1.41x)**, turbo3 158.5 -> **224.2 (1.41x)**
  over the FIFO baseline; solo regression <=0.07%. Zero-config spot
  check **234-239 t/s** aggregate (90.9% graph-cache hits).
- Decode @26K (server replay, fp8 basin): classic config 143.0 /
  **full default stack 176.3 t/s** (+23%). Echo ceiling (repetitive
  traffic, wide suffix): 26K zero-config server **400.6 t/s**.
- Live Claude-Code traffic (07-10, 9 scored task trials, 430
  requests): **231.3 t/s aggregate**, per-request median 225 / p75 277
  / peak 378 t/s. Qwopus fine-tune: **+5.7% decode at quality TIE**
  (246.5 vs 233.1 t/s aggregate, 5.65 vs 5.31 tok/rnd; matched 21-task
  sweep, 07-11).
- Prefill (fp8 batched TTFT): 8K 2.35s | 32K 10.4s | 128K 59.4s (~2200 t/s).
- Cross-engine: **+47% decode vs tuned llama.cpp's best config** (07-10
  protocol run; ~15 points of the 47 are bit-width, the rest is
  mechanism) and 202.7 vs 117.1 t/s vs vLLM on public SWE-bench agentic
  tasks -- see Benchmarks. llama's ngram spec still wins pure file
  re-emission, a mode mutually exclusive with its production draft-mtp
  config (decomposition: BUILDLOG 07-10/07-13).
- 3090 (24GB, turbo3 + h16, 07-12): **102.2 t/s median** live CC
  decode at **131K ctx** -- +19% decode at +60% context over mainline
  llama.cpp's strongest vanilla config.
- Weight tiers (07-12): q6k's matched-protocol PPL 7.9127 beats every
  measured GGUF of this model incl. unsloth's 26 GB flagship. On fp8 KV
  the quality tiers cost context (262144 / 192512 / 122880 auto-ctx for
  default/q6/q6k on a 32GB 5090, re-measured 07-17 under the calibrated
  auto-ctx) -- pair them with `Q27_KV=turbo3`, which keeps the full
  262144 window on every tier.
- q4s tier (07-16): 15.46 GB / 4.55 bpw. Paired-protocol PPL 8.0197
  vs default's 8.0409 (-0.26%, the third measured error-cancellation
  win), suite +5.2% (186.2 vs 177.0 t/s same-day) -- smaller, faster,
  AND lower perplexity. Exists for VRAM-starved cards: the 2.27 GB it
  returns is ~167K tokens of turbo3 KV budget. Field-measured on an
  A10 (22.6 GiB usable, issue #1): the arc ran 28,672 stock ->
  49,152 with `Q27_MAXD=4` -> 212,992 on q4s -> **262,144** (the full
  native window) at a31108a with the v0.3.0 capture gates
  (`Q27_SAMPLED=0`); default weights reach 102,400 on the same card.

### Carried state (pre-campaign, still in force)

- Width-12 verify (07-10): widths 9..12 belong to the suffix drafter
  (live agentic AL 10.6 on ~62% of decode); byte-identical at old
  widths by construction. W_MAX stays 12 -- W16 measured as a
  per-token LOSS, reopening only for file-re-emission traffic
  (`q27-server-w16`; BUILDLOG 07-13).
- The CLI stays fp16/reference so the bitwise canonicals hold; the
  server's fp8/mma stack is tolerance-class by policy (see Serving).
  Long-context validated on both compact KVs: needle 6/6 at 361K;
  allocation ceilings fp8 294,912 / turbo3 655,360 (W12, 5090).
- Tolerant tool-call parser: nine cataloged drift modes + the 07-11
  inference tie-break; every engine on this harness depends on it
  (strict parsing scores 0.000 on the hardest task class; BUILDLOG
  07-08). `--constrain-tools` stays opt-in (in-call cost 3.1x at depth).
- Multi-slot serving (`--slots N`): batching default since 07-16;
  `Q27_BATCH=0` restores R1b round-granularity time-slicing (the
  measured FIFO baseline). P8 stable-prefix snapshot (warm turns ~1.3s)
  + P9 same-session checkpoint ring own the prefill side.
- Measured NO-GOs with do-not-retry bars (deep-MTP ladder, GDN chunk,
  fdmma orchestration variants, W16 cap, P4 mixer co-residency,
  prefill FA2 relayout, draft-head shortlist): receipts in
  [docs/BUILDLOG.md](docs/BUILDLOG.md); parked levers in
  [docs/notes.md](docs/notes.md).

## Why this model is a good target

- Dense-ish 27B that fits entirely in 32 GB VRAM at 4-bit -- no expert offload, no DRAM scatter, none of the DSV4 pain
- MTP draft head trained into the checkpoint: self-speculation without a separate draft model
- Hybrid Gated-DeltaNet: near-O(1) memory per token for 48 of 65 layers.
  KV lives only in the 17 full-attention layers (16 + MTP, all global, no
  windowing): 72 KB/token fp16 (73728 B; the engine
  allocates 18 K/V pairs -- 17 attention layers + the MTP head). A dense 65-layer build would need ~68 GB
  @256K.
- Measured 5090 KV ceilings: fp16 ~180K | fp8 (36 KB/tok) **294,912** |
  turbo3 3-bit (14.1 KB/tok) **655,360** -- 2.5x the 262K native window.
  Auto-ctx caps at 262144 for fp8/turbo3, 131072 for fp16; explicit
  `--ctx` overrides. turbo3 position-bucket NLL is flat through 297K
  (tracks fp8 within +0.65-1.2% every bucket); the agentic quality gate
  closed PASS 07-16 (within +0.39% at CC depths on a real 154K CC
  transcript; shape-matched CC scores tie).
- The catch the per-token-memory napkin misses: attention KV is RESTORABLE state (any prefix row range replays for free) while GDN recurrent state is all-or-nothing per sequence -- you can only resume from a position you snapshotted. Hybrids make per-user context cheap but make context REUSE an engineering problem (prefix cache, mid-history divergence, multi-doc serving). That trade is where P8/checkpoint work lives; the measured cost of ignoring it was 7.9x wall-clock on agentic traffic (see build log P8/P9)
- The opponent, tuned honestly: llama.cpp's best measured config on this
  box is Q5_K_M + draft-mtp10 + p_min 0.5 (**~117 t/s @2K** single-stream;
  the win over stock is p_min, not draft depth -- swept 07-06). All
  cross-engine numbers use that config; see Reference numbers.

## Why paged-KV engines can't cache this model -- and how q27 turns that into wall time

vLLM's serving story is PagedAttention: KV memory is a global pool of
16-token blocks and the prefix cache shares blocks by content hash. On a
pure-attention transformer that is close to a free lunch -- attention KV
is an append-only, position-addressed log, so any cached prefix block
replays for free.

Hybrid GDN breaks the assumption the lunch depends on. 48 of this model's
65 layers carry no KV at all; their state is a dense recurrent summary
(128x128 per head + a conv ring) that REPLACES the token log. That state is
order-dependent and all-or-nothing: it cannot be paged, cannot be shared by
hash, and cannot be reconstructed from any cached block -- only replayed
from position 0 or restored from a snapshot you took yourself. A block
cache covers 17/65 layers; without the matching GDN state those blocks are
dead weight. Measured consequence (SWE-bench agentic, 07-15): vLLM's
prefix cache got **0% reuse** and re-prefilled every turn -- competitive
decode (117 t/s) but the WORST wall time of all five engines tested
(133 s/instance).

q27 treats the GDN summary as a first-class object instead of a cache miss:

- **P8 stable-prefix snapshot**: one device-side snapshot of all 48 GDN
  states at the last ChatML-stable boundary, plus split-encode at that
  boundary so tokenization itself is prefix-stable across turns.
- **P9 checkpoint ring**: pinned-host copies every 4096 tokens during
  prefill, so mid-history divergence rewinds to the nearest checkpoint
  instead of position 0.
- Attention KV needs neither: rows below the divergence point are
  append-only and stay valid in place.

A warm CC turn is therefore restore + suffix-only prefill -- real traffic
looks like `prompt=25473 hit=24136 pf=1337` in the `[req]` log, ~1.3 s
instead of a 10-20 s full re-prefill at p50 agentic depth. That arithmetic,
times every turn of a 30-90-turn trajectory, is the whole wall-time story:
**q27 47 s/instance vs vLLM 133 s** on identical tasks, with decode speed
(203 vs 117 t/s) explaining less than half the gap. The continuous-batching
stack (07-14..16) is independent of this machinery and stacks on top:
snapshots own prefill, batching owns decode.

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
fp8/turbo3 KV) or (b) more accepted positions per weight read (ladder,
suffix width) -- at batch 1 there is no third lever. The 07-14..16
continuous-batching campaign is lever (b) pointed across users --
concurrent slots SHARE the weight read (2-slot aggregate 1.41x) while
the per-stream arithmetic above still holds inside each fused round
(docs/multislot-throughput.md).

### Decode methodology (canonical, 2026-07-02)

These numbers are NOT interchangeable -- each answers a different question:

- **Short-bench suite** (SOTA-comparable): 5 fixed genre-diverse short
  prompts x 128 tokens, `--spec`, STOCK clocks -- `tools/shortbench_suite.sh`.
  **Current (v0.2.0, vanilla baseline): 177.4 t/s mean** (per-prompt
  170.7-185.5; series history in BUILDLOG). The per-prompt spread is
  trajectory/acceptance variance -- no single short prompt may carry a
  cross-engine number. It is also the param/launch-overhead CANARY: it
  caught the width-12 param-copy regression the depth gates missed.
- **Canonical prompt** (bitwise gate, NOT a benchmark): 128 tokens from the
  5-token canonical prompt -- vanilla baseline md5 `a2982c51...` (the
  standard; 144.2 t/s, 2.61 t/round at v0.2.0), Qwopus `4c4120c7...` for
  fine-tune gating. Held bitwise through every default-path kernel
  change since fd2 (the full list is in BUILDLOG). Tie-lottery
  sensitivity is why it gates bitwise identity and nothing else. It
  gates the CLI's reference defaults; the server's CC defaults are
  deliberately tolerance-class (fp8+mma) -- `Q27_PROFILE=ref` restores
  reference behavior there.
- **Tie-lottery methodology** (the project's most subtle measurement
  concept): tolerance-class numerics changes (fp8 paths, mma, split-count)
  re-roll greedy argmax ties -- **neutral in expectation, deterministic
  per build**. A quality flip on ONE benchmark basin is read via a basin
  MATRIX across tasks plus a re-roll on the next binary, never a single
  retrial (the mma case study is in BUILDLOG 07-10).
  Acceptance-sensitive decisions must name their basin; cross-BUILD
  text comparisons are invalid (same-binary legs only).
- **Depth numbers**: the current predictors are the Reference-numbers
  block (cctx 26K 143.0/176.3, live CC 213-246 aggregate); the 07-08-era
  61K/74K series and the 2K-soak long-generation series live in BUILDLOG
  with their cross-era bridges. Depth numbers, not 2K numbers, predict
  agentic wall time.

OC policy: headline + SOTA comparisons are reported STOCK (community numbers
aren't OC'd; sidesteps the non-ECC tail-risk conversation). +3000 stays a
supervised-bench option (+2.3% short-bench measured); the weight-checksum
tool (`--verify-weights`, `/health?verify=1`) exists for OC sessions. The
soft-error incident that set this policy is in the BUILDLOG appendix.

## Design decisions

- **Weights**: custom 4-bit symmetric groupwise (group 64, fp16 scales), packed for coalesced 128B warp loads, dequant fused into GEMV. Embeddings, lm_head, MTP layer, norms at 8-bit/f32. Repacked offline from the BF16 GGUF (container spec: docs/FORMAT.md).
- **KV cache**: fp16 by default; fp8 E4M3 is the server default since
  07-03 (`Q27_KV=fp8` on the CLI). Scale-free saturating conversion --
  measured amax sits 3.8x under the 448 E4M3 max, so per-row scales buy
  nothing. 36 KB/token, +11% decode @28.5K, PPL -0.05%, KL 3.4e-5. The
  CLI stays fp16 so the bitwise canonicals hold.
- **turbo3 3-bit KV** (`Q27_KV=turbo3`; `turbo3v` = fp16-K diagnostic):
  WHT-rotated 50-byte blocks per 128 dims (ported from
  [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant),
  src/turbo3.cuh), 14.1 KB/token, full stack (decode, fdmma verify,
  batched prefill). PPL fp16 7.317 / fp8 7.327 / turbo3 7.381; K costs
  +0.17% -- the GQA=6 K-crater the source fork guards against does not
  exist on this model. Acceptance ties fp8 exactly on basin-matched
  replay.
- **MTP**: first-class. Draft + verify in one pipeline under a single CUDA graph. No separate draft context, no re-prefill.
- **Stack**: plain CUDA C++. No CUTLASS, no deps beyond CUDA runtime. Offline tools are Python: tools/repack.py (runs once; docs/FORMAT.md) and tools/gguf_to_hf.py (certified GGUF -> HF inversion, 866/866 tensors byte-exact, for cross-engine reference runs).
- **Serving**: OpenAI, Anthropic (Claude Code-grade), and OpenAI Responses (Codex-grade) shapes on one binary. Since 2026-07-03 the SERVER defaults to fp8 KV on sm_89+ (and to
turbo3 on sm_86 since v0.3.0) (--kv-fp16 or Q27_KV=fp16 opts out); the CLI keeps fp16 so decode canonicals stay bitwise.
- **Numerics contracts (batching)**: every fused lane family carries
  the ninv N-invariance gate -- **bitwise-when-untrimmed** -- plus its
  seam and twin legs; fused rounds run a UNION GEMM-family policy so
  batched numerics match solo; canonical + sampled-seed EXACT at every
  merge. The only text forks are the documented tolerance classes (A1
  suffix-trim, turbo3 concurrency tie re-rolls).
- **Two-tier batching guard**: user-EXPLICIT `Q27_BATCH=1` plus an
  incompatible env stays fail-fast FATAL; profile-DEFAULT plus an
  incompatible env auto-disables with one banner line and serves
  exactly as pre-batching (a default must never kill a
  formerly-working invocation).

## Serving

```
make build/q27-server
./build/q27-server model.q27 model.tok --port 8080
```

**Defaults (2026-07-16) = the measured Claude-Code stack.** A bare server
serves the exact config every live trial and record number was earned on:
fp8 KV + `Q27_FD=mma` (e4m3 on sm_89+, fp16-MMA h16 on sm_80..88; fp8 KV
itself needs sm_89+; sm_86/Ampere defaults to turbo3 since v0.3.0 --
a bare w8 boot serves the full 262144 window on a 24GB 3090 -- with
fp16 via `--kv-fp16`), `Q27_PMIN=0.5`,
`Q27_MAXD=auto7`, suffix drafter at width 12, fast-head, no-think, phase
stats; `--ctx` auto-sizes the KV budget to free VRAM (capped at the
262144 native window for fp8/turbo3, 131072 fp16; single-slot).
Continuous batching is default since 2026-07-16 (`Q27_BATCH=1
Q27_BATCH_GRAPH=1`, graph-cache cap 64, shrunk to fit VRAM headroom;
`Q27_BATCH=0` disables; single-slot/solo traffic is byte-identical to
pre-batch). Every knob keeps its env/flag override (user env always
wins), `Q27_PROFILE=ref` restores the conservative reference behavior
(fp16, ungated, no suffix, fd2, no batching), and the **CLI binary keeps
reference defaults** so the bitwise canonical gates are untouched.
Escapes: `--kv-fp16 --no-fast-head --think`, any individual `Q27_*`.

`--slots N` auto-sizes too (since 2026-07-18): with `--ctx` omitted the
free-VRAM budget is split across the N co-resident engines and every slot
gets the same computed window (logged `--ctx auto: <ctx> per slot`). Pass an
explicit `--ctx` to set slot 0 by hand and `--slot1-ctx` for the background
slots.

**Auth.** Off by default -- loopback-only binding is the actual safety net
(see `docs/SECURITY-MODEL.md`); this is a convenience for the cases that
doc's own recommendation (put a real reverse proxy in front) is overkill
for, not a replacement for it under real multi-tenant/production exposure.
`--api-key KEY` (repeatable), `--api-key-file PATH` (one key per line, `#`
comments ignored), and `Q27_API_KEY` (env -- preferred in containers, where
CLI args are visible via `ps` but orchestrator-injected env vars are not)
all add keys; any of them is accepted. Every endpoint except `/health`
requires one once at least one key is configured. Both header conventions
work, so neither client family needs special handling:
`Authorization: Bearer <key>` (set via `OPENAI_API_KEY` for OpenAI-compatible
clients, or Codex's `env_key` in `~/.codex/config.toml`) or `x-api-key: <key>`
(set via `ANTHROPIC_API_KEY` for `claude` / Claude Code). Binding non-loopback
with no key configured prints a warning at boot but is not refused --
some deployments intentionally terminate auth at their own reverse proxy.

Behavior note (thinking): the default profile is no-think for speed -- it
prefills an empty `<think></think>` block so the model answers directly.
`--think` flips the server default the other way (prefills an open `<think>`
so the model reasons in a real block, closed with `</think>`, before it
answers).

Either default is overridable **per request** -- the server profile just sets
the default a request can override in either direction, via any of:
`enable_thinking: <bool>` (OpenAI/Qwen top-level), nested
`chat_template_kwargs.enable_thinking` (llama.cpp/GLM), or
`thinking: {"type": "enabled"|"disabled"}` (Anthropic -- Claude Code's own
toggle). Thinking-on routes the reasoning trace to `reasoning_content`
(OpenAI) / a `thinking` content block (Anthropic), never into the answer text.
Give a thinking request enough `max_tokens` to cover the trace **and** the
answer -- a tight budget is spent entirely on reasoning and truncates the
answer. When a client omits `max_tokens` the server defaults it to 8192
(unified across all three API shapes, clamped to the context window); a long
thinking trace wants more, set it explicitly.

A reasoning model handed zero reasoning budget over-refuses a narrow class of
borderline requests; mitigated 2026-07-13 by injecting a minimal default system
prompt when the client sends none (never fires for real Claude Code;
`Q27_BARE=1` opts out). For compliance-sensitive workloads default-on `--think`
remains the stronger lever (BUILDLOG 2026-07-13).

Three API shapes on one server:
- **OpenAI**: `/v1/chat/completions`, `/v1/completions` (text)
- **Anthropic**: `/v1/messages` -- native Messages API with thinking
  blocks, tool_use/tool_result, input_json_delta streaming, exact
  `/v1/messages/count_tokens`, an anthropic-shaped context-limit error
  (400) so Claude Code compacts instead of retry-looping, and cch
  billing-header normalization that keeps the prefix cache warm across
  CC turns. Claude Code-compatible:
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
(src/stream_split.h) that also routes `<think>`. The tolerant tool-call
parser recovers nine observed drift modes, logging each recovery for the
drift catalog. Greedy (spec decode) by default; `temperature>0` routes to
sampled SPEC decode -- top-p nucleus + Gumbel-max with rejection-sampled
spec acceptance at spec speed, seeded and reproducible, greedy left
bitwise-unchanged (docs/sampling-design.md). The exit-gate A/B passed
(docs/sampling-exit-gate.md), so the server can default sampling on for
clients that send no temperature via `Q27_FORCE_TEMP`/`Q27_FORCE_TOP_P`
(an explicit request temperature still wins). `--fast-head` trades output
exactness for ~7% more t/s.

Confidence-gated depth (P12 + P14): `Q27_PMIN=theta` caps verify width on
the drafter's top1-top2 margin; `Q27_DEXIT` stops drafting at the first
sub-theta margin. The adaptive 4..7 ladder lives in src/depthctl.h
(thresholds and knobs documented there); measured +2.7% geomean over
d4-gated, +4.2% over fixed-d5 on real-CC replay. Greedy output is
bitwise-identical under gating at every ceiling -- round segmentation
varies, tokens never do. The CLI and `Q27_PROFILE=ref` leave it all unset.

## Benchmarks

Cross-engine comparison against llama.cpp (mainline, and TheTom's `ngram-mod`
fork) on a **public, reproducible** agentic task set: 12 pinned
SWE-bench_Verified instances driven through Claude Code, the **same**
Qwen3.6-27B-MTP model on every engine, all 5090-only + q8 KV + greedy. Real
agentic decode throughput:

| engine | decode | wall/inst |
|---|---|---|
| **q27** (MTP + SuffixDraft, fused) | **202.7 t/s** | **47 s** |
| vLLM NVFP4 + MTP | 117.1 t/s | 133 s |
| llama mainline + MTP (`--spec-type draft-mtp`) | 116.3 t/s | 80 s |
| llama `ngram-mod` (fork) | 61.1 t/s | 118 s |
| llama mainline (no spec) | 62.0 t/s | 120 s |

With the *same* MTP head, enabling MTP nearly doubles stock llama.cpp, and
**two independent engines (llama.cpp and vLLM) land on the same ~117 t/s**
-- yet q27 is a further ~1.73x on top: the residual is the engine, not the
drafter choice. ngram-mod adds ~nothing on real coding; vLLM's wall/inst
is worst because its prefix caching is dead on this hybrid-GDN arch.
Quality converged to the model across engines (11-12/12
edited-gold-file; both tool protocols validated first -- unvalidated
tool parsing is how engines DO move quality).

Full methodology, fairness controls, the payload microbench, and reproduce
steps: [docs/BENCHMARKING.md](docs/BENCHMARKING.md). Harness, pinned task set,
and raw per-instance results: [bench/swebench/](bench/swebench/).

## Open items (2026-07-16)

- ~~**turbo3-vs-fp8 quality gate**~~ **CLOSED 2026-07-16, verdict PASS**
  (BUILDLOG "TURBO3 AGENTIC QUALITY GATE"): agentic-corpus NLL on a real
  154K-token CC transcript is within +0.39% of fp8 in every CC-depth
  bucket, and the shape-matched CC study (2x48K both legs, n=3/leg)
  ties/favors turbo3 -- the 07-15 band gap was the shape confound. No
  quality asterisk on turbo3; fp8 stays the sm_89+ CC serving default
  on speed alone -- turbo3 is the capacity lever there (2x96K vs 2x48K
  on 32GB) and, since v0.3.0, the outright default on Ampere.
- **Saguaro-style 3090 off-path drafting** -- the one uncommissioned
  engine idea left standing from 07-10.
- **Prefill follow-ons**: the retired Phase-3's filed successor (async
  producer/consumer + mbarrier rewrite of the prefill-attention
  kernel). (The serial-threshold call shipped in v0.3.0 as
  `Q27_PF_BATCH_MIN` -- TTFT 350->31-33ms / 567->53-55ms; the CLI
  default is unchanged, so the canonical still holds.)
- **Strict-parser zero-rescue config**: engage the constrain grammar on
  a bare `{"name"` opener too, closing the wrapper-less bypass
  (strict-parser A/B verdict: BUILDLOG 2026-07-08).
- **Graph-cache cap under churn**: live CC already draws 44+ keys vs
  the bench's 28; cap 64 swallows today's alphabet, revisit
  `Q27_BATCH_GRAPH_CAP` if multi-tenant composition churn widens it.

Measured-and-parked levers (chunked-WY delta scan, cross-session
checkpoint pool, AWQ-style scales, P11 split path, and friends):
[docs/notes.md](docs/notes.md).

## History

The full chronological record -- every DONE block with its numbers,
every negative result, the early milestones (M0..E6), the progress-log
table (43.4 -> 177.4 t/s), and the M6 prefill history -- lives in
[docs/BUILDLOG.md](docs/BUILDLOG.md). The BUILDLOG is the ledger, not
git history (commits sometimes batch a day's entries). Design docs and
phase plans: [docs/plans/](docs/plans/). Standing risk register, P10-A
status, and parked levers: [docs/notes.md](docs/notes.md). Multi-slot
throughput analysis (rewritten post-campaign):
[docs/multislot-throughput.md](docs/multislot-throughput.md).

## License

MIT -- see [LICENSE](LICENSE).
