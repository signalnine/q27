# Quasar

A narrow inference engine for **Qwen3.6-27B-MTP and Qwen3.8-27B-MTP** (hybrid GDN+attention, trained-in MTP heads) and their fine-tunes on a single RTX 5090 (3090 and 4090/Ada also supported; Apple-silicon Metal backend for the q4s tier). One model family, one GPU, as fast as possible. In the spirit of [antirez/ds4](https://github.com/antirez/ds4).

## Why this is interesting

- **Fastest of the four engines tested, on this harness, across two independent
  runs.** Mean wall per SWE-bench instance, same 12 tasks, one harness,
  unchanged competitor binaries:

  | leg | 2026-08-17 | 2026-08-19 |
  |---|--:|--:|
  | **q27** q5f | 46.8 s | **46.3 s** |
  | **q27** q4s | 48.5 s | 49.6 s |
  | llama.cpp | 71.4 s | 59.9 s |
  | vLLM | 84.5 s | 78.8 s |
  | ninfer NVFP4 | 96.8 s | 113.8 s |
  | ninfer int8 | 327.4 s | 266.2 s |

  q27 is the only leg that reproduced (46.8 -> 46.3 s); every competitor moved
  7-19% on identical binaries, so the ordering is stable and the margins are
  not. n=1 per instance, sequential legs, each engine's own sampling defaults.
  Caveats: [FINDINGS.md](bench/crossengine/FINDINGS.md#6-caveats). The win is
  prefix reuse, not raw decode -- ninfer decodes faster and still takes 2-7x
  the wall time. **Where q27 loses:** ninfer's NVFP4 peaks 1.57x higher at 8
  concurrent streams (834 vs 531 t/s). Logged at the same rate as the wins.
  *(Dated 2026-09-06: ninfer shipped an agent prefix-reuse fix; on the same
  artifact and harness, current master measures 51 s/inst at 91.3% reuse --
  [bench/crossengine/NINFER-REBENCH.md](bench/crossengine/NINFER-REBENCH.md).
  Dated 2026-09-09: with both engines on DFlash2 drafters and ~97% reuse the
  wall ordering has flipped -- ninfer 36 s/inst, q27 108 s -- at decode
  rates within 6%; q27's sessions run 25 turns and 18K output tokens per
  instance against ninfer's 15 and 6K, and why is the open question:
  [bench/crossengine/agentic-2026-09-09/](bench/crossengine/agentic-2026-09-09/README.md).)*
- **Self-speculation as the whole design**: trained-in MTP ladder + free
  suffix drafter through one shared-KV MMA verify -- 5.3-5.8 accepted tokens
  per weight read on live traffic (231-246 t/s aggregate on a 5090).
- **Continuous batching on top of it** (default since 2026-07-16): concurrent
  slots decode through one fused weight sweep, rounds replayed as shape-keyed
  CUDA graphs -- 2-slot aggregate **1.41x**, solo cost <=0.07%, byte-identity
  gated, zero config.
- **A 24GB card becomes a 262K box**: turbo3 3-bit KV (14.1 KB/token, needle
  6/6 at 361K) and turbo5k 5-bit-K (the Ampere default) are capacity levers a
  32K-at-fp16 card cashes directly. Ported from
  [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant),
  then taken past the fork's own limits.
- **Native Anthropic Messages endpoint at Claude-Code grade**: thinking
  blocks, tool_use with input_json_delta streaming, exact `count_tokens`,
  anthropic-shaped context-limit errors, billing-header normalization so the
  prefix cache survives real CC turns, and a tolerant tool-call parser any
  engine on this harness needs in order to score. One env var points Claude
  Code at it; OpenAI and Codex (Responses) shapes ride the same binary.
- **Receipts for everything**: bitwise canonical gates, negative results
  logged at the same rate as wins, every number here traceable to a dated
  [BUILDLOG](docs/BUILDLOG.md) entry.

**Baseline model: vanilla Qwen3.6-27B-MTP** (canonical md5 `a2982c51...`) --
bench rigs and gate scripts default to it. Fine-tunes stay supported
(`MODEL=`/`TOK=`/`CANON_MD5=` overrides). Qwen3.8 tiers carry their own
canonicals in the HF model card.

## Quickstart

Requirements: an NVIDIA GPU with 24GB+ VRAM, CUDA toolkit 12.8+ at
`/usr/local/cuda`, gcc. `make` builds one tri-arch binary (sm_86 + sm_89 +
sm_120; arch dispatch at runtime). 12.8 is the floor on every card: the build
links an `sm_120a` object unconditionally. Prebuilt linux x86_64 binaries
(CUDA statically linked, driver r580+) are on the
[releases page](https://github.com/signalnine/q27/releases).

On 24GB cards (3090-class): build `make build/q27-server-w8` -- the default
width-12 build OOMs at graph setup. Prefer the q4s tier; its 2.27GB goes
straight to KV budget. Sub-24GiB cards (A10-class) add `Q27_MAXD=4` and
`Q27_SAMPLED=0`; field-measured on a 22.6 GiB A10, that boots the full 262,144
native window (issue #1). A power-capped 3090 decodes at roughly half the
350-420W figure (issue #6). `--ctx` auto-sizes to measured free VRAM.

Pick a quant. Seven 3.6 tiers, one repo
([signalnine/Qwen3.6-27B-MTP-q27](https://huggingface.co/signalnine/Qwen3.6-27B-MTP-q27)),
all serving identically:

| tier | bpw | GPU | pick it when |
|---|--:|---|---|
| **default** | 5.25 | 24GB+ | the reference tier: bitwise canonical, most measured |
| q4s | 4.55 | 24GB+ | max context on small cards; +5% decode, PPL 0.26% BETTER than default (error cancellation is real) |
| q5f | 5.30 | 24GB+ | best quality that fits 24GB: PPL 7.9491 matches q6 at 2.3GB less |
| q6 | 6.0 | 32GB | superseded by q6f at the same size |
| q6f | 6.11 | 32GB | the 32GB pick: PPL 7.9189 nearly matches q6k at 2.25GB less |
| q6k | 6.8 | 32GB | matches the best GGUFs of this model, ~10% slower decode |
| q8 | 8.1 | 48GB+ | near-lossless reference; does not fit 32GB |

Task scores measure the same across tiers -- the quality tiers buy perplexity
margin, not benchmark wins. When in doubt take the default.

### Qwen3.8-27B (v2 recipes)

A repack-only port ([docs/PORTING.md](docs/PORTING.md) predicted it from the
config, and the prediction held), but the quant recipes were re-derived from a
fresh per-tensor sensitivity sweep -- the 3.6 recipes measurably do not
transfer (the `ssm_out` promotion they carry compounds through the GDN
recurrence at depth). Tiers at
[signalnine/Qwen3.8-27B-MTP-q27](https://huggingface.co/signalnine/Qwen3.8-27B-MTP-q27):

| tier | GB | wikitext PPL | HumanEval+ | needle |
|---|--:|--:|--:|---|
| q4s (v2) | 15.70 | 7.3765 | 30/30 | 6/6 @ ~120K |
| **default (v2)** | 17.00 | 7.3121 | 30/30 | 6/6 @ ~120K |
| q6 (v2) | 19.76 | 7.2233 | 28/30 | 6/6 @ ~100K |
| q6k (v2) | 22.52 | 7.1718 | 29/30 | 6/6 @ ~40K |

Serve 3.8 for agentic use with `--think` and the card sampler:
`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0`. Under that
recipe the full 19-task suite scores **0.928 hidden / 0.895 composite**
(BUILDLOG 2026-08-25); under the old defaults (greedy, 16K think budget) the
same suite scored 0.511 -- the recipe is the difference, not the checkpoint.
The engine auto-selects 3.8's trained XML tool dialect from the artifact name.

```bash
# 1. tokenizer + your chosen tier (Apache-2.0)
huggingface-cli download signalnine/Qwen3.6-27B-MTP-q27 \
  --include qwen36-27b-mtp.q27 qwen36-27b-mtp.tok CHECKSUMS.md5 \
  --local-dir models/qwen36-27b-mtp
# verify: (cd models/qwen36-27b-mtp && md5sum -c CHECKSUMS.md5 --ignore-missing)

# 2. build (CLI + server + test suites)
git clone https://github.com/signalnine/q27 && cd q27
make

# 3. smoke test (128 tokens; output md5 = canonical a2982c51...)
./build/q27 ../models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
  --tokens "760,6511,314,9338,369" -n 128 --ctx 2048 --spec

# 4. serve -- zero config resolves the full measured stack; binds
#    127.0.0.1 only. Reaching it from containers or other machines is an
#    explicit opt-in, and wants a key: --host 0.0.0.0 --api-key <key>
./build/q27-server ../models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
  ../models/qwen36-27b-mtp/qwen36-27b-mtp.tok --port 8080
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

Expect ~170-230 t/s decode on a 5090 depending on traffic shape, warm
multi-turn prefills from the prefix cache, and `count_tokens` plus
anthropic-shaped context-limit errors so Claude Code compacts correctly.

## State of the engine (2026-09-09)

One binary serves Claude Code, Codex, and OpenAI clients on a 5090 with a
DFlash2 block drafter (K=7, MMA verify) as the production decode path, a
persistent prefix cache that hits on real agentic traffic, and a tool-call
parser measured against a labelled corpus of the model's own drift. Current
release: [v0.11.0](https://github.com/signalnine/q27/releases).

Headline numbers, each dated in the BUILDLOG and in the campaign READMEs
under [bench/crossengine/](bench/crossengine/):

- Claude Code traffic, 12 SWE-bench instances, medium effort (the only
  level both engines render): **q27 207 t/s** aggregate decode (219 median,
  3.89 tok/round, 96.9% prefix reuse) vs ninfer's DFlash2 arm 221 (240,
  4.19, 96.8%). Wall per instance 108 s vs 36 s: q27's sessions run 25 turns
  and 18K output tokens per instance against 15 and 6K -- the open question
  of this release (09-09).
- Production at Claude Code's default effort (xhigh): DFlash2 is +22%
  aggregate decode over the MTP ladder on the same instances; the
  prefix-cache tiers with the shared system-block cut took the prefill wall
  from 255 s to 110 s on a 12-instance run (09-08).
- Decode round at K=7: 17.8 ms = draft 2.5 + verify 15.1 + host 0.2 ms;
  wider K loses on the round wall at every depth measured (09-08).
- Concurrency ladder at 8 slots / 16K: **530.6 t/s** aggregate (08-19, the
  MTP ladder; DFlash2 serving is single-slot today).
- 3090 (24GB): **102.2 t/s** median live CC decode at 131K context.
- Cross-engine long-context (08-27, Qwen3.8, all four engines):
  decode does not erode with context on ANY engine -- q27 leads decode
  (~150-180 t/s), vLLM leads cold prefill ~2.5x. Tables and the vLLM
  spec-decode retest: [bench/crossengine/LONGCTX.md](bench/crossengine/LONGCTX.md).

Speed parity is not task parity: a quality table over the 09-08 campaigns
found three of twelve tasks per DFlash2 arm dying on their first turn from
tool calls the streaming parser never saw. That is fixed in v0.11.0
(undeclared-name pass-through, the wrapper-family openers; 12/12 non-empty
on 09-09), and the table now ships with every campaign readout
([bench/swebench/quality_table.py](bench/swebench/quality_table.py)).

fp4 note, because the story inverted twice: block-scaled fp4 MMA does exist on
consumer Blackwell (sm_120a only; under plain sm_120 the failure is
indistinguishable from missing silicon), runs at 780-868 TFLOPS here
(`tools/microbench_mxf4`), and still loses at decode -- nvfp4 moves 1.06x the
bytes of Q4_G64 for the same weights, and decode is a byte count, not a FLOP
count. TMA bulk-tensor copies exist there too (09-08) and lifted a bitwise
W4A8 prefill GEMM to 1.33-1.39x of the incumbent, short of the 1.6x that
would have justified the port.

## Why this model is a good target

- Dense-ish 27B that fits entirely in 32 GB VRAM at 4-bit -- no expert
  offload, no DRAM scatter.
- MTP draft head trained into the checkpoint: self-speculation without a
  separate draft model.
- Hybrid Gated-DeltaNet: KV lives only in the 17 full-attention layers
  (72 KB/token fp16); the other 48 layers carry a fixed-size recurrent
  summary. Measured 5090 KV ceilings: fp8 294,912 tokens, turbo3 655,360 --
  2.5x the native window.
- The catch the napkin misses: attention KV is restorable state, GDN state is
  all-or-nothing per sequence. Hybrids make context cheap but make context
  REUSE an engineering problem -- and the measured cost of ignoring it was
  7.9x wall-clock on agentic traffic. That trade is where this engine lives.

## Prefix reuse on a hybrid, and why it decides the benchmark

vLLM's PagedAttention caches KV as content-hashed blocks -- close to a free
lunch on a pure-attention transformer, where KV is an append-only log. Hybrid
GDN breaks the assumption: 48 of 65 layers carry no KV, only a dense
order-dependent recurrent summary that cannot be paged, shared by hash, or
rebuilt from cached blocks. A block cache covers 17/65 layers; without the
matching GDN state those blocks are dead weight.

Measured consequence (08-17 four-engine run): ninfer got **0% reuse** on real
Claude-Code traffic -- 541 requests, every one a `full_reset` -- and paid
97-327 s/instance against q27's 47 while *decoding faster*. At the time this
was by design: two resume offsets, arbitrary prefix reuse a documented
non-goal. That position changed -- their 2026-09-03 exact-identity reuse fix
measures **91.3% reuse and 51 s/inst** on the same artifact and harness
([NINFER-REBENCH.md](bench/crossengine/NINFER-REBENCH.md)); the hybrid-GDN
analysis above stands, the ninfer example is now historical. llama.cpp reaches 93.9% by checkpointing
recurrent state per slot at ~1 GiB each, which caps it at 6 slots on 32 GB.
vLLM reaches 89.8% (fixed upstream since the 07-15 run measured 0%).

q27 treats the GDN summary as a first-class object:

- **P8 stable-prefix snapshot**: all 48 GDN states at the last ChatML-stable
  boundary, plus split-encode so tokenization itself is prefix-stable.
- **P9 checkpoint ring**: pinned-host copies every 4096 tokens, so mid-history
  divergence rewinds to the nearest checkpoint instead of position 0.
- **P16 persistent prefix cache** (`--prefix-cache DIR`, opt-in): survives
  restarts; entries are verified token-by-token before any state is read, and
  a second entry cut inside the system+tools block is shared across
  conversations. Details: `docs/plans/2026-07-24-persistent-prefix-cache.md`.

A warm CC turn is restore + suffix-only prefill: `prompt=25473 hit=24136
pf=1337` in the `[req]` log, ~1.3 s instead of 10-20 s. That arithmetic, times
every turn of a 30-90-turn trajectory, is the whole wall-time story.

## Architecture facts (ground truth from GGUF metadata)

| | |
|---|---|
| arch | `qwen35` (Qwen3-Next-style hybrid) |
| layers | 65: 48 Gated DeltaNet + 16 full attention (every 4th) + 1 MTP layer |
| hidden | 5120 |
| FFN | SwiGLU, intermediate 17408 |
| full attention | GQA 24Q/4KV, head_dim 256, QK-norm, gated output |
| RoPE | partial, dim 64 of 256, M-RoPE sections, freq_base 1e7 |
| vocab | 248320, embeddings + lm_head untied |
| MTP | 1 nextn layer: eh_proj combines (embedding, hidden) -> attn + FFN -> shared lm_head |
| context | 262144 native |

Per-layer forward-pass semantics: [docs/SPEC.md](docs/SPEC.md).

## Performance model

Single-stream decode is weight-read-bound: t/s = BW x eff / bytes-per-step x
accepted-tokens-per-round. The GPU offers hundreds of ops per byte of DRAM
bandwidth and batch-1 decode uses ~2, so >99% of compute idles while weights
stream. Datacenters close the gap by batching users per weight read; q27
batches with itself -- the width-12 verify amortizes one read across the MTP
ladder plus the suffix drafter's free lanes, 5.3-5.8 accepted tokens per round
on live traffic. That is how 231 t/s clears a ~105 t/s plain ceiling on the
5090 (and 102 clears ~52 on the 3090, where ncu puts the GEMV family at
81-90% of DRAM speed-of-light -- the ceiling is real, not assumed).

Corollary, proven repeatedly: every decode win is (a) fewer bytes per step or
(b) more accepted positions per weight read. There is no third lever at batch
1; continuous batching is lever (b) pointed across users.

Decode numbers are not interchangeable -- the short-bench suite, the bitwise
canonical, tie-lottery methodology for tolerance-class changes, and the depth
numbers each answer a different question. The methodology, with the OC policy,
lives in [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

## Design decisions

- **Weights**: custom 4-bit symmetric groupwise (group 64, fp16 scales),
  packed for coalesced 128B warp loads, dequant fused into GEMV. Repacked
  offline from the BF16 GGUF ([docs/FORMAT.md](docs/FORMAT.md)).
- **KV cache**: fp16 on the CLI (canonicals stay bitwise); the server
  defaults to fp8 E4M3 on sm_89+ (scale-free saturating conversion -- amax
  sits 3.8x under the E4M3 max) and turbo5k on Ampere. turbo3 3-bit
  (14.1 KB/token) is the capacity lever; turbo5k (5-bit K) exists because the
  08-01 tail study found turbo3 at 6x fp8's catastrophic-position rate, and
  cuts that 43% -- also the study that showed **PPL ranks KV formats
  backwards** (signed error cancels in the mean; gate on tail metrics).
- **MTP**: first-class. Draft + verify in one pipeline under a single CUDA
  graph; no separate draft context, no re-prefill. Confidence-gated depth
  (`Q27_PMIN`, adaptive 4..7 ladder in src/depthctl.h) is bitwise-neutral on
  output -- round segmentation varies, tokens never do.
- **Stack**: plain CUDA C++. No CUTLASS, no deps beyond the CUDA runtime.
  Offline tools are Python (tools/repack.py; tools/gguf_to_hf.py, certified
  866/866 tensors byte-exact).
- **Numerics contracts**: fused lane families carry an N-invariance gate
  (bitwise-when-untrimmed); batched numerics match solo via a union
  GEMM-family policy; the only text forks are documented tolerance classes.
- **Batching guard**: user-explicit `Q27_BATCH=1` plus an incompatible env is
  fail-fast fatal; a profile default in the same position auto-disables with
  a banner -- a default must never kill a formerly-working invocation.

## Security

The model chooses every byte the tool-call parser sees, so the parser is
attack surface. [docs/SECURITY.md](docs/SECURITY.md) documents the trust
boundaries (a memory-safety bug is GPU-host code execution; a semantic bug is
client-sandbox execution), why the vLLM `eval()` class cannot reach this
codebase (no eval, no template engine), and the defences: coverage-guided
fuzzing under ASan+UBSan (`make fuzz`, millions of inputs clean), per-shape
adversarial tests, and a replayed drift corpus (below) that gates parser
changes.

## Serving

```
make build/q27-server
./build/q27-server model.q27 model.tok --port 8080
```

**Sampler: greedy by default, and that is a choice you may want to change.**
Claude Code sends no sampling fields, so agentic serving is greedy until you
say otherwise -- and for Qwen3.8 the measured recipe is the model card's
sampler: **`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05
--think-budget 0`**. The A/B (BUILDLOG 08-22/23): greedy 0.758, matched
sampler 0.847 vs llama.cpp 0.878, no task separating the engines once
matched (pooled p=0.824). The third flag matters as much as the first two:
the default think budget is half of `max_tokens`, Claude Code sends 64000,
and a force-closed reasoning block can end a session on an empty turn --
measured at 5 of 6 trials scoring 0.000 before `--think-budget 0`. The
startup banner always records which sampler ran.

**Defaults = the measured Claude-Code stack.** A bare server serves the exact
config the record numbers were earned on: fp8 KV (sm_89+; turbo5k on Ampere),
fused-MMA verify, `Q27_PMIN=0.5`, adaptive depth, suffix drafter at width 12,
continuous batching, auto-sized `--ctx`. Every knob keeps its env/flag
override, `Q27_PROFILE=ref` restores conservative reference behavior, and the
CLI binary keeps reference defaults so the bitwise canonicals are untouched.

**Persistent prefix cache**: `--prefix-cache DIR` (opt-in). Restart TTFT
8.15 s -> 1.20 s; entries verified token-by-token; LRU capped by
`--prefix-cache-max-gb`. Stores conversation content on disk in plaintext --
see [docs/SECURITY-MODEL.md](docs/SECURITY-MODEL.md). The pinned host-RAM
tier above it (`--prefix-cache-ram-gb`) is off by default on measured grounds.

**DFlash2 drafter (the production decode path since 2026-09-08).** The
drafter is z-lab's
[Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
(3.8 GB bf16, Apache-2.0). Ready-made packs are in the
[signalnine/Qwen3.8-27B-MTP-q27](https://huggingface.co/signalnine/Qwen3.8-27B-MTP-q27)
model repo next to the tiers (`qwen38-dflash2-q8-serve.d2w`, 2.1 GB, and the
Q4 pack, 1.2 GB; md5s in its CHECKSUMS.md5). To build one yourself:

```
pip install torch safetensors numpy
python3 tools/dflash2_pack.py /path/to/Qwen3.8-27B-DFlash2 qwen38-dflash2-q8-serve.d2w --q8
```

The serving pack carries only the drafter (about 2.1 GB): the engine
supplies the embedding and the head from its own Q8 tensors. `--q8` is the
production choice (bitwise with the fp16 pack on the acceptance gate);
without it the matmuls go Q4 (1.2 GB, -1.5 to -3.6% tok/round measured).
`--with-target <hf_dir>` adds the fp16 embed and head for the CLI's
`q27 --dflash2` path and the numerics A/Bs; serving never needs it. Then:

```
Q27_DFLASH2=qwen38-dflash2-q8-serve.d2w Q27_BATCH=0 Q27_DFLASH2_RESERVE_GB=3 \
  ./build/q27-server model.q27 model.tok --port 8080 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0
```

`Q27_BATCH=0` because DFlash2 serving is single-slot today, and the reserve
keeps the KV pool from taking the VRAM the drafter needs. K=7 is the
measured default (`Q27_DFLASH2_K`; wider loses on the round wall).
[tools/launch_q27_38.sh](tools/launch_q27_38.sh) is the exact production
recipe, prefix-cache tiers included.

**Auth**: off by default -- loopback-only binding is the safety net.
`--api-key KEY`, `--api-key-file PATH`, or `Q27_API_KEY` (preferred in
containers) all add keys; both `Authorization: Bearer` and `x-api-key` work.
Binding non-loopback with no key warns but is not refused.

**Metrics**: `--enable-metrics` (opt-in, default off) exposes `GET /metrics`
in Prometheus text exposition -- per-API request/token counters, prefill
computed/cached split, TTFT/E2E/ITL latency histograms, live token counters
(dashboard tok/s moves *during* generation, not only at request end),
KV/prefix-cache/inflight gauges, and the MTP accept-ratio lanes. Auth-exempt
like `/health`; 404 and zero bookkeeping cost when the flag is absent.
Series reference and consumer recipes:
[docs/metrics-endpoint.md](docs/metrics-endpoint.md).

**Thinking**: the default profile is no-think (prefills an empty
`<think></think>`); `--think` flips it. Per-request control is opt-in behind
`--request-think` so a harness that sends `enable_thinking:true` cannot
silently flip a no-think server. Reasoning inside a prompt-seeded think block
is budgeted (default: half of `max_tokens`, force-closed so the answer still
fits; `--think-budget 0` opts out, `N>0` also arms later blocks); trips are
reported in `usage`. Why it defaults on: a 240-scenario A/B found unbudgeted
block mode truncating 28 times against inline's 0 at level accuracy.

Three API shapes on one server:

- **OpenAI**: `/v1/chat/completions`, `/v1/completions`
- **Anthropic**: `/v1/messages` -- thinking blocks, tool_use/tool_result,
  input_json_delta streaming, exact `count_tokens`, anthropic-shaped
  context-limit errors, cch billing-header normalization.
  `ANTHROPIC_BASE_URL=http://host:8080 claude`
- **OpenAI Responses**: `/v1/responses` -- Codex-compatible: function tools,
  `custom` freeform tools, reasoning items; event set verified against the
  codex-rs source.

Codex config (`~/.codex/config.toml`):
```toml
model_provider = "q27"
model = "gpt-5-codex"

[model_providers.q27]
name = "q27 local"
base_url = "http://localhost:8080/v1"
wire_api = "responses"
```

**The tool-call parser** renders tools per the qwen35 chat template and
recovers a cataloged set of drift modes -- currently numbered up to 22, from
dropped openers through parameter-as-opener spellings -- because strict
parsing scores 0.000 on the hardest task class. Since 2026-08-25 the catalog
is gated by a replayed drift corpus: redacted shapes captured from real
traffic, human-labelled, and re-parsed on every change (`make corpus-check`,
100% agreement on 159 shapes / 1,442 turns). `--constrain-tools` stays opt-in
(3.1x in-call cost at depth).

## Benchmarks

Four engines, one 5090, one harness, one accounting convention, same weights.
12 pinned SWE-bench_Verified instances driven through Claude Code; requests
normalized so each engine sees identical bytes; timing client-side. Sampling
is each engine's own defaults; n=1 per instance. Full methodology:
[docs/BENCHMARKING.md](docs/BENCHMARKING.md); harness and raw data:
[bench/swebench/](bench/swebench/) and [bench/crossengine/](bench/crossengine/).

**Real agentic traffic, DFlash2 era** (2026-09-09, both engines on block
drafters, effort pinned to medium, q27 on the production recipe with a
fresh cache root; [readout](bench/crossengine/agentic-2026-09-09/README.md)):

| engine | decode agg / median | tok/round | prefix reuse | wall/inst | turns, out tok /inst | gold |
|---|--:|--:|--:|--:|--:|--:|
| **q27** production (Q8 pack, K=7) | 207.2 / 219.4 t/s | 3.89 | 96.9% | 108 s | 25.0, 18.2K | 9/12 |
| ninfer DFlash2 k=7 (NVFP4) | 220.6 / 240.1 t/s | 4.19 | 96.8% | **36 s** | 15.0, 6.2K | 11/12 |

Decode within 6%, reuse equal, wall 3x apart because q27's sessions take
1.7x the turns and 3x the output tokens -- consistent across the 09-07,
09-08 and 09-09 runs, cause not yet attributed (quant tier, the effort
rendering, or the parser). n=1 per instance.

**Real agentic traffic** (2026-08-17, ninfer before its prefix-reuse fix):

| engine | decode | wall/inst | prefix reuse | gold |
|---|--:|--:|--:|--:|
| **q27** q5f | **223.4 t/s** | **47 s** | 92.1% | 10/12 |
| **q27** q4s | 215.5 t/s | 48 s | 88.7% | 9/12 |
| ninfer NVFP4 | 250.2 t/s | 97 s | **0%** | 11/12 |
| ninfer int8 | 261.5 t/s | 327 s | **0%** | 11/12 |
| llama.cpp Q5_K_M + MTP | 120.5 t/s | 71 s | 93.9% | 11/12 |
| vLLM NVFP4 (no spec) | 67.8 t/s | 84 s | 89.8% | 9/12 |

**Concurrency ladder** (aggregate t/s, 8 slots / 16K, 2026-08-19 re-run,
competitor binaries byte-identical to 08-17):

| engine | C=1 | C=2 | C=4 | C=8 |
|---|--:|--:|--:|--:|
| **q27** q4s | 141.3 | 229.7 | 352.3 | **530.6** |
| **q27** q5f | 134.7 | 173.8 | 299.9 | 509.7 |
| ninfer NVFP4 | 157.2 | 299.7 | 442.3 | **834.3** |
| ninfer int8 | 137.0 | 179.5 | 219.5 | 353.6 |
| vLLM NVFP4 | 67.4 | 121.4 | 215.7 | 438.7 |
| llama.cpp | 84.8 | 168.3 | 149.1 | 193.1 |

q27's rows carry one opt-in flag worth +8.6% at C=8 only (stock: 488.4, still
+18.3% over its 08-17 self on defaults). ninfer's batch peak is not an fp4
win -- their own `text_policy()` locks their int tier out of their batch
kernel, so the within-engine comparison was never a format control. Everyone
pays for hybrid-GDN state somewhere: llama.cpp OOMs past 6 slots, vLLM's pool
holds 6.53 sequences at 16K, ninfer trades reuse away, q27 reaches 8 slots
because record-then-fold cut its per-slot GDN cost.

**Quality does not separate them**: 75 deterministic-verifier scenarios at
temp 0 put four engines and five quant formats inside a 2-point band -- a
null result at n=1, not a demonstration of equivalence, but on this evidence
the throughput numbers carry no quality asterisk.

**Long-context arm** (2026-08-27, Qwen3.8, think-on, matched sampler):
decode does not erode with context on any engine -- flat to gently declining
from 2K to 51K; q27 leads decode, vLLM leads cold prefill by ~2.5x. The
08-17 finding that vLLM's MTP path corrupts under agentic load **did not
reproduce on the current nightly**: spec-on is coherent across 24 Claude Code
sessions and worth +60% decode, with two caveats (one session in 24 fell into
a repetition loop; vLLM 400s any request whose prompt plus `max_tokens`
exceeds the context window, which walls Claude Code sessions at ~42.5K prompt
where q27 and ninfer clamp). Tables, raw records, and the walls paid for:
[bench/crossengine/LONGCTX.md](bench/crossengine/LONGCTX.md).

The earlier three-engine run (2026-07-14) is superseded but kept: it is the
reason ngram-style drafting is not in this engine -- it added ~nothing on
real coding while MTP nearly doubled stock llama.cpp.

## Open items

- **DFlash2 serving is single-slot.** The batched decode path (8 slots,
  530 t/s aggregate) is the MTP ladder's; the drafter's fused commits
  bypass the ring mirrors, so batching it is real integration work, and the
  serial SWE-bench harness cannot show whether sustained concurrency exists
  to pay for it (queue wait was 2.3% of a 12-instance run).
- **Cache persistence stops at 65536 tokens.** A 69.7K-token conversation
  re-prefilled 29K after a side request; raising `--prefix-cache-max-tokens`
  costs pinned staging memory per slot. Not yet measured against 128K
  admission.
- **The request replay has no corpus yet.** `Q27_REQ_LOG` and
  `bench/replay/` are gated (two fresh boots agree on every output); a
  recorded multi-session Claude Code log is the missing input for a
  binary-vs-binary A/B that does not ride on the harness's trajectory noise.
- **W4A8 prefill GEMM on the shelf**: bitwise at 1.33-1.39x of the incumbent
  with TMA fills, short of the 1.6x bar; `tools/gemm_w4a8_spike.cu` holds
  it if a 1.35x is ever wanted as-is.
- **Graph-cache cap under churn**: live CC draws 44+ keys against the bench's
  28; cap 64 covers today, revisit `Q27_BATCH_GRAPH_CAP` if multi-tenant
  churn widens the alphabet.
- **Does a budgeted think-block beat inline?** Not yet run: the 07-28 A/B
  leaves block and inline level on accuracy while the block arm eats 28
  truncations to inline's 0; a budget converting even part of those should
  put the block arm ahead.
- **turbo5k's tail is not a subset of turbo3's**: it fixes 89 of turbo3's 114
  catastrophic positions and introduces 40 new ones. Net -43%, but a workload
  fine on one is not guaranteed fine on the other. TCQ filed as its own plan,
  not started.
- **`billing-header cch normalize`** fails in `test_tokenizer` on HEAD
  (`5a81225` last touched that normalizer). Filed, not fixed.

Closed items keep their receipts in [docs/BUILDLOG.md](docs/BUILDLOG.md);
measured-and-parked levers in [docs/notes.md](docs/notes.md).

## History

The full chronological record -- every DONE block with its numbers, every
negative result, the progress table (43.4 -> 219.5 t/s single-stream) --
lives in [docs/BUILDLOG.md](docs/BUILDLOG.md). The BUILDLOG is the ledger,
not git history. Design docs and phase plans: [docs/plans/](docs/plans/).
Standing risk register and parked levers: [docs/notes.md](docs/notes.md).
Multi-slot throughput analysis: [docs/multislot-throughput.md](docs/multislot-throughput.md).
What a new Qwen checkpoint must match before it loads:
[docs/PORTING.md](docs/PORTING.md).

## License

MIT -- see [LICENSE](LICENSE).
