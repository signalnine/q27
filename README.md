# Quasar

A narrow inference engine for **Qwen3.6-27B-MTP and Qwen3.8-27B-MTP** (hybrid GDN+attention, trained-in MTP heads), their fine-tunes, and PrismML's **Ternary Bonsai 2 27B** (2-bit, Hadamard-folded) on a single RTX 5090 (3090 and 4090/Ada also supported; Apple-silicon Metal backend for the q4s tier). One model family, one GPU, as fast as possible. In the spirit of [antirez/ds4](https://github.com/antirez/ds4).

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
  instance against ninfer's 15 and 6K
  ([bench/crossengine/agentic-2026-09-09/](bench/crossengine/agentic-2026-09-09/README.md)).
  Attributed the same day: on one identical prompt with 24 seeds, q27's
  per-turn reasoning matches llama.cpp serving a Q8_0 of the model, and
  ninfer's NVFP4 arm is the one that reasons 1.5x shorter -- the shorter
  sessions are ninfer's quant or sampler, not a q27 deficit. Two q27
  defects were found and fixed on the way (the served model name made
  Claude Code drop prior thinking blocks; the `<tools>` block had been
  rendered compact and key-sorted since 08-22) without moving the gap:
  [bench/crossengine/agentic-2026-09-09-echo/](bench/crossengine/agentic-2026-09-09-echo/README.md).
  Corrected 2026-09-10: most of the per-turn gap was q27's -- its tokenizer
  spelled the `<tool_call>`/`<tool_response>` tags as text on every agentic
  prompt, which a turn-0 probe barely sees. v0.11.4 fixes it: 40% less
  thinking per turn, 70-75 s/inst; what remains is turn count
  ([bench/crossengine/agentic-2026-09-10-effort/](bench/crossengine/agentic-2026-09-10-effort/README.md)).)*
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

### Bonsai 2 27B (ternary Qwen3.8, 2026-09-18)

[PrismML's Ternary Bonsai 2 27B](https://prismml.com/news/bonsai-2-27b) is
Qwen3.8-27B with every projection ternary (one fp16 scale per 128) in a
Hadamard-rotated basis and no MTP block. q27 serves it natively: the repack
keeps the ternary values exact in a 2-bit container (`T2_G128`, 9.44 GB for
the whole model, embeddings and head in exact Q8), the engine rotates the
activations itself (sign + Walsh-Hadamard per 1024-block, fused into the
activation quantizers), decode runs 2-bit dp4a GEMVs, prefill and the
speculative verify run the int8 tensor-core GEMMs off a 2-bit staging
unpack, and DFlash2 drafts against the ternary target with the Qwen3.8 Q8
pack (the pack has no MTP head, so DFlash2 is the only drafter). The pack
is at [signalnine/Bonsai-2-27B-q27](https://huggingface.co/signalnine/Bonsai-2-27B-q27)
with the tokenizer and checksums. Bit-exact
containers mean the port is checkable against the reference fork: teacher-
forced logits agree at top-1 0.9974 over 383 positions and wikitext PPL
matches `llama-perplexity` on the same GGUF to 0.15% (8.2767 vs 8.2643 at
2048). `docs/plans/2026-09-18-bonsai2-ternary.md` has the design and log.

| tier | GB | wikitext PPL | HumanEval+ | needle | 5090 decode (DFlash2 K=7) |
|---|--:|--:|--:|---|---|
| Qwen3.8 default (v2) | 17.00 | 7.3121 | 30/30 | 6/6 @ ~120K | ~220 t/s |
| Bonsai 2 (T2) | 9.44 | 9.2508 | 25/30 | 6/6 @ ~90K | 233 t/s on the 700-token prompt, 346 on cities; round 14.8 ms; plain 110 t/s |
| Bonsai 2 (T3, `--bonsai2-container t3 --slim`) | 6.06 | 9.2513 (bitwise the T2 pack on the 3090) | same model | same model | the 8 GB-card pack; decode at the T2 pack's speed |

Same protocol as the table above (chunk 512, fp8 KV; the same-card pair on
the 3090 is 9.2513 vs 7.3102). The 2.3 GB engine stack plus the 9.44 GB
weights leave 23 GB of KV on a 32 GB card (auto context 262K). Read the PPL
and HumanEval+ columns before picking it: it is a different checkpoint, not
a quant tier of the one above. On the 12-instance Claude Code SWE-bench
campaign it lands the same patches (gold 11/12, identical to the Qwen3.8
legs) at 205 vs 222 t/s aggregate, but reasons about twice as long per
instance (37 vs 22 API turns, 75K vs 35K thinking chars), so wall is 2.3x
(`bench/crossengine/agentic-2026-09-18-bonsai2/`).

The Qwen3.8 drafter works against this target as-is. A drafter trained on
the ternary target does better: ProCreations'
[Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2)
(independent, Apache-2.0, the z-lab architecture retrained on Bonsai
features) packs unchanged through `tools/dflash2_pack.py --q8` and on the
same campaign gives 3.80 tok/round and 227.7 t/s aggregate against 3.47 and
205 -- above the Qwen3.8 default tier's own aggregate on this traffic, at
the same trajectory shape. Identity-gated against plain decode on the 3090
(the 5090's greedy is not width-invariant for any drafter, a known
instrument property; BUILDLOG 2026-09-18 (as)).

Multi-slot works too: without a drafter a Bonsai member rides the fused
round as a width-2 lane pair whose second lane is never accepted, so the
conductor's union sweep serves every slot from one weight pass (BUILDLOG
2026-09-18 (at); `Q27_BONSAI_FUSED=0` pins members to solo rounds). On the
5090 at 8 slots / 16K: 105 / 150 / 200 / **329 t/s** aggregate at C = 1 / 2
/ 4 / 8 against 111 / 116 / 116 / 117 time-sliced, byte-identical to plain
decode on the 3090 gate. Half the lanes are dummies there. With a drafter in
the pack the lanes fill: ProCreations'
[Ternary-Bonsai-2-27B-MTP](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP)
head (independent, Apache-2.0, the Qwen3.8 MTP block distilled onto Bonsai)
repacks as the pack's blk.64 with `tools/repack.py ... --mtp-safetensors
model_mtp.safetensors` (9.87 GB), the engine runs its ordinary gated ladder
against the ternary target (the MTP layer stays unrotated), and the same
ladder reads 173 / 231 / 384 / **512 t/s** at C = 1 / 2 / 4 / 8 -- the
Qwen tiers' class from a 9.9 GB pack (BUILDLOG 2026-09-18 (au); the
ladder is bitwise vs plain decode at 1500 CLI tokens, and the width-4-plus
verify has the engine's own near-tie flips on either model).

On a 12 GB card (3060 class): `repack.py --slim` stores the embedding and
head as T2 too (exact; 7.2 GB, or 7.6 GB with the MTP head), the
`build/q27-server-12g` target is a single sm_86 image with 8 lanes and
256-row prefill chunks, and `Q27_FIXED_STACK_GB=0.9` tells the pool sizer
what that build actually costs. Simulated at 11.7 GB free on the 3090: 32K
context plain, 20K with the MTP ladder, bitwise the full pack's plain
decode; the 2.1 GB DFlash2 pack leaves only 4K there, so the MTP pack is
the 12 GB drafter (BUILDLOG 2026-09-19 (av)).

On an 8 GB card: `repack.py --bonsai2-container t3 --slim` packs the body
as `T3_G128`, five trits per byte (1.6 bpw; 6.06 GB, or 6.49 GB with the
MTP head). The CUDA decode GEMV reads it in a layout built to sum exactly
what the T2 kernel sums, so the pack is bitwise the T2 pack at every width
(400/400 matrices, same PPL to six digits, same server texts); prefill
converts each matrix into a 22 MB T2 scratch on the fly (+4% wall). With
`Q27_FIXED_STACK_GB=0.6` and the Ampere-default turbo5k KV, a headless
RTX 3060 Ti measures 36.9K context, 42 t/s decode and 481 tok/s prefill
(first field report, 09-22); the 3090 simulation put 45K at 8.0 GB free,
12K with the MTP ladder (`Q27_FIXED_STACK_GB=0.8`), and 24K plain with a
display on the card. Decode runs at the T2 pack's speed rather than 24%
under it -- the digit extraction is exposed on the 3090 (BUILDLOG
2026-09-20 (aw)).

```bash
# Bonsai 2: repack the PTQ1_0 GGUF (exact, ~3 min, default container t2),
# serve with the Qwen3.8 tokenizer and DFlash2 pack (tools/launch_q27_38.sh
# has a `bonsai2` mode with this config)
python3 tools/repack.py Ternary-Bonsai-2-27B-PTQ1_0.gguf models/bonsai2-27b-t2.q27
Q27_KV=fp8 Q27_BATCH=0 Q27_DFLASH2=models/qwen38-dflash2-q8-serve.d2w Q27_DFLASH2_RESERVE_GB=3 \
  ./build/q27-server models/bonsai2-27b-t2.q27 models/qwen38-27b-mtp.tok --think \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0
```

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

## State of the engine (2026-09-21)

One binary serves Claude Code, Codex, and OpenAI clients on a 5090 with a
DFlash2 block drafter (K=7, MMA verify) as the production decode path, a
persistent prefix cache that hits on real agentic traffic, and a tool-call
parser measured against a labelled corpus of the model's own drift. Current
release: [v0.14.1](https://github.com/signalnine/q27/releases).

Headline numbers, each dated in the BUILDLOG and in the campaign READMEs
under [bench/crossengine/](bench/crossengine/):

- **Ternary Bonsai 2 27B (v0.12.0)**: PrismML's 2-bit Qwen3.8 served
  natively from a 9.44 GB pack -- exact ternary containers, activation
  rotation fused into the quantizers, 2-bit GEMVs for decode and a 2-bit
  staging unpack for the tensor-core prefill and verify GEMMs, DFlash2 on
  the ternary target. On the same 12-instance Claude Code run: **227.7 t/s
  aggregate at 3.80 tok/round** with a target-trained drafter (205 / 3.47
  with the Qwen3.8 one), 23 GB of KV left on a 32 GB card. The checkpoint
  itself measures +27% wikitext PPL and 25/30 HumanEval+ against the
  Qwen3.8 default tier and reasons about twice as long per instance --
  same patches landed, 2.3x the wall (09-18; the tier table below). Fused
  multi-slot rounds serve it at 329 t/s aggregate over 8 slots at 16K, and
  512 t/s with a third-party MTP head repacked into the pack. Small cards
  (v0.14.0): slim packs and a sm_86 build put it on 12 GB (32K context),
  and a five-trits-per-byte container, bitwise the 2-bit pack, on 8 GB
  (6.06 GB; a 3060 Ti measures 36.9K context at 42 t/s;
  `tools/install-bonsai2-8gb.sh`).
- Claude Code traffic, 12 SWE-bench instances, medium effort (the only
  level both engines render), 2026-09-10 re-bench: **q27 v0.11.3 218-222
  t/s** aggregate decode (232-233 median, 4.05-4.10 tok/round, two runs)
  vs ninfer's DFlash2 arm 218 (241, 4.11) the same day -- decode parity, up
  from 207 vs 221 on 09-09; the sampler-order fix is +4-6% of it against a
  same-day control. Prefix reuse 95.7% vs 96.1%. Wall per instance 98 s vs
  24 s: q27's sessions run 25 turns and 15K output tokens per instance
  against 11 and 4K.
- **Tokenizer fix (v0.11.4)**: q27's encoder never matched the
  vocab's `<tool_call>` / `<tool_response>` added tokens, so every agentic
  prompt since July showed the model its own tool calls and results spelled
  out as text. Fixed (0/34 -> 34/34 recorded Claude Code prompts identical
  to AutoTokenizer's ids): on the same traffic per-turn reasoning drops
  40-43%, decode rises to 228-232 t/s at 4.21-4.27 tok/round, wall 70-75 s
  per instance, 10-11/12 gold (two runs, 2026-09-10). What remains of the
  gap to ninfer is turn count: q27 still reproduces and verifies before it
  edits, as the Q8_0 reference does on the same states. v0.11.5 finishes the
  job for Unicode (NFC and the regex's character classes, taken from the
  reference itself): `tools/tok_parity.py` matches AutoTokenizer on every
  Unicode scalar value.
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

**Multi-slot windows are elastic** (`--slots N` with auto `--ctx`, since
2026-09-10, [issue #42](https://github.com/signalnine/q27/issues/42)): the
slots share one paged KV pool and any one of them may hold up to the whole
of it -- on a 5090, 262K tokens per slot at 4 slots, 152K at 8, where the
old per-slot division gave 49K and 2K. With continuous batching (the
default), a request is entitled to its prompt plus 4K rows and grows as it
writes, instead of reserving its whole max_tokens up front; every grant
passes a banker's-algorithm safety check, so some request can always
finish and one that cannot grow sits out a round rather than deadlocking
([plan](docs/plans/2026-09-10-incremental-kv.md)). What that buys is cache
survival: four 50K-token Claude-Code-shaped sessions on a 4-slot 5090 keep
all four conversations cached (warm turns 0.4 s) where up-front reservations
evicted each other on every turn (15 s re-prefills). A request the free
pages cannot cover reclaims idle conversations' caches, least recently used
first, then waits for a running one to finish. `Q27_KV_INCREMENTAL=0`
restores up-front reservation; an explicit `--ctx` / `--slot1-ctx` still
fixes the windows.

DFlash2 serving is single-slot: it needs `Q27_BATCH=0`, and with `--slots N`
every slot loads its own drafter (~2 GB for the Q8 pack plus ring and
scratch) and the slots take turns on the GPU, so extra slots buy separate
caches and windows, not throughput. The pool sizing reserves the drafter's
VRAM per slot (v0.11.7; before that once per process, and a 2-slot boot
could end at 0 MB free).

**When a request seems stuck**, read the startup lines and the `[wait]`
lines. Each slot needs a fixed stack plus a 16K-token KV floor, so on a
card with less free VRAM `--slots N` can bring up fewer slots; the server
now says so (`WARNING: --slots 4 requested, 2 slots came up (not enough
free VRAM ...)`), and requests past that count wait for a free slot.
Anything that waits longer than `Q27_WAIT_LOG_MS` (default 5000; 0 turns
it off) prints a `[wait]` line with its request id -- the same `rid` as
its final `[req]` line -- and the reason: all slots busy, the KV pool
short or unsafe to grow into, a parked KV growth, the GPU gate before a
prefill, or a prefill time-sliced with other requests' prefills. It
repeats every 30 s while the wait lasts and once more when it ends. The
`http:` startup line gives the HTTP worker count; connections past it
queue inside the HTTP layer before q27 sees them, with no log line.

Two things to know on WSL2 or any box that caps pinned host memory. Each
slot lazily pins a GDN checkpoint ring for mid-history divergence (up to 16
x ~157 MB); if that pin fails the ring turns off for the slot with a
`[ckpt]` line instead of taking the server down (before v0.11.7 it was
fatal on a long prompt), and `Q27_CKPT_INTERVAL=0` skips it up front. And
the Windows driver pages VRAM to system RAM instead of failing when the
card is oversubscribed, which shows up as decode at a third of its speed;
`vram: free ... at ready` near 0 is the tell.

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

**Real agentic traffic, DFlash2 era** (2026-09-10 re-bench, same harness
and effort pin as 09-09, both engines on block drafters, q27 on the
production recipe with a fresh cache root, same-day control;
[readout](bench/crossengine/agentic-2026-09-10/README.md)):

| engine | decode agg / median | tok/round | prefix reuse | wall/inst | turns, out tok /inst | gold |
|---|--:|--:|--:|--:|--:|--:|
| **q27** v0.11.3 (Q8 pack, K=7) | 218.0 / 232.2 t/s | 4.05 | 95.7% | 98 s | 25.2, 15.3K | 10/12 |
| q27 before PR #43 (control) | 209.7 / 223.1 t/s | 3.98 | --* | --* | 23.2, 16.8K | 12/12 |
| ninfer DFlash2 k=7 (NVFP4) | 218.2 / **240.5** t/s | 4.11 | 96.1% | **24 s** | 10.8, 4.0K | 11/12 |
| **q27 v0.11.4** (tokenizer fix; two runs) | **228.0-231.5** / 242.9-247.2 t/s | **4.21-4.27** | 95.7-96.5% | 70-75 s | 22.6-22.9, 11.2-11.7K | 10-11/12 |

The last row is the tokenizer fix from the same day
([readout](bench/crossengine/agentic-2026-09-10-effort/README.md)): q27's
encoder never matched the `<tool_call>`/`<tool_response>` added tokens, so
every agentic prompt spelled them as text. Fixed, q27 thinks 40-43% less
per turn on this traffic (1295-1387 chars vs 2281; ninfer 954) and drafts
better. A second v0.11.3 run measured 221.8 / 233.2 t/s at 4.10 tok/round. The
sampler-order fix (PR #43) is worth +4-6% aggregate decode on this traffic
against the same-day control and closes the tokens-per-round gap to ninfer
(3.89 vs 4.19 on 09-09). Wall stays apart on trajectory length, attributed
below. \*The control's reuse and wall (and the first v0.11.3 run's) are
confounded -- the shared /dev/shm filled mid-run and prefix-cache writes
failed (readout, read 4); the v0.11.3 row above is a rerun with a healthy
cache. The 09-09 table:

| engine | decode agg / median | tok/round | prefix reuse | wall/inst | turns, out tok /inst | gold |
|---|--:|--:|--:|--:|--:|--:|
| **q27** production (Q8 pack, K=7) | 207.2 / 219.4 t/s | 3.89 | 96.9% | 108 s | 25.0, 18.2K | 9/12 |
| ninfer DFlash2 k=7 (NVFP4) | 220.6 / 240.1 t/s | 4.19 | 96.8% | **36 s** | 15.0, 6.2K | 11/12 |

Decode within 6%, reuse equal, wall 3x apart because q27's sessions take
1.7x the turns and 3x the output tokens -- consistent across the 09-07,
09-08 and 09-09 runs. n=1 per instance, and a same-day control showed
that n=1 swings the aggregate by +-3 turns. Attributed
([readout](bench/crossengine/agentic-2026-09-09-echo/README.md)): on one
identical turn-0 prompt, 24 seeds per arm, q27's per-turn thinking
(production, ladder, fp16 KV, q6 tier: median 276-340 chars) is
indistinguishable from llama.cpp serving a Q8_0 of the same model (314,
p 0.3-0.7), while ninfer's NVFP4 arm reasons a median 209 (p=0.0003
against the reference). The shorter sessions are ninfer's quant or
sampler making the model terser than it is at 8 bits; per-token cost at
equal reasoning is the comparable number, and there the engines are
within 6%. Drafter, KV dtype, tier and the sampler chain are excluded on
q27. Two q27 defects surfaced and were fixed without moving the gap: the
served model name in responses made Claude Code drop prior thinking blocks
(now echoes the requested model), and the serving path's `<tools>` block
had been the compact key-sorted dump since 08-22 (now the template's
client-ordered spaced form, +5% prompt tokens on a 28-tool request).
[Corrected 2026-09-10: most of the per-turn gap WAS q27's. Its tokenizer
spelled the tool tags as text on every agentic prompt; a turn-0 probe
holds only a few tags, which is why it read "q27 = reference". Fixed, per-turn
reasoning drops 40-43% on Claude Code traffic. On identical mid-session
states, with identical token ids, the Q8_0 reference still matches q27
(1.11x, p=0.80) and ninfer sits within 5%; what remains is turn count --
q27 reproduces and verifies before it edits. See the 09-10 effort readout.]

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
| **q27** Bonsai 2 T2 (no drafter, 09-18) | 105.5 | 150.0 | 200.4 | 328.9 |
| **q27** Bonsai 2 T2 + MTP head (09-18) | 173.2 | 231.0 | 384.1 | 512.2 |
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
- **Cache persistence stops at 65536 tokens by default.** Returning to an
  evicted 69K-token conversation re-prefills 45K tokens (16.4 s); with
  `--prefix-cache-max-tokens 131072` the same return restores in 0.5 s, for
  about 4.6 GB more pinned host memory per slot
  ([docs/perf-next-2026-09-12.md](docs/perf-next-2026-09-12.md)). Raise it
  where the host RAM allows; the incidence on recent Claude Code campaigns
  was 0-3 such misses per 12 instances.
- **Incremental KV admits optimistically.** When long outputs outgrow the
  pool together, the request first in the safe order runs and the others
  park until it finishes: same total wall as up-front reservation in the
  gate (201 vs 203 s), but three of four requests finished later. There is
  no preemption and no fairness beyond the safety check, and the block
  table still uploads from pageable host memory on each growth.
- **Every q27 agentic number before 2026-09-10 ran with the tokenizer bug**,
  the drift corpus included (v0.11.4, deployed 2026-09-10 on a fresh
  prefix-cache root); whether the tool-call drift shapes (issue #38) came from
  it is untested.
- **Turn count is the remaining gap.** With correct tokens q27 thinks within
  5% of ninfer per turn on identical states, and the Q8_0 reference agrees
  with q27, but q27 still runs about twice the turns: it tries to reproduce
  and verify in harness containers that have no repo dependencies. The
  gold-file proxy cannot say whether that verification pays; SWE-bench's
  test images would. Effort low is not a lever on the evidence so far.
- **The request replay has a corpus now, locally.** `Q27_REQ_LOG` recorded
  five 12-instance Claude Code legs on 2026-09-10 (session content, kept out
  of the repo); `bench/replay/` has not yet been run on them.
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
