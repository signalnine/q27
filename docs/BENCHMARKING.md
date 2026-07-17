# q27 benchmarking methodology

How the q27 numbers are produced, and how to reproduce the cross-engine
comparison against the [llama-cpp-turboquant](https://github.com/) fork with
`ngram-mod`. Two engines are compared throughout:

| engine | build | quant | spec-decode |
|---|---|---|---|
| **q27** | git `94e645a` | NVFP4 **5.25 bpw** | MTP head + SuffixDraft (fused verify) |
| **llama-cpp-turboquant** (TheTom fork) | git `c3e6dbb13` | Q5_K_M **~5.5 bpw** | `--spec-type ngram-mod` (n_match=24, n_max=64, n_min=48) |
| **llama.cpp mainline** | git `13e67386` (2026-07-01) | Q5_K_M **~5.5 bpw** | run two ways: none (stock), and `--spec-type draft-mtp --spec-draft-n-max 6` (same MTP head as q27) |
| **vLLM** | `vllm/vllm-openai:nightly` | NVFP4 (`unsloth/Qwen3.6-27B-NVFP4`, compressed-tensors) | `speculative-config {method:mtp, num_speculative_tokens:3}` — the model's MTP head |

All three serve the **same base model** (Qwen3.6-27B-MTP; the `qwen35` GGUF arch,
which mainline supports as `LLM_ARCH_QWEN35`). The only unavoidable confound is
quantization (both llama builds carry ~0.25 bpw more than q27's NVFP4); it is
disclosed on every result and it favors llama, so any q27 win is conservative.
Mainline is the **stock, no-drafter floor**: it isolates how much of the gap is
the base decode kernel vs the drafter, and shows exactly what `ngram-mod` adds
over vanilla llama.cpp. (Mainline `13e67386` is ~2 weeks behind true nightly but
carries the `qwen35` arch and the Anthropic `/v1/messages` endpoint; a fresh
rebuild risks the known sm_120 toolchain traps and would not move the base-kernel
number materially.)

> **Serving headline (2026-07-16, v0.2.0):** 2 slots batch through one
> fused weight sweep + graph replay -- **1.41x aggregate over FIFO on both
> KV formats**, solo cost <=0.07%, zero-config. Full table + methodology:
> "Single-box serving: 2-slot continuous-batching aggregate" below.

## Fairness controls (every cross-engine run)

- **Single GPU, RTX 5090 only.** llama is pinned with `CUDA_VISIBLE_DEVICES=0` —
  otherwise llama.cpp layer-splits onto the second GPU (a 3090 here) and gets an
  unearned bandwidth edge. Verify the second GPU's memory is untouched after
  startup.
- **KV cache matched.** llama runs `-ctk q8_0 -ctv q8_0` (≈ q27's fp8 KV) and
  `-np 1` (single slot = single user, as q27 is).
- **Greedy decode** (`temperature 0`) on both sides.
- **Decode-only timing.** Throughput excludes prefill/TTFT so the number
  isolates the decode+drafter path, not prompt processing. (llama has the faster
  tensor-core prefill; q27 has the faster decode. Mixing them hides the real
  difference.)
- **Identical payloads / pinned tasks / same harness** on both engines.

## Method A — payload decode microbench (`/v1/completions`)

Isolates spec-decode effectiveness across regimes. Send identical
`{prompt, max_tokens, temperature: 0}` to each engine's `/v1/completions`.

- q27 decode t/s from its `[req]` journal line (`tps=`).
- llama decode t/s from the response `timings.predicted_per_second`
  (decode-only, excludes `prompt_ms`), with `draft_n` / `draft_n_accepted` for
  acceptance.

Three regimes, chosen to bracket drafter behavior:

| payload | regime | what it tests |
|---|---|---|
| `echo_ctx12k` (256 tok) | pure verbatim echo | drafter saturates |
| `fileemit_verbatim` (1024 tok) | partial-echo code continuation | near-repeat tolerance |
| `novel_prose` (400 tok) | novel generation | drafter with nothing to match |

**Two gotchas that will corrupt this bench if ignored:**

1. **ngram-mod persists its n-gram table across requests.** Re-sending an
   identical novel prompt makes it echo its own prior greedy output, so the rate
   climbs run-to-run (novel 56 → 79 → 97 t/s). Use the **cold** run (first call,
   or restart the server between prompts). q27 keeps no server-side table, so its
   number is request-invariant.
2. **Cross-quant divergence.** Q5 vs NVFP4 can pick a different greedy token on
   the same prompt, after which the two engines generate different text (and
   different token counts). Such a payload is no longer a like-for-like
   comparison — drop it (this is why `echo_ctx26k` is excluded).

## Method B — agentic traffic via SWE-bench Verified (reproducible)

Measures the engines on **realistic multi-turn tool-use coding traffic** that
anyone can rerun. This replaces the earlier private task set (see "History"):
private tasks can't be reproduced off this box, public SWE-bench instances can.

**Task set** — 12 instances pinned from `princeton-nlp/SWE-bench_Verified`,
biased to fast-test repos (`requests`, `flask`, `pytest`, `pylint`, `xarray`)
and `<15 min fix` difficulty, chosen deterministically (per-repo quota, sorted
by id). Regenerate with `bench/swebench/select_instances.py`; the frozen list is
`bench/swebench/manifest.json`. Pinned by `instance_id` + `base_commit` → exact
reproducibility.

**Harness** — Claude Code (`claude -p`, `--output-format stream-json`) pointed at
the engine's Anthropic `/v1/messages` API via `ANTHROPIC_BASE_URL`. Both engines
serve `/v1/messages` natively (q27 by design; the turboquant fork too — verified
streaming SSE, `tool_use`, thinking blocks, and `count_tokens`), so no
translation proxy is needed.

**Sandbox** — each instance runs the agent inside the `thunderdome/claude-code`
Docker image (claude CLI + node) via plain `docker run`, **not** the private
orchestration. Required flags:
- `--user 1000:1000` + `-e HOME=/home/node` — Claude Code refuses
  `--dangerously-skip-permissions` as root; run as the image's `node` user.
- `--add-host host.docker.internal:host-gateway` — reach the host engine on
  `:8081` from inside the container.

Running the untrusted upstream repo + autonomous agent in a container (not on the
host) is a safety requirement, not just convenience.

**Per instance** — clone `repo@base_commit` from a local bare mirror, feed the
`problem_statement` plus a "fix the code, don't run tests" instruction, let the
agent read/edit, then `git diff` = the candidate patch.

**Metrics**
- **decode t/s** (engine telemetry over the run window) and **wall-to-wall** time.
- **cheap quality signal**: non-empty diff, and *edited-the-gold-file* — overlap
  of the changed files with the gold patch's files (from the dataset). This is
  **not** the official resolve-rate.

**Why not official resolve grading.** SWE-bench's real grading applies the patch
and runs `FAIL_TO_PASS`/`PASS_TO_PASS` in per-instance Docker eval images
(multi-GB each) — heavy on a disk-constrained box, and, more to the point, since
both engines run the **same base model**, resolve-rate is a model property that
does not differentiate engines. The file-overlap signal is enough to confirm the
agent did something on-target; anyone wanting the resolve % can run the `swebench`
harness over the same pinned patches.

## Reproduce

Prereqs: Docker + the `thunderdome/claude-code` image (or any image with node +
the `claude` CLI), `pip install datasets`, and an engine serving the Anthropic
API on `:8081`.

```bash
# 1. materialize the pinned task set (or use the frozen manifest.json)
HF_HOME=/mnt/ai/hf_cache python3 bench/swebench/select_instances.py

# 2. one-time: bare mirrors of the 5 repos into $SWEBENCH_CACHE (default /mnt/ai/swebench-cache)
for r in pallets/flask psf/requests pydata/xarray pylint-dev/pylint pytest-dev/pytest; do
  git clone --bare "https://github.com/$r" "/mnt/ai/swebench-cache/${r/\//__}.git"
done

# 3. start q27 on :8081, run the set
#    (q27-server <model.q27> <model.tok> --port 8081 --host 0.0.0.0)
bash bench/swebench/run.sh q27

# 4. swap the engine on :8081 to the ngram-mod fork (5090-only + q8 KV), run the same set
#    CUDA_VISIBLE_DEVICES=0 llama-server -m Qwen3.6-27B-MTP-Q5_K_M.gguf \
#      -ngl 99 -fa on -c 131072 -np 1 -ctk q8_0 -ctv q8_0 --spec-type ngram-mod --jinja \
#      --host 0.0.0.0 --port 8081
bash bench/swebench/run.sh llama

# 4b. mainline baselines (mainline llama-server, git 13e67386):
#     stock (no drafter):
#       CUDA_VISIBLE_DEVICES=0 llama-server -m ...Q5_K_M.gguf -ngl 99 -fa on -c 131072 \
#         -np 1 -ctk q8_0 -ctv q8_0 --jinja --host 0.0.0.0 --port 8081
bash bench/swebench/run.sh llamamain     # -> results.llamamain.jsonl (unit llamamain-eval)
#     with the MTP head (fairest vs q27 -- same head, same model):
#       ...same launch... --spec-type draft-mtp --spec-draft-n-max 6
bash bench/swebench/run.sh llamammtp     # -> results.llamammtp.jsonl (unit llamammtp-eval)

# 4c. (vLLM) needs two extra pieces: vLLM has no /v1/messages, so Claude Code talks
#     to a litellm Anthropic->OpenAI shim on :8081 that forwards to vLLM on :8080.
#     - vLLM (5090-only, single-seq, MTP, qwen3_coder tool parser for the XML tool format):
#         docker run --gpus '"device=0"' -p 8080:8000 -v <hf_cache>:/root/.cache/huggingface \
#           vllm/vllm-openai:nightly --model unsloth/Qwen3.6-27B-NVFP4 --served-model-name vllm-qwen \
#           --max-num-seqs 1 --max-model-len 131072 --gpu-memory-utilization 0.96 --kv-cache-dtype fp8 \
#           --trust-remote-code --enable-auto-tool-choice --tool-call-parser qwen3_coder \
#           --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
#     - litellm shim (config wildcard-routes to openai/vllm-qwen @ vLLM):
#         docker run -p 8081:4000 -v <config.yaml>:/app/config.yaml ghcr.io/berriai/litellm:main-stable \
#           --config /app/config.yaml --port 4000
#     Decode t/s comes from vLLM /metrics deltas (not journalctl): see scratchpad/vllm_swebench.sh.
bash bench/swebench/run.sh vllm          # -> results.vllm.jsonl

# 5. compare bench/swebench/results.{q27,llama,llamamain,llamammtp,vllm}.jsonl
```

Long runs should be launched under `systemd-run --user` (a crashed shell
otherwise tears the job's cgroup down).

## Results (2026-07-14, RTX 5090)

### Method A — payload decode (decode-only t/s)

| payload | regime | q27 | llama ngram-mod | winner |
|---|---|---|---|---|
| echo_ctx12k | pure echo | **603** (11.6 tok/rnd) | 529 (96% acc) | q27 |
| fileemit_verbatim | partial-echo | 178 (3.0 tok/rnd) | **409** (89% acc) | llama |
| novel_prose | novel | **157** (2.6 tok/rnd) | 56 cold / 97 warm | q27 |

No clean winner at the payload level: ngram-mod's 24-tok lookup / 64-tok drafts
win partial-echo continuations; q27's fused MTP wins pure echo (once acceptance
saturates) and novel generation (MTP drafts every round; ngram-mod has nothing
to match).

### Method B — SWE-bench Verified, 12 instances

| engine | decode agg | wall/inst | nonempty diff | edited gold file |
|---|---|---|---|---|
| **q27** (MTP + SuffixDraft, fused) | **202.7 t/s** | **47 s** | 12/12 | 11/12 |
| **vLLM** NVFP4 + MTP (`method:mtp`, n=3) | 117.1 t/s | 133 s | 12/12 | 11/12 |
| **llama mainline + MTP** (`--spec-type draft-mtp`, n-max 6) | 116.3 t/s | 80 s | 12/12 | 11/12 |
| **llama ngram-mod** (fork) | 61.1 t/s | 118 s | 12/12 | 11/12 |
| **llama mainline** (no spec) | 62.0 t/s | 120 s | 12/12 | 12/12 |

(vLLM decode is aggregate from `/metrics` deltas — `generation_tokens_total` /
`inter_token_latency_seconds_sum`; the others are per-`[req]` telemetry.)

The five engines decompose the gap cleanly (all same model + MTP head available):

- **ngram-mod adds ~nothing on real agentic traffic.** The fork (61.1 t/s) is
  within noise of — marginally *below* — stock mainline (62.0). At 34% draft
  acceptance the failed drafts + table bookkeeping cancel the wins. ngram-mod's
  advantage is real only on synthetic high-echo re-emission (Method A), a small
  slice of real coding.
- **MTP is the real lever, and two independent engines confirm it.** Turning on
  the MTP head nearly **doubles** stock mainline (62 → 116–117 t/s), and llama's
  MTP (116.3) and vLLM's MTP (117.1) land on essentially the **same number** from
  completely different codebases — strong evidence this is the drafter's ceiling
  for a mainstream engine on this model, not a one-off.
- **On that same MTP head, q27 is still ~1.73× faster than both** (202.7 vs
  ~117). That residual is q27's engine — the fused shared-KV MTP+SuffixDraft
  verify, NVFP4 kernels, and tie/tolerance discipline — not the drafter *choice*.
  It matches Method A, where q27 leads llama+MTP on novel generation (157 vs 92
  t/s) but ties on echo (178 vs 184).
- **vLLM pays a wall-time tax this benchmark exposes.** Its decode (117) is
  competitive, but its **wall/inst (133 s) is the worst of all five** because
  vLLM's prefix caching is dead on this hybrid-GDN arch (0% reuse) — every
  agentic turn re-prefills the whole growing context — and it runs behind a
  litellm Anthropic→OpenAI shim (vLLM has no `/v1/messages`). q27 and llama both
  reuse prefix/checkpoint state across turns, so they convert competitive decode
  into far lower wall time. This is an arch-support gap, not raw kernel speed, but
  it's real for anyone serving this model agentically on vLLM today.
- **Quality is engine-independent** (11–12/12 edited-gold-file across all five) —
  the model is identical; the engine only changes speed. The 1-instance spread is
  agentic noise.

Gap decomposition (real agentic decode): stock llama.cpp **62** → +ngram-mod
**~62** (≈0) → +MTP **116–117** (×1.9, and vLLM independently agrees) → q27's MTP
engine **203** (×1.73 on top).

## Single-box serving: 2-slot continuous-batching aggregate

The cross-engine numbers above are single-stream by design. This table is
q27-only: what continuous batching (a serving default since 2026-07-16)
adds when two slots decode at once. Method: `tools/batch_ab.sh`
(`LEGS="A B D" REPS=3 MAXTOK=512`) — fresh w16 server per leg, two ~25-27K
prompts (codegen + docs) warmed once so per-slot prefix snapshots land,
then 3 measured reps firing both simultaneously; the metric is
`(dec_codegen + dec_docs) / concurrent window`, median over reps. Leg A
pins `Q27_BATCH=0` (the FIFO round-interleave baseline); leg B is the
defaults-on path (`Q27_BATCH=1`; graph replay + cap 64 land from the
serving profile — the script's gate env sets only KV/PMIN/MAXD, so it
cannot suppress them); leg D replays the payloads solo under both to
price the k=1 fallthrough (bar: |delta| < 2%).

Measured 2026-07-16 at `c0c5c5e` (v0.2.0, rebuilt binaries):

| KV | A: FIFO interleave | B: batched + graphs | B/A | solo delta (D) |
|---|---|---|---|---|
| fp8 | 168.9 t/s | **237.7 t/s** | **1.41x** | +0.06% / +0.00% |
| turbo3 | 158.5 t/s | **224.2 t/s** | **1.41x** | +0.07% / -0.06% |

The arc that got here (2 slots, fp8 aggregate): FIFO 1.00x → P1 fused
verify 1.21x → P2 sweep fusion 1.31x → P3 shape-keyed CUDA-graph replay
1.41x, solo cost ~0% at every stage (BUILDLOG 2026-07-14..16). Greedy
text: docs is byte-identical A-vs-B on both KVs; codegen can fork through
the documented A1 suffix-trim policy (it did here — 4 trim rounds/leg fp8,
2 turbo3); turbo3 replays are additionally trajectory-sensitive to
concurrency rep-to-rep on BOTH legs (quantized-KV tie re-rolls; the docs
md5 sets still match A vs B).

## vs club-3090 community recipes (their harness, our silicon)

club-3090 maintains the largest public cross-rig benchmark matrix for
this model family (vLLM / llama.cpp / ik_llama / beellama recipes on
3090/4090/5090-class cards). On 2026-07-16 we ran THEIR canonical
harness verbatim against q27 -- endpoint-only mode, zero q27-side
special-casing:

```
cd club-3090 && URL=http://localhost:8020 CONTAINER=none PP=1 bash scripts/bench.sh
```

Their protocol: 3 warmups + 5 measured runs of two fixed prompts
(narrative essay `max_tokens=1000`, Python quicksort `max_tokens=800`),
streaming `/v1/chat/completions` at temperature 0.6 / top_p 0.95 with
`enable_thinking:false`; **wall TPS** = completion_tokens / wall (their
headline) and **decode TPS** = completion_tokens / (wall - client TTFT);
plus salted, cache-busted prefill probes at ~10K and ~90K token depths
(prefill t/s = prompt_tokens / TTFT, client-observed). Token counts come
from the OpenAI `stream_options.include_usage` usage chunk -- the q27
server grew that (commit `aa991de`) as the comparability prerequisite.
Each GPU was benched TWICE (their own repeatability practice); both
passes shown.

q27 rows (vanilla qwen36-27b-mtp, single slot, bare v0.2.0 serving
defaults, this rig, 2026-07-16):

| q27 config | ctx | narr wall (decode) t/s | code wall (decode) t/s | TTFT | prefill t/s @10K / @90K |
|---|---|---|---|---|---|
| **5090** (W12, fp8 KV + fdmma, auto-ctx), pass 1 | 262144 | **144.15 (151.81)** | **193.04 (210.92)** | 350 ms | 3372 / 2559 |
| 5090, pass 2 | 262144 | 143.97 (151.62) | 192.82 (210.65) | 350 ms | 3350 / 2560 |
| **3090** (w8, fp16 KV + h16 mma, `--ctx 24576`), pass 1 | 24576 | **84.06 (88.59)** | **105.76 (115.08)** | 609 ms | 1124 / SKIP (>ctx) |
| 3090, pass 2 | 24576 | 83.41 (87.88) | 105.67 (114.97) | 611 ms | 1123 / SKIP (>ctx) |

In-run CV <= 0.3%, pass-to-pass <= 0.8%. q27's own `[req]` telemetry
agrees with their client-side decode numbers within 0.5% (5090: 151.4
t/s at 2.64 tok/round narrative, 209.8 at 3.86 code; 3090: 88.2 /
114.5) -- two independent instruments, one answer.

Their published rows (club-3090 `BENCHMARKS.md`, quoted as-is with rig +
date -- their numbers on their community rigs, NOT re-measured here):

| their row (rig, date) | ctx | narr t/s | code t/s |
|---|---|---|---|
| 1x5090 vLLM DFlash fp8 (@efschu, 575 W, 2026-05-07) | 49K | 126.53 wall (127.98 decode) | 200.11 wall (204.80 decode) |
| 1x3090 ik_llama two-stage (370 W, 2026-05-24) | 200K | 59.4 decode | 97.8 decode |
| 1x3090 ik_llama MTP (370 W, 2026-05-23) | 200K | 59.67 wall (60.39 decode) | 68.78 wall (72.40 decode) |
| 1x3090 llama.cpp MTP (2026-05-23) | 200K | 49.69 wall (50.27 decode) | 57.50 wall (58.92 decode) |
| 1x3090 beellama DFlash (370 W, 2026-05-30) | 102K | 50.2 wall (50.4 decode) | 99.7 wall (101.3 decode) |
| 1x3090 vLLM long-text | 90K | ~50 | ~67 |
| *multi-GPU context:* 2x3090 vLLM dual (290 W, 2026-07-09) | 262K | 96 decode | 127 decode |
| *multi-GPU context:* 2x5090 vLLM dual (2026-06-25) | 262K | 153.41 wall (154.62 decode) | 196.91 wall (200.13 decode) |

Read (decode-to-decode, spec-on vs spec-on):

- **5090**: q27 **+19% narrative** over their best single-5090 row
  (151.8 vs 127.98 decode); code is a near-tie (**+3%**, 210.9 vs 204.80
  -- DFlash N=5 is strongest exactly on token-predictable code). On wall
  TPS q27 wins narrative +14% and cedes 3.5% on code. A single q27 5090
  lands within 2-6% of their DUAL-5090 vLLM aggregate row (144/193 vs
  153/197 wall).
- **3090**: q27 **+47% narrative** over the best published single-3090
  decode (88.6 vs ik MTP 60.39) and **+14% code** over the best (115.1
  vs beellama DFlash 101.3; +18% vs ik two-stage 97.8, +95% vs mainline
  llama.cpp MTP 58.92). A single q27 3090 reaches ~91% of their 2x3090
  vLLM dual decode row (96/127).
- **Prefill**: on the 3090 their llama-family rows publish the same
  client-observed PP instrument -- mainline 1025, ik 1109 -- and q27
  measures 1092-1126: parity on Ampere prefill (consistent with the
  07-12 raw-kernel A/B where llama led). The 5090 client-observed
  2.56K t/s at 90K depth matches q27's engine-side pf rate (2557-2583).
- **TTFT on tiny prompts** is q27's worst number in their table: ~350 ms
  (5090) / ~610 ms (3090) prefill-pipeline floor on 25-token prompts,
  where their vLLM rows publish ~51-53 ms. Gone by the first token;
  stated because their table shows it.

Caveats, theirs and ours, stated plainly:

- **Cross-rig variance dominates.** Their own rulebook: "variations ...
  usually trace back to power caps, PCIe lane counts, or pin." These are
  OUR numbers on OUR silicon vs their contributors' rigs. Concretely:
  most of their 3090 rows ran at a 370 W cap and they document -29..-42%
  decode going 370->230 W; our 3090 ran at stock (observed draw ~417 W).
  Part of the 3090 gap is power, not engine.
- **Sampling parity is temp+top_p only.** Their harness pins temperature
  0.6 / top_p 0.95 in the request; their engines compose a server-default
  top_k=20 on top, and q27 has no top_k. A sampling-distribution nuance,
  TPS-neutral -- every run on both sides decodes to the max_tokens cap.
- **Wall vs decode preserved** per their two-metric convention; both are
  quoted wherever they publish both.
- **Spec-to-spec is fair**: every headline row of theirs is
  speculation-on (MTP / two-stage / DFlash); q27 runs its own MTP +
  suffix stack. Nobody here is compared spec-off.
- **Their FAIL-drop rule** (failed runs fall out of the summary stats):
  no q27 run failed in any of the four passes. The one non-measurement
  is the 3090 90K prefill depth, which their harness SKIPped cleanly on
  q27's context-limit 400 (24576 ctx < 90K) -- their documented over-ctx
  path, not a failure.
- **Context ceiling is the honest trade on the 3090 row.** Their 3090
  recipes serve 102K-200K; this q27 3090 config serves 24576 (fp16 KV on
  24 GB; the 5090-calibrated auto-ctx anchor over-sizes on sm_86 -- auto
  36864 and explicit 32768 both OOM at spec-graph instantiation under
  the 07-16 defaults, 24576 boots). *(Auto-ctx recalibrated 2026-07-17
  -- measured-free sizing, exact per-token KV; picks now boot on both
  cards. See the BUILDLOG entry.)* q27's turbo3 KV serves 131072 on
  this same card at 102.2 t/s live agentic decode (BUILDLOG 2026-07-12);
  that config was not the one benched here.
- **Quant tiers differ across all rows** (theirs: AutoRound-INT4,
  Q4_K_M, IQ4_KS, Q5_K_S; q27: its nvfp4-family v1.4 tier, 17.73 GB) --
  same model family, not identical checkpoints. And their rows are dated
  2026-05..07 on the engine pins of those days; engines move.

### Matched-bpw rerun: q4s-v1 in the community 4-bit band (2026-07-16)

The run above carried the "quant tiers differ" caveat: the 5.25-bpw
default tier sits above their 4-bit recipes. The q4s-v1 repack (15.46
GB, **4.55 bpw**: Q8 promotions cut to 41 tensors, single Q4 lm_head;
BUILDLOG "q4s tier SHIPPED" + "q4s-v1 REPACK VALIDATION") drops q27
into their band, so this is the harness rerun with weights-bpw matched.
Same command, endpoint-only, two passes per GPU, bare serving defaults
(only `--ctx` differs on the 3090, see below). q27 bpw figures are the
tier labels (whole-model bytes / params, same denominator as the 5.25
default). Their AutoRound INT4 rows are ~4.1 bpw on the *linears* but
~5.0-5.3 whole-model once the fp16 embeddings + lm_head are counted --
q4s (Q8 embed, Q4 head) undercuts even that.

| config (weights bpw) | ctx | narr wall (decode) t/s | code wall (decode) t/s |
|---|---|---|---|
| **q27 q4s-v1, 5090** (4.55), pass 1 | 262144 | **146.82 (154.20)** | **181.60 (196.16)** |
| q27 q4s-v1, 5090, pass 2 | 262144 | 146.58 (153.94) | 181.49 (196.02) |
| q27 v1.4, 5090 (5.25), best pass (above) | 262144 | 144.15 (151.81) | 193.04 (210.92) |
| their 1x5090 vLLM DFlash (AutoRound INT4 ~4.1 linears / ~5.0-5.3 whole) | 49K | 126.53 (127.98) | 200.11 (204.80) |
| **q27 q4s-v1, 3090** (4.55), pass 1 | **61440** | **88.51 (93.16)** | **99.39 (106.92)** |
| q27 q4s-v1, 3090, pass 2 | 61440 | 87.84 (92.43) | 99.21 (106.72) |
| q27 v1.4, 3090 (5.25), best pass (above) | 24576 | 84.06 (88.59) | 105.76 (115.08) |
| their 1x3090 ik_llama two-stage (IQ4_KS ~4.25) | 200K | 59.4 decode | 97.8 decode |
| their 1x3090 ik_llama MTP (IQ4_KS ~4.25) | 200K | 59.67 (60.39) | 68.78 (72.40) |
| their 1x3090 llama.cpp MTP (Q4_K_M ~4.85) | 200K | 49.69 (50.27) | 57.50 (58.92) |
| their 1x3090 beellama DFlash (Q5_K_S ~5.36 + IQ4_XS draft) | 102K | 50.2 (50.4) | 99.7 (101.3) |

Same instruments as above: in-run CV <= 0.3%, pass-to-pass <= 0.8%;
engine `[req]` telemetry agrees with their client-side decode within
0.5% (5090: 153.5-153.9 t/s at 2.58 tok/round narrative, 195.1-195.3
at 3.45 code; 3090: 92.1-93.0 at 2.67, 106.3-106.5 at 3.17). 3090 at
stock power (observed ~417 W) vs their mostly 370 W-capped rows -- the
power caveat from the v1.4 run stands unchanged.

Read, decode-to-decode:

- **3090 matched-bpw**: q4s **+54% narrative** over the best published
  single-3090 decode (93.2 vs ik MTP 60.39, now at comparable bpw) and
  **+5.5% code** over beellama DFlash (106.9 vs 101.3 -- and DFlash
  carries 0.8 MORE bpw here). vs our own v1.4 rows: narrative +5.2%,
  code **-7.2%** -- the single-Q4-head tier re-rolls the code
  acceptance basin (tok/round 3.86 -> 3.45 on the 5090 code prompt;
  narrative 2.64 -> 2.58 barely moves), so the byte savings win on
  low-acceptance traffic and lose on high-acceptance code. Same
  pattern both GPUs.
- **5090**: q4s narrative 154.2 stays +20% over their best single-5090
  decode (127.98); code 196.2 now cedes **-4.2%** to their DFlash
  204.80 (v1.4 was +3%) -- the code-acceptance price above, stated
  plainly.
- **Context ceiling, the headline q4s buys**: the 3090 config serves
  **61440** (boots; 65536 OOMs at spec-graph instantiation; auto-ctx
  still 5090-calibrated, picks 69632 and OOMs) -- **2.5x** the v1.4
  config's 24576 on the same card, same fp16-KV defaults. The 2.27 GB
  of freed weights went straight to KV. Their 3090 rows still serve
  102-200K; closing the rest is the turbo3 lever (131K on this card,
  BUILDLOG 2026-07-12), not weights. *(Both flags resolved 2026-07-17:
  auto-ctx recalibrated to measured-free sizing, and q4s + turbo3
  boots the full 262144 window on this 3090 -- needle 6/6 at a 233K
  prompt. See the BUILDLOG entry and the addendum below.)*
- **KV-bits asymmetry, theirs-favoring**: our 3090 leg runs fp16 KV
  (16 bits/token-channel) vs their 3090 rows' q4_0/q5_0 KV (~4-5
  bits). They spend ~3x fewer bits on KV -- that is where their 200K
  ceilings come from, and it makes the decode comparison conservative
  in their favor on memory traffic (their KV reads are smaller).
- **Quality ladder at 0.66 bpw less** (q4s vs v1.4, validation run
  2026-07-16, BUILDLOG "q4s-v1 REPACK VALIDATION"): wikitext-2 PPL
  **8.0197 vs 8.0409 (-0.26%, q4s BETTER**; +1.29% vs the Q5_K_M bar
  vs v1.4's +1.55%); agentic-corpus NLL flat at CC depths except one
  content-diverse bucket (16k-32k **+2.63%**, generic-corpus control
  -1.29..+1.21% across the same depths => content noise, not
  systematic); needle spot 2/2 exact at ~149K/~24K depth in a 248.7K
  prompt. The cheaper quant does not pay a measured quality tax; it
  pays a code-acceptance speed tax.

### Addendum 2026-07-17 -- turbo3 closes the 3090 context column

Same card, same harness, q4s + `Q27_KV=turbo3`, auto-ctx (recalibrated
this day) lands the **full 262144 native window on the 24 GB 3090**
(0.67 GB spare at ready; needle 6/6 verbatim on a ~233K-token haystack,
deepest plant ~95%). Decode does not pay for it -- it gains:

| 3090 leg (q4s) | narr wall/decode | code wall/decode | TTFT | ctx |
|---|---|---|---|---|
| fp16 KV (07-16 leg) | 88.5 / 93.2 | 99.4 / 106.9 | ~565ms | 61440 |
| turbo3 KV (07-17) | **89.45 / 94.23** | **108.32 / 117.38** | 567ms | **262144** |

The 5090's turbo3 decode tax inverts on Ampere: 800 B vs 4096 B per KV
pair per token, and on a bandwidth-starved part the KV-read savings beat
the dequant compute (code +9.8% decode). Prefill tax ~3% at 10K (1096
vs 1131 tok/s); the 90K-class prefill leg is measurable for the first
time on this card (643 tok/s cache-busted, previously SKIPped as
over-ctx). Against their published single-3090 rows this config now
leads every column at once: narrative decode 94.2 vs ik's 60.4, code
117.4 vs beellama DFlash's 101.3, at 262K ctx vs their 102-200K
ceilings. The context trade flagged above is closed.

## History / non-reproducible baselines

An earlier cross-engine run used 3 **private** greenfield tasks (not
redistributable, hence Method B above). For the record, on those tasks q27
decoded **289.8 t/s agg / 236.5 med** vs llama **61.0 / 55.5** (~4.75×), with
ngram-mod accepting only **34%** on agentic traffic — the same effect Method B
captures on public tasks.

## Honest caveats

- Quantization confound: the two llama builds are Q5_K_M (+0.25 bpw vs q27's
  NVFP4, favors llama); vLLM is NVFP4 (the *same* quant family as q27, so the
  cleanest comparison), but from a different checkpoint (`unsloth`, the multimodal
  variant — its unused vision tower costs VRAM, which is why vLLM ran at 131072
  ctx not higher).
- vLLM carries two extra confounds the others don't: a **litellm proxy hop**
  (Anthropic↔OpenAI translation adds per-request latency) and **no prefix
  caching** on this arch (re-prefill every turn). Both inflate vLLM's wall time;
  neither touches its decode-t/s number. Read vLLM's wall/inst as "vLLM serving
  this model agentically *today*," and its decode-t/s as the cleaner
  engine-vs-engine number.
- Agentic runs are non-deterministic: wall time, turn count, and which files get
  edited vary run-to-run. Treat single-run Method-B numbers as indicative, not
  precise; average more instances/trials for tighter bounds.
- *edited-gold-file* ≠ correctness — it confirms the agent worked the right file,
  not that the fix passes tests.
- Single box, single run per engine.
