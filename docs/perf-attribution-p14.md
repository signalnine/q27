# P14 Task 1 -- post-fd2 decode + prefill attribution, go/no-go data

Measurement-only pass (no product code changed) on branch `p14-perf-levers`, HEAD
`b5c5ea0` (P13). Binary verified current via full `make` (nothing-to-do) + canonical
gate `4c4120c72056aba2bc2d2561471eafce` EXACT before any run. RTX 5090 (GPU 0),
memory OC STOCK, vox-transcriber untouched on the 3090.

**Headline reconciliation vs the BUILDLOG ballparks:** the round *cost* (ms/round)
reproduces the fd2 BUILDLOG within ~2-3% at every ctx; the *t/s* runs lower than the
wikitext ballparks because this pass drove a **repo-docs** prompt whose acceptance
(tok/round) is lower than the wikitext continuations the BUILDLOG used. **ms/round is
the binary-cost-comparable metric; t/s is acceptance-(prompt-)dependent.** Every gate
delta below is measured on the *same* prompt, so acceptance is controlled.

## Workload recipe (deterministic, reproducible)

- Corpus: `sorted(docs/*.md) + README.md` (12 files, 194,793 chars), concatenated with
  `\n\n`, tiled to 124,506 tokens, HF-tokenized with
  `/mnt/ai/models/qwopus-27b-mtp/hf-bf16/tokenizer.json` (same vocab as the `.tok`,
  exported from the same GGUF). Script: `scratchpad/prep_tokens.py`.
- Server text prompts (`prompt2k/16k/61k.txt`): HF-decode of the first 2000/16000/61000
  ids; server re-encodes to prompt=2000/16001/61003 (round-trip stable, verified from the
  `[req]` `prompt=` field).
- CLI token files (`toks60k.txt`, `toks75k.txt`): space-separated ids for `--tokens-file`.
- kvstats synthetic tokens (`synthtoks.bin`): 140,000 int32 ids (tiled `toks75k`, max id
  248,068 < VOCAB 248,320). Prefill timing is value-independent, so synthetic ids are
  faithful for timing.
- Server: `CUDA_VISIBLE_DEVICES=0 Q27_PF_XG=32 ./build/q27-server <model> <tok> --port 8081
  --ctx 131072 --no-think --fast-head` (single slot). `Q27_PMIN` is read once at
  `build_spec_graphs`, so each gate setting = one server restart (12 s teardown guard).
  Decode t/s / ms/round / tok/round come from the server `[req]` log (`tps`, `dec_ms`,
  `rounds`, `dec`), never from nsys-inflated walls.

---

## Step 2 -- ground-truth decode walls (greedy, no profiler)

`/v1/completions`, `max_tokens=384`, 1 warmup + n=3 (spread <0.2%, near-deterministic).
Prompt fully prefix-cached after warmup (`hit`~=prompt, `pf_ms`~=14), so decode is clean.

| ctx | metric | ungated | Q27_PMIN=0.5 | Q27_PMIN=1.0 |
|----:|--------|--------:|-------------:|-------------:|
| 2K  | t/s        | 112.9 | 121.1 (+7.3%) | 119.7 (+6.0%) |
| 2K  | ms/round   | 21.25 | 18.98 | 18.23 |
| 2K  | tok/round  | 2.400 | 2.299 | 2.182 |
| 16K | t/s        | 126.1 | 128.1 (+1.6%) | 121.4 (-3.7%) |
| 16K | ms/round   | 23.07 | 20.53 | 19.64 |
| 16K | tok/round  | 2.909 | 2.630 | 2.385 |
| 61K | t/s        | 109.9 | 114.9 (+4.6%) | 112.5 (+2.4%) |
| 61K | ms/round   | 29.87 | 26.95 | 25.49 |
| 61K | tok/round  | 3.282 | 3.097 | 2.866 |

- ms/round matches the fd2 BUILDLOG (2K ~20.3, 16K ~22.5, 61K ~29.2) within ~2-3%.
- Gate direction confirmed at 61K on docs traffic (both thetas positive), but **smaller
  than the wikitext BUILDLOG (+7-10.8%) and theta=0.5 > theta=1.0 here** -- the docs
  prompt's acceptance profile differs. **theta=1.0 over-narrows at 16K (-3.7%)**: a real
  regime where an aggressive gate hurts. Takeaway for Task 3/4 tuning: theta is
  acceptance-(prompt-)dependent; 0.5 is the safer default across ctx on low-acceptance text.
- CLI `--tokens-file` legs were dropped from Step 2: the CLI `--spec` path serial-prefills
  (`step_with` per token, `engine.cu:876`), so a 60K prompt is a ~15-20 min prefill and the
  server (batched prefill) is the authoritative, faster decode-wall harness. The gate delta
  is the same physics either way (verify-width narrowing).

---

## Step 3 -- sampled-vs-greedy (never measured before)

Same 61K prompt, server ungated, `max_tokens=384`, seed 42.

| 61K            | t/s   | ms/round | tok/round |
|----------------|------:|---------:|----------:|
| greedy         | 109.9 | 29.87    | 3.282     |
| T=0.7 top_p0.95| 80.0  | 30.19    | 2.415     |

**The sampled per-round KERNEL tax is +1.07% (30.19 vs 29.87 ms/round).** The large t/s
drop (-27%) is almost entirely lower acceptance (tok/round -26%), which is inherent to
rejection sampling, not a kernel cost. The nucleus/gumbel/spec_accept kernels add ~0.3
ms/round at depth. (2K sampled = 132.2 t/s > greedy 112.9 is the short-ctx tie lottery,
not representative.) `[sample-stats]` reported 2.415 tok/round at T=0.7 @61K.

---

## Step 4 -- post-fd2 round budget @61K (definitive; supersedes BUILDLOG:664)

nsys `--trace=cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi
--capture-range-end=stop`, `Q27_PROF_DECODE=1`, one 61K request `max_tokens=800`
(hit eos at 429 tok). Kernel-sum/round vs the Step-2 ground-truth wall closes within
~5% (per-node overhead cancels; node-tracing inflates *wall* ~1.75x, kernel sums honest).

- ungated: 130 rounds, kernel-sum **28.83 ms/round** vs wall 29.87 (**96.5%**).
- gated theta=1.0: 152 rounds (same 429 greedy tokens, more rounds = gate narrows verify),
  kernel-sum **24.08 ms/round** vs wall 25.49 (**94.5%**).

| kernel family | ungated ms/round | gated1.0 ms/round |
|---|---:|---:|
| batched verify GEMV (q4_n/q8_n)      | 12.24 (42%) | 10.74 |
| attention (fd2 draft+verify)         |  9.19 (32%) |  6.63 |
| draft/single GEMV (q4/q8/f16)        |  3.11       |  3.06 |
| GDN (delta/conv/gates/norms)         |  1.72       |  1.22 |
| rmsnorm/add/silu/rope/embed          |  1.16       |  1.15 |
| draft scans (k_margin+k_argmax) [Task 2] | 0.585   |  0.576 |
| attention combine                    |  0.398      |  0.338 |
| quantize                             |  0.386      |  0.339 |
| finish/prep/kv_store                 |  0.032      |  0.032 |
| **KERNEL SUM**                       | **28.83**   | **24.08** |

Structure @61K: weight-stream GEMVs (batched verify + single) = **15.4 ms/round (53%)**,
attention = **9.6 ms/round (33%)**, everything else = ~4 ms/round. The gate cuts the round
by narrowing verify width: batched GEMV -1.5, attention -2.57 (verify widths spread across
2/3/4/5 instead of always-5). `k_margin`'s 0.545 ms/round is the dead ungated-default scan
Task 2 fuses away.

### fd2 draft/verify split (nsys, per grid z)

| config | z=1 draft | z=5 (or width) verify |
|---|---|---|
| ungated @61K | 4/round x **127.8 us** = 0.51 ms/round | 16/round x **542.6 us** = 8.68 ms/round |
| gated1.0 @61K | 4/round x 127.7 us = 0.51 ms/round | widths 2/3/4/5 = 256/371/465/539 us; 6.11 ms/round total |

Model shape read off the split: 16 attention layers (16 verify fd2/round, ntok=5) + a
single-layer MTP draft head (4 draft fd2/round, ntok=1).

---

## Step 5 -- fd2 cross-lane KV traffic, ratio R (gates Task 5/6)

**R ~= 4.25 (nsys time-proxy on a BW-bound kernel). R >= 2 -> Task 5 GO.**

fd2 is DRAM-bandwidth-bound (BUILDLOG: ~808 GB/s = 45% of peak), so per-instance time is
proportional to DRAM bytes. Evidence, all self-consistent, from the Step-4 decode capture
(capture-range cleanly isolates post-prefill decode at 61K):

1. **Verify(z=5)/draft(z=1) per-instance time = 542.6/127.8 = 4.25x.** Both attend the same
   ~61K KV slice; verify carries 5 tokens, draft 1. If the 5 verify lanes shared the KV via
   L2, verify would cost ~1-1.5x draft (KV read once, extra compute hidden on a BW-bound
   kernel). Instead it is 4.25x -> the lanes re-stream KV from DRAM.
2. **Per-instance verify time is ~linear in width** (gated: w2 256, w3 371, w4 465, w5 539
   us; increments +128/+115/+94/+74 us/lane, ~= one full draft KV stream per added lane).
   Linear-in-lane-count on a BW-bound kernel = each lane independently re-streams. The
   slightly *decreasing* increments show L2 absorbs a growing minority at higher widths, so
   R is a bit under the naive 5 (hence ~4).

KV/layer @61K fp8 = 4 kv-heads x 128 head-dim x 61,440 pos x 2(K+V) x 1B ~= 63 MB; a z=5
verify instance re-streaming ~4-5x = ~250-315 MB vs the z=1 draft's ~63 MB.

**Direct ncu DRAM bytes: UNMEASURED-DIRECT** (`sudo -n ncu` works). Two confounds, recorded
for the next attempt: (a) on this Blackwell/sm_120 `dram__bytes_read.sum` is silently
dropped -- the collectable name is `dram__bytes_op_read.sum`/`dram__sectors_op_read.sum`;
(b) the server MTP-warms KV chunk-by-chunk *during* the 61K batched prefill, launching many
fd2 at growing context, which pollutes any `--launch-skip` window (the skip-60 window landed
on ~11K-ctx prefill-warm fd2: L2-hit z1=2.5% vs z5=73%, dur ~20 us -- not the 61K decode).
ncu-on-CLI is worse (serial-prefill launches 60K fd2 before decode). The nsys time-proxy is
the established BUILDLOG methodology and is decisive here (4.25x >> the 1.3 SKIP floor).

The ncu L2-hit numbers that DID land (draft z=1 = 2.5% hit at high ctx = pure DRAM stream;
verify z=5 = 52-73% = partial cross-lane reuse) independently corroborate "verify re-streams
most, but not all, of the KV" -> exactly the gap Task 5's axis-swap converts to L2 hits.

---

## Step 6 -- draft-attention share (gates Task 5b)

**Draft attention (fd2 z=1) = 0.51 ms/round @61K (4 inst x 127.8 us). < 1.5 ms/round -> Task
5b SKIP.** Draft attention is 1.7% of the ungated round; retuning FD2_NS for the ntok=1 draft
cannot pay for a canonical re-derive.

---

## Step 7 -- prefill decay attribution, 16K vs 128K (feeds a future plan)

`--nll synthtoks.bin --kvstats N --ctx N` under nsys `--trace=cuda` (batched prefill;
timing printed before the fp8 KV-scan refusal; production default g64, no Q27_PF_XG).

- 16K: 5.12 s, **3201 t/s** (matches BUILDLOG 3180). Kernel time 5.10 s.
- 128K: 76.34 s, **1717 t/s** (BUILDLOG g64 71.5 s / 1834; nsys adds ~7%). Kernel time 75.91 s.

| bucket | 16K | 128K |
|---|---:|---:|
| prefill attention (`k_attn_prefill_mma`) | 0.69 s (13.5%) | **41.16 s (54.2%)** |
| GEMM/GEMV weights (`k_gemm_mma_T`+`f16_T`) | 3.41 s (66.9%) | 26.92 s (35.5%) |
| GDN delta/WY scan (`k_delta_wy`)          | 0.54 s (10.7%) | 4.35 s (5.7%) |
| quantize                                   | 0.22 s (4.3%)  | 1.72 s (2.3%) |
| norm/rope/act/embed                        | 0.22 s (4.3%)  | 1.75 s (2.3%) |

**The 3201 -> 1717 t/s decay is driven by prefill attention's O(N^2) term:** attention time
0.69 -> 41.16 s = ~60x for 8x length (quadratic), while GEMM weight-streaming is linear
(3.41 -> 26.92 s = ~7.9x) and GDN is linear (0.54 -> 4.35 = ~8x). Per-token: GEMM ~205 us/tok
flat at both lengths; attention 42 -> 314 us/tok = the entire +268 us/tok decay. At 16K GEMM
dominates (67%); by 128K quadratic attention overtakes it (54%). A future prefill-attention
lever (block-sparse / paged / longer-tile flash-prefill) is where 128K prefill speed lives;
weights and GDN are already length-flat per token.

---

## Go / no-go matrix

| item | gate (plan thresholds) | measured | decision |
|---|---|---|---|
| **Task 5** fd2 lane-innermost grid | R >= 2 GO; R < 1.3 SKIP | **R ~= 4.25** (verify/draft time, BW-bound; linear-in-width) | **GO** |
| **Task 5b** FD2_NS retune (draft) | draft-attn >= 1.5 ms/round @61K | **0.51 ms/round** | **SKIP** |
| **Task 6** fd2 lane-pair fusion | R >= 2 AND Task 5 shipped AND attn still >= ~2x BW floor AND Gabe approval | R>=2 met; rest pending Task 5 | **DEFER** -- re-decide on Task 5 results (Task 5 may capture most of the R~4.25 headroom for free) |
| **Task 7** gate_maxd 6-8 brief | uses measured draft/verify economics | draft step ~0.13 ms attn + GEMV; per-lane verify ~+100 us + ~127 us/verify-attn-instance; P12b depth-5 already loses on low-acceptance docs (BUILDLOG) | **PROCEED (write brief)**; preliminary: GO only for high-acceptance agentic traffic. Needs post-Task-4 (early-exit) draft cost to finalize |
| **nucleus-rewrite follow-on** | sampled-path tax >= 3% | **+1.07% ms/round @61K** | **NO-GO** |

## Surprises vs the BUILDLOG ballparks

- **t/s ran ~13% below the wikitext ballpark at 61K (109.9 vs ~126) purely from prompt
  acceptance** (tok/round 3.28 vs 3.68); ms/round (29.87 vs 29.2) matched. Same-day,
  same-binary, same-prompt gate deltas are the trustworthy comparison.
- **theta=1.0 HURTS at 16K (-3.7%) and theta=0.5 beats theta=1.0 at 61K** on docs traffic --
  the opposite ordering to the wikitext BUILDLOG. The gate is acceptance-sensitive; 0.5 is
  the safer cross-ctx default. Relevant to Task 3/4 default-theta choice.
- **The sampled tax is almost all acceptance, not kernels** (+1.07% ms/round). Kills the
  nucleus-rewrite follow-on; Task 3's gate (acceptance side) is the sampled-path lever, not a
  faster nucleus kernel.
- **fd2 draft/verify split is 0.51 vs 8.68 ms/round** -- verify attention is 17x the draft
  attention, entirely because the 16 attention layers x 5-lane re-stream vs the MTP head's
  4 x 1-lane. This is the whole Task 5 rationale in one number.
- **Blackwell ncu metric drift**: `dram__bytes_read.sum` silently no-collects (use
  `dram__bytes_op_read.sum`); and server prefill MTP-warm fd2 pollutes launch-skip. Direct
  DRAM-byte R is recorded UNMEASURED-DIRECT; the nsys time-proxy stands in and is decisive.

---

## Task 3 -- P12 gate ported to the sampled spec path (measured)

Same 61K docs prompt + a 2K docs prompt, server (bench template), `/v1/completions`,
`max_tokens=384`, 1 warmup + n=3 (spread <0.2%). Sampled = `temperature=0.7, top_p=0.95,
seed=42`. Each `Q27_PMIN` = one server restart (read once at `build_spec_graphs`). t/s /
ms/round / tok/round from the `[req]` log (`tps`, `dec_ms`, `dec/rounds`). Binary = branch
tip with Task 3 (per-width sampled verify graphs + capped accept walk). Baseline re-measured
on this binary (Task 2 shifted timing; the doc's Step-3 80.0 t/s / 2.415 tok/round predates
Task 2 and its tok/round came from `[sample-stats]`, a different counter than `dec/rounds`).

### Sampled (T=0.7), the target metric

| ctx | metric      | ungated | Q27_PMIN=0.5 | Q27_PMIN=1.0 |
|----:|-------------|--------:|-------------:|-------------:|
| 2K  | t/s         | 135.7   | 146.6 (+8.0%)| 120.3 (-11.3%)|
| 2K  | ms/round    | 21.94   | 20.31        | 18.56        |
| 2K  | tok/round   | 2.977   | 2.977        | 2.233        |
| 61K | t/s         | 95.1    | 95.1 (+0.0%) | 91.0 (-4.3%) |
| 61K | ms/round    | 30.38   | 25.73        | 23.98        |
| 61K | tok/round   | 2.887   | 2.446        | 2.182        |

**Perf-gate verdict: NOT MET at 61K.** The gate criterion (best theta >= +2% at 61K-class
ctx, no 2K regression > 1.5%) fails: best sampled theta @61K is 0.5 at **+0.0%**. At 2K,
theta=0.5 is a clean **+8.0%** (tok/round IDENTICAL 2.977 -> the narrowing only skipped
would-be-rejected lanes, pure per-round savings) and does not regress. theta=1.0 over-narrows
and regresses at both ctx.

Mechanism: the gate narrows verify width from the DRAFTER's top1-top2 margin. For greedy
verify, a low-margin draft is usually argmax-rejected anyway, so narrowing loses nothing.
For SAMPLED (rejection) verify, a low-margin draft still has a real chance of being accepted,
so at 61K the narrowing cuts accept-able lanes -> tok/round drops 2.887 -> 2.446, the extra
rounds (133 -> 157) exactly offset the cheaper round (30.38 -> 25.73 ms), netting +0.0%. At
2K the acceptance profile happens to align (stream unchanged), so it is pure win. The gate is
acceptance-(prompt-)dependent -- consistent with the Step-2 greedy finding.

### Greedy cross-check @61K (same binary) -- machinery is healthy

| 61K greedy | ungated | Q27_PMIN=0.5 | Q27_PMIN=1.0 |
|------------|--------:|-------------:|-------------:|
| t/s        | 116.1   | 119.5 (+2.9%)| 123.8 (+6.6%)|
| ms/round   | 29.28   | 26.55        | 24.42        |
| tok/round  | 3.398   | 3.174        | 3.024        |

The greedy gate still delivers its win (+6.6% at theta=1.0) on this binary, proving the
shared gate machinery (draft graph + margin readback + cap loop + per-width verify) is NOT
regressed by Task 3. The sampled +0.0% is therefore a genuine rejection-acceptance effect,
not an implementation bug. (Note the greedy vs sampled theta ordering INVERTS: greedy prefers
theta=1.0 at 61K, sampled prefers 0.5 -- the aggressive gate that helps greedy hurts sampled.)

Deep-ctx aside: `prompt16k.txt` actually tokenizes to ~94K tokens (mislabeled by Task 1);
sampled gated theta=1.0 there collapses to 1.272 tok/round / 53.1 t/s -- theta=1.0 degrades
monotonically with ctx on sampled decode.

### Takeaway for Task 4

Task 3's verify-side gate is neutral for sampled 61K docs traffic. The remaining sampled
lever is the DRAFT side: Task 4 (per-step draft early-exit) stops DRAFTING at the first
sub-theta margin, saving ~1.5 ms/draft-pass regardless of whether the verify would have
accepted -- a cost the verify-only gate cannot touch. Task 3 is the necessary substrate
(per-width sampled verify graphs + capped accept walk) for that. Default remains ungated
(`Q27_PMIN` unset), so this change is zero-risk to production sampled traffic.
