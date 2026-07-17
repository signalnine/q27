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
| **Task 6** fd2 lane-pair fusion | R >= 2 AND Task 5 shipped AND attn still >= ~2x BW floor AND Gabe approval | R>=2 met; **Task 5 shipped +2.7% (marginal) -- captured only ~10% of the verify per-instance time, so most of the R~4.25 headroom REMAINS** | **DEFER (Gabe's call)** -- Task 5 did NOT capture most of the headroom; the lane-pair fusion still has a real target, but it is the expensive kernel rewrite and needs explicit go |
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

---

## Task 4 -- draft early-exit (margin-gated per-step draft graphs, `Q27_DEXIT`)

Same harness as Task 3 (same server template, same docs prompts, `/v1/completions`
`max_tokens=384`, 1 warmup + n=3 medians, sampled = `temperature=0.7, top_p=0.95, seed=42`,
fields from the `[req]` log). Binary = branch tip with Task 4 (per-step draft graphs +
early-exit loop in both gated branches). A/B lever: `Q27_DEXIT=1` (early-exit, default when
gated) vs `Q27_DEXIT=0` (monolithic gated draft = the Task-3 behavior) -- one server restart
per (theta, dexit) config. Baselines re-measured on THIS binary per the plan's measurement
rigor (see the session-drift footnote below).

### Identity gate (plan Step 5) -- ALL IDENTICAL

For every cell of theta {0.5, 1.0} x ctx {2K, 61K} x {greedy, sampled seed 42}, 3 runs each:
`Q27_DEXIT=1` vs `Q27_DEXIT=0` emitted bytes IDENTICAL and round counts IDENTICAL
(rounds: 61K g 131/121, s 180/152; 2K g 176/167, s 173/167 for theta 1.0/0.5). The sampled
identity is exact as predicted (same drafts, same caps, same Philox keys). A `Q27_MAXD=auto`
greedy CLI smoke (2K, theta=1.0) is also byte-identical at 46 rounds both -- the P13 EMA sees
identical n and md_used per round on both settings (caps identical), so adaptive-maxd
trajectories cannot diverge.

### Perf: Q27_DEXIT=1 vs =0 (n=3 medians, docs prompts)

| theta | ctx | mode | DEXIT=1 t/s | DEXIT=0 t/s | delta | ms/round (1 vs 0) | tok/round | rounds |
|------:|----:|------|------------:|------------:|------:|-------------------|----------:|-------:|
| 1.0 | 61K | greedy  | **122.1** | 118.3 | **+3.2%** | 24.02 vs 24.79 (-0.77) | 2.931 | 131 |
| 1.0 | 61K | sampled | **95.0**  | 90.1  | **+5.4%** | 22.46 vs 23.69 (-1.23) | 2.133 | 180 |
| 1.0 | 2K  | greedy  | 130.6 | 123.3 | +5.9% | 16.70 vs 17.69 (-0.99) | 2.182 | 176 |
| 1.0 | 2K  | sampled | 125.5 | 119.5 | +5.0% | 17.69 vs 18.58 (-0.89) | 2.220 | 173 |
| 0.5 | 61K | greedy  | 121.1 | 119.4 | +1.4% | 26.21 vs 26.58 (-0.37) | 3.174 | 121 |
| 0.5 | 61K | sampled | **100.5** | 97.6 | **+3.0%** | 25.13 vs 25.89 (-0.76) | 2.526 | 152 |
| 0.5 | 2K  | greedy  | 129.2 | 124.3 | +3.9% | 17.80 vs 18.50 (-0.70) | 2.299 | 167 |
| 0.5 | 2K  | sampled | 122.6 | 118.5 | +3.5% | 18.75 vs 19.40 (-0.65) | 2.299 | 167 |

**Plan gates:** greedy 61K theta=1.0 **+3.2% >= +3% MET**; theta=0.5 positive (+1.4%) MET.
tok/round and rounds are IDENTICAL between DEXIT settings in every cell (cap semantics
unchanged) -- the entire delta is per-round draft cost, exactly the design.

**2K note (positive surprise, not a gate miss):** the plan expected 2K "neutral (+-1.5%)"
on the nothing-to-skip assumption, but the docs 2K prompt is LOW-acceptance at these thetas
(tok/round 2.18-2.30 => many cap-0/1 rounds => 2-3 drafts skipped/round of a ~18 ms round),
so early-exit pays +3.9..+5.9%. The honest "nothing to skip" neutrality check is the
worst-case probe below.

### Worst-case sync-overhead probe (`Q27_PMIN=0.01`, greedy)

Every margin passes theta => the loop always runs all md_used steps with md_used
D2H+syncs/round vs the monolithic's one -- pure added-overhead measurement, nothing skipped:

| ctx | DEXIT=1 | DEXIT=0 | delta |
|----:|--------:|--------:|------:|
| 61K | 115.4 t/s (29.43 ms/r) | 116.0 t/s (29.31 ms/r) | -0.5% |
| 2K  | 115.6 t/s (20.76 ms/r) | 116.1 t/s (20.67 ms/r) | -0.4% |

+0.09-0.12 ms/round = the 3 extra 4-byte D2H+sync round-trips (~30-40 us each). Within the
+-1.5% band -- high-acceptance traffic pays only this. Sync-count delta per round:
monolithic gated = 1 margin sync; early-exit = min(cap+1, md_used) margin syncs (cap=0 pays
1, the top-up adds launches but no sync; worst case md_used = +3 syncs).

### Gated-vs-ungated on this binary (the production question)

Ungated reference re-measured on this binary: 61K greedy 116.4 / sampled 97.0;
2K greedy 115.5 / sampled 116.5 t/s.

| vs ungated | greedy | sampled |
|---|---|---|
| 61K theta=1.0 + dexit | **+4.9%** | -2.1% |
| 61K theta=0.5 + dexit | +4.0% | **+3.6%** |
| 2K theta=1.0 + dexit | +13.1% | +7.7% |
| 2K theta=0.5 + dexit | +11.9% | +5.2% |

**Task 3's sampled wash is resolved:** verify-narrowing alone was +0.0% @61K theta=0.5;
adding draft-side early-exit at the SAME acceptance cost (caps unchanged, tok/round 2.526
either way) takes it to **+3.6% over ungated**. Recommendation: gate ON for both paths with
**theta=0.5 as the cross-path default** (sampled theta=1.0 still nets negative vs ungated
at 61K: the acceptance loss from aggressive narrowing exceeds the draft savings); greedy
tolerates/prefers theta=1.0 at depth (+4.9%). `Q27_DEXIT` should stay default-ON: it is
positive or neutral in every measured cell.

### Session-drift footnote (measurement hygiene)

Round counts for the SAME prompt+theta drift across sessions/binaries (61K greedy theta=1.0:
Task 1 = 134, Task 3 = 127, this session = 131) while each session's A/B cells are internally
exact. The drift pre-dates Task 4 (it exists between Task 1 and Task 3) and correlates with
prefix-cache state (`hit=61002` here), not with any code change -- every comparison in this
section is same-server-config A/B, so it cancels. Do not compare tok/round across sessions;
compare only within a table row.

Commands: `scratchpad/task4_matrix.sh` (identity + perf matrix), `task4_extra.sh` (overhead
probe + auto smoke), `task4_ungated.sh` (ungated reference), all driving
`srv4.sh`/`measure4.py`; per-run texts + rounds in `scratchpad/id_*` for the identity diffs.

---

## Task 5 -- fd2 lane-innermost grid scheduling (bitwise L2 fix)

**GO condition (Task 1 Step 5): R ~= 4.25 >= 2.** The verify fd2 grid was
`dim3(n_kv_heads, FD2_NS, ntok)` -- the token lane (`blockIdx.z`) is the SLOWEST-varying
axis, so all of lane 0's blocks schedule before lane 1's and each verify lane re-streams the
full KV slice from DRAM (zero cross-lane L2 reuse by construction). The KV read address
depends on `(pos, kv_head)` only, NOT on the token lane, so the `vw` same-`(head,split)`
blocks read byte-identical KV.

### The change (spec3.cu, `k_attn_fd2` + `attn_decode3_fd2`)

Pure index remap -- lane becomes the FASTEST-varying axis so the same-`(head,split)` blocks
of all verify lanes co-schedule onto the same ~1MB KV chunk:

```
- launch:  dim3 g1(n_kv_heads, FD2_NS, ntok);
+ launch:  dim3 g1(ntok, FD2_NS, n_kv_heads);
- kernel:  const int kvh = blockIdx.x, sp = blockIdx.y, t = blockIdx.z;
+ kernel:  const int t = blockIdx.x, sp = blockIdx.y, kvh = blockIdx.z;
```

Per-block work, per-lane fp accumulation order, the scratch-cell addressing per
`(head, split, lane)` triple (`part + (pair*FD2_NS + sp)*FD_ST`, `pair = t*(NKV*GQA) +
kvh*GQA + j`), and the combine kernel are byte-for-byte unaffected -- only the block
enumeration ORDER differs. The v1 fallback (`k_attn_fd`, `Q27_FD=v1`) is untouched. The
ntok=1 draft launches share the wrapper (grid `dim3(1, FD2_NS, NKV)`) -- harmless (x=t=0).

### Gates -- ALL PASS

1. **Build.** Full `make`, sm_86 + sm_120: no warnings from spec3.cu (only the pre-existing
   engine.cu/server.cu/tokenizer.cpp warnings).
2. **test_kernels ALL PASS** (0 FAIL): the fd2 gates (fd2-vs-v1 tolerance, run-to-run
   bitwise determinism, default-dispatch bitwise) pass unchanged over the full
   seq {1,47,1024,16384,61440} x ntok {1,5} x {fp8,fp16} matrix.
   **Bitwise-vs-pre-change PROVEN on the FULL matrix** (stronger than the plan's substitute):
   a temporary FNV-1a fingerprint over every fd2 output byte of every matrix cell (both
   dtypes, all ntok lanes) is IDENTICAL pre vs post = `5f0e1d98593d2283` (pre binary built by
   stashing only the spec3 remap, keeping the fingerprint helper; helper reverted before the
   commit so the landed diff is spec3.cu-only). PLUS the plan's named substitute: 61K greedy
   `generated:` text byte-IDENTICAL pre vs post (1308 bytes, 3/3 runs). tok/round and
   round-counts are also identical pre vs post in every bench cell.
3. **Canonical EXACT `4c4120c72056aba2bc2d2561471eafce`** on the pre binary, the post binary,
   and the final committed binary.

### Bench (server, docs prompts, greedy, n=3 medians, spread <0.1%)

Pre baseline re-measured on the saved pre-change server binary
(`scratchpad/pre/q27-server`); post on the committed binary. Same server template
(`Q27_PF_XG=32 --ctx 131072 --no-think --fast-head`), `/v1/completions`, `max_tokens=384`,
1 warmup + n=3, fields from the `[req]` log. tok/round and rounds are IDENTICAL pre vs post
in every cell (bitwise fd2), so the entire delta is per-round attention cost.

| config | ctx | PRE t/s | POST t/s | delta | PRE ms/round | POST ms/round | tok/round | rounds |
|--------|----:|--------:|---------:|------:|-------------:|--------------:|----------:|-------:|
| ungated            | 61K | 116.1 | **119.3** | **+2.7%** | 29.27 | 28.50 | 3.398 | 113 |
| gated theta=0.5+dexit | 61K | 121.8 | **124.2** | +2.0% | 26.06 | 25.55 | 3.174 | 121 |
| ungated            | 2K  | 115.5 | 115.5 | +0.0% | 20.78 | 20.77 | 2.400 | 160 |
| gated theta=0.5+dexit | 2K  | 129.6 | 129.5 | -0.1% | 17.75 | 17.76 | 2.299 | 167 |

**Decision (plan rule on 61K ungated t/s): +2.7% is in the [+1.5%, +3%) band -> KEEP, flagged
MARGINAL.** Not >= +3% (would be an unqualified keep) but well above the +1.5% revert floor.
2K does NOT regress (ungated +0.0%, gated -0.1%, both << the 1.5% floor) -- the empty-split
early-return scheduling is unaffected by the axis order.

### Mechanism confirmation (nsys, direct)

Step-4-style capture on the POST binary (`Q27_PROF_DECODE=1`,
`nsys --trace=cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi
--capture-range-end=stop`, one 61K request, `cuda_gpu_kern_sum`), compared against the Task-1
pre-change fd2 row (same node-traced methodology, so pre/post are apples-to-apples):

| fd2 per-instance (ns) | PRE (Task 1) | POST | delta |
|---|---:|---:|---:|
| Med (verify z=5 dominant) | 542,076 | **487,325** | **-10.1%** |
| Min (draft z=1)           | 125,151 | 128,191 | ~flat (single-lane draft: no cross-lane question) |
| Max (verify z=5)          | 549,788 | 506,109 | -8.0% |

The verify per-instance time drops ~10% toward the draft z=1 floor while the draft z=1 time
is unchanged -- the exact signature of cross-lane KV L2 reuse (co-scheduled verify lanes now
hit L2 for a portion of the KV slice instead of each re-streaming it from DRAM). 16 verify
fd2/round x -54.8 us = **-0.88 ms/round** predicted from the per-instance drop, consistent
with the observed **-0.77 ms/round** ungated round reduction. The reuse is PARTIAL (verify
did not collapse to the draft floor): the 5090's L2 absorbs a minority of the ~63 MB/layer
fp8 KV slice per co-scheduled wave, not all of it -- which is why the win is real but
marginal (+2.7%), not the full R~4.25 headroom. That residual headroom is the Task 6
(lane-pair fusion) target, still DEFERRED pending Gabe's call on this marginal result.
[RESOLVED 2026-07-07/09: Task 6 was built as attn-fd3 and KILLED on its perf gate
(-4.0% @61K vs the >= +5% floor); the whole KV-sharing-restructure family was then
closed with data on 07-09. See docs/attn-fd3-design.md.]

Commands: pre binary from `git stash push -- src/spec3.cu` + `make` (fingerprint helper
retained), saved to `scratchpad/pre/`; server A/B via `scratchpad/srv5.sh BIN PMIN DEXIT` +
`measure4.py`; nsys capture in `scratchpad/nsys_post61k.{nsys-rep,sqlite}`; pre fd2 row in
`scratchpad/kern_ung.csv` (Task 1).
