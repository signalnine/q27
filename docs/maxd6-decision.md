# maxd6 decision brief: raise gate_maxd to 6 (speculatively 7-8)?

> **OUTCOME: the 2026-07-08 GO at the bottom was executed same day --
> auto-ladder 4..6 SHIPPED** (BUILDLOG 2026-07-08 "maxd6 build": cctx auto
> +4.2% vs d5 same-harness (the entry's +4.7% headline mixed harnesses,
> corrected in the 07-09 review rerun), byte-identical, canonical EXACT;
> the width-7 lane cost, the last extrapolation, was measured in the
> build). Follow-on maxd7
> (BUILDLOG 2026-07-09): depth-7 machinery BUILT, ships OPT-IN; auto stays
> 4..6 (the width-8 round cost came in ~2x the extrapolation). The 07-07
> NO-GO and its 07-08 reversal below are both preserved as decision history.

2026-07-07. The question: now that P14 landed per-width verify graphs (Task 3),
draft early-exit with the `min(W, md_used)` width-floor top-up (Task 4), fused
draft argmax+margin (Task 2), and the fd2 lane-innermost L2 fix (Task 5), does
raising the confidence-gated draft ceiling `gate_maxd` from its default 4 to 6
(and speculatively 7-8) pay on high-acceptance agentic traffic -- and what
exactly would it cost to build? All numbers same box, RTX 5090, memory OC STOCK,
branch `p14-perf-levers` HEAD c27e353. Companion data: `docs/perf-attribution-p14.md`
(Tasks 1-5), BUILDLOG P12/P12b (`:1286`), burst-depth (`:371`), P14 Tasks 2-5.

## Context -- why the old verdict is stale, and what P14 actually changed

The standing verdict is "do not raise depth" from the 2026-07-04 burst-depth
measurement (BUILDLOG:371): the cost slope was **+6.5 ms/depth** (d4 27.44 ->
d5 33.94 ms/round @16K), breakeven **~3.5 ms/depth**, so fixed d6/d8/d10 lost
-5/-13/-18% throughput despite outstanding deep acceptance (chain survival
92-94% flat to depth 10 on code, mean chain 7.62, p(chain>=10)=54%). P12b
(BUILDLOG:1317) confirmed it live: opt-in `Q27_MAXD=5` was **+2.6% on agentic
3.6K but -8% on docs 60K**, because that path always drafted to `gate_maxd`, so
the 5th MTP pass was pure cost whenever acceptance was low.

That +6.5-ms/-breakeven-3.5 model assumed **ungated, fixed-depth drafting always
to max, with full-width verify every round.** P14 changed both sides of the
ledger:

- **Draft side (Task 4).** Early-exit stops drafting at the first sub-theta
  margin. A round only pays for the deep draft steps when the drafter's leading
  margin run is that deep -- i.e. roughly when the deep lane is going to be
  accepted. The cost is now *correlated* with the payoff instead of flat.
- **Verify side (Task 3 + P12).** Verify width follows the cap
  (`verify_graph_w[cap+1]`), so a shallow round runs a narrow verify, not always
  width-5/6. Narrowing is bitwise-free for greedy (independent grid lanes).
- **Both sides cheaper (Tasks 2, 5).** The dead `k_margin` scan is fused away
  (-0.545 ms/round), and verify fd2 per-instance dropped ~10% from the L2 fix.

So the honest question is no longer "does fixed depth-6 pay" (it does not) but
"under early-exit + width-gated verify, does *raising the ceiling* pay on the
rounds where the drafter is confident that deep?"

**New measurement taken for this brief** (61K docs, greedy, server, n=3 medians,
`Q27_MAXD` x `Q27_PMIN` x `Q27_DEXIT` sweep; commands + raw in
`scratchpad/maxd_probe.sh` / `maxd_probe_results.txt`). This is the first time
the depth-5 ceiling has been run *with* the full P14 machinery (P12b measured it
pre-early-exit):

| config (61K docs greedy) | t/s | ms/round | tok/round | rounds | dec wall | vs d4-gate |
|---|---:|---:|---:|---:|---:|---:|
| **d4 θ0.5 dexit1 (current default)** | **124.7** | 25.45 | 3.174 | 121 | 3079 ms | ref |
| d5 θ0.5 dexit1 (early-exit) | 120.6 | 27.46 | 3.310 | 116 | 3185 ms | **-3.3%** |
| d5 θ0.5 dexit0 (monolithic = P12b-like) | 116.9 | 28.32 | 3.310 | 116 | 3285 ms | -6.3% |
| d5 θ1.0 dexit1 (tight gate + early-exit) | 123.8 | 24.42 | 3.024 | 127 | 3101 ms | -0.7% |
| d5 θ1.0 dexit0 | 117.1 | 25.82 | 3.024 | 127 | 3279 ms | -6.1% |

Three facts this pins, all decision-relevant:

1. **Early-exit is real but partial.** dexit1 vs dexit0 recovers **+3.2%** (θ0.5)
   / **+5.7%** (θ1.0) of the deep-draft tax -- more at the tighter gate, where
   more rounds fail early and skip the 5th draft. This halves P12b's docs
   penalty (dexit0 -6.3% -> dexit1 -3.3% at θ0.5; -6.1% -> -0.7% at θ1.0).
2. **Depth-5 still loses on docs even with everything P14 built.** -3.3% (θ0.5),
   -0.7% (θ1.0). The deep lane *does* land tokens on docs (tok/round 3.174 ->
   3.310, +4.3%) but the wall grows faster (+3.4%), so net throughput drops. The
   marginal token at depth is bought below the machine's average rate.
3. **The old "don't raise depth" verdict is softened, not overturned, for
   non-agentic traffic.** P14 roughly halved the slope, but depth-5 is still
   net-negative on docs. Depth-6 (one draft + one verify lane deeper) is
   necessarily worse on the same traffic.

The entire GO case therefore rests on **agentic traffic having materially higher
deep-lane acceptance than docs** -- which the burst data suggests is real for raw
code chains but which has **never been measured at depth-5 with the P14
machinery on live agentic serving traffic** (see UNMEASURED).

## Measured post-Task-4 economics (the breakeven inputs)

All @61K from `docs/perf-attribution-p14.md` Step 4 unless noted.

### Cost of one draft step

| component | ms/round (over 4 drafts) | ms/step | source |
|---|---:|---:|---|
| draft/single GEMV (MTP block + lm_head) | 3.11 | 0.78 | attribution Step 4 |
| draft attention (fd2 z=1) | 0.51 | 0.128 | attribution Step 4 (4 x 127.8 us) |
| fused draft scan (k_argmax_top2, post-Task-2) | ~0.30 | ~0.075 | Task 2 (one full-vocab pass) |
| early-exit D2H margin sync | -- | ~0.035 | Task 4 (~30-40 us/step) |
| **one draft step** | | **~1.0 ms** | |

Cross-check: Task 4 measured DEXIT=1 vs =0 = **-0.77 ms/round** @61K θ1.0 (d4
ceiling), and this brief's live d5 θ1.0 saved **-1.40 ms/round** (deeper ceiling
= more drafts to skip). Net saving per *skipped* draft ~0.7-0.9 ms (the first
failing draft still runs, and the width-floor top-up adds one launch at cap==0).
The burst entry's "~1.5 ms/draft pass" was an older 16K, pre-fusion measurement;
**~1.0 ms/step is the current post-fd2/post-fusion number.**

### Cost of one verify lane (width w -> w+1)

The verify attention (fd2) increment is the dominant per-lane term. Measured
per-instance, gated widths @61K (attribution Step 4, PRE-Task-5):

| verify width | fd2 per-instance | increment | x16 layers/round |
|---:|---:|---:|---:|
| 2 | 256 us | -- | |
| 3 | 371 us | +115 us | +1.84 ms |
| 4 | 465 us | +94 us | +1.50 ms |
| 5 | 539 us | +74 us | +1.18 ms |

The increment *decreases* with width (L2 absorbs a growing minority). Task 5's
lane-innermost fix dropped verify per-instance a further ~10% (542 -> 487 us at
width-5). The batched verify GEMV adds little per lane -- weights are streamed
once and shared across lanes (gemv10 micro-ratio 0.91: 1x10 lanes = 0.91x of
2x5), so a lane past 5 adds ~0.3-0.5 ms/round of compute on already-resident
weights (burst: "~16% of nb5 cost per lane past 5"). Extrapolating the two
curves (attention increment decaying ~+55/+45 us/instance for w5->6/w6->7,
post-Task-5 -10%, x16, plus ~0.3-0.5 GEMV):

- **per verify lane, w5->w6 ~ +1.0-1.2 ms/round; w6->w7 ~ +0.9-1.1 ms/round.**
  **UNMEASURED** -- width-7 fd2 is not instantiated (gemv ceiling is `case 6`,
  gate_maxd is hard-clamped to <=5 at engine.cuh:892), so no width-7 verify
  graph has ever run. These are extrapolations from width<=5 data.

## Breakeven model

**Under early-exit, raising the ceiling from d5 to d6 is free on every round that
does not reach cap>=5, and identical to d5 there.** The only rounds affected are
those whose leading margin run reaches 5 (the drafter is confident to depth 5).
On such a round, d6 adds:

- one 6th draft step: **~1.0 ms** (always, since the round drafted that deep),
- one 7th verify lane when the 6th margin also passes (cap=6, verify width 7):
  **~1.0 ms** (UNMEASURED, extrapolated),
- and on the fraction where the 6th margin *fails* (cap stays 5), it pays the
  ~1.0 ms draft for nothing.

Marginal cost per fired (cap>=5) round: **~2.0 ms** when the 6th draft is
accepted (the productive case), ~1.0 ms wasted when it is not.

Marginal benefit: with probability `p_acc6` (accept prob of the 6th draft on the
rounds that got that deep), commit **+1 token** this round.

The extension pays *on a fired round* iff its marginal tokens-per-ms beats the
round's baseline tokens-per-ms:

    p_acc6 / 2.0 ms  >  tok/round / ms/round

- **High-acceptance agentic (burst-derived).** Raw MTP chain survival is 92-94%
  to depth 10 on code, so `p_acc6 ~ 0.92` if fired rounds inherit that rate.
  Marginal = 0.92 / 2.0 = **0.46 tok/ms**, vs 61K baseline ~0.11-0.13 tok/ms.
  **~4x the machine average -> strongly positive on fired rounds.** The KV and
  weights are already streamed; the marginal deep draft+verify is cheap relative
  to spinning a fresh round.
- **Docs (measured, d4->d5 proxy).** The live probe *is* this calculation one
  depth down: +0.136 tok/round for +2.01 ms/round (θ0.5, dexit1) = **0.068
  tok/ms**, below the round's 0.125 tok/ms baseline -> **net loss** (the observed
  -3.3%). The deep lane's served acceptance on docs is too low to clear breakeven.

**The pivot is `p_acc6` on the fired rounds, which flips the sign.** The breakeven
served-acceptance for a fired round is `p >= (tok/round / ms/round) * 2.0 ms`
~= 0.25 at 61K. Docs delivers below that at depth-5; code *raw* chains deliver
0.92 -- but the served, gated, margin-truncated rate at depth-5 on real agentic
traffic is the unmeasured quantity (raw MTP acceptance overstates it, because the
theta gate truncates the chain earlier than the model would).

**Contrast with the old ungated model.** Old: +6.5 ms/round paid on **every**
round (ungated always drafts to max, full-width verify), breakeven 3.5, so d6 =
-5%. New: the ~2.0 ms marginal is paid **only on cap>=5 rounds** and is
*correlated with the token landing*. The +6.5/breakeven-3.5 arithmetic is retired
for the gated+early-exit path; it only ever described ungated fixed depth. But
the live probe shows the correlation is not tight enough on docs to turn depth
positive -- P14 shrank the penalty (halved it), it did not invert it.

## Implementation cost -- what a depth-6 build actually touches

`gate_maxd` is hard-clamped `<=5` today (engine.cuh:892); the batched verify
pipeline is hardwired at **6-lane capacity** (width-6 = the P12b d5 ceiling + 1
bonus lane). Raising to depth-6 is a **full 6->7 lane widening** of that pipeline
-- structurally the same change as P12b's 5->6, one lane wider, and it will hit
the same landmines. P12b (BUILDLOG:1337) took 4 commits and the quantize3
lane-count bug cost a whole session. Budget one P12b-sized effort **per depth
increment** (d6, then d7, then d8), or one larger up-front refactor (below).

Checklist for d6 (`gate_maxd=6`, perm mod-7, verify width 7):

1. **Depth clamp.** engine.cuh:891-892 raise the cap 5 -> 6 (and the
   `Q27_MAXD=auto` promote ceiling `cur_maxd` max, engine.cuh:1150). Trivial, but
   gates everything below.
2. **perm mod-6 -> mod-7.** 4 real modulo sites -- engine.cuh **208, 217**
   (`SBuf`/`RBuf`: `(role+perm)%6`) and **1167, 1263** (`perm = (perm+(n-1))%6`).
   The 3 hits in `prefill.cu` (184/196/821) are PTX operand refs (`%6`), NOT
   modulo -- ignore. Plus every graph-array perm dimension `[6] -> [7]`:
   `spec_graph`, `draft_graph`, `verify_graph`, `draft_graph_lo`,
   `spec_sample_graph` (all `[6]`), `verify_graph_w[7][6] -> [8][7]`,
   `verify_sample_graph_w[6][6]`, `draft_step_graph[5][6] -> [6][7]`.
3. **7th GDN state buffer: `S_spare6`/`ring_spare6`. +~157 MB** (48 GDN layers
   [64 total - 16 attn] x [3.0 MiB S-state (GDN_HEADS 48 x GDN_DIM 128 x 128 x 4)
   + 0.12 MiB ring]). Touch: the ternary chains SBuf/RBuf (engine.cuh:209-214,
   218-223), the alloc loop (425-434), the two memset loops (918-928, ~1597), and
   the snapshot/restore paths. d7/d8 add `S_spare7`/`S_spare8` at the same
   +157 MB each -> **d6/d7/d8 = +157/+314/+471 MB**.
4. **Batched verify pipeline 6 -> 7 lanes (the bulk).** ~19 distinct 6-lane
   buffer bundles (`x1_f`, `qkv_f`, `z_f`, `alpha_f`, `betar_f`, `g_f`, `kbuf_f`,
   `vbuf_f`, `qg_f`, `og_f`, `o_f`, `y_f`, `h_f`, `ffn_g_f`, `ffn_u_f`,
   `attnout_f`, `convout_f`, `beta_f`, `d_pos_f`) across ~66 reference sites, each
   needs a 7th `_g` lane + its alloc. Signatures widen 6 -> 7 params: `mm5`
   (engine.cuh:638, add `out_g`), `qx5`->`qx6`, `gemv_f16_3`, `gdn_gates3`. GEMV
   template needs **`case 7`** in `k_gemv_q4_n`/`q8_n` (kernels.cu:330-354; today
   `case 2..6,10`). `logits2` 6*VOCAB -> 7*VOCAB (+~1 MB), flash-decode `scratch`
   6 -> 7 lanes (engine.cuh:276,310). Warm the width-7 graphs so capture never
   lazy-compiles (engine.cuh:934).
5. **quantize3-class lane audit (run now).** `grep -n "t==3\|t == 3\|?n3:n4"
   src/*.cu*` finds the real landmine at **kernels.cu:449-452**: `quantize3`
   selects its per-lane output by explicit pointers
   `t==0?n0:...:t==4?n4:n5` (6 lanes, n0..n5). At ntok=7 lane 6 falls through to
   n5 and silently overwrites lane 5 -- this is *exactly* the P12b bug class
   (memcheck-blind: valid buffer, wrong lane). MUST add `n6/e6/s6/i6` and the
   `t==5?n5:n6` branch. The other grep hit, test_kernels.cu:1473, is a comment
   (false positive). Manually audit every batched kernel's per-lane select while
   here: the `k_finish_sampled` `src = n==5?x1e:...:x1a` (blocks.cu:737) is the
   same explicit-pointer pattern (sampled path -- see item 6).
6. **Sampled path stays capped at 4 (separate line item).** `k_spec_accept`,
   `k_finish_sampled`, and the `x1a..x1e` fanout (blocks.cu:637/728,
   blocks.cuh:139/148) are hardwired to `max_draft<=4` / `n<=5`. The greedy
   depth-6 build does NOT widen sampled -- it stays depth-4 (its ceiling is fixed
   at 4 today). Widening sampled to depth-6 = a second P12b-class change: `dr5`,
   `dr6`, `x1f`, `x1g`, `d_nuc` 5 -> 7 lanes (engine.cuh:290), and the
   `k_finish_sampled` src-select extended. Do not conflate the two.
7. **P12b bisect recipe (for when d6 diverges).** `Q27_PMIN=100` forces width-2
   (PASS = the narrow path is clean); `cap<=4` via a low theta forces width<=5
   (PASS narrows the blast radius to width-6/7 only); `Q27_FD=v1` rules attention
   in/out. A divergence that survives `Q27_FD=v1` and appears only at width>=7 is
   a lane-count landmine (item 5). Do not fix forward past a canonical mismatch.

**Cheaper alternative if going past d6:** replace the explicit `_a.._g` lane
fanout with a pointer-array (`float* lanes[MAXW]`) throughout mm/qx/gdn/quantize.
Larger up-front change, but it kills the quantize3 landmine class permanently and
scales to any width in one shot -- worth it if d6/d7/d8 are all wanted, not worth
it for d6 alone.

## Comparison

| | fixed gate_maxd=6 | adaptive auto -> 6 | Task 6 lane-pair fusion |
|---|---|---|---|
| who it helps | all traffic (taxes non-agentic) | agentic streaks only (EMA demotes elsewhere) | **all traffic, every round** |
| measured today | docs d5 already -3.3% (θ0.5); d6 worse | untested at d6 | Task 5 got +2.7%; ~90% of R~4.25 headroom (~6 ms/round) remains |
| upside | agentic-only, conditional, UNMEASURED | agentic-only, self-limiting, UNMEASURED | universal; residual verify-attn is ~6 ms/round |
| downside risk | high (fixed ceiling taxes docs/math) | low (auto demotes on low yield) | med (expensive kernel rewrite; occupancy gate) |
| VRAM | +157 MB | +157 MB | ~0 |
| build effort | 1 P12b-class 6->7 lane widening + quantize3 landmine | same + P13 EMA range bump | fd3 design doc + paired-lane kernel (Gabe-gated) |
| depends on traffic mix | yes (bad) | yes (managed) | no |

## Recommendation

**NO-GO on a fixed `gate_maxd=6` default. GO-IF, narrowly, on extending the
`Q27_MAXD=auto` ceiling to 6 -- gated on one cheap agentic measurement that
nobody has taken. Do the measurement before either build; if a build must be
chosen, do Task 6 first.**

Rationale, in order:

1. **Fixed depth-6 is NO-GO.** The live probe settles it: with *all* of P14's
   machinery, depth-5 already loses **-3.3%** on docs (θ0.5) and only reaches
   -0.7% at the tight gate. Depth-6 is one draft + one verify lane deeper on the
   same traffic -- necessarily worse. A fixed ceiling cannot tell agentic from
   docs/math, so it taxes the mixed production stream. P14 halved the depth
   penalty; it did not invert it.

2. **The only viable form is adaptive.** The P13 EMA (promote 4->5 when
   `sat_ema >= maxd_hi=0.50`, demote when `yield_ema < maxd_lo=0.10`) is precisely
   the mechanism that confines deep drafting to acceptance streaks and demotes
   back on docs -- exactly where the fixed ceiling fails. A depth-6 build should
   ship *only* as an `auto` ceiling bump (4..6), never a fixed default.

3. **GO-IF condition (pick X/Y from the rig, not from raw acceptance).** Extend
   the auto ceiling to 6 iff a `--stats` run on **real agentic serving traffic**
   (the P12 Phase 0 rig) shows both:
   - **X: cap>=5 (margin-run-depth reaches 5) on >= 30% of gated rounds** -- below
     this the +157 MB and P12b-class build move too few rounds to matter; and
   - **Y: depth-5 `sat_ema` >= 0.50 sustained** (P13's own promote threshold) AND
     a depth-5-vs-depth-4 A/B on that traffic is **net positive** (the docs A/B in
     this brief is net *negative* at -3.3%, so this is a real gate, not a
     formality).
   If X/Y hold, the fired-round economics (0.46 vs 0.11 tok/ms) make d6 a clear
   agentic win; if they do not, depth stays at the d4/d5-auto optimum.

4. **Sequencing: measure before you build; Task 6 before depth-6.** The entire
   depth-6 case rests on `p_acc6` at depth-5 on agentic traffic, which is
   UNMEASURED -- and the one traffic class we *can* measure (docs) says no. So the
   cheapest, highest-information next step is not a build at all: it is the
   agentic depth-5 A/B + margin-run-depth histogram (~1 session on the existing
   Phase 0 rig, zero engine risk). If a *build* must be picked, **Task 6
   (lane-pair fusion) first**: it targets the ~6 ms/round verify-attention
   residual (R~4.25, Task 5 captured only ~10%), it helps **every round of every
   traffic class** (not just agentic streaks), it costs ~0 VRAM, and by shrinking
   the per-verify-lane cost it *lowers depth-6's own breakeven* -- making a later
   depth-6 cheaper to justify. Depth-6 is a large, conditional, agentic-only bet
   whose payoff is unproven at depth with the current machinery; it should come
   after both the measurement and the universal lever, not before.

## UNMEASURED (what would settle it)

- **`p_acc6` / margin-run-depth distribution at depth-5 on real agentic serving
  traffic.** The whole sign of the decision. Burst gives *raw* MTP chain survival
  (92-94% to d10 on code), but the served, theta-gated, margin-truncated rate at
  depth-5 is lower and unmeasured. Settle with a depth-5 `--stats`/`--burst-stats`
  run on live CC/CRUSH agentic traffic (P12 Phase 0 rig): report the fraction of
  gated rounds reaching cap>=5 (the X threshold) and the depth-5 vs depth-4 A/B
  t/s (the Y net-positive gate).
- **Width-7 (and 8, 9) verify-lane cost.** gate_maxd is clamped <=5 and the gemv
  ceiling is `case 6`, so no width-7 verify graph has ever run. The +1.0 ms/round
  per-lane figure is extrapolated from the decaying width<=5 fd2 increments
  post-Task-5. Settle by instantiating width-7 (item 4) and running the forced-cap
  sweep (`Q27_PMIN` at the depth-6 build).
- **Auto-ceiling behavior on mixed traffic at depth-6.** The comparison assumes
  the P13 EMA demotes fast enough to keep the docs -3.3% off the mixed stream at
  depth-6; only tested at 4..5 today. Settle with an `Q27_MAXD=auto` docs vs
  agentic A/B once the depth-6 ceiling exists.

## 2026-07-07 MEASURED -- GO-IF evaluated: **NO-GO**

The UNMEASURED numbers above were taken same-day (telemetry 42ccf6d: `gch`/`gnh`
gated-round histograms in the `[req]` log; full data in BUILDLOG "maxd6 GO-IF
measurement" + results/q27-maxd6-*.log).

- **X PASS**: cap>=5 on **79.3%** of 5336 gated rounds (one full T8 trial, real CC
  traffic, ctx to 81K, score 0.796).
- **Y saturation PASS**: depth-5 n=6 on **65.2%** (>=0.50); p(5th lane | fired) 82.2%;
  5.03 tok/round.
- **Y throughput FAIL**: identical-request 26K replay A/B, d5 vs d4 (theta 0.5, dexit1):
  repro +2.9% (97.6% fired), T8-style codegen **-5.4%** (56% fired), fresh testgen
  **-3.9%** (52% fired). Interpolated at the live operating point (79% fired / 82%
  yield): **-0.8%..+0.1%**. Depth-5 is breakeven at best on live-matched agentic
  traffic; it wins only in near-verbatim (>~90% fired) regimes.

**The breakeven model in this brief is refuted by the measurement.** The ~2.0 ms
fired-round marginal understated true cost 1.5-2x (measured +2.1..+3.8 ms/round
d4->d5 across all rounds): the theta gate predicts drafter CONFIDENCE, not verifier
ACCEPTANCE -- on codegen traffic 44% of fired rounds waste the deep lanes. The 0.46
tok/ms "strongly positive" agentic estimate does not survive contact with served
traffic; the measured net at 79%/82% is ~0.

**Decision: NO-GO on extending the auto ceiling to 6.** With depth-5 at breakeven on
the very traffic that fires it hardest, a strictly-deeper lane on a smaller fired
fraction cannot pay for the P12b-class 6->7 widening + 157 MB. Ceiling stays 4/5-auto.
Retry bar: a gate that predicts acceptance rather than confidence, or a materially
cheaper verify lane. Side finding: P13's HI=0.5 promote threshold lands exactly at the
measured win/loss crossover (45%-sat traffic loses -5.4%, 96%-sat wins +2.9%).

## 2026-07-08 RERUN on refreshed economics -- GO-IF conditions now MET: **GO, narrowly (auto-ladder-6)**

Retry bar satisfied on both prongs since the NO-GO: verify-gemv made the lane cheaper
(+5.9% decode) and accept-gate Phase 1 shipped the acceptance-tracking bar
(conditional lane-5 yield, lo=0.35 = measured crossover). Full data: BUILDLOG
2026-07-08 accept-gate Phase 0 + this rerun; rig tools/burst_sim.py + accept_ab.sh.

**New instrument.** --burst-stats CSVs (10-deep chains + margins per free position)
drive an exact offline ROUND simulation at any ceiling/theta (tools/burst_sim.py) --
round-sampled, killing the per-position discount. Caveat discovered en route: chains
seeded from serial-path hiddens differ in ULPs from live verify-lane hiddens; deep
chains amplify near-tie flips, so the sim UNDERESTIMATES tok/round, mildly on low-sat
flavors (echo -5% vs CLI ground truth) and severely on echo-heavy ones (cctx -27%).
Sim positives are trustworthy; sim negatives on high-sat flavors are not
decision-grade. Emitted-token identity across serial/spec/server paths reconfirmed
throughout (the divergence lives only in draft chains near theta/argmax ties).

**The missing payload found.** A real CC transcript replayed raw (cctx, 25.8K tok,
built from a thunderdome bench session transcript; NOT committed -- private) finally
reproduces the live-T8 profile: sat5 0.714 (live T8 0.652), 5.29 tok/round (5.03).
Measured CLI legs: d4 204.1 t/s (4.56 t/r, sat4 .807) -> d5 218.5 t/s (5.29 t/r) =
**+7.0%** -- the leg class that measured -0.8..+0.1% pre-verify-gemv now clearly PASSES Y2.

**GO-IF conditions, re-evaluated:**
- X (cap>=5 on >=30% of gated rounds): PASS -- live T8 79.3% (07-07 histograms,
  unchanged) + cctx fired5 >= sat5 .714.
- Y1 (depth-5 saturation >= 0.50): PASS -- live T8 65.2%; cctx 71.4%.
- Y2 (d5-vs-d4 net positive at the live point): PASS -- cctx +7.0% measured;
  26K envelope +0.2..+5.6% (accept-gate Phase 0); only 61K low-yield docs negative
  (-1.7%), which the Phase-1 bar demotes out of depth-5 anyway.

**Depth-6 estimate (measured-sat extrapolation, cctx):** per-level sat decay
.885 (=.714/.807); d4->d5 tok/round gain +0.73 ~= sat5 .714 validates the model
(each new top lane lands a token on rounds that saturate one level deeper).
d6: +0.63 tok/round at dms +1.6-1.9 (width-7 lane EXTRAPOLATED from the decaying
fd2 increments; still never instantiated) -> **+4-5% t/s on CC-flavor traffic**.
Constructed low/mid-sat payloads: ~0 (sim and extrapolation agree; echo +0.7%,
docs -1.7..testgen +3.7% sim, all inside the bias band). Ladder confinement:
promote 5->6 at sat5 >= HI=0.5 admits ONLY cctx-class traffic (echo .26, codegen
.16, testgen .26, docs .46, docs61k .12 all stay at 5); demote on y6 < bar_6
(~0.35, same breakeven structure) bounds any misfire to a grace window.

**Decision: GO -- as an `auto` ladder extension 4..6 ONLY (never a fixed default),
one P12b-class build session.** Scope per the 2026-07-07 checklist items 1-7
(6->7 lane widening, S_spare6 +157 MB, quantize3-landmine audit, gemv case 7,
perm mod-7) + a level-6 pair in depthctl (unit-testable). Build-phase gates before
any default: (a) measure the width-7 verify lane via forced-cap sweep (kills the
last extrapolation), (b) canonical 4c4120c7 EXACT + token identity + round
determinism, (c) replay A/B on cctx + the 26K envelope: SUCCESS = cctx >= +3% with
no constructed payload below its d5 baseline; (d) glf/gla extended to lane 6 so the
live yield is observable from day one. d7/d8: NOT in scope -- re-derive from live
lane-6 telemetry after d6 ships (sat-decay extrapolation says marginal-positive on
cctx, but margin-run truncation at depth is the unmodeled term).

Expected value honestly stated: +4-5% on the traffic q27 actually serves (CC
agentic), ~0 elsewhere, ladder-protected; also the first step toward the P4-echo
+45% tail that llama's depth-10 still owns.
