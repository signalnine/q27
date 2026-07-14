The draft ladder is at its SOL floor. No free win exists; the on-path lever is a gated quality tradeoff whose entire fate rides on one unmeasured number, and the higher-ceiling lever is a separate architectural project. Here is the plan.

---

# Draft ladder: cut the 2.91 ms MTP head+layer stream

2026-07-13. Target: the MTP draft ladder, the dominant non-verify cost on the
novel-prose path (the 224 t/s headline). Not touched by this session's
suffix/verify/delta work. HEAD 90a4974.

## Honest expectation FIRST

**There is no free, bitwise win here. The 2.91 ms is genuine SOL weight-streaming
at 90-94% of the 5090's DRAM ceiling, re-streamed 4x by an irreducible serial
dependency, with the argmax already fused to a 1 MB pass.** The only real levers
trade acceptance (approximate draft head, B/E) or architecture (off-path draft on
the 3090, D). Both are gated by measurements that do not exist yet, and both are
smaller than the headline framing suggests. Arithmetic:

**Byte budget per draft step** (engine.cuh:900-938, all leaf `blk.64.*`):
eh_proj 52 MB + attn q/k/v/o 105 MB + ffn 267 MB = **427 MB Q8 MTP layer**;
head `output_q4.weight` 248320x5120x0.5 = **635 MB Q4** (+~40 MB scales) = **~1.1
GB/step**. x4 serial steps = ~4.4 GB. At the measured ~1580-1687 GB/s that IS the
2.91 ms (head 1.60 ms = 61%, MTP layer 1.10 ms = 39%, attn ~0.21 ms). Reclaiming
only the head caps any on-path win at 61% of 2.91 ms.

**The 12% headline uses the wrong denominator -- correct it before sizing anything.**
The 2.91/24.3 = 12% divides draft cost by a **width-12 round**. Width-12 rounds
are SUFFIX rounds (`sfx_width()` up to `W_MAX=12`), and suffix rounds skip the MTP
chain entirely -- **zero draft cost** (engine.cuh:1530-1548). The 2.91 ms draft
NEVER coexists with a width-12 verify. On a genuine novel *ladder* round the
verify width is `gate_maxd+1 <= 8`, held strictly below `gemm_min=9` by the
build-time abort (engine.cuh:1284), so novel verify runs the per-lane GEMV path,
not the flat `k_vgemm`. That verify is weight-flat regardless: full model 17.7 GB
/ ~1580 GB/s = 11.2 ms floor + delta + narrow attn ~13-14 ms at low ctx. At the
224 t/s headline (tok/round ~3, ms/round ~13.4), the draft is **~22% of a novel
round, the head alone ~12%** -- roughly 2x the headline. At 61K ctx (53-80 t/s,
attention-dominated) the fixed 2.91 ms draft is **<5% of the round** -- nearly
worthless.

**But 2.91 ms is a SATURATING round (all 4 margins passed theta) -- a high-acceptance
round, not typical novel prose.** With `dexit_on` (default true, `Q27_PMIN=0.5`)
the ladder launches steps one at a time and breaks at the first sub-theta margin,
width-floored at 2 (engine.cuh:1561-1595). The realized average is `gs.draft_ms /
gated_rounds` **< 2.91 ms**, and it is already logged (server.cu:450). The true
prize is that realized fraction, a number nobody has read. It sits between the
12% (too low, wrong denominator) and 22% (too high, saturating peak).

**What an on-path head lever can actually claim.** Head is 61% of draft bytes.
Even a *free* head is bounded: at K=32768 the gathered rows are 32768x2560 B ~82
MB < 128 MB L2, so draft steps 2-4 hit L2 (savings beat the linear K/248320
estimate) -- but the **427 MB MTP-layer Q8 does not fit L2** and stays a ~1.10 ms
floor. So the draft floor is ~1.31 ms; max head reclaim ~1.6 ms. Ceiling at the
short-ctx headline ~10-12% (p=0), realistic +4-6% once you pay scatter-below-SOL,
the cold step-1 stream, and recall < 1.

**Acceptance is the whole risk, and it compounds.** A shortlist miss at step k
truncates the ladder at n=k (finish_round stops at the first draft/verify
mismatch). Model it as a truncating geometric, `tok/round = 1 + sum_{k=1..4}
(a*r)^k`, r = per-step probability the true argmax is in-shortlist. At baseline
a=0.85 (tok/round 3.70), break-even against a ~10% ceiling lands at **r ~0.95 per
step (off-shortlist p ~5%)**; r=0.90 loses ~15% tokens (net negative), r=0.98 nets
~+7%. Higher baseline acceptance makes the bar *tighter* (longer forfeited tail).
Novel prose is exactly where the argmax is least predictable and r is lowest, and
the 4 steps predict forward positions the verify never scored, so r **decays with
depth**. The relevant recall is **shortlist-vs-Q4**, NOT the cited 98.1%
Q4-vs-Q8 agreement -- do not reuse that number.

**B collapses into E.** The shortlist needs a per-step SOURCE with high argmax
recall on novel prose. Static (unigram/BPE) is dead (content-driven argmax). The
prev-round verify top-M drifts forward over the 4 steps. The only cheap high-recall
all-vocab scorer *is* a low-rank head -- i.e. B becomes E, and E needs an offline
SVD/distill artifact that **does not exist in-tree** (grep clean; `turbo3` is
KV-only, unwired as a head format).

**Off-path D is the higher ceiling but the wrong project for novel prose.** It
hides 100% of the draft (head AND layer) at zero acceptance cost, verify
untouched. But draft(R+1) consumes `h_next` produced at the END of verify(R)
(spec3.cu:904-916), so overlap requires speculating verify(R)'s outcome n before
it lands -- and the suffix path already strips the draft where n is predictable
(engine.cuh:1530). On novel prose n is least predictable, so hiding fraction is
anticorrelated with the presence of the cost. Ceiling ~20% at headline if
perfectly hidden; realistic at n-hit-rate p_n ~0.2-0.5: +3-9% minus misprediction
penalty, for a weeks-long dual-GPU pipeline on a contended sm_86 3090. Cost it
only if `gate_n_hist` comes back sharply peaked (it will not, on novel prose).

## VERDICT

**Not a free win. A *possible* worth-it quality tradeoff (B/E), currently
NOT greenlit -- gated behind a mandatory zero-cost acceptance probe that most
likely kills it on novel prose.** The draft is at its SOL floor. **DO NOT write
the gathered-GEMV kernel unless the P1 probe returns per-step shortlist-vs-Q4
recall >= 0.97 at draft positions 3-4 and >= 0.95 at positions 1-2, at a K whose
reclaimed head bytes clear >= 4% of the *measured* novel-round ms.** If it does
not, the draft stays at its floor and the only remaining lever is off-path D as a
separate architectural plan -- worth starting only if decode t/s becomes the
headline (it is not today; prefill dominates agentic wall time per
drafter-probe-plan.md) AND `gate_n_hist` is peaked.

## What is already there (no work)

- **The verify escape hatch -- correctness holds under ANY draft head.**
  `k_finish_round` (spec3.cu:904-916) commits `nt = v[n-1]` (verify verdict) and
  carries `h_next = x1s.p[n-1]` (a verify hidden). A bad draft lowers n, never the
  token or the propagated state. Safety is "finish_round commits the verify
  verdict," true under both heads -- NOT "Q8 bitwise" (that holds only on the
  no-fast-head CLI gate; `--fast-head` is the production server default,
  README:465, under which verify is also Q4).
- **The draft head is ALREADY Q4** (engine.cuh:930-933, `output_q4.weight` when
  present). The canonical gate `a2982c5197c627551b27d76a0a94b220`
  (tools/shortbench_suite.sh:28) was produced with a Q4 draft head + Q8 verify
  (no `--fast-head`). A further draft-head change moves along a boundary the gate
  already sits on; it cannot move the md5 unless you flip `--fast-head` (verify
  goes Q4) or exceed the commit modulus (`W_MAX=12 > max n=8`). Both invariants
  hold.
- **Realized draft cost + step count telemetry**: `gs.draft_ms`, `gs.draft_steps`
  accumulated per gated round (engine.cuh:1578-1583), printed at server.cu:450.
  This is the dexit-averaged baseline the sizing needs.
- **Per-lane acceptance + n histograms**: `gch/gnh/glf/gla` (server.cu:475-493,
  behind `pmin_theta>0`). `gate_n_hist` peakedness gates off-path D; `gate_lane_acc`
  prices depth.
- **Adaptive depth is shipped and spent** (depthctl.h k_min=4, cur=4, floats up
  only; dexit early-exit + width-floor-2, engine.cuh:1561-1595). Per-round depth
  is already margin-adaptive, finer than the EMA ceiling.
- **The offline probe harness exists**: `--burst-stats` writes d1..d10 + m1..m10
  per free position (engine.cu:40,58,235), consumed by tools/ladder_price.py and
  tools/burst_sim.py. P1 extends its CSV, no new rig.
- **The gather is trivial to build**: `k_gemv_q4` is warp-per-row, rows fully
  contiguous, group-64 scales indexed by the same `row` (kernels.cu:198-207:
  `wr = W + row*(cols/2)`, `sr = S + row*(cols/64)`). A shortlist fork is
  `row = idx[warp]` -- weights and scales gather together, no repacking, VRAM cost
  `idx[K]*4B ~128 KB`.

## What changes -- file:line checklist

### P1 probe (measurement only; no hot-path kernel)
1. **engine.cuh (draft path, near mtp_forward 933-937)**: behind a new
   `Q27_DRAFT_RECALL` env flag, D2H the full `mtp_logits` (248320 f32, ~1 MB) for
   each draft step and the prior round's resident verify logits (`logits2`,
   engine.cuh:1188), compute host-side whether the draft argmax rank is `< K` in
   the verify ordering, per draft position 1..4. Emit to the burst-stats CSV. This
   is measurement-only (gated, cold path, ~4 MB D2H/round) -- it does not touch the
   shipped graph.
2. **engine.cu (~235, burst-stats CSV writer)**: add `rank1..rank4` columns
   alongside `d1..d10`.
3. **tools/ladder_price.py**: add a rank-CDF reducer -- `p_novel(K, step) =
   P(rank >= K)` per draft position, and the truncating-geometric
   `E[n] = sum_{k} prod_{j<=k} r_j` vs baseline.

### Kernel (conditional -- only after P1 passes gate 3)
4. **kernels.cu (~192, new `k_gemv_q4_shortlist`)**: fork `k_gemv_q4`, replace
   `row` with `idx[blockIdx.x*(blockDim.x/32)+threadIdx.x/32]`, launch
   `ceil(K/warps)` blocks.
5. **blocks.cu (~398, argmax over K)**: run `k_argmax_top2` over K logits, then
   remap local winner `tok -> idx[tok]`.
6. **engine.cuh (mm dispatch ~738 + mtp_forward 933)**: build `idx[K]` once per
   round from a top-M scan of the prior verify logits (reuse the resident
   `logits2`), pass it to the shortlist head for all 4 steps (idx stable across
   the ladder -> L2 residency on steps 2-4).
7. **New build target + flag** (Makefile, engine.cu arg parse): `Q27_SHORTLIST_K`,
   off by default. `--fast-head` must stay OFF for the gate to bind.

## Costs to measure

- **Realized draft fraction on novel prose** = `gs.draft_ms/round` / `ms/round`
  (server.cu:450). The true prize denominator. Also log realized `gs.draft_steps`
  -- if it is < 4, the 2.91 ms baseline already overcounts.
- **`gate_n_hist` peakedness** on novel prose (server.cu:475-493). Peaked -> D
  viable; flat/spread -> D dead on this traffic.
- **`p_novel(K, step)`** per draft position 1..4 (P1). The single kill number.
- **Effective head cut**: nsys the shortlist head with idx resident -- confirm the
  L2 hit on steps 2-4 (target < linear K/248320 x 1.60 ms) and the residual
  ~1.10 ms MTP-layer floor that no head lever touches.
- **Acceptance A/B vs the md5 gate**: passing md5 + lower t/s = acceptance dropped
  (kill); failing md5 = you touched verify (bug), not the lever.

## Gates

1. **P0 (zero code, ~30 min, DO FIRST):** run the server on a real novel-prose
   prompt, greedy, `Q27_PMIN=0.5`, phase stats on; read `gs.draft_ms`,
   `gs.draft_steps`, `ms/round`, and `gate_n_hist`. Deliverables: the true realized
   draft fraction (settles 12% vs 22%), the dexit-averaged baseline, the
   n-distribution. **If `gate_n_hist` is flat: off-path D is dead on novel prose --
   do not scope it. If realized draft < ~12% of the round: no on-path lever clears
   its own effort -- STOP, the draft is at its floor.**
2. **P1 (measurement instrumentation, ~0.5-1 session, BEFORE any kernel):** build
   the rank-CDF probe (checklist 1-3), run on novel prose, compute `p_novel(K,
   step)` and `E[n]`. **KILL if per-step recall < 0.95 at positions 1-2 or < 0.97
   at positions 3-4, or if it decays past those by step 4, at the smallest K
   whose head reclaim clears >= 4% of the measured round. No CUDA is written.**
3. **Kernel greenlight (only if gate 2 passes):** build checklist 4-7. Gate on
   canonical md5 UNCHANGED (a2982c51...) with `--fast-head` off, `test_kernels`
   ALL PASS, and measured t/s up at the headline op-point. A passing md5 with
   *lower* t/s means acceptance dropped below break-even -- revert.
4. **Off-path D (separate plan):** open only if gate 1 shows `gate_n_hist` sharply
   peaked AND decode t/s is the headline metric. Not part of this plan.

## Baselines that must not move

Canonical md5 `a2982c5197c627551b27d76a0a94b220` (tools/shortbench_suite.sh,
greedy, no `--fast-head`); `test_kernels` ALL PASS; gated==ungated token identity.
Any draft-head change is bitwise-neutral on this gate by construction (finish_round
commits the verify verdict; `W_MAX=12 > max n`); a moved md5 is a bug in the verify
path, not the lever.

## Effort and recommendation

| Step | Effort | Risk | Expected size (novel headline) |
|---|---|---|---|
| P0 read | ~30 min, zero code | none | decides everything |
| P1 probe | 0.5-1 session, no hot-path kernel | none | kill/greenlight number |
| Shortlist kernel | 1-2 sessions | low kernel, all risk in P1's number | +4-6% if recall clears; net-negative if not |
| Off-path D | weeks, dual-GPU pipeline | high | +3-9% if `gate_n_hist` peaked; ~0 on flat novel n |

**Recommendation: run P0 now, then P1; build no kernel until P1 clears gate 2.**
My expectation is that P1 kills the shortlist on novel prose -- the argmax is
content-driven, recall decays over 4 serial steps, and the only high-recall source
collapses B into E (an artifact the repo lacks). In that case the honest finding is
"the draft is at its SOL floor; the next real lever is off-path drafting, a separate
architectural project justified only when decode t/s becomes the headline." P0+P1
are cheap and decisive; committing to either kernel or pipeline before them is
premature.

## Adversarial findings ledger

- **DEXIT double-count (2.91 ms is saturating):** INCORPORATED. Baseline is realized
  `gs.draft_ms/round` (engine.cuh:1578-1583, server.cu:450), not the saturating peak.
- **Wrong denominator (12% is a width-12 suffix round with no draft):** INCORPORATED.
  Novel ladder round verifies at width <= 8 < gemm_min=9 (engine.cuh:1284); true share
  ~15-22% at short ctx, <5% at long ctx.
- **"Q8 verify / bitwise-safe" conflates CLI gate with production:** INCORPORATED.
  `--fast-head` is the server default; safety restated as "finish_round commits the
  verify verdict," head-agnostic. Canonical gate is the no-fast-head Q8 path.
- **Acceptance model:** INCORPORATED. Truncating geometric `E[n]=sum prod r_j`,
  shortlist-vs-Q4 recall per position -- NOT the 98.1% Q4-vs-Q8 figure.
- **L2 residency upside (idx stable across 4 steps, 82 MB < 128 MB L2):** INCORPORATED
  as an upside; bounded by the 427 MB MTP-layer floor that stays in VRAM.
- **B collapses into E (no cheap high-recall source; low-rank artifact absent):**
  INCORPORATED -- treated as one lever family, gated on the same probe, flagged as the
  missing artifact.
- **Off-path bubble (h_next at end of verify(R); suffix strips draft where n is
  predictable):** INCORPORATED -- D scoped out for novel prose, gated on `gate_n_hist`.
- **Gather mechanics (group-64 scales ride the row index, ~10-line fork):** INCORPORATED
  as low kernel risk; the cost is the source, not the gather.
- **Lever A (free win -- redundant traffic, argmax fusion, spill/occupancy, width-4
  batch):** OVERRULED as material. All four die on arithmetic: reads are 1.1 GB/step >>
  128 MB L2 and serially dependent (no reuse); argmax already reads the 1 MB logits, not
  the 635 MB weight (fusion saves ~0.02%); single-token `k_gemv_q4/q8` carry no
  `__launch_bounds__` and sit at the DRAM wall (no spill cliff, unlike the batched
  latency-bound `_n` kernels); the width-4 batch is impossible through the serial argmax.
  Cheapest confirmation if doubted: one `ncu gpu__dram_throughput.avg.pct_of_peak` line
  on the warm `k_gemv_q4` (full-path + `sudo -n`, not on PATH) -- it reads >90%.
- **Lever C (lower k_min below 4):** OVERRULED. dexit already trims per-round below the
  ceiling and floors verify at width 2 (engine.cuh:1584); depthctl only floats up and
  resets to k_min per request (depthctl.h:69). Lowering k_min buys nothing -- C's
  savings are already inside the realized `gs.draft_ms`. (This overrules the draftpath
  survey's #1 ranking, which predated the dexit analysis.)