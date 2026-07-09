# Acceptance-gate design: gate on verifier acceptance, not drafter confidence

2026-07-08. Successor to the maxd6 NO-GO retry bar (docs/maxd6-decision.md: "a
gate that predicts acceptance rather than confidence") and the depth-match P4
finding (BUILDLOG:1861: llama +45% in the near-verbatim tail; "raising the
ceiling is already priced (needs an acceptance-predicting gate, not theta)").
All baselines RTX 5090, production serving config (fp8 KV, --fast-head), greedy.

## Problem -- what the theta gate gets wrong, measured

The P12 gate thresholds the drafter's top1-top2 margin (theta): cap = leading
run of margin >= theta, verify width = cap+1. That is a CONFIDENCE signal. The
maxd6 GO-IF measurement (BUILDLOG:1648) showed confidence and acceptance
decouple by traffic regime at the same theta=0.5:

| payload (26K replay A/B) | d5 fired | p(5th lane \| fired) | d5 net vs d4 |
|---|---:|---:|---:|
| code-repro | 97.6% | 95.2% | +2.9% |
| T8-style codegen | 56.3% | 56.2% | **-5.4%** |
| fresh unit-test gen | 51.9% | 48.6% | -3.9% |

Same margins, same theta -- on codegen 44% of fired rounds waste the deep
draft+verify. At the live T8 operating point (79.3% fired / 82.2% yield) depth-5
is breakeven at best (-0.8..+0.1%). No function of the margin alone can separate
these regimes: the margin -> acceptance mapping itself shifts with traffic.

Two consequences:

1. **The current demote bar is an order of magnitude below breakeven.** P13
   demotes 5->4 when yield_ema < maxd_lo=0.10. Rearranging the throughput
   condition (d5 wins iff dtok/dms per fired round > baseline tok/ms) on the
   measured deltas gives a breakeven lane-5 yield of **~0.7-0.8** -- consistent
   with "d5 wins only in >~90%-fired regimes". A stream whose 5th lane pays 30%
   keeps burning it forever (codegen's 56% never demotes; only the HI=0.5
   promote bar accidentally protects the auto path by never promoting there).
2. **The ceiling can never be raised safely without this gate.** P4-echo showed
   +45% left on the table at 100% acceptance vs llama depth-10. maxd6 priced
   d6 as NO-GO *because* the theta gate cannot confine deep lanes to rounds
   that accept. An acceptance-predicting gate is the precondition; the d6 build
   itself stays out of scope here.

## Why yield feedback, not static margin calibration

The maxd6 verdict floated "margin-calibrated accept prob". Rejected as the
primary mechanism: a static calibration p_acc(margin, depth) is exactly the
map the measurement showed to be regime-dependent (95% vs 56% yield at the
same margin bin). Realized per-lane yield is regime-aware by construction, and
the mechanism is already validated in-tree: P13's EMA controller with HI=0.5
landed empirically at the measured win/loss crossover (BUILDLOG:1691).

Margin calibration keeps a supporting role: `--stats` today bins acceptance
only by the PASS-2 margin (engine.cu:187-341 computes m3/m4/m5 then voids
them). Phase 0 extends the binning to per-pass margin x depth. If
p(acc_k | m_k) turns out steep at the operating margins, a per-depth theta
schedule (Q27_PMIN comma-list) is a cheap static complement; if flat, theta
stays scalar and yield feedback is the whole game. Decided by data, not taste.

## Mechanism -- per-lane yield bars, generalizing P13

Draft-lane indexing: lane j (j=1..gate_maxd) fired iff cap >= j, accepted iff
n >= j+1. Per-lane conditional yield:

    y_j = EMA( n >= j+1 | cap >= j )        (alpha = maxd_ema_a, default 1/16)

P13's yield_ema is y_5 restricted to depth-5 rounds; sat_ema is the depth-4
saturation EMA. The generalization keeps P13's promote/demote structure per
level, extended from the single 4<->5 pair to a ceiling k* in [k_min..gate_maxd]:

- **demote** k* -> k*-1 when y_{k*} < bar_{k*} (the lane stopped paying);
- **promote** k* -> k*+1 when sat at k* (n >= k*+1) >= HI (the ceiling binds);
- hysteresis as today: on promote, seed the new level's yield EMA just above
  its bar (bounded grace window); on demote, zero the sat EMA.

Per-round composition is unchanged in shape:

    cap_eff = min( leading margin run >= theta, k* )

Margins answer "how deep is the drafter confident THIS round"; yields answer
"how deep has confidence been PAYING lately". Only the deepest active level's
EMAs update per round (mutually exclusive by md_used, as today).

Bars come from measured economics, not taste:

    bar_j = c_j * r        c_j = ms per fired lane-j (draft step + verify
                                 lane increment, from forced-width sweeps)
                           r   = baseline tok/ms of the operating band

v1 uses static bars tuned for the 26-75K agentic band (env-overridable,
`Q27_MAXD_LO` retained as the j=5 bar). If the A/B shows ctx sensitivity, the
extension is r from the engine's own tok/round / ms/round EMAs -- deferred,
not built speculatively.

## Why now -- the economics moved

verify-gemv (40ec3a2, +5.5% decode @61K) landed AFTER the maxd6 A/B: the
verify side got cheaper, which lowers every bar_j and shifts d5 breakeven in
depth's favor. All maxd6 depth economics are stale in the pessimistic-for-depth
direction. Phase 0 re-runs the A/B before any bar is set.

## Phases

- **Phase 0 -- telemetry + refreshed economics (no engine-behavior change).**
  (a) Per-lane fired/accepted counters `gate_lane_fired/acc[j]` beside gch/gnh
  (host longs, printed in `[req]` as glf/gla): give per-lane live yields
  p(n>=j+1 | cap>=j), which the marginals cannot -- the informative margin of
  the full (cap, n) joint at a fraction of the print width. (b) `--stats` per-pass margin bins: p(acc_k |
  m_k bin, prefix ok) for k=2..5. (c) Re-run the 3-payload 26K replay A/B
  (maxd6 rig, BUILDLOG:1655 methodology, payloads reconstructed per the
  depth-match recipe incl. the instant-EOS gotcha) on current HEAD: fresh
  c_j from forced-width sweeps (Q27_PMIN forcing) + fresh d4/d5 deltas.
  GATE: lane counters show sub-bar lanes on codegen-flavored traffic (else the
  build has nothing to convert and stops at Phase 1).
- **Phase 1 -- knob-level acceptance tuning (no new mechanism).** Set
  maxd_lo to the refreshed lane-5 bar (plus the promote-seed clamp this
  implies: seed = min(1, just-above-bar)). A/B on the 3 payloads + live-T8-
  matched replay. This alone converts the "56% yield never demotes" failure.
- **Phase 2 -- per-lane yield gate (the build).** Generalized controller as
  above. Scope set by Phase 0: if only lane 5 sits sub-bar on real traffic,
  the controller stays 4<->5 and Phase 2 collapses into Phase 1. Host-side
  logic only (spec_round tail), zero new device memory, no kernel changes.
  CPU unit tests for the controller (toolconstrain-style); canonical bitwise
  + token-identity + round-determinism gates.
- **Phase 3 -- validation at the live operating point.** T8-style live trial
  + replay A/Bs. SUCCESS: T8-matched replay net >= +1% (was -0.8..+0.1%);
  codegen payload >= 0% under auto; repro payload keeps >= +2%; no canonical
  regressions. Honest expectation: low single digits on mixed traffic -- the
  strategic payoff is unlocking the ceiling question (P4's +45% tail) later.

## Invariants and testing

Same claim as P12/P13/P14: the gate changes ROUND GROUPING ONLY (draft depth,
verify width, round count) -- the emitted token sequence is invariant vs the
ungated canonical path. Verify lanes are independent grid indices; narrowing
never changes which token emits. Gates: canonical md5 4c4120c7 EXACT (ungated
AND gated legs), test_kernels ALL PASS, gated-vs-ungated token identity at
2K/16K/61K, deterministic round counts across runs. Controller logic gets
host-side unit tests (EMA update, promote/demote/hysteresis, seed clamp,
level exclusivity) with no GPU dependency.

## Out of scope

- Raising gate_maxd past 5 (P12b-class 6->7 widening + quantize3-landmine
  audit + 157 MB) -- separately priced in maxd6-decision.md; this gate is its
  prerequisite, not its vehicle.
- Sampled path (ceiling fixed at 4; maxd6 checklist item 6) and the
  constrained tool-split path (P15 refinish dynamics) -- both keep today's
  behavior.
- Drafter retraining (docs/drafter-probe-plan.md, PARKED) and static
  margin->p_acc EV gating as primary mechanism (regime-dependence, above).

## UNMEASURED (what Phase 0 settles)

- Per-lane live yields y_2..y_5 by traffic flavor (glf/gla counters; marginals
  gch/gnh cannot give this).
- Per-pass margin -> acceptance steepness at depth (decides the theta-schedule
  complement).
- Post-verify-gemv c_j and d4/d5 deltas (all current numbers pre-date +5.5%).
- Whether any lane below 5 ever sits sub-bar on real traffic (sets Phase 2
  scope).

## 2026-07-08 RESOLVED -- Phases 0-1 SHIPPED, Phase 2 dead, verdict

Phase 0 (BUILDLOG "accept-gate Phase 0") refreshed everything: the d5 crossover
is **y5 ~ 0.35 conditional** (-1.7% at .282 @61K, +0.2% at .355, +2.7% at .45+
@26K) -- half the maxd6-era estimate; verify-gemv moved it. y2..y4 sit at
.42-.95 everywhere -> **no lane below 5 approaches the bar; the per-lane ladder
(Phase 2 / plan Task 7) is dead by measurement**, exactly the gate the plan set.
Margin bins: instrumented (--stats own-pass bins) but not needed -- yield
feedback alone carried the result; theta stays scalar.

Phase 1 shipped the acceptance gate at depth-5 scale (conditional yield + seed
clamp + lo=0.35): auto beats BOTH fixed legs overall (+2.7% geomean vs the
d4-gated production rec, +0.6% vs fixed-5) and self-protects on the one losing
flavor. Production rec: `Q27_PMIN=0.5 Q27_MAXD=auto`. Residual: -1.1%
promote-churn on 61K docs (pre-existing; escalating promote-hysteresis is the
known shave, unbuilt). The strategic follow-on this unlocks: re-run the maxd6
ceiling-6 GO-IF on these economics.
