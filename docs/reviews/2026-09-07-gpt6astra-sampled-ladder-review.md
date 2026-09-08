# gpt-6-astra review: sampled-ladder depth widening (2026-09-07)

Adversarial static review of the diff implementing
docs/plans/2026-09-07-sampled-ladder-depth.md (sampled spec path widened from
the hard depth-4/width-5 cap to the adaptive ceiling, tail kernels to lane
packs, outcome unified onto the greedy OUTCOME_INTS layout).

VERDICT: "no production correctness regression found in static review."

Production paths verified:
- Rejection chain + bounds: accept counters (P, 2, k, 0) for k=0..6, stop
  draw (P, 3, token, 0) -- no counter reuse introduced; one stop draw per
  round needs no stop-lane component. logits2 has W_MAX lanes, d_nuc 16.
- Outcome migration: no remaining executable reader assumes the old 7-int
  layout or pending at [6]. Conductor D2H, solo/fused sampled commits,
  CLI/DFlash2 readers, metrics/server telemetry all read the wide layout.
- Capture ordering: warm at widest sampled shapes, explicit 4/5 reset before
  the monolithic capture, per-width loop, vw=5 restore; the retained
  dmax=gate_maxd feeds no later capture (fold takes explicit T_, DFlash2
  installs its own width).
- Fused widths 6-8: draft rows + hidden buffers exist through depth 7; trim
  only shrinks grants; exec-cache key (ordered engines, granted widths,
  sampled mask, suffix class, KV kind) needs no depth component since
  drafting is outside the verify capture.
- Depth-4 equivalence: no changed probability arithmetic, Philox inputs,
  token selection, or hidden-state choice. (Token identity does not prove
  identical buffer traffic -- pending moved [6]->[17], D2H grew 28->72 B --
  by design.)

Findings (both TEST defects, both fixed same session):
1. [P2] The random depth-7 chain gate could pass with lanes 5-7 disabled:
   expected observations at n>=6/7/8 are ~29.7/11.3/1.55 in 8192 trials and
   the tolerance exceeded the expected probability. FIX: dominant-prefix
   conditional test -- lanes 0..4 forced-accept, lanes 5-6 measured
   conditionally with thousands of samples (now in test_kernels (h)).
2. [P2] All-accept loop `md + 1 < LANES` tested depths 1..6 only. FIX:
   `md < LANES` (depth 7 / n=8 now forced).

Residual gap it names: no synthetic harness drives finish_sampled /
nucleus_multi directly (covered indirectly by the seeded old-vs-new E2E
equivalence at depth<=5 and by shared-buffer identity with the greedy tail at
n=6..8; a dedicated harness is future work). Also: for any DEXIT=0 depth-7
A/B use fixed Q27_MAXD=7 -- auto7 clamps to 5 under Q27_DEXIT=0.

Raw transcript: session scratchpad sampled_depth_codex.out (139K tokens).
