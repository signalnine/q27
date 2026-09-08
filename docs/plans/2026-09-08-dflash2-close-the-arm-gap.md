# Close the DFlash2 arm gap (queued 2026-09-07, next session)

Where the 09-07 arc landed (docs/perf-attribution-decode-2026-09-07b.md,
docs/plans/2026-09-07-sampled-ladder-depth.md,
bench/crossengine/rebench-2026-09-07/): q27's ENGINE beats ninfer at equal
acceptance (ladder round 17.5 ms vs their MTP ~19.0; +13% t/s over their MTP3
on identical prompts). Their remaining +36-45% net lead is entirely the
DFlash2 arm: 3.65 tok/round at ZERO round-wall premium, vs our dflash2-sampled
3.08 tok/round at +4.8 ms over our own ladder. Two work items, in order.

Instrument for everything here: `bench/ladder/drive_seeded.py <base> <model>
<tag> [think|echo]` (seeded prompts, 6 x {12.5K, 50K}); tok/round + round wall
from q27 [req] journals / ninfer --request-log-jsonl. Unseeded arm-A sweeps
are +-10-20% acceptance noise -- do not use them for A/Bs. ninfer serve rig:
release artifact + `--spec dflash2 --draft-tokens 7 --kv-dtype int8
--kv-capacity 131072 --max-context 131072` (defaults come up 8K/bf16).

## Item 1: kill the +4.8 ms dflash2 round premium (22.3 -> ~19.5 ms)

Premium decomposition (Q27_D2_TIMING + tonight's journals): drafter ~2.0 ms
(Q4 serving pack, graphed, compute-bound) + always-width-8 verify vs the
ladder's gated ~3.5 avg (~1-1.5 ms) + top16/walk ~0.3 + ingest ~0.2 + sampled
tail ~0.2. ninfer's premium is ~0 (their d2 round is FASTER than their MTP
round -- one drafter forward replaces sequential MTP steps).

- (a) ADAPTIVE D2 VERIFY WIDTH (the 09-06 gpt-6-astra lever #1): keep K=7
  drafts, verify W < 8 when the drafter is unconfident. The selector walk
  (k_d2_walk) already computes per-position margins on device. Host-loop
  version is cheap now: capture per-width d2 verify graphs (greedy + sampled
  twins, tap-enabled -- verify_graph_w lack taps, capture like d2_setup does
  at each W), D2H the margin run (the d2 round already syncs on the outcome,
  and a 4 B pre-verify read costs ~0.2 ms) or a device SWITCH later.
  Watch: emitted tokens must stay identical for greedy (equality chain only
  reads accepted prefix -- width change is behavior-neutral greedy; sampled
  is distribution-neutral). Gate like the ladder's P12.
- (b) DRAFTER READ COST: ours 2.0 ms vs their ~0.5 (nvfp4). Floor check
  first: serving pack is 1.2 GB Q4 -> ~0.7 ms pure-BW; where do the other
  1.3 ms go (draft_compute is captured -- nsys the graph once)? Then either
  kernel work or an fp4 repack (5090 fp4 MMA WORKS -- sm_120a block-scaled
  mxf4nvf4, see memory reference_sm120_no_fp4_mma) or fp8. Any repack needs
  the Phase-4 numerics gate (tok/round within 5% of fp16 taps).
- (c) small: overlap ingest with the next draft's staging; sampled-tail
  kernels are serial 1-block launches (~0.2 ms) -- batch if it shows.

## Item 2: close the acceptance delta (3.08 vs 3.65 tok/round, same drafter class)

Cheap-first:
- (a) QUANT: our Q4 pack measured -1.5..-3.6% tok/round vs fp16 taps
  (Phase 4). Build a q8 SERVING pack (tools/dflash2_pack.py --q8 --no-head
  --no-embed; the existing qwen38-dflash2-q8.d2w has head+embed) and A/B on
  the seeded instrument. If Q8 recovers a chunk, fp8/fp4 repack per 1(b).
- (b) PER-LANE LOCALIZATION: tonight's telemetry is already on disk --
  ninfer accepted_per_position in
  bench/crossengine/rebench-2026-09-07/rb_ninfer_reqlog.jsonl vs our d2
  gnh/lane counters in the q27-d2test journal window (and re-runnable).
  Early-lane deficit = drafter quality/taps; tail deficit = walk/selection.
- (c) RING/TAP COVERAGE: our prefill tap capture seeds the last 2048 prompt
  tokens (D2_SEED_WINDOW); check ninfer's effective window; A/B a bigger
  seed window (ring alloc is 4096).
- (d) PACK PROVENANCE: our pack is a Q4 conversion of the z-lab bf16
  checkpoint; ninfer's may be retrained (check ninfer-master model-cards /
  release notes). If theirs is retrained, that bounds what (a)-(c) can
  recover and the answer is a retrain or their-weights import, priced
  separately.

Combined honest target: dflash2-sampled from ~139 to ~175-190 t/s think
(ninfer at 203 on the same instrument); echo already wins at 50K (+10% over
our ladder).

## Standing cautions

- Round truncation/forced-transition d2 state sync + ctx reserve fixes are
  in (3ec524d) -- do not regress verify_w_max() when touching widths.
- A SAMPLED on_round callback returning m<n without forcing/cancelling is
  unsafe (pre-existing contract gap, flagged in
  docs/reviews/2026-09-07-gpt6astra-sampled-dflash2-review.md).
- Serving greedy is NOT width-invariant across depth configs (batch-gemm
  reduction shapes) -- CLI gemv-path gates only.
- Canonical prompt token files now live in bench/dflash2/toks/ (were only in
  session scratchpads before).
