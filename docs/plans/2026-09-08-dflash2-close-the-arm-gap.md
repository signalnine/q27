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

## Progress 2026-09-07 late: item 2 root-caused and mostly closed

Item 2 was NOT quant/taps/ring/provenance. Cheap-first diagnostics:

- (d) provenance CLOSED: ninfer's pack is a straight conversion of the same
  z-lab checkpoint (docs/maintainer/qwen3.8-27b-dflash2.md pins revision
  50307d4c; artifact doc: matrices W8G32, norms/conv/selector BF16). No retrain.
- (b) per-lane localization (dflash2_round now feeds gnh/glf/gla, so a d2
  profile reads off the [req] journal exactly like the ladder's):

      q27 greedy-walk 12.5K  P>=j: 0.645 0.486 0.354 0.249 0.170 0.105 0.077
                             cond: 0.645 0.755 0.728 0.703 0.683 0.618 0.733
      ninfer          12.5K  P>=j: 0.754 0.609 0.444 0.347 0.243 0.160 0.129
                             cond: 0.754 0.809 0.728 0.783 0.699 0.658 0.805

  Lanes 3..7 conditional match; lane 1 (-0.11) and 2 carry the deficit, and
  greedy no-think acceptance was already identical -> the SAMPLED proposal
  law. q27's selector walk was greedy in sampled rounds (one-hot q, accept
  prob p(argmax E)); ninfer draws the path from softmax(E/T) and rejects
  against that sparse q (candidate_selector_path.cu draw_rank +
  speculative_round.cuh speculative_sparse_warp_accept; expected accept
  sum_v min(p, q)).

SHIPPED: sampled selector walk + sparse-q rejection tail (k_d2_walk sampled
branch + second draft graph; k_d2_spec_accept / k_d2_sample_stop /
k_d2_stop_fallback; spec_verify_tail_sampled_d2; Q27_D2_WALK=greedy = old
behaviour). Design + implementation reviewed by gpt-6-astra
(docs/reviews/2026-09-07-gpt6astra-d2-sampled-walk-*.md). Gates:
test_kernels --sampling-only test_d2_walk_reject (device walk + device tail:
output ~ p by chi-square, accept rate == sum min(p,q), one-hot q bit-identical
to the ladder's tail, cap and empty-residual paths), greedy CLI byte-identity,
seeded think driver.

Result (same instrument, same night, quiet box):

      q27 sampled-walk 12.5K  P>=j: 0.751 0.575 0.424 0.278 0.184 0.109 0.076
                              cond: 0.751 0.765 0.738 0.655 0.663 0.593 0.697
      tok/round 12.5K 3.075 -> 3.383 (ninfer 3.69); 50K 3.036 -> 3.234 (3.59)
      t/s       12.5K 138.9 -> 153.3 (+10%);        50K 132.0 -> 141.7 (+7%)
      round wall unchanged (22.2 / 23.3 ms) -- the walk costs nothing.

Lane 1 is at parity (0.751 vs 0.754). The residual (-8% tok/round) now sits
in lanes 4..7 (cond 0.655/0.663/0.593/0.697 vs 0.783/0.699/0.658/0.805):
deeper mask rows, i.e. drafter numerics (Q4-g64 vs their W8) and/or ring
coverage, not the walk. Q8 serving pack A/B: see below.

WARM-TURN RING STARVATION -- FOUND AND SHIPPED (same night). d2_prefill_begin
reset the ring every turn and prefill re-seeded only the UNCACHED tail, so a
prefix-cache warm turn started the drafter with 1-5 context rows: tok/round
3.28 (cold, pf=3023) -> 3.16 (warm, pf=5) -> 3.05 (warm, pf=1) on three
identical seeded requests. Agentic traffic is almost all warm turns.

Shipped: d2_prefill_align(prompt, base) keeps ring rows for positions below
min(LCP(d2_seq, prompt), base) -- d2_seq is the host token sequence the ring
was built from (maintained per accepted lane and at the truncation
rollback), so the rule is lineage-agnostic and exact: a RAM/disk restore over
a ring built from another conversation gets LCP ~ 0 -> reset. Dflash2 tracks
ctx_end/ctx_contig; rollback_to(pos) resets on non-contiguous coverage.
Q27_D2_RING=reset = old behaviour. That contiguity check exposed a
PRE-EXISTING HOLE: the last prompt token (NP-1) went through step_with
(graph_exec, no tap capture) and was never in the ring -- every turn's
drafter lacked its most recent context row. Fixed with the eager
token_launches(d2_vtaps) (graph_exec's own launch sequence plus the tap
copies) + a one-row ingest.

Measured (bench/ladder/drive_warm_turn.py, keep vs Q27_D2_RING=reset):
  (A) identical request x3:  keep 3.82 / 3.82 / 3.82 tok/round (67 rounds each,
      identical streams: warm == cold now) vs reset 3.56 / 3.12 / 3.12
  (B) two-turn conversation, turn 2: keep 5.69, 4.92 vs reset 4.57, 3.88
  Standard 12.5K seeded think (all cold): 3.383 -> 3.593 tok/round (+6%),
  153.3 -> 162.9 t/s -- see BUILDLOG 2026-09-07 (g). Slide exercised (2400-
  token generation, then a repeat + a follow-up turn): no fault; a prompt
  that repeats a range the ring has slid past correctly keeps 0 rows.
  gpt-6-astra review P2s (overlapping compaction copy, fused-batch bypass,
  latched contiguity flag) fixed before commit.

## Progress 2026-09-07 late (2): item 1 measured; small lever shipped

Serving d2 round (12.5K think, vox stopped): draft 3.7 -> 3.55 ms after the
top-16 bitonic rewrite + windowed attention loop (bitwise-neutral), verify
17.7, host 0.58, plus post_round's fold ~0.4 -> ~22 ms vs ninfer 18.0.
Drafter decomposition (nsys, BUILDLOG 2026-09-07 (h)): 47 Q4 gemvs 0.89 ms
(near the 1.2 GB floor), Q4 head 0.4, attention 0.24-0.5 (ring 1.2K-2.5K
rows), top-16 now 0.09, ~110 tiny kernels ~0.2, walk 0.01, residual graph
gaps. Remaining levers, ranked:
- (i) attention flash-decoding split or K/V shared across the 4 GQA heads
  per block: 0.2-0.4 ms (the naive smem-tiled version was SLOWER; see (h)).
- (ii) head: ninfer's optimized route (131072-row draft head, 0.34 GB Q4)
  ~0.2 ms, acceptance impact to measure; or an fp8/fp4 head.
- (iii) fp4 backbone repack (sm_120a block-scaled MMA works): ~0.45 ms,
  needs the Phase-4 numerics gate.
- (iv) fuse the tiny kernels (norm+quantize, dconv+quantize): ~0.1 ms.
- (v) adaptive verify width (K=7 drafts, W<8 verify when unconfident): the
  width_bench says ~0.12 ms/lane, so ~0.5 ms at best, minus the sync and
  the acceptance cap -- last.
Honest ceiling for the drafter: ~1.5 ms (from 3.55) = -2 ms/round = +10%
t/s; the verify (17.7 vs ninfer ~16.5) and post_round (~1 ms) are separate
engine work.

## Drafter attention SHIPPED (2026-09-07 late, BUILDLOG (j))

Serving nsys: k_d2_attn was 1.87 ms of the 3.95 ms draft graph (375 us per
launch at a 2.3K-row ring). Flash-decoding rewrite (32 key splits x 8 kv
heads, K/V shared across the 4 GQA heads, online softmax, combine kernel):
draft 3.57 -> 1.95 ms, round 22.0 -> 20.4 ms, +12% t/s, tok/round
unchanged (3.59 -> 3.65). Drafter budget now: Q8 gemvs 1.3 (floor), head
0.43, top-16 0.09, tiny ~0.15. Remaining round gap to ninfer (20.4 vs
18.0): verify 17.85 vs ~16.5, and fold 0.38 + host 0.58 + ingest 0.13 vs
their 0.6 -- the fold can overlap the draft graph on a side stream (~0.4).

## Agentic standings 2026-09-07 late (bench/crossengine/agentic-2026-09-07)

Claude Code on the 12 SWE-bench instances, effort medium on both engines:
q27 ladder 162.1 t/s (3.13 tok/round) | q27 DFlash2 Q4 173.3 (3.94) | q27
DFlash2 Q8 176.3 (4.03) | ninfer DFlash2 228.5 (4.28) | ninfer MTP3 148.3
(2.90). DFlash2 is now a +9% serving win on q27 (was -7%); q27 leads ninfer
+9% at equal drafter class; their DFlash2 arm leads ours +30% on +6%
tok/round -> the round wall is the whole remaining gap on agentic traffic
too. Confound: the model thinks 2.5x less per message on ninfer's quant.

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
