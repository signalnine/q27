# Sampled ladder to depth gate_maxd (lever 1 of the 09-07 part-2 attribution)

Motivation: `perf-attribution-decode-2026-09-07b.md` section 5. Live serving
always samples (temp 1.0) and the sampled spec path is hard-capped at depth-4
draft / width-5 verify, while ninfer drafts 7 on identical traffic and takes
+0.26 tok/round from positions 5-7 at an equal round wall. Per-lane accept
decay matches theirs for lanes 1-4, and the live cap histogram shows 63% of
gated rounds SATURATED at cap=4 -- the margin gate wanted to go deeper.

## Design

Depth policy for sampled rounds -- REVISED after measurement: sampled rounds
ride the SAME P13 dctl adaptive ceiling as greedy (4..gate_maxd on saturation
evidence) and FEED dctl exactly like greedy rounds. The first cut used a
fixed gate_maxd ceiling (mirroring ninfer's fixed-7) and LOST 2-16% across
the whole arm-A sweep: ninfer's DFlash2 drafts all 7 positions in one forward
(depth is free), while q27's MTP draft steps are SEQUENTIAL (~0.45 ms each) --
unconditional deep drafting on think traffic burns more than the
~0.14/0.07/0.05-accept tail lanes pay. The theta margin gate alone does not
price that (63% of rounds clear theta 0.5 to the ceiling); the dctl
saturation ladder is the referee built for exactly this economics. An
operator restores the old sampled behavior with Q27_MAXD=4.

Mechanical widening (the same W16 shape the greedy tail got):

- `struct IP3` moves spec3.cuh -> kernels.cuh (next to CP3) so blocks.cuh can
  take lane packs.
- `q27k::spec_accept`: dr1..dr4 named pointers -> `IP3 drafts`; the accept
  walk reads `*drafts.p[k]` for k < max_draft (max_draft now up to
  W_PLUMB-1; practically vw-1 <= 7). Philox counters already key on lane k --
  KIND_SPEC_ACCEPT word k in 0..6, no collision with any other draw site.
- `q27k::finish_sampled`: dr1..dr4 + x1a..x1e -> `IP3 drafts` + `CP3 x1s`
  (LANESW(x1)); `src = x1s.p[n-1]`; outcome moves to the GREEDY layout
  ({n, t1, dr1..dr15, pending} = OUTCOME_INTS, pending at [OUTCOME_INTS-1]) --
  writes all W_PLUMB-1 draft slots like k_finish_round.
- `d_nuc`: 5x4 -> W_PLUMB x4 floats (lane packs rule: plumb-wide).
- `spec_verify_tail_sampled`: build the packs exactly like spec_verify_tail;
  nucleus_multi already takes L=vw.
- `spec_sample_round`: md_used 4 -> gate_maxd; outcome read OUTCOME_INTS,
  pending from [OUTCOME_INTS-1].
- `draft_md_used(sampled)`: `sampled ? gate_maxd : (maxd_auto ? dctl.cur :
  gate_maxd)` -- covers the conductor's fused sampled rounds (draft_and_gate
  and spec_verify_tail_sampled are member functions; the fused path inherits
  the widening).
- `commit_outcome` sampled branch: pending from [OUTCOME_INTS-1].
- `build_spec_graphs` sampled phase: warm at dmax=gate_maxd / vw=gate_maxd+1,
  monolithic spec_sample_graph capture stays dmax=4/vw=5 (ungated sampled
  path unchanged), per-width sampled verify captures W=2..gate_maxd+1 (was
  2..5): +2-3 graphs.

Not touched: greedy tail + graphs (canonical bitwise gates), suffix (greedy-
only), toolgram (accept-cap path unchanged), dctl, DFlash2.

## Gates (all run 2026-09-07)

1. `test_kernels --sampling-only` ALL PASS, including three NEW deep gates:
   depth-7 accept-chain vs the CPU Philox reconstruction, all-accept at
   depths 1..7, and (after gpt-6-astra found the random chain statistically
   toothless at lanes 5-6) a dominant-prefix conditional test -- lane-5
   accept 0.3798 vs 0.3806 served, lane-6 0.1360 vs 0.1371 (3111 samples).
2. REFACTOR EQUIVALENCE: old vs new binary at Q27_MAXD=4 sampled seeded --
   token-md5 identical, same rounds. PASS.
3. Ungated sampled: old vs new token-md5 identical. PASS.
4. Greedy (auto7, seeded): old vs new token-md5 identical. PASS.
5. Determinism: auto7 sampled, same seed twice -> identical. PASS.
6. gpt-6-astra adversarial diff review: "no production correctness
   regression found in static review"; production paths (Philox counters,
   outcome migration, capture ordering, fused widths 6-8, depth-4
   equivalence) all check out. Two P2 TEST defects found and fixed (the
   toothless chain tolerance; an off-by-one keeping all-accept at depths
   1..6).

## Measured outcome

- FIXED gate_maxd ceiling (first cut): arm-A sweep -2..-16% across all six
  ctx points -> reverted to the adaptive policy (the "Design" note above).
- Adaptive policy: CLI seeded A/B reproduces the old depth-4 numbers exactly
  on short think traffic (dctl holds at 4 without saturation evidence) =
  no regression; deep rounds engage only where realized accepts justify
  them. The n>5 mass now appears in live gnh histograms when traffic
  saturates. The UPSIDE is gated on lever 3 (bar/theta retune): the margin
  gate prices greedy confidence, but sampled acceptance is p_served (temp
  1.0) -- margins cleared 4-deep on 63% of sweep rounds while only 24%
  accepted 4 -- so promotion under sampling needs bars keyed to realized
  accepts (dctl already is) plus a theta/economics pass. Structural fix for
  deep-draft economics (depth at zero marginal cost) = lever 2, sampled
  DFlash2 verify.
