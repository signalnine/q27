# gpt-6-astra design review: DFlash2 sampled selector walk + sparse-q rejection (2026-09-07)

Reviewer: gpt-6-astra via codex exec (read-only sandbox, xhigh), before
implementation. Input: the design note (scratchpad d2_stochastic_walk_design.md,
reproduced in docs/plans/2026-09-08-dflash2-close-the-arm-gap.md's item-2
resolution) plus src/dflash2.cu, src/blocks.cu, src/engine.cuh and ninfer's
candidate_selector_path.cu / speculative_round.cuh.

## Finding that triggered it

Same drafter checkpoint class as ninfer (z-lab Qwen3.8-27B-DFlash2, straight
conversion; theirs W8G32, ours Q4-g64). Greedy no-think acceptance identical
(4.86 vs 4.87 tok/round). Sampled serving (temp 1.0 / top-p 0.95 / top-k 20 /
min-p 0.05, think-on, seeded 6x{12.5K,50K}): q27 3.08 tok/round vs ninfer
3.69. Per-lane (new gnh/glf/gla counters in dflash2_round):

    q27 d2 (greedy walk) 12.5K  P>=j: 0.645 0.486 0.354 0.249 0.170 0.105 0.077
                                cond: 0.645 0.755 0.728 0.703 0.683 0.618 0.733
    ninfer d2            12.5K  P>=j: 0.754 0.609 0.444 0.347 0.243 0.160 0.129
                                cond: 0.754 0.809 0.728 0.783 0.699 0.658 0.805

Front-loaded deficit with identical greedy behaviour = the walk's proposal law
under sampling: q27 walked argmax (one-hot q, accept prob p(argmax)); ninfer
draws the path from softmax(E/T) and rejects against that sparse q (expected
accept sum_v min(p,q)).

## Verdict (verbatim structure, condensed)

Q1 yes: with normalized truncated p, proposals drawn from the retained q, and
independent accept/correction randomness, accepted mass is min(p,q) and the
rejected mass is redistributed as normalized (p-q)+; q support outside the
nucleus is harmless. Preserves q27's implemented target distribution.

1. P1 -- preserve the cap branch; subtract q ONLY when exclude >= 0 (capped and
   all-accepted rounds draw plain p; the bonus lane's q[K] does not exist).
2. P1 -- Q27_D2_WALK=greedy must keep the one-hot verification tail (the greedy
   walk never writes q rows).
3. P1 -- fp32 cdf can equal Philox's maximum uniform with q[15] == 0; strict
   u < cdf then falls back to a zero-probability slot and `p >= q` would accept
   an out-of-nucleus token (p = q = 0). Fall back to a positive-q slot. (ninfer
   draw_rank has the same weakness.)
4. P1 -- pre-existing: alloc() never initialised d_ctx_n, and capture_draft()'s
   warm run reads it in k_d2_attn. Initialise before any warm run; warm the new
   tail kernels before capture; keep sampled mode through the eager fallback.
5. P2 -- the "-1e30 key offset" residual fallback in the note is wrong (fp32
   collapses all such keys; am_pack then picks the lowest tied id, possibly
   the rejected token). Skip non-positive residuals and detect emptiness
   explicitly.
6. P2 -- determinism contract: kinds 2/3/4 are disjoint; overlapping rounds only
   reuse proposal-suffix randomness that never determined a committed prefix
   (safe); round TRUNCATION can revisit evaluated keys; seed+prompt alone does
   not give cache-independent output (ring reset + prefill reseeding change q).
7. P3 -- the acceptance improvement is empirical; request temperature is a
   heuristic (max overlap for a fixed candidate set S is p(S), at q = p(.|S)).

## Resolution in the implementation

- (1) k_d2_spec_accept keeps `*cap -> {n=1, stop=0, exclude=-1}`;
  k_d2_sample_stop gates the subtraction on `exclude >= 0`.
- (2) d2_setup captures the OLD spec_verify_tail_sampled when
  Q27_D2_WALK=greedy; the sparse-q tail only pairs with the sampled walk.
- (3) k_d2_walk falls back to the last candidate with q > 0; k_d2_spec_accept
  additionally requires p > 0 to accept.
- (4) alloc() memsets d_ctx_n; capture_draft() uploads ctx_n before its warm
  runs; d2_setup runs spec_verify_tail_sampled_d2 once eagerly before capture;
  draft() passes `sampling` through to the eager draft_compute.
- (5) k_d2_sample_stop skips r <= 0 (and the rejected token), contributes the
  zero pack when a thread finds nothing, and k_d2_stop_fallback (sample_stop's
  body, gated on *best == 0) supplies the exclude-d draw for a numerically
  empty residual. test_d2_walk_reject (f) constructs that case.
- (6) documented in the commit; the sampled d2 stream is reproducible for
  matching drafter state (cold-vs-cold seeded runs reproduced round counts
  exactly; warm turns differ in the uncached tail the ring is seeded from).
- (7) measured, not assumed: lane-1 0.645 -> 0.751 (ninfer 0.754), tok/round
  3.075 -> 3.383 at 12.5K, 3.036 -> 3.234 at 50K, round wall unchanged.

Raw transcript: session scratchpad d2walk_codex_design.out.
