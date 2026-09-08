# gpt-6-astra implementation review: DFlash2 sampled walk + sparse-q tail (2026-09-07)

Reviewer: gpt-6-astra via codex exec (read-only, xhigh) on the uncommitted
diff, after the design review (2026-09-07-gpt6astra-d2-sampled-walk-design.md).
GPU execution unavailable to the reviewer; test results below are ours.

## Verdict

Cap branch, greedy A/B routing (old one-hot tail retained), the Philox move
(bodies text-identical, kinds disjoint), shared draft buffers under
single-stream ordering, sampled mode through the eager fallback, the p>0
guard, d_ctx_n init: all confirmed. Greedy arithmetic/routing: no regression
identified (bitwise identity left to our CLI gate).

1. P1 -- k_d2_stop_fallback gated on `*best`, which its own blocks write: a
   block scheduled after another block's atomicMax skips its tokens; even a
   block with no eligible token writes am_pack(-FLT_MAX, 0), so the block
   owning the sole valid token can be suppressed and token 0 emitted outside
   the nucleus. Predicate must be an immutable snapshot.
2. P2 -- the eager tail warm ran before any drafter computation: d_cand
   uninitialised, logits lanes past the ladder's warm width never computed.
3. P2 -- K bound checked against W_MAX only; the drafter's buffers are
   D2_WMAX rows (pre-existing mismatch for wider W_MAX builds).
4. P2 -- test gaps: zeroed codebooks AND hidden removed all predecessor
   dependence (wrong chaining would pass); V=64 puts the whole vocabulary in
   one block so the fallback race is invisible; no max-K/bonus/mode-switch/
   captured-vs-eager cases.
5. P3 -- the fp32-cdf fallback slot's stored q excludes the cdf tail it wins
   (P(u >= cdf) ~ 1e-7): stored law != realised law by that much.

## Resolution

1. k_d2_stop_latch<<<1,1>>> writes scratch word 1 = (word 0 == 0) between the
   residual kernel and the fallback; the fallback reads word 1 only and
   reduces into word 0. d_amax (engine) and the test scratch are now 2 u64.
   Test (h): 40000-token vocabulary, sole nucleus token 39000 in a high
   block, 64 seeded rounds, d2 fallback == sample_stop == 39000.
2. d_cand memset at alloc; d2_setup zeroes every d_draft_L slot before the
   tail warm; stale logits/nucleus floats are only compared, never indexed.
3. d2_setup bounds K by min(W_MAX, D2_WMAX) - 1.
4. Test (g): random half codebooks + hidden projection; per trial the host
   recomputes q_pos conditioned on the DEVICE's previous pick and the stored
   rows must match (2e-4), the pick must sit in a q>0 slot, and picks must not
   all be argmax. Max-K / captured-vs-eager / mode-switch remain E2E-covered
   (seeded serving runs, greedy CLI identity) rather than unit-covered.
5. The walk now stores q_eff: the last q>0 slot absorbs 1 - cdf when the fp32
   cdf ends below 1, so the stored row equals the realised draw law.

Raw transcript: session scratchpad d2walk_codex_impl.out.
