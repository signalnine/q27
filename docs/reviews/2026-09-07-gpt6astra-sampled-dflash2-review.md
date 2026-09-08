# gpt-6-astra review: sampled DFlash2 verify (2026-09-07, two-part)

Review of commit dcbab9b (DFlash2 sampled verify + the forced-pending and
truncation-rollback state fixes). Part 1 ran adversarial-first and hit the
provider's usage limit mid-pass -- but its open thread (round truncation vs
drafter state) is what surfaced the ring-rollback bug that landed IN dcbab9b.
Part 2 (after the limit reset) completed the pass.

VERDICT: request changes -- two findings, both fixed same evening; "the
sampled rejection construction is sound."

1. [P1] ctx-guard reserve excluded DFlash2's width: with Q27_MAXD=4 +
   Q27_SUFFIX=0 + K=7 the reserve was 6 while the d2 verify writes 8 KV
   lanes -- a prompt admitted at max_ctx-6 wrote past the KV allocation
   (the depth-5-era overrun class). PRE-EXISTING on greedy d2, newly
   reachable by sampled requests. FIX: verify_w_max() now includes d2_k+1
   when d2_on.
2. [P2] Dflash2::rollback() only shrank the host ctx_n; attention reads the
   device mirror *d_ctx_n, which only ingest() uploaded -- the next draft
   (before its ingest) still attended the discarded rows. FIX: draft()'s
   per-round H2D staging now refreshes d_ctx_n unconditionally (4 B/round),
   so every draft sees the current host count.

Verified sound:
- Truncation ordering + m==n (rows/position kept, pending refreshed --
  needed because greedy grammar engagement can change pending without
  truncating; the shipped code does refresh pending for all m<=n).
- Forced install: overwrites device pending + d2_pending before drafting,
  correctly leaves ring/position alone (the replacement pending is not yet
  forwarded); sampled staging disables the bootstrap.
- Sampled ingest positions: n = pending + accepted drafts at P+1..P+n; the
  resampled/bonus token stays pending and is ingested next round.
- Rejection theorem with drafter-dependent proposals: conditional on
  drafter history the proposal is a point mass; accept with p(d), else
  sample p excluding d, recovers p exactly -- cache-dependent drafter state
  does not invalidate it (cold/warm seed divergence is compatible with
  distribution preservation).
- Greedy dispatch/verification unchanged; no token-selection regression.
- Flagged contract gap (pre-existing, not fixed): a future SAMPLED on_round
  callback returning m<n without forcing/cancelling would consume a
  provisional argmax unsafely -- the current callbacks never do.

Raw transcripts: session scratchpad d2samp_codex.out / d2samp_codex2.out.
