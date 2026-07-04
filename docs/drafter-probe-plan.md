# Drafter probe plan (PARKED -- decode-side, not active)

Question this answers before any DSpark-grade work: **does a purpose-trained
drafter beat Qwopus's existing MTP head enough to justify a q27 draft-path
fork?** Cheap experiment first (days), full parallel-backbone build only if
the probe says yes. See [[reference-dspark]] for why: acceptance is already
near-ceiling (measured p(d5|prefix4)=96.8%), so a new drafter's only real win
is the round-cost curve (~7%/depth MTP -> ~0.3%/depth parallel), which only
matters if decode t/s is the priority. It currently is NOT (wall time is
prefill + was in-call constraint, now P11).

## Probe design (Eagle-MLP, days not weeks)

- **Drafter arch**: pure-MLP head (NO attention) so the hybrid GDN target is
  irrelevant -- the drafter only consumes the target's last hidden state
  (5120-dim) + the embedding of the current token, predicts next hidden ->
  reuse frozen lm_head for logits. 2-3 MLP layers, ~0.2-0.4B params. This
  sidesteps DeepSpec's dense-only limitation entirely (its drafters have
  attention; ours doesn't).
- **Why MLP is a fair probe**: if even a cheap MLP drafter can't beat the
  MTP head's acceptance on our traffic, a fancier one won't either. If it
  ties/beats, THEN the parallel-backbone build is justified.

## Steps

1. **Data gen (hours on 5090)**: run q27 (or llama.cpp Qwopus) over the
   SERVING distribution -- agentic/code/no-think, NOT generic chat. Reuse
   Thunderdome transcripts + a code corpus. Collect (hidden@output_norm,
   next_token) trajectories via the existing `--stats` hidden-dump path
   (hid_dump already exists in engine.cu E7 gate). ~10-50M tokens.
2. **Train (overnight, 1x5090)**: teacher-force the MLP to match the target's
   next hidden (smooth-L1) + CE on next token. Optional: DSpark's
   total-variation term for acceptance. PyTorch, standalone, not in q27.
3. **Acceptance A/B (the actual result)**: measure p(d1..d4) of the trained
   MLP drafter vs the MTP head on held-out agentic traffic, using the same
   margin-bin harness as `--stats`. Decision gate: MLP drafter's accepted
   tok/round > MTP head's 4.36 by enough to matter (>~10%) AND on the
   agentic/think-heavy distribution where the MTP head is unmeasured.

## If the probe says yes

- Parallel-backbone drafter (DSpark/DFlash style) trained via adapted
  DeepSpec (hybrid arch work) -- weeks.
- New q27 draft path: parallel forward replacing the 4 sequential MTP passes
  (round-critical surgery, same code P11 touched).
- Margin-gated adaptive depth (portable from DSpark, uses the free --stats
  margin signal) -- ships independently of the drafter, do this first.

## Priority

Behind all prefill/wall-time work. The measured agentic gap is prefill +
per-turn re-prefill, not draft depth. Revisit only if decode t/s becomes the
headline again.
