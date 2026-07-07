# constrain-tools bundle: engage-lag fix + serving-state gates (2026-07-07)

Queue #6. Goal: `--constrain-tools` safe to default on in eval serving; unblock the
strict-parser A/B (zero rescues both legs) and the constraint-cost soak.

## The bug (07-04, score-0 basins)

The grammar engages when `<tool_call>` completes in DECODED text, but a spec round
decides up to maxd+1 tokens + the new pending token in one launch. Every decision made
in the round where the marker completes -- the round tail em[k+1..n-1] AND the new
pending -- samples UNMASKED. If those tokens enter the name region, the name prefix is
chosen free (hallucinatable); the mask then engages MID-NAME and forces a trie-legal
char into a diverged name ("getg_project" splice), the next byte goes illegal, the
grammar disengages, and greedy loops on tool-not-found.

Invariant the fix must establish: **every tool-name byte (and every in-grammar byte)
is decided under the mask** -- equivalently, the first post-marker decision onward is
masked. llama.cpp gets this for free (token-at-a-time); q27 must handle the round tail.

## Fix design: round truncation + re-finish at lane m (no snapshots)

Key structural fact (engine.cuh SBuf/RBuf + spec_round): the round's per-lane GDN
states live in the 6 rotating role buffers -- lane j's delta_step writes role j
(lane 0 in place at role 0), and finish "commits" by advancing `perm` by n-1 so old
role n-1 becomes the new role 0. Therefore state-after-lane-(m-1) for any accepted
m<=n is STILL RESIDENT post-round in old role m-1. Same for conv rings. Lane hiddens
x1/x1_b..x1_f and lane logits logits2[lane*VOCAB] are also resident. KV/MTP-KV rows
past the kept position are overwrite-safe (next round rewrites them). Decode-time
checkpoints don't exist (P9 saves only in prefill chunks).

So a mid-round engage can REWIND the round to m kept tokens and re-decide the pending
token under the just-staged mask, entirely from resident state:

`Engine::refinish_round(m, n)` (host-side, between rounds, ~zero cost, rare):
1. `perm = (perm + (m - n) + 6) % 6`  -- role 0 becomes state-after-lane-(m-1)
2. upload `*d_P = P_before_round + m` (4B H2D)
3. D2D `h_next <- x1_lane(m-1)`
4. `argmax_masked(logits2 + (m-1)*VOCAB, ..., d_mask_ids slot 0, d_token)` -- the new
   pending, decided under the engage mask staged earlier on the same stream
5. sync; read back `d_token` -> new `last_pending`; return it

generate() gains an `on_round(em, n) -> m` hook (server-only, like on_pending; CLI
never sets it -> canonical path untouched): called right after spec_round, BEFORE
emission. Return -1 = no action; m in [1..n] = marker completed at em[m-1]: truncate
emission to m tokens and refinish. m==n degenerates to pending-only re-decide (no perm
/ d_P change). Greedy-path only (constrained+sampled stays Phase 3).

ToolConstrainer moves to `src/toolconstrain.h` as
`template <class EngineT, class TokT> struct BasicToolConstrainer` (server aliases the
real Engine/Tokenizer; tests drive a FakeEngine/FakeTok -- api_common.h pattern):
- `scan_round(em, n)`: owns the rolling-tail marker detection (moved out of on_id).
  Sequentially decodes em[]; on marker completion at index j: tg.reset + advance(rem)
  + apply(tg) (stages mask + cap=1), sets skip_feed=j+1, returns j+1. Illegal rem or
  pool-full: no engage, return -1 (parser fallback path, logged).
- `on_id(id)`: active-state grammar feeding only (+closer/disengage); skips the first
  skip_feed tokens after a scan engage (they were already consumed by scan_round).
- on_pending / on_drafts / end: unchanged semantics.

## Serving-state gates (07-05 audit items a/b/c)

- (c) clear-at-claim: `generate()` entry clears a leftover device constraint iff one
  was set (`h_mask_id0 >= 0 || h_cap0 != 0` guard -> CLI/canonical path issues no new
  device ops). Kills the stale lane-0 mask + accept-cap-1 leak from a non-CUDA throw
  between generate() and tc.end().
- (b) pool-full parity: a -1 from mask_pool_add makes the constrainer STICKY-disabled
  for the remainder of the request (one log line + counter), instead of today's silent
  per-mask drop. Deterministic, visible, per-request scoped.
- (a) split-brain check: mask_id() validates a cached per-slot pool id against
  `eng->mask_pool_used`; out-of-range (the "someone reset the pool" case) logs and
  re-uploads instead of using a stale id.
- [req] line gains `tg=engaged/disengaged/pool` counters after `end=` (reqlog_gate-safe).

## Contract

- [x] C1 scan_round engages + truncates at the marker token: unit (FakeEngine): marker
      mid-round -> return j+1, mask staged, cap set, skip_feed set
- [x] C2 marker spanning tokens AND rounds engages exactly once (tail carry): unit
- [x] C3 rem bytes after the marker advance the grammar; illegal rem -> no engage,
      return -1, logged: unit
- [x] C4 no double-feed: kept tokens re-delivered via on_id don't re-advance tg: unit
- [x] C5 marker at last emitted token -> return n (pending-only refinish): unit
- [x] C6 pool-full -> sticky per-request disengage + counter; later calls in the same
      request don't re-engage; next request (fresh begin()) tries again: unit
- [x] C7 split-brain: stale host2dev id >= mask_pool_used detected + re-added: unit
- [x] C8 clear-at-claim: constraint left set is cleared at next generate() entry;
      never-set path issues no clear: unit-level via engine state probe + code gate
- [x] C9 refinish_round: constrained E2E decode emits a grammar-VALID call (name in
      tools, JSON parses, zero disengages) on a prompt where the marker completes
      mid-round: CLI harness (new `--tools-json` flag, greedy)
- [x] C10 round-phase invariance: C9 output bytes IDENTICAL across Q27_PMIN unset /
      0.5 / 1.0 (different round groupings shift where the marker lands in a round)
- [x] C11 canonical unchanged: md5 4c4120c72056aba2bc2d2561471eafce (no flag)
- [x] C12 test_kernels PASS; compute-sanitizer clean on the C9 run
- [x] C13 post-close decode unconstrained (closer disengages; text after the call
      decodes full-width): part of C9 assertions

## Measurement (after the fix lands)

- Constraint-cost soak: identical-request tool-heavy replay, --constrain-tools on vs
  off (wall + t/s + in-call fraction). The in-call cap is 1/round (~22 t/s in bodies).
- Strict-parser A/B: tolerant-parser rescues disabled (env), grammar ON both q27 legs
  vs llama grammar leg; gate = zero rescues both legs at non-degraded score.
- Q27_TOOL_SPLIT stays forbidden under --slots (P11 race, unchanged).
