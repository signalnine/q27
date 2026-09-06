# DFlash2 drafter integration -- design v2 (2026-09-06)

Status: DESIGN, pre-measurement. Supersedes
`docs/dflash-block-verify-design.md` (2026-07-09, v1 drafter, parked at
Phase 0). This is a delta document: the v1 doc's motivation, bitwise
contract, and Phase-0 discipline carry forward; the drafter generation, the
verify plan, and two of the hardest problems changed -- both in our favor.

## Why reopen it

Measured 2026-09-06 (bench/crossengine/NINFER-REBENCH.md): ninfer's DFlash2
integration runs Qwen3.8-27B at 221/181 t/s (short/51K ctx) against a
paired MTP3 control at 157/133 on the same artifact, engine, and
instrument -- the whole +40-50% margin is the drafter, at only 37-39% lane
acceptance with K=7. That is the first working DFlash-family win on this
box (mainline vLLM ran it degraded and lost; llama.cpp does not wire it),
and it beats our own 3.8 sweep rows. The v1 doc's question stands with new
evidence behind it: does the drafter beat OUR ladder, on OUR quant, same
harness -- but the prior on GO moved substantially.

## The v2 drafter (z-lab/Qwen3.8-27B-DFlash2, MIT)

Verified against the checkpoint on disk and ninfer's algorithm contract
(their doc cross-checks vLLM and SGLang reference implementations):

- 5 non-causal sliding-window (2048) layers, H=5120, I=17408, 32Q/8KV
  heads, D=128, RoPE theta 1e7 NEOX, our tokenizer (vocab 248320).
- **1.92B params, 3.85 GB bf16, NO embedding or lm_head of its own** --
  it reuses the target's, exactly like the MTP head. Q4-family repack
  lands ~1.0-1.2 GB resident (fc [5120,25600] quantizes; ninfer stores it
  W8G32).
- Target feature taps at layers **[5,19,33,47,61]** (v1: [1,16,31,46,61]):
  s_t = concat of the five residual streams (25600), c_t =
  rmsnorm(W_fc s_t) * w_context. Context K/V for all five draft layers
  project DIRECTLY from c_t -- no input norm, no conv, no Q path on
  context rows. Cheaper context maintenance than v1's window modes.
- Block semantics: anchor = the pending target token; K masks predict
  absolute positions F+1..F+K in ONE forward (no shift). K is a runtime
  choice 1..15, no reconversion; K=7 is the recommended and
  ninfer-measured point.
- Candidate selector (the genuinely new machinery vs v1): top-M=16
  logits per mask position, a rank-256 hidden projection, and
  successor/predecessor codebooks scoring adjacent transitions; one path
  walk picks the K draft tokens. Three small tensors; the walk is
  host-trivial or a tiny kernel.

## What changed since the v1 design -- two scope collapses

1. **No S=16 GEMM verify build.** K=7 means verify width W=K+1=8, and the
   engine ALREADY captures per-width verify graphs 2..8 (M1b set, in
   every serving boot log). DFlash2's drafts enter the same fused verify
   the MTP ladder feeds today; the batched-prefill verify path the v1 doc
   designed is unnecessary for bring-up (it returns only if a K>=11
   sweep says wider blocks pay).
2. **GDN partial accept is (probably) already solved.** The v1 doc's
   Phase-2 monster -- GDN state does not truncate on partial accepts --
   predates M1 record-then-fold (shipped 07-14..16). The ladder does
   prefix-partial accepts at width <=12 every round today, and DFlash2
   acceptance is the same shape: a path accepted up to the first
   mismatch is a prefix accept. Working assumption: the existing fold
   machinery covers it unchanged. **P0c below exists to confirm this in
   code before it is believed.**

What did NOT change: the bitwise contract (greedy output byte-identical to
the plain/MTP paths; canonical gates hold EXACT with the feature on), the
drafter-vs-quant risk (trained on BF16 target hiddens; our 5.25-bpw
residuals + fp8 KV differ -- P0a exists for this), and coexistence
(`Q27_DFLASH2=1` opt-in beside the ladder, never replacing it until beaten
same-harness).

## The composition play nobody else has

Every other integration REPLACES its MTP path with DFlash2. q27's verify
already merges two draft sources (MTP ladder + suffix drafter) into one
width-12 round. DFlash2 at K=7 occupies 8 columns and leaves 4 -- enough
for suffix lanes to ride on top exactly as they do now. On echo-heavy
stretches the suffix drafter is the reason live traffic hits 5.3-5.8
tok/round; DFlash2's wins are on prose/code where suffix goes quiet. The
two are anti-correlated by construction, which is the best case for
stacking. Phase 3 measures DFlash2-alone vs DFlash2+suffix vs
ladder+suffix; the stack is the candidate default.

## Cost model (to be measured, not believed)

- Drafter forward: reads ~1.0-1.2 GB (Q4) once per round at W=8 columns
  -> ~0.6-0.9 ms weight-bound on a 5090, plus small SDPA over the 2048
  window and the selector. Estimate 1-2 ms/round; bf16 bring-up ~2.5-3 ms.
- Tap capture: 5 x 5120 x W values per verify round -- device copies at
  the layer boundaries, microseconds; fc+norm per committed token ~0.26 GB
  read amortized over accepted tokens.
- Round arithmetic at the cctx anchor (re-measure in P0b): ladder+suffix
  runs ~5.3 tok/round over ~26 ms. GO needs
  AL_dflash2 / (round_ms + drafter_ms) > 1.10x today's tok/ms with
  suffix contribution held equal on both sides -- the honest comparison
  is (dflash2+suffix) vs (ladder+suffix), same replay.
- Evidence prior: ninfer's 37-39% lane acceptance at K=7 yields ~3.6
  tok/round from the block alone before any suffix contribution.

## VRAM (131K serving, 5090)

Today ~24 GB. Adds: drafter ~1.0-1.2 GB Q4 (3.85 GB bf16 for Phase-1
numerics-first bring-up at reduced ctx), context ring ~5120 x window x 2B
per resident conversation slice (bounded with the sliding window at 2048:
~21 MB, plus K/V context stores per draft layer ~105 MB), selector
tensors ~10 MB. Q4 path fits with headroom; bf16 path is viable for
bring-up.

## Phases

0. **Measurement, zero engine risk (1 session).**
   - P0a acceptance-on-our-taps: capture q27's five-layer residual taps +
     committed tokens on cctx/CC-replay traffic (stats-path plumbing),
     run the bf16 drafter offline in torch against those taps, measure
     the AL distribution. Validate drafter logits against the vLLM/SGLang
     reference on fixed inputs FIRST (selector + RoPE details silently
     crater AL if wrong). Kill: AL-equivalent < the ladder's yield on the
     same replay, or quant-tap degradation > 25% vs BF16-target taps.
   - P0b cycle cost: drafter forward timed in torch as the upper bound;
     verify width-8 cost is already known from the ladder's own graphs.
   - P0c fold audit: code-read that record-then-fold's partial-accept
     path has no ladder-specific assumption a block drafter violates
     (draft provenance, pending-token bookkeeping, dexit interplay).
1. **Drafter runtime + offline E2E (1-2 sessions).** Repack (or bf16
   load) behind `Q27_DFLASH2=1`; tap capture in the verify path; drafter
   forward + selector; feed the existing width-8 verify; byte-identity vs
   plain greedy on the canonicals; no graphs, no tuning.
2. **Performance (1 session).** Drafter under its own CUDA graph, K sweep
   (5/7/9/11), suffix-stacking wiring, fp8-KV interaction check.
3. **Same-harness verdict (1 session).** cctx replay + live CC trial:
   ladder+suffix vs dflash2+suffix vs dflash2-alone; canonical/width
   gates; BUILDLOG verdict and default decision.

## Risks

- Selector correctness: codebook/transition semantics are the new
  surface; the reference-logit validation in P0a is the guard.
- Quant-tap mismatch (P0a's whole reason to exist; v1 risk, unchanged).
- Tap capture on the CLI/reference path must stay dormant so bitwise
  canonicals are untouched (server-only, like fp8 KV).
- The composition bet could fail: suffix and DFlash2 may fight over the
  same accept opportunities on mixed traffic. Phase 3's three-way A/B is
  designed to catch exactly that.

## Prior art

z-lab/Qwen3.8-27B-DFlash2 (MIT, on disk at
`/mnt/ai/models/qwen38-27b-dflash2-bf16`); ninfer's algorithm contract and
op checklist (docs/maintainer/qwen3.8-27b-dflash2.md at their master,
cross-checked against vLLM `qwen3_dflash2.py` and SGLang's DFlash worker);
FlashRT and lucebox-hub/dflash per the v1 doc's credits.
