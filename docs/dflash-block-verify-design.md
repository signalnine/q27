# DFlash block-drafter + batched S=16 verify -- design (2026-07-09)

Status: DESIGN, pre-measurement. Nothing built. Phase-0 gates below decide
GO/NO-GO before any engine work (maxd6/fd3 discipline: measure the economics
first, kill cheaply).

## Motivation

The post-d7 decode ceiling is structural on the current machinery, by
measurement:

- Sequential draft tax: one MTP pass per depth level; d7's marginal lane
  cost (+2.33 ms/round post-launch_bounds) never pays even at y7 .81
  (BUILDLOG 2026-07-09 maxd7).
- Verify-side GEMV lane-amortization exhausts past ~7 lanes (width-8 ncu
  attribution: latency/issue-bound, DRAM demand FALLS at nb=8).
- fd2 decode attention pays +15-17%/lane for per-lane KV re-reads; the
  fd3 smem-sharing family is closed (bitwise-equal v3 still +44-61%).

A block-diffusion drafter changes BOTH cost curves at once:

- ONE drafter forward proposes 15 tokens (no sequential chain).
- ONE batched S=16 target forward verifies the block -- a GEMM (weights
  read once per tile of 16 tokens), NOT 16 GEMV lanes. This is the
  prefill cost curve, which we already ride at 2206 t/s.

## Prior art (FlashRT + z-lab, both permissive)

FlashRT (Apache-2.0, flashrt-project/FlashRT) ships this for the SAME
model (Qwen3.6-27B): drafter = z-lab/Qwen3.6-27B-DFlash (MIT, 3.3 GB BF16,
5 layers, block_size=16, vocab 248320 -- our tokenizer). Their measured
result on Thor (SM110), same process, greedy, vs their own MTP-K6 spec:

    robot-task JSON   AL 2.87 -> 4.57   33.7 -> 48.9 tok/s  (+45%)
    navigation plan   AL 2.59 -> 3.25   30.5 -> 34.8        (+14%)
    prose             AL 2.43 -> 3.00   28.5 -> 31.7        (+11%)

Lossless: verify is greedy ground truth, outputs byte-identical to their
MTP reference. Cycle anatomy: S=16 verify (~86 ms Thor, weight-read
bound) + drafter graph replay (~7 ms) + one ~10 us host sync for the
accept decision. Partial accepts restore per-step checkpoints written
during the verify -- no recovery forward.

Context for our numbers: their MTP baseline accepts 2.4-2.9/cycle at
fixed K=6; our adaptive ladder already runs 5.3-5.8 tok/round on live CC
traffic. Their DFlash gains are RELATIVE to a weak MTP baseline -- the
Phase-0 question is whether the drafter beats OUR ladder, not theirs.

## Drafter contract (from FlashRT's reference impl, verified against
lucebox-hub/dflash)

Inputs per cycle:
- prev accepted token id (scalar).
- hidden taps: (5, 5120) -- target hidden states at layers
  [1, 16, 31, 46, 61] from the most recent target step.
- Feature-window context: fc-projected target features of committed
  tokens. Two modes: per-token window (128-256 slots, one per committed
  token -- HIGHER acceptance, Thor default) vs per-cycle shift (legacy,
  lower AL). Build the per-token mode.

Forward (5 layers, all M=16 GEMMs + one M=1 fc):
    input_ids = [prev_token, MASK x 15] -> embed (16, 5120)
    target_feat = rmsnorm(fc @ concat(taps)) * hidden_norm   (1 row)
    per layer: rmsnorm -> QKV (K/V get target_feat row prepended,
    kv_seq = window+16), q/k head-norms, RoPE NEOX theta=1e7,
    SDPA non-causal (GQA 4:1), o-proj residual, SwiGLU FFN residual
    final rmsnorm -> lm_head (16, 248320); rows 1..15 = candidates

## q27 integration sketch

1. **Drafter runtime**: repack drafter to q27 Q4/Q8 (reuse repack
   pipeline; ~1 GB resident) or keep fp16 GEMMs (3.3 GB, simpler
   numerics first). 5-layer forward at M=16 is a handful of k_gemm_*_T
   launches + one small SDPA -- a single CUDA graph, ~1-3 ms on sm_120.
2. **S=16 verify**: reuse the batched prefill path (prefill_chunk(T=16)
   + per-row logits via the nll-path mmT(head)) with argmax per row.
   KV/attention rows land exactly as prefill does today.
3. **Hidden taps**: capture h at 5 layer boundaries for the LAST
   accepted row during verify -- five 20 KB device copies per cycle.
4. **Feature window**: ring of fc-projected features (256 x 5120 x 2 B
   = 2.6 MB) appended N+1 entries per cycle.
5. **Acceptance**: match-mask argmin + one host sync (their measured
   10 us), or device-side accept loop later.
6. **Partial accept -- THE hard q27 problem (P15 class)**: attention KV
   truncates for free (rows past accept point are dead, positions
   rewind -- ctx_round_reserve() bounds apply at S=16, reserve becomes
   ~18). GDN recurrent state does NOT truncate: the chunked scan
   advances S through all 16 positions. Options:
   a. Per-step S snapshots during the scan: ~3.1 MB/layer x n_gdn
      layers x 16 steps ~= 2.4 GB. Too hot at 131K serving VRAM.
   b. Checkpoint every 4th step (~600 MB) + serial GDN-only replay of
      <=3 tokens from the checkpoint (GDN serial step is cheap; conv
      ring is 48 KB). DEFAULT CANDIDATE.
   c. Recompute: GDN-only forward of the accepted prefix from
      round-start state (S_snap machinery exists). Simplest, costs a
      partial forward per partial accept.
   Decision deferred to Phase-2 with (b) as the working assumption;
   refinish_round is the rewind precedent.
7. **Coexistence**: `Q27_DFLASH=1` opt-in path beside the MTP ladder,
   never replacing it until beaten same-harness. Greedy only first;
   spec-sampled later (rejection machinery exists from Phase 2
   sampling).

## Bitwise contract

Verify is the greedy ground truth: emitted tokens MUST be byte-identical
to the plain/MTP paths per architecture. Canonical gates (a2982c51 base
5090, 6894254e 3090-fp16, 4c4120c7 qwopus) must hold EXACT with
Q27_DFLASH=1. The relaxed-thinking acceptance variant (TensorRT-LLM
policy, FlashRT +43% on think-heavy flavors) breaks exactness INSIDE
think blocks by design -- separate opt-in knob, tolerance-class, NOT part
of this build (we serve --no-think).

## VRAM budget (131K serving, 5090)

Today ~24 GB. Adds: drafter ~1 GB (Q4) or 3.3 GB (fp16) + window 3 MB +
GDN checkpoints ~600 MB (option b) + verify scratch (prefill buffers
already allocated). Q4 drafter fits with ~5 GB headroom; fp16 drafter is
tight but viable for Phase-1 bring-up at reduced ctx.

## Phase 0 -- measurement gates (no engine code)

- **P0a, acceptance on OUR traffic and OUR quant**: capture q27 hidden
  taps + committed tokens on cctx/CC-transcript traffic (extend
  --dump-logits-class plumbing or --stats), run the z-lab drafter
  offline in PyTorch against those taps, measure AL distribution.
  CRITICAL: drafter was trained against BF16 target hiddens; our 5.25
  bpw hiddens differ -- BF16-HF-target AL numbers do NOT transfer.
- **P0b, cycle cost**: time prefill_chunk(T=16) + 16-row head at 26K and
  61K depth (width_bench extension); drafter forward estimated from
  GEMM shapes, then measured in torch as an upper bound.
- **GO-IF**: AL / cycle_ms > (live tok/round) / (live round_ms) x 1.10
  at the cctx operating point (5.29 tok/round over ~26 ms today ->
  need AL/cycle_ms > 0.224 tok/ms). Example: AL 6.5 at 26 ms cycle
  passes; AL 5 at 26 ms dies.
- **Kill criteria**: P0a AL < 5.5 on cctx-class traffic (no headroom
  over the ladder), or P0b verify > ~35 ms at 26K (GEMM path not
  materializing), or quant-degraded taps collapse AL vs BF16 taps by
  >25% (drafter would need retraining -- out of scope).

## Phases

0. Measurements above (1 session, zero engine risk).
1. Drafter runtime + offline E2E: drafter in-engine, verify via existing
   prefill path, host-driven loop, correctness vs plain greedy
   (byte-identity), no graphs, no partial-accept optimization
   (recompute option c). 1-2 sessions.
2. Performance: CUDA graphs both sides, GDN checkpoint option (b),
   accept-loop tightening, feature-window per-token mode. 1-2 sessions.
3. Server integration + same-harness A/B vs the ladder (cctx replay +
   live CC trial), canonical/width gates, BUILDLOG verdict. 1 session.

## Risks

- GDN partial-accept state machinery is P15-class complexity (the
  split-brain/refinish territory) -- Phase 2's whole budget.
- Drafter-vs-quant mismatch (P0a exists to catch this first).
- RoPE NEOX theta=1e7 / head-norm details must match exactly or AL
  silently craters -- validate drafter logits against FlashRT's torch
  reference on fixed inputs before trusting any AL number.
- Our verify depth reserve grows to ~18 rows near ctx ceiling
  (ctx_round_reserve generalizes; the P0 #1 centralization pays off).

## Prior-art credits

FlashRT (flashrt-project/FlashRT, Apache-2.0) -- integration reference;
z-lab/Qwen3.6-27B-DFlash (MIT) -- drafter checkpoint;
lucebox-hub/dflash -- original block-diffusion drafter recipe.

---

## Phase-0 VERDICT (2026-07-09): NO-GO for the primary target

Measured same-day on the design's own gates (rig: `--dump-taps` +
`--p0b` in engine.cu, torch harness scratchpad/dflash_p0a.py):

**P0a -- drafter AL on q27-captured hiddens (base model, 5.25 bpw):**

    cctx (agent transcript, the serving flavor)  AL 2.10  (67% -> 46%
        cycles accept nothing; ctx-rows-raw fix took 1.41 -> 2.10)
    cctx, taps shifted one layer (+-1 convention) AL 2.18 -- a wash
    cctx, window 1 (per-cycle mode)               AL 1.70 -- ring works
    prose @8K                                     AL 2.65

Prose 2.65 vs FlashRT's published prose 3.0 (Thor, their impl)
validates the harness to first order. KILL BAR was AL < 5.5: measured
2.1-2.2 on the traffic that matters. The ladder's trained-in MTP nets
5.29 tok/round on the SAME traffic -- the 5-layer block drafter is not
close, independent of verify cost.

**P0b -- S=16 verify cost on the existing prefill path:**

    depth 1.7K: 41.3 ms | depth 26K: 47.2 ms | depth 61K: 53.2 ms
    (chunk+head; +1.1 ms for the 16-row head)

~Flat in depth => kernel-shape cost (T=1024-tuned prefill kernels at
T=16), not attention. Weight-read floor is ~10 ms; a DEDICATED small-T
GEMM verify could plausibly reach 13-18 ms -- but even a
floor-touching 12 ms cycle needs AL > 2.7 to match the ladder, and
needs AL ~5.5+ to justify the build. Not there.

**Why FlashRT's +30-60% didn't transfer:** their gains are measured
against a WEAK MTP baseline (fixed K=6, AL 2.4-2.9). Our
margin-gated adaptive ladder already extracts 5.3 tok/round from the
same trained-in MTP heads on agentic traffic; their best DFlash flavor
(4.57) never reaches our live baseline.

**Honest nuance:** on LOW-acceptance flavors (docs61k-class, ladder
~3 tok/round at ~26 ms = 0.12 tok/ms), drafter AL ~2.65 at a
hypothetical 13 ms dedicated cycle = 0.20 tok/ms -- DFlash could win
+60-70% THERE. But that requires the dedicated small-T verify (the
expensive Phase-1/2 build, P15-class GDN state machinery) and pays
only on the flavors our serving profile cares least about. Not
commissioned.

**DO-NOT-RETRY unless:** (a) a drafter retrained on q27-quant hiddens
exists, or (b) any block drafter shows AL >= 6 on agent-transcript
traffic, or (c) the serving profile shifts to low-acceptance flavors
AND the small-T verify gets built for another reason.
[2026-07-16 note: leg (c)'s second half has since materialized -- a
dedicated small-T tensor-core GEMM verify shipped 2026-07-13 as k_vgemm
(for the MTP/suffix ladder, union widths >= 9; BUILDLOG "GEMM-verify").
The serving profile has NOT shifted to low-acceptance flavors, so this
NO-GO stands; only the verify-cost half of any future re-evaluation is
now free.] The rig stays:
`--dump-taps <file>` (eager tap capture, graphs untouched, canonical
EXACT) and `--p0b` (S=16 cycle timing) in engine.cu;
scratchpad/dflash_p0a.py.
