# Continuous Batching P2c: Fused Draft Steps Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use conclave:executing-plans to implement this plan task-by-task.

**Goal:** Fuse the per-engine MTP draft steps into one weight sweep per step
(union over still-active engines), recovering the ~2.4 ms/round of serial
draft wall that P2a's overlap could not touch (weight-BW-bound physics), and
carrying the 2-slot aggregate from the measured 1.25x across the 1.3x bar.

**Architecture:** The exact P0/P1 pattern applied to `mtp_forward`
(engine.cuh:910-948, layer 64): union multi-lane weight ops (eh_proj mm, the
MTP ffn, the dominant 248320-row head mm) + per-engine sub-launches for the
MTP attention (`attn_block` on each engine's own mtp_k/mtp_v) and the
argmax_margin tail. At k=2-4 the union stays on the dp4a GEMV family
(< gemm_min) -- the SAME family each engine's solo draft uses -- and gemv
N-invariance is ninv-proven, so margins are bitwise identical to solo,
therefore caps/widths/verify/bytes are identical: the P2 master gate
(byte-identity vs scratchpad/p2_baseline/refs.md5) applies with NO
tolerance class. Fused steps run EAGERLY (cross-engine, no graphs); the
solo path keeps its captured draft graphs untouched.

**Tech Stack:** CUDA C++, existing multi-lane kernel family (embed3,
rmsnorm3, quantize3/mm5-pattern, add3), md5 gates.

---

## Context primer

- Branch p2-overlap (continues; P2c was gated GO by Task 4's 1.25x miss +
  the plan's >=1.5ms-draft-wall criterion -- drafts are ~5.6ms serial at
  k=2, P2a realized ~0). Baselines: scratchpad/p2_baseline/ (NOTE the
  refs.md5 methodology note: md5 = completion text + TRAILING NEWLINE).
- Read first: docs/plans/2026-07-15-batch-p2-overlap.md (all of it,
  including addenda B1-B8 -- they bind here too) and the P2b commit
  message (e36cb1f) for the gate discipline.
- mtp_forward anatomy (engine.cuh:910-948, il=64):
  1. embed_row_q8(tok) -> e_hn            [union: embed3-style per-lane]
  2. rmsnorm enorm(e_hn); rmsnorm hnorm(h_src) -> e_hn+N_EMBD   [union]
  3. qx(e_hn, 2*N_EMBD); mm(nextn.eh_proj) -> x_mtp        [UNION SWEEP]
  4. rmsnorm(attn_norm) -> x1; attn_block(64, x1, y, mtp_k, mtp_v, pos)
                                          [PER-ENGINE: own MTP KV]
  5. add_inplace; rmsnorm(post_attn); ffn(64, x1, y)       [UNION SWEEP]
  6. add_inplace; rmsnorm(shared_head_norm); qx; mm(head Q4) -> mtp_logits
                                          [UNION SWEEP -- the big one]
  7. argmax_margin -> draft token + margin [per-engine (own dst) or
     per-lane loop -- same kernel, own buffers]
- Per-engine state the fused step must respect: e_hn, x_mtp, x1, y,
  mtp_logits, xq slots, h_src (h_next chain / hs[k] D2D), tok_src chain
  (d_token/d_draft_L), pos ptrs (d_pos_m..m7), d_draft_L dst, margins,
  d_am_blk scratch, mtp_k/mtp_v. ALL Engine members -- build a small
  MtpLaneView (k <= MAX_K lanes) mirroring LaneView's pattern.
- The step CHAIN (spec_draft_step_launches, :1125-1142): step k>0 first
  does a D2D of x1 -> hs[k] on the engine's stm. In the fused world these
  per-engine D2Ds happen on cstm (or the engine's stm fenced properly --
  simplest: cstm, since the whole fused step runs on cstm and the P2a
  interleaved loop's per-engine launches are replaced wholesale for the
  fused path).
- The P2a interleaved loop (conductor.h, Conductor::draft_widths) is the
  integration point: per step, instead of per-engine draft_step_launch on
  member stms, ONE fused_draft_step(active_engines, k_step, cstm) +
  per-engine margin D2H + ONE sync + the same drop-out arithmetic (B8
  assert stays). draft_done events: record on cstm after each engine's
  last contribution (or simply after the loop -- the verify already runs
  on cstm, so ordering is free; keep the events for the API, record them
  on cstm).
- GEMM family: union k <= 4 << gemm_min 9 -> gemv family always; set the
  MtpLaneView's gemm_min = 99 with a comment (defensive; the head mm at
  union width must NEVER take vgemm since solo drafts use gemv -- same
  A1/Task-9 policy logic).
- Sampled members: draft_sample_bootstrap runs per-engine before step 0
  (unchanged); sampled ceilings differ (md_used=4) -- the active-set logic
  already handles per-member md_used.
- Suffix members don't draft (suffix_propose) -- excluded from fused steps
  exactly as they are from the interleaved loop today.

## Task 1: MtpLaneView + mtp_forward parameterization (solo byte-neutral)

**Files:** src/engine.cuh.
Mirror P0 exactly: `struct MtpLaneView { ... }` (per-lane: e_hn, x_mtp,
x1, y, lg(mtp_logits), xq slot, tok, pos, draft_dst, margin_dst, h_src;
shared: vw(=k), stm, gemm_min } + `mtp_solo_view()`. Parameterize
mtp_forward's per-lane ops into multi-lane calls over the view
(embed3-family, rmsnorm3, quantize3, mm5-family, add3) with the mixer
seam: `mtp_pre(v)` (steps 1-3), `mtp_attn(st)` (step 4, member-based,
stream param like gdn_mix), `mtp_post(v)` (steps 5-6), `mtp_tail(v)`
(step 7 per-lane loop). Solo mtp_forward = composition over
mtp_solo_view() -- byte-identical BY CONSTRUCTION, and the draft graphs
capture the same launch sequence (B-A8). CAREFUL: mm() and qx() are the
single-lane forms; the multi-lane forms need the SAME kernels the solo
path uses at ntok=1 (gemv_*_n at ntok=1, quantize3 at vw=1, rmsnorm3 at
vw=1) OR keep the single-lane kernels for the solo composition and use
multi-lane only in the fused path -- DECIDE by checking whether
gemv_q4_n(ntok=1) is bitwise == the mm() path's kernel (they may be the
SAME kernel; grep mm()'s implementation). If they differ, the solo path
KEEPS single-lane kernels (zero risk) and the fused path uses multi-lane
with the ninv guarantee making fused-vs-solo margins bitwise anyway.
State which branch reality took.
Gates: make + w16 + fused_smoke rebuild; test_kernels; canonical EXACT;
sampled-seed EXACT; test_conductor; fused_smoke fp16+turbo3 all legs;
master refs (fp8+t3, like-composition) EXACT. Commit.

## Task 2: fused_draft_step + interleaved-loop integration

**Files:** src/conductor.h (+ engine accessors if needed).
`build_mtp_union_view(Engine** es, const int* step_k, int k, cstm)` (each
engine may be at a different chain position ONLY in topup edge cases --
verify: the P2a loop advances all active members in lockstep, so step is
shared; top-ups happen after drop-out and are per-engine SOLO launches --
keep top-ups on the solo path per engine, they are rare (cap==0 only)).
`fused_draft_step(es, k, step, cstm)`: union pre -> per-engine mtp_attn on
cstm (k<=4 tiny attentions; side-stream forking is NOT worth it here --
state why: each is one token vs the verify's W lanes) -> union post ->
per-engine tail. Wire into Conductor::draft_widths behind the existing
loop structure: launch fused step, per-engine margin D2H on cstm, one
cstm sync, same margins/caps/drop-outs, B8 assert unchanged. draft_done
events recorded on cstm. Solo top-up launches stay on engine stms fenced
by an event from cstm (or move them to cstm too -- simpler, do that,
with a comment).
[SHIPPED CORRECTION (P2 exit review): the implementation keeps top-ups on
the MEMBER stm and records draft_done on e->stm (fused_round), not cstm --
correct because the per-step cstm sync completes the fused writes before
any top-up is issued (host program order), and draft_done on the member
stm then fences everything a member contributed OFF cstm (top-ups, suffix
prep/H2D, sampled bootstrap) while the fused steps need no event at all:
the verify runs on the same in-order cstm.]
Gates: same battery as Task 1 + the B4 double-run.
Commit.

## Task 3: measure + close P2

Full batch_ab REPS=3 legs A/B/D, fp8 + turbo3. Bar: fp8 B/A >= 1.3x.
Report phd walls (should collapse toward ~1ms). Solo-regression <2%.
BUILDLOG entry with the full P2 arc (P2a ~0, P2b +3%, P2c +X%). Then the
P2 Task 6 exit runs (fresh battery, TWO reviews, CC sanity, Makefile
conductor.h dep fix for build/q27-server (SECURITY approval needed),
merge/push only with Gabe's go.

## Non-goals
P3 shape-graphs (revisit only if Task 3 attribution says the launch tax
now binds); >=3-engine draft-fusion tuning; anything touching solo
capture paths beyond the byte-neutral Task 1 refactor.
