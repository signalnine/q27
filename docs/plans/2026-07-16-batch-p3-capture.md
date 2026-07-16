# Continuous Batching P3: Fused-Verify Capture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use conclave:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the measured 3.4 ms/round eager-launch tax (2,610
launches x ~1.66us GPU starvation each) by capturing the fused verify body
as CUDA graphs keyed on the exact round shape, taking 2-slot aggregate from
1.31x toward the ~1.44-1.48x ceiling. Bar: fp8 >= 1.38x.

**Architecture (design workflow wf_3bdf79af-8c0, judged synthesis):**
Approach A with D-grafts. The fused verify launch sequence is ROUND-
INVARIANT for a fixed key (ordered engine tuple, exact width vector,
sfx/gemm-family class, sampled mask, kv_kind) -- every pointer
build_union_view bakes is an init-fixed engine member EXCEPT the GDN role
buffers, whose (role+perm)%W_MAX resolution is host-side and consumed ONLY
by the conv_step/delta_step chains (judge-verified: engine.cuh:388-395,
:1355-1363; all other grep hits are comments). Therefore: (1) TABLE TWINS
of those two kernels index a once-uploaded [engine][gdn-layer][W_MAX]
role-pointer table via a per-engine device perm scalar (k pinned ints H2D
per round); (2) capture the WHOLE fused_verify_round per exact shape key,
cache ~32 execs LRU, eager fallback on miss, capture-then-launch same
round on first sight. Empirical shape alphabet: 28 keys/159 rounds, top-16
= 85-90% coverage (D's census). FALLBACK if spikes fail: D's segmented
capture (break at GDN mixers, no twins, ~1.7-2.0 ms). KILLED: device-side
graph launch (conditional nodes are host-launch-only, CUDA 13.2 headers).
DEFERRED: delta-chain kernel fusion (+0.4 ms busy; does NOT stack with
capture on launch gaps -- never sum them; forces re-capture if done later).

**Measured basis** (scratchpad/p3_measure/, nsys x2 modes + bare run,
cross-checked <2%): tax = 3.43 ms verify micro-gaps + 0.27 draft + 0.08
sync tails per 26.8 ms k=2 round; 2,610 launches/round (~2,075 verify);
capture ceiling 3.1-3.4 ms, net realistic 2.2-2.9. CAVEAT THAT SCOPES T3:
host gating rides on PAGEABLE D2H blocking semantics (margins + outcome,
~19 ms/round of host wait absorbed there) -- the capture boundary EXCLUDES
all D2H copies; do not move them.

---

## Context primer

Branch `p3-capture` off master 1a486ee. Everything from the P2 plans binds:
docs/plans/2026-07-15-batch-p2-overlap.md (B1-B8), the refs.md5
trailing-newline convention, systemd-run for ALL GPU runs, racecheck NEVER
on full-engine binaries, sanitizers in own scopes, no Makefile edits
without SECURITY approval (new test binaries get explicit nvcc lines or a
target bundle offered at exit). Baselines: scratchpad/p2_baseline/refs.md5
(the P2 master gate STAYS the master gate -- capture must be byte-neutral),
p3_measure/ (the tax numbers T4 is judged against). Key code:
conductor.h fused_verify_round (:530-602, launch-only body; draft_done
waits at :535 stay OUTSIDE the capture), draft_widths per-step host sync
(:1149, hard boundary -- draft phase is NOT capturable, the 07-14
GPU-side-depth NO-GO stands), B8 assert (:1195-1216, the discipline the
new key assert mirrors), engine.cuh RBuf/SBuf (:388-395), gdn_mix chains
(:1355-1363), build_union_view (conductor.h:395-411).

## Tasks (judge's skeleton, gates verbatim where given)

### T0: S0 standalone capture spike (0.5d) -- FIRST, kills cheap
Standalone .cu (tools/p3_capture_spike.cu, explicit nvcc line, no engine
code): on sm_120, stream-capture under Relaxed mode ~2000 dummy kernels
shaped like the launch census (incl. one fdmma-shaped ~100KB dynamic-smem
node with cudaFuncSetAttribute latched pre-capture) with per-layer 2-4
side-stream fork/join record/wait choreography (the P2b pattern). Measure:
(a) cross-stream fork/join capture legality on driver 580.119.02;
(b) replay saving per launch (need >= ~0.75us; the 1.66us starvation
invariant gives 2x margin); (c) instantiate cost at ~2000 nodes (bar
<= ~50ms, warmup-hiccup class); (d) per-exec device memory x 32 LRU.
KILL P3a -> fallback D on (a); KILL P3 entirely only if (b) < ~0.4us.

### T1: S1 decisive byte spike + key cardinality (1d)
Q27_P3_SPIKE env hack (NOT for merge): capture the REAL fused_verify_round
at the dominant (w,w) shape -- capture-without-execute, then launch the
instantiated exec for the SAME round -- through batch_ab LEGS=B. GATE:
refs.md5 byte-identical (any diff = capture-semantics state; hunt or
kill). Plus 2h census: A's FULL key cardinality from [bat] lines +
sfx/sampled classes in the p2c_t3/p2_exit logs; hot-key coverage >= ~85%
else rescope to D. Spike code carries a REVERT-BEFORE-T2 marker.

### T2: conv_step/delta_step table twins (1d)
Once-uploaded [engine][layer][W_MAX] role-pointer tables (~9KB/engine),
per-round k perm ints via PINNED staging + cudaMemcpyAsync on cstm before
graph launch (pinned specifically to avoid new pageable-blocking
semantics). Solo path keeps the original kernels untouched. GATE: new ninv
seam leg, twin-vs-original bitwise on BOTH arches (mirrors the P2c
single-vs-multi-lane leg) -- mismatch is DOA, no tolerance class; ptxas -v
spill=0 + register-delta review twin-vs-original (the P10 guard); refs.md5
recomparison (twins active in the eager path first).

### T3: conductor integration (1.5-2d)
Shape-keyed exec cache: key = (ordered engine tuple, exact width vector,
sfx/gemm_min class, sampled mask, kv_kind). Capture-on-demand under
Relaxed/ThreadLocal; LRU ~32 (whole-round zoo ~100 max, smaller than D's
segment zoo); eager fallback on miss; capture-then-instantiate-then-launch
same round; ALWAYS-ON key assert (re-derive the pointer table from host
ints, compare vs the hit's key -- B8 discipline); optional startup
pre-capture of the top-16 gated width pairs (hides 10-50ms instantiate in
warmup). Capture boundary EXCLUDES: draft phase, draft_done waits, timing
event records, ALL D2H copies. GATE: fused_smoke all legs both KVs;
test_conductor; B8 + key assert live and silent.

### T4: full battery + the bar (1-1.5d)
Master refs.md5 both KVs byte-identical; B4 self-determinism x2; canonical
a2982c5197c627551b27d76a0a94b220; sampled-seed; ninv (incl. twin + seam
legs); solo-regression leg D ~0% (solo untouched by construction);
memcheck small-footprint own-scope. THE BAR: batch_ab REPS=3 fp8 >= 1.38x
(turbo3 reported alongside). BUILDLOG entry with the P3 arc + the
capture-hit-rate/instantiate telemetry.

### T5 CONDITIONAL P3b: draft-step micro-graphs (+2d)
Only if T4 lands 1.35-1.38x (short by <= ~0.3 ms): per-(tuple, step, na)
draft-step graphs; margin D2H + per-step sync stay OUTSIDE (hard host
boundary). Same battery. Else skip to exit.

### T6: exit
Fresh battery, ultracode multi-lens review + adversarial verify (the P2
exit pattern), CC sanity pair, memory/BUILDLOG, Makefile target bundle if
any new tools (SECURITY), merge/push ONLY on Gabe's go.

## Non-goals
Kernel fusion (deferred, non-stacking); device-side graph launch (closed,
header-verified); mixer co-residency (separate lever, ~3.5 ms, own session);
moving margin/outcome D2H into graphs (scoped out by the pageable-gating
caveat).
