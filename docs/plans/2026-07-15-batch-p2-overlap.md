# Continuous Batching P2 (Mixer + Draft Overlap) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use conclave:executing-plans to implement this plan task-by-task.

**Goal:** Recover the two measured serialization costs in the fused round --
per-engine mixers (9.1 ms serial at k=2) and per-engine draft phases (5.6 ms
serial) -- via stream-level overlap, taking the 2-slot aggregate from 1.21x
(fp8) toward the ~1.75x arithmetic in BUILDLOG 2026-07-15.

**Architecture:** Scheduling-only changes. (P2a) The conductor's per-engine
dexit draft loops interleave stepwise so engines' draft graphs run
concurrently on their own streams. (P2b) Per-mixer-layer fork/join: each
engine's gdn_mix/attn_mix launches on a conductor-owned side stream, fenced
by events around the union pre/post sections. No kernel changes, no new
numeric paths: every kernel receives byte-identical inputs and parameters,
so per-stream output must be BITWISE identical to P1's -- and that is the
main gate. (P2c, conditional) True draft fusion (union MTP weight sweep)
only if post-P2a measurement still shows >1.5 ms/round of draft wall.

**Tech Stack:** CUDA streams/events, plain CUDA C++, md5 gates. No Makefile
changes (targets exist since f45a9ad).

---

## Context primer

- Repo /mnt/ai/projects/q27, branch `p2-overlap` off master (014a920 or
  later). Main checkout, no worktree (standing preference).
- Read first: docs/plans/2026-07-14-continuous-batching-design.md ("Fused
  round anatomy" -- P2 is its step-3 second sentence), the P1 plan's context
  primer (gate commands, model paths, systemd-run rule, sensitive-file rule),
  and BUILDLOG entries "CONTINUOUS BATCHING P1" + the two 2026-07-15
  addenda (the measured walls this plan spends).
- Key code (line numbers at 014a920):
  - src/conductor.h:446-470 `fused_verify_round` -- the per-layer loop;
    mixers at :460 (attn) / :464 (gdn), all serial on cstm today.
  - src/conductor.h `Conductor::fused_round` -- calls draft_and_gate /
    suffix_propose PER MEMBER SEQUENTIALLY, then trim, then
    fused_verify_round. This sequential host loop is WHY drafts serialize.
  - src/engine.cuh:1905 `draft_and_gate` -- per-step graph launch + margin
    D2H + stream sync + theta check; :1942 `suffix_propose`.
  - src/engine.cuh:1110 `gdn_mix(il, st)` / :1167 `attn_mix(il, st)` --
    already stream-parameterized (P1 Task 8). Mixer state is per-engine
    (S/RBuf roles, kcache, d_part scratch) -- VERIFY this isolation claim
    for every buffer a mix touches before Task 3 ships (grep the members;
    any shared-DeviceModel scratch would be a data race under overlap).
  - src/engine.cuh:910 `mtp_forward` -- single-lane mm() calls (:926,:943);
    this is why P2c is expensive and deferred.
- Measured basis (batch_ab, w16 2x32K, k=2): fused round 29.9 ms = drafts
  5.6 (serial) + union weights ~11.3 + mixers 9.1 (serial) + ~3.9 tax.
  Overlap ceilings: mixers -> ~max(4.5,4.5)+eps saves up to ~4.5 ms; drafts
  -> saves up to ~2.8 ms. Post-P2 arithmetic ~20.6-22 ms -> ~1.6-1.75x.
- THE P2 MASTER GATE (new, stronger than P1's): scheduling-only means the
  batched CONCURRENT outputs must be byte-identical to the SAME inputs run
  at the P1 HEAD. Task 0 captures those references; every subsequent task
  re-runs them. Any diff = a bug, no tolerance class applies.
- All the P1 gates still apply on engine.cuh-touching commits: canonical
  a2982c5197c627551b27d76a0a94b220, sampled-seed vs
  scratchpad/p0_baseline/sampled_seed.md5, test_kernels, test_conductor,
  fused_smoke (all legs; run once with Q27_KV=turbo3 too), and the
  Q27_BATCH-unset replay (scratchpad/t12_defaultoff.sh pattern).

## Task 0: P2 baselines + branch

**Files:** create scratchpad/p2_baseline/ (gitignored); branch.
**Dependencies:** none.

1. `cd /mnt/ai/projects/q27 && git checkout -b p2-overlap && git rev-parse HEAD`
2. Fresh gates at branch point: make; test_kernels; canonical; sampled-seed
   (all must pass before anything else -- record outputs).
3. Concurrent byte references: run tools/batch_ab.sh legs "B" REPS=1
   MAXTOK=512 twice, once per KV:
   `KV=fp8 LEGS=B REPS=1 OUT=scratchpad/p2_baseline/fp8 bash tools/batch_ab.sh`
   `KV=turbo3 LEGS=B REPS=1 OUT=scratchpad/p2_baseline/t3 bash tools/batch_ab.sh`
   (via systemd-run --wait). Save every completion body + md5 list to
   scratchpad/p2_baseline/refs.md5. ALSO record the leg-B aggregate numbers
   (these are the perf baseline P2 must beat).
4. Commit nothing (scratchpad ignored); note HEAD in p2_baseline/HEAD.

## Task 1: fused-round phase stats (measurement first)

**Files:** src/conductor.h (+ src/engine.cuh only if GenStats needs a field).
**Dependencies:** Task 0.

The Task 9 TODO: fused rounds print phd/phv zeros. Add COARSE per-round
cudaEvent brackets on cstm: ev_round0 (before draft-wait), ev_draft_end
(after the draft_done waits land), ev_verify_end (after tails); after the
round's one sync, cudaEventElapsedTime accumulates into the members'
gs.draft_ms/verify_ms (split evenly per member? NO -- attribute the full
wall to EACH member's gs with a `fused` marker; document that fused walls
are shared, matching the [req] parse). Events live in a small per-Conductor
pool (create once, destroy in ~Conductor). Gate: batch_ab debug pass shows
non-zero phd/phv on fully-fused requests; references from Task 0 still
byte-identical (events don't change scheduling of work, only add markers);
canonical EXACT. Commit: "P2: fused-round phase walls via cstm events
(shared-wall semantics documented)".

## Task 2: P2a -- stepwise draft overlap

**Files:** src/conductor.h (the fused_round member loop), src/engine.cuh
(new step-granular entrypoints beside draft_and_gate).
**Dependencies:** Task 0 (Task 1 recommended first for attribution).

1. Engine entrypoints (mechanical extraction from draft_and_gate, keep
   draft_and_gate itself intact for the solo/smoke paths):
   - `void draft_step_launch(int k)` -- graph launch + margin D2H on stm,
     NO sync (lines :1913-1917 minus the sync/check).
   - `float draft_margin(int k) const` -- h_draft_margin[k] (host read,
     valid only after the caller synced stm).
   - `void draft_floor_topup(int cap, int md_used)` -- the :1925 loop.
   - sampled bootstrap (:1907-1910) stays in a `draft_sample_bootstrap()`
     called once before step 0.
2. Conductor: replace the sequential draft_and_gate calls with an
   interleaved loop over gated (non-suffix) members:
   active = members needing drafts; for k = 0.. while active nonempty:
   launch step k on every active member (their own stm); sync each active
   member's stm ONCE (cudaStreamSynchronize per stm, in member order --
   the syncs overlap the OTHER engines' still-running steps); read margins;
   drop members whose margin < theta (record their cap) or k == md_used-1.
   Then per member: floor top-up, want width, EXACTLY the values
   draft_and_gate would have produced (assert-equal in debug: run
   draft_and_gate's arithmetic on the recorded margins and compare).
   Suffix members keep suffix_propose (already one-shot, no loop).
   Per-member md_used comes from the same dctl/gate_maxd read -- hoist it
   before the loop, per member.
3. Gates: Task 0 references BYTE-IDENTICAL (both KVs); fused_smoke ALL legs
   (it drives the conductor -- rebuild first); canonical + sampled-seed
   EXACT; test_conductor ALL PASS. Then a quick measured sanity: batch_ab
   KV=fp8 LEGS=B REPS=1 -- aggregate should improve by roughly +1.5-2.5
   t/s-class... no: expected round saving ~2-3 ms of 29.9 -> aggregate
   ~+8-10%. Record the number, no bar here (Task 5 is the bar).
   Commit: "P2a: stepwise draft overlap -- engines' draft graphs run
   concurrently; per-member gate values identical (byte-identity refs green)".

## Task 3: P2b -- mixer side-stream fork/join

**Files:** src/conductor.h (fused_verify_round + Conductor stream pool).
**Dependencies:** Task 2 (same file region; serial to avoid conflicts).

1. Pre-flight isolation audit (blocking): list every device buffer
   gdn_mix/attn_mix touch (read engine.cuh :1110-1230): S/RBuf roles,
   conv scratch, kcache/vcache, d_part, per-lane arrays, plus any
   DeviceModel-owned scratch. Each must be per-engine. If ANY is shared
   across engines (DeviceModel member), STOP and report -- that is a P2b
   blocker requiring per-engine duplication first.
2. Conductor owns `cudaStream_t side[MAX_K]` (created in the ctor with
   cudaStreamNonBlocking, destroyed in ~Conductor -- NOT the engines' stm,
   to keep the draft_done/stm ordering contract untouched).
3. fused_verify_round, per mixer layer il: record ev_fork on cstm (after
   the union pre + any preceding union ops); for m: side[m] waits ev_fork,
   es[m]->{attn,gdn}_mix(il, side[m]), record ev_mix[m] on side[m]; cstm
   waits every ev_mix[m]; union post continues on cstm. Event pool sized
   MAX_K+1 per layer-use, REUSED round-robin (events are reusable after
   the wait lands; one pool of 2*(MAX_K+1) is enough with the per-round
   sync -- justify in a comment).
4. Gates: Task 0 references BYTE-IDENTICAL both KVs (the big one -- mixers
   on different streams MUST NOT change any lane's bytes); fused_smoke all
   legs incl Q27_KV=turbo3 run; canonical + sampled-seed; test_kernels.
   Plus a race gate: compute-sanitizer --tool racecheck on fused_smoke
   (racecheck instruments shared memory -- the smoke's small shapes keep it
   tractable; if runtime explodes, --tool memcheck on the small-footprint
   server config from scratchpad/t12_sanitizer.sh instead, and say so).
   Commit: "P2b: per-engine mixers fork onto conductor side streams,
   join before union post (byte-identity refs green, racecheck clean)".

## Task 4: measure -- the P2 bar

**Files:** docs/BUILDLOG.md (+ tools/batch_ab.sh only if a flag is needed).
**Dependencies:** Tasks 2+3.

Run tools/batch_ab.sh full legs "A B" REPS=3, KV=fp8 AND KV=turbo3 (w16
2x32K, the standing shape). Numbers to report: aggregate A, aggregate B,
B/A ratio, per-request medians, phd/phv fused walls (Task 1), trim counts.
THE BAR: fp8 B/A >= 1.3x (the original design bar P1 missed at 1.21x).
Stretch reference: ~1.6-1.75x arithmetic. Solo-regression check (LEGS=D)
must stay within 2%. If the bar fails AGAIN: attribute with the Task 1
walls (mixer overlap realized? draft overlap realized? tax grew?), write
the honest BUILDLOG entry, and STOP -- P2c/P3 decisions go back to Gabe
with the numbers. If it passes: BUILDLOG entry with the full table.
Commit: "P2: 2-slot aggregate <X.XX>x (bar 1.3x) -- draft+mixer overlap
measured; fp8 + turbo3 tables".

## Task 5 (CONDITIONAL): P2c true draft fusion -- go/no-go

**Dependencies:** Task 4.
Go criterion: Task 4's fused walls still show >= 1.5 ms/round of draft
wall at k=2 AND the bar needs it (or Gabe wants the ceiling). The work:
lane-parameterize mtp_forward (mm() -> mm5-family over a 2-4 lane
cross-engine view, same LaneView pattern as P0) so all engines' draft step
k is ONE weight sweep. This is a P0-sized refactor of the MTP path with
its own byte gates (draft margins bitwise vs solo) -- write it as its own
plan if triggered; do NOT improvise it inside this one. If no-go: state so
in the BUILDLOG entry and close P2.

## Task 6: phase exit

**Dependencies:** Task 4 (or 5 if triggered).
- Fresh full battery: make clean build, test_kernels, canonical,
  sampled-seed, test_conductor, ninv_test, fused_smoke (fp16 + turbo3),
  Q27_BATCH-unset replay vs p0_baseline, Task 0 refs one last time.
- Code review (conclave:requesting-code-review), then the SECOND pass
  (state-heavy rule) focused on the new stream/event choreography.
- Live CC sanity (single run, not a study): one concurrent T2+T5 pair on
  the batched turbo3 2x96K shape -- zero errors + bat>=2 + walls in family.
- Merge to master + push ONLY with Gabe's go (ask, do not assume this
  time either). Memory + BUILDLOG updates.

## Sequencing

Task 0 -> 1 -> 2 -> 3 -> 4 -> (5?) -> 6, strictly serial (Tasks 1-3 share
conductor.h; every task is gated on the byte-identity references).

## Non-goals

P3 shape-graphs / device perm indirection (only if Task 4's tax attribution
says launch overhead is the new binding constraint -- it was ~1.9 ms at P1);
>2-engine-specific tuning; Q27_BATCH default flip; turbo3-vs-fp8 quality
(separate PPL+needle session).
