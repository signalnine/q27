# Continuous batching across active slots (2026-07-14)

## Problem

Concurrent generations time-slice the GPU behind the R1b FIFO gate at round
granularity. Decode is weight-bandwidth-bound, so N active requests each get
~1/N of the weight-read bandwidth: aggregate throughput ~= single-stream.
Under Claude Code subagent fan-out (2-4 concurrent conversations, warm prefix
caches, decode-dominant) the box leaves ~half or more of its aggregate decode
on the table.

Measured basis (07-14 triage, BUILDLOG): warm decode wall splits verify 83% /
draft 15% / suffix 2.5%; inside a wide round the weight GEMV is ~73% and is at
its SOL floor (k_vgemm, 07-13). The only structural lever left is sharing the
weight sweep across requests: the multi-lane kernels' marginal lane cost is
~9% (gemv) to ~flat (vgemm at N>=9), so one sweep serving 2-4 requests' lanes
approaches Nx aggregate.

## Decisions (interactive brainstorm, Gabe)

1. TARGET: CC subagent fan-out. Success = aggregate decode t/s at 2-4
   concurrency approaching Nx solo, solo path untouched. Prefill keeps
   time-slicing (chunk granularity behind the gate) as today.
2. ARCHITECTURE: lockstep conductor over existing engines. Engines keep ALL
   state (KV, GDN roles, snapshots, ckpt ring, suffix index, depthctl).
   Batched weight sweep over the union lane set; per-engine mixer
   (attention/GDN) and accept/commit sub-launches. No mega-engine refactor.
3. LANE POLICY: per-engine gates + cap. Each engine keeps its own P12-gated
   width and suffix width; if the union exceeds W_MAX, trim the widest
   requesters first, suffix lanes before gated lanes.
4. DETERMINISM: bitwise-when-untrimmed. A stream's bytes in a batch == its
   solo bytes whenever its width was not trimmed (requires N-invariant
   batched GEMV, gated by test); trimmed rounds are tolerance-class (same
   family as the known fdmma width forks); same batch composition always
   reproduces the same bytes. Canonical CLI gate untouched by construction
   (solo path never enters the fused round).
5. PHASING: verify-only first (P1), draft fusion and mixer overlap later
   (P2), shape-graphs only if measured necessary (P3).

## Feasibility facts (verified in source, 07-14)

- P3/CP3/XQ3/IP3/FCP3 are PER-LANE POINTER structs (kernels.cuh:67,
  spec3.cuh:13, fdmma.cuh:33). The GEMV/vgemm/elementwise path can take lanes
  from different engines today; cross-engine batching there is plumbing.
- launch_fdmma takes ONE kc/vc (fdmma.cuh:412): attention shares K/V tiles
  across lanes of one sequence. Cross-sequence attention has zero shared
  reads, so batching it has no value; it stays per-engine. Same for the GDN
  chain and kv_store (per-sequence state).
- spec_round_launches() (engine.cuh:1236) is the eager round body that graph
  capture records; it already runs eagerly at warm-up. The fused round reuses
  this code, refactored to lane-set parameters -- no second forward is
  invented.
- k_vgemm loads lane columns with tt<T zero-pad guards over a fixed K-stage
  order; each output element is an exact MMA dot product whose accumulation
  order does not depend on lane count or slot position. Bitwise N-invariance
  is realistic and testable. The narrow gemv family (independent per-lane
  accumulators, fixed K order) likewise.
- The union width exists ONLY in the weight sweep. Per-engine attention runs
  at that engine's own width (2-8 typical), so fdmma W>=14 occupancy cliffs
  never apply. vgemm engages at union >= 9 (flat in W). W16 build
  (-DQ27_W_MAX=16, W_PLUMB=16 plumbing shipped) is the natural serving
  target for 3-4 slot headroom.

## Scheduler

A dedicated CONDUCTOR thread owns decode scheduling; request threads own
everything else.

- Request thread: claim slot -> tokenize -> prefill (behind GpuGate, as
  today) -> REGISTER engine with conductor -> block on a per-request token
  queue, streaming SSE as tokens arrive -> on EOS/n_max/cancel, unregister
  and reclaim the slot.
- Conductor loop: collect decode-ready engines -> form this round's batch ->
  GpuGate acquire -> one fused round -> post each engine's committed tokens
  to its request thread -> release/yield -> repeat.
- Solo case (1 registered engine): conductor launches that engine's existing
  captured round graphs. The solo path is byte-for-byte today's path; fusion
  only engages at >= 2.
- Membership changes at round boundaries only. Joins land at the next round;
  cancellation marks the member and its lanes are absent from the next batch
  (no mid-round teardown).
- GpuGate survives unchanged as arbiter between the conductor and prefill
  chunks; the conductor is one well-behaved gate client, so cold prefills
  interleave at chunk granularity exactly as today.
- Token callbacks (SSE writes, StreamSplitter, tool-constrain) move to
  request threads. Tool-constrain hooks mutate engine state mid-generation:
  those callbacks stay synchronous with the conductor loop (posted and
  acknowledged before that engine's next round).

## Fused round anatomy

1. Per-engine DRAFT: each engine's existing captured draft-step graphs run
   serially with their dexit margin checks (P1; fused draft steps are P2).
2. WIDTH DECISION: each engine computes gated width / suffix width exactly as
   today; trim rule caps the union at W_MAX (suffix before gated). Granted
   (post-trim) widths feed depthctl/suffix bookkeeping.
3. FUSED VERIFY FORWARD (eager): layer loop over the union lane set. Union
   rmsnorm3/quantize3/gemv/vgemm calls take pointer structs filled from
   different engines' lane buffers. At each mixer layer, drop to per-engine
   sub-launches (own kc/vc, GDN roles, own width). P1 runs mixers serially on
   the conductor stream; P2 forks them onto per-engine side streams with
   join events (they are independent).
4. Per-engine ACCEPT/COMMIT: existing kernels unchanged, including perm
   rotation (the commit), suffix index update, depthctl EMA.
5. One outcome D2H + stream sync for the whole batch.

Eager launch is fine here: the host enqueues a layer (~12 launches x ~3us =
~36us) far faster than the GPU consumes one (~200us); graphs are revisited
only on measurement (P3 gate below). No new device allocations of
consequence: the batch is pointer plumbing over buffers engines already own.

## Expected value (from the measured split: draft 15 / weights ~60 / mixers ~25)

- P1, 2 slots: serial drafts (30) + fused weights (~62) + serial mixers (50)
  ~= 142 units per 2 streams -> ~1.4x aggregate.
- P2 mixer overlap: ~117 -> ~1.7x. P2 draft fusion: ~105 -> ~1.9x.
- 3-4 slots (W16 build): weight sweep stays ~flat -> toward 2.5-3x.
These are planning numbers, not promises; the P1 A/B is the arbiter.
- MEASURED P1: 1.21x at 2 slots (bar 1.3x missed) -- serial mixers + serial
  drafts came in as priced; the miss is +2.0 ms/round GEMV-family width
  scaling and +1.9 ms/round eager launch tax, both unpriced above; P2
  arithmetic from the measured walls ~1.75x (BUILDLOG 2026-07-15).

## Gates

1. Canonical CLI gate a2982c51 EXACT after P0 refactor (and every phase).
2. N-invariance kernel gate: vgemm_test/gemv tests extended -- identical lane
   inputs at different union widths and slot positions must be bitwise
   identical per lane.
3. Solo-equivalence replay gate: two conversations served batched vs solo,
   per-stream md5 identical on untrimmed traffic (adaptive_sfx_gate.sh
   pattern).
4. Composition determinism: same batch mix twice -> identical bytes.
5. Headline A/B: 2/3/4 concurrent replays, conductor vs FIFO-gate baseline
   (Q27_NO_INTERLEAVE and default interleave both). P1 bar: >= 1.3x
   aggregate at 2 slots.
6. test_kernels + compute-sanitizer (dual-arch build) as always.

## Phasing

- P0: refactor spec_round_launches() into lane-set-parameterized helpers.
  ZERO behavior change; gates 1/3/6 green before any batching lands.
- P1: conductor + fused verify, k <= 4, greedy AND sampled verify (serving
  default is T<=0.7 -- greedy-only would never fire in production), trim
  policy, gates 1-6, measure at 2 slots.
- P2: mixer side-stream overlap; fused draft steps (engines drop out at
  their own early-exit depth); re-measure at 2-4 slots.
- P3 (conditional): shape-graphs via device-side perm indirection, only if
  P1/P2 measure the eager host tax > ~2-3% of round wall.
  [RESOLVED 2026-07-16: condition met (eager dispatch tax measured 3.4
  ms/round); P3 built (table twins + shape-keyed LRU graph cache) and
  passed its bar -- 1.41x fp8 / 1.40x turbo3, BUILDLOG 2026-07-16 "P3 T4".]

## Risks

- P0 touches what graph capture records: that is why it ships alone behind
  byte-identity gates before any scheduler code exists.
- Trim/bookkeeping interaction: depthctl and suffix EMAs must see granted
  widths, or adaptive depth drifts under contention.
- Sampled-path parity: spec_sample_round mirrors spec_round; the fused
  forward must serve both or batching is dead in production.
- Slot ctx asymmetry (131K slot0 + 32K slots1+): acceptable for the fan-out
  target; subagent conversations measured 11-18K (R0).
- Eager host tax: bounded and measured at P1; priced escape hatch at P3.
- Env: Q27_BATCH=1 gates the whole feature (default off until the P1 gates
  pass; flips default only after live CC validation).
  [RESOLVED 2026-07-16: exactly that sequence ran -- gates passed, live CC
  validated (4 clean validations, zero errors), defaults flipped ON in the
  serving profile (Q27_BATCH=1 Q27_BATCH_GRAPH=1 CAP=64; Q27_PROFILE=ref
  stays the no-batch reference). BUILDLOG 2026-07-16 "DEFAULTS FLIPPED ON".]

## Not in scope

General multi-client serving (prefill+decode co-scheduling, admission,
fairness), paged/virtual KV, mega-engine batch-dim refactor, cross-GPU.
