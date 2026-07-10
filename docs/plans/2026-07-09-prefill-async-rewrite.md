# Prefill attention: full async rewrite (Phase 3, planned 2026-07-09)

> Successor to docs/plans/2026-07-09-prefill-fa2-relayout.md (Phase 3a,
> KILLED same-day with the decisive diagnostic). Read that verdict first.

**Goal:** restructure `k_attn_prefill_mma_fp8q` from a synchronous per-tile
pipeline into an async producer/consumer kernel, targeting the measured
binding constraint: Eligible Warps Per Scheduler 0.44 / No Eligible 73.9%,
INVARIANT under occupancy doubling (Phase 3a proof). Occupancy is
explicitly NOT the target metric.

**Prize (measured frame):** attn = 54% of 128K prefill; FlashRT's async FA2
averages ~2900 t/s at 256K vs our 2206 at 128K. If eligibility roughly
doubles, expect +20-27% prefill at 128K (TTFT 59.7 -> ~47-49s), +5-6% live
wall, +~20% cold long-context.

## Why the current kernel serializes

Every PP=32 tile iteration runs: wait(cp.async) -> __syncthreads ->
all-threads V-convert -> __syncthreads -> QK^T MMA -> softmax -> PV MMA.
All 6 warps are gated through identical phases by full-CTA barriers; during
load/convert phases zero math is eligible, during math phases zero load
issue. The schedulers idle 74% of cycles by construction.

## Target structure

- **Warp specialization**: 1-2 PRODUCER warps own K/V movement (cp.async
  into a 3-4 stage smem ring + V convert); 6 CONSUMER warps (one per GQA
  head, keeping the proven o[32][4] layout) run QK^T/softmax/PV against
  ready stages. Consumers never execute loads; producers never barrier on
  math.
- **Stage synchronization via mbarrier** (`cuda::pipeline<thread_scope_
  block>` preferred; raw PTX mbarrier if the library path underperforms).
  Fallback if sm_120 friction: two named barriers (`bar.sync` with IDs,
  producer-set / consumer-set) -- Ampere-compatible, coarser but still
  removes full-CTA phase gating.
- **Ring**: 3 stages x (K 32x272 fp8 + V 32x256 fp8) = 49.5KB + s_q 25KB
  ~= 74.5KB < 99KB. Producer runs 2 stages ahead; consumer waits per-stage
  (mbarrier arrive/wait), not per-phase.
- Registers: consumers keep today's 254-class budget; CTA = 8 warps x ~200
  avg = ~51K regs -> 1 CTA/SM, 8 warps. Eligibility, not residency, is the
  thesis.

## Phases (each gated, each killable)

**A. Async skeleton, math-identical (1-2 sessions).** Producer/consumer +
ring + mbarriers; V still converted to half (by producer), PV MMA
unchanged, per-row math order preserved -> aim BITWISE vs fp8q (the 3a
d-split proved exact-transform is achievable; same bar here).
GO gate: ncu Eligible Warps/scheduler >= 0.9 (2x) AND TTFT >= +8% @128K,
bitwise logits @32K. Kill: eligibility rises but TTFT < +5% (would mean
DRAM/L2 or MMA-pipe is the next wall -- re-attribute before more work);
or mbarrier overhead swamps PP=32 granularity (visible as barrier-stall
migration into mbarrier waits).

**B. fp8 PV (tolerance-class, ~1 session).** Drop s_v + the convert phase
entirely; PV consumes fp8 V directly (P quantized e4m3 like Q in the QK^T
precedent). Frees 16.5KB smem -> 4-5 ring stages; removes the producer's
biggest job. Gates: the shipped fp8q tolerance battery (logit cosine >=
0.9999 + argmax MATCH @131K, needle 6/6, --pf continuation) -- same class
that cleared Phase 2's default-on.

**C. Consumer widening (optional, data-driven).** If A/B leave MMA-pipe
headroom and eligibility < ~1.5: apply the BITWISE-validated d-split
(warp pairs per head, 144 regs) to fit 12 consumers + producer(s);
registers now fit because 3a measured them. Only if ncu says issue-bound
remains the wall.

**D. Default-on + ship.** Same protocol as fp8q Phase 2: opt-in env
(Q27_PF_ASYNC) through A-C, flip default after the full battery + needle
+ canonical (fp16 path untouched throughout; canonicals are fp16-serial
and never see this kernel -- the gate is --pf-side).

## Measurement discipline

- ncu sections per phase: SchedulerStats + WarpStateStats (Eligible/
  scheduler, No-Eligible %, stall breakdown migration), MemoryWorkload
  (DRAM/L2 utilization -- watch for the wall MOVING).
- TTFT ladder per phase: 8K / 32K / 128K (+256K once, VRAM permitting)
  fp8 KV, base model, --pf harness, 3 runs median, same-binary env A/B.
- The 3a lesson institutionalized: any variant that raises a resource
  metric (occupancy, stages, warps) must show the ELIGIBILITY needle move
  in the same ncu pass, or it gets killed regardless of how good the
  resource story sounds.

## Risks

- mbarrier/pipeline codegen quality on sm_120 consumer silicon (nvcc 13.x)
  -- prototype the ring in a 200-line standalone microbench BEFORE
  touching the kernel (tools/, day 1 of Phase A; kill the library path
  early if `cuda::pipeline` emits full-CTA barriers under the hood).
- Producer starvation at high nsplit (short KV chunks per CTA at 128K:
  chunk/PP ~= 128 tiles -- ring depth 3 is 2.3% of the stream; fine. At
  nsplit-heavy SHALLOW prefills the per-CTA stream is ~32 tiles; prologue
  overhead grows -- measure the 8K rung, don't assume).
- Numerics: A is bitwise by construction or it doesn't ship; B is the
  only numerics change and rides a proven gate battery.
- Barrier-ID fallback semantics differ across arch (named-barrier count
  16); keep IDs static and documented.

## Non-goals

f16-MMA kernel, split-KV combine, GDN prefill, decode paths, PP/TT
geometry changes (PP=32/TT=16 stay until A's attribution says otherwise).
