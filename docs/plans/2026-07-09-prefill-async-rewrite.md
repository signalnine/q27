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

---

## Phase A day-1 gate result (2026-07-09): mechanism GO, Phase-A EV cut -- plan INVERTED

tools/mbar_ring.cu (committed): producer/consumer cuda::pipeline ring at
the kernel's real stage sizes vs the current CTA-barrier phase structure,
identical work, MATH_ITERS sweep across load:math balances:

    load-heavy  85: 1.20x | balanced 170: 1.48x GO | 340: 1.23x |
    680: 1.11x | math-heavy 1360: 1.04x

Mechanism verdict: cuda::pipeline is SOUND on sm_120 (1.48x at balance --
no CTA-barrier degeneration; the library path is usable).

EV verdict for Phase A proper: the REAL kernel sits near the math-heavy
end of this curve -- 95.6% L2-hit and the Phase-1 cp.async ping-pong
already overlap the global loads, so the async rewrite's remaining prey
is only the V-convert phase + wait granularity: projected +5-10% on the
attn kernel ~= +3-6% prefill, BELOW the +8% Phase-A GO bar. The 30%
long_scoreboard stall lives INSIDE the math phase (smem/MMA operand
latency), which producer/consumer specialization does not touch.

**Plan inversion:** Phase B (fp8 PV: delete s_v + the V-convert phase +
one sync per tile, tolerance-gated, ~1 session) is now FIRST -- it
removes the same serialized phase the async structure would have hidden,
at a fraction of the complexity, and its attribution decides whether any
mbarrier work remains worth doing. The FlashRT gap beyond that is
math-phase engineering (MMA operand pipelining, instruction mix), not
async structure -- a different, register-level project if B's numbers
say it's still worth chasing.

---

## fp8-PV cut (promoted-first Phase B): GATED GO -- run it next session

Two macro-gated probes added to k_attn_prefill_mma_fp8q (default-inert;
canonical a2982c51 EXACT with them present):

**Tolerance (-DQ27_PV8_PROBE): round-trip P through e4m3, keep f16 MMA.**
V is already e4m3 in the KV cache, so this isolates the only NEW loss a
true fp8-PV MMA adds -- quantizing softmax P. @131K vs the default fp8q:
cosine 0.99996269 (bar >= 0.9999 PASS), argmax MATCH, top5 4/5. Marginally
below the shipped fp8q gate (0.9999827, top5 5/5) but inside the Phase-B
bar. Needle 6/6 still required on the real kernel before default-on.

**Convert-phase cost (-DQ27_PV_CONVX=N): repeat the V-convert N times.**
@131K: N=1 59.0s | N=4 71.0s | N=8 86.0s = ~3.9s PER CONVERT = 6.6% of
prefill. THIS REVISES THE DAY-1 EV-CUT: the microbench modeled loads+math
but NOT the convert phase, which sits between two __syncthreads on the
critical path (cp.async hides the GLOBAL loads, not this smem->smem ALM
phase). Deleting it -> ~55s @128K = +7% prefill, at/above the GO bar.

**Verdict: fp8-PV MMA is GO.** Tolerance passes; payoff measured ~6.6-7%.
Implementation (next session, ~1-2 sessions): true m16n8k32.e4m3 PV where
V is the B operand read STRIDED from s_vraw (no s_v, no convert phase) --
the mirror of the QK^T path but with the key/dim axes swapped, so the
B-fragment indexing gathers 4-consecutive-keys-at-fixed-dim (strided by
HD in s_vraw) instead of QK^T's contiguous 4-dims. Risk: strided smem
B-reads bank-conflict; net win = (6.6% convert saved) - (conflict cost),
measured on the real kernel. Frees s_v's 16.5KB -> deeper cp.async ring
as a bonus. Gates: fp8q tolerance battery (cosine/argmax/top5 @131K +
needle 6/6 + --pf continuation) + TTFT ladder 8/32/128K + canonical
(fp16 path untouched). Rigs kept: Q27_PV8_PROBE, Q27_PV_CONVX.

---

## fp8-PV IMPLEMENTED (Q27_PF_PV8) -- CORRECT + tolerance-safe, but net ~0%: NEGATIVE

Built the true m16n8k32.e4m3 PV kernel k_attn_prefill_mma_pv8 (opt-in;
canonical a2982c51 EXACT by default). Layout de-risked first via a
standalone microtest (tools/pv8_mma_test.cu, max abs err 0.00000 vs CPU
ref): P relaid from QK^T accumulator -> A-frag via per-warp s_P +
__syncwarp; V consumed raw as the strided e4m3 B operand; V double-buffered
in s_vraw; s_v + the convert phase deleted.

**Correctness:** logits @131K cosine 0.99996543 / argmax MATCH / top5 4/5 --
matches the isolated Q27_PV8_PROBE (0.99996269) to 5 digits, confirming the
layout. Tolerance passes the Phase-B bar.

**Perf:** 128K 59.66 -> 59.30s (+0.6%), 32K 10.68 -> 10.64s (+0.4%). The
measured 6.6% convert-phase saving is ~entirely offset by the strided-V
B-operand gather (256 scattered byte-reads/thread/tile, HD-strided ->
bank-conflicted, INSIDE the MMA loop). LESSON: getting V from [key][dim]
into the PV B-operand layout is an IRREDUCIBLE data-movement cost; fp8-PV
relocates the layout transform (convert -> gather) rather than removing it.

**The one variant that could still win: transpose-V.** A phase that writes
s_vt[dim][key] fp8 (8KB, HALF the convert's 16KB half-write) makes the B
reads clean contiguous uint32 (no in-loop scatter). Predicted ceiling
~+3% (half the convert cost + scatter moved out of the MMA loop). That is
the next attempt if prefill TTFT becomes the priority; kernel + microtest
kept as its scaffold (proven-correct P-relayout + fp8 PV MMA). Q27_PF_PV8
stays opt-in, documented negative. Do-not-ship as default (no win).
