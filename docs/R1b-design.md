# R1b: round-granularity interleaved scheduling

2026-07-04. Prereq landed (c874500 + aadd111): WY scratch per-engine.

## Problem

R1 gave each conversation its own engine (slot) but rounds still serialize
behind one `std::mutex gpu` held across the WHOLE `generate()` -- prefill
plus every decode round. A request that arrives while another slot is
mid-generation waits for that generation to finish end-to-end. Measured on
the R1 acceptance rerun (Claude Code session, tinylog task, --slots 2):
118.3s wall, 24.4s of it queue wait. The client parallelizes (main thread +
Explore subagent); the server makes the parallelism worthless during
contention windows.

Time-slicing at round granularity removes the head-of-line blocking: a
decode round is ~27ms (spec_round host-syncs each round), a prefill chunk
is PF_T=1024 tokens ~320ms. Both are natural preemption points where the
engine's device state is fully consistent and its stream can be drained
cheaply.

## Design

**GpuGate (api_common.h, pure host).** FIFO ticket lock replacing the gpu
mutex. `acquire()` takes a ticket, blocks until served; `release()` serves
the next ticket; `contended()` is one relaxed atomic load; `maybe_yield()`
= if contended, release + re-acquire (re-enqueues at the tail -> strict
round-robin among active requests). Solo request: `maybe_yield()` is a
single atomic load per round, no syscalls, no syncs.

**Engine yield hook (engine.cuh).** `std::function<bool()> on_round_gap`,
default null. Called (a) at the bottom of each decode round, after
`on_pending`; (b) after each prefill chunk in both chunk loops. Returns
true when a yield actually happened; engine accumulates `gs.gw_ms` (time
parked) and `gs.yields`. Serial path (<32-token prompts, sub-second) does
not yield. The CLI binary never sets the hook -- canonical path untouched,
zero new code executes.

**Sync discipline.** The decode loop is host-synced at every round bottom
already (spec_round reads the outcome). The prefill loop runs host-ahead
with async chunk launches, so the server's yield lambda must
`cudaStreamSynchronize(eng.stm)` BEFORE releasing the gate -- the GPU is
genuinely idle when the other engine gets it. The sync only happens when
contended; the solo prefill pipeline is unchanged.

**Slot claiming (server.cu).** With whole-request exclusion gone, an
engine can be mid-generation (yielded) when a new request routes. New
per-slot `busy` flag under a small `route_m` mutex: `pick_slot` scans FREE
engines only (reuse_len on a busy engine reads mid-generation state --
meaningless and racy); tiers unchanged (reuse_len > cache_empty > LRU). If
no free engine fits, the handler waits on `route_cv` until one frees
(condvar wakeup, barging accepted -- bounded by <=4 slots and self-limiting
clients; documented, not defended). Prompt fits NO slot even when free:
route to the largest free engine for generate()'s clean refusal, exactly
today's behavior. Flow per request:

    claim engine (route_m: pick free, mark busy)   -- may wait
    gate.acquire()                                  -- FIFO from here
    install on_round_gap; generate(); uninstall
    gate.release()
    free engine (route_m: busy=false, LRU stamp); route_cv.notify_all()

qw keeps its meaning: time from request arrival to generate() start (now =
engine-claim wait + first gate acquire). New telemetry after `end=` (the
reqlog_gate regex stops there): `gw=%.0f yields=%d`. pf_ms/dec_ms remain
wall-inclusive of yield waits; the analyzer subtracts gw_ms.

**Escape hatch.** `Q27_NO_INTERLEAVE=1` skips installing the hook: exact
R1 whole-request serialization behind the gate. Debug lever for the known
rare mid-round host-interaction flake class (P11 split crash + the
one-time canonical md5 flake) -- R1b adds host activity between rounds, so
a kill switch that restores the old timing is cheap insurance.

**Shared host state under the gate.** tool_mask_cache mutations, tokenizer
decode_one, SSE sink writes, and all Engine callback work happen inside
rounds while holding the gate -- same exclusion as the old mutex. A slow
SSE client still holds the GPU during its round (pre-existing; cb_ms~0
measured on real traffic; unchanged).

## Correctness argument

Per-engine isolation is complete since R1 + the WY prereq: KV, GDN
snapshot/spare sets, ckpt ring, mask pool, WY scratch, graphs, stream --
all Engine members; weights are shared read-only. Time-slicing serializes
device work (yield drains the stream before release), so an engine's
computation is bit-identical regardless of what runs between its rounds.
Greedy decode => interleaved output text must be BYTE-IDENTICAL to the
same request run solo. That property is the acceptance gate, not just a
design claim.

## Not doing (scoped out)

- Concurrent kernel execution across slots (space-sharing): splits weight
  bandwidth, both slots slow ~2x during overlap, and the target win --
  overlapping one slot's GPU work with the OTHER's tool-wait idle -- needs
  only time-slicing. Fused batch-10 remains the separately-scoped upgrade.
- Priority/weighted scheduling: FIFO round-robin v1; add weights only if
  [req] telemetry shows the main slot starving.
- Light no-spec utility slots, preemption/eviction mid-generation: later.

## Gates

1. GpuGate unit tests (test_tokenizer.cpp, host-only, no -pthread needed
   on glibc 2.39): FIFO round-robin order under forced contention;
   maybe_yield()==false when solo; N-thread stress completes (no deadlock,
   no lost wakeup).
2. tools/interleave_gate.sh (server-level, like reqlog_gate.sh; GPU free):
   --slots 2, long streaming request A + short request B fired mid-A.
   Asserts: B completes while A still streaming (overlap happened); A and
   B texts byte-identical to their solo runs (determinism); A's [req] line
   shows yields>0, gw>0; B's qw small; Q27_NO_INTERLEAVE=1 restores
   serialization (B waits, yields=0). RED against the R1 build: overlap
   assert fails.
3. Existing: test_kernels ALL PASS, canonical n=128 md5 EXACT (CLI
   untouched), --pf 200 seq+32 IDENTICAL, reqlog_gate.sh PASS (new fields
   appended after end=, parse regex unaffected).
4. Soak: repeated concurrent-pair generations (the flake-pattern watch).

## Acceptance

Rerun the R0/R1 workload (claude -p + Explore subagent, tinylog RFC3339
task, --slots 2, same day): expect queue wait 24.4s -> ~0-5s, wall 118.3s
-> ~95-105s, task success, zero output divergence vs solo. Solo-path
regression check: canonical short bench within noise of 177.3 t/s.
