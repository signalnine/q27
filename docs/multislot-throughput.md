# Why q27 multi-slot doesn't scale aggregate throughput like vLLM

2026-07-11, written against the day's 2-slot turbo3 measurements. Companion
to docs/P10-decision.md (which priced the alternative and rejected it).

## The observation

Two turbo3 slots at 131K each on the 5090 (30.6/32.6 GB), real CC sessions:

- Solo T8: 132-148s wall, per-request decode median ~205 t/s.
- Concurrent T8 pair: 253/256s (~1.8x per session), per-request median
  103.5 t/s, p90 200.8. Engine busy-aggregate stays ~200 t/s -- the same
  total rate, split between tenants.
- Concurrent T2 pair: 152/162s -- NEAR-SOLO walls two-up. T11 pair: 69/70s.

vLLM at c=2 on the same class of hardware gains ~1.6x aggregate (P10 brief:
160.7 t/s c=2 vs 102 single). q27 gains ~1.0x. That is not a bug; it is the
engine's central design bet, and the p90=200.8 shows there is no scheduler
tax -- a slot that briefly holds the GPU alone runs at full solo rate.

## Why: the weight stream is spent on speculative width, not users

Decode on this box is bandwidth-bound on the 17.7 GB weight stream. Every
decode round streams (nearly) the full weight set once. The only way any
engine beats that per token is amortization: feed MORE tokens through each
weight read.

- vLLM amortizes ACROSS USERS: continuous batching concatenates live
  sequences, one weight read feeds N users' tokens. Aggregate scales until
  compute or KV bandwidth binds.
- q27 amortizes WITHIN ONE USER: the MTP ladder + width-12 verify feed up
  to ~13 lanes of ONE sequence through each weight read (tok/rnd 5.8 on
  cctx2 means each weight stream yields ~5.8 committed tokens). That
  amortization is what makes 200+ t/s single-stream possible at all --
  it IS "batching", pointed at latency instead of throughput.

R1b multi-slot is therefore round-granularity TIME-SLICING: at any instant
one slot's kernels own the GPU; the other waits for a round boundary
(handovers are measured in Engine::GenStats gw_ms). Two sustained decoders
each stream the weights for their own rounds -- 2x the weight traffic per
token-pair vs a batched engine -- hence the clean ~2x per-request split on
decode-heavy traffic. No cross-slot kernel launches share anything.

Why CC traffic often doesn't feel it: agentic sessions are BURSTY. T2/T11
sessions spend most wall time in host-side tool execution; the slots'
decode bursts rarely collide, so both ride near-solo (152/162s two-up).
Only sustained simultaneous decode (T8-class synthesis turns) pays the
full interleave split.

## Why we didn't build cross-user batching (and still haven't)

P10 Option A ("fused batch-10", 2026-07-03) priced it: extend the n-lane
GEMVs 5->10(now 12->24) lanes, per-lane SEQUENCE plumbing in verify
attention (lanes assume one sequence today -- separate KV base pointers,
positions, GDN states per lane group), and -- the killer -- the CUDA-graph
zoo cannot bake per-perm pointers across two independent mod-12 rotations
(a 12^2 permutation product), forcing device-side indirection through the
round-critical path that every canonical bitwise gate protects. Estimated
~1.9x aggregate, hard-capped at 2 users by per-slot GDN role VRAM (each
slot carries its own 12-set rotation; turbo3 shrinks KV, NOT that), for a
~0-15% wall edge over just routing c>=2 elsewhere. Rejected then; nothing
today changes the cost side. turbo3 does not lift the 2-user cap either:
17.7 + 3x(~4.8 fixed + 1.78 KV) > 32.6 GB.

## What multi-slot IS for (what turbo3 changed)

Capacity and admission, not aggregate: two FULL-context tenants (131K+131K
vs fp8's 131K+~23K), zero queue latency for the second user, fair splits
under contention, near-zero cost on bursty real traffic. The six concurrent
sessions of 2026-07-11 (T8/T2/T11 pairs) completed 6/6 with T2/T11 at
solo-class scores AND walls.

If aggregate throughput ever becomes the goal on this box: that's vLLM's
game (with its own measured costs here -- 0% prefix-cache reuse on
hybrid-GDN, 4.7x wall on CC agentic traffic, thunderdome 2026-07-09), or
it's reopening P10-A with the graph-indirection price paid. The latency
engine keeps its bet.
