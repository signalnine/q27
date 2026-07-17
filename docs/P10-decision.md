# P10 decision brief: fused batch-10 serving vs latency-engine positioning

> **2026-07-16 OUTCOME: Option A is what shipped -- and the price this brief
> named is the price paid.** The recommendation below was left unpicked;
> the program took C's stepping stone (R1/R1b, "A1") and the 2026-07-11
> multislot doc re-affirmed the Option-A rejection. Then the
> continuous-batching campaign (P0-P3, BUILDLOG 2026-07-14..16) built
> exactly A's core: conductor-fused cross-user verify (one weight sweep
> serves the union of everyone's lanes), with all three costs paid --
> (a) the graph indirection through the round-critical path (conv/delta
> table twins + a shape-keyed LRU graph cache replacing the per-perm
> baked-pointer zoo, spill-0, bitwise-gated), (b) union-width lane
> plumbing, (c) per-lane sequence/KV plumbing. Measured at 2 slots:
> **1.41x aggregate both KVs, solo cost 0.00%, byte-identity held at every
> phase; default-ON since v0.2.0.** What the estimate got right: it priced
> the work as expensive (three days, ~15 gated commits) and the ~1.9x was
> the no-graphs k=2 ceiling (eager fusion measured 1.21-1.31x; graph
> replay carried it to 1.41x). The 2-big-context-user VRAM cap (GDN role
> sets) still stands. Current analysis: docs/multislot-throughput.md
> ("The price P10-A named, and P3 paid").

2026-07-03. The question: build fused 2-slot batch-10 multi-slot serving, or keep
q27 a single-user latency engine and route concurrent work to vLLM. All numbers
same box, RTX 5090.

## Context

q27 single-stream: **177.5 t/s** stock short-bench, **218.6 t/s** 2K soak (+3000
OC); Thunderdome time-tracker trials run ~90-107s each, serial only (single-slot
server) -- two trials cost ~190-200s wall. vLLM nightly on Qwen3.6-27B
INT4-AutoRound + MTP k=3: **102 t/s** single-stream (120 greedy), **160.7 t/s**
aggregate at c=2, MTP acceptance 97.6% on agentic traffic -- but MTP LOSES to the
no-MTP baseline at c=4 (197.6 vs 296.6 aggregate), so speculative decode stops
paying for vLLM too somewhere past c=2-3. Measured wall clock for 2 concurrent
time-tracker trials on vLLM: **118s total** (101/104s each, ~3% mutual slowdown,
scores 0.848).

q27 2-slot VRAM budget: weights 17.7 GB + 2x GDN 5-set rotation 6.0 GB + ~1 GB
scratch leaves ~6.8 GB KV = **2 slots x ~50K ctx fp16 / ~100K fp8; 3 slots not
viable** (each slot adds 3 GB of GDN buffers). Either way the prerequisite is
fp8 KV as default -- quality already gated (PPL -0.05%, needle 6/6 to 361K).

## Option A -- fused batch-10

Both slots' verify lanes share each weight read: one weight stream feeds 10
lanes (2 slots x batch-5 verify), the same amortization trick that makes MTP the
whole game at batch 1, applied across users.

What it buys: est **~1.9x aggregate (~400 t/s)**, ~200 t/s per user -- each user
keeps near-solo q27 speed while sharing the GPU. That is ~2.5x vLLM's c=2
aggregate and ~2x its per-user rate; 2 concurrent trials should land near solo
wall clock (~100-115s est vs vLLM's measured 118s).

What it costs: three pieces of engine surgery. (a) Device-side graph indirection
to replace the per-perm baked-pointer graphs -- two independent mod-5 rotations
multiply to a 5^n permutation product, so baking pointers per perm explodes.
(b) n-lane GEMV templates extended 5 -> 10 lanes. (c) Per-lane sequence/KV
plumbing in verify attention (lanes currently assume one sequence).

Risks: the ~1.9x is an estimate, not a measurement -- the 4->5 lane step cost
+14% round tax (P3) and the scaling at 10 lanes is unknown. Graph indirection
rewrites the round-critical path every canonical gate protects. The 2-slot VRAM
cap is permanent: at c=3+ the comparison flips to vLLM no matter what. And
decode is not the whole trial -- the wall-clock edge over just-using-vLLM is
~0-15% est.

## Option B -- latency engine + vLLM for batch

What it buys: c=2 concurrency today at **118s wall, measured, score 0.848, zero
engine changes**. vLLM keeps scaling where q27 physically cannot (296.6 t/s
no-MTP at c=4). q27 stays the single-user latency engine (218.6 soak vs 102) and
roadmap effort goes to items with no vLLM substitute: sampling, adaptive depth,
checkpoint pool.

What it forfeits: per-user speed under concurrency -- vLLM users get ~102 t/s,
half of a fused slot's est ~200. The q27 serving stack does not travel:
P8/P9 prefix+checkpoint caches (1.3s warm turns), Anthropic-native shape, and
constrained tool decoding all stop at the routing boundary. Single-stack
simplicity is gone.

Operational shape: the two engines do not co-reside in 32 GB (17.7 GB q27
weights + vLLM's own allocation), so routing means either a process swap on the
5090 (model-load latency between modes) or vLLM pinned to the 3090 (unmeasured,
slower part). The vLLM leg also serves Qwen3.6-27B INT4-AutoRound, not the
Qwopus checkpoint -- concurrent requests get different weights unless Qwopus
gets its own AutoRound quant (unverified).

## Option C -- interleaved-only (the cheap middle)

Each spec round serves one slot: 5 graphs x 2 slots baked to per-slot buffers,
no kernel changes -- two engine states sharing read-only weights. Rejected as a
destination: two busy slots split one weight stream, so each decodes at ~half
speed and the pair would NOT beat vLLM's 118s. It only wins when slots alternate
compute and tool-wait (agentic overlap) -- real, but narrower than either A or
B. Legitimate as a stepping stone: per-slot state + scheduler are prerequisites
for A anyway, and they ship a usable overlap win on their own.

## Comparison

| | A: fused batch-10 | B: q27 solo + vLLM | C: interleaved |
|---|---|---|---|
| per-user t/s @ c=2 | ~200 est | 102 measured | ~110 est (both busy) |
| aggregate @ c=2 | ~400 est | 160.7 measured | ~220 est |
| 2-trial wall clock | ~100-115s est | 118s measured | >118s |
| c=3+ path | none (VRAM) | yes (296.6 @ c=4) | none |
| engine work | graph indirection + 10-lane GEMV + per-lane KV | none | per-slot state + scheduler |
| risk to gated paths | high (round-critical) | none | low |
| ctx per slot (fp8) | ~100K | vLLM-managed | ~100K |

## Recommendation

Not picked here. The strongest case for each side:

**For A:** it is the only option where concurrency costs no per-user speed.
Fused slots hold est ~200 t/s each vs vLLM's 102, and every measured q27 win --
218.6 soak, 1.3s warm turns, 361K validated ctx, constrained tools -- extends to
2 users instead of stopping at 1. The batch-1 arithmetic-intensity argument cuts
the same way here: the idle ops-per-byte MTP converts into latency can absorb a
second user's verify lanes nearly free, off bytes already being read. And A
stages cleanly: C's per-slot state first (useful alone for tool-wait overlap),
fusion second, with a measured go/no-go between.

**For B:** the wall-clock problem is already solved -- 118s measured, today,
score 0.848, zero lines of engine code -- and A improves on that by an estimated
0-15%, purchased with the riskiest surgery on the roadmap through the paths all
canonical gates protect, for a benefit the VRAM budget hard-caps at 2 users
exactly where vLLM accelerates. q27's differentiated value is single-user
latency and long context; every hour on multi-slot plumbing is an hour not spent
on sampling, adaptive depth, or the checkpoint pool -- items nothing else on the
box can deliver.
