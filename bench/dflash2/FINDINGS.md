# DFlash2 into q27: a drafter that wins single-turn and loses live

**TL;DR:** We integrated the z-lab DFlash2 block drafter into q27 as an
alternative to the MTP ladder. It is byte-identical to plain greedy, its
acceptance beats the ladder token-for-token, and on single-turn CLI
benchmarks it wins decode t/s on every traffic type (echo +43%, code-edit
+20%, code-write +6%, prose +4%). Then we pointed real Claude Code at it and
it lost by 7% (164.6 vs 177.5 t/s). The gap is entirely cold-start: agentic
traffic is many short turns, the drafter's context ring resets each turn, and
a cold drafter proposes badly -- while the MTP head always drafts from the
target's hidden state, which carries context through the KV cache. The
single-turn benchmarks warmed the ring once and never paid that cost again.
DFlash2 is a real win with warm context and not a serving default until the
prefill feeds the ring. This is the whole reason to run a live trial.

Full working log: `docs/plans/2026-09-06-dflash2-integration.md`. Rig and raw
data: this directory.

## Why we looked

The trigger was a re-bench of ninfer (`bench/crossengine/NINFER-REBENCH.md`):
their DFlash2 integration ran Qwen3.8-27B at +40-50% over their own MTP3
control on our instrument -- the first working DFlash-family win on this box,
and it beat our own 3.8 sweep rows. DFlash2 is a 1.92B block drafter that
predicts K tokens in one forward from five residual-stream taps of the
target, with no embedding or head of its own (it reuses the target's). The
question was direct: does that drafter beat OUR MTP ladder, on OUR quant, in
OUR engine.

## How it works in q27

The integration reuses almost everything q27 already had for speculation.

- **One fused verify, unchanged.** DFlash2 at K=7 drafts 7 tokens into an
  8-wide block. q27 already captures per-width verify graphs 2..8 and already
  folds partial accepts into GDN state every round (the record-then-fold
  path). The drafter's proposals enter the exact same verify the MTP ladder
  feeds; nothing about the target forward changed.
- **Byte-identical by construction.** The verify decides every emitted token
  from the target's own argmax, so the drafter can only change *how many*
  tokens a round commits, never *which*. Greedy output stays bitwise equal to
  the plain path regardless of drafter quality. Every phase gate checked this
  and it held.
- **The drafter runtime** (`src/dflash2.cu`) is a small five-layer transformer
  with two kernels q27 did not already have -- a grouped dynamic causal
  convolution and a bidirectional sliding-window attention -- plus the
  selector (top-16 per position, then a codebook path-walk) that turns raw
  logits into the K proposals. Feature taps are captured out of the verify
  forward into a device buffer and fed back as the drafter's context.

Three optimizations moved it from correct-but-slow to fast:

1. **Q4 drafter.** The first cut ran the drafter's matmuls in fp16 through a
   gemv that re-reads the weight once per verify column -- so it read its
   3.85 GB of weights eight times per round. Repacking to Q4-g64 and routing
   through the engine's weight-shared `gemv_q4_n` (the same kernel the verify
   uses) cut that to one shared read. Numerics gate: Q4 vs fp16 acceptance
   within 3.6%, output still byte-identical.
2. **Engine head reuse.** The drafter's logits go through the engine's own
   quantized output head instead of a packed fp16 copy. The Q8-vs-fp16 head
   did not shift acceptance at all, and it saved the single most expensive
   kernel in the round.
3. **Verify graph.** The whole verify (forward with tap capture, plus the
   accept/finish tail) is a fixed launch sequence over fixed buffers, so it
   captures once and replays -- the same trick the ladder uses, now with the
   tap copies inside the capture.

## Single-turn: it wins everywhere

K=7, fp8 KV, byte-identical to plain greedy, on the CLI against the MTP
ladder (same binary, same prompts):

| traffic    | ladder t/s | dflash2 t/s | delta |
|------------|-----------:|------------:|------:|
| code-write |        154 |         164 |   +6% |
| prose      |        124 |         130 |   +4% |
| code-edit  |        189 |         226 |  +20% |
| echo       |        252 |         361 |  +43% |

The drafter wins tok/round on all four (+18% to +95%), and once the verify is
graphed the extra tokens turn into a t/s win, largest exactly where the
tok/round lead is largest. Two things we learned pinning this down:

- **The verify is free in width.** A width-2 through width-12 verify forward
  measures 20.1 to 21.3 ms -- the block is weight-bandwidth-bound, so the
  wider verify costs almost nothing. This is the premise of batched
  speculation, and it means the only real cost DFlash2 adds is the drafter
  forward itself.
- **K stops at 7.** Identity holds for K<=7 (width <= 8) and breaks at K>=8
  (width >= 9), at a fixed token regardless of K. Width 8 is the ladder's
  structural maximum (D_MAX_MTP=7); widths 9-12 are reached only by suffix
  rounds through captured graphs, and the eager verify/fold path above width
  8 has a latent, pre-existing engine bug that has nothing to do with the
  drafter. K=7 is the default anyway.

## Live Claude Code: it loses

We wired the drafter into the server (`Q27_DFLASH2=<pack>`, single-slot) and
ran real Claude Code sessions against it -- SWE-bench flask and requests
instances through the agentic harness. It worked and it was correct: the
sessions ran to completion, edited the right files, and hit prefix-cache warm
turns. Then we ran the same instances against the ladder, changing only the
drafter:

| | agg decode t/s | tok/round | reqs |
|---|--:|--:|--:|
| dflash2 (cold-start) | 164.6 | 3.17 | 27 |
| ladder + suffix      | 177.5 | 3.43 | 28 |
| delta                | **-7.2%** | | |

The single-turn wins did not transfer. The cause is cold-start. Agentic CC is
many short turns -- think a bit, call a tool, read the result, repeat. The
drafter's context ring resets at the start of each turn: warm turns restore
the target's recurrent state from the prefix cache, but nothing recomputes the
drafter's taps for the restored prefix. So for the first stretch of every
turn the drafter attends to nothing and proposes badly. The MTP head has no
such problem -- it drafts from the target's hidden state, which always carries
context through the KV cache. The CLI benchmarks each had one prefill and one
long decode, so the ring warmed once and stayed warm; that is not what real
traffic looks like.

The ladder's suffix drafter accepted zero tokens on these instances, so the
incumbent we lost to was pure MTP, not MTP plus its echo trick.

## What it would take

The entire -7% is cold-start, so the fix is to warm the ring:

1. **Prefill tap capture.** Capture the last ~2048 prompt tokens' taps during
   the batched prefill and seed the ring with them, so the drafter starts each
   turn warm. This is the one high-value follow-up.
2. **Composition with the ladder.** Keep the MTP drafts for the cold rounds
   and let DFlash2 ride once the ring warms. More invasive, lower priority.

Until (1) lands, DFlash2 is a warm-context win and not a serving default. That
is the honest result, and it is exactly what a live trial is for -- the
single-turn numbers were real but not representative.

## Serving-integration notes (paid for once)

- The KV pool greedily takes all free VRAM before the drafter loads, so a
  co-resident drafter has to reserve its footprint up front or the load OOMs
  (`Q27_DFLASH2_RESERVE_GB`, default ~2 GB).
- Reuse the engine's own Q8 head and Q8 embedding for the drafter rather than
  packing fp16 copies -- that is 5 GB of VRAM back, which is the difference
  between a 24K and a 131K context window on a 32 GB card.
- Compare a drafter run against a plain run at the SAME KV setting. fp8 vs
  fp16 KV changes greedy tokens, and an early "divergence" here was just a
  stale fp16-KV baseline, not a bug.

## Reproduce

Single-turn (CLI): `q27 --dflash2 <pack.d2w> --k 7 --tokens-file <ids>` and
compare against `q27 --spec`. Rig scripts and the tap-replay gate are in this
directory (see `README.md`).

Serving: `Q27_DFLASH2=<serving-pack> Q27_BATCH=0 q27-server ... --ctx 131072`,
then drive it with `bench/swebench/run.sh`. The serving pack drops the fp16
head and embed (`dflash2_pack.py --no-head --no-embed`).
