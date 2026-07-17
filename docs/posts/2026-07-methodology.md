# Quasar: the build log is the product

*(draft -- not published. Quasar is the engine, built under the codename
q27; the names are used interchangeably below. The headline results
live in two companion posts -- the 5090 speed post and the 3090/turbo3
post; this one is about the methodology that produced them: basins,
tie lotteries, kill criteria, and what happened when an outside audit
pointed at my cache code.)*

*(Update 2026-07-16 -- the "What's left" section resolved. The filed
tensor-core GEMM verify shipped 2026-07-13 (k_vgemm, bitwise by
construction, +21% echo / +5-7% agentic). turbo3 did not become the
serving default: it is the capacity lever (2x96K context on 32 GB) while
fp8 stays the speed default. And a lever this post never filed landed
biggest -- cross-user continuous batching (P0-P3), default-on at 1.41x
aggregate for 2 slots with solo serving bitwise-untouched. BUILDLOG
2026-07-14..16; docs/multislot-throughput.md.)*

Day one of this project was July 2nd. A few days in I told a reviewer I
wouldn't write this post until q27 hit at least parity with tuned
llama.cpp on the same model, same GPU, same harness. On July 5th I filed
the measurement protocol: three trials per task, both engines at their
strongest config, same day. On July 10th -- eight days in -- the protocol ran
and came back +47% decode over llama's best configuration, with score
medians converged. The acceptance criteria predate the result by five
days, which is the only reason you should believe any of the numbers
below.

## What q27 is

q27 is a from-scratch CUDA inference engine for exactly one model family:
Qwen3.6-27B-MTP, a hybrid Gated-DeltaNet/attention model with trained-in
MTP draft heads, running on a single RTX 5090. No abstraction layers, no
model zoo, no portability. One model, one GPU, as fast as I can make it.
The repo is in the spirit of antirez/ds4: the engine is the artifact, and
the build log records everything including the failures.

Two design commitments shaped everything else. Greedy decode is
bitwise-reproducible and gated: a canonical prompt's 128-token output has
held byte-identical through every kernel rewrite, width change, and graph
restructure since the gate existed, and anything that would break it
ships as a separately-gated tolerance-class path. And negative results
get the same writeup as wins, with attribution. About a third of this post is
things that didn't work.

## The suffix drafter, and what's actually novel about it

The model's MTP heads give you self-speculation for free: draft a few
tokens with the cheap head, verify them in one batched forward, keep the
prefix that matches. q27 runs that with an adaptive depth ladder (4 to 7
drafts, promoted and demoted per-stream from realized acceptance).

Agentic traffic has a second structure worth exploiting: Claude Code
re-emits tool results, rewrites files, and echoes its own earlier output
constantly. When the committed stream's suffix recurs earlier in the
context, the continuation of the earlier occurrence is a free draft -- no
model call at all. q27's suffix drafter indexes the stream with 4-gram
lookup and fills the verify lanes from the match.

Stream-lookup speculation is an old idea. llama.cpp has ngram speculation
in the same family, and on a degenerate pure-loop payload it beats q27
badly (889 vs 318 t/s) because its drafts are effectively unbounded while
q27 caps at 12 lanes. The part I'll claim is the composition: per-round
arbitration between the trained MTP ladder and the free suffix drafter,
both verified through one shared-KV MMA kernel, with the suffix taking
the echo-heavy stretches so the ladder isn't starved. On live traffic
the cap doesn't bind -- measured acceptance length is 9 to 10.6 tokens
per fired round, under the 12-lane ceiling, on 30-60% of all decoded
tokens.

Getting there meant widening the whole verify architecture from 8 lanes
to 12: pointer-struct kernel signatures, twelve rotating GDN state
buffers, a permutation modulus change, and a captured-graph zoo that grew
1.5x. The widening is byte-identical at old widths by construction --
role addressing is fully indirected, so a larger modulus only relabels
which physical buffer holds a role. The plan predicted uncapped
acceptance around 10.5 based on offline simulation; live traffic
delivered 10.61 on the first trial. Same fire rate as the old width,
+63% tokens per fire. That's a cap release, and it behaved exactly like
one.

## A day of kernel tuning, including the failures

The verify-attention kernel (fdmma) scores all lanes against a shared KV
stream with fp8 tensor-core MMA. At the start of the day it ran 61K
context, width 12, in 354.7us. By the end: 202.6us, 5.6x over the
register-accumulator baseline it replaced. Two changes did all of it,
and three careful attempts did nothing.

ncu said the kernel idled 89% of cycles with no eligible warps: one
192-thread CTA per SM, three full-CTA barriers per tile, nothing to hide
latency with. The fix that worked was boring: single-buffer the K/V
staging, shrink shared memory to 49KB, get the register count to 168,
and let two CTAs co-reside per SM so they interleave each other's stalls.
+17-26% depending on width, bitwise-identical because the arithmetic is
shared code.

The second win was wave quantization. The kernel launched 512 CTAs (128
splits x 4 KV heads) against 340 resident slots -- a full wave plus a
half-empty one. Setting splits to SMs*2/kv_heads = 85 fills exactly one
wave: another -29% at the width that matters. The split count is now
computed from the device at startup.

The three failures are more instructive than the wins:

- Reordering the prefetch to overlap the V-transpose: measured wash. The
  copies were already hidden. Reverted.
- Warp-pair PV (halve the output accumulator, split PV dims across
  paired warps, sync with named barriers): bitwise-correct, lost 6-15%
  against the 2-CTA kernel.
- Full warp-specialized producer/consumer (dedicated staging warp,
  arrive/wait barrier ring, zero CTA-wide barriers in the loop):
  bitwise-correct, lost by 2x.

All three fail for the same reason, which is the useful part: for this
kernel family, CTA count dominates intra-CTA orchestration. Two
independent barrier domains hide stalls better than any clever
choreography inside one domain, and the clever choreography costs the
shared memory that buys the second domain. I measured that conclusion
three different ways in one afternoon and I'm confident enough to put a
do-not-retry bar in the log: don't revisit warp specialization here
unless the staging fits under 49KB and 85 registers at the same time.

## The regression my own benchmark caught

The best story of the day is an embarrassing one. After the widening
shipped -- gates green, canonical byte-identical, sanitizer clean, live
trial faster -- I ran the short-context benchmark suite on the vanilla
model and got 149.5 t/s against a 161.8 reference. Identical output
trajectories. Pure per-round cost. The depth-focused gates never noticed
because they measure at 26-61K context where attention dominates.

nsys narrowed it to exactly the kernels that take lane-pointer structs
as by-value parameters and index them with blockIdx. ptxas showed the
stack frame doubled from 128 to 256 bytes. When you dynamically index a
by-value kernel parameter, the compiler copies the whole struct to
per-thread local memory. The widening doubled the struct, which doubled
a copy that had been silently taxing every multi-lane kernel since the
structs were introduced weeks earlier.

The fix is one attribute: `__grid_constant__` pins the parameter in
constant memory and makes dynamic indexing legal without the copy.
Addressing-only, bitwise. The suite went to 172.2 -- above the baseline
that predated the regression, because the fix removed the original 128
bytes too. Two lessons went into the log as standing rules: the
short-context suite is the launch-overhead canary and runs after any
plumbing change, and an instrument that demonstrably catches what the
other gates miss retroactively strengthens every number it ever
produced.

## Ties, basins, and how to not fool yourself

Greedy decode over a 27B has argmax ties, and any change to floating
point accumulation order re-rolls them: fp8 attention, split-count
changes, sometimes just a rebuild. The output is deterministic for a
given binary and neutral in expectation across binaries, and one
benchmark in my suite has a hidden test gate that lands on a knife-edge
tie -- the same task scores 0.85 or 0.55 depending on which side of one
early tie the trajectory falls, on every engine I've run through it,
llama included.

That forces some measurement discipline. Bitwise gates only compare legs
from the same binary. Tolerance-class changes get judged by a basin
matrix across several tasks plus a re-roll on the next binary, never by
a single trial on the bimodal task. The fp8 attention kernel looked like
it was systematically tanking that benchmark until the matrix showed
three other tasks at parity-or-better and the next binary re-rolled the
bad basin good with the identical kernel. The flip was a per-binary
lottery; the kernel steered nothing. If you benchmark greedy LLM
inference and haven't hit this,
you will.

## The result

Protocol as filed July 5th: three trials each of three agentic coding
tasks (a collab server, a task queue, an analytics dashboard), scored by
deterministic hidden tests, Claude Code as the harness driving each
engine's native API, no-think greedy both sides, same day, same 5090,
vanilla model both sides. q27 at its shipped defaults; llama.cpp at its
best measured config (Q5_K_M, MTP draft speculation depth 10, p-min 0.5,
flash attention).

Score medians: 0.83 vs 0.83, 0.78 vs 0.79, and the bimodal task drew 2/3
good basins for q27 against 1/3 for llama. Across the nine draws per
engine, q27 landed in-band 8/9 and llama 5/9, including one hard zero --
but n=9 cannot separate those proportions (Fisher's exact p is about
0.29). It stays an observation until a bigger n separates it. Quality is the model's,
and quality parity here is a system-level claim: both harness paths
depend on tolerant tool-call parsing, and strict parsing scores zero on
some of these tasks for any engine.

Decode, within-leg, 430 requests of telemetry: q27 231.3 t/s aggregate
(median 225, peak 378) vs llama 157.4 (median 155, peak 274). That's
+47%. The n=1 pilot the day before read +40%; the number strengthened
under replication, which is what happens when the pilot wasn't a lucky
draw.

Decompose it before someone else does: q27 reads about 15.8GB of weights
per step at 5.25 bits/weight; Q5_K_M reads about 18.2GB. On a
bandwidth-bound decode that's roughly 15 points of the 47 from bit-width
alone. A 30-trial quality A/B against Q5_K_M scored dead even months of
tuning ago, so the quant is paid for; the remaining ~22% is mechanism --
suffix drafter, depth ladder, the fdmma kernel, and prefix-cache
behavior under a real multi-turn client. End-to-end task wall favored
q27 3-4x on the clean tasks, but wall is trajectory-confounded (llama
generated 2.3x the tokens on its own trajectories), so the decode
telemetry is the claim and the wall is context.

For the record, the trajectory of this comparison: tuned llama beat q27
by 31% at depth on July 6th. Parity on July 7th. +47% on the 10th. I
was building until the same benchmark flipped, and the benchmark is the
one llama was winning.

## Zero-config, because the config was the last bug

Until yesterday the fast configuration was six environment variables of
tribal knowledge. Now `q27-server model.q27 model.tok` resolves the full
measured stack -- fp8 KV, MMA attention, the confidence gate, the depth
ladder, the suffix drafter at width 12 -- arch-gated so older GPUs fall
back cleanly, with the context size auto-fitted to free VRAM (capped at the 262K native window for the compact KV formats since day nine). Every knob
keeps its override, `Q27_PROFILE=ref` restores the conservative
reference behavior, and the CLI binary keeps reference defaults so the
bitwise gates still mean something. The bare command on the vanilla
model hits 400 t/s on a repetitive-payload ceiling test -- quote that as
a bound. The headline is 231 t/s on real agentic traffic, from a command
line with two arguments.

## The receipts

The engine was built with Claude Code driving the sessions, and the token
bill for the whole eight days is $3,398 -- 2.64 billion tokens, of which
16M were input, 8.9M were output, and 2.53 billion (96%) were cache
reads. Read that last number again, because it's the thesis of the engine
printed as an invoice: agentic traffic is overwhelmingly the same context
re-read every turn with a small live suffix. That workload shape is why
q27's prefix cache is built around the recurrent state, why the suffix
drafter exists at all, and why a one-line billing-header fix on day four
(Claude Code's mutating prompt head was forcing a full re-prefill every
turn) was one of the highest-leverage changes of the week.

There's also a recursion in the usage report I want on the record: from
day three onward, `q27-qwopus-27b` appears in the models column of its
own build receipt. Once the server spoke the Anthropic API well enough, I
pointed Claude Code at it and some of the tokens that built q27 were
decoded by q27, at zero dollars per token. And for scale: the week's
entire 8.9M output tokens amount to about 10.7 hours of decode at
tonight's measured 231 t/s. The engine can now retype its own source
history in an afternoon.

## Day nine: the 3-bit KV cache

Before q27 existed I had a llama.cpp fork with an experimental KV quant
called turbo3: QuaRot-style, L2-normalize each 128-dim group, rotate it
through a baked Walsh-Hadamard transform, then 3-bit quantize against 8
Lloyd-Max centroids with a corrected per-block norm. 50 bytes per 128
dims. On day nine I ported it into q27: 13.4 KB per token against fp8's
34 and fp16's 68.

The port went microtest-first. The block format, the rotation tables,
and the quantizer were validated bit-for-bit against the fork's CPU
reference before any engine wiring existed, and every engine kernel that
followed was tested against the same oracle -- write a failing test,
watch it fail, then build the kernel. The failing-test step earned its
keep immediately: the stub run exposed that my own comparison harness
swallowed NaNs through std::max, which means it would have passed a
garbage kernel. The fp8 and fp16 paths held byte-identical through all
three phases of the port; the canonical gate ran before and after every
one.

The correctness contract is the pretty part. Dequantized K equals
rotated K, so the attention dot needs Q rotated once after rope, and the
rotation is orthonormal so the scores are unchanged. V accumulates in
the rotated basis and one inverse transform on the pooled output
un-rotates everything at once. Three small kernels, no change to the
softmax anywhere.

## The finding: 3-bit K survives where its own author said it wouldn't

My fork refuses to run turbo3 on K when the GQA ratio is 6 or higher --
it silently upgrades K to q8_0, because a 7:1 model cratered to
perplexity 2887 in early testing and the guard got written wide. This
model sits at exactly 6.0. That guard was the single biggest open
question in the port plan, so instead of trusting it I measured it:
teacher-forced NLL over the same wikitext chunks, four KV configs, same
binary.

fp16 7.317, fp8 7.327, turbo3-V-only 7.368, turbo3 K+V 7.381. Quantizing
V to 3 bits costs +0.70%. Adding 3-bit K on top costs another +0.17%.
The crater does not exist here -- probably because at head_dim 256 the
rotation runs over two independent 128-groups per head, each with its
own norm. The guard is over-conservative for this model, and there is a
number below for exactly what that conservatism costs.

The rest of the quality file, so nobody has to take the PPL row on
faith: position-bucketed NLL over a single 297K-token pass is flat and
tracks fp8 within +0.65-1.2% in every bucket, so the delta does not
compound with depth. Needle retrieval is 6/6 exact at a 361,513-token
prompt, two needles past the native window -- the deepest retrieval this
engine has produced under any KV format. And on basin-matched replay of
a real Claude Code transcript, speculative acceptance ties fp8 to the
third decimal: 5.818 tokens per round on both legs. The 3-bit cache
costs the drafter nothing.

Wall clock needed one more kernel. The verify-attention MMA path reads
raw fp8 tiles, so turbo3 initially fell back to the slower register
kernel and paid 26% at 27K context. Teaching the tile loader to expand
blocks to e4m3 in shared memory -- the MMA math untouched -- closed
that to -4.4% at 27K, and at 61K turbo3 comes out 9.6% ahead of fp8
because it reads 2.56x fewer KV bytes into a bandwidth-bound kernel.

Ceilings, all measured the same evening on the 5090: fp16 allocates to
roughly 180K, fp8 to 294,912 (the estimate in the log was 285K; the
ladder says 295), turbo3 to 655,360 -- two and a half times the native
window, bounded by VRAM and nothing else. The serving cap that used to
sit at 131K was a habit from the fp8 era; it now sits at the 262,144
native window for the compact formats, and the eval server boots there
zero-config.

## What a 24GB card is for

The fun demonstrations came after the gates. A 3090 running this model
could historically serve 32K of context -- the first trial run proved
the point by dying on the context wall at turn ten, three times out of
three, while the server calmly returned well-formed context-limit
errors. That cap was never compute; it was KV bytes. With turbo3 the
same card serves Claude Code at 131K context and 70 tokens per second
median, and completed three full agentic benchmark sessions without a
single protocol failure. The fork the quant came from tops out at 98K
on the same card, partly because of its heavier weight file and partly
because of that K guard -- refusing 3-bit K at ratio 6.0 costs it
exactly the 98K-to-131K gap. On Ampere its kernels decode 15% faster
than mine; the engine with the measured K answer holds the context
ceiling. I will take that trade, and both numbers are in the log.

On the 5090, the same bytes buy tenancy instead: two slots at a full
131K each, validated with pairs of concurrent Claude Code sessions.
Light agentic tasks interleave through each other's tool-execution gaps
and land solo-class scores at solo-class wall times, two-up. Heavy
sustained decode splits the GPU fairly at about 1.8x per-session wall.
There is no vLLM-style aggregate speedup hiding in there, and the log
now has a short doc on why: this engine already spends its weight-read
amortization on speculative width within one user -- 5.8 committed
tokens per weight stream is the batch. Multi-slot buys capacity and
zero-queue admission, and with fp8 the second tenant got 23K of
context. With turbo3 both tenants are whole.

## The trial that debugged the parser

The first fp8-vs-turbo3 benchmark run of the evening produced garbage in
the most instructive way available. The fp8 leg scored 0.00 three times
in nine seconds each: the model emitted its very first tool call in a
known malformed shape, the tolerant parser declined to rescue it, Claude
Code saw prose with no tool call, and every session ended at turn one.

The decline was the bug, and it was a designed decline. The parser
infers a tool name for name-dropped calls by matching argument keys
against the registered tool schemas, and refuses on a scoring tie
because guessing wrong is worse than not rescuing. Two days ago that was
sound. Current Claude Code registries carry property twins -- Bash and
Monitor both take command and description -- so the orphaned arguments
tied 4-4 and the refusal fired. What surfaced it was a benign
accumulation-order reroll from a header change that nudged the greedy
trajectory onto the malformed shape (see the ties section above; it
always comes back). The fix is a tie-break: a tied candidate whose
required parameters are absent from the arguments could never have
validated anyway, so eliminate it, and rescue only a unique survivor.
The exact bytes from the failing trial are now a unit test, drift mode
catalog entry ten, and the same benchmark re-run scores normally. The
benchmark caught a serving bug that every future harness version would
have hit. That is the second time this week an instrument paid for
itself; I plan to keep buying instruments.

## Day ten: untaking the trade

The day-nine ledger ended with llama decoding 15% faster on the 3090 and
me claiming the context ceiling made it a fine trade. That lasted one
night. The 3090 has no fp8 tensor cores, so my verify attention ran on
the register-accumulator fallback while the 5090 got the MMA kernel --
an architecture gap, which means a fixable one.

Sizing first, per the standing rule: phase telemetry on a real replay
put the verify wall at 53% of 3090 decode. Worth building. The f16 MMA
verify kernel turned out SIMPLER than the fp8 one it mirrors, for a
reason I did not expect: 16-bit fragments get hardware paths that 8-bit
fragments lack. The fp8 kernel relays its probability fragments through
shared memory byte-by-byte and pre-transposes V, because e4m3 has no
ldmatrix-transpose and no direct fragment identity. The f16 kernel
converts probabilities to PV fragments in registers and loads V
transposed straight from its natural layout. Two shared-memory
structures and their barriers, gone. The prefill kernel had solved all
of this months of subjective time ago; I lifted its idioms line for
line.

One bug in the first cut, and it is a classic: the fp16 tile fill
copied a uint2 where eight halves need a uint4, leaving half of every
tile group uninitialized. The NaN-hardened comparison I added to the
test harness two days earlier -- after catching my own harness
swallowing NaNs through std::max -- flagged it on the first run.
Instruments compound.

Numbers, same replay both days: 90.3/93.1 tokens per second on the
fallback kernel, 119.9/123.0 on the MMA kernel, +32%. Verify wall down
25%. Live benchmark sessions: three for three good basins at 0.82, and
102.2 t/s median against llama's 80.7 on identical traffic -- from 13%
behind to 27% ahead overnight, while keeping the 131K-vs-98K context
lead. The canonical gates held bitwise on all eight legs, and the new
kernel measures 1.2-2.7e-3 against the exact path -- two orders of
magnitude tighter than the fp8 kernel's own noise floor, because
16-bit Q and P round less. Ampere also quietly got something nobody
asked for: fp16-KV serving now has an MMA verify path on every
architecture, because the same kernel takes all three cache formats
through one template parameter.

The postscript is the part I would print on a poster. With attention
fixed, I profiled again: the GEMV weight stream now owns 68% of the
round and runs at 81-90% of the card's DRAM roofline. That single nsys
table killed two planned optimizations in an afternoon -- the async
staging variant of the new kernel (attention is now 0.3% of the round;
there is nothing left to hide it behind) and a wider-lane build (probed
anyway: it costs half the context, decodes 17-20% slower because wide
rounds fall off the MMA kernel, and the width cap wasn't binding in
the first place). Both negatives took one measurement each and both
went in the log with do-not-retry bars. The 3090 is parked at the
memory wall now, where the only remaining lever is reading fewer
weight bytes -- a quantization-policy study with a quality budget, not
a kernel afternoon. 70 to 102 tokens per second in a day, and the last
30% of that day was spent proving there is no cheap fourth act.

## The audit, the alias, and the ledger

Two days after the numbers stabilized, an outside audit of the repo
came back with four findings, ranked. Three were hygiene (an
unauthenticated server binding all interfaces by default, a stale
hardcoded model name, an overclaimed traceability sentence). The
fourth was pointed at the checkpoint ring, and it was the worst CLASS
of bug this engine can have: on a mid-history divergence, the
re-prefill overwrote KV rows with the new conversation while ring
entries and the snapshot covering those rows survived -- so a later
request matching the OLD conversation could restore recurrent state
over foreign attention rows. Silent mixed-conversation state,
corrupting exactly the feature (the prefix cache) that carries the
cross-engine result. No existing gate could see it: the canonical is
CLI-serial, and the serial path's guard never covered the batched
divergence route.

The handling is the methodology in miniature. First a RED receipt:
craft the divergence-then-replay shape, watch the buggy binary restore
4,468 tokens of state over ~1,300 rows another conversation had
overwritten -- measure the corruption before touching the code. Then
the fix, scoped with a principled exception: entries that remain a
prefix of the new prompt stay valid, because deterministic prefill
rewrites their rows with identical values. GREEN: the replay restores
exactly the surviving checkpoint. Canonical unchanged (host-side cache
policy only). Shipped same day as a point release.

Then the part that separates a ledger from a highlight reel: the
exposure audit. "No published number rode the bug" has to be a checked
claim. journald still held every [gen] line since the 10th; grepping
for divergence-restores found 552 traversals of the alias condition --
every single one on July 11th, live agentic sessions where Claude
Code's subagent sidechains interleave two conversation branches on one
engine. Every replay bench, the protocol A/B, and everything measured
on the 12th: zero traversals, structurally clean. The July 11th
session scores are annotated provisional in the docs until rerun. And
the divergence-then-replay shape is now a standing gate
(tools/ckpt_gate.sh), the same way every parser drift mode became a
fixture: bugs earn their regression tests by demonstrating a shape,
and the shape stays in the battery forever.

## What's left

The deep-MTP question is closed by pricing rather than building: ceilings
past 8 lose money even on the friendliest measured traffic, and ceiling
8's theoretical +2.7% evaporates against live cap distributions plus the
suffix drafter already owning the saturating stretches. The wide-lane
cost curve says the marginal is batched-GEMV-bound, so the one filed
lever for the future is a tensor-core GEMM verify -- which happens to be
the shape llama's ngram speculation exploits on that degenerate loop.
If pure-echo traffic ever matters, that's the pivot, and the microbench
for it already exists in the tree. The turbo3 follow-ons are smaller:
the 3-bit cache is one default flip away from being the serving
standard if acceptance parity holds over more traffic, and the parked
weight-bit policy study is now the only lever the 3090 profile left
standing.

Everything above is reproducible from the repo: the build log carries
every number in this post with its gate output, the negative results
have their own entries, and the benchmark harness is public. The engine
is a hobby project for one model on one GPU. It is also, as of this
week, the fastest way I know of to run this model on this GPU by a
margin I specified before I could meet it.
