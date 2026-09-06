# DFlash2 drafter integration -- design v2 (2026-09-06)

Status: Phases 0-6 EXECUTED 2026-09-06. dflash2 wins single-turn CLI t/s
on all four traffic types at K=7, byte-identical to plain greedy (Phase 5:
code-write 164/154, code-edit 226/189, echo 361/252). **But the Phase-6
live-CC trial says NOT a serving win yet: -7% vs the ladder on real agentic
traffic, because the drafter ring cold-starts each turn (no prefill tap
capture).** Verdict: promising, correct, wired -- but gated on prefill tap
capture before it beats the always-warm MTP ladder on serving. K>7 is
separately capped by a pre-existing engine width>8 verify bug. Supersedes
`docs/dflash-block-verify-design.md` (2026-07-09, v1 drafter, parked at
Phase 0). This is a delta document: the v1 doc's motivation, bitwise
contract, and Phase-0 discipline carry forward; the drafter generation, the
verify plan, and two of the hardest problems changed -- both in our favor.

## Why reopen it

Measured 2026-09-06 (bench/crossengine/NINFER-REBENCH.md): ninfer's DFlash2
integration runs Qwen3.8-27B at 221/181 t/s (short/51K ctx) against a
paired MTP3 control at 157/133 on the same artifact, engine, and
instrument -- the whole +40-50% margin is the drafter, at only 37-39% lane
acceptance with K=7. That is the first working DFlash-family win on this
box (mainline vLLM ran it degraded and lost; llama.cpp does not wire it),
and it beats our own 3.8 sweep rows. The v1 doc's question stands with new
evidence behind it: does the drafter beat OUR ladder, on OUR quant, same
harness -- but the prior on GO moved substantially.

## The v2 drafter (z-lab/Qwen3.8-27B-DFlash2, MIT)

Verified against the checkpoint on disk and ninfer's algorithm contract
(their doc cross-checks vLLM and SGLang reference implementations):

- 5 non-causal sliding-window (2048) layers, H=5120, I=17408, 32Q/8KV
  heads, D=128, RoPE theta 1e7 NEOX, our tokenizer (vocab 248320).
- **1.92B params, 3.85 GB bf16, NO embedding or lm_head of its own** --
  it reuses the target's, exactly like the MTP head. Q4-family repack
  lands ~1.0-1.2 GB resident (fc [5120,25600] quantizes; ninfer stores it
  W8G32).
- Target feature taps at layers **[5,19,33,47,61]** (v1: [1,16,31,46,61]):
  s_t = concat of the five residual streams (25600), c_t =
  rmsnorm(W_fc s_t) * w_context. Context K/V for all five draft layers
  project DIRECTLY from c_t -- no input norm, no conv, no Q path on
  context rows. Cheaper context maintenance than v1's window modes.
- Block semantics: anchor = the pending target token; K masks predict
  absolute positions F+1..F+K in ONE forward (no shift). K is a runtime
  choice 1..15, no reconversion; K=7 is the recommended and
  ninfer-measured point.
- Candidate selector (the genuinely new machinery vs v1): top-M=16
  logits per mask position, a rank-256 hidden projection, and
  successor/predecessor codebooks scoring adjacent transitions; one path
  walk picks the K draft tokens. Three small tensors; the walk is
  host-trivial or a tiny kernel.

## What changed since the v1 design -- two scope collapses

1. **No S=16 GEMM verify build.** K=7 means verify width W=K+1=8, and the
   engine ALREADY captures per-width verify graphs 2..8 (M1b set, in
   every serving boot log). DFlash2's drafts enter the same fused verify
   the MTP ladder feeds today; the batched-prefill verify path the v1 doc
   designed is unnecessary for bring-up (it returns only if a K>=11
   sweep says wider blocks pay).
2. **GDN partial accept is (probably) already solved.** The v1 doc's
   Phase-2 monster -- GDN state does not truncate on partial accepts --
   predates M1 record-then-fold (shipped 07-14..16). The ladder does
   prefix-partial accepts at width <=12 every round today, and DFlash2
   acceptance is the same shape: a path accepted up to the first
   mismatch is a prefix accept. Working assumption: the existing fold
   machinery covers it unchanged. **P0c below exists to confirm this in
   code before it is believed.**

What did NOT change: the bitwise contract (greedy output byte-identical to
the plain/MTP paths; canonical gates hold EXACT with the feature on), the
drafter-vs-quant risk (trained on BF16 target hiddens; our 5.25-bpw
residuals + fp8 KV differ -- P0a exists for this), and coexistence
(`Q27_DFLASH2=1` opt-in beside the ladder, never replacing it until beaten
same-harness).

## The composition play nobody else has

Every other integration REPLACES its MTP path with DFlash2. q27's verify
already merges two draft sources (MTP ladder + suffix drafter) into one
width-12 round. DFlash2 at K=7 occupies 8 columns and leaves 4 -- enough
for suffix lanes to ride on top exactly as they do now. On echo-heavy
stretches the suffix drafter is the reason live traffic hits 5.3-5.8
tok/round; DFlash2's wins are on prose/code where suffix goes quiet. The
two are anti-correlated by construction, which is the best case for
stacking. Phase 3 measures DFlash2-alone vs DFlash2+suffix vs
ladder+suffix; the stack is the candidate default.

## Cost model (to be measured, not believed)

- Drafter forward: reads ~1.0-1.2 GB (Q4) once per round at W=8 columns
  -> ~0.6-0.9 ms weight-bound on a 5090, plus small SDPA over the 2048
  window and the selector. Estimate 1-2 ms/round; bf16 bring-up ~2.5-3 ms.
- Tap capture: 5 x 5120 x W values per verify round -- device copies at
  the layer boundaries, microseconds; fc+norm per committed token ~0.26 GB
  read amortized over accepted tokens.
- Round arithmetic at the cctx anchor (re-measure in P0b): ladder+suffix
  runs ~5.3 tok/round over ~26 ms. GO needs
  AL_dflash2 / (round_ms + drafter_ms) > 1.10x today's tok/ms with
  suffix contribution held equal on both sides -- the honest comparison
  is (dflash2+suffix) vs (ladder+suffix), same replay.
- Evidence prior: ninfer's 37-39% lane acceptance at K=7 yields ~3.6
  tok/round from the block alone before any suffix contribution.

## VRAM (131K serving, 5090)

Today ~24 GB. Adds: drafter ~1.0-1.2 GB Q4 (3.85 GB bf16 for Phase-1
numerics-first bring-up at reduced ctx), context ring ~5120 x window x 2B
per resident conversation slice (bounded with the sliding window at 2048:
~21 MB, plus K/V context stores per draft layer ~105 MB), selector
tensors ~10 MB. Q4 path fits with headroom; bf16 path is viable for
bring-up.

## Phases

0. **Measurement, zero engine risk (1 session).**
   - P0a acceptance-on-our-taps: capture q27's five-layer residual taps +
     committed tokens on cctx/CC-replay traffic (stats-path plumbing),
     run the bf16 drafter offline in torch against those taps, measure
     the AL distribution. Validate drafter logits against the vLLM/SGLang
     reference on fixed inputs FIRST (selector + RoPE details silently
     crater AL if wrong). Kill: AL-equivalent < the ladder's yield on the
     same replay, or quant-tap degradation > 25% vs BF16-target taps.
   - P0b cycle cost: drafter forward timed in torch as the upper bound;
     verify width-8 cost is already known from the ladder's own graphs.
   - P0c fold audit: code-read that record-then-fold's partial-accept
     path has no ladder-specific assumption a block drafter violates
     (draft provenance, pending-token bookkeeping, dexit interplay).
1. **Drafter runtime + offline E2E (1-2 sessions).** Repack (or bf16
   load) behind `Q27_DFLASH2=1`; tap capture in the verify path; drafter
   forward + selector; feed the existing width-8 verify; byte-identity vs
   plain greedy on the canonicals; no graphs, no tuning.
2. **Performance (1 session).** Drafter under its own CUDA graph, K sweep
   (5/7/9/11), suffix-stacking wiring, fp8-KV interaction check.
3. **Same-harness verdict (1 session).** cctx replay + live CC trial:
   ladder+suffix vs dflash2+suffix vs dflash2-alone; canonical/width
   gates; BUILDLOG verdict and default decision.

## Risks

- Selector correctness: codebook/transition semantics are the new
  surface; the reference-logit validation in P0a is the guard.
- Quant-tap mismatch (P0a's whole reason to exist; v1 risk, unchanged).
- Tap capture on the CLI/reference path must stay dormant so bitwise
  canonicals are untouched (server-only, like fp8 KV).
- The composition bet could fail: suffix and DFlash2 may fight over the
  same accept opportunities on mixed traffic. Phase 3's three-way A/B is
  designed to catch exactly that.

## Phase 0 results (2026-09-06, same day)

Rig: z-lab's own `dflash` package (clone at `/mnt/ai/projects/dflash`) run
against the checkpoint on disk, target = the BF16 HF checkpoint loaded as
`Qwen3_5ForCausalLM` (needs transformers 5.15; venv `/mnt/ai/venvs/dflash2`
over system torch). CPU-only, so the serving GPUs stayed untouched. Scripts:
`bench/dflash2/`.

**P0a -- drafter works, and the tap ids are config truth.** The z-lab class
maps our drafter checkpoint exactly (1.92B params, taps [5,19,33,47,61] read
from `dflash_config`, not the generic-formula fallback, which would give the
v1 ids). E2E greedy through their `dflash_generate` on four 192-token
prompts, BF16 taps, K=7 -- and the same four prompts through live q27-38
serving (temperature 0, `[req] dec/rounds`) as the same-traffic incumbent:

| traffic    | ladder+suffix tok/round @ round ms | DFlash2 BF16 tok/round | tok/ms ratio* |
|------------|-----------------------------------:|-----------------------:|--------------:|
| code-write |                        2.53 @ 16.2 |                   4.55 |     **1.65x** |
| prose      |                        2.29 @ 15.8 |                   2.94 |     **1.18x** |
| code-edit  |                        3.31 @ 16.4 |                   3.24 |         0.90x |
| echo       |                        6.86 @ 18.4 |        7.35 (cap is 8) |         0.99x |

*DFlash2 charged the incumbent's full round wall plus 1.5 ms of drafter --
conservative twice over: a dflash2 round drops the ladder's MTP passes from
the wall, and the Q4 drafter floor is under 1 ms (P0b).

Two findings that reshape the plan:

1. **The 5.3-5.8 tok/round live figure is an echo-heavy cctx anchor, not
   what the ladder yields on think-prose.** On the traffic that dominates
   CC decode time (thinking + code-write) the incumbent runs 2.3-2.5
   tok/round, and DFlash2 beats it by up to 80% -- consistent with
   ninfer's +40% on think-on decode. The GO gate as originally written
   (beat 5.3/26 ms) compared against the wrong denominator.
2. **The composition thesis was backwards.** DFlash2 is STRONGEST on echo
   (22/26 rounds accepted the full 8-token block) and weakest on prose --
   both drafters feast on echo; they are correlated, not anti-correlated.
   The stack still matters, but for a different reason: dflash2-alone
   REGRESSES echo (capped at 8 vs suffix rounds reaching 12), so suffix
   lanes 8..11 stacked on top (or K=11) are what protects echo-heavy
   stretches, while the drafter's real margin is prose/code-write. Phase
   2's K sweep and stacking A/B are now data-motivated, not a bet.

Greedy identity vs plain HF generate diverged once in 128 tokens at a
0.375-logit margin (~3 bf16 ulps at logit scale 25): accumulated
batched-vs-serial bf16 cache divergence, both continuations coherent. Not a
rig bug and not a q27 concern -- q27's verify lanes are bitwise twins of
plain decode, so identity holds by construction there.

Still open, and it is the one kill gate left: every DFlash2 number above is
on BF16 target hiddens. The quant-tap leg (q27 5.25-bpw residuals + fp8 KV)
needs the engine's tap-dump plumbing -- first thing in Phase 1.

**P0b -- drafter cost.** Eager bf16 on the 3090: 9.7-10.9 ms/round, flat in
context rows and accepted count = launch/python-bound, decomposed as forward
6.5 ms + propose 3.2 ms (the propose is a 7-step python loop; in q27 it is
one tiny kernel). Weight-read floors: bf16 3.85 GB = 4.1 ms on the 3090,
2.2 ms on the 5090; Q4 ~1.1 GB = ~0.6 ms. The design's 1-2 ms/round Q4
estimate stands; bf16 bring-up ~2.5-3 ms is plausible once graphed.

**P0c -- fold audit: CONFIRMED safe.** `k_finish_round`'s acceptance is a
pure equality chain over draft slots that are bare device ints (the MTP
chain writes 1..7 by name, the suffix drafter stages 8..W_PLUMB-1 -- the
pre-staged-draft pattern DFlash2 needs already exists); the record arena is
lane-indexed with no draft provenance; `flush_fold` replays by accepted
count alone; `refinish_round` is generic in m. No ladder assumption a block
drafter violates. One Phase-1 delta it surfaces: dflash2 rounds need a
verify-graph variant in which the MTP chain does not write the same slots
the drafter staged.

**Port-surface addendum.** Every draft layer wraps BOTH its attention and
its MLP in a `GroupedDynamicCausalConv` (kernel 2, group 16: a per-token
dynamic kernel added to a learned base, prepare/finish pair around each
block). The context rows still bypass all of it -- the "no conv on context
rows" claim above holds -- but the noise-row path is more than the plain
5-layer transformer the cost model described. Small tensors, elementwise
cost, real implementation surface.

**Tap contract for Phase 1.** HF `hidden_states[i+1]` = the residual stream
after layer i's second residual add, no final norm (entry 0 = embedding;
z-lab's `extract_context_feature` uses offset=1). So q27 taps x after the
post-FFN residual add at layers 5/19/33/47/61 -- matching ninfer's r_t^l.

## Phase 1: quant-tap gate (2026-09-06, same day) -- PASSED

The last kill gate is closed, and not narrowly. Method: `DFLASH_TAPS`
bumped to the v2 ids {5,19,33,47,61} (the v1 `--dump-taps` rig from the
parked design still works end to end), the four Phase-0 prompts dumped
through the CLI plain-greedy path on the 5.25-bpw artifact, and the drafter
replayed offline over those taps (`bench/dflash2/p1_qtap_al.py`). Controls:
the replay rig reproduces `dflash_generate`'s own acceptance on the same
trajectory (4.44 vs 4.55 tok/round, recompute-vs-cache numerics); and the
controlled leg teacher-forces the SAME q27 token streams through the BF16
target so only the tap source differs:

| traffic    | BF16 taps | q27 taps | delta |
|------------|----------:|---------:|------:|
| code-write |      3.18 |     3.24 |   +2% |
| prose      |      2.45 |     2.55 |   +4% |
| code-edit  |      4.06 |     4.44 |   +9% |
| echo       |      7.64 |     7.64 |    0% |

q27 taps are not degraded at all -- they score marginally HIGHER on every
non-echo type, which has a plausible mechanism: the live taps are
self-consistent with the trajectory that produced them, while the BF16
teacher-forced taps are slightly off-policy. (The uncontrolled per-prompt
swings, e.g. code-write 4.55 BF16-trajectory vs 3.24 q27-trajectory, are
trajectory content variance, not tap quality -- the controlled table is the
comparison that counts.) The int8 embedding rows feeding the noise columns
remain a small untested delta; they are high-fidelity (q8 + fp16 row
scales) and get covered by the bring-up parity test.

## Phase 1 complete (2026-09-06, same day): runtime + in-engine E2E

**Runtime (src/dflash2.cu + tools/dflash2_pack.py).** Eager bring-up module:
flat fp16/fp32 pack, context ingest (fc + context_norm + per-layer K/V with
k_norm/rope into an append ring), draft block (new grouped-dynamic-conv and
bidirectional-window-attention kernels; gemv/rmsnorm/rope3 reuse; host-side
top-16 + selector walk). Parity vs the z-lab torch reference on the same
dumps: 46/57 rounds propose the identical 7 tokens, 91.7% token match
(divergences are fp16-vs-bf16 tie cascades), AL equal within 2%.

**In-engine E2E (`q27 --dflash2 <pack.d2w>`).** The suffix-round pattern
with the DFlash2 drafter: host proposals staged into `d_draft_L[0..6]`,
`prep_round`, EAGER width-8 `spec_verify_forward` (new defaulted `taps`
arg retains each lane's five residual streams -- host branch, graphs and
the fused mirror byte-identical) + `spec_verify_tail`, fold, ingest the
accepted lanes' taps. Gate matrix, 192 tokens per prompt:

| traffic    | vs plain greedy | E2E tok/round | offline replay |
|------------|-----------------|--------------:|---------------:|
| code-write | IDENTICAL       |          3.25 |           3.24 |
| prose      | IDENTICAL       |          2.49 |           2.55 |
| code-edit  | IDENTICAL       |          4.43 |           4.44 |
| echo       | IDENTICAL       |          7.65 |           7.64 |

`--spec` on the same binary stayed byte-identical to plain (the shared
verify path is untouched; the taps param defaults to nullptr everywhere
else). Echo already runs 153.8 t/s fully eager.

**The one real bug of the bring-up, worth remembering:** `gdn_mix` and the
record arena run at the MEMBER `vw` (the LaneView's `vw` only drives the
attention/FFN sweep). The member defaults to 5, so a width-8 eager round
recorded only 4 speculative GDN rows -- any round accepting n >= 6 folded
unrecorded garbage into committed state and corrupted the stream (caught
by the byte-identity gate at token 6). Fix: `set_round_width(D2_W)` before
the loop, legal here because this mode captures no graphs.

**Eager round wall: ~47-50 ms** (vs ~18 ms for the graphed ladder round) --
drafter gemvs eagerly submitted, 7 MB logits D2H + host top-16/walk per
round, fp16 target head instead of the engine's quantized head, no graphs.
That cost order is Phase 2's whole job: drafter under its own graph,
on-device top-16 + selector walk, reuse the engine head/int8 embeddings
(measure the int8-embed delta then), tap-enabled verify graph captures
(the tap buffer is init-fixed, so capture is legal), K sweep 5/7/9/11,
suffix stacking at widths 9..12.

## Phase 2 (2026-09-06, same day): perf + K sweep

Four things landed; the wall analysis pointed at the two that were worth
doing and away from the two that weren't.

**On-device selector.** Per-round top-16 + selector walk moved off the host
(`k_d2_top16a/b` two-stage over the whole grid; `k_d2_walk` one block).
Proposals stay on device and reach the engine's `d_draft_L` by D2D copy --
no host sync in the round. The first cut (one iterative top-16 block) cost
4 ms/round; the two-stage version is negligible. tok/round unchanged
(3.29/2.51/4.34/7.64 on the smoke, exact vs the Phase-1 host walk -- tie
semantics preserved by ordering stage 2 on value-desc then id-asc).

**Engine quantized head reuse (the win).** The drafter's logits ran through
the pack's fp16 `target.head` (2.5 GB, 10.4 ms/round at DDR-limited fp16
bandwidth). Routing them through the engine's own Q8/Q4 output head instead
(`set_engine_head`, opt-out `Q27_D2_FP16HEAD=1`): **+37% throughput** --
code-write 64.4 -> 88.2 t/s, echo 147 -> 196 t/s -- and tok/round stayed
byte-identical (3.25/2.49/4.43/7.65). The Q8-vs-fp16 head numerics did not
shift acceptance, and E2E output is verify-decided so it stays identical to
plain greedy regardless.

**K sweep (offline, on the q27 taps).** tok/round by K:

| traffic    | K=5  | K=7  | K=9  | K=11 |
|------------|-----:|-----:|-----:|-----:|
| code-write | 2.94 | 3.29 | 3.47 | 3.35 |
| echo       | 5.79 | 7.64 | 9.10 | 10.61 |

Echo scales with K to the width-12 ceiling; code-write peaks at K=9 then
falls (deeper masks propose worse). K=7 is the balanced default; K=9 is the
echo-favoring point and the natural DFlash2 width when suffix is off. The
verify wall grows with width, so the throughput-optimal K is a Phase-3
tok/ms question, not a tok/round one.

**fp8 KV.** E2E under `Q27_KV=fp8` (the serving config) is byte-identical
to plain-fp8 at the same tok/round -- the drafter taps read the fp32
residual stream before the KV store, so KV quant never reaches drafter
acceptance. code-write 89.2 t/s, echo 200.2 t/s under fp8.

**Wall breakdown (nsys, decode phase, K=7).** Round ~37 ms. Split: width-8
verify through the 27B (`k_gemv_q4_n<8>` + tails) ~40%, the eager fp16
drafter forward + batched ingest (`k_gemv_f16_3`) ~35%, GDN/attn/norms the
rest. GPU-busy tracks wall -- the round is compute/bandwidth-bound, not
launch-bound, so CUDA-graph capture buys ~10-15%, not a multiple. The one
lever that would move the drafter third materially is a Q4 repack of the
drafter weights (3.85 GB fp16 -> ~1.1 GB, the design's ~0.6 ms floor), but
that shifts drafter numerics and deserves its own acceptance gate rather
than riding in on a perf commit. Both are deferred to after the Phase-3
verdict says whether the wall is even the thing to fix.

**Not done, deliberately:** suffix stacking (dflash2's 8 columns + suffix
lanes 9..11 in one verify) is the composition A/B -- it IS Phase 3's
three-way comparison, so it lives there, not here.

## Phase 3 (2026-09-06, same day): same-harness verdict

Same binary, same four prompts, fp8 KV (serving config), greedy. `--spec`
is the ladder; `--dflash2` is this drafter. Two numbers per cell:
tok/round (the fair, graph-independent measure) and t/s (wall).

| traffic    | ladder t/r / t/s | dflash2 K=7 t/r / t/s | dflash2 K=9 t/r / t/s |
|------------|-----------------:|----------------------:|----------------------:|
| code-write |  2.76 / 154      |  3.25 / 89            |  3.25 / 81            |
| prose      |  2.21 / 124      |  2.68 / 74            |  2.97 / 74            |
| code-edit  |  3.43 / 189      |  4.51 / 122           |  4.73 / 116           |
| echo       |  4.64 / 252      |  7.65 / 199           |  9.05 / 213           |

**Two things this settles.**

1. **The drafter is clearly better.** dflash2 wins tok/round on all four,
   by +18% (code-write) to +95% (echo). That is the drafter's whole reason
   to exist and it is proven on our quant, our engine, our traffic.
2. **It loses on t/s today, and exactly one thing is why.** The verify is
   free: `--p0b` shows the width-2..12 verify forward is 20.1 -> 21.3 ms,
   i.e. width-8 costs 0.7 ms more than width-2 (weight-bandwidth-bound, the
   premise of batched verify). The drafter, measured alone on the 5090
   (`dflash2_smoke`, drafter+ingest), is 26 ms/round -- and `k_gemv_f16_3`
   is 98% of it, one call maxing at 10.4 ms. Root cause, confirmed at the
   kernel: `k_gemv_f16_3` grids `(rows, ntok)` and each `(row, token)`
   block re-reads the full weight row, so the drafter reads its 3.85 GB of
   fp16 weights ONCE PER VERIFY COLUMN -- ~30 GB/round at W=8. The verify's
   own `gemv_q4_n` reads each weight once and shares it across all columns
   (~10 us/call for the same shapes). The drafter is on the wrong kernel.

**Suffix note.** On these single-turn prompts the suffix drafter never
fires (needs >=12-token committed echo), so `--spec` and `Q27_SUFFIX=1
--spec` are identical here -- the Phase-0 suffix wins were real agentic
echo. The composition A/B (dflash2 + suffix lanes 9..11) is therefore not
measurable on this corpus and is deferred to the live-CC trial. It is also
now lower-priority: dflash2's own K knob already reaches echo's ceiling
(K=9 gives 9.05 tok/round on echo, beating ladder+suffix's live ~6.86),
so the drafter may not need suffix stacking at all.

**Projection (the GO case).** Round = drafter + verify. Verify+fold ~10 ms
(the 36.5 ms eager round minus the 26 ms drafter). Put the drafter on the
`gemv_q4_n` path (weight read once, shared across columns, int4): the
design's ~0.6-2 ms floor. Round -> ~12-13 ms, and at the measured
tok/round: code-write ~3.25/13 ms = ~250 t/s (vs ladder 154), echo K=9
~9.05/13 ms = ~450+ t/s (vs 252). dflash2 wins t/s across the board once
the drafter is quantized, most decisively exactly where the tok/round lead
is largest.

**Verdict: GO**, conditional on Phase 4 (drafter Q4-g64 repack onto
`gemv_q4_n`, with a numerics gate: Q4-drafter acceptance vs the bf16
drafter, kill if >5-10% tok/round loss). Everything upstream of the repack
is done and byte-identity-gated; the repack is the one remaining piece
between a proven-better drafter and a shipped t/s win. Suffix stacking and
CUDA-graph capture fold into the live-CC trial after the repack lands.

## Phase 4 (2026-09-06, same day): the Q4 drafter repack

The Phase-3 root cause was that `gemv_f16_3` re-reads the weight per verify
column. Fix: pack the drafter's 47 matmul weights as Q4-g64
(tools/dflash2_pack.py `quant_q4`, identical to the engine's own format) and
route them through `gemv_q4_n` (`Dflash2::mmq`: quantize the activation once,
then the weight is read ONCE and shared across all W columns). Norm weights
and conv base kernels stay fp32; codebooks and the target embedding stay
fp16. Pack is `qwen38-dflash2-q4.d2w`; `--q8` builds a Q8 sibling as the
numerics fallback.

**Numerics gate: PASSED.** Q4-drafter tok/round vs the fp16 drafter, same
q27 taps: code-write 3.24 vs 3.29 (-1.5%), prose 2.42 vs 2.51 (-3.6%),
code-edit 4.24 vs 4.34 (-2.3%), echo 7.64 vs 7.64 (0%). All inside the
5-10% kill line; Q8 came back byte-identical to fp16 as expected. And
crucially the E2E output stays BYTE-IDENTICAL to plain greedy at K=7 on all
four (verify-decided), so the drafter's small acceptance loss costs nothing
in correctness.

**Cost.** Drafter+ingest on the 5090: 26.1 -> 12.2 ms/round (2.1x) -- the
weight-reread is gone. E2E round ~22 ms (drafter 12 + verify ~10).

**E2E throughput, K=7, fp8 KV, engine head (byte-identical to plain):**

| traffic    | ladder t/s | dflash2 Q4 t/s | delta |
|------------|-----------:|---------------:|------:|
| code-write |        154 |            148 |   -4% |
| prose      |        124 |            117 |   -6% |
| code-edit  |        189 |            205 |   +9% |
| echo       |        252 |            328 |  +30% |

dflash2 now wins where its tok/round lead is largest (echo, code-edit) and
sits at ~parity where it is smallest (code-write, prose). That is the
expected shape: round = ~12 ms drafter + ~10 ms verify, and the extra
tokens pay for the wider round only when acceptance is high.

**Two things between here and winning everywhere.**

1. **The drafter is still eager (12 ms, launch-bound), not the ~2 ms
   bandwidth floor.** 47 `gemv_q4_n` + attention/dconv/norm launches per
   round, each with a `quantize3`, all eager. CUDA-graph capture of the
   drafter forward (fixed shape per K, all pointers init-fixed) is the next
   lever; projected round ~13 ms -> code-write ~250 t/s, prose ~180, i.e. a
   win on all four. This is Phase 5.
2. **K=9 (width-10) has an identity edge case** -- code-write and prose
   diverge late (token 188 / 61) at K=9 while code-edit/echo hold; K=7
   (width-8) is byte-identical everywhere. The ladder captures verify
   graphs only for widths 2..8, and width-10 is a path exercised only by
   this mode; the fold/verify interaction above width 8 needs a look before
   K>7 ships. K=7 is the default and the shipped-safe point.

Baseline hygiene note (cost a detour): the Phase-1/2 plain baselines were
fp16-KV; the Phase-3/4 runs are fp8-KV, and fp8 vs fp16 KV changes greedy
tokens. Always compare a drafter run against a plain run at the SAME KV
setting -- the first "divergence" here was a stale fp16-KV baseline, not a
bug (plain greedy is deterministic run-to-run, and Q4/Q8/fp16 drafters are
all byte-identical to the matched fp8 plain).

## Phase 5 (2026-09-06, same day): verify graph -> wins everywhere; the width-8 wall

**The Phase-4 "drafter is 12 ms" number was a smoke artifact.** The
`dflash2_smoke` tool uses the fp16 PACK head (it has no engine), so 10.4 of
its 12 ms was one `gemv_f16_3` head call per round. Profiling the real E2E
(engine Q8 head) showed no pathological kernel: the round is the eager sum of
a small drafter (~2 ms) and the eager 64-layer width-8 verify (~18-20 ms).
The verify was the lever, not the drafter.

**Verify graph.** `spec_verify_forward` (with tap capture) + `spec_verify_tail`
is a fixed launch sequence over init-fixed buffers; every per-round input
(staged drafts, positions, d_token, d_P) is a device buffer written before
the round. So it captures once and replays -- exactly how the ladder graphs
its own verify, now with the tap D2D copies inside the capture (the tap
buffer is init-fixed). `Q27_D2_NOGRAPH=1` keeps the eager path; the two are
byte-identical and the graph is deterministic run-to-run.

**Result -- dflash2 wins t/s on all four at K=7, byte-identical to plain
greedy (fp8 KV, engine head):**

| traffic    | ladder | dflash2 K=7 | delta |
|------------|-------:|------------:|------:|
| code-write |    154 |         164 |   +6% |
| prose      |    124 |         130 |   +4% |
| code-edit  |    189 |         226 |  +20% |
| echo       |    252 |         361 |  +43% |

The graph added +10-15% over the Phase-4 eager verify and flipped code-write
and prose from parity to wins. Round is now ~20 ms (drafter ~2 ms eager +
graphed verify ~18 ms).

**The width-8 wall (why K stays at 7).** Mapping identity vs K on prose:
K<=7 (width<=8) byte-identical, K>=8 (width>=9) diverges -- and always at the
same token regardless of K, i.e. a hard width boundary at 8, not a
drafter-quality effect. Width 8 is exactly the ladder's structural max
(D_MAX_MTP=7 -> gate_maxd+1=8); widths 9..12 are reached only by suffix
rounds through *captured* graphs, and the eager verify/GDN/fold at width>8
has a latent bug the ladder never exercises. It is pre-existing engine-core
code, not the drafter integration, and K=7 is the right default anyway (the
Phase-0 balanced point, and it already wins echo by +43% without needing the
wider block). Fixing width>8 would unlock the K=9 echo/code-edit gains but is
its own engine task, tracked separately.

**Remaining, both optional.** (1) Wire dflash2 behind a server flag
(`Q27_DFLASH2`) for the live-CC trial and the suffix-composition A/B -- the
CLI single-turn wins are clean, but real agentic traffic (long context, echo
bursts) is the last validation before a default flip. (2) The ~2 ms eager
drafter tail: graphing it needs a device-indexed embedding lookup (the anchor
token currently bakes into a launch arg); ~1 ms/round, low priority now that
every traffic type already wins.

## Phase 6 (2026-09-06, same day): server wiring + the live-CC verdict

Wired the drafter into the serving path (`Q27_DFLASH2=<pack>`, single-slot /
`Q27_BATCH=0`): a per-engine `Dflash2` with a sliding context ring,
`dflash2_round` hooked into `decode_step` in place of `spec_round`, the
verify graph captured once in `d2_setup`, and the drafter reusing the
engine's own Q8 head AND Q8 embedding (the serving pack drops the fp16
head+embed -> 1.2 GB, and the KV pool reserves ~2 GB for it up front, or it
OOMs since the pool otherwise eats all free VRAM). The ring cold-resets each
turn -- warm turns restore the target's state but NOT the drafter's prefix
taps, a deliberate first-cut limitation.

**It works and it is correct.** Real Claude Code sessions (SWE-bench flask +
requests, via the agentic harness) ran to completion, edited the right
files, and hit prefix-cache warm turns (rid=3 prompt 23979, computed 176).
Output is verify-decided, so correct regardless of drafter quality.

**But it LOSES on live agentic traffic** (same two instances, same
single-slot config, only the drafter differs):

| | agg decode t/s | tok/round | reqs |
|---|--:|--:|--:|
| dflash2 (cold-start) | 164.6 | 3.17 | 27 |
| ladder + suffix      | 177.5 | 3.43 | 28 |
| delta                | **-7.2%** | | |

The single-turn CLI wins do NOT transfer. The reason is cold-start: agentic
CC is many short turns, the drafter ring resets each turn, and its attention
has no prefix context for the first stretch of every turn -- while the
ladder's MTP head drafts from the target's hidden state, which always
carries context through the KV cache. The CLI benchmarks each had one
prefill + one long decode, so the ring warmed once and stayed warm; live
traffic never gives that. (The suffix drafter accepted zero tokens on these
instances, so the incumbent here is pure MTP.)

**What this means.** The drafter is genuinely better token-for-token when it
has context (Phases 0-5), but on this box, in this engine, the MTP ladder's
always-warm context beats DFlash2's cold-per-turn ring on real agentic
traffic. Two things would close it, and both are real work, not tuning:

1. **Prefill tap capture** -- get the last ~2048 prompt tokens' taps into
   the ring during the batched prefill (`prefill_chunk`), so the drafter
   starts each turn warm. This is the single highest-value follow-up; the
   cold-start penalty is the whole -7%.
2. **Composition with the ladder** -- keep the MTP/suffix drafts for the
   cold rounds and let DFlash2 ride once the ring warms. More invasive
   (fused verify), lower priority than (1).

Until (1) lands, DFlash2 stays a CLI/warm-context win and NOT a serving
default. That is the honest state, and it is exactly what a live-CC trial
is for: the single-turn numbers were real but not representative.

## Prior art

z-lab/Qwen3.8-27B-DFlash2 (MIT, on disk at
`/mnt/ai/models/qwen38-27b-dflash2-bf16`); ninfer's algorithm contract and
op checklist (docs/maintainer/qwen3.8-27b-dflash2.md at their master,
cross-checked against vLLM `qwen3_dflash2.py` and SGLang's DFlash worker);
FlashRT and lucebox-hub/dflash per the v1 doc's credits.
