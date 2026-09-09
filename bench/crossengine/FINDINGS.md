# ninfer vs q27 — real-world head-to-head, 2026-08-17

One 5090, one harness, one accounting convention, four legs. Vanilla Qwen3.6-27B
on both engines.

    q27 77f46bf | ninfer 604bdc5 | driver 580.119.02 | Claude Code 2.1.170

| leg | artifact | bytes | role |
|---|---|--:|---|
| `q4s` | `qwen36-27b-mtp-q4s.q27` | 15.46 GB | q27 canonical anchor (`f64e7c02`) |
| `nint` | `qwen3_6_27b.ninfer` | 17.50 GB | ninfer groupwise-int — the format control |
| `q5f` | `qwen36-27b-mtp-q5f.q27` | 18.22 GB | q27, bpw-matched to nvfp4 within 0.5% |
| `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | 18.32 GB | ninfer NVFP4 — their headline |

The 08-15 A/B compared q4s against nvfp4 at +18% bits with no correctness arm.
`q5f` exists so the quality comparison is bpw-matched and the format effect is
separable from the engine effect.

---

## Headline

**The result splits by workload, and the two halves do not contradict each other.**

- **Real agentic traffic → q27**, 4.10x effective prefill at matched bpw, on the
  strength of prefix reuse.
- **Batch throughput → ninfer NVFP4**, by 1.91x, on the strength of fp4 MMA at width.
- **Quality → a tie.** At matched bpw, 66/75 vs 65/75.

---

## 1. Agentic (12 SWE-bench_Verified instances, Claude Code driving each engine)

| leg | inst | non-empty | gold | wall/inst | decode t/s | reqs |
|---|--:|--:|--:|--:|--:|--:|
| q4s | 12 | 10 | 9 | 48 s | 215.5 | 254 |
| nint | 12 | 12 | **11** | 327 s † | **261.5** | 308 |
| q5f | 12 | 11 | 10 | 47 s | 223.4 | 266 |
| nvfp4 | 12 | 12 | **11** | 97 s | 250.2 | 230 |

† nint hit the harness's 700 s cap on 3 of 12 instances (2103 s of its 3929 s
total). Its 9 uncapped instances averaged **203 s**. The true uncapped mean is
higher than 327 s, not lower — treat 327 as a floor.

`decode` is tokens ÷ summed decode windows, client-observed through the tap, one
convention for both engines.

**Decode is close, and ninfer is ahead.** 261.5 / 250.2 against 215.5 / 223.4.
Wall-clock is where q27 wins, and it is entirely a prefill story.

---

## 2. Prefix reuse — the finding that decides the agentic half

Reuse is taken from each engine's **own** telemetry, not inferred: q27's
`[gen] prompt=N prefix_hit=M` server lines, ninfer's
`result.prefix_cache_hit_tokens` reqlog field.

| leg | reqs | reuse (token-wtd) | reqs w/ any hit | cold prefill | **effective** | prompt tok |
|---|--:|--:|--:|--:|--:|--:|
| q4s | 254 | **88.72%** | 84.3% | 3332 | **24,224** | 8,041,948 |
| nint | 311 | **0.00%** | 0% | 2776 | **2,717** | 10,173,421 |
| q5f | 266 | **92.13%** | 88.0% | 3300 | **31,314** | 8,846,341 |
| nvfp4 | 230 | **0.00%** | 0% | 7765 | **7,641** | 7,200,662 |

`effective` = all prompt tokens ÷ all TTFT, i.e. the rate that actually sets
wall-clock. The comparison that counts is the **bpw-matched** one:

    q5f / nvfp4  = 4.10x     <- the honest number
    q4s / nint   = 8.92x     <- unmatched bpw, flattering

**4.10x, not 8.9x.** nvfp4's cold prefill is genuinely 2.35x q5f's (7765 vs
3300) — the fp4 format advantage is real and it claws back more than half the
reuse gap. Quoting 8.9x would be comparing q27's best tier against ninfer's
slower format.

ninfer's own per-request telemetry, not our measurement:

    nint    311 reqs   100% full_reset   hit_tokens=0   computed_prefill == prompt
    nvfp4   230 reqs   100% full_reset   hit_tokens=0   computed_prefill == prompt

The traffic is identically shaped on both sides — a monotonically growing
transcript, the ideal case for reuse:

    q27 q4s                       ninfer nint
    turn  in_tok   ttft_ms        turn  in_tok   ttft_ms
       6   31628      4847           4   29799     10359
       7   31815       381 ←reuse    5   30014     10471
       8   32643       387 ←reuse    6   30223     10501
       9   32757       497 ←reuse    7   30514     10625
      11   32960       153 ←reuse    9   30808     10731

q27 appends ~100–800 tokens per turn and prefills only those. ninfer re-prefills
all ~30k every turn, TTFT pinned at ~10.5 s.

### Why — design-bounded, but the exact failing predicate is not pinned

Be careful here; an earlier draft of this document overstated the case.

`request_plan_impl.h:186-202` tests for a prefix match at exactly **two**
offsets — `sequence.execution_frontier` (append) and one
`sequence.rewrite_checkpoint` (restore). There is no scan, so there is no
degraded hit: you match one of the two or you get `full_reset`. Their maintainer
docs state the boundary directly:

    ### 1.1 Non-goals
    - arbitrary longest-common-prefix reuse;

    当前 Qwen3.6 target 的 Linear Attention state 不能从 KV prefix 单独重建
    落在这两个 checkpoint 之外的更短 token prefix ... 本设计不支持

The underlying constraint is real and shared: Qwen3.6 is hybrid GDN, and
linear-attention state cannot be rebuilt from a KV prefix, so a hit must be
proven by a complete `SequenceState` snapshot.

**But that alone does not explain a zero**, and the distinction matters. The
same document says append-at-frontier *is* supported, and Claude Code traffic is
append-shaped — q27's hits sit at 96–98% median prefix depth, i.e. exactly the
append case. Yet across every logged request in this run, ninfer's
`append_frontier` path fired **zero** times. Its `restore_turn_checkpoint` path
*does* fire (38 times) — but only on short benchlocal prompts, never on agentic
traffic. So ninfer's reuse machinery is not broken; it simply never engages here.

The most likely mechanism, from the source but **not confirmed by instrumenting
the failing predicate**: `AppendAtFrontier` compares against a ledger of raw
*generated* token IDs, while an agentic client re-sends that turn re-rendered and
re-tokenized (tool calls become structured blocks, then get re-serialized), so
the byte sequences differ and the match fails. The documented non-goal then
explains why there is no fallback to catch it.

Honest summary: **the design bounds reuse to two offsets and provides no
partial-hit fallback; on agentic traffic neither offset ever matched.** Which
predicate rejects is unpinned.

**Robustness.** Restricting to agentic-shaped requests (prompt > 5k tokens),
reuse is 0% in *both* harness configurations — 299 + 218 requests post-thinkstrip
and 178 + 200 pre-thinkstrip, 695 in total, every one `full_reset`. The zero is
not an artifact of the request normalization.

**Ruled out:** thinking mode (102 requests with `enable_thinking=false` also got
zero); concurrency (`max_concurrency: 1`, never contended); config
(`prefix_reuse: true` throughout).

### Three engines, three answers to the same GDN problem

| engine | strategy | agentic reuse |
|---|---|--:|
| ninfer | 2 fixed resume offsets; LCP an explicit non-goal | **0%** |
| llama.cpp | N recurrent-state checkpoints (`--ctx-checkpoints`, default 32) | bounded by N |
| q27 | arbitrary LCP | **88.7–92.1%** |

q27 is the only one of the three doing arbitrary LCP on a GDN model. This run
prices that at **4.10x effective prefill** against a bpw-matched ninfer tier.

---

## 3. Concurrency ladder — ninfer wins this half

| leg | C=1 | C=2 | C=4 | C=8 | peak | scaling |
|---|--:|--:|--:|--:|--:|--:|
| q4s | 135.8 | 218.8 | 307.6 | 412.9 | 412.9 | 3.04x |
| nint | 138.8 | 160.7 | 218.2 | 349.8 | 349.8 | 2.52x |
| q5f | 127.1 | 194.2 | 290.9 | 385.9 | 385.9 | 3.04x |
| nvfp4 | 159.3 | 277.5 | 494.9 | **790.0** | **790.0** | **4.96x** |

**ninfer NVFP4 peaks at 1.91x q27's best.** Its 4.96x scaling reproduces their
published 5.67x closely, on our silicon, through our harness.

> **CORRECTED 2026-08-18. The measurement stands; the mechanism below was wrong.**
> The original text read:
>
> > This is a **format** effect, not a batching-machinery effect: ninfer's own
> > int leg scales only 2.52x and peaks *below* q27. fp4 MMA engaging at batch
> > width is doing the work, exactly as the 08-14 notes predicted.
>
> fp4 MMA is not doing the work, and cannot be. T2 built a decode-shaped nvfp4
> block-scaled MMA tile and measured it at the union width batched decode
> actually runs: **9.5% of the card's dense fp4 peak at M=16**. Nothing at
> decode width is compute-bound, so there is no arithmetic bottleneck for fp4 to
> relieve -- and on the axis that does bind, nvfp4 is the LARGER format (4.50
> bpw against Q4_G64's 4.25, 1.0588x the bytes). Measured against q27's actual
> decode GEMM the fp4 tile was 1.06x, and 0.95x once the kernel-technique gap
> was divided out. BUILDLOG 2026-08-18 (c).
>
> **What is actually happening**, from ninfer's own reqlogs (`server.{nvfp4,nint}
> .ladder.reqlog.jsonl`, shipped instances, 15 `request_done` each):
>
> | leg | tok/round C=1 -> C=8 | round wall C=1 / 2 / 4 / 8 |
> |---|---|---|
> | nvfp4 | 2.397 -> 2.384 | 15.04 / **15.00** / 16.18 / 18.90 ms |
> | nint | 2.305 -> 2.325 | 16.58 / 22.02 / 38.77 / 43.40 ms |
>
> Both legs hold `draft_window=3` and ~0.46 acceptance at every rung. The two
> tiers differ by **1.025x on tokens per round and 2.30x on round wall**: the
> gap is not speculation and it is not arithmetic, it is that nint's round gets
> 2.6x slower with concurrency while nvfp4's grows 26%. **nvfp4's round wall is
> flat from C=1 to C=2 (15.04 -> 15.00 ms)** -- doubling the batch cost nothing.
> No arithmetic-throughput story produces that; a "the small-batch kernel was
> leaving the machine idle" story does.
>
> The code names it. `text_policy()` returns `AllowA4` for `QType::NVFP4` and
> `A16Only` for every other qtype (`src/targets/qwen3_6_27b/impl/variant.cpp`),
> and the W4A4 route engages above ~4 tokens
> (`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:87`). So ninfer's
> int tier is locked out of its own batch-capable kernel family by policy and
> runs eight near-independent small-T GEMVs. **fp4 did not make nvfp4 fast;
> A16Only made nint slow, and the qtype is the flag that gates the good path.**
> That also means the within-engine control does not license "format effect" --
> it holds the scheduler fixed but not the kernel family.
>
> **This does not transfer to q27.** q27 already has what nvfp4 has and nint
> lacks: `k_vgemm` is a real batch-capable MMA GEMM measured at 84% of streaming
> SOL at M=16. There is no idle machine to reclaim, which is exactly why T2
> measured 1.06x rather than 1.71x. q27's own gap decomposes, per-stream at
> C=8, as **2.31x = 1.34x tokens-per-round x 1.71x round wall** (q4s: 1.777
> tok/round at 32.41 ms, from its own `[req]` lines) -- a speculation-throttle
> term and a round-overhead term, neither of which is a weight-format question.
> Follow-on: `docs/plans/2026-08-18-batch-round-budget.md`.
>
> The leg-level control is otherwise clean, and was re-checked: `harness/legs.sh`
> launches both ninfer legs from one shared branch with identical flags
> (`--kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft --prefill-chunk
> 1024 --no-thinking --max-concurrency 8 --max-context 16384 --kv-capacity
> 131072`), differing only in `--port`, `--model-id` and artifact path. The
> confound is at the ARTIFACT level, not the flag level: the two tiers also
> differ in embedding/head quant and in which tensors are kept at BF16, so
> "nvfp4 vs nint" was never a clean single-variable format comparison.

q27's 412.9 t/s at C=8 lands inside the 406–420 band from M3a — independent
confirmation via a different measurement path.

**Convention calibration.** Client-side accounting agrees with q27's own `[req]`
telemetry to within 0.1% at every rung (135.8 vs 135.8, 412.9 vs 412.6). This is
what licenses comparing ninfer's 790 to q27's 412.9 and to the existing ladder
history.

---

## 4. Quality — a tie at matched bits

benchlocal `--medium`, 5 packs / 75 deterministic-verifier scenarios, temp 0.

| leg | total | toolcall | instrfollow | structout | dataextract | reasonmath |
|---|--:|--:|--:|--:|--:|--:|
| q4s | 64/75 | 14 | 12 | 14 | 13 | 11 |
| nint | 64/75 | 14 | 12 | 13 | 13 | 12 |
| q5f | **66/75** | 15 | 12 | 14 | 13 | 12 |
| nvfp4 | **65/75** | 13 | 13 | 14 | 13 | 12 |

**The bpw-matched pair is 66 vs 65** — one scenario out of 75, i.e. noise. All
four legs sit in a 64–66 band.

Engine choice does not measurably affect correctness here, which is the expected
result for two engines serving the same weights, and it means the performance
numbers above carry no quality asterisk. Note also q4s→q5f buys 2 scenarios, so
q27's own 4.6→5.25 bpw step is worth about as much as the engine choice: nothing
measurable on these packs.

---

## 5. Methodology — bugs found, and what each would have gotten wrong

Every one of these produced a plausible, publishable, **wrong** number first.

| # | Bug | Would have concluded |
|---|---|---|
| 1 | Tap used `HTTPResponse.read(8192)`, which blocks until the buffer fills or EOF — sub-8 KB replies arrived as one block, TTFT absorbed the decode | ~⅔ of requests reporting 4–5-digit token rates |
| 2 | Claude Code sends `output_config:{"effort":"high"}`; q27 ignores it, ninfer 400s | "ninfer cannot run Claude Code" |
| 3 | ninfer validates `model` on `/v1/chat/completions`; q27 ignores it | "ninfer scores 0/75 on quality" |
| 4 | `--max-context 16384` made ninfer's auto KV capacity resolve to 16384 **total**, so one sequence fit and the ladder serialized | **"ninfer does not scale with concurrency"** — the exact opposite of true |
| 5 | benchlocal's `--no-thinking` sends `chat_template_kwargs.enable_thinking`; ninfer wants it top-level | quality 0/75 again |
| 6 | Claude Code's `thinking:{"type":"adaptive"}` shadowed ninfer's `--no-thinking` | decode ordering **reversed** (see below) |

Bug 1 was caught with a regression test using a fake engine emitting 10 deltas
50 ms apart; reverting to `read()` reproduces the pathology exactly (0.17 ms
decode window, 57299 t/s), so the test pins the bug rather than merely passing.

### Normalization applied

The tap strips `output_config` and `thinking`, and moves
`chat_template_kwargs.enable_thinking` to top-level — **uniformly on every leg**.
q27 ignores all three, so this is a no-op there; it was verified as such twice
(quality 64/75 identical via direct and tap paths; agentic 9/12 gold and 10/12
non-empty reproduced across harness versions). The alternative — one engine
silently discarding what the other enforces — is not the same request.

### Bug 6 is worth its own note

`translate.cpp` resolves `request.enable_thinking.value_or(server.enable_thinking)`,
so an engaged optional shadows `--no-thinking`. 167/190 nint requests ran with
reasoning on while q27 ran hard server-side `think=0`. Correcting it:

| nint | before | after |
|---|--:|--:|
| gold | 10/12 | 11/12 |
| decode agg | 174.4 | **261.5** |
| MTP acceptance | 74.9% | 78.5% |

**The ordering reversed** — ninfer went from losing decode to winning it. The
mechanism is visible in ninfer's telemetry: thinking-off switches its sampler
profile (temp 0.7 / top_p 0.8 instead of 1.0 / 0.95), a tighter tail makes the
draft head land more often, and speculative decode gets cheaper.

Generalization worth keeping: **decode throughput on a spec-decode engine is a
function of the sampler.** Any cross-engine decode number quoted without pinning
sampling is close to meaningless.

---

## 6. Caveats

- **n=1 per instance.** No repeats. Gold-hit counts are indicative, not precise.
- **Reuse rate is coupled to conversation length**, and trajectories diverged
  between legs (per-instance request counts range 2 to 82). The reuse comparison
  is between different conversation populations, not identical ones.
- **An earlier draft reported 80%/83% reuse** from a tap-side threshold heuristic
  (prefill_tps > 15000 as a proxy for "warm"). The numbers here come from q27's
  direct `prefix_hit` telemetry instead. The heuristic undercounted; prefer the
  direct instrument.
- **Neither engine is deterministic.** q27 logged `T=1.000 top_p=1.000`; ninfer
  draws a fresh random seed per request (`translate.cpp:46-52`), so its runs are
  structurally non-reproducible. Sampling is *not* matched between engines — each
  applies its own defaults when the client sends none, which is the real-world
  configuration but not a controlled one.
- **nint's 3 timeouts** cap 53% of its wall time. 327 s/inst is a floor.
- **Legs ran sequentially**, not interleaved. Thermal drift not controlled.
- **The `Explore` early-quit** (2 on q4s, 1 on q5f) is a model trajectory
  artifact, *not* an engine property — the ninfer legs never generated that call
  at all, rather than generating and recovering from it. Do not read it as q27
  parsing more strictly.
- **VRAM is not comparable as stated** (q4s/q5f 30656 MiB, nint 27410, nvfp4
  28200): q27 auto-sizes context to fill VRAM while ninfer was given an explicit
  `--max-context`. Different policies, not different efficiency.
- **Not comparable to the 2026-07-14 baseline** (11/12 gold, 202.7 t/s): that run
  was greedy, q8 KV, ~5.25 bpw, and an older Claude Code.

---

## 7. What this means for q27

1. **Prefix reuse is q27's strongest differentiator, and it is now measured.**
   4.10x effective prefill against a bpw-matched ninfer tier on real agentic
   traffic, against a competitor for whom the gap is architectural and
   documented as a non-goal. That is a better claim than "q27 is fast" and it is
   defensible. Note it is 4.10x and not the 8.92x the q4s/nint pair suggests --
   ninfer's fp4 cold prefill is 2.35x q27's and claws back over half the gap.

2. **The prefill-performance plan's own ROI gate now has its number.** Measured
   cache miss rate is **7.9–11.3%** of prompt tokens on real agentic traffic
   (token-weighted, from q27's own `prefix_hit` telemetry), with only default
   in-memory reuse (`--prefix-cache` was not enabled). The plan states its value
   scales with miss rate, not prefill share, and that "not worth it yet" is a
   legitimate outcome. At a 17–20% miss rate, a free perfect prefill buys little
   on this workload.

3. **The batch ladder is where q27 loses, and the cause is identified.** 412.9 vs
   790.0 at C=8, entirely attributable to fp4 MMA at width — ninfer's int leg
   peaks below q27. The phase-2 fp4 NO-GO was decided on *prefill* quality and
   VRAM grounds; this is a different question (fp4 at batch width for decode
   throughput) and the ladder says it is worth 1.91x.

4. **Quality parity at matched bpw removes an objection.** Any future fp4 tier
   has a clean quality reference point on the same packs.

---

## Reproduce

    bash bench/crossengine/harness/run_all.sh <outdir> [arms] [legs]
      arms  agentic,quality,ladder
      legs  q4s,nint,q5f,nvfp4,llama,vllm

`harness/analyze.py bench/crossengine` regenerates every table in RESULTS.txt
from the data committed beside it.

Resumable per leg/arm; a leg/arm whose output exists is skipped.
`harness/test_tap.py` is the measurement proxy's own regression test.

`isolation/` holds the two attribution experiments the headline claims rest on:
the vLLM MTP corruption A/B (3 configs x 4 instances) and llama.cpp's slot
ceiling walk (--parallel 8 OOMs, 6 boots).

Superseded runs -- six of them, each a harness bug caught before it reached a
conclusion -- are not committed; they were bulky and their lesson is recorded in
section 5 instead. Quality JSONs are committed with the per-scenario transcripts
stripped (1.1 MB -> 5 KB each); the scores and per-pack breakdowns are intact.
Captured request bodies are stripped from the tap logs: those are Claude Code's
own system prompt and tool schemas.

## 2026-09-09 addendum: the DFlash2-era standing (see agentic-2026-09-09/README.md)

Both engines now run block drafters and ~97% prefix reuse. q27's
production recipe decodes within 6% of ninfer's DFlash2 arm (207 vs 221
t/s aggregate; 3.89 vs 4.19 tok/round) but takes 3x the wall per instance
(108 vs 36 s) because its Claude Code sessions run 25 turns and 18K output
tokens per instance against 15 and 6K -- the same pattern on 09-07 and
09-08. Section 2's finding (reuse decides the agentic half) is now a tie;
what decides it is trajectory length, and that is unattributed: quant
tier (Q4_G64 vs NVFP4), the rendering of "medium" effort, or the parser.
The 08-17 table below is the pre-fix ninfer; keep both.

## 2026-08-22 addendum: Qwen3.8 quality legs, q27 vs llama.cpp

A `llama38` leg (Qwen3.8-27B-MTP as Q5_K_M, 19.5 GB, within 0.3 GB of q27's q6
tier) joins `q38`. Quality arm only, 75 problems, pass@3, greedy, no-think.

| leg | total | toolcall | instruct | structout | dataextract | reasonmath |
|---|---:|---:|---:|---:|---:|---:|
| q38 (q27, default tier) | 62/75 | 13 | 13 | 14 | 11 | 11 |
| llama38 (Q5_K_M, q8_0 KV) | 54/75 | 13 | 13 | 14 | 3 | 11 |

**Read the dataextract column with care.** Re-issuing those prompts at both
engines directly (no tap proxy, three thinking-field variants) reproduced the
gap deterministically and showed what it is: llama.cpp emits the correct object
wrapped in a one-element array (`[ { ... } ]`) on seven of the eight, and q27
does the same on DE-02 where llama.cpp does not. Both engines flip a near-tie
`{`/`[` first token; the verifier's top-level shape check scores a
wrapped-correct answer 0/10. The other four packs are identical to the problem.
On fixed prompts the two engines are at parity; the 54 is a verifier artifact of
a coin-flip first token, not an extraction-quality gap.

Also measured the same day, for the record: for a tool-free request the two
renderers are byte-identical (DE-01, 1898 chars, empty diff); q27's speculative
greedy path is token-identical to plain greedy (256/256); fp8 KV moves the
greedy trajectory at token 39 of a continuation and costs +0.04% NLL on 51,000
predictions; q6 is -1.2% NLL against the default tier. `llama-perplexity`
segfaults on this GGUF with every flag combination tried, so there is no
llama.cpp-side NLL here.
