# Performance implementation, 2026-09-09

## 1. Sampler order and vocabulary scans

Corrected the CUDA filter chain to temperature-scaled weights, then top-k,
top-p, min-p. Previously min-p reduced top-p's denominator, removing an extra
boundary token on 5/36 saved production logit rows. Temperature-last filtering
is equivalent at the production T=1; no equivalence is claimed at other T.

The new synthetic regression uses weights `[1, .1, .055, .04]`, k=4, p=.95,
min-p=.05. The old kernel retains two instead of three tokens and fails both
support and probability checks (maximum probability error .04762). Corrected
and optimized kernels pass. Tests now compare probabilities with min-p enabled,
exercise a ragged 248321-token vocabulary, k=20/32/64, and require bitwise
single/multi-lane identity. The existing sampling and rejection tests pass.

Optimization: stop top-k bisection when its support has exactly k members.
For k<=32, compact and sort the surviving values, compute top-p on that small
set, then apply min-p. Larger supports use the general bisection path. The
exported full-softmax log normalizer and retained mass still describe the
actual final support for speculative rejection.

Scope limitation: top-k still uses the engine's pre-existing logit-threshold
representation. Exact ties crossing the top-k boundary cannot be split by
token ID with that representation; this change fixes filter order rather
than redefining that separate edge case.

RTX 5090, eight distinct saved logit rows, CUDA-event microbenchmark:

| Kernel | Median ms |
|---|---:|
| Corrected reference | 0.7036 |
| Optimized | 0.2335 |

Both match the intended support on all 36 saved rows. These timings compare
the corrected target distribution; the previous shipped kernel had different
semantics and measured 0.5925 ms in the earlier investigation.

Full 15-request replay, fresh server and private cache per arm, order
reference/optimized/optimized/reference, K=7, target T=1/p=.95/k=20/min-p=.05:

| Arm | Output tokens | Rounds | Decode ms | Decode tok/s |
|---|---:|---:|---:|---:|
| Reference A | 5067 | 1214 | 21861 | 231.78 |
| Optimized A | 5067 | 1214 | 21330 | 237.55 |
| Optimized B | 5067 | 1214 | 21377 | 237.03 |
| Reference B | 5067 | 1214 | 21941 | 230.94 |

All 15 output hashes match across all four runs. Combined decode time falls
2.50%; mean round time falls from 18.040 to 17.589 ms. Prefill is unaffected
by this decode optimization, so total request wall improvement is smaller.
The production service was restored to its original binary after testing.

Local artifacts: `build/perf-sampler-20260909/` contains the baseline and
candidate binaries, compiler/test logs, cubins, microbenchmarks, replay driver,
per-request results, and server telemetry. These artifacts are ignored by git.

## 2. Independent proposal temperature

`Q27_D2_TEMP` overrides only the DFlash2 selector's
proposal temperature. Unset follows each request's target temperature, as
before. The override is captured in both eager and graphed draft paths; the
walk stores its actual q for acceptance and residual correction. Target
sampler parameters are not modified.

GPU walk/rejection tests pass with target T=.9 and draft T=.9/.5/2, including
stored-q probabilities, draw histograms, target rejection histograms, greedy
identity, conditional codebooks, and residual fallback. Adding the control
with its default reproduces every output of the sampler-only candidate.

The same 15-request replay, fresh server/cache per temperature, K=7 and fixed
target sampler:

| Draft T | Decode tok/s | Tokens/round |
|---|---:|---:|
| Default (1.0) | 238.03 | 4.174 |
| 0.70 | 227.65 | 3.990 |
| 0.85 | 223.47 | 3.923 |
| 1.15 | 233.61 | 4.113 |
| 1.30 | 219.34 | 3.856 |
| Explicit 1.0 repeat | 237.22 | 4.174 |

Every alternative also loses separately on the first eight fixed-400-token
requests and the seven real Claude Code requests. Keep the default. This is
a local calibration sweep, not proof of an optimum for all tasks or contexts.
The default and explicit-1.0 runs have identical output hashes and round counts.
Production was restored after the sweep.

## 3. Reasoning effort

Ran two repeats each of ledger T14 and scheduler T12, at Claude Code
medium and default/high effort (rendered as xhigh by q27). Every trial uses a
fresh candidate server, private prefix cache, and disposable benchmark checkout.
The candidate includes the sampler fix/optimization and uses default draft
temperature. Agent task wall excludes setup; correctness grading follows code
inspection. No global reasoning-effort change has been made.

Ledger T14 completed; all four source patches were inspected and independently
passed the unchanged 40-test suite:

| Effort | Trial | Task wall s | Output tokens | Claude Code turns | Tests |
|---|---:|---:|---:|---:|---:|
| Medium | 1 | 98.54 | 17277 | 44 | 40/40 |
| Medium | 2 | 94.61 | 18367 | 39 | 40/40 |
| Default/high | 1 | 94.28 | 17594 | 42 | 40/40 |
| Default/high | 2 | 123.69 | 24159 | 14 | 40/40 |

Medium averages 96.57 s versus 108.99 s, but the first pair favors default
effort and the second default trial accounts for the difference. Two repeats
of this one task do not justify a global setting.

Scheduler T12, independently graded against the validation tag's 38 hidden
tests after source inspection (all five artifacts also pass TypeScript):

| Effort | Trial | Task wall s | Output tokens | Own tests | Hidden tests |
|---|---:|---:|---:|---:|---:|
| Default/high | 1 | 475.37 | 77661 | 23/23 | 38/38 |
| Medium | 1 | 421.93 | 69071 | 29/29 | 37/38 |
| Medium | 2 | 526.23* | 77927 | 20/20 | 38/38 |
| Default/high | 2 | 334.21 | 55451 | None written | 36/38 |
| Medium, 262K ctx rerun | 3 | 952.94 | 115247 | 27/27 | 37/38 |

Medium 1 schedules only 89/100 events in the large case; that test alone takes
59.6 s. Default/high 2 schedules 29/30 and 85/100 in the scaling cases and
writes no own tests. The fastest completed scheduler run therefore also has
the weakest hidden-test result. These are functional and typecheck results,
not the benchmark's combined coverage/lint/quality score.

*Medium 2 is excluded from effort wall/completion comparisons: the isolated
server used `--ctx 131072`, and Claude Code terminated with `Prompt is too long`
after prompts of 130655 and 130699 tokens. Production allows 262144. Its
existing artifact passing hidden tests does not make that a completed agent
run, and the context cap is not an effort-quality failure. A fresh medium
trial at production's context capacity completed in 952.94 s, with 59 Claude
Code turns and a maximum prompt of 157845 tokens. It passes 27 own tests and
TypeScript, but still fails the hidden 100-event scheduling case (37/38).
The two valid completed medium scheduler trials average 687.43 s, versus
404.79 s for default/high; neither completed medium run passes all hidden
tests. This is a small pilot, with different context caps for the rerun,
not a statistically powered estimate or proof that default always wins.

Across the initial eight trials, decode consumes roughly 73–89% of agent wall;
prefill consumes 10–21%, and tools/client overhead accounts for the remainder.
Different output lengths and trajectories dominate the effort comparison.
The 262K rerun spends 601.59 s decoding, 73.67 s prefilling, and 277.69 s
outside those measured engine phases, including a CPU search experiment the
agent eventually stops itself. That is about 29% of its wall time outside
decode/prefill, illustrating why GPU throughput alone does not predict
agent completion time.
Keep default effort: this pilot does not show a repeatable quality-preserving
global wall-time improvement from medium.

## 4. Proposal-only head shortlist (pending replay)

Added opt-in `Q27_D2_SHORTLIST=/path/to/ids.i32`. At startup it validates a
unique, bounded token-ID list and gathers the target head's existing Q4/Q8
bytes and scales into a smaller proposal head. It does not requantize them.
Top-16 selection returns original vocabulary IDs, including tie breaking;
the target head, target sampling, embedding, and codebooks remain unchanged.
The full head remains the default. A 131072-row Q4 head adds approximately
340 MiB of device storage because the target still needs its full head.

`tools/d2_shortlist.py` creates this binary map from token-frequency counts
and tokenizer special IDs. The experimental map uses NInfer checkout
487f897's `tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64` and
the local Qwen3.8 tokenizer configuration. It keeps 131072 rows, including
all 21 special tokens; training count coverage is 99.3912%, held-out token
count coverage 99.0936%, held-out acceptance-count coverage 99.1926%. These
are corpus coverage figures, not measured DFlash2 acceptance or task quality.
Map SHA256: `c348bf8d2e70d502718ebb8eeefdc794936dc8462ae357b802ba0a987fad756c`.

The kernel suite builds for sm_86, sm_89, and sm_120. On the RTX 3090 and 5090 it passes
the sampler/rejection suite, Q4/Q8 row-gather byte identity, and mapped top-16
tests with ties, excluded high logits, and ragged vocabulary size. The RTX
5090 is running the shortlist replay after the effort trials.
