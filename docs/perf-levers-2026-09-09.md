# Performance levers after the September 9 review

q27 is close to ninfer on engine cost per token. A decisive agentic wall-time
win needs better speculative acceptance, fewer tokens/turns at maintained task
quality, or a workload where serving policy still costs seconds. Small kernel
work can help establish a lead, but the measured remaining housekeeping costs
cannot explain a threefold task-wall difference.

This review used q27 HEAD `e81ea6f`, the existing production server binary,
the retained campaigns, and the local ninfer benchmark checkout `487f897`.
New experiments ran on the RTX 5090. Production was paused with user permission
and restored afterward with its original launch recipe; no engine or production
configuration files were changed.

## New measurements

Fresh servers replayed `build/reqlog-gate.jsonl`: eight seeded, 400-token
OpenAI requests at roughly 2.3K/6.6K context, followed by seven recorded Claude
Code requests. Each trial had an empty private tmpfs prefix cache. Trial order
was K=7, 5, 6, 7. All model load digests matched `b743d26b1f0562a9`.

| K | Output tokens | Rounds | Tokens/round | Decode tok/s | Round ms |
|---|---:|---:|---:|---:|---:|
| 7, first | 5,026 | 1,251 | 4.018 | 224.19 | 17.920 |
| 5 | 4,848 | 1,281 | 3.785 | 219.44 | 17.247 |
| 6 | 5,172 | 1,363 | 3.795 | 216.02 | 17.566 |
| 7, repeat | 5,026 | 1,251 | 4.018 | 224.24 | 17.916 |

The K=7 repetitions matched all 15 output hashes and every request's round
count; replay wall was 40.4 seconds in both. Narrowing makes rounds cheaper
but loses on aggregate decode here. On the fixed-output first eight requests,
K=7/5/6 delivered 225.1/216.7/207.9 tok/s. The seven Claude Code requests alone
slightly favor the narrower blocks on decode rate, so the evidence does not
exclude a request-specific policy. That subset is too small to establish one.
Different K values sample different continuations; these are replay results,
not task-success measurements.

The test context allocation was 131,072, compared with production's 262,144;
the private cache budget was 3 GB, compared with production's 40 GB. All four
arms used those same limits. The recording is heavily weighted toward cold
synthetic requests, so its 17.7-second prefill total is not a production
prefill-share estimate.

An Nsight Systems node trace of the first K=7 request reproduced its output
and 95 rounds but inflated decode from about 1.66 s to 4.32 s. Do not use its
wall time to price host overhead. Its kernel attribution identifies:

| Kernel group | GPU milliseconds per round |
|---|---:|
| Main Q4 verify GEMM, excluding head | 9.03 |
| Width-8 Q8 GEMVs, across draft and verify | 1.65 |
| Fused norm/quantize | 0.82 |
| GDN verify mixer | 0.77 |
| GDN commit scan, partly overlapped | 0.69 |
| Target nucleus | 0.61 |
| Drafter vocabulary head | 0.45 |
| Target vocabulary head | 0.43 |
| Verify split-K reduction | 0.36 |

These are kernel durations, not additive critical-path savings: the fold runs
on a side stream. Clean server telemetry puts draft around 2.45 ms, verify
around 15.0–15.2 ms, and the timed host remainder around 0.23 ms.

## 1. Target sampler: a concrete small speed lever and an unresolved semantic mismatch

`src/blocks.cu:nucleus_body` uses one CTA per lane to repeatedly scan 248,320
logits: 24 top-k bisection iterations and 16 top-p iterations, plus normalization
passes. With the production top-k=20, the useful final candidate set is tiny.
A tiled top-k selection followed by filtering/normalizing the small candidate
set is the most concrete remaining local kernel experiment.

The shipped kernel was extracted from the server executable and invoked via
the CUDA driver on eight distinct saved prompt-end logit rows. Five batches
of 100 launches measured **0.5924–0.5925 ms per eight-lane call**, independently
confirming the profiler's 0.61 ms. Reducing this to 0.1–0.2 ms would save
approximately **2.2–2.8% of a 17.9 ms round**. This is a target, not a measured
implementation speedup; eliminating the entire kernel would save only 3.3%.

Before optimizing, settle the intended filter semantics:

- q27 CUDA applies min-p before calculating the mass used for top-p.
- ninfer `487f897`, in `sampling_normalize_support`, calculates top-p's target
  mass before applying min-p. Its speculative path uses that helper too.
- q27's CPU comparison in `src/test_kernels.cu:check_nucleus_contract` also
  applies top-p before min-p, despite the CUDA implementation doing otherwise.

Min-p's threshold is invariant to normalization, but removing its rejected
tokens changes the denominator for top-p. Thus the filters do not generally
commute. Matching temperature/top-k/top-p/min-p values does not match the law.

Using 36 retained baseline prompt-end logit dumps, the actual shipped GPU
kernel kept a different support on **5/36 rows** from the other order evaluated
on the same FP32 logits. The additional support carried up to **4.83%** of
probability mass; mean total variation across all rows was 0.52%. The same
CPU filter-order comparison on the 36 fold-last candidate dumps differed on
4/36 rows. These dumps are warm-turn probes, not a representative sample of
every token in agentic reasoning.

This establishes a mismatch, not its causal effect on reasoning length.
ninfer's weight/KV formats, BF16 logit materialization, and RNG still differ.
The existing single-prompt reasoning study does not rule sampling out.
Add a counterexample to the CUDA/CPU contract test, choose the intended law,
then benchmark a faster implementation against that law. A new exact top-k
path also needs explicit tie/boundary handling rather than assuming byte
identity with bisection.

## 2. Improve DFlash2 proposals at fixed K=7

`src/dflash2.cu:k_d2_walk` uses the target's inverse temperature for the
proposal selector. There is no separate proposal-temperature control.
Calibrating the proposal temperature, or the relative unary/selector-edge
score scale, is a cheap experiment: only the 16-candidate walk changes.

Keep the target sampler fixed. Rejection sampling can preserve the target
distribution under a different proposal provided the stored q is exactly the
law used to draw it, including CDF rounding/fallback behavior. Seeded output
identity across different proposals is not expected.

Measure acceptance by lane and request class on full identical histories.
First sweep a small proposal-temperature range around the current value;
do not change both temperature and edge scaling in the first experiment.
Promote only on held-out replay and task quality, not a fitted acceptance plot.

The September 9 production trace puts 68% of output tokens in long requests;
those average 3.83 tokens/round versus about 4.30 on short requests. Merely
raising 3.89 to 4.30 at unchanged round cost would improve decode throughput
10.5%. That is an illustrative payoff, not evidence that calibration can
achieve it. It is more promising than widening: K>7 already lost in the
retained controlled sweep, and K<7 lost in today's aggregate replay.

If calibration plateaus, target a drafter trained/calibrated on the actual
quantized target's agentic traces. That is a larger project with an acceptance
objective and a fixed target law, rather than another broad target-quant port.

## 3. Reduce the drafter vocabulary-head cost

q27's drafter computes all 248,320 vocabulary rows before selecting 16.
The fresh trace prices that head at 0.45 ms/round. ninfer supports an optional
131,072-row proposal head with an explicit mapping back to vocabulary IDs;
see its [DFlash2 specification](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/qwen3.8-27b-dflash2.md).
The retained campaign command did not request its optimized head, so this
capability should not be presented as the explanation for ninfer's result.

A compact, calibrated proposal-only head could roughly halve the head's
weight traffic. The optimistic gross saving is about 0.2 ms/round, or 1.1%,
before acceptance loss and mapping cost. It is a bounded engineering project,
but lower priority than proposal calibration or target sampling. Preserve
global token IDs for codebook access and rejection correction; never apply
the shortlist to the target's final sampling support.

The existing all-Q4 drafter already lost acceptance relative to Q8. Repeating
that downgrade without a selective quantization/calibration hypothesis has
little value.

## 4. Optimize task completion, with actual quality measurements

The SWE-bench campaign's 108 s versus 36 s coincided with roughly 18.2K versus
6.2K output tokens per instance. Its nonempty-diff and gold-file-hit counts
are not correctness tests. The newer Thunderdome medium pair emitted nearly
equal volume, 223.5K versus 227.8K, at 186.5 versus 191.8 tok/s. The large
trajectory gap is workload-dependent.

Effort and sampling are therefore legitimate product levers if the objective
is elapsed time at maintained task quality. Use a preselected task/seed set,
include failures and prematurely terminated sessions, and compare complete
task outcomes, output volume, turns and wall time together. The short T14
task and the longer T12 task should both be represented: medium already had
one fast but incomplete q27 T12 trial, whereas the retained xhigh trials were
stronger. Do not globally lower effort based on token count alone.

For scale, 10% fewer output tokens at otherwise unchanged behavior removes
about 9% of engine service time when decode is 90% of service. That exceeds
the entire removable sampler cost. It remains a quality/latency trade to
measure, not a kernel improvement or a reason to dismiss task wall time.

## 5. Reopen cache/concurrency work only for the traffic that needs it

Production persistence still caps snapshots at 65,536 tokens while serving
admits 262,144. Returning to a long conversation after interleaving can thus
restore an older frontier and redo a large suffix. Test 64K–128K interleaving
before pricing a larger cache limit, because pinned staging memory grows with
the cap. The reviewed September 9 trace never exceeded 63,907 prompt tokens,
so it cannot measure this risk.

DFlash2 remains single-slot. True concurrent agent/subagent workloads are
a separate opportunity, but the retained sequential SWE-bench campaign had
only about 2% accumulated queue wait and cannot justify a batching rewrite.
Measure actual overlapping requests and critical-path queue delay first.

q4s is also primarily a capacity candidate here: inspecting the two local
Qwen3.8 tensor tables found unchanged bulk tensor dtypes; q4s replaces the
Q8 output head and omits the separate Q4 head copy. Production verification
already uses a Q4 head. The first-token prefill path still uses `output.weight`,
so changing tiers is not automatically numerically neutral, nor an obvious
bulk decode speedup.

## What stays parked

Existing measurements already price wider K, forced prefill attention splits,
empty-ring reseeding, the TMA/W4A8 prefill port, and repeated cp.async verify
experiments as poor trades. Fold-last saves about 12 ms per eligible turn,
roughly 0.3% over the cited full campaign, while changing first-token
numerics. A generic host-sync rewrite targets a roughly 0.23 ms timed
remainder; it does not have a hidden multi-millisecond budget.

New local artifacts are under `build/perf-review-20260909/`: `run.py`,
the four replay JSONLs/server logs, `summary.jsonl`, `k7profile.nsys-rep`,
`kernels.csv`, and `probe_nucleus.py`/`gpu-nucleus.json`. The probe consumes
the retained `pfdump_old` files in the prior session scratchpad. Request
bodies and raw logits have not been copied into this document.
