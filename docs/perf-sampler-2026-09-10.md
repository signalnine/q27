# CUDA sampler filter order and small-top-k optimization

The CUDA sampler now applies top-k, top-p, then min-p to temperature-scaled
weights. Previously min-p reduced the denominator used by top-p, which could
remove an additional boundary token. This is enabled in the default sampler
path; no environment variable or launch flag is required.

For example, with weights `[1, .1, .055, .04]`, k=4, p=.95, and min-p=.05,
top-p needs the first three tokens. Applying min-p before top-p incorrectly
leaves only two. The regression checks both the surviving support and its
renormalized probabilities.

## Implementation

- Stop the top-k bisection as soon as the retained support has exactly k tokens.
- For k<=32, compact the survivors into shared memory and sort their values.
  Compute top-p on that small support, then apply min-p.
- Use the existing general bisection path for larger supports or compact-buffer
  overflow from ties.
- Continue exporting the full-softmax log normalizer and the final retained
  mass, so speculative rejection uses probabilities for the actual served law.

The single-lane and multi-lane entry points share the implementation. There
are no new allocations, sampler parameters, or graph-capture requirements.
Top-k retains the existing logit-threshold representation: exact ties across
its boundary cannot be split by token ID. This change does not redefine that
separate limitation. Temperature-last filtering is equivalent at the serving
T=1; equivalence at other temperatures is not claimed.

## Measurements

The original RTX 5090 experiment compared the optimized kernel with a
reference that already had the filter-order correction:

| Measurement | Corrected reference | Optimized |
|---|---:|---:|
| Eight-lane nucleus, CUDA-event median | 0.7036 ms | 0.2335 ms |
| Mean decode round, 15-request replay | 18.040 ms | 17.589 ms |

Fresh-server/cache replay in reference/optimized/optimized/reference order
produced identical hashes for all 15 requests, with 5067 output tokens and
1214 rounds in each run. Combined decode time fell 2.50%. This is a decode
improvement; total agent wall time also includes prefill and tool execution.
Both corrected and optimized kernels matched the intended support on all
36 saved production logit rows.

## Integration validation

This sampler-only port is based on master `bd81f73`, including incremental KV
entitlements. It carries the sampler changes and their tests from the earlier
performance branch, without changing proposal-temperature or head selection.

- Standard `make -j2 build/test_sampling build/test_kernels build/q27-server`
  succeeds for the default sm_86, sm_89, and sm_120 targets.
- CPU sampling tests pass.
- `build/test_kernels --sampling-only` passes on RTX 3090 and RTX 5090. This
  includes filter-order support/probability checks, a ragged 248321-token
  vocabulary, k=20/32/64, single/multi-lane identity, and speculative rejection.
- The integrated server reproduces all 15 output hashes from the original
  optimized sampler replay, with the same 5067 tokens and 1214 rounds.
  Production was restored after this isolated integration check.

Experiment logs and replay recordings are retained under
`build/perf-sampler-20260909/` in the original q27 checkout. The earlier
performance measurements are documented on `codex/perf-sampler-20260909` in
`docs/perf-implementation-2026-09-09.md`.
