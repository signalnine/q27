# gpt-6-astra review: host-sync lever re-measurement (2026-09-07)

Second opinion on the re-measurement that killed the attribution doc's lever #1
("per-round host sync ~2.2 ms"). Brief + questions: the session's
hostsync_brief.md; the review ran adversarial-first as requested.

## Verdict on the method (Q1): holes exist, one validation closes them

- The nsys node-trace subtraction assumes profiled kernel duration == clean
  kernel duration; a 3% drift doubles the residual. The -n 2 baseline cancels
  common WORK, not measured DURATION (clock/thermal/cache/instrumentation
  drift); 58 ms of uncancelled startup kernels across 108 rounds equals the
  whole claimed residual. Population mismatch: kernel estimate averages rounds
  2..109, clean wall averages all 109.
- Device-side memcpy time misses submission/staging/synchronization-return
  cost; pageable cudaMemcpyAsync can block.
- The CLI "wall" is a loop-elapsed interval (host-created GPU idle included) --
  fine, but name it that.
- CHEAPEST VALIDATION (adopted, ran same evening): graph-LEVEL nsys trace
  (near-zero overhead), inspect device gaps between graph replays directly.
  RESULT: profiled wall 16.04 vs clean 15.92 ms/round (instrument ~transparent);
  GPU busy union 1528.6 ms of a 1550 ms decode window = 1.4% idle =
  ~0.22 ms/round. The re-measurement HOLDS; subtraction had slightly
  OVERSTATED the gap (0.54 vs ~0.2-0.35 true).
- Bonus: the attribution doc's "capture-range broken on nsys 2025.6.3" claim is
  wrong-footed -- cudaProfilerStart/Stop live in the serving generate path
  (engine.cuh ~4386, env-gated Q27_PROF_DECODE) and the CLI spec loop never
  calls them; an empty CLI capture needs no profiler bug.

## Verdict on the DEXIT A/B (Q2): net win, not a bound

W_off - W_on = GPU work saved - extra sync cost = 0.57 ms; it does not bound
the sync cost without independently estimating the GPU savings. Config trap
found: auto7 allows depth 7 with dexit on but clamps to 5 with it off --
resolved for THIS comparison (both runs stayed depth-4, identical tokens/
histograms) but a confound for broader A/Bs. Keeping DEXIT on is supported;
pricing a conditional-graph replacement at 0.3-0.4 ms is not yet supported
(now bounded above by the 1.4%-idle graph-level result anyway).

## Verdict on the re-ranking (Q3): measure before rewrite -- agreed order

1. Matched serving attribution for NINFER and q27 (same traffic, same
   instrument) BEFORE any implementation.
2. Targeted attention measurements -- with two corrections: (a) width_bench /
   attn_fdw_bench re-run 50 calls against the SAME KV allocation; 56 MiB at
   28K fits the 5090's L2, so inter-call cache retention can flatter the
   slope vs real inference (weight reads evict between attention calls);
   measure widths 1..8 under production-like cache conditions before
   multiplying per-layer numbers by 16. (b) collect DRAM read bytes / L2
   hit rates and normalize against one cold KV sweep (2 x ctx x 4 x 256 B
   per fp8 layer) -- near 1.0x means reuse already works, near W x means
   duplication remains.
3. Retire "ninfer reads the same 17 GB so the gemv can't differ": equal bytes
   is a common lower bound, not equal achieved time. Keep the hypothesis open
   until ninfer's round is decomposed directly.

Also confirmed from source: fd2 already reuses each loaded K/V vector across
the 6 GQA heads and splits context 128 ways; what it lacks is explicit
cross-lane sharing (token-first grid ordering encourages L2 reuse only). The
fdmma leg (live default on sm_89+) stages KV through smem once for all lanes.

Raw transcript: session scratchpad codex_hostsync_review.txt.
