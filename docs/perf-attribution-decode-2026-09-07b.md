# Decode attribution, part 2: the levers re-measured, ninfer decomposed (2026-09-07 evening)

Follow-up to `perf-attribution-decode-2026-09-07.md`. That doc's ranked levers
were re-measured the same day; neither survives in its stated form, and a
direct decomposition of ninfer's decode replaces the inferred gap story.
gpt-6-astra reviewed the method adversarially
(`docs/reviews/2026-09-07-gpt6astra-hostsync-remeasure.md`); its cheapest-
validation suggestion (graph-level trace) is what settles Q1 below.

## 1. Lever #1 "per-round host sync ~2.2 ms" -- retired (instrument artifact)

Gated ladder CLI (fp8, Q27_PMIN=0.5, Q27_MAXD=auto7, code-write, -n 256,
109 rounds, deterministic):

| measurement | ms/round |
|---|--:|
| clean wall (no profiler) | 15.92 |
| nsys node-trace wall | 17.19 (instrument +1.27) |
| kernel time (node-trace subtraction -n 256 minus -n 2) | 15.38 |
| graph-level-trace wall | 16.04 (instrument +0.12) |
| GPU idle, graph-level, decode window | **1.4% = ~0.22** |

The part-1 doc's "wall 18.8 / ~1.5 ms host+graph gaps" was read off the
node-trace-profiled run: ~70%+ of that gap was nsys per-launch overhead.
Sampled path: same split (15.97 wall / 15.45 kernel). Device-side memcpys in
the decode window: ~4 us/round (6 copies) -- the pageable 4 B margin reads and
40 B outcome read do not register. (They remain the bit-flip-exposed pageable
path per the 5090 corruption finding -- pinning them is hygiene, not perf.)

DEXIT A/B (same prompt, identical tokens, both stayed depth-4): dexit=1
15.92 vs dexit=0 (monolithic draft, one margin sync) 16.49 ms/round. Early
exit's GPU savings beat its per-step sync cost; with total idle at 1.4%, a
conditional-graph round restructure is bounded at ~0.2-0.3 ms (~1.5%). PARKED.
(Review caveats: the A/B is a net-win proof, not a sync-cost bound, and auto7
clamps to 5 under dexit=0 -- benign here, both runs sat at depth 4.)

## 2. Lever #2 "fd2 attention slope" -- measured on the wrong leg

Live serving (cc profile, sm_89+) runs `Q27_FD=mma`. The fdmma kernel streams
the lanes' KV union ONCE through smem per (split, kv-head) block -- the
W-times KV re-read that produces fd2's slope is exactly what fdmma removed
(docs/plans/2026-07-10-fdmma-verify-attn.md). attn_fdw_bench today (per
layer; x16 for a round): mma 92 us at 26K/W4, 215 us at 61K/W8 (vs fd2
202/803). The engine's auto split count (SMs*2/kv_heads = 85) is already the
sweep optimum. Headroom vs one cold KV sweep: ~1 ms/round at 26K, ~2.3 at 61K
-- and the review flags that these microbench numbers may be L2-flattered
(56 MiB KV at 28K fits the 5090's L2; the bench re-reads one allocation 50x),
so treat headroom as an upper bound pending counter-based measurement.

## 3. ninfer decomposed directly (no more inference from our side)

ninfer master, released Qwen3.8-27B-nvfp4 artifact (DFlash2 bound in),
`--spec dflash2 --draft-tokens 7 --kv-dtype int8 --kv-capacity 131072`, nsys
node-trace, arm-A traffic (think-on, sampled, same instrument as
bench/crossengine). The 26K decode window is exactly req#5's 153 rounds:

| bucket | ms/round | calls/round |
|---|--:|--:|
| nvfp4_w4a4_mma (+ its act-quant 0.18) | 5.77 | 112 |
| fp8_small_t | 5.25 | 120 |
| fp8_mma | 1.74 | 24 |
| w8_small_t_mma | 1.17 | 21 |
| causal_attention_small_t_i8_tiled (+reduce 0.08) | 1.00 | 16 |
| head + ksplit_topk | 1.52 | 2 |
| GDN-ish (record/gating/fold/conv) | 1.05 | ~60 |
| norms + misc | ~0.7 | ~180 |
| **kernel total** | **18.2** | ~815 |

Their measured round: 18.8 ms under trace => host+gap ~0.6 ms/round (3%) --
same shape as ours. Their own clean telemetry reports host 0.1-0.4%. There is
no engine-side magic: weight matmuls dominate (14.7 ms/round at width 8,
NVFP4 W4A4 tensor-core + fp8 legs), attention 1.0 ms at 26K, GDN ~1.

## 4. Where the standing actually is (same instrument, arm A think-on)

Fresh q27 ladder row, TODAY's build (post GDN register/tile), vs the 09-06
ninfer rows (bench/crossengine/rebench-2026-09-06):

| ctx | q27 ladder | ninfer DFlash2 | ninfer MTP3 |
|--:|--:|--:|--:|
| ~100 | 156.2 | 221.4 | 157.3 |
| 3.3K | 179.9 | 223.9 | 146.3 |
| 6.5K | 170.8 | 216.1 | 155.4 |
| 12.5K | 181.9 | 203.3 | 147.7 |
| 26K | 175.9 | 186.0 | 135.9 |
| 50K | 163.2 | 181.0 | 133.3 |

ROUND WALLS ARE EQUAL: q27 17.85 ms clean at 12.5K vs ninfer 18.4 under-trace
(same-day driver, their reqlog rounds); 18.6 vs 18.8 at 26K. The entire
ninfer-DFlash2 lead is tok/round: 3.51 vs 3.24 at 12.5K (+8% == their mid-ctx
t/s edge), widening at short ctx (~3.5 vs ~2.5 == the 221-vs-156 gap).

Per-lane accept decay MATCHES for lanes 1-4 (q27 0.80/0.64/0.48/0.38, ninfer
0.79/0.63/0.51/0.31); their margin comes from positions 5-7 (+0.26 tok/round),
which they draft for free (one DFlash2 forward drafts all 7 every round).

## 5. The actual defect found: the sampled ladder is capped at depth 4

Live serving always samples (temp 1.0), and the sampled spec path is
hard-capped at depth-4 draft / width-5 verify (`engine.cuh` ~702, ~2664
"the sampled tail is always depth-4", `spec_sample_round` md_used=4). The
auto4..7 ladder exists ONLY for greedy rounds. Confirmed live: the sweep's
[req] gnh histograms have zero mass above n=5. So production q27 never drafts
past 4 while ninfer drafts 7 on identical traffic. The greedy promote bar
(sat >= 0.50) is also stale -- tuned when a width-8 round cost +3 ms; on this
traffic sat[4] ~= 0.24 so even greedy sits at 4, yet ninfer's position data
shows lanes 5-7 still pay at today's width-flat verify cost.

## Re-ranked levers

1. **Extend the sampled ladder to depth gate_maxd** (sampled per-width verify
   captures W=6..8, widen the sampled outcome layout, md_used from gate_maxd,
   seeded-determinism + parity gates). Mirrors shipped greedy P12b/P14 work.
   Expected +3-6% live t/s; removes a structural cap on the DEFAULT serving
   path.
2. **Sampled DFlash2 verify** (rejection-sample over the dflash2 draft) --
   the drafter whose depth is free; q27's is greedy-only today
   (`d2_on && !t.sampling`), so it cannot field its best drafter exactly where
   ninfer wins. Bigger lift, bigger win at short ctx.
3. **Retune depthctl promote bars** for the flattened width cost (greedy
   side; cheap once (1) lands and re-measures the width-cost curve).
4. Attention mma headroom at long ctx -- only after counter-based (not
   L2-flattered) measurement; upper bound ~1 ms at 26K.
5. Host-side conditional-graph restructure: bounded at ~0.2-0.3 ms. Dead
   unless something else resurrects it.

## Tool notes

- nsys node-trace ADDS ~1.3 ms/round of host overhead on this workload --
  never read walls off it; graph-level trace is near-transparent (+0.12) and
  shows inter-graph gaps directly.
- The part-1 "capture-range broken on nsys 2025.6.3" note is unproven: the
  cudaProfilerStart/Stop bracket is env-gated (Q27_PROF_DECODE=1) AND lives on
  the serving generate path; the CLI spec loop never calls it. An empty CLI
  capture needs no profiler bug.
- ninfer-serve rig: defaults come up KV 8,192/bf16 -- pass
  `--kv-dtype int8 --kv-capacity 131072 --max-context 131072` to match the
  rebench config; `--request-log-jsonl` emits per-request rounds +
  accepted_per_position (the AL profile used above).
- Fresh q27 row: bench/crossengine/rebench-2026-09-06/xe_q27lad_0907.jsonl.
