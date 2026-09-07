# Decode round-wall attribution (2026-09-07)

Measurement pass scoped by gpt-6-astra (codex) for the "close the ninfer gap"
work. Goal: find where the decode round wall's time actually goes before touching
a kernel. Method and every number below are reproducible; scope limits are stated
inline.

## Method

- Shipping decode path (MTP ladder, `--spec --fast-head`, `Q27_KV=fp8`), CLI
  `build/q27`, model `qwen38-27b-mtp.q27`. The width-8/width-5 verify GEMV is
  shared between the ladder and dflash2, so this attributes both.
- Per-kernel decode isolation by SUBTRACTION: `nsys --cuda-graph-trace=node`
  trace at `-n 256` minus a `-n 2` baseline, divided by the round delta. Prefill
  and one-time graph capture are identical in both runs and cancel exactly.
  - `--cuda-graph-trace=node` is REQUIRED: without it, graph replays collapse to
    a single node and decode kernels are invisible (a plain trace subtraction
    returned 0.07 ms/round against an 18.8 ms wall).
  - `--capture-range=cudaProfilerApi` is BROKEN on nsys 2025.6.3 (empty reports
    regardless of flags). The BUILDLOG idiom worked on an older nsys. Subtraction
    is the workaround.
- Isolated kernel SOL via `ncu` (2026.1.0) on `build/width_bench`. ncu counters
  need elevated perms here (`ERR_NVGPUCTRPERM`); `sudo -n ncu ...` works. This
  corrects the stale "ncu not on PATH / unusable" gotcha: ncu is installed at
  `/opt/nvidia/nsight-compute/2026.1.0/ncu` and usable under sudo.

## 8K decode attribution (ground truth, nsys node-trace subtraction, 70 rounds)

Total decode kernel time: **17.32 ms/round** (wall 18.8 ms/round; ~1.5 ms is
host/graph gaps).

| bucket | ms/round | % |
|---|---:|---:|
| GEMV (quant weight-read) | 12.71 | 73.4 |
| attention (fd2/full) | 1.73 | 10.0 |
| GDN recurrence | 1.22 | 7.0 |
| norms | 0.76 | 4.4 |
| act-quant | 0.23 | 1.3 |
| misc elementwise/kv | 0.31 | 1.8 |
| rest | ~0.4 | ~2 |

Dominant single kernel: **`k_gemv_q4_n<5>` = 9.68 ms/round (56%)** (the ladder
verifies at width ~5, not 8; width 8 was the dflash2 K=7 framing).

## ncu: the dominant GEMV is memory-bound and register-trapped

`k_gemv_q4_n<5>`, isolated FFN matrix 17408x5120 Q4 (~44.6 MB), width 5:

- **DRAM Throughput 72%** (1.27 TB/s), Compute (SM) 35%. Limiter: DRAM.
- **70.6% of cycles stall on long-scoreboard** (waiting on memory loads).
- Eligible warps/scheduler **0.40** (75% of cycles issue nothing).
- **Achieved occupancy 45%, theoretical 50%, capped by 80 reg/thread**
  (Block Limit Registers = 3).

Read: the 28% gap to peak bandwidth is memory latency that low occupancy fails to
hide, and occupancy is register-limited. The 80 registers ARE the WxRow FP32
accumulators, which the bitwise-verify contract freezes. Codex lever #1
(prefetch) needs MORE registers -> block limit 3->2 -> occupancy falls -> likely
net-negative. Lever #2 (drop registers for occupancy) is blocked by the numerics
contract. The two obvious levers fight, and the safe one is contract-blocked.

Q8 head GEMV (`k_gemv_q8_n<5>`, 248320x5120): **DRAM 90% = near floor.** Not a
lever.

## Attention context slope (width_bench fd2, fp8 KV, x16 layers)

| context | fd2 ms/round |
|---|---:|
| ~8K (nsys) | 1.7 |
| 28K | 3.92 |
| ~48K (interp) | ~6.5 |
| 61K | 8.38 |

Wall confirms: 8K 18.8 ms/round -> 48K 24.4 ms/round (+5.6 ms). The context slope
is almost entirely attention. So the decode mix shifts from GEMV 73% / attn 10%
at 8K to roughly GEMV ~52% / attn ~27% at 48K (GEMV is context-independent weight
read; attention grows with KV).

(Scope limit: the 48K per-kernel nsys subtraction was abandoned - the 2.2 GB
node-trace reps produce 8-15 GB sqlites and risked filling root. The 48K split
above is derived from the isolated width_bench fd2 sweep + the wall delta, not a
direct 48K nsys subtraction.)

## Verdict on the GEMV rewrite: effectively NO-GO for the ninfer gap

1. The dominant GEMV is 72% DRAM-bound and register-trapped. Best realistic
   capture (72%->~82% via occupancy) is ~1.4 ms/round, but the path is blocked by
   the bitwise contract and the two levers fight. High risk, not a quick win.
2. The Q8 head is already at 90% DRAM. No headroom.
3. **ninfer reads the same 17 GB of weights**, so it cannot have a materially
   faster verify GEMV - it hits the same memory floor. Acceptance already matches
   (4.86 vs 4.87 tok/round). Therefore ninfer's +16% is NOT the GEMV.

The ninfer gap (our round ~21.2 ms vs their ~18.55 ms) is the **~2.2 ms/round host
overhead** and the round structure, plus the attention slope at long context -
NOT the verify GEMV, which is near the hardware memory floor for both engines.

## Recommended next levers (ranked)

1. **Per-round host sync (~2.2 ms/round).** This is the largest tractable
   difference from ninfer and touches every decode path. Profile the host side
   of the round (the D2H outcome transfer + ingest + fold submission that the
   verify graph excludes).
2. **fd2 attention at long context.** Grows to ~27% at 48K; more structural
   freedom than the register-trapped GEMV (KV reuse across the verify tokens, L2
   grid ordering). The right lever for the 20-60K contexts real CC runs.
3. **GEMV occupancy** only if a register-neutral, bitwise-exact decomposition is
   found. Biggest single kernel, but contract-blocked today.
