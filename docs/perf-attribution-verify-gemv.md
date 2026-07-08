# Verify-GEMV Phase 0 attribution (ncu, 2026-07-08)

**Verdict: PROCEED to Task 2.** The batched verify GEMV (`k_gemv_q4_n<5>`) at 61K decode is
**latency-bound at ~40% of DRAM peak**, not at the weight roofline. The dominant stall is a
single class -- **long-scoreboard on L1TEX (90.4% of the 68.8-cycle inter-issue gap)** -- and
ncu flags the load access pattern (**10 of 32 bytes per sector used** on global loads), so the
Task 2 tactic is load-pattern coalescing/vectorization + latency-hiding, both bitwise-safe.

## Method

- Branch `verify-gemv` @ 7050173 (master 1709b93 + Task 0). 5090, stock memory.
- Ungated server (uniform width-5 verify): `Q27_PF_XG=32 ./build/q27-server <M> <T> --port
  8081 --ctx 131072 --no-think --fast-head`, one 61K request (`scratchpad/prompt61k.txt`,
  max_tokens=800).
- ncu (root, RmProfilingAdminOnly=1): `sudo -n env CUDA_VISIBLE_DEVICES=0 Q27_PF_XG=32
  /usr/local/cuda/bin/ncu --set full --graph-profiling=node -k "regex:k_gemv_q4_n"
  --launch-skip 2000 --launch-count 4 --kill 1 --target-processes all -o
  scratchpad/vg_gemv --force-overwrite <server cmd>` -- skip lands ~6.5 rounds into decode
  (305 q4_n launches/round), 4 launches profiled (4 distinct weight matrices), server
  self-kills after. Report: `scratchpad/vg_gemv.ncu-rep` (root-owned).
- Baseline context (Task 0, nsys): batched verify GEMV 12.38 ms/round, weight-stream total
  15.49 ms/round (53%) -- reproduces P14 Step 4 within 0.6%.

## Measured (4 launches, distinct matrices, width N=5)

| grid | dur (us) | DRAM % of peak | achieved occ % |
|---|---:|---:|---:|
| (640,1,1)x256  | 68.5 | 39.3 | 63.3 |
| (1280,1,1)x256 | 37.7 | 42.0 | 83.9 |
| (768,1,1)x256  | 25.7 | 39.0 | 72.1 |
| (2176,1,1)x256 | 57.8 | 46.5 | 89.3 |

Common counters (all 4 launches, tight range):

| metric | value | reading |
|---|---|---|
| DRAM throughput | **39-47% of 1.79 TB/s** (~690-830 GB/s) | far below the weight roofline |
| Memory (L1/L2) throughput | 54-62% | not saturated either |
| Compute (SM) throughput | 21-26% | not compute-bound |
| Issue slots busy | **10.7-13.2%**, IPC 0.44-0.57 | SMs starving |
| Warp cycles / issued inst | **64-72** | ~70 cycles stalled per issue |
| -- of which long-scoreboard (L1TEX dep) | **43.7-62.2 cyc = ~90%** | THE stall |
| Achieved occupancy | 63-89% (theoretical 100%) | occupancy is NOT the limiter |
| L1/TEX hit | 96.6-97.2% | per-column activation re-reads are L1-resident |
| L2 hit | 17-29% | weights stream from DRAM (as designed) |
| **Load sector utilization** | **10.0 / 32 bytes** | weight-read pattern wastes ~2/3 of each sector |
| Store sector utilization | 4.0 / 32 bytes | output writes scattered (small total traffic) |
| Registers | 40/thread; block limits: regs 6 / warps 6 | mild |
| ncu OPT estimate | **43.9% speedup** on the L1TEX stall | consistent with the 37% roofline gap |

## Interpretation

The plan's decision matrix, row 2, exactly: **achieved BW < 70% of peak with scoreboard
stalls dominating -> real headroom below the roofline, PROCEED.** The kernel is not
bandwidth-limited; it is **memory-LATENCY-limited with a wasteful access pattern**:

1. **10/32-byte load sectors.** If the weight loads produce partially-used sectors, DRAM
   moves up to ~3x the useful bytes -- this both inflates the apparent traffic and caps
   effective bandwidth. Fixing the access geometry (wider/coalesced per-thread reads,
   uint4-aligned weight walks) is the direct attack and is bitwise-safe (load pattern only,
   same per-output fp accumulation order).
2. **90% long-scoreboard at 63-89% occupancy** means in-flight memory per warp is too low:
   each warp issues a load and sits ~62 cycles. More loads in flight per warp (wider loads,
   manual unroll/prefetch of the next weight chunk) hides the latency without touching the
   reduction structure.

Both tactics are Task 2 (Phase 1, bitwise-safe). The dp4a issue rate is NOT the limiter
(issue 11-13%), so the tensor-core rewrite (Task 3) is unlikely to be justified -- re-evaluate
only if Task 2 stalls out well short of the roofline.

**Honest ceiling reminder (plan):** weight-stream floor ~9.7 ms/round vs 15.4 today -> max
~1.24x decode if fully captured. Task 2 keep-bar: >=2% end-to-end decode t/s (full engine,
n=3, canonical bitwise).

## Task 2 result (Phase 1, bitwise-safe) -- +5.9% decode @61K, KEEP

Two applications of one change: activation reads 4x uint2 -> 2x uint4 (same bytes, same
component order into the same dp4a sequence; integer-exact, fp acc order untouched ->
greedy bitwise BY CONSTRUCTION, canonical held exactly both times). No smem, no geometry
change, registers flat -- deliberately inside the -4%-smem-lesson constraints.

| step | 61K gated decode (n=3 median) | delta |
|---|---|---|
| baseline (Task 0) | 163.2 t/s (dec_ms 4901) | -- |
| `k_gemv_q4_n` (batched verify) | **172.2 t/s** (4645) | **+5.5%** |
| + `k_gemv_q4` (single/draft) | **172.9 t/s** (4627) | +0.4% (noise-level; bounded +0.7% by the 1.6 ms/round share; kept for pattern consistency) |

Same 173 rounds every config (engine-level bitwise confirmation). Captures ~1.6 ms/round
of the ~5.7 ms/round roofline gap (~28%). `k_gemv_q8_n` untouched (already uint4 loads).

## Plan verdict after Task 2

**Task 3 (tensor-core verify, canonical-breaking): NOT JUSTIFIED.** Phase 0 showed dp4a
issue was never the limiter (issue 11-13%); Phase 1 captured the cheap latency win. The
residual ~4 ms/round gap is L1TEX-latency structure (per-column re-reads at 63-89%
occupancy), and the remaining bitwise-safe levers (prefetch depth, register-pressure
tuning) are in the marginal band with occupancy risk -- the fd2/smem precedents say stop
here. Re-open only with a new fact (e.g. a future activation layout that batches columns
contiguously, making the x-reads one coalesced stream).

## Commands

Every number above: the Method commands + `ncu --import scratchpad/vg_gemv.ncu-rep --page
details`. Baseline harness: `scratchpad/prep_tokens.py` (fixtures), `bench61k.py` (session
scratchpad; 3x 61K requests, tps= from req_log).
