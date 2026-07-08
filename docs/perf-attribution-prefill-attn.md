# Prefill-attention Phase 0 attribution (ncu, 2026-07-07)

**Verdict: PROCEED.** `k_attn_prefill_mma` at deep context is **latency/occupancy-starved**,
not bandwidth- or tensor-bound. The FLOP-derived "~39% of peak tensor" estimate is
ground-truthed at **33%** (right ballpark, slightly high). One material correction to the
plan/review: the occupancy limiter is **dual (registers AND shared memory)**, so a "2
resident CTAs" play needs a register cut too, not just a smem shrink.

## Method

- HEAD `623cdb1` on branch `prefill-attn`, existing binary (canonical `4c4120c7...` verified,
  no rebuild). 5090 free (vox on the 3090). Fixture `scratchpad/synthtoks.bin` = 140k random
  int32 < VOCAB (prefill timing is value-independent).
- Baseline wall (no profiler): **128K prefill 75.45 s / 1737 t/s** -- in the P14 band.
- ncu 2026.1, `sudo -n` (RmProfilingAdminOnly=1). `-k k_attn_prefill_mma --launch-skip 1900
  --launch-count 3` -> profiled 3 launches at **chunk ~118, base ~121K** (deep, quadratic
  regime). PF_T=1024 -> 16 attn layers x 128 chunks = **2048 total launches** (the first ncu
  run's `--launch-skip 6000` overshot this and profiled nothing -- fixed).
- Report: `scratchpad/pf_attn_128k.ncu-rep`. All 3 launches identical (~44.8 ms, grid
  (4,64,8)=2048 blocks, nsplit=8). Numbers below are the deepest launch.

## Measured (deepest launch, base ~121K, 44.78 ms)

| Speed-of-Light | value | reading |
|---|---:|---|
| Compute (SM) throughput | **33.2%** | not compute-bound |
| Memory throughput | 23.5% | not memory-bound |
| **DRAM throughput** | **1.98%** (35 GB/s of 1790) | **nowhere near bandwidth** |
| L1/TEX hit rate | 87.4% | KV staged, well-reused |
| **L2 hit rate** | **95.6%** | at depth the KV is **L2-resident, not re-streamed from DRAM** |
| Tensor pipe active | 33-35% | ncu: "well-utilized, **should not be a bottleneck**" |
| Issued IPC active | **0.42** | SMs mostly idle |
| Issue slots busy | 9.9% | ~90% of issue slots empty |
| Warp cycles / issued inst | **14.23** | 14 cycles of stall per issue |

| Occupancy | value |
|---|---:|
| Theoretical / Achieved | **12.5% / 12.49%** (no scheduling loss) |
| Active warps / SM | **6.00** (= 1 CTA x 6 warps) |
| Registers / thread | **248** |
| Dynamic smem / block | **84.48 KB** (static 0) |
| **Block limit -- registers** | **1** |
| **Block limit -- shared mem** | **1** |
| Block limit -- warps / barriers / SM | 8 / 24 / 24 |

**Both registers (248) and smem (84.48 KB) independently cap the kernel to 1 CTA/SM.** On
sm_120 the per-SM smem carveout admits one 84.48 KB block (2x would need <=~50 KB), and
65536 regs/SM / (192 threads x 248) = 1.37 -> 1 block. So occupancy is 12.5% and **cutting
smem alone will not raise it** -- registers co-bind.

### Stall breakdown (cycles per issued instruction, sums to ~14.2)

| stall reason | cyc | share | fixed by |
|---|---:|---:|---|
| long_scoreboard (global/L2 load latency) | 4.24 | 30% | **cp.async prefetch** (Phase 1) |
| math_pipe_throttle (bursty MMA issue) | 3.96 | 28% | fp8 MMA (fewer/fatter ops) + occupancy |
| barrier (`__syncthreads`) | 2.13 | 15% | occupancy / deeper pipeline |
| wait (instruction dependency) | 1.99 | 14% | occupancy |
| selected (actual issue) | 1.00 | -- | -- |
| short_scoreboard (smem latency) | 0.34 | 2% | -- |

No single stall dominates -- the signature of **occupancy starvation**: with only 6 warps,
every latency source (L2 loads, MMA bursts, barriers, dependencies) surfaces as a stall
because there aren't enough warps to interleave.

## Interpretation

The P4 split fix already solved the DRAM problem: at 121K the kernel is 95.6% L2-hit and
1.98% DRAM. The "quadratic wall" at depth is **not** re-streaming memory -- it is that each
launch does ~O(N) more MMA+staging work while the SMs sit ~67% idle (SM busy 33%, IPC 0.42)
because 6 warps/SM cannot hide the pipeline latency. The kernel has huge headroom (33% ->
toward roofline), gated entirely by occupancy and latency-hiding, exactly the class the
prefill-attn plan targets.

## Verdict for the plan levers

**PROCEED. Plan ordering validated, with one added constraint.**

1. **Phase 1 (cp.async double-buffer) -- first, bitwise-safe.** Directly attacks the largest
   single stall (long_scoreboard, 30%) by prefetching the next tile's K/V while the current
   tile's MMAs run, raising IPC within the existing 6 warps -- no occupancy required. Best
   ROI, lowest risk. The 95.6% L2-hit means the prefetch overlaps L2 latency (~200 cyc), not
   DRAM, but that latency is exactly what the 6 warps fail to hide.
2. **Phase 2 (fp8 MMA) -- well-motivated on two counts now.** fp8 m16n8k32 halves the QK^T MMA
   instruction count (attacks math_pipe_throttle, 28%) AND halves K/V staging smem -- a step
   toward 2 CTAs/SM. Still tolerance-gated.
3. **NEW -- the "2 resident CTAs" goal (review item 4 / Phase 2) needs a REGISTER cut, not
   just smem.** Registers (248/thread) co-limit to 1 block independent of smem. The
   `o[32][4]` output accumulator is 128 registers/thread on its own -- restructuring it (smem
   spill of O, or a narrower tile) is the prerequisite for occupancy >12.5%. Shrinking smem
   without cutting registers stays at 1 CTA/SM and buys nothing on occupancy. This is the
   correction to the plan's Task 2/Phase-2 smem framing.

**Re-ranking:** Phase 1 (cp.async) unchanged as the first move. Task 4 (longer tiles) and any
occupancy-doubling must be paired with the register cut -- fold that into Phase 2 rather than
treating smem as the sole lever. The maximum from occupancy alone (12.5% -> 25%, 2 CTAs) is
bounded and expensive; the cp.async latency-hiding + fp8 MMA path captures the stall profile
more directly and is where the plan should spend first.

## Caveats / follow-ups

- All 3 profiled launches were at one depth (~121K); occupancy/stall structure is
  depth-independent, but the tensor-% and stall *mix* shift with the p0-loop length -- P14
  Step 7 already has the 16K-vs-128K time sweep. A mid-depth (~16K) ncu point would confirm
  the stall mix at the crossover if Phase 1 tuning needs it.
- Report kept at `scratchpad/pf_attn_128k.ncu-rep` (root-owned). Regenerate with the Task-1
  command in the plan if fixtures are re-cleaned.

## Phase 1 result (cp.async K/V prefetch) -- +5.4% (CORRECTED; first pass was a test artifact)

cp.async double-buffered prefetch of the next PP-tile's raw fp8 K/V (fp8 path;
`Q27_PF_CPASYNC`, default on), convert-on-consume. Bitwise-identical (canonical 4c4120c7;
fp8-path greedy A/B on==off).

**Correction:** the first measurement used fp16 KV (default; `--kvstats` forbids fp8), where
cp.async is dead code (`CPA = sizeof(CT)==1` false) -- so the "76.30 vs 76.40 = +0.2%
neutral" was the fp16 blocking path measured against itself, not cp.async. cp.async never
executed.

Redone on the real fp8 path (`Q27_KV=fp8`, `--pf 131072 --ctx 133120`, `Q27_PF_NOSERIAL=1`):

| config | 128K prefill wall |
|---|---|
| blocking (no cp.async) | 72.10 s (1818 t/s) |
| **cp.async** | **68.20 s (1922 t/s)** |

**cp.async = +5.4% on the fp8 prefill wall (~+10% on the attention kernel, which is ~54% of
prefill).** Common-mode-clean (the d_gen OOB, since fixed, was identical across configs).
Phase 1 is a KEEP. The 95.6%-L2-hit occupancy analysis still holds -- cp.async helps because
it overlaps the L2 *load* latency (long_scoreboard 30%) with the MMAs even at 6 warps; the
earlier "L2 not DRAM so cp.async won't help" reasoning was too pessimistic. Bench lesson:
always `Q27_KV=fp8` for this path; `--kvstats` is fp16-only.

## Phase 2 result (fp8 QK^T MMA, `Q27_PF_FP8MMA`) -- +11.8% @128K

fp8 e4m3 QK^T (`mma.sync.m16n8k32.e4m3.e4m3.f32`) consuming the fp8 KV cache directly,
via the smem relayout the ec1a54c revert prescribed: Q staged as e4m3 (s_q 50.7->25.3KB),
s_k dropped (K read raw from a **double-buffered** s_kraw so cp.async prefetch and
fp8-direct-read coexist), PV stays fp16 (V still converted to s_v). New separate kernel
`k_attn_prefill_mma_fp8q`, opt-in; the default fp16/fp8 kernels are untouched (canonical
4c4120c7 holds -- it is a distinct compiled kernel).

**Bank-conflict padding was load-bearing.** The fp8 QK^T packs 4 consecutive e4m3 into a
uint32; from contiguous s_q/s_kraw the gid dimension gives an 8-way smem bank conflict (the
f16 path dodges it via padded LDH=264). Padding s_q to LDQ=260 and s_kraw to LDK=272
(+~1.4KB smem; total ~66KB, under the 99KB cap) lifted the win from +4.9% to +11.8%.

| 128K prefill (`Q27_KV=fp8 --pf 131072 --ctx 133120 NOSERIAL`) | wall | t/s |
|---|---|---|
| default fp8 (f16-MMA) | 68.27s | 1920 |
| fp8q unpadded | 64.57s | 2030 (+4.9%) |
| **fp8q padded** | **60.20s** | **2177 (+11.8%)** |

+11.8% end-to-end 128K ~= **+22% on the attention kernel** (~54% of prefill) -- clears the
plan's >=10% attn keep-bar. The fp8 MMA attacks the measured 28% `math_pipe_throttle`
stall at the unchanged 12.5% occupancy, exactly the Phase-0 prediction.

**Correctness (greedy/identity gates).** `--nll` prefills per-token via `step_with` (decode
attention), NOT the batched prefill kernel, so fp8q is validated through the batched-prefill
routes:
- canonical 4c4120c7 unchanged (decode + default prefill paths bit-identical)
- serial-vs-batched continuation **IDENTICAL @ pf=512 and pf=4096** (fp8q batched prefill vs
  the serial per-token reference -- the fp8 QK^T never flips the greedy argmax at depth)
- `--pfcache` warm-vs-cold **IDENTICAL**, and mid-divergence checkpoint-restore **IDENTICAL**,
  under fp8q

**Deep logit A/B (default-on quality gate) -- PASS.** Wired `--dump-logits` into the `--pf`
batched leg (dumps the post-prefill position-N logits, the only route that hits this kernel;
`--nll` is per-token). fp8q vs default-fp8 (f16-MMA) at **position 131072** over a 131072-token
varied prompt (101,863 distinct ids from synthtoks):

| metric | fp8q vs default @128K | P2 fp8-KV reference |
|---|---|---|
| cosine | **0.9999827** | 0.9995 |
| max\|dlogit\| | 0.100 | -- |
| KL(def\|\|fp8q) | 1.9e-4 | 3.4e-5 |
| argmax | **MATCH** (95726) | -- |
| top-5 | **5/5** | -- |

Tighter cosine than the shipped fp8 KV cache; identical greedy top-1 and top-5 at max depth.
The fp8 QK^T adds essentially no quality delta even at 128K -- the "silent quality loss on
deep prompts" risk is measured and absent. Repro: `Q27_KV=fp8 [Q27_PF_FP8MMA=1]
Q27_PF_NOSERIAL=1 ./build/q27 <model> --tokens-file scratchpad/pf_toks.txt --pf 131072
--ctx 133120 --dump-logits <out>`, then scratchpad/cmp_logits.py.

**Opt-in KEEP; quality-gated for default-on.** The one gate not yet run is a needle-retrieval
sweep (retrieval is more sensitive than a single next-token logit) -- the logit A/B bounds
per-position drift, a needle would confirm depth-placed recall. Cheap now that the dump path
exists.
