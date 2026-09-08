# Prefill attack plan (2026-09-08)

Status: PLAN, ready to execute from a fresh context. Evidence and numbers
are in docs/perf-attribution-prefill-2026-09-08.md (the recon); this file is
the executable part. Written so a session with no memory of the recon can
run it: every phase has the launch command, the instrument, the bar, and
the traps.

## 0. State at hand-off (2026-09-08 evening)

- Production q27-38 runs the DFlash2 Q8 config with NO prefix-cache flags
  (tools/launch_q27_38.sh mode `d2`). Baseline on the 12 pinned SWE-bench
  instances at Claude Code's default effort (q27 renders xhigh):
  bench/crossengine/agentic-2026-09-08/prodd2-xhigh.req.txt -- 299 requests,
  prefill wall 255 s, decode wall 1640 s, 201.0 t/s decode aggregate,
  6 eviction full misses (80 s), 12 cold first turns of 20-25 K (80 s).
- Everything committed on master (recon 3a67fc5, this plan + launch
  script + pf_misses.py in the following commit). No branch, no worktree.
- Instruments in the repo: bench/crossengine/agentic-2026-09-08/pf_agg.py
  (prefill profile by size bucket), pf_misses.py (miss anatomy by
  conversation), lanes_agg.py (decode per-lane), tools/cublaslt_peak.cu
  (`make build/cublaslt_peak`, vendor GEMM ceiling), the `--pf N` CLI leg
  (engine.cu:1320-1380; `Q27_KV=fp8 Q27_PF_NOSERIAL=1` for the serving
  path), build/microbench_mxf4 (gemm_q4_T TOPS on real projection weights).
- Session traps that bit this week, all still live: vox transcribers add
  host jitter (`sudo -n systemctl stop vox-transcriber vox-transcriber-gmrs`,
  restart after); readiness must key on the unit's InvocationID (the
  launch script does); [req] counters are cumulative per invocation;
  run.sh picks its telemetry parser by label PREFIX -- pass
  SWEBENCH_TELEMETRY=q27 for any label not starting with q27; heredoc
  python discards piped stdin; background shells need
  /usr/local/cuda/bin/nvcc on PATH or a "PASS" comes from a stale binary;
  `pkill -f` matches its own shell; q27-38 is transient (stop deletes it).

## Phase 0 -- switch the shipped cache tiers on (no code, ~40 min)

Why: two thirds of the prefill wall is misses the P16/P16b/P16c tiers
already handle. Every returning turn after an interleaved side request
missed fully (0/6), and first turns never reuse the ~22 K system block.

1. Relaunch production with the tiers:
   `bash tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1`
   (tmpfs disk tier at /dev/shm/q27-pfx, 40 GB budget, RAM tier 16 GB,
   max-tokens 65536, step 8192 default). Q27_SYSBLK=1 logs
   sys_off/sys_len/stable_off per request -- the tool for "why did a
   cross-conversation hit miss".
   Check the boot log: `[pfx]` line reporting the root and budget; no
   pinned-allocation failure (staging is pfx_bytes(65536) ~2.2 GB host
   per slot; 128 GB box).
2. Run the identical traffic (vox ON, as the baseline had it):
   `SWEBENCH_UNIT=q27-38 SWEBENCH_HOST=172.17.0.1 SWEBENCH_EFFORT=high SWEBENCH_TELEMETRY=q27 bash bench/swebench/run.sh prodpfx 2>&1 | tee bench/crossengine/agentic-2026-09-08/prodpfx.log`
   then capture the journal for that invocation:
   `journalctl --user _SYSTEMD_INVOCATION_ID=$(systemctl --user show q27-38 -p InvocationID --value) -o cat --no-pager | grep -E "^\[(req|pfx|gen)\]" > bench/crossengine/agentic-2026-09-08/prodpfx-xhigh.req.txt`
3. Read it:
   `python3 bench/crossengine/agentic-2026-09-08/pf_misses.py prodd2 .../prodd2-xhigh.req.txt prodpfx .../prodpfx-xhigh.req.txt`
   `python3 bench/crossengine/agentic-2026-09-08/pf_agg.py ...` and
   `python3 bench/crossengine/agentic-2026-09-08/lanes_agg.py` for decode.
   `grep "\[pfx\]" prodpfx-xhigh.req.txt` for every restore (alloc/read/
   import ms) and persist.

Bars (against prodd2-xhigh):
- returning turns after another conversation: full misses 6 -> 0; each
  such turn shows hit >= prompt - 8192 - new content (one step of
  re-prefill at most) and a `[pfx] restore` line under 1 s.
- first turns 2..12: `pfx=` >= 20000 (the system-block entry). If they
  still read 0, Q27_SYSBLK=1 tells whether sys_len differs between
  conversations (07-24 saw a 2,144-token drift between sessions; ninfer's
  22,449-token shared hits on this harness say it is stable here) or the
  billing-header normalizer missed CC 2.1.170's format (api_common.h
  normalize_cc_billing_header).
- prefill wall <= 150 s (from 255); decode t/s within +-5% of 201 agg;
  no `[pfx] read FAILED`; no persist on the critical path above 0.2 s
  (persist D2H is ~50 ms/GB; the 07-25 `alloc 7016 ms` was a restore
  joining the writer thread -- watch for it).
- Quality columns (nonempty/gold) are n=1 and NOT a bar.

Decision: if the bars hold, `d2-pfx` becomes production (update the
mode comment in tools/launch_q27_38.sh, campaign.sh's relaunch line,
BUILDLOG, memory). If restores are slow, try RAM tier off (tmpfs alone,
0.47 s restores measured 07-24) or step 4096. If first turns still miss,
that is a separate P16b/normalizer fix, not a reason to hold the
eviction win.

Known interaction: after a pfx restore the DFlash2 ring has no rows for
the restored prefix (d2_prefill_align keeps only VRAM-resident rows), so
the first rounds of that turn draft from a shallow ring -- the same
cold-start the warm-turn fix removed for same-slot turns. Measure, don't
guess: compare tok/round on restored turns vs same-conv turns in
lanes_agg.py. If it costs, persisting the last 2048 tap rows with the
blob is the fix (~40 MB per entry).

## Phase 1 -- slot routing (multi-slot configs only, defer)

All requests landed on slot 0 in every run. The ladder config (multi-slot)
could route a conversation to the slot holding its prefix and send new
conversations to the slot with the cheapest state to lose. Not applicable
to the single-slot DFlash2 production config; phase 0 covers it. Do this
only if the ladder config comes back into production.

## Phase 2 -- the int8 GEMM to the vendor shape (kernel, 2-4 sessions)

Why: cuBLASLt int8 reaches 648-897 TOPS at M=1024 on our projection shapes;
`gemm_q4_T` (prefill.cu:248, `k_gemm_mma_T<Q4IN,XG64,NT>`) sits at
310-322 -- 36-48% of the vendor ceiling. GEMM is ~55-60% of a 25-50 K cold
prefill, so 2x on the GEMM is ~1.4x on the cold wall. ninfer's fp4 GEMM
(the whole of its 2.2x) runs at exactly the level a vendor-class int8
kernel reaches on our existing weights.

Facts a fresh context needs (prefill.cu:232-262):
- Weights: Q4_G64 nibble-packed, fp16 scale per 64 (Q4IN), unpacked to s8
  at the reg->smem store. Activations: int8 per-64 (XG64 nat64/s64) so two
  K=32 MMAs chain in int32 before ONE fp32 dequant step per 64-group; the
  fp32 sum over groups is sequential in K.
- Current tile: MR=64 rows x NT=128 tokens x KS=128 K-stage, 8 warps
  (4 row x 2 token), single-buffered smem with a register-staged next
  stage. The comment says double-buffering and __launch_bounds__ both
  measured SLOWER in THIS shape -- that is the local optimum the vendor
  shape escapes, not a reason to stop.
- Split-K (`gemm_splitk_nsp`, prefill.cu:735) fires only when
  blocks*2 <= nsm, i.e. small T or small rows; it regroups the fp32 sum and
  is tolerance-gated already. The XG64 path itself is NOT serial-vs-batched
  identical and is gated by tolerance + PPL + canonical (policy 2026-07-04).
  So the gate for a new kernel is bitwise vs the CURRENT g64 nsp==1 kernel
  where achievable (same per-64 int32 partials, same sequential fp32 group
  order -> bit-identical regardless of tile shape), else the same
  tolerance battery.
- pf4.cu already carries the modern structure for this codebase (BM/BN/BK
  128/128/256, 8 warps, 2-stage cp.async, swizzled smem, ldmatrix) and
  ninfer's TMA kernel is BlockM 256 x N 128 x K 128, 3-stage mbarrier,
  8 consumer warps + 128 producer threads.

Tasks:
1. Baseline, same session: `make build/microbench_mxf4` and record
   gemm_q4_T TOPS on the four shapes at M=1024 (expect 280-322); run
   `build/cublaslt_peak 0` alongside for the ceiling. Stop vox first.
2. Spike (tools/gemm_w4a8_spike.cu, standalone, synthetic Q4_G64 data):
   a W4A8 kernel with BM=128 tokens x BN=128 rows x BK=128 (two g64
   groups) per stage, 3-stage cp.async pipeline for packed q4 + int8
   activations + scales, ldmatrix for the activation fragments, nibble
   unpack to s8 in registers (lop3/prmt) on the B side, int32 accumulate
   per stage, fp32 fold per 64-group in the SAME order as the incumbent.
   Bitwise check against a straight port of the incumbent's math on the
   same synthetic inputs. Bar to continue: >= 1.6x the incumbent's TOPS
   at M=1024 on ffn_gate (17408x5120) and attn_out (5120x8192).
3. Port into prefill.cu as a new template instantiation behind
   `Q27_PF_GEMM=w4a8v2` (default off), keep the split-K route for the
   small-T shapes untouched. Gates: `--pf 200 seq+32` identity pattern
   (docs/plans/2026-08-17-prefill-performance.md P1 bar), N-invariance
   `tools/ninv_test.cu`, canonical md5 unchanged (NP=5 takes serial
   prefill so it must not move), deep logit A/B at 131072 via
   `--dump-logits` if not bitwise (repro line in
   docs/perf-attribution-prefill-attn.md).
4. End-to-end: `Q27_KV=fp8 Q27_PF_NOSERIAL=1 ./build/q27 <model> --tokens-file
   <toks> --pf 25600` old vs new, and the 12-instance run at xhigh with
   phase 0 on. Bar: cold 25 K first turn 7.2 s -> <= 5.5 s in [req]
   pf_ms; 128 K --pf wall -20% or better.
5. Default on; then the follow-on fusions ninfer has and we do not:
   gate/up in one launch with SwiGLU in the epilogue, residual add in the
   down-proj epilogue. Each bitwise-gated separately.

## Phase 3 -- trims (bitwise, half a session)

- `qxT` (engine.cuh:3772-3780) launches quantize_x (g32) AND quantize_x_g64
  on every projection group; the g32 output is unused when the g64 route
  is taken (default; `Q27_PF_XG=32` is the exact legacy leg). Guard the g32
  launch on the route flag. 2.3-4.3% of prefill. Gate: `--pf` output
  byte-identical on both routes.
- `mtp_warm_T` (engine.cuh:5013, 5042) runs per chunk even when DFlash2 is
  the drafter and the MTP head is never consulted (eh_proj 5120x10240 +
  k/v GEMMs, ~1.5%). Guard on `d2_on`. Gate: seeded DFlash2 output
  byte-identical with and without (bench/ladder/drive_seeded.py).
- Skip the dead work in the 64-256-token turns (20% of requests, 137 ms
  mean, ~1400 tok/s): measure the floor first with a 128-token `--pf`
  under nsys before touching anything -- it may be launch overhead (1600
  eager launches per chunk), the per-chunk sync, or the DFlash2 tap
  copies on the tail.

## Phase 4 -- attention and delta-scan (the 128 K levers, later)

- Prefill attention (`k_attn_prefill_mma_pv8`, prefill.cu:1971): 13.5% of
  prefill at 16 K, 54% at 128 K. Occupancy is proven NOT the lever (FA2
  relayout to 25% moved TTFT -1%, killed 07-09); the warp-specialized
  async rewrite in docs/plans/2026-07-09-prefill-async-rewrite.md
  (+20-27% at 128 K estimated) was never built. Re-attribute with nsys at
  25 K and 128 K AFTER phase 2 lands -- the GEMM cut raises attention's
  share.
- `k_delta_wy` (prefill.cu:2890): 192 blocks, 16 serialized WY_C=64 chunk
  steps per GDN layer per 1024-chunk, warp-0-only substitution; 6-11% of
  prefill, length-flat. Candidate: parallelize the chunk recurrence
  across the 48 heads x 4 column groups more aggressively or fold the
  state passing into a two-level scan (ninfer: prepare / state_passing /
  output as three kernels).

## Phase 5 -- format (not planned)

A native fp4 tier remains dead on the 08-15/08-18 quality verdicts; phase
2 captures most of the fp4 GEMM advantage on the existing weights.

## Order and expected outcome on the prodd2 traffic mix

| after phase | prefill wall (255 s baseline) | first-turn TTFT | mid-conv stalls |
|---|--:|--:|--:|
| 0 | ~130 s | ~2 s (from 7) | gone (restores) |
| 0+2 | ~95 s | ~1.5 s | -- |
| 0+2+3 | ~90 s | -- | -- |

Prefill is 13.5% of the engine wall at xhigh, so the whole plan is worth
~8-9% of total wall at production effort and ~15% at medium; the
user-visible part (the 7 s and 15.7 s stalls) is what phase 0 removes.
