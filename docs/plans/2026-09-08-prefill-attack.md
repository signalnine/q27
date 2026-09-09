# Prefill attack plan (2026-09-08)

Status: phase 0 DONE (2026-09-08 evening, see "Phase 0 result" below);
phase 2 spike DONE, bar not met (TMA W/X-only is the one remaining spike);
phase 3 DONE (trims shipped, A deferred, B opt-in off, C parked); phase 4
remains PLAN. The ranked agenda after all of it is BUILDLOG 2026-09-08 (p)
/ docs/reviews/2026-09-08-gpt6astra-what-next.md: width>8 investigation
first (bounded), turn-replay/quality/queue attribution alongside, then
shared-cut promotion + cache failure paths, then one TMA spike. Evidence and
numbers are in docs/perf-attribution-prefill-2026-09-08.md (the recon); this
file is the executable part. Written so a session with no memory of the
recon can run it: every phase has the launch command, the instrument, the
bar, and the traps.

## 0. State at hand-off (2026-09-08 evening; updated after phase 0)

- Production q27-38 runs `tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1`
  (DFlash2 Q8 + prefix-cache tiers on tmpfs, P16b shared cut) since the
  phase 0 rerun. The pre-phase-0 baseline (mode `d2`, no cache) on the 12
  pinned SWE-bench instances at Claude Code's default effort (q27 renders
  xhigh): bench/crossengine/agentic-2026-09-08/prodd2-xhigh.req.txt -- 299
  requests, prefill wall 255 s, decode wall 1640 s, 201.0 t/s decode
  aggregate, 6 eviction full misses (80 s), 12 cold first turns of 20-25 K
  (80 s). After phase 0 (prodpfx2-xhigh.req.txt): prefill wall 110 s, 0
  full misses, round wall unchanged.
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

How the tiers actually persist and restore (reviewer-verified, read this
before judging a miss):
- Persist fires only at two prefill boundaries, never "on eviction":
  the P16b system-block cut (last chunk boundary <= sys_len, only when
  base == 0, at most once per cold prefill) and the P8 stable boundary
  (stable_len < NP always holds on /v1/messages: server.cu:2752 splits
  the encode at the assistant-open). Both go through pfx_should_persist
  (engine.cuh:4273): cache enabled AND writer not busy AND
  4096 <= L <= max_tokens AND (last_persist == 0 OR L - last_persist >=
  step 8192) AND !has(prompt, L). A busy writer skips the boundary with
  no retry; a failed disk write still advances pfx_last_persist.
- The step gate is SHARED with the system save: a system cut at 21,504
  makes a 28,000-token stable boundary ineligible (growth 6,496) until
  the conversation reaches 29,696. So after an eviction the restored
  entry can be up to one step (8192) plus the growth since behind, and
  the returning turn re-prefills that much -- expected, not a bug. Step
  4096 halves it at the cost of more saves (D2H ~50 ms/GB each).
- Tier resolution (engine.cuh:4873) is P8 -> P9 -> RAM -> disk, first
  match wins, NOT longest match: a shorter RAM system-block entry can
  hide a longer disk conversation entry. Diagnose the selected tier and
  restored length (`[pfx] restore L=`, `[gen] ... pfx=`) before blaming
  persistence. `hit` is the reuse length whatever the tier; `pfx` > 0
  only for RAM/disk restores (a P9 hit has pfx=0).
- The ~350-token foreign request cannot persist (below min 4096); it
  destroys the slot's P8/P9 entries but leaves RAM/disk entries intact.
  That is exactly the case phase 0 fixes.
- Memory: pfx_bytes(65536) is ~2.44 GB (fp8 KV + GDN state); each engine
  pins TWO staging buffers of that size (~4.9 GB pinned) and the RAM tier
  adds budget/2.44 GB slots on top. The launch script defaults the RAM
  tier OFF (tmpfs alone restored in 0.47 s on 07-24; RAM ON was 0.53 s
  first-restore) -- PFX_RAM_GB=16 turns it on (6 slots, ~15 GB more
  pinned). tmpfs budget 40 GB on a 62 GB-free /dev/shm. Disk eviction is
  by write mtime, not access LRU. The flags do not change GPU context
  (pool-clamped, server.cu:1080).

1. Relaunch production with the tiers:
   `bash tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1`
   (tmpfs disk tier at /dev/shm/q27-pfx, 40 GB budget, max-tokens 65536,
   step 8192 default, RAM tier off). Q27_SYSBLK=1 logs
   sys_off/sys_len/stable_off per request -- the tool for "why did a
   cross-conversation hit miss" (logging only; it enables nothing).
   Check the boot log: the `prefix-cache:` line (server.cu:878) reporting
   root, entries, budget, min/max/step; the cache must be ENABLED there
   or sys_len is never computed and the RAM tier never inits.
2. Run the same harness and settings as the baseline (vox ON as the
   baseline had it; the traffic itself is fresh Claude Code sessions, so
   turn counts and trajectories differ -- compare miss CLASSES, not
   request counts). Record the image identity first:
   `docker image inspect thunderdome/claude-code:latest --format '{{.Id}}'`.
   `SWEBENCH_UNIT=q27-38 SWEBENCH_HOST=172.17.0.1 SWEBENCH_EFFORT=high SWEBENCH_TELEMETRY=q27 bash bench/swebench/run.sh prodpfx 2>&1 | tee bench/crossengine/agentic-2026-09-08/prodpfx.log`
   then capture the FULL journal for that invocation (keep `[sysblk]`,
   `[d2]`, `prefix-cache:` write-failure and eviction lines; filter later):
   `journalctl --user _SYSTEMD_INVOCATION_ID=$(systemctl --user show q27-38 -p InvocationID --value) -o cat --no-pager > bench/crossengine/agentic-2026-09-08/prodpfx-xhigh.journal`
   and derive `grep "^\[req\]" ... > prodpfx-xhigh.req.txt` for the
   aggregators.
2b. Controlled eviction case (deterministic, 5 min, do it BEFORE the full
   run): drive conversation A to ~28 K and again to ~48 K with
   bench/ladder/drive_warm_turn.py or captured requests, send one ~350-
   token request from a different conversation, then extend A by one
   turn. Record the last persisted boundary and the returned hit/pfx.
   This is the acceptance test for the mechanism; the 12-instance run is
   the traffic-level confirmation.
3. Read it:
   `python3 bench/crossengine/agentic-2026-09-08/pf_misses.py prodd2 .../prodd2-xhigh.req.txt prodpfx .../prodpfx-xhigh.req.txt`
   `python3 bench/crossengine/agentic-2026-09-08/pf_agg.py ...` and
   `python3 bench/crossengine/agentic-2026-09-08/lanes_agg.py` for decode.
   `grep "\[pfx\]" prodpfx-xhigh.req.txt` for every restore (alloc/read/
   import ms) and persist.

Bars (against prodd2-xhigh; measured acceptance targets, not code
guarantees -- see the predicate above):
- returning turns after another conversation: full misses 6 -> 0; each
  such turn shows hit >= (last persisted boundary) with a `[pfx] restore`
  line under 1 s, and the re-prefilled remainder explained by the step
  gate + growth since that boundary.
- first turns 2..12: `hit` >= 20000 (the system-block entry, via RAM/disk
  `pfx` > 0 or a surviving P9 entry). A hit needs identical tokens
  THROUGH the saved cut: effort, tool set and declaration order, system
  text and the billing header all matter (server.cu:2681,
  api_common.h:744; normalizer at api_common.h:233 handles `cch=` and the
  4th+ cc_version component). If first turns miss, Q27_SYSBLK=1 gives
  sys_len per request -- equal sys_len does not prove equal tokens and
  unequal does not prove a miss (the difference may sit past the cut);
  dump two first-turn prompts and compute the token LCP against the saved
  cut. ninfer's 22,449-token shared hits on this harness are supporting
  evidence under ITS rendering, not proof under ours.
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

Known interaction (reviewer-checked: the flags do not break the
Q27_BATCH=0 DFlash2 config; the fold event is waited before restore/
reset, and reset() leaves the drafter ring alone): a RAM/disk restore
carries no drafter rows, but the re-prefilled suffix re-seeds the ring
through the normal tap capture, so any restore whose suffix is >= 2048
tokens (D2_SEED_WINDOW) rebuilds the FULL seed window; only shorter
suffixes draft from a shallow ring. Measure, don't guess: add a grouping
to lanes_agg.py for restored turns (pfx > 0) vs same-conv turns (its
current groups are completion-length buckets) and compare tok/round. If
it costs, persisting the 2048 tap rows with the blob is 200 MiB per
entry (5 taps x 5120 floats x 2048 rows), so a cheaper fix would be to
recompute the last 2048 tokens' taps on restore.

### Phase 0 result (2026-09-08, 14:36)

Run: bench/crossengine/agentic-2026-09-08/README.md "Prefill attack phase
0". Controlled case passed every bar (restore of a 48,852-token entry in
254 ms after a foreign 369-token request, re-prefill 44 tokens; system-block
entry restored for a new conversation in 44 ms). 12 instances: the
after-other-conversation class went 6 full misses (80 s) -> 2 restores (0.2 s
each, 0 misses); same-conversation turns all hit; no read failures; persist
exports median 74 ms, four at 202-217 ms. Decode +12% on different
trajectories (traffic). `d2-pfx` is production.

First turns: 0 of 26 hit. Root cause from the entry token vectors: all
sessions share EXACTLY 22460 tokens and diverge inside Claude Code's
gitStatus section (per-repo "Recent commits:" hashes), and the P16b cut at
the last chunk boundary <= sys_len (22528) lies 68 tokens past that. Fix
(same day): `PrefixCache::shared_prefix(prompt, sys_len)` = longest prefix an
indexed entry shares with the prompt (token vectors only); the engine cuts
the system entry at that length when it is shorter than sys_len (cut lands
at 21504 here). Session 1 cuts at sys_len, session 2 at the shared length,
session 3 onward restores. Gates: tools/test_prefix_cache.cpp
(test_shared_prefix_across_sessions), the three-session live probe
(bench/ladder/pfx_shared_probe.py: shared body ending just under a chunk
boundary, sys_len just over it, so old and new cuts differ), then the
12-instance rerun on a fresh root (prodpfx2): bar = first turns 3..26 hit
>= 20000 with a `[pfx] restore L=21504` under 0.3 s, first-turn prefill wall
-70% or better.

Rerun result (14:41): shipped and measured. Session 2 cut at 21504; 12 of
the 13 later first turns that carry a system block restored L=21504 in
79 + 45 ms and re-prefilled 2.3-4.5K tokens (1.0-1.7 s vs 7.0-8.4 s cold);
the three cold ones after bootstrap had no system block (Claude Code side
calls). Both miss classes 0; prefill wall 255 -> 110 s at matched turn
counts; first-turn wall 85.5 -> 44.3 s (-48%, of which two cold bootstrap
turns are 14 s -- the -70% bar assumed hits from session 2, the mechanism
needs one extra session); round wall 18.63 = baseline; quality unchanged.
Live probes are in bench/ladder/pfx_evict_probe.py and pfx_shared_probe.py.
Phase 0 is CLOSED; `d2-pfx` is production (launch script, campaign.sh
relaunch line, BUILDLOG 2026-09-08 (g)). gpt-6-astra reviewed the shared
cut afterwards (docs/reviews/2026-09-08-gpt6astra-shared-cut.md, BUILDLOG
(i)): the concurrent-writer race and the eviction/publish race are fixed;
one item is DEFERRED as a follow-up: a restored prefill (base > 0) never
runs the shared-cut discovery, so after a client change a short matching
entry can pin every session to it. Fix = an explicit promotion policy (run
shared_prefix on restored prefills too and persist a longer system cut when
the restored base is a system-class entry shorter than the shared length;
the 8192 step gate must not suppress it). Not urgent: today's worst case is
a shorter hit, never a miss.

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

Facts a fresh context needs (prefill.cu:232-262, 442-488, 502, 874):
- Weights: Q4_G64 nibble-packed, fp16 scale per 64 (Q4IN), unpacked to s8
  at the reg->smem store. Activations: int8 per-64 (XG64 nat64/s64).
  Numerics per 64-group: a FRESH int32 accumulator, two K=32 MMAs chained,
  then one fp32 update `acc += (wscale*xscale) * int_sum` into the
  persistent fp32 accumulator; groups are visited in increasing global
  order (gg=0 then gg=1 inside each 128-K stage). The epilogue only
  stores. So bitwise equivalence across tile shapes IS achievable: keep
  separate int32 accumulation per 64-group (summing two differently
  scaled groups into one stage accumulator is WRONG, not merely
  non-bitwise), the increasing group order, the same scale product and
  rounding, and the same FMA contraction behaviour.
- LIVE dispatch on sm_120 at saturated large T (the production case) is
  NOT `k_gemm_mma_T<..,128>` but `k_gemm_mma_ntx<..,96>`: MR=128 rows x
  NT=96 tokens, two row minitiles per warp (prefill.cu:502, 874; the +3.4%
  ntx route). `k_gemm_mma_T` MR=64 x NT=128 x KS=128 is the numerical
  reference and the small-T route. Activation ldmatrix is already on by
  default in both (prefill.cu:36). Both are single-buffered smem with a
  register-staged next stage; the "double-buffering measured slower"
  comment is about THIS shape -- the vendor shape escapes it, not a
  reason to stop. build/microbench_mxf4 measures live dispatch
  (microbench_mxf4.cu:1133), so its 310-322 IS the ntx number.
- Split-K (`gemm_splitk_nsp`, prefill.cu:735, 797) auto-fires only when
  blocks*2 <= nsm AND g64 AND scratch is present; forced mode bypasses the
  occupancy threshold. It regroups the fp32 sum and is tolerance-gated.
  The XG64 path itself is NOT serial-vs-batched identical and is gated by
  tolerance + PPL + canonical (policy 2026-07-04). Gate for a new kernel:
  bitwise vs the current g64 nsp==1 kernels (both MR=64 and ntx must
  agree with it today -- confirm first) via a direct old/new GEMM
  comparison across tile boundaries and partial tiles (T not a multiple
  of the tile, rows not a multiple of MR); `tools/ninv_test.cu` covers the
  decode/verify families, NOT gemm_q4_T, so this comparison has to be
  written (into the spike or microbench_mxf4). If not bitwise: state the
  thresholds up front (deep logit A/B at 131072: cosine >= 0.99998,
  argmax MATCH, top-5 5/5, as the fp8q gate had) plus the agentic
  long-context quality battery, AND a separate prefix-cache root per
  numerical variant -- cache compatibility (prefix_cache.h:57) does not
  encode kernel numerics, so a restored blob from the old kernel would
  silently mix numerics with the new one.
- pf4.cu already carries the modern structure for this codebase (BM/BN/BK
  128/128/256, 8 warps, 2-stage cp.async, swizzled smem, ldmatrix) and
  ninfer's TMA kernel is BlockM 256 x N 128 x K 128, 3-stage mbarrier,
  8 consumer warps + 128 producer threads.

Tasks:
1. Baseline, same session: `make build/microbench_mxf4` and record the
   live-dispatch TOPS on the four shapes at M=1024 (expect 280-322; that
   is the ntx kernel at T=1024); run `build/cublaslt_peak 0` alongside
   for the ceiling. Stop vox first. Also record the MR=64 reference
   kernel's output on the same inputs -- it is the bitwise reference.
2. Spike (tools/gemm_w4a8_spike.cu, standalone, synthetic Q4_G64 data):
   a W4A8 kernel with BM=128 tokens x BN=128 rows x BK=128 (two g64
   groups) per stage, 3-stage cp.async pipeline for packed q4 + int8
   activations + scales, ldmatrix for the activation fragments, nibble
   unpack to s8 in registers (lop3/prmt) on the B side, a SEPARATE int32
   accumulator per 64-group (two per 128-K stage), fp32 fold per group in
   the incumbent's increasing order with the incumbent's scale product.
   Bitwise check against the MR=64 reference on the same synthetic inputs,
   including partial tiles. Bar to continue: >= 1.6x the live-dispatch
   TOPS at M=1024 on ffn_gate (17408x5120) and attn_out (5120x8192).
3. Port into prefill.cu as a new template instantiation behind
   `Q27_PF_GEMM=w4a8v2` (default off), keep the split-K route for the
   small-T shapes untouched. Gates: (a) the direct old/new GEMM
   comparison from task 2 on real weights; (b) `--pf N` serial-vs-batched
   identity is a LEGACY-PATH gate only (`Q27_PF_XG=32`; on the default g64
   route a mismatch exits successfully, engine.cu:1374 -- it does not
   enforce new-kernel correctness); (c) canonical md5 unchanged (NP=5
   takes serial prefill so it must not move); (d) deep logit A/B at
   131072 via `--dump-logits --ctx 133120` if not bitwise (repro line in
   docs/perf-attribution-prefill-attn.md), thresholds as stated above.
4. End-to-end: `Q27_KV=fp8 Q27_PF_NOSERIAL=1 ./build/q27 <model> --tokens-file
   <toks> --pf 25600 --ctx 27648` old vs new (the CLI --ctx default is
   2048 and the run refuses without it), and the 12-instance run at xhigh
   with phase 0 on and a FRESH cache root. Bar: cold 25 K first turn
   7.2 s -> <= 5.5 s in [req] pf_ms; 128 K --pf wall -20% or better.
5. Default on; then the follow-on fusions ninfer has and we do not:
   gate/up in one launch with SwiGLU in the epilogue, residual add in the
   down-proj epilogue. Each bitwise-gated separately.

### Phase 2 status (2026-09-08 evening, task 2 spike done, bar not met)

BUILDLOG 2026-09-08 (h). tools/gemm_w4a8_spike.cu is a bitwise W4A8 kernel
(all shapes, all T incl. tails) at 1.30x (ffn_gate) / 1.41x (attn_out) at
M=1024 -- short of the 1.6x bar. Measured ceilings: IMMA pipe 1020 TOPS,
register-only exact fold 827, kernel without the stage fill 609, fill alone
6.8 TB/s; the gap is fill/compute non-overlap, not the fold or occupancy.
Task 2 continues, in the order gpt-6-astra set
(docs/reviews/2026-09-08-gpt6astra-w4a8-spike.md, BUILDLOG (j)):
(a) TMA (cp.async.bulk.tensor, single-CTA, expect_tx mbarriers) for the W
and X tiles ONLY, issued by an elected thread inside the consumer CTA,
scale loads unchanged -- isolates TMA's effect before any layout change
(the cp.async producer-warp attempt regressed to 326-350 because it put 56
copies per lane per stage on one warp and changed register allocation;
not a verdict on specialization); (b) only then transposed scale sidecars
(W [ngrp][rows] fp16 at load, xs [ngrp][T] from the quantizer; inner box
a multiple of 16 B, strides 16-B aligned, hardware swizzle validated
against the fragment layout); (c) the structural matrix: 128x192 (4x3) and
192x128 (6x2), 64x128 / 128x64, 4x2 vs 2x4, grouped rasterization over
2-8 row tiles, BK 64/128/256 on smaller tiles, the operand-role swap
(tokens on the MMA's M side: a new fragment/permutation mapping, not a
pointer swap), split-K only for underfilled grids. smem: 128x128x256/s2
and 256x128x128/s3 are 102 KiB > the 99 KiB block limit. Probes and
traps in tools/probes/README.md (ptxas hoists loop-invariant mma:
register-only IMMA loops must perturb an input per iteration).

Task 3 (the port behind Q27_PF_GEMM=w4a8v2) gate list, from the same
review: the fold ported as explicit mul.rn/fma.rn asm (identical source
expressions are a compiler-dependent contract), build flags recorded and
the production SASS inspected; FOLD==2 (cvt16) never ships (conditionally
exact below 16*FLT_MIN); nat64p is an ADDITIONAL quantizer output from
the same rounded q0/q1 (never a replacement: Q8 and the fallbacks read
nat64), with Q8 exercised on the same XQuant right after Q4 and
Q27_PF_XG=32 / Q27_PREFILL=dp4a / missing-g64 routes covered; preserve
the split-K DECISION (scratch capacity, forced counts, uneven partitions)
and route w4a8v2 only where the incumbent would not have split; shape
contract explicit (cols % 128, rows 1/7/8/15/16/17 and BN boundaries,
every T 1..17 and the dispatch boundaries, pipelines shorter than the
stage count); numerics gate against BOTH incumbent kernels on real
projection weights and captured activations plus endpoint patterns (all
nibble values, both activation signs, max dots, exact cancellation,
one-hot K at every position, distinct adjacent-group scales, halfway
rounding, tiny products, overflow, signed zero); exact-sized allocations
with checked red zones (the spike does this now); buffers, sidecars and
descriptors allocated on the existing init paths under the arena's
per-chunk claim discipline, no process-global mutable state and no lazy
allocation while another engine may be capturing; acceptance = unchanged
legacy --pf identity, unchanged decode canonicals, direct g64 old/new,
mixed Q4/Q8, short suffixes as well as 1024-token chunks, quantizer and
sidecar cost included. Default off until the bar is met or the smaller
gain is explicitly accepted.

## Phase 3 -- trims (bitwise, half a session)

- `qxT` (engine.cuh:3772-3780) launches quantize_x (g32) AND quantize_x_g64
  on every projection group; the g32 output is unused when the g64 MMA
  route is taken (default). Guard the g32 launch on the route flag, but
  the guard must keep g32 alive for `Q27_PF_XG=32` (exact legacy leg) AND
  for `Q27_PREFILL=dp4a` (prefill.cu:900 consumes the g32 buffers
  regardless of the g64 preference). 2.3-4.3% of prefill. Gate: `--pf`
  output byte-identical on all three routes.
- `mtp_warm_T` (engine.cuh:5013, 5042) runs per chunk even when DFlash2 is
  the drafter and the MTP head is never consulted (eh_proj 5120x10240 +
  k/v GEMMs, ~1.5%). Guard on `d2_on`. Gate: seeded DFlash2 output
  byte-identical with and without (bench/ladder/drive_seeded.py). Side
  effect: the persisted prefix blobs carry the MTP KV (engine.cuh:4101);
  a blob written with the warm pass skipped would restore stale MTP state
  into a LADDER config later. Use a separate cache root for d2 production
  (or version the entry) before the ladder ever reads that root.
- Skip the dead work in the 64-256-token turns (20% of requests, 137 ms
  mean, ~1400 tok/s): measure the floor first with a 128-token `--pf`
  under nsys before touching anything -- it may be launch overhead (1600
  eager launches per chunk), the per-chunk sync, or the DFlash2 tap
  copies on the tail.

### Phase 3 status (2026-09-08 evening): trims SHIPPED, floor measured

BUILDLOG 2026-09-08 (k). Shipped bitwise: qxT skips the g32 quantize on
the g64 route (-9.7 ms per 1024-chunk); mtp_warm_T skipped under DFlash2
(/dev/shm/q27-pfx is thereby a DFlash2-only root -- blobs carry unwarmed
MTP rows); k_gemm_f16_T warp-per-token register-tree retile (212 -> 61.5
us, -14.5 ms per chunk). CLI 1024-token prefill 285 -> 261 ms.

The small-turn floor (nsys, production config): a warm 41-token turn is
76 ms = chunk A 35 + a SECOND full chunk for the ~5 post-boundary tokens
25 + the eager last-token step 14; each streams the whole 13.5 GB weight
set (small-T GEMM at 0.7-1 TB/s), and the prefill attention kernel runs 4
blocks for 320 us per layer at any T <= 64 (5 ms per chunk). Levers, all
numerics-class (tolerance/quality gates, separate cache root, user's
call): (A) mid-chunk GDN snapshot at the stable boundary so chunk B
disappears (-25 ms per warm turn); (B) fold the last prompt token into the
batched chunk, logits from the head GEMV on that row (-12 ms; first
decode token moves to g64 numerics; canonical NP=5 prompts stay serial);
(C) engage the attention position split when the grid underfills, not
only at deep base_pos (-4.5 ms per chunk). Together 76 -> ~35 ms per warm
turn; the production 64-256-token turns (137 ms mean) about -30%.

Order after the gpt-6-astra advisory (docs/reviews/2026-09-08-gpt6astra-
small-turn-levers.md, BUILDLOG (l)): (0) DONE, BUILDLOG (m): the DFlash2
last-token forward is a graph (-2 ms per turn, bitwise; the eager step had
only 2.4 ms of submission gaps, the rest is its weight stream); (1) C
MEASURED AND PARKED, BUILDLOG (n): the Q27_PF_SPLIT sweep at 3K/25K/45K
shows the depth rule already splits 6-8 ways at Claude Code depths, so an
underfill rule is worth <= 3 ms per turn there (7-8 ms only below ~8K) --
not worth a numerics-class change; (2) B BUILT AND GATED, BUILDLOG (o):
-12 ms per warm turn, but the batched per-token NLL is +1.37% over serial
on the agentic corpus (inside the +2% rule) and the first-token logits
move 10-100x more than the accepted prefix class (KL median 1e-2 vs 1e-4);
shipped OPT-IN (Q27_PF_FOLDLAST=1), default off -- the user's call whether
~1.3% of run wall is worth a +1.4% first-token NLL. Original item: append the last token to an
existing post-snapshot chunk with room, existing head GEMV, MTP/DFlash2
traps as listed; (3) A only after repricing, with explicit boundary-state
export (snap/ckpt/pfx copy LIVE state today). Every numerics-class lever
needs the teacher-forced turn-replay gate through generate_prefill (the
batched NLL loop bypasses it), the +2% NLL rejection ceiling, DFlash2
tok/round and request wall, and its own cache root.

After (p) (gpt-6-astra "what next"): A stays DEFERRED (192 eligible turns
x 25 ms = 4.8 s = 0.6% of the run for 3-5 sessions + boundary-state export
risk); B's run-wall fraction corrected to 0.3% (219 x 12 ms = 2.6 s); the
prefill chunk graphs are deferred until the removable submission gaps are
bounded (the trace was 88% GPU-busy; the 55 ms intercept is a fixed cost,
not launch overhead). The next work on this plan is item 3 of (p): the
shared-cut promotion with discovery + the cold-only save predicate + the
8192 step gate handled, and the cache failure paths (the reservation leak
is fixed in the (p) commit; eviction-by-mtime is to be measured first).

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

## Order and expected outcome on the prodd2 traffic mix (conditional estimates)

| after phase | prefill wall (255 s baseline) | first-turn TTFT | mid-conv stalls |
|---|--:|--:|--:|
| 0 | ~130 s | ~2 s (from 7) | restores + <= one step of re-prefill |
| 0+2 | ~95 s | ~1.5 s | -- |
| 0+2+3 | ~90 s | -- | -- |

Assumptions behind the table: phase 0's hits land as the predicate above
allows (first turns share the system block through the cut); phase 2
delivers 2x on a 55-60% GEMM share, which is 1.38-1.43x on the cold
prefill wall -- it closes most, not all, of ninfer's 2.2x (the rest is the
fp4 operand and their fused epilogues). Prefill is 13.5% of the engine
wall at xhigh, so the whole plan is worth ~8-9% of total wall at
production effort and ~15% at medium; the user-visible part (the 7 s and
15.7 s stalls) is what phase 0 removes.

## Review

gpt-6-astra reviewed this plan statically before hand-off
(docs/reviews/2026-09-08-gpt6astra-prefill-plan.md); its eleven items are
folded in above: the persist predicate and shared step gate, first-match
tier resolution, memory accounting (RAM tier defaulted off), the P16b
hit conditions, the DFlash2 seed-window correction (>= 2048-token
suffixes re-seed fully; taps are 200 MiB not 40 MB), the stale kernel
shape (live dispatch is ntx MR=128 x NT=96), separate int32 accumulators
per 64-group, `--ctx` on the --pf command, the legacy-only nature of the
`--pf` identity gate on g64, ninv_test not covering gemm_q4_T, separate
cache roots per numerical variant, the dp4a consumer of g32, the MTP
contents of persisted blobs, and "same harness" instead of "identical
traffic".
