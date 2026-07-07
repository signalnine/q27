# Prefill Attention O(N^2): Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use conclave:executing-plans to implement this plan task-by-task.

**Goal:** Cut deep-context prefill wall (TTFT on cold prompts + the R1b re-prefill class) by attacking the one length-quadratic kernel. P14 Step 7 measured `k_attn_prefill_mma` at **41.16 s = 54.2% of 128K prefill kernel time** (0.69 s / 13.5% at 16K), driving the 3201 -> 1717 t/s decay; GEMM and GDN are length-flat per token. This plan takes 128K prefill from **~71.5 s toward the high-40s** by raising the attention kernel's sustained utilization and (Phase 2) consuming the already-fp8 KV cache directly in the MMA.

**Scope boundary:** decode is untouched. This kernel (`k_attn_prefill_mma`, prefill.cu) is disjoint from the fd2 decode-attention kernel. No change here can move a decode canonical.

**Architecture:** measure-first, then two independent perf phases on one branch (`prefill-attn`), gated hardest-last. Phase 1 (cp.async software pipeline) is a **bitwise-safe** memory-scheduling change -- no fp reorder -- so it lands under the existing canonical md5 and the prefill bitwise-identity gate. Phase 2 (fp8 MMA) changes the numeric path and lands under the **tolerance-gate** battery (P1.5/risk-6 precedent), one gate + prefix-cache-identity cycle. Phase 0 (profile) produces the go/no-go occupancy data and must run first because the "~39% of peak" figure below is FLOP-derived, not a profiler counter.

**Tech Stack:** CUDA 13.2 C++ (sm_120 primary, sm_86 must also compile), `mma.sync`/`ldmatrix`, `cp.async` (sm_80+ PTX, works on sm_120), dynamic smem, Makefile (`make` full builds ONLY), test_kernels harness, nsys/ncu profiling.

---

## Non-negotiable project rules (read before every task)

These come from the BUILDLOG's paid-for lessons. Violating them has burned whole sessions.

1. **Full `make` always.** Never `make <target>` -- target-scoped builds leave stale sibling binaries. The tell for a stale binary: numbers identical to the previous run.
2. **Canonical gate** (exact recipe, verified 2026-07-06):
   ```bash
   CUDA_VISIBLE_DEVICES=0 ./build/q27 /mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27 \
     --tokens "760,6511,314,9338,369" --ctx 2048 --spec -n 128 2>/dev/null \
     | grep '^generated:' | md5sum
   ```
   must print `4c4120c72056aba2bc2d2561471eafce`. An EMPTY-input md5 (`d41d8cd9...`) means the run failed -- usually the back-to-back VRAM teardown race (sleep 12-15 s and retry). **This gate exercises decode, not deep prefill** -- necessary but NOT sufficient here (the 2K canonical barely enters the attention kernel's quadratic regime and never triggers position splits). The prefill-specific gates are in each task below.
3. **GPU discipline:** `nvidia-smi` before any bench (vox-transcriber owns the 3090; the 5090 must show ~0 MiB used). Long jobs (>2 min) via `systemd-run --user --unit=<name> /usr/bin/bash -c "cd /mnt/ai/projects/q27 && <cmd>"` -- session crashes kill setsid jobs (cgroup teardown). Units are transient: `systemctl --user stop` deletes them; `reset-failed` after crashes. Sleep 10-15 s between back-to-back CLI model loads (VRAM teardown OOM race).
4. **Benchmarks:** memory OC must be STOCK for any recorded number. Prefill wall from the `--kvstats N --ctx N` timing print (value-independent, so synthetic ids are faithful for TIMING). Never record nsys-inflated walls (node tracing inflates ~7% here, 2.3x on decode-node traces) -- no-profiler runs are ground truth. Prefill wall is near-deterministic; n=3, report median, spread >2% => stale binary or GPU contention, stop.
5. **nsys/ncu:** attention prefill is profiled on a pure `--kvstats N` prefill (NOT a `--tokens` CLI run -- CLI prefills serially and hides the batched kernel). `ncu` needs `sudo -n`. Use `--trace=cuda` for the kernel-time breakdown; `ncu --set full` on `k_attn_prefill_mma` for occupancy/pipe counters.
6. **Lane-count / grid-shape landmine:** any batched kernel selecting per-lane outputs by explicit pointers instead of struct indexing corrupts silently when widths change (the quantize3 bug). This kernel hardcodes `gqa=6`, `head_dim=256`, warp=q-head; touching the tiling must keep those identities exact. Grep the assumptions before editing.
6a. **Latent `t0>0` split-write corruption (CUDA review #4, docs/cuda-review-2026-07-07.md).** `k_attn_prefill_mma`'s split-partial write uses the ABSOLUTE token index `tr0 = t0 + gid` (`prefill.cu:1047`) while the combine reads RELATIVE rows `[0,SB)` (`prefill.cu:1085`) and `part` is sized for `SB` rows. It is correct today ONLY because the sole caller passes `t0=0` (`engine.cuh:1446`). If any task here introduces `t0>0` sub-batching (e.g. tiling the chunk into smaller attention calls), the split path silently overwrites the next head's partials. Before any such change: fix the write to relative rows (`tr0 - t0`), or size scratch from `t0+SB`, or assert `t0==0` when `part != nullptr`. Do NOT assume the documented "arbitrary `t0`" contract in `prefill.cuh:44` actually holds.
7. **Commit style:** each task commits with a BUILDLOG entry (`docs/BUILDLOG.md`, chronological, at the end) recording what/why/numbers. Do NOT push -- Gabe reviews and pushes (public repo).
8. **Determinism contracts:** greedy output bitwise-stable run-to-run; prefill bitwise-identical to the serial GEMV path is the current invariant for the fp16 path (gated on identical warm-vs-cold continuations). Phase 2 relaxes this to a tolerance gate for the fp8 path ONLY, with the fp16 path kept bitwise as the reference and `Q27_ATTN_PF=lite` + an fp8-off fallback preserved.

Model/tokenizer paths: `/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27` + `.tok`.
Server bench template: `CUDA_VISIBLE_DEVICES=0 ./build/q27-server <model> <tok> --port 8081 --ctx 131072 --no-think --fast-head`.
Existing knobs on this kernel: `Q27_ATTN_PF=lite` (falls back to `k_attn_prefill_T`, the pre-P1.5 reference), `Q27_PF_SPLIT=N` (forces P4 position splits; 1 = pre-split bit-identical path), `Q27_KV=fp8` (E4M3 cache), `Q27_PF_XG=32` (exact activation-group path vs g64 default).

---

### Task 0: Branch + fixtures + baseline

**Files:** none (git + fixtures + baseline capture)
**Dependencies:** none

**Step 1:** Clean tree at HEAD `623cdb1`, 5090 free:
```bash
cd /mnt/ai/projects/q27 && git status --short && git log --oneline -1
nvidia-smi --query-compute-apps=pid,name --format=csv
ls -lt src/ | head -3   # mtimes predate this session
git checkout -b prefill-attn
```

**Step 2:** Regenerate the kvstats fixtures -- **they were cleaned from the tree** (the P14 attribution used `scratchpad/synthtoks.bin`, 140,000 int32 ids, and `scratchpad/prep_tokens.py`; both are gone). Recreate per docs/perf-attribution-p14.md lines 20-27: 140K int32 token ids (tiled from a ~75K real transcript slice, max id < VOCAB 248,320). If `prep_tokens.py` is unrecoverable from git history (`git log --all --oneline -- scratchpad/prep_tokens.py`), a synthetic file of 140,000 random int32 in `[0, 248320)` is faithful for TIMING (prefill wall is value-independent -- stated in the doc and re-proven by the fp8-refusal-after-timing trick). Store under `scratchpad/` (gitignored). Record the exact regen command in the BUILDLOG so the next session isn't blocked again.

**Step 3:** Full build + kernel tests + canonical:
```bash
make 2>&1 | tail -3 && ./build/test_kernels 2>&1 | tail -5
CUDA_VISIBLE_DEVICES=0 ./build/q27 <model> --tokens "760,6511,314,9338,369" --ctx 2048 --spec -n 128 2>/dev/null | grep '^generated:' | md5sum
# expect 4c4120c72056aba2bc2d2561471eafce
```

**Step 4:** Capture the baseline prefill breakdown at the two anchor lengths (this plan's before-numbers). fp8 KV (production default), g64 default:
```bash
# wall (no profiler, ground truth):
CUDA_VISIBLE_DEVICES=0 ./build/q27 <model> --nll scratchpad/synthtoks.bin --kvstats 16384  --ctx 16384  2>&1 | grep -iE 'prefill|t/s'
CUDA_VISIBLE_DEVICES=0 ./build/q27 <model> --nll scratchpad/synthtoks.bin --kvstats 131072 --ctx 131072 2>&1 | grep -iE 'prefill|t/s'
```
Expected from P14 Step 7 (sanity, not gates): 16K ~5.1 s / ~3200 t/s; 128K ~71-76 s / ~1717-1834 t/s. If these are off by >5%, STOP -- the tree or GPU state is wrong.

**Gate:** clean build, test_kernels ALL PASS, canonical exact, baseline walls within 5% of P14. Commit is doc-only (this plan + BUILDLOG "starting prefill-attn" entry); no code yet.

---

### Task 1: Phase 0 -- profile `k_attn_prefill_mma`, confirm the bottleneck class

**Files:** none (measurement + a BUILDLOG attribution entry)
**Dependencies:** Task 0

**Why:** the ranked levers below assume the kernel is **compute-pipeline/occupancy bound**, not bandwidth bound. Derivation: useful attention FLOPs at 128K = 16 attn layers x 24 q-heads x (N^2/2 causal) x (2 for QK^T + 2 for PV) x 256 head_dim ~= 3.4 PFLOP; over 41.16 s that is ~82 TFLOPS sustained ~= **39% of the 5090's ~209 TFLOPS fp16-MMA peak**. Minimum KV DRAM traffic ~= 1.1 TB => ~0.6 s at 1.79 TB/s, so bandwidth is >60x from binding. This must be confirmed with real counters before committing to kernel rewrites -- if ncu instead shows a memory or latency stall, the lever ranking changes.

**Step 1:** ncu one 128K prefill, target the attention kernel:
```bash
sudo -n ncu --set full --launch-count 3 --launch-skip 200 -k k_attn_prefill_mma \
  -o scratchpad/pf_attn_128k \
  CUDA_VISIBLE_DEVICES=0 ./build/q27 <model> --nll scratchpad/synthtoks.bin --kvstats 131072 --ctx 131072
```
(`--launch-skip` past the shallow early tiles into the deep-context regime where the quadratic term dominates; adjust so the captured launches are late-context.)

**Step 2:** Read out and record in a new `docs/perf-attribution-prefill-attn.md`:
- Achieved occupancy vs theoretical. Prediction: theoretical is **~1 block / 6 warps per SM** -- `__launch_bounds__(192,1)` + ~82.5 KB dynamic smem ((6*16 + 2*32) * 264 * 2 B) forces one block per SM on the 100 KB budget, ~12% occupancy. Confirm the smem figure is what caps it (vs registers/launch bound).
- Tensor pipe active %, memory pipe active %, warp stall reasons (expect `short_scoreboard` / `barrier` at the `__syncthreads` before/after K/V staging dominating -- 6 warps cannot hide the smem-stage latency, and the PP=32 loop serializes stage against MMA).
- Sustained tensor TFLOPS -- ground-truth the ~39%-of-peak estimate.

**Gate / decision:** if tensor-pipe-idle + low-occupancy + barrier-stall is confirmed (expected), Phase 1 (overlap staging with MMA) and Phase 2 (more MMA throughput + smaller smem) both attack it -- PROCEED. If instead the kernel is already >70% tensor-active (surprise), Phase 1's headroom is small; re-rank toward Phase 2 (fp8 doubles MMA throughput regardless) and longer tiles, and note the correction. This task is cheap and decides the rest.

> **MEASURED 2026-07-07 -- PROCEED** (full data in `docs/perf-attribution-prefill-attn.md`). Deep launch (base ~121K, 44.8 ms): DRAM **1.98%**, L2 hit **95.6%** (KV L2-resident, not re-streamed), tensor **33%** ("should not be a bottleneck"), IPC **0.42**, achieved occupancy **12.5%** (6 warps/SM, 1 CTA). Stalls spread across long_scoreboard 30% / math_pipe_throttle 28% / barrier 15% / wait 14% = occupancy-starvation signature. Two corrections to the levers below: (1) the FLOP-derived ~39% tensor is really ~33%; (2) **occupancy is dual-limited by registers (248/thread) AND smem** -- Block Limit Registers = 1 AND Block Limit Shared Mem = 1 -- so Phase 2's "smaller smem" alone will NOT raise occupancy; see the register-cut note added to Task 3.

---

### Task 1.5 (Phase 0.5): grid-remap for KV L2 reuse -- cheapest bitwise-safe probe (perf-review item 2c)

**Files:** `src/prefill.cu` (`attn_prefill_launch` grid + `k_attn_prefill_mma` blockIdx index derivation)
**Dependencies:** Task 1 (do the ncu first so we can measure the L2-hit delta), but independent of Tasks 2-3 -- run it first as the lowest-risk bite.

**What:** the launch grid is `dim3 grid(n_kv_heads, tiles, nsplit)` (`prefill.cu:1217`), so the 4 KV heads vary fastest (blockIdx.x). Adjacent CTAs are thus *different* KV heads reading *disjoint* K/V -- zero adjacent-CTA reuse. Remap so query **tiles** vary fastest (`grid(tiles, n_kv_heads, nsplit)`, swap the blockIdx.x/y roles in the kernel's `kvh`/`t0` derivation): adjacent CTAs become consecutive tiles of the *same* KV head, which share the causal KV prefix (tile t reads K/V[0..t], tile t+1 reads [0..t+1]) -> L2 reuse. This is the prefill analog of the proven fd2 Task-5 "lane-innermost L2 fix" (+2.7% decode). Pure grid/index remap -- **no math change, bitwise-safe.**

**Caveats:** CTA-to-SM scheduling order is not contractually x-fastest, so the reuse is a heuristic bet on launch-order locality; with only 4 KV heads the current mapping is especially reuse-hostile, which is why the remap is plausible. Measure the L2-hit-rate delta in the Task 1 ncu counters, not just wall.

**Gates:** canonical md5 unchanged (`4c4120c7...`, decode untouched); prefill bitwise-identity vs `Q27_ATTN_PF=lite` (grid remap must not perturb output); `test_kernels` ALL PASS; needle 3/3 @64K. **KEEP if 128K prefill wall improves at all with L2-hit measurably up; DROP if neutral/negative** (it's free to try and free to revert). Do this before Task 2 so the cp.async pipeline is built on the better-scheduled grid.

---

### Task 2: Phase 1 -- cp.async double-buffered K/V staging (BITWISE-SAFE)

**Files:** `src/prefill.cu` (`k_attn_prefill_mma`)
**Dependencies:** Task 1 confirms occupancy/barrier stall

> **MEASURED 2026-07-07 -- +5.4%, IMPLEMENTED + KEEP (corrected).** cp.async prefetch (fp8 path, `Q27_PF_CPASYNC` default on) is bitwise-identical (canonical 4c4120c7) and cuts the fp8 128K prefill wall 72.10s -> 68.20s = **+5.4%** (~+10% on the attention kernel). The FIRST pass measured "+0.2% neutral" but that used fp16 KV (default; `--kvstats` forbids fp8) where cp.async is dead code -- it never ran. Real number is on `Q27_KV=fp8 --pf 131072 --ctx 133120 Q27_PF_NOSERIAL=1`. Phase 1 is a keep. See docs/perf-attribution-prefill-attn.md.

**What:** the inner loop today is `__syncthreads -> stage K/V (global->smem, elementwise convert) -> __syncthreads -> QK^T MMA -> softmax -> PV MMA`, fully serializing the next tile's loads behind the current tile's compute with only 6 warps to hide either. Convert to a software pipeline: prefetch tile `p+1`'s K/V with `cp.async` (`cp.async.cg.shared.global`) into a second smem buffer while tile `p`'s MMAs run, `cp.async.wait_group` + `__syncthreads` at the boundary, ping-pong the two buffers. This is a **memory-scheduling change only** -- the MMA inputs, accumulation order, softmax, and outputs are bit-for-bit unchanged, so the fp16 path stays bitwise-identical.

**Subtlety (the one real correctness trap):** `cp.async` copies raw bytes; the current staging does a per-element *convert* (fp8/fp16 -> half in `kv2h`) during the copy. cp.async cannot convert. Two options -- pick per Phase-0 data:
- (a) **fp16 KV only:** cp.async the raw halves directly (already the smem type); the fp8 path keeps the existing convert-staging (no regression, just no Phase-1 speedup until Phase 2). Simplest, lowest risk, and Phase 2 makes fp8 the fast path anyway.
- (b) cp.async raw fp8 bytes into a staging sub-buffer, convert to half in a separate pass. More smem, more complexity; only if Phase 0 says fp8 prefill matters before Phase 2 ships.
Default to (a): cp.async the fp16 path, leave fp8 staging as-is, let Phase 2 own fp8.

**smem budget:** double-buffering K+V adds one more `[PP][LDH]` half pair = +2*32*264*2 = +33.8 KB, total ~116 KB. **Phase 0 measured the per-SM smem carveout admits only ONE 84.48 KB block already (Block Limit Shared Mem = 1), so ~116 KB stays at 1 CTA/SM** -- fine, because cp.async raises IPC within the existing 6 warps and does not need more occupancy. But do NOT exceed the sm_120 max-dynamic-smem cap (bump `cudaFuncAttributeMaxDynamicSharedMemorySize`; verify the launch still succeeds). If the extra buffer pushes past the cap, double-buffer K only (+17 KB) or reduce PP.

**Gates (all required):**
- **Prefill bitwise identity:** the fp16 path must stay bit-identical. Compare against `Q27_ATTN_PF=lite` (the reference `k_attn_prefill_T`) via the existing pf/pfcache identity check and a needle/nll-long A/B. For the fp16 MMA path specifically, dump attention output at a fixed prompt with cp.async on vs a `Q27_PF_CPASYNC=0` compile/runtime toggle and assert byte-equal.
- **Canonical md5** unchanged (`4c4120c7...`) -- decode untouched, must hold trivially; if it moves, something leaked outside the kernel.
- **test_kernels** ALL PASS (add a split=5-vs-1 and cp.async-on-vs-off unit A/B on the attention kernel at edge shapes T=23/base=37, matching the P1.5/P4 precedent).
- **needle** 3/3 @64K, verbatim-identical answers vs baseline.
- **Perf:** record attn kernel s and end-to-end 128K wall. Target from Phase-0 math: 39% -> 60-70% of peak, attn ~41 -> ~24-27 s, 128K wall ~71.5 -> ~54-57 s. **KEEP if attn kernel improves >=10%**; below that it's in the fd2-Task-5 marginal band and needs Gabe's call to keep the added smem/complexity.

**Keep `Q27_ATTN_PF=lite` and the pre-cp.async path reachable** (env or `#if`) for bisection.

---

### Task 3: Phase 2 -- fp8 K/V consumed directly in MMA (TOLERANCE-GATED)

**Files:** `src/prefill.cu` (`k_attn_prefill_mma`, fp8 template instantiation)
**Dependencies:** Task 2 (pipeline in place; fp8 becomes the fast staging path)
**Requires Gabe's explicit go** -- this is the expensive numeric-path kernel rewrite, the Phase-2 analog of the P14 "expensive rewrite needs explicit approval" rule.

> **2026-07-07 ATTEMPT (2a: fp8 QK^T only) -- REVERTED. Found a hard design conflict + two
> gotchas; captured here for a proper pass.** Implemented `mma.sync.m16n8k32.e4m3.e4m3.f32`
> for QK^T (K straight from `s_kraw`, Q cast half->e4m3; fragment k-layout: a/b regs hold
> `k=tg*4+{0..3}` and `+16`, accumulator layout identical to the f16 path so softmax/PV are
> untouched). It compiles on both arches (fp8 mma is sm_89+, guarded; sm_86 gets a no-op),
> and `fp8q` engages via the `--pf`/generate path (`[pfattn] szCT=1 use_mma=1 cpa=1 fp8=1`).
> **Blockers:**
> 1. **smem conflict (fundamental).** cp.async prefetches tile p+1 into `s_kraw` BEFORE the
>    MMA, but the fp8 QK^T reads `s_kraw` DURING the MMA -- so it reads the wrong tile (p+1).
>    The f16 path is immune (it reads the converted `s_k`). Double-buffering `s_kraw` to fix
>    it needs +16 KB -> 117 KB, over the 99 KB sm_120 cap. **cp.async-prefetch and
>    fp8-direct-read cannot coexist in smem as currently laid out.** The way out is to shrink
>    smem FIRST -- stage Q as fp8 (`s_q` 50.7 KB -> 25.3 KB), which frees room for a
>    double-buffered `s_kraw` AND is the step toward 2 CTAs/SM. So fp8-MMA must start from the
>    Q-fp8 + smem-relayout, not bolt onto the Phase-1 cp.async buffer.
> 2. **Test the prefill kernel via `--pf N --ctx >N`, NOT `-n`.** The normal `-n` generation
>    path prefills through a route that never calls `attn_prefill_launch` (diagnostic never
>    fired); only `generate()` (used by `--pf`) does. All greedy/logit A/Bs via `-n` are
>    meaningless for this kernel. For fp8-MMA correctness, use `--pf` without
>    `Q27_PF_NOSERIAL` (serial-vs-batched continuation gate) or add a logit dump to the `--pf`
>    path.
> 3. In 2a the K convert is still run (redundant when fp8q) -- neutral timing until removed.
>
> Revert hygiene: `src/prefill.cu` restored to HEAD (clean cp.async, +5.4%); canonical 4c4120c7
> re-verified. fp8-MMA is a from-the-smem-layout rewrite, gated on Gabe's go.

**What (the elegant part):** under `Q27_KV=fp8` the K and V caches are *already* E4M3 bytes. Today the kernel up-converts them to fp16 in smem and runs `mma.sync.m16n8k16.f16`. Blackwell (sm_120) supports fp8 MMA (`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`): consume the cached fp8 bytes directly in QK^T. This (a) doubles MMA throughput on the QK^T phase, (b) halves K staging smem and traffic (no half up-convert, and cp.async moves half the bytes), (c) with the freed smem, makes longer tiles feasible (see Task 4). Only Q (qk-normed, bounded -- the P2 kvstats probe measured K amax <=21.8, well inside E4M3 range) needs casting to fp8, and the P (softmax probs in [0,1]) x V phase can stay fp16-accumulate or also go fp8 depending on the tolerance result.

**Occupancy reality (Phase 0):** point (c) is necessary but NOT sufficient. Phase 0 measured **Block Limit Registers = 1** (248 regs/thread) alongside Block Limit Shared Mem = 1 -- so freeing smem does NOT reach 2 CTAs/SM until registers also drop to <=170 (2 blocks x 192 threads x 170 = 65280 <= 65536). The `o[32][4]` output accumulator is 128 regs/thread by itself; a 2-CTA occupancy win requires restructuring it (smem-spill O, or a narrower per-warp tile) TOGETHER with the smem shrink. Do not expect longer tiles / higher occupancy from fp8 smem savings alone. The fp8 MMA still pays for itself via (a) doubled QK^T throughput (attacks the measured 28% math_pipe_throttle stall) even at unchanged 12.5% occupancy -- treat that as the primary Phase 2 payoff, with 2-CTA occupancy a separate, register-gated sub-goal.

**Why it's tolerance not bitwise:** the QK^T integer/fp structure changes (fp8 mantissa in the MMA vs fp16). Precedent: P1.5's fp16 MMA prefill already runs a tolerance gate (3.8e-4 vs FA-lite) and P2's fp8 KV cache passed cosine 0.9995 / KL 3.4e-5 / PPL -0.05%. The K/V values are *identical bytes* to what the fp8-cache decode path already consumes losslessly; the only new error is Q-cast + the fp8 QK accumulate. Expect error at or below the P1.5 fp16-MMA tolerance.

**Gates (the full tolerance battery -- P1.5/P2 recipe):**
- **Unit A/B:** fp8-MMA QK^T vs the fp16-MMA reference on the dequantized cache, at edge shapes; assert max rel err below a pre-registered bound (start 5e-4, the P1.5 band; tighten if measured lower).
- **PPL:** full-corpus `--nll` delta < 0.1% vs the fp16 prefill (P2 spent -0.05%; budget is 0.1%).
- **nll-long:** `--nll-long 65536` buckets flat, within 0.3% of fp16 per bucket (the P2/P4 long-context gate).
- **needle:** 6/6 on the 361.5K haystack including the two beyond-native placements (the P2 deep gate), verbatim answers.
- **Prefix-cache identity:** pf vs pfcache must still agree under the fp8-MMA path (warm-turn correctness).
- **Canonical md5** unchanged -- decode path must not move (this touches prefill only; if the fp8 attention feeds a warm continuation, re-verify the warm-vs-cold gate).
- **fp16 path stays bitwise:** fp8-MMA is an opt-in under `Q27_KV=fp8`; the fp16 KV path keeps Task-2's bitwise identity. `Q27_ATTN_PF=lite` still falls back to the reference.
- **Perf:** stacked on Task 2, target attn ~24 -> ~13-16 s, end-to-end **128K ~54 -> ~46-48 s (~1.5x vs the 71.5 s baseline)**. The lever grows with context: at 256K native the attention share is ~70%, so this approaches ~1.9x there -- record a 256K point if VRAM allows.

**Do NOT proceed to a tolerance-gated commit if any correctness gate misses.** A PPL or needle regression here is a silent quality loss on every deep prompt; the whole project's discipline is that prefill never trades quality for speed without a passed gate + a re-derived canonical set.

---

### Task 4 (CONDITIONAL): longer tiles (TT 16 -> 32)

**Files:** `src/prefill.cu`
**Dependencies:** Task 3 shipped (fp8 shrinks smem enough to fit); Phase-0 said staging traffic/sync-count is still a top stall after Tasks 2-3
**Requires Gabe's go** (kernel retiling; folds into the Task 3 tolerance cycle if pursued together).

**What:** double the token tile so each K/V stage feeds twice the MMA work, halving the staging traffic and `__syncthreads` count per token. Blocked at fp16 Q (s_q alone at TT=32 is ~101 KB > budget); becomes feasible only after Task 3 removes the fp16 K staging and/or casts Q to fp8. Secondary lever -- only build if Phase 0 / post-Task-3 profiling shows staging still dominates. Same tolerance gates as Task 3 (it's the same numeric path, just retiled); if it stays fp16-Q it must hold bitwise.

**Measured-and-parked note:** if post-Task-3 profiling shows the kernel is now tensor-bound (staging hidden), SKIP and record it -- do not retile a kernel that's no longer staging-limited.

---

### Explicitly out of scope (park with reasons)

- **Block-sparse / sliding-window attention** (the P14 doc's own suggestion): changes model *output*, not just numerics -- it drops attention mass. Against this project's exactness culture that is an opt-in quality-gated *feature* (with needle + PPL + task-score gates), not a prefill perf lever. Park as a separate feature proposal; do not fold into this plan.
- **GEMM tile tuning / GDN:** length-flat per token (P14 Step 7); no quadratic decay to attack. The Amdahl ceiling for this plan is set by them: non-attention prefill is ~32.5 s real at 128K, so even *free* attention caps end-to-end at ~2.2x. The honest headline target is 71.5 s -> high-40s, not sub-30s.

---

## Success criteria (whole plan)

1. Phase 0 attribution doc with real ncu counters replacing the FLOP-derived 39% estimate.
2. `k_attn_prefill_mma` measured >=1.4x faster at 128K (attn kernel s), with fp16 path bitwise-identical and fp8 path inside the tolerance battery.
3. End-to-end 128K prefill wall in the high-40s s (from ~71.5 s), recorded no-profiler, stock mem, n=3 median.
4. Canonical md5 `4c4120c7...` unchanged throughout (decode never moved).
5. Every number has its exact command in `docs/perf-attribution-prefill-attn.md`; each task is one commit with a BUILDLOG entry; nothing pushed (Gabe reviews).

## Risk register (this plan)

- **cp.async correctness (Task 2):** the convert-during-stage trap -- mitigated by defaulting to the fp16-raw-copy path (option a) and leaving fp8 to Phase 2.
- **fp8 MMA precision (Task 3):** mitigated by the full P1.5/P2 tolerance battery and fp16 reference kept bitwise; do-not-commit-on-miss.
- **Occupancy regression from bigger smem (Tasks 2/4):** the added double-buffer/tile smem could drop below 1 block/SM and *lose* -- Phase 0 gives the smem-vs-occupancy curve; each task re-checks achieved occupancy and backs off buffering if it inverts.
- **Fixture drift:** the kvstats fixtures were already lost once (Task 0 Step 2); regen command goes in the BUILDLOG so the timing baseline is reproducible.
