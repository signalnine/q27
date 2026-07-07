# Verify-GEMV decode roofline investigation: plan

> **For Claude:** REQUIRED SUB-SKILL: Use conclave:executing-plans to implement this plan task-by-task.

**Goal:** Decide -- with a profiler, not a guess -- whether the batched verify GEMV
(`k_gemv_q4_n` / `k_gemv_q8_n`) has bandwidth headroom worth capturing, and if so, capture
it. This is the largest single decode lever by share: at 61K the weight-stream GEMVs are
**15.4 ms/round = 53% of the round** (batched verify + single draft;
docs/perf-attribution-p14.md Step 4), and the verify alone is ~42%.

**The honest ceiling (compute it before touching code).** The verify GEMV reads the full
model's Q4 weights once per round and reuses them across the N verify columns (weights in
registers, activations re-read per column -- see `kernels.cu:255-267`). So it is
overwhelmingly **weight-bandwidth-bound**, and N (verify width) barely changes the bytes.
Weight traffic/round ~= full model once (~14.8 GB) + ~4 MTP draft-head reads (~0.64 GB
each) ~= 17 GB; at 1.79 TB/s that is a **~9.7 ms/round DRAM floor**. Actual is 15.4 ms, so
the GEMVs run at **~63% of the weight roofline** -- there is ~37% headroom, worth up to
**~1.24x decode** if fully captured (a -5.7 ms round on a ~29 ms round). That is the
absolute ceiling; realistic capture is less, and the shared-mem-GEMV attempt already
regressed **-4% end-to-end despite positive microbenchmarks** (BUILDLOG) -- this headroom
is known-hard to reach.

**Two things tensor cores can and cannot do.** They cannot beat the ~9.7 ms weight-BW
floor (they raise the compute peak, not bandwidth; arithmetic intensity is ~12 MAC/byte at
N=6, far under any roofline ridge). They *can* help reach it **iff** Phase 0 shows the
current dp4a kernel is issue-, occupancy-, or latency-bound *below* the DRAM roofline --
i.e. dp4a cannot issue fast enough to saturate DRAM, and an MMA that does a whole tile per
instruction closes the gap. That is the entire question this plan answers.

**Architecture:** measure-first, safe-first. Phase 0 (ncu) is a hard go/no-go that can
terminate the plan. Phase 1 is cheap **bitwise-safe** kernel tuning (must hold the greedy
canonical). Phase 2 is the tensor-core skinny GEMM -- which **breaks greedy bitwise**
(int8 MMA cannot reproduce the dp4a fp-reduction structure; P1 established this for the
prefill GEMM), so it requires an explicit go, a re-derived canonical, and the full quality
battery. Branch `verify-gemv`.

**Tech Stack:** CUDA 13.2 C++ (sm_120 primary, sm_86 must compile), `mma.sync` m16n8k32
s8s8s32 (already implemented for prefill in `k_gemm_mma_T`), CUDA graphs, `ncu`/`nsys`,
Makefile (`make` full builds ONLY), test_kernels harness.

---

## Non-negotiable project rules (read before every task)

1. **Full `make` always.** Never `make <target>` -- stale sibling binaries. Tell: numbers identical to the previous run.
2. **Canonical gate** (exact recipe, verified 2026-07-06):
   ```bash
   CUDA_VISIBLE_DEVICES=0 ./build/q27 /mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27 \
     --tokens "760,6511,314,9338,369" --ctx 2048 --spec -n 128 2>/dev/null \
     | grep '^generated:' | md5sum
   ```
   must print `4c4120c72056aba2bc2d2561471eafce`. **This kernel is ON the greedy critical path** -- the verify GEMV computes the logits whose argmax decides token acceptance. Phase 1 MUST reproduce this md5 exactly. Phase 2 WILL change it (see its gate) and must re-derive + re-verify the full canonical set.
3. **GPU discipline:** `nvidia-smi` before any bench (vox-transcriber owns the 3090; the 5090 must show ~0 MiB used). Long jobs via `systemd-run --user --unit=<name> ...`. Sleep 10-15 s between back-to-back CLI model loads (VRAM teardown OOM race).
4. **Benchmarks in the FULL ENGINE, never microbench.** The load-bearing lesson of this plan: the shared-mem GEMV won a microbench and lost -4% end-to-end. Decode t/s from the server req_log `tps=` or CLI timing; stock memory OC; n=3 median; spread >2% => stale binary or contention, stop.
5. **Decode profiling recipe:** `ncu`/`nsys` on decode needs `Q27_PROF_DECODE=1` + `--cuda-graph-trace=node --capture-range=cudaProfilerApi` (the rig brackets the decode loop). NEVER profile a long `--tokens` CLI run (it prefills serially and hides the decode kernels). `ncu` needs `sudo -n`. Node tracing inflates wall ~2.3x -- use no-profiler runs for any recorded t/s.
6. **`-lineinfo` for source-correlated ncu** (`Makefile:5`) is a real help here but the Makefile is a gated file: surface it as a `SECURITY TRIGGERED` diff for approval before editing, don't add it silently.
7. **Commit style:** each task = one commit with a BUILDLOG entry (what/why/numbers). Do NOT push -- Gabe reviews and pushes.
8. **Determinism contract:** greedy bitwise-stable run-to-run. Phase 1 preserves it. Phase 2 explicitly renegotiates it (its own gate).

Model/tok: `/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27` + `.tok`.
Server bench: `CUDA_VISIBLE_DEVICES=0 Q27_PMIN=0.5 ./build/q27-server <model> <tok> --port 8081 --ctx 131072 --no-think` (production gate config; the verify widths this plan targets only appear with the gate ON).
Kernel: `k_gemv_q4_n<N>` at `src/kernels.cu:236` (dispatch `:330-335`, widths 2-6 + 10); Q8 twin at `:278`. It reads the **legacy** activation quant (`L.eo/xs/is`). The int8 MMA GEMM to reuse in Phase 2 is `k_gemm_mma_T` (prefill.cu, P1) and reads the **g64** quant (`nat64`).

---

### Task 0: Branch + baseline + a stable 61K decode-profiling harness

**Files:** none (git + fixtures + baseline).
**Dependencies:** none.

- Clean tree at HEAD `623cdb1`, 5090 free, `git checkout -b verify-gemv`.
- Regenerate the 61K bench fixtures if absent (the kvstats/prompt fixtures were cleaned once -- see the prefill-attn plan Task 0; `prompt61k.txt` / `toks60k.txt` per docs/perf-attribution-p14.md lines 20-27).
- Full build, `test_kernels` ALL PASS, canonical `4c4120c7...` exact.
- Baseline: capture the no-profiler decode t/s at 61K with the production gate config (n=3 median), and the per-kernel decode breakdown via the Q27_PROF_DECODE rig (this reproduces the 15.4 ms/round GEMV attribution -- if it doesn't land within ~5% of Step 4, the harness is wrong, STOP).
- Commit: doc-only (this plan + BUILDLOG "starting verify-gemv" entry).

---

### Task 1 (Phase 0): ncu the verify GEMV -- the go/no-go gate

**Files:** none (measurement + a new `docs/perf-attribution-verify-gemv.md`).
**Dependencies:** Task 0.

**This task decides whether Tasks 2-3 happen at all.** ncu `k_gemv_q4_n<5>` (and `<6>`) at
61K decode inside the full engine (Q27_PROF_DECODE rig), source-correlated if the gated
`-lineinfo` build is approved. Collect the Blackwell counters the review named:
- `dram__bytes_read.sum` and achieved DRAM throughput **vs peak (1.79 TB/s)** -- the roofline position.
- `smsp__issue_active` / issue-slot utilization -- is dp4a issue the limiter?
- `smsp__warp_issue_stalled_long_scoreboard` (and short-scoreboard) -- is it waiting on the per-column `__ldg` activation reads?
- Achieved occupancy + register/thread -- is it occupancy-limited (256 threads/block, 8 warps, row-per-warp)?

**Decision matrix (record the verdict in the attribution doc):**
- **Achieved DRAM BW >= ~85% of peak** -> the kernel is at the weight roofline; nothing (tensor cores included) can beat ~9.7 ms/round. **NO-GO. Kill Tasks 2-3, record, stop.** The 53% GEMV slice is simply the cost of reading the model at batch 1, and the decode lever is elsewhere (fewer rounds via acceptance -- see the perf review's item 5).
- **Achieved BW < ~70% of peak with issue/occupancy/scoreboard stalls dominating** -> real headroom below the roofline. **PROCEED to Task 2.** Note which stall class dominates -- it picks the Task 2 tactic (issue -> fewer/fatter ops; scoreboard -> prefetch/vectorize activations; occupancy -> register/geometry).
- **In between (70-85%)** -> marginal; Task 2 (cheap, bitwise-safe) is still worth one attempt, Task 3 (canonical-breaking) is NOT justified by the residual.

Cheapest, highest-information step in the plan; ~1 session on the existing decode rig, zero engine risk.

---

### Task 2 (Phase 1): cheap BITWISE-SAFE kernel tuning to approach the roofline

**Files:** `src/kernels.cu` (`k_gemv_q4_n` / `k_gemv_q8_n`).
**Dependencies:** Task 1 says PROCEED and names the dominant stall.

Before any canonical-breaking rewrite, spend the safe bites -- these keep the dp4a
fp-reduction structure, so **greedy stays bitwise** (canonical `4c4120c7...` must hold
exactly). Pick per Task 1's stall class:
- **Activation prefetch / wider loads** (if long-scoreboard on `__ldg(xp+u)`): the inner loop re-reads N columns' activations per weight chunk; stage or vectorize them, or hoist the per-column `xss[n]`/`iss[n]` scalars. The activation reorder must not change the fp reduction (accumulate order per output is fixed by the `n`/`u`/`ch` structure -- keep it).
- **Launch geometry / occupancy** (if occupancy-limited): rows-per-block, warps-per-block, `__launch_bounds__`. Pure scheduling, trivially bitwise.
- **Accumulator/register pressure** (if issue-bound): the `acc[N]` + 2-6 accumulator structure; tune without reordering the fp adds.

**Gates:** canonical md5 unchanged (bitwise); `test_kernels` ALL PASS; **full-engine** decode
t/s at 61K, n=3 median (NOT a microbench). **KEEP only if end-to-end decode improves >=2%**
(the microbench-win/engine-loss trap is exactly what killed the shared-mem attempt -- a
positive ncu number here is necessary but NOT sufficient; the engine t/s is the gate).
Record how much of the ~37% headroom Phase 1 captured -- that number decides whether Task 3
is even worth proposing.

---

### Task 3 (Phase 2): tensor-core skinny GEMM verify -- BREAKS GREEDY BITWISE

**Files:** `src/kernels.cu` (new MMA verify dispatch), `src/engine.cuh` (verify call site).
**Dependencies:** Task 1 PROCEED + Task 2 shipped + **Task 2 left a residual gap worth the cost**.
**Requires Gabe's explicit go AND explicit acceptance that greedy output changes.**

**What:** route the batched verify through an int8 MMA GEMM instead of the dp4a GEMV -- pad
the verify width N (4-6) to the MMA's N=8 tile and reuse the existing `k_gemm_mma_T`
machinery (P1 already unpacks Q4 nibbles to s8 with the -8 offset folded and runs
`mma.sync.m16n8k32.s8s8s32`). On a weight-BW-bound kernel the padded 3/8 wasted MMA compute
is free (weights read once regardless), and the MMA issues a whole tile per instruction, so
if Task 1 found dp4a *issue* the limiter, this is what closes to the roofline.

**The synergy with perf-review item 2b:** the MMA GEMM reads the **g64** activation format,
which `qxT` already computes every step (`engine.cuh:1390`) and which item 2b calls "wasted
in decode." Switching decode verify to MMA makes that g64 quant **used** -- so do NOT drop
the decode-side g64 requant if this task ships; the two items are coupled.

**Why it breaks bitwise (state it plainly):** P1 found the MMA accumulator cannot reproduce
dp4a's per-32-block fp multiply-add structure without infeasible register cost -- the
integer dots are exact, but the fp reduction across chunks reorders (~1e-6). On the verify
path that ~1e-6 can flip an argmax on a rare tie, changing which draft tokens are accepted,
changing greedy output. So the canonical `4c4120c7...` **will move**. This is the project's
most sacred invariant; do not treat the re-derivation as a formality.

**Gates (the full battery, plus a re-derived canonical):**
- **Tolerance unit A/B:** MMA verify logits vs dp4a verify logits at edge shapes, max rel err below a pre-registered bound (P1 band ~1e-6 on the integer-exact dots; the fp-reorder is the only source).
- **Re-derive the canonical:** produce the new greedy md5 under the MMA verify, commit it as the new reference with the diff from `4c4120c7...` documented; prove the change is pure fp-reorder (not a logic bug) by showing the token stream diverges only where logit margins are within fp noise.
- **Quality battery:** full-corpus `--nll` PPL delta < 0.1% vs dp4a; needle 3/3 @64K; a task-score A/B (Thunderdome/CC greedy) showing no regression -- because this changes *accepted tokens*, a t/s win with a score loss is a NO-GO.
- **Spec correctness:** acceptance rate (tok/round) must not drop -- if the MMA verify changes acceptance materially, the round-count effect can erase the per-round speed win (measure tok/round, not just ms/round).
- **Perf:** full-engine 61K decode t/s, n=3 median. **KEEP only if end-to-end decode improves >=5%** AND no quality-gate regression. Keep the dp4a `k_gemv_q4_n` as a runtime fallback (`Q27_VERIFY=dp4a`) for bisection and as the bitwise-greedy reference path.

If the quality battery shows any regression, this is a **NO-GO on principle** regardless of
speed -- the engine does not trade greedy correctness for decode t/s.

---

## Adjacent cheap levers (tracked here, not the plan spine)

- **Perf-review item 2b (quant phase-split).** `qxT` (`engine.cuh:1386-1390`) unconditionally
  runs both `quantize_x` (legacy) and `quantize_x_g64`. In **prefill** the MMA GEMM consumes
  only g64, so legacy is wasted (~1-2% of the quant bucket); symmetrically in **decode** the
  GEMV consumes legacy, so g64 is wasted -- **unless Task 3 ships**, which makes decode g64
  used. So: phase-split the prefill side freely (drop legacy when `Q27_PREFILL != dp4a`); on
  the decode side, gate the drop on whether Task 3 is active. The author's "always-fresh so
  dispatch is safe" comment is the invariant to preserve -- assert no live legacy consumer
  before dropping. Modest, do after Task 1's ncu quantifies the quant bucket.
- **Perf-review item 2c (prefill-attn grid remap for L2)** lives in the prefill-attn plan
  (docs/plans/2026-07-07-prefill-attn.md), not here -- it is a prefill-attention change.
- **Perf-review item 5 (acceptance-tuning).** If Task 1 says NO-GO (GEMV at roofline), the
  decode lever is fewer rounds, not faster rounds -- i.e. proposal-temperature / context-aware
  gating to raise tok/round. That is a separate research plan and must wait on the
  CUDA-review #2/#3 sampler fixes (they corrupt the acceptance traces). Note the pointer;
  don't fold it in.

---

## Success criteria

1. Task 1 attribution doc with the roofline verdict (achieved DRAM BW vs peak, dominant
   stall class) -- this alone is a valuable, cheap output even if it kills the plan.
2. If PROCEED: Task 2 captures some headroom **bitwise** (canonical held), measured
   end-to-end (not microbench).
3. Task 3 only if the residual justifies breaking bitwise, and only through the full quality
   battery + a re-derived, documented canonical.
4. Every number has its exact command in `docs/perf-attribution-verify-gemv.md`; each task
   one commit + BUILDLOG entry; nothing pushed.

## Risk register

- **Microbench win / engine loss** (the shared-mem GEMV precedent): mitigated by rule 4 --
  every keep-decision is a full-engine n=3 median, ncu is necessary-not-sufficient.
- **Task 3 breaks greedy correctness silently:** mitigated by the re-derived canonical +
  task-score A/B; a t/s win with a score loss is an explicit NO-GO.
- **Acceptance-rate regression** eats the per-round win: measure tok/round, not just ms/round.
- **NO-GO is a real outcome:** if Task 1 shows the GEMV at the roofline, the honest result is
  "decode is weight-bound at batch 1, this lever is closed" -- record it and redirect to
  acceptance (item 5). Do not force a rewrite past a roofline verdict.
