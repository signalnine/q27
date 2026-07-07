# P14: Performance Levers Bundle -- Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use conclave:executing-plans to implement this plan task-by-task.

**Goal:** Close the tuned-llama depth gap (190 t/s @75.5K vs q27 145.6 pre-P12) by porting the P12/P13 confidence gate to the production sampled path, adding draft-side early exit, fixing fd2 cross-lane KV scheduling, fusing the draft argmax+margin scans, and re-attributing the post-fd2 round -- all under the repo's canonical/tolerance gate discipline.

**Architecture:** Six code tasks + one measurement task + one decision brief, sequential on one branch (`p14-perf-levers`) because they share hot files (engine.cuh, blocks.cu, spec3.cu) and one GPU. Bitwise-safe changes (Tasks 2,3,4,5) land under the existing canonical md5 `4c4120c7...`; fp-order changes (Task 5b/6, conditional) batch into one tolerance-gate + canonical-re-derive cycle per the g64/fd2 precedent. Measurement (Task 1) runs first and produces the go/no-go data for the conditional tasks.

**Tech Stack:** CUDA 13.2 C++ (sm_120 primary, sm_86 must also compile), CUDA graphs, Makefile (`make` full builds ONLY), test_kernels harness, nsys/ncu profiling.

---

## Non-negotiable project rules (read before every task)

These come from the BUILDLOG's paid-for lessons. Violating them has burned whole sessions.

1. **Full `make` always.** Never `make <target>` -- target-scoped builds leave stale sibling binaries. The tell for a stale binary: numbers identical to the previous run.
2. **Canonical gate** (exact recipe, verified 2026-07-06 on this branch):
   ```bash
   CUDA_VISIBLE_DEVICES=0 ./build/q27 /mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27 \
     --tokens "760,6511,314,9338,369" --ctx 2048 --spec -n 128 2>/dev/null \
     | grep '^generated:' | md5sum
   ```
   must print `4c4120c72056aba2bc2d2561471eafce`. (Token IDs inline -- no tokenizer arg; `tools/sampling_gate.sh` is the reference implementation and its gate 1 is an acceptable substitute.) An EMPTY-input md5 (`d41d8cd9...`) means the run failed -- check stderr, usually the back-to-back VRAM teardown race (sleep 12-15s and retry). Any task below marked CANONICAL-EXACT must reproduce this md5 unchanged. `Q27_FD=v1` reproduces the pre-fd2 kernel (old canonical `58b6ae85...`) -- useful for bisecting attention changes.
3. **GPU discipline:** `nvidia-smi` before any bench (vox-transcriber owns the 3090; the 5090 must show ~0 MiB used). Long jobs (>2 min) via `systemd-run --user --unit=<name> /usr/bin/bash -c "cd /mnt/ai/projects/q27 && <cmd>"` -- session crashes kill setsid jobs. Units are transient: `systemctl --user stop` deletes them; `reset-failed` after crashes. Sleep 10-15s between back-to-back CLI model loads (VRAM teardown OOM race).
4. **Benchmarks:** memory OC must be STOCK for any recorded number. Decode t/s from the server req_log `tps=` field or CLI timing output; never nsys-inflated walls (node tracing inflates wall 2.3x -- use no-profiler runs as ground truth). n=1 t/s is near-deterministic for greedy decode; scores are NOT (basins) -- this plan only measures t/s, never task scores.
5. **nsys decode profiling:** needs `--cuda-graph-trace=node --capture-range=cudaProfilerApi` with `Q27_PROF_DECODE=1` (the rig brackets the decode loop). Never profile a long `--tokens` CLI run (CLI prefills serially). `ncu` needs `sudo -n`.
6. **Lane-count landmine:** any batched kernel selecting per-lane outputs by explicit pointers (`t==3?n3:n4`) instead of struct indexing corrupts silently when widths change (the quantize3 bug). Grep for this pattern whenever touching widths.
7. **Commit style:** each task commits with a BUILDLOG entry (`docs/BUILDLOG.md`, chronological section at the end) recording what/why/numbers. Do NOT push -- Gabe reviews and pushes (public repo).
8. **Determinism contracts:** greedy output must be bitwise-stable run-to-run. Sampled output must be seeded-reproducible (same seed -> same bytes). Never introduce atomics that reorder fp adds on any path feeding tokens.

Model/tokenizer paths: `/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27` + `.tok`.
Server bench template (single-slot is fine for benches):
`CUDA_VISIBLE_DEVICES=0 Q27_PF_XG=32 ./build/q27-server <model> <tok> --port 8081 --ctx 131072 --no-think --fast-head`

**Measurement rigor (applies to every bench in this plan):** fixed seed (`--seed 42`) for anything sampled; one discarded warmup run, then n=3 and report the median (decode t/s is near-deterministic; if spread >2%, suspect a stale binary or GPU contention and stop); compare only same-day, same-binary-class numbers; record the exact command next to every number in the attribution doc.

**When a gate fails (bisect recipe):** each task is one commit, so `git bisect` across the branch with the canonical md5 as the test script settles which task broke it. Within a task, narrow with the config knobs before touching code: `Q27_PMIN` unset (gate off), `Q27_DEXIT=0` (monolithic draft), `Q27_FD=v1` (pre-fd2 attention -- rules attention in/out), `Q27_SAMPLE_PLAIN=1` (plain vs spec sampling), `Q27_PMIN=100` (forces width-2, the P12b narrow-width probe). A divergence that survives `Q27_FD=v1` and appears only at a specific width is a lane-count landmine (rule 6). Do not fix forward past a canonical mismatch.

---

### Task 0: Branch + baseline sanity

**Files:** none (git + build only)
**Dependencies:** none

**Step 1:** Confirm clean tree at `b5c5ea0` and no other session is active:
```bash
cd /mnt/ai/projects/q27 && git status --short && git log --oneline -1
nvidia-smi --query-compute-apps=pid,name --format=csv
ls -lt src/ | head -3   # mtimes must predate this session
```
Expected: clean tree, HEAD b5c5ea0, 5090 free.

**Step 2:** `git checkout -b p14-perf-levers`

**Step 3:** Full build + kernel tests:
```bash
make 2>&1 | tail -3 && ./build/test_kernels 2>&1 | tail -5
```
Expected: clean build sm_86+sm_120, ALL PASS.

**Step 4:** Canonical gate (rule 2). Expected md5: `4c4120c72056aba2bc2d2561471eafce`.

**Step 5:** Commit nothing; this is a checkpoint only.

---

### Task 1: Measurement pass -- post-fd2 attribution + go/no-go data

**Files:**
- Create: `docs/perf-attribution-p14.md`
**Dependencies:** Task 0

No product code changes. This produces the decision data for Tasks 5/5b/6/7. All numbers into the doc as they land, with exact commands.

**Step 1: Workload prep.** Read the CLI arg parsing in `src/engine.cu` (search `--tokens-file`) to learn its format (added in P12 for >128KB prompts). Build a ~60K-token prompt file from repo docs (deterministic: `cat docs/SPEC.md docs/BUILDLOG.md docs/FORMAT.md ...` repeated, tokenized to the file format the flag expects) and a ~75K variant. Store under the session scratchpad, record the recipe in the doc.

**Step 2: Ground-truth decode walls (no profiler), greedy.** Server up (bench template above), then `/v1/completions` continuations at 2K / 16K / 61K wikitext-style prompts + the 60K tokens-file via CLI. Record ms/round + t/s + tok/round for: ungated, `Q27_PMIN=0.5`, `Q27_PMIN=1.0`. Expected ballpark from BUILDLOG: 61K ungated ~126 t/s serving / 60K CLI ~90 -> ~100 gated.

**Step 3: Sampled-vs-greedy head-to-head (never measured).** Same 60K prompt, same server: greedy vs `temperature=0.7, top_p=0.95` (or `Q27_FORCE_TEMP=0.7 Q27_FORCE_TOP_P=0.95` env if per-request fields are inconvenient), n=256 decode. Record t/s + tok/round both. This isolates the sampled-path tax (nucleus kernels are 5 single-block full-vocab scans/round -- Task 3 shrinks the count, a multi-block nucleus rewrite is a possible follow-on if the tax is >=3%).

**Step 4: nsys round attribution @61K.** Per rule 5, server-side, `Q27_PROF_DECODE=1`, both greedy-ungated and greedy-gated(theta=1.0), ~200+ rounds. Produce a per-kernel ms/round table: gemv_q4_n/q8_n, k_attn_fd2 + combine, delta_step/conv_step/gdn_gates, rmsnorm*, quantize*, argmax*/margin, memcpys, and (sampled run) nucleus/gumbel/spec_accept. Sum must close within ~5% of the no-nsys wall delta method (per-node overhead cancels between configs).
Deliverable: the definitive post-fd2 round budget (the pre-fd2 one at BUILDLOG:664 is stale).

**Step 5: fd2 cross-lane KV traffic (gates Task 5/6).**
```bash
sudo -n ncu --kernel-name regex:k_attn_fd2 --launch-count 24 \
  --metrics dram__bytes_read.sum,lts__t_sector_hit_rate.pct,gpu__time_duration.sum \
  ./build/q27 <model> <tok> --tokens-file <60K file> -n 8 --spec
```
The launches MIX draft (grid z=1) and verify (grid z=5) instances -- ncu reports the grid per launch; compute R **from the z=5 verify instances only** (draft instances have no cross-lane question). Compare per-instance `dram__bytes_read` against 1x KV-slice (lanes L2-shared) vs ~5x (lanes re-stream DRAM). KV bytes/layer at 61K fp16 = 2*4*256*61440*2B ~= 252MB (fp8 half that; note which the CLI ran). Record the ratio R = measured / 1x.
**Go/no-go:** R >= 2 -> Task 5 (axis swap) GO. R < 1.3 -> Tasks 5 and 6 SKIP (L2 already absorbs it); record and move on.

**Step 6: Draft-attention share (gates Task 5b).** From the Step 4 table: ms/round of ntok=1 fd2 instances (the 4 draft passes). >= 1.5ms/round @61K -> Task 5b candidate; else SKIP.

**Step 7: Prefill decay attribution @128K.** `--kvstats 131072` run (needs `--nll` int32 tokens; synthetic tokens fine -- prefill timing is value-independent; it prints timing before the fp8 refusal) under nsys. Bucket kernel time: prefill attention vs delta_wy/delta scan vs GEMM vs quantize/other, at 16K vs 128K. Deliverable: which term drives 3180 t/s @16K -> ~1830 @128K. No optimization in this plan -- attribution only (feeds a future plan).

**Step 8:** Write `docs/perf-attribution-p14.md` with all tables + the go/no-go matrix (Tasks 5, 5b, 6, 7 + nucleus-rewrite follow-on). Commit: `git add docs/perf-attribution-p14.md && git commit -m "P14: post-fd2 decode + prefill attribution, go/no-go matrix"`

---

### Task 2: Fused draft argmax+margin (`k_argmax_top2`)

Kills the 4-5 dead single-block 248320-wide `k_margin` scans baked into the DEFAULT (ungated) `spec_graph`, and halves full-vocab draft reductions when gated. CANONICAL-EXACT.

**Files:**
- Modify: `src/blocks.cu` (new kernels after k_argmax_extract, ~line 302), `src/blocks.cuh` (declaration)
- Modify: `src/engine.cuh:594-595, 725-752` (spec_draft_launches)
- Modify: `src/engine.cuh` scratch alloc near line 275 (`d_draft_margin`)
- Test: `src/test_kernels.cu` (follow the existing per-kernel test pattern; find the k_margin test first)

**Dependencies:** Task 0 (Task 1 can be in flight; no file overlap)

**Step 1: Write the failing test** in test_kernels.cu: random logits (n=248320, plus crafted cases: all-equal ties, max at index 0, max at last index, duplicated max values):
- fused token == existing `argmax()` token (exact int equality -- tie semantics must match: per-thread strict `>` keeps lowest index per thread, packed-u64 max keeps highest index across threads; DO NOT change either)
- fused margin == CPU exact top1-top2 (exact float equality; selection only, no fp arithmetic besides the final subtract)
- fused margin == existing `margin()` output on the same buffers

**Step 2:** `make && ./build/test_kernels` -- expected: new test FAILS (symbol missing).

**Step 3: Implement.** In blocks.cu:

```cpp
// P14: fused top-2 for the draft path -- one full-vocab pass produces the
// argmax token (bitwise-identical tie semantics to k_argmax: per-thread
// strict >, packed-u64 max across threads/blocks) AND the top1-top2 margin
// (P12 gate signal). Replaces k_argmax + k_margin on the draft path; both
// stay in-tree (verify path / tests).
__global__ void k_argmax_top2(const float* __restrict__ x, int n,
                              unsigned long long* __restrict__ blk1,
                              float* __restrict__ blk2) {
    float v1 = -FLT_MAX, v2 = -FLT_MAX; int i1 = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float xi = x[i];
        if (xi > v1) { v2 = v1; v1 = xi; i1 = i; }
        else if (xi > v2) { v2 = xi; }
    }
    __shared__ unsigned long long s1[256];
    __shared__ float s2[256];
    s1[threadIdx.x] = am_pack(v1, i1);
    s2[threadIdx.x] = v2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            unsigned long long a1 = s1[threadIdx.x], b1 = s1[threadIdx.x + s];
            float a2 = s2[threadIdx.x], b2 = s2[threadIdx.x + s];
            // top1: packed max, identical lattice to k_argmax's reduction.
            // top2: k_margin's pairwise merge (blocks.cu:328-331) on unpacked values.
            float a1v = am_unpack_val(a1), b1v = am_unpack_val(b1);
            s1[threadIdx.x] = max(a1, b1);
            s2[threadIdx.x] = (a1v >= b1v) ? fmaxf(a2, b1v) : fmaxf(b2, a1v);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { blk1[blockIdx.x] = s1[0]; blk2[blockIdx.x] = s2[0]; }
}
```
with the exact inverse of `am_pack`'s monotonic map (am_pack: `u = (u & 0x80000000u) ? ~u : (u | 0x80000000u)`):
```cpp
__device__ __forceinline__ float am_unpack_val(unsigned long long p) {
    unsigned u = (unsigned)(p >> 32);
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u);
}
```
Subtlety the merge must respect: when a1v == b1v (duplicated max value), `(a1v >= b1v)` takes the a-side branch and b1v feeds the top2 candidate -- same as k_margin, so margin==0 for duplicated maxima. The test's duplicated-max case pins this.
```cpp
__global__ void k_top2_finalize(const unsigned long long* __restrict__ blk1,
                                const float* __restrict__ blk2, int nblk,
                                int* __restrict__ tok, float* __restrict__ margin_out) {
    // single block <=256 threads: same pairwise (top1,top2) merge over nblk
    // entries; write tok = low 32 bits of global max pack; margin = v1 - v2.
}
```
Host wrapper `argmax_margin(x, n, d_tok, d_margin, d_blk1, d_blk2, st)` launching `k_argmax_top2<<<128,256>>>` + `k_top2_finalize<<<1,128>>>`. Write the real merge code (the sketch above marks the two invariants that make it bitwise: pack/unpack must invert `am_pack`'s monotonic map exactly, and the top1 lattice must be the same packed-u64 max as k_argmax so tie-breaks are identical).
Allocate `d_am_blk1` (128 u64) + `d_am_blk2` (128 float) beside `d_draft_margin` (engine.cuh ~line 275).

**Step 4:** In `spec_draft_launches` (engine.cuh:725-752): replace each `argmax(mtp_logits, VOCAB, d_draft{k}, d_amax, stm)` + `margin(mtp_logits, VOCAB, d_draft_margin+k, stm)` pair with one `argmax_margin(mtp_logits, VOCAB, d_draft{k}, d_draft_margin+k, d_am_blk1, d_am_blk2, stm)`. Keep the mtp_forward / h_next D2D structure untouched. Note the ordering: margin for draft k must land in `d_draft_margin[k-1]`? -- NO: preserve the existing slot mapping exactly (read lines 736-750 carefully; margins are indexed 0..4 for drafts 1..5).

**Step 5:** `make && ./build/test_kernels` -- ALL PASS.

**Step 6: Canonical gate** (rule 2) -- md5 EXACT `4c4120c7...`. Also gated identity: run the 16K prompt with `Q27_PMIN=0.5` before/after this change (same binary can't do "before" -- use the previous commit's binary or compare against Task 1's recorded gated tok/round + t/s): emitted text identical, round count identical (margins are value-identical so caps are identical).

**Step 7: Bench sanity:** 2K shortbench within noise; 61K server t/s within noise (this task removes work; expect 0 to +1%).

**Step 8: Commit** with BUILDLOG entry: `git add -A && git commit -m "P14: fuse draft argmax+margin into one full-vocab pass (k_argmax_top2)"`

---

### Task 3: Port the P12 gate + gated-round plumbing to the sampled spec path

Production is cleared to default T<=0.7 sampling; today `spec_sample_round` (engine.cuh:1097) launches the full-width graph unconditionally, so default traffic misses the +10.8%@60K gate. CANONICAL-EXACT (greedy untouched); sampled realization SHIFTS (allowed -- rejection walk truncates differently), seeded reproducibility must hold.

**Files:**
- Modify: `src/blocks.cu:554-590` (k_spec_accept + wrapper: add `int max_draft`), `src/blocks.cuh`
- Modify: `src/engine.cuh`: `spec_verify_launches_sampled` (822-831), `build_spec_graphs` (943-979 area), `spec_sample_round` (1097-1111), graph member decl near line 159
- Test: `src/test_kernels.cu` (--sampling-only section)

**Dependencies:** Task 2 (margins now come from the fused kernel; the draft graphs this task reuses must be final)

**Step 1: Write the failing kernel test:** extend the existing spec-accept test (find it via `grep -n spec_accept src/test_kernels.cu`) to cover `max_draft` in {1,2,3,4}: (a) the walk never accepts more than max_draft drafts; (b) all-accept at max_draft=m yields stop_lane==m (bonus lane m); (c) empirical accept rate per lane still equals p_served under each cap (the Phase-2 property).

**Step 2:** `make && ./build/test_kernels --sampling-only` -- new cases FAIL (signature mismatch first; fix call sites as you go).

**Step 3: Kernel change.** `k_spec_accept` (blocks.cu:554): add `int max_draft` param; `stop_lane` init becomes `max_draft` (not 4); loop `for (k = 0; k < max_draft; k++)`. Wrapper `spec_accept` passes it through. All-accept semantics: lanes 0..max_draft-1 are drafts, lane max_draft is the bonus draw -- exactly the width-5 behavior when max_draft=4.

**Step 4: Per-width sampled verify.** `spec_verify_launches_sampled` (engine.cuh:822): nucleus loop `k < vw` (not 5); `spec_accept(..., vw - 1, ...)`. `k_finish_sampled` needs no change (keys on n<=vw; src select covers n in 1..5). Verify `logits2`/`d_nuc` indexing stays lane-relative (it does -- offsets are `k*VOCAB` / `k*4`).

**Step 5: Graph capture.** In `build_spec_graphs`, after the sampled monolithic set (971-979): capture `verify_sample_graph_w[6][6]` for W=2..5 (`dmax=4; vw=W;` then capture `spec_verify_launches_sampled()`), mirroring the greedy per-width loop at 950-958. New member `cudaGraphExec_t verify_sample_graph_w[6][6] = {};` beside `verify_graph_w`. Update the "spec graphs captured" fprintf to mention sampled gated widths.

**Draft-graph selection rule for sampled+gated (fixed here, not at implementation time):** the sampled tail is 4-draft, so the sampled+gated path ALWAYS launches a depth-4 draft graph. `draft_graph[perm]` is depth-4 only when `gate_maxd==4`; therefore in `build_spec_graphs`, capture `draft_graph_lo[perm]` whenever `gate_maxd == 5` (today it is captured only under `maxd_auto` -- widen that condition; one extra graph per perm, no new buffers). The round then uses:
```cpp
cudaGraphExec_t dg = (gate_maxd == 5) ? draft_graph_lo[perm] : draft_graph[perm];
int md_used = 4;  // sampled ceiling is 4 this phase; cap <= 4 by construction
```
A depth-4 draft graph writes margins [0..3] -- indexing is consistent with the cap loop unchanged. No `min(cap,4)` clamp is needed once the draft itself is depth-4; assert it anyway (`cap <= 4`).

**Step 6: Round logic.** `spec_sample_round`: mirror `spec_round`'s gated branch:
```cpp
if (pmin_theta > 0.f) {
    int md_used = 4; // sampled tail is 4-draft; under Q27_MAXD=auto use draft_graph_lo
    cudaGraphExec_t dg = (maxd_auto || gate_maxd == 5) ? /* depth-4 graph: draft_graph_lo
        under auto; under fixed Q27_MAXD=5 there is no lo graph -- capture one, or
        simpler: sampled+gated forces the depth-4 draft graph; read the capture code
        and pick the cleanest correct option, documenting it */ : draft_graph[perm];
    CUDA_CHECK(cudaGraphLaunch(dg, stm));
    // margins -> cap exactly as spec_round:1027-1032, but clamp cap <= 4
    CUDA_CHECK(cudaGraphLaunch(verify_sample_graph_w[W][perm], stm));
} else {
    if (samp_first) { ... existing bootstrap ... }
    CUDA_CHECK(cudaGraphLaunch(spec_sample_graph[perm], stm));
}
```
Keep the `samp_first` bootstrap correct on the gated branch too (first token samples from retained prefill logits BEFORE any spec round -- study lines 1098-1101 and replicate). P13 EMA: do NOT update sat/yield EMAs from sampled rounds this phase (sampled ceiling is 4; document as limitation in the BUILDLOG entry).

**Step 7:** `make && ./build/test_kernels --sampling-only` -- ALL PASS (no model needed; runs even if a server holds the GPU).

**Step 8: Live gates** (GPU, server down):
- Canonical EXACT (greedy path untouched).
- `tools/sampling_gate.sh` PASS (existing seeded-identity / seed-varies / sampled!=greedy gates).
- NEW: sampled+gated reproducibility: same prompt, `--temp 0.7 --top-p 0.95 --seed 42 Q27_PMIN=1.0`, 2 runs -> byte-identical. 3rd run seed 43 -> differs.
- NEW first-token bootstrap gate: gated (`Q27_PMIN=1.0`) vs ungated sampled runs at the SAME seed must emit an IDENTICAL FIRST token -- token 0 comes from the retained prefill logits via `sample_g` kind 0 on both paths, before any spec round. A differing first token means the gated branch broke the `samp_first` bootstrap (the known silent-bug class here).
- Accept-vs-temp band: `[sample-stats]` at T=0.7 gated should stay ~3.4+ tok/round (gate trims low-confidence tails; large regression = bug).

**Step 9: Perf gate.** 60K tokens-file, T=0.7: `Q27_PMIN=1.0` vs unset. Expect >= +7% t/s (greedy analog was +10.8%; narrower widths also skip nucleus scans, so it may exceed it). Record in the attribution doc. 2K: neutral within 1.5%.

**Step 10: Commit** with BUILDLOG entry: `"P14: P12 confidence gate on the sampled spec path (per-width sampled verify graphs, capped accept walk)"`

---

### Task 4: Draft early-exit (per-step draft graphs) -- the other half of p_min

llama's p_min stops DRAFTING (llama.cpp common/speculative.cpp:332); q27's gate only narrows verify while drafts always run to gate_maxd (~1.5ms/pass). Split the draft into per-step graphs and stop at the first sub-theta margin. CANONICAL-EXACT; token AND round-count identical to the monolithic gated path (cap semantics unchanged: leading run >= theta).

**Files:**
- Modify: `src/engine.cuh` only: refactor `spec_draft_launches` (725-752) into a per-step function; capture `draft_step_graph[5][6]` in `build_spec_graphs`; early-exit loop in `spec_round` + `spec_sample_round` gated branches.

**Dependencies:** Task 3 (both gated branches get the loop; capture code is shared)

**Step 1: Verify the staleness proof against the code (proof already derived; re-check, don't re-derive).** The concern: skipped drafts skip their mtp_k/mtp_v row writes. The proof that this is safe, from k_prep_round (spec3.cu:511-527) and mtp_forward/attn_block (engine.cuh:562-596, 475-500):

1. Draft k's position is `pos_m{k} = P + k` (prep_round), its input token OCCUPIES row P+k, and inside `attn_block` the `kv_store` at that row executes BEFORE `attn_decode` reads rows 0..P+k (engine.cuh:493-494). So within a round, every draft writes its own row before attending it, and attends no row beyond its own.
2. Early-exit runs drafts 1..cap+1 (the FIRST FAILING draft still runs -- its margin is what stops the loop -- and therefore still writes row P+cap+1). Rows written this round: P+1..P+cap+1. The verify commits n <= cap+1 tokens, so the new base P' = P+n <= P+cap+1: **every committed position's row is written**.
3. The rows early-exit skips (P+cap+2..P+dmax) are exactly rows > P'+1's predecessors... more precisely: next round's draft j sits at P'+j and attends rows 0..P'+j; rows P'+1..P'+j are freshly written by that round's drafts 1..j (point 1) BEFORE any read, and rows <= P' are committed rows covered by induction. The monolithic path's "extra" rows P+cap+2..P+dmax hold speculative K/V that verify rejected -- they are likewise overwritten by a future draft before any draft attends them. So the read-set of every future attention is IDENTICAL under early-exit and monolithic. No new staleness class exists.
4. The one pre-existing gap class (full-accept bonus: n = W commits position P+W whose row no draft wrote, dmax=4 path) is UNCHANGED by this task -- it happens only when cap==dmax, where early-exit and monolithic are the same.
5. Corollary: draft token streams, caps, round grouping, and emitted tokens are all EXACTLY identical to the monolithic gated path -- which is why Step 5 gates round-COUNT identity, not just token identity, and why any round-count divergence means the implementation (not the design) is wrong.

Re-check each cited line against HEAD before coding; if any indexing differs from the above (e.g. kv_store ordering, pos_m derivation), STOP and report rather than adapting the design silently.

**Step 2: Refactor for identical capture.** Extract `spec_draft_step_launches(int k)` = { k==0: prep_round + first mtp_forward + fused argmax_margin; k>0: h_next D2D + mtp_forward k + fused argmax_margin } such that `spec_draft_launches()` becomes `for (k = 0; k < dmax; k++) spec_draft_step_launches(k);` and the recorded kernel sequence of the monolithic graphs is byte-identical to before the refactor (this keeps every existing graph, and the canonical, untouched). Build + canonical EXACT before proceeding.

**Step 3: Capture per-step graphs.** In the perm loop of `build_spec_graphs`: `draft_step_graph[k][p]` for k=0..gate_maxd-1 (capture each step alone). Under `maxd_auto`, steps 0..4 exist; `draft_graph_lo` becomes redundant for the gated path but LEAVE IT (constrained path + non-early-exit fallback use it).

**Step 4: Early-exit loop.** In both gated branches (Q27_DEXIT env, default ON when pmin_theta>0; `Q27_DEXIT=0` restores the monolithic-draft behavior for A/B):
```cpp
int cap = 0;
for (int k = 0; k < md_used; k++) {
    CUDA_CHECK(cudaGraphLaunch(draft_step_graph[k][perm], stm));
    CUDA_CHECK(cudaMemcpyAsync(h_draft_margin + k, d_draft_margin + k, 4,
                               cudaMemcpyDeviceToHost, stm));
    CUDA_CHECK(cudaStreamSynchronize(stm));
    if (h_draft_margin[k] < pmin_theta) break;
    cap++;
}
```
then the existing W floor + per-width verify launch. Worst case adds md_used-1 extra syncs/round (~15us each) vs one -- noise against a >=20ms round; each skipped draft saves ~1.5ms.

**Step 5: Identity gates.** For theta in {0.5, 1.0} x ctx {2K, 16K, 60K} x {greedy, sampled seed 42}: `Q27_DEXIT=1` vs `Q27_DEXIT=0` -> emitted bytes identical AND round counts identical (round count printed in CLI stats / server req_log). Canonical EXACT (ungated path never enters the loop).

**Step 6: Perf gate.** 60K docs tokens-file, greedy theta=1.0: expect >= +3% over Task-3-era gated (that prompt's margins fail often -- that is where drafts get skipped). 2K neutral (+-1%). High-acceptance agentic-style prompt: neutral (nothing to skip). Record all three in the attribution doc.

**Step 7: Commit** with BUILDLOG entry: `"P14: draft early-exit -- margin-gated per-step draft graphs (p_min draft-stop parity with llama)"`

---

### Task 5: fd2 lane-innermost grid scheduling (bitwise L2 fix)

**GATED on Task 1 Step 5: run only if R >= 2** (verify lanes re-stream KV from DRAM). The verify attention grid is `dim3(n_kv_heads, FD2_NS, ntok)` (spec3.cu:444); z (lane) is the SLOWEST axis, so lane t+1's blocks schedule after lane t has streamed the whole KV slice -- zero cross-lane L2 reuse by construction. Swapping lane to the fastest axis co-schedules the vw same-split blocks onto the same ~1MB KV chunk. Pure index remap: per-lane fp order unchanged -> bitwise. CANONICAL-EXACT.

**Files:**
- Modify: `src/spec3.cu` (k_attn_fd2 blockIdx mapping + its launch in attn_decode3_fd2 wrapper, ~317-456)
- Test: `src/test_kernels.cu` (fd2 section already exists)

**Dependencies:** Task 1 (go/no-go), Task 4 (bench baseline current)

**Step 1:** Read k_attn_fd2's blockIdx usage (spec3.cu:317-372). Change launch to `dim3(ntok, FD2_NS, n_kv_heads)` and the kernel's axis reads to match (`t = blockIdx.x; split = blockIdx.y; h = blockIdx.z;`). Touch NOTHING else -- scratch indexing, per-block work, and loop order stay identical. Check whether the ntok=1 draft launches share the wrapper (they do -- same remap applies, harmless at z=... x=1).

**Step 2:** `make && ./build/test_kernels` -- the fd2 gates (fd2-vs-v1 tolerance, run-to-run bitwise, default-dispatch bitwise) must PASS unchanged. Add one assertion if absent: post-change fd2 output bitwise vs pre-change fd2 on the seq {1,47,1024,16384,61440} x ntok {1,5} matrix (generate reference with the previous commit's test binary if needed).

**Step 3:** Canonical EXACT.

**Step 4: Bench.** 61K server greedy ungated + 60K CLI gated: keep if >= +3% at 61K; revert the commit if < +1.5% (record either way in the attribution doc -- a negative here validates the L2 measurement methodology). Expect the win to grow at 128K-class ctx: also record the 75K tokens-file number.

**Step 5: Commit** (or revert + BUILDLOG negative entry): `"P14: fd2 lane-innermost grid order -- cross-lane KV L2 reuse"`

---

### Task 5b (conditional): FD2_NS retune for ntok=1 draft attention

**GATED on Task 1 Step 6** (draft-attn >= 1.5ms/round @61K). Draft attention launches only 4*FD2_NS*1 = 512 blocks (~3/SM). Making the split count ntok-dependent (e.g. FD2_NS_1 = 512 for ntok==1) changes the combine merge count -> fp ORDER changes -> NOT bitwise: tolerance-gate class, canonical re-derive required. **Batch with Task 6 if it also fires** (one re-derive cycle) -- otherwise weigh whether the measured ms justify a canonical re-derive alone; if not, record SKIP in the attribution doc and leave for a future fp-order batch. Gates when pursued (the fd2/g64 policy, docs/attn-fd2-design.md:42-56): unit tolerance vs double reference, run-to-run bitwise, PPL within noise of 7.1889/7.1928, --nll-long 160K bucket-flat, acceptance parity ~2%, NEW canonical md5 derived in the landing commit.

---

### Task 6 (conditional, requires explicit go from Gabe): fd2 lane-pair fusion

**GATED on:** Task 1 R >= 2 AND Task 5 shipped but total attn still >= ~2x its BW floor at 61K. This is the expensive kernel rewrite; Task 5 may capture most of the win for free. **Do not start without Gabe's explicit approval on the Task 5 results.**

Design constraints (write `docs/attn-fd3-design.md` first, mirroring attn-fd2-design.md):
- Pair lanes per block (grid `dim3(ceil(ntok/2), NS, 4)`): 2x s_q (12.3KB smem), 2x acc[6][8] = 96 acc regs/lane -- est ~150 total regs. Occupancy floor (hard gate, verified with `-Xptxas -v` on a skeleton BEFORE writing the full kernel): <= 168 regs/thread AND >= 3 resident blocks/SM at 128 threads (>= 12 warps/SM) -- below either bound, the pair design loses fd2's latency-hiding win and must be redesigned (narrower acc ownership or NW retune), not shipped anyway.
- Per-lane online-softmax sequence kept in the lane's own registers in the SAME position order -> bitwise vs fd2 achievable; if the compiler forces spills that change nothing semantically, still bitwise.
- Odd ntok: last block runs single-lane (guard, not a separate kernel).
- Kill criteria: <5% at 61K, or short-bench regression >2%, or any fd2 unit gate fails -> revert, keep the design doc + BUILDLOG negative.

---

### Task 7: gate_maxd 6-8 decision brief (no code)

**Files:**
- Create: `docs/maxd6-decision.md`
**Dependencies:** Tasks 1, 4 (needs post-early-exit draft economics)

Using measured data ONLY (no estimates where a measurement exists): post-Task-4 cost of one draft step (attribution table), per-lane verify cost (width sweep from the per-width graphs: time gated rounds at forced caps), and the acceptance distributions (BUILDLOG:371 burst data: 92-94% chain survival to d10, agentic mean chains 6.2-7.6). Compute the breakeven for gate_maxd=6 under early-exit + width-gated verify (the old +6.5ms/depth economics assumed ungated fixed depth). Include the implementation checklist if GO: S_spare6/ring_spare6 (+~155MB), perm mod-7 (7 graph sets everywhere -- audit every `% 6`), width-7 gemv instantiation + warm, logits2/nuc/scratch sizing, the quantize3-class lane audit (`grep -n "t==3\|t == 3\|?n3:n4" src/*.cu*` plus manual review of every batched kernel's per-lane selection), and the P12b bisect recipe (theta=100 width test) for when it diverges. Recommendation + numbers to Gabe; implementation is a separate plan.

Commit the doc.

---

### Task 8: Docs sync + wrap-up

**Files:**
- Modify: `README.md` (State section: sampled-path gate, early-exit, any fd2 win; retire the "sampled path misses the gate" gap), `docs/BUILDLOG.md` (P14 summary entry if the per-task entries need a capstone)
**Dependencies:** all landed tasks

**Step 1:** README State/roadmap sync with measured numbers (honest framing per the red-team precedent: n=1 t/s deltas are fine, no score claims).
**Step 2:** Graph-variant inventory: the engine now carries spec_graph, draft_graph, draft_graph_lo, draft_step_graph, verify_graph, verify_graph_w, spec_sample_graph, verify_sample_graph_w, sample_graph (x6 perms each). Add a short "graph zoo" comment block above the members in engine.cuh (which path uses which, and which are now redundant-but-kept) so the next width/depth change doesn't miss one. If any variant is genuinely dead after this bundle (e.g. the monolithic gated draft use-case), note it as removable in the BUILDLOG entry -- do not remove it in this plan.
**Step 3:** Full `make && ./build/test_kernels` + canonical gate one last time on the branch tip.
**Step 4:** Commit. Leave the branch UNPUSHED and UNMERGED -- summarize the merge decision for Gabe (merge-to-master + push is his call on this public repo).

---

## Execution notes

- Execute tasks strictly in order 0,1,2,3,4,(5),(5b),(6),7,8; conditionals per their gates.
- One Opus 4.8 subagent per task, fresh context each; the orchestrator reviews the diff + gate outputs between tasks (conclave:subagent-driven-development).
- Every subagent reads this plan's "Non-negotiable project rules" section plus its own task before touching anything.
- If a gate fails: stop, report, do not "fix forward" past a canonical mismatch -- bisect with the P12b recipe (narrow width/theta first, then Q27_FD=v1 to rule attention in/out).
- If the GPU is busy (vox-transcriber is 3090-only and fine; anything on the 5090 is not), stop and report rather than queueing behind it.
