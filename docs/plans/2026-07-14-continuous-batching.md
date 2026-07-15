# Continuous Batching P0+P1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use conclave:executing-plans to implement this plan task-by-task.

**Goal:** Batch the verify-forward weight sweep across 2-4 concurrent decode requests so one weight read serves all active slots (design: `docs/plans/2026-07-14-continuous-batching-design.md`).

**Architecture:** P0 refactors the engine's eager round body (`spec_round_launches` and callees) into lane-view-parameterized helpers with zero behavior change, gated by canonical md5 + byte-identity replay. P1 adds a conductor thread that runs a fused eager verify forward over the union lane set (per-lane pointer structs filled from multiple engines), with per-engine mixer/tail sub-launches, behind `Q27_BATCH=1`.

**Tech Stack:** CUDA C++ (sm_120/sm_86 dual-arch), plain Makefile, no test framework -- gates are md5 comparisons, kernel self-tests, and shell scripts.

---

## Context primer (read first, zero-context engineer)

- Repo: `/mnt/ai/projects/q27`, branch `continuous-batching` off master `11191a0`.
- Model for ALL gates/benches: `/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27` (+`.tok`).
  NEVER benchmark with qwopus (standing rule).
- Build: `make` (targets `build/q27`, `build/q27-server`, `build/test_kernels`, ...).
  A full rebuild is ~5-10 min; incremental is fine. `build/q27-server-w16` is a
  hand-built `-DQ27_W_MAX=16` variant -- do not break its build.
- THE CANONICAL GATE (run after every engine-touching commit):
  ```
  cd /mnt/ai/projects/q27 && CUDA_VISIBLE_DEVICES=0 build/q27 \
    /mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
    --tokens "760,6511,314,9338,369" -n 128 --ctx 2048 --spec 2>&1 \
    | grep '^generated:' | md5sum
  ```
  Expected: `a2982c5197c627551b27d76a0a94b220`. Any other value = STOP, the
  change is not behavior-preserving.
- Kernel self-test: `build/test_kernels` -> must end `ALL PASS`.
- Sanitizer (dual-arch build REQUIRED, sm_120-only throws fatbin noise):
  full path, not on PATH: `sudo -n /usr/local/cuda/bin/compute-sanitizer ...`
  (run at phase ends, not every commit).
- The GPU must be free (check `nvidia-smi`); a `q27-eval` systemd unit may be
  serving on :8081 -- if so, ask before stopping it.
- Long-running benches: launch via `systemd-run --user` (session crashes kill
  plain background jobs).
- Key source geography (line numbers at branch point `11191a0`):
  - `src/engine.cuh:1236 spec_round_launches()` -- eager round body; graph
    capture records it (`build_spec_graphs`, :1360-1420). It runs eagerly at
    warm-up (:1356), so eager execution is already exercised.
  - `:1159 spec_verify_forward()` -- embed3 -> N_LAYER x {rmsnorm3,
    attn_pair|gdn_pair, add3, rmsnorm3, ffn_pair, add3} -> output_norm ->
    qx5 -> head `mm5` into `logits2`.
  - `:998 gdn_pair(il)`, `:1039 attn_pair(il)`, ffn_pair below it.
  - `:953 qx5()` / `:962 mm5()` -- the union points. `mm5` picks `k_vgemm`
    (flat-in-W MMA GEMM) at `vw >= gemm_min` (9) else the dp4a GEMV family.
  - `:1193 spec_verify_launches()` = forward + greedy tail (argmax_masked x vw
    + finish_round). `:1220 spec_verify_launches_sampled()` = same forward +
    sampled tail. DO NOT fork the forward: both tails share it by design.
  - `:1507 spec_round(int* emit)` -- host round driver: gated dexit draft loop
    (:1561-1616), verify graph launch, outcome D2H (:1622), host bookkeeping.
  - `vw` (:944) is a HOST member read at capture/launch time = the verify
    width. `gemm_min` guardrail at :1284 aborts if the ladder ever reaches
    the GEMM path -- respect it.
  - Per-lane pointer structs: `P3/CP3` (`kernels.cuh:67`), `XQ3` (:69),
    `IP3` (`spec3.cuh:13`), `FCP3/FIP3` (`fdmma.cuh:33`). 16 slots, lanes are
    already independent in addressing. `LANESW(name)` macro builds a struct
    from member arrays `name_L`.
  - Server: `src/server.cu` -- slots (:276), GpuGate use (:869), generate call
    sites for 3 API shapes. `GpuGate` in `src/api_common.h:74`.
- Style: match surrounding code -- dense comments explaining WHY, measured
  numbers in comments, no C++ exceptions in kernels path, CUDA_CHECK on every
  CUDA call.
- SENSITIVE FILE RULE: any `Makefile` edit requires printing
  `SECURITY TRIGGERED:` + the diff and waiting for explicit `APPROVED` from
  Gabe. Tasks below avoid Makefile edits by compiling new test binaries with
  explicit nvcc/g++ commands inside gate scripts; a Makefile target bundle is
  offered once at the end.

---

## P0 -- lane-view refactor (zero behavior change)

### Task 1: LaneView struct + solo view builder

**Files:**
- Modify: `src/engine.cuh` (near :940, above `qx5`)

**Dependencies:** none

**Step 1: Add the struct and builder (no call sites yet)**

Add above `qx5()`:

```cpp
    // Continuous-batching P0: everything the verify forward reads that is
    // PER-LANE, gathered behind one view so the fused (cross-engine) round
    // can point slot k at any engine's lane buffers. Solo path: solo_view()
    // returns a view over this engine's own lanes -- pointer-identical to
    // the member arrays, so behavior is unchanged by construction.
    // Mixer state (RBuf/SBuf roles, kcache, d_pos scalars) is deliberately
    // NOT here: mixers stay per-engine (design 2026-07-14).
    struct LaneView {
        std::array<float*, W_PLUMB> x1, qkv, z, alpha, betar, g, beta,
            convout, o, og, y, qg, kbuf, vbuf, h, lg;
        q27k::XQuant xq[W_PLUMB];
        q27k::IP3 vtok;               // verify_tokens(): embed sources
        q27k::WIP3 pos;               // per-lane position ptrs (rope)
        std::array<int*, W_PLUMB> dv; // per-lane argmax outputs (greedy tail)
        float* vgemm_ws;              // workspace for k_vgemm (>= union width)
        int vw;                       // live width THIS round
        cudaStream_t stm;             // stream the round runs on
    };
    LaneView solo_view() {
        LaneView v{};
        v.x1 = x1_L; v.qkv = qkv_L; v.z = z_L; v.alpha = alpha_L;
        v.betar = betar_L; v.g = g_L; v.beta = beta_L; v.convout = convout_L;
        v.o = o_L; v.og = og_L; v.y = y_L; v.qg = qg_L; v.kbuf = kbuf_L;
        v.vbuf = vbuf_L; v.h = h_L;
        for (int t = 0; t < W_PLUMB; t++) {
            v.xq[t] = xq_L[t];
            v.lg[t] = logits2 + (size_t)(t < W_MAX ? t : 0) * VOCAB;
            v.dv[t] = d_v_L[t];
        }
        v.vtok = verify_tokens();
        v.pos = lane_pos();
        v.vgemm_ws = d_vgemm_ws;
        v.vw = vw;
        v.stm = stm;
        return v;
    }
```

Check the actual member array names against the `LANESW` uses in
`spec_verify_forward`/`gdn_pair`/`attn_pair`/`ffn_pair` before committing --
the list above was read from :998-1190 but ffn_pair's arrays and any h_L
naming must be verified in source. Add/remove fields to exactly cover what
Tasks 2-4 need; nothing more (YAGNI).

**Step 2: Build**

Run: `make 2>&1 | tail -3`
Expected: compiles clean (struct is unused so far).

**Step 3: Commit**

```bash
git add src/engine.cuh && git commit -m "P0: LaneView + solo_view() -- per-lane state behind one view (no call sites yet)"
```

### Task 2: Parameterize qx5/mm5 on the view

**Files:**
- Modify: `src/engine.cuh:953-996` (qx5, mm5)

**Dependencies:** Task 1

**Step 1: Change signatures**

`qx5(const std::array<float*,W_PLUMB>& x, int cols)` becomes
`qx5(const LaneView& v, const std::array<float*,W_PLUMB>& x, int cols)`:
read `v.xq[i]` instead of `xq_L[i]`, launch on `v.stm` with `v.vw`.
`mm5(const DevTensor& w, const std::array<float*,W_PLUMB>& ys_a)` becomes
`mm5(const LaneView& v, ...)`: `v.xq`, `v.vgemm_ws`, `v.vw`, `v.stm`.
Update ALL call sites (gdn_pair/attn_pair/ffn_pair/spec_verify_forward and
any draft-path callers -- grep `mm5(` and `qx5(`) to pass `solo_view()`.
Build a `const LaneView sv = solo_view();` ONCE at the top of
`spec_verify_forward()` and thread it through the pair helpers as a
parameter (`gdn_pair(il, sv)` etc.) rather than rebuilding per call.
NOTE: if the draft path (`mtp_forward`) also calls mm5/qx5, give it the same
treatment -- same view, `vw` as it uses today. Verify with grep, do not
assume.

**Step 2: Build + canonical gate + test_kernels**

Run: `make 2>&1 | tail -3 && build/test_kernels 2>&1 | tail -1`
then the canonical gate command (header).
Expected: `ALL PASS`; md5 `a2982c5197c627551b27d76a0a94b220` EXACT.

**Step 3: Commit**

```bash
git add src/engine.cuh && git commit -m "P0: qx5/mm5 read lane state via LaneView (solo view; canonical EXACT)"
```

### Task 3: Split pair helpers into pre/mix/post on the view

**Files:**
- Modify: `src/engine.cuh:998-1098` (gdn_pair, attn_pair, ffn_pair)

**Dependencies:** Task 2

**Step 1: Mechanical split, launch order UNCHANGED**

- `gdn_pre(il, v)`: :1000-1016 (qx5, mm5 qkv, mm5 gate, gemv_f16 alpha/beta,
  gdn_gates3) -- all reads via `v`.
- `gdn_mix(il)`: :1022-1030 verbatim (conv chain, l2norm3, delta chain) --
  MEMBER state (RBuf/SBuf, member stm, member vw). Untouched code.
- `gdn_post(il, v)`: :1031-1036 (gated_norm3, qx5 og, mm5 out).
- `gdn_pair(il, v) { gdn_pre(il, v); gdn_mix(il); gdn_post(il, v); }`.
- Same treatment for `attn_pair` (pre = q/k/v mm5s + rmsnorm_heads + rope3;
  mix = turbo3 rotate + kv_store + attention + combine, member-based;
  post = o-proj mm5) and `ffn_pair` (entirely pre -- keep one function,
  rename param only). READ THE CURRENT BODIES FIRST; the exact boundary is
  "everything touching RBuf/SBuf/kcache/attention scratch = mix".
- gotcha: `gdn_mix` uses member `vw` while pre/post use `v.vw`. In solo these
  are equal. Add an assert in `gdn_pair`: `assert(v.vw == vw ||  /* fused */ true)`
  -- actually: document instead, asserts in hot path are noise; the fused
  driver (P1 Task 8) never calls the composed `gdn_pair`, only pre/mix/post
  separately.

**Step 2: Build + gates (same as Task 2 Step 2)**

Expected: `ALL PASS`; canonical EXACT.

**Step 3: Rebuild the w16 variant and gate it**

Run: the same nvcc command that produced `build/q27-server-w16` (check
`docs/BUILDLOG.md` part-15/16 or `git log --grep=w16` for the exact command;
it is `make`'s q27-server rule with `-DQ27_W_MAX=16`).
Expected: compiles. (Its runtime gate happens at P1 Task 12.)

**Step 4: Commit**

```bash
git add src/engine.cuh && git commit -m "P0: gdn/attn/ffn pairs split pre(view)/mix(member)/post(view), launch order unchanged (canonical EXACT)"
```

### Task 4: Parameterize the verify forward + tails on the view

**Files:**
- Modify: `src/engine.cuh:1159-1234`

**Dependencies:** Task 3

**Step 1: Thread the view**

`spec_verify_forward(const LaneView& v)`: embed3 uses `v.vtok`/`v.h`;
rmsnorm3/add3 build their P3/CP3 from `v.h`/`v.x1`/`v.y`; head mm5 writes
`v.lg`. Greedy tail `spec_verify_launches()`: argmax loop reads
`v.lg[t]`/`v.dv[t]` but keeps member `d_mask_pool/d_mask_ids/d_amax` and
`finish_round` exactly as-is (tail is per-engine forever). Sampled tail:
same -- forward takes the view, tail members untouched.
`spec_verify_launches()` becomes `spec_verify_launches(const LaneView& v)`
called with `solo_view()` from `spec_round_launches()`.

**Step 2: Build + canonical + test_kernels + sampled smoke**

Canonical gate as usual, PLUS a sampled-path smoke (the sampled graphs
capture the same forward):
```
CUDA_VISIBLE_DEVICES=0 build/q27 /mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27 \
  --tokens "760,6511,314,9338,369" -n 64 --ctx 2048 --spec --temp 0.7 --top-p 0.95 --seed 42 \
  2>&1 | grep '^generated:' | md5sum
```
Record the md5 BEFORE this task (at Task 3's commit) and compare: must be
identical (sampled path is deterministic at fixed seed).

**Step 3: Server replay byte-identity gate (the P0 exit gate)**

Fresh `build/q27-server`, replay `scratchpad/accept_payload_codegen.json`
4x at the accept_ab.sh config, on THIS commit vs the branch point:
```
bash scratchpad/dexit_ab.sh   # or a trimmed 1-payload variant; compare
# per-replay completion text md5s vs the same run at 11191a0 (git stash /
# checkout master build in build-master/ if needed -- simplest: run the
# gate ONCE at branch point now, save md5s to scratchpad/p0_baseline.md5)
```
Do the baseline capture FIRST (before Task 1) if not already done --
see Task 0 note below. Expected: byte-identical completions.

**Step 4: Commit**

```bash
git add src/engine.cuh && git commit -m "P0: verify forward + tails on LaneView; P0 exit gates green (canonical, sampled-seed, replay byte-identity)"
```

### Task 0 (do before Task 1): capture P0 baselines

**Files:** create `scratchpad/p0_baseline/` (gitignored)

**Dependencies:** none -- FIRST thing on the branch.

At `11191a0`: run the canonical gate, the sampled-seed smoke (Task 4 cmd),
and one server replay (codegen payload, accept_ab config, 4 replays) saving
all completion bodies + md5s into `scratchpad/p0_baseline/`. These are the
compare targets for Tasks 2-4. Commit nothing (scratchpad is ignored).

---

## P1 -- conductor + fused verify

### Task 5: N-invariance kernel gate (vgemm + gemv + gemv_f16)

**Files:**
- Create: `tools/ninv_test.cu`
- Reference: `tools/vgemm_test.cu` (harness pattern: host ref, device run,
  bitwise compare), `src/vgemm.cuh`, `src/kernels.cuh`

**Dependencies:** none (parallel with P0)

**Step 1: Write the failing-by-construction test**

For each kernel family {`vgemm_verify` (q4 + q8 weight legs),
`gemv_q4_n`, `gemv_q8_n`, `gemv_f16_3`}: fill lanes with fixed random
activations; run once at T=N1 with payload lanes in slots {0,1,2} and once
at T=N2>N1 with the SAME payload lanes in different slots (e.g. {1,4,7}),
padding other lanes with junk (not zeros -- junk proves isolation); compare
payload-lane outputs BITWISE across the two runs. Cover
(N1,N2) in {(2,5),(3,9),(5,12),(9,16)} x both weight dtypes. Exit nonzero
with a per-case diff count on any mismatch, print `NINV ALL PASS` on
success. Keep the harness style of vgemm_test (no framework).

**Step 2: Build + run (explicit command, no Makefile edit)**

```
nvcc -O2 -std=c++17 -arch=sm_120 -I src tools/ninv_test.cu src/vgemm.cu src/kernels.cu \
  -o build/ninv_test    # check vgemm_test's actual link line in Makefile and mirror it
build/ninv_test
```
Expected: `NINV ALL PASS`. IF ANY CASE FAILS: stop, report which kernel/width
-- the determinism contract downgrades per design ("bitwise-when-untrimmed"
holds only for kernels that pass; a gemv failure means batched rounds must
force the vgemm path or the contract note in the design doc gets amended).
This is a finding, not a blocker for throughput.

**Step 3: Commit**

```bash
git add tools/ninv_test.cu && git commit -m "P1: N-invariance gate for vgemm/gemv/gemv_f16 lanes (bitwise-when-untrimmed contract)"
```

### Task 6: Trim policy, pure host function + unit test

**Files:**
- Create: `src/conductor.h` (header-only, like `depthctl.h`/`toolconstrain.h`)
- Create: `tools/test_conductor.cpp`

**Dependencies:** none

**Step 1: Write the failing test first**

`tools/test_conductor.cpp` (CPU-only, style of `tools/test_depthctl.cpp`):
cases for `q27::trim_widths(int* want, bool* is_suffix, int k, int cap)`:
- fits: {4,5} cap 12 -> unchanged
- overflow trims widest first: {8,7} cap 12 -> {6,6} or {5,7}? DEFINE:
  repeatedly decrement the current widest (ties: suffix lanes first, then
  higher slot index) until sum <= cap. {8,7} cap 12 -> {6,6}.
- suffix-first: want {12(sfx), 6} cap 12 -> {6,6} (suffix absorbs all trim
  before the gated lane loses any).
- floor: no lane below 2 (engine floor "no width-1 gemv"); k=4 all-suffix
  {12,12,12,12} cap 16 -> {4,4,4,4}.
- k=1 never trims (solo bypasses fusion anyway).

**Step 2: Run to verify it fails**

```
g++ -O2 -std=c++17 -Wall -I src tools/test_conductor.cpp -o build/test_conductor && build/test_conductor
```
Expected: FAIL to compile (trim_widths undefined).

**Step 3: Implement `trim_widths` in src/conductor.h, minimal**

**Step 4: Run: expected ALL PASS. Commit**

```bash
git add src/conductor.h tools/test_conductor.cpp && git commit -m "P1: trim policy (widest-first, suffix-before-gated, floor 2) + CPU unit test"
```

### Task 7: DecodeTask extraction from generate()

**Files:**
- Modify: `src/engine.cuh` (generate(), :2400-2500 region -- read the whole
  function first)

**Dependencies:** Task 4

**Step 1: Read generate() end to end. Map its decode-loop state**

Everything the loop carries across rounds (emit budget/n_max, dec counters,
EOS/stop state, on_token/on_round_gap hooks, suffix bookkeeping, tc hooks)
becomes `struct DecodeTask { ... }`. The loop body becomes
`bool decode_step(DecodeTask& t)` = one spec_round/spec_sample_round + the
per-token host bookkeeping; returns false when generation is done.
`generate()` = prefill/setup + `DecodeTask t = make_decode_task(...); while (decode_step(t));` + teardown.
NO functional edits in the same commit as the extraction.

**Step 2: Build + canonical + sampled-seed + replay byte-identity (P0 gates)**

All three must be EXACT vs `scratchpad/p0_baseline/`.

**Step 3: Commit**

```bash
git add src/engine.cuh && git commit -m "P1: generate() decomposed into prefill + DecodeTask/decode_step loop (byte-identical, P0 gates green)"
```

### Task 8: Fused verify round (the core)

**Files:**
- Modify: `src/conductor.h` (union view builder + fused round driver)
- Modify: `src/engine.cuh` (expose the needed engine internals to the
  conductor: pre/mix/post, tails, LaneView pieces -- prefer a small
  `friend struct q27::Conductor;` or public accessors, match project taste)

**Dependencies:** Tasks 3, 4, 6

**Step 1: Union view builder**

```cpp
// conductor.h -- slot k of the union view points at (engines[m], lane j).
// Engine lane buffers are written IN PLACE by the union sweep; mixers and
// tails then read their own members untouched. vgemm_ws: engines[0]'s (any
// engine's is sized for W_MAX >= union cap).
```
`build_union_view(Engine** es, const int* w, int k) -> {LaneView, LaneMap}`
where LaneMap records (engine, lane) per union slot for scatter-free
bookkeeping. Positions/vtok/xq/lg/dv per slot come from the owning engine's
solo_view() fields at its lane index.

**Step 2: Fused round driver**

`fused_verify_round(Engine** es, int* granted_w, int k, cudaStream_t cstm)`:
1. per-engine: drafts + width decision ALREADY DONE by caller (Task 9);
   each engine's draft work ran on ITS OWN stm; record event per engine,
   `cudaStreamWaitEvent(cstm, ev_e)` for each.
2. build union view (vw = sum granted_w, stm = cstm).
3. eager fused forward: embed3(union) then per layer:
   `rmsnorm3(union); if attn: {for e: attn_pre? NO}` -- careful: pre is
   UNION (weight ops), mix is PER-ENGINE:
   `pre(il, uview); for e: e->mix_on(il, cstm); post(il, uview);`
   `gdn_mix`/`attn_mix` currently launch on member stm at member vw -- add a
   `(cudaStream_t st, int width, int lane0)` variant? NO -- lane0 offset is
   wrong: each engine's mix reads its OWN member lane arrays 0..w_e-1, which
   the union sweep wrote in place. Only the STREAM must be overridable:
   give mix helpers an explicit stream param defaulting to member stm, and
   a width param defaulting to member vw (the granted width this round --
   set member vw = granted_w[e] before the round; that is also what tails
   read). So: `for e: { e->gdn_mix(il, cstm); }` with e->vw pre-set.
4. union output_norm + qx5 + head mm5 with per-slot lg -> each engine's own
   logits2 lanes.
5. per-engine tails on cstm: `e->spec_verify_launches_tail(...)` -- factor
   the tail (post-forward part) of spec_verify_launches/_sampled into
   callable pieces during this task (mechanical, gate with canonical).
6. per-engine outcome D2H (their pinned/pageable oc as today) + ONE
   `cudaStreamSynchronize(cstm)`.
7. record event on cstm; each engine's NEXT draft (own stm) waits on it.
CRITICAL union-sweep precondition: every engine's lane buffers for lanes
0..w_e-1 are engine-owned and distinct -- assert no engine appears twice.

**Step 3: Two-engine smoke (CLI-level, no server)**

Write `tools/fused_smoke.cu` (or extend an existing tool): construct 2
engines on one DeviceModel (the server already proves this pattern), prefill
two SHORT prompts (canonical tokens + a second fixed prompt), run N=32
tokens each: leg A solo-sequential (existing generate), leg B conductor
fused rounds. Compare emitted token ids per stream: byte-identical when no
trim occurred (widths at ctx 2048 stay <= 6, union <= 12, no trim).
Build with explicit nvcc line mirroring q27's link (loader/tokenizer/
device_model/kernels/spec3/vgemm/prefill objects -- copy the Makefile q27
rule's file list).
Expected: `FUSED SMOKE PASS: streamA identical, streamB identical`.

**Step 4: Run test_kernels + canonical (solo path untouched proof). Commit**

```bash
git add src/conductor.h src/engine.cuh tools/fused_smoke.cu && \
git commit -m "P1: fused verify round -- union weight sweep + per-engine mixers/tails; 2-engine smoke byte-identical to solo"
```

### Task 9: Conductor thread + registration + token queues

**Files:**
- Modify: `src/conductor.h` (Conductor class: registry, round loop, queues)
- Test: extend `tools/test_conductor.cpp` (membership bookkeeping, CPU-only:
  fake round fn via std::function, verify join-at-round-boundary /
  leave-on-done / solo-fallthrough sequencing)

**Dependencies:** Tasks 7, 8

**Step 1: Failing CPU test for the scheduler skeleton**

Test Conductor<FakeEngine> (template on engine type, duck-typed: needs
`draft_and_gate() -> want_width`, `set_granted(w)`, `finish_round() -> done`)
: 3 fake engines join at staggered "rounds"; assert every round's member
set changes only at boundaries; a member marked done leaves before the next
round; single member -> `solo_round()` called, not fused.

**Step 2: Implement Conductor (minimal for the test): run + commit**

**Step 3: Real wiring**

- registration: `Conductor::register_task(Engine*, DecodeTask*, TokenSink)`;
  request thread blocks on a `TokenQueue` (mutex+cv, vector<int> batches +
  done flag + error slot).
- conductor thread: owns the fused round loop; per round: for each member
  run the host side of `decode_step` split: draft+gate (per-engine, own stm,
  dexit host loop as today), trim (Task 6) -> set granted vw per engine,
  fused_verify_round (Task 8), per-member host bookkeeping (the rest of
  decode_step: emit tokens -> queue, EOS/n_max -> done), GpuGate
  acquire/release around the GPU section, maybe_yield between rounds
  (prefill interleave preserved).
- tool-constrain hooks: run on the conductor thread exactly where
  decode_step runs them today (they are engine methods; the queue only
  carries tokens OUT).
- kill switch: `Q27_BATCH` env, default OFF. When off, conductor is never
  constructed; server behaves exactly as at 11191a0.

**Step 4: Build; canonical; commit**

```bash
git add src/conductor.h tools/test_conductor.cpp && git commit -m "P1: conductor thread, registration, token queues, gate interplay (Q27_BATCH=1, default off)"
```

### Task 10: Server integration behind Q27_BATCH=1

**Files:**
- Modify: `src/server.cu` (the 3 generate call sites: /v1/messages stream +
  non-stream, /v1/completions; grep `eng.generate` / `generate(` to find all)

**Dependencies:** Task 9

**Step 1:** When `Q27_BATCH=1`: after prefill (request thread, under gate as
today), build DecodeTask, register with the conductor, drain TokenQueue into
the existing on_token consumer logic (StreamSplitter/SSE/tool_buf paths --
the lambda bodies move, they do not change). When off: the pre-existing
inline generate() path runs BYTE-FOR-BYTE (do not refactor it in this task).
[req] logging: conductor mode fills the same GenStats fields; add ` bat=k`
(mean batch width this request) after `sfx=`.

**Step 2: E2E gates**

- `Q27_BATCH=0` replay = byte-identical vs `scratchpad/p0_baseline/`.
- `Q27_BATCH=1`, ONE request (codegen replay): solo fallthrough, tokens
  byte-identical to Q27_BATCH=0.
- `Q27_BATCH=1`, TWO concurrent replays (codegen + docs payloads,
  `curl ... & curl ... & wait`): both complete, each stream's text
  byte-identical to its solo run (no-trim traffic at 32K ctx: widths <= 7
  each, union <= 14 <= 16 on the w16 build; on the W12 build trim WILL fire
  -- run this gate on q27-server-w16, that is the serving target).
- same two-concurrent run twice: byte-identical to itself (composition
  determinism).

**Step 3: Commit**

```bash
git add src/server.cu && git commit -m "P1: server conductor mode (Q27_BATCH=1) -- register/queue/drain; E2E solo-equivalence + composition gates green"
```

### Task 11: Aggregate A/B -- the headline gate

**Files:**
- Create: `tools/batch_ab.sh` (pattern: `tools/accept_ab.sh` + the 07-14
  `scratchpad/dexit_ab.sh`)

**Dependencies:** Task 10

**Step 1:** Script: vanilla qwen, w16 server build, fp8 KV, PMIN 0.5, MAXD
auto, 32K ctx x 2 slots (`--slots 2 --slot1-ctx 32768`). Legs:
- baseline: `Q27_BATCH=0` (today's FIFO interleave), 2 concurrent replay
  payloads (codegen+docs), 1 cold + 3 warm rounds each; metric = summed
  decode tokens / wall of the concurrent window (and per-[req] tps).
- batched: `Q27_BATCH=1`, same payloads.
Also run 1-concurrent legs both modes (solo regression check: batched-mode
solo must be within noise of baseline solo).

**Step 2: Run via systemd-run --user; read results**

Expected (design planning number): batched/baseline aggregate >= 1.3x at 2
concurrent. Record actual numbers in the script output block + BUILDLOG.
If < 1.3x: profile before concluding -- nsys the fused round; the usual
suspects are (a) serial mixers (P2 lever, known), (b) eager launch tax (P3
lever, measure host enqueue vs GPU consume), (c) draft phases not
overlapping (check the event wiring from Task 8 step 1).

**Step 3: Commit + BUILDLOG entry**

```bash
git add tools/batch_ab.sh docs/BUILDLOG.md && \
git commit -m "P1: 2-slot aggregate A/B -- <measured>x vs FIFO baseline (bar 1.3x)"
```

### Task 12: Phase exit -- sanitizer, w16 gate, review

**Dependencies:** Task 11

- compute-sanitizer memcheck on the 2-concurrent batched server run
  (dual-arch build): 0 errors.
- `build/test_kernels` + canonical + sampled-seed one more time, fresh.
- `build/ninv_test`: still ALL PASS.
- Full-suite verification per Completion Gate, then request code review
  (conclave:requesting-code-review), then the SECOND review pass
  (state-heavy task rule).
- Offer Gabe the Makefile target bundle (ninv_test, test_conductor,
  fused_smoke) as ONE diff for SECURITY TRIGGERED approval.
- Update memory + BUILDLOG with measured numbers; the design doc's EV table
  gets the ACTUAL 2-slot number.

## Sequencing / parallelism

- Wave 1: Task 0, then Tasks 1-4 strictly serial (same file, each gated).
  Tasks 5, 6 can run parallel to Wave 1 (different files).
- Wave 2: Task 7, then 8, then 9 (8 and 9 share conductor.h; serial).
- Wave 3: Tasks 10 -> 11 -> 12 serial.
- Every task ends with the canonical gate if src/engine.cuh was touched.

## Explicit non-goals (P2/P3, do not build now)

Mixer side-stream overlap; fused draft steps; shape-graphs / device perm
indirection; >=3-slot tuning; Q27_BATCH default-on. Each needs the P1
measurement first.

---

## Consensus-review addenda (2026-07-14, resolve before Tasks 7-10)

**A1. Task 5 / Task 10 contract resolution.** If `ninv_test` FAILS for a
kernel family, the Task 10 solo-equivalence gate DOWNGRADES for traffic that
exercises that family: the gate then requires (a) composition determinism
(same mix twice -> same bytes) and (b) a quality spot-check, and the design
doc's determinism section gets amended in the same commit with the measured
finding. Solo-equivalence remains required for traffic on passing families.
There is no configuration where Task 10 silently accepts divergence.

**A2. Error/abort path.** CUDA_CHECK failures stay PROCESS-FATAL (unchanged
single-operator posture, docs/SECURITY-MODEL.md). The conductor adds no
recovery machinery. Non-CUDA failures are cancellations (A3). A member's
HOST-side exception during bookkeeping must not leave the conductor holding
the gate: the round loop's gate hold is scoped RAII (GpuGate::Lease), and
queue `done`/error posting happens in the unwind path (mirror HookGuard).

**A3. Cancellation semantics.** `DecodeTask` gains `std::atomic<bool> cancel`.
Request thread sets it (SSE write failure / client disconnect / shutdown).
Conductor checks at round boundaries ONLY: a cancelled member gets its
teardown run (tc-hook clear via the HookGuard pattern, GenStats finalized,
`done` posted, slot freed via the existing slot_guard path on the request
thread after the queue drains). Its lanes are absent from the next round.
No mid-round teardown, ever.

**A4. Encapsulation model: thin engine-owned entrypoints.** The conductor
calls a narrow public Engine surface only: `solo_view()`, `fused_pre/mix/
post(il, ...)` (or equivalently pre/mix/post as split in Task 3),
`draft_and_gate()`, `verify_tail(view)`, `decode_step` bookkeeping pieces
(Task 7), `set_granted_width(w)`. No `friend`, no raw member reaches from
conductor.h. If Task 8 finds itself needing another member, it adds an
accessor in engine.cuh with a comment saying why.

**A5. Trim-active gate (append to Task 10 Step 2).** On the W12 build, run
2 concurrent suffix-saturating echo payloads (`scratchpad/echo_ctx4k.json`
pattern) so trim MUST fire: assert both streams complete, `bat=` telemetry
shows trim, and composition determinism holds. No solo-equivalence claim on
this leg.

**A6. build_union_view audit (append to Task 8 Step 3).** fused_smoke
additionally dumps the union view's slot->pointer table and asserts each
slot's pointers equal the owning engine's solo_view() entries at the mapped
lane (host-side, exact pointer equality). Plus `assert` no engine appears
twice and union vw <= W_MAX.

**A7. Gate-ownership invariant (Task 9).** In batch mode, gate holders are
exactly: the conductor (decode rounds) and request threads (prefill chunks).
Invariant: the conductor never blocks on a request thread while holding the
gate (token posts are non-blocking); request threads never hold the gate
while waiting on a TokenQueue. State this as a comment on the conductor's
round loop and keep it true.

**A8. Graph-capture non-interference (explicit).** build_spec_graphs
captures the SAME refactored functions (Tasks 2-4); the canonical +
sampled-seed + replay gates are the capture-equivalence proof and run on
every engine.cuh-touching commit. No separate capture path may be added.

**A9. vgemm_ws sizing assert (Task 8).** `build_union_view` asserts
engines[0]'s workspace covers the union width (read the allocation size, do
not assume W_MAX).

**A10. Solo-latency regression check (Task 11).** The A/B script's
1-concurrent legs also compare per-request p50 tps: batched-mode solo must
be within noise (<2%) of baseline solo, or the solo fallthrough is not a
fallthrough.
