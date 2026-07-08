# Acceptance Gate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use conclave:executing-plans to implement this plan task-by-task.

**Goal:** Make q27's speculative-decode depth control track verifier ACCEPTANCE
(realized per-lane yield) instead of drafter confidence (theta margin), per
docs/acceptance-gate-design.md.

**Architecture:** Phase 0 adds telemetry (per-lane fired/accepted counters in
the live gate path; per-pass margin bins in `--stats`) and refreshes the d4/d5
economics on post-verify-gemv HEAD. Phase 1 fixes the P13 yield semantics
(conditional on fired), clamps the promote seed, and retunes `maxd_lo` to the
measured breakeven. Phase 2 (conditional on Phase-0 data) generalizes P13 to a
per-lane ceiling controller extracted into a host-testable header. Phase 3
validates at the live operating point.

**Tech Stack:** CUDA C++ (sm_120), host-side control logic only (no kernel
changes), bash rigs in tools/, canonical gates via tools/shortbench_suite.sh.

**Baselines that must not move:** canonical md5 4c4120c72056aba2bc2d2561471eafce
(greedy, fd2 default); test_kernels ALL PASS; gated==ungated token identity.

**Constants (from BUILDLOG:1648 maxd6 measurement, PRE-verify-gemv -- Phase 0
refreshes these):** d5-vs-d4 26K replay: repro +2.9% (97.6% fired / 95.2%
yield), codegen -5.4% (56.3%/56.2%), testgen -3.9% (51.9%/48.6%); live-T8
interpolation -0.8..+0.1%. Derived breakeven lane-5 yield ~0.7-0.8.

**Rig conventions:** model /mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27,
tokenizer .tok sibling. Production serving config = `Q27_KV=fp8` + `--fast-head`.
fp8 KV is OPT-IN (see reference_q27_bench_gotchas). Fresh server per leg, 1 cold
prefill + 3 identical replays, medians, replay spread <= 0.3% or rerun. Greedy.
5090 must be free (`nvidia-smi` first).

---

### Task 0: Branch + baseline gates

**Files:** none (commands only)

**Dependencies:** none

**Step 1:** `cd /mnt/ai/projects/q27 && git checkout -b acceptance-gate && make -j`
Expected: clean build, no new warnings.

**Step 2:** `build/test_kernels | tail -5`
Expected: ALL PASS.

**Step 3:** `bash tools/shortbench_suite.sh`
Expected: `canonical: md5 OK (4c4120c72056aba2bc2d2561471eafce)`.
Record the suite mean t/s as the session baseline.

---

### Task 1: Per-lane fired/accepted counters (live gate telemetry)

The marginals gch/gnh cannot give p(lane j accepted | lane j fired); these
counters can, and they are what Phase-1/2 bars act on.

**Files:**
- Modify: `src/engine.cuh:241-242` (member decl), `src/engine.cuh:1203` (update)
- Modify: `src/server.cu:297,314-322` (print)

**Dependencies:** Task 0

**Step 1: Add members** next to the existing hists (engine.cuh:241-242):

```c++
    long gate_cap_hist[6] = {}; // [cap]
    long gate_n_hist[7] = {};   // [n]; index 0 unused
    // acceptance-gate Phase 0: per-draft-lane conditional acceptance on gated
    // rounds. Lane j (1..5) FIRED iff cap >= j; ACCEPTED iff n >= j+1. Yields
    // p(acc_j | fired_j) that the marginals above cannot reconstruct.
    long gate_lane_fired[6] = {}, gate_lane_acc[6] = {}; // [j]; index 0 unused
```

**Step 2: Update in spec_round** (engine.cuh:1203, extend the existing line):

```c++
        if (gate_cap >= 0) {
            gate_cap_hist[gate_cap]++; gate_n_hist[n]++;
            for (int j = 1; j <= gate_cap; j++) {
                gate_lane_fired[j]++;
                if (n >= j + 1) gate_lane_acc[j]++;
            }
        }
```

**Step 3: Print in [req]** (server.cu): widen `gatebuf[192]` -> `gatebuf[384]`
(22 longs + labels overflow 192) and append to the existing snprintf:

```c++
                                " gch=... gnh=..."
                                " glf=%ld,%ld,%ld,%ld,%ld gla=%ld,%ld,%ld,%ld,%ld",
                                ...existing args...,
                                e.gate_lane_fired[1], ..., e.gate_lane_fired[5],
                                e.gate_lane_acc[1],   ..., e.gate_lane_acc[5]),
```

**Step 4: Build + smoke.** `make -j`, then start the server with the gate on and
send one completion; verify glf/gla appear and are consistent (glf[1] >= glf[2]
>= ... ; gla[j] <= glf[j]; glf[1] == sum(gch[1..5])):

```bash
Q27_PMIN=0.5 Q27_MAXD=auto build/q27-server $MODEL $TOK --port 8199 --ctx 8192 --no-think &
curl -s localhost:8199/v1/completions -d '{"prompt":"int main() {","max_tokens":128}' >/dev/null
grep -o 'gch=.* gla=[0-9,]*' server.log | tail -1
```

**Step 5: Regression gates.** `build/test_kernels` ALL PASS;
`bash tools/reqlog_gate.sh` (C2 field-parse must tolerate the new fields) PASS;
`bash tools/shortbench_suite.sh` canonical OK (host counters cannot move tokens,
prove it anyway).

**Step 6: Commit.** `git add -A && git commit -m 'accept-gate Phase 0a: per-lane fired/accepted gate telemetry (glf/gla)'`

---

### Task 2: --stats per-pass margin bins

Today `--stats` bins depth-3/4/5 acceptance by the PASS-2 margin only; m3/m4/m5
are computed then voided (engine.cu:317,329,340). Bin by each pass's OWN margin
to measure p(acc_k | m_k bin, prefix ok) -- this decides the theta-schedule
complement (design doc, "supporting role").

**Files:**
- Modify: `src/engine.cu:194-207` (Pend structs + counters), `:316-341`
  (capture sites), `:358-386` (prints)

**Dependencies:** Task 0 (independent of Task 1)

**Step 1: Carry own margins.** Pend3 gains `float margin3;` (Pend4:
`margin4`, Pend5: `margin5`). Capture sites stop voiding:

```c++
            auto [d3, d3b, m3] = top2(l1);
            (void)d3b;
            if (known_idx + 3 < (int)pend3.size()) pend3[known_idx + 3] = {d3, m2, d1, d2, m3};
```

(same shape for m4 at :328 and m5 at :339 -- keep field order matching the
struct).

**Step 2: Own-margin counters.** Next to c3n/c3pre/c3ok add
`long o3n[5]={0}, o3ok[5]={0};` (and o4/o5). In the scoring block, when the
prefix is accepted, also bin by the own margin:

```c++
                if (p3.d1 == seq[known_idx - 2] && p3.d2 == seq[known_idx - 1]) {
                    c3pre[b]++;
                    int ob = bin(p3.margin3); o3n[ob]++;
                    if (p3.pred == seq[known_idx]) { c3ok[b]++; o3ok[ob]++; }
                }
```

(same for depth 4/5; note o-counters condition on prefix-ok, so
o3ok/o3n = p(d3 | prefix ok, m3 bin) directly).

**Step 3: Print** after the existing depth-5 block:

```c++
        printf("  acceptance by OWN-pass margin (prefix ok) [accept-gate Phase 0b]:\n");
        for (int d = 3; d <= 5; d++) { /* per depth: */
            printf("    d%d: ", d);
            for (int b = 0; b < 5; b++)
                printf("%s=%4.1f%%(n=%ld) ", bl[b], 100.0 * oNok[b] / (oNn[b] ? oNn[b] : 1), oNn[b]);
            printf("\n");
        }
```

(unroll per depth; no macro heroics needed.)

**Step 4: Run.** `build/q27 $MODEL --stats 512 --tokens-file scratchpad/toks8k.txt`
Expected: new block prints; d3 row's total n equals c3pre total; existing E3
lines unchanged.

**Step 5: Interpret + record.** GATE QUESTION: does p(acc_k | m_k) rise steeply
with m_k at the operating range (0.5-2), or is it flat per regime? Record the
table in the BUILDLOG entry. Steep -> theta-schedule complement goes on the
Phase-2 menu; flat -> theta stays scalar (expected per design).

**Step 6: Commit.** `git add -A && git commit -m 'accept-gate Phase 0b: --stats own-pass margin acceptance bins'`

---

### Task 3: Replay A/B rig (tools/accept_ab.sh) + payloads

The maxd6 rig lived in scratchpad and is gone; commit this one. Three ~26K
payloads spanning the saturation axis, per the depth-match recipe
(BUILDLOG:1841): repro = docs+self-copy, code = q27 src concat, transcript =
CC-log flavor. GOTCHA: greedy raw-completion of plain prose EOSes instantly at
depth -- every payload needs an open continuation (mid-echo cut or list form).

**Files:**
- Create: `tools/accept_ab.sh` (server A/B: legs = env-config x payloads,
  1 prefill + 3 replays, medians, parses tps + gch/gnh/glf/gla from [req])
- Create: `tools/make_payloads.py` (build the 3 payloads at ~26K tokens from
  docs/ + src/ + a runs/ transcript; verify token count via /v1/messages/count_tokens)

**Dependencies:** Task 1 (needs glf/gla in [req])

**Step 1:** Write `tools/make_payloads.py`; verify each payload:
26K +- 1K tokens, and a 32-token greedy probe does NOT hit EOS.

**Step 2:** Write `tools/accept_ab.sh` with legs:

```
for payload in repro code transcript:
  A: Q27_KV=fp8 Q27_PMIN=0.5 Q27_MAXD=4 --fast-head   (d4 baseline)
  B: Q27_KV=fp8 Q27_PMIN=0.5 Q27_MAXD=5 --fast-head   (fixed d5)
  C: Q27_KV=fp8 Q27_PMIN=0.5 Q27_MAXD=auto --fast-head (P13 auto, current LO)
```

Each leg: fresh server, decode 256, 1 cold prefill + 3 replays, print median
tps, tok/round, and the per-lane yields gla[j]/glf[j].

**Step 3:** Dry-run one leg (repro/A), confirm replay spread <= 0.3%.

**Step 4: Commit.** `git add -A && git commit -m 'accept-gate Phase 0c: committed replay A/B rig + payload builder'`

---

### Task 4: Phase-0 measurement + bar derivation  [DECISION GATE]

**Files:**
- Modify: `docs/BUILDLOG.md` (new dated section), `docs/acceptance-gate-design.md`
  (fill UNMEASURED)

**Dependencies:** Tasks 1-3. GPU free.

**Step 1:** `bash tools/accept_ab.sh 2>&1 | tee results-accept-ab.log`
(~9 legs x ~2-4 min).

**Step 2: Derive.** For each payload: d5 net% vs d4, fired = glf[5]/gated
rounds, yield y5 = gla[5]/glf[5]. Regress net=0 across the 3 payloads ->
**bar_5** (refreshed breakeven yield). Also read y2..y4 per payload.

**Step 3: DECISION GATE.**
- If y5 < bar_5 on the codegen-flavored payload while fixed-d5 is negative
  there (expected): Phase 1 proceeds.
- If post-verify-gemv economics moved so much that fixed-d5 is now >= 0
  everywhere: STOP after retuning nothing; write verdict (the gate's premise
  is gone; reopen maxd6 instead).
- If any y_j (j<5) sits below ~bar_5-0.15 on any payload: Phase 2 scope
  includes levels below 5 (k_min < 4); else Phase 2 collapses into Phase 1
  (controller extraction still happens for testability, no new levels).

**Step 4:** Record everything in BUILDLOG + design doc; commit.

---

### Task 5: Extract P13 controller to src/depthctl.h (host-testable, TDD)

Pure refactor first: identical semantics, unit tests pin them, GPU gates prove
nothing moved. Mirrors the P15 toolconstrain.h precedent.

**Files:**
- Create: `src/depthctl.h` (struct DepthCtl: cur_maxd, sat_ema, yield_ema,
  alpha, hi, lo, rounds4/5, promotes/demotes, `void update(int md_used,
  int gate_cap, int n)` -- body = today's engine.cuh:1209-1224 verbatim)
- Create: `tools/test_depthctl.cpp` (plain g++, no CUDA)
- Modify: `src/engine.cuh` (replace inline block + members with `DepthCtl dctl;`,
  env parsing at :944-952 writes dctl fields)
- Modify: `src/server.cu:307-310` (`e.maxd_rounds4` -> `e.dctl.rounds4` etc.)
- Modify: `Makefile` (test_depthctl target, add to `all` next to test_kernels)

**Dependencies:** Task 0 (can run parallel to Tasks 1-4)

**Step 1: Failing tests first** (`tools/test_depthctl.cpp`): promote at
sat_ema crossing hi after ceil(ln((hi-1)/(sat0-1))/ln(1-a)) saturated rounds;
demote at yield < lo; promote seeds yield = 2*lo; demote zeroes sat; EMA
half-life ~11 rounds at a=1/16; level exclusivity (depth-4 rounds never touch
yield_ema). Compile: `g++ -std=c++17 -O1 tools/test_depthctl.cpp -o build/test_depthctl`.
Expected: FAIL (header absent).

**Step 2:** Write `src/depthctl.h` with the verbatim P13 semantics. Tests PASS.

**Step 3:** Wire into Engine; `make -j`; `build/test_kernels` ALL PASS;
`bash tools/shortbench_suite.sh` canonical OK; 16K gated token-identity spot
check (Q27_PMIN=0.5 Q27_MAXD=auto before/after binaries -> byte-identical
output, same round count).

**Step 4: Commit.** `'accept-gate: extract P13 depth controller to depthctl.h + CPU tests (pure refactor)'`

---

### Task 6: Phase 1 -- conditional yield + seed clamp + bar retune

The two semantic fixes, TDD'd in depthctl, then the knob measured and flipped.

**Files:**
- Modify: `src/depthctl.h`, `tools/test_depthctl.cpp`
- Modify: `docs/BUILDLOG.md`

**Dependencies:** Tasks 4, 5

**Step 1: Failing tests:** (a) at depth-5, rounds with gate_cap < 5 do NOT
update yield_ema (lane 5 unfired = no evidence; dexit makes it near-free);
(b) promote seed = min(1.0f, 2*lo) (at lo=0.6 old code seeded 1.2 --
undemotable for ~1/alpha extra rounds); (c) with lo=bar_5 and a 56%-yield
hit stream (the codegen regime), demote fires within ~2 half-lives.

**Step 2:** Implement both changes in depthctl.h. Tests PASS. GPU gates
(test_kernels, canonical, gated identity) PASS -- yield-update conditioning
changes round grouping only via later demotes; token identity must hold.

**Step 3: Knob A/B (env only, no default change yet):** rerun accept_ab.sh
leg C with `Q27_MAXD_LO=<bar_5 from Task 4>` on all 3 payloads.
SUCCESS: codegen-flavored payload recovers to >= d4 baseline (auto demotes);
repro keeps >= +2% (auto stays at 5). If repro regresses: bar too hot;
bisect LO between 0.10 and bar_5.

**Step 4:** Flip the default `lo` in depthctl.h to the validated value with a
comment citing the measurement; update the seed-clamp comment. Rerun leg C
WITHOUT env overrides -> same numbers.

**Step 5:** BUILDLOG entry + commit:
`'accept-gate Phase 1: conditional lane-5 yield + seed clamp + maxd_lo -> <bar> (measured breakeven)'`

---

### Task 7: Phase 2 -- per-lane ceiling controller  [CONDITIONAL on Task 4 gate]

Skip entirely if Phase 0 found no sub-bar lane below 5.

**Files:**
- Modify: `src/depthctl.h` (per-level {sat, yield} EMAs, k_min..k_max ladder,
  promote/demote per design doc "Mechanism"), `tools/test_depthctl.cpp`
- Modify: `src/engine.cuh` (cap_eff = min(margin run, dctl.cur) -- ALREADY the
  structure via md_used; only the ladder range changes. Note: k_min < 4 means
  draft_step_graph/verify_graph_w for shallower md exist already, widths 2..6;
  no graph work.)

**Dependencies:** Tasks 4, 6

**Steps:** TDD ladder tests (multi-level promote/demote paths, hysteresis at
each level, no oscillation on a 50/50 alternating stream -- assert transition
count bounded), wire, GPU gates, A/B on the payload that motivated it, BUILDLOG,
commit. Bars per level from Task-4 y_j data (bar_j = c_j * r; c_j from the
per-width fd2 increments in docs/perf-attribution-p14.md scaled by the
verify-gemv refresh).

---

### Task 8: Phase 3 -- live-operating-point validation + merge

**Files:**
- Modify: `docs/BUILDLOG.md`, `docs/acceptance-gate-design.md` (verdict),
  `MEMORY` update out-of-repo.

**Dependencies:** Task 6 (and 7 if it ran)

**Step 1:** T8-matched replay A/B (transcript payload, auto vs d4-fixed vs
old-LO auto). SUCCESS per design: net >= +1% vs the -0.8..+0.1% status quo,
no payload regresses below its d4 baseline.

**Step 2:** Full gate suite fresh: test_kernels, shortbench_suite canonical,
reqlog_gate, test_depthctl, gated-vs-ungated identity at 2K/16K/61K,
round-count determinism (2 runs, same rounds).

**Step 3:** BUILDLOG verdict section (include the honest expectation vs
measured), design-doc UNMEASURED section resolved.

**Step 4:** Merge to master per repo convention (fast-forward or merge commit
matching prior P-branches), push.

---

## Execution notes

- Every measurement leg needs the 5090 free; `nvidia-smi` before each rig run.
- Servers in rigs: launched by the rig scripts themselves; do not leave a
  server resident between tasks (canonical gates need the GPU).
- Long background pieces (if any) via `systemd-run --user` (session-crash
  survival, per ops memory).
- Minimal bench scope (user preference): the 3 payloads ARE the suite; do not
  add more flavors without a decision reason.
