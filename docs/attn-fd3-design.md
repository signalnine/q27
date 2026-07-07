# attn-fd3: lane-pair fused flash-decode

**STATUS: KILLED 2026-07-07 (implemented, all bitwise gates passed, bench
lost -4.0% @61K ungated vs the >= +5% keep floor; kernel reverted, this
doc kept per the kill protocol). See "Outcome" at the end -- the negative
is fully attributed: the 16->12 warps/SM occupancy cost exceeds the
halved-KV-traffic win; fd2 is latency-hiding-bound before it is BW-bound
at this occupancy.**

2026-07-07. Follows the P14 Task-5 result (docs/perf-attribution-p14.md):
fd2 is DRAM-BW-bound and the verify lanes re-stream the KV slice (R ~= 4.25
at 61K); the Task-5 lane-innermost axis swap recovered only ~10% of the
verify per-instance time because L2 absorbs a minority of the ~63 MB/layer
fp8 KV slice per co-scheduled wave. fd3 fuses PAIRS of verify lanes into
one block so each KV byte read from DRAM serves 2 lanes out of registers --
the paired lanes' KV traffic halves instead of relying on L2 capacity.

## Kernel (k_attn_fd3, replaces k_attn_fd2 as default; fd2 kept in-tree
## as Q27_FD=fd2 fallback, v1 kept as Q27_FD=v1)

### Pairing scheme + grid

Block `bx` owns lanes `t0 = 2*bx` and `t1 = 2*bx + 1` (lane 1 exists iff
`t1 < ntok`). Grid preserves the Task-5 lane-innermost order with the pair
index on the fastest axis:

    dim3(ceil(ntok/2), FD2_NS, n_kv_heads)     // was dim3(ntok, FD2_NS, n_kv_heads)

so same-(head,split) pair-blocks still co-schedule onto the same KV chunk
(the Task-5 cross-PAIR L2 reuse is retained on top of the in-register
pair sharing). Block stays NW=4 warps = 128 threads. ntok=1 (draft) runs
grid x=1 with the lane-1 guard off -- same kernel, no separate path.

### Per-lane state, position loop

Per-thread registers: `m[2][6], l[2][6], acc[2][6][8]` -- lane s of the
pair keeps its ENTIRE online-softmax state (max, sum, accumulator) in its
own register slice, all indices compile-time (unrolled) per the
lane-count-landmine rule (no dynamic register indexing, no per-lane
pointer ternaries). Thread dim ownership is unchanged from fd2:
D(l) = {4l..4l+3, 128+4l..128+4l+3}, K/V rows load via fd2_ld8 (two
4B/8B words per row per thread).

Each lane s has its own split geometry, exactly fd2's:

    seq_s   = *pos.p[t_s] + 1
    chunk_s = ceil(seq_s / FD2_NS)
    plo_s   = sp * chunk_s;  phi_s = min(seq_s, plo_s + chunk_s)
    active_s = plo_s < seq_s          (inactive lanes write nothing)

If neither lane is active the block returns (fd2's empty-split early
return, per pair). The position loop iterates a RELATIVE index so both
lanes advance in lockstep with fd2's warp phase:

    for (i = warp; i < max(phi0-plo0, phi1-plo1); i += NW):
        p0 = plo0 + i; p1 = plo1 + i
        u0 = p0 < phi0; u1 = has1 && p1 < phi1
        if (u0): load K/V row p0 (once) -> update lane 0 (j = 0..5)
        if (u1): if row p1 != loaded row: load K/V row p1
                 update lane 1 (j = 0..5)

Lane s's warp-w subsequence is `{plo_s + w + k*NW} ∩ [plo_s, phi_s)` in
increasing order -- IDENTICAL positions, identical warp assignment,
identical per-position arithmetic to fd2 running lane s alone. When the
lanes' split bases align (`plo0 == plo1`, i.e. equal chunk -- the common
case for consecutive verify positions -- or sp == 0), the row-p1 reload
never fires and one KV read serves both lanes from registers. When a
pair straddles a chunk-count boundary (ceil(seq/128) differs, ~1/128 of
positions) the lanes' rows diverge and both loads issue -- bitwise-safe
degradation to fd2 traffic for that instance, no special-casing.

### Update body

The per-(position, head) update is fd2's expression verbatim (float4
s_q reads, 8-term dot, shfl_down reduce + lane-0 broadcast, exact expf
online-softmax rescale, 8 FMA accumulator updates), instantiated per
lane with the lane's own s_q base and register slice. Same source
expression, same contraction shape -> same PTX arithmetic per lane.

### smem budget

    s_q[2][6][256]   12,288 B   (2x fd2's q tile, one per lane)
    s_mrg[6][256]     6,144 B   (cross-warp merge buffer, REUSED per lane)
    s_ml[2][NW][6][2]   384 B   (static; per-lane per-warp {m,l})
    total            ~18.8 KB dynamic+static

(The plan's 12.3KB estimate assumed the merge buffer could alias dead
s_q space; a separate reused buffer costs 6KB more and stays far from
the smem occupancy bound -- smem allows >= 4 blocks/SM on both archs,
registers are the binding limit. Chosen for clarity over the alias.)

### Epilogue (per lane, serialized)

For s in {0, 1}, lane active: zero s_mrg; barrier; NW barrier passes
adding warp w's `acc[s]` rescaled by `exp(m - m_block)` in FIXED warp
order 0..NW-1 (fd2's bitwise-determinism contract -- no atomics); then
per-head {m_block, l_block, s_mrg} written to the lane's OWN scratch
cell `pair = t_s*(n_kv*gqa) + kvh*gqa + j`, `dst = part +
(pair*FD2_NS + sp)*FD_ST` -- byte-identical layout and values to fd2,
so **k_attn_fd_combine is untouched** and scratch sizing is unchanged.
Lane 1's merge reuses s_mrg after a barrier. All merge arithmetic per
lane is fd2's expression on that lane's own state.

### The bitwise argument

fd3 output must be BITWISE-IDENTICAL to fd2 (this is the contract, not
an aspiration -- no tolerance gate, no canonical re-derive). Per lane:
(1) position set and order per warp are identical (same plo/phi/NW
phase); (2) every fp op in the update is the same expression on the
same values (lane-private m/l/acc, shared K/V values are loads of the
same bytes); (3) the cross-warp merge runs the same NW-pass serialized
order on the lane's own {m,l,acc}; (4) the scratch partial and the
untouched combine give the same final output bytes. Lane interleaving
inside a block orders INDEPENDENT register updates only -- no shared fp
state exists between lanes anywhere.

### Odd ntok / lane guard

Last block at odd ntok has `t1 == ntok` -> `has1 = false`: lane-1 q
load, position-loop updates, epilogue, and the `pos.p[t1]` deref are
all guarded off. Single kernel, no second launch, no tail kernel.
ntok=1 is the degenerate all-blocks-single-lane case (draft path).

### Dispatch / fallback

`attn_decode3` reads Q27_FD at launch time (graph capture bakes it):

    Q27_FD unset / other  -> fd3 (new default)
    Q27_FD=fd2            -> k_attn_fd2 (Task-5 unpaired kernel, bitwise-equal)
    Q27_FD=v1             -> k_attn_fd (pre-fd2 kernel, old canonical 58b6ae85)

`attn_decode3_fd2` stays exported (fd2-vs-v1 unit gates unchanged);
`attn_decode3_fd3` is exported for the fd3 unit gates. Bisect recipe:
Q27_FD=fd2 rules the pairing in/out with zero output change expected.

## Occupancy gate (HARD, before any kernel logic ships)

Skeleton with the true resource shape (2x s_q smem, m/l/acc[2][6][8]
register state, real update arithmetic), compiled `-Xptxas -v` for
sm_86 AND sm_120 at NW=4 (128 threads):

    PASS requires, on BOTH archs:
      regs/thread <= 168
      resident blocks/SM >= 3 (>= 12 warps/SM), from regs AND smem

(168 regs is the 3-block ceiling: 3 blocks x 4 warps x ceil(168*32/256)
alloc units = 64,512 of 65,536 regs/SM. smem at ~18.8KB/block allows
>= 4 blocks on both archs' 100KB SMs -- registers bind.) Below either
bound the pair design loses fd2's latency-hiding and is NOT shipped;
one quick variant allowed (launch_bounds reg cap / narrower acc
ownership / NW retune), else record the ptxas numbers as a BUILDLOG
negative and stop.

Measured (Stage B skeleton, CUDA 13.2 V13.2.51, -O2, project flags,
fp8 and fp16 instantiations identical) -- GATE PASS on both archs:

    sm_86:  168 regs, 0 spills, 18,816B smem/block (18,432 dyn + 384 static)
            -> blocks/SM: regs 3 (3x4x5376 = 64,512 of 65,536), smem 5
            -> 3 resident blocks = 12 warps/SM
    sm_120: 168 regs, 0 spills, 18,816B smem/block
            -> blocks/SM: regs 3, smem 5 -> 3 resident = 12 warps/SM

    (The 96B stack frame is the CP3/IP3 param-pointer arrays dynamically
    indexed by t0/t1, stored/reloaded ONCE in the prologue -- fd2 has the
    identical 96B frame; no hot-loop local traffic, 0 register spills.
    fd2 reference on the same flags: 119 regs sm_86 / 122 sm_120 ->
    4 blocks / 16 warps/SM; fd3's 12 warps/SM meets the floor.)

## Gates (all must pass; fd3 is bitwise so the canonical is UNCHANGED)

- test_kernels: every existing fd2 gate PASSES unchanged.
- fd3 section, seq {1,47,1024,16384,61440} x ntok {1,2,3,4,5} x
  {fp8,fp16}: (a) fd3 vs fd2 BITWISE (err 0 against 1e-30, the fd2
  bitwise-gate style); (b) fd3 run-to-run bitwise determinism;
  (c) default dispatch == fd3 bitwise; (d) odd ntok {1,3,5} exercises
  the single-lane guard (and even {2,4} the pure-paired path).
- Canonical md5 EXACT `4c4120c72056aba2bc2d2561471eafce` (bitwise
  kernel -> unchanged). A mismatch means the kernel is NOT bitwise:
  fix or stop; the tolerance-gate + canonical-re-derive policy
  (attn-fd2-design.md:42-56) is NOT invoked without the orchestrator.
- Bench (server rig, greedy, 1 warmup + n=3 medians): tok/round and
  round counts IDENTICAL to the Task-5 POST baselines in every cell
  (any round-count change is a bug).

## Perf targets and kill criteria

Task-5 POST baselines (same rig, docs/perf-attribution-p14.md):
61K ungated 119.3 t/s / 28.50 ms/round (113 rounds), 61K gated0.5
124.2 / 25.55, 2K ungated 115.5 / 20.77 (160 rounds), 2K gated 129.5 /
17.76. Verify fd2 per-instance ~487 us vs the ~128 us single-lane
draft floor; ideal pair-fusion traffic at width 5 is 3 streams vs 5.

KILL (revert the kernel commit, KEEP this doc, BUILDLOG negative):
- 61K ungated gain < +5%, or
- 2K regression > 2%, or
- any fd2/fd3 unit gate fails.

## Outcome (2026-07-07): KILLED on the perf gate; every correctness gate passed

The kernel was implemented exactly as above and passed everything except
the bench:

- Stage-B occupancy gate: PASS (numbers above; exactly the 168-reg bound,
  0 spills, 3 blocks / 12 warps per SM on both archs).
- test_kernels: 384 checks ALL PASS, 0 FAIL. All 132 fd2-section gates
  unchanged. All 160 fd3-section gates (fd3-vs-fd2 bitwise, run-to-run
  determinism, default-dispatch==fd3, Q27_FD=fd2 fallback; seq
  {1,47,1024,16384,61440} x ntok {1,2,3,4,5} x {fp8,fp16}) exact err
  0.000e+00 against tol 1e-30. The bitwise argument HELD in practice.
- Canonical md5 EXACT 4c4120c72056aba2bc2d2561471eafce on the fd3 binary.

Bench (server rig, docs prompts, greedy, max_tokens=384, 1 warmup + n=3
medians, spread <0.1%; same binary, same session: Q27_FD=fd2 = the
unpaired Task-5 kernel as the pre side -- it reproduced the recorded
Task-5 POST baselines within 0.3% in every cell, so no session drift):

| cell            | fd2 (pre) t/s / ms/r | fd3 t/s / ms/r | delta  | tok/r, rounds |
|-----------------|----------------------|----------------|--------|---------------|
| 61K ungated     | 118.9 / 28.58        | 114.2 / 29.75  | -4.0%  | 3.398, 113 (identical) |
| 61K gated 0.5   | 124.3 / 25.53        | 118.1 / 26.87  | -5.0%  | 3.174, 121 (identical) |
| 2K ungated      | 115.4 / 20.79        | 115.1 / 20.84  | -0.3%  | 2.400, 160 (identical) |
| 2K gated 0.5    | 129.5 / 17.75        | 129.4 / 17.77  | -0.1%  | 2.299, 167 (identical) |

KILL: 61K ungated -4.0% << +5% floor. (2K within the 2% band; all unit
gates passed -- the perf criterion alone killed it.)

Mechanism (nsys node-traced decode capture on the fd3 binary, 61K
ungated, vs the Task-5 POST fd2 per-instance rows, same methodology):

| instance class      | fd2 (Task 5) | fd3           | delta |
|---------------------|--------------|---------------|-------|
| draft, grid x=1     | 128.2 us     | 218.9 us avg  | +71%  |
| verify, x=3 (pairs) | 487.3 us     | 548.2 us avg  | +13%  |

Predicted round delta 4x0.091 + 16x0.061 = +1.34 ms/round; measured
+1.17 ms/round (29.75 - 28.58) -- consistent.

The draft row is the diagnosis: at ntok=1 fd3 does EXACTLY fd2's work
(no pair exists), differing only in resource shape (168 regs / 3 blocks
/ 12 warps per SM vs fd2's 119-122 regs / 4 blocks / 16 warps) -- and
runs 71% slower. The kernel is latency-hiding-bound before it is
DRAM-BW-bound at this occupancy: losing 4 warps/SM costs far more than
halving KV bytes saves, on both instance classes. The fd2 design doc
said as much ("the NW 8->4 probe proved it latency-hiding-bound"); the
occupancy floor this design gated on (>= 12 warps/SM) was too low --
the pairing needs >= fd2's 16 warps/SM to break even, i.e. <= ~128
regs/thread, and the honest per-lane state alone (120 floats) makes
that unreachable without changing per-lane fp order (register halving
via split dim ownership changes the shfl reduce lattice = not bitwise
= tolerance-gate + canonical re-derive territory).

Any future re-attempt should either (a) accept the fp-order change and
batch it into a tolerance-gate cycle (per attn-fd2-design.md:42-56, the
orchestrator's call), or (b) attack the residual R-headroom without
per-thread state duplication (e.g. __ldcs streaming hints, or a
persistent-block KV tile walk that shares loads across lanes through
smem instead of registers -- smem tiles do not double the register
state but reintroduce the RMW-serialization fd2 removed; measure first).
