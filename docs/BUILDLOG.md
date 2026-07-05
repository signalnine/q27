# q27 build log (chronological)

Verbatim DONE blocks for every milestone and P-step, moved out of the
README 2026-07-03. Each block reflects the state of knowledge AT THE
TIME; bracketed [superseded -- ...] marks point at what replaced a
number. Canonical current numbers live in the README ("State of the
engine", "Decode methodology").

Context for the 2026-07-02 reordering: decode is in good shape (see "Decode
methodology" above for the canonical numbers -- 177.5 stock short-bench /
218.6 OC soak; the 188.9 in this section's history is the depth-3-era +4000
short bench; three consecutive micro-opt attempts on the remaining tail came
back negative). The user-visible gaps are cold prefill (~8x behind the fork's
tensor-core GEMM; 61s cold TTFT @26.7K, warm turns 1.3s via prefix cache
[superseded -- P1-P6 cut cold 28.5K TTFT to ~15.0s; P8 fixed the warm-turn
cache]) and two advertised claims with no measurement behind them.
Cheapest-blocking-first:

**P0 -- claims gates: DONE 2026-07-02.**
- Long-context: validated to 64K (risk 5: flat NLL-by-position, +2.0%
  cross-engine at [32K,64K), needle 3/3). `--nll` / `--nll-long` added to the
  CLI (batched teacher-forced NLL, protocol-identical to llama-perplexity;
  gated vs a serial reference path).
- PPL delta: **+3.35% vs Q5_K_M** (7.2135 vs 6.9797), marginally over the 3%
  bar -> NEW ITEM below.

**P0.5 -- DONE 2026-07-02: v1.4 = ssm_out + attn_output promoted to Q8.**
Sensitivity study (full-corpus --nll per candidate repack; baseline v1.3
PPL 7.2139, Q5_K_M 6.9797):

| candidate (Q4 -> Q8) | +GB | PPL | verdict |
|---|---|---|---|
| ffn_down first4+last4 | 0.35 | 7.2079 | dud |
| ffn_down first8+last8 | 0.69 | 7.2074 | dud |
| ffn_down ALL (ceiling probe) | 2.76 | 7.1466 | 29% of gap, unshippable ratio |
| attn_qkv (GDN in-proj) | 1.22 | 7.2396 | **WORSE than baseline** |
| residual writers, late-only | 0.12 | 7.2112 | dud (not concentrated) |
| **residual writers ALL = v1.4** | **0.98** | **7.1928** | shipped |

Findings: (1) ffn_down sensitivity is spread uniformly across layers -- no
cheap subset exists; (2) promoting the GDN in-projections HURTS: v1.3's Q4
errors there partially cancel against downstream quant errors, and breaking
the correlation costs +0.36% PPL; (3) v1.4's decode got FASTER (+3.3% on a
2000-token soak, 203.9 -> 210.6 t/s) because cleaner residual writers raise
MTP draft acceptance (3.47 -> 3.67 tokens/round) by more than the +0.98 GB
of reads cost -- quant policy and speculation acceptance are coupled.
Remaining gap to Q5_K_M: +3.05% (was +3.35%); closing more via uniform
promotion has terrible ROI (see ceiling probe) -- importance-weighted scales
(AWQ-style) are the real path if quality ever becomes the priority.
v1.3 archived at qwopus-27b-mtp-v13.q27 (old canonical sequences apply to
it); all canonical gates re-derived for v1.4 in this commit.

**P1 -- DONE 2026-07-02: prefill 2.35x, cold 28.1K TTFT 63.8s -> 35.7s.**
Kernel: k_gemm_mma_T (prefill.cu), one mma.sync per 32-element quant block,
Q4 nibbles unpacked to s8 (offset folded) in smem staging; unit-gated at
1e-6 vs dp4a on all four weight shapes; full-corpus PPL +0.04% (fp reorder);
needle 3/3 @64K; decode canonicals untouched; Q27_PREFILL=dp4a keeps the
exact-reference path. nsys at 28K now shows attention prefill dominant
(~16.4s of the remaining ~36s wall) with delta_scan next (~3.8s) -- the
long-context TTFT lever is attention prefill now, NOT GEMM. GEMM tile
tuning (ldmatrix, cp.async, wider N tile to cut the 8x weight re-read)
still pays at short ctx but is deferred. Original plan below for reference.

**P1 (original plan) -- prefill via int8 tensor-core MMA:**
mma.sync m16n8k32 s8s8s32 (sm_80+ PTX, works on sm_120) replaces dp4a in the
prefill GEMM. CORRECTION (2026-07-02, found during implementation): the
within-chunk int32 dots are exact under MMA, but the per-32-block activation
scales force one fp multiply-add per chunk, and dp4a's fp structure per
output (32 stride-32 lane-partials + shuffle tree, matching serial GEMV)
cannot be reproduced by an MMA accumulator without infeasible register cost.
So the ORIGINAL review point stands: the MMA path needs a tolerance gate,
not the bitwise gate. Divergence is pure fp-reorder noise (~1e-6 rel;
integer dots exact), so tolerance is tight. Gates: unit test (MMA chunk
dots == dp4a exactly), --pfdbg state maxdiff at fp-noise level, full-corpus
--nll PPL delta < 0.02% vs dp4a prefill, needle 3/3, Q27_PREFILL=dp4a env
keeps the exact path as reference/fallback, decode canonicals untouched.
Still no CUTLASS; "plain CUDA" stays true. dp4a prefill: ~590 t/s @512 /
~300 @26K; fork reference 2,300-2,400. Realistic target 2-4x -> cold 26.7K
TTFT from 61s toward ~15-25s.

**P2 -- DONE 2026-07-02: fp8 (E4M3) KV cache, opt-in via `Q27_KV=fp8`.**
Halves KV (68 -> 34 KB/token); measure-first probe (`--kvstats`, 8K wikitext
tokens through prefill) showed K amax <= 21.8 / V amax <= 118.6 vs the E4M3
448 saturation bound with negligible sub-denormal mass, so the design is
scale-free saturating conversion -- per-row scales only pay when a float
format is range-limited, and this one is not. All 3 store kernels + 3
attention consumers (flash-decode, MMA and lite FA prefill) templated on the
cache element type; fp16 stays the default and its canonicals hold bitwise
(md5 re-verified post-refactor). Gates: unit tests prove store == host
saturating cvt bitwise and fp8 kernels == fp16 kernels on the dequantized
cache bitwise (E4M3 is exact in fp16); corpus PPL 7.1889 vs 7.1928 (-0.05%,
noise); `--nll-long 65536` buckets flat and within 0.3% of fp16; needle 3/3
@55K through the server; logit A/B @512-tok prompt: cosine 0.9995, top-1
exact, KL 3.4e-5; pf/pfcache identical under fp8; acceptance 3.64 vs 3.67
t/round (-1% @2K soak). Wins: decode @28.5K ctx 105.7 -> 117.2 t/s (+11%),
alloc ceiling ~180K -> ~370K [superseded -- P3's 5th GDN buffer set: ~355K]
(262K native allocates and runs; cold TTFT unchanged). Risk-6 tolerance-gate machinery now exists for future fp paths.
Correctness then validated to the new ceiling (see risk 5): fp16-vs-fp8 NLL
A/B flat to 160K (+-0.06% per bucket), fp8 NLL flat to 370K on a 783K-token
corpus, needle 6/6 on a 361.5K-token haystack including two placements
beyond the 262K native limit; 19.1 t/s decode at 361K depth.

**P1.5 -- DONE 2026-07-02: cold 28.1K TTFT 35.7s -> 24.3s.**
k_attn_prefill_mma: fp16 flash-attention prefill on mma.sync.m16n8k16 --
block per (kv head, 16-token tile), 6 warps = 6 GQA q-heads, K/V slabs
smem-shared, S->P register-identity reuse, fp32 softmax/O. Attention at 28K:
~16s -> ~4s. Gates: unit A/B vs FA-lite at 3.8e-4 (edge shapes T=23/base=37),
PPL 7.2139 (+0.006%), needle 3/3 @64K, canonicals untouched, 3-lens
adversarial kernel review with 0 confirmed findings; Q27_ATTN_PF=lite
fallback. Weight-checksum tool also landed: baseline at load, --verify-weights
(CLI) and /health?verify=1 (server), 867 tensors in ~20 ms. Review minors
fixed: prompt>ctx now refused in generate() (was a silent KV overrun,
pre-existing), dead pf_scratch removed (~100 MB @32K). Remaining minors
noted: both attention prefill kernels hardcode gqa=6/head_dim=256 (fine for
this model, silent wrongness if reused elsewhere); Q fp16 saturation above
65504 is theoretical for rmsnormed heads. Next long-ctx cost: delta_scan_T
(~3.8s @28K), then GEMM tile tuning at short ctx.

**P3 -- DONE 2026-07-02: depth-4 speculation (2K soak 210.4 -> 218.6 t/s,
28.5K-depth fp8 decode 117.2 -> 126.6 t/s).**
- Round-cost audit (nsys, --cuda-graph-trace=node): the argmax hypothesis was
  WRONG -- all 7 argmax chains total ~0.03 ms/round (0.2%); graphs already
  amortize launch overhead. Real budget: 4-lane batch GEMVs ~65%, the 3
  sequential MTP draft passes ~13% (of which ~1.1 ms = three 636 MB Q4
  draft-head reads, bandwidth-irreducible per pass), delta_step ~5%,
  small-kernel soup the rest. Argmax batching NOT built (measured pointless).
- Depth-4 gate: --stats pass-4 chain measured p(d4|prefix-3) = 97.4% overall
  (bar was ~60%), projecting +0.89 t/round ungated. BUILT: 5th lane (e),
  batch-5 verify, 4-pass draft chain, 5 GDN state buffers with mod-5 perm,
  5 captured graphs, ctx guard P+6. Soak: 4.36 t/round (was 3.67), 71% of
  rounds accept all 5; round cost +14% (4th MTP pass + 5-lane scaling) ->
  net +3.9% @2K, +8% @28.5K depth (rounds are weight-dominated there, so the
  extra lane's KV sweep doesn't bite). Canonical n=128 md5 UNCHANGED --
  greedy output stays bit-identical, as with E6. Cost: +610 MB (5th GDN
  buffer set) -> fp8 ctx ceiling ~370K -> ~355K; 262K native still fits.
- pf_scratch remnants removed (dead `scratch`/`max_ctx` params dropped from
  attn_prefill_T; the allocation itself died in P1.5).

**P4 -- DONE 2026-07-02: split-position FA prefill (long-ctx prefill ~2x;
cold 361.5K request 1324s -> 764s).** nsys on a pure 26.6K prefill showed
attention at 27% and climbing quadratically (~85% at 361K), and the cause was
SM starvation, not bandwidth: grid (4 kv heads, chunk/16 tiles) = 64 blocks
of 6 warps on a 170-SM part, ~20 TFLOPS sustained (~10% of fp16 MMA peak).
Fix: gridDim.z splits each tile's causal position range into PP-aligned
slices; each split emits an unnormalized {m, l, O[256]} partial per (q-head,
row) and k_attn_pf_combine merges (flash-decode's trick applied to prefill).
nsplit auto-scales with depth ((base+SB)/4096, capped 8, Q27_PF_SPLIT
overrides; 1 = bit-identical pre-split path, always used at short ctx).
Partials scratch 51 MB; combine cost 0.1%. Measured: attention kernel
6.01s -> 3.12s @26.6K (1.93x); 128K prefill wall ~153s -> ~78s (1.96x)
[method differs from P6's fp16-KV kvstats numbers -- see roadmap open
verification]; cold
28.5K TTFT 24.7s -> 21.4s (attention is only ~27% there); cold 361.5K
1324s -> 764s (1.73x, ~12.6 min) with the deep needle still retrieving
exactly. Gates: unit split=5-vs-1 at 1.9e-5 (empty tail slices exercised),
fp8==fp16(deq) bitwise identities hold under splits, canonical md5 exact,
pf/pfcache IDENTICAL, nll-long 32K split-on/off equal to the 4th decimal,
needle 3/3 @55K with verbatim-identical answers. Remaining long-ctx costs:
delta_scan_T (~50s @361K), GEMM tile tuning at short ctx.

**P5 -- DONE 2026-07-02: GEMM tile tuning (prefill GEMM -36%/-48%, cold
28.5K TTFT 21.4s -> 16.8s [superseded -- P6: 15.0s], short-prompt prefill
1388 -> 1790 t/s).**
Ladder, each step measured on an ffn_gate micro (17408x5120, T=256):
1. Grid swap (token blocks on blockIdx.x): the old row-major schedule
   re-read weights from DRAM once per 32-token block (8x traffic at T=256);
   swapped order gets 94% L2 hit rate (ncu). Alone it was NEUTRAL --
   the kernel was latency-bound, not BW-bound (2 blocks/SM, reg-limited,
   31% occupancy).
2. Register-pipelined staging: stage st+1's global loads issue right after
   stage st's smem stores, hiding DRAM latency behind mma work. 258 -> ~248.
3. Vectorized Q4 nibble unpack (byte_perm + vsub4, 2 u32 smem stores
   replacing 8 byte stores): 258 -> 217.
4. NT 32 -> 64 (halves token blocks, L2 traffic, and duplicated staging):
   217 -> 204 us.
Negatives (local optimum, do not retry): __launch_bounds__ 3 blocks/SM
(spills, 254), double-buffered smem at NT=64 (225) and NT=32 (237).
In-engine @26.6K: Q4 GEMM 9.78 -> 6.26s, Q8 1.44 -> 0.75s; whole prefill
GPU 22.2 -> 15.2s vs the pre-P4 morning baseline (1.46x today combined).
128K prefill wall ~78 -> ~57s (prefill-only) [does not reconcile with P6's
fp16-KV 117.6s -- likely fp8-KV runs; see roadmap open verification]. All
arithmetic unchanged --
unit errors byte-identical, canonical md5 exact, pf/pfcache IDENTICAL,
soak 217.0 (decode untouched). Prefill cost order @26K is now
delta_scan_T 25% / attention 21% / Q4 GEMM 41%.

**P6 -- DONE 2026-07-02: column-split delta scan, kernel 1.81x, 26K prefill
wall 15.0 -> 13.5s.** k_delta_scan_T was the same SM-starvation disease P4
fixed for attention: 48 blocks (one per GDN head) on 170 SMs, 756us per
256-token launch, 25% of prefill @26K. The 128 S-columns per head are
independent (pred_j, dj and o_j read only column j), so
k_delta_scan_split<NCOL> slices them across gridDim.x blocks and deepens
row-parallelism per column (NTILE=512/NCOL row tiles instead of 4; the two
serial 32-iteration row loops shrink to NCOL/4). Same 4-barriers-per-token
structure -- the legacy kernel's 5th trailing barrier is provably covered by
the next token's sq/sk barrier. Q27_DS_SPLIT forces 1/2/4/8; 1 = untouched
legacy kernel; auto default 8 (measured 748 -> 428/456/413us for 2/4/8 --
4 is reproducibly WORST: 192 blocks is the awkward wave count on 170 SMs;
384 balances best). No combine kernel needed (unlike P4): the split is pure
parallelism, only sq/sk staging duplicates (CS x 1KB/token from L2).
Measured: 26K prefill 15.02 -> 13.48s (-10.3%), 28.5K 16.69 -> 14.96s, 128K
125.5 -> 117.6s (all fp16-KV kvstats-method A/B, same binary; --kvstats now
prints prefill wall time). Gates: split-vs-exact 5e-8 at T=1 AND T=64
(tolerance-gated like P4 -- row reductions reorder), full-corpus PPL 7.1931
vs 7.1928 (+0.004%), canonical md5 exact at split 1/4/8, --pf 200
continuations IDENTICAL, --pfdbg maxdiffs same order as the split=1
baseline. Test lesson worth keeping: the first unit-test run FAILED at 8e-2
with raw-normal conv data -- the delta update S += beta*k(v - k'S) is
chaotic when ||k||~11, amplifying legitimate 1e-7 reorder noise; the engine
l2-normalizes q/k per head before the scan (l2norm_heads_T), and with
in-contract data the split matches to 5e-8. Test data must honor kernel
input contracts before a tolerance FAIL means anything. Prefill cost order
@26K is now Q4 GEMM ~46% / attention ~23% / delta_scan ~15%.

**Task-level quality A/B + P7 constrained decoding -- DONE 2026-07-03.**
q27 v1.4 4-bit vs Q5_K_M (llama.cpp + MTP), Thunderdome standard suite
T1-T10, CRUSH harness, no-think + greedy both legs, n=3 per task:
**overall 0.786 vs 0.786 -- DEAD EVEN (30 trials/leg).**
Per-task deltas: collab-server +0.103 (q27), fts +0.023, task-queue
+0.022, plugin/ecommerce/monorepo/ssg within +-0.002, time-tracker -0.016,
phantom-invoice -0.063, analytics -0.073 (bimodal 0.48/0.83 on BOTH legs
-- task variance, not quant). Greedy determinism made n=3 near-zero
variance on most tasks. **The +3.05% PPL does not appear in agentic
coding.** What DID appear: five tool-format drift modes under no-think
greedy (dropped <tool_call> wrapper; unterminated JSON w/ </file> junk;
<content>-tagged raw values; {"tool_call": JSON-keyed opener; raw control
chars inside JSON strings) -- structurally masked on the llama leg by
grammar-constrained decoding, initially FATAL on q27 (task-queue 0.000,
plugin 0.185 with zero writes executed), now fully recovered by the
tolerant parser chain in api_common.h (17 recoveries in the final rerun,
scores 0.782/0.899). Verdict: the quant is clean; tool-call discipline is
a SERVING-LAYER property. **P7 constrained decoding SHIPPED 2026-07-03**
(`--constrain-tools`): ToolGrammar + lazy mask cache + slot-0 masked
verify + in-grammar acceptance cap + pending-token mask staging. E2E
clean on time-tracker 0.84 / task-queue / collab 0.836 / plugin 0.903 --
zero disengages, zero fallbacks needed for wrapped calls. (An early
deterministic "0x65 rejected" disengage was root-caused to one-token-
lagged masks before on_pending staging existed -- stale masks FORCE
illegal tokens; gone since.) In-call throughput ~22 t/s (acceptance
capped at 1/round in-grammar; drafts are generated inside the round
graph and cannot be host-constrained -- split draft/verify graphs is the
known optimization if tool-span speed ever matters). [P11 SHIPPED
2026-07-03: split draft/verify graphs -- host reads the 4 drafts back and
stages per-lane masks, verify uncapped; in-call 49 -> 204 t/s (4.2x),
token-identical to the capped path.]

**P8 -- DONE 2026-07-03: stable-prefix snapshot.** Root cause of the 7.9x
eval wall-time: the snapshot included the volatile prompt tail
(assistant-open + no-think prefill), which every re-rendering client
replaces next turn -- divergence ~6 tokens before snapshot end, voided by
the all-or-nothing check -> full re-prefill EVERY turn. The old --pfcache
gate appended raw tokens (a flow no real client takes) and hid it since
M6.5. Fix: chatml_prompt reports the boundary (end of last input message,
always abutting <|im_start|> so split-encoding is tokenization-invariant),
generate() prefills in two stages and snapshots at the boundary. Gate v2
uses a tail-divergent turn 2: warm restore, continuations IDENTICAL.
Measured: collab-server trial 2434s -> 536s (4.5x), score unchanged
(0.836), prefix_hit=54-58K logged turn over turn -- the first real-traffic
cache hits this server has ever had. Wall-time refresh on the full
P7+P8+P9 stack (2026-07-03, worst two remaining tasks, n=3):
analytics-dashboard 1954s -> 667s AND score 0.641 -> 0.820 (constraint
fixed drift that was costing points, now beats the Q5 leg's 0.715);
ecommerce-backend 1025s -> 199s (score 0.518 unchanged). The 13-19x
pathological multipliers are now a uniform ~3-4x vs llama.cpp; the
residual is per-turn suffix prefill + the in-call constraint cap.
Remaining for TRUE mid-history edits (client compaction, edited files):
periodic GDN checkpoints -- snapshot S + conv rings every N tokens,
restart from nearest checkpoint <= divergence (llama.cpp PR #24785 /
commit b9180 n_rs_seq is the reference design). State is ~28 MB/layer-set
snapshot. [superseded -- P9 shipped the same-session checkpoint ring,
below; the cross-session pool stays parked]

**P9 -- DONE 2026-07-03: same-session GDN checkpoint ring.** Host-pinned
snapshots on a ring; mid-history divergence restores from the nearest
checkpoint <= the divergence point instead of a cold re-prefill (the P8
note's design, built). Gate: pfcache v3, warm == cold continuations
IDENTICAL. Cross-session checkpoint pool remains parked (see roadmap).

**Depth-5 gate -- MEASURED 2026-07-03, parked (pass-5 stats rig).**
**p(d5|prefix4) = 96.8%**, p(prefix4) = 89.0%, +0.862 t/round ungated
per-position. Applying the
stats-vs-live discount P3 exhibited (+0.890 projected -> +3.9% live),
depth-5 nets **~+2-4% @2K** against ~+12-14% round cost (5th sequential
MTP head pass + 6-lane verify) and a 6th GDN buffer set (-610MB fp8 ctx
ceiling). Real but modest -- build only if decode t/s becomes the
priority again. Margin-gating buys little on the soak (ungated chains
already clean); the think-heavy/high-entropy acceptance measurement
remains the open question before ANY depth change ships.

## 2026-07-04 -- the llama.cpp wall-clock chase (bench-time-tracker)

Goal: beat llama.cpp Q5_K_M wall-to-wall on Thunderdome bench-time-tracker at
equal-or-better score. Same-day bar (basins reroll daily -- prompts embed
dates; llama's basin is day-stable): q5km 19/19/20s @ 0.856. q27 start of
day: 96-138s with one score-0 day; end: 23/24/24s @ 0.849.

Landed:
- `--constrain-tools` OFF for serving. The capped grammar path has an
  engage-lag hole: the first post-engage token samples unmasked, a
  hallucinated tool name gets one grammar-forced char spliced in
  (`getg_project`), the grammar disengages on the next illegal byte, greedy
  loops on "tool not found" -> score 0. The bare-call parser chain alone
  matches llama.cpp behavior at 3.7x less wall (107s -> 29s).
- Prefill 1990 -> 2614 t/s @16K, bitwise: PF_T 256->1024 + GEMM NT 64->128
  (b1d7d88); attn ldmatrix.trans V-frags + vectorized KV staging (a7d209e,
  1.31x on that kernel). Note: chunking changes attention nsplit boundaries,
  so long-prompt greedy basins reroll even when canonical stays bitwise.
- Chunked-WY delta scan, DEFAULT OFF (76b524d, Q27_DS_MODE=wy): GDN
  recurrence as 64-token-chunk products + warp-private forward substitution,
  log-space decay ratios (lambda underflows f32 over a chunk). Derivation
  1e-15 in f64; 16-case tolerance test at 1e-7 incl. underflow regime;
  continuations IDENTICAL. Perf round pending: scalar phases run
  latency-bound (2.51 vs 1.67 ms/call) -- warp-tiling next.

Measured-dead (don't retry without new facts):
- Fixed depth-5/6 spec: built fully (7b25921, 871c852; canonical bitwise),
  -3 to -12% everywhere, reverted (79ff1e5, 27b663a).
- ADAPTIVE depth-4/5: built, all gates passed, still net-negative -- at 16K a
  d4 round is 26.7ms and a d5 round 33.4ms, so a 100%-accepted d5 round
  (6/33.4 = 180 t/s) loses to a full d4 round (5/26.7 = 187 t/s). No trigger
  can beat that inequality; work stashed. Reopen only if the d5 round delta
  drops under ~+5ms (suspects: nb=6 gemv template register cliff, 5th MTP
  pass attention).
- f16-GEMM tile restructure (-1.5%); 5-lane GEMV smem staging (-4% in-engine
  despite +6-11% in an isolated micro -- isolated micros lie for smem-hungry
  kernels, in-graph occupancy dominates); GEMM ldmatrix (+3%, not ported);
  PF_T 2048 / fp16-KV / PF_SPLIT basin tickets (0.836-0.85).

Profiling: decode profiling REQUIRES `nsys --cuda-graph-trace=node` (default
kern_sum silently omits graph-launched kernels). Round anatomy @2K: verify
batch-5 GEMVs 51% at ~60% DRAM BW; drafts 4-5ms of ~23ms. Q4 prefill GEMM
244 TOPS = 29% int8 peak; int-accumulate ceiling probe 312 TOPS.

The residual 4.2s vs llama decomposes ~2.2s engine + ~2.3s trajectory (their
basin writes 26% fewer output chars across 6 turns vs our 8 -- same model,
quant numerics pick the basin). Open: delta-WY tiling, activation regroup
32->64 (breaks the serial-vs-batched identity gate BY DESIGN -- policy
decision), basin lottery for the last stretch.

## 2026-07-04 (later) -- both open prefill levers landed; 16K wall 2556 -> 3180 t/s

- Delta-WY warp tiling (60deb75): k_delta_wy 2.51 -> 0.55 ms/call (4.5x).
  Warp-tiled register GEMMs for all three scalar phases, smem-A blocked
  forward substitution (8-row diagonal blocks in registers), QKt.R fold
  pre-substitution, shuffle-scan log-lambda. wy path 2913 t/s @16K vs seq
  2560 back-to-back. Rejected with numbers: 512-thread fat blocks (no gain,
  per-block latency chain dominates), L1 prefetch hints. Next lever if ever
  needed: cp.async K/Q panels (~3100 ceiling with subst zeroed).
- g64 activation regroup (this commit, `Q27_PF_XG`, DEFAULT ON): batched-
  prefill activations requantized per-64 (matches the Q4_G64 weight group) on
  xqT only; two K=32 mma.sync steps chain in int32 (new accumulating form)
  before ONE fp dequant step per 64 -- the fp chain was ~22% of GEMM cost.
  Q4 kernel regs 252 -> 174, zero spills. 16K kvstats A/B same binary:
  2556.3 (Q27_PF_XG=32 exact path) -> 2780.8 t/s (+8.8%); with wy on top,
  3179.6 t/s. PPL 7.1931 -> 7.1921 (noise). GATE POLICY CHANGE (Gabe
  sign-off 2026-07-04): per-64 amax changes int8 values vs the decode path's
  per-32, so serial-vs-batched identity CANNOT hold on the default path --
  replaced by test_kernels g64-vs-exact (g64 quantization pushed through the
  dp4a exact path via duplicated-scale expansion: same integer dots, fp
  grouping only; passed at 6e-7..6e-6 vs 1e-4), corpus PPL, canonical md5
  (canonical CLI prefills serially -- still bitwise, verified both modes),
  and a thunderdome spot-check. `--pf` enforces identity only under
  Q27_PF_XG=32 (verified IDENTICAL); on the g64 default it reports mismatch
  as expected-not-fatal (this probe happened to stay IDENTICAL at 8K).
  Decode lanes untouched: canonical md5 58b6ae85... exact in both modes.

Stack state after both: 16K prefill 3179.6 t/s (was 2556-2634 band same-day),
with Q27_DS_MODE=wy still opt-in pending the time-tracker rerun; flip to
default if the n=3 rerun holds. Remaining vs llama: decode gap + basin.

## 2026-07-04 (evening) -- burst-depth measured DEAD; deep acceptance is NOT the constraint

`--burst-stats` rig added (chain 10 MTP draft passes per free-region
position on the serial path, CSV dump of drafts + top1-top2 margins;
production-faithful: prompt phase does actual-next-token MTP KV warmup,
free region chains from the accepted token). Run: 2489 positions of code
continuation (time-tracker task.md + src + tests as 4.7K-token prompt,
no-think, greedy).

Result: the chain barely decays -- p(d_k|prefix k-1) = 92-94% FLAT to
depth 10 on code; mean chain 7.62; p(chain>=10) = 54%. The model would
happily accept 10-deep drafts half the time.

Economics kill it anyway. Round simulation on the actual trajectory
(faithful rejection clustering): t/round 4.51 (d4) -> 7.80 (d10) -- only
+73% tokens for +150% depth, because rounds restart at rejection points.
Cost slope MEASURED at serving conditions (fp8 KV, 16K depth, same
prompt, wall/rounds): d4 27.44 ms/round, d5 (871c852 build) 33.94 --
**+6.5 ms/depth, same as fp16's +6.75** (slope is draft-pass + verify-GEMV
dominated; fp8 attention savings are noise). At +6.5: fixed d6/d8/d10 =
-5/-13/-18%, best gated config (full-accept ramp) = -7%. Breakeven slope
~3.5 ms/depth = the engine would need to nearly halve per-depth cost
(ffn verify GEMVs add ~16% of the nb5 cost PER LANE past 5 -- gemv10
micro ratio 0.91 -- and each sequential draft pass is ~1.5 ms).

Verdict: no depth increase wins at the current engine structure, gated or
not, despite outstanding deep acceptance. The acceptance data is the
durable asset: if drafting were ~free (parallel draft backbone a la
DSpark -- needs training, not portable), the ceiling is large. Decode
stays at the d4 local optimum. Rig kept (--burst-stats N / --burst-out).

## 2026-07-04 (night) -- R0: real-work telemetry + anatomy; goal reframed to real agentic wall-clock

Gabe reframed the goal: optimize wall-to-wall for REAL agentic work (Claude
Code + subagents, long CRUSH tasks), not the bench-time-tracker sprint. R0 =
instrument, measure, rank levers. Codex-suggested improvements evaluated
against measurements below.

**Shipped (commits 8b73164, 6960d83, 60bc172):**
- `[req]` per-request stderr line, all six API call sites: conv fingerprint
  (fnv1a64 system+first-user), qw_ms queue wait, tok_ms render+encode,
  prompt/hit/ckpt/pf tokens, pf_ms, dec/dec_ms, cb_ms (client-write time
  inside on_token), rounds, tps, end reason, t= ms since server start.
  Engine::GenStats fills from generate() -- host-side only, no new syncs.
  Server-level gate tools/reqlog_gate.sh (11 asserts, RED watched at HEAD).
- UTF-8 crash fix: the FIRST real Claude Code session killed the server --
  BPE split an em dash across tokens, the per-token SSE delta was invalid
  UTF-8, nlohmann dump() threw type_error.316, uncaught -> std::terminate.
  ~100s of CRUSH trials never tripped it; Claude-flavored output did in
  minutes. Fix: q27::Utf8Gate (api_common.h) buffers incomplete trailing
  sequences on all 6 generation text pipelines, flush() -> U+FFFD; plus
  dump-time error_handler_t::replace on all response serialization.
  Self-test (9 cases) in test_tokenizer.
- Gates: canonical md5 58b6ae85 EXACT, --pf 200 seq+32 IDENTICAL,
  test_kernels ALL PASS, reqlog 11/11, utf8 self-test PASS.

**Finding -- pf-identity gate is N-sensitive (pre-existing, stash-verified
at HEAD):** --pf 64 seq+32 MISMATCHES (divergence past ~16 continuation
tokens) while the documented --pf 200 passes. Full exact pins
(PREFILL=dp4a ATTN_PF=lite DS_SPLIT=1 PF_SPLIT=1) restore identity at 64.
Read: the degenerate 5-token-cycle prompt sits on argmax near-ties that flip
on 1e-7 reorder noise from the auto-split/mma paths; the gate is
"argmax-stable at N=200", not "bitwise at all N". Gate invocations should
pin N=200 (or the full exact env set).

**Anatomy A -- real Claude Code session (claude -p + Explore subagent,
tinylog task; SUCCESS 12/12 tests, 178s wall, 35 reqs, 2 convs):**
- GPU 99% busy: prefill 79.9s (45%) + decode 95.5s (54%); idle 1%.
- Interleave phase: main (28-33K) x subagent (11-18K) strictly alternate,
  EVERY request hit=0 (each conv switch does reset()+ckpt_clear()).
  44.3s = 25% OF SESSION WALL was re-prefill of already-served context;
  queue waits up to 14s (client parallelizes, single slot serializes).
- Solo phase: near-perfect reuse to 60.6K (warm turns 0.2-2.8s). P8 holds
  on real Claude Code re-rendering traffic.
- Decode vs depth: ~130 t/s @30K -> 73-105 @55-61K.
- tok_ms 8-26ms even at 61K prompts; cb_ms ~0. Codex items "tokenizer BPE
  rewrite" and "decouple SSE writes" are MEASURED DEAD for real work.

**Anatomy B -- CRUSH long tasks, same-day q27 vs llama (n=1 each; q27
serving stack: fast-head, fp8 KV, no-think, PF_XG=32, constraint OFF):**
- collab-server: q27 223s @ 0.840 (98% prefix reuse, prefill 28.4s, decode
  145.4s = 65% of wall, 16.4K tok @ 113 t/s avg, ctx to 65K, idle 19%)
  vs llama 112s @ 0.844. First q27 attempt was a 20s score-0 one-shot-quit
  basin (2 turns, 3.9K diff, ended without tool calls) -- retry normal.
- analytics-dashboard: q27 257s @ 0.847 (decode 195.6s = 76% of wall,
  20.2K tok @ 103 t/s, ctx to 83K, 97% reuse, idle 31%) vs llama 132s @
  0.481 (llama's greedy basin FAILED hidden tests today, 0.000 subscore).
- Late-leg llama per-request decode: 104-153 t/s at comparable depth --
  q27's decode-rate gap at depth is ~1.2-1.4x, NOT 2x. The 2x wall gap is
  mostly OUTPUT TOKEN VOLUME (llama basins write roughly half the tokens;
  the time-tracker "26% fewer chars" pattern generalizes) on top of that
  1.2-1.4x. Old "3-4x multipliers" (constraint-era) are obsolete.

**Lever ranking for real-work wall (post-R0):**
1. P10-A1 per-slot engine state + longest-stable-prefix routing: kills the
   25%-of-wall interleave re-prefill class + subagent queue serialization.
   Asymmetric slots (main full-fat 131K; 1-2 light utility slots, no-spec,
   ~1.2GB each) fit VRAM; A1a landed, WY scratch must become per-engine
   first (prefill.cu:1749 process-global statics -- codex #5 confirmed).
2. Decode at depth (30-83K = where real sessions live; 65-76% of long-task
   wall): FIRST an nsys decode-slice at ~60K fp8 (the 16K attribution may
   not transfer -- attention share grows with depth), THEN pick between
   batch-verify tensor-core GEMV (codex #1; test_kernels gemv10 already
   shows 1x10 = 0.94x cost of 2x5 at kernel level) and quant-for-acceptance
   v1.5. Ceiling vs llama-at-depth ~1.2-1.4x.
3. Sampling (greedy loop escape; analytics/collab one-shot-quit and
   getg_project-class basins are all greedy pathologies) -- per
   docs/sampling-design.md; also caps worst-case wall on real tasks.
4. Output-volume basin gap vs llama: NOT engine-actionable (quant numerics
   lottery); for real use a terse-style system prompt is free wall time.
   For the bench goal it remains the binding ~2x.
- Parked with data: P11 (constraint off), SSE decouple, tokenizer rewrite,
  ckpt arena (minor; revisit when per-slot rings multiply the footprint),
  ldmatrix port (+1-3% prefill on a non-bottleneck).

Ops notes: q27-eval and llama-q5km-eval are TRANSIENT systemd-run units --
`systemctl --user stop` deletes them; recreate with systemd-run (specs in
this entry's session; q27-eval: Q27_PF_XG=32, --port 8081 --ctx 131072
--no-think --fast-head). reset-failed before recreate after a crash. Disk
on / at 99% (6.9G free) -- single trials fit (~2-100MB persisted) but the
07-02 full-suite no-space incident will recur on any big run; 23G of old
runs in ~/thunderdome/results/runs is the obvious reclaim, needs Gabe's
call. 3090 is occupied by vox-transcriber (20.4GB) -- P10-decision's
"vLLM on 3090" fallback is not currently viable.

## 2026-07-04 (late night) -- R1 LANDED: multi-slot serving; -34% real Claude Code session wall

**R1 (e8f71fd + hardening c618c91, pushed): `--slots N --slot1-ctx M`.**
N engines borrow the one uploaded weight set (P10-A1a ctor); each slot owns
its GDN snapshot, ckpt ring, KV, and P7 mask-pool ids. Routing under the gpu
mutex via Engine::reuse_len (have_snap-gated strict snapshot extension, else
best P9 checkpoint -- the same predicate generate() honors) > empty slot >
LRU; per-slot n_max re-clamp; largest-slot no-fit fallback. Rounds still
serialize (R1b later); the win is that interleaved conversations stop
destroying each other's prefix caches.

**Acceptance (same task, prompt, greedy, same day):** the R0 Claude Code
session (claude -p + Explore subagent, tinylog RFC3339 task) rerun against
--slots 2: **178.3s -> 118.3s wall (-33.7%)**; avoidable re-prefill 44.3s ->
0.0s; total prefill 79.9 -> 31.4s; queue wait 54.5 -> 24.4s; task succeeded
both runs (12/12 and 14/14 tests). Main pinned slot 0, subagent slot 1,
every interleaved turn warm (e.g. main turn 2: pf 1.6s vs 11.8s single-slot).

**Review (10-angle finder pass + sweep, 12 findings filed, all fix-class
CONFIRMED items landed in c618c91):** worst finds -- serial-path (<32-tok)
requests left stale snap_toks + a retained ckpt ring whose KV rows the short
request had overwritten (silent-corruption path, PRE-EXISTING since P9,
now cleared); n_max clamped against slot-0 ctx not the routed slot (ctx-guard
truncation surfaced as end_turn on the deployed 131K/32K config); the OAI
stream empty-piece skip had disabled per-token disconnect probing. Deferred,
documented: per-slot mask-pool mapping split-brain (latent while
constrain-tools is off), [req] schema triplication + six-site handler
boilerplate, gate python dedup, Utf8Gate per-site wiring.

**Serving:** q27-eval now runs `--slots 2` (slot 0 = 131072, slot 1 = 32768
fp8; ~27.7 GB resident, ~5 GB headroom). Subagent conversations larger than
~32K route to slot 0 by eligibility (logged, acceptable; raise --slot1-ctx
if telemetry shows it).

Gates at HEAD: reqlog two-phase PASS (interleave-warm C12 a/b hit=115/74),
canonical md5 58b6ae85 EXACT, --pf 200 seq+32 IDENTICAL, test_kernels ALL
PASS, utf8 self-test PASS.

## 2026-07-04 (post-R1) -- R1b prereq: WY scratch per-engine

delta_scan_wy's KKt/QKt panels were process-global statics (prefill.cu:1749)
-- fine while the gpu mutex serializes whole prefills, fatal for R1b's
interleaved rounds: two engines with chunks in flight would race one panel
set across streams (k_delta_wy_kk writes what the other engine's k_delta_wy
is still reading), and the lazy regrow cudaFrees panels the other stream's
queued kernels still reference. Now caller-owned q27k::WyScratch (kkt/qkt/
cap_nch), one per Engine (wy_scratch member), threaded through delta_scan_T;
regrow syncs the owning stream before freeing, since this engine's own
earlier chunks may still be reading the old panels. seq path ignores it.

TDD red->green: new test_kernels case "wy stream isolation" -- two contexts
(T=512/1024, different nch so each regrows its own panels), ITERS=48 chained
scans interleaved across two streams with no host sync, compared BITWISE
against isolated serial references (same kernels + data + launch config, so
only scratch sharing can differ). RED against the shared statics: ctx A
failed with 2.4e6 mismatched words (ctx B's kk pass stomps A's 8 panels; B
passed on scheduling luck -- one context failing is the demonstration).
GREEN with per-context scratch: both bitwise-exact, err 0.

Gates: test_kernels ALL PASS, canonical n=128 md5 58b6ae85 EXACT (177.3 t/s
short bench, in family), --pf 200 seq+32 IDENTICAL (batched 1575 t/s),
q27-eval recreated on the new binary (--slots 2 config, health checked).
R1b round-granularity interleaving is now unblocked on the prefill side.
