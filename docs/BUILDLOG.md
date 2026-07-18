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

Review hardening (8-angle finder pass + adversarial verify, second commit):

(1) CONFIRMED pre-existing uninit read: k_delta_wy's QKt fold reads the
strict-upper triangle k_delta_wy_kk never writes (skip at ~1451: `ss > tt`
guards both stores; fold at ~1630 runs the full row range), neutralized only
by R == 0 at those positions -- 0 * NaN/Inf from recycled cudaMalloc pages is
NaN, poisoning live oT rows (S itself is safe: state update reads only
rhat/K; KKt reads all producer-guarded). The kernel zeroes rhat for exactly
this hazard class but missed the global-memory operand; a 0xFF-memset buffer
in engine init is an in-process NaN source for recycled pages. Fix: panels
cudaMemsetAsync'd to zero once per allocation in wy_grow -- no kernel stores
those entries, so zeros persist; 0 * 0 keeps the exact-zero semantics bitwise
(canonical unchanged, verified). Predates this work; per-engine allocation
merely widened fresh-page exposure.

(2) Eager reserve (3 finders converged): serving T is chunk-capped at
PF_T=1024 (engine.cuh prefill loops), so panels are a fixed 8 MiB/engine;
wy_scratch_reserve(PF_T) now runs at Engine init next to the other fal
buffers, making the lazy regrow unreachable in serving -- previously a short
first prompt (< 1024 tok) sized panels low and the first long prompt then
regrew MID-SERVING (stream drain + cudaFree/cudaMalloc under the global
allocator lock, hiccuping the sibling slot; also capture-unsafe if R1b ever
re-captures a slot while another serves). Lazy grow retained as the fallback
for callers that skip reserve (tests). WyScratch contract doc now states the
one-stream pin explicitly (regrow drains only the caller's stream).

(3) Test cleanups: shared l2norm_qk_host helper (contract block was
triplicated across test_delta_split/test_delta_wy/isolation; chaos rationale
now lives once), Ctx aggregate init de-noised, whole-buffer memcmp, unified
teardown idiom. Accepted residuals, documented: WyScratch* is required
non-null even on seq-pinned paths (do NOT normalize nullptr-passing -- wy is
the default mode, a null deref there is a mid-prefill segfault); regrow
branch has no in-flight-work test coverage (it is now test-only surface;
a deterministic UAF test would be timing-flaky); Engine still never frees
device buffers (house lifetime model, panels included, 8 MiB/engine).

Hardening gates: test_kernels ALL PASS (isolation bitwise both ctx),
canonical md5 58b6ae85 EXACT, --pf 200 seq+32 IDENTICAL, q27-eval recreated,
health ok.

## 2026-07-04 (late) -- R1b: round-granularity interleaved scheduling

Design docs/R1b-design.md; commits 3568823 + c615d8f + 0449131. R1 left
rounds serialized behind a whole-generation mutex: a request arriving
mid-generation waited for the other conversation's ENTIRE generation (R1
acceptance: 24.4s residual queue wait). R1b time-slices the GPU at the two
boundaries where engine device state is coherent -- decode rounds (~27ms,
spec_round host-syncs each) and prefill chunks (PF_T=1024, ~320ms).

**Mechanism.** q27::GpuGate (api_common.h): FIFO ticket lock; maybe_yield()
takes the re-enqueue ticket INSIDE the handover critical section (naive
release+acquire loses queue position to the next yielder when descheduled
-- caught RED by the C1 self-test, sequence 0,1,2,...,0,2,1). Solo path =
one relaxed atomic load per round. Engine::on_round_gap hook called between
rounds/chunks; the server's lambda drains the engine stream BEFORE
maybe_yield so handovers are real (prefill loop runs host-ahead). CLI never
sets the hook: canonical bitwise untouched (md5 EXACT, 177.5 t/s short
bench). Slot claiming: Slot::busy under route_m -- routing reads only
settled engines; all-busy waiters block on route_cv (barging accepted,
documented). LRU stamped at FREE (completion recency), not claim -- R1's
serialization made those equivalent, R1b doesn't; refused-class claims keep
their old stamp. Q27_NO_INTERLEAVE=1 = exact R1 serialization (flake-class
debug lever). Telemetry: gs.gw_ms/gs.yields, printed as gw=/yields= after
end= ([req] parse regexes stop there); pf_ms/dec_ms stay wall-inclusive, so
GPU-busy = pf+dec-gw; [gen-done] prints decode-phase parks only (pass-2
review fix: request-total gw over decode-only dt over-corrects).

**Gates (all green at 0449131).** tools/interleave_gate.sh NEW: overlap
(short B completes while long A streams, 400 deltas after), byte-identity
solo-vs-interleaved for both A and B, chunk-granularity admission proven
schedule-wise (B fully served 0.95s BEFORE cold-A's first delta), third
request queues and completes, kill switch serializes with yields=0. RED
against R1 build failed exactly on overlap+telemetry. GpuGate 4-case
self-test in test_tokenizer (FIFO rotation, solo fast path, 8x200 stress
exclusion). reqlog_gate PASS both phases; test_kernels ALL PASS; --pf 200
seq+32 IDENTICAL; 10x concurrent soak (alternating cold/warm A) byte-stable,
21 yielding requests. Two review passes (first: approve, 9 findings, LRU
stamp + S6b strengthening + stale-port guard landed; second adversarial
state-lifecycle pass: approve, 0 blockers, [gen-done] phase fix landed).
Deferred, documented: P7 hook-leak-on-throw class now includes on_round_gap
(harmless: every claim site reinstalls); same-conversation duplicate
requests fork to a second slot (cache-efficiency delta only, greedy output
unchanged); /health?verify comment corrected (default-stream xsum BARRIERS
engine streams briefly -- pre-existing).

**Acceptance A/B (same day, same workload, fresh server per leg): tinylog
RFC3339 task, claude -p + TWO parallel Explore subagents, --slots 2 prod
config.** No-interleave legs (=R1): 142.2s and 129.0s wall, qw_sum
105.1/69.1s, qw_max 17.3/15.1s, 34/25 reqs. R1b leg: **114.7s wall (beats
both controls), qw_sum 19.4s, 1024 yields, gw_sum 79.2s** -- and 18.3s of
that 19.4 is TWO engine-claim waits (big prompts needing busy slot 0:
13.5s + 4.8s); gate-level waits across the other 20 requests total ~1.1s.
Task success 14/14 tests all three legs. Per-request GPU work identical
across legs (4.1 vs 4.2s/req) -- the delta is scheduling, not workload
luck. READ: the R1 whole-generation queue-wait class is dead; what remains
is ENGINE-claim head-of-line when conversations outnumber slots (3 convs on
2 slots here) -- that is the --slots ceiling / light-utility-slot lever
(P10 asymmetric-slots note), not a gate problem.

Ops: acceptance legs ran claude -p with ANTHROPIC_BASE_URL at a local
server on 8082; leg driver + workspace seed rebuildable in minutes (tinylog
package, 8 logger tests + 6 RFC3339 contract tests incl. millis rounding
carry). q27-eval recreated on the new binary after the GPU window.

## 2026-07-04 (night) -- decode-at-depth ATTRIBUTED: k_attn_fd is 99% of the depth cost, latency-hiding-bound

Methodology: Q27_PROF_DECODE=1 brackets the decode loop with a cudaProfiler
range (engine.cuh; done() is the exit funnel) so `nsys profile
--cuda-graph-trace=node --capture-range=cudaProfilerApi` records ONLY the
decode slice from the SERVER (batched prefill, fp8 default) -- the CLI
--tokens serial-prefill trap is bypassed entirely. CAVEAT measured: node
tracing inflates round WALL ~2.3x at 16K (+34ms/round host overhead) but
kernel-execution sums stay honest (kern_sum @16K = 26.5ms vs 27.3 true);
attribution uses ground-truth anchors (no nsys) + per-kernel deltas between
depths, which cancel the per-node overhead (same node count per round at
both depths). Workload: wikitext continuation, /v1/completions, 600 tok.

**Ground truth (no nsys): 16K = 27.3ms/round (108.2 t/s; matches the known
27.4), 61K = 47.2ms/round (78.0 t/s). Depth cost = +19.9ms/round over +44K
ctx (~0.45ms per 1K).** Per-kernel delta/round (nsys, 188/60 rounds):
k_attn_fd +19.92ms (7.59 -> 27.51, 20 inst/round both depths) = 99% OF THE
DELTA; k_gemv_q4_n +0.10 (10.7 -> 10.8 -- the verify GEMVs are DEPTH-FLAT);
everything else <=0.02. Kernel-sum delta 20.1 vs wall delta 19.9 (closes
within 1%). **The P10/roadmap "batch-verify tensor-core GEMV vs
quant-for-acceptance" fork is MEASURED-DEAD for the depth bucket: both
target weight-stream cost, which does not grow with depth.**

**k_attn_fd efficiency: ~91 GB/s effective KV read = 5% of DRAM peak, at
BOTH depths** (61K: 125MB/instance in 1.375ms avg). Probe 1 -- FD_NS 16->64
(4x block count; starvation hypothesis): 61K 47.2 -> 45.5ms/round only.
NOT grid-bound. (First attempt measured the OLD binary -- spec3.cuh is
missing from the Makefile dep lists, numbers identical to 0.1 t/s was the
tell; touch spec3.cu forced it. Dep-line fix proposed to Gabe separately.)
Probe 2 -- NW 8->4 @128 threads + FD_NS=64 (smem 55->30KB, resident warps
8 -> 12/SM): 61K 42.5ms/round (-10% wall, attn ~-17%), 16K 25.8. More
resident warps = directly faster with 12 still far under the 48-64 cap:
**latency-hiding-bound, occupancy capped by the 55KB smem accumulator**
(per warp: 6 heads x 256-float acc RMW in smem per position + 12 expf + 16
byte-granular K/V loads).

Fix design (next work item, kernel rewrite): (1) accumulator to REGISTERS
-- each lane owns acc[lane+32u], 48 regs/lane, smem drops to the 6KB q
tile, warps/SM 8 -> ~20+; (2) vectorized fp8 K/V loads (int2/int4, 16
byte-loads -> 2-4); (3) FD_NS then re-tuned for grid fill (composes).
Ceiling: attn at 30-50% DRAM BW = 3-6ms/round -> 61K round ~24-27ms =
**~120-140 t/s at depth** (from 78; llama late-leg samples 104-153).
Numerics: any split/warp-shape change reorders fp merges -- observed
trajectory tie-flips (16K leg: eos@177 vs n_max@600 across probe builds;
also warm-vs-cold at 61K diverged 163 vs 188 rounds via the 16K request's
PARTIAL last chunk changing WY chunk width -- same argmax-tie class as the
--pf 64 finding, first time seen serving-side). The rewrite therefore
lands with re-derived canonicals + PPL + needle + acceptance-rate gates
(g64 gate-policy precedent), not bitwise identity.

Probes REVERTED (spec3.* clean); shipped this session: the env-gated
profiler range only (canonical md5 EXACT, off-path zero-cost).

## 2026-07-05 (early) -- attn-fd2 LANDED: register-accumulator flash-decode, +62% decode at 61K

Design docs/attn-fd2-design.md; the fix for the decode-at-depth attribution
(previous entry). k_attn_fd2: per-lane register accumulators (48/lane, lane
owns dims {4l..4l+3, 128+4l..+3}), uchar4/uint2-vectorized K/V loads (16
byte-loads -> 4 word loads), smem 55.3KB -> 12.3KB (occupancy stops being
smem-capped), NW=4 x 128 threads, cross-warp merge barrier-SERIALIZED in
warp order (smem atomics would break run-to-run bitwise determinism).
FD2_NS=128 splits (own constant; FD_NS=16 frozen so Q27_FD=v1 reproduces
the historical kernel BIT-FOR-BIT -- verified, old canonical 58b6ae85
exact under the fallback). Empty splits early-return WITHOUT writing
partials; the combine kernel derives the used-split count from pos and
skips the rest (bitwise-identical for v1 -- skipped partials contributed
exactly 0; kills the +2.4%/round empty-split tax at 2K, now +1.3%).
Combine takes ns + pos at runtime. Scratch sized by FD_MAXNS (engine,
tests). FD2_NS sweep: 128 (0.156ms/inst @61K micro) beats 256 (0.169) and
16 (0.624); v1 = 0.768 -> 4.9x per instance, ~808 GB/s = 45% of DRAM peak
(from 5%).

**Serving ground truth (fp8, --fast-head, wikitext continuation): 61K
47.2 -> 29.2 ms/round = 78.0 -> 126.2 t/s (+62%); 16K 27.3 -> 22.5
ms/round (-18%, 166 t/s on this text). Acceptance parity exact: 163
rounds / 3.68 t/round at 61K on BOTH kernels. 2K short bench 20.03 ->
20.3 ms/round (+1.3%); printed tps 177.5 -> 160 is t/round lottery on the
tie-riddled canonical prompt (3.56 -> 3.25), not kernel cost.** The
depth slope of attention fell ~3x (0.45 -> ~0.15 ms/round per 1K ctx);
decode-at-depth was 65-76% of long-task wall, so CRUSH-class tasks should
gain ~25-40% wall. llama late-leg (104-153 t/s at depth) is now matched
from below at 61K.

TDD: RED on missing attn_decode3_fd2 symbol; unit gate = fd2 + v1 BOTH
against an exact double-precision host softmax reference (arbitration --
first fd2-vs-v1 comparison used per-element relative error on
statistically-zero random-input outputs and false-failed at 4e-3; metric
now max-abs/RMS), + run-to-run bitwise determinism, + default-dispatch==
fd2 bitwise, across seq {1,47,1024,16384,61440} x ntok {1,5} x {fp8,fp16}.
Numerics gates: --pf 200 seq+32 IDENTICAL; PPL fp16 7.1918 (bar 7.1928),
fp8 7.1833 (bar 7.1889) -- both in noise, both better; --nll-long 160K
fp8 BUCKET-IDENTICAL to 4dp vs v1 at all 8 depth buckets; interleave gate
PASS both phases (fd2 x R1b interplay, byte-identical solo-vs-interleaved).

**CANONICAL RE-DERIVED (gate policy per docs/attn-fd2-design.md, g64
precedent): n=128 md5 = 4c4120c72056aba2bc2d2561471eafce (fd2 default,
run-stable); 58b6ae856e8e10d10549878ac44417a4 remains valid ONLY under
Q27_FD=v1. Recipe unchanged (grep '^generated:' | md5sum, -n 128 --spec).
SOLVED EN ROUTE: 4c4120c7 is byte-identical to the 2026-07-04 "one-time
canonical md5 flake" (1-of-10, never reproduced) -- the canonical prompt
sits on an argmax tie whose two sides are the v1 and fd2 accumulation
orders; the adaptive-agent's experimental build must have perturbed the
same tie once. That flake is no longer unexplained; the P11 split crash
remains the only open item in the flake-pattern class.**

Stale-binary lesson AGAIN, new variant: after the Makefile dep fix, a
target-scoped `make build/test_kernels` left build/q27-server old --
measured a whole "fd2" server run on the pre-fd2 binary (identical
numbers + identical trajectory = the tell, second time this session).
Full `make`, always, no target names.

## 2026-07-05 -- same-day CRUSH A/B: fd2 banked; decode rate now BEATS llama at depth; residual gap = output volume

Both legs same day (basins reroll daily), n=1/task per the minimal-scope
policy, identical harness (CRUSH no-think greedy, T2 collab + T8 analytics,
greenfield/complex). q27 = fd2 serving stack (--slots 2, fp8, fast-head);
llama = Q5_K_M + draft-mtp n_max 6, q8 KV (the standing config, recovered
from the journal).

| task | q27 (fd2) | llama Q5_K_M | read |
|---|---|---|---|
| T2 collab | 230s @ 0.847 | 120s @ 0.843 | 1.92x wall, equal score |
| T8 analytics | 180s @ 0.825 | 190s @ 0.478 | q27 WINS wall AND score |

Telemetry decomposition (q27-eval [req] aggregates): T2 = 59 reqs, 22039
decode tokens at **161.3 t/s effective**, 137s decode + 47s prefill of 230s
wall, ctx to 74.7K; T8 = 62 reqs, 21785 tokens at **164.0 t/s**, ctx to
69.3K. Pre-fd2 the same trial class ran 103-113 t/s -- fd2 delivered +50%
real-trial decode rate, and q27's rate now exceeds llama's own late-leg
samples (108.8-153.9 t/s in its eval-time lines). The residual T2 wall gap
is OUTPUT VOLUME: q27's basin wrote 22K tokens where llama's wrote ~11K --
the R0 conclusion stands (not engine-actionable; terse-prompt / sampling
lever). T8: llama's analytics basin failed hidden tests (0.040) for the
second day running (0.481 on 07-04) -- q27 took both axes outright.

Wall-history for the same tasks, q27 leg: collab 2434s (pre-P8, 07-02) ->
223s (07-04) -> 230s today at DEEPER ctx (74.7K vs 30-83K band) and +50%
decode rate -- today's basin simply wrote more context; cross-day walls are
not comparable, which is why the A/B is same-day. README "State of the
engine" refreshed to 2026-07-05 (fd2 depth numbers, 160.2/209.2 short-ctx
labels, A/B row, sampling next).

Ops: llama-q5km-eval recreate command recovered via `journalctl --user -u
llama-q5km-eval` (transient units log their full bash -c in the journal --
easier than BUILDLOG archaeology). Disk / now at 80% (Gabe reclaimed old
runs). Servers swapped sequentially (both do not fit 32GB); q27-eval
restored and healthy after the llama leg.

## 2026-07-05 (later) -- red-team pass on the README: five claims audited, two retracted, one refuted back

External red-team review of the README's public claims; every item verified
against trial data / server logs / code before acting. Outcomes:

**1. "analytics WINS outright" RETRACTED.** Pooled analytics draws this week:
q27 {0.490, 0.600, 0.799, 0.825, 0.830, 0.831, 0.834, 0.847}, llama {0.478,
0.481, 0.483, 0.830, 0.831} -- bimodal on BOTH engines (q27 drew its own low
basin twice on 07-03 morning), and the 30-trial A/B scored analytics -0.073
AGAINST q27. llama's two low-basin days are ~2 draws at its ~3/5 empirical
low rate; Fisher exact on the pooled high/low split is p~0.24. One greedy
draw per leg separates nothing. README reframed: "llama sampled its low
basin, q27 didn't"; the analytics wall win (180 vs 190s) marked
basin-confounded. The 1.92x collab line also now states its DIRECTION (q27
slower) -- it read as a win.

**2. llama leg VERIFIED NOT HANDICAPPED (the load-bearing one).** The A/B
llama build is mainline b9857 (2026-07-01, includes the n_rs_seq machinery
P9 was modeled on; n_rs_seq auto-set to draft n_max=6). Its server log
(llama-q5km-eval.log; the M.SS.mmm.uuu-prefixed 7m23s instance starting
~23:51 PDT IS the A/B) shows hybrid context checkpoints active (212-278 MiB
creates + rotation during collab) and, through the analytics leg at 62-65K
ctx, "selected slot by LCP similarity ... f_keep = 0.99x" with per-turn
prompt evals of only 179-1260 tokens (suffix-only, zero re-prefill) and
draft-mtp mean chain 4.8-7.0. The collab volume story survives the
strongest-available-opponent test on this build; the README now says so
where the claim lives. Community-config sweep (draft depth, p_min; reported
"mean 140.7 @ Q6 patched" unreproduced) stays open before headline claims.

**3. constrain-tools honesty.** README no longer lists the flag as bare
"available": off in eval serving (engage-lag hole, 07-04 entry), in-grammar
cap 1/round = ~22 t/s in call bodies, 0.786 tie parser-carried (17
rescues), strict-parser rerun = open gate blocked on the engage-lag fix,
Anthropic-path-only wiring.

**4. Single-prompt short bench RETIRED as a benchmark; suite shipped.**
tools/shortbench_suite.sh: 5 fixed genre-diverse prompts (ids baked from
llama-tokenize --no-bos on the source GGUF) x 128 tok, stock, greedy
--spec, canonical prompt kept as leading bitwise gate. First run (fd2,
stock, offset verified 0): canonical md5 EXACT at 159.6 t/s; suite 157.2 /
174.7 / 158.5 / 190.8 / 165.6 = mean 169.4 t/s, t/round 3.20-3.88. The
+-10% per-prompt spread on trajectory alone is the point: the old single
number (177.5/160.2 tie lottery) sat inside its own noise band next to the
community's ~160; 169.4-over-5 is the number that can face a rerun.

**5. Constrained-decoding state audit (multi-slot x interleave x restore x
prefill-continuation).** The reviewer's specific fear -- P9 restore
resuming with a desynced grammar automaton -- is REFUTED in code:
ckpt_restore/snap_restore/reset touch only S/conv rings/positions/perm
(engine.cuh), grammar is per-request (ToolConstrainer local per handler)
and engages only on decoded output; no gate needed there. The audit found
three REAL latent cells instead, now queued as gates in the README roadmap:
(a) the R1-deferred split-brain is precisely located -- mask bytes+identity
in the process-global tool_mask_cache (server.cu:275) vs per-slot host2dev
map (server.cu:238) vs per-engine device pool (engine.cuh:96-101), coherent
only because nothing ever resets host2dev/mask_pool_used -- an unenforced
invariant, no assert; (b) pool-full divergence: per-engine pool caps at 512
while the shared cache is unbounded, and a full pool silently drop()s the
constraint on that slot only; (c) cross-request leak: d_mask_ids/
d_accept_cap are cleared only via tc.end() -- a non-CUDA throw between
generate() and tc.end() leaves the NEXT request on that slot decoding under
a stale lane-0 mask + accept-cap-1 (cheap fix when constrain-tools work
resumes: clear device constraint at request claim). R1b interleaving itself
is clean for the default capped path (all cache-touching callbacks precede
the round_gap yield; per-engine buffers drained at handover); Q27_TOOL_SPLIT
stays forbidden under --slots (P11 race, engine.cuh:789-803). Assistant-
prefill continuations ending mid-tool-call decode unconstrained by design.

**6. sampling-design.md gained an exit criterion:** quality A/B + drift
catalog re-run under the production sampling config before sampling
defaults on anywhere -- every quality number today is greedy-no-think
scoped, and temperature moves acceptance and drift together.

Strategic read accepted: the defensible claim is "fastest agentic wall
clock at depth with quality parity" (prefix cache + depth-4 + fd2 +
interleaving), not "fastest short-context decode" -- items 1-3 above are
what stood between that claim and being airtight.

## 2026-07-05 (evening) -- Claude Code head-mutation fix: billing-header cch normalization

First real Claude-Code-as-harness leg (thunderdome claude-code-q27-haight,
CC 2.1.119 in-container) exposed a P8-class hole: EVERY request logged
hit=0 ckpt=-1 with a per-request conv fingerprint, full 126K-token cold
prefill (~72s) per turn, ~50 turns/hour. Root cause: CC (<= 2.1.1xx era)
prefixes its system prompt with "x-anthropic-billing-header: cc_version=...;
cc_entrypoint=cli; cch=a5145;You are Claude Code..." and the cch stamp
CHANGES ON EVERY REQUEST -- the prompt HEAD mutates, so no token-level
prefix survives and both the P8 stable-prefix snapshot and P9 checkpoint
routing are voided for the entire conversation. P8 was built for volatile
TAILS; this is the volatile-HEAD dual. llama.cpp already ships the
countermeasure (normalize_anthropic_billing_header, ggml-org/llama.cpp
PR #21793) -- their Anthropic path was more Claude-Code-hardened than ours.
Host CC 2.1.200 (cc_entrypoint=sdk-cli) no longer sends cch -- verified by
tapping two sessions (byte-identical system, md5 5a18db38) -- so the class
is version-scoped but MUST be normalized to serve pinned/older CCs.

Fix: q27::normalize_cc_billing_header (api_common.h) pins the stamp bytes
to 'f' (same canonical bytes as llama's normalizer; tolerant of 1-16-char
stamps, header-segment-bounded, non-CC prompts untouched), applied in the
Anthropic anthropic_msgs system assembly AND in conv_fp so telemetry hashes
what the engine actually prefills. Self-test block in test_tokenizer
(mutating stamps normalize equal + cch=fffff; no-cch 2.1.200 header
untouched; stray body cch= untouched; long stamp pinned). Gates:
test_tokenizer 9/9 PASS; live pair A/B on the serving stack: two requests
differing only in cch -> conv IDENTICAL (c50bc714984869c1), turn 2
hit=376/397 pf=21 (was: hit=0 pf=full-context). CLI/decode paths untouched
(server-only), canonical unaffected.

Ops note: the failed leg also showed CC driving the T2 conversation to
126K+ tokens (slot-0 ctx is 131072) -- watch end= reasons near the ctx
ceiling on the rerun; CC's own compaction assumptions target Anthropic
model context windows, not ours.

## 2026-07-05 (late) -- Claude-Code-harness A/B: q27+CC vs llama+CC, native Anthropic both legs

First same-HARNESS cross-engine A/B: Claude Code (2.1.119, thunderdome
claude-code image) against q27 (port 8081, native /v1/messages) and llama
b9857 (port 8080, its own native /v1/messages -- no proxy either leg). New
adapters claude-code-q27-haight / claude-code-q5km-haight (byte-identical
modulo port/model alias) + thunderdome.yaml entries; thunderdome tree left
uncommitted with the rest of the eval scaffolding. Greedy + no-think parity:
q27 --no-think (greedy-only by construction); llama --temp 0 verified
effective via /slots after a real CC request (temperature 0.0 -- CC sends
no temperature field) + --chat-template-kwargs '{"enable_thinking":false}'
because --reasoning-budget 0 is a SILENT NO-OP for the qwen35 template
(thinking blocks still emitted; llama-leg gotcha #1). Gotcha #2: CC's
Workflow tool schema (maxLength 524288 on `script`) makes llama's
json-schema-to-grammar fail at sampler init ("failed to parse grammar") --
EVERY CC request 400s; bisected via a request tap + per-tool probes;
upstream-reportable. Workaround: --disallowed-tools Workflow on BOTH legs
(CC never invokes it un-prompted; tool block must stay byte-identical).

Results (same-day, n=1/task, matched 27-tool block unless noted):

| leg                        | T2 collab                  | T8 analytics                |
|----------------------------|----------------------------|-----------------------------|
| q27+CC (28-tool, pre-parity)| 374s @ 0.840 (43.0K dec tok)| 577s @ 0.292 CRASHED        |
| q27+CC (matched)           | 8s @ 0.00 one-shot-quit    | **167s @ 0.82**             |
| llama+CC (matched)         | **95s @ 0.85**             | 222s @ 0.50 (low basin)     |
| q27+CRUSH (same morning)   | 230s @ 0.847               | 180s @ 0.825                |
| llama+CRUSH (same morning) | 120s @ 0.843               | 190s @ 0.478                |

T8 head-to-head (both legs did real agentic work): q27 FASTER AND HIGHER
(167s@0.82 vs 222s@0.50). q27 T8 anatomy: 53 reqs, 16.6K decode tok (105s),
84K suffix prefill (41s), 2.47M tokens CACHE-HIT (the cch fix saved ~20 min
of re-prefill on this single trial), ONE P9 checkpoint-ring restore -- the
first real-client (Claude Code) restore sighting -- max ctx 63K, all eos.
Decode at depth 182-215 t/s. R1b interleaving live on CC's background
calls (gw/yields nonzero, slot 1 concurrent with slot 0).

T2 has NO matched-basin pair: q27's 27-tool T2 basin is a one-shot-quit
(model answers the task prompt conversationally, 184 tok, zero tool calls,
CC exits; retry 4s -- DETERMINISTIC same-day, the cache-hit replay proves
byte-identical prompts, so "retry normal" does NOT apply when nothing
perturbs the bytes). The 28-tool T2 run (374s @ 0.840, 43K output tokens)
proves the quit is basin, not structure: one extra tool schema in the
system block rerolled a full agentic run. llama's T2 basin was terse and
excellent (95s@0.85). Volume basin still owns the T2 wall story: q27 43K
output tokens vs llama ~13K (llama total across BOTH tasks: 29.6K dec in
210s, 327K prefill in 132s -- llama re-prefills 4x more than q27 post-fix
but rides a lean trajectory).

The pre-parity q27 T8 crash was a compound engine-adjacent failure worth
its own list of levers: (a) a single greedy generation blew past CC's
32000-output-token cap (mega-generation pathology; second live greedy
exhibit after one-shot-quit -- both are sampling's case); (b) the
conversation passed slot-0 ctx (131072) and q27 answered end=refused,
which CC treats as retryable rather than compact-now -- an
anthropic-shaped context-limit error would let CC recover; (c) q27 has NO
/v1/messages/count_tokens (404 in-log) so CC's compaction timing flies
blind -- llama implements it. (a) is model/sampling; (b)+(c) are cheap
server work, queued.

Verdict, stated carefully (this morning's red-team rules apply to our own
numbers): score comparisons at n=1/day are basin draws -- today q27 drew
{T2-quit, T8-high}, llama drew {T2-high, T8-low}; neither separates the
engines on quality. What the experiment DID establish: (1) q27 serves
Claude Code end-to-end at full cache efficiency (the cch fix was
load-bearing; hit=0 -> 97%+), (2) P8/P9/R1b all fired for a real CC
client, (3) on the one matched-basin pair (T8), q27+CC beat llama+CC on
wall AND score, (4) four concrete engine gaps now have names and owners.
Claude Code as a harness is VIABLE on q27 today; the wall story at depth
remains rate-won, volume-bound.

## 2026-07-05 (night) -- CC-harness robustness pair: /v1/messages/count_tokens + anthropic-shaped ctx-limit 400

Queue #1 from the A/B post-mortem, both halves of the compaction contract
Claude Code expects from an Anthropic endpoint:

count_tokens: POST /v1/messages/count_tokens now answers {"input_tokens": N}
(was 404 -> CC compaction timing flew blind). N is EXACTLY what /v1/messages
would report as usage.input_tokens for the same body: the request mapping
(anthropic_msgs + anthropic_tools_json, cch normalization included) moved
from server lambdas into api_common.h and both paths share it. count encodes
the whole rendered string, the served path split-encodes at the P8 boundary;
equal only because the boundary abuts the <|im_start|> added token --
gated in test_tokenizer alongside new shape self-tests for the mapping
(integration: count=186 == usage=186 on a tools+system body). CPU-only
handler: no slot claim, no GPU gate, safe concurrent with generations.

ctx-limit error: an over-ceiling prompt now 400s BEFORE slot claim / SSE
with the real API's envelope and byte format Claude Code substring-matches
for compact-now: {"type":"error","error":{"type":"invalid_request_error",
"message":"prompt is too long: N tokens > M maximum"}}. The old path
(engine end=refused inside an empty 200) read as retryable -- the T8
pre-parity crash looped exactly there. M = max_slot_ctx - 7 (n_max>=1 plus
spec rows P+1..P+6), so the refusal rule and the reported maximum agree by
construction. Streamed requests get the same plain-JSON 400 (provider not
yet started). Codex /v1/responses keeps its own context_length_exceeded.

Bad-JSON 400s upgraded from bare {"type":"error"} to the full envelope
(the SDK reads error.message; bare shape surfaced as empty errors).
Latent crash fixed en route: a message without "content" aborted on
nlohmann const operator[] assertion (json.hpp:21449, reproduced in test) --
any garbage POST could kill the server; now guarded in anthropic_msgs.

Verification: test_tokenizer all-PASS (new: anthropic api shapes 12 checks,
count==split-encode gate); 10/10 integration matrix on a throwaway
--ctx 4096 server (exact refusal string vs live count_tokens, stream and
nonstream, regression on small requests); q27-eval recreated on the new
binary, count_tokens live on 8081. Canonical gate not re-derived: no model
change, 400 fires before generate(), decode path untouched.

CC-level effect (compaction actually triggering instead of the retry loop)
gets validated the next time a deep-context trial runs -- queue: sampling.

## 2026-07-05 (night) -- Sampling Phase 1: plain-path sampler kernels (roadmap #2)

First increment of the sampling feature (docs/sampling-design.md). Phase 1 =
the three sampler primitives validated on-device in isolation, greedy path
bitwise-untouched. Spec rejection sampling (Phase 2), constrained+sampled
(Phase 3), graph capture, and server plumbing compose on top once the
numerics are proven -- this is the correctness foundation, nothing wired into
the decode loop yet.

New q27k::sample(logits, n, inv_temp, top_p, seed, pos, draw_kind, d_out,
d_scratch, d_nuc) in blocks.cu (temp>0 only; greedy stays on argmax via a host
branch, so canonical md5 is definitionally unchanged -- no edit touches the
greedy path):
- k_nucleus (single block, no atomics): max M, logsumexp logZ at inv_temp,
  and the top-p logit threshold via a fixed 12-iteration bisection on a prob
  cutoff (no sort). Single-block tree reduction = deterministic; fixed
  geometry = graph-capturable later. thresh clamped <= M so the argmax token
  is always in the nucleus (guards tiny/degenerate top_p).
- k_gumbel: argmax_i(inv_temp*x_i + G_i) over {x_i >= thresh}, G_i Gumbel from
  Philox4x32-10. Reuses argmax's am_pack+atomicMax (integer, order-independent
  -> deterministic token). Gumbel-max over the nucleus samples EXACTLY
  softmax(inv_temp*x) renormalized over the nucleus (Leviathan identity), so
  the sampler needs no CDF scan and no float-atomic nondeterminism.
- Philox stateless: counter=(pos,draw_kind,vocab_i), key=seed. Nothing mutable
  advances -> graph replay / prefix-cache restore / ckpt all stay consistent
  for free when this wires into the engine (design section 5).

test_kernels test_sample (synthetic 64-logit vectors, no model), all PASS:
- seeded identity: same (seed,pos) -> byte-identical token id.
- tiny top_p (1e-4) collapses nucleus to {argmax}; 32 draws all == argmax
  (ties the sampler to the greedy token, validates threshold+gumbel jointly).
- nucleus mass=0.9532 at top_p=0.95 (cnt=2): enough AND near-minimal (dropping
  the min member falls below top_p).
- chi-square vs analytic renormalized-truncated-softmax over the SAME GPU
  nucleus, 8192 deterministic draws (T=0.8, top_p=0.95): chi2=1.06, df=1,
  bound=52.3 -- near-perfect fit, and the test is deterministic so the bound
  never flakes. Any out-of-nucleus draw is a fatal (chi2=1e9) guard.
- temperature effect: nucleus 1 token at T=0.5 -> 31 at T=1.5.

Full test_kernels ALL PASS (existing dequant/gemv/mma/attn/fd2/wy/fp8 suite
unchanged); full make clean sm_120; test_tokenizer unaffected; q27-eval
recreated on the fresh binary (sample() is linked but never called by the
server yet, so serving behavior is byte-identical). Canonical md5 not
re-derived: no model/decode-path/graph edit.

Next (Phase 2, per design): k_spec_accept + k_sample_stop in a second 5-perm
graph set (greedy drafts + target-dist rejection acceptance), --stats
acceptance-vs-temp telemetry, spec==non-spec distribution gate. Then the
param-block + server plumbing that lets a request actually set temperature,
and the exit-criterion quality A/B under production sampling.

## 2026-07-05 (night) -- Sampling Phase 1 COMPLETE: param block + sampled graph + server plumbing

Wired the Phase-1 sampler kernels into an end-to-end sampled decode path. A
request that sets temperature>0 now samples; temperature==0/absent stays on the
greedy spec path, bitwise (canonical re-verified 4c4120c7 AFTER the build_graph
and generate() edits -- proof the greedy path did not move).

Param block: kernels refactored to read {inv_temp, top_p, seed} from a device
q27k::SampleParams* (was host scalars). One captured graph serves every request
-- the host rewrites *d_samp before the decode loop, the pointer is fixed at
capture. k_gumbel_d reads the key position from *d_pos (device), so graph replay
/ prefix restore / ckpt stay consistent with nothing mutable to advance.

Sampled graph: token_launches_sampled() = the plain forward with sample_g at the
tail instead of argmax; captured as a second graph (sample_graph) in build_graph
after a warm+reset, alongside (not replacing) the greedy token graph.

Decode integration (engine.cuh): sample_round() produces exactly one token so it
drops into generate()'s existing decode loop in place of spec_round (reusing its
ctx-guard / eos / on_token / round-gap logic -- minimal new loop code). First
token is an EAGER sample_g from the retained prefill logits (kind 0, no
forward -- re-forwarding the last prompt token would double-apply its GDN
recurrent update); later tokens replay sample_graph (kind 1). Eager-kind-0 vs
graph-kind-1 keying means the two never share a Philox counter even at the same
*d_pos. No MTP, no spec -- one token/round, correctness-first (Phase 2 adds spec
rejection sampling for speed). samp is an Engine member the server sets per
request, so generate()'s signature is unchanged across all 6 call sites.

Server plumbing (all 3 API shapes): parse_sample(body) maps
temperature/top_p/seed -> SampleParams (temp<=0 => greedy; top_p default 1; seed
honored for reproducible A/B). Set on eng before generate at every site; the 3
streaming providers capture the parsed params by value (body is out of scope in
the post-return chunked callback). Tool constraining is disabled under sampling
(tc.enabled &= inv_temp<=0; generate() also guards on_pending) -- constrained
+sampled is Phase 3.

Verification: canonical greedy md5 4c4120c72056... BITWISE OK; test_kernels
test_sample ALL PASS through the device-param sample_g (seeded identity,
tiny-top_p==argmax, nucleus 0.9532, chi2=0.34/df1, temp widens 1->31); full make
clean sm_120; live /v1/messages (throwaway --ctx 4096): greedy deterministic,
sampled seeded-identity (same seed -> byte-identical output), seed 123 vs 456
diverge coherently ("expanse of water..." vs "expanse of saltwater..."),
default-seed deterministic, sampled != greedy, high-temp terminates, streaming
+temperature emits deltas and stops. q27-eval recreated on the fresh binary
(CC path unchanged -- CC sends no temperature, so it stays greedy/bitwise).

Phase 1 DONE. Next: Phase 2 spec rejection sampling (k_spec_accept + k_sample_stop,
2nd 5-perm graph set, greedy drafts, --stats accept-vs-temp) for sampled decode
at spec speed; then the exit-criterion quality A/B + drift catalog under
production sampling. Open Phase-1 limitation: sampled decode is the slow plain
path (no MTP) -- fine for correctness/A-B, Phase 2 restores throughput.

**Sampling Phase 2 -- DONE 2026-07-05 (all gates pass).**
Sampled decode now rides the depth-4 speculative round instead of the slow plain
path. Draft half is byte-identical (greedy drafts); only the verify TAIL forks
into a 2nd captured 5-perm set (spec_sample_graph[]), so greedy stays bitwise.
New kernels (blocks.cu, launched only under temp>0): k_spec_accept walks the
Leviathan/Chen rejection test over the 4 greedy drafts -- accept dr_k with prob
p_served(dr)=softmax_full(dr)/nucleus_mass (q is a delta at the greedy draft, so
min(1,p/q)=p), first reject stops the chain; k_sample_stop resamples the new
pending from the stop lane's nucleus (exclude the rejected draft) via a
device-indirected Gumbel-max; k_finish_sampled does the h_next/d_P/outcome
bookkeeping keyed on the accepted count n. Refinement over the sec-1 sketch:
k_nucleus_d gained out[3]=nucleus_mass -- the accept prob is the RENORMALIZED
served prob, and using softmax_full(dr) under-accepts by 1/mass at top_p<1
(mass==1 when top_p>=1, so temp-only is unaffected). Philox: accept draws kind 2,
stop draws kind 3, both keyed on *d_P (round-start committed pos, strictly
increasing -> no cross-round collision; disjoint from the plain path's kinds 0/1).
Engine: spec_verify_forward factored out and shared by the greedy and sampled
tails; generate() + the CLI --spec loop route temp>0 to spec_sample_round
(Q27_SAMPLE_PLAIN=1 forces the plain sampler for the spec==non-spec A/B). CLI
gained --temp/--top-p/--seed. Acceptance-vs-temp telemetry on the [sample-stats]
line. Kernel gates (test_kernels --sampling-only, runs with no model resident):
nucleus mass vs CPU 6e-8; empirical accept rate 0.4333 vs p_served 0.4215
(softmax_full 0.388 would miss by ~8 sigma -> the mass fix is load-bearing);
rejection-sampling committed-token chi2 5.9/df4; seeded identity; rejected draft
never re-emitted -- ALL PASS. Build clean sm_86+sm_120, no new warnings. LIVE
GATES ALL PASS (tools/sampling_gate.sh, q27-eval briefly stopped for GPU 0):
greedy canonical md5 4c4120c7 UNCHANGED (greedy bitwise -- the spec_verify_forward
extraction + build_spec_graphs lambda refactor were inert), sampled seeded
identity, seed-varies, sampled!=greedy, spec/plain both valid. Acceptance-vs-temp
(tokens/round): greedy 3.43, T=0.3 3.59, T=0.7 3.45, T=1.0 1.90, T=1.5 1.00 --
holds near greedy through T<=0.7 (sharp draft head, E3's 98.1% agreement), sags
at high temp as the served target flattens (distribution-fidelity cost, as the
design predicted -- NOT an implementation cost). q27-eval restarted on the
Phase-2 binary (greedy unchanged so CC/eval traffic identical; gains the
sampled-spec path; log shows "5 greedy + 5 sampled perms"). See
docs/sampling-phase2-impl.md. NEXT: the exit-criterion Thunderdome quality A/B +
drift catalog under production sampling before sampling defaults on anywhere.

## 2026-07-06 -- sampling exit gate PASSED; tool-call PARSING was the agentic ceiling

Ran the exit gate (docs/sampling-exit-gate.md). Prereqs: **P1** `Q27_FORCE_TEMP`/
`Q27_FORCE_TOP_P` server default so CC/CRUSH (which send no temperature) exercise the
sampled path -- explicit request temp still wins; forced req draws a distinct LOGGED seed;
env-unset is byte-identical greedy (canonical 4c4120c7 is CLI-generated, untouched). **P2**
per-mode `[drift]` logging in parse_bare_tool_calls (modes 1-5 tag; UN-RESCUED flags an
intended call the chain couldn't recover). Both verified by standalone header tests + live.

The gate immediately surfaced the real story: **q27's low agentic scores were the
SERVING-LAYER tool-call parser, not quant or sampling.** Batch-1 (T8 n=5 CRUSH) scored
sampled 0.356 vs greedy 0.095 with BOTH legs dominated by un-rescued tool calls; the CC
harness one-shot-quit (7s, 1 turn, score 0). Three parser drifts, each root-caused from
REAL captured bytes and fixed in api_common.h (greedy canonical untouched throughout):
- **write-content**: file writes are `"content": "CODE</content>` -- JSON-quote-open value,
  stray `</content>` tag-close, unescaped quotes/newlines/braces in the body break the
  string and the code's braces corrupt the depth scan. escape_content_tags only fired on a
  `<content>` OPENING tag; extended it to the quote-open/tag-close shape (anchor `"content":`,
  rewrite the raw span). 6/6 of the batch-1 CRUSH failures.
- **CC mode-6**: `{"name":\n{ARGS}}` -- name string AND "arguments" key both absent. Added
  infer_tool_name (match orphaned arg-keys to a tool schema; exact required-set wins; refuse
  on tie) + scan_namedropped for the unbalanced `{"name":{ARGS}{"name":{ARGS}` batch. Tools
  JSON plumbed to parse_bare_tool_calls (streaming provider captures `tools` by value).
- **dangling-prefix hybrid** ("read all files in parallel"): a bare `{"name":\n` prepended to
  a batch of VALID calls; the dangling opener sent the scan into the truncated path whose
  `break` discarded the valid calls after it. Fixed: break only when the repair actually
  recovered a call, else advance-and-continue. `[drift] UN-RESCUED` gained `(ntools=N)` to
  separate plumbing (<=0) from schema-miss (>0) -- the diagnostic that pinned this one.

Live effect (fixed server, greedy T8): **CC 0.00 -> 0.466-0.553** (one-shot-quit -> 65-92
turn full trajectories, writes the whole project), **CRUSH greedy 0.095 -> 0.484**.

**Gate verdict -- PASS.** Fixed-path re-gate (T8 n=5/leg, same-day, greedy vs sampled-T0.7):
CC 0.466->0.553 (+.087), CRUSH 0.484->0.510 (+.026). Sampled >= greedy on BOTH harnesses;
no new drift mode; sampled:CC had FEWER un-rescued than greedy:CC (sampling's variation
escapes greedy's DETERMINISTIC failure basins). n=5 high-variance so "no regression detected,
plausible small benefit", not a nailed win. Speed neutral (T<=0.7 is the flat accept-vs-temp
region). **Sampling cleared to default at T<=0.7 / top_p 0.95.** The T8 bimodality (score
~0.48 vs ~0.85; hidden_tests ~0.04 usual vs 0.92-0.96 occasional) is MODEL quality --
structurally-complete but behaviorally-imperfect code -- orthogonal to the serving gate and
a separate thread. Residual un-rescued after all three fixes: a small non-fatal mode-6 tail.
Tree UNCOMMITTED (P1/P2 + three fixes + docs). Harnesses: claude-code-q27-haight (mode-6
exhibit), crush-q27-greedy-haight (the 0.786 baseline). [committed 3d60607; README 5a86007.]

## 2026-07-06 (later) -- measurement gaps closed (128K prefill; strongest-llama sweep)

Two open measurement debts, both resolved.

**128K prefill re-measure.** The roadmap flagged P5's "~57s" vs P6's fp16-KV "117.6s" as
un-reconcilable. Measured directly with `--kvstats 131072` (needs `--nll` int32 tokens;
prefill timing is value-independent -- it is GEMM + attention over positions, not token
content -- so a 140K synthetic-token file is faithful for TIMING; the fp8 refusal at the
KV-scan is AFTER the timing print, so fp8 still yields the number). Four combos:

    KV=fp8  g64-default  71.46s (1834 t/s)      <- production default, the honest number
    KV=fp16 g64-default  76.55s (1712 t/s)
    KV=fp8  Q27_PF_XG=32 75.47s (1737 t/s)      <- exact/identity path
    KV=fp16 Q27_PF_XG=32 80.36s (1631 t/s)

So current 128K prefill is **~71-80s** (KV format ~6%, g64-vs-exact ~6%). P6's 117.6s is
STALE -- it predates the g64 activation regroup (+8.8%) and delta-WY warp tiling (both
2026-07-04 later); P5's 57s was optimistic/extrapolated. (Gotcha: back-to-back CLI model
loads with no sleep hit the VRAM-teardown OOM race -- two combos came back empty until
re-run with a gap.)

**Strongest-opponent llama sweep.** Our cross-engine A/B ran llama draft=6 untuned; the
r/LocalLLM reference is draft=10 + p_min 0.5. Swept the 4 corners on llama-server (Q5_K_M
+ draft-mtp, single-stream greedy, ~2K ctx, decode t/s = `/completion` timings
predicted_per_second; llama-CLI hung in conversation mode even with -no-cnv -- server is
the reliable harness):

    draft=6  p_min=0.0   102.4 t/s   <- our old A/B config
    draft=6  p_min=0.5   116.6
    draft=10 p_min=0.0   102.7
    draft=10 p_min=0.5   117.9       <- r/LocalLLM ref

The r/LocalLLM +15-20 t/s claim reproduces (+15%), BUT the lever is **p_min 0.5, not draft
depth**: p_min gives +14 t/s at either depth; draft 6->10 buys ~0-1. So our A/B
UNDER-STATED llama decode by ~15% -- the honest strongest-llama baseline is **~117 t/s @2K**
(q27 @2K 169-209 still wins clearly). A depth-matched cross-engine run is still owed before
any headline wall/rate claim. README measurement-debts section + progress-log P5/P6 rows
updated; both numbers now honest.

## 2026-07-06 (later still) -- depth-matched cross-engine: TUNED llama beats q27 at depth

The owed depth-matched run. Both engines, single-stream greedy, the SAME ~75.5K-token prompt
(230KB transcript slice; q27 tokenized 75533, llama 75528 -- matched), decode 256, read
q27-server req_log `tps=` and llama-server /completion `predicted_per_second`:

    q27  (fp8, greedy spec depth-4, LOSSLESS head)  145.6 t/s  (4.57 tok/round, 56 rounds)
    llama tuned   (draft-mtp, n_max=10, p_min=0.5)  190.3 t/s  <- ~31% FASTER than q27
    llama untuned (draft-mtp, n_max=6,  p_min=0.0)  139.5 t/s

**Claim overturned:** the README's "decode rate at depth beats llama" held ONLY vs the untuned
opponent (q27 145.6 > 139.5). Against the STRONGEST llama (p_min 0.5) q27 LOSES decode rate at
depth by ~23%. p_min is worth +36% for llama here (139.5 -> 190.3) vs only +15% @2K -- at depth
the deep-KV verify pass is expensive, so skipping it on low-confidence positions is a big win,
and it grows with ctx. q27 fast-head would add only ~7% (~156), still well short of 190. (q27's
145.6 @75.5K actually beats its own stale 126@61K headline -- q27 is healthy; the tuned opponent
is just faster at depth.) Caveat: n=1 per config, but decode t/s is near-deterministic and the
margin is large + mechanistically explained (a multi-prompt confirmation would firm it).

Consequence: **confidence-gated depth (q27's p_min equivalent) is now the clear highest-value
decode lever** -- the roadmap reopen candidate is empirically the way to close the depth gap.
README State/Why-this-model/measurement-debts/roadmap all corrected to the honest depth-matched
numbers; the "beats llama at depth" claim is retired.

## 2026-07-06 (P12) -- confidence-gated depth SHIPPED (branch p12-confidence-gated-depth)

The lever the depth-match called for. Per-step drafter-margin gate (the top1-top2 margin = the
p_min analog): cap the verify to only the lanes the drafter is confident about, skipping the
expensive deep-KV verify when it is not. `Q27_PMIN=theta` engages it; unset = the canonical
depth-4 round. Greedy output stays **bitwise-identical** -- each verify lane is an independent
grid index, so narrowing the batch changes only round count + verify width, never which token
emits (the same property that made the batch-5 verify match the serial path).

**Phase 0/0b (gate measurement first, per the roadmap).** `--stats`/`--burst-stats` margin bins
on think-heavy + real thunderdome CC agentic traffic. Findings: the winning mechanism is a
per-step p_min early-stop (dominates single-signal m1/m2 gates); agentic decode is FAR more
draftable than math (mean accept-len 6.2 vs 3.9), so depth is traffic-dependent. Offline
throughput bounds predicted a context-adaptive theta (short-ctx neutral, long-ctx wins, higher
theta at longer ctx) -- all confirmed live below. (FINDINGS.md + analyze.py in the session
scratchpad, not the repo.)

**P12 depth-4 gate (the robust win, DEFAULT).** draft width-5 + `k_margin` (top1-top2, a
SEPARATE pass that never touches the canonical k_argmax) -> read 4 margins -> cap at the leading
run >= theta -> launch a pre-captured `verify_graph_w[cap+1][perm]` (per-width verify graphs,
widths 2..5). `finish_round` gained a `max_draft` cap so a narrow verify never commits an
uncomputed lane. Measured decode t/s (`--tokens-file` added to get long prompts past the 128KB
single-arg limit):

    ctx   OFF     theta 0.5       theta 1.0
    2K    167.4   167.4 (neutral) --
    16K   122.8   129.4 (+5.4%)   129.9 (+5.8%)
    60K   90.05   96.31 (+7.0%)   99.76 (+10.8%)

Token-identity gated==ungated verified at 2K/16K/60K; test_kernels ALL PASS; k_margin exact vs CPU.

**P12b depth-5 (opt-in, `Q27_MAXD=5`).** Adds the 6th verify lane: perm goes mod-5 -> mod-6 with a
6th GDN state buffer (S_spare5); depth-4 rounds stay a bitwise-identical subset. Correct
(token-identity, test_kernels PASS, 6-tok rounds fire) but **traffic-dependent**: agentic 3.6K
+2.6% (fires heavily) vs docs 60K depth-5 91.66 < depth-4-gate 99.76 = -8%. Path C always drafts
to gate_maxd, so the 5th MTP pass is pure cost when acceptance is low -> depth-4 is the default,
depth-5 is opt-in. This is the "tok/round overstates deep-maxd" caveat from Phase 0b, now
confirmed on real t/s. Follow-on: ADAPTIVE maxd (key draft depth to recent acceptance) would make
deep drafting a universal win; a long-ctx AGENTIC A/B is the untested regime where depth-5 should
win biggest.

**The bug that cost the session (worth remembering).** Depth-5 first diverged, and
compute-sanitizer memcheck was CLEAN. Root cause: `quantize3` (qx5) was the ONE batched verify
kernel that selects its per-lane output by EXPLICIT pointers (`t==3?n3:n4`) instead of CP3 struct
indexing -- at ntok=6 lane 5 fell through to n4 and OVERWROTE lane 4's quantized activation
(a valid buffer, wrong lane, so memcheck was blind). Bisect trail: theta=100 (width-2) PASS,
cap<=4 (width<=5) PASS -> width-6 only; `Q27_FD=v1` also failed -> not attention -> a shared
kernel -> quantize3. Also sized-for-6: the flash-decode scratch and logits2. **Lesson: any
batched kernel using explicit per-lane pointers instead of the CP3/P3 struct is a lane-count
landmine.**

Commits (branch): e80cd59 P12 gate; 339e9df structs->6 + gemv width-6; f5b4dac mod-6 perm + 6th
GDN buffer; bea6833 6th-lane wiring + quantize3 fix; edc3250 default depth-4 / Q27_MAXD=5 opt-in.

## 2026-07-06 (P14 Task 2) -- fuse draft argmax+margin into one full-vocab pass (k_argmax_top2)

The Task-1 attribution (docs/perf-attribution-p14.md) pinned `k_margin` at **0.545 ms/round @61K
ungated** -- a DEAD scan: the ungated default never reads a margin, but P12 captured a separate
248320-wide single-block `k_margin` per draft into the monolithic `spec_graph`, so every ungated
round paid for 4 top-2 scans nobody reads. This task fuses each draft's argmax and its margin into
ONE full-vocab pass. CANONICAL-EXACT.

**Kernels (`blocks.cu`).** `k_argmax_top2<<<128,256>>>` does k_argmax's per-thread strict-`>`
top-1 (packed via `am_pack`) AND k_margin's top-2 in the same grid-stride loop, then a shared-mem
pairwise merge; `k_top2_finalize<<<1,128>>>` combines the 128 block partials and writes tok +
(top1-top2). Two invariants make it bit-identical: (1) the grid + packed-u64 max lattice are the
SAME as k_argmax, so the token and all tie-breaks match exactly (per-thread lowest index, highest
index across threads); (2) `am_unpack_val` is the exact inverse of am_pack's monotonic map, and the
margin is pure selection (no fp reassociation, only the final subtract), so it equals k_margin's
value order-independently. `k_argmax` and `k_margin` both stay in-tree (verify path / tests).

**Wiring (`engine.cuh`).** `mtp_forward` gains an optional `margin_dst` -- non-null (draft path
only) swaps its trailing `q27k::argmax` for the fused `argmax_margin`, null keeps the plain argmax
for every other caller (deviation from the plan sketch, which put the call in `spec_draft_launches`:
the argmax was already INSIDE mtp_forward at :595, so folding the fused call there avoids a redundant
second argmax and keeps one code path). `spec_draft_launches` drops the 4-5 separate `margin()`
calls and passes `d_draft_margin + k` -- slot mapping (0..4 for drafts 1..5) unchanged. Scratch
`d_am_blk1` (128 u64) + `d_am_blk2` (128 float) alloc'd beside `d_draft_margin`.

**Gates.** test_kernels ALL PASS incl. 3 new fused assertions (token==argmax(), margin==CPU
top1-top2, margin==margin(), all err 0.0e+00 across random n=248320 + all-equal + max@0 + max@last +
duplicated-max). Canonical md5 **4c4120c72056aba2bc2d2561471eafce** EXACT. 16K gated identity
(`Q27_PMIN=0.5`, before/after binaries): emitted text **byte-identical**, **46 rounds** both,
2.80 tok/round both. Build sm_86+sm_120 warning-clean.

**Bench (server, docs prompts, ungated, n=3 median):** 61K **29.87 -> 29.28 ms/round (-0.59**, =
the removed k_margin cost), 116.1 t/s (was 109.9); 2K 21.25 -> 20.93 ms/round (-0.32), 114.7 t/s
(was 112.9). Round counts deterministic across runs (no stale-binary tell). Small, non-negative,
as predicted.

Files: blocks.cu, blocks.cuh, engine.cuh, test_kernels.cu.

## 2026-07-06 (P14 Task 3) -- P12 confidence gate on the sampled spec path

**What.** Ported the P12/P12b confidence gate + gated-round plumbing to the production
sampled spec path (`spec_sample_round`, engine.cuh). Default sampled traffic (T<=0.7) can now
narrow the sampled verify per-round like greedy does. CANONICAL-EXACT (greedy untouched):
canonical md5 **4c4120c72056aba2bc2d2561471eafce** EXACT.

**Kernel (`blocks.cu`).** `k_spec_accept` + wrapper gain `int max_draft`: `stop_lane` inits to
`max_draft` (was 4), the accept walk loops `k < max_draft` (was `< 4`). A width-W sampled
verify walks `max_draft = W-1` drafts; all-accept commits `max_draft` drafts + the bonus lane
(n=max_draft+1). max_draft=4 reproduces the pre-P14 width-5 behavior bit-for-bit.

**Engine (`engine.cuh`).** `spec_verify_launches_sampled` is now per-width: nucleus loop `k <
vw` (was `<5`), `spec_accept(..., vw-1, ...)`. **`k_finish_sampled` needs NO change** -- it
keys on `n = spec[0]` and its `src` select (`n==5?x1e:...:x1a`) covers n in 1..5; with
max_draft=vw-1 and vw<=5, n<=5 always. `build_spec_graphs` captures a new
`verify_sample_graph_w[6][6]` for W=2..5 (mirrors the greedy per-width loop; sampled tail is
always depth-4 so W<=5 regardless of gate_maxd). Draft-graph selection is FIXED: the
sampled+gated path ALWAYS drafts depth-4, so `draft_graph_lo` is now captured whenever
`gate_maxd==5` (widened from maxd_auto-only -- so a fixed `Q27_MAXD=5` also has a depth-4
draft); one extra graph/perm, no new buffers, greedy selection unchanged. `spec_sample_round`
gains the gated branch mirroring `spec_round`: launch depth-4 draft, D2H the 4 margins,
`cap = leading run >= theta` (assert cap<=4), launch `verify_sample_graph_w[max(cap+1,2)]`.
The `samp_first` bootstrap stays ABOVE the gate/ungated branch, so token 0 (retained prefill
logits, `sample_g` kind 0) is identical on both paths.

**LIMITATION (P13 EMA).** sat/yield EMAs are NOT updated from sampled rounds this phase (the
sampled ceiling is fixed at 4); Q27_MAXD=auto adaptive-maxd applies to greedy only.

**Correctness gates -- ALL PASS.** `test_kernels --sampling-only` incl. new max_draft cases:
cap honored (n<=md+1 for md 1..4), lane-0 accept==p_served under each cap, per-lane conditional
accept==p_served at md=4 (lanes 0..3), all-accept stop_lane==max_draft (viol=0). Full
`test_kernels` ALL PASS. `tools/sampling_gate.sh` ALL PASS (greedy canonical, seeded identity,
seed-varies, sampled!=greedy, accept-vs-temp). NEW live gates: sampled+gated seeded
reproducibility (seed 42 byte-identical x2, seed 43 differs); first-token identity
gated(Q27_PMIN=1.0) vs ungated same seed 42 (both first tok=11751); gated T=0.7 = 3.00
tok/round (no acceptance collapse). Build sm_86+sm_120 warning-clean.

**PERF -- gate NOT met at 61K (honest, calibration finding, not a bug).** Sampled T=0.7 docs
prompt, server n=3 medians (`[req]` dec/rounds): 61K best theta = 0.5 at **+0.0%** (95.1 ->
95.1 t/s), theta=1.0 **-4.3%**; 2K theta=0.5 **+8.0%** (146.6 vs 135.7, tok/round IDENTICAL --
pure per-round savings), theta=1.0 -11.3%. The +2% @61K target (calibration-updated from the
greedy analog) is missed. Mechanism: the drafter-margin gate narrows verify; greedy would
argmax-reject a low-margin draft anyway (narrowing free), but sampled rejection can still
accept it, so at 61K the narrowing cuts accept-able lanes (tok/round 2.887->2.446) and the
extra rounds offset the cheaper round -> net 0. GREEDY cross-check on THIS binary confirms the
shared machinery is healthy: 61K greedy +6.6% (theta 1.0) / +2.9% (theta 0.5). So the sampled
result is a genuine rejection-acceptance effect. Full tables in docs/perf-attribution-p14.md
(Task 3 section). Gate stays OFF by default (Q27_PMIN unset) -> zero risk to production
sampled traffic. The sampled win, if any, is on the DRAFT side (Task 4 early-exit); Task 3 is
its substrate. `verify_sample_graph_w` IS exercised now (the gated sampled branch drives it
whenever Q27_PMIN>0 under sampling) and is proven correct by the reproducibility/first-token
gates; it just does not yet pay for itself at 61K on docs traffic.

Files: blocks.cu, blocks.cuh, engine.cuh, test_kernels.cu, docs/perf-attribution-p14.md.

## 2026-07-06 (P14 Task 4) -- draft early-exit: margin-gated per-step draft graphs

**What.** llama's `p_min` stops DRAFTING at the first low-confidence draft
(common/speculative.cpp:332); q27's P12 gate only narrowed VERIFY while the gated draft
always ran to gate_maxd. This task splits the draft into per-step CUDA graphs
(`draft_step_graph[5][6]`, one per draft step per perm) and makes both gated rounds
(greedy `spec_round`, sampled `spec_sample_round`) launch steps one at a time, D2H the
4-byte margin, and stop at the first sub-theta margin. `Q27_DEXIT=0` restores the
monolithic gated draft (A/B lever); default ON when `Q27_PMIN>0`; the ungated/constrained
paths never touch any of it. CANONICAL-EXACT: **4c4120c72056aba2bc2d2561471eafce** at the
refactor checkpoint AND the final binary.

**Refactor.** `spec_draft_launches` -> `for (k < dmax) spec_draft_step_launches(k)`; step 0
= prep_round + draft 1, step k>0 = h_next{k+1} D2D + mtp_forward k+1. The concatenated
kernel sequence is byte-identical to the old monolithic body (the D2D that used to trail
step k now leads step k+1 -- same stream order), so every existing graph capture
(spec/draft/draft_lo/sample) is untouched -- proven by canonical EXACT at the
refactor-only checkpoint before any behavior change landed.

**The staleness-proof re-check EARNED ITS STEP (paid-for lesson).** The plan's Step-1 proof
claimed early-exit is safe because "verify commits n <= cap+1". Re-checking each cited
invariant against HEAD found the fourth one FALSE at one edge: the no-width-1-gemv floor
(`W = max(2, cap+1)`, engine.cuh) means a cap==0 round runs a width-2 verify whose finish
walks max_draft=W-1=1 draft, so **n can reach 2 while only draft 1 ran**. Counterexample
trace: cap==0 (draft-1 margin < theta) but the verifier ACCEPTS d1 (margin gates the
drafter's confidence, not the verifier's verdict -- common on low-margin docs traffic) =>
n=2 => base P'=P+2, but naive early-exit never wrote MTP KV row P+2 (monolithic did, via
draft 2) => next round's draft 1 at P+3 attends a stale row P+2 => round-count divergence
(greedy) / byte divergence (sampled). **Fix: top up draft-step launches to min(W, md_used)
before the verify** -- fires only at cap==0, one extra step whose inputs (d_draft, x1 after
step 0) are final, so it writes exactly what monolithic step 1 writes. With the top-up,
drafts_launched >= n everywhere except the pre-existing full-accept bonus row (unwritten
identically on both paths). The other three invariants held as written (kv_store before
attn_decode in attn_block; pos_m{k}=P+k in k_prep_round; drafts are the sole MTP-KV writers
during decode). Plan doc amended in this commit to record the corrected proof + loop.

**Gates -- ALL PASS.** (1) Refactor checkpoint canonical EXACT. (2) Identity matrix: theta
{0.5,1.0} x ctx {2K,61K} x {greedy, sampled T=0.7 seed 42}, 3 runs/cell: DEXIT=1 vs 0
emitted bytes IDENTICAL and rounds IDENTICAL in all 8 cells; plus Q27_MAXD=auto greedy
smoke byte-identical (46 rounds both; EMA sees identical n/md_used by construction, caps
identical). (3) Final canonical EXACT. (4) test_kernels ALL PASS (no kernel changes --
graph/host restructuring only).

**Perf (server, docs prompts, n=3 medians, DEXIT=1 vs 0 same binary).** 61K greedy
theta=1.0 **+3.2%** (122.1 vs 118.3, -0.77 ms/round) -- plan gate >=+3% MET; theta=0.5
+1.4%. 61K SAMPLED **+5.4%** (theta 1.0) / **+3.0%** (theta 0.5): tok/round IDENTICAL by
construction (2.133/2.526), so unlike Task 3's verify-narrowing (a wash at depth from
acceptance loss) this is pure per-round savings -- sampled gated theta=0.5 is now **+3.6%
over ungated** on this binary (Task 3's +0.0% resolved). 2K +3.9..+5.9% (docs 2K is
low-acceptance at these thetas -- lots to skip; NOT the neutral case). Worst-case overhead
probe `Q27_PMIN=0.01` (nothing ever skipped, md_used syncs vs 1): -0.4/-0.5%, i.e. the +3
extra 4-byte D2H syncs cost ~0.1 ms/round = noise. Sync delta/round: monolithic 1 margin
sync; early-exit min(cap+1, md_used). Full tables + gated-vs-ungated matrix + session-drift
footnote in docs/perf-attribution-p14.md (Task 4 section). Recommendation recorded there:
gate ON both paths, theta=0.5 cross-path default (sampled theta=1.0 still nets -2.1% vs
ungated at 61K); Q27_DEXIT default-ON (positive or neutral in every measured cell).

Files: engine.cuh (spec_draft_step_launches + capture + both gated branches),
docs/plans/2026-07-06-p14-perf-levers.md (corrected Step-1 proof + Step-4 loop),
docs/perf-attribution-p14.md.

## 2026-07-07 (P14 Task 5) -- fd2 lane-innermost grid order: cross-lane KV L2 reuse (+2.7%, MARGINAL, KEPT)

**What.** Task-1 measured R~=4.25 (verify fd2 per-instance time linear in verify width on a
BW-bound kernel => each verify lane re-streams the full KV slice from DRAM, zero cross-lane
L2 reuse). Cause: the verify grid `dim3(n_kv_heads, FD2_NS, ntok)` puts the token lane on
`blockIdx.z`, the SLOWEST-varying axis, so all of lane 0's blocks schedule before lane 1's.
But the KV read address depends on `(pos, kv_head)` only -- NOT the lane -- so the `vw`
same-`(head,split)` blocks read byte-identical KV. **Fix: make the lane the FASTEST axis** so
those blocks co-schedule and share the ~1MB KV chunk in L2. PURE INDEX REMAP, two lines in
spec3.cu:

```
- dim3 g1(n_kv_heads, FD2_NS, ntok);   const int kvh=blockIdx.x, sp=blockIdx.y, t=blockIdx.z;
+ dim3 g1(ntok, FD2_NS, n_kv_heads);   const int t=blockIdx.x, sp=blockIdx.y, kvh=blockIdx.z;
```

Per-block work, per-lane fp accumulation order, scratch-cell addressing per
`(head,split,lane)`, and the combine kernel are byte-for-byte unaffected -- only the block
enumeration ORDER differs. v1 fallback (`k_attn_fd`, `Q27_FD=v1`) untouched. Task 5b SKIP
(draft-attn 0.51 ms/round < 1.5 floor, Task 1 Step 6). CANONICAL-EXACT.

**Gates -- ALL PASS.** (1) Full make sm_86+sm_120: no spec3.cu warnings. (2) test_kernels ALL
PASS (0 FAIL): fd2-vs-v1 tolerance, run-to-run bitwise, default-dispatch bitwise all
unchanged over seq {1,47,1024,16384,61440} x ntok {1,5} x {fp8,fp16}. **Bitwise-vs-pre-change
proven on the FULL matrix**: a temporary FNV-1a fingerprint over every fd2 output byte of
every matrix cell is IDENTICAL pre vs post (`5f0e1d98593d2283`; pre binary built by stashing
ONLY the spec3 remap and keeping the fingerprint helper; helper reverted before commit so the
landed diff is spec3.cu-only). Plus the named substitute: 61K greedy `generated:` text
byte-identical pre vs post (1308 B, 3/3). (3) Canonical **4c4120c72056aba2bc2d2561471eafce**
EXACT on pre, post, and final binary.

**Bench (server, docs prompts, greedy, n=3 medians, spread <0.1%; pre re-measured on the
saved pre-change binary).** tok/round and round-counts IDENTICAL pre vs post in every cell
(bitwise fd2), so the whole delta is per-round attention cost. **61K ungated 116.1 -> 119.3
t/s = +2.7%** (29.27 -> 28.50 ms/round). 61K gated theta=0.5+dexit 121.8 -> 124.2 = +2.0%.
2K ungated +0.0% (115.5=115.5), 2K gated -0.1% -- no short-ctx regression (empty-split
early-return unaffected by axis order). **Decision (plan rule on 61K ungated): +2.7% is in
the [+1.5%, +3%) band -> KEEP, flagged MARGINAL** (above the +1.5% revert floor, below the
+3% unqualified-keep bar).

**Mechanism (nsys, direct).** Step-4-style POST capture vs the Task-1 pre-change fd2 row
(same node-traced methodology): verify fd2 per-instance Med **542.1 -> 487.3 us (-10.1%)**,
Max 549.8 -> 506.1 (-8.0%), while the draft z=1 Min is flat (125.2 -> 128.2 us; single-lane
draft has no cross-lane question). Verify time drops ~10% toward the draft floor = the exact
L2-reuse signature. 16 verify/round x -54.8 us = -0.88 ms/round predicted, matches the
observed -0.77 ms/round. **Reuse is PARTIAL** -- the 5090 L2 absorbs a minority of the
~63 MB/layer fp8 KV per co-scheduled wave, not all of it -- which is why the win is real but
marginal, NOT the full R~4.25 headroom. The residual headroom is the Task 6 (lane-pair
fusion) target, which therefore is NOT made redundant by Task 5; still DEFERRED pending
Gabe's explicit go (expensive kernel rewrite).

Files: src/spec3.cu (2-line index remap), docs/perf-attribution-p14.md (Task 5 section +
go/no-go Task 6 row updated).

## 2026-07-07 (P14 Task 8) -- capstone: docs sync + graph-zoo inventory

Wrap-up for the P14 perf-levers bundle (branch `p14-perf-levers`, Tasks 0-5 + 7 landed;
5b measured-SKIP, 6 deferred pending go). **The bundle:** fused draft argmax+margin
(Task 2, `k_argmax_top2` -- kills the dead ungated `k_margin` scan, -0.545 ms/round); the
P12 confidence gate ported to the production SAMPLED spec path (Task 3, per-width sampled
verify graphs + capped accept walk); draft early-exit (Task 4, `Q27_DEXIT` default-ON when
`Q27_PMIN` set -- margin-gated per-step draft graphs with the `min(W,md_used)` width-floor
top-up, llama's p_min draft-stop parity); and the fd2 lane-innermost grid order (Task 5,
partial cross-lane KV L2 reuse). **Headline same-config @61K docs greedy: ungated 109.9 ->
119.3 t/s across the bundle (Task 2 fusion + Task 5 L2); production gated config
(`Q27_PMIN=0.5` + `Q27_DEXIT`) 124-125 t/s. Sampled @61K gated+dexit +3.6% over ungated.**
Greedy stays bitwise throughout (canonical **4c4120c72056aba2bc2d2561471eafce** unchanged
end to end). **Task-4 proof-catch (the paid-for step):** the Step-1 re-check of the
staleness proof found the width-2 floor lets a cap==0 verify commit n=2 while only draft 1
ran -- a stale-KV-row divergence -- caught before any code was written; the `min(W,md_used)`
top-up fixes it. Task 5 is MARGINAL-KEPT (+2.7%, in the [+1.5%,+3%) band); it captured only
~10% of the R~4.25 cross-lane headroom, so the residual is the deferred Task 6 (lane-pair
fusion) target. See docs/perf-attribution-p14.md (Tasks 1-5 attribution + go/no-go matrix)
and docs/maxd6-decision.md (gate_maxd=6: NO-GO fixed default, GO-IF adaptive gated on one
unmeasured agentic depth-5 A/B; Task 6 recommended first).

**This commit (docs-only + comment-only):** (1) README State/serving/roadmap sync -- new
P14 State bullet, four progress-log rows, a serving-section gated-config block, and roadmap
open-lever entries (maxd6 GO-IF measurement + Task 6 lane-pair fusion), all framed per the
red-team precedent (same-prompt A/B t/s deltas only, no cross-prompt/score claims, Task 5
flagged marginal). (2) A GRAPH-ZOO comment block above the perm-indexed graph members in
engine.cuh inventorying all nine graph sets (spec_graph, spec_sample_graph, verify_graph_w,
verify_sample_graph_w, draft_step_graph, draft_graph, draft_graph_lo, verify_graph, plus the
two non-spec singletons graph_exec/sample_graph) and which decode path launches each --
comment-only, zero codegen change (canonical EXACT confirms). **Removable-candidate noted,
NOT removed:** with `Q27_DEXIT` default-ON the gated early-exit path drives
`draft_step_graph`, so monolithic `draft_graph` / `draft_graph_lo` are redundant FOR THE
GATED PATH; both stay live for the P11 constrained-tool path and the `Q27_DEXIT=0` A/B
baseline, so neither is removable. `draft_graph_lo` is the closest to dead (unique callers:
constrained-tool-under-auto + the DEXIT=0 auto/sampled fallback) -- a future
graph-pruning pass, not this bundle. `verify_graph` (monolithic width-5) is now used only by
the constrained-tool path.

**Final gates on the branch tip (engine.cuh touched -> rebuild REQUIRED, not optional):**
full `make` clean (sm_86+sm_120, only pre-existing tokenizer.cpp warnings); `test_kernels`
ALL PASS; canonical **4c4120c72056aba2bc2d2561471eafce** EXACT. Branch left UNPUSHED +
UNMERGED for Gabe (public repo; merge is his call).

Files: README.md, src/engine.cuh (comment-only graph-zoo block), docs/BUILDLOG.md.

## 2026-07-07 (P14 Task 6) -- NEGATIVE: fd2 lane-pair fusion (k_attn_fd3) killed on the perf gate

**What:** implemented the deferred Task-6 lane-pair fusion (Gabe's explicit go on the Task-5
marginal result): `k_attn_fd3` pairs verify lanes per block -- grid
`dim3(ceil(ntok/2), FD2_NS, n_kv_heads)` (Task-5 lane-innermost order kept, pair index
fastest), per-lane online-softmax state in per-lane register slices (`m[2][6], l[2][6],
acc[2][6][8]`, all compile-time-indexed per the quantize3 landmine rule), ONE KV row read
serving both lanes when the split bases align (always for consecutive verify positions except
~1/128 chunk-boundary straddles), odd-ntok last block single-lane via a `has1` guard, scratch
layout + combine untouched, `Q27_FD=fd2` fallback to the unpaired kernel (v1 fallback
unchanged). Design + full post-mortem: **docs/attn-fd3-design.md** (kept per the kill
protocol; kernel reverted).

**Every correctness gate passed; the BENCH killed it.** Stage-B occupancy skeleton
(`-Xptxas -v`, CUDA 13.2, both archs): exactly **168 regs, 0 spills, 18,816B smem/block ->
3 blocks / 12 warps per SM** on sm_86 AND sm_120 (gate floor: <=168 regs, >=3 blocks -- PASS
at the boundary; fd2 reference: 119/122 regs -> 4 blocks / 16 warps). test_kernels **384
checks ALL PASS, 0 FAIL**: all 132 fd2 gates unchanged; 160 new fd3 gates (fd3-vs-fd2
BITWISE, run-to-run determinism, default-dispatch==fd3, Q27_FD=fd2 fallback; seq
{1,47,1024,16384,61440} x ntok {1,2,3,4,5} x {fp8,fp16}) ALL **err 0.000e+00** -- the
bitwise-vs-fd2 contract held exactly. Canonical **4c4120c72056aba2bc2d2561471eafce** EXACT
on the fd3 binary.

**Bench (kill):** server rig, docs prompts, greedy, 1 warmup + n=3 medians (spread <0.1%),
same-binary same-session A/B via `Q27_FD=fd2` (which reproduced the recorded Task-5 POST
baselines within 0.3% in every cell -- no session drift; tok/round + rounds IDENTICAL
fd2-vs-fd3 in every cell, live bitwise confirmation):
61K ungated **118.9 -> 114.2 t/s (-4.0%)** (28.58 -> 29.75 ms/round, 113 rounds);
61K gated0.5+dexit **124.3 -> 118.1 (-5.0%)**; 2K ungated 115.4 -> 115.1 (-0.3%); 2K
gated0.5 129.5 -> 129.4 (-0.1%). Kill criterion: 61K ungated gain < +5% -> REVERT.

**Why it lost (nsys per-instance, fd3 binary @61K vs Task-5 fd2 rows):** draft (grid x=1,
single lane -- fd2's exact work, only the resource shape differs) **128.2 -> 218.9 us
(+71%)**; verify (x=3) 487.3 -> 548.2 us (+13%) despite ~halved pair KV traffic. Predicted
round delta +1.34 ms/round vs measured +1.17 -- consistent. **The kernel is
latency-hiding-bound before it is DRAM-BW-bound: dropping 16 -> 12 warps/SM costs more than
halving KV bytes saves.** The honest per-lane register state (120 floats) pins regs >= ~168
-> 3 blocks/SM; breaking even needs fd2's 16 warps (~128 regs), unreachable without changing
per-lane fp order (= tolerance-gate + canonical-re-derive class, NOT this task's contract).
The >=12-warp Stage-B floor was too permissive -- for future occupancy gates on this kernel
family, the bar is "no worse than the incumbent's warps/SM", not an absolute 12.

**Follow-on options recorded in the design doc:** (a) accept fp-order change and batch a
narrower-state pair kernel into a tolerance-gate + canonical-re-derive cycle (orchestrator's
call, attn-fd2-design.md:42-56 policy); (b) smem-tile KV sharing / __ldcs -- measure first.
The R~4.25 headroom stands unclaimed; Task 5's +2.7% remains the shipped fd2 state.

**Revert hygiene:** src/spec3.cu, src/spec3.cuh, src/test_kernels.cu restored to the Task-8
tip (7c2ff95) BEFORE this commit; full `make` rebuilt; test_kernels ALL PASS and canonical
EXACT re-verified on the reverted tree. This commit lands docs only.

Files: docs/attn-fd3-design.md (new, design + post-mortem), docs/BUILDLOG.md.

## 2026-07-07 (maxd6 GO-IF measurement) -- VERDICT: NO-GO on auto-ceiling-6; depth stays 4/5-auto

The docs/maxd6-decision.md GO-IF gates, measured on real agentic serving traffic. Telemetry
commit 42ccf6d (host-side gated-round histograms: margin-run depth `gch` cap 0..gate_maxd,
accepted length `gnh` n 1..gate_maxd+1, cumulative per engine, appended to `[req]` after
`end=`; canonical 4c4120c7 EXACT ungated AND gated depth-5; test_kernels ALL PASS).

**Rig.** Server `Q27_PMIN=0.5 Q27_MAXD=5` (dexit=1, greedy), one full thunderdome T8 trial
via claude-code-q27-haight (67 reqs, ctx to 81K, score 0.796 -- high basin, hidden 0.906;
logs results/q27-maxd6-server.log, runs/2026-07-07T15-50-33). A/B = identical-request
26K-deep replay (P13 methodology; live-agent A/B diverges), fresh server per leg, 1 prefill
+ 3 replays, medians (spread <=0.3%); three payloads spanning the saturation axis
(logs results/q27-maxd6-ab-{d5,d4,d5mid}.log).

**X + Y1 (live T8, 5336 gated rounds): BOTH PASS.**
gch=[148,189,259,239,268,4233] gnh=[193,427,444,384,408,3480].
X: cap>=5 fired on **79.3%** of gated rounds (gate >=30%). Y1: depth-5 saturation (n=6)
**65.2%** (gate >=0.50), grew 45%->65% as ctx deepened; p(5th lane accepted | fired)
**82.2%**; mean **5.03 tok/round** -- real CC traffic is far more repro-flavored than fresh
codegen (tool-call bodies echo file content).

**Y2 (d5-vs-d4 replay A/B): FAIL.**
| payload | d4 t/s (tok/rnd) | d5 t/s (tok/rnd) | d5 net | d5 fired | p(5th|fired) |
|---|---|---|---|---|---|
| code-repro | 205.2 (4.93) | 211.2 (5.88) | **+2.9%** | 97.6% | 95.2% |
| T8-style codegen | 169.8 (3.70) | 160.6 (3.98) | **-5.4%** | 56.3% | 56.2% |
| fresh unit-test gen | 152.6 (3.34) | 146.7 (3.53) | **-3.9%** | 51.9% | 48.6% |
Interpolated at the live-T8 operating point (79.3% fired / 82.2% yield): **-0.8%..+0.1% --
breakeven at best.** Fixed depth-5 wins ONLY in near-verbatim regimes (>~90% fired).

**The brief's fired-round model is REFUTED.** Predicted ~2.0 ms marginal on fired rounds
only ("cost correlated with payoff", 0.46 tok/ms -> strongly positive); measured ms/round
d4->d5 grew +2.1..+3.8 across ALL rounds (repro 23.99->27.80, mid 21.73->24.72, fresh
21.90->24.05) -- ~1.5-2x the model. Mechanism: **theta gates on drafter confidence, not
verifier acceptance** -- at mid, 44% of fired rounds wasted the deep draft+verify anyway
(p|fired 56%). The correlation the model assumed only tightens in repro-like traffic.

**Verdict: NO-GO.** Depth-6 is the same structure one lane deeper on a strictly smaller
fired fraction, against a P12b-class 6->7 widening + quantize3-landmine audit + 157 MB.
With depth-5 at breakeven on live-matched agentic traffic, the ceiling stays at the
d4/d5-auto optimum. Do-not-retry without new facts (cheaper verify lane or a gate that
predicts ACCEPTANCE, not confidence -- e.g. margin-calibrated accept prob).

**Bonus validation: P13's HI=0.5 promote threshold is empirically well-placed.** It
excludes exactly the regimes where d5 measured negative (mid 45% d4-sat -> stays 4,
avoiding -5.4%; fresh 37% -> stays 4, avoiding -3.9%) and admits the winners (repro 96% ->
promotes, +2.9%; live T8 ~73% d4-sat -> promotes, ~breakeven).

Files: docs/maxd6-decision.md (verdict appended), src/engine.cuh + src/server.cu (42ccf6d).
Next per queue: prefill-attn O(N^2) (54% of prefill @128K).

## 2026-07-07 (prefill-attn Phase 0) -- ncu attribution: latency/occupancy-starved, NOT bandwidth/tensor bound

Queue-top item started (branch prefill-attn; plan docs/plans/2026-07-07-prefill-attn.md).
Phase 0 = ncu on `k_attn_prefill_mma` at 128K to decide the lever ranking before any rewrite.

Rig: existing HEAD binary (canonical 4c4120c7 verified, no rebuild), 5090 free, fixture
scratchpad/synthtoks.bin (140k random int32 < VOCAB; prefill timing value-independent).
Baseline no-profiler 128K prefill 75.45s / 1737 t/s (P14 band). ncu 2026.1 sudo -n,
`-k k_attn_prefill_mma --launch-skip 1900 --launch-count 3` -> 3 launches at chunk ~118,
base ~121K (PF_T=1024 -> 16 attn x 128 chunks = 2048 total launches; first attempt's
--launch-skip 6000 overshot and profiled nothing). Report scratchpad/pf_attn_128k.ncu-rep.

Deepest launch (44.8ms, grid (4,64,8)=2048 blocks, nsplit=8): **DRAM 1.98%** (35 GB/s of
1790), **L2 hit 95.6%** (KV L2-resident at depth, NOT re-streamed -- P4 split already killed
the DRAM problem), **tensor 33%** ("should not be a bottleneck"), **IPC 0.42**, issue slots
9.9% busy, 14.2 stall-cyc/inst. **Achieved occupancy 12.5%** (6 warps/SM, 1 CTA). Occupancy
DUAL-limited: **Block Limit Registers = 1 (248 regs/thread) AND Block Limit Shared Mem = 1
(84.48 KB)**. Stalls spread: long_scoreboard 30% / math_pipe_throttle 28% / barrier 15% /
wait 14% = occupancy-starvation signature (6 warps hide nothing).

VERDICT: PROCEED. Two corrections to the plan: (1) FLOP-derived ~39% tensor is really ~33%;
(2) shrinking smem ALONE won't raise occupancy -- registers co-bind (the o[32][4] O
accumulator is 128 regs/thread), so any 2-CTA play needs a register cut too. Phase 1
(cp.async) confirmed as the first move: attacks the largest stall (long_scoreboard 30%) by
prefetching the next tile's K/V, raising IPC within the existing 6 warps, no occupancy
needed, bitwise-safe. Full data: docs/perf-attribution-prefill-attn.md.

Files: docs/plans/2026-07-07-prefill-attn.md (plan + Task-1 verdict + register-cut notes),
docs/perf-attribution-prefill-attn.md (new). Docs only; no code changed.

## 2026-07-07 (prefill-attn Phase 1) -- cp.async K/V prefetch: MEASURED NEUTRAL (~0%), kept as Phase-2 scaffolding

Implemented cp.async double-buffered prefetch in k_attn_prefill_mma (fp8 path only;
`Q27_PF_CPASYNC`, default on). Raw fp8 K/V of the next PP-tile is cp.async-prefetched into a
+16 KB smem buffer (total 100,864 B <= the 101,376 B sm_120 optin cap) while the current
tile's MMAs run; convert-on-consume feeds kv2h the identical bytes -> bitwise-identical.
fp16 path lacks smem room, keeps the blocking staging. New PTX helpers cpasync16/commit/
wait_all; 27 cp.async in the generated PTX (confirmed engaged -- cuobjdump/ncu were just
missing from PATH, hence earlier empty greps).

GATES: full make clean (sm_86+sm_120). Canonical 4c4120c7 EXACT. Prefill bitwise A/B
(greedy -n 8, 8K prompt, nsplit=2): CPASYNC=1 == CPASYNC=0 == 36b83fd8 -- BITWISE PASS.

PERF: 128K prefill wall, same binary, flag-only A/B (so the delta is the isolated attention
kernel): CPASYNC=1 76.30s vs CPASYNC=0 76.40s = **+0.2% on attention. NEUTRAL.** Root cause
(consistent with Phase 0): (1) at depth the K/V loads are already 95.6% L2-hit (~200cyc), not
the DRAM cost cp.async shines against; (2) fp8 forces a separate smem->smem convert pass
(fp8->half for the fp16 MMA) that eats the saving; (3) the kernel is occupancy-starved (6
warps) so there's little independent work to overlap the async load behind. cp.async is an
occupancy-INDEPENDENT lever; this kernel's bottleneck is occupancy.

DECISION: keep the scaffolding (raw-fp8 prefetch pipeline is exactly what fp8-MMA reuses) and
proceed to Phase 2 (fp8-MMA) -- consume raw fp8 directly in the MMA: removes the convert,
halves K/V smem, doubles QK^T throughput (attacks the measured 28% math_pipe_throttle).
Tolerance-gated (breaks greedy bitwise). Gabe approved proceeding 2026-07-07.

Files: src/prefill.cu (cp.async helpers + prefetch pipeline + launch),
docs/perf-attribution-prefill-attn.md (Phase 1 result appended), docs/plans/2026-07-07-prefill-attn.md (Task 2 verdict).

## 2026-07-07 (prefill-attn Phase 1 CORRECTION) -- cp.async is +5.4%, NOT neutral -- prior test used fp16 KV where cp.async is dead code

**The Phase-1 "neutral" verdict above is WRONG -- methodology error.** `kv_fp8` defaults to
false (fp16 KV); fp8 is opt-in via `Q27_KV=fp8` (engine.cuh:284/306). cp.async lives only on
the fp8 path (`CPA = sizeof(CT)==1`). Every Phase-0/1 run used fp16 KV -- and `--kvstats`
even *forbids* fp8 (engine.cu:458) -- so `CPASYNC=1` and `=0` both ran the identical fp16
blocking path. "Bitwise + same speed" meant cp.async never executed, not that it's neutral.

**Redone on the real fp8 path** (`Q27_KV=fp8`, `--pf 131072 --ctx 133120`, `Q27_PF_NOSERIAL=1`
-- the kvstats fp8-ban forced this harness):
| config | 128K prefill wall |
|---|---|
| CPASYNC=0 (blocking, f16 mma) | **72.10 s** (1818 t/s) |
| CPASYNC=1 (cp.async, f16 mma) | **68.20 s** (1922 t/s) |
| CPASYNC=1 + FP8MMA=1 (fp8 QK^T) | 68.16 s (1923 t/s) |

**cp.async = 72.10 -> 68.20 s = +5.4% on the fp8 prefill wall (~+10% on the attention
kernel).** The +5.4% is common-mode-clean (the d_gen OOB below was identical across configs).
So cp.async is a REAL WIN on the production path -- Phase 1 is a keep, not a wash. (The fp16
path stays on blocking staging, no smem room; production is fp8 so that's fine.)

fp8 QK^T (Phase 2 2a) = 68.20 -> 68.16 = neutral SO FAR, as predicted: 2a still runs the now-
redundant K convert that offsets the fewer-MMA saving. Removing that convert when fp8q (2b) is
where the fp8 throughput shows up. Greedy output identical f16-vs-fp8 QK^T (fp8 perturbation
below the argmax-flip threshold on this prompt); logit-level tolerance check pending.

Lesson: ALWAYS set `Q27_KV=fp8` when benching the prefill-attn path; `--kvstats` is fp16-only
so use `--pf N --ctx >N Q27_PF_NOSERIAL=1` for fp8 prefill timing.

Files: docs (correction). Numbers on the pre-d_gen-fix binary; delta is common-mode so valid.

## 2026-07-07 (fix) -- d_gen long-context OOB write (CUDA-review #1 / the top single-user bug)

`d_gen` was a fixed 65,536-int buffer (`MAX_GEN_TRACK`); batched prefill's final
`step_with(prompt[NP-1])` runs `k_advance` writing `d_gen[NP-1]` (blocks.cu:102) with no
capacity check. Any prompt > 65,536 tokens at `--ctx > 65,536` therefore wrote OOB -- and the
prefill-attn benchmarking (131072-token prefills) was triggering it on every deep run. The
write is post-forward (advance sits at the end of the step_with graph, after logits), so it's
a small scattered scribble that left the timing/logit measurements valid, but it is genuine
UB and the #1 documented correctness bug.

Fix: allocate `d_gen` from `max_ctx` instead of the fixed 65,536 (engine.cuh:336). NP is
already bounded `<= max_ctx` by the generate() guard, so `d_gen[NP-1]` is now always in
bounds -- exact, no per-write branch. Cost: negligible (d_gen goes 256 KB -> max_ctx*4).

GATES: full make clean. Canonical 4c4120c7 EXACT (d_gen is a tracking buffer, not in the
decode compute path -- output unchanged). compute-sanitizer memcheck on a 66,000-token
prefill (>65,536): **ERROR SUMMARY: 0 errors** (was an OOB global write pre-fix).

Files: src/engine.cuh (d_gen alloc). Addresses docs/cuda-review-2026-07-07.md #1,
docs/SECURITY-MODEL.md carve-out #4.

## 2026-07-07 (review bug-fix pass) -- all remaining confirmed review bugs fixed

Cleared the confirmed bugs from both review docs before resuming perf work (d_gen was
already done). Two commits, each full-make + canonical 4c4120c7 (all changes are
near-zero-head / sampled / discarded-value / never-triggered / prompt-content paths, so
greedy is untouched):

fd0f504 (batch): CUDA #6 L2-eps (prefill batched path max(sum,eps)->eps^2, ggml semantics);
CUDA #2 Philox u==1 clamp (top ~128 x0 rounded to 2^32 -> infinite Gumbel; sampled only);
CUDA #4 split-prefill only when t0==0 (absolute-vs-relative row corruption, latent); CUDA #5
dp4a staging tt<nt guard (OOB read past T, discarded values, latent); Security #1 null-content
guard (const operator[] SIGABRT on content-less message, OpenAI endpoints); Security #2 refuse
NP<1 (empty prompt decoded from stale recurrent state); Model move-assign UB (munmap instead
of ~Model()+reuse); DeviceModel copy deleted (raw-CUDA-ptr double-free); --ctx floor 32.

4fa9d24: CUDA #3 top-p (k_nucleus_d bisects the logit threshold over a 40-logit window, 16
steps, vs the old prob-cutoff [0,1] 12-step that degenerated to full-vocab on diffuse
distributions; sampled only); Security #7 ChatML injection (strip <|im_start|>/<|im_end|>
from untrusted content+roles in chatml_prompt; no-op for marker-free content so the prefix
cache is unchanged).

Out of scope (per SECURITY-MODEL, single-operator engine on own artifacts): the
multi-tenant network findings and the untrusted-tokenizer bugs (#9-11: zero-length special
loop, unchecked fread, embedding OOB). Left as documented, not fixed. Ledgers
(docs/cuda-review-2026-07-07.md, docs/SECURITY-MODEL.md) updated to FIXED status.

Sampled-path fixes (Philox, top-p) verified by canonical-no-regression only; a dedicated
sampled A/B would firm the top-p diffuse-case behavior. ChatML strip is a targeted
mitigation; the fuller fix (token-wise encode with specials off for content) is deferred as
it touches the prefix-cache path.

## 2026-07-07 (4th session) -- depth-match multi-prompt confirm: the -31% did NOT generalize

The owed multi-prompt confirmation of the 2026-07-06 n=1 depth-match (q27 145.6 vs tuned
llama 190.3, "~31% faster at depth"). Same methodology, 4 payload flavors at ~75.4K matched
tokens (q27 count_tokens vs llama /tokenize within ~15 tok), single-stream greedy, decode
256, raw completion endpoints (q27 /v1/completions "prompt" = raw encode, no template;
llama /completion), fresh server per leg, 1 cold prefill + 3 identical replays (replay
spread <=0.3%). q27 legs = HEAD fa028d2 production serving config (fp8 KV, --fast-head),
gated Q27_PMIN=0.5 (Q27_DEXIT default-on) and ungated; llama = Q5_K_M, draft-mtp
n_max=10 p_min=0.5, q8_0 KV. Logs thunderdome/results/{q27-depthmatch-*,llama-depthmatch-tuned}.log.

    ~75.4K ctx, decode 256      q27 ungated  q27 gated(0.5)  llama tuned   winner
    P1 transcript (CC A/B log)     118.2        123.3 (3.56)   111.4 (69%)  q27 +10.7%
    P2 repro (docs + self-copy)    141.7        153.0 (4.41)   154.4 (87%)  tie  -0.9%
    P3 code (CUDA src concat)       83.8         93.1 (2.37)    99.5 (76%)  llama +6.9%
    P4 echo (3KB block repeated)      --        158.0 (4.92)  229.9 (100%)  llama +45%
    (parens: q27 tok/round | llama draft-acceptance)  P1-P3 geomean: 120.6 vs 119.6

**Verdict: the 07-06 -31% was prompt-specific AND pre-P12.** On the current build the
mixed-flavor picture is PARITY (geomean +0.9% q27); q27 wins transcript, ties repro, loses
code ~7%. llama's remaining edge is structural and lives in the near-verbatim tail: at
P4's 100% acceptance its depth-10 drafts (11-tok rounds) beat q27's depth-4 ceiling
(4.92 tok/round = saturated) by +45% -- exactly the >90%-fired regime the maxd6 NO-GO
measurement identified, and raising the ceiling is already priced (needs an
acceptance-predicting gate, not theta). Secondary results: (a) first multi-prompt
depth confirmation of the P12 gate: +4.3%/+8.0%/+11.1% on P1/P2/P3 -- biggest where
acceptance is LOWEST; (b) acceptance is strongly flavor-dependent at depth (2.37-4.92
tok/round across payloads), so any single-prompt depth number is a draw, same lesson as
score basins; (c) the 07-06 prompt behaved like repro-flavor (its 4.57 tok/round sits at
P2's 4.41) where llama is at its best. Methodology gotcha: greedy raw-completion of
plain markdown docs EOSes immediately (dec=0) at 75K depth -- prose payloads need an
open continuation (mid-echo cut or list form); two payload shapes were discarded for
instant-EOS before P2 landed as docs+self-copy.

## 2026-07-07 (4th session, later) -- P15: constrain-tools engage-lag FIXED + serving-state gates

The 07-04 hole that kept `--constrain-tools` off in serving is closed. Root cause
recap: the grammar engaged on decoded text, but a spec round decides the round tail +
the new pending token in one launch -- everything decided in the marker's round sampled
UNMASKED, so a hallucinated name prefix could be chosen free and the mask then spliced
a trie-legal char into it mid-name ("getg_project" -> disengage -> tool-not-found
greedy loop -> score 0).

**Fix: round truncation + re-finish (no snapshots).** The round's per-lane GDN states
survive in the six rotating role buffers (the round commits by advancing `perm`, never
by copying), lane hiddens sit in x1..x1_f and lane logits in logits2, and KV/MTP rows
past the kept position are overwrite-safe. So when the marker completes at em[m-1],
`Engine::refinish_round(m, n)` rewinds: perm -= (n-m), d_P = P+m, h_next <- x1[m-1],
and the new pending is a masked re-argmax of lane m-1's resident logits -- the mask is
live from the FIRST post-marker decision, hence over every tool-name byte (the llama
grammar property). generate() gained an `on_round(em, n) -> m` hook (server-only,
greedy-only; CLI/canonical path never sets it); the constrainer moved to
src/toolconstrain.h (`BasicToolConstrainer<EngineT, TokT>`, unit-testable CPU-side)
with trigger detection in scan_round() -- the round batch is scanned PRE-emission, so
discarded tail tokens are never emitted, fed, or parsed.

**Serving-state gates (07-05 audit a/b/c) shipped with it:**
- clear-at-claim: generate() entry clears any leftover device constraint (stale lane-0
  mask + accept-cap-1 leak from a non-CUDA throw); guarded on host mirrors so the
  canonical path issues zero new device work.
- pool-full parity: mask-pool exhaustion now sticky-disengages for the REST of the
  request (one log + counter) instead of silently dropping per-mask; a cached failed
  add (-1 in host2dev) retries instead of caching the failure forever.
- split-brain: a per-slot pool id outside the engine's live pool is detected,
  logged, re-uploaded (rebinds counter).
- `[req]` gained `tg=engaged,disengaged,pool_drops,rebinds` after end= (parser-safe).

**Verification (tools/constrain_gate.sh + tools/test_toolconstrain.cpp):** 9 CPU unit
tests (engage/truncate/span/rem/skip-feed/sticky-pool/rebind/closer) ALL PASS; E2E
(tools/constrain_e2e.cu, production chatml_prompt preamble, tools named getg_project/
run_tests to force the hallucination case): emitted call has a REGISTERED name, JSON
parses, zero disengages, and refinish+truncation actually fired (trunc=1 -- the marker
completed mid-round); output bytes IDENTICAL across Q27_PMIN unset/0.5/1.0 (round-
phase invariance); clear-at-claim leak test RED->GREEN verified (guard disabled =
forced-token corruption, guard on = byte-identical legs); canonical 4c4120c7 EXACT;
test_kernels ALL PASS; compute-sanitizer memcheck 0 errors on the constrained E2E
(dual-arch build; sm_120-only builds throw fatbin-probe noise under the sanitizer).
Server smoke: non-stream + stream Anthropic paths both return registered-name
tool_use blocks with tg= telemetry.

Still open before default-on: constraint-cost soak (in-call cap is 1/round) and the
strict-parser A/B (zero rescues both legs) -- the latter needs a strict-parser knob
(tolerant rescues are currently unconditional). Q27_TOOL_SPLIT stays forbidden under
--slots (P11 race, unchanged). Constrained+sampled remains Phase 3 (tc.enabled gates
on greedy).

**Constraint-cost soak (same session).** Identical-request replay at 75.7K depth
(p1-transcript payload + write_note tool, response = one ~195-token tool call, 1 cold +
3 replays/leg, gated production config): OFF 102.2 t/s (2.67 tok/rnd, 1.9s warm turn)
vs ON 33.0 t/s (1.02 tok/rnd -- the in-grammar cap=1 -- 5.9s warm turn). Both legs
emitted byte-identical calls (the model was emitting a valid call anyway), so this is a
clean cost read: **the capped grammar is 3.1x slower inside call bodies at depth**
(+4s per call-turn at 75K; in-call rate 33 t/s, better than the old ~22 estimate but
still the dominant cost). Default-on verdict: keep OFF for speed-sensitive eval
serving; the flag is now SAFE (no score-0 basins) and buys structural validity when
robustness matters -- the strict-parser A/B can run with it on. The known speed fix
remains the P11 split path (proven token-identical, 4.2x in-call) blocked on its
orchestration race.

**P15 adversarial review pass (same session).** Independent reviewer confirmed the
refinish/rewind math correct on every reachable path (gated widths, maxd auto depth-5,
DEXIT, EOS-in-kept/discarded, R1b boundary, CLI-canonical unreachability) and the
invariant as structural, not just empirical. Findings fixed: **M1** dangling tc-hook
lambdas surviving a non-CUDA throw out of generate() (httplib catches at routing, next
request on a hook-less path would call a destroyed stack frame) -- HookGuard RAII nulls
on_pending/on_drafts/on_round on any exit, constructed after slot_guard so hooks clear
before the slot frees; **m3** pool_dead set mid-scan now stops the scan (a second
same-round marker with a cached entry mask could bypass the pool and disengage
nondeterministically mid-call later); **m4** a call completing inside the entry token's
remainder no longer stages a closed-state mask (defensive; not producible with the real
BPE vocab); **m5** the gate now asserts trunc>=1 so the perm-rewind path (not just the
m==n degenerate) stays exercised. **m2** documented as a comment: the split-brain check
is range-only -- safe while the pool is append-only; pool reset/eviction would need
whole-map epoch invalidation. m3/m4 landed RED->GREEN in test_toolconstrain (11 tests).

**prefill-attn Phase 2 (fp8 QK^T MMA) -- START + design (2026-07-07, branch
`prefill-attn-fp8mma` off master 535093f).** Baseline re-confirmed on this box: fp8 128K
prefill **67.80s / 1933 t/s** (`Q27_KV=fp8 --pf 131072 --ctx 133120 Q27_PF_NOSERIAL=1`),
within 0.6% of the Phase-1 68.20s -- cp.async is the current before-number. Canonical
4c4120c7 holds. **Design (resolves the ec1a54c revert's smem conflict):** the revert kept
s_q fp16 (50.7KB) + s_k + a double-buffered s_kraw = 117KB > 99KB cap. Fix: stage Q as
e4m3 (s_q 50.7->25.3KB) and DROP s_k (K read straight from fp8), freeing room for a
double-buffered s_kraw (2x8KB). New fp8-QK^T layout: s_q fp8 25.3 + s_v half 16.9 +
s_kraw fp8 x2 16 + s_vraw fp8 8 = ~66KB, comfortably under cap. QK^T becomes
`mma.sync.m16n8k32.e4m3.e4m3.f32` (8 k32 steps vs 16 k16); a-regs = uint32 of 4
consecutive e4m3 at k=tg*4+{0..3} and +16 (rows gid/gid+8), b-regs = same for K[n=gid];
accumulator layout identical to the f16 path so softmax/PV/output are byte-untouched. PV
stays fp16 (V still converted to s_v). Gated behind a 2nd template instantiation
(FP8MMA=true, CT=fp8 only) selected by `Q27_PF_FP8MMA` -- default path is a separate
compiled kernel, bit-identical. Tolerance-gated (fp16 path stays bitwise). Next:
implement + unit A/B <=5e-4 + prefix-cache identity + canonical + the P2 deep battery
(PPL/nll-long/needle 6/6).

**prefill-attn Phase 2 (fp8 QK^T MMA) -- SHIPPED (opt-in), +11.8% @128K.** New kernel
`k_attn_prefill_mma_fp8q` (mma.sync.m16n8k32.e4m3), engaged by `Q27_PF_FP8MMA=1` on the
fp8 path; the default fp16/fp8 kernels are untouched (separate compiled kernel, canonical
4c4120c7 unchanged). The revert's smem conflict is resolved by staging Q as e4m3 (s_q
50.7->25.3KB) and dropping s_k so s_kraw can double-buffer. **Bank-conflict padding was
load-bearing**: the fp8 QK^T packs 4 e4m3/uint32 and hits an 8-way conflict from contiguous
s_q/s_kraw; padding LDQ=260 / LDK=272 (+~1.4KB, total ~66KB) took it from +4.9% -> +11.8%.
128K wall (fp8, NOSERIAL): default f16-MMA 68.27s / fp8q 60.20s (2177 t/s) = **+11.8%
end-to-end ~= +22% on the attn kernel** (attacks the Phase-0 28% math_pipe_throttle at
12.5% occ). Correctness (greedy/identity -- `--nll` is per-token so it does NOT hit this
kernel; validated via batched-prefill routes): canonical unchanged, serial-vs-batched
continuation IDENTICAL @ pf=512 AND pf=4096, `--pfcache` warm/cold + mid-divergence
checkpoint-restore IDENTICAL, all under fp8q. Opt-in KEEP (EXPERIMENTAL until the deep
battery passes); default-on gated on a batched-prefill PPL/needle path (filed).
attribution: docs/perf-attribution-prefill-attn.md.

**Consensus review fixes (same session, 1/3 reviewers -- Claude/Gemini timed out, Codex
landed).** Applied: (#2, the important one) runtime device-capability guard in
attn_prefill_launch -- `Q27_PF_FP8MMA` now falls back to f16-MMA + warns on <sm_89 instead
of hitting the mma_e4m3 no-op stub and emitting silent garbage on the 3090; (#3)
static_assert(LDQ%4==0 && LDK%4==0) so a future stride change can't silently break the
uint32 fp8-load alignment; (#1) documented the t0==0 split-write contract (rule 6a) in the
new kernel; (#4) documented that the a/b fragment reads never touch the pad tail [HD,LD).
Skipped w/ reason: #5 one-time cudaFuncSetAttribute (matches existing code, single-GPU per
SECURITY-MODEL); #6 grid-remap divergence (Task 1.5 was never adopted -- L2 already 95.6%,
no default-path remap to diverge from). Re-verified after fixes: canonical 4c4120c7,
serial-vs-batched IDENTICAL @ pf=512, --pfcache IDENTICAL, fp8q 128K unchanged.

**Deep logit A/B wired + PASSED (default-on quality gate).** `--dump-logits` now fires on the
`--pf` batched leg, dumping the post-prefill position-N logits -- the only route through
k_attn_prefill_mma[_fp8q] (`--nll` prefills per-token via step_with, never hits it; this was
the plan's stated gate gap). fp8q vs default-fp8 at **position 131072** (131072-token varied
prompt, 101,863 distinct ids): **cosine 0.9999827** (tighter than the P2 fp8-KV 0.9995),
max|dlogit| 0.100, KL 1.9e-4, **argmax MATCH (95726), top-5 5/5**. So the fp8 QK^T adds
essentially zero quality delta at max depth -- the "silent deep-prompt quality loss" risk is
measured and absent. Perf reconfirmed same runs: default 68.00s / fp8q 60.19s = +11.5%.
Only remaining default-on gate: a needle-retrieval sweep (retrieval > single-logit
sensitivity), cheap now the dump path exists. Repro in the attribution doc.

**Needle sweep (fp8q) -- 6/6, default-on quality gate CLEARED.** q27-server on the fp8q path
(Q27_KV=fp8 Q27_PF_FP8MMA=1), 6-needle W&P sweep 10-95% depth. VRAM capped ctx to ~320K on
the 32GB card (fp8 KV 34KB/tok + 17.7GB model OOMs >~330K -- NOT fp8q-specific, default fp8
OOMs the same), so haystack trimmed to ~317K; deepest needle ~301K still BEYOND the 262K
native limit. **6/6 PASS with verbatim-correct sentences at every depth** (@301K: "...tidal
array is 88231."), matching the established default 6/6. So the full Phase 2 gate battery is
green: +11.8% @128K, greedy-identical (serial-vs-batched + pfcache), deep logit A/B cosine
0.9999827 / argmax MATCH @131K, needle 6/6 to 301K. fp8q trades no measurable quality for the
speedup -- default-on is defensible (the only caveat, VRAM at 34KB/tok on 32GB, is orthogonal
and hits both paths).

**FLIPPED DEFAULT-ON (Gabe's call, same session).** `Q27_PF_FP8MMA` now defaults to 1 on the
fp8 KV path -- fp8q is the fp8 prefill kernel by default; `Q27_PF_FP8MMA=0` forces the old
f16-MMA path (and the <sm_89 guard still auto-falls-back). The fp16 KV path and the fp16
canonical (4c4120c7) are untouched -- fp8q only ever engages under `Q27_KV=fp8`. Verified
post-flip: canonical 4c4120c7 unchanged; fp8 default (no env) now runs fp8q -- 128K 59.6s
(2199 t/s, +12.7% vs the 68.3s f16-MMA fallback), serial-vs-batched IDENTICAL @ pf=512;
`Q27_PF_FP8MMA=0` restores 68.3s. Branch prefill-attn-fp8mma pushed to origin.

**test_kernels fallout from the default-on flip -- FIXED (caught starting verify-gemv;
the flip commit skipped the test_kernels gate, own goal).** `fp8 attn prefill mma ==
fp16(deq) (bitwise)` FAILed (err 4.265e-2 vs tol 1e-30): the mma-mode fp8 leg now routes
through fp8q, which is tolerance-class BY DESIGN (fp8 QK^T), so the bitwise assertion no
longer applies to the default path. Fix: (1) `attn_prefill_launch` now caches only the
ARCH check and re-reads `Q27_PF_FP8MMA` per launch (was a static latch -- one process
couldn't test both paths; getenv at launch frequency, once per chunk x layer, is noise);
(2) the bitwise legs pin `Q27_PF_FP8MMA=0` -- they isolate the f16-STAGING invariant
(fp8 bytes -> half -> same MMA), which still holds bitwise; (3) NEW check `fp8q attn
prefill vs fp16(deq) (tol)` at bound 1e-1 (observed 4.265e-2; a fragment-layout bug gives
O(1)+ garbage). The same-err match between the old FAIL and the new tolerance PASS proves
fp8q engages under =1 and disengages under =0 in one process. test_kernels ALL PASS (225),
canonical 4c4120c7 holds.

**verify-gemv Task 0 -- branch + fixtures + baseline (2026-07-08, branch `verify-gemv` off
master 1709b93).** Fixtures regenerated (they were cleaned again): `scratchpad/prep_tokens.py`
rewritten per docs/perf-attribution-p14.md lines 20-27 -- corpus `sorted(docs/*.md)+README.md`
HF-tokenized with `hf-bf16/tokenizer.json`, tiled to 124,506 ids (max 248,068, matches the P14
recipe exactly); emits prompt2k/16k/61k.txt + toks60k/75k.txt. Regen: `python3
scratchpad/prep_tokens.py` from the repo root. NOTE the docs corpus GREW since P14 (18 files
now), so text != P14's -- acceptance is content-dependent and much higher on the new corpus
(templated BUILDLOG text): gated 61K decode is 163.2 t/s median n=3 (173 rounds/800 tok, 4.62
tok/rnd) vs P14-era 124-125 -- NOT comparable, do not read as a speedup. The content-independent
anchor reproduces: ungated nsys (node-trace, Q27_PROF_DECODE rig, 170 rounds) kernel-sums/round
= batched verify GEMV **12.38 ms/rnd** (P14 12.24, +1.2%), single/draft **3.11** (exact),
weight-stream total **15.49 vs 15.4 (+0.6%)** -- Task 0's 5% sanity gate PASS, harness valid.
test_kernels ALL PASS, canonical 4c4120c7 exact.

**verify-gemv Task 1 (Phase 0 ncu) -- PROCEED.** ncu'd `k_gemv_q4_n<5>` inside the full
engine (ungated server, 61K request, `--graph-profiling=node -k regex:k_gemv_q4_n
--launch-skip 2000 --launch-count 4 --kill 1`, root). The verify GEMV is **latency-bound at
39-47% of DRAM peak**, NOT at the weight roofline: long-scoreboard (L1TEX dependency) is
**90.4% of the 68.8-cycle inter-issue gap**, issue slots 11-13% busy, occupancy 63-89% (not
the limiter), L1 hit ~97% (activations resident) / L2 17-29% (weights stream). Smoking gun:
**global loads use 10.0 of 32 bytes per sector** -- the weight-read pattern wastes ~2/3 of
each DRAM transaction. ncu's own OPT estimate: 43.9% speedup on the L1TEX stall, consistent
with the plan's ~37% roofline gap (15.4 vs ~9.7 ms/round floor). Decision matrix row 2 ->
**Task 2 GO** (bitwise-safe: coalesce/vectorize the weight walk + more loads in flight);
dp4a issue is NOT the limiter, so Task 3 (tensor-core, canonical-breaking) looks unjustified
-- revisit only if Task 2 stalls short. Full counters + commands:
docs/perf-attribution-verify-gemv.md.

**verify-gemv Task 2 (Phase 1, bitwise-safe) -- +5.5% decode @61K, KEEP.** One surgical
change in `k_gemv_q4_n`: the per-column activation reads go 4x uint2 -> 2x uint4 (same 32
bytes, same component order into the same dp4a sequence -- integer-exact, fp acc order
untouched, so greedy is bitwise by construction; alignment holds, eo + 32*ch is a
16B-multiple into a cudaMalloc base). This halves the L1TEX wavefronts on exactly the loads
Phase 0 fingered (8B at 32B lane stride = 10/32 bytes/sector, long_scoreboard 90%).
Deliberately NO smem (the -4% smem-staging lesson: in-graph occupancy dominates). Gates:
canonical 4c4120c7 EXACT; test_kernels ALL PASS (225); full-engine 61K production-gate
decode **163.2 -> 172.2 t/s median n=3 (+5.5%, spread 0.1%)**, dec_ms 4901 -> 4645, same
173 rounds (engine-level bitwise confirmation). Captures ~1.48 ms/round of the ~5.7
ms/round roofline gap (~26%). NOT applied to k_gemv_q8_n (its activation loads are already
uint4) or the single-column kernels (draft path, 3.1 ms/round, same pattern applies --
candidate follow-up).

**verify-gemv Task 2 close-out: single-column k_gemv_q4 + plan verdict.** Same 2x-uint4
change on the draft-path k_gemv_q4: 172.2 -> 172.9 t/s median n=3 (+0.4%, noise-level,
bounded at +0.7% by the 1.6 ms/round share; KEPT for strictly-positive median + load-shape
consistency across the q4 kernels). Canonical EXACT both commits, test_kernels ALL PASS,
same 173 rounds throughout. **Task 2 total: 163.2 -> 172.9 t/s (+5.9%) @61K gated, greedy
bitwise, ~28% of the roofline gap captured.** **Task 3 (tensor-core verify, breaks
canonical): NOT JUSTIFIED -- killed.** Phase 0 showed dp4a issue was never the limiter;
the residual gap is L1TEX-latency structure and the remaining safe levers are
marginal-band with occupancy risk (fd2/smem precedents). verify-gemv plan COMPLETE:
attribution + verdicts in docs/perf-attribution-verify-gemv.md.

**Strict-parser knob (Q27_TOOL_STRICT=1) -- the strict-parser A/B is now runnable.** The
tolerant rescues were unconditional; the knob severs ALL of them: parse_tool_call goes
plain-JSON-only (no <content>-tag rewrite/mode-3, double-encoded arguments REJECTED
instead of unwrapped) and parse_bare_tool_calls (drift modes 1-6, both server call sites)
returns empty. Suppressed rescues log `[q27-strict]` so a campaign can count what the
tolerant chain would have carried; the tolerant leg keeps its existing `[drift]`/
`[tool-fallback]` counters. Read-once env (one leg per server run). Gates: build clean,
test_tokenizer PASS both modes, E2E smoke = byte-identical well-formed tool_use in both
modes on a clean call (no rescue fired, strict did not over-suppress). Default behavior
byte-unchanged (knob off = old code path).

**Strict-parser A/B campaign (same session) -- VERDICT: NOT engine-true; mode-1 rescue is
load-bearing.** CC greedy, T2+T8, canonical server config (PMIN=0.5, slots 2), three legs:
tolerant T8 **0.837**/150s with **12 rescued calls** (mode-1 dropped-wrapper, 2 fallback
events, one on the OPENING turn); strict T8 **0.000**/4s -- the first assistant turn emits
its tool calls wrapper-less, strict suppresses, CC sees text-only and one-shot-quits;
strict + --constrain-tools T8 **0.549**/491s -- the grammar carries every wrapped call
(session survives, 2.17M tokens) but ONE mid-session wrapper-less turn still bypasses it
(the constrainer engages at <tool_call>; bare-JSON turns are invisible to it) and the lost
calls degrade hidden_tests 0.94 -> 0.22. T2 was uninformative (one-shot-quit basin fired on
BOTH legs -- tolerant's opener was un-rescuable even by the full chain, [drift] UN-RESCUED).
Fairness: llama-server's chat parser has the same tolerance class (its own wrapper-less
recovery), so cross-engine A/Bs remain apples-to-apples as SYSTEM comparisons; the refuted
claim is only that q27's scores survive with rescues off. FOLLOW-UP LEVER (not built):
engage the constrain grammar on a bare {"name" opener too -- closes the wrapper-less bypass
and would make strict+constrain the true zero-rescue configuration. Logs:
parser-{tolerant,strict,strictc}-server.log (session scratchpad). README quality-gates
updated; this closes the last open red-team gate.


## 2026-07-08 (accept-gate Phase 0) -- depth economics REFRESHED post-verify-gemv: d5 crossover yield is ~0.35, not ~0.7

Phase 0 of docs/acceptance-gate-design.md (plan docs/plans/2026-07-08-acceptance-gate.md),
branch acceptance-gate. Telemetry first, then the maxd6 A/B re-run on current HEAD -- the
maxd6 economics predate verify-gemv's +5.5% decode and are stale in the
pessimistic-for-depth direction.

**Telemetry (shipped).** (a) Per-lane fired/accepted counters `glf/gla` in `[req]`
(9113e8e): lane j fired iff cap>=j, accepted iff n>=j+1 -- the conditional yields
p(acc_j|fired_j) that the gch/gnh marginals cannot reconstruct. (b) `--stats` own-pass
margin bins (fe37595): p(acc_k | m_k bin, prefix ok) for k=3..5; m3/m4/m5 were computed
and voided since E3. Also fixed there: the stats pend arrays are sized N+8 but scoring
indexed them by known_idx across the WHOLE prompt -- any prompt longer than --stats N
read heap garbage and zeroed every counter (bug present since E3; --stats needs
prompt << N usage or this fix). Rig fix en route: reqlog_gate.sh predated the 2f47508
anthropic-shaped ctx-limit 400 and crashed on it uncaught (bf782c7); C1/C9/C15 updated
(no [req] line for validation-rejected requests -- /v1/completions silently returns
0 tokens for prompt>ctx instead of a 400, noted, not fixed here).

**A/B rig (5db7b23).** tools/make_payloads.py + tools/accept_ab.sh: frozen ~26K
raw-completion payloads spanning the saturation axis + a 61K docs leg; fresh server
per leg, fp8 KV + --fast-head, greedy, 1 cold prefill + 3 identical replays (medians,
det asserted). Payload lessons paid for: plain mid-function cuts EOS in ~8 tokens at
26K depth (twice); an explicit open stub forces sustained generation; self-copy at 13K
token echo distance does NOT saturate acceptance (y5 0.53) -- the >0.9 regime needs
short-range repetition, and even a 2.5KB block repeated (depth-match P4 flavor) only
reached y5 0.51 here.

**Measured (scratchpad/accept_ab_run3.log, accept_ab_61k.log), Q27_PMIN=0.5 dexit1:**

    payload      ctx   d4 t/s   d5 t/s  d5 net   auto    y5(d5)  fired5
    echo         28K   151.6    155.7   +2.7%    155.7   0.514   0.507
    docs         25K   168.1    177.5   +5.6%    175.2   0.769   0.600
    codegen      26K   163.9    164.2   +0.2%    169.5   0.355   0.443
    testgen      27K   162.0    168.3   +3.9%    166.1   0.447   0.576
    docs61k      61K   114.4    112.5   -1.7%    113.0   0.282   0.433

**Findings, decision-grade:**
1. **The maxd6-era loss regime has mostly inverted at 26K.** Fixed-d5 is >= d4 on every
   26K flavor (geomean +3.1%); the maxd6 measurement had -5.4% codegen at y5 0.56 --
   post-verify-gemv, +0.2% at y5 0.355. The crossover yield is **~0.35** (bracketed:
   -1.7% at 0.282, +0.2% at 0.355, +2.7% at 0.45+), half the maxd6-derived ~0.7.
2. **maxd_lo=0.10 is 3.5x below the crossover AND in the wrong units.** yield_ema today
   is p(n=6) over ALL depth-5 rounds (~y5 x fired); at docs61k steady state that is
   ~0.12 > 0.10 -> auto would sit at depth-5 losing -1.7% indefinitely. The
   acceptance-gate premise stands, at depth-5 scale: demote on CONDITIONAL yield
   (p(acc5|fired), the glf/gla quantity) against a bar at the measured crossover.
3. **Promote/grace churn is a real live cost.** auto on docs61k: fired5 0.032 with
   y5 0.000 -- promote, burn an 11-round grace window with zero deep accepts, demote,
   repeat = **-1.2% vs plain d4**. Conditional yield + bar 0.35 demotes on the same
   evidence in the same window; HI=0.5 stays (correctly excludes docs61k sat~0.42,
   admits every 26K winner -- second revalidation of P13's bar).
4. **auto is the right default shape.** It beat fixed-d5 on codegen (+3.4% vs d4 by
   demoting through low-yield stretches) and tracked within ~1.5% of the best fixed
   leg elsewhere; with the Phase-1 bars it should dominate both fixed legs everywhere.
5. **Per-lane ladder below 5 is DEAD (plan Task 7 gate).** y2..y4 sit at 0.42-0.95
   across all flavors -- no measured lane below 5 approaches the ~0.35 bar. Phase 2
   collapses into Phase 1 as the plan anticipated.
6. **maxd6/ceiling-6 reopens LATER, with these numbers.** With d5 breakeven-to-winning
   across the 26K envelope and a working acceptance bar, the d6 GO-IF calculus
   (maxd6-decision.md) deserves a re-run -- but only after Phase 1 ships and only as
   the separately-priced P12b-class build. Not this branch.

Own-pass margin bins (--stats, Task 2): populated and sane on synthetic prompts, but
the frozen-payload bins are the useful calibration -- deferred to a follow-on --stats
pass on the payload set if the theta-schedule question ever becomes live; the yield
feedback path does not need it (design doc "supporting role").

## 2026-07-08 (accept-gate Phase 1) -- SHIPPED: conditional yield + retuned bars; auto is now the recommended gated config (+2.7% geomean vs d4-gated)

The Phase-0 measurement made Phase 1 knob-shaped and killed Phase 2 (per-lane ladder:
no lane below 5 approaches the bar; plan Task 7 gate). Changes, all in the extracted
controller (src/depthctl.h, ecf7a31 refactor -- canonical-exact, round-identical):

1. **Conditional yield.** yield_ema updates only on rounds where the 5th lane FIRED
   (gate_cap >= md_used); unfired rounds carry no evidence and, under early-exit,
   barely pay. The old unconditional EMA (~y5 x fired) sat above lo=0.10 on traffic
   where fixed-d5 measured -1.7% -- a sticky-loss regime the controller could never
   leave. Now the EMA measures y5 in the same units as the bar.
2. **Promote-seed clamp** min(1, 2*lo) -- 2*lo past 1.0 would stretch the demote grace
   window arbitrarily.
3. **maxd_lo 0.10 -> 0.35** = the measured crossover (Phase 0: -1.7% at y5 .282,
   +0.2% at .355, +2.7% at .45+). maxd_hi stays 0.50 (third revalidation: excludes
   docs61k sat~0.42, admits every 26K winner). Both still env-overridable.

17 CPU tests (tools/test_depthctl.cpp, `make` target build/test_depthctl) pin the old
semantics + the three changes (unfired-no-evidence, seed clamp, 33%-yield demotes /
50%-yield holds at the production bar).

**A/B (frozen Phase-0 payloads, scratchpad/accept_ab_phase1.log):**

    payload      d4      d5      auto(old)  auto(NEW)  new vs best fixed
    echo         151.6   155.7   155.7      156.4      +0.4% (beats both)
    docs         168.1   177.5   175.2      176.2      -0.7% of d5
    codegen      163.9   164.2   169.5      170.0      +3.5% (beats both)
    testgen      162.0   168.3   166.1      166.6      -1.0% of d5
    docs61k      114.4   112.5   113.0      113.1      -1.1% of d4

auto(new) geomean vs the current production rec (d4-gated): **+2.7%**; vs fixed-5:
+0.6% -- auto dominates both fixed legs overall. **Production rec becomes
`Q27_PMIN=0.5 Q27_MAXD=auto`** (binary defaults unchanged: auto remains opt-in, so
nothing moves for existing configs; lo=0.35 applies only under auto).

**Honest residuals.** (a) docs61k auto pays -1.1% vs plain d4 -- promote/grace churn
(sat bursts past 0.5, zero-yield window, demote, repeat; fired5 .032, y5 .000).
Identical in the old controller; a demote-count promote-escalator could shave it,
not built (YAGNI at 1% worst-flavor). (b) The plan's T8-matched >= +1% criterion is
satisfied by envelope interpolation only (live T8 ran fired .79 / y5 .82 -- past the
docs point, where d5 = +5.6%); the maxd6 T8 payload artifacts are gone and were not
reconstructed. (c) One plan criterion technically missed: "no payload below its d4
baseline" -- docs61k at -1.1%, see (a); Q27_MAXD=4 remains available for pure
long-ctx docs serving.

Gates fresh at HEAD: test_kernels ALL PASS, test_depthctl 17/17, canonical 4c4120c7
EXACT (ungated + gated-5 + auto), reqlog_gate both phases PASS, shortbench mean 179.8
(baseline 179.7), replay determinism OK on all legs. Files: depthctl.h, engine.cuh,
engine.cu, server.cu, Makefile, tools/{test_depthctl.cpp,accept_ab.sh,make_payloads.py,
reqlog_gate.sh}, docs/{acceptance-gate-design.md,plans/2026-07-08-acceptance-gate.md}.

**Follow-on unlocked (not this branch): reopen maxd6/ceiling-6.** With the crossover
at ~0.35 and a bar that tracks it, the d6 GO-IF arithmetic deserves a re-run on the
refreshed economics -- echo/docs flavors sit at y5 .5-.8 with headroom above the
ceiling, and the depth-match P4 tail (llama +45% at 100% acceptance) is still on the
table. Cost side unchanged: P12b-class 6->7 widening + quantize3-landmine audit +
157 MB (maxd6-decision.md items 1-7).

## 2026-07-08 (maxd6 GO-IF RERUN) -- VERDICT FLIPPED: GO on auto-ladder-6 (was NO-GO 07-07)

Rerun of the maxd6-decision.md GO-IF on post-verify-gemv economics + the Phase-1
acceptance bar; full verdict appended to docs/maxd6-decision.md. Headline: all three
GO-IF conditions now PASS; depth-6 is worth one P12b-class build session as an auto
ladder extension (4..6), est +4-5% on CC-flavor traffic, ~0 elsewhere, bar-protected.

**New rig: tools/burst_sim.py** -- exact offline gated-round simulation (any ceiling,
any theta) from --burst-stats chain CSVs, round-sampled. Validated against CLI spec
legs on the same trajectory: shallow-round histograms match; deep rounds UNDERCOUNT
because burst chains seed from serial-path hiddens which differ in ULPs from live
verify-lane hiddens -- deep chains amplify near-tie flips (echo tok/round -5% vs CLI,
echo-heavy cctx -27%). Sim = conservative bound; measured-sat extrapolation used
where the bias bites. En-route finding worth keeping: emitted TOKENS are identical
across serial/spec/server paths (reconfirmed), but DRAFT CHAINS are not ULP-stable
across those paths -- any future draft-side instrumentation must seed from the same
path it models, or eat this bias.

**The payload that settled it: cctx** -- a real CC bench-session transcript replayed
as a raw 25.8K completion (recipe in decision doc; payload NOT committed, private
transcript). First constructed payload to reproduce the live-T8 saturation profile
(sat5 .714 vs live .652; 5.29 tok/round vs 5.03). Measured d4->d5 on it: 204.1 ->
218.5 t/s = **+7.0%** -- the same replay class that measured breakeven-at-best on
07-07. Constructed-payload lesson extended: docs/code/echo builds never exceeded
sat5 ~0.46; only a REAL agent transcript (tool bodies echoing file content) reaches
the live regime -- keep one on hand for depth work.

Numbers (CLI legs, serial prefill, fp8 KV + fast-head, greedy, theta 0.5):
cctx d4 4.56 t/r sat4 .807 | d5 5.29 t/r sat5 .714 | sat decay/level .885;
d4->d5 tok/round gain +0.73 ~= sat5 (model check ~2%). d6 projection +0.63 t/r at
+1.6-1.9 ms/rnd -> +4-5%. Ladder admits only cctx-class traffic to level 6
(sat5 elsewhere: docs .46, echo .26, testgen .26, codegen .16, docs61k .12).

Build gates (before any default flip): width-7 lane measured via forced-cap sweep;
canonical + identity + determinism; cctx >= +3% replay A/B with no envelope payload
below its d5 baseline; glf/gla extended to lane 6. d7/d8 explicitly deferred to
live lane-6 telemetry. Files: tools/burst_sim.py (new), docs/maxd6-decision.md
(verdict), scratchpad burst_*.csv + payloads (uncommitted).

## 2026-07-08 (maxd6 build) -- SHIPPED: auto-ladder 4..6; cctx auto +4.7% vs d5, byte-identical, canonical EXACT

The GO verdict built (branch maxd6-ladder; plan docs/plans/2026-07-08-maxd6-build.md).
The P12b-class 6->7 widening plus a 3-bar depthctl ladder; ~5 hours measurement-to-merge,
no lost sessions -- the quantize3-landmine audit was priced in and the lane fix landed
with the widening, not after a divergence hunt.

**Engine widening (b24864d).** perm mod-6 -> mod-7 + 7th GDN role (S_spare6/ring_spare6,
+157 MB); all graph arrays [6]->[7] (verify_graph_w -> [8][7], draft_step_graph ->
[6][7]); 7th verify lane (_g) through qx5/mm5/gdn_pair/attn_pair/ffn_pair + the vw>6
chains; gemv_q4_n/q8_n case 7; quantize3 7th explicit lane (THE P12b landmine class,
fixed pre-emptively); logits2 7*VOCAB; fd scratch 7 lanes; outcome {n,t1,dr1..dr6,pending}
(9 ints); prep/finish_round widened (a6 chain, n<=7); d_draft6/h_next6/d_pos_m6/d_pos_g;
margins [6]; em[7] callers; glf/gla/gch/gnh lane-6/n-7 telemetry. Sampled path untouched
(ceiling 4); constrained path unchanged (depth-4 draft via draft_graph_lo, now captured
for gate_maxd>=5). Q27_MAXD=6 fixed accepted for testing; auto = the ladder; auto+DEXIT=0
clamps to 4..5 (no depth-5 monolithic graph under a depth-6 ceiling; dexit is default-on).

**depthctl ladder (135e4a0 + 6066ef7, 30 CPU tests).** Levels 4..6, per-level sat/yld
EMAs, fresh-stint hysteresis on every level entry (enter(): sat=0, yld=min(1,2*lo) --
prevents stale-EMA 6->5->4 cascades; behavior-identical at k_max=5). THREE bars, each
measured onto a real failure mode:
- hi=0.50 (4->5 promote, P13, revalidated again);
- hi6=0.60 (5->6 promote): docs-flavor sat5 is BURSTY around .46-.5 and level-6 there
  measured -1.9..-2.9%; cctx sustains .71. 0.60 alone did NOT stop docs (bursts cross it);
- flo6=0.45 (level-6 fired-rate demote): the discriminator that finally separated docs
  from cctx is NOT yield (both y6 ~.70 when fired) but HOW OFTEN margins run 6-deep
  (docs ~.3 vs cctx .6+). The 6th draft step runs on every cap>=5 round; rare 6-deep
  firing = pure tax the conditional-yield bar cannot see. Level-6 only: at level 5,
  low-fired/high-yield flavors WIN (testgen fired .30, +3.9%) -- a fired bar there
  would be wrong.
All env-tunable: Q27_MAXD_HI/HI6/FLO6/LO/EMA.

**Measured (CLI legs + server replay A/B, fp8 KV + fast-head, greedy theta 0.5):**

    cctx (real-CC flavor)   d4 202.6 | d5 216.1 | d6 222.0 (+2.7% vs d5, 7-tok
    rounds on 64%) | auto(server, warm) 222.6 -- auto matches-or-beats fixed-6
    by demoting through weak stretches. [CORRECTED 2026-07-09, review: this
    line originally claimed +4.7% vs d5, but 222.6 is a warm-SERVER number and
    the d5 216.1 here is a CLI leg -- cross-harness, so the +4.7% is not a
    valid delta; vs the CLI d5 it is +3.0%. Same-harness rerun on the
    post-review binary: see the 2026-07-09 review-fixes entry.]
    Emitted text BYTE-IDENTICAL d4/5/6/auto.
    Ladder live: 2 promotes 0 demotes, final ceiling 6.
    Envelope auto vs Phase-1 auto: echo/codegen/testgen/docs61k never promote to 6
    (sat5 .12-.30 vs hi6) -> within noise (+0.6..-1.2%); docs (the bursty boundary
    flavor at sat5 ~.46-.49) was -2.9% pre-flo6, now 175.3 = noise vs Phase-1
    (-0.5%), -1.2% vs fixed-d5 (the same auto-vs-fixed gap Phase 1 had).

Width-7 lane cost MEASURED (closes the extrapolation): d5->d6 at fired6 .64 costs
+2.05 ms/round (24.48 -> 26.53 CLI) ~= 3.2 ms per fired-6 round all-in (6th draft +
width-7 verify + re-segmentation) -- the maxd6-decision extrapolation said 1.6-1.9;
benefit side was dead-on (tok/round 5.89 predicted 5.92).

Gates fresh at HEAD: test_kernels ALL PASS, test_depthctl 30/30, canonical 4c4120c7
EXACT (ungated + gated 4/5/6/auto), reqlog both phases, shortbench 177.6 (noise band),
replay det=OK on converged legs (NONDET across replays on mid-convergence auto legs is
the ladder's carried EMA state re-segmenting rounds -- tokens identical, documented).

**Production rec unchanged in shape, better in depth: `Q27_PMIN=0.5 Q27_MAXD=auto`**
(now the 4..6 ladder). Binary defaults untouched -- gate opt-in as always. NONDET
round-segmentation across replays under auto is expected telemetry behavior.

**Residuals / next.** (a) docs-class boundary traffic keeps a ~1% auto-vs-fixed gap
(promote exploration; bounded by flo6). (b) d7/d8: cctx sat6 .64 STILL saturates --
the ladder extends by the same recipe (S_spare7, [8]->[9] arrays, hi7/flo7) IF live
lane-6 telemetry (glf/gla now report it) shows sustained sat6 >= .6 on real serving;
the pointer-array lane refactor (maxd6-decision.md) becomes worth it at d7+. (c) The
P4-echo tail (llama depth-10 +45%) remains open above d6.

## 2026-07-08 (new model onboarded) -- unsloth Qwen3.6-27B-MTP (base) repacked + running

The BASE Qwen3.6-27B-MTP (unsloth/Qwen3.6-27B-MTP-GGUF, BF16 split-GGUF ->
llama-gguf-split --merge -> tools/repack.py v1.4 `(ssm_out|attn_output)\.`) now
runs on the maxd6-era engine. Files: /mnt/ai/models/qwen36-27b-mtp/{.q27,.tok}
(17.73 GB, 867 tensors incl output_q4); merged BF16 kept at
/mnt/ai/models/qwen36-27b-mtp-gguf/ for future repacks (split shards deleted).

Architecture metadata is an EXACT match to Qwopus (same qwen35 hybrid, 65
blocks, M-RoPE [11,11,10,0], nextn=1, vocab 248320, 866 GGUF tensors); the
exported tokenizer is BYTE-IDENTICAL to qwopus-27b-mtp.tok. Quant RMSE profile
matches the Qwopus repack band (worst ~0.117 Q4 tensors).

Baselines (this model's own -- Qwopus gates do NOT transfer):
- canonical md5 **a2982c5197c627551b27d76a0a94b220** (128 tok, ctx 2048, --spec;
  shortbench_suite.sh now takes CANON_MD5 env for non-default models);
- shortbench suite mean **166.1 t/s** (159.9-173.7, t/round 3.10-3.37) vs
  Qwopus 179.7 (3.20-3.88) -- same kernels/shapes, the ~8% is pure MTP
  acceptance, slightly lower on the base model;
- server smoke: /v1/messages chat coherent (--no-think, fast-head).

Ops notes: hf download stalled twice on the 49.9 GB shard at 160 MB (single
connection); aria2c -x8 against the resolve URL pulled it at full rate --
prefer aria2c for multi-GB HF shards on this box. The GGUF repo also carries
mmproj-* files (the base model is multimodal-capable); q27 uses the text tower
only.

## 2026-07-08 (qwen36 base thunderdome) -- T8 0.842 / T2 0.851 after drift-mode-7 parser rescue (was 0.00/one-shot-quit)

First thunderdome trials of the base Qwen3.6-27B-MTP through q27 + Claude Code
(claude-code-q27-haight, ~/thunderdome rig; server Q27_PMIN=0.5 Q27_MAXD=auto,
fp8 KV, fast-head, --no-think, ctx 131072).

**The 07-06 lesson repeated verbatim on a new model: tool-call PARSING was the
whole ceiling.** At greedy the base model deterministically opens its first CC
tool call as `{"name":\n{"function":{ARGS}}}` -- wrapper-less AND name-dropped
(mode 6) AND the orphaned args nested under a lone "function" shell. The mode-6
key-signature rescue could not see through the shell -> CC got text-only ->
one-shot-quit -> T8 0.00 (84 output tokens, 8s). Thinking mode did NOT help
(empty think block, identical drift).

**Drift mode 7 rescue shipped** (api_common.h `infer_tool_name_unwrapped`):
raw mode-6 inference first (existing rescues undisturbed), then peel bounded
single-key object shells (function/arguments/parameters/input/tool_call) and
retry; the recovered call carries the INNER args. Gated by 2 new
test_tokenizer cases (the observed bytes verbatim + a raw-mode-6
no-regression case); reqlog_gate both phases PASS.

**Scores (1 trial each, greedy):** T8 analytics-dashboard **0.842** (312s,
hidden 0.938 / agent 1.000 / cov 0.815 / metrics 1.000; Qwopus refs
0.796-0.837); T2 collab-server **0.851** (136s, hidden+agent 1.000;
Qwopus refs 0.847-0.848). The base model is AT-OR-ABOVE Qwopus on both
tasks once the drift mode is rescued -- n=1 greedy, so treat as basin
samples, not a cross-model verdict.

**maxd6 ladder, first LIVE-CC validation (T8 trial, 77 reqs, 7018 gated
rounds):** md6 on **84%** of rounds (5865) vs md5 16% / md4 1%; 13 promotes /
12 demotes (bounded churn); cap=6 fired on 68% of all rounds; live lane
yields y1..y6 = .985/.957/.917/.869/.817/**.782** -- lane 6 far above both
bars (yld .35 / fired .45); live sat6 ~.64 = the cctx replay number. The
d7 headroom signal is now confirmed on real serving traffic.

Ops: ~/thunderdome (NOT /mnt/ai/projects/thunderdome) is the live rig with
the *-haight adapters; `./thunderdome run --orchestrator claude-code-q27-haight
--task T8 --trials 1`; adapter expects the engine on host port 8081.

## 2026-07-08 (cross-engine, base qwen36) -- q27 2.26x llama decode at matched depth; trials q27 0.842/0.851 vs llama 0.818/0.841

Same BASE model (Qwen3.6-27B-MTP) both engines, quant bracket around q27 v1.4's
5.25 bpw: llama Q4_K_M 16.8 GB (4.98) and Q5_K_M 19.5 GB (5.79), quantized
locally from the merged BF16 (llama-quantize needed a rebuild -- stale-lib
symbol error). llama = mainline build 1491, tuned config (--spec-type draft-mtp
--spec-draft-n-max 10 --spec-draft-p-min 0.5, q8_0 KV, temp 0,
enable_thinking:false), CUDA_VISIBLE_DEVICES=0 (auto-split OOMs onto the busy
3090 at 131K ctx).

**Rate at matched depth (cctx replay, 25.8K tok, n=256, warm medians):**
llama Q4_K_M **98.4 t/s** (draft accept 179/271 = 66% at n_max 10) vs q27 d4
202.6 / d5 216.1 / d6 222.0 / **auto 222.6 -- 2.26x**. llama Q5_K_M leg
CRASHED serving the first request (segv in update_slots, llama-side,
unreported). July context: tuned llama measured 111-190 t/s on Qwopus flavors
at 75K; on THIS transcript-flavor payload the gap is structural -- q27's
ladder runs 5.3-5.9 tok/round where llama's chain nets ~3.3.

**CC trials (claude-code-*-haight, n=1 greedy, same harness/tool block):**
T8 q27 **0.842 @ 312s** vs llama **0.818 @ 178s**; T2 q27 **0.851 @ 136s** vs
llama **0.841 @ 165s**. Wall is trajectory-confounded as always (q27's T8 run
did ~1.7x the prompt volume: 5.4M vs 3.2M tokens served); trial-weighted
decode rates (q27 177 t/s @ up-to-109K vs llama 185 t/s on its shallower mix)
are NOT comparable across different trajectories -- the replay number above is
the controlled comparison. Prefill effective: q27 2309-2569 t/s (96.7% prefix
cache) vs llama 1720 t/s (its own cache also ~97% effective).

Scores: q27 above llama on both tasks at n=1; treat as basin samples. Both
engines' first-turn behavior on the base model depends on their tolerant
parsers (q27 needed the mode-7 rescue; llama's own wrapper-less recovery
carried its legs -- same tolerance class, system-fair).

## 2026-07-09 (three-stack cctx shootout) -- q27 222.6 vs vLLM-NVFP4+MTP 155.5 vs llama 98.4 t/s

Third leg of the cross-engine comparison: PrismaSCOUT NVFP4 (rdtand mixed
NVFP4+BF16 export of BASE Qwen3.6-27B, 5.31 bpp, compressed-tensors) on vLLM
nightly v0.23.1rc1.dev748, 5090-pinned, kv fp8, 32K len (vllm-serve keys
prismascout-27b[-mtp] added in the 5090-local-llm repo). Same cctx payload
(25.8K CC transcript), decode isolated as wall(256) - wall(1) since vLLM
AUTO-DISABLES prefix caching on hybrid-GDN models (enable_prefix_caching=False
in the engine config; hit rate 0%) -- every request re-prefills.

    decode @ cctx 25.8K       t/s     bpw    notes
    q27 auto ladder          222.6    5.25   96.7% prefix-cache in live serving
    vLLM PrismaSCOUT MTP k=3 155.5    5.31   MTP +112% on this stack/payload
    llama Q4_K_M tuned        98.4    4.98   draft-mtp 10/0.5, accept 66%
    vLLM PrismaSCOUT no-MTP   73.5    5.31

Two findings that update priors: (1) the July "vLLM MTP net-negative on 5090"
does NOT generalize -- that was modelopt+marlin at short ctx; on
compressed-tensors + current nightly + echo-heavy 26K traffic MTP k=3 is
+112%. Both results stand in their contexts; the finding is
backend/payload-conditional. (2) The bigger structural gap for AGENTIC serving
is prefix caching: the hybrid-GDN recurrent state makes APC all-or-nothing,
vLLM punts (0% reuse -> every CC turn re-prefills the whole conversation,
2.5-2.7s at 26K and growing with depth), llama checkpoint-searches, and q27's
P8/P9 snapshot+checkpoint machinery ran 96.7% reuse over the live trials.
Batch-1 decode at matched depth: q27 +43% over vLLM's best leg despite NVFP4
tensor-core dequant -- bandwidth-bound decode doesn't care about MMA peak.

## 2026-07-09 (maxd7) -- BUILT AND MEASURED: depth-7 machinery ships OPT-IN; auto stays 4..6 (width-8 round cost ~2x extrapolation)

The d6 verdict's own gate (live sat6 >= ~.6; measured .64) authorized the build.
Widening = the maxd6 recipe one lane deeper (branch maxd7-ladder, plan
docs/plans/2026-07-09-maxd7.md): _h lane, perm mod-8, S_spare7 (+157 MB),
width-8 verify, gemv case 8, quantize3 8th explicit lane, outcome
{n,t1,dr1..dr7,pending}, hists to cap7/n8, glf/gla lane 7; depthctl level 7
(hi7=.60, flo7=.45, 40 CPU tests). Gates: canonical 4c4120c7 EXACT at
{ungated, 4,5,6,7,auto}; cctx emitted text BYTE-IDENTICAL across all six
ceiling configs; test_kernels ALL PASS.

**En-route incident, fully forensic'd (half the session): the cctx trajectory
FLIPPED between 07-08 and 07-09 builds.** Yesterday's replay legs (b4a02285,
4.56 tok/round, empty-think basin) vs today (a225f6a7, 3.21, thinking basin)
-- divergence at TOKEN 3, a near-tie newline flip after <think>. Exonerated in
order: the maxd7 widening (stash test), the hi6-era source (exact b24864d
rebuild reproduces TODAY's text), rebuild determinism (two same-source builds
byte-differ in ELF metadata only, behaviorally identical), the model file
(byte-identical to a fresh deterministic re-repack from the intact BF16;
an unexplained mtime bump at 21:22 was benign -- content verified). Verdict:
**cross-BUILD near-tie re-rolls are a real, known class** (the canonical
177.5->160.2 re-roll documented it); cross-build/cross-day emitted-TEXT
comparisons on tie-heavy payloads are NOT a valid identity gate. Same-binary
legs only -- which is what the A/B discipline always said. Hardening:
/mnt/ai/models/qwen36-27b-mtp/CHECKSUMS.md5 now records the model file md5s.
Corollary: 07-08's cctx numbers (222.6 etc.) and 07-09's are different BASINS
of the same payload; each is internally valid, they are not comparable to
each other. The three-stack shootout ordering is unaffected (each engine ran
its own greedy basin; the structural findings -- APC-dead-on-hybrids, byte
economics -- do not rest on basin luck).

**Live T8 with the 4..7 ladder** (base qwen36, 30 reqs, ~1.95K gated rounds):
level 7 SUSTAINED on merit where chosen -- md7 on 26% of rounds, lane-7 live
fired .61 (bar .45) / yield .596 (bar .35), 15 promotes / 13 demotes, weighted
decode 175.7 t/s (parity with the ladder-6 trial's 175.4 on a different
basin). Trial score 0.262 = a 92s low basin (cross-build tie lottery again;
scores are basin samples, telemetry is the signal).

**The decisive same-build A/B (cctx2 = fresh transcript replay, HIGH-sat:
y1..y7 = .98/.93/.89/.89/.88/.80/.81):**

    auto 4..6   210.8 t/s  <- best (revalidated after the default revert)
    fixed d6    209.8
    auto 4..7   204.3      (-2.6% vs fixed-6)
    fixed d7    196.3      (-6.4% vs fixed-6; tok/round +4.4% but ms/round
                            25.96 -> 28.98 = +3.0 ms)

**Depth-7 as built does NOT pay even in its best regime**: the width-8 round
costs +3.0 ms over width-7 -- ~2x the decayed-increment extrapolation -- and
+0.24 tok/round cannot cover it. The fired/yield bars are blind to this
failure mode (the lane is PRODUCTIVE; the machinery is what's expensive).
DECISION: **auto default reverts to the 4..6 ladder** (validated above);
`Q27_MAXD=auto7` opts into 4..7 and fixed `Q27_MAXD=7` stays, for the retune
after the follow-on: **attribute the width-8 cost** (ncu on the 8-lane verify
-- occupancy/L2/graph cliff?) -- if it shrinks to the extrapolated ~1.5 ms,
depth-7 re-enters via the same A/B. Also fixed en route: hi7/flo7 env parses
(Q27_MAXD_HI7/FLO7) were never wired -- the capped-auto A/B leg silently ran
uncapped; caught by y7 appearing in its telemetry.

Do-not-retry without new facts: ladder-7 by default (this A/B), d8 (strictly
worse cost curve until width-8 is attributed).

## 2026-07-09 (width-8 attribution) -- SOLVED: one real cliff fixed (+2.3% on d7), the rest is structural lane-amortization exhaustion

The maxd7 follow-on. Micro rig tools/width_bench.cu (gemv q4/q8 at nb=5..8 on
real shapes, L2-rotated; fd2 fp8 at ntok=5..8, verify-shaped positions,
28.7K + 61K). The +3.0 ms width-8 marginal decomposed exactly:

1. **q8_n register cliff at nb=8 -- REAL, FIXED.** 94 regs -> 2 CTAs/SM (vs 3
   at nb<=7); the v1.4 Q8 residual writers are mid-size and latency-sensitive
   (the giant head masked it in micros). `__launch_bounds__(256, N<=5?4:3)`
   pins 3 CTAs through nb=8 (80 regs + 40B stack spill). q4_n got the same
   treatment (68->64 regs, 4 CTAs at nb=8) -- neutral there, kept for headroom.
   Register allocation only; canonical 4c4120c7 EXACT (ungated/d7/auto),
   test_kernels ALL PASS. Same-build cctx2: **fixed-d7 196.3 -> 200.9 t/s
   (+2.3%), width-8 marginal 3.02 -> 2.33 ms**; d6 209.7 / auto 209.9 neutral.
2. **q4_n at nb=8: NOT occupancy.** ncu (Gabe-run, sudo): L1 hit 95.7/95.6,
   occupancy 62/61%, no-eligible 75% BOTH widths -- but DRAM demand DROPS
   998 -> 754 GB/s at nb=8. The kernel is latency-bound with the weight
   stream no longer binding: each warp's serial per-lane work (+14% at 8/7)
   converts ~1:1 into runtime (+17%/call = ~1.5 ms/round over ~400 calls).
   **The batched-GEMV lane-amortization curve is EXHAUSTED past ~7 lanes** --
   the marginal lane costs a full lane of latency, not a shared-stream ride.
3. **fd2 +15-17%/lane at 7->8 = +0.8 ms/round** (per-lane KV re-read;
   the old decaying-increment extrapolation was a width<=5 artifact).

Verdict unchanged: d7 stays OPT-IN (auto7), auto stays 4..6 (-4.2% for fixed
d7 vs d6 on cctx2 even after the fix). **The named lever for d7+ (and for
every gated width): lane-split GEMV** -- 2 warps/row x N/2 lanes halves the
per-warp latency chain at the cost of L2-absorbed double weight reads (DRAM
headroom exists: 43-57%). Medium kernel rework; pairs with the deferred fd3
lane-pair fusion as the "deep-width machinery" pair. Re-run the d7 A/B when
either lands. Rig committed: tools/width_bench.cu (+ Makefile target).

## 2026-07-09 (lane-split GEMV) -- NEGATIVE: warp-pair lane split is neutral-to-worse; the width-8 GEMV residual is structural

The named lever from the width-8 attribution, built and measured same-day.
k_gemv_q4_n2/q8_n2<N>: warp PAIR per row in one block (4 rows x 2 warps),
lanes split [0,NH)/[NH,N), per-lane chunk order and fp accumulation identical
to the single-warp kernel (bitwise by construction), dispatch nb>=6 behind
Q27_GEMV_SPLIT. Micro (L2-rotated q4 ffn): nb=6/7 neutral, **nb=8 WORSE
(0.0437 vs 0.0411 ms, +6%)**. Mechanism: each warp of the pair still reads
the row's ENTIRE weight stream, so per-warp weight-load instructions DOUBLE
-- exactly cancelling the halved per-lane chains; the L1-twin-hit hope buys
bytes, not issue slots, and this kernel stalls on issue/latency, not bytes.
Code REVERTED (not kept as dead opt-in); this entry is the record.

Do-not-retry without a design that shares weight LOADS (not just weight
bytes) across the pair: chunk-split breaks the fp summation tree (bitwise
-> tolerance-gate territory), smem staging already measured -4% (verify-gemv
Task 2 era). The surviving width-8 economics: +2.33 ms/round marginal
(post launch-bounds), structural in equal parts GEMV lane-loop and fd2
per-lane KV. Depth-7 stays opt-in; the ladder 4..6 remains the optimum.

## 2026-07-09 (external review 623cdb1..f2dddbc) -- 2 P0 + 4 P1 all verified and fixed; DepthCtl lifetime made a deliberate, measured choice

External review of the maxd6/maxd7/accept-gate range landed 6 findings (2 P0,
4 P1) + notables. Every finding verified against the code before fixing; all
confirmed (one sub-claim relocated: string-value \uXXXX was already strict --
the incomplete-escape hole was the KEY-string escape path).

**P0 #1 -- ctx admission was depth-5-era at every guard.** The maxd6/7
widenings never re-audited the context guards: generate()'s loop stopped at
`P+7 > max_ctx`, the CLI spec loop likewise, and all six server n_max clamps
reserved `-6` -- but a full-width round at gate_maxd 6/7 writes attention-KV
rows through P+8 (MTP rows through P+7), overrunning the caches by up to 2
rows near the ceiling (silent corruption the prefix cache could then reuse).
All seven sites now derive from one accessor, `Engine::ctx_round_reserve()
= gate_maxd + 2`. Boundary-tested at --ctx 32 / Q27_MAXD=7 under
compute-sanitizer (fp16 + fp8 KV): guard stops at P=24 (24+9>32), 0 errors.

**P0 #2 -- direct CLI had no prompt/generation bounds.** Ingestion stepped
an arbitrary tokens-file into caches sized --ctx (the "8K tokens-file dies
at default ctx" gotcha was this bug wearing a stats costume), and the final
d_gen copy read `prompt+n_gen` ints from a max_ctx-sized buffer. Now: refuse
prompt > ctx, clamp n_gen, refuse --nll-chunk > ctx (nll-long already
clamped).

**P1 #3 -- mask cache could cross tool allowlists.** ToolGrammar::signature()
(the server-global mask-cache key) omitted names_, so requests with different
tool sets collided at equal grammar states -- B could be steered into A's
tools. Signature now appends a sorted allowlist key. Regression R1.

**P1 #4 -- top-p broke at high temperature.** The cutoff bisection searched
raw logits [M-40, M] while mass is inv_temp-scaled: at T=10 the window spans
4 scaled nats and the threshold pinned at M-40 (a two-tier test distribution
kept 44% for top_p=0.95). Window is now 40*max(1,T) raw = >=40 scaled nats;
T<=1 bitwise-unchanged. GPU test 3b: thresh -61.3, mass 0.9502.

**P1 #5 -- ChatML stripping missed the OpenAI path + tools.** apply_chat_template
concatenated raw roles/content (server OpenAI chat bypassed api_common's
strip_ctrl entirely); tools_preamble dumped declarations verbatim. Both
boundaries now strip <|im_start|>/<|im_end|>; forged messages tokenize
identically to their pre-stripped equivalents (test).

**P1 #6 -- ToolGrammar accepted JSON that json::parse rejects.** Single J_NUM
state took 1..2e+-3 / 01 / 1e5e5; ',' before '}' / ']' fell into the
empty-object/array cases (trailing commas); J_KEYESC took '\u' in one hop
(\uZZ in keys). Full JSON number FSM (8 states, leading-zero rule), _REQ
states after commas, 4-hex key-escape states. Regression R2 (25 cases).

**Notables:**
- **DepthCtl lifetime**: reviewer flagged docs-vs-behavior mismatch
  (engine-lifetime state described as per-stream). Built reset() + hooked it
  per-request, then MEASURED it: cctx auto 207.9 vs d5 211.3 (-1.6%) -- a
  50-round request spends its life re-earning depth; docs61k neutral.
  Decision: carry stays the DEFAULT (single-user rig, multi-tenant out of
  scope per SECURITY-MODEL.md, consecutive requests are turns of one
  conversation); Q27_MAXD_RESET=1 opts into per-request isolation.
  reset() + 6 CPU checks stay.
- **Slot admission floor** derived from a depth-5-era "5 GDN sets ~3GB"
  constant -> engines report exact gdn_state_bytes, admission uses it.
- **S_spare6/7 always allocated** (~157MB each at any gate_maxd): accepted
  and documented -- perm rotation is uniformly mod-8, so all 8 sets enter
  rotation even at shallow ceilings; a width-dependent modulus would
  complicate the P15-hardened refinish machinery for pure VRAM savings.
- **BUILDLOG 2026-07-08 "+4.7%" corrected**: 222.6 was warm-server vs a CLI
  d5 leg (cross-harness). Same-harness rerun on the post-review binary
  (server replay, fresh server per leg, 1 cold + 3 warm medians, Q27_KV=fp8
  Q27_PMIN=0.5 Q27_DEXIT=1): cctx @25.8K d5 211.9 vs auto 220.7 = **+4.2%**;
  docs61k @61K d5 111.5 vs auto 109.9 (-1.4%, known bursty-flavor noise;
  round-grouping flickers across replays under carry -- emitted tokens
  identical). No 100K payload exists; rerun scope is 26K + 61K.
  README's three +4.7% sites corrected to the same-harness numbers.

**Gates:** canonical 4c4120c7 EXACT at MAXD 4/5/6/7/auto/auto7 (+ auto/auto7
with Q27_MAXD_RESET=1); base-model canonical a2982c51 EXACT; test_kernels
ALL PASS (incl. new high-T nucleus); test_depthctl 43 PASS; test_toolconstrain
ALL PASS (incl. R1/R2); test_tokenizer suite PASS (incl. both-boundary
sanitize); compute-sanitizer clean on the ctx-boundary rig (fp16 + fp8 KV).

Out of this pass: Makefile wiring for test_toolconstrain (pending approval --
build/run documented in the test header meanwhile).

## 2026-07-09 (baseline switch) -- vanilla Qwen3.6-27B-MTP is the benchmark standard; fine-tunes stay supported

Policy: `qwen36-27b-mtp` (canonical a2982c51...) is now the default MODEL/TOK
and CANON_MD5 in every bench/gate rig (shortbench_suite, accept_ab,
constrain_gate, sampling_gate, reqlog_gate, interleave_gate); fine-tunes ride
the same rigs via MODEL/TOK/CANON_MD5 env overrides (Qwopus: 4c4120c7...).
Historical numbers in README/BUILDLOG were measured on Qwopus unless noted.

Baseline numbers on the standard model (post-review binary, all gates green
on the new defaults):
- shortbench suite 161.8 t/s (canonical leg 131.5, 2.61 tok/round);
  sampling gate ALL PASS; constrain gate ALL PASS (canonical a2982c51 EXACT,
  test_kernels PASS).
- cctx same-harness server replay @25.8K (Q27_KV=fp8 Q27_PMIN=0.5
  Q27_DEXIT=1): d5 161.2 vs auto 163.5 = +1.4%. The base model saturates
  less than Qwopus (y5 .611/fired5 .49 vs .81/.84), so the ladder promotes
  less and pays less -- the ~8% acceptance gap vs the fine-tune shows up as
  both lower absolute t/s and a smaller auto edge.

Also corrected here: the review-fixes entry claimed 46 DepthCtl checks; the
suite has 43 (external re-count caught it).

## 2026-07-09 (review follow-up) -- 4 residual findings fixed; DepthCtl gets lineage-aware reset

Reviewer re-audit of the review-fixes range surfaced 5 more (1 High, 3
Medium, 1 Low). All verified and fixed; the Low (Makefile header deps) plus
the earlier check-count slip (46 -> 43, fixed in the baseline entry) round
it out.

1. HIGH, CLI diagnostic budgets: --burst-stats/--stats/--pfdbg bypassed the
   P0 #2 prompt+n_gen bound and still overran KV/MTP-KV/d_gen (burst probes
   chain 10 positions past the step). Per-mode refusals incl. lookahead;
   all three verified firing at --ctx 256.
2. MEDIUM, mask-pool exhaustion: the P1 #3 allowlist key went on EVERY
   state signature (and stale name_pref_ came along), duplicating identical
   argument-state masks per tool set/name until the 512-entry pool filled
   -> constraint silently off. Legality can only depend on names up to
   NAME_VAL, so the key is now name-phase-only + ToolMaskCache dedupes by
   bitset content. R3 regression: pool does not grow for a second tool set.
3. MEDIUM, empty-200 near the limit: preflights now share one
   reserve-aware max_prompt across all three routes, and claim_slot skips
   zero-budget slots. Live-verified at --ctx 512 (400s at 504+).
4. MEDIUM, fp8 dispatch: gate keys on k_arch_probe (__CUDA_ARCH__ of the
   LOADED image) instead of the device attribute -- an Ada card running the
   sm_86 image can no longer enable the mma_e4m3 no-op stub. 5090
   unaffected (test_kernels fp8q PASS).
5. DepthCtl lifecycle (reviewer suggestion adopted): claim_slot resets the
   ladder when a non-prefix-restoring request takes the slot over; warm
   carry preserved within a conversation lineage. Q27_MAXD_RESET=1 stays as
   the strict knob.

Gates: canonical a2982c51 (base, default) + 4c4120c7 (Qwopus, auto7 leg)
both EXACT; test_kernels/test_depthctl/test_toolconstrain/test_tokenizer
ALL PASS; F3 server smoke live (400/400/normal). Makefile dep wiring
(finding 5) pending approval.

## 2026-07-09 (comprehensive baseline campaign) -- vanilla Qwen3.6-27B-MTP reference numbers, master 197d6b6

Full campaign on the new benchmark standard. Production config throughout
(Q27_KV=fp8, Q27_PMIN=0.5, Q27_DEXIT=1, --fast-head); server-replay legs are
fresh-server, 1 cold + 3 warm, medians (tools/accept_ab.sh); ran under
systemd-run (unit qbench-2026-07-09), logs /tmp/qbench.

**Decode envelope @~26K (t/s | tok/round):**

    payload   d4            d5            auto
    echo      172.4 | 3.88  188.9 | 4.41  184.5 | 4.49
    docs      171.8 | 3.71  179.0 | 4.06  173.0 | 3.94
    codegen   177.2 | 3.94  182.1 | 4.27  182.8 | 4.27
    testgen   160.1 | 3.56  159.0 | 3.61  160.2 | 3.56 (never promotes)

d5 >= d4 on every flavor (echo +9.6%) -- the post-verify-gemv inversion holds
on the base model; auto tracks the winner and correctly refuses to promote on
testgen (fired5 = 0).

**cctx (real-CC flavor, 25.8K): d4 157.0 | d5 159.9 | d6 165.7 | auto 162.1
| auto7 162.5.** Base saturation (y5 .61, fired5 .38-.49) sits UNDER the hi6
promote bar much of the time, so auto is conservative and fixed d6 leads it
by +2.2% on this flavor -- but fixed d6 would tax testgen-class traffic
(y5 .36), so auto stays the production rec. auto7 == auto (level 7 never
sustained).

**docs61k @61K: d4 133.1 | d5 118.1 | auto 144.7 -- BASIN-CONFOUNDED, do not
read as config deltas.** d5 took 82 rounds vs d4's 74 for the same 256
tokens, impossible under width-invariance on a shared trajectory: the
tie-heavy payload forks into different greedy basins per config (each leg
internally deterministic, det=OK). Cross-config comparison at this depth
needs token-file CLI replay, not text-completion replay.

**Prefill (fp8 batched TTFT): 8K 2.35s (3491 t/s) | 32K 10.39s (3155 t/s) |
128K 59.4s (2206 t/s)** -- 128K matches the Qwopus-era 59.6s (same weights
shape; the O(N^2) attention share grows with depth as known).

**Sampling tax @8K: greedy 137.7 t/s (2.61 tok/rnd) vs T=0.7/top-p 0.95
129.3 (2.59) = -6.1%**, acceptance preserved through sampled verify.

**Fine-tune delta, same binary + payload (cctx): Qwopus d5 210.0 / auto
219.0 (+4.3%) vs base d5 159.9 / auto 162.1 -> +35% at auto, pure
acceptance (5.82 vs 3.56 tok/round; y5 .81 vs .61, fired5 .79 vs .38).**
Same-day rerun drift vs the morning legs (211.9/220.7): ~1%.

Shortbench suite (current binary): 161.1 (canonical leg 131, a2982c51
EXACT). Not run: 100K+ payloads (none exist), concurrency (out of scope for
the single-stream engine).

## 2026-07-09 (thunderdome validation) -- post-review binary + base model: no agentic regression

Two CC tasks via claude-code-q27-haight (base qwen36 on :8081, production
config, 131K/32K slots), refs = 2026-07-08 pre-review trials:

- T2 collab-server: 0.851 @138s (ref 0.851 @136s -- identical).
- T8 analytics-dashboard: trial 1 landed the documented bad auth-chain
  basin (0.564 @612s, hidden 0.219 / agent 1.000 / metrics 1.000, 134
  turns); retrial 0.846 @155s, hidden 0.938 (ref 0.842). The bimodality is
  the known eval artifact, not engine quality; two same-config trials
  bracketing both basins is the expected signature.

Live ladder on real CC traffic (base model): 66% of gated rounds at depth 6
(md6 2404 / md5 1008 / md4 239, 15 promotes / 13 demotes) -- live traffic
saturates far above the cctx replay's fired5 ~.38, re-confirming that only
real agent transcripts reach the deep regime. Serving-path changes under
test: ChatML both-boundary strip, ctx preflights, lineage DepthCtl reset,
strict tool grammar, mask-cache keying.

## 2026-07-09 (thunderdome T1-T10) -- base model 0.834 mean over T1-T9; T10 surfaced drift mode 8 (fixed)

Full T1-T10, claude-code-q27-haight, base qwen36, one trial each (T8/T10
retrialed), post-review binary:

    T1  time-tracker        0.841 @51s    hidden 1.000
    T2  collab-server       0.851 @138s   hidden 1.000
    T3  fts-search          0.760 @89s
    T4  phantom-invoice     0.833 @23s
    T5  task-queue marathon 0.789 @155s   hidden 0.930
    T6  monorepo-disaster   0.850 @42s
    T7  plugin-marketplace  0.890 @91s    hidden 1.000
    T8  analytics-dashboard 0.846 @155s   (good basin; trial 1 bad auth-basin
                                           0.564 @612s -- known artifact)
    T9  ssg-toolkit         0.850 @97s
    T10 ecommerce-backend   crash-class x2 (0.22 @2-4s) -> DRIFT MODE 8

Mean over scored T1-T9: 0.834. T10 was a stable greedy one-shot-quit: the
first turn batches four well-formed Read calls as {"function": "Read",
"arguments": {...}} (string-valued alias key) behind a dangling {"name":
line -- no rescue mode matched, all calls dropped, CC quit (metrics 0.9:
the model was fine). Fixed as drift mode 8 (resolve_aliased_call,
registered-name-validated; dfd3b12); live T10 revalidation pending an
eval-server restart onto the new binary.

Live throughput across the whole campaign (344 requests): decode 168.0 t/s
aggregate (179K tokens), 5.29 tok/round, prefix cache serving 95% of prompt
tokens; per-request median 166 t/s, p75 186, peak 254.

## 2026-07-09 (three-engine thunderdome) -- same model family, same harness: quality converges, prefix-cache architecture decides wall

T1-T10, claude-code-*-haight adapters (byte-identical modulo endpoint),
greedy + no-think all legs, n=1/task, same day, same binary/build per leg:
q27 base repack 5.25 bpw | llama.cpp mainline 1491, Q4_K_M 4.98 bpw,
draft-mtp 10/0.5, q8 KV | vLLM nightly 0.23.1rc1.dev748, PrismaSCOUT NVFP4
~5.31 bpw, MTP k=3, fp8 KV, native /v1/messages (anthropic router).

    task                q27            llama.cpp      vLLM
    T1 time-tracker     0.841 @51s     0.843 @57s     0.400 @696s*
    T2 collab-server    0.851 @138s    0.802 @131s    0.870 @593s
    T3 fts-search       0.760 @89s     0.970 @104s    0.710 @637s
    T4 phantom-invoice  0.833 @23s     0.833 @29s     0.833 @92s
    T5 task-queue       0.789 @155s    0.799 @125s    0.797 @511s
    T6 monorepo         0.850 @42s     0.850 @56s     0.850 @124s
    T7 plugin-mkt       0.890 @91s     0.902 @84s     0.903 @485s
    T8 analytics        0.846 retrial  0.531/0.550    0.549 (bad basin)
       (bimodal)        (t1 bad .564)  (07-08 good .818)
    T9 ssg-toolkit      0.850 @97s     0.850 @114s    0.850 @107s
    T10 ecommerce       parser-quit    0.518 @50s     0.520 @124s
                        (mode 8 fixed)
    mean T1-7+T9        0.833          0.856          0.777
    wall  T1-7+T9       686s           700s           3245s (4.7x)

(*coverage-file crash-class; scored work included. T8 excluded from means:
the auth-chain basin lottery is engine-independent -- every engine drew the
bad basin at least once today. T10 excluded: q27's pre-mode-8 parser quit;
llama/vLLM both scored ~0.52 partials.)

Read: with the tolerant-parser class equalized, SCORES converge to the
model (T4/T6/T9 three-way identical; T2/T7 within noise; T3 a llama
outlier win, T1 a vLLM outlier loss). WALL separates on serving
architecture: q27 and llama both reuse conversation KV across turns (q27
95% prompt-token cache-hit measured live); vLLM cannot prefix-cache
hybrid-GDN models (0% reuse, known) and re-prefills every turn -- 4.7x
wall despite competitive decode (avg generation ~92 t/s reported;
replay-controlled decode remains q27 162 vs vLLM+MTP 155.5 vs llama 98.4
on cctx @25.8K).

Live decode aggregates over each leg's own trajectories
(trajectory-confounded, NOT controlled): q27 168.0 t/s (5.29 tok/round,
344 reqs) | llama 180.1 t/s (median 182/req, 347 reqs). Live traffic
feeds llama's n_max-10 chains far better than the cctx replay predicted
-- same live-vs-replay saturation effect as q27's ladder; the controlled
same-payload replay stays the rate comparison of record.

Leg gotchas fixed en route (5090-local-llm 82e43fa): prismascout keys had
the hermes tool parser (qwen3.6 emits qwen3_coder XML function calls ->
returned as TEXT -> CC one-shot-quit in 6s), 32K max-len, and no
no-think/greedy defaults; vLLM nightly's --default-chat-template-kwargs +
--override-generation-config close the parity gap server-side. New
thunderdome adapter claude-code-vllm-haight (approved) mirrors the other
legs byte-for-byte.

## 2026-07-09 (q27 on the RTX 3090, sm_86) -- same scores, 7-12x wall; 32K is the practical ctx ceiling; arch-probe fix validated on real Ada-less silicon

Base model, production config, GPU 1 (vox transcribers unloaded for the run).

**Canonical forks ACROSS ARCHITECTURES (fp16 KV): 3090 = 6894254e...** vs
5090 a2982c51 -- internally width-invariant (ungated == gated == auto), so
the bitwise contract holds PER-ARCH; sm_86 codegen/ULP differences fork the
tie-heavy trajectory exactly like cross-build lotteries. (The fp8-KV leg
happens to land on the 5090 md5 -- coincidence of this trajectory, not a
guarantee.) shortbench_suite gained BENCH_GPU= and the 3090 gates on its
own canonical.

**Throughput:** shortbench suite 87.9 t/s (55% of the 5090's 161.1, on 52%
of the bandwidth; per-prompt tok/round identical -- pure hardware scaling).
cctx replay d5 58.2 / auto 59.7 (36% of 5090; trajectory basin-forked, own
numbers). Canonical-fixture decode 84.5 vs 131.5.

**Serving envelope:** --ctx 65536 OOMs at CUDA-graph instantiation (model
17.7GB + graph set + 2.2GB KV > 24GB); --ctx 32768 serves at 23.1GB.
[pfattn] logged "loaded image sm_86 < sm_89: fp8-MMA prefill unavailable"
-- the review follow-up #4 arch-probe gate doing its job on real hardware.

**Trials (claude-code-q27-haight -> :8081 on GPU 1):**
    T4 phantom-invoice  0.83 @164s   (5090: 0.833 @23s)
    T6 monorepo         0.85 @496s   (5090: 0.850 @42s)
    T2 collab-server    NOT VIABLE at 32K: conversation outgrows ctx; the
    server 400s correctly (5x ctx-limit) but CC does NOT compact against a
    32K window (its compaction assumptions target Anthropic-sized windows,
    the 07-05 ops-note risk realized) -- terminal "Prompt is too long".

Read: scores are IDENTICAL to the 5090 -- quality is the model + parser,
fully preserved on sm_86; wall is 7-12x (bandwidth + no fp8-MMA prefill +
f16 fallbacks). The 3090 is a functional overflow/test device for small-ctx
agentic work, not a serving peer. CC-vs-small-window compaction is the real
limiter for bigger tasks, not the engine.

## 2026-07-09 (llama.cpp on the 3090) -- decode TIES q27 on sm_86; both engines cap at 32K; CC-vs-32K fragility is engine-independent

Mirror of the q27-3090 leg: mainline 1491, base Q4_K_M, draft-mtp 10/0.5,
q8 KV, GPU 1. -c 131072 and 65536 both fail context creation on 24GB (then
segv on llama's own error path); -c 32768 serves at 23.85GB -- MORE than
q27's 23.1GB at the same ctx (fp8 KV + graphs beats q8 KV on footprint).

**cctx replay @25.8K, warm medians: llama 60.4 t/s vs q27 58.2 (d5) /
59.7 (auto) -- a TIE.** The 5090's 1.65x q27 lead on this payload
evaporates on sm_86: no fp8-MMA, f16 fallbacks, and wider verify rounds
cost more per ms, so both engines sit on the same bandwidth wall
(basin-forked trajectories; +/-5% is noise here).

**Trials at the 32K ceiling:** llama T4 crashed @37s (0.65 partial, CC
exit after 17 turns at ~30.6K in) and T6 crashed @40s (0.81 partial, 36
turns, ~30.4K) -- both died at conversation ~= ctx - margin, where llama's
context error is a shape CC treats as terminal. q27 completed T4 0.83 /
T6 0.85 at the same window because its trajectories stayed under 32K --
its T2 died identically when the conversation outgrew the window. Verdict:
at 32K windows CC agentic work is trajectory-lottery-gated on EVERY
engine; the 3090 is a small-task/overflow device, and CC's
compaction-vs-small-window mismatch (07-05 ops note) is the binding
constraint, not either engine.

3090 box score (base model, same day): decode ~58-60 t/s both engines;
q27 completes 2/3 trials at identical-to-5090 scores; llama 0/2 at this
window. vox transcribers were stopped for the runs.

## 2026-07-09 (levers survey, post-DFlash-kill) -- deep-chain data flips the decode roadmap: GEMM-verify deep ladder is the new #1

**Keystone measurement (rig: --burst-stats SD=10 on cctx, 1488 positions,
scored vs the .seq stream):** the trained-in MTP chain's CONDITIONAL
acceptance is FLAT to depth 10 -- p(dk | prefix ok) = .845/.790/.765/.775/
.800/.775/.781/.796/.833/.862 for k=1..10. It does not decay; it RISES in
deep echo regions. Depth was never acceptance-limited -- only our GEMV
verify machinery capped it (width-8 lane exhaustion, +2.33ms marginal).

**Lever A -- GEMM-verify deep ladder (NEW #1).** Replace GEMV lanes with a
dedicated batched T<=16 GEMM verify (prefill-class kernels shaped for
small T) and raise the ladder ceiling to 10+. Gated round sim on the
burst data (theta .5, sim-vs-sim so the known -27% cctx bias cancels):

    GEMV c7 vs c5:      +1%   (matches the measured d7 wash -- sim sane)
    GEMM c10 @13ms:    +46%
    GEMM c10 @15ms:    +31%
    GEMM c10 @18ms:    +13%
    GEMM c10 @22ms:     -4%   (breakeven ~21ms)

Everything rides on ONE number: dedicated small-T verify cycle cost.
P0b showed the current path's 47ms is kernel shape, flat in depth;
floor math says 12-16ms is plausible (10ms weight floor + fd2-class
16-row attention ~4ms + head ~1ms). External proof of mechanism: llama's
batch verify at chain depth 10 does 180 t/s LIVE on this model (vs our
168) with far weaker per-token kernels. NEXT STEP: 1-day spike -- extend
width_bench with T=8/16 GEMM sweeps over the layer shapes + fd2 ntok=16
to bound the cycle; build only if <=~16ms. Also: fixed-cost verify wants
a different gate (fire deep at moderate confidence) -- more upside than
the sim shows, which reuses the GEMV-era theta gate.

**Ranked levers behind it:**
- B. Prefill Phase-3 / FA2-class smem relayout (standing, Gabe-gated):
  externally validated by FlashRT (~2900 t/s @256K vs our 2206 @128K).
  +6% LIVE wall (prefix cache already eats 95% of prompts) but +16% on
  cold long-context / TTFT. 2-4 sessions, tolerance-class precedent.
- C. fd3-mma tensor-core decode attention (filed): kills fd2's
  +15-17%/lane KV re-read growth and the 26K->61K decode sag; COMPOUNDS
  with A (deep verify wants cheap 16-row attention -- the fd2 ntok=16
  number from the A-spike doubles as C's motivation data). Untested MMA
  thesis; the two fd3 kills were smem-sharing, not MMA.
- D. Small-window CC compaction shim (new, from the 3090 runs): find why
  CC compacts at 131K but terminates at 32K (request-tap forensics),
  serve the shape that triggers compaction. <=1 session, unlocks
  3090/32K agentic serving. Robustness, not throughput.
- E. Relaxed think-phase acceptance (FlashRT +43% think-heavy):
  tolerance-class, ~1 session, margins already computed -- but we serve
  no-think; park until a thinking workload exists.
- F. Constrain-tools bare-JSON engage (quality): closes the grammar
  bypass (T8 hidden 0.94->0.22 case), makes strict+constrain the
  zero-rescue config. ~1 session.
- G. KV pruning (quality-class), H. ladder churn polish (~1-3% bursty
  flavors): behind A-C.

Sequencing: A-spike first (1 day, pure measurement); if GO, A build
(3-5 sessions, P15-class GDN-state risk at verify widths >8 -- the
mod-8 role-buffer machinery caps at 8 lanes, so >8 needs chunked-scan
GDN with checkpoint/replay, the DFlash Phase-2 problem inherited). B
next for TTFT. C rides A's spike data.

## 2026-07-09 (GEMM-verify spike) -- NO-GO at the <=16ms bar; the lever collapses into the tensor-core pivot

Spike protocol: Q27_P0B_T width sweep + nsys per-kernel attribution
(tiny-prompt run isolates pure T=16 instances; 103 chunks).

**T-shape sweep @26K (chunk+head ms):** T=8 45.8 | 16 47.1 | 32 49.2 |
64 54.0 | 128 68.5 | 256 107.2 => cost = ~44ms FIXED + 0.25ms/token.

**Attribution of the fixed 44ms (per T=16 cycle):** q4 weight GEMMs
(k_gemm_mma_T<1,1>, 304 calls) 28.6ms + q8 GEMMs 5.9ms + attention
(k_attn_prefill_mma_fp8q, 16x200us) 3.2ms + GDN scan 1.5ms + misc ~2.5ms.
The weight GEMMs stream at 29% of DRAM BW at M=16 (34.5ms vs the 9.9ms
whole-model floor) -- the T=1024-era tiling collapses at small M (grid
over rows only; too few CTAs to saturate DRAM).

**Verdict vs the sim:** win zone (13-15ms cycle, +31-46%) needs ~85% BW
at M=16; our best demonstrated small-M class (tuned GEMV) peaks 50-55%,
which lands the realistic dedicated-path cycle at 20-26ms = the
breakeven zone (sim: -4% at 22ms, +13% at 18ms). NOT worth 3-5 sessions
+ P15-class GDN state work for a breakeven-to-+13% range.

**What would reopen it:** a tensor-core weight-GEMM path (in-kernel
dequant q4->fp8/bf16 + MMA, CUTLASS-class) that demonstrates >=70% BW at
M=16 -- i.e., the already-assessed NVFP4/tensor-core pivot. The
deep-ladder lever is not independent of that pivot; it is one of its
payoffs. The chain-acceptance finding STANDS (flat 0.78-0.86 to d10) and
makes the pivot's decode case materially stronger than when we assessed
it (then: "batch-1 bandwidth-bound, not recommended"; now: batch-16
verify with measured-flat acceptance is the workload the pivot serves).

Levers roadmap after the spike: B (prefill FA2 relayout) resumes as #1
actionable; the tensor-core pivot gets a strengthened case file (chain
data + 29%-BW finding + FlashRT/CUTLASS reference); C/D unchanged.
Rig kept: Q27_P0B_T sweep param.

## 2026-07-09 (prefill FA2 relayout, Phase 3a) -- KILLED same-day: occupancy doubled to 25%, TTFT -1%; the kernel is barrier-serialized, not occupancy-bound

Built the plan's 3a shape (docs/plans/2026-07-09-prefill-fa2-relayout.md):
384-thread CTA, warp-pair d-split (each warp owns 128 of the head's 256 O
dims, redundant full QK^T per pair), 144 regs zero-spill vs fp8q's 254,
LOGITS BITWISE IDENTICAL (exact-math transform -- the d-split technique is
validated and reusable). ncu: achieved occupancy 24.98%, 11.99 warps/SM --
the designed doubling, delivered exactly. TTFT 128K: 59.7 -> 60.4s.

The tell: Issued/scheduler 0.26, No Eligible 73.9% -- unchanged with 2x
warps. Warp pairs are lockstep clones stalling on the same shared chains at
the same cycles; per-tile __syncthreads serializes every warp through the
same pipeline phases. Phase-0's "occupancy-bound" read is RETIRED: the
binding constraint is the synchronous tile pipeline. FlashRT's FA2 wins via
async structure (software-pipelined tiles, warp-specialized
producer/consumer, no full-CTA barriers on the hot path) -- that full
rewrite (2-4 sessions) is what Phase 3 actually costs, with a sharper
target metric now: Eligible Warps/scheduler (0.44 today), not occupancy.

Kernel + launcher reverted per kill protocol (canonical a2982c51 EXACT
post-revert); plan doc carries the verdict + do-not-retry (register cuts,
CTA repackaging, split-kk all preserve the binding barrier structure).

## 2026-07-10 -- suffix drafter LIVE-CC trial (GO for CC serving) + GDN chunk P0 (STOP -> attention re-attribution)

Suffix trial, T8 x claude-code-q27-haight, qwopus, production env
(Q27_KV=fp8 PMIN=0.5 MAXD=auto) +- Q27_SUFFIX=1, fresh server per leg:

- ON leg: score 0.815 @ 167s (the known matched basin), 53 reqs,
  22,057 decode tok, 4,006 rounds. **sfx = 1,964 rounds / 12,734 tok =
  6.48 tok/round fired, 57.7% of ALL decode via suffix rounds, 49% of
  rounds.** MTP rounds 4.57 tok/rnd (composition: suffix takes the echo
  cream). Ladder NOT starved live: 81% of auto rounds at ceiling 6
  (md6=2954, mprom=11) -- the repro-payload starvation was small-sample.
- OFF leg: score 0.834, hidden_tests identical (0.906 both), 100% pass
  both. 54 reqs / 16,324 tok / 5.00 tok/rnd blended.
- PAIRING IS IMPOSSIBLE cross-run: legs forked at request 5 -- same conv,
  same prompt SIZE (33,188), different bytes (tool output carries
  wall-clock-dependent content; pytest timings etc). Same class as the
  cross-build tie-lottery: within-run telemetry is the only currency for
  CC trials. Byte-identity of the engine itself is construction-level +
  5-payload gated; the fork is environmental.
- Within-leg value estimate: rounds saved vs MTP-at-saturating-rate on
  the same stretches (~158 rounds, ~4%) + zero draft cost on 49% of
  rounds (~4-5% of decode wall) ~= 6-9% decode-wall win on this
  trajectory, CAPPED at width 7 (fired AL 6.48 = ceiling; offline sim
  says the same fires reach AL 10-12 uncapped). Deep verify multiplies
  this. RECOMMENDATION: Q27_SUFFIX=1 in the CC serving env (binary
  default stays off); revisit default-on after the width work.

GDN chunked-scan P0 (tools/gdn_chunk_bench.cu, 3090): STOP-rule fired.
delta chunk 1.21-1.29x (state-WRITE-bound: the 3.1MB/step role snapshot
is contractual; smem residency only removes the read); conv chunk
2.9-7.1x but ~0.05ms/lane component. Bitwise identity on every leg
(identical per-step arithmetic + reduction order) -- numerics approach
validated for future GDN work. WIDTH-COST RE-ATTRIBUTION: ~1.4ms/lane =
attention ~0.6 (fd2 re-reads full KV per lane; 16 layers x ~53MB fp8 @
26K ctx, grows with ctx) + delta ~0.23 + conv ~0.05 + batched per-lane
compute ~0.3-0.5. Ceiling-8+ target REVISED to a shared-KV W-query
verify attention kernel (one KV pass scores all W lanes, split-KV; the
tokenspeed-MLA/FlashRT verify shape; NOT the retired Task-6 lane-pair
fusion). Falsifiable check queued: per-lane marginal should ~double at
61K ctx (phase-width run on docs61k). Plan: docs/plans/2026-07-10-gdn-chunk.md.

## 2026-07-10 -- deep-ladder ceilings 7/8 re-priced on the fdmma flat width curve

Method: tools/ladder_price.py simulates the exact gated-round policy
(theta=0.5 margin cap, early-exit draft steps at 0.81ms, width-floor
top-up, leading-run acceptance) over REAL measured 10-deep MTP chains
(--burst-stats CSVs) and prices rounds with the MEASURED width curves.
Sim validates against live: mma-vs-fd2 gap at 61K reproduces the
measured +18.7% A/B.

Width curves (ms/round, measured): mma@26K W4..8 = 16.6/17.7/17.9/19.0/
21.8 (fd2: 18.5..25.8, W8 -15.5%); mma@61K W4..8 = 18.1/19.2/19.4/20.7/
23.3 (fd2: 22.6..32.8, W8 -29%).

Chain data, two poles:
- HOT (echo-heavy, fp16-basin cctx chains, tok/rnd 3.9-4.6): under fd2
  ceilings 5..8 are a WASH (149->151 t/s @61K -- the historical maxd6/7
  NO-GO reproduced by the sim). Under mma: c7 = +0.6% over c5, c8 =
  +4.9% (185.8 vs 177.2 @61K). Deep flips from wash to positive; the
  win concentrates at ceiling 8, with a local dip at 7 (y7-conditional
  acceptance on this traffic doesn't quite cover the W8 step; y8 does).
- COLD (fp8-basin cctx chains regenerated tonight, tok/rnd 2.6): deep
  loses monotonically on every curve (mma@26K c5 129.4 -> c8 121.0,
  -6.5%). Confirmed live: forced maxd7 on docs61k = 122.7 vs auto 142.0.
  The AUTO LADDER's sat/flo bars are the protection -- on this traffic
  it never promotes past 4/5 (measured md4=252 md5=0 at 61K), so deep
  ceilings are never engaged and cost nothing.

VERDICT:
1. auto7 under Q27_FD=mma: SAFE TO ENABLE (ladder-protected on cold,
   ~wash-to-positive on hot; W8 rounds now 21.8-23.3ms). Modest MTP-only
   upside; production rec for the live-CC trial = Q27_FD=mma
   Q27_MAXD=auto7 Q27_SUFFIX=1 -- the suffix drafter is the bigger
   width-8 consumer (live fired AL 6.48 was WIDTH-CAPPED; sim says the
   same fires reach 10.7 at K=16).
2. Ceiling 8+ (W=9..16 verify): the MTP-only case is +4.9% on hot
   traffic; the suffix case wants W 9-12 outright. fdmma solved the
   attention share (marginal lane ~1.3ms, W9 extrapolates +1.3); the
   REMAINING blocker is architecture, not perf: CP3/P3/IP3 p[8] structs,
   8-lane role-buffer rotation (perm mod 8), and the GDN serial chains
   at width>8 (the known P15-class problem). That plumbing project is
   now the gate on everything deeper than ceiling 7.
3. The maxd7 auto-ladder (ea87ccf, 7th sess) needs no code change --
   auto7 exists; the flip is an env-config decision post live-CC trial.

Chains: scratchpad/burst_cctx_fp8.csv (+docs61k_fp8 when its 61K
token-walk completes). Historical footnote: the fp16-vs-fp8 basin split
of the SAME cctx payload (echo-heavy vs cold, 35% vs 0.8% suffix fire,
3.9-4.6 vs 2.6 tok/rnd) is the sharpest tie-lottery exhibit yet -- any
acceptance-sensitive decision MUST name its basin.

Addendum (docs61k fp8 chains completed): the MID pole (3.52-3.79 tok/rnd)
also prefers shallow -- mma@61K c5 158.9 -> c8 150.2 (-5.5%). The c8 win
exists ONLY on genuinely saturating traffic (y flat >= ~.82); even
3.5-3.8 tok/rnd flavors shouldn't promote to 7. This is exactly what the
shipped hi6/flo bars enforce (docs stays ceiling 5, cold cctx 4) --
auto7's safety case rests on those bars staying strict, and all three
measured poles now validate their calibration.

## 2026-07-10 -- full-stack live-CC trials (mma + auto7 + suffix), T8 x2

Config: Q27_FD=mma Q27_MAXD=auto7 Q27_SUFFIX=1, qwopus, fp8, fresh server
per trial.

PERF (both trials, within-run telemetry): trial 1: 249.3 t/s aggregate
decode (75 reqs, 34K tok) -- +27% over the suffix-only trial (195.8),
+42% over base; suffix 7.49 tok/rnd on 63.3% of decode (width-8 uncap
from 6.48@W7); ladder promoted to ceiling 7 on 66% of auto rounds
(md7=3407) -- LIVE traffic clears the strict bars that no replay payload
ever has. Trial 2: 232.9 t/s, suffix 7.15/rnd on 48.7%, md7=1728. The
stack composes exactly as priced.

QUALITY: 0/2 good basins (0.560, 0.561; hidden_tests 0.219 both) vs 2/2
good without mma (0.815, 0.834; hidden_tests 0.906 both). agent_tests
1.000 / code_metrics 1.000 / coverage ~0.8 in ALL FOUR trials -- the
code works; the failure is the T8 auth-chain hidden gate. Forensics:
register returns 500 on the hidden test's spec payload (reproduced
locally in the archived workspace; contract itself is spec-compliant
and identical across good/bad trials -- the 500 is an internal handler/
schema design choice, likely tenant-FK-class, that the agent's own
tests don't exercise). Attribution is CLEAN by construction: suffix is
token-identical, auto7 is width-invariant -- any behavioral shift is
Q27_FD=mma's tolerance-class attention numerics steering early design
decisions into a punished fork on this task. n=2v2 on a bimodal gate
(P ~ 0.1-0.2 under no-effect), suggestive not conclusive.

VERDICT:
- Q27_SUFFIX=1 + Q27_MAXD=auto7: quality-safe BY CONSTRUCTION; their
  wins (suffix 63% coverage at 7.5/rnd; ceiling-7 residency 66%) are
  real on live traffic. GO for CC serving env alongside whatever
  attention kernel is active.
- Q27_FD=mma: perf validated (+18.7% replay, +27% live compound) but
  stays OPT-IN. Before any flip: (a) broaden the task sample (T2 +
  2-3 other benches, mma-only vs off) to separate basin-lottery from
  systematic steering; (b) if steering is real, trial the fp8q-PV
  fallback (f16 PV halves the numerics perturbation) which the design
  doc specified for exactly this contingency.

## 2026-07-10 -- mma basin-steering matrix: NOT systematic; one deterministic T8 tie-flip

{T2,T5,T11,T8} x {off,mma}, minimal config (mma the only variable), fresh
server per trial: T2 0.839->0.844, T5 0.776->0.797, T11 0.850->0.850,
T8 0.815->0.559. With all history: T8 off = 0.815/0.834/0.815, T8 mma =
0.560/0.561/0.559 -- three near-identical scores per leg. VERDICT: mma
does NOT systematically steer (3/4 tasks parity or better); it behaves
exactly like a REBUILD in the documented cross-build tie-lottery sense
-- tie-class token flips, neutral in expectation, and T8 has one
load-bearing near-tie (the register-handler schema fork) that fp8
attention numerics deterministically resolve into the punished side of
that task's hidden auth gate. agent_tests/code_metrics 1.000 throughout.

DECISION: Q27_FD=mma cleared for the CC SERVING env (full stack:
+Q27_SUFFIX=1 +Q27_MAXD=auto7, 249 t/s aggregate measured) -- real-work
tie-flips are symmetric (T5 went UP); the T8 fork is a benchmark-gate
idiosyncrasy, same risk class as every shipped numerics change (fp8q,
fp8-PV) under the house tolerance protocol. Binary default stays fd2
(benchmark comparability + the cross-build rule: same-binary legs only).
Reconciliation path if T8-class forks ever matter: the fp8q-PV fdmma
variant (f16 PV) from the design doc's contingency ladder.

## 2026-07-10 -- width-12 P0 DONE: mechanical lane widening 8->12, canonical EXACT, serving-VRAM finding

Plan docs/plans/2026-07-10-width12-verify.md P0, executed as filed.
Surface: p[16] struct family (P3/CP3/XQ3, IP3 + new WIP3, FCP3/FIP3,
Q4Lanes/Q8Lanes [10]->[16]); prep_round/finish_round converted to
pointer-struct signatures at the 17/25-param wall (finish acceptance
chain generalized to 11 drafts / 12 verdicts, leading-run semantics
preserved verbatim; outcome 10->14 ints {n, t1, dr1..dr11, pending});
4 new verify lanes i..l with full activation sets + XQuant + pos/draft/
verdict scalars (NO h_next/pos_m -- MTP ladder stays 4..7 by policy);
+4 GDN role sets (S_spare8..11 + rings, +627MB/engine); perm mod 12
(SBuf/RBuf 12-way, advance sites, refinish +12 fixup, lanes[12]); graph
zoo re-dimensioned [8]->[12] perms (verify_graph_w [13][12], capture
loops 12); logits2/FD-scratch/d_mask_ids/h_mask_ids5 12-lane; argmax
chain + logits2 offsets to 12; em[12]/oc[14]; ctx_round_reserve keyed
on verify_w_max() (value-identical until the P1 suffix width knob).
Fix in passing: reset() memset stopped at spare 5 (stale P12b list,
benign only because roles are write-before-read within a round) -- now
resets all 11 spares; stale mod-7/mod-6 comments corrected. gemv N=12
instantiation and fdmma W>8 deliberately NOT touched (P1/P2 gates).

GATES all green:
- canonical EXACT both models, both binaries: vanilla a2982c51 (fp16),
  qwopus 4c4120c7.
- replay byte-identity vs pre-widen b69cbd9 binary: gated fp8 auto7 and
  the full CC stack (fp8+PMIN0.5+auto7+SUFFIX+FD=mma), 256 tok each,
  byte-identical. (Modulus proof carried by construction: role access
  is fully SBuf/RBuf-indirected, so a larger modulus only relabels
  which physical buffer holds a role.)
- test_kernels ALL PASS (3090 leg); depthctl/toolconstrain/suffixdraft
  CPU suites PASS; compute-sanitizer memcheck 0 errors (gated fp8 leg).
- capture wall: 12 perms x 23 graph sets at auto7 -- server slot-0
  startup unchanged-feeling (~1s capture log gap); CLI canonical run
  captures 12 perms without measurable startup regression at ctx 2048.

FINDING (P1 blocker for the live trial): the widened 2-slot eval config
(qwopus, 131072 + 32768 fp8, the 249 t/s env) OOMs at slot-1 sampled
graph instantiation on the 32GB 5090 -- slot 0 comes up, slot 1 dies at
engine.cuh cudaGraphInstantiate. Real headroom was ~2.5GB, not the
plan's ~5GB estimate; +627MB/engine roles x2 + ~1.5x graph memory
crosses it. q27-eval RESTORED on the pre-widen binary (byte-identical
serving; snapshot kept at build/prewiden-b69cbd9/). P1 must pick the
trial serving shape first: single-slot 131K (fits trivially), or
131K + smaller slot-1, or reclaim graph memory. Widened binaries at
build/{q27,q27-server} md5-differ from snapshot as expected.

## 2026-07-10 -- width-12 P0 review pass: 3 real latents fixed (adversarial 4-lens workflow, 11 agents)

Second-pass review of c399d70 (4 lenses: missed-sites / perm-state-machine
/ memory-safety / graph-zoo, each finding adversarially verified): 7
confirmed findings deduping to 3 real items, 0 refuted -- ALL latent
until P1 launches widths > 8, i.e. exactly the class the byte-identity
gates cannot see:
1. CLI round-outcome hist[8] (engine.cu, 3 lenses independently): the
   em[] widening's missed sibling -- hist[n-1] at n=9..12 is a stack OOB
   in the very replay leg P2 depends on. -> hist[W_MAX] + 12-bucket print.
2. k_quantize_x3 flat 8-lane arg list (kernels.cu): the terminal ternary
   fall-through would alias lanes 8..11 onto lane 7's XQuant buffers --
   the documented P12b "lane-count landmine" reborn at the next width.
   -> lanes ride the XQ3 struct, one slot per lane by construction.
3. Q27_PHASE_STATS vw buckets capped <= 8 (engine.cuh): widened to
   W_MAX+1 mechanically; NOTE the binding gap is that suffix rounds are
   not phase-stamped at all, so the P2 width curve needs suffix stamping
   + phwn/phwm print + rig extension (filed in the plan's P1 list).
Plan doc gained the review addenda: P1 serving-shape blocker (2-slot OOM),
P1 suffix stamping, P2 fdmma gate+switch co-widen + launch return check,
P3 gate-hist widening before MTP ceilings 8+.
Re-gated after fixes: canonical a2982c51 + qwopus 4c4120c7 EXACT, full-CC
replay leg byte-identical vs pre-widen binary, test_kernels ALL PASS,
sanitizer memcheck 0 errors. conclave auto-review attempted, hung on
external CLIs (timed out) -- workflow review is the pass of record.

## 2026-07-10 -- width-12 P1 DONE: suffix width 12 LIVE (Q27_SUFFIX_W), gemv N=12 no-cliff, eval flipped to single-slot widened

Q27_SUFFIX_W knob decouples the SUFFIX verify width from the MTP gated
width (ladder stays 4..7): envs parsed pre-capture, warm runs at the
widest width, ONE extra per-perm verify graph captured at exactly sfx_w
(12 graphs, not 4 widths x 12), suffix branch proposes sfx_w-1 and
launches verify_graph_w[sfx_w], reserve rides verify_w_max(). Suffix
rounds now phase-stamped into their OWN GenStats bucket (sfx_ms/
sfx_rounds; server [req] appends sfxm/sfxn under Q27_PHASE_STATS --
phwn/phwm untouched, parsers safe). gemv_q4_n/q8_n N=9,11,12
instantiated (10 existed); width_bench sweep extended 5..12.

CLI FIX in passing: the --tokens --spec loop drove spec_round with an
EMPTY suffix index (only Engine::generate wired sfx.reset/append) --
Q27_SUFFIX could never fire on CLI replays, zero-fire by construction.
Now wired; CLI is a valid suffix venue for the first time (explains a
chunk of the 07-09 'replay payloads cannot evaluate this drafter').

MEASURED (5090):
- gemv width sweep (L2-rotated q4 ffn / resident q8 head): N=12 per-lane
  cost only +9% vs N=8 (5.98 vs 5.48 us/lane); q8 head +18.6% total for
  +50% lanes. NO register cliff -- mma16 GEMM contingency NOT needed.
- CLI echo leg (6x5-tok periodic prompt, fp8+PMIN+auto7+SUFFIX_W=12):
  suffix fires 17-20 rounds x 12 tok = ~80% of decode at 12 tok/round.
- BYTE-IDENTITY: under fd2, wide-suffix leg is byte-identical to both
  the suffix-off leg and the PRE-WIDEN binary (width-invariance holds
  bitwise, 20 wide rounds live). Under Q27_FD=mma the wide leg FORKS
  (~token 100): W=12 rounds fall through to fd2 while W<=8 rounds run
  mma, so round grouping selects the attention kernel per position --
  the documented mma tolerance-class regime, now width-coupled. NOT a
  widening bug; P2's fdmma W<=16 lift removes the fork.
- canonical a2982c51 EXACT on the final binary; sanitizer memcheck 0
  errors on the wide leg; test_kernels ALL PASS.

SERVING (Gabe: "let's do one slot"): q27-eval RECREATED on the WIDENED
binary, single slot 131072, full stack + Q27_SUFFIX_W=12 +
Q27_PHASE_STATS=1: 27.0GB/32.6GB = 5.6GB headroom (2-slot OOM resolved).
Smoke (echo completion): 200 tok / 18 rounds = 11.1 tok/rnd avg, suffix
17x12, **tps=326.5** -- highest live per-request decode ever recorded on
this engine (prior per-req peak 254). P2 next: live CC/T8 trial (does
real traffic reach AL ~10.5? wall delta vs 249 t/s stack?), fdmma W<=16
lift, width-curve at 12 via sfxm/sfxn.

## 2026-07-10 -- width-12 P2 DONE: LIVE T8 CONFIRMS THE THESIS -- suffix AL 10.61 on 61.6% of decode, good basin, per-req peak 294 t/s

fdmma W<=16 lift shipped (7b85f06: s_geo re-stride, cases 9..12, honored
launch return; fdmma_test W=4..12 modeled-EXACT; the test's own p[8]
combine-fork P3 was the W=9 crash -- fixed). Wide suffix rounds now run
the MMA kernel. Documented: wide-vs-narrow round grouping stays
tolerance-class under mma (union-window tile phase p_beg is a function of
the round's lane set) -- inherent to shared-KV scoring, same regime the
basin matrix cleared for mma itself.

LIVE T8 TRIAL (widened single-slot stack: fp8+PMIN0.5+auto7+SUFFIX_W=12
+FD=mma+PHASE_STATS, thunderdome claude-code-q27-haight, concurrent
openrouter campaign on host = CPU-contended wall):
- score 0.84 @ 210s, hidden_tests 0.9375, agent_tests/code_metrics 1.0
  = GOOD basin (this morning's mma legs went 0/2 bad at 0.56; n=1 draw
  on the new binary landed well -- tie-lottery neutral-in-expectation
  holds).
- **suffix AL 10.61** (1953 fired rounds, 20721 tok) = 61.6% of ALL
  decode (33635 tok, 71 reqs) -- plan predicted ~10.5-10.7 uncapped, live
  traffic DELIVERS it (was 6.48 @W7-cap, 7.49 @W8).
- aggregate 7.09 tok/round; decode 224.9 t/s aggregate SINGLE-slot
  (2-slot baseline was 249/233 aggregate with overlap; per-req median
  211 / p75 248 / **peak 294** vs prior per-req peak 254).
- width-12 cost point (sfxm/sfxn, mixed T8 ctx): 39.0 ms/wide-round =
  3.68 ms/token committed, vs gated rounds' 26.3 ms at 4.62 tok/rnd =
  5.68 ms/token -- wide suffix rounds are 1.5x more wall-efficient per
  token. Suffix rounds carried 51% of decode wall for 62% of tokens.
- fire rate unchanged vs width-8 trial (~1950 rounds/trajectory): the
  widening converts the SAME fires into +63% tokens each, exactly the
  cap-release mechanism the plan bet on.
VERDICT: width-12 GO for the CC serving env. q27-eval stays on the
widened single-slot config (27.0/32.6GB). Replay accept_ab deferred --
live trial is the venue of record (and the CLI can now fire suffix for
future replay work). P3 (MTP ceilings 8..10 pricing, GDN deferred-
snapshot) remains optional per plan.

## 2026-07-10 -- fdmma tuning pass: STAGES=1 2-CTA variant DEFAULT (+17-26% kernel at depth, bitwise-identical)

Measure-first (standing 3a rule: eligibility is the metric). ncu on the
shipped 1-CTA kernel: Eligible Warps 0.11/scheduler, No-Eligible 89.5%,
DRAM 26%, SM 20% -- severely latency-bound; 6 resident warps and THREE
full-CTA barriers per tile leave nothing to hide latency with. W=12 also
broke the flat width curve (+56% over W=8 at 61K: 5 of 6 warps live =
compute serialization on top of the same stream). Prefetch-reorder
experiment (issue next tile before the V transpose): WASH -- copies were
already hidden; reverted.

THE LEVER: k_attn_fdmma<W, STAGES>. STAGES=1 single-buffers K/V (fetch
tile at loop top; trailing uniform barrier guards buffer reuse) and
shrinks s_q to live rows (fdmma_qrows -- 96 was a fixed bound nothing
read past), landing smem 49.1KB@W12 / 40.9@W8 and 168 regs (zero spill
at W=12) under __launch_bounds__(192, 2): TWO CTAs co-reside and hide
each other's barrier/memory gaps -- inter-CTA overlap replaces the
intra-CTA ping-pong. Per-tile arithmetic extracted to a SHARED
fdmma_tile_compute (both variants compile the same math -- outputs
bitwise-identical by construction, verified).

MEASURED (attn_fdw_bench, S1 vs S2): 26K W4/8/12 = 1.07/1.05/1.16x;
61K W4/8/12 = 1.18/1.17/1.26x. 61K W=12: 354.7 -> 282.4us = 4.0x over
fd2 (was 3.15x). bitwise=OK every leg. Engine share: ~-1.1ms per wide
round at 61K (attention ~15% of the 39ms wide round) -- real but below
T8 trajectory noise; kernel bench is the measurement of record.

SHIPPED DEFAULT-ON (bitwise class, same as the P14 grid remap
precedent): spec3.cu passes stages=1; Q27_FDMMA_STAGES=2 restores the
old staging for A/B. Gates: fdmma_test 42/42 across BOTH variants
(modeled-EXACT everywhere), canonical a2982c51 EXACT, S1 engine wide
leg byte-identical to the S2 binary's, sanitizer memcheck 0,
test_kernels ALL PASS. q27-eval restarted on the tuned binary.
Remaining headroom: 282us vs ~130us floor at 61K W12 -- next lever
would be the d-split O-accumulator halving (warp-pair PV, 3-CTA
territory) or cross-CTA split-count retune; neither commissioned.

## 2026-07-10 -- fdmma warp-pair PV (d-split) = NEGATIVE; reverted, attribution clean

Built k_attn_fdmma<W, STAGES, PAIRED>: 384 threads, even warp scores
(QK^T + softmax + s_P relayout) and takes PV dims 0..127, odd partner
takes 128..255 of the same rows -- o[16][4] = 64 regs (half), sc factors
transported per-gid through s_sc, pair-scoped bar.sync (id 1+rowblock,
64 threads), per-tile CTA barriers proven sufficient to fence s_P slab
reuse. Bitwise-equal outputs CONFIRMED at every (ctx, W, staging) --
the correctness design was right.

PERF: NO-GO. vs the shipped S1 2-CTA kernel at 61K: W=8 247/278us
(P2/P1) vs 194.6; W=12 300.7/330.8 vs 282.8. Only cell that won: 26K
W=12 P2 = 174.7 vs 186.1 (+6%) -- not worth a seq-dispatched variant
for ~15% of wide-round wall. ATTRIBUTION: the S1 win was INDEPENDENT
BARRIER DOMAINS (two CTAs interleaving each other's stalls), not warp
count -- pairing doubles warps inside ONE domain, adds a pair-bar per
tile, and idles the partner through the score phase. Register relief
(168 -> ~120) bought nothing: regs were not binding after S1.

DO-NOT-RETRY unless the kernel first moves to a structure where the
partner warp has score-phase work (e.g. warp-specialized K-transpose or
software-pipelined QK of tile i+1) -- i.e. as part of a full
producer/consumer rewrite, not as a bolt-on split. Machinery REVERTED
to 1f8f3d3 (fdmma.cuh + bench); fdmma_test re-verified ALL PASS on the
reverted tree. Remaining headroom at 61K W=12: 282.8us vs ~130 floor;
next candidates: cross-CTA split-count retune (ns as an fdmma-only
knob), or the full warp-specialized rewrite -- neither commissioned.

## 2026-07-10 -- fdmma split-count retune: ns = SMs*2/kv_heads (85) DEFAULT = -22..-36% kernel, 5.6x fd2 at 61K W12

Wave quantization was the next binding constraint after S1: grid
(128, 4) = 512 CTAs vs 170 SMs x 2 resident = 340 slots -> a half-empty
second wave. Bench sweep (S1 kernel): ns=85 (exactly one wave) = 26K
W8 83.0us (-28%), 61K W8 150.8 (-22%), 61K W12 202.6 (-29%); ns=64
wins 26K W12 (119.4, -36%); ns=96 shows the textbook overflow-wave
penalty (WORSE than 128 at 61K W8). 61K W12 is now 202.6us = 5.6x fd2
(was 3.15x this morning pre-tuning); day total on that cell: 354.7 ->
202.6 = 1.75x.

SHIPPED: spec3.cu computes fdmma_ns = clamp(SMs*2/n_kv_heads, 16,
FD_MAXNS) once (85 on the 5090), passes it to launch_fdmma AND the
combine; Q27_FDMMA_NS pins it for A/B; fd2 keeps FD2_NS=128 (its own
sweep). NOT bitwise across ns (split boundaries move -> combine fp
order) -- rebuild-class tie-lottery, the regime the basin matrix
cleared for mma. Gates: fdmma_test parameterized over ns {128, 85} =
84 legs ALL PASS (modeled-EXACT at both; ref chunk math had a stale
NS -- caught by 56 honest failures before the fix); canonical
a2982c51 EXACT; same-binary repeat identical; Q27_FDMMA_NS=128 escape
verified; sanitizer 0. q27-eval restarted on the retuned binary.

## 2026-07-10 -- fdmma warp-specialized rewrite = NEGATIVE (0.49-0.61x vs S1@ns85); occupancy story CLOSED

Prototype (tools/attn_fdw_bench.cu k_attn_fdmma_ws, kept as rig): 224
threads = 6 consumers + 1 producer warp owning cp.async K/V staging +
the V transpose into a double-buffered s_vt ring; stage handoff via
named-barrier arrive/sync pairs (ready/consumed ids 1..4, count 224);
consumers hit ZERO CTA-wide barriers in the tile loop; per-row math =
fdmma_tile_compute verbatim. Choreography CORRECT: bitwise-equal
outputs at every leg. Perf: 61K W12 353.3us vs champion 202.4 (0.57x),
61K W8 306.7 vs 150.8 (0.49x), 26K similar.

ATTRIBUTION (closes the day's occupancy arc): the WS CTA has 7 warps
in ONE barrier domain vs the champion's 12 across TWO independent
domains -- multi-CTA interleave hides staging strictly better than a
dedicated producer once smem permits 2 CTAs, and WS's fat smem (double
K/V raw + double vt = ~74KB @W12) forbids the second CTA. Same
conclusion as the prefill Phase-A EV-cut and the warp-pair PV
negative, now measured three ways: for THIS kernel family, occupancy
(CTA count) dominates intra-CTA orchestration. DO-NOT-RETRY unless a
redesign gets specialized staging under ~49KB smem AND <=85 regs
(PP=16 half-tiles would be the entry point -- changes mask/softmax
tile granularity, tolerance-class, unpriced).

fdmma day summary (61K W=12, 126MB stream): 354.7 (morning W12 lift)
-> 282.4 (S1 2-CTA) -> 202.6 (ns=85) = 1.75x tuning day, 5.6x over
fd2, vs ~130-140us modeled floor. Shipped: S1 default + computed
fdmma_ns. Rejected with attribution: prefetch reorder (wash),
warp-pair PV (barrier-domain insight), warp-specialized producer
(this entry).

## 2026-07-10 -- tuned-stack trial triplet (T2/T5/T8): scores at/above every reference band, wide-round wall -6% live, peak 320 t/s

Full stack (width-12 + S1 + ns=85), single slot, CPU-contended host
(same openrouter campaign as the morning trial -- walls comparable):
- T2 collab-server 0.85 @ 100s (refs .839-.851: band top)
- T5 task-queue    0.81 @ 164s (refs .776-.797: ABOVE band, best T5 recorded)
- T8 analytics     0.82 @ 196s (good basin .815-.84; morning leg 0.84)
The ns=85 basin reshuffle drew good everywhere -- 4th/5th/6th good-basin
draws today on the widened stack.

Engine telemetry per task ([req] segmented by journal marks):
- T8: 221.7 t/s agg, 6.50 tok/rnd, suffix AL 10.07 on 52% of decode,
  wide round 36.6ms (MORNING pre-tuning: 224.9 agg, AL 10.61 on 62%,
  39.0ms) -- the -2.4ms/wide-round is the fdmma tuning's live
  signature; agg parity is trajectory-lottery (different suffix share).
- T2: 197.5 agg, AL 8.99 on 32%; T5: 199.0 agg, AL 10.04 on 35% --
  the wide suffix generalizes well beyond T8's echo-heavy profile.
- Per-request peaks: T8 320 t/s (NEW RECORD; 294 this morning, 254
  before today), T5 296, T2 280. Wide-round wall 35.7-36.6ms
  consistent across tasks.

## 2026-07-10 -- width-12 P3 DONE: MTP ceilings 9/10 NO-GO, ceiling 8 NO-BUILD (suffix-shadowed), GDN deferred-snapshot stays shelved (wide marginal is GEMV-N-bound)

Pre-req shipped: gate_cap_hist/gate_n_hist/gate_lane_* widened to W_MAX
sizes (the review's only-safe-under-4..7 landmine).

NEW INSTRUMENT: engine-true wide-round curve via suffix legs (server,
Q27_MAXD=4 + Q27_SUFFIX_W=W, open-cut echo payload at 26K/61K -- the
first run EOS'd instantly on a cleanly-terminated prompt, the known
open-continuation gotcha). Verify-only rounds, ms/round:
  W=5/8/10/12 @26K: 18.63 / 23.41 / 28.83 / 35.31
  W=5/8/10/12 @61K: 19.56 / 24.49 / 30.02 / 36.30
Curve is nearly CTX-INDEPENDENT (+~1ms at 61K, every width: post-tuning
fdmma is that flat) and the marginal lane ACCELERATES: ~1.6ms/lane thru
W8, 2.7 at W9-10, 3.2 at W11-12. Attribution: attention flat (kernel
bench), GDN chains ~0.32ms/lane linear -> the accelerating term is
GEMV-N (width_bench: q4 per-call +12..28%/step past N=8). Per-token at
full accept flattens: W8 2.93 / W10 2.88 / W12 2.94 ms/tok.

PRICING (ladder_price.py extended to ceilings 9/10 + --curve flag; real
d10 chains; W2..4 spliced from old curves -- common across ceilings so
splice precision cannot affect the ranking):
- HOT cctx (fp16-basin): c5 176.4 -> c7 179.5 -> c8 184.3 (+2.7% over
  c7) -> c9 181.7 -> c10 178.0. CEILINGS 9/10 NEGATIVE even here (chain
  acceptance saturates at 4.79 tok/rnd; W10/11 lanes cost 2.7-3.2ms).
- COLD cctx + docs61k: monotonically negative past 5-6, as always.
LIVE CHECK (today's cumulative gch/gnh on the tuned stack): cap=7 fires
16% of gated rounds, n=8 commits 10%, sat7 ~= 25% -- far under any hi8
promotion bar, and the saturating stretches are suffix-owned (AL ~10 on
32-52% of decode). Ceiling 8's hot-chain +2.7% (MTP-wall-only) shrinks
under the live cap mix to <1% engine.

VERDICTS: ceilings 9/10 NO-GO (priced negative everywhere); ceiling 8
NO-BUILD (reopen if live gnh[8] share rises materially or a suffix-off
deployment appears); GDN deferred-snapshot SHELVED (the curve says the
wide marginal is GEMV-N-bound, not GDN-bound -- the honest reopening
lever for cheaper wide rounds is the mma16 NT=16 GEMM pivot, tools/
mma16_bench.cu, 76% SOL flat W2..16). Width-12 plan fully closed:
P0/P1/P2 shipped, P3 priced and resolved. q27-eval restored (widened
single-slot full stack).

## 2026-07-10 -- vanilla-qwen bench caught a P0-era regression; __grid_constant__ fix lands ABOVE the historical baseline (suite 172.2 vs 161.8)

The base-model bench (the whole point of keeping vanilla qwen as the
standard) surfaced what the depth-focused width-12 gates could not:
shortbench suite 149.5 vs the 161.8 reference (-7.6%) with canonical
EXACT -- identical trajectories, pure ms/round. Bisect: fully present
at c399d70 (P0). nsys kernel diff: ONLY the dynamically-indexed
struct-param kernels regressed (k_rmsnorm3 +51%, k_gemv_f16_3 +43%,
k_quantize_x3 +45%, k_attn_fd2 +58%; compile-time-indexed k_gemv_q?_n
+2%). ptxas: stack frame 128 -> 256B -- the by-value lane structs are
copied to per-thread LOCAL memory when indexed by blockIdx (the classic
param trap), and p[8] -> p[16] doubled the copy. The 128B tax had been
there SINCE P10-A0.

FIX: __grid_constant__ const on every struct-by-value __global__ param
(20 kernels: spec3.cu 13, kernels.cu 6, fdmma.cuh 1) -- guaranteed
const-bank residency, no local copy, addressing-only change (bitwise).

RESULTS (vanilla qwen, all gates green: canonical a2982c51 EXACT,
test_kernels + fdmma_test ALL PASS, sanitizer 0):
- shortbench suite 149.5 -> 172.2 = +15% over the regressed state and
  +6.4% ABOVE the 07-09 baseline (161.8); canonical 121.5 -> 140.1.
- echo 2K full stack: 266.7 -> 317.9 t/s (+19%, identical trajectory).
- cctx 26K server replay: classic 127.1 -> 143.0, full stack 154.2 ->
  176.3 (+23% stack-over-classic on the same payload).
Vanilla bench summary on the fixed binary: suite 172.2 / cctx classic
143.0 / cctx full-stack 176.3 / echo full-stack 317.9. LESSON promoted
to the standing gate set: short-ctx suite is the param/launch-overhead
canary -- depth gates alone missed an 8% engine-wide tax.

## 2026-07-10 -- CC DEFAULTS SHIPPED: bare `q27-server model tok` = the full measured stack; 26K echo 400.6 t/s zero-config. Plus llama.cpp cross-engine legs

DEFAULTS FLIP (server only; CLI keeps reference defaults so the bitwise
canonical world is untouched): a bare invocation now resolves to fp8 KV
+ Q27_FD=mma (sm_89+ arch-gated, else fp16+fd2) + PMIN 0.5 + MAXD auto7
+ SUFFIX_W 12 + PHASE_STATS + fast-head + no-think, and --ctx
auto-sizes the KV budget to free VRAM (measured anchor 22.6GB fixed +
34KB/tok fp8; cap 131072, floor 16384, single-slot; multi-slot keeps
explicit --ctx). Mechanism: setenv(overwrite=0) pre-engine, so user env
ALWAYS wins; Q27_PROFILE=ref restores the conservative reference
behavior including the flag defaults (tri-state --fast-head/--think);
Q27_SUFFIX / Q27_PHASE_STATS parses are now value-aware (=0 disables --
were presence-only). Startup prints the resolved profile line. README
Serving section rewritten. GATES: bare-vs-explicit-env server BYTE-
IDENTICAL on the 26K echo payload; profile banners verified both ways;
canonical CLI a2982c51 EXACT; bare server on vanilla@26K echo =
**400.6 t/s** (32 wide rounds) -- the best engine number ever recorded,
from a zero-config command line. q27-eval unit is now env-free:
`systemd-run --user --unit=q27-eval build/q27-server qwopus.q27
qwopus.tok --port 8081` (auto-ctx picked 131072, 27.0GB).

LLAMA.CPP CROSS-ENGINE (same base qwen Q4_K_M, same 5090, same day;
llama-server /completion timings; q27 numbers = today's vanilla bench):
- plain: pp512 3721, pp8192 3562 (q27 fp8 prefill ~3480 @8K = parity),
  tg128 78.7 (plain decode -- both engines BW-bound here).
- spec short-gen: llama draft-mtp10/p0.5 = 133.3 t/s (accept 97/134)
  vs q27 suite 172.2 = q27 +29% same-model same-GPU.
- echo (pure 8-token loop, degenerate best case): llama draft-mtp10
  287.8 vs q27 317.9 (+10% q27); BUT llama ngram-mod = 889.2 t/s
  (251/251 drafts accepted, ~24-tok ngram drafts) -- unbounded draft
  length crushes the pure-loop case where q27 is lane-capped at W=12.
  NOT commissioned as a counter: live CC traffic runs AL ~10.6 (< the
  12 cap; the cap does not bind on real work), and the P3 curve says
  wide lanes are GEMV-N-bound -- the structural answer if pure-echo
  ever mattered is the mma16 GEMM-verify pivot (batched-GEMM verify =
  llama's shape), already on file. Ops notes: this llama build's
  llama-cli hangs in conversation mode (-no-cnv is a warning now),
  llama-completion binary is stale (symbol error) -- use llama-server;
  and pkill -f self-matches the invoking shell (use pkill -x, again).

## 2026-07-10 -- VANILLA A/B TRIPLET, q27-vs-llama.cpp through Claude Code: q27 +40% decode, 1.8-3.5x wall at matched scores

Cleanest cross-engine read ever taken here: BOTH engines on vanilla
qwen36 (q27 5.25bpw zero-config defaults vs llama Q5_K_M + its best
config draft-mtp10/p-min0.5, -fa, no-think), same 5090, same CC
harness (claude-code-q27-haight / claude-code-q5km-haight twins), same
day, back-to-back.

SCORES (converge to the model, as always): T2 0.84 == 0.84; T5 q27
0.78 vs llama 0.81 (both at the reference band); T8 q27 0.85 GOOD
basin vs llama 0.52 = the documented engine-independent bad-basin
auth-gate artifact (per the retrial rule it is NOT read as quality;
llama has historical T8 goods).

WALL (end-to-end CC task time): T2 93s vs 323s (3.5x), T5 96s vs 171s
(1.8x), T8 144s vs 109s (llama bad basin truncates early -- not
comparable). DECODE ([req] vs llama eval-time lines, decode-only):
q27 213/227/222 t/s aggregate per task (med 209-264, PEAK 362 = new
live record), suffix AL 8.4-9.6 on 24-31% of decode; llama 157.1
aggregate (med 169, p75 202, peak 259) over 89 reqs / 73.8K tok.
=> q27 +40% aggregate decode, +40% peak, and the wall gap is larger
than the decode gap (prefix-cache architecture + trajectory length --
llama generated ~1.8x the tokens on its own trajectories; documented
cross-run lottery, within-leg telemetry is the rate comparison).

Context for the record: 07-06's depth-match had TUNED llama +31% over
q27 at 75K, retired to PARITY on 07-07; today, post width-12 + fdmma
tuning + defaults, the same-model comparison is q27 +40% decode and
multiples on wall. The llama echo-ngram degenerate case (889 t/s,
earlier entry) is their one winning cell. [2026-07-13 CORRECTION: it is
NOT just the degenerate loop -- see the cap-binding entry that date;
ngram wins the realistic file-re-emission regime too, 653 vs 377, and
q27's "cap never binds" claim was regime-limited. Corrected in README +
speed post.]

## 2026-07-10 -- n>=3 CROSS-ENGINE PROTOCOL RUN: q27 +47% decode, score medians converge, 8/9 vs 5/9 trial robustness

The standing final-A/B protocol (07-05: n>=3 per task, both legs
strongest config, same day) executed: 3 x {T2,T5,T8} per engine, vanilla
qwen both sides, q27 zero-config defaults vs llama Q5_K_M
draft-mtp10/p-min0.5/fa, no-think greedy CC harness, back to back.

SCORES: T2 q27 {.84,.81,.83} vs llama {.45,.84,.83} -- medians 0.83 ==
0.83; T5 q27 {.78,.78,.78} (three identical draws) vs llama
{.79,0.00,.82} -- medians 0.78 vs 0.79; T8 q27 {.83,.85,.56} vs llama
{.57,.53,.83} -- the documented bimodality on BOTH engines, q27 2/3
good draws vs llama 1/3. Trial robustness across the 9 draws: q27 8/9
in-band; llama 5/9 (one hard 0.00 at 443s -- harness-terminal -- plus a
0.45 and two bad T8 basins). Serving-layer signal, held at n=9.

DECODE (within-leg, the rate currency): q27 231.3 t/s aggregate (430
reqs / 137.5K tok; median 225, p75 277, PEAK 378 = new live record;
suffix AL 9.4 on 37%) vs llama 157.4 (197 reqs / 313.5K tok; median
155, p75 187, peak 274) = **q27 +47%** -- stronger than the n=1's +40%.
WALL: q27 medians 86/97/145s vs llama 286/443/146s -- 3-4x on T2/T5 but
trajectory-confounded (llama ~2.3x token volume on its own
trajectories); decode telemetry is the claim, wall is context.

Publish-gate ledger: parity bar passed at +47% on the protocol-grade
measurement. Remaining before any writeup: nothing on the measurement
side. qwopus standing env restored after the run.

## 2026-07-10 -- statistical register correction on the n=3 robustness read

The 8/9-vs-5/9 trial-robustness split and the T8 2/3-vs-1/3 basin split
CANNOT carry weight at n=9/leg (Fisher's exact p ~= 0.29 on the
robustness table; the T8 split is one draw). Both demoted from
"signal" to descriptive observations in the README; the claims that
stand statistically are the score-median convergence and the +47%
within-leg decode gap (430 requests of telemetry). Also promoted into
the README: the protocol's 07-05 filing date (acceptance criteria
predate the result -- the credibility asset), replication direction
(+40% n=1 -> +47% n=3), and quality parity scoped as a SYSTEM-level
claim with the strict-parser reality stated plainly (strict = 0.000 on
T8-class for any engine; the tolerant parser is load-bearing).

## 2026-07-10/11 -- codex-vs-q27: my adapter had the wrong wire_api; the endpoint already shipped

CORRECTION to the entry as first filed. The first codex triplet crashed
at 0s because MY adapter set `wire_api = "chat"`, which current codex
rejects at config load ("no longer supported, set wire_api=responses",
codex discussion 7782). The premise I wrote -- "q27 does not serve
/v1/responses" -- was wrong: q27-server has a complete native Responses
handler (POST /v1/responses; input items + instructions -> ChatML,
function_call + custom apply_patch tool bridging, reasoning items,
response.created/output_item.done/output_text.delta/completed SSE with a
usage block; the codex-rs wire facts are documented at the handler).
Live curl on the standing server returns the correct event sequence.
Fix was one line in the adapter (wire_api=responses). Lesson: read the
server's own route table before declaring a capability missing -- I had
a "blocked, needs a third API surface" writeup for an endpoint that was
already built and gated.

## 2026-07-11 -- codex ENABLED on q27 (third harness): two real /v1/responses fixes, T5 is a model basin

Codex now runs against q27 (adapter codex-q27-haight, wire_api=responses).
Path to green, two genuine handler bugs found by running it:
1. OUTPUT-ITEM LIFECYCLE. The existing streaming handler emitted
   response.output_text.delta with no open item; codex 0.143 enforces
   the lifecycle ("OutputTextDelta without active item" -> turn abort).
   T2 survived only because its first output was a tool call (a complete
   done-only item); T5/T8 led with text and died at 14-22s. Fixed: emit
   output_item.added + content_part.added before the first delta, and
   output_text.done + content_part.done + output_item.done at flush.
   T8 recovered 0.00 -> 0.85.
2. BARE-CALL RECOVERY (parity with the Anthropic path). Added
   parse_bare_tool_calls over the accumulated text on finalize -- the
   model sometimes emits {"name":...,"arguments":...} without the
   <tool_call> wrapper.

RESULT: T2 0.85, T8 0.85 (matching/beating Claude-Code-on-q27 .85/.82);
q27 served codex at 157-352 t/s/request, 96% prefix-cache reuse (hit
49152 every turn), suffix drafter firing. T5 stays 0.00, and it is a
MODEL-TRAJECTORY BASIN, not a wire defect: Qwopus chose to write the
whole index.ts as ONE bare exec_command heredoc (14,662 chars) that
truncated mid-string -- unrecoverable because the content was never
finished emitting. Same class as the T8 auth-gate bimodality; CC handles
T5 fine on the identical model, so it's codex-instruction-style-specific
model behavior, not q27. Three harnesses now proven on q27 (Claude Code
native Anthropic, CRUSH OpenAI chat, codex OpenAI Responses).

## 2026-07-11 -- vanilla qwen x codex: T2 parity with CC, T5 is a model x harness basin (3 shapes, un-recoverable)

Ran the codex triplet on VANILLA qwen (the published-numbers model).
Result: T2 0.85 (== CC-on-vanilla 0.84-0.85), T8 auth-gate basin
(0.51/0.23 draws -- bimodal on every engine), T5 0.00.

T5 is a MODEL x CODEX-HARNESS basin, confirmed by three distinct
un-recoverable malformations across three runs -- Qwopus: one 14,662-char
bare exec_command heredoc, truncated mid-string; vanilla run 1: dropped
the opening quote on the "arguments" key; vanilla run 2: emitted
`{"name": "exec_command",` and abandoned the call mid-object. The common
cause is the model, not the wire: under codex's tool-instruction style
this model family mis-formats or abandons tool calls on T5, differently
each trajectory; Claude Code drives the SAME models through T5 at
0.78-0.81. q27 faithfully relays what the model emits.

Two REAL parser fixes landed along the way (both correct, unit-tested,
not basin-overfit -- they recover terminated bare calls in production):
(1) Responses bare-call recovery no longer guards on non-empty `tools`
-- codex registers its shell tool as a hosted type this handler skips,
so `tools` is empty yet the model still emits bare calls for it; the
parser only recovers well-shaped name+arguments JSON and codex validates
names itself, so a spurious recovery is harmless. (2) drift mode 9:
repair a dropped opening quote on the "arguments" key
({"name":"X",\narguments":{...}} -> valid) -- "arguments" is the schema
key so the repair is unambiguous inside these segments; unit-verified on
the exact vanilla-run-1 bytes (recovers to a valid exec_command call)
with no false-fire on prose containing `arguments"`.

VERDICT: codex is a working third harness on q27 across both models
(T2 parity with CC). T5's tool-formatting basin is a codex x
model-family interaction to note, not a q27 bug to fix. q27-eval
restored to the qwopus standing env.

## 2026-07-11 -- full 21-task sweep, Claude Code x vanilla q27: the breadth number

First full-suite sweep (was always a 3-task keyhole before). CC on
vanilla qwen, q27 zero-config defaults, n=1/task, single slot.

DISTRIBUTION (N=21): mean 0.748, MEDIAN 0.830, stdev 0.140, range
0.455-0.889. ZERO crashes -- min 0.455 is a real low score, not a wire
failure; the engine served all 21 categories (greenfield, features,
bugfix, marathon, recovery, algorithmic, reasoning, correctness,
ambiguity, real-repo) without a single 0.00.

Clean category split -- median 0.830 because the easy/medium/greenfield
band clusters tight while the hard-reasoning tail pulls the MEAN down:
- 0.83-0.89 (13 tasks): plugin-marketplace .889, then a wall of .85
  (monorepo-disaster, fts-search, ssg-toolkit, yaml-escapes,
  yaml-trailing-comma, financial-ledger[correctness/hard],
  debug-nightmare[bugfix/hard]), collab .837, phantom-invoice .833,
  time-tracker .830.
- 0.72-0.82 (4): constraint-scheduler[algo/hard] .817, analytics .816,
  task-queue[marathon] .789, structural-merge[algo/hard] .724.
- 0.45-0.59 (5, the hard-reasoning tail): factory-reset[reason/hard]
  .585, permission-maze[ambiguity/hard] .555, ecommerce .510,
  reactive-spreadsheet[algo/hard] .499, beam-splitter[reason/hard] .455.

READ: the tail is a MODEL capability ceiling on reasoning/algorithmic/
ambiguity-hard, not an engine effect -- a faithful engine serving this
27B would show the same shape (correctness/hard financial-ledger scored
.85, so it is reasoning depth, not correctness, that bites). The
engine's job is to serve every category without failing, and it did:
21/21, no crashes, median 0.83.

DECODE over the whole sweep (671 requests, 255K tokens): 233.1 t/s
aggregate, median 220, p75 254, PEAK 394 (new record); suffix drafter
AL 9.4 on 38% of decode. Throughput held across all categories.

Blog quality claim now has breadth behind it: median 0.83 across 21
tasks / 10 categories, zero crashes, at 233 t/s aggregate.

## 2026-07-11 -- Q27_W_MAX build knob: narrow builds for smaller cards (3090 restored to ~49K)

FINDING first: width-12 broke 3090 (24GB) serving. The default build
OOMs at graph instantiation on the 3090 REGARDLESS of --ctx (fails at
16K same as 49K) -- the 12-perm graph zoo + 12 GDN role sets + weights
overflow 24GB before KV is even allocated. Width-8 era ran 32K at
23.1GB; width-12 added ~1.5-2GB (4 role sets + 50% more captured graphs)
and pushed it over. Also found: --ctx auto's fixed anchor (22.6GB) was
calibrated on the 5090 fp8/mma path and (a) didn't scale with the graph
zoo, (b) forced a 16384 floor that itself OOMed rather than failing
clean.

FIX: `-DQ27_W_MAX=N` (default 12, floor 8, cap 12). W_MAX controls the
memory-scaling axis -- GDN role count, perm modulus, graph-zoo
dimension, max verify width -- so a narrow build reclaims one role set
(~157MB) + one perm's graphs (~130MB) per width dropped. Crucially
separated from W_PLUMB=12, the FIXED lane-buffer plumbing (per-lane
buffers i..l, p[16] structs, the 14-int finish outcome) that must stay
12-wide regardless -- the narrow build's conflation of these was 5
"too many initializer" compile errors, now clean. auto-ctx now
W_MAX-scales the anchor and clamps to what fits (warns instead of
forcing an OOM floor).

GATES: default W_MAX=12 canonical a2982c51 EXACT (guards are alloc-only,
zero behavior change); W_MAX=8 build canonical EXACT AND its
gated-fp8-auto7 leg BYTE-IDENTICAL to the W_MAX=12 build (760801e0) --
proving the narrow build is a pure memory reduction, no behavior change
at the widths it supports (widths <=8 use the same role prefix by
construction). 3090 measured with the W_MAX=8 + Q27_KV=fp8 build:
serves 32K at 23.2GB, 49K at 23.8GB (generates correctly) -- BETTER
than the old 32K because fp8 KV halves the per-token cost.

Build narrow (until a Makefile knob lands):
  nvcc <NVCCFLAGS> -DQ27_W_MAX=8 <sources> -o build/q27-server-w8
3090 serving: build W_MAX=8, run with Q27_KV=fp8 (sm_86 CC profile
arch-gates fp8 off, but fp8 KV storage works on sm_86 -- only the fdmma
MMA kernel needs sm_89; force it), explicit --ctx up to ~49152.

## 2026-07-11 -- matched 21-task vanilla-vs-qwopus: "vanilla beats qwopus" was an ARTIFACT (median tie)

Ran both models through the full 21-task suite, same day/GPU/harness
(vanilla run 13-14-19, qwopus 14-07-43). My pkill clobbered q27-eval
mid-qwopus-sweep (corrupting 5 consecutive trials); those 5 + the
documented-bimodal analytics were re-run and substituted (structural
0.76, financial-ledger 0.85, permission-maze 0.51, reactive 0.76,
circuit-debugger 0.27, analytics 0.00-retrial).

PAIRED RESULT: vanilla median 0.830 / qwopus median 0.836 -- a TIE.
Paired-delta median +0.000; 13 of 21 tasks tie within +-0.02. The mean
gap (qwopus -0.024) is TWO bimodal-basin draws swinging BOTH ways:
qwopus lost analytics (0.816->0.00) and circuit-debugger (0.626->0.27)
but WON beam-splitter (0.455->0.845) and reactive-spreadsheet
(0.499->0.760). Net wash, basin lottery on the hard-reasoning tasks;
identical on the 13 stable ones. This is the documented "scores
converge to the model" -- qwopus is a SPEED fine-tune (+35% acceptance,
219 vs 162 t/s cctx replay), same quality as base. The earlier
"vanilla beats qwopus" read was the broad-vs-narrow artifact (vanilla
had a 21-task sweep, qwopus only the 3-task keyhole); matched, they tie.

OPS: canonical a2982c51 re-verified clean (closes W_MAX commit f551561,
whose in-session check returned the co-located-CLI-OOM empty-md5).
Lesson re-logged: pkill -x q27-server also kills the standing eval unit
-- name the PID or CUDA_VISIBLE_DEVICES-scope it.

## 2026-07-11 -- turbo3 KV port phase 1: format+WHT+quant/dequant VALIDATED (microtest ALL PASS)

Porting TurboQuant 3-bit KV (Gabe's llama-cpp-turboquant fork @c3e6dbb13)
into q27. Phase-1 = de-risk the format before any engine wiring, done
microtest-first against the CPU reference (ggml-turbo-quant.c).

turbo3 = QuaRot-style: per-128-group L2-normalize -> forward Walsh-
Hadamard (baked, seed-42 s1/s2 diagonals + butterfly) -> 8 Lloyd-Max
centroids {+-0.190207,0.118786,0.066822,0.021663} scaled by a corrected
per-block norm (grp_norm/recon_norm). block_turbo3 = 50B (fp16 norm +
qs[32] 2-bit + signs[16] 1-bit) covering 128 dims. head_dim=256 = TWO
groups, TURBO_D=128, no padding.

SHIPPED: src/turbo3.cuh (shared device format: struct, centroids,
TURBO_S1/S2[128] verbatim, WHT inv_sqrt_128, nearest-centroid, dequant
helper); tools/turbo3_test.cu (faithful device port validated vs CPU
oracle). RESULTS (5090): device quant == CPU (1/4096 midpoint tie),
norm/dequant EXACT, round-trip q->deq->inverse-WHT cosine 0.983 (= the
3-bit quality floor), and THE READ CONTRACT: WHT(q).dequant(K) vs true
q.K score cosine 0.983 over 512 pairs -- proving the Q-rotate +
rotated-K-storage math (the silent-wrong-output risk) is correct.
Debug lesson: the device __constant__ arrays loaded fine; the initial
"4096 mismatches" was a hand-typed oracle-array transcription bug in the
test, not a port bug (device-only round-trip 0.985 isolated it).

Build (Makefile target pending): nvcc -O2 -std=c++17 -arch=sm_120
tools/turbo3_test.cu -o build/turbo3_test.
NEXT (phase 1 cont., own session): block-addressed KV storage (8
blocks/row = 400B, 2.56x vs fp8), k_kv_store3 turbo3 cooperative write,
fd2 turbo3 read (k_attn_fd2_turbo3 + fd2_ld8_turbo3), forward-WHT on Q
after rope + inverse-WHT on attnout post-combine, all behind
Q27_KV=turbo3 with fp8/fp16 BITWISE. fdmma (dequant-to-e4m3 smem)
deferred. Then the fresh quality gate (no head_dim=256 oracle -- PPL +
needle on the q27 model). Full port spec: wf_f94f54d8-2ab.

## 2026-07-11 -- turbo3 KV phase 1 WIRED: store/read/rotate live behind Q27_KV=turbo3 (decode-only), fp8/fp16 bitwise-EXACT

Phase-1 engine wiring for the TurboQuant 3-bit KV port (format validated
last session, spec docs/plans/2026-07-11-turbo3-kv-port-spec.md -- recovered
from workflow wf_f94f54d8-2ab into the repo).

SHIPPED (all TDD'd: stubs -> failing microtests -> kernels -> green):
- spec3.cu k_kv_store_t3: cooperative per-128-group store (one block per
  group, 128 thr: fixed-order L2 reduce -> s1/butterfly/s2 forward WHT ->
  nearest-centroid pack via shfl+ballot -> corrected fp16 norm). Multi-token
  (CP3/IP3, grid (8, K|V, ntok)); k_plain leg stores K as plain fp16 rows
  (turbo3v). Named _t3 -- k_kv_store3 was taken (verify multi-token store).
- spec3.cu k_wht3<INV>: in-place per-128-group Walsh-Hadamard rotate;
  operand order matches the CPU fwht exactly => device == CPU BITWISE
  (tested). Forward on Q post-rope (only when K is turbo3), inverse on
  attnout post-combine PRE-sigmoid-gate (elementwise gate does not commute
  with rotation). Gate half of qg untouched (stride-aware).
- spec3.cu k_attn_fd2_t3<KT3,NW> + fd2_ld8_t3: fd2 with block-addressed
  turbo3 loads (row = pos*n_kv*2 blocks; lane's 4 dims = one qs byte;
  signs[l>>1] bits (l&1)*4+i; hoisted per-block norm). Softmax/merge copied
  byte-identical from k_attn_fd2; fp8/fp16 instantiations untouched.
  KT3=false = turbo3v (fp16 K + turbo3 V, Q unrotated).
- Dispatch: attn_decode/attn_decode3/attn_decode3_fd2 widened bool fp8 ->
  int kvk (KvKind in cuda_common.h; 0/1 keep old meaning so existing call
  sites/bools stay correct). turbo3 always routes to fd2 -- Q27_FD=v1/mma
  fall through silently (no block-cache leg there).
- engine.cuh: kv_kind parse (turbo3|turbo3v), kv_bytes(is_v) sizing at all
  5 alloc/memset site-pairs (main per-layer caches + MTP + resets); K and V
  sized separately for turbo3v. attn_block + attn_pair insertions (host
  branches on init-fixed kv_kind => graph-capture-safe; MTP draft path
  rides attn_block automatically). turbo3.cuh: _LIST macros share the
  sign/centroid token sequences between device __constant__ and host oracle
  mirrors; shared turbo3_butterfly128 helper.
- Phase-1 guards: prefill_chunk aborts (no turbo3 leg in kv_store_T /
  attn_prefill_T yet -> serving gated off; q27-server refuses turbo3 at
  startup; --kvstats refuses turbo3 up front).
- Makefile: turbo3.cuh deps + build/turbo3_test target.

GATES:
- test_kernels: +9 turbo3 checks x 32 legs ALL PASS. Store idx = 100%
  bit-match vs CPU quantizer (0 midpoint ties in sample; budget 0.2%);
  norms <=1 ULP; k_plain K rows bitwise fp16. wht3 fwd BITWISE == CPU;
  inv(fwd) 5e-7. e2e turbo3 attention vs double host-ref over same blocks:
  <=2.3e-6 (bound 2e-4) both modes, seq 1..8192, ntok 1/5; deterministic
  bitwise. Quality floor vs fp16 attention: cosine 0.966-0.982 (t3),
  0.982-0.984 (t3v) -- the 0.983 read contract, as expected. NaN-hardened
  comparisons (stub RED run exposed max(0,NaN) false-pass).
- Canonical bitwise (C6): master-W8 vs turbo3-W8 same-flag builds on the
  5090, 3 legs (default fp16 / Q27_KV=fp8 / Q27_FD=v1): generated: lines
  byte-IDENTICAL; fp16 leg == stored canonical a2982c51 exactly.
- Smoke (5090, qwopus): turbo3 plain 80.2 t/s, spec 162.7 t/s @ 3.05
  tok/rnd; turbo3v spec 182.6 t/s @ 3.42 tok/rnd with token stream
  IDENTICAL to fp16 leg (and same rounds) over 64 tokens -- K exact + 3-bit
  V barely perturbs; turbo3 K+V diverges only at token ~60 (tolerance
  class). Banner sizes: ~13.4 KB/token K+V (2.56x under fp8, 5.1x under
  fp16).
- Sanitizer: memcheck (turbo3 spec + turbo3v plain) + racecheck (turbo3
  spec): memcheck 0 errors both legs. racecheck: FULL-ENGINE run is a
  HOST-RAM BOMB (sanitizer tracking ballooned q27 to 118-120 GB anon RSS,
  global OOM killed the tmux scope TWICE) -- rerun kernel-filtered
  (--kernel-name 'regex=kv_store_t3|wht3|attn_fd2_t3', note regex= not
  regex:) under systemd-run -p MemoryMax=80G: 0 hazards. Standing ops rule:
  sanitizer on q27 = kernel-filtered + memory-capped unit, always.
- Guards verified live: --pf aborts with message; q27-server exits at
  startup; kvstats refuses.

QUALITY TRIAGE (the port-spec GQA=6 K-risk gate; wikitext-2 test, 16 chunks
of 2048, teacher-forced serial NLL via --nll-serial, qwopus, same chunks all
legs, 16368 predictions):
  fp16     NLL 1.990229  PPL 7.3172   (baseline)
  fp8      NLL 1.991537  PPL 7.3268   (+0.13%, prod default)
  turbo3v  NLL 1.997168  PPL 7.3682   (+0.70% -- 3-bit V alone)
  turbo3   NLL 1.998863  PPL 7.3807   (+0.87% -- 3-bit K+V)
VERDICT: turbo3-K does NOT crater at GQA=6 -- K adds only +0.17% on top of
V's +0.70% (the fork's 7:1 disaster was PPL 2887 vs 7.4; the baked WHT
rotation + two independent 128-groups per 256-dim head evidently carry K
here). SYMMETRIC turbo3 K+V CLEARS the 5% gate with room to spare => stays
the phase-1 config; turbo3v remains a diagnostic escape and the fd2
mixed-type refactor is NOT needed. Serving caveat (post-prefill-port):
canonical-prompt smoke showed acceptance 3.05 vs 3.42 tok/rnd (turbo3 vs
fp16) -- run the accept-rate A/B on real traffic before any default flip.

NEXT: prefill port (kv_store_T turbo3 + attn_prefill read leg) -> unlocks
serving + needle; fdmma turbo3 (dequant-to-e4m3 smem) deferred; InnerQ
skipped (auto-off at head_dim>128).

## 2026-07-11 -- turbo3 KV phase 2: PREFILL ported -- serving + needle UNLOCKED, fp8/fp16 bitwise-EXACT again

Batched prefill gets turbo3 legs; the phase-1 decode-only guards are gone.
Q27_KV=turbo3 now runs the full stack: prefill (f16-MMA or lite) + decode +
verify + MTP warm, server included.

SHIPPED (TDD'd; stubs -> 6 failing checks -> green):
- prefill.cu k_kv_store_T_t3: prefill store over T tokens (grid (8, K|V, T)),
  quant pipeline EXTRACTED to turbo3.cuh turbo3_quant_group and the decode
  store (spec3.cu) refactored onto it -- the two writers cannot drift.
- prefill.cu k_wht_T<INV>: flat [T][row] per-128-group WHT (Q forward
  post-rope / attnout inverse pre-gate), shares turbo3_butterfly128.
- k_attn_prefill_T / k_attn_prefill_mma widened to <CT, KT3, VT3>: ONLY the
  smem staging changes (turbo3_deq_elem per-element for lite;
  turbo3_stage8_h2 2-qs-bytes+1-sign-byte+norm -> 4x half2 for the mma
  tiles); every MMA past the tiles is format-agnostic. cp.async prefetch
  stays fp8-only (CPA guard); fp8q/pv8 kernels untouched. turbo3 = mma
  default (lite via Q27_ATTN_PF=lite); turbo3v = <half,false,true>.
- attn_prefill_T bool fp8 -> int kvk; engine attn_block_T + mtp_warm_T
  wired; prefill_chunk/server guards REMOVED; server auto-ctx per_tok
  (13.6e3 turbo3 / 41.6e3 turbo3v) + slot admission now derives bytes/row
  from Engine::kv_bytes; auto-ctx log KV label fixed.

GATES:
- test_kernels +6 checks ALL PASS (suite green on 3090): store_T 100%
  bit-match vs CPU quantizer at base offset + k_plain bitwise fp16;
  prefill-vs-double-host-ref lite t3/t3v 1.6e-6/1.9e-6 (exact-staging
  class), mma t3 4.8e-3 (fp16 staging of rotated Q + dequant values; the
  scalar mma-vs-lite gate class is 5e-3).
- Canonical bitwise: 6 legs (pre-change binary vs new, fp16/fp8/v1,
  vanilla, 5090) byte-IDENTICAL; fp16 == a2982c51 stored canonical.
- Batched NLL (prefill path, same 16x2048 protocol as the phase-1 serial
  triage): fp16 7.3214 (vs serial 7.3172, +0.06% = the known g64-regroup
  design delta -- control leg); turbo3v 7.3390; turbo3 7.3978 (+1.04% vs
  batched fp16). K/V cost attribution SHIFTS vs serial (batched: V +0.24%
  K +0.80%; serial: V +0.70% K +0.17%) -- the mma leg stages WHT-rotated Q
  to fp16, and that noise lands only on the rotated-Q (turbo3-K) leg;
  TOTALS agree (~+0.9-1.0%) and stay well inside the 5% gate.
- --pf 512 M6 smoke under turbo3: batched-vs-serial continuations
  IDENTICAL (32/32) -- same blocks from both writers, greedy absorbed the
  attention-order noise; batched 2827 t/s vs serial 74.4 (38x) at 512.
- NEEDLE 6/6 at ~122K prompt tokens (turbo3 server, auto-ctx 131072,
  needles at 10..95% depth, exact retrieval every time) -- the port-spec
  quality gate is now fully closed: PPL + needle both pass on 3-bit K+V.
- Sanitizer (kernel-filtered + MemoryMax-capped per the standing ops
  rule): memcheck 0 errors (turbo3 --pf 384 full run), racecheck 0
  hazards (filtered to kv_store_T_t3|wht_T|attn_prefill_*). No OOMs this
  time -- the kernel-filter + MemoryMax + own-unit recipe holds.

NEXT: perf pass on the turbo3 prefill staging (dequant loads are scalar
byte reads; cp.async-style block prefetch if profiles say it matters at
128K), fdmma turbo3 verify leg (dequant-to-e4m3 smem) still deferred,
accept-rate A/B on real CC traffic before any serving-default discussion
(decode smoke saw 3.05 vs 3.42 tok/rnd).

## 2026-07-11 -- turbo3 accept-rate A/B on real CC traffic: acceptance TIES fp8 (basin-matched); wall gap is KERNEL-side (no fdmma leg)

tools/accept_kv_ab.sh (new; accept_ab.sh with KV/kernel legs at the fixed CC
serving point PMIN=0.5/auto7/profile-suffix): fp8-as-served (mma), fp8+fd2
(kernel control), turbo3 (fd2 fallthrough). 1 cold + 3 warm replays each,
qwopus, ctx 32K. Payloads: cctx/cctx2 (real CC transcripts) + repro.

RESULTS (~27K prompts):
  cctx2 (BASIN-ALIGNED: all legs dec=256, rounds ~44 -- the only clean read):
    fp8/mma  250.8 t/s  5.818 tok/rnd  23.20 ms/rnd  fired5 .598
    fp8/fd2  206.0 t/s  5.689 tok/rnd  27.62 ms/rnd  fired5 .595
    turbo3   184.9 t/s  5.818 tok/rnd  31.48 ms/rnd  fired5 .692
    => turbo3 ACCEPTANCE == fp8 EXACTLY (5.818 both; ladder fires MORE at
    depth 5+). The phase-1 canonical-smoke worry (3.05 vs 3.42 tok/rnd) does
    NOT reproduce on real traffic -- it was a 64-token basin artifact.
    => the -26%% wall is kernel: mma->fd2 = 4.4 ms/rnd, fd2->fd2-t3 dequant
    = 3.9 ms/rnd. The deferred fdmma turbo3 verify leg is THE lever.
  cctx / repro: legs FORK BASINS (rounds 27/88/30 and 83/69/58; the
  documented tie-lottery on these payloads) -- cross-leg deltas there are
  basin artifacts, not acceptance signal. FWIW turbo3 landed favorable
  basins in both (8.53 tok/rnd cctx, 4.41 repro, beating fp8's own legs).
  Determinism: warm replays byte-identical everywhere; cctx2 round-count
  drift across replays ([44,44,41], dec identical) = dctl carry, known.

61K DEPTH LEG (docs61k, fp8 vs turbo3; basin caveat applies):
  fp8/mma  128.6 t/s  2.535 tok/rnd  19.70 ms/rnd  rounds=101 fired5 .000
  turbo3    99.7 t/s  2.876 tok/rnd  28.84 ms/rnd  rounds=89  fired5 .154
  => basins forked (rounds 101 vs 89) so tok/rnd is not attributable, but
  the wall story is clear: NO claw-back at depth vs fp8-AS-SERVED, because
  the mma shared-KV verify advantage grows with ctx just as fast as the
  KV-byte savings do (fdmma was 3.65x over fd2 at 61K). -22%% wall at 61K.

VERDICT: turbo3's KV quality costs ZERO acceptance on basin-matched real
traffic. Serving default stays fp8 (mma wall advantage); turbo3 is cleared
as the long-context/memory option (2.56x KV, ~1%% PPL, needle 6/6). Next
perf lever: fdmma turbo3 verify (dequant-to-e4m3 smem) to close the
kernel gap; fd2-t3 read cost (+3.9 ms/rnd @27K) shrinks in relative terms
as ctx deepens (2.56x fewer KV bytes).

## 2026-07-11 -- fdmma turbo3 verify leg SHIPPED: dequant-to-e4m3 tiles, turbo3 joins the mma serving path

The A/B's 4.4 ms/rnd mma->fd2 deficit is closed at the source: Q27_FD=mma
now engages for Q27_KV=turbo3. k_attn_fdmma<W, STAGES, T3>: the STAGES==1
tile fetch expands 3-bit blocks to e4m3 (turbo3_stage8_e4m3: 2 qs bytes +
1 sign byte + norm -> 8 e4m3) in place of the raw cp.async copy; transpose,
MMA, epilogue untouched. T3 is single-buffered only (the default config;
blocking expand leans on the 2nd resident CTA like the raw variant --
Q27_FDMMA_STAGES=2 has no turbo3 leg). turbo3v stays fd2 (diagnostic).

GATES:
- test_kernels ALL PASS on sm_120 (new fdmma-t3 checks: control-fallthrough
  bitwise==fd2 without the env; engaged (!=fd2) with it; rel gated RELATIVE
  to a same-harness fp8 CONTROL: the SHIPPING fp8 fdmma-vs-fd2 output rel
  is 0.16-0.24 (e4m3 Q/P pipeline -- fdmma was only ever gated
  modeled-ref/acceptance-class, never small-output-rel); turbo3 lands at
  1.2-2.2x that floor = the predicted one-extra-e4m3-rounding, gate <=2.5x).
  First absolute bound (3e-2) was wrong and is retired by the control.
- Canonical (p3base vs new, vanilla, 5090): fp16 EXACT (a2982c51), v1
  EXACT, fp8+mma (THE serving path) EXACT; fp8+fd2 leg RE-ROLLED
  (51df96c1 -> 559ea54f, fork at token ~57): cuobjdump confirms the
  k_attn_fd2<fp8> SASS changed (TU codegen drift from the new header
  inlines), fd2-fp8 correctness checks all green (vs host-ref 1e-4, vs v1,
  deterministic) -- the documented rebuild-class near-tie re-roll, accepted
  per the attn-fd2-design canonical-replacement policy. Serving path
  bitwise-stable.
- Accept A/B rerun (cctx2 @32K, docs61k @65K; fp8-as-served vs
  turbo3-now-mma):
  cctx2 (basin-matched, dec=256 both, 44 rounds):
    fp8/mma    252.2 t/s  5.818 tok/rnd  23.07 ms/rnd  fired5 .598
    turbo3/mma 241.2 t/s  5.818 tok/rnd  24.14 ms/rnd  fired5 .689
    => acceptance still TIES EXACTLY; wall -4.4%% (was -26%% on fd2). The
    +1.07 ms/rnd is the in-kernel block-expand vs raw cp.async.
  docs61k (basin fork: rounds 101 vs 84 -- tok/rnd not attributable):
    fp8 128.4 t/s / 2.535 tok/rnd; turbo3 140.7 t/s / 3.048 tok/rnd
    => turbo3 now FASTER at 61K (+9.6%%) in its basin (was -22%%).
- Sanitizer (filtered + capped): memcheck 0 errors (turbo3+mma spec run), filtered racecheck
  (k_attn_fdmma) 0 hazards.

NEXT: turbo3 serving trial (live CC session on the eval box) now that the
kernel gap is closed; prefill staging perf if profiles warrant; ctx-ceiling
sweep (turbo3 should push far past 131K -- 13.4 KB/token).

## 2026-07-11 -- thunderdome T8 A/B (fp8 vs turbo3 serving) + mode-6/7 inference tie-break fix

Live CC-harness trial (thunderdome claude-code-q27-haight, T8 x3 per leg,
same binary 68fd707+parser-fix, same 10-day-old CC image as the 07-09 runs).

ROUND 1 found a SERVING BUG, not an engine delta: the fp8 leg one-shot-quit
all 3 trials (score 0.00, 81 tok, turns=1) -- the first tool call came out
in the name-dropped mode-6/7 drift shape and the rescue REFUSED it: the
modern CC registry carries property-twins (Bash and Monitor both have
{command, description}), orphaned Bash args scored a 4-4 tie, and
infer_tool_name refuses ties. Cross-BUILD tie-reroll (the fd2-fp8 SASS
drift; MTP drafts ride fd2) moved T8's opening call into that shape today.
FIX (api_common.h, TDD'd in test_tokenizer ok12/ok13): on a score tie,
eliminate candidates whose REQUIRED params are not covered by the args
(such a call could never validate); a UNIQUE survivor wins; ties among
required-satisfied candidates still refuse. The exact trial bytes are the
test. Meanwhile the turbo3 leg (round 1, old parser) rolled a GOOD basin
and scored 0.852.

ROUND 2 (patched parser, both legs fresh):
  fp8/mma    0.54 / 0.54 / 0.54  (173/152/130s)  hidden .219  4.00M tok
  turbo3/mma 0.85 / 0.85 / 0.85  (145/132/148s)  hidden .969  2.79M tok
  Live decode telemetry ([req], per-leg): fp8 median 230.9 t/s (219 reqs),
  turbo3 median 205.0 t/s (171 reqs) = -11% median; aggregate confounded
  by traffic mix (bad-basin fp8 sessions flail: 74.8K vs 42.0K decode tok).
SCORE READ: T8 is the documented BIMODAL eval (auth-chain gate artifact;
fp8's 0.54/hidden-.219 IS the known low mode from the 07-10 full-stack
trials, turbo3's 0.852/.969 the high mode). Six turbo3 trials today (2
rounds x 3) all landed the good basin; that is basin-lottery evidence, not
an engine-quality claim -- but it IS the strongest live validation yet
that turbo3 serving is production-shaped: full 100-turn CC sessions,
wall-clock equal-or-better per trial, -11% median decode, zero protocol
failures.
VERDICT: the performance picture is UNCHANGED in kind -- turbo3 costs ~11%
median decode on live traffic (vs -4.4% at matched 27K replay) and nothing
else; scores are basin lottery. Serving default remains fp8; turbo3 is
fully cleared for long-context serving. The parser tie-break fix is the
real catch of the trial (it protects EVERY leg on the modern CC registry).
Gates: test_tokenizer (incl. new ok12/ok13) + test_toolconstrain PASS;
parser is host-side only (canonical CLI legs unaffected); live E2E = the
round-2 legs themselves (100-turn sessions through the patched path).

## 2026-07-11 -- turbo3 ctx-ceiling sweep: allocates to 655K, quality FLAT to 297K, needle 6/6 at a 361K prompt

Ladder (5090, CLI --spec boot + decode, W_MAX=12 build, explicit --ctx):
196608 / 262144 / 327680 / 393216 / 458752 / 524288 / 589824 / 655360 ALL
OK -- no OOM through the top rung (KV 8.9GB + fixed ~22.6GB = 31.5 of
32.6GB). turbo3's ceiling is VRAM-bound around ~660K, 2.5x the 262K native
window and ~2.2x fp8's practical max (MEASURED 2026-07-11 evening
ladder, W12 build: fp8 OK at 294912, OOM at 299102 and 311296 -- so the
earlier "297054 did not fit" line, written from VRAM math before the
probe, happened to be right; the standing ~285K estimate was ~3% low).

Position-bucket NLL (--nll-long, wikitext-2 297054 tokens, ONE pass, no
resets): turbo3 buckets 5.02-6.13 PPL from 0-2k through 256k-320k -- FLAT
(the 256-320K bucket reads 5.431, indistinguishable from mid-range; bucket
wiggle is corpus content). fp8 overlay at its 262144 max: turbo3 tracks
fp8 within +0.65-1.2% in EVERY bucket -- the short-ctx quality delta does
NOT compound with depth.

Needle (turbo3 server --ctx 372736 on :8080, original 355K-haystack
needle_deep protocol): 6/6 EXACT at prompt=361,513 tokens, including the
two beyond-native depths (78%, 95%) -- the deepest validated retrieval on
this engine (fp8's record was 301K).

READ: at 13.4 KB/token the 131072 auto-ctx cap is pure policy for turbo3.
Raising the auto cap for Q27_KV=turbo3 (e.g. to 262144 native) is a
one-line change -- left for a deliberate default decision, since serving
wall at depth and needle latency (355K prefill ~458s first-hit) belong in
that conversation.

## 2026-07-11 -- 3090 + turbo3 thunderdome: a 24GB card serves Claude Code at 131K ctx

W_MAX=8 build (build/q27-server-w8, not a Makefile target), Q27_KV=turbo3,
sm_86 => fd2 path (no fdmma), qwopus, T8 x3 via the same CC harness leg.

- Fit: needs the full card -- with vox-transcriber's 2.7GB resident the W8
  fixed stack OOMs at ANY ctx (fixed ~22.2GB measured). vox paused (Gabe
  approved): 32K boots at 22.7GB; 131072 boots at 23.9/24.6GB. turbo3 KV
  at 131K = 1.78GB -- fp16 (8.5GB) and even fp8 (4.5GB) CANNOT fit beside
  the fixed stack. turbo3 is the difference between a 32K box and a 131K
  box on this card.
- 32K first attempt: all 3 trials died ON THE CONTEXT WALL at ~75s -- T8
  grows past 32K by ~turn 10; the server behaved perfectly (anthropic-shaped
  400 ctx-limit, no crash). Confirms the 07-09 "3090 caps 32K" note is a
  KV-BYTES cap that turbo3 removes.
- 131K run: 3/3 full sessions (509/297/466s, ~3.07M tok each), scores
  0.55/0.52/0.54 = the T8 bimodal low mode again (hidden .198; same basin
  class as the 5090 fp8 leg -- lottery, not hardware). Live decode over
  197 requests: median 70.0 t/s, p90 91.6 -- consistent with the ~60 t/s
  3090 class from 07-09, now WITH 4x the context. Wall 2-3.5x the 5090.
VERDICT: turbo3 makes the 3090 a viable long-context CC box (131K, 70 t/s
median). Constraint to note: the card must be dedicated (vox's 2.7GB is
the difference between boot and OOM).

## 2026-07-11 -- 3090 head-to-head: q27+turbo3 vs llama.cpp+turbo-KV (the fork turbo3 came from)

Same card (vox paused), same model family (qwopus MTP; q27 17.7GB internal
quant vs llama Q5_K_M 19.5GB), same CC harness leg, T8 x3. llama =
llama-cpp-turboquant fork bin (mainline-recent, --spec-type draft-mtp
n-max 10 p-min 0.5, -fa on, single slot).

Fit ladder (llama): q8_0 KV 82K OK (23.4GB, not probed higher);
-ctk/-ctv turbo3: 65K/98K OK (23.1GB), 131K OOM -- its auto-asymmetric
rule fired live ("GQA ratio 6:1 -- upgrading K from turbo3 to q8_0"), so
llama's turbo3 mode on this model is q8_0-K + turbo3-V, and its 19.5GB
weights artifact eats the rest. q27's symmetric 3-bit K+V (validated
no-crater at GQA 6, this morning's triage) + 17.7GB weights reach 131K.

T8 x3 @ each engine's max ctx:
  q27+turbo3 @131K:   0.52/0.54/0.55 (3x low basin), walls 297-509s,
                      decode median 70.0 t/s (p90 91.6, 197 reqs)
  llama+turboKV @98K: 0.84/0.56/0.57 (mixed basins),  walls 242-479s,
                      decode median 80.7 t/s (p90 112.5, 89 gens)
Scores = basin lottery at n=3 (llama drew one high basin; the 5090
campaigns established score-parity at scale -- scores converge to the
model). Substantive deltas: llama +15% median decode on sm_86 (q27 has no
fdmma leg there -- fd2 only); q27 +33% ctx ceiling (symmetric 3-bit K+V,
i.e. the K-no-crater finding is worth exactly the 98K->131K gap here).
READ: on Ada/Blackwell q27 wins wall via fdmma; on Ampere llama's kernels
are stronger but its own shipped turbo-KV config can't match q27's ceiling
because it refuses 3-bit K at GQA>=6 -- which this morning's PPL triage
showed is over-conservative for this model.

## 2026-07-11 -- turbo3 2-slot on the 5090: TWO full 131K slots, two concurrent CC sessions

Q27_KV=turbo3 --slots 2 --slot1-ctx 131072: BOTH slots admitted at 131,072
(30.6/32.6GB; per-slot KV 1.78GB x2). fp8 could never do this -- its
2-slot config left slot 1 ~23K after the per-slot fixed cost (P10-era
budget); turbo3 turns the second slot into a FULL-context peer.

Load test: thunderdome T8 --trials 2 --parallel 2 = two concurrent CC
sessions on one 5090 (R1b round-interleaving). Both completed, 253s/256s
(0.83 + 0.46 -- basin lottery), vs solo turbo3 132-148s: ~1.8x per-session
wall => 2 users in 256s vs ~280s serialized, plus zero queue latency for
the second user. Per-request decode median under contention 103.5 t/s
(p90 200.8 -- the interleave gaps favor whichever slot holds the GPU).

READ: turbo3 makes the 5090 a genuine 2-user 131K box. Serving default
still fp8 single-slot pending a deliberate flip; the 2-slot turbo3 config
is validated and one flag away.

## 2026-07-11 -- 2-slot turbo3 task breadth: T2 + T11 concurrent pairs

Same 2x131K turbo3 config, each task run as a CONCURRENT pair (both slots
loaded):
  T2  (Collab Server, greenfield/complex): 0.85 / 0.84 @ 152/162s,
      hidden tests 1.000 both -- the historical SOLO good-basin score
      (0.851, fp8 era) at near-solo walls, two-up.
  T11 (Debug Nightmare, bugfix/hard):      0.85 / 0.85 @ 69/70s.
With T8's pair (0.83/0.46 @ 253/256s): six concurrent-session runs on the
2-slot box, zero protocol/ctx failures, scores at the single-user record
basins on T2/T11. Lighter tasks (T2/T11) barely feel the second tenant;
T8's heavy decode shows the ~1.8x interleave cost. q27-eval restored to
default fp8 single-slot.

## 2026-07-11 -- docs/multislot-throughput.md: why 2-slot is capacity, not vLLM-style aggregate

Written against the day's measurements: R1b = round-granularity
time-slicing (p90 200.8 shows zero scheduler tax; the ~2x per-request
split on dual sustained decode is pure weight-bandwidth physics). q27
spends its weight-read amortization on speculative WIDTH within one user
(tok/rnd 5.8 = its "batch"), vLLM spends it across users. Cross-user
batching remains rejected per P10-A pricing (12^2 perm product kills the
baked-pointer graph zoo; per-lane sequence plumbing; 2-user VRAM cap that
turbo3 does NOT lift -- GDN role sets, not KV, are the per-slot cost).
Multi-slot's actual product: 2x full-context tenants + zero-queue
admission + near-zero cost on bursty CC traffic (T2/T11 pairs at
solo-class score AND wall).

## 2026-07-11 -- fp8 ctx ceiling MEASURED on the 5090 (W12): 294912 OK, 299102 OOM

Ladder (CLI --spec boot + decode): 262144 / 278528 / 294912 OK; 299102 and
311296 OOM. Replaces the ~285K post-width-12 estimate (+3%). For the
record alongside turbo3's 655360 (2.2x) and fp16's ~180K. Also corrects
the ctx-sweep entry's untested "297054 did not fit" aside -- probed now,
it indeed does not (barely: the boundary sits in [294912, 299102)).

## 2026-07-11 -- auto-ctx cap raised to the native window for compact KV (Gabe sign-off)

server.cu: auto-ctx cap is now format-aware -- 262144 (the advertised
native window) for fp8/turbo3/turbo3v, 131072 kept for fp16. Rationale:
the old cap was a TTFT + estimate-margin guard from the fp8-era formula
distrust, not a VRAM fact (fp8 measured to 294912 today, turbo3 to
655360). Verified: bare boots auto-size 262144 and reach slot-ready on
both fp8 and turbo3; explicit --ctx still overrides both ways. q27-eval
now serves 262144 fp8 zero-config. Cold-prefill TTFT at full window is
the accepted tradeoff (~4-5 min worst case at 262K).

## 2026-07-12 -- fdmma-h16: fp16-MMA verify for Ampere (and fp16-KV everywhere)

Goal: beat llama.cpp wall clock on 3090 thunderdome tasks (yesterday:
llama 80.7 t/s median vs q27 70.0; the gap was fd2-only verify on sm_86).
Sizing first: Q27_PHASE_STATS on a 3090 cctx2 replay showed verify wall =
53% of decode. Plan: docs/plans/2026-07-12-fdmma-f16.md.

SHIPPED: k_attn_fdmma_h16<W, FMT> (fdmma.cuh) -- m16n8k16.f16 MMA verify,
sm_80+. The e4m3 kernel's geometry/split/two-sided-mask/online-softmax/
fd2-partial-epilogue verbatim; the f16 prefill kernel's mma idioms make it
SIMPLER than its donor: S->PV A-frag register identity (no s_P relayout)
and ldmatrix.trans on natural V rows (no s_vt transpose). Single-buffered
half tiles, blocking fill, ~59KB at W8 = 1 CTA/SM on sm_86; launcher ns =
SMs/kv_heads (one 1-CTA wave). FMT covers all three KV formats landing as
half: fp16 raw (uint4 -- uint2 first cut left 4 halves uninitialized,
caught by the unit test as NaN via the NaN-hardened compare), fp8 kv2h,
turbo3 stage8_h2. W 4..8 only (99KB smem cap); 9..12 fall to fd2.
Dispatch: sm_89+ keeps e4m3 for fp8/turbo3; fp16-KV + sm_80..88 all
formats route H16 under Q27_FD=mma. Server profile now sets Q27_FD=mma on
cc_arch 80..88 too.

GATES:
- test_kernels ALL PASS both arches. H16-vs-fd2 rel 1.2-2.7e-3 across
  fp16/fp8/turbo3 x W {5,8} x seq {47,4096} -- an order of magnitude
  tighter than the e4m3 fdmma's 0.16-0.24 floor (fp16 Q and P, exact kv2h).
- Canonical 8 legs (p4base vs new, 5090): fp16/fp8/v1/fp8mma ALL bitwise
  EXACT (no reroll this build).
- Sanitizer: racecheck (3090, turbo3+mma) 0 hazards; memcheck (5090,
  fp16+mma) 0 errors. Filtered + capped per the standing rule.
- 3090 like-for-like replay (same server config as the morning fd2
  instrumentation, cctx2 x3): tps 90.3/93.1 -> 119.9/123.0 = +32%
  decode; verify wall 1456-1481 -> 1096-1111 ms (-25%); dec_ms -24%.
- Thunderdome T8 x3 @131K turbo3+mma vs yesterday's llama best
  (98K turboKV fork, 242-479s walls, 80.7 t/s median):
  q27-h16 @131K: 0.82 / 0.82 / 0.82 (285/304/433s, hidden .938 -- three
  good basins vs llama's one-of-three), live decode median 102.2 t/s
  (p90 133.2, 166 reqs, busy-agg 107.4) vs llama's 80.7 = q27 +27%,
  flipped from -13% yesterday. Raw wall medians 304 vs 267s are
  trajectory-confounded (q27 sessions pushed 2.84M tokens each vs llama's
  1.48M mean -- 1.9x the work); per the standing methodology the decode
  telemetry is the rate currency, and on matched work q27 now wins wall
  outright. VERDICT: on Ampere, q27 beats llama's best on decode rate
  (+27%), ctx ceiling (+33%), and basin draws (3/3 vs 1/3). Both engines
  measured at their strongest same-day configs.

## 2026-07-12 -- 3090 post-h16 profile: the round is at the weight roofline; cp.async-h16 plan KILLED by data

nsys (turbo3+mma decode, 3090): GEMV weight stream = 68.4% of GPU time
(k_gemv_q4 27.5 + q4_n 22.8 + q8 10.8 + q8_n 7.3); h16 verify attention
0.3%, fd2_t3 wide rounds 0.4%, GDN delta_step 3.4%. ncu on k_gemv_q4:
80.8-90.3% DRAM SOL (735-823 GB/s of 936) -- the big FFN gemvs sit at
~90%, essentially the roofline; the small-grid attn projections at ~81%
(sub-saturating grid, ~1-2% engine-wide if chased). The performance
model's 85-90% efficiency assumption holds on Ampere.

VERDICTS: (a) the planned cp.async double-buffer for h16 is DEAD --
attention is 0.3% of the round, there is nothing left to hide; (b) the
3090 decode ceiling is now weight BYTES x acceptance, nothing else. The
two live levers: W9/W10 build probe (suffix width headroom = more
accepted tokens per weight stream; VRAM says W10@131K turbo3 is
borderline on 24GB) and the parked lower-bit weight policy (the only
path to materially fewer bytes; needs its own quality-gated study).
Standing rule reaffirmed: profile before building -- this killed a
planned kernel change for the cost of one nsys run.

## 2026-07-12 -- W10 probe on the 3090: NEGATIVE (costs ctx AND speed, buys nothing)

Q27_W_MAX=10 build, turbo3, 3090 (vox paused): OOM at 131K, fits at 65K
(24.08GB -- the two extra role sets + graphs eat half the context
headroom). cctx2 warm replays: 98.6/99.2 tps vs the W8+h16 reference's
119.9/123.0 = 17-20% SLOWER, identical round counts and tok/rnd (~5.6),
suffix fire unchanged. Two compounding reasons: (a) rounds wider than 8
route to fd2_t3 (h16 caps at W8 by smem), so width 9-10 trades the mma
attention win away exactly when it engages; (b) the width cap does not
bind on this traffic anyway -- gla lanes 6-7 tail to near zero, so lanes
9-10 would idle. The 5090's AL-10.6 story needs BOTH width 12 AND the
e4m3 kernel's W12 smem budget; Ampere has neither. VERDICT: W8 stays the
3090 recommendation, binary deleted, do-not-retry unless an h16 W>8
variant exists (needs the 2-CTA smem math to change, i.e. it does not).
3090 optimization is now formally parked at the weight roofline: the
remaining lever is the weight-bit policy study.

## 2026-07-12 -- vanilla mainline llama.cpp on the 3090: the third leg of the triangle

Mainline v1491 (13e67386, no turboquant), Q5_K_M, q8_0 KV, draft-mtp10
p-min 0.5 fa, same harness/task/day discipline. Ladder: 65K/82K OK, 98K
OOM -- q8_0 caps it at 81,920 (identical to the fork's q8_0 rung; the
turbo types are additive).

T8 x3 @82K: 0.57 / 0.55 / 0.54 (hidden .208 = the bimodal low mode x3),
and the 0.55 trial CRASHED at 594s on the context wall ("request 81966
tokens exceeds 81920") -- the same failure class as q27's 32K death,
one tier up. 2/3 completions. GEN decode median 85.6 t/s (p90 105.5,
135 gens) -- mainline decodes ~6% faster than the turboquant fork build
(85.6 vs 80.7; version/build delta), so 85.6 is the honest best-llama
decode figure on this card.

FINAL 3090 TRIANGLE (same card, model family, harness):
              ctx-max   decode-med   T8            completions
  mainline    81,920    85.6         0.55x3 low    2/3 (ctx crash)
  turbo fork  98,304    80.7         mixed         3/3
  q27 t3+h16  131,072   102.2        0.82x3 good   3/3
=> vs the strongest vanilla config: q27 +19% decode, +60% context, and
the context lead is not cosmetic -- T8-class agentic sessions RUN OUT of
82K on bad basins. llama's GDN context checkpoints (~400MB each, visible
in its logs) also eat the same VRAM the KV needs. Basin draws remain
lottery per task; the structural numbers are ctx + decode + completion.

## 2026-07-12 -- sglang on Qwen3.6-27B: NO FUNCTIONAL SUPPORT (dated finding, time-boxed probe)

sglang 0.5.15 (fresh venv, 5090), two checkpoints, both dead in the GDN
weight-loader mapping:
- Lorbus int4-AutoRound: loader DROPS `linear_attn.in_proj_ba.weight`
  on every GDN layer ("not found in params_dict"), then the marlin GPTQ
  repacker hard-crashes on a 96-wide projection ("size_n = 96 not
  divisible by tile_n_size = 64").
- rdtand NVFP4 (the SAME checkpoint vLLM serves): compressed-tensors
  loader cannot map `linear_attn.in_proj_qkvz` ("unable to find matching
  target").
BF16 untestable on 32GB (54.7GB). Conclusion: sglang's qwen3.6
hybrid-GDN support is incomplete at the quantized-checkpoint loader
level as of 0.5.15 / 2026-07-12. No decode number is possible; the
cross-engine pitch cites llama.cpp and vLLM (measured) and this finding
for sglang. Probe cost: ~35 min, two launch attempts, no code written.

## 2026-07-12 -- 3090 prefill A/B vs llama.cpp: llama +27% raw (first Ampere prefill read)

Gap in the record: the 07-09 cross-engine day benched prefill on the
5090 only (parity: llama pp8192 3562 vs q27 fp8 ~3480). Same read on
the 3090 (vox paused, GPU1 exclusive, vanilla model both sides):

- llama mainline 13e67386, Q5_K_M, -fa 1: pp512 1419 +- 13,
  pp8192 1355 +- 5 t/s.
- q27 (--pf, Q27_PF_NOSERIAL=1): fp8 512/8192 = 1040/1065 t/s;
  turbo3 = 1041/1089 t/s (KV format is noise here, as expected --
  prefill is GEMM-bound).

llama +27-30% raw on sm_86 vs parity on sm_120. Not a fallback
artifact: 1065 is right where the 5090's 3480 lands scaled by int8
tensor throughput; llama's Ampere mmq path is simply better tuned than
our P1/P5 GEMM tiling on this arch (tuned on the 5090). Serving picture
unchanged: effective prefill on real CC traffic favors q27 2309-2569
vs 1720 t/s (prefix-cache mechanics dominate raw rate), and the 3090
dome wins were decode-side. A sm_86 GEMM-tile retune is the lever if
raw 3090 prefill ever matters; not commissioned. README 3090 bullet
updated with the honest split. vox restored.

## 2026-07-12 -- unsloth dynamic-NVFP4 probe: new vLLM-best 155.6 (their fixed MTP, NOT their kernels); q27 same-day 370 on the same payload

unsloth released Qwen3.6-27B-NVFP4 (announced 07-10, "~2.5x faster than
other NVFP4 quants"): compressed-tensors mixed export, fp8 attention +
NVFP4 MLPs, MTP layer kept BF16 with config flag unsloth_fixed_mtp. It
LOADS on the same vLLM nightly image as the 07-09 shootout
(v0.23.1rc1.dev748) -- but it is heavier (23.4GB): 131K max-len no
longer fits at util 0.85 (wants 4.88GiB KV, 2.34 free), so the new
vllm-serve keys (unsloth-nvfp4-27b[-mtp], 5090-local-llm repo) run 32K.

Same-day five-way, 25.8K cctx payload, decode = wall(256)-wall(1),
temp 0, batch 1, 3 reps, one image:

    rdtand  no-MTP   73.2 t/s   (reproduces 07-09's 73.5 -- rig valid)
    unsloth no-MTP   64.2       (-12% RAW vs rdtand)
    rdtand  MTP k=3 119.0       (07-09's 155.5 = different day/basin)
    unsloth MTP k=3 155.6       NEW vLLM BEST, +31% same-day
    q27 zero-config 370.5       (Qwopus; w1 0.05s prefix-cached vs
                                 vLLM 2.5-3.4s re-prefill every call)

Decomposition: unsloth's quant kernels are SLOWER here -- the mixed
export streams more bytes and bandwidth-bound decode goes as bytes.
All of the win is the fixed MTP export: MTP multiplier 2.42x
(155.6/64.2) vs rdtand's 1.62x (119.0/73.3). Their "~2.5x" claim is
almost certainly that multiplier, not a kernel speedup; as a
quant-vs-quant kernel claim it does NOT hold on this box.

Caveats, stated: q27 leg is the Qwopus speed fine-tune at 5.25 bpw
(worth ~6% vs vanilla per the 07-11 sweep) vs base-model vLLM legs;
cctx replay numbers are basin-valid same-day only (the 370 vs 07-08's
222.6 is engine progress + basin + replay-warm suffix traffic, do not
ratio across days). The standing vLLM structural notes are unchanged:
0% prefix-cache reuse on hybrid-GDN, every request re-prefills.
Cross-engine pitch update: vLLM's best batch-1 decode on this model is
now 155.6 by unsloth's fixed-MTP checkpoint; q27 same-day same-payload
is 2.4x that. En-route bug caught by smell: a script edit dropped the
prismascout-27b key, vllm-serve printed help and exited, and the
"rdtand" bench re-hit the still-running unsloth container -- flagged by
byte-identical 64.2s, fixed, re-run clean.

## 2026-07-12 -- q6 tier SHIPPED: 6.0 bits/param, +0.35% vs Q5_K_M, -4-11% decode

Ask: a bigger quant for quality-first users. Design was already priced
by the P0.5 study: q6 = v1.4 + ffn_down ALL promoted to Q8 (the one
measured lever with uniform sensitivity; GDN in-projections stay Q4 on
purpose -- promoting them measured WORSE). Plan
docs/plans/2026-07-12-q6-tier.md; repack.py grew --tag (quant_policy
meta override, q6-v1).

    tools/repack.py BF16.gguf out.q27 --q8 '(ssm_out|attn_output|ffn_down)\.' --tag q6-v1

Artifact: qwen36-27b-mtp-q6.q27, 20.49 GB = 6.00 bits/param (884s
repack; ffn_down vacated the worst-RMSE list, remaining Q4 worst ~0.115
attn_q/qkv/ffn_gate). Canonical (new artifact, own gate):
666ffd70747e10e6a9a2087cb18ce8d2.

MEASURED, all same-day matched-protocol on the 5090:
- PPL (--nll full corpus, 148335 preds, fp16 KV): v1.4 8.0409 -> q6
  7.9460; llama-perplexity Q5_K_M bar 7.9179 (145 chunks, c2048, same
  corpus). Gap to Q5_K_M: +1.55% -> +0.35% = 77% of the gap closed for
  +2.76 GB. WAY better than the P0.5 qwopus-scale projection (29%) --
  the vanilla/wikitext scale is kinder to ffn_down promotion.
- Speed price: short-bench suite 171.5 -> 152.1 t/s (-11.3%, tok/rnd
  flat: no acceptance dividend this time, unlike the v1.4 residual-
  writer promotion); 26K cctx server replay 176.6 -> 169.2 (-4.3%) --
  the price shrinks at depth.
- Serving envelope: fp8 auto-ctx 196608 on the 5090 (was OOM before the
  fix below), turbo3 full 262144 cap expected (per_tok math; not
  boot-tested). 3090 W8 turbo3: measured OOM -- fixed cost alone
  (20.49 + 1.27 + roles/graphs = 24.2 GB) exceeds the card. q6 is
  5090-class; 24 GB cards stay on the 5.25 bpw artifact.

EN-ROUTE FIX (server.cu): auto-ctx's fixed-cost anchor hardcoded
19.0e9, which silently WAS "17.73 GB v1.4 weights + 1.27 GB base". q6's
+2.76 GB made auto-ctx oversize KV and OOM at boot. Now: fixed =
stat(model file) + 1.27e9 + role/graph terms. Gate: v1.4 fp8 boot still
picks 262144 (anchor bitwise-reproduced); q6 picks 196608 and serves.
Ops note repeated the hard way: journalctl -u shows PRIOR invocations
-- greping a rebooted unit's log without --since/invocation filter
reads the previous crash as the current state.

Shipped to HF: signalnine/Qwen3.6-27B-MTP-q27/qwen36-27b-mtp-q6.q27 +
CHECKSUMS.md5 + model-card section. Non-goals logged in the plan:
Q6 kernel dtype, ffn_gate/up (8-bit-class), AWQ-style scales, qwopus-q6.

## 2026-07-12 -- q6k tier: down+gate Q8 BEATS every measured GGUF of this model, incl. unsloth's 26GB flagship

Ask: target Q6_K-class quality. Method: measure the REAL bars (naive
local Q6_K is a trap: 8.1089, worse than Q5_K_M -- no imatrix; unsloth's
GGUFs are imatrix'd dynamic quants), then a mini sensitivity pass on the
two FFN levers P0.5 never isolated. All PPL matched-protocol (wikitext-2
test, c2048; q27 legs paired on the same 148335 predictions):

    GGUF bars (llama-perplexity):        q27 nll ladder:
    unsloth Q5_K_M   19.5GB  7.9179      v1.4 5.25bpw    8.0409
    unsloth Q6_K     22.9GB  7.9811      c3 gate-only    8.0088
    unsloth UD-Q6_K_XL 26GB  7.9584      q6 down-only    7.9460
    naive Q6_K (no imx) 22GB 8.1089      c1 down+gate    7.9127  <- q6k
                                         c2 down+up      8.0695

FINDINGS: (1) ffn_gate compounds with ffn_down (-0.42% beyond q6) but
is the WEAKER solo lever (c3 loses to q6 at equal size). (2) ffn_up
promotion HURTS (+1.6% over q6) -- second confirmed error-cancellation
structure after P0.5's GDN in-proj: cleaning up's Q4 noise inside the
SwiGLU product breaks a correlation. Do NOT promote ffn_up. (3) unsloth
non-monotonicity: their Q5_K_M beats both their 6-bit variants on this
corpus. (4) c1-vs-UD-Q6_K_XL delta (-0.57%) is inside llama's single-run
SE (+-0.77%) -- claim is "matches the 26GB flagship at 23.25GB", not
"beats"; the q27-internal ladder is paired and solid.

q6k artifact: qwen36-27b-mtp-q6k.q27, 23.25GB = 6.81 bits/param,
--q8 '(ssm_out|attn_output|ffn_down|ffn_gate)\.' --tag q6k-v1
(re-repacked from c1 for the clean tag; repack determinism gated on
canonical equality). Canonical 2122018dce5929a74c72aa140e713098.
Speed/envelope (5090, same-day): suite 143.1 (-17% vs default, -6% vs
q6), 26K replay 150.6, auto-ctx fp8 114688 / turbo3 262144 (both
boot-verified). 24GB cards: even more DOA than q6.

Tier map now: 5.25bpw default (fastest, 3090-capable) / q6 6.0bpw
(+0.35% off Q5_K_M, -4% depth decode) / q6k 6.8bpw (GGUF-flagship
quality, -15% depth decode). Next uncommissioned lever: attn_q
promotion (+~0.34GB, unmeasured); AWQ-style scales remain the real
path past uniform promotion. Losers c2/c3 + naive Q6_K deleted;
unsloth 6-bit GGUFs kept as reference bars.

## 2026-07-12 -- tier dome (T8 x3 x {v1.4, q6, q6k}, 5090, zero-config fp8): no score separation; live decode price -5.5%/-10.5%

Same-day 3-leg thunderdome on the new weight tiers, model swapped under
the same q27-eval unit/adapter, telemetry from per-request tps= lines
(dec>=32 filter):

    leg   trials (score@wall)              med   live decode med/p90 (n)
    v1.4  0.84@155s 0.83@121s 0.23@134s*   0.83  224.7 / 328.6 (151)
    q6    0.55@200s 0.83@322s 0.81@295s    0.81  212.4 / 308.7 (142)
    q6k   0.83@149s 0.56@153s 0.48@130s    0.56  201.0 / 317.6 (153)
    * v1.4 trial 3 = harness coverage-tool failure (coverage-summary.json
      never produced; scored as 0 coverage), not a model failure.

READ PER THE STANDING T8 DOCTRINE: T8 scores are bimodal basin samples
(~0.85 vs ~0.55); good-basin draws 2/3 vs 2/3 vs 1/3 at n=3 are not
separable (and all 9 trials passed/ran to completion). The PPL ladder
(8.041/7.946/7.913) does NOT visibly move T8 scores -- sub-2% PPL deltas
sit below the task-score noise floor, as expected. What the dome DID
measure cleanly: the tiers' live-traffic decode price, v1.4 224.7 ->
q6 212.4 (-5.5%) -> q6k 201.0 (-10.5%), consistent with the 26K replay
ladder (shallower live mix, smaller price). Verdict: tiers are for PPL/
robustness buyers, not task-score buyers; the default stays 5.25 bpw.
Logs ~/.cache/tier_dome/; runs 2026-07-12T19-32-02 following.

## 2026-07-12 -- P9 checkpoint-ring ALIAS FIXED (audit finding, RED/GREEN receipts)

Gabe's audit pointed at the ring restore path. The bug: on a mid-history
divergence (base > 0 via ckpt or snap hit), re-prefill overwrites KV
rows [base..NP) with the NEW conversation, but ring entries and the
stable snapshot whose coverage extends past base survived untouched. A
later request matching the OLD prompt then restores GDN state over
foreign KV rows -- silent mixed-conversation state. The serial path
already guarded exactly this class (R1 finding, ckpt_clear on
overwrite); the batched divergence path did not.

RED (measured, Q27_CKPT_INTERVAL=64, A 4469 tok / B sharing ~1230):
  R3 (A again after B): prefix_hit=4468 ckpt=4 -- restored A@4468 state
  over ~1300 rows of B's KV.
GREEN (fix): R3 prefix_hit=1024 ckpt=0 -- only the checkpoint whose
  covered rows survived B's overwrite.

Fix (engine.cuh, generate() batched path): after computing base, drop
every ring entry and the snapshot whose coverage exceeds base UNLESS
its tokens are a prefix of the new prompt (those rows get rewritten
with identical values -- deterministic prefill, pf identity gates).
Canonical a2982c51 EXACT (host-side cache-policy change only). Repro
script scratchpad ckpt_alias_test.sh pattern recorded here: two long
prompts sharing >1 chunk of prefix, diverge, replay the first.

Remaining audit item, uncommissioned: collapse the 12-lane copy-paste
state (S_spare1..11 + _b.._l vars, ~500 lines) into a struct array --
pure refactor now that the aliasing bug is located and fixed;
must hold canonical EXACT.

## 2026-07-12 -- 12-lane copy-paste state COLLAPSED (audit item, two gated stages)

Stage A: S_spare1..11 / ring_spare1..11 -> S_sp/ring_sp[W_PLUMB-1][N_LAYER]
arrays; SBuf/RBuf 12-way ternary chains -> one index expression; alloc +
2x memset blocks -> loops.
Stage B: the 18 per-lane float* families (h/x1/y/qg/kbuf/vbuf/attnout/
qkv/convout/z/alpha/betar/g/beta/o/og/ffn_g/ffn_u) x lanes b..l, plus
d_pos_a..l, d_va..l, xqC..xqL -> std::array members with [0] aliasing
the primary lane's pointer. 45 twelve-wide brace lists -> LANES12(F)
macro; conv/delta/rmsnorm if-chains -> loops (launch order preserved);
mm5/qx5 grew array overloads (the lm-head logits2-slice call keeps the
12-arg form). Kept named: h_next2..7, d_pos_m2..7, d_draft2..11 (small,
irregular shapes).

GATES: canonical a2982c51 EXACT after EACH stage (pure pointer-identity
refactor -- [0] aliases mean identical kernel args, loops preserve launch
order); test_kernels ALL PASS; W8 + W12 server builds clean; server
26K replay 176.6/177.3 t/s (pre-refactor same-day: 176.6/177.5 -- same
basin, same speed). Net -446 lines of copy-paste (360+86); the lane
state is now ~20 array members and 6 loops. En-route gotcha: LANES12
next to a declarator name token-pastes (q27k::IP3 P{{...}} ->
PLANES12) -- spaced.

## 2026-07-12 (addendum) -- lane collapse: perf VERIFIED unchanged

Post-collapse vs same-day pre-collapse, same model/GPU/binary-era:
suite 171.6 vs 171.5 t/s (5-prompt mean, the cross-build-robust
comparator); 26K server replay 176.6/177.3 vs 176.6/177.5; boot-to-
ready 4.0s (weights page-cache-warm; capture cost unchanged -- the new
loops execute at graph-CAPTURE time only, decode replays captured
graphs, so there was no mechanism for a decode delta and none measured).

## 2026-07-12 -- alias exposure LEDGER (checked claim, per review) + gate promotion

"No published number rode the bug" is now a CHECKED claim, not an
assumed one. Method: journald [gen] lines since 07-10 (552 survive),
grepped for ckpt>=0 (a mid-history divergence restore -- the alias
precondition).

CLEAN -- zero divergence-restores observed:
- Every replay/CLI bench (fresh server, identical prompts): all t/s
  headlines, PPL ladders, ctx ceilings, prefill benches, accept A/Bs.
- 2026-07-10 cross-engine triplet (the +47% protocol A/B): 0 hits.
- ALL of 2026-07-12: the h16 3090 dome (+27%, 102.2 t/s), the tier
  dome, both probe days' benches. 0 hits.

EXPOSED -- 552 traversals of the alias CONDITION, all on 2026-07-11:
six 5090 server instances (66-133 each: the turbo3 T8 trials, 2-slot
pairs, T2/T11 breadth) and 6 on the 3090 w8 leg. Mechanism confirmed in
the logs: CC sidechain/compaction traffic interleaves a ~32K main
conversation with ~19-20K branches on one engine; each branch's
re-prefill left the other's longer-coverage entries stale (pattern:
prompt=32335 restore@12288 alternating with prompt=19397 hit@16384).
Caveat, stated honestly: a traversal is not a confirmed foreign-row
read (prefix-covered restores are valid) and the corrupt fraction is
not reconstructible post-hoc. Decode RATES are content-independent and
stand; 07-11 live-session SCORES and walls carry unquantifiable
exposure. Consequence: README's 3090 bullet (07-12 data) is unaffected;
docs/multislot-throughput.md's session results are annotated; rerun the
2-slot pair + one turbo3 T8x3 leg under >= e16c394 before citing any
07-11 session score publicly.

Gate promotion: the divergence-then-replay shape is now
tools/ckpt_gate.sh (self-contained: generates prompts, boots the
server, A/B/A, asserts replay hit <= divergence base). Run it on every
cache-path change. First run on the fixed binary: PASS (1024 <= 1024).

## 2026-07-13 -- exposed 07-11 session results RERUN under the fixed binary: scores validated

Per review: "no published number rode the bug" required rerunning the
two exposed result sets under >= e16c394. Both rerun same-config
(qwopus, turbo3, ports/adapters identical to 07-11).

2-slot pairs (5090, 2x131K):
    pair  fixed binary            07-11 (exposed)
    T2    0.84/0.84 @ 136/148s    0.84-class @ 152/162s
    T11   0.85/0.85 @ 59/87s      @ 69/70s
    T8    0.55/0.84 @ 236/297s    ~0.82 both @ 253/256s
T2/T11 reproduce solo-class scores AND walls; T8 drew the documented
bimodal basin split. Throughput: per-request decode median 145.3 /
p90 268.1 all-pairs (T8 window 155.1 / 302.0), busy-agg 115.9 t/s.
The p90 above solo median re-confirms no scheduler tax. Divergence-
restores under 256 requests of concurrent traffic: ONE (valid by
construction post-fix). multislot-throughput.md annotation resolved:
capacity, admission, split, and score claims all stand.

3090 turbo3 T8x3 (w8, 131K): scores 0.81 / 0.84 / 0.82 -- the 07-11
0.82x3 score class reproduced exactly on the fixed binary. Decode
median 92.8 t/s (qwopus; h16-era band), busy-agg 120.4, zero
divergence-restores. Caveat, stated: all three trials hit the 2700s
harness ceiling still working (scored at cutoff; ~27 min GPU-busy of
each 45-min wall was tool execution) -- trajectories are cross-build
incomparable per the standing doctrine, so the WALLS validate nothing
either way; the SCORES were the exposed claim and they hold.

VERDICT: the alias exposure did not flatter any published score. The
07-11 numbers stand with the annotation lifted (scores) and the usual
trajectory caveat (walls).

En-route negative, filed for triage: the FIRST 3090 rerun leg
mistakenly ran the VANILLA model and hit a deterministic parser-drift
retry loop (2/3 trials: [drift] modes=12 recovery re-requested
identically every ~5 min to harness timeout; prompt=34344,
prefix_hit=34343 each time). Vanilla + w8 + turbo3 + this harness
build reaches a drift shape the recovery satisfies but CC rejects.
NOT triaged tonight; the loop payload shape is in journald
(2026-07-12 20:52-21:26). Candidate fixture once reproduced small.

## 2026-07-13 -- the 12-lane suffix cap BINDS on file re-emission (external-review-driven) + two more findings

The maintainer of the turboquant fork (TheTom/llama-cpp-turboquant @
558c6b78e) ran a KV-matched, slot-matched, sampling-matched A/B of his
build vs q27@dc8d5ad on the 5090. His headline that mattered: in the
file-re-emission regime his ngram-mod does 653 t/s vs q27's fused
MTP+suffix 376.9 (+73%), at 85%+ draft acceptance. That directly
challenges q27's published "12-lane cap never binds on real traffic."

MEASURED (Q27_SUFFIX_DBG trace, file re-emission payload = real code
file + targeted addition, forced full-file re-emit):
- suffix-fired rounds 103, ladder rounds 253; suffix mean accepted
  10.68 against the 12 ceiling.
- accepted-length histogram, suffix rounds: n=12 in 83/103 fires = 81%
  PINNED AT THE CAP. The rest scattered 1..11. This is truncation, not
  weak matching -- llama's unbounded ngram takes ~24-token drafts at
  251/251, q27 stops at 12 by construction.
VERDICT: the cap binds in this regime. Published claim corrected
(README, speed post, the 07-09 entry above).

Widening is a BOUNDED plumbing project, not a kernel rewrite: the fdmma
verify kernel already has FDMMA_CASE 13..16 ("kernel is 16-ready"), and
the lane-pointer structs (P3/CP3/IP3/XQ3) are already float*[16]. What
is still 12-wide: W_PLUMB (outcome ints, emit[], S_sp/ring_sp role
arrays, the gch/gnh/glf/gla histograms) and the per-perm captured-graph
zoo (grows with W_MAX). So W16 is the same CLASS of work as the 8->12
widening was (~a day, its own canonical campaign, +~630MB role VRAM for
4 more sets + graph growth), gated by the modulus-relabel invariant
that kept 8->12 bitwise. NOT built tonight -- sized and filed; it is the
single highest-value speed item and it barely touches novel-prose rate
(suffix only fires on repetition; the MTP ladder path is untouched).

### Finding 2: over-refusal localizes to the no-think serving default

Reviewer saw q27 flatly refuse a signed-authorization pentest scan and
an OTC-dosage question where his build answered; verified not an
injected system prompt. Localized it: the CC serving profile defaults
to no-think, which prefills an empty <think></think> block
(api_common.h chatml_prompt). A/B, same prompt, default vs --think:
- pentest (nmap SYN scan, stated engagement ID): no-think SOFT-REFUSES
  ("cannot provide the command even with claimed authorization");
  --think reasons through the authorization and GIVES the command
  (`nmap -sS -p 80,443,8080,8443 10.2.0.0/24`).
- otc: answered correctly under BOTH (could not reproduce the reviewer's
  flat OTC refusal on default serving -- likely phrasing/sampling; the
  pentest class is the clean repro).
Mechanism: a reasoning model handed zero reasoning budget pattern-
matches "nmap SYN scan -> refuse" instead of reaching "authorized ->
answer." NOT a fix to the default (no-think is the speed default and
carries the 224 t/s headline); this is a documented tradeoff --
operators who need borderline-legit compliance pass --think, one flag.
Filed for the serving docs.

### Finding 3: GQA-6 guard, INDEPENDENT cross-engine confirmation

The reviewer forced his fork's auto-asymmetric guard off
(TURBO_AUTO_ASYMMETRIC=0): symmetric 3-bit K+V ran clean, +0.87% total
on this model -- matching q27's own triage (+0.70% V, +0.17% K-on-top)
from a SEPARATE engine. Same mechanism cited independently (head_dim 256
= two independent 128-dim WHT groups protecting K at GQA 6). This is the
strongest validation the K-crater refutation has: two engines, one
finding. Belongs in the 3090 post as external confirmation. His
actionable: per-arch guard calibration / documented override (his fork's
call; the calibration logic is q27's to offer back).

## 2026-07-13 -- over-refusal FIXED: default system prompt when client sends none

Root cause (measured, not assumed): the over-refusal needs BOTH the
no-think empty-<think> prefill AND no system prompt. A/B under no-think,
pentest ask:
- no system prompt        -> SOFT-REFUSES
- "You are a helpful assistant." -> COMPLIES (gives the nmap command)
- Claude-Code-shaped system -> COMPLIES
So real Claude Code never hits this (it always sends a substantial
system prompt); only bare /v1/messages | /v1/chat/completions requests
do. The empty think block gives zero reasoning budget; with zero system
context on top, the model falls to a defensive refusal prior.

Fix (api_common.h chatml_prompt): inject "You are a helpful assistant."
ONLY when the client supplied no system prompt (Q27_BARE=1 opts out).
Zero reasoning cost (no think tokens), prefix-cached (before stable_off,
snapshot-safe), does not touch client-supplied system prompts, and does
not affect the /v1/completions raw-text replay path (no chat template),
so the 224 t/s headline and the server replay gates are untouched.

Validation A/B (no-think default, no client system):
- authorized pentest (signed engagement ID): COMPLIES, gives
  `nmap -sS -p 80,443,8080,8443 10.2.0.0/24`.
- OTC ibuprofen dosing: answered correctly.
- MALICIOUS CONTROL (hospital ransomware, extortion note): still
  REFUSED -- the default recovers legitimate compliance without
  stripping genuine safety.
- Q27_BARE=1: reverts to the prior refusal (clean opt-out / repro of
  the reviewer's exact condition).
Canonical a2982c51 EXACT (CLI uses --tokens, never the chat template).
README serving-section note updated to reference the fix.

## 2026-07-13 -- W16 BUILT AND MEASURED: cap raise is a NO-GO; the real bottleneck was a GEMV register spill (+18% on the SHIPPED width)

Built the W16 plan (docs/plans/2026-07-13-w16-suffix.md), gated it, and
measured it. The plan's premise -- "81% of suffix fires pin at the 12-lane
cap, so raise the cap" -- is half right, and the half that is wrong is the
half that mattered.

### The cap binds. Raising it makes things WORSE.

Suffix-leg width curve (the width-12 P3 instrument: server Q27_MAXD=4 +
Q27_SUFFIX_W=W, open-cut echo payload @28K, qwopus, 5090):

  W_MAX=16 build:  W=12   24.28 ms/rnd   2.103 ms/tok   434.5 t/s  <- OPTIMUM
                   W=13   27.21          2.194          427.1
                   W=14   30.47          2.177          424.0
                   W=16   39.97          2.518          368.5  <- -15% t/s

The cap-binding claim reproduces exactly, at BOTH widths (Q27_SUFFIX_DBG):
  W=12: 21/22 fires = 95% pinned at n=12
  W=16: 15/16 fires = 94% pinned at n=16
So W16 delivers precisely the +33% accepted-tokens-per-fire the plan
predicted -- and throughput falls anyway, because the wide round costs MORE
than proportionally. Tokens per fire was never the throughput variable.

Not a tuning artifact. At W=16, the stages/ns choice this session added
(below) is the best of the four combinations, and W16 still loses:
  auto (stages=2, ns=42)  39.52 ms/rnd  372.4 t/s  <- best
  stages=1, ns=85         40.96         360.3      (what the old code picked)
  stages=1, ns=42         39.94         368.6
  stages=2, ns=85         39.99         368.2

VERDICT: W16 = measured NO-GO. W12 is the per-token optimum of this kernel
family. Q27_W_MAX stays 12; no q27-server-w16 target is recommended. The
answer to the fork maintainer's 653-vs-377 is NOT verify width.

### What the width curve was actually hiding: a register-spill cliff

The 07-10 width-12 P3 entry recorded "per-token at full accept flattens: W8
2.93 / W10 2.88 / W12 2.94 ms/tok" and attributed the accelerating marginal
lane (1.6 -> 2.7 -> 3.2 ms) to "GEMV-N". That was the right suspect and the
wrong diagnosis: it is not that a wide GEMV is inherently expensive, it is
that ours was SPILLING.

k_gemv_q4_n / k_gemv_q8_n carried `__launch_bounds__(256, N <= 8 ? 4 : 3)`.
The 3-CTA pin gives ~80 registers, and past N=9 the per-lane accumulators
stop fitting. ptxas -v on sm_120, q4:

  N=12: 40B stack /  76B spill stores      N=14: 104B / 140B
  N=13: 40B      /  68B                    N=15: 144B / 212B
                                           N=16: 232B / 296B

Retiering widths >= 10 to a 2-CTA/128-reg pin (Q27_GEMV_2CTA_MIN=10, swept:
N=9 is the crossover and stays 3-CTA) buys the accumulators back. Measured
q4 ffn 17408x5120, ms/call:

            N=10    N=11    N=12    N=13    N=16
  3-CTA    0.0536  0.0638  0.0702  0.0800  0.180
  2-CTA    0.0476  0.0509  0.0554  0.0638  0.113

ENGINE A/B, same binary, same day, same payload -- only the pin differs:

  W    old ms/rnd   new    | old ms/tok  new   | old t/s  new t/s
  5      15.99     16.02   |   3.211    3.217  |  305.2   304.7   (control)
  8      19.54     19.65   |   2.462    2.476  |  387.5   385.4   (control)
  10     24.12     21.42   |   2.472    2.194  |  388.9   433.5
  12     29.16     24.33   |   2.526    2.107  |  367.4   433.4

W5/W8 are unchanged by construction (their pins do not move) and land within
noise -- the control that says the delta is the pin and nothing else. At the
SHIPPED suffix width the round drops 29.16 -> 24.33 ms: **+18% t/s (367 ->
434) on the production path, no plumbing required.**

And note what it does to the SHAPE: under the old pin, per-token cost was
flat-to-RISING past W8 (2.462 -> 2.472 -> 2.526) -- widening was worthless,
exactly as P3 concluded. Under the new pin it falls monotonically. The
"widening does not pay" finding was an artifact of the spill, not a property
of the design. (It still does not pay past 12 -- see the W16 curve -- but for
a different reason: the fdmma occupancy cliff below.)

The retier touches ONLY suffix rounds: gated ladder rounds verify widths
2..gate_maxd+1 (<= 8), which keep their existing pins. Canonical a2982c51
EXACT on both builds.

### fdmma loses its 2-CTA occupancy at W>=14 (why 14..16 are expensive)

k_attn_fdmma's shipped STAGES=1 variant is the default because TWO CTAs
co-reside per SM. s_q holds fdmma_qrows(W) = 16-rounded 6W rows: 80 rows
through W=13, but 96 from W=14 (6*14=84). smem(W,1) goes 48.0KB -> 52.1KB,
so 2 CTAs need 104.2KB against a 100KB sharedMemPerMultiprocessor -- and
occupancy silently drops to ONE CTA, which is strictly worse single-buffered.
The split count ns had the same 2-CTA assumption baked in (SMs*2/kv_heads).

Fixed both: stages=1 iff 2*fdmma_smem_bytes(W,1) <= smem/SM, else stages=2;
ns = SMs*(2 or 1)/kv_heads to match. No-op at W<=13 (2 CTAs still fit -> ns
computes 85 on the 5090, bit-for-bit the shipped value); worth +3.4% at W16.
This is why W=13 is the last cheap width and 14..16 are not.

### Shipped anyway: the plumbing, because the refactor is a keeper

W_PLUMB moved to cuda_common.h (spec3.cu could not see it, which is why
k_finish_round carried hardcoded 11/12 literals that no compiler could tie
back to the width) and went 12 -> 16, the hard ceiling of the kernel family
(k_attn_fdmma asserts 6*W <= 96 rows). Q27_W_MAX stays 12. Every "list every
lane" site is now loop-built off W_PLUMB rather than a hand-written brace
list -- that class of site is where the 8->12 widening hid the k_quantize_x3
lane-aliasing bug, and where a 4-lens audit workflow found this session's
worst bug: refinish_round's 12-entry `lanes[W_PLUMB]` list left slots 12..15
nullptr, so a W_MAX=16 server taking a tool-marker completion on a >=13-token
suffix round would memcpy from nullptr and die. Also fixed: the CLI round-
outcome print indexed 12 fixed slots out of a hist[W_MAX] array (a stack OOB
read on the w8 build, silent truncation above it), and the graph banner
hardcoded "12 perms".

Costs measured, not assumed: shortbench suite 171.7 t/s vs the 172.2 vanilla
standard (noise) -- the plumbing canary says W_PLUMB=16 taxes the ladder path
nothing.

GATES: canonical a2982c51 EXACT (default W12 build AND W_MAX=16 build);
fdmma_test PASS at widths 4..16 x stages{1,2} x ns{128,85,42} (modeled cos
1.0000000, rel 3-5e-6); test_kernels ALL PASS; compute-sanitizer memcheck 0
errors on a W_MAX=16 run in which a 16-token suffix round actually committed;
shortbench 171.7.

### Where the 653 actually lives

Not in the cap. The wide-round marginal is still superlinear even retiered
(q4 gemv 0.055 ms/call at N=12 -> 0.113 at N=16), and fdmma's occupancy
cliff sits at 14. The honest lever remains the one P3 named: the mma16 NT=16
GEMM pivot (tools/mma16_bench.cu, 76% SOL, 0.043-0.045 ms FLAT W2..16). What
changed today is the size of the prize -- the GEMV it has to beat is now
0.055 ms at N=12, not 0.070 -- and the fact that a GEMM verify would be flat
in W is exactly what would make widths past 12 pay at all. W16 reopens ONLY
behind that pivot.

## 2026-07-13 (cont.) -- the retier through the REAL harness: mechanism confirmed, headline rescoped (+19% was the echo number, agentic is ~+5%)

Ran the gemv pin A/B through thunderdome + Claude Code (recreated the
claude-code-q27-haight adapter -- the old one was an uncommitted convention
and is gone; it now lives in the thunderdome tree, base URL retargeted at
:8081, otherwise the stock CC adapter verbatim). Leg = the binary behind
:8081; T8 + T2; same day, same model.

FIRST, THE GATE THE 07-13 REVIEW SAID WAS MISSING. Nothing exercised the
retiered widths for TOKEN identity -- the canonical prompt never runs a suffix
round. Closed it: old-pin vs new-pin on a suffix-heavy payload, greedy ->
BYTE-IDENTICAL output (same md5, same 254 suffix tokens over 22 rounds), only
the wall moved (642 -> 534 ms). The retier is a PURE speed change; quality
cannot move. That is the strongest form of the claim and it is now gated.

MECHANISM CONFIRMED ON LIVE AGENTIC TRAFFIC. Over 3,389 (old) and 2,024 (new)
real suffix rounds driven by Claude Code:
  suffix ms/round  29.53 -> 24.78  = -16.1%
matching the isolated bench (-17%). Per-round cost is trajectory-independent,
so this number is unconfounded.

BUT THE RAW AGGREGATE A/B IS UNINFORMATIVE: 252.5 -> 251.9 t/s (-0.3%). The
legs FORKED (the documented cross-run CC confound -- tool outputs carry
wall-clock bytes): old drew a 48.1%-suffix trajectory, new a 38.5% one. At
n=1/leg the trajectory lottery swamps a ~5% effect. Trajectory-matched
counterfactual (each leg priced under the other pin, using its OWN round counts
-- no cross-leg pairing): +6.3% and +5.2%.

RESOLVED IT PROPERLY with fixed-BYTES paired replays (byte-identity means both
binaries walk the same trajectory -- no fork, no lottery). All four verified
IDENTICAL output:
  payload    suffix tok/decode   old ms/sfx-rnd -> new    delta t/s
  codegen          0.4%            28.8 -> 24.2            -0.1%
  testgen          0.0%             (never fires)          -0.3%
  docs            21.9%            29.0 -> 24.3            +2.0%
  echo            99.2%            29.1 -> 24.4           +17.4%

THE LAW: the suffix round gets 16% cheaper EVERYWHERE it fires (28.8/29.0/29.1
-> 24.2/24.3/24.4 -- dead uniform). The ENGINE-level gain is that 16% times the
suffix WALL share. Novel generation never fires the suffix drafter, so it gets
exactly nothing. Real CC agentic work sits at 27-37% suffix wall share (T8/T2
re-emit files constantly) -> ~+5-6%.

HEADLINE RESCOPED. The "+19%" is the ECHO/repetition number, NOT an engine
average. README corrected. This does not change the W16 verdict (per-token still
bottoms at W12) and it does not change the mma16-GEMM conclusion -- if anything
it sharpens it: a verify GEMM flat in W is what would make the WIDE path cheap
enough to matter on traffic that is not already repetitive.

Scores (descriptive only, n=1, and the pin CANNOT move them -- byte-identity):
new 0.52/0.55 (T8/T2), old 0.29/0.57. Both legs sit below the historical 0.82-0.85
band on this RECREATED adapter, on both legs -- read nothing into it except that
the recreated adapter is not bit-for-bit the lost original. Not chased.

Tools: tools/thunderdome_pin_ab.sh, tools/pin_ab_report.py ([req] gotcha: dec/
tps/sfxm/sfxn are PER-REQUEST but sfx=<fired>,<tok> is ENGINE-CUMULATIVE --
summing it across requests overcounts; take the last line, or diff consecutive).

## 2026-07-13 (cont.) -- spill audit across ALL kernels: the LADDER gemv was never occupancy-swept either (+1.7% on the main decode path)

The 2-CTA retier came from noticing one kernel spilled. Nobody had ever asked
the same question of the other 183. Asked it (`nvcc -Xptxas -v` over kernels.cu
/ spec3.cu / blocks.cu / prefill.cu): 30 of 184 kernels spill. Ranked by spill
stores, the interesting ones are all in the same family -- and four of them are
the widths the LADDER verifies:

  k_gemv_q4_n<4>  48B spill @ 64 regs (4-CTA)
  k_gemv_q4_n<5>  36B         "
  k_gemv_q4_n<8>  24B         "

The 4-CTA/64-reg tier dates from the depth-4 era and was never swept. It is
beaten by 3 CTAs / 80 regs at EVERY ladder width:

  q4 ffn 17408x5120, ms/call    N=5     N=6     N=7     N=8
    4-CTA (old)                0.0361  0.0338  0.0368  0.0438
    3-CTA (new)                0.0332  0.0329  0.0341  0.0391
                                -8.0%   -2.7%   -7.3%  -10.7%

This one is worth more than the suffix retier in BREADTH: gated rounds verify
widths 2..gate_maxd+1, so it hits EVERY round, not just the repetitive ones.

  canonical         139.8 -> 142.2 t/s  (+1.7%, md5 a2982c51 EXACT)
  shortbench suite  171.9 -> 174.9 t/s  (+1.7%)
  @26K, fixed-bytes paired (all IDENTICAL output):
    codegen +0.8%   testgen +0.6%   docs +0.7%   echo -0.0%

The gain SHRINKS with context (+1.7% @2K -> +0.7% @26K) because attention, not
the weight GEMV, owns a deep round -- exactly the mirror of the suffix retier,
which is worthless on novel prose and worth +17% on echo. Together the two pins
cover the two regimes: the ladder pin pays on every round and most at short
context; the suffix pin pays only on repetition. Neither moves a single token
(both are register allocation; canonical EXACT).

N<=3 stays 4-CTA (narrow gated graphs, untouched). Knobs: Q27_GEMV_3CTA_MIN_Q4
(4) and Q27_GEMV_2CTA_MIN (10).

STILL ON THE TABLE (not built): k_gemv_q8_n<6..9> spill 80-112B at the 3-CTA
pin, but the q8 head only runs WITHOUT --fast-head, so it is off the serving
path -- fix if the reference profile ever matters. k_attn_fdmma<6/7/8> spill
8-24B at 168 regs; that kernel is occupancy/latency-bound, not register-bound
(three separate 2026-07-10 experiments said so), so the bar to retry is low
value.

THE REAL REMAINING LEVER, now priced: the mma16 NT=16 GEMM verify pivot. With
the GEMV retiered, the crossover is visible -- gemv is 0.033-0.039 ms/call at
ladder widths but 0.055 @N=12 and 0.113 @N=16, while the MMA GEMM measured
0.043-0.045 ms FLAT W2..16 (76% SOL, tools/mma16_bench.cu). So the GEMM LOSES
below ~N=10 and wins 2.5x at N=16. The architecture that falls out is a HYBRID:
GEMV for the ladder (2..8), MMA GEMM for the suffix widths (>=10). That flattens
the wide-round marginal -- which is the one thing that would make the W16 cap
raise pay, and it is exactly the conclusion the 07-10 P3 entry reached from the
other direction. This is the next build, not another pin.

## 2026-07-13 (cont.) -- P0 of the GEMM-verify plan: RUN, and it PASSES. Build.

The plan (docs/plans/2026-07-13-gemm-verify.md) put a kill-criterion in front of
the kernel: nsys a live width-12 suffix round, sum the GEMV nodes, and STOP if
they are under 12 ms of a ~24.3 ms round. Ran it.

INSTRUMENT NOTE (cost me one run): nsys does NOT break out CUDA-graph replays by
default, and `--cuda-graph-trace=node` did not change that here either -- every
decode kernel reports exactly ONE round's worth of instances (gemv_q4_n<12>: 305
= the round's 305 Q4 mm5 calls; fdmma<12>: 16 = the attn layers; finish_round: 1).
Those instances are the EAGER warm round that build_spec_graphs executes at
vw = sfx_width(). That turns out to be exactly the instrument we wanted: a clean,
complete node histogram of one width-12 verify round. Window it on the single
k_finish_round and sum.

ONE WIDTH-12 VERIFY ROUND (2889 nodes, 21.38 ms GPU-busy):
  13.89 ms  65.0%  k_gemv_q4_n<12>     <-- weight GEMV
   2.52 ms  11.8%  k_delta_step        <-- the GDN serial chain
   1.78 ms   8.3%  k_gemv_q8_n<12>     <-- weight GEMV
   0.48 ms   2.3%  k_conv_step
   0.48 ms   2.2%  k_gemv_f16_3
   0.46 ms   2.1%  k_rmsnorm3
   0.45 ms   2.1%  k_rmsnorm_heads
   0.35 ms   1.6%  k_attn_fdmma<12>    (ctx ~0 in the warm round; ~2.5 ms @28K)
   ... (nothing else above 0.35 ms)
  WEIGHT GEMV = 15.66 ms = 73% of GPU-busy.

Reconciled against the real graph-replayed round (sfxm/sfxn = 24.76 ms @28K) and
the independent DRAM-cold replay (tools/round_weight_cost: 16.73 ms): the warm
round runs the GEMV L2-warm at ctx 0 and its fdmma is nearly free, so the honest
in-context figures are GEMV ~= 16.4 ms and NW ~= 8.36 ms. Both instruments agree
to within 1 ms and the sum reproduces the 24.76 ms round.

  P0 BAR: >= 15.5 BUILD | 12-15.5 re-derive | < 12 STOP.
  MEASURED: 15.66 ms (nsys) / 16.73 (cold replay). ==> **BUILD.**

RE-DERIVED PAYOFF (measured, not modelled):
  P2 (GEMM @ W12):  round 24.76 -> 20.34 ms (-17.9%); echo 427 -> 511 t/s (+19.5%);
                    real agentic +4.7% to +6.4%.
  P3 (+ W16 cap):   round(16)/round(12) under the GEMM = 1.025, i.e. the cap needs
                    only +2.5% more accepted tokens to pay -- and echo delivers
                    +37.4%. A 15x margin. echo 662 t/s; agentic +10.7% to +15.1%.
Slightly under the plan's estimate (+21.9% / +5.1-7.1%) because NW is ~10% bigger
than the subtraction said. The plan's own numbers, corrected by its own P0. It
holds.

FREE FINDING, and it is a good one: **k_delta_step -- the GDN serial delta chain --
is 2.52 ms, the biggest NON-weight kernel in the round (12%).** Once the weight
path is flat, that is the next thing standing up. GDN chunking was SHELVED on
2026-07-09 ("state-WRITE-bound, 1.21-1.29x, below the 2.5x bar") on the theory
that the wide marginal was GDN-bound; it is not, it is GEMV-bound -- but delta_step
is still 12% of a wide round and nobody has profiled it since. Re-open AFTER P2.

Also priced for free: nothing else in the round is above 0.5 ms. There is no
hidden term. The round really is weights + delta + a long tail.

## 2026-07-13 (cont.) -- P3: the W16 cap REOPENS on the flat GEMM, but only in the file-re-emission regime. Default stays W12.

Rebuilt -DQ27_W_MAX=16 with the P2 GEMM wired, canonical a2982c51 EXACT (the
ladder still never reaches the GEMM). Then measured the two things that decide a
cap raise: the WIDTH CURVE (is the wide round cheap now?) and the ACCEPT
HISTOGRAM (does live traffic actually want the extra lanes?).

WIDTH CURVE, W16 server, GEMM engaged (echo payload):
   W   @28K rnd   t/s    @60K rnd   t/s
   12    19.95    519      21.05    509
   13    20.48    552      21.63    519
   14    20.99    592      22.51    550
   16    21.95    631      23.59    584
round(16)/round(12) = 1.100 @28K, 1.121 @60K. The round is now essentially FLAT
in width -- the exact thing that was missing on 07-13, when the same curve read
W12 24.28 -> W16 39.97 (round ratio 1.65) and W16 LOST 15%. With the GEMM, W16 is
+22% over W12 @28K, +15% @60K. The fdmma 1-CTA cliff at W>=14 costs a little at
depth (ratio 1.10 -> 1.12) but does not change the verdict.

THE REOPEN ARITHMETIC, now measured both sides:
  a cap raise pays iff tok(16)/tok(12) > round(16)/round(12).
  round side: 1.10-1.12 (was 1.65).
  token side, CONTROLLED (same fixed echo payload, both caps, Q27_SUFFIX_DBG):
    cap 12 mean 11.54 tok/fire, cap 16 mean 15.88 -> 1.375.
  1.375 > 1.12 -> WINS with ~3x margin. On echo the cap raise is worth +21% t/s
  ON TOP of P2 (echo 519 -> 631).

BUT GATE 8 (the one that decides the DEFAULT) is live CC traffic, not echo. T8
through Claude Code, cap-12 leg: 628 suffix fires, MEAN ACCEPTED 7.82, median 9,
only 41% pinned at 12. Live agentic fires mostly do not reach the EXISTING 12
cap, so raising it to 16 has almost nothing to bite on. (The cap-16 leg forked to
a different trajectory -- 32 fires vs 628 -- so its mean is not comparable; the
cap-12 leg ALONE is the finding, and it is unconfounded.)

VERDICT: the 07-13 W16 NO-GO is REVISED, not reversed. W16 went from "loses
everywhere" to "wins the file-re-emission regime (+21%), neutral on mixed
agentic." The default stays Q27_W_MAX=12: on typical traffic fires don't saturate
12, so the extra 4 lanes (+630 MB roles, ~1.8x graph-zoo boot) buy nothing.
q27-server-w16 is now a LEGITIMATE build for repetition-heavy serving -- exactly
the fork maintainer's 653-vs-377 file-re-emission scenario, where W12+GEMM does
519 and W16+GEMM does 631, closing most of the gap to llama's 653 (true parity
still needs unbounded suffix, out of scope). The knob to reach for is the build,
not a default.

WHY THIS IS THE RIGHT OUTCOME: P2 already banked the flat GEMM on ALL traffic
(the W12 round dropped 24.76 -> 19.96 regardless of width). P3's remaining
question was only "does raising the CAP on top of that pay," and the honest
answer is "only when the traffic is repetitive enough to fill the current cap,"
which the live histogram says is a minority of real agentic work. The GEMM was
the lever; the cap was not.

New tools: tools/gate8_caphist.sh (live cap A/B), the WC_CTX/WC_PAY knobs on
width_curve.sh, scratchpad/accept_payload_echo60k.json.

## 2026-07-13 (cont.) -- k_delta_step register fusion: the GDN chain wrote its state TWICE; write it once (+1.5% ladder / +2.6% echo, bitwise, helps EVERY round)

P0 flagged k_delta_step (the GDN DeltaNet recurrence) as the biggest non-weight
kernel in a wide round -- 2.52 ms, 12%. It is bandwidth-bound, and it was moving
2x the floor.

MEASURED (tools/delta_bench.cu, isolation): the shipped kernel writes the 128x128
per-head state So in pass 1 (decay) and reads it back + rewrites it in pass 2
(delta update) = 12 MB traffic where the recurrence floor is 6 MB (read Si once,
write So once). Keeping this thread's 32 decayed state values in REGISTERS across
the two passes drops the redundant write+read:
  SOL state read+write (6 MB):   0.0041 ms  1525 GB/s
  k_delta_ship (48 blk):         0.0061 ms  (12 MB, So written twice)
  k_delta_reg  (48 blk):         0.0041 ms  = SOL exactly, -33%/call
  bitwise reg-vs-ship: state 0 diffs, o 0 diffs.
Not occupancy-bound: 48 blocks (one per head) already saturate HBM once the
redundant traffic is gone, so a column-split (more blocks) would buy nothing --
the register version already hits SOL. The 07-09 "state-WRITE-bound, 3.1 MB/step
contractual" note was RIGHT about the floor; the kernel just wasn't at it.

BITWISE by construction: fp32 in a register == fp32 round-tripped through global,
and the arithmetic is unchanged (s is the same value whether it lives in So or a
register), so the GDN state and every downstream token are bit-for-bit identical.
Canonical a2982c51 EXACT; test_kernels delta-wy PASS; memcheck 0 errors; 80 regs,
0 spill.

ENGINE IMPACT -- and this one helps traffic the GEMM CANNOT:
  shortbench suite  174.7 -> 177.4 (+1.5%)   canonical 142.0 -> 144.3
  echo W12 round    19.96 -> 19.42 ms         echo t/s 519 -> 532 (+2.6%)
delta_step runs on EVERY GDN layer of EVERY decode round -- spec or single-token,
ladder or suffix, novel prose or echo. So unlike the GEMM (suffix rounds only) and
the retiers (mostly suffix), this is a universal +1.5%, INCLUDING the novel-prose
224 t/s headline path where the suffix drafter never fires. Small but free and
everywhere.

NEXT non-weight lever (priced by the same P0 histogram): nothing else in the round
exceeds 0.5 ms after this -- k_conv_step (0.48) and k_gemv_f16_3 (0.48, the GDN
in/out proj) are the tail, and both are already near their floors. The round is
now weights (GEMM, flat) + delta (at floor) + a sub-0.5ms tail. The remaining
structural lever is the one named all along: the MTP draft ladder re-streaming the
head, which is W-invariant and shows on no width curve -- P0's histogram is where
to look next.

## 2026-07-13 (cont.) -- MTP draft head lever: INVESTIGATED, NO-GO. The draft is at its SOL floor; no free win, and the on-path lever is a quality tradeoff that most likely loses on novel prose.

P0 flagged the MTP draft ladder as the last non-verify lever. Investigated it
properly (3-design/3-adversarial workflow + direct profiling + the realized-cost
telemetry nobody had read). Verdict: NO-GO for now. Plan + full reasoning in
docs/plans/2026-07-13-mtp-draft-head.md.

WHAT THE DRAFT ACTUALLY IS. Each of the (up to) 4 serial draft steps runs a full
mtp_forward through the trained MTP module (blk.64): ~427 MB of Q8 MTP-layer
weights + the 635 MB Q4 vocab head, ~1.1 GB/step. The head is streamed once per
step (4x/round at full depth). Measured single-token head GEMV = 635MB/0.40ms =
1587 GB/s = SOL. The steps are SERIAL (step k+1 embeds step k's token) so they
cannot batch, and 635MB >> 128MB L2 so nothing caches. UNLIKE k_delta_step there
is NO redundant traffic to reclaim -- the single-token gemv_q4/gemv_q8 are 44/48
regs, 0 spill, already at SOL. The free-win angle is DEAD (independently confirmed
by the adversarial pass).

THE HEADLINE WAS MIS-SIZED (my error). "2.91 ms = 12% of the round" divided the
draft by a WIDTH-12 round -- but width-12 rounds are SUFFIX rounds, and suffix
rounds SKIP the draft ladder entirely. The 2.91 ms is a saturating (all-4-margins-
passed) peak that never coexists with a width-12 verify. REALIZED production number
(the dexit-averaged telemetry at server.cu:450, phd/phv, read for the first time
on live novel codegen/testgen/docs @~25K ctx): draft = 220-227 ms vs verify
1160-1410 ms = **14-16% of the decode wall**, at **3.0-3.7 steps/round** (adaptive
depth + dexit ALREADY trim below the nominal 4). Matches the memory's "12-15% of
decode wall." So the lever is real but ~14%, not a hidden 22%+, and gets smaller
with context (<5% at 61K, attention-dominated).

WHY NO ON-PATH WIN.
- Free/bitwise: none. At SOL, serial, non-redundant.
- SHORTLIST head (project only top-K vocab rows for the draft argmax): correctness
  is safe (the verify recomputes the true token; a shortlist miss only lowers n,
  the finish_round equality walk still commits the verify verdict). BUT (1) a
  static unigram/BPE shortlist is dead -- the argmax is content-driven; the only
  high-recall all-vocab scorer IS a low-rank/distilled head, which is an offline
  artifact that DOES NOT EXIST in-tree (turbo3 is KV-only). (2) Acceptance
  COMPOUNDS: a miss at step k truncates the ladder at n=k; modeled as a truncating
  geometric, break-even against a ~10% head-only ceiling needs per-step
  shortlist-vs-Q4 recall ~0.95 (off-shortlist p ~5%), and higher baseline
  acceptance makes the bar TIGHTER. Novel prose is exactly where argmax is least
  predictable and recall is lowest, and the 4 steps predict forward positions the
  verify never scored, so recall DECAYS with depth. (3) Even a FREE head caps the
  win at 61% of the draft (the 427 MB MTP layer stays a ~1.1 ms floor, doesn't fit
  L2) -> ~4-6% engine at the short-ctx headline, ~0 at depth.
  Concentration probe (small corpus, 2048 tok/631 distinct): suggestive (top-32K
  covers this sample) but OVERFIT to 4 prompts -- the 248K vocab exists for the
  multilingual/code/symbol tail that diverse traffic hits. Not decisive; the real
  measurement is per-step shortlist-vs-Q4 recall on held-out diverse traffic, which
  the plan makes P1 and which most likely fails.
- ADAPTIVE DEPTH is already shipped and spent (depthctl, dexit): the 3.0-3.7
  steps/round IS the shipped adaptive trim. No headroom there.

THE REAL (UN-BUILT) LEVER is OFF-PATH drafting on the idle 3090 -- hides 100% of
the draft (head + layer) at zero acceptance cost. But draft(R+1) consumes h_next
produced at the END of verify(R), so overlap requires speculating verify(R)'s
accepted-count n before it lands -- and n is LEAST predictable on novel prose
(exactly where the draft cost lives). Ceiling ~14-18% if perfectly hidden (matches
the memory's Saguaro estimate); realistic +3-9% minus misprediction, for a
weeks-long dual-GPU pipeline on a contended sm_86 3090. Worth starting ONLY if
decode t/s becomes the headline (today prefill dominates agentic wall) AND
gate_n_hist comes back peaked (it will not, on novel prose).

DECISION: do NOT build a shortlist kernel on spec. The draft stays at its floor.
If revisited, P1 is the zero-cost acceptance probe (log draft argmax vs Q4-head
argmax per step on diverse traffic, compute per-step recall); build only if recall
>= 0.97 at steps 3-4 and >= 0.95 at 1-2. This is the session's discipline applied
to a negative: measured the bound, priced the tradeoff, declined the speculative
build. New tool signal: the phd/phv/phs [req] fields ARE the realized draft
telemetry -- read them before sizing any draft work.

## 2026-07-13 (cont.) -- PREFILL profiled: it is NOT attention-bound and NOT dominant-aggregate. The lever is the weight GEMM's LSU/occupancy floor (ldmatrix untried). Diagnosis banked; rewrite is a green-light decision.

Chased "prefill dominates agentic wall." Two corrections to standing beliefs, both measured.

PREMISE CORRECTED. On real cold T8 agentic traffic (215 reqs, this session's harness
[req] telemetry): prefill is 34% of request wall at the MEDIAN (decode 66%), because
generation-heavy requests dominate the aggregate. BUT prefill is 88-96% for
large-prompt/short-output requests (read a 24-50K file, make a small edit) -- common in
agentic coding. So prefill is a large minority overall and dominates the read-heavy
request class, but "dominates agentic wall" (aggregate) is FALSE on this traffic; decode
does. (These are hit=0 cold; real interactive caching lowers prefill further.)

MEMORY CORRECTED. The standing belief (07-07) was that prefill-attention (O(N^2),
k_attn_prefill_mma) is the top lever. nsys of a 65K near-pure prefill (23.6s, 2763 tok/s):
  weight GEMM (k_gemm_mma_T)   8624 ms  63.9%   <- the real cost
  attn O(N^2) (prefill_mma_pv8) 2330 ms  17.3%
  GDN delta/conv                1298 ms   9.6%
  quant/norm/elt                1180 ms   8.7%
Attention is only 17% at 65K (it grows quadratically, so it overtakes only at ~130K+).
At agentic context (24-65K) prefill is WEIGHT-GEMM-bound, not attention-bound.

THE BOUND, ncu'd (k_gemm_mma_T, NT=128, on ffn_gate at T=1024):
  Compute(SM) SoL 59% | DRAM 6% (NOT memory-bound) | ALU 13% (dequant is cheap)
  LSU pipe 63% (the hottest unit) | warps_active 16.6% | issue_active 36% (schedulers
  64% IDLE) | 168-252 regs -> 1 block/SM.
Diagnosis: the kernel builds MMA fragments with ~80 SCALAR 32-bit smem loads per thread
per stage; the LSU pipe is the bottleneck and low occupancy (1 block/SM, reg-limited)
can't hide the load latency (issue 36%). NOT tensor-bound, NOT DRAM-bound, NOT
bank-conflicted (LDX=144 padding makes the layout conflict-free -- verified by hand).
The author already tried double-buffering + __launch_bounds__ occupancy forcing (both
SLOWER, "local optimum, do not retry") and swept NT=64->128 (128 won, for A-fragment
reuse). Register reduction for 2 blocks/SM is blocked by the acc[8][4]=32-reg
accumulator (the NT=128 tile the author chose).

THE UNTRIED LEVER: ldmatrix (LDSM). One ldmatrix loads a full MMA fragment vs 4 scalar
loads, cutting LSU instruction count ~4x -- directly at the 63% bottleneck. Plausible
1.2-1.4x on the GEMM (59% SoL -> ~80%), = ~15-25% on prefill = ~5-8% on the read-heavy
request class. HONEST RISK: (a) it is an intricate, correctness-critical rewrite of a
bitwise-gated kernel (canonical a2982c51); (b) reducing LSU may shift the bottleneck to
occupancy/issue, capping the win; (c) the author's "local optimum" suggests diminishing
returns. Realistic EV is a real-but-modest win with a genuine chance of near-floor.

STATUS: diagnosis complete and banked. The ldmatrix rewrite is a materially larger,
uncertain undertaking than this session's surgical decode wins -- a green-light decision,
not an autonomous surgical fix. Cheapest decisive experiment = an ldmatrix fork of
k_gemm_mma_T as a standalone microbench (fork the kernel, one shape ffn_gate T=1024,
tolerance-check vs the shipped kernel, measure LSU% + time). Build it only on go.
Everything else in prefill (attn 17%, GDN 10%, quant 9%) is smaller and mostly at floor.

## 2026-07-13 (cont.) -- prefill GEMM ldmatrix SPIKE: +4.1% bitwise ceiling (occupancy-limited beyond). Lever is real but small; NOT the hoped 1.4x.

Built the ldmatrix spike (tools/gemm_ldm_spike.cu) -- a fork of the XG64 prefill
GEMM on real ffn_gate weights at T=1024, A/B'd against the scalar reference,
tolerance-gated. Answered "does ldmatrix beat the scalar smem fragment loads?":
  scalar     : 0.6392 ms
  ldmatrix-A : 0.6634 ms  -3.7%   (A is amortized 8x across subtiles -- ldmatrix
                                    instruction overhead > the load it saves)
  ldmatrix-B : 0.6143 ms  +4.1%   <- the win: B is loaded ONCE per subtile (32/gg)
  ldmatrix-AB: 0.6321 ms  +1.1%   (A's -3.7% cancels most of B's +4.1%)
All BITWISE identical to scalar (rel 0.00, 0/17.8M floats differ -- ldmatrix moves
the same bytes into the same registers; my A-operand x4 and B-operand x2 address
formulas were correct first try).

VERDICT: the ldmatrix lever is REAL but SMALL. +4.1% on the GEMM = +2.6% on prefill
(GEMM is 64% of prefill) = ~+0.9% on median agentic wall, up to ~+2.5% on read-heavy
(prefill-dominant) requests. NOT the hoped 1.2-1.4x. WHY: ncu confirms the kernel
stays occupancy-limited after ldmatrix -- LSU drops but warps_active is still 16.6%
(1 block/SM, reg-limited) and issue_active 35% (schedulers idle). ldmatrix relieves
the LSU THROUGHPUT pressure but not the LATENCY/occupancy wall, so most of the
theoretical 59%->85% SoL headroom is unreachable without 2 blocks/SM -- which needs
a register cut the NT=128 acc[8][4] accumulator blocks. The author already found
NT=64 (which could fit 2 blocks/SM) SLOWER -- pre-ldmatrix; whether NT=64 + ldmatrix-B
+ 2-block occupancy wins is an open follow-on, a full re-tune not a spike.

RECOMMENDATION: the honest ceiling for the prefill GEMM's CURRENT SHAPE is ~+4%
(ldmatrix-B, bitwise). Worth integrating as free prefill-path perf (bounded: B-load
swap in the 4 real kernel variants Q4/Q8 x XG32/XG64, canonical-gated), but it is a
~+1-2.5% engine win, not a headline. The bigger prize (NT=64 + ldmatrix + 2 blocks/SM)
is a genuine GEMM re-tune with uncertain payoff (author's NT=64 negative), a separate
green-light. Prefill attention (17% @65K, grows to ~130K+) and GDN (10%) remain smaller
and mostly at floor. Spike tool kept for the follow-on. This is the session's discipline
on a marginal lever: measured the ceiling, reported it honestly, did not oversell +4% as
the 1.4x the framing hoped.

## 2026-07-14 -- P1 SHIPPED: ldmatrix-B integrated into the prefill GEMM, bitwise, +1.5% batched prefill (below the +2.6% projection)

External review P1: integrate the ldmatrix-B spike (tools/gemm_ldm_spike.cu,
07-13: +4.1% GEMM, bitwise, 0/17.8M floats differ; B-side only -- A loses, AB
cancels) into the production prefill GEMM (src/prefill.cu k_gemm_mma_T, scalar
B-loads). Done: added a plain `ldm_x2` helper (non-trans m8n8.x2.shared.b16) and
swapped the scalar activation-fragment loads for `ldmatrix.x2` in BOTH branches --
XG64 (two x2 per gg, b0..b3) and XG32 (one x2 per cc, b0,b1) -- behind
`#if Q27_GEMM_LDMB` (default 1; -DQ27_GEMM_LDMB=0 = scalar reference, the in-binary
A/B leg). ldmatrix address = the spike's exact formula (lane->token lane%8, K-half
(lane%16)/8), which reuses production's own MR/NT/KS/LDX layout, so it's a direct
swap.

GATES (vanilla qwen, --pf batched prefill = the k_gemm_mma_T path):
  - BITWISE: dump-logits(LDMB=1) == dump-logits(LDMB=0), byte-for-byte (993280-B
    logit vector), on BOTH branches -- default XG64 AND Q27_PF_XG=32. ldmatrix moves
    the same bytes into the same registers; confirmed in the production kernel on
    real weights, not just the synthetic spike.
  - canonical a2982c51 EXACT (decode path unaffected -- it uses the GEMV, not this
    kernel).
  - Q8 (Q4IN=false) NOT run (no Q8 weights on hand) but covered BY CONSTRUCTION:
    the B/activation load I changed is Q4IN-independent (Q4IN only touches the
    weight/A side + scale unpack), identical code in each branch.

PERF (--pf 4096, batched TTFT t/s, 3 runs median, <0.3% spread):
  LDMB ON  3507.3 t/s   LDMB OFF 3455.4 t/s   = +1.50% end-to-end batched prefill.

HONEST DELTA: +1.5% measured, BELOW the reviewer's +2.6% projection. The +4.1% was
the ffn_gate GEMM in ISOLATION at T=1024; end-to-end batched prefill at T=4096 runs
all layers' GEMMs plus attention/GDN/other kernels, so the isolated GEMM win does
not translate 1:1 (either the GEMM is <64% of prefill at this shape or the in-context
per-GEMM benefit is smaller). Still a FREE, bitwise, universal-on-prefill win with
zero VRAM/quality cost -- default on. Session discipline again: measure the real
end-to-end number, don't ship the isolated-kernel projection. NOTE: the CLI carries
it; server binaries (q27-server[-w16]) need a rebuild to pick up the prefill.cu
change. Reference binary build/q27-ldmoff kept for A/B.

## 2026-07-14 -- P2: short-tail prefill GEMM = runtime NT dispatch (+13-16% short prefill, bitwise); serial-threshold is the bigger follow-on

External review P2: the NT=128 prefill GEMM collapses below T~32 (a mostly-empty
128-token tile + high smem = low occupancy). Fix per the plan (docs/plans/
2026-07-13-gemm-verify.md:414): prefill is NOT graph-captured, so NT can be picked
at launch for free; NT is BITWISE-invariant (same per-output FP accumulation order),
so it is a pure speed choice.

MICROBENCH (scratchpad/gemm_nt_sweep.cu, real ffn_gate Q4, templated NT):
  T    best-NT   ms(best vs NT=128)
  16   NT=16     0.025 vs 0.067  (2.7x)
  32   NT=32     0.027 vs 0.068  (2.5x)
  64   NT=64     0.047 vs 0.072  (1.5x)
  128  NT=128    optimum
  192  NT=64     +34% (128+64 wastes a half-tile)
Smaller tile fills small T and uses less smem (more occupancy). Big win <= T=64.

SHIPPED: templated k_gemm_mma_T<Q4IN,XG64,NT>; launch_gemm_mma_x dispatches
nt = T<=16?16 : T<=32?32 : T<=64?64 : 128 (Q27_PF_NT forces a fixed tile for A/B).
4 NT x 4 (Q4IN,XG64) = 16 instantiations; each sets its own smem attr once.

GATES (vanilla qwen): canonical a2982c51 EXACT; BITWISE auto-dispatch ==
forced-NT=128 dump-logits byte-for-byte @T=16 (auto->16), T=48 (auto->64), T=512
(auto->128). NT invariance proven, not asserted.

PERF (--pf batched TTFT, 3-4 run median): +16% @T=32, +13% @T=64 end-to-end. Diluted
from the isolated 2.5-3x GEMM by the non-GEMM prefill (attention/GDN/norms). Applies
to prefix-cache suffixes and prefill tails: prefill_chunk runs with small Tc even on
long prompts (engine.cuh:2343 chunks [base..NP) at PF_T; short suffix Tc=NP-base ->
NT dispatch), so the common agentic short-turn case benefits. Free, bitwise, no VRAM.

SEPARATE FINDING -- serial-threshold lowering: a 4x win gated by a POLICY choice, not
a bug (earlier "latent bug" call was a MISDIAGNOSIS, corrected here). Total prompts <
32 tokens skip batched prefill (engine.cuh:2291 `NP >= 32`) and take the SERIAL
per-token path -- 230ms @T=16 (~16x14ms). Lowering the threshold (Q27_PF_MINBATCH
knob) DID give 4.0x @NP=16 (230->58ms), 2.2x @NP=8. The --pf serial-vs-batched M6
identity FAILS under XG32 at NP=6,8,10 (PASSES 5,12,16). FIRST READ: a small-NP bug.
WRONG -- follow-up shows XG32 batched != serial at NATIVE batched sizes TOO (N=33,64,
128 MISMATCH; 40,512 IDENTICAL), and those sizes are what production runs every day
without issue. So it is NOT a small-NP bug: batched prefill is INHERENTLY tolerance-
class vs serial (different FP reduction orders in the batched attention/GDN kernels,
NOT the GEMM -- reproduces with Q27_PF_NT=128), and the greedy continuation matches
serial only for CONFIDENT content; tie-prone content flips. Short prompts flip more
because short context is uncertain. The "XG32 = exact serial-vs-batched identity"
claim (prefill.cu:219) is inaccurate -- it holds for confident content, not
universally; the M6 gate has been passing because it is run at a size/content that
happens to be confident.

CONSEQUENCE: the 32-threshold is NOT load-bearing for a bug -- it is the boundary
below which short prompts get the EXACT serial path. Lowering it is a POLICY choice:
trade the 4x short-prompt speed for switching short prompts from exact-serial to the
(already-shipped) tolerance-class batched path. That CHANGES short-prompt greedy
outputs, INCLUDING the canonical (NP=5 -> batched -> new md5, needs re-baseline).
Deferred to a quality/policy call, not a bug fix. No code shipped for it (knob
reverted). Discipline note: "fix the bug" turned up that there was no bug -- the
measurement (XG32 mismatches at native batched sizes) refuted the premise.

## 2026-07-14 -- security/robustness review triage: 4 in-scope fixes, 3 out-of-scope (hostile artifact), 1 deferred

Second external security review (server/loader/tokenizer). Cross-checked against the
authoritative docs/SECURITY-MODEL.md (single-operator localhost engine; hostile-
artifact + adversarial-DoS findings out of scope by design). Verified each against
current code before acting.

FIXED (in-scope -- bite the operator's own benign workflow; all built + smoke-verified):
- #1 empty-prompt crash (HIGH, REAL, a GAP past the 2026-07-07 fix). reuse_len() ->
  ckpt_best() runs at SLOT SELECTION, before the engine-entry `NP>=1` guard
  (engine.cuh:2259). ckpt_best did `c.toks.size() > prompt.size()-1` -> size_t
  underflow to SIZE_MAX on an empty prompt -> nothing skipped -> std::equal derefs the
  empty vector's begin() (nullptr) -> crash, once any checkpoint exists. Fix:
  `c.toks.size()+1 > prompt.size()` (no underflow) + reject empty prompt at the
  handler (server.cu, 400 before slot claim). Smoke: `{"prompt":""}` and missing
  prompt both -> HTTP 400, server stays alive.
- #2 disconnect keeps generating (Anthropic + Responses). The OpenAI streaming
  callback already returns the sink.write() result (stops on disconnect); Anthropic
  (server.cu:1032) and Responses returned `true` unconditionally. Fix: the ev SSE
  emitter sets a captured `alive` flag on write failure; both callbacks return
  `alive`. Smoke: Anthropic stream, max_tokens=800, client cut at 3s -> engine
  stopped at dec=408 (was: generate to 800).
- #3 quadratic BPE (the one #3 sub-point the security model itself flags). bpe_word
  is O(n^2) (full pair rescan + erase-in-loop); a no-whitespace blob (minified JS/
  base64) collapses to one huge word and stalls tokenization. Fix: WORD_CAP=1024,
  chunk over-cap words -> O(n*WORD_CAP). Inert below 1024B, so normal text +
  canonical byte-identical. (The unbounded-request-size half of #3 stays out --
  network/DoS, `--host 127.0.0.1` is the mitigation.)
- #7 missing terminal finish_reason (OpenAI streaming). Every chunk had
  finish_reason:null then [DONE]; clients never learned stop vs length. Fix: emit a
  terminal chunk with "length" (produced>=nm) or "stop" before [DONE]. Smoke: stream
  now ends with finish_reason:"length".

OUT OF SCOPE (require a hostile model/tokenizer artifact -- SECURITY-MODEL.md
dispositions these as #9/#10/#11; q27 loads one self-produced model+tokenizer):
- #4 model tensor bounds integer overflow (loader.cpp), #5 model metadata negative
  layer index (engine.cuh:445), #6 truncated tokenizer header reads (tokenizer.cpp:58).
  Real code observations, but they need an attacker-supplied artifact q27 does not face.
  (If q27 ever ingests third-party model zoos, these re-activate -- per the model doc.)

DEFERRED:
- #8 capturing structured bindings is a C++20 feature (server.cu:1303 make_item_cbs
  4-tuple, 32 refs). NVCC warns but builds+runs; a real fix is a struct-return refactor
  of 32 sites for a portability nicety. Low; left as a warning.

Gates: canonical a2982c51 EXACT; test_tokenizer self-tests PASS; server smokes above.

**2026-07-14 -- CROSS-ENGINE: q27 vs llama-cpp-turboquant (TheTom) ngram-mod, side-by-side.**
Both engines, vanilla qwen, greedy, decode-only t/s (excludes prefill). q27 =
NVFP4 5.25bpw git 94e645a (MTP+SuffixDraft, its shipped config). llama =
Qwen3.6-27B-MTP-Q5_K_M ~5.5bpw git c3e6dbb13 with `--spec-type ngram-mod`
(n_match=24, n_max=64, n_min=48 defaults; self-speculative, no draft model).
IDENTICAL /v1/completions payloads on both. Decode-only comparison isolates
spec-decode effectiveness (both base kernels ~50-65 t/s single-stream, so the
delta IS the drafter):

  payload                     regime                 q27        llama ngram-mod
  echo_ctx12k (256 tok)       pure verbatim echo     603 t/s*   529 t/s (96% acc)
  fileemit_verbatim (1024)    partial-echo cont.     178 t/s    409 t/s (89% acc)*
  novel_prose (400)           novel generation       157 t/s*   56 cold / 97 warmed
  echo_ctx26k (256)           CONFOUNDED (diverged)  290 t/s    49 t/s (0 drafts)
  (* = winner; tok/round q27: 11.6 / 3.0 / 2.6 respectively)

NO clean winner -- complementary drafter strengths:
1. PURE SHORT ECHO (drafter fires hard, 108 fires, 11.6 tok/rnd): q27 WINS
   603 vs 529. Fused MTP+suffix verify is faster per accepted token than
   ngram-mod's separate draft/verify once acceptance is near-saturated.
2. PARTIAL-ECHO CONTINUATION (12K ctx, 1024 tok code): llama WINS 409 vs 178.
   q27's SuffixDraft fired only 25-38x/291 rounds (3.0 tok/rnd) where
   ngram-mod hit 89%. ROOT CAUSE = mechanism: q27 SuffixDraft needs EXACT
   suffix repetition + is greedy-gated + capped at width 12; ngram-mod's
   24-token lookup drafts up to 64 forward from ANY in-context match, so it
   tolerates near-repeats that break q27's suffix match. THIS is TheTom's
   "653 file-re-emit" regime -- real, and q27's weakest spot.
3. NOVEL GENERATION: q27 WINS decisively 157 vs 56 (cold). MTP head drafts a
   learned guess every round regardless of echo material; ngram-mod has
   nothing to match -> falls to base decode. NOTE: ngram-mod PERSISTS its
   table across requests, so repeated identical novel prompts warm 56->79->97
   by echoing their own prior greedy output; the cold 56 is the honest
   single-shot number. q27's 157.5 is request-invariant (no server-side table).
4. echo_ctx26k is NOT comparable: Q5 vs NVFP4 quant divergence made the greedy
   outputs differ (llama 169 tok w/ 0 drafts vs q27 256 tok) -> different text,
   drop it.

bpw confound: llama Q5_K_M ~5.5bpw has MORE bits than q27's 5.25 (mild edge to
llama) yet still loses novel + pure-echo -> qualitative conclusion robust.

ENGINEERING LEVER (not yet built): the fileemit regime is where q27 leaves the
most on the table. Adding an ngram-style long-lookahead lookup (24-tok match ->
draft-forward-N) to COMPLEMENT MTP+SuffixDraft (not replace) would capture
llama's partial-echo win without touching the novel-prose MTP advantage. The
current SuffixDraft is deliberately conservative (exact-suffix + greedy-gate +
W12) for bitwise determinism; an ngram path would need the same tie/tolerance
discipline as the wide fdmma verify. Candidate follow-on.

**2026-07-14 -- CROSS-ENGINE #2: llama-ngram-mod ON REAL THUNDERDOME AGENTIC TRAFFIC.**
Drove Claude Code (the claude-code-q27-haight adapter -- retargets ANTHROPIC_BASE_URL
at :8081) against the llama-cpp-turboquant fork on the SAME 3 tasks q27 ran. The fork
serves the Anthropic /v1/messages API natively (streaming, tool_use, thinking blocks,
count_tokens all verified) so no proxy needed. llama config made FAIR to q27:
5090-only (CUDA_VISIBLE_DEVICES=0, no 3090 layer-split), q8_0 KV (~ q27 fp8), single
slot, --spec-type ngram-mod --jinja, Q5_K_M ~5.5bpw. Same base model.

                          wall    exit       score   q27 wall  q27 score
  analytics-dashboard     741s    completed  0.609    150s     0.512
  time-tracker            173s    completed  0.653     54s     0.791
  structural-merge        363s    completed  0.943     90s     0.911

  DECODE:   llama 61.0 t/s agg / 55.5 med   vs   q27 289.8 agg / 236.5 med
  ngram-mod draft acceptance on agentic traffic = 34% (14370/42116)

q27 is ~4.75x faster decode and ~4x faster wall-to-wall on real agentic coding, and
wins DESPITE its known cold-prefill disadvantage (llama has the tensor-core GEMM) --
because agentic turns are decode-bound (long generation), and decode is where q27's
MTP dominates. The 34% ngram-mod acceptance is the whole story: real agentic coding is
mostly NOVEL generation (writing new code), not re-emission, so ngram-mod falls to
llama's ~55 t/s base rate. It only hit 89% on the synthetic fileemit payload. q27's
MTP head drafts every round regardless of echo -> 5.46 tok/round -> 289.8 t/s.
Both engines COMPLETED all tasks; scores are run-to-run noisy (agentic nondeterminism),
not the signal. Cost column ignored (synthetic price, meaningless for local).

Net across both cross-engine studies: ngram-mod's win is REAL but NARROW (high-echo
re-emission only); on the actual q27 workload q27 wins decisively. The fileemit lever
(add ngram-lookahead to complement MTP) would help the narrow echo case without
touching this agentic win. See prior 2026-07-14 entry for the payload-level split.

**2026-07-14 -- CROSS-ENGINE #3: reproducible SWE-bench agentic bench, THREE engines.**
Replaced the private thunderdome tasks (not redistributable) with a public, pinned
task set: 12 SWE-bench_Verified instances (fast-test repos: requests/flask/pytest/
pylint/xarray, <15min difficulty), Claude Code driving each engine's Anthropic
/v1/messages API, sandboxed per-instance in the thunderdome/claude-code Docker image
via plain `docker run` (NOT the private orchestration; --user 1000:1000 node, since
claude refuses --dangerously-skip-permissions as root). Artifacts in bench/swebench/
(run.sh, manifest.json, select_instances.py, results.*.jsonl). Full method +
reproduce steps in docs/BENCHMARKING.md. Fair config: all 5090-only, q8 KV, greedy,
same base model (Qwen3.6-27B-MTP), llama Q5_K_M ~5.5bpw (+0.25 vs q27 NVFP4).

  engine                        decode agg  med    wall/inst  gold-file
  q27 (MTP+SuffixDraft)         202.7 t/s   208.4   47 s      11/12
  llama ngram-mod (fork c3e6d)  61.1 t/s    56.9   118 s      11/12
  llama MAINLINE (13e67386,none) 62.0 t/s   62.5   120 s      12/12

MAINLINE BASELINE IS THE PAYOFF: it loads the qwen35 GGUF fine (LLM_ARCH_QWEN35 +
/v1/messages both upstream as of Jul-01) and runs stock autoregressive (no spec).
Result: fork ngram-mod (61.1) == mainline (62.0) within noise -- actually fork is
marginally LOWER (failed-draft + table overhead at 34% acceptance ~cancels wins).
So on REAL agentic coding ngram-mod adds ~nothing; its Method-A win (409 vs 178 on
synthetic file-emit) does NOT generalize. Base decode kernels are comparable (~62
t/s), so the ENTIRE ~3.3x gap is q27's MTP head drafting productively on novel
generation where prompt-lookup/ngram have nothing to match. Quality identical
(11-12/12 gold-file, model is the same; engine only changes speed). q27 finishes
each instance in ~40% of the llama wall time despite +0.25bpw and slower cold prefill.

**2026-07-14 -- CROSS-ENGINE #4: llama.cpp WITH the MTP head (apples-to-apples).**
Ran mainline llama.cpp (13e67386, which has --spec-type draft-mtp + auto-discovers the
GGUF's MTP head) with --spec-type draft-mtp --spec-draft-n-max 6 on the same 12
SWE-bench instances. Same MTP head + same model as q27 -> isolates ENGINE quality, not
drafter choice. Fair config (5090-only, q8 KV, greedy). results.llamammtp.jsonl.

  engine                                 decode agg  med    wall/inst  gold
  q27 (MTP + SuffixDraft, fused)         202.7 t/s   208.4   47 s      11/12
  llama mainline + MTP (n-max 6)         116.3 t/s   127.3   80 s      11/12
  llama ngram-mod (fork)                  61.1 t/s    56.9  118 s      11/12
  llama mainline (no spec)                62.0 t/s    62.5  120 s      12/12

GAP DECOMPOSITION (real agentic decode, all same model): stock 62 -> +ngram-mod ~62
(x1.0, adds nothing) -> +MTP 116 (x1.9, MTP is the real lever, 58% accept on agentic vs
ngram's 34%) -> q27 203 (x1.74 ON TOP of llama+MTP). So MTP nearly doubles stock, and
q27's engine (fused shared-KV MTP+SuffixDraft verify, NVFP4 kernels, tie/tolerance
discipline) is another ~1.74x over mainline's MTP -- with the IDENTICAL head. Payload
smoke agrees: q27 vs llama+MTP = 157 vs 92 novel (x1.7), 178 vs 184 fileemit (tie);
agentic is novel-heavy so the mix lands at x1.74. Quality engine-independent (11-12/12).
Answers "what about llama.cpp with MTP": it closes ~half the gap to q27; the residual
1.74x is q27's engine, not the drafter. Full 4-engine table in docs/BENCHMARKING.md.

**2026-07-15 -- CROSS-ENGINE #5: vLLM NVFP4 + MTP on the SWE-bench agentic bench.**
Added vLLM as a 5th engine. vLLM has no /v1/messages, so Claude Code drives it via a
litellm Anthropic<->OpenAI shim (:8081 -> vLLM :8080). Model: unsloth/Qwen3.6-27B-NVFP4
(compressed-tensors, has mtp_num_hidden_layers=1; multimodal Qwen3_5ForConditionalGeneration
variant, vision tower unused). vLLM nightly, 5090-only, single-seq, fp8 KV, MTP spec
(method:mtp n=3). Gotchas hit + fixed: (a) hermes tool-parser returns null -- this model
emits Qwen XML tool calls <function=x><parameter=y>, needs --tool-call-parser qwen3_coder;
(b) KV didn't fit 131072 at 0.92 util -> 0.96; (c) first run 4/12 empty = ContextWindowExceeded
(Claude Code requests ~32k max_tokens, prompt+output > the 98304 I'd set) -> raised to 131072,
re-ran clean 12/12. Harness/scripts in bench/swebench/vllm/. Decode t/s from vLLM /metrics
(generation_tokens_total / inter_token_latency_seconds_sum).

  engine                          decode agg  wall/inst  gold
  q27 (MTP + SuffixDraft, fused)  202.7 t/s    47 s      11/12
  vLLM NVFP4 + MTP (n=3)          117.1 t/s   133 s      11/12
  llama mainline + MTP (n-max 6)  116.3 t/s    80 s      11/12
  llama ngram-mod (fork)           61.1 t/s   118 s      11/12
  llama mainline (no spec)         62.0 t/s   120 s      12/12

TWO KEY TAKEAWAYS: (1) vLLM's MTP (117.1) == llama's MTP (116.3) to within noise -- two
independent codebases converge on the same ~117 t/s MTP ceiling, and q27 is a further
~1.73x on top with the SAME head. Strong evidence the lead is q27's engine, not a one-off.
(2) vLLM has the WORST wall/inst (133s) despite competitive decode, because prefix caching
is dead on hybrid-GDN (0% reuse) -> re-prefill every turn, plus the litellm hop. q27/llama
reuse prefix state across turns and convert competitive decode into low wall time. Quality
engine-independent (11-12/12). Full 5-engine table + vLLM caveats in docs/BENCHMARKING.md.

**2026-07-14 -- EXTERNAL PERF-REVIEW TRIAGE: top-3 items measured, all three priced
at <=0.5%; the review's value ranking inverted reality.** A 6-item external review
(tokenizer prefix cache / exact-BPE heap / GPU-side draft-depth / continuous batching /
ckpt traffic / reset clears) triaged by measurement before building anything.

(1) Review #3 "keep draft-depth decisions on the GPU" (claimed strongest single-request
lever): dexit A/B on vanilla qwen (fp8, PMIN=0.5, MAXD=auto, codegen+echo replays,
scratchpad/dexit_ab.sh, phd/phv/phs from [req]): per-draft-step wall 818us with
Q27_DEXIT=1 (launch+D2H+sync EVERY step) vs 810us with Q27_DEXIT=0 (monolithic, one
sync) on codegen; echo 820 vs 813 (det=OK). The entire per-step launch+sync+pageable-
staging overhead is ~8us/step = ~2-3ms/request = 0.15% of decode wall. A device-side
continuation flag / conditional graph node has nothing to recover; dexit's win is
SKIPPING steps (283 vs 405 launched, +4-6% tps), already shipped as default. NO-GO.

(2) Review #1/#2 (cache tokenized prefixes; exact priority-queue BPE): whole-prompt
re-encode measured at 2.8-3.7M tok/s (scratchpad/tok_bench.cpp on real replay prompts):
22ms @61K-tok prompt, 9ms @26K; server-side [req] tok_ms=12 @26.8K agrees. A perfect
prefix cache saves ~20ms of TTFT per turn on a maxed conversation -- <1% of any real
turn. The BPE loop is O(word^2) per WORD, words are whitespace-delimited and tiny; the
1024-byte cap only rechunks degenerate no-whitespace blobs (deliberate, 94e645a). LOW,
not built.

(3) Decode wall split, codegen @26.8K warm (the serving shape): verify 83% / draft 15% /
suffix 2.5% / residual ~0. The wall is the verify round's weight GEMV -- already at the
07-13 SOL floor (k_vgemm). No review item touches it.

(4) Review #5 ckpt traffic: ~150MB async D2H per checkpoint every 4096 tok on the
compute stream = ~100ms inside an 8.2s cold 26.8K prefill (1.2%); warm turns snapshot-hit
(pf=1, ckpt never fires). Review #6 reset() clears: cold-path only, single-digit ms; the
full-clear is deliberate (graph-capture state match, width-12 fix). Both LOW.

(5) Review #4 continuous batching across slots: the only structurally real item --
aggregate throughput under CC subagent fan-out. Scheduler+kernel refactor, needs a
design pass; filed, not started.

Hygiene noted, not built: h_draft_margin / oc[] / h_sfx_prop are pageable (each async
copy takes a staging hop) -- but that cost is already inside the measured 8us/step, so
pinning is worth ~1ms/request at best. Prefill datapoint banked: 3258 t/s @26.8K fp8
cold. METHOD (again): measure before building -- the review's "highest value" item was
worth 0.15% and its "smaller wins" were already priced into an 8us number.

**2026-07-15 -- CONTINUOUS BATCHING P1 -- 2-slot aggregate A/B.** The headline
gate (plan Task 11): Q27_BATCH=1 conductor vs the FIFO-interleave baseline.
METHOD: codegen+docs accept payloads at max_tokens 512 (longer decode window =
cleaner overlap), fired concurrently, 1 warmup pass + 3 measured reps, fresh
q27-server-w16 per leg (fp8 KV, PMIN 0.5, MAXD auto, 32K x 2 slots), aggregate
= summed dec / concurrent-window wall, median of 3 (tools/batch_ab.sh).

MEASURED: **1.21x -- MISSES the 1.3x bar** (design projection ~1.4x). Not tuned;
attributed (below) and shipped honest -- P2/P3 exist for exactly these levers.
- A FIFO:            169.1 t/s agg (169.1/169.1/169.1; per-req 85.0/88.6)
- B batched:         204.0 t/s agg (204.0/205.9/200.6; per-req 102.6/107.6;
                     bat=2.0, 159 fused rounds/req -- batching engaged)
- C batched+GEMM=1:  201.0 t/s agg -> C/B = -1.5%. The always-vgemm union sweep
  is FASTER per round (27.9 vs 29.9 ms) but loses tok/round (2.99 vs 3.22; its
  tolerance-class text is different traffic). The A1 solo-matching family
  policy costs nothing at 2 slots -- keep it.
- D solo regression (A10): BATCH=1 solo p50 vs BATCH=0: codegen -0.06%, docs
  +0.00% -- the conductor k==1 fallthrough is free. PASS.

ATTRIBUTION (diagnostic reps with Q27_PHASE_STATS=1 + Q27_BATCH_DBG=1, one per
leg, unmeasured): baseline is honest -- under FIFO each request's GPU-active
round is exactly solo pace (phd+phv = 18.2 ms/round == solo warmup; dec_ms
36.4 ms/round-pair with yields=159 = pure serialization, zero interleave tax).
Fused k=2 round wall = 29.9 ms vs the design price 26.0 (serial drafts 2x2.8 +
fused weights ~11.3 + serial mixers 2x4.6 -> 1.40x). The +3.9 ms/round excess
is the whole miss (36.4/29.9 = 1.22x observed). The plan's three suspects:
(a) serial mixers: present AS PRICED, 9.1/29.9 = 30% of the round (plan said
    ~27%). Not the miss; the P2 side-stream lever is worth ~4.6 ms/round.
(c) serial drafts: as priced (5.6 ms). P2 draft-fusion lever ~2.8 ms.
(b) the unpriced +3.9 ms splits in two, both measured: ~2.0 ms = the GEMV
    family scaling worse at union widths 7-12 than the flat weights price
    (leg C's family swap recovers exactly 2.0 ms of round wall); ~1.9 ms =
    eager-launch/sync tax of the fused body (~1100 launches/round vs solo's
    ONE graph launch; C residual after priced components = 1.9). P3
    shape-graphs + P2 family/width tuning. Trim is NOT a factor (4/159 dbg
    rounds, all 16->11s suffix; unions <=12 vs cap 16 otherwise), and the
    conductor is NOT a factor (D = 0%; B codegen's 12-round solo tail runs
    phv 15.8 ms/rnd ~= solo 15.4).
Post-P2/P3 arithmetic from these walls: 29.9 - 4.6 (mixers) - 1.9 (graphs)
- 2.8 (draft fusion) ~= 20.6 ms -> ~1.75x, consistent with the design's P2
projection (~1.7x). The 1.3x bar is reachable with the mixer lever alone.

TEXTS (greedy): docs A == B byte-identical 3/3 (all-gated, untrimmed --
bitwise contract holds). codegen B forks vs A at bytes 673/429/425 across
reps == the documented pre-existing w16-vs-w12 suffix-width fork (Task 10:
672 cold / 428 warm), entering ONLY via the 16->11s trimmed suffix rounds
(A1 policy fork); rep-to-rep md5s differ because join alignment + cumulative
depthctl move the trim point. dec=512 on every request, every leg. C forks
both payloads (vgemm tolerance family, priced) but is rep-deterministic.
Task 9 TODO check: fused rounds have no phd/phv wall buckets -- fully-fused
[req] prints clean zeros (phd=0.0 phv=0.0 phs=0), solo-tail rounds fill
normally, no garbage fields.
Sanitizer (review-fix pass, 07-15): memcheck 0 errors with FULL allocation
tracking at a small footprint (w16 server, Q27_BATCH=1 fp8, 8K x 2 slots,
2 concurrent ~240-token prompts x 64 tokens, 32 fused k=2 rounds, bat=2.0
both [req] lines; scratchpad/t12_san/sanitizer.small.log) -- the 32K w16
config OOMs memcheck's own tracking (documented limitation, standing
kernel-filtered+memory-capped rule).

**2026-07-15 -- CONTINUOUS BATCHING P1: LIVE CC VALIDATION = PASS (stability +
same-shape quality), merged to master f45a9ad and pushed.** Method: two
thunderdome CC tasks (T2=bench-collab-server, T5=bench-task-queue) run
CONCURRENTLY against the batched server, then an identical-shape control with
Q27_BATCH=0. Vanilla qwen, W12 build, fp8, 2 slots x 49152 (the w16 build's
2-slot shape maxes at 2x32K on 32GB -- too small for CC tasks, which grew past
33K by turn ~17 and crashed the first attempt at that shape; W12 2x48K is the
CC-viable batch-serving config until P2 re-prices).

  leg                       task-queue      collab-server   med tps  errors
  Q27_BATCH=1 (concurrent)  0.301 completed 0.551 completed 191.4    0
  Q27_BATCH=0 (same shape)  0.289 completed 0.303 CRASHED   148.7    0

Batched leg: 180 reqs, ZERO end=error / [req-error] / 5xx, server survived the
full window, 53 reqs with fused rounds (mean width 1.98, deepest bat=2.0,24).
Same-shape scores equal-or-better with batching ON (control even drew a crash
basin on collab-server); absolute scores sit below the 131K-era bands on BOTH
legs -- that is the 2x48K compaction squeeze (26 vs 8 ctx-limit 400s), a
context-shape effect, not a batching effect. n=1/leg, tie-lottery caveats
apply; the byte-level correctness claims rest on the Task 10 gates, not on
these scores. OPS notes: two same-second `thunderdome run` invocations race on
the results/latest symlink (stagger >=2s); first validation attempt at w16
2x32K died at the ctx wall, not in the engine (17 clean reqs then repeated
"prompt is too long" 400s CC could not compact out of at that window).
Q27_BATCH stays DEFAULT-OFF; flip is a product call now that stability is
proven -- P2 (mixer overlap, ~1.75x arithmetic) is the remaining perf lever.

**2026-07-15 -- TURBO3 KV x CONTINUOUS BATCHING: VALIDATED (correctness +
stability + capacity); quality scare at n=1 RETIRED by n=3 (lottery); perf
tax measured.** Rerun of the CC validation with Q27_KV=turbo3. Gates first:
fused_smoke honors caller-pinned Q27_KV (94e3684) and ALL legs pass
byte-identical under turbo3 (solo/fused/conductor/A2-error); then the Task-10
byte-identity gate under turbo3 (all-gated concurrent replay, suffix off,
scratchpad/t3_byteident/): 4/4 streams byte-identical to solo, cold AND warm,
with heavy fusion (bat=2.0,65). The fused path is bytewise sane on turbo3.

CAPACITY: turbo3's 2.56x smaller rows turn the W12 2-slot shape from 2x48K
(fp8) into 2x96K on 32GB (29.6GB used) -- ZERO ctx-limit 400s across every
turbo3 CC run (fp8 2x48K drew 8-26 per run). The compaction squeeze that
capped the fp8 CC validation is gone.

QUALITY (n=3/leg, medians, batched vs same-shape control): task-queue 0.577
vs 0.540, collab-server 0.287 vs 0.568. Directions MIXED, within-leg spreads
dominate (task-queue batched spans 0.215-0.607; rep1's alarming 0.215/0.272-
vs-0.540/0.631 gap inverted at rep2) -- per the standing statistical register
(n<=9/leg cannot separate binary tables) this is the documented basin
lottery, NOT a batching effect. CC-agent crashes appeared in BOTH legs and in
the fp8 control too (eval-artifact class). Absolute turbo3 scores pool below
the fp8-131K-era bands (0.78-0.85) -- turbo3-vs-fp8 quality remains an OPEN
dedicated question (the pending PPL+needle gate from the 07-11 port), NOT
answerable from these shape-confounded runs.

PERF TAX: turbo3 cold prefill 1483 t/s vs fp8 3258 (2.2x slower, 26.8K
payload); batched-leg per-request decode med 113 t/s (t3 attention dequant +
P1 serial mixers compound). Stability: ~900 batched requests across the day,
zero end=error / [req-error] / 5xx, zero server crashes.

SERVING GUIDANCE: fp8 W12 2x48K stays the CC batch-serving default (faster,
quality-known); Q27_KV=turbo3 is the capacity lever when >48K/slot matters
more than speed. OPS: bare `wait` in a script that backgrounded the server
waits forever (use explicit pids); pkill -f self-matches the invoking shell
(use pkill -x -- relearned the hard way).

**2026-07-15 addendum -- turbo3 2-slot AGGREGATE (batch_ab, w16 2x32K, same
shape/payloads as the fp8 measurement):** FIFO 159.2 -> batched 197.6 t/s
aggregate = 1.24x (fp8: 169.1 -> 204.0 = 1.21x). turbo3 batched aggregate
lands within ~3% of fp8's despite the t3 attention tax -- the tax hits solo
throughput too, so the batching RATIO is slightly better. Per-request medians
99.4/102.2 batched vs 80.0/85.1 FIFO. bat_med=2.0, 154 fused rounds/req, 2
trim events; one codegen rep-3 text fork (suffix-trim class, docs 3/3
identical) -- the documented tolerance classes, nothing new. batch_ab.sh KV
now env-overridable (KV=turbo3).

**2026-07-16 -- P2a+P2b MEASURED: fp8 1.25x / turbo3 1.27x (bar 1.3x, miss)
-- P2c triggered.** Full A/B (batch_ab REPS=3, w16 2x32K): fp8 168.1->209.6
agg (P1 was 169.1->204.0 = 1.21x), turbo3 159.3->201.8. Solo regression
0.06-0.17% (4/4 PASS). Attribution: P2a draft overlap realized ~0 (draft
steps are weight-BW-bound; two engines reading the same MTP weights share
one bandwidth stream -- overlap is worthless, only FUSION recovers draft
time); P2b mixer fork/join realized ~1ms/round of its ~4.5ms ceiling (phv
24.2->23.1ms; co-residency limited, fdmma smem footprint suspected).
Byte-identity master gate held throughout (24/24 like-composition positions
both KVs; fork/join bisect-proven byte-neutral). P2c (fused draft steps, one
MTP weight sweep per step for all active engines, ~2.4ms/round projected)
triggered per the plan's go criterion; plan at
docs/plans/2026-07-16-batch-p2c-draft-fusion.md. OPS casualties banked:
racecheck-on-full-engine = 120GB host OOM (killed two sessions; racecheck
only on synthetic drivers, sanitizers only in own systemd scopes); w16
memcheck full tracking impossible on 32GB (device tracking + 26GB anchor);
md5 refs are text+trailing-newline (a hashlib comparison without it cost a
bisect cycle).

**2026-07-16 -- P2c FUSED DRAFT STEPS: THE BAR PASSES. fp8 1.31x / turbo3
1.35x (bar 1.3x), solo regression 0%.** The full P2 arc, all measured on the
same shape (batch_ab REPS=3, w16 2x32K, medians):

  stage                    fp8 agg (B)   ratio    turbo3 agg (B)  ratio
  P1 (fused verify only)      204.0      1.21x       197.8        1.24x
  +P2a draft overlap          204.5      ~1.21x      198.1        ~1.24x
  +P2b mixer fork/join        209.6      1.25x       201.8        1.27x
  +P2c fused draft steps      221.6      1.31x       214.6        1.35x

P2c = one MTP weight sweep per draft step across active engines
(MtpLaneView + mtp_pre/attn/post/tail seams mirror the P0 pattern;
per-step chain pointers via Engine::mtp_step_view; k==1 falls back to the
captured solo graphs -- no nbatch=1 multi-lane kernel exists). Margins
bitwise identical to solo (gemv N-invariance, ninv 28/28; B8 assert silent
across every run), so caps/widths/bytes identical: master gate 32/32
like-composition positions across both KVs + B4 self-determinism. Solo
regression 4/4 PASS (one anomalous turbo3 control leg at 146.0 re-run:
151.9, -0.07% -- one-off slow control, direction was inverted anyway).
Attribution held end-to-end: draft time yielded to FUSION not overlap
(weight-BW-bound), mixers to OVERLAP not fusion (state-bound) -- the two
lessons of this phase. Remaining known taxes: mixer co-residency (~3.5ms
of ceiling unrealized; fdmma smem suspected), eager launch (~1.9ms, P3).
Plans: docs/plans/2026-07-15-batch-p2-overlap.md +
2026-07-16-batch-p2c-draft-fusion.md. Exit phase (reviews, CC sanity,
merge) next; Q27_BATCH remains default-off.

**2026-07-16 -- P3 S1: real-round capture byte spike + key census. Byte gate
PASS (16/16 like-composition positions, 623 replayed rounds); census at the
bar (28 keys/KV, top-16 ~85%, alphabet fits LRU-32) -- GO for T3 approach
A.** Q27_P3_SPIKE=1 env hack (written, gated, REVERTED -- not merged; diff +
full evidence in scratchpad/p3_s1/): Conductor::fused_round wrapped the REAL
fused_verify_round in stream capture EVERY fused round -- draft_done waits
hoisted onto cstm before BeginCapture (waits on externally-recorded events
are capture-illegal), phase-timing records hoisted outside,
BeginCapture(Relaxed) -> the whole verify body incl. the P2b side-stream
fork/join (T0's proven topology) -> EndCapture -> Instantiate -> launch the
exec for the SAME round (capture-without-execute; the round happens once,
via replay); outcome D2Hs + the one host sync stay outside behind the graph
launch; destroy per round after the sync. No capture rejection ever fired:
623/623 rounds (both KVs, mixed suffix+gated and trim-active shapes
included, up to 3574 nodes) captured, instantiated and replayed clean.
BYTES: master-refs procedure (batch_ab LEGS=B REPS=1 MAXTOK=512, fp8 +
turbo3, systemd-run, GPU idle-checked) vs scratchpad/p2_baseline/refs.md5
(trailing-newline md5 convention): turbo3 8/8 EXACT; fp8 dbg pass 4/4 EXACT;
fp8 measured leg drew the known docs-first arrival composition and matches
the like-composition pre-spike reference (p2_exit/fp8: codegen 905c96d5,
docs a9f1e759) bitwise -- an arrival fork reproduced exactly, not a capture
leak. Zero byte deviations anywhere = no capture-semantics state leak.
COSTS (fresh capture per round, throwaway pattern): body+capture ~1.0-1.1ms
+ EndCapture ~0.03 + instantiate ~1.4-1.6ms (first-call ~9ms) = ~2.4-2.7ms
median/round at 2462 (fp8) / 2526 (t3) median nodes; T0's 3.5ms @2192
number was the warmup-inclusive analog. Served aggregate under the spike:
fp8 220.4 vs 221.6 eager (-0.5%), t3 209.0 vs 214.6 (-2.6%) -- the replay
saving (~3.5-4.3ms arithmetic) nearly pays the every-round capture tax
already, live corroboration that cached execs (tax -> first-sight only)
clear the T4 bar arithmetic. CENSUS (9 Bdbg logs: p2c_t3, p2_exit x3, p2_t4
x2, p3_s1 x2; key = engine tuple + granted width vector + sfx class + gemm
family + sampled mask + kv_kind): fp8 795 rounds / 28 distinct keys, top-16
= 84.9%, top-32 = 100%; turbo3 611 / 28, top-16 = 85.3%, top-32 = 100%.
Sampled mask (0,0) throughout (greedy payloads); gemm family gemv99
throughout (29 mixed suffix+gated rounds, no all-suffix round observed);
kv_kind partitions the table per server config; engine tuple constant (0,1)
by the md5 composition witness -- NOTE the [gen] prefill-start order is NOT
that witness (p2_exit/t3 dbg: docs-first [gen], canonical bytes). VERDICT:
per-KV top-16 sits at the ~85% bar, and the stronger fact decides -- the
full per-KV alphabet (28) fits the T3 LRU-32 with headroom, so steady-state
hit rate is 100% after ~28 first-sight captures (~2.4ms each, warmup-class).
GO: T1 gates pass, proceed to T2 (conv/delta table twins) + T3 approach A
(whole-round shape-keyed exec cache); fallback D (segmented capture) NOT
triggered. T3 caveat banked: a mid-server composition flip (member re-join)
flips the tuple and can double the live alphabet past 32; LRU + cheap
recapture keeps it benign, revisit the cap under multi-tenant churn.

**2026-07-16 -- P3 T4: THE BAR PASSES. fp8 1.41x / turbo3 1.40x (bar 1.38x),
solo regression 0.00%.** Full battery at 3500a9c: clean rebuild; test_kernels/
test_conductor/ninv (NINV+SEAM+TWIN legs, both arches) ALL PASS; canonical +
sampled-seed EXACT; fused_smoke all legs incl. the graph leg, fp16+turbo3;
master refs graphs-ON 32/32 EXACT (B4 x2 both KVs, all canonical
composition); memcheck small-footprint with Q27_BATCH_GRAPH=1: 0 errors, 64
fused graph-replay rounds under the tool. THE BAR (batch_ab REPS=3 legs
A/B/D, MAXTOK=512):

  KV       leg A    leg B (graphs)  ratio   solo delta
  fp8      168.8    237.5 t/s       1.41x   +0.00/+0.00%
  turbo3   158.7    221.6 t/s       1.40x   +0.00/-0.06%

The P3 arc: eager dispatch tax measured 3.4 ms/round (2,610 launches x
~1.66us GPU starvation, nsys attribution) -> T2 table twins make rounds
perm-invariant (+0.2-0.3 ms eager cost, accepted) -> T3 shape-keyed LRU-32
exec cache (28-key alphabet, 100% steady-state hits, always-on stale-key
guard, first-sight capture ~10 ms warmup-class) -> steady phv/round -2.9 ms
-> aggregate 221.6 -> 237.5 (fp8). T5 draft micro-graphs SKIPPED (bar
exceeded; the +0.27 ms draft pool stays on the shelf). Cumulative
continuous-batching arc at 2 slots: FIFO 1.00x -> P1 1.21x -> P2 1.31x ->
P3 1.41x (fp8; turbo3 1.40x), solo cost zero at every stage, byte-identity
to the P2 references held through every phase. Landing at the top of the
design workflow's 1.39-1.44x projection. Remaining shelf: mixer
co-residency (~3.5 ms, the biggest unexplored pool), draft pool 0.27 ms,
twins' 0.2-0.3 ms eager cost (moot under graphs -- rounds replay).

**2026-07-16 -- P4 MIXER CO-RESIDENCY: MEASURED NO-GO (closed-architectural,
nothing built).** The post-P3 shelf said ~3.5 ms of unrealized mixer overlap.
Attribution (nsys, scratchpad/p4_measure/ATTRIBUTION.md) decomposed it: the
figure was mostly LAUNCH TAX DOUBLE-COUNTED -- P3's graphs already harvested
the GDN share (loss 1.04 -> 0.16 ms/round, node-level profile) -- leaving a
true residual of ~0.7-0.8 ms/round concentrated in fdmma/fd2 WAVE
SERIALIZATION: each engine's verify attention fills exactly one full GPU
wave by design (ns=85 x 4 heads = 340 CTAs = 2/SM x 170 SMs, third CTA
forbidden by regs AND smem), so two engines board only in each other's
drain tails. The one candidate lever (k-aware half-wave split, ns=85/k) was
mechanism-probed at ns=42 vs 85 on like-shape fused rounds (trajectory-clean;
fd2 controls identical): co-residency becomes TOTAL (both/min 53% -> 98%,
wall collapses onto max) BUT per-kernel time inflates 1.3-1.6x (each engine
streams its KV pass through half the machine), netting -13.7 us/window =
~0.10 ms/round live -- 3-4x short of the +1.5% bar, at the price of the
batched-vs-solo bitwise gate on attn rounds. The free end-to-end spike
(process-wide NS=42 batch_ab) was discarded as confounded: the tolerance
fork re-rolled the codegen trajectory (dec 512->397, fused rounds 159->72),
a caution for any future numeric-knob A/B. CLOSING PHYSICS, completing the
P2 lesson: weight-BW-bound work -> fusion only; state-latency-bound work ->
overlap; SATURATED work (attn KV streaming at depth) -> neither. The
continuous-batching campaign ends at fp8 1.41x / turbo3 1.40x (2 slots,
solo 0%), with the residual ~0.6 ms/round booked closed-architectural and
the remaining shelf (draft pool 0.27 ms, kernel-fusion node floor ~0.3 ms)
priced below build cost.

**2026-07-16 -- P3 LIVE CC A/B: graphs vs eager on real agentic traffic =
TRANSFERS. Depth-matched fused phv/round -4.9 to -5.5 ms (-17 to -19%),
fused tps +15-17%; solo untouched; zero errors both legs. One live finding:
the CC key alphabet (44+) outruns the LRU-32 cap -- churn is benign.**
Method: two thunderdome CC tasks (T2 collab-server + T5 task-queue, one pair
per leg, staggered 15 s) driven CONCURRENTLY at the same server, same day,
same build (master 765933d, build/q27-server W12), same shape: Q27_KV=turbo3
Q27_MAXD=auto Q27_BATCH=1, 2 slots x 98304, --no-think --fast-head, port
8081, systemd unit q27-eval; leg GRAPHS adds Q27_BATCH_GRAPH=1, leg EAGER
omits it (GRAPHS ran first). Both legs +Q27_PHASE_STATS=1 +Q27_BATCH_DBG=1
(phv + gcache/bat telemetry). Per the 07-15 CC-validation register: CC
trajectories fork on wall-clock bytes, so WALLS AND SCORES ARE
TRAJECTORY-CONFOUNDED; the engine's own [req] lines are the currency.
Evidence: scratchpad/p3_cc_ab/{graphs,eager}/ (journal + req.lines +
harness logs + analyze.py/bucket.py).

STABILITY (gate 1): both legs ZERO end=error / [req-error] / 5xx; servers
survived and stopped clean; GRAPHS 92 reqs (92 eos; http 92x200+2x404 HEAD
probes), EAGER 138 reqs (137 eos, 1 n_max; 138x200+2x404). GCACHE (GRAPHS):
rounds=954 hits=822 misses=132 evictions=100 guard_trips=0 (86.2% hit).
The "misses ~= alphabet, 0 trips" expectation HALF-held: gt=0, but live CC
drew 44+ distinct keys (vs the batch_ab census's 28) -- maxd-auto width
churn on real traffic is richer than bench payloads, the LRU-32 cap binds
and ~88 of the misses are eviction-churn recaptures. Benign: recapture tax
~132 x 2.4 ms ~= 0.3 s across the 42 s fused window (<1%), and the hit rate
held 86%. Q27_BATCH_GRAPH_CAP is the knob if multi-tenant traffic ever
widens this (cap 64 would swallow the observed alphabet).

THE COMPARISON (fused traffic, [req] currency). Raw distributions
(trajectory-confounded mixes -- GRAPHS drew SHORT trajectories, EAGER long,
see scores): fused (bat>=1.5) tps med/p75/max GRAPHS 132.5/140.0/177.0
(n=36) vs EAGER 123.3/147.4/190.6 (n=95); fully-fused (bat=2.0) phv/round
weighted GRAPHS 24.23 ms (1193 rounds) vs EAGER 30.31 ms (9692 rounds).
The de-confounded cut -- fully-fused reqs bucketed by prompt depth, with
tok/round verified matched (3.7-4.3 both legs, GRAPHS wider at [30-40K)):

  ctx bucket    phv/round(w)  GRAPHS vs EAGER      tps med          phv/token
  [20K,30K)     23.80 vs 29.33  (-5.5ms, -18.9%)   132.4 vs 114.6   6.41 vs 7.52
  [30K,40K)     24.95 vs 29.87  (-4.9ms, -16.5%)   137.9 vs 117.8   5.76 vs 7.39
  [40K,60K)*    26.98 vs 30.82  (-3.8ms)           151.0 vs 129.6   5.87 vs 7.15
  [60K,100K)*   28.57 vs 31.55  (-3.0ms)           125.8 vs 120.5   4.80 vs 6.66
  (* GRAPHS side of the deep buckets = the same-shape P3-exit CC sanity
  dataset (graphs ON, long trajectories) -- this leg's short draw produced
  no fused traffic past 40K; two independent graphs datasets agree at
  every matched depth.)

VERDICT: P3 TRANSFERS. Live fused rounds save 3.0-5.5 ms/round depth-matched
(T4 bench arithmetic said -2.9 steady; mid-depth live rounds beat it --
md-auto runs deeper draft chains per round than the w16 bench shape, so a
round carries more launches for the graph to erase), fused-window tps +15-17%
at matched depth (bench aggregate increment was +9.8% turbo3). Solo tps:
raw medians 202.5 vs 232.2 look like a gap but do NOT replicate -- depth-
matched buckets split both directions ([40-60K) 213.8 vs 235.2, [60-100K)
261.6 vs 270.1, and the prior graphs sanity drew 266.8 at [40-60K), ABOVE
eager) -- mix noise, consistent with T4's solo delta 0.00% (graphs only
touch fused rounds). Aggregate tokens/overlap-window (801 vs 985 tok/s) is
NOT comparable: dec_ms window reconstruction compresses gate waits and the
fused samples differ 8x in size; descriptive only, non-load-bearing.

SCORES (descriptive, basin-lottery caveat -- n=1/task/leg cannot separate
anything): GRAPHS T2 0.287 (completed, 87 s, 36 turns) / T5 0.607
(completed, 180 s); EAGER T2 0.530 (completed, 1160 s) / T5 0.615
(completed, 960 s); prior same-shape graphs sanity T2 0.565 / T5 0.575.
All four completed, zero crashes = the only score-shaped signal. The
lottery cut BOTH ways on walls (GRAPHS leg drew 87/180 s, EAGER 1160/960 s)
-- which is exactly why walls and scores are not the currency here. Context
for the running table: 07-15 turbo3 CC validation (P1-era batching, same
2x96K capacity): per-req decode med 113 t/s, ~900 reqs 0 errors; this A/B's
fused medians (GRAPHS 132.5 live vs that 113) also carry P2's fusion gains,
not graphs alone. Serving call stands: Q27_BATCH default-off is a product
call, but when batching is on, Q27_BATCH_GRAPH=1 is now validated live --
stability clean, solo-neutral, and the fused-round win is real on agentic
traffic.

**2026-07-16 -- CONTINUOUS BATCHING DEFAULTS FLIPPED ON (product call,
owner-authorized).** The CC serving profile now sets Q27_BATCH=1
Q27_BATCH_GRAPH=1 Q27_BATCH_GRAPH_CAP=64 (setenv overwrite=0, the house
pattern: user env always wins; Q27_PROFILE=ref skips the block entirely, so
ref stays the conservative no-batch reference; the CLI binary is untouched
and every bitwise canonical gate rides the CLI). Evidence line: THE BAR
1.41x fp8 / 1.40x turbo3 aggregate at 2 slots (bar 1.38x), live CC A/B
fused rounds -17..-19% phv/round depth-matched (+15-17% fused tps), solo
cost 0.00%, and four clean live CC validations (07-15 turbo3 2x96K ~900
reqs, P3-exit sanity, live A/B GRAPHS + EAGER legs) with zero errors.
CAP=64 rationale: the live CC key alphabet drew 44+ distinct graph keys vs
the LRU-32 default (86% hits, ~88 eviction-churn recaptures, benign at
<1% tax) -- 64 swallows the observed alphabet at ~460 MB worst case
(8 MB/exec budget), and the conductor ctor's headroom check
SHRINKS-never-aborts, so tight configs self-protect. TWO-TIER M2 GUARD
(the semantic change): batch_env_user is captured BEFORE the profile
setenv block; user-EXPLICIT Q27_BATCH=1 + incompatible env {PMIN<=0,
DEXIT=0, SAMPLE_PLAIN, TOOL_SPLIT} keeps the fail-fast FATAL exit(1)
(you asked for a config that cannot run), while profile-DEFAULT +
incompatible env prints one line -- "continuous batching: OFF
(auto-disabled: <reason>)" -- skips conductor construction, and serves
exactly as pre-P1 (a default must never kill a formerly-working
invocation; ref-profile runs never even reach the guard). The banner now
states provenance: "ON (serving default since 2026-07-16 | Q27_BATCH=1
explicit env)" / "OFF (Q27_BATCH=0 | Q27_PROFILE=ref | auto-disabled:
reason)". Kill switches: Q27_BATCH=0, Q27_BATCH_GRAPH=0, Q27_PROFILE=ref.
Gates at this commit: make + w16 + fused_smoke rebuilds clean;
test_kernels ALL PASS; test_conductor PASS; canonical EXACT
a2982c5197c627551b27d76a0a94b220 + sampled-seed EXACT vs p0_baseline (the
flip lives in the SERVER profile block only -- CLI defaults proven
untouched); bare-server W12 codegen replay position-wise == p0_baseline r1
with bat= present (k=1 solo fallthrough byte-identity); kill switches +
both guard tiers exercised live; 2-slot smoke (32K+32K, concurrent
codegen/docs) bat>=1.5 fused, gcache ON at cap 64, zero errors.

**2026-07-16 -- v0.2.0 BENCHMARK REFRESH (master c0c5c5e, rebuilt
binaries; first re-run of the records with continuous batching DEFAULT-ON
in the server profile).** Gates first, fresh build/q27 at stock clocks
(mem offset verified 0): canonical a2982c5197c627551b27d76a0a94b220 EXACT
+ sampled-seed 8b6aacf912d8e4c7a50a021623c6c276 EXACT vs p0_baseline.
Single-stream: shortbench suite mean **177.4 t/s** (hash-table 170.7 /
merge-sorted 172.1 / planets 185.5 / translate-fr 173.5 / tcp-vs-udp
185.0; canonical prompt 144.2 t/s, 2.61 t/round) -- lands on the 07-13
delta-fusion number EXACTLY; the README's 172.2 reference line was stale
and is refreshed. THE BAR re-run (batch_ab LEGS="A B D" REPS=3
MAXTOK=512, w16 build; leg B is the bare-defaults path: Q27_BATCH=1
explicit + GRAPH/CAP=64 riding in from the profile -- GATEENV audited,
it sets only KV/PMIN/MAXD, none in the incompatible set):

  KV      leg A    leg B (graphs)  ratio   solo delta (D)
  fp8     168.9    237.7 t/s       1.41x   +0.06% / +0.00%
  turbo3  158.5    224.2 t/s       1.41x   +0.07% / -0.06%

vs T4 (3500a9c): fp8 168.8/237.5 -> 168.9/237.7 (unchanged, noise);
turbo3 B 221.6 -> 224.2 (+1.2%), ratio 1.40x -> 1.41x -- both KVs now at
1.41x. Banner proof per leg: A/D0 "continuous batching: OFF
(Q27_BATCH=0)"; B/D1 "ON (Q27_BATCH=1 explicit env, union cap 16)" +
"[gcache] fused-verify graph cache ON (Q27_BATCH_GRAPH=1, cap 64)".
bat_med=2.0 both KVs (fused_med 159 fp8 / 154 turbo3); debug passes show
trim lines (4 fp8 / 2 turbo3) and non-zero phd/phv/phs on fused [req].
Text: docs A==B both KVs; codegen forks through the documented A1
suffix-trim (fp8 B-only; turbo3 rep-NONDET on BOTH legs -- concurrency
re-rolls quantized-KV ties, the docs md5 SETS still match A vs B).

ZERO-CONFIG SPOT CHECK (the release claim): bare `q27-server model tok
--ctx 32768 --slots 2 --slot1-ctx 32768`, NO env (W12 binary): banners
"continuous batching: ON (serving default since 2026-07-16, union cap
12)" + gcache ON at cap 64; concurrent codegen+docs 512-tok reps
aggregate **234.3 / 238.7 t/s**, bat=2.0/1.9 (159 fused rounds/req),
warm turns pf=1 off the prefix snapshots, zero errors; DBG pass gcache
h=289 m=29 (90.9% hit) ev=0 gt=0 over 318 fused rounds -- CAP=64 swallows
the 2-slot alphabet with zero evictions. ONE EDGE found by running it
bare-bare (no --ctx at all): --slots N does not auto-size --ctx (8192
default + a "pass --ctx" warning), so a ~26K prompt serializes onto
slot 1 and nothing fuses -- documented in README Serving. Docs refreshed
with these numbers: README reference/State/Serving lines +
docs/BENCHMARKING.md 2-slot aggregate table.

**2026-07-16 -- TURBO3 AGENTIC QUALITY GATE (master eccc641 / v0.2.0).**
The open question from the 07-15 turbo3 x batching validation: turbo3 CC
scores pooled below the fp8-131K-era bands in SHAPE-CONFOUNDED runs (fp8
ran 131K windows, turbo3 ran 2x96K/2x48K squeezes). Already measured, NOT
redone here: generic-corpus (wikitext-2) position-bucket NLL turbo3 vs fp8
+0.65-1.2%/bucket flat to 297K (07-11); needle 6/6 @361K (07-11);
acceptance TIES fp8 on basin-matched CC replay (07-11 accept_kv_ab). What
was never measured: NLL on AGENTIC-shaped text (ChatML + tools + code +
tool_response blocks) and a shape-MATCHED CC score comparison.

PRE-DECLARED DECISION RULE (written before any result was collected):
turbo3 PASSES as agentic-quality-safe unless
  (a) agentic-corpus NLL delta (turbo3 vs fp8) at CC depths (16-100K
      buckets) substantially exceeds the +0.87% generic figure -- i.e.
      >+2% in any CC-depth bucket, sustained across buckets; OR
  (b) the shape-matched CC study (n=3/leg) shows a gross consistent
      deficit: BOTH tasks' medians lower by >0.15 with matched
      ctx-squeeze.
Per the standing statistical register, n=3/leg cannot separate small
effects -- anything short of (a)/(b) reads "no detectable tax; the 07-15
band gap attributed to shape confound", reported descriptively.

LEG 1 -- agentic-corpus position-bucket NLL A/B (deterministic). Corpus:
the LARGEST captured real CC conversation (scratchpad/ccreplay
req_0031.json, 2026-07-14 capture_proxy traffic) rendered to its full
ChatML token sequence by the server's OWN code path (anthropic_msgs ->
anthropic_tools_json -> chatml_prompt think=false, then
Tokenizer::encode; renderer = scratchpad/t3_quality/render_req.cpp
including src/api_common.h verbatim): 15 messages, 615,147 chars ->
**154,160 tokens**, one contiguous stream, no concatenation (provenance +
sha256 in scratchpad/t3_quality/corpus/). Run: `--nll <corpus> --nll-long
154160 --ctx 163840`, CUDA_VISIBLE_DEVICES=0, one pass, no resets,
Q27_KV=fp8 vs turbo3 (the two serving formats), vanilla qwen:

  bucket      n      fp8 NLL  turbo3   dNLL     dPPL%
  0-2k        2047   2.8235   2.8810   +0.0575  +5.92
  2k-8k       6144   1.9144   1.9022   -0.0122  -1.21
  8k-16k      8192   2.1390   2.1461   +0.0071  +0.71
  16k-32k    16384   2.1347   2.1386   +0.0039  +0.39   << CC range
  32k-48k    16384   0.0248   0.0280   +0.0032  +0.32   << CC range
  48k-64k    16384   0.0016   0.0020   +0.0004  +0.04   << CC range
  64k-96k    32768   0.0016   0.0022   +0.0006  +0.06   << CC range
  96k-128k   32768   0.0002   0.0002   +0.0000  +0.00
  128k-160k  23088   0.0113   0.0120   +0.0007  +0.07

Rule (a): NOT triggered -- every CC-depth bucket is far under +2%; the
worst CC-depth bucket (16k-32k, the only content-diverse one, n=16384)
is +0.39%, UNDER even the +0.87% generic short-context figure. Shape
notes, reported descriptively: buckets >=32k are echo-dominated (NLL
0.0002-0.028 -- the transcript's depth is one ~252K-char assistant turn,
i.e. model-generated text that teacher-forcing re-predicts near-argmax;
0% duplicate 2K-char chunks, so this is own-output echo, not copy-paste
repetition). That IS the agentic serving regime at depth, and turbo3
does not disturb it (absolute dNLL <= +0.0032 everywhere past 32k). The
0-2k +5.92% and 2k-8k -1.21% wiggle (n=2047/6144, system-preamble +
task-statement content) is outside the rule's CC-depth scope and
sign-flips, i.e. content noise, consistent with the known ~+1% short-ctx
class. Novel-content-at-depth coverage stays with the 07-11 generic
corpus (+0.65-1.2%/bucket flat to 297K) -- the two corpora are
complementary and BOTH clear their bars.

LEG 2 -- shape-matched CC study (n=3/leg, alternating fp8,t3,...). Config:
v0.2.0 serving defaults (batching+graphs ON), W12 build/q27-server, fresh
server per rep as unit q27-eval on :8081, --ctx 49152 --slots 2
--slot1-ctx 49152 BOTH legs (the fp8-max 2x48K shape -- turbo3 takes the
SAME squeeze, removing the 07-15 confound), CUDA_VISIBLE_DEVICES=0,
vanilla qwen. Per rep: thunderdome T2 (bench-collab-server), sleep 15
(latest-symlink stagger), T5 (bench-task-queue) CONCURRENT; scores from
trials meta.json composite_score (harness exit codes ignored per the
standing note). Runner: scratchpad/t3_quality/leg2_run.sh; raw per-rep
logs + meta in scratchpad/t3_quality/leg2/.

  rep          T2-collab           T5-taskq            ctx400  med tps
  1 fp8        0.303 crashed 271s  0.200 completed 10s   10     215.2
  2 turbo3     0.631 completed     0.653 completed 411s  17     204.9
  3 fp8        0.303 crashed 205s  0.200 completed 11s    7     216.8
  4 turbo3     0.303 completed     0.200 crashed 164s    25     179.7
  5 fp8        0.277 crashed 381s  0.200 completed 10s   19     240.8
  6 turbo3     0.272 completed     0.293 crashed 305s     4     150.5

  leg     T2 median (spread)        T5 median (spread)
  fp8     0.303 (0.277-0.303)       0.200 (0.200-0.200)
  turbo3  0.303 (0.272-0.631)       0.293 (0.200-0.653)

Rule (b): NOT triggered -- T2 medians TIE (0.303 both), T5 turbo3 is
HIGHER (+0.093). No deficit on either task, let alone >0.15 on both.
Confound counters comparable: ctx-limit 400s fp8 36 vs turbo3 46 total
(overlapping per-rep ranges; turbo3's 25 came from its longest full
session, 804s/5.04M tok) -- the squeeze bound BOTH legs as designed.
Stability: 549 requests across 6 reps, ZERO end=error / [req-error] /
5xx, zero server crashes. Decode context: per-rep med tps fp8
215-241 vs turbo3 151-205 (the known t3 decode tax; turbo3 reps also ran
the longer, deeper sessions). Eval artifacts hit BOTH legs and are
score-dominant: fp8's T5 leg is a DETERMINISTIC one-shot-quit (3/3 reps
byte-identical 24,109 tokens, 10-11s -- the [drift] UN-RESCUED
first-tool-call class, greedy determinism re-drawing the same
trajectory), fp8's T2 crashed 3/3 (CC-agent crash class), turbo3's T5
crashed 2/3; turbo3 drew the only two high basins (rep 2). Per the
standing register these are basin/artifact modes, NOT KV-quality signal
-- which is exactly why the rule demanded a GROSS consistent deficit,
and there is none. Both legs pool well under the fp8-131K-era bands
(0.78-0.85), confirming the band gap tracks the 2x48K squeeze, not the
KV format.

VERDICT: **PASS -- turbo3 is agentic-quality-safe.** Neither trigger
fired: (a) agentic-corpus NLL at CC depths max +0.39% (bar >+2%
sustained); (b) shape-matched medians tie/favor turbo3 (bar: both
tasks -0.15). Finding of record: NO detectable turbo3 quality tax on
agentic serving; the 07-15 "turbo3 pools below fp8-era bands"
observation is attributed to the SHAPE CONFOUND (131K windows vs
2x96K/2x48K squeezes), as suspected. With quality now settled on top of
the 07-11 gates (generic NLL flat to 297K, needle 6/6 @361K, acceptance
ties) and the capacity picture already established (turbo3 2x96K vs fp8
2x48K on 32GB, W12), the serving guidance becomes: fp8 stays the CC
default on SPEED (turbo3 costs ~5-30% median decode depending on depth
mix); choose turbo3 whenever capacity matters -- >48K/slot, more slots,
or the 3090 -- with no quality asterisk. The turbo3-vs-fp8 quality gate
(open since the 07-11 port) is CLOSED.

**2026-07-16 -- CLUB-3090 HARNESS ON OUR SILICON (their bench.sh verbatim,
endpoint-only; 5090 + 3090, two passes each).** Prerequisite shipped
first: the OpenAI streaming paths now emit the spec
`stream_options.include_usage` final usage chunk (`aa991de`; both
/v1/chat/completions and /v1/completions, one shared handler; absent
option = framing byte-identical). Gates on that change: make + w16 + w8
rebuilds; canonical a2982c5197c627551b27d76a0a94b220 + sampled-seed
8b6aacf912d8e4c7a50a021623c6c276 EXACT (CLI untouched); bare-server W12
codegen replay text == p0_baseline r1 EXACT; live curl A/B -- no usage
chunk without the option, spec-shape chunk with it, both API shapes.
Then their harness unmodified from their repo (read end-to-end first;
endpoint-only mode): `URL=http://localhost:8020 CONTAINER=none PP=1 bash
scripts/bench.sh` -- 3 warm + 5 measured per prompt (narrative 1000 tok /
code 800 tok), temp 0.6 top_p 0.95, streamed usage counts, salted prefill
probes at 10K/90K. q27 = vanilla qwen, single slot, bare defaults,
systemd-run, GPU-exclusive.
  5090 (W12, fp8+fdmma, auto-ctx 262144), two passes: narr 144.15/143.97
wall (151.81/151.62 decode), code 193.04/192.82 (210.92/210.65), TTFT
350 ms, prefill 3372/3350 t/s @10K, 2559/2560 @90K (client-observed);
in-run CV <=0.2%, pass delta <=0.15%. [req] cross-check: 151.4 / 209.8
t/s engine at 2.64 / 3.86 tok/round.
  3090 (w8, fp16 KV + h16, banner fd=mma, `--ctx 24576`), two passes:
narr 84.06/83.41 (88.59/87.88), code 105.76/105.67 (115.08/114.97), TTFT
~610 ms, prefill 1124/1123 @10K; the 90K depth SKIPped by their harness
on q27's context-limit 400 -- their documented over-ctx path, not a
failure. Engine cross-check 88.2 / 114.5 t/s.
  vs their published rows (decode-to-decode, all spec-on): 5090 +19%
narr / +3% code over their best single-5090 (vLLM DFlash 127.98/204.80
decode), within 2-6% of their DUAL-5090 row on wall; 3090 +47% narr /
+14% code over the best published single-3090 decode (ik MTP 60.39 narr,
beellama DFlash 101.3 code), ~91% of their 2x3090 vLLM dual decode row
(96/127) on one card -- honest asterisks: their 3090 rows are mostly
370W-capped (ours drew ~417 W; they document -29..-42% at 230 W) and
serve 102-200K ctx vs our 24576 on this config. ANOMALY logged: bare
auto-ctx (36864) and explicit `--ctx 32768` both OOM at
verify_sample_graph instantiation on the 24 GB card under the 07-16
defaults; 24576 boots -- the auto-ctx anchor is 5090-calibrated, exactly
the miss the README warns about; turbo3 remains the 131K lever on that
card. Full method, tables, and caveats: docs/BENCHMARKING.md "vs
club-3090 community recipes (their harness, our silicon)".

## 2026-07-16 -- q4s tier SHIPPED: 4.55 bpw for VRAM-starved cards -- 2.27 GB smaller, +5.2% suite, AND PPL beats default (-0.26%)

Ask came from GitHub issue #1 (A10 cloud card: 24 DECIMAL GB minus ECC
= 22.6 GiB usable): default weights + w8 + turbo3 measured a 28672 ctx
ceiling -- the fixed stack eats 95% of that card, and the graph-zoo
env knobs cap out at +280 MiB (Q27_MAXD=4, his measured sweep; DEXIT
is not a capture knob, the per-step graphs capture unconditionally).
The real lever is weight bytes. Design: strip the v1.4 above-Q4 mass
that doesn't pay. repack.py grew --q4-head: emit output.weight ITSELF
at Q4_G64 and drop the output_q4.weight dupe (one lm_head serves
draft/verify/plain; all four name-keyed call sites fall back
correctly -- engine change ZERO), plus no --q8, so the v1.4
residual-writer promotion (ssm_out+attn_output) reverts to Q4.
KEPT Q8: token_embd (phase-2 candidate, unmeasured), blk.64 MTP
(FORMAT.md: draft/verify agreement craters), attn_k/v (~0.17 GB,
errors persist in KV).

Artifact: qwen36-27b-mtp-q4s.q27, 15.46 GB = 4.55 bpw (repack 678s,
866 tensors, md5 7e5454e0c0ded717136ad3e42634ba25, tag q4s-v1; worst
RMSE 0.1226 = same class as v1.4's residual Q4 worst 0.115, and the
Q4 head is NOT in the worst-15). Bytes that moved: lm_head mass
1.966 GB (Q8 head + Q4 dupe) -> 0.675 (one Q4 head); promotions
2.045 -> 1.070. Total -2.27 GB = ~167K tokens of turbo3 KV budget.

MEASURED (same-day, master e58c063 fresh build, 5090):
- canonical: v1.4 a2982c51 EXACT first (build sanity); q4s canonical
  f64e7c02252ca4c40cea62db662205e0, deterministic x2, 2.84 t/round.
- PPL (--nll, 148335 preds, c2048, paired): v1.4 8.0409 reproduced
  EXACT; q4s **8.0197 = -0.26% BETTER than default**. Third measured
  error-cancellation structure (after ffn_up and GDN in-proj) -- the
  v1.4 promotion was tuned on qwopus; on vanilla wikitext it was
  hurting, not helping. Ladder: Q5_K_M 7.9179 / q6k 7.9127 / q6
  7.9460 / q4s 8.0197 / v1.4 8.0409.
- suite: 186.2 vs 177.0 t/s same-day (+5.2%) -- decode goes as weight
  bytes; the 12.8% byte cut beats the acceptance mix shift.
- serving: turbo3 boots, auto-ctx picks the 262144 cap on the 5090
  (24.0 GB used), coherent completions through /v1/messages.
- NOT run: task dome (tier-dome precedent says no score separation;
  run before quality-critical recommendations); phase-2 token_embd
  demotion (-0.6 GB more) unmeasured.

Small-card math (A10 fixed 21907 MiB measured at maxd4): q4s fixed
~19.7 GB -> ceiling ~215K by arithmetic, up from 49152 tuned / 28672
stock. A 24 GiB 3090 reaches the 262144 cap by the same arithmetic
(not boot-verified; vox resident). Tier map now: q4s 4.55 (max ctx,
fastest) / default 5.25 (reference, canonical a2982c51) / q6 6.0 /
q6k 6.8. Shipped: README tier row + small-cards rewrite, CHECKSUMS,
HF upload alongside the existing tiers.

FIELD CONFIRMATION (07-17, issue #1, same A10, commit 666b7d9): the
reporter downloaded q4s off HF before the announcement even posted
and ran his own ladder. Measured: v1.4+maxd4 49,152 re-confirmed
(30 MiB spare); **q4s+maxd4 212,992** (22,567 MiB used, 22 MiB
spare; 217,088 OOMs at the sampled-verify instantiate) = 7.4x his
stock 28,672. Predicted 219K vs measured 213K -- the delta is the
new gcache exec reserve, whose headroom guard his logs show working
as designed at the brim (cap 64 -> 2 with LRU recapture). Boot also
dropped 18.1s -> 6.0s on the smaller weight stream. Scored tasks
(07-17, 5090): HOLD at default-tier level, tier-dome precedent
repeats. q4s A10 row promoted to the README.

## 2026-07-16 -- q4s-v1 REPACK VALIDATION: full ladder GREEN, anchors minted, club-3090 matched-bpw rerun (both GPUs)

Independent validation ladder on the q4s artifact above (master
e58c063 binaries; file md5 7e5454e0c0ded717136ad3e42634ba25 verified
against CHECKSUMS.md5). Where this ladder re-measured the SHIPPED
entry's numbers it CONVERGED exactly (canonical md5, paired PPL) --
two sessions, same answers. v1.4 stays the reference tier; nothing
here changes a serving default; adoption is a product call.

**q4s-v1 CANONICAL ANCHORS (new-artifact anchors -- v1.4's a2982c51 /
8b6aacf9 are NOT replaced and still gate the default tier):**
- canonical (`--tokens "760,6511,314,9338,369" -n 128 --ctx 2048
  --spec`, grep '^generated:' | md5sum):
  **f64e7c02252ca4c40cea62db662205e0**, x2 runs EXACT.
- sampled-seed (same prompt, `-n 64 --temp 0.7 --top-p 0.95
  --seed 42`): **900031e9b86df8f52493e6c1f4040c2e**, x2 runs EXACT.

Ladder (all GPU work under systemd-run, GPU idle-checked per rung):

1. SMOKE + test_kernels(q4s): generation sane both recipes, anchors
   above. test_kernels: **83/83 executed checks PASS** -- including
   dequant/gemv/gemv_n/gemm-MMA/g64 on the Q4 head via dtype dispatch
   -- then the suite ABORTS (illegal memory access,
   test_kernels.cu:498) inside `test_gemv10_scaling`: the P10-A0 perf
   probe HARDCODES `gemv_q8_n` on `output.weight` and reads 2x past a
   Q4 head allocation. Harness fixture assumption (default-tier Q8
   verify head), NOT a q4s weight fault -- same-day v1.4 control on
   the same binary: 315 checks ALL PASS (which also covers the
   model-independent synthetics the crash skipped). LOOSE END: teach
   test_gemv10_scaling to skip or dispatch on `q4_head` files.
2. PPL, exact v1.4 protocol (wiki.test.qwopus.i32, `--nll --nll-chunk
   2048 --ctx 2048`, 145 chunks, 148335 preds, fp16 KV, paired legs):
   q4s **8.0197** / v1.4 re-run **8.0409** (recorded value reproduced
   EXACT). q4s = **-0.26% vs v1.4 (better)**, **+1.29% vs the Q5_K_M
   bar 7.9179** (v1.4 +1.55%). Matches the SHIPPED entry's paired run.
3. AGENTIC NLL (t3_quality leg-1 method: 154,160-tok real-CC corpus,
   `--nll-long 154160 --ctx 163840`, BOTH legs fp8 KV; the v1.4 leg
   reproduces the turbo3-gate fp8 column to the 4th decimal):

     bucket      n      v1.4     q4s      dNLL     dPPL%
     0-2k        2047   2.8235   2.7857   -0.0378  -3.71
     2k-8k       6144   1.9144   1.8949   -0.0195  -1.93
     8k-16k      8192   2.1390   2.1823   +0.0433  +4.43
     16k-32k    16384   2.1347   2.1607   +0.0260  +2.63  << CC range
     32k-48k    16384   0.0248   0.0277   +0.0029  +0.29  << CC range
     48k-64k    16384   0.0016   0.0014   -0.0002  -0.02  << CC range
     64k-96k    32768   0.0016   0.0017   +0.0001  +0.01  << CC range
     96k-128k   32768   0.0002   0.0002   +0.0000  +0.00
     128k-160k  23088   0.0113   0.0123   +0.0010  +0.10

   FLAG + disposition: 16k-32k (the one content-diverse CC bucket)
   reads **+2.63%**, over the 2% red-flag bar in that single bucket;
   every other CC bucket <= +0.29%, pooled 16k-96k +0.58%, and the
   echo-dominated depth that IS the CC serving regime is untouched.
   "Sustained across buckets" is structurally untestable on this
   corpus (deeper CC buckets are all own-output echo), so the
   established disambiguator was run: the 07-11 generic-corpus
   position-bucket NLL (wikitext-2, one pass, both legs fp8 at the
   262144 window -- content-diverse at EVERY depth). Result, q4s vs
   v1.4: 0-16k **-3.6/-6.8/-7.2% (q4s better)**; 16k-96k
   -1.29/-0.31/+0.96/+1.21% (pooled +0.35%); then FLAT
   +1.16..+1.35% out to 256k, non-compounding. VERDICT: the agentic
   +2.63% is content noise on an n=1 transcript (same sign-flipping
   class as the turbo3 gate's excluded shallow wiggles), not a
   systematic CC-depth defect. Ladder proceeded; flag on the record.
4. NEEDLE spot (needle_deep method, haystack trimmed to the fp8
   262144 native window -- the full 355K haystack is turbo3-only;
   prompt 248,726 tok, deepest-first): depth 60% (~149K) and 10%
   (~24K) both PASS EXACT -- **2/2**.
5. SERVING SANITY (bare W12 2-slot, `--ctx 32768 --slots 2
   --slot1-ctx 32768`, zero env): warm pass + 2 measured concurrent
   codegen+docs reps. Composition determinism: rep1 == rep2
   completion text BYTE-IDENTICAL both payloads (raw JSON differs
   only in `created`). bat=2.0/1.8 fused, gcache ON cap 64, warm reps
   pf=1 off prefix snapshots, ZERO errors. codegen solo-warm vs
   fused-rep fork = the documented A1 suffix-trim class (docs payload
   solo==fused) -- v1.4-identical behavior.
6. CLUB-3090 matched-bpw rerun (their bench.sh verbatim,
   endpoint-only, x2 passes per GPU; full table in
   docs/BENCHMARKING.md "Matched-bpw rerun"):
   - 5090 (bare W12, auto-ctx 262144): narr 146.8/146.6 wall
     (154.2/153.9 decode), code 181.6/181.5 (196.2/196.0), TTFT
     326ms, prefill 3442/3420 @10K, 2608/2562 @90K. In-run CV <=0.3%,
     pass delta <=0.2%, [req] agrees within 0.5%.
   - 3090 (vox-transcriber stopped for the window, restarted +
     verified active after; w8, fp16 KV + h16, stock ~417 W,
     `--ctx 61440`): narr 88.5/87.8 (93.2/92.4), code 99.4/99.2
     (106.9/106.7), TTFT ~565ms, prefill 1131/1122 @10K, 90K SKIP
     (>ctx, their documented path).
   - **3090 ctx FINDING (boot-verified)**: q4s + fp16 KV boots
     **61440** (49152/57344 also boot; **65536 OOMs** at
     spec_sample_graph instantiation; auto-ctx picks 69632 and OOMs
     -- the 5090-calibrated anchor miss again) = **2.5x** v1.4's
     24576 on the same club-config defaults. The 2.27 GB of freed
     weights went straight to KV, arithmetic-clean (68 KB/token).
     The SHIPPED entry's "262144 cap by arithmetic" claim is the
     TURBO3 ceiling; this is the fp16-KV bench config's.
   - READ vs v1.4 (decode): narrative **+1.6% (5090) / +5.2%
     (3090)**; code **-7% on BOTH GPUs** -- the single-Q4-head tier
     re-rolls the code acceptance basin (5090 code 3.86 -> 3.45
     tok/round; narrative 2.64 -> 2.58 barely moves), so the
     shipped-entry suite gain (+5.2%, 5-prompt mean) and a code-basin
     loss coexist: q4s wins low-acceptance traffic on bytes and gives
     ground where acceptance was carrying v1.4. vs THEIR rows at
     matched bpw: 3090 narr +54% over ik MTP (93.2 vs 60.39), code
     +5.5% over beellama DFlash (106.9 vs 101.3); 5090 narr +20% over
     their best single-5090, code -4.2% vs DFlash 204.80 (v1.4 was
     +3%). KV-bits asymmetry stated in the doc: our 3090 leg spends
     fp16 KV vs their q4_0/q5_0.
   - Ops lesson RE-paid (cost: three phantom OOM readings): a failed
     systemd-run unit needs `systemctl reset-failed` before the name
     is reused, and journal greps must scope to
     _SYSTEMD_INVOCATION_ID -- the ctx ladder's first three rungs
     read ONE stale crash as three "OOMs" (24576-40960!) until the
     identical CPU-time lines gave it away. Same trap as the 07-12
     journalctl note, new costume.

VERDICT: **q4s-v1 VALIDATED** at 0.66 bpw under the default tier.
Quality GREEN (PPL better, agentic NLL flat at depth with one
disambiguated content-noise flag, needle exact, serving deterministic,
zero errors across every server boot). Speed trade legible: narrative
+2-5%, code -7% (acceptance basin), +2.27 GB KV budget, 3090 club
config 2.5x ctx. Open items: test_gemv10_scaling q4-head skip; task
dome before quality-critical recommendations (per the SHIPPED entry);
phase-2 token_embd demotion unmeasured.

## 2026-07-16 -- q4s-v1 CONSISTENCY STUDY: do real agentic task scores hold on the repack? (thunderdome, single-slot 131K fp8)

The validation ladder above left one open item ("task dome before
quality-critical recommendations") and one legible trade: the
single-Q4-head tier re-rolls the code acceptance basin (5090 code 3.86
-> 3.45 tok/round, -7% code decode). Question: does that acceptance
re-roll (or anything else in the tier) move REAL agentic task scores?
Master 4d0a3c2, build/q27-server (W12, mtime after last src commit
aa991de), q4s md5 re-verified against CHECKSUMS.md5.

DESIGN (pre-declared, written before any result was collected):
- Tasks: the CONSISTENT SCORERS at this shape -- T2 collab-server
  (historical band 0.83-0.85; draws 0.851/0.851/0.839-0.844), T11
  debug-nightmare (0.850 x2), T5 task-queue (0.78-0.81; draws
  0.789/0.776-0.797). NOT the documented lottery modes (T8's
  auth-chain basin; the T2/T5 crash classes seen only at the 2x48K
  squeeze shape in the turbo3 gate -- the bands here were measured at
  single-slot 131K, which is what this study reproduces).
- Shape: single-slot --ctx 131072, Q27_KV=fp8, bare v0.2.0 defaults
  (fast-head + no-think + fp8 are the defaults; env pins fp8
  explicitly), port 8081, unit q27-eval, CUDA_VISIBLE_DEVICES=0,
  vanilla model files. One task at a time, sequential -- no
  concurrency anywhere in this study.
- Legs: q4s-v1 (qwen36-27b-mtp-q4s.q27) vs v1.4 (qwen36-27b-mtp.q27),
  n=3 per task per leg, alternating leg-blocks with a fresh server per
  block: q4s[T2,T11,T5], v14[T2,T11,T5], x3 = 18 runs. GPU-free check
  + reset-failed before each block; "slot 0 ready: ctx=131072"
  verified from the CURRENT invocation's journal (invocation-ID
  scoped, per the 07-16 phantom-OOM lesson).
- Thunderdome: `./thunderdome run --orchestrator claude-code-q27-haight
  --task TN --trials 1`; scores from trials meta.json composite_score
  + exit_reason (harness exit codes ignored per the standing note).
- PASS RULE: q4s passes if per-task medians land within +-0.03 of the
  same-day v1.4 leg medians AND within/above the historical bands;
  single outlier draws reported descriptively (standing register:
  n=3 cannot separate small effects). A consistent multi-task deficit
  >0.05 = flag for the owner, no rationalizing.
- Also collected per run: decode tps + tok/round from [req] telemetry
  (the code-acceptance question: do T2/T5 turn walls/tps shift on
  q4s?), errors (must be zero), wall times (descriptive only).

RESULTS (18/18 runs completed, ZERO [req-error]/end=error across all
runs and 6 server boots; every boot's "slot 0 ready: ctx=131072"
verified invocation-scoped; study wall ~53 min):

  block  leg  T2 collab          T11 debug        T5 task-queue
  b1     q4s  0.553 @175s        1.000 @38s       0.619 @141s
  b2     v14  0.535 @212s        1.000 @39s       0.200 @9s
  b3     q4s  0.553 @307s        1.000 @38s       0.620 @145s
  b4     v14  0.547 @533s        1.000 @50s       0.200 @10s
  b5     q4s  0.545 @146s        1.000 @37s       0.625 @153s
  b6     v14  0.272 @67s         1.000 @48s       0.200 @9s

  task  q4s med (spread)      v14 med (spread)      delta    hist band
  T2    0.553 (0.545-0.553)   0.535 (0.272-0.547)   +0.018   0.83-0.85
  T11   1.000 (1.000-1.000)   1.000 (1.000-1.000)    0.000   0.850
  T5    0.620 (0.619-0.625)   0.200 (0.200-0.200)   +0.420   0.78-0.81

Basin anatomy (sub-scores): T2 draws the SAME basin on BOTH legs all
6 runs -- hidden_tests 0.07 / tests 0 / agent 1.00; composite spread
is coverage/code_metrics wiggle (v14 b6 0.272 = outlier draw that
also lost agent_tests, 67s). T11: tests+static 1.00 flat, both legs,
all 6. T5: v14 lands the DOCUMENTED deterministic one-shot-quit 3/3
(the [drift] UN-RESCUED first-tool-call class -- 2 reqs, 151 decode
tokens, 9-10s, identical every rep; same class the turbo3 gate logged
for fp8 T5 at 2x48K, now on the reference tier at single-slot 131K);
q4s escapes it 3/3 into full 141-153s sessions (hidden 0.18, coverage
0.88-0.91). Per the standing register that is a basin-lottery re-roll
at the first tool call (different tier bytes, different trajectory),
not KV/weight quality signal -- but the direction is q4s ABOVE.

Telemetry ([req] aggregates per run; trajectory-confounded,
descriptive only -- the code-acceptance question):

  task  leg  tok/round (med)   per-req med tps (med of 3)   agg tps
  T2    q4s  5.29/5.47/5.77 (5.47)   233/254/255 (253.7)    259-294
  T2    v14  4.64/5.65/8.66 (5.65)   224/235/237 (235.2)    233-401*
  T11   q4s  4.81/4.88/4.99 (4.88)   259/259/260 (259.9)    234-244
  T11   v14  4.70/4.80/4.87 (4.80)   232/239/246 (238.6)    227-235
  T5    q4s  4.53/4.59/5.27 (4.59)   235/240/241 (240.2)    228-263
  T5    v14  (151-token quits, not comparable)
  (* v14 b4 = 533s/140K-tok echo-deep outlier: tok/round 8.66, agg 401)

The club-bench -7% code-decode re-roll does NOT reproduce on real CC
traffic: q4s tok/round is -3% on T2 (5.47 vs 5.65) and at parity on
T11, while per-request median tps reads q4s +8% (T2 253.7 vs 235.2)
and +9% (T11 259.9 vs 238.6) -- the 12.8% byte cut outruns the
acceptance mix shift on agentic turn shapes, exactly as it did on the
5-prompt suite. Wall times: T11 q4s 37-38s vs v14 39-50s every draw;
T2/T5 walls are volume-basin dominated (T2 v14 67-533s).

VERDICT (against the pre-declared rule, stated plainly): the rule as
written does NOT return a clean PASS -- the band clause fails on T2
(0.553 < 0.83) and T5 (0.620 < 0.78), and T5's median delta (+0.420)
is outside +-0.03. But every one of those violations is either shared
by or in favor of q4s: the same-day v1.4 CONTROL misses the same
bands identically (T2 0.535) or worse (T5 0.200), and the +0.420 is
q4s ABOVE the control. The deficit flag (consistent multi-task
deficit >0.05) does NOT fire -- q4s median >= v1.4 median on all
three tasks. Finding of record: **NO q4s-attributable score deficit;
real agentic task scores HOLD on the repack** (T2 +0.018, T11 tie at
ceiling, T5 +0.420 via basin escape). The historical bands themselves
did not reproduce ON THE REFERENCE TIER -- thunderdome is unchanged
since 03-18 (same tasks, same rubrics, same image), so the band drift
is the ENGINE era: the bands were minted on 07-08/07-10 binaries and
defaults, and the documented cross-build tie-lottery (fdmma/fast-head
defaults, W12, gemm-verify, suffix/auto7 -- many rebuilds since) has
re-rolled the greedy trajectories. FLAGS FOR THE OWNER (independent
of q4s, no rationalizing): (1) the fp8-131K-era bands are STALE under
master 4d0a3c2 -- re-base before using them as gates again; (2) v1.4
now draws T2's hidden_tests-0.07 basin and T5's deterministic
one-shot-quit at the canonical single-slot 131K shape -- the [drift]
first-tool-call rescue miss is worth a look on its own; (3) T11's
composite now reads 1.000 both legs (tests+static only in the
rubric's fired components) vs the 0.850 era draws. The validation
ladder's open item "task dome before quality-critical
recommendations" is CLOSED: q4s holds real agentic task scores at the
reference shape.

## Appendix: early milestones, progress log, and M6 prefill history (moved verbatim from the README, 2026-07-16)

The README carried these from the start of the project; they moved here
in the 2026-07-16 editorial slim-down. Each row/block reflects the state
of knowledge at the time; current canonical numbers live in the README
("State of the engine", "Decode methodology").

### Milestones

- **M0** DONE -- repack tool: BF16 GGUF -> q27 4-bit format (policy v1.2)
- **M1** DONE -- correctness: greedy decode, output verified vs llama.cpp
- **M2** DONE -- dp4a GEMVs + CUDA-graph decode: 80.1 t/s plain
- **M3** DONE -- MTP speculative pipeline, lossless (token-identical):
  depth-2 drafting, batched verify, 3-perm cyclic state graphs. **146.0 t/s**
  (llama.cpp MTP fork on same model/GPU: 101.5). Stretch target was 165;
  verify-GEMV bandwidth floor makes the remaining gap ~1-2%/iteration work.
- **M4** DONE -- dual lm_head (Q4 draft / Q8 verify), grid merges, device-side
  round bookkeeping. `--fast-head` opt-in: **156.5 t/s**
- **M5** DONE -- HTTP serving: OpenAI + Anthropic + OpenAI Responses, exact
  byte-level BPE tokenizer (gated 21/21 vs llama-tokenize), tool calling
- **E6** DONE -- ungated depth-3 speculation: measured p(d3 | d1,d2 correct)
  = 83.7% offline (docs/E6-design.md), so the round always drafts 3 and
  batch-4-verifies {pending, d1, d2, d3}. 4 GDN buffers under a mod-4 role
  permutation, 4 captured graphs. 3.12 tok/round, **188.9 t/s** @2k
  (204.8 long-gen) [superseded -- P3 depth-4; see the README's Decode
  methodology]; 8000-token output bit-identical to depth-2. Also fixed
  two latent bugs found en route: flash-decode scratch under-allocation at
  ctx<4128, and missing ctx guard letting spec rounds write KV rows past
  max_ctx (silent corruption the prefix cache could then reuse).
- **CB (P0-P3)** DONE 2026-07-15/16 -- continuous batching across slots:
  P0 `LaneView` state split (07-15), P1 lockstep conductor + fused
  verify sweep (07-15, **1.21x** 2-slot aggregate), P2a/b/c
  overlap-vs-fusion attribution + fused draft steps (07-16, **1.31x**),
  P3 table twins + shape-keyed CUDA-graph round replay (07-16,
  **1.41x** both KVs, live-CC-validated). Solo cost ~0% and
  byte-identity (ninv + seam + twin legs) at every stage; serving
  default since 07-16. **P4 mixer co-residency: measured NO-GO**
  (07-16) closes the campaign -- saturated-work physics, ~0.1ms/round
  net for a numerics-gate price.

### Progress log (tg t/s, greedy, token-identical output verified each step)

Chronological -- each row supersedes the previous. Current canonical numbers
live in the README ("Decode methodology").

| change | t/s |
|---|---|
| reference kernels e2e | 43.4 |
| dp4a int8-activation GEMVs | 58.8 |
| coalesced delta state + wide norms + multiblock argmax | 66.5 |
| CUDA-graph token replay, device-chained decode | 75.9 |
| delta_step i-parallel v2 | 80.1 |
| + speculative decode depth-1 (host-driven) | 84.2 |
| + direct-write batched GEMV | 92.2 |
| + parity-pair captured graphs | 109.3 |
| + depth-2 drafting (2.13 tok/round) | 107.3 |
| + grid-merged 3-token small kernels | 115.1 |
| + dual lm_head: Q4 drafts, Q8 verify (v1.3 repack) | 121.1 |
| steady state (128-token bench, 2.39 tok/round) | **133.5** |
| `--fast-head` opt-in (Q4 verify; output differs, coherent) | 143.0 |
| + full grid merges (l2/f16/gates/rope/kv/attn/sigmoid/embed x3) | 145.8 lossless / 156.5 fast |
| + device-side round bookkeeping (1 sync + 16B readback/round) | 146.0 lossless / 156.5 fast |
| E1: display compositor off GPU 0 (cosmic-comp/Xwayland stole ~10%) | **157.4** lossless / **168.5** fast |
| warp-cooperative decode attention (coalesced K/V) | **168.6** lossless @2k; 65.8 @8k ctx (~2x long-ctx) |
| flash-decode (split-K, K/V shared across GQA heads) | **173.1** @2k / **159.6** @8k ctx lossless; 178.1 fast |
| fp16 KV cache (attn + MTP) | 169.7 @2k / 159.7 @8k; halves KV bytes, -2.1GB @32k ctx |
| E2: GDDR7 mem offset +4000 (tools/mem_oc.py, volatile) | **176.6** lossless / **185** fast-head; prefill ~+6% |
| E6: ungated depth-3 speculation (3.12 tok/round; batch-4 verify) | **188.9** @2k (128-tok) / **204.8** long-gen; 8000-token output bit-identical to depth-2 |
| P1: int8 tensor-core prefill GEMM (mma.sync m16n8k32) | prefill **1380 t/s** @600 / **1384** @4K (dp4a: 592/580, 2.35x); cold 28.1K TTFT **63.8s -> 35.7s**; PPL delta vs dp4a +0.04% (fp reorder only) |
| P1.5: fp16 tensor-core flash-attention prefill (m16n8k16) | cold 28.1K TTFT **35.7s -> 24.3s** (63.8s at day start, 2.63x total); prefill 1408 @600 / **1508** @4K; PPL 7.2139 (+0.006% vs exact); needle 3/3 @64K; kernel review: 0 confirmed bugs |
| v1.4 quant policy (ssm_out + attn_output -> Q8, +0.98 GB) | PPL **7.1928** (-0.29%); decode **+3.3%** on 2000-tok soak (acceptance 3.47 -> 3.67 t/round -- cleaner residual writers agree better with the MTP draft head); all gates re-derived |
| P2: fp8 E4M3 KV cache (opt-in, `Q27_KV=fp8`) | decode @28.5K ctx **105.7 -> 117.2 t/s** (+11%); 2K soak 208.3 vs 210.4 (-1%, acceptance 3.64 vs 3.67); ctx ceiling **~180K -> ~370K** (262K native fits); PPL 7.1889 (-0.05%), needle 3/3 @55K, logit KL 3.4e-5 |
| P3: depth-4 speculation (batch-5 verify, mod-5 perm) | 2K soak **210.4 -> 218.6 t/s** (4.36 t/round, 71% of rounds accept 5); 28.5K-depth fp8 **117.2 -> 126.6** (+8%; +19.8% vs pre-P2); canonical md5 unchanged (lossless); gate: p(d4\|prefix-3) measured 97.4% |
| P4: split-position FA prefill (SM-starvation fix) | attention kernel **1.93x** @26.6K; 128K prefill **~1.96x** (153 -> 78s); cold 28.5K TTFT **24.7 -> 21.4s**; cold **361.5K request 1324 -> 764s** (~12.6 min, needle exact); split-vs-exact 1.9e-5, combine cost 0.1% |
| P5: GEMM tile tuning (grid swap + reg pipeline + vector unpack + NT=64) | Q4 GEMM **-36%** / Q8 **-48%** @26.6K; prefill **1388 -> 1790 t/s** @600; cold 28.5K TTFT **21.4 -> 16.8s** [superseded -- P6: 15.0s]; 128K prefill ~78 -> ~57s [re-measured 2026-07-06: current 128K prefill is ~71-80s (fp8 g64 71.5 / fp16 exact 80.4); both this 57s and P6's 117.6s superseded]; arithmetic bitwise-unchanged (canonical + pf IDENTICAL) |
| P6: column-split delta scan (SM-starvation fix #2) | kernel **748 -> 413 us** @T=256 (1.81x, 48 -> 384 blocks); 26K prefill wall **15.0 -> 13.5s** (-10.3%); 28.5K **16.7 -> 15.0s**; 128K **125.5 -> 117.6s** (fp16-KV kvstats method) [superseded 2026-07-06: current 128K prefill ~71-80s after g64 regroup + delta-WY tiling]; split-vs-exact 5e-8, PPL 7.1931 (+0.0003 = fp reorder), canonical md5 exact, pf IDENTICAL |
| fd2: register-accumulator flash-decode (SM-starvation/occupancy fix #3, attn was 99% of depth cost at 5% DRAM BW) | 61K depth **78.0 -> 126.2 t/s** (+62%, 47.2 -> 29.2 ms/round); 16K **-18%/round**; instance 0.768 -> 0.156 ms @61K (45% DRAM BW); 2K +1.3%/round; acceptance parity exact; PPL in noise both KV modes; nll-long 160K bucket-identical; CANONICAL RE-DERIVED 4c4120c7 (old 58b6ae85 under Q27_FD=v1) |
| P12: confidence-gated depth (`p_min` equiv; `Q27_PMIN=theta`) -- gate verify width on the drafter's top1-top2 margin, skip the deep-KV verify when unconfident | decode **grows with ctx: 2K neutral / 16K +5.8% / 60K +10.8%** (theta 1.0; +7.0% theta 0.5); greedy output BITWISE-IDENTICAL (lanes are independent grid indices -> only round count + verify width change); higher theta wins at longer ctx (context-adaptive theta confirmed). P12b depth-5 (`Q27_MAXD=5`, opt-in): agentic +2.6% but docs -8% (always drafts to max, so the 5th MTP pass is pure cost at low acceptance) -> depth-4 stays default; adaptive maxd is the follow-on |
| P14 Task 2: fuse draft argmax+margin (`k_argmax_top2`) -- kills the dead ungated `k_margin` scan | -0.545 ms/round @61K (the removed scan); canonical 4c4120c7 EXACT (bitwise); test_kernels +3 fused assertions (token==argmax, margin==CPU top1-top2, all err 0) |
| P14 Task 3: P12 confidence gate ported to the sampled spec path (per-width sampled verify graphs, capped accept walk) | sampled verify-narrowing ALONE is a wash @61K docs (+0.0% theta0.5 -- a low-margin draft the sampler may accept gets skipped, tok/round drops, extra rounds offset the cheaper round); greedy cross-check healthy on the same binary (+6.6% theta1.0); substrate for Task 4; canonical 4c4120c7 EXACT (greedy untouched) |
| P14 Task 4: draft early-exit (`Q27_DEXIT`, margin-gated per-step draft graphs, `min(W,md_used)` width-floor top-up) | same-binary A/B @61K docs: greedy **+3.2%** (theta1.0), sampled **+5.4%**; emitted bytes + round counts bitwise-identical to the monolithic draft in all 8 identity cells; sampled gated+dexit now **+3.6% over ungated** (Task 3's sampled wash resolved); canonical 4c4120c7 EXACT |
| P14 Task 5: fd2 lane-innermost grid order (partial cross-lane KV L2 reuse; R~4.25 measured) | same-session pre/post A/B @61K ungated **116.1 -> 119.3 t/s (+2.7%, MARGINAL-KEPT)**; verify fd2 per-instance -10% toward the draft floor; 2K neutral (+0.0%); canonical 4c4120c7 EXACT (2-line index remap, bitwise on the full fd2 matrix) |
| prefill-attn Phase 1: cp.async K/V double-buffered prefetch (fp8 path) | fp8 128K prefill **72.1 -> 68.2s (+5.4%)**; bitwise (convert-on-consume of identical bytes); first "neutral" reading was an fp16-KV test artifact -- cp.async is dead code off the fp8 path |
| prefill-attn Phase 2: fp8 QK^T MMA (`mma.sync.e4m3`, Q staged fp8, bank-conflict padding) -- DEFAULT-ON on fp8 KV | 128K prefill **68.3 -> 59.6s (+11.8%**, ~2200 t/s); logit cosine 0.9999827 + argmax MATCH @131K; needle **6/6 to ~301K**; fp16 path + canonical untouched; `Q27_PF_FP8MMA=0` opts out |
| verify-gemv: activation reads 4x uint2 -> 2x uint4 in `k_gemv_q4_n` (+ single-col) | decode @61K **163.2 -> 172.9 t/s (+5.9%)** on 2026-07-08 fixtures; GEMV was LATENCY-bound (long_scoreboard 90%, 39-47% DRAM peak) -- weights were fine, the per-column activation loads hammered L1TEX; bitwise BY CONSTRUCTION (same bytes, same dp4a order); tensor-core verify NOT justified |
| accept-gate Phase 1: conditional lane-5 yield + `maxd_lo` 0.10 -> 0.35 (the measured d5 crossover) | `Q27_MAXD=auto` becomes the production rec: **+2.7% geomean over d4-gated** across the 5-payload envelope, beats BOTH fixed ceilings; the old unconditional yield EMA sat above the demote bar on traffic where fixed-d5 measured -1.7% |
| maxd6: adaptive ladder 4..6 (7-lane verify, perm mod-7, +157 MB; 3-bar depthctl hi/hi6/flo6) | real-CC-transcript @25.8K: d4 202.6 / d5 216.1 / d6 222.0 (7-tok rounds on 64%); **auto 220.7 vs d5 211.9 = +4.2% same-harness** (2026-07-09 review rerun; original +4.7% claim mixed harnesses); text byte-identical at every ceiling; canonical 4c4120c7 EXACT; non-saturating flavors never promote past 5 |

2K-soak series (2000-token generation, the long-generation methodology
tier; headline for agentic reply-length outputs): **209.2 t/s STOCK
fd2-era** (4.32 t/round; pre-fd2 213.2/4.36 -- the ~2% is the short-ctx
split tax).

k_vgemm T8 figure carried only in the README until 2026-07-16: real T8
agentic suffix rounds measured 24.76 -> 20.85 ms on the live harness
(the echo/W12 replay figure of record is 24.76 -> 19.96 ms; see the
2026-07-13 GEMM-verify entries).

Headline numbers from E2 onward include a GDDR7 offset. Consumer GDDR7
has no ECC and weights load once, so a marginal OC can plant a persistent
silent error the token-identity gates can't see. That happened on
2026-07-02 at +4000 (one wrong canonical run after 30 min of heat, then
clean again -- binary confirmed innocent). Daily offset is +3000 since:
the band above it bought ~0.4% and produced the soft error. +4000 only
for short supervised benches; `--verify-weights` / `/health?verify=1` is
the detector; offset is volatile across reboots.

### Prefill (M6)

Batched prefill: 256-token chunks, smem-staged dp4a GEMM (16 rows/block share
one activation tile; per-lane accumulation order matches the serial GEMV
exactly, so prefill is bitwise-identical to the serial path -- gated on
identical continuations). GDN state scans sequentially inside one kernel with
S resident in shared memory; attention runs two-pass softmax in 32-token
sub-batches; MTP warm skips attention/FFN (only the K/V stores matter).

| prompt | serial | batched | speedup |
|---|---|---|---|
| 512 | 76 t/s | 567 t/s | 7.5x |
| 4096 | 53 t/s | 453 t/s | 8.5x |

**Prefix cache (M6.5)**: GDN state + conv rings snapshotted after prefill
(attention/MTP KV rows are append-only, so prefix rows stay valid); next
request LCP-matches the snapshot and prefills only the suffix. Claude Code
turn 2 on a 26.7k-token context: **1.3s** (26,670/26,693 tokens reused)
[superseded -- see P8: this gate replayed raw tokens, a flow no real client
takes; re-rendering clients missed the cache 100% of the time until the P8
stable-prefix snapshot]. Unconditionally correct: any mismatch falls back to
full prefill; warm-vs-cold continuations gated identical.

Real-world (Claude Code `claude -p`, 26.7k-token system prompt):
| | TTFT |
|---|---|
| pre-M6 (serial prefill) | 15-min timeout, 0 tokens |
| M6 (batched) | 139s |
| + coalesced attention prefill | 90s |
| + GEMM tuning + FA-lite attention | 61s |
| turn 2+ with prefix cache | **1.3s** |

[historical -- cold 28.5K TTFT is ~15.0s after P1-P6 and cold 128K is 59.6s
after the 2026-07-07/08 prefill-attn pair; the warm-turn number required the
P8 stable-prefix snapshot to hold on real re-rendering traffic]
## 2026-07-17 -- auto-ctx recalibrated (measured-free sizing) + turbo3 takes the 3090 to 262144

Two asks in one pass: fix the 5090-calibrated auto-ctx anchor that
over-sized on the 3090 (69632 pick vs 61440 real ceiling), and find out
how far q4s + turbo3 pushes a 24 GB card.

**Root causes, stated exactly (src/server.cu sizing block):**
1. All four per-token KV constants omitted the MTP pair: 34e3/68e3/
   13.6e3/41.6e3 are 17-pair (attn-only) numbers; the engine allocates
   18 K/V pairs (17 attn + 1 MTP; Engine::kv_bytes). Exact per-token:
   fp8 36864, fp16 73728, turbo3 14400, turbo3v 44064 B.
2. Weights entered the estimate as stat(model file) -- upload alignment
   and tier drift unmodeled.
3. The non-weight base (1.27 GB) was anchored pre-P1-P3; the graph zoo
   grew ~1.4 GB since. Measured today (in-process free deltas): 5090 W12
   fp8@131072 non-KV stack = 4.49 GB; 3090 w8 fp16@61440 = 4.22 GB;
   3090 w8 turbo3@262144 = 4.30 GB. The 07-16 club rerun's 61440 fit was
   a knife edge: **free at ready = 0.00 GB**.

**Fix (server.cu only):** sizing moved AFTER upload_all() -- budget from
MEASURED free VRAM with weights resident (tier size, alignment, and any
co-tenant process fall out of the measurement). Exact 18-pair per-token
bytes (MIRROR WARNING against Engine::kv_bytes). Arch-calibrated base:
sm_120 0.89 GB, sm_86 1.77 GB (fd2/h16 workspaces are heavier), + the
existing (W_MAX+1)*0.157 role + W_MAX*0.13 graph terms. Slack 1.0 ->
0.25 GB (it was sized for the stat-based estimate this replaces; the
4096 floor adds 0-0.3 GB more). Two new banner lines: `vram: free X.XX
GB post-weights` and `... at ready` -- the calibration probe is now a
standing part of every boot.

**Verification matrix (all boots real, q4s tier unless noted):**
- 3090 fp16 auto: **57344**, boots, 0.29 GB at ready (was: picks 69632,
  OOMs). One deliberate 4096-step below the 61440 knife edge.
- 3090 turbo3 auto: **262144**, boots, 0.67 GB at ready.
- 5090 fp8 auto: 262144 (cap-bound), 2.95 GB at ready.
- 5090 fp8 tier boots: v1.4 **262144** / q6 **192512** / q6k **122880**
  (0.75 / 0.53 / 0.31 GB at ready). README tier numbers refreshed.
- Retro-check: the corrected model predicts the hand-found v1.4 3090
  ceiling (24576) exactly.
- Gates: diff is server.cu-only (sizing + two prints; no engine/kernel
  touch); both arches build clean; serving determinism smoke 2x
  byte-identical on the new binary.

**turbo3 x q4s on the 3090 (the capacity headline):** a 24 GB card now
boots the FULL 262144 native window (4.27x the fp16 61440, 2x the 07-11
131K-on-v1.4 record). turbo3 KV at 262144 = 3.77 GB vs fp16's 4.53 GB
at 61440 -- the format IS the ceiling on this card. Needle 6/6 PASS on
a ~233K-token haystack (depths 10-95%, deepest ~222K), first-hit
2m59s wall for the entire 6-ask run -- 3090 turbo3 prefill sustained
>1300 tok/s, no measured 2.2x prefill tax at this shape. Club-harness
decode bench (their bench.sh verbatim, 3 warmup + 5 measured, vs the
07-16 fp16@61440 leg on the same card):

| leg (3090, q4s, turbo3@262144) | wall t/s | decode t/s | TTFT |
|---|---|---|---|
| narrative (n=5, CV 0.2%) | 89.45 | 94.23 | 567ms |
| code (n=5, CV 0.0%) | 108.32 | 117.38 | 570ms |
| prefill 10K cache-busted (n=3) | 1096 tok/s | -- | 9.2s |
| prefill 90K-class cache-busted (n=3, ~70K-tok runs) | 643 tok/s | -- | 145s |

vs the 07-16 fp16@61440 leg, same card, same harness: narrative +0.9%
wall / +1.1% decode, code **+9.0% wall / +9.8% decode**, TTFT equal.
The 5090's turbo3 decode tax (5-30% by depth) INVERTS on Ampere: fp16
streams 4096 B per KV pair per token, turbo3 800 B, and on a
bandwidth-starved part the KV-read savings beat the dequant compute --
the same physics triad, applied to KV bytes. Prefill tax is ~3% here
(1096 vs 1131 tok/s at 10K), not the 5090's 2.2x. NET: on sm_86,
turbo3 is BOTH the capacity lever and the speed pick -- narrative
94.2 / code 117.4 decode at 262144 ctx beats every published
single-3090 club row on both axes while quadrupling their best q27
fp16 window. OPEN (product call, not taken here): the Ampere CC
profile still defaults fp16 KV with "turbo3 opt-in recommended"
(server.cu profile block); today's numbers argue for flipping that
default on sm_86.

vox-transcriber stopped for the test window and restarted after
(3090 must be dedicated; its 2.7 GB is the difference between boot and
OOM at these shapes). Open (unchanged): --slots still defaults --ctx
8192 (multi-slot auto-size is its own roadmap item); test_gemv10_scaling
q4-head fixture gap.
## 2026-07-17 -- Ampere pass: turbo3 default on sm_86, TTFT 8x (serial-prefill fix), 4-stream 5090

Three results from the afternoon block (Gabe: "we should default to
turbo3 on ampere for sure" + "do an Ampere tuning pass" + "could we get
4 streams on 5090?").

**1. turbo3 is the sm_86 serving default** (Gabe sign-off). CC profile
now sets Q27_KV=turbo3 on cc_arch 80..88 (overwrite=0: user env wins;
ref profile keeps fp16). Verified: a bare `q27-server-w8 model tok`
boots kv=turbo3, auto-ctx 262144 on the 3090.

**2. Four streams on the 5090: YES, on the w8 build.** Slot-cost
ladder measured with q4s (free 16.83 GB post-weights):
- fp8 W12 4x48K: slots 0-1 ready, slot 2 SKIPPED (3.9 < 5.2 GB) --
  the shipping 2x48K shape is the fp8 W12 ceiling.
- turbo3 W12 4x48K: slots 0-2 ready (0.39 GB spare) -- turbo3 buys
  the THIRD stream on the standard build.
- turbo3 w8 4x48K: ALL FOUR ready, 0.18 GB spare. Per-slot fixed is
  the wall (~4.6-4.8 GB each on W12: borrowing engines carry their own
  role sets + graph zoo; KV is minor at these shapes), so the narrower
  build's ~1.2 GB/slot savings is what unlocks slot 3. Aggregate
  throughput at 4 lanes is UNMEASURED (w8 union cap = 8 -- 4 lanes at
  trim floor 2 saturate it); capacity claim only.

**3. TTFT root cause found and fixed -- the club-table anomaly dies.**
generate_prefill routed prompts < 32 tokens down a SERIAL walk: per
token, two ungraphed full forwards (trunk + MTP) and two stream syncs
-- measured ~22 ms/token on sm_86, ~11 on sm_120. The club bench
prompts are 17-23 tokens: 23 x 22 ms = the entire 567 ms "TTFT floor"
(5090's 350 ms = same structure at its speed). Worse, the serial path
CLEARS the slot's snapshot + checkpoint ring -- a tiny prompt routed to
a slot destroyed its conversation cache (live-CC relevant, not just
bench optics). The chunked path already handles arbitrary small tail
chunks, so the fix is a threshold knob: Q27_PF_BATCH_MIN (engine
default 32 = exact old behavior; CC profile sets 2; floor 2 because
NP=1 would snap_save an empty prefix). Measured on the 3090: 23-token
prompt pf_ms 533 -> 69 (7.7x); 7-token 151 -> 62.

SERIAL-vs-CHUNKED IS NOT BITWISE (measured: tiny-prompt greedy text
diverges between paths) -- expected, the chunked path is what every
prompt >= 32 already takes, and path identity is part of the config.
Consequences handled:
- The 5-token canonical prompt is serial-path BY CONSTRUCTION under
  the CLI default 32: canonical a2982c5197c627551b27d76a0a94b220
  (vanilla) and f64e7c02252ca4c40cea62db662205e0 (q4s) both EXACT on
  the new binary, script-exact extraction.
- Sub-default settings print a banner ("the 5-token canonical md5 does
  NOT hold here") -- same failure class as the gemm_min guardrail, but
  a banner not a refusal since the server profile sets 2 deliberately.
- test_kernels ALL PASS; same-state repeat determinism byte-identical.
- A 3-request probe showed request 3 diverging from 1-2 on REPEATED
  identical prompts: depth-ladder lineage carry (md4/md5 round-mix
  shifts 51/4 -> 51/57 -> 96/67), the documented Q27_MAXD_RESET
  semantics -- pre-existing on both paths, not this change.

**Decode-side profile (the tuning-pass measurement, 3090 turbo3):**
round wall 31 ms = verify 25.2 (81%) + draft 5.7 (18%), 3.94 tok/round
on the code probe. Per-width verify: W2 21.3 ms -> W6 28.4 ms -- a
gentle +1.5 ms/width slope on a ~21 ms floor, NO spill cliff. The
floor is the weight stream itself: 15.46 GB / 936 GB/s = 16.5 ms
theoretical => sm_86 decode already runs ~78% BW efficiency. Remaining
per-kernel headroom ~4.5 ms/round; identified follow-up levers, NOT
taken today: Q27_GEMV_2CTA_MIN is compile-time (=10, never fires on
w8 widths -- an sm_86 occupancy sweep needs rebuild-per-point) and
vgemm-below-width-9 collides with the gemm_min canonical guardrail
(gate_maxd+1 >= gemm_min refuses). Both are half-day items for a
bounded ~5-15%; the TTFT fix was the big fish.

**Club-harness bench, both cards, post-fix (their bench.sh verbatim):**

| card / leg (q4s, auto-ctx) | wall t/s | decode t/s | TTFT | pre-fix TTFT |
|---|---|---|---|---|
| 5090 fp8 narrative (n=5, CV 0.1%) | 161.14 | 161.97 | **31ms** | 350ms |
| 5090 fp8 code (n=5, CV 0.0%) | 191.99 | 193.67 | **33ms** | ~350ms |
| 3090 turbo3 narrative (n=5, CV 0.3%) | 89.65 | 90.07 | **53ms** | 567ms |
| 3090 turbo3 code (n=5, CV 0.0%) | 115.50 | 116.43 | **55ms** | 570ms |

3090 prefill unchanged (10K 1095, 90K-class 655.6 tok/s cache-busted).
Wall throughput absorbs the TTFT win directly: 5090 narrative wall
146.8 -> 161.1 (+9.7%), 3090 code wall 108.3 -> 115.5 (+6.6%). Decode
rates re-roll +-1-5% with the new tiny-prompt continuation text (the
chunked-path numerics produce a different greedy transcript; same
class as any config change). TTFT columns vs club: their best
single-5090 ~51ms -> we're 31-33; their 3090 class ~51ms -> 53-55 =
parity. The last column they led is gone.

vox-transcriber stopped for the window and restarted after.

## 2026-07-17 -- 4-stream aggregate: fits, but the ceiling is ~250 t/s at 2 lanes

Gabe asked what 4 slots actually yield. Scaling curve on the 5090, q4s,
turbo3 KV, campaign payload methodology (tools/batch_ab.sh style: warmup
lands per-slot snapshots, N payloads fired simultaneously, aggregate =
sum(dec)/window, median of 3; scripts scratchpad/batch_ab_4slot*.sh;
q4s BY NECESSITY -- v1.4 cannot boot the 4-slot shape):

| streams | build | aggregate t/s | per-stream | vs solo |
|---|---|---|---|---|
| 1 | w8  | 161.9 | 161.9 | 1.00x |
| 2 | w8  | 250.4 | 136.2 | 1.55x |
| 3 | W12 (3x45056) | 248.6 | 86.9 | 1.54x |
| 4 | w8  (4x40960) | 221.2 | 60.6 | 1.37x |

READ: the union weight sweep amortizes the full weight stream by 2
lanes -- aggregate PLATEAUS at ~250 and 4 lanes REGRESSES (w8 union cap
8 = ~2-wide verify per lane at 4 lanes; acceptance collapses, bat 3.0-
3.1 avg lanes steady). Stream count past 2 buys concurrent users, not
tokens: 2x136 / 3x87 / 4x61. The 07-15 "W12 2x48K = CC-viable batch
shape" default stands; 3-4 slots are a fan-out option, not a
throughput lever.

**CRASH FOUND + WORKAROUND (open robustness item):** the first 4-lane
attempt (4x49152, GRAPH_CAP=64 default, 0.18 GB headroom) served 23
fused rounds then DIED: cudaGraphInstantiate OOM at conductor.h:1698
-- the P3 exec cache instantiates graphs LAZILY per new shape key, and
the boot-time cap-shrink guard does not cover runtime growth. A tight
boot passes health and then crashes mid-traffic. Workaround measured:
step ctx down one notch (4x40960 frees 0.47 GB) + Q27_BATCH_GRAPH_CAP
=24 (LRU evicts within headroom) -> 16/16 clean. PROPOSED FIX (not
implemented): wrap the instantiate site in evict-LRU-and-retry, then
fall back to the ungraphed fused round -- same shrink-never-abort
philosophy as the ctor guard; the 4x49152/cap-64 shape is the natural
regression test. Second finding: rep 1 of a fresh 4-lane server ran
95.3 t/s vs 221 steady (graph-capture warmup tax at 4-lane alphabet
size); W12 3-lane showed no such dip.

## 2026-07-17 -- gcache instantiate-OOM fixed: evict-at-cap reorder + shrink-never-abort retry

The 4-lane crash (previous entry) root-caused and fixed in
conductor.h's graph_round miss path. TWO defects:
1. Evict-AFTER-instantiate: at the cap boundary the path held cap+1
   execs transiently, so the ctor headroom guard's shrunk cap (176 MB
   -> cap 20) still overflowed by one exec instantiating entry 21.
   Eviction now runs BEFORE capture/instantiate.
2. No OOM recovery: cudaGraphInstantiate ran under CUDA_CHECK =
   process abort. Now: on cudaErrorMemoryAllocation, clear the sticky
   error, evict LRU, retry; if the cache is EMPTY and it still OOMs,
   destroy the capture, banner, graphs_on_ = false for the rest of the
   run, re-stage perms (M1 posture) and serve the round eagerly --
   the same recovery shape as the guard trip, extending the ctor's
   shrink-never-abort contract to runtime growth. Non-OOM instantiate
   errors keep the loud-abort contract.

GATES: fused_smoke ALL PASS (A2 error leg + graph legs byte-identical
to solo, both passes). REGRESSION (the exact crashing shape, 4x49152
default cap): 16/16 served, ZERO OOM events -- the reorder alone
covers it; the retry/disable path stays as armor for per-exec-size
drift (4-lane execs can exceed the 8 MB estimate). Aggregate
reproduced 221.4 vs 221.2 protocol-matched. S1 solo 162.0 vs 161.9
EXACT. S2/S4 medians moved -2..-3% across reruns but the FIXED binary's
own protocol/run-to-run spread spans the delta (S4 221.4 vs 216.9 same
binary same hour; S2 229.5 fresh-boot vs 242.4 after-S1): harness
variance, hit path untouched by the diff.

## 2026-07-17 -- sm_86 GEMV occupancy sweep = NEGATIVE (kernel win, zero round transfer)

Ampere-pass item #2 (Gabe: "let's start with #2"). The 4/3/2-CTA
launch_bounds tier boundaries (Q27_GEMV_2CTA_MIN=10, 3CTA_MIN_Q4=4,
3CTA_MIN_Q8=6) were swept on the 5090; re-swept on the 3090 for the
w8 ladder range N=2..8. Tool: tools/gemv_tier_sweep.cu (kept; build
line below). Vanilla model (benchmark rule).

WHAT THE MICROBENCH FOUND (isolated gemv, 100-rep cudaEvent, 3090):
the q8 head/writers (248320-row) SPILL under the shipped 4/3-CTA
register caps, and 2-CTA (128-reg, 0 spill) is fastest-or-tied at
EVERY N=2..8 -- N=4 is a 2.24 ms -> 1.47 ms cliff (-52%), others
-12..-30%. ptxas confirms: 4-CTA q8 spills at N=2,3,4,7,8; 2-CTA
clean everywhere. q4 ffn was already near-optimal (<=3.4% off,
non-monotonic). Looked like a clean, large, monotonic win.

WHY IT DOES NOT SHIP -- adversarial round-level verify (the win did
NOT transfer):
- Implemented as an __CUDA_ARCH__<890-gated q8->2-CTA pin (sm_89+
  bit-identical). Correctness PASSED cleanly: 5090 canonical
  a2982c51/f64e7c02 EXACT; sm_86 old-vs-new CLI byte-identical on
  BOTH models (register-alloc-only, values invariant); test_kernels
  ALL PASS both arches.
- Club decode A/B (3090 w8 turbo3, same-session old vs new binary):
  narr 90.01 -> 89.42, code 116.46 -> 116.35 = FLAT (narr dip inside
  OLD's 0.3% CV).
- Q27_PHASE_STATS A/B (codegen payload, real ~20K prompt, per-width
  verify-ms buckets) = the decisive read: phv 3617.6 vs 3621.7;
  dominant width-6 bucket (59 rounds) 2044.2 vs 2044.5 ms; EVERY
  phwm bucket identical within 0.2%. The microbench's 13% q8 win at
  width 6 produced ZERO round change.

ROOT CAUSE / LESSON: kernel-isolation microbench overstated the win.
In a tight back-to-back loop the q8 head's spill local-mem traffic
contends on the 3090's 6 MB L2; in the actual verify round that single
head GEMV is bandwidth-floored (~1.4 ms streaming 1.27 GB either way)
and dwarfed/overlapped by the 65-layer forward. The round is
weight-stream-bound at ~78% BW efficiency (the 07-17 Ampere profile
said exactly this) -- a register retier cannot beat the weight stream.
REVERTED (src/kernels.cu untouched from 1c0b1b1); tool + sweep data
kept for future arch re-sweeps. Verdict stands with the physics triad:
BW-bound work has no occupancy lever.

Sweep repro (sm_86, 3 forced-tier builds, transcriber stopped):
  nvcc -O2 -std=c++17 -gencode arch=compute_86,code=sm_86 -DTIER_TAG='"3CTA"' \
    -DQ27_GEMV_3CTA_MIN_Q4=0 -DQ27_GEMV_3CTA_MIN_Q8=0 -DQ27_GEMV_2CTA_MIN=99 \
    tools/gemv_tier_sweep.cu src/kernels.cu src/spec3.cu src/vgemm.cu src/blocks.cu \
    src/prefill.cu src/device_model.cu src/loader.cpp -o gemv_3CTA
  (4CTA: all _MIN=99; 2CTA: all _MIN=0; run on GPU 1, vanilla model)

## 2026-07-17 -- W12-on-3090 = NO-GO (width ceiling is 8 on sm_86); vgemm-below-9 closed; auto-ctx W-slope fix

Ampere-pass item #1 (Gabe green-light). q4s freed the VRAM that forced
the w8 build, so the W12 fatbin was tried on the 3090 for the first
time. VERDICT: w8 is not a VRAM compromise on Ampere -- it is the
per-token optimum, the sm_86 twin of the 07-13 "W16 no-go, W12
optimum" finding. Every card has a width ceiling; Ampere's is 8.

GATES FIRST (h16 verify had never run widths 9-12 on sm_86): ninv on
the 3090 ALL PASS incl. W=12 + TWIN legs (vanilla model; ninv aborts
on the q4s Q4-head -- same fixture gap as test_gemv10_scaling, on the
books). Serving determinism 2x byte-identical; first-request output
byte-matched w8 (bitwise-when-untrimmed: the gated ladder is
build-invariant, only suffix rounds differ).

CAPACITY: W12 non-KV fixed on sm_86 = 6.57 GB vs w8's 4.22 (measured
at ready, ctx 32768). The four extra widths cost 0.59 GB each -- the
width-9..12 graph zoo runs 0.43 GB/width on sm_86 vs 0.13 on sm_120.
turbo3 ceiling: ~143K vs w8's 262144 (-44%).

THROUGHPUT (3090, turbo3, ctx 32768, q4s):
- club bench FLAT (narr 90.00 / code 116.58 vs w8 90.01/116.46) --
  short prompts, identical gated ladder.
- codegen payload (26.8K prompt, 512 tok, 3 reps): W12 92.3/91.0/70.6
  tps vs w8 100.3/96.9/96.9 -- REGRESSES. Suffix rounds DO pin at the
  new cap (12.0 tok/round) but cost 97 ms vs 43 ms at width 8: 2.25x
  the wall for 1.5x the tokens. The wide round is NOT flat in width on
  sm_86 (unlike the 5090 post-GEMM-pivot).
- echo payload: 11.6 tok/round at 98 ms -- same shape.
- ATTRIBUTION (one boot, Q27_GEMM_MIN=13 forces the GEMV family at
  width 12; guardrail-legal): suffix round 92.6 ms vs vgemm's 97 --
  within 5%. vgemm is NOT the culprit; the width cost is structural
  (attention at ntok=12 + GDN chain on 82 SMs). THIS ALSO CLOSES
  LEVER #3 (vgemm-below-9 on sm_86): the two families are within 5%
  at width 12, so there is no vgemm advantage to harvest at 6-8.

SHIPPED (one real defect found): a bare `q27-server` (the W12 default
binary) + turbo3 on a 3090 auto-picked 217088 and DIED in
build_spec_graphs (engine.cuh:1967, no runtime recovery in the solo
zoo -- same class the conductor got armored against, on the books).
auto-ctx graph term is now piecewise: 0.13 GB/width up to 8 on all
arches, 0.43 GB/width above 8 on sm_86 (calibrated from today's two
measured points; sm_120 and all W<=8 arithmetic unchanged). Verified:
W12 auto on the 3090 now picks 131072 and boots, 0.43 GB spare.

AMPERE PASS CLOSES: TTFT fix + turbo3 default SHIPPED; occupancy
sweep NEGATIVE; W12 NO-GO; vgemm-below-9 CLOSED. The 3090 config is
settled: w8 + turbo3 + q4s @ 262144, narr 90 / code 116.5 decode,
TTFT 53 ms. The only remaining Ampere decode lever is weight bytes
(the quant ladder) -- the 16.5 ms weight-stream floor itself.

## 2026-07-17 -- sm_86 depth/pmin policy sweep = NEUTRAL (defaults transfer; no flip)

Ampere lever: the dctl bars + auto7 + pmin 0.5 encode 5090 round
economics; the 3090's steeper width slope (+8%/width) suggested a
shallower optimum. Env-only sweep, 3090 w8 turbo3 q4s ctx 32768,
fresh boot per config, codegen/docs/echo x4 (rep1 warm, medians;
scratchpad/depth_sweep_3090.sh + depth_sweep_out/).

- AUTO6 == BASE EXACTLY on all three payloads: md7=0 in every BASE
  row -- the auto7 ladder never promotes past 6 on this traffic. The
  controller's own saturation bars already do the arch adjustment;
  the auto7-vs-auto6 question is moot on sm_86.
- PMIN06: docs -1.8%, codegen -15.3% (trajectory-confounded draw:
  early-eos re-roll) -- dead.
- PMIN04: docs +4.2% (tok/rnd 2.96->3.18), echo +2.5%, codegen -4.1%.
  De-confound leg (BASE vs PMIN04, codegen x7, median of 6, rule
  pre-declared at <=2% deficit to flip): deficit HOLDS at -2.7% with
  huge per-draw variance (81.0-107.8 vs BASE's 96.2-96.3) and lower
  acceptance (3.37 vs 3.74 tok/rnd). NO FLIP.

VERDICT: the 5090-tuned policy defaults transfer to sm_86 unchanged.
Q27_PMIN=0.4 goes on the record as a docs/echo-flavored OPTION
(+4.2/+2.5%) that costs codegen consistency -- not a default. The
software side of the Ampere pass is now fully closed: shipped = TTFT
fix, turbo3 default, two auto-ctx calibrations; closed-negative =
occupancy retier, W12, vgemm<9, policy re-tune. Remaining levers are
non-kernel: memory OC (hardware, moves the BW floor directly),
2-slot batching on the 3090 (throughput, untried), prefill constants
PF_T/PF_SB (sized for 170 SMs, long-prompt TTFT only), and the quant
ladder (the decode floor itself).

## 2026-07-17 -- PF_T/PF_SB sweep on sm_86 = shipped constants already optimal

Last untried Ampere software item (Gabe: "let's look at prefill
constants"). PF_T=1024 was sized to fill 170 SMs; hypothesis was 82
SMs might prefer smaller (plus 0.4-0.8 GB scratch back to turbo3 ctx)
and PF_SB=32 might matter at depth. Made both -D-overridable
(Q27_PF_T / Q27_PF_SB, defaults unchanged -- behavior-invariant,
canonical a2982c51 EXACT on the rebuilt binary), parallel-built five
sm_86 w8 variants, cache-busted prefill probes (unique-head prompts,
hit=0, pf_ms from [req]) at ~10K and ~70K tokens x3, turbo3, ctx
98304, transcriber down:

  T1024_SB32 (shipped)  10k 1175.7   70k 916.0   free@ready 2.88 GB
  T512_SB32             -3.1%        -1.6%       +0.40 GB
  T2048_SB32            -1.2%        +0.1%       -0.81 GB
  T1024_SB16            -1.8%        -0.4%       --
  T1024_SB64            -1.6%        -0.4%       --

VERDICT: the 5090-sized point is ALSO the sm_86 optimum at both
depths -- the GEMM saturates by T=1024 on 82 SMs and the T=2048
weight-re-read halving nets ~zero. T512's 0.40 GB scratch refund
(+29K turbo3 tokens) is not worth -3%/-1.6% with ctx already at
262144. No change; overridability kept (future sweeps are one -D).
This closes the Ampere software surface COMPLETELY: every lever is
now either shipped (TTFT, turbo3 default, auto-ctx cals) or closed
with data (occupancy, W12, vgemm<9, depth/pmin policy, prefill
constants). Remaining: memory OC (hardware), 3090 2-slot batching
(throughput), quant ladder (the floor).

## 2026-07-17 -- graph-zoo capture gates (issue #1): sampled set env-gated, mono D/V auto-skipped

The two in-thread promises from issue #1, landed. The zoo had two
capture-unconditional sets with narrow consumers:

1. **Sampled set** (sample_graph + spec_sample_graph[12] +
   verify_sample_graph_w[4][12]): serves ONLY temperature>0 requests.
   Now Q27_SAMPLED=0 skips capture entirely. The server refuses
   temp>0 with a 400 (code sampling_disabled, all three API shapes,
   preflight before slot claim); Q27_SAMPLED=0 + Q27_FORCE_TEMP>0 is
   a boot-time FATAL (two-tier precedent: contradictory explicit
   config refuses loudly); make_decode_task carries a belt that
   forces any slipped-through sampled task GREEDY with a stderr line
   (task-filler layer: never half-build, never null-launch).
2. **Monolithic draft pair + verify_graph** (P11 split set): consumers
   are exactly the constrained-tool path and the Q27_DEXIT=0 A/B
   (header map; sampled-gated-dexit uses per-step graphs only, line
   ~2482). The server now clears capture_constrained when booted
   without --constrain-tools, so STANDARD boots skip them
   automatically -- no new knob. CLI leaves both gates default-true:
   the canonical zoo is byte-identical.

MEASURED (same binary, full-zoo --constrain-tools control, ctx 8192):
  sm_120: mono 80 MB, sampled 340 MB (combined 420 MB)
  sm_86:  mono 150 MB, sampled 600 MB (combined 750 MB = ~55K turbo3
          tokens -- the sm_86 sampled graphs are fatter, consistent
          with the W12 zoo finding)
auto-ctx now deducts the measured savings per arch (constrain_tools /
sampled_on aware). Verified on the 3090: bare boot auto 262144 with
0.81 GB at ready (was 0.67); Q27_SAMPLED=0 boot 262144 with 1.42 GB.
For the A10 in issue #1 the arithmetic says a greedy-only boot now
reaches the full 262144 native window (his 212,992 + ~55K, capped).

GATES -- ALL PASS: canonical a2982c51 + f64e7c02 EXACT; sampled-seed
anchor 900031e9 EXACT (the sampled zoo is untouched when on);
test_kernels + fused_smoke (graph legs byte-identical); serving
matrix: DEFAULT boot serves greedy+sampled with mono skipped;
SAMPLED=0 400s temp>0, greedy byte-deterministic 2x; FORCE_TEMP
contradiction FATALs; DEXIT=0 boot captures mono D and serves (the
banner's three-state marker covers D/V independently). One false
alarm en route: a "determinism DIVERGE" that was md5-of-full-JSON
(created timestamp) -- text-only extraction passes; harness lesson
re-learned same-day.

## 2026-07-17 -- v0.3.0 RELEASED (tag @ e8a6e46)

github.com/signalnine/q27/releases/tag/v0.3.0 -- "small cards get the
whole window". 23 commits since v0.2.0, one day: turbo3 = Ampere
serving default (262144 on a 24GB 3090, decode faster than fp16 was),
TTFT 8-11x (Q27_PF_BATCH_MIN), q4s tier validated end-to-end with its
own anchors, auto-ctx rebuilt on measured free VRAM + exact 18-pair
KV bytes + capture-gate awareness, Q27_SAMPLED=0 + mono auto-skip
(~750MB back on sm_86), gcache OOM fix, 4-stream measured (~250 t/s
ceiling at 2 lanes), four negative results on the record (occupancy,
W12-on-3090, vgemm<9, depth/pmin), tools needle_check.py +
gemv_tier_sweep.cu. Pre-tag doc-staleness sweep: 4-reader parallel
audit vs ground truth found 19 stale claims (5 blocking: front-page
131K headline, fp16-default claim, serial-threshold listed as open,
multislot 2-cap rationale, notes fp8-opt-in) -- all fixed in e8a6e46.
Assets: q27-v0.3.0-linux-x86_64.tar.gz (4 binaries + MIT LICENSE,
sha256 af7118e4...) + SHA256SUMS-0.3.0. Release-binary canonical
a2982c51 EXACT re-verified at package time.

## 2026-07-17 -- issue #1 receipts: A10 confirms 262144; brim-serving verified; the "identical VRAM" anomaly is malloc granularity

Field results at a31108a (chaudhryfaisal): **q4s + Q27_SAMPLED=0 +
maxd4 boots the FULL 262144 on the 22.6 GiB A10** (prediction held);
v1.4 doubled to 102400 (was 49152). His v1.4 106496 OOM died at the
suffix verify instantiate (engine.cuh:1968) -- the solo-zoo
no-runtime-recovery edge, already on the books.

He also reported a runtime OOM under live Claude Code at ctx 212992
on the OLDER build (666b7d9 era, 22-31 MiB boot spare). Reproduction
attempt on the 3090, CURRENT build, his exact recipe at 262144 with a
1.2 GB balloon pinning free to 154 MiB: 200K-token cache-busted
prefill (2m28s) AND 1024-token generation at 200K depth both served
cleanly -- free VRAM byte-identical before/after (zero runtime
allocation; boots = serves holds on this build). His old build
predates the gcache evict-order fix and the capture gates and ran at
5-7x less headroom; ask for the crash's "at src/..." line if it
recurs on a31108a.

His "used_after identical at 258048 vs 262144" observation SOLVED,
exactly: turbo3 KV per buffer = ctx x 400 B; at 262144 that is
EXACTLY 100 MiB, at 258048 it is 98.44 MiB -- and cudaMalloc's 2 MiB
granularity rounds both to the same 100 MiB granule across all 36
buffers (delta = 0). His 253952 row confirms: 96.88 -> 98 MiB granule,
a 72 MiB step (he measured 68). Reproduced on the 3090: both ctx
values read 1.70 GB at ready. The window's last 4096 tokens are
allocation-free. Benign; physics, not engine.

## 2026-07-17 (late) -- first 4090 (sm_89): field test on a RunPod pod; tri-arch fatbin ships

Gabe spun up a RunPod 4090 for q27's first Ada run. Everything below
was executed remotely over SSH on the pod.

TOOLCHAIN FINDING: **CUDA 12.4+ is a hard floor for sm_89** -- the
pod's stock CUDA 12.0 ptxas rejects the e4m3 MMA forms ("Unexpected
instruction types specified for 'mma'", prefill.ptx). Installed the
12.6 toolkit alongside (driver untouched); tri-arch (86/89/120)
compiles clean there and on our 13.2.

GATES on sm_89 -- ALL PASS, with a cross-arch determinism result:
- CLI canonical (q4s) = **8196e65e... = the sm_86 anchor, byte-exact.**
  One canonical anchor per SASS family (8.x vs 12.x), not per chip.
- Serving greedy probe = byte-identical to the 5090's output for the
  same prompt (fp8 path agreeing across sm_89/sm_120).
- Serving determinism 2x byte-identical. Model md5 verified.

NUMBERS (q4s, w8 build, their bench.sh verbatim on-pod, CVs <=0.1%):
fp8 (arch default): narr 102.06/102.58, code 135.48/136.64, TTFT 49ms
@ auto-ctx 110592. turbo3: code 130.95/132.03, TTFT 50ms @ the FULL
262144 (0.27 GB spare). vs their best published single-4090 (ik
two-stage 82.5/120.9 @160K): +24% narr / +13% code at 1.6x context.
turbo3 tax ladder across arches complete: 5090 fp8-wins-big, 4090
fp8-wins-small (-3.4% code on turbo3), 3090 turbo3-wins-outright.
Prefill on Ada near-5090-class (~3.1K tok/s @10K on turbo3).

SHIPPED:
- Makefile (Gabe-approved): sm_89 gencode added -- the fatbin now
  covers 3090/4090/5090. v0.3.0 release binaries predate this; README
  notes the workaround (build from source with 12.4+, or Q27_KV=turbo3
  explicit) until the next release.
- auto-ctx sm_89 base = 2.13 GB (two pod boots agreeing within 75 MB;
  Ada's fixed stack runs ~1 GB over the sm_120 constants -- the
  pre-calibration pick survived on a 40 MB margin). The >8-width graph
  slope on sm_89 is unmeasured and deliberately shares sm_86's fat
  slope (under-pick beats a dead boot).
- README: CUDA 12.4 floor, tri-arch build, 4090 guidance; BENCHMARKING
  4090 addendum.
- Post-change gates on the tri-arch binaries: canonical a2982c51 +
  f64e7c02 + sampled-seed 900031e9 EXACT, test_kernels ALL PASS.

ODDITY on the record: the tokenizer file vanished from the pod's
overlay disk between download (listed, 7.2 MB) and first use
("cannot open"). Re-downloaded + checksummed, did not recur. Cloud
overlay-fs distrust noted; CHECKSUMS verification is now part of the
pod recipe.

## 2026-07-17 (late) -- external review of BENCHMARKING.md: label + claim fixes; q8-v1 repacked

Review feedback (via Gabe) on the five-engine doc, all points taken:
1. "NVFP4" mislabel FIXED everywhere it described q27's own tiers --
   v1.4/q4s/q6/q6k are q27's Q4_G64/Q8_G128 integer group quant, NOT
   NVIDIA's e2m1+fp8-scale NVFP4; only vLLM's unsloth checkpoint is
   literally NVFP4. Now labeled "q27 4-bit, 5.25 bpw effective".
2. "Quality is engine-independent" softened to "quality converged to
   the model once both tool protocols were validated" -- the
   strict-parser episode (T8 0.00) proved engines DO move quality
   through tool-protocol failures.
3. 4090 addendum re-dated 07-17-late (UTC slip).
4. "vLLM has no /v1/messages" annotated: current vLLM Python frontend
   serves it natively (Anthropic->OpenAI double adapter, ~04/2026);
   our 07-14 run used the litellm shim; native-endpoint rerun flagged.
5. Method B n=1 disclosure -> n=3 seal RUNNING tonight (q27 +
   llama+MTP ceiling legs).

Also tonight: q8-v1 tier repacked (--q8 '.*', 28.45 GB / ~8.1 bpw,
867 tensors, worst Q8 rel-RMSE 0.0153; keeps the v1.3-style Q4
draft-head copy -- a --q8-head acceptance variant is the obvious
follow-up). UNVALIDATED: no card in the house fits it (weights +
fixed stack > 32 GB); upload to HF in progress; the ladder runs on
rented 48 GB+ hardware (PRO 6000 / RTX 6000 Ada -- doubles as those
cards' field test).

## 2026-07-17 (late) -- Method B n=3 seal: 1.59x sealed; an early-quit reproducer surfaces

Review item #3 executed: 12 pinned SWE-bench instances x3 reps, q27
(bare-boot reproduce recipe, vanilla, HEAD de5564c) then mainline
llama+MTP (13e67386, doc launch line verbatim). Throughput SEALED:
q27 192.4 t/s (678 reqs / 143.7K tok) vs llama 120.9 (703 / 213.6K)
= 1.59x decode, 1.58x wall (n=1 claim was 1.74x; both engines moved
a few percent with n and engine evolution). llama's 12/12 nonempty
reproduced exactly; q27 drew 10-11/12: pallets__flask-5014 is a
DETERMINISTIC early-eos (45 tok, end=eos, byte-identical x3, the
dome's one-shot-quit class, was 12/12 on the 07-14 build) and
pydata__xarray-4094 a 2-of-3 no-edit lottery (one 63-turn spin =
the drift-loop shape). Doc updated with the sealed table + honest
quality columns. flask-5014 = the standing reproducer for the
early-quit class; the [drift] first-tool-call rescue item now has a
deterministic test case. Results archived scratchpad/results.*.rep*.

## 2026-07-18 -- q8 ladder on an RTX PRO 6000 (96GB): anchors minted, quad-262K boots; TWO bugs surfaced

Gabe rented an RTX PRO 6000 Blackwell Server (96GB, sm_120, driver
570/CUDA 12.8, 224 cores) for the q8-v1 ladder. Results and the two
bugs the trip surfaced, in order:

**BUG 1 -- v0.3.0 release binaries have an undocumented driver floor.**
The tarball's statically-linked CUDA 13.2 runtime refuses drivers
older than r580: "driver version is insufficient" at first CUDA call
on this pod's 570.195 (CUDA 12.8) stack. Every ladder rung silently
produced empty output on the release binaries (and the ladder's
2>/dev/null ate the diagnostic -- harness lesson re-learned AGAIN;
ladder v2 keeps stderr and fail-fasts on build-sanity). Workaround:
source-build with the driver's toolkit (CUDA 12.8 = first sm_120
support; builds clean, canonical EXACT). TODO: state the driver floor
in the release notes + README; consider a 12.8-runtime build for the
next release's binaries.

**LADDER (source-built sm_120 @ HEAD, all on-pod):**
- Build sanity: vanilla canonical a2982c51 EXACT (fourth silicon).
- q8-v1 canonical MINTED: a5eddc71c12a1f3a43ebec479cb1458b x2 EXACT.
- q8-v1 sampled-seed MINTED: e85bded3f5e2b99481b812bbe263cd62 x2.
- Serving: bare boot = fp8 @ auto 262144, 56 GB spare at ready;
  determinism 2x byte-identical; needle 6/6 at ~233K.
- Club bench (their bench.sh): narr 97.45/97.99, code 131.11/132.42,
  TTFT 56ms. Code decode 132 vs the pure byte-scaling prediction
  ~105 from q4s's 192: the Q8 ACCEPTANCE RECOVERY IS REAL (+26%
  over bytes -- the q4s Q4-head acceptance loss un-happens at Q8).
- **Quad-slot: 4 x 262144 turbo3 ALL READY with 28.89 GB still
  free** -- 1,048,576 tokens of resident context on one card, W12
  build, stock binary. The --slots clamp (4) is now the binding
  limit on this hardware, not VRAM.
- PPL: paired serial-path run IN PROGRESS (see bug 2); absolute
  serial numbers are not comparable to the batched-path anchors
  (g64 regroup, notes.md #6), but the paired delta is valid.

**BUG 2 -- batched --nll path has a latent OOB read, sanitizer-
deterministic, lottery in the wild.** First seen as sm_0 + "illegal
memory access" at prefill.cu:2085 on the pod (both tiers), then
reproduced at HEAD on the 5090. Bisect misdirection: four suspect
commits all tested clean; a tri-vs-dual-arch build split also
evaporated on 3x reruns (n=1 observations -- the harness-variance
lesson applied to crashes). Ground truth via compute-sanitizer
memcheck: k_embed_rows_q8_T reads emb + tok*cols with a GARBAGE
token -- Invalid __global__ read, block (1,17,0), IDENTICAL address
across runs and across dual/tri builds, 3426 errors. So: not a
race in the upload (same-stream ordered), not codegen, not arch --
a deterministic-under-instrumentation bad token index in the
batched nll flow, timing-lottery without instrumentation. The
serving prefill path shows no such fault (needle at 233K, live CC,
club prefill legs all clean + sanitizer-clean runs elsewhere).
REPRODUCER: compute-sanitizer --tool memcheck build/q27 <model>
--nll wiki.test.raw --ctx 2048 (fires in <60s). Scope: quality
measurement only; serving unaffected. OPEN, top of tomorrow's list.

## 2026-07-18 -- nll "bug 2" RETRACTED: wrong input file; one real fix shipped (step_with) + input guards

Resolution of yesterday's "batched --nll latent OOB": there was no
batched-path bug. The --nll flag reads RAW INT32 TOKEN IDS; the house
corpus is wiki.test.qwopus.i32 (tokenizer byte-identical across
vanilla/qwopus). I fed wiki.test.raw -- raw TEXT -- on both machines:
ASCII reinterpreted as int32 yields ~540M "token ids" (the guard now
names it: id 540876810 at offset 0), embed reads emb + id*cols
terabytes out of bounds. Same bytes = same addresses everywhere
(the "deterministic OOB"); usually lands on mapped memory (the six
"clean" runs read silent garbage); occasionally unmapped (the
natural crashes, incl. both pod tiers). The 07-16 PPL anchors used
the correct .i32 and are untouched.

What the chase surfaced anyway, both kept:
1. **step_with stack-lifetime fix (REAL bug, independent):**
   cudaMemcpyAsync(d_token, &token) from the parameter's stack slot
   with no sync -- a driver that defers pageable staging reads a dead
   frame. Now a synchronous 4-byte copy (cold paths only; ordering
   note documented: relies on stm being a BLOCKING stream). step_taps
   same fix. Canonical a2982c51 + f64e7c02 + sampled-seed 900031e9
   all EXACT after.
2. **--nll input guards:** both int32 loaders now scan ids against
   VOCAB and refuse with a diagnostic naming the first bad id --
   raw text can never masquerade as a corpus again. (First guard
   attempt landed twice in the wrong loader -- regex patching; the
   verify caught it because the text file still "worked".)

Method lesson for the ledger: the bisect chased two phantom
correlations (commit suspects, tri-vs-dual fatbin) that n=1 crash
observations manufactured. The sanitizer's DETERMINISTIC address was
the only honest witness, and its "nearest allocation 512 B" line was
pointing at input data, not engine state, from the first report.

## 2026-07-18 -- q8-v1 ladder CLOSED on the RTX PRO 6000; default-tier PPL anchor reproduced cross-machine

Final rungs, correct corpus (wiki.test.qwopus.i32 -- the tokenizer is
byte-identical across vanilla/qwopus), exact anchor protocol
(--nll-chunk 2048, 148,335 predictions):
- **default tier: PPL 8.0409 -- the 07-16 anchor to FOUR DECIMALS**,
  on different silicon (PRO 6000 vs 5090), different toolchain (12.8
  vs 13.2), different machine. The measurement stack reproduces.
- **q8-v1: PPL 7.9942** (-0.58% vs default). Better than default and
  q4s (8.0197); NOT the family floor -- q6 7.9460 / q6k 7.9127 still
  lead on wikitext. Third data point on error cancellation: the
  tuned mixed promotions beat blanket-Q8; PPL is non-monotonic in
  bits in this family. q8's distinct value is elsewhere: the
  acceptance recovery (code decode 132 on this card vs ~105 pure
  byte-scaling -- the q4s Q4-head acceptance loss un-happens) and
  near-lossless verify-side weights as the reference point.

q8-v1 LADDER SUMMARY (all on the PRO 6000, v0.3.0-era HEAD):
canonical a5eddc71 x2 + sampled-seed e85bded3 x2 MINTED; build-sanity
a2982c51 EXACT; needle 6/6 @233K; serving deterministic; club narr
97.99 / code 132.42 @ fp8 auto-262144; 2-slot aggregate ~181 t/s
(bat 2.0, the 1.55x multiplier holds on Blackwell-Pro); quad-slot
4x262144 turbo3 ALL READY with 28.89 GB spare. Not run: the task
dome (Gabe's call whether q8 needs one; q4s precedent says the
score-lottery dominates PPL-class deltas anyway). README tier table
gains the q8 row + the release-binary driver floor (r580+) is now
stated in the Quickstart.

## 2026-07-18 -- PRO 6000 q4s club leg completes the four-card table

Same tier, same harness, apples-to-apples at last: PRO 6000 Server
Edition q4s fp8 @ auto-262144 = narr 138.55/139.32, code
169.35/170.81, TTFT 40ms (CV <=0.5%). ~12% UNDER the 5090 despite
the bigger die: the Server Edition's inline GDDR7 ECC + passive-SKU
clocks tax exactly the bandwidth decode lives on. Fleet reading:
5090 = single-stream king (162/194); PRO 6000 = the capacity card
(85.1 GB post-weights free, quad-262K, 181 t/s 2-slot, q8 host);
4090 = the fp8 midpoint; 3090 = the 262K value card. Same-card tier
read: q8 costs 22% decode vs q4s for 84% more weight bytes --
acceptance recovery holds it sublinear. Pod work complete.
