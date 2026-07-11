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
earlier entry) remains their one winning cell.

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
