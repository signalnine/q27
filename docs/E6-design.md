# E6: depth-3 speculation -- design

> **2026-07-02 Phase-0b RESULT (supersedes the gated design below):**
> measured p(d3 | d1,d2 correct) = 83.7% ungated over 512 positions
> (design assumed ~65%). Every margin bin pays for itself (worst bin:
> 25.9% prefix x 71.4% d3 = +0.185 t/round vs ~5.5% round tax), so E6
> ships **ungated always-on depth-3**: no D/V graph split, no second host
> sync, no top2_margin kernel, no theta. One fused graph per perm (4
> perms), round drafts 3 (pass 3 chains from pass-2 hidden) and
> batch-4-verifies {t1, dr1, dr2, dr3}. Everything else below (4-buffer
> mod-4 rollback, P3->p[4] widening, finish_round4, lane-d buffers, MTP
> KV semantics, phased gates) applies unchanged. Conservative net
> estimate ~+15% decode (round-level prefix acceptance 64.7% x 83.7% =
> +0.54 t/round on 2.53, minus ~5.5% round-time cost).

Goal: extend the depth-2 MTP self-speculation round to depth 3 when the pass-2
draft is confident, per E3 data (margin>=2 covers 64% of rounds at ~86% draft-2
acceptance). Projected +5-8% net decode. Lossless: emitted tokens remain exactly
the main model's greedy sequence; depth only changes how far ahead we verify.

## Current round (depth-2, one fused graph per perm)

1. `prep_round`: pos_a=P+1, pos_b=P+2, pos_c=P+3, pos_m=P+1, pos_m2=P+2;
   outcome[1] = t1 (pending token).
2. MTP pass 1: (h_next, t1) @ pos_m -> dr1. Pass 2 chains MTP's own
   post-head-norm hidden (x1): (x1, dr1) @ pos_m2 -> dr2.
3. Batch-3 verify of {t1, dr1, dr2} @ {P+1, P+2, P+3} through the main model
   (grid-merged kernels, lanes indexed by blockIdx.y via P3{p[3]} structs).
4. GDN state: 3 physical buffers, role r -> physical (r+perm)%3.
   conv/delta chains: a: R0->R0 (in place), b: R0->R1, c: R1->R2.
5. `finish_round`: n = 1 + [va==dr1] + [va==dr1 && vb==dr2]; d_token=v[n-1];
   h_next = x1 lane n-1; dP += n; outcome {n, t1, dr1, dr2}. perm += n-1 (mod 3).
6. Host: one graph launch + 16B outcome readback + sync per round.

## Phase 0a: scratch sizing fix (independent bug)

Flash-decode partials need ntok * N_HEAD * FD_NS * FD_ST floats
(3*24*16*258 = 297K floats = 1.16MB), independent of ctx. Allocated:
3 * N_HEAD * max_ctx * 4 bytes (589KB at ctx=2048) -- overrun at ctx < 4128,
masked because all downstream activation buffers are write-before-read per use.
Fix: allocate 4 * N_HEAD * FD_NS * FD_ST * 4 bytes (4 lanes, E6-ready = 1.58MB).
Gate: canonical 32/128-token output identical, perf flat.

## Phase 0b: measurement gate (E7 precedent: measure before building)

E3 measured p(d1)=88.1%, p(d2|chain)=73.9%, and draft-2 acceptance by pass-2
margin -- but never p(d3 | chain-2). The +5-8% projection assumed it.
Extend --stats N with a pass-3 chain: (x1 of pass 2, dr2) @ pos+3 -> d3,
pend3 keyed to seq position idx+3, storing {d3, margin2, d1, d2}.
At scoring: prefix_ok = (d1, d2 both matched truth); report per margin2-bin
and cumulative theta in {0.5, 1, 2, 4}:
  f(theta), p(prefix_ok | >=theta), p(d3 | prefix_ok, >=theta),
  extra tokens/round ~= f * p_prefix * p_d3.
DECISION: proceed only if projected net >= +3% at some theta
(cost model: ~4-5% of round for the extra MTP pass on gated rounds only,
+~1% batch-4 verify delta on gated rounds, +~0.1% second sync).

## Target round (gated depth-3)

Graphs (9 executables):
- **D** (draft, perm-independent -- MTP touches no GDN state):
  prep_round4 (adds pos_d=P+4, pos_m3=P+3) + pass 1 + pass 2 + `top2_margin`
  kernel on pass-2 mtp_logits -> d_margin (raw logit units, matching E3 bins).
- **V3[perm]** (4): embed3 + batch-3 verify + finish_round (unchanged semantics).
- **V4[perm]** (4): copy x1->h_next3 + MTP pass 3 ((h_next3, dr2) @ pos_m3 ->
  dr3) + embed4 + batch-4 verify of {t1,dr1,dr2,dr3} @ {P+1..P+4} +
  finish_round4: n = 1+[a]+[ab]+[abc], outcome {n, t1, dr1, dr2, dr3} (24B).

Host loop per round: launch D; readback d_margin (4B) + sync; pick V4 if
margin >= theta (env Q27_D3_MARGIN, default from Phase 0b, inf = disabled);
launch V; readback outcome + sync. Two syncs per round; added GPU idle
~10-20us on a ~14ms round (~0.1%). CUDA conditional graph nodes are the
escalation path if the gap measures worse.

GDN state: 4 physical buffers (adds S_spare3/ring_spare3, ~157MB), role r ->
physical (r+perm)%4. V4 chains d: R2->R3. perm += n-1 (mod 4). Invariant
unchanged: role 0 = last-committed state; delta_step/conv_step write dst fully,
so stale roles never leak.

Kernel widening: P3/CP3/IP3/XQ3 gain a 4th slot (p[4]); existing {{a,b,c}}
brace inits compile unchanged (p[3]=nullptr, unused at width 3). Launchers gain
an ntok param (default 3); lanes ride blockIdx.y (gdn_gates3: blockIdx.x).
mm3 -> mm_n via existing gemv_q4_n/gemv_q8_n (nbatch 2..4 already supported).
New: xqD (4th XQuant), lane-d activation buffers (~10MB), logits2 4*VOCAB,
d_pos_d, d_pos_m3, d_draft3, d_vd, d_margin, h_next3, outcome widened to 32B.

MTP KV: pass 3 writes speculative row P+3; identical semantics to today's
speculative pass-2 row (stale chained rows only degrade future draft
acceptance, never emitted tokens -- verify uses main-model KV only).

## Phased gates (canonical 32/128-token sequence must match at every commit)

1. **P1**: 4th buffer set + perm mod 4, still fused depth-2 graphs (4 perms).
   Gate: canonical + perf flat.
2. **P2**: graph split D + V3[4], margin kernel + readback + live margin
   histogram (sanity vs E3 bins), always depth-2. Gate: canonical + perf
   loss <= 0.5%.
3. **P3**: V4 path + finish_round4 + theta gate (first theta=inf -> identical;
   then enable). Gate: canonical IDENTICAL + tokens/round up + net t/s up.
   Sweep theta in {1, 2, 4} on 512-token gen @2k.
4. **P4**: @8k bench, server smoke, README + memory update.

## Cost/benefit (to be re-grounded by Phase 0b)

Gain: f * p(prefix|gate) * p(d3|prefix,gate) extra tokens/round on a 2.53
tokens/round base. With f=0.64, p_prefix~=0.76, p_d3~=0.65: +0.32 t/r = +12%
raw. Cost: ~5% MTP pass 3 * f + ~1% batch-4 delta * f + ~0.1% sync = ~4%.
Net ~+8% (upper end of E3 projection; p_d3 is the unknown).

## Risks

- p(d3|chain-2) collapses (chains degrade) -> Phase 0b kills E6 cheaply.
- Graph-split idle gap larger than estimated -> measure at P2 before V4 work.
- Warm-up must execute every kernel the graphs contain (lazy module init)
  before capture: warm one D + one V3-shaped + one V4-shaped round, then reset
  all state (incl. spare3) exactly as build_spec_graphs does today.
- emit array widens to 4 (engine.cuh generate loop, engine.cu spec loop,
  server.cu) -- audit all spec_round callers.
- pos_d = P+4 must stay < max_ctx (same guard class as existing P+3).
