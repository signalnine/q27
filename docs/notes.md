# q27 standing notes: risk register + program status

Institutional state that is neither chronological (BUILDLOG) nor
methodology (BENCHMARKING). Moved verbatim from the README in the
2026-07-16 editorial slim-down; statuses are current as of that date.

## Risk register

1. **Gated DeltaNet decode kernel** is the new risk center (was "simple dense" until we read the GGUF). llama.cpp's implementation is the semantic reference; validate per-layer.
2. 4-bit quality on a 27B: keep sensitive tensors high-bit, add importance-weighted scaling if PPL regresses > ~3% vs Q5_K_M. **STATUS: MEASURED + MITIGATED 2026-07-02** -- v1.3 measured +3.35% vs Q5_K_M (7.2135 vs 6.9797, identical tokens/protocol via `--nll`); v1.4 policy (residual writers to Q8, chosen by a 6-candidate sensitivity study -- see build log P0.5) lands at **7.1928 = +3.05%**, and decode got faster (+3.3%, acceptance-coupled). Still marginally over the 3% bar; the study shows uniform promotion cannot close the rest at acceptable cost (ffn_down ceiling probe: 29% of gap for +2.76 GB) -- importance-weighted scales are the documented path if ever needed. The t/s comparison remains bit-width-assisted (15.8 GB reads vs 18.2).
3. M-RoPE sections must match exactly or long-context quality silently degrades.
4. MTP acceptance rate must survive quantization (draft and verify disagreeing more = less speedup). STATUS: measured -- Q4 vs Q8 draft-head argmax agreement 98.1% (E3); depth-3 runtime acceptance 85.7%.
5. **Long-context correctness: VALIDATED to 361K (2026-07-02, fp8 KV).** Original 64K validation: (a) `--nll-long 65536` flat buckets; (b) cross-engine vs llama-perplexity +2.0% at [32K,64K) (smaller than the short-context delta, so no length-dependent divergence); (c) needle 3/3 @55K. Extended after P2 with a 783K-token corpus (War and Peace, tokenized with the model's own vocab; `--nll-long` buckets now reach 320K+): (d) fp16-vs-fp8 NLL A/B at 163840, bucket deltas within +-0.06% at ALL depths (fp8 cost does not grow with position); (e) fp8 `--nll-long 370000` single pass: buckets flat 7.2-7.6 to 256K, then a graceful +3% drift beyond the native 262K (7.89 at 256-320K, 7.69 at 320K+) -- no blowup even in RoPE-extrapolation territory; (f) needle retrieval **6/6 on a 361.5K-token haystack** (depths 35K/124K/213K/248K within native + 276K/337K BEYOND native, all exact, think traces naming surrounding chapters); (g) `--kvstats 131072`: K amax 21.7 / V amax 128.4, zero E4M3 saturation at depth. Decode at 361K depth: 19.1 t/s (fp8, spec). Caveat: each distinct long prompt is a full cold prefill (~22 min @361K) -- the GDN recurrent snapshot makes the prefix cache all-or-nothing, so mid-document divergence cannot reuse state [mitigated same-session by P9's checkpoint ring -- restore from nearest checkpoint <= divergence; cross-session pool still parked]. Risk 3 is covered to 64K by (a)+(b).
6. **fp-precision paths break the bitwise gate.** Batched prefill is currently bit-identical to serial because dp4a's int32 block sums are order-independent and the per-group fp scale-and-add matches serial order. RESOLVED for prefill: the int8 mma.sync path keeps int32 accumulation, so the bitwise gate survives tensor-core prefill (P1). **RESOLVED for fp8 KV (P2, 2026-07-02):** the tolerance-gate machinery now exists and passed -- logit A/B vs the fp16 path (cosine 0.9995, top-1 exact, KL 3.4e-5 @512-tok prompt), corpus PPL delta -0.05%, needle 3/3 -- and fp8 shipped opt-in, later becoming the sm_89+ serving default
(the CLI's fp16 default stays bitwise-canonical; since v0.3.0 sm_86
serving defaults to turbo3). The same gate recipe applies to any future fp16/fp8 MMA decode path. **AMENDED for the g64 activation regroup (2026-07-04, policy sign-off):** batched-prefill activations now default to per-64 quantization (`Q27_PF_XG`, matching the Q4 weight group so two K=32 mmas chain in int32 before one fp dequant step). Per-64 amax changes the int8 values vs the decode path's per-32, so serial-vs-batched identity no longer holds BY DESIGN on the default path. Replacement gates: test_kernels g64-vs-exact (same quantized inputs through the dp4a exact path, rounding-noise bound), corpus PPL delta, canonical md5 (the canonical CLI run prefills serially and stays bitwise), scored-task spot-check. `Q27_PF_XG=32` restores the exact path and the `--pf` identity gate enforces it there.

## P10-A status (multi-slot program)

A0 PASSED, A1 SHIPPED (R1 multi-slot + R1b round interleaving;
whole-generation queue waits gone; analysis in docs/P10-decision.md).
A2 (fused cross-sequence verify) is SUPERSEDED in mechanism by the
07-14..16 continuous-batching conductor -- fused verify across active
slots, default-on (BUILDLOG 2026-07-14..16; docs/multislot-throughput.md
carries the current 1.41x table). The conversations-outnumber-slots
trigger it waited on now just means raising `--slots` -- with the
07-17 caveat that aggregate saturates ~250 t/s at 2 lanes; slots past
2 divide the same ceiling among more users.

## Parked levers (measured, receipts in BUILDLOG)

- Fixed depth-5 (+2-4% @2K for +12-14% round cost) and ungated burst
  depth (BUILDLOG 2026-07-04 "burst-depth measured DEAD").
- Chunked-WY delta scan (`Q27_DS_MODE=wy`, default OFF).
- Cross-session checkpoint pool (P9 covers same-session).
- Importance-weighted scales, AWQ-style -- only path left on the +3.05%
  PPL gap (risk 2 above); scored task trials say the gap doesn't bite on
  agentic coding.
- P11 split-path (`Q27_TOOL_SPLIT`): unexplained crash under accumulated
  multi-request state -- flake hunt required before any split/adaptive
  path ships; keep OFF under `--slots`.
- docs-class promote churn (~1%): known shave is a demote-count
  promote-escalator; not built (YAGNI at 1%, worst flavor only).
