# DFlash2 verify width > 8: the matched-family identity gate (2026-09-08)

Item 1 of the 2026-09-08 (p) agenda (docs/reviews/2026-09-08-gpt6astra-what-
next.md). Result in BUILDLOG 2026-09-08 (q).

The 09-06 (d) entry recorded a "width-8 wall": CLI `--dflash2 --k K` was
byte-identical to the ladder for K <= 7 and diverged at one fixed token for
every K >= 8, and attributed it to a latent engine bug in the eager
verify/GDN/fold at width > 8. The reviewer's counter-hypothesis: `gemm_min`
defaults to 9, so `mm5` switches the big projections from the GEMV family
(`gemv_q4_n<N>`) to the MMA family (`k_vgemm`) at exactly width 9 -- a
same-token divergence for every K >= 8 is what a dispatch switch looks like.

## Scripts

- `width_gate.sh <outdir> smoke|full [n_gen]` -- CLI sweep. Per prompt
  (bench/dflash2/toks): a plain greedy run, the ladder (`--spec`), then
  `--dflash2 <full Q8 pack> --k K` for K in {1,6,7,8,9,10,11} in two arms,
  each with the verify graph and eager (`Q27_D2_NOGRAPH=1`):
  - `gemv`: `Q27_GEMM_MIN=99` -- every width on the GEMV family;
  - `dflt`: gemm_min 9 -- widths >= 9 on the MMA family.
  Both arms pass `--spec` alongside `--dflash2`: `Q27_GEMM_MIN` is parsed
  in `build_spec_graphs`, which the CLI calls only with `--spec`. Stops
  production (q27-38) and relaunches it on exit.
- `width_gate_cmp.py <outdir> [--matrix]` -- first divergence of every run
  vs the ladder and vs plain, tok/round and t/s, optional pairwise matrix.
- `width_serve.sh <outdir> [K list]` -- serving side, production config
  (Q8 serving pack, MMA verify at width K+1 by default, sampled walk, fold
  overlap), one boot per K (default order 7 10 7 10 8 9 11): seeded streams
  pass A + pass B, decode at 12.5K and 50K depth. Saves each boot's `[req]`
  and `[d2timing]` lines.
- `width_serve_cmp.py <outdir>` -- per boot and prompt class: tok/round,
  decode t/s, mean dec_ms; the cumulative `[d2timing]` breakdown; stream
  shas (compare A vs B within a boot only -- across K the sampled walk
  realises different tokens under different proposals by design).
- `width_repeat.sh <outdir>` -- repeats of the one MMA-arm outlier.

## Traps

- The reference is the LADDER stream, not plain greedy: the plain decode
  graph uses `k_gemv_q4`, which is not `k_gemv_q4_n<1>` (engine.cuh:1709),
  so plain differs from every width-N verify at some token (prose 32,
  code-write 52, code-edit 127; echo happens to agree). The 09-06 gate
  compared against `--spec` too.
- K=1 fails in the CLI: the drafter's own GEMV has no single-row kernel
  (`gemv_q4_n: bad nbatch 1`). Unrelated to verify width.
- Round counts ARE reproducible for a fixed configuration and history (graph
  vs eager: identical streams and identical round counts on every GEMV-arm
  run; serving boots repeat every hash and round count). The (o)-era
  "not reproducible" note was ring-history drift between passes, not
  drafter nondeterminism.
- The 5090's ~1%-per-load weight corruption (memory: pageable DMA path,
  rate scales with bytes) applies to every CLI run here (13.5 GB weights +
  7 GB pack per process). The scripts print `wsum` per run; discard any run
  whose digest is not the tier's modal value (`b743d26b1f0562a9` for
  qwen38-27b-mtp.q27). The first sweep ran without the print and had one
  sporadic outlier (code-write, K=10, graph) that 4 repeats did not
  reproduce.
