# GPU micro-probes (prefill plan phase 2, 2026-09-08)

The three .cu files here are standalone:
`nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o build/<name> tools/probes/<name>.cu`.
(The margin_predictor_*.py scripts in this directory are the older
acceptance-gate analysis tools, unrelated.) All numbers below: RTX 5090,
production q27-38 idle on the same GPU.

- `imma_chain.cu` -- m16n8k32 s8 IMMA pipe ceiling via DEPENDENT accumulator
  chains (immune to ptxas hoisting): **1020 TOPS**, 10.9 ns per IMMA per warp at
  8 warps/SM, saturated even with one chain per warp.
- `imma_fold.cu` -- register-only IMMA pair + the W4A8 per-group fold mix:
  no fold 968-973, exact 4-op fold (IADD, FFMA, FMUL, FFMA) 817-827, 3-op
  cvt fold (I2F, FMUL, FFMA) 889-896. I2F is NOT slow on sm_120. Inputs are
  perturbed per iteration: PTX has no `volatile`, ptxas hoists loop-invariant
  mma out of loops and a register-only "IMMA-only" loop then reports 2-4x the
  pipe peak (the trap the first version of this probe fell into).
- `stage_bw.cu` -- the spike's cp.async stage pipeline with no compute:
  **6.8 TB/s** (74 us for attn_out T=1024 vs 201 us for the full kernel) and a
  plain 16-B-load L2 probe (8.0-8.3 TB/s).
