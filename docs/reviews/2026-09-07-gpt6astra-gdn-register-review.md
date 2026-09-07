# gpt-6-astra review: GDN register-resident state conversion (2026-09-07)

External review (codex, gpt-6-astra, xhigh, read-only) of the spec3.cu change
converting k_gdn_delta_all / k_gdn_delta_chunk3 from 64KB smem state to
float sreg[32] per thread. Verdict + how each concern was resolved:

VERDICT: sound. "The indexing and state chaining match, and the retained
barriers appear sufficient for all shared scratch." Removing the post-load
__syncthreads is correct (each thread reads only its own sreg; no cross-thread
dependency on the load).

Concerns raised, and resolution:
1. `#pragma unroll 8` could keep sreg in local memory (spill) rather than
   registers. RESOLVED: ptxas -v reports 0 spill stores / 0 spill loads for
   both kernels -- sreg is register-resident. (k_delta_step uses full unroll;
   ours matches the original's unroll-8, which does not reassociate the
   single-accumulator pred/acc chains, so it stays bitwise.)
2. FMA contraction (fmad=true default) means "fp32 reg == fp32 mem" alone is
   insufficient; the operation sequence must match. RESOLVED: the sequence is
   identical, and the bitwise gates below confirm it empirically.
3. gdn_fuse_eq compares the two MODIFIED kernels for the speculative lanes
   (register-vs-register), so a common speculative-lane regression could pass;
   only committed state + lane-0 have the independent delta_step reference.
   RESOLVED: ran ninv_test, whose CHUNK+FOLD legs compare the chunk/all
   speculative outputs against a SERIAL delta_step chain (independent,
   known-good) -- "CHUNK+FOLD BITWISE: ALL PASS".

Validation summary (all green):
- build/gdn_fuse_eq: BITWISE IDENTICAL at widths 2,3,5,8,12,16 (lane-0/committed
  vs known-good k_delta_step).
- build/ninv_test: CHUNK+FOLD BITWISE ALL PASS (speculative lanes vs serial
  delta_step, the independent reference codex asked for).
- ptxas: 0 spill bytes on both kernels.
- serving wsum b743d26b1f0562a9 (canonical clean; weights untouched).

Measured: dflash2 width-8 verify 16.77 -> 16.50 ms/round; same kernel is in the
MTP ladder verify so every decode path benefits.

Next lever (astra, not yet done): column-tile split 48 -> 192 CTAs to fill more
SMs (the grid is only 48 blocks). A block/thread remapping -- do it behind the
same gdn_fuse_eq + ninv_test gates.
