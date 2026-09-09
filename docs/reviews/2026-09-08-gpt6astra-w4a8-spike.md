# gpt-6-astra review: W4A8 prefill GEMM spike (2026-09-08)

Reviewer: gpt-6-astra via codex exec (read-only sandbox, xhigh), static review of tools/gemm_w4a8_spike.cu, tools/probes/, BUILDLOG 2026-09-08 (h) and the plan's phase 2 against src/prefill.cu. No builds or runs on the reviewer's side. Dispositions in BUILDLOG 2026-09-08 (j); the porting gates (section C) are folded into the plan's phase 2 task 3.

## Report (reviewer's words)

The spike demonstrates a useful, credible **1.30×/1.41× speedup**, but it has not met the 1.6× bar or established an unconditional bitwise contract. The integer arithmetic and permutation are sound under the current shape contract; the main numerical exception is `FOLD==2`, and the main integration risks are dispatch, activation-buffer ownership, and untested tails. The evidence supports reducing staging overhead, but does not prove complete fill/math serialization or guarantee that TMA alone fixes it. I recommend another bounded spike before porting. This review was static: no builds, kernels, or tests were run.

**A. Bitwise equivalence**

1. **[P1] `FOLD==2` is conditionally exact; the author’s subnormal analysis is incomplete.**

   **Evidence:** [gemm_w4a8_spike.cu:134](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:134) converts `16d` directly, while [line 319](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:319) scales `wsc` by `1/16`. The reference first rounds `wsc*xs`, then folds `d`: [prefill.cu:479](/mnt/ai/projects/q27/src/prefill.cu:479).

   For finite fp16 `wsc`, converting to fp32 and dividing by 16 is exact—even the smallest nonzero half becomes the normal fp32 value `2^-28`. The potential error is the **subsequent multiplication**:

   ```
   reference: p = RN(wsc * xs);        acc = FMA(p, d, acc)
   cvt16:     q = RN((wsc/16) * xs);   acc = FMA(q, 16d, acc)
   ```

   Away from underflow and overflow, power-of-two scaling commutes with rounding, so these agree. However, `q` can lose bits while `p` is still normal. The relevant boundary is approximately **16·FLT_MIN = 2^-122 ≈ 1.88e-37**, not simply “subnormal original products” or `1.4e-37`.

   An analytical counterexample, with round-to-nearest and gradual underflow: let `wsc=2^-24`, `xs=(1+2^-23)·2^-100`, `d=1`, and `acc=0`. The reference returns `2^-124 + 2^-147`; `cvt16` returns `2^-124`. Overflow also differs: the reference scale product can overflow while the scaled product remains finite, changing cancellation and `d=0` behavior.

   **Recommendation:** Keep the magic conversion for the unconditional path, or explicitly range-gate `cvt16` with an exact fallback. An exact **three-operation fold** exists if unpacking makes MMA produce **`d` directly**: integer-to-float conversion, rounded scale multiplication, fused accumulation. That moves additional work into unpacking. With the existing `16d` representation, `d16 >> 4; cvt; mul; fma` is an exact **four-operation** alternative; moving `1/16` onto a scale does not establish an unconditional three-operation identity.

2. **[P1] Identical source expressions establish a compiler-dependent contract, not a permanent bitwise guarantee.**

   **Evidence:** The spike’s fold is ordinary C++ at [gemm_w4a8_spike.cu:344](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:344), matching [prefill.cu:479](/mnt/ai/projects/q27/src/prefill.cu:479) and [prefill.cu:651](/mnt/ai/projects/q27/src/prefill.cu:651). `lag2` explicitly uses `mul.rn` and `fma.rn` in its main body at [line 743](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:743), but its final drain uses the ordinary-expression helper at [line 669](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:669).

   The required recurrence is `acc = RN(RN(wsc*xs)*d + acc)`, with the outer operation fused. Separate multiplication/addition or reassociation changes rounding. A passing gate supports the tested compiler configuration; it cannot cover later changes to contraction, FTZ, compilation units, or toolchain. NVIDIA documents contraction as controlled by `--fmad`. [NVCC documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#fmad-true-false-fmad)

   **Recommendation:** Record the supported build flags and inspect the eventual production instruction sequence. Encode the intended rounding explicitly in the new fold, including lag drains, and compare against both incumbent kernels. Add cancellation, halfway-rounding, tiny-product, overflow, and signed-zero cases. Treat NaN payload identity as a separate contract rather than assuming it follows from ordinary finite-data equivalence.

3. **[P2] The high-nibble bound and magic conversion are sound under the stated integer contract.**

   **Evidence:** Packing is even-low/odd-high at [FORMAT.md:33](/mnt/ai/projects/q27/docs/FORMAT.md:33); unpacking is at [gemm_w4a8_spike.cu:123](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:123); the quantizer clamps to `[-127,127]` at [kernels.cu:154](/mnt/ai/projects/q27/src/kernels.cu:154). Each group starts with a zero accumulator and chains exactly two MMAs at [spike line 335](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:335).

   Consequently,

   ```
   |d| ≤ 64·8·127 = 65,024
   |16d| ≤ 1,040,384 < 2^22.
   ```

   Even allowing activation byte `-128` gives only `|16d|≤2^20`. Neither the final dot nor an intermediate integer sum approaches int32 overflow.

   Within this bound, adding `16d` to the bits `0x4B400000` produces the float **12,582,912 + 16d**, in a binade whose spacing is one. Multiplication by `1/16` and subtraction of `786,432` yield exactly `d`. The conversion itself does not depend on FMA contraction to avoid intermediate rounding.

   **Recommendation:** Preserve the fresh accumulator per 64-element scale group. Add deterministic endpoint tests for all nibble values, both activation signs, maximum dots, and exact cancellation. These validate the implementation of an otherwise sound proof.

4. **[P2] The K permutation and delayed folds preserve arithmetic; padding must remain confined to invalid outputs.**

   **Evidence:** [gemm_w4a8_spike.cu:154](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:154) implements the activation permutation; [line 302](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:302) constructs matching weight fragments. The lagged fold retains its scales and drains the final queued result at [line 548](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:548). Input predicates and output guards are at [line 206](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:206) and [line 365](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:365).

   The permutation is `k → floor(k/2)+16·(k mod 2)` **within each 32-element block**. Applying it to both operands preserves the integer dot and never crosses a scale-group boundary. `lag`/`lag2` delay updates without changing each output’s group order, subject to finding 2.

   Token padding becomes actual integer zero. Packed weight padding is different: zero bytes unpack to **−128**, whereas the incumbent pads with `0x88`, which unpacks to zero ([prefill.cu:316](/mnt/ai/projects/q27/src/prefill.cu:316)). This is harmless for discarded row outputs: MMA does not mix different output rows. It must not be mistaken for generally neutral weight padding.

   **Recommendation:** Add one-hot K tests at every position, distinct adjacent-group scales, and mixed row/token tails. Preserve uniform participation in MMA and synchronization; do not return individual tail lanes early.

**B. Performance diagnosis**

5. **[P2] Staging is a major cost, but “fill and math do not overlap” overstates the evidence.**

   **Evidence:** The timings and stalls are recorded at [BUILDLOG.md:15738](/mnt/ai/projects/q27/docs/BUILDLOG.md:15738). The actual pipeline issues future copies before current computation at [gemm_w4a8_spike.cu:274](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:274).

   Async transfer completion can already overlap math; issuing the copy instructions, address calculations, and predicates still consumes consumer-warp issue slots. The evidence fits **insufficient overlap and substantial staging issue cost**, without proving complete serialization.

   Using the reported numbers, 609 TOPS corresponds to roughly **141 µs** for the 201-µs case. Adding the standalone 74-µs fill gives **215 µs**, not 201 µs. That difference admits some overlap or measurement/configuration differences.

   More importantly, [stage_bw.cu:19](/mnt/ai/projects/q27/tools/probes/stage_bw.cu:19) copies only W/X, without scales or the production swizzle, despite its opening description. Its resource use differs too. The no-fill ablation changes synchronization and staging state; only `STAGES-1` slots are initially filled ([spike line 253](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:253)), so consuming every ring slot without further fills is not a controlled, initialized substitute.

   **Recommendation:** Repeat later with an exact staging-only clone and initialized no-refill slots, preserving launch geometry and documenting registers/residency. Separately remove scale copies, copy issuance, and barriers. Keep the fold in the diagnosis: 609 versus 807 TOPS shows that it remains material after staging is removed.

6. **[P2] TMA is an appropriate next experiment on sm_120a; transposed scales need not precede the first useful test.**

   **Evidence:** The proposal is at [BUILDLOG.md:15759](/mnt/ai/projects/q27/docs/BUILDLOG.md:15759); current scale copies are small and strided at [gemm_w4a8_spike.cu:218](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:218).

   TMA directly addresses per-lane copy issuance. Use ordinary single-CTA `cp.async.bulk.tensor` with transaction-counted mbarriers, retaining `ldmatrix` and `mma.sync`. Do not assume a datacenter Blackwell `tcgen05` or cluster-multicast design transfers unchanged. [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)

   **Recommendation:** First replace **W and X copies only**, retaining existing scale loads. Try an elected issuing thread within the existing consumer CTA before adding producer warps. This isolates TMA’s benefit without changing every layout.

   Then compare transposed sidecars against ordinary scale loads or aligned scale overfetch. For TMA sidecars, pad the physical weight-scale row count and activation-scale token stride: arbitrary `rows*2` or `T*4` need not satisfy the required 16-byte stride alignment. Inner box sizes must be multiples of 16 bytes, not merely at least 16 bytes. Validate the hardware swizzle against the fragment layout. [Tensor-map API requirements](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)

7. **[P2] The producer-warp regression does not disprove specialization; its copy workload and register layout are unfavorable.**

   **Evidence:** The producer loops are at [gemm_w4a8_spike.cu:882](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:882), and the extra warp changes the launch bound at [line 841](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:841). The 168-register result is recorded at [BUILDLOG.md:15753](/mnt/ai/projects/q27/docs/BUILDLOG.md:15753).

   The checked-in 128²/BK128 producer issues **56 copies per lane per stage**: 16 W, 32 X, four W-scale, four X-scale copies, plus addressing and synchronization. Concentrating that work into one warp creates a plausible producer bottleneck. Adding a ninth or seventeenth warp also changes register allocation; this is not an otherwise identical overlap experiment. The base and specialized kernels additionally use different shared-memory scale layouts.

   **Recommendation:** Compare matched scale layouts and test two/four copy producers on a smaller tile before ruling out `cp.async` specialization. For TMA, evaluate a four-warp producer group with explicit register redistribution. `setmaxnreg` is supported on **sm_120a**, but its participation rules are warpgroup-wide; applying producer-only redistribution to one warp in a mixed warpgroup is invalid. [NVIDIA register-allocation instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg)

   The reported 255-register and 128-register configurations leave little resource slack. Measure spills and residency for each structural change rather than carrying over “occupancy did not matter.”

8. **[P2] Several cheap structural comparisons remain worthwhile before a full TMA rewrite.**

   **Evidence:** Tile dimensions and warp ownership are defined at [gemm_w4a8_spike.cu:179](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:179); rasterization is fixed at [line 195](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:195). Historical experiments are summarized at [BUILDLOG.md:15746](/mnt/ai/projects/q27/docs/BUILDLOG.md:15746), but are not all represented in the current table.

   **Recommendation:** Use the following targeted matrix. Dimensions below are **weight rows × tokens × K**.

   | Experiment | Concrete comparison and purpose |
   |---|---|
   | Rectangular tiles | Compare `256×128×128` against `128×256×128`, plus `128×64×128` and `64×128×128`. The first balances packed-W and int8-X bytes; smaller tiles improve grid size and register headroom. |
   | Intermediate tile sizes | Try `128×192×128` with `4×3` warps or `192×128×128` with `6×2`, checking row tails and wave counts. |
   | Warp layout | Compare `4×2` against `2×4` on 128²: this changes operand reuse and fragment live ranges without changing output area. Consider `8×2` consumers for `256×128`. |
   | Rasterization | Compare token-fast, row-fast, and grouped traversal over 2–8 neighboring row tiles. Measure both warm-L2 and streaming behavior. Token-fast already has historical justification at [prefill.cu:272](/mnt/ai/projects/q27/src/prefill.cu:272). |
   | BK | Compare BK64 with more stages against BK128; test BK256 on smaller tiles. Keep the FP fold per 64 elements regardless of BK. |
   | Operand-role swap | Put activations on MMA’s 16-row A dimension and weights on its 8-column B dimension, as the plan’s B-side unpack suggests ([plan:267](/mnt/ai/projects/q27/docs/plans/2026-09-08-prefill-attack.md:267)). This may improve stores to token-major `y` and change unpack reuse; it requires new fragment/permutation mapping, not merely swapped pointers. |
   | Split-K | For underfilled small-T/small-row grids, compare preserved incumbent split-K against smaller unsplit tiles. Treat new split-K numerics separately. |

   Shared memory constrains this matrix: the current formula gives **102 KiB** for either `128×128×256` with two stages or `256×128×128` with three stages, before barriers. Both exceed sm_120’s 99-KiB per-block limit. [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#occupancy)

9. **[P3] Preserve the experiment matrix and distinguish diagnostic throughput from end-to-end cost.**

   **Evidence:** [gemm_w4a8_spike.cu:1036](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1036) currently registers two base kernels and six specialized kernels, all BK128; lag, lag2, and ablation macros remain unregistered. Quantization and permutation happen before timing at [line 1097](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1097). The final success message is unconditional even with `--notest` at [line 1162](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1162).

   The register-only probes appropriately use dependency chains or changing operands ([imma_chain.cu:21](/mnt/ai/projects/q27/tools/probes/imma_chain.cu:21), [imma_fold.cu:36](/mnt/ai/projects/q27/tools/probes/imma_fold.cu:36)). Their ceilings do not include realistic operand loading or live ranges. Also, every nominal L2-size case in [stage_bw.cu:60](/mnt/ai/projects/q27/tools/probes/stage_bw.cu:60) clamps to the same 8-MiB X allocation.

   **Recommendation:** Retain every measured configuration with its exact flags and resource report. Report zero selected tests as an error and `--notest` as untested. Include quantizer/sidecar production in the eventual comparison, alongside kernel-only timing.

**C. Gates before porting**

10. **[P1] Make the supported shape contract explicit; two launchers silently truncate unsupported K.**

   **Evidence:** Every kernel uses `cols/BK`. Base and lag launchers reject remainders at [gemm_w4a8_spike.cu:383](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:383) and [line 807](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:807); `launch_lag2` and `launch_ws` lack that check ([line 790](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:790), [line 1007](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1007)).

   **Recommendation:** Add gates for:

   - `cols<BK`, `cols==BK`, and `cols%BK!=0`. Fall back where the incumbent supports the shape; it currently requires multiples of 128 ([prefill.cu:864](/mnt/ai/projects/q27/src/prefill.cu:864)).
   - BK64 with odd `cols/64`: its aligned-pair scale load assumes an even group count ([spike:221](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:221)). Otherwise row bases can be misaligned and the final pair can overread.
   - Rows `1,7,8,15,16,17` and immediately around BN boundaries.
   - Every `T=1…17`, plus token-tile and dispatch boundaries such as `31/32/33`, `63/64/65`, `95/96/97`, and `127/128/129`.
   - Defined zero-size behavior and pipeline lengths shorter than the stage count.

   For future K-tail support, zero invalid activation elements and skip nonexistent scale groups. Appending extra zero-valued FP folds can itself change signed-zero results.

11. **[P1] Preserve the existing split-K decision, not just the old split-K implementation.**

   **Evidence:** The harness always passes null scratch at [gemm_w4a8_spike.cu:1115](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1115). Production dispatch considers scratch capacity at [prefill.cu:798](/mnt/ai/projects/q27/src/prefill.cu:798), and splitting changes FP grouping by design ([prefill.cu:241](/mnt/ai/projects/q27/src/prefill.cu:241)).

   **Recommendation:** Before taking `w4a8v2`, determine whether the original dispatcher would have selected split-K. Preserve that route, including forced split counts, as the plan requests.

   Test split off/auto/forced, null and undersized scratch, uneven partitions, and counts near the number of K stages. If a later spike implements split-K, compare partials and the final reduction against the incumbent using identical partition boundaries. Unsplit bitwise equality does not validate split output or justify changing production partition selection.

12. **[P1] `nat64p` must be an additional output; replacing `nat64` would corrupt Q8 and fallbacks.**

   **Evidence:** `XQuant` currently exposes natural g64 bytes and scales at [kernels.cuh:26](/mnt/ai/projects/q27/src/kernels.cuh:26). Both Q4 and Q8 use the common MMA dispatcher ([prefill.cu:894](/mnt/ai/projects/q27/src/prefill.cu:894), [prefill.cu:925](/mnt/ai/projects/q27/src/prefill.cu:925)). The plan explicitly requires a second buffer at [plan:305](/mnt/ai/projects/q27/docs/plans/2026-09-08-prefill-attack.md:305).

   **Recommendation:** Generate `nat64p` from the **same already-rounded `q0/q1`** as `nat64`, preserving `s64` exactly. Test both outputs bytewise against the existing quantizer plus standalone permutation.

   Route the new kernel only for Q4_G64, g64 activation mode, supported device/shapes, and valid new buffers. Exercise Q8 immediately after Q4 using the same `XQuant`; also exercise `Q27_PF_XG=32`, `Q27_PREFILL=dp4a`, missing g64 buffers, and supported in-process route changes. Existing `eo` is a different per-eight-element, g32 layout and cannot substitute for `nat64p`.

13. **[P1] Add independent numerical and memory-safety gates; current data and allocations hide important failures.**

   **Evidence:** Synthetic scales stay near `0.002…0.032`, activations near `−2…2` ([gemm_w4a8_spike.cu:1083](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1083)). All four row dimensions are multiples of 256. Inputs and outputs are allocated for `Tmax=4096`, while comparison covers only the active prefix ([line 1105](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1105)).

   **Recommendation:** Require direct comparisons against **both forced MR64 and ntx references on identical inputs**, with split disabled and route flags recorded. Add real projection weights and captured activations, plus structured integer patterns and the floating-point adversaries in A.

   Use exact-sized allocations or checked red zones, distinct row/group sentinels, and repeated launches with changing T. Later run memory, race, and synchronization sanitizers over every shipped pipeline variant. An out-of-range access into the currently oversized allocation may neither fault nor affect the compared outputs.

14. **[P1] New buffers and descriptors must follow both standalone-engine and shared-arena lifetimes.**

   **Evidence:** Engines can either alias `PrefillArena::xqT` or allocate their own ([engine.cuh:1216](/mnt/ai/projects/q27/src/engine.cuh:1216)). Arena handoff drains the previous stream at [prefill_arena.h:129](/mnt/ai/projects/q27/src/prefill_arena.h:129). Allocation is intentionally performed before serving because other threads may be capturing graphs ([prefill_arena.h:59](/mnt/ai/projects/q27/src/prefill_arena.h:59)).

   **Recommendation:** Allocate `nat64p`, any transposed activation scales, and persistent descriptors during the existing initialization paths. Include ownership, capacity, accounting, and cleanup. Shared buffers must remain inside the arena’s per-chunk claim discipline; overlapping independent streams need distinct writable storage.

   Gate graph capture/replay with stable pointers, first use, engine destruction/recreation, and alternating engines/streams with different T/K. Prefill is presently uncaptured ([prefill.cu:859](/mnt/ai/projects/q27/src/prefill.cu:859)), but that does not make lazy allocation safe during another engine’s global capture. Avoid process-global mutable descriptors and unsynchronized/per-device-incomplete initialization flags such as the spike’s `static bool attr`.

15. **[P2] Port acceptance needs route-aware integration and performance evidence.**

   **Evidence:** The plan specifies default-off routing and real-weight comparison at [plan:273](/mnt/ai/projects/q27/docs/plans/2026-09-08-prefill-attack.md:273), and explicitly warns that the default g64 `--pf` mismatch does not enforce correctness. End-to-end targets are at [plan:283](/mnt/ai/projects/q27/docs/plans/2026-09-08-prefill-attack.md:283).

   **Recommendation:** Require unchanged legacy `--pf` identity, unchanged decode canonicals, direct g64 old/new comparisons, mixed Q4/Q8 execution, and unchanged preserved split-K behavior. Measure short suffixes as well as 1024-token chunks, including the extra quantizer stores and any scale-layout work.

   If any shipped path intentionally changes numerics, apply the plan’s deep-logit/quality gates and separate prefix-cache roots. Keep `w4a8v2` default-off until those gates pass and either the 1.6× bar is met or the smaller measured gain is explicitly accepted.
The spike demonstrates a useful, credible **1.30×/1.41× speedup**, but it has not met the 1.6× bar or established an unconditional bitwise contract. The integer arithmetic and permutation are sound under the current shape contract; the main numerical exception is `FOLD==2`, and the main integration risks are dispatch, activation-buffer ownership, and untested tails. The evidence supports reducing staging overhead, but does not prove complete fill/math serialization or guarantee that TMA alone fixes it. I recommend another bounded spike before porting. This review was static: no builds, kernels, or tests were run.

**A. Bitwise equivalence**

1. **[P1] `FOLD==2` is conditionally exact; the author’s subnormal analysis is incomplete.**

   **Evidence:** [gemm_w4a8_spike.cu:134](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:134) converts `16d` directly, while [line 319](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:319) scales `wsc` by `1/16`. The reference first rounds `wsc*xs`, then folds `d`: [prefill.cu:479](/mnt/ai/projects/q27/src/prefill.cu:479).

   For finite fp16 `wsc`, converting to fp32 and dividing by 16 is exact—even the smallest nonzero half becomes the normal fp32 value `2^-28`. The potential error is the **subsequent multiplication**:

   ```
   reference: p = RN(wsc * xs);        acc = FMA(p, d, acc)
   cvt16:     q = RN((wsc/16) * xs);   acc = FMA(q, 16d, acc)
   ```

   Away from underflow and overflow, power-of-two scaling commutes with rounding, so these agree. However, `q` can lose bits while `p` is still normal. The relevant boundary is approximately **16·FLT_MIN = 2^-122 ≈ 1.88e-37**, not simply “subnormal original products” or `1.4e-37`.

   An analytical counterexample, with round-to-nearest and gradual underflow: let `wsc=2^-24`, `xs=(1+2^-23)·2^-100`, `d=1`, and `acc=0`. The reference returns `2^-124 + 2^-147`; `cvt16` returns `2^-124`. Overflow also differs: the reference scale product can overflow while the scaled product remains finite, changing cancellation and `d=0` behavior.

   **Recommendation:** Keep the magic conversion for the unconditional path, or explicitly range-gate `cvt16` with an exact fallback. An exact **three-operation fold** exists if unpacking makes MMA produce **`d` directly**: integer-to-float conversion, rounded scale multiplication, fused accumulation. That moves additional work into unpacking. With the existing `16d` representation, `d16 >> 4; cvt; mul; fma` is an exact **four-operation** alternative; moving `1/16` onto a scale does not establish an unconditional three-operation identity.

2. **[P1] Identical source expressions establish a compiler-dependent contract, not a permanent bitwise guarantee.**

   **Evidence:** The spike’s fold is ordinary C++ at [gemm_w4a8_spike.cu:344](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:344), matching [prefill.cu:479](/mnt/ai/projects/q27/src/prefill.cu:479) and [prefill.cu:651](/mnt/ai/projects/q27/src/prefill.cu:651). `lag2` explicitly uses `mul.rn` and `fma.rn` in its main body at [line 743](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:743), but its final drain uses the ordinary-expression helper at [line 669](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:669).

   The required recurrence is `acc = RN(RN(wsc*xs)*d + acc)`, with the outer operation fused. Separate multiplication/addition or reassociation changes rounding. A passing gate supports the tested compiler configuration; it cannot cover later changes to contraction, FTZ, compilation units, or toolchain. NVIDIA documents contraction as controlled by `--fmad`. [NVCC documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#fmad-true-false-fmad)

   **Recommendation:** Record the supported build flags and inspect the eventual production instruction sequence. Encode the intended rounding explicitly in the new fold, including lag drains, and compare against both incumbent kernels. Add cancellation, halfway-rounding, tiny-product, overflow, and signed-zero cases. Treat NaN payload identity as a separate contract rather than assuming it follows from ordinary finite-data equivalence.

3. **[P2] The high-nibble bound and magic conversion are sound under the stated integer contract.**

   **Evidence:** Packing is even-low/odd-high at [FORMAT.md:33](/mnt/ai/projects/q27/docs/FORMAT.md:33); unpacking is at [gemm_w4a8_spike.cu:123](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:123); the quantizer clamps to `[-127,127]` at [kernels.cu:154](/mnt/ai/projects/q27/src/kernels.cu:154). Each group starts with a zero accumulator and chains exactly two MMAs at [spike line 335](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:335).

   Consequently,

   ```
   |d| ≤ 64·8·127 = 65,024
   |16d| ≤ 1,040,384 < 2^22.
   ```

   Even allowing activation byte `-128` gives only `|16d|≤2^20`. Neither the final dot nor an intermediate integer sum approaches int32 overflow.

   Within this bound, adding `16d` to the bits `0x4B400000` produces the float **12,582,912 + 16d**, in a binade whose spacing is one. Multiplication by `1/16` and subtraction of `786,432` yield exactly `d`. The conversion itself does not depend on FMA contraction to avoid intermediate rounding.

   **Recommendation:** Preserve the fresh accumulator per 64-element scale group. Add deterministic endpoint tests for all nibble values, both activation signs, maximum dots, and exact cancellation. These validate the implementation of an otherwise sound proof.

4. **[P2] The K permutation and delayed folds preserve arithmetic; padding must remain confined to invalid outputs.**

   **Evidence:** [gemm_w4a8_spike.cu:154](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:154) implements the activation permutation; [line 302](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:302) constructs matching weight fragments. The lagged fold retains its scales and drains the final queued result at [line 548](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:548). Input predicates and output guards are at [line 206](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:206) and [line 365](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:365).

   The permutation is `k → floor(k/2)+16·(k mod 2)` **within each 32-element block**. Applying it to both operands preserves the integer dot and never crosses a scale-group boundary. `lag`/`lag2` delay updates without changing each output’s group order, subject to finding 2.

   Token padding becomes actual integer zero. Packed weight padding is different: zero bytes unpack to **−128**, whereas the incumbent pads with `0x88`, which unpacks to zero ([prefill.cu:316](/mnt/ai/projects/q27/src/prefill.cu:316)). This is harmless for discarded row outputs: MMA does not mix different output rows. It must not be mistaken for generally neutral weight padding.

   **Recommendation:** Add one-hot K tests at every position, distinct adjacent-group scales, and mixed row/token tails. Preserve uniform participation in MMA and synchronization; do not return individual tail lanes early.

**B. Performance diagnosis**

5. **[P2] Staging is a major cost, but “fill and math do not overlap” overstates the evidence.**

   **Evidence:** The timings and stalls are recorded at [BUILDLOG.md:15738](/mnt/ai/projects/q27/docs/BUILDLOG.md:15738). The actual pipeline issues future copies before current computation at [gemm_w4a8_spike.cu:274](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:274).

   Async transfer completion can already overlap math; issuing the copy instructions, address calculations, and predicates still consumes consumer-warp issue slots. The evidence fits **insufficient overlap and substantial staging issue cost**, without proving complete serialization.

   Using the reported numbers, 609 TOPS corresponds to roughly **141 µs** for the 201-µs case. Adding the standalone 74-µs fill gives **215 µs**, not 201 µs. That difference admits some overlap or measurement/configuration differences.

   More importantly, [stage_bw.cu:19](/mnt/ai/projects/q27/tools/probes/stage_bw.cu:19) copies only W/X, without scales or the production swizzle, despite its opening description. Its resource use differs too. The no-fill ablation changes synchronization and staging state; only `STAGES-1` slots are initially filled ([spike line 253](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:253)), so consuming every ring slot without further fills is not a controlled, initialized substitute.

   **Recommendation:** Repeat later with an exact staging-only clone and initialized no-refill slots, preserving launch geometry and documenting registers/residency. Separately remove scale copies, copy issuance, and barriers. Keep the fold in the diagnosis: 609 versus 807 TOPS shows that it remains material after staging is removed.

6. **[P2] TMA is an appropriate next experiment on sm_120a; transposed scales need not precede the first useful test.**

   **Evidence:** The proposal is at [BUILDLOG.md:15759](/mnt/ai/projects/q27/docs/BUILDLOG.md:15759); current scale copies are small and strided at [gemm_w4a8_spike.cu:218](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:218).

   TMA directly addresses per-lane copy issuance. Use ordinary single-CTA `cp.async.bulk.tensor` with transaction-counted mbarriers, retaining `ldmatrix` and `mma.sync`. Do not assume a datacenter Blackwell `tcgen05` or cluster-multicast design transfers unchanged. [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)

   **Recommendation:** First replace **W and X copies only**, retaining existing scale loads. Try an elected issuing thread within the existing consumer CTA before adding producer warps. This isolates TMA’s benefit without changing every layout.

   Then compare transposed sidecars against ordinary scale loads or aligned scale overfetch. For TMA sidecars, pad the physical weight-scale row count and activation-scale token stride: arbitrary `rows*2` or `T*4` need not satisfy the required 16-byte stride alignment. Inner box sizes must be multiples of 16 bytes, not merely at least 16 bytes. Validate the hardware swizzle against the fragment layout. [Tensor-map API requirements](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)

7. **[P2] The producer-warp regression does not disprove specialization; its copy workload and register layout are unfavorable.**

   **Evidence:** The producer loops are at [gemm_w4a8_spike.cu:882](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:882), and the extra warp changes the launch bound at [line 841](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:841). The 168-register result is recorded at [BUILDLOG.md:15753](/mnt/ai/projects/q27/docs/BUILDLOG.md:15753).

   The checked-in 128²/BK128 producer issues **56 copies per lane per stage**: 16 W, 32 X, four W-scale, four X-scale copies, plus addressing and synchronization. Concentrating that work into one warp creates a plausible producer bottleneck. Adding a ninth or seventeenth warp also changes register allocation; this is not an otherwise identical overlap experiment. The base and specialized kernels additionally use different shared-memory scale layouts.

   **Recommendation:** Compare matched scale layouts and test two/four copy producers on a smaller tile before ruling out `cp.async` specialization. For TMA, evaluate a four-warp producer group with explicit register redistribution. `setmaxnreg` is supported on **sm_120a**, but its participation rules are warpgroup-wide; applying producer-only redistribution to one warp in a mixed warpgroup is invalid. [NVIDIA register-allocation instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg)

   The reported 255-register and 128-register configurations leave little resource slack. Measure spills and residency for each structural change rather than carrying over “occupancy did not matter.”

8. **[P2] Several cheap structural comparisons remain worthwhile before a full TMA rewrite.**

   **Evidence:** Tile dimensions and warp ownership are defined at [gemm_w4a8_spike.cu:179](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:179); rasterization is fixed at [line 195](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:195). Historical experiments are summarized at [BUILDLOG.md:15746](/mnt/ai/projects/q27/docs/BUILDLOG.md:15746), but are not all represented in the current table.

   **Recommendation:** Use the following targeted matrix. Dimensions below are **weight rows × tokens × K**.

   | Experiment | Concrete comparison and purpose |
   |---|---|
   | Rectangular tiles | Compare `256×128×128` against `128×256×128`, plus `128×64×128` and `64×128×128`. The first balances packed-W and int8-X bytes; smaller tiles improve grid size and register headroom. |
   | Intermediate tile sizes | Try `128×192×128` with `4×3` warps or `192×128×128` with `6×2`, checking row tails and wave counts. |
   | Warp layout | Compare `4×2` against `2×4` on 128²: this changes operand reuse and fragment live ranges without changing output area. Consider `8×2` consumers for `256×128`. |
   | Rasterization | Compare token-fast, row-fast, and grouped traversal over 2–8 neighboring row tiles. Measure both warm-L2 and streaming behavior. Token-fast already has historical justification at [prefill.cu:272](/mnt/ai/projects/q27/src/prefill.cu:272). |
   | BK | Compare BK64 with more stages against BK128; test BK256 on smaller tiles. Keep the FP fold per 64 elements regardless of BK. |
   | Operand-role swap | Put activations on MMA’s 16-row A dimension and weights on its 8-column B dimension, as the plan’s B-side unpack suggests ([plan:267](/mnt/ai/projects/q27/docs/plans/2026-09-08-prefill-attack.md:267)). This may improve stores to token-major `y` and change unpack reuse; it requires new fragment/permutation mapping, not merely swapped pointers. |
   | Split-K | For underfilled small-T/small-row grids, compare preserved incumbent split-K against smaller unsplit tiles. Treat new split-K numerics separately. |

   Shared memory constrains this matrix: the current formula gives **102 KiB** for either `128×128×256` with two stages or `256×128×128` with three stages, before barriers. Both exceed sm_120’s 99-KiB per-block limit. [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#occupancy)

9. **[P3] Preserve the experiment matrix and distinguish diagnostic throughput from end-to-end cost.**

   **Evidence:** [gemm_w4a8_spike.cu:1036](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1036) currently registers two base kernels and six specialized kernels, all BK128; lag, lag2, and ablation macros remain unregistered. Quantization and permutation happen before timing at [line 1097](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1097). The final success message is unconditional even with `--notest` at [line 1162](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1162).

   The register-only probes appropriately use dependency chains or changing operands ([imma_chain.cu:21](/mnt/ai/projects/q27/tools/probes/imma_chain.cu:21), [imma_fold.cu:36](/mnt/ai/projects/q27/tools/probes/imma_fold.cu:36)). Their ceilings do not include realistic operand loading or live ranges. Also, every nominal L2-size case in [stage_bw.cu:60](/mnt/ai/projects/q27/tools/probes/stage_bw.cu:60) clamps to the same 8-MiB X allocation.

   **Recommendation:** Retain every measured configuration with its exact flags and resource report. Report zero selected tests as an error and `--notest` as untested. Include quantizer/sidecar production in the eventual comparison, alongside kernel-only timing.

**C. Gates before porting**

10. **[P1] Make the supported shape contract explicit; two launchers silently truncate unsupported K.**

   **Evidence:** Every kernel uses `cols/BK`. Base and lag launchers reject remainders at [gemm_w4a8_spike.cu:383](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:383) and [line 807](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:807); `launch_lag2` and `launch_ws` lack that check ([line 790](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:790), [line 1007](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1007)).

   **Recommendation:** Add gates for:

   - `cols<BK`, `cols==BK`, and `cols%BK!=0`. Fall back where the incumbent supports the shape; it currently requires multiples of 128 ([prefill.cu:864](/mnt/ai/projects/q27/src/prefill.cu:864)).
   - BK64 with odd `cols/64`: its aligned-pair scale load assumes an even group count ([spike:221](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:221)). Otherwise row bases can be misaligned and the final pair can overread.
   - Rows `1,7,8,15,16,17` and immediately around BN boundaries.
   - Every `T=1…17`, plus token-tile and dispatch boundaries such as `31/32/33`, `63/64/65`, `95/96/97`, and `127/128/129`.
   - Defined zero-size behavior and pipeline lengths shorter than the stage count.

   For future K-tail support, zero invalid activation elements and skip nonexistent scale groups. Appending extra zero-valued FP folds can itself change signed-zero results.

11. **[P1] Preserve the existing split-K decision, not just the old split-K implementation.**

   **Evidence:** The harness always passes null scratch at [gemm_w4a8_spike.cu:1115](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1115). Production dispatch considers scratch capacity at [prefill.cu:798](/mnt/ai/projects/q27/src/prefill.cu:798), and splitting changes FP grouping by design ([prefill.cu:241](/mnt/ai/projects/q27/src/prefill.cu:241)).

   **Recommendation:** Before taking `w4a8v2`, determine whether the original dispatcher would have selected split-K. Preserve that route, including forced split counts, as the plan requests.

   Test split off/auto/forced, null and undersized scratch, uneven partitions, and counts near the number of K stages. If a later spike implements split-K, compare partials and the final reduction against the incumbent using identical partition boundaries. Unsplit bitwise equality does not validate split output or justify changing production partition selection.

12. **[P1] `nat64p` must be an additional output; replacing `nat64` would corrupt Q8 and fallbacks.**

   **Evidence:** `XQuant` currently exposes natural g64 bytes and scales at [kernels.cuh:26](/mnt/ai/projects/q27/src/kernels.cuh:26). Both Q4 and Q8 use the common MMA dispatcher ([prefill.cu:894](/mnt/ai/projects/q27/src/prefill.cu:894), [prefill.cu:925](/mnt/ai/projects/q27/src/prefill.cu:925)). The plan explicitly requires a second buffer at [plan:305](/mnt/ai/projects/q27/docs/plans/2026-09-08-prefill-attack.md:305).

   **Recommendation:** Generate `nat64p` from the **same already-rounded `q0/q1`** as `nat64`, preserving `s64` exactly. Test both outputs bytewise against the existing quantizer plus standalone permutation.

   Route the new kernel only for Q4_G64, g64 activation mode, supported device/shapes, and valid new buffers. Exercise Q8 immediately after Q4 using the same `XQuant`; also exercise `Q27_PF_XG=32`, `Q27_PREFILL=dp4a`, missing g64 buffers, and supported in-process route changes. Existing `eo` is a different per-eight-element, g32 layout and cannot substitute for `nat64p`.

13. **[P1] Add independent numerical and memory-safety gates; current data and allocations hide important failures.**

   **Evidence:** Synthetic scales stay near `0.002…0.032`, activations near `−2…2` ([gemm_w4a8_spike.cu:1083](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1083)). All four row dimensions are multiples of 256. Inputs and outputs are allocated for `Tmax=4096`, while comparison covers only the active prefix ([line 1105](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1105)).

   **Recommendation:** Require direct comparisons against **both forced MR64 and ntx references on identical inputs**, with split disabled and route flags recorded. Add real projection weights and captured activations, plus structured integer patterns and the floating-point adversaries in A.

   Use exact-sized allocations or checked red zones, distinct row/group sentinels, and repeated launches with changing T. Later run memory, race, and synchronization sanitizers over every shipped pipeline variant. An out-of-range access into the currently oversized allocation may neither fault nor affect the compared outputs.

14. **[P1] New buffers and descriptors must follow both standalone-engine and shared-arena lifetimes.**

   **Evidence:** Engines can either alias `PrefillArena::xqT` or allocate their own ([engine.cuh:1216](/mnt/ai/projects/q27/src/engine.cuh:1216)). Arena handoff drains the previous stream at [prefill_arena.h:129](/mnt/ai/projects/q27/src/prefill_arena.h:129). Allocation is intentionally performed before serving because other threads may be capturing graphs ([prefill_arena.h:59](/mnt/ai/projects/q27/src/prefill_arena.h:59)).

   **Recommendation:** Allocate `nat64p`, any transposed activation scales, and persistent descriptors during the existing initialization paths. Include ownership, capacity, accounting, and cleanup. Shared buffers must remain inside the arena’s per-chunk claim discipline; overlapping independent streams need distinct writable storage.

   Gate graph capture/replay with stable pointers, first use, engine destruction/recreation, and alternating engines/streams with different T/K. Prefill is presently uncaptured ([prefill.cu:859](/mnt/ai/projects/q27/src/prefill.cu:859)), but that does not make lazy allocation safe during another engine’s global capture. Avoid process-global mutable descriptors and unsynchronized/per-device-incomplete initialization flags such as the spike’s `static bool attr`.

15. **[P2] Port acceptance needs route-aware integration and performance evidence.**

   **Evidence:** The plan specifies default-off routing and real-weight comparison at [plan:273](/mnt/ai/projects/q27/docs/plans/2026-09-08-prefill-attack.md:273), and explicitly warns that the default g64 `--pf` mismatch does not enforce correctness. End-to-end targets are at [plan:283](/mnt/ai/projects/q27/docs/plans/2026-09-08-prefill-attack.md:283).

   **Recommendation:** Require unchanged legacy `--pf` identity, unchanged decode canonicals, direct g64 old/new comparisons, mixed Q4/Q8 execution, and unchanged preserved split-K behavior. Measure short suffixes as well as 1024-token chunks, including the extra quantizer stores and any scale-layout work.

   If any shipped path intentionally changes numerics, apply the plan’s deep-logit/quality gates and separate prefix-cache roots. Keep `w4a8v2` default-off until those gates pass and either the 1.6× bar is met or the smaller measured gain is explicitly accepted.
