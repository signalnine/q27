# gpt-6-astra perf review: DFlash2 decode round (2026-09-06)

External review (codex, gpt-6-astra, xhigh, read-only) of the DFlash2 decode
round, requested to find the ~2ms/round gap to ninfer at matched acceptance
(4.87 tok/round). Verbatim below; savings are the model's engineering
estimates, not measurements.

Key takeaways (my summary):
- Methodology catch: our per-phase timing in Engine::dflash2_round is
  misattributed. The GDN fold runs in post_round (outside the round fn) and
  ingest is submitted outside the event window, so "verify 16.77 + host 0.83"
  double-counts. Fix instrumentation (fold+ingest events, completion-to-
  completion) before trusting the split.
- Biggest levers: (1) adaptive verify width 0.5-2ms (gate on the walk's
  selector-margin, keep drafting K=7, verify W=2..8 via a device SWITCH node);
  (2) GDN state in registers + column-tile split 0.3-0.9ms (helps EVERY decode
  path, not just DFlash2 -- gdn_delta_all is the serving path); (3) drafter
  attention layout 0.15-0.5ms.
- Safe cumulative drafter wins: dedup 25 activation quantizations (q/k/v and
  gate/up share ny0; ingest k/v share s_ct), graph the ingest, collapse the 40
  tap-copy graph nodes to 5 gathers. ~0.5ms combined, byte-identical.

---

**The best opportunities are adaptive verification, register-resident GDN state, and the drafter’s attention kernel.** I would not budget the full 0.83 ms as removable host overhead.

This is a read-only review. Savings below are engineering estimates, not measurements: the NVIDIA driver was unavailable locally. Ranges overlap and should not be added blindly.

First, fix the attribution: [engine.cuh:2848](/mnt/ai/projects/q27/src/engine.cuh:2848) records the first event **after previously queued work**, while [engine.cuh:2866](/mnt/ai/projects/q27/src/engine.cuh:2866) submits ingest outside the measured GPU intervals. `post_round` also submits the fold outside this function at [engine.cuh:4518](/mnt/ai/projects/q27/src/engine.cuh:4518). Consequently, “host” can include outstanding fold/ingest execution and pageable-copy staging. Add separate ingest/fold events and measure completion-to-completion round latency before assigning that residual to synchronization.

1. **Adaptive verify width: potentially 0.5–2 ms/round, conditional on retaining token yield.**

   Compute confidence in [k_d2_walk, dflash2.cu:269](/mnt/ai/projects/q27/src/dflash2.cu:269): thread 0 already scans the 16 **selector-adjusted** scores. Retain the best and second-best score at each position and write their margin. Using only `d_cval` margins would omit the predecessor/codebook contribution that actually determines the proposal.

   Keep drafting K=7, then choose verify W=2…8. Fit prefix survival against those margins using actual acceptance outcomes, and choose width using measured `verify_ms[W]`. The relevant objective is tokens/time:

   \[
   \text{beneficial if expected tokens lost}<\frac{4.87}{19.96}\times\text{milliseconds saved}.
   \]

   Thus, saving **2 ms permits losing fewer than 0.49 tokens/round**; saving 1 ms permits fewer than 0.24. The aggregate 55% acceptance does not establish whether this gate will win.

   Two implementation details matter:

   - **Existing graphs are not directly reusable.** [engine.cuh:2621](/mnt/ai/projects/q27/src/engine.cuh:2621) captures `spec_verify_launches`, whose forward receives no taps. Capture DFlash2 variants using `spec_verify_forward(v, d2_vtaps)` plus `spec_verify_tail(v)`, as [engine.cuh:2794](/mnt/ai/projects/q27/src/engine.cuh:2794) does. Set both member `vw` and view width during capture because `gdn_mix` reads the member. Also ensure all required widths exist; the current capture loop depends on `gate_maxd`.
   - Avoid an extra steady-state host rendezvous by putting the width bodies under a CUDA conditional SWITCH, with the walk setting its handle. This requires constructing conditional body graphs, not inserting existing `cudaGraphExec_t` handles. [NVIDIA documents device-evaluated SWITCH nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

   **Do not initially shrink the draft itself:** its attention is bidirectional across noise rows, so reducing K changes even the retained proposals. Narrowing only verification gives a cleaner experiment.

2. **Keep GDN state in registers and split the independent value columns: approximately 0.3–0.9 ms/round.**

   [spec3.cu:197](/mnt/ai/projects/q27/src/spec3.cu:197) stages the entire 128×128 state in shared memory. Both passes at [spec3.cu:249](/mnt/ai/projects/q27/src/spec3.cu:249) and [spec3.cu:265](/mnt/ai/projects/q27/src/spec3.cu:265) repeatedly load and store it.

   **Each state element has one owning thread throughout every lane.** Cross-thread communication involves `sq`, `sk`, `part`, and `dj`; the state matrix itself need not be shared. Use a fully unrolled `float sreg[32]`, retaining it across all eight steps. The single-step implementation already demonstrates this ownership at [blocks.cu:221](/mnt/ai/projects/q27/src/blocks.cu:221).

   Preserve the current arithmetic order: decay → prediction → four-part reduction → delta → state update → output reduction. Store committed state only after lane 0, then continue speculative updates in registers.

   Next, benchmark splitting `j` into 32-column tiles:

   - Grid `(48 heads, 4 column tiles)`, 128 threads/block.
   - `j = tile*32 + (tid&31)`, `it = tid>>5`, retaining each thread’s 32-element `i` accumulation.
   - Shared `part[4][32]`, `dj[32]`, and complete 128-element Q/K vectors.

   This expands the current **48 CTAs to 192**, distributing work across more SMs without duplicating state traffic or changing the reduction partition. Also benchmark 64-column tiles.

   The current allocation is 64 KiB dynamic plus 3.5 KiB static shared memory, allowing only one such CTA per SM under sm_120’s 128 KiB shared-memory limit. [Blackwell resource limits](https://docs.nvidia.com/cuda/blackwell-tuning-guide/).

   Check register spilling before accepting the result. Apply the same implementation strategy to [delta_chunk3, spec3.cu:81](/mnt/ai/projects/q27/src/spec3.cu:81) where that standalone path remains used. The serving path already calls `delta_all`; optimizing only `delta_chunk3` would miss it.

3. **Bound and parallelize drafter attention: approximately 0.15–0.5 ms/round.**

   [dflash2.cu:93](/mnt/ai/projects/q27/src/dflash2.cu:93) iterates over every retained context row. Visibility is applied only **after** the QK dot product at line 106, and the V accumulation at line 123 still visits masked rows. Since ingest compacts only at capacity, the kernel scans roughly 2048–4096 context rows for a 2048-position window.

   First restrict iteration to the newest at-most-2048 context rows, retaining the exact per-query visibility predicate. A device-managed circular buffer can also eliminate compaction and make addressing graph-stable; compaction itself is too infrequent to be a major average-time saving.

   Then replace the attention body:

   - QK currently assigns adjacent warp lanes to different tokens, producing **4096-byte-stride K accesses** at each component load. Map warp lanes across the 128 head components instead.
   - [dflash2.cu:110](/mnt/ai/projects/q27/src/dflash2.cu:110) makes one thread scan all scores for max and exponential sum. Use parallel reductions.
   - Exponentials are evaluated again at line 119. Store the first-pass exponentials and normalize them once.
   - For the longer contexts, use split-context CTAs with partial max/sum/output and a merge kernel, shortening the serial V accumulation.

   Reduce the fixed shared-memory reservation at [dflash2.cu:574](/mnt/ai/projects/q27/src/dflash2.cu:574) to the bounded window or tile size. Keep FP32 KV initially to isolate the execution-layout change. Parallel reductions change drafter numerics, so recheck acceptance.

4. **Graph ingest and remove pageable round-control copies: approximately 0.1–0.35 ms/round.**

   [Dflash2::ingest, dflash2.cu:453](/mnt/ai/projects/q27/src/dflash2.cu:453) currently submits approximately **33 kernels plus control copies** for a decode chunk: 11 quantizations, 11 matmuls, one hidden norm, five K norms, and five RoPE kernels.

   Capture one ingest graph per accepted count, but fix the moving destinations first: `kr0`/`vr0` at line 463 bake the current append offset into kernel arguments. Project into fixed staging buffers, then scatter into ring slots using device counters. Generate positions on-device.

   Replace stack-backed `oc` at [engine.cuh:2859](/mnt/ai/projects/q27/src/engine.cuh:2859) with persistent pinned storage. Likewise eliminate the pageable anchor/position uploads at [dflash2.cu:490](/mnt/ai/projects/q27/src/dflash2.cu:490): an opening graph kernel can read existing `d_token`/`d_P` and initialize the drafter inputs. Pageable `cudaMemcpyAsync` may block or synchronize for staging. [CUDA copy semantics](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).

   Once control is device-resident, capture prep, draft, proposal transfer, and verify together. Replace the seven four-byte proposal copies at [engine.cuh:2853](/mnt/ai/projects/q27/src/engine.cuh:2853) with one scatter kernel.

   **The outcome wait remains necessary for the current host callbacks.** An event wait only helps if work is queued beyond the outcome copy. Potential overlap is next-draft computation against the target fold, using separate streams and making verify wait for both.

   Preserve the callback boundary: [engine.cuh:4507](/mnt/ai/projects/q27/src/engine.cuh:4507) can change accepted count and pending token, while DFlash2 currently ingests and updates its mirrors beforehand. Reconcile those values after truncation before committing ring rows or starting the next draft.

5. **Replace repeated top-16 rescans with hierarchical selection: approximately 0.05–0.2 ms/round.**

   [top16a, dflash2.cu:150](/mnt/ai/projects/q27/src/dflash2.cu:150) rescans each roughly 485-element slice 16 times, with growing exclusion checks and block-wide reductions.

   [top16b, dflash2.cu:195](/mnt/ai/projects/q27/src/dflash2.cu:195) repeats that pattern over **8192 candidates in only seven CTAs**. This is the more obvious occupancy and serial-work problem.

   Load each stage-A slice once into registers, select/sort its top 16, then add an intermediate merge: groups of 16 slice lists produce one top-16 list. That gives **32 merge CTAs per row, 224 total**, leaving only 512 candidates per row for the final merge. Preserve ordering by `(value descending, token ID ascending)`.

   The walk itself is a lower priority: [dflash2.cu:253](/mnt/ai/projects/q27/src/dflash2.cu:253) has a real predecessor dependency across positions, and its successor-codebook payload is only about 56 KiB at K=7. Retain that sequential walk; add confidence there.

6. **Collapse the verify’s 40 feature-tap copy nodes: approximately 0.05–0.15 ms/round.**

   [engine.cuh:2289](/mnt/ai/projects/q27/src/engine.cuh:2289) captures eight separate 20 KiB D2D copies at each of five tap layers.

   Replace each eight-copy sequence with one gather kernel over `(lane, hidden component)`: **40 nodes become five**. Better, specialize the preceding residual-add kernel at those five layers to write both the residual and its tap destination, eliminating the copy nodes and source rereads entirely.

   The total payload is only about 0.82 MB; this optimization targets node scheduling overhead.

7. **Remove 25 redundant quantizations per round: approximately 0.04–0.1 ms/round.**

   Split [mmq, dflash2.cu:344](/mnt/ai/projects/q27/src/dflash2.cu:344) into activation quantization and multiplication from an existing `XQuant`.

   The exact redundancies are:

   - Q/K/V share `ny0`: remove two quantizations per layer at [dflash2.cu:555](/mnt/ai/projects/q27/src/dflash2.cu:555).
   - Gate/up share `ny0`: remove one per layer at [dflash2.cu:591](/mnt/ai/projects/q27/src/dflash2.cu:591).
   - Head/selector share `nhf`: reuse `hxq` at [dflash2.cu:621](/mnt/ai/projects/q27/src/dflash2.cu:621), removing one.
   - All ten ingest K/V projections share unchanged `s_ct`: quantize once outside the layer loop at [dflash2.cu:460](/mnt/ai/projects/q27/src/dflash2.cu:460), removing nine.

   That is **16 draft + nine ingest** quantization launches removed without changing quantized values.

   Weight sharing across lanes is already implemented correctly. Also, the actual Q4 N=8 launch tier already permits three CTAs/SM; the four-CTA comment near the kernel is stale. See [kernels.cu:23](/mnt/ai/projects/q27/src/kernels.cu:23).

8. **Small fusion/reduction improvements: approximately 0.03–0.1 ms/round combined.**

   For [gdn_convnorm3, spec3.cu:337](/mnt/ai/projects/q27/src/spec3.cu:337), retain shared-memory reduction stages 64 and 32, then perform offsets 16…1 with warp shuffles. Broadcast the final result through shared memory. This preserves the existing addition tree while cutting the norm reduction from eight block barriers to three.

   For [k_d2_dconv, dflash2.cu:57](/mnt/ai/projects/q27/src/dflash2.cu:57), the current `(20,W) × 256` geometry already covers H=5120 with coalesced channel accesses. The clearer saving is fusing finish-convolution with residual addition at [dflash2.cu:580](/mnt/ai/projects/q27/src/dflash2.cu:580) and [dflash2.cu:597](/mnt/ai/projects/q27/src/dflash2.cu:597): ten fewer launches. Read the unchanged `no` buffer and add directly into `nx`; writing convolution results back into its input would introduce cross-row hazards.

For validation, the existing [gdn_fuse_eq harness](/mnt/ai/projects/q27/tools/gdn_fuse_eq.cu:101) checks the right surfaces: committed state, ring, every lane’s output, and record buffers. Use those comparisons for GDN changes, then compare round time **and tokens/round** on the live workload for attention and adaptive-width changes.
tokens used
124,235
**The best opportunities are adaptive verification, register-resident GDN state, and the drafter’s attention kernel.** I would not budget the full 0.83 ms as removable host overhead.

This is a read-only review. Savings below are engineering estimates, not measurements: the NVIDIA driver was unavailable locally. Ranges overlap and should not be added blindly.

First, fix the attribution: [engine.cuh:2848](/mnt/ai/projects/q27/src/engine.cuh:2848) records the first event **after previously queued work**, while [engine.cuh:2866](/mnt/ai/projects/q27/src/engine.cuh:2866) submits ingest outside the measured GPU intervals. `post_round` also submits the fold outside this function at [engine.cuh:4518](/mnt/ai/projects/q27/src/engine.cuh:4518). Consequently, “host” can include outstanding fold/ingest execution and pageable-copy staging. Add separate ingest/fold events and measure completion-to-completion round latency before assigning that residual to synchronization.

1. **Adaptive verify width: potentially 0.5–2 ms/round, conditional on retaining token yield.**

   Compute confidence in [k_d2_walk, dflash2.cu:269](/mnt/ai/projects/q27/src/dflash2.cu:269): thread 0 already scans the 16 **selector-adjusted** scores. Retain the best and second-best score at each position and write their margin. Using only `d_cval` margins would omit the predecessor/codebook contribution that actually determines the proposal.

   Keep drafting K=7, then choose verify W=2…8. Fit prefix survival against those margins using actual acceptance outcomes, and choose width using measured `verify_ms[W]`. The relevant objective is tokens/time:

   \[
   \text{beneficial if expected tokens lost}<\frac{4.87}{19.96}\times\text{milliseconds saved}.
   \]

   Thus, saving **2 ms permits losing fewer than 0.49 tokens/round**; saving 1 ms permits fewer than 0.24. The aggregate 55% acceptance does not establish whether this gate will win.

   Two implementation details matter:

   - **Existing graphs are not directly reusable.** [engine.cuh:2621](/mnt/ai/projects/q27/src/engine.cuh:2621) captures `spec_verify_launches`, whose forward receives no taps. Capture DFlash2 variants using `spec_verify_forward(v, d2_vtaps)` plus `spec_verify_tail(v)`, as [engine.cuh:2794](/mnt/ai/projects/q27/src/engine.cuh:2794) does. Set both member `vw` and view width during capture because `gdn_mix` reads the member. Also ensure all required widths exist; the current capture loop depends on `gate_maxd`.
   - Avoid an extra steady-state host rendezvous by putting the width bodies under a CUDA conditional SWITCH, with the walk setting its handle. This requires constructing conditional body graphs, not inserting existing `cudaGraphExec_t` handles. [NVIDIA documents device-evaluated SWITCH nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

   **Do not initially shrink the draft itself:** its attention is bidirectional across noise rows, so reducing K changes even the retained proposals. Narrowing only verification gives a cleaner experiment.

2. **Keep GDN state in registers and split the independent value columns: approximately 0.3–0.9 ms/round.**

   [spec3.cu:197](/mnt/ai/projects/q27/src/spec3.cu:197) stages the entire 128×128 state in shared memory. Both passes at [spec3.cu:249](/mnt/ai/projects/q27/src/spec3.cu:249) and [spec3.cu:265](/mnt/ai/projects/q27/src/spec3.cu:265) repeatedly load and store it.

   **Each state element has one owning thread throughout every lane.** Cross-thread communication involves `sq`, `sk`, `part`, and `dj`; the state matrix itself need not be shared. Use a fully unrolled `float sreg[32]`, retaining it across all eight steps. The single-step implementation already demonstrates this ownership at [blocks.cu:221](/mnt/ai/projects/q27/src/blocks.cu:221).

   Preserve the current arithmetic order: decay → prediction → four-part reduction → delta → state update → output reduction. Store committed state only after lane 0, then continue speculative updates in registers.

   Next, benchmark splitting `j` into 32-column tiles:

   - Grid `(48 heads, 4 column tiles)`, 128 threads/block.
   - `j = tile*32 + (tid&31)`, `it = tid>>5`, retaining each thread’s 32-element `i` accumulation.
   - Shared `part[4][32]`, `dj[32]`, and complete 128-element Q/K vectors.

   This expands the current **48 CTAs to 192**, distributing work across more SMs without duplicating state traffic or changing the reduction partition. Also benchmark 64-column tiles.

   The current allocation is 64 KiB dynamic plus 3.5 KiB static shared memory, allowing only one such CTA per SM under sm_120’s 128 KiB shared-memory limit. [Blackwell resource limits](https://docs.nvidia.com/cuda/blackwell-tuning-guide/).

   Check register spilling before accepting the result. Apply the same implementation strategy to [delta_chunk3, spec3.cu:81](/mnt/ai/projects/q27/src/spec3.cu:81) where that standalone path remains used. The serving path already calls `delta_all`; optimizing only `delta_chunk3` would miss it.

3. **Bound and parallelize drafter attention: approximately 0.15–0.5 ms/round.**

   [dflash2.cu:93](/mnt/ai/projects/q27/src/dflash2.cu:93) iterates over every retained context row. Visibility is applied only **after** the QK dot product at line 106, and the V accumulation at line 123 still visits masked rows. Since ingest compacts only at capacity, the kernel scans roughly 2048–4096 context rows for a 2048-position window.

   First restrict iteration to the newest at-most-2048 context rows, retaining the exact per-query visibility predicate. A device-managed circular buffer can also eliminate compaction and make addressing graph-stable; compaction itself is too infrequent to be a major average-time saving.

   Then replace the attention body:

   - QK currently assigns adjacent warp lanes to different tokens, producing **4096-byte-stride K accesses** at each component load. Map warp lanes across the 128 head components instead.
   - [dflash2.cu:110](/mnt/ai/projects/q27/src/dflash2.cu:110) makes one thread scan all scores for max and exponential sum. Use parallel reductions.
   - Exponentials are evaluated again at line 119. Store the first-pass exponentials and normalize them once.
   - For the longer contexts, use split-context CTAs with partial max/sum/output and a merge kernel, shortening the serial V accumulation.

   Reduce the fixed shared-memory reservation at [dflash2.cu:574](/mnt/ai/projects/q27/src/dflash2.cu:574) to the bounded window or tile size. Keep FP32 KV initially to isolate the execution-layout change. Parallel reductions change drafter numerics, so recheck acceptance.

4. **Graph ingest and remove pageable round-control copies: approximately 0.1–0.35 ms/round.**

   [Dflash2::ingest, dflash2.cu:453](/mnt/ai/projects/q27/src/dflash2.cu:453) currently submits approximately **33 kernels plus control copies** for a decode chunk: 11 quantizations, 11 matmuls, one hidden norm, five K norms, and five RoPE kernels.

   Capture one ingest graph per accepted count, but fix the moving destinations first: `kr0`/`vr0` at line 463 bake the current append offset into kernel arguments. Project into fixed staging buffers, then scatter into ring slots using device counters. Generate positions on-device.

   Replace stack-backed `oc` at [engine.cuh:2859](/mnt/ai/projects/q27/src/engine.cuh:2859) with persistent pinned storage. Likewise eliminate the pageable anchor/position uploads at [dflash2.cu:490](/mnt/ai/projects/q27/src/dflash2.cu:490): an opening graph kernel can read existing `d_token`/`d_P` and initialize the drafter inputs. Pageable `cudaMemcpyAsync` may block or synchronize for staging. [CUDA copy semantics](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html).

   Once control is device-resident, capture prep, draft, proposal transfer, and verify together. Replace the seven four-byte proposal copies at [engine.cuh:2853](/mnt/ai/projects/q27/src/engine.cuh:2853) with one scatter kernel.

   **The outcome wait remains necessary for the current host callbacks.** An event wait only helps if work is queued beyond the outcome copy. Potential overlap is next-draft computation against the target fold, using separate streams and making verify wait for both.

   Preserve the callback boundary: [engine.cuh:4507](/mnt/ai/projects/q27/src/engine.cuh:4507) can change accepted count and pending token, while DFlash2 currently ingests and updates its mirrors beforehand. Reconcile those values after truncation before committing ring rows or starting the next draft.

5. **Replace repeated top-16 rescans with hierarchical selection: approximately 0.05–0.2 ms/round.**

   [top16a, dflash2.cu:150](/mnt/ai/projects/q27/src/dflash2.cu:150) rescans each roughly 485-element slice 16 times, with growing exclusion checks and block-wide reductions.

   [top16b, dflash2.cu:195](/mnt/ai/projects/q27/src/dflash2.cu:195) repeats that pattern over **8192 candidates in only seven CTAs**. This is the more obvious occupancy and serial-work problem.

   Load each stage-A slice once into registers, select/sort its top 16, then add an intermediate merge: groups of 16 slice lists produce one top-16 list. That gives **32 merge CTAs per row, 224 total**, leaving only 512 candidates per row for the final merge. Preserve ordering by `(value descending, token ID ascending)`.

   The walk itself is a lower priority: [dflash2.cu:253](/mnt/ai/projects/q27/src/dflash2.cu:253) has a real predecessor dependency across positions, and its successor-codebook payload is only about 56 KiB at K=7. Retain that sequential walk; add confidence there.

6. **Collapse the verify’s 40 feature-tap copy nodes: approximately 0.05–0.15 ms/round.**

   [engine.cuh:2289](/mnt/ai/projects/q27/src/engine.cuh:2289) captures eight separate 20 KiB D2D copies at each of five tap layers.

   Replace each eight-copy sequence with one gather kernel over `(lane, hidden component)`: **40 nodes become five**. Better, specialize the preceding residual-add kernel at those five layers to write both the residual and its tap destination, eliminating the copy nodes and source rereads entirely.

   The total payload is only about 0.82 MB; this optimization targets node scheduling overhead.

7. **Remove 25 redundant quantizations per round: approximately 0.04–0.1 ms/round.**

   Split [mmq, dflash2.cu:344](/mnt/ai/projects/q27/src/dflash2.cu:344) into activation quantization and multiplication from an existing `XQuant`.

   The exact redundancies are:

   - Q/K/V share `ny0`: remove two quantizations per layer at [dflash2.cu:555](/mnt/ai/projects/q27/src/dflash2.cu:555).
   - Gate/up share `ny0`: remove one per layer at [dflash2.cu:591](/mnt/ai/projects/q27/src/dflash2.cu:591).
   - Head/selector share `nhf`: reuse `hxq` at [dflash2.cu:621](/mnt/ai/projects/q27/src/dflash2.cu:621), removing one.
   - All ten ingest K/V projections share unchanged `s_ct`: quantize once outside the layer loop at [dflash2.cu:460](/mnt/ai/projects/q27/src/dflash2.cu:460), removing nine.

   That is **16 draft + nine ingest** quantization launches removed without changing quantized values.

   Weight sharing across lanes is already implemented correctly. Also, the actual Q4 N=8 launch tier already permits three CTAs/SM; the four-CTA comment near the kernel is stale. See [kernels.cu:23](/mnt/ai/projects/q27/src/kernels.cu:23).

8. **Small fusion/reduction improvements: approximately 0.03–0.1 ms/round combined.**

   For [gdn_convnorm3, spec3.cu:337](/mnt/ai/projects/q27/src/spec3.cu:337), retain shared-memory reduction stages 64 and 32, then perform offsets 16…1 with warp shuffles. Broadcast the final result through shared memory. This preserves the existing addition tree while cutting the norm reduction from eight block barriers to three.

   For [k_d2_dconv, dflash2.cu:57](/mnt/ai/projects/q27/src/dflash2.cu:57), the current `(20,W) × 256` geometry already covers H=5120 with coalesced channel accesses. The clearer saving is fusing finish-convolution with residual addition at [dflash2.cu:580](/mnt/ai/projects/q27/src/dflash2.cu:580) and [dflash2.cu:597](/mnt/ai/projects/q27/src/dflash2.cu:597): ten fewer launches. Read the unchanged `no` buffer and add directly into `nx`; writing convolution results back into its input would introduce cross-row hazards.

For validation, the existing [gdn_fuse_eq harness](/mnt/ai/projects/q27/tools/gdn_fuse_eq.cu:101) checks the right surfaces: committed state, ring, every lane’s output, and record buffers. Use those comparisons for GDN changes, then compare round time **and tokens/round** on the live workload for attention and adaptive-width changes.
