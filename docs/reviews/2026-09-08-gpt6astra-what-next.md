# gpt-6-astra advisory: what next after 2026-09-08 (ranked plan)

Reviewer: gpt-6-astra via codex exec (read-only sandbox, xhigh), static review of the 2026-09-06..08 BUILDLOG entries, the prefill plan, the campaign README and the engine/drafter code, asked to rank the author's candidate list, add what is missing, localize the width > 8 divergence, and name tomorrow's experiment. Dispositions in BUILDLOG 2026-09-08 (p).

## Report (reviewer's words)

**Spend tomorrow on a bounded width investigation, then prioritize cache robustness and one TMA spike.** Decode dominates, but K=10 plausibly adds only **3–6% tokens/round before its extra cost**, so this is a measured opportunity, not a large promised win. Crucially, the reported width-8 wall coincides exactly with the CLI’s GEMV→MMA numerical-family switch; establish whether a state bug remains with that switch disabled. Keep lever B off, defer A and chunk graphs, and stop tuning already-measured dead ends. The 216-versus-219 t/s result establishes practical parity; it does not establish equivalent task quality or completion cost. This review was static: no builds, execution of project code, or changes.

Session estimates below are engineering estimates; projected gains are conditional.

| Rank | Candidate | Expected value on current traffic | Effort, sessions | Risk | Decision |
|---:|---|---|---:|---|---|
| 1 | **(1) Resolve width wall; sweep K=8…11** | Approximately 3–6% gross decode gain at K=10; net gain unknown | ½–1 diagnosis; 1–2 sweep/gates if localized | Medium–high correctness | **Do, bounded** |
| 2 | **Added: reproducible turn replay, quality and queue attribution** | Makes small gains distinguishable from ring/history drift; identifies larger serving opportunities | 1–2 initially | Low | **Do alongside experiments** |
| 3 | **(5a) Shared-cut promotion and persistence failure paths** | Protects multi-second TTFT savings after client changes; current incidence unknown | 1–2 | Medium cache correctness | **Do next** |
| 4 | **(2) TMA W/X-only spike** | Roughly 1–3% total wall if the kernel gate passes; more valuable for cold TTFT | 1 spike; 2–3 additional port/gates | Low isolated; medium integrated | **Do one spike** |
| 5 | **(5b) Restore drafter context** | Recovers about 17% of affected requests’ inflated decode time, before reseeding cost; aggregate incidence unknown | ½–1 attribution; 1–3 implementation | Medium state/memory | **Measure now; implement selectively** |
| 6 | **(3) Lever A** | Up to ~25 ms per eligible warm turn; about 0.6% of the cited run | 3–5 | High snapshot/numerics | **Defer** |
| 7 | **(4) Prefill chunk graphs** | Removable submission gaps, probably milliseconds per chunk—not 55 ms | ½ attribution; 2–4 implementation | Medium capture/state | **Defer implementation** |
| 8 | **(6) Drafter numerics, lanes 6–7** | Likely sub-1% overall without new evidence | 1–2 | Low target risk; acceptance regression | **Defer** |
| 9 | **Added: DFlash2 batching** | Potentially large aggregate throughput gain under sustained concurrency; unpriced today | 4–8+ | High scheduling/state/memory | **Defer until queue evidence** |

1. **[P1] The first suspect is the width-9 dispatch transition; GDN orchestration is the next suspect.**

   The historical divergence is real evidence, but its attribution to an eager verify/GDN/fold bug is not yet established by the cited experiment. `gemm_min` defaults to **9**, and `mm5` switches eligible projections to MMA at that width. The code explicitly says GEMV and MMA are different numerical families. A fixed divergence position across K≥8 is therefore consistent with the dispatch transition. Evidence: [engine.cuh:417](/mnt/ai/projects/q27/src/engine.cuh:417), [engine.cuh:1992](/mnt/ai/projects/q27/src/engine.cuh:1992), [engine.cuh:2870](/mnt/ai/projects/q27/src/engine.cuh:2870), [BUILDLOG.md:15701](/mnt/ai/projects/q27/docs/BUILDLOG.md:15701).

   If a genuine corruption survives that control, inspect **member/view width consistency, recorded rows, and accepted-prefix folding** first. `gdn_mix` consumes member `vw`; projections consume `LaneView::vw`. However, the original missing-width fix is present in **both** CLI and serving setup. The record arena already allocates `W_MAX−1` rows, and fold graphs cover every corresponding accepted length. There is no obvious remaining eight-row allocation ceiling in these entry points. Evidence: [engine.cu:234](/mnt/ai/projects/q27/src/engine.cu:234), [engine.cuh:2858](/mnt/ai/projects/q27/src/engine.cuh:2858), [engine.cuh:2065](/mnt/ai/projects/q27/src/engine.cuh:2065), [engine.cuh:1302](/mnt/ai/projects/q27/src/engine.cuh:1302), [engine.cuh:2741](/mnt/ai/projects/q27/src/engine.cuh:2741).

   **Do not widen `W_PLUMB`.** It is already 16; engine and drafter capacities are 12, sufficient for K≤11. Top-16 means candidates **per draft position**, not sixteen verified positions. Poor proposals alone cannot explain incorrect greedy output from a correct verifier. Evidence: [cuda_common.h:34](/mnt/ai/projects/q27/src/cuda_common.h:34), [dflash2.h:28](/mnt/ai/projects/q27/src/dflash2.h:28), [dflash2.cu:359](/mnt/ai/projects/q27/src/dflash2.cu:359).

2. **[P1] Tomorrow morning’s single experiment: a matched-family width crossover, followed by a production K comparison if it passes.**

   Start with the existing **greedy identity-versus-plain/`--spec` gate**, using the prose reproducer, then code-write, code-edit and echo; generate at least 512 tokens to cross the reported late failures. Test **W={2,7,8,9,10,11,12}**, hence K={1,6,7,8,9,10,11}, with identical model, head selection, prompt and fp8 KV.

   For the CLI diagnostic, use `Q27_KV=fp8`, `Q27_GEMM_MIN=99`, `Q27_SUFFIX=0`, `Q27_SAMPLED=0`, and unset serving’s `Q27_DFLASH2`. **Include `--spec` alongside `--dflash2 <full-pack> --k K`: `Q27_GEMM_MIN` is parsed inside `build_spec_graphs`, which the CLI invokes only with `--spec`.** A bare `--dflash2` run otherwise leaves the default threshold intact. Use a full CLI pack containing the embedding, not the stripped serving pack. Evidence: [engine.cu:127](/mnt/ai/projects/q27/src/engine.cu:127), [engine.cuh:2500](/mnt/ai/projects/q27/src/engine.cuh:2500), [dflash2.cu:709](/mnt/ai/projects/q27/src/dflash2.cu:709).

   Compare each width with `Q27_D2_NOGRAPH` **unset**, then `Q27_D2_NOGRAPH=1`. Setting it to `0` also disables the CLI verify graph because the check is presence-based. This knob does **not** disable serving verification graphs. Evidence: [engine.cu:245](/mnt/ai/projects/q27/src/engine.cu:245), [engine.cuh:3038](/mnt/ai/projects/q27/src/engine.cuh:3038).

   If matched-GEMV identity fails, use one temporary diagnostic gate with **fixed proposal tokens**, identical incoming state, and each committed length `n=1…W`. Compare per-layer activations, committed lane-0 state, every recorded row, and post-fold S/conv history against sequential execution. Poison unused buffers; assert member `vw == view.vw`. First mismatch before folding implicates forward/plumbing; first mismatch after folding implicates record/commit. Existing `gdn_fuse_eq` and `ninv_test` provide useful foundations, but their current width sets omit several transition widths. Evidence: [gdn_fuse_eq.cu:68](/mnt/ai/projects/q27/tools/gdn_fuse_eq.cu:68), [ninv_test.cu:51](/mnt/ai/projects/q27/tools/ninv_test.cu:51).

   If GEMV passes, do not manufacture a core fix: validate the production MMA family at W=8…12. Use `Q27_DFLASH2_K=7…11`, `Q27_D2_VGEMM=1`, `Q27_BATCH=0`, the Q8 serving pack, reserve 3 GB, sampled walk/ring retention, and `Q27_PF_FOLDLAST=0`. Start with `Q27_D2_FOLD=sync`, then repeat with normal overlap. A serving **all-GEMV** control requires **both** `Q27_D2_VGEMM=0` and `Q27_GEMM_MIN=99`; the former alone still permits MMA above width 8.

   Finish with interleaved K=7/K=10 comparisons on matched initialized conversation histories at 12.5K and 50K, then representative warm `/v1/messages` turns. Measure **delivered tokens per total round wall**, not just acceptance. My continuation bar would be a repeatable **≥2% request-wall improvement**, after correctness gates. If the fixed-family failure remains unlocalized after one session, keep K=7 and park widening.

3. **[P1] K=10’s honest acceptance budget is approximately +0.10 to +0.24 tokens/round—not three extra tokens.**

   Let \(S_j=P(A\ge j)\). Ignoring end-of-request truncation,

   \[
   E[N_K]=1+\sum_{j=1}^{K}S_j,\qquad
   \Delta E[N_{10}]=S_8+S_9+S_{10}.
   \]

   September 7(f) gives \(S_7=0.076\), with late conditional continuation probabilities around 0.6–0.7. Extrapolating 0.65–0.70 gives **+0.102–0.117 tokens/round**, roughly **3–3.5%** over its reported 3.383 baseline. Those later probabilities were not measured. Evidence: [BUILDLOG.md:16618](/mnt/ai/projects/q27/docs/BUILDLOG.md:16618).

   The newer production profile is more favorable: \(S_7=0.137\), with conditionals near 0.75. Extending that slope predicts \(S_8,S_9,S_{10}\approx0.103,0.077,0.058\): **+0.238 tokens/round**, taking 3.738 to approximately **3.98**, or **+6.4%**. Evidence: [BUILDLOG.md:16260](/mnt/ai/projects/q27/docs/BUILDLOG.md:16260).

   These are planning estimates. Wider drafts change the bidirectional noise-row attention, so even the first seven proposal distributions can change. More concretely, drafter attention processes **32 query vectors per group**: W=8 fits one group; W≥9 requires two. Its backbone/head also remain on GEMV. Thus the target’s flat MMA cost does not establish a flat round. A 3–6% acceptance gain tolerates only roughly **0.5–1.1 ms** extra on a 17.35 ms round. Evidence: [dflash2.cu:109](/mnt/ai/projects/q27/src/dflash2.cu:109), [dflash2.cu:115](/mnt/ai/projects/q27/src/dflash2.cu:115), [dflash2.cu:520](/mnt/ai/projects/q27/src/dflash2.cu:520).

4. **[P1] Control ring history and quality before interpreting small performance differences.**

   Identical seeds and requests are insufficient: the log records unchanged output streams with substantially different round counts after prior requests changed ring contents. Reproduce the **preceding conversation sequence and cache state** in each arm; report first-round behavior separately from steady decode. Evidence: [BUILDLOG.md:15749](/mnt/ai/projects/q27/docs/BUILDLOG.md:15749).

   Extend the integrated sampled walk/rejection gate through K=11: it currently uses **K=3**, as does the top-16 test. Exercise late rejection, full acceptance, bonus/cap draws, empty residual, truncation and context-limit admission. Different-K sampled streams need not be byte-identical; require repeatability within an arm and the correct target distribution. Evidence: [test_kernels.cu:3050](/mnt/ai/projects/q27/src/test_kernels.cu:3050), [test_kernels.cu:3436](/mnt/ai/projects/q27/src/test_kernels.cu:3436), [engine.cuh:639](/mnt/ai/projects/q27/src/engine.cuh:639).

   Keep first-token distribution/turn replay and actual task success in the quality battery. The campaign’s nonempty/gold-file counts are single-pass proxies, and trajectories differ greatly. Do not optimize toward ninfer’s token count without establishing task quality. Evidence: [campaign README:31](/mnt/ai/projects/q27/bench/crossengine/agentic-2026-09-08/README.md:31), [small-turn review:63](/mnt/ai/projects/q27/docs/reviews/2026-09-08-gpt6astra-small-turn-levers.md:63).

5. **[P2] Promotion is worthwhile robustness work; include two concrete cache failure cases.**

   Promotion must address **discovery, the cold-only save predicate, and the shared 8192-token step gate**. Merely calling `shared_prefix` on restored requests will not publish anything. Require a strictly longer verified shared prefix, save at an actually reached chunk boundary, and bound promotion frequency. Test an old short entry followed by several new-client sessions with a longer common prefix. Evidence: [engine.cuh:4985](/mnt/ai/projects/q27/src/engine.cuh:4985), [engine.cuh:4313](/mnt/ai/projects/q27/src/engine.cuh:4313), [engine.cuh:4301](/mnt/ai/projects/q27/src/engine.cuh:4301).

   **New static finding:** `pfx_persist` reserves the key, then returns on staging-allocation failure without releasing it. `has()` treats that reservation as present, suppressing retries for the process lifetime. Put reservation ownership around the entire export-to-writer handoff; test allocation failure followed by retry. Also reconcile failed writes with `pfx_last_persist`, which advances before write success. Evidence: [engine.cuh:4326](/mnt/ai/projects/q27/src/engine.cuh:4326), [engine.cuh:4335](/mnt/ai/projects/q27/src/engine.cuh:4335), [prefix_cache.h:287](/mnt/ai/projects/q27/src/prefix_cache.h:287), [engine.cuh:4342](/mnt/ai/projects/q27/src/engine.cuh:4342).

   Longer term, eviction is by **write mtime**, not access recency. A useful old shared entry can age out despite frequent hits; measure that before introducing a more elaborate policy. Evidence: [prefix_cache.h:388](/mnt/ai/projects/q27/src/prefix_cache.h:388).

6. **[P2] Give TMA one session; retain the existing stop bar.**

   Test exactly the proposed intervention: **W/X bulk copies, an elected consumer thread, unchanged scale loading and arithmetic**. Preserve the ≥1.6× live-dispatch gate on both ffn_gate and attn_out at T=1024. Include quantizer/permutation costs when pricing integration. The measured 1.30×/1.41× result does not justify silently lowering the bar. Evidence: [prefill plan:301](/mnt/ai/projects/q27/docs/plans/2026-09-08-prefill-attack.md:301).

   Even applying 1.6× to a 60% GEMM share across *all* 14% prefill yields only **3.15% total-wall savings**; restricting the win to cold/saturated chunks lowers that. The isolated experiment remains attractive because it is bounded and could materially improve cold TTFT.

   If it fails, stop the port and further fold/occupancy/pipeline permutations. If it passes, preserve split-K dispatch decisions and the exact fold, and complete real-weight/activation gates; the current spike uses synthetic inputs. Evidence: [prefill plan:327](/mnt/ai/projects/q27/docs/plans/2026-09-08-prefill-attack.md:327), [gemm_w4a8_spike.cu:1090](/mnt/ai/projects/q27/tools/gemm_w4a8_spike.cu:1090).

7. **[P2] Restore reseeding needs an explicit state source and a break-even calculation.**

   Group requests by **valid ring rows after alignment and suffix length**, across P8/P9/RAM/disk—not merely `pfx>0`. The successful phase-0 campaign does not resolve the tiny-suffix case: every measured restore there re-prefilled at least 2.3K tokens. Evidence: [engine.cuh:2959](/mnt/ai/projects/q27/src/engine.cuh:2959), [campaign README:176](/mnt/ai/projects/q27/bench/crossengine/agentic-2026-09-08/README.md:176).

   You cannot simply replay the preceding 2048 tokens from the restored state at L: prefill mutates recurrent state forward. You need an earlier compatible checkpoint, saved taps, or saved derived drafter K/V. Raw taps cost **200 MiB** per 2048-row window; the existing five-layer fp32 drafter K/V representation is **80 MiB**, plus positions, and is another design worth pricing. Its compatibility must include drafter weights/numerics. Evidence: [engine.cuh:3906](/mnt/ai/projects/q27/src/engine.cuh:3906), [dflash2.h:25](/mnt/ai/projects/q27/src/dflash2.h:25), [dflash2.h:67](/mnt/ai/projects/q27/src/dflash2.h:67).

   With +20% rounds in the affected class, repairing it saves roughly one-sixth of its current decode time. A reseeding operation costing \(C\) pays only when affected decode time exceeds approximately \(6C\). Blanket recomputation is particularly unattractive for short tool-call responses.

8. **[P2] Reprice A and graphs using whole-run arithmetic.**

   With B default-off, a typical two-chunk warm turn still performs **two batched weight passes plus the final-token pass**. Graphing that final pass removed submission gaps, not its weight traffic. Likewise, the ~55 ms intercept is a fixed-cost observation, not a 55 ms launch-overhead measurement; the trace reported 88% GPU busy. Evidence: [engine.cuh:5064](/mnt/ai/projects/q27/src/engine.cuh:5064), [engine.cuh:5125](/mnt/ai/projects/q27/src/engine.cuh:5125), [engine.cuh:5169](/mnt/ai/projects/q27/src/engine.cuh:5169), [BUILDLOG.md:15916](/mnt/ai/projects/q27/docs/BUILDLOG.md:15916).

   At 192 eligible turns, A’s optimistic saving is **192×25 ms = 4.8 s**, approximately **0.6% of 800 s**. That is useful TTFT improvement but weak compensation for 3–5 sessions and boundary-state export risk. The log’s B estimate also needs denominator reconciliation: 219 turns×12 ms is approximately 2.6 s, not 1.3% of an 800 s run. Evidence: [BUILDLOG.md:15896](/mnt/ai/projects/q27/docs/BUILDLOG.md:15896), [BUILDLOG.md:15761](/mnt/ai/projects/q27/docs/BUILDLOG.md:15761).

   Defer both implementations. For graphs, first bound removable gaps, then solve host-valued base/T and shared-arena ownership without a graph per absolute position. For A, retain explicit boundary S **and raw convolution history**; current snapshots/export copy live state. Evidence: [engine.cuh:3864](/mnt/ai/projects/q27/src/engine.cuh:3864), [engine.cuh:4082](/mnt/ai/projects/q27/src/engine.cuh:4082), [engine.cuh:4227](/mnt/ai/projects/q27/src/engine.cuh:4227).

9. **[P2] Serving work could outrank kernel work, but measure the actual constraint.**

   There are already requests with **1–2 seconds of queue wait**. Attribute queue time by arrival concurrency and request class before deciding on batching or scheduling; a 12-instance campaign alone does not establish sustained GPU concurrency. DFlash2 batching requires real integration because fused commits currently bypass its ring mirrors. Evidence: [prodpfx2 requests:18](/mnt/ai/projects/q27/bench/crossengine/agentic-2026-09-08/prodpfx2-xhigh.req.txt:18), [server.cu:535](/mnt/ai/projects/q27/src/server.cu:535).

   The measured **0.2 ms round host component** is already small, and its timer excludes subsequent callbacks and other `post_round` work. Profile the complete round before pursuing another host-overhead rewrite. The sampled nucleus kernel’s prior **0.62 ms** is a more concrete bounded target if a fresh profile still shows it. Evidence: [engine.cuh:3064](/mnt/ai/projects/q27/src/engine.cuh:3064), [engine.cuh:4788](/mnt/ai/projects/q27/src/engine.cuh:4788), [BUILDLOG.md:16337](/mnt/ai/projects/q27/docs/BUILDLOG.md:16337).

   Audit **actual pool-clamped context and free VRAM after all graphs/drafter allocations**, then 128K admission, warm continuation, eviction and tool-call quality. Cache persistence currently stops at 65536, and a 69.7K conversation already demonstrated the resulting re-prefill cliff. This deserves measurement before a 128K attention rewrite. Evidence: [server.cu:1080](/mnt/ai/projects/q27/src/server.cu:1080), [launch_q27_38.sh:37](/mnt/ai/projects/q27/tools/launch_q27_38.sh:37), [campaign README:84](/mnt/ai/projects/q27/bench/crossengine/agentic-2026-09-08/README.md:84).

   Two smaller corrections: current code already honors `output_config.effort`; verify rendering rather than implementing it again. Conversely, the launch helper still recognizes `serving ON`, despite the log documenting that this precedes listener readiness. Evidence: [api_common.h:408](/mnt/ai/projects/q27/src/api_common.h:408), [server.cu:2681](/mnt/ai/projects/q27/src/server.cu:2681), [launch_q27_38.sh:52](/mnt/ai/projects/q27/tools/launch_q27_38.sh:52).

10. **[P3] Explicitly stop spending time on these now.**

    Keep **lever C parked and B default-off**. Stop GEMV rewrites under the old bitwise contract, further exact-fold micro-tuning, native-fp4 revival, and interpreting the 216/219 standings as a remaining performance gap. Do not retune the ladder/suffix path for current production: DFlash2 wins this workload and suffix acceptance was absent in the cited campaign. Defer lanes 6–7 numerics until the wider-K experiment identifies a specific acceptance deficit. Evidence: [BUILDLOG.md:15769](/mnt/ai/projects/q27/docs/BUILDLOG.md:15769), [BUILDLOG.md:15757](/mnt/ai/projects/q27/docs/BUILDLOG.md:15757), [BUILDLOG.md:16315](/mnt/ai/projects/q27/docs/BUILDLOG.md:16315), [prefill plan:434](/mnt/ai/projects/q27/docs/plans/2026-09-08-prefill-attack.md:434), [BUILDLOG.md:16465](/mnt/ai/projects/q27/docs/BUILDLOG.md:16465).
