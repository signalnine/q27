# gpt-6-astra review: prefill attack plan (2026-09-08)

Reviewer: gpt-6-astra via codex exec (read-only sandbox, xhigh), static review of
docs/plans/2026-09-08-prefill-attack.md + docs/perf-attribution-prefill-2026-09-08.md
against the code, before the plan was handed to a fresh context. No GPU run on the
reviewer's side. Every numbered item below was folded into the plan the same evening;
the plan text is the authority, this file is the evidence for why it says what it says.

## Verdict (reviewer's words)

**Phase 0 can handle these conversations, but the plan overstates its guarantees. Phase 2 also has a stale baseline and ineffective gates.** Static inspection only; no builds, workloads, or services were run.

1. **Anthropic does provide a persistable stable boundary.** `chatml_prompt` sets `stable_off` before appending the assistant opener and think block ([api_common.h:804]). `/v1/messages` encodes that prefix, records `stable_len`, then appends the separately encoded tail ([server.cu:2752]). Thus `stable_len < NP` on this path; ordinary extending turns also have `base < stable_len`.

   The stable save at [engine.cuh:5032] therefore reaches this additional predicate ([engine.cuh:4273]):
   ```
   cache enabled && !writer_busy
   && 4096 <= L <= 65536
   && (last_persist == 0 || L - last_persist >= 8192)
   && !cache.has(prompt, L)
   ```
   **Change plan lines 37–39 and 62–64:** explain this predicate explicitly; persistence happens at eligible prefill boundaries, not automatically when the slot is evicted.

2. **The system save can block the first conversation save through the shared step gate.** For example, saving a system cut at 21,504 makes a 28,000-token stable boundary ineligible: growth is only 6,496. Subsequent turns can save once their stable boundary reaches 29,696. A 48K turn likewise need not save its exact current boundary if a sufficiently recent shorter entry exists.

   `has()` checks the **same length and token hash**, so an existing system entry does not itself block a longer stable entry ([prefix_cache.h:223]). A successful export publishes to RAM before the disk writer finishes ([engine.cuh:4304]). The ~350-token foreign request cannot persist under the 4096 minimum; it destroys incompatible VRAM snapshots/checkpoints but leaves host/disk entries available ([engine.cuh:4960]).

   **Change the “one step at most” claim to a conditional target.** Busy writers skip eligible boundaries without queuing a retry; failed disk writes still leave `pfx_last_persist` advanced. Also, a shared P9 hit can carry the previous conversation’s scalar `pfx_last_persist`, since only `base==0` clears it ([engine.cuh:4940]).

3. **Restore selection can defeat the proposed hit-depth guarantee.** Resolution is P8 → P9 → host RAM → disk, stopping at the first tier with a match; it does not choose the longest match across tiers ([engine.cuh:4873]). A shorter RAM system entry can therefore hide a longer disk conversation entry.

   **Add to Phase 0:** diagnose selected tier and restored length before blaming persistence. Require a controlled `A → foreign ~350-token request → extended A` case at both 28K and 48K, recording the last saved boundary and returned `hit`. Treat “six full misses become zero” as a measured acceptance target, not a code guarantee.

4. **65536 removes the relevant save-size cap; memory accounting needs correction.** The cap applies to saved entry length `L`, not incoming prompt length. Disk `find()` does not enforce `max_tokens` on restores ([prefix_cache.h:179]); even the old 32768 cap permits a 48K request to restore an older shorter prefix.

   For the stated fp8 geometry, `pfx_bytes(65536)` is approximately **2.44 GB**, including recurrent state ([engine.cuh:4134]). A 16 GB RAM budget yields **six slots**, and each engine additionally pins **two** staging buffers: approximately **19.5 GB pinned total** for one engine, plus up to 40 GB of tmpfs files ([prefix_ram.h:48], [server.cu:1185]).

   **Replace plan lines 47–49:** check actual RAM-slot count, both staging allocations, free host/tmpfs capacity, and slot-0 context with decode reserve. RAM allocation failure reduces/disables that tier; staging is the fallback. Cache flags do not enlarge GPU context, which can be pool-clamped ([server.cu:1080]). Disk eviction uses write-time `mtime`, without touching entries on reads, so hot entries are not protected by true access LRU ([prefix_cache.h:283]).

5. **P16b cross-conversation reuse is plausible, not established by these logs.** Disk-cache initialization must succeed: RAM setup and `sys_len` computation depend on `pfx_cache.enabled()`; `Q27_SYSBLK` only enables logging ([server.cu:2758]). With `base==0`, ordinary messages save at `floor(sys_len/PF_T)*PF_T`, subject to the same minimum, step, busy, and `has()` gates ([engine.cuh:4285], [engine.cuh:5019]). Missing that single opportunity does not schedule another system save.

   Hits require identical tokens **through the saved cut**. Equal `sys_len` does not prove equality; unequal lengths do not prove a miss if differences occur after the cut. Effort, tool selection/declaration order, system content, and billing normalization all matter ([server.cu:2681], [api_common.h:744]). The normalizer handles `cch=` and fourth-plus version components, requires the header at string start, and preserves stamp length ([api_common.h:233]).

   **Replace plan lines 65–70:** verify token LCP against the saved cut; ninfer’s hits are supporting evidence, not proof under q27’s rendering. Use `hit` for successful reuse and `pfx` to identify host/disk restores: a valid P9 hit can have `pfx=0`.

6. **No apparent DFlash2 configuration or ordering break from these flags.** Both launch modes retain the same `Q27_BATCH=0` environment ([launch_q27_38.sh:17]); the server explicitly enforces that requirement ([server.cu:541]). Prefill waits on the previous fold event before restoring/resetting target state ([engine.cuh:4843]); the fold producer records it after flushing ([engine.cuh:4740]). `reset()` does not reset the drafter ring; `d2_prefill_align` separately retains rows below `min(token_LCP, base)` ([engine.cuh:2938]).

   **Rewrite plan lines 84–90:** host/disk restores lack stored drafter rows, but a recomputed suffix of at least 2048 tokens rebuilds the complete seed window; only shorter suffixes risk a shallow ring ([engine.cuh:2956], [engine.cuh:5060]). Raw 2048-row tap storage is **200 MiB**, not ~40 MB: five taps × 5120 floats per row ([dflash2.h:24]). Add RAM/disk restore validation with both short and long suffixes.

7. **Bitwise g64 equivalence across tile shapes is achievable; the 128-K stage does not impose a pairwise fp32 reduction.** Each stage processes `gg=0`, then `gg=1`; each group creates a fresh int32 accumulator, chains two K=32 MMAs, and immediately updates the persistent fp32 accumulator ([prefill.cu:442], [prefill.cu:474]). The final epilogue merely stores it ([prefill.cu:488]).

   **Tighten plan lines 123–126 and 140–143:** preserve increasing global g64-group order, `(wscale*xscale)` multiplication/rounding, and FMA contraction behavior. Replace “int32 accumulate per stage” with **“separate int32 accumulation per 64-group.”** Summing the two differently scaled groups into one stage accumulator would be wrong, not merely non-bitwise.

8. **The stated current kernel shape is stale.** On sm_120, saturated large-T g64 dispatch defaults to `k_gemm_mma_ntx<…,96>`: **MR=128, NT=96**, with two row minitiles per warp ([prefill.cu:502], [prefill.cu:874]). Activation `ldmatrix` is already enabled by default even in `k_gemm_mma_T` ([prefill.cu:36]).

   **Replace plan lines 103–118 and evidence lines 191–195.** Benchmark against live dispatch; use the MR=64 unsplit kernel as an explicit numerical reference. The microbench already measures live dispatch ([microbench_mxf4.cu:1133]). Keep small-T/split-K dispatch explicit: automatic splitting also requires g64 and sufficient scratch, while forced mode bypasses the occupancy threshold ([prefill.cu:735], [prefill.cu:797]).

9. **Phase 2’s commands and gates need repairs before hand-off.**
   - Add `--ctx 27648` to the 25,600-token command at plan lines 153–156. CLI context defaults to **2048** ([engine.cu:18]). Preserve `--ctx 133120` for the referenced 131072-token logit run.
   - Label `Q27_PF_XG=32 --pf …` as a legacy-path regression gate. Default g64 mismatch explicitly exits successfully, so that command does not enforce new-kernel correctness ([engine.cu:1374]).
   - `tools/ninv_test.cu` tests decode/verify families, not `gemm_q4_T` ([ninv_test.cu:22]). Add direct old/new prefill-GEMM comparisons across tile boundaries and partial tiles.
   - Specify numerical thresholds and an agentic long-context quality battery for the non-bitwise fallback. Use separate cache roots per numerical variant: cache compatibility omits kernel/numerics settings ([prefix_cache.h:57]).

10. **Phase 0’s instrumentation currently discards necessary evidence.** Plan line 53 drops `[sysblk]`, `[d2]`, and `prefix-cache:` messages—including write failures and eviction. Root/budget boot logging is `prefix-cache:`, not `[pfx]` ([server.cu:878]). Preserve the full invocation journal, then derive filtered views. The old restore/writer-join problem is already addressed by separate buffers ([engine.cuh:4108]); lowering step to 4096 increases save frequency and reduces recomputation, not import latency.

   **Replace “identical traffic” with “same harness/settings.”** `run.sh` launches fresh Claude sessions using a default `:latest` image ([run.sh:14], [run.sh:61]); record the image identity and use captured requests for deterministic eviction checks. `high` correctly maps to q27 xhigh ([api_common.h:379]). `lanes_agg.py` also needs a new grouping to compare restored versus same-conversation turns; its current groups are completion-length buckets ([lanes_agg.py:19]).

11. **Two final plan corrections:** Phase 3’s g32 guard must preserve `Q27_PREFILL=dp4a`, which consumes g32 buffers regardless of the g64 preference ([prefill.cu:900]). Skipping `mtp_warm_T` under DFlash2 also changes persisted MTP contents; isolate/version those entries before ladder reuse ([engine.cuh:4101]). Finally, qualify the outcome table as conditional estimates: a 2× improvement in a 55–60% GEMM share yields approximately 1.38–1.43× cold-prefill speedup, and does not establish the plan’s claim about explaining ninfer’s whole 2.2× advantage.
