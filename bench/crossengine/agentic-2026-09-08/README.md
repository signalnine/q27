# Agentic cross-engine campaign, 2026-09-08 rerun (build d4947fd)

Same harness, legs, instances, effort pin (medium on both), ninfer parser
patch and fairness controls as ../agentic-2026-09-07/README.md; the only
change is the q27 build: ring retention + last-token row, flash-decoding
drafter attention, side-stream fold, the bitwise verify batch (lane-packed
head norms, fused alpha/beta gemv, gdn_delta_all tile split, fused
norm+quantize) and the d2 verify on the deterministic MMA path (vgemm,
default). ninfer legs are unchanged controls rerun in the same session.

## Results (09:31 finish, one pass over the 12 instances per leg)

| leg | decode t/s agg | median | tok/round | prefix reuse | wall/inst | turns | out tok | nonempty | gold |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| q27lad (ladder+suffix, production) | 162.8 | 179.7 | 3.079 | 90.0% | 111 s | 16.7 | 170513 | 11/12 | 10/12 |
| q27d2q4 (DFlash2 Q4 pack) | 215.1 | 228.5 | 3.911 | 92.3% | 112 s | 23.8 | 191283 | 12/12 | 11/12 |
| q27d2q8 (DFlash2 Q8 pack) | 216.2 | 228.3 | 4.034 | 89.2% | 75 s | 16.7 | 124175 | 9/12 | 7/12 |
| ninferd2 (DFlash2 k=7) | 218.5 | 246.1 | 4.105 | 97.2% | 28 s | 12.9 | 59957 | 12/12 | 10/12 |
| ninfermtp (MTP3 control) | 150.2 | 165.8 | 2.905 | 95.6% | 38 s | 14.0 | 54976 | 12/12 | 10/12 |

vs 2026-09-07 (same harness, build 0cc4720 era): q27d2q4 173.3 -> 215.1
(+24%), q27d2q8 176.3 -> 216.2 (+23%), q27lad 162.1 -> 162.8, ninferd2
228.5 -> 218.5 (-4%, unchanged binary), ninfermtp 148.3 -> 150.2 (+1%).

Reads:

1. q27's DFlash2 arm is now at parity with ninfer's on agentic traffic in
   the same session (216 vs 219 t/s, -1%), from -23% the night before; the
   two unchanged ninfer legs bound the session drift at about +-4%.
2. DFlash2 on q27 is +33% over the production ladder (216 vs 163).
3. tok/round: q27 Q8 4.03 vs ninfer 4.10 (-2%); the thinking-length
   confound from 09-07 persists (ninfer's trajectories carry far fewer
   output tokens per turn), so ninfer's traffic remains the easier draft.
4. Quality signals are one sampled pass each: gold-file hits range 7-11 of
   12 across the q27 legs with no engine-level pattern (the Q8 leg drew a
   7; the Q4 leg an 11). Not a differentiator at n=1.
5. The ladder leg's flat t/s despite the bitwise verify batch: its
   trajectories differ a lot between the runs (turns 12.7 -> 16.7, output
   tokens 111K -> 171K), and per-request t/s in agentic traffic is dominated
   by the thinking/echo mix; the seeded instrument saw +4% for the same
   changes.

Standing caveats as before: one pass, effort medium on both engines, ninfer
patched for undeclared tool names.

## Production at xhigh: DFlash2 vs ladder on the same instances (12:14 finish)

Production q27-38 on the DFlash2 config (Q8 pack) vs the ladder config, both
at Claude Code's default effort (q27 renders xhigh), 12 instances each.
Aggregates via lanes_agg.py over the [req] journals (prodd2-xhigh.req.txt,
prodlad-xhigh.req.txt):

| config | reqs | dec tok | t/s agg | t/s med | tok/round | round ms | reuse | nonempty | gold |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DFlash2 (Q8) | 299 | 329672 | 201.0 | 218.3 | 3.738 | 18.60 | 93.3% | 9/12 | 8/12 |
| ladder+suffix | 271 | 212254 | 164.2 | 177.2 | 3.134 | 19.08 | 92.4% | 11/12 | 10/12 |

DFlash2 +22% agg / +23% median at xhigh (vs +33% at the medium pin): xhigh
makes 80% of decoded tokens long thinking turns (3.78 vs 3.12 tok/round
there; 4.36 vs 3.78 on short tool-call turns). Rounds within 0.5 ms; the
lead is acceptance. Quality is one sampled pass per config on different
trajectories (the DFlash2 run had one 700 s-capped instance): not a signal.
Note: run.sh's own decode line read 0 for these runs -- the labels do not
start with q27, so its telemetry fell to the llama parser; use
SWEBENCH_TELEMETRY=q27 for such labels.

## Prefill attack phase 0: the shipped cache tiers ON (prodpfx, 14:36 finish)

docs/plans/2026-09-08-prefill-attack.md phase 0. Same DFlash2 production
config plus `tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1` (P16 disk tier
on tmpfs /dev/shm/q27-pfx, 40 GB, min 4096, max 65536, step 8192, RAM tier
off), fresh cache root, same 12 instances at Claude Code's default effort
(xhigh), vox ON as in the baseline. Files: prodpfx-xhigh.journal (full
invocation), prodpfx-xhigh.req.txt, pf_restore_agg.py (joins [gen] pfx with
[req]: cold / vram / restore classes).

Controlled eviction case first (bench/ladder/pfx_evict_probe.py, /v1/messages, 6.6K-token
system block): conversation A at 28,484 tokens persisted the system cut
(L=6144, 0.37 GB, export 168 ms) and the stable boundary (L=28479, 1.15 GB,
53 ms); A at 48,857 hit P8 at 28479 and persisted L=48852 (1.86 GB, 80 ms); a
369-token request from another conversation; A's next turn restored L=48852
from tmpfs (read 164 + import 90 ms) and re-prefilled 44 tokens, pf_ms 342
(a full miss is ~14 s); a NEW conversation with the same system block
restored the L=6144 entry (28 + 16 ms), pf 721. A first probe that overshot
to 69.7K tokens showed the max-tokens trap live: the 69.7K boundary was
silently never persisted and the returning turn re-prefilled 29K.

12 instances, miss anatomy (pf_misses.py) against prodd2-xhigh:

| | prodd2 (tiers off) | prodpfx (tiers on) |
|---|--:|--:|
| requests / conversations | 299 / 30 | 223 / 26 |
| prefill wall | 255 s | 154 s |
| first turns: n, wall, with a system-block hit | 30, 85.5 s (33%), 0 | 26, 92.1 s (60%), 0 |
| returning after another conversation: hits / full misses | 0 / 6 (79.9 s, 31%) | 2 / 0 |
| same-conversation turns: hits / misses | 263 / 0 | 195 / 0 |
| restores | -- | 2 x L=44229, 1.70 GB, 214-217 ms |
| persists (export ms med / max) | -- | 24 (74 / 217) |
| read FAILED / evictions | -- | 0 / 0 (29 GB of 40 used) |
| decode t/s agg / med, tok/round, round ms | 201.0 / 218.3, 3.738, 18.60 | 224.9 / 215.7, 4.284, 19.05 |
| instances nonempty / gold | 9 / 8 | 9 / 7 |

Reads:

1. The eviction class is closed. Every returning turn after an interleaved
   side request restored (2 of 2, 214-217 ms for 1.70 GB) instead of
   re-prefilling 28-48K (8.5-15.9 s each, 6 of them in the baseline). The
   traffic is fresh Claude Code sessions, so the class occurred less often
   this run (2 vs 6); compare classes, not totals.
2. First turns still never hit (0 of 26) and are now 60% of the prefill
   wall. Root cause found from the entries themselves (the .q27pc files
   carry the token vectors): all five sessions checked share EXACTLY 22460
   tokens and diverge inside Claude Code's gitStatus section ("Recent
   commits:" followed by per-repo hashes). The P16b cut sits at the last
   chunk boundary <= sys_len (22528), 68 tokens past the divergence, so no
   session can ever hit another's entry and every first turn writes its
   own 0.94 GB entry (6 identical-L entries in the root). Fix: cut at the
   longest prefix an indexed entry shares with the prompt (PrefixCache::
   shared_prefix; cut lands at 21504 here) -- see the next section.
3. Restore cost is what the 07-24 numbers said: ~0.2 s from tmpfs; no
   alloc stalls, no read failures. Persist exports: 20 of 24 under 130 ms,
   four 0.94 GB system-cut exports at 202-217 ms (same size usually takes
   46 ms; likely contention with the previous entry's tmpfs write).
4. Decode moved +12% agg on different trajectories (tok/round 4.28 vs 3.74,
   P(accept>=1) 0.827 vs 0.800): traffic, not the cache. Round wall 19.05 vs
   18.60 ms is inside the run-to-run band seen all week with vox on.
   Quality at n=1 (gold 7 vs 8) is not a signal.
5. The restored-turn drafter question (shallow ring after a restore) has
   n=2 here (276 decoded tokens) -- no read.

## Phase 0 rerun with the P16b shared cut (prodpfx2, 14:41 finish)

Same launch, fresh cache root, the engine now cutting the system-block entry
at the longest prefix an indexed entry shares with the prompt
(PrefixCache::shared_prefix; engine.cuh generate()). Files: prodpfx2-xhigh.
journal / .req.txt. Live gate first (bench/ladder/pfx_shared_probe.py, three
sessions with a 7.1K shared body + per-session gitStatus tail, a 375-token
foreign request between them so each first turn is cold): S1 cut at 7168
(nothing shared), S2 "shares 7084 with an indexed entry -> cut at 6144",
S3 restored L=6144 (33 + 16 ms) and re-prefilled 1083 tokens, pf_ms 435 vs
2025 cold. Without the foreign request the P9 checkpoint ring serves S2/S3
at 4096 and the cut logic never runs -- the ring is what a side request
clears in production.

| | prodd2 (tiers off) | prodpfx (tiers on) | prodpfx2 (tiers on + shared cut) |
|---|--:|--:|--:|
| requests / conversations | 299 / 30 | 223 / 26 | 228 / 30 |
| prefill wall | 255 s | 154 s | 110 s |
| first turns: n, wall, with a hit | 30, 85.5 s, 0 | 26, 92.1 s, 0 | 30, 44.3 s, 10 |
| first turns >= 20K: cold / restored | 12 / 0 | 12 / 0 | 3 / 12 |
| returning after another conversation: hits / full misses | 0 / 6 (79.9 s) | 2 / 0 | 6 / 0 |
| restores (n, read+import ms med / max) | -- | 2, 214 / 217 | 16, 124 / 212 |
| persists (n, export ms med / max) | -- | 24, 74 / 217 | 16, 68 / 209 |
| read FAILED / evictions / root used | -- | 0 / 0 / 29 GB | 0 / 0 / 19 GB |
| prefix reuse | 93.3% | 94.5% | 96.2% |
| decode t/s agg / med, tok/round, round ms | 201.0 / 218.3, 3.738, 18.60 | 224.9 / 215.7, 4.284, 19.05 | 212.8 / 229.2, 3.964, 18.63 |
| harness wall, turns, output tokens | 2168 s, 221, 204K | 1373 s, 150, 138K | 840 s, 219, 133K |
| instances nonempty / gold | 9 / 8 | 9 / 7 | 9 / 8 |

Reads:

1. First turns now hit from the third session on. Journal sequence: session
   1 cut at 22528 (nothing indexed); session 2 "system block 22574 tokens,
   shares 22460 with an indexed entry -> cut at 21504" and persisted 0.91
   GB; sessions 3-15 with a system block restored L=21504 (79 + 45 ms) and
   re-prefilled 2.3-4.5K tokens: 1.0-1.7 s per first turn instead of
   7.0-8.4 s cold. The three cold first turns after the bootstrap carried no
   system block at all (no [sysblk] line: 27393, 13749 and 5476-token
   prompts, Claude Code side calls), so no system entry applies to them.
2. Both miss classes are closed on this traffic: 0 full misses among
   returning turns (6 restores of 29.7-44.2K conversation entries after
   side requests) and 12 of 15 system-block first turns restored (the other
   3 are the two bootstrap sessions plus one 27K prompt without a system
   block). Prefill wall 255 -> 110 s (-57%) with turn counts matched (221
   vs 219); the bootstrap costs two cold first turns per fresh root, once.
3. Restored turns draft normally: tok/round 3.88 (restore class, 16 turns)
   vs 3.96 (VRAM hits) vs 4.07 (cold), the same spread the completion-length
   mix explains; every restore here re-prefilled >= 2.3K tokens, which
   re-seeds the DFlash2 ring through the normal tap capture.
4. Round wall 18.63 ms = the baseline's 18.60: the 19.05 in prodpfx was
   run-to-run jitter, not the pinned staging buffers. Decode +6% agg on
   different trajectories (traffic). The harness wall halving is mostly
   fewer decoded tokens this pass (146K vs 330K), not the cache; the
   attributable part is the 145 s of prefill wall.
5. Persist exports: median 68 ms, four at 150-209 ms (0.9-2.4 GB); each is on
   the critical path once per boundary. Restore max 212 ms (2.4 GB entry).
6. tmpfs use 19 GB after 12 instances at a 40 GB budget; a day of sessions
   will cycle the LRU. Entries never encode kernel numerics -- a numerics
   change needs a fresh PFX_DIR (launch script comment).

Decision: `d2-pfx` (with the shared cut) is production. The remaining
prefill cost on this traffic is the two bootstrap cold prefills per fresh
root, the ~2.3-4.5K re-prefill after each system-block restore (the shared
prefix ends 956 tokens past the 21504 chunk boundary, plus the per-session
gitStatus tail and the first user message), and the step-gate remainder
after conversation restores (up to 8192 tokens + growth). Phase 2 (the GEMM)
is what makes the remaining cold prefills cheaper.
