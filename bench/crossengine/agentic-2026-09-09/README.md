# Agentic campaign 2026-09-09: the v0.11.0 standing (q27 production config vs ninfer DFlash2)

Same harness as 09-07/08 (bench/crossengine/agentic-2026-09-07/campaign.sh,
12 pinned SWE-bench_Verified instances driven through Claude Code, one pass
per leg, sequential legs, effort pinned to medium -- the only level both
engines render; production serves xhigh). Legs: `q27prod` = the production
recipe of tools/launch_q27_38.sh d2-pfx (DFlash2 Q8 pack, K=7, MMA verify,
prefix-cache tiers on a FRESH tmpfs root, so the run pays its own bootstrap
turns) on master c010961; `ninferd2` = ninfer-serve `--spec dflash2
--draft-tokens 7` on the NVFP4 release artifact, unchanged since 09-07.

## Results (23:22 finish)

| leg | decode t/s agg | median | tok/round | prefix reuse | wall/inst | turns/inst | out tok/inst | nonempty | gold |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| q27prod (production, medium) | 207.2 | 219.4 | 3.887 | 96.9% | 108 s | 25.0 | 18243 | 12/12 | 9/12 |
| ninferd2 (DFlash2 k=7, medium) | 220.6 | 240.1 | 4.188 | 96.8% | 36 s | 15.0 | 6246 | 12/12 | 11/12 |

q27prod detail (lanes_agg.py over swebench_q27prod.journal): 305 requests,
prefill wall 126 s, decode 1093 s; 4 cold >= 20K prompts (the bootstrap of
a fresh root), 285 requests restored or hit, 96.9% of prompt tokens served
from cache; P(accept >= lane j) 0.811 0.623 0.471 0.358 0.267 0.206 0.157;
long thinking turns are 68% of decoded tokens at 3.83 tok/round, short
tool-call turns 4.30. No parser recovery fired on this run (0 pass-through
/ mode 21-22 lines) and no instance ended on its first turn.

## Reads

1. Decode is within 6% of ninfer's DFlash2 arm (207 vs 221 aggregate, 219
   vs 240 median), the same standing as 09-08 (216 vs 219) within the
   +-4% session drift the unchanged ninfer binary has shown across runs.
   tok/round 3.89 vs 4.19.
2. Both engines now reuse ~97% of prompt tokens. ninfer's agent-prefix fix
   (NINFER-REBENCH.md) removed the 0%-reuse handicap that made q27 the wall
   winner on 08-17; the ordering on wall has flipped.
3. Wall per instance is 108 s vs 36 s and the decode rate does not explain
   it: q27's sessions ran 25 turns and 18.2K output tokens per instance
   against 15 turns and 6.2K on ninfer. The same pattern held on 09-07/08
   (q27d2q8 16.7 turns / 10.3K vs ninferd2 12.9 / 5.0K), so it is not one
   run's trajectory luck. Per instance the gap is concentrated: xarray-4094
   74 turns / 72K vs 20 / 6K, pytest-10081 25 / 21K vs 5 / 1K, pylint-4970
   50 / 44K vs 40 / 21K. Same model family and sampler chain, different
   quant tier (Q4_G64 vs NVFP4), a different rendering of "medium" effort
   (q27 renders it in the system prompt; ninfer passes the name to the
   template), and different tool-call parsers. Which of those makes
   Claude Code loop longer on q27 is the open question this campaign
   leaves, and the first thing to run the request replay at.
   *Answered the same day in
   [agentic-2026-09-09-echo/](../agentic-2026-09-09-echo/README.md):
   none of the three. q27's per-turn reasoning equals a Q8_0 reference
   served by llama.cpp; ninfer's NVFP4 arm reasons 1.5x shorter than that
   reference, and the harness's n=1 turn counts swing by +-3 on a same-day
   control. Two q27 defects were fixed on the way (model-name echo so
   Claude Code keeps thinking history; the compact `<tools>` rendering)
   without moving the gap.*
4. Task outcomes: 12/12 non-empty diffs on both; gold-file hits 9/12 vs
   11/12, single sampled trajectories, not a signal on their own but read
   next to (3).

Caveats: n=1 per instance; the q27 leg ran without Q27_PRINT_WSUM (added to
the campaign env afterwards), so the load digest is unverified -- one load,
~1% prior of the 5090's pageable-DMA flip; effort medium is not
production's xhigh (the 09-08 README has the xhigh production numbers:
212.8 agg / 229.2 med, 3.964 tok/round, prefill wall 110 s).

Files: q27prod.log, ninferd2.log (run.sh output), results.*.jsonl,
swebench_q27prod.journal (+ q27prod.req.txt = its [req] lines),
swebench_ninferd2.journal, ninferd2.reqlog.jsonl (ninfer's own request
log), summarize.py, lanes_agg.py (copied from agentic-2026-09-08).
