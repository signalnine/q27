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
