# Agentic re-bench 2026-09-10: v0.11.3 vs its control vs ninfer

Same harness, instances and effort pin as the 09-09 campaign
(bench/crossengine/agentic-2026-09-09/): Claude Code 2.1.x in the
thunderdome image driving 12 pinned SWE-bench instances one at a time,
CLAUDE_CODE_EFFORT_LEVEL=medium, card sampler (T 1.0, top-p 0.95, top-k
20, min-p 0.05). `campaign.sh` in agentic-2026-09-07 with four new legs,
run in this order:

- `q27v0113` -- the production recipe (DFlash2 Q8 pack, K=7, batching off,
  prefix-cache tiers on a fresh tmpfs root) on the v0.11.3 binary
  (818990a, md5 0e48d741).
- `ninferd2b` -- ninfer DFlash2 k=7 on the NVFP4 artifact, unchanged binary
  and args from 09-09, re-measured the same day.
- `q27pre0113` -- the same recipe on the kept pre-v0.11.3 binary (bd81f73,
  md5 5870dff0): identical engine minus PR #43, the same-day control.
- `q27v0113b` -- the v0.11.3 leg again, after the first three showed the
  tmpfs confound (read 4): production's cache cleared first
  (`CLEAR_PROD_PFX=1`) so the leg had its full 40 GB budget; zero failed
  writes, 32 persists, 31 restores.

All three q27 legs loaded wsum b743d26b1f0562a9. ninfer rejected only Claude
Code's two thinking-disabled title requests, as on 09-09.

| leg | decode agg / median | tok/round | prefix reuse | wall/inst | turns/inst | out tok/inst | nonempty | gold |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **q27 v0.11.3, rerun (healthy cache)** | **218.0 / 232.2 t/s** | **4.052** | **95.7%** | **98 s** | 25.2 | 15.3K | 12/12 | 10/12 |
| q27 v0.11.3, first run | 221.8 / 233.2 t/s | 4.101 | 92.8%* | 108 s* | 23.8 | 15.2K | 12/12 | 10/12 |
| q27 before PR #43 | 209.7 / 223.1 t/s | 3.978 | 90.0%* | 123 s* | 23.2 | 16.8K | 12/12 | 12/12 |
| ninfer DFlash2 | 218.2 / 240.5 t/s | 4.113 | 96.1% | 24 s | 10.8 | 4.0K | 12/12 | 11/12 |
| *09-09 q27 (v0.11.0)* | *207.2 / 219.4* | *3.887* | *96.9%* | *108 s* | *25.0* | *18.2K* | *12/12* | *9/12* |
| *09-09 ninfer* | *220.6 / 240.1* | *4.188* | *96.8%* | *36 s* | *15.0* | *6.2K* | *12/12* | *11/12* |

\* confounded: the legs' tmpfs cache root filled and prefix-cache writes
failed (read 4). Decode columns are unaffected; the rerun row is the
valid reuse and wall.

## Reads

1. **Decode is at parity.** v0.11.3 decodes 218.0 and 221.8 t/s aggregate
   in two runs against ninfer's 218.2 the same day, tokens per round
   4.05-4.10 against 4.11; ninfer's median is higher (240.5 vs 232-233).
   The aggregates are over different request mixes -- q27 decoded ~190K
   tokens over ~310 requests, ninfer 48K over 126 -- so parity, not a lead,
   is the claim.
2. **PR #43 holds up on Claude Code traffic.** Against the same-day control
   the sampler-order fix plus the sorted small-top-k nucleus is +4.0% and
   +5.8% aggregate decode across the two runs, +4% median, +2-3% tokens per
   round -- in line with the +5.8% Codex measured on the 15-request replay.
   The old kernel's smaller nucleus rejected drafts the target law should
   have accepted.
3. **Wall is still trajectory length.** With a healthy cache q27 takes 98 s
   per instance against ninfer's 24 s, at 25 turns and 15K output tokens
   against 11 and 4K (09-09: 25/18K vs 15/6K), unchanged in kind:
   per-turn reasoning on an identical prompt puts q27 on the llama.cpp Q8_0
   reference and ninfer's NVFP4 arm 1.5x under it
   (agentic-2026-09-09-echo).
4. **The first v0.11.3 run's and the control's prefix reuse and wall are
   CONFOUNDED; the rerun row is the valid one.** /dev/shm (a 62 GB tmpfs) held production's own 38 GB prefix
   cache plus 5.3 GB of stale scratch roots, so each leg's fresh cache root
   had about 19 GB. It filled at 14:52:57, and from then on the disk tier's
   writes failed ("prefix-cache: write failed"): 26 of the v0.11.3 leg's 41
   persists and all 60 of the control's. The reuse drop to 90-93% and the
   restores that fell back to the shared 23,552-token system entry after a
   side request are those failed writes -- every deeper entry of the
   conversation examined (45490, 50546, 51596, 54859, 59571) is on the
   failed list. An earlier draft of this readout blamed Claude Code's
   thinking blocks for it; that is retracted, there is no evidence of a
   reuse regression. Decode rate and tokens per round are measured over
   decode only and stand. What does hold from the digging: on every
   consecutive same-conversation pair, here and on 09-09, the warm turn
   reuses the previous prompt minus its 5-token generation prompt and
   re-prefills the previous turn's own output -- by design, the snapshot
   sits at the prompt's stable prefix. campaign.sh now refuses a q27 leg
   whose cache root cannot get its budget and deletes the root after the
   leg. **The rerun with a healthy cache measures 95.7% reuse** against
   ninfer's 96.1% and 09-09's 96.9%, and 98 s per instance: no reuse
   regression.

Files: `results.*.jsonl` and `*.log` (harness), `q27*.req.txt` (the q27
legs' [req] lines, rerun included), `summarize.py` (the 09-09 script).
Journals and the ninfer request log stay local. n=1 per instance.
