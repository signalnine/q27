# q27 vs ninfer on live Claude Code, Qwen3.8-27B (2026-09-06)

Head-to-head on real agentic traffic: q27's shipping MTP ladder vs ninfer's
current best (their DFlash2 drafter), both serving Qwen3.8-27B, both driven by
Claude Code through the same tapproxy (client-observed decode t/s, the
`output_config`/`thinking` normalization applied to both), same 12 pinned
SWE-bench_Verified instances, both --no-thinking.

- **q38** = q27-server + qwen38-27b-mtp.q27, fp8 KV, MTP ladder (harness q27
  commit 746034a).
- **nvfp4m** = ninfer master (604bdc5) + their released Qwen3.8-27B NVFP4
  artifact + `--spec dflash2 --draft-tokens 7`.

## Result: ninfer +27% on decode t/s

Per-request decode t/s from the tap, median over substantial requests
(out_tok >= 64), which is robust to the ~12x difference in tokens generated:

| leg | decode t/s (median) | p25 | p75 | nonempty | gold | quit@1 |
|---|--:|--:|--:|--:|--:|--:|
| q27 3.8 ladder    | 224 | 205 | 259 | 7/12  | 6/12  | 5 |
| ninfer 3.8 dflash2 | 285 | 234 | 351 | 12/12 | 10/12 | 0 |

ninfer decodes ~27% faster on this traffic, AND completed more cleanly
(12/12 vs 7/12 nonempty). q27's aggregate token count was 12x higher (252K vs
20K): five q27 sessions quit after one turn and one pylint instance looped for
701 s, so the aggregate decode t/s (dtok/summed-windows) is not comparable --
the per-request median is. The quit@1 / loop behavior is likely tied to the
no-think agentic config and is a separate issue from decode speed.

## Where this leaves us

The +27% is the DFlash2 drafter -- the same drafter we integrated into q27
(see docs/plans/2026-09-06-dflash2-integration.md and bench/dflash2/). On
single-turn CLI benchmarks our DFlash2 wins, but on live agentic serving it
LOSES to our own ladder because the drafter's context ring cold-starts every
turn. ninfer's DFlash2 does not pay that cost -- their agent-prefix-reuse keeps
the drafter warm across turns. So this gap and our own live-CC regression have
the same root cause and the same fix: warm the drafter ring from the prefill
(prefill tap capture). Until that lands, ninfer's 3.8 decode leads q27's by
~27% on real Claude Code traffic.

Raw: agentic.{q38,nvfp4m}.jsonl (per-instance), tap.{q38,nvfp4m}.jsonl
(per-request timing). Harness: bench/crossengine/harness/, leg `nvfp4m` added
to legs.sh.
