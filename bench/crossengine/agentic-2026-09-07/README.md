# Agentic cross-engine campaign, 2026-09-07 (Claude Code on SWE-bench)

The question: after the DFlash2 work of 2026-09-07 (sampled selector walk,
ring retention, last-token row, bitonic top-16), where does q27 stand vs
ninfer on REAL agentic coding traffic -- multi-turn, tool-heavy, prefix-cache
warm turns, thinking on -- rather than on the seeded single-turn instrument?

## Instrument

bench/swebench/run.sh: Claude Code 2.1.170 (thunderdome/claude-code image,
`claude -p --dangerously-skip-permissions`, 700 s cap per instance) against
the engine's Anthropic Messages API on host port 8081, over the 12 pinned
SWE-bench Verified instances in bench/swebench/manifest.json. Per instance:
wall, turns, output tokens, nonempty diff, gold file edited. Engine metrics
over the run window: aggregate decode t/s (tokens / decode time, requests
>= 8 tokens), median per-request t/s, tok/round, prefix-cache reuse. q27 and
llama parse their journals; ninfer parses its --request-log-jsonl (new).

campaign.sh runs the legs back to back as transient user units, both vox
transcribers stopped (host jitter), production relaunched at the end.
summarize.py prints the table.

## Legs

| leg | engine | config |
|---|---|---|
| q27lad | q27 build 0cc4720 | production: fp8 KV, ladder auto4..7 + suffix, batch mode 1 slot |
| q27d2q4 | q27 | + Q27_BATCH=0, DFlash2 K=7 sampled walk, Q4-g64 serving pack (1.2 GB) |
| q27d2q8 | q27 | + DFlash2 Q8 serving pack (2.08 GB), reserve 3 GB |
| ninferd2 | ninfer master 487f897, release Qwen3.8-27B nvfp4 | --spec dflash2 --draft-tokens 7, int8 KV 131072, max-context 131072 |
| ninfermtp | ninfer | --spec mtp --draft-tokens 3 (content control) |

## Fairness controls

- Same harness, same instances, same Claude Code, same prompts; single stream.
- Sampler: temperature 1.0 (Claude Code sends it), top-p 0.95, top-k 20,
  min-p 0.05 pinned server-side on both (ninfer's thinking default has
  min-p 0, so it is passed explicitly). Thinking on, no budget, on both.
- REASONING EFFORT = MEDIUM ON BOTH ENGINES. Claude Code 2.1.170 sends
  output_config.effort (low | medium | high); the Qwen3.8 template exposes
  low | medium | xhigh, ninfer passes the name through and 400s on 'high',
  q27 ignores the field and renders its boot default (xhigh). Medium is the
  only effort reachable on both, so run.sh pins CLAUDE_CODE_EFFORT_LEVEL=
  medium and the q27 legs set Q27_REASONING_EFFORT=medium. This is NOT
  production's xhigh: thinking is shorter, acceptance higher, and the
  numbers are comparable across legs here but not to earlier xhigh runs.
  Follow-up: teach q27's /v1/messages to honour output_config.effort so the
  client's setting is respected on both engines.
- NINFER PARSER PATCHED LOCALLY (ninfer-undeclared-tools.patch, one line):
  Claude Code 2.1.170 declares 28 tools but defers Grep/Glob, and its prompt
  still names them, so Qwen emits `<function=Grep>` next to a valid Bash
  call. ninfer master (487f897) rejects the WHOLE tool-call batch as plain
  text when any name is undeclared (fallback_reason undeclared_tool), so the
  agent's turn ends after one request; q27 surfaces the calls, the client
  answers the unknown one with an error result, and the agent continues.
  The patch sets enforce_declared_names = false so ninfer behaves like q27
  here. Everything else in their parser (argument typing by schema) is
  untouched. The smoke that found it: one instance, ninfer 1 request / 6 s
  / no diff vs q27 17 requests / 65 s / gold file edited.
- Quants differ (q27 q4s/Q8 mix vs nvfp4), so the agent's trajectories
  diverge after the first token: per-instance wall and turns are context,
  not the engine metric. Decode t/s and tok/round over the run window are.
- The MTP3 control separates content mix from the DFlash2 arm, as in the
  09-07 rebench.

## Results (2026-09-07 21:23-22:40, one pass over the 12 instances per leg)

| leg | decode t/s agg | median | tok/round | prefix reuse | wall/inst | turns | out tok | nonempty | gold |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| q27lad (ladder+suffix, production) | 162.1 | 174.5 | 3.132 | 91.2% | 131 s | 12.7 | 111520 | 10/12 | 9/12 |
| q27d2q4 (DFlash2 Q4 pack) | 173.3 | 186.3 | 3.941 | 91.6% | 79 s | 17.5 | 119002 | 11/12 | 9/12 |
| q27d2q8 (DFlash2 Q8 pack) | 176.3 | 185.9 | 4.033 | 90.6% | 86 s | 17.2 | 124522 | 10/12 | 9/12 |
| ninferd2 (DFlash2 k=7) | 228.5 | 252.9 | 4.276 | 96.6% | 28 s | 13.8 | 54875 | 12/12 | 10/12 |
| ninfermtp (MTP3 control) | 148.3 | 165.4 | 2.898 | 94.0% | 54 s | 13.6 | 78071 | 12/12 | 10/12 |

Per-lane acceptance on this traffic (P(accept >= lane j), DFlash2 legs):

    q27d2q8   0.820 0.641 0.494 0.380 0.294 0.231 0.182   cond 0.820 0.782 0.770 0.770 0.772 0.787 0.786
    ninferd2  0.830 0.665 0.526 0.419 0.338 0.274 0.228   cond 0.830 0.801 0.791 0.796 0.807 0.812 0.832

Reads:

1. DFlash2 on q27 is now an agentic serving WIN: +7% (Q4) / +9% (Q8) decode
   t/s over the production ladder, tok/round 3.13 -> 3.94 / 4.03. On 09-06
   the same harness had it at -7% (greedy-only drafter, ring cold every turn,
   last prompt row missing). Q8 is the better serving pack here too.
2. Engine vs engine at the same drafter class: q27 ladder 162.1 vs ninfer
   MTP3 148.3 = q27 +9% t/s (tok/round 3.13 vs 2.90), consistent with the
   seeded rebench (+13%).
3. ninfer's DFlash2 arm leads q27's by +30% t/s (228.5 vs 176.3) on +6%
   tok/round (4.28 vs 4.03): the remaining gap is the round wall, as the
   seeded instrument said (their ~18 vs our ~22 ms). Lane 1 is at parity
   (0.83 vs 0.82); lanes 2-7 trail by 2-3 points each.
4. CONFOUND, in ninfer's favour: with the same rendered prompt (both engines
   emit no effort line at medium) the model thinks 2.5x LESS per assistant
   message on ninfer (327 chars/msg vs 572-803 on q27, from Claude Code's
   own logs), takes fewer turns (13.8 vs 17) and emits half the output
   tokens. Less thinking = easier-to-draft traffic and shorter decode
   stretches; nvfp4 + int8 KV vs q4s + fp8 KV is the only difference in
   what the model sees. Trajectory-level metrics (wall, turns, out tokens)
   are therefore not comparable across engines; decode t/s and tok/round
   are engine metrics on each engine's own trajectories.
5. Prefix reuse is real on both (91% q27, 94-97% ninfer -- their
   agent-prefix-reuse fix works). The q27 suffix drafter fired 0 times on
   the ladder leg: on this traffic it contributes nothing.
6. Quality signals are flat across legs (gold file 9-10/12, nonempty
   10-12/12), as expected for one base model.

Standing caveats: one pass (the 07-17 seal used 3 x 12); reasoning effort
medium on both (not production's xhigh); ninfer runs with the one-line
parser patch above. Reproduce: `systemd-run --user --unit agentic-campaign
bash bench/crossengine/agentic-2026-09-07/campaign.sh`, then summarize.py.
