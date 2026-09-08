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

## Results

(appended by the session that ran the campaign)
