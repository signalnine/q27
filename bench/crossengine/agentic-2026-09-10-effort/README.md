# Reasoning-length gap, 2026-09-10: effort, rendering, and a tokenizer bug

The question from the 09-09/09-10 campaigns: on the same 12 Claude Code
SWE-bench instances q27 thinks about twice as long per turn as ninfer and runs
twice the turns, so it pays 4x the wall at equal decode speed. Harness,
instances, card sampler (T 1.0, top-p 0.95, top-k 20, min-p 0.05) and effort
pin (medium unless a leg says low) are the 09-10 re-bench's
(bench/crossengine/agentic-2026-09-10/).

**Answer.** Most of the per-turn gap was an engine bug. q27's tokenizer never
matched four of the vocab's added tokens -- `<tool_call>`, `</tool_call>`,
`<tool_response>`, `</tool_response>` -- so every tool-bearing prompt showed
the model its tool-format instructions, its own past calls and every tool
result as spelled-out text ("<", "tool", "_call", ">") instead of the single
tokens it was trained on and emits itself. 0 of 34 recorded Claude Code
prompts tokenized like transformers' AutoTokenizer; after the fix, 34 of 34.
On the same traffic the fixed binary thinks 43% less per turn at medium
effort (2281 -> 1295 chars; ninfer 954), decodes 8% faster (4.27 vs 3.98
tok/round -- the drafter was trained on the right tokens too) and hits the
gold file on 11/12. It does NOT run fewer turns: 21-23 per instance against
ninfer's 11. What is left of the wall gap is trajectory length -- q27 still
reproduces and verifies more before it edits -- and on identical mid-session
states the llama.cpp Q8_0 reference behaves like q27, not like ninfer. A
replicate leg reproduces the fix (1387 chars/turn, 70 s, 10/12). Effort low
does not stack with it (85 s, one runaway session) and its earlier turn cut
on v0.11.3 is inside the harness noise.

## 1. Turn-0 probe: what makes ninfer terse, and whether effort moves q27

`probe_think.py` (agentic-2026-09-09-echo), one recorded turn-0 body (28
tools, the pytest-10081 task), 24 seeds per arm. Thinking chars, median with a
bootstrap 95% CI, Mann-Whitney against q27 at medium:

| arm | median [CI] | mean | p90 | vs q27 medium |
|---|--:|--:|--:|--:|
| q27 production recipe, medium | 291 [239, 337] | 351 | 548 | -- |
| q27, effort low (template's low line; +26 prompt tokens) | 382 [345, 423] | 405 | 521 | p=0.034 (longer) |
| ninfer NVFP4, int8 KV (DFlash2) | 234 [205, 266] | 240 | 306 | p=0.009 |
| ninfer NVFP4, bf16 KV (DFlash2) | 248 [232, 279] | 274 | 370 | p=0.10 |
| ninfer groupwise-int weights, int8 KV (no spec) | 248 [240, 301] | 286 | 379 | p=0.20 |
| llama.cpp Q8_0 (09-09) | 314 [269, 352] | 343 | 446 | p=0.76 |
| q27 fixed tokenizer, medium | 303 [240, 436] | 446 | 850 | p=0.70 |
| q27 fixed tokenizer, effort low | 344 [286, 386] | 341 | 444 | p=0.45 |

Neither ninfer's KV dtype, its weight format nor its drafter makes the
difference (all three arms 234-248; the int-weights arm ran without
speculation). ninfer renders effort exactly as q27 does (medium = no line,
`src/targets/qwen3_6/impl/frontend/chat_template.cpp`), and its sampler is
q27's post-PR #43 order (top-k, then top-p over the top-k mass, then min-p).
The low-effort line makes q27 think LONGER at turn 0 on this prompt. At turn
0 the gap is small (1.2x) anyway -- the session gap is mid-session. The
tokenizer fix does not move turn 0 either (303 vs 291): the turn-0 prompt
holds the tool tags only in the template's tool-format paragraph, a mid-session
prompt holds two per call and two per result.

## 2. Effort low on Claude Code: fewer turns, same per-turn reasoning

Legs `q27low` (production recipe, Claude Code at effort low) and `q27v0113c`
(the same-day medium control), both on the v0.11.3 binary with fresh cache
roots (0 failed writes, wsum b743d26b1f0562a9), request bodies recorded
(`REQBODY_LOG`, kept local).

| leg | dec t/s agg / med | tok/round | reuse | wall/inst | turns | out tok/inst | gold |
|---|--:|--:|--:|--:|--:|--:|--:|
| q27 v0.11.3, medium (control) | 213.9 / 231.9 | 3.977 | 95.2% | 106 s | 21.8 | 16.6K | 10/12 |
| q27 v0.11.3, effort low | 216.1 / 233.8 | 3.975 | 95.3% | 69 s | 18.1 | 11.2K | 10/12 |
| **q27 fixed tokenizer + 3.8 history, medium** | **231.5 / 242.9** | **4.265** | 95.7% | **75 s** | 22.9 | **11.2K** | **11/12** |
| q27 fixed, medium, replicate | 228.0 / 247.2 | 4.211 | 96.5% | 70 s | 22.6 | 11.7K | 10/12 |
| q27 fixed, effort low | 218.4 / 248.4 | 4.070 | 96.2% | 85 s | 24.2 | 13.5K | 11/12 |
| ninfer DFlash2 (09-10 re-bench) | 218.2 / 240.5 | 4.113 | 96.1% | 24 s | 10.8 | 4.0K | 11/12 |

From the transcripts (`turns_cmp.py`, `verify_cmp.py`, `edit_phase.py`):

| leg | turns | thinking K chars | per turn | turns >2K chars | tool calls | runs code | before 1st edit | after last edit |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| q27 medium (control) | 20.8 | 47.5 | 2281 | 26% | 20.8 | 5.8 | 14.4 | 3.8 |
| q27 effort low | 16.8 | 31.4 | 1864 | 21% | 17.1 | 4.9 | 11.2 | 3.9 |
| q27 fixed, medium | 21.2 | 27.5 | 1295 | 17% | 21.9 | 8.1 | 14.1 | 4.9 |
| q27 fixed, medium, replicate | 21.4 | 29.7 | 1387 | 19% | 21.6 | 8.0 | 13.6 | 5.2 |
| q27 fixed, effort low | 22.2 | 36.4 | 1637 | 19% | 23.2 | 4.6 | 17.7 | 3.8 |
| ninfer | 10.5 | 10.0 | 954 | 10% | 9.8 | 2.8 | 5.6 | 2.7 |

(per instance; "runs code" = Bash calls invoking python/pytest/pip.)

On the v0.11.3 binary effort low cut turns (18 vs 22), not per-turn
reasoning (1864 vs 2281 chars/turn, both far above ninfer). On the fixed
binary it does neither: 22.2 turns, 1637 chars/turn, 85 s -- one runaway
xarray-4094 session (80 turns, 52K tokens) dominates the wall, and without it
the leg looks like the medium ones. The v0.11.3 turn cut sits inside the
harness's +-3 turn noise and did not reproduce, so effort low is not a lever
worth recommending on this evidence. The fixed-binary medium legs replicate
each other: 1295 and 1387 chars/turn, 70-75 s, 10-11/12 gold (seven of the
twelve sessions repeat exactly between the two runs -- same turns, thinking
and output tokens, seed-0 determinism -- the other five diverged on external
nondeterminism).

The extra q27 turns sit BEFORE the first edit: on
pylint-4970 q27 tries to run pylint, finds astroid missing in the harness
container, builds a stub astroid and a harness, fetches pages, and only then
edits; ninfer reads the file, edits, re-reads, done. The harness containers
have no repo dependencies, so this verification mostly fails -- and the
gold-file proxy cannot reward it either way (no SWE-bench test images here).

## 3. Identical mid-session states: q27, ninfer and the reference agree on what to do

34 recorded turns from the control leg (tool-turn positions 2 and 5 of each
main session, `select_turns.py`), 4 seeds each, on every engine
(`replay_think.py` over /v1/messages; `raw_think.py` sends byte-identical
pre-rendered prompts to a raw completion endpoint, which takes every template
out of the comparison). Per-body mean thinking, geometric-mean ratio and
Wilcoxon signed-rank:

| arm (same 34 states) | think mean | vs q27 old tokenizer | vs q27 fixed tokenizer | next action: Bash / Read / none |
|---|--:|--:|--:|--:|
| q27 /v1/messages (v0.11.3) | 1820 | 0.93, p=0.26 | | 60% / 24% / 6% |
| q27 raw, untrimmed history, old tokenizer | 2052 | 1.00 | | 51% / 29% / 8% |
| q27 raw, template-trimmed history, old tokenizer | 2038 | 1.02, p=0.39 | | 55% / 27% / 7% |
| q27 raw, trimmed, **fixed tokenizer** | 2110 | 0.81, p=0.58 | 1.00 | 55% / 29% / 5% |
| llama.cpp Q8_0 raw, trimmed (HF-identical ids) | 1904 | 0.90, p=0.37 | 1.11, p=0.80 | 57% / 25% / 6% |
| ninfer /v1/messages | 1411 | 0.77, p=0.057 | 0.95, p=0.035 | 58% / 26% / 6% |

On a fixed state all engines choose the same kind of next step, and the
reference's thinking is indistinguishable from q27's. With correct token ids
q27 and ninfer sit within 5% per turn. The session-level gap therefore builds
up along the trajectory -- q27's own earlier turns, rendered the way it saw
them, steer the later ones.

## 4. Two prompt-fidelity bugs found on the way

Both commits are on master: `5c28eaf` (history rendering), `6084562`
(tokenizer).

**Tokenizer (the one that matters).** `src/tokenizer.cpp` matched CONTROL
tokens plus a hardcoded `<think>`/`</think>`; the Qwen3.6/3.8 vocab has six
USER_DEFINED added tokens and HF (and llama.cpp) match all of them in text.
Found when llama.cpp tokenized a q27-rendered prompt to 26544 tokens and q27
to 26578. Also fixed: the pretokenizer's `\s*[\r\n]+` stopped at the first
newline ("\n \n" is one HF token). Parity, q27 ids vs AutoTokenizer on the 34
prompts (26K-86K tokens): 0/34 -> 34/34. `test_tokenizer` gained an HF-parity
block (9/9; 3/9 on the old encoder). Every tool-bearing prompt tokenizes
differently now, so a deploy needs a fresh prefix-cache root. The bug is as
old as the tokenizer (2026-07-01) and the 3.6 vocab has the same six tokens
at the same ids: every q27 agentic measurement to date ran on it.

**History rendering (conformance, no measurable effect).** The 3.8 template
trims every content and reasoning_content, puts "\n\n" between text and the
first call, walks arguments in the client's key order and spaces non-string
values. q27 showed its own turns back as "...\n\n</think>\n\n\n\ntext",
Edit calls with new_string before old_string, and `[1,2]`. Fixed for the
XML (3.8) dialect; new llama.cpp-captured golden
`tools/golden/qwen38_history_request.*`. Same 34 states, trimmed vs
untrimmed: 1.02x (p=0.39).

## 5. Files

`results.*.jsonl`, `*.log` (harness), `summarize.py`, `turns_cmp.py`-style
tables from `verify_cmp.py` / `edit_phase.py` / `toolseq.py`,
`turns_from_bodies.py`, `select_turns.py` + `select.txt` (the 34 recorded
seqs), `render_tpl.py` (template-exact history), `cmp_arms.py` (probe stats),
`run_*.sh` (as run; scratch paths). Per-sample lengths (no content) live next
to the probe tools in `../agentic-2026-09-09-echo/` (`gap_*`, `replay_*`,
`raw_*`.jsonl). Journals, recorded bodies and rendered prompts are session
content and stay local. n=1 campaign per leg unless noted; run-to-run noise
on these instances is about +-3 turns and +-4K tokens per instance.
