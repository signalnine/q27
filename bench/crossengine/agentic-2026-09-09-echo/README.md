# 2026-09-09: why q27's Claude Code sessions ran 1.7x the turns of ninfer's

The 09-09 release campaign left one open question: at decode rates within
6% and equal prefix reuse, q27's sessions took 25 turns and 18.2K output
tokens per instance against ninfer's 15 and 6.2K, so q27 paid 3x the wall.
This run attributes it and measures the fix. Instances, harness, effort pin
(medium), sampler chain and binary are the 09-09 campaign's, plus one
change in `src/server.cu`.

## What the transcripts said

Turn structure is the same on both engines (12 instances, Claude Code
2.1.265, retained `out.jsonl` transcripts):

| | q27prod (09-09) | ninferd2 (09-09) |
|---|--:|--:|
| API turns / instance | 23.8 | 14.0 |
| tool calls / turn | 0.9 | 0.9 |
| turns with no tool call | 4% | 7% |
| tool errors (mostly Bash exit 1, same shapes) | 34 | 21 |
| thinking chars / turn, mean / median | 2222 / 386 | 1106 / 287 |
| turns with >2K chars of thinking | 27% | 14% |
| share of thinking in those turns | 87% | 68% |

No looping tool sequences, no parser recovery, every q27 request ended on
EOS, sampler chains identical (temp 1.0, top-p 0.95, top-k 20, min-p 0.05,
no budget), both rendering medium effort. The excess is more turns AND
longer thinking per turn, spread across instances.

## Where it came from: the model never saw its own prior reasoning

Prompt growth between consecutive requests of one session minus the
previous request's completion tokens is the tool result plus template
overhead when the client sends the previous turn back whole, and goes
negative when part of it is missing:

| engine | pairs | growth minus completion | negative |
|---|--:|--:|--:|
| ninfer (request log) | 156 | min +19 | 0% |
| q27 (`[req]` lines) | 266 | e.g. requests-1921 turn 17: completion 5995, growth 615 | 45% |

Every long-thinking q27 turn is strongly negative: the thinking never came
back. The recorded request bodies from a real production session
(`Q27_REQ_LOG`, 7 requests) confirm it from the client side -- zero
`thinking` blocks in any assistant history message, only `text` and
`tool_use`. On ninfer the growth accounting shows the thinking blocks are
sent back and rendered (ninfer keeps reasoning for every assistant turn
after the last real user query, as the Qwen3.8 template does; q27's
renderer does the same when it is given reasoning, `api_common.h`
`chatml_prompt`).

Both engines return `context_management: null` (neither implements the
`clear_thinking` edit Claude Code asks for), both sign thinking blocks with
a placeholder (`q27-local` / the message id), both stream the same block
shapes. The one difference in what Claude Code sees: the response's `model`
field. q27 returned its served name (`qwen38-27b-mtp`); ninfer echoes the
model the client asked for (`claude-opus-4-8`), as the Anthropic API does.
Claude Code tags each assistant message with the response model and drops
prior thinking blocks from the history it sends back when that tag differs
from the model it is requesting. (Inferred from behaviour; the rule itself
was not located in the binary.)

Live confirmation on production (`echo_test/` in the session scratchpad):
one short session against the old binary recorded no thinking in history;
against the patched binary the bodies carried 1, 2, 3 thinking blocks
(signature `q27-local`, response tags `claude-opus-4-8`), and on this
campaign's echo leg growth minus completion was never negative.

## The fix

`src/server.cu` `/v1/messages`: the response's `model` is the request's
`model` when the client sent one (`resp_model`), the served name otherwise.
`Q27_ECHO_MODEL=0` restores the served name (the `q27noecho` control leg
here). OpenAI-shaped endpoints are unchanged. Claude Code's Grep-not-declared
and other harness traps are unaffected.

## Side check: rendering

Rendering a real 28-tool Claude Code turn-0 body through q27's renderer
(`build/render_request`) and through the HF `apply_chat_template` of the
Qwen3.8 checkpoint gives byte-identical text through the end of the
`<tools>` block (first divergence at char 90015 of 98483, inside the
tool-call instruction paragraph, where the offline tool picks the JSON
dialect because it has no model header; the server picks XML). ninfer's
turn-0 prompt is a constant 1355 tokens larger than q27's on the seven
instances whose first requests pair cleanly -- ninfer renders something
extra; q27 matches the trained template.

## Results: the fix lands, the gap does not move

`turns_cmp.py` over the retained transcripts. `q27prod` and `ninferd2` are
the 09-09 legs; `q27noecho` is the same-day control (patched binary, echo
off, i.e. the 09-09 wire behaviour); `q27echo` is the fix.

| leg | turns/inst | thinking K chars/inst | thinking chars/turn | turns >2K chars | tool calls/inst | out tok/inst | gold | wall/inst |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| q27prod (09-09) | 23.8 | 53.0 | 2222 | 27% | 24.0 | 18.2K | 9/12 | 108 s |
| q27noecho (control) | 21.1 | 40.1 | 1903 | 22% | 21.2 | 14.5K | 10/12 | 105 s |
| q27echo (fix) | 20.2 | 41.9 | 2078 | 26% | 20.5 | 15.3K | 12/12 | 95 s |
| ninferd2 (09-09) | 14.0 | 15.5 | 1106 | 14% | 14.0 | 6.2K | 11/12 | 36 s |

Per instance (turns / thinking K chars / output tokens, `*` = patch touched
a gold file):

| instance | q27prod | q27noecho | q27echo | ninferd2 |
|---|--:|--:|--:|--:|
| flask-5014 | 15 / 10 / 4704* | 15 / 10 / 4704* | 11 / 7 / 3246* | 11 / 3 / 1996* |
| requests-1142 | 16 / 17 / 7524* | 18 / 56 / 16367* | 16 / 21 / 8163* | 13 / 8 / 4352* |
| requests-1724 | 13 / 22 / 8153 | 8 / 9 / 3925 | 17 / 35 / 15073* | 11 / 12 / 5579 |
| requests-1766 | 12 / 4 / 2743* | 12 / 4 / 2743* | 15 / 29 / 10320* | 22 / 15 / 5670* |
| requests-1921 | 34 / 81 / 26692 | 19 / 37 / 15126* | 29 / 57 / 20039* | 18 / 52 / 17734* |
| xarray-4075 | 28 / 56 / 19163 | 40 / 68 / 24503 | 26 / 70 / 22501* | 17 / 14 / 6155* |
| xarray-4094 | 72 / 226 / 72480* | 50 / 118 / 40516* | 51 / 137 / 51301* | 20 / 19 / 6943* |
| pylint-4970 | 50 / 128 / 44830* | 45 / 89 / 33931* | 29 / 80 / 26276* | 39 / 53 / 21284* |
| pylint-6903 | 10 / 11 / 6237* | 10 / 11 / 6237* | 12 / 19 / 7399* | 4 / 2 / 1381* |
| pytest-10081 | 21 / 69 / 21201* | 21 / 69 / 21201* | 20 / 38 / 13417* | 5 / 4 / 1597* |
| pytest-5262 | 8 / 6 / 3425* | 8 / 6 / 3425* | 12 / 9 / 4395* | 3 / 2 / 1139* |
| pytest-5809 | 7 / 3 / 1766* | 7 / 3 / 1766* | 4 / 1 / 917* | 5 / 2 / 1118* |

Reads:

1. **The control reproduces the 09-09 run bit-for-bit on 6 of 12
   instances** (flask-5014, requests-1766, pylint-6903, pytest-10081,
   pytest-5262, pytest-5809: identical thinking, text and tool sequence on
   every turn). q27 samples with seed 0 when the client sends none, so a
   session is a deterministic function of its prompts; the other six
   diverged at turn 1-18 on something outside the engine (a tool result
   with a timestamp, a pip download, a web fetch) and then wandered:
   requests-1921 34 -> 19 turns, xarray-4075 28 -> 40. The harness's
   run-to-run noise on the aggregate is therefore about +-3 turns and +-4K
   tokens per instance, from a handful of instances.
2. **The echo fix is inside that noise.** 20.2 turns / 15.3K tokens
   against the control's 21.1 / 14.5K; thinking per turn 2078 vs 1903
   chars. The model now sees its own prior reasoning (prompt growth minus
   completion never negative on this leg) and gold hits went 10 -> 12, but
   the trajectory length did not move. What was found and fixed is real
   and stays; it is not the answer to the question.
3. **The gap is per-turn and engine-side.** On the identical turn-0 prompt
   (same system prompt, tools and task on both engines; q27's three legs
   produce the same turn-0 output by determinism) q27 thinks 376 chars on
   average (median 342) and ninfer 177 (166) over the 12 tasks; pooled
   over turns 0-2 it is 1248 vs 590. Twelve paired prompts, 2x, before any
   trajectory can differ. The fixed-prompt probe below takes it from there.

## Fixed-prompt probe: q27 matches the near-lossless reference; ninfer is the terse one

`probe_think.py` (Anthropic path) / `probe_think_oai.py` (llama.cpp,
`/v1/chat/completions` with `--jinja`, the HF template) send one identical
turn-0 body -- Claude Code's system prompt and 28 tool schemas from a
recorded production session, the pytest-10081 task as the user turn,
medium effort, thinking on -- 24 times with seeds 1..24, temp 1.0 / top-p
0.95 / top-k 20 / min-p 0.05 / no budget everywhere, and record the
thinking length of the first response. Every sample on every arm ended in
a tool call. Medians with a bootstrap 95% CI; Mann-Whitney against the
llama.cpp Q8_0 arm (the closest thing to the bf16 model this box can
serve) and against ninfer. `probe_*.jsonl` are the per-sample records.

| arm | thinking chars, mean / median [CI] | p90 | out tok mean | vs llama Q8_0 | vs ninfer |
|---|--:|--:|--:|--:|--:|
| q27 production (Q4_G64, DFlash2 Q8, fp8 KV) | 362 / 317 [279, 354] | 429 | 172 | p=0.58 | p<0.0001 |
| q27 MTP ladder (no DFlash2) | 339 / 262 [221, 392] | 573 | 154 | p=0.73 | p=0.008 |
| q27 production, fp16 KV | 679 / 337 [272, 389] | 596 | 237 | p=0.40 | p=0.0002 |
| q27 q6 tier | 341 / 340 [266, 366] | 452 | 158 | p=0.62 | p=0.0002 |
| llama.cpp Q8_0 (made from the BF16 GGUF today) | 343 / 314 [266, 352] | 393 | 147 | -- | p=0.0003 |
| llama.cpp Q5_K_M | 272 / 246 [216, 266] | 350 | 136 | p=0.024 | p=0.09 |
| ninfer DFlash2 (NVFP4, int8 KV) | 219 / 209 [175, 244] | 293 | 129 | p=0.0003 | -- |

Reads:

1. **q27 is the reference, not the outlier.** Every q27 arm is
   indistinguishable from llama.cpp at Q8_0 (p 0.4-0.7); ninfer's NVFP4
   arm thinks 1.5x shorter than that reference (p=0.0003), and llama.cpp's
   own Q5_K_M sits between. The per-turn gap in the sessions is ninfer
   producing less reasoning than the model does at 8 bits, not q27
   producing more. Which of NVFP4 weights, int8 KV, or ninfer's sampler
   does it is ninfer's question; the same sampler chain is pinned on all
   arms, and the render bisect below rules out the prompt.
2. **Drafter, KV dtype and tier are excluded on q27** (four arms, same
   medians; the fp16-KV mean is one 6.6K-char sample).
3. **Rendering, bisected** (`render_bisect.py`, max_tokens=1 with sections
   removed): HF template 24586 tokens for this body, ninfer 24701 (tools
   block +153, system -38), q27 server 23347 -- tools block 1236 short.
   That one is ours, see below; it does not explain the thinking length
   (llama.cpp Q8_0 renders the HF template and lands on q27's numbers).

## A rendering bug found on the way: the serving path never used the tools declaration

`prepare_anthropic_prompt` in `server.cu` built the client-ordered,
template-spaced `<tools>` declaration (the 2026-08-22 feature) from the raw
body with a keep-filter of `selected.names` -- two lines after
`selected.names` had been moved into `tool_names`. The filter saw an empty
list, matched no tool, the declaration came back empty and `tools_preamble`
fell back to the compact key-sorted dump: `{"function":{"description":...,
"name":...,"parameters":...},"type":"function"}` with no spaces. Every
Claude Code request since 08-22 was served that way (production journal:
sys block 22506 tokens where the template gives 23772). `render_request`
builds the declaration before any move, so the offline corpus, the golden
test and the flip gate all saw the right prompt while the server did not.
The integration harness passed the raw body as `nullptr`, so its byte-exact
copy of the lambda never exercised the path. Fixed (`&tool_names`), test
16b added (declaration present with raw body, sorted fallback without),
and the probe re-run on the corrected prompt:

| arm | thinking chars, mean / median [CI] | p90 | out tok mean | vs q27 before | vs llama Q8_0 | vs ninfer |
|---|--:|--:|--:|--:|--:|--:|
| q27 production, declaration fixed (24593 tokens = template) | 389 / 276 [231, 350] | 606 | 168 | p=0.32 | p=0.66 | p=0.0013 |

The corrected prompt (`count_tokens` 24593 against the template's 24586,
tools block 22289 vs 22279) does not move the thinking length: still the
Q8_0 reference, still 1.3x ninfer. The bug was real and is fixed; it was
not the cause either.

## Where this leaves the question

The 1.7x turns / 3x tokens of the 09-09 table decompose as: (a) a harness
whose per-instance trajectories are seed-0 deterministic and swing by
+-3 turns on the aggregate from a handful of instances, so the 09-09
q27prod row (23.8) and today's control (21.1) bracket the same engine;
(b) a per-turn thinking length on which q27 equals the 8-bit model and
ninfer runs 1.3-1.5x short, compounding over a session into the turn and
token counts (the model that thinks less also stops sooner: ninfer's
sessions end at 14 turns with 11/12 gold, q27's at 20-24 with 9-12/12).
Two real defects were found and fixed on q27's side without moving the
gap -- the dropped thinking history (model echo) and the compact tools
block -- and the drafter, KV dtype, tier and sampler chain are excluded.
What remains is on ninfer's side of the table: NVFP4 weights, int8 KV, or
its sampler making the model terser than it is at 8 bits. Wall per
instance is therefore not a q27 engine deficit to chase; the honest
comparison is per-token cost at equal reasoning, where the engines are
within 6%.

Legs: `q27echo` then `q27noecho`, `campaign.sh` in agentic-2026-09-07

Legs: `q27echo` then `q27noecho`, `campaign.sh` in agentic-2026-09-07
(`LEGS="q27echo q27noecho" CAMPAIGN_DIR=<this dir>`), each on a fresh
`/dev/shm/q27-pfx-<leg>` root, vox transcribers stopped, production
relaunched after. Transcripts under `/mnt/ai/swebench-work/<leg>/`.
