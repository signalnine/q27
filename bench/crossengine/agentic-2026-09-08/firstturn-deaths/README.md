# First-turn deaths in the 2026-09-08 production arms (item 2 of the (p) agenda)

The seven Claude Code first turns that ended a SWE-bench session after ONE
assistant message in the q27 arms of this campaign (BUILDLOG 2026-09-08 (s)).
Each file is the assistant's visible text of that turn, verbatim, from
`/mnt/ai/swebench-work/<arm>/<iid>/logs/out.jsonl`. `tools_cc.json` is the
28-tool Claude Code 4.8 schema those requests carried (no Grep, no Glob).

Replay one through the SAME streaming path the /v1/messages handler uses:

    make build/stream_probe
    ./build/stream_probe bench/crossengine/agentic-2026-09-08/firstturn-deaths/tools_cc.json \
        bench/crossengine/agentic-2026-09-08/firstturn-deaths/prodpfx2_psf__requests-1142.txt 4

| file | shape | after the (s) fixes |
|---|---|---|
| prodpfx2_psf__requests-1142 | `<function_calls>\n<invoke>\n<parameter=file_path>...</invoke>` (unnamed invoke, EOS) | Read (mode 21, `</invoke>` closer) |
| prodpfx2_pydata__xarray-4075 | `<function=Grep>` x2 with `<parameter=-n>` -- Grep not declared by Claude Code 4.8 | 2x Grep passed through; the client answers "no such tool" |
| prodlad_pydata__xarray-4094 | `<parameter=function=Bash>` opener | Bash (mode 22) |
| q27lad_psf__requests-1921 | `<tool_calls>\n<invoke>` x2, second truncated at EOS | first call Read; the truncated one stays text |
| prodd2_pydata__xarray-4075 | nested `<parameter=client>proxy<parameter=calls>...` junk | refused (correct) |
| prodpfx2_pylint-dev__pylint-6903 | hallucinated `<system-warning>` block, no call | refused (correct) |
| q27d2q8_pytest-dev__pytest-5262 | `<tool_use>` + parameters, last value unterminated at EOS | refused (policy: a truncated value is never executed) |

Sampling without a client seed is deterministic per prompt (prodpfx and
prodpfx2 produced these texts byte-for-byte on the same instances), so a
live re-run of the instance reproduces the same first turn -- which is what
made the live gate possible (bench/swebench/run.sh <label> <iid>).
