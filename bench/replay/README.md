# Request recording + sequential replay (2026-09-08, item 2 of the (p) agenda)

q27 samples with seed 0 when the client sends none (Claude Code never does),
so a turn's output is deterministic per prompt AND per preceding sequence:
the DFlash2 drafter ring and every prefix-cache tier are history-dependent,
and the same prompt after a different history draws different tokens
(BUILDLOG (s): the same instance re-run through the harness produced a
different first turn). An A/B between two binaries therefore has to feed
both the identical sequence. This is that instrument.

## Record

    REQ_LOG=/mnt/ai/projects/q27/build/reqlog-$(date +%F).jsonl bash tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1

or `Q27_REQ_LOG=<file>` in any q27-server environment. One JSONL line per
request on `/v1/messages`, `/v1/responses`, `/v1/chat/completions` and
`/v1/completions`: `seq` (arrival order), `t_ms` (epoch ms at receipt),
`api`, `path`, `body` (the raw request body, verbatim). It is real session
content: keep it local, delete it when done, never fold it into the repo.

## Replay

    bench/replay/replay.py build/reqlog-2026-09-08.jsonl --base http://172.17.0.1:8081 --out A.jsonl

posts every body in order to the same path, one at a time (waits for each
response), streaming when the body says so, and records per request the
sha256 of the delivered output in order (text, thinking, tool_use names
and their JSON), output tokens, TTFT and wall. `--limit N` / `--start SEQ`
cut the sequence.

    bench/replay/replay_diff.py A.jsonl B.jsonl

per request: identical output or not, tokens, TTFT, wall for both arms;
then the count of identical outputs and the FIRST divergent seq. Two
fresh boots of the same binary replaying the same log must agree on every
sha (the determinism gate); two binaries that differ only in speed must
too; the first divergence localises a numerics or parser change.

Boot each arm FRESH (`systemctl --user stop q27-38`, relaunch on the arm's
binary, replay, stop): a replay against a server that has served anything
since boot starts from a different ring/cache history. Do not run two
arms concurrently on one GPU. The server-side [req] lines of each boot
carry dec/rounds/dec_ms for tok/round and round-wall comparisons
(bench/crossengine/agentic-2026-09-08/queue_attr.py-style parsing).
