#!/bin/bash
# GATE: divergence-then-replay must not restore cache state whose KV rows
# were overwritten by another branch (P9 alias, found by audit 2026-07-12,
# fixed e16c394; 552 live traversals of the condition measured on 07-11
# traffic). Promoted to the standing battery: run on EVERY cache-path
# change, the way the parser drift modes became fixtures.
#
# Shape: prompt A (multi-chunk, ring saves) -> prompt B sharing >1 chunk
# then diverging (forces a base>0 restore + re-prefill) -> prompt A again.
# PASS = the replay restores only surviving-coverage state (its prefix_hit
# does not exceed the divergence base). FAIL = it restores past the
# divergence point, i.e. recurrent state over the other branch's KV rows.
#
# Usage: tools/ckpt_gate.sh build/q27-server [model.q27 model.tok]
set -u
BIN=${1:?usage: ckpt_gate.sh build/q27-server [model tok]}
MODEL=${2:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27}
TOK=${3:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.tok}
PORT=${CKPT_GATE_PORT:-8196}
S="${TMPDIR:-/tmp}/q27-ckpt-gate"; mkdir -p "$S"

python3 - "$S" <<'PY'
import random, sys
S = sys.argv[1]
words = ("valve boiler gauge copper steam piston flange rivet lathe crane "
         "winch pulley girder beam truss anvil forge ingot billet furnace "
         "damper throttle governor manifold gasket").split()
def para(seed, n):
    r = random.Random(seed)
    return " ".join(r.choice(words) for _ in range(n))
shared = "Inspection log for the harbor works. " + para(1, 1300)
open(f"{S}/pA.txt", "w").write(shared + " SECTION ALPHA. " + para(2, 1800))
open(f"{S}/pB.txt", "w").write(shared + " SECTION BRAVO. " + para(3, 500))
PY

Q27_CKPT_INTERVAL=64 "$BIN" "$MODEL" "$TOK" --port "$PORT" --ctx 16384 \
    2>"$S/srv.log" &
SRV=$!
for _ in $(seq 1 60); do
    curl -s -m 2 "localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 3
    kill -0 $SRV 2>/dev/null || { echo "CKPT GATE FAIL (server died)"; exit 1; }
done
for P in pA pB pA; do
    python3 -c "import json;print(json.dumps({'model':'x','prompt':open('$S/$P.txt').read(),'max_tokens':4,'temperature':0}))" > "$S/body.json"
    curl -s "localhost:$PORT/v1/completions" -H 'content-type: application/json' \
        -d @"$S/body.json" >/dev/null
done
kill $SRV 2>/dev/null; wait 2>/dev/null

grep "\[gen\]" "$S/srv.log"
R2=$(grep "\[gen\]" "$S/srv.log" | sed -n 2p | grep -oE "prefix_hit=[0-9]+" | cut -d= -f2)
R3=$(grep "\[gen\]" "$S/srv.log" | sed -n 3p | grep -oE "prefix_hit=[0-9]+" | cut -d= -f2)
if [ -n "${R3:-}" ] && [ -n "${R2:-}" ] && [ "$R2" -gt 0 ] && [ "$R3" -le "$R2" ]; then
    echo "CKPT GATE PASS (replay hit $R3 <= divergence base $R2)"
else
    echo "CKPT GATE FAIL (replay hit ${R3:-none}, divergence base ${R2:-none})"
    exit 1
fi
