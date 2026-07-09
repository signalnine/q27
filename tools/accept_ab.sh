#!/usr/bin/env bash
# accept_ab.sh -- acceptance-gate Phase 0 replay A/B (docs/acceptance-gate-design.md).
#
# For each payload (repro/code/testgen) x each depth leg (d4/d5/auto):
# fresh server, 1 cold prefill + 3 identical replays (P13 methodology,
# BUILDLOG:1655), greedy. Reports per leg: median warm decode t/s, tok/round,
# rounds (must be identical across replays -- greedy determinism), and the
# cumulative per-lane yields gla[j]/glf[j] from the final [req] line.
#
# Usage: bash tools/accept_ab.sh [PAYLOAD ...]   (default: all three)
# Env: MODEL, TOK, PORT, LEGS ("4 5 auto"), MAXTOK override.
# Needs the 5090 free; run tools/make_payloads.py first.

set -u
cd "$(dirname "$0")/.."
MODEL=${MODEL:-/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27}
TOK=${TOK:-/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.tok}
PORT=${PORT:-8199}
CTX=${CTX:-32768}
LEGS=${LEGS:-"4 5 auto"}
PAYLOADS=${*:-echo docs codegen testgen}
SRV=""

stop_server() { [ -n "$SRV" ] && kill "$SRV" 2>/dev/null && wait "$SRV" 2>/dev/null; SRV=""; }
trap stop_server EXIT

for pay in $PAYLOADS; do
  BODY=scratchpad/accept_payload_${pay}.json
  [ -f "$BODY" ] || { echo "missing $BODY (run tools/make_payloads.py)"; exit 1; }
  for leg in $LEGS; do
    LOG=$(mktemp /tmp/accept_ab.XXXXXX.log)
    Q27_KV=fp8 Q27_PMIN=0.5 Q27_MAXD=$leg \
      build/q27-server "$MODEL" "$TOK" --port "$PORT" --ctx "$CTX" --no-think \
      --fast-head >"$LOG" 2>&1 &
    SRV=$!
    for i in $(seq 1 120); do
      curl -s -m 2 "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 2
    done
    for r in 1 2 3 4; do
      curl -s -m 600 "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
        --data-binary @"$BODY" >/dev/null
    done
    stop_server
    python3 - "$LOG" "$pay" "$leg" <<'PYEOF'
import re, statistics, sys
log, pay, leg = sys.argv[1:4]
reqs = [l for l in open(log) if "[req]" in l]
assert len(reqs) == 4, f"{pay}/{leg}: want 4 [req] lines, got {len(reqs)}"
def f(pat, l, cast=float):
    m = re.search(pat, l)
    return cast(m.group(1)) if m else None
warm = reqs[1:]
tps = [f(r" tps=([\d.]+)", l) for l in warm]
dec = [f(r" dec=(\d+)", l, int) for l in warm]
rnd = [f(r" rounds=(\d+)", l, int) for l in warm]
dms = [f(r" dec_ms=([\d.]+)", l) for l in warm]
det = "OK" if len(set(rnd)) == 1 and len(set(dec)) == 1 else f"NONDET rounds={rnd} dec={dec}"
prompt = f(r" prompt=(\d+)", reqs[0], int)
last = reqs[-1]
def vec(name, l):
    m = re.search(rf" {name}=([\d,]+)", l)
    return [int(x) for x in m.group(1).split(",")] if m else None
glf, gla, gch = vec("glf", last), vec("gla", last), vec("gch", last)
y = ["%.3f" % (a / fd) if fd else "--" for a, fd in zip(gla or [], glf or [])]
gated = sum(gch) if gch else 0
fired5 = "%.3f" % (glf[4] / gated) if glf and gated else "--"
if not rnd[0] or not dec[0]:
    sys.exit(f"{pay}/{leg}: dec={dec[0]} rounds={rnd[0]} prompt={prompt} -- "
             "zero-output leg (prompt>ctx returns 0 tokens on /v1/completions; "
             "or instant EOS). Fix the payload, this leg measured nothing.")
if dec[0] < 200:
    print(f"  WARNING: dec={dec[0]} < 200 (early EOS -- payload not open enough?)")
print(f"{pay:8s} d{leg:4s} prompt={prompt} tps_med={statistics.median(tps):7.1f} "
      f"tok/rnd={dec[0]/rnd[0]:5.3f} ms/rnd={statistics.median(dms)/rnd[0]:6.2f} "
      f"rounds={rnd[0]} det={det} y1..5={','.join(y)} fired5={fired5}")
PYEOF
    rm -f "$LOG"
  done
done
