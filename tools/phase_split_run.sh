#!/usr/bin/env bash
# phase_split_run.sh -- Saguaro draft-fraction measurement (survey 2026-07-09).
# Per leg: fresh server, 1 cold + 3 warm identical greedy replays, parse [req].
# Legs: cctx+phase, cctx control (no phase, determinism/overhead check), docs+phase.
set -u
cd "$(dirname "$0")/.."
MODEL=${MODEL:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27}
TOK=${TOK:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.tok}
PORT=${PORT:-8207}
CTX=${CTX:-32768}
SRV=""
stop_server() { [ -n "$SRV" ] && kill "$SRV" 2>/dev/null && wait "$SRV" 2>/dev/null; SRV=""; }
trap stop_server EXIT

run_leg() { # $1=payload $2=phase(0/1) $3=tag
  local BODY=scratchpad/accept_payload_$1.json
  [ -f "$BODY" ] || { echo "missing $BODY"; exit 1; }
  local LOG=/tmp/phase_split_$3.log
  local PH=()
  [ "$2" = 1 ] && PH=(Q27_PHASE_STATS=1)
  env "${PH[@]}" Q27_KV=fp8 Q27_PMIN=0.5 Q27_MAXD=auto \
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
  echo "=== leg $3 (payload=$1 phase=$2) ==="
  grep "\[req\]" "$LOG" | sed 's/conv=[0-9a-f]*//'
}

run_leg cctx 1 cctx_phase
run_leg cctx 0 cctx_ctrl
run_leg docs 1 docs_phase
echo ALL_LEGS_DONE
