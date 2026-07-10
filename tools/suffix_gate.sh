#!/usr/bin/env bash
# suffix_gate.sh -- live gate for the Q27_SUFFIX echo drafter.
# Per payload x {off,on}: fresh server, 1 cold + 3 warm greedy replays,
# responses SAVED. Gates: (1) completion text byte-identical off-vs-on;
# (2) cctx sfx fired > 0 and tps up; (3) docs sfx fired ~= 0, tps flat.
set -u
cd "$(dirname "$0")/.."
MODEL=${MODEL:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27}
TOK=${TOK:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.tok}
PORT=${PORT:-8209}
CTX=${CTX:-32768}
SRV=""
stop_server() { [ -n "$SRV" ] && kill "$SRV" 2>/dev/null && wait "$SRV" 2>/dev/null; SRV=""; }
trap stop_server EXIT

run_leg() { # $1=payload $2=sfx(0/1)
  local BODY=scratchpad/accept_payload_$1.json
  local LOG=/tmp/sfxgate_$1_$2.log
  local SFX=()
  [ "$2" = 1 ] && SFX=(Q27_SUFFIX=1)
  env "${SFX[@]}" Q27_KV=fp8 Q27_PMIN=0.5 Q27_MAXD=auto \
    build/q27-server "$MODEL" "$TOK" --port "$PORT" --ctx "$CTX" --no-think \
    --fast-head >"$LOG" 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do
    curl -s -m 2 "localhost:$PORT/health" >/dev/null 2>&1 && break; sleep 2
  done
  for r in 1 2 3 4; do
    curl -s -m 600 "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
      --data-binary @"$BODY" > /tmp/sfxgate_${1}_${2}_r${r}.json
  done
  stop_server
  echo "=== $1 sfx=$2 ==="
  grep "\[req\]" "$LOG" | sed 's/conv=[0-9a-f]*//' | \
    grep -oE "(rid=[0-9]+|dec=[0-9]+|dec_ms=[0-9.]+|rounds=[0-9]+|tps=[0-9.]+|sfx=[0-9,]+)" | paste - - - - - - 2>/dev/null || \
    grep "\[req\]" "$LOG" | sed 's/conv=[0-9a-f]*//'
}

for pay in cctx docs; do
  run_leg "$pay" 0
  run_leg "$pay" 1
  echo "--- text identity $pay (warm r2..r4):"
  for r in 2 3 4; do
    a=$(python3 -c "import json,sys;print(json.load(open('/tmp/sfxgate_${pay}_0_r${r}.json'))['choices'][0]['text'])" | md5sum | cut -d' ' -f1)
    b=$(python3 -c "import json,sys;print(json.load(open('/tmp/sfxgate_${pay}_1_r${r}.json'))['choices'][0]['text'])" | md5sum | cut -d' ' -f1)
    [ "$a" = "$b" ] && echo "  r$r IDENTICAL ($a)" || echo "  r$r MISMATCH off=$a on=$b"
  done
done
echo ALL_DONE
