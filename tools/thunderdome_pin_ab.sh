#!/usr/bin/env bash
# gemv-pin A/B through the REAL agentic harness (thunderdome + Claude Code).
#
# Leg = whichever q27-server binary sits behind :8081. Same commit, same model,
# same day, same tasks; the ONLY difference is Q27_GEMV_2CTA_MIN (10 = shipped
# 2-CTA retier, 99 = the old 3-CTA pin).
#
# WHAT TO READ: the engine's own [req] decode telemetry, not the wall clock.
# Cross-run CC legs FORK (tool outputs carry wall-clock bytes, so the two legs
# stop sharing a trajectory within a few requests) -- walls and token volumes
# are trajectory-confounded, and scores are a documented tie-lottery. Decode
# rate over hundreds of requests is the currency (BUILDLOG 2026-07-10).
#
# Usage: tools/thunderdome_pin_ab.sh <leg-name> <server-binary> [tasks...]
set -u
LEG="${1:?usage: thunderdome_pin_ab.sh <leg> <server-binary> [T8 T2 ...]}"
BIN="${2:?}"
shift 2
TASKS=("${@:-T8}")
TD=/mnt/ai/projects/thunderdome
MODEL=/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27
TOK=/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.tok
OUT=/tmp/pinab_$LEG

systemctl --user stop q27-eval 2>/dev/null
sleep 3
systemd-run --user --unit=q27-eval "$BIN" "$MODEL" "$TOK" \
  --port 8081 --host 0.0.0.0 >/dev/null 2>&1
for _ in $(seq 180); do
  curl -s -m 1 http://127.0.0.1:8081/health >/dev/null 2>&1 && break
  sleep 2
done
curl -s -m 3 http://127.0.0.1:8081/health | grep -q ok || { echo "$LEG: server never came up" >&2; exit 1; }
START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$LEG] serving $(basename "$BIN") on :8081 since $START"

cd "$TD"
for T in "${TASKS[@]}"; do
  echo "[$LEG] running $T ..."
  ./thunderdome run --orchestrator claude-code-q27-haight --task "$T" --trials 1 \
    >"$OUT.$T.harness.log" 2>&1
  echo "[$LEG] $T harness exit=$?"
done

# harvest the engine's [req] lines for the window this leg served
journalctl --user -u q27-eval --since "$START" --no-pager -o cat 2>/dev/null \
  | grep '^\[req\]' > "$OUT.req"
echo "[$LEG] captured $(wc -l < "$OUT.req") [req] lines -> $OUT.req"
