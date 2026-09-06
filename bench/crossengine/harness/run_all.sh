#!/usr/bin/env bash
# Orchestrator for the ninfer-vs-q27 head-to-head.
#
# Serial by construction: one engine owns the 5090 at a time. Nothing here may
# run concurrently -- two legs sharing the card would make every number a
# contention measurement.
#
# Resumable: a leg/arm whose output file already exists is skipped, so a crash
# or a stop costs one arm, not the run.
#
#   usage: run_all.sh <outdir> [arms] [legs]
#     arms  csv of agentic,quality,ladder   (default: all three)
#     legs  csv of q4s,nint,q5f,nvfp4,llama,vllm,q38,q38q4s,llama38 (default:
#           the six Qwen3.6 legs; q38/q38q4s are Qwen3.8-27B-MTP on q27 and
#           llama38 is the same checkpoint as Q5_K_M on llama.cpp -- the
#           size-matched 3.8 competitor, added 2026-08-22)
set -u

SP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SP/legs.sh"

OUT="${1:?usage: run_all.sh <outdir> [arms] [legs]}"
ARMS="${2:-agentic,quality,ladder}"
LEGS="${3:-q4s,nint,q5f,nvfp4,llama,vllm}"
TAP_PORT=8081
mkdir -p "$OUT"

MANIFEST="$OUT/manifest.txt"
{
  echo "run: $(date -Is)"
  echo "arms: $ARMS"
  echo "legs: $LEGS"
  echo "q27 commit: $(git -C /mnt/ai/projects/q27 rev-parse --short HEAD)"
  echo "ninfer commit: $(git -C "$(dirname "$(dirname "$(dirname "$NINFER_BIN")")")" rev-parse --short HEAD 2>/dev/null)"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0)"
} >"$MANIFEST"
cat "$MANIFEST"

has_arm() { [[ ",$ARMS," == *",$1,"* ]]; }
has_leg() { [[ ",$LEGS," == *",$1,"* ]]; }

# Applied on EVERY leg so both engines see the same request.
#
# output_config -- Claude Code 2.1.170 sends {"effort":"high"}; q27 ignores
#   unknown top-level knobs, ninfer's Qwen3.6 template 400s on it.
#
# thinking -- Claude Code sends {"type":"adaptive"}. ninfer maps any non-
#   "disabled" type to enable_thinking=true, and translate.cpp resolves
#   request.enable_thinking.value_or(server.enable_thinking), so an engaged
#   optional SHADOWS --no-thinking: 167/190 nint requests ran with reasoning on
#   while q27 ran hard server-side think=0. Dropping the field lets the server
#   default apply, which is what q27 already does. Without this the decode t/s
#   and gold-hit columns compare different amounts of work.
#
# Normalizing here keeps the A/B honest -- the alternative is one engine
# silently discarding what the other enforces, which is not the same request.
STRIP_FIELDS="${STRIP_FIELDS:-output_config,thinking}"

TAP_PID=""
tap_start() {
  local log="$1"
  python3 "$SP/tapproxy.py" --listen $TAP_PORT --upstream "127.0.0.1:$2" --log "$log" \
    --strip-fields "$STRIP_FIELDS" --translate-thinking >"$OUT/tap.stderr" 2>&1 &
  TAP_PID=$!
  sleep 2
}
tap_stop() {
  [ -n "$TAP_PID" ] && kill "$TAP_PID" 2>/dev/null
  TAP_PID=""
  # the listen socket must be free before the next arm rebinds it
  sleep 3
}

cleanup() { tap_stop; [ -n "${CUR_LOG:-}" ] && leg_stop "$CUR_LOG"; }
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------- serving arms
for LEG in q4s nint q5f nvfp4 llama vllm q38 q38q4s llama38 nvfp4m; do
  has_leg "$LEG" || continue
  { has_arm agentic || has_arm quality; } || continue
  [ -s "$OUT/agentic.$LEG.jsonl" ] && [ -s "$OUT/quality.$LEG.json" ] && \
    { echo "[skip] $LEG serving arms already done"; continue; }

  echo ""; echo "################ LEG $LEG (serving) ################"
  CUR_LOG="$OUT/server.$LEG.agentic.log"
  T0=$(date +%s)
  PORT=$(leg_start "$LEG" agentic "$CUR_LOG") || { echo "start FAILED"; continue; }
  if ! leg_wait "$PORT" 900; then
    echo "[$LEG] boot timeout -- tail:"; tail -20 "$CUR_LOG"; leg_stop "$CUR_LOG"; continue
  fi
  echo "[$LEG] booted in $(( $(date +%s) - T0 ))s on :$PORT, VRAM $(vram_now) MiB"

  tap_start "$OUT/tap.$LEG.jsonl" "$PORT"

  if has_arm agentic && [ ! -s "$OUT/agentic.$LEG.jsonl" ]; then
    bash "$SP/agentic.sh" "$LEG" "$TAP_PORT" "$OUT" 2>&1 | tee "$OUT/agentic.$LEG.log"
  fi
  if has_arm quality && [ ! -s "$OUT/quality.$LEG.json" ]; then
    # Through the TAP, not the engine directly: benchlocal's --no-thinking
    # sends chat_template_kwargs.enable_thinking, which ninfer 400s on, so
    # quality needs the same normalization the agentic arm gets. The extra
    # local hop costs nothing that matters to a correctness score.
    bash "$SP/quality.sh" "$LEG" "$TAP_PORT" "$OUT"
  fi

  echo "[$LEG] peak VRAM $(vram_now) MiB"
  tap_stop
  leg_stop "$CUR_LOG"; CUR_LOG=""
done

# ----------------------------------------------------------------- ladder arm
if has_arm ladder; then
for LEG in q4s nint q5f nvfp4 llama vllm q38 q38q4s llama38 nvfp4m; do
  has_leg "$LEG" || continue
  [ -s "$OUT/ladder.$LEG.txt" ] && { echo "[skip] $LEG ladder already done"; continue; }

  echo ""; echo "################ LEG $LEG (ladder) ################"
  CUR_LOG="$OUT/server.$LEG.ladder.log"
  PORT=$(leg_start "$LEG" ladder "$CUR_LOG") || { echo "start FAILED"; continue; }
  if ! leg_wait "$PORT" 900; then
    echo "[$LEG] boot timeout -- tail:"; tail -20 "$CUR_LOG"; leg_stop "$CUR_LOG"; continue
  fi
  # how many slots actually admitted is itself a result: q5f and nvfp4 carry
  # ~2.8 GB more weights than q4s, and the 8-slot figure was measured on q4s.
  grep -iE "slot|admit|skipped" "$CUR_LOG" | tail -15 | sed "s/^/[$LEG] /"
  echo "[$LEG] VRAM after boot $(vram_now) MiB"

  tap_start "$OUT/tapladder.$LEG.jsonl" "$PORT"
  SRV_ARG=(); [ "$(leg_engine "$LEG")" = q27 ] && SRV_ARG=(--server-log "$CUR_LOG")
  for C in 1 2 4 8; do
    python3 "$SP/ladder_tap.py" --url "http://127.0.0.1:$TAP_PORT" \
      --tap "$OUT/tapladder.$LEG.jsonl" --c "$C" --label "$LEG" \
      "${SRV_ARG[@]}" 2>&1 | tee -a "$OUT/ladder.$LEG.txt"
    sleep 5
  done
  tap_stop
  leg_stop "$CUR_LOG"; CUR_LOG=""
done
fi

echo ""; echo "################ DONE ################"
echo "results in $OUT"
ls -la "$OUT"
