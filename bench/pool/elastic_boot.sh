#!/usr/bin/env bash
# Elastic multi-slot windows (issue #42): boot matrix. Each config boots a
# transient q27-server on 127.0.0.1:$PORT, prints its sizing lines, and stops.
# Journal reads are keyed on the unit's InvocationID -- an unfiltered
# `journalctl -u` matches a PREVIOUS invocation's "listening on" and reports
# a dead boot as healthy (2026-09-10, the stale-instrument trap).
# Refuses to run if anything but production's q27-server holds the GPU.
# usage: elastic_boot.sh <label> [server args...]   (one config per call)
#   env: GPU_UUID (default: the 5090), KV (fp8), PORT (8099), BIN, MODEL, TOK,
#        EXTRA_ENV (e.g. "-E CUDA_VISIBLE_DEVICES=1 -E Q27_KV_POOL=0"), KEEP=1
set -u
Q=/mnt/ai/projects/q27
BIN=${BIN:-$Q/build/q27-server}
MODEL=${MODEL:-/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27}
TOK=${TOK:-/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok}
PORT=${PORT:-8099}
GPU_UUID=${GPU_UUID:-GPU-e592b842}
UNIT=q27-elastic-boot
label=$1; shift
others=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | /usr/bin/grep "$GPU_UUID" | /usr/bin/grep -v "$BIN" || true)
if [ -n "$others" ]; then echo "=== $label: REFUSED -- GPU busy: $others"; exit 3; fi
systemctl --user reset-failed $UNIT 2>/dev/null
systemd-run --user --unit $UNIT -E Q27_KV=${KV:-fp8} ${EXTRA_ENV:-} $BIN $MODEL $TOK --host 127.0.0.1 --port $PORT --think "$@" >/dev/null 2>&1
inv=$(systemctl --user show $UNIT -p InvocationID --value)
log() { journalctl --user _SYSTEMD_INVOCATION_ID="$inv" -o cat --no-pager 2>/dev/null; }
state=TIMEOUT
for i in $(seq 1 150); do
  if log | /usr/bin/grep -q 'listening on'; then state=BOOTED; break; fi
  if ! systemctl --user is-active --quiet $UNIT; then state=DIED; break; fi
  sleep 2
done
echo "=== $label: $* -> $state (invocation $inv)"
log | /usr/bin/grep -E 'vram:|--ctx auto|\[pool\]|slot [0-9]+ (ready|SKIPPED)|listening|out of memory|FATAL|CUDA error' \
    | /usr/bin/grep -v 'paged KV:' | cut -c1-170 | sed 's/^/  /'
[ "$state" = BOOTED ] && [ "${KEEP:-0}" = 1 ] && { echo "  (left running for tests: unit $UNIT)"; exit 0; }
systemctl --user stop $UNIT; sleep 3
[ "$state" = BOOTED ]
