#!/usr/bin/env bash
# Production launch recipe for the q27-38 serving unit (Qwen3.8-27B-MTP on the
# 5090). q27-38 is a TRANSIENT systemd-run --user unit: `systemctl --user stop
# q27-38` deletes it, this script recreates it. Modes:
#   d2       production since 2026-09-08: DFlash2 Q8 pack, sampled walk, MMA verify
#   d2-pfx   d2 + the shipped prefix-cache tiers ON (tmpfs-backed disk tier +
#            host-RAM tier); docs/plans/2026-09-08-prefill-attack.md phase 0
#   ladder   the pre-09-08 production config (MTP ladder + suffix drafter)
# Extra `-E K=V` after the mode are passed to systemd-run (e.g. -E Q27_SYSBLK=1).
set -euo pipefail
Q=/mnt/ai/projects/q27
BIN=$Q/build/q27-server
MODEL=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27
TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
PACK8=/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q8-serve.d2w
ARGS="--host 172.17.0.1 --port 8081 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0"
D2ENV="-E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_BATCH=0 -E Q27_DFLASH2=$PACK8 -E Q27_DFLASH2_RESERVE_GB=3 -E Q27_D2_TIMING=1"
# P16 disk tier on tmpfs (zero SSD wear; /dev/shm had 62 GB free on 09-08),
# P16c RAM tier, max_tokens raised so 48K+ conversations persist (default
# 32768 would silently never persist them), step left at the 8192 default.
PFX_DIR=${PFX_DIR:-/dev/shm/q27-pfx}
PFXARGS="--prefix-cache $PFX_DIR --prefix-cache-max-gb ${PFX_MAX_GB:-40} --prefix-cache-ram-gb ${PFX_RAM_GB:-16} --prefix-cache-max-tokens ${PFX_MAX_TOK:-65536}"
mode=${1:-}; shift || true
systemctl --user stop q27-38 2>/dev/null || true
systemctl --user reset-failed q27-38 2>/dev/null || true
case "$mode" in
  d2)      systemd-run --user --unit q27-38 $D2ENV "$@" $BIN $MODEL $TOK $ARGS ;;
  d2-pfx)  mkdir -p "$PFX_DIR"
           systemd-run --user --unit q27-38 $D2ENV "$@" $BIN $MODEL $TOK $ARGS $PFXARGS ;;
  ladder)  systemd-run --user --unit q27-38 -E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 "$@" $BIN $MODEL $TOK $ARGS ;;
  *) echo "usage: $0 d2|d2-pfx|ladder [-E K=V ...]" >&2; exit 2 ;;
esac
# readiness: key on THIS invocation (a --since window can match the previous
# unit's "serving ON" line)
inv=$(systemctl --user show q27-38 -p InvocationID --value)
for i in $(seq 1 240); do
  if journalctl --user _SYSTEMD_INVOCATION_ID="$inv" -o cat --no-pager 2>/dev/null | grep -q "serving ON"; then
    echo "q27-38 ($mode) serving; invocation $inv"; exit 0
  fi
  systemctl --user is-active --quiet q27-38 || { echo "q27-38 died during startup" >&2; journalctl --user _SYSTEMD_INVOCATION_ID="$inv" -o cat --no-pager | tail -5; exit 1; }
  sleep 2
done
echo "q27-38 never reported serving ON" >&2; exit 1
