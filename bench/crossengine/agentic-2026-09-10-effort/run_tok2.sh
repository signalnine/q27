#!/usr/bin/env bash
# Fixed binary follow-ups: turn-0 probe (medium, low; 24 seeds), then Claude
# Code legs q27tokb (medium replicate) and q27toklow (effort low).
set -u
S=/tmp/claude-1000/-home-gabe/22610e4a-d806-40af-82cc-4755ef136fd9/scratchpad
Q=/mnt/ai/projects/q27; M=/mnt/ai/projects/q27-master; D=$M/bench/crossengine/agentic-2026-09-09-echo
BIN=$M/build/q27-server
MODEL=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27; TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
PACK8=/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q8-serve.d2w
Q27ARGS="--host 0.0.0.0 --port 8081 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0"
log() { echo "[tok2 $(date '+%H:%M:%S')] $*"; }
portfree() { for i in $(seq 1 90); do [ -z "$(ss -tanH '( sport = :8081 )')" ] && return 0; sleep 1; done; log "8081 still has sockets"; }
busy=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | /usr/bin/grep GPU-e592b842 | /usr/bin/grep -v "$Q/build/q27-server" || true)
[ -n "$busy" ] && { log "5090 busy: $busy"; exit 3; }
log "fixed binary md5 $(md5sum $BIN | cut -c1-8)"
systemctl --user stop q27-38; sleep 3; portfree
unit=probe-q27tok; systemctl --user reset-failed $unit 2>/dev/null
systemd-run --user --unit $unit -E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_BATCH=0 -E Q27_DFLASH2=$PACK8 -E Q27_DFLASH2_RESERVE_GB=3 $BIN $MODEL $TOK $Q27ARGS >/dev/null 2>&1
inv=$(systemctl --user show $unit -p InvocationID --value); ok=0
for i in $(seq 1 150); do case "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health)" in 200|401) ok=1; break;; esac; systemctl --user is-active --quiet $unit || break; sleep 2; done
if [ $ok = 1 ]; then
  for b in body_turn0 body_turn0_low; do
    python3 $D/probe_think.py http://127.0.0.1:8081 gap_q27tok-${b#body_turn0} $S/probe/$b.json 24 4096 2>&1 | tail -1
  done
else log "probe server never came up"; journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager | tail -4; fi
journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager | /usr/bin/grep -m1 -o 'wsum: [0-9a-f]*' | sed 's/^/  /'
systemctl --user stop $unit; sleep 3; portfree
log "=== campaign legs q27tokb q27toklow"
CAMPAIGN_DIR=$M/bench/crossengine/agentic-2026-09-10-effort LEGS="q27tokb q27toklow" CLEAR_PROD_PFX=1 REQBODY_LOG=$S/bodies/req \
  bash $M/bench/crossengine/agentic-2026-09-07/campaign.sh
log "TOK2-DONE"
