#!/usr/bin/env bash
# After the raw run: (1) the FIXED binary (tokenizer + 3.8 history) on the same
# trimmed raw prompts -- isolates the tokenizer against raw_q27_t; (2) a Claude
# Code campaign leg on the fixed binary (q27tok), production cache cleared,
# bodies recorded. Production is relaunched (on the production binary) at the end.
set -u
S=/tmp/claude-1000/-home-gabe/22610e4a-d806-40af-82cc-4755ef136fd9/scratchpad
B=$S/bodies; R=$B/rendered; Q=/mnt/ai/projects/q27; M=/mnt/ai/projects/q27-master
D=$M/bench/crossengine/agentic-2026-09-09-echo
BIN=$M/build/q27-server
MODEL=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27; TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
PACK8=/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q8-serve.d2w
Q27ARGS="--host 0.0.0.0 --port 8081 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0"
log() { echo "[tok $(date '+%H:%M:%S')] $*"; }
portfree() { for i in $(seq 1 90); do [ -z "$(ss -tanH '( sport = :8081 )')" ] && return 0; sleep 1; done; log "8081 still has sockets"; }
until grep -q 'RAW-DONE' $B/run_raw.log; do sleep 10; done
log "raw run done; fixed binary md5 $(md5sum $BIN | cut -c1-8)"
busy=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | /usr/bin/grep GPU-e592b842 | /usr/bin/grep -v "$Q/build/q27-server" || true)
[ -n "$busy" ] && { log "5090 busy: $busy"; exit 3; }
systemctl --user stop q27-38; sleep 3; portfree
unit=raw-q27tok; systemctl --user reset-failed $unit 2>/dev/null
systemd-run --user --unit $unit -E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_BATCH=0 -E Q27_DFLASH2=$PACK8 -E Q27_DFLASH2_RESERVE_GB=3 $BIN $MODEL $TOK $Q27ARGS >/dev/null 2>&1
inv=$(systemctl --user show $unit -p InvocationID --value); ok=0
for i in $(seq 1 150); do case "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health)" in 200|401) ok=1; break;; esac; systemctl --user is-active --quiet $unit || break; sleep 2; done
if [ $ok = 1 ]; then
  log "=== q27tok up"; python3 $D/raw_think.py http://127.0.0.1:8081 raw_q27tok_t q27 $R t $B/select.txt ${N:-4} 16384 2>&1 | tail -1
else log "q27tok never came up"; journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager | tail -4; fi
journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager | /usr/bin/grep -m1 -o 'wsum: [0-9a-f]*' | sed 's/^/  /'
systemctl --user stop $unit; sleep 3; portfree
log "=== campaign leg q27tok"
CAMPAIGN_DIR=$M/bench/crossengine/agentic-2026-09-10-effort LEGS=q27tok CLEAR_PROD_PFX=1 REQBODY_LOG=$B/req \
  bash $M/bench/crossengine/agentic-2026-09-07/campaign.sh
log "TOK-DONE"
