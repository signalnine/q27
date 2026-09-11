#!/usr/bin/env bash
# Mid-session replay: the same recorded Claude Code turns (medium effort, from
# the q27v0113c leg) on q27 production recipe, ninfer DFlash2 NVFP4, and the
# llama.cpp Q8_0 reference (5090+3090 split, so vox is stopped for the arm and
# restarted after). N seeds per body. Production relaunched at the end.
set -u
S=/tmp/claude-1000/-home-gabe/22610e4a-d806-40af-82cc-4755ef136fd9/scratchpad
B=$S/bodies; Q=/mnt/ai/projects/q27; D=/mnt/ai/projects/q27-master/bench/crossengine/agentic-2026-09-09-echo
MODEL=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27; TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
PACK8=/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q8-serve.d2w
Q27ARGS="--host 0.0.0.0 --port 8081 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0"
NINFER=/mnt/ai/projects/ninfer-master/build/apps/ninfer-serve
NV=/mnt/ai/models/ninfer/qwen38-nvfp4-release/qwen3_8_27b_nvfp4.ninfer
NARGS="--host 0.0.0.0 --port 8081 --kv-dtype int8 --kv-capacity 131072 --max-context 131072 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.05"
LS=/mnt/ai/projects/llama.cpp/build/bin/llama-server; G8=/mnt/ai/models/qwen38-27b-mtp-gguf/Qwen3.8-27B-MTP-Q8_0.gguf
N=${N:-6}; ARMS=${ARMS:-"q27 ninfer llamaq8"}; BODIES=${BODIES:-$B/req.q27v0113c.jsonl}; SEL=${SEL:-$B/select.txt}
log() { echo "[replay $(date '+%H:%M:%S')] $*"; }
portfree() { for i in $(seq 1 90); do [ -z "$(ss -tanH '( sport = :8081 )')" ] && return 0; sleep 1; done; log "8081 still has sockets"; }
busy=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | /usr/bin/grep GPU-e592b842 | /usr/bin/grep -v "$Q/build/q27-server" || true)
[ -n "$busy" ] && { log "5090 busy: $busy"; exit 3; }
systemctl --user stop q27-38; sleep 3; portfree
up() { # unit -> 0 when healthy
  for i in $(seq 1 300); do
    case "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health)" in 200|401) return 0;; esac
    systemctl --user is-active --quiet $1 || return 1; sleep 2
  done; return 1; }
for arm in $ARMS; do
  unit=replay-$arm; systemctl --user reset-failed $unit 2>/dev/null; extra=""
  case $arm in
    q27) systemd-run --user --unit $unit -E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_BATCH=0 -E Q27_DFLASH2=$PACK8 -E Q27_DFLASH2_RESERVE_GB=3 $Q/build/q27-server $MODEL $TOK $Q27ARGS >/dev/null 2>&1 ;;
    ninfer) systemd-run --user --unit $unit $NINFER $NV $NARGS --spec dflash2 --draft-tokens 7 >/dev/null 2>&1 ;;
    llamaq8) sudo -n systemctl stop vox-transcriber vox-transcriber-gmrs; sleep 3
      b3=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | /usr/bin/grep GPU-5a723c5e || true)
      if [ -n "$b3" ]; then log "3090 busy after vox stop: $b3 -- skipping llamaq8"; sudo -n systemctl start vox-transcriber vox-transcriber-gmrs; continue; fi
      systemd-run --user --unit $unit -E CUDA_VISIBLE_DEVICES=0,1 $LS -m $G8 -ngl 999 -c 131072 -np 1 --host 0.0.0.0 --port 8081 --jinja --reasoning-format deepseek --cache-prompt -fa on --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.05 >/dev/null 2>&1
      extra=--llama ;;
  esac
  inv=$(systemctl --user show $unit -p InvocationID --value)
  if up $unit; then
    log "=== $arm up"
    python3 $D/replay_think.py http://127.0.0.1:8081 replay_$arm $BODIES $SEL $N 16384 $extra 2>&1
  else
    log "$arm never came up"; journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager | tail -4
  fi
  journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager | /usr/bin/grep -m1 -o 'wsum: [0-9a-f]*' | sed 's/^/  /'
  systemctl --user stop $unit; sleep 3; portfree
  [ $arm = llamaq8 ] && sudo -n systemctl start vox-transcriber vox-transcriber-gmrs
done
bash $Q/tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1 | tail -1
log "vox: $(systemctl is-active vox-transcriber) $(systemctl is-active vox-transcriber-gmrs)"
log "REPLAY-DONE"
