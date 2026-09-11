#!/usr/bin/env bash
# Reasoning-length gap probe: what makes ninfer terse, and what moves q27.
# Same fixed turn-0 Claude Code body (28 tools, pytest-10081 task), 24 seeds
# per arm, the card sampler on every arm. Production relaunched at the end.
set -u
S=/tmp/claude-1000/-home-gabe/22610e4a-d806-40af-82cc-4755ef136fd9/scratchpad/probe
Q=/mnt/ai/projects/q27; D=/mnt/ai/projects/q27-master/bench/crossengine/agentic-2026-09-09-echo
MODEL=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27; TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
PACK8=/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q8-serve.d2w
Q27ARGS="--host 0.0.0.0 --port 8081 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0"
NINFER=/mnt/ai/projects/ninfer-master/build/apps/ninfer-serve
NV=/mnt/ai/models/ninfer/qwen38-nvfp4-release/qwen3_8_27b_nvfp4.ninfer
NI=/mnt/ai/models/ninfer/qwen3_8_27b.ninfer
NARGS="--host 0.0.0.0 --port 8081 --kv-capacity 131072 --max-context 131072 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.05"
N=${N:-24}
log() { echo "[gap $(date '+%H:%M:%S')] $*"; }
busy=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | /usr/bin/grep GPU-e592b842 | /usr/bin/grep -v "$Q/build/q27-server" || true)
[ -n "$busy" ] && { log "5090 busy: $busy"; exit 3; }
# the low-effort body: same request, output_config.effort = low
python3 - "$S" <<'EOF'
import json, sys
S = sys.argv[1]; b = json.load(open(f"{S}/body_turn0.json"))
b.setdefault("output_config", {})["effort"] = "low"; json.dump(b, open(f"{S}/body_turn0_low.json", "w"))
EOF
systemctl --user stop q27-38; sleep 4
for i in $(seq 1 30); do ss -ltn | /usr/bin/grep -q ':8081 ' || break; sleep 1; done
arm() { # label body cmd...
  local label=$1 body=$2; shift 2
  local unit=gap-$label; systemctl --user reset-failed $unit 2>/dev/null
  systemd-run --user --unit $unit "$@" >/dev/null 2>&1
  local inv; inv=$(systemctl --user show $unit -p InvocationID --value)
  local ok=0
  for i in $(seq 1 150); do
    case "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health)" in 200|401) ok=1; break;; esac
    systemctl --user is-active --quiet $unit || break; sleep 2
  done
  if [ $ok = 1 ]; then
    log "=== $label"
    python3 $D/probe_think.py http://127.0.0.1:8081 gap_$label $S/$body $N 4096 2>&1 | tail -1
  else
    log "$label never came up"; journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager | tail -3
  fi
  journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager | /usr/bin/grep -m1 -o 'wsum: [0-9a-f]*' | sed 's/^/  /'
  systemctl --user stop $unit; sleep 3
  for i in $(seq 1 30); do ss -ltn | /usr/bin/grep -q ':8081 ' || break; sleep 1; done
}
arm ninfer-nvfp4-int8kv body_turn0.json     $NINFER $NV $NARGS --kv-dtype int8 --spec dflash2 --draft-tokens 7
arm ninfer-nvfp4-bf16kv body_turn0.json     $NINFER $NV $NARGS --kv-dtype bf16 --spec dflash2 --draft-tokens 7
arm ninfer-int-int8kv   body_turn0.json     $NINFER $NI $NARGS --kv-dtype int8
arm q27-medium          body_turn0.json     -E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_BATCH=0 -E Q27_DFLASH2=$PACK8 -E Q27_DFLASH2_RESERVE_GB=3 $Q/build/q27-server $MODEL $TOK $Q27ARGS
arm q27-low             body_turn0_low.json -E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_BATCH=0 -E Q27_DFLASH2=$PACK8 -E Q27_DFLASH2_RESERVE_GB=3 $Q/build/q27-server $MODEL $TOK $Q27ARGS
bash $Q/tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1 | tail -1
log "GAP-DONE"
