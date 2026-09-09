#!/usr/bin/env bash
# Agentic (Claude Code / SWE-bench) cross-engine campaign, 2026-09-07:
# q27 ladder+suffix (production config) vs q27 DFlash2 (Q4 and Q8 serving
# packs) vs ninfer DFlash2 k=7 vs ninfer MTP3 (content control). Every leg
# serves the Anthropic Messages API on 0.0.0.0:8081 as a transient user unit,
# runs bench/swebench/run.sh over the 12 pinned instances, then stops. Both
# vox transcribers are stopped for the duration (host jitter) and production
# q27-38 is relaunched at the end.
#   LEGS="ninferd2" INSTANCE=psf__requests-1142 bash campaign.sh   # smoke
#   systemd-run --user --unit agentic-campaign bash campaign.sh    # full
set -u
Q=/mnt/ai/projects/q27
DIR=${CAMPAIGN_DIR:-$Q/bench/crossengine/agentic-2026-09-07}
MODEL=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27
TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
Q27=$Q/build/q27-server
Q27ARGS="--host 0.0.0.0 --port 8081 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0"
# Reasoning effort: Claude Code 2.1.170 sends output_config.effort (low |
# medium | high); ninfer passes the name to the Qwen3.8 template, which only
# exposes low | medium | xhigh, so 'high' is a 400 there; q27 ignores the
# field and renders its boot default (xhigh). The only effort reachable on
# BOTH engines is medium: run.sh pins CLAUDE_CODE_EFFORT_LEVEL=medium and the
# q27 legs render medium too. This is NOT production's xhigh -- say so.
# Q27_PRINT_WSUM so every leg's weight digest is in its journal: the 5090's
# pageable-DMA load corruption (~1%/load) is otherwise invisible in a
# campaign result. Added 2026-09-09; the 09-07/08/09 legs ran without it.
Q27ENV="-E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_REASONING_EFFORT=medium"
PACK4=/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q4-serve.d2w
PACK8=/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q8-serve.d2w
NINFER=/mnt/ai/projects/ninfer-master/build/apps/ninfer-serve
ART=/mnt/ai/models/ninfer/qwen38-nvfp4-release/qwen3_8_27b_nvfp4.ninfer
# same sampler chain as q27's agentic recipe (Claude Code sends temperature
# only; the rest comes from server defaults -- ninfer's thinking default has
# min_p 0, so pin it); thinking on (template default effort), no budget.
NARGS="--host 0.0.0.0 --port 8081 --kv-dtype int8 --kv-capacity 131072 --max-context 131072 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.05"
LEGS=${LEGS:-q27lad q27d2q4 q27d2q8 ninferd2 ninfermtp}
INSTANCE=${INSTANCE:-}
mkdir -p $DIR
log() { echo "[campaign $(date '+%H:%M:%S')] $*"; }

start_engine() { # $1 label
  local unit=$1-eval
  systemctl --user reset-failed $unit 2>/dev/null
  case "$1" in
    q27lad)   systemd-run --user --unit $unit $Q27ENV $Q27 $MODEL $TOK $Q27ARGS ;;
    q27d2q4)  systemd-run --user --unit $unit $Q27ENV -E Q27_BATCH=0 -E Q27_DFLASH2=$PACK4 $Q27 $MODEL $TOK $Q27ARGS ;;
    q27d2q8)  systemd-run --user --unit $unit $Q27ENV -E Q27_BATCH=0 -E Q27_DFLASH2=$PACK8 -E Q27_DFLASH2_RESERVE_GB=3 $Q27 $MODEL $TOK $Q27ARGS ;;
    # the production config (tools/launch_q27_38.sh d2-pfx: DFlash2 Q8 pack +
    # prefix-cache tiers on a FRESH tmpfs root, so the run pays the cold
    # bootstrap turns itself) at the campaign's effort pin (medium, the only
    # level both engines render -- production serves xhigh; see the 09-08
    # README for the xhigh numbers). Added 2026-09-08 for the v0.11.0 table.
    q27prod)  rm -rf /dev/shm/q27-pfx-campaign; mkdir -p /dev/shm/q27-pfx-campaign
              systemd-run --user --unit $unit $Q27ENV -E Q27_BATCH=0 -E Q27_DFLASH2=$PACK8 -E Q27_DFLASH2_RESERVE_GB=3 -E Q27_SYSBLK=1 $Q27 $MODEL $TOK $Q27ARGS \
                --prefix-cache /dev/shm/q27-pfx-campaign --prefix-cache-max-gb 40 --prefix-cache-ram-gb 0 --prefix-cache-max-tokens 65536 ;;
    # 2026-09-09 model-echo A/B (turn-count investigation): the q27prod config
    # with /v1/messages echoing the client's model name (server.cu resp_model)
    # so Claude Code keeps prior thinking blocks in the history it sends back.
    # q27noecho = the same binary with Q27_ECHO_MODEL=0 (the 09-09 q27prod
    # leg's wire behaviour) as the same-day control. Own pfx root each.
    q27echo|q27noecho)
              rm -rf /dev/shm/q27-pfx-$1; mkdir -p /dev/shm/q27-pfx-$1
              systemd-run --user --unit $unit $Q27ENV -E Q27_BATCH=0 -E Q27_DFLASH2=$PACK8 -E Q27_DFLASH2_RESERVE_GB=3 -E Q27_SYSBLK=1 \
                -E Q27_ECHO_MODEL=$([ "$1" = q27echo ] && echo 1 || echo 0) $Q27 $MODEL $TOK $Q27ARGS \
                --prefix-cache /dev/shm/q27-pfx-$1 --prefix-cache-max-gb 40 --prefix-cache-ram-gb 0 --prefix-cache-max-tokens 65536 ;;
    ninferd2) systemd-run --user --unit $unit $NINFER $ART $NARGS --spec dflash2 --draft-tokens 7 --request-log-jsonl $DIR/$1.reqlog.jsonl ;;
    ninfermtp) systemd-run --user --unit $unit $NINFER $ART $NARGS --spec mtp --draft-tokens 3 --request-log-jsonl $DIR/$1.reqlog.jsonl ;;
    *) log "unknown leg $1"; return 1 ;;
  esac
  for i in $(seq 1 150); do
    case "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health)" in 200|401) return 0 ;; esac
    systemctl --user is-active --quiet $unit || { log "$unit died during startup"; journalctl --user -u $unit --no-pager -o cat | tail -5; return 1; }
    sleep 2
  done
  log "$unit never became healthy"; return 1
}
stop_engine() { systemctl --user stop $1-eval 2>/dev/null; sleep 3; }

systemctl --user stop q27-38 q27-d2test 2>/dev/null
sudo -n systemctl stop vox-transcriber vox-transcriber-gmrs 2>/dev/null
sleep 3
log "vox: $(systemctl is-active vox-transcriber) $(systemctl is-active vox-transcriber-gmrs); legs: $LEGS; instance filter: '${INSTANCE:-all}'"
for leg in $LEGS; do
  log "=== leg $leg ==="
  if ! start_engine $leg; then log "leg $leg SKIPPED (engine failed)"; stop_engine $leg; continue; fi
  export SWEBENCH_UNIT=$leg-eval SWEBENCH_HOST=127.0.0.1
  case "$leg" in ninfer*) export SWEBENCH_REQLOG=$DIR/$leg.reqlog.jsonl ;; *) unset SWEBENCH_REQLOG ;; esac
  bash $Q/bench/swebench/run.sh $leg $INSTANCE 2>&1 | tee $DIR/$leg.log
  cp $Q/bench/swebench/results.$leg.jsonl $DIR/ 2>/dev/null
  cp $Q/bench/swebench/swebench_$leg.journal $DIR/ 2>/dev/null
  stop_engine $leg
done
sudo -n systemctl start vox-transcriber vox-transcriber-gmrs 2>/dev/null
systemctl --user reset-failed q27-38 2>/dev/null
# production config lives in ONE place: tools/launch_q27_38.sh (d2-pfx since
# 2026-09-08 evening = DFlash2 Q8 pack + prefix-cache tiers; BUILDLOG (e), (g)).
# Relaunching with an inline systemd-run here silently reverted production to
# a cache-less config once.
bash $Q/tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1
log "done; production relaunched; vox: $(systemctl is-active vox-transcriber) $(systemctl is-active vox-transcriber-gmrs)"
echo "CAMPAIGN DONE" > $DIR/DONE
