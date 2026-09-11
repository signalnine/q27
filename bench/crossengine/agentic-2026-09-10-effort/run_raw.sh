#!/usr/bin/env bash
# Raw-prompt arms: q27 untrimmed (control: must reproduce the Anthropic-path
# replay), q27 template-trimmed, llama.cpp Q8_0 on trimmed and untrimmed.
# Byte-identical prompts per body across engines. vox stopped for the llama
# arms (5090+3090 split) and restarted after; production relaunched at the end.
set -u
S=/tmp/claude-1000/-home-gabe/22610e4a-d806-40af-82cc-4755ef136fd9/scratchpad
B=$S/bodies; R=$B/rendered; Q=/mnt/ai/projects/q27; D=/mnt/ai/projects/q27-master/bench/crossengine/agentic-2026-09-09-echo
MODEL=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27; TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
PACK8=/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q8-serve.d2w
Q27ARGS="--host 0.0.0.0 --port 8081 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0"
LS=/mnt/ai/projects/llama.cpp/build/bin/llama-server; G8=/mnt/ai/models/qwen38-27b-mtp-gguf/Qwen3.8-27B-MTP-Q8_0.gguf
N=${N:-4}; SEL=$B/select.txt; ARMS=${ARMS:-"q27 llama"}
log() { echo "[raw $(date '+%H:%M:%S')] $*"; }
portfree() { for i in $(seq 1 90); do [ -z "$(ss -tanH '( sport = :8081 )')" ] && return 0; sleep 1; done; log "8081 still has sockets"; }
up() { for i in $(seq 1 300); do case "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health)" in 200|401) return 0;; esac; systemctl --user is-active --quiet $1 || return 1; sleep 2; done; return 1; }
busy=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | /usr/bin/grep GPU-e592b842 | /usr/bin/grep -v "$Q/build/q27-server" || true)
[ -n "$busy" ] && { log "5090 busy: $busy"; exit 3; }
systemctl --user stop q27-38; sleep 3; portfree
first=$(head -1 $SEL)
for arm in $ARMS; do
  unit=raw-$arm; systemctl --user reset-failed $unit 2>/dev/null
  if [ $arm = q27 ]; then
    systemd-run --user --unit $unit -E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_BATCH=0 -E Q27_DFLASH2=$PACK8 -E Q27_DFLASH2_RESERVE_GB=3 $Q/build/q27-server $MODEL $TOK $Q27ARGS >/dev/null 2>&1
    mode=q27
  else
    sudo -n systemctl stop vox-transcriber vox-transcriber-gmrs; sleep 3
    b3=$(nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | /usr/bin/grep GPU-5a723c5e || true)
    if [ -n "$b3" ]; then log "3090 busy: $b3 -- skipping llama"; sudo -n systemctl start vox-transcriber vox-transcriber-gmrs; continue; fi
    systemd-run --user --unit $unit -E CUDA_VISIBLE_DEVICES=0,1 $LS -m $G8 -ngl 999 -c 131072 -np 1 --host 0.0.0.0 --port 8081 -fa on --cache-prompt >/dev/null 2>&1
    mode=llama
  fi
  inv=$(systemctl --user show $unit -p InvocationID --value)
  if up $unit; then
    log "=== $arm up"
    if [ $mode = llama ]; then
      python3 - $R/t_$first.txt $R/counts.txt $first <<'PY'
import json, sys, urllib.request
t = open(sys.argv[1]).read(); want = {l.split()[0]: int(l.split()[2]) for l in open(sys.argv[2])}[sys.argv[3]]
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8081/tokenize", data=json.dumps({"content": t}).encode(), headers={"content-type": "application/json"})))
print(f"  token parity: llama tokenizes t_{sys.argv[3]} to {len(r['tokens'])} (q27 untrimmed render of the same body: {want})")
PY
    fi
    python3 - http://127.0.0.1:8081 $mode $R/t_$first.txt <<'PY'
import json, sys, urllib.request
base, mode, p = sys.argv[1:4]; t = open(p).read()
b = dict(prompt=t, temperature=1.0, top_p=0.95, top_k=20, min_p=0.05, seed=1, stream=False)
if mode == "q27": url = "/v1/completions"; b["max_tokens"] = 256
else: url = "/completion"; b["n_predict"] = 256
r = json.load(urllib.request.urlopen(urllib.request.Request(base + url, data=json.dumps(b).encode(), headers={"content-type": "application/json", "authorization": "Bearer local"})))
txt = r["choices"][0]["text"] if mode == "q27" else r["content"]
print("  smoke:", repr(txt[:160]))
PY
    for pre in t p; do
      [ $mode = q27 ] || [ $pre = t ] || [ "${LLAMA_P:-1}" = 1 ] || continue
      log "--- $arm prefix $pre"
      python3 $D/raw_think.py http://127.0.0.1:8081 raw_${arm}_$pre $mode $R $pre $SEL $N 16384 2>&1 | tail -2
    done
  else
    log "$arm never came up"; journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager | tail -4
  fi
  journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager | /usr/bin/grep -m1 -o 'wsum: [0-9a-f]*' | sed 's/^/  /'
  systemctl --user stop $unit; sleep 3; portfree
  [ $mode = llama ] && sudo -n systemctl start vox-transcriber vox-transcriber-gmrs
done
bash $Q/tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1 | tail -1
log "vox: $(systemctl is-active vox-transcriber) $(systemctl is-active vox-transcriber-gmrs)"
log "RAW-DONE"
