#!/usr/bin/env bash
# Boot a production-config server on a FRESH cache root, run the promotion
# probe (shape A then B), save the journal + root listing.
# usage: pfx_promote_run.sh <outdir> <binary> [tag]   (pair control and fix on the SAME tag: the filler embeds it)
set -u
Q=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$1; BIN=$2; TAG=${3:-promo}
ROOT=/dev/shm/q27-pfx-probe
MODEL=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27; TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
PACK8=/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q8-serve.d2w
ARGS="--host 172.17.0.1 --port 8081 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0"
PFXARGS="--prefix-cache $ROOT --prefix-cache-max-gb 40 --prefix-cache-ram-gb 0 --prefix-cache-max-tokens 65536"
B=http://172.17.0.1:8081
mkdir -p "$OUT"; rm -rf "$ROOT"; mkdir -p "$ROOT"
systemctl --user stop q27-38 2>/dev/null
trap 'systemctl --user stop q27-ab 2>/dev/null; echo "=== relaunch production $(date +%T)"; bash $Q/tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1 2>&1 | tail -1' EXIT
systemctl --user reset-failed q27-ab 2>/dev/null
systemd-run --user --unit q27-ab -E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_BATCH=0 -E Q27_DFLASH2=$PACK8 -E Q27_DFLASH2_RESERVE_GB=3 -E Q27_SYSBLK=1 $BIN $MODEL $TOK $ARGS $PFXARGS >/dev/null 2>&1
inv=$(systemctl --user show q27-ab -p InvocationID --value)
for w in $(seq 1 120); do
  journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager 2>/dev/null | grep -q 'listening on' && break
  systemctl --user is-active --quiet q27-ab || { echo "server died"; journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager | tail -3; exit 1; }
  sleep 2
done
echo "=== shape A $(date +%T)"; python3 $Q/bench/ladder/pfx_promote_probe.py $B ${TAG}A A
echo "=== shape B $(date +%T)"; python3 $Q/bench/ladder/pfx_promote_probe.py $B ${TAG}B B
sleep 3
journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager | grep -E '^\[pfx\]|^\[gen\]|^\[req\]|^\[sysblk\]|prefix-cache' > $OUT/journal.txt
ls -la --time-style=+%H:%M:%S $ROOT > $OUT/root.txt
echo "=== entries:"; awk '{print $5, $6, $7}' $OUT/root.txt | grep q27pc
echo "=== [req] hit/pf per request:"; grep '^\[req\]' $OUT/journal.txt | awk '{for(i=1;i<=NF;i++) if($i~/^(rid|prompt|hit|pf|pf_ms)=/) printf "%s ", $i; print ""}'
echo "=== [pfx] lines:"; grep '^\[pfx\]' $OUT/journal.txt | cut -c1-120
echo "=== done $(date +%T)"
