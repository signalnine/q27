#!/usr/bin/env bash
# Serving-side width sweep (item 1, step 2): production config (Q8 serving
# pack, MMA verify at width K+1, sampled walk, ring retention, fold overlap),
# one boot per K, interleaved 7/10 first so drift between boots is visible.
# Per boot: seeded streams pass A + pass B (within-arm repeatability), then
# decode-at-depth probes at 12.5K and 50K. The [req] and [d2timing] journal
# lines of each boot are saved for the comparison.
# usage: width_serve.sh <outdir> [K list]
#   env SEEDED=0 skips the seeded passes; DEPTHS="25000 50000" DSEEDS=8 shape
#   the decode-at-depth probes (defaults: 12500 50000, 2 seeds)
set -u
Q=$(cd "$(dirname "$0")/../../.." && pwd); S=$(dirname "$0")
OUT=$1; shift
KS=${*:-"7 10 7 10 8 9 11"}
PACK8=/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q8-serve.d2w
MODEL=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27; TOK=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok
ARGS="--host 172.17.0.1 --port 8081 --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0"
B=http://172.17.0.1:8081
mkdir -p "$OUT"
systemctl --user stop q27-38 2>/dev/null
trap 'systemctl --user stop q27-ab 2>/dev/null; echo "=== relaunch production $(date +%T)"; bash $Q/tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1 2>&1 | tail -1' EXIT
i=0
for K in $KS; do
  i=$((i+1)); tag=k$K.b$i
  echo "=== $tag $(date +%T)"
  systemctl --user reset-failed q27-ab 2>/dev/null
  systemd-run --user --unit q27-ab -E Q27_KV=fp8 -E Q27_PRINT_WSUM=1 -E Q27_BATCH=0 -E Q27_DFLASH2=$PACK8 -E Q27_DFLASH2_RESERVE_GB=3 -E Q27_D2_TIMING=1 -E Q27_DFLASH2_K=$K $Q/build/q27-server $MODEL $TOK $ARGS >/dev/null 2>&1
  inv=$(systemctl --user show q27-ab -p InvocationID --value)
  ok=0
  for w in $(seq 1 120); do
    journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager 2>/dev/null | grep -q 'listening on' && { ok=1; break; }
    systemctl --user is-active --quiet q27-ab || break
    sleep 2
  done
  if [ $ok = 0 ]; then echo "$tag: server did not come up"; journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager | tail -3; continue; fi
  if [ "${SEEDED:-1}" = 1 ]; then
    python3 $S/seeded_hash.py $B ${tag}.A > $OUT/$tag.seededA.txt 2>&1
    python3 $S/seeded_hash.py $B ${tag}.B > $OUT/$tag.seededB.txt 2>&1
  fi
  for dp in ${DEPTHS:-12500 50000}; do
    python3 $S/width_depth_probe.py $B $tag $dp ${DSEEDS:-2} > $OUT/$tag.depth$dp.txt 2>&1
  done
  sleep 1
  journalctl --user _SYSTEMD_INVOCATION_ID=$inv -o cat --no-pager | grep -E '^\[req\]|^\[d2timing\]|serving ON' > $OUT/$tag.journal.txt
  systemctl --user stop q27-ab
  echo "$tag: $(grep -c '^\[req\]' $OUT/$tag.journal.txt) requests, $(grep -c '^\[d2timing\]' $OUT/$tag.journal.txt) timing lines"
done
echo "=== serve sweep done $(date +%T)"
