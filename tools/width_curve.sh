#!/usr/bin/env bash
# Suffix-leg wide-round curve: ms/round and ms/token as a function of the
# suffix verify width W. The engine-true answer to "does a wider round pay?"
#
# Rig (BUILDLOG 2026-07-10, the width-12 P3 instrument): run the server with
# the MTP ladder pinned shallow (Q27_MAXD=4) and the suffix width pinned to W,
# then feed it the OPEN-CUT echo payload. Every wide round is then a suffix
# round of exactly W lanes at ~100% acceptance, so sfxm/sfxn in the [req] line
# is a clean per-width round cost. (The payload must end mid-flow: greedy
# raw-completion of a cleanly-terminated prompt EOSes instantly.)
#
# Usage: tools/width_curve.sh <server-binary> <W> [<W> ...]
#   BENCH_GPU=0  MODEL=...  PORT=8099
set -u
BIN="${1:?usage: width_curve.sh <server-binary> <W>...}"; shift
MODEL="${MODEL:-/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27}"
TOK="${TOK:-/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.tok}"
PORT="${PORT:-8099}"
PAY="$(dirname "$0")/../scratchpad/accept_payload_echo.json"
[ -f "$PAY" ] || { echo "missing $PAY -- run tools/make_payloads.py" >&2; exit 1; }

printf '%-4s %10s %8s %9s %9s %8s\n' W ms/round tok/rnd ms/token t/s dec
for W in "$@"; do
  CUDA_VISIBLE_DEVICES="${BENCH_GPU:-0}" \
  Q27_KV=fp8 Q27_FD=mma Q27_MAXD=4 Q27_PMIN=0.5 Q27_SUFFIX=1 Q27_SUFFIX_W="$W" \
  Q27_PHASE_STATS=1 \
    "$BIN" "$MODEL" "$TOK" --port "$PORT" --ctx 32768 --no-think --fast-head \
    >/tmp/wc_$W.log 2>&1 &
  SRV=$!
  for _ in $(seq 120); do
    curl -s -m 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
  done
  # two passes: the first warms the prefix cache, the second is the measurement
  for _ in 1 2; do
    curl -s -m 300 -X POST "http://127.0.0.1:$PORT/v1/completions" \
      -H 'content-type: application/json' -d @"$PAY" >/dev/null 2>&1
  done
  # [req] carries sfxm=<ms in suffix rounds> and sfxn=<suffix rounds> PER REQUEST,
  # but sfx=<fired>,<tokens> is ENGINE-CUMULATIVE -- so with NPASS identical
  # (greedy, prefix-cached) passes the per-request suffix tokens are sfx_tok/NPASS.
  # (Cross-check: sfx_fired/NPASS == sfxn exactly.)
  NPASS=2
  line="$(grep '^\[req\]' /tmp/wc_$W.log | tail -1)"
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  sfxm=$(sed -nE 's/.*sfxm=([0-9.]+).*/\1/p' <<<"$line")
  sfxn=$(sed -nE 's/.*sfxn=([0-9]+).*/\1/p' <<<"$line")
  dec=$(sed -nE 's/.*dec=([0-9]+).*/\1/p' <<<"$line")
  tps=$(sed -nE 's/.*tps=([0-9.]+).*/\1/p' <<<"$line")
  # sfx=<fired>,<tokens>
  sfxt=$(sed -nE 's/.* sfx=[0-9]+,([0-9]+).*/\1/p' <<<"$line")
  python3 - "$W" "${sfxm:-0}" "${sfxn:-0}" "${sfxt:-0}" "${dec:-0}" "${tps:-0}" "$NPASS" <<'PY'
import sys
W, ms, n, sfxtok_cum, dec, tps, npass = sys.argv[1], *map(float, sys.argv[2:])
tok = sfxtok_cum / npass          # per-request suffix tokens
if n and tok:
    print(f"{W:<4} {ms/n:10.2f} {tok/n:8.2f} {ms/tok:9.3f} {tps:9.1f} {int(dec):8d}")
else:
    print(f"{W:<4} {'NO SUFFIX ROUNDS':>10}  (dec={int(dec)}, tps={tps})")
PY
done
