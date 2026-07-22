#!/usr/bin/env bash
# Sampling Phase 2 LIVE gates (docs/sampling-phase2-impl.md sec Gates).
# Runs on GPU 0 and loads the 17.7GB model, so free GPU 0 first (stop any
# resident q27-server). Kernel-level correctness is already proven by
# test_kernels --sampling-only; these confirm the end-to-end engine plumbing.
#
# Usage: tools/sampling_gate.sh [model.q27]
set -u
MODEL="${1:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27}"
BIN="$(dirname "$0")/../build/q27"
# baseline greedy canonical: vanilla qwen36-27b-mtp (benchmark standard
# 2026-07-09); other tiers/fine-tunes override via CANON_MD5= env --
#   q4s: f64e7c02252ca4c40cea62db662205e0
#   q5f: 683f7f4450ca4c60837abdb603ee3237  (Q4-head + ffn_down, 5.30bpw)
#   Qwopus: 4c4120c7...
CANON_MD5="${CANON_MD5:-a2982c5197c627551b27d76a0a94b220}"
CANON_IDS="760,6511,314,9338,369"
export CUDA_VISIBLE_DEVICES=0
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
fail=0

gen() { # $1=outfile ; extra args after
  local out="$1"; shift
  "$BIN" "$MODEL" --tokens "$CANON_IDS" --ctx 2048 --spec "$@" >"$out" 2>>"$tmp/err.log"
  grep '^generated:' "$out"
}

echo "== gate 1: greedy canonical md5 (bitwise -- greedy path must be untouched)"
gen "$tmp/canon.out" -n 128 >/dev/null
md5="$(grep '^generated:' "$tmp/canon.out" | md5sum | cut -d' ' -f1)"
if [[ "$md5" == "$CANON_MD5" ]]; then echo "  OK  md5=$md5"
else echo "  FAIL md5=$md5 want $CANON_MD5" >&2; fail=1; fi

echo "== gate 2: sampled seeded identity (same seed -> identical stream)"
a="$(gen "$tmp/s1.out" -n 48 --temp 0.85 --top-p 0.95 --seed 42)"
b="$(gen "$tmp/s2.out" -n 48 --temp 0.85 --top-p 0.95 --seed 42)"
if [[ "$a" == "$b" && -n "$a" ]]; then echo "  OK  (identical across runs)"
else echo "  FAIL seeded runs differ" >&2; fail=1; fi

echo "== gate 3: seed varies + sampled != greedy (sanity)"
c="$(gen "$tmp/s3.out" -n 48 --temp 0.85 --top-p 0.95 --seed 7)"
g="$(grep '^generated:' "$tmp/canon.out" | head -c 400)"
[[ "$a" != "$c" ]] && echo "  OK  seed 42 != seed 7" || { echo "  FAIL seeds gave identical output" >&2; fail=1; }
[[ "$a" != "$g" ]] && echo "  OK  sampled != greedy" || { echo "  FAIL sampled == greedy" >&2; fail=1; }

echo "== gate 4: spec==plain trajectories both valid (full chi-square is kernel-proven)"
sp="$(gen "$tmp/spec.out" -n 48 --temp 0.85 --top-p 0.95 --seed 3)"
pl="$(Q27_SAMPLE_PLAIN=1 gen "$tmp/plain.out" -n 48 --temp 0.85 --top-p 0.95 --seed 3)"
ns="$(wc -w <<<"$sp")"; np="$(wc -w <<<"$pl")"   # word count includes the 'generated:' token
if [[ "$ns" -ge 40 && "$np" -ge 40 ]]; then echo "  OK  spec=$((ns-1)) tok, plain=$((np-1)) tok (both produced; distributions match by kernel gate)"
else echo "  FAIL a path under-produced (spec=$ns plain=$np words)" >&2; fail=1; fi

echo "== gate 5: acceptance-vs-temp (tokens/round should sag as T rises)"
for T in 0.0 0.3 0.7 1.0 1.5; do
  if [[ "$T" == "0.0" ]]; then
    "$BIN" "$MODEL" --tokens "$CANON_IDS" -n 96 --ctx 2048 --spec >"$tmp/t.out" 2>/dev/null
    tag="greedy"
  else
    "$BIN" "$MODEL" --tokens "$CANON_IDS" -n 96 --ctx 2048 --spec --temp "$T" --top-p 0.95 --seed 1 >"$tmp/t.out" 2>/dev/null
    tag="T=$T"
  fi
  tpr="$(sed -n 's/.*= [0-9.]* t\/s (\([0-9.]*\) tokens\/round.*/\1/p' "$tmp/t.out")"
  printf "  %-8s %s tokens/round\n" "$tag" "${tpr:-?}"
done

echo ""
[[ "$fail" == 0 ]] && echo "SAMPLING GATES: ALL PASS" || { echo "SAMPLING GATES: FAILED"; cat "$tmp/err.log" >&2; }
exit "$fail"
