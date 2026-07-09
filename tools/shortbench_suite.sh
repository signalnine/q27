#!/usr/bin/env bash
# Short-bench SUITE: the SOTA-comparable short-context decode number.
#
# Why this exists: the single 5-token canonical prompt is a BITWISE GATE,
# not a benchmark. Its 128-token greedy trajectory sits on argmax ties that
# legitimate fp accumulation-order changes re-roll (measured: fd2 moved the
# single-prompt number 177.5 -> 160.2 t/s, -10%, while per-round cost moved
# +1.3%). A number that swings 10% on a tie re-roll cannot carry a
# cross-engine comparison. This suite spreads the number over 5 fixed,
# genre-diverse short prompts (code / technical prose / list / translation /
# comparative reasoning) so no single degenerate trajectory owns it, and
# reports the MEAN. The canonical prompt still runs first, as the gate.
#
# Protocol: STOCK clocks (verify offset 0 before trusting a headline), 128
# tokens, --ctx 2048, --spec, greedy, fp16 KV (CLI default). Prompt token
# ids were produced once with llama-tokenize --no-bos on the source GGUF
# (Jackrong/Qwopus3.6-27B-v2-MTP-GGUF) and are baked in so the suite needs
# no tokenizer at run time. Deterministic: same binary + GPU arch => same
# trajectories, so n=1 per prompt is exact.
#
# Usage: tools/shortbench_suite.sh [model.q27]
set -u
MODEL="${1:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27}"
BIN="$(dirname "$0")/../build/q27"
# Baseline canonical: vanilla qwen36-27b-mtp (the benchmark standard,
# 2026-07-09). Fine-tunes carry their own: CANON_MD5=<md5> env overrides
# (Qwopus fd2-era: 4c4120c72056aba2bc2d2561471eafce; Q27_FD=v1 -> 58b6ae85...).
CANON_MD5="${CANON_MD5:-a2982c5197c627551b27d76a0a94b220}"

CANON_IDS="760,6511,314,9338,369"
declare -a NAMES=(hash-table merge-sorted planets translate-fr tcp-vs-udp)
declare -a PROMPTS=(
  "814,20139,1204,264,5010,1898,13081,45776,321,948,1754,8024,14387,13"
  "7734,264,12654,709,421,78161,1330,10300,11140,1083,799,10300,1103,13"
  "826,279,7810,31784,314,279,12570,1785,440,799,6821,2029,883,1754,13"
  "26583,310,8323,25,561,8831,369,6037,3242,11,321,567,1220,4088,310,279,2981,1518,424,32147,13"
  "3710,513,279,6355,61782,1881,25804,321,40986,364,264,1865,7019,37446,1746,30"
)

run_one() { # $1=ids -> prints "tps t_per_round" on stdout, full output to $2
  CUDA_VISIBLE_DEVICES=${BENCH_GPU:-0} "$BIN" "$MODEL" --tokens "$1" -n 128 --ctx 2048 --spec >"$2" 2>&1
  sed -n 's/^spec decode:.*= \([0-9.]*\) t\/s (\([0-9.]*\) tokens\/round.*/\1 \2/p' "$2"
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "== canonical gate (bitwise, not part of the mean)"
read -r ctps ctpr < <(run_one "$CANON_IDS" "$tmp/canon.out")
md5="$(grep '^generated:' "$tmp/canon.out" | md5sum | cut -d' ' -f1)"
if [[ "$md5" == "$CANON_MD5" ]]; then
  echo "canonical: md5 OK ($md5)  ${ctps} t/s  ${ctpr} t/round"
else
  echo "canonical: MD5 MISMATCH got $md5 want $CANON_MD5" >&2
  exit 1
fi

echo "== suite (5 prompts, 128 tok each, stock, greedy --spec)"
sum=0
for i in "${!PROMPTS[@]}"; do
  read -r tps tpr < <(run_one "${PROMPTS[$i]}" "$tmp/p$i.out")
  [[ -z "${tps:-}" ]] && { echo "FAIL: no tps line for ${NAMES[$i]}" >&2; exit 1; }
  printf "%-14s %7.2f t/s  %.2f t/round\n" "${NAMES[$i]}" "$tps" "$tpr"
  sum=$(awk -v a="$sum" -v b="$tps" 'BEGIN{print a+b}')
done
mean=$(awk -v s="$sum" -v n="${#PROMPTS[@]}" 'BEGIN{printf "%.1f", s/n}')
echo "== suite mean: ${mean} t/s over ${#PROMPTS[@]} prompts"
