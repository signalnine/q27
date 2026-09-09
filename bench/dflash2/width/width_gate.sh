#!/usr/bin/env bash
# Width>8 investigation, item 1 of the 2026-09-08 (p) agenda.
# Matched-family greedy identity gate: for each prompt, a plain greedy reference
# (GEMV family, width 1) and the ladder (--spec) reference; then DFlash2 at
# K in KS (verify width K+1) in two arms:
#   gemv : Q27_GEMM_MIN=99 (every width stays on the GEMV family; needs --spec
#          so build_spec_graphs parses the env)
#   dflt : gemm_min 9 (widths >= 9 take the MMA family) -- same flags, env differs
# each arm with the verify graph (default) and eager (Q27_D2_NOGRAPH=1).
# usage: width_gate.sh <outdir> <mode: smoke|full> [n_gen]
set -u
Q=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=$1; MODE=${2:-smoke}; NGEN=${3:-512}
MODEL=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27
PACK=/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q8.d2w
mkdir -p "$OUT"
case $MODE in
  smoke) PROMPTS="prose"; KS="7 8"; NGS="graph" ;;
  full)  PROMPTS="prose code-write code-edit echo"; KS="1 6 7 8 9 10 11"; NGS="graph nograph" ;;
  *) echo "mode smoke|full" >&2; exit 2 ;;
esac
systemctl --user stop q27-38 2>/dev/null
trap 'echo "=== relaunch production $(date +%T)"; bash $Q/tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1 2>&1 | tail -1' EXIT
run() { # $1=tag $2..=env K=V ... -- args
  local tag=$1; shift
  local envs=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  local t0=$(date +%s)
  env Q27_KV=fp8 Q27_PRINT_WSUM=1 Q27_SUFFIX=0 Q27_SAMPLED=0 "${envs[@]}" $Q/build/q27 $MODEL "$@" > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  local rc=$?
  printf '%-32s rc=%d %4ds wsum=%s  %s\n' "$tag" $rc $(( $(date +%s) - t0 )) "$(grep -o 'wsum: [0-9a-f]*' "$OUT/$tag.err" | cut -c7-14)" "$(tail -1 "$OUT/$tag.out" | cut -c1-80)"
}
for p in $PROMPTS; do
  TF=$Q/bench/dflash2/toks/toks_$p.txt
  COMMON="--tokens-file $TF -n $NGEN --ctx 2048 --fast-head"
  run $p.plain -- $COMMON
  run $p.ladder -- $COMMON --spec
  for K in $KS; do
    for ng in $NGS; do
      NGE=(); [ $ng = nograph ] && NGE=(Q27_D2_NOGRAPH=1)
      run $p.gemv.k$K.$ng Q27_GEMM_MIN=99 "${NGE[@]}" -- $COMMON --spec --dflash2 $PACK --k $K
      run $p.dflt.k$K.$ng "${NGE[@]}" -- $COMMON --spec --dflash2 $PACK --k $K
    done
  done
done
echo "=== sweep done $(date +%T)"
