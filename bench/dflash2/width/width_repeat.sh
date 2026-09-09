#!/usr/bin/env bash
# Repeat the one MMA-arm outlier (code-write, K=10, verify graph diverged at
# token 93 from every other MMA run; its eager twin did not). Deterministic
# divergence = a capture bug at width 11; sporadic = nondeterminism.
# usage: width_repeat.sh <outdir>
set -u
Q=$(cd "$(dirname "$0")/../../.." && pwd); OUT=$1; mkdir -p "$OUT"
MODEL=/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27
PACK=/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q8.d2w
systemctl --user stop q27-38 2>/dev/null
trap 'echo "=== relaunch production $(date +%T)"; bash $Q/tools/launch_q27_38.sh d2-pfx -E Q27_SYSBLK=1 2>&1 | tail -1' EXIT
run() { local tag=$1; shift; local envs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  env Q27_KV=fp8 Q27_PRINT_WSUM=1 Q27_SUFFIX=0 Q27_SAMPLED=0 "${envs[@]}" $Q/build/q27 $MODEL "$@" > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  printf '%-30s rc=%d  %s\n' "$tag" $? "$(tail -1 "$OUT/$tag.out" | cut -c1-90)"; }
p=code-write; TF=$Q/bench/dflash2/toks/toks_$p.txt; COMMON="--tokens-file $TF -n 512 --ctx 2048 --fast-head"
run $p.ladder -- $COMMON --spec
for r in 1 2 3 4; do run $p.dflt.k10.graph.r$r -- $COMMON --spec --dflash2 $PACK --k 10; done
for r in 1 2; do run $p.dflt.k10.nograph.r$r Q27_D2_NOGRAPH=1 -- $COMMON --spec --dflash2 $PACK --k 10; done
for r in 1 2; do run $p.dflt.k8.graph.r$r -- $COMMON --spec --dflash2 $PACK --k 8; done
for r in 1 2; do run $p.gemv.k10.graph.r$r Q27_GEMM_MIN=99 -- $COMMON --spec --dflash2 $PACK --k 10; done
echo "=== repeat done $(date +%T)"
