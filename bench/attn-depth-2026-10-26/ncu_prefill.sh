#!/usr/bin/env bash
# ncu attribution of the prefill attention kernel at ~63K depth, production KV (fp8).
# --pf prefills the 65484-token payload in 1024-chunks: 16 attn layers x 64 chunks =
# 1024 attention launches; skip 1000 so the profiled launches sit at the deepest chunks.
set -euo pipefail
cd /mnt/ai/projects/q27-master
NCU=/opt/nvidia/nsight-compute/2026.1.1/ncu
OUT=bench/attn-depth-2026-10-26/ncu_prefill_attn_63k
export CUDA_VISIBLE_DEVICES=0 Q27_KV=fp8 Q27_PRINT_WSUM=1 Q27_PF_NOSERIAL=1
$NCU --target-processes all -k regex:k_attn_prefill_mma --launch-skip 1000 --launch-count 2 \
  --section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats \
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
  -o "$OUT" -f \
  ./build/q27 /mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27 \
    --tokens-file bench/attn-depth-2026-10-26/ids_65536.txt --pf 65484 --ctx 70000 \
  > "$OUT.stdout" 2> "$OUT.stderr"
$NCU -i "$OUT.ncu-rep" --page details > "$OUT.txt" 2>&1 || true
echo DONE > bench/attn-depth-2026-10-26/NCU_DONE
