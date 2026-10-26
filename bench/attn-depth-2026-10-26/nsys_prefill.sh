#!/usr/bin/env bash
# nsys kernel-time attribution of a cold 65484-token prefill (fp8 KV, production path).
# No perf counters needed (ncu is blocked for non-root here: ERR_NVGPUCTRPERM).
set -euo pipefail
cd /mnt/ai/projects/q27-master
OUT=bench/attn-depth-2026-10-26/nsys_prefill_63k
rm -f "$OUT.nsys-rep" "$OUT.sqlite"
export CUDA_VISIBLE_DEVICES=0 Q27_KV=fp8 Q27_PRINT_WSUM=1 Q27_PF_NOSERIAL=1
/usr/local/cuda/bin/nsys profile -t cuda --cuda-graph-trace=node -o "$OUT" --force-overwrite true \
  ./build/q27 /mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27 \
    --tokens-file bench/attn-depth-2026-10-26/ids_65536.txt --pf 65484 --ctx 70000 \
  > "$OUT.stdout" 2> "$OUT.stderr"
/usr/local/cuda/bin/nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv -o "$OUT" "$OUT.nsys-rep" > "$OUT.stats.log" 2>&1
/usr/local/cuda/bin/nsys stats --report cuda_gpu_trace --format csv -o "$OUT" "$OUT.nsys-rep" >> "$OUT.stats.log" 2>&1
echo DONE > bench/attn-depth-2026-10-26/NSYS_DONE
