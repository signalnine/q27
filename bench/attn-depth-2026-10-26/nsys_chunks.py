#!/usr/bin/env python3
"""Attribute a cold 64K prefill from the nsys GPU trace: kernel time by family for the whole
run and for the deepest 1024-token chunk (the span between the 1008th and 1024th attention
launches, i.e. chunk 64 at ~63K depth)."""
import csv, sys, collections, re
trace = sys.argv[1] if len(sys.argv) > 1 else "nsys_prefill_63k_cuda_gpu_trace.csv"
rows = []
with open(trace, newline="") as f:
    rd = csv.DictReader(f)
    for r in rd:
        try: rows.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"]))
        except (KeyError, ValueError): continue
rows.sort()
def fam(name):
    n = name
    if "attn_prefill" in n: return "attn_prefill"
    if "attn_fd" in n or "fdmma" in n: return "attn_decode"
    if "gemm" in n or "gemv" in n: return "gemm/gemv"
    if "delta" in n or "conv" in n or "gdn" in n or "ssm" in n: return "gdn/delta"
    if "norm" in n or "rms" in n: return "norm"
    if "quant" in n or "qx" in n: return "quant"
    return "other"
attn_idx = [i for i, (_, _, n) in enumerate(rows) if "attn_prefill" in n and "combine" not in n]
print(f"kernels {len(rows)}, attention launches {len(attn_idx)}, total GPU kernel time {sum(d for _,d,_ in rows)/1e9:.2f} s")
def report(sub, title):
    tot = sum(d for _, d, _ in sub)
    by = collections.Counter(); cnt = collections.Counter()
    for _, d, n in sub: by[fam(n)] += d; cnt[fam(n)] += 1
    print(f"\n== {title}: {tot/1e6:.1f} ms of kernel time ==")
    for k, v in by.most_common(): print(f"  {k:14} {v/1e6:9.2f} ms  {v/tot:6.1%}  ({cnt[k]} launches)")
    top = collections.Counter()
    for _, d, n in sub: top[re.sub(r'\(.*', '', n)[:70]] += d
    print("  top kernels:")
    for k, v in top.most_common(6): print(f"    {v/1e6:8.2f} ms  {k}")
report(rows, "whole 65484-token prefill (all depths)")
if len(attn_idx) >= 1024:
    a, b = attn_idx[1008], attn_idx[1024] if len(attn_idx) > 1024 else len(rows)
    last = rows[a:b]
    report(last, "deepest chunk (attention launches 1009-1024, depth ~63K)")
    first = rows[attn_idx[0]:attn_idx[16]]
    report(first, "first chunk (attention launches 1-16, depth ~0)")
