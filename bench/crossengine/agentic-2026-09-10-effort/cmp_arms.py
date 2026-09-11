#!/usr/bin/env python3
"""Thinking-length comparison across probe arms: median with bootstrap 95% CI,
p90, and two-sided Mann-Whitney against a reference arm.
usage: cmp_arms.py <ref_label> <label=path.jsonl> ..."""
import json, random, sys
from scipy.stats import mannwhitneyu
ref = sys.argv[1]; arms = dict(a.split("=", 1) for a in sys.argv[2:])
data = {k: [json.loads(l)["think"] for l in open(p) if l.strip()] for k, p in arms.items()}
def med(x): s = sorted(x); n = len(s); return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
def ci(x, B=4000):
    r = random.Random(7); m = sorted(med([r.choice(x) for _ in x]) for _ in range(B))
    return m[int(.025 * B)], m[int(.975 * B)]
def p90(x): s = sorted(x); return s[min(len(s) - 1, int(.9 * len(s)))]
print(f"{'arm':28s} {'n':>3s} {'mean':>5s} {'median [95% CI]':>18s} {'p90':>5s} {'max':>5s}  vs {ref}")
for k, x in data.items():
    lo, hi = ci(x); p = mannwhitneyu(x, data[ref]).pvalue if k != ref else float('nan')
    print(f"{k:28s} {len(x):3d} {sum(x)/len(x):5.0f} {med(x):6.0f} [{lo:4.0f}, {hi:4.0f}] {p90(x):5d} {max(x):5d}  p={p:.4f}")
