#!/usr/bin/env python3
"""Paired comparison of replay_think.py arms over the same recorded bodies:
render parity (prompt tokens per body), per-body mean thinking chars, a
Wilcoxon signed-rank test of each arm against the reference arm, the
geometric-mean ratio, and the next-action mix (first tool called).
usage: replay_cmp.py <ref_label> <label=path.jsonl> ..."""
import json, math, sys
from collections import Counter, defaultdict
from scipy.stats import wilcoxon
ref = sys.argv[1]; arms = dict(a.split("=", 1) for a in sys.argv[2:])
per = {}   # arm -> seq -> [rows]
for k, p in arms.items():
    d = defaultdict(list)
    for l in open(p):
        if l.strip(): r = json.loads(l); d[r["seq"]].append(r)
    per[k] = d
seqs = sorted(set.intersection(*(set(d) for d in per.values())))
print(f"{len(seqs)} bodies common to all arms")
print("render parity (prompt tokens, arm minus ref): " + ", ".join(
    f"{k}: median {sorted((per[k][s][0]['prompt'] or 0) - (per[ref][s][0]['prompt'] or 0) for s in seqs)[len(seqs)//2]:+d}"
    for k in per if k != ref))
mean = lambda xs: sum(xs) / len(xs)
bm = {k: {s: mean([r["think"] for r in per[k][s]]) for s in seqs} for k in per}
print(f"\n{'arm':12s} {'samples':>7s} {'think mean':>10s} {'median of body means':>20s} {'geo ratio vs ' + ref:>18s} {'wilcoxon p':>10s} {'bodies arm>ref':>14s}")
for k in per:
    xs = [bm[k][s] for s in seqs]; ys = [bm[ref][s] for s in seqs]
    allr = [r["think"] for s in seqs for r in per[k][s]]
    gr = math.exp(mean([math.log((x + 50) / (y + 50)) for x, y in zip(xs, ys)]))
    p = wilcoxon(xs, ys).pvalue if k != ref else float("nan")
    print(f"{k:12s} {len(allr):7d} {mean(allr):10.0f} {sorted(xs)[len(xs)//2]:20.0f} {gr:18.2f} {p:10.4f} {sum(x > y for x, y in zip(xs, ys)):>8d}/{len(seqs)}")
print("\nnext action (first tool called, share of samples):")
for k in per:
    c = Counter((r["tools"][0] if r["tools"] else "(none)") for s in seqs for r in per[k][s])
    n = sum(c.values())
    print(f"  {k:12s} " + "  ".join(f"{t} {v / n:.0%}" for t, v in c.most_common(6)))
print("\nper body (mean thinking chars; prompt tokens of the ref arm):")
print(f"  {'seq':>4s} {'msgs':>4s} {'prompt':>6s} " + " ".join(f"{k:>10s}" for k in per))
for s in seqs:
    r0 = per[ref][s][0]
    print(f"  {s:4d} {r0.get('msgs', -1):4d} {r0['prompt'] or 0:6d} " + " ".join(f"{bm[k][s]:10.0f}" for k in per))
