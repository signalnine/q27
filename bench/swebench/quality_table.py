#!/usr/bin/env python3
"""Task-quality table across SWE-bench campaign arms (item 2): per arm, the
single-pass proxies (nonempty, gold_hit) and the harness result, plus
per-instance agreement so a speed difference can be read against what the
runs actually did. usage: quality_table.py <results.jsonl>..."""
import json, os, sys
from collections import Counter
arms = {}
for p in sys.argv[1:]:
    name = os.path.basename(p).replace("results.", "").replace(".jsonl", "")
    arms[name] = {r["iid"]: r for r in (json.loads(l) for l in open(p))}
print(f"{'arm':16s} {'n':>2s} {'nonempty':>8s} {'gold':>5s} {'success':>7s} {'other results':22s} {'turns':>6s} {'out_tok':>8s} {'wall_s':>7s} {'diff_ln':>7s}")
for name, rows in arms.items():
    rs = list(rows.values()); n = len(rs)
    res = Counter(r["result"] for r in rs)
    other = ",".join(f"{k}:{v}" for k, v in res.items() if k != "success")
    print(f"{name:16s} {n:2d} {sum(r['nonempty'] for r in rs):8d} {sum(r['gold_hit'] for r in rs):5d} "
          f"{res.get('success', 0):7d} {other:22s} {sum(r['turns'] for r in rs) / n:6.1f} {sum(r['out_tok'] for r in rs) / n:8.0f} "
          f"{sum(r['wall_s'] for r in rs) / n:7.0f} {sum(r['diff_lines'] for r in rs) / n:7.1f}")
iids = sorted(set.intersection(*(set(a) for a in arms.values())))
print(f"\nper instance (gold-hit G / nonempty-only N / empty . ; turns) over {len(iids)} shared instances:")
print(f"{'iid':28s} " + " ".join(f"{a[:10]:>10s}" for a in arms))
for i in iids:
    cells = []
    for a in arms.values():
        r = a[i]
        m = "G" if r["gold_hit"] else ("N" if r["nonempty"] else ".")
        cells.append(f"{m}{r['turns']:>3d}/{r['out_tok'] // 1000:>2d}k")
    print(f"{i:28s} " + " ".join(f"{c:>10s}" for c in cells))
