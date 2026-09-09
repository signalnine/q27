#!/usr/bin/env python3
"""Summarise width_serve.sh boots: per boot and prompt class, tok/round, decode
t/s, mean dec_ms; the last cumulative [d2timing] line; stream shas across
passes and boots. usage: width_serve_cmp.py <outdir>"""
import glob, os, re, sys
from collections import defaultdict
d = sys.argv[1]
def kv(line):
    return {k: v for k, v in re.findall(r"(\w+)=([^\s]+)", line)}
def cls(prompt):
    p = int(prompt)
    return "s2k" if p < 4000 else "s6k" if p < 9000 else "d12k" if p < 20000 else "d25k" if p < 40000 else "d50k"
boots = sorted(glob.glob(f"{d}/k*.b*.journal.txt"), key=lambda f: int(re.search(r"\.b(\d+)\.", f).group(1)))
print(f"{'boot':10s} {'class':5s} {'n':>2s} {'tok/rnd':>8s} {'dec t/s':>8s} {'dec_ms':>7s} {'rounds':>7s}")
for jf in boots:
    tag = os.path.basename(jf).split(".journal")[0]
    reqs = [kv(l) for l in open(jf) if l.startswith("[req]")]
    reqs = [r for r in reqs if int(r.get("dec", 0)) > 0]
    by = defaultdict(list)
    for r in reqs:
        by[cls(r["prompt"])].append(r)
    by["ALL"] = reqs
    for c in ("s2k", "s6k", "d12k", "d25k", "d50k", "ALL"):
        rs = by.get(c, [])
        if not rs:
            continue
        dec = sum(int(r["dec"]) for r in rs); rnd = sum(int(r["rounds"]) for r in rs)
        ms = sum(float(r["dec_ms"]) for r in rs)
        print(f"{tag:10s} {c:5s} {len(rs):2d} {dec / max(rnd, 1):8.3f} {dec / max(ms, 1) * 1000:8.1f} {ms / len(rs):7.0f} {rnd / len(rs):7.1f}")
    tl = [l.strip() for l in open(jf) if l.startswith("[d2timing]")]
    if tl:
        print(f"{'':10s} {tl[-1]}")
print()
print("stream shas per (tgt, seed): boot.pass -> sha (identical across boots = width-invariant target numerics)")
shas = defaultdict(dict)
for f in sorted(glob.glob(f"{d}/k*.b*.seeded?.txt")) + sorted(glob.glob(f"{d}/k*.b*.depth*.txt")):
    for l in open(f):
        m = kv(l)
        if "sha" not in m:
            continue
        key = (m.get("tgt") or ("depth" + m["depth"]), m["seed"])
        col = os.path.basename(f).split(".seeded")[0] + ("." + f[-5] if ".seeded" in f else "")
        shas[key][col] = (m["sha"], m["gen"])
cols = sorted({c for v in shas.values() for c in v}, key=lambda c: (int(re.search(r"\.b(\d+)", c).group(1)), c))
print(f"{'key':18s} " + " ".join(f"{c:>12s}" for c in cols))
for key in sorted(shas, key=lambda k: (k[0], int(k[1]))):
    row = shas[key]
    print(f"{key[0] + '/' + key[1]:18s} " + " ".join(f"{row[c][0][:8] + '/' + row[c][1]:>12s}" if c in row else f"{'-':>12s}" for c in cols))
