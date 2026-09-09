#!/usr/bin/env python3
"""Compare width_gate.sh outputs. Reference = the ladder (--spec) stream (the
width-N gemv family; the plain decode graph uses k_gemv_q4 != k_gemv_q4_n<1>,
engine.cuh:1709, so plain is NOT bitwise with any width-N verify). Prints the
first divergence vs the ladder and vs plain, plus a pairwise matrix.
usage: width_gate_cmp.py <outdir> [--matrix]"""
import glob, os, re, sys
d = sys.argv[1]
matrix = "--matrix" in sys.argv
def toks(path):
    for line in open(path):
        if line.startswith("generated:"):
            return [int(x) for x in line.split()[1:]]
    return None
def stats(path):
    s = open(path).read()
    m = re.search(r"= ([\d.]+) t/s \(([\d.]+) tokens/round over (\d+) rounds", s)
    return (float(m.group(1)), float(m.group(2)), int(m.group(3))) if m else None
def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None
def fmt(fd):
    return "SAME" if fd is None else f"@{fd}"
prompts = sorted({os.path.basename(f).split(".")[0] for f in glob.glob(f"{d}/*.ladder.out")})
for p in prompts:
    ref = toks(f"{d}/{p}.ladder.out")
    plain = toks(f"{d}/{p}.plain.out") if os.path.exists(f"{d}/{p}.plain.out") else None
    if ref is None:
        print(f"{p}: ladder reference missing"); continue
    print(f"== {p}: ladder {len(ref)} tokens; plain vs ladder: {fmt(first_diff(ref, plain)) if plain else 'n/a'}")
    arms = []
    for f in sorted(glob.glob(f"{d}/{p}.*.out"), key=lambda x: (len(x), x)):
        tag = os.path.basename(f)[len(p) + 1:-4]
        if tag in ("plain", "ladder"):
            continue
        t = toks(f)
        if t is None:
            err = open(f[:-4] + ".err").read().strip().splitlines()
            print(f"  {tag:24s} NO OUTPUT: {err[-1][:80] if err else '?'}"); continue
        arms.append((tag, t))
        st = stats(f)
        extra = f"  {st[0]:6.1f} t/s  {st[1]:.3f} tok/round  {st[2]:4d} rounds" if st else ""
        print(f"  {tag:24s} vs ladder {fmt(first_diff(ref, t)):8s} vs plain {fmt(first_diff(plain, t)) if plain else 'n/a':8s}{extra}")
    if matrix and arms:
        names = [a for a, _ in arms]
        print("  pairwise first divergence:")
        print("  " + " " * 24 + " ".join(f"{n[:10]:>10s}" for n in names))
        for a, ta in arms:
            print(f"  {a:24s}" + " ".join(f"{fmt(first_diff(ta, tb)):>10s}" for _, tb in arms))
