#!/usr/bin/env python3
"""Decode acceptance by prefix-reuse class, from a FULL q27 journal (not the
[req]-only extract): the [req] line has no pfx field, so each [req] is joined
with the [gen] line that preceded it (single slot, Q27_BATCH=0: strictly
sequential). Classes: cold (hit=0), vram (P8/P9 hit, pfx=0), restore (RAM/disk
tier, pfx>0). Also lists every [pfx] persist/restore line with its timing.

usage: pf_restore_agg.py <label> <journal> [<label> <journal> ...]
Written for docs/plans/2026-09-08-prefill-attack.md phase 0 (does a restored
turn draft from a shallow DFlash2 ring? compare tok/round per class).
"""
import re, sys, statistics as st

GEN = re.compile(r"^\[gen\] prompt=(\d+) prefix_hit=(\d+) snap=(\d+) ckpt=(-?\d+) pfx=(\d+)")
REQ = re.compile(r"^\[req\] .*? prompt=(\d+) hit=(\d+) ckpt=(-?\d+) pf=(\d+) pf_ms=(\d+) dec=(\d+) dec_ms=(\d+) .*? rounds=(\d+)")
PFX = re.compile(r"^\[pfx\] (persisting|restore) L=(\d+) .*?(export|read) (\d+) ms(?: \+ import (\d+) ms)?")

def run(label, path):
    rows = []; pend = None; persists = []; restores = []
    for line in open(path, errors="replace"):
        g = GEN.match(line)
        if g:
            pend = int(g.group(5)); continue
        p = PFX.match(line)
        if p:
            (persists if p.group(1) == "persisting" else restores).append(
                (int(p.group(2)), int(p.group(4)) + int(p.group(5) or 0)))
            continue
        r = REQ.match(line)
        if r:
            prompt, hit, ckpt, pf, pf_ms, dec, dec_ms, rounds = map(int, r.groups())
            cls = "cold" if hit == 0 else ("restore" if (pend or 0) > 0 else "vram")
            rows.append((cls, prompt, hit, pf, pf_ms, dec, dec_ms, rounds))
            pend = None
    if not rows:
        print(f"== {label}: no [req] lines"); return
    print(f"== {label}: {len(rows)} reqs; persists {len(persists)} (export ms med "
          f"{st.median(x[1] for x in persists) if persists else 0:.0f}, max "
          f"{max((x[1] for x in persists), default=0)}); restores {len(restores)} "
          f"(read+import ms med {st.median(x[1] for x in restores) if restores else 0:.0f}, max "
          f"{max((x[1] for x in restores), default=0)})")
    print(f"   {'class':8s} {'n':>4} {'pf tok':>8} {'pf ms':>8} {'pf tok/s':>8} {'dec tok':>8} {'tok/round':>9} {'round ms':>8}")
    for cls in ("cold", "vram", "restore"):
        b = [r for r in rows if r[0] == cls]
        if not b: continue
        pf = sum(r[3] for r in b); pfms = sum(r[4] for r in b)
        d = [r for r in b if r[5] >= 8]
        dec = sum(r[5] for r in d); dms = sum(r[6] for r in d); rnd = sum(r[7] for r in d)
        print(f"   {cls:8s} {len(b):>4} {pf:>8} {pfms:>8} {1000*pf/max(pfms,1):>8.0f} {dec:>8} "
              f"{dec/max(rnd,1):>9.3f} {dms/max(rnd,1):>8.2f}")
    # long-decode subset only (thinking turns), same classes -- the mix differs per class
    print("   long turns (dec >= 1024) only:")
    for cls in ("cold", "vram", "restore"):
        d = [r for r in rows if r[0] == cls and r[5] >= 1024]
        if not d: continue
        dec = sum(r[5] for r in d); rnd = sum(r[7] for r in d)
        print(f"   {cls:8s} {len(d):>4} tok/round {dec/max(rnd,1):.3f}")

args = sys.argv[1:]
for i in range(0, len(args), 2):
    run(args[i], args[i + 1])
