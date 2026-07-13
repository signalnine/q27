#!/usr/bin/env python3
"""Summarize a gemv-pin A/B leg from the engine's own [req] telemetry.

GOTCHA (BUILDLOG 2026-07-13): in a [req] line, dec/tps/rounds/sfxm/sfxn are
PER-REQUEST, but `sfx=<fired>,<tokens>` is ENGINE-CUMULATIVE across every
request the process has served. So suffix totals come from the LAST line, not
from summing. sfxm/sfxn (per-request suffix wall + rounds) ARE summable.

Usage: pin_ab_report.py <leg-name> <reqfile> [<leg-name> <reqfile> ...]
"""
import re
import sys


def parse(path):
    reqs = []
    for line in open(path):
        if not line.startswith("[req]"):
            continue
        g = lambda p, d=0.0: (float(m.group(1)) if (m := re.search(p, line)) else d)
        reqs.append(
            dict(
                dec=g(r"dec=(\d+)"),
                dec_ms=g(r"dec_ms=(\d+)"),
                tps=g(r"tps=([\d.]+)"),
                rounds=g(r"rounds=(\d+)"),
                sfxm=g(r"sfxm=([\d.]+)"),
                sfxn=g(r"sfxn=(\d+)"),
                sfx_tok_cum=g(r"sfx=\d+,(\d+)"),
                prompt=g(r"prompt=(\d+)"),
            )
        )
    return [r for r in reqs if r["dec"] > 0]


def summarize(name, path):
    r = parse(path)
    if not r:
        print(f"{name}: no requests")
        return None
    dec = sum(x["dec"] for x in r)
    dec_ms = sum(x["dec_ms"] for x in r)
    sfx_ms = sum(x["sfxm"] for x in r)
    sfx_rnd = sum(x["sfxn"] for x in r)
    sfx_tok = max(x["sfx_tok_cum"] for x in r)  # cumulative -> take the max
    tps = sorted(x["tps"] for x in r)
    agg = dec / (dec_ms / 1000) if dec_ms else 0
    print(f"\n=== {name} ===")
    print(f"  requests            {len(r)}")
    print(f"  decode tokens       {int(dec)}")
    print(f"  AGGREGATE decode    {agg:.1f} t/s   (total tokens / total decode ms)")
    print(f"  per-req median      {tps[len(tps)//2]:.1f} t/s   p90 {tps[int(.9*len(tps))]:.1f}   peak {tps[-1]:.1f}")
    print(f"  suffix tokens       {int(sfx_tok)}  ({100*sfx_tok/dec:.1f}% of decode)")
    if sfx_rnd:
        print(f"  suffix rounds       {int(sfx_rnd)}   AL {sfx_tok/sfx_rnd:.2f} tok/round")
        print(f"  suffix ms/round     {sfx_ms/sfx_rnd:.2f} ms   <- the thing the retier moves")
        print(f"  suffix wall share   {100*sfx_ms/dec_ms:.1f}% of decode ms")
    return dict(name=name, agg=agg, med=tps[len(tps)//2], dec=dec, sfx_tok=sfx_tok,
                sfx_rnd=sfx_rnd, sfx_ms=sfx_ms, dec_ms=dec_ms, n=len(r))


legs = [summarize(sys.argv[i], sys.argv[i + 1]) for i in range(1, len(sys.argv), 2)]
legs = [x for x in legs if x]
if len(legs) == 2:
    a, b = legs
    print(f"\n=== {b['name']} vs {a['name']} ===")
    print(f"  aggregate decode   {a['agg']:.1f} -> {b['agg']:.1f} t/s   ({100*(b['agg']/a['agg']-1):+.1f}%)")
    print(f"  per-req median     {a['med']:.1f} -> {b['med']:.1f} t/s   ({100*(b['med']/a['med']-1):+.1f}%)")
    if a["sfx_rnd"] and b["sfx_rnd"]:
        am, bm = a["sfx_ms"] / a["sfx_rnd"], b["sfx_ms"] / b["sfx_rnd"]
        print(f"  suffix ms/round    {am:.2f} -> {bm:.2f} ms   ({100*(bm/am-1):+.1f}%)")
    print("  NOTE walls/token-volumes are trajectory-confounded (CC legs fork on")
    print("  tool-output wall-clock bytes); decode RATE is the currency.")
