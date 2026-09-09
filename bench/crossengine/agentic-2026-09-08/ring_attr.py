#!/usr/bin/env python3
"""Item 5 (2026-09-08 (p) agenda): drafter ring context at decode start vs
acceptance, from a production journal. Joins each request's [gen] (restore
tier), [d2] ring align (rows the ring kept for this prompt) and [req]
(dec/rounds/dec_ms). Rows available to the drafter when decode starts =
rows kept + the re-prefilled suffix (seeded from prefill taps), capped at the
2048-row window. usage: ring_attr.py <journal> [<journal>...]"""
import re, sys
from collections import defaultdict
WINDOW = 2048
def kv(line):
    return {k: v for k, v in re.findall(r"(\w+)=([^\s]+)", line)}
reqs = []
for path in sys.argv[1:]:
    gen = d2 = None
    for line in open(path, errors="replace"):
        if line.startswith("[gen] prompt="):
            gen = kv(line); d2 = None
        elif line.startswith("[d2] ring align:"):
            m = re.search(r"keep (\d+) rows \(lcp-capped (\d+), base (\d+), seq (\d+)\)", line)
            d2 = dict(keep=int(m.group(1)), lcp=int(m.group(2)), base=int(m.group(3)), seq=int(m.group(4)))
        elif line.startswith("[req]"):
            r = kv(line)
            if gen is None or d2 is None or int(r.get("dec", 0)) < 8:
                gen = d2 = None; continue
            hit = int(r["hit"]); pf = int(r["pf"]); dec = int(r["dec"]); rounds = int(r["rounds"])
            tier = "cold" if hit == 0 else ("P8" if int(gen.get("snap", 0)) >= hit and int(gen.get("pfx", 0)) == 0 and int(gen.get("ckpt", -1)) < 0
                                            else "P16" if int(gen.get("pfx", 0)) > 0 else "P9")
            rows = min(WINDOW, d2["keep"] + pf)
            reqs.append(dict(rid=int(r["rid"]), prompt=int(r["prompt"]), hit=hit, pf=pf, keep=d2["keep"], rows=rows,
                             dec=dec, rounds=rounds, dec_ms=float(r["dec_ms"]), tier=tier, file=path.split("/")[-1]))
            gen = d2 = None
n = len(reqs); tot_dec = sum(r["dec"] for r in reqs); tot_ms = sum(r["dec_ms"] for r in reqs)
print(f"{n} decoding requests, {tot_dec} tokens, {tot_ms / 1000:.0f} s decode")
def table(title, keyf, keys):
    print(f"\n{title}")
    print(f"{'class':28s} {'n':>4s} {'dec tok':>8s} {'dec s':>6s} {'share':>6s} {'tok/round':>9s} {'t/s':>6s} {'mean pf':>8s} {'mean keep':>9s}")
    ref = None
    for k in keys:
        rs = [r for r in reqs if keyf(r) == k]
        if not rs: continue
        dec = sum(r["dec"] for r in rs); rnd = sum(r["rounds"] for r in rs); ms = sum(r["dec_ms"] for r in rs)
        print(f"{k:28s} {len(rs):4d} {dec:8d} {ms / 1000:6.1f} {ms / tot_ms * 100:5.1f}% {dec / max(rnd, 1):9.3f} {dec / max(ms, 1) * 1000:6.1f} "
              f"{sum(r['pf'] for r in rs) / len(rs):8.0f} {sum(r['keep'] for r in rs) / len(rs):9.0f}")
def rows_bin(r):
    x = r["rows"]
    return "rows 0-63" if x < 64 else "rows 64-255" if x < 256 else "rows 256-1023" if x < 1024 else "rows 1024-2047" if x < WINDOW else "rows 2048 (full)"
table("by drafter rows at decode start (kept + re-prefilled suffix, cap 2048)", rows_bin,
      ["rows 0-63", "rows 64-255", "rows 256-1023", "rows 1024-2047", "rows 2048 (full)"])
def cls(r):
    if r["hit"] == 0: return "cold prefill (no restore)"
    if r["keep"] == 0: return f"restored {r['tier']}, ring EMPTY"
    if r["keep"] < WINDOW: return f"restored {r['tier']}, ring partial"
    return f"restored {r['tier']}, ring full"
table("by restore tier and ring state", cls, sorted({cls(r) for r in reqs}))
# the affected class: restored, ring empty, tiny suffix (< 512 tokens re-prefilled)
aff = [r for r in reqs if r["hit"] > 0 and r["keep"] == 0 and r["pf"] < 512]
full = [r for r in reqs if r["rows"] >= WINDOW]
if aff and full:
    a_tpr = sum(r["dec"] for r in aff) / sum(r["rounds"] for r in aff)
    f_tpr = sum(r["dec"] for r in full) / sum(r["rounds"] for r in full)
    a_ms = sum(r["dec_ms"] for r in aff)
    print(f"\nAFFECTED (restored, ring empty, suffix < 512): n={len(aff)}, decode {a_ms / 1000:.1f} s ({a_ms / tot_ms * 100:.1f}% of decode), "
          f"tok/round {a_tpr:.3f} vs full-ring {f_tpr:.3f} ({(f_tpr / a_tpr - 1) * 100:+.1f}% rounds if reseeded to full)")
    print(f"  ceiling on savings if the class ran at full-ring acceptance: {a_ms / 1000 * (1 - a_tpr / f_tpr):.1f} s per run; "
          f"per affected request {a_ms / len(aff) * (1 - a_tpr / f_tpr):.0f} ms (a reseed must cost less than that to pay)")
    print("  affected requests (rid prompt hit pf keep dec rounds tok/round):")
    for r in sorted(aff, key=lambda r: -r["dec_ms"])[:12]:
        print(f"    {r['rid']:4d} {r['prompt']:6d} {r['hit']:6d} {r['pf']:5d} {r['keep']:5d} {r['dec']:5d} {r['rounds']:4d} {r['dec'] / r['rounds']:.2f}")
