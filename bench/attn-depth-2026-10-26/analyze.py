#!/usr/bin/env python3
"""Tables for the shape sweep: ms/round per leg and depth vs the default legs' band,
verify/draft split from the 200-round [d2timing] windows (attributed to a request only
when >= 90% of the window falls inside it), prefill ms for the split legs, and a
traffic-weighted delta using the 09-10 Claude Code campaign's decode-time shares by depth."""
import json, os, re, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = {4096: 0.04, 24576: 0.16, 45056: 0.60, 65536: 0.19}  # dec_ms share by depth bucket, 09-10 campaign
rows = [json.loads(l) for l in open(f"{HERE}/results.jsonl")]
errors = [r for r in rows if "ERROR" in r]
rows = [r for r in rows if "ERROR" not in r]
legs = []
for r in rows:
    if r["leg"] not in legs: legs.append(r["leg"])
depths = sorted({r["depth"] for r in rows})

def windows(leg):
    """per request index -> (wall, draft, verify, host) averaged over clean 200-round windows"""
    path = f"{HERE}/journal_{leg}.log"
    if not os.path.exists(path): return {}
    cum = []  # (n, sums dict)
    reqs = []  # cumulative round boundary after each request
    out = collections.defaultdict(list)
    for line in open(path, errors="replace"):
        m = re.match(r"\[d2timing\] (\d+) rounds: wall ([\d.]+) = fold ([\d.]+) \+ draft ([\d.]+) \+ verify ([\d.]+) \+ host ([\d.]+)", line)
        if m:
            n = int(m.group(1)); avg = [float(x) for x in m.group(2, 4, 5, 6)]
            prev_n, prev = (cum[-1] if cum else (0, [0, 0, 0, 0]))
            sums = [a * n for a in avg]
            win = [(s - p) / (n - prev_n) for s, p in zip(sums, prev)]
            cum.append((n, sums))
            lo, hi = prev_n, n
            # which request will this window belong to? decided once the [req] line arrives
            out["_pending"].append((lo, hi, win))
            continue
        m = re.match(r"\[req\] .*? rounds=(\d+)", line)
        if m:
            start = reqs[-1] if reqs else 0; end = start + int(m.group(1)); reqs.append(end)
            k = len(reqs) - 1
            keep = []
            for lo, hi, win in out["_pending"]:
                ov = max(0, min(hi, end) - max(lo, start))
                if ov >= 0.9 * (hi - lo): out[k].append(win)
                elif hi > end: keep.append((lo, hi, win))  # straddles into the next request
            out["_pending"] = keep
    res = {}
    for k, wins in out.items():
        if k == "_pending" or not wins: continue
        res[k] = [sum(w[i] for w in wins) / len(wins) for i in range(4)]
    return res

win = {leg: windows(leg) for leg in legs}
by = {(r["leg"], r["depth"]): r for r in rows}
def dflt(d, key):
    v = [by[(l, d)][key] for l in legs if l.startswith("default") and (l, d) in by and key in by[(l, d)]]
    return (sum(v) / len(v), min(v), max(v)) if v else (None, None, None)

for d in depths:
    base, lo, hi = dflt(d, "ms_round")
    print(f"\n=== depth {d}  (default band ms/round {lo:.2f}..{hi:.2f}, mean {base:.2f})" if base else f"\n=== depth {d}")
    print(f"{'leg':10} {'ms/round':>8} {'delta':>7} {'verify':>7} {'draft':>6} {'host':>5} {'tok/rnd':>7} {'pf_ms':>7} {'pf tok/s':>8}")
    for leg in legs:
        r = by.get((leg, d))
        if not r: continue
        k = depths.index(d)
        w = win[leg].get(k)
        v = f"{w[2]:7.2f} {w[1]:6.2f} {w[3]:5.2f}" if w else f"{'-':>7} {'-':>6} {'-':>5}"
        delta = f"{r['ms_round'] - base:+7.2f}" if base else f"{'':>7}"
        print(f"{leg:10} {r['ms_round']:8.2f} {delta} {v} {r['tok_round']:7.2f} {r['pf_ms']:7.0f} {r['pf']/r['pf_ms']*1000:8.0f}")

print("\n=== traffic-weighted delta vs default mean (ms/round; negative = faster) ===")
for leg in legs:
    if leg.startswith("default"): continue
    tot = 0; wsum = 0
    for d in depths:
        r = by.get((leg, d)); base = dflt(d, "ms_round")[0]
        if r and base: tot += WEIGHTS[d] * (r["ms_round"] - base); wsum += WEIGHTS[d]
    if wsum: print(f"{leg:10} {tot / wsum:+.3f}  (covers {wsum:.0%} of traffic weight)")
band = [dflt(d, "ms_round") for d in depths]
print("default band half-width per depth:", ", ".join(f"{d}: {(b[2]-b[1])/2:.2f}" for d, b in zip(depths, band) if b[0]))
if errors: print("\nERRORS:", *[json.dumps(e)[:200] for e in errors], sep="\n")
