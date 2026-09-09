#!/usr/bin/env python3
# One table over the campaign's legs: per-leg decode t/s (aggregate over the
# run window), tok/round, prefix reuse, wall, turns, and the cheap quality
# signals (nonempty diff, gold file edited). Reads <leg>.log (run.sh's
# ENGINE DECODE block) and results.<leg>.jsonl written by campaign.sh.
import json, os, re, sys, statistics as st
DIR = os.path.dirname(os.path.abspath(__file__))
legs = sys.argv[1:] or ['q27lad', 'q27d2q4', 'q27d2q8', 'ninferd2', 'ninfermtp']
print(f"{'leg':10s} {'inst':>4s} {'dec t/s':>8s} {'med':>6s} {'tok/rnd':>7s} {'reuse':>6s} {'wall/i':>7s} {'turns':>6s} {'out_tok':>8s} {'nonempty':>8s} {'gold':>5s}")
for leg in legs:
    lf = os.path.join(DIR, leg + '.log'); rf = os.path.join(DIR, f'results.{leg}.jsonl')
    if not os.path.exists(lf): print(f"{leg:10s} (no log)"); continue
    log = open(lf).read()
    m = re.search(r'decode: ([0-9.]+) t/s agg / ([0-9.]+) med', log)
    tr = re.search(r'tok/round: ([0-9.]+)', log)
    pr = re.search(r'prefix reuse: ([0-9.]+)%', log)
    rows = [json.loads(l) for l in open(rf)] if os.path.exists(rf) else []
    n = len(rows)
    wall = sum(r['wall_s'] for r in rows) / n if n else 0
    turns = sum(r['turns'] for r in rows) / n if n else 0
    out = sum(r['out_tok'] for r in rows) if n else 0
    ne = sum(r['nonempty'] for r in rows); gh = sum(r['gold_hit'] for r in rows)
    print(f"{leg:10s} {n:4d} {float(m.group(1)) if m else 0:8.1f} {float(m.group(2)) if m else 0:6.1f} "
          f"{float(tr.group(1)) if tr else 0:7.3f} {float(pr.group(1)) if pr else 0:5.1f}% {wall:7.0f} {turns:6.1f} {out:8d} {ne:5d}/{n:<2d} {gh:3d}/{n}")
