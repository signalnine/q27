#!/usr/bin/env python3
"""Prefill profile of real Claude Code traffic from q27 [req] journal lines.

usage: pf_agg.py <label> <file> [<label> <file> ...]
Reports per file: requests, prompt tokens, computed prefill tokens (pf), cached
(hit), prefill wall, decode wall, aggregate prefill tok/s, and a by-size table
(pf tokens bucket -> n, tok/s, mean ms) so we see what sizes actually occur.
"""
import re, sys

RX = re.compile(r"prompt=(\d+) hit=(\d+) ckpt=(-?\d+) pf=(\d+) pf_ms=(\d+) dec=(\d+) dec_ms=(\d+)")
QW = re.compile(r"qw_ms=(\d+) tok_ms=(\d+)")
BUCKETS = [(0, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 16384), (16384, 65536), (65536, 10**9)]

def run(label, path):
    rows = []
    for line in open(path, errors="replace"):
        m = RX.search(line)
        if not m:
            continue
        q = QW.search(line)
        prompt, hit, ckpt, pf, pf_ms, dec, dec_ms = map(int, m.groups())
        qw, tk = (int(q.group(1)), int(q.group(2))) if q else (0, 0)
        rows.append((prompt, hit, pf, pf_ms, dec, dec_ms, qw, tk))
    if not rows:
        print(f"{label}: no [req] lines"); return
    n = len(rows)
    P = sum(r[0] for r in rows); H = sum(r[1] for r in rows); PF = sum(r[2] for r in rows)
    PFMS = sum(r[3] for r in rows); D = sum(r[4] for r in rows); DMS = sum(r[5] for r in rows)
    QW_ = sum(r[6] for r in rows); TK = sum(r[7] for r in rows)
    hits = sum(1 for r in rows if r[1] > 0)
    print(f"== {label}: {n} reqs, prompt {P} tok, cached {H} ({100*H/max(P,1):.1f}%), computed pf {PF}")
    print(f"   prefill wall {PFMS/1000:.1f} s  decode wall {DMS/1000:.1f} s  queue {QW_/1000:.1f} s  tokenize {TK/1000:.1f} s"
          f"  -> prefill = {100*PFMS/max(PFMS+DMS,1):.1f}% of prefill+decode")
    print(f"   aggregate prefill {1000*PF/max(PFMS,1):.0f} tok/s (computed tokens / prefill ms); reqs with a hit: {hits}/{n}")
    print(f"   {'pf bucket':>14} {'n':>5} {'tok':>8} {'ms':>8} {'tok/s':>7} {'mean ms':>8} {'share of pf ms':>14}")
    for lo, hi in BUCKETS:
        b = [r for r in rows if lo <= r[2] < hi]
        if not b: continue
        t = sum(r[2] for r in b); ms = sum(r[3] for r in b)
        print(f"   {lo:>6}-{min(hi, 10**6):<7} {len(b):>5} {t:>8} {ms:>8} {1000*t/max(ms,1):>7.0f} {ms/len(b):>8.0f} {100*ms/max(PFMS,1):>13.1f}%")
    big = sorted(rows, key=lambda r: -r[3])[:5]
    print("   top-5 prefill walls (prompt/hit/pf/pf_ms):", ", ".join(f"{r[0]}/{r[1]}/{r[2]}/{r[3]}" for r in big))

args = sys.argv[1:]
for i in range(0, len(args), 2):
    run(args[i], args[i + 1])
