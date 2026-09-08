#!/usr/bin/env python3
"""Prefix-cache miss anatomy of q27 [req] journal lines from Claude Code traffic.

usage: pf_misses.py <label> <file> [<label> <file> ...]

For each file: first turns vs returning turns, and for returning turns whether
the immediately preceding request belonged to the same conversation. The
2026-09-08 finding: a returning turn hit iff the previous request was the same
conversation -- every interleaved side request (another conv=) produced a full
re-prefill. Also lists the full misses with their neighbourhood so a run with
the cache tiers on can be checked turn by turn.
"""
import re, sys

RX = re.compile(r"rid=(\d+) .*?conv=(\w+) .*?prompt=(\d+) hit=(\d+) ckpt=(-?\d+) pf=(\d+) pf_ms=(\d+) dec=(\d+) dec_ms=(\d+)")

def run(label, path, show=8):
    rows = [m.groups() for l in open(path, errors="replace") for m in [RX.search(l)] if m]
    if not rows:
        print(f"== {label}: no [req] lines"); return
    seen = set(); prev = None
    cls = {"same-conv": [0, 0, 0, 0], "after-other-conv": [0, 0, 0, 0]}  # hits, full misses, full-miss ms, partial-miss ms(pf>4096 with hit>0)
    firsts = [0, 0, 0]  # n, ms, n with hit>0
    misses = []
    for i, r in enumerate(rows):
        conv, prompt, hit, pf, ms = r[1], int(r[2]), int(r[3]), int(r[5]), int(r[6])
        if conv in seen:
            k = "same-conv" if prev == conv else "after-other-conv"
            if hit > 0:
                cls[k][0] += 1
                if pf > 4096: cls[k][3] += ms
            else:
                cls[k][1] += 1; cls[k][2] += ms
                if pf > 4096: misses.append(i)
        else:
            seen.add(conv); firsts[0] += 1; firsts[1] += ms; firsts[2] += (hit > 0)
        prev = conv
    tot = sum(int(r[6]) for r in rows)
    print(f"== {label}: {len(rows)} reqs, {len(seen)} conversations, prefill wall {tot/1000:.0f}s")
    print(f"   first turns {firsts[0]:3d}: {firsts[1]/1000:6.1f}s ({100*firsts[1]/tot:.0f}%), {firsts[2]} of them with a prefix hit (system-block entry)")
    for k, v in cls.items():
        print(f"   returning, {k:>17}: hits {v[0]:3d}  full misses {v[1]:3d}  full-miss prefill {v[2]/1000:6.1f}s ({100*v[2]/tot:.0f}%)  big partial re-prefill {v[3]/1000:.1f}s")
    for i in misses[:show]:
        print(f"   ---- full miss rid {rows[i][0]} (prompt {rows[i][2]}, {int(rows[i][6])/1000:.1f}s)")
        for j in range(max(0, i - 3), min(len(rows), i + 2)):
            rr = rows[j]
            print(f"      rid={rr[0]:>4} conv={rr[1][:6]} prompt={rr[2]:>6} hit={rr[3]:>6} pf={rr[5]:>6} pf_ms={rr[6]:>5}")

args = sys.argv[1:]
for i in range(0, len(args), 2):
    run(args[i], args[i + 1])
