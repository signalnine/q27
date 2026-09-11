#!/usr/bin/env python3
"""Per-turn thinking chars from Q27_REQ_LOG bodies: each session's LAST tool
body carries every earlier assistant turn (thinking included since the model
echo). usage: turns_from_bodies.py <bodies.jsonl>"""
import json, sys
sess = []; cur = None
for ln in open(sys.argv[1]):
    d = json.loads(ln); b = json.loads(d["body"])
    if not b.get("tools"): continue
    ms = b["messages"]
    if cur is None or len(ms) < cur[1]: sess.append([d["seq"], len(ms), b]); cur = sess[-1]
    else: cur[1] = len(ms); cur[2] = b
allt = []
for first, n, b in sess:
    th = []
    for m in b["messages"]:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            th.append(sum(len(x.get("thinking", "")) for x in m["content"] if x.get("type") == "thinking"))
    allt += th
    print(f"session@{first:4d}: {len(th):3d} turns, thinking {sum(th)/1000:6.1f}K chars, median {sorted(th)[len(th)//2] if th else 0:5d}, >2K {sum(1 for t in th if t > 2000):3d}")
allt.sort(); n = len(allt)
print(f"ALL: {len(sess)} sessions {n} turns, mean {sum(allt)/n:.0f}, median {allt[n//2]}, p90 {allt[int(.9*n)]}, >2K {sum(1 for t in allt if t>2000)/n:.0%}, share in >2K {sum(t for t in allt if t>2000)/max(1,sum(allt)):.0%}")
