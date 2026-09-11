#!/usr/bin/env python3
"""Turn-count A/B across legs: API turns per instance, thinking chars, text
chars, tool calls, output tokens (harness result record), gold hit, wall.
Reads the retained Claude Code transcripts under /mnt/ai/swebench-work/<leg>/
<iid>/logs/out.jsonl and the campaign results.<leg>.jsonl next to this file
(or in the 09-09 dir for the reference legs).
usage: turns_cmp.py [leg ...]   default: q27prod ninferd2 q27noecho q27echo"""
import json, os, sys, glob
from collections import defaultdict
W = "/mnt/ai/swebench-work"
HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = [HERE, os.path.join(HERE, "..", "agentic-2026-09-09"), os.path.join(HERE, "..", "agentic-2026-09-10"),
        os.path.join(HERE, "..", "agentic-2026-09-10-effort")]

def transcript(path):
    msgs = {}; order = []
    for ln in open(path, errors="replace"):
        try: d = json.loads(ln)
        except: continue
        if d.get("type") != "assistant": continue
        m = d["message"]; mid = m.get("id")
        if mid not in msgs: msgs[mid] = dict(think=0, text=0, tools=0); order.append(mid)
        for b in m.get("content", []):
            t = b.get("type")
            if t == "thinking": msgs[mid]["think"] += len(b.get("thinking", ""))
            elif t == "text": msgs[mid]["text"] += len(b.get("text", ""))
            elif t == "tool_use": msgs[mid]["tools"] += 1
    return [msgs[m] for m in order]

def results(leg):
    for d in DIRS:
        p = os.path.join(d, f"results.{leg}.jsonl")
        if os.path.exists(p):
            return {json.loads(l)["iid"]: json.loads(l) for l in open(p) if l.strip()}
    return {}

legs = sys.argv[1:] or ["q27prod", "ninferd2", "q27noecho", "q27echo"]
per = {}
for leg in legs:
    R = results(leg); rows = {}
    for f in sorted(glob.glob(f"{W}/{leg}/*/logs/out.jsonl")):
        iid = f.split("/")[5]; T = transcript(f); r = R.get(iid, {})
        rows[iid] = dict(turns=len(T), think=sum(t["think"] for t in T), text=sum(t["text"] for t in T),
                         tools=sum(t["tools"] for t in T), out=r.get("out_tok", 0), gold=r.get("gold_hit"),
                         wall=r.get("wall_s", 0), longthink=sum(1 for t in T if t["think"] > 2000))
    per[leg] = rows
iids = sorted(set().union(*[set(r) for r in per.values()]))
print(f"{'leg':10s} {'inst':>4s} {'turns/i':>7s} {'think K/i':>9s} {'think/turn':>10s} {'>2K turns':>9s} {'text K/i':>8s} {'tools/i':>7s} {'out tok/i':>9s} {'gold':>5s} {'wall/i':>6s}")
for leg in legs:
    r = per[leg]; n = len(r) or 1; T = sum(x["turns"] for x in r.values()) or 1
    print(f"{leg:10s} {len(r):4d} {sum(x['turns'] for x in r.values())/n:7.1f} {sum(x['think'] for x in r.values())/n/1000:9.1f} "
          f"{sum(x['think'] for x in r.values())/T:10.0f} {sum(x['longthink'] for x in r.values())/T:9.0%} {sum(x['text'] for x in r.values())/n/1000:8.1f} "
          f"{sum(x['tools'] for x in r.values())/n:7.1f} {sum(x['out'] for x in r.values())/n:9.0f} {sum(1 for x in r.values() if x['gold']):2d}/{len(r):<2d} {sum(x['wall'] for x in r.values())/n:6.0f}")
print()
hdr = f"{'iid':26s}" + "".join(f" {leg[:9]+' t/thK/out':>20s}" for leg in legs)
print(hdr)
for iid in iids:
    line = f"{iid:26s}"
    for leg in legs:
        x = per[leg].get(iid)
        line += f" {'-':>20s}" if not x else f" {x['turns']:4d}/{x['think']/1000:5.0f}/{x['out']:7d}{'*' if x['gold'] else ' '}"
    print(line)
print("\n* = patch touched a gold file")
