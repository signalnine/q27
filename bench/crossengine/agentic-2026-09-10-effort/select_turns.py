#!/usr/bin/env python3
"""Pick replay bodies from a Q27_REQ_LOG: each MAIN session's tool-bearing
turns at the given positions (0 = the first tool turn), skipping subagent
sessions (their first user message is not the harness task). Writes one seq
per line. usage: select_turns.py <bodies.jsonl> <out.txt> <pos,pos,...>"""
import json, sys
src, out, pos = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
sess = []
for ln in open(src):
    d = json.loads(ln); b = json.loads(d["body"])
    if not b.get("tools"): continue
    first = b["messages"][0]; c = first["content"]
    txt = c if isinstance(c, str) else "".join(x.get("text", "") for x in c if x.get("type") == "text")
    main = "is checked out at /workspace" in txt
    if not sess or len(b["messages"]) < sess[-1]["n"] or sess[-1]["main"] != main:
        sess.append({"main": main, "n": 0, "seqs": [], "task": txt[txt.find("The repository"):][:60]})
    sess[-1]["n"] = len(b["messages"]); sess[-1]["seqs"].append(d["seq"])
pick = []
for s in sess:
    if not s["main"] or len(s["seqs"]) < 6: continue
    got = [s["seqs"][p] for p in pos if p < len(s["seqs"])]
    pick += got
    print(f"{len(s['seqs']):3d} turns {s['task']!r}: {got}")
open(out, "w").write("\n".join(map(str, pick)) + "\n")
print(len(pick), "bodies")
