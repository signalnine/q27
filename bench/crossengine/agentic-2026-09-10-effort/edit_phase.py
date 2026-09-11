#!/usr/bin/env python3
"""Per leg: tool calls before the first Edit/Write, between first and last,
and after the last Edit/Write (the verify-or-finish tail). usage: edit_phase.py leg..."""
import json, sys, glob
W = "/mnt/ai/swebench-work"
print(f"{'leg':12s} {'pre-edit/i':>10s} {'mid/i':>6s} {'post-edit/i':>11s} {'edits/i':>7s} {'no-edit inst':>12s}")
for leg in sys.argv[1:]:
    files = sorted(glob.glob(f"{W}/{leg}/*/logs/out.jsonl")); n = len(files)
    pre = mid = post = ed = noed = 0
    for p in files:
        calls = []
        for ln in open(p, errors="replace"):
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("type") == "assistant":
                calls += [x["name"] for x in r["message"].get("content", []) if x.get("type") == "tool_use"]
        idx = [i for i, c in enumerate(calls) if c in ("Edit", "Write", "MultiEdit", "NotebookEdit")]
        if not idx: noed += 1; pre += len(calls); continue
        pre += idx[0]; mid += idx[-1] - idx[0] + 1 - len(idx); post += len(calls) - idx[-1] - 1; ed += len(idx)
    print(f"{leg:12s} {pre/n:10.1f} {mid/n:6.1f} {post/n:11.1f} {ed/n:7.1f} {noed:12d}")
