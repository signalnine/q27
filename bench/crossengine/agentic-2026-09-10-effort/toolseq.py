#!/usr/bin/env python3
"""Tool-call sequence of a Claude Code stream-json transcript: per assistant
turn, thinking chars and the tool + a short arg summary (no content dumped)."""
import json, sys
for p in sys.argv[1:]:
    print("==", p)
    n = 0
    for ln in open(p):
        try: r = json.loads(ln)
        except Exception: continue
        if r.get("type") != "assistant": continue
        for c in r["message"].get("content", []):
            if c.get("type") == "thinking": print(f"   think {len(c.get('thinking',''))}")
            elif c.get("type") == "tool_use":
                n += 1; i = c.get("input", {})
                arg = i.get("command") or i.get("file_path") or i.get("pattern") or i.get("path") or ""
                print(f"{n:3d} {c['name']:10s} {str(arg)[:90]!r}")
            elif c.get("type") == "text" and c.get("text","").strip(): print(f"   text {len(c['text'])}")
