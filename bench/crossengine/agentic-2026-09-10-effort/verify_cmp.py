#!/usr/bin/env python3
"""Per leg: tool calls by kind, and how many Bash calls try to RUN code
(python/pytest/pip) and how many of those hit a missing-dependency error.
usage: verify_cmp.py leg ..."""
import json, sys, glob, re
W = "/mnt/ai/swebench-work"
RUN = re.compile(r"\b(python3?|pytest|pip3?|py\.test|tox)\b")
MISS = re.compile(r"ModuleNotFoundError|No module named|ImportError|command not found|cannot import name")
print(f"{'leg':12s} {'tools/i':>7s} {'Bash/i':>6s} {'run/i':>6s} {'run-fail/i':>10s} {'missing-dep/i':>13s} {'Read/i':>6s} {'Grep+Glob/i':>11s} {'Edit+Write/i':>12s} {'Web/i':>5s} {'Task/i':>6s}")
for leg in sys.argv[1:]:
    files = sorted(glob.glob(f"{W}/{leg}/*/logs/out.jsonl")); n = len(files)
    c = dict(tools=0, bash=0, run=0, runfail=0, miss=0, read=0, grep=0, edit=0, web=0, task=0)
    for p in files:
        pend = {}
        for ln in open(p, errors="replace"):
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("type") == "assistant":
                for x in r["message"].get("content", []):
                    if x.get("type") != "tool_use": continue
                    c["tools"] += 1; nm = x["name"]; inp = x.get("input", {})
                    if nm == "Bash":
                        c["bash"] += 1
                        if RUN.search(inp.get("command", "")): c["run"] += 1; pend[x["id"]] = True
                    elif nm == "Read": c["read"] += 1
                    elif nm in ("Grep", "Glob"): c["grep"] += 1
                    elif nm in ("Edit", "Write", "MultiEdit"): c["edit"] += 1
                    elif nm.startswith("Web"): c["web"] += 1
                    elif nm in ("Task", "Agent"): c["task"] += 1
            elif r.get("type") == "user":
                for x in (r.get("message", {}).get("content") or []):
                    if not isinstance(x, dict) or x.get("type") != "tool_result": continue
                    if x.get("tool_use_id") not in pend: continue
                    txt = x.get("content"); txt = txt if isinstance(txt, str) else json.dumps(txt)
                    if x.get("is_error"): c["runfail"] += 1
                    if MISS.search(txt): c["miss"] += 1
    f = lambda k: c[k] / max(1, n)
    print(f"{leg:12s} {f('tools'):7.1f} {f('bash'):6.1f} {f('run'):6.1f} {f('runfail'):10.1f} {f('miss'):13.1f} {f('read'):6.1f} {f('grep'):11.1f} {f('edit'):12.1f} {f('web'):5.1f} {f('task'):6.1f}")
