#!/usr/bin/env python3
"""Per-turn thinking length on a FIXED prompt, many seeds: isolates the
engine's per-turn behaviour from trajectory effects. Sends the same
/v1/messages body N times with seed=1..N (both q27 and ninfer honour a
body-level seed on the Anthropic path), non-streaming, and records per
sample the thinking chars, text chars, tool calls, output tokens and
stop reason.
usage: probe_think.py <base_url> <label> <body.json> [N] [max_tokens]
writes <label>.jsonl next to this file and prints the summary."""
import json, os, sys, time, urllib.request, statistics as st
base, label, body_path = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 32
max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
here = os.path.dirname(os.path.abspath(__file__))
body = json.load(open(body_path))
body["stream"] = False; body["max_tokens"] = max_tokens
out = open(os.path.join(here, f"{label}.jsonl"), "w")
rows = []
for seed in range(1, N + 1):
    b = dict(body); b["seed"] = seed
    req = urllib.request.Request(base.rstrip("/") + "/v1/messages", data=json.dumps(b).encode(),
                                 headers={"content-type": "application/json", "x-api-key": "local",
                                          "authorization": "Bearer local", "anthropic-version": "2023-06-01"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r: resp = json.load(r)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read()[:300]); sys.exit(1)
    dt = time.time() - t0
    think = sum(len(c.get("thinking", "")) for c in resp.get("content", []) if c.get("type") == "thinking")
    text = sum(len(c.get("text", "")) for c in resp.get("content", []) if c.get("type") == "text")
    tools = [c.get("name") for c in resp.get("content", []) if c.get("type") == "tool_use"]
    u = resp.get("usage", {})
    row = dict(seed=seed, think=think, text=text, tools=tools, out=u.get("output_tokens"), stop=resp.get("stop_reason"),
               model=resp.get("model"), secs=round(dt, 2))
    rows.append(row); out.write(json.dumps(row) + "\n"); out.flush()
    print(f"  seed {seed:2d} think {think:6d} text {text:5d} tools {tools} out {u.get('output_tokens')} stop {resp.get('stop_reason')} {dt:.1f}s", flush=True)
th = [r["think"] for r in rows]; ot = [r["out"] or 0 for r in rows]
print(f"== {label}: n={N} thinking chars mean {st.mean(th):.0f} median {st.median(th):.0f} p90 {sorted(th)[int(0.9 * N) - 1]} max {max(th)} | "
      f"output tokens mean {st.mean(ot):.0f} median {st.median(ot):.0f} | stop {dict((s, sum(1 for r in rows if r['stop'] == s)) for s in set(r['stop'] for r in rows))} | "
      f"tool calls {sum(1 for r in rows if r['tools'])}/{N}")
