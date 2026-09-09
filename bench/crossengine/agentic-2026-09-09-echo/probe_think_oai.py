#!/usr/bin/env python3
"""probe_think.py for engines that only speak /v1/chat/completions
(llama.cpp): converts the recorded Anthropic body (system blocks, one user
turn, tools) to the OpenAI shape, asks the template for medium effort with
thinking on via chat_template_kwargs, pins the same sampler chain, and
records the same per-sample fields from reasoning_content/content.
usage: probe_think_oai.py <base_url> <label> <body.json> [N] [max_tokens]"""
import json, os, sys, time, urllib.request, statistics as st
base, label, body_path = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 24
max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
here = os.path.dirname(os.path.abspath(__file__))
a = json.load(open(body_path))
sysb = a["system"]; system = "\n\n".join(x["text"] for x in sysb) if isinstance(sysb, list) else sysb
msgs = [{"role": "system", "content": system}]
for m in a["messages"]:
    c = m["content"]
    msgs.append({"role": m["role"], "content": c if isinstance(c, str) else "\n\n".join(x.get("text", "") for x in c if x.get("type") == "text")})
tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}} for t in a.get("tools", [])]
effort = (a.get("output_config") or {}).get("effort", "medium")
body = dict(model="qwen38", messages=msgs, tools=tools, stream=False, max_tokens=max_tokens,
            temperature=1.0, top_p=0.95, top_k=20, min_p=0.05,
            chat_template_kwargs={"enable_thinking": True, "reasoning_effort": effort})
out = open(os.path.join(here, f"{label}.jsonl"), "w"); rows = []
for seed in range(1, N + 1):
    b = dict(body); b["seed"] = seed
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(b).encode(),
                                 headers={"content-type": "application/json", "authorization": "Bearer local"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r: resp = json.load(r)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read()[:300]); sys.exit(1)
    dt = time.time() - t0
    ch = resp["choices"][0]; msg = ch["message"]
    think = len(msg.get("reasoning_content") or ""); text = len(msg.get("content") or "")
    tools_called = [t["function"]["name"] for t in (msg.get("tool_calls") or [])]
    u = resp.get("usage", {})
    row = dict(seed=seed, think=think, text=text, tools=tools_called, out=u.get("completion_tokens"), stop=ch.get("finish_reason"),
               model=resp.get("model"), secs=round(dt, 2))
    rows.append(row); out.write(json.dumps(row) + "\n"); out.flush()
    print(f"  seed {seed:2d} think {think:6d} text {text:5d} tools {tools_called} out {u.get('completion_tokens')} stop {ch.get('finish_reason')} {dt:.1f}s", flush=True)
th = [r["think"] for r in rows]; ot = [r["out"] or 0 for r in rows]
print(f"== {label}: n={N} thinking chars mean {st.mean(th):.0f} median {st.median(th):.0f} p90 {sorted(th)[int(0.9 * N) - 1]} max {max(th)} | "
      f"output tokens mean {st.mean(ot):.0f} median {st.median(ot):.0f} | stop {dict((s, sum(1 for r in rows if r['stop'] == s)) for s in set(r['stop'] for r in rows))} | "
      f"tool calls {sum(1 for r in rows if r['tools'])}/{N}")
