#!/usr/bin/env python3
"""Mid-session thinking length on FIXED prompts, many seeds, any engine that
serves /v1/messages (q27, ninfer, llama.cpp): the multi-turn twin of
probe_think.py. Reads request bodies recorded by q27's Q27_REQ_LOG, picks the
tool-bearing turns listed in a selection file (one recorded seq per line),
and sends each body N times with seed=1..N, non-streaming, recording per
sample the thinking chars, text chars, tool calls, prompt and output tokens
and stop reason.

  replay_think.py <base_url> <label> <bodies.jsonl> <select.txt> [N] [max_tokens] [--llama]

--llama adds chat_template_kwargs {enable_thinking, reasoning_effort} from the
body's own effort: llama.cpp's Anthropic bridge drops output_config and maps
only thinking.type == "enabled", so without it the HF template would render
its xhigh default (or no thinking at all for Claude Code's "adaptive").
The stock template also REFUSES Claude Code 2.1.x bodies (a system message
after the first: "System message must be at the beginning") -- start
llama-server with --chat-template-file bench/crossengine/harness/
qwen38_sysinline.jinja, or use raw_think.py with q27-rendered prompts, which
takes every template out of the comparison.
Bodies are real session content: keep them and the per-sample text out of
the repo; this script writes lengths only (<label>.jsonl next to itself)."""
import json, os, sys, time, urllib.request
args = [a for a in sys.argv[1:] if not a.startswith("--")]
llama = "--llama" in sys.argv
base, label, bodies_path, sel_path = args[:4]
N = int(args[4]) if len(args) > 4 else 8
max_tokens = int(args[5]) if len(args) > 5 else 16384
here = os.path.dirname(os.path.abspath(__file__))
want = [int(x) for x in open(sel_path).read().split()]
recs = {}
for ln in open(bodies_path):
    d = json.loads(ln)
    if d["seq"] in want: recs[d["seq"]] = json.loads(d["body"])
out = open(os.path.join(here, f"{label}.jsonl"), "w")
for seq in want:
    body = recs[seq]
    body["stream"] = False; body["max_tokens"] = max_tokens
    # Claude Code sends temperature only; pin the rest of the card sampler in
    # the body too (llama.cpp's bridge passes top_p/top_k through but not
    # min_p -- start llama-server with --min-p 0.05 --top-k 20 --top-p 0.95)
    body.setdefault("temperature", 1.0); body.setdefault("top_p", 0.95); body.setdefault("top_k", 20)
    if llama:
        eff = (body.get("output_config") or {}).get("effort", "medium")
        body["chat_template_kwargs"] = {"enable_thinking": True, "reasoning_effort": eff}
    ths = []; pin = -1
    for seed in range(1, N + 1):
        b = dict(body); b["seed"] = seed
        req = urllib.request.Request(base.rstrip("/") + "/v1/messages", data=json.dumps(b).encode(),
                                     headers={"content-type": "application/json", "x-api-key": "local",
                                              "authorization": "Bearer local", "anthropic-version": "2023-06-01"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r: resp = json.load(r)
        except urllib.error.HTTPError as e:
            print(f"seq {seq} seed {seed}: HTTP {e.code} {e.read()[:300]}", flush=True); continue
        dt = time.time() - t0
        c = resp.get("content", [])
        think = sum(len(x.get("thinking", "")) for x in c if x.get("type") == "thinking")
        text = sum(len(x.get("text", "")) for x in c if x.get("type") == "text")
        tools = [x.get("name") for x in c if x.get("type") == "tool_use"]
        u = resp.get("usage", {})
        pin = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
        row = dict(seq=seq, msgs=len(body["messages"]), seed=seed, think=think, text=text, tools=tools,
                   prompt=pin, out=u.get("output_tokens"), stop=resp.get("stop_reason"), secs=round(dt, 2))
        out.write(json.dumps(row) + "\n"); out.flush(); ths.append(think)
    ths.sort()
    print(f"  seq {seq:4d} msgs {len(body['messages']):3d} prompt {pin:6d}: think median {ths[len(ths)//2] if ths else -1:6d} "
          f"mean {sum(ths)/max(1,len(ths)):7.0f} max {max(ths) if ths else -1:6d} (n={len(ths)})", flush=True)
print(f"== {label}: done")
