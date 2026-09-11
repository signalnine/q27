#!/usr/bin/env bash
# Regenerate a template golden from a RUNNING llama-server (CPU only, -ngl 0)
# via /apply-template: this is what llama.cpp feeds the model, not a jinja2
# approximation. Run from the repo root with ~20 GB of free RAM; no VRAM, so
# it can run alongside a GPU job.
#   tools/capture_template_golden.sh <fixture.anthropic.json> <out.prompt>
# e.g. tools/golden/qwen38_tools_request.anthropic.json -> ...request.prompt
#      tools/golden/qwen38_history_request.anthropic.json -> ...history_request.prompt
# The template is the checkpoint's chat_template.jinja with ONE change
# (bench/crossengine/harness/qwen38_sysinline.jinja): a system message after
# the first renders inline as its own trimmed system turn instead of raising.
# Claude Code 2.1.x sends such messages mid-conversation; ninfer renders them
# the same way. Everything else is the stock template.
# (2026-09-10: arguments now pass through argv -- the old quoted heredoc read a
# literal "${1:?...}" as the output dir.)
set -uo pipefail
FIX="${1:?usage: capture_template_golden.sh <fixture.anthropic.json> <out.prompt>}"
OUT="${2:?usage: capture_template_golden.sh <fixture.anthropic.json> <out.prompt>}"
LC=/mnt/ai/projects/llama.cpp/build/bin/llama-server
GG=/mnt/ai/models/qwen38-27b-mtp-gguf/Qwen3.8-27B-MTP-Q5_K_M.gguf
TPL=bench/crossengine/harness/qwen38_sysinline.jinja
LOG=$(mktemp)
$LC -m $GG --port 8193 --host 127.0.0.1 -ngl 0 --jinja --chat-template-file $TPL --reasoning off -c 4096 --no-warmup >$LOG 2>&1 &
P=$!
for i in $(seq 1 120); do [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8193/health)" = 200 ] && break; sleep 5; done
python3 - "$FIX" "$OUT" <<'PY'
import json, sys, urllib.request
fix, out_path = sys.argv[1:3]
req = json.load(open(fix))
sysc = req["system"] if isinstance(req["system"], str) else "".join(b["text"] for b in req["system"])
msgs = [{"role": "system", "content": sysc}]
for m in req["messages"]:
    c = m["content"]
    if isinstance(c, str): msgs.append({"role": m["role"], "content": c}); continue
    tcs = []; text = ""; think = ""
    for b in c:
        if b["type"] == "text": text += b["text"]
        elif b["type"] == "thinking": think += b["thinking"]
        elif b["type"] == "tool_use":
            tcs.append({"type": "function", "id": b["id"], "function": {"name": b["name"], "arguments": json.dumps(b["input"])}})
        elif b["type"] == "tool_result":
            rc = b["content"] if isinstance(b["content"], str) else "".join(x.get("text", "") for x in b["content"])
            msgs.append({"role": "tool", "content": rc, "tool_call_id": b["tool_use_id"]})
    if m["role"] == "assistant":
        e = {"role": "assistant", "content": text}
        if think: e["reasoning_content"] = think
        if tcs: e["tool_calls"] = tcs
        msgs.append(e)
    elif m["role"] != "user" or text:
        msgs.append({"role": m["role"], "content": text})
tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in req["tools"]]
# no effort field -> the template's default (xhigh), as q27 renders a 3.8 boot
kw = {"enable_thinking": True}
eff = (req.get("output_config") or {}).get("effort")
if eff: kw["reasoning_effort"] = eff
body = {"messages": msgs, "tools": tools, "chat_template_kwargs": kw}
r = urllib.request.Request("http://127.0.0.1:8193/apply-template", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
out = json.loads(urllib.request.urlopen(r, timeout=120).read())
p = out.get("prompt") if isinstance(out, dict) else str(out)
open(out_path, "w").write(p)
print("minja golden:", len(p), "chars ->", out_path)
PY
rc=$?
kill $P; wait $P 2>/dev/null; rm -f $LOG
exit $rc
