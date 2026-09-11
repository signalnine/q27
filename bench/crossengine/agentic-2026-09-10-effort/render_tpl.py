#!/usr/bin/env python3
"""Render an Anthropic body's HISTORY per the Qwen3.8 chat_template.jinja
(every content and reasoning |trim, tool-call separators from the template,
non-string args via tojson with spaced separators; mid-conversation system
messages as ninfer renders them), reusing q27's own system/tools block (the
prefix of q27's render up to the first message) so only the history rules
differ. usage: render_tpl.py <body.json> <q27_render.txt> <out.txt>"""
import json, sys
body = json.load(open(sys.argv[1])); q = open(sys.argv[2]).read()
head_end = q.index("<|im_end|>\n") + len("<|im_end|>\n")
assert q.startswith("<|im_start|>system\n")
out = [q[:head_end]]
WS = " \t\r\n"
def tr(s): return s.strip(WS)
def tojson(v): return json.dumps(v, ensure_ascii=False)
msgs = body["messages"]
i = 0
while i < len(msgs):
    m = msgs[i]; role = m["role"]; c = m["content"]
    blocks = [{"type": "text", "text": c}] if isinstance(c, str) else c
    if role == "system":
        out.append("<|im_start|>system\n" + tr("".join(b.get("text", "") for b in blocks if b.get("type") == "text")) + "<|im_end|>\n")
    elif role == "user":
        trs = [b for b in blocks if b.get("type") == "tool_result"]
        txt = [b for b in blocks if b.get("type") == "text"]
        assert not (trs and txt), "mixed user message"
        if trs:
            s = "<|im_start|>user"
            for b in trs:
                rc = b.get("content", "")
                if isinstance(rc, list): rc = "".join(x.get("text", "") for x in rc if x.get("type") == "text")
                s += "\n<tool_response>\n" + tr(rc) + "\n</tool_response>"
            out.append(s + "<|im_end|>\n")
        else:
            out.append("<|im_start|>user\n" + tr("".join(b["text"] for b in txt)) + "<|im_end|>\n")
    elif role == "assistant":
        think = tr("".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking"))
        text = tr("".join(b.get("text", "") for b in blocks if b.get("type") == "text"))
        s = "<|im_start|>assistant\n<think>\n" + think + "\n</think>\n\n" + text
        calls = [b for b in blocks if b.get("type") == "tool_use"]
        for k, b in enumerate(calls):
            s += ("\n\n" if text else "") if k == 0 else "\n"
            s += "<tool_call>\n<function=" + b["name"] + ">\n"
            for pk, pv in (b.get("input") or {}).items():
                s += "<parameter=" + pk + ">\n" + (pv if isinstance(pv, str) else tojson(pv)) + "\n</parameter>\n"
            s += "</function>\n</tool_call>"
        out.append(s + "<|im_end|>\n")
    i += 1
out.append("<|im_start|>assistant\n<think>\n")
open(sys.argv[3], "w").write("".join(out))
