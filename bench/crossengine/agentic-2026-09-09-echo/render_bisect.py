#!/usr/bin/env python3
"""Localize a cross-engine prompt-size difference by section: send the same
/v1/messages body with sections removed (max_tokens=1, thinking off so the
answer is one token) and print the engine's prompt token count per variant.
Run against each engine; the per-variant deltas say which section renders
differently. usage: render_bisect.py <base_url> <label> <body.json>"""
import json, sys, urllib.request
base, label, body_path = sys.argv[1:4]
body = json.load(open(body_path))
def count(b):
    b = dict(b); b["stream"] = False; b["max_tokens"] = 1
    req = urllib.request.Request(base.rstrip("/") + "/v1/messages", data=json.dumps(b).encode(),
                                 headers={"content-type": "application/json", "x-api-key": "local",
                                          "authorization": "Bearer local", "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=300) as r: u = json.load(r).get("usage", {})
    return (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
variants = {}
variants["full"] = body
v = dict(body); v.pop("tools", None); variants["no_tools"] = v
v = dict(body); v["tools"] = body["tools"][:1]; variants["one_tool"] = v
v = dict(body); v["system"] = "You are a helpful assistant."; variants["tiny_system"] = v
v = dict(body); v.pop("tools", None); v["system"] = "You are a helpful assistant."; variants["tiny_system_no_tools"] = v
v = dict(body); v["messages"] = [{"role": "user", "content": "hi"}]; variants["user_hi"] = v
v = dict(body); v["thinking"] = {"type": "disabled"}; v.pop("output_config", None); variants["thinking_off"] = v
print(f"== {label}")
res = {}
for k, b in variants.items():
    try: res[k] = count(b)
    except Exception as e: res[k] = f"ERR {str(e)[:80]}"
    print(f"  {k:22s} {res[k]}")
if all(isinstance(x, int) for x in res.values()):
    print(f"  tools block = {res['full'] - res['no_tools']}, system = {res['full'] - res['tiny_system']}, "
          f"user = {res['full'] - res['user_hi']}, thinking prefix = {res['full'] - res['thinking_off']}")
