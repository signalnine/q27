#!/usr/bin/env python3
# Decode at depth: one conversation of ~depth filler tokens, then a seeded
# 400-token generation (streamed, content hashed). The [req] journal line
# carries dec/rounds/dec_ms for the arm comparison.
# usage: width_depth_probe.py <base> <tag> <depth> [seeds=2]
import hashlib, json, requests, sys, time
base, tag, depth = sys.argv[1], sys.argv[2], int(sys.argv[3])
nseeds = int(sys.argv[4]) if len(sys.argv) > 4 else 2
SAMPLER = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.05)
def filler(approx_tokens, nonce):
    unit = ("def h_{n}(recs,cfg):  # {x}\n    o=[]\n    for i,r in enumerate(recs):\n"
            "        if not ok(r,cfg.s_{n}): continue\n        o.append(tx(r,{n}.0,i))\n    return o\n"
            "[2026-08-27T0{m}:{nn}:12Z] INFO w-{n} did {n}00 recs in {n}.4s {x}\n\n")
    s = []; n = 0; c = 0
    while c < approx_tokens * 2.1:
        s.append(unit.format(n=(n % 97) + 1, nn=(n % 59) + 1, m=(n % 9), x=nonce)); c += len(s[-1]); n += 1
    return "".join(s)
GEN = ("Ignore the material above. Think step by step about which three of the functions above would be "
       "hardest to test, then output a numbered list of 30 distinct two-word English noun phrases about weather.")
for seed in range(1, nseeds + 1):
    content = filler(depth, f"depth-{depth}") + "\n\n" + GEN
    body = dict(model="qwen38", messages=[{"role": "user", "content": content}], max_tokens=400,
                stream=True, stream_options={"include_usage": True}, seed=seed, **SAMPLER)
    h = hashlib.sha256(); comp = 0; prompt = 0; t0 = time.time()
    with requests.post(base + "/v1/chat/completions", json=body, stream=True, timeout=900) as r:
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "): continue
            d = line[6:]
            if d == b"[DONE]": break
            j = json.loads(d)
            if j.get("usage"): comp = j["usage"].get("completion_tokens", 0); prompt = j["usage"].get("prompt_tokens", 0)
            for ch in j.get("choices", []):
                delta = ch.get("delta", {})
                for k in ("reasoning_content", "reasoning", "content"):
                    if delta.get(k): h.update(delta[k].encode())
    print(f"{tag} depth={depth} seed={seed} prompt={prompt} gen={comp} sha={h.hexdigest()[:16]} wall={time.time()-t0:.2f}s", flush=True)
