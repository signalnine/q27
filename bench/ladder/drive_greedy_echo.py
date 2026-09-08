#!/usr/bin/env python3
# Greedy echo probe: temperature 0, fixed prompts -> tokens are width-invariant
# across depthctl bar settings, so rounds/t/s deltas are EXACT (zero noise).
import requests, time, sys

base = sys.argv[1]; tag = sys.argv[2]

def filler(approx_tokens, nonce):
    unit = ("def h_{n}(recs,cfg):  # {x}\n    o=[]\n    for i,r in enumerate(recs):\n"
            "        if not ok(r,cfg.s_{n}): continue\n        o.append(tx(r,{n}.0,i))\n    return o\n"
            "[2026-08-27T0{m}:{nn}:12Z] INFO w-{n} did {n}00 recs in {n}.4s {x}\n\n")
    s = []; n = 0; c = 0
    while c < approx_tokens*3:
        s.append(unit.format(n=(n%97)+1, nn=(n%59)+1, m=(n%9), x=nonce)); c += len(s[-1]); n += 1
    return "".join(s)

GEN_ECHO = ("Reproduce the definitions of h_1 through h_12 from the material above, "
            "verbatim, exactly as written, including comments. Nothing else.")

for tgt in [8000, 32000]:
    for trial in range(1, 4):
        content = filler(tgt, f"gecho-{tgt}-{trial}") + "\n\n" + GEN_ECHO
        body = dict(model="qwen38-27b-mtp", temperature=0,
                    messages=[{"role":"user","content":content}], max_tokens=512,
                    stream=True, stream_options={"include_usage":True})
        t0 = time.time(); tf = None; comp = prompt = None
        import json as J
        with requests.post(base+"/v1/chat/completions", json=body, stream=True, timeout=900) as r:
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "): continue
                d = line[6:]
                if d == b"[DONE]": break
                try: j = J.loads(d)
                except Exception: continue
                chs = j.get("choices") or []
                delta = (chs[0].get("delta") if chs else {}) or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    if tf is None: tf = time.time()
                if j.get("usage"):
                    prompt = j["usage"].get("prompt_tokens"); comp = j["usage"].get("completion_tokens")
        t1 = time.time(); tf = tf or t1
        dt = max(t1-tf, 1e-6)
        print(f"{tag} tgt={tgt} trial={trial} prompt={prompt} gen={comp} dec={dt:.2f}s dtps={(comp-1)/dt:.1f}" if comp else f"{tag} {tgt} {trial} no usage", flush=True)
