#!/usr/bin/env python3
# Paired-seed arm-A probe: fixed nonce+seed per trial so two engine builds see
# byte-identical requests; depth-4-identical builds then produce identical
# tokens and the diff isolates the feature under test.
import requests, time, json, sys

base = sys.argv[1]; model = sys.argv[2]; tag = sys.argv[3]
SAMPLER = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.05)

def filler(approx_tokens, nonce):
    unit = ("def h_{n}(recs,cfg):  # {x}\n    o=[]\n    for i,r in enumerate(recs):\n"
            "        if not ok(r,cfg.s_{n}): continue\n        o.append(tx(r,{n}.0,i))\n    return o\n"
            "[2026-08-27T0{m}:{nn}:12Z] INFO w-{n} did {n}00 recs in {n}.4s {x}\n\n")
    s = []; n = 0; c = 0
    while c < approx_tokens*3:
        s.append(unit.format(n=(n%97)+1, nn=(n%59)+1, m=(n%9), x=nonce)); c += len(s[-1]); n += 1
    return "".join(s)

GEN = ("Ignore the material above. Output a numbered list, one item per line, of 50 "
       "distinct two-word English noun phrases about weather. Nothing else.")

for tgt in [8000, 32000]:
    for seed in range(1, 7):
        content = filler(tgt, f"pair-{tgt}-{seed}") + "\n\n" + GEN
        body = dict(model=model, messages=[{"role":"user","content":content}],
                    max_tokens=512, stream=True, stream_options={"include_usage":True},
                    seed=seed, **SAMPLER)
        t0 = time.time(); tf = None; comp = prompt = None
        with requests.post(base+"/v1/chat/completions", json=body, stream=True, timeout=900) as r:
            if r.status_code != 200:
                print(tag, tgt, seed, "HTTP", r.status_code); continue
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "): continue
                d = line[6:]
                if d == b"[DONE]": break
                try: j = json.loads(d)
                except Exception: continue
                chs = j.get("choices") or []
                delta = (chs[0].get("delta") if chs else {}) or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    if tf is None: tf = time.time()
                if j.get("usage"):
                    prompt = j["usage"].get("prompt_tokens"); comp = j["usage"].get("completion_tokens")
        t1 = time.time()
        if tf is None: tf = t1
        dt = max(t1-tf, 1e-6)
        print(f"{tag} tgt={tgt} seed={seed} prompt={prompt} gen={comp} dec={dt:.2f}s dtps={(comp-1)/dt:.1f}" if comp else f"{tag} {tgt} {seed} no usage", flush=True)
