#!/usr/bin/env python3
# Warm-turn probe for the DFlash2 ring (2026-09-07): (A) three identical
# seeded requests -- runs 2 and 3 are prefix-cache warm turns (pf = a few
# tokens); (B) a two-turn conversation -- turn 2 re-renders turn 1's answer
# plus a new user message, the realistic agentic warm turn. Pair with the
# [req] journal (prompt/hit/pf/rounds/dec) to read tok/round per request:
#   journalctl --user -u <unit> _SYSTEMD_INVOCATION_ID=<id> -o cat | grep '\[req\]'
import requests, sys, time

base = sys.argv[1]; model = sys.argv[2]; tag = sys.argv[3]
SAMPLER = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.05)

def filler(approx_tokens, nonce):
    unit = ("def h_{n}(recs,cfg):  # {x}\n    o=[]\n    for i,r in enumerate(recs):\n"
            "        if not ok(r,cfg.s_{n}): continue\n        o.append(tx(r,{n}.0,i))\n    return o\n"
            "[2026-08-27T0{m}:{nn}:12Z] INFO w-{n} did {n}00 recs in {n}.4s {x}\n\n")
    s = []; n = 0; c = 0
    while c < approx_tokens * 3:
        s.append(unit.format(n=(n % 97) + 1, nn=(n % 59) + 1, m=(n % 9), x=nonce)); c += len(s[-1]); n += 1
    return "".join(s)

GEN = ("Ignore the material above. Output a numbered list, one item per line, of 50 "
       "distinct two-word English noun phrases about weather. Nothing else.")
GEN2 = ("Now the same again, but 40 distinct two-word English noun phrases about the ocean. "
        "Nothing else.")

def chat(messages, seed, max_tokens):
    body = dict(model=model, messages=messages, max_tokens=max_tokens, stream=False, seed=seed, **SAMPLER)
    t0 = time.time()
    r = requests.post(base + "/v1/chat/completions", json=body, timeout=900).json()
    m = r["choices"][0]["message"]; u = r.get("usage", {})
    print(f"{tag} seed={seed} prompt={u.get('prompt_tokens')} gen={u.get('completion_tokens')} "
          f"wall={time.time()-t0:.2f}s", flush=True)
    return m.get("content") or ""

# (A) three identical requests
content = filler(2000, f"{tag}-A") + "\n\n" + GEN
for i in range(3):
    chat([{"role": "user", "content": content}], 7, 256)

# (B) two-turn conversation, repeated for two seeds
for seed in (11, 12):
    c1 = filler(2000, f"{tag}-B{seed}") + "\n\n" + GEN
    a1 = chat([{"role": "user", "content": c1}], seed, 320)
    chat([{"role": "user", "content": c1}, {"role": "assistant", "content": a1},
          {"role": "user", "content": GEN2}], seed, 256)
