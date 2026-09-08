#!/usr/bin/env python3
# Controlled eviction probe for the P16 disk tier + P16b system-block entry
# (docs/plans/2026-09-08-prefill-attack.md phase 0 step 2b). All requests go
# through /v1/messages (the Claude Code path: system field, P8 stable split).
#   A1: sys S (~6.5K) + U1 (~21.5K)            -> cold; expect sys-cut persist + ~28K persist
#   A2: + a1 + U2 (~20K)                       -> P8 hit ~28K; expect ~48K persist
#   B : sys S_B + ~300 tok, other conversation -> evicts A's P8/P9 (below min, no persist)
#   A3: + a2 + U3 (short)                      -> expect [pfx] restore L=~48K from disk
#   C : sys S + short user, new conversation   -> expect restore of the system-block entry
#   A4: + a3 + U4 (short)                      -> second restore of A after C
import requests, sys, time

base = sys.argv[1] if len(sys.argv) > 1 else "http://172.17.0.1:8081"
TAG = sys.argv[2] if len(sys.argv) > 2 else "pfxprobe"

def filler(approx_tokens, nonce):
    unit = ("def h_{n}(recs,cfg):  # {x}\n    o=[]\n    for i,r in enumerate(recs):\n"
            "        if not ok(r,cfg.s_{n}): continue\n        o.append(tx(r,{n}.0,i))\n    return o\n"
            "[2026-08-27T0{m}:{nn}:12Z] INFO w-{n} did {n}00 recs in {n}.4s {x}\n\n")
    s = []; n = 0; c = 0
    while c < approx_tokens * 2.1:  # measured 2.1 chars/token on this filler (probe1)
        s.append(unit.format(n=(n % 97) + 1, nn=(n % 59) + 1, m=(n % 9), x=nonce)); c += len(s[-1]); n += 1
    return "".join(s)

S = "You are a code auditor. Reference corpus follows.\n\n" + filler(6500, f"{TAG}-sys")
S_B = "You are a title generator. Reply with a short title only."
ASK = "\n\nReply with exactly the word OK and nothing else."

def msg(label, system, messages, max_tokens=24):
    body = dict(model="qwen38", system=system, messages=messages, max_tokens=max_tokens,
                temperature=1.0, top_p=0.95, stream=False)
    t0 = time.time()
    r = requests.post(base + "/v1/messages", json=body, timeout=900)
    wall = time.time() - t0
    j = r.json()
    u = j.get("usage", {})
    txt = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
    print(f"{label:3s} http={r.status_code} in={u.get('input_tokens')} out={u.get('output_tokens')} "
          f"wall={wall:.2f}s text={txt[:30]!r}", flush=True)
    return txt

U1 = filler(21500, f"{TAG}-u1") + ASK
U2 = filler(20000, f"{TAG}-u2") + ASK
U3 = "Second short follow-up: list two risks in the corpus above." + ASK
U4 = "Third short follow-up: name one function from the corpus." + ASK
A = "OK"

msg("A1", S, [{"role": "user", "content": U1}])
time.sleep(1.5)  # let the persist writer finish before the next boundary
msg("A2", S, [{"role": "user", "content": U1}, {"role": "assistant", "content": A},
              {"role": "user", "content": U2}])
time.sleep(1.5)
msg("B ", S_B, [{"role": "user", "content": "Title for: " + filler(300, f"{TAG}-b")}])
msg("A3", S, [{"role": "user", "content": U1}, {"role": "assistant", "content": A},
              {"role": "user", "content": U2}, {"role": "assistant", "content": A},
              {"role": "user", "content": U3}])
msg("C ", S, [{"role": "user", "content": "New conversation, same system block. " + filler(200, f"{TAG}-c") + ASK}])
msg("A4", S, [{"role": "user", "content": U1}, {"role": "assistant", "content": A},
              {"role": "user", "content": U2}, {"role": "assistant", "content": A},
              {"role": "user", "content": U3}, {"role": "assistant", "content": A},
              {"role": "user", "content": U4}])
