#!/usr/bin/env python3
# Depth x turn-size probe for the prefill attention split sweep (BUILDLOG 2026-09-08 (n)):
# run one server per Q27_PF_SPLIT setting, then this at depths 3000/25000/45000.
# One conversation at a given depth, then follow-up turns of ~5/40/128/512 new
# tokens; read pf/pf_ms from the [req] journal afterwards. /v1/messages.
import requests, sys, time
base, tag, depth = sys.argv[1], sys.argv[2], int(sys.argv[3])
def filler(approx_tokens, nonce):
    unit = ("def h_{n}(recs,cfg):  # {x}\n    o=[]\n    for i,r in enumerate(recs):\n"
            "        if not ok(r,cfg.s_{n}): continue\n        o.append(tx(r,{n}.0,i))\n    return o\n"
            "[2026-08-27T0{m}:{nn}:12Z] INFO w-{n} did {n}00 recs in {n}.4s {x}\n\n")
    s = []; n = 0; c = 0
    while c < approx_tokens * 2.1:
        s.append(unit.format(n=(n % 97) + 1, nn=(n % 59) + 1, m=(n % 9), x=nonce)); c += len(s[-1]); n += 1
    return "".join(s)
def msg(label, msgs):
    body = dict(model="qwen38", system="You are a code auditor.", messages=msgs, max_tokens=8,
                temperature=1.0, top_p=0.95, stream=False)
    t0 = time.time(); r = requests.post(base + "/v1/messages", json=body, timeout=900); u = r.json().get("usage", {})
    print(f"{tag} depth={depth} {label} http={r.status_code} in={u.get('input_tokens')} wall={time.time()-t0:.2f}s", flush=True)
msgs = [{"role": "user", "content": filler(depth, f"{tag}-{depth}") + "\n\nReply OK."}]
msg("cold", msgs)
for k, nt in enumerate((5, 40, 128, 512)):
    msgs.append({"role": "assistant", "content": "OK"})
    msgs.append({"role": "user", "content": (filler(nt, f"{tag}-{depth}-t{k}") if nt > 8 else "Go on.") + " Reply OK."})
    msg(f"turn+{nt}", msgs)
