#!/usr/bin/env python3
# Live gate for the P16b shared cut: three "sessions" whose system block is a
# shared ~7.1K-token body plus a per-session ~80-token tail (the Claude Code
# gitStatus shape), each a NEW conversation. Sizes are chosen so the shared
# body ends just BEFORE a 1024-chunk boundary (7168) and sys_len just after it:
#   S1: cold, nothing indexed        -> expect persist at 7168 (sys_len cut, old rule)
#   S2: shares ~7.1K with S1's entry -> expect "[pfx] system block ... shares N -> cut at 6144", persist L=6144
#   S3: shares with both             -> expect [pfx] restore L=6144 and hit=6144
# Read the [sysblk]/[pfx]/[gen]/[req] lines of the unit's journal afterwards.
import requests, sys, time

base = sys.argv[1] if len(sys.argv) > 1 else "http://172.17.0.1:8081"
TAG = sys.argv[2] if len(sys.argv) > 2 else "shared1"
BODY_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 7110

def filler(approx_tokens, nonce):
    unit = ("def h_{n}(recs,cfg):  # {x}\n    o=[]\n    for i,r in enumerate(recs):\n"
            "        if not ok(r,cfg.s_{n}): continue\n        o.append(tx(r,{n}.0,i))\n    return o\n"
            "[2026-08-27T0{m}:{nn}:12Z] INFO w-{n} did {n}00 recs in {n}.4s {x}\n\n")
    s = []; n = 0; c = 0
    while c < approx_tokens * 2.1:  # ~2.1 chars/token on this filler
        s.append(unit.format(n=(n % 97) + 1, nn=(n % 59) + 1, m=(n % 9), x=nonce)); c += len(s[-1]); n += 1
    return "".join(s)

SHARED = "You are a code auditor. Reference corpus follows.\n\n" + filler(BODY_TOK, f"{TAG}-sys") + \
         "\n\ngitStatus: snapshot at the start of the conversation.\n\nRecent commits:\n"
def tail(i):
    return "".join(f"{(0x1a2b3c4d * (i + 1) * (k + 7)) & 0xffffffff:08x} Merge pull request #{1700 + i * 13 + k} from user{i}/branch-{k}\n"
                   for k in range(5))

def msg(label, system, user, max_tokens=16):
    body = dict(model="qwen38", system=system, messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=1.0, top_p=0.95, stream=False)
    t0 = time.time()
    r = requests.post(base + "/v1/messages", json=body, timeout=900)
    u = r.json().get("usage", {})
    print(f"{label} http={r.status_code} in={u.get('input_tokens')} wall={time.time()-t0:.2f}s", flush=True)

for i in range(3):
    msg(f"S{i+1}", SHARED + tail(i), f"Session {i+1} ({TAG}): reply with the word OK only.")
    time.sleep(1.5)  # let the persist writer finish before the next cold prefill
    # Production shape: a ~300-token side request from another conversation lands
    # between sessions and clears the slot's P8/P9 VRAM entries, so the next
    # first turn is COLD (base == 0). Without it the P9 ring serves session 2.
    msg(f"F{i+1}", "You are a title generator. Reply with a short title only.",
        "Title for: " + filler(250, f"{TAG}-f{i}"))
