#!/usr/bin/env python3
# Live gate for P16b shared-cut PROMOTION (2026-09-08 (p) item 3): an old
# client's short entry is indexed, then several new-client sessions arrive
# whose system block shares a LONGER prefix among themselves. Before the fix
# every new session restored the old cut and re-prefilled the rest forever;
# after it, the second new session writes an exploratory sys_len entry, the
# third promotes the shared length, and later ones restore that.
#
#   O1: old client, system = HEAD(~7.1K) + OLD_TAIL      -> cold, entry at 7168
#   N1: new client, HEAD + MID + tail_1                   -> cold, shares 7110 with O1 -> entry at 6144
#   N2: HEAD + MID + tail_2, restores 6144                -> (fix) explores: sys_len-cut entry
#   N3: HEAD + MID + tail_3                               -> (fix) restores N2's entry (shape A) or promotes (B)
#   N4: HEAD + MID + tail_4                               -> (fix) restores the promoted entry, writes nothing
# Shape A (MID ~3000, tail ~80): the sys_len boundary (9216) lies INSIDE the
#   shared region, so N2's exploratory entry is itself the promoted one.
# Shape B (MID ~2800, tail ~600): the boundary (10240) lies inside the tail,
#   so N2's entry is a prefix of nobody; N3 cuts at the shared length (9216).
# A ~300-token foreign request follows every session so the next first turn
# is COLD for the VRAM tiers (P8/P9), as in production.
# Read [pfx]/[gen]/[req] from the unit's journal; usage: <base> <tag> <A|B>
import requests, sys, time

base = sys.argv[1] if len(sys.argv) > 1 else "http://172.17.0.1:8081"
TAG = sys.argv[2] if len(sys.argv) > 2 else "promo"
SHAPE = (sys.argv[3] if len(sys.argv) > 3 else "A").upper()
HEAD_TOK = 7110
MID_TOK, TAIL_TOK = (3000, 0) if SHAPE == "A" else (2800, 600)

def filler(approx_tokens, nonce):
    unit = ("def h_{n}(recs,cfg):  # {x}\n    o=[]\n    for i,r in enumerate(recs):\n"
            "        if not ok(r,cfg.s_{n}): continue\n        o.append(tx(r,{n}.0,i))\n    return o\n"
            "[2026-08-27T0{m}:{nn}:12Z] INFO w-{n} did {n}00 recs in {n}.4s {x}\n\n")
    s = []; n = 0; c = 0
    while c < approx_tokens * 2.1:  # ~2.1 chars/token on this filler
        s.append(unit.format(n=(n % 97) + 1, nn=(n % 59) + 1, m=(n % 9), x=nonce)); c += len(s[-1]); n += 1
    return "".join(s)

HEAD = "You are a code auditor. Reference corpus follows.\n\n" + filler(HEAD_TOK, f"{TAG}-head")
GITSTAT = "\n\ngitStatus: snapshot at the start of the conversation.\n\nRecent commits:\n"
OLD_TAIL = "\n\ngitStatus (old client).\n\nRecent commits:\n" + "".join(
    f"{(0x5eed * (k + 3)) & 0xffffffff:08x} Merge pull request #{900 + k} from legacy/branch-{k}\n" for k in range(5))
MID = filler(MID_TOK, f"{TAG}-mid") + GITSTAT
def tail(i):
    t = "".join(f"{(0x1a2b3c4d * (i + 1) * (k + 7)) & 0xffffffff:08x} Merge pull request #{1700 + i * 13 + k} from user{i}/branch-{k}\n"
                for k in range(5))
    if TAIL_TOK:
        t += "\nWorking tree notes:\n" + filler(TAIL_TOK, f"{TAG}-tail{i}")
    return t

def msg(label, system, user, max_tokens=16):
    body = dict(model="qwen38", system=system, messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=1.0, top_p=0.95, stream=False)
    t0 = time.time()
    r = requests.post(base + "/v1/messages", json=body, timeout=900)
    u = r.json().get("usage", {})
    print(f"{TAG} {label} http={r.status_code} in={u.get('input_tokens')} wall={time.time()-t0:.2f}s", flush=True)

def foreign(i):
    time.sleep(1.5)  # let the persist writer finish before the next cold prefill
    msg(f"F{i}", "You are a title generator. Reply with a short title only.",
        "Title for: " + filler(250, f"{TAG}-f{i}"))

msg("O1", HEAD + OLD_TAIL, f"Session O1 ({TAG}): reply with the word OK only.")
foreign(0)
for i in range(1, 5):
    msg(f"N{i}", HEAD + MID + tail(i), f"Session N{i} ({TAG}): reply with the word OK only.")
    foreign(i)
