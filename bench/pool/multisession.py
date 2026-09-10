#!/usr/bin/env python3
"""Multi-session cache survival under KV reservations (issue #42 step 2).

A finished request's pages stay with its conversation as prefix cache until
another conversation needs them. Up front, each Claude-Code-shaped turn
reserves prompt + max_tokens (64K) and its idle lineage keeps all of it, so a
few long sessions overflow the pool and evict each other's caches; reserved
incrementally, an idle lineage holds only what was written. This drives
`sessions` conversations with distinct ~base_tokens prompts for `rounds`
turns each, one request at a time (round-robin across sessions, max_tokens
64000, one-word replies), and prints per-turn walls. Cache survival is read
from the server's [req] hit= by the gate: a warm continuation re-prefills a
few dozen tokens, an evicted one re-prefills the whole conversation.
usage: multisession.py <base_url> <base_tokens> [sessions=4] [rounds=3]"""
import json, sys, time, random, urllib.request, urllib.error

BASE = sys.argv[1].rstrip("/")
BASE_TOK = int(sys.argv[2])
SESS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
ROUNDS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
WORDS = ("river stone cloud amber lantern orchard velvet harbor meadow copper whisper "
         "falcon granite willow ember saffron thistle cobalt juniper marble tundra quartz "
         "maple sparrow canyon indigo glacier fennel obsidian heron").split()

def salad(n_words, seed):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) + str(r.randint(0, 99)) for _ in range(n_words))

def chat(messages, max_tokens):
    body = {"model": "q27", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "authorization": "Bearer x"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            return r.status, json.load(r), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:200].decode("utf-8", "replace")}, time.time() - t0

st, j, _ = chat([{"role": "user", "content": salad(2000, 0)}], 1)
assert st == 200, j
tpw = (j["usage"]["prompt_tokens"] - 20) / 2000.0
sessions = [[{"role": "user", "content": salad(int(BASE_TOK / tpw), 500 + s) + "\nReply with one word."}]
            for s in range(SESS)]
t_all = time.time(); walls = []; bad = 0
for r in range(ROUNDS):
    row = []
    for s in range(SESS):
        st, j, dt = chat(sessions[s], 64000)
        if st != 200:
            bad += 1; row.append(f"s{s}:{st}"); continue
        ans = j["choices"][0]["message"].get("content") or "ok"
        sessions[s] = sessions[s] + [{"role": "assistant", "content": ans},
                                     {"role": "user", "content": "One more word, please."}]
        walls.append(dt); row.append(f"s{s}:{j['usage']['prompt_tokens']}t/{dt:.1f}s")
    print(f"round {r + 1}: " + "  ".join(row), flush=True)
total = time.time() - t_all
warm = walls[SESS:]
print(f"TOTAL wall {total:.0f}s over {len(walls)} turns; rounds 2+ mean {sum(warm)/max(len(warm),1):.1f}s/turn; "
      f"{'ALL 200' if not bad else f'{bad} non-200'}")
sys.exit(1 if bad else 0)
