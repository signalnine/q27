#!/usr/bin/env python3
"""Incremental KV entitlements (issue #42 step 2) against a live multi-slot
q27-server. Two scenarios, each with a per-request wall bound (a hang fails):

  burst: four concurrent requests shaped like Claude Code turns -- a
    ~window/6 prompt and max_tokens 64000, short answers. Up front each
    reserves prompt + 64K and the pool admits two at a time; incremental,
    all four should start together. Reported: per-request queue wait from
    the response wall and the server's [req] qw_ms (read by the gate).
  growth: four concurrent requests whose prompts nearly fill the pool
    (4 x (prompt + 4K headroom) just under it) and that write long outputs,
    so their lineages must grow past the pool together -- the banker's check
    parks some at round boundaries and they finish as others free pages.
    Pass = all complete AND every output is an unbroken count from 1 (KV
    corruption from a bad remap mid-decode would break the sequence); the
    gate reads [kv-grow] lines for the parking.
  decode: three concurrent ~0.23-window prompts that each write ~3K tokens
    with max_tokens 64000. Up front two fit and the third waits for a whole
    turn; incremental, all three decode together.

usage: incremental_admission.py <base_url> <window_tokens> <pool_tokens> [burst|growth|decode|all]"""
import json, sys, time, random, threading, urllib.request, urllib.error

BASE = sys.argv[1].rstrip("/")
WINDOW, POOL = int(sys.argv[2]), int(sys.argv[3])
WHICH = sys.argv[4] if len(sys.argv) > 4 else "all"
WORDS = ("river stone cloud amber lantern orchard velvet harbor meadow copper whisper "
         "falcon granite willow ember saffron thistle cobalt juniper marble tundra quartz "
         "maple sparrow canyon indigo glacier fennel obsidian heron").split()

def salad(n_words, seed):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) + str(r.randint(0, 99)) for _ in range(n_words))

def chat(messages, max_tokens, timeout=1800):
    body = {"model": "q27", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "authorization": "Bearer x"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r), t0, time.time()
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:300].decode("utf-8", "replace")}, t0, time.time()

st, j, _, _ = chat([{"role": "user", "content": salad(2000, 0)}], 1)
assert st == 200, j
TPW = (j["usage"]["prompt_tokens"] - 20) / 2000.0
def words_for(tokens): return max(1, int(tokens / TPW))
print(f"calibration {TPW:.3f} tokens/word; window {WINDOW}, pool {POOL} tokens")

fails = []
def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok: fails.append(name)

def concurrent(prompts, max_tokens, bound_s):
    res = [None] * len(prompts)
    def one(i):
        res[i] = chat([{"role": "user", "content": prompts[i]}], max_tokens, timeout=bound_s)
    ths = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    t0 = time.time(); [t.start() for t in ths]; [t.join(bound_s) for t in ths]
    return res, time.time() - t0

if WHICH in ("burst", "all"):
    n = WINDOW // 6
    prompts = [salad(words_for(n), 100 + i) + "\nReply with exactly one word." for i in range(4)]
    res, wall = concurrent(prompts, 64000, 900)
    ok = all(r is not None and r[0] == 200 for r in res)
    detail = ", ".join(f"{(r[1].get('usage') or {}).get('prompt_tokens','?')}t/"
                       f"{(r[1].get('usage') or {}).get('completion_tokens','?')}out/{r[3]-r[2]:.0f}s" if r else "HUNG"
                       for r in res)
    check("burst-4x-64K-max_tokens", ok, f"wall {wall:.0f}s: {detail}")

def count_run(resp):
    """Longest unbroken 1,2,3,... run over the answer lines (reasoning too)."""
    if not resp: return 0
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    best = 0
    for field in ("content", "reasoning_content"):
        nxt = 1
        for line in (msg.get(field) or "").splitlines():
            s = line.strip().rstrip(".,")
            if s.isdigit():
                v = int(s)
                if v == nxt: nxt += 1
                elif v == 1: nxt = 2   # a fresh count restarts the run
            best = max(best, nxt - 1)
    return best

if WHICH in ("growth", "all"):
    # 4 x (prompt + 4K headroom + reserve) just under the pool; long outputs
    # push the sum past it, so lineages must grow and some must park.
    n = (POOL - 4 * (4096 + 32)) // 4 - 2000
    ask = "\nNow write the integers from 1 to 4000 in order, one per line, with no other text."
    prompts = [salad(words_for(n), 200 + i) + ask for i in range(4)]
    res, wall = concurrent(prompts, 9000, 1500)
    ok = all(r is not None and r[0] == 200 for r in res)
    outs = [(r[1].get("usage") or {}).get("completion_tokens", 0) if r else 0 for r in res]
    runs = [count_run(r[1]) if r else 0 for r in res]
    detail = ", ".join(f"{(r[1].get('usage') or {}).get('prompt_tokens','?')}t/{o}out/run{u}/{r[3]-r[2]:.0f}s" if r else "HUNG"
                       for r, o, u in zip(res, outs, runs))
    check("growth-past-pool", ok, f"wall {wall:.0f}s: {detail}")
    grown = sum(1 for o in outs if o > 4096)
    check("growth-exercised", grown >= 2, f"{grown} of 4 wrote past the 4K admission headroom")
    check("growth-content-intact", all(u >= 500 for u in runs),
          f"unbroken 1..N counts per output: {runs} (>= 500 each)")

if WHICH in ("decode", "all"):
    n = int(WINDOW * 0.23)
    ask = "\nNow write the integers from 1 to 700 in order, one per line, with no other text."
    prompts = [salad(words_for(n), 300 + i) + ask for i in range(3)]
    res, wall = concurrent(prompts, 64000, 1500)
    ok = all(r is not None and r[0] == 200 for r in res)
    runs = [count_run(r[1]) if r else 0 for r in res]
    detail = ", ".join(f"{(r[1].get('usage') or {}).get('prompt_tokens','?')}t/"
                       f"{(r[1].get('usage') or {}).get('completion_tokens','?')}out/run{u}/{r[3]-r[2]:.0f}s" if r else "HUNG"
                       for r, u in zip(res, runs))
    check("decode-burst-3x-64K-max_tokens", ok and all(u >= 300 for u in runs), f"wall {wall:.0f}s: {detail}")

print("RESULT:", "ALL PASS" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
