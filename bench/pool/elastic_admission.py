#!/usr/bin/env python3
"""Admission behaviour under elastic multi-slot windows (issue #42).

Against a running multi-slot q27-server (auto ctx, KV pool on), checks:
  1. beyond-old-cap: one request larger than the old divided window admits
     and completes (the 4-slot 5090 divided window was 45056);
  2. four concurrent mid-size requests all complete;
  3. pressure: four concurrent requests whose entitlements sum past the pool
     all complete -- the excess waits for a finishing slot and scavenges it,
     none hangs (per-request wall bounded);
  4. oversized: a prompt past the pool window is refused fast (400), not
     queued;
  5. continuation: a second turn of test 1's conversation reuses its prefix.
Prompts are distinct pseudo-random word salads (no cross-request reuse);
temperature 0, tiny max_tokens, so walls are prefill-dominated.
usage: elastic_admission.py <base_url> <window_tokens> [old_cap]"""
import json, sys, time, random, threading, urllib.request, urllib.error

BASE = sys.argv[1].rstrip("/")
WINDOW = int(sys.argv[2])                 # per-slot window the boot printed
OLD_CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 45056
WORDS = ("river stone cloud amber lantern orchard velvet harbor meadow copper whisper "
         "falcon granite willow ember saffron thistle cobalt juniper marble tundra quartz "
         "maple sparrow canyon indigo glacier fennel obsidian heron").split()

def salad(n_words, seed):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) + str(r.randint(0, 99)) for _ in range(n_words))

def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "authorization": "Bearer x"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:300].decode("utf-8", "replace")}, time.time() - t0

def chat(messages, max_tokens=8):
    return post("/v1/chat/completions", {"model": "q27", "messages": messages, "max_tokens": max_tokens,
                                          "temperature": 0, "stream": False})

# calibrate words -> tokens on this tokenizer (one small request)
st, j, _ = chat([{"role": "user", "content": salad(2000, 0)}], 1)
assert st == 200, j
TPW = (j["usage"]["prompt_tokens"] - 20) / 2000.0
def words_for(tokens): return max(1, int(tokens / TPW))
print(f"calibration: {TPW:.3f} tokens/word; window {WINDOW}, old divided cap {OLD_CAP}")

fails = []
def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok: fails.append(name)

# 1. beyond the old divided cap
n1 = int(min(OLD_CAP * 1.35, WINDOW * 0.5))
msgs1 = [{"role": "user", "content": salad(words_for(n1), 1) + "\nReply with one word."}]
st, j, dt = chat(msgs1)
check("beyond-old-cap", st == 200 and j["usage"]["prompt_tokens"] > OLD_CAP,
      f"status {st}, prompt {j.get('usage', {}).get('prompt_tokens')} tokens (old cap {OLD_CAP}), {dt:.1f}s")

def run_concurrent(sizes, seed0, label, bound_s):
    res = [None] * len(sizes)
    def one(i, n):
        res[i] = chat([{"role": "user", "content": salad(words_for(n), seed0 + i) + "\nReply with one word."}])
    ths = [threading.Thread(target=one, args=(i, n)) for i, n in enumerate(sizes)]
    t0 = time.time(); [t.start() for t in ths]; [t.join(bound_s) for t in ths]
    wall = time.time() - t0
    ok = all(r is not None and r[0] == 200 for r in res)
    detail = ", ".join(f"{(r[1].get('usage') or {}).get('prompt_tokens', '?')}t/{r[2]:.0f}s/{r[0]}" if r else "HUNG" for r in res)
    check(label, ok and wall < bound_s, f"wall {wall:.0f}s (bound {bound_s}s): {detail}")

# 2. four concurrent mid-size requests (fit together)
run_concurrent([int(WINDOW * 0.12)] * 4, 100, "four-concurrent", 600)
# 3. pressure: sum of entitlements ~1.4x the window -> the last must wait and scavenge
run_concurrent([int(WINDOW * 0.35)] * 4, 200, "pressure-over-pool", 900)
# 4. oversized
st, j, dt = chat([{"role": "user", "content": salad(words_for(int(WINDOW * 1.08)), 300)}])
check("oversized-refused", st == 400 and dt < 60, f"status {st} in {dt:.1f}s: {str(j)[:120]}")
# 5. continuation of test 1's conversation
if st is not None:
    first = chat(msgs1)
    ans = first[1]["choices"][0]["message"]["content"] if first[0] == 200 else "ok"
    st, j, dt = chat(msgs1 + [{"role": "assistant", "content": ans}, {"role": "user", "content": "And one more word?"}])
    check("continuation", st == 200, f"status {st}, prompt {j.get('usage', {}).get('prompt_tokens')} tokens, {dt:.1f}s "
          "(prefix reuse: see the server's [req] hit= for this request)")
print("RESULT:", "ALL PASS" if not fails else f"FAILED {fails}")
sys.exit(1 if fails else 0)
