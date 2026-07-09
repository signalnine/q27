#!/usr/bin/env bash
# interleave_gate.sh -- server-level gate for R1b round-granularity
# interleaved scheduling (docs/R1b-design.md).
#
# Phase 1 (--slots 2, interleaving on):
#   S1 overlap: short B (fired mid-A) completes while long A still streams
#   S2 determinism: A and B concurrent texts byte-identical to solo runs
#   S3 telemetry: A [req] shows yields>0 gw>0; B qw small; lines parse
#   S4 third request C (both engines busy) queues, completes, healthy
#   S6 B fired during A's multi-chunk prefill (A prompt ~4K = several PF_T)
# Phase 2 (Q27_NO_INTERLEAVE=1): B serializes behind A, A yields=0
#
# Usage: bash tools/interleave_gate.sh [BIN [PORT]]
# Env:   MODEL, TOK override model/tokenizer paths.
# Needs the GPU free (~25 GB peak): stop any resident server first.

set -u
BIN=${1:-build/q27-server}
PORT=${2:-8198}
MODEL=${MODEL:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27}
TOK=${TOK:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.tok}
SRV=""

start_server() { # args: logfile, extra flags...
    local log=$1; shift
    # a stale listener would make the whole phase test the wrong binary
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[gate] FAIL: port $PORT already serving -- stop it first"; return 1
    fi
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
      "$BIN" "$MODEL" "$TOK" --port "$PORT" --ctx 8192 --no-think "$@" >"$log" 2>&1 &
    SRV=$!
    for i in $(seq 1 120); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 $SRV 2>/dev/null || { echo "[gate] FAIL: server died at startup"; tail -5 "$log"; return 1; }
        sleep 2
    done
    echo "[gate] FAIL: no health after 240s"; return 1
}
stop_server() {
    [ -n "$SRV" ] && kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=""
}
trap 'stop_server' EXIT

# ---------------- Phase 1: interleaving on ----------------
LOG1=$(mktemp /tmp/interleave_gate1.XXXXXX.log)
echo "[gate] phase 1 (--slots 2, interleave): log $LOG1"
start_server "$LOG1" --slots 2 --slot1-ctx 4096 || exit 1

PORT=$PORT LOG=$LOG1 python3 - <<'PYEOF'
import json, os, re, sys, threading, time, urllib.request

port, log = os.environ["PORT"], os.environ["LOG"]
base = f"http://127.0.0.1:{port}"

def post(path, body, timeout=300):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()

def msg_text(raw):
    return "".join(b.get("text", "") for b in json.loads(raw)["content"]
                   if b.get("type") == "text")

# A: multi-chunk prefill (~4K tokens = several PF_T=1024 chunks) + long
# greedy decode (counting resists early EOS). B/C: short, distinct convs.
PAD = "alpha bravo charlie delta echo foxtrot golf hotel " * 500
A_BODY = {"model": "q27", "max_tokens": 400, "stream": True,
          "system": "You are a meticulous counting machine.",
          "messages": [{"role": "user", "content": PAD +
                        "\nCount from one to two hundred in English words, "
                        "one number per line, no digits."}]}
B_BODY = {"model": "q27", "max_tokens": 32,
          "system": "You are a terse geography quiz bot.",
          "messages": [{"role": "user", "content":
                        "Name the capital of France in one word."}]}
C_BODY = {"model": "q27", "max_tokens": 24,
          "system": "You are a laconic arithmetic engine.",
          "messages": [{"role": "user", "content": "What is 17 plus 25?"}]}

def stream_a(out):
    req = urllib.request.Request(base + "/v1/messages",
                                 json.dumps(A_BODY).encode(),
                                 {"Content-Type": "application/json"})
    text, delta_ts = [], []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            try:
                j = json.loads(line[6:])
            except Exception:
                continue
            if j.get("type") == "content_block_delta" and \
               j["delta"].get("type") == "text_delta":
                text.append(j["delta"]["text"])
                delta_ts.append(time.monotonic())
    out["text"] = "".join(text)
    out["delta_ts"] = delta_ts
    out["done"] = time.monotonic()

# solo baselines (greedy => concurrent reruns must reproduce these exactly)
a_solo = {}
stream_a(a_solo)
b_solo_text = msg_text(post("/v1/messages", B_BODY))
# evict A's snapshot: a fresh tiny conversation routes LRU to slot 0 and
# its serial path clears the cache -- so concurrent A below re-prefills
# COLD through multiple PF_T chunks (exercises the prefill yield path;
# without this, A warm-hits its solo run and B lands during decode).
post("/v1/messages", {"model": "q27", "max_tokens": 8,
                      "messages": [{"role": "user", "content": "hi"}]})
time.sleep(1.0)

# concurrent: A streams (cold, ~6 chunk prefill); B fires 0.35s in (mid-
# prefill of A); C 0.9s in (both engines busy -> queues for a free one)
a_run, b_run, c_run = {}, {}, {}
def run_b():
    time.sleep(0.35)
    b_run["text"] = msg_text(post("/v1/messages", B_BODY))
    b_run["done"] = time.monotonic()
def run_c():
    time.sleep(0.9)
    c_run["text"] = msg_text(post("/v1/messages", C_BODY))
    c_run["done"] = time.monotonic()
t0 = time.monotonic()
ta = threading.Thread(target=stream_a, args=(a_run,))
tb = threading.Thread(target=run_b)
tc = threading.Thread(target=run_c)
ta.start(); tb.start(); tc.start()
ta.join(); tb.join(); tc.join()

with urllib.request.urlopen(base + "/health", timeout=10) as r:
    assert json.loads(r.read())["status"] == "ok", "server unhealthy after phase 1"

# [req] lines: full R0 schema + R1b fields (gw=, yields= after end=)
pat = re.compile(
    r"\[req\] rid=(\d+) api=(\w+) conv=([0-9a-f]{8,16}) qw_ms=([\d.]+) "
    r"tok_ms=([\d.]+) prompt=(\d+) hit=(\d+) ckpt=(-?\d+) pf=(\d+) "
    r"pf_ms=([\d.]+) dec=(\d+) dec_ms=([\d.]+) cb_ms=([\d.]+) rounds=(\d+) "
    r"tps=([\d.]+) end=([\w-]+)")
xtra = re.compile(r" gw=([\d.]+) yields=(\d+)")
recs = []
for l in open(log):
    if "[req]" not in l:
        continue
    m = pat.search(l)
    assert m, f"unparseable [req] line: {l!r}"
    x = xtra.search(l)
    recs.append(dict(rid=int(m.group(1)), prompt=int(m.group(6)),
                     hit=int(m.group(7)), pf=int(m.group(9)),
                     qw=float(m.group(4)), end=m.group(16),
                     gw=float(x.group(1)) if x else None,
                     yields=int(x.group(2)) if x else None))
recs.sort(key=lambda r: r["rid"])  # [req] lines land in COMPLETION order

def chk(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)

ok = True
ok &= chk(len(recs) == 6, f"P1 six [req] lines (got {len(recs)})")
# rid = arrival order: a_solo, b_solo, evictor, A, B, C
if len(recs) == 6:
    ra, rb, rc = recs[3], recs[4], recs[5]
    a_after_b = [t for t in a_run.get("delta_ts", []) if t > b_run["done"]]
    ok &= chk(b_run["done"] < a_run["done"], "S1a B completes before A")
    ok &= chk(len(a_after_b) > 0, f"S1b A still streaming after B done "
              f"({len(a_after_b)} deltas after)")
    ok &= chk(a_run.get("text") == a_solo.get("text") and len(a_solo.get("text", "")) > 200,
              "S2a A text identical solo vs interleaved")
    ok &= chk(b_run.get("text") == b_solo_text and len(b_solo_text) > 0,
              "S2b B text identical solo vs interleaved")
    ok &= chk(ra["yields"] is not None and ra["yields"] > 0,
              f"S3a A yields>0 (got {ra['yields']})")
    ok &= chk(ra["gw"] is not None and ra["gw"] > 0, f"S3b A gw>0 (got {ra['gw']})")
    ok &= chk(rb["qw"] < 3000, f"S3c B qw under 3s (got {rb['qw']:.0f}ms)")
    ok &= chk(len(c_run.get("text", "")) > 0 and rc["end"] in ("end_turn", "n_max", "max_tokens", "eos"),
              f"S4 third request served (end={rc['end']})")
    ok &= chk(ra["hit"] == 0 and ra["pf"] > 3000,
              f"S6a A cold multi-chunk prefill (hit={ra['hit']} pf={ra['pf']})")
    # schedule-based, not threshold-based: B fully served before A's FIRST
    # delta proves B ran inside A's prefill window at chunk granularity. If
    # a future prefill speedup shrinks that window below B's runtime, this
    # fails LOUDLY -- restage with a bigger PAD rather than weakening it.
    a_first = a_run["delta_ts"][0] if a_run.get("delta_ts") else 0
    ok &= chk(b_run["done"] < a_first,
              f"S6b B served entirely inside A's prefill "
              f"(B done {a_first - b_run['done']:.2f}s before A's first delta)")
sys.exit(0 if ok else 1)
PYEOF
RC1=$?
stop_server
if [ $RC1 -ne 0 ]; then echo "[gate] phase 1 FAIL (log kept: $LOG1)"; exit 1; fi
echo "[gate] phase 1 PASS"

# ---------------- Phase 2: Q27_NO_INTERLEAVE=1 serializes ----------------
sleep 10  # VRAM teardown
LOG2=$(mktemp /tmp/interleave_gate2.XXXXXX.log)
echo "[gate] phase 2 (Q27_NO_INTERLEAVE=1): log $LOG2"
export Q27_NO_INTERLEAVE=1
start_server "$LOG2" --slots 2 --slot1-ctx 4096 || exit 1
unset Q27_NO_INTERLEAVE

PORT=$PORT LOG=$LOG2 python3 - <<'PYEOF'
import json, os, re, sys, threading, time, urllib.request

port, log = os.environ["PORT"], os.environ["LOG"]
base = f"http://127.0.0.1:{port}"

def post(path, body, timeout=300):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()

PAD = "alpha bravo charlie delta echo foxtrot golf hotel " * 500
A_BODY = {"model": "q27", "max_tokens": 400,
          "system": "You are a meticulous counting machine.",
          "messages": [{"role": "user", "content": PAD +
                        "\nCount from one to two hundred in English words, "
                        "one number per line, no digits."}]}
B_BODY = {"model": "q27", "max_tokens": 32,
          "system": "You are a terse geography quiz bot.",
          "messages": [{"role": "user", "content":
                        "Name the capital of France in one word."}]}

a_run, b_run = {}, {}
def run_a():
    post("/v1/messages", A_BODY, timeout=600)
    a_run["done"] = time.monotonic()
def run_b():
    time.sleep(0.3)
    post("/v1/messages", B_BODY)
    b_run["done"] = time.monotonic()
ta = threading.Thread(target=run_a)
tb = threading.Thread(target=run_b)
ta.start(); tb.start(); ta.join(); tb.join()

xtra = re.compile(r" gw=([\d.]+) yields=(\d+)")
recs = []
for l in open(log):
    if "[req]" not in l:
        continue
    x = xtra.search(l)
    recs.append(dict(gw=float(x.group(1)) if x else None,
                     yields=int(x.group(2)) if x else None))

def chk(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)

ok = True
ok &= chk(len(recs) == 2, f"P2 two [req] lines (got {len(recs)})")
ok &= chk(b_run["done"] > a_run["done"], "S5a B serialized behind A")
ok &= chk(all(r["yields"] == 0 for r in recs),
          f"S5b no yields when disabled (got {[r['yields'] for r in recs]})")
sys.exit(0 if ok else 1)
PYEOF
RC2=$?
stop_server
trap - EXIT
if [ $RC2 -eq 0 ]; then echo "[gate] interleave gate PASS (both phases)"; else
    echo "[gate] phase 2 FAIL (log kept: $LOG2)"; fi
exit $RC2
