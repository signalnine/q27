#!/usr/bin/env bash
# reqlog_gate.sh -- server-level gate for [req] telemetry + multi-slot routing.
#
# Phase 1 (default single-slot server):
#   C1 exactly one [req] line per generation request (all APIs, stream+non)
#   C2 all fields parse; C3 rid increasing; C4 conv fingerprint semantics
#   C5 Anthropic turn-2 warm-hits the stable-prefix snapshot
#   C6 cold request hit=0/pf=prompt; C7 timing sanity; C8 api tags
#   C9 prompt>ctx refused cleanly, server stays healthy
# Phase 2 (--slots 2 --slot1-ctx 4096): R1 interleave-warm routing
#   C12 A1,B1,A2,B2 interleaved -> A2.hit>0 AND B2.hit>0
#   C13 slot affinity: A1.slot==A2.slot, B1.slot==B2.slot, A.slot!=B.slot
#   C15 oversize prompt -> end=refused, healthy
#
# Usage: bash tools/reqlog_gate.sh [BIN [PORT]]
# Env:   MODEL, TOK override model/tokenizer paths.
# Needs the GPU free (~25 GB peak in phase 2): stop any resident server first.

set -u
BIN=${1:-build/q27-server}
PORT=${2:-8199}
MODEL=${MODEL:-/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27}
TOK=${TOK:-/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.tok}
SRV=""

start_server() { # args: logfile, extra flags...
    local log=$1; shift
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

# ---------------- Phase 1: single slot, R0 battery ----------------
LOG1=$(mktemp /tmp/reqlog_gate1.XXXXXX.log)
echo "[gate] phase 1 (single slot): log $LOG1"
start_server "$LOG1" || exit 1

PORT=$PORT LOG=$LOG1 python3 - <<'PYEOF'
import json, os, re, sys, urllib.error, urllib.request

port, log = os.environ["PORT"], os.environ["LOG"]
base = f"http://127.0.0.1:{port}"

def post(path, body, timeout=300):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()

SYS_A = ("You are a terse assistant for a warehouse inventory system. Rules: "
         "answer in one short sentence, never apologize, prefer metric units, "
         "cite item codes verbatim, and treat every question as being about "
         "the Fremont warehouse unless another site is named. The Fremont "
         "site stores fasteners, adhesives, abrasives, and packaging film in "
         "aisles one through nine, with hazardous storage restricted to the "
         "caged section of aisle nine under compliance rule seventeen.")
SYS_B = ("You are a verbose poetry tutor who explains meter and rhyme with "
         "long florid examples drawn from nineteenth century verse, always "
         "quoting at least two lines and naming the poet and the year of "
         "publication before giving any judgement about the student's work.")
U1 = "Which aisle holds the packaging film, and what is restricted in aisle nine?"
U2 = "And which compliance rule covers that caged section?"

r1 = json.loads(post("/v1/messages", {"model": "q27", "max_tokens": 48,
    "system": SYS_A, "messages": [{"role": "user", "content": U1}]}))
reply = "".join(b.get("text", "") for b in r1["content"] if b.get("type") == "text")
post("/v1/messages", {"model": "q27", "max_tokens": 48, "system": SYS_A,
    "messages": [{"role": "user", "content": U1},
                 {"role": "assistant", "content": reply or "Aisle seven."},
                 {"role": "user", "content": U2}]})
post("/v1/messages", {"model": "q27", "max_tokens": 32, "system": SYS_B,
    "messages": [{"role": "user", "content": "Comment on: the sea, the sea."}]})
post("/v1/chat/completions", {"model": "q27", "max_tokens": 8,
    "messages": [{"role": "system", "content": SYS_A},
                 {"role": "user", "content": "Say OK."}]})
post("/v1/messages", {"model": "q27", "max_tokens": 24, "stream": True,
    "system": SYS_A, "messages": [{"role": "user", "content": U1}]})
# T6 oversize prompt: since 2f47508 the server returns an anthropic-shaped
# HTTP 400 (ctx-limit) instead of a 200 -- expect it, then require health.
try:
    post("/v1/messages", {"model": "q27", "max_tokens": 16,
        "messages": [{"role": "user", "content": "alpha bravo charlie delta " * 4000}]})
    sys.exit("T6: oversize prompt unexpectedly succeeded (want HTTP 400)")
except urllib.error.HTTPError as e:
    assert e.code == 400, f"T6: expected 400, got {e.code}"
with urllib.request.urlopen(base + "/health", timeout=10) as r:
    assert json.loads(r.read())["status"] == "ok", "server unhealthy after T6"

pat = re.compile(
    r"\[req\] rid=(\d+) api=(\w+) conv=([0-9a-f]{8,16}) qw_ms=([\d.]+) "
    r"tok_ms=([\d.]+) prompt=(\d+) hit=(\d+) ckpt=(-?\d+) pf=(\d+) "
    r"pf_ms=([\d.]+) dec=(\d+) dec_ms=([\d.]+) cb_ms=([\d.]+) rounds=(\d+) "
    r"tps=([\d.]+) end=([\w-]+)")
lines = [l for l in open(log) if "[req]" in l]
recs = []
for l in lines:
    m = pat.search(l)
    assert m, f"C2 FAIL: unparseable [req] line: {l!r}"
    g = m.groups()
    recs.append(dict(rid=int(g[0]), api=g[1], conv=g[2], qw=float(g[3]),
                     tok=float(g[4]), prompt=int(g[5]), hit=int(g[6]),
                     ckpt=int(g[7]), pf=int(g[8]), pf_ms=float(g[9]),
                     dec=int(g[10]), dec_ms=float(g[11]), cb=float(g[12]),
                     rounds=int(g[13]), tps=float(g[14]), end=g[15]))

def chk(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    return cond

ok = True
# 5, not 6: since 2f47508 the T6 oversize prompt 400s at validation (asserted
# above) and never reaches generation, so it emits no [req] line.
ok &= chk(len(recs) == 5, f"C1 five [req] lines (got {len(recs)})")
if len(recs) == 5:
    t1, t2, t3, t4, t5 = recs
    ok &= chk(all(recs[i]["rid"] < recs[i+1]["rid"] for i in range(4)),
              "C3 rid increasing")
    ok &= chk(t1["conv"] == t2["conv"], "C4a same conversation -> same conv")
    ok &= chk(t1["conv"] != t3["conv"], "C4b different system -> different conv")
    ok &= chk(t2["hit"] >= 50, f"C5 warm turn hit>=50 (got {t2['hit']})")
    ok &= chk(t2["pf"] == t2["prompt"] - t2["hit"], "C5b pf == prompt-hit")
    ok &= chk(t1["hit"] == 0 and t1["pf"] == t1["prompt"], "C6 cold: hit=0, pf=prompt")
    ok &= chk(t1["pf_ms"] > 0 and t1["dec"] > 0 and t1["dec_ms"] > 0
              and t1["rounds"] > 0 and t1["tps"] > 10, "C7 timing sanity")
    ok &= chk(t1["qw"] >= 0 and t1["tok"] >= 0 and t1["cb"] >= 0, "C7b nonneg")
    ok &= chk(t4["api"] == "oai" and t1["api"] == "anth" and t5["api"] == "anth",
              "C8 api tags")
    # C9 (oversize refused cleanly) is the T6 400-assert + post-T6 health probe
    # in the request section above; no [req] line to check anymore.
sys.exit(0 if ok else 1)
PYEOF
RC1=$?
stop_server
if [ $RC1 -ne 0 ]; then echo "[gate] phase 1 FAIL (log kept: $LOG1)"; exit 1; fi
echo "[gate] phase 1 PASS"

# ---------------- Phase 2: --slots 2, interleave-warm routing ----------------
sleep 10  # VRAM teardown before the second, larger instance
LOG2=$(mktemp /tmp/reqlog_gate2.XXXXXX.log)
echo "[gate] phase 2 (--slots 2): log $LOG2"
start_server "$LOG2" --slots 2 --slot1-ctx 4096 || exit 1

PORT=$PORT LOG=$LOG2 python3 - <<'PYEOF'
import json, os, re, sys, urllib.error, urllib.request

port, log = os.environ["PORT"], os.environ["LOG"]
base = f"http://127.0.0.1:{port}"

def post(path, body, timeout=300):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()

SYS_A = ("You are a terse assistant for a warehouse inventory system. Rules: "
         "answer in one short sentence, never apologize, prefer metric units, "
         "cite item codes verbatim, and treat every question as being about "
         "the Fremont warehouse unless another site is named. The Fremont "
         "site stores fasteners, adhesives, abrasives, and packaging film in "
         "aisles one through nine, with hazardous storage restricted to the "
         "caged section of aisle nine under compliance rule seventeen.")
SYS_B = ("You are a verbose poetry tutor who explains meter and rhyme with "
         "long florid examples drawn from nineteenth century verse, always "
         "quoting at least two lines and naming the poet and the year of "
         "publication before giving any judgement about the student's work.")
UA = "Which aisle holds the packaging film, and what is restricted in aisle nine?"
UB = "Comment on the line: the sea, the sea, the ever-rolling sea."

ra = json.loads(post("/v1/messages", {"model": "q27", "max_tokens": 48,
    "system": SYS_A, "messages": [{"role": "user", "content": UA}]}))
reply_a = "".join(b.get("text", "") for b in ra["content"] if b.get("type") == "text")
rb = json.loads(post("/v1/messages", {"model": "q27", "max_tokens": 48,
    "system": SYS_B, "messages": [{"role": "user", "content": UB}]}))
reply_b = "".join(b.get("text", "") for b in rb["content"] if b.get("type") == "text")
post("/v1/messages", {"model": "q27", "max_tokens": 48, "system": SYS_A,
    "messages": [{"role": "user", "content": UA},
                 {"role": "assistant", "content": reply_a or "Aisle seven."},
                 {"role": "user", "content": "And which compliance rule applies?"}]})
post("/v1/messages", {"model": "q27", "max_tokens": 48, "system": SYS_B,
    "messages": [{"role": "user", "content": UB},
                 {"role": "assistant", "content": reply_b or "A fine anapest."},
                 {"role": "user", "content": "Name the poet you quoted."}]})
# C15 oversize prompt: anthropic-shaped 400 since 2f47508 (no [req] line),
# then the server must stay healthy.
try:
    post("/v1/messages", {"model": "q27", "max_tokens": 16,
        "messages": [{"role": "user", "content": "alpha bravo charlie delta " * 4000}]})
    sys.exit("C15: oversize prompt unexpectedly succeeded (want HTTP 400)")
except urllib.error.HTTPError as e:
    assert e.code == 400, f"C15: expected 400, got {e.code}"
with urllib.request.urlopen(base + "/health", timeout=10) as r:
    assert json.loads(r.read())["status"] == "ok", "server unhealthy after oversize"

hitp = re.compile(r"\[req\] rid=(\d+) .*? hit=(\d+) .*? end=([\w-]+)")
slotp = re.compile(r" slot=(\d+)")
recs = []
for l in open(log):
    if "[req]" not in l:
        continue
    m = hitp.search(l)
    assert m, f"unparseable [req] line: {l!r}"
    s = slotp.search(l)
    recs.append(dict(rid=int(m.group(1)), hit=int(m.group(2)), end=m.group(3),
                     slot=int(s.group(1)) if s else None))

def chk(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    return cond

ok = True
ok &= chk(len(recs) == 4, f"C1' four [req] lines (got {len(recs)})")
if len(recs) == 4:
    a1, b1, a2, b2 = recs
    ok &= chk(a2["hit"] > 0, f"C12a interleaved A2 warm (hit={a2['hit']})")
    ok &= chk(b2["hit"] > 0, f"C12b interleaved B2 warm (hit={b2['hit']})")
    ok &= chk(all(r["slot"] is not None for r in recs), "C13a slot= present")
    if all(r["slot"] is not None for r in recs):
        ok &= chk(a1["slot"] == a2["slot"], "C13b A sticky slot")
        ok &= chk(b1["slot"] == b2["slot"], "C13c B sticky slot")
        ok &= chk(a1["slot"] != b1["slot"], "C13d A,B on different slots")
    # C15 (oversize refused, healthy) is the 400-assert + health probe above.
sys.exit(0 if ok else 1)
PYEOF
RC2=$?
stop_server
trap - EXIT
if [ $RC2 -eq 0 ]; then echo "[gate] reqlog gate PASS (both phases)"; else
    echo "[gate] phase 2 FAIL (log kept: $LOG2)"; fi
exit $RC2
