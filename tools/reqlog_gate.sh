#!/usr/bin/env bash
# reqlog_gate.sh -- server-level gate for the [req] per-request telemetry line.
#
# Starts a real q27-server on a test port, drives the OpenAI + Anthropic APIs
# with curl, then asserts on the server's stderr log:
#   C1 exactly one [req] line per generation request
#   C2 all fields parse (rid api conv qw_ms tok_ms prompt hit ckpt pf pf_ms
#      dec dec_ms cb_ms rounds tps end)
#   C3 rid strictly increasing
#   C4 conv fingerprint stable within a conversation, distinct across systems
#   C5 Anthropic turn 2 warm-hits the stable-prefix snapshot (hit>0)
#   C6 cold request: hit==0, pf==prompt
#   C7 timing sanity (pf_ms/dec_ms/rounds/tps positive on a normal gen)
#   C8 api tag matches endpoint (oai vs anth)
#   C9 prompt>ctx refused cleanly: end=refused, server stays healthy
#
# Usage: bash tools/reqlog_gate.sh [BIN [PORT]]
# Env:   MODEL, TOK override model/tokenizer paths.
# Needs the GPU free (~20 GB at --ctx 8192): stop any resident q27-server first.

set -u
BIN=${1:-build/q27-server}
PORT=${2:-8199}
MODEL=${MODEL:-/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27}
TOK=${TOK:-/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.tok}
LOG=$(mktemp /tmp/reqlog_gate.XXXXXX.log)

echo "[gate] server: $BIN port $PORT log $LOG"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  "$BIN" "$MODEL" "$TOK" --port "$PORT" --ctx 8192 --no-think >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT

for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo "[gate] FAIL: server died at startup"; tail -5 "$LOG"; exit 1; }
    sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[gate] FAIL: no health after 240s"; exit 1; }
echo "[gate] server healthy, driving requests"

PORT=$PORT LOG=$LOG python3 - <<'PYEOF'
import json, os, re, sys, urllib.request

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

# T1 anth cold
r1 = json.loads(post("/v1/messages", {"model": "q27", "max_tokens": 48,
    "system": SYS_A, "messages": [{"role": "user", "content": U1}]}))
reply = "".join(b.get("text", "") for b in r1["content"] if b.get("type") == "text")
# T2 anth warm turn 2 (history + real turn-1 reply)
post("/v1/messages", {"model": "q27", "max_tokens": 48, "system": SYS_A,
    "messages": [{"role": "user", "content": U1},
                 {"role": "assistant", "content": reply or "Aisle seven."},
                 {"role": "user", "content": U2}]})
# T3 anth different conversation
post("/v1/messages", {"model": "q27", "max_tokens": 32, "system": SYS_B,
    "messages": [{"role": "user", "content": "Comment on: the sea, the sea."}]})
# T4 openai chat
post("/v1/chat/completions", {"model": "q27", "max_tokens": 8,
    "messages": [{"role": "system", "content": SYS_A},
                 {"role": "user", "content": "Say OK."}]})
# T5 anth streaming
post("/v1/messages", {"model": "q27", "max_tokens": 24, "stream": True,
    "system": SYS_A, "messages": [{"role": "user", "content": U1}]})
# T6 prompt > ctx (8192): must refuse cleanly
post("/v1/messages", {"model": "q27", "max_tokens": 16,
    "messages": [{"role": "user", "content": "alpha bravo charlie delta " * 4000}]})
# server still alive?
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
ok &= chk(len(recs) == 6, f"C1 six [req] lines (got {len(recs)})")
if len(recs) == 6:
    t1, t2, t3, t4, t5, t6 = recs
    ok &= chk(all(recs[i]["rid"] < recs[i+1]["rid"] for i in range(5)),
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
    ok &= chk(t6["end"] == "refused", f"C9 oversize end=refused (got {t6['end']})")
sys.exit(0 if ok else 1)
PYEOF
RC=$?
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; trap - EXIT
if [ $RC -eq 0 ]; then echo "[gate] reqlog gate PASS"; else
    echo "[gate] reqlog gate FAIL (log kept: $LOG)"; fi
exit $RC
