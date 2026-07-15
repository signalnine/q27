#!/usr/bin/env bash
# batch_ab.sh -- continuous-batching P1 headline A/B
# (docs/plans/2026-07-14-continuous-batching.md Task 11).
#
# Measures whether the Q27_BATCH=1 conductor (fused verify round across 2
# concurrent decode requests, one union weight sweep) pays vs the FIFO
# round-interleave baseline. Vanilla qwen ONLY (standing rule), w16 serving
# build, fp8 KV, PMIN 0.5, MAXD auto, 2 slots x 32K.
#
# Legs (fresh server per leg on $PORT):
#   A  baseline-concurrent  Q27_BATCH=0                 codegen+docs fired together
#   B  batched-concurrent   Q27_BATCH=1                 same procedure
#   C  batched-vgemm        Q27_BATCH=1 Q27_BATCH_GEMM=1  prices the A1 bitwise
#      GEMM-family policy (unions forced onto k_vgemm instead of the
#      solo-matching GEMV family)
#   D  solo-regression (A10)  single sequential requests, 3x each payload,
#      under Q27_BATCH=0 and =1: per-request tps p50 must match within 2%
#      (the conductor k==1 fallthrough must be free)
#
# Procedure per concurrent leg: 1 sequential warmup pass of both payloads
# (lands per-slot prefix snapshots so measured reps are decode-dominated),
# then $REPS measured repetitions firing both payloads simultaneously and
# re-firing the pair only after both return. PRIMARY metric per rep:
#   aggregate = (dec_codegen + dec_docs) / (max(curl end) - min(curl start))
# median over reps. Secondary: per-request tps medians, bat= telemetry,
# completion-text md5s (greedy: A and B texts must match per payload unless
# a suffix-round trim fork fires -- the documented A1 policy fork, Task 10).
#
# After each batched leg's measured server, ONE extra UNMEASURED server pass
# runs with Q27_BATCH_DBG=1 Q27_PHASE_STATS=1 (env is process-level, so the
# debug stderr needs its own server -- measured reps stay debug-free) to
# capture per-round want->granted trim lines and the phase-field behavior of
# fused rounds ([req] phd/phv under BATCH=1: fused rounds have no wall
# buckets yet, Task 9 TODO -- fields must be zeros/absent, not garbage).
#
# Payloads are jq copies of the accept payloads with max_tokens raised
# 256 -> 512: the longer decode window makes the concurrent overlap dominate
# the window wall (prefill is snapshot-hit on measured reps) and shrinks the
# fixed HTTP+prefill edges' share of the aggregate metric.
#
# Usage: bash tools/batch_ab.sh          (GPU 0 must be free;
#        launch via systemd-run --user for crash-safety on long runs)
# Env:   MODEL TOK PORT(8199) REPS(3) LEGS("A B C D") PAYLOADS("codegen docs")
#        BIN(build/q27-server-w16) MAXTOK(512) OUT(scratchpad/batch_ab_out)
set -u
cd "$(dirname "$0")/.."
MODEL=${MODEL:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.q27}
TOK=${TOK:-/mnt/ai/models/qwen36-27b-mtp/qwen36-27b-mtp.tok}
PORT=${PORT:-8199}
REPS=${REPS:-3}
LEGS=${LEGS:-"A B C D"}
PAYLOADS=${PAYLOADS:-"codegen docs"}
BIN=${BIN:-build/q27-server-w16}
MAXTOK=${MAXTOK:-512}
OUT=${OUT:-scratchpad/batch_ab_out}
# KV env-overridable (KV=turbo3 for the t3 aggregate leg, 2026-07-15)
GATEENV="Q27_KV=${KV:-fp8} Q27_PMIN=0.5 Q27_MAXD=auto"
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$OUT"
SRV=""
stop_server() { [ -n "$SRV" ] && kill "$SRV" 2>/dev/null && wait "$SRV" 2>/dev/null; SRV=""; }
trap stop_server EXIT

for p in $PAYLOADS; do
  SRCJ=scratchpad/accept_payload_${p}.json
  [ -f "$SRCJ" ] || { echo "missing $SRCJ (run tools/make_payloads.py)"; exit 1; }
  jq --argjson mt "$MAXTOK" '.max_tokens = $mt' "$SRCJ" \
    > "scratchpad/batch_ab_${p}_${MAXTOK}.json"
done

start_server() { # $1=extra env ("Q27_BATCH=1 ...")  $2=log path
  env $GATEENV $1 "$BIN" "$MODEL" "$TOK" --port "$PORT" --ctx 32768 \
    --slots 2 --slot1-ctx 32768 --no-think --fast-head >"$2" 2>&1 &
  SRV=$!
  for i in $(seq 1 150); do
    curl -s -m 2 "localhost:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SRV" 2>/dev/null || break
    sleep 2
  done
  echo "FATAL: server did not come up ($1)"; tail -5 "$2"; exit 1
}

fire() { # $1=payload name  $2=output tag -- records start/end + text md5
  local body="scratchpad/batch_ab_${1}_${MAXTOK}.json" t0 t1 md5
  t0=$(date +%s.%N)
  curl -s -m 900 "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    --data-binary @"$body" >"$OUT/$2.json"
  t1=$(date +%s.%N)
  md5=$(jq -r '.choices[0].text // empty' "$OUT/$2.json" | md5sum | cut -d' ' -f1)
  echo "$1 $t0 $t1 $md5" >"$OUT/$2.time"
}

rep_concurrent() { # $1=tag prefix -- fire both payloads simultaneously
  local pids=() p
  for p in $PAYLOADS; do fire "$p" "${1}_${p}" & pids+=($!); done
  wait "${pids[@]}"
}

run_concurrent_leg() { # $1=leg name  $2=extra env
  echo "=== leg $1 ($2) ==="
  start_server "$2" "$OUT/server_$1.log"
  for p in $PAYLOADS; do fire "$p" "${1}_warm_${p}"; done   # snapshots land
  for r in $(seq 1 "$REPS"); do rep_concurrent "${1}_r${r}"; done
  stop_server
}

run_debug_pass() { # $1=leg name  $2=extra env -- unmeasured, DBG+PHASE on
  echo "=== leg $1 debug pass (unmeasured) ==="
  start_server "$2 Q27_BATCH_DBG=1 Q27_PHASE_STATS=1" "$OUT/server_${1}dbg.log"
  for p in $PAYLOADS; do fire "$p" "${1}dbg_warm_${p}"; done
  rep_concurrent "${1}dbg_r1"
  stop_server
}

run_solo_leg() { # $1=leg name  $2=extra env -- sequential singles (A10)
  echo "=== leg $1 solo ($2) ==="
  start_server "$2" "$OUT/server_$1.log"
  for p in $PAYLOADS; do fire "$p" "${1}_warm_${p}"; done
  for r in $(seq 1 "$REPS"); do
    for p in $PAYLOADS; do fire "$p" "${1}_r${r}_${p}"; done
  done
  stop_server
}

for leg in $LEGS; do
  case $leg in
    A) run_concurrent_leg A "Q27_BATCH=0" ;;
    B) run_concurrent_leg B "Q27_BATCH=1"
       run_debug_pass    B "Q27_BATCH=1" ;;
    C) run_concurrent_leg C "Q27_BATCH=1 Q27_BATCH_GEMM=1"
       run_debug_pass    C "Q27_BATCH=1 Q27_BATCH_GEMM=1" ;;
    D) run_solo_leg D0 "Q27_BATCH=0"
       run_solo_leg D1 "Q27_BATCH=1" ;;
    *) echo "unknown leg $leg"; exit 1 ;;
  esac
done

python3 - "$OUT" "$REPS" $PAYLOADS <<'PYEOF'
import glob, os, re, statistics as st, sys
out, reps = sys.argv[1], int(sys.argv[2])
pays = sys.argv[3:]

def times(tag):
    f = os.path.join(out, tag + ".time")
    if not os.path.exists(f): return None
    p, t0, t1, md5 = open(f).read().split()
    return dict(pay=p, t0=float(t0), t1=float(t1), md5=md5)

def reqs(leg):
    f = os.path.join(out, f"server_{leg}.log")
    if not os.path.exists(f): return []
    return [l for l in open(f) if "[req]" in l]

def fld(l, name, cast=float):
    m = re.search(rf" {name}=([\d.]+)", l)
    return cast(m.group(1)) if m else None

def batfld(l):
    m = re.search(r" bat=([\d.]+),(\d+)", l)
    return (float(m.group(1)), int(m.group(2))) if m else None

# payload attribution: warmup fired in $PAYLOADS order -> prompt= per payload
def attr_map(leg):
    rl = reqs(leg)
    return {fld(rl[i], "prompt", int): pays[i] for i in range(len(pays))} if len(rl) >= len(pays) else {}

conc = {}
for leg in ("A", "B", "C"):
    rl = reqs(leg)
    if not rl: continue
    am = attr_map(leg)
    body = rl[len(pays):]                       # drop warmup lines
    assert len(body) == reps * len(pays), f"leg {leg}: {len(body)} measured [req]"
    aggs, per, md5s, decs, bats = [], {p: [] for p in pays}, {p: set() for p in pays}, {p: set() for p in pays}, []
    for r in range(1, reps + 1):
        ts = [times(f"{leg}_r{r}_{p}") for p in pays]
        window = max(t["t1"] for t in ts) - min(t["t0"] for t in ts)
        chunk = body[(r - 1) * len(pays): r * len(pays)]
        dec = sum(fld(l, "dec", int) for l in chunk)
        aggs.append(dec / window)
        for l in chunk:
            p = am[fld(l, "prompt", int)]
            per[p].append(fld(l, "tps"))
            decs[p].add(fld(l, "dec", int))
            b = batfld(l)
            if b: bats.append(b)
        for t in ts: md5s[t["pay"]].add(t["md5"])
    conc[leg] = dict(agg=st.median(aggs), aggs=aggs, per=per, md5=md5s, dec=decs, bat=bats)

print("\n================ batch_ab results ================")
for leg in ("A", "B", "C"):
    if leg not in conc: continue
    c = conc[leg]
    pstr = "  ".join(f"{p}: tps_med={st.median(c['per'][p]):6.1f} dec={sorted(c['dec'][p])}" for p in pays)
    bstr = f"  bat_med={st.median(b[0] for b in c['bat']):.1f} fused_med={int(st.median(b[1] for b in c['bat']))}" if c["bat"] else ""
    det = " ".join(f"{p}:{'OK' if len(c['md5'][p]) == 1 else 'NONDET'}" for p in pays)
    print(f"leg {leg}: aggregate_med={c['agg']:7.1f} t/s  reps={['%.1f' % a for a in c['aggs']]}{bstr}")
    print(f"        {pstr}  rep-det: {det}")
if "A" in conc and "B" in conc:
    r = conc["B"]["agg"] / conc["A"]["agg"]
    print(f"\nB/A aggregate ratio = {r:.2f}x  (bar 1.3x) -> {'PASS' if r >= 1.3 else 'FAIL'}")
if "B" in conc and "C" in conc:
    d = 100 * (conc["C"]["agg"] / conc["B"]["agg"] - 1)
    print(f"C/B (always-vgemm vs family-match policy) = {d:+.1f}%")
# cross-leg text identity (greedy): A vs B vs C per payload
for p in pays:
    sets = {leg: conc[leg]["md5"][p] for leg in conc}
    ref = sets.get("A")
    diffs = [leg for leg in ("B", "C") if leg in sets and sets[leg] != ref]
    if ref is not None:
        print(f"text {p}: A md5 {sorted(ref)} " +
              ("== B/C" if not diffs else f"DIVERGES in {diffs}: " +
               " ".join(f"{l}={sorted(sets[l])}" for l in diffs)))
# leg D solo regression (A10)
d0, d1 = reqs("D0"), reqs("D1")
if d0 and d1:
    print("\nleg D solo regression (A10, bar: |delta| < 2%):")
    am0 = attr_map("D0")
    for p in pays:
        def tps_of(rl):
            return [fld(l, "tps") for l in rl[len(pays):] if am0.get(fld(l, "prompt", int)) == p]
        t0m, t1m = st.median(tps_of(d0)), st.median(tps_of(d1))
        dl = 100 * (t1m / t0m - 1)
        print(f"  {p:8s} BATCH=0 p50={t0m:6.1f}  BATCH=1 p50={t1m:6.1f}  "
              f"delta={dl:+.2f}%  {'PASS' if abs(dl) < 2 else 'FAIL'}")
# debug-pass evidence: trim lines + phase fields under BATCH=1
for leg in ("B", "C"):
    f = os.path.join(out, f"server_{leg}dbg.log")
    if not os.path.exists(f): continue
    bat = [l.strip() for l in open(f) if l.startswith("[bat]")]
    trims = [l for l in bat if "->" in l and re.search(r"(\d+)->(\d+)", l) and
             any(int(a) != int(b) for a, b in re.findall(r"(\d+)->(\d+)", l))]
    rl = [l for l in open(f) if "[req]" in l][len(pays):]
    ph = [(fld(l, "phd"), fld(l, "phv"), fld(l, "phs", int)) for l in rl]
    print(f"\nleg {leg} debug pass: {len(bat)} [bat] round lines, {len(trims)} with trim; "
          f"phase fields (phd,phv,phs) on measured [req]: {ph}")
    for l in bat[:3]: print(f"  {l}")
print("=================================================")
PYEOF
