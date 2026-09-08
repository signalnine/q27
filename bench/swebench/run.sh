#!/usr/bin/env bash
# swebench_run.sh <engine_label> [instance_filter]
# Public, reproducible agentic bench: drive Claude Code (sandboxed in the
# thunderdome/claude-code image, but WITHOUT the private orchestration -- just
# `docker run`) against N pinned SWE-bench_Verified instances, pointed at whatever
# engine serves the Anthropic API on host:8081. Measures decode t/s + wall time +
# a cheap "edited the right file" signal (overlap with the gold patch's files).
# No official Docker eval matrix (disk-light). engine_label picks the journal unit.
set -u
# self-contained: manifest lives next to this script; work/cache dirs overridable via env.
SP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE=${SWEBENCH_CACHE:-/mnt/ai/swebench-cache}   # bare repo mirrors (one-time clone)
WORK=${SWEBENCH_WORK:-/mnt/ai/swebench-work}      # per-instance workspaces (ephemeral)
IMG=${SWEBENCH_IMG:-thunderdome/claude-code:latest}
PORT=${SWEBENCH_PORT:-8081}
HOST=${SWEBENCH_HOST:-127.0.0.1}   # where the health probe looks; containers always use host.docker.internal
MAN="$SP/manifest.json"
ENGINE=${1:?usage: run.sh <q27|llama> [instance_filter]}
FILTER=${2:-}
UNIT=${SWEBENCH_UNIT:-$ENGINE-eval}
RES="$SP/results.$ENGINE.jsonl"
: >"$RES"

case "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://$HOST:$PORT/health)" in 200|401) ;; *) echo "engine $HOST:$PORT not up" >&2; exit 1;; esac
# Telemetry family (2026-09-07): q27 [req] journal lines, llama-server eval
# lines, or ninfer's --request-log-jsonl (SWEBENCH_REQLOG=<file>). Defaults by
# label prefix so q27lad / q27d2q8 / ninferd2 pick the right parser.
case "${SWEBENCH_TELEMETRY:-}" in
  q27|llama|ninfer) TELEM=$SWEBENCH_TELEMETRY ;;
  *) case "$ENGINE" in q27*) TELEM=q27 ;; ninfer*) TELEM=ninfer ;; *) TELEM=llama ;; esac ;;
esac
REQLOG=${SWEBENCH_REQLOG:-}
echo "=== SWE-bench agentic bench | engine=$ENGINE (:8081, journal $UNIT, telemetry $TELEM) | img=$IMG ==="
RUN_START="$(date '+%Y-%m-%d %H:%M:%S')"
RUN_EPOCH_MS=$(( $(date +%s) * 1000 ))

mapfile -t IDS < <(python3 -c "import json;print('\n'.join(m['instance_id'] for m in json.load(open('$MAN'))))")
for IID in "${IDS[@]}"; do
  [ -n "$FILTER" ] && [ "$IID" != "$FILTER" ] && continue
  # pull this instance's fields
  read -r REPO BASE REPOKEY < <(python3 -c "
import json
m=[x for x in json.load(open('$MAN')) if x['instance_id']=='$IID'][0]
print(m['repo'], m['base_commit'], m['repo'].replace('/','__'))")
  WS=$WORK/$ENGINE/$IID
  rm -rf "$WS"; mkdir -p "$WS/logs"
  git clone --quiet "$CACHE/$REPOKEY.git" "$WS/repo" 2>/dev/null
  git -C "$WS/repo" checkout -q "$BASE" 2>/dev/null || { echo "  $IID: checkout FAIL"; continue; }
  # task prompt (problem_statement + fix instruction), mounted read-only
  python3 -c "
import json
m=[x for x in json.load(open('$MAN')) if x['instance_id']=='$IID'][0]
open('$WS/task.txt','w').write(
'The repository \'%s\' is checked out at /workspace. Below is a GitHub issue.\n'
'Investigate and fix it by editing the source files under /workspace. Do NOT add\n'
'new test files and do NOT run the test suite -- just make the code changes needed.\n\n'
'ISSUE:\n%s\n' % (m['repo'], m['problem_statement']))"

  echo "[run] $IID ($REPO) ..."
  T0=$(date +%s)
  docker run --rm --user 1000:1000 --add-host host.docker.internal:host-gateway \
    -v "$WS/repo:/workspace" -w /workspace \
    -v "$WS/task.txt:/task.txt:ro" \
    -v "$WS/logs:/logs" -e HOME=/home/node \
    -e ANTHROPIC_BASE_URL=http://host.docker.internal:$PORT \
    -e ANTHROPIC_API_KEY=local -e ANTHROPIC_AUTH_TOKEN=local \
    -e CLAUDE_CODE_EFFORT_LEVEL=${SWEBENCH_EFFORT:-medium} \
    --entrypoint bash "$IMG" \
    -c 'P=$(cat /task.txt); timeout 700 claude -p --output-format stream-json --verbose --dangerously-skip-permissions -- "$P" >/logs/out.jsonl 2>/logs/err.log; echo $? >/logs/exit' \
    >/dev/null 2>&1
  T1=$(date +%s); WALL=$((T1-T0))
  git -C "$WS/repo" diff >"$WS/patch.diff" 2>/dev/null
  echo "[run] $IID done wall=${WALL}s"

  # per-instance metrics + gold-file overlap
  python3 - "$IID" "$REPO" "$WALL" "$WS" "$MAN" >>"$RES" <<'PY'
import json,sys,re,subprocess
iid,repo,wall,ws,man=sys.argv[1:6]
m=[x for x in json.load(open(man)) if x['instance_id']==iid][0]
diff=open(f"{ws}/patch.diff").read()
files=sorted(set(re.findall(r'^\+\+\+ b/(.+)$', diff, re.M)))
gold=set(m['gold_files'])
gold_hit=bool(set(files)&gold)
diff_lines=sum(1 for l in diff.splitlines() if l[:1] in '+-' and not l.startswith(('+++','---')))
turns=out_tok=0; exit_reason='?'
try:
    for ln in open(f"{ws}/logs/out.jsonl"):
        try: d=json.loads(ln)
        except: continue
        if d.get('type')=='result':
            turns=d.get('num_turns',0); out_tok=(d.get('usage') or {}).get('output_tokens',0)
            exit_reason=d.get('subtype','?')
except Exception: pass
try: exitc=open(f"{ws}/logs/exit").read().strip()
except Exception: exitc='?'
print(json.dumps({"iid":iid,"repo":repo,"wall_s":int(wall),"turns":turns,"out_tok":out_tok,
    "files_changed":files,"gold_files":sorted(gold),"gold_hit":gold_hit,"diff_lines":diff_lines,
    "nonempty":bool(files),"exit":exitc,"result":exit_reason}))
PY
done

# ---- engine decode telemetry for the run window ----
echo ""; echo "=== ENGINE DECODE ($ENGINE) over run window ==="
journalctl --user -u "$UNIT" --since "$RUN_START" --no-pager -o cat 2>/dev/null >"$SP/swebench_$ENGINE.journal"
python3 - "$TELEM" "$SP/swebench_$ENGINE.journal" "$RES" "$REQLOG" "$RUN_EPOCH_MS" <<'PY'
import sys,re,json,statistics as st
eng,jf,res,reqlog,t0ms=sys.argv[1:6]; t0ms=int(t0ms)
tps=[]; tok=0; ms=0.0; a=g=0; rounds=0; hit=prompt=0
if eng=='ninfer':
    # ninfer --request-log-jsonl: one request_done row per request; decode
    # seconds + completion tokens are the same split q27's [req] line carries.
    for ln in open(reqlog):
        try: d=json.loads(ln)
        except Exception: continue
        if d.get('event')!='request_done' or d.get('timestamp_unix_ms',0) < t0ms: continue
        r=d.get('result',{}); t=d.get('timings_seconds',{}); sp=d.get('speculative',{})
        n=r.get('completion_tokens',0); dec=t.get('decode',0.0)
        if n>=8 and dec>0:
            tps.append(n/dec); tok+=n; ms+=dec*1000.0; rounds+=sp.get('rounds',0)
            hit+=r.get('prefix_cache_hit_tokens',0); prompt+=r.get('prompt_tokens',0)
else:
  for ln in open(jf):
    if eng=='q27':
        m=re.search(r'^\[req\].* prompt=(\d+) hit=(\d+) .* dec=(\d+) dec_ms=([0-9.]+) .* rounds=(\d+) tps=([0-9.]+)', ln)
        if m and int(m.group(3))>=8:
            tps.append(float(m.group(6))); tok+=int(m.group(3)); ms+=float(m.group(4))
            rounds+=int(m.group(5)); prompt+=int(m.group(1)); hit+=int(m.group(2))
        s=re.search(r'sfx=(\d+),(\d+)', ln)
        if s: a+=int(s.group(1)); g+=int(s.group(2))
    else:
        m=re.search(r'eval time =\s*([0-9.]+) ms /\s*(\d+) tokens \(.*?,\s*([0-9.]+) tokens per second\)', ln)
        if m and 'prompt eval' not in ln and int(m.group(2))>=8:
            tps.append(float(m.group(3))); tok+=int(m.group(2)); ms+=float(m.group(1))
        d=re.search(r'draft acceptance = [0-9.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)', ln)
        if d: a+=int(d.group(1)); g+=int(d.group(2))
agg=tok/(ms/1000.0) if ms else 0
rows=[json.loads(l) for l in open(res)]
done=len(rows); ne=sum(r['nonempty'] for r in rows); gh=sum(r['gold_hit'] for r in rows)
wall=sum(r['wall_s'] for r in rows)
print(f"  instances: {done}  nonempty-diff: {ne}/{done}  edited-gold-file: {gh}/{done}")
print(f"  total wall: {wall}s ({wall/done:.0f}s/inst avg)" if done else "  no instances")
print(f"  decode: {agg:.1f} t/s agg / {st.median(tps) if tps else 0:.1f} med  ({len(tps)} reqs, {tok} tok)")
if rounds: print(f"  tok/round: {tok/rounds:.3f} ({rounds} rounds)")
if prompt: print(f"  prefix reuse: {hit/prompt*100:.1f}% of prompt tokens served from cache ({hit}/{prompt})")
if eng=='q27': print(f"  suffix drafter: {a} fires / {g} tok")
elif eng=='llama': print(f"  ngram-mod accept: {(a/g*100) if g else 0:.0f}% ({a}/{g})")
PY
echo "=== END ($ENGINE) ==="
