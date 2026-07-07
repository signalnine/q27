#!/usr/bin/env bash
# P15 constrain-tools gate (docs/plans/2026-07-07-constrain-tools.md C9-C13):
#  1. constrained E2E emits a grammar-valid call: name REGISTERED, JSON parses,
#     zero disengages, closer present, engage fired (C9/C13)
#  2. round-phase invariance: output bytes identical across Q27_PMIN unset /
#     0.5 / 1.0 (the marker lands at different in-round offsets) (C10)
#  3. canonical bitwise gate 4c4120c72056aba2bc2d2561471eafce (C11)
#  4. test_kernels PASS (C12; sanitizer run is a separate manual step)
# Needs the GPU free (stops nothing itself). Run from the repo root.
set -u
cd "$(dirname "$0")/.."
MODEL=${MODEL:-/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.q27}
TOK=${TOK:-/mnt/ai/models/qwopus-27b-mtp/qwopus-27b-mtp.tok}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
OUT=${OUT:-/tmp/constrain_gate.$$}
mkdir -p "$OUT"
fails=0
note() { echo "[gate] $*"; }
bad() { echo "[gate] FAIL: $*"; fails=$((fails + 1)); }

note "building constrain_e2e (dual-arch, matches NVCCFLAGS -- sm_120-only builds throw fatbin-probe errors under compute-sanitizer)..."
$NVCC -O2 -std=c++17 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120 -Xcompiler -Wall \
    tools/constrain_e2e.cu src/blocks.cu src/prefill.cu src/kernels.cu src/spec3.cu \
    src/device_model.cu src/loader.cpp src/tokenizer.cpp -o build/constrain_e2e \
    || { bad "constrain_e2e build"; echo "[gate] $fails FAILURES"; exit 1; }

run_leg() { # $1=label $2=env-assignments
    local label=$1 envs=$2
    note "leg $label (env: ${envs:-none})"
    env $envs ./build/constrain_e2e "$MODEL" "$TOK" -n 250 \
        --out "$OUT/$label.txt" > "$OUT/$label.stats" 2> "$OUT/$label.log"
    cat "$OUT/$label.stats"
}

run_leg base ""
sleep 12   # VRAM teardown race between back-to-back CLI loads
run_leg pmin05 "Q27_PMIN=0.5"
sleep 12
run_leg pmin10 "Q27_PMIN=1.0"

# C9/C13 assertions on the base leg
python3 - "$OUT" <<'EOF' || fails=$((fails + $?))
import json, re, sys
out = sys.argv[1]
fails = 0
stats = open(f"{out}/base.stats").read()
m = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", stats)}
if m.get("engaged", 0) < 1: print("[gate] FAIL: never engaged"); fails += 1
if m.get("disengaged", 0) != 0: print(f"[gate] FAIL: {m['disengaged']} disengages"); fails += 1
if m.get("refinish", 0) < 1: print("[gate] FAIL: refinish never fired"); fails += 1
text = open(f"{out}/base.txt").read()
calls = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S)
if not calls:
    print("[gate] FAIL: no closed <tool_call> block in output"); fails += 1
else:
    body = json.loads(calls[0])  # raises -> script fails via except? keep explicit
    name = body.get("name")
    if name not in ("getg_project", "run_tests"):
        print(f"[gate] FAIL: unregistered tool name {name!r}"); fails += 1
    if not isinstance(body.get("arguments"), dict):
        print(f"[gate] FAIL: arguments not an object"); fails += 1
    print(f"[gate] call OK: name={name} args={body.get('arguments')}")
sys.exit(fails)
EOF

# C10: byte-identity across round phrasings
if cmp -s "$OUT/base.txt" "$OUT/pmin05.txt" && cmp -s "$OUT/base.txt" "$OUT/pmin10.txt"; then
    note "round-phase invariance OK (3 legs byte-identical)"
else
    bad "round-phase invariance: legs differ"
    ls -la "$OUT"
fi

# C8: clear-at-claim leak test (plants a stale restrictive constraint)
sleep 12
if ./build/constrain_e2e "$MODEL" "$TOK" --leak-test 2>"$OUT/leak.log" | grep -q "LEAK_TEST=PASS"; then
    note "clear-at-claim leak test OK"
else
    bad "clear-at-claim leak test"
fi

# C11: canonical bitwise
sleep 12
MD5=$(./build/q27 "$MODEL" --tokens "760,6511,314,9338,369" --ctx 2048 -n 128 --spec 2>/dev/null \
    | grep '^generated:' | md5sum | cut -d' ' -f1)
if [ "$MD5" = "4c4120c72056aba2bc2d2561471eafce" ]; then
    note "canonical OK ($MD5)"
else
    bad "canonical changed: $MD5"
fi

# C12: kernel tests
sleep 12
if ./build/test_kernels "$MODEL" 2>/dev/null | tail -1 | grep -q "ALL PASS"; then
    note "test_kernels PASS"
else
    bad "test_kernels"
fi

echo
if [ "$fails" -eq 0 ]; then echo "[gate] ALL PASS (artifacts: $OUT)"; else echo "[gate] $fails FAILURES (artifacts: $OUT)"; fi
exit "$fails"
