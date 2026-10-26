#!/usr/bin/env python3
"""Per-shape dispatch sweep on the SERVED q27 binary (DFlash2 d2 config, no prefix cache).

One scratch systemd unit per leg (the knobs are process-wide statics), the
four depth prompts served in sequence, the [req] line per request recorded.
Judged on ms/round (dec_ms/rounds) and verify ms/round (phv/rounds), never on t/s.
Every load is wsum-gated against the tier's modal digest; a mismatched load is
torn down and retried (3 loads max). Every failure is an explicit ERROR row.
"""
import json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
Q = "/mnt/ai/projects/q27-master"
BIN = f"{Q}/build/q27-server"
MODEL = "/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.q27"
TOK = "/mnt/ai/models/qwen38-27b-mtp/qwen38-27b-mtp.tok"
PACK8 = "/mnt/ai/models/qwen38-27b-dflash2-bf16/qwen38-dflash2-q8-serve.d2w"
WSUM_REF = "b743d26b1f0562a9"
PORT = int(os.environ.get("SWEEP_PORT", "18082"))
ARGS = f"--host 127.0.0.1 --port {PORT} --think --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --think-budget 0".split()
BASE_ENV = {"CUDA_VISIBLE_DEVICES": "0", "Q27_KV": "fp8", "Q27_PRINT_WSUM": "1", "Q27_BATCH": "0",
            "Q27_DFLASH2": PACK8, "Q27_DFLASH2_RESERVE_GB": "3", "Q27_D2_TIMING": "1"}
DEPTHS = [int(x) for x in os.environ.get("SWEEP_DEPTHS", "4096,24576,45056,65536").split(",")]
MAX_TOKENS = int(os.environ.get("SWEEP_MAX_TOKENS", "1600"))

LEGS = {  # name -> extra env
    "kv_turbo3": {"Q27_KV": "turbo3"}, "kv_turbo5k": {"Q27_KV": "turbo5k"}, "kv_int8g64": {"Q27_KV": "int8g64"}, "kv_fp16": {"Q27_KV": "fp16"},
    "default_a": {}, "ns16": {"Q27_FDMMA_NS": "16"}, "ns32": {"Q27_FDMMA_NS": "32"},
    "ns43": {"Q27_FDMMA_NS": "43"}, "ns64": {"Q27_FDMMA_NS": "64"},
    "default_b": {}, "ns85": {"Q27_FDMMA_NS": "85"}, "ns128": {"Q27_FDMMA_NS": "128"},
    "stages2": {"Q27_FDMMA_STAGES": "2"}, "vgemm0": {"Q27_D2_VGEMM": "0"},
    "ds1": {"Q27_DS_SPLIT": "1"}, "ds2": {"Q27_DS_SPLIT": "2"}, "ds4": {"Q27_DS_SPLIT": "4"}, "default_c": {},
}

def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def journal(inv):
    return sh(["journalctl", "--user", f"_SYSTEMD_INVOCATION_ID={inv}", "-o", "cat", "--no-pager"]).stdout

def unit_active(unit):
    return sh(["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0

def stop_unit(unit):
    sh(["systemctl", "--user", "stop", unit])
    for _ in range(60):
        if not unit_active(unit): break
        time.sleep(1)
    sh(["systemctl", "--user", "reset-failed", unit])

def boot(leg, env, attempt):
    unit = f"sweep-{leg}-{attempt}"
    stop_unit(unit)
    cmd = ["systemd-run", "--user", "--unit", unit, "--collect"]
    for k, v in {**BASE_ENV, **env}.items(): cmd += ["-E", f"{k}={v}"]
    cmd += [BIN, MODEL, TOK] + ARGS
    r = sh(cmd)
    if r.returncode != 0: return unit, None, f"systemd-run failed: {r.stderr.strip()}"
    inv = sh(["systemctl", "--user", "show", unit, "-p", "InvocationID", "--value"]).stdout.strip()
    t0 = time.time()
    while time.time() - t0 < 420:
        j = journal(inv)
        if "listening on" in j:
            m = re.search(r"wsum: ([0-9a-f]{16})", j)
            return unit, inv, ("ok" if m and m.group(1) == WSUM_REF else f"wsum {m.group(1) if m else 'MISSING'}")
        if not unit_active(unit):
            return unit, inv, "died during startup: " + " | ".join(j.strip().splitlines()[-3:])
        time.sleep(2)
    return unit, inv, "boot timeout"

def request(text):
    body = json.dumps({"prompt": text, "max_tokens": MAX_TOKENS, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        resp.read()
    return time.time() - t0

def last_req_line(inv, seen):
    for _ in range(30):
        lines = [l for l in journal(inv).splitlines() if l.startswith("[req]")]
        if len(lines) > seen: return lines[-1], len(lines)
        time.sleep(1)
    return None, seen

def parse_req(line):
    d = dict(re.findall(r"(\w+)=([-\d.]+)", line))
    out = {k: float(d[k]) for k in ("prompt", "pf", "pf_ms", "dec", "dec_ms", "rounds", "phv", "phfd", "phd", "tps") if k in d}
    if out.get("rounds", 0) > 0:
        out["ms_round"] = out["dec_ms"] / out["rounds"]
        out["verify_round"] = out.get("phv", 0) / out["rounds"]
        out["draft_round"] = out.get("phfd", 0) / out["rounds"]
        out["tok_round"] = out["dec"] / out["rounds"]
    return out

def main():
    legs = sys.argv[1:] or list(LEGS)
    texts = {d: open(f"{HERE}/text_{d}.txt", "rb").read().decode("utf-8", "replace") for d in DEPTHS}
    results_path = f"{HERE}/results.jsonl"
    for leg in legs:
        env = LEGS[leg]
        inv = None; status = None
        for attempt in (1, 2, 3):
            unit, inv, status = boot(leg, env, attempt)
            print(f"[{leg}] boot {attempt}: {status}", flush=True)
            if status == "ok": break
            stop_unit(unit)
        if status != "ok":
            with open(results_path, "a") as f:
                f.write(json.dumps({"leg": leg, "env": env, "ERROR": f"boot: {status}"}) + "\n")
            continue
        seen = 0
        for d in DEPTHS:
            row = {"leg": leg, "env": env, "depth": d, "unit": unit, "inv": inv}
            try:
                row["wall_s"] = round(request(texts[d]), 2)
                line, seen = last_req_line(inv, seen)
                if line is None: row["ERROR"] = "no [req] line after request"
                else: row.update(parse_req(line)); row["req_line"] = line
            except Exception as e:
                row["ERROR"] = f"request: {e}"
            with open(results_path, "a") as f: f.write(json.dumps(row) + "\n")
            print(f"[{leg}] depth {d}: " + (row.get("ERROR") or
                  f"ms/round {row.get('ms_round', 0):.2f} verify {row.get('verify_round', 0):.2f} draft {row.get('draft_round', 0):.2f} tok/round {row.get('tok_round', 0):.2f} prompt {row.get('prompt')}"), flush=True)
        with open(f"{HERE}/journal_{leg}.log", "w") as f: f.write(journal(inv))
        stop_unit(unit)
    open(f"{HERE}/DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))

if __name__ == "__main__":
    main()
