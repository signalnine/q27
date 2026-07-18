#!/usr/bin/env python3
"""RunPod Serverless handler for q27 (stateless single-shot).

Design: q27 preallocates weights + the whole CUDA-graph zoo at boot
("boots = serves"), so booting per request would pay that cost every
time. Instead the server is started ONCE here at module import -- the
worker's warm lifetime amortizes it, and RunPod FlashBoot can snapshot
a ready worker. Each job just forwards to the local server.

This is the STATELESS shape: no conversation state is preserved between
jobs (q27's checkpoint/prefix-cache advantage needs session affinity,
which serverless does not give -- see deploy README). Single-shot
completions and one-turn chat are the right fit.

Input schema (job["input"]):
  prompt:      str   -- text completion (uses /v1/completions), OR
  messages:    list  -- [{role, content}...] one-turn chat (/v1/messages)
  max_tokens:  int   = 512
  temperature: float = 0.0   (>0 requires a sampled build; see README)
  top_p:       float = 1.0
  stop:        list  (optional, completions only)
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import runpod

PORT = int(os.environ.get("Q27_PORT", "8081"))
BIN = os.environ.get("Q27_BIN", "/opt/q27/q27-server-w8")
MODEL = os.environ.get("Q27_MODEL", "/runpod-volume/qwen36-27b-mtp-q4s.q27")
TOK = os.environ.get("Q27_TOK", "/runpod-volume/qwen36-27b-mtp.tok")
BOOT_TIMEOUT_S = int(os.environ.get("Q27_BOOT_TIMEOUT_S", "600"))
GEN_TIMEOUT_S = int(os.environ.get("Q27_GEN_TIMEOUT_S", "600"))
BASE = f"http://127.0.0.1:{PORT}"


def _log(*a):
    print("[q27-handler]", *a, file=sys.stderr, flush=True)


def _boot_server():
    for p in (MODEL, TOK, BIN):
        if not os.path.exists(p):
            raise RuntimeError(f"missing required path: {p} (attach the network volume / check env)")
    # localhost-only: the handler is the only client, no external surface.
    cmd = [BIN, MODEL, TOK, "--port", str(PORT), "--host", "127.0.0.1"]
    _log("starting:", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=dict(os.environ))
    t0 = time.time()
    while time.time() - t0 < BOOT_TIMEOUT_S:
        if proc.poll() is not None:
            raise RuntimeError(f"q27-server exited during boot (code {proc.returncode})")
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                if b"ok" in r.read():
                    _log(f"ready in {time.time() - t0:.0f}s")
                    return proc
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"q27-server did not become healthy within {BOOT_TIMEOUT_S}s")


# Heavy init at MODULE scope: FlashBoot snapshots a ready worker; a
# cold worker pays the weight-load + graph-zoo build once, then serves.
_SERVER = _boot_server()


def _post(path, body):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-api-key": "local"})
    with urllib.request.urlopen(req, timeout=GEN_TIMEOUT_S) as r:
        return json.load(r)


def handler(job):
    inp = job.get("input") or {}
    max_tokens = int(inp.get("max_tokens", 512))
    temperature = float(inp.get("temperature", 0.0))
    top_p = float(inp.get("top_p", 1.0))
    try:
        if "messages" in inp:  # one-turn chat via the Anthropic endpoint
            body = {"model": "q27", "max_tokens": max_tokens,
                    "messages": inp["messages"]}
            if temperature > 0:
                body["temperature"] = temperature
                body["top_p"] = top_p
            out = _post("/v1/messages", body)
            text = "".join(b.get("text", "") for b in out.get("content", [])
                           if b.get("type") == "text")
            return {"text": text, "usage": out.get("usage", {}), "stop_reason": out.get("stop_reason")}
        prompt = inp.get("prompt")
        if not prompt:
            return {"error": "provide 'prompt' (completion) or 'messages' (chat)"}
        body = {"model": "q27", "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": top_p}
        if inp.get("stop"):
            body["stop"] = inp["stop"]
        out = _post("/v1/completions", body)
        ch = out["choices"][0]
        return {"text": ch.get("text", ""), "finish_reason": ch.get("finish_reason"),
                "usage": out.get("usage", {})}
    except urllib.error.HTTPError as e:
        return {"error": f"q27 {e.code}: {e.read().decode()[:300]}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


runpod.serverless.start({"handler": handler})
