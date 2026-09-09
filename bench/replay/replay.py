#!/usr/bin/env python3
"""Sequential request replayer (2026-09-08, item 2 of the (p) agenda).

Feeds a server the exact request bodies a recording captured, one at a time
in recorded order, so two binaries see the IDENTICAL preceding sequence --
the only condition under which q27's seed-0 sampling reproduces a turn
(the drafter ring and every cache tier are history-dependent). Records per
request: the content hash of the streamed (or non-streamed) output in
delivery order, output tokens, TTFT and wall, and the tool_use names.

    bench/replay/replay.py <reqlog.jsonl> --base http://172.17.0.1:8081 --out A.jsonl
        [--limit N] [--start SEQ] [--timeout S] [--quiet]

The recording comes from q27-server with Q27_REQ_LOG=<file> (one JSONL line
per request: seq, t_ms, api, path, body as a string). Compare two replays
with bench/replay/replay_diff.py.
"""
import argparse, hashlib, json, sys, time
import requests

def sse_events(resp):
    """Yield parsed SSE data payloads (dicts) from a streaming response."""
    for raw in resp.iter_lines():
        if not raw or not raw.startswith(b"data:"):
            continue
        d = raw[5:].strip()
        if d == b"[DONE]":
            break
        try:
            yield json.loads(d)
        except json.JSONDecodeError:
            continue

def consume(path, body, resp):
    """Hash the output in delivery order; return (sha, out_tok, ttft_ms, tools)."""
    h = hashlib.sha256(); out_tok = None; ttft = None; tools = []
    t0 = time.time()
    def first():
        nonlocal ttft
        if ttft is None: ttft = (time.time() - t0) * 1000
    if body.get("stream"):
        if path.startswith("/v1/messages"):
            for ev in sse_events(resp):
                t = ev.get("type")
                if t == "content_block_start":
                    cb = ev.get("content_block", {})
                    if cb.get("type") == "tool_use":
                        first(); tools.append(cb.get("name", "")); h.update(("<tool_use:" + cb.get("name", "") + ">").encode())
                elif t == "content_block_delta":
                    d = ev.get("delta", {})
                    s = d.get("text") or d.get("thinking") or d.get("partial_json")
                    if s: first(); h.update(s.encode())
                elif t == "message_delta":
                    u = ev.get("usage") or {}
                    if "output_tokens" in u: out_tok = u["output_tokens"]
        else:  # OpenAI chat / completions
            for ev in sse_events(resp):
                if ev.get("usage"): out_tok = ev["usage"].get("completion_tokens", out_tok)
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {}) if "delta" in ch else {"content": ch.get("text")}
                    for k in ("reasoning_content", "reasoning", "content"):
                        if d.get(k): first(); h.update(d[k].encode())
                    for tc in d.get("tool_calls", []) or []:
                        fn = tc.get("function", {})
                        if fn.get("name"): first(); tools.append(fn["name"]); h.update(("<tool_use:" + fn["name"] + ">").encode())
                        if fn.get("arguments"): first(); h.update(fn["arguments"].encode())
    else:
        first()
        j = resp.json()
        if path.startswith("/v1/messages"):
            for b in j.get("content", []):
                if b.get("type") == "text": h.update(b["text"].encode())
                elif b.get("type") == "thinking": h.update(b.get("thinking", "").encode())
                elif b.get("type") == "tool_use":
                    tools.append(b.get("name", "")); h.update(("<tool_use:" + b.get("name", "") + ">").encode())
                    h.update(json.dumps(b.get("input"), sort_keys=True).encode())
            out_tok = (j.get("usage") or {}).get("output_tokens")
        else:
            for ch in j.get("choices", []):
                m = ch.get("message", {}) if "message" in ch else {"content": ch.get("text")}
                for k in ("reasoning_content", "reasoning", "content"):
                    if m.get(k): h.update(m[k].encode())
                for tc in m.get("tool_calls", []) or []:
                    fn = tc.get("function", {}); tools.append(fn.get("name", ""))
                    h.update(("<tool_use:" + fn.get("name", "") + ">" + fn.get("arguments", "")).encode())
            out_tok = (j.get("usage") or {}).get("completion_tokens")
    return h.hexdigest(), out_tok, ttft, tools

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log"); ap.add_argument("--base", default="http://172.17.0.1:8081")
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    n = 0; t_all = time.time()
    with open(a.out, "w") as out:
        for line in open(a.log):
            if not line.strip(): continue
            rec = json.loads(line)
            if rec["seq"] < a.start: continue
            body = json.loads(rec["body"])
            path = rec["path"]
            t0 = time.time()
            resp = requests.post(a.base + path, json=body, stream=bool(body.get("stream")), timeout=a.timeout)
            status = resp.status_code
            sha, out_tok, ttft, tools = consume(path, body, resp) if status == 200 else ("", None, None, [])
            wall = (time.time() - t0) * 1000
            row = dict(seq=rec["seq"], path=path, status=status, sha=sha, out_tok=out_tok,
                       ttft_ms=None if ttft is None else round(ttft, 1), wall_ms=round(wall, 1), tools=tools)
            out.write(json.dumps(row) + "\n"); out.flush()
            n += 1
            if not a.quiet:
                print(f"seq={rec['seq']:4d} {path:22s} {status} out={out_tok} ttft={row['ttft_ms']} wall={wall:.0f}ms sha={sha[:12]} tools={','.join(tools)[:40]}", flush=True)
            if a.limit and n >= a.limit: break
    print(f"replayed {n} requests in {time.time() - t_all:.1f} s -> {a.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
