#!/usr/bin/env python3
"""Needle-in-haystack retrieval check for long-context serving.

Plants six distinct "calibration constant" needles at 10/35/60/70/78/95%
depth in a filler haystack, then asks the server to read each back
verbatim. PASS = exact number at every depth. This is the quality check
for a big --ctx: if turbo3 KV (or any config) degraded retrieval at
depth, it shows up here first.

Server (separate terminal):
  Q27_KV=turbo3 ./build/q27-server-w8 <model> <tok> --port 8081 --ctx <N>
Then: python3 tools/needle_check.py [--port 8081] [--depth CHARS]
  --depth is characters; ~3 chars/token (700000 ~= a 233K-token prompt).
  Size it under your --ctx. First ask pays the full cold prefill;
  the rest ride the prefix cache.
"""
import argparse
import json
import sys
import urllib.request

FILLER = (
    "The maintenance log records routine telemetry from the orbital relay. "
    "Sensor arrays report nominal drift within expected tolerances. "
    "Ground control confirms handshake integrity across the downlink. "
    "Thermal cycling remains inside the qualified operating envelope. "
)
NEEDLES = [
    ("tidal array", "88231"),
    ("phase regulator", "40917"),
    ("cryo manifold", "72608"),
    ("beacon lattice", "15540"),
    ("dampener coil", "63182"),
    ("aperture gate", "29475"),
]


def build_prompt(depth_chars):
    # plant needles at fractional depths; pad with filler to ~depth_chars
    fracs = [0.10, 0.35, 0.60, 0.70, 0.78, 0.95]
    slots = sorted(zip(fracs, NEEDLES))
    out = []
    cur = 0
    for frac, (name, val) in slots:
        target = int(depth_chars * frac)
        while cur < target:
            out.append(FILLER)
            cur += len(FILLER)
        s = f"\nIMPORTANT FACT: the calibration constant for the {name} is {val}.\n"
        out.append(s)
        cur += len(s)
    while cur < depth_chars:
        out.append(FILLER)
        cur += len(FILLER)
    return "".join(out)


def ask(port, haystack, name):
    q = (haystack + f"\n\nBased ONLY on the text above, what is the calibration "
         f"constant for the {name}? Answer with just the number.")
    body = json.dumps({
        "model": "q27", "max_tokens": 24, "temperature": 0,
        "messages": [{"role": "user", "content": q}],
    }).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/messages", data=body,
                                 headers={"Content-Type": "application/json",
                                          "x-api-key": "x"})
    r = json.load(urllib.request.urlopen(req, timeout=2400))
    return "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--depth", type=int, default=200000, help="approx prompt chars")
    args = ap.parse_args()
    hay = build_prompt(args.depth)
    print(f"haystack ~{len(hay)} chars (~{len(hay)//3} tok)", file=sys.stderr)
    passed = 0
    for name, val in NEEDLES:
        ans = ask(args.port, hay, name)
        ok = val in ans
        passed += ok
        print(f"  {name:16s} want {val}: {'PASS' if ok else 'FAIL'}  got {ans.strip()[:40]!r}")
    print(f"\nNEEDLE {passed}/{len(NEEDLES)} -> {'PASS' if passed == len(NEEDLES) else 'FAIL'}")
    return 0 if passed == len(NEEDLES) else 1


if __name__ == "__main__":
    sys.exit(main())
