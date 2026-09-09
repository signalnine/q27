#!/usr/bin/env python3
"""Compare two replays of the same recording (bench/replay/replay.py outputs).

    bench/replay/replay_diff.py A.jsonl B.jsonl

Per request: identical output (sha), output tokens, TTFT and wall for both
arms. The first request whose sha differs is where the arms' numerics (or
parser) diverged; everything after it is on a different trajectory only if
the client would have sent different follow-ups, which a replay does not
model -- so the count of identical requests and the position of the first
divergence are the two numbers to read. Totals compare wall and tokens.
"""
import json, sys

def load(p):
    return {r["seq"]: r for r in (json.loads(l) for l in open(p) if l.strip())}

a, b = load(sys.argv[1]), load(sys.argv[2])
seqs = sorted(set(a) & set(b))
same = 0; first_diff = None
print(f"{'seq':>4s} {'path':22s} {'same':4s} {'tokA':>5s} {'tokB':>5s} {'ttftA':>7s} {'ttftB':>7s} {'wallA':>7s} {'wallB':>7s} tools")
for s in seqs:
    ra, rb = a[s], b[s]
    eq = ra["sha"] == rb["sha"] and ra["status"] == rb["status"]
    same += eq
    if not eq and first_diff is None: first_diff = s
    print(f"{s:4d} {ra['path']:22s} {'yes' if eq else 'NO ':4s} {str(ra['out_tok']):>5s} {str(rb['out_tok']):>5s} "
          f"{str(ra['ttft_ms']):>7s} {str(rb['ttft_ms']):>7s} {ra['wall_ms']:7.0f} {rb['wall_ms']:7.0f} "
          f"{','.join(ra['tools'])[:30]}{'' if ra['tools'] == rb['tools'] else ' | ' + ','.join(rb['tools'])[:30]}")
wa = sum(a[s]["wall_ms"] for s in seqs); wb = sum(b[s]["wall_ms"] for s in seqs)
ta = sum(a[s]["out_tok"] or 0 for s in seqs); tb = sum(b[s]["out_tok"] or 0 for s in seqs)
print(f"\n{len(seqs)} requests: {same} identical outputs; first divergence at seq {first_diff}")
print(f"A: {ta} output tokens, wall {wa / 1000:.1f} s | B: {tb} tokens, wall {wb / 1000:.1f} s | B/A wall {wb / max(wa, 1):.3f}")
