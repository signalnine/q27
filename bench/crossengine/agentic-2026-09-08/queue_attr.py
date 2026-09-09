#!/usr/bin/env python3
"""Queue-wait attribution from q27 [req] lines (item 2 of the (p) agenda).
Each [req] carries qw_ms (queue wait), pf_ms, dec_ms, cb_ms and t (ms since
server start at completion). Reconstruct each request's service interval
[t - pf_ms - dec_ms - cb_ms, t], its arrival = start - qw_ms, and attribute
every queued request to what was being served when it arrived.
usage: queue_attr.py <req.txt> [--top N]"""
import re, sys
from collections import Counter, defaultdict
path = sys.argv[1]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 12
def kv(line):
    return {k: v for k, v in re.findall(r"(\w+)=([^\s]+)", line)}
reqs = []
for l in open(path):
    if not l.startswith("[req]"):
        continue
    r = kv(l)
    qw, pf, dec, cb, t = (float(r.get(k, 0)) for k in ("qw_ms", "pf_ms", "dec_ms", "cb_ms", "t"))
    svc = pf + dec + cb
    reqs.append(dict(rid=int(r["rid"]), api=r.get("api"), conv=r.get("conv"), prompt=int(r["prompt"]),
                     hit=int(r.get("hit", 0)), pf=int(r.get("pf", 0)), pf_ms=pf, dec=int(r.get("dec", 0)),
                     dec_ms=dec, qw=qw, svc=svc, end=t, start=t - svc, arrive=t - svc - qw,
                     slot=r.get("slot"), yields=int(r.get("yields", 0))))
reqs.sort(key=lambda r: r["rid"])
n = len(reqs); span = (max(r["end"] for r in reqs) - min(r["arrive"] for r in reqs)) / 1000
busy = sum(r["svc"] for r in reqs) / 1000
qtot = sum(r["qw"] for r in reqs) / 1000
queued = [r for r in reqs if r["qw"] > 0]
print(f"{n} requests over {span:.0f} s; service {busy:.0f} s ({busy / span * 100:.0f}% of span); "
      f"queue wait total {qtot:.1f} s ({qtot / (busy + qtot) * 100:.1f}% of service+queue); "
      f"{len(queued)} requests queued (>0 ms), {sum(1 for r in reqs if r['qw'] >= 1000)} >= 1 s, max {max(r['qw'] for r in reqs):.0f} ms")
print("slots:", dict(Counter(r["slot"] for r in reqs)), " yields>0:", sum(1 for r in reqs if r["yields"] > 0))
def cls(r):
    if r["pf"] >= 8000: return "cold-long-prefill"
    if r["pf"] >= 1000: return "prefill-1k-8k"
    return "small-turn"
# what was in service at each queued request's arrival
attr = defaultdict(lambda: [0, 0.0])   # blocker class -> [count, queue ms]
pair = defaultdict(lambda: [0, 0.0])
conc = Counter()
for q in queued:
    blockers = [r for r in reqs if r is not q and r["start"] <= q["arrive"] < r["end"]]
    inq = [r for r in reqs if r is not q and r["arrive"] <= q["arrive"] and r["start"] > q["arrive"]]  # ahead in the queue
    conc[len(blockers) + len(inq)] += 1
    key = ",".join(sorted(cls(b) for b in blockers)) or ("queue-only:" + str(len(inq)))
    attr[key][0] += 1; attr[key][1] += q["qw"]
    pair[(cls(q), key)][0] += 1; pair[(cls(q), key)][1] += q["qw"]
print("\nqueued requests by what was in service at arrival (blocker class -> n, queue s, mean ms):")
for k, (c, ms) in sorted(attr.items(), key=lambda x: -x[1][1]):
    print(f"  {k:40s} {c:4d} {ms / 1000:7.1f} s  mean {ms / c:6.0f} ms")
print("\nby (queued request class, blocker):")
for (qc, k), (c, ms) in sorted(pair.items(), key=lambda x: -x[1][1]):
    print(f"  {qc:18s} behind {k:40s} {c:4d} {ms / 1000:7.1f} s")
print("\nrequests ahead (in service + queued) at arrival:", dict(sorted(conc.items())))
print(f"\ntop {top} queue waits:")
for q in sorted(queued, key=lambda r: -r["qw"])[:top]:
    blockers = [r for r in reqs if r is not q and r["start"] <= q["arrive"] < r["end"]]
    print(f"  rid={q['rid']:4d} qw={q['qw']:5.0f} ms  {cls(q):18s} prompt={q['prompt']:6d} pf={q['pf']:6d} | in service: "
          + "; ".join(f"rid={b['rid']} {cls(b)} pf={b['pf']} pf_ms={b['pf_ms']:.0f} dec_ms={b['dec_ms']:.0f}" for b in blockers))
# service-time composition: how much of the busy time is prefill vs decode
pf_s = sum(r["pf_ms"] for r in reqs) / 1000; dec_s = sum(r["dec_ms"] for r in reqs) / 1000
print(f"\nservice composition: prefill {pf_s:.0f} s, decode {dec_s:.0f} s; "
      f"cold-long-prefill requests {sum(1 for r in reqs if cls(r) == 'cold-long-prefill')} = {sum(r['pf_ms'] for r in reqs if cls(r) == 'cold-long-prefill') / 1000:.0f} s of prefill")
