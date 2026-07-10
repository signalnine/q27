#!/usr/bin/env python3
"""ladder_price.py -- re-price deep-ladder ceilings on measured width curves.

Simulates the gated round policy (P12/P14 semantics: cap = leading run of
draft margins >= theta up to the ceiling; early-exit drafting stops at the
first sub-theta margin; verify width W = cap+1 floored at 2; commit n =
leading run of draft==actual capped at cap, +1) over REAL measured chain
data (--burst-stats CSV: d1..d10 draft tokens + m1..m10 margins at every
position of a real committed stream), and prices each round with a MEASURED
verify-width cost curve plus the measured per-draft-step cost.

Usage: ladder_price.py chains.csv stream.seq prompt_len curve_name...
Curves are defined inline (measured, BUILDLOG 2026-07-09/10).
"""
import csv
import sys

DRAFT_MS = 0.81  # measured MTP step (Q27_PHASE_STATS, 07-09)
THETA = 0.5      # production Q27_PMIN

# measured verify wall (ms/round) by width W=2..9. W9 = extrapolated
# (+1 lane at each curve's local marginal) -- ceiling 8 needs W9 verify,
# which ALSO needs the p[8]-struct widening (not built; pricing only).
CURVES = {
    # 61K ctx (docs61k, phase-width runs)
    'fd2@61k':  {2: 17.9, 3: 19.8, 4: 22.6, 5: 24.5, 6: 26.2, 7: 28.8, 8: 32.8, 9: 36.8},
    'mma@61k':  {2: 17.8, 3: 19.8, 4: 18.1, 5: 19.2, 6: 19.4, 7: 20.7, 8: 23.3, 9: 24.6},
    # 26K ctx (docs/cctx-scale)
    'fd2@26k':  {2: 15.6, 3: 16.8, 4: 18.5, 5: 19.9, 6: 20.7, 7: 22.6, 8: 25.8, 9: 28.9},
    'mma@26k':  None,  # filled from the measured 26K run via --curve26 flag
}


def simulate(rows, actual, ceiling, curve):
    """Walk the stream; one gated round per iteration. Returns (tok, ms, rounds)."""
    pos = min(rows)  # first decode-region chain row
    n_tok = 0
    ms = 0.0
    rounds = 0
    last = max(r for r in rows)
    while pos in rows and pos < last:
        d, m = rows[pos]
        # early-exit draft: steps = leading margins >= theta (up to ceiling),
        # +1 for the step whose margin stopped the loop (still ran)
        cap = 0
        while cap < ceiling and cap < len(m) and m[cap] >= THETA:
            cap += 1
        steps = min(cap + 1, ceiling)
        W = max(2, cap + 1)
        # width-floor top-up: cap==0 runs draft step 1 anyway
        if cap == 0 and ceiling >= 2:
            steps = max(steps, 2)
        # acceptance: leading run of draft==actual within cap
        n = 0
        while n < cap and pos + n + 1 <= last and d[n] == actual.get(pos + n, None):
            n += 1
        n += 1  # pending/correction token always commits
        ms += curve[W] + steps * DRAFT_MS
        n_tok += n
        rounds += 1
        pos += n
    return n_tok, ms, rounds


def main():
    chains_csv, seq_path, plen_s = sys.argv[1], sys.argv[2], sys.argv[3]
    plen = int(plen_s)
    toks = [int(t) for t in open(seq_path).read().split()]
    actual = {i: toks[i + 1] for i in range(plen - 1, len(toks) - 1)}
    rows = {}
    with open(chains_csv) as f:
        for row in csv.DictReader(f):
            q = int(row['q'])
            d = [int(row[f'd{k}']) for k in range(1, 11)]
            m = [float(row[f'm{k}']) for k in range(1, 11)]
            rows[q] = (d, m)
    # chains are recorded at FREE positions q >= plen-1? burst rig records
    # prompt-phase too; keep only decode-region rows with a known actual
    rows = {q: v for q, v in rows.items() if q in actual}
    if len(sys.argv) > 4 and sys.argv[4].startswith('--curve26='):
        vals = [float(x) for x in sys.argv[4].split('=')[1].split(',')]
        CURVES['mma@26k'] = dict(zip(range(2, 2 + len(vals)), vals))
        # extrapolate W9 with the W7->W8 marginal
        c = CURVES['mma@26k']
        c[9] = c[8] + (c[8] - c[7])
    # width-12 P3: generic measured-curve flag, name:v2,v3,...,vNN (W from 2)
    for arg in sys.argv[4:]:
        if arg.startswith('--curve='):
            name, vals_s = arg.split('=', 1)[1].split(':', 1)
            vals = [float(x) for x in vals_s.split(',')]
            CURVES[name] = dict(zip(range(2, 2 + len(vals)), vals))
    print(f"chains: {len(rows)} decode positions from {chains_csv}")
    ceils = (5, 6, 7, 8, 9, 10)  # width-12 P3: price the deep ceilings
    print(f"{'curve':>12} | " + " | ".join(f"ceil {c}: t/s (tok/rnd)" for c in ceils))
    for cname, curve in CURVES.items():
        if curve is None:
            continue
        cells = []
        for ceil in ceils:
            if ceil + 1 not in curve:  # curve doesn't cover this width
                cells.append(f"{'--':>14}")
                continue
            tok, ms, rnd = simulate(rows, actual, ceil, curve)
            cells.append(f"{tok * 1000 / ms:7.1f} ({tok / rnd:4.2f})")
        print(f"{cname:>12} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
