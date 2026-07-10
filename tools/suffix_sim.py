#!/usr/bin/env python3
"""suffix_sim.py -- Phase-0 gate for a suffix-tree/echo drafter (survey
2026-07-09: sonar/Arctic suffix decoding, rtp-llm advice-prompt). Offline
acceptance simulation on a real committed token stream; zero engine risk.

At each decode position i: find the longest suffix of stream[:i] that
recurs earlier in the stream (prompt + generated so far), propose the K
tokens that followed that earlier occurrence, score against what the
engine actually committed. Fire rule: propose only when match_len >= L.

Usage: suffix_sim.py stream.seq prompt_len [K]
"""
import sys
from collections import defaultdict

def main():
    toks = [int(t) for t in open(sys.argv[1]).read().split()]
    plen = int(sys.argv[2])
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    n = len(toks)
    NG = 4  # index key size
    idx = defaultdict(list)  # 4-gram -> positions of its LAST token
    for i in range(NG - 1, plen):
        idx[tuple(toks[i - NG + 1:i + 1])].append(i)

    results = []  # (match_len, al)
    for i in range(plen, n):
        key = tuple(toks[i - NG:i])
        best_m, best_end = 0, -1
        for p in idx.get(key, []):
            # p == i-1 is the suffix matching ITSELF (lag 0): proposing the
            # literal future, always "correct" -- exclude. Lag >= 1 overlap
            # (periodic extension) is a real proposer capability: leading-
            # match scoring makes direct future-comparison equivalent to the
            # lag-copy walk by induction.
            if p >= i - 1:
                continue
            # extend match backwards from p
            m = NG
            while m < 256 and p - m >= 0 and i - m - 1 >= 0 and toks[p - m] == toks[i - m - 1]:
                m += 1
            # ties -> most recent occurrence (list is position-ordered)
            if m >= best_m:
                best_m, best_end = m, p
        al = 0
        if best_end >= 0:
            while (al < K and best_end + 1 + al < n and i + al < n
                   and toks[best_end + 1 + al] == toks[i + al]):
                al += 1
        results.append((best_m, al))
        # grow index with the newly committed position
        if i >= NG - 1:
            idx[tuple(toks[i - NG + 1:i + 1])].append(i)

    N = len(results)
    print(f"stream {sys.argv[1]}: {N} decode positions, K={K}, index 4-gram")
    print(f"{'fire L>=':>9} {'fired%':>7} {'AL|fired':>9} {'AL>=8%':>7} {'AL>=16%':>8} "
          f"{'tok/pos overall':>16}")
    for L in (4, 6, 8, 12, 16, 24, 32):
        fired = [(m, al) for m, al in results if m >= L]
        if not fired:
            print(f"{L:>9} {0:>7.1f}")
            continue
        fr = 100.0 * len(fired) / N
        mal = sum(al for _, al in fired) / len(fired)
        a8 = 100.0 * sum(1 for _, al in fired if al >= 8) / len(fired)
        a16 = 100.0 * sum(1 for _, al in fired if al >= K) / len(fired)
        # tokens/position if we drafted ONLY when fired (else 0 bonus)
        tot = sum(al for _, al in fired) / N
        print(f"{L:>9} {fr:>7.1f} {mal:>9.2f} {a8:>7.1f} {a16:>8.1f} {tot:>16.2f}")
    # distribution of match lengths at decode positions
    from collections import Counter
    c = Counter(min(m, 40) for m, _ in results)
    print("match-len histogram (capped 40):",
          sorted(c.items())[:12], "...tail", sum(v for k, v in c.items() if k >= 24))

if __name__ == "__main__":
    main()
