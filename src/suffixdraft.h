// SuffixDraft -- zero-model echo drafter (Phase 0: tools/suffix_sim.py,
// 2026-07-09). Longest-suffix-match proposer over the committed stream
// (prompt + emitted tokens): when the current suffix recurs earlier, the
// tokens that followed the earlier occurrence become the round's draft
// lanes -- no MTP passes, CPU-side, pennies per round. The verify half
// guarantees emitted tokens are greedy-identical regardless of draft
// content, so this trades only ROUND COUNT, never correctness.
//
// Phase-0 numbers (real streams, K=16): cctx fire@L>=12 35% of positions
// at AL 11.5; docs fire@L>=12 = 0% -- the gate goes silent on neutral
// traffic. Lag-0 self-match is excluded (proposing the literal future);
// lag>=1 overlap is kept -- with leading-match acceptance the direct
// continuation equals the lag-copy walk by induction.
//
// Host-side only, no CUDA. Unit tests: tools/test_suffixdraft.cpp.
#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace q27 {

class SuffixDraft {
    static constexpr int NG = 4;       // index key: 4-gram
    static constexpr int MAXCAND = 64; // most recent occurrences kept per key
    static constexpr int MAXEXT = 256; // backward match-extension cap
    std::vector<int> t;                // committed stream (prompt + emitted)
    std::unordered_map<uint64_t, std::vector<int>> idx; // key4 -> ascending positions

    static uint64_t key4(const int* p) {
        uint64_t h = 1469598103934665603ULL;
        for (int i = 0; i < NG; i++) { h ^= (uint32_t)p[i]; h *= 1099511628211ULL; }
        return h;
    }
    void index_pos(int i) { // i = position of the 4-gram's LAST token
        auto& v = idx[key4(&t[i - NG + 1])];
        v.push_back(i);
        if ((int)v.size() > MAXCAND) v.erase(v.begin());
    }

public:
    void reset(const std::vector<int>& prompt) {
        t = prompt;
        idx.clear();
        for (int i = NG - 1; i < (int)t.size(); i++) index_pos(i);
    }
    void append(int tok) {
        t.push_back(tok);
        if ((int)t.size() >= NG) index_pos((int)t.size() - 1);
    }
    size_t size() const { return t.size(); }

    // Longest suffix match of (stream + [pending]) against the stream, and
    // the k-token continuation that followed the match. pending rides along
    // VIRTUALLY: it is the round's d_token (spec3 outcome[9], not yet in em),
    // and proposals fill d_draft1.. which continue after it. Returns the
    // match length (0 = no match); out[0..k) valid only when > 0. Ties break
    // to the most recent occurrence. Proposals that would read past the
    // stream end continue from the proposal buffer itself (lag-copy periodic
    // extension -- computable from history, no future reference).
    int propose_with(int pending, int k, int* out) const {
        const int n = (int)t.size();
        if (n < NG - 1) return 0;
        const int N = n + 1; // conceptual stream length incl pending
        auto s = [&](int i) { return i < n ? t[i] : pending; };
        int key[NG];
        for (int j = 0; j < NG; j++) key[j] = s(N - NG + j);
        auto it = idx.find(key4(key));
        if (it == idx.end()) return 0;
        int best_m = 0, best_p = -1;
        for (int p : it->second) {
            if (p >= N - 1) continue; // lag-0 self-match: reading the future
            bool ok = true;           // hash-collision guard: verify the key tokens
            for (int j = 0; j < NG; j++)
                if (t[p - j] != s(N - 1 - j)) { ok = false; break; }
            if (!ok) continue;
            int m = NG;
            while (m < MAXEXT && p - m >= 0 && N - 1 - m >= 0 && t[p - m] == s(N - 1 - m)) m++;
            if (m >= best_m) { best_m = m; best_p = p; } // >= : most recent wins ties
        }
        if (best_p < 0) return 0;
        for (int i = 0; i < k; i++) {
            int src = best_p + 1 + i;
            // src >= N would reference a future position: its predicted value
            // is the proposal already made for it (periodic extension). src in
            // [n, N) is the pending itself.
            out[i] = src < N ? s(src) : out[src - N];
        }
        return best_m;
    }
};

} // namespace q27
