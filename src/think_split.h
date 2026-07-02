// Streaming splitter for Qwopus <think>...</think> reasoning markers.
// Routes a token-by-token decoded text stream into THINK vs TEXT channels,
// holding back any tail that could be a partial marker so markers split across
// token boundaries are still detected.
#pragma once
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace q27 {

struct ThinkSplitter {
    enum Chan { TEXT, THINK };
    Chan chan = TEXT;  // Qwopus opens with <think>, flipping us to THINK immediately
    std::string hold;
    const std::string OPEN = "<think>";
    const std::string CLOSE = "</think>";

    std::vector<std::pair<Chan, std::string>> feed(const std::string& piece) {
        hold += piece;
        std::vector<std::pair<Chan, std::string>> out;
        for (;;) {
            const std::string& marker = chan == TEXT ? OPEN : CLOSE;
            size_t p = hold.find(marker);
            if (p != std::string::npos) {
                if (p > 0) out.push_back({chan, hold.substr(0, p)});
                hold.erase(0, p + marker.size());
                chan = chan == TEXT ? THINK : TEXT;
                continue;
            }
            // no full marker: emit all but the longest suffix that is a prefix of marker
            size_t keep = 0, maxk = std::min(hold.size(), marker.size() - 1);
            for (size_t k = maxk; k > 0; k--)
                if (hold.compare(hold.size() - k, k, marker, 0, k) == 0) { keep = k; break; }
            if (hold.size() > keep) {
                out.push_back({chan, hold.substr(0, hold.size() - keep)});
                hold.erase(0, hold.size() - keep);
            }
            break;
        }
        return out;
    }

    std::vector<std::pair<Chan, std::string>> flush() {
        std::vector<std::pair<Chan, std::string>> out;
        if (!hold.empty()) { out.push_back({chan, hold}); hold.clear(); }
        return out;
    }
};

inline std::string strip_ws(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

} // namespace q27
