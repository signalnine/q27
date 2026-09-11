#include "tokenizer.h"

#ifdef Q27_TOKENIZER_TESTING
#include <atomic>
#endif

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <unordered_map>
#include <memory>

namespace q27 {

// ---- GPT-2 byte <-> unicode-char mapping ----
// bytes 0x21..0x7E, 0xA1..0xAC, 0xAE..0xFF map to themselves (as codepoints);
// the rest map to 256+k in order.
static void build_byte_maps(std::string b2u[256], std::unordered_map<std::string, uint8_t>& u2b) {
    auto cp_to_utf8 = [](int cp) {
        std::string s;
        if (cp < 0x80) s += (char)cp;
        else if (cp < 0x800) {
            s += (char)(0xC0 | (cp >> 6));
            s += (char)(0x80 | (cp & 0x3F));
        } else {
            s += (char)(0xE0 | (cp >> 12));
            s += (char)(0x80 | ((cp >> 6) & 0x3F));
            s += (char)(0x80 | (cp & 0x3F));
        }
        return s;
    };
    int k = 0;
    for (int b = 0; b < 256; b++) {
        bool direct = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
        int cp = direct ? b : 256 + k++;
        b2u[b] = cp_to_utf8(cp);
        u2b[b2u[b]] = (uint8_t)b;
    }
}

#ifdef Q27_TOKENIZER_TESTING
static std::atomic<int> tokenizer_live_impls{0};
#endif

struct Tokenizer::Impl {
#ifdef Q27_TOKENIZER_TESTING
    Impl() { tokenizer_live_impls.fetch_add(1, std::memory_order_relaxed); }
    ~Impl() { tokenizer_live_impls.fetch_sub(1, std::memory_order_relaxed); }
#endif
    std::unordered_map<std::string, int> tok2id;
    std::unordered_map<std::string, int> merge_rank; // "left right" -> rank
    std::string b2u[256];
    std::unordered_map<std::string, uint8_t> u2b;
    std::vector<std::pair<std::string, int>> specials; // control tokens, longest first
};

#ifdef Q27_TOKENIZER_TESTING
int tokenizer_live_impls_for_test() {
    return tokenizer_live_impls.load(std::memory_order_relaxed);
}
#endif

struct FileCloser {
    void operator()(FILE* f) const { if (f) fclose(f); }
};

template <typename T>
static void read_exact(FILE* f, T* out, size_t count) {
    if (count && fread(out, sizeof(T), count, f) != count)
        throw std::runtime_error("tok: truncated");
}

static std::string read_lp(FILE* f) {
    uint16_t n = 0;
    read_exact(f, &n, 1);
    std::string s(n, 0);
    read_exact(f, s.data(), n);
    return s;
}

Tokenizer::Tokenizer(const std::string& path) : impl_(std::make_unique<Impl>()) {
    std::unique_ptr<FILE, FileCloser> f(fopen(path.c_str(), "rb"));
    if (!f) throw std::runtime_error("tok: cannot open " + path);
    uint32_t magic = 0, ver = 0, n = 0, bos = 0, eos = 0;
    read_exact(f.get(), &magic, 1);
    read_exact(f.get(), &ver, 1);
    read_exact(f.get(), &n, 1);
    read_exact(f.get(), &bos, 1);
    read_exact(f.get(), &eos, 1);
    if (magic != 0x54373251) throw std::runtime_error("tok: bad magic");
    bos_ = (int)bos; eos_ = (int)eos;
    tokens_.reserve(n);
    for (uint32_t i = 0; i < n; i++) tokens_.push_back(read_lp(f.get()));
    types_.resize(n);
    read_exact(f.get(), types_.data(), n);
    uint32_t nm = 0;
    read_exact(f.get(), &nm, 1);
    for (uint32_t i = 0; i < nm; i++) impl_->merge_rank.emplace(read_lp(f.get()), (int)i);

    for (uint32_t i = 0; i < n; i++) impl_->tok2id.emplace(tokens_[i], (int)i);
    build_byte_maps(impl_->b2u, impl_->u2b);
    // Added tokens match in text as whole tokens, like HF's added-token split
    // and llama.cpp: type 3 (CONTROL) and type 4 (USER_DEFINED). Until
    // 2026-09-10 only CONTROL plus a hardcoded <think>/</think> matched, so
    // the other USER_DEFINED tokens of the Qwen3.6/3.8 vocab -- <tool_call>,
    // </tool_call>, <tool_response>, </tool_response> -- encoded as plain
    // text ("<", "tool", "_call", ">") in every prompt: the template's
    // tool-format instructions, every past tool call, every tool result. The
    // checkpoint was trained on (and itself emits) the single added tokens.
    for (uint32_t i = 0; i < n; i++)
        if (types_[i] == 3 || types_[i] == 4) impl_->specials.push_back({tokens_[i], (int)i});
    // <think>/</think> by name as well, for a vocab that types them NORMAL
    // (BPE merges cannot form them; think-block prefills depend on this)
    for (const char* s : {"<think>", "</think>"}) {
        auto it = impl_->tok2id.find(s);
        if (it != impl_->tok2id.end() && types_[it->second] != 3 && types_[it->second] != 4)
            impl_->specials.push_back({s, it->second});
    }
    // longest-first for greedy matching
    std::sort(impl_->specials.begin(), impl_->specials.end(),
              [](auto& a, auto& b) { return a.first.size() > b.first.size(); });
}

// split a UTF-8 string into unicode chars (as utf8 substrings)
static std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        int len = 1;
        uint8_t c = s[i];
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

std::vector<int> Tokenizer::bpe_word(const std::string& word) const {
    // Bounded-word guard: the merge loop below is O(word^2) (full pair rescan +
    // erase-in-loop per merge). A pathological no-whitespace blob (minified JS,
    // base64) collapses to one huge "word" and stalls tokenization single-threaded
    // before any context check. Cap the word and BPE in WORD_CAP-byte chunks so cost
    // is O(n * WORD_CAP), not O(n^2). Only over-cap words (already-degenerate input)
    // get different token boundaries; normal whitespace-delimited text is far under
    // the cap and byte-identical (the canonical is unaffected).
    static constexpr size_t WORD_CAP = 1024;
    if (word.size() > WORD_CAP) {
        std::vector<int> out;
        for (size_t off = 0; off < word.size(); off += WORD_CAP) {
            auto chunk = bpe_word(word.substr(off, WORD_CAP));
            out.insert(out.end(), chunk.begin(), chunk.end());
        }
        return out;
    }
    // word is raw bytes; map to byte-encoded char strings
    std::vector<std::string> parts;
    parts.reserve(word.size());
    for (unsigned char c : word) parts.push_back(impl_->b2u[c]);

    while (parts.size() > 1) {
        int best = INT32_MAX, bi = -1;
        for (size_t i = 0; i + 1 < parts.size(); i++) {
            auto it = impl_->merge_rank.find(parts[i] + " " + parts[i + 1]);
            if (it != impl_->merge_rank.end() && it->second < best) {
                best = it->second;
                bi = (int)i;
            }
        }
        if (bi < 0) break;
        parts[bi] += parts[bi + 1];
        parts.erase(parts.begin() + bi + 1);
    }
    std::vector<int> out;
    for (auto& p : parts) {
        auto it = impl_->tok2id.find(p);
        if (it != impl_->tok2id.end()) out.push_back(it->second);
        else // byte fallback: emit each char separately (should be in vocab)
            for (auto& ch : utf8_chars(p)) {
                auto it2 = impl_->tok2id.find(ch);
                if (it2 != impl_->tok2id.end()) out.push_back(it2->second);
            }
    }
    return out;
}

// qwen35 pretokenizer approximation:
//  (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n L N]?[L M]+ | N |
//  ?[^\s L M N]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// with L = letter (ASCII alpha or any byte >= 0x80), N = single digit.
std::vector<std::string> Tokenizer::pretokenize(const std::string& t) const {
    std::vector<std::string> out;
    size_t i = 0, n = t.size();
    auto is_l = [&](size_t j) {
        return j < n && (isalpha((unsigned char)t[j]) || (unsigned char)t[j] >= 0x80);
    };
    auto is_d = [&](size_t j) { return j < n && isdigit((unsigned char)t[j]); };
    auto is_sp = [&](size_t j) { return j < n && isspace((unsigned char)t[j]); };
    auto is_nl = [&](size_t j) { return j < n && (t[j] == '\r' || t[j] == '\n'); };

    while (i < n) {
        // contractions (case-insensitive)
        if (t[i] == '\'' && i + 1 < n) {
            char c1 = tolower(t[i + 1]);
            char c2 = i + 2 < n ? tolower(t[i + 2]) : 0;
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                out.push_back(t.substr(i, 2)); i += 2; continue;
            }
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                out.push_back(t.substr(i, 3)); i += 3; continue;
            }
        }
        // [^\r\n L N]? [L M]+   (optional leading non-letter joins a letter run)
        {
            size_t j = i;
            bool lead = false;
            if (!is_nl(j) && !is_l(j) && !is_d(j) && j < n && is_l(j + 1)) { lead = true; j++; }
            if (is_l(j)) {
                size_t k = j;
                while (is_l(k)) k++;
                out.push_back(t.substr(lead ? i : j, k - (lead ? i : j)));
                i = k;
                continue;
            }
        }
        // single digit
        if (is_d(i)) { out.push_back(t.substr(i, 1)); i++; continue; }
        // " ?[^\s L M N]+[\r\n]*"  (punct run, optional leading space)
        {
            size_t j = i;
            if (t[j] == ' ' && j + 1 < n && !is_sp(j + 1) && !is_l(j + 1) && !is_d(j + 1)) j++;
            if (j < n && !is_sp(j) && !is_l(j) && !is_d(j)) {
                size_t k = j;
                while (k < n && !is_sp(k) && !is_l(k) && !is_d(k)) k++;
                while (is_nl(k)) k++;
                out.push_back(t.substr(i, k - i));
                i = k;
                continue;
            }
        }
        // \s*[\r\n]+ : \s* is greedy over ALL whitespace, newlines included,
        // and backtracks only far enough for [\r\n]+ -- the match is the
        // whitespace run up to and including its LAST \r/\n. (Until 2026-09-10
        // \s* stopped at the first newline, so "\n \n" -- a blank line holding
        // a space, common in diffs and tool output -- split as "\n" + " \n"
        // where HF has the single token "\n \n".)
        {
            size_t k = i;
            while (is_sp(k)) k++;
            size_t last = std::string::npos;
            for (size_t p = i; p < k; p++)
                if (is_nl(p)) last = p;
            if (last != std::string::npos) {
                out.push_back(t.substr(i, last + 1 - i));
                i = last + 1;
                continue;
            }
        }
        // \s+(?!\S) | \s+
        if (is_sp(i)) {
            size_t k = i;
            while (is_sp(k)) k++;
            // \s+(?!\S): trailing run keeps all; else leave last space for next token
            if (k < n && k - i > 1) k--;
            out.push_back(t.substr(i, k - i));
            i = k;
            continue;
        }
        out.push_back(t.substr(i, 1)); // fallback single byte
        i++;
    }
    return out;
}


Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    size_t i = 0;
    while (i < text.size()) {
        // greedy special-token match
        bool matched = false;
        for (auto& [s, id] : impl_->specials) {
            if (text.compare(i, s.size(), s) == 0) {
                out.push_back(id);
                i += s.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;
        // find next special occurrence; encode the plain span before it
        size_t next = std::string::npos;
        for (auto& [s, id] : impl_->specials) {
            size_t p = text.find(s, i);
            if (p != std::string::npos && p < next) next = p;
        }
        size_t end = next == std::string::npos ? text.size() : next;
        std::string span = text.substr(i, end - i);
        for (auto& w : pretokenize(span)) {
            auto ids = bpe_word(w);
            out.insert(out.end(), ids.begin(), ids.end());
        }
        i = end;
    }
    return out;
}

std::string Tokenizer::decode_one(int id) const {
    if (id < 0 || id >= (int)tokens_.size()) return "";
    if (types_[id] == 3) return ""; // control tokens invisible in output
    std::string out;
    for (auto& ch : utf8_chars(tokens_[id])) {
        auto it = impl_->u2b.find(ch);
        if (it != impl_->u2b.end()) out += (char)it->second;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) out += decode_one(id);
    return out;
}

std::vector<std::string> Tokenizer::vocab_bytes() const {
    std::vector<std::string> out(tokens_.size());
    for (size_t i = 0; i < tokens_.size(); i++)
        if (types_[i] != 3) out[i] = decode_one((int)i);
    return out;
}

int Tokenizer::token_id(const std::string& s) const {
    for (size_t i = 0; i < tokens_.size(); i++)
        if (tokens_[i] == s) return (int)i;
    return -1;
}

// Strip ChatML role delimiters from untrusted roles/content so they can't
// forge prompt structure (Security #7 -- same policy as api_common.h
// strip_ctrl, applied here so EVERY template caller is covered; review
// 2026-07-09 P1 #5: the OpenAI chat path reached this function with raw
// client strings and bypassed the Anthropic path's sanitizer).
static std::string strip_chatml(std::string s) {
    for (const std::string& m : {std::string("<|im_start|>"), std::string("<|im_end|>")})
        for (size_t p; (p = s.find(m)) != std::string::npos;) s.erase(p, m.size());
    return s;
}

std::vector<int> Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages, bool think) const {
    std::string p;
    for (auto& [role, content] : messages)
        p += "<|im_start|>" + strip_chatml(role) + "\n" + strip_chatml(content) + "<|im_end|>\n";
    p += "<|im_start|>assistant\n";
    std::vector<int> ids = encode(p);
    if (!think) {
        // <think>/</think> are added tokens BPE cannot form from text --
        // append their ids directly (string fallback if a future vocab
        // lacks them)
        int t1 = token_id("<think>"), t2 = token_id("</think>");
        std::vector<int> nn = encode("\n\n");
        if (t1 >= 0 && t2 >= 0) {
            ids.push_back(t1);
            ids.insert(ids.end(), nn.begin(), nn.end());
            ids.push_back(t2);
            ids.insert(ids.end(), nn.begin(), nn.end());
        } else {
            std::vector<int> s = encode("<think>\n\n</think>\n\n");
            ids.insert(ids.end(), s.begin(), s.end());
        }
    }
    return ids;
}

} // namespace q27
