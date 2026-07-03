#include "tokenizer.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <unordered_map>

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

struct Tokenizer::Impl {
    std::unordered_map<std::string, int> tok2id;
    std::unordered_map<std::string, int> merge_rank; // "left right" -> rank
    std::string b2u[256];
    std::unordered_map<std::string, uint8_t> u2b;
    std::vector<std::pair<std::string, int>> specials; // control tokens, longest first
};

static std::string read_lp(FILE* f) {
    uint16_t n;
    if (fread(&n, 2, 1, f) != 1) throw std::runtime_error("tok: truncated");
    std::string s(n, 0);
    if (n && fread(s.data(), 1, n, f) != n) throw std::runtime_error("tok: truncated");
    return s;
}

Tokenizer::Tokenizer(const std::string& path) : impl_(new Impl) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("tok: cannot open " + path);
    uint32_t magic, ver, n, bos, eos;
    fread(&magic, 4, 1, f); fread(&ver, 4, 1, f); fread(&n, 4, 1, f);
    fread(&bos, 4, 1, f); fread(&eos, 4, 1, f);
    if (magic != 0x54373251) throw std::runtime_error("tok: bad magic");
    bos_ = (int)bos; eos_ = (int)eos;
    tokens_.reserve(n);
    for (uint32_t i = 0; i < n; i++) tokens_.push_back(read_lp(f));
    types_.resize(n);
    fread(types_.data(), 1, n, f);
    uint32_t nm;
    fread(&nm, 4, 1, f);
    for (uint32_t i = 0; i < nm; i++) impl_->merge_rank.emplace(read_lp(f), (int)i);
    fclose(f);

    for (uint32_t i = 0; i < n; i++) impl_->tok2id.emplace(tokens_[i], (int)i);
    build_byte_maps(impl_->b2u, impl_->u2b);
    for (uint32_t i = 0; i < n; i++)
        if (types_[i] == 3) impl_->specials.push_back({tokens_[i], (int)i});
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
        // \s*[\r\n]+
        {
            size_t j = i;
            while (is_sp(j) && !is_nl(j)) j++;
            if (is_nl(j)) {
                size_t k = j;
                while (is_nl(k)) k++;
                out.push_back(t.substr(i, k - i));
                i = k;
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

int Tokenizer::token_id(const std::string& s) const {
    for (size_t i = 0; i < tokens_.size(); i++)
        if (tokens_[i] == s) return (int)i;
    return -1;
}

std::vector<int> Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages, bool think) const {
    std::string p;
    for (auto& [role, content] : messages)
        p += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
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
