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
#include <array>

#include "unicode_tables.h"

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

// ---- Unicode: code points, pre-tokenizer classes, NFC ----
// The Qwen3.6/3.8 tokenizer.json runs NFC, then this Split regex, then byte-level
// BPE:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|
//    ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Until 2026-09-10 q27 approximated it on BYTES (every byte >= 0x80 a letter,
// ASCII-only digits and whitespace) and skipped NFC, so "don’t", "a—b", "٣",
// "a b" or a decomposed "é" split differently from HF. The classes and NFC
// data now come from the reference itself (tools/gen_unicode_tables.py probes
// the tokenizers library; src/unicode_tables.h), and the matcher below walks
// code points alternative by alternative in the regex's order.
namespace {
enum : uint8_t { C_O = 0, C_LM = 1, C_N = 2, C_S = 3, C_NL = 4 };

template <class R, size_t N>
bool in_ranges(const R (&r)[N], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < r[mid].lo) hi = mid;
        else if (cp > r[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

uint8_t cp_class(uint32_t cp) {
    if (cp == '\n' || cp == '\r') return C_NL;
    if (cp < 0x80) {
        static const auto ascii = [] {
            std::array<uint8_t, 128> a{};
            for (uint32_t c = 0; c < 128; c++)
                a[c] = in_ranges(unicode::kLetterMark, c) ? C_LM
                     : in_ranges(unicode::kNumber, c)     ? C_N
                     : in_ranges(unicode::kSpace, c)      ? C_S : C_O;
            a['\n'] = a['\r'] = C_NL;
            return a;
        }();
        return ascii[cp];
    }
    if (in_ranges(unicode::kLetterMark, cp)) return C_LM;
    if (in_ranges(unicode::kNumber, cp)) return C_N;
    if (in_ranges(unicode::kSpace, cp)) return C_S;
    return C_O;
}

// Decode one scalar at s[i]; *len = bytes consumed. Invalid or truncated UTF-8
// yields one byte as its own "code point" (cp >= 0x110000 marks it) so every
// input byte still reaches BPE, as before.
uint32_t decode_cp(const std::string& s, size_t i, int* len) {
    const auto b = [&](size_t k) { return (uint32_t)(uint8_t)s[k]; };
    const uint32_t c = b(i);
    int n = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
    if (n == 1) { *len = 1; return c; }
    if (n == 0 || i + n > s.size()) { *len = 1; return 0x110000 + c; }
    uint32_t cp = c & (0x7F >> n);
    for (int k = 1; k < n; k++) {
        if ((b(i + k) & 0xC0) != 0x80) { *len = 1; return 0x110000 + c; }
        cp = (cp << 6) | (b(i + k) & 0x3F);
    }
    static const uint32_t kMin[5] = {0, 0, 0x80, 0x800, 0x10000};
    if (cp < kMin[n] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) { *len = 1; return 0x110000 + c; }
    *len = n;
    return cp;
}

void append_utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += (char)cp;
    else if (cp < 0x800) { o += (char)(0xC0 | cp >> 6); o += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        o += (char)(0xE0 | cp >> 12); o += (char)(0x80 | ((cp >> 6) & 0x3F));
        o += (char)(0x80 | (cp & 0x3F));
    } else {
        o += (char)(0xF0 | cp >> 18); o += (char)(0x80 | ((cp >> 12) & 0x3F));
        o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F));
    }
}

uint8_t ccc_of(uint32_t cp) {
    if (cp < 0x300) return 0;
    const auto& r = unicode::kCcc;
    size_t lo = 0, hi = sizeof(r) / sizeof(r[0]);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < r[mid].lo) hi = mid;
        else if (cp > r[mid].hi) lo = mid + 1;
        else return r[mid].ccc;
    }
    return 0;
}

constexpr uint32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = 588, kSCount = 11172;

void decompose_cp(uint32_t cp, std::vector<uint32_t>& out) {
    if (cp >= kSBase && cp < kSBase + kSCount) {
        const uint32_t s = cp - kSBase;
        out.push_back(kLBase + s / kNCount);
        out.push_back(kVBase + (s % kNCount) / kTCount);
        if (s % kTCount) out.push_back(kTBase + s % kTCount);
        return;
    }
    const auto& d = unicode::kDecomp;
    size_t lo = 0, hi = sizeof(d) / sizeof(d[0]);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < d[mid].cp) hi = mid;
        else if (cp > d[mid].cp) lo = mid + 1;
        else {
            out.insert(out.end(), unicode::kDecompFlat + d[mid].off,
                       unicode::kDecompFlat + d[mid].off + d[mid].len);
            return;
        }
    }
    out.push_back(cp);
}

uint32_t compose_pair(uint32_t a, uint32_t b) {
    if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount)
        return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
    if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase &&
        b < kTBase + kTCount)
        return a + (b - kTBase);
    const uint64_t key = ((uint64_t)a << 21) | b;
    const auto& c = unicode::kComp;
    size_t lo = 0, hi = sizeof(c) / sizeof(c[0]);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (key < c[mid].pair) hi = mid;
        else if (key > c[mid].pair) lo = mid + 1;
        else return c[mid].cp;
    }
    return 0;
}
}  // namespace

// NFC (UAX #15: full canonical decomposition, canonical ordering, canonical
// composition). Text below U+0300 is already NFC, so ASCII and Latin-1 traffic
// returns untouched without decoding: a byte >= 0xCC is the only way to start a
// code point at or above U+0300. Invalid UTF-8 is returned unchanged (the
// reference cannot even receive it).
std::string Tokenizer::nfc(const std::string& s) {
    bool need = false;
    for (unsigned char c : s)
        if (c >= 0xCC) { need = true; break; }
    if (!need) return s;
    std::vector<uint32_t> d;
    d.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        int len = 0;
        const uint32_t cp = decode_cp(s, i, &len);
        if (cp >= 0x110000) return s;
        decompose_cp(cp, d);
        i += len;
    }
    for (size_t i = 1; i < d.size(); i++) {  // canonical ordering (stable)
        const uint8_t c = ccc_of(d[i]);
        if (c == 0) continue;
        for (size_t j = i; j > 0 && ccc_of(d[j - 1]) > c; j--) std::swap(d[j - 1], d[j]);
    }
    std::vector<uint32_t> r;
    r.reserve(d.size());
    size_t starter = SIZE_MAX;
    int last_cc = -1;  // ccc of the last char appended after the starter
    for (uint32_t ch : d) {
        const int cc = ccc_of(ch);
        if (starter != SIZE_MAX) {
            const bool adjacent = r.size() == starter + 1;
            const bool blocked = !adjacent && (last_cc == 0 || last_cc >= cc);
            if (!blocked) {
                if (const uint32_t comp = compose_pair(r[starter], ch)) {
                    r[starter] = comp;
                    continue;
                }
            }
        }
        if (cc == 0) { starter = r.size(); last_cc = -1; }
        else last_cc = cc;
        r.push_back(ch);
        if (cc == 0) last_cc = -1;
    }
    std::string o;
    o.reserve(s.size());
    for (uint32_t cp : r) append_utf8(o, cp);
    return o;
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& t) const {
    struct Cp { uint32_t cp; uint32_t off; uint8_t cls; };
    std::vector<Cp> v;
    v.reserve(t.size());
    for (size_t i = 0; i < t.size();) {
        int len = 0;
        const uint32_t cp = decode_cp(t, i, &len);
        v.push_back({cp, (uint32_t)i, cp >= 0x110000 ? (uint8_t)C_O : cp_class(cp)});
        i += len;
    }
    const size_t n = v.size();
    std::vector<std::string> out;
    auto emit = [&](size_t a, size_t b) {
        const size_t e = b < n ? v[b].off : t.size();
        out.push_back(t.substr(v[a].off, e - v[a].off));
    };
    // (?i:...) folds: ASCII case, and U+017F (long s) folds to 's' -- the one
    // non-ASCII fold that reaches this branch (probed against tokenizers).
    auto fold = [&](size_t j) -> uint32_t {
        if (j >= n) return 0;
        const uint32_t c = v[j].cp;
        if (c == 0x17F) return 's';
        return c < 0x80 ? (uint32_t)tolower((int)c) : 0;
    };
    size_t i = 0;
    while (i < n) {
        const uint8_t ci = v[i].cls;
        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (v[i].cp == '\'' && i + 1 < n) {
            const uint32_t c1 = fold(i + 1), c2 = fold(i + 2);
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') { emit(i, i + 2); i += 2; continue; }
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                emit(i, i + 3); i += 3; continue;
            }
        }
        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+  (a mark is a legal prefix too, but the
        // letter run would absorb it anyway, so L and M share one class)
        {
            size_t j = i;
            if (ci != C_NL && ci != C_LM && ci != C_N && j + 1 < n && v[j + 1].cls == C_LM) j++;
            if (v[j].cls == C_LM) {
                size_t k = j;
                while (k < n && v[k].cls == C_LM) k++;
                emit(i, k); i = k; continue;
            }
        }
        // \p{N}
        if (ci == C_N) { emit(i, i + 1); i++; continue; }
        // " ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"  (the optional space is U+0020 only)
        {
            size_t j = i;
            if (v[j].cp == ' ' && j + 1 < n && v[j + 1].cls == C_O) j++;
            if (v[j].cls == C_O) {
                size_t k = j;
                while (k < n && v[k].cls == C_O) k++;
                while (k < n && v[k].cls == C_NL) k++;
                emit(i, k); i = k; continue;
            }
        }
        // \s*[\r\n]+ : \s* is greedy over ALL whitespace, newlines included, and
        // backtracks only far enough for [\r\n]+ -- the match is the whitespace
        // run up to and including its LAST newline. (Before 2026-09-10 \s*
        // stopped at the first newline, so "\n \n" split in two.)
        size_t k = i;
        while (k < n && (v[k].cls == C_S || v[k].cls == C_NL)) k++;
        size_t last = SIZE_MAX;
        for (size_t p = i; p < k; p++)
            if (v[p].cls == C_NL) last = p;
        if (last != SIZE_MAX) { emit(i, last + 1); i = last + 1; continue; }
        // \s+(?!\S) | \s+ : a run followed by non-space leaves its last char
        // for the next token's prefix; a lone space is \s+
        if (k > i) {
            if (k < n && k - i > 1) k--;
            emit(i, k); i = k; continue;
        }
        emit(i, i + 1);  // unreachable for valid text: every class has a branch
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
        const std::string span = nfc(text.substr(i, end - i));
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
