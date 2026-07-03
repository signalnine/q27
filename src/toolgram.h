// P7: char-level pushdown machine for the <tool_call> body. Once the model
// emits <tool_call>, decode is constrained so that only
//   {"name": "<registered tool>", "arguments": <valid JSON object>}
// (plus surrounding whitespace and the </tool_call> closer token) is
// sampleable. Built against the five observed drift modes of Qwopus v1.4
// under no-think greedy (see api_common.h parse_bare_tool_calls): dropped
// wrapper, unterminated JSON, <content>-tagged values, {"tool_call": opener,
// raw control chars inside strings. The machine advances per accepted char;
// token legality is checked by simulating a token's bytes on a copy
// (token_ok). EOS/im_end must be masked upstream until done().
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace q27 {

struct ToolGrammar {
    void reset(const std::vector<std::string>& tool_names) {
        names_ = tool_names;
        st_ = WS_OBJ_OPEN;
        lit_ = 0;
        name_pref_.clear();
        stack_.clear();
        dead_ = false;
    }

    bool advance(char c) {
        if (dead_) return false;
        if (!step(c)) { dead_ = true; return false; }
        return true;
    }

    bool advance_str(const std::string& s) {
        for (char c : s)
            if (!advance(c)) return false;
        return true;
    }

    // full call body consumed (outer object closed); trailing ws still legal
    bool done() const { return !dead_ && st_ == DONE_; }

    // would every byte of s be legal from the current state?
    bool token_ok(const std::string& s) const {
        ToolGrammar copy = *this;
        return copy.advance_str(s);
    }

  private:
    enum St {
        WS_OBJ_OPEN,   // expect '{'
        KEY_OPEN_Q,    // expect '"' of "name"
        KEY_NAME,      // matching literal name
        KEY_COLON,     // expect ':'
        NAME_OPEN_Q,   // expect '"' of the tool-name value
        NAME_VAL,      // tool-name chars (prefix of a registered name)
        ARG_COMMA,     // expect ','
        ARGKEY_OPEN_Q, // expect '"' of "arguments"
        ARGKEY,        // matching literal arguments
        ARGS_COLON,    // expect ':'
        J_VALUE,       // JSON value start (first one must be '{')
        J_STR,         // inside string
        J_STR_ESC,     // after backslash
        J_STR_U1, J_STR_U2, J_STR_U3, J_STR_U4, // \uXXXX
        J_NUM,         // inside number
        J_LIT,         // inside true/false/null
        J_KEY,         // expect '"' of an object key (or '}' if empty obj)
        J_KEYSTR,      // inside object key string
        J_KEYESC,      // escape inside key string
        J_KEYCOLON,    // expect ':' after key
        J_AFTER_VAL,   // expect ',' or closer
        OBJ_CLOSE,     // expect final '}' of the outer call object
        DONE_
    };

    static bool is_ws(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }
    static bool is_hex(char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    }
    // legal raw byte inside a JSON string: anything except control chars,
    // '"' and '\' (those switch state). UTF-8 continuation bytes are < 0 as
    // signed char -- allowed.
    static bool str_byte(char c) { return (unsigned char)c >= 0x20; }

    bool pop_or_finish(char c) {
        // c is the matching closer already validated against stack top
        stack_.pop_back();
        if (stack_.empty()) { st_ = OBJ_CLOSE; return true; }
        st_ = J_AFTER_VAL;
        (void)c;
        return true;
    }

    // dispatch c as the start of a JSON value (stack non-empty context)
    bool value_start(char c) {
        if (c == '{') { stack_.push_back('{'); st_ = J_KEY; return true; }
        if (c == '[') { stack_.push_back('['); st_ = J_VALUE; return true; }
        if (c == '"') { st_ = J_STR; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) { st_ = J_NUM; return true; }
        if (c == 't') { lit_word_ = "true"; lit_ = 1; st_ = J_LIT; return true; }
        if (c == 'f') { lit_word_ = "false"; lit_ = 1; st_ = J_LIT; return true; }
        if (c == 'n') { lit_word_ = "null"; lit_ = 1; st_ = J_LIT; return true; }
        if (c == ']' && !stack_.empty() && stack_.back() == '[')
            return pop_or_finish(c); // empty array
        return false;
    }

    // number ended by a structural char: re-dispatch that char
    bool num_end_redispatch(char c) {
        st_ = J_AFTER_VAL;
        return step(c);
    }

    bool step(char c) {
        switch (st_) {
            case WS_OBJ_OPEN:
                if (is_ws(c)) return true;
                if (c == '{') { st_ = KEY_OPEN_Q; return true; }
                return false;
            case KEY_OPEN_Q:
                if (is_ws(c)) return true;
                if (c == '"') { lit_ = 0; return st_ = KEY_NAME, true; }
                return false;
            case KEY_NAME: {
                static const char* KW = "name";
                if (lit_ < 4) { if (c != KW[lit_]) return false; lit_++; return true; }
                if (c == '"') { st_ = KEY_COLON; return true; }
                return false;
            }
            case KEY_COLON:
                if (is_ws(c)) return true;
                if (c == ':') { st_ = NAME_OPEN_Q; return true; }
                return false;
            case NAME_OPEN_Q:
                if (is_ws(c)) return true;
                if (c == '"') { name_pref_.clear(); st_ = NAME_VAL; return true; }
                return false;
            case NAME_VAL: {
                if (c == '"') {
                    for (auto& n : names_)
                        if (n == name_pref_) { st_ = ARG_COMMA; return true; }
                    return false; // no exact tool name
                }
                std::string next = name_pref_ + c;
                for (auto& n : names_)
                    if (n.compare(0, next.size(), next) == 0 && next.size() <= n.size()) {
                        name_pref_ = next;
                        return true;
                    }
                return false;
            }
            case ARG_COMMA:
                if (is_ws(c)) return true;
                if (c == ',') { st_ = ARGKEY_OPEN_Q; return true; }
                return false;
            case ARGKEY_OPEN_Q:
                if (is_ws(c)) return true;
                if (c == '"') { lit_ = 0; st_ = ARGKEY; return true; }
                return false;
            case ARGKEY: {
                static const char* KW = "arguments";
                if (lit_ < 9) { if (c != KW[lit_]) return false; lit_++; return true; }
                if (c == '"') { st_ = ARGS_COLON; return true; }
                return false;
            }
            case ARGS_COLON:
                if (is_ws(c)) return true;
                if (c == ':') { st_ = J_VALUE; return true; }
                return false;
            case J_VALUE:
                if (is_ws(c)) return true;
                if (stack_.empty()) {
                    // arguments value itself must be an object
                    if (c == '{') { stack_.push_back('{'); st_ = J_KEY; return true; }
                    return false;
                }
                return value_start(c);
            case J_STR:
                if (c == '"') { st_ = J_AFTER_VAL; return true; }
                if (c == '\\') { st_ = J_STR_ESC; return true; }
                return str_byte(c);
            case J_STR_ESC:
                if (c == 'u') { st_ = J_STR_U1; return true; }
                if (strchr_esc(c)) { st_ = J_STR; return true; }
                return false;
            case J_STR_U1: if (!is_hex(c)) return false; st_ = J_STR_U2; return true;
            case J_STR_U2: if (!is_hex(c)) return false; st_ = J_STR_U3; return true;
            case J_STR_U3: if (!is_hex(c)) return false; st_ = J_STR_U4; return true;
            case J_STR_U4: if (!is_hex(c)) return false; st_ = J_STR; return true;
            case J_NUM:
                if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
                    c == '+' || c == '-')
                    return true;
                if (is_ws(c) || c == ',' || c == '}' || c == ']')
                    return num_end_redispatch(c);
                return false;
            case J_LIT:
                if (lit_ < lit_word_.size()) {
                    if (c != lit_word_[lit_]) return false;
                    lit_++;
                    if (lit_ == lit_word_.size()) st_ = J_AFTER_VAL;
                    return true;
                }
                return false;
            case J_KEY:
                if (is_ws(c)) return true;
                if (c == '"') { st_ = J_KEYSTR; return true; }
                if (c == '}' && !stack_.empty() && stack_.back() == '{')
                    return pop_or_finish(c); // empty object
                return false;
            case J_KEYSTR:
                if (c == '"') { st_ = J_KEYCOLON; return true; }
                if (c == '\\') { st_ = J_KEYESC; return true; }
                return str_byte(c);
            case J_KEYESC:
                if (c == 'u' || strchr_esc(c)) { st_ = J_KEYSTR; return true; }
                return false; // \uXXXX in keys: rare; accept simple escapes only
            case J_KEYCOLON:
                if (is_ws(c)) return true;
                if (c == ':') { st_ = J_VALUE; return true; }
                return false;
            case J_AFTER_VAL:
                if (is_ws(c)) return true;
                if (c == ',') {
                    st_ = (!stack_.empty() && stack_.back() == '{') ? J_KEY : J_VALUE;
                    return true;
                }
                if (c == '}' && !stack_.empty() && stack_.back() == '{')
                    return pop_or_finish(c);
                if (c == ']' && !stack_.empty() && stack_.back() == '[')
                    return pop_or_finish(c);
                return false;
            case OBJ_CLOSE:
                if (is_ws(c)) return true;
                if (c == '}') { st_ = DONE_; return true; }
                return false;
            case DONE_:
                return is_ws(c); // only ws until the </tool_call> token
        }
        return false;
    }

    static bool strchr_esc(char c) {
        return c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' ||
               c == 'r' || c == 't';
    }

    std::vector<std::string> names_;
    std::string name_pref_;
    std::string lit_word_;
    std::vector<char> stack_;
    size_t lit_ = 0;
    St st_ = WS_OBJ_OPEN;
    bool dead_ = false;

  public:
    // state signature for mask caching: two states with equal signatures
    // accept identical token sets (state enum + stack + literal progress +
    // name prefix fully determine transitions)
    std::string signature() const {
        std::string s;
        s += (char)('A' + (int)st_);
        s += dead_ ? '!' : '.';
        s.append(stack_.begin(), stack_.end());
        s += '|';
        s += std::to_string(lit_);
        s += '|';
        s += lit_word_;
        s += '|';
        s += name_pref_;
        return s;
    }
};

// Signature-hashed lazy cache of vocab legality bitmasks. Exact: a mask is
// built by simulating every vocab token's bytes from the (copied) grammar
// state. Masks are append-only (stable indices -> device-resident pool in
// phase 2/3). Wiring rules encoded here: the </tool_call> closer id is legal
// iff done(); EOS is never legal inside the grammar (enforcement disengages
// after the closer, upstream).
struct ToolMaskCache {
    // vocab: decoded byte strings per token id (specials included verbatim)
    void init(const std::vector<std::string>* vocab, int closer_id) {
        vocab_ = vocab;
        closer_id_ = closer_id;
        words_ = ((int)vocab->size() + 31) / 32;
    }

    // returns stable mask index for the grammar's current state
    int get(const ToolGrammar& g) {
        std::string sig = g.signature();
        auto it = index_.find(sig);
        if (it != index_.end()) return it->second;
        std::vector<uint32_t> m(words_, 0);
        const auto& v = *vocab_;
        for (size_t id = 0; id < v.size(); id++) {
            bool ok;
            if ((int)id == closer_id_)
                ok = g.done();
            else if (v[id].empty())
                ok = false; // specials/EOS and empty entries: never legal in-grammar
            else
                ok = g.token_ok(v[id]);
            if (ok) m[id >> 5] |= 1u << (id & 31);
        }
        int idx = (int)masks_.size();
        masks_.push_back(std::move(m));
        index_.emplace(std::move(sig), idx);
        return idx;
    }

    const std::vector<uint32_t>& mask(int idx) const { return masks_[idx]; }
    size_t size() const { return masks_.size(); }
    int words() const { return words_; }

  private:
    const std::vector<std::string>* vocab_ = nullptr;
    int closer_id_ = -1;
    int words_ = 0;
    std::unordered_map<std::string, int> index_;
    std::vector<std::vector<uint32_t>> masks_;
};

} // namespace q27
