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
#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace q27 {

struct ToolGrammar {
    void reset(const std::vector<std::string>& tool_names) {
        names_ = tool_names;
        // canonicalized allowlist key for mask caching (review 2026-07-09 P1
        // #3): token legality depends on names_ (NAME_VAL prefix matching), so
        // two requests with different tool sets must never share a cached
        // mask. Sorted so registration order doesn't fragment the cache.
        // '\x1f' can't appear in sampleable tool names (str_byte rejects
        // control chars), so the join is unambiguous.
        std::vector<std::string> sorted = tool_names;
        std::sort(sorted.begin(), sorted.end());
        names_key_.clear();
        for (auto& n : sorted) { names_key_ += n; names_key_ += '\x1f'; }
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
    bool done() const { return !dead_ && (st_ == DONE_ || st_ == CLOSER_ || st_ == CLOSED_); }
    // literal </tool_call> fully matched -- constraint can disengage
    bool closed() const { return !dead_ && st_ == CLOSED_; }

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
        J_VALUE_REQ,   // value required after array ',' (']' illegal here)
        J_STR,         // inside string
        J_STR_ESC,     // after backslash
        J_STR_U1, J_STR_U2, J_STR_U3, J_STR_U4, // \uXXXX
        // strict JSON number FSM (review 2026-07-09 P1 #6: a single J_NUM
        // state accepted arbitrary sign/dot/exponent orderings like 1..2e+-3,
        // which downstream json::parse then rejects -- the constraint exists
        // to make that impossible)
        J_NUM_MINUS,   // after leading '-': digit required
        J_NUM_ZERO,    // leading '0': only '.', e/E or a terminator may follow
        J_NUM_INT,     // integer digits
        J_NUM_DOT,     // after '.': fraction digit required
        J_NUM_FRAC,    // fraction digits
        J_NUM_E,       // after e/E: sign or exponent digit required
        J_NUM_ESIGN,   // after e+/e-: exponent digit required
        J_NUM_EXP,     // exponent digits
        J_LIT,         // inside true/false/null
        J_KEY,         // expect '"' of an object key (or '}' if empty obj)
        J_KEY_REQ,     // key required after object ',' ('}' illegal here)
        J_KEYSTR,      // inside object key string
        J_KEYESC,      // escape inside key string
        J_KEY_U1, J_KEY_U2, J_KEY_U3, J_KEY_U4, // \uXXXX inside key string
        J_KEYCOLON,    // expect ':' after key
        J_AFTER_VAL,   // expect ',' or closer
        OBJ_CLOSE,     // expect final '}' of the outer call object
        DONE_,         // body complete; expect ws then the literal closer
        CLOSER_,       // matching "</tool_call>" (model emits it as BPE text)
        CLOSED_        // closer consumed; anything goes (host deactivates)
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
    // allow_close: ']' may close the array here (legal for the first slot --
    // empty array -- but not right after a ',': trailing commas are invalid
    // JSON and must stay unsampleable)
    bool value_start(char c, bool allow_close = true) {
        if (c == '{') { stack_.push_back('{'); st_ = J_KEY; return true; }
        if (c == '[') { stack_.push_back('['); st_ = J_VALUE; return true; }
        if (c == '"') { st_ = J_STR; return true; }
        if (c == '-') { st_ = J_NUM_MINUS; return true; }
        if (c == '0') { st_ = J_NUM_ZERO; return true; }
        if (c >= '1' && c <= '9') { st_ = J_NUM_INT; return true; }
        if (c == 't') { lit_word_ = "true"; lit_ = 1; st_ = J_LIT; return true; }
        if (c == 'f') { lit_word_ = "false"; lit_ = 1; st_ = J_LIT; return true; }
        if (c == 'n') { lit_word_ = "null"; lit_ = 1; st_ = J_LIT; return true; }
        if (allow_close && c == ']' && !stack_.empty() && stack_.back() == '[')
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
            case J_VALUE_REQ:
                if (is_ws(c)) return true;
                return value_start(c, /*allow_close=*/false);
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
            case J_NUM_MINUS:
                if (c == '0') { st_ = J_NUM_ZERO; return true; }
                if (c >= '1' && c <= '9') { st_ = J_NUM_INT; return true; }
                return false;
            case J_NUM_ZERO:
                if (c == '.') { st_ = J_NUM_DOT; return true; }
                if (c == 'e' || c == 'E') { st_ = J_NUM_E; return true; }
                if (is_ws(c) || c == ',' || c == '}' || c == ']')
                    return num_end_redispatch(c);
                return false; // JSON forbids digits after a leading 0
            case J_NUM_INT:
                if (c >= '0' && c <= '9') return true;
                if (c == '.') { st_ = J_NUM_DOT; return true; }
                if (c == 'e' || c == 'E') { st_ = J_NUM_E; return true; }
                if (is_ws(c) || c == ',' || c == '}' || c == ']')
                    return num_end_redispatch(c);
                return false;
            case J_NUM_DOT:
                if (c >= '0' && c <= '9') { st_ = J_NUM_FRAC; return true; }
                return false;
            case J_NUM_FRAC:
                if (c >= '0' && c <= '9') return true;
                if (c == 'e' || c == 'E') { st_ = J_NUM_E; return true; }
                if (is_ws(c) || c == ',' || c == '}' || c == ']')
                    return num_end_redispatch(c);
                return false;
            case J_NUM_E:
                if (c == '+' || c == '-') { st_ = J_NUM_ESIGN; return true; }
                if (c >= '0' && c <= '9') { st_ = J_NUM_EXP; return true; }
                return false;
            case J_NUM_ESIGN:
                if (c >= '0' && c <= '9') { st_ = J_NUM_EXP; return true; }
                return false;
            case J_NUM_EXP:
                if (c >= '0' && c <= '9') return true;
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
            case J_KEY_REQ: // after ',': a key must follow (no trailing comma)
                if (is_ws(c)) return true;
                if (c == '"') { st_ = J_KEYSTR; return true; }
                return false;
            case J_KEYSTR:
                if (c == '"') { st_ = J_KEYCOLON; return true; }
                if (c == '\\') { st_ = J_KEYESC; return true; }
                return str_byte(c);
            case J_KEYESC:
                if (c == 'u') { st_ = J_KEY_U1; return true; }
                if (strchr_esc(c)) { st_ = J_KEYSTR; return true; }
                return false;
            // \uXXXX in keys: the old single-hop 'u' acceptance let incomplete
            // escapes like "\uZZ" through (review 2026-07-09 P1 #6) -- require
            // 4 hex digits exactly as the string-value states do
            case J_KEY_U1: if (!is_hex(c)) return false; st_ = J_KEY_U2; return true;
            case J_KEY_U2: if (!is_hex(c)) return false; st_ = J_KEY_U3; return true;
            case J_KEY_U3: if (!is_hex(c)) return false; st_ = J_KEY_U4; return true;
            case J_KEY_U4: if (!is_hex(c)) return false; st_ = J_KEYSTR; return true;
            case J_KEYCOLON:
                if (is_ws(c)) return true;
                if (c == ':') { st_ = J_VALUE; return true; }
                return false;
            case J_AFTER_VAL:
                if (is_ws(c)) return true;
                if (c == ',') {
                    // _REQ states: the next key/value is mandatory, so '}'/']'
                    // right after the comma (trailing comma) stays unsampleable
                    st_ = (!stack_.empty() && stack_.back() == '{') ? J_KEY_REQ : J_VALUE_REQ;
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
                if (is_ws(c)) return true;
                if (c == '<') { lit_word_ = "</tool_call>"; lit_ = 1; st_ = CLOSER_; return true; }
                return false;
            case CLOSER_:
                if (lit_ < lit_word_.size()) {
                    if (c != lit_word_[lit_]) return false;
                    lit_++;
                    if (lit_ == lit_word_.size()) st_ = CLOSED_;
                    return true;
                }
                return false;
            case CLOSED_:
                return true; // constraint disengages host-side at closed()
        }
        return false;
    }

    static bool strchr_esc(char c) {
        return c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' ||
               c == 'r' || c == 't';
    }

    std::vector<std::string> names_;
    std::string names_key_; // sorted allowlist join (mask-cache key component)
    std::string name_pref_;
    std::string lit_word_;
    std::vector<char> stack_;
    size_t lit_ = 0;
    St st_ = WS_OBJ_OPEN;
    bool dead_ = false;

  public:
    // state signature for mask caching: two states with equal signatures
    // accept identical token sets (state enum + stack + literal progress +
    // name prefix + tool-name allowlist fully determine transitions). The
    // allowlist component (review 2026-07-09 P1 #3): the cache is shared
    // across requests (server-global), and NAME_VAL legality depends on which
    // names are registered -- without it, request B could inherit request A's
    // masks and be steered into A's tool names.
    std::string signature() const {
        std::string s;
        s += (char)('A' + (int)st_);
        s += dead_ ? '!' : '.';
        s.append(stack_.begin(), stack_.end());
        s += '|';
        s += std::to_string(lit_);
        s += '|';
        s += lit_word_;
        // The name-prefix and allowlist components only matter where token
        // legality can depend on them: a token from any state up to NAME_VAL
        // can span into name-prefix matching, but from ARG_COMMA onward the
        // grammar can never re-enter NAME_VAL (one name, then arguments) and
        // name_pref_ is dead state. Keying them on EVERY state duplicated
        // identical argument-state masks per tool set (and per chosen name)
        // and exhausted the 512-entry device pool (review follow-up
        // 2026-07-09 #2).
        if (st_ <= NAME_VAL) {
            s += '|';
            s += name_pref_;
            s += '|';
            s += names_key_;
        }
        return s;
    }
};

// Signature-hashed lazy cache of vocab legality bitmasks. Exact: a mask is
// built by simulating every vocab token's bytes from the (copied) grammar
// state. Masks are append-only (stable indices -> device-resident pool in
// phase 2/3). Wiring rules encoded here: the </tool_call> closer id is legal
// iff done(); EOS is never legal inside the grammar (enforcement disengages
// after the closer, upstream). Templated on the grammar type so the same
// caching machinery serves both the JSON dialect (ToolGrammar) and the XML
// dialect (ToolGrammarXml); both expose signature() + token_ok() so the body
// is identical. Instantiated explicitly at the call sites (server, metal,
// tests) since the vocabulary cache is owned per-engine.
template <class G>
struct ToolMaskCache {
    // vocab: decoded byte strings per token id (specials included verbatim)
    void init(const std::vector<std::string>* vocab, int closer_id) {
        vocab_ = vocab;
        closer_id_ = closer_id;
        words_ = ((int)vocab->size() + 31) / 32;
    }

    // returns stable mask index for the grammar's current state
    int get(const G& g) {
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
        // content dedupe (review follow-up 2026-07-09 #2): distinct
        // signatures with identical bitsets share one entry, so the
        // append-only device pool holds unique masks only (backstop behind
        // the state-conditional allowlist key above)
        std::string key((const char*)m.data(), (size_t)words_ * 4);
        auto ct = content_.find(key);
        if (ct != content_.end()) {
            index_.emplace(std::move(sig), ct->second);
            return ct->second;
        }
        int idx = (int)masks_.size();
        masks_.push_back(std::move(m));
        content_.emplace(std::move(key), idx);
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
    std::unordered_map<std::string, int> content_; // bitset bytes -> index
    std::vector<std::vector<uint32_t>> masks_;
};

// XML-dialect sibling of ToolGrammar. The JSON machine above constrains the
// body inside <tool_call>...</tool_call> to the 3.6/JSON format
// (`{"name":..., "arguments":...}`); this one constrains it to the 3.8-
// trained XML format
// (`<function=NAME><parameter=KEY>VALUE</parameter>...</function>`), so
// `--constrain-tools` can be enabled on Qwen3.8 without the grammar fighting
// the model's trained emission.
//
// Schema-aware: when the full tool name matches, the parameter-key allowlist
// switches to that tool's declared parameters (params_per_name overload of
// reset). The names-only overload leaves parameter keys unrestricted -- a
// permissive fallback for callers that don't carry the schema.
//
// Values are multi-line free text. `<` inside a value is legal ONLY as the
// start of `</parameter>` or `</function>`; `</tool_call>` inside a value is
// rejected (the value must close via `</parameter>` then the function via
// `</function>` then the body via `</tool_call>`). That is the prevention:
// a model that tries to write `<result>`, a stray `<function=...>`, or any
// other nested tag, hits a dead state and the decoder masks it out. This is
// what stops drift modes like the live chimera `{"name":"bash",</parameter>
// </function>` at the source rather than leaving the parser to refuse an
// unrecoverable call.
//
// Same external surface as ToolGrammar (reset/advance/advance_str/token_ok/
// done/closed/signature) so the templated ToolMaskCache<G> works unchanged.
struct ToolGrammarXml {
    // names-only reset (no schema): parameter keys unrestricted
    void reset(const std::vector<std::string>& tool_names) {
        reset(tool_names, {});
    }
    // schema-aware reset: params_per_name aligned with tool_names
    void reset(const std::vector<std::string>& tool_names,
               const std::vector<std::vector<std::string>>& params_per_name) {
        names_ = tool_names;
        params_per_name_ = params_per_name;
        std::vector<std::string> sorted = tool_names;
        std::sort(sorted.begin(), sorted.end());
        names_key_.clear();
        for (auto& n : sorted) { names_key_ += n; names_key_ += '\x1f'; }
        cur_params_key_.clear();
        cur_name_idx_ = -1;
        st_ = WS0;
        name_pref_.clear();
        key_pref_.clear();
        lit_word_.clear();
        lit_ = 0;
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
    // body fully consumed (after </function> + ws); </tool_call> sampleable
    bool done() const {
        return !dead_ && (st_ == DONE_ || st_ == CT_CLOSE || st_ == CLOSED_);
    }
    bool closed() const { return !dead_ && st_ == CLOSED_; }
    bool token_ok(const std::string& s) const {
        ToolGrammarXml copy = *this;
        return copy.advance_str(s);
    }

  private:
    enum St {
        WS0,            // ws before first element; first element MUST be <function
        FUNC_LT,        // consumed '<', expect 'f' for <function
        FUNC_LIT,       // matching 'function'
        FUNC_EQ,        // '='
        NAME,           // tool name (allowlist prefix), term '>'
        GT1,            // '>' after name
        WS1,            // ws after function opener, before next tag
        PARAM_LT,       // consumed '<', expect 'p' (<parameter) or '/' (closer)
        PARAM_LIT,      // matching 'parameter'
        PARAM_EQ,       // '='
        KEY,            // param key (allowlist vs cur tool params), term '>'
        GT2,            // '>' after key
        VAL,            // value body (printable + ws)
        VAL_LT,         // VAL saw '<'; expect '/' only (closer)
        SLASH,          // consumed '</', pick closer: 'p'-></parameter>, 'f'-></function>
        PARAM_CLOSE,    // matching '</parameter>'
        FUNC_CLOSE,     // matching '</function>'
        DONE_,          // ws after </function>; </tool_call> sampleable
        CT_CLOSE,       // matching '</tool_call>'
        CLOSED_         // past closer
    };

    static bool is_ws(char c) {
        return c == ' ' || c == '\n' || c == '\r' || c == '\t';
    }
    static bool val_byte(char c) {
        // value byte: printable (>= 0x20) and not '<' (handled separately)
        return (unsigned char)c >= 0x20 && c != '<';
    }

    bool step(char c) {
        switch (st_) {
            case WS0:
                if (is_ws(c)) return true;
                if (c == '<') { st_ = FUNC_LT; return true; }
                return false;
            case FUNC_LT:
                // first element: only <function legal (empty body handled by
                // engagement's "call closed within entry token")
                if (c == 'f') { lit_word_ = "function"; lit_ = 1; st_ = FUNC_LIT; return true; }
                return false;
            case FUNC_LIT:
                if (lit_ < lit_word_.size()) {
                    if (c != lit_word_[lit_]) return false;
                    lit_++;
                    if (lit_ == lit_word_.size()) st_ = FUNC_EQ;
                    return true;
                }
                return false;
            case FUNC_EQ:
                if (is_ws(c)) return true;
                if (c == '=') { st_ = NAME; name_pref_.clear(); return true; }
                return false;
            case NAME: {
                if (c == '>') {
                    for (size_t i = 0; i < names_.size(); i++)
                        if (names_[i] == name_pref_) {
                            cur_name_idx_ = (int)i;
                            cur_params_key_.clear();
                            if (i < params_per_name_.size()) {
                                auto pk = params_per_name_[i];
                                std::sort(pk.begin(), pk.end());
                                for (auto& k : pk) { cur_params_key_ += k; cur_params_key_ += '\x1f'; }
                            }
                            st_ = GT1;
                            return true;
                        }
                    return false; // undeclared tool name
                }
                std::string next = name_pref_ + c;
                for (auto& n : names_)
                    if (n.compare(0, next.size(), next) == 0 && next.size() <= n.size()) {
                        name_pref_ = next;
                        return true;
                    }
                return false;
            }
            case GT1:
                if (is_ws(c)) return true;
                if (c == '<') { st_ = PARAM_LT; return true; }
                return false;
            case WS1:
                if (is_ws(c)) return true;
                if (c == '<') { st_ = PARAM_LT; return true; }
                return false;
            case PARAM_LT:
                if (c == 'p') { lit_word_ = "parameter"; lit_ = 1; st_ = PARAM_LIT; return true; }
                if (c == '/') { st_ = SLASH; return true; }
                return false;
            case PARAM_LIT:
                if (lit_ < lit_word_.size()) {
                    if (c != lit_word_[lit_]) return false;
                    lit_++;
                    if (lit_ == lit_word_.size()) st_ = PARAM_EQ;
                    return true;
                }
                return false;
            case PARAM_EQ:
                if (is_ws(c)) return true;
                if (c == '=') { st_ = KEY; key_pref_.clear(); return true; }
                return false;
            case KEY: {
                if (c == '>') {
                    if (cur_params_key_.empty()) {
                        // no schema: permissive (accept any exact key)
                        st_ = GT2;
                        return true;
                    }
                    if (cur_name_idx_ >= 0 && cur_name_idx_ < (int)params_per_name_.size()) {
                        for (auto& k : params_per_name_[cur_name_idx_])
                            if (k == key_pref_) { st_ = GT2; return true; }
                    }
                    return false;
                }
                std::string next = key_pref_ + c;
                if (cur_params_key_.empty()) {
                    key_pref_ = next;
                    return true;
                }
                bool ok = false;
                if (cur_name_idx_ >= 0 && cur_name_idx_ < (int)params_per_name_.size()) {
                    for (auto& k : params_per_name_[cur_name_idx_])
                        if (k.compare(0, next.size(), next) == 0 && next.size() <= k.size()) {
                            ok = true; break;
                        }
                }
                if (!ok) return false;
                key_pref_ = next;
                return true;
            }
            case GT2:
                if (is_ws(c)) return true;
                st_ = VAL;
                return step(c);
            case VAL:
                if (is_ws(c)) return true;
                if (val_byte(c)) return true;
                if (c == '<') { st_ = VAL_LT; return true; }
                return false;
            case VAL_LT:
                // inside a value: only closer legal, so '</' only
                if (c == '/') { st_ = SLASH; return true; }
                return false;
            case SLASH:
                if (c == 'p') { lit_word_ = "</parameter>"; lit_ = 3; st_ = PARAM_CLOSE; return true; }
                if (c == 'f') { lit_word_ = "</function>"; lit_ = 3; st_ = FUNC_CLOSE; return true; }
                return false;
            case PARAM_CLOSE:
                if (lit_ < lit_word_.size()) {
                    if (c != lit_word_[lit_]) return false;
                    lit_++;
                    if (lit_ == lit_word_.size()) st_ = WS1;
                    return true;
                }
                return false;
            case FUNC_CLOSE:
                if (lit_ < lit_word_.size()) {
                    if (c != lit_word_[lit_]) return false;
                    lit_++;
                    if (lit_ == lit_word_.size()) st_ = DONE_;
                    return true;
                }
                return false;
            case DONE_:
                if (is_ws(c)) return true;
                if (c == '<') { lit_word_ = "</tool_call>"; lit_ = 1; st_ = CT_CLOSE; return true; }
                return false;
            case CT_CLOSE:
                if (lit_ < lit_word_.size()) {
                    if (c != lit_word_[lit_]) return false;
                    lit_++;
                    if (lit_ == lit_word_.size()) st_ = CLOSED_;
                    return true;
                }
                return false;
            case CLOSED_:
                return true;
        }
        return false;
    }

    std::vector<std::string> names_;
    std::vector<std::vector<std::string>> params_per_name_;
    std::string names_key_;
    std::string cur_params_key_;
    std::string name_pref_;
    std::string key_pref_;
    std::string lit_word_;
    int cur_name_idx_ = -1;
    size_t lit_ = 0;
    St st_ = WS0;
    bool dead_ = false;

  public:
    // Same signature scheme as ToolGrammar: state + lit progress + allowlist
    // components only where token legality depends on them. NAME depends on
    // names_key_ + name_pref_; KEY depends on cur_params_key_ + key_pref_;
    // literal-match states depend on lit_. Argument/closer states never
    // re-enter the allowlist branches.
    std::string signature() const {
        std::string s;
        s += (char)('a' + (int)st_);
        s += dead_ ? '!' : '.';
        if (st_ == NAME) {
            s += '|';
            s += name_pref_;
            s += '|';
            s += names_key_;
        } else if (st_ == KEY) {
            s += '|';
            s += key_pref_;
            s += '|';
            s += cur_params_key_;
        } else if (st_ == FUNC_LIT || st_ == PARAM_LIT) {
            s += '|';
            s += std::to_string(lit_);
            s += '|';
            s += lit_word_;
        } else if (st_ == PARAM_CLOSE || st_ == FUNC_CLOSE || st_ == CT_CLOSE) {
            s += '|';
            s += std::to_string(lit_);
            s += '|';
            s += lit_word_;
        }
        return s;
    }
};

} // namespace q27
