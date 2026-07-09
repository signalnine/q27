// CPU unit tests for src/toolconstrain.h (BasicToolConstrainer): the engage-lag
// fix host logic (scan_round truncation semantics, skip_feed dedup) and the
// serving-state gates (sticky pool-full disengage, split-brain id validation).
// Hermetic: FakeEngine + FakeTok, real ToolGrammar/ToolMaskCache.
//
// Build+run: g++ -std=c++17 -I src tools/test_toolconstrain.cpp -o build/test_toolconstrain && ./build/test_toolconstrain
#include "toolconstrain.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

// ---- fakes ----------------------------------------------------------------
struct FakeTok {
    std::vector<std::string> vocab;
    std::string decode_one(int id) const { return vocab[(size_t)id]; }
};
struct FakeEngine {
    int mask_pool_used = 0;
    int pool_cap = 512;
    int add_calls = 0;
    std::vector<int> constraint_log; // every set_tool_constraint arg, in order
    int last_masks5[5] = {-9, -9, -9, -9, -9};
    int mask_pool_add(const void*) {
        add_calls++;
        if (mask_pool_used >= pool_cap) return -1;
        return mask_pool_used++;
    }
    void set_tool_constraint(int id) { constraint_log.push_back(id); }
    void set_tool_masks5(const int ids[5]) {
        for (int i = 0; i < 5; i++) last_masks5[i] = ids[i];
    }
};

using TC = q27::BasicToolConstrainer<FakeEngine, FakeTok>;

// vocab ids used by the tests
enum {
    T_HELLO = 0,   // "Hello "
    T_MARK,        // "<tool_call>"
    T_NL,          // "\n"
    T_JOPEN,       // "{\""
    T_NAME,        // "name"
    T_COLONQ,      // "\": \""
    T_GET,         // "get"
    T_PROJ,        // "_project"
    T_MARKA,       // "<tool"  (spanning half 1)
    T_MARKB,       // "_call>" (spanning half 2)
    T_MARKREM,     // "<tool_call>\n{\""  (marker + legal rem)
    T_MARKBAD,     // "<tool_call>xq"     (marker + illegal rem)
    T_CLOSER,      // "</tool_call>"
    T_BODY,        // "\", \"arguments\": {}}"  (rest of a full call body)
    T_N
};

static FakeTok mk_tok() {
    FakeTok t;
    t.vocab.resize(T_N);
    t.vocab[T_HELLO] = "Hello ";
    t.vocab[T_MARK] = "<tool_call>";
    t.vocab[T_NL] = "\n";
    t.vocab[T_JOPEN] = "{\"";
    t.vocab[T_NAME] = "name";
    t.vocab[T_COLONQ] = "\": \"";
    t.vocab[T_GET] = "get";
    t.vocab[T_PROJ] = "_project";
    t.vocab[T_MARKA] = "<tool";
    t.vocab[T_MARKB] = "_call>";
    t.vocab[T_MARKREM] = "<tool_call>\n{\"";
    t.vocab[T_MARKBAD] = "<tool_call>xq";
    t.vocab[T_CLOSER] = "</tool_call>";
    t.vocab[T_BODY] = "\", \"arguments\": {}}";
    return t;
}

struct Rig {
    FakeTok tok = mk_tok();
    FakeEngine eng;
    q27::ToolMaskCache cache;
    std::vector<int> host2dev;
    TC tc;
    Rig() {
        cache.init(&tok.vocab, T_CLOSER);
        tc.eng = &eng;
        tc.tok = &tok;
        tc.cache = &cache;
        tc.host2dev = &host2dev;
        tc.enabled = true;
        tc.begin({"get_project", "run_tests"});
    }
};

static int fails = 0;
#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
            fails++;                                                                   \
        }                                                                              \
    } while (0)

// C1: marker completing mid-round engages, stages the mask, truncates at m=j+1
static void test_c1_engage_truncate_midround() {
    Rig r;
    int em[4] = {T_HELLO, T_MARK, T_GET, T_PROJ};
    int m = r.tc.scan_round(em, 4);
    CHECK(m == 2);                    // keep "Hello ", "<tool_call>"; discard the rest
    CHECK(r.tc.active);
    CHECK(r.tc.engaged == 1);
    CHECK(!r.eng.constraint_log.empty() && r.eng.constraint_log.back() >= 0);
    CHECK(r.eng.add_calls == 1);      // one mask built + uploaded
    CHECK(r.tc.skip_feed == 2);       // kept tokens must not re-feed the grammar
}

// C2: marker spanning tokens AND rounds engages exactly once via the tail carry
static void test_c2_marker_spans_rounds() {
    Rig r;
    int a[2] = {T_HELLO, T_MARKA};
    CHECK(r.tc.scan_round(a, 2) == -1);
    CHECK(!r.tc.active);
    int b[2] = {T_MARKB, T_GET};
    int m = r.tc.scan_round(b, 2);
    CHECK(m == 1);                    // completes at index 0 of round B
    CHECK(r.tc.active);
    CHECK(r.tc.engaged == 1);
}

// C3a: rem bytes after the marker advance the grammar (legal rem)
static void test_c3_rem_advances() {
    Rig r;
    int em[1] = {T_MARKREM};
    int m = r.tc.scan_round(em, 1);
    CHECK(m == 1);                    // m == n: pending-only refinish
    CHECK(r.tc.active);
    // grammar consumed "\n{\"" -- next legal continuation is the "name" key
    q27::ToolGrammar probe = r.tc.tg;
    CHECK(probe.advance_str("name"));
    CHECK(!q27::ToolGrammar(r.tc.tg).advance('x')); // 'x' is not a legal key start here
}

// C3b: illegal rem -> no engage, scan continues (no truncation), tail keeps working
static void test_c3_illegal_rem_drops() {
    Rig r;
    int em[2] = {T_MARKBAD, T_HELLO};
    int m = r.tc.scan_round(em, 2);
    CHECK(m == -1);
    CHECK(!r.tc.active);
    CHECK(r.tc.disengaged == 1);
    // a later clean marker still engages (tail was not corrupted)
    int em2[1] = {T_MARK};
    CHECK(r.tc.scan_round(em2, 1) == 1);
    CHECK(r.tc.active);
}

// C4: kept tokens re-delivered via on_id do not re-advance the engaged grammar
static void test_c4_no_double_feed() {
    Rig r;
    int em[4] = {T_HELLO, T_MARK, T_GET, T_PROJ};
    int m = r.tc.scan_round(em, 4);
    CHECK(m == 2);
    std::string sig0 = r.tc.tg.signature();
    r.tc.on_id(T_HELLO); // kept token 0
    r.tc.on_id(T_MARK);  // kept token 1 (the marker token itself)
    CHECK(r.tc.tg.signature() == sig0);
    CHECK(r.tc.skip_feed == 0);
    // the NEXT token (a re-decoded, masked one) does feed
    r.tc.on_id(T_JOPEN);
    CHECK(r.tc.tg.signature() != sig0);
}

// C5: marker completing at the LAST emitted token returns n (pending-only refinish)
static void test_c5_marker_at_last_token() {
    Rig r;
    int em[2] = {T_HELLO, T_MARK};
    int m = r.tc.scan_round(em, 2);
    CHECK(m == 2);
    CHECK(r.tc.active);
    CHECK(r.tc.skip_feed == 2);
}

// C6: pool-full -> sticky per-request disengage; fresh begin() re-arms
static void test_c6_pool_full_sticky() {
    Rig r;
    r.eng.pool_cap = 0; // every mask_pool_add fails
    int em[2] = {T_MARK, T_GET};
    int m = r.tc.scan_round(em, 2);
    CHECK(m == -1);
    CHECK(!r.tc.active);
    CHECK(r.tc.pool_dead);
    CHECK(r.tc.pool_drops == 1);
    long eng_before = r.tc.engaged;
    int em2[1] = {T_MARK};
    CHECK(r.tc.scan_round(em2, 1) == -1); // sticky: no re-engage attempt this request
    CHECK(r.tc.engaged == eng_before);
    r.eng.pool_cap = 512;
    r.tc.begin({"get_project", "run_tests"}); // next request re-arms
    CHECK(!r.tc.pool_dead);
    int em3[1] = {T_MARK};
    CHECK(r.tc.scan_round(em3, 1) == 1);
    CHECK(r.tc.active);
}

// C7: stale per-slot pool id (>= mask_pool_used) is detected and re-uploaded
static void test_c7_split_brain_rebind() {
    Rig r;
    int em[1] = {T_MARK};
    CHECK(r.tc.scan_round(em, 1) == 1); // engage -> host2dev[ci] = pool id 0
    CHECK(r.eng.add_calls == 1);
    // simulate an engine whose pool was reset behind the map's back
    r.eng.mask_pool_used = 0;
    r.tc.begin({"get_project", "run_tests"});
    int em2[1] = {T_MARK};
    CHECK(r.tc.scan_round(em2, 1) == 1);
    CHECK(r.eng.add_calls == 2);  // stale id detected -> mask re-uploaded
    CHECK(r.tc.rebinds == 1);
    CHECK(!r.eng.constraint_log.empty() && r.eng.constraint_log.back() == 0);
}

// m3 (review): pool_dead set mid-scan must stop the scan -- a second marker in
// the SAME round must not re-engage (a cached mask id could otherwise bypass
// the pool and later disengage nondeterministically mid-call)
static void test_m3_pool_dead_stops_scan() {
    Rig r;
    r.eng.pool_cap = 0;
    int em[4] = {T_MARK, T_HELLO, T_MARK, T_GET};
    int m = r.tc.scan_round(em, 4);
    CHECK(m == -1);
    CHECK(r.tc.pool_dead);
    CHECK(r.tc.engaged == 1);    // first marker only; second never attempted
    CHECK(r.tc.pool_drops == 1);
    CHECK(!r.tc.active);
}

// m4 (review): a call that COMPLETES inside the entry token's remainder bytes
// (marker+body+closer in one piece) must not stage a mask for a closed state
static void test_m4_closed_within_entry_token() {
    Rig r;
    r.tok.vocab.push_back("<tool_call>{\"name\": \"run_tests\", \"arguments\": {}}</tool_call>");
    int one_shot = (int)r.tok.vocab.size() - 1;
    int em[2] = {one_shot, T_HELLO};
    int m = r.tc.scan_round(em, 2);
    CHECK(m == -1);              // nothing to truncate; call already complete
    CHECK(!r.tc.active);         // closed, not left engaged
    // no constraint may be left staged for the closed state
    CHECK(r.eng.constraint_log.empty() || r.eng.constraint_log.back() == -1);
}

// C13 (host half): a full call closes; closer disengages; constraint cleared
static void test_closer_disengages() {
    Rig r;
    int em[1] = {T_MARKREM}; // engage with rem "\n{\"" already consumed
    CHECK(r.tc.scan_round(em, 1) == 1);
    r.tc.on_id(T_MARKREM); // skip_feed swallows the kept token
    // masked in-grammar tokens arrive one per round (cap=1)
    for (int id : {T_NAME, T_COLONQ, T_GET, T_PROJ, T_BODY, T_CLOSER}) r.tc.on_id(id);
    CHECK(!r.tc.active);
    CHECK(!r.eng.constraint_log.empty() && r.eng.constraint_log.back() == -1);
    CHECK(r.tc.disengaged == 0); // clean close is not a disengage
}

// R1 (review 2026-07-09 P1 #3): masks are keyed by the tool-name allowlist.
// The cache is server-global and shared across requests; before the fix, two
// grammars at the same state with DIFFERENT allowlists collided on one cache
// entry, so request B could be steered into request A's tool names.
static void test_r1_allowlist_in_cache_key() {
    FakeTok tok = mk_tok();
    q27::ToolMaskCache cache;
    cache.init(&tok.vocab, T_CLOSER);
    q27::ToolGrammar ga, gb;
    ga.reset({"get_project"});
    gb.reset({"run_tests"});
    CHECK(ga.signature() != gb.signature());
    const char* pre = "{\"name\": \"";
    CHECK(ga.advance_str(pre));
    CHECK(gb.advance_str(pre));
    int ia = cache.get(ga), ib = cache.get(gb);
    CHECK(ia != ib);
    const auto& ma = cache.mask(ia);
    const auto& mb = cache.mask(ib);
    CHECK(ma[T_GET >> 5] & (1u << (T_GET & 31)));    // "get" legal under A
    CHECK(!(mb[T_GET >> 5] & (1u << (T_GET & 31)))); // ...and illegal under B
    // registration order must not fragment the cache (canonicalized key)
    q27::ToolGrammar gc, gd;
    gc.reset({"aa", "bb"});
    gd.reset({"bb", "aa"});
    CHECK(gc.signature() == gd.signature());
}

// R3 (review follow-up 2026-07-09 #2): allowlist isolation must not duplicate
// allowlist-independent masks. Argument-state signatures carry no allowlist
// component (token legality past NAME_VAL cannot depend on names_), and the
// content-dedupe backstop collapses identical bitsets -- so N tool sets share
// one pool entry per argument state instead of exhausting the 512-entry pool.
static void test_r3_no_mask_duplication_across_allowlists() {
    FakeTok tok = mk_tok();
    q27::ToolMaskCache cache;
    cache.init(&tok.vocab, T_CLOSER);
    q27::ToolGrammar ga, gb;
    ga.reset({"get_project"});
    gb.reset({"run_tests"});
    CHECK(ga.advance_str("{\"name\": \"get_project\", \"arguments\": {\"x\": "));
    CHECK(gb.advance_str("{\"name\": \"run_tests\", \"arguments\": {\"x\": "));
    // identical argument state -> identical signature -> same cache entry
    CHECK(ga.signature() == gb.signature());
    int ia = cache.get(ga);
    size_t n_masks = cache.size();
    int ib = cache.get(gb);
    CHECK(ia == ib);
    CHECK(cache.size() == n_masks); // pool did not grow for the second tool set
    // name-phase states still keyed by allowlist (isolation preserved)
    q27::ToolGrammar na, nb;
    na.reset({"get_project"});
    nb.reset({"run_tests"});
    CHECK(na.advance_str("{\"name\": \""));
    CHECK(nb.advance_str("{\"name\": \""));
    CHECK(na.signature() != nb.signature());
    CHECK(cache.get(na) != cache.get(nb));
}

// R2 (review 2026-07-09 P1 #6): the grammar must reject malformed JSON that
// json::parse downstream would reject -- otherwise constrained decode can
// still produce an unparseable tool call.
static bool accepts_args(const std::string& args_json) {
    q27::ToolGrammar g;
    g.reset({"t"});
    if (!g.advance_str("{\"name\": \"t\", \"arguments\": " + args_json + "}")) return false;
    return g.done();
}
static void test_r2_strict_json() {
    // valid JSON stays sampleable
    CHECK(accepts_args("{\"x\": 1}"));
    CHECK(accepts_args("{\"x\": -0.5}"));
    CHECK(accepts_args("{\"x\": 0}"));
    CHECK(accepts_args("{\"x\": -0}"));
    CHECK(accepts_args("{\"x\": 1e5}"));
    CHECK(accepts_args("{\"x\": 1E+5}"));
    CHECK(accepts_args("{\"x\": 0.25e-3}"));
    CHECK(accepts_args("{\"x\": [1, 2.5, -3e2]}"));
    CHECK(accepts_args("{\"x\": []}"));
    CHECK(accepts_args("{\"x\": {}}"));
    CHECK(accepts_args("{\"x\": \"\\u0041\"}"));
    CHECK(accepts_args("{\"\\u0041\": 1}"));
    // malformed numbers (old single J_NUM state accepted all of these)
    CHECK(!accepts_args("{\"x\": 1..2}"));
    CHECK(!accepts_args("{\"x\": 1e+-3}"));
    CHECK(!accepts_args("{\"x\": --1}"));
    CHECK(!accepts_args("{\"x\": +1}"));
    CHECK(!accepts_args("{\"x\": 1e}"));
    CHECK(!accepts_args("{\"x\": 1e5e5}"));
    CHECK(!accepts_args("{\"x\": 01}"));
    CHECK(!accepts_args("{\"x\": 1.}"));
    CHECK(!accepts_args("{\"x\": .5}"));
    CHECK(!accepts_args("{\"x\": -}"));
    // trailing commas
    CHECK(!accepts_args("{\"x\": 1,}"));
    CHECK(!accepts_args("{\"x\": [1,]}"));
    // incomplete \u escape in an object key (old J_KEYESC took 'u' in one hop)
    CHECK(!accepts_args("{\"\\uZZ00\": 1}"));
}

int main() {
    test_c1_engage_truncate_midround();
    test_c2_marker_spans_rounds();
    test_c3_rem_advances();
    test_c3_illegal_rem_drops();
    test_c4_no_double_feed();
    test_c5_marker_at_last_token();
    test_c6_pool_full_sticky();
    test_c7_split_brain_rebind();
    test_m3_pool_dead_stops_scan();
    test_m4_closed_within_entry_token();
    test_closer_disengages();
    test_r1_allowlist_in_cache_key();
    test_r2_strict_json();
    test_r3_no_mask_duplication_across_allowlists();
    if (fails) {
        fprintf(stderr, "test_toolconstrain: %d FAILED\n", fails);
        return 1;
    }
    printf("test_toolconstrain: ALL PASS\n");
    return 0;
}
