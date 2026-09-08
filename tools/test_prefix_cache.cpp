// CPU unit tests for the P16 persistent prefix cache (src/prefix_cache.h).
// Host-only by design -- no CUDA, no engine, no model. The device-side copies
// (Engine::pfx_export / pfx_import) are covered by the live restart gate in
// docs/plans/2026-07-24-persistent-prefix-cache.md, not here.
//
// Build+run: g++ -std=c++17 -I src tools/test_prefix_cache.cpp -o build/test_prefix_cache && ./build/test_prefix_cache
#include "prefix_cache.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

static const uint64_t COMPAT_A = 0xAAAA1111AAAA1111ULL;
static const uint64_t COMPAT_B = 0xBBBB2222BBBB2222ULL;

static std::string tmproot(const char* tag) {
    std::string p = std::string("/tmp/q27_pfx_test_") + tag;
    // best-effort clean between runs
    std::string cmd = "rm -rf '" + p + "'";
    if (system(cmd.c_str()) != 0) { /* fresh run, nothing to remove */ }
    return p;
}

static std::vector<int> seq(int n, int start = 1000) {
    std::vector<int> v((size_t)n);
    for (int i = 0; i < n; i++) v[(size_t)i] = start + i;
    return v;
}

static q27::PrefixCacheCfg cfg_for(const std::string& root, int min_tokens = 8) {
    q27::PrefixCacheCfg c;
    c.root = root;
    c.min_tokens = min_tokens;
    c.max_tokens = 1 << 20;
    c.step_tokens = 1;
    return c;
}

static void test_roundtrip_write_find_read() {
    const std::string root = tmproot("rt");
    q27::PrefixCache pc;
    CHECK(pc.init(cfg_for(root), COMPAT_A));
    CHECK(pc.enabled());

    const std::vector<int> toks = seq(64);
    const std::string gdn = "GDNSTATEGDNSTATE", kv = "KVROWSKVROWSKVROWS";
    CHECK(pc.write(toks, 32, gdn.data(), gdn.size(), kv.data(), kv.size()));
    CHECK(pc.size() == 1);
    CHECK(pc.has(toks, 32));
    CHECK(!pc.has(toks, 31));

    // a longer prompt that extends the stored prefix must hit
    std::vector<int> prompt = seq(100);
    q27::PrefixCache::Entry e;
    CHECK(pc.find(prompt, &e));
    CHECK(e.L == 32);

    std::vector<char> buf(gdn.size() + kv.size());
    CHECK(pc.read_state(e, buf.data(), buf.size()));
    CHECK(std::string(buf.data(), gdn.size()) == gdn);
    CHECK(std::string(buf.data() + gdn.size(), kv.size()) == kv);
}

static void test_divergent_prompt_misses() {
    const std::string root = tmproot("div");
    q27::PrefixCache pc;
    CHECK(pc.init(cfg_for(root), COMPAT_A));
    const std::vector<int> toks = seq(64);
    CHECK(pc.write(toks, 32, "g", 1, "k", 1));

    std::vector<int> prompt = seq(100);
    prompt[17] = 999999; // diverges INSIDE the stored prefix
    q27::PrefixCache::Entry e;
    CHECK(!pc.find(prompt, &e));
}

static void test_exact_length_prompt_misses() {
    // L == prompt.size() would leave no token to decode from -- the same
    // predicate generate() applies to the P8 snapshot (snap_toks <= NP-1).
    const std::string root = tmproot("exact");
    q27::PrefixCache pc;
    CHECK(pc.init(cfg_for(root), COMPAT_A));
    const std::vector<int> toks = seq(32);
    CHECK(pc.write(toks, 32, "g", 1, "k", 1));
    q27::PrefixCache::Entry e;
    CHECK(!pc.find(toks, &e));               // exactly as long: no
    CHECK(pc.find(seq(33), &e));             // one token longer: yes
    CHECK(e.L == 32);
}

static void test_longest_prefix_wins() {
    const std::string root = tmproot("longest");
    q27::PrefixCache pc;
    CHECK(pc.init(cfg_for(root), COMPAT_A));
    const std::vector<int> toks = seq(256);
    CHECK(pc.write(toks, 16, "g", 1, "k", 1));
    CHECK(pc.write(toks, 64, "g", 1, "k", 1));
    CHECK(pc.write(toks, 32, "g", 1, "k", 1));
    q27::PrefixCache::Entry e;
    CHECK(pc.find(seq(200), &e));
    CHECK(e.L == 64);
}

static void test_below_min_tokens_never_matches() {
    const std::string root = tmproot("min");
    q27::PrefixCache pc;
    q27::PrefixCacheCfg c = cfg_for(root, /*min_tokens=*/64);
    CHECK(pc.init(c, COMPAT_A));
    const std::vector<int> toks = seq(256);
    CHECK(pc.write(toks, 32, "g", 1, "k", 1)); // writable, but under the floor
    q27::PrefixCache::Entry e;
    CHECK(!pc.find(seq(200), &e));
}

// THE gate this whole file exists for: a key collision must not restore
// another conversation's state. We forge it -- write an entry, then overwrite
// its token payload in place with different tokens so the filename key still
// "matches" a prompt it has nothing to do with.
static void test_forged_key_collision_is_refused() {
    const std::string root = tmproot("collide");
    q27::PrefixCache pc;
    CHECK(pc.init(cfg_for(root), COMPAT_A));
    const std::vector<int> toks = seq(64);
    CHECK(pc.write(toks, 32, "g", 1, "k", 1));

    q27::PrefixCache::Entry e;
    CHECK(pc.find(seq(100), &e)); // sanity: hits before tampering

    // rewrite the stored token vector, leaving header+key untouched
    FILE* f = fopen(e.path.c_str(), "r+b");
    CHECK(f != nullptr);
    if (f) {
        std::vector<int> other = seq(32, /*start=*/500000);
        fseek(f, (long)sizeof(q27::PfxHdr), SEEK_SET);
        fwrite(other.data(), sizeof(int), other.size(), f);
        fclose(f);
    }
    q27::PrefixCache pc2;
    CHECK(pc2.init(cfg_for(root), COMPAT_A));
    q27::PrefixCache::Entry e2;
    CHECK(!pc2.find(seq(100), &e2)); // key still matches, TOKENS do not -> refuse
}

static void test_compat_mismatch_is_invisible() {
    const std::string root = tmproot("compat");
    {
        q27::PrefixCache pc;
        CHECK(pc.init(cfg_for(root), COMPAT_A));
        CHECK(pc.write(seq(64), 32, "g", 1, "k", 1));
        CHECK(pc.size() == 1);
    }
    q27::PrefixCache other;                    // different model / KV format
    CHECK(other.init(cfg_for(root), COMPAT_B));
    CHECK(other.size() == 0);                  // not indexed at all
    q27::PrefixCache::Entry e;
    CHECK(!other.find(seq(100), &e));
}

static void test_truncated_file_is_not_indexed() {
    const std::string root = tmproot("trunc");
    q27::PrefixCache pc;
    CHECK(pc.init(cfg_for(root), COMPAT_A));
    const std::string big(4096, 'x');
    CHECK(pc.write(seq(64), 32, big.data(), big.size(), big.data(), big.size()));
    q27::PrefixCache::Entry e;
    CHECK(pc.find(seq(100), &e));
    CHECK(truncate(e.path.c_str(), (off_t)(sizeof(q27::PfxHdr) + 32 * 4 + 100)) == 0);

    q27::PrefixCache pc2;
    CHECK(pc2.init(cfg_for(root), COMPAT_A));
    CHECK(pc2.size() == 0); // size != header's declared payload -> skipped
}

static void test_rescan_survives_restart() {
    const std::string root = tmproot("restart");
    {
        q27::PrefixCache pc;
        CHECK(pc.init(cfg_for(root), COMPAT_A));
        CHECK(pc.write(seq(64), 32, "gdn", 3, "kvkv", 4));
    }
    q27::PrefixCache fresh; // a new process pointed at the same directory
    CHECK(fresh.init(cfg_for(root), COMPAT_A));
    CHECK(fresh.size() == 1);
    q27::PrefixCache::Entry e;
    CHECK(fresh.find(seq(100), &e));
    CHECK(e.L == 32);
    char buf[7] = {0};
    CHECK(fresh.read_state(e, buf, 7));
    CHECK(std::string(buf, 7) == "gdnkvkv");
}

static void test_eviction_respects_budget() {
    const std::string root = tmproot("evict");
    q27::PrefixCache pc;
    q27::PrefixCacheCfg c = cfg_for(root);
    c.max_bytes = 4096; // tiny budget, forces eviction
    CHECK(pc.init(c, COMPAT_A));
    const std::string blob(1500, 'z');
    for (int i = 0; i < 6; i++) {
        std::vector<int> t = seq(64, 1000 + i * 1000);
        CHECK(pc.write(t, 32, blob.data(), blob.size(), blob.data(), blob.size()));
    }
    CHECK(pc.bytes() <= c.max_bytes);
    CHECK(pc.size() >= 1);
}

// P16b shared cut (2026-09-08): Claude Code sessions agree on the system block
// up to a per-session gitStatus tail (five sessions shared exactly 22460 of a
// 22544-22578-token block), so an entry cut at sys_len is hit by nobody. The
// cut has to land at the longest prefix an indexed entry shares with THIS
// prompt; this is the primitive that finds it.
static void test_shared_prefix_across_sessions() {
    const std::string root = tmproot("shared");
    q27::PrefixCache pc;
    CHECK(pc.init(cfg_for(root, /*min_tokens=*/16), COMPAT_A));
    CHECK(pc.shared_prefix(seq(100), 60) == 0);        // empty cache: nothing to share
    std::vector<int> s1 = seq(100); s1[50] = 777;      // session 1: tail differs from 50
    CHECK(pc.write(s1, 64, "g", 1, "k", 1));           // its system entry, cut at 64
    std::vector<int> s2 = seq(100); s2[50] = 888;      // session 2: same block, own tail
    CHECK(pc.shared_prefix(s2, 60) == 50);             // agrees through token 49
    CHECK(pc.shared_prefix(s2, 40) == 40);             // capped at upto
    CHECK(pc.shared_prefix(s2, 200) == 50);            // upto past the prompt/entry is fine
    CHECK(pc.shared_prefix(seq(100, 500000), 60) == 0); // foreign prompt shares nothing
    std::vector<int> s3 = seq(100); s3[58] = 999;      // an entry that agrees further wins
    CHECK(pc.write(s3, 64, "g", 1, "k", 1));
    std::vector<int> s4 = seq(100); s4[58] = 1111;
    CHECK(pc.shared_prefix(s4, 60) == 58);
    CHECK(pc.write(seq(100, 300000), 8, "g", 1, "k", 1)); // below min_tokens: ignored
    CHECK(pc.shared_prefix(seq(100, 300000), 60) == 0);
    q27::PrefixCache off;                               // disabled cache answers 0
    CHECK(off.shared_prefix(s4, 60) == 0);
}

static void test_bad_root_disables() {
    q27::PrefixCache pc;
    q27::PrefixCacheCfg c = cfg_for("/proc/definitely/not/writable/q27");
    CHECK(!pc.init(c, COMPAT_A));
    CHECK(!pc.enabled());
    q27::PrefixCache::Entry e;
    CHECK(!pc.find(seq(100), &e)); // disabled cache answers, never crashes
}

int main() {
    test_roundtrip_write_find_read();
    test_divergent_prompt_misses();
    test_exact_length_prompt_misses();
    test_longest_prefix_wins();
    test_below_min_tokens_never_matches();
    test_forged_key_collision_is_refused();
    test_compat_mismatch_is_invisible();
    test_truncated_file_is_not_indexed();
    test_rescan_survives_restart();
    test_eviction_respects_budget();
    test_shared_prefix_across_sessions();
    test_bad_root_disables();
    if (failures) { fprintf(stderr, "%d FAILURE(S)\n", failures); return 1; }
    fprintf(stderr, "all prefix-cache tests passed\n");
    return 0;
}
