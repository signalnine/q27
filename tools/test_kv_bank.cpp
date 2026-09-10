// CPU unit test for src/kv_bank.h (incremental KV entitlements, issue #42
// step 2): the banker's safety check and the page arithmetic it mirrors.
#include "kv_bank.h"

#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(cond, name)                                          \
    do {                                                           \
        if (cond) printf("  %-66s PASS\n", name);                  \
        else { printf("  %-66s FAIL\n", name); fails++; }          \
    } while (0)

int main() {
    using q27::KvClaim;
    using q27::kv_bank_safe;
    using q27::kv_pages_for_rows;

    // page arithmetic: 16 attention pairs at ceil(r/64) + MTP at ceil((r+1)/64)
    CHECK(kv_pages_for_rows(0) == 0, "pages(0) = 0");
    CHECK(kv_pages_for_rows(64) == 16 * 1 + 2, "pages(64): MTP row 65 spills a page");
    CHECK(kv_pages_for_rows(63) == 16 * 1 + 1, "pages(63): everything in one page");
    CHECK(kv_pages_for_rows(4096) == 16 * 64 + 65, "pages(4096) = 1089");

    // Units below are pages. A 4-slot 5090 pool holds ~71K pages per side;
    // use round numbers.
    const long POOL = 70000;
    const long P30K = kv_pages_for_rows(30000), P34K = kv_pages_for_rows(34096);
    const long P94K = kv_pages_for_rows(94000);

    // The reason for step 2: four Claude Code turns, 30K prompt + 64K
    // max_tokens each. Up front they need 4 x 94K rows > the pool; reserved
    // incrementally (prompt + 4K) the state is safe -- any one of them can
    // finish, and each finisher frees enough for the next.
    {
        std::vector<KvClaim> c(4, KvClaim{P34K, P94K});
        const long avail = POOL - 4 * P34K;
        CHECK(kv_bank_safe(c, avail), "4 x (30K prompt + 64K max) incremental: safe");
        std::vector<KvClaim> up(4, KvClaim{P94K, P94K});
        CHECK(4 * P94K > POOL, "same four up front would not fit the pool at all");
    }
    // Unsafe: every claim needs more than what is free, even after any single
    // finisher -- admitting the last one could deadlock, so it must wait.
    {
        std::vector<KvClaim> c = {{20000, 60000}, {20000, 60000}, {20000, 60000}};
        CHECK(!kv_bank_safe(c, POOL - 60000), "3 x (need 40K) with 10K free: unsafe");
        c.pop_back();
        CHECK(!kv_bank_safe(c, 30000), "2 x (need 40K) with 30K free: nobody can finish -> unsafe");
        CHECK(kv_bank_safe(c, 40000), "2 x (need 40K) with 40K free: one finishes, frees 20K -> safe");
    }
    // Order independence: the check sorts by remaining need.
    {
        std::vector<KvClaim> a = {{10000, 50000}, {30000, 32000}};
        std::vector<KvClaim> b = {{30000, 32000}, {10000, 50000}};
        CHECK(kv_bank_safe(a, 10000) == kv_bank_safe(b, 10000), "claim order does not matter");
        CHECK(kv_bank_safe(a, 10000), "need-2K claim finishes first, 40K then covers the need-40K one");
        CHECK(!kv_bank_safe(a, 5000), "5K: the small one finishes but 35K < 40K -> unsafe");
        CHECK(!kv_bank_safe(a, 1000), "neither need fits in 1K: unsafe");
    }
    // Held above max (a lineage that kept rows from an earlier, longer turn)
    // counts as zero need, never negative.
    {
        std::vector<KvClaim> c = {{50000, 40000}, {1000, 20000}};
        CHECK(kv_bank_safe(c, 0), "over-held lineage needs nothing and frees its pages");
    }
    // Negative availability (a grant larger than free + idle) is never safe.
    {
        std::vector<KvClaim> c;
        CHECK(!kv_bank_safe(c, -1), "avail < 0: unsafe even with no claims");
        CHECK(kv_bank_safe(c, 0), "no claims, nothing borrowed: safe");
    }
    // The head-of-order property the no-deadlock argument rests on: in a safe
    // state, growing the smallest-need claim by any amount that fits keeps the
    // state safe.
    {
        std::vector<KvClaim> c = {{P30K, P94K}, {40000, 45000}};
        long avail = POOL - P30K - 40000;
        bool all = kv_bank_safe(c, avail);
        for (long d = 1; d <= 5000 && all; d += 499) {
            std::vector<KvClaim> g = c;
            g[1].held += d; // claim 1 has the smaller need (5000)
            all = kv_bank_safe(g, avail - d);
        }
        CHECK(all, "growing the smallest-need claim within its max stays safe");
    }
    printf(fails ? "test_kv_bank: %d FAILED\n" : "test_kv_bank: ALL PASS\n", fails);
    return fails ? 1 : 0;
}
