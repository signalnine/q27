// Incremental KV entitlements (issue #42 step 2, plan
// docs/plans/2026-09-10-incremental-kv.md): the banker's-algorithm safety
// check that makes reserve-as-you-go deadlock-free without preemption.
//
// Each ACTIVE generation holds some pages now and has declared a maximum it
// may grow to (prompt + max_tokens + round reserve). A grant -- an admission
// or a mid-decode extension -- is made only if the resulting state is SAFE:
// some completion order exists in which every active generation can reach
// its maximum, where a generation that finishes returns its held pages to
// the idle pool (they become a reclaimable conversation cache). In a safe
// state the generation with the smallest remaining need can always grow, so
// at least one generation always makes progress and parked ones only wait
// for others to finish, never for each other.
//
// Pages here are PER SIDE: the K and V sides of the pool carry identical page
// counts for a given row count (one page per 64 rows per layer pair on each
// side), so the two sides are one resource and the caller passes the
// smaller side's availability.
//
// Pure host code, no CUDA: tools/test_kv_bank.cpp drives it directly.
#pragma once
#include <algorithm>
#include <vector>

namespace q27 {

// Pages per side the pool maps for `rows` attention rows: 16 attention
// pairs at ceil(rows/64) plus the MTP pair at ceil((rows+1)/64). MIRRORS
// server.cu's pages_for and Engine::kv_entitle.
inline long kv_pages_for_rows(int rows) {
    if (rows <= 0) return 0;
    return 16L * ((rows + 63) / 64) + (long)((rows + 1 + 63) / 64);
}

struct KvClaim {
    long held; // pages this generation's lineage maps now
    long max;  // pages it may hold at its declared maximum (>= held)
};

// true iff the state is safe. `avail` = pages obtainable without touching an
// active generation: free pages plus pages held by idle lineages (scavenged
// LRU-first when a grant actually needs them). A single resource, so ordering
// claims by remaining need is an optimal completion order: if the claim with
// the smallest need cannot finish now, no claim can.
inline bool kv_bank_safe(std::vector<KvClaim> claims, long avail) {
    if (avail < 0) return false;
    std::sort(claims.begin(), claims.end(), [](const KvClaim& a, const KvClaim& b) {
        return (a.max - a.held) < (b.max - b.held);
    });
    for (const KvClaim& c : claims) {
        const long need = c.max > c.held ? c.max - c.held : 0;
        if (need > avail) return false;
        avail += c.held;
    }
    return true;
}

} // namespace q27
