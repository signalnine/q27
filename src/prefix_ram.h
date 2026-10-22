#pragma once
// P16c: host-RAM tier between the VRAM tiers (P8 snapshot, P9 ring) and the
// P16 disk tier. Keeps recently used prefix blobs in PINNED host memory so a
// restore skips the file read entirely.
//
// SIZE THE LEVER BEFORE BELIEVING IN IT (measured 2026-07-24, 5090, 1.09 GB
// entry): a restore is read + import. Import (H2D from pinned) is **38 ms** and
// is the floor no tier can beat. The read is 681 ms cold and 206 ms once the
// page cache holds the file. So this tier removes ~206 ms from a restore that
// otherwise costs ~245 ms -- real, but it is the last 3% of an 8.15 s -> 0.68 s
// win the disk tier already delivered, and it costs GBs of pinned RAM. It is
// therefore OFF by default (`--prefix-cache-ram-gb 0`) and worth turning on
// mainly when the page cache is under pressure from other work, where the
// alternative is the 681 ms cold read on every hit rather than 206 ms.
//
// Fixed-size slots (one entry's worth each) rather than a general allocator:
// every blob is bounded by pfx_bytes(max_tokens), cudaMallocHost is slow enough
// that it must not land on a restore, and same-size slots make reuse trivial
// with no fragmentation.
//
// Verification is the same rule as everywhere else in P16: an entry is matched
// by comparing its FULL token vector against the request prefix, never by hash.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <vector>

namespace q27 {

class PrefixRam {
  public:
    struct Blob {
        std::vector<int> toks;  // published prefix (empty = slot in flight)
        char* p = nullptr;      // pinned
        size_t cap = 0;
        size_t bytes = 0;
        bool inflight = false;  // handed out by acquire(), not yet published/abandoned
        ~Blob() {
            if (p) cudaFreeHost(p);
        }
    };
    using BlobPtr = std::shared_ptr<Blob>;

    // slot_bytes = pfx_bytes(max_tokens); budget rounds DOWN to whole slots.
    void init(size_t budget_bytes, size_t slot_bytes) {
        slot_bytes_ = slot_bytes;
        max_slots_ = slot_bytes ? (int)(budget_bytes / slot_bytes) : 0;
        if (max_slots_ > 0)
            fprintf(stderr, "prefix-cache RAM tier: %d slot%s x %.2f GB\n", max_slots_,
                    max_slots_ == 1 ? "" : "s", slot_bytes / 1e9);
        else if (budget_bytes)
            fprintf(stderr,
                    "prefix-cache RAM tier: budget %.2f GB < one %.2f GB slot -- disabled\n",
                    budget_bytes / 1e9, slot_bytes / 1e9);
    }
    bool enabled() const { return max_slots_ > 0; }

    // Pin every slot at boot. acquire() would otherwise call cudaMallocHost on
    // the first restore that lands in a fresh slot -- measured 133-141 ms at
    // 1.3 GB during P16c, on the request critical path, for a size that is
    // known before the first request arrives. Slots go in with empty toks, so
    // find() skips them and acquire() treats them as evictable.
    void prealloc() {
        if (!enabled()) return;
        auto t0 = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lk(m_);
        while ((int)lru_.size() < max_slots_) {
            BlobPtr b = std::make_shared<Blob>();
            if (cudaMallocHost((void**)&b->p, slot_bytes_) != cudaSuccess) {
                b->p = nullptr;
                fprintf(stderr,
                        "prefix-cache RAM tier: pinned alloc %.2f GB FAILED at boot -- "
                        "tier capped at %d slot%s\n",
                        slot_bytes_ / 1e9, (int)lru_.size(), lru_.size() == 1 ? "" : "s");
                max_slots_ = (int)lru_.size();
                break;
            }
            b->cap = slot_bytes_;
            lru_.push_back(b);
        }
        if (max_slots_ > 0)
            fprintf(stderr, "prefix-cache RAM tier: %d slot%s pinned at boot (%.0f ms)\n",
                    max_slots_, max_slots_ == 1 ? "" : "s",
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count());
    }

    // Longest published blob that is an exact prefix of `prompt` and leaves at
    // least one token to decode from. Returned by shared_ptr, so eviction by
    // another thread cannot free it while a restore is copying out of it.
    BlobPtr find(const std::vector<int>& prompt) {
        if (!enabled() || prompt.empty()) return nullptr;
        std::lock_guard<std::mutex> lk(m_);
        BlobPtr best;
        for (auto& b : lru_) {
            if (b->toks.empty() || b->toks.size() + 1 > prompt.size()) continue;
            if (best && b->toks.size() <= best->toks.size()) continue;
            if (std::equal(b->toks.begin(), b->toks.end(), prompt.begin())) best = b;
        }
        if (best) touch(best);
        return best;
    }

    // A slot to fill. Reuses an evictable allocation (refcount 1 = only the LRU
    // holds it) before allocating, so the pinned malloc happens at most
    // max_slots_ times over the server's life -- and zero times once prealloc()
    // has run, which is the serving path.
    //
    // In-flight slots count against max_slots_ (2026-10-04): acquire() takes
    // the slot out of lru_, so counting lru_ alone let a second concurrent
    // acquire() see a free slot and cudaMallocHost one more -- over the pinned
    // budget, on the serving path -- and publish() never trimmed it back.
    BlobPtr acquire(size_t need) {
        if (!enabled() || need > slot_bytes_) return nullptr;
        std::lock_guard<std::mutex> lk(m_);
        if ((int)lru_.size() + inflight_ >= max_slots_) {
            for (auto it = lru_.begin(); it != lru_.end(); ++it) {
                if (it->use_count() > 1) continue;  // someone is still reading it
                BlobPtr b = *it;
                lru_.erase(it);
                b->toks.clear();
                b->bytes = need;
                b->inflight = true;
                inflight_++;
                return b;
            }
            return nullptr;  // every slot busy: caller falls back to its own staging
        }
        BlobPtr b = std::make_shared<Blob>();
        if (cudaMallocHost((void**)&b->p, slot_bytes_) != cudaSuccess) {
            fprintf(stderr, "prefix-cache RAM tier: pinned alloc %.2f GB FAILED -- tier off\n",
                    slot_bytes_ / 1e9);
            b->p = nullptr;
            max_slots_ = 0;
            return nullptr;
        }
        b->cap = slot_bytes_;
        b->bytes = need;
        b->inflight = true;
        inflight_++;
        return b;
    }

    // Give back an acquired slot that was never filled (e.g. the disk read
    // failed). It returns to the LRU empty, oldest position, so its pinned
    // allocation is reused instead of freed here and re-allocated later.
    void abandon(const BlobPtr& b) {
        if (!b) return;
        std::lock_guard<std::mutex> lk(m_);
        if (b->inflight) { b->inflight = false; inflight_--; }
        b->toks.clear();
        lru_.erase(std::remove(lru_.begin(), lru_.end(), b), lru_.end());
        lru_.insert(lru_.begin(), b);
    }

    // Publish a filled slot. Replaces any entry with the same prefix length and
    // content (a re-persist of the same prefix).
    void publish(const BlobPtr& b, const std::vector<int>& prompt, int L) {
        if (!b || !enabled() || L <= 0 || (size_t)L > prompt.size()) return;
        std::lock_guard<std::mutex> lk(m_);
        if (b->inflight) { b->inflight = false; inflight_--; }
        b->toks.assign(prompt.begin(), prompt.begin() + L);
        // a duplicate (two engines restored the same entry at once) keeps
        // its pinned allocation: it goes to the LRU FRONT with no tokens, so
        // acquire() reuses it next. Erasing it freed the slot for good and
        // the next acquire() paid a cudaMallocHost on the serving path (bug
        // hunt 2026-10-22).
        std::vector<BlobPtr> dups;
        for (auto& x : lru_)
            if (x != b && x->toks.size() == b->toks.size() &&
                std::equal(x->toks.begin(), x->toks.end(), b->toks.begin()))
                dups.push_back(x);
        for (auto& x : dups) {
            lru_.erase(std::remove(lru_.begin(), lru_.end(), x), lru_.end());
            x->toks.clear();
            lru_.insert(lru_.begin(), x);
        }
        lru_.erase(std::remove(lru_.begin(), lru_.end(), b), lru_.end());
        lru_.push_back(b);  // back = most recently used
    }

    int count() const {
        std::lock_guard<std::mutex> lk(m_);
        return (int)lru_.size();
    }

  private:
    void touch(const BlobPtr& b) {  // caller holds m_
        auto it = std::find(lru_.begin(), lru_.end(), b);
        if (it != lru_.end()) {
            lru_.erase(it);
            lru_.push_back(b);
        }
    }
    mutable std::mutex m_;
    std::vector<BlobPtr> lru_;  // front = least recently used
    size_t slot_bytes_ = 0;
    int inflight_ = 0;  // acquired slots not yet published/abandoned (guarded by m_)
    int max_slots_ = 0;
};

}  // namespace q27
