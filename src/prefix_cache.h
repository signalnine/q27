#pragma once
// P16: persistent stable-prefix cache -- a disk tier under the P8 stable
// snapshot and the P9 checkpoint ring (docs/plans/2026-07-24-persistent-prefix-cache.md).
//
// Both existing tiers die with the process, so a server restart and every new
// conversation that shares Claude Code's 20-25K-token system block pay a full
// cold prefill (measured 2026-07-24 on a 5090: ~3.4K t/s, so ~7.3 s at 25K).
// This tier keeps the state that P8 already identifies as stable -- everything
// before chatml_prompt's stable_off -- in a file, keyed on the token prefix
// itself, so a restart or a fresh conversation restores instead of recomputing.
//
// One file per entry:  {root}/{key16}-{L}.q27pc
//     header   PfxHdr, fixed 64 B
//     tokens   L int32   <- THE verification payload
//     state    gdn_bytes of recurrent state, then kv_bytes of attention+MTP rows
//
// The filename key is a hash and narrows the candidate set, nothing more.
// Every load compares the stored token vector against the request prefix
// element by element before a single byte of state is read -- the same rule
// ckpt_best() already applies in RAM (engine.cuh). A hash-keyed restore that
// skips this silently continues ANOTHER conversation's state; that is the bug
// CachyLLama shipped on its SSD tier and patched on 2026-07-23 (outside PR
// fix/ssd-cache-prefix-verification). The disk tier inherits the strict rule,
// not the cheap one.
//
// Host-only (no CUDA here) so it unit-tests on CPU; the device-side copies
// live in Engine::pfx_export / pfx_import.
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace q27 {

inline uint64_t pfx_fnv1a64(const void* p, size_t n, uint64_t h = 1469598103934665603ULL) {
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Everything that has to match for a stored blob to be loadable: the model
// bytes, the buffer geometry, and the KV format. A mismatch means the entry is
// not consulted at all -- there is no partial or coerced load.
inline uint64_t pfx_compat_hash(const std::string& model_path, uint64_t model_bytes, int n_layer,
                                int n_kv, int head_dim, int gdn_heads, int gdn_dim, int gdn_ch,
                                int kv_kind, int format_version) {
    uint64_t h = pfx_fnv1a64(model_path.data(), model_path.size());
    const int64_t v[] = {(int64_t)model_bytes, n_layer,  n_kv,    head_dim,      gdn_heads,
                         gdn_dim,              gdn_ch,   kv_kind, format_version};
    return pfx_fnv1a64(v, sizeof v, h);
}

#pragma pack(push, 1)
struct PfxHdr {
    uint32_t magic;     // 'Q27P'
    uint32_t version;   // PFX_VERSION
    uint64_t compat;    // pfx_compat_hash
    uint64_t tok_hash;  // pfx_fnv1a64 over the L tokens
    int32_t L;
    int32_t pad;
    uint64_t gdn_bytes;
    uint64_t kv_bytes;
    uint64_t reserved[2];
};
#pragma pack(pop)
static_assert(sizeof(PfxHdr) == 64, "PfxHdr must stay 64 bytes");

static constexpr uint32_t PFX_MAGIC = 0x50373251;  // 'Q27P' little-endian
static constexpr uint32_t PFX_VERSION = 1;

struct PrefixCacheCfg {
    std::string root;
    size_t max_bytes = 20ull << 30;  // eviction budget
    int min_tokens = 4096;           // below this, prefill is cheap enough
    int max_tokens = 32768;          // caps the staging buffer the engine pins
    int step_tokens = 8192;          // growth required before re-persisting a chain
};

class PrefixCache {
  public:
    struct Entry {
        std::string path;
        int L = 0;
        uint64_t key = 0;
        size_t bytes = 0;
        long mtime = 0;
    };

    // Creates the directory if needed and indexes what is already there.
    // Returns false (disabled) if the directory cannot be used.
    bool init(const PrefixCacheCfg& cfg, uint64_t compat) {
        cfg_ = cfg;
        compat_ = compat;
        if (cfg.root.empty()) return false;
        if (mkdir(cfg.root.c_str(), 0700) != 0 && errno != EEXIST) {
            fprintf(stderr, "prefix-cache: cannot create %s (%s) -- disabled\n", cfg.root.c_str(),
                    strerror(errno));
            return false;
        }
        struct stat st;
        if (stat(cfg.root.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            fprintf(stderr, "prefix-cache: %s is not a directory -- disabled\n", cfg.root.c_str());
            return false;
        }
        enabled_ = true;
        rescan();
        prefetch_recent();
        return true;
    }

    // Boot prefetch: pull the most recently used entries into the page cache
    // while the server is still uploading weights (~10-40 s of free cover). A
    // cold read of a 1.09 GB entry measured 681 ms vs 206 ms warm, and that
    // 475 ms lands on the FIRST request after a restart -- exactly the request
    // this feature exists to make fast.
    //
    // This READS the file rather than calling posix_fadvise(WILLNEED). MEASURED
    // 2026-07-24: fadvise did nothing for a 1 GB entry (restore still took the
    // full 681 ms cold) -- it is advisory, and the kernel declines a readahead
    // request that size. An explicit chunked read is the only way to actually
    // guarantee the pages. Runs on a detached thread; if the entry is never
    // used the cost is background I/O the page cache would drop anyway.
    void prefetch_recent(int max_entries = 2) {
        std::vector<Entry> es;
        {
            std::lock_guard<std::mutex> lk(m_);
            es = index_;
        }
        std::sort(es.begin(), es.end(),
                  [](const Entry& a, const Entry& b) { return a.mtime > b.mtime; });
        if (es.size() > (size_t)max_entries) es.resize((size_t)max_entries);
        if (es.empty()) return;
        size_t total = 0;
        for (const auto& e : es) total += e.bytes;
        fprintf(stderr, "prefix-cache: warming %.2f GB of recent entries in the background\n",
                total / 1e9);
        std::thread([es] {
            std::vector<char> buf(8u << 20);
            for (const auto& e : es) {
                int fd = ::open(e.path.c_str(), O_RDONLY);
                if (fd < 0) continue;
                while (::read(fd, buf.data(), buf.size()) > 0) {}
                ::close(fd);
            }
        }).detach();
    }

    bool enabled() const { return enabled_; }
    const PrefixCacheCfg& cfg() const { return cfg_; }
    size_t size() const {
        std::lock_guard<std::mutex> lk(m_);
        return index_.size();
    }
    size_t bytes() const {
        std::lock_guard<std::mutex> lk(m_);
        size_t t = 0;
        for (const auto& e : index_) t += e.bytes;
        return t;
    }

    // Longest indexed entry that is an EXACT prefix of `prompt` and strictly
    // shorter than it (L <= NP-1: a full-length match would leave nothing to
    // decode from, mirroring the snapshot predicate in generate()). The token
    // vector is read back from disk and compared element by element; the key
    // hash only picks candidates.
    bool find(const std::vector<int>& prompt, Entry* out) const {
        if (!enabled_ || prompt.empty()) return false;
        std::vector<Entry> cands;
        {
            std::lock_guard<std::mutex> lk(m_);
            cands = index_;
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Entry& a, const Entry& b) { return a.L > b.L; });
        for (const auto& e : cands) {
            if (e.L < cfg_.min_tokens) continue;
            if ((size_t)e.L + 1 > prompt.size()) continue;  // needs >=1 token to decode from
            if (pfx_fnv1a64(prompt.data(), (size_t)e.L * sizeof(int)) != e.key) continue;
            if (!verify(e, prompt)) {
                fprintf(stderr, "prefix-cache: key hit but TOKENS DIFFER for %s -- ignoring\n",
                        e.path.c_str());
                continue;
            }
            *out = e;
            return true;
        }
        return false;
    }

    // P16b shared cut: the longest prefix of prompt[0,upto) that some indexed
    // entry's stored tokens also begin with (0 when nothing qualifies).
    // Measured 2026-09-08 on Claude Code traffic: five sessions agreed on
    // exactly 22460 tokens of a 22544-22578-token system block (the per-repo
    // gitStatus tail differs), so an entry cut at sys_len was hit by nobody and
    // every first turn wrote its own 0.94 GB entry. Cut at the shared length
    // and the next session restores it. Reads token vectors only (never the
    // state), in two stages so a root full of unrelated entries costs 1 KB
    // per entry: a 256-token head first, the full vector only when the head
    // matches completely and the entry could still beat the current best.
    // The engine calls this once per cold prefill that carries a system
    // block. I/O runs OUTSIDE the index lock on purpose (a persist or an
    // eviction must not wait on it); an entry evicted between the index copy
    // and the open just fails the open and is skipped.
    int shared_prefix(const std::vector<int>& prompt, int upto) const {
        if (!enabled_ || upto <= 0 || prompt.empty()) return 0;
        upto = std::min(upto, (int)prompt.size());
        std::vector<Entry> cands;
        {
            std::lock_guard<std::mutex> lk(m_);
            cands = index_;
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Entry& a, const Entry& b) { return a.L > b.L; });
        constexpr int HEAD = 256;
        int best = 0;
        std::vector<int> toks;
        auto lcp_from = [&](int from, int n) {
            int l = from;
            while (l < n && toks[(size_t)l] == prompt[(size_t)l]) l++;
            return l;
        };
        for (const auto& e : cands) {
            if (e.L < cfg_.min_tokens) continue;
            const int n = std::min(e.L, upto);
            if (n <= best) continue;  // cannot beat the current best
            int fd = ::open(e.path.c_str(), O_RDONLY);
            if (fd < 0) continue;
            const int h = std::min(n, HEAD);
            toks.resize((size_t)h);
            bool ok = read_full(fd, toks.data(), (size_t)h * sizeof(int), sizeof(PfxHdr));
            int l = ok ? lcp_from(0, h) : 0;
            if (ok && l == h && n > h) {  // head matched: read the rest
                toks.resize((size_t)n);
                ok = read_full(fd, toks.data() + h, (size_t)(n - h) * sizeof(int),
                               sizeof(PfxHdr) + (off_t)h * sizeof(int));
                if (ok) l = lcp_from(h, n);
            }
            ::close(fd);
            if (!ok) continue;
            best = std::max(best, l);
        }
        return best;
    }

    // Read the state region (gdn then kv, contiguous) into `dst`.
    bool read_state(const Entry& e, void* dst, size_t dst_n) const {
        int fd = ::open(e.path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        PfxHdr h{};
        bool ok = ::pread(fd, &h, sizeof h, 0) == (ssize_t)sizeof h && h.magic == PFX_MAGIC &&
                  h.version == PFX_VERSION && h.compat == compat_ && h.L == e.L;
        if (ok && h.gdn_bytes + h.kv_bytes != dst_n) {
            fprintf(stderr, "prefix-cache: state size %zu != expected %zu for %s\n",
                    (size_t)(h.gdn_bytes + h.kv_bytes), dst_n, e.path.c_str());
            ok = false;
        }
        if (ok) {
            const off_t off = (off_t)sizeof(PfxHdr) + (off_t)e.L * 4;
            ok = read_full(fd, dst, dst_n, off);
        }
        ::close(fd);
        return ok;
    }

    // Indexed OR in flight (reserved by a writer that has not published yet):
    // both mean "do not export this key again".
    bool has(const std::vector<int>& toks, int L) const {
        if (!enabled_ || L <= 0 || (size_t)L > toks.size()) return false;
        const uint64_t k = pfx_fnv1a64(toks.data(), (size_t)L * sizeof(int));
        std::lock_guard<std::mutex> lk(m_);
        for (const auto& e : index_)
            if (e.L == L && e.key == k) return true;
        for (const auto& f : inflight_)
            if (f.second == L && f.first == k) return true;
        return false;
    }

    // Claim (key, L) for one writer. All engines of a server share one cache
    // (server.cu) while the export/writer state is per engine, so two cold
    // slots choosing the same cut could both pass has() and both write the
    // same file (gpt-6-astra review 2026-09-08, P1: with a shared .tmp path
    // and O_TRUNC one writer could publish the other's half-written state).
    // Call before the D2H export; write() releases the claim when it is done
    // (published or failed). Returns false when the key is indexed or already
    // claimed -- the caller skips this boundary, exactly as for a has() hit.
    bool reserve(const std::vector<int>& toks, int L) {
        if (!enabled_ || L <= 0 || (size_t)L > toks.size()) return false;
        const uint64_t k = pfx_fnv1a64(toks.data(), (size_t)L * sizeof(int));
        std::lock_guard<std::mutex> lk(m_);
        for (const auto& e : index_)
            if (e.L == L && e.key == k) return false;
        for (const auto& f : inflight_)
            if (f.second == L && f.first == k) return false;
        inflight_.emplace_back(k, L);
        return true;
    }

    // Atomic write: full content to a per-writer .tmp in the same directory,
    // then rename + index update under the lock (so an eviction can never
    // interleave between a publication and its indexing: review P2). A torn
    // write can therefore never be indexed. Releases the reserve() claim.
    bool write(const std::vector<int>& toks, int L, const void* gdn, size_t gdn_n, const void* kv,
               size_t kv_n) {
        if (!enabled_ || L <= 0 || (size_t)L > toks.size()) return false;
        const uint64_t key = pfx_fnv1a64(toks.data(), (size_t)L * sizeof(int));
        char name[64];
        snprintf(name, sizeof name, "%016llx-%d.q27pc", (unsigned long long)key, L);
        const std::string final_path = cfg_.root + "/" + name;
        char tsuf[48];
        snprintf(tsuf, sizeof tsuf, ".tmp.%ld.%u", (long)::getpid(), tmp_seq_.fetch_add(1));
        const std::string tmp_path = final_path + tsuf;
        struct Release {  // the claim goes away on every exit path
            PrefixCache* c; uint64_t k; int L;
            ~Release() { c->release(k, L); }
        } rel{this, key, L};
        int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) {
            fprintf(stderr, "prefix-cache: cannot write %s (%s)\n", tmp_path.c_str(),
                    strerror(errno));
            return false;
        }
        PfxHdr h{};
        h.magic = PFX_MAGIC;
        h.version = PFX_VERSION;
        h.compat = compat_;
        h.tok_hash = key;
        h.L = L;
        h.gdn_bytes = gdn_n;
        h.kv_bytes = kv_n;
        bool ok = write_full(fd, &h, sizeof h) &&
                  write_full(fd, toks.data(), (size_t)L * sizeof(int)) &&
                  write_full(fd, gdn, gdn_n) && write_full(fd, kv, kv_n);
        if (ok) ok = ::fsync(fd) == 0;
        ::close(fd);
        if (!ok) {
            ::unlink(tmp_path.c_str());
            fprintf(stderr, "prefix-cache: write failed for %s\n", final_path.c_str());
            return false;
        }
        Entry e;
        e.path = final_path;
        e.L = L;
        e.key = key;
        e.bytes = sizeof(PfxHdr) + (size_t)L * 4 + gdn_n + kv_n;
        e.mtime = (long)time(nullptr);
        {
            std::lock_guard<std::mutex> lk(m_);
            if (::rename(tmp_path.c_str(), final_path.c_str()) != 0) {
                ::unlink(tmp_path.c_str());
                fprintf(stderr, "prefix-cache: publish failed for %s (%s)\n", final_path.c_str(),
                        strerror(errno));
                return false;
            }
            index_.erase(std::remove_if(index_.begin(), index_.end(),
                                        [&](const Entry& x) { return x.path == final_path; }),
                         index_.end());
            index_.push_back(e);
        }
        evict_to_budget();
        return true;
    }

    // LRU by mtime down to the byte budget. Whole entries only. The unlink
    // happens under the same lock as the index removal so it can never hit a
    // replacement that a writer published to the same path in between.
    void evict_to_budget() {
        if (!enabled_) return;
        std::vector<Entry> doomed;
        {
            std::lock_guard<std::mutex> lk(m_);
            size_t total = 0;
            for (const auto& e : index_) total += e.bytes;
            if (total <= cfg_.max_bytes) return;
            std::sort(index_.begin(), index_.end(),
                      [](const Entry& a, const Entry& b) { return a.mtime < b.mtime; });
            while (total > cfg_.max_bytes && !index_.empty()) {
                total -= index_.front().bytes;
                doomed.push_back(index_.front());
                ::unlink(index_.front().path.c_str());
                index_.erase(index_.begin());
            }
        }
        for (const auto& e : doomed)
            fprintf(stderr, "prefix-cache: evicted %s (%.2f GB budget)\n", e.path.c_str(),
                    cfg_.max_bytes / 1e9);
    }

    // Drop a reserve() claim without writing (the caller could not stage the
    // export). Without this the claim outlives the process and has() keeps
    // reporting the key as present, so the boundary is never retried
    // (gpt-6-astra advisory 2026-09-08 (p), P2).
    void release(const std::vector<int>& toks, int L) {
        if (L <= 0 || (size_t)L > toks.size()) return;
        release(pfx_fnv1a64(toks.data(), (size_t)L * sizeof(int)), L);
    }

    void release(uint64_t key, int L) {
        std::lock_guard<std::mutex> lk(m_);
        inflight_.erase(std::remove_if(inflight_.begin(), inflight_.end(),
                                       [&](const std::pair<uint64_t, int>& f) {
                                           return f.first == key && f.second == L;
                                       }),
                        inflight_.end());
    }

    // Index every well-formed, compat-matching entry in the directory.
    // Files that fail the header check are left alone (another model's cache
    // may legitimately share the directory), with one exception: a leftover
    // .tmp is this cache's own crashed or killed write (the writer runs in the
    // background, so SIGTERM mid-write leaves one). It can never be indexed --
    // only the rename publishes an entry -- so sweeping it on startup keeps a
    // killed write from leaking gigabytes forever.
    void rescan() {
        std::vector<Entry> found;
        DIR* d = ::opendir(cfg_.root.c_str());
        if (!d) return;
        while (dirent* de = ::readdir(d)) {
            const std::string n = de->d_name;
            if (n.find(".q27pc.tmp") != std::string::npos) {  // ".q27pc.tmp.<pid>.<n>"
                const std::string stale = cfg_.root + "/" + n;
                if (::unlink(stale.c_str()) == 0)
                    fprintf(stderr, "prefix-cache: swept incomplete write %s\n", stale.c_str());
                continue;
            }
            if (n.size() < 7 || n.compare(n.size() - 6, 6, ".q27pc") != 0) continue;
            const std::string path = cfg_.root + "/" + n;
            int fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) continue;
            PfxHdr h{};
            struct stat st {};
            const bool ok = ::pread(fd, &h, sizeof h, 0) == (ssize_t)sizeof h &&
                            h.magic == PFX_MAGIC && h.version == PFX_VERSION &&
                            h.compat == compat_ && h.L > 0 && ::fstat(fd, &st) == 0 &&
                            (uint64_t)st.st_size ==
                                sizeof(PfxHdr) + (uint64_t)h.L * 4 + h.gdn_bytes + h.kv_bytes;
            ::close(fd);
            if (!ok) continue;
            Entry e;
            e.path = path;
            e.L = h.L;
            e.key = h.tok_hash;
            e.bytes = (size_t)st.st_size;
            e.mtime = (long)st.st_mtime;
            found.push_back(e);
        }
        ::closedir(d);
        std::lock_guard<std::mutex> lk(m_);
        index_.swap(found);
    }

  private:
    // The rule: never trust the key. Read the stored tokens and compare.
    bool verify(const Entry& e, const std::vector<int>& prompt) const {
        int fd = ::open(e.path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        std::vector<int> toks((size_t)e.L);
        const bool ok = read_full(fd, toks.data(), (size_t)e.L * sizeof(int), sizeof(PfxHdr));
        ::close(fd);
        return ok && std::equal(toks.begin(), toks.end(), prompt.begin());
    }

    static bool read_full(int fd, void* dst, size_t n, off_t off) {
        char* p = (char*)dst;
        while (n > 0) {
            const ssize_t r = ::pread(fd, p, n > (1u << 30) ? (1u << 30) : n, off);
            if (r <= 0) return false;
            p += r;
            off += r;
            n -= (size_t)r;
        }
        return true;
    }
    static bool write_full(int fd, const void* src, size_t n) {
        const char* p = (const char*)src;
        while (n > 0) {
            const ssize_t w = ::write(fd, p, n > (1u << 30) ? (1u << 30) : n);
            if (w <= 0) return false;
            p += w;
            n -= (size_t)w;
        }
        return true;
    }

    PrefixCacheCfg cfg_;
    uint64_t compat_ = 0;
    bool enabled_ = false;
    mutable std::mutex m_;
    std::vector<Entry> index_;
    std::vector<std::pair<uint64_t, int>> inflight_;  // reserve()d keys, under m_
    std::atomic<unsigned> tmp_seq_{0};                 // per-writer .tmp suffix
};

}  // namespace q27
