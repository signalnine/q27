# gpt-6-astra review: P16b shared cut (2026-09-08)

Reviewer: gpt-6-astra via codex exec (read-only sandbox, xhigh), static review of commits b01543a and 19d4178 (src/prefix_cache.h shared_prefix, engine.cuh pfx_sys_cut) against the surrounding cache/engine/server code. No builds or runs on the reviewer's side. Disposition of each item is recorded in BUILDLOG 2026-09-08 (i).

## Report (reviewer's words)

The two-stage comparison looks correct, and the measured 22,460 → 21,504 cut is consistent with the engine’s chunk policy. I would address the **P1 publication race before relying on this across slots**, and the **P2 promotion gap before treating the three-session bootstrap as a general guarantee**. The writer and eviction races below predate these commits; shared cuts increase their relevance by directing different sessions toward the same cache key. This was static inspection only: no builds, tests, or repository programs were run, and no files were changed.

1. **P1 — Concurrent writers can publish an incomplete state blob.**  
   **Evidence:** All slots share one cache ([server.cu:1172](/mnt/ai/projects/q27/src/server.cu:1172)), but `pfx_busy` and writer threads belong to individual engines. `has()` checks only indexed entries ([prefix_cache.h:278](/mnt/ai/projects/q27/src/prefix_cache.h:278)); it does not reserve an in-flight key. Every writer for the same `(hash, L)` opens the same `.tmp` with `O_TRUNC` ([prefix_cache.h:294](/mnt/ai/projects/q27/src/prefix_cache.h:294)), and publication is only locked when updating the index.

   Two cold slots can select the same shared cut and both pass `has()`. Writer A can finish writing GDN; writer B then truncates the file and writes its header/tokens; A resumes at its existing offset, writes KV, and renames the file while B remains unfinished. The published file can have a valid header, matching tokens, expected size, and a hole where GDN belongs. `read_state()` accepts that shape ([prefix_cache.h:263](/mnt/ai/projects/q27/src/prefix_cache.h:263)). B can also continue modifying the inode after publication.

   **Recommendation:** Use a unique temporary file per writer and coordinate publication per cache key. A shared in-flight reservation would additionally prevent duplicate exports and writes. The per-engine busy flag is insufficient.

2. **P2 — A shorter restore prevents learning a better system cut indefinitely.**  
   **Evidence:** Shared-prefix discovery requires `base == 0` ([engine.cuh:4956](/mnt/ai/projects/q27/src/engine.cuh:4956)), and system persistence independently requires it ([engine.cuh:4289](/mnt/ai/projects/q27/src/engine.cuh:4289)). Once an 8,192-token entry matches a new client version, subsequent sessions restore it even if that version consistently shares 22,460 tokens. Those requests never discover or persist the longer system prefix. Conversation entries may still accumulate, but their session-specific suffixes do not provide a reusable system entry.

   This applies to snapshot/checkpoint hits too. The BUILDLOG explicitly says the probe inserted a foreign request because otherwise a 4,096-token checkpoint prevented the cut logic from running ([BUILDLOG.md:15818](/mnt/ai/projects/q27/docs/BUILDLOG.md:15818)). The shorter entry can remain the effective ceiling until eviction or another source supplies a better entry.

   **Recommendation:** Permit system-prefix promotion when a restored request crosses a useful boundary beyond `base`. Give promotion an explicit policy instead of merely removing the guard: disk/RAM restores set `pfx_last_persist = base`, so the existing 8,192-token growth gate could still suppress worthwhile improvements.

3. **P2 — Eviction can unlink a replacement entry and leave `has()` permanently suppressing its repair.**  
   **Evidence:** Eviction removes entries under the mutex, then unlinks their paths after releasing it ([prefix_cache.h:339](/mnt/ai/projects/q27/src/prefix_cache.h:339)). Meanwhile, a writer can rename a replacement onto that same path and insert it into the index ([prefix_cache.h:316](/mnt/ai/projects/q27/src/prefix_cache.h:316)). A delayed eviction then deletes the replacement, leaving an indexed path that no longer exists. `has()` still returns true from its length/hash metadata.

   This is distinct from the benign case where `shared_prefix()` opens an entry that was simply evicted. Here, subsequent lookups miss while persistence of that exact entry is suppressed.

   **Recommendation:** Coordinate final-path publication, index updates, and unlinking, or use generation-specific filenames so eviction cannot delete a newer generation.

4. **P3 — The minimum-length check happens before rounding, allowing an unusable shared cut.**  
   **Evidence:** The engine accepts `shared >= min_tokens` ([engine.cuh:4959](/mnt/ai/projects/q27/src/engine.cuh:4959)), then persists at a preceding chunk boundary. `pfx_should_persist()` checks the resulting boundary against the minimum ([engine.cuh:4277](/mnt/ai/projects/q27/src/engine.cuh:4277)). With configurable `min_tokens = 4500` and `shared = 4600`, the engine selects the shared cut but reaches boundary 4096, which is rejected. It consequently loses the system entry that the original, longer cut could have produced.

   Production’s aligned minimum of 4096 avoids this case.

   **Recommendation:** Validate the actual eligible chunk boundary before replacing the original system cut.

5. **P3 — Several test assertions do not establish the bounds their comments claim.**  
   **Evidence:** The “capped at L” assertion uses a prompt that already diverges at token 700, before the entry ends at 1000 ([test_prefix_cache.cpp:249](/mnt/ai/projects/q27/tools/test_prefix_cache.cpp:249)). Returning 700 does not demonstrate the entry-length cap. Likewise, the `upto = 200` case diverges at 50, before either the 64-token entry or 100-token prompt ends ([test_prefix_cache.cpp:231](/mnt/ai/projects/q27/tools/test_prefix_cache.cpp:231)).

   The head-boundary assertions correctly exercise comparison results, but an implementation that unnecessarily reads every tail could pass them all. They do not establish the claimed I/O reduction.

   **Recommendation:** Add fully matching cases that stop specifically at entry length and prompt length, plus read instrumentation or fault injection for the skipped-tail and partial-read paths.

The remaining correctness checks did not reveal another defect:

- **Index snapshot and concurrent scans:** The index is copied **under** the mutex; file I/O happens outside it ([prefix_cache.h:220](/mnt/ai/projects/q27/src/prefix_cache.h:220)). Each call owns its candidates, token buffer, and `best`. An unlink before `open()` causes a skip; an unlink after `open()` leaves the descriptor readable. Concurrent insertion can make the heuristic temporarily stale. These are safe behaviors, subject to the publication defects above.
- **Two-stage boundaries:** The head covers `[0, h)`. Continuation starts at token `h`, with destination `toks.data() + h` and file offset `64 + 4h`; for `h = 256`, that is byte 1088 ([prefix_cache.h:241](/mnt/ai/projects/q27/src/prefix_cache.h:241)). Resizing preserves the head. Positive short reads are accumulated; EOF or errors discard the candidate. `EINTR` causes a conservative miss because `read_full()` does not retry it ([prefix_cache.h:418](/mnt/ai/projects/q27/src/prefix_cache.h:418)).
- **Cut semantics:** A nonaligned shared length intentionally rounds down: 22,460 becomes 21,504 with `PF_T = 1024`. `shared == sys_len` correctly retains the original cut. A conversation entry is valid evidence of token sharing: the engine exports its current state at the chosen boundary rather than attempting to truncate the source entry’s recurrent state.
- **Gates and RAM:** Cold requests reset `pfx_last_persist` before selecting the shorter cut, so an earlier larger value does not suppress it. An outstanding writer can still make the single cut opportunity fail `pfx_busy`, without retry. Below-minimum entries are excluded correctly. Omitting RAM entries affects learning when a divergent entry exists only in RAM—during disk publication, after disk eviction, or after a failed write. Exact RAM hits already restore earlier in tier resolution. Thus this omission costs opportunities, not state correctness; production currently disables that tier.
- **Server propagation:** `sys_len` is computed per request, assigned after slot claim, and captured by value for streaming ([server.cu:2758](/mnt/ai/projects/q27/src/server.cu:2758), [server.cu:2813](/mnt/ai/projects/q27/src/server.cu:2813), [server.cu:2974](/mnt/ai/projects/q27/src/server.cu:2974)). I found no stale request-length propagation here.

For **cost**, a budget-respecting 40 GB root containing entries of at least 0.4 GB has at most 100 entries. Entries rejected within the head require at most **100 KiB of requested token reads total**. If every head matches, requested bytes are bounded by `4 × Σ min(Lᵢ, sys_len, prompt.size())`: conservatively **9.03 MB at sys_len = 22,574**, or **26.2 MB when all scanned lengths are bounded by 65,536**. CPU work is `O(N log N + Σ nᵢ)`, including comparisons and buffer initialization; ordinary full reads require up to two `pread`s plus open/close per candidate. Filesystem readahead can exceed those requested-byte counts, so these are not physical-I/O or latency bounds.

The **`n <= best` pruning is correct**: no candidate can share more than its `n` tokens. Because descending `L` also makes `n` nonincreasing, `break` would be valid there; `continue` merely does extra loop work. The 1 KiB fast path only applies when divergence occurs within the first 256 tokens—different sessions sharing the same template head still require tail reads. Also, the configured 65,536 maximum limits engine persistence, not entries accepted during rescan, so older larger entries require using their actual lengths in the bound.
The two-stage comparison looks correct, and the measured 22,460 → 21,504 cut is consistent with the engine’s chunk policy. I would address the **P1 publication race before relying on this across slots**, and the **P2 promotion gap before treating the three-session bootstrap as a general guarantee**. The writer and eviction races below predate these commits; shared cuts increase their relevance by directing different sessions toward the same cache key. This was static inspection only: no builds, tests, or repository programs were run, and no files were changed.

1. **P1 — Concurrent writers can publish an incomplete state blob.**  
   **Evidence:** All slots share one cache ([server.cu:1172](/mnt/ai/projects/q27/src/server.cu:1172)), but `pfx_busy` and writer threads belong to individual engines. `has()` checks only indexed entries ([prefix_cache.h:278](/mnt/ai/projects/q27/src/prefix_cache.h:278)); it does not reserve an in-flight key. Every writer for the same `(hash, L)` opens the same `.tmp` with `O_TRUNC` ([prefix_cache.h:294](/mnt/ai/projects/q27/src/prefix_cache.h:294)), and publication is only locked when updating the index.

   Two cold slots can select the same shared cut and both pass `has()`. Writer A can finish writing GDN; writer B then truncates the file and writes its header/tokens; A resumes at its existing offset, writes KV, and renames the file while B remains unfinished. The published file can have a valid header, matching tokens, expected size, and a hole where GDN belongs. `read_state()` accepts that shape ([prefix_cache.h:263](/mnt/ai/projects/q27/src/prefix_cache.h:263)). B can also continue modifying the inode after publication.

   **Recommendation:** Use a unique temporary file per writer and coordinate publication per cache key. A shared in-flight reservation would additionally prevent duplicate exports and writes. The per-engine busy flag is insufficient.

2. **P2 — A shorter restore prevents learning a better system cut indefinitely.**  
   **Evidence:** Shared-prefix discovery requires `base == 0` ([engine.cuh:4956](/mnt/ai/projects/q27/src/engine.cuh:4956)), and system persistence independently requires it ([engine.cuh:4289](/mnt/ai/projects/q27/src/engine.cuh:4289)). Once an 8,192-token entry matches a new client version, subsequent sessions restore it even if that version consistently shares 22,460 tokens. Those requests never discover or persist the longer system prefix. Conversation entries may still accumulate, but their session-specific suffixes do not provide a reusable system entry.

   This applies to snapshot/checkpoint hits too. The BUILDLOG explicitly says the probe inserted a foreign request because otherwise a 4,096-token checkpoint prevented the cut logic from running ([BUILDLOG.md:15818](/mnt/ai/projects/q27/docs/BUILDLOG.md:15818)). The shorter entry can remain the effective ceiling until eviction or another source supplies a better entry.

   **Recommendation:** Permit system-prefix promotion when a restored request crosses a useful boundary beyond `base`. Give promotion an explicit policy instead of merely removing the guard: disk/RAM restores set `pfx_last_persist = base`, so the existing 8,192-token growth gate could still suppress worthwhile improvements.

3. **P2 — Eviction can unlink a replacement entry and leave `has()` permanently suppressing its repair.**  
   **Evidence:** Eviction removes entries under the mutex, then unlinks their paths after releasing it ([prefix_cache.h:339](/mnt/ai/projects/q27/src/prefix_cache.h:339)). Meanwhile, a writer can rename a replacement onto that same path and insert it into the index ([prefix_cache.h:316](/mnt/ai/projects/q27/src/prefix_cache.h:316)). A delayed eviction then deletes the replacement, leaving an indexed path that no longer exists. `has()` still returns true from its length/hash metadata.

   This is distinct from the benign case where `shared_prefix()` opens an entry that was simply evicted. Here, subsequent lookups miss while persistence of that exact entry is suppressed.

   **Recommendation:** Coordinate final-path publication, index updates, and unlinking, or use generation-specific filenames so eviction cannot delete a newer generation.

4. **P3 — The minimum-length check happens before rounding, allowing an unusable shared cut.**  
   **Evidence:** The engine accepts `shared >= min_tokens` ([engine.cuh:4959](/mnt/ai/projects/q27/src/engine.cuh:4959)), then persists at a preceding chunk boundary. `pfx_should_persist()` checks the resulting boundary against the minimum ([engine.cuh:4277](/mnt/ai/projects/q27/src/engine.cuh:4277)). With configurable `min_tokens = 4500` and `shared = 4600`, the engine selects the shared cut but reaches boundary 4096, which is rejected. It consequently loses the system entry that the original, longer cut could have produced.

   Production’s aligned minimum of 4096 avoids this case.

   **Recommendation:** Validate the actual eligible chunk boundary before replacing the original system cut.

5. **P3 — Several test assertions do not establish the bounds their comments claim.**  
   **Evidence:** The “capped at L” assertion uses a prompt that already diverges at token 700, before the entry ends at 1000 ([test_prefix_cache.cpp:249](/mnt/ai/projects/q27/tools/test_prefix_cache.cpp:249)). Returning 700 does not demonstrate the entry-length cap. Likewise, the `upto = 200` case diverges at 50, before either the 64-token entry or 100-token prompt ends ([test_prefix_cache.cpp:231](/mnt/ai/projects/q27/tools/test_prefix_cache.cpp:231)).

   The head-boundary assertions correctly exercise comparison results, but an implementation that unnecessarily reads every tail could pass them all. They do not establish the claimed I/O reduction.

   **Recommendation:** Add fully matching cases that stop specifically at entry length and prompt length, plus read instrumentation or fault injection for the skipped-tail and partial-read paths.

The remaining correctness checks did not reveal another defect:

- **Index snapshot and concurrent scans:** The index is copied **under** the mutex; file I/O happens outside it ([prefix_cache.h:220](/mnt/ai/projects/q27/src/prefix_cache.h:220)). Each call owns its candidates, token buffer, and `best`. An unlink before `open()` causes a skip; an unlink after `open()` leaves the descriptor readable. Concurrent insertion can make the heuristic temporarily stale. These are safe behaviors, subject to the publication defects above.
- **Two-stage boundaries:** The head covers `[0, h)`. Continuation starts at token `h`, with destination `toks.data() + h` and file offset `64 + 4h`; for `h = 256`, that is byte 1088 ([prefix_cache.h:241](/mnt/ai/projects/q27/src/prefix_cache.h:241)). Resizing preserves the head. Positive short reads are accumulated; EOF or errors discard the candidate. `EINTR` causes a conservative miss because `read_full()` does not retry it ([prefix_cache.h:418](/mnt/ai/projects/q27/src/prefix_cache.h:418)).
- **Cut semantics:** A nonaligned shared length intentionally rounds down: 22,460 becomes 21,504 with `PF_T = 1024`. `shared == sys_len` correctly retains the original cut. A conversation entry is valid evidence of token sharing: the engine exports its current state at the chosen boundary rather than attempting to truncate the source entry’s recurrent state.
- **Gates and RAM:** Cold requests reset `pfx_last_persist` before selecting the shorter cut, so an earlier larger value does not suppress it. An outstanding writer can still make the single cut opportunity fail `pfx_busy`, without retry. Below-minimum entries are excluded correctly. Omitting RAM entries affects learning when a divergent entry exists only in RAM—during disk publication, after disk eviction, or after a failed write. Exact RAM hits already restore earlier in tier resolution. Thus this omission costs opportunities, not state correctness; production currently disables that tier.
- **Server propagation:** `sys_len` is computed per request, assigned after slot claim, and captured by value for streaming ([server.cu:2758](/mnt/ai/projects/q27/src/server.cu:2758), [server.cu:2813](/mnt/ai/projects/q27/src/server.cu:2813), [server.cu:2974](/mnt/ai/projects/q27/src/server.cu:2974)). I found no stale request-length propagation here.

For **cost**, a budget-respecting 40 GB root containing entries of at least 0.4 GB has at most 100 entries. Entries rejected within the head require at most **100 KiB of requested token reads total**. If every head matches, requested bytes are bounded by `4 × Σ min(Lᵢ, sys_len, prompt.size())`: conservatively **9.03 MB at sys_len = 22,574**, or **26.2 MB when all scanned lengths are bounded by 65,536**. CPU work is `O(N log N + Σ nᵢ)`, including comparisons and buffer initialization; ordinary full reads require up to two `pread`s plus open/close per candidate. Filesystem readahead can exceed those requested-byte counts, so these are not physical-I/O or latency bounds.

The **`n <= best` pruning is correct**: no candidate can share more than its `n` tokens. Because descending `L` also makes `n` nonincreasing, `break` would be valid there; `continue` merely does extra loop work. The 1 KiB fast path only applies when divergence occurs within the first 256 tokens—different sessions sharing the same template head still require tail reads. Also, the configured 65,536 maximum limits engine persistence, not entries accepted during rescan, so older larger entries require using their actual lengths in the bound.
