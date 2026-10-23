# Security model: q27 is a single-operator engine

**Status:** authoritative scoping doc. Written 2026-07-07 in response to an external
security review that evaluated q27 as an exposed multi-tenant service. Most of that
review's HIGH findings are correct *code observations* but assume a threat model q27
does not operate under. This doc states the actual threat model, explains why the
multi-tenant / untrusted-artifact findings are out of scope by design, and -- so real
bugs are not lost in the dismissal -- carves out the findings that still bite a single
user and remain live.

---

## The actual threat model

q27 is a personal research inference engine. Every trust boundary the review assumes is
adversarial is, in this deployment, controlled by one person:

| Boundary | Review assumed | Reality for q27 |
|---|---|---|
| **Network** | Internet-exposed, hostile clients | localhost (or trusted LAN on `haight`), driven by the operator's own `claude-code-q27-haight` harness |
| **Client** | Untrusted, may send crafted requests to exfiltrate/abuse | the operator's own Claude Code / CLI, sending well-formed OpenAI/Anthropic requests |
| **Model + tokenizer artifacts** | Attacker supplies a hostile `.q27` / `.tok` | self-produced quants of the operator's own model (`qwopus-27b-mtp.q27` / `.tok`), built on the same box |
| **Tenancy** | Many mutually-distrusting users share one server | one user; `--slots` multiplexes *the operator's own* concurrent requests, not distinct principals |

Under this model q27's job is to be **fast and numerically correct for a cooperative
caller**, not to be **hostile-input-hardened**. Those are different products. The review
graded the second; q27 is the first. That is a deliberate scope choice, not an
oversight -- hardening has real cost (validation on every hot path, allocation from
checked bounds, per-request cache scoping) and buys nothing against a threat that is not
present.

### What is explicitly NOT defended against

- Untrusted network clients / missing auth / missing TLS.
- Denial of service from adversarial request volume, body size, or pathological input.
- Cross-request / cross-tenant isolation and information leakage.
- Hostile model or tokenizer files.
- Resource-exhaustion abuse (holding a GPU slot, stalling inference).

If any of these become real -- see **"When this model breaks"** at the end -- the
dismissed findings re-activate immediately and this doc is void.

---

## Findings dispositioned OUT (multi-tenant / untrusted-artifact only)

Each of these requires an assumption from the "Review assumed" column above. They are
left as-is by design. (Finding numbers follow the external review.)

**#3 -- Unsafe network defaults, unbounded admission, quadratic BPE.**
`host="0.0.0.0"` default (`server.cu:218`), no auth/TLS, httplib `SIZE_MAX` payload
default, unbounded work queue, O(n^2) BPE on long mergeable words. Every one of these is
a network-exposure or adversarial-input concern. On localhost with a cooperative client
there is no attacker to send an oversized body or a pathological merge word, and the
operator does not DoS themselves. *Update 2026-07-12 (0b34934): the default bind is now
`127.0.0.1`; a non-loopback bind is an explicit `--host` opt-in, so the bind half of this
finding is closed in code, not just dispositioned.* *Update 2026-07-19: opt-in
`--api-key`/`--api-key-file`/`Q27_API_KEY` auth exists now (see the addendum below) --
the auth half of this finding has a mitigation available, but it is a shared-secret
gate, not the TLS/rate-limiting/request-size-bounding the finding also names. Those
remain open; see "When this model breaks" for the recommended disposition if any of them
become relevant.* *One caveat with a self-inflicted
edge:* the quadratic BPE (`tokenizer.cpp:110-121`, erase-in-loop, no word-length cap) can
bite the operator's own coding-agent workload -- a single large no-whitespace blob
(minified JS, base64) collapses to one word and tokenizes O(n^2) single-threaded before
any context check. Not a security issue, but the one sub-point of #3 worth a bounded-word
fix on performance grounds. *Shipped 2026-07-14 (94e645a): WORD_CAP=1024 chunks over-cap
words; inert below 1024 B, canonical byte-identical.* *Update 2026-10-22: the
availability sub-points of this finding are bounded now; see the addendum of
that date. The WORD_CAP chunking is gone with them: the BPE merge is the
standard heap merge (2026-10-19), 198 ms per MB and HF-identical on long words,
where the chunked path split them differently.*

**#5 -- Disconnected clients keep consuming GPU; public `/health?verify` stalls
inference.** Both are resource-abuse vectors: a hostile client disconnects mid-generation
to waste GPU, or hammers the weight-checksum endpoint to stall decode. A single operator
who disconnects has simply cancelled their own work, and does not attack their own health
endpoint. Cancellation-on-disconnect is a *nicety* (frees your own GPU sooner), not a
security control. *The nicety shipped 2026-07-14 (94e645a): all three streaming APIs
(OpenAI, Anthropic, Responses) now stop generating when the client drops.*
*Update 2026-10-19: it covers a request abandoned while queued (dropped at the
queue, 499), the window between the slot claim and the prefill, and the
non-stream legs (polled every 16 tokens); before that a closed non-stream
request decoded to `max_tokens` for nobody. `/health?verify` is key-gated
inside the handler now; the pre-routing exemption had let a form-urlencoded
body carry `verify=1` past it.*

**#6 -- Tool parsing "fails open."** The tolerant parser scans prose for call-shaped
JSON, repairs truncation, and does not check parsed tool names against the request's
allowlist. This is a concern *only if the tool-call text is adversarial*. Here the
tool-call text is produced by the operator's own model answering the operator's own
prompts, and the tools are executed by the operator's own trusted client. The tolerance
is a **feature** in this setting -- it is exactly what lifted CC 0.00 -> 0.55 by rescuing
the model's slightly-malformed calls (see BUILDLOG 2026-07-06 parser-drift fixes).
Removing it to satisfy a strict-framing security posture would regress the engine's
actual job. (The *name-inference-picks-wrong-tool* sub-point has a small correctness
angle unrelated to security -- see carve-outs.)

**#8 -- Constrained tool masks not isolated between requests.** `signature()` omits tool
names from the grammar-cache key (`toolgram.h:277-289`); the cache is global
(`server.cu:310`). Cross-request mask reuse is a *tenant-isolation* bug: it only bites when
two concurrent requests carry *different* tool sets with a colliding name-prefix, so one
gets a mask built from the other's names. It requires the opt-in `--constrain-tools` flag
(off by default, greedy-path only) AND concurrent `--slots` with differing tool sets. A
single operator running one stable tool set never triggers it. (Note: the comment at
`toolgram.h:274-276` actively *asserts* the name prefix fully determines transitions --
which is the bug, not a documented caveat -- so if `--slots` + `--constrain-tools` ever run
together with varied tool sets, this is a real latent defect, not a known-safe corner.) Out
of scope until multi-principal / multi-tool-set serving exists.

**#9, #10, #11 -- Malformed model metadata / offsets / tokenizer corrupt host or GPU
memory.** Negative `attn_layers` index, wrap-prone offset bounds (`loader.cpp`),
unchecked tokenizer header reads (`tokenizer.cpp:58`), zero-length special-token
infinite loop (`tokenizer.cpp:224`), unbounded embedding-gather index
(`kernels.cu:519`). All confirmed real -- **and all require a hostile artifact.** q27
loads exactly one model and one tokenizer, both self-produced on the same machine. A
malicious `.q27` is not in the threat model.

> **PARTIALLY CLOSED 2026-08-19**, after an external review pointed out that the
> quickstart now distributes artifacts through Hugging Face -- which weakens the
> "self-produced on the same machine" premise this disposition rests on. Two
> changes, both in the direction this paragraph already prescribed:
>
> - **The negative `attn_layers` index is fixed**, along with a second path in the
>   same loop: `find_first_of` returns `npos` on a truncated `meta_json` and the
>   result was indexed before the loop head could catch it.
> - **The "single startup validator" now exists**: `validate_tensor_manifest()`
>   (`engine.cuh`) pins the name, dtype class and exact shape of every tensor the
>   engine consumes, before any `cudaMalloc` and before `upload_all()`. It closes
>   the class where a container-valid file declares a shape the kernels then use
>   to write into compile-time-sized buffers. Shapes were read off 21 shipped
>   artifacts and are invariant; `tools/test_manifest.cu` runs both legs.
>
> **Still open in this class:** the tokenizer findings (#10/#11 above), the loader's
> offset arithmetic, and the embedding-gather bound. The manifest checks the tensor
> TABLE, not the payload bytes behind it. Treat third-party `.q27`/`.tok` files as
> out of scope still -- this is a narrower door, not a closed one.
>
> **Code check 2026-10-01:** the loader half is narrower than the line above
> says. `loader.cpp` bounds-checks every data and scale offset against the
> file with overflow-safe arithmetic, requires 256-byte alignment, and
> `validate_tensor_payload()` checks each payload's exact size and the T2/T3
> code ranges (since 2026-07-28 / 2026-08-01). Tokenizer header reads throw on
> truncation (`read_exact`, 2026-08-02). The embedding-gather kernels
> (`k_embed_row_q8` / `k_embed_row_t2`) still index the row by token id with
> no bound.

**Also out: `Q27_TOOL_SPLIT` documented race** (`engine.cuh:1343`). Already known,
already opt-in, already documented OFF under `--slots`. Not a new finding.

---

## Carve-out: findings that bite EVEN a single user

> **STATUS 2026-07-07: all carve-out bugs below are FIXED** -- d_gen OOB (#4, prefill-attn
> branch), null-content abort (#1), empty-prompt stale-state (#2), ChatML injection (#7),
> and the correctness bugs (L2-eps, Model move-assign UB, DeviceModel double-free, --ctx
> floor) in commits fd0f504 and 4fa9d24. Canonical 4c4120c7 unchanged. The multi-tenant /
> untrusted-artifact findings above remain dispositioned out by design (not "unfixed vulns").

> **STATUS 2026-07-14 (second triage, BUILDLOG "security/robustness review triage"):** the
> remaining single-user polish landed in 94e645a -- empty prompts now rejected HTTP 400 at the
> handler plus a `ckpt_best()` size_t-underflow fix (a REAL crash found past #2's analysis: it
> ran at slot selection, before the engine-entry `NP>=1` guard, once any checkpoint existed),
> disconnect cancellation on all streaming APIs, BPE WORD_CAP=1024, terminal `finish_reason`.
> The hostile-artifact findings (#9/#10/#11 class) stayed dispositioned out per this doc.

The "ignore multi-tenant" instruction must not bury these. None of them need an attacker.
A benign malformed request, an oversized-but-honest prompt, or untrusted *content* (not
an untrusted client) from the operator's own workflow trips them. These are the ones
worth an actual fix pass, ranked by how easily normal use hits them.

> A later CUDA-focused review re-confirmed the memory-safety carve-outs and added sampler
> bugs; full triage with corrected line numbers and a fix queue is in
> `docs/cuda-review-2026-07-07.md`. Finding #1 there is #4 below (`d_gen` OOB); its #6 is
> the L2-eps item below.

1. **#4 -- Prompts > 65,536 tokens corrupt GPU memory (in the deep-context config).**
   `d_gen` is a fixed 65,536-entry allocation (`MAX_GEN_TRACK`, `engine.cuh:36/336`);
   batched prefill's final `step_with` runs `k_advance`, which writes `d_gen[*d_step]` with
   `d_step = NP-1` and no capacity check (`blocks.cu:101`). Confirmed reachable in the
   serving path, not CLI-only. **Precondition:** the launch ctx must exceed 65,536 -- at
   default ctx (8192 / slot1 32768) a 65,537-token prompt is *refused* by the `NP > max_ctx`
   guard (`engine.cuh:1702`), so it never reaches the write. But the advertised deep-context
   config the operator actually benches with (`--ctx 131072`) admits >65,536-token prompts,
   and then a normal deep prompt writes out of bounds (~525 KB overwrite at 200K). **This is
   the top single-user bug precisely because it is live in the config you run.** Fix:
   allocate `d_gen` from checked `max_ctx`, or drop it from the serving path (it is only read
   back on the CLI).

2. **#1 -- One malformed request aborts the whole server (OpenAI endpoints only).**
   `{"messages":[{"role":"user"}]}` with no `content` reaches nlohmann's const `operator[]`
   (`server.cu:536`, `m["content"]`), which asserts -> `SIGABRT` (Makefile builds with no
   `-DNDEBUG`, so the assert is live) that httplib cannot catch, killing every in-flight
   generation. **Scope that matters for your deployment:** this fires only on the OpenAI
   `/v1/chat/completions` and `/v1/completions` paths. The Anthropic `/v1/messages` path --
   the one the `claude-code-q27-haight` harness uses -- was already given exactly this guard
   (`api_common.h:259`, `if (!m.is_object() || !m.contains("content")) continue;`), and
   `/v1/responses` operates on a non-const body so it auto-vivifies instead of asserting. So
   it is unreachable from your primary harness and only bites if you drive the OpenAI text
   endpoints. Still a one-line guard worth adding for symmetry.

3. **#2 -- Empty prompt yields garbage output (NOT a crash -- review overstated this).**
   The review's "size_t underflow reads beyond the empty vector at `engine.cuh:1555`" is
   **refuted**: on the empty-prompt path the value computed is `int P = (int)prompt.size()-1
   = -1` (signed, cast before subtract -- no wrap), the batched ckpt path that does contain a
   genuine `size_t` `prompt.size()-1` is gated behind `NP >= 32` and never entered, and no
   host vector is indexed OOB. What actually happens: `reset()` does not clear `d_token`, so a
   zero-token prompt (missing `prompt` on `/v1/completions`) decodes from stale recurrent
   state and echoes the *prior request's* pending token (`spec3.cu:533`). No crash, no memory
   unsafety -- a LOW-severity degenerate-input correctness bug, OpenAI-text-endpoint only.
   Fix: reject or BOS-seed empty prompts; require `NP >= 1` at engine entry.

4. **#7 (content half) -- ChatML control tokens in untrusted CONTENT.** The client is
   trusted, but the *content* the operator feeds it may not be -- a fetched web page, a file,
   or a tool result containing `<|im_end|><|im_start|>system...` becomes real role delimiters
   because the tokenizer recognizes specials anywhere (`tokenizer.cpp:218`) and content is
   concatenated raw (`api_common.h:188/200`). This is the one injection finding that survives
   the single-user model, *conditional on the operator routing untrusted text through q27*.
   If q27 only ever sees operator-authored prompts, it is moot; the moment it summarizes a
   web page, it is live. Fix: tokenize untrusted content with special-token recognition
   disabled.

5. **Correctness bugs (no attacker, no threat model needed):**
   - **L2-norm epsilon mismatch:** batched prefill uses `max(sum, eps)` (`prefill.cu:644`)
     while serial/spec use `max(sum, eps^2)` (`blocks.cu:44`). A genuine batched-vs-serial
     numeric divergence -- relevant to this project's *bitwise-prefill-identity* invariant,
     which the warm-vs-cold gates depend on. Worth reconciling on the merits, quite
     apart from security.
   - **`Model` move-assign UB** (`loader.cpp:46`, explicit `~Model()` then assign into dead
     members) and **`DeviceModel` implicitly copyable** despite owning raw CUDA pointers
     (double-free on copy). Latent C++ UB; only bites if those paths are exercised, but free
     to fix.
   - **`--ctx < 7` warmup OOB** (`engine.cuh:981`): operator-misconfiguration, not attack;
     add a min-ctx floor.
   - Dev hygiene: tokenizer test passes with zero cases on a missing fixture
     (`test_tokenizer.cpp:257`); `make` can leave stale server binaries (`Makefile:29`,
     already rule #1 in the plans).

**Recommended disposition for the carve-outs:** **#4 is the only near-term fix** -- it turns
a normal deep-context prompt (in the `--ctx 131072` config you bench with) into a GPU
memory overwrite, and no attacker is involved. #1 is a cheap one-line guard for symmetry but
is already covered on the harness path you actually use. #2, #7, the quadratic-BPE
word-cap, and the C++ correctness bugs (L2 eps, move-assign UB, `--ctx < 7`) are a modest
hardening/cleanup pass whenever convenient. None require adopting the review's
network/multi-tenant posture.

---

## Addendum 2026-07-16: the continuous-batching surface

v0.2.0 ships continuous batching ON by default in the serving profile (BUILDLOG
2026-07-14..16): a conductor thread fuses concurrent slots' decode rounds into shared
weight sweeps (plus replayed CUDA graphs) and streams tokens back through per-request
TokenQueues. What this does and does not change here:

- **The tenancy row above is unchanged.** Batch mode co-schedules *the operator's own*
  concurrent requests on one GPU; no new principal exists, so cross-request co-residency
  in a fused round is a performance structure, not a trust boundary. The moment slots
  serve distinct principals, the "When this model breaks" triggers below apply exactly
  as written.
- **Cross-request contamination is a gated numeric claim, not an assumption.** Batched
  output is byte-identical to the solo references (master-gate refs, canonical,
  sampled-seed, ninv incl. the table-twin leg), and a request alone on the box runs the
  k==1 solo path (0.00% measured delta).
- **Error posture (A2).** CUDA failures stay process-fatal -- the server exits rather than
  serve past a poisoned GPU context. A host-side exception in one member's round
  bookkeeping fails *that member's* queue only; the fused round continues for the rest.
- **Config guard is two-tier (M2).** User-explicit `Q27_BATCH=1` plus an incompatible env
  (`Q27_PMIN<=0`, `Q27_DEXIT=0`, `Q27_SAMPLE_PLAIN`, `Q27_TOOL_SPLIT`) is a fail-fast
  FATAL exit; profile-*default* batching auto-disables with a one-line notice and serves
  exactly as pre-batching. Kill switches: `Q27_BATCH=0`, `Q27_BATCH_GRAPH=0`,
  `Q27_PROFILE=ref`.

Nothing in this addendum re-activates or weakens the dispositions above; the DoS /
multi-tenant / hostile-artifact scoping is unaffected.

---

## Addendum 2026-07-19: opt-in API key authentication

`--api-key`/`--api-key-file`/`Q27_API_KEY` add an optional shared-secret gate
(`server.cu`, httplib pre-routing handler; `api_common.h` for the constant-time
compare and header parsing). This does **not** change the threat model or
tenancy row above, and does not supersede this doc's closing recommendation
("bind `127.0.0.1`, terminate auth/TLS/rate-limiting at an authenticated
reverse proxy" for real exposure) -- it is a lighter-weight option for the
middle ground between pure loopback-trust and standing up a full proxy: a
trusted-LAN deployment, a quick tunnel, or a container where the operator
wants *something* between "wide open" and "not reachable at all" without the
operational overhead of nginx/Caddy in front.

**What it is:** a single shared secret (or a small rotation set via
`--api-key-file`/multiple `--api-key` flags), checked via
`Authorization: Bearer <key>` or `x-api-key: <key>` before route dispatch.
Constant-time comparison; `/health` stays open for infra health checks.

**What it explicitly is NOT** -- the rest of finding #3 and the DoS/tenancy
findings above are otherwise unchanged by this:
- Not TLS. The key travels in plaintext over an unencrypted connection
  unless the operator adds TLS separately (a reverse proxy, an SSH tunnel,
  a VPN). A shared secret sent in the clear over an untrusted network is
  interceptable.
- Not rate-limiting or request-size bounding (finding #3's other
  sub-points -- unbounded httplib payload size, unbounded work queue --
  are still open). *Bounded since 2026-10-19/22: see the addendum of
  2026-10-22. Rate limiting is still the proxy's.*
- Not multi-tenant isolation. One valid key authenticates the caller as
  *the* operator, full stop -- there is still one principal, one KV/mask
  cache, one set of slots. This narrows "who can send requests" without
  adding any of the per-tenant isolation findings #2/#8 above would need
  for actual multi-tenant serving.
- Not a defense against a compromised or malicious *client* -- it defends
  the boundary between "has the key" and "does not," not what a key-holder
  can do once authenticated (still everything, per the existing threat
  model).

**Revised tripwire guidance:** the first bullet under "When this model
breaks" below (non-loopback bind) is now a *spectrum* rather than a binary
-- non-loopback + no key is unchanged (fully open, findings #3/#5/#8 fully
live); non-loopback + `--api-key` narrows the exposure to "anyone with the
key" but the rest of finding #3 (no TLS, no size/rate bounding) and the
tenancy findings (#2 leak-half, #8) are unaffected and still require the
proxy-based disposition this doc recommends if the deployment is genuinely
multi-tenant or internet-facing.

---

## When this model breaks (re-activation triggers)

This entire doc is contingent. The out-of-scope findings become live again the instant any
of these becomes true -- treat this list as the tripwire:

- q27 binds anything other than loopback / trusted LAN, or goes behind the anarres reverse
  proxy for anyone but the operator (findings #3, #5, #8 re-activate).
- q27 serves more than one principal, or one principal's requests must be isolated from
  another's (#2 leak-half, #8).
- q27 loads a model or tokenizer it did not produce (#9, #10, #11 -- ship the startup
  validator first).
- The operator's own client stops being trusted to only send well-formed, non-hostile
  tool-call text (#6).

If exposure ever changes, the cheapest correct move is **not** to harden q27's internals to
multi-tenant grade -- it is to keep q27 single-trust and put the trust boundary in front of
it: bind `127.0.0.1`, and terminate auth/TLS/rate-limiting at an authenticated reverse
proxy. That preserves the "fast, correct, cooperative-caller" design and re-scopes findings
#3/#5 to the proxy where they belong. Only the memory-safety carve-outs (#4, #1) must be
fixed in q27 itself regardless, because they are reachable by the operator today.

## Addendum 2026-07-24: the persistent prefix cache writes conversation state to disk

`--prefix-cache DIR` (P16, v0.6.0, **off by default**) is the first q27
feature that persists anything derived from request content past the process
lifetime. It does not change the tenancy row or the threat model above, but
it does move data across a boundary this doc previously did not have to
discuss, so state it plainly.

**What lands on disk.** For each cached prefix, one file containing:
- the **token IDs** of the prefix, verbatim (they are the verification
  payload -- a hash alone cannot safely key a restore, see BUILDLOG
  2026-07-24), and
- the model's **KV rows and recurrent state** for those tokens.

The tokens are the user's and operator's prompt text in tokenized form:
system prompts, tool definitions, and the beginning of the conversation are
recoverable from them with nothing more than the tokenizer. Treat a
`--prefix-cache` directory as **conversation content in plaintext**, at the
same sensitivity as a chat log. It is not encrypted and it is not obfuscated.

**Consequences the operator owns:**
- Files are created mode 0600 in a directory created 0700, so the default is
  owner-only. That is the whole access control; anything more (full-disk
  encryption, a tmpfs, a per-user directory) is the operator's to add.
- Do NOT point `--prefix-cache` at a shared, world-readable, backed-up, or
  synced location without deciding that a chat log belongs there.
- The cache is keyed on prefix CONTENT, not on identity. If two principals
  ever reach one q27 (which this doc already tells you not to do -- there is
  one principal by design), an entry written by one is restorable by the
  other, but only by a request that already contains the same tokens. It
  leaks no content to a caller who cannot already produce that prefix; it is
  a timing-observable shared cache, not a read primitive.
- Entries survive until LRU eviction (`--prefix-cache-max-gb`) or manual
  deletion. There is no TTL and no "forget this conversation" call.

**What it does not weaken.** The restore path re-verifies the full token
vector before loading state, and every entry carries a compat hash over the
model bytes, buffer geometry, and KV format, so a stale, corrupt, foreign, or
truncated file is refused rather than coerced (`tools/test_prefix_cache.cpp`
forges each case). A malicious file in the cache directory cannot make the
server continue a conversation whose tokens the requester did not supply --
though anyone who can WRITE to that directory can already do far worse to the
process, and is outside the model here as elsewhere.

## Addendum 2026-08-30: the /metrics surface

`--enable-metrics` (default off) adds an auth-exempt `GET /metrics`
(Prometheus text exposition; see `docs/metrics-endpoint.md`). Disposition:
metadata-only telemetry -- request and token counters, latency histograms,
occupancy gauges, server uptime. No prompt text, no generated tokens, no
conversation state reaches the endpoint, and the prefix cache's on-disk
content is not readable through it. It widens the unauthenticated surface
the same way `/health` does -- an on-path observer learns that this is a
q27 server and how busy it is -- so it stays opt-in and rides on the same
loopback / api-key posture as everything else. When the flag is absent the
route is not registered at all (404). Nothing in this addendum re-activates
or weakens the dispositions above.

---

## Addendum 2026-10-22: the availability sub-points after bug hunts 5 and 6

Finding #3's dismissal rested on "the operator does not DoS themselves".
Two hunts (BUILDLOG (bg) 2026-10-19 and (bh) 2026-10-22; the posture list is
in `docs/SECURITY.md`) found the single operator doing exactly that through
their own tooling, and bounded the paths inside q27 rather than at a proxy,
because each one bites on loopback:

- A coding-agent harness that opens connections and never finishes a
  request, or a browser tab, could hold every HTTP worker (one worker per
  connection for its life, 23 growing to 92 on this box) before any key
  check runs. Each request's read now has to finish within `--read-deadline`
  seconds of its first byte (30; writes are unaffected, a long streaming
  response is fine) and one peer address holds at most `--max-conn-per-ip`
  connections (64). Past the pool the accept queue holds 256 sockets.
- Request-size bounding: the body cap is 100 MB (the vendored httplib's
  default since the 0.53.1 update, so the `SIZE_MAX` in #3 is stale), and a
  rendered prompt that cannot fit the window is a 400 before it is
  tokenized. The three quadratic request-text paths (NFC ordering, the
  specials scan, the BPE merge) are gone, chunking included; the worst
  crafted megabyte went from 107 s to 131 ms.
- Browser guards that matter precisely in the single-operator model: a
  form-content-type POST is a 415 and, on a loopback bind with no key, a
  `Host` header naming anything but the loopback address is a 403. Both
  close the paths a web page has to a loopback service through the
  operator's own browser (DNS rebinding, simple-request form posts), which
  a proxy in front would never have seen.
- An abandoned request is dropped at every stage (the 2026-10-19 update to
  #5 above), so the operator's own cancelled work frees the GPU.
- Strict `Q27_*` parsing: an empty export or a typo refuses to boot naming
  the variable, instead of silently selecting fp16 KV or a zero drafter
  reserve.

The threat model is unchanged: one principal, no TLS, rate limiting still
a proxy's job, and `--max-conn-per-ip` is the one knob a proxy deployment
has to raise or disable, since every client arrives from the proxy's
address. What changed is the list of things that need the proxy: payload
size, slow senders and the quadratic tokenizer have come off it.
