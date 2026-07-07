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
operator does not DoS themselves. *Mitigation already available:* `--host 127.0.0.1`
closes the bind concern in one flag the day it matters. *One caveat with a self-inflicted
edge:* the quadratic BPE (`tokenizer.cpp:110-121`, erase-in-loop, no word-length cap) can
bite the operator's own coding-agent workload -- a single large no-whitespace blob
(minified JS, base64) collapses to one word and tokenizes O(n^2) single-threaded before
any context check. Not a security issue, but the one sub-point of #3 worth a bounded-word
fix on performance grounds. See carve-outs.

**#5 -- Disconnected clients keep consuming GPU; public `/health?verify` stalls
inference.** Both are resource-abuse vectors: a hostile client disconnects mid-generation
to waste GPU, or hammers the weight-checksum endpoint to stall decode. A single operator
who disconnects has simply cancelled their own work, and does not attack their own health
endpoint. Cancellation-on-disconnect is a *nicety* (frees your own GPU sooner), not a
security control -- tracked as ordinary polish, not a fix-now item.

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
memory.** Negative `attn_layers` index (`engine.cuh:313`), wrap-prone offset bounds
(`loader.cpp:127`), unchecked tokenizer header reads (`tokenizer.cpp:58`), zero-length
special-token infinite loop (`tokenizer.cpp:224`), unbounded embedding-gather index
(`kernels.cu:519`). All confirmed real -- **and all require a hostile artifact.** q27
loads exactly one model and one tokenizer, both self-produced on the same machine. A
malicious `.q27` is not in the threat model. The correct disposition is a *single
startup validator* if q27 ever ingests third-party artifacts (e.g. a public model zoo);
until then this is defending a door that only the operator has a key to.

**Also out: `Q27_TOOL_SPLIT` documented race** (`engine.cuh:1343`). Already known,
already opt-in, already documented OFF under `--slots`. Not a new finding.

---

## Carve-out: findings that bite EVEN a single user

> **STATUS 2026-07-07: all carve-out bugs below are FIXED** -- d_gen OOB (#4, prefill-attn
> branch), null-content abort (#1), empty-prompt stale-state (#2), ChatML injection (#7),
> and the correctness bugs (L2-eps, Model move-assign UB, DeviceModel double-free, --ctx
> floor) in commits fd0f504 and 4fa9d24. Canonical 4c4120c7 unchanged. The multi-tenant /
> untrusted-artifact findings above remain dispositioned out by design (not "unfixed vulns").

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
     which is load-bearing for the warm-vs-cold gates. Worth reconciling on the merits, quite
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
