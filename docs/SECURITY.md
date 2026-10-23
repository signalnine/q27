# Security model

q27 runs a language model's output through a hand-written parser and turns it
into tool calls that something else executes. The model chooses every byte that
parser sees. This document says what that does and does not put at risk, what
has been tested, and what has not.

The framing is Boyd Kane's, ["LLMs could control their host machines by
exploiting inference engines"](https://boydkane.com/essays/llms-could-control-their-host-machines-by-exploiting-inference-engines):
a model that emits a token sequence its own inference engine mishandles is
attacking a machine that holds its weights and sits inside a trusted network.
The argument is worth taking seriously here, because q27 is precisely that
shape of program.

## Trust boundaries

```
  model weights ──> q27-server (GPU host) ──> HTTP client (Claude Code, in Docker)
                    ^ parser runs HERE          ^ tool calls EXECUTE here
```

Two facts follow:

- **A parser bug is code execution on the GPU host**, not merely a bad tool
  call. The parser shares a process with the engine and the weights.
- **A *semantic* parser bug is code execution in the client's sandbox.** q27
  does not run tool calls; it emits them, and the client runs them with the
  client's permissions. Recovering a call the model did not intend is a real
  escalation even though no memory is corrupted.

The mitigation Kane recommends -- keep the token parser off the GPU host --
is half-satisfied by accident: execution already happens in a container that
is not the GPU host. The parser itself is not separated.

## What q27 does not have

The specific vulnerability class in the essay does not reach this codebase.

- **No `eval`.** CVE-2025-9141 was vLLM's XML tool parser passing tool-call
  arguments to `eval()`. q27 has no `eval`, `system`, `popen`, `exec*`,
  `dlopen`, or `fork` anywhere in the parser or the server.
- **No template engine.** Prompt rendering is hardcoded C++
  (`chatml_prompt`, `tools_preamble`). There is no Jinja, so the ~35-template
  surface the essay describes does not exist. This is also why q27 cannot
  accept user-supplied chat templates (issue #38) -- the same decision, priced
  both ways.
- **No dynamic model-architecture dispatch.** One architecture, one loader.

## What q27 does have: a deliberately forgiving parser

The real surface is `parse_bare_tool_calls` and the drift chain around it --
23 numbered recovery modes (mode 23 added after this doc was first
written), brace repair, dialect-closer blanking, tool-name inference,
and recovery from inside reasoning. Every one exists because a model emitted
that shape in production and lost a call. Every one is also a rule that turns
model-authored bytes into control flow.

Two distinct risks, and they need different defences.

### 1. Memory safety (would give GPU-host code execution)

~120 `substr()` calls in `api_common.h` alone (2026-10-01 count; ~109 when
this was first written) plus dozens of span computations over model-authored
strings.
The characteristic failure is a span whose `source_begin` is `npos` or behind
the cursor, making `raw.substr(cursor, source_begin - cursor)` underflow to
`SIZE_MAX`. That exact bug was found in review on 2026-08-20 (mode 20 leaving
middle calls unstamped) and is now guarded at both call sites.

**Tested by fuzzing** (`make fuzz`). The harness drives every entry point the
server reaches with generated text: the drift chain with and without EOF
repair, `parse_tool_call`, the splitter feeding `resolve_ordered_tool_segments`,
the streaming `BareToolTextHoldback` on both TEXT and THINK, and
`recover_unclosed_tool_tail`. It asserts the span invariant the handler
depends on (`source_begin <= source_end <= size`) rather than waiting for the
underflow downstream.

`make fuzz-schema` (2026-10-07) is the same chain with the TOOL SCHEMA varying
alongside the model bytes: the fixed three-tool schema above never reached the
2026-10-04 crash (an MCP-style `"type":["string","null"]` threw out of the
streaming parse), so this harness feeds `anthropic_tools_json` whatever JSON
the input starts with and runs every schema-reading entry point, plus the JSON
tool grammar, over the rest.

Status 2026-08-24: 1.7M inputs under ASan+UBSan with the standalone mutator
(six seeds, zero hits), plus a coverage-guided libFuzzer session at ~3,100
exec/s that added 8,161 coverage-increasing inputs without a crash. Neither is
a proof. Run it after touching the parser.

Re-run on every hunt since. 2026-10-10: `make fuzz` 592,842 runs and `make
fuzz-schema` 1,280,475 runs, 20 minutes each, clean. 2026-10-19 and
2026-10-22: every CPU suite under ASan+UBSan with no report and 60,000
`fuzz-gcc` iterations clean each time, plus (10-19) compute-sanitizer
memcheck over nine decode/prefill configs on the 5090, 0 errors.

### 2. Semantic escalation (would give client-sandbox execution)

A well-formed parse that produces a call the model never made. This is the
likelier failure here, and fuzzing cannot find it: there is no crash oracle,
only the question "did this text mean to be a call?"

The standing rule is that **a model writing *about* a call must not make one**.
Markup inside a code fence or a backtick span stays prose, enforced by the
display-context lexer (`markdown_lex.h`) that every opener check consults.
This is why recovery goes through `BareToolTextHoldback` and never calls
`parse_bare_tool_calls` directly -- the parser alone will happily execute a
fenced example, verified while fixing issue #38.

Defences are per-shape and adversarial by construction: each recovery mode
ships with a test that it does NOT fire on the displayed form of the same
bytes. When adding a drift mode, add both.

`--constrain-tools` is the strict alternative: grammar-constrained decoding
makes malformed calls unrepresentable rather than recoverable. It costs
generality and is off by default. Recovery and constraint are the two ends of
this trade; q27 currently sits at the permissive end.

## Serving posture

- Binds `127.0.0.1` by default. `--host 0.0.0.0` without `--api-key` accepts
  unauthenticated requests from anyone who can reach the port; the flags are
  documented together in the README for that reason.
- `--api-key` comparison is constant-time (`secure_compare`), deliberately
  without early exit on length or first difference.
- `--enable-metrics` (default off) adds an auth-exempt `GET /metrics`,
  reachable without a key for the same reason `/health` is: pollers scrape
  without credentials. It exposes operational metadata only -- request and
  token counters, latency histograms, occupancy gauges, uptime -- never
  prompt text or generated content. Without the flag the route does not
  exist (404).
- No auth by default is a loopback-only assumption. Do not expose the port.
- Slow connections: a request's bytes (request line, headers, body) must
  arrive within `--read-deadline` seconds of the first one (30 by default),
  and one peer address may hold at most `--max-conn-per-ip` connections (64,
  below the 92 worker threads the pool grows to). Each HTTP connection
  occupies one worker for its whole life, so without these limits a modest
  number of connections that never finish sending their headers could hold
  every worker (found 2026-10-22, before any key check runs). Many sources
  can still fill the pool; put a reverse proxy in front of an exposed port.
- Browser guards: form content-types are refused (415) on every POST, and on a
  loopback bind with no key a `Host` header that does not name the loopback
  address is refused (403). Both close the paths a web page has to a
  loopback service through the operator's browser.
- Log lines never carry raw client or model bytes: the request path and
  method, tool names, and every parser rescue line go through a sanitizer
  (control bytes become `?`, lengths are capped), and tool names with control
  characters are rejected at the request.
- Size limits before any work: the HTTP body cap is 100 MB, and a rendered
  prompt longer than any window could hold (`max_prompt` x the longest vocab
  token, 128 bytes) is a 400 on all three APIs and `count_tokens` before it
  is tokenized (2026-10-19). The three request-text paths that were
  quadratic (NFC ordering over a combining run, the specials scan, the BPE
  merge loop) are linear or n log n now; the worst crafted megabyte went
  from 107 s to 131 ms. Past the worker pool the accept queue holds 256
  sockets and closes the rest.
- A client that leaves is dropped: while queued (499 in the log), between
  the slot claim and the prefill, and every 16 tokens on the non-stream legs
  (2026-10-19). Before that a closed non-stream request decoded to
  `max_tokens` for nobody.
- A failed generation is never presented as a finished answer: each API's
  stream ends in its own error shape and a non-stream leg returns 500 even
  with partial text (2026-10-22). The failure this closes is an agent acting
  on a truncated answer it believes complete.
- Configuration is parsed strictly: every `Q27_*` knob is checked once at
  boot against a range, an empty export counts as unset, and a value that
  does not parse refuses to boot naming the variable (2026-10-22). The
  failure this replaces was silent: an empty `Q27_KV=` from an unset shell
  variable meant fp16 KV and half the context, and an empty DFlash2 reserve
  meant a zero reserve and an out-of-memory at the first request.

## Bug hunts

Each hunt is a set of read-only reviews over surfaces no earlier hunt had
read, plus a live probe set against the production config, and each shipped
as a release. The receipts are the BUILDLOG entries.

| date | BUILDLOG | release | closed |
|---|---|---|---|
| 2026-10-01 | (az) | v0.14.2 | a silent T3 GEMV error on unused widths, the CLI DFlash2 `map::at`, an estimator mismatch |
| 2026-10-04 | (ba) | v0.14.3 | a server-killing tool schema, a stack-overflow body, a cache key that trusted file size |
| 2026-10-07 | (be) | v0.14.4 | the plain-lanes forced close broken both ways, a grammar cache that mixed schemas, a re-engage that never fired |
| 2026-10-10 | (bf) | v0.14.5 | request semantics against a live server (`max_tokens`/`n` 400s, Responses dialect, image placeholders), a sampler barrier race, the cache key by drafter mode |
| 2026-10-19 | (bg) | v0.14.6 | three quadratic paths in request text, the marker-strip bypass that forged role boundaries, the `/health?verify` key bypass, dead clients still served |
| 2026-10-22 | (bh) | v0.14.7 | the request-read deadline and per-address cap, failed generations reported as finished, a dropped second tool call on the stream legs, a 500 on short prompts for no-MTP packs, strict knob parsing |

Still open from the hunts, listed under the README's open items: stream
vs non-stream divergence for two reasoning shapes, no SIGTERM drain, the
keepalive on the slot wait only, the disk-tier restore under the GPU gate,
and the `--constrain-tools` mask-pool growth.

## Reporting

Open an issue at https://github.com/signalnine/q27 for anything in the
parser. If you have a token sequence that makes q27 execute a call the model
did not intend, that is the interesting one -- include the raw generated text
and the declared tools, since both matter to the recovery chain.
