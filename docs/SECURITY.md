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

## Reporting

Open an issue at https://github.com/signalnine/q27 for anything in the
parser. If you have a token sequence that makes q27 execute a call the model
did not intend, that is the interesting one -- include the raw generated text
and the declared tools, since both matter to the recovery chain.
