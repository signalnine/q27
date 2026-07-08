#!/usr/bin/env python3
"""Build the accept-gate A/B payloads (docs/acceptance-gate-design.md Phase 0).

Three ~26K-token raw-completion payloads spanning the saturation axis, per the
depth-match recipe (BUILDLOG:1841) and the maxd6 rig (BUILDLOG:1655):

  repro    docs + verbatim self-copy cut mid-line -> continuation echoes
           (the >90%-fired regime where depth-5 measured +2.9%)
  code     src concat cut mid-function -> in-style fresh-ish codegen
           (the T8-style mid regime where depth-5 measured -5.4%)
  testgen  src concat + open unit-test stub -> fresh generation
           (lowest-saturation regime, -3.9%)

GOTCHA (BUILDLOG:1870): greedy raw completion of closed prose EOSes instantly
at depth -- every payload ends mid-flow (open continuation).

Emits scratchpad/accept_payload_{repro,code,testgen}.json -- ready-to-POST
/v1/completions bodies (greedy, max_tokens 256). ~3.9 chars/token estimate;
the [req] prompt= field reports the true count on first use.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "scratchpad"
# ~26K tokens. chars/tok measured per flavor (BUILDLOG text tokenizes DENSE:
# 97898 chars -> 33137 tok = 2.95; code ~3.2). The rig's first [req] prompt=
# field is the ground truth -- keep prompts well under --ctx 32768.
TARGET_TOK = 26_000
DOCS_CPT, CODE_CPT = 2.95, 3.2

def read(*rel):
    return "\n\n".join((ROOT / r).read_text(errors="replace") for r in rel)

def cut_open(text, n):
    """Trim to n chars ending MID-LINE (keep the partial line -- an
    incomplete statement forces continuation; dropping it lets greedy EOS,
    measured dec=7 on the code payload). Back up past whitespace/close-punct
    so the cut lands mid-expression."""
    t = text[:n]
    while t and (t[-1] in " \t\n.,;!?)}]\"'" ):
        t = t[:-1]
    return t

def emit(name, prompt):
    body = {"prompt": prompt, "max_tokens": 256, "temperature": 0}
    p = OUT / f"accept_payload_{name}.json"
    p.write_text(json.dumps(body))
    print(f"{p.name}: {len(prompt)} chars (~{len(prompt)/3.0:.0f} tok est)")

# echo: a ~2.5KB code block repeated to depth, cut mid-block -> continuation
# is short-range verbatim echo. The high-acceptance anchor (depth-match P4
# flavor: llama hit 100% acceptance here). Long-range self-copy does NOT
# saturate (measured y5 0.53 at a 13K-token echo distance -- run 2).
code_chars = int(TARGET_TOK * CODE_CPT)
block = read("src/blocks.cu")[:2_500]
reps = code_chars // (len(block) + 20) + 1
echo = ("\n// ---- copy ----\n".join([block] * reps))
emit("echo", cut_open(echo, code_chars))

# docs: docs + verbatim self-copy cut mid-line (run-1 "repro" recipe) -- the
# measured near-breakeven mid point (y5 0.556, d5 +0.2% in run 1).
docs = read("docs/BUILDLOG.md")
half = int(TARGET_TOK * DOCS_CPT) // 2
base = docs[:half]
emit("docs", base + "\n\n[archive copy for verification]\n\n" + cut_open(base, int(half * 0.92)))

# codegen: source context + open fresh-implementation stub (plain
# mid-function cuts EOS within ~8 tokens on greedy raw completion at this
# depth -- measured twice; an explicit stub forces sustained generation).
code = read("src/engine.cuh", "src/blocks.cu", "src/kernels.cu")
cg_stub = ("\n\n// ---- generated: pointer-array lane fanout (replaces the explicit\n"
           "// _a.._g per-lane selects; see maxd6-decision.md item 5). ----\n"
           "__global__ void k_quantize_lanes(const float* const* __restrict__ src,\n"
           "                                 uint8_t* const* __restrict__ dst,\n"
           "                                 int n, int nlanes) {\n    int t =")
emit("codegen", cut_open(code, code_chars - 2_000) + cg_stub)

# testgen: different source mix + open test stub (frozen since run 1).
src = read("src/blocks.cu", "src/spec3.cu", "src/prefill.cu")[: code_chars - 6_000]
stub = ("\n\n// ---- unit tests for the kernels above (test_kernels.cu harness"
        " style: CPU\n// reference, max-abs-err tolerance, PASS/FAIL lines)."
        " ----\n\nstatic void test_argmax_top2_random() {\n    const int n =")
emit("testgen", cut_open(src, code_chars - 6_000) + stub)
