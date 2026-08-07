#!/usr/bin/env bash
# Verifies tools/test_chat_completions_integration.cpp's embedded
# build_prompt/handle lambdas are byte-for-byte identical to src/server.cu's.
# Run this after ANY edit to either handle() or the integration harness --
# CI should fail loudly on drift rather than silently testing stale logic.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
import sys

def extract(text, marker):
    i = text.index(marker)
    j = text.index('{', i)
    depth = 0
    k = j
    while True:
        c = text[k]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[i:k + 1]
        k += 1

real = open('src/server.cu').read()
fake = open('tools/test_chat_completions_integration.cpp').read()
ok = True
for marker, label in (('auto build_prompt = [&]', 'build_prompt'),
                      ('auto prepare_anthropic_prompt = [&]', 'prepare_anthropic_prompt'),
                      ('auto handle = [&]', 'handle')):
    r, f = extract(real, marker), extract(fake, marker)
    if r != f:
        ok = False
        print(f'DRIFT in {label}(): harness no longer matches src/server.cu', file=sys.stderr)
        # cheap first-diff-line pointer
        rl, fl = r.splitlines(), f.splitlines()
        for n, (a, b) in enumerate(zip(rl, fl)):
            if a != b:
                print(f'  first mismatch at line {n}:', file=sys.stderr)
                print(f'    real: {a}', file=sys.stderr)
                print(f'    fake: {b}', file=sys.stderr)
                break
    else:
        print(f'{label}(): byte-for-byte match')
sys.exit(0 if ok else 1)
PY
