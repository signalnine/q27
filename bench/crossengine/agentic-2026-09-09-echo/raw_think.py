#!/usr/bin/env python3
"""Thinking length on FIXED, byte-identical raw prompts across engines: sends
pre-rendered prompt text (ending in "<|im_start|>assistant\\n<think>\\n") to a
raw completion endpoint N times with seed=1..N and the card sampler, and
parses the output itself -- thinking = text before </think>, first tool =
the first <function=NAME>. Removes every engine-side template difference, so
two engines differ only in numerics.

  raw_think.py <base_url> <label> <q27|llama> <prompt_dir> <prefix> <select.txt> [N] [max_tokens]

q27 uses /v1/completions, llama.cpp /completion. Prompts are
<prompt_dir>/<prefix>_<seq>.txt (real session content: keep out of the repo);
this writes lengths only, <label>.jsonl next to this file."""
import json, os, re, sys, time, urllib.request
base, label, mode, pdir, prefix, sel = sys.argv[1:7]
N = int(sys.argv[7]) if len(sys.argv) > 7 else 4
max_tokens = int(sys.argv[8]) if len(sys.argv) > 8 else 16384
here = os.path.dirname(os.path.abspath(__file__))
FN = re.compile(r"<function=([^>\n]+)>")
out = open(os.path.join(here, f"{label}.jsonl"), "w")
for seq in [int(x) for x in open(sel).read().split()]:
    prompt = open(os.path.join(pdir, f"{prefix}_{seq}.txt")).read()
    ths = []
    for seed in range(1, N + 1):
        samp = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.05, seed=seed, stream=False)
        if mode == "q27":
            url = "/v1/completions"; b = dict(prompt=prompt, max_tokens=max_tokens, **samp)
        else:
            url = "/completion"; b = dict(prompt=prompt, n_predict=max_tokens, cache_prompt=True, **samp)
        req = urllib.request.Request(base.rstrip("/") + url, data=json.dumps(b).encode(),
                                     headers={"content-type": "application/json", "authorization": "Bearer local"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r: resp = json.load(r)
        except urllib.error.HTTPError as e:
            print(f"seq {seq} seed {seed}: HTTP {e.code} {e.read()[:300]}", flush=True); continue
        dt = time.time() - t0
        if mode == "q27":
            ch = resp["choices"][0]; text = ch.get("text", ""); ntok = resp.get("usage", {}).get("completion_tokens"); stop = ch.get("finish_reason")
            npr = resp.get("usage", {}).get("prompt_tokens")
        else:
            text = resp.get("content", ""); ntok = resp.get("tokens_predicted"); stop = resp.get("stop_type")
            npr = resp.get("tokens_evaluated")
        closed = "</think>" in text
        think = len(text.split("</think>")[0].strip()) if closed else len(text.strip())
        after = text.split("</think>", 1)[1] if closed else ""
        tools = FN.findall(after)
        row = dict(seq=seq, seed=seed, think=think, closed=closed, tools=tools, out=ntok, prompt=npr,
                   stop=stop, text_after=len(after.strip()), secs=round(dt, 2))
        out.write(json.dumps(row) + "\n"); out.flush(); ths.append(think)
    ths.sort()
    print(f"  seq {seq:4d}: think median {ths[len(ths)//2] if ths else -1:6d} mean {sum(ths)/max(1,len(ths)):7.0f} (n={len(ths)})", flush=True)
print(f"== {label}: done")
