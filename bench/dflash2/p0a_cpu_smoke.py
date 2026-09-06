# P0a CPU smoke: E2E DFlash2 rig validation, BF16 target + BF16 drafter, greedy.
# Leg 1: plain target greedy generate (baseline tokens)
# Leg 2: dflash_generate greedy (token identity vs baseline + acceptance lengths)
import sys, time, json, collections
sys.path.insert(0, '/mnt/ai/projects/dflash')
import torch
torch.manual_seed(0)
torch.set_num_threads(12)

import dflash.model as dm
dm._cuda_time = time.perf_counter  # CPU run; keep CUDA context off the serving GPUs

from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
from dflash.model import DFlash2DraftModel, dflash_generate

TARGET = '/mnt/ai/models/qwen38-27b-hf'
DRAFT = '/mnt/ai/models/qwen38-27b-dflash2-bf16'
N_NEW = 128

t0 = time.time()
tok = AutoTokenizer.from_pretrained(TARGET)
try:
    target = AutoModelForCausalLM.from_pretrained(TARGET, attn_implementation='sdpa', dtype=torch.bfloat16)
    route = 'AutoModelForCausalLM'
except ValueError as e:
    print(f'[load] causal-lm route rejected ({str(e)[:80]}), falling back', flush=True)
    target = AutoModelForImageTextToText.from_pretrained(TARGET, attn_implementation='sdpa', dtype=torch.bfloat16)
    route = 'AutoModelForImageTextToText'
target = target.eval()
print(f'[load] target {type(target).__name__} via {route} in {time.time()-t0:.0f}s', flush=True)

t0 = time.time()
draft = DFlash2DraftModel.from_pretrained(DRAFT, attn_implementation='sdpa', dtype=torch.bfloat16).eval()
print(f'[load] draft in {time.time()-t0:.0f}s, taps {draft.target_layer_ids}, K={draft.block_size-1}', flush=True)

msgs = [{"role": "user", "content": "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H59M' into a datetime.timedelta. Handle edge cases."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
input_ids = tok.encode(prompt, return_tensors='pt', add_special_tokens=False)
print(f'[prompt] {input_ids.shape[1]} tokens', flush=True)

eos = target.generation_config.eos_token_id or tok.eos_token_id
stop_ids = [eos] if isinstance(eos, int) else list(eos)

t0 = time.time()
with torch.inference_mode():
    base = target.generate(input_ids, max_new_tokens=N_NEW, do_sample=False, eos_token_id=stop_ids)
wall = time.time() - t0
base_new = base[0, input_ids.shape[1]:]
print(f'[base] {len(base_new)} tokens in {wall:.0f}s ({len(base_new)/wall:.2f} t/s)', flush=True)
print('[base text]', json.dumps(tok.decode(base_new)[:400]), flush=True)

t0 = time.time()
out = dflash_generate(draft, target, input_ids, N_NEW, stop_ids, temperature=0.0, return_stats=True)
wall = time.time() - t0
new = out.output_ids[0, input_ids.shape[1]:]
al = out.acceptance_lengths
print(f'[dflash] {len(new)} tokens in {wall:.0f}s | rounds={len(al)} | tok/round mean={sum(al)/len(al):.2f}', flush=True)
print('[dflash] tok/round histogram:', dict(sorted(collections.Counter(al).items())), flush=True)

n = min(len(new), len(base_new))
match = bool((new[:n] == base_new[:n]).all().item())
print(f'[identity] first {n} tokens identical: {match}', flush=True)
if not match:
    div = int((new[:n] != base_new[:n]).nonzero()[0, 0])
    print(f'[identity] diverges at {div}: base={tok.decode(base_new[div:div+8])!r} vs dflash={tok.decode(new[div:div+8])!r}', flush=True)
print('[dflash text]', json.dumps(tok.decode(new)[:400]), flush=True)
print('[done]', flush=True)
