# P0a part 2: (1) tie-flip diagnostic at the smoke's divergence point,
# (2) AL sweep across traffic types, dflash leg only (identity already smoked).
import sys, time, json, collections
sys.path.insert(0, '/mnt/ai/projects/dflash')
import torch
torch.manual_seed(0)
torch.set_num_threads(12)
import dflash.model as dm
dm._cuda_time = time.perf_counter

from transformers import AutoModelForCausalLM, AutoTokenizer
from dflash.model import DFlash2DraftModel, dflash_generate

TARGET = '/mnt/ai/models/qwen38-27b-hf'
DRAFT = '/mnt/ai/models/qwen38-27b-dflash2-bf16'

tok = AutoTokenizer.from_pretrained(TARGET)
target = AutoModelForCausalLM.from_pretrained(TARGET, attn_implementation='sdpa', dtype=torch.bfloat16).eval()
draft = DFlash2DraftModel.from_pretrained(DRAFT, attn_implementation='sdpa', dtype=torch.bfloat16).eval()
print('[load] ok', flush=True)

eos = target.generation_config.eos_token_id or tok.eos_token_id
stop_ids = [eos] if isinstance(eos, int) else list(eos)

def encode(user):
    msgs = [{"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(prompt, return_tensors='pt', add_special_tokens=False)

# ---- Part 1: tie diagnostic ----
iso = "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H59M' into a datetime.timedelta. Handle edge cases."
input_ids = encode(iso)
out = dflash_generate(draft, target, input_ids, 64, stop_ids, temperature=0.0, return_stats=True)
new = out.output_ids[0, input_ids.shape[1]:]
DIV = 57
prefix = torch.cat([input_ids[0], new[:DIV]])[None]
with torch.inference_mode():
    l_full = target(prefix, logits_to_keep=1).logits[0, -1].float()
    half = prefix.shape[1] // 2
    from dflash.model import _make_cache
    c = _make_cache(target.config)
    target(prefix[:, :half], past_key_values=c, use_cache=True)
    l_split = target(prefix[:, half:], past_key_values=c, use_cache=True, logits_to_keep=1).logits[0, -1].float()
for name, l in (('full-prefill', l_full), ('split-prefill', l_split)):
    v, i = torch.topk(l, 3)
    print(f'[tie/{name}] top3: ' + ' | '.join(f'{tok.decode([t])!r} {x:.4f}' for x, t in zip(v.tolist(), i.tolist()))
          + f' | margin {v[0]-v[1]:.4f}', flush=True)

# ---- Part 2: AL sweep ----
PROMPTS = {
    'code-write': iso,
    'prose': "Explain the tradeoffs between mmap and read() for sequential file IO on Linux. Cover page cache behavior and readahead.",
    'code-edit': "Here is a function:\n\ndef retry(fn, n=3):\n    for i in range(n):\n        try:\n            return fn()\n        except Exception:\n            pass\n\nFix it so the last exception propagates and add exponential backoff with jitter. Show the full corrected function.",
    'echo': 'Here is a JSON config:\n{"host": "172.17.0.1", "port": 8080, "kv": "fp8", "think_budget": 0, "temp": 1.0, "top_p": 0.95, "model": "qwen38-27b-mtp"}\nRepeat it back exactly, changing only the port to 8081.',
}
results = {}
for name, user in PROMPTS.items():
    ids = encode(user)
    t0 = time.time()
    out = dflash_generate(draft, target, ids, 192, stop_ids, temperature=0.0, return_stats=True)
    al = out.acceptance_lengths
    n_new = out.output_ids.shape[1] - ids.shape[1]
    results[name] = dict(tok_per_round=sum(al)/len(al), rounds=len(al), tokens=n_new,
                         hist=dict(sorted(collections.Counter(al).items())))
    print(f'[al/{name}] {n_new} tok, {len(al)} rounds, {sum(al)/len(al):.2f} tok/round, '
          f'hist={results[name]["hist"]} ({time.time()-t0:.0f}s)', flush=True)
print('[summary]', json.dumps(results), flush=True)
print('[done]', flush=True)
