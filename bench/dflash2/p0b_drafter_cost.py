# P0b: DFlash2 drafter per-round cost, bf16, faithful to the dflash_generate loop:
# warm draft KV cache (ctx rows), then per round: forward(target_hidden=produced fresh
# rows, noise=8 masks) + propose(). Run on whatever GPU CUDA_VISIBLE_DEVICES exposes.
import sys, time
sys.path.insert(0, '/mnt/ai/projects/dflash')
import torch
from dflash.model import DFlash2DraftModel, _make_cache

DRAFT = '/mnt/ai/models/qwen38-27b-dflash2-bf16'
dev = 'cuda:0'
print('[gpu]', torch.cuda.get_device_name(0), flush=True)

m = DFlash2DraftModel.from_pretrained(DRAFT, attn_implementation='sdpa', dtype=torch.bfloat16).to(dev).eval()
W = m.block_size  # 8
H = m.config.hidden_size
TAP = len(m.target_layer_ids) * H  # 25600

# fake lm_head on-device (vocab x H) -- propose() needs it; count its cost too
vocab = m.config.vocab_size
head = torch.nn.Linear(H, vocab, bias=False, dtype=torch.bfloat16, device=dev)

torch.manual_seed(0)
def rnd(*s): return torch.randn(*s, dtype=torch.bfloat16, device=dev) * 0.02

@torch.inference_mode()
def run(warm_ctx, produced, iters=50):
    cache = _make_cache(m.config)
    pos = torch.arange(warm_ctx + W, device=dev)[None]
    # warm the cache: one big context ingest
    m(target_hidden=rnd(1, warm_ctx, TAP), noise_embedding=rnd(1, W, H),
      position_ids=pos, past_key_values=cache, use_cache=True)
    cache.crop(-W)
    anchor = torch.tensor([9], device=dev)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in range(iters):
        th = rnd(1, produced, TAP)
        noise = rnd(1, W, H)
        p = torch.arange(warm_ctx - produced, warm_ctx + W, device=dev)[None]
        hid = m(target_hidden=th, noise_embedding=noise, position_ids=p,
                past_key_values=cache, use_cache=True)[:, 1 - W:, :]
        cache.crop(-(produced + W))
        toks, cand, _ = m.propose(hid, anchor, head, 0.0)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f'[p0b] warm_ctx={warm_ctx} produced={produced}: {ms:.2f} ms/round', flush=True)
    return ms

run(512, 4, iters=10)   # warmup + compile paths
for ctx in (512, 2048):
    for produced in (2, 4, 8):
        run(ctx, produced)
print('[done]', flush=True)
