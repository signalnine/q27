# Phase 1 quant-tap gate: DFlash2 acceptance on q27-captured taps vs BF16.
#
# Modes:
#   q27 <taps.bin> <prompt_len> [name]
#       Replay the drafter over a `q27 --dump-taps` dump (per step: int32
#       committed token + 5*5120 fp32 residual taps at layers {5,19,33,47,61})
#       and report tokens/round against the committed greedy stream.
#   hfcontrol <toks_file> [n_new]
#       Validate the replay rig itself: run z-lab dflash_generate E2E (BF16
#       target), then extract taps for that same trajectory with one batched
#       forward and replay -- E2E and replay tokens/round should agree.
#
# Both modes share replay(): full-context recompute per round (no drafter KV
# cache; bidirectional sliding-window attention makes recompute value-equal),
# produced = accepted prefix + 1 bonus, mirroring dflash_generate.
import sys, json, struct, collections
sys.path.insert(0, '/mnt/ai/projects/dflash')
import numpy as np
import torch
import torch.nn.functional as Fn

torch.manual_seed(0)
torch.set_num_threads(12)
from dflash.model import DFlash2DraftModel

TARGET = '/mnt/ai/models/qwen38-27b-hf'
DRAFT = '/mnt/ai/models/qwen38-27b-dflash2-bf16'
H, NTAP, WINDOW = 5120, 5, 2048


def load_drafter():
    return DFlash2DraftModel.from_pretrained(DRAFT, attn_implementation='sdpa',
                                             dtype=torch.bfloat16).eval()


def load_embed_head():
    # just the two shared tensors, not the whole 52 GB target
    from safetensors import safe_open
    idx = json.load(open(f'{TARGET}/model.safetensors.index.json'))['weight_map']
    out = {}
    for key in ('model.language_model.embed_tokens.weight', 'lm_head.weight'):
        with safe_open(f'{TARGET}/{idx[key]}', 'pt') as f:
            out[key] = f.get_tensor(key)
    embed = out['model.language_model.embed_tokens.weight']
    head = torch.nn.Linear(H, embed.shape[0], bias=False, dtype=torch.bfloat16)
    head.weight.data = out['lm_head.weight']
    return embed, head


@torch.inference_mode()
def replay(tokens, taps, prompt_len, model, embed_w, head):
    """tokens: list[int] length M; taps: [M, 5*H] bf16 (taps[i] = residuals of
    position i). Returns per-round produced counts (accepted prefix + bonus)."""
    M = len(tokens)
    K = model.block_size - 1
    mask = model.mask_token_id
    F = prompt_len
    als = []
    while F + 1 < M:
        c0 = max(0, F - WINDOW)
        th = taps[c0:F][None]
        noise_ids = torch.tensor([[tokens[F]] + [mask] * K])
        noise = Fn.embedding(noise_ids, embed_w)
        pos = torch.arange(c0, F + 1 + K)[None]
        hid = model(target_hidden=th, noise_embedding=noise, position_ids=pos,
                    past_key_values=None, use_cache=False)[:, -K:, :]
        prop = model.propose(hid, noise_ids[:, 0], head, 0.0)[0][0].tolist()
        al = 0
        for j in range(min(K, M - 1 - F)):
            if prop[j] == tokens[F + 1 + j]:
                al += 1
            else:
                break
        produced = min(al + 1, M - F - 1)
        als.append(produced)
        F += produced
    return als


def report(name, als):
    hist = dict(sorted(collections.Counter(als).items()))
    print(f'[{name}] {sum(als)} tok, {len(als)} rounds, '
          f'{sum(als)/len(als):.2f} tok/round, hist={hist}', flush=True)


def mode_q27(taps_bin, prompt_len, name):
    raw = open(taps_bin, 'rb').read()
    step = 4 + NTAP * H * 4
    M = len(raw) // step
    assert len(raw) % step == 0, f'{taps_bin}: not a multiple of step size'
    tokens, taps = [], np.empty((M, NTAP * H), dtype=np.float32)
    for i in range(M):
        o = i * step
        tokens.append(struct.unpack_from('<i', raw, o)[0])
        taps[i] = np.frombuffer(raw, np.float32, NTAP * H, o + 4)
    model = load_drafter()
    embed, head = load_embed_head()
    als = replay(tokens, torch.from_numpy(taps).to(torch.bfloat16),
                 prompt_len, model, embed, head)
    report(f'q27-tap/{name}', als)


def mode_hfcontrol(toks_file, n_new):
    from transformers import AutoModelForCausalLM
    from dflash.model import dflash_generate
    import dflash.model as dm
    import time
    dm._cuda_time = time.perf_counter
    tokens_in = torch.tensor([[int(x) for x in open(toks_file).read().split()]])
    target = AutoModelForCausalLM.from_pretrained(
        TARGET, attn_implementation='sdpa', dtype=torch.bfloat16).eval()
    model = load_drafter()
    eos = target.generation_config.eos_token_id
    stop = [eos] if isinstance(eos, int) else list(eos)
    out = dflash_generate(model, target, tokens_in, n_new, stop,
                          temperature=0.0, return_stats=True)
    report('hf-e2e', out.acceptance_lengths)
    seq = out.output_ids
    with torch.inference_mode():
        hs = target(seq, output_hidden_states=True).hidden_states
    taps = torch.cat([hs[i + 1] for i in model.target_layer_ids], -1)[0].to(torch.bfloat16)
    als = replay(seq[0].tolist(), taps, tokens_in.shape[1], model,
                 target.get_input_embeddings().weight, target.lm_head)
    report('hf-replay', als)


def mode_bf16taps(pairs):
    """The controlled leg: teacher-force each q27 dump's token stream through
    the BF16 target and replay on ITS taps -- same tokens, only the tap source
    differs from mode q27. pairs = [(taps.bin, prompt_len, name), ...]."""
    from transformers import AutoModelForCausalLM
    target = AutoModelForCausalLM.from_pretrained(
        TARGET, attn_implementation='sdpa', dtype=torch.bfloat16).eval()
    model = load_drafter()
    for taps_bin, prompt_len, name in pairs:
        raw = open(taps_bin, 'rb').read()
        step = 4 + NTAP * H * 4
        tokens = [struct.unpack_from('<i', raw, i * step)[0]
                  for i in range(len(raw) // step)]
        with torch.inference_mode():
            hs = target(torch.tensor([tokens]), output_hidden_states=True).hidden_states
        taps = torch.cat([hs[i + 1] for i in model.target_layer_ids], -1)[0].to(torch.bfloat16)
        als = replay(tokens, taps, prompt_len, model,
                     target.get_input_embeddings().weight, target.lm_head)
        report(f'bf16-tap/{name}', als)


if __name__ == '__main__':
    if sys.argv[1] == 'bf16taps':
        args = sys.argv[2:]
        mode_bf16taps([(args[i], int(args[i + 1]), args[i + 2])
                       for i in range(0, len(args), 3)])
    elif sys.argv[1] == 'q27':
        mode_q27(sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else '?')
    elif sys.argv[1] == 'hfcontrol':
        mode_hfcontrol(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 192)
    else:
        sys.exit('mode: q27 | hfcontrol')
