# DFlash2 Phase 0 measurement rig (2026-09-06)

Offline validation of the z-lab Qwen3.8-27B-DFlash2 drafter before any
engine work. Design + results: `docs/plans/2026-09-06-dflash2-integration.md`
(Phase 0 results section). Raw output: `p0a.log`, `p0a_sweep.log`.

Setup (one-time):

```
git clone --depth 1 https://github.com/z-lab/dflash /mnt/ai/projects/dflash
python3 -m venv --system-site-packages /mnt/ai/venvs/dflash2
/mnt/ai/venvs/dflash2/bin/pip install 'transformers==5.15.0'
```

The venv rides the system CUDA torch; transformers 5.15 is required both for
`DynamicCache.activate_past_recording` (the loop's GDN-state rewind) and to
load the composite Qwen3.8 checkpoint directly as `Qwen3_5ForCausalLM`
(5.5.0 fails on `vocab_size`).

- `p0a_cpu_smoke.py` -- E2E greedy: plain HF generate vs `dflash_generate`
  on one prompt; token identity + acceptance lengths. CPU-only (~5 min),
  serving GPUs untouched. Run with `CUDA_VISIBLE_DEVICES=` set empty.
- `p0a_al_sweep.py` -- the tie-flip diagnostic at the smoke's divergence
  point + acceptance across four traffic types (code-write / prose /
  code-edit / echo), 192 tokens each. The incumbent side of the table comes
  from sending the same four prompts to live q27 serving at temperature 0
  and reading `dec`/`rounds` off the `[req]` log lines.
- `p0b_drafter_cost.py` -- drafter per-round cost in eager torch, faithful
  to the generate loop's shape (warm draft KV, few fresh context rows + 8
  noise rows per round). Point CUDA_VISIBLE_DEVICES at a free GPU.

Model paths are hardcoded for haight (`/mnt/ai/models/qwen38-27b-hf`,
`/mnt/ai/models/qwen38-27b-dflash2-bf16`).
