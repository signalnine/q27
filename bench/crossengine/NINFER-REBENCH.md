# ninfer re-bench, 2026-09-06: the 0%-reuse verdict is dead

**TL;DR:** ninfer merged `fix(engine): preserve exact agent prefix reuse`
(a140e7a, 2026-09-03) plus a DFlash2 speculative path, 254 commits past the
binary this directory's head-to-head pinned. Re-run of the agentic arm --
same 12 SWE-bench instances, same tap, same accounting, and the SAME 3.6
nvfp4 artifact bytes as the 08-17/08-19 legs -- on a fresh master build:
**51 s/inst (was 97 / 113.8), 91.3% token-weighted prefix reuse (was 0%,
then documented as a design non-goal), quality held (12/12 nonempty, 11/12
gold)**. The wall-time story that decided the original head-to-head -- ninfer
decodes fast and re-prefills itself to death -- no longer holds; ninfer now
sits at q27's door (46-49 s/inst) instead of 2-7x behind it. The README and
FINDINGS rows quoting 0% reuse are dated by this file, the same discipline
as the vLLM MTP-corruption note.

## Method

- Engine: ninfer master `487f897` (2026-09-06), built fresh in a worktree;
  the pinned 08-15 binary at `/mnt/ai/projects/ninfer/build` is untouched.
- Agentic arm: `harness/agentic.sh nvfp4m` -- 12 pinned SWE-bench_Verified
  instances via Claude Code, `tapproxy.py` in front (client-observed timing,
  `--strip-fields output_config,thinking --translate-thinking`), artifact =
  the same local `qwen3_6_27b_nvfp4.ninfer` the 08-17/08-19 legs served, so
  the delta is engine behavior alone.
- Decode sweeps: `xengine_longctx.py` arm A (cold unique prefixes, 512
  tokens, median of 3, think-on) against ninfer's released
  `Qwen3.8-27B-nvfp4-NInfer` artifact -- which now carries the DFlash2
  drafter bound in -- once with `--spec dflash2 --draft-tokens 7` and once
  with `--spec mtp --draft-tokens 3` as the paired control.

## Agentic arm (same artifact bytes as 08-17/08-19)

| | 08-17 | 08-19 | 2026-09-06 master |
|---|--:|--:|--:|
| wall/inst (mean) | 97 s | 113.8 s | **51 s** |
| prefix reuse (token-weighted) | 0% | 0% | **91.3%** |
| gold / nonempty | 11/12, 12/12 | -- | 11/12, 12/12 |
| decode (tap median) | ~250 t/s | -- | 219-227 t/s |

Reuse agrees to the decimal between ninfer's own reqlog
(`computed_prefill_tokens` 1,059,015 of `prompt_tokens` 12,174,815) and the
tap's `cache_read_input_tokens` -- a field ninfer did not emit at all in
08-17. A representative warm turn: prompt 51,326 -> computed 33. Per-instance
walls: [6, 14, 15, 19, 22, 22, 31, 43, 46, 64, 164, 166] -- half the
instances now finish in under 25 s.

That lands ninfer's reuse in the same band as q27 (88.7-92.1%) and llama.cpp
(93.9%). What remains engine-specific is HOW: their design doc frames it as
exact-identity checkpointing -- a client that reorders tool-JSON members or
rewrites history falls back to an earlier checkpoint, where q27's
normalization layer absorbs those rewrites. Claude Code's traffic evidently
matches their exact path now (they special-case its attribution metadata in
the system array).

## Qwen3.8-27B decode sweeps (released nvfp4 artifact, our instrument)

Decode t/s vs context (arm A, median of 3, think-on):

| leg | ~0 | 3K | 6K | 13K | 26K | 51K |
|---|--:|--:|--:|--:|--:|--:|
| ninfer master DFlash2 k=7 | 221.4 | 223.9 | 216.1 | 203.3 | 186.0 | 181.0 |
| ninfer master MTP3 (paired control) | 157.3 | 146.3 | 155.4 | 147.7 | 135.9 | 133.3 |
| q27-def (08-27 row, for scale) | 145 | 153 | 179 | 178 | 166 | 164 |
| ninfer 08-15 MTP3 (08-27 row) | 156 | 126 | 148 | 147 | 148 | 135 |

DFlash2 on ninfer is the first working case of that drafter beating a native
MTP path anywhere on this box (mainline vLLM ran it only degraded and lost
to its own MTP; llama.cpp does not wire it). Their own corpus puts the
DFlash2-over-MTP3 delta at +19% single-request; the paired control row above
isolates the same question on our instrument, separating drafter gain from
the 254 commits of engine movement. The control answers it: master MTP3
measures the same as the 08-15 binary did (the engine work went elsewhere),
so the whole DFlash2 margin is the drafter -- +40-50% at short context,
+36% at 51K, larger than the +19% their own corpus reports on different
traffic.

## Walls hit (and the recurring one)

- The chat template rejects Claude Code's `reasoning effort: high` -- 400,
  session dead at turn 1. That is now the THIRD engine template with this
  exact wall (vLLM's Qwen3.8 render, flash-next's GGUF template, ninfer's);
  q27 remains the only engine that aliases the effort tier. The harness tap
  strips `output_config`/`thinking` for ninfer legs; a shell-quoting slip
  that passed the strip-list unexpanded reproduced the 08-29-style turn-1
  massacre instantly. Probe one instance before trusting a leg.
- The 08-27-converted local 3.8 artifact was not needed: ninfer now
  publishes release artifacts (`neroued/Qwen3.8-27B[-nvfp4]-NInfer`), and
  the published numbers were measured on those bytes -- download, don't
  reconvert.

## What this dates

- README "Why this is interesting": the ninfer wall-time rows (97-327
  s/inst) and "re-prefills every turn" attribution -- historically true,
  measured on the 08-15 binary; current master measures 51 s/inst at 91.3%
  reuse on the same artifact.
- README "Why paged-KV engines can't cache this model": the "ninfer gets 0%
  reuse ... explicit non-goal" paragraph -- their position changed with
  a140e7a; the hybrid-GDN analysis stands, the ninfer example is dated.
- FINDINGS.md caveats quoting ninfer reuse as structurally absent.

The 08-17/08-19 head-to-head tables themselves stand as recorded -- pinned
binaries, dated. This file is the successor measurement, not a correction.
