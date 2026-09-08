# Cross-engine re-bench 2026-09-07 evening (post levers 1+2)

Instrument: bench/ladder/drive_seeded.py -- seeded prompts (6 seeds x
{12.5K, 50K} ctx), think + echo tasks, sampler temp 1.0 / top-p 0.95 /
top-k 20 / min-p 0.05, think-on, 512 tokens decode, single stream. Same
request bytes to every engine; round telemetry from q27 [req] journals and
ninfer --request-log-jsonl. q27 = tonight's build (sampled ladder e533baf +
sampled dflash2 dcbab9b). ninfer = master 487f897, release
Qwen3.8-27B-nvfp4 artifact, int8 KV 131072.

## Decode t/s (mean of 6 seeds)

| leg                      | think 12.5K | think 50K | echo 12.5K | echo 50K |
|--------------------------|--:|--:|--:|--:|
| q27 ladder               | 148.8 | 129.2 | 163.1 | 147.4 |
| q27 dflash2-sampled      | 138.9 | 131.2 | 161.6 | 162.7 |
| ninfer dflash2 k=7       | 203.1 | 186.9 | 236.4 | 220.8 |
| ninfer MTP3 (control)    | 131.3 | ~124  | -- | -- |

## Decomposition (think 12.5K means)

| leg | tok/round | round wall |
|---|--:|--:|
| q27 ladder | 2.53 | 17.5 ms |
| q27 dflash2 | 3.08 | 22.3 ms |
| ninfer dflash2 | 3.65 | 18.0 ms |
| ninfer MTP3 | 2.50 | ~19.0 ms |

## Verdict

1. ENGINE: q27's round wall BEATS ninfer's at equal acceptance -- ladder
   17.5 ms vs their MTP round ~19.0 ms; q27 ladder +13% t/s over their MTP3
   on identical prompts (148.8 vs 131.3). The old "engine gap" is not just
   closed, it is inverted.
2. Their entire lead is the DFlash2 arm: 3.65 tok/round at +0 wall premium
   (their dflash2 round is FASTER than their MTP round -- one drafter
   forward replaces 3 sequential MTP steps). Content mix is exonerated by
   the MTP3 control (2.50 == our 2.53 on the same prompts).
3. q27's dflash2 closes a third of the acceptance gap (3.08) but pays a
   +4.8 ms round premium (22.3 vs ladder 17.5): drafter ~2.0 + width-8
   verify vs the ladder's gated ~3.5 avg + walk/ingest. Net: parity with
   our own ladder, 30%+ behind their dflash2 arm.

## Next engineering targets (both dflash2-integration, ranked)

1. The +4.8 ms d2 round premium: adaptive verify width for d2 rounds
   (drafter-confidence-gated, the old astra lever), drafter cost (Q4 pack
   read ~2 ms -- theirs rides nvfp4 at ~0.5), overlap ingest with draft.
2. The 3.08-vs-3.65 acceptance delta at the same drafter class: their
   pack/taps (nvfp4, possibly retrained) vs our Q4 conversion of the z-lab
   checkpoint (known -1.5..-3.6% from quant, rest unexplained -- per-lane
   accept profiles are in the logged telemetry for a follow-up).
