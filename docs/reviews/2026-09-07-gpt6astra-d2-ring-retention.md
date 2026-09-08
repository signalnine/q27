# gpt-6-astra review: DFlash2 ring retention across turns (2026-09-07)

Reviewer: gpt-6-astra via codex exec (read-only, xhigh), static review of the
uncommitted diff (d2_prefill_align + Dflash2 ctx_end/ctx_contig/rollback_to +
the last-token tap seed). No GPU run on the reviewer's side.

## Verdict

Bookkeeping checks out for solo serving: alignment assigns the prompt
covering both the chunk and final-token ingests; rounds append the verified
INPUT tokens oc[1..n]; truncation removes the same n-m suffix from the ring,
the position and the sequence; sampled bootstrap and forced-pending
replacements happen before ingestion; cancellation leaves at most an
unreported committed suffix that the next LCP removes; RAM/disk, checkpoint
and snapshot restores all pass through alignment with no token-lineage
error found. The eager token_launches(d2_vtaps) tail step retains the
captured launch sequence's KV/GDN/logits and advance's d_pos/d_step/d_gen
effects. The serial (chunked) prefill path still captures no prompt taps
(missing coverage, not wrong coverage; tiny prompts only).

Three P2s, two pre-existing:

1. Compaction could overlap: keep = D2_WINDOW with ctx_n = 4095, T = 2 copies
   rows [2047, 4095) onto [0, 2048) -- an overlapping cudaMemcpyAsync
   (undefined), for K, V and positions, with the host still believing the
   ring intact.
2. Fused batch rounds (conductor commit_outcome) advance the target without
   touching d2_pos / the ring / d2_seq; a later solo d2 round would label
   taps with stale positions while every host invariant still holds.
3. A disconnected seed window (base < NP - 2048) latched ctx_contig = false
   permanently; after a later slide the coverage is contiguous again but the
   next identical request would reset a valid ring -- warm starvation again.

## Resolution

1. ingest keeps min(ctx_n, D2_WINDOW - T) rows on a slide: everything older
   cannot fall inside the window once the chunk lands (lossless), and with
   ctx_cap = 2 * D2_WINDOW the copied tail always starts past its
   destination (asserted) -- overlap impossible.
2. Q27_DFLASH2 now refuses to start unless Q27_BATCH=0 (single-slot, solo
   rounds), the only mode it was ever wired for.
3. A chunk that does not continue at ctx_end drops the retained rows before
   appending (they are all > D2_WINDOW behind any future query, so lossless)
   instead of latching a flag; the ring stays contiguous by construction.

Raw transcript: session scratchpad ring_codex.out.
