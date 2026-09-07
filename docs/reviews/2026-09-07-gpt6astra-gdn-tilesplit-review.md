# gpt-6-astra review: GDN fold column-tile split (2026-09-07)

Review of the <<<48,512>>> -> <<<192,128>>> column-tile split of k_delta_scan_T
(the commit-fold), converting one-block-per-head to CT=4 tiles of 32 columns.

VERDICT: "bitwise-safe as a source transformation." No indexing, reduction-
order, or synchronization defect. Exhaustively verified:
- Coverage exact: every (column, row-tile) pair and all 16384 state elements
  per head owned exactly once across the 4 blocks; each column's 4 row-tile
  threads are all in one block.
- sq/sk full-128 load correct (row-indexed reads need all rows; column-tile
  offset must NOT be added to i0+k).
- Reduction order unchanged: ((part0+part1)+part2)+part3 with identical row
  partitions and ascending-k accumulation -> per-column bitwise.
- All 5 barriers correct with 4 warps; single-writer on every shared entry;
  no inter-block sync needed (blocks share no writable state).
- Global addressing (vj, oT, Sgh) uses full column j; cc only for scratch.
  Shared mem 3584 -> 1664 B/block. 768 warps spread over more blocks.

Two caveats it could not check source-only, RESOLVED empirically here:
- FMA contraction could change bits vs the smem version. -> E2E --dflash2 is
  byte-identical to plain greedy over 192 tokens x 4 prompts; any FMA-induced
  bit change would diverge. Confirmed identical.
- ninv_test FOLD compares committed state (Sg), not the oT output rows. -> the
  fold's oT (fold_o) is write-only scratch, never read downstream (only S is
  advanced), so oT identity is moot; and E2E covers anything consumed.

Gates green: ninv_test FOLD BITWISE PASS (vs serial delta_step), E2E
byte-identical, 0 ptxas spills, wsum canonical-clean. Measured k_delta_scan_T
22.77 -> 15.37 ms (-32%; -45% vs original smem).
