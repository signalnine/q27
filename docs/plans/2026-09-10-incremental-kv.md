# Plan: incremental KV entitlements (issue #42 step 2)

Step 1 (2026-09-10, v0.11.2) let every slot's window grow to the whole
paged pool. What still made a burst queue instead of share was the size of
each request's reservation: `claim_slot` entitled prompt + max_tokens +
round reserve up front, and Claude Code asks for 64K on every turn, so four
30K-prompt turns wanted 4 x 94K rows of a ~269K-token 4-slot pool on a
5090 and ran two at a time. Most turns write a few thousand tokens.

## Design

- **Admission** entitles prompt + 4096 rows + round reserve (capped at the
  declared maximum) instead of the maximum itself. The declared maximum,
  `Slot::ent_max`, is prompt + max_tokens (as the admission limits resolved
  it) + the forced reasoning-close tokens + round reserve, clamped to the
  window. The close tokens were left out before this change and could write
  a few rows past the mapped pages at the very end of a response.
- **Growth** happens at round boundaries. `Engine::pre_round` (the shared
  pre-check of the solo and fused decode paths) compares the rows the next
  round may write, `Ph + ctx_round_reserve()`, with the lineage's
  entitlement `kv_rows`. Short: it calls the server's `on_need_rows` hook,
  which grows the lineage by 4096 rows (or just what the round needs) under
  `route_m`. If the hook cannot grant, the task is marked **parked**: it
  passed its pre-checks but may not write this round.
- **The conductor** skips parked members for the round and re-checks them
  at the next boundary; they keep their slot, KV and position. A round in
  which every live member is parked runs nothing, releases the GPU gate
  and waits (2 ms, or until a join or a freed slot pokes it) -- spinning
  while holding the gate would starve the prefill the head of line is
  waiting on.
- **Only conductor members park.** Every decode runs in the conductor when
  it exists (all eight `generate()` call sites are `conductor ?
  batch_generate : eng.generate`). Without the conductor (Q27_BATCH=0,
  DFlash2 serving) requests still reserve their maximum up front and the
  pre_round check is a belt that stops rather than write past the
  entitlement. Incremental mode also needs more than one slot and the pool;
  `Q27_KV_INCREMENTAL=0` turns it off.

## Why it cannot deadlock (no preemption needed)

Every grant -- an admission or a growth -- is made only if the resulting
state is **safe** in the banker's-algorithm sense (`src/kv_bank.h`): there
is an order in which every active request can reach its declared maximum,
where a request that finishes returns its pages to the idle pool (idle
lineages are reclaimable cache; `kv_grant` scavenges them LRU-first). With
one resource type, ordering by remaining need is optimal, so the check is a
sort and a scan.

In a safe state the request with the smallest remaining need can always be
granted any growth within its maximum, and the state stays safe (its need
and the available pages drop by the same amount). So at any moment at
least one active request can progress:

- if it is a conductor member, its growth is granted and it runs;
- if it is still in prefill, it needs no growth (admission covered its
  prompt and first round) and will register with the conductor;
- when it finishes, its pages become idle, which only raises what is
  available to everyone else.

Parked requests therefore wait only for other requests to finish, never
for each other. A request that has finished decoding but not yet freed its
slot counts as needing nothing more (`Engine::kv_growth_done`, set by
`finish_decode`), so its unused max_tokens do not hold others back.

What this does not provide: fairness. A parked request waits while newly
admitted ones run, as long as the state stays safe; the same barging
already applied to slot waits. And an idle conversation's cache is still
the first thing reclaimed under pressure.

## Lock order

The growth hook runs on the conductor thread inside a round, so it holds
the GPU gate and then takes `route_m`. No `route_m` holder ever takes the
gate (`claim_slot` waits on `route_cv`, which releases `route_m`;
`free_slot` and `/metrics` hold it briefly), so the order cannot invert.
`kv_entitle` uploads the block table on the member engine's own stream
before the round's draft kernels, which the fused verify waits on.

## Gates

- CPU: `tools/test_kv_bank.cpp` (safety check, page arithmetic, the
  head-of-line growth property), `tools/test_conductor.cpp` parking cases
  (parked member skipped and kept, rejoins, all-parked idle round, solo
  path among parked members, cancel beats park).
- GPU: `fused_smoke` (conductor union-vs-solo byte identity), the boot
  matrix (`bench/pool/elastic_boot.sh`), `bench/pool/elastic_admission.py`
  (step-1 regression), `bench/pool/incremental_admission.py` (burst of four
  64K-max_tokens turns; four long outputs growing past the pool), and the
  burst with `Q27_KV_INCREMENTAL=0` as the control.
