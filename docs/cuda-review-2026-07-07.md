# CUDA correctness review triage (2026-07-07)

**Provenance:** external CUDA-focused security/correctness review. Each finding
independently verified against source at HEAD `623cdb1`; line numbers below are the
*corrected* ones (the review's were mostly a few lines stale). This is the tracking
ledger.

> **STATUS 2026-07-07: all six findings (#1-#6) and the additional correctness risks
> (L2-eps, move-assign UB, DeviceModel copy, --ctx floor) are FIXED** -- d_gen in the
> prefill-attn branch, the rest in commits fd0f504 (batch) and 4fa9d24 (top-p). Canonical
> 4c4120c7 unchanged throughout. The Q27_TOOL_SPLIT note stays opt-in / off by default.

**Disposition in one line:** all six are real code observations, but only **#1 fires on
the greedy production path**, and only at `--ctx > 65536`. #2/#3 are sampled-path only
(greedy uses `k_argmax`, never the Gumbel/top-p kernels). #4/#5 are latent (no live caller
reaches the buggy branch). #6 is on the greedy prefill path but dormant (the trigger state
never occurs in real activations, which is why the canonical md5 has held).

**Scope legend:** *greedy* = affects the bitwise-canonical greedy path Gabe measures;
*sampled* = only when `inv_temp > 0`; *latent* = present in code but unreachable in the
current engine.

| # | What | Verdict | Fires on | Severity | Fix cost |
|---|---|---|---|---|---|
| 1 | `d_gen` long-context OOB write | CONFIRMED | greedy + sampled | **fix-now (live at ctx>64K)** | small |
| 2 | Philox `u==1` -> infinite Gumbel | CONFIRMED | sampled | low | 1 line |
| 3 | top-p bisection -> full-vocab | CONFIRMED | sampled (`top_p<1`) | low-med | kernel edit |
| 4 | split prefill breaks for `t0>0` | CONFIRMED but LATENT | none today | none (refactor trap) | small |
| 5 | dp4a over-reads partial tiles | CONFIRMED, near-harmless | greedy + sampled | very low | 1 predicate |
| 6 | L2-norm eps differs by path | CONFIRMED, dormant | greedy prefill | invariant break, never triggers | 1 char |

Overlaps with the earlier review: **#1 == that review's #4** and **#6 == its L2 item** --
both already noted in `SECURITY-MODEL.md`'s carve-out. **#4** is cross-referenced from
`docs/plans/2026-07-07-prefill-attn.md` (that plan rewrites this kernel).

---

## #1 -- `d_gen` long-context OOB device write [FIX-NOW]

**Fires on:** greedy + sampled prefill. **Precondition:** launch ctx > 65,536.

`d_gen` is a fixed 65,536-int tracking buffer (`MAX_GEN_TRACK`, `engine.cuh:36`; allocated
`engine.cuh:336`). The batched-prefill final `step_with(prompt[NP-1])` sets `d_step = NP-1`
(`engine.cuh:1776`) and runs `k_advance`, which writes without a capacity check:

```cpp
// blocks.cu:101-102
__global__ void k_advance(int* d_pos, int* d_step, int* d_gen, const int* d_token) {
    d_gen[*d_step] = *d_token;
```

The only guard is `NP > max_ctx` (`engine.cuh:1702`); nothing checks `NP` against
`MAX_GEN_TRACK`. At default ctx (8192 / slot1 32768) a 65,537-token prompt is *refused*, so
the write is unreachable. But the advertised deep-context config that Gabe benches with
(`--ctx 131072`) admits >64K-token prompts, and then a normal deep prompt writes past
`d_gen` (a ~525 KB global overwrite at ~200K tokens). Confirmed serving-reachable via the
prefill `step_with`, not CLI-only (`server.cu` generate path). This is the one finding that
turns a normal operator action into memory corruption, with no attacker involved.

**Fix:** allocate `d_gen` from checked `max_ctx`, or drop it from the serving path entirely
(it is only read back on the CLI at `engine.cu:992`) and bounds-check the tracking write
independently of position advancement.

**FIXED 2026-07-07** (engine.cuh:336 -- `d_gen` sized to `max_ctx`; NP is bounded `<= max_ctx`
by the generate() guard, so the write is now always in bounds). Canonical 4c4120c7 unchanged.
Surfaced during prefill-attn benchmarking, whose 131072-token prefills were triggering it.

---

## #2 -- Philox `u==1` yields an infinite Gumbel [sampled, low]

**Fires on:** sampled path only (greedy uses `k_argmax`, no Gumbel).

The uniform conversion claims an open interval but isn't one:

```cpp
// blocks.cu:490-491
// (0,1): never exactly 0 or 1, so -log(-log(u)) is finite   <-- COMMENT IS WRONG
return ((float)x0 + 0.5f) * (1.0f / 4294967296.0f);
```

For the top 128 values of `x0` (`x0 >= 4294967168`), `(float)x0` rounds to `2^32` in fp32
(mantissa spacing is 256 there), `+0.5f` is absorbed, and `x 2^-32` gives **exactly
`1.0f`**. Then:

```cpp
// blocks.cu:587
float g = -logf(-logf(u));   // u==1 -> -logf(0) -> +inf
```

The `+inf` key force-wins the Gumbel-max (`blocks.cu:588-589`), selecting an arbitrary
in-nucleus token regardless of its logit. The low end is fine (`x0=0` -> `u = 2^-33 > 0`);
only the `u==1` end is broken.

**Two corrections to the review's numbers:**
- The 0.737% is the **`top_p >= 1` worst case** (whole vocabulary draws Gumbels). The draw
  is gated behind the nucleus test (`blocks.cu:585`, `if (x[i] < thresh) continue;` runs
  *before* the Philox call), so with truncation the rate is `~nucleus_size x 2^-25` --
  orders of magnitude rarer at the default `top_p 0.95`.
- Greedy is entirely unaffected.

**Compounding with #3:** when #3 fires it makes the nucleus full-vocab, which then re-exposes
#2 at its full 0.737% rate. They interact on diffuse positions.

**Fix:** clamp in `philox_uniform`, e.g. `u = fminf(u, 0x1.fffffep-1f)` (largest float < 1),
or guard the Gumbel against a non-finite `g`. The **`p==1` spec-accept sub-claim** shares
this root cause and is plausible, but the exact accept comparison was not read here --
confirm that line before touching the accept path.

---

## #3 -- top-p silently degrades to full-vocabulary sampling [sampled, low-med]

**Fires on:** sampled path with `top_p < 1` on high-entropy positions.

`k_nucleus_d` finds the nucleus threshold by a fixed 12-iteration bisection on a probability
cutoff `tau in [0,1]` (`blocks.cu:527`), so it can only resolve `tau` to `2^-12 ~= 2.4e-4`.
On a diffuse distribution where every token's prob is below that cutoff, `mass(p>=tau)` is
always `< top_p`, `s_lo` never leaves 0, and:

```cpp
// blocks.cu:546
else { float tau = s_lo; thresh = (tau <= 0.f) ? -FLT_MAX : M + (logf(tau)+logZ)/inv_temp; }
```

`tau==0` -> `thresh = -FLT_MAX` -> the nucleus is the **entire vocabulary** and `mass==1`.
So top-p silently stops truncating exactly when the distribution is most diffuse. The kernel
test only ever used 64 peaked logits (`test_kernels.cu:1305`), so this was never exercised.
Temperature `T<=0.7` sharpens distributions and reduces the trigger but does not remove it.

**Fix:** log-domain bisection (bisect on the logit threshold directly, unbounded below), or a
selection-based nucleus. Add a diffuse-distribution case to the kernel test.

---

## #4 -- split prefill attention breaks for nonzero `t0` [LATENT]

**Fires on:** nothing in the current engine.

In `k_attn_prefill_mma`, the split-partial write uses the **absolute** token index while the
scratch is sized for `SB` relative rows:

```cpp
// prefill.cu:1047 (write): tr0 = t0 + gid, trows = gridDim.y*TT (sized for SB)
size_t b0 = ((size_t)(qh * trows + tr0) * nsp + sp) * PF_PART_STRIDE;
// prefill.cu:1085 (combine): t = blockIdx.x in [0, SB) -- RELATIVE
const float* pp = part + (size_t)(qh * trows + t) * nsp * PF_PART_STRIDE;
```

With `t0 > 0` and splitting on, writes land at rows `[t0, t0+SB)` -- past the `[0,SB)` range
`trows` allots -- overlapping the next head's partials and exceeding scratch for the last
head, while the combine reads the (now-empty) relative rows. **But the sole engine caller
passes `t0` literally 0** (`engine.cuh:1446`: `..., base, 0, T, ...`), and the model always
prefills a chunk as one `t0=0` call, so this never executes. The defect is an API-contract
lie: `prefill.cuh:44` documents "a sub-batch of SB tokens starting at (base_pos+t0)",
implying arbitrary `t0` works with splitting -- it doesn't.

**Fix:** make the write use relative rows (`tr0 - t0`) *or* size scratch from `t0+SB` *or*
document and assert `t0==0` when `part != nullptr`. **Cross-ref:** the prefill-attn plan
rewrites this kernel; a future `t0>0` sub-batching refactor would silently corrupt here.

---

## #5 -- dp4a prefill over-reads partial token tiles [near-harmless]

**Fires on:** greedy + sampled prefill, last (partial) tile only.

The Q4/Q8 staging loops read every `tt in [0,TB)` without the `tt < nt` guard the compute and
write already have:

```cpp
// prefill.cu:52-56 (Q4 staging -- no nt guard):
s_eo[...] = c0+cc < n_chunks ? __ldg(eo + (size_t)(t0+tt)*ept + ...) : make_uint2(0,0);
// prefill.cu:78 (compute -- guarded):   if (tt >= nt) break;
// prefill.cu:94 (write -- guarded):     if (lane==0 && i < nt) y[...] = v;
```

(Same shape for Q8 at `prefill.cu:121-126`.) For the last tile where `nt < TB`, staging reads
rows `[t0+nt, t0+TB)` which are `>= T` -- an OOB read of the activation arrays. The staged
values are **discarded** (compute breaks at `tt>=nt`), so there is no correctness impact; the
only risk is a fault if an activation buffer ends exactly on a page boundary, which
over-provisioned CUDA allocations make unlikely. The kernel test masks it by padding `XQuant`
(`test_kernels.cu:255`); the public API does not require that padding.

**Fix:** add the `t0+tt < T` predicate to the staging loads (cheap), or document the padded
allocation contract the test already relies on.

---

## #6 -- L2-norm epsilon differs between prefill and decode [dormant invariant break]

**Fires on:** greedy prefill (and every path), but the trigger state never occurs.

```cpp
// blocks.cu:43-44 (decode) and spec3.cu:26 -- the DOCUMENTED semantics:
// ggml: y = x / max(sqrt(sum), eps)  == x * rsqrt(max(sum, eps^2))
float inv = rsqrtf(fmaxf(sh[0], eps * eps));
// prefill.cu:644 -- the DEVIATION:
float inv = rsqrtf(fmaxf(sh[0], eps));
```

So batched prefill regularizes with `max(sum, eps)` while decode/spec use `max(sum, eps^2)`.
With `eps=1e-6`, a head whose `sum(x^2)` falls below `eps` normalizes by up to `1000x`
differently between the two paths (`rsqrt(1e-6)=1e3` vs `rsqrt(1e-12)=1e6`). This violates the
prefill-bitwise-identity invariant -- **but only for heads with `sum(x^2) < 1e-6`**, which
RMSnormed real activations never produce, which is exactly why the canonical greedy md5 has
been stable through it. `prefill.cu:644` is the odd one out and should match the documented
ggml semantics.

**Fix:** `eps` -> `eps*eps` at `prefill.cu:644` (one char). Free; do it next time the prefill
gate cycle runs.

---

## Fix queue

Ordered by real payoff. All touch canonical-gated hot paths, so each needs the full
`make` + canonical md5 (`4c4120c7...`) + `test_kernels` cycle on the 5090 before landing --
Gabe's review-and-push gate.

1. **#1** -- `d_gen` bounds/alloc. Only one that corrupts memory in a config Gabe runs.
2. **#6** -- one-char eps fix; restores the prefill-identity invariant for free.
3. **#2** -- one-line `u` clamp; correctness of the sampled path (low rate, but a real bug
   with a wrong comment asserting it can't happen).
4. **#3** -- top-p log-domain/selection rewrite + a diffuse-dist test. Larger; sampled-path
   quality on high-entropy positions.
5. **#5** -- staging predicate, folded into any future prefill-kernel touch.
6. **#4** -- fix or assert the `t0==0` contract when the prefill-attn plan rewrites the
   kernel (tracked there).

Greedy production today is exposed only to **#1** (at ctx>64K); #5/#6 are greedy-path but
dormant; #2/#3 are sampled-only; #4 is latent.
