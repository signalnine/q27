# Sampling design (temperature / top-p vs spec-verify acceptance)

Status: design. Roadmap item 3 (post-P8). Everything today is greedy argmax:
plain decode (engine.cuh:414), MTP drafts (mtp_forward), the 5 verify lanes
(argmax_masked in spec_round_launches). Spec acceptance is exact-match
(k_finish_round: `va == dr1` chain) -- lossless FOR GREEDY only. Rule zero:
the greedy path stays bitwise untouched (canonical md5 gates depend on it).

## 1. Sampled spec acceptance

Exact-match acceptance is wrong under sampling: it collapses the output
distribution toward the draft head's argmax. The standard fix (Leviathan/Chen
2023 rejection sampling): accept draft token d with prob min(1, p(d)/q(d))
(p = target, q = draft); on reject, resample from norm(max(0, p - q)) and
stop the chain. Output distribution is exactly the target's, for ANY q.

With q27's greedy drafts (section 2), q is a delta at the draft argmax, so the
test degenerates to: accept d with prob p(d); on reject, resample from p with
d zeroed and renormalized. No q bookkeeping at all.

"p" here means the SERVED distribution: temp-scaled, top-p-truncated,
renormalized. Truncation happens before the accept test -- a draft outside the
nucleus has p(d) = 0 and auto-rejects.

What the kernels need (per round, per lane over VOCAB = 248320 logits):
- **max + logsumexp** at 1/T: one grid-reduction pass, same shape/cost class
  as k_argmax (the 7 argmax chains measured 0.03 ms/round total in P3 --
  logits are 1 MB/lane; this is noise next to 15.5 GB/step of weights).
- **top-p threshold**: no sort. Binary search on a prob cutoff tau: fixed 12
  iterations of (sum p_i, count) over {p_i >= tau}; nucleus = {p_i >= tau*}.
  Fixed iteration count keeps it graph-capturable. ~12 MB of logit re-reads.
- **device RNG**: Philox4x32 counter-based, STATELESS -- counter keyed on
  (seed, abs pos from d_P, lane, draw kind, vocab index). No mutable RNG
  buffer to advance: graph replay, prefix-cache restore, ckpt_restore all
  stay consistent for free.
- **sampling = Gumbel-max**: argmax(logit_i/T + G_i) over the nucleus mask,
  G_i from Philox. This is k_argmax_masked plus a noise add -- one new kernel
  cloned from an existing one, no CDF scan, no atomics-order nondeterminism.

Where the logic lives: k_finish_round is NOT extended in place (it must stay
byte-identical for greedy). New pair, launched only in sampled graphs:
- **k_spec_accept** (1 block, serial like k_prep_round): reads the 4 drafts,
  per-lane p(d_next) (from the logsumexp pass + one logit lookup), draws 4
  uniforms, walks the accept chain, writes {n, stop lane, exclude token}.
- **k_sample_stop** (grid): Gumbel-max over the stop lane's logits (base +
  stop*VOCAB indirection via device buffer -- fixed base pointer, graph-safe),
  nucleus- and exclude-token-masked. All 4 accepted: stop lane = slot e, no
  exclusion (the free bonus sample). Finish bookkeeping (h_next copy,
  d_P += n, outcome[]) keys off k_spec_accept's n, not the equality chain.

Exactly ONE full sample per round regardless of where the chain stops.

## 2. MTP draft distribution: greedy drafts for v1

Options:
- **(a) greedy drafts + target-dist acceptance** -- valid (rejection sampling
  is correct for any q, including a delta), zero changes to the 4 sequential
  mtp_forward passes or their captured graphs, accept prob per lane is just
  p(d). Cost: acceptance = E[p(argmax)] sags as temp rises.
- **(b) sampled drafts** -- draft head gets its own softmax + Philox draw per
  pass (4x), q(d_k) stored for the ratio test; min(1, p/q) accepts more at
  high temp. Strictly better acceptance, strictly more machinery.

**v1 ships (a).** Correctness is identical (both sample the exact target);
(a) touches nothing inside the draft passes; and E3 (98.1% Q4/Q8 draft-argmax
agreement) says the draft head is sharp -- (b)'s win only appears at temps
this engine's agentic workloads rarely run. Instrument acceptance-vs-temp in
--stats first; build (b) only if the curve says so.

## 3. Interaction with P7 constrained decoding

Masked sampling = renormalize over legal tokens: the grammar mask applies
BEFORE softmax -- masked logits go to -inf ahead of the max/logsumexp pass, so
normalization and the top-p nucleus are computed over legal tokens only
(masking after normalization leaks mass to illegal tokens and skews the
nucleus cut). Same d_mask_ids slot-0 plumbing as today.

The acceptance-cap path already does the hard part: in-grammar rounds force
n = 1 (d_accept_cap), so sampled in-grammar decode is a single masked sample
from slot 0 -- k_spec_accept sees cap and skips the chain, k_sample_stop runs
masked. Rejection logic never meets the grammar. EOS stays masked until
done(), as now.

## 4. Determinism and gates

- temperature == 0 or absent routes to the existing argmax/argmax_masked
  kernels via an explicit branch: the SAME captured greedy graphs as today,
  sampled kernels never launch, canonical md5 gates stay greedy and bitwise.
- Sampled graphs are a second set of 5 captured perms, selected per request.
- New gates for the sampled path:
  - **seeded identity**: fixed (seed, prompt, params) -> token-identical
    across runs and builds on the same GPU (Philox is deterministic and the
    Gumbel-max reduction has a fixed comparison order like k_argmax).
    Per-GPU-arch canonical, not cross-arch.
  - **statistical**: empirical next-token histogram (>= 512 draws, ~8 probe
    contexts, T = 0.8 / top_p = 0.95) chi-square vs HF reference softmax on
    the same logits protocol as the P2 logit A/B.
  - **spec == non-spec distribution**: rejection sampling guarantees the spec
    path samples the same target as phase 1's plain path; gate it with a
    two-sample chi-square between the two paths' histograms (token-wise match
    is NOT expected -- they consume different draw counts).

## 5. CUDA graph compatibility

Same pattern as P7's mask ids: one device param block {inv_temp, top_p, seed,
flags}, host-rewritten between graph launches, pointers fixed at capture.
Philox counters derive from d_P + lane on device -- nothing else advances.
Fixed launch geometry everywhere (12-step threshold search, fixed grids). No
re-capture on param change; greedy<->sampled graph-set picked per request.

## 6. Cost

New kernels: k_logsumexp3 (5-lane), k_topp_threshold, k_sample_gumbel (clone
of k_argmax_masked + noise), k_spec_accept; plain-path variants of the first
three. Changed: spec_round_launches grows a sampled variant, finish
bookkeeping reads k_spec_accept's n, forward()'s argmax gets a sampled
sibling, server plumbs temperature/top_p/seed through all 3 API shapes.
Untouched: GEMVs, MTP passes, attention, GDN, prefill, greedy graphs.

Perf at temp > 0: added DRAM is ~20-70 MB/round of logit re-reads. Softmax
over vocab reads 1 MB/lane -- argmax cost class, NOT head-GEMV class (the
head GEMV reads the ~1.3 GB lm_head). Kernel-time impact ~1-2% of a ~20 ms
round. The real cost is acceptance: tokens/round falls with temp -- a
distribution-fidelity cost, not an implementation one. Measure and publish
the t/s-vs-temp curve.

## Phased plan

- **Phase 1 -- plain-path sampling**: logsumexp + threshold + Gumbel-max
  kernels, param block, API plumbing, seeded-identity + chi-square gates,
  greedy canonicals re-verified untouched. Validates the samplers without
  spec complexity.
- **Phase 2 -- spec rejection sampling**: k_spec_accept + k_sample_stop in a
  second 5-perm graph set, greedy drafts, --stats acceptance-vs-temp
  telemetry, spec==non-spec distribution gate.
- **Phase 3 -- constrained + sampled**: -inf mask pre-softmax, single-sample
  cap path; gate = zero grammar violations over N sampled tool calls plus
  chi-square vs a masked-renormalized HF reference.
- **Exit criterion (before sampling defaults on anywhere)**: re-run the
  Thunderdome quality A/B and the tool-format drift catalog under the
  production sampling config. Every quality number backing the engine
  today (the 0.786 tie, the five drift modes, acceptance 4.32-4.36
  t/round, the depth-4 gate p(d4|prefix-3)=97.4%) is greedy-no-think
  scoped; temperature moves acceptance AND drift behavior simultaneously,
  so the greedy numbers do not transfer by argument. This is an exit gate
  for the feature, not a follow-up.

## Open questions

- Sampled drafts (2b): at what measured temp does the acceptance curve pay
  for 4 extra softmax+draw passes?
- min-p / top-k: the threshold machinery gives min-p nearly free; top-k needs
  selection -- support or refuse?
- --fast-head (Q4 verify) shifts the target distribution -- own statistical
  reference, or fast-head + sampling unsupported?
- Repetition/presence penalties need a device pass over emitted history --
  deferred; where does the history buffer live re: graphs?
