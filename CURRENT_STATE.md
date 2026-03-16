# Current State

**Last updated:** 2026-03-16
**Updated by:** codex-gpt5
**Status:** in_progress
**Current phase:** Phase 0 - Scaffold, prereg alignment, and methodology hardening

## Active Thesis Lock

- `known`: this repo is now a standalone local workspace on a MacBook Pro.
- `known`: oracle-alpha must be treated as an upper bound on the routing signal available in standard architectures, more precisely as the routing signal recoverable from fixed standard-model representations, and as a lower bound on the benefit of depth routing once routing and computation can co-adapt.
- `known`: the scaffold intentionally removes all Modal assumptions from the source `braindstorms` experiment.
- `known`: the required lanes are now locked into the workspace structure:
  - oracle-alpha against preregistered nulls
  - Figure 8 validation
  - tool-breakage demonstration
  - softmax-constrained vs unconstrained vs top-k routing
  - `w_l` analog geometry for learned router queries
  - block-structure test around the `8 clusters` hypothesis
  - safety analysis around refusal or honesty-related routing differences
- `known`: the local methodology audit surfaced non-trivial implementation hazards that must remain fixed in all future code:
  - no Ward linkage directly on Jensen-Shannon distances
  - uniform routing should reconstruct the original logits after final normalization, not `logits / L`
  - exact per-source routed-logit decomposition must use the shared final normalization factor from the full mixture
  - claim-bearing significance should default to the sequence-level unit
  - `resid_post` is insufficient for final Figure 8 claims; use sublayer outputs
- `known`: the local paper cache now exists under `background-work/papers/files` and is indexed in `background-work/papers/DOWNLOAD_MANIFEST.md`
- `known`: a second red-team review identified four additional publishability risks that must stay fixed:
  - tool-breakage cannot rely on raw logit-lens monotonicity as the vanilla baseline
  - claim-bearing work needs a pilot/confirmatory split
  - router input wording must stay explicitly per-token, e.g. `h_1[t]`
  - refusal-feature discovery must precede safety-lane causal claims

## Immediate Next Steps

1. Publicly pre-register the analysis plan on LessWrong before claim-bearing runs.
2. Freeze dependencies before claim-bearing scientific execution.
3. Define and save the pilot/confirmatory split before any method-tuning runs.
4. Build the first implementation slice for Phase 1 oracle-alpha infrastructure on local MPS.
5. Start with reconstruction and cache-validity tests before any claim-bearing oracle-alpha optimization.

## Phase 1 Gate

The first execution gate remains the preregistered one from `research/decision-matrix.md`:

- show mean cross-entropy reduction greater than `0.01` nats over uniform routing on at least `100` sequences
- require `p < 0.01`
- compare against random, magnitude-proportional, and last-layer-only nulls
- on the small-scale validation tranche, require `d > 0.2` versus the random baseline before scaling up

## Operational Locks

- `known`: Figure 8 tests must use operationalized metrics:
  - locality score
  - `alpha_0` embedding persistence across depth
  - `Entropy(pre-attn) > Entropy(pre-MLP)`
  - off-diagonal peaks above `2/L`
- `known`: tool-breakage requires a non-monotonic logit-lens demonstration on factual recall, with a target of non-monotonic curves on `>50%` of prompts before making a strong breakage claim.
- `known`: raw logit lens is not assumed monotonic in the vanilla model; the strong tool-breakage claim requires additional instability under routing relative to the original-model baseline and a tuned-lens-aware comparison.
- `known`: router training success is not just "it trains"; the local target gate is `R^2 > 0.5` when approximating oracle-alpha.
- `known`: clustering must be informative enough to clear `silhouette > 0.2` before we claim task-structured routing.
- `known`: if hierarchical clustering is run on Jensen-Shannon distances directly, use average or complete linkage rather than Ward.
- `known`: any paired t-test gate is interpreted over per-sequence mean deltas unless a stronger dependence-aware method is written down first.
- `known`: authored control documents and the local paper cache passed a final existence audit on 2026-03-16.
- `known`: claim-bearing prompts and thresholds must come from a pilot/confirmatory split rather than one blended prompt pool.
