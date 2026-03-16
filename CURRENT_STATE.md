# Current State

**Last updated:** 2026-03-16
**Updated by:** codex-gpt5
**Status:** in_progress
**Current phase:** Phase 0 - Scaffold and prereg alignment

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

## Immediate Next Steps

1. Keep the scaffold verification green after the second-pass thesis alignment changes.
2. Publicly pre-register the analysis plan on LessWrong before claim-bearing runs.
3. Build the first implementation slice for Phase 1 oracle-alpha infrastructure on local MPS.
4. Preserve the upper-bound and lower-bound framing split in all future docs and summaries.

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
- `known`: router training success is not just "it trains"; the local target gate is `R^2 > 0.5` when approximating oracle-alpha.
- `known`: clustering must be informative enough to clear `silhouette > 0.2` before we claim task-structured routing.
