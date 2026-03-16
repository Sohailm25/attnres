# Current State

**Last updated:** 2026-03-16
**Updated by:** codex-gpt5
**Status:** in_progress
**Current phase:** Phase 0 - Scaffold and prereg alignment

## Active Thesis Lock

- `known`: this repo is now a standalone local workspace on a MacBook Pro.
- `known`: oracle-alpha must be treated as an upper bound on the routing signal available in standard architectures, not as a direct stand-in for co-adaptation inside trained Attention Residuals models.
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

1. Finish scaffold verification and keep the structure test green.
2. Convert the master research document into explicit local prereg gates.
3. Build the first implementation slice for Phase 1 oracle-alpha infrastructure on local MPS.
4. Preserve the co-adaptation caveat in all future experiment docs and results summaries.

## Phase 1 Gate

The first execution gate remains the preregistered one from `research/decision-matrix.md`:

- show mean cross-entropy reduction greater than `0.01` nats over uniform routing on at least `100` sequences
- require `p < 0.01`
- compare against random, magnitude-proportional, and last-layer-only nulls
