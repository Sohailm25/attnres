# Second Review Readiness

## Purpose

This memo records the second-pass review against the full local `research/` directory and the more detailed reference scaffold on the Mac mini.

## Findings Resolved

1. Framing is now explicit:
   - oracle-alpha is an upper bound on the routing signal recoverable from fixed standard-model representations
   - oracle-alpha is a lower bound on the benefit of depth routing once routing and computation can co-adapt
2. The prereg now operationalizes Figure 8 rather than naming it abstractly.
3. The prereg now defines the comparison regimes more concretely.
4. The tool-breakage lane now has a concrete success condition.
5. Router training now names the local architecture and readiness target.
6. The scaffold now includes adapted `RESEARCH_POSITIONING.md` and `PROPOSAL_REVIEW.md` files, matching the source experiment's rigor more closely.

## Remaining Non-Structural Preconditions Before Claim-Bearing Implementation

- Keep the local prereg in `history/PREREG.md` as the claim-bearing prereg artifact.
- Freeze the dependency set used for scientific runs.
- Keep the `.venv` versus conda divergence explicit; `.venv` is the local workspace standard.

## Readiness Verdict

The experiment shape is now structurally aligned with the local research documents. Implementation can begin once the environment freeze and other active Phase 0 blockers are done.
