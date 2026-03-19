# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: the broad factual frame audit is done, the simple aggregation comparison is done, the width-only capacity comparison is done, the first router-family comparison is done, the supervision-granularity audit is done, the richer-supervision comparison is done, the family rerun under all-token supervision is now done, and the residual author `novel_title` split is bounded as a lexical-surface sidecar. The main active implementation next step is `resattn-afu`, the bounded tokenwise teacher-target comparison against the retained all-token MLP baseline.
- Experimental status: the center of gravity stays on the primary Gemma oracle lane. The strongest live story is still primary-model effective depth mixture plus factual family-plus-prompt-frame-conditioned route modes on the strongest stratum. Tool-breakage is frozen at a truthful boundary, Figure 8 stays frozen, and the Phase 6 pilot now has a better baseline: `all_tokens_target_mse` materially improved held-out fit and reopened the family question just enough for the widened MLP to beat the linear head, but the gain stayed modest and the lane is still below readiness.
- Critical reminder: do not drift back to pooled reruns, donor-remap-by-inertia, moon redesigns on the old `v4` surface, frozen Figure 8 rescue work, more arbitrary span masks, another blind width sweep under the old sequence objective, or another blind head-family tweak without moving onto the improved all-token supervision baseline first.
