# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-o3n`
- Focus: `resattn-o3n` has now decomposed the matched-family donor controls; the next meaningful move is `resattn-0mu`, which expands the matched-family surface before rerunning the donor-arm counterfactual on more prompts.
- Experimental status: the repo now has a sharper but still bounded dynamic-control read. On `tool_breakage_factual_recall_v2`, routed is slightly worse than both explicit within-family and cross-family donor controls on mean tuned KL, but the margins are tiny and the within-family aggregate is carried mainly by the element family.
- Critical reminder: the same-model tool-breakage lane is no longer blocked by baseline quality or by an obviously wrong dynamic control. The remaining question is breadth: whether the narrow donor-arm advantage survives a larger balanced matched-family surface.
