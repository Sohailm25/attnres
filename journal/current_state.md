# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-cky`
- Focus: `resattn-cky` has now landed the locked `32`-prompt confirm baseline on `tool_breakage_factual_recall_v3`; the next meaningful move is `resattn-4g2`, the larger donor-arm counterfactual on that same surface.
- Experimental status: the larger balanced confirm read slightly strengthened the aligned same-model breakage story rather than washing it out. The tuned KL delta rose to `+2.9127`, while the relative rank metrics stayed alive on the 32-prompt surface.
- Critical reminder: the next uncertainty is now the donor-arm control on the broader confirm set, not whether the larger matched-family surface itself is viable.
