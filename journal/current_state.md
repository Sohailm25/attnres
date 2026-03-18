# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-4ny`
- Focus: `resattn-4ny` has now confirmed the matched-family tool-breakage surface; the next meaningful move is the dynamic-routing counterfactual on `tool_breakage_factual_recall_v2`.
- Experimental status: the repo now has a better-aligned bounded baseline for the Gemma tool-breakage lane. `tool_breakage_factual_recall_v2` beats the old mixed `v1` surface on the KL-primary confirm read while keeping the rank metrics alive on a larger confirm set.
- Critical reminder: the stronger same-model claim is still blocked on the explicit dynamic-routing counterfactual, not on another baseline refresh.
