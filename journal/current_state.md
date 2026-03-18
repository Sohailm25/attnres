# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-qww`
- Focus: `resattn-qww` has now run the matched-family dynamic-routing counterfactual; the next meaningful move is `resattn-o3n`, which splits the donor controls into explicit within-family and cross-family arms.
- Experimental status: the repo now has a stronger aligned tool-breakage baseline plus a mixed but sharper dynamic-control result. On `tool_breakage_factual_recall_v2`, routed remains much worse than the fixed-alpha control, but the current cyclic prompt-permuted control nearly matches routed on mean tuned KL while routed stays worse on final-position tuned KL.
- Critical reminder: the stronger same-model claim is still blocked on prompt-specific dynamic-control evidence, not on baseline quality. The current prompt-permuted control is already within-family on `12 / 16` prompts, so the next read should separate within-family and cross-family donor effects instead of rerunning the same aggregate control.
