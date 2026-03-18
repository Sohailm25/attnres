# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-0mu`
- Focus: `resattn-0mu` has now landed the expanded balanced `tool_breakage_factual_recall_v3` surface and its first pilot baseline; the next meaningful move is `resattn-cky`, the locked `32`-prompt confirm baseline on that larger surface.
- Experimental status: the larger matched-family pilot did not wash out the aligned same-model breakage signal. The tuned KL delta stayed at `+2.6019` versus `+2.5849` on `v2`, and final-position tuned KL improved to `+3.0324`.
- Critical reminder: the next uncertainty is still breadth, not geometry. The donor-arm result is promising but narrow, so the right next check is the larger `v3` confirm baseline rather than another control redesign.
