# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-w39`
- Focus: `resattn-w39` has now completed the first full stratified `registry_v5` Gemma oracle campaign; the next meaningful move is saved-artifact structure analysis, not another immediate scale-up.
- Experimental status: the larger prompt-surface expansion paid off. `registry_v5` stayed strongly positive on both oracle loss recovery and held-out predictiveness at `256 / 1024`, alpha-shape recovery improved to `R^2 = 0.2392`, and the new stratum split is informative rather than cosmetic. Factual recall is the strongest and cleanest stratum; general text is the loosest on alpha shape.
- Critical reminder: do not reopen feature or target sweeps on the oracle lane before extracting the grouped and stratum-conditioned structure from the saved `registry_v5` artifact.
