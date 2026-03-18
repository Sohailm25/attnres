# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-rh0`
- Focus: `resattn-rh0` is now the planning freeze for the next larger primary-model Gemma oracle campaign; after it lands, the immediate execution step is `resattn-gad`
- Experimental status: the repo has enough positive primary-model oracle signal that the next clean leverage point is prompt-surface expansion rather than more method tuning. The frozen plan is a stratified `registry_v5` with four explicit strata, `64` pilot prompts and `256` confirm prompts per stratum, plus a required calibration slice before any full launch.
- Critical reminder: the next scale-up should change the prompt surface only. Do not reopen target-family or feature-family redesign inside the first `registry_v5` campaign.
