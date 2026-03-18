# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-gad`
- Focus: `resattn-gad` has now generated the stratified `registry_v5` surface and cleared the repaired calibration slice; the next meaningful move is the first larger `registry_v5` Gemma oracle campaign.
- Experimental status: the prompt-surface expansion worked. `registry_v5` now provides four explicit oracle strata with `256` pilot prompts and `1024` confirm prompts total, and the repaired small calibration stayed positive on both oracle loss recovery and held-out predictiveness with the current best primary-model method. The next question is no longer “can v5 load and run?” It is how the larger stratified campaign behaves.
- Critical reminder: keep the first `registry_v5` campaign method-fixed to the current best primary-model feature source plus `oracle_alpha_logit_vector`.
