ABOUTME: Summarizes the first full stratified `registry_v5` Gemma oracle-alpha campaign on the fixed primary-model method surface.
ABOUTME: Records whether the broader four-stratum prompt expansion strengthens the primary-model oracle and predictiveness story enough to pivot from reruns to analysis.

# Motivation

`resattn-w39` exists because the primary-model Gemma oracle lane had already
cleared positive held-out routed-loss recovery on the saved `registry_v4`
surface, and the most underexplored promising direction was breadth rather than
another predictor redesign. The repo froze `registry_v5` as a stratified prompt
surface with four explicit families:

- factual recall
- reasoning and math
- code and procedural text
- general narrative or expository text

The point of this run was not just “more prompts.” It was to test whether the
current best primary-model oracle path stays positive, useful, and structurally
interesting on a materially broader prompt surface without reopening feature or
target search.

# Methods

- Model: `google/gemma-2-2b`
- Device: local `mps`
- Prompt registry: `prompts/registry_v5.yaml`
- Collection: `oracle_alpha_phase1_v1`
- Split sizes:
  - pilot: `256`
  - confirm: `1024`
- Strata:
  - `stratum_factual_recall`
  - `stratum_reasoning_math`
  - `stratum_code_procedural`
  - `stratum_general_text`
- Fixed oracle method:
  - feature source:
    `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat`
  - target: `oracle_alpha_logit_vector`
  - regularization grid: `1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100`
  - optimization: `20` Adam steps, `lr = 0.1`, `seed = 11`
- Output directory:
  `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/`
- Durability:
  - original launch in `tmux` session `oa-v5-gemma-full`
  - prompt-level oracle checkpoints and feature caches written during the run
  - exact-command rerun executed after completion to verify reuse on the final
    output directory

# Results

- The broad primary-model oracle result stayed clearly positive:
  - mean oracle improvement over uniform: `+1.6299` nats
  - bootstrap interval: `[1.5865, 1.6758]`
  - positive prompts: `1024 / 1024`
- The confirm oracle beat every prereg null on mean loss:
  - `uniform = 4.7341`
  - `random_dirichlet = 9.4581`
  - `magnitude_proportional = 6.9846`
  - `last_layer_only = 20.9150`
  - `optimized = 3.1042`
- Held-out predictiveness strengthened rather than collapsing:
  - selected feature source:
    `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat`
  - selected target: `oracle_alpha_logit_vector`
  - selected ridge penalty: `100.0`
  - predicted mean improvement over uniform: `+0.8292` nats
  - predicted bootstrap interval: `[0.7805, 0.8758]`
  - predicted positive prompts: `958 / 1024`
  - confirm `R^2 = 0.2392`
  - mean JS divergence to oracle alpha: `0.0985`
- The stratum breakdown is now scientifically informative:
  - factual recall:
    - oracle mean improvement `= +2.2488`
    - predicted mean improvement `= +1.6297`
    - predicted positives `= 239 / 256`
    - mean JS `= 0.0580`
  - reasoning and math:
    - oracle mean improvement `= +0.8613`
    - predicted mean improvement `= +0.4848`
    - predicted positives `= 247 / 256`
    - mean JS `= 0.0803`
  - code and procedural text:
    - oracle mean improvement `= +1.4595`
    - predicted mean improvement `= +0.5376`
    - predicted positives `= 238 / 256`
    - mean JS `= 0.1181`
  - general text:
    - oracle mean improvement `= +1.9501`
    - predicted mean improvement `= +0.6645`
    - predicted positives `= 234 / 256`
    - mean JS `= 0.1374`
- Resume reuse worked on the real large output, not just on calibration:
  - original runtime: `2544.10` seconds
  - exact-command rerun against the completed output directory: `319.63`
    seconds
  - the rerun emitted only predictiveness-progress lines, which is consistent
    with reusing saved oracle and feature artifacts rather than recomputing the
    full campaign

# Interpretation

- This is the strongest primary-model oracle artifact in the repo so far.
- The old “best positive result lives on the development model” paper risk is
  now much smaller. The primary Gemma spine is positive on a much broader prompt
  surface and no longer looks like a fragile confirm-split win.
- The broader surface also changes the interpretation problem in a good way:
  the new question is less “does the primary-model signal exist?” and more
  “what grouped or stratum-specific structure explains that signal?”
- Factual recall is currently the clearest high-signal stratum and looks like
  the best bridge between the oracle lane and the bounded Gemma tool-breakage
  lane.

# Limitations

- This is still the current final-output residual-source oracle slice, not the
  full multi-surface claim-bearing oracle analysis.
- The selected ridge penalty remains `100.0`, so the broader success does not
  mean the predictiveness geometry is fully solved in every descriptive sense.
- This artifact does not answer the block-structure question by itself; it only
  supplies a much stronger saved surface for that analysis.

# Next Steps

- Close `resattn-w39`.
- Run stratified primary-model pattern analysis on the saved `registry_v5`
  artifact before launching another expensive oracle campaign.
- Use that saved-artifact analysis to decide whether the next oracle follow-up
  should be factual-recall-focused, broader grouped-structure work, or a shift
  to another major lane.
