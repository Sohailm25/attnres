ABOUTME: Summarizes the repaired small-slice calibration for the generated stratified Gemma oracle registry_v5 surface.
ABOUTME: Records whether registry_v5 works end to end before the first larger primary-model campaign launch.

# Motivation

`resattn-gad` exists to turn the frozen `resattn-rh0` plan into a real next
oracle surface. The first requirement was not a claim-bearing run. It was a
small calibration on the generated `registry_v5` prompt surface to verify:

- deterministic registry generation
- four-stratum prompt coverage on a small slice
- checkpoint and summary artifact writing
- resume reuse on the saved output directory

The first calibration attempt exposed a real generator bug in the general-text
prompt bank (`a lemons`, `a optician`). This `v2` artifact supersedes that first
attempt after repairing the prompt phrases and regenerating `prompts/registry_v5.yaml`.

# Methods

- Model: `google/gemma-2-2b`
- Device: local `mps`
- Prompt registry: `prompts/registry_v5.yaml`
- Collection: `oracle_alpha_phase1_v1`
- Slice size:
  - pilot `8`
  - confirm `8`
- Ordering: round-robin across the four strata, so the slice covers:
  - factual recall
  - reasoning and math
  - code and procedural text
  - general narrative or expository text
- Fixed oracle method:
  - feature source:
    `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat`
  - target: `oracle_alpha_logit_vector`
  - regularization grid: `1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100`
  - optimization: `20` Adam steps, `lr = 0.1`, `seed = 11`
- Output directory:
  `results/oracle_alpha/20260318-gemma2-registry-v5-calibration-v2/`

# Results

- The repaired `registry_v5` surface ran end to end without code changes to the
  oracle runner or campaign path.
- The small confirm oracle result stayed strongly positive:
  - mean improvement over uniform: `+1.4363` nats
  - positive prompts: `8 / 8`
- Held-out predictiveness on the same small slice also stayed positive:
  - predicted mean improvement over uniform: `+0.5896` nats
  - predicted positive prompts: `5 / 8`
  - mean JS divergence to oracle alpha: `0.1195`
  - `R^2 = -0.0454`
  - selected ridge penalty: `100.0`
- The saved manifest and checkpoint directories were created as intended:
  - `campaign_manifest.json`
  - `checkpoints/oracle_runs/`
  - `checkpoints/feature_vectors/`
  - `predictiveness_progress.json`
  - `predictiveness_summary.json`
- Resume reuse worked on the repaired surface:
  - rerunning the exact command on the same output directory completed in
    `16.94` seconds
  - the rerun emitted only predictiveness-progress lines, which is consistent
    with reusing the saved oracle and feature artifacts rather than recomputing
    them

# Interpretation

- `registry_v5` is operationally ready for a larger primary-model Gemma oracle
  campaign.
- The small-slice result is encouraging enough to justify scaling:
  the oracle signal stayed clearly positive across all four strata, and the
  fixed primary-model predictiveness path stayed positive rather than collapsing
  on the broader surface.
- The repaired prompt-bank issue was worth catching here. It was a real quality
  bug, but it looks like a prompt-generation defect rather than a scientific
  defect in the oracle lane.

# Limitations

- This is a calibration artifact, not a claim-bearing confirm run.
- The slice is tiny and uses only `8 / 8` prompts.
- The selected ridge penalty remains `100.0` on this tiny slice, so the
  calibration does not resolve the larger predictiveness-shape question by
  itself.

# Next Steps

- Close `resattn-gad` once the generator, tests, and calibration artifact land.
- Launch the first larger `registry_v5` primary-model Gemma oracle campaign on
  the fixed feature-source and target path.
