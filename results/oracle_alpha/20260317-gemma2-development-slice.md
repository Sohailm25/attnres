ABOUTME: Summarizes the first bounded primary-model oracle-alpha slice on Gemma-2.
ABOUTME: Records whether the existing runner executes cleanly on the primary spine before any larger primary-model run.

# Motivation

`resattn-7cs` exists because the repo's strongest positive oracle-alpha result
still lived on the development model even after primary-spine reconstruction
readiness was cleared. The smallest honest next question was whether the
existing bounded oracle-alpha runner also executes cleanly on the primary
`google/gemma-2-2b` spine.

# Methods

- Model: `google/gemma-2-2b`
- Device: local `mps`
- Prompt registry: `prompts/registry_v4.yaml`
- Collection: `oracle_alpha_phase1_v1`
- Split: `pilot`
- Exploratory: `true`
- Slice size: `2` prompts
- Optimization settings:
  - `20` Adam steps
  - learning rate `0.1`
  - seed `11`
- Output artifact:
  `results/oracle_alpha/20260317-gemma2-development-slice.json`

# Results

- The bounded primary-model runner executed cleanly without Gemma-specific code
  changes.
- Mean sequence improvement over uniform: `+2.1634` nats on `2` pilot prompts.
- Mean null losses on the same slice:
  - `uniform = 4.3030`
  - `random_dirichlet = 6.6138`
  - `magnitude_proportional = 6.5578`
  - `last_layer_only = 24.7532`
- The runner used `53` residual sources on the primary model:
  embedding plus `26` attention outputs and `26` MLP outputs.

# Limitations

- This is still a `2`-prompt smoke slice and is not claim-bearing.
- The result does not yet clear any primary-model stability or predictiveness
  gate.
- The prompt surface is exploratory pilot only.

# Next Steps

- Close `resattn-7cs` as a successful primary-model feasibility slice.
- Move the next oracle-alpha step to `resattn-a7j`, the primary-model pilot
  stability suite on the saved pilot prompt surface.
