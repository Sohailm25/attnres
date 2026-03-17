# GPT-2 XL Development-Model Oracle-Alpha Slice

## Motivation

The repo needed a real oracle-alpha execution path that consumed the saved prompt registry and control registry by default. The goal of this run was not a claim-bearing result; it was to prove the first development-model harness actually works.

## Methods

- Model: `gpt2-xl`
- Device: local `mps`
- Collection: `oracle_alpha_phase1_v1`
- Split: `pilot`
- Number of prompts: `2`
- Objective: optimize one softmax alpha vector per sequence over the final-output residual-source decomposition
- Nulls reported: `uniform`, `random_dirichlet`, `magnitude_proportional`, `last_layer_only`
- Optimization settings:
  - `20` Adam steps
  - learning rate `0.1`
  - seed `11`

## Results

- Mean sequence improvement over uniform: `1.2141` nats
- Bootstrap interval over the two-sequence mean: `[1.0230, 1.4053]`
- Mean null losses:
  - `uniform = 4.0151`
  - `random_dirichlet = 4.7303`
  - `magnitude_proportional = 5.6835`
  - `last_layer_only = 11.3386`
- Per-prompt optimized loss stayed below uniform on both pilot prompts.

## Limitations

- This is a two-prompt development smoke, not a preregistered Phase 1 feasibility test.
- The runner currently covers the final-output decomposition only, not the broader claim-bearing multi-layer analysis surface.
- No restart stability, resampling, or paraphrase perturbation checks have been run yet.

## Next Steps

- Scale the runner to a larger pilot batch and execute the preregistered stability perturbations.
- Revisit the MIB anchor once the runner can support a controlled sanity task.
