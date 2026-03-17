# GPT-2 XL Held-Out Predictiveness Loss-Aware Target Comparison

## Motivation

`resattn-xaa` followed the compressed-target tradeoff from `resattn-3ns`. The bounded question was whether the earlier target disagreement mostly reflected a bad tuning rule. The smallest direct fix was to keep the current best feature surface fixed, compare the existing raw-simplex, full-logit, and compressed-logit targets on the pilot split, and tune by predicted routed-loss improvement over uniform rather than by descriptive alpha metrics alone.

## Methods

- Model: `gpt2-xl`
- Device: local `mps`
- Collection: `oracle_alpha_phase1_v1`
- Train split: `pilot` (`8` prompts)
- Eval split: `confirm` (`8` prompts)
- Feature source: `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`
- Candidate targets:
  - `oracle_alpha_vector`
  - `oracle_alpha_logit_vector`
  - `oracle_alpha_depth_type_band_logit_vector`
- Predictiveness model: ridge regression
- Tuning rule:
  - leave-one-out pilot selection over targets and `lambda in {1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100}`
  - primary metric: mean predicted routed-loss improvement over uniform
  - secondary metric: mean JS divergence to oracle alpha
- Oracle-alpha optimization settings:
  - `20` Adam steps
  - learning rate `0.1`
  - seed `11`

## Results

- Loss-aware pilot tuning still selected the raw-simplex target:
  - selected target: `oracle_alpha_vector`
  - selected ridge penalty: `100.0`
  - pilot tuning mean predicted improvement over uniform: `+0.0990` nats
  - pilot tuning mean JS: `0.2435`
- The other pilot candidates were close but slightly worse on the primary metric:
  - full-source logit target: `+0.0952` nats, mean JS `0.2544`, `lambda=100.0`
  - compressed depth-type-band logit target: `+0.0923` nats, mean JS `0.2465`, `lambda=100.0`
- Confirm performance for the selected target reverted to the earlier raw-simplex token-aware result:
  - confirm `R^2 = -0.2154`
  - confirm mean JS `= 0.2377`
  - predicted mean improvement over uniform `= -0.0011` nats
  - `4 / 8` confirm prompts improved over uniform
  - oracle mean improvement over uniform remained `+1.3409` nats
- This means the loss-aware tuning change landed methodologically, but it did not clear the blocker.

## Limitations

- The pilot surface is still only `8` prompts, so even the more aligned tuning metric can select a path that fails to generalize.
- This artifact evaluates the selected candidate on confirm, not every target candidate under a matched confirm rerun in one file.
- The result therefore isolates the effect of the new pilot selection rule more than it settles the broader target-object question.

## Next Steps

- Treat this as a real negative result for the “selection metric alone fixes the tradeoff” hypothesis.
- Follow `resattn-7mb`: test a more fundamental predictiveness redesign rather than another target-selection tweak.
- The current best bets are a materially larger saved pilot surface, a token/span-level routing target, or another lower-variance target object that does not assume one full alpha vector per sequence is the right supervised object.
