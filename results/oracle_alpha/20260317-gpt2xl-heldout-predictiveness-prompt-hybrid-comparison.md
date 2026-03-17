# GPT-2 XL Held-Out Predictiveness Prompt-Hybrid Comparison

## Motivation

`resattn-27f` asked whether the current held-out predictiveness blocker was just missing prompt-level information. The smallest next step was to compare simple prompt-shape baselines, a mean token-embedding baseline, and hybrids that appended those prompt-level features to the current best token-aware `h_4[t]` summary.

## Methods

- Model: `gpt2-xl`
- Device: local `mps`
- Collection: `oracle_alpha_phase1_v1`
- Train split: `pilot` (`8` prompts)
- Eval split: `confirm` (`8` prompts)
- Oracle-alpha target: final sequence-level `final_alpha` vector from the development-model runner
- Feature-source candidates, compared on the pilot split only:
  - `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`
  - `prompt_shape_scalar_features_v1`
  - `mean_pooled_token_embedding`
  - `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_prompt_shape_scalar_features_v1_concat`
  - `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat`
- Selection rule: choose the feature source with the lowest leave-one-out pilot mean JS divergence
- Predictiveness model: ridge regression
- Ridge tuning: pilot-only leave-one-out JS over the grid `1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100`
- Oracle-alpha optimization settings:
  - `20` Adam steps
  - learning rate `0.1`
  - seed `11`

## Results

- Pilot-only feature comparison:
  - `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`: tuning mean JS `0.2435`
  - `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_prompt_shape_scalar_features_v1_concat`: tuning mean JS `0.2436`
  - `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat`: tuning mean JS `0.2444`
  - `prompt_shape_scalar_features_v1`: tuning mean JS `0.2483`
  - `mean_pooled_token_embedding`: tuning mean JS `0.2492`
- The selected feature source stayed `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`
- Confirm-split predictiveness therefore stayed unchanged:
  - `R^2 = -0.2154`
  - mean JS to oracle alpha: `0.2377`
  - predicted-alpha mean improvement over uniform: `-0.0011` nats
  - oracle-alpha mean improvement over uniform: `1.3409` nats
- The prompt-level and hybrid candidates were not totally uninformative, but none of them improved enough to displace the current token-aware baseline

## Limitations

- The pilot split is still only `8` prompts, so small ranking differences remain noisy.
- Every tested candidate again selected the strongest ridge penalty (`100.0`), which suggests the held-out predictor is still operating in a heavily shrunk regime.
- Because the selected feature source did not change, this artifact does not move the core blocker forward; it mainly narrows what the blocker is not.

## Next Steps

- Treat this as evidence against “we only needed simple prompt-level or hybrid features.”
- Follow `resattn-23p`: review whether the held-out predictiveness setup itself is the problem, focusing on the sequence-level alpha target, ridge-regression family, prompt scale, and pilot JS tuning objective.
- Keep strong oracle-alpha interpretation blocked until the confirm-split predictor improves on the routed-loss metric rather than only descriptive alpha similarity.
