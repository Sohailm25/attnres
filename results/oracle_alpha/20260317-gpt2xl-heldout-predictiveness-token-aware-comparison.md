# GPT-2 XL Held-Out Predictiveness Token-Aware Comparison

## Motivation

`resattn-7ve` asked whether the weak held-out predictiveness result was mainly a consequence of overly lossy sequence summaries. The smallest next step was to add minimal token-aware `h_4[t]` summaries, keep feature selection pilot-only, and rerun the confirm check without changing the saved prompt registry.

## Methods

- Model: `gpt2-xl`
- Device: local `mps`
- Collection: `oracle_alpha_phase1_v1`
- Train split: `pilot` (`8` prompts)
- Eval split: `confirm` (`8` prompts)
- Oracle-alpha target: final sequence-level `final_alpha` vector from the development-model runner
- Feature-source candidates, compared on the pilot split only:
  - `mean_pooled_h_1[t]_resid_post_layer_0`
  - `mean_pooled_h_4[t]_resid_post_layer_3`
  - `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`
  - `start_mid_end_h_4[t]_resid_post_layer_3_concat`
  - `mean_pooled_h_1[t]_plus_h_4[t]_concat`
  - `final_token_h_1[t]_plus_h_4[t]_concat`
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
  - `mean_pooled_h_4[t]_resid_post_layer_3`: tuning mean JS `0.2490`
  - `mean_pooled_h_1[t]_plus_h_4[t]_concat`: tuning mean JS `0.2508`
  - `mean_pooled_h_1[t]_resid_post_layer_0`: tuning mean JS `0.2530`
  - `start_mid_end_h_4[t]_resid_post_layer_3_concat`: tuning mean JS `0.2581`
  - `final_token_h_1[t]_plus_h_4[t]_concat`: tuning mean JS `0.3044`
- The selected feature source was `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`
- Confirm-split predictiveness under the selected feature source:
  - `R^2 = -0.2154`
  - mean JS to oracle alpha: `0.2377`
  - predicted-alpha mean improvement over uniform: `-0.0011` nats
  - oracle-alpha mean improvement over uniform: `1.3409` nats
- Relative to the prior mean-pooled `h_4[t]` baseline, the selected token-aware summary improved confirm `R^2` and mean JS modestly, but it did not produce a better mean predicted-loss improvement over uniform

## Limitations

- The pilot split is still only `8` prompts, so feature ranking remains noisy even when confirm is kept clean.
- Every tested candidate again selected the strongest ridge penalty (`100.0`), which suggests the predictor is still operating in a heavily shrunk regime.
- The token-aware summary improved alpha-shape similarity more than routed-loss recovery, so this artifact still does not clear the held-out predictiveness blocker.

## Next Steps

- Keep `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat` as the current best token-aware internal summary, not as a solved predictor.
- Follow `resattn-27f`: compare prompt-level baselines and hybrid prompt-shape plus internal-state summaries on the pilot split only.
- Keep strong oracle-alpha interpretation blocked until the confirm-split predictor becomes materially positive on both alpha similarity and routed-loss metrics.
