# GPT-2 XL Held-Out Predictiveness Feature Comparison

## Motivation

`resattn-k2e` followed the failed first held-out predictiveness check by asking a narrower question: does a slightly richer sequence-level early-state summary improve the confirm-split predictor without touching the saved prompt registry or leaking the confirm split into feature selection?

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
  - `mean_pooled_h_4[t]_resid_post_layer_3`: tuning mean JS `0.2490`
  - `mean_pooled_h_1[t]_plus_h_4[t]_concat`: tuning mean JS `0.2508`
  - `mean_pooled_h_1[t]_resid_post_layer_0`: tuning mean JS `0.2530`
  - `final_token_h_1[t]_plus_h_4[t]_concat`: tuning mean JS `0.3044`
- The selected feature source was `mean_pooled_h_4[t]_resid_post_layer_3`
- Confirm-split predictiveness under the selected feature source:
  - `R^2 = -0.2314`
  - mean JS to oracle alpha: `0.2409`
  - predicted-alpha mean improvement over uniform: `-0.0010` nats
  - oracle-alpha mean improvement over uniform: `1.3409` nats
- Relative to the original mean-pooled `h_1[t]` baseline, the selected `h_4[t]` summary slightly improved both mean JS and predicted loss, but it did not flip the held-out check into a clearly positive result

## Limitations

- The pilot split is still only `8` prompts, so feature-source ranking is noisy even when confirm is kept clean.
- All candidate summaries still preferred the strongest tested ridge penalty (`100.0`), which suggests the current feature family remains weak for the `98`-dimensional alpha target.
- The improvement over the `h_1[t]` baseline is real but small; this artifact does not clear the predictiveness blocker.

## Next Steps

- Treat mean-pooled `h_4[t]` as the best tested internal-state summary so far, not as a locked solution.
- Move to `resattn-7ve`: token-aware or prompt-level feature surfaces, still selected on the pilot split only.
- Keep strong oracle-alpha interpretation blocked until a confirm-split predictor becomes materially positive.
