ABOUTME: Summarizes the first primary-model Gemma held-out oracle-alpha predictiveness check on the saved prereg-scale split.
ABOUTME: Records whether the primary spine clears held-out routed-loss recovery and how that relates to the weaker alpha-shape metrics.

# Motivation

`resattn-js8` exists because the repo's strongest oracle-alpha result had still
been carrying a development-model-only risk even after Gemma cleared exact
reconstruction, a bounded oracle slice, and the saved pilot stability suite.
The next honest question was whether the primary `google/gemma-2-2b` spine also
stays positive on the saved `96 / 128` pilot/confirm held-out predictiveness
check rather than only looking good in exploratory pilot artifacts.

# Methods

- Model: `google/gemma-2-2b`
- Device: local `mps`
- Prompt registry: `prompts/registry_v4.yaml`
- Collection: `oracle_alpha_phase1_v1`
- Train split: `pilot` (`96` prompts)
- Eval split: `confirm` (`128` prompts)
- Oracle target family: `oracle_alpha_logit_vector`
- Predictiveness model: ridge regression with the saved control-plan metrics
  (`mean_predicted_improvement_over_uniform` primary, `mean_js_divergence`
  secondary)
- Candidate feature sources:
  - `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`
  - `prompt_shape_scalar_features_v1`
  - `mean_pooled_token_embedding`
  - `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_prompt_shape_scalar_features_v1_concat`
  - `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat`
- Regularization grid: `1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100`
- Oracle-alpha optimization settings:
  - `20` Adam steps
  - learning rate `0.1`
  - seed `11`
- MIB handling: omitted for this custom prompt slice because it is not a
  benchmark-compatible task surface

# Results

- Selected feature source:
  `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat`
- Selected regularization strength: `100.0`
- Pilot tuning summary:
  - mean predicted improvement over uniform: `0.8466` nats
  - mean JS divergence: `0.1481`
- Confirm predicted-alpha result:
  - mean improvement over uniform: `+1.0428` nats
  - bootstrap interval: `[0.9495, 1.1331]`
  - positive prompts: `123 / 128`
- Confirm oracle-alpha result:
  - mean improvement over uniform: `+2.5525` nats
  - bootstrap interval: `[2.4603, 2.6442]`
  - positive prompts: `128 / 128`
- Confirm alpha-recovery metrics:
  - `R^2 = -0.0632`
  - mean JS divergence to oracle alpha: `0.1478`
- Confirm oracle mean losses:
  - `uniform = 6.3916`
  - `random_dirichlet = 10.5157`
  - `magnitude_proportional = 7.3886`
  - `last_layer_only = 20.4461`
  - `optimized = 3.8391`
- The confirm oracle beat every prereg null on all `128` prompts.

# Interpretation

- This is a real primary-model held-out predictiveness pass on the repo's
  primary metric, not just another bounded feasibility slice.
- The biggest paper-shape weakness from the external review is materially
  reduced: the strongest positive oracle-alpha story is no longer confined to
  `gpt2-xl`.
- The result is still mixed in one important way: routed-loss recovery is
  strong, but the Euclidean alpha-shape metrics remain weaker than the loss
  story (`R^2 < 0` and `lambda = 100.0`).
- The selected hybrid source suggests that the primary model benefits from
  combining the current best token-aware internal summary with prompt-level
  embedding information rather than using either surface alone.

# Limitations

- This remains the current final-output residual-source development slice, not
  the broader claim-bearing multi-surface oracle analysis.
- MIB is still omitted because this prompt registry is not a benchmark-
  compatible task surface.
- The current ridge helper is operationally inefficient in the `n << d` regime;
  this bounded run completed cleanly, but it took about an hour and produced no
  intermediate progress artifact during the numerical sweep.

# Next Steps

- Close `resattn-js8` as a successful primary-model held-out predictiveness
  artifact with mixed alpha-shape metrics.
- Move the next oracle-alpha question to `resattn-2sb`, the primary-model Gemma
  prereg-scale pattern analysis on the saved oracle artifact.
- Track the new `n << d` ridge-runtime bottleneck separately in `resattn-b4q`
  rather than treating it as a scientific weakness in the result itself.
