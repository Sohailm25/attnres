ABOUTME: Summarizes the first primary-model Gemma oracle-alpha pilot stability suite.
ABOUTME: Records the restart and prompt-perturbation behavior of the saved pilot surface on the primary spine.

# Motivation

`resattn-a7j` followed the first bounded `2`-prompt Gemma oracle-alpha slice.
The question was whether the primary-model oracle lane stays positive and
methodologically usable when scaled to the saved `8`-prompt pilot surface with
restart, paraphrase, and resample checks.

# Methods

- Model: `google/gemma-2-2b`
- Device: local `mps`
- Collection: `oracle_alpha_phase1_v1`
- Split: `pilot`
- Number of prompts: `8`
- Objective: optimize one softmax alpha vector per sequence over the final-output
  residual-source decomposition, then compare aggregate and prompt-matched
  alpha stability under restarts and prompt perturbations
- Nulls reported: `uniform`, `random_dirichlet`, `magnitude_proportional`,
  `last_layer_only`
- Stability settings:
  - restart seeds from the saved control registry
  - saved prompt perturbation: `prompt_paraphrase`
  - prompt resampling: `3` resamples of size `6`
- Optimization settings:
  - `20` Adam steps
  - learning rate `0.1`

# Results

- Base pilot mean improvement over uniform: `1.7613` nats
- Base bootstrap interval: `[1.4843, 2.0403]`
- Restart stability over aggregate `final_alpha` distributions:
  - mean pairwise Jensen-Shannon divergence: `7.63e-08`
  - mean top-4 Jaccard overlap: `1.0000`
  - mean top-1 source agreement: `1.0000`
- Restart prompt-matched stability:
  - mean pairwise Jensen-Shannon divergence: `4.77e-07`
  - mean top-4 Jaccard overlap: `1.0000`
  - mean top-1 source agreement: `0.9500`
- Saved paraphrase perturbation versus the base run:
  - aggregate mean pairwise Jensen-Shannon divergence: `0.0200`
  - aggregate mean top-4 Jaccard overlap: `0.6000`
  - aggregate mean top-1 source agreement: `1.0000`
- Prompt-matched paraphrase stability:
  - mean pairwise Jensen-Shannon divergence: `0.1434`
  - mean top-4 Jaccard overlap: `0.3310`
  - mean top-1 source agreement: `0.3750`
- Prompt resampling versus the base run:
  - aggregate mean pairwise Jensen-Shannon divergence: `0.0151`
  - aggregate mean top-4 Jaccard overlap: `0.4698`
  - aggregate mean top-1 source agreement: `0.5000`
- Prompt-matched resample stability:
  - mean pairwise Jensen-Shannon divergence: `6.64e-07`
  - mean top-4 Jaccard overlap: `1.0000`
  - mean top-1 source agreement: `1.0000`

# Interpretation

- This is a real primary-model pilot stability pass.
- The base pilot improvement remains strongly positive on the primary model.
- Restart stability is effectively exact at both the aggregate and prompt-matched
  levels, which means the Gemma pilot lane is not currently bottlenecked on
  optimizer randomness.
- Paraphrases and prompt resampling move the aggregate and prompt-matched alpha
  summaries materially, which is the same qualitative pattern seen earlier on
  the development model rather than a primary-model collapse.
- The strongest honest read is that the primary-model oracle lane is now beyond
  feasibility and ready for its first held-out structure test.

# Limitations

- This is still an exploratory pilot artifact, not a claim-bearing primary-model
  result.
- The lane still lacks held-out predictiveness and any confirm-split primary-
  model result.
- The current runner still operates on the final-output residual-source
  decomposition rather than the broader multi-surface claim-bearing setup.

# Next Steps

- Close `resattn-a7j` as a successful primary-model stability pass.
- Move the next oracle-alpha step to the first primary-model held-out
  predictiveness check on the saved confirm split.
