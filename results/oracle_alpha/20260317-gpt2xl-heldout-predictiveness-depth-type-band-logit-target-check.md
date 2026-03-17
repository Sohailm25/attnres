# GPT-2 XL Held-Out Predictiveness Depth-Type-Band Logit Target Check

## Motivation

`resattn-3ns` asked for the next bounded redesign after the mixed full-source logit-target result. The smallest non-speculative compressed target was to collapse the `98`-source alpha vector into deterministic depth-band-by-source-type groups, predict that lower-dimensional object in logit coordinates, and then lift it back to full-source alpha using train-only within-group templates.

## Methods

- Model: `gpt2-xl`
- Device: local `mps`
- Collection: `oracle_alpha_phase1_v1`
- Train split: `pilot` (`8` prompts)
- Eval split: `confirm` (`8` prompts)
- Oracle-alpha target:
  - training target: `oracle_alpha_depth_type_band_logit_vector`
  - groups: `embed`, `pos_embed`, and depth-band (`early`, `mid`, `late`) crossed with source type (`attn`, `mlp`)
  - lift-back rule: decode predicted group logits to a group simplex, then distribute each group mass across sources using train-split average within-group templates
- Feature source: `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`
- Predictiveness model: ridge regression
- Tuning rule:
  - single feature source
  - leave-one-out pilot tuning over `1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100`
  - primary metric `r_squared`
  - secondary metric `mean_js_divergence`
- Oracle-alpha optimization settings:
  - `20` Adam steps
  - learning rate `0.1`
  - seed `11`

## Results

- The selected ridge penalty dropped from the previous `100.0` saturation to `0.0001`
- Pilot leave-one-out tuning improved relative to the full-source logit target:
  - compressed target: `R^2 = -0.3340`, mean JS `= 0.2465`
  - full-source logit target: `R^2 = -0.4431`, mean JS `= 0.2544`
- Confirm descriptive alpha-recovery metrics also improved relative to the full-source logit target and nearly returned to the earlier raw-simplex token-aware baseline:
  - compressed depth-type-band logit target: `R^2 = -0.2241`, mean JS `= 0.2380`
  - full-source logit target: `R^2 = -0.3291`, mean JS `= 0.2457`
  - earlier raw-simplex token-aware baseline: `R^2 = -0.2154`, mean JS `= 0.2377`
- The routed-loss metric did not survive the compression:
  - compressed depth-type-band logit target: predicted mean improvement over uniform `= -0.0031` nats
  - full-source logit target: predicted mean improvement over uniform `= +0.0693` nats
  - earlier raw-simplex token-aware baseline: `-0.0011` nats
  - oracle-alpha mean improvement over uniform: `+1.3409` nats
- Only `3 / 8` confirm prompts improved over uniform under the compressed target, versus `5 / 8` for the full-source logit target

## Limitations

- The prompt split is still only `8 / 8`, so target compression may be helping descriptive stability simply by reducing variance rather than by recovering a truer routing object.
- The lift-back path uses train-only within-group templates, which is principled enough for a bounded comparison but still imposes a coarse structure on the final full-source alpha prediction.
- Because routed loss is still the primary claim-bearing objective for this lane, the compressed target does not clear the blocker even though its descriptive alpha metrics look better.

## Next Steps

- Treat this as a real comparison result, not a pass.
- Follow `resattn-xaa`: make target and regularization selection explicitly loss-aware when comparing predictiveness paths, while keeping `R^2` and mean JS as secondary diagnostics.
- Keep strong oracle-alpha interpretation blocked until a held-out predictor is positive on routed loss without collapsing the descriptive alpha metrics into an obviously worse regime.
