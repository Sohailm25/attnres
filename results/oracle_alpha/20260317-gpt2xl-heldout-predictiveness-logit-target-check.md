# GPT-2 XL Held-Out Predictiveness Logit-Target Check

## Motivation

`resattn-qq2` addressed the strongest design-review critique of the current predictiveness path: the runner was fitting ridge directly to simplex-valued oracle-alpha vectors in unconstrained Euclidean coordinates and only clipping and renormalizing afterward. The smallest next step was to keep the current best feature source fixed and change only the target geometry.

## Methods

- Model: `gpt2-xl`
- Device: local `mps`
- Collection: `oracle_alpha_phase1_v1`
- Train split: `pilot` (`8` prompts)
- Eval split: `confirm` (`8` prompts)
- Oracle-alpha target:
  - training target changed from raw sequence-level `final_alpha` vectors to `oracle_alpha_logit_vector`
  - evaluation stayed on recovered simplex alpha distributions
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

- The selected ridge penalty stayed at `100.0`
- Pilot leave-one-out tuning on the constrained target remained weak:
  - `R^2 = -0.4431`
  - mean JS `= 0.2544`
- Confirm-split descriptive alpha-recovery metrics got worse relative to the previous raw-simplex token-aware baseline:
  - constrained logit target: `R^2 = -0.3291`, mean JS `= 0.2457`
  - previous raw-simplex baseline: `R^2 = -0.2154`, mean JS `= 0.2377`
- The routed-loss metric moved in the direction we actually care about:
  - predicted-alpha mean improvement over uniform: `+0.0693` nats
  - previous raw-simplex baseline: `-0.0011` nats
  - oracle-alpha mean improvement over uniform: `+1.3409` nats
- `5 / 8` confirm prompts improved over uniform under the predicted alphas
- The constrained target therefore helped routed-loss recovery without improving alpha-shape recovery

## Limitations

- The prompt split is still only `8 / 8`, so the predictor remains badly sample-limited relative to the `98`-source target.
- The selected ridge penalty still saturates at `100.0`, so the fit remains in a heavy-shrinkage regime.
- Because the descriptive alpha metrics regressed while the loss metric improved, the repo still does not have a single clean held-out success criterion for this lane.

## Next Steps

- Treat this as partial progress, not a cleared blocker.
- Follow `resattn-3ns`: compare at least one compressed target that reduces target dimensionality while preserving the sequence-level confirmatory unit.
- Keep strong oracle-alpha interpretation blocked until a held-out predictor improves routed-loss recovery without collapsing the descriptive alpha metrics into a clearly worse regime.
