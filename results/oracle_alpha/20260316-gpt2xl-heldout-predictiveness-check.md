# GPT-2 XL Held-Out Predictiveness Check

## Motivation

`resattn-53q` was the first real confirm-split identifiability check for the development-model oracle-alpha runner. The goal was to test whether the sequence-level oracle-alpha vectors recovered on the pilot prompts could be predicted out of sample on the confirm prompts from an early hidden-state summary rather than merely described after the fact.

## Methods

- Model: `gpt2-xl`
- Device: local `mps`
- Collection: `oracle_alpha_phase1_v1`
- Train split: `pilot` (`8` prompts)
- Eval split: `confirm` (`8` prompts)
- Oracle-alpha target: final sequence-level `final_alpha` vector from the current development-model runner
- Feature source: mean-pooled `h_1[t]` operationalized as `resid_post` at layer `0`, averaged across token positions
- Predictiveness model: ridge regression
- Ridge tuning: choose the regularization strength by leave-one-out pilot JS divergence over the grid `1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100`
- Oracle-alpha optimization settings:
  - `20` Adam steps
  - learning rate `0.1`
  - seed `11`
- Runner-stage MIB handling: omitted for this development-model prompt slice, with rationale recorded explicitly in the artifact rather than silently ignored

## Results

- Selected ridge regularization strength: `100.0`
- Pilot tuning mean JS divergence: `0.2530`
- Confirm-split predictiveness summary:
  - `R^2 = -0.2456`
  - mean Jensen-Shannon divergence to oracle alpha: `0.2434`
- Confirm-split oracle mean improvement over uniform: `1.3409` nats
- Confirm-split predicted-alpha mean improvement over uniform: `-0.0348` nats
- `4 / 8` confirm prompts improved over uniform under the predicted alpha vectors, and `4 / 8` were worse than uniform

## Limitations

- This is still a tiny development-model split, not the preregistered `100`-sequence feasibility tranche.
- The current feature object is deliberately narrow: one mean-pooled early hidden-state summary rather than a richer per-token or multi-layer feature surface.
- Negative held-out `R^2` on this slice does not falsify the broader oracle-alpha thesis by itself, but it does fail the current predictiveness check for this specific sequence-level feature summary.

## Next Steps

- Treat the current held-out predictiveness check as failed for the mean-pooled `h_1[t]` feature spec.
- Follow up in `resattn-k2e` by comparing richer early-state summaries or alternative prompt-level features on the pilot split only before rerunning the confirm check.
- Keep strong oracle-alpha pattern interpretation blocked until an out-of-sample predictor clears a materially stronger confirm-split result.
