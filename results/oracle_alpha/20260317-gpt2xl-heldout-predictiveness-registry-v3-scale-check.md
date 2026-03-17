# GPT-2 XL Held-Out Predictiveness Registry V3 Scale Check

## Motivation

`resattn-0vx` followed the positive `registry_v2` pilot-expansion result. The bounded question was whether that improvement would survive a meaningfully larger saved pilot and confirm surface without changing the feature source, target candidates, or tuning rule.

## Methods

- Model: `gpt2-xl`
- Device: local `mps`
- Prompt registry: `prompts/registry_v3.yaml`
- Train split: `pilot` (`32` prompts)
- Eval split: `confirm` (`16` prompts)
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

- The larger split kept the logit target selected:
  - selected target: `oracle_alpha_logit_vector`
  - selected ridge penalty: `100.0`
  - pilot tuning mean predicted improvement over uniform: `+0.1012` nats
  - pilot tuning mean JS: `0.2315`
- The other pilot candidates stayed worse on the primary metric:
  - raw simplex target: `+0.0838` nats, mean JS `0.2259`, `lambda=100.0`
  - compressed depth-type-band logit target: `+0.0429` nats, mean JS `0.2212`, `lambda=100.0`
- Held-out confirm performance stayed positive and strengthened in absolute terms:
  - confirm `R^2 = -0.1954`
  - confirm mean JS `= 0.2421`
  - predicted mean improvement over uniform `= +0.1277` nats
  - oracle mean improvement over uniform `= +1.2726` nats
  - `12 / 16` confirm prompts improved over uniform
- Relative to `registry_v2`, the main headline is that the logit path remained selected and the held-out routed-loss gain stayed positive on a larger saved split.

## Limitations

- The result is much stronger than the earlier `8 / 8` and `16 / 8` checks, but it is still below the preregistered `100`-sequence Phase 1 scale.
- Confirm mean JS got slightly worse than the `16 / 8` run even though held-out routed loss improved, so descriptive alpha similarity and routed-loss recovery remain only partially aligned.
- The prompt surface is still a local inline collection, not a benchmark-backed evaluation regime.

## Next Steps

- Treat the `oracle_alpha_logit_vector` path as the current best predictiveness path to scale.
- Follow `resattn-9jq`: scale this same method toward the prereg-sized oracle-alpha gate rather than switching targets again.
- Keep strong oracle-alpha interpretation blocked until the larger held-out surface and the preregistered Phase 1 gate are both cleared.
