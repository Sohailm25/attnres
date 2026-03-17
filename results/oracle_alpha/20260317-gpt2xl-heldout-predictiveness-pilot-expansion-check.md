# GPT-2 XL Held-Out Predictiveness Pilot Expansion Check

## Motivation

`resattn-7mb` tested the least invasive redesign after `resattn-xaa`: keep the loss-aware tuning rule, feature source, and confirm set fixed, but materially enlarge the pilot split. The question was whether the earlier failure mostly reflected a data-starved pilot selection surface rather than a fully wrong target family.

## Methods

- Model: `gpt2-xl`
- Device: local `mps`
- Prompt registry: `prompts/registry_v2.yaml`
- Train split: `pilot` (`16` prompts)
- Eval split: `confirm` (`8` prompts, unchanged from `registry_v1`)
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

- Doubling the pilot split changed the selected path:
  - selected target: `oracle_alpha_logit_vector`
  - selected ridge penalty: `0.01`
  - pilot tuning mean predicted improvement over uniform: `+0.1092` nats
  - pilot tuning mean JS: `0.2286`
- The raw-simplex and compressed targets no longer won pilot selection:
  - raw simplex target: `+0.0713` nats, mean JS `0.2220`, `lambda=0.01`
  - compressed depth-type-band logit target: `+0.0528` nats, mean JS `0.2198`, `lambda=100.0`
- Held-out confirm performance improved relative to the earlier `8 / 8` loss-aware run:
  - confirm `R^2 = -0.2616`
  - confirm mean JS `= 0.2377`
  - predicted mean improvement over uniform `= +0.0857` nats
  - oracle mean improvement over uniform `= +1.3409` nats
  - `5 / 8` confirm prompts improved over uniform
- Relative to the earlier `registry_v1` loss-aware comparison, the main change is that pilot expansion selected the full-logit path and restored a positive held-out routed-loss delta.

## Limitations

- This is still only a `16 / 8` prompt slice, so the result is stronger than the old `8 / 8` run but still far from a claim-bearing scale.
- Confirm `R^2` remains negative, so the predictor is not yet recovering the oracle alpha object in a strong descriptive sense even though the routed-loss metric improved.
- The new pilot prompts are still local inline prompts, not a benchmark-backed evaluation surface.

## Next Steps

- Treat pilot size as a real lever, not a solved problem.
- Follow `resattn-0vx`: scale the `registry_v2` loss-aware logit path beyond the current `16 / 8` slice before treating this as a stable unlock.
- Keep strong oracle-alpha interpretation blocked until the larger prompt surface and the held-out routed-loss advantage persist together.
