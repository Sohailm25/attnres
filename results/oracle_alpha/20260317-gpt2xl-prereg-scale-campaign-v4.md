# GPT-2 XL Prereg-Scale Oracle-Alpha Campaign V4

## Motivation

`resattn-9jq` was the first prereg-scale oracle-alpha campaign on the saved `registry_v4` prompt surface after the checkpointed campaign runner landed. The question was no longer whether the held-out predictiveness path could stay positive on a slightly larger split; it was whether the development-model oracle lane would survive a materially larger `96 / 128` pilot/confirm surface, clear the preregistered oracle-loss gate on at least `100` confirm sequences, and still preserve a positive held-out routed-loss signal.

## Methods

- Model: `gpt2-xl`
- Device: local `mps`
- Prompt registry: `prompts/registry_v4.yaml`
- Train split: `pilot` (`96` prompts)
- Eval split: `confirm` (`128` prompts)
- Feature source: `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`
- Candidate targets:
  - `oracle_alpha_vector`
  - `oracle_alpha_logit_vector`
  - `oracle_alpha_depth_type_band_logit_vector`
- Predictiveness model: ridge regression
- Tuning rule:
  - pilot-only target and regularization selection
  - primary metric: mean predicted routed-loss improvement over uniform
  - secondary metric: mean JS divergence to oracle alpha
  - regularization grid: `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100}`
- Oracle-alpha optimization settings:
  - `20` Adam steps
  - learning rate `0.1`
  - seed `11`
- Campaign persistence:
  - prompt-level oracle checkpoints under `checkpoints/oracle_runs/`
  - prompt-level feature caches under `checkpoints/feature_vectors/`
  - final summaries in `oracle_train_run.json`, `oracle_eval_run.json`, `predictiveness_summary.json`, and `campaign_manifest.json`

## Results

- The prereg-scale oracle stage cleared the development-model Phase 1 gate on the confirm split:
  - confirm mean oracle improvement over uniform: `+1.2993` nats
  - `128 / 128` confirm prompts improved over uniform
  - bootstrap interval: `[1.2604, 1.3417]`
  - paired one-sided t-test versus uniform: `p = 9.72e-98`
  - paired Cohen's `d` versus uniform: `5.54`
- The oracle result also beat the preregistered nulls strongly on every confirm prompt:
  - versus `random_dirichlet`: mean delta `+1.7611`, `p = 4.50e-83`, `d = 4.20`
  - versus `magnitude_proportional`: mean delta `+2.1361`, `p = 9.64e-81`, `d = 4.02`
  - versus `last_layer_only`: mean delta `+6.8152`, `p = 2.52e-99`, `d = 5.71`
- Held-out predictiveness remained positive at prereg scale, but the selector moved back to the raw-simplex target:
  - selected target: `oracle_alpha_vector`
  - selected ridge penalty: `100.0`
  - pilot tuning mean predicted improvement over uniform: `+0.1131` nats
  - pilot mean JS: `0.2365`
- Confirm predictiveness stayed positive in the routed-loss metric:
  - predicted mean improvement over uniform: `+0.1162` nats
  - paired one-sided t-test versus uniform: `p = 1.82e-11`
  - paired Cohen's `d`: `0.64`
  - `95 / 128` confirm prompts improved over uniform
  - oracle mean improvement over uniform: `+1.2993` nats
- Descriptive alpha-shape recovery is still weak despite the positive routed-loss result:
  - confirm `R^2 = -0.0884`
  - confirm mean JS `= 0.2427`
- Relative to the earlier `32 / 16` run, the headline is mixed but favorable:
  - the held-out routed-loss signal remained positive on a much larger confirm surface
  - the target selector no longer preferred `oracle_alpha_logit_vector`
  - strong alpha-shape recovery is still not present

## Limitations

- This is a development-model result on `gpt2-xl`, not yet the primary Gemma-2 lane.
- Strong interpretation remains blocked: the routed-loss metric is positive, but alpha-shape recovery is still weak (`R^2 < 0`, mean JS `≈ 0.24`).
- The saved prompt surface is larger and confirm-locked, but it is still a local inline collection rather than a benchmark-backed evaluation set.
- The campaign runner exposed an observability problem: top-level summary files stayed stale until the long pilot tuning sweep completed, even though the expensive oracle checkpoints were already done.

## Next Steps

- Treat the development-model oracle gate as cleared and move the next oracle task to prereg-scale pattern analysis on the saved `registry_v4` artifact.
- Keep strong oracle-alpha interpretation blocked until later work either improves alpha-shape recovery materially or shows that routed-loss-positive predictiveness is the right confirmatory object.
- Carry the same campaign logic toward the primary Gemma-2 lane once the model-specific backend path is ready.
- Address `resattn-9co` so future large campaigns expose progress during the summary stage rather than only at the final write.
