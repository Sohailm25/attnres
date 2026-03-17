# Held-Out Predictiveness Review Alignment

## Motivation

`resattn-23p` followed an external review of the held-out oracle-alpha predictiveness setup. The review did not argue that the oracle optimization was fake; it argued that the current predictiveness setup was asking the wrong question in the wrong geometry. Before changing the target formulation itself, the repo needed two no-regret fixes:

- make the saved control registry actually govern feature and regularization selection
- report prompt-matched per-sequence alpha stability rather than only aggregate alpha stability

## Changes Landed

- The runner now consults the saved predictiveness control metrics from [configs/oracle_alpha_controls_v1.yaml](/Users/sohailmo/resattn/configs/oracle_alpha_controls_v1.yaml) when comparing candidate feature sources and ridge regularization strengths.
- The control layer now exposes reusable metric-comparison helpers, with explicit directionality for `r_squared` and `mean_js_divergence`.
- The stability suite now reports prompt-matched per-sequence alpha stability alongside the previous aggregate alpha-distribution metrics.
- The design-review follow-up for the harder problem was split out into `resattn-qq2`, which will handle the constrained-or-compressed target reformulation instead of smuggling that redesign into a small infrastructure patch.

## Results

- No new model-backed oracle-alpha experiment was run in this slice.
- The value of this change is methodological alignment:
  - the control registry is now less decorative and more authoritative
  - restart/paraphrase/resample stability evidence is now more relevant to the held-out predictiveness object

## Limitations

- This slice does **not** fix the main geometric critique that the predictor is learning a simplex-distributed alpha target in unconstrained Euclidean ridge coordinates.
- This slice does **not** fix the scale critique that `8` pilot / `8` confirm prompts are very small relative to the current target dimension on `gpt2-xl`.
- This slice does **not** rerun the held-out experiments, because the predictor formulation itself remains under review.

## Next Steps

- Follow `resattn-qq2`: implement a constrained or compressed predictiveness target.
- Revisit whether the prompt surface should be enlarged after the target formulation is fixed.
