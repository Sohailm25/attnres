ABOUTME: Records the locked confirm baseline on the matched-family Gemma factual-recall tool-breakage surface.
ABOUTME: Tests whether the better-aligned `v2` surface should replace the old mixed factual baseline before the dynamic counterfactual step.

# Motivation

`resattn-4ny` exists because the matched-family pilot on
`tool_breakage_factual_recall_v2` stayed strongly positive on the same-model
tuned-lens baseline. The next honest question was confirmatory: does the locked
`16`-prompt confirm split preserve that signal strongly enough that `v2` should
replace the old mixed factual `v1` surface as the main bounded baseline for the
Gemma tool-breakage lane?

# Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v2` confirm split (`16` prompts)
- Families:
  - `capital_fact`
  - `element_symbol`
  - `author_fact`
  - `moon_fact`
- Family balance: `4` confirm prompts per family
- Lens comparison:
  - raw logit lens
  - custom Gemma-2 tuned lens from
    `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`
- Primary metric: KL to the model's final output distribution
- Secondary diagnostics:
  - mean top-1 agreement
  - final-position KL / top-1
  - routed-versus-original final-target-rank worsening
  - routed-versus-original target-rank-range increase
- Durability:
  - prompt-level checkpoints under
    `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-confirm-v2/checkpoints/prompt_results`
- exact-command rerun after completion to verify checkpoint reuse
- tracked machine-readable artifact:
  `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-confirm-v2/metrics.json`

# Results

- The matched-family confirm surface stays strongly positive on the tuned-lens
  baseline:
  - mean tuned KL to the final distribution rises from `3.1966` to `5.9610`
    (`+2.7644`)
  - final-position tuned KL rises from `5.7886` to `9.2363` (`+3.4477`)
  - mean tuned top-1 drops from `0.5036` to `0.3196`
  - final-position tuned top-1 drops from `0.1490` to `0.0962`
- The relative rank metrics remain clearly informative on the larger confirm
  set:
  - tuned final-target-rank worsening on `11 / 16` prompts (`0.6875`)
  - tuned best-target-rank worsening on `8 / 16` prompts (`0.5000`)
  - tuned target-rank-range increase on `11 / 16` prompts (`0.6875`)
  - tuned routed-versus-original non-monotonicity increase remains `0 / 16`
- Family-level confirm behavior is heterogeneous but consistently positive on
  final-position tuned KL deltas:
  - capitals:
    - mean final-position tuned KL delta `= +4.7066`
    - mean final target-rank delta `= +1944.25`
  - moons:
    - mean final-position tuned KL delta `= +4.0721`
    - mean final target-rank delta `= +71.25`
  - elements:
    - mean final-position tuned KL delta `= +3.5016`
    - mean final target-rank delta `= +2.0`
    - mean range delta is negative, so these prompts still tend to compress
      target-rank range rather than broaden it
  - authors:
    - mean final-position tuned KL delta `= +2.8658`
    - mean final target-rank delta `= +166.0`
- Direct comparison to the old mixed factual baseline favors `v2` on the
  KL-primary read:
  - old `v1` mean tuned KL delta `= +2.5227` on `8` prompts
  - new `v2` mean tuned KL delta `= +2.7644` on `16` prompts
  - old `v1` final-position tuned KL delta `= +2.9096`
  - new `v2` final-position tuned KL delta `= +3.4477`
  - old `v1` tuned final-target-rank worsening `= 0.625`
  - new `v2` tuned final-target-rank worsening `= 0.6875`
  - old `v1` tuned target-rank-range increase `= 0.875`
  - new `v2` tuned target-rank-range increase `= 0.6875`
- Durability check passed:
- exact-command rerun completed in `10.11` seconds
- all `16` prompt checkpoints kept their original timestamps
- only `summary.json` was rewritten on rerun

# Interpretation

- `tool_breakage_factual_recall_v2` should now be treated as the main bounded
  same-model baseline surface for this lane.
- The aligned surface is stronger than the old mixed factual surface on the
  KL-primary confirm read while using twice as many confirm prompts, so there is
  no good reason to keep centering `v1`.
- The next real blocker is therefore no longer baseline quality. It is the
  controlled dynamic-routing counterfactual on this new surface.

# Limitations

- The stronger same-model claim is still blocked on the dynamic-routing
  counterfactual.
- Element prompts still behave differently from the other families on the range
  metric.
- Multi-token answer strings remain a first-token caveat for some author prompts.

# Next Steps

- Close `resattn-4ny`.
- Run `resattn-qww`: the dynamic-routing counterfactual on
  `tool_breakage_factual_recall_v2`.
- Treat the old mixed factual `v1` baseline as contextual evidence only.
