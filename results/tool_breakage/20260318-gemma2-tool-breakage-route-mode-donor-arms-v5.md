ABOUTME: Summarizes the donor-arm counterfactual on the narrowed route-mode-aware Gemma factual surface.
ABOUTME: Tests whether the v5 bridge keeps supportive donor-arm controls once the confirm baseline is fixed.

# Gemma-2 Route-Mode-Aware Donor-Arm Decomposition v5

## Motivation

`resattn-hth` is the control-stage follow-up to the positive `v5` confirm
baseline. The narrowed bridge already showed that routed traces are worse than
the original-model baseline across all targeted route modes. The next honest
question is stricter:

- does routed still beat the fixed `pilot_mean_alpha` objection on `v5`
- does the donor-arm read stay supportive on the narrowed bridge
- if the result stays mixed, is the mixture family-level or route-mode-specific

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v5` confirm split (`20` prompts)
- Families:
  - `subcategory_capital_fact`
  - `subcategory_element_symbol`
  - `subcategory_author_fact`
- Route modes:
  - capitals: `clusters 2 / 3 / 4`
  - elements: `clusters 1 / 5 / 9`
  - authors: `clusters 6 / 7 / 11 / 12`
- Baseline source:
  `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-confirm-v5/summary.json`
- Fixed-alpha source:
  `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-pilot-v5/summary.json`
- Counterfactual runner:
  `scripts/run_tool_breakage_dynamic_counterfactual.py`
- Saved-artifact profile runner:
  `scripts/run_tool_breakage_counterfactual_profile.py`
- Control arms:
  - `prompt_permuted_alpha`
  - `within_family_permuted_alpha`
  - `cross_family_permuted_alpha`
  - `pilot_mean_alpha`
- Primary metric: tuned-lens mean KL to the final output distribution
- Secondary diagnostics:
  - tuned final-position KL
  - tuned routed-versus-control final-target-rank worsening
  - tuned routed-versus-control target-rank-range increase

Artifacts:

- full counterfactual summary:
  `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-donor-arms-v5/summary.json`
- route-mode and family profile:
  `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-donor-arms-v5/profile.json`

Runtime durability:

- prompt checkpoints live under
  `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-donor-arms-v5/checkpoints/prompt_results`
- the exact-command rerun finished in `10.03` seconds with unchanged checkpoint
  timestamp hash
- rerunning the saved-artifact profile command reproduced the same JSON hash

## Results

The fixed-alpha objection stays clearly weaker than routed on the narrowed
surface:

- routed minus `pilot_mean_alpha` mean tuned KL: `+1.0898`
- routed minus `pilot_mean_alpha` final-position tuned KL: `+1.1861`
- routed worsens final target rank versus `pilot_mean_alpha` on `9 / 20` prompts
- routed increases tuned target-rank range versus `pilot_mean_alpha` on
  `16 / 20` prompts

The donor-arm read is still mixed on the pooled aggregate:

- routed minus `within_family_permuted_alpha` mean tuned KL: `-0.1021`
- routed minus `prompt_permuted_alpha` mean tuned KL: `-0.1502`
- routed minus `cross_family_permuted_alpha` mean tuned KL: `-0.3728`

The family split is more informative than the pooled aggregate:

- authors:
  - routed minus `within_family_permuted_alpha`: `-0.3415`
  - routed minus `prompt_permuted_alpha`: `-0.5756`
  - routed minus `cross_family_permuted_alpha`: `+0.0803`
- capitals:
  - routed minus `within_family_permuted_alpha`: `+0.0004`
  - routed minus `prompt_permuted_alpha`: `-0.0159`
  - routed minus `cross_family_permuted_alpha`: `-0.6346`
- elements:
  - routed minus `within_family_permuted_alpha`: `+0.1146`
  - routed minus `prompt_permuted_alpha`: `+0.2829`
  - routed minus `cross_family_permuted_alpha`: `-0.7151`

So the narrowed bridge does not fail uniformly. The donor-arm signal is still
alive in elements, nearly tied in capitals, and negative mainly in authors.

The route-mode split sharpens that picture further:

- positive on all four arms:
  - `route_mode_author_cluster_7`
- positive on `within_family_permuted_alpha` and `prompt_permuted_alpha`:
  - `route_mode_capital_cluster_3`
  - `route_mode_element_cluster_5`
  - `route_mode_element_cluster_9`
- positive on `cross_family_permuted_alpha`:
  - `route_mode_author_cluster_7`
  - `route_mode_element_cluster_1`
  - `route_mode_element_cluster_9`
- negative on every dynamic donor arm:
  - `route_mode_author_cluster_12`

The prompt-permuted and within-family donor arms are not fully independent on
this surface. Their route-mode summaries are numerically identical on `7 / 10`
targeted modes, which reflects how the locked confirm ordering reuses the same
local donor pairings for many modes.

## Interpretation

- `v5` keeps the fixed-alpha objection clearly weaker than routed. That part of
  the same-model story survives the narrowed bridge cleanly.
- `v5` still does not broadly reopen the stronger prompt-specific donor-arm
  claim. The pooled `within_family_permuted_alpha` and `prompt_permuted_alpha`
  reads remain slightly negative on the primary metric.
- The mixed result is structured rather than mushy:
  - elements remain modestly supportive
  - capitals are near a tie
  - authors are the blocker
  - one author mode (`cluster 7`) is very strong while another (`cluster 12`)
    is negative against every dynamic donor arm
- The truthful boundary is therefore narrower than “route-matched routing is
  broadly worse than donor swaps.” The repo can now support a bounded
  route-mode-aware bridge with a cleared fixed-alpha objection, but not a broad
  donor-arm escalation.

## Limitations

- Moon prompts remain out of scope for this narrowed bridge.
- `prompt_permuted_alpha` and `within_family_permuted_alpha` partially collapse
  on this confirm ordering, so they are not maximally independent controls on
  `v5`.
- Each targeted route mode still has only `2` confirm prompts, so the
  route-mode-level read is informative but still brittle.

## Next Steps

- Close `resattn-hth` as a mixed donor-arm result on the narrowed `v5` bridge.
- Keep the tool-breakage claim boundary restricted to:
  - positive routed-versus-original baseline on `v5`
  - positive routed-versus-fixed-alpha counterfactual
  - mixed donor-arm result with route-mode heterogeneity
- If tool-breakage resumes, audit the `v5` donor assignment geometry before any
  further rerun, especially the author modes and the `prompt_permuted` /
  `within_family` collapse.
