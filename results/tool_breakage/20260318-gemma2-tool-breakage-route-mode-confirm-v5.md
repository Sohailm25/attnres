# ABOUTME: Summarizes the locked Gemma confirm baseline on the route-mode-aware v5 tool-breakage surface.
# ABOUTME: Tests whether the narrowed factual bridge survives on the confirm split when read by route mode as well as by family.

# Gemma-2 Route-Mode-Aware One-Token Confirm v5

## Motivation

`resattn-138` is the confirm follow-up to the positive `v5` pilot from
`resattn-xfg`. The pilot established signs of life on the narrowed
capital/element/author bridge, but that was still one prompt per targeted route
mode. The confirm question is stricter:

- does the narrowed `v5` bridge stay positive on the locked split
- does the route-mode read survive beyond the pilot
- is the remaining mixed signal still concentrated in specific modes rather than
  spread across the whole narrowed surface

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
- Exclusions:
  - moon prompts remain outside the main collection
  - the singleton author outlier remains outside the main collection
- Baseline runner:
  `scripts/run_tool_breakage_factual_recall_baseline.py`
- Saved-artifact profile runner:
  `scripts/run_tool_breakage_baseline_profile.py`
- Primary metric: tuned-lens mean KL to the final output distribution under
  routing minus the original-model baseline
- Secondary diagnostics:
  - tuned final-position KL increase under routing
  - tuned final-target-rank worsening
  - tuned target-rank-range increase

Artifacts:

- tracked metrics summary:
  `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-confirm-v5/metrics.json`
- route-mode and family profile:
  `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-confirm-v5/profile.json`
- full local summary:
  `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-confirm-v5/summary.json`

Runtime durability:

- prompt checkpoints live under
  `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-confirm-v5/checkpoints/prompt_results`
- the exact-command rerun reused the full prompt checkpoint set in `9.99`
  seconds with an unchanged checkpoint timestamp hash
- rerunning the saved-artifact profile command reproduced the same JSON hash in
  `3.38` seconds

## Results

The locked `v5` confirm baseline stayed clearly positive overall:

- mean tuned KL increase under routing: `+2.9065`
- mean final-position tuned KL increase under routing: `+4.2024`
- tuned final-target-rank worsening fraction: `0.35`
- tuned target-rank-range increase fraction: `0.85`

The family-level confirm read is clean across the narrowed bridge families:

- authors: `+3.2542`, final-rank worsening fraction `= 0.375`
- capitals: `+2.4749`, final-rank worsening fraction `= 0.3333`
- elements: `+2.8743`, final-rank worsening fraction `= 0.3333`

All three narrowed bridge families are `100%` positive on the primary tuned
mean-KL metric in the locked confirm run.

The route-mode confirm read also holds throughout:

- author `cluster 7`: `+4.9136`, final-rank worsening fraction `= 1.0`
- author `cluster 6`: `+3.2753`, `0.0`
- author `cluster 11`: `+2.8350`, `0.0`
- author `cluster 12`: `+1.9931`, `0.5`
- capital `cluster 4`: `+2.6248`, `0.0`
- capital `cluster 2`: `+2.4993`, `0.5`
- capital `cluster 3`: `+2.3006`, `0.5`
- element `cluster 1`: `+3.1169`, `0.0`
- element `cluster 5`: `+2.7566`, `0.5`
- element `cluster 9`: `+2.7495`, `0.5`

So the key confirm result is stronger than the pilot boundary:

- all `10 / 10` targeted confirm modes are positive on tuned mean KL increase
  under routing
- the remaining mixed signal is now clearly mode-specific rather than
  family-global
- author `cluster 7` is the strongest confirm mode, while author `cluster 12`
  is the weakest but still positive

## Interpretation

- `v5` now has a real confirm pass for the narrowed factual tool-breakage
  bridge.
- The confirm split did not collapse the pilot result. It strengthened the
  claim that the narrowed capital/element/author bridge is real on the primary
  Gemma spine.
- The next honest question is therefore no longer “does `v5` have signs of
  life?” It is whether the same-model donor-arm controls stay supportive on this
  narrowed surface.
- The claim boundary still matters:
  - this is a confirm pass for the narrowed route-mode bridge
  - it is not a broad reopening of the full factual tool-breakage lane because
    moon prompts remain outside the main collection

## Limitations

- This confirm artifact still excludes moon prompts by design.
- The singleton author outlier is still unresolved.
- The strong same-model prompt-specific claim remains gated on the controlled
  donor-arm counterfactual, which has not yet been rerun on `v5`.

## Next Steps

- Close `resattn-138`.
- Take `resattn-hth` next: rerun the same-model donor-arm controls on the
  narrowed `v5` confirm surface and keep the write-up stratified by route mode
  as well as by family.
- Keep `resattn-a1w` as the moon-family sidecar rather than folding moon prompts
  back into the main collection.
