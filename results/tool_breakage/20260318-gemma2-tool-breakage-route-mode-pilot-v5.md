# ABOUTME: Summarizes the first bounded Gemma tool-breakage pilot on the route-mode-aware v5 surface.
# ABOUTME: Tests whether the new one-token bridge is more informative when summarized by intended route modes as well as by family.

# Gemma-2 Route-Mode-Aware One-Token Pilot v5

## Motivation

`resattn-xfg` is the first execution follow-up to the `v5` prompt-surface
redesign from `resattn-q38`. The `v4` one-token surface was family-aligned but
mode-undercovered:

- capitals covered only `1 / 3` saved capital modes
- elements covered only `1 / 3` saved element modes
- authors covered only `2 / 5` saved author modes

The point of `v5` is therefore not just to run another smaller pilot. It is to
test whether the factual tool-breakage bridge becomes more informative when the
surface targets the robust capital, element, and author route modes directly
and keeps moon prompts out of the main collection.

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v5` pilot split (`10` prompts)
- Targeted families:
  - `subcategory_capital_fact`
  - `subcategory_element_symbol`
  - `subcategory_author_fact`
- Targeted route modes:
  - capitals: `clusters 2 / 3 / 4`
  - elements: `clusters 1 / 5 / 9`
  - authors: `clusters 6 / 7 / 11 / 12`
- Exclusions:
  - moon prompts stay outside the main collection
  - the singleton author outlier (`cluster 10`) stays outside the main
    collection
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

- baseline summary:
  `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-pilot-v5/summary.json`
- route-mode and family profile:
  `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-pilot-v5/profile.json`

Runtime durability:

- prompt checkpoints live under
  `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-pilot-v5/checkpoints/prompt_results`
- the exact-command rerun reused the full prompt checkpoint set in `10.16`
  seconds with an unchanged checkpoint timestamp hash
- rerunning the saved-artifact profile command reproduced the same JSON hash in
  `3.59` seconds

## Results

The route-mode-aware `v5` pilot stayed clearly positive overall:

- mean tuned KL increase under routing: `+2.8378`
- mean final-position tuned KL increase under routing: `+4.1369`
- tuned final-target-rank worsening fraction: `0.5`
- tuned target-rank-range increase fraction: `0.7`

The family-level pilot read is now clean across the narrowed bridge families:

- authors: `+3.2970`
- capitals: `+2.5546`
- elements: `+2.5088`

All three families are `100%` positive on the primary tuned mean-KL metric in
this pilot.

The route-mode profile is the more important read. Every targeted pilot mode is
positive on tuned mean KL increase under routing:

- author `cluster 11`: `+4.1085`
- author `cluster 7`: `+4.1652`
- author `cluster 6`: `+2.8493`
- author `cluster 12`: `+2.0651`
- capital `cluster 2`: `+2.6735`
- capital `cluster 3`: `+2.3949`
- capital `cluster 4`: `+2.5954`
- element `cluster 1`: `+2.4678`
- element `cluster 5`: `+2.4426`
- element `cluster 9`: `+2.6161`

The route-mode profile also exposes heterogeneity that the old pooled family
read could not:

- tuned final-target-rank worsening appears in `5 / 10` pilot modes
- the clearest worsening modes are:
  - author `cluster 6`: mean final target-rank delta `= +126`
  - author `cluster 11`: `+68`
  - capital `cluster 3`: `+101`
  - capital `cluster 4`: `+5`
  - element `cluster 9`: `+4`
- the other five pilot modes stayed positive on tuned mean KL without final
  target-rank worsening

## Interpretation

- `v5` is a real positive narrowing result.
- The route-mode-aware bridge did not just keep the pilot healthy overall. It
  made the heterogeneity more legible:
  - all targeted modes are positive on the primary tuned mean-KL metric
  - rank-instability is now visibly concentrated in a subset of author, capital,
    and element modes instead of being hidden inside pooled family averages
- That is enough to justify a confirmatory follow-up on `v5` rather than
  returning to `v4` or broad pooled reruns.

## Limitations

- This is still a pilot artifact with one prompt per targeted route mode.
- Because moon prompts are excluded by design, this artifact should not be
  read as a broad reopening of the full factual tool-breakage lane.
- Some apparent improvement versus `v4` may come from narrowing the surface away
  from moon prompts rather than from route-mode targeting alone.
- The singleton author outlier is still unresolved and intentionally excluded.

## Next Steps

- Close `resattn-xfg`.
- Take `resattn-138` next: run the locked `v5` confirm baseline and keep the
  write-up stratified by intended route mode and family.
- Keep `resattn-a1w` as the moon-family sidecar rather than folding moon prompts
  back into the main collection prematurely.
