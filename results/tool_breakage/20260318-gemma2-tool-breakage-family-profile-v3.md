ABOUTME: Profiles family-conditioned heterogeneity in the broader Gemma matched-family donor-arm artifact.
ABOUTME: Preserves the prompt-level donor structure so the pooled `v3` summary is not over-interpreted.

# Gemma-2 Matched-Family Family Profile v3

## Motivation

`resattn-mxf` exists because the broader `tool_breakage_factual_recall_v3`
donor-arm artifact is scientifically mixed in a way that the pooled summary
cannot explain. The aggregate says routed loses to the within-family donor arm
on mean tuned KL by `0.0768` nats, but that number hides a real family split.
This artifact profiles that split directly from the saved prompt checkpoints so
future tool-breakage work does not reopen the lane from a flattened average.

## Methods

- Source artifact:
  `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-donor-arms-v3/summary.json`
- Prompt-level inputs:
  `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-donor-arms-v3/checkpoints/prompt_results/*.json`
- Prompt registry:
  `prompts/registry_v5.yaml`
- Collection: `tool_breakage_factual_recall_v3` confirm (`32` prompts)
- Families:
  - `subcategory_capital_fact`
  - `subcategory_element_symbol`
  - `subcategory_author_fact`
  - `subcategory_moon_fact`
- Analysis:
  - group prompt-level routed-versus-arm deltas by family
  - keep the same tuned-lens primary quantity as the saved counterfactual:
    mean routed-minus-arm KL to the final output distribution across layer
    traces
  - also retain prompt-level donor assignments, alpha Jensen-Shannon distance
    to each arm, final-position KL deltas, and rank-instability diagnostics
  - annotate target format with simple word-count / whitespace checks

The machine-readable summary is
`results/tool_breakage/20260318-gemma2-tool-breakage-family-profile-v3.json`.

## Results

The pooled `v3` summary is hiding three different stories rather than one.

### 1. The cleanest positive family is still `subcategory_element_symbol`

- routed minus `within_family_permuted_alpha` mean tuned KL: `+0.1624`
- routed minus `cross_family_permuted_alpha` mean tuned KL: `+0.7711`
- routed minus `pilot_mean_alpha` mean tuned KL: `+1.2521`
- within-family positive prompt fraction on mean tuned KL: `0.625`
- within-family alpha JS distance: `0.2085`

This is the strongest family-level support for the bounded same-model story on
the broader surface.

### 2. `subcategory_author_fact` is the main pooled drag, and not because its donor alphas are unusually close

- routed minus `within_family_permuted_alpha` mean tuned KL: `-0.3678`
- routed minus `cross_family_permuted_alpha` mean tuned KL: `-0.5923`
- routed minus `pilot_mean_alpha` mean tuned KL: `-0.0561`
- within-family positive prompt fraction on mean tuned KL: `0.625`
- within-family alpha JS distance: `0.3211`

Two details matter here.

First, the author family is negative not only against the within-family donor
arm but also against the cross-family donor arm and slightly against the fixed
pilot-mean alpha arm. So this is not just “within-family donor alphas are too
similar.”

Second, the within-family donor alphas are actually *less* similar for authors
than for capitals or elements:

- author within-family alpha JS: `0.3211`
- capital within-family alpha JS: `0.2458`
- element within-family alpha JS: `0.2085`

So alpha closeness alone does not explain the author-family reversal.

The more plausible confound visible in the saved prompt surface is target
format:

- author multiword-target fraction: `1.0`
- capital multiword-target fraction: `0.375`
- element multiword-target fraction: `0.0`
- moon multiword-target fraction: `0.0`

That does not prove the effect is only a first-token artifact, but it makes the
current author-family drag harder to interpret as a clean routing-family result.

The largest negative within-family prompt outliers are:

- `tb3-confirm-023`: `Margaret Atwood` routed minus within-family mean tuned KL
  `= -2.5799` against donor `tb3-confirm-024` (`Charlotte Bronte`)
- `tb3-confirm-018`: `Gabriel Garcia Marquez` routed minus within-family mean
  tuned KL `= -1.6841` against donor `tb3-confirm-019` (`Toni Morrison`)

### 3. Capitals and moons are near-tie families, not clean passes or collapses

Capitals:

- routed minus `within_family_permuted_alpha` mean tuned KL: `-0.0297`
- routed minus `cross_family_permuted_alpha` mean tuned KL: `+0.4090`
- routed minus `pilot_mean_alpha` mean tuned KL: `+1.3831`
- within-family positive prompt fraction on mean tuned KL: `0.5`
- mean final-layer routed-minus-within-family final-position KL: `+0.0598`

Moons:

- routed minus `within_family_permuted_alpha` mean tuned KL: `-0.0720`
- routed minus `cross_family_permuted_alpha` mean tuned KL: `-0.3638`
- routed minus `pilot_mean_alpha` mean tuned KL: `+0.6432`
- within-family positive prompt fraction on mean tuned KL: `0.5`
- mean final-layer routed-minus-within-family final-position KL: `+0.3308`

So both families are mixed, but in different ways:

- capitals look close to neutral against the within-family donor arm while still
  keeping a strong fixed-alpha gap
- moons keep a positive fixed-alpha gap and positive final-layer final-position
  KL versus the within-family arm, but mean tuned KL still washes slightly
  negative and the cross-family donor arm is also negative

## Interpretation

- The broader `v3` donor-arm result should remain frozen as a bounded,
  family-conditioned mixed artifact rather than a single pooled claim.
- The positive same-model prompt-specific signal is strongest in
  `subcategory_element_symbol`.
- The author-family reversal is the most important residual caveat, and the
  saved artifact points more toward target-format / output-space confounding
  than toward simple within-family alpha homogeneity.
- The fixed-alpha objection is clearly weaker only in the pooled sense. At the
  family level, that statement fails for authors and becomes modest for moons.

## Limitations

- This is post-run analysis on the existing `v3` prompt surface, not a new
  confirmatory run.
- The target-format annotation is intentionally simple; it identifies the
  multiword pattern but does not fully retokenize the answer surface.
- Author prompts remain subject to the existing first-token caveat.

## Next Steps

- Close `resattn-mxf`.
- If the tool-breakage lane reopens later, start from a target-format-aware
  family redesign rather than another pooled rerun.
- Keep the current same-model tool-breakage claim boundary frozen at the family
  level unless a new prompt surface separates routing-family structure from the
  current answer-format confound.
