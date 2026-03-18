ABOUTME: Summarizes the donor-arm counterfactual on the broader matched-family Gemma factual-recall surface.
ABOUTME: Tests whether the narrow v2 donor-arm advantage survives on the locked 32-prompt `tool_breakage_factual_recall_v3` confirm set.

# Gemma-2 Matched-Family Donor-Arm Decomposition v3

## Motivation

`resattn-4g2` exists because the broader matched-family `v3` confirm baseline
stayed strongly positive on the tuned-lens primary metric. The last honest
question in this batch was whether the narrower donor-arm advantage from the
smaller `v2` surface survives on the larger balanced confirm set, or whether
the stronger same-model escalation still collapses once the explicit
within-family control is judged on more prompts.

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v3` confirm split (`32` prompts)
- Families:
  - `subcategory_capital_fact`
  - `subcategory_element_symbol`
  - `subcategory_author_fact`
  - `subcategory_moon_fact`
- Baseline source:
  `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-confirm-v3/summary.json`
- Fixed-alpha source:
  `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-pilot-v3/summary.json`
- Tuned lens:
  `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`
- Control arms:
  - `prompt_permuted_alpha`: legacy cyclic confirm donor
  - `within_family_permuted_alpha`: next donor within the same family block
  - `cross_family_permuted_alpha`: same within-family index, but from the next
    family block
  - `pilot_mean_alpha`: fixed alpha equal to the mean oracle alpha over the
    `v3` pilot prompts
- Primary metric: tuned-lens mean KL to the final output distribution
- Secondary diagnostics:
  - tuned final-position KL
  - tuned routed-versus-control final-target-rank worsening
  - tuned routed-versus-control best-target-rank worsening
  - tuned routed-versus-control target-rank-range increase
- Runtime durability:
  - prompt-level checkpoints under
    `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-donor-arms-v3/checkpoints/prompt_results`
  - exact-command rerun completed in `9.98` seconds after the first run

## Results

The broader `v3` surface preserves the strong fixed-alpha gap but does not
preserve a positive routed-minus-within-family mean tuned KL aggregate.

Against the fixed-alpha arm, routed remains clearly worse:

- `pilot_mean_alpha` mean tuned KL over original: `+2.1072`
- routed minus `pilot_mean_alpha` mean tuned KL: `+0.8056`
- routed minus `pilot_mean_alpha` final-position tuned KL: `+1.1740`
- routed worsens final target rank versus `pilot_mean_alpha` on `17 / 32`
  prompts
- routed worsens best target rank versus `pilot_mean_alpha` on `16 / 32` prompts
- routed increases target-rank range versus `pilot_mean_alpha` on `23 / 32`
  prompts

Against the broader donor controls, the read is mixed:

- `within_family_permuted_alpha` mean tuned KL over original: `+2.9895`
- routed minus `within_family_permuted_alpha` mean tuned KL: `-0.0768`
- routed minus `within_family_permuted_alpha` final-position tuned KL: `+0.2198`
- routed worsens final target rank versus `within_family_permuted_alpha` on
  `16 / 32` prompts
- routed worsens best target rank versus `within_family_permuted_alpha` on
  `17 / 32` prompts
- routed increases target-rank range versus `within_family_permuted_alpha` on
  `20 / 32` prompts

- `cross_family_permuted_alpha` mean tuned KL over original: `+2.8567`
- routed minus `cross_family_permuted_alpha` mean tuned KL: `+0.0560`
- routed minus `cross_family_permuted_alpha` final-position tuned KL: `+0.2850`
- routed worsens final target rank versus `cross_family_permuted_alpha` on
  `13 / 32` prompts
- routed worsens best target rank versus `cross_family_permuted_alpha` on
  `14 / 32` prompts
- routed increases target-rank range versus `cross_family_permuted_alpha` on
  `14 / 32` prompts

So the fixed-alpha objection stays cleared, but the broader within-family donor
control still blocks a stronger prompt-specific escalation on the primary
metric.

The family breakdown shows why:

- routed beats the within-family donor arm on mean tuned KL on `18 / 32` prompts
- but the aggregate is pulled negative by author prompts:
  - `subcategory_author_fact`: mean routed-minus-within-family KL `= -0.3678`
  - `subcategory_capital_fact`: `-0.0297`
  - `subcategory_element_symbol`: `+0.1624`
  - `subcategory_moon_fact`: `-0.0720`
- final-position tuned KL versus the within-family arm stays positive in every
  family, but that is not enough to overturn the primary mean-KL read

## Interpretation

- The tool-breakage lane now has a strong bounded same-model baseline story and
  a clearly weaker fixed-alpha objection on the broader surface.
- The broader `v3` donor-arm control does not support a stronger prompt-specific
  escalation on the primary tuned mean-KL metric.
- The right scientific boundary is therefore:
  - routed-versus-original breakage is real on the matched-family surface
  - fixed non-uniform routing is not enough to explain all of it
  - but the broader within-family donor control still prevents a stronger
    prompt-specific same-model claim

## Limitations

- The main residual story is family-conditioned heterogeneity, not a single
  pooled effect.
- The donor-arm artifact does not yet have a dedicated family-conditioned
  write-up; that is tracked separately.
- Multi-token author targets remain a first-token caveat.

## Next Steps

- Close `resattn-4g2` as mixed.
- Freeze the stronger same-model tool-breakage claim at this broader donor-arm
  boundary.
- If the lane resumes later, start with `resattn-mxf` and profile the
  family-conditioned donor-arm heterogeneity before spending another run.
