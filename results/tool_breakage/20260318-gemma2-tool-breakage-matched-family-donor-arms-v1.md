ABOUTME: Summarizes the donor-arm decomposition on the matched-family Gemma factual-recall counterfactual surface.
ABOUTME: Separates within-family and cross-family donor controls while keeping the legacy cyclic arm for comparison.

# Gemma-2 Matched-Family Donor-Arm Decomposition v1

## Motivation

`resattn-o3n` exists because the matched-family dynamic counterfactual on
`tool_breakage_factual_recall_v2` was still mixed under the old cyclic
`prompt_permuted_alpha` arm. The key ambiguity was structural: because the
confirm ids are grouped by family, that cyclic arm is already within-family on
`12 / 16` prompts and cross-family only on the four boundary transitions. The
next honest question was whether routed still exceeds explicit within-family and
cross-family donor controls once those are separated cleanly.

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v2` confirm split (`16` prompts)
- Families:
  - `subcategory_capital_fact`
  - `subcategory_element_symbol`
  - `subcategory_author_fact`
  - `subcategory_moon_fact`
- Baseline source:
  `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-confirm-v2/summary.json`
- Fixed-alpha source:
  `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-pilot-v2/summary.json`
- Tuned lens:
  `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`
- Control arms:
  - `prompt_permuted_alpha`: legacy cyclic confirm donor
  - `within_family_permuted_alpha`: next donor within the same family block
  - `cross_family_permuted_alpha`: same within-family index, but from the next
    family block
  - `pilot_mean_alpha`: fixed alpha equal to the mean oracle alpha over the
    matched-family pilot prompts
- Primary metric: tuned-lens mean KL to the final output distribution
- Secondary diagnostics:
  - tuned final-position KL
  - tuned routed-versus-control final-target-rank worsening
  - tuned routed-versus-control best-target-rank worsening
  - tuned routed-versus-control target-rank-range increase
- Runtime durability:
  - prompt-level checkpoints under
    `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-donor-arms-v1/checkpoints/prompt_results`
  - exact-command rerun completed in `9.83` seconds after the first run

## Results

The donor split improves the old `qww` control read, but only narrowly.

Against the fixed-alpha arm, routed remains clearly worse:

- `pilot_mean_alpha` mean tuned KL over original: `+1.6727`
- routed minus `pilot_mean_alpha` mean tuned KL: `+1.0917`
- routed minus `pilot_mean_alpha` final-position tuned KL: `+1.3894`
- routed worsens final target rank versus `pilot_mean_alpha` on `11 / 16`
  prompts
- routed worsens best target rank versus `pilot_mean_alpha` on `11 / 16` prompts
- routed increases target-rank range versus `pilot_mean_alpha` on `13 / 16`
  prompts

Against the explicit donor controls, routed now exceeds both arms on the tuned
primary metric, but only by very small margins:

- `within_family_permuted_alpha` mean tuned KL over original: `+2.7265`
- routed minus `within_family_permuted_alpha` mean tuned KL: `+0.0380`
- routed minus `within_family_permuted_alpha` final-position tuned KL: `+0.3949`
- routed worsens final target rank versus `within_family_permuted_alpha` on
  `7 / 16` prompts
- routed worsens best target rank versus `within_family_permuted_alpha` on
  `8 / 16` prompts
- routed increases target-rank range versus `within_family_permuted_alpha` on
  `9 / 16` prompts

- `cross_family_permuted_alpha` mean tuned KL over original: `+2.7555`
- routed minus `cross_family_permuted_alpha` mean tuned KL: `+0.0089`
- routed minus `cross_family_permuted_alpha` final-position tuned KL: `+0.2056`
- routed worsens final target rank versus `cross_family_permuted_alpha` on
  `6 / 16` prompts
- routed worsens best target rank versus `cross_family_permuted_alpha` on
  `6 / 16` prompts
- routed increases target-rank range versus `cross_family_permuted_alpha` on
  `8 / 16` prompts

The aggregate is not broad:

- routed beats the within-family donor arm on mean tuned KL on `10 / 16` prompts
- routed beats the within-family donor arm on final-position tuned KL on
  `8 / 16` prompts
- family-level mean routed-minus-within-family tuned mean KL:
  - `subcategory_element_symbol`: `+0.2385`
  - `subcategory_author_fact`: `-0.0244`
  - `subcategory_moon_fact`: `-0.0217`
  - `subcategory_capital_fact`: `-0.0406`

So the element family is doing most of the work in the current within-family
aggregate.

## Interpretation

- The donor-arm decomposition answers the immediate geometry question. The old
  dynamic blocker was too pessimistic: routed does slightly exceed both explicit
  donor arms on the tuned primary metric.
- That is still not a clean strong same-model pass. The margins are tiny, the
  per-prompt counts are modest, and the within-family aggregate is carrying a
  clear family concentration.
- The right next move is therefore not another control redesign. It is a larger
  balanced matched-family surface that can test whether this narrow donor-arm
  advantage survives more prompts.

## Limitations

- The confirm surface is still only `16` prompts.
- The donor-arm advantage is small enough that prompt composition can still move
  the aggregate meaningfully.
- Multi-token author targets remain a first-token caveat.

## Next Steps

- Close `resattn-o3n` as mixed.
- Run `resattn-0mu` to expand the matched-family surface around the same four
  families.
- Rerun the donor-arm counterfactual on that larger balanced surface before
  making stronger same-model language.
