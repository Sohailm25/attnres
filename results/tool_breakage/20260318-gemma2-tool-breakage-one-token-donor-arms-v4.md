ABOUTME: Summarizes the donor-arm counterfactual on the one-token matched-family Gemma factual-recall surface.
ABOUTME: Tests whether the redesigned v4 surface reopens the stronger prompt-specific same-model tool-breakage claim.

# Gemma-2 One-Token Matched-Family Donor-Arm Decomposition v4

## Motivation

`resattn-apy` reruns the explicit donor-arm counterfactual on
`tool_breakage_factual_recall_v4`, the one-token matched-family surface. The
locked `v4` confirm baseline already showed that every family stayed positive on
mean tuned KL under routing. The real question here is whether that redesign
was enough to improve the donor-arm controls as well, or whether the stronger
prompt-specific same-model claim would still remain mixed.

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v4` confirm split (`32` prompts)
- Families:
  - `subcategory_capital_fact`
  - `subcategory_element_symbol`
  - `subcategory_author_fact`
  - `subcategory_moon_fact`
- Surface constraint: every `target_text` is exactly one next token under the
  `google/gemma-2-2b` tokenizer
- Baseline source:
  `results/tool_breakage/20260318-gemma2-tool-breakage-one-token-confirm-v4/summary.json`
- Fixed-alpha source:
  `results/tool_breakage/20260318-gemma2-tool-breakage-one-token-pilot-v4/summary.json`
- Tuned lens:
  `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`
- Control arms:
  - `prompt_permuted_alpha`: legacy cyclic confirm donor
  - `within_family_permuted_alpha`: next donor within the same family block
  - `cross_family_permuted_alpha`: same within-family index, but from the next
    family block
  - `pilot_mean_alpha`: fixed alpha equal to the mean oracle alpha over the
    `v4` pilot prompts
- Primary metric: tuned-lens mean KL to the final output distribution
- Secondary diagnostics:
  - tuned final-position KL
  - tuned routed-versus-control final-target-rank worsening
  - tuned routed-versus-control best-target-rank worsening
  - tuned routed-versus-control target-rank-range increase
- Runtime durability:
  - prompt-level checkpoints under
    `results/tool_breakage/20260318-gemma2-tool-breakage-one-token-donor-arms-v4/checkpoints/prompt_results`
  - exact-command rerun finished in `10.07` seconds with unchanged checkpoint
    timestamp hash

The machine-readable summary for this artifact is
`results/tool_breakage/20260318-gemma2-tool-breakage-one-token-donor-arms-v4/summary.json`.

## Results

The one-token redesign materially improved the donor-arm picture relative to the
saved `v3` artifact, but it did not clear every donor control.

Against the fixed-alpha arm, routed is still clearly worse:

- routed minus `pilot_mean_alpha` mean tuned KL: `+0.2642`
- routed minus `pilot_mean_alpha` final-position tuned KL: `+0.8221`
- routed worsens final target rank versus `pilot_mean_alpha` on `19 / 32`
  prompts
- routed worsens best target rank versus `pilot_mean_alpha` on `19 / 32` prompts
- routed increases target-rank range versus `pilot_mean_alpha` on `24 / 32`
  prompts

Against the donor controls, the read is mixed but healthier than `v3`:

- routed minus `prompt_permuted_alpha` mean tuned KL: `-0.0139`
- routed minus `prompt_permuted_alpha` final-position tuned KL: `+0.2545`
- routed worsens final target rank versus `prompt_permuted_alpha` on `15 / 32`
  prompts
- routed worsens best target rank versus `prompt_permuted_alpha` on `13 / 32`
  prompts
- routed increases target-rank range versus `prompt_permuted_alpha` on `19 / 32`
  prompts

- routed minus `within_family_permuted_alpha` mean tuned KL: `+0.0165`
- routed minus `within_family_permuted_alpha` final-position tuned KL: `+0.2960`
- routed worsens final target rank versus `within_family_permuted_alpha` on
  `16 / 32` prompts
- routed worsens best target rank versus `within_family_permuted_alpha` on
  `14 / 32` prompts
- routed increases target-rank range versus `within_family_permuted_alpha` on
  `18 / 32` prompts

- routed minus `cross_family_permuted_alpha` mean tuned KL: `-0.2297`
- routed minus `cross_family_permuted_alpha` final-position tuned KL: `-0.2900`
- routed worsens final target rank versus `cross_family_permuted_alpha` on
  `11 / 32` prompts
- routed worsens best target rank versus `cross_family_permuted_alpha` on
  `12 / 32` prompts
- routed increases target-rank range versus `cross_family_permuted_alpha` on
  `14 / 32` prompts

Relative to `v3`, the pooled improvement is real:

- routed minus `within_family_permuted_alpha` mean tuned KL moved from
  `-0.0768` to `+0.0165`
- routed minus `prompt_permuted_alpha` mean tuned KL moved from `-0.0982` to
  `-0.0139`
- routed minus `pilot_mean_alpha` stayed clearly positive, though smaller in
  magnitude (`+0.8056` to `+0.2642`)
- routed minus `cross_family_permuted_alpha` regressed from `+0.0560` to
  `-0.2297`

The later saved family-conditioned profile in
`results/tool_breakage/20260318-gemma2-tool-breakage-family-profile-v4.md`
should govern any family-level interpretation of this pooled artifact.

## Interpretation

- The one-token redesign is a real improvement over the saved `v3` donor-arm
  result. The old author-format confound was materially suppressing the
  within-family donor read.
- The stronger prompt-specific same-model claim is still not broadly reopened.
  The fixed-alpha objection remains clearly weaker, and the within-family donor
  arm is now slightly positive, but the cross-family donor arm is still stronger
  than routed on aggregate.
- The residual blocker is no longer just “authors are multi-token.” The moon
  family now dominates the remaining cross-family failure.

## Limitations

- This artifact still relies on a pooled aggregate as the primary claim surface;
  the saved family-conditioned follow-up lives in
  `results/tool_breakage/20260318-gemma2-tool-breakage-family-profile-v4.md`.
- The moon prompts are somewhat more stylized than capitals and elements, so the
  cross-family pairing may still be conflating donor mismatch with prompt style.
- The positive within-family margin is real but small (`+0.0165`), so it should
  not be overstated.

## Next Steps

- Close `resattn-apy` as a mixed improvement.
- Use `resattn-8h7` to write the family-conditioned `v4` donor-arm profile
  explicitly.
- Decide from that profile whether the next honest move is moon-prompt redesign,
  donor-pair remapping, or freezing the stronger same-model claim at this new
  mixed boundary.
