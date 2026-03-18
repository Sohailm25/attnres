ABOUTME: Records the first pilot baseline on the expanded matched-family Gemma factual-recall surface.
ABOUTME: Tests whether the larger balanced `tool_breakage_factual_recall_v3` surface preserves the same-model tuned-lens degradation signal.

# Gemma-2 Matched-Family Tool-Breakage Pilot v3

## Motivation

`resattn-0mu` exists because the donor-arm decomposition on
`tool_breakage_factual_recall_v2` became promising but still narrow. Before
spending another run on a larger donor-arm counterfactual, the repo needed to
know whether the aligned same-model breakage signal survives a materially larger
balanced factual surface.

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v3` pilot split (`16` prompts)
- Families:
  - `capital_fact`
  - `element_symbol`
  - `author_fact`
  - `moon_fact`
- Family balance: `4` pilot prompts per family
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
    `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-pilot-v3/checkpoints/prompt_results`
  - exact-command rerun after completion to verify checkpoint reuse
- Tracked machine-readable artifact:
  `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-pilot-v3/metrics.json`

## Results

- The larger balanced pilot keeps the tuned-lens breakage signal strongly
  positive:
  - mean tuned KL to the final distribution rises from `3.1583` to `5.7602`
    (`+2.6019`)
  - final-position tuned KL rises from `5.6496` to `8.6820` (`+3.0324`)
  - tuned final-target-rank worsening on `8 / 16` prompts (`0.5000`)
  - tuned best-target-rank worsening on `6 / 16` prompts (`0.3750`)
  - tuned target-rank-range increase on `11 / 16` prompts (`0.6875`)
- The broader surface does not dilute the old `v2` pilot:
  - `v2` mean tuned KL delta `= +2.5849`
  - `v3` mean tuned KL delta `= +2.6019`
  - `v2` final-position tuned KL delta `= +2.9118`
  - `v3` final-position tuned KL delta `= +3.0324`
- Durability check passed:
  - exact-command rerun completed in `10.01` seconds
  - all `16` prompt checkpoints kept their original timestamps
  - only the local untracked `summary.json` was rewritten on rerun

## Interpretation

- The aligned same-model breakage signal survives a materially larger balanced
  prompt surface.
- That makes the next confirm step worth the time. The repo no longer has a
  good reason to stay on the smaller `v2` pilot when testing this lane.
- The next meaningful move is the locked `32`-prompt confirm baseline on
  `tool_breakage_factual_recall_v3`.

## Limitations

- This is still a pilot artifact, not the locked confirm read.
- The moon family uses more repeated fact templates than the other families,
  because that subcategory has fewer clean direct-recall variants than capitals
  or element symbols.
- The full local `summary.json` is intentionally untracked because it exceeds
  the repo's large-file pre-commit limit.

## Next Steps

- Close `resattn-0mu`.
- Run `resattn-cky`: the locked confirm baseline on
  `tool_breakage_factual_recall_v3`.
- If the confirm read stays positive, spend the next run on the larger donor-arm
  counterfactual rather than more prompt curation.
