ABOUTME: Records the locked confirm baseline on the expanded matched-family Gemma factual-recall surface.
ABOUTME: Tests whether the larger balanced `tool_breakage_factual_recall_v3` surface preserves the aligned same-model tuned-lens breakage signal.

# Gemma-2 Matched-Family Tool-Breakage Confirm v3

## Motivation

`resattn-cky` exists because the expanded balanced `v3` pilot stayed strongly
positive on the tuned-lens baseline. The next honest question was confirmatory:
does the locked `32`-prompt `tool_breakage_factual_recall_v3` surface preserve
that signal strongly enough that the final run in this batch should be the
larger donor-arm counterfactual?

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v3` confirm split (`32` prompts)
- Families:
  - `capital_fact`
  - `element_symbol`
  - `author_fact`
  - `moon_fact`
- Family balance: `8` confirm prompts per family
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
    `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-confirm-v3/checkpoints/prompt_results`
  - exact-command rerun after completion to verify checkpoint reuse
- Tracked machine-readable artifact:
  `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-confirm-v3/metrics.json`

## Results

- The larger locked confirm surface stays strongly positive on the tuned-lens
  baseline:
  - mean tuned KL to the final distribution rises from `3.2837` to `6.1964`
    (`+2.9127`)
  - final-position tuned KL rises from `5.5944` to `9.0248` (`+3.4304`)
  - tuned final-target-rank worsening on `22 / 32` prompts (`0.6875`)
  - tuned best-target-rank worsening on `21 / 32` prompts (`0.65625`)
  - tuned target-rank-range increase on `22 / 32` prompts (`0.6875`)
- The broader confirm surface slightly strengthens the old `v2` baseline rather
  than diluting it:
  - `v2` mean tuned KL delta `= +2.7644`
  - `v3` mean tuned KL delta `= +2.9127`
  - `v2` final-position tuned KL delta `= +3.4477`
  - `v3` final-position tuned KL delta `= +3.4304`
- Durability check passed:
  - exact-command rerun completed in `10.01` seconds
  - all `32` prompt checkpoints kept their original timestamps
  - only the local untracked `summary.json` was rewritten on rerun

## Interpretation

- The expanded matched-family surface is now a real confirmatory baseline, not
  just a pilot idea.
- The broader surface slightly strengthens the aligned same-model tool-breakage
  story on the tuned primary metric.
- That makes the donor-arm counterfactual on the same `v3` surface the only
  remaining high-value run in this batch.

## Limitations

- The full local `summary.json` is intentionally untracked because it exceeds
  the repo's large-file pre-commit limit.
- The moon family still uses more repeated fact templates than the other
  families.
- The donor-arm control question remains unanswered on this larger surface until
  the follow-up run lands.

## Next Steps

- Close `resattn-cky`.
- Run `resattn-4g2`: the donor-arm counterfactual on the locked `v3` confirm
  surface.
- Stop this batch there; that is the only remaining high-value control question.
