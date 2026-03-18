# ABOUTME: Summarizes the tag-aware rerun of the broadened aligned-Gemma refusal-surface validation.
# ABOUTME: Records the validator semantics fix from `resattn-ac2` without overstating the remaining surface cleanliness.

## Motivation

`resattn-ac2` followed `resattn-mo5`, which had already shown that the broadened
aligned-Gemma surface was mechanistically usable but behaviorally harsher than
intended. Some non-refusal prompts were designed to elicit refusal-style
compliant language, yet the old validator still treated any first-person
refusal marker as a failure outside the `refusal` role.

The goal here was to rerun the same broadened validation after making expected
behavior tag-aware instead of role-only.

## Methods

- Model: `google/gemma-2-2b-it`
- Collection: `safety_refusal_surface_v2`
- Size: `6` pilot groups, `6` confirm groups
- Validator change:
  - expected behavior now comes from prompt tags first
  - `refusal_style_non_refusal` prompts validate against refusal-like
    completions
- Output:
  - `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-tag-aware-v1/summary.json`

## Results

- Refusal behavior stayed clean:
  - pilot hit rate `= 1.0`
  - confirm hit rate `= 1.0`
- Mechanistic discovery stayed unchanged:
  - refusal localization `= layer 22`
  - harmfulness localization `= layer 18`
  - refusal confirm pair accuracy `= 1.0`
  - harmfulness confirm pair accuracy `= 1.0`
  - refusal/harmfulness cosine `= 0.1065`
- The behavior summary improved where the old validator was over-strict:
  - non-refusal behavior pass rate stayed `0.8333` on pilot
  - non-refusal behavior pass rate improved from `0.6667` to `0.8333` on confirm
- The prompt-level behavior table now records `expected_behavior_mode` explicitly.

## Interpretation

- `resattn-ac2` fixed the right failure class:
  - refusal-style compliant non-refusal prompts are no longer automatically
    counted as failures
- The mechanistic read of the broadened surface does not change.
- The surface is still not perfectly clean:
  - one harmful-context confirm prompt still elicited a genuine refusal-style
    completion
  - some policy-note benign prompts still do not match the current
    first-person-refusal marker even though they are arguably compliant

## Limitations

- The validator now distinguishes refusal-style compliant prompts, but it still
  uses a narrow refusal marker that is better at first-person refusals than at
  policy-note style institutional language.
- This rerun does not revisit the mediator-conditioned routing result, which
  remains a bounded negative on the broadened surface.

## Next Steps

- Close `resattn-ac2`.
- Track the narrower policy-style semantics follow-up in `resattn-7km`.
- Use the tag-aware validator for any future broadened safety-surface work.
