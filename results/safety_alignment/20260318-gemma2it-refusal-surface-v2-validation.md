# ABOUTME: Summarizes the broadened aligned-Gemma refusal-surface validation on the v2 prompt collection.
# ABOUTME: Records stable direction discovery alongside the stricter behavior-matching caveat introduced by refusal-style non-refusal prompts.

## Motivation

`resattn-mo5` asked for the smallest aligned-Gemma prompt-surface follow-up that
could plausibly break the current role-collapsed mediator partition. The first
step was to broaden the prompt collection itself while keeping the mechanism
stack fixed.

`safety_refusal_surface_v2` keeps the same three-role group structure but mixes
standard non-refusal prompts with refusal-style non-refusal prompts that ask for
safe refusal wording or policy language rather than direct model refusal.

## Methods

- Model: `google/gemma-2-2b-it`
- Collection: `safety_refusal_surface_v2`
- Size: `6` pilot groups, `6` confirm groups
- Checkpoints:
  - prompt-level residual checkpoints saved under
    `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation/checkpoints/prompt_residuals/`
- Readouts:
  - behavior checks
  - refusal and harmfulness layer localization
  - held-out paired direction-separation accuracy

## Results

- Refusal behavior stayed clean:
  - pilot hit rate `= 1.0`
  - confirm hit rate `= 1.0`
- Refusal-direction discovery stayed stable:
  - selected layer `= 22`
  - confirm primary pair accuracy `= 1.0`
- Harmfulness-direction discovery also stayed stable:
  - selected layer `= 18`
  - confirm primary pair accuracy `= 1.0`
- Refusal and harmfulness directions remained reasonably distinct:
  - cosine `= 0.1065`
- The broadened-surface caveat is behavioral:
  - non-refusal behavior pass rate fell to `0.8333` on pilot
  - non-refusal behavior pass rate fell to `0.6667` on confirm

## Interpretation

- The broadened surface did **not** break the refusal-feature workflow itself.
  Localization and held-out direction separation remained strong on the new
  collection.
- The important change is that the old behavior heuristic is now too strict for
  some prompts by design. Several `refusal_style_non_refusal` prompts elicited
  explicit first-person refusal text while still following the meta-instruction
  to draft refusal or policy language.
- So this artifact is a usable validation pass for the broadened surface, but it
  is not behaviorally equivalent to the original `safety_refusal_discovery_v1`
  collection.

## Limitations

- The existing non-refusal behavior rule treats any first-person refusal marker
  as a failure, even on prompts explicitly asking for refusal-style wording.
- That means this artifact broadens the surface successfully, but it also
  exposes a validator-semantics gap that future safety-surface work should not
  ignore.

## Next Steps

- Use the same broadened collection in the mediator-conditioned routing analysis.
- Record the validator follow-up separately so future broadened-surface runs can
  distinguish intentional refusal-style compliance from true role collapse.
