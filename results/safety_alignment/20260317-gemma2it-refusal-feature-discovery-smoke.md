# ABOUTME: Summarizes the smoke run for the aligned-Gemma refusal-feature discovery workflow.
# ABOUTME: Records the first checkpointed prompt-level safety artifact before the full validation run.

## Motivation

Verify that the first refusal-feature discovery runner works end to end on an aligned
Gemma model before paying for the full pilot/confirm split.

## Methods

- Model: `google/gemma-2-2b-it`
- Collection: `safety_refusal_discovery_v1`
- Size: `1` pilot group and `1` confirm group
- Positions:
  - harmfulness localization at `instruction_final`
  - refusal localization at `assistant_prefill`
- Checks:
  - prompt-level residual checkpoints
  - greedy refusal-marker behavior check
  - paired refusal and harmfulness direction separation

## Results

- Prompt behavior matched expectation on all `6 / 6` prompts.
- Refusal localization peaked at assistant-prefill layer `22`.
- Harmfulness localization peaked at instruction-final layer `25`.
- Both primary direction checks separated their intended pairs perfectly on the tiny
  split:
  - refusal confirm pair accuracy `= 1.0`
  - harmfulness confirm pair accuracy `= 1.0`

## Limitations

- This is only a runner smoke. The `1 / 1` split is too small to interpret the
  cross-direction metrics or any broader safety story.

## Next Steps

- Run the full `6 / 6` aligned-Gemma workflow artifact.
- Keep any claim boundary at “workflow validation” until the held-out split stays
  clean and later causal mediator checks exist.
