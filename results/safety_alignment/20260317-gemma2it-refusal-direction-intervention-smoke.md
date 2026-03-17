# ABOUTME: Summarizes the 1/1 smoke for the aligned-Gemma refusal-direction intervention runner.
# ABOUTME: Records whether the bounded causal mediator check is operational before the full confirm run.

## Motivation

`resattn-73l` needed one bounded causal mediator check before any later
mediator-conditioned routing analysis on the safety lane. The smallest honest
version was a `1 / 1` smoke on the frozen aligned-Gemma prompt collection.

## Methods

- Model: `google/gemma-2-2b-it`
- Collection: `safety_refusal_discovery_v1`
- Size: `1` pilot group, `1` confirm group
- Localizations reused from the same smoke:
  - refusal: assistant-prefill layer `22`
  - harmfulness: instruction-final layer `25`
- Intervention type:
  - replace the scalar projection on the discovered direction with the pilot mean
    projection for the comparison role while leaving orthogonal residual content
    unchanged
- Arms:
  - refusal suppression on refusal prompts
  - harmfulness suppression on refusal prompts
  - refusal injection on harmful-context prompts
  - refusal injection on benign prompts
  - harmfulness injection on benign prompts
- Readouts:
  - primary behavior check: greedy refusal-marker hit rate
  - secondary preference check: mean logprob margin between the group-matched
    safe refusal continuation and the matched non-refusal continuation

## Results

- The original binary refusal-marker readout was too coarse on its own:
  - no arm changed the `1 / 1` greedy refusal marker
- The continuation-preference readout moved in the expected direction for the
  refusal arms:
  - refusal suppression on refusal prompts: margin delta `-0.2049`
  - refusal injection on harmful-context prompts: margin delta `+0.1583`
  - refusal injection on benign prompts: margin delta `+0.2171`
- The matched harmfulness controls stayed flat on the same prompts:
  - harmfulness suppression on refusal prompts: margin delta `0.0`
  - harmfulness injection on benign prompts: margin delta `0.0`

## Limitations

- This is still only a `1 / 1` smoke and should not be interpreted as a
  confirmatory causal result.
- The greedy refusal marker did not flip even when the preference metric moved,
  so the runner needed both readouts before the full run.

## Next Steps

- Keep the continuation-preference metric as a required secondary readout.
- Run the full frozen `6 / 6` confirm artifact on the same aligned-Gemma setup.
