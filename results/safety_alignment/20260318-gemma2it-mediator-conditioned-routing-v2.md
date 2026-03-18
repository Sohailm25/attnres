# ABOUTME: Summarizes the broadened aligned-Gemma mediator-conditioned routing follow-up on the v2 prompt surface.
# ABOUTME: Records a bounded negative result: the broadened surface still does not break the role-collapsed mediator partition.

## Motivation

`resattn-mo5` asked whether a broader aligned-Gemma prompt surface could expose
mediator-active prompts inside the non-refusal roles, rather than keeping the
refusal mediator perfectly aligned with outright refusal prompts.

## Methods

- Model: `google/gemma-2-2b-it`
- Collection: `safety_refusal_surface_v2`
- Size: `6` pilot groups, `6` confirm groups
- Localized directions:
  - refusal at assistant-prefill layer `22`
  - harmfulness at instruction-final layer `18`
- Checkpoint surface:
  - prompt-level residual checkpoints saved under
    `results/safety_alignment/20260318-gemma2it-mediator-conditioned-routing-v2/checkpoints/prompt_residuals/`
- Confirm readouts:
  - pilot-derived mediator threshold
  - mediator-active versus mediator-inactive partition
  - refusal-direction trajectories by role
  - intervention-conditioned trajectory comparisons

## Results

### Mediator partition

- The broadened-surface threshold was `135.4067`.
- The confirm partition still collapsed exactly onto role labels:
  - active prompts: `6`, all `refusal`
  - inactive prompts: `12`, split evenly across `harmful_context` and `benign`
- Refusal-direction selected-layer means remained sharply separated:
  - refusal `= 342.1530`
  - harmful_context `= -53.1067`
  - benign `= -70.1827`

### Intervention-conditioned trajectories

- Refusal suppression on refusal prompts still moved the trajectory strongly:
  - selected-layer delta `= -416.9160`
  - final-layer delta `= -306.1021`
- Refusal injection on harmful-context prompts still produced a large shift:
  - selected-layer delta `= +395.6044`
  - final-layer delta `= +284.3705`
- Refusal injection on benign prompts also remained large:
  - selected-layer delta `= +412.6804`
  - final-layer delta `= +315.3803`

## Interpretation

- The broadened surface answered the core `mo5` question negatively.
- Even after adding refusal-style non-refusal prompts, the mediator-active subset
  still lands exactly on outright refusal prompts rather than cutting across the
  non-refusal roles.
- That makes the current aligned-Gemma refusal mediator look more like an
  overt-refusal detector than a broader safety-routing partition on this small
  prompt family.
- At the same time, the intervention-conditioned trajectory story remains real:
  manipulating the refusal direction still propagates strong full-depth shifts on
  both refusal and non-refusal prompts.

## Limitations

- The broadened collection also exposed a validator-semantics gap: some
  refusal-style non-refusal prompts are flagged as behavior mismatches by the
  legacy refusal-marker heuristic.
- So this result rules out the stronger role-separation hope on the current
  broadened surface, but it does not mean every future safety-surface expansion
  would be uninformative.

## Next Steps

- Close `resattn-mo5` as a bounded negative result on the broadened surface.
- Track the validator-semantics follow-up separately before any future safety
  prompt-surface expansion.
