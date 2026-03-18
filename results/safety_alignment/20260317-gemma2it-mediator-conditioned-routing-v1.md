# ABOUTME: Summarizes the first mediator-conditioned safety routing artifact on aligned Gemma.
# ABOUTME: Records bounded trajectory evidence without overstating prompt-level mediator separation.

## Motivation

The aligned-Gemma safety lane already had two prerequisites in place:

- a held-out refusal-feature discovery validator
- a bounded causal refusal-direction intervention artifact

`resattn-h1p` asked for the smallest honest third-stage follow-up: reuse the
validated refusal mediator and the frozen prompt collection to test whether
mediator-conditioned depth trajectories reveal anything beyond a relabeled
refusal-versus-non-refusal split.

## Methods

- Model: `google/gemma-2-2b-it`
- Collection: `safety_refusal_discovery_v1`
- Size: `6` pilot groups, `6` confirm groups
- Checkpoint surface:
  - reused saved prompt-level residual checkpoints under
    `results/safety_alignment/20260317-gemma2it-mediator-conditioned-routing-v1/checkpoints/prompt_residuals/`
- Pilot-localized directions:
  - refusal at assistant-prefill layer `22`
  - harmfulness at instruction-final layer `25`
- Confirm readouts:
  - mediator-active versus mediator-inactive partition induced by the pilot
    refusal threshold
  - refusal-direction and harmfulness-direction mean depth trajectories by role
  - intervention-conditioned trajectory comparisons for:
    - refusal suppression on refusal prompts
    - harmfulness suppression on refusal prompts
    - refusal injection on harmful-context prompts
    - refusal injection on benign prompts

## Results

### Mediator partition

- The pilot-derived refusal threshold was `79.0206`.
- The confirm partition still collapsed exactly onto prompt roles:
  - active prompts: `6`, all `refusal`
  - inactive prompts: `12`, split evenly across `harmful_context` and `benign`
- Mean refusal-direction projection at layer `22` stayed widely separated:
  - active mean `= 323.2004`
  - inactive mean `= -163.4050`

### Role trajectories

- Refusal-direction trajectories on assistant-prefill residuals remained sharply
  separated at the selected layer:
  - refusal `= 323.2003`
  - harmful_context `= -183.3988`
  - benign `= -143.4112`
- The same ordering persisted at the final layer:
  - refusal `= 252.7739`
  - harmful_context `= -174.2734`
  - benign `= -132.8563`
- Harmfulness-direction trajectories on instruction-final residuals kept the
  intended role ordering:
  - harmful_context `= 192.7267`
  - refusal `= 74.5561`
  - benign `= 45.9237`

### Intervention-conditioned trajectories

- Refusal suppression on refusal prompts moved the refusal-direction trajectory
  strongly in the negative direction:
  - selected-layer delta `= -511.8212`
  - final-layer delta `= -372.8501`
- Refusal injection on harmful-context prompts moved the same trajectory toward
  the refusal manifold:
  - selected-layer delta `= +506.8928`
  - final-layer delta `= +369.9403`
- Refusal injection on benign prompts produced a similarly strong shift:
  - selected-layer delta `= +466.9052`
  - final-layer delta `= +358.7685`
- Harmfulness suppression on refusal prompts also moved the targeted projection
  strongly negative:
  - selected-layer delta `= -125.8243`
  - final-layer delta `= -125.8243`

## Interpretation

- This artifact clears the narrow workflow question behind `resattn-h1p`.
- The mediator-conditioned read is no longer just a relabeled prompt partition,
  because interventions at refusal layer `22` propagate large shifts to the
  final-layer refusal-direction trajectory on both refusal and non-refusal
  prompts.
- At the same time, the confirm mediator partition is still exactly
  `refusal` versus `non-refusal` on this frozen prompt set. So this is not
  evidence that the current collection exposes a subtler mediator-active subset
  inside a role class.
- The strongest honest claim is therefore:
  - the aligned-Gemma refusal mediator supports a bounded mediator-conditioned
    depth-trajectory analysis
  - causal manipulation of that mediator changes the full-depth refusal-style
    trajectory, not just the local selected-layer coefficient
- It is not yet evidence for a broader safety-routing law over a richer prompt
  surface.

## Limitations

- The confirm collection is still small and templated.
- The mediator-active partition remains role-collapsed on this prompt set.
- The harmfulness intervention site is the final layer, so its trajectory delta
  is not evidence of deeper-layer propagation in the same way as the refusal
  interventions at layer `22`.
- This artifact still does not localize routed alpha differences or establish a
  mediator-conditioned routing mechanism in the oracle-alpha sense.

## Next Steps

- Close `resattn-h1p` as a bounded stage-3 safety artifact.
- Keep strong safety-routing claims blocked on `resattn-mo5`, which should test
  a broader prompt surface that can break the current role collapse.
- Return repo priority to the open Figure 8 and tool-breakage decision issues.
