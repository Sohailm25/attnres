# ABOUTME: Summarizes the first confirm-split causal refusal-direction intervention check on aligned Gemma.
# ABOUTME: Records bounded mediator evidence before any safety-routing analysis claims.

## Motivation

The aligned-Gemma refusal-feature workflow was already validated, but the safety
lane still lacked the prereg-required causal mediator check before any
mediator-conditioned routing analysis. `resattn-73l` asked for the smallest
honest follow-up: intervene on the localized refusal and harmfulness directions
and test whether the refusal mediator moves behavior or safe continuation
preference on the frozen confirm split.

## Methods

- Model: `google/gemma-2-2b-it`
- Collection: `safety_refusal_discovery_v1`
- Size: `6` pilot groups, `6` confirm groups
- Localized sites from the frozen pilot groups:
  - refusal: assistant-prefill layer `22`
  - harmfulness: instruction-final layer `25`
- Intervention type:
  - replace the scalar projection on the discovered direction with the pilot mean
    projection for the comparison role while keeping orthogonal residual content
    fixed
- Arms:
  - refusal suppression on refusal prompts
  - harmfulness suppression on refusal prompts
  - refusal injection on harmful-context prompts
  - refusal injection on benign prompts
  - harmfulness injection on benign prompts
- Readouts:
  - primary behavior check: greedy refusal-marker hit rate
  - secondary preference check: mean logprob margin between a group-matched safe
    refusal continuation and a matched non-refusal continuation

## Results

### Localized mediator geometry

- Refusal localization stayed at assistant-prefill layer `22`.
- Harmfulness localization stayed at instruction-final layer `25`.
- Direction cosine stayed near zero: `0.0064`.

### Refusal prompts

- Refusal suppression on refusal prompts did not flip the greedy marker:
  - baseline refusal rate `= 1.0`
  - intervened refusal rate `= 1.0`
- But it did reduce the safe refusal preference margin:
  - baseline margin `= 2.4995`
  - intervened margin `= 2.3105`
  - margin delta `= -0.1891`
- The matched harmfulness suppression control stayed flat:
  - margin delta `= 0.0`

### Harmful-context prompts

- Refusal injection on harmful-context prompts produced the only greedy flip in
  the artifact:
  - baseline refusal rate `= 0.0`
  - intervened refusal rate `= 0.1667` (`1 / 6`)
- The same arm also moved the safe refusal preference margin upward:
  - baseline margin `= -0.6816`
  - intervened margin `= -0.5075`
  - margin delta `= +0.1741`

### Benign prompts

- Refusal injection on benign prompts did not trigger explicit greedy refusal:
  - baseline refusal rate `= 0.0`
  - intervened refusal rate `= 0.0`
- But it still moved the refusal-versus-benign continuation margin upward:
  - baseline margin `= -1.7987`
  - intervened margin `= -1.5613`
  - margin delta `= +0.2374`
- The matched harmfulness injection control stayed flat:
  - margin delta `= 0.0`

## Interpretation

- This is a real causal-mediator pass for the safety lane.
- The localized refusal direction has causal bite on the confirm split:
  suppressing it weakens refusal-style continuation preference on refusal
  prompts, and injecting it increases refusal-style continuation preference on
  harmful-context prompts.
- The harmfulness controls do not recapitulate the same effect at their own
  localized site, which is the strongest current evidence that the aligned-Gemma
  refusal mediator is not just generic harmfulness recognition.
- The binary refusal marker is still mostly saturated, so the strongest honest
  claim is about safe continuation preference plus one observed greedy flip, not
  large-scale binary behavior reversal.
- The benign-prompt shift means the mediator is not a perfectly selective
  “harmful request only” switch. It is better read as a refusal-style execution
  direction with limited specificity at this bounded intervention surface.

## Limitations

- This is not yet a routing analysis and does not license any claim that routing
  differences are mediated by this direction.
- The intervention is one-dimensional coefficient replacement, not full circuit
  tracing or a richer concept-cone analysis.
- The prompt collection remains small and templated.

## Next Steps

- Close `resattn-73l` as the bounded causal mediator check.
- Keep future safety-routing claims conditioned on this mediator evidence rather
  than on descriptive separability alone.
- Open the next safety issue for mediator-conditioned routing analysis on the
  aligned Gemma lane.
