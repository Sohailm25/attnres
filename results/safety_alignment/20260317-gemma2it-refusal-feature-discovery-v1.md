# ABOUTME: Summarizes the first held-out refusal-feature discovery workflow validation on aligned Gemma.
# ABOUTME: Records layer localization, direction discovery, and behavior checks before any safety-routing claims.

## Motivation

Validate the refusal-feature discovery workflow required by `resattn-3f1` before the
safety lane makes any routing claim. The workflow needed to show three things on a
frozen prompt split:

1. the model actually exhibits refusal on the intended prompts,
2. candidate refusal and harmfulness layers can be localized separately,
3. a simple mechanistic anchor can generalize out of sample.

## Methods

- Model: `google/gemma-2-2b-it`
- Collection: `safety_refusal_discovery_v1`
- Size: `6` pilot groups and `6` confirm groups
- Prompt structure per group:
  - `refusal`: harmful request expected to elicit refusal
  - `harmful_context`: high-level harmful-context explanation expected to answer safely
  - `benign`: safe control expected to answer normally
- Positions:
  - harmfulness localization at `instruction_final`
  - refusal localization at `assistant_prefill`
- Discovery:
  - localize layers by mean paired cosine divergence on the pilot groups
  - discover a refusal direction from `refusal - harmful_context`
  - discover a harmfulness direction from `harmful_context - benign`
- Validation:
  - greedy refusal-marker behavior checks
  - held-out paired projection margins and pair accuracy
  - cross-direction controls at the same evaluation points
- Reproducibility:
  - prompt-level residual checkpoints saved after each prompt under the artifact
    directory
  - rerunning the exact command reused the saved checkpoints without rewriting them

## Results

### Behavior checks

- Refusal behavior hit rate:
  - pilot `= 1.0`
  - confirm `= 1.0`
- Non-refusal behavior pass rate:
  - pilot `= 1.0`
  - confirm `= 1.0`

### Layer localization

- Refusal localization peaked at assistant-prefill layer `22`
  with mean divergence `= 0.4812`.
- Harmfulness localization peaked at instruction-final layer `25`
  with mean divergence `= 0.3888`.

### Held-out direction validation

- Refusal direction on confirm:
  - primary mean margin `= 506.5991`
  - primary pair accuracy `= 1.0`
  - cross-direction mean margin `= 0.7924`
  - cross-direction pair accuracy `= 0.6667`
- Harmfulness direction on confirm:
  - primary mean margin `= 146.8030`
  - primary pair accuracy `= 1.0`
  - cross-direction mean margin `= 0.4251`
  - cross-direction pair accuracy `= 0.8333`
- Direction cosine:
  - refusal versus harmfulness `= 0.0064`

## Interpretation

- The workflow validates cleanly as a first aligned-Gemma safety artifact.
- The model/prompt surface is not the blocker:
  refusal and non-refusal behavior matched the intended labels on the full frozen
  split.
- The localized refusal and harmfulness anchors are strongly distinct at the level
  this workflow asked for:
  different preferred positions, different peak layers, and almost zero cosine
  similarity.
- The cross-direction metrics are not zero, so this is not evidence for a single
  perfectly disentangled safety direction.

## Limitations

- This is still workflow validation, not a mediator-conditioned routing result.
- The prompt triples are intentionally small and templated; they support method
  validation, not broad claims about real-world safety behavior.
- The current validation is held-out separability, not a causal intervention study.

## Next Steps

- Close `resattn-3f1` as a workflow-validation success.
- Keep strong safety-routing claims blocked on a later causal mediator check.
- Use the saved prompt-level checkpoints for the next aligned-Gemma safety follow-up
  instead of rerunning the full prompt surface.
