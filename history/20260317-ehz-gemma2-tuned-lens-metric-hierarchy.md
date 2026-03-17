ABOUTME: Records the tuned-lens metric hierarchy decision for the Gemma-2 tool-breakage lane.
ABOUTME: Anchors future routed-versus-original factual-recall work to the saved same-model viability artifact.

# EHZ Decision Memo: Gemma-2 Tuned-Lens Metric Hierarchy

## Motivation

`resattn-ehz` exists because `resattn-5k9` proved that a custom same-model tuned lens on `google/gemma-2-2b` is operationally viable, but it also made the metric story asymmetric. The held-out distributional metrics improved strongly, while final-position factual-recall top-1 improved only slightly. Before the routed-versus-original tool-breakage lane starts, the repo needs an explicit answer to a simple question: what counts as the primary tuned-lens-aware baseline metric?

## Decision

- Use held-out KL to the model's final output distribution as the primary tuned-lens baseline metric for the Gemma-2 tool-breakage lane.
- Require mean top-1 agreement, final-position KL, and final-position top-1 to be reported as secondary diagnostics alongside the primary KL metric.
- Do not block the routed-versus-original tool-breakage lane on sharpening the tuned-lens objective for answer-token recovery right now.
- Do block answer-token-facing interpretations on explicit final-position evidence; KL improvement alone does not license claims about recovering or obscuring the correct factual-recall answer token.

## Rationale

- The saved viability artifact is strongest on held-out distributional recovery:
  - mean KL improved from `11.3881` to `3.4580`
  - mean top-1 improved from `0.1863` to `0.5119`
  - final-position KL improved from `14.8685` to `7.5721`
  - final-position top-1 improved only from `0.1010` to `0.1250`
- Layerwise behavior tells the same story:
  - `25 / 26` layers improved on held-out KL
  - `25 / 26` layers improved on mean top-1
  - `25 / 26` layers improved on final-position KL
  - only `4 / 26` layers improved on final-position top-1
- The tool-breakage lane is about whether interpretability traces become less faithful or more unstable under routing relative to the original-model baseline. For that comparison, a metric that tracks full held-out distributional fidelity is the cleaner primary baseline than a weak answer-token proxy.
- At the same time, the task surface is factual recall, so final-position metrics remain scientifically important. They should constrain interpretation even if they are not the primary gating metric for entering the routed lane.

## Rejected Alternatives

- Make final-position top-1 the primary tuned-lens baseline metric now:
  - rejected because the current pilot does not support it, and forcing that choice would delay the routed-versus-original lane until a different tuned-lens objective is built
  - also rejected because it would conflate "better global lens fidelity" with "better answer-token recovery"
- Immediately replace the residual-MSE objective with a logit-space or answer-token-focused objective:
  - rejected as the default next step because the same-model Gemma viability question is already answered
  - retained as a future option if later tool-breakage claims need stronger final-position behavior than the current baseline provides

## Consequences

- The next routed-versus-original factual-recall work should center held-out KL as the primary tuned-lens-aware baseline metric.
- Every tool-breakage write-up should still report final-position metrics and avoid answer-token-facing language unless those metrics support it directly.
- If a future lane explicitly wants stronger claims about correct-answer visibility rather than broader lens faithfulness, that lane should either sharpen the tuned-lens objective or add a separate answer-token-focused tuned-lens variant rather than quietly repurposing the current baseline.
