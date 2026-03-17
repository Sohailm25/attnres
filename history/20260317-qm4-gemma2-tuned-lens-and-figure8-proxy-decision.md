# QM4 Decision Memo: Gemma-2 Tuned Lens and Figure 8 Proxy

## Motivation

`resattn-qm4` existed to resolve two implementation-shaping decisions before the Figure 8 and tool-breakage lanes begin real execution:

1. whether the tuned-lens-aware comparison for tool-breakage should stay on the primary Gemma-2 model or move to a secondary model with better off-the-shelf lens support
2. what reproducible proxy is required before making strong claims that frozen-model routing aligns with trained AttnRes Figure 8 patterns

## Decision

- Tool-breakage will stay on the primary `google/gemma-2-2b` lane and use a custom Gemma-2 tuned lens trained locally.
- A secondary-model tuned-lens comparison is allowed only as supplementary context, not as the substitute that satisfies the primary tuned-lens requirement.
- Strong Figure 8 alignment claims require a small local AttnRes reproduction as the reproducible proxy.
- Until that proxy exists, the Figure 8 lane is limited to comparison against the published AttnRes pattern surface rather than claims of trained-routing alignment.

## Rationale

- The repo already locks Gemma-2-2B as the primary model for main results plus SAE and circuit tooling in `configs/experiment.yaml`. Moving the tuned-lens comparison to a secondary model would weaken the exact-model tool-breakage control the lane is supposed to provide.
- The deep-research review explicitly says there is no off-the-shelf Gemma-2 tuned lens and that the tool-breakage lane therefore needs a custom trained lens on the target model rather than silently dropping the comparison.
- The prior bounded `qm4` sidecar recommendation already converged on this same path: custom Gemma-2 tuned lens plus small local AttnRes reproduction.
- For Figure 8, the repo already treats strong trained-routing language as gated on a reproducible proxy. The strongest repo-grounded proxy is a small local AttnRes reproduction, because it directly addresses the co-adaptation objection and does not depend on non-released public small-scale checkpoints.

## Rejected Alternatives

- Secondary-model-only tuned-lens comparison:
  - rejected because it would satisfy the tuned-lens requirement on the wrong model and weaken the tool-breakage claim for the primary Gemma-2 lane
- Open depth-mixing baseline as the default Figure 8 proxy:
  - rejected as the default because it is less direct than a small AttnRes reproduction for claims specifically about AttnRes-style learned routing patterns
  - retained only as a fallback if local AttnRes reproduction proves infeasible and that infeasibility is logged explicitly

## Consequences

- The next tool-breakage implementation issue should train and validate a custom Gemma-2 tuned lens before claim-bearing routed-vs-original comparisons.
- The next strong Figure 8 alignment issue should build a small local AttnRes reproduction rather than overclaim from visual similarity to the published figure alone.
