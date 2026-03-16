# Preregistration

## Scope

This preregistration covers the local depth-routing experiment defined by:

- `research/master-research-document.docx`
- `research/decision-matrix.md`
- `research/artifact2.md`
- `research/artifact3.md`

## Framing Lock

- Oracle-alpha is an upper bound on the routing signal available in standard architectures, more precisely the routing signal recoverable from fixed standard-model representations.
- Oracle-alpha is a lower bound on the benefit of depth routing once routing and computation can co-adapt.
- Oracle-alpha is not a trained AttnRes simulation and does not include the co-adaptation feedback loop.
- The strongest frozen-model claim is recovery of an effective depth mixture; stronger language about an internal router requires additional causal validation.
- Any claim about trained Attention Residuals beyond comparison or motivation must be marked as inference.

## Primary Hypothesis

Standard transformer residual streams contain latent, input-dependent depth-routing structure: there exist non-uniform softmax-constrained routing weights that improve next-token loss relative to uniform routing.

## Secondary Hypotheses

1. The recovered routing patterns are structured rather than random.
2. The routing patterns correlate with input strata, token position, and difficulty.
3. The routing patterns recover the qualitative Figure 8 signatures:
   - diagonal dominance
   - embedding persistence
   - layer-type specialization
   - learned skip connections
4. The softmax-constrained regime outperforms or meaningfully differs from unconstrained and top-k routing.
5. The learned router query vectors are a useful w_l analog for geometry analysis.
6. Routing structure supports a meaningful block hypothesis around approximately 8 clusters.
7. Refusal or honesty-related features show measurable routing differences in the safety lane.

## Null Models

- uniform
- random Dirichlet routing
- magnitude-proportional routing
- last-layer-only routing

## Required Regimes

- softmax-constrained
- unconstrained
- top-k

## Phase Gates

### Phase 1: Oracle-Alpha Feasibility and Stability

- Minimum sample size: `100` sequences
- Acceptance gate: mean cross-entropy improvement greater than `0.01` nats over uniform
- Significance gate: `p < 0.01`
- Test: paired t-test over per-sequence mean loss deltas
- All null models must be reported
- Small-scale scale-up gate: `d > 0.2` versus the random baseline
- Use a pilot tranche for method selection and debugging, and hold out a confirmatory tranche for claim-bearing statistics
- Use bootstrap confidence intervals with at least `1000` resamples for claim-bearing estimates
- Run a stability suite over optimization restarts and prompt resamples or paraphrases before moving to pattern interpretation
- Run an out-of-sample predictiveness check on the confirmatory tranche; if recovered alpha structure does not generalize beyond descriptive fitting, weaken the claim accordingly
- Use MIB as a benchmark anchor or sanity control when the task-model pair is compatible; if omitted, document why

### Phase 2: Pattern Analysis

- Primary routing distance: Jensen-Shannon divergence
- If hierarchical clustering is run directly on Jensen-Shannon distances, use average or complete linkage
- Ward linkage is only allowed after an explicit Euclidean embedding step
- Routing clusters must beat the random-baseline structure check
- Token-position alignment must be preserved when correlating routing with SAE activations
- Primary clustering threshold: `silhouette > 0.2`
- Statistical reporting must include Cohen's d and bootstrap confidence intervals

### Phase 3: Figure 8 Validation

`Figure 8` here refers to Figure 8 in `research/Attention_Residuals.pdf`.

Each pattern must be operationalized before inspecting aggregate heatmaps:

- diagonal dominance: `locality score > 1/L`
- embedding persistence: `alpha_0` shows above-uniform weight at deep layers
- layer-type specialization: `Entropy(pre-attn) > Entropy(pre-MLP)`
- learned skip connections: structural off-diagonal peaks where `α* > 2/L`
- Strong claims that frozen-model routing matches trained routing require a reproducible proxy, such as a small local AttnRes reproduction or another open depth-mixing comparison. Without that, this lane is interpreted as comparison against the published pattern surface only.

### Phase 4: Comparison Regimes

All routing regimes use matched initialization.

- softmax-constrained: `α = softmax(z)`
- unconstrained: sigmoid gating without sum-to-1 constraint
- top-k: `k ∈ {2,4,8,L/4,L/2}`

If softmax-constrained does not beat or separate from unconstrained, no claim about competition being uniquely beneficial may be made.

### Phase 5: Tool-Breakage Demonstration

The tool-breakage lane is mandatory.

- preferred demonstration: factual recall on Gemma-2-2B
- required outputs: routing-aware lens comparison and intervention sensitivity analysis
- primary figure: original-model versus routed-model traces under both raw logit lens and tuned lens
- if Gemma-2 lacks an off-the-shelf tuned lens, train a custom lens or move the tuned-lens comparison to a secondary model rather than silently dropping it
- raw logit lens is not assumed to be smooth or monotonic in the original model
- retain the legacy threshold language for continuity: `non-monotonic curves on >50% of prompts`, but only interpret it relative to the original-model baseline and tuned-lens-aware comparison
- success threshold for the strong claim: routing increases non-monotonicity or rank-instability relative to the original-model baseline on >50% of prompts, with tuned-lens-aware comparison reported alongside raw logit lens
- failure condition: if routing leaves raw and tuned-lens behavior qualitatively unchanged relative to the original-model baseline, the breakage claim must be weakened
- controlled dynamic-routing counterfactual required as a confirmatory control
- MIB-compatible or other benchmarked causal-localization controls should be used where the task-model pair permits them

### Phase 6: Router Training and Geometry

- router architecture: 2-layer MLP on h_1[t]
- router input is per-token, not a single global sequence vector
- pilot comparison: test `h_1[t]` against an early contextual state such as `h_4[t]` on the pilot tranche, then lock the choice before confirmatory training
- training plan: oracle-alpha distillation, then end-to-end refinement
- learned router query vectors must be analyzed as a w_l analog
- geometry outputs must include cosine structure and layer-function hypotheses
- router readiness target: `R^2 > 0.5`

### Phase 7: Block Structure and Safety

- explicitly test whether approximately 8 clusters emerge
- run the safety lane on refusal or honesty-related features
- safety lane runs as three stages: layer localization, feature discovery and validation, then mediator-conditioned routing analysis
- safety lane begins with refusal-feature discovery and validation; do not assume pre-labeled refusal features already exist in the local SAE workflow
- distinguish harmfulness-encoding features from refusal-execution features before causal interpretation
- follow `background-work/SAFETY_PUBLICATION_POLICY.md` for any external write-up touching refusal or jailbreak-adjacent findings

## Overclaim Guardrails

Do not claim:

- that frozen-model oracle-alpha reveals what co-adapted AttnRes layers would necessarily learn
- that visual resemblance to Figure 8 is enough by itself
- that safety-relevant routing patterns imply practical control without causal evidence

## Implementation Constraints

- Uniform-routing reconstruction sanity checks target agreement with the model's original logits after the model's own final normalization, not `logits / L`.
- Exact per-source routed-logit decomposition must use the shared final normalization factor from the full routed mixture. Per-source LayerNorm or per-source RMSNorm is not exact.
- Claim-bearing Figure 8 and layer-type-specialization analyses require sublayer outputs rather than `resid_post`-only caches.
- Token-level significance tests require explicit dependence-aware justification. The default unit for claim-bearing tests is the sequence.
- Tool-breakage claims must compare against the original-model baseline and include a tuned lens comparison; raw logit lens behavior alone is insufficient.
- Claim-bearing pattern interpretation requires stability reporting and out-of-sample alpha predictiveness rather than descriptive reconstruction alone.

## Reproducibility

- Fix random seeds before running claim-bearing experiments.
- Log hyperparameters before runs.
- Save a pilot/confirmatory split before tuning prompts, thresholds, or architecture choices.
- Freeze the local dependency set before scientific runs.
- Treat training-dynamics claims from checkpoints as within-run evolution claims unless stronger counterfactual evidence is recorded.
- Publicly pre-register this plan on LessWrong before claim-bearing execution.
