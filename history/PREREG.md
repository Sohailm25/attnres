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

### Phase 1: Oracle-Alpha Feasibility

- Minimum sample size: `100` sequences
- Acceptance gate: mean cross-entropy improvement greater than `0.01` nats over uniform
- Significance gate: `p < 0.01`
- Test: paired t-test
- All null models must be reported
- Small-scale scale-up gate: `d > 0.2` versus the random baseline

### Phase 2: Pattern Analysis

- Primary routing distance: Jensen-Shannon divergence
- Routing clusters must beat the random-baseline structure check
- Token-position alignment must be preserved when correlating routing with SAE activations
- Primary clustering threshold: `silhouette > 0.2`
- Statistical reporting must include Cohen's d and bootstrap confidence intervals

### Phase 3: Figure 8 Validation

Each pattern must be operationalized before inspecting aggregate heatmaps:

- diagonal dominance: `locality score > 1/L`
- embedding persistence: `alpha_0` shows above-uniform weight at deep layers
- layer-type specialization: `Entropy(pre-attn) > Entropy(pre-MLP)`
- learned skip connections: structural off-diagonal peaks where `α* > 2/L`

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
- primary figure: standard versus oracle-alpha logit-lens traces
- success threshold for the strong claim: non-monotonic curves on >50% of prompts
- failure condition: if reweighting leaves standard tool outputs qualitatively unchanged, the breakage claim must be weakened

### Phase 6: Router Training and Geometry

- router architecture: 2-layer MLP on h_1
- training plan: oracle-alpha distillation, then end-to-end refinement
- learned router query vectors must be analyzed as a w_l analog
- geometry outputs must include cosine structure and layer-function hypotheses
- router readiness target: `R^2 > 0.5`

### Phase 7: Block Structure and Safety

- explicitly test whether approximately 8 clusters emerge
- run the safety lane on refusal or honesty-related features

## Overclaim Guardrails

Do not claim:

- that frozen-model oracle-alpha reveals what co-adapted AttnRes layers would necessarily learn
- that visual resemblance to Figure 8 is enough by itself
- that safety-relevant routing patterns imply practical control without causal evidence

## Reproducibility

- Fix random seeds before running claim-bearing experiments.
- Log hyperparameters before runs.
- Freeze the local dependency set before scientific runs.
- Publicly pre-register this plan on LessWrong before claim-bearing execution.
