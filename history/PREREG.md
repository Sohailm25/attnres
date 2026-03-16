# Preregistration

## Scope

This preregistration covers the local depth-routing experiment defined by:

- `research/master-research-document.docx`
- `research/decision-matrix.md`
- `research/artifact2.md`
- `research/artifact3.md`

## Framing Lock

- Oracle-alpha is an upper bound on the routing signal available in standard architectures under fixed representations.
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
- All null models must be reported

### Phase 2: Pattern Analysis

- Routing clusters must beat the random-baseline structure check
- Token-position alignment must be preserved when correlating routing with SAE activations

### Phase 3: Figure 8 Validation

Each pattern must be operationalized before inspecting aggregate heatmaps:

- diagonal dominance
- embedding persistence
- layer-type specialization
- learned skip connections

### Phase 4: Comparison Regimes

All routing regimes use matched initialization.

- softmax-constrained
- unconstrained
- top-k

If softmax-constrained does not beat or separate from unconstrained, no claim about competition being uniquely beneficial may be made.

### Phase 5: Tool-Breakage Demonstration

The tool-breakage lane is mandatory.

- preferred demonstration: factual recall on Gemma-2-2B
- required outputs: routing-aware lens comparison and intervention sensitivity analysis
- failure condition: if reweighting leaves standard tool outputs qualitatively unchanged, the breakage claim must be weakened

### Phase 6: Router Training and Geometry

- learned router query vectors must be analyzed as a w_l analog
- geometry outputs must include cosine structure and layer-function hypotheses

### Phase 7: Block Structure and Safety

- explicitly test whether approximately 8 clusters emerge
- run the safety lane on refusal or honesty-related features

## Overclaim Guardrails

Do not claim:

- that frozen-model oracle-alpha reveals what co-adapted AttnRes layers would necessarily learn
- that visual resemblance to Figure 8 is enough by itself
- that safety-relevant routing patterns imply practical control without causal evidence
