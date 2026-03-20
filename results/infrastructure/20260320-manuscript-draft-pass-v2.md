ABOUTME: Refines manuscript prose with concrete table/figure placeholders and tightened mixed-lane caveat wording.
ABOUTME: Uses exact C1-C6 metrics from saved artifacts without adding new experiments.

# Manuscript Draft Pass v2

## Working Title

Latent Depth-Routing Spectroscopy in Standard Transformers: Primary-Model
Oracle Evidence, Structured Regimes, and Boundaries

## Abstract (Refined Draft)

We analyze frozen `google/gemma-2-2b` with oracle-alpha to test whether standard
transformers expose an input-dependent effective depth mixture. On a stratified
confirm surface (`1024` prompts), optimized routing beats uniform and all
preregistered null baselines (`+1.6299` nats vs uniform; `1024/1024` prompts
positive), and held-out predicted routing remains positive (`+0.8292` nats;
`958/1024` positive). Competitive softmax routing is stronger than matched
unconstrained and top-k alternatives on confirm (`128/128` wins versus each
tested alternative). Pattern analyses are non-random and task-conditioned, with
strongest raw-source organization in factual recall. Extension lanes are
bounded: tool-breakage remains mixed under seeded dynamic donor controls, safety
mediator activity remains refusal-collapsed on the v3 quartet surface, and the
OIH anchor is positive but calibration-sensitive (`+0.0478` dynamic minus
calibrated static). We explicitly treat oracle-alpha as frozen-model effective
depth-mixture analysis, not trained-router equivalence.

## 1. Introduction

This work asks whether frozen standard transformers expose recoverable,
input-dependent depth-routing structure. The target object is an effective depth
mixture over fixed residual sources. The analysis does not claim that a
co-adapted trained routing system would recover the same routing object.

The paper contributes a broad primary-model evidence surface with confirm-split
controls, explicit regime comparisons, and bounded extension-lane claims.

## 2. Methods and Controls

- Model spine: `google/gemma-2-2b` on local MPS.
- Main oracle surface: `registry_v5` (`256` pilot / `1024` confirm).
- Statistical unit: sequence-level for claim-bearing tests.
- Reconstruction constraints: original-logit target after final normalization;
  shared final normalization for exact source decomposition.
- Regime comparison: matched-init softmax, unconstrained sigmoid, and top-k.
- Extension controls:
  - tool-breakage: routed-vs-original plus dynamic counterfactual arms.
  - safety: localization and behavior validation before mediator partition.
  - OIH: explicit MIB-compatible anchor lane with calibrated static comparator.

## 3. Results

### 3.1 Core Claims (C1-C3)

**Table 1 (insert in manuscript): Core quantitative anchors**

| Claim | Metric | Value |
|---|---|---|
| C1: oracle signal vs uniform | Oracle mean improvement | `+1.6299` nats |
| C1: oracle signal robustness | Positive prompts | `1024/1024` |
| C1: held-out predictiveness | Predicted mean improvement | `+0.8292` nats |
| C1: held-out predictiveness | Predicted positives | `958/1024` |
| C1: alpha-shape diagnostics | Confirm `R^2` / mean JS | `0.2392` / `0.0985` |
| C3: regime comparison | Softmax vs unconstrained mean advantage | `+0.4888` nats (`128/128` wins) |
| C3: regime comparison | Softmax vs each top-k (`k=2,4,8,13,26`) | `128/128` wins per `k` |

**Figure 1 placeholder (insert):** Confirm loss by routing regime
(softmax, unconstrained, top-k family) with confidence intervals and per-prompt
win counts.

### 3.2 Structure Claim (C2)

Pattern-analysis artifacts show non-random and task-conditioned routing
structure. Factual recall is the strongest raw-source signal stratum, and
grouped structure persists more broadly across strata.

**Figure 2 placeholder (insert):** Stratum-conditioned routing structure
summary: factual recall vs reasoning/math vs code/procedural vs general text,
including predicted deltas and JS metrics.

## 4. Extension Lanes (C4-C6)

**Table 2 (insert in manuscript): Bounded extension-lane outcomes**

| Claim | Status | Primary quantitative anchor | Bounded interpretation |
|---|---|---|---|
| C4: tool-breakage | mixed | routed minus `prompt_permuted_alpha = -0.0343`; routed minus `within_family_permuted_alpha = -0.2084`; routed minus `cross_family_permuted_alpha = -0.1344`; routed minus `pilot_mean_alpha = +1.0898` | Routed-vs-fixed-alpha is positive, but dynamic donor controls remain mixed/negative on pooled primary metric. |
| C5: safety | mixed | v3 validation gates all clean (`1.0` behavior/pair metrics); confirm mediator-active roles `{'refusal': 3}` only | Refusal/harmfulness signals are present, but mediator-expansion claim is blocked on current surface. |
| C6: OIH | mixed | predicted `+0.2290`, oracle `+0.5709`, dynamic minus calibrated static `+0.0478` | Positive extension anchor with modest dynamic-over-calibrated-static margin. |

**Figure 3 placeholder (insert):** Extension-lane boundary panel with three
subplots/tables for tool-breakage, safety mediator partition, and OIH calibrated
comparison.

## 5. Tightened Caveat Language (for manuscript copy)

- **Frozen-model boundary:** “These results characterize recoverable routing
  signal in fixed representations and do not establish trained-router
  equivalence.”
- **Tool-breakage boundary:** “Dynamic counterfactual controls leave this lane
  mixed; strongest same-model prompt-specific breakage language is not
  supported broadly.”
- **Safety boundary:** “Mediator-conditioned role expansion is not supported on
  current v3 prompts despite clean validation and localized refusal/harmfulness
  structure.”
- **OIH boundary:** “Dynamic-over-static signal is positive against calibrated
  static comparator but modest, so this lane is extension evidence rather than
  a standalone central claim.”

## 6. Explicit Non-Claims

- No frozen-model oracle-alpha equals trained AttnRes routing claim.
- No strong Figure 8 trained-routing alignment claim.
- No router-distillation readiness claim (`R^2 > 0.5` gate remains unmet).
- No claim-bearing training-dynamics result in this cycle.

## 7. Next Writing Move

Integrate this v2 draft into manuscript sections with final figure/table assets
and keep claim wording pinned to C1-C6 status labels (`supported` vs `mixed`)
from the claim matrix.
