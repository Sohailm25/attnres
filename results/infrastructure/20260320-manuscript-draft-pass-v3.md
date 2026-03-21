ABOUTME: Compresses manuscript prose to a reviewer-efficient narrative while preserving C1-C6 claim boundaries.
ABOUTME: Keeps mixed-lane caveats explicit and removes repetitive exposition from v2.

# Manuscript Draft Pass v3 (Compressed)

## Abstract

On frozen `google/gemma-2-2b`, oracle-alpha recovers a strong input-dependent
effective depth mixture signal on a stratified confirm surface (`1024` prompts):
oracle routing improves over uniform by `+1.6299` nats (`1024/1024` positive),
and held-out predicted routing remains positive (`+0.8292`, `958/1024` positive).
Softmax-constrained routing outperforms matched unconstrained and top-k regimes
on confirm (`128/128` wins against each tested alternative). Routing structure
is non-random and task-conditioned, with strongest raw-source organization in
factual recall. Extension lanes are bounded: tool-breakage is mixed under seeded
dynamic donor controls, safety mediator activity remains refusal-collapsed on
v3 prompts, and OIH is positive but calibration-sensitive
(`+0.0478` dynamic minus calibrated static). These are frozen-model results, not
trained-router equivalence claims.

## Core Methods (Condensed)

- Primary surface: `registry_v5` (`256` pilot / `1024` confirm).
- Statistical unit: sequence-level for claim-bearing tests.
- Reconstruction discipline: final-normalization-consistent uniform/logit checks
  and shared-final-norm source decomposition.
- Regimes: matched-init softmax, unconstrained sigmoid gating, and top-k.
- Mixed-lane controls:
  - tool-breakage includes seeded dynamic donor counterfactuals,
  - safety includes v3 behavior/localization validation before mediator read,
  - OIH uses calibrated static comparison.

## Results Snapshot

### C1-C3 (Supported)

| Claim | Anchor |
|---|---|
| C1 (oracle signal) | Oracle `+1.6299` vs uniform; `1024/1024` positive; predicted `+0.8292`; `R^2=0.2392`, mean JS `0.0985` |
| C2 (structured routing) | Non-random/task-conditioned structure with strongest factual-recall raw-source organization |
| C3 (regime separation) | Softmax vs unconstrained: `+0.4888` nats (`128/128` wins); softmax vs each top-k (`k=2,4,8,13,26`): `128/128` wins |

### C4-C6 (Mixed, Bounded)

| Claim | Anchor | Bound |
|---|---|---|
| C4 tool-breakage | routed minus prompt/within/cross dynamic arms: `-0.0343 / -0.2084 / -0.1344`; routed minus `pilot_mean_alpha`: `+1.0898` | Routed-vs-fixed-alpha positive, dynamic donor controls mixed/negative on pooled primary metric |
| C5 safety | v3 validation clean (`1.0` behavior/pair metrics); mediator-active confirm roles refusal-only | Mediator-expansion claim blocked on current surface |
| C6 OIH | predicted `+0.2290`, oracle `+0.5709`, dynamic minus calibrated static `+0.0478` | Positive extension anchor with modest dynamic-over-calibrated-static margin |

## Figure/Table Placeholders

1. Figure 1: confirm loss by routing regime with per-prompt wins.
2. Figure 2: stratum-conditioned routing structure summary.
3. Figure 3: extension-lane boundary panel (tool-breakage, safety, OIH).
4. Table 1: C1-C3 quantitative anchors.
5. Table 2: C4-C6 bounded outcomes.

## Explicit Non-Claims

- No frozen-model oracle-alpha equals trained AttnRes routing claim.
- No strong Figure 8 trained-routing alignment claim.
- No router-distillation readiness claim (`R^2 > 0.5` remains unmet).
- No claim-bearing training-dynamics result in this cycle.
