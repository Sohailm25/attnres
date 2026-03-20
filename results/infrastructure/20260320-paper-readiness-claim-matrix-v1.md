ABOUTME: Summarizes paper-readiness status by mapping current claims to concrete artifacts and explicit blockers.
ABOUTME: Freezes the current claim boundaries so writing can proceed without accidental overclaiming.

# Paper-Readiness Claim Matrix v1

## Scope

This matrix reflects the current strongest defensible story on the primary `google/gemma-2-2b` spine, using saved artifacts already registered in `results/RESULTS_INDEX.md`.

Claim statuses:
- `supported`: evidence is sufficient for a bounded paper claim.
- `mixed`: real evidence exists, but interpretation needs explicit caveats.
- `blocked`: not ready for claim-bearing paper language.

## Claim Matrix

| Claim | Status | Primary evidence | Required caveat / blocker |
|---|---|---|---|
| Frozen-model oracle routing signal exists on primary Gemma and beats prereg nulls | supported | `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1.md`, `results/oracle_alpha/20260318-gemma2-registry-v5-synthesis-v1.md` | Keep framing as `effective depth mixture`, not trained-router equivalence |
| Routing structure is non-random and task-conditioned (especially factual recall) | supported | `results/pattern_analysis/20260318-gemma2-registry-v5-pattern-analysis-v1.md`, `results/block_structure/20260318-gemma2-factual-recall-routing-synthesis-v1.md` | Do not overstate global `~8`-cluster success |
| Softmax competitive routing is meaningfully better than unconstrained/top-k on confirm prompts | supported | `results/comparison_regimes/20260318-gemma2-regime-comparison-v1.md` | Keep claim tied to tested regimes and current prompt surface |
| Tool-breakage lane shows bounded routed-vs-original differences with tuned-lens-aware analysis | mixed | `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-confirm-v5.md`, `results/tool_breakage/20260320-gemma2-tool-breakage-route-mode-dynamic-seeded-v1.md` | Seeded dynamic donor controls remain mixed/negative on pooled primary metric |
| Safety lane shows refusal-related routing structure with causal interventions | mixed | `results/safety_alignment/20260320-gemma2it-refusal-surface-v3-validation-v3.md`, `results/safety_alignment/20260320-gemma2it-mediator-conditioned-routing-v3.md` | Mediator-active confirm partition remains refusal-only on v3 |
| OIH external anchor shows positive dynamic signal over calibrated static comparator | mixed | `results/oracle_alpha/20260320-resattn-oih-full-v1.md` | Dynamic-over-calibrated-static margin is positive but modest (`+0.0478` nats) |
| Figure 8 strong alignment to trained routing | blocked | `results/figure8_validation/*` proxy artifacts | Lane remains descriptive/proxy-level; strong trained-routing alignment not supported |
| Router distillation reaches prereg readiness (`R^2 > 0.5`) | blocked | `results/router_training/20260323-gemma2-router-distillation-block-split-supervision-v1.md`, `results/router_training/20260323-gemma2-router-distillation-hybrid-late-third-v1.md` | Current best is `R^2 = 0.4099`; readiness gate unpassed |
| Training-dynamics extension supports checkpoint-level claims | blocked | no claim-bearing training-dynamics artifact yet | Lane requires either a minimal checkpoint artifact or explicit omission in paper scope |

## Writing-Ready Core

The current writing-ready core is:
- primary-model oracle signal on Gemma with prereg null controls,
- route-structure evidence with strongest support in factual recall,
- regime-comparison result showing competitive softmax routing utility.

## Non-Claims To Keep Explicit

- No equivalence claim between frozen-model oracle alpha and trained AttnRes routing.
- No strong Figure 8 trained-routing alignment claim.
- No claim that router-distillation readiness has been reached.
- No strong tool-breakage claim beyond currently bounded mixed evidence.

## Remaining Paper-Readiness Blockers

1. Decide paper scope for training-dynamics lane: include minimal checkpoint artifact or declare out-of-scope with explicit rationale.
2. Freeze final claim language around mixed lanes (`tool_breakage`, `safety_alignment`, `router_training`) before drafting.
3. Produce the final claim-evidence table for the manuscript draft directly from this matrix and `results/RESULTS_INDEX.md`.
