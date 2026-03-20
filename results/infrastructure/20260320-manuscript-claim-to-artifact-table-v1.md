ABOUTME: Provides a manuscript-ready appendix table mapping each intended claim to concrete artifacts and caveats.
ABOUTME: Restricts the table to supported and mixed claims that are currently in-scope for drafting.

# Manuscript Claim-to-Artifact Table v1

## Usage

Use this table directly in methods/appendix drafting. Keep claim wording bounded to the caveats column.

## Claim Mapping

| Claim ID | Draft claim (bounded wording) | Status | Primary artifact(s) | Key quantitative anchor | Required caveat |
|---|---|---|---|---|---|
| C1 | On primary Gemma, optimized oracle routing improves loss versus uniform and prereg null baselines. | supported | `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1.md`; `results/oracle_alpha/20260318-gemma2-registry-v5-synthesis-v1.md` | Confirm oracle mean improvement over uniform remains strongly positive on saved confirm surface. | Frozen-model oracle alpha is an effective-depth-mixture analysis, not trained-router equivalence. |
| C2 | Routing structure is non-random and task-conditioned, with strongest support in factual recall. | supported | `results/pattern_analysis/20260318-gemma2-registry-v5-pattern-analysis-v1.md`; `results/block_structure/20260318-gemma2-factual-recall-routing-synthesis-v1.md` | Factual-recall structure exceeds random baseline checks and remains stable in saved analyses. | Do not generalize to a global universal cluster taxonomy. |
| C3 | Competitive softmax routing is empirically stronger than tested unconstrained/top-k alternatives on this prompt surface. | supported | `results/comparison_regimes/20260318-gemma2-regime-comparison-v1.md` | Softmax-constrained regime outperforms tested alternatives across confirm prompts. | Claim is limited to the tested model/prompt/regime setup. |
| C4 | Tool-breakage evidence is present but bounded after dynamic donor-counterfactual controls. | mixed | `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-confirm-v5.md`; `results/tool_breakage/20260320-gemma2-tool-breakage-route-mode-dynamic-seeded-v1.md` | Routed-vs-fixed-alpha remains positive while seeded dynamic donor-arm deltas remain mixed/negative. | Present as mixed evidence; avoid strong broad same-model breakage claims. |
| C5 | Safety-lane refusal analyses show measurable structure, but mediator expansion remains refusal-collapsed on v3. | mixed | `results/safety_alignment/20260320-gemma2it-refusal-surface-v3-validation-v3.md`; `results/safety_alignment/20260320-gemma2it-mediator-conditioned-routing-v3.md` | Validation gates are clean, but confirm mediator-active prompts remain refusal-only. | Avoid broad mediator-control claims beyond current v3 prompt families. |
| C6 | On the OIH MCQA anchor, dynamic routing is positive and remains above a calibrated static comparator. | mixed | `results/oracle_alpha/20260320-resattn-oih-full-v1.md` | Predicted dynamic over uniform is positive and exceeds calibrated static by `+0.0478` nats. | Treat as bounded extension evidence; dynamic-vs-static margin is modest and calibration-sensitive. |

## Explicit Non-Claims

- No claim that frozen-model oracle alpha recovers trained AttnRes routing.
- No strong trained-routing Figure 8 alignment claim.
- No claim that router-distillation readiness (`R^2 > 0.5`) has been reached.
- No training-dynamics claim in the current manuscript cycle.
