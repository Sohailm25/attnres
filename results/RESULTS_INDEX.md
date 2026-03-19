# Results Index

Register every experimental artifact here. Never delete entries; mark superseded artifacts explicitly.

## Rules

- Every artifact saved under `results/` must appear here.
- Every entry should state the relevant hypothesis or lane.
- Every entry should state `pass`, `fail`, `mixed`, `partial`, or `planning`.
- Every summary must respect the oracle-alpha framing lock.

## Infrastructure

| Artifact | Lane | Status | Path |
|---|---|---|---|
| Phase 1 dependency freeze and first validation slice | infrastructure | partial | `results/infrastructure/20260316-phase1-freeze-and-validation.md` |
| GPT-2 XL model-backed reconstruction smoke | infrastructure | pass | `results/infrastructure/20260316-gpt2xl-reconstruction-smoke.md` |
| Gemma-2 primary-spine model-backed reconstruction smoke | infrastructure | pass | `results/infrastructure/20260317-gemma2-reconstruction-smoke.md` |
| Pilot/confirmatory prompt registry and confirm-only access guard | infrastructure | pass | `results/infrastructure/20260316-pilot-confirm-registry.md` |
| Pilot/confirmatory prompt registry v2 pilot expansion | infrastructure | pass | `results/infrastructure/20260317-pilot-confirm-registry-v2.md` |
| Pilot/confirmatory prompt registry v3 scale-up | infrastructure | pass | `results/infrastructure/20260317-pilot-confirm-registry-v3.md` |
| Pilot/confirmatory prompt registry v4 prereg-scale freeze | infrastructure | pass | `results/infrastructure/20260317-pilot-confirm-registry-v4.md` |
| Oracle-alpha Phase 1 control suite | infrastructure | pass | `results/infrastructure/20260316-oracle-alpha-control-suite.md` |
| Held-out predictiveness review alignment | infrastructure | pass | `results/infrastructure/20260316-heldout-predictiveness-review-alignment.md` |
| Prereg-scale oracle-alpha campaign build-out | infrastructure | pass | `results/infrastructure/20260317-prereg-scale-campaign-buildout.md` |
| Gemma-shaped `n << d` ridge runtime benchmark | infrastructure | pass | `results/infrastructure/20260318-gemma2-ridge-runtime-benchmark-b4q.md` |
| Oracle campaign progress smoke for summary-stage observability | infrastructure | pass | `results/infrastructure/20260318-oracle-campaign-progress-smoke-9co.md` |

## Oracle Alpha

| Artifact | Lane | Status | Path |
|---|---|---|---|
| Gemma-2 `registry_v5` oracle synthesis v1 | oracle_alpha | pass | `results/oracle_alpha/20260318-gemma2-registry-v5-synthesis-v1.md` |
| GPT-2 XL development-model oracle-alpha slice | oracle_alpha | partial | `results/oracle_alpha/20260316-gpt2xl-development-slice.md` |
| GPT-2 XL pilot stability suite | oracle_alpha | partial | `results/oracle_alpha/20260316-gpt2xl-pilot-stability-suite.md` |
| GPT-2 XL held-out predictiveness check | oracle_alpha | fail | `results/oracle_alpha/20260316-gpt2xl-heldout-predictiveness-check.md` |
| GPT-2 XL held-out predictiveness feature comparison | oracle_alpha | mixed | `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-feature-comparison.md` |
| GPT-2 XL held-out predictiveness token-aware comparison | oracle_alpha | mixed | `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-token-aware-comparison.md` |
| GPT-2 XL held-out predictiveness prompt-hybrid comparison | oracle_alpha | fail | `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-prompt-hybrid-comparison.md` |
| GPT-2 XL held-out predictiveness logit-target check | oracle_alpha | mixed | `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-logit-target-check.md` |
| GPT-2 XL held-out predictiveness depth-type-band logit target check | oracle_alpha | mixed | `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-depth-type-band-logit-target-check.md` |
| GPT-2 XL held-out predictiveness loss-aware target comparison | oracle_alpha | fail | `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-loss-aware-target-comparison.md` |
| GPT-2 XL held-out predictiveness pilot expansion check | oracle_alpha | mixed | `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-pilot-expansion-check.md` |
| GPT-2 XL held-out predictiveness registry v3 scale check | oracle_alpha | mixed | `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-registry-v3-scale-check.md` |
| GPT-2 XL prereg-scale oracle-alpha campaign v4 | oracle_alpha | mixed | `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4.md` |
| Gemma-2 primary-model oracle-alpha development slice | oracle_alpha | partial | `results/oracle_alpha/20260317-gemma2-development-slice.md` |
| Gemma-2 primary-model pilot stability suite | oracle_alpha | partial | `results/oracle_alpha/20260317-gemma2-pilot-stability-suite.md` |
| Gemma-2 primary-model held-out predictiveness check | oracle_alpha | mixed | `results/oracle_alpha/20260317-gemma2-heldout-predictiveness-check.md` |
| Gemma-2 registry_v5 calibration v2 | oracle_alpha | positive | `results/oracle_alpha/20260318-gemma2-registry-v5-calibration-v2.md` |
| Gemma-2 registry_v5 stratified campaign v1 | oracle_alpha | pass | `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1.md` |

## Pattern Analysis

| Artifact | Lane | Status | Path |
|---|---|---|---|
| GPT-2 XL prereg-scale sequence-level pattern analysis v1 | pattern_analysis | superseded | `results/pattern_analysis/20260317-gpt2xl-prereg-scale-pattern-analysis-v1.md` |
| GPT-2 XL prereg-scale pattern analysis grouped-view robustness v1 | pattern_analysis | mixed | `results/pattern_analysis/20260317-gpt2xl-prereg-scale-pattern-analysis-ojq-v1.md` |
| Gemma-2 primary-model prereg-scale pattern analysis v1 | pattern_analysis | mixed | `results/pattern_analysis/20260318-gemma2-prereg-scale-pattern-analysis-v1.md` |
| Gemma-2 registry_v5 stratified pattern analysis v1 | pattern_analysis | mixed | `results/pattern_analysis/20260318-gemma2-registry-v5-pattern-analysis-v1.md` |

## Figure 8 Validation

| Artifact | Lane | Status | Path |
|---|---|---|---|
| Local Block AttnRes proxy viability smoke v1 | figure8_validation | mixed | `results/figure8_validation/20260317-attnres-proxy-viability-smoke-v1.md` |
| Local Block AttnRes proxy viability scale v1 | figure8_validation | mixed | `results/figure8_validation/20260317-attnres-proxy-viability-scale-v1.md` |
| Local Block AttnRes proxy compact-subword v1 | figure8_validation | mixed | `results/figure8_validation/20260317-attnres-proxy-compact-subword-v1.md` |
| Local Block AttnRes proxy compact-subword capacity-first v1 | figure8_validation | mixed | `results/figure8_validation/20260317-attnres-proxy-compact-subword-capacity-v1.md` |
| Local Block AttnRes proxy compact-subword horizon-first v1 | figure8_validation | fail | `results/figure8_validation/20260317-attnres-proxy-compact-subword-horizon-v1.md` |
| Local Block AttnRes proxy compact-subword `wikitext-103` calibration | figure8_validation | partial | `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-calibration.md` |
| Local Block AttnRes proxy compact-subword `wikitext-103` v1 | figure8_validation | mixed | `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-v1.md` |
| Local Block AttnRes proxy compact-subword `wikitext-103` best-checkpoint v1 | figure8_validation | mixed | `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-bestcheck-v1.md` |
| Local Block AttnRes proxy regularization-first `wikitext-103` sweep v1 | figure8_validation | fail | `results/figure8_validation/20260317-attnres-proxy-regularization-sweep-bux-v1.md` |

## Block Structure

| Artifact | Lane | Status | Path |
|---|---|---|---|
| Gemma-2 factual-recall raw-source cluster profile v1 | block_structure | mixed | `results/block_structure/20260318-gemma2-factual-recall-cluster-profile-v1.md` |
| Gemma-2 factual route-mode characterization v1 | block_structure | pass | `results/block_structure/20260318-gemma2-factual-route-modes-v1.md` |
| Gemma-2 factual-recall routing synthesis v1 | block_structure | pass | `results/block_structure/20260318-gemma2-factual-recall-routing-synthesis-v1.md` |
| Gemma-2 factual route-mode frame audit v1 | block_structure | pass | `results/block_structure/20260318-gemma2-factual-route-mode-frame-audit-v1.md` |
| Gemma-2 author novel-title residual audit v1 | block_structure | mixed | `results/block_structure/20260318-gemma2-author-novel-title-residual-audit-v1.md` |

## Comparison Regimes

| Artifact | Lane | Status | Path |
|---|---|---|---|
| Gemma-2 primary-model routing-regime comparison v1 | comparison_regimes | pass | `results/comparison_regimes/20260318-gemma2-regime-comparison-v1.md` |

## Tool Breakage

| Artifact | Lane | Status | Path |
|---|---|---|---|
| Gemma-2 tuned-lens viability pilot v1 | tool_breakage | mixed | `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1.md` |
| Gemma-2 routed-versus-original factual-recall baseline pilot v1 | tool_breakage | mixed | `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1.md` |
| Gemma-2 routed-versus-original factual-recall baseline confirm v1 | tool_breakage | mixed | `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1.md` |
| Gemma-2 controlled dynamic-routing counterfactual smoke | tool_breakage | mixed | `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-smoke.md` |
| Gemma-2 controlled dynamic-routing counterfactual confirm v1 | tool_breakage | mixed | `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-confirm-v1.md` |
| Gemma-2 factual-routing tool-breakage bridge v1 | tool_breakage | mixed | `results/tool_breakage/20260318-gemma2-factual-routing-tool-breakage-bridge-v1.md` |
| Gemma-2 matched-family tool-breakage pilot v2 | tool_breakage | partial | `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-pilot-v2.md` |
| Gemma-2 matched-family tool-breakage confirm v2 | tool_breakage | pass | `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-confirm-v2.md` |
| Gemma-2 matched-family dynamic-routing counterfactual v2 | tool_breakage | mixed | `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-counterfactual-v2.md` |
| Gemma-2 matched-family donor-arm decomposition v1 | tool_breakage | mixed | `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-donor-arms-v1.md` |
| Gemma-2 matched-family tool-breakage pilot v3 | tool_breakage | pass | `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-pilot-v3.md` |
| Gemma-2 matched-family tool-breakage confirm v3 | tool_breakage | pass | `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-confirm-v3.md` |
| Gemma-2 matched-family donor-arm decomposition v3 | tool_breakage | mixed | `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-donor-arms-v3.md` |
| Gemma-2 matched-family family-conditioned donor-arm profile v3 | tool_breakage | mixed | `results/tool_breakage/20260318-gemma2-tool-breakage-family-profile-v3.md` |
| Gemma-2 one-token matched-family pilot v4 | tool_breakage | pass | `results/tool_breakage/20260318-gemma2-tool-breakage-one-token-pilot-v4.md` |
| Gemma-2 one-token matched-family confirm v4 | tool_breakage | pass | `results/tool_breakage/20260318-gemma2-tool-breakage-one-token-confirm-v4.md` |
| Gemma-2 one-token matched-family donor-arm decomposition v4 | tool_breakage | mixed | `results/tool_breakage/20260318-gemma2-tool-breakage-one-token-donor-arms-v4.md` |
| Gemma-2 one-token matched-family family-conditioned donor-arm profile v4 | tool_breakage | mixed | `results/tool_breakage/20260318-gemma2-tool-breakage-family-profile-v4.md` |
| Gemma-2 one-token factual route-mode coverage v4 | tool_breakage | mixed | `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-coverage-v4.md` |
| Gemma-2 route-mode-aware one-token factual surface v5 | tool_breakage | planning | `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-surface-v5.md` |
| Gemma-2 route-mode-aware one-token pilot v5 | tool_breakage | pass | `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-pilot-v5.md` |
| Gemma-2 route-mode-aware one-token confirm v5 | tool_breakage | pass | `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-confirm-v5.md` |
| Gemma-2 route-mode-aware donor-arm decomposition v5 | tool_breakage | mixed | `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-donor-arms-v5.md` |
| Gemma-2 factual route-mode bridge v5 | tool_breakage | pass | `results/tool_breakage/20260318-gemma2-tool-breakage-factual-route-mode-bridge-v5.md` |
| Gemma-2 donor geometry audit v5 | tool_breakage | mixed | `results/tool_breakage/20260318-gemma2-tool-breakage-donor-geometry-audit-v5.md` |
| Gemma-2 moon prompt-style audit v4 | tool_breakage | mixed | `results/tool_breakage/20260318-gemma2-tool-breakage-moon-style-audit-v4.md` |

## Training Dynamics

| Artifact | Lane | Status | Path |
|---|---|---|---|

## Router Training

| Artifact | Lane | Status | Path |
|---|---|---|---|
| Gemma-2 router-distillation readiness audit v1 | router_training | pass | `results/router_training/20260318-gemma2-router-distillation-readiness-audit-v1.md` |
| Gemma-2 router-distillation pilot export v1 | router_training | pass | `results/router_training/20260318-gemma2-router-distillation-pilot-export-v1.md` |
| Gemma-2 router-distillation pilot v1 | router_training | fail | `results/router_training/20260318-gemma2-router-distillation-pilot-v1.md` |
| Gemma-2 router-distillation target comparison v1 | router_training | mixed | `results/router_training/20260318-gemma2-router-distillation-target-comparison-v1.md` |
| Gemma-2 router-distillation aggregation comparison v1 | router_training | fail | `results/router_training/20260318-gemma2-router-distillation-aggregation-comparison-v1.md` |
| Gemma-2 router-distillation capacity comparison v1 | router_training | mixed | `results/router_training/20260318-gemma2-router-distillation-capacity-comparison-v1.md` |
| Gemma-2 router-distillation family comparison v1 | router_training | mixed | `results/router_training/20260318-gemma2-router-distillation-family-comparison-v1.md` |
| Gemma-2 router-distillation supervision-granularity audit v1 | router_training | pass | `results/router_training/20260318-gemma2-router-distillation-supervision-granularity-audit-v1.md` |
| Gemma-2 router-distillation supervision-objective comparison v1 | router_training | pass | `results/router_training/20260318-gemma2-router-distillation-supervision-objective-comparison-v1.md` |

## Safety Alignment

| Artifact | Lane | Status | Path |
|---|---|---|---|
| Gemma-2 IT refusal-feature discovery smoke | safety_alignment | partial | `results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-smoke.md` |
| Gemma-2 IT refusal-feature discovery validation v1 | safety_alignment | pass | `results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-v1.md` |
| Gemma-2 IT refusal-direction intervention smoke | safety_alignment | partial | `results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-smoke.md` |
| Gemma-2 IT refusal-direction intervention v1 | safety_alignment | pass | `results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-v1.md` |
| Gemma-2 IT mediator-conditioned routing v1 | safety_alignment | mixed | `results/safety_alignment/20260317-gemma2it-mediator-conditioned-routing-v1.md` |
| Gemma-2 IT broadened refusal-surface validation v2 | safety_alignment | superseded | `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation.md` |
| Gemma-2 IT broadened refusal-surface validation v2 tag-aware v1 | safety_alignment | superseded | `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-tag-aware-v1.md` |
| Gemma-2 IT broadened refusal-surface validation v2 policy-style v2 | safety_alignment | mixed | `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-policy-style-v2.md` |
| Gemma-2 IT broadened refusal-surface policy budget check v1 | safety_alignment | mixed | `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-policy-budget-check-v1.md` |
| Gemma-2 IT broadened refusal-surface prohibition-style v1 | safety_alignment | mixed | `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-prohibition-style-v1.md` |
| Gemma-2 IT broadened mediator-conditioned routing v2 | safety_alignment | fail | `results/safety_alignment/20260318-gemma2it-mediator-conditioned-routing-v2.md` |

## Figures

| Artifact | Lane | Status | Path |
|---|---|---|---|
