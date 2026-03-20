ABOUTME: Provides the first prose manuscript draft pass from the locked skeleton and claim matrix.
ABOUTME: Keeps claims bounded to calibrated OIH, mixed tool-breakage/safety lanes, and explicit non-claims.

# Manuscript Draft Pass v1

## Working Title

Latent Depth-Routing Spectroscopy in Standard Transformers: Primary-Model
Oracle Evidence, Structured Regimes, and Boundaries

## Abstract (Draft)

We study whether a frozen standard transformer (`google/gemma-2-2b`) exhibits
an input-dependent effective depth mixture under oracle-alpha analysis. On a
large stratified confirm surface (`1024` prompts), optimized routing is strongly
better than uniform and preregistered null baselines (`+1.6299` nats vs uniform;
`1024/1024` positive), and held-out predicted routing remains positive
(`+0.8292` nats; `958/1024` positive). Competitive softmax routing is stronger
than matched unconstrained and top-k alternatives on the confirm split
(`128/128` prompt-level wins vs each tested alternative). Pattern analyses show
non-random, task-conditioned routing structure, with the strongest raw-source
organization in factual recall and broad grouped structure elsewhere. Extension
lanes are explicitly bounded: tool-breakage remains mixed under dynamic donor
controls, safety mediator activity remains refusal-collapsed on the broadened
quartet surface, and router-distillation and strong Figure 8 trained-routing
alignment claims remain blocked. We frame oracle-alpha as a frozen-model
effective-depth-mixture analysis, not evidence of trained router equivalence.

## 1. Introduction (Draft)

The central question is whether standard transformers expose latent,
input-dependent depth-routing structure that can be recovered without changing
model weights. The analysis target is not a literal trained router variable, but
an effective depth mixture over cached residual sources in a fixed model.

This framing matters for interpretation quality. A positive oracle-alpha signal
can establish that useful input-dependent routing structure is recoverable from
frozen representations. It does not establish what a co-adapted trained routing
system would necessarily learn. The manuscript therefore separates supported
frozen-model findings from stronger trained-routing claims.

The primary contribution is a broad primary-model evidence surface on Gemma-2
with preregistered controls, confirm-split evaluation, and explicit mixed-lane
boundaries.

## 2. Methods and Controls (Draft)

We optimize per-sequence softmax routing weights over fixed residual-source
decompositions and evaluate sequence-level loss improvements against uniform and
preregistered nulls. Claim-bearing tests use sequence-level units by default.
The implementation enforces reconstruction and normalization constraints:
uniform-routing comparisons target original logits after final normalization,
and exact per-source decomposition uses the shared final normalization factor.

Prompt and threshold choices are governed by saved pilot/confirm splits and
control registries. The main primary-model oracle campaign uses a stratified
`registry_v5` surface (pilot `256`, confirm `1024`) with four strata:
factual recall, reasoning/math, code/procedural, and general text.

Regime comparisons are run with matched initialization for:
softmax-constrained, unconstrained sigmoid gating, and top-k routing.

Extension-lane controls are explicit:
- Tool-breakage requires routed-vs-original baselines and dynamic-routing
  counterfactuals, not raw-lens behavior alone.
- Safety requires localization and role-separated validation before
  mediator-conditioned interpretation.
- OIH is treated as a MIB-compatible anchor extension with calibrated static
  comparators, not as replacement of the main oracle spine.

## 3. Primary Results (Draft)

### 3.1 Oracle Signal and Null Baselines

On the stratified Gemma confirm surface (`1024` prompts), oracle routing
improves mean loss over uniform by `+1.6299` nats
(`95% CI [1.5865, 1.6758]`, `1024/1024` positive). The same oracle run beats
all preregistered nulls on mean loss (`random_dirichlet`, `magnitude_proportional`,
`last_layer_only`).

Held-out predicted routing is also positive:
`+0.8292` nats over uniform (`95% CI [0.7805, 0.8758]`, `958/1024` positive),
with confirm `R^2 = 0.2392` and mean JS `= 0.0985` to oracle alpha.

### 3.2 Regime Comparison

On locked confirm prompts (`128`), softmax-constrained routing outperforms
matched unconstrained and all tested top-k variants on every prompt in the
comparison table:
- softmax vs unconstrained mean advantage: `+0.4888` nats (`128/128` wins)
- softmax vs top-k (`k ∈ {2,4,8,13,26}`): all `128/128` wins per `k`

This supports a bounded competition claim: on this tested surface, zero-sum
depth competition is empirically stronger than tested independent gating and
sparse top-k families.

### 3.3 Structure and Stratum Conditioning

Pattern analyses on saved artifacts show non-random routing structure and clear
task conditioning. The strongest raw-source organization appears in factual
recall, while grouped structure generalizes more broadly. On the stratified
surface, factual recall also has the strongest predicted signal among strata
(`+1.6297` predicted mean improvement; mean JS `0.0580`), with weaker but still
positive predicted improvements in other strata.

The manuscript uses this as structured-but-heterogeneous evidence, not as a
global clean `~8`-cluster claim.

## 4. Extension Lanes (Bounded Evidence)

### 4.1 Tool-Breakage (Mixed)

The same-model tool-breakage baseline remains positive for routed-vs-original,
and routed-vs-fixed-alpha controls remain positive. However, dynamic donor-arm
controls remain mixed after seeded-derangement pairing removed an ordering
artifact:
- routed minus `prompt_permuted_alpha`: `-0.0343`
- routed minus `within_family_permuted_alpha`: `-0.2084`
- routed minus `cross_family_permuted_alpha`: `-0.1344`
- routed minus `pilot_mean_alpha`: `+1.0898`

Interpretation stays bounded: evidence of disruption exists, but strongest
prompt-specific same-model dynamic-control claims are not supported broadly.

### 4.2 Safety (Mixed / Bounded Negative for Mediator Expansion)

Validation on `safety_refusal_surface_v3` is clean
(`refusal/non-refusal` behavior gates and pair accuracies all `= 1.0`).
Localization and separation are stable (`refusal layer 22`, harmfulness layer
`25`, direction cosine `-0.0162`).

Mediator-conditioned routing remains role-collapsed on confirm:
active prompts are refusal-only (`3` active refusal; all non-refusal roles
inactive). This blocks stronger mediator-expansion claims on the current v3
surface.

### 4.3 OIH Anchor (Positive but Calibration-Sensitive)

On `resattn_oih_mcqa_v1` (`110/50`), dynamic held-out signal is positive:
- predicted over uniform: `+0.2290`
- oracle over uniform: `+0.5709`
- `R^2 = 0.0906`, mean JS `= 0.0758`

Static comparison requires calibrated reading:
- pruned static baseline: `-3.5916` (`0/50` positive)
- pilot-mean static baseline: `+0.1813` (`49/50` positive)
- calibrated static policy selects pilot-mean static
- dynamic minus calibrated static: `+0.0478`

This supports a bounded dynamic-over-static statement while avoiding overreliance
on the weak pruned-static comparator.

## 5. Limitations and Explicit Non-Claims (Draft)

- This is a frozen-model oracle analysis on fixed representations.
  It is not evidence that trained AttnRes routing would match these recovered
  alphas after co-adaptation.
- Strong trained-routing Figure 8 alignment is not supported by current local
  proxy artifacts.
- Router-distillation readiness remains below prereg gate
  (best bounded confirm result below `R^2 > 0.5`).
- Tool-breakage and safety lanes remain mixed and must be described as such.
- Training-dynamics is out-of-scope for claim-bearing language in this draft
  cycle.

## 6. Discussion and Next Steps (Draft)

The current evidence supports a coherent primary claim set on the main Gemma
spine: a strong recoverable oracle routing signal, meaningful regime separation
favoring softmax competition, and structured task-conditioned routing behavior.

The highest-value near-term writing path is to finalize these supported claims
with explicit mixed-lane caveats rather than reopening broad exploratory runs.
Any extension-lane escalation should require a specific methodological trigger
(for example, a concrete flaw in the calibrated OIH comparator, or a
mechanistically justified mediator redefinition in safety).

## Source Artifacts Used

- `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1.md`
- `results/comparison_regimes/20260318-gemma2-regime-comparison-v1.md`
- `results/pattern_analysis/20260318-gemma2-registry-v5-pattern-analysis-v1.md`
- `results/block_structure/20260318-gemma2-factual-recall-routing-synthesis-v1.md`
- `results/tool_breakage/20260320-gemma2-tool-breakage-route-mode-dynamic-seeded-v1.md`
- `results/safety_alignment/20260320-gemma2it-refusal-surface-v3-validation-v3.md`
- `results/safety_alignment/20260320-gemma2it-mediator-conditioned-routing-v3.md`
- `results/oracle_alpha/20260320-resattn-oih-full-v1.md`
- `history/PREREG.md`
