ABOUTME: Reviewer-facing progress report summarizing the experiment status, evidence, and claim boundaries as of 2026-03-17.
ABOUTME: Meant to let an external reviewer assess alignment between the repo's execution so far and the project's overall scientific goal.

# External Review Progress Report

## Purpose

This memo is a repo-grounded status report for an external reviewer. It is not a
new experimental artifact. Its job is to summarize:

- the overall experimental goal
- the scientific framing and non-negotiable claim boundaries
- what has actually been implemented and observed so far
- which parts of the thesis are currently supported, mixed, or still blocked
- where the repo is aligned well versus where it is still at risk

Everything below is derived from the current repo state, especially:

- `CURRENT_STATE.md`
- `history/PREREG.md`
- `configs/experiment.yaml`
- `results/RESULTS_INDEX.md`
- the cited artifact memos under `results/`

## Overall Experimental Goal

The central question is:

Can standard frozen transformers expose a latent, input-dependent depth-routing
structure that becomes visible under oracle-alpha analysis, and does that
structure support the interpretability thesis motivated by Attention
Residuals?

The repo is explicitly **not** trying to prove that a standard frozen model
contains a literal trained AttnRes router. The locked framing is narrower:

- `known`: oracle-alpha on frozen models is a post-hoc analysis over fixed
  representations
- `known`: oracle-alpha is treated as an upper bound on routing signal
  recoverable from fixed standard-model representations
- `known`: oracle-alpha is treated as a lower bound on the benefit of depth
  routing once routing and computation can co-adapt
- `known`: the strongest default frozen-model claim is recovery of an
  `effective depth mixture`

That framing is consistent across `history/PREREG.md`,
`configs/experiment.yaml`, and `CURRENT_STATE.md`.

## Reviewer Bottom Line

The repo has made real progress and is mostly aligned with the overall goal.
The strongest current positive result is no longer "the infrastructure exists."
It is this:

- `observed`: the prereg-scale development-model oracle-alpha lane on `gpt2-xl`
  clears the Phase 1 loss gate strongly on a locked `96 / 128` pilot/confirm
  split, with `+1.2993` nats mean confirm improvement over uniform,
  `128 / 128` confirm prompts positive, `p = 9.72e-98`, and strong wins over
  all preregistered nulls

The strongest current limitation is also no longer missing infrastructure. It is
interpretation:

- `observed`: held-out predictiveness stays positive on routed loss at prereg
  scale, but alpha-shape recovery remains weak (`R^2 = -0.0884`, mean
  JS `= 0.2427`)
- `observed`: prereg-scale pattern analysis shows robust coarse routing
  structure, especially in grouped source-type views, but does not pass the raw
  block-structure gate and does not support a clean `~8`-cluster claim
- `observed`: the local Figure 8 proxy is operationally real, but still does
  not recover a persuasive trained-routing pattern surface
- `observed`: the same-model Gemma tool-breakage lane shows real routed-versus-
  original degradation, but the dynamic counterfactual weakens the strongest
  claim that prompt-matched input-dependent routing is uniquely responsible

So the repo is aligned with the goal in the important sense that it has built
the right lanes, passed the key development-model feasibility gate, and refused
to overclaim from mixed results. It is not yet aligned in the stronger sense of
having fully closed the secondary interpretability story.

## Scientific Locks That Are Being Respected

The strongest sign of alignment is not a single positive result. It is that the
repo has repeatedly preserved the scientific locks when results turned mixed.

- `known`: claim-bearing statistics default to the sequence-level unit
- `known`: uniform-routing reconstruction targets the original logits after the
  model's own final normalization, not `logits / L`
- `known`: exact per-source logit decomposition uses the shared final
  normalization factor from the full routed mixture
- `known`: no Ward linkage is used directly on Jensen-Shannon distances
- `known`: Figure 8 and layer-type-specialization claims require sublayer
  outputs, not `resid_post`-only shortcuts
- `known`: tool-breakage claims must compare original versus routed behavior and
  include a tuned-lens-aware baseline
- `known`: strong tool-breakage language requires a controlled dynamic-routing
  counterfactual
- `known`: safety claims require layer localization first, then feature
  discovery and validation, then mediator-conditioned analysis, while separating
  harmfulness from refusal
- `known`: strong trained-routing alignment language requires a reproducible
  proxy rather than visual similarity to published AttnRes figures

The repo has repeatedly updated its execution path to obey these locks rather
than silently relaxing them.

## Phase and Lane Status

### Phase 1: Oracle-alpha infrastructure and feasibility

This is the strongest lane so far.

- `observed`: the Phase 1 dependency freeze and full lock landed cleanly
- `observed`: reconstruction math was validated both by pure tests and by a
  model-backed `gpt2-xl` smoke
- `observed`: pilot/confirm prompt registries are frozen and code-enforced
- `observed`: the prereg control suite is codified and importable
- `observed`: the oracle-alpha runner, stability suite, held-out predictiveness
  checks, and checkpointed campaign runner all exist and have been exercised on
  real artifacts

The main empirical result is:

- `observed`: on `gpt2-xl`, the prereg-scale `registry_v4` campaign reached
  `+1.2993` nats mean confirm improvement over uniform on `128` confirm prompts
  with `128 / 128` prompts positive
- `observed`: oracle alpha also beat `random_dirichlet`,
  `magnitude_proportional`, and `last_layer_only` on every confirm prompt

This means the repo now has strong development-model evidence for the primary
hypothesis that non-uniform softmax-constrained routing can outperform uniform
routing in frozen standard transformers.

What remains limited:

- `observed`: held-out predictiveness is positive on the confirm routed-loss
  metric (`+0.1162` nats, `95 / 128` prompts positive) but still weak as an
  alpha-recovery story (`R^2 = -0.0884`, mean JS `= 0.2427`)
- `inferred`: the lane supports "effective depth mixture with real
  out-of-sample loss signal" more strongly than it supports "the recovered alpha
  object is itself well-predicted in a way that justifies stronger router-like
  language"

### Phase 2: Pattern analysis and block structure

This lane is scientifically useful but still mixed.

Raw-source result:

- `observed`: best raw-source silhouette on the prereg-scale confirm artifact is
  `0.1428` versus matched random control `0.1093`
- `observed`: the raw structure is stable under resampling but dominated by
  `k = 2` and extremely imbalanced clusters
- `observed`: the prereg raw block-structure gate remains unpassed

Grouped-view result:

- `observed`: grouped `source_type` routing shows stronger coarse structure
  (`0.6652` oracle silhouette versus `0.5569` random)
- `observed`: grouped `depth_thirds_by_type` also beats matched random
  (`0.3451` versus `0.2305`)
- `observed`: those views still mostly collapse to `k = 2`, not a clean
  `~8`-cluster decomposition

Current honest read:

- `inferred`: the repo has evidence for coarse routing-regime variation,
  especially attention-versus-MLP balance differences across prompts
- `known`: it does not yet have evidence for the prereg raw block-structure
  hypothesis around approximately `8` clusters

### Phase 2: Figure 8 validation

This is the most stubborn mixed lane.

Progress made:

- `observed`: the repo resolved the Figure 8 ambiguity against
  `research/Attention_Residuals.pdf`
- `observed`: a local small AttnRes proxy exists and is checkpointed,
  reproducible, and operationally real
- `observed`: the proxy beat the matched baseline in early smoke settings
- `observed`: compact-subword tokenization, width changes, horizon changes,
  larger corpus, checkpoint-trajectory export, and matched regularization have
  all now been tested

Best current proxy read on the widened compact-subword `wikitext-103` regime:

- `observed`: best-checkpoint baseline eval loss `= 6.5929`
- `observed`: best-checkpoint AttnRes eval loss `= 6.6315`
- `observed`: loss delta `= +0.0386` in favor of the matched baseline
- `observed`: deep embedding persistence improved to `0.1689`
- `observed`: mean pre-attn entropy remained below mean pre-MLP entropy, with
  entropy gap `= -0.0549`

Current honest read:

- `observed`: the proxy exists and is credible as a bounded local reproduction
- `observed`: some paper-facing metrics move in the right direction under more
  faithful regimes
- `known`: the current proxy does not independently recover a paper-like Figure
  8 surface
- `known`: the repo has explicitly frozen custom objective rescue on this tiny
  proxy

This is a methodological strength, not a weakness. The repo has stopped trying
to force a positive Figure 8 result out of a fidelity-limited proxy and moved
the next step to a strategic decision issue (`resattn-1lk`) about whether to
revisit the lane through a more faithful proxy or keep the strong lane frozen.

### Phase 3: Tool-breakage on Gemma-2

This lane is more mature than the Figure 8 lane and more mixed than the oracle
lane.

What is solid:

- `observed`: the repo kept the tool-breakage lane on primary
  `google/gemma-2-2b` and trained a custom Gemma-2 tuned lens locally
- `observed`: the tuned lens is a real held-out improvement over raw logit lens
  on the original model in distributional metrics
- `observed`: on the locked confirm split, routing worsens the same-model tuned-
  lens baseline:
  - mean tuned KL `3.3834 -> 5.9061`
  - final-position tuned KL `5.9851 -> 8.8947`
  - mean tuned top-1 `0.5095 -> 0.3024`
  - final-position tuned top-1 `0.1442 -> 0.1154`
- `observed`: the old absolute non-monotonicity boolean is uninformative here,
  but the relative rank metrics are informative:
  - tuned target-rank-range increase on `7 / 8` confirm prompts
  - tuned final-target-rank worsening on `5 / 8`

What blocks the strongest claim:

- `observed`: the controlled dynamic-routing counterfactual clears the weaker
  static-routing objection, because routed traces are worse than a fixed
  `pilot_mean_alpha` control
- `observed`: the same counterfactual weakens the stronger claim, because a
  prompt-permuted dynamic control is at least as damaging as the prompt-matched
  routed trace on the tuned primary KL surface

Current honest read:

- `inferred`: routing clearly perturbs the same-model Gemma lens surface and
  often worsens answer-token rank stability
- `known`: the strongest same-model claim that prompt-matched input-dependent
  routing is uniquely responsible for the damage is currently blocked
- `known`: the repo has frozen that claim boundary instead of moving the
  goalposts

### Phase 3: Safety alignment on aligned Gemma

This lane is methodologically strong and intentionally narrow.

What is solid:

- `observed`: the repo correctly moved the safety lane to
  `google/gemma-2-2b-it` after confirming that the base model was invalid for
  refusal discovery
- `observed`: refusal and harmfulness are localized separately:
  - refusal at assistant-prefill layer `22`
  - harmfulness at instruction-final layer `25`
- `observed`: held-out direction validation is strong:
  - confirm pair accuracy `= 1.0` for both primary refusal and harmfulness
    directions
  - direction cosine `= 0.0064`
- `observed`: the causal mediator pass shows refusal-direction interventions move
  safe continuation preference on the frozen confirm split
- `observed`: the mediator-conditioned trajectory artifact shows large
  full-depth refusal-trajectory shifts under refusal-direction interventions

What remains limited:

- `observed`: the current mediator-active partition still collapses exactly to
  refusal versus non-refusal prompt roles on the frozen prompt set
- `known`: this supports a bounded mediator-conditioned depth-trajectory claim,
  not a broader safety-routing law

Current honest read:

- `inferred`: the safety lane has successfully validated its mechanistic anchor
  and causal intervention surface
- `known`: stronger safety-routing language remains blocked on a broader prompt
  surface that breaks the current role collapse

### Not yet executed enough to review strongly

These lanes remain required by the thesis but are not yet mature enough to
support a substantive scientific read:

- comparison regimes (`softmax` vs `unconstrained` vs `top-k`)
- block structure as its own dedicated lane beyond the current pattern-analysis
  artifacts
- training dynamics on `pythia-2.8b`
- router training plus `w_l` analog geometry

This is a real gap relative to the full end-state thesis. It is not a hidden
gap; the repo keeps these lanes visible and unclaimed.

## Alignment To The Overall Goal

### Where alignment is strong

1. The repo is testing the right core question.
   The strongest executed lane still maps directly onto the primary hypothesis:
   whether standard transformers contain exploitable non-uniform depth-routing
   structure relative to uniform routing.

2. The repo has built the controls needed to keep a positive result honest.
   Reconstruction sanity checks, null models, pilot/confirm split, restart and
   perturbation stability, out-of-sample predictiveness, tuned-lens-aware
   comparisons, and dynamic controls are all present.

3. The repo is not faking success on the AttnRes-specific thesis.
   The Figure 8 lane and the strong same-model Gemma tool-breakage claim both
   stayed mixed, and the repo tightened claim boundaries instead of rephrasing
   them as wins.

4. The safety lane is aligned methodologically.
   It did not jump straight to routing claims. It localized layers, separated
   harmfulness from refusal, validated directions out of sample, and then ran a
   bounded mediator intervention.

### Where alignment is only partial

1. The strongest positive oracle-alpha result is still on the development model,
   not the primary Gemma-2 spine.

2. The route from "effective depth mixture exists" to "this gives a strong new
   interpretability surface analogous to AttnRes-trained routing" is still not
   fully closed.

3. The required comparison-regime and router-training lanes remain largely
   future work, so the thesis is not yet closed end to end.

4. The Figure 8 lane has not yielded a convincing trained-routing proxy result.
   That is now a strategic decision, not a mere implementation omission.

## Claims Currently Supported

These claims look supported by the current repo evidence.

- `supported`: on a development model (`gpt2-xl`), frozen-model oracle-alpha
  strongly beats uniform routing and the preregistered nulls at prereg-scale
  confirm counts
- `supported`: the resulting routing structure is not random noise; it contains
  stable coarse source-type routing variation across prompts
- `supported`: same-model routing on Gemma-2 changes tuned-lens behavior
  relative to the original-model baseline and often worsens target-rank
  stability on factual recall prompts
- `supported`: aligned Gemma contains separable refusal and harmfulness
  directions on the current held-out prompt surface, and the refusal direction
  has causal leverage on safe continuation preference

## Claims Explicitly Not Supported Yet

These stronger claims remain blocked.

- `blocked`: frozen-model oracle-alpha recovers a well-generalized alpha object
  strongly enough to justify literal router language
- `blocked`: prereg raw block structure around roughly `8` clusters
- `blocked`: strong Figure 8 / trained-routing alignment on the current local
  proxy
- `blocked`: the strongest same-model Gemma tool-breakage claim that prompt-
  matched input-dependent routing is uniquely responsible for the observed lens
  degradation
- `blocked`: any broad safety-routing claim beyond the current bounded aligned-
  Gemma mediator evidence
- `blocked`: comparison-regime conclusions about softmax versus unconstrained
  versus top-k routing
- `blocked`: router-training and `w_l`-analog geometry claims

## Main Risks For The Reviewer To Inspect

These are the places where the project is strongest if they hold, and weakest if
they do not.

1. Whether the reviewer buys the current interpretation of the prereg-scale
   oracle lane:
   positive held-out routed-loss recovery despite weak alpha-shape recovery.

2. Whether the grouped-view pattern result is best understood as meaningful
   coarse routing structure or as a dimensionality-compression artifact that is
   scientifically narrower than it looks.

3. Whether the current Gemma tool-breakage result is already enough to count as
   a meaningful "tool-breakage demonstration" even with the strong same-model
   causal claim still blocked.

4. Whether freezing the current Figure 8 proxy lane rather than adding custom
   rescue objectives is the right methodological choice.

5. Whether the safety lane's current prompt surface is too role-collapsed to
   be worth extending immediately, or whether the validated mechanistic anchor
   already makes it a high-value next lane.

## Recommended Reviewer Read Order

If the reviewer wants the shortest path through the repo:

1. `CURRENT_STATE.md`
2. `history/PREREG.md`
3. `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4.md`
4. `results/pattern_analysis/20260317-gpt2xl-prereg-scale-pattern-analysis-ojq-v1.md`
5. `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1.md`
6. `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-confirm-v1.md`
7. `results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-v1.md`
8. `results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-v1.md`
9. `results/safety_alignment/20260317-gemma2it-mediator-conditioned-routing-v1.md`
10. `history/20260317-qm4-gemma2-tuned-lens-and-figure8-proxy-decision.md`
11. `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-bestcheck-v1.md`
12. `history/20260317-9fo-no-objective-redesign-for-local-proxy.md`

## Current Best Next Questions

The current ready issues are:

- `resattn-1lk`: decide whether to revisit Figure 8 via a more faithful proxy or
  keep the strong-claim lane frozen
- `resattn-mo5`: broaden the aligned-Gemma safety prompt surface beyond the
  current role-collapsed mediator split
- `resattn-9co`: add progress artifacts for campaign summary stage

From the repo's current scientific position, the most reviewer-relevant next
decision is `resattn-1lk`. That decision determines whether the project still
has a credible path to strong Figure 8 / trained-routing alignment, or whether
the strongest final paper should instead center on:

- robust frozen-model oracle-alpha feasibility
- coarse routing-structure analysis
- mixed but real same-model tool-breakage evidence
- bounded aligned-Gemma safety mediator evidence

## Final Assessment

The repo is in a stronger position than a superficial pass would suggest.

- `observed`: it has already cleared a real prereg-scale oracle feasibility gate
- `observed`: it has real mixed results, not just missing work, in the hard
  lanes that matter most for the paper story
- `observed`: when those mixed results appeared, the repo generally responded by
  tightening claim boundaries rather than by quietly changing the question

That means the project is currently strongest as a careful, methodologically
disciplined investigation of latent depth-routing structure in standard
transformers, with one strong positive development-model lane and several
important but still mixed interpretability extensions.

The key open question for an external reviewer is not whether there has been
progress. There clearly has. The key question is whether the remaining mixed
lanes still point toward a coherent end-state contribution, and especially
whether the Figure 8 strong-claim lane should be pursued further or frozen.
