# Decisions Log

## [2026-03-16T00:00:00-0500] DECISION: Initialize `resattn/` as its own git repository

- Trigger: the workspace was not a standalone repo and inherited `/Users/sohailmo` as the git root.
- Decision: initialize `resattn/` independently and create a task branch for the scaffold work.
- Rationale: branch, history, and hooks need to apply to this experiment rather than the home-directory repo.
- Impact: all subsequent tracking and commits are local to this experiment.

## [2026-03-16T00:05:00-0500] DECISION: Reuse the `braindstorms` operating scaffold but strip experiment-specific assumptions

- Trigger: the source experiment had the right rigor but the wrong subject matter and runtime model.
- Decision: copy the structural discipline of the source experiment while removing persona-circuits and Modal-specific assumptions.
- Rationale: the value is in the operating system of the research, not in inherited execution details.
- Impact: `AGENTS.md`, state docs, prereg, results index, and directories are adapted to the depth-routing experiment.

## [2026-03-16T00:10:00-0500] DECISION: Lock the oracle-alpha framing to the local thesis correction

- Trigger: the default execution plan risked overstating what frozen-model routing optimization proves.
- Decision: treat oracle-alpha as an upper bound on the routing signal available in standard architectures under fixed representations.
- Rationale: frozen-model optimization misses the co-adaptation feedback loop emphasized by the thesis.
- Impact: all core docs and prereg entries must preserve this distinction.

## [2026-03-16T00:15:00-0500] DECISION: Promote the missing thesis lanes to mandatory status

- Trigger: the local gap review identified missing lanes relative to the core thesis.
- Decision: make the following mandatory: Figure 8 validation, block-structure testing, softmax vs unconstrained vs top-k comparison, tool-breakage demonstration, `w_l` analog geometry, and safety routing analysis.
- Rationale: without these lanes, the project drifts away from the claim that AttnRes creates a new interpretability surface.
- Impact: directory layout, prereg, current state tracking, and results indexing all expose these lanes explicitly.

## [2026-03-16T00:30:00-0500] DECISION: Tighten the scaffold after the second-pass review against the full research set

- Trigger: the second review found that the scaffold was directionally correct but too coarse in its operational detail.
- Decision: add the missing research-positioning docs, correct the upper-bound/lower-bound framing split, and operationalize the main prereg metrics and gates.
- Rationale: implementation errors tend to come from vague research scaffolding, not just vague code.
- Impact: the repo now encodes Figure 8 metrics, comparison-regime definitions, tool-breakage thresholds, router training targets, and the public prereg requirement directly in the control docs.

## [2026-03-16T13:45:00-0500] DECISION: Promote the methodology audit and source-style operating rigor to first-class control docs

- Trigger: the third review found that the current `AGENTS.md` was materially leaner than the source experiment and that several hidden implementation hazards were still only implicit in the research docs.
- Decision: expand `AGENTS.md` with adapted operating discipline from `braindstorms` and encode the newly surfaced hazards in a dedicated methodology audit plus the prereg, config, and guidance docs.
- Rationale: the next failure mode is no longer a missing thesis lane; it is a technically plausible but invalid implementation.
- Impact: future implementation must respect the JSD-versus-Ward correction, the final-normalization decomposition rule, the sequence-level statistical unit, and the sublayer-output requirement.

## [2026-03-16T14:05:00-0500] DECISION: Adopt the high-signal research-taste rules from the external sprint prompt, but only in adapted form

- Trigger: Sohail provided an external "vibe research" prompt and asked whether any of it was worth incorporating into `AGENTS.md`.
- Decision: import the smallest-experiment-first heuristic, the explicit target-plot planning step, the tight-feedback-loop preference, the structured experiment write-up, and the requirement to route approved new sprints through a bd issue.
- Rationale: these rules improve research taste and visibility without importing incompatible assumptions like `uv`, monorepo sprint directories, or fresh branch-per-run workflow.
- Impact: future experimental planning should be more explicit and cheaper to iterate on before large runs.

## [2026-03-16T14:15:00-0500] DECISION: Make tmux plus resumable checkpoints mandatory for long-running local work

- Trigger: Sohail called out the risk of losing hours or days of work when moving locations or getting disconnected while running experiments on the MacBook Pro.
- Decision: require `tmux` for long-running jobs and require explicit checkpoint paths, checkpoint cadence, log paths, and a resume command before launch.
- Rationale: local-only research is not operationally serious unless long jobs survive normal laptop interruptions.
- Impact: future agents should default to resumability rather than terminal persistence.

## [2026-03-16T14:35:00-0500] DECISION: Tighten the publishability constraints after a second red-team review

- Trigger: a secondary review found four remaining ways the experiment could produce a technically interesting but publication-weak result.
- Decision: add a pilot/confirmatory split, make the tool-breakage lane tuned-lens-aware, clarify router input as per-token `h_1[t]`, and require refusal-feature discovery before safety-lane claims.
- Rationale: these are the remaining places where a competent implementation could still tell the wrong story.
- Impact: the prereg, state docs, references, and review history now better protect the paper-quality interpretation.

## [2026-03-16T15:30:00-0500] DECISION: Adopt the deep-research deltas that materially harden methodology, not the ones that only add scope

- Trigger: Sohail added `research/deepresearch1.md` and `research/deepresearch2.md` and asked for a thorough review against the current repo setup.
- Decision: adopt five spec-level changes:
  - treat `effective depth mixture` as the strongest default frozen-model claim
  - use MIB as a benchmark anchor or sanity control when the task-model pair fits it
  - require a stability suite plus out-of-sample predictiveness before high-claim interpretation
  - require a controlled dynamic-routing counterfactual for the strong tool-breakage claim
  - strengthen the safety lane to separate harmfulness from refusal after layer localization
- Rationale: these changes materially reduce reviewer-credible failure modes without dragging the repo into unnecessary new infrastructure.
- Impact: `AGENTS.md`, `CURRENT_STATE.md`, `history/PREREG.md`, `configs/experiment.yaml`, `background-work/RESEARCH_POSITIONING.md`, `background-work/MECH_INTERP_GUIDANCE.md`, and the paper cache now encode the updated scientific floor.

## [2026-03-16T15:35:00-0500] DECISION: Resolve the Figure 8 ambiguity directly from the local AttnRes PDF

- Trigger: `deepresearch1.md` flagged uncertainty about which paper and figure the current Figure 8 lane actually referred to.
- Decision: treat Lane 2 as validation against Figure 8 in `research/Attention_Residuals.pdf`, whose text explicitly names diagonal dominance, embedding persistence, layer specialization, learned skip connections, and Block AttnRes with `N = 8` preserving the structure.
- Rationale: leaving the figure ambiguous would make the lane easy to misimplement and easy to overclaim from.
- Impact: the repo can now use a precise Figure 8 reference while still requiring a reproducible proxy before making strong trained-routing alignment claims.

## [2026-03-16T16:00:00-0500] DECISION: Drop the public LessWrong prereg requirement and keep the local prereg as the only mandatory prereg artifact

- Trigger: Sohail explicitly requested removal of `resattn-c12` and the public LessWrong prereg requirement.
- Decision: the repo no longer treats public preregistration as a blocker. `history/PREREG.md` remains the mandatory prereg artifact for claim-bearing work in this workspace.
- Rationale: the scientific guardrail we actually need here is a stable local prereg plus the pilot/confirmatory split and dependency freeze, not publication on a specific external platform.
- Impact: active control docs no longer list LessWrong as a prerequisite, and the corresponding tracker issue should be retired.

## [2026-03-16T20:55:51-0500] DECISION: Freeze the local Phase 1 environment in-place and start with backend-agnostic reconstruction checks

- Trigger: `resattn-syn` was the highest-priority ready blocker, and the current `.venv` only contained bootstrap tooling.
- Decision: pin the direct dependencies in `requirements.txt`, generate a fully resolved `requirements.lock.txt`, install the pinned stack into the existing `.venv`, and implement the first reconstruction/cache-validity checks as pure validation utilities before wiring a model backend.
- Rationale: this is the smallest reproducible Phase 1 slice that makes the environment real, preserves the shared-final-normalization rule, and avoids pretending a model-backed oracle-alpha path exists before the reconstruction math is tested.
- Impact: future Phase 1 work can assume a pinned, installed scientific environment plus a tested normalization-aware validation module, but model-backed reconstruction on the development model is still the next gate.

## [2026-03-16T21:10:00-0500] DECISION: Treat `wip/resattn-scaffold` as the repo trunk branch and merge completed task branches back into it

- Trigger: Sohail asked that completed task work not remain isolated on long-lived `wip/*` branches.
- Decision: treat `wip/resattn-scaffold` as the practical trunk branch for this repo and merge completed task branches back into it before considering the work landed.
- Rationale: leaving scientific and infrastructure work isolated on task branches creates repo drift and weakens continuity across sessions.
- Impact: future closeout should include merging the active task branch back into `wip/resattn-scaffold`, pushing that branch, and only then treating the task as finished.

## [2026-03-16T21:18:40-0500] DECISION: Sum cached residual writes in forward order for model-backed reconstruction on MPS

- Trigger: the first `gpt2-xl` model-backed smoke showed exact per-layer residual identities and near-exact logits, but the final residual reconstruction still differed by about `8.5e-4` when the residual stack was combined with a bulk reduction.
- Decision: reconstruct the final residual by accumulating cached write vectors sequentially in forward order rather than calling `residual_stack.sum(dim=0)`.
- Rationale: the forward-order accumulation matches the model's actual residual update path and avoids spurious reduction-order mismatch on local MPS.
- Impact: the `gpt2-xl` smoke now reconstructs the final residual and logits exactly, and this sequential accumulation rule should remain the default for claim-bearing reconstruction checks.

## [2026-03-16T16:20:00-0500] DECISION: Make the thought log explicitly permissive for ongoing reflection and sidecar research

- Trigger: Sohail wanted `THOUGHT_LOG.md` to preserve the model's feel for the experiment, including internal monologue-like reflections, hunches, guesses, and interesting tangential findings.
- Decision: treat `THOUGHT_LOG.md` as a first-class reflective research log that agents should update whenever useful, and explicitly allow bounded parallel sidecar research when it informs the paper without blocking the main line of work.
- Rationale: the experiment will be more valuable at the end if it preserves not just validated results but also the evolving intuitions, tensions, and side observations that shaped the work.
- Impact: future agents should feel free to record research reflections throughout the project and summarize useful sidecar findings in a durable place instead of losing them in ephemeral outputs.

## [2026-03-16T22:05:00-0500] DECISION: Save the pilot/confirmatory split as a versioned prompt registry with a code-level confirm lock

- Trigger: `resattn-gke` was the highest-priority remaining Phase 1 blocker, and the repo still had policy language about a pilot/confirmatory split without any saved artifact or enforcement path.
- Decision: add `prompts/registry_v1.yaml` as the versioned split artifact, load it through `prompts/registry.py`, point the main experiment config at that file, and require confirmatory access to go through a guard that rejects exploratory mode.
- Rationale: the split only matters scientifically if prompt membership is durable and the confirm set cannot quietly leak into exploratory iterations under time pressure.
- Impact: the repo now has a reusable source of truth for prompt collections plus a scriptable confirm-only access rule, and the next Phase 1 blocker moves back to identifiability, MIB, and predictiveness controls.

## [2026-03-16T22:40:00-0500] DECISION: Encode the Phase 1 identifiability controls as a saved control registry before any oracle-alpha runner exists

- Trigger: `resattn-9wn` required the stability suite, held-out predictiveness check, and MIB anchor to exist before claim-bearing oracle-alpha work, but the repo still had no oracle-alpha execution harness.
- Decision: add `configs/oracle_alpha_controls_v1.yaml` plus `validation/oracle_alpha_controls.py` so the current Phase 1 control suite is durable, validated, and importable even before runner code exists.
- Rationale: the honest current need is methodological discipline, not fake benchmark wiring. Saving the control plan now prevents later runner code from quietly changing bootstrap size, perturbation types, held-out evaluation, or MIB handling.
- Impact: claim-bearing oracle-alpha work now has a source-of-truth control config, reusable bootstrap/stability/predictiveness helpers, and an explicit `planned` MIB anchor with a revisit trigger once a runner exists.

## [2026-03-16T23:08:00-0500] DECISION: Start the first oracle-alpha execution path as a final-output development slice

- Trigger: `resattn-myh` needed to turn the saved prompt and control registries into a real execution path without pretending the full claim-bearing experiment stack already existed.
- Decision: implement the first runner as a development-model slice that optimizes one softmax alpha vector per sequence over the final-output residual-source decomposition, reports against the preregistered nulls, and writes a JSON artifact.
- Rationale: this is the smallest runner that genuinely exercises the fixed-representation oracle-alpha idea, the prompt/control registries, and the final-normalization-correct reconstruction path while avoiding premature expansion into full multi-layer claim-bearing analysis.
- Impact: the repo now has a runnable oracle-alpha path plus a `gpt2-xl` pilot artifact, and the next follow-up shifts from “build any runner at all” to “scale and harden the runner with the preregistered stability perturbations.”

## [2026-03-16T23:35:00-0500] DECISION: Make the pilot stability suite actually test restart variation and saved prompt perturbations

- Trigger: the first `resattn-83v` stability attempt exposed two validity problems: zero-logit initialization made restart seeds deterministic, and one saved paraphrase was malformed enough to confound the prompt-perturbation comparison.
- Decision: initialize alpha logits with tiny seed-dependent noise, summarize stability from the terminal `final_alpha` vectors rather than best-loss snapshots, and keep paraphrase perturbations as explicit saved prompt text in the registry.
- Rationale: restart stability is meaningless if every seed follows the same optimizer path, and prompt-perturbation metrics are not interpretable if the perturbation text is broken or generated ad hoc.
- Impact: the rerun artifact now measures a real exploratory stability surface, and future stability or predictiveness work should treat `final_alpha` plus saved perturbation text as the source of truth.

## [2026-03-17T00:18:00-0500] DECISION: Match the held-out predictiveness run to the saved ridge-regression plan and the current sequence-level oracle target

- Trigger: `resattn-53q` exposed two repo-state ambiguities: the control plan specified `ridge_regression` while the helper only implemented plain least squares, and the saved feature description still allowed either internal-state summaries or prompt-level features.
- Decision: implement a real ridge-regression predictiveness path, tune regularization on the pilot split only using leave-one-out mean JS divergence, and use a sequence-level summary of `h_1[t]` for the current development-model runner by mean-pooling `resid_post` at layer `0` across token positions.
- Rationale: the current oracle-alpha runner produces one alpha vector per sequence, so the predictor should consume a sequence-level early-state summary rather than inventing a different target object. Pilot-only tuning keeps the confirm split clean, and mean JS is a more stable tiny-sample tuning metric than fold-level `R^2` with one held-out sequence.
- Impact: the first held-out artifact is now methodologically aligned with the saved control plan, but it also shows that the present mean-pooled `h_1[t]` feature summary does not generalize well enough for strong interpretation.

## [2026-03-17T00:35:00-0500] DECISION: Record MIB as omitted for the current development-model predictiveness slice while keeping the global anchor planned

- Trigger: `resattn-53q` required an explicit MIB decision now that a development-model predictiveness runner exists.
- Decision: omit MIB for this specific development-model prompt-slice run, while keeping the broader control-suite anchor planned for a later benchmark-compatible lane.
- Rationale: the current pilot/confirm oracle-alpha surface is a custom local prompt registry on `gpt2-xl`, not a benchmark-compatible task-model pair where a MIB-style sanity task would be informative rather than artificial.
- Impact: the runner-stage artifact now records a justified omission instead of silently skipping MIB, and future compatible lanes should revisit the anchor rather than treating it as closed.

## [2026-03-17T01:08:00-0500] DECISION: Keep feature-source selection pilot-only and treat `h_4[t]` mean pooling as the current best internal summary, not a locked predictor

- Trigger: `resattn-k2e` compared a small richer set of internal-state summaries to see whether the failed held-out predictiveness result was just an overly weak `h_1[t]` feature choice.
- Decision: rank candidate feature sources on pilot leave-one-out mean JS only, select `mean_pooled_h_4[t]_resid_post_layer_3` as the best tested internal-state summary, and keep the confirm split for a single held-out rerun.
- Rationale: using confirm to compare feature families would quietly contaminate the gate. The pilot-only ranking improved the predictor modestly, which is useful information, but the confirm result stayed too weak to justify locking the feature source as solved.
- Impact: future predictiveness work should treat mean-pooled `h_4[t]` as the current best internal baseline and move next to token-aware or prompt-level features rather than continuing to iterate tiny pooled-state variants.

## [2026-03-16T22:59:08-0500] DECISION: Treat token-aware `position_thirds` pooling over `h_4[t]` as the current best alpha-shape predictor, but keep the blocker open

- Trigger: `resattn-7ve` tested whether minimal token-aware sequence summaries could recover held-out oracle-alpha structure better than the current mean-pooled `h_4[t]` baseline.
- Decision: keep pilot-only feature selection, adopt `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat` as the best tested token-aware internal summary by pilot leave-one-out mean JS, and keep strong interpretation blocked because confirm-split predicted loss stayed slightly below uniform on average.
- Rationale: the token-aware summary improved confirm `R^2` and mean JS modestly, which is real signal, but it did not translate into a positive routed-loss advantage. That is not enough to treat the predictor as substantively recovered.
- Impact: future predictiveness work should move to prompt-level or hybrid feature surfaces rather than continuing to search within `h_4[t]`-only token-aware pooling variants.

## [2026-03-16T23:13:05-0500] DECISION: Stop iterating simple prompt-level feature families and escalate to a design review

- Trigger: `resattn-27f` compared prompt-shape baselines, mean token-embedding baselines, and hybrids that appended those prompt-level features to the best token-aware `h_4[t]` summary.
- Decision: keep `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat` as the best tested feature source, treat the prompt-level and hybrid comparisons as failed to improve the blocker, and move next to a bounded design review of the held-out predictiveness setup.
- Rationale: none of the new candidates beat the existing token-aware baseline on pilot leave-one-out JS, so the confirm artifact stayed effectively unchanged. That makes more small feature-family additions look low-yield relative to checking whether the target object, tuning metric, sample size, or predictor family is the real problem.
- Impact: the next predictiveness issue should review the sequence-level alpha target, the ridge-regression choice, the `8`-prompt pilot/confirm scale, and the alignment between pilot JS tuning and confirm routed-loss recovery before further runner changes.

## [2026-03-16T23:26:00-0500] DECISION: Land the no-regret design-review fixes before changing predictiveness geometry

- Trigger: the external review of the held-out predictiveness setup identified two implementation mismatches that were clearly worth fixing before any deeper predictor redesign: the runner was not actually governed by the saved predictiveness metrics, and the stability suite only reported aggregate alpha stability.
- Decision: keep the current predictor family for now, but make the runner consult the saved primary/secondary predictiveness metrics and add prompt-matched per-sequence alpha stability metrics for restart/paraphrase/resample runs.
- Rationale: these changes improve methodological honesty without prejudging the larger redesign choice between constrained alpha coordinates and compressed target objects.
- Impact: the next real redesign step can now focus on geometry and target formulation rather than on config drift or overly weak stability reporting.
