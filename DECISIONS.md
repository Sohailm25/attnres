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

## [2026-03-17T16:35:00-0500] DECISION: Land alpha-logit coordinates as the first constrained predictiveness target and treat the result as mixed

- Trigger: `resattn-qq2` was the smallest direct response to the design-review critique that the predictiveness runner was learning simplex-valued alpha targets in the wrong geometry.
- Decision: add `oracle_alpha_logit_vector` as the first constrained target, keep evaluation on recovered simplex alpha distributions, and rerun the held-out check with the current best token-aware feature source fixed.
- Rationale: geometry mismatch was the clearest no-regret design flaw, and changing only the target coordinates isolates that fix from further feature-family churn.
- Impact: the held-out routed-loss metric improved to `+0.0693` nats over uniform on average, but confirm `R^2` and mean JS both regressed and the selected ridge penalty remained `100.0`; the next issue should test compressed or lower-dimensional targets rather than iterating this raw logit-target path further.

## [2026-03-17T17:15:00-0500] DECISION: Use a compressed depth-type-band logit target as the first lower-dimensional predictiveness comparison

- Trigger: `resattn-3ns` needed one bounded compressed target that could be compared directly against the full-source logit target without inventing speculative clusters or changing the feature surface.
- Decision: add `oracle_alpha_depth_type_band_logit_vector`, which compresses oracle alpha into deterministic depth-band-by-source-type group logits and lifts those predictions back to full-source alpha using train-only within-group templates.
- Rationale: this is the smallest lower-dimensional target that is architecture-grounded, reproducible across runs, and consistent with the current constraint that claim-bearing statistics still operate at the sequence level.
- Impact: the compressed target improved confirm `R^2` and mean JS and eliminated the `lambda=100` shrinkage regime, but routed-loss recovery fell back slightly below uniform; the next issue should make selection explicitly loss-aware rather than assume the descriptive alpha metrics are enough.

## [2026-03-17T08:20:00-0500] DECISION: Make held-out predictiveness tuning loss-aware, then treat a negative result as evidence against selection-only fixes

- Trigger: `resattn-xaa` followed the mixed raw-logit-versus-compressed-target result and the external review that said the runner was still asking the right question in the wrong selection geometry.
- Decision: change the saved predictiveness control plan so pilot tuning is governed first by mean predicted routed-loss improvement over uniform and secondarily by mean JS divergence, and extend the runner to compare multiple target parameterizations directly on one fixed feature surface.
- Rationale: routed-loss recovery is the actual claim-bearing objective for this lane, so pilot tuning should stop pretending that descriptive alpha metrics alone are the primary decision rule.
- Impact: the resulting `gpt2-xl` comparison still selected the raw-simplex target and reverted to the same slightly negative confirm result, which means the next issue should stop focusing on target/regularization selection and move to a more fundamental redesign such as a larger pilot surface or a different target object.

## [2026-03-17T08:52:22-0500] DECISION: Test pilot size before another predictor redesign, and treat the positive result as promising but not decisive

- Trigger: `resattn-7mb` had multiple plausible redesign options, and the least invasive one was to increase only the pilot surface while preserving the existing confirm set and loss-aware comparison.
- Decision: create `prompts/registry_v2.yaml`, double the oracle-alpha pilot split from `8` to `16` prompts with saved paraphrases, point the default loader and experiment config at `registry_v2`, and rerun the same loss-aware target comparison without changing the feature source or confirm set.
- Rationale: this isolates the sample-size hypothesis directly. If pilot size is a real bottleneck, the selected target and regularization should become more stable before we invest in token/span-level target redesign.
- Impact: the enlarged pilot surface changed selection from the raw-simplex target to `oracle_alpha_logit_vector`, moved `lambda` from `100.0` to `0.01`, and restored a positive confirm routed-loss delta (`+0.0857` nats). The next issue should scale this path further rather than switch targets again immediately.

## [2026-03-17T09:08:00-0500] DECISION: Scale the logit path again before redesigning it

- Trigger: `resattn-0vx` was the direct follow-up after the positive `registry_v2` result, and the main question was whether the logit-target win would survive a larger saved prompt surface.
- Decision: create `prompts/registry_v3.yaml`, expand the oracle-alpha split to `32` pilot prompts and `16` confirm prompts, point the default loader and config at `registry_v3`, and rerun the same loss-aware comparison without changing the feature surface or target set.
- Rationale: if the logit path is the real current best method, it should survive more data before we spend effort on another redesign. Scaling the saved prompt surface is the cleanest way to test that.
- Impact: the `oracle_alpha_logit_vector` path stayed selected, the held-out routed-loss metric remained positive and strengthened to `+0.1277` nats, and `12 / 16` confirm prompts improved over uniform. The next issue should scale this same path toward the prereg-sized Phase 1 gate.

## [2026-03-17T09:29:48-0500] DECISION: Build the next oracle-alpha scale-up as a resumable campaign instead of another one-shot rerun

- Trigger: after `resattn-0vx`, the next live question was no longer whether to scale but how to do it without repeatedly paying for nearby reruns as the prompt surface grows toward the prereg gate.
- Decision: add a checkpointed campaign layer that persists prompt-level oracle sequence results, feature-vector caches, and the full target-by-regularization tuning grid under one output directory before launching the larger run itself.
- Rationale: the current `32 / 16` result is strong enough to justify scaling, but the old held-out script was still operationally disposable: no prompt-level resume, no reusable feature cache, and no saved full lambda table. Those would force avoidable reruns once the run became expensive.
- Impact: the repo now has a reusable prereg-scale launch path, but the actual larger scientific run remains pending on freezing the next saved prompt surface and launching it in `tmux`.

## [2026-03-17T10:28:19-0500] DECISION: Freeze the prereg-scale oracle-alpha prompt surface as `registry_v4` before launching the big run

- Trigger: `resattn-9jq` had the checkpointed campaign infrastructure but still lacked a large saved prompt surface that justified paying the cost of a tmux-backed prereg-scale run.
- Decision: add `prompts/registry_v4.yaml`, expand the oracle-alpha split to `96` pilot prompts and `128` confirm prompts, and point the default registry loader plus the main experiment config at `registry_v4`.
- Rationale: a prereg-scale launch is only worth doing once the prompt surface is large enough to answer the near-term sample-size question without immediately forcing another registry bump. Freezing the bigger split now turns the next run into the real campaign rather than another intermediate rehearsal.
- Impact: the repo now has both pieces required for the launch path: checkpointed campaign infrastructure and a materially larger saved prompt surface. The next `resattn-9jq` step is operational, not architectural.

## [2026-03-17T11:12:00-0500] DECISION: Treat the `registry_v4` prereg-scale `gpt2-xl` campaign as a development-model oracle gate clear, but not as a full interpretation clear

- Trigger: the first prereg-scale campaign on `prompts/registry_v4.yaml` completed with a full `96 / 128` pilot/confirm split, final summary artifacts, and strong sequence-level oracle improvements against uniform and the preregistered nulls.
- Decision: record the `gpt2-xl` `registry_v4` artifact as the first development-model clear of the preregistered Phase 1 oracle-loss gate, while keeping strong interpretive language blocked because confirm alpha-shape recovery is still weak and the result is not yet on the primary Gemma-2 lane.
- Rationale: the repo now has direct evidence that the oracle optimization itself is real at prereg scale on the development model, but the held-out predictor still selects the raw-simplex target with heavy shrinkage and only weakly recovers oracle-alpha shape (`R^2 < 0`, mean JS `≈ 0.24`). That is enough to advance the lane operationally, but not enough to claim a solved routing-recovery story.
- Impact: the next oracle issue should move to prereg-scale pattern analysis on the saved artifact rather than keep treating basic oracle feasibility as unresolved, while the stronger interpretation gate stays open.

## [2026-03-17T11:14:00-0500] DECISION: Keep checkpoint caches and bulky raw campaign JSONs local, and track compact summaries plus the technical memo

- Trigger: the completed prereg-scale campaign output directory contained `448` prompt-level checkpoint files (`33 MB`) in addition to the top-level summary JSONs and memo.
- Decision: ignore `results/**/checkpoints/`, `results/**/*.log`, and the bulky raw campaign JSONs in git so resumable caches remain available locally under `results/`, while the committed repo surface carries the manifest, compact machine-readable summaries, and the markdown memo.
- Rationale: the checkpoint tree and full raw JSON payloads are operationally valuable for local reuse, but versioning hundreds of per-prompt cache files plus multi-megabyte raw JSONs would clutter the repository without improving the scientific handoff nearly as much as compact summaries and the memo do.
- Impact: future large campaigns can still write reusable checkpoint trees and full raw JSONs under `results/`, but closeout should commit the manifest, compact split/predictiveness summaries, and the markdown write-up rather than the raw cache directory or oversized JSON dumps.

## [2026-03-17T12:12:00-0500] DECISION: Start Phase 2 pattern analysis from saved raw oracle alphas, not the compact summaries

- Trigger: `resattn-tpw` needed the first prereg-scale pattern-analysis slice after the development-model oracle gate cleared, and the compact campaign JSONs did not preserve per-sequence `final_alpha` vectors.
- Decision: add `validation/pattern_analysis.py` plus `scripts/run_oracle_alpha_pattern_analysis.py`, and make the first reusable pattern-analysis path consume the saved raw `oracle_eval_run.json`, summarize source-type mass, and scan average-linkage clustering on Jensen-Shannon distances against a matched random Dirichlet control.
- Rationale: the saved raw oracle artifact is the smallest honest input that still preserves the sequence-level routing object we need for pattern analysis. Using average linkage directly on JSD respects the methodology lock, while a matched random control keeps the first clustering read anchored without pretending this already answers the broader block-structure question.
- Impact: the repo now has a reusable sequence-level pattern-analysis module and the first prereg-scale artifact. The resulting structure is weakly above random but outlier-driven and below the prereg `silhouette > 0.2` gate, so the next pattern follow-up should stress-test grouped-source and resampling robustness rather than overclaim a clean block structure.

## [2026-03-17T12:32:00-0500] DECISION: Keep tuned-lens and Figure 8 strong-claim validation on the primary scientific objects

- Trigger: `resattn-qm4` needed to resolve two branching decisions before the tool-breakage and Figure 8 lanes start implementation: whether tuned-lens-aware tool-breakage stays on Gemma-2 or moves to a secondary model, and what reproducible proxy is required for strong Figure 8 alignment claims.
- Decision: train a custom Gemma-2 tuned lens locally for the primary tool-breakage lane, treat any secondary-model tuned-lens comparison as supplementary only, and require a small local AttnRes reproduction as the default reproducible proxy for strong Figure 8 / trained-routing alignment claims.
- Rationale: the repo already locks Gemma-2-2B as the primary model for main results and the tool-breakage lane is supposed to compare routed-versus-original behavior on that exact model. Satisfying the tuned-lens requirement on a secondary model would weaken the claim instead of hardening it. For Figure 8, the small local AttnRes reproduction is the most direct way to address the co-adaptation objection and the lack of released small-scale checkpoints.
- Impact: `resattn-qm4` can now close at the decision level, and the next implementation work should be split into explicit follow-up issues for custom Gemma-2 tuned-lens training and the small local AttnRes reproduction rather than leaving either path implicit.

## [2026-03-17T12:56:00-0500] DECISION: Treat the first Gemma-2 tuned-lens pilot as a viability pass with metric caveats

- Trigger: `resattn-5k9` finished the first full-surface original-model tuned-lens pilot on `google/gemma-2-2b`, and the repo needed to decide whether that was strong enough to keep the primary tool-breakage lane on Gemma rather than revisiting the secondary-model escape hatch.
- Decision: treat `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1.md` as a real viability pass for the custom Gemma tuned-lens path, close `resattn-5k9`, and keep the next tool-breakage follow-up focused on metric choice or objective sharpening rather than on re-proving basic feasibility.
- Rationale: the held-out pilot on `96` train prompts and `8` factual-recall eval prompts improved mean KL to the final distribution from `11.3881` to `3.4580` and mean top-1 agreement from `0.1863` to `0.5119`, while also improving final-position KL from `14.8685` to `7.5721`. That is enough evidence that a same-model tuned-lens-aware baseline is operationally credible on Gemma-2. The caveat is that final-position top-1 only moved from `0.1010` to `0.1250`, so the current strength is distributional recovery, not strong answer-token recovery.
- Impact: the tool-breakage lane no longer needs to spend time debating whether custom Gemma tuned-lens training is feasible. The next work should decide whether later routed-versus-original analysis will use held-out KL as the primary tuned-lens baseline metric or whether the lens objective should be sharpened for factual-recall final-position behavior before making answer-token-facing claims.

## [2026-03-17T13:05:00-0500] DECISION: Make Gemma tool-breakage metric hierarchy KL-primary and keep answer-token interpretation gated

- Trigger: `resattn-ehz` needed to convert the saved Gemma tuned-lens viability artifact into an explicit control decision before the routed-versus-original factual-recall lane starts.
- Decision: use held-out KL to the model's final output distribution as the primary tuned-lens baseline metric for Gemma-2 factual recall, require mean top-1 agreement plus final-position KL/top-1 as secondary diagnostics, and do not delay the routed lane to re-optimize the tuned lens for final-position answer-token recovery right now.
- Rationale: the saved pilot is strongest on distributional fidelity, not final answer-token recovery. `25 / 26` layers improved on held-out KL, `25 / 26` improved on mean top-1, `25 / 26` improved on final-position KL, but only `4 / 26` improved on final-position top-1. That is enough to trust KL as the cleaner same-model tuned-lens baseline for later lens-breakage comparisons, while also making it explicit that answer-token-facing claims need direct final-position evidence.
- Impact: the tool-breakage lane can move forward on Gemma-2 without re-opening the tuned-lens feasibility question, but later routed-versus-original write-ups must report the secondary final-position diagnostics and avoid letting KL improvements stand in for answer-token claims.

## [2026-03-17T13:45:00-0500] DECISION: Treat `resattn-28b` as a baseline implementation pass and block confirm on stronger relative tool-breakage metrics

- Trigger: the first same-model Gemma routed-versus-original factual-recall pilot finished on the saved `8`-prompt pilot split with prompt-level checkpoints and a complete `summary.json`.
- Decision: land `resattn-28b` as the first baseline implementation pass, keep the resulting pilot artifact as valid evidence of routed-versus-original degradation on the KL-primary metrics, and block any confirmatory factual-recall run on the new follow-up `resattn-ypj` rather than interpreting the current non-monotonicity boolean directly.
- Rationale: the pilot shows broad same-model degradation under routing:
  - tuned mean KL worsened on `8 / 8` prompts
  - tuned final-position KL worsened on `8 / 8` prompts
  - raw mean KL worsened on `7 / 8` prompts
  - raw and tuned mean top-1 both worsened on `8 / 8` prompts
  That is enough to say the runner works and the routed-versus-original baseline is scientifically useful. But the prereg legacy threshold is phrased around non-monotonicity relative to the original baseline, and the boolean version of that metric is already saturated here: original-model raw and tuned traces are non-monotonic on all `8 / 8` pilot prompts before routing is applied. A blind confirm run would therefore ask the wrong decision question.
- Impact: the next tool-breakage issue should codify stronger relative success metrics from the saved pilot traces, likely using target-rank degradation or another routed-versus-original instability summary, and only then move to a confirmatory Gemma factual-recall run. The current pilot remains a real baseline artifact, not a strong-claim pass.

## [2026-03-17T14:05:00-0500] DECISION: Use relative target-rank instability as the next Gemma tool-breakage decision surface

- Trigger: `resattn-ypj` compared candidate relative metrics directly on the saved `resattn-28b` pilot traces because the legacy non-monotonicity boolean was already saturated on the original-model baseline.
- Decision: keep the legacy routed-versus-original non-monotonicity increase metric in the summary for continuity, but promote relative target-rank metrics to the actionable decision surface before the confirm run:
  - routed-versus-original final-target-rank worsening
  - routed-versus-original best-target-rank worsening
  - routed-versus-original target-rank-range increase
- Rationale: on the saved pilot traces, non-monotonicity increase stayed `0 / 8` under both raw and tuned lens, so it carries almost no discriminative value. The rank metrics are more informative on the same artifact:
  - tuned routed traces increase target-rank range on `7 / 8` prompts
  - tuned routed traces worsen the best target rank on `4 / 8` prompts
  - tuned routed traces worsen the final-layer target rank on `4 / 8` prompts
  That makes rank-based summaries the smallest honest way to preserve the prereg “non-monotonicity or rank-instability” language without inventing a new runner or rerunning Gemma before the confirm step.
- Impact: the next tool-breakage issue should run the locked confirm split using the codified relative rank metrics, while the later controlled dynamic-routing counterfactual remains a separate blocker for the strong claim.

## [2026-03-17T13:52:00-0500] DECISION: Treat the locked Gemma confirm split as a confirmatory rank-instability pass, not a final strong-claim pass

- Trigger: `resattn-6te` completed the first locked same-model Gemma factual-recall confirm run using the metric surface codified in `resattn-ypj`.
- Decision: record the confirm artifact as a successful confirmatory baseline pass for routed-versus-original KL degradation and rank instability, but keep the strong tool-breakage claim blocked on the later controlled dynamic-routing counterfactual.
- Rationale: the confirm result reproduces the same-model degradation story on unseen prompts:
  - mean tuned-lens KL worsened from `3.3834` to `5.9061`
  - final-position tuned-lens KL worsened from `5.9851` to `8.8947`
  - tuned rank-range increase is `7 / 8`
  - tuned final-target-rank worsening is `5 / 8`
  - raw final-target-rank worsening is `6 / 8`
  At the same time, routed-versus-original non-monotonicity increase remains `0 / 8` under both raw and tuned lens, so the confirm result still does not rescue the old absolute boolean story. The honest confirm read is “routing increases rank instability and often worsens answer-token rank,” not “routing newly induces non-monotonic curves.”
- Impact: the next tool-breakage issue should focus on the prereg-required controlled dynamic-routing counterfactual rather than on re-running the same confirm baseline or revisiting the metric surface again.

## [2026-03-17T14:07:00-0500] DECISION: Use two controlled alpha baselines in `resattn-g09`, with prompt-permuted confirm alphas as the primary counterfactual

- Trigger: `resattn-g09` is the next live blocker for the Gemma tool-breakage lane, and the repo needs a concrete counterfactual design before any implementation starts.
- Decision: implement the controlled dynamic-routing counterfactual with two saved control arms in one runner:
  - a primary prompt-permuted confirm-alpha control built by deterministic cyclic reassignment of the saved `resattn-6te` confirm oracle alphas
  - a secondary fixed-alpha control built from the mean oracle alpha over the saved `resattn-28b` pilot prompts
- Rationale: the prompt-permuted control is the cleanest way to preserve non-uniform routed mixtures while breaking prompt alignment, which is the actual causal question behind the strong claim. The pilot-mean alpha control is cheaper to add in the same pass and answers the simpler static-routing objection without needing a second run. Reusing the saved baseline oracle alphas keeps the counterfactual artifact anchored to the already-registered Gemma baseline instead of silently re-optimizing a nearby variant.
- Impact: `resattn-g09` should compare routed traces against the original baseline and both control arms on the existing tuned-KL and relative target-rank metrics. If routed degradation remains stronger than both controls, the tool-breakage lane clears a much stronger confirmatory bar; if not, the strong claim boundary tightens immediately.

## [2026-03-17T14:19:00-0500] DECISION: Treat `resattn-g09` as a mixed / negative control result that blocks the strongest same-model Gemma claim

- Trigger: the locked `resattn-g09` confirm artifact completed on the `8`-prompt Gemma factual-recall confirm split with both the prompt-permuted and fixed-alpha controls saved.
- Decision: close `resattn-g09` as a successful implementation of the preregistered dynamic counterfactual and record the scientific outcome as mixed:
  - the fixed `pilot_mean_alpha` control is weaker than the prompt-matched routed trace
  - the prompt-permuted dynamic control is more damaging than the prompt-matched routed trace on the tuned primary KL metric
  - the strong same-model Gemma tool-breakage claim therefore stays blocked
- Rationale: the new artifact answers the preregistered control question directly. On the tuned primary metric:
  - routed minus `pilot_mean_alpha` mean KL is `+0.6091`
  - routed minus `prompt_permuted_alpha` mean KL is `-0.3082`
  - routed minus `prompt_permuted_alpha` final-position tuned KL is `-0.6541`
  The rank-based confirm read points in the same direction: routed worsens tuned final target rank versus the prompt-permuted control on only `4 / 8` prompts and increases tuned rank range on only `2 / 8`. That is not enough to say the prompt-matched route is uniquely responsible for the observed lens damage.
- Impact: the Gemma lane should now be written up as a routed-versus-original degradation that is stronger than a fixed non-uniform control but not stronger than the current prompt-misaligned dynamic control. The next highest-value implementation step should move to another lane, while any future Gemma follow-up must be framed as a new control-design question rather than a missing prerequisite.

## [2026-03-17T14:33:00-0500] DECISION: Start `resattn-7hb` with an `8`-block local Block AttnRes viability slice, not a paper-scale reproduction

- Trigger: after `resattn-g09` landed, `resattn-7hb` became the highest-value ready issue, and the repo needed a concrete minimum viable proxy definition before adding any training code.
- Decision: define the first local AttnRes proxy slice as a small decoder-only Block AttnRes reproduction with:
  - `8` routing blocks so the proxy can actually express the `~8`-block Figure 8 structure the paper highlights
  - a matched standard-residual baseline
  - explicit sublayer-granular routing export for later Figure 8 metrics
  - a local feasibility-first training target rather than a paper-scale benchmark target
- Rationale: the proxy exists to make trained-routing alignment claims reproducible, not to replicate the paper’s absolute benchmark numbers on local hardware. An `8`-block model is the smallest setup that still preserves the key structural object from the paper, while a matched baseline prevents the proxy from collapsing into a one-off architecture demo. Jumping straight to a larger 194M+ reproduction before verifying stable training and exportability on local hardware would be the wrong order of operations.
- Impact: the first `7hb` implementation should build the training/export scaffolding and run a tiny viability slice before any longer tmux-backed training campaign is launched. If that slice fails to train stably or cannot export routing at the needed granularity, the lane should pause and reconsider architecture scope before burning more compute.

## [2026-03-17T15:05:00-0500] DECISION: Close `resattn-7hb` on local-proxy viability, not on Figure 8 alignment

- Trigger: the first smoke and scaled Wikitext runs from `resattn-7hb` both completed with checkpoints, rerunnable resume paths, exported routing summaries, and a real eval-loss win for the local Block AttnRes proxy over the matched baseline.
- Decision: treat `resattn-7hb` as successful once the repo lands the local `8`-block proxy scaffold plus the two viability artifacts, but do not treat the lane as a Figure 8 alignment pass.
- Rationale: the issue asked for the smallest local AttnRes reproduction and matched baseline setup that could support later Figure 8 checks. The scaled artifact now proves that this setup is operationally real. At the same time, the saved proxy metrics remain mixed:
  - deep embedding persistence stayed low at `0.1049`
  - mean pre-attn entropy remained below mean pre-MLP entropy (`1.4674 < 1.5264`)
  - the stronger run improved the loss comparison without improving the paper-facing metric surface
  This makes the right next issue a proxy-regime decision, not more pretending that `7hb` already solved trained-routing alignment.
- Impact: the next Figure 8 follow-up becomes `resattn-3l6`, which decides whether to scale the current char-level regime further or move to a stronger tokenizer / corpus setup before claim-bearing proxy analysis.

## [2026-03-17T15:25:00-0500] DECISION: Make the next Figure 8 proxy run more realistic by changing tokenization, not by extending char-level training again

- Trigger: `resattn-3l6` exists because the saved `7hb` char-level Wikitext artifacts were operationally successful but scientifically mixed, and Sohail explicitly gave room for a longer or bigger run if it adds real signal.
- Decision: keep the current `8`-block architecture, dataset family, and matched-baseline setup fixed, but move the next proxy run from character-level modeling to compact remapped GPT-2 subword tokenization on the same Wikitext corpus before spending more time on another char-level rerun.
- Rationale: the scaled char-level run already answered the “just train it longer” question well enough. Loss improved more strongly, but the Figure 8-facing metrics did not move in the right direction:
  - the proxy beat the baseline by `0.0327` eval-loss points
  - deep embedding persistence dropped from `0.1824` to `0.1049`
  - the entropy ordering stayed inverted (`mean_pre_attn_entropy < mean_pre_mlp_entropy`)
  That pattern is more consistent with a regime mismatch than with a simple undertraining problem. Switching to compact subword ids is the smallest higher-fidelity change because it preserves the local training setup while removing the most obvious mismatch with normal LM training.
- Impact: `validation/attnres_reproduction.py` and `scripts/run_attnres_proxy_viability.py` should support a compact-subword mode, and the next run should be a checkpointed Wikitext compact-subword viability slice on the same `8`-block architecture. If that run remains mixed, the next follow-up should debate corpus or capacity, not tokenizer realism again.

## [2026-03-17T15:40:00-0500] DECISION: Keep compact subword as the Figure 8 proxy default and move the next follow-up to capacity or optimization, not back to characters

- Trigger: the compact-subword Wikitext artifact from `resattn-3l6` completed at the same `1500`-step horizon as the earlier char-level scaled run.
- Decision: close the tokenizer-vs-char decision in favor of compact remapped GPT-2 subword tokenization as the default local Figure 8 proxy regime, and create the next follow-up around capacity or optimization scaling rather than another tokenization change.
- Rationale: the compact-subword run moved the paper-facing metrics in the right direction relative to the char-level scaled artifact:
  - deep embedding persistence improved from `0.1049` to `0.1364`
  - the entropy gap improved from `-0.0590` to `-0.0349`
  At the same time, tokenization realism alone did not solve the proxy:
  - the compact-subword proxy lost the longer-horizon baseline comparison (`6.6764` vs `6.6463`)
  - the entropy ordering still remained inverted
  That means the next bottleneck is more likely model capacity or optimization horizon than tokenization choice.
- Impact: `resattn-3l6` can close once the new artifact and state docs land, and the next Figure 8 issue becomes `resattn-111` rather than another char-level rerun.

## [2026-03-17T16:40:00-0500] DECISION: Treat width-only scaling as informative but insufficient, and move the next Figure 8 follow-up to optimization horizon

- Trigger: `resattn-111` widened the compact-subword local Block AttnRes proxy from `d_model=96`, `d_ff=384` to `d_model=160`, `d_ff=640` while keeping the `1500`-step horizon, corpus, and tokenization fixed.
- Decision: close the capacity-isolation issue as a useful mixed result, keep the widened compact-subword regime as the current Figure 8 proxy default, and make the next follow-up a horizon-only continuation rather than another width or tokenization change.
- Rationale: width helped the proxy somewhat, but not in the way needed for the claim:
  - the widened AttnRes proxy improved from `6.6764` to `6.6496` best eval loss
  - the widened matched baseline improved more strongly from `6.6463` to `6.5899`
  - deep embedding persistence improved again from `0.1364` to `0.1515`
  - the entropy gap regressed from `-0.0349` to `-0.0561`, leaving the ordering inverted
  This is enough to say capacity alone is not the full fix at the fixed `1500`-step horizon. The remaining clean next question is whether the widened proxy is simply under-optimized.
- Impact: `resattn-111` can close once the artifact and state docs land, and the next Figure 8 issue becomes `resattn-jci` rather than another width or tokenization sweep.

## [2026-03-17T17:15:00-0500] DECISION: Treat the widened compact-subword horizon follow-up as a negative result and stop scaling the same regime blindly

- Trigger: `resattn-jci` resumed the widened compact-subword Figure 8 proxy from `1500` to `4500` total steps without changing width, tokenization, corpus, or seed.
- Decision: close the horizon-only follow-up as a negative result for the “it just needs more optimization budget” hypothesis, and move the next Figure 8 task to a bounded redesign decision instead of another width or horizon rerun on the same local regime.
- Rationale: the longer run did not improve the best widened comparison at all:
  - baseline best eval loss stayed `6.5899`
  - AttnRes best eval loss stayed `6.6496`
  - the loss delta stayed `+0.0598`
  The Figure-8-facing read also remained mixed:
  - deep embedding persistence softened from `0.1515` to `0.1415`
  - the entropy gap remained negative (`-0.0460`)
  - the entropy ordering stayed inverted
  That is enough evidence to stop treating optimization horizon as the obvious next lever on this exact widened compact-subword setup.
- Impact: `resattn-jci` can close once the artifact and state docs land, and the next Figure 8 issue becomes `resattn-du2`, which should choose one redesign lever explicitly instead of extending the current regime again.

## [2026-03-17T17:32:00-0500] DECISION: Use aligned Gemma for the safety lane, not the base primary-model spine

- Trigger: `resattn-3f1` needed a real refusal-analysis precondition check before any safety workflow code could be trusted.
- Decision: keep `google/gemma-2-2b` as the primary frozen-model spine for oracle-alpha and tool-breakage, but use `google/gemma-2-2b-it` as the safety-alignment model for refusal-feature discovery and later safety-lane follow-ups.
- Rationale: a direct preflight showed the base `google/gemma-2-2b` model complied with a harmful request instead of refusing, while `google/gemma-2-2b-it` refused the same request and answered the high-level harmful-context control normally. A refusal-feature workflow on the base model would have been invalid by construction.
- Impact: `configs/experiment.yaml` now encodes the safety-model override, and the safety lane can proceed without conflating the main frozen-model spine with the aligned-refusal spine.

## [2026-03-17T17:47:00-0500] DECISION: Close `resattn-3f1` on workflow validation, not on a causal safety claim

- Trigger: the first full aligned-Gemma refusal-feature artifact completed on the frozen `6 / 6` pilot/confirm prompt triples with prompt-level checkpoints, held-out summaries, and resume verification.
- Decision: treat `resattn-3f1` as complete once the repo lands the aligned-Gemma workflow validator, the frozen prompt collection, and the held-out validation artifact, while keeping mediator-conditioned safety-routing claims blocked on a later causal intervention issue.
- Rationale: the workflow now clears the blocker it was meant to clear:
  - behavior checks matched expectation on `36 / 36` prompts
  - refusal localized to assistant-prefill layer `22`
  - harmfulness localized to instruction-final layer `25`
  - confirm pair accuracy stayed `1.0` for both primary directions
  - refusal and harmfulness directions were nearly orthogonal (`cosine = 0.0064`)
  At the same time, the cross-direction confirm metrics are not zero (`0.6667` and `0.8333`), so this is not evidence for a fully disentangled single-direction safety story, and it is not yet a mediator-conditioned routing result.
- Impact: `resattn-3f1` can close honestly as a workflow-validation success, and the next safety follow-up is `resattn-73l`, which adds the causal refusal-direction intervention check before any stronger safety-routing claim.

## [2026-03-17T18:05:00-0500] DECISION: Make corpus size the next Figure 8 redesign lever and defer objective and architecture changes

- Trigger: `resattn-du2` followed the compact-subword width-only and horizon-only negatives and needed one bounded redesign decision before more compute was spent on the Figure 8 proxy lane.
- Decision: keep compact remapped GPT-2 subword tokenization, keep the widened `d_model=160`, `d_ff=640`, `8`-block local proxy, keep the standard next-token objective, and make the next Figure 8 run a corpus-first follow-up on `wikitext/wikitext-103-raw-v1`.
- Rationale: the saved artifacts point most strongly at corpus-size mismatch and overfitting on `wikitext-2-raw-v1`, not at a simple optimization shortage or an obviously wrong proxy architecture:
  - compact subword improved the Figure-facing metrics relative to the char-level run
  - width improved deep embedding persistence again
  - horizon from `1500` to `4500` changed the best eval losses by exactly `0.0` while train loss kept falling
  That combination is more consistent with a too-small corpus than with “just add more width,” “just train longer,” or “change the LM objective.”
- Impact: `resattn-du2` can close once the redesign memo and state docs land, and the next Figure 8 issue becomes `resattn-7y4`, which runs the widened compact-subword proxy on `wikitext-103` while deferring objective, architecture, and sequence-length changes.

## [2026-03-17T18:24:00-0500] DECISION: Keep the widened Figure 8 training surface fixed on `wikitext-103`, but enlarge the validation slice for lower-noise readout

- Trigger: `resattn-7y4` needs one final run specification before launch, and the compact-subword vocabulary sweep on `wikitext-103` showed that `2048` train texts remain within the existing `vocab_size=20000` limit while larger train slices would force an embedding-table change.
- Decision: keep the training surface identical to the widened `wikitext-2` compact-subword runs (`2048` train texts, `d_model=160`, `d_ff=640`, `8` blocks, `1500` steps, `vocab_size=20000`) and increase only the validation slice from `256` to `512` texts for the `wikitext-103` artifact.
- Rationale: this preserves the corpus-first redesign discipline by avoiding a hidden architecture change through a larger compact vocabulary, while still buying a more stable evaluation readout on the larger corpus. The measured sweep showed:
  - `2048 / 256` uses observed compact vocabulary `18494`
  - `2048 / 512` uses observed compact vocabulary `19250`
  - `2048 / 1024` would exceed the current limit at `20646`
  - `4096 / 256` would force a larger vocabulary at `24235`
- Impact: `resattn-7y4` becomes a clean corpus-first follow-up with a stronger held-out evaluation surface, and any future move to larger train slices now clearly counts as a later architecture-linked redesign rather than an unnoticed continuation of the same regime.

## [2026-03-17T18:49:00-0500] DECISION: Treat `resattn-7y4` as a mixed corpus-first result and move the next Figure 8 follow-up to checkpoint-level optimization visibility

- Trigger: the widened compact-subword `wikitext-103` run finished with a narrower baseline gap and improved deep embedding persistence, but still did not restore a routed win or the prereg entropy ordering.
- Decision: close `resattn-7y4` as a useful mixed result, do not interpret it as a solved Figure 8 follow-up, and create `resattn-fby` as the next Figure 8-specific issue for best-checkpoint / eval-trajectory support before another optimization redesign.
- Rationale: the larger corpus changed the failure mode enough to justify a different next question:
  - widened `wikitext-2` loss delta: `+0.0598`
  - widened `wikitext-103` loss delta: `+0.0386`
  - deep embedding persistence improved from `0.1515` to `0.1615`
  - the entropy ordering stayed inverted (`1.4318 < 1.4893`)
  - the exact same `wikitext-103` regime was positive at `200` steps during calibration, then negative at the full `1500`-step horizon
  That means corpus size helped, but the runner still cannot answer the now-live question: whether an earlier more competitive checkpoint also has a better Figure 8 surface. The current runner only preserves the final model plus a scalar `best_eval_loss`, which is not enough.
- Impact: the repo should stop treating more same-regime training as the obvious next Figure 8 move. The next Figure 8 work should first preserve checkpoint-level eval history and best-model state. Overall repo priority can reasonably shift to `resattn-73l` while that Figure 8 follow-up waits.

## [2026-03-17T18:13:35-0500] DECISION: Close `resattn-73l` on bounded causal mediator evidence, with continuation preference as the decisive readout

- Trigger: the aligned-Gemma causal mediator run on the frozen `6 / 6` confirm split completed after a smoke showed the original greedy refusal-marker readout was too coarse on its own.
- Decision: treat `resattn-73l` as complete once the repo lands the bounded coefficient-replacement intervention runner plus the confirm artifact, and interpret the result through both the greedy refusal marker and a matched safe continuation-preference metric.
- Rationale: the full artifact shows a real causal mediator effect while staying inside the repo’s safety guardrails:
  - refusal suppression on refusal prompts lowered the refusal-versus-context preference margin by `0.1891`
  - the matched harmfulness suppression control stayed exactly flat
  - refusal injection on harmful-context prompts raised the same preference margin by `0.1741` and flipped the greedy refusal marker on `1 / 6` prompts
  - refusal injection on benign prompts also raised refusal preference (`+0.2374`) without producing greedy refusal flips
  - the matched harmfulness injection control on benign prompts stayed flat
  This is enough to clear the “mediator must be causal, not just separable” blocker. It is not enough to claim a highly selective harmful-only refusal switch, because the benign preference shift is real and the binary refusal marker stays mostly saturated.
- Impact: `resattn-73l` can close honestly as a bounded causal mediator pass. The next safety issue becomes `resattn-h1p`, which moves to mediator-conditioned routing analysis on aligned Gemma. Overall repo priority should now return to `resattn-fby`, because the Figure 8 lane has the stronger cross-lane blocker.

## [2026-03-17T18:46:49-0500] DECISION: Treat best-checkpoint export as necessary Figure 8 instrumentation, not as the missing scientific fix

- Trigger: `resattn-fby` added eval-history plus best-checkpoint persistence to the widened compact-subword `wikitext-103` proxy and reran the full `1500`-step artifact on the same seed and architecture.
- Decision: close the trajectory-support issue as a useful mixed result, keep the widened `wikitext-103` regime as the current local Figure 8 default, and stop treating checkpoint selection alone as the next explanatory lever.
- Rationale: the new artifact changed the measured shape, but not the lane conclusion:
  - the matched baseline best step was `900`
  - the AttnRes best step was `850`
  - final loss delta was `+0.1272`, while best-checkpoint loss delta softened to `+0.0386`
  - deep embedding persistence improved from `0.1615` at the final checkpoint to `0.1689` at the best checkpoint
  - the entropy gap improved only slightly from `-0.0574` to `-0.0549`, leaving the ordering inverted
  That is enough to say the earlier runner was understating the regime by evaluating the final checkpoint alone, but not enough to say optimization-shape fixes the Figure 8 problem. The baseline still wins and the paper-facing layer-type-specialization signature still points the wrong way.
- Impact: `resattn-fby` can close once the artifact and state docs land. The next Figure 8 follow-up should be framed as a bounded optimization or objective redesign question rather than another blind rerun of the same widened compact-subword `wikitext-103` setup; that follow-up is now tracked as `resattn-8xu`.

## [2026-03-17T19:00:00-0500] DECISION: Treat grouped-source pattern structure as coarse-regime evidence, not as a prereg block-structure pass

- Trigger: `resattn-ojq` extended the prereg-scale `gpt2-xl` pattern-analysis artifact with grouped-source views and `128` prompt-resampling checks over `96`-prompt subsamples.
- Decision: close the robustness follow-up as a useful mixed result, keep the raw-source block-structure gate unpassed, and treat the grouped-source gains as evidence for coarse source-type routing variation rather than for a clean `~8`-cluster decomposition.
- Rationale: the control extensions sharpened the interpretation instead of flipping it:
  - raw-source clustering survives resampling, but only weakly (`0.1428` versus random `0.1093`) and with extreme dominance (`126 / 2` full sample, best `k = 2` on `128 / 128` resamples)
  - grouped views raise the above-random signal:
    - `source_type`: `0.6652` versus `0.5569`, `87 / 41` at `k = 2`
    - `depth_thirds_by_type`: `0.3451` versus `0.2305`, but still `117 / 11` at `k = 2`
  - the stronger grouped view is only `3` dimensions and therefore too coarse to support the prereg `~8`-cluster block hypothesis
  - the more expressive `7`-group depth-banded view still collapses mostly to `k = 2`
  This is enough to defend a coarse routing-regime story at grouped source type, but not enough to claim broad block structure in the raw routing object.
- Impact: `resattn-ojq` can close once the artifact and state docs land. The next overall repo step should move to another major lane rather than spending more time polishing the current raw block-structure story.

## [2026-03-17T19:26:00-0500] DECISION: Close `resattn-h1p` on bounded mediator-conditioned trajectory evidence, not on prompt-level mediator separation

- Trigger: the strengthened aligned-Gemma mediator-conditioned routing runner completed with confirm trajectory summaries plus intervention-conditioned trajectory comparisons on the frozen `6 / 6` prompt groups.
- Decision: treat `resattn-h1p` as complete once the repo lands the mediator-partition summary, the role-trajectory artifact, and the intervention-conditioned trajectory comparisons, while keeping stronger safety-routing claims blocked on a later prompt-surface follow-up.
- Rationale: the new artifact clears the narrow question that blocked the safety lane:
  - the confirm mediator partition is still exactly role-collapsed (`6` active refusal prompts versus `12` inactive non-refusal prompts)
  - but refusal-direction interventions at assistant-prefill layer `22` propagate large shifts to the final-layer refusal-direction trajectory:
    - refusal suppression on refusal prompts: `-372.8501`
    - refusal injection on harmful-context prompts: `+369.9403`
    - refusal injection on benign prompts: `+358.7685`
  This is enough to move beyond a descriptive role relabeling result. It is not enough to claim the current prompt set exposes a subtler mediator-active subset inside the non-refusal roles.
- Impact: `resattn-h1p` can close honestly as a bounded stage-3 safety artifact. The repo should now treat stronger safety-routing language as dependent on `resattn-mo5`, a broader or less role-collapsed prompt-surface follow-up, rather than on more rewrites of this same frozen collection.

## [2026-03-17T19:45:00-0500] DECISION: Make the next Figure 8 redesign regularization-first and defer objective changes

- Trigger: `resattn-8xu` needed to turn the mixed widened `wikitext-103` best-checkpoint artifact into a concrete next question instead of another blind rerun.
- Decision: keep the widened compact-subword `wikitext-103` proxy and the standard next-token objective fixed, and make the next Figure 8 follow-up a bounded regularization sweep rather than an objective-level redesign.
- Rationale: the saved proxy trail now says two things at once:
  - the current objective can reach a healthier operating region (`200`-step positive calibration and best-checkpoint loss delta `+0.0386` instead of final-checkpoint `+0.1272`)
  - but the proxy does not sustain a routed win and still misses the entropy ordering at the best checkpoint (`-0.0549`)
  That is a stronger case for stabilization failure than for “the standard objective can never express the desired regime.” Changing the objective now would make any positive result less faithful to the local AttnRes proxy story.
- Impact: `resattn-8xu` can close once the memo and state docs land. The next Figure 8 issue is `resattn-bux`, which compares three matched regularization settings on the existing widened `wikitext-103` best-checkpoint-enabled regime before any objective-level change is allowed.

## [2026-03-17T19:48:30-0500] DECISION: Treat matched regularization as exhausted for the widened `wikitext-103` Figure 8 proxy

- Trigger: `resattn-bux` completed the three-arm regularization-first sweep on the best-checkpoint-enabled widened compact-subword `wikitext-103` proxy.
- Decision: close the regularization-first follow-up as a useful negative result and move the Figure 8 lane to an objective-level redesign question rather than another matched stabilization sweep.
- Rationale: the three new arms all stayed inside the control band on the metrics that actually matter:
  - control loss delta `= +0.0386`
  - new loss deltas `= +0.0378`, `+0.0376`, `+0.0368`
  - control entropy gap `= -0.0549`
  - best new entropy gap `= -0.0533`
  Dropout improved deep embedding persistence, but no arm restored a routed win or materially improved the layer-type-specialization signature. That means the faithful “just stabilize this regime” explanation is now weak.
- Impact: `resattn-bux` can close once the artifact and state docs land. The next Figure 8 issue becomes `resattn-9fo`, which should choose the smallest honest objective-level redesign instead of reopening regularization or another same-regime rerun.

## [2026-03-17T19:58:00-0500] DECISION: Freeze the current Gemma dynamic-control claim boundary and do not add a finer same-model control now

- Trigger: `resattn-5d9` asked whether the mixed `resattn-g09` counterfactual result justified a narrower follow-up control or whether that would amount to moving the goalposts.
- Decision: close `resattn-5d9` by freezing the current same-model Gemma claim boundary and declining any finer dynamic-control follow-up for now.
- Rationale: the existing counterfactual already answers the decisive question:
  - routed is clearly worse than the fixed `pilot_mean_alpha` control
  - routed is not worse than the prompt-permuted dynamic control on the tuned primary metric
  - the rank-based confirm diagnostics versus the prompt-permuted control are also weak (`4 / 8` final-rank worsening, `2 / 8` rank-range increase)
  That is enough to block the strong prompt-matched-routing claim. A narrower donor-matching control on the same `8`-prompt confirm surface would currently read more like post-hoc rescue than like a missing prerequisite.
- Impact: `resattn-5d9` can close once the memo and state docs land. The Gemma lane should stay frozen at the current mixed claim boundary unless a later methodological defect or larger confirm surface justifies reopening the control design.

## [2026-03-17T20:05:00-0500] DECISION: Do not add a custom objective to the current tiny Figure 8 proxy

- Trigger: `resattn-9fo` asked for the smallest honest objective-level redesign after the negative regularization sweep on the widened compact-subword `wikitext-103` proxy.
- Decision: decline any objective-level redesign on the current local Block AttnRes proxy and freeze the strong Figure 8 / trained-routing lane at the current descriptive boundary.
- Rationale: the repo has already exhausted the faithful small-proxy adjustments:
  - tokenization
  - width
  - horizon
  - corpus
  - best-checkpoint export
  - matched regularization
  The obvious next objective changes would be custom proxy-shaping losses, not paper-faithful training choices. That would make a positive result harder to interpret than the current negative one.
- Impact: `resattn-9fo` can close once the memo and state docs land. The next Figure 8 strategic issue is `resattn-1lk`, which decides whether to keep the strong lane frozen or revisit it through a more faithful proxy path rather than through a custom loss on the current tiny model.

## [2026-03-17T20:12:00-0500] DECISION: Reprioritize the next scientific move to the primary Gemma oracle path and treat `resattn-1lk` as second-order

- Trigger: the external review correctly pointed out that the biggest remaining paper-shape risk is still that the strongest positive oracle-alpha result lives on the development model, while the repo's ready queue only exposed `resattn-1lk`, `resattn-mo5`, and `resattn-9co`.
- Decision: create `resattn-2s0` and `resattn-7cs`, elevate primary-spine Gemma oracle readiness above the Figure 8 strategic decision, and update the reviewer memo plus state docs so they reflect that ordering explicitly.
- Rationale: a frozen-model paper that clears its main oracle gate only on the development model remains structurally weaker than one that at least begins to replicate on the primary spine. `resattn-1lk` still matters, but it is a freeze-or-escalate decision on a currently mixed lane; it is not the next strongest way to reduce overall thesis risk.
- Impact: the next top scientific action is a bounded Gemma oracle-alpha feasibility slice (`resattn-7cs`). `resattn-1lk` stays next as a strategic Figure 8 decision, `resattn-mo5` stays the strongest extension candidate after that, and `resattn-9co` remains deferred cleanup.

## [2026-03-17T20:18:00-0500] DECISION: Treat the primary-spine Gemma reconstruction smoke as green and move the blocker forward

- Trigger: `resattn-2s0` ran the existing TransformerLens-backed reconstruction smoke unchanged on `google/gemma-2-2b`.
- Decision: close primary-spine backend readiness for the reconstruction layer as solved on local MPS and move the oracle-alpha blocker forward from reconstruction to first bounded primary-model execution.
- Rationale: the smoke came back exact on the primary model:
  - `53` sources
  - `final_residual_max_abs_error = 0.0`
  - `uniform_logits_max_abs_error = 0.0`
  - all `26` `resid_mid` and `resid_post` identities exact
  That removes the main remaining reason to keep the primary model out of the oracle lane.
- Impact: `resattn-2s0` can close once the artifact and state docs land. The next oracle-alpha issue is `resattn-7cs`, the first bounded Gemma feasibility slice.

## [2026-03-17T20:24:00-0500] DECISION: Treat the first bounded Gemma oracle-alpha slice as a clean primary-spine feasibility pass

- Trigger: `resattn-7cs` ran the existing bounded development-slice runner unchanged on `google/gemma-2-2b` after primary-spine reconstruction readiness cleared.
- Decision: close the first primary-model oracle-alpha slice as a successful feasibility pass and move the next blocker forward to the saved pilot stability suite.
- Rationale: the runner executed cleanly on the primary spine and returned a materially positive bounded artifact:
  - mean sequence improvement over uniform `= +2.1634` nats on `2` pilot prompts
  - mean null losses stayed above the optimized slice (`random_dirichlet = 6.6138`, `magnitude_proportional = 6.5578`, `last_layer_only = 24.7532`)
  That is enough to show that the existing oracle runner is not development-model-only in its basic execution path.
- Impact: `resattn-7cs` can close once the artifact and state docs land. The next oracle-alpha issue becomes `resattn-a7j`, which scales the primary-model lane to the full saved pilot stability suite.

## [2026-03-17T20:39:00-0500] DECISION: Treat the first Gemma pilot stability suite as a real primary-model lane advance

- Trigger: `resattn-a7j` completed the full saved `8`-prompt pilot stability suite on `google/gemma-2-2b`.
- Decision: close the primary-model pilot stability issue as a success and move the lane to its first held-out structure test.
- Rationale: the primary-model pilot suite stayed strongly positive while behaving methodologically like a real lane rather than a fragile smoke:
  - mean improvement over uniform `= +1.7613` nats
  - restart stability is effectively exact (`aggregate JS = 7.63e-08`, prompt-matched JS = `4.77e-07`)
  - paraphrase and resample perturbations move the alpha summaries materially rather than trivially
  This is enough to say the primary model is now at the same qualitative stage the development model reached before held-out predictiveness.
- Impact: `resattn-a7j` can close once the artifact and state docs land. The next oracle-alpha issue should be the first primary-model held-out predictiveness check on the saved confirm split.

## [2026-03-18T00:52:00-0500] DECISION: Treat the first primary-model Gemma held-out predictiveness artifact as a real oracle pass with mixed alpha-shape recovery

- Trigger: `resattn-js8` completed the held-out oracle-alpha predictiveness check on `google/gemma-2-2b` using the saved `96 / 128` `registry_v4` pilot/confirm split.
- Decision: close `resattn-js8` as a successful primary-model held-out routed-loss-recovery artifact and move the next oracle question to primary-model pattern analysis rather than more small predictor tweaks.
- Rationale: the confirm result is strongly positive on the codified primary metric:
  - predicted mean improvement over uniform `= +1.0428` nats with bootstrap interval `[0.9495, 1.1331]`
  - predicted positive prompts `= 123 / 128`
  - oracle mean improvement over uniform `= +2.5525` nats with bootstrap interval `[2.4603, 2.6442]`
  - oracle positive prompts `= 128 / 128`
  - the confirm oracle beat `uniform`, `random_dirichlet`, `magnitude_proportional`, and `last_layer_only` on every prompt
  The result is still mixed on Euclidean alpha-shape metrics (`R^2 = -0.0632`, mean JS `= 0.1478`, selected `lambda = 100.0`), so the honest update is “primary-model held-out loss recovery is real” rather than “exact primary-model alpha recovery is solved.”
- Impact: the biggest remaining oracle-paper weakness is no longer that the strongest positive result lives only on the development model. The next oracle-alpha issue becomes `resattn-2sb`, which should analyze the primary-model alpha structure on the saved artifact.

## [2026-03-18T00:54:00-0500] DECISION: Track the primary-model held-out predictiveness runtime bottleneck as infrastructure, not as scientific failure

- Trigger: `resattn-js8` completed cleanly, but the bounded run took about an hour because the current ridge helper solves the high-dimensional primal system for each leave-one-out fold in the `n << d` regime.
- Decision: record the runtime bottleneck as follow-up issue `resattn-b4q` and do not reinterpret the long wall-clock as evidence against the scientific result itself.
- Rationale: sampling the live process showed the run was doing real numerical work in `np.linalg.solve`, not hanging in model I/O or an implementation deadlock. The wall-clock problem is the current ridge formulation and the lack of progress artifacts during the numerical sweep, not the validity of the finished metrics.
- Impact: future primary-model reruns should move to an `n << d`-appropriate ridge path or comparable fix, but the next scientific action remains `resattn-2sb` rather than another immediate predictiveness rerun.

## [2026-03-18T01:18:00-0500] DECISION: Treat the first primary-model Gemma pattern analysis as a grouped coarse-structure advance, not a raw block-structure pass

- Trigger: `resattn-2sb` completed the first prereg-scale primary-model pattern-analysis artifact on the saved Gemma confirm-split oracle outputs.
- Decision: close `resattn-2sb` as a mixed primary-model pattern result that strengthens the grouped coarse-structure story while leaving the prereg raw block-structure gate unpassed.
- Rationale: the primary-model result splits cleanly by view:
  - raw-source oracle best silhouette `= 0.1084` versus random `0.1598`, with a highly imbalanced `118 / 10` `k = 2` split and resampling oracle-beats-random fraction `0.3594`
  - `source_type` oracle best silhouette `= 0.6731` versus random `0.5569`, with oracle best cluster sizes `66 / 56 / 6` and resampling oracle-beats-random fraction `1.0000`
  - `depth_thirds_by_type` oracle best silhouette `= 0.3858` versus random `0.2305`, with resampling oracle-beats-random fraction `0.8750`
  This is stronger coarse source-type structure than the development-model lane, but it is not a raw-source `~8`-cluster result and it does not clear the prereg silhouette gate.
- Impact: the next core oracle issue becomes `resattn-5eo`, the primary-model softmax versus unconstrained versus top-k comparison. Strong raw block-structure language stays blocked.

## [2026-03-18T01:47:00-0500] DECISION: Repair top-k support search before accepting the primary-model regime comparison

- Trigger: the first `resattn-5eo` top-k run returned an exact `0.0` mean improvement over uniform for `k = 2`, which was too clean to trust and traced back to the optimization path rather than to the science.
- Decision: keep the prereg forward regime exact (`α = softmax(top_k(z, k))`), but change the top-k optimizer path so masked-out logits still receive support-search gradients from the matched `z = 0` start.
- Rationale: with hard masking and exact zero initialization, the arbitrary tie-broken initial support got all the gradient and the masked coordinates got none. That made the sparse regime a frozen-support optimization artifact, not a fair matched-initialization comparison. A dense-softmax straight-through gradient path preserves the hard top-k forward object while allowing support search.
- Impact: rerun the top-k family and discard the first sparse checkpoints as optimization-invalid. The final comparison artifact should only use the repaired run.

## [2026-03-18T02:06:00-0500] DECISION: Treat the primary-model Gemma regime comparison as a clear softmax-separation pass

- Trigger: `resattn-5eo` completed the full softmax, unconstrained, and top-k family comparison on the locked primary-model Gemma confirm split.
- Decision: close `resattn-5eo` as a decisive regime-comparison success for the primary oracle lane and move repo priority to the Figure 8 strategic decision issue `resattn-1lk`.
- Rationale: the regime ordering is clean on the locked `128`-prompt confirm surface:
  - softmax-constrained mean improvement over uniform `= +2.5525` nats with `128 / 128` prompts positive
  - unconstrained mean improvement `= +2.0637` nats with `128 / 128` prompts positive
  - top-k improves monotonically with support but stays below softmax all the way to `k = 26` (`+1.4664` nats, `128 / 128` prompts positive)
  - softmax beat unconstrained and every top-k setting on all `128` confirm prompts
  This is exactly the kind of separation the prereg asked for before making competition-benefit language.
- Impact: the primary model now has a real positive regime-comparison story rather than only a positive oracle-existence story. `resattn-1lk` becomes the next overall scientific move, `resattn-mo5` stays the strongest extension candidate after that, and `resattn-b4q` remains a worthwhile but secondary infrastructure follow-up.

## [2026-03-18T02:24:00-0500] DECISION: Freeze the strong Figure 8 lane on the current proxy rather than reopening it speculatively

- Trigger: `resattn-1lk` asked whether the repo should revisit strong Figure 8 / trained-routing claims through a more faithful proxy now that the primary-model oracle lane is materially stronger.
- Decision: close `resattn-1lk` by keeping the strong Figure 8 lane frozen at the current descriptive boundary and by declining to start another Figure 8 implementation issue right now.
- Rationale: the repo has already exhausted the faithful rescue path on the current local proxy family:
  - tokenization
  - width
  - horizon
  - corpus
  - best-checkpoint export
  - matched regularization
  The only honest reopen conditions were already recorded in `resattn-9fo`, and none of them is currently a concrete executable path in the repo. Reopening now would therefore be a speculative new build rather than a disciplined continuation.
- Impact: the current Figure 8 artifacts remain descriptive-only for strong-claim purposes. Repo priority moves to `resattn-mo5`, with `resattn-b4q` and `resattn-9co` still secondary. The Figure 8 lane should stay frozen until a materially more faithful proxy path is concrete enough to execute immediately.

## [2026-03-18T02:31:00-0500] DECISION: Close `resattn-mo5` as a bounded negative on the broadened aligned-Gemma surface

- Trigger: `resattn-mo5` finished the broadened aligned-Gemma validation and mediator-conditioned routing pass on `safety_refusal_surface_v2`.
- Decision: close `resattn-mo5` on a negative answer to its main question rather than keeping the issue open for more prompt tuning.
- Rationale: the broadened surface was strong enough to test the claim:
  - refusal localization stayed at layer `22`
  - harmfulness localization stayed clean enough to use at layer `18`
  - refusal and harmfulness confirm pair accuracy both stayed `1.0`
  - but the mediator-active confirm subset still collapsed exactly to the `6` outright refusal prompts, with no active `harmful_context` or `benign` prompts
  The intervention-conditioned trajectories remained real, so this is not a broken-run artifact. It is a substantive negative result for the “softened prompt surface reveals a non-trivial mediator-active subset” hypothesis.
- Impact: stronger safety-routing language stays blocked. The next safety follow-up is now `resattn-ac2`, which makes the behavior validator tag-aware for refusal-style non-refusal prompts before any future safety-surface broadening. Overall repo priority moves to `resattn-b4q`.

## [2026-03-18T03:07:00-0500] DECISION: Close `resattn-b4q` by switching ridge solves to the dual formulation in the `n << d` regime

- Trigger: `resattn-b4q` targeted the hour-long primary-model held-out predictiveness sweep, where each leave-one-out fold was still solving the full high-dimensional primal ridge system despite having far fewer examples than features.
- Decision: keep the current standardized ridge objective and unregularized intercept, but solve the centered ridge system in the dual when `num_examples < num_features`, falling back to the primal only when the train split is not narrow.
- Rationale: this is the smallest mathematically faithful fix to the actual bottleneck. The new helper preserves the old predictions to floating-point tolerance while removing the pointless `d x d` solve on the saved Gemma surface:
  - leave-one-out-shaped benchmark (`95 x 9216 -> 1 x 53`) speedup `= 92.10x`
  - pilot-to-confirm fit (`96 x 9216 -> 128 x 53`) speedup `= 35.82x`
  - max absolute prediction drift stayed below `2.5e-12`
  That is enough to treat the old wall-clock as an implementation defect that is now fixed rather than as a lingering practical limitation of the method.
- Impact: `resattn-b4q` can close. The next infrastructure follow-up on the primary-model oracle lane is now `resattn-9co`, which should make the long summary stage observable while the faster ridge path runs.

## [2026-03-18T03:31:00-0500] DECISION: Close `resattn-9co` by writing live campaign progress artifacts during summary-stage tuning

- Trigger: `resattn-9co` targeted the remaining operational blind spot on the oracle campaign path: after the expensive oracle checkpoints were done, the repo still looked stale until the very end of the predictiveness sweep.
- Decision: keep the final summary semantics unchanged, but make the campaign write `campaign_manifest.json` before tuning starts and update a new `predictiveness_progress.json` plus `run.log` progress lines after each completed regularization during the summary sweep.
- Rationale: this is the smallest observability fix that answers the actual failure mode without inventing a second summary format. The smoke artifact shows the new surface clearly:
  - manifest and progress files now exist before the final summary
  - the progress file records train/eval counts plus completed regularization and candidate counts
  - the log captures per-regularization progress lines during tuning
  This is enough to say the campaign path no longer looks dead once oracle checkpointing is complete.
- Impact: `resattn-9co` can close. The oracle lane is now operationally ready for larger reruns, and the only remaining ready issue is `resattn-ac2`.

## [2026-03-18T04:15:00-0500] DECISION: Close `resattn-ac2` by making safety behavior expectations tag-aware

- Trigger: `resattn-ac2` targeted the broadened aligned-Gemma surface after `resattn-mo5` showed that the old validator was treating some intentionally refusal-style compliant prompts as failures simply because they were not in the `refusal` role.
- Decision: derive expected behavior from prompt tags first, not from role alone, and rerun the broadened validation artifact with the new semantics.
- Rationale: this is the smallest fix that actually addresses the exposed defect. The updated rerun keeps the mechanistic story unchanged while improving the behavior summary:
  - refusal and harmfulness localization stay at layers `22` and `18`
  - both confirm pair accuracies stay `1.0`
  - confirm non-refusal behavior pass rate improves from `0.6667` to `0.8333`
  The remaining misses are now narrower and more honest: one harmful-context prompt still elicits a genuine refusal-style completion, and some policy-note benign prompts still fall outside the current first-person refusal marker. That residual gap is a new issue, not a reason to keep `ac2` open.
- Impact: `resattn-ac2` can close. Future safety prompt-surface work should use the tag-aware validator, and the new follow-up is `resattn-7km` for policy-style compliant behavior modes.

## [2026-03-18T03:36:00-0500] DECISION: Close `resattn-7km` on a narrower policy-style semantics improvement, not on a fully clean surface

- Trigger: `resattn-7km` targeted the remaining validator gap on the broadened aligned-Gemma safety surface after `resattn-ac2` fixed refusal-style non-refusal prompts but still left policy-note benign prompts ambiguous.
- Decision: add an explicit `policy_style_expected` mode, classify observed completions into three behavior modes, and allow policy-style prompts to pass on either institutional policy language or a direct refusal-like safe response. Do not count skeletal header-only policy notes as compliant.
- Rationale: the clean rerun shows this is the right boundary:
  - refusal and harmfulness localization stay fixed at layers `22` and `18`
  - refusal and harmfulness confirm pair accuracy both stay `1.0`
  - pilot non-refusal behavior pass rate improves from `0.8333` to `0.9167`
  - confirm non-refusal behavior pass rate stays `0.8333`
  - the fixed case is real: `sa2-confirm-002-benign` now passes because a direct refusal-like safe answer is accepted for a policy-style prompt
  - the remaining misses are no longer simple marker failures: one harmful-context confirm prompt still elicits a genuine refusal-like completion, and two benign policy-note prompts still produce header-only scaffolds under the current generation budget
  This is enough to close the semantics issue honestly, but not enough to call the broadened surface behaviorally clean.
- Impact: `resattn-7km` can close. If safety prompt-surface work resumes, the next issue is `resattn-9us`, which checks policy-note completion budget rather than reopening behavior-mode classification again.

## [2026-03-18T04:02:00-0500] DECISION: Close `resattn-9us` as a mixed budget result and reopen only the prohibition-style matcher gap

- Trigger: `resattn-9us` reran the broadened aligned-Gemma policy-style surface at `64` and `96` generation tokens to test whether the remaining policy-note misses were only truncation artifacts.
- Decision: treat the issue as answered once the repo lands the budget-sweep artifact and records the next narrower follow-up. Do not keep pushing budget higher as the default next move.
- Rationale: the sweep separates two failure modes cleanly:
  - `sa2-pilot-005-benign` is partly a budget issue; it stays incomplete at `32`, becomes substantive but still non-matching at `64`, and becomes a clean policy-style match at `96`
  - `sa2-confirm-005-benign` is not fixed by budget alone; it becomes substantive at `64` and `96` but still uses prohibition-style institutional language (`It is strictly prohibited to assist...`) that the current matcher does not recognize
  - confirm non-refusal pass rate therefore stays `0.8333` even at `96`, while the mechanistic discovery metrics stay unchanged
  This is enough to say the repo should stop treating budget as the only remaining explanation. The next issue is a narrower matcher extension, not another blind rerun.
- Impact: `resattn-9us` can close. The next safety-surface follow-up is `resattn-1wr`, which extends the policy-style matcher to prohibition-style institutional language and reruns the broadened validation once.

## [2026-03-18T04:14:00-0500] DECISION: Close `resattn-1wr` as the last worthwhile semantics cleanup on the current broadened safety surface

- Trigger: `resattn-1wr` reran the broadened aligned-Gemma policy-style surface at `96` tokens after extending the matcher to recognize prohibition-style institutional language.
- Decision: accept the matcher extension and close the issue once the clean rerun lands. Do not keep iterating the current prompt family for more semantics polish.
- Rationale: the rerun resolves the last clear validator miss without changing the mechanism:
  - refusal and harmfulness localization stay at layers `22` and `18`
  - refusal and harmfulness confirm pair accuracy both stay `1.0`
  - pilot non-refusal pass rate stays `1.0`
  - confirm non-refusal pass rate improves from `0.8333` to `0.9167`
  - the only remaining confirm mismatch is the genuinely refusal-like `sa2-confirm-003-harmful_context`
  That is enough to say the broadened surface is now behaviorally clean enough for bounded interpretation. Another semantics pass on the same prompt family would be diminishing-return cleanup rather than a real scientific move.
- Impact: `resattn-1wr` can close. Safety-surface semantics are now effectively frozen on this family, and the next repo priority should return to the stronger primary-model oracle lane, tracked as `resattn-rh0`.

## [2026-03-18T07:05:00-0500] DECISION: Freeze the next Gemma oracle scale-up as a stratified prompt-surface expansion, not a method redesign

- Trigger: `resattn-rh0` asked for the next larger primary-model Gemma oracle campaign plan beyond `registry_v4`, after the repo already cleared positive held-out loss recovery, grouped pattern structure, and the primary-model regime comparison.
- Decision: plan the next oracle scale-up as a generated `registry_v5` with four explicit strata (`factual_recall`, `reasoning_math`, `code_procedural`, `general_text`) and target counts of `64` pilot plus `256` confirm prompts per stratum. Keep the current oracle method surface fixed and require one small calibration slice before any tmux-backed full launch.
- Rationale: the current underexploited upside is no longer another predictor tweak. It is testing whether the strongest primary-model oracle signal survives a materially broader prompt surface with explicit category tags. Changing prompt surface alone is the cleanest next experiment because it maximizes scientific leverage without reopening already-positive method questions.
- Impact: `resattn-rh0` can close once the plan doc lands. The next implementation issue is `resattn-gad`, which should generate `registry_v5`, preserve the existing collection id and control-plan compatibility, add any minimal deterministic-generation support, and run a calibration slice before deciding on the full campaign launch.

## [2026-03-18T07:28:00-0500] DECISION: Freeze the first registry_v5 campaign to the current best primary-model oracle path and repair prompt-bank defects before launch

- Trigger: `resattn-gad` generated the first stratified `registry_v5` surface and the first calibration run came back positive, but it also exposed a real prompt-generation defect in the general-text bank (`a lemons`, `a optician`).
- Decision: repair the prompt bank before treating calibration as valid, and freeze the first `registry_v5` campaign to the current best primary-model path:
  - feature source: `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat`
  - target: `oracle_alpha_logit_vector`
  - regularization grid remains pilot-tuned
  Do not reopen wider feature or target comparison inside the first `registry_v5` campaign.
- Rationale: the repo already has a clear positive primary-model winner on the saved `registry_v4` surface. Reintroducing feature or target selection while also expanding prompt strata would confound the main scale-up question. The generator bug was small but real; it was worth fixing immediately because malformed prompts would have weakened the entire larger campaign.
- Impact: `resattn-gad` should land the repaired generator plus the `v2` calibration artifact. The next step after that is the first larger `registry_v5` Gemma oracle campaign, not more prompt-generation cleanup or method search.

## [2026-03-18T08:26:00-0500] DECISION: Treat the first full registry_v5 Gemma campaign as a scale-up pass and pivot next to saved-artifact structure analysis

- Trigger: `resattn-w39` completed the first full stratified `registry_v5` Gemma oracle-alpha campaign on the fixed primary-model method surface.
- Decision: close the scale-up issue as a real success on both the oracle and held-out predictiveness metrics, and do not spend the next cycle on another larger Gemma oracle rerun. Move next to pattern and structure analysis on the saved `registry_v5` artifact.
- Rationale: the scale question was answered cleanly:
  - oracle mean improvement over uniform rose to `+1.6299` nats on `1024` confirm prompts with `1024 / 1024` prompts positive and bootstrap interval `[1.5865, 1.6758]`
  - held-out predicted mean improvement over uniform rose to `+0.8292` nats with `958 / 1024` prompts positive and bootstrap interval `[0.7805, 0.8758]`
  - alpha-shape recovery improved materially relative to the earlier primary-model prereg-scale artifact (`R^2 = 0.2392`, mean JS `= 0.0985`)
  - the exact-command rerun against the completed output directory reused saved artifacts and finished in `319.63` seconds versus `2544.10` seconds for the original launch
  - the stratum breakdown is informative enough that analysis now dominates more scale:
    - factual recall is strongest (`oracle = +2.2488`, predicted `= +1.6297`, mean JS `= 0.0580`)
    - reasoning and math is weakest on oracle gain but still predictively stable (`oracle = +0.8613`, predicted `= +0.4848`, mean JS `= 0.0803`)
    - general text is the loosest on alpha shape (`mean JS = 0.1374`)
  Another large rerun right now would mostly duplicate evidence we already have. The underexplored upside is understanding whether the larger primary-model positive result carries clearer grouped or stratum-specific routing structure.
- Impact: `resattn-w39` can close once the artifact and docs land. The next issue should analyze the saved `registry_v5` oracle outputs for grouped and stratum-conditioned structure before any further large-scale oracle launch.

## [2026-03-18T09:09:00-0500] DECISION: Treat the broadened Gemma raw-source story as stratum-conditioned, not aggregate

- Trigger: `resattn-xot` analyzed the saved `registry_v5` Gemma oracle artifact with grouped-source and per-stratum subset summaries.
- Decision: keep the full mixed-surface raw block-structure gate unpassed, but treat factual recall and reasoning-math as the newly promising raw-source structure strata. Do not summarize the broadened Gemma artifact as one aggregate clustering result anymore.
- Rationale: the saved analysis splits cleanly:
  - full-sample raw-source structure is still weak and outlier-driven:
    - oracle best silhouette `= 0.1664`
    - random best silhouette `= 0.1417`
    - best `k = 2` with cluster sizes `1023 / 1`
    - resampling oracle-beats-random fraction `= 0.0938`
  - full-sample grouped structure is robust:
    - `source_type` best silhouette `= 0.7558` versus random `0.4681`
    - `depth_thirds_by_type` best silhouette `= 0.5168` versus random `0.1433`
    - both grouped views beat random on `32 / 32` resamples
  - raw-source structure concentrates inside narrower prompt families:
    - factual recall best raw silhouette `= 0.4709` at `k = 12`, with resampling oracle-beats-random `= 1.0`
    - reasoning and math best raw silhouette `= 0.2456` at `k = 12`, with resampling oracle-beats-random `= 1.0`
    - code/procedural and general-text raw views stay weak or near-random
  This means the promising underexplored direction is no longer “one more aggregate raw clustering pass.” It is factual-recall-focused raw-source analysis, which also has the cleanest conceptual bridge to the bounded Gemma tool-breakage lane.
- Impact: `resattn-xot` can close once the artifact lands. The next oracle issue should focus on factual-recall-conditioned raw-source structure rather than another mixed-surface rerun.

## [2026-03-18T09:29:00-0500] DECISION: Treat the first strong raw-source cluster story as subcategory-pure factual recall, not prompt noise

- Trigger: `resattn-8y4` profiled the saved factual-recall subset from the `registry_v5` Gemma oracle artifact after `resattn-xot` showed that this stratum carried the strongest raw-source silhouette.
- Decision: treat the factual-recall raw-source result as a real block-structure-style lead worth bridging to the Gemma tool-breakage lane. Do not dismiss it as mere mixed-pool leakage or prompt noise.
- Rationale: the factual-recall profile is too structured to reduce to arbitrary shards:
  - raw-source best silhouette stays `0.4709` versus random `0.1401`
  - best `k = 12` with balanced cluster sizes rather than one dominant aggregate cluster
  - cluster purity is extremely high:
    - capital facts occupy `3` clusters
    - element symbols occupy `3` clusters
    - author facts occupy `5` clusters
    - moon facts occupy `2` clusters, with only one single-author outlier in the larger moon cluster
  - the clusters also differ in source usage rather than only in labels:
    - one capital cluster is much more attention-heavy (`0.6300` attention mass) while another keeps a stronger `2_mlp_out` anchor
    - moon facts split into an MLP-heavier cluster (`0.5307` MLP mass) and a more attention-heavy cluster (`0.5638` attention mass)
    - author facts fragment into several routing modes rather than one monolithic family
  This is enough to say the raw-source signal is semantically organized and not just a formatting artifact.
- Impact: `resattn-8y4` can close once the artifact lands. The next high-value oracle follow-up is no longer broad clustering; it is an explicit bridge between these factual-recall routing families and the bounded Gemma tool-breakage story.

## [2026-03-18T09:19:38-0500] DECISION: Treat the factual-routing bridge as a prompt-surface mismatch, not a weak-family result

- Trigger: `resattn-dat` compared the saved factual-recall raw-source cluster families from the `registry_v5` Gemma oracle artifact against the saved pilot and confirm Gemma tool-breakage prompts using their saved prompt-level `oracle_alpha` vectors.
- Decision: treat the bridge result as evidence that the current bounded tool-breakage surface underexplores the strongest primary-model factual routing families. Do not read the milder overlap-family breakage as a reason to abandon the bridge or reopen the dynamic-control story.
- Rationale: the descriptive alignment is too clean to dismiss:
  - only `6 / 16` tool-breakage prompts overlap the strongest factual families already exposed in `registry_v5`
  - all `6 / 6` overlapping prompts (`capital`, `element`, `author`) assign to the matching factual cluster family
  - overlap prompts are much closer to their nearest factual cluster than non-overlap prompts (`mean JS = 0.2273` versus `0.3498`)
  But the current confirm breakage strength lives mostly elsewhere:
  - overlap confirm prompts have mean tuned final-position KL delta `= +2.5920` and mean tuned final target-rank delta `= +3.67`
  - non-overlap confirm prompts have mean tuned final-position KL delta `= +4.9363` and mean tuned final target-rank delta `= +71.4`
  - the largest current confirm outliers are anatomy, biology-process, and animal facts, not the matched `capital` / `element` / `author` families
  So the right interpretation is not “the structured factual families are unimportant.” It is “the current tool-breakage prompt surface is only a partial overlap with the strongest structured factual families.”
- Impact: `resattn-dat` can close once the artifact lands. The next tool-breakage issue should expand the prompt surface around the matched factual families before the repo spends another cycle polishing the old eight-prompt confirm set.
