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
