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
