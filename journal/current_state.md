# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: `resattn-mo5` is now green and the next overall scientific priority is `resattn-b4q`, the `n << d` ridge bottleneck fix on the primary-model oracle path
- Experimental status: the primary `google/gemma-2-2b` spine now has the strongest unfrozen story in the repo: reconstruction, bounded oracle smoke, pilot stability, held-out predictiveness, grouped pattern analysis, and a decisive prereg routing-regime comparison where softmax beat unconstrained and every top-k setting on all `128` confirm prompts. The strategic Figure 8 lane is now explicitly frozen at the descriptive boundary. The broadened aligned-Gemma safety follow-up also completed and answered its main question negatively: `safety_refusal_surface_v2` kept refusal and harmfulness direction discovery clean, but the mediator-active subset still collapsed exactly onto outright refusal prompts even after adding refusal-style non-refusal prompts. That means the safety lane now has stronger negative evidence about the current mediator's scope, plus a new validator-semantics issue (`resattn-ac2`) for any future prompt-surface broadening.
- Critical reminder: the best next move is now a contained improvement on the strongest live lane, not more prompt gardening on frozen or negative lanes. Keep the strong Figure 8 lane frozen, keep the Gemma tool-breakage claim boundary frozen, and only revisit safety-surface expansion after the validator semantics are made tag-aware.
