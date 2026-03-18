# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: `resattn-5eo` is now green and the next overall scientific priority is `resattn-1lk`, the conservative Figure 8 freeze-or-escalate decision, with `resattn-mo5` still the strongest experimental extension issue after that
- Experimental status: the primary `google/gemma-2-2b` spine now has a real full-lane oracle story rather than only feasibility fragments: exact reconstruction, bounded oracle smoke, pilot stability, held-out predictiveness, grouped pattern analysis, and now the prereg routing-regime comparison. The new regime result is clean and important: softmax-constrained routing improved over uniform by `+2.5525` nats, unconstrained improved by `+2.0637`, top-k improved monotonically from `+0.0115` at `k = 2` to `+1.4664` at `k = 26`, and softmax beat every alternative on all `128` confirm prompts. That materially strengthens the primary-model competition story. The grouped-pattern read is still stronger than the raw block-structure read, the Gemma factual-recall tool-breakage lane remains a bounded mixed result because the prompt-permuted dynamic control is at least as damaging as the prompt-matched routed trace, the widened compact-subword `wikitext-103` Figure 8 lane is still operationally real but frozen for strong claims on the current proxy, and the aligned-Gemma safety lane still has a role-collapsed mediator partition despite real intervention-conditioned trajectory movement.
- Critical reminder: the primary model now supports both oracle loss recovery and regime separation, so the next repo move should reduce paper-shape ambiguity rather than re-prove existing oracle facts. Keep the Figure 8 lane in conservative freeze-or-escalate territory, keep the safety lane honest about the role-collapsed prompt surface, and treat `resattn-b4q` as useful infrastructure rather than as the next scientific headline.
