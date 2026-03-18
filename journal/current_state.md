# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: `resattn-7km` is now the last landed safety-semantics cleanup; the repo has no remaining ready oracle-infrastructure work, and the next optional safety follow-up is `resattn-9us`
- Experimental status: the primary `google/gemma-2-2b` oracle lane is operationally ready for larger reruns, and the broadened aligned-Gemma safety surface now has both tag-aware and policy-style-aware validator semantics. The clean policy-style rerun improved pilot non-refusal pass rate from `0.8333` to `0.9167` while keeping confirm at `0.8333` and leaving the mechanistic discovery results unchanged. The remaining misses are now one genuine refusal-like harmful-context completion plus two header-only policy-note benign outputs, so the next safety question is prompt completeness rather than behavior-mode classification.
- Critical reminder: if I continue after this, it should be because a new scientific lane is chosen or because `resattn-9us` is deliberately prioritized, not because the repo still has obvious unfinished cleanup.
