# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: `resattn-ac2` is now green; the repo has no remaining ready oracle-infrastructure work, and the next optional safety follow-up is `resattn-7km`
- Experimental status: the primary `google/gemma-2-2b` oracle lane is operationally ready for larger reruns, and the broadened aligned-Gemma safety surface now has the correct tag-aware validator semantics. The updated rerun fixed the old refusal-style false negatives on the confirm split, moving non-refusal pass rate from `0.6667` to `0.8333` while keeping the mechanistic discovery results unchanged. The remaining semantic gap is narrower and explicitly tracked as a policy-style follow-up rather than being conflated with the broader `mo5` result.
- Critical reminder: if I continue after this, it should be because a new scientific lane is chosen or because `resattn-7km` is deliberately prioritized, not because the repo still has obvious unfinished cleanup.
