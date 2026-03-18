# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: `resattn-9us` is the current bounded safety-surface sweep; the repo still has no ready oracle-infrastructure work, and the next narrow safety follow-up after this is `resattn-1wr`
- Experimental status: the primary `google/gemma-2-2b` oracle lane remains the strongest positive result path, while the broadened aligned-Gemma safety surface now has a clearer residual boundary. The policy-budget sweep showed that more tokens help but do not fully fix the surface: pilot non-refusal pass rises to `1.0` at `96`, confirm stays `0.8333`, and the remaining confirm miss is now clearly prohibition-style institutional wording rather than a header-only truncation artifact.
- Critical reminder: if I continue after this, it should be because `resattn-1wr` is worth one more narrow semantics pass or because the repo is ready to pivot back to a stronger primary-model oracle-scale issue.
