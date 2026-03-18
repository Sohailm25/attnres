# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: `resattn-1wr` is the last narrow safety-surface semantics pass; after it lands, the next meaningful move is `resattn-rh0` on the stronger primary-model oracle lane
- Experimental status: the broadened aligned-Gemma safety surface is now largely behaviorally clean for this prompt family. The prohibition-style rerun keeps the mechanistic results unchanged and moves confirm non-refusal pass rate to `0.9167`, leaving only one genuinely refusal-like harmful-context mismatch. That means the safety semantics cleanup is basically done, while the biggest remaining upside is still on the primary `google/gemma-2-2b` oracle path.
- Critical reminder: if I continue after this, it should be because I am deliberately pivoting to `resattn-rh0`, not because the current safety surface still has obvious hygiene debt.
