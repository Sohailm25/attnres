# ABOUTME: Tracks the repo state relative to the research plan for quick session recovery.
# ABOUTME: Overwritten as the experiment state changes so future sessions can restart without rereading the full repo.

# Journal Current State

- Date: 2026-03-18
- Repo: standalone and initialized
- Branch: `wip/resattn-4xn`
- Focus: `resattn-4xn` has now frozen the next Gemma tool-breakage follow-up as a one-token matched-family prompt-surface redesign rather than a metric rescue on the current `v3` surface.
- Experimental status: the lane remains frozen at the family-conditioned `v3` boundary. The next legitimate reopening is `resattn-t0p`: build `tool_breakage_factual_recall_v4` with one-token targets under the Gemma tokenizer and rerun the pilot baseline.
- Critical reminder: keep the model, tuned-lens baseline, donor-arm controls, and KL-primary metric fixed when `t0p` lands. The point is to isolate the answer-format confound, not to mix surface and metric changes.
