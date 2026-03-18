ABOUTME: Summarizes the locked one-token matched-family Gemma tool-breakage confirm baseline on the redesigned v4 surface.
ABOUTME: Tests whether the one-token surface stays positive enough to justify rerunning donor-arm controls.

# Gemma-2 One-Token Matched-Family Confirm v4

## Motivation

`resattn-czd` is the confirmatory follow-up to the positive `v4` pilot. The
pilot showed that once every answer was constrained to a single Gemma token,
every family was positive on mean tuned KL under routing. The next question was
whether that healthier family read survives on the locked confirm split strongly
enough to justify returning to donor-arm controls on the redesigned surface.

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v4` confirm split (`32` prompts)
- Families:
  - `subcategory_capital_fact`
  - `subcategory_element_symbol`
  - `subcategory_author_fact`
  - `subcategory_moon_fact`
- Surface constraint: every `target_text` tokenizes to exactly one token under
  the `google/gemma-2-2b` tokenizer
- Baseline runner:
  `scripts/run_tool_breakage_factual_recall_baseline.py`
- Primary metric: tuned-lens mean KL to the final output distribution under
  routing minus the original-model baseline
- Secondary diagnostics:
  - mean final-position tuned KL under routing
  - final-position target-rank worsening
  - target-rank-range increase
- Runtime durability:
  - prompt-level checkpoints under
    `results/tool_breakage/20260318-gemma2-tool-breakage-one-token-confirm-v4/checkpoints/prompt_results`
  - exact-command rerun finished in `9.98` seconds with unchanged checkpoint
    timestamp hash

The machine-readable summary for this artifact is
`results/tool_breakage/20260318-gemma2-tool-breakage-one-token-confirm-v4/metrics.json`.

## Results

The locked `v4` confirm surface stayed strongly positive overall:

- mean tuned KL delta under routing: `+2.7121`
- mean final-position tuned KL delta under routing: `+3.0329`
- tuned final-target-rank worsening fraction: `0.5`
- tuned target-rank-range increase fraction: `0.6875`

The important family read is that all four families remained positive on mean
tuned KL under routing on the confirm split:

- capitals: `+2.9547`
- elements: `+2.9847`
- authors: `+2.0532`
- moons: `+2.8556`

Every family is `8 / 8` positive on mean tuned KL under routing on this locked
confirm surface.

That means the one-token redesign did not just help on pilot. It survives the
confirm split cleanly and removes the old “authors are the unique negative
family at baseline” read from the saved `v3` boundary.

## Interpretation

- The one-token surface is now strong enough to justify rerunning donor-arm
  controls.
- This confirm artifact still does **not** by itself reopen the stronger
  prompt-specific same-model claim, because it remains a routed-versus-original
  baseline comparison.
- But it does show that the redesigned surface is stable enough that a donor-arm
  rerun is no longer premature.

## Limitations

- This is still a baseline artifact, not a dynamic-control artifact.
- The moon prompts remain somewhat more stylized than the old `v3` “largest moon
  of X” prompts.
- The confirm result does not tell us yet whether the within-family donor-arm
  control will stay positive on this surface.

## Next Steps

- Close `resattn-czd`.
- Run `resattn-apy`: the donor-arm counterfactual on the one-token `v4` surface.
- Treat the donor-arm rerun as the real reopening test for the stronger
  prompt-specific same-model claim.
