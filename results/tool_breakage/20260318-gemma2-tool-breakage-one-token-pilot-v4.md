ABOUTME: Summarizes the one-token matched-family Gemma tool-breakage pilot on the redesigned v4 surface.
ABOUTME: Tests whether the family-conditioned breakage story stays healthier when every answer is a single Gemma token.

# Gemma-2 One-Token Matched-Family Pilot v4

## Motivation

`resattn-t0p` is the first direct follow-up to the `v3` family-profile artifact.
`resattn-mxf` showed that the broader matched-family donor-arm result was being
dragged hardest by the author family, and `resattn-4xn` froze the next honest
reopening as a one-token matched-family surface rather than a metric rescue.

This pilot tests whether the routed-versus-original baseline still looks healthy
once every target is a single next token under the Gemma tokenizer.

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v4` pilot split (`16` prompts)
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
    `results/tool_breakage/20260318-gemma2-tool-breakage-one-token-pilot-v4/checkpoints/prompt_results`
  - exact-command rerun finished in `10.57` seconds with unchanged checkpoint
    timestamp hash

The machine-readable summary for this artifact is
`results/tool_breakage/20260318-gemma2-tool-breakage-one-token-pilot-v4/metrics.json`.

## Results

The one-token `v4` pilot stayed strongly positive overall:

- mean tuned KL delta under routing: `+2.7594`
- mean final-position tuned KL delta under routing: `+2.9217`
- tuned final-target-rank worsening fraction: `0.5`
- tuned target-rank-range increase fraction: `0.6875`

The important family read is that all four families are now positive on the
baseline tuned mean-KL metric:

- capitals: `+2.9918`
- elements: `+2.7245`
- authors: `+2.2556`
- moons: `+3.0658`

All `4 / 4` pilot prompts in every family were positive on mean tuned KL under
routing.

That is the main contrast with the saved `v3` family-profile boundary:

- on the `v3` donor-arm profile, authors were the unique family that went
  negative against the within-family, cross-family, and fixed-alpha controls
- on the `v4` routed-versus-original pilot baseline, authors are no longer the
  unique weak family and remain clearly positive (`+2.2556`)

The author-family pilot is not the strongest family, but it no longer looks
like a baseline outlier:

- author mean tuned KL delta under routing: `+2.2556`
- author mean final-position tuned KL delta under routing: `+3.0250`
- author mean final-layer final-position tuned KL delta under routing: `+4.3709`

## Interpretation

- The one-token redesign looks promising enough to justify confirmatory follow-up.
- The result does **not** yet reopen the stronger prompt-specific same-model
  claim, because this pilot only tests routed-versus-original baseline
  degradation and does not rerun donor-arm controls.
- But it does weaken the worry that the author family is intrinsically weak at
  the baseline stage. Once the answer surface is one-token throughout, authors
  no longer stand out negatively.

## Limitations

- This is a pilot-only baseline artifact, not a confirm artifact.
- The comparison to the saved `v3` family profile is not apples-to-apples on the
  control structure: `v3` is a donor-arm analysis, while this pilot is routed
  versus original only.
- Some one-token moon prompts are still more stylized than the old “largest moon
  of X” surface.

## Next Steps

- Close `resattn-t0p`.
- Run `resattn-czd`: the locked `v4` confirm baseline.
- Only revisit donor-arm controls on the one-token surface if the confirm
  baseline stays positive and the author-family improvement survives.
