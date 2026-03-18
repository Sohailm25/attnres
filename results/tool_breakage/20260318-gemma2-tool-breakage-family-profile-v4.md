ABOUTME: Profiles family-conditioned heterogeneity in the one-token Gemma matched-family donor-arm artifact.
ABOUTME: Preserves the prompt-level donor structure so the pooled v4 summary is not over-interpreted.

# Gemma-2 One-Token Family Profile v4

## Motivation

`resattn-8h7` exists because the pooled `tool_breakage_factual_recall_v4`
donor-arm artifact is better than `v3` but still mixed. The aggregate says the
one-token redesign repaired the within-family donor arm and almost repaired the
legacy prompt-permuted arm, yet the cross-family donor arm remains negative.
This artifact profiles the saved prompt checkpoints directly so the next step is
driven by the actual residual family structure rather than by another pooled
average.

## Methods

- Source artifact:
  `results/tool_breakage/20260318-gemma2-tool-breakage-one-token-donor-arms-v4/summary.json`
- Prompt-level inputs:
  `results/tool_breakage/20260318-gemma2-tool-breakage-one-token-donor-arms-v4/checkpoints/prompt_results/*.json`
- Prompt registry:
  `prompts/registry_v5.yaml`
- Collection: `tool_breakage_factual_recall_v4` confirm (`32` prompts)
- Families:
  - `subcategory_capital_fact`
  - `subcategory_element_symbol`
  - `subcategory_author_fact`
  - `subcategory_moon_fact`
- Analysis:
  - group prompt-level routed-versus-arm deltas by family
  - keep the same tuned-lens primary quantity as the saved counterfactual:
    mean routed-minus-arm KL to the final output distribution across layer
    traces
  - retain alpha Jensen-Shannon distance, final-position KL deltas, and
    rank-instability diagnostics for each arm
  - inspect the worst moon-family cross-family donor pairings directly from the
    saved prompt checkpoints
- Determinism:
  - rerunning the exact profile command reproduced the same JSON hash
  - the exact-command rerun completed in `3.36` seconds

The machine-readable summary is
`results/tool_breakage/20260318-gemma2-tool-breakage-family-profile-v4.json`.

## Results

The one-token `v4` surface is not one pooled story. It is three different
family regimes.

### 1. `subcategory_element_symbol` is now the only clean all-arm positive family

- routed minus `within_family_permuted_alpha` mean tuned KL: `+0.1624`
- routed minus `prompt_permuted_alpha` mean tuned KL: `+0.3549`
- routed minus `cross_family_permuted_alpha` mean tuned KL: `+0.8831`
- routed minus `pilot_mean_alpha` mean tuned KL: `+1.0578`
- within-family positive prompt fraction on mean tuned KL: `0.625`
- within-family alpha JS distance: `0.2085`

This is the clearest prompt-specific same-model signal on the one-token
surface.

### 2. `subcategory_author_fact` and `subcategory_capital_fact` are near-tie families, not clear failures

Authors:

- routed minus `within_family_permuted_alpha` mean tuned KL: `-0.0310`
- routed minus `prompt_permuted_alpha` mean tuned KL: `-0.1172`
- routed minus `cross_family_permuted_alpha` mean tuned KL: `+0.2805`
- routed minus `pilot_mean_alpha` mean tuned KL: `+0.0892`
- within-family positive prompt fraction on mean tuned KL: `0.375`
- within-family alpha JS distance: `0.2835`

Capitals:

- routed minus `within_family_permuted_alpha` mean tuned KL: `-0.0950`
- routed minus `prompt_permuted_alpha` mean tuned KL: `-0.0262`
- routed minus `cross_family_permuted_alpha` mean tuned KL: `+0.1704`
- routed minus `pilot_mean_alpha` mean tuned KL: `+1.0661`
- within-family positive prompt fraction on mean tuned KL: `0.5`
- within-family alpha JS distance: `0.2496`

The important update relative to `v3` is that authors are no longer the unique
dominant failure family. They are now closer to capitals: slightly negative on
the tighter dynamic controls, but positive on the broader cross-family and
fixed-alpha controls.

### 3. `subcategory_moon_fact` is the real blocker on the one-token surface

- routed minus `within_family_permuted_alpha` mean tuned KL: `+0.0296`
- routed minus `prompt_permuted_alpha` mean tuned KL: `-0.2670`
- routed minus `cross_family_permuted_alpha` mean tuned KL: `-2.2528`
- routed minus `pilot_mean_alpha` mean tuned KL: `-1.1565`
- within-family positive prompt fraction on mean tuned KL: `0.75`
- within-family alpha JS distance: `0.3325`

So moons are not failing in the same way authors failed on `v3`.

- The within-family donor arm is slightly positive.
- The prompt-permuted arm is negative.
- The cross-family arm is strongly negative.
- Even the fixed pilot-mean alpha arm is negative.

That makes moons the only family where the fixed-alpha objection still survives
at the family level on `v4`.

The worst cross-family moon outliers are concrete and all map to capital donors:

- `tb4-confirm-029`: `Ariel` routed minus cross-family mean tuned KL
  `= -3.6003` against donor `tb4-confirm-005` (`Lima`)
- `tb4-confirm-025`: `Europa` routed minus cross-family mean tuned KL
  `= -3.0117` against donor `tb4-confirm-001` (`Canberra`)
- `tb4-confirm-028`: `Miranda` routed minus cross-family mean tuned KL
  `= -2.6972` against donor `tb4-confirm-004` (`Nairobi`)
- `tb4-confirm-027`: `Hyperion` routed minus cross-family mean tuned KL
  `= -2.3595` against donor `tb4-confirm-003` (`Bangkok`)

## Interpretation

- The one-token redesign genuinely improved the tool-breakage lane. The old
  “authors are the unique failure family” story is no longer correct.
- The correct boundary is now family-conditioned:
  - elements are a clean positive signal
  - authors and capitals are near ties
  - moons are the real blocker
- Because moons are negative not only on the cross-family donor arm but also on
  the prompt-permuted and pilot-mean controls, the next honest move is not
  another pooled rerun. It is a narrow moon-family audit.

## Limitations

- This is saved-artifact analysis, not a new confirmatory model run.
- The profile identifies the moon-family failure sharply, but it does not by
  itself prove whether that failure is due to prompt style, donor pairing, or a
  genuine null for the stronger same-model claim.
- The family-conditioned story still depends on the current four-family `v4`
  surface rather than a larger balanced confirm set.

## Next Steps

- Close `resattn-8h7`.
- Freeze the one-token `v4` pooled claim at this family-conditioned mixed
  boundary.
- Preserve `resattn-a1w` as the only immediate tool-breakage follow-up worth
  doing: audit the moon-family prompt style before any further redesign or
  donor-pair remap.
