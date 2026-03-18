ABOUTME: Audits the moon-family miss on the one-token v4 tool-breakage surface using saved artifacts only.
ABOUTME: Separates prompt-style drift, donor-pair mismatch, and genuine-null interpretations before any moon redesign.

# Gemma-2 Moon Prompt-Style Audit v4

## Motivation

`resattn-a1w` follows the saved `v4` family profile. That profile already
showed that moons, not authors, were the real blocker on the family-balanced
one-token surface. But it did not yet say why.

The narrow question here is:

- are moon failures mostly a prompt-style problem
- are they mostly a donor-pair mismatch problem
- or do they show a genuine null for prompt-specific same-model breakage

## Methods

- Model: `google/gemma-2-2b`
- Family profile source:
  `results/tool_breakage/20260318-gemma2-tool-breakage-family-profile-v4.json`
- Prompt registry source: `prompts/registry_v5.yaml`
- Collection: `tool_breakage_factual_recall_v4`
- Family: `subcategory_moon_fact`
- Procedure:
  - compare moon pilot versus confirm prompt wording
  - split confirm moon prompts into direct descriptor prompts versus
    mythological `named after` prompts
  - inspect prompt-level donor sources and routed-minus-arm tuned-KL deltas for
    all moon prompts
  - isolate whether the moon `prompt_permuted_alpha` gap versus
    `within_family_permuted_alpha` is broad or driven by one boundary donor
    mismatch
  - inspect the subcategories used by the moon cross-family donor arm

Machine-readable artifact:

- `results/tool_breakage/20260318-gemma2-tool-breakage-moon-style-audit-v4.json`

## Results

### The moon confirm surface drifted in style relative to the moon pilot

The moon pilot is entirely direct descriptor style:

- pilot descriptor prompts: `4`
- pilot `named after` prompts: `0`

The moon confirm surface is mostly a different style:

- confirm descriptor prompts: `2`
- confirm `named after` prompts: `6`

So the saved `v4` moon family is not a clean pilot-to-confirm continuation of
one surface. It quietly changes from direct astronomical descriptors to mostly
mythological clue prompts.

### The prompt-permuted moon failure is almost entirely one donor mismatch

The family-level moon means from the saved `v4` profile are:

- routed minus `within_family_permuted_alpha` mean tuned KL: `+0.0296`
- routed minus `prompt_permuted_alpha` mean tuned KL: `-0.2670`

That gap is not a broad moon-family effect. It is almost entirely one prompt:

- only `tb4-confirm-032` differs between `prompt_permuted_alpha` and
  `within_family_permuted_alpha`
- `tb4-confirm-032`:
  - prompt: `Saturn's moon named after the Titan who held up the sky is`
  - target: `Atlas`
  - prompt-permuted donor: `tb4-confirm-001` (`Canberra`, capital fact)
  - within-family donor: `tb4-confirm-025` (`Europa`, moon fact)
  - prompt-versus-within delta gap: `-2.3728`

That single boundary wrap exactly accounts for the family-level
`prompt_permuted` versus `within_family` separation on moons.

So the moon prompt-permuted failure is mainly donor-pair mismatch, not a broad
family null.

### The cross-family moon failure is also pure donor geometry

All moon cross-family donors in the saved `v4` artifact are capitals:

- cross-family donor subcategory for moon prompts:
  - `subcategory_capital_fact = 8 / 8`

And every moon cross-family prompt is negative on the primary metric. The worst
saved outliers are all moon-to-capital pairings:

- `tb4-confirm-029` (`Ariel`) vs `tb4-confirm-005` (`Lima`): `-3.6003`
- `tb4-confirm-025` (`Europa`) vs `tb4-confirm-001` (`Canberra`): `-3.0117`
- `tb4-confirm-028` (`Miranda`) vs `tb4-confirm-004` (`Nairobi`): `-2.6972`

So the cross-family moon failure should also be read as donor-pair mismatch,
not as evidence that moons are globally inert.

### The remaining weakness is the confirm-only `named after` prompt style

The within-family split by style is the useful residual signal:

- descriptor confirm prompts (`2` prompts):
  - mean within-family tuned-KL delta `= +0.2831`
  - mean prompt-permuted tuned-KL delta `= +0.2831`
  - mean pilot-mean tuned-KL delta `= -0.8186`
- `named after` confirm prompts (`6` prompts):
  - mean within-family tuned-KL delta `= -0.0549`
  - mean prompt-permuted tuned-KL delta `= -0.4504`
  - mean pilot-mean tuned-KL delta `= -1.2691`

This is the clearest prompt-style finding in the saved moon family:

- the direct descriptor prompts are still modestly positive on the within-family
  arm
- the confirm-only `named after` prompts are already slightly negative even on
  the within-family arm
- both styles remain negative against the pilot-mean arm, which says the
  family-balanced `v4` moon slice is not ready for a stronger prompt-specific
  same-model claim

## Interpretation

- The moon-family miss is not best described as a clean genuine null.
- The negative `prompt_permuted_alpha` read is mostly one boundary donor
  mismatch.
- The negative `cross_family_permuted_alpha` read is capital-donor mismatch on
  every moon prompt.
- The residual real weakness is prompt-surface design:
  - the confirm moon prompts drift heavily into a `named after` style that is
    absent from the moon pilot
  - that style is weaker than the descriptor style even under within-family
    donors

So the truthful recommendation is:

- leave the one-token tool-breakage claim frozen at its family-conditioned
  boundary
- do not run a moon donor-remap rerun by default
- if the moon sidecar is ever reopened, do a small moon-prompt rewrite first so
  pilot and confirm share one style, then decide whether any donor remap is
  worth testing

## Limitations

- This is saved-artifact analysis, not a new confirmatory rerun.
- The style split is small (`2` descriptor versus `6` `named after` prompts),
  so it is diagnostic rather than final.
- The result only applies to the family-balanced `v4` moon sidecar, not to the
  narrower `v5` main bridge, which already excludes moons.

## Next Steps

- Close `resattn-a1w`.
- Keep the moon-family question frozen on `v4` rather than reopening the main
  tool-breakage lane.
- If moons ever become strategically necessary again, prefer a moon-only prompt
  rewrite before any donor-pair remap.
