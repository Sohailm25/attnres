ABOUTME: Audits the narrowed v5 donor-control geometry on saved Gemma tool-breakage artifacts.
ABOUTME: Determines whether the prompt-permuted versus within-family collapse is mainly ordering artifact or a real donor-pairing limitation.

# Gemma-2 Donor Geometry Audit v5

## Motivation

`resattn-oi7` follows the narrowed `v5` donor-arm result. The bridge is now
well aligned with the strongest factual route modes, so the remaining honest
question is narrower:

- is the partial collapse between `prompt_permuted_alpha` and
  `within_family_permuted_alpha` mostly a confirm-ordering artifact
- do author modes `6 / 11 / 12` really drive the negative within-family read
- should the repo freeze this lane, remap donors, or do nothing

## Methods

- Model: `google/gemma-2-2b`
- Source artifact:
  `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-donor-arms-v5/profile.json`
- Surface: `tool_breakage_factual_recall_v5` confirm split (`20` prompts)
- Families:
  - `subcategory_capital_fact`
  - `subcategory_element_symbol`
  - `subcategory_author_fact`
- Procedure:
  - compare `prompt_permuted_alpha` and `within_family_permuted_alpha` donor
    source prompt ids at the prompt level
  - compare the same two arms at the route-mode aggregate level
  - decompose the negative `within_family_permuted_alpha` aggregate by family
    and author route mode
  - use the saved donor-selection code path in `validation/tool_breakage.py` to
    interpret any collapse structurally rather than heuristically

Machine-readable artifact:

- `results/tool_breakage/20260318-gemma2-tool-breakage-donor-geometry-audit-v5.json`

## Results

### The collapse is mostly a confirm-ordering artifact

The donor-selection logic makes the structural issue explicit:

- `prompt_permuted_alpha` uses the next prompt in global confirm order
- `within_family_permuted_alpha` uses the next prompt inside the same family
- the `v5` confirm prompts are grouped by family

That means the two arms are identical except at family-block endpoints. The
saved artifact confirms exactly that:

- identical donor source prompt on `17 / 20` prompts (`0.85`)
- identical route-mode mean tuned-KL delta on `7 / 10` targeted modes (`0.70`)
- the only differing prompts are:
  - `tb5-confirm-006` (`route_mode_capital_cluster_4`)
  - `tb5-confirm-012` (`route_mode_element_cluster_9`)
  - `tb5-confirm-020` (`route_mode_author_cluster_12`)
- the only differing route modes are therefore the family-block endpoint modes:
  - `route_mode_capital_cluster_4`
  - `route_mode_element_cluster_9`
  - `route_mode_author_cluster_12`

So the earlier partial collapse was real, but it was not mysterious. It is a
direct consequence of the locked confirm ordering.

### The ordering artifact explains most of the pooled negative sum, but not all of it

The overall `within_family_permuted_alpha` mean tuned-KL delta is:

- routed minus `within_family_permuted_alpha` mean tuned KL `= -0.1021`

Decomposed by prompt type:

- the `3` differing family-endpoint prompts contribute `82.9%` of the total
  negative within-family sum
- the `17` identical-donor prompts contribute only `17.1%`
- mean within-family delta on differing prompts: `-0.5642`
- mean within-family delta on identical-donor prompts: `-0.0205`

So the pooled negative aggregate is mostly driven by the family-endpoint
artifact. But it does not disappear completely once that artifact is isolated.

### The remaining real limitation is concentrated in author modes 6 and 11

The family contribution decomposition is decisive:

- authors contribute `-2.7318` to the within-family sum
- capitals contribute `+0.0024`
- elements contribute `+0.6874`

So the author family alone more than accounts for the entire pooled negative
result.

Inside authors, the important split is not “cluster 12 blocks everything.” It
is narrower:

- `route_mode_author_cluster_6`:
  - within-family mean tuned-KL delta `= -1.9465`
  - prompt-permuted mean tuned-KL delta `= -1.9465`
  - contribution sum `= -3.8931`
- `route_mode_author_cluster_11`:
  - within-family mean tuned-KL delta `= -0.6423`
  - prompt-permuted mean tuned-KL delta `= -0.6423`
  - contribution sum `= -1.2846`
- `route_mode_author_cluster_12`:
  - within-family mean tuned-KL delta `= -0.0519`
  - prompt-permuted mean tuned-KL delta `= -0.9886`
  - contribution sum `= -0.1038`
- `route_mode_author_cluster_7` offsets the others:
  - within-family mean tuned-KL delta `= +1.2749`
  - contribution sum `= +2.5497`

This means:

- author modes `6` and `11` remain negative even where
  `prompt_permuted_alpha` and `within_family_permuted_alpha` are exactly the
  same donor mapping
- author mode `12` matters much more as the family-endpoint ordering artifact
  carrier than as the main within-family negative contributor

So the ordering artifact is real, but it is not the whole story.

## Interpretation

- The partial `prompt_permuted` versus `within_family` collapse is mostly a
  confirm-ordering artifact on `v5`.
- But the negative donor-arm result is not reducible to that artifact alone.
- The real residual limitation is author-family donor pairing, especially
  route modes `6` and `11`.
- A donor remap could change the pooled aggregate by removing the three
  family-endpoint mismatches, but it would not cleanly rescue the author-family
  signal that stays negative under identical donor assignments.

The truthful recommendation is therefore:

- freeze the current donor-arm claim boundary
- do not launch another donor-remap rerun by inertia
- keep the lane at:
  - positive routed-versus-original `v5` baseline
  - positive routed-versus-fixed-alpha control
  - mixed donor-arm result with a mostly ordering-driven collapse plus a real
    author-family donor-pairing limitation

## Limitations

- This is a saved-artifact audit, not a new intervention run.
- The `v5` confirm surface is still only `20` prompts, so route-mode-level
  decomposition remains brittle even when it is structurally informative.
- A future donor-remap design could still be worth exploring if the paper
  later needs a stronger same-model donor-arm claim, but that is no longer the
  default next move.

## Next Steps

- Close `resattn-oi7`.
- Freeze the main tool-breakage donor-arm boundary rather than redesigning the
  donor controls immediately.
- Keep `resattn-a1w` as the bounded moon-family sidecar if this lane resumes.
- Keep the repo centered on the stronger primary-model oracle and regime
  results instead of spending another cycle on a mixed supporting lane.
