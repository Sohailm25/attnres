# ABOUTME: Audits the saved Gemma `registry_v5` reasoning/math raw-source structure without another oracle run.
# ABOUTME: Decides whether reasoning/math is a better bridge lane than factual recall or a secondary structured result.

# Gemma-2 Reasoning/Math Route-Mode Audit v1

## Motivation

`resattn-v7h` exists because the saved `registry_v5` Gemma artifact already
showed that reasoning/math is the second-strongest raw-source stratum after
factual recall:

- raw-source silhouette `= 0.2456`
- random-control silhouette `= 0.1401`
- resampling oracle-beats-random `= 1.0`

That made reasoning/math the clearest underexplored positive result on the
primary model. The question here was not whether the stratum is positive at all.
It was narrower:

- are the reasoning/math modes interpretable?
- are they mainly prompt-frame artifacts?
- or are they more task-like than the already-solved factual family-plus-frame
  story?

## Methods

This is a saved-artifact audit only. No new model run was launched.

Source artifacts:

- broad oracle anchor:
  `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/oracle_eval_run.json`
- new reasoning/math raw-source profile:
  `results/block_structure/20260319-gemma2-reasoning-math-cluster-profile-v1.json`
- new reasoning/math route-mode summary:
  `results/block_structure/20260319-gemma2-reasoning-math-route-modes-v1.json`
- new deterministic frame audit:
  `results/block_structure/20260319-gemma2-reasoning-math-route-mode-frame-audit-v1.json`
- prompt surface:
  `prompts/registry_v5.yaml`

Analysis steps:

1. Reuse the saved `registry_v5` confirm artifact and profile the
   `stratum_reasoning_math` subset with the existing raw-source cluster script.
2. Reuse the existing route-mode helper at the subcategory level.
3. Derive deterministic frame labels from the fixed `registry_v5` template
   construction for each reasoning/math subcategory.
4. Compare cluster dominance by subcategory versus cluster dominance by frame,
   then inspect frame concentration inside each subcategory’s route modes.

## Results

### 1. Reasoning/math is a real raw-source result, but still weaker than factual recall

The saved reasoning/math subset retains real raw-source structure:

- best raw-source silhouette `= 0.2456`
- random-control best silhouette `= 0.1401`
- best `k = 12`
- cluster sizes:
  `48 / 35 / 33 / 31 / 31 / 16 / 16 / 15 / 13 / 10 / 7 / 1`

Subcategory support across the `k = 12` split:

- `subcategory_arithmetic_story`: `6` clusters
- `subcategory_schedule_reasoning`: `6` clusters
- `subcategory_number_sequence`: `3` clusters
- `subcategory_magnitude_comparison`: `4` clusters

So reasoning/math is not a trivial one-mode-per-task stratum. It fragments
meaningfully.

### 2. At the cluster level, reasoning/math is more operation-dominated than frame-dominated

Across the saved `k = 12` split:

- weighted dominant-subcategory majority share `= 0.8125`
- weighted dominant-frame majority share `= 0.5938`

That is a meaningful contrast with factual recall. Factual recall’s strongest
current read is family-plus-frame-conditioned. Reasoning/math is not frame-only
in the same way. The top-level clustering is better explained by operation type
than by prompt frame.

The clearest pure or near-pure operation clusters are:

- schedule-reasoning clusters `3 / 4 / 5 / 12`
- number-sequence cluster `1`
- magnitude-comparison cluster `9`

The mixed clusters are concentrated in a few cross-operation joins:

- cluster `2`: arithmetic `counted_lost_added` plus number-sequence
  `{sequence_continues, extend_pattern}`
- cluster `8`: arithmetic `had_gave_bought` plus number-sequence
  `worksheet_lists` with one magnitude spillover
- cluster `10`: arithmetic `starts_gives_gets` plus magnitude
  `between_amounts`

So the residual mixing is not random. It is structured around a few recurring
surface forms.

### 3. Inside each operation, frame still organizes the submodes strongly

Per-subcategory overall frame-majority share:

- arithmetic story: `0.9844`
- number sequence: `0.7500`
- magnitude comparison: `0.6875`
- schedule reasoning: `0.6875`

This means the best current reasoning/math read is two-stage:

- operation dominates the top-level clustering
- prompt frame still organizes the within-operation modes

Subcategory-specific patterns:

#### Arithmetic story

Arithmetic is almost deterministic by frame:

- `had_gave_bought`: `14` prompts in one main mode plus `1` singleton spill
- `after_give_then_buy`: `15` prompts in one main mode plus `1` singleton spill
- `counted_lost_added`: `16` prompts in one pure mode
- `starts_gives_gets`: `16` prompts plus `1` spill in a mixed mode

This is the most frame-organized reasoning/math subcategory in the saved audit.

#### Number sequence

Number sequence has three clean route modes:

- `sequence_continues` and `extend_pattern` merge into one `32`-prompt mode
- `continuing_pattern` forms its own `16`-prompt mode
- `worksheet_lists` forms its own `16`-prompt mode

So number sequence is not one-frame-per-mode. It has one merged continuation
mode plus two single-frame modes.

#### Schedule reasoning

Schedule reasoning is operation-pure but still frame-split:

- `starts_minutes_later` and `schedule_shows` merge into one `32`-prompt mode
- `if_begins_lasts` and `wraps_up_at` split across `13 / 10 / 7 / 1 / 1`

That is more fragmented than a simple frame grouping, but it is still visibly
frame-conditioned.

#### Magnitude comparison

Magnitude comparison has one mixed frame mode and two near-pure frame modes:

- `note_lists` and `if_measures` merge into a `34`-prompt mode
- `holds_vs_holds` forms a `15`-prompt near-pure mode
- `between_amounts` forms a `14`-prompt near-pure mode plus `1` singleton spill

So magnitude is also not one universal comparison route.

## Interpretation

The strongest honest read is:

- reasoning/math is a meaningful secondary raw-source result on the primary
  Gemma spine
- it is more operation-dominated than factual recall at the top cluster level
- but its within-operation route modes are still strongly frame-conditioned

That makes reasoning/math interesting, but it does **not** beat factual recall
as the repo’s main structured-interpretability center.

Why factual recall should still stay central:

- factual recall has the stronger raw-source signal overall
  (`0.4709` vs `0.2456`)
- factual recall already has the strongest downstream bridge into the bounded
  tool-breakage lane
- reasoning/math still lacks a comparably mature extension lane

So the project story should tighten to:

- factual recall is the strongest and most actionable structured stratum
- reasoning/math is the strongest supporting contrast because it shows a more
  operation-dominated but still templated routing structure

## Limitations

- This is still a saved-artifact sequence-level audit.
- The frame labels are deterministic labels recovered from the fixed
  `registry_v5` template construction, not causal rewrites.
- The mixed arithmetic/sequence and arithmetic/magnitude clusters are suggestive
  of shared reasoning form, but this artifact does not establish that causally.
- This does not justify a new oracle run or a new downstream bridge lane by
  itself.

## Next Steps

- Close `resattn-v7h`.
- Keep factual recall as the main structured-interpretability center and bridge
  lane.
- Keep `resattn-b4h` as the next active implementation step.
- If reasoning/math returns later, make the next step a targeted audit of the
  mixed cross-operation clusters rather than another broad rerun.
