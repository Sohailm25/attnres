ABOUTME: Quantifies prompt-frame concentration inside the saved Gemma factual route modes.
ABOUTME: Sharpens the strongest current claim from family-only structure to family-plus-frame-conditioned route modes.

# Gemma-2 Factual Route-Mode Frame Audit v1

## Motivation

`resattn-chz` established that the strongest raw-source structure on the
primary Gemma spine lives inside factual recall and that the route modes looked
family-plus-prompt-frame-conditioned from saved examples.

That wording was still too soft. The real next question was narrower:

- how much of the covered capital, element, and author route-mode structure is
  actually explained by prompt frame
- whether the strongest current claim should stop at semantic family or move to
  family-plus-frame conditioning
- whether the bounded factual tool-breakage bridge should now be read as
  family-conditioned only or as family-plus-frame-conditioned

This pass answers that using saved artifacts only.

## Methods

- Model: `google/gemma-2-2b`
- Source artifacts:
  - `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/oracle_eval_run.json`
  - `results/block_structure/20260318-gemma2-factual-recall-cluster-profile-v1.json`
  - `results/tool_breakage/20260318-gemma2-tool-breakage-factual-route-mode-bridge-v5.json`
- Prompt registry: `prompts/registry_v5.yaml`
- Subset: locked factual-recall `confirm` prompts only
- Cluster assignments:
  - reused the saved factual raw-source split with best `k = 12`
  - reassigned confirm prompts to those same average-linkage JSD clusters
- Families audited:
  - `subcategory_capital_fact`
  - `subcategory_element_symbol`
  - `subcategory_author_fact`
- Frame labels:
  - derived deterministically from the fixed prompt templates in `registry_v5`
  - no new model run or new clustering was launched

For each family, the audit reports:

- frame counts inside each covered route mode
- frame-majority share across covered prompts
- unique-entity spread inside each mode as a check against a content-only read

## Results

## 1. Capitals and elements are deterministic frame-group splits, not just family splits

Covered capital modes (`3 / 3`):

| Cluster | Size | Frame counts | Unique entities | Max count per entity |
|---|---:|---|---:|---:|
| `3` | `32` | `direct_capital = 16`, `travel_guide = 16` | `16` | `2` |
| `2` | `16` | `map_surface = 16` | `16` | `1` |
| `4` | `16` | `quiz_surface = 16` | `16` | `1` |

Covered element modes (`3 / 3`):

| Cluster | Size | Frame counts | Unique entities | Max count per entity |
|---|---:|---|---:|---:|
| `9` | `32` | `chemistry_notes = 16`, `lab_chart = 16` | `16` | `2` |
| `1` | `16` | `direct_symbol = 16` | `16` | `1` |
| `5` | `16` | `periodic_table = 16` | `16` | `1` |

Two things are clear.

First, the capital and element route modes are not fragmenting by answer
content. Each mode spans all `16` countries or all `16` elements. The merged
clusters simply contain two prompts per entity because the paired frames route
together.

Second, the split is still highly frame-organized:

- capitals: `64 / 64` prompts are explained by a deterministic three-way frame
  grouping:
  - `{direct_capital, travel_guide}`
  - `{map_surface}`
  - `{quiz_surface}`
- elements: `64 / 64` prompts are explained by a deterministic three-way frame
  grouping:
  - `{chemistry_notes, lab_chart}`
  - `{direct_symbol}`
  - `{periodic_table}`

So for these two families, the strongest honest read is already stronger than
`family-conditioned` and still narrower than `content-conditioned`:

- the modes are family-plus-frame-conditioned
- the only ambiguity is which nearby frame pair merges inside each family

## 2. Authors are also frame-dominated, but one frame still fragments further

Covered author modes (`4 / 5`; uncovered outlier `cluster 10` excluded from the
bridge):

| Cluster | Size | Frame counts | Unique entities | Max count per entity |
|---|---:|---|---:|---:|
| `12` | `18` | `direct_author = 16`, `novel_title = 2` | `16` | `2` |
| `6` | `16` | `literature_students = 16` | `16` | `1` |
| `7` | `16` | `library_catalog = 16` | `16` | `1` |
| `11` | `13` | `novel_title = 13` | `13` | `1` |

Residual uncovered author outlier:

| Cluster | Size | Frame counts | Entity |
|---|---:|---|---|
| `10` | `1` | `novel_title = 1` | `Things Fall Apart` |

The author family is still overwhelmingly frame-conditioned:

- covered-mode frame-majority share `= 61 / 63 = 0.9683`
- all `16` `literature_students` prompts form their own mode
- all `16` `library_catalog` prompts form their own mode
- all `16` `direct_author` prompts stay together in `cluster 12`

What remains is a narrower residual split inside the `novel_title` frame:

- `13 / 16` novel-title prompts form `cluster 11`
- `2 / 16` spill into the direct-author cluster:
  - `Moby-Dick`
  - `Frankenstein`
- `1 / 16` becomes the singleton uncovered outlier:
  - `Things Fall Apart`

So the author family is not merely family-plus-frame-conditioned in a clean
one-mode-per-frame way. It is frame-dominated with a small residual within-frame
split on novel-title prompts.

## 3. The bounded factual tool-breakage bridge should now be read as family-plus-frame-conditioned

The narrowed `v5` bridge already covers:

- all capital modes
- all element modes
- all non-outlier author modes

After this frame audit, that means the bridge is not only family-aligned. It is
already aligned to the main covered frame groups inside those families:

- capital direct/travel, map, and quiz routes
- element direct, periodic-table, and notes/chart routes
- author literature, catalog, direct-author, and the main novel-title route

That does not make the tool-breakage lane content-pure. It sharpens the
interpretation in a narrower direction:

- the bounded factual tool-breakage lane samples family-plus-frame-conditioned
  route modes
- the remaining ambiguity is the residual within-frame author-title split, not
  whether the bridge is still missing the main capital or element route groups

## Main Claim Update

The strongest current factual-structure claim should now read:

- on the primary Gemma spine, the strongest raw-source route modes inside
  factual recall are family-plus-prompt-frame-conditioned
- capitals and elements are effectively deterministic frame-group splits within
  family
- authors are frame-dominated with a small residual within-frame split on
  novel-title prompts

The strongest current claim should **not** read:

- family-conditioned only
- content-only routing disentangled from prompt frame
- one universal raw `~8`-cluster story

## Limitations

- This is still a saved-artifact sequence-level audit.
- The frame labels are deterministic template labels from `registry_v5`, not a
  causal intervention on prompt wording.
- The author-family residual split is too small to interpret as a new positive
  story by itself.
- This does not reopen any frozen Figure 8 or broad donor-arm claims.

## Next Steps

- Close `resattn-i6i`.
- Keep `resattn-914` as the next main implementation step.
- If another saved-artifact scientific follow-up is needed later, narrow it to
  the residual author novel-title split rather than reopening the broad
  family-versus-frame question.
