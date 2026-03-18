ABOUTME: Bridges the strongest factual-recall raw-source route modes to the narrowed v5 Gemma tool-breakage surface.
ABOUTME: Records whether the current bounded bridge already covers the strongest factual routing modes or only a subset.

# Gemma-2 Factual Route-Mode Bridge v5

## Motivation

`resattn-lnu` follows the new oracle synthesis pass. The repo now has a strong
primary-model factual-recall raw-source result and a narrowed `v5`
tool-breakage bridge, but those two facts are only useful together if the
bridge is actually targeting the strongest factual routing modes.

The specific question here is narrower than the older family-level bridge:

- does `v5` cover the strongest factual route modes rather than just the right
  broad families
- which factual modes remain outside the current bridge
- does the remaining gap still justify more bridge redesign

## Methods

- Model: `google/gemma-2-2b`
- Oracle source artifact:
  `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/oracle_eval_run.json`
- Factual route-mode source artifact:
  `results/block_structure/20260318-gemma2-factual-route-modes-v1.json`
- Tool-breakage source artifacts:
  - `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-pilot-v5/summary.json`
  - `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-confirm-v5/summary.json`
- Prompt registry: `prompts/registry_v5.yaml`
- Factual subset: `stratum_factual_recall` on the locked confirm split
- Tool-breakage surface: `tool_breakage_factual_recall_v5` pilot plus confirm
  (`30` prompts total)
- Procedure:
  - reuse `scripts/run_tool_breakage_route_mode_bridge.py`
  - assign each `v5` tool-breakage prompt to the nearest saved factual
    raw-source route mode
  - count coverage by family-conditioned mode
  - compare the resulting `v5` coverage against the older saved `v4`
    coverage artifact

Machine-readable artifact:

- `results/tool_breakage/20260318-gemma2-tool-breakage-factual-route-mode-bridge-v5.json`

## Results

The narrowed `v5` bridge now covers the strongest factual route modes far more
completely than `v4`.

### Overall mode coverage

- factual family-conditioned modes in the saved artifact: `13`
- modes with any `v5` prompt assignment: `10 / 13`
- modes with matching-family `v5` prompt assignment: `10 / 13`

This is a large improvement over the saved `v4` route-mode coverage artifact:

- `v4` modes with matching-family assignment: `5 / 13`
- `v5` modes with matching-family assignment: `10 / 13`

### Family-level coverage

| Family | Family-conditioned modes | Matching-family modes covered by `v5` | Uncovered mode cluster labels |
|---|---:|---:|---|
| author | `5` | `4` | `10` |
| capital | `3` | `3` | none |
| element | `3` | `3` | none |
| moon | `2` | `0` | `8 / 10` |

So the current bridge is no longer a thin slice of the core factual families:

- capitals: full coverage (`3 / 3`)
- elements: full coverage (`3 / 3`)
- authors: near-full coverage (`4 / 5`)

The only uncovered author mode is `cluster 10`, which is the singleton outlier
the repo intentionally excluded from the main bridge. The remaining uncovered
family is moons, which the repo also intentionally kept out of the main
collection.

### Prompt-level alignment

The `v5` bridge is now fully aligned at the family level:

- overlap prompt fraction: `1.0`
- matching subcategory fraction: `1.0`
- matching subcategory fraction on overlap prompts: `1.0`

Every `v5` prompt assigns to a nearest factual route mode in the matching
family. This is stronger than the older `dat` family bridge, where only
`6 / 16` prompts overlapped the strongest factual families at all.

### Mode-level breakage magnitudes

Each covered `v5` mode is represented by `3` prompts (`1` pilot, `2` confirm).
The mean tuned final-position KL delta under routing differs materially by mode:

- strongest covered mode:
  - `subcategory_author_fact::cluster_7 = +12.7356`
- next strongest covered modes:
  - `subcategory_author_fact::cluster_12 = +5.1216`
  - `subcategory_element_symbol::cluster_9 = +4.7830`
  - `subcategory_capital_fact::cluster_3 = +4.6148`
  - `subcategory_element_symbol::cluster_5 = +4.4919`
- milder covered modes:
  - `subcategory_author_fact::cluster_6 = +1.7056`
  - `subcategory_capital_fact::cluster_2 = +2.1446`
  - `subcategory_capital_fact::cluster_4 = +2.2515`
  - `subcategory_element_symbol::cluster_1 = +2.2107`
  - `subcategory_author_fact::cluster_11 = +2.7926`

So `v5` is not just aligned in coverage. It also spans both mild and strong
breakage modes inside the matched factual families.

## Interpretation

- The narrowed `v5` bridge already covers the strongest factual routing modes
  for the core bridge families the repo actually cares about:
  - all capital modes
  - all element modes
  - all non-outlier author modes
- This materially upgrades the old bridge story. The repo no longer has to say
  “the bridge points in the right direction but only overlaps loosely.”
- The remaining gaps are now explicit and narrow:
  - the singleton author outlier `cluster 10`
  - the excluded moon modes
- That means the current bounded bridge is already well aligned with the
  strongest factual raw-source structure on the primary model. If tool-breakage
  stays mixed from here, the main explanation is no longer poor mode coverage.
  It is donor-arm geometry and route-mode heterogeneity inside the covered
  families.

## Limitations

- This remains a saved-artifact bridge, not a new intervention run.
- The uncovered author `cluster 10` mode is a singleton outlier, so treating it
  as a mandatory main-surface target would probably overfit.
- Moon modes remain intentionally outside the main bridge, so this artifact does
  not say the full factual family space is covered.

## Next Steps

- Close `resattn-lnu`.
- Treat the main factual bridge as structurally aligned for capitals, elements,
  and authors.
- If tool-breakage resumes, prioritize `resattn-oi7` over another bridge
  redesign, because donor geometry now looks like the real remaining bottleneck.
- Keep `resattn-a1w` as the moon-specific sidecar rather than reopening the
  main bridge around moon coverage.
