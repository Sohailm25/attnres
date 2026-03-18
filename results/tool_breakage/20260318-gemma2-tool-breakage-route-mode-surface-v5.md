# ABOUTME: Documents the route-mode-aware redesign of the one-token factual tool-breakage surface.
# ABOUTME: Explains why the main collection now narrows to robust capital, element, and author modes.

# Motivation

`resattn-q38` follows the saved factual route-mode characterization in
`results/block_structure/20260318-gemma2-factual-route-modes-v1.md` and the
coverage audit in
`results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-coverage-v4.md`.
That audit showed that the one-token matched-family `v4` surface was still too
coarse:

- capitals covered only `1 / 3` capital modes
- elements covered only `1 / 3` element modes
- authors covered only `2 / 5` author modes
- the stronger redesign target was broad route-mode undercoverage, not moon
  prompts by themselves

So the next truthful move was not another pooled rerun on `v4`. It was a new
prompt surface built around the robust missing route modes.

# Methods

- Source artifact 1:
  `results/block_structure/20260318-gemma2-factual-route-modes-v1.json`
- Source artifact 2:
  `results/tool_breakage/20260318-gemma2-tool-breakage-route-mode-coverage-v4.json`
- Prompt registry:
  `prompts/registry_v5.yaml`
- New collection id:
  `tool_breakage_factual_recall_v5`

Design rules:

1. keep the surface one-token under the `google/gemma-2-2b` tokenizer
2. target route modes explicitly rather than family labels alone
3. keep moon prompts in the sidecar lane
4. keep the singleton author outlier out of the main collection until a later
   stability check justifies treating it as a real mode

The collection therefore targets the following robust route modes:

| Family | Route modes in main collection | Prompt forms |
|---|---|---|
| capital | `2`, `3`, `4` | `On most maps ... appears as`, `The capital city ... is`, `In a geography quiz ... would be` |
| element | `1`, `5`, `9` | `The chemical symbol ... is`, `In the periodic table ... is abbreviated as`, `Chemistry notes ...` / `A lab chart ...` |
| author | `6`, `7`, `11`, `12` | `Literature students learn ...`, `Most library catalogs list ... under`, `The novel ... was written by`, `The author of ... is` |

Collection shape:

- pilot prompts: `10`
- confirm prompts: `20`
- family mix:
  - capitals: `3` pilot / `6` confirm
  - elements: `3` pilot / `6` confirm
  - authors: `4` pilot / `8` confirm

# Results

The new `tool_breakage_factual_recall_v5` surface now encodes the route-mode
design directly in prompt tags:

- `route_mode_capital_cluster_2`
- `route_mode_capital_cluster_3`
- `route_mode_capital_cluster_4`
- `route_mode_element_cluster_1`
- `route_mode_element_cluster_5`
- `route_mode_element_cluster_9`
- `route_mode_author_cluster_6`
- `route_mode_author_cluster_7`
- `route_mode_author_cluster_11`
- `route_mode_author_cluster_12`

Two explicit guardrails now hold in the registry:

- the main `v5` collection contains no moon prompts
- every pilot and confirm target remains a single Gemma token

This means future tool-breakage runs can test the route-mode bridge directly
instead of reusing a family-balanced surface that already proved mode-blind.

# Limitations

- This is a prompt-surface design artifact, not a new Gemma run.
- The collection does not yet prove that the new prompts land on the intended
  modes under the model; it only makes the intended coverage explicit.
- The singleton author outlier (`cluster 10`) remains excluded from the main
  collection, so the redesign is intentionally conservative rather than fully
  exhaustive.
- Moon prompts remain unresolved and intentionally bounded to the sidecar lane.

# Next Steps

- Close `resattn-q38`.
- Take the next tool-breakage execution issue on `tool_breakage_factual_recall_v5`
  rather than returning to pooled `v4` reruns.
- Keep `resattn-a1w` as the moon-family sidecar.
- Treat the singleton author outlier as a separate stability question rather
  than silently folding it into the main surface.
