ABOUTME: Summarizes how the one-token Gemma tool-breakage surface covers the saved factual route modes.
ABOUTME: Decides whether the next extension should stay on the current surface or move to a route-mode-aware redesign.

# Motivation

`resattn-unp` follows the saved factual route-mode artifact and the one-token
`tool_breakage_factual_recall_v4` lane. The family-level `v4` read was already
mixed but useful. The missing question was stricter:

- does the current one-token surface actually cover the factual route modes that
  showed up in the primary-model oracle lane
- or does it only cover a thin slice of each family while missing the stronger
  within-family structure

# Methods

- Model: `google/gemma-2-2b`
- Oracle source artifact:
  `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/oracle_eval_run.json`
- Factual route-mode source artifact:
  `results/block_structure/20260318-gemma2-factual-route-modes-v1.json`
- Tool-breakage source artifacts:
  - `results/tool_breakage/20260318-gemma2-tool-breakage-one-token-pilot-v4/summary.json`
  - `results/tool_breakage/20260318-gemma2-tool-breakage-one-token-confirm-v4/summary.json`
- Prompt registry: `prompts/registry_v5.yaml`
- Factual subset: `stratum_factual_recall` on the locked `confirm` split
- Tool-breakage surface: `tool_breakage_factual_recall_v4` pilot plus confirm
  (`48` prompts total)
- Procedure:
  - rebuild the factual prompt-to-mode bridge with the saved `cluster_count = 12`
  - treat the saved factual route-mode artifact as the family-conditioned mode
    map
  - audit how many modes receive:
    - any `v4` prompt assignment
    - matching-family `v4` prompt assignment

# Results

## Overall coverage

- factual family-conditioned modes in the saved artifact: `13`
  - the count is `13`, not `12`, because the saved factual route-mode artifact
    preserves one author outlier inside the moon-dominant raw cluster as its own
    family-conditioned mode
- modes with any `v4` prompt assignment: `8 / 13`
- modes with a matching-family `v4` prompt assignment: `5 / 13`

So the current one-token surface is not just partial. It is materially
under-covering the factual route-mode map.

## Family-level coverage

| Family | Family-conditioned modes | Modes with any assignment | Modes with matching-family assignment | Uncovered mode cluster labels |
|---|---:|---:|---:|---|
| author | `5` | `4` | `2` | `6 / 10 / 11` |
| capital | `3` | `1` | `1` | `2 / 4` |
| element | `3` | `2` | `1` | `5 / 9` |
| moon | `2` | `1` | `1` | `8` |

The main undercoverage is not moons. It is that capitals and elements each
collapse onto a single covered mode.

## The current `v4` prompts collapse whole families onto one mode

- Capitals:
  - all `12` capital prompts map to capital mode `cluster 3`
  - no capital prompt covers capital modes `2` or `4`
- Elements:
  - all `12` element prompts map to element mode `cluster 1`
  - no element prompt covers element modes `5` or `9`
- Authors:
  - only `5 / 12` author prompts land on matching author modes
  - `4` author prompts cover author mode `12`
  - `1` author prompt covers author mode `7`
  - `7` author prompts miss the author family entirely and instead map to the
    dominant capital mode `3`
- Moons:
  - only `1 / 12` moon prompts lands on a matching moon mode
  - none cover moon mode `8`
  - most moon prompts map to author or element modes instead

## Interpretation

The next extension should stay on capitals, elements, and authors, but it
should not stay on the current `v4` surface unchanged.

The right read is:

- the factual-family bridge is still correct in direction
- the current one-token surface is too narrow at the mode level
- the next honest move is a route-mode-aware prompt-surface redesign

This is why the moon-family sidecar should stay bounded:

- moon undercoverage is real
- but the stronger, more general problem is that the main bridge families are
  also missing modes
- so the next main step should not let moon cleanup drive the whole lane

# Limitations

- This is saved-artifact analysis, not a new confirmatory model run.
- Coverage is defined by nearest-mode assignment under the saved factual bridge,
  so it is descriptive rather than causal.
- The `13` family-conditioned modes include one author outlier inside a
  moon-dominant raw cluster; that is a faithful representation of the saved
  route-mode artifact, not a claim that there are `13` clean raw clusters.

# Next Steps

- Close `resattn-unp`.
- Take `resattn-q38` next: redesign the one-token factual surface around the
  missing capital, element, and author modes.
- Keep `resattn-a1w` as the moon-specific sidecar rather than the main bridge
  issue.
