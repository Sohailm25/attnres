# Pilot/Confirmatory Prompt Registry

## Motivation

The repo required a saved pilot/confirmatory split before any claim-bearing runs, but the split previously existed only as policy text in the prereg and review docs. This artifact records the first concrete registry and its enforcement path.

## Methods

- Added `prompts/registry_v1.yaml` as a versioned prompt registry with inline prompt collections.
- Implemented `prompts/registry.py` to load the registry, validate prompt metadata, and reject confirmatory access when exploratory mode is enabled.
- Added `scripts/export_prompt_split.py` as a direct script-level access path so future runs can consume the registry without rewriting split logic.
- Covered the registry and script guard in `tests/test_prompt_registry.py`.

## Results

- The saved registry now includes:
  - `oracle_alpha_phase1_v1` with an `8` prompt pilot split and an `8` prompt confirmatory split.
  - `tool_breakage_factual_recall_v1` with an `8` prompt pilot split and an `8` prompt confirmatory split.
- The registry inherits the preregistered default null set:
  - `uniform`
  - `random_dirichlet`
  - `magnitude_proportional`
  - `last_layer_only`
- The registry inherits the default claim-bearing statistics:
  - `paired_t_test`
  - `bootstrap_1000`
- Exploratory access to confirmatory prompts now fails through both the importable helper and the export script.

## Limitations

- This is a control artifact, not a scientific result.
- The prompt collections are intentionally small and inline; larger lane-specific registries may still need to be added later.
- Safety-lane prompt handling is still intentionally absent here because that lane requires additional policy and feature-validation work first.

## Next Steps

- Add identifiability, MIB, and out-of-sample predictiveness controls before claim-bearing oracle-alpha runs.
- Extend the registry only when a lane needs a larger or more specialized prompt pool.
