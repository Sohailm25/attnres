# Pilot/Confirmatory Prompt Registry V3

## Motivation

`resattn-0vx` needed a stronger saved prompt surface after `registry_v2` first restored positive held-out routed loss. The next bounded move was to expand both the pilot and confirm oracle-alpha splits while preserving the same method path.

## Methods

- Added [registry_v3.yaml](/Users/sohailmo/resattn/prompts/registry_v3.yaml) as the new default prompt registry.
- Expanded the oracle-alpha pilot split from `16` prompts to `32` prompts.
- Expanded the oracle-alpha confirm split from `8` prompts to `16` prompts.
- Kept the tool-breakage collection unchanged.
- Pointed the default loader in [registry.py](/Users/sohailmo/resattn/prompts/registry.py) and the main config in [experiment.yaml](/Users/sohailmo/resattn/configs/experiment.yaml) at `prompts/registry_v3.yaml`.

## Results

- The repo now defaults to the larger `registry_v3` prompt surface.
- The oracle-alpha prompt surface is large enough to move beyond the earlier `8 / 8` and `16 / 8` checks without changing the rest of the runner.
- The prompt-registry and scaffold tests passed after the update.

## Limitations

- The expanded prompt surface is still an inline local collection, not a benchmark dataset.
- Only the oracle-alpha collection changed in `v3`; the tool-breakage collection is still the same as `v2`.

## Next Steps

- Use `registry_v3` for the next prereg-aligned oracle-alpha scaling step.
- Keep strong interpretation blocked until the larger held-out surface remains positive and the broader Phase 1 gate is tested.
