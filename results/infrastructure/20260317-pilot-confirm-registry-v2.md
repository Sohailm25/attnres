# Pilot/Confirmatory Prompt Registry V2

## Motivation

`resattn-7mb` needed the smallest fundamental oracle-alpha predictiveness redesign after the loss-aware selection fix still failed on the original `8 / 8` prompt split. The least invasive next step was to expand only the pilot surface while preserving the locked confirm set.

## Methods

- Added [registry_v2.yaml](/Users/sohailmo/resattn/prompts/registry_v2.yaml) as the new default prompt registry.
- Kept the oracle-alpha confirm split unchanged at `8` prompts.
- Expanded the oracle-alpha pilot split from `8` prompts to `16` prompts.
- Added saved paraphrases for all new pilot prompts so the existing perturbation workflow remains valid.
- Pointed the default loader in [registry.py](/Users/sohailmo/resattn/prompts/registry.py) and the main config in [experiment.yaml](/Users/sohailmo/resattn/configs/experiment.yaml) at `prompts/registry_v2.yaml`.

## Results

- The repo now defaults to `prompts/registry_v2.yaml`.
- The oracle-alpha pilot split is materially larger without contaminating the old confirm set.
- The registry/scaffold tests passed after the update.

## Limitations

- The expanded pilot surface is still an inline local prompt collection rather than a benchmark dataset.
- Only the oracle-alpha pilot split changed in `v2`; the tool-breakage collection is unchanged.

## Next Steps

- Use `registry_v2` for the next oracle-alpha loss-aware comparison.
- Keep claim-bearing interpretation blocked until the larger pilot surface is paired with stronger held-out evidence.
