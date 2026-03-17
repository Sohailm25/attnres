# Pilot/Confirmatory Prompt Registry V4

## Motivation

`resattn-9jq` needed a saved prompt surface that was large enough to justify the new checkpointed campaign runner. The previous `32 / 16` split showed real held-out routed-loss signal, but it was still small enough that another near-term scale-up was likely.

## Methods

- Added [registry_v4.yaml](/Users/sohailmo/resattn/prompts/registry_v4.yaml) as the new default prompt registry.
- Expanded the oracle-alpha pilot split from `32` prompts to `96` prompts.
- Expanded the oracle-alpha confirm split from `16` prompts to `128` prompts.
- Kept the tool-breakage collection unchanged from `registry_v3`.
- Pointed the default loader in [registry.py](/Users/sohailmo/resattn/prompts/registry.py) and the main config in [experiment.yaml](/Users/sohailmo/resattn/configs/experiment.yaml) at `prompts/registry_v4.yaml`.

## Results

- The repo now defaults to the larger `registry_v4` prompt surface.
- The oracle-alpha collection spans `oa-pilot-001..096` and `oa-confirm-001..128`.
- The prompt-registry and scaffold expectations were updated to require the `96 / 128` split, and the full unit suite plus pre-commit hooks passed after the switch.

## Limitations

- This freezes a larger local prompt surface, not a claim-bearing oracle-alpha campaign result.
- The prompt registry is still an inline local collection rather than a benchmark dataset.
- Only the oracle-alpha collection changed in `v4`; the tool-breakage collection is still the same as `v3`.

## Next Steps

- Log the tmux session, checkpoint path, cadence, log path, and resume command in `SCRATCHPAD.md`.
- Launch the prereg-scale oracle-alpha campaign from `registry_v4` through [run_oracle_alpha_predictiveness_campaign.py](/Users/sohailmo/resattn/scripts/run_oracle_alpha_predictiveness_campaign.py).
- Verify that checkpoints materialize on disk before treating the launch as active.
