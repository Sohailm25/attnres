# Prereg-Scale Oracle-Alpha Campaign Build-Out

## Motivation

`resattn-0vx` made the current oracle-alpha path worth scaling, but the existing held-out script was still a disposable rerun path: no prompt-level resume, no reusable feature cache, and no saved full regularization grid. The next scale-up needed to be operationally durable before it became scientifically expensive.

## Methods

- Added `validation/oracle_alpha_campaign.py` as a filesystem-backed wrapper around the existing runner.
- Extended `validation/oracle_alpha_runner.py` so:
  - collection runs can reuse cached prompt-level oracle results
  - feature extraction can reuse cached prompt-level feature vectors
  - each feature-source and target candidate now records the full regularization grid, not only the selected `lambda`
- Added `scripts/run_oracle_alpha_predictiveness_campaign.py` to launch a checkpointed campaign under one output directory.
- Added unit coverage for:
  - full regularization-grid reporting
  - campaign-manifest and checkpoint creation
  - campaign resume without recomputing oracle optimization or feature vectors

## Results

- The repo can now materialize prompt-level oracle checkpoints under `checkpoints/oracle_runs/<split>/`.
- The repo can now materialize reusable feature caches under `checkpoints/feature_vectors/<split>/<feature_source>/`.
- A campaign run now writes:
  - `campaign_manifest.json`
  - `oracle_train_run.json`
  - `oracle_eval_run.json`
  - `predictiveness_summary.json`
- This is infrastructure only. No new scientific oracle-alpha result is claimed from this slice.

## Limitations

- The larger prereg-scale prompt surface is not frozen yet, so this build-out does not itself clear `resattn-9jq`.
- The campaign runner still uses the current final-output development-model oracle-alpha surface rather than the later multi-layer claim-bearing analysis surface.
- The runner now saves the full tested `lambda` grid, but only for the feature sources and targets included in a given launch.

## Next Steps

- Freeze the next saved oracle-alpha prompt surface, likely `registry_v4`, with a materially larger confirm tranche.
- Launch the prereg-scale campaign inside `tmux` using the new checkpointed script.
- Record the tmux session, checkpoint paths, cadence, log path, and resume command in `SCRATCHPAD.md` before launch.
