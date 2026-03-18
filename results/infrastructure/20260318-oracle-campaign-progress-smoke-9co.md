# ABOUTME: Summarizes the smoke artifact for campaign summary-stage progress observability.
# ABOUTME: Records the new manifest and progress files plus log lines added for `resattn-9co`.

## Motivation

`resattn-9co` targeted an operational blind spot in the oracle campaign runner.
Once the expensive oracle checkpoints were complete, the repo still looked stale
until the very end of the predictiveness sweep. The goal here was to prove that
the campaign now exposes real progress during that summary stage without
changing the final summary format.

## Methods

- Script:
  - `scripts/run_oracle_alpha_predictiveness_campaign.py`
- Model:
  - `tiny-stories-1M`
- Device:
  - `cpu`
- Collection:
  - `oracle_alpha_phase1_v1`
- Smoke size:
  - `2` pilot prompts
  - `1` confirm prompt
- Candidate set:
  - one feature source
  - one target
  - regularization grid `{0.001, 0.1}`
- Output directory:
  - `results/infrastructure/20260318-oracle-campaign-progress-smoke-9co/`

## Results

- The campaign now writes `campaign_manifest.json` before the final
  predictiveness summary lands.
- The campaign now writes `predictiveness_progress.json` during the tuning
  sweep, not just at the end.
- The final progress artifact recorded:
  - `status = complete`
  - `num_train_sequences = 2`
  - `num_eval_sequences = 1`
  - `total_candidate_pairs = 1`
  - `completed_candidate_pairs = 1`
  - `total_regularization_evaluations = 2`
  - `completed_regularization_evaluations = 2`
- The smoke log captured the expected live progress lines during tuning:
  - one line after the first regularization
  - one line after the second regularization

## Interpretation

- The campaign no longer looks partially failed once the oracle stage is done.
- The observability fix is intentionally narrow:
  - manifest exists early
  - progress counts update during tuning
  - final summary semantics stay unchanged
- That is enough to close the original UX failure mode from the prereg-scale
  campaign.

## Limitations

- This is a tiny smoke artifact on `tiny-stories-1M`, not a large rerun on the
  full Gemma primary-model surface.
- It validates observability mechanics, not scientific metrics.

## Next Steps

- Close `resattn-9co`.
- Treat the oracle campaign path as operationally ready for larger reruns.
- Move the remaining ready queue to `resattn-ac2`.
