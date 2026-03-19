# ABOUTME: Compares the retained all-token Gemma router baseline against a bounded tokenwise teacher target.
# ABOUTME: Reuses the frozen all-token family-summary split so only teacher fidelity changes.

# Gemma-2 Router-Distillation Teacher-Target Comparison v1

## Motivation

`resattn-d36` left Phase 6 with a bounded but improved pilot baseline:

- `mlp + h_4[t] + oracle_alpha_logit_vector + mean_token_logits_then_softmax + all_tokens_target_mse`

That baseline still repeats one sequence-level oracle target at every token. The
next honest question was therefore narrower than a full export redesign:

- does a bounded oracle-conditioned tokenwise teacher beat the retained
  repeated-sequence target on the same frozen split?

`resattn-afu` tests that directly while keeping the model family, input field,
target parameterization, inference aggregation, optimizer, and split fixed.

## Methods

- Input dataset:
  `results/router_training/20260318-gemma2-router-distillation-pilot-export-v1/`
- Frozen baseline / split source:
  `results/router_training/20260318-gemma2-router-distillation-family-comparison-all-tokens-v1/summary.json`
- Frozen baseline:
  - router family: `mlp`
  - hidden dim: `512`
  - input: `h_4[t]`
  - target: `oracle_alpha_logit_vector`
  - inference aggregation: `mean_token_logits_then_softmax`
  - split: exact saved `192 / 64` train/eval prompt ids from the frozen family
    artifact
- Candidate supervision objectives:
  - `all_tokens_target_mse`
  - `next_token_positions_sequence_target_mse`
  - `next_token_positions_oracle_alpha_target_logit_contribution_mse`

Teacher definitions:

- `all_tokens_target_mse`:
  repeat the saved prompt-level `oracle_alpha_logit_vector` target on every
  token
- `next_token_positions_sequence_target_mse`:
  same repeated prompt-level target, but only on positions with a defined next
  token
- `next_token_positions_oracle_alpha_target_logit_contribution_mse`:
  bounded oracle-conditioned per-token source preferences derived from
  shared-final-norm pre-softcap contributions to the ground-truth next-token
  logit

Optimization:

- Adam
- learning rate `0.001`
- weight decay `0.0001`
- batch size `16`
- max epochs `300`
- patience `40`
- seed `11`

Selection rule:

- primary metric: held-out `R^2`
- secondary metric: held-out mean JS divergence

Readiness gate:

- prereg `R^2 > 0.5`

Machine-readable artifact:

- `results/router_training/20260318-gemma2-router-distillation-teacher-target-comparison-v1/summary.json`

Implementation checks:

- the first launch failed before training because the new teacher path rejected
  Gemma's `RMSPre` final norm and had not imported `_fixed_residual_sources`
  explicitly
- those were root-cause implementation defects, not scientific anomalies:
  - the builder now accepts both `RMS` and `RMSPre`
  - `RMSPre` is treated as unit-weight RMS normalization
  - `_fixed_residual_sources` is now imported explicitly from
    `validation.oracle_alpha_runner`
- the exact-command rerun preserved the `summary.json` hash on MPS:
  `97872d32c0f5fa3d6bf5c75bd7f06c7acab5874a`

## Results

Held-out pilot comparison:

| Supervision objective | Held-out `R^2` | Held-out mean JS | Best epoch |
|---|---:|---:|---:|
| `all_tokens_target_mse` | `0.4211` | `0.0785` | `34` |
| `next_token_positions_sequence_target_mse` | `0.4201` | `0.0792` | `34` |
| `next_token_positions_oracle_alpha_target_logit_contribution_mse` | `-11.2639` | `0.4320` | `2` |

Selected objective:

- `all_tokens_target_mse`
- readiness cleared: `false`

Relative to the retained all-token baseline:

- `next_token_positions_sequence_target_mse`
  - `R^2` delta `= -0.0010`
  - mean JS delta `= +0.0008`
- `next_token_positions_oracle_alpha_target_logit_contribution_mse`
  - `R^2` delta `= -11.6850`
  - mean JS delta `= +0.3536`

The bounded tokenwise teacher was not a near miss. It collapsed quickly,
selected `best_epoch = 2`, and produced much lower predicted entropy
(`2.5624` versus `3.7136` for the retained baseline).

## Interpretation

This is a real negative result for the specific bounded teacher we chose.

The truthful read is:

- the retained all-token repeated sequence target is still the best objective on
  this saved pilot surface
- merely restricting supervision to positions with a valid next token does not
  change the story; the matched-mask control is effectively a tie with the
  baseline
- the shared-final-norm next-token contribution teacher is the wrong target for
  this Phase 6 objective

The main implication is narrower than "tokenwise supervision is bad." The
bounded tokenwise approximation is bad. That matters because it blocks a lazy
interpretation that any teacher made to look more token-local must help.

What this result does **not** show:

- it does **not** show that exact tokenwise oracle teachers would fail
- it does **not** show that target fidelity is irrelevant
- it does **not** justify reopening width or family search by inertia

What it does show:

- the current repeated all-token sequence target remains the least-bad baseline
- the next honest target-fidelity step must use a more faithful tokenwise
  oracle-teacher construction, or stay on the repeated all-token target
- architecture search should remain frozen while that question is unresolved

## Limitations

- The bounded teacher is still an approximation, not a true tokenwise oracle
  optimization target.
- The comparison is still pilot-only on the saved `192 / 64` split.
- The retained baseline remains below the prereg readiness gate.

## Next Steps

- Close `resattn-afu` as a stable negative result for the bounded shared-final-
  norm tokenwise teacher path.
- Keep the saved Phase 6 baseline as:
  `mlp + h_4[t] + oracle_alpha_logit_vector + mean_token_logits_then_softmax + all_tokens_target_mse`
- Take one narrower follow-up before any full export redesign:
  regenerate a small exact tokenwise oracle-teacher slice on a bounded subset
  and test whether true tokenwise supervision actually beats the retained
  baseline.
