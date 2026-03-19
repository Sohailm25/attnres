# ABOUTME: Compares exact tokenwise oracle teachers against the retained all-token Gemma router baseline on a bounded subset.
# ABOUTME: Reuses the frozen all-token family-summary split and changes only teacher fidelity on a deterministic stratum-balanced slice.

# Gemma-2 Router-Distillation Exact-Teacher Subset v1

## Motivation

`resattn-afu` closed one tempting but weak story: a bounded shared-final-norm
contribution teacher was not just unhelpful, it was actively bad. But that
still left a fairer question open:

- if we optimize the true per-token routed loss directly, does exact tokenwise
  supervision beat the retained repeated sequence-level target on the same
  prompt slice?

`resattn-7xo` answers that question on the smallest honest slice that is still
worth reading:

- keep the retained all-token MLP baseline
- keep the repeated sequence-target controls
- change only the teacher fidelity
- use a deterministic stratum-balanced subset so exact tokenwise oracle
  optimization stays bounded

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
- Deterministic bounded subset:
  - train: `32` prompts (`8` per stratum)
  - eval: `16` prompts (`4` per stratum)
  - strata:
    - `stratum_factual_recall`
    - `stratum_reasoning_math`
    - `stratum_code_procedural`
    - `stratum_general_text`

Teacher definitions:

- `all_tokens_target_mse`:
  repeat the saved prompt-level `oracle_alpha_logit_vector` target on every
  token
- `next_token_positions_sequence_target_mse`:
  same repeated prompt-level target, but only on positions with a defined next
  token
- `next_token_positions_exact_oracle_alpha_logit_mse`:
  optimize a separate softmax alpha for each token position directly against the
  next-token cross-entropy from routed logits under Gemma-2's own final norm and
  logits softcap; then convert the optimized alphas to centered log-alpha
  targets

Exact-teacher settings:

- optimization steps per token: `30`
- learning rate: `0.1`

Router optimization:

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

- `results/router_training/20260319-gemma2-router-distillation-exact-teacher-subset-v1/summary.json`

Implementation checks:

- a tiny `8 / 4` calibration slice completed cleanly before the real run, so the
  exact-teacher path was no longer an unverified implementation jump
- the exact-command rerun preserved the `summary.json` hash on MPS:
  `ff6c608adfbb05dcf5895a11ce8740553304f9b2`

## Results

Held-out bounded-subset comparison:

| Supervision objective | Held-out `R^2` | Held-out mean JS | Best epoch |
|---|---:|---:|---:|
| `all_tokens_target_mse` | `0.1567` | `0.1014` | `23` |
| `next_token_positions_sequence_target_mse` | `0.1803` | `0.0998` | `23` |
| `next_token_positions_exact_oracle_alpha_logit_mse` | `-0.1126` | `0.1486` | `20` |

Selected objective:

- `next_token_positions_sequence_target_mse`
- readiness cleared: `false`

Relative to the retained all-token baseline on this subset:

- `next_token_positions_sequence_target_mse`
  - `R^2` delta `= +0.0235`
  - mean JS delta `= -0.0016`
- `next_token_positions_exact_oracle_alpha_logit_mse`
  - `R^2` delta `= -0.2694`
  - mean JS delta `= +0.0472`

Runtime:

- first real run wall-clock: `566.14s`
- exact-command rerun wall-clock: `559.56s`

## Interpretation

This is a negative result for the main tokenwise-teacher hypothesis.

The exact teacher is not just worse than the retained all-token baseline. It is
worse than both repeated-sequence controls on the bounded subset. So the repo
should **not** read this as “tokenwise supervision is the obvious next export
upgrade.”

The truthful read is:

- exact tokenwise oracle teachers do not justify a broader export redesign from
  this result
- the retained repeated sequence-level target remains the safer Phase 6 default
- architecture search should stay frozen while the target mismatch is better
  understood

There is one useful nuance:

- on this bounded subset, `next_token_positions_sequence_target_mse` modestly
  beats `all_tokens_target_mse`

That is real on this slice, but it is not enough to overturn the larger
full-split evidence from `resattn-afu`, where the matched next-token mask
control was effectively a tie with the retained all-token baseline. So the
subset result is a bounded hint about support, not a new global baseline change.

The exact-teacher failure narrows the next honest question:

- is the problem high within-prompt variation in exact tokenwise routes, which a
  mean-token sequence decoder cannot absorb cleanly?
- or is the exact tokenwise teacher simply misaligned with the sequence-level
  alpha object the pilot is still scored on?

That is now a diagnosis question on the saved artifact, not a license for a
larger tokenwise export build.

## Limitations

- This is a bounded deterministic subset, not the full saved `192 / 64` split.
- The exact teacher is exact with respect to the local routed next-token loss on
  this subset, but evaluation is still sequence-level after mean-token
  aggregation.
- The retained baseline and every comparison row remain below the prereg
  readiness gate.

## Next Steps

- Close `resattn-7xo` as a real negative result for exact tokenwise teacher
  rescue on the bounded Gemma subset.
- Do **not** redesign the full export around tokenwise teachers from this
  result.
- Keep the retained broader Phase 6 baseline as:
  `mlp + h_4[t] + oracle_alpha_logit_vector + mean_token_logits_then_softmax + all_tokens_target_mse`
- Take one cheaper diagnosis follow-up on the saved artifact:
  audit whether exact-tokenwise failure is better explained by within-prompt
  teacher variance or by sequence-aggregation mismatch.
