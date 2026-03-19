ABOUTME: Re-runs the saved Gemma router-family comparison under the improved all-token supervision baseline.
ABOUTME: Uses the frozen 192/64 family-summary split unchanged so only the family question is reopened.

# Gemma-2 Router-Distillation Family Comparison Under All-Token Supervision v1

## Motivation

`resattn-tqn` changed the honest Phase 6 baseline.

On the saved Gemma pilot export, `all_tokens_target_mse` materially improved the
retained linear baseline relative to the old sequence-level objective:

- old baseline: `linear + h_4[t] + oracle_alpha_logit_vector + mean_token_logits_then_softmax + sequence_target_mse`
- new baseline: `linear + h_4[t] + oracle_alpha_logit_vector + mean_token_logits_then_softmax + all_tokens_target_mse`

That meant the old `resattn-zic` family conclusion was no longer enough. The
bounded next question was:

- does the linear-versus-MLP ranking change once the comparison is rerun under
  the improved all-token supervision objective on the same frozen split?

## Methods

- Input dataset:
  `results/router_training/20260318-gemma2-router-distillation-pilot-export-v1/`
- Frozen supervision summary:
  `results/router_training/20260318-gemma2-router-distillation-supervision-objective-comparison-v1/summary.json`
- Frozen split:
  - exact `192 / 64` train/eval prompt IDs reused unchanged from the saved
    family-summary baseline
- Frozen settings:
  - input: `h_4[t]`
  - target: `oracle_alpha_logit_vector`
  - inference aggregation: `mean_token_logits_then_softmax`
  - supervision objective: `all_tokens_target_mse`
- Candidate router families:
  - `linear`
  - `mlp`
- MLP width:
  - `512`
- Optimization:
  - Adam
  - learning rate `0.001`
  - weight decay `0.0001`
  - batch size `16`
  - max epochs `300`
  - patience `40`
  - seed `11`
  - device `mps`
- Selection rule:
  - primary metric: held-out `R^2`
  - secondary metric: held-out mean JS divergence
- Readiness gate:
  - prereg `R^2 > 0.5`

Machine-readable artifact:

- `results/router_training/20260318-gemma2-router-distillation-family-comparison-all-tokens-v1/summary.json`

Implementation checks:

- the retained linear row under `all_tokens_target_mse` matched the saved
  supervision-objective artifact exactly on the same split:
  - held-out `R^2 = 0.4076`
  - held-out mean JS `= 0.0797`
- the exact-command rerun preserved the `summary.json` hash on MPS:
  `0ceba4be9932c94ff312c7724cf8528fb856c15c`

## Results

Held-out pilot results:

| Router family | Hidden width | Held-out `R^2` | Held-out mean JS | Best epoch |
|---|---:|---:|---:|---:|
| `linear` | `N/A` | `0.4076` | `0.0797` | `117` |
| `mlp` | `512` | `0.4211` | `0.0785` | `34` |

Selected pilot family:

- router family: `mlp`
- input: `h_4[t]`
- target: `oracle_alpha_logit_vector`
- aggregation: `mean_token_logits_then_softmax`
- supervision objective: `all_tokens_target_mse`
- readiness cleared: `false`

Relative to the retained all-token linear baseline:

- `R^2` delta `= +0.0135` for `mlp`
- mean JS delta `= -0.0012` for `mlp`
- eval loss delta `= -0.0081` for `mlp`

So the family ranking did change under the better supervision surface, but only
modestly.

## Interpretation

This is a real positive update for the MLP family, but it is not a license to
reopen broad architecture search.

The truthful read is:

- the old linear-favored family result was conditional on the weaker
  sequence-level supervision objective
- under `all_tokens_target_mse`, the widened MLP now wins on both held-out
  `R^2` and mean JS
- the gain is real but small, and the pilot still misses the prereg readiness
  gate by a meaningful margin

So the architectural conclusion tightens to:

- move the saved Phase 6 baseline from `linear` back to `mlp`, but only under
  the improved all-token supervision objective
- do **not** interpret this as evidence that more blind family or width search
  is the main missing rescue

The more plausible next blocker is still target fidelity. Repeating one
sequence-level teacher target across all tokens helped, and a slightly richer
family helped again, but neither change got the lane close to readiness. That
points more toward bounded tokenwise teacher targets than toward another
open-ended architecture sweep.

## Limitations

- This is still pilot-only and reuses a held-out split inside the pilot export,
  not the confirmatory surface.
- The supervision objective is denser, but it still uses the same sequence-level
  oracle target at every token rather than a true tokenwise oracle target.
- Only two router families were compared.

## Next Steps

- Close `resattn-d36`.
- Move the saved pilot baseline to:
  `mlp + h_4[t] + oracle_alpha_logit_vector + mean_token_logits_then_softmax + all_tokens_target_mse`
- Do not start another architecture sweep by inertia.
- Take `resattn-afu` next to compare bounded tokenwise teacher targets against
  the retained all-token MLP baseline.
