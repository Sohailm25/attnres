ABOUTME: Compares linear and MLP router families on the saved Gemma router-distillation pilot export.
ABOUTME: Keeps the current Phase 6 baseline fixed so the family question can be answered before another redesign.

# Gemma-2 Router-Distillation Family Comparison v1

## Motivation

`resattn-3ak` showed that widening the current 2-layer MLP only nudged the best
pilot path upward:

- retained best width-only baseline:
  `oracle_alpha_logit_vector + mean_token_logits_then_softmax + h_4[t] + hidden_dim=512`
- held-out `R^2 = 0.3270`
- held-out mean JS `= 0.0837`

That result mattered, but it did not clear the prereg readiness gate and it did
not justify another width-by-inertia sweep.

`resattn-zic` keeps the saved export, split, input, target, and aggregation
fixed and changes only the router family:

- `linear`
- `mlp`

The goal is to test whether nonlinearity is the missing lever or whether the
next blocker lies elsewhere.

## Methods

- Input dataset:
  `results/router_training/20260318-gemma2-router-distillation-pilot-export-v1/`
- Model:
  saved `google/gemma-2-2b` pilot export only
- Frozen input:
  - `h_4[t]`
- Frozen target:
  - `oracle_alpha_logit_vector`
- Frozen aggregation:
  - `mean_token_logits_then_softmax`
- Candidate router families:
  - `linear`
  - `mlp`
- MLP width:
  - `512`
- Split:
  - pilot only
  - same stratified `192 / 64` train/eval split reused across both families
- Optimization:
  - Adam
  - learning rate `0.001`
  - weight decay `0.0001`
  - batch size `16`
  - max epochs `300`
  - patience `40`
  - seed `11`
- Selection rule:
  - primary metric: held-out `R^2`
  - secondary metric: held-out mean JS divergence
- Readiness gate:
  - prereg `R^2 > 0.5`

Machine-readable artifact:

- `results/router_training/20260318-gemma2-router-distillation-family-comparison-v1/summary.json`

## Results

Held-out pilot results after the exact-command rerun:

| Router family | Hidden width | Held-out `R^2` | Held-out mean JS | Best epoch |
|---|---:|---:|---:|---:|
| `linear` | `N/A` | `0.3445` | `0.0853` | `8` |
| `mlp` | `512` | `0.3270` | `0.0837` | `5` |

Selected pilot family:

- router family: `linear`
- input: `h_4[t]`
- target: `oracle_alpha_logit_vector`
- aggregation: `mean_token_logits_then_softmax`
- readiness cleared: `false`

Relative to the widened MLP baseline:

- `R^2` delta `= +0.0175` for `linear`
- mean JS delta `= +0.0016` for `linear` (worse)

So the family comparison does **not** show a clean nonlinearity win. The
primary metric actually prefers the linear head, while the secondary metric
still prefers the MLP slightly.

Rerun note:

- the exact-command rerun preserved the `summary.json` hash unchanged on MPS:
  `89f3b2682375aa7956f969879af7479e85852fb9`

## Interpretation

This is another bounded Phase 6 result, and it cuts against the “just add a
better nonlinear head” story.

The truthful read is:

- current MLP nonlinearity is not the missing rescue
- the least-bad family on the primary metric is now `linear`
- both families still miss the readiness gate by a large margin
- the next blocker is no longer simple width or the first obvious family split

The key signal is not that linear is suddenly “solved.” It is that once the
input, target, aggregation, and split are frozen, a simple linear head already
matches or slightly exceeds the MLP on held-out `R^2`.

That means the next disciplined Phase 6 move should not be:

- a wider MLP
- a deeper MLP
- another blind family tweak

The next honest question is whether the supervision object is too coarse for the
saved export. In other words: are we now bottlenecked by sequence-level target
granularity rather than by another architecture scalar?

## Limitations

- This is still pilot-only and uses a held-out split inside the pilot export,
  not the confirmatory surface.
- Only two router families were compared.
- The family ordering is metric-mixed: `linear` wins on the primary metric and
  `mlp` stays slightly better on JS, so the write-up should not overstate the
  linear preference beyond the fixed selection rule.

## Next Steps

- Close `resattn-zic` as a stable negative result for nonlinear-family rescue on
  the saved pilot export.
- Take `resattn-but` next to audit whether the remaining Phase 6 blocker is
  supervision granularity rather than another architecture tweak.
