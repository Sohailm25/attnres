ABOUTME: Compares richer token/span supervision objectives on the saved Gemma router-distillation pilot export.
ABOUTME: Reuses the frozen linear baseline and family-summary split so only the supervision objective changes.

# Gemma-2 Router-Distillation Supervision-Objective Comparison v1

## Motivation

`resattn-but` showed that the retained Phase 6 baseline

- `linear + h_4[t] + oracle_alpha_logit_vector + mean_token_logits_then_softmax`

was no longer mainly bottlenecked by simple width or family tweaks. The
remaining held-out errors were materially stratified by prompt regime and only
weakly explained by coarse scalar surrogates.

That diagnosis was useful, but it still left an open question:

- does richer token/span supervision actually buy us anything on the same saved
  pilot surface?

`resattn-tqn` tests the smallest honest version of that question without
pretending we already have tokenwise oracle targets:

- keep the model family fixed
- keep the input fixed
- keep the target fixed
- keep the inference aggregation fixed
- change only the training supervision objective

## Methods

- Input dataset:
  `results/router_training/20260318-gemma2-router-distillation-pilot-export-v1/`
- Frozen baseline / split source:
  `results/router_training/20260318-gemma2-router-distillation-family-comparison-v1/summary.json`
- Frozen baseline:
  - router family: `linear`
  - input: `h_4[t]`
  - target: `oracle_alpha_logit_vector`
  - inference aggregation: `mean_token_logits_then_softmax`
- Frozen split:
  - exact `192 / 64` train/eval prompt IDs from the saved family-comparison
    artifact
- Candidate supervision objectives:
  - `sequence_target_mse`
  - `all_tokens_target_mse`
  - `last_third_tokens_target_mse`
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

Machine-readable artifacts:

- `results/router_training/20260318-gemma2-router-distillation-supervision-objective-comparison-v1/summary.json`
- `results/router_training/20260318-gemma2-router-distillation-supervision-objective-comparison-v1/objective_audits.json`

Implementation checks:

- exact-command rerun preserved the `summary.json` hash on MPS:
  `13cbb0d9d69f05cdb5d97e0ed69b66d0f72ea350`
- the frozen `sequence_target_mse` row matched the saved family baseline exactly:
  - held-out `R^2 = 0.3445`
  - held-out mean JS `= 0.0853`

## Results

Held-out pilot comparison:

| Supervision objective | Held-out `R^2` | Held-out mean JS | Best epoch |
|---|---:|---:|---:|
| `sequence_target_mse` | `0.3445` | `0.0853` | `8` |
| `all_tokens_target_mse` | `0.4076` | `0.0797` | `117` |
| `last_third_tokens_target_mse` | `0.2140` | `0.1103` | `29` |

Selected objective:

- `all_tokens_target_mse`
- readiness cleared: `false`

Relative to the retained sequence-level baseline:

- `all_tokens_target_mse`
  - `R^2` delta `= +0.0630`
  - mean JS delta `= -0.0057`
- `last_third_tokens_target_mse`
  - `R^2` delta `= -0.1306`
  - mean JS delta `= +0.0250`

Per-stratum mean-JS deltas from `objective_audits.json`:

| Stratum | Baseline JS | `all_tokens` JS | Delta | `last_third` JS | Delta |
|---|---:|---:|---:|---:|---:|
| `stratum_factual_recall` | `0.0409` | `0.0395` | `-0.0014` | `0.0751` | `+0.0343` |
| `stratum_reasoning_math` | `0.0576` | `0.0522` | `-0.0054` | `0.0613` | `+0.0037` |
| `stratum_code_procedural` | `0.0627` | `0.0568` | `-0.0059` | `0.0745` | `+0.0118` |
| `stratum_general_text` | `0.0754` | `0.0723` | `-0.0031` | `0.0949` | `+0.0195` |

So the richer all-token objective improved every stratum modestly, while the
last-third objective hurt every stratum and hurt factual recall badly.

## Interpretation

This is the first Phase 6 result that turns the supervision diagnosis into a
real improvement path.

The truthful read is:

- denser token-level supervision helps on the frozen baseline
- that help is broad rather than confined to one stratum
- end-concentrated supervision is the wrong simplification here
- the current lane is still below readiness, but the gap is smaller and the
  next honest lever has changed

The important nuance is what this result does **not** mean:

- it does **not** mean we now have tokenwise oracle supervision
- it does **not** mean the router should only care about the final span
- it does **not** yet reopen broad architecture search by inertia

What it does mean is narrower and useful:

- repeating the same sequence-level target across all tokens materially improves
  held-out fit
- the earlier supervision-granularity diagnosis now has payoff
- the right next Phase 6 question is how the best current architecture behaves
  under this denser supervision surface, or whether a more faithful tokenwise
  target can improve further

The negative `last_third_tokens_target_mse` result also matters. It argues
against the easy story that the missing signal lives only near the end of the
prompt. The current gain looks more like distributed token coverage than
answer-position localization.

## Limitations

- The richer objectives still reuse the same sequence-level oracle target, so
  this is denser/localized supervision, not tokenwise oracle supervision.
- This is still pilot-only and still uses a held-out split inside the pilot
  export, not the confirmatory surface.
- Even the improved `all_tokens` objective remains below the prereg readiness
  gate (`R^2 > 0.5`).

## Next Steps

- Close `resattn-tqn` as a real positive result for denser all-token
  supervision on the frozen Gemma pilot split.
- Treat `all_tokens_target_mse` as the new least-bad Phase 6 supervision
  baseline on this saved export.
- Take one bounded follow-up that re-tests the next architecture question under
  the improved supervision objective rather than under the old sequence-level
  baseline.
