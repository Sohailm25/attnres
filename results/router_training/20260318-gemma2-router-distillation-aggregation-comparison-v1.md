ABOUTME: Compares mean-token versus last-token sequence aggregation on the saved Gemma router-distillation pilot export.
ABOUTME: Keeps the alpha-logit target fixed so the next blocker can be identified cleanly before any capacity sweep.

# Gemma-2 Router-Distillation Aggregation Comparison v1

## Motivation

`resattn-4hj` fixed the biggest visible Phase 6 design flaw by moving the pilot
router target from raw alpha to `oracle_alpha_logit_vector`.

That pass improved the held-out pilot materially, but it still left the lane
below the readiness gate:

- selected baseline: `oracle_alpha_logit_vector + h_4[t]`
- held-out `R^2 = 0.3028`
- held-out mean JS `= 0.0835`
- readiness gate: `R^2 > 0.5`

The next honest question was whether the remaining error was mostly a lossy
sequence aggregation problem.

`resattn-914` keeps the saved export, pilot split discipline, and target
geometry fixed, and changes only the token-to-sequence aggregation rule:

- `mean_token_logits_then_softmax`
- `last_token_logits_then_softmax`

## Methods

- Input dataset:
  `results/router_training/20260318-gemma2-router-distillation-pilot-export-v1/`
- Model:
  saved `google/gemma-2-2b` pilot export only
- Candidate inputs:
  - `h_1[t]`
  - `h_4[t]`
- Frozen target:
  - `oracle_alpha_logit_vector`
- Candidate aggregations:
  - `mean_token_logits_then_softmax`
  - `last_token_logits_then_softmax`
- Router:
  - 2-layer MLP
  - hidden width `256`
- Split:
  - pilot only
  - same stratified `192 / 64` train/eval split reused across all four
    input-aggregation combinations
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

- `results/router_training/20260318-gemma2-router-distillation-aggregation-comparison-v1/summary.json`

## Results

The stronger token-specific alternative did not help.

Held-out pilot results after the exact-command rerun:

| Aggregation | Input | Held-out `R^2` | Held-out mean JS | Best epoch |
|---|---|---:|---:|---:|
| `mean_token_logits_then_softmax` | `h_1[t]` | `0.2891` | `0.0867` | `7` |
| `mean_token_logits_then_softmax` | `h_4[t]` | `0.3116` | `0.0831` | `7` |
| `last_token_logits_then_softmax` | `h_1[t]` | `0.2294` | `0.0979` | `81` |
| `last_token_logits_then_softmax` | `h_4[t]` | `0.2639` | `0.0966` | `14` |

Selected pilot combination:

- aggregation: `mean_token_logits_then_softmax`
- input: `h_4[t]`
- readiness cleared: `false`
- aggregation changed input ranking: `false`

Relative to the current mean-token baseline, last-token aggregation is worse for
both inputs:

- `h_1[t]`
  - `R^2` delta `= -0.0597`
  - mean JS delta `= +0.0112`
- `h_4[t]`
  - `R^2` delta `= -0.0478`
  - mean JS delta `= +0.0135`

So the answer-position-only alternative does not recover the remaining pilot
gap. The current best combination remains:

- `oracle_alpha_logit_vector`
- `mean_token_logits_then_softmax`
- `h_4[t]`

The exact-command rerun changed the saved summary hash and nudged the decimals
again on MPS, but it preserved every qualitative conclusion:

- `mean_token_logits_then_softmax` stayed selected
- `h_4[t]` stayed best
- the ranking did not flip across aggregations
- readiness still failed

## Interpretation

This is a negative result for the hypothesis that sequence aggregation alone is
the main remaining blocker on the saved pilot export.

The truthful read is:

- target geometry mattered
- this simple answer-position aggregation change did not help
- the current mean-token baseline is still the best tested aggregation
- the next honest blocker is now model capacity or router family, not another
  blind aggregation tweak on this fixed export

This does **not** prove aggregation is irrelevant in the absolute sense. It
does show that the most obvious stronger token-specific alternative on the saved
prompt-only export is not the rescue path.

That means the next disciplined move should be:

- keep `oracle_alpha_logit_vector` fixed
- keep `mean_token_logits_then_softmax` fixed
- test capacity next before inventing a richer router family

## Limitations

- This is still pilot-only and uses a held-out split inside the pilot export,
  not the confirmatory surface.
- The aggregation comparison is intentionally small: mean versus last token.
- Because the saved export is prompt-only, a last-token loss should be read as
  evidence against this answer-position-only rescue, not against every possible
  token-aware aggregation.

## Next Steps

- Close `resattn-914`.
- Take `resattn-3ak` next for a bounded capacity comparison with
  `oracle_alpha_logit_vector` and `mean_token_logits_then_softmax` frozen.
