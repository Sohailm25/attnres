ABOUTME: Compares hidden-width capacity on the saved Gemma router-distillation pilot export.
ABOUTME: Keeps the alpha-logit target and mean-token aggregation fixed so width can be tested cleanly before a router-family redesign.

# Gemma-2 Router-Distillation Capacity Comparison v1

## Motivation

`resattn-914` closed the most obvious sequence-aggregation escape hatch on the
saved Gemma pilot export.

The retained baseline was:

- target: `oracle_alpha_logit_vector`
- aggregation: `mean_token_logits_then_softmax`
- best input: `h_4[t]`
- held-out `R^2 = 0.3116`
- held-out mean JS `= 0.0831`

That left two honest hypotheses for the next pilot blocker:

- the current router is under-capacity
- the router family itself is the wrong next lever

`resattn-3ak` tests the smallest width-only comparison that can discriminate
between those stories.

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
- Frozen aggregation:
  - `mean_token_logits_then_softmax`
- Candidate hidden widths:
  - `256`
  - `512`
- Router:
  - 2-layer MLP
- Split:
  - pilot only
  - same stratified `192 / 64` train/eval split reused across all four
    input-width combinations
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

- `results/router_training/20260318-gemma2-router-distillation-capacity-comparison-v1/summary.json`

## Results

Held-out pilot results after the exact-command rerun:

| Hidden width | Input | Held-out `R^2` | Held-out mean JS | Best epoch |
|---|---|---:|---:|---:|
| `256` | `h_1[t]` | `0.2831` | `0.0874` | `7` |
| `256` | `h_4[t]` | `0.3236` | `0.0837` | `7` |
| `512` | `h_1[t]` | `0.2692` | `0.0882` | `5` |
| `512` | `h_4[t]` | `0.3270` | `0.0837` | `5` |

Selected pilot combination:

- hidden width: `512`
- input: `h_4[t]`
- readiness cleared: `false`
- capacity changed input ranking: `false`

Relative to the current `256` baseline:

- `h_4[t]`
  - `R^2` delta `= +0.0034`
  - mean JS delta `= -0.0000` (improvement `= 0.0000087`)
- `h_1[t]`
  - `R^2` delta `= -0.0139`
  - mean JS delta `= +0.0008`

So width helped only slightly, only on `h_4[t]`, and not nearly enough to move
the readiness story.

Implementation note:

- this pass also fixed a real reproducibility bug in the router fitter by
  seeding Torch model initialization from the run seed instead of only seeding
  the split and batch order
- after that fix, the exact-command rerun preserved the
  `summary.json` hash unchanged on MPS:
  `0cd7c7bee688d503d44f06b54ea9200d634c8b57`

## Interpretation

This is a real but non-rescuing capacity result.

The truthful read is:

- hidden width is not irrelevant
- hidden width alone is not the main remaining blocker
- `h_4[t]` remains the best tested input
- the lane is still far below the prereg readiness gate

The important negative evidence is not merely that `512` failed to clear
`R^2 > 0.5`. It is that widening the same router family only bought a few
thousandths on the best path while leaving the input ranking unchanged.

That means the next disciplined Phase 6 move should be:

- keep `oracle_alpha_logit_vector` fixed
- keep `mean_token_logits_then_softmax` fixed
- keep `h_4[t]` as the current baseline input
- compare router families rather than running another blind width sweep

The smallest honest next question is whether a simpler or differently structured
router head changes the held-out fit materially on this same saved pilot split.

## Limitations

- This is still pilot-only and uses a held-out split inside the pilot export,
  not the confirmatory surface.
- Only two widths were compared, and both stay within the same 2-layer MLP
  family.
- The stable rerun is specific to this fixed-seed path after the seeding fix;
  the write-up should still treat the qualitative ordering as the important
  scientific signal.

## Next Steps

- Close `resattn-3ak` as a mixed width-only result.
- Take `resattn-zic` next to compare router families on the same saved pilot
  export before any wider capacity or confirmatory router-training run.
