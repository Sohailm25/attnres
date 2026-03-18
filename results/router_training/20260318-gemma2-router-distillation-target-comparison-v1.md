# ABOUTME: Registers the first target-parameterization comparison for Gemma pilot router distillation on the saved export.
# ABOUTME: Records whether raw-alpha or alpha-logit targets are the right next baseline before aggregation or capacity sweeps.

# Gemma-2 Router-Distillation Target Comparison v1

## Motivation

`resattn-m6r` showed that the first pilot router fit on the saved Gemma export
failed in a very specific way:

- both `h_1[t]` and `h_4[t]` stayed near the uniform-entropy ceiling
- held-out `R^2` stayed negative for both inputs
- the strongest hypothesis was target/objective mismatch rather than missing
  supervision or a locked input failure

`resattn-4hj` keeps the saved export, held-out split discipline, and sequence
aggregation fixed, and changes only the target parameterization:

- `oracle_alpha_vector`
- `oracle_alpha_logit_vector`

The goal is to identify whether the next real blocker is still target geometry
or whether the lane should move next to aggregation or model capacity.

## Methods

- Input dataset:
  `results/router_training/20260318-gemma2-router-distillation-pilot-export-v1/`
- Model:
  saved `google/gemma-2-2b` pilot export only
- Candidate inputs:
  - `h_1[t]`
  - `h_4[t]`
- Candidate targets:
  - `oracle_alpha_vector`
  - `oracle_alpha_logit_vector`
- Router:
  - 2-layer MLP
  - hidden width `256`
- Token-to-sequence aggregation:
  - `mean_token_logits_then_softmax`
- Split:
  - pilot only
  - same stratified `192 / 64` train/eval split reused across all four
    input-target combinations
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

- `results/router_training/20260318-gemma2-router-distillation-target-comparison-v1/summary.json`

## Results

Raw-alpha targets reproduced the `m6r` failure pattern:

- `oracle_alpha_vector` + `h_1[t]`
  - held-out `R^2 = -0.0619`
  - held-out mean JS `= 0.1584`
  - mean predicted entropy `= 3.9693`
- `oracle_alpha_vector` + `h_4[t]`
  - held-out `R^2 = -0.0486`
  - held-out mean JS `= 0.1565`
  - mean predicted entropy `= 3.9684`

Alpha-logit targets changed the pilot materially:

- `oracle_alpha_logit_vector` + `h_1[t]`
  - held-out `R^2 = 0.2722`
  - held-out mean JS `= 0.0872`
  - mean predicted entropy `= 3.6176`
- `oracle_alpha_logit_vector` + `h_4[t]`
  - held-out `R^2 = 0.3028`
  - held-out mean JS `= 0.0835`
  - mean predicted entropy `= 3.6248`

Selected pilot combination:

- target: `oracle_alpha_logit_vector`
- input: `h_4[t]`
- readiness cleared: `false`
- target changed input ranking: `false`

Relative to the raw-alpha path, the logit target:

- moves held-out `R^2` from negative to clearly positive
- cuts mean JS nearly in half
- pulls predicted entropy much closer to the oracle mixture entropy

The exact-command rerun on the same saved output directory changed the summary
hash and nudged the numeric values slightly on MPS, but it did not change any
qualitative conclusion:

- `oracle_alpha_logit_vector` stayed selected
- `h_4[t]` stayed the best input
- the input ranking stayed unchanged across targets
- the readiness gate still failed

## Interpretation

This is a real pilot-design improvement, not a full Phase 6 pass.

The truthful read is:

- target geometry was a genuine blocker
- it is no longer the main blocker
- `h_4[t]` remains provisionally best, but not by enough to treat the input
  choice as scientifically settled
- the next honest blocker is sequence aggregation before model capacity

Why aggregation first:

- the target-only change already moved the fit substantially without changing
  the split or the hidden-state ranking
- the lane still compresses token evidence through one fixed
  `mean_token_logits_then_softmax` rule even though the intended router object is
  per-token
- the remaining error pattern still looks too smooth relative to the oracle
  mixture, which is more suggestive of lossy sequence aggregation than of
  missing supervision

So the next Phase 6 move should not be confirmatory router training and should
not be a blind width sweep. It should be an aggregation comparison on the same
saved pilot export with `oracle_alpha_logit_vector` frozen as the baseline
target.

## Limitations

- This is still pilot-only and uses a held-out split inside the pilot export,
  not the confirmatory surface.
- The router is still tiny relative to the source dimension and no alternative
  aggregation rule was tested here.
- MPS reruns move the exact numbers slightly, so the write-up should treat the
  qualitative ordering as the stable signal, not the fourth decimal place.

## Next Steps

- Close `resattn-4hj` as a partial pass on target geometry.
- Take `resattn-914` next to compare sequence aggregation rules on the same
  saved export with `oracle_alpha_logit_vector` fixed.
