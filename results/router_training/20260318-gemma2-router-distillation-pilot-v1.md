ABOUTME: Registers the first pilot-only Gemma router-distillation comparison on the saved per-token export.
ABOUTME: Records the held-out h_1[t] versus h_4[t] comparison and the next modeling blocker before confirmatory router training.

# Gemma-2 Router-Distillation Pilot v1

## Motivation

`resattn-1ot` removed the Phase 6 data blocker by exporting pilot-only token ids,
`h_1[t]`, `h_4[t]`, and sequence-level `final_alpha` targets from the saved
Gemma `registry_v5` campaign.

`resattn-m6r` asks the first honest modeling question on top of that export:

- can a small prereg-style pilot router recover useful held-out oracle-alpha
  structure at all
- does `h_1[t]` or `h_4[t]` look better on the same held-out pilot slice
- is the lane ready for confirmatory router training and later `w_l`-analog
  geometry, or is there still a pilot-stage modeling blocker

## Methods

- Input dataset:
  `results/router_training/20260318-gemma2-router-distillation-pilot-export-v1/`
- Model:
  `google/gemma-2-2b` saved export only; no new forward-pass cache export
- Candidate inputs:
  - `h_1[t]`
  - `h_4[t]`
- Router:
  - 2-layer MLP
  - hidden width `256`
- Objective:
  - MSE on the saved sequence-level `final_alpha` vectors
- Token-to-sequence aggregation:
  - `mean_token_logits_then_softmax`
- Split:
  - pilot only
  - stratified by subcategory tag
  - `192` train prompts / `64` held-out eval prompts
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

- `results/router_training/20260318-gemma2-router-distillation-pilot-v1/summary.json`

## Results

The pilot router fit ran cleanly, but it did not clear the readiness gate.

Held-out comparison:

- `h_1[t]`
  - best epoch: `1`
  - train loss: `0.0003731`
  - eval loss: `0.0003803`
  - eval `R^2 = -0.0599`
  - eval mean JS `= 0.1582`
- `h_4[t]`
  - best epoch: `2`
  - train loss: `0.0003694`
  - eval loss: `0.0003776`
  - eval `R^2 = -0.0524`
  - eval mean JS `= 0.1572`

So `h_4[t]` narrowly won the pilot comparison, but only by a tiny margin, and
both candidates failed badly relative to the prereg readiness target.

The important diagnostic is collapse:

- mean predicted entropy:
  - `h_1[t] = 3.9692`
  - `h_4[t] = 3.9690`
- mean oracle entropy:
  - `3.5135`

A rerun after compacting the saved summary artifact moved the exact metrics
slightly on MPS, but the qualitative result stayed the same:

- `h_4[t]` remained the narrow winner
- both inputs still failed the readiness gate
- both predictions still collapsed near the uniform-entropy ceiling

With `53` sources, the pilot router is predicting an almost-uniform mixture on
held-out prompts rather than recovering prompt-specific routing structure.

## Interpretation

This is not a data-readiness failure anymore. It is a pilot modeling failure.

The truthful read is:

- do not lock the input choice yet
- do not proceed to confirmatory router training yet
- do not start `w_l`-analog geometry analysis yet

`h_4[t]` is provisionally better than `h_1[t]`, but the gap is too small, too
low-quality, and too sensitive to minor rerun drift to treat as a real locked
choice. The stronger signal is that the current raw-alpha MSE setup plus
`mean_token_logits_then_softmax` aggregation collapses toward a near-static
average mixture.

That points the next honest follow-up at pilot-stage modeling design, not more
data plumbing.

## Limitations

- This is pilot-only and uses a held-out split inside the pilot export, not the
  confirmatory surface.
- Only one token-to-sequence aggregation rule was tested here.
- The objective is raw-alpha MSE only; no logit-space target or end-to-end LM
  refinement was attempted.
- No learned query geometry is analyzed here because the pilot router does not
  yet clear the readiness gate.

## Next Steps

- Close `resattn-m6r`.
- Start `resattn-4hj`: compare target parameterizations for the same saved
  pilot export before revisiting the input lock.
