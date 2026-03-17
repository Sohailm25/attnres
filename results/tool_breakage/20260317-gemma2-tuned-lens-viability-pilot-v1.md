ABOUTME: Summarizes the first held-out Gemma-2 tuned-lens viability pilot for the tool-breakage lane.
ABOUTME: Records what the custom tuned lens does and does not establish before routed-versus-original comparisons begin.

# Gemma-2 Tuned-Lens Viability Pilot v1

## Motivation

`resattn-5k9` exists to answer one narrow question before the full tool-breakage lane starts: can a custom tuned lens on the primary `google/gemma-2-2b` model materially outperform the raw logit lens on held-out factual-recall prompts, so later routed-versus-original comparisons stay on the same model rather than shifting the tuned-lens requirement to a secondary system?

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Train collection: `oracle_alpha_phase1_v1` pilot split (`96` prompts, `1067` positions)
- Eval collection: `tool_breakage_factual_recall_v1` pilot split (`8` prompts, `58` positions)
- Lens family: per-layer low-rank affine translator into the final residual basis
- Rank: `16`
- Training objective: residual-space MSE to final residuals
- Evaluation objective: held-out raw-vs-tuned comparison after the model's own final normalization and unembedding
- Runtime durability: prompt-level residual caches plus a resumable `training_state.pt` checkpoint under `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoints`

## Results

- Mean held-out KL to the final distribution improved from `11.3881` to `3.4580`.
- Mean held-out top-1 agreement improved from `0.1863` to `0.5119`.
- Final-position held-out KL improved from `14.8685` to `7.5721`.
- Final-position held-out top-1 agreement improved only from `0.1010` to `0.1250`.
- The saved output also proves the operational path is usable locally:
  - all `96` training prompts and `8` eval prompts were cached to disk
  - the full run wrote `checkpoints/training_state.pt`
  - rerunning the exact resume command completed successfully against the saved checkpoint tree

The strongest per-layer gains are in the middle-to-late stack. Early layers move from essentially unusable raw lens behavior to materially lower KL, while layers `20` through `24` also show clear top-1 gains. Final-position answer-token recovery remains weak overall, even though the distributional gap narrows substantially.

## Limitations

- This is an original-model-only pilot. It does not yet compare routed versus original behavior.
- The training objective is residual MSE, not a direct logit-space tuned-lens loss. That was an intentional viability simplification to keep the first local Gemma run memory-safe and resumable.
- The evaluation surface is only the saved factual-recall pilot split, so this artifact establishes baseline viability, not the final tool-breakage claim.
- The strongest gain is distributional (`KL`), not final-position top-1. That means later tool-breakage analysis should not silently substitute “better answer-token recovery” for the tuned-lens-aware baseline without further justification.

## Next Steps

- Treat the custom Gemma tuned-lens path as viable enough to keep the primary tool-breakage lane on Gemma-2.
- Keep the next tool-breakage issue explicit about metric choice: decide whether routed-versus-original analysis will use held-out KL as the primary tuned-lens baseline metric or whether the lens objective should be sharpened for final-position factual-recall recovery.
- Reuse this checkpointed path and cached prompt artifacts when the routed-versus-original factual-recall comparisons begin.
