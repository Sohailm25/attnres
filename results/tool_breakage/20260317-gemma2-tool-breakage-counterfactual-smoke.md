ABOUTME: Records the tiny runner-smoke artifact for the Gemma dynamic counterfactual path.
ABOUTME: Preserves the first `2`-prompt checkpointed control-arm run as implementation evidence, not as a claim-bearing result.

# Gemma-2 Tool-Breakage Counterfactual Smoke

## Motivation

Verify that the new `resattn-g09` control-arm runner reuses the saved baseline prompt results, writes checkpointed prompt artifacts, and produces sane routed-versus-control deltas before the locked confirm launch.

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: first `2` prompts from the locked factual-recall confirm split
- Control arms:
  - `prompt_permuted_alpha`
  - `pilot_mean_alpha`
- Output directory: `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-smoke/`

## Results

- The runner wrote per-prompt checkpoints and a resumable `summary.json`.
- Routed tuned KL stayed worse than the fixed `pilot_mean_alpha` control (`routed minus control = +1.2240`).
- The prompt-permuted control was more damaging than the prompt-matched route on this tiny subset (`routed minus control = -0.7417`).

## Limitations

- This is only a `2`-prompt smoke artifact.
- The prompt-permuted control is just a single donor swap here, so the result is too pairing-sensitive to interpret scientifically.

## Next Steps

- Use the smoke only as implementation evidence.
- Judge the dynamic control on the full locked confirm run, not on this tiny subset.
