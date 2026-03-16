# Mech Interp Guidance

## Local Guardrails

- Stay in float32 unless a narrower format is explicitly justified.
- Prefer PyTorch-native models with hook access. Do not pivot to GGUF or llama.cpp for interpretability work.
- Keep token positions matched when comparing routing with SAE activations.
- Do not treat a pretty heatmap as evidence; preregistered operationalizations come first.
- Preserve the oracle-alpha framing boundary: fixed computations, no co-adaptation.

## Common Failure Modes

- Wrong residual decomposition makes the routing analysis look better than it is.
- Comparing softmax to unconstrained with mismatched initialization confounds the regime study.
- Clustering random baselines into apparently meaningful groups usually means the metric is wrong.
- A tool-breakage demo that leaves logit-lens behavior monotonic does not support a strong breakage claim.

## Local Runtime Notes

- MacBook Pro MPS is the default target.
- CPU fallback is acceptable when a library op is unsupported on MPS.
- no Modal
