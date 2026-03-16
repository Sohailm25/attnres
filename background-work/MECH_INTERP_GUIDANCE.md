# Mech Interp Guidance

## Local Guardrails

- Stay in float32 unless a narrower format is explicitly justified.
- Prefer PyTorch-native models with hook access. Do not pivot to GGUF or llama.cpp for interpretability work.
- Keep token positions matched when comparing routing with SAE activations.
- Do not treat a pretty heatmap as evidence; preregistered operationalizations come first.
- Preserve the oracle-alpha framing boundary: fixed computations, no co-adaptation.
- Use sequence-level aggregation for claim-bearing hypothesis tests unless a dependence-aware token-level method is explicitly justified.
- Use the model's shared final normalization factor when decomposing routed mixtures; do not normalize each source independently.
- Cache sublayer outputs for claim-bearing runs that touch Figure 8 or layer-type specialization. `resid_post` alone is a smoke-test shortcut, not a final method.

## Common Failure Modes

- Wrong residual decomposition makes the routing analysis look better than it is.
- Comparing softmax to unconstrained with mismatched initialization confounds the regime study.
- Clustering random baselines into apparently meaningful groups usually means the metric is wrong.
- A tool-breakage demo that leaves logit-lens behavior monotonic does not support a strong breakage claim.
- Using Ward linkage directly on Jensen-Shannon distances is invalid and can manufacture structure.
- Validating uniform routing against `logits / L` will flag a false bug; the correct target is the original logits after the model's own final normalization.
- Computing "exact" per-source contributions with per-source LayerNorm or RMSNorm is wrong; exactness requires the shared routed-mixture normalization factor.
- Treating token losses as independent samples inflates significance through pseudoreplication.

## Local Runtime Notes

- MacBook Pro MPS is the default target.
- CPU fallback is acceptable when a library op is unsupported on MPS.
- no Modal
