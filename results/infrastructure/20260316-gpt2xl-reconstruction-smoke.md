# Motivation

The repo needed a real backend-backed check, not just toy linear-algebra validation, before broader oracle-alpha work could rely on cached sublayer outputs.

# Methods

- Loaded the configured development model `gpt2-xl` through TransformerLens on `mps`.
- Cached `hook_embed`, `hook_pos_embed`, per-layer `attn_out`, `mlp_out`, and residual-state hooks.
- Reconstructed the final pre-`ln_final` residual by accumulating the cached write vectors in forward order.
- Applied the model's own `ln_final` and `unembed` path to the reconstructed mixture and compared it with the original logits.
- Checked per-layer residual identities using `resid_pre`, `resid_mid`, `resid_post`, `attn_out`, and `mlp_out`.

# Results

- `final_residual_max_abs_error = 0.0`
- `uniform_logits_max_abs_error = 0.0`
- All per-layer `resid_mid` and `resid_post` identity checks were exact on this prompt.
- The artifact JSON is saved at `results/infrastructure/20260316-gpt2xl-reconstruction-smoke.json`.

# Limitations

- This is still a smoke check on a single prompt, not a broader stability or null-model study.
- The result validates the reconstruction path on the configured development model, but it does not yet cover the primary Gemma-2 lane.

# Next Steps

- Keep claim-bearing oracle-alpha work gated on the pilot/confirmatory split and the identifiability controls.
- Reuse the new validation module when wiring the broader oracle-alpha infrastructure and when extending the check to Gemma-2.
