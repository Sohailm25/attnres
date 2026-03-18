ABOUTME: Summarizes the primary-spine Gemma-2 model-backed reconstruction smoke on local MPS.
ABOUTME: Records whether cached sublayer writes reconstruct the original Gemma-2 logits exactly before any primary-model oracle-alpha run.

# Motivation

The strongest positive oracle-alpha result in the repo still lived on the
development model. Before primary-model oracle-alpha work could start, the repo
needed to verify that the same model-backed reconstruction path used on
`gpt2-xl` also works on the primary `google/gemma-2-2b` spine.

# Methods

- Loaded `google/gemma-2-2b` through TransformerLens on local `mps`.
- Cached `hook_embed`, `hook_pos_embed`, per-layer `attn_out`, `mlp_out`, and
  residual-state hooks.
- Reconstructed the final pre-final-norm residual by accumulating the cached
  write vectors in forward order.
- Applied the model's own final normalization and unembedding to the
  reconstructed mixture and compared it with the original logits.
- Checked per-layer residual identities using `resid_pre`, `resid_mid`,
  `resid_post`, `attn_out`, and `mlp_out`.

# Results

- `num_sources = 53`
- `final_residual_max_abs_error = 0.0`
- `uniform_logits_max_abs_error = 0.0`
- All `26` per-layer `resid_mid` and `resid_post` identity checks were exact on
  this prompt.
- The artifact JSON is saved at
  `results/infrastructure/20260317-gemma2-reconstruction-smoke.json`.

# Limitations

- This is still a single-prompt smoke check, not a broader stability or null
  study.
- The result clears primary-spine reconstruction readiness, but it does not yet
  run the primary-model oracle-alpha optimizer itself.

# Next Steps

- Close `resattn-2s0` as a successful backend-readiness pass on the primary
  spine.
- Move the next scientific priority to `resattn-7cs`, the first bounded
  primary-model Gemma oracle-alpha feasibility slice.
