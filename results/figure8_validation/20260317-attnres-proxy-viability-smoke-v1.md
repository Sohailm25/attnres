# Motivation

`resattn-7hb` needed the smallest honest local AttnRes reproduction that could later support Figure 8 metrics without overclaiming from the published paper alone. This smoke run existed to validate the operational path before spending more time on scale.

# Methods

- Dataset: `wikitext/wikitext-2-raw-v1`
- Tokenization: character-level vocabulary learned from the selected texts
- Train / eval slice: `256` train texts, `64` validation texts
- Architecture: matched `8`-layer tiny baseline and `8`-block Block AttnRes proxy with `d_model=96`, `n_heads=4`, `d_ff=384`, `seq_len=64`
- Optimization: `200` steps, batch size `8`, learning rate `3e-4`, weight decay `0.01`, seed `11`
- Output dir: `results/figure8_validation/20260317-attnres-proxy-viability-smoke-v1/`

# Results

- The run completed on local MPS, wrote checkpoints for both models, and the exact rerun command reused those checkpoints successfully.
- The Block AttnRes proxy slightly beat the matched baseline on eval loss:
  - baseline best eval loss: `2.7338`
  - AttnRes best eval loss: `2.7130`
  - delta: `-0.0207`
- The routing export surface is real and already exposes the prereg Figure 8 summary hooks:
  - deep embedding persistence: `0.1824`
  - mean pre-attn entropy: `1.4953`
  - mean pre-MLP entropy: `1.5561`

# Limitations

- This is a viability smoke, not a claim-bearing Figure 8 result.
- The current pattern read is mixed rather than paper-like: deep embedding persistence is low and the entropy ordering is inverted relative to the prereg expectation.
- The proxy is still character-level and lightly trained, so the negative pattern read is ambiguous between undertraining and a poor proxy regime.

# Next Steps

- Keep the architecture fixed and scale only data plus optimization before changing the proxy design.
- Treat this artifact as superseded by the scaled run once that larger result is registered.
