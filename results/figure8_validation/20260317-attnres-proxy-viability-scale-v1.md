# Motivation

The smoke run proved that the local `8`-block Block AttnRes proxy trains, checkpoints, and exports routing summaries. This scaled run tests whether the same proxy starts to show stronger training signal and a more informative Figure 8 pattern surface when only data and optimization scale are increased.

# Methods

- Dataset: `wikitext/wikitext-2-raw-v1`
- Tokenization: character-level vocabulary with `vocab_size=512`
- Train / eval slice: `2048` train texts, `256` validation texts
- Architecture: matched `8`-layer tiny baseline and `8`-block Block AttnRes proxy with `d_model=96`, `n_heads=4`, `d_ff=384`, `seq_len=64`
- Optimization: `1500` steps, batch size `16`, learning rate `3e-4`, weight decay `0.01`, seed `11`
- Output dir: `results/figure8_validation/20260317-attnres-proxy-viability-scale-v1/`
- Resume check: rerunning the exact command reused the saved checkpoints and completed in `9.822` seconds

# Results

- The scaled proxy remained stable and outperformed the matched baseline:
  - baseline best eval loss: `2.1458`
  - AttnRes best eval loss: `2.1130`
  - delta: `-0.0327`
- The learned routing surface is non-trivial and increasingly diffuse at deeper targets:
  - `0_pre_mlp` locality: `0.7767`
  - `1_pre_attn` locality: `0.7148`
  - `7_pre_attn` locality: `0.1041`
  - `final_output` top source: `block_6` with mean weight `0.2224`
- The Figure 8 proxy metrics remain mixed rather than clearly aligned:
  - deep embedding persistence: `0.1049`
  - mean pre-attn entropy: `1.4674`
  - mean pre-MLP entropy: `1.5264`
  - entropy gap stayed negative (`-0.0590`)

# Limitations

- This is still not a strong Figure 8 alignment pass. The local proxy is operationally real, but the saved pattern surface does not yet justify claims that the trained-routing proxy matches the published AttnRes Figure 8 signatures.
- The regime is still character-level Wikitext on a very small model, so the negative or mixed Figure 8 read may reflect proxy choice rather than the absence of the pattern in trained depth routing.
- The first larger launch failed because the wider text slice exceeded `vocab_size=256`; the saved scaled artifact uses `vocab_size=512` and is the only run that should be interpreted.

# Next Steps

- Close `resattn-7hb` as a build-and-viability success, not as a Figure 8 alignment success.
- Decide the next credible proxy regime before spending more compute: either more optimization on this setup or a more realistic tokenizer / corpus regime.
