# Motivation

This calibration existed to verify that the widened compact-subword
`wikitext-103` Figure 8 regime stayed inside the existing `vocab_size=20000`
envelope, wrote resumable checkpoints cleanly, and provided an early read before
the full `1500`-step corpus-first run.

# Methods

- Dataset: `wikitext/wikitext-103-raw-v1`
- Tokenization: compact remapped GPT-2 subword ids
- Train / eval slice: `2048` train texts, `512` validation texts
- Architecture: matched `8`-layer tiny baseline and `8`-block Block AttnRes
  proxy with `d_model=160`, `n_heads=4`, `d_ff=640`, `seq_len=64`
- Optimization: `200` steps, batch size `16`, learning rate `3e-4`, weight
  decay `0.01`, seed `11`
- Output dir:
  `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-calibration/`

# Results

- The widened `wikitext-103` regime stayed within the existing compact-subword
  vocabulary cap:
  - observed compact vocabulary size: `19250`
- The early loss read was positive for the routed proxy:
  - baseline eval loss: `6.9299`
  - AttnRes eval loss: `6.8764`
  - delta: `-0.0534`
- The early Figure 8 read was also stronger than the earlier widened
  `wikitext-2` artifact on deep embedding persistence:
  - deep embedding persistence: `0.2022`
- The entropy ordering still remained inverted:
  - mean pre-attn entropy: `1.4557`
  - mean pre-MLP entropy: `1.5164`

# Limitations

- This is an operational and early-training artifact, not the main
  corpus-first result.
- The run is too short to support any direct Figure 8 claim.
- The main value of this artifact is showing that the widened `wikitext-103`
  regime starts in a healthier place than the earlier widened `wikitext-2`
  configuration.

# Next Steps

- Use the full `1500`-step `wikitext-103` run as the main scientific artifact.
- Keep this calibration as evidence that the larger same-family corpus changes
  the early optimization behavior before the longer horizon changes the read.
