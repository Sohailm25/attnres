# Motivation

`resattn-111` existed to isolate the next likely Figure 8 proxy bottleneck after `resattn-3l6`. The compact-subword run improved the paper-facing Figure 8 metrics relative to the earlier char-level scaled artifact, but it no longer beat the matched baseline at the `1500`-step horizon. This follow-up keeps tokenization, corpus, and horizon fixed and scales width only, so the run answers whether model capacity is the next credible lever.

# Methods

- Dataset: `wikitext/wikitext-2-raw-v1`
- Tokenization: compact remapped GPT-2 subword ids
- Train / eval slice: `2048` train texts, `256` validation texts
- Architecture: matched `8`-layer tiny baseline and `8`-block Block AttnRes proxy with `d_model=160`, `n_heads=4`, `d_ff=640`, `seq_len=64`
- Comparison anchor: the previous compact-subword artifact used the same setup with `d_model=96`, `d_ff=384`
- Optimization: `1500` steps, batch size `16`, learning rate `3e-4`, weight decay `0.01`, seed `11`
- Output dir: `results/figure8_validation/20260317-attnres-proxy-compact-subword-capacity-v1/`
- Resume check: rerunning the exact final command after completion reused the saved checkpoints and finished in `7.52` seconds

# Results

- Width alone did not restore the loss comparison:
  - widened baseline best eval loss: `6.5899`
  - widened AttnRes best eval loss: `6.6496`
  - delta: `+0.0598`
- Both models improved relative to the smaller compact-subword run, but the baseline improved more:
  - baseline improved from `6.6463` to `6.5899`
  - AttnRes improved from `6.6764` to `6.6496`
- The Figure-8-facing read stayed mixed:
  - deep embedding persistence improved from `0.1364` to `0.1515`
  - mean pre-attn entropy: `1.4381`
  - mean pre-MLP entropy: `1.4942`
  - entropy gap regressed from `-0.0349` to `-0.0561`

# Limitations

- This is a width-isolation artifact, not a Figure 8 alignment pass.
- The negative loss delta at a fixed `1500`-step horizon does not prove capacity is unhelpful. The widened proxy may still be undertrained relative to the matched baseline.
- The entropy ordering remained inverted, so the run does not clear the prereg layer-type-specialization expectation.

# Next Steps

- Close `resattn-111` as a useful bottleneck-isolation result rather than an open-ended scaling issue.
- Keep compact subword and the widened `d_model=160`, `d_ff=640` regime fixed.
- Move the next Figure 8 follow-up to optimization horizon (`resattn-jci`) rather than more tokenization or width changes.
