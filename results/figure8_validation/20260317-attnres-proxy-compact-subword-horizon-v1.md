# Motivation

`resattn-jci` existed to test whether the widened compact-subword Figure 8 proxy from `resattn-111` was simply under-optimized. The run keeps tokenization, width, corpus, and seed fixed and extends the widened `d_model=160`, `d_ff=640` regime from `1500` to `4500` total steps by resuming the saved checkpoints.

# Methods

- Dataset: `wikitext/wikitext-2-raw-v1`
- Tokenization: compact remapped GPT-2 subword ids
- Train / eval slice: `2048` train texts, `256` validation texts
- Architecture: matched `8`-layer tiny baseline and `8`-block Block AttnRes proxy with `d_model=160`, `n_heads=4`, `d_ff=640`, `seq_len=64`
- Optimization: resumed from the saved `1500`-step checkpoints and continued to `4500` total steps, batch size `16`, learning rate `3e-4`, weight decay `0.01`, seed `11`
- Output dir: `results/figure8_validation/20260317-attnres-proxy-compact-subword-horizon-v1/`
- Resume check: rerunning the exact final command after completion reused the saved checkpoints and finished in `7.96` seconds

# Results

- The longer horizon did not improve the widened proxy's best loss at all:
  - baseline best eval loss: `6.5899`
  - AttnRes best eval loss: `6.6496`
  - delta: `+0.0598`
- Relative to the earlier widened compact-subword artifact:
  - baseline best eval loss stayed unchanged at `6.5899`
  - AttnRes best eval loss stayed unchanged at `6.6496`
- The Figure-8-facing metrics remained mixed and did not strengthen meaningfully:
  - deep embedding persistence moved from `0.1515` to `0.1415`
  - mean pre-attn entropy: `1.4267`
  - mean pre-MLP entropy: `1.4727`
  - entropy gap remained negative at `-0.0460`

# Limitations

- This is a horizon-isolation artifact, not a Figure 8 alignment pass.
- Because the best eval losses remained unchanged for both models, this result should be read as evidence against “more optimization on the same widened regime” rather than as evidence against the broader local AttnRes proxy idea.
- The proxy still uses a tiny local `8`-block setting on Wikitext, so the next redesign question should focus on proxy regime rather than overclaiming from this negative result.

# Next Steps

- Close `resattn-jci` as a negative result for the “it just needs more optimization budget” hypothesis.
- Stop scaling the current widened compact-subword regime along horizon alone.
- Move the next Figure 8 follow-up to `resattn-du2`, a bounded redesign decision over corpus, proxy objective, or local architecture.
