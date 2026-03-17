# Motivation

`resattn-3l6` existed to decide the next credible Figure 8 proxy regime after the mixed char-level `7hb` artifacts. The concrete question was whether the next no-regret change was simply more char-level optimization or a more realistic tokenization regime on the same local `8`-block proxy.

# Methods

- Dataset: `wikitext/wikitext-2-raw-v1`
- Tokenization: compact remapped GPT-2 subword ids
- Train / eval slice: `2048` train texts, `256` validation texts
- Architecture: matched `8`-layer tiny baseline and `8`-block Block AttnRes proxy with `d_model=96`, `n_heads=4`, `d_ff=384`, `seq_len=64`
- Optimization:
  - smoke: `200` steps
  - scale continuation: resumed the same checkpoints to `1500` steps
- Output dir: `results/figure8_validation/20260317-attnres-proxy-compact-subword-v1/`
- Resume check: the `200`-step smoke wrote resumable checkpoints, and the `1500`-step continuation reused the same output directory

# Results

- The compact-subword smoke was healthy and initially beat the matched baseline:
  - baseline best eval loss: `7.0926`
  - AttnRes best eval loss: `7.0459`
  - delta: `-0.0467`
- At the `1500`-step horizon, the Figure 8 proxy metrics moved in the right direction relative to the char-level scaled run, but the loss comparison flipped:
  - compact-subword baseline best eval loss: `6.6463`
  - compact-subword AttnRes best eval loss: `6.6764`
  - delta: `+0.0300`
- Relative to the char-level scaled artifact:
  - deep embedding persistence improved from `0.1049` to `0.1364`
  - the entropy gap improved from `-0.0590` to `-0.0349`
  - the entropy ordering still remained inverted (`mean_pre_attn_entropy < mean_pre_mlp_entropy`)

# Limitations

- This is a regime-decision artifact, not a Figure 8 alignment pass.
- Tokenizer realism alone did not solve the proxy problem. The pattern surface improved modestly, but the AttnRes proxy no longer beat the matched baseline at the longer horizon.
- Because both the corpus and model remain small, the current negative result cannot yet distinguish “subword regime still underpowered” from “the remaining bottleneck is corpus or capacity.”

# Next Steps

- Keep compact-subword tokenization as the preferred proxy regime over character-level modeling.
- Do not spend more time on additional char-level reruns.
- The next Figure 8 follow-up should keep the compact-subword regime and scale either model capacity or optimization horizon, rather than revisiting tokenization again.
