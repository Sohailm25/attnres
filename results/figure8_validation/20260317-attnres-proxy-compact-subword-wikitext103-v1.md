# Motivation

`resattn-7y4` existed to run the corpus-first Figure 8 follow-up chosen in
`resattn-du2`. The concrete question was whether moving the widened
compact-subword local Block AttnRes proxy from `wikitext-2` to the larger
same-family `wikitext-103` corpus would restore competitiveness with the matched
baseline and strengthen the Figure-8-facing metric surface without changing the
local architecture or objective.

# Methods

- Dataset: `wikitext/wikitext-103-raw-v1`
- Tokenization: compact remapped GPT-2 subword ids
- Train / eval slice: `2048` train texts, `512` validation texts
- Architecture: matched `8`-layer tiny baseline and `8`-block Block AttnRes
  proxy with `d_model=160`, `n_heads=4`, `d_ff=640`, `seq_len=64`
- Optimization:
  - calibration: `200` steps
  - full run: `1500` steps
- Shared training config: batch size `16`, learning rate `3e-4`, weight decay
  `0.01`, seed `11`, `checkpoint_every_steps=50`
- Output dirs:
  - calibration:
    `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-calibration/`
  - full run:
    `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-v1/`
- Resume check: rerunning the exact completed `1500`-step command reused the
  saved checkpoints and finished in `9.28` seconds

# Results

- The `200`-step calibration was promising:
  - baseline eval loss: `6.9299`
  - AttnRes eval loss: `6.8764`
  - delta: `-0.0534`
  - deep embedding persistence: `0.2022`
- The full `1500`-step corpus-first run remained mixed:
  - baseline best eval loss: `6.5929`
  - AttnRes best eval loss: `6.6315`
  - delta: `+0.0386`
- Relative to the widened `wikitext-2` compact-subword artifact:
  - the loss gap improved from `+0.0598` to `+0.0386`
  - deep embedding persistence improved from `0.1515` to `0.1615`
  - the entropy gap stayed effectively unchanged and negative:
    - `wikitext-2`: `1.4381 - 1.4942 = -0.0561`
    - `wikitext-103`: `1.4318 - 1.4893 = -0.0574`
- The saved Figure 8 surface is therefore still incomplete:
  - deep embedding persistence improved modestly
  - mean pre-attn entropy remained below mean pre-MLP entropy
  - the matched baseline still won at the full horizon
- The run also exposed a new optimization-shape fact:
  - the early `200`-step read was positive
  - the final `1500`-step read was negative versus the baseline
  - the current runner does not preserve checkpoint-level eval history or the
    best-model checkpoint needed to inspect that crossover directly

# Limitations

- This is still not a Figure 8 alignment pass.
- The corpus-first change improved the widened compact-subword regime, but it
  did not restore a routed win at the main `1500`-step horizon.
- Because the runner only preserves the final model plus a scalar
  `best_eval_loss`, the repo cannot yet inspect whether earlier more competitive
  checkpoints also express a stronger Figure 8 pattern surface.
- The larger validation slice lowers eval noise, but it also means the top-line
  losses are not directly comparable numerically to the older `256`-text
  validation artifacts.

# Next Steps

- Close `resattn-7y4` as a useful mixed corpus-first result rather than as a
  solved Figure 8 follow-up.
- Use `resattn-fby` to add best-checkpoint / eval-trajectory support before the
  next Figure 8 optimization follow-up.
- Do not treat this as evidence that more horizon alone is the next answer.
- If effort shifts away from the Figure 8 lane next, `resattn-73l` is now the
  stronger overall repo follow-up.
