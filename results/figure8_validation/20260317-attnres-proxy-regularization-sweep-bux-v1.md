# ABOUTME: Summarizes the regularization-first Figure 8 follow-up on the widened compact-subword `wikitext-103` proxy.
# ABOUTME: Records the last standard-objective stabilization check before objective-level redesign becomes live.

## Motivation

`resattn-bux` existed because the best-checkpoint widened `wikitext-103`
artifact changed the Figure 8 lane diagnosis. The local Block AttnRes proxy was
not simply failing under the standard next-token objective; it was reaching a
healthier best-checkpoint regime and then failing to sustain it. The right next
question was therefore whether matched regularization could stabilize that regime
before the repo moved to any objective-level redesign.

## Methods

- Control reference:
  - `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-bestcheck-v1/summary.json`
- Shared training surface for all new arms:
  - dataset: `wikitext/wikitext-103-raw-v1`
  - tokenization: compact remapped GPT-2 subword ids
  - train / eval slice: `2048` train texts, `512` validation texts
  - architecture: matched `8`-layer tiny baseline and `8`-block Block AttnRes
    proxy with `d_model=160`, `n_heads=4`, `d_ff=640`, `seq_len=64`
  - objective: standard next-token language modeling
  - optimization: `1500` steps, batch size `16`, learning rate `3e-4`, seed
    `11`, best-checkpoint export enabled
- New regularization arms:
  1. `dropout=0.1`, `weight_decay=0.01`
  2. `dropout=0.0`, `weight_decay=0.05`
  3. `dropout=0.1`, `weight_decay=0.05`
- Output root:
  - `results/figure8_validation/20260317-attnres-proxy-regularization-sweep-bux-v1/`
- Resume check:
  - rerunning the exact sweep command reused the saved checkpoints for all three
    arms and finished in `30.67` seconds

## Results

### Control reference

- best-checkpoint baseline loss: `6.5929`
- best-checkpoint AttnRes loss: `6.6315`
- loss delta: `+0.0386`
- deep embedding persistence: `0.1689`
- entropy gap: `-0.0549`

### New regularization arms

- `dropout=0.1`, `weight_decay=0.01`
  - best-checkpoint baseline loss: `6.5857`
  - best-checkpoint AttnRes loss: `6.6235`
  - loss delta: `+0.0378`
  - deep embedding persistence: `0.1855`
  - entropy gap: `-0.0556`
- `dropout=0.0`, `weight_decay=0.05`
  - best-checkpoint baseline loss: `6.5877`
  - best-checkpoint AttnRes loss: `6.6253`
  - loss delta: `+0.0376`
  - deep embedding persistence: `0.1675`
  - entropy gap: `-0.0533`
- `dropout=0.1`, `weight_decay=0.05`
  - best-checkpoint baseline loss: `6.5810`
  - best-checkpoint AttnRes loss: `6.6178`
  - loss delta: `+0.0368`
  - deep embedding persistence: `0.1836`
  - entropy gap: `-0.0546`

## Interpretation

- This is a useful negative result for the regularization-rescue hypothesis.
- All three matched regularization arms stayed in the same narrow loss-gap band
  as the control (`+0.0368` to `+0.0378` versus control `+0.0386`).
- None of the arms restored a routed win.
- None of the arms materially improved the Figure 8 entropy ordering:
  - the best entropy gap only moved from `-0.0549` to `-0.0533`
- Dropout helped deep embedding persistence, but not the real blocker.
- The strongest honest update is therefore:
  - matched regularization can slightly reshape secondary proxy metrics
  - matched regularization does not rescue the widened `wikitext-103` Figure 8
    lane under the standard next-token objective

## Limitations

- This is still a tiny local proxy, not a trained-routing alignment pass.
- The sweep uses one seed and one fixed architecture.
- The regularization arms were intentionally narrow; this result rules out a
  small faithful stabilization fix, not every conceivable optimizer schedule.

## Next Steps

- Close `resattn-bux` as a useful negative result for the regularization-first
  rescue hypothesis.
- Stop spending Figure 8 cycles on further matched regularization sweeps of this
  widened `wikitext-103` regime.
- Move the next Figure 8 decision to `resattn-9fo`, which should choose the
  smallest honest objective-level redesign.
