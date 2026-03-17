# ABOUTME: Summarizes the best-checkpoint trajectory follow-up on the widened compact-subword `wikitext-103` Figure 8 proxy.
# ABOUTME: Records whether checkpoint selection, rather than final-checkpoint drift, changes the local trained-routing proxy conclusion.

## Motivation

`resattn-fby` existed because the widened compact-subword `wikitext-103`
artifact was mixed in a suspicious way: a `200`-step calibration was positive,
but the full `1500`-step artifact finished behind the matched baseline while the
runner only preserved the final model plus a scalar `best_eval_loss`. The
question here was narrow: does saving eval trajectory plus the best checkpoint
materially change the Figure 8 read on this regime, or does it only soften the
negative result?

## Methods

- Dataset: `wikitext/wikitext-103-raw-v1`
- Tokenization: compact remapped GPT-2 subword ids
- Train / eval slice: `2048` train texts, `512` validation texts
- Architecture: matched `8`-layer tiny baseline and `8`-block Block AttnRes
  proxy with `d_model=160`, `n_heads=4`, `d_ff=640`, `seq_len=64`
- Optimization: `1500` steps, batch size `16`, learning rate `3e-4`, weight
  decay `0.01`, seed `11`, `checkpoint_every_steps=50`
- New runner support:
  - save `*_eval_history.json`
  - save `*_best_state.pt`
  - export both final-checkpoint and best-checkpoint AttnRes Figure 8 proxy
    summaries in one `summary.json`
- Output dir:
  - `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-bestcheck-v1/`
- Resume check: rerunning the exact finished command reused the saved
  checkpoints and completed cleanly

## Results

- The trajectory support worked as intended:
  - baseline best step `= 900`
  - AttnRes best step `= 850`
  - both models peaked well before the final `1500`-step checkpoint
- Final-checkpoint comparison was meaningfully worse than the best-checkpoint
  comparison:
  - final baseline eval loss `= 6.7562`
  - final AttnRes eval loss `= 6.8834`
  - final loss delta `= +0.1272`
  - best baseline eval loss `= 6.5929`
  - best AttnRes eval loss `= 6.6315`
  - best loss delta `= +0.0386`
- Best-checkpoint Figure 8 metrics improved modestly relative to the final
  checkpoint:
  - deep embedding persistence: `0.1615 -> 0.1689`
  - mean pre-attn entropy: `1.4318 -> 1.4454`
  - mean pre-MLP entropy: `1.4893 -> 1.5003`
  - entropy gap: `-0.0574 -> -0.0549`
- The directional interpretation did not flip:
  - the matched baseline still won at the best checkpoint
  - the entropy ordering remained inverted
  - the routed proxy still did not recover a paper-like Figure 8 surface

## Interpretation

- This is a useful mixed result, not a rescue.
- The earlier runner was understating the regime by relying on the final
  checkpoint alone. Best-checkpoint export clearly matters on this local proxy.
- But checkpoint selection is not enough to clear the Figure 8 problem:
  best-checkpoint metrics improved only modestly, the loss comparison stayed
  negative, and layer-type-specialization still points the wrong way.
- The right update is therefore narrower than “optimization fixed it” and
  stronger than “the whole regime is hopeless”:
  the widened compact-subword `wikitext-103` proxy has an earlier, healthier
  operating region, but the current training setup still does not sustain a
  competitive or paper-like routed solution.

## Limitations

- This is still a tiny local proxy, not a trained-routing alignment pass.
- The trajectory comparison uses one seed and one fixed architecture.
- Best-checkpoint selection improved the read, but it does not reveal which next
  redesign lever is best by itself; it only rules out “final-checkpoint drift
  was the whole problem.”

## Next Steps

- Close `resattn-fby` as a successful trajectory-support issue with a mixed
  scientific result.
- Stop treating checkpoint selection alone as the missing fix for the Figure 8
  lane.
- Make the next Figure 8 follow-up a bounded optimization or objective redesign
  question rather than another blind rerun of the same regime.
