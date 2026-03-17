# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: loss-aware target selection is now implemented, and the next live need is a more fundamental predictiveness redesign rather than another tuning-rule tweak
- Experimental status: the pinned Phase 1 environment exists, the reconstruction checks are green, the development-model runner is live, prompt-matched per-sequence stability is reported, and the saved predictiveness control plan now tunes by pilot predicted routed-loss improvement first and mean JS second; on the current `gpt2-xl` `8 / 8` split, that still selected the raw-simplex target and reverted to the same slightly negative confirm routed-loss result
- Critical reminder: keep the oracle-alpha framing honest, preserve the deep-research control additions, and stop assuming target/regularization selection is the main blocker for held-out predictiveness
