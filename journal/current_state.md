# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: the first token-aware feature comparison is now in place, and the next live need is prompt-level or hybrid predictiveness work rather than more `h_4[t]`-only pooling variants
- Experimental status: the pinned Phase 1 environment exists, the reconstruction checks are green, the development-model runner is live, `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat` is now the best tested token-aware internal summary, and the confirm-split predictiveness blocker still stands because predicted routed loss remains near-neutral to slightly negative
- Critical reminder: keep the oracle-alpha framing honest, preserve the deep-research control additions, and treat the current near-neutral predicted-loss result as insufficient rather than “close enough”
