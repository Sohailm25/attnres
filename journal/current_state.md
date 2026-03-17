# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: the first compressed target comparison is now in place, and the next live need is loss-aware predictiveness selection rather than another blind target swap
- Experimental status: the pinned Phase 1 environment exists, the reconstruction checks are green, the development-model runner is live, the saved control metrics govern predictiveness selection, prompt-matched per-sequence stability is reported, and the current target comparison is split: the full-source logit target is the only path with positive held-out routed-loss recovery, while the compressed depth-type-band logit target has better confirm `R^2` / mean JS and no ridge saturation
- Critical reminder: keep the oracle-alpha framing honest, preserve the deep-research control additions, and treat routed-loss recovery as the primary objective when comparing predictiveness paths
