# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: the first constrained alpha-logit predictiveness path is now in place, and the next live need is a compressed target or another lower-dimensional reformulation rather than more simple feature-family changes
- Experimental status: the pinned Phase 1 environment exists, the reconstruction checks are green, the development-model runner is live, the saved control metrics govern predictiveness selection, prompt-matched per-sequence stability is reported, and the first constrained target rerun produced a mixed result: confirm predicted routed loss is now positive on average, but confirm `R^2` and mean JS both regressed and the selected ridge penalty still saturates at `100.0`
- Critical reminder: keep the oracle-alpha framing honest, preserve the deep-research control additions, and do not treat the first positive held-out routed-loss delta as sufficient while alpha-shape recovery is still weak
