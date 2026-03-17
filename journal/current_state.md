# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: the larger `registry_v3` split kept the logit path selected, so the next live need is to scale that same path toward the prereg-sized gate rather than redesign the predictor again
- Experimental status: the pinned Phase 1 environment exists, the reconstruction checks are green, the development-model runner is live, prompt-matched per-sequence stability is reported, and the saved predictiveness control plan now tunes by pilot predicted routed-loss improvement first and mean JS second; with `prompts/registry_v3.yaml`, the same loss-aware target comparison still selects `oracle_alpha_logit_vector` and recovers `+0.1277` nats over uniform on a `32 / 16` split
- Critical reminder: keep the oracle-alpha framing honest, preserve the deep-research control additions, and treat the positive `32 / 16` result as strong encouragement to scale, not as a claim-bearing endpoint
