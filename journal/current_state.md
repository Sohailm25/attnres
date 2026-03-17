# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: the first pilot-size redesign improved oracle-alpha held-out predictiveness, and the next live need is to scale that `registry_v2` path rather than switch targets again
- Experimental status: the pinned Phase 1 environment exists, the reconstruction checks are green, the development-model runner is live, prompt-matched per-sequence stability is reported, and the saved predictiveness control plan now tunes by pilot predicted routed-loss improvement first and mean JS second; with `prompts/registry_v2.yaml`, the same loss-aware target comparison selects `oracle_alpha_logit_vector` at `lambda=0.01` and recovers `+0.0857` nats over uniform on confirm
- Critical reminder: keep the oracle-alpha framing honest, preserve the deep-research control additions, and treat the positive `16 / 8` result as promising but still well below claim-bearing scale
