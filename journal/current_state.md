# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: `resattn-7hb` has now built the first local Block AttnRes proxy and matched baseline; the next highest-value follow-up is `resattn-3l6`, which decides whether the Figure 8 proxy should scale further on the current char-level regime or move to a stronger tokenizer/corpus setup
- Experimental status: the `96 / 128` `registry_v4` campaign on `gpt2-xl` clears the development-model oracle-loss gate strongly against uniform and the preregistered nulls; the first sequence-level pattern slice shows weak above-random but below-threshold structure; the Gemma-2 tuned-lens pilot remains KL-strong and answer-token-weaker; the locked confirm Gemma factual-recall baseline reproduces broad KL degradation under routing with `7 / 8` tuned rank-range increase and `5 / 8` tuned final-rank worsening; the dynamic counterfactual shows routed traces remain worse than the fixed `pilot_mean_alpha` control but not worse than the prompt-permuted dynamic control on the tuned KL surface; and the new local `8`-block Wikitext proxy beats its matched baseline on eval loss while still showing mixed Figure 8 metrics (`deep_embedding_persistence=0.1049`, `mean_pre_attn_entropy < mean_pre_mlp_entropy`)
- Critical reminder: keep the oracle-alpha framing honest, preserve the KL-primary tuned-lens metric hierarchy, and do not treat the new local AttnRes proxy as a Figure 8 alignment pass until the proxy metrics themselves look credible
