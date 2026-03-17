# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-ypj`
- Focus: `resattn-ypj` is codifying the next Gemma tool-breakage decision surface so the confirm run uses relative rank-instability and rank-degradation metrics instead of the saturated non-monotonicity boolean
- Experimental status: the `96 / 128` `registry_v4` campaign on `gpt2-xl` clears the development-model oracle-loss gate strongly against uniform and the preregistered nulls; the first sequence-level pattern slice shows weak above-random but below-threshold structure; the Gemma-2 tuned-lens pilot remains KL-strong and answer-token-weaker; and the saved Gemma factual-recall pilot now shows `0 / 8` routed-versus-original non-monotonicity increase but `7 / 8` tuned rank-range increase plus `4 / 8` tuned best-rank worsening
- Critical reminder: keep the oracle-alpha framing honest, preserve the KL-primary tuned-lens metric hierarchy, and let the next confirmatory tool-breakage read be driven by the codified relative rank metrics rather than the saturated legacy boolean
