# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: `resattn-6te` has now landed the locked confirm-split Gemma factual-recall baseline; the next tool-breakage blocker is `resattn-g09`, the controlled dynamic-routing counterfactual
- Experimental status: the `96 / 128` `registry_v4` campaign on `gpt2-xl` clears the development-model oracle-loss gate strongly against uniform and the preregistered nulls; the first sequence-level pattern slice shows weak above-random but below-threshold structure; the Gemma-2 tuned-lens pilot remains KL-strong and answer-token-weaker; and the locked confirm Gemma factual-recall run reproduces broad KL degradation under routing with `7 / 8` tuned rank-range increase, `5 / 8` tuned final-rank worsening, and `0 / 8` non-monotonicity increase
- Critical reminder: keep the oracle-alpha framing honest, preserve the KL-primary tuned-lens metric hierarchy, and treat the controlled dynamic-routing counterfactual as the next real blocker before any strong Gemma tool-breakage claim
