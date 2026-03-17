# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-g09`
- Focus: `resattn-g09` has landed the locked Gemma dynamic counterfactual and the strongest same-model tool-breakage claim remains blocked; the next highest-value implementation lane is now `resattn-7hb`
- Experimental status: the `96 / 128` `registry_v4` campaign on `gpt2-xl` clears the development-model oracle-loss gate strongly against uniform and the preregistered nulls; the first sequence-level pattern slice shows weak above-random but below-threshold structure; the Gemma-2 tuned-lens pilot remains KL-strong and answer-token-weaker; the locked confirm Gemma factual-recall baseline reproduces broad KL degradation under routing with `7 / 8` tuned rank-range increase and `5 / 8` tuned final-rank worsening; and the new dynamic counterfactual shows routed traces remain worse than the fixed `pilot_mean_alpha` control but not worse than the prompt-permuted dynamic control on the tuned KL surface
- Critical reminder: keep the oracle-alpha framing honest, preserve the KL-primary tuned-lens metric hierarchy, and do not overstate the Gemma tool-breakage lane beyond the mixed counterfactual result that is now saved
