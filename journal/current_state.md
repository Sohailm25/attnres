# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-28b`
- Focus: `resattn-28b` now has the first checkpointed same-model Gemma routed-versus-original factual-recall baseline; the next tool-breakage blocker is `resattn-ypj`, which must replace the saturated non-monotonicity boolean with stronger relative routed-versus-original success metrics before a confirm run
- Experimental status: the `96 / 128` `registry_v4` campaign on `gpt2-xl` clears the development-model oracle-loss gate strongly against uniform and the preregistered nulls; the first sequence-level pattern slice shows weak above-random but below-threshold structure; the Gemma-2 tuned-lens pilot remains KL-strong and answer-token-weaker; and the new Gemma factual-recall baseline pilot shows broad KL degradation under routing on the pilot split but a saturated original-model non-monotonicity baseline
- Critical reminder: keep the oracle-alpha framing honest, preserve the KL-primary tuned-lens metric hierarchy, and do not let the saturated non-monotonicity boolean drive the next tool-breakage decision surface
