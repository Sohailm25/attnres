# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-ehz`
- Focus: `resattn-ehz` is codifying the Gemma tool-breakage metric hierarchy so `resattn-28b` can proceed with KL as the primary tuned-lens baseline metric while keeping answer-token claims gated on final-position evidence
- Experimental status: the `96 / 128` `registry_v4` campaign on `gpt2-xl` clears the development-model oracle-loss gate strongly against uniform and the preregistered nulls; the first sequence-level pattern slice shows weak above-random but below-threshold structure; and the Gemma-2 tuned-lens pilot is now explicitly treated as a KL-strong, answer-token-weaker baseline artifact
- Critical reminder: keep the oracle-alpha framing honest, preserve the deep-research control additions, and do not let distributional tuned-lens gains silently substitute for answer-token factual-recall claims
