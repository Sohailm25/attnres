# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-5k9`
- Focus: `resattn-5k9` now has a real Gemma-2 viability artifact, so the live choice is whether to take `resattn-ehz` for tuned-lens metric sharpening or move sideways to `resattn-ojq`, `resattn-7hb`, or `resattn-3f1`
- Experimental status: the `96 / 128` `registry_v4` campaign on `gpt2-xl` clears the development-model oracle-loss gate strongly against uniform and the preregistered nulls; the first sequence-level pattern slice shows weak above-random but below-threshold structure; and the first full-surface Gemma-2 tuned-lens pilot is now a real held-out pass on KL/distributional recovery with only modest final-position top-1 gains on factual recall
- Critical reminder: keep the oracle-alpha framing honest, preserve the deep-research control additions, and do not overstate the Gemma tuned-lens pilot as an answer-token recovery result or a routed-versus-original tool-breakage result
