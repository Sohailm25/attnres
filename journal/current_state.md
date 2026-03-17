# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: the first prereg-scale `gpt2-xl` pattern-analysis artifact is now landed, so the next live question is whether the weak above-random clustering survives grouped-source and resampling robustness checks or whether work should shift to `resattn-qm4` / `resattn-3f1`
- Experimental status: the `96 / 128` `registry_v4` campaign on `gpt2-xl` clears the development-model oracle-loss gate strongly against uniform and the preregistered nulls, held-out predicted routed loss remains positive at prereg scale (`+0.1162` nats, `95 / 128` confirm prompts positive), and the first sequence-level pattern slice finds weak above-random structure (`silhouette = 0.1428` vs random `0.1093`) that is still below the prereg `0.2` gate and dominated by a `126 / 2` outlier split
- Critical reminder: keep the oracle-alpha framing honest, preserve the deep-research control additions, and treat the current pattern result as a development-model Phase 2 entry artifact, not as a block-structure pass, a Figure 8 pass, or a primary-model replication
