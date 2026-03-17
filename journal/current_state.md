# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: the first prereg-scale `gpt2-xl` oracle-alpha campaign is now complete, so the next live oracle task is pattern analysis on the saved `registry_v4` artifact rather than more scale-up work
- Experimental status: the `96 / 128` `registry_v4` campaign on `gpt2-xl` clears the development-model oracle-loss gate strongly against uniform and the preregistered nulls, while held-out predicted routed loss remains positive at prereg scale (`+0.1162` nats, `95 / 128` confirm prompts positive) but alpha-shape recovery stays weak (`R^2 = -0.0884`, mean JS `= 0.2427`) and the selector returns to the raw-simplex target with ridge `100.0`
- Critical reminder: keep the oracle-alpha framing honest, preserve the deep-research control additions, and treat this as a development-model gate clear plus a mixed predictiveness result, not as a full interpretation clear or a primary-model replication
