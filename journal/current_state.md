# Journal Current State

- Date: 2026-03-17
- Repo: standalone and initialized
- Branch: `wip/resattn-scaffold`
- Focus: `registry_v4` is now frozen as the prereg-scale prompt surface, so the next live need is to launch the larger tmux-backed campaign rather than keep extending the registry or redesigning the predictor
- Experimental status: the pinned Phase 1 environment exists, the reconstruction checks are green, the development-model runner is live, prompt-matched per-sequence stability is reported, the saved predictiveness control plan tunes by pilot predicted routed-loss improvement first and mean JS second, and the checkpointed campaign runner can now consume `prompts/registry_v4.yaml`, which expands the oracle-alpha surface to `96` pilot prompts and `128` confirm prompts while keeping the tool-breakage collection unchanged
- Critical reminder: keep the oracle-alpha framing honest, preserve the deep-research control additions, and treat both the positive `32 / 16` result and the new `96 / 128` prompt freeze as setup for the real scale-up, not as the scale-up itself
