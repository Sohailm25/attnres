ABOUTME: Summarizes the first locked confirm-split same-model Gemma-2 factual-recall tool-breakage baseline.
ABOUTME: Records the confirmatory routed-versus-original result under the KL-primary and rank-instability metric surface.

# Gemma-2 Tool-Breakage Baseline Confirm v1

## Motivation

`resattn-6te` exists to answer the first confirmatory question in the Gemma tool-breakage lane: on the locked factual-recall confirm split, do routed traces degrade the same-model original baseline under the KL-primary tuned-lens hierarchy, and do the codified relative rank metrics remain informative on unseen prompts after the pilot metric-hardening step?

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v1` confirm split (`8` prompts, `exploratory=false`)
- Target definition: first next token contributed by `target_text` after tokenizing `prompt + target_text`
- Original traces: per-layer `resid_post` states projected through the model's own final normalization and unembedding
- Routed traces: prefix-renormalized oracle-alpha mixtures over the decomposed residual sources available up to each layer depth
- Lens comparison:
  - raw logit lens
  - custom Gemma-2 tuned lens checkpoint from `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`
- Primary metric: KL to the model's final output distribution
- Secondary diagnostics: mean top-1 agreement plus final-position KL/top-1
- Relative instability metrics:
  - routed-versus-original non-monotonicity increase
  - final-target-rank worsening
  - best-target-rank worsening
  - target-rank-range increase
- Runtime durability: prompt-level JSON checkpoints under `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1/checkpoints/prompt_results`

## Results

The confirm split reproduces broad same-model degradation under routing on the KL-primary surface:

- mean raw KL to the final distribution increased from `11.1598` to `16.0153`
- mean tuned-lens KL increased from `3.3834` to `5.9061`
- final-position tuned-lens KL increased from `5.9851` to `8.8947`
- mean tuned-lens top-1 dropped from `0.5095` to `0.3024`
- final-position tuned-lens top-1 dropped from `0.1442` to `0.1154`

Prompt-level confirm coverage remains broad on the core degradation metrics:

- raw final-target-rank worsened on `6 / 8` prompts
- tuned final-target-rank worsened on `5 / 8` prompts
- raw best-target-rank worsened on `4 / 8` prompts
- tuned best-target-rank worsened on `4 / 8` prompts
- raw target-rank range increased on `5 / 8` prompts
- tuned target-rank range increased on `7 / 8` prompts

The confirm result sharpens the earlier pilot read:

- routed-versus-original non-monotonicity increase is still `0 / 8` under both raw and tuned lens
- the old boolean therefore remains uninformative on Gemma factual recall
- the tuned rank-range metric is now the cleanest confirmatory instability read, because it crosses the prereg `>50%` threshold by a wide margin (`7 / 8`)
- tuned final-target-rank worsening also clears the threshold (`5 / 8`)

So the confirm split does not support the old absolute non-monotonicity story, but it does support a rank-instability story: on unseen prompts, routing usually broadens target-rank instability and often worsens the final answer-token rank relative to the original baseline.

## Limitations

- This confirm artifact still uses the first next token from `target_text`, not a full answer span. Multi-token answers remain an answer-string caveat.
- The current run does not include the prereg-required controlled dynamic-routing counterfactual.
- The confirm artifact therefore supports the same-model routed-versus-original baseline and the rank-instability read, but not the final strong tool-breakage claim.

## Next Steps

- Treat `resattn-6te` as the first confirmatory baseline pass for the Gemma factual-recall lane.
- Use `resattn-g09` to add the controlled dynamic-routing counterfactual before making the strong tool-breakage claim.
- Keep answer-token-facing interpretation subordinate to the KL-primary metric hierarchy and the first-token target caveat.
