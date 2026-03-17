ABOUTME: Summarizes the controlled dynamic-routing counterfactual for the locked Gemma-2 factual-recall confirm split.
ABOUTME: Tests whether the routed-versus-original degradation from resattn-6te is stronger than prompt-misaligned and fixed-alpha controls.

# Gemma-2 Tool-Breakage Counterfactual Confirm v1

## Motivation

`resattn-g09` exists to satisfy the preregistered confirmatory control for the same-model Gemma factual-recall lane: are the routed-versus-original KL and rank-instability degradations from `resattn-6te` actually specific to prompt-matched routing, or would nearby non-uniform routed mixtures damage the same traces generically?

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v1` confirm split (`8` prompts, `exploratory=false`)
- Baseline source of truth: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1/summary.json`
- Tuned lens: `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`
- Original and routed arms: reused from the saved `resattn-6te` confirm artifact
- Control arms:
  - `prompt_permuted_alpha`: deterministic cyclic reassignment of the saved confirm oracle alphas across prompts
  - `pilot_mean_alpha`: fixed alpha equal to the mean oracle alpha over the saved `resattn-28b` pilot prompts
- Primary metric: tuned-lens mean KL to the final output distribution
- Secondary diagnostics:
  - tuned final-position KL
  - tuned routed-versus-control final-target-rank worsening
  - tuned routed-versus-control best-target-rank worsening
  - tuned routed-versus-control target-rank-range increase
- Runtime durability: prompt-level JSON checkpoints under `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-confirm-v1/checkpoints/prompt_results`

## Results

The two control arms separate the scientific read cleanly.

Against the fixed non-uniform control, the saved routed trace remains more damaging:

- baseline routed mean tuned KL over original: `+2.5227`
- `pilot_mean_alpha` mean tuned KL over original: `+1.9137`
- routed minus `pilot_mean_alpha` mean tuned KL: `+0.6091`
- routed minus `pilot_mean_alpha` final-position tuned KL: `+0.8750`
- routed worsens final target rank versus `pilot_mean_alpha` on `5 / 8` prompts
- routed worsens best target rank versus `pilot_mean_alpha` on `4 / 8` prompts
- routed increases target-rank range versus `pilot_mean_alpha` on `4 / 8` prompts

Against the prompt-misaligned dynamic control, the saved routed trace is not uniquely worst:

- `prompt_permuted_alpha` mean tuned KL over original: `+2.8310`
- routed minus `prompt_permuted_alpha` mean tuned KL: `-0.3082`
- routed minus `prompt_permuted_alpha` final-position tuned KL: `-0.6541`
- routed worsens final target rank versus `prompt_permuted_alpha` on `4 / 8` prompts
- routed worsens best target rank versus `prompt_permuted_alpha` on `3 / 8` prompts
- routed increases target-rank range versus `prompt_permuted_alpha` on `2 / 8` prompts

So the counterfactual resolves the lane in a mixed direction:

- the weaker static-routing objection is cleared, because the prompt-matched routed trace is worse than the fixed pilot-mean alpha control on the tuned primary metric
- the stronger input-dependent-routing claim is not cleared, because a prompt-misaligned dynamic control is at least as damaging as the prompt-matched route on the tuned KL surface

## Limitations

- The confirm split is still `8` prompts, so the control read is confirmatory but small.
- The prompt-permuted control is deterministic cyclic reassignment, not a whole family of donor-match samplers.
- The target remains the first next token implied by `target_text`, so multi-token answer caveats still apply.

## Next Steps

- Treat `resattn-g09` as satisfying the preregistered dynamic counterfactual requirement while weakening, not strengthening, the strongest same-model Gemma claim.
- Keep the strong Gemma tool-breakage claim blocked.
- Prefer moving the next major implementation step to the Figure 8 proxy lane unless a new follow-up issue justifies a narrower dynamic-control study without moving the goalposts.
