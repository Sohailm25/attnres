ABOUTME: Summarizes the first same-model Gemma-2 routed-versus-original factual-recall baseline for the tool-breakage lane.
ABOUTME: Records what the pilot establishes, what it leaves ambiguous, and why confirm work is blocked on stronger relative success metrics.

# Gemma-2 Tool-Breakage Baseline Pilot v1

## Motivation

`resattn-28b` exists to build the first same-model routed-versus-original factual-recall baseline on `google/gemma-2-2b`, using the saved custom tuned lens from `resattn-5k9` and the KL-primary metric hierarchy from `resattn-ehz`. The immediate goal is not the final strong claim. It is to prove that the repo can generate checkpointed prompt-level routed traces, compare them against the original-model baseline under both raw and tuned lens, and surface whether routing materially destabilizes the same-model lens baseline on the saved factual-recall pilot prompts.

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v1` pilot split (`8` prompts)
- Target definition: first next token contributed by `target_text` after tokenizing `prompt + target_text`
- Original traces: per-layer `resid_post` states projected through the model's own final normalization and unembedding
- Routed traces: prefix-renormalized oracle-alpha mixtures over the decomposed residual sources available up to each layer depth
- Lens comparison:
  - raw logit lens
  - custom Gemma-2 tuned lens checkpoint from `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`
- Primary metric: KL to the model's final output distribution
- Required secondary diagnostics: mean top-1 agreement plus final-position KL/top-1
- Runtime durability: prompt-level JSON checkpoints under `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1/checkpoints/prompt_results`

## Results

- Mean raw KL to the final distribution increased under routing from `11.3515` to `16.9148`.
- Mean tuned-lens KL increased under routing from `3.3544` to `5.8709`.
- Final-position tuned-lens KL increased under routing from `7.5721` to `11.4121`.
- Mean tuned-lens top-1 agreement dropped under routing from `0.5152` to `0.3200`.
- Final-position tuned-lens top-1 dropped from `0.1250` to `0.0433`.

Prompt-level coverage is broad on the KL-primary metric:

- tuned mean KL worsened on `8 / 8` pilot prompts
- tuned final-position KL worsened on `8 / 8` pilot prompts
- raw mean KL worsened on `7 / 8` pilot prompts
- raw final-position KL worsened on `5 / 8` pilot prompts
- raw and tuned mean top-1 both worsened on `8 / 8` pilot prompts

The rank-style read is more mixed and therefore more informative than the saturated non-monotonicity boolean:

- final-layer target rank worsened under routing on `4 / 8` prompts for both raw and tuned lens
- best-layer target rank also worsened on `4 / 8` prompts for both raw and tuned lens
- the clearest tuned-lens degradations are on prompts such as `Jupiter` (`1 -> 39` at the final layer), `William Shakespeare` (`2 -> 63`), and `giraffe` (`1 -> 101`)

The non-monotonicity boolean is not useful on this pilot by itself:

- `8 / 8` prompts were already non-monotonic on the original-model baseline under the raw lens
- `8 / 8` prompts were also already non-monotonic on the original-model baseline under the tuned lens
- routed traces stayed `8 / 8` non-monotonic under both views

So the pilot does establish same-model degradation under routing on KL and agreement metrics, but it does not establish the prereg “routing increases non-monotonicity on >50% of prompts” threshold because that boolean is saturated before routing is applied.

## Limitations

- This is a pilot-split artifact only. It is not a confirmatory result.
- The current factual-recall target is the first next token from `target_text`, not a full multi-token answer span. That is deliberate for the current `P(correct_token)` trace, but it weakens answer-string claims on multi-token targets.
- The non-monotonicity summary is too coarse for the current Gemma factual-recall baseline because the original-model traces are already non-monotonic on every pilot prompt.
- The current pilot does not include the controlled dynamic-routing counterfactual required for the strong tool-breakage claim.

## Next Steps

- Treat `resattn-28b` as the first baseline implementation pass, not as a strong-claim pass.
- Use `resattn-ypj` to codify relative routed-versus-original success metrics from the saved pilot traces before any confirmatory run.
- Keep later answer-token-facing interpretation subordinate to the KL-primary metric hierarchy unless a stronger final-position target surface is logged explicitly.
