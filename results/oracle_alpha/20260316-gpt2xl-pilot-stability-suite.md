# GPT-2 XL Pilot Stability Suite

## Motivation

`resattn-83v` needed to harden the first development-model oracle-alpha runner beyond the two-prompt smoke. The goal of this run was still exploratory: scale to the full saved pilot split and measure whether the learned alpha distributions stay stable under restart seeds and prompt perturbations.

## Methods

- Model: `gpt2-xl`
- Device: local `mps`
- Collection: `oracle_alpha_phase1_v1`
- Split: `pilot`
- Number of prompts: `8`
- Objective: optimize one softmax alpha vector per sequence over the final-output residual-source decomposition, then aggregate the terminal alpha distributions across prompts
- Nulls reported: `uniform`, `random_dirichlet`, `magnitude_proportional`, `last_layer_only`
- Stability settings:
  - restart seeds from the saved control registry: `11`, `17`, `23`, `31`, `47`
  - saved prompt perturbation: `prompt_paraphrase`
  - prompt resampling: `3` resamples of size `6`
- Optimization settings:
  - `20` Adam steps
  - learning rate `0.1`
  - seed-dependent near-uniform alpha-logit initialization

## Results

- Base pilot mean improvement over uniform: `1.2432` nats
- Base bootstrap interval: `[1.1482, 1.3361]`
- Base mean null losses:
  - `uniform = 3.8717`
  - `random_dirichlet = 4.5633`
  - `magnitude_proportional = 5.6110`
  - `last_layer_only = 10.8013`
- Restart stability over aggregate `final_alpha` distributions:
  - mean pairwise Jensen-Shannon divergence: `2.87e-07`
  - mean top-4 Jaccard overlap: `1.0000`
  - mean top-1 source agreement: `0.6000`
- Saved paraphrase perturbation versus the base run:
  - mean pairwise Jensen-Shannon divergence: `0.0305`
  - mean top-4 Jaccard overlap: `0.1429`
  - mean top-1 source agreement: `0.0000`
- Prompt resampling versus the base run:
  - mean pairwise Jensen-Shannon divergence: `0.0191`
  - mean top-4 Jaccard overlap: `0.4222`
  - mean top-1 source agreement: `0.0000`
- The mean sequence improvement stayed positive under all three perturbation families:
  - restart mean: `1.2433`
  - paraphrase mean: `1.2374`
  - resample mean: `1.2525`

## Limitations

- This is still a development-model pilot on the exploratory split, not a claim-bearing feasibility result.
- The current runner is limited to the final-output residual-source decomposition rather than the broader multi-layer claim-bearing surface.
- Restart stability is now valid, but the aggregate alpha summaries still need held-out predictiveness on the confirmatory split before they support a stronger interpretation.

## Next Steps

- Run `resattn-53q`: held-out alpha predictiveness on the confirmatory split using the saved control plan.
- Only after predictiveness is wired should the oracle-alpha lane scale toward the preregistered `100`-sequence gate.
