ABOUTME: Records the first pilot baseline on the matched-family Gemma factual-recall tool-breakage surface.
ABOUTME: Tests whether the same-model degradation signal survives once the prompt surface is aligned to the strongest factual routing families.

# Motivation

`resattn-qcn` follows the factual-routing bridge artifact. That bridge showed two
things at once:

- the strongest primary-model factual routing families are real and easy to
  localize
- the old eight-prompt tool-breakage surface only partially overlaps them

The next honest step was to build a tool-breakage prompt surface centered on the
matched families and check whether the same-model degradation signal survives on
that better-aligned surface.

# Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v2` pilot split (`8` prompts)
- Families:
  - `capital_fact`
  - `element_symbol`
  - `author_fact`
  - `moon_fact`
- Family balance:
  - `2` pilot prompts per family
  - `4` confirm prompts per family reserved for the next step
- Lens comparison:
  - raw logit lens
  - custom Gemma-2 tuned lens from
    `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`
- Primary metric: KL to the model's final output distribution
- Secondary diagnostics:
  - mean top-1 agreement
  - final-position KL / top-1
  - routed-versus-original final-target-rank worsening
  - routed-versus-original target-rank-range increase
- Durability:
  - prompt-level checkpoints under
    `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-pilot-v2/checkpoints/prompt_results`
  - exact-command rerun after completion to verify checkpoint reuse

# Results

- The matched-family pilot keeps a strong same-model tuned-lens degradation
  signal:
  - mean tuned KL to the final distribution increases from `3.3374` to `5.9222`
    (`+2.5849`)
  - final-position tuned KL increases from `6.2867` to `9.1985` (`+2.9118`)
  - mean tuned top-1 drops from `0.5494` to `0.3264`
  - final-position tuned top-1 drops from `0.1154` to `0.0721`
- The relative rank metrics remain informative on the aligned surface:
  - tuned final-target-rank worsening on `4 / 8` prompts
  - tuned best-target-rank worsening on `3 / 8` prompts
  - tuned target-rank-range increase on `5 / 8` prompts
  - tuned routed-versus-original non-monotonicity increase remains `0 / 8`
- Family-level pilot heterogeneity is real but not disqualifying:
  - capitals:
    - mean final-position tuned KL delta `= +3.9622`
    - mean final target-rank delta `= +6.0`
  - moons:
    - mean final-position tuned KL delta `= +3.2368`
    - mean final target-rank delta `= +1.5`
  - elements:
    - mean final-position tuned KL delta `= +2.4228`
    - mean final target-rank delta `= +0.5`
    - mean range delta is negative, so these prompts often compress rather than
      broaden the tuned target-rank trajectory
  - authors:
    - mean final-position tuned KL delta `= +0.8424`
    - mean final target-rank delta `= +2.0`
- Durability check passed:
  - exact-command rerun completed in `10.17` seconds
  - all `8` prompt checkpoints kept their original timestamps
  - only `summary.json` was rewritten on rerun

# Interpretation

- The aligned prompt surface is viable. Better family alignment did not wash out
  the same-model tool-breakage effect.
- The pilot also makes the next question sharper. We no longer need to ask
  whether a matched-family surface can carry any degradation signal at all. It
  can.
- The next meaningful question is confirmatory: does the locked `16`-prompt
  confirm split on this matched-family surface preserve the tuned-lens KL signal
  and the rank-instability read strongly enough to matter?

# Limitations

- This is still a pilot artifact.
- The confirm split has not been run yet.
- Authors are notably weaker than the other matched families on this pilot.
- The old dynamic-control boundary remains unchanged.

# Next Steps

- Close `resattn-qcn`.
- Run `resattn-4ny`: the locked confirm baseline on
  `tool_breakage_factual_recall_v2`.
- Keep the old mixed factual surface as contextual evidence rather than the main
  next baseline.
