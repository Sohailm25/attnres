ABOUTME: Summarizes the controlled dynamic-routing counterfactual on the matched-family Gemma factual-recall surface.
ABOUTME: Tests whether the aligned `tool_breakage_factual_recall_v2` routed trace is uniquely damaging relative to fixed-alpha and prompt-permuted controls.

# Gemma-2 Matched-Family Dynamic-Routing Counterfactual v2

## Motivation

`resattn-qww` exists because `tool_breakage_factual_recall_v2` replaced the old
mixed factual baseline as the main bounded same-model tool-breakage surface.
The next honest question was confirmatory: on this aligned surface, is the
prompt-matched routed trace uniquely more damaging than nearby non-uniform
controls, or does the stronger same-model claim still collapse once dynamic
counterfactuals are added?

## Methods

- Model: `google/gemma-2-2b`
- Device: `mps`
- Prompt surface: `tool_breakage_factual_recall_v2` confirm split (`16` prompts)
- Baseline source of truth:
  `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-confirm-v2/summary.json`
- Fixed-alpha source:
  `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-pilot-v2/summary.json`
- Tuned lens:
  `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`
- Original and routed arms: reused from the saved matched-family confirm
  baseline
- Control arms:
  - `prompt_permuted_alpha`: deterministic cyclic reassignment of the saved
    confirm oracle alphas across prompts
  - `pilot_mean_alpha`: fixed alpha equal to the mean oracle alpha over the
    saved matched-family pilot prompts
- Primary metric: tuned-lens mean KL to the final output distribution
- Secondary diagnostics:
  - tuned final-position KL
  - tuned routed-versus-control final-target-rank worsening
  - tuned routed-versus-control best-target-rank worsening
  - tuned routed-versus-control target-rank-range increase
- Runtime durability:
  - prompt-level JSON checkpoints under
    `results/tool_breakage/20260318-gemma2-tool-breakage-matched-family-counterfactual-v2/checkpoints/prompt_results`
  - exact-command rerun completed in `10.22` seconds after the first run

## Results

The aligned `v2` surface weakens the fixed-alpha objection cleanly but still
does not clear the stronger prompt-specific dynamic claim.

Against the fixed non-uniform control, routed remains substantially worse:

- `pilot_mean_alpha` mean tuned KL over original: `+1.6727`
- routed minus `pilot_mean_alpha` mean tuned KL: `+1.0917`
- routed minus `pilot_mean_alpha` final-position tuned KL: `+1.3894`
- routed worsens final target rank versus `pilot_mean_alpha` on `11 / 16`
  prompts
- routed worsens best target rank versus `pilot_mean_alpha` on `11 / 16` prompts
- routed increases target-rank range versus `pilot_mean_alpha` on `13 / 16`
  prompts

Against the current cyclic prompt-permuted dynamic control, the read is much
tighter and mixed:

- `prompt_permuted_alpha` mean tuned KL over original: `+2.7768`
- routed minus `prompt_permuted_alpha` mean tuned KL: `-0.0123`
- routed minus `prompt_permuted_alpha` final-position tuned KL: `+0.3533`
- routed worsens final target rank versus `prompt_permuted_alpha` on `7 / 16`
  prompts
- routed worsens best target rank versus `prompt_permuted_alpha` on `8 / 16`
  prompts
- routed increases target-rank range versus `prompt_permuted_alpha` on `8 / 16`
  prompts

The donor mapping matters for interpretation. Because the confirm ids are
grouped by family, the current cyclic prompt-permuted control is already
within-family on `12 / 16` prompts and crosses family boundaries only on the
four boundary transitions:

- capital -> capital on `3 / 4` capital prompts, then capital -> element
- element -> element on `3 / 4` element prompts, then element -> author
- author -> author on `3 / 4` author prompts, then author -> moon
- moon -> moon on `3 / 4` moon prompts, then moon -> capital

Family-conditioned tension stays visible:

- element prompts are the clearest family where routed is worse than the
  prompt-permuted arm on both mean and final-position tuned KL
- capital prompts are mixed but slightly favor routed being worse on final
  position
- author prompts are the clearest family pulling the mean-KL aggregate the
  other way
- moon prompts remain mixed

## Interpretation

- The aligned `v2` surface now gives a sharper tool-breakage story than the old
  mixed factual surface.
- The weaker static-routing objection is cleared again: routed is substantially
  worse than the fixed pilot-mean alpha control on the tuned primary metric and
  the relative rank diagnostics.
- The stronger same-model dynamic-routing claim is still blocked on the current
  control surface because routed does not clearly beat the mostly within-family
  cyclic donor arm on mean tuned KL.
- The remaining uncertainty is now narrower than before. The next useful
  follow-up is not another baseline refresh or a bigger aggregate rerun. It is
  an explicit within-family versus cross-family donor decomposition.

## Limitations

- The confirm split is still `16` prompts, so the control read is confirmatory
  but modest in size.
- The current prompt-permuted control is deterministic cyclic reassignment, not
  a richer donor family.
- The first-token answer caveat still applies to multi-token author targets.

## Next Steps

- Close `resattn-qww` as mixed.
- Run `resattn-o3n` to split the donor control into explicit within-family and
  cross-family arms on the same matched-family surface.
- Keep the stronger same-model Gemma tool-breakage claim blocked until routed
  clearly beats the within-family donor control on the tuned primary metric.
