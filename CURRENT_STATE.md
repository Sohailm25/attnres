# Current State

**Last updated:** 2026-03-17
**Updated by:** codex-gpt5
**Status:** in_progress
**Current phase:** Phase 1 - Oracle-alpha infrastructure, reconstruction sanity checks, and stability gates

## Active Thesis Lock

- `known`: this repo is now a standalone local workspace on a MacBook Pro.
- `known`: oracle-alpha must be treated as an upper bound on the routing signal available in standard architectures, more precisely as the routing signal recoverable from fixed standard-model representations, and as a lower bound on the benefit of depth routing once routing and computation can co-adapt.
- `known`: the strongest frozen-model framing is recovery of an `effective depth mixture` or latent routing signal; stronger router language requires extra causal validation.
- `known`: the scaffold intentionally removes all Modal assumptions from the source `braindstorms` experiment.
- `known`: the required lanes are now locked into the workspace structure:
  - oracle-alpha against preregistered nulls
  - Figure 8 validation
  - tool-breakage demonstration
  - softmax-constrained vs unconstrained vs top-k routing
  - `w_l` analog geometry for learned router queries
  - block-structure test around the `8 clusters` hypothesis
  - safety analysis around refusal or honesty-related routing differences
- `known`: the local methodology audit surfaced non-trivial implementation hazards that must remain fixed in all future code:
  - no Ward linkage directly on Jensen-Shannon distances
  - uniform routing should reconstruct the original logits after final normalization, not `logits / L`
  - exact per-source routed-logit decomposition must use the shared final normalization factor from the full mixture
  - claim-bearing significance should default to the sequence-level unit
  - `resid_post` is insufficient for final Figure 8 claims; use sublayer outputs
- `known`: the local paper cache now exists under `background-work/papers/files` and is indexed in `background-work/papers/DOWNLOAD_MANIFEST.md`
- `known`: a second red-team review identified four additional publishability risks that must stay fixed:
  - tool-breakage cannot rely on raw logit-lens monotonicity as the vanilla baseline
  - claim-bearing work needs a pilot/confirmatory split
  - router input wording must stay explicitly per-token, e.g. `h_1[t]`
  - refusal-feature discovery must precede safety-lane causal claims
- `known`: Figure 8 has now been verified locally in `research/Attention_Residuals.pdf`; its pattern surface is diagonal dominance, embedding persistence, layer specialization, learned skip connections, and Block AttnRes with `N = 8` preserving that structure.
- `known`: a deep-research review added four more control requirements:
  - use MIB as a benchmark anchor or sanity control when the task-model pair fits it
  - clear a stability suite and an out-of-sample predictiveness check before high-claim interpretation
  - require a controlled dynamic-routing counterfactual for the strong tool-breakage claim
  - separate harmfulness from refusal and localize safety layers before mediator-conditioned safety claims
- `known`: the Phase 1 dependency freeze has landed:
  - `requirements.txt` now pins the direct stack used for Phase 1
  - `requirements.lock.txt` captures the fully resolved transitive environment
  - `.venv` now imports the pinned `torch`, `transformer-lens`, `sae-lens`, `nnsight`, and `pyvene` stack successfully on this machine, and `torch.backends.mps.is_available()` is `True`
- `known`: the first reconstruction correctness surface now exists in `validation/reconstruction.py` with tests covering:
  - cache reconstruction from embedding plus sublayer outputs
  - uniform-routing agreement with the original logits after the model's final normalization
  - exact shared-final-norm routed-logit decomposition
  - rejection of the per-source normalization shortcut as exact
- `known`: model-backed Phase 1 reconstruction is now exercised in `validation/model_backed.py` and by the `gpt2-xl` smoke artifact:
  - cached embedding plus per-sublayer writes reconstruct the final residual exactly when accumulated in forward order
  - applying `ln_final` plus `unembed` to that reconstructed mixture recovers the original logits exactly on the smoke prompt
  - per-layer `resid_mid` and `resid_post` identities were exact in the smoke check on local MPS
- `known`: the pilot/confirmatory split is now saved and code-enforced:
  - `prompts/registry_v1.yaml` is the versioned prompt registry for the current inline prompt collections
  - `prompts/registry.py` centralizes registry loading plus the confirm-only access guard
  - `scripts/export_prompt_split.py` refuses confirmatory reads when exploratory mode is enabled
  - the first saved collections cover Phase 1 oracle-alpha prompts and the factual-recall tool-breakage lane
- `known`: the preregistered Phase 1 control suite is now saved and importable:
  - `configs/oracle_alpha_controls_v1.yaml` records the current stability suite, held-out predictiveness check, and explicit MIB anchor plan
  - `validation/oracle_alpha_controls.py` provides the registry loader plus reusable bootstrap, restart-stability, and held-out predictiveness helpers
  - the current MIB stance is `planned`, not `omitted`: the control is recorded now, while execution remains blocked on a future oracle-alpha runner
- `known`: the first oracle-alpha execution harness now exists:
  - `validation/oracle_alpha_runner.py` implements a final-output development slice that optimizes a per-sequence softmax alpha vector over fixed cached residual sources
  - `scripts/run_oracle_alpha_development_slice.py` consumes the saved prompt and control registries by default and writes a JSON artifact
  - the first `gpt2-xl` pilot smoke on `2` prompts completed on local MPS and improved mean sequence loss over uniform by `1.2141` nats, but this remains a runner smoke rather than a claim-bearing result
- `known`: the scaled development-model pilot stability suite now exists:
  - `scripts/run_oracle_alpha_pilot_stability_suite.py` runs restart, saved-paraphrase, and prompt-resample checks against the same saved prompt and control registries
  - the exploratory `gpt2-xl` pilot artifact on `8` prompts improved mean sequence loss over uniform by `1.2432` nats with a bootstrap interval of `[1.1482, 1.3361]`
  - restart variation in the aggregate `final_alpha` distributions was tiny but non-zero (`mean JS = 2.87e-07`, top-1 agreement `0.60`), while saved paraphrases and prompt resampling moved the aggregate alpha distributions more strongly (`JS = 0.0305` and `0.0191`)
  - this is still a development-model stability hardening result, not confirm-split predictiveness or a claim-bearing feasibility pass
- `known`: the first held-out predictiveness artifact now exists, and it is a real blocker rather than a positive result:
  - `scripts/run_oracle_alpha_heldout_predictiveness_check.py` runs the pilot-to-confirm predictiveness check using mean-pooled `h_1[t]` features and a pilot-tuned ridge regressor
  - on `gpt2-xl`, the confirm-split result was weak for the current feature spec: `R^2 = -0.2456`, mean JS to oracle alpha `= 0.2434`, and predicted alpha vectors were slightly worse than uniform on average (`-0.0348` nats)
  - the oracle alpha itself remained strong on the confirm split (`1.3409` nats over uniform), so the failure is in the current predictor surface rather than in the confirm-split oracle run
  - strong oracle-alpha interpretation remains blocked until a stronger out-of-sample predictor exists
- `known`: the first richer feature-summary comparison improved the predictor slightly but did not clear the blocker:
  - `resattn-k2e` compared a small pilot-only set of internal-state summaries and selected `mean_pooled_h_4[t]_resid_post_layer_3` by leave-one-out pilot mean JS
  - the selected `h_4[t]` summary modestly improved the confirm metrics (`R^2 = -0.2314`, mean JS `= 0.2409`, predicted mean improvement `= -0.0010` nats) relative to the `h_1[t]` baseline
  - that is still not strong enough for claim-bearing interpretation, so the next follow-up moves to token-aware or prompt-level feature surfaces rather than declaring the problem solved
- `known`: the first token-aware feature comparison improved alpha-shape recovery slightly but still did not clear the held-out predictiveness blocker:
  - `resattn-7ve` added minimal token-aware `h_4[t]` summaries and selected `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat` by pilot leave-one-out mean JS
  - the selected token-aware summary improved the confirm metrics again (`R^2 = -0.2154`, mean JS `= 0.2377`) relative to the mean-pooled `h_4[t]` baseline
  - predicted routed loss remained effectively neutral to slightly negative on confirm (`-0.0011` nats versus uniform), so the stronger interpretation gate is still blocked
  - the next follow-up is prompt-level or hybrid feature surfaces rather than more `h_4[t]`-only pooling variants
- `known`: the prompt-level and hybrid feature comparison failed to beat the current token-aware baseline:
  - `resattn-27f` compared prompt-shape scalars, mean-pooled token embeddings, and hybrids that appended those prompt-level features to `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`
  - none of those candidates beat the token-aware `h_4[t]` baseline on pilot leave-one-out mean JS, so the selected feature source and confirm metrics stayed unchanged (`R^2 = -0.2154`, mean JS `= 0.2377`, predicted mean improvement `= -0.0011` nats)
  - this is stronger evidence that the current blocker is not just a missing simple feature family
  - the next follow-up is a bounded design review of the held-out predictiveness setup rather than more small feature-surface additions
- `known`: the first bounded design-review alignment slice has landed:
  - `resattn-23p` made the runner consult the saved predictiveness control metrics instead of hard-coding pilot JS selection
  - the stability suite now reports prompt-matched per-sequence alpha stability alongside the previous aggregate alpha-distribution metrics
  - this improves methodological alignment but does not change the core blocker: the predictor still learns the raw oracle-alpha simplex target in unconstrained Euclidean ridge coordinates on a very small prompt split
  - the next follow-up is a constrained-or-compressed predictiveness target rather than another infrastructure-only tweak

## Immediate Next Steps

1. Implement a constrained or compressed oracle-alpha predictiveness target in `resattn-qq2`.
2. Decide the tuned-lens path for Gemma-2 tool-breakage: custom lens training versus a secondary-model comparison.
3. Validate the refusal-feature discovery workflow before the safety lane becomes active.
4. Port the model-backed reconstruction smoke from the development model to the primary Gemma-2 lane when the Gemma-specific backend path is ready.
5. Expand the saved prompt registry or control registry only when a lane needs a larger or more specialized confirmatory surface.

## Phase 1 Gate

The first execution gate remains the preregistered one from `research/decision-matrix.md`:

- show mean cross-entropy reduction greater than `0.01` nats over uniform routing on at least `100` sequences
- require `p < 0.01`
- compare against random, magnitude-proportional, and last-layer-only nulls
- on the small-scale validation tranche, require `d > 0.2` versus the random baseline before scaling up

## Operational Locks

- `known`: Figure 8 tests must use operationalized metrics:
  - locality score
  - `alpha_0` embedding persistence across depth
  - `Entropy(pre-attn) > Entropy(pre-MLP)`
  - off-diagonal peaks above `2/L`
- `known`: tool-breakage requires a non-monotonic logit-lens demonstration on factual recall, with a target of non-monotonic curves on `>50%` of prompts before making a strong breakage claim.
- `known`: raw logit lens is not assumed monotonic in the vanilla model; the strong tool-breakage claim requires additional instability under routing relative to the original-model baseline and a tuned-lens-aware comparison.
- `known`: the strong tool-breakage claim also requires a controlled dynamic-routing counterfactual or another explicitly logged confirmatory failure metric.
- `known`: router training success is not just "it trains"; the local target gate is `R^2 > 0.5` when approximating oracle-alpha.
- `known`: clustering must be informative enough to clear `silhouette > 0.2` before we claim task-structured routing.
- `known`: if hierarchical clustering is run on Jensen-Shannon distances directly, use average or complete linkage rather than Ward.
- `known`: any paired t-test gate is interpreted over per-sequence mean deltas unless a stronger dependence-aware method is written down first.
- `known`: authored control documents and the local paper cache passed a final existence audit on 2026-03-16.
- `known`: claim-bearing prompts and thresholds must come from a pilot/confirmatory split rather than one blended prompt pool.
- `known`: the confirm set is now code-locked against exploratory access through the saved prompt registry helpers and export script.
- `known`: claim-bearing oracle-alpha runs must also load an explicit control plan that fixes bootstrap size, stability perturbations, held-out predictiveness evaluation, and the MIB plan or omission rationale.
- `known`: the current oracle-alpha runner is a development slice over final-output residual sources, not yet the full claim-bearing multi-layer analysis surface.
- `known`: MIB is a benchmark/control anchor, not the project spine; if it is omitted for a lane, that omission must be justified in `DECISIONS.md`.
- `known`: claim-bearing pattern interpretations require stability and out-of-sample alpha predictiveness on the confirmatory split.
- `known`: strong Figure 8 or trained-routing match claims require a reproducible proxy or local small-scale depth-mixing reproduction; otherwise Lane 2 is an internal prediction-surface comparison against the published AttnRes Figure 8.
- `known`: safety analysis must distinguish harmfulness-encoding from refusal-execution and start with layer localization.
- `known`: checkpoint studies support within-run evolution claims, not counterfactual schedule claims, unless additional evidence is logged.
