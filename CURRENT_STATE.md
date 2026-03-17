# Current State

**Last updated:** 2026-03-16
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

## Immediate Next Steps

1. Add the identifiability, MIB, and out-of-sample predictiveness controls before claim-bearing oracle-alpha optimization.
2. Decide the tuned-lens path for Gemma-2 tool-breakage: custom lens training versus a secondary-model comparison.
3. Port the model-backed reconstruction smoke from the development model to the primary Gemma-2 lane when the Gemma-specific backend path is ready.
4. Expand the saved prompt registry when a lane needs a larger or more specialized confirmatory pool, without reopening the access-enforcement rule.
5. Keep broader oracle-alpha optimization blocked until the preregistered controls are green.

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
- `known`: MIB is a benchmark/control anchor, not the project spine; if it is omitted for a lane, that omission must be justified in `DECISIONS.md`.
- `known`: claim-bearing pattern interpretations require stability and out-of-sample alpha predictiveness on the confirmatory split.
- `known`: strong Figure 8 or trained-routing match claims require a reproducible proxy or local small-scale depth-mixing reproduction; otherwise Lane 2 is an internal prediction-surface comparison against the published AttnRes Figure 8.
- `known`: safety analysis must distinguish harmfulness-encoding from refusal-execution and start with layer localization.
- `known`: checkpoint studies support within-run evolution claims, not counterfactual schedule claims, unless additional evidence is logged.
