# Current State

**Last updated:** 2026-03-17
**Updated by:** codex-gpt5
**Status:** in_progress
**Current phase:** Phase 3 - Tool-breakage and safety routing analysis

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
  - `prompts/registry_v4.yaml` is the current default prompt registry for the inline prompt collections, while `prompts/registry_v1.yaml`, `prompts/registry_v2.yaml`, and `prompts/registry_v3.yaml` remain as earlier saved prompt-surface snapshots
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
- `known`: the first constrained predictiveness target is now implemented and rerun:
  - `resattn-qq2` added the `oracle_alpha_logit_vector` target path, so the runner now fits ridge in constrained alpha-logit coordinates and decodes back to simplex alpha distributions for evaluation
  - on `gpt2-xl`, that geometry change improved the held-out routed-loss metric from `-0.0011` nats to `+0.0693` nats over uniform on average, with `5 / 8` confirm prompts improving
  - the descriptive alpha-recovery metrics got worse at the same time (`R^2 = -0.3291`, mean JS `= 0.2457` versus the previous `-0.2154` and `0.2377`), and the selected ridge penalty still saturated at `100.0`
  - strong interpretation therefore remains blocked; the next follow-up is a compressed target or another lower-dimensional reformulation rather than more raw logit-target iteration
- `known`: the first compressed predictiveness target is now implemented and compared:
  - `resattn-3ns` added `oracle_alpha_depth_type_band_logit_vector`, which compresses oracle alpha into deterministic depth-band-by-source-type group logits and lifts back to full-source alpha with train-only within-group templates
  - on `gpt2-xl`, that compressed target improved confirm descriptive alpha recovery relative to the full-source logit target (`R^2 = -0.2241`, mean JS `= 0.2380`) and broke the previous `100.0` ridge-saturation pattern by selecting `0.0001`
  - the held-out routed-loss metric fell back slightly below uniform (`-0.0031` nats, `3 / 8` prompts positive), so compression helped the descriptive side but not the current primary claim-bearing objective
  - the next follow-up is to make target and regularization selection loss-aware rather than to keep swapping targets blindly
- `known`: the loss-aware target-selection follow-up has now landed, and it did not clear the blocker:
  - `resattn-xaa` changed the saved predictiveness control plan so pilot tuning is now governed first by mean predicted routed-loss improvement over uniform and only secondarily by mean JS divergence
  - `validation/oracle_alpha_runner.py` can now compare multiple target parameterizations on the same feature surface during pilot tuning, and `scripts/run_oracle_alpha_heldout_predictiveness_check.py` exposes that path directly
  - on `gpt2-xl`, loss-aware pilot tuning across the raw-simplex, full-logit, and compressed-logit targets still selected the raw-simplex target with `lambda=100.0`
  - the confirm result reverted to the earlier raw-simplex token-aware baseline (`R^2 = -0.2154`, mean JS `= 0.2377`, predicted mean improvement `= -0.0011` nats, `4 / 8` prompts positive), while oracle alpha itself remained strong at `+1.3409` nats over uniform
  - the blocker is now sharper: target/regularization selection alone does not solve held-out predictiveness on the current `8 / 8` prompt split
- `known`: the first pilot-size redesign has now landed and improved the held-out result:
  - `resattn-7mb` introduced `prompts/registry_v2.yaml`, which doubles the oracle-alpha pilot split from `8` to `16` prompts while keeping the `8` confirm prompts fixed
  - on `gpt2-xl`, the same loss-aware target comparison now selected `oracle_alpha_logit_vector` with `lambda=0.01` instead of the raw-simplex target with `lambda=100.0`
  - the held-out confirm routed-loss metric turned positive again (`+0.0857` nats over uniform, `5 / 8` prompts positive) while confirm mean JS stayed at `0.2377`
  - confirm `R^2` remains negative (`-0.2616`), so pilot size looks like a real lever but not a full resolution of the predictiveness blocker
  - the next oracle-alpha follow-up is to scale the `registry_v2` path beyond the current `16 / 8` slice rather than to switch targets again
- `known`: the next scale-up kept the logit path alive on a larger saved split:
  - `resattn-0vx` introduced `prompts/registry_v3.yaml`, which expands the oracle-alpha prompt surface to `32` pilot prompts and `16` confirm prompts
  - on `gpt2-xl`, the same loss-aware target comparison still selected `oracle_alpha_logit_vector`, now with `lambda=100.0`
  - the held-out confirm routed-loss metric remained positive and strengthened in absolute terms (`+0.1277` nats over uniform, `12 / 16` prompts positive)
  - confirm `R^2` improved to `-0.1954`, while confirm mean JS was `0.2421`; descriptive alpha recovery is still weak, but the routed-loss signal now survives a meaningfully larger split
  - the next oracle-alpha follow-up is to scale this same logit path toward the prereg-sized Phase 1 gate rather than to change targets again
- `known`: the prereg-scale campaign build-out now exists even though the large launch has not started:
  - `validation/oracle_alpha_campaign.py` materializes prompt-level oracle checkpoints and feature-vector caches under a reusable campaign output directory
  - `scripts/run_oracle_alpha_predictiveness_campaign.py` launches that checkpointed path and writes a manifest, split-level oracle summaries, and a final predictiveness summary
  - `validation/oracle_alpha_runner.py` now accepts cached per-prompt oracle results and cached feature vectors, and each feature-source/target candidate records the full regularization grid rather than only the selected `lambda`
  - the repo is now operationally ready for tmux-backed prereg-scale runs, and the first `registry_v4` campaign artifact now exists under `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/`
- `known`: the next prereg-scale prompt surface is now frozen as `prompts/registry_v4.yaml`:
  - the default oracle-alpha collection now spans `96` pilot prompts and `128` confirm prompts, with prompt ids `oa-pilot-001` through `oa-pilot-096` and `oa-confirm-001` through `oa-confirm-128`
  - the tool-breakage factual-recall collection is unchanged from `registry_v3`
  - the repo has passed the full unit suite and pre-commit hooks after switching the default loader and main config to `registry_v4`
  - `resattn-9jq` has now completed its launch step and produced the first prereg-scale development-model artifact
- `known`: the first prereg-scale oracle-alpha campaign artifact now exists on the development model:
  - `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/` contains the saved `96 / 128` `gpt2-xl` campaign outputs plus prompt-level reusable checkpoints
  - the oracle confirm run clears the preregistered development-model loss gate strongly:
    - mean oracle improvement over uniform `= +1.2993` nats on `128` confirm prompts
    - paired one-sided t-test versus uniform `p = 9.72e-98`
    - paired Cohen's `d = 5.54`
    - all `128 / 128` confirm prompts improve over uniform
    - oracle alpha also beats the preregistered `random_dirichlet`, `magnitude_proportional`, and `last_layer_only` nulls on every confirm prompt
  - held-out predictiveness remains positive at prereg scale but still mixed:
    - selected target `oracle_alpha_vector` with ridge `100.0`
    - predicted mean improvement over uniform `= +0.1162` nats, `95 / 128` prompts positive, paired one-sided `p = 1.82e-11`, paired `d = 0.64`
    - confirm `R^2` stays negative at `-0.0884`, and mean JS to oracle alpha is `0.2427`
  - interpretation: the development-model oracle gate is now cleared, but strong interpretation remains blocked by weak alpha-shape recovery and by the lack of primary-model replication
- `known`: the prereg-scale campaign also surfaced a new operational follow-up:
  - `resattn-9co` tracks summary-stage observability, because top-level summary files stayed stale until the end of the long pilot tuning sweep even after the expensive oracle checkpoints were complete
- `known`: the first prereg-scale sequence-level pattern-analysis artifact now exists on the saved `registry_v4` confirm oracle outputs:
  - `validation/pattern_analysis.py` and `scripts/run_oracle_alpha_pattern_analysis.py` now provide the reusable sequence-level pattern-analysis slice over saved oracle run JSONs
  - `results/pattern_analysis/20260317-gpt2xl-prereg-scale-pattern-analysis-v1.json` summarizes the confirm-split `gpt2-xl` `final_alpha` distributions with source-type mass and average-linkage JSD clustering against a matched random Dirichlet control
  - the descriptive source-mass read is attention-heavy but diffuse: mean embedding mass `= 0.0349`, mean attention mass `= 0.5560`, mean MLP mass `= 0.4090`, mean entropy `= 3.8986`, and mean effective sources `= 49.4540`
  - the best oracle clustering result is weakly above the matched random control but still below the prereg block-structure gate: best silhouette `= 0.1428` at `k = 2` versus random-control `0.1093`
  - the apparent structure is outlier-driven rather than broad block structure: cluster sizes are `126 / 2` at `k = 2`, `119 / 5 / 2 / 2` at `k = 4`, and `114 / 4 / 3 / 2 / 2 / 1 / 1 / 1` at `k = 8`
  - interpretation: this is a valid Phase 2 entry artifact showing weak above-random routing structure on the development model, but it does not clear the prereg `silhouette > 0.2` gate and does not support a clean `~8`-cluster claim
- `known`: `resattn-qm4` is now resolved at the decision level:
  - tool-breakage stays on the primary `google/gemma-2-2b` lane and will use a custom Gemma-2 tuned lens trained locally rather than satisfying the tuned-lens requirement on a secondary model
  - a secondary-model tuned-lens comparison is allowed only as supplementary context, not as the primary confirmatory control
  - strong Figure 8 / trained-routing alignment claims require a small local AttnRes reproduction as the reproducible proxy
  - until that proxy exists, the Figure 8 lane is limited to comparison against the published AttnRes pattern surface rather than claims of direct trained-routing alignment
- `known`: `resattn-5k9` now has a real original-model viability artifact on the primary Gemma lane:
  - `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1.md` is the first full-surface custom tuned-lens pilot on `google/gemma-2-2b`
  - the pilot trained a low-rank affine residual translator on `96` oracle-alpha pilot prompts and evaluated on the `8` factual-recall pilot prompts from `tool_breakage_factual_recall_v1`
  - held-out mean KL to the final distribution improved from `11.3881` to `3.4580`, and mean held-out top-1 agreement improved from `0.1863` to `0.5119`
  - final-position held-out KL also improved from `14.8685` to `7.5721`, but final-position top-1 only moved from `0.1010` to `0.1250`
  - interpretation: the custom Gemma tuned-lens path is operationally viable and good enough to keep the primary tool-breakage lane on Gemma-2, but later routed-versus-original work should treat distributional metrics as the current strength and not overstate answer-token recovery
- `known`: the tuned-lens pilot is now checkpoint-hardened:
  - prompt-level residual caches and `training_state.pt` live under `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoints`
  - rerunning the exact launch command after completion reused the saved checkpoint tree successfully, so future routed-versus-original work can build on the same durability path rather than starting from a throwaway script
- `known`: `resattn-ehz` now fixes the tuned-lens metric hierarchy for the Gemma tool-breakage lane:
  - primary tuned-lens baseline metric: held-out KL to the model's final output distribution
  - required secondary diagnostics: mean top-1 agreement, final-position KL, and final-position top-1
  - interpretation lock: KL improvement alone is enough to keep the same-model tuned-lens baseline viable, but it does not license answer-token-facing factual-recall claims without explicit final-position evidence
- `known`: `resattn-28b` now provides the first same-model Gemma routed-versus-original factual-recall baseline artifact:
  - `validation/tool_breakage.py` implements the checkpointed prompt-level baseline runner and `scripts/run_tool_breakage_factual_recall_baseline.py` is the launch entry point
  - `prompts/registry_v4.yaml` now carries explicit `target_text` metadata for every `tool_breakage_factual_recall_v1` prompt so factual-recall token traces are defined by the saved registry rather than handwritten per run
  - `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1.md` is the first `8`-prompt pilot artifact comparing original versus routed traces under both raw and tuned lens on `google/gemma-2-2b`
  - the pilot shows broad KL-primary degradation under routing:
    - mean raw KL to final increased from `11.3515` to `16.9148`
    - mean tuned-lens KL to final increased from `3.3544` to `5.8709`
    - final-position tuned-lens KL increased from `7.5721` to `11.4121`
    - mean tuned top-1 dropped from `0.5152` to `0.3200`
    - final-position tuned top-1 dropped from `0.1250` to `0.0433`
  - prompt-level coverage is broad on the KL-primary metrics:
    - tuned mean KL worsened on `8 / 8` prompts
    - tuned final-position KL worsened on `8 / 8` prompts
    - raw mean KL worsened on `7 / 8` prompts
    - raw and tuned mean top-1 both worsened on `8 / 8` prompts
  - the prereg non-monotonicity boolean is already saturated on the original baseline (`8 / 8` prompts non-monotonic under both raw and tuned lens), so this pilot validates the runner and shows broad routed-versus-original degradation, but it does not yet clear the relative strong-claim threshold
  - `resattn-ypj` has now hardened the relative decision surface on the saved pilot traces:
    - routed-versus-original non-monotonicity increase is `0 / 8` under both raw and tuned lens
    - routed traces increase target-rank range on `4 / 8` prompts under the raw lens and `7 / 8` prompts under the tuned lens
    - routed traces worsen the best observed target rank on `4 / 8` prompts under both raw and tuned lens
    - routed traces worsen the final-layer target rank on `4 / 8` prompts under both raw and tuned lens
  - interpretation: the current pilot does not support a strong claim via the legacy non-monotonicity boolean, but it does support moving to a confirmatory run on a clearer rank-based instability surface
  - `resattn-6te` is now the next tool-breakage blocker: run the locked confirm split with the codified relative rank metrics before the later controlled dynamic-routing counterfactual

## Immediate Next Steps

1. Use `resattn-6te` to run the first confirmatory Gemma factual-recall baseline with the codified relative rank metrics.
2. Take `resattn-7hb` to build the small local AttnRes reproduction that will serve as the strong Figure 8 proxy.
3. Validate the refusal-feature discovery workflow in `resattn-3f1` before the safety lane becomes active.
4. Use `resattn-ojq` to test whether grouped-source and prompt-resampled clustering views produce a more robust pattern story than the current outlier-driven raw-source result.
5. Port the model-backed reconstruction smoke from the development model to the primary Gemma-2 lane when the Gemma-specific backend path is ready.

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
- `known`: the tuned-lens-aware comparison is now KL-primary on Gemma-2 factual recall; final-position metrics must still be reported and control answer-token-facing interpretation.
- `known`: the current Gemma pilot shows that the simple non-monotonicity boolean can saturate on the original-model baseline, so later tool-breakage decisions must use an explicitly relative routed-versus-original metric rather than that boolean alone; the current codified fallback is relative target-rank instability and target-rank degradation.
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
