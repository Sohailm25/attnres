ABOUTME: Audits whether the saved Gemma registry_v5 oracle campaign already contains the supervision needed for Phase 6 router distillation.
ABOUTME: Records the smallest honest next implementation slice for router distillation and w_l-analog geometry.

# Gemma-2 Router-Distillation Readiness Audit v1

## Motivation

The primary-model Gemma oracle lane is now strong enough that the next core
interpretability move should not be another broad oracle rerun by inertia. The
prereg Phase 6 lane is router distillation plus `w_l`-analog geometry, but that
lane is only ready if the repo already has the right supervision cached.

`resattn-5qd` answers the narrow readiness question:

- do the saved `registry_v5` campaign artifacts already contain the per-token
  supervision needed for Phase 6
- can a saved-artifact export start the pilot router lane immediately
- if not, what is the smallest honest implementation slice

## Methods

- Saved campaign:
  `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/`
- Checkpoint code:
  `validation/oracle_alpha_campaign.py`
- Runner code:
  `validation/oracle_alpha_runner.py`
- Prereg target:
  `history/PREREG.md`, Phase 6
- Audit procedure:
  - inspect the saved campaign directory structure
  - inspect one oracle checkpoint and one feature-vector checkpoint
  - inspect the checkpoint dataclasses and feature-vector builder in code
  - compare what is persisted against what Phase 6 requires

Machine-readable artifact:

- `results/router_training/20260318-gemma2-router-distillation-readiness-audit-v1.json`

## Results

### What the saved campaign already has

The saved `registry_v5` campaign is strong for sequence-level predictiveness:

- prompt-level oracle checkpoints exist under
  `checkpoints/oracle_runs/{pilot,confirm}/`
- feature-vector checkpoints exist under
  `checkpoints/feature_vectors/{pilot,confirm}/`
- the oracle checkpoints persist, per sequence:
  - `prompt_id`
  - `prompt`
  - `split`
  - `num_sources`
  - `source_labels`
  - `uniform_loss`
  - `optimized_loss`
  - `null_losses`
  - `best_alpha`
  - `final_alpha`
- the feature checkpoints persist, per sequence:
  - `prompt_id`
  - `split`
  - one aggregated feature vector under `values`

So the current campaign is already sufficient for the existing sequence-level
oracle and predictiveness lane.

### What the saved campaign does not have

The Phase 6 prereg lane is defined on per-token router inputs:

- router architecture: `2`-layer MLP on `h_1[t]`
- pilot comparison: `h_1[t]` versus an early contextual state such as `h_4[t]`
- learned router query vectors analyzed as a `w_l` analog

The current saved campaign does not persist the data needed for that lane:

- no token ids per prompt checkpoint
- no per-token hidden-state tables
- no cached `h_1[t]`
- no cached `h_4[t]`
- no per-token oracle target
- no saved token-to-sequence aggregation object for a router loss

This is visible in both code and artifacts:

- `OracleAlphaSequenceResult` only stores sequence-level losses and alpha
  vectors
- `_save_feature_vector` only writes one aggregated vector per prompt
- `_resid_post_states` and `_fixed_residual_sources` compute token-level states
  internally, but those tensors are discarded after sequence-level summaries are
  built

So a saved-artifact export from the current campaign cannot start the prereg
router-distillation pilot by itself.

## Interpretation

The main blocker is now concrete:

- the Gemma oracle lane is scientifically ready for Phase 6
- the Gemma campaign artifacts are not operationally ready for Phase 6

That is a good problem. It means the next core move is not another expensive
oracle campaign. It is a targeted data/export implementation slice.

The smallest honest next implementation slice is:

1. Extend the oracle campaign checkpointing path to optionally save pilot-only
   per-token export artifacts.
2. Persist, per pilot prompt:
   - token ids
   - `h_1[t]`
   - `h_4[t]`
   - the sequence-level `final_alpha`
   - prompt metadata needed to join back to the oracle result
3. Build a tiny Phase 6 pilot dataset export on top of those checkpoints before
   any router training code is written.
4. Only after that, fit the prereg pilot comparison:
   - `h_1[t]` input versus `h_4[t]` input
   - sequence-level aggregation choice logged explicitly

That is the smallest slice that preserves truthfulness. Anything smaller still
leaves the training lane blocked by missing supervision.

## Limitations

- This is a readiness audit, not a training artifact.
- The audit does not resolve the future token-to-sequence aggregation choice for
  router training; it only shows that the current saved campaign does not store
  enough information to test that choice.
- No new oracle run was launched here.

## Next Steps

- Close `resattn-5qd`.
- Open the concrete follow-up implementation issue for pilot-only per-token
  export.
- Keep the main story centered on the saved Gemma oracle results while Phase 6
  data readiness catches up.
