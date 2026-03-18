# ABOUTME: Summarizes the runtime benchmark for the adaptive ridge helper on the Gemma predictiveness shape.
# ABOUTME: Records the speedup over the legacy primal solve without changing predictiveness artifact semantics.

## Motivation

`resattn-b4q` targeted a pure implementation defect on the strongest live lane.
The primary-model held-out predictiveness artifact was scientifically valid, but
its ridge helper still solved the full high-dimensional primal system inside
leave-one-out tuning even when the train split was much narrower than the
feature dimension.

The goal here was not to change the method. It was to verify that the adaptive
ridge helper removes that avoidable bottleneck while preserving the old
predictions.

## Methods

- Surface matched to the saved primary-model Gemma predictiveness run:
  - pilot count `= 96`
  - confirm count `= 128`
  - feature source
    `= position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat`
  - feature dimension `= 9216`
  - target `= oracle_alpha_logit_vector`
  - target dimension `= 53`
  - regularization strength `= 100.0`
- Benchmark script:
  - `scripts/run_ridge_runtime_benchmark.py`
- Compared two implementations on synthetic arrays with the same matrix shapes:
  - legacy primal ridge solve
  - new adaptive helper from `validation/oracle_alpha_controls.py`
- Benchmarked two cases:
  - leave-one-out-shaped fold: `95 x 9216 -> 1 x 53`
  - full pilot-to-confirm fit: `96 x 9216 -> 128 x 53`

## Results

- Leave-one-out-shaped fold:
  - primal seconds `= 2.3811`
  - adaptive seconds `= 0.0259`
  - speedup `= 92.10x`
  - max abs prediction difference `= 1.15e-12`
- Full pilot-to-confirm fit:
  - primal seconds `= 2.2246`
  - adaptive seconds `= 0.0621`
  - speedup `= 35.82x`
  - max abs prediction difference `= 2.44e-12`

## Interpretation

- The old wall-clock problem was the wrong ridge formulation for the matrix
  shape, not a deeper limitation of the predictiveness path.
- The new helper preserves the old predictions to floating-point tolerance while
  making the narrow-train regime cheap enough that future reruns should not be
  dominated by the primal solve.
- This is an infrastructure result, not a scientific one. It changes how fast
  the method runs, not what the method means.

## Limitations

- The benchmark isolates the linear-algebra subproblem with synthetic arrays; it
  is not an end-to-end wall-clock claim for the full predictiveness runner.
- Other runtime costs still exist, especially the long summary-stage sweep and
  its current weak progress visibility.

## Next Steps

- Close `resattn-b4q`.
- Move the remaining oracle-lane infrastructure priority to `resattn-9co`.
- Keep future runtime debugging focused on summary-stage observability rather
  than on the ridge solve itself.
