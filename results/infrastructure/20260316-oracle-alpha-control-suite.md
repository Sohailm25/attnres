# Oracle-Alpha Phase 1 Control Suite

## Motivation

The repo already required stability reporting, held-out alpha predictiveness, and a MIB benchmark anchor before claim-bearing oracle-alpha interpretation. Those requirements existed only as control-doc language, not as a durable config or reusable code path.

## Methods

- Added `configs/oracle_alpha_controls_v1.yaml` as the saved Phase 1 control registry for the current oracle-alpha lane.
- Implemented `validation/oracle_alpha_controls.py` to:
  - load and validate the control registry against the saved prompt registry
  - compute bootstrap confidence intervals over sequence-level means
  - summarize restart stability with pairwise Jensen-Shannon divergence and top-k Jaccard overlap
  - summarize held-out alpha predictiveness with linear regression, `R^2`, and mean Jensen-Shannon divergence
  - force explicit `planned` versus `omitted` MIB handling
- Added `tests/test_oracle_alpha_controls.py` to pin the config contract and helper behavior.

## Results

- The current Phase 1 oracle-alpha control plan is now saved under `configs/oracle_alpha_controls_v1.yaml`.
- The plan fixes:
  - sequence-level aggregation
  - `1000` bootstrap resamples
  - restart seeds `11, 17, 23, 31, 47`
  - prompt perturbations `prompt_resample` and `prompt_paraphrase`
  - held-out predictiveness from the pilot split to the confirmatory split
- The current MIB status is `planned`, with a written rationale and revisit trigger rather than a silent omission.
- The helpers and config are covered by tests, but no oracle-alpha optimization has been run yet.

## Limitations

- This is infrastructure, not a scientific result.
- The registry currently covers the Phase 1 oracle-alpha lane only.
- The MIB anchor is planned but not executed because the repo still lacks an oracle-alpha runner and MIB adapter.

## Next Steps

- Build the first oracle-alpha execution harness and make it consume the saved control registry by default.
- Revisit the MIB plan once the runner can support a controlled sanity task.
