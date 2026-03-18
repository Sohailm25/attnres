ABOUTME: Summarizes the first prereg-scale primary-model pattern analysis on the saved Gemma oracle artifact.
ABOUTME: Records whether the primary spine strengthens the coarse-structure story or the raw block-structure story.

# Motivation

`resattn-2sb` follows the first primary-model held-out oracle-alpha
predictiveness pass on `google/gemma-2-2b`. The next direct question is whether
the saved confirm-split Gemma oracle distributions show stronger raw-source
structure, stronger grouped coarse structure, or the same weak pattern seen on
the development-model lane.

# Methods

- Model: `google/gemma-2-2b`
- Source artifact:
  `results/oracle_alpha/20260317-gemma2-heldout-predictiveness-check-eval-run.json`
- Split: `confirm`
- Number of prompts: `128`
- Number of sources: `53`
- Clustering:
  - Jensen-Shannon distance between per-sequence `final_alpha` vectors
  - average linkage only
  - `k` scan from `2` through `12`
  - matched random Dirichlet control
- Grouped views:
  - `raw_source`
  - `source_type`
  - `depth_thirds_by_type`
- Resampling stability:
  - `64` prompt resamples
  - sample size `96`
  - oracle versus matched random control for each view

# Results

- Descriptive source-mass read:
  - mean embedding mass `= 0.0113`
  - mean attention mass `= 0.5502`
  - mean MLP mass `= 0.4384`
  - mean entropy `= 3.5003`
  - mean effective sources `= 33.2125`
  - mean top-1 mass `= 0.0858`
- Raw-source clustering stays weak:
  - oracle best silhouette `= 0.1084` at `k = 2`
  - random-control best silhouette `= 0.1598` at `k = 2`
  - oracle best cluster sizes `= 118 / 10`
  - random best cluster sizes `= 127 / 1`
  - resampling oracle-beats-random fraction `= 0.3594`
- Grouped coarse structure is stronger:
  - `source_type` oracle best silhouette `= 0.6731` versus random `0.5569`
  - `source_type` oracle best cluster sizes `= 66 / 56 / 6`
  - `source_type` resampling oracle-beats-random fraction `= 1.0000`
  - `depth_thirds_by_type` oracle best silhouette `= 0.3858` versus random `0.2305`
  - `depth_thirds_by_type` oracle best cluster sizes `= 121 / 7`
  - `depth_thirds_by_type` resampling oracle-beats-random fraction `= 0.8750`
- Top-1 mass is distributed across attention-heavy mid-depth sources rather than
  collapsing to one dominant layer:
  - `10_attn_out` appears `18` times
  - `14_attn_out` appears `12` times
  - `8_attn_out` appears `12` times

# Interpretation

- The primary model strengthens the grouped coarse-structure story relative to
  the development-model lane.
- The primary model does not strengthen the raw block-structure story. On the
  raw-source view, the matched random control still clusters at least as well as
  the oracle distributions, and the best split remains a highly imbalanced
  `118 / 10` `k = 2` partition.
- The honest update is therefore:
  - stronger coarse source-type and depth-band organization on the primary spine
  - no pass on the prereg raw block-structure gate
  - no clean `~8`-cluster story

# Limitations

- This analysis uses the current final-output residual-source oracle slice, not
  the broader multi-surface claim-bearing setup.
- The grouped views are informative but they compress the source space, so they
  support coarse-routing claims rather than a raw-source block-structure pass.
- The saved Gemma eval-run artifact is a compact per-sequence alpha packaging
  derived from the `resattn-js8` raw backup rather than a fresh oracle rerun.

# Next Steps

- Close `resattn-2sb` as a mixed primary-model pattern-analysis result.
- Keep the prereg raw block-structure gate unpassed on the primary model.
- Move the next core oracle issue to `resattn-5eo`, the primary-model softmax
  versus unconstrained versus top-k comparison.
