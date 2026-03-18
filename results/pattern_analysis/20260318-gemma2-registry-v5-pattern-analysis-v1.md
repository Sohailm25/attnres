ABOUTME: Summarizes the grouped-source and per-stratum follow-up on the first full stratified Gemma registry_v5 oracle artifact.
ABOUTME: Records whether the broadened primary-model routing structure is broad, grouped, and weakly aggregate or concentrated in specific prompt strata.

# Motivation

`resattn-xot` follows the first full stratified `registry_v5` Gemma oracle run.
That run answered the scale question cleanly on loss recovery and held-out
predictiveness, but it left a more interesting structure question open:

- does the broader primary-model artifact now carry stronger raw-source
  structure overall?
- or is the new signal concentrated inside particular prompt families?

This follow-up reuses the saved oracle artifact instead of launching another
Gemma run.

# Methods

- Model: `google/gemma-2-2b`
- Source artifact:
  `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/oracle_eval_run.json`
- Prompt registry: `prompts/registry_v5.yaml`
- Split: `confirm`
- Number of prompts: `1024`
- Number of sources: `53`
- Views:
  - `raw_source`
  - `source_type`
  - `depth_thirds_by_type`
- Subset groups: registry tags beginning with `stratum_`
- Clustering:
  - Jensen-Shannon divergence between per-sequence `final_alpha` vectors
  - average linkage only
  - `k` scan from `2` through `12`
  - matched random Dirichlet control
- Resampling stability:
  - `32` resamples
  - fixed sample size `192`
  - same resampling budget used for the full artifact and each stratum subset

# Results

## Full mixed surface

- Descriptive source-mass read:
  - mean embedding mass `= 0.0114`
  - mean attention mass `= 0.5344`
  - mean MLP mass `= 0.4542`
  - mean entropy `= 3.5186`
  - mean effective sources `= 33.8275`
- Full-sample raw-source structure is only weakly above the random control and
  remains aggregate-heavy:
  - oracle best silhouette `= 0.1664`
  - random best silhouette `= 0.1417`
  - best `k = 2`
  - cluster sizes `= 1023 / 1`
  - resampling oracle-beats-random fraction `= 0.0938`
- Full-sample grouped views are strong and robust:
  - `source_type`:
    - oracle best silhouette `= 0.7558`
    - random best silhouette `= 0.4681`
    - best `k = 2`
    - cluster sizes `= 1020 / 4`
    - resampling oracle-beats-random fraction `= 1.0`
  - `depth_thirds_by_type`:
    - oracle best silhouette `= 0.5168`
    - random best silhouette `= 0.1433`
    - best `k = 2`
    - cluster sizes `= 1022 / 2`
    - resampling oracle-beats-random fraction `= 1.0`

## Stratum-conditioned raw-source structure

- Factual recall is the clearest raw-source result in the repo so far:
  - raw-source oracle best silhouette `= 0.4709`
  - raw-source random best silhouette `= 0.1401`
  - raw-source best `k = 12`
  - raw-source cluster sizes
    `= 33 / 32 / 32 / 32 / 18 / 16 / 16 / 16 / 16 / 16 / 16 / 13`
  - raw-source resampling oracle-beats-random fraction `= 1.0`
- Reasoning and math also shows real raw-source structure:
  - raw-source oracle best silhouette `= 0.2456`
  - raw-source random best silhouette `= 0.1401`
  - raw-source best `k = 12`
  - raw-source resampling oracle-beats-random fraction `= 1.0`
- Code/procedural and general-text raw-source views stay weak:
  - code/procedural:
    - oracle best silhouette `= 0.1339`
    - random best silhouette `= 0.1401`
    - raw-source resampling oracle-beats-random fraction `= 0.375`
  - general text:
    - oracle best silhouette `= 0.1301`
    - random best silhouette `= 0.1401`
    - best `k = 2`
    - cluster sizes `= 254 / 2`
    - raw-source resampling oracle-beats-random fraction `= 0.125`

## Stratum-conditioned grouped views

- All four strata retain strong grouped coarse structure, even when their
  raw-source views differ:
  - `source_type` silhouettes range from `0.6095` to `0.7153`
  - `depth_thirds_by_type` silhouettes range from `0.4482` to `0.6044`
  - both grouped views beat the matched random control on `32 / 32` resamples
    for every stratum

# Interpretation

- The broadened primary-model structure story is now clearly two-level:
  - broad, robust grouped coarse structure across the full prompt surface
  - stronger raw-source structure only inside tighter semantic strata
- The aggregate raw-source prereg gate is still unpassed on the full mixed
  surface. The best full-sample raw split is still effectively binary and
  heavily imbalanced.
- The promising underexplored result is factual recall, not the mixed pool.
  That stratum now carries the strongest raw-source structure and also has the
  cleanest conceptual bridge to the bounded Gemma tool-breakage lane.
- Reasoning/math also looks real, but less immediately central to the repo's
  strongest existing cross-lane story.

# Limitations

- The fixed `192`-prompt resample size is a runtime-conscious budget for this
  larger artifact, so the resampling numbers should be compared within this
  artifact rather than directly against earlier `75%`-sample analyses.
- This remains a final-output, sequence-level analysis. It still does not test
  token-level routing variation or sublayer-level Figure 8 structure.
- Factual recall's raw-source result does not by itself pass the exact prereg
  `~8`-cluster story; its best `k` is `12`, not `8`.

# Next Steps

- Close `resattn-xot`.
- Open the next oracle follow-up around factual-recall-focused raw-source
  structure, not another mixed-surface aggregate rerun.
- Keep the mixed full-surface raw block-structure gate unpassed even while
  promoting the grouped-structure and factual-recall-stratum results.
