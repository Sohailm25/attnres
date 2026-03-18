# ABOUTME: Summarizes the grouped-source and prompt-resampling robustness follow-up on the prereg-scale GPT-2 XL routing pattern artifact.
# ABOUTME: Records whether the weak raw-source clustering signal survives control extensions and whether any broader grouped-view structure emerges.

## Motivation

`resattn-ojq` followed the first prereg-scale pattern-analysis artifact, which
showed weak above-random clustering on raw `final_alpha` vectors but looked
heavily outlier-driven (`126 / 2` at `k = 2`). The next honest question was not
“can we squeeze a better silhouette out of the same artifact?” It was whether
the apparent structure survives two harder checks:

- grouped-source compressions that collapse the `98` raw sources into coarser,
  prereg-consistent views
- prompt-resampling stability on the confirm split rather than one full-sample
  clustering read

## Methods

- Input artifact:
  `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/oracle_eval_run.json`
- Model / split: `gpt2-xl`, `128` confirm prompts from `registry_v4`
- Raw routing object: per-sequence `final_alpha` over `98` final-output sources
- Grouped-source views:
  - `raw_source` (`98` sources)
  - `source_type` (`embedding`, `attention`, `mlp`)
  - `depth_thirds_by_type` (`embedding` plus early / middle / late attention and
    MLP bands; `7` groups total)
- Distance: Jensen-Shannon divergence
- Linkage: average
- Cluster scan: `k = 2..12`
- Random control: matched symmetric Dirichlet for each view
- Prompt-resampling stability:
  - `128` resamples
  - sample size `96` prompts without replacement per resample

## Results

### Raw-source view stays weak and outlier-driven

- The original raw-source result is unchanged at full sample:
  - oracle best silhouette `= 0.1428`
  - random-control best silhouette `= 0.1093`
  - delta `= +0.0335`
- Cluster dominance remains extreme:
  - `k = 2`: `126 / 2` (largest fraction `= 0.9844`)
  - `k = 4`: `119 / 5 / 2 / 2`
  - `k = 8`: `114 / 4 / 3 / 2 / 2 / 1 / 1 / 1` (largest fraction `= 0.8906`)
- Resampling confirms this is a stable weak result, not a one-shot fluke:
  - oracle best silhouette mean `= 0.1356 ± 0.0153`
  - random best silhouette mean `= 0.1118`
  - oracle beats random on `102 / 128` resamples (`0.7969`)
  - best `k` is `2` on all `128 / 128` resamples
  - mean largest-cluster fraction stays very high:
    - `k = 2`: `0.9793`
    - `k = 8`: `0.8546`

### Grouped views reveal stronger coarse structure

- The `source_type` view shows the strongest grouped signal:
  - oracle best silhouette `= 0.6652`
  - random best silhouette `= 0.5569`
  - delta `= +0.1082`
  - `k = 2`: `87 / 41`
  - `k = 8`: `42 / 34 / 31 / 7 / 6 / 4 / 3 / 1`
  - resampling:
    - oracle best silhouette mean `= 0.6432 ± 0.0380`
    - oracle beats random on `125 / 128` resamples (`0.9766`)
    - best `k` is usually `2` or `3`, not cleanly fixed
- The `depth_thirds_by_type` view strengthens the signal relative to raw
  sources, but it does not remove the main imbalance:
  - oracle best silhouette `= 0.3451`
  - random best silhouette `= 0.2305`
  - delta `= +0.1146`
  - `k = 2`: `117 / 11`
  - `k = 8`: `95 / 12 / 9 / 4 / 3 / 3 / 1 / 1`
  - resampling:
    - oracle best silhouette mean `= 0.4555 ± 0.0920`
    - oracle beats random on `117 / 128` resamples (`0.9141`)
    - best `k` is still `2` on `121 / 128` resamples

## Interpretation

- This follow-up strengthens the pattern-analysis lane, but not in the
  preregistered way.
- The raw-source routing structure is now clearly real enough to survive
  resampling, yet it still looks like weak, mostly binary, outlier-heavy
  structure rather than a broad `~8`-cluster block decomposition.
- Grouped-source views reveal a stronger coarse pattern, especially when the
  routing mass is compressed to source types. The honest interpretation is that
  sequence-level routing differs meaningfully in coarse attention-versus-MLP
  balance across prompts.
- That does **not** amount to a prereg block-structure pass:
  - the grouped views raise the matched random-control silhouette too
  - the strongest grouped view is only `3` dimensions and therefore too coarse
    to support a meaningful `~8`-cluster claim
  - the `7`-group depth-banded view still collapses mostly to `k = 2`

## Limitations

- This remains a development-model result on `gpt2-xl`, not the primary Gemma
  lane.
- The analysis is sequence-level and final-alpha-only; it still does not test
  token-level routing variation or sublayer-level Figure 8 structure.
- Grouped-source compression can make coarse regimes look cleaner simply by
  reducing dimensionality, which is why the matched random-control comparison is
  still essential.

## Next Steps

- Close `resattn-ojq` as a useful robustness follow-up, not as a block-structure
  pass.
- Keep the prereg `silhouette > 0.2` raw block-structure gate unpassed.
- Treat the grouped-view result as evidence for coarse source-type routing
  variation while keeping stronger block-structure language blocked.
