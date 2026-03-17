# GPT-2 XL Prereg-Scale Pattern Analysis V1

## Motivation

`resattn-tpw` was the first prereg-scale pattern-analysis slice on the saved `registry_v4` oracle artifact after the development-model oracle gate cleared. The immediate question was not whether we could already make a broad Figure 8 or block-structure claim; it was whether the saved confirm-split oracle alphas showed any sequence-level structure above a simple matched random control, and whether that structure looked broad enough to justify pushing deeper into Phase 2.

## Methods

- Input artifact: `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/oracle_eval_run.json`
- Model / split: `gpt2-xl`, `128` confirm prompts from `prompts/registry_v4.yaml`
- Routing object: saved per-sequence `final_alpha` vectors over the `98` final-output residual sources
- Distance: Jensen-Shannon divergence between sequence-level alpha distributions
- Linkage: average
- Cluster scan: `k = 2..12`
- Structure null: symmetric Dirichlet control matched on sequence count and source dimension
- Descriptive summaries:
  - mean entropy and effective source count
  - top-1 source frequencies
  - mean mass on embeddings, attention outputs, and MLP outputs

## Results

- The confirm-split oracle alphas are diffuse rather than sharply concentrated:
  - mean entropy: `3.8986`
  - mean effective sources: `49.4540`
  - mean top-1 mass: `0.0444`
- Source-type mass is tilted toward attention over MLP, with embeddings small on average:
  - embedding mass: `0.0349`
  - attention mass: `0.5560`
  - MLP mass: `0.4090`
- Top-1 routing is not uniform across sources:
  - `pos_embed` is the most common top-1 source (`36 / 128`)
  - `0_attn_out` is next (`19 / 128`)
  - several later attention and MLP sources appear, but much less often
- The best average-linkage clustering result is above the matched random control, but only weakly:
  - oracle best silhouette: `0.1428` at `k = 2`
  - random-control best silhouette: `0.1093` at `k = 2`
- The apparent structure is not broad block structure:
  - `k = 2` partition sizes: `126, 2`
  - `k = 4` partition sizes: `119, 5, 2, 2`
  - `k = 8` partition sizes: `114, 4, 3, 2, 2, 1, 1, 1`
- The best `k = 2` split is therefore mostly an outlier carve-out, not a clean large-scale partition of prompts into multiple routing regimes.

## Limitations

- This is still a development-model result on `gpt2-xl`, not the primary Gemma-2 lane.
- The analysis is sequence-level and uses saved final-output `final_alpha` vectors only. It does not yet touch sublayer outputs, token-level routing variation, or Figure 8 metrics.
- The clustering signal remains below the prereg block-structure gate (`silhouette > 0.2`).
- Because the strongest split is dominated by a tiny outlier cluster, the current artifact is evidence for weak above-random structure, not for a broad `~8`-cluster block hypothesis.

## Next Steps

- Treat the current result as a valid Phase 2 entry artifact but not as a block-structure pass.
- Use `resattn-ojq` to test whether grouped-source views and prompt-resampling stability produce a more robust clustering story before making any stronger pattern claim.
- Keep Figure 8 and trained-routing-style language blocked until a richer source surface exists and the primary-model lane is ready.
