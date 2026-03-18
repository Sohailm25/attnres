ABOUTME: Summarizes the within-family factual route-mode characterization from the saved Gemma registry_v5 oracle artifact.
ABOUTME: Decides whether the next oracle-to-extension bridge should stay on factual families or shift to another stratum.

# Motivation

`resattn-2mx` follows the saved `registry_v5` Gemma oracle, stratified
pattern-analysis, and factual cluster-profile artifacts. Those earlier results
already established that factual recall is the strongest raw-source structure
surface on the primary model. The open question here was narrower:

- are the factual families internally differentiated in a meaningful way
- what actually distinguishes those within-family routing modes
- should the next bridge stay on factual families or shift to
  reasoning and math

# Methods

- Model: `google/gemma-2-2b`
- Source artifacts:
  - `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/oracle_eval_run.json`
  - `results/block_structure/20260318-gemma2-factual-recall-cluster-profile-v1.json`
- Prompt registry: `prompts/registry_v5.yaml`
- Subset: `stratum_factual_recall` on the locked `confirm` split
- Family tags:
  - `subcategory_capital_fact`
  - `subcategory_element_symbol`
  - `subcategory_author_fact`
  - `subcategory_moon_fact`
- Cluster count: reused saved factual best `k = 12`
- Profiling step:
  - assign the factual confirm prompts to the saved `k = 12` raw-source split
  - group the clusters back into factual families
  - summarize each family by:
    - number of routing modes
    - within-family centroid JS separation
    - family-level top sources
    - per-mode top sources
    - per-mode source enrichments relative to the family mean
    - per-mode source-type mass deltas relative to the family mean

# Results

## Family-level mode counts and separation

| Family | Prompt count | Mode count | Mode sizes | Mean within-family centroid JS |
|---|---:|---:|---|---:|
| author | `64` | `5` | `18 / 16 / 16 / 13 / 1` | `0.1533` |
| capital | `64` | `3` | `32 / 16 / 16` | `0.1610` |
| element | `64` | `3` | `32 / 16 / 16` | `0.2096` |
| moon | `64` | `2` | `32 / 32` | `0.1770` |

The strongest factual stratum is therefore not just family-pure. Every factual
family splits into multiple route modes, and the element family has the largest
within-family mode separation.

## Route-mode signatures are mechanistic, not just semantic labels

- Capitals split into three clear regimes:
  - one attention-heavier mode led by `17_attn_out` with positive attention
    shift `(+0.0529)` and negative MLP shift `(-0.0547)`
  - one MLP-heavier mode led by `2_mlp_out` and `14_mlp_out`
  - one larger mixed mode with a milder MLP tilt and enriched
    `12_attn_out` / `6_mlp_out`
- Elements split into three strongly separated regimes:
  - one MLP-heavier mode enriched on `14_attn_out` and `10_mlp_out`
  - one near-balanced attention mode led by `19_attn_out` and `11_attn_out`
  - one attention-heavier mode enriched on `5_attn_out` and `18_attn_out`
- Authors fragment the most:
  - one clearly MLP-heavy mode led by `8_mlp_out`, `12_mlp_out`, and
    `16_mlp_out`
  - multiple attention-heavier or mixed modes led by `6_attn_out`,
    `18_attn_out`, `5_attn_out`, and `4_attn_out`
  - one singleton attention-heavy outlier
- Moons split cleanly into two modes:
  - an attention-heavier mode enriched on `12_attn_out` and `16_attn_out`
  - an MLP-heavier mode enriched on `4_mlp_out` and `11_mlp_out`

## Bridge decision

The next oracle-to-extension bridge should stay on factual families, not shift
to reasoning and math yet.

The reason is not only that factual recall still has the stronger raw-source
structure (`best silhouette = 0.4709` versus `0.2456` for reasoning/math). It
is that factual recall now also has a richer and more actionable internal mode
story:

- capitals, elements, and authors each expose multiple route modes with clear
  source signatures
- those families are already the cleanest conceptual bridge into the bounded
  tool-breakage lane
- moons are structurally real, but they remain extension-risky because the
  current one-token tool-breakage lane is still mixed on moon prompts

So the next bridge question is no longer family-only coverage. It is
mode-aware coverage inside the factual families we already know matter.

# Limitations

- This is still a saved-artifact sequence-level analysis.
- The cluster count is inherited from the saved factual best `k = 12` result,
  not re-optimized for a different downstream purpose.
- The singleton author mode should be treated as an outlier until a later pass
  shows it is stable under a stricter robustness check.
- This does not reopen the strong same-model tool-breakage claim. It only
  sharpens where the next bridge should look.

# Next Steps

- Close `resattn-2mx`.
- Take `resattn-unp` next: audit one-token tool-breakage `v4` coverage of these
  factual route modes rather than only the family labels.
- Keep `resattn-a1w` as the bounded moon-family sidecar, not as the main bridge
  issue.
- Do not shift the main extension bridge to reasoning/math yet.
