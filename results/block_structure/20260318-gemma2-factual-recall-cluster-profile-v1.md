ABOUTME: Summarizes the factual-recall raw-source cluster profile from the saved Gemma registry_v5 oracle artifact.
ABOUTME: Records whether the strongest raw-source silhouette on the primary model is semantically coherent enough to matter for the block-structure lane.

# Motivation

`resattn-8y4` follows the stratified `registry_v5` pattern-analysis artifact.
That run showed that the strongest raw-source structure on the primary model was
not global. It was concentrated in the factual-recall subset. The next honest
question was whether that subset-level raw-source result reflects:

- meaningful semantic organization
- different routing modes within factual recall
- or just prompt-template shards

# Methods

- Model: `google/gemma-2-2b`
- Source artifact:
  `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/oracle_eval_run.json`
- Prompt registry: `prompts/registry_v5.yaml`
- Subset: `stratum_factual_recall` on the `confirm` split
- Number of prompts: `256`
- Raw-source clustering:
  - Jensen-Shannon divergence between per-sequence `final_alpha` vectors
  - average linkage
  - `k` scan from `2` through `12`
  - matched random Dirichlet control
- Profiling step:
  - use the factual-recall best `k = 12` raw-source split
  - profile cluster purity by saved `subcategory_*` tag
  - report source-type mass and top mean source labels per cluster

# Results

- The factual-recall raw-source result stays strong:
  - oracle best silhouette `= 0.4709`
  - random best silhouette `= 0.1401`
  - best `k = 12`
  - cluster sizes
    `= 33 / 32 / 32 / 32 / 18 / 16 / 16 / 16 / 16 / 16 / 16 / 13`
  - largest cluster fraction `= 0.1289`
- The clusters are almost perfectly pure by saved factual subcategory:
  - `subcategory_capital_fact` occupies `3` clusters
  - `subcategory_element_symbol` occupies `3` clusters
  - `subcategory_author_fact` occupies `5` clusters
  - `subcategory_moon_fact` occupies `2` clusters
  - all clusters are fully pure except one `33`-prompt moon-fact cluster with a
    single author-fact outlier
- The subcategories do not collapse to one routing mode each:
  - capitals split into three modes
  - elements split into three modes
  - authors split into five modes
  - moon facts split into two modes
- The cluster profiles show different source-usage regimes, not just label
  purity:
  - capital clusters:
    - one cluster is strongly attention-heavy (`0.6300` attention mass) with
      `17_attn_out` on top
    - another keeps a stronger `2_mlp_out` anchor with more balanced
      attention/MLP mass
  - moon-fact clusters:
    - one cluster is MLP-heavier (`0.5307` MLP mass)
    - the other is more attention-heavy (`0.5638` attention mass)
  - author-fact clusters fragment into several distinct modes, including a more
    MLP-heavy cluster led by `16_mlp_out`

# Interpretation

- This is not a toy clustering artifact. The strongest raw-source result on the
  primary model is semantically organized and internally differentiated.
- The right reading is not “Gemma has a clean global `~8`-cluster raw routing
  structure.” The right reading is narrower:
  - the full mixed-surface prereg gate is still unpassed
  - factual recall contains a real family-conditioned raw-source structure
  - those factual families themselves split into multiple routing modes
- This matters because the repo's bounded same-model tool-breakage lane is also
  factual-recall-based. Capitals, elements, and author facts now look like the
  most natural bridge between the strongest raw-source structure result and the
  existing tool-breakage evidence.

# Limitations

- This artifact is subset-conditioned and does not change the full mixed-surface
  prereg gate.
- The factual-recall result still peaks at `k = 12`, not `~8`.
- This is still a final-output, sequence-level analysis rather than token-level
  or sublayer-level routing structure.

# Next Steps

- Close `resattn-8y4`.
- Use the factual-recall cluster families to define the next bridge into the
  bounded Gemma tool-breakage lane.
- Keep the full mixed-surface raw block-structure gate unpassed.
