# ABOUTME: Synthesizes the strongest factual-recall routing result on the primary Gemma spine from saved artifacts only.
# ABOUTME: States what the factual route modes support, what they do not support, and how they connect to the bounded tool-breakage bridge.

# Gemma-2 Factual-Recall Routing Synthesis v1

## Motivation

The saved `registry_v5` Gemma oracle artifact changed the center of gravity of
the repo.

The strongest broad result is now primary-model effective depth mixture on
Gemma. But the strongest *structured* raw-source result is narrower:

- it is not the full mixed prompt surface
- it is not a clean global raw `~8`-cluster story
- it is factual recall

`resattn-chz` exists to make that narrower claim explicit and useful. It
reuses the strongest saved factual artifacts and answers four questions in one
place:

1. What is actually structured inside factual recall?
2. What distinguishes the factual route modes mechanistically?
3. How much of that structure is already represented in the bounded
   tool-breakage bridge?
4. Which claims does this enable, and which claims remain blocked?

## Methods

This is a saved-artifact synthesis only. No new oracle or tool-breakage run was
launched.

Source artifacts:

- broad oracle anchor:
  `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1.md`
- factual cluster profile:
  `results/block_structure/20260318-gemma2-factual-recall-cluster-profile-v1.md`
- factual route modes:
  `results/block_structure/20260318-gemma2-factual-route-modes-v1.md`
- family bridge:
  `results/tool_breakage/20260318-gemma2-factual-routing-tool-breakage-bridge-v1.md`
- route-mode bridge:
  `results/tool_breakage/20260318-gemma2-tool-breakage-factual-route-mode-bridge-v5.md`

## Results

### 1. Factual recall is the strongest semantically organized raw-source structure in the repo

On the saved `registry_v5` Gemma artifact:

- full-surface raw-source structure is only weakly above random and remains
  aggregate-heavy
- factual recall is the strongest subset-conditioned raw-source result:
  - best silhouette `= 0.4709`
  - random-control best silhouette `= 0.1401`
  - best `k = 12`
  - cluster sizes
    `= 33 / 32 / 32 / 32 / 18 / 16 / 16 / 16 / 16 / 16 / 16 / 13`
  - resampling oracle-beats-random fraction `= 1.0`

This already makes the right high-level claim narrower than the original block
hypothesis:

- the repo does **not** have a global raw-source `~8`-cluster result
- the repo **does** have a strong stratum-conditioned raw-source result inside
  factual recall

### 2. The factual raw-source result is not just family-pure; it splits into multiple route modes

The factual clusters are nearly perfectly pure by saved factual subcategory:

- capitals occupy `3` clusters
- elements occupy `3` clusters
- authors occupy `5` clusters
- moons occupy `2` clusters
- all clusters are fully pure except one `33`-prompt moon cluster containing a
  single author outlier

Within-family route-mode structure:

| Family | Prompt count | Mode count | Mode sizes | Mean within-family centroid JS |
|---|---:|---:|---|---:|
| author | `64` | `5` | `18 / 16 / 16 / 13 / 1` | `0.1533` |
| capital | `64` | `3` | `32 / 16 / 16` | `0.1610` |
| element | `64` | `3` | `32 / 16 / 16` | `0.2096` |
| moon | `64` | `2` | `32 / 32` | `0.1770` |

Two important points follow:

- factual recall is not one routing regime per factual family
- the element family has the strongest within-family separation, while authors
  have the most modes

### 3. The route modes are mechanistically different, not just label shards

The route modes differ in source usage in ways that are easy to summarize and
hard to dismiss as random noise.

Capitals:

- family mean is anchored by `17_attn_out`, `2_mlp_out`, `8_attn_out`,
  `2_attn_out`
- three modes split into:
  - cluster `2`: attention-heavier map-style mode
    - attention delta `= +0.0529`
    - top sources: `17_attn_out`, `2_attn_out`, `21_attn_out`
  - cluster `4`: MLP-heavier quiz-style mode
    - MLP delta `= +0.0210`
    - top sources: `2_mlp_out`, `14_mlp_out`, `18_attn_out`
  - cluster `3`: larger mixed direct-fact mode
    - mild MLP tilt
    - top sources: `2_mlp_out`, `17_attn_out`, `8_attn_out`, `12_attn_out`

Elements:

- family mean is anchored by `1_attn_out`, `15_attn_out`, `14_attn_out`
- three modes split into:
  - cluster `5`: attention-heavier abbreviated/table mode
    - attention delta `= +0.0626`
    - top sources: `5_attn_out`, `18_attn_out`, `8_attn_out`
  - cluster `9`: MLP-heavier chart/notes mode
    - MLP delta `= +0.0301`
    - top sources: `14_attn_out`, `10_mlp_out`, `1_attn_out`
  - cluster `1`: near-balanced symbol-definition mode
    - source-type deltas near zero
    - top sources: `19_attn_out`, `11_attn_out`, `1_attn_out`, `5_mlp_out`

Authors:

- family mean is anchored by `6_attn_out`, `12_attn_out`, `16_mlp_out`,
  `6_mlp_out`, `2_mlp_out`
- four non-outlier modes are practically relevant:
  - cluster `6`: clearly MLP-heavy literature-students mode
    - attention delta `= -0.0613`
    - MLP delta `= +0.0645`
    - top sources: `16_mlp_out`, `8_mlp_out`, `12_mlp_out`
  - cluster `7`: strong breakage catalog/indexing mode
    - top sources: `4_attn_out`, `12_attn_out`, `2_attn_out`, `3_mlp_out`
  - cluster `11`: attention-heavier novel-title mode
    - attention delta `= +0.0297`
    - top sources: `6_attn_out`, `18_attn_out`, `2_mlp_out`, `8_attn_out`
  - cluster `12`: mixed direct author/novel mode
    - top sources: `6_mlp_out`, `12_attn_out`, `2_mlp_out`, `6_attn_out`
- cluster `10` is a singleton attention-heavy outlier and should stay treated
  as such

### 4. The route modes appear partly prompt-frame-conditioned, not purely content-conditioned

The saved example prompts strongly suggest that many within-family modes track
prompt framing as well as answer family.

Examples:

- capitals:
  - cluster `2`: `On most maps, the capital of ... appears as`
  - cluster `4`: `In a geography quiz, the capital of ... would be`
  - cluster `3`: direct-fact variants such as `The capital city of ... is`
    and `Travel guides note that the capital of ... is`
- elements:
  - cluster `1`: `The chemical symbol for ... is`
  - cluster `5`: `In the periodic table, ... is abbreviated as`
  - cluster `9`: `Chemistry notes write ... as` and `A lab chart would mark ...`
- authors:
  - cluster `6`: `Literature students learn that ... was written by`
  - cluster `7`: `Most library catalogs list ... under`
  - cluster `11`: `The novel ... was written by`
  - cluster `12`: mixed direct prompts including `The author of ... is`

This is a meaningful interpretability result, but it sharpens the claim
boundary:

- the route modes are not just about semantic answer family
- they appear to reflect a family-plus-prompt-frame routing structure
- the repo has **not** yet cleanly disentangled prompt frame from latent task
  computation causally

### 5. The bounded factual tool-breakage bridge is already well aligned for the core families

The narrowed `tool_breakage_factual_recall_v5` bridge covers the strongest
factual route modes much more completely than the older `v4` surface:

- total factual modes: `13`
- modes with matching-family `v5` assignment: `10 / 13`

Family coverage:

| Family | Matching-family modes covered by `v5` | Uncovered |
|---|---:|---|
| author | `4 / 5` | singleton outlier `10` |
| capital | `3 / 3` | none |
| element | `3 / 3` | none |
| moon | `0 / 2` | `8 / 10` |

So the core factual bridge now covers:

- all capital modes
- all element modes
- all non-outlier author modes

It also spans both mild and strong routed-versus-original breakage modes:

- strongest covered mode:
  - `subcategory_author_fact::cluster_7`
  - mean tuned final-position KL delta `= +12.7356`
- other strong covered modes:
  - `subcategory_author_fact::cluster_12 = +5.1216`
  - `subcategory_element_symbol::cluster_9 = +4.7830`
  - `subcategory_capital_fact::cluster_3 = +4.6148`
  - `subcategory_element_symbol::cluster_5 = +4.4919`
- milder covered modes still remain positive:
  - `subcategory_author_fact::cluster_6 = +1.7056`
  - `subcategory_capital_fact::cluster_2 = +2.1446`
  - `subcategory_capital_fact::cluster_4 = +2.2515`
  - `subcategory_element_symbol::cluster_1 = +2.2107`
  - `subcategory_author_fact::cluster_11 = +2.7926`

That matters because it removes a weak explanation for the mixed donor-arm
result. The current bounded tool-breakage lane is no longer obviously mixed
because it failed to cover the important factual route modes.

## Claims Enabled

This synthesis supports the following stronger repo-level claims:

- On the primary Gemma spine, the strongest raw-source routing structure is
  concentrated in factual recall rather than in the full mixed surface.
- Within factual recall, routing is meaningfully structured below the family
  level: capitals, elements, and authors each split into multiple route modes
  with distinct source-usage signatures.
- The strongest primary-model factual route modes are already represented in the
  narrowed `v5` factual tool-breakage bridge for capitals, elements, and
  non-outlier authors.
- The mixed donor-arm boundary in tool-breakage should now be read as a
  heterogeneity/control issue, not primarily as a bridge-coverage failure.

## Claims Still Blocked

This synthesis does **not** support the following claims:

- a global raw-source `~8`-cluster result on the full prompt surface
- a clean content-only routing story disentangled from prompt frame
- a broad prompt-specific same-model donor-arm tool-breakage claim
- a trained-routing or Figure 8 alignment claim
- any reopening of moon prompts or the singleton author outlier as part of the
  main factual bridge

## Interpretation

The main interpretability discovery in the repo is now more specific than the
original broad block-structure hope.

The truthful story is:

- standard frozen Gemma exposes a recoverable effective depth mixture
- that mixture is strongest and most interpretable in factual recall
- inside factual recall, routing is organized jointly by semantic family and
  prompt frame
- those route modes have distinct attention/MLP balances and are already linked
  to a bounded same-model tool-breakage surface

This is a meaningful result because it is:

- on the primary model
- held-out and prereg-scale on the oracle side
- sharper than grouped source-type summaries alone
- more honest than a forced full-surface raw cluster story

## Limitations

- This is a synthesis over saved sequence-level artifacts, not a new causal
  intervention.
- The prompt-frame-conditioned interpretation is strongly suggested by the
  saved examples, but not yet quantified as a dedicated frame-versus-content
  analysis.
- The bridge remains bounded:
  moons are excluded and donor-arm controls are still mixed.
- Nothing here upgrades the Figure 8 lane or the trained-routing claim
  boundary.

## Next Steps

- Close `resattn-chz`.
- Keep `resattn-914` as the next implementation issue for Phase 6.
- Add one saved-artifact follow-up for factual prompt-frame-conditioned route
  mode audit before launching any new broad oracle rerun.
