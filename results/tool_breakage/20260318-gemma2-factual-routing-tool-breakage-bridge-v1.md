ABOUTME: Bridges the saved Gemma factual-recall routing families to the existing bounded tool-breakage prompts.
ABOUTME: Records whether the strongest primary-model factual routing structure is actually represented in the current tool-breakage surface.

# Motivation

`resattn-dat` exists because the saved `registry_v5` Gemma oracle artifact now
has a real factual-recall raw-source structure result, and the current bounded
Gemma tool-breakage lane is also factual-recall-based. The honest next question
was whether those two lanes are talking about the same semantic families or only
overlapping loosely.

# Methods

- Model: `google/gemma-2-2b`
- Oracle source artifact:
  `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/oracle_eval_run.json`
- Factual cluster source artifact:
  `results/block_structure/20260318-gemma2-factual-recall-cluster-profile-v1.json`
- Tool-breakage source artifacts:
  - `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1/summary.json`
  - `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1/summary.json`
- Prompt metadata source: `prompts/registry_v5.yaml`
- Factual oracle subset:
  - `stratum_factual_recall`
  - `256` confirm prompts
  - best saved raw-source split `k = 12`
- Bridge procedure:
  - rebuild the `12` factual cluster centroids from the saved per-prompt
    `final_alpha` vectors
  - use the saved prompt-level tool-breakage `oracle_alpha` vectors for all
    `16` pilot+confirm factual prompts
  - assign each tool-breakage prompt to the nearest factual cluster centroid by
    Jensen-Shannon distance
  - compare the saved prompt subcategory tag against the dominant subcategory of
    the nearest factual cluster
  - summarize tuned-lens breakage deltas alongside the family assignments

# Results

- The current tool-breakage surface only partially overlaps the strongest
  factual routing families:
  - factual cluster families in the saved oracle artifact:
    `capital`, `element`, `author`, `moon`
  - overlapping tool-breakage families:
    `capital`, `element`, `author`
  - overlap prompts: `6 / 16` (`0.375`)
- The overlap prompts align perfectly with the factual routing families:
  - `6 / 6` overlap prompts assign to the matching factual cluster family
  - matching fraction on overlap prompts `= 1.0`
  - mean JS to nearest factual cluster:
    - overlap prompts `= 0.2273`
    - non-overlap prompts `= 0.3498`
- The confirm split shows a mismatch between clean family alignment and largest
  breakage magnitude:
  - overlap confirm prompts (`capital`, `element`, `author`):
    - mean tuned final-position KL delta `= +2.5920`
    - mean tuned final target-rank delta `= +3.67`
    - mean tuned target-rank-range delta `= +6019.7`
  - non-overlap confirm prompts:
    - mean tuned final-position KL delta `= +4.9363`
    - mean tuned final target-rank delta `= +71.4`
    - mean tuned target-rank-range delta `= +128475.0`
- The current largest confirm breakage outliers are outside the matched routing
  families:
  - anatomy fact:
    - nearest factual family `author`
    - tuned final-position KL delta `= +14.3234`
    - tuned final target-rank delta `= +144`
  - biology-process fact:
    - nearest factual family `element`
    - tuned final-position KL delta `= +4.4488`
    - tuned final target-rank delta `= +201`
  - animal fact:
    - nearest factual families `capital` or `author`
    - tuned final-position KL deltas `= +5.7230` and `+3.0625`

# Interpretation

- The bridge itself is real. The current bounded Gemma tool-breakage lane is
  not disconnected from the strongest primary-model factual routing structure.
- But the bridge is only partial. The prompts that actually overlap the clean
  `capital` / `element` / `author` routing families are the better aligned
  prompts, not the prompts producing the biggest current confirm breakage.
- So the right conclusion is not “the factual-family bridge failed.” The right
  conclusion is that the current tool-breakage surface is not maximizing the new
  primary-model oracle lead.
- This makes the next high-value move clear: expand the factual tool-breakage
  prompt surface around the matched routing families before spending more time
  polishing the old eight-prompt confirm set.

# Limitations

- This is a saved-artifact descriptive bridge, not a new intervention or causal
  control.
- The tool-breakage prompt surface is still tiny (`16` prompts total).
- The current overlap family set does not include `moon` facts, even though that
  family is present in the stronger factual oracle structure.
- The strong same-model tool-breakage claim boundary remains unchanged by this
  artifact.

# Next Steps

- Close `resattn-dat`.
- Use `resattn-qcn` to expand the Gemma factual-recall tool-breakage surface
  around the matched `capital` / `element` / `author` families.
- Keep the current dynamic-control claim boundary frozen unless a real
  methodological defect appears.
