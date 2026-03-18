ABOUTME: Records the target-format-aware redesign decision for the Gemma matched-family tool-breakage lane.
ABOUTME: Freezes the next follow-up as a one-token prompt-surface test rather than another pooled rerun or metric rescue.

# 20260318 4xn Target-Format-Aware Tool-Breakage Redesign

## Motivation

`resattn-mxf` sharpened the broader `tool_breakage_factual_recall_v3` donor-arm
result into an explicit family-conditioned boundary:

- `subcategory_element_symbol` stays positive against the within-family,
  cross-family, and fixed-alpha controls
- `subcategory_author_fact` goes negative against all three controls
- capitals and moons sit near tie on the within-family mean-KL read
- authors are the only family with a `1.0` multiword-target fraction

The next question is not “rerun the same pooled surface again.” It is whether
the current author-family drag is mostly an answer-format confound or a real
family-level failure of the prompt-specific routing story.

## Decision

If the Gemma tool-breakage lane reopens later, the next follow-up should be a
**one-token matched-family prompt-surface redesign** rather than a metric
redesign on the current `v3` surface.

Concretely:

- keep the same model: `google/gemma-2-2b`
- keep the same tuned-lens baseline and donor-arm counterfactual machinery
- keep tuned mean KL as the primary metric
- keep the relative rank-instability diagnostics as secondary metrics
- do **not** start by changing the evaluation objective or adding a new
  continuation metric
- instead, build a new `tool_breakage_factual_recall_v4` surface in which every
  confirmatory target is a **single next token under the Gemma tokenizer**

This should remain a matched-family surface. The right design is:

- `subcategory_capital_fact`: only one-token capitals
- `subcategory_element_symbol`: unchanged style, already one-token
- `subcategory_author_fact`: switch from full-name completions to
  one-token surname or equivalent one-token author-answer prompts
- `subcategory_moon_fact`: only one-token moon names

## Why This Is The Smallest Honest Redesign

### 1. It isolates the most plausible current confound with one change

The family profile does not support the simpler story that the author-family
reversal is caused by unusually close within-family donor alphas:

- author within-family alpha JS `= 0.3211`
- capital within-family alpha JS `= 0.2458`
- element within-family alpha JS `= 0.2085`

So the cleanest remaining confound is answer format, not alpha similarity. A
single-token surface tests that directly.

### 2. It preserves comparability to the current lane

The existing same-model tool-breakage stack is already working:

- tuned-lens baseline
- routed-versus-original baseline
- within-family / cross-family / fixed-alpha donor controls
- KL-primary metric discipline
- prompt-level checkpointing and resume

Changing the prompt surface while holding the rest fixed is cleaner than
changing metrics and prompts together.

### 3. Metric redesign first would answer the wrong question

The current primary metric is already a distributional KL to the model's final
output distribution, not only a first-token rank heuristic. The author-family
reversal persists on that primary metric. So a metric-first redesign would mix
two uncertainties:

- whether the current surface is confounded by multi-token answers
- whether a new metric is merely more forgiving

The prompt-surface redesign is the better first discriminator.

## Feasibility Probe

A quick Gemma tokenizer check confirms that a one-token matched-family surface
is feasible locally.

Examples that are one token under `google/gemma-2-2b`:

- capitals: `Canberra`, `Cairo`, `Bangkok`, `Nairobi`, `Rome`, `Madrid`
- authors / surnames: `Lee`, `Morrison`, `Shelley`, `Tolstoy`, `Kafka`,
  `Atwood`, `Austen`, `Orwell`
- moons: `Moon`, `Titan`, `Triton`, `Europa`, `Io`, `Rhea`, `Hyperion`,
  `Miranda`, `Ariel`

Examples from the current `v3` surface that are *not* one token and therefore
should not survive unchanged into `v4`:

- capitals: `Brasilia`
- authors: full names such as `Margaret Atwood`, `Gabriel Garcia Marquez`
- moons: `Ganymede`

## What Not To Do

- Do not reopen the pooled `v3` surface with another donor-arm rerun.
- Do not try to “rescue” the current mixed result by changing the metric first.
- Do not drop the author family entirely; that would turn a confound into
  silent selection bias.
- Do not mix prompt-surface redesign with a new tuned-lens objective in the
  same first follow-up.

## Smallest Next Implementation Slice

The next concrete issue should be:

1. add `tool_breakage_factual_recall_v4` to the saved registry
2. require one-token targets under the Gemma tokenizer for every prompt
3. keep the four matched families
4. run the pilot baseline only
5. compare the family-conditioned tuned mean-KL deltas against `v3`

That pilot is enough to falsify the redesign:

- if authors stop being uniquely negative, the `v3` mixed result was partly a
  target-format confound and the lane is worth reopening
- if authors remain negative even on the one-token surface, the family-level
  mixed boundary is probably real and the stronger same-model claim should stay
  frozen

## Conclusion

The next legitimate reopening of the Gemma tool-breakage lane is not another
larger pooled rerun and not a new metric stack. It is a one-token matched-family
prompt surface that keeps the current machinery fixed and tests the answer-format
confound directly.
