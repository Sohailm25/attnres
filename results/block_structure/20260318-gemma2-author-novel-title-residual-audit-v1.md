ABOUTME: Audits the residual split inside the saved author novel-title route mode on Gemma.
ABOUTME: Uses only saved registry_v5 factual-route artifacts to decide whether the residual looks lexical, semantic, or unresolved.

# Gemma-2 Author Novel-Title Residual Audit v1

## Motivation

`resattn-i6i` closed the broad frame audit with a strong result:

- capitals and elements are deterministic frame-group splits
- authors are overwhelmingly frame-conditioned
- the only residual ambiguity is inside the `novel_title` frame

That residual looked like:

- `13 / 16` `novel_title` prompts in `cluster 11`
- `2 / 16` `novel_title` prompts spilling into the direct-author mode
  `cluster 12`
- `1 / 16` `novel_title` prompt as the singleton outlier `cluster 10`

`resattn-0kc` asks the narrow follow-up question: does that leftover split look
more like title-shape / lexical-surface heterogeneity, answer-entity
heterogeneity, or does it remain unresolved?

## Methods

Saved artifacts only:

- `results/block_structure/20260318-gemma2-factual-route-mode-frame-audit-v1.md`
- `results/block_structure/20260318-gemma2-factual-route-modes-v1.json`
- `results/block_structure/20260318-gemma2-factual-recall-cluster-profile-v1.json`

What was checked:

- the exact `novel_title` prompt counts by cluster from the saved frame audit
- the specific spillover titles and singleton outlier
- whether the exceptional prompts share answer entities or instead share unusual
  title surface properties

No new oracle run or clustering pass was performed.

## Results

Saved split:

- main `novel_title` mode:
  - `cluster 11`
  - `13 / 16` prompts
  - example titles include `Invisible Man`, `The Odyssey`, and `Jane Eyre`
- spillovers into the direct-author mode:
  - `cluster 12`
  - `2 / 16` prompts
  - `Moby-Dick`
  - `Frankenstein`
- singleton outlier:
  - `cluster 10`
  - `1 / 16` prompt
  - `Things Fall Apart`

The exceptional titles do **not** support an answer-entity explanation:

- `Moby-Dick` maps to Herman Melville
- `Frankenstein` maps to Mary Shelley
- `Things Fall Apart` maps to Chinua Achebe

So the residual is not clustering a shared author or answer type.

What the exceptions do share is unusual title surface relative to the main
`cluster 11` novel-title pool:

- `Moby-Dick` is hyphenated punctuation-heavy surface text
- `Frankenstein` is an eponymous single-token title that also looks like a
  surname/person name
- `Things Fall Apart` is the longest clause-like multiword title in the
  exceptional set and becomes the singleton outlier rather than joining either
  main author mode

That makes lexical/title-shape heterogeneity the best current explanation of
the `13 / 2 / 1` residual split.

## Interpretation

This is a bounded cleanup result, not a new positive route-mode discovery.

The truthful read is:

- the broader factual conclusion does not change
- authors are still best described as family-plus-prompt-frame-conditioned
- the residual within-frame `novel_title` split looks more lexical than semantic
- the evidence for that lexical read is still small-`n`, so it should stay
  explicitly bounded

Most importantly, the residual does **not** justify reopening the general
family-versus-frame question, and it does **not** supply evidence for a new
content-only author mode.

## Limitations

- This audit relies on saved prompts and saved cluster assignments rather than a
  dedicated machine-readable per-prompt frame table.
- The lexical-surface interpretation is based on only three exceptional titles.
- No causal intervention or alternate prompt rewrite was run here.

## Next Steps

- Close `resattn-0kc` as a bounded saved-artifact clarification.
- Keep the main next implementation step on `resattn-but`.
