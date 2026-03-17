# ABOUTME: Records the bounded Figure 8 proxy redesign decision after the negative width and horizon follow-ups.
# ABOUTME: Chooses the next lever explicitly and names the levers that are intentionally deferred.

## Question

After the mixed or negative Figure 8 proxy results on:

- char-level Wikitext scale-up
- compact-subword Wikitext
- widened compact-subword width-only follow-up
- widened compact-subword horizon-only follow-up

what is the single next redesign lever that should change before we spend more
compute?

## Evidence

### What already improved

- Moving from character-level to compact remapped GPT-2 subword tokenization
  improved the paper-facing metrics:
  - deep embedding persistence increased from `0.1049` to `0.1364`
  - the entropy gap improved from `-0.0590` to `-0.0349`
- Widening the compact-subword regime improved deep embedding persistence again
  from `0.1364` to `0.1515`.

These results say the proxy architecture is not completely broken and that the
tokenization correction was real.

### What failed cleanly

- Width alone did not restore competitiveness with the matched baseline:
  - widened baseline best eval loss `= 6.5899`
  - widened AttnRes best eval loss `= 6.6496`
- Extending the same widened run from `1500` to `4500` total steps did not change
  the best eval losses at all.
- The horizon artifact also showed the clearest operational symptom:
  training loss kept falling while best eval loss stayed frozen.

That is stronger evidence for data/regime overfitting than for a simple
optimization-budget shortage.

### What this rules out as the immediate next lever

- **More horizon on the same widened regime** is ruled out directly by
  `resattn-jci`.
- **More width on the same compact-subword regime** is not the best next lever,
  because the baseline absorbed the added capacity better than the routed proxy.
- **Reopening tokenization** is not justified, because compact subword was the
  last change that moved the Figure-facing metrics in the right direction.

## Decision

The next bounded redesign lever is **corpus size**, not objective and not local
architecture.

Specifically:

- keep compact remapped GPT-2 subword tokenization
- keep the widened `d_model=160`, `d_ff=640`, `8`-block local proxy
- keep the standard next-token language-model objective
- keep the current optimizer and seed discipline
- change the corpus from `wikitext/wikitext-2-raw-v1` to the larger same-family
  `wikitext/wikitext-103-raw-v1`

This is the smallest redesign that directly tests the strongest remaining
hypothesis: the current regime is overfitting a tiny corpus before the routed
proxy can show stable Figure 8 structure.

## Why this lever wins

1. It is the cleanest reading of the saved artifacts.
   The unchanged best eval loss from `1500` to `4500` steps is exactly what a
   too-small corpus looks like.

2. It preserves comparability.
   Changing only corpus size keeps tokenization, objective, and local architecture
   fixed, so the next result will actually answer a causal question rather than
   creating another confounded regime change.

3. It is operationally cheap relative to the alternatives.
   The existing runner already supports dataset swaps. No new model code,
   objective code, or routing export code is needed.

## Explicitly Deferred

### Deferred: proxy objective changes

Do **not** change the proxy objective next.

Reason:
- the current target is still to reproduce a trained-routing Figure 8 pattern
  under a standard autoregressive LM setup
- changing the objective before testing the corpus-size hypothesis would make the
  next result harder to interpret

### Deferred: local architecture changes

Do **not** change the local Block AttnRes architecture next.

Reason:
- the proxy already trains stably
- the char-level scaled run showed the routed model can beat the matched baseline
- the compact-subword and width artifacts changed the Figure-facing metrics in the
  right direction even though the baseline comparison stayed weak

That is not what an obviously wrong architecture looks like.

### Deferred: another sequence-length or horizon change

Do **not** bundle sequence-length or horizon changes into the next run.

Reason:
- horizon alone already failed on the widened regime
- changing corpus and context length together would make the next artifact harder
  to interpret

## Next Step

Create a follow-up issue for a **corpus-first Figure 8 run** on
`wikitext-103-raw-v1` with the widened compact-subword setup held fixed.

If that larger same-family corpus still fails to restore competitiveness or move
the entropy ordering in the right direction, then objective or architecture
redesign becomes the live question.
