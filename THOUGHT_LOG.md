# Thought Log

Running list of open questions, risks, and follow-up ideas for the depth-routing experiment.

## Open Actions

- [ ] Confirm that oracle-alpha beats uniform routing on the preregistered Phase 1 gate before any interpretability claims.
- [ ] Operationalize all Figure 8 predictions before inspecting heatmaps.
- [ ] Compare softmax-constrained routing against unconstrained and top-k regimes using matched initialization.
- [ ] Design the tool-breakage demonstration around a concrete factual recall circuit on Gemma-2-2B.
- [ ] Treat learned router query vectors as the closest `w_l` analog and define their geometry analysis.
- [ ] Test whether hierarchical clustering reveals approximately `8 clusters` consistent with the Block AttnRes story.
- [ ] Check whether refusal or honesty-related SAE features correspond to distinct routing patterns.

## Notes

- The strongest invalidation risk is overclaiming from frozen-model routing optimization.
- The strongest differentiation opportunity is the combination of tool-breakage, Figure 8 validation, and safety routing analysis.
