# Thought Log

Research reflections for the depth-routing experiment.

Purpose:

- capture the qualitative feel of the experiment as it unfolds
- preserve hunches, predictions, guesses, and competing hypotheses
- note surprises, tensions, weird facts, and confidence shifts
- leave behind a readable reflective trail for the end-of-project write-up

Important rule:

- entries here are not claim-bearing evidence
- use this file for research reflections, intuitions, and interesting facts
- validated conclusions still belong in `CURRENT_STATE.md`, `DECISIONS.md`, run artifacts, and the prereg-aware result docs
- useful sidecar research findings belong here if they help preserve why a tangential paper, result, or intuition seemed important at the time

Suggested entry format:

```text
## [TIMESTAMP] [short title]
- Stage: [planning / implementation / analysis / synthesis]
- Feel of the Experiment: [1-3 sentences]
- Working Hypotheses:
  - [hypothesis]
- Hunches and Guesses:
  - [guess]
- Predictions:
  - [what I think will happen]
- Surprises and Tensions:
  - [unexpected fact or inconsistency]
- Confidence:
  - [low / medium / high] in [what]
- Interesting facts:
  - [paper fact, implementation detail, or pattern worth remembering]
- Sidecar research:
  - [useful tangent from parallel reading or bounded exploration]
```

## Working Hypotheses

- [ ] Confirm that oracle-alpha beats uniform routing on the preregistered Phase 1 gate before any interpretability claims.
- [ ] Operationalize all Figure 8 predictions before inspecting heatmaps.
- [ ] Compare softmax-constrained routing against unconstrained and top-k regimes using matched initialization.
- [ ] Design the tool-breakage demonstration around a concrete factual recall circuit on Gemma-2-2B.
- [ ] Treat learned router query vectors as the closest `w_l` analog and define their geometry analysis.
- [ ] Test whether hierarchical clustering reveals approximately `8 clusters` consistent with the Block AttnRes story.
- [ ] Check whether refusal or honesty-related SAE features correspond to distinct routing patterns.

## Hunches and Guesses

- The strongest invalidation risk is overclaiming from frozen-model routing optimization.
- The strongest differentiation opportunity is the combination of tool-breakage, Figure 8 validation, and safety routing analysis.
- The first real implementation bugs are more likely to come from reconstruction and normalization mistakes than from the optimizer itself.

## Feel of the Experiment

- The project looks strongest when framed as recovering an effective depth mixture rather than claiming hidden trained routers.
- The paper path likely depends more on methodological discipline than on finding a flashy first positive result.

## Predictions

- The first feasibility tranche will probably show non-uniform routing before it shows especially clean block structure.
- The safety lane will likely bottleneck on feature validation rather than on routing analysis.

## Surprises and Tensions

- The most differentiated part of the thesis is still the tool-breakage lane, but it is also the easiest lane to overstate.
- Figure 8 is a useful prediction surface, but strong trained-routing match claims still need a reproducible proxy.

## Interesting facts

- Figure 8 in `research/Attention_Residuals.pdf` explicitly names diagonal dominance, embedding persistence, layer specialization, learned skip connections, and Block AttnRes preserving the structure with `N = 8`.
- The strongest current reviewer-facing additions are stability, out-of-sample predictiveness, and harmfulness-versus-refusal separation.

## [2026-03-16T20:55:51-0500] Dependency Freeze And First Validation Slice
- Stage: implementation
- Feel of the Experiment: The dependency freeze was cleaner than expected. The work still feels fragile, but the fragility has moved from packaging risk toward model-hooking and cache-extraction risk, which is the right direction.
- Working Hypotheses:
  - The first serious implementation failures are still most likely to be normalization or cache-accounting mistakes rather than optimizer pathologies.
- Hunches and Guesses:
  - Python 3.14 is probably not the immediate source of trouble for this repo, at least on this machine, even though I expected it to be more hostile.
- Predictions:
  - The next real snag will likely be extracting the exact sublayer outputs we need from the development model while preserving a reconstruction identity within tolerance.
- Surprises and Tensions:
  - The install path for the pinned Phase 1 stack was materially smoother than expected, but that does not buy much scientific confidence by itself.
  - A toy reconstruction module is useful as a tripwire, but it is also dangerously easy to overestimate; the real gate is model-backed validation.
- Confidence:
  - medium in the environment freeze being stable enough for Phase 1
  - low-to-medium that the first model-backed reconstruction pass will be clean on the first try
- Interesting facts:
  - `torch.backends.mps.is_available()` is `True` in the pinned `.venv`, so the local backend assumption is now checked rather than merely stated.
  - The current validation tests encode the two easiest mistakes to make: comparing uniform routing to `logits / L` and pretending per-source normalization is exact.

## [2026-03-16T21:18:40-0500] GPT-2 XL Reconstruction Smoke
- Stage: implementation
- Feel of the Experiment: This is the first point where the repo felt like it touched the real experiment rather than just the scaffold. The most interesting part was that the bug signal looked scientific at first but turned out to be numerical bookkeeping.
- Working Hypotheses:
  - Model-backed reconstruction errors are still more likely to come from accumulation order or hook selection than from final normalization itself.
- Hunches and Guesses:
  - The primary Gemma-2 lane will probably surface different problems than GPT-2 XL, likely around architecture quirks rather than the basic residual algebra.
- Predictions:
  - The next real blocker is no longer reconstruction math; it is keeping the prompt split and identifiability controls ahead of any tempting oracle-alpha optimization.
- Surprises and Tensions:
  - A bulk `sum(dim=0)` on the residual stack produced a false-looking final residual mismatch on MPS even while the logits and layer identities were effectively right.
  - Reconstructing in the model's actual forward order made the mismatch disappear entirely, which is a useful reminder that “vectorized” is not automatically the right scientific implementation.
- Confidence:
  - high in the current GPT-2 XL reconstruction path
  - medium that Gemma-2 will be comparably clean on the first pass
- Interesting facts:
  - The `gpt2-xl` smoke on local MPS ended with exact final-residual and logit reconstruction on the prompt `"The capital of France is"`.
  - All `resid_mid` and `resid_post` layer identities were exact once the cache filter and summation order matched the actual forward pass.

## [2026-03-16T22:05:00-0500] Prompt Split Discipline Finally Feels Real
- Stage: implementation
- Feel of the Experiment: This was less glamorous than the reconstruction work, but it is probably more protective against bad science. The split stops being a virtue-signal once the confirm prompts are actually trapped behind code.
- Working Hypotheses:
  - The first meaningful use of the confirm split will be to keep the Phase 1 loss gate honest while the identifiability controls are still being built.
- Hunches and Guesses:
  - The pressure to “just peek” at confirm prompts would have shown up quickly once tool-breakage plots started looking interesting, so this lock is arriving at the right time.
- Predictions:
  - The next bugs are more likely to be around registry growth or model-specific dataset plumbing than around the pilot/confirm guard itself.
- Surprises and Tensions:
  - The first direct script-level check failed because Python executed the script with `scripts/` rather than the repo root on `sys.path`, which is banal but exactly the kind of infrastructure paper-cut that would have weakened trust in the guard.
- Confidence:
  - high in the current split artifact being durable enough for the next Phase 1 slice
  - medium that the first registry version will remain the final prompt mix for every lane
- Interesting facts:
  - The initial registry deliberately covers two near-term needs rather than every future lane: Phase 1 oracle-alpha prompts and factual-recall prompts for tool-breakage.
  - The guard now exists in both importable Python code and a runnable export script, which lowers the chance that someone bypasses it accidentally.

## [2026-03-16T22:40:00-0500] Control Suite Before Runner
- Stage: implementation
- Feel of the Experiment: This is the part of research infrastructure that looks boring until it saves the project. The repo now has a way to say “these are the controls we promised” without pretending that promise has already been executed.
- Working Hypotheses:
  - The next temptation will be to start runner work and quietly drift from the saved control plan unless the loader becomes the default path immediately.
- Hunches and Guesses:
  - MIB was the easiest place to accidentally bluff, so forcing it into `planned` versus `omitted` felt more important than wiring a toy adapter.
- Predictions:
  - The first oracle-alpha runner will probably reveal that the control suite needs one or two additional reporting fields, but not a different backbone.
- Surprises and Tensions:
  - The repo had already accumulated enough methodological language that the missing piece was not another decision memo; it was a config and code path that future execution cannot casually bypass.
- Confidence:
  - high in the saved control-plan format being the right floor
  - medium in the exact stability metrics list being final
- Interesting facts:
  - The MIB control is now explicit but still honest: planned benchmark anchor, no fake execution claim, revisit once a runner exists.
  - The new helper module stays runner-agnostic, which should make it usable for both the eventual development-model path and the primary Gemma path.

## [2026-03-16T23:08:00-0500] First Runner Smoke
- Stage: implementation
- Feel of the Experiment: The project crossed from “careful scaffold” into “actual experiment code” again. The result is encouraging, but the more important fact is that the runner is now forced through the saved prompt and control contracts.
- Working Hypotheses:
  - The biggest remaining implementation risk is no longer whether a basic oracle-alpha optimization can run; it is whether the same path stays stable under restart and prompt perturbation pressure.
- Hunches and Guesses:
  - A surprisingly large loss drop on a tiny pilot slice is more likely to reflect the freedom of the per-sequence final-output objective than any strong scientific story yet.
- Predictions:
  - Scaling from `2` prompts to a real pilot batch will surface the first practical bottlenecks around cache reuse and runtime, not around the basic alpha optimizer.
- Surprises and Tensions:
  - The first `gpt2-xl` slice improved over uniform more strongly than I expected for such a small harness, which increases my suspicion rather than my confidence until restart and perturbation checks exist.
- Confidence:
  - high that the runner wiring is real
  - low-to-medium that the raw magnitude of this first smoke improvement will survive more disciplined scaling
- Interesting facts:
  - The final-output development slice used `98` sources on `gpt2-xl`: embeddings plus `48` attention outputs and `48` MLP outputs.
  - The first two pilot prompts both improved over uniform, but that is still a smoke artifact and not a preregistered claim-bearing gate.

## [2026-03-16T23:52:00-0500] Pilot Stability Suite Feels More Useful Than The Raw Improvement Number
- Stage: implementation
- Feel of the Experiment: The most important outcome was not that the base pilot stayed positive. It was that the first restart artifact turned out to be partly fake, and the repo now has a cleaner path that measures something real.
- Working Hypotheses:
  - Prompt wording and sample composition are exerting more pressure on the aggregate alpha distributions than optimizer restarts do, at least in this final-output pilot slice.
- Hunches and Guesses:
  - If that pattern survives the confirm split, the next scientific bottleneck will be whether the alpha summaries predict anything out of sample rather than whether they merely look stable under reruns.
- Predictions:
  - The held-out predictiveness check is more likely to expose weak structure than the restart suite was.
- Surprises and Tensions:
  - The restart metrics became almost perfectly stable once the initialization bug was fixed, which makes the zero-top1 agreement under paraphrases and resamples stand out more sharply than I expected.
  - One malformed saved paraphrase was enough to make the first perturbation artifact scientifically annoying, which is a good reminder that “small prompt curation details” are part of the method here, not cosmetic text.
- Confidence:
  - medium that the current pilot stability artifact is methodologically honest
  - low-to-medium that the present alpha structure will clear a held-out predictiveness bar on the confirm split
- Interesting facts:
  - The corrected `gpt2-xl` pilot suite on `8` prompts kept mean improvement over uniform at `1.2432` nats.
  - Restart divergence in aggregate `final_alpha` was only `2.87e-07`, while paraphrase and resample divergence rose to `0.0305` and `0.0191`.

## [2026-03-17T00:35:00-0500] The Confirm Split Finally Pushed Back
- Stage: implementation
- Feel of the Experiment: This is the most useful kind of disappointing result. The confirm split did not merely look noisy; it cleanly refused the current story that a simple sequence-level summary of `h_1[t]` was already enough to predict the recovered alpha vectors.
- Working Hypotheses:
  - The current failure is more likely a feature-summary failure than a collapse of the underlying oracle-alpha signal, because the confirm-split oracle improvement stayed strong while the predictor itself went weak.
- Hunches and Guesses:
  - Mean-pooling `h_1[t]` is probably too lossy for a sequence-level alpha target, especially when the pilot set is only `8` prompts and the predicted object has `98` coordinates.
- Predictions:
  - A richer early-state summary or a different sequence aggregation will probably move the held-out result more than changing the ridge grid again.
- Surprises and Tensions:
  - The selected ridge regularization slammed into the strongest tested value (`100.0`), which feels less like a subtle tuning problem and more like the model asking for a simpler or more faithful feature surface.
  - Four confirm prompts still improved under predicted alphas, but the average was negative, which makes this feel like weak partial overlap rather than total nonsense.
- Confidence:
  - high that the current held-out result should be treated as a failure for this feature spec
  - low-to-medium that the next feature-summary pass will recover a clearly positive confirm-split predictor without expanding the prompt surface
- Interesting facts:
  - The confirm-split oracle alpha itself improved mean loss over uniform by `1.3409` nats, so the blocker is predictiveness rather than the confirm oracle run.
  - The current mean-pooled `h_1[t]` ridge predictor reached `R^2 = -0.2456` and mean JS `= 0.2434` on the confirm split.

## [2026-03-17T01:08:00-0500] Slightly Better Still Isn’t Good Enough
- Stage: implementation
- Feel of the Experiment: The richer-feature comparison was worth doing because it moved the result in the right direction without needing any story-telling. But the movement was small enough that it mostly sharpened the next question instead of answering it.
- Working Hypotheses:
  - The real missing ingredient is probably token-awareness or a less lossy sequence summary, not merely “a slightly later layer”.
- Hunches and Guesses:
  - Mean-pooled `h_4[t]` looks like a better internal baseline than mean-pooled `h_1[t]`, but it still feels like a compressed shadow of the object we are asking it to predict.
- Predictions:
  - The next meaningful improvement will come from features that preserve more positional structure or encode prompt shape more directly.
- Surprises and Tensions:
  - `h_4[t]` beat the other tested internal summaries, including the `h_1+h_4` concatenations, which makes me suspect that “better early contextualization” matters more here than raw feature count.
  - Even the best candidate still wanted the strongest ridge penalty, which is an uncomfortable sign that the predictor is mostly trying not to say too much.
- Confidence:
  - medium that mean-pooled `h_4[t]` is the right internal baseline to beat

## [2026-03-17T16:35:00-0500] Geometry Helped The Loss Metric More Than The Alpha Metric
- Stage: analysis
- Feel of the Experiment: This result is more interesting than another clean negative. Changing only the target geometry finally made the held-out routed-loss delta positive on average, but it did it in a way that made the descriptive alpha metrics uglier rather than cleaner.
- Working Hypotheses:
  - The logit target is probably closer to the right geometry than raw-simplex ridge, but it is still asking the model to predict too large and too fragile an object for an `8 / 8` split.
- Hunches and Guesses:
  - The next useful move is to compress the target, not to keep polishing feature summaries against the full `98`-source alpha vector.
- Predictions:
  - A lower-dimensional target will likely trade away some alpha-shape fidelity while making the routed-loss readout more stable and interpretable.
- Surprises and Tensions:
  - The first positive held-out routed-loss delta arrived at the same time that confirm `R^2` and mean JS got worse.
  - That makes the project feel less like “find the best descriptive alpha metric” and more like “decide what predictiveness objective actually matters for the thesis.”
- Confidence:
  - medium that the geometry fix was worth landing
  - low-to-medium that raw logit-coordinate prediction alone can clear the interpretation blocker
- Interesting facts:
  - The constrained target moved predicted mean improvement to `+0.0693` nats over uniform with `5 / 8` confirm prompts positive, while oracle alpha stayed at `+1.3409`.
  - The selected ridge penalty still saturated at `100.0`, which means the shrinkage story did not disappear just because the target geometry got better.

## [2026-03-17T17:15:00-0500] Compression Fixed The Shape Story But Not The Loss Story
- Stage: analysis
- Feel of the Experiment: This is the cleanest tradeoff we have seen so far. Compression made the predictor look saner in the descriptive metrics and stopped the pathological regularization choice, but it also gave back the one concrete win the full logit target had earned, which was positive held-out routed loss.
- Working Hypotheses:
  - The repo is now looking less like “find the right target” and more like “align target selection with the actual objective.”
- Hunches and Guesses:
  - The next improvement probably comes from loss-aware pilot selection or a hybrid criterion, not from another single target swap.
- Predictions:
  - If we keep choosing by `R^2` and JS alone, we will keep finding predictors that look tidier than they behave.
- Surprises and Tensions:
  - The compressed target almost exactly recovered the old raw-simplex descriptive metrics while the full logit target kept the only positive routed-loss delta.
  - The ridge strength dropping from `100.0` to `0.0001` is a strong sign that compression changed the geometry/conditioning story materially, not just a tiny bit.
- Confidence:
  - medium that target dimensionality matters
  - medium that selection-objective mismatch is now the dominant blocker in the oracle-alpha lane
- Interesting facts:
  - The compressed depth-type-band logit target reached confirm `R^2 = -0.2241` and mean JS `= 0.2380`, almost matching the old raw-simplex token-aware baseline.
  - That same target dropped predicted mean improvement back to `-0.0031` nats and only `3 / 8` confirm prompts improved over uniform.
  - low that further pooled-state variants alone will solve the confirm-split predictiveness problem
- Interesting facts:
  - The best tested internal feature summary improved predicted mean loss from `-0.0348` to `-0.0010` nats versus uniform.
  - The confirm-split `R^2` stayed negative (`-0.2314`) even after that improvement.

## [2026-03-16T22:59:08-0500] Token Awareness Helped The Shape More Than The Outcome
- Stage: implementation
- Feel of the Experiment: This result is more interesting than the raw delta suggests. Preserving coarse position structure in `h_4[t]` did seem to help the predictor find alpha vectors that look a bit more like the oracle ones, but that shape improvement still refused to turn into a real routed-loss gain.
- Working Hypotheses:
  - The predictor may now be close to the right internal-state family but still missing prompt-level context or another summary that matters for which alpha differences are loss-relevant rather than merely distributionally similar.
- Hunches and Guesses:
  - The next useful surface is probably a hybrid: some coarse prompt-shape signal plus the best internal-state summary, not another pure `h_4[t]` pooling trick.
- Predictions:
  - Prompt-level or hybrid features will either finally flip predicted loss slightly positive or make it obvious that the current sequence-level target is too coarse for this prompt scale.
- Surprises and Tensions:
  - `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat` beat plain mean-pooling by pilot JS and confirm JS, which suggests that positional structure matters.
  - The predicted mean improvement got microscopically worse (`-0.0011` vs `-0.0010` nats), which is a frustrating hint that matching the alpha distribution better is not the same as preserving the useful part of the routing signal.
- Confidence:
  - medium that token-awareness is part of the answer
  - low that internal-state-only summaries are sufficient by themselves at the current prompt scale
- Interesting facts:
  - The selected token-aware summary improved confirm `R^2` from `-0.2314` to `-0.2154` and mean JS from `0.2409` to `0.2377`.
  - Every tested candidate still preferred the strongest ridge penalty (`100.0`), which continues to look like the model saying “be conservative; the feature surface is still weak.”

## [2026-03-16T23:13:05-0500] The Prompt-Level Detour Mostly Strengthened The Case For A Deeper Review
- Stage: implementation
- Feel of the Experiment: This is the kind of negative result that is actually useful. The prompt-level and hybrid ideas were the obvious next cheap shot, and the repo is better off knowing they do not help than half-believing they might have if only we had tried them.
- Working Hypotheses:
  - The held-out predictiveness bottleneck is now more likely about the target or predictor class than about missing a small auxiliary feature family.
- Hunches and Guesses:
  - If there is still a recoverable signal here, it may need either a different target object than one sequence-level alpha vector or a predictor that respects token structure more directly.
- Predictions:
  - A design review will probably focus on at least one of four things: sequence-level target coarseness, linear ridge bias, tiny prompt count versus feature dimension, or pilot JS as the wrong tuning metric for the loss objective.
- Surprises and Tensions:
  - The hybrids did not even beat the plain token-aware `h_4[t]` baseline on pilot JS, which is stronger evidence against “we just forgot prompt shape” than I expected.
  - The prompt-shape hybrid came very close to the best pilot JS, which is mildly annoying because it says the added features are not nonsense, just not enough to matter.
- Confidence:
  - high that simple prompt-level or hybrid features are not the missing piece
  - medium that the next correct move is to step back and review the predictiveness formulation itself
- Interesting facts:
  - `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_prompt_shape_scalar_features_v1_concat` nearly tied the best pilot JS (`0.24356` vs `0.24350`) but still lost.
  - The selected feature source and all confirm metrics remained exactly unchanged because none of the new candidates beat the existing baseline.

## [2026-03-16T23:26:00-0500] The Review Mostly Changed What Not To Trust
- Stage: implementation
- Feel of the Experiment: The external review did not hand us a clean new predictor. What it really did was make two quiet weaknesses impossible to ignore: the control config was not truly in charge, and the stability evidence was less relevant to the learned object than it looked.
- Working Hypotheses:
  - Fixing metric governance and prompt-matched stability will not solve the predictiveness blocker by themselves, but they are worth landing because they remove two ways of fooling ourselves.
- Hunches and Guesses:
  - The next actual progress will come from changing the target geometry, not from another layer of summary features or another selection heuristic.
- Predictions:
  - Once the predictor is moved into a constrained or compressed target space, the repeated preference for extreme ridge shrinkage should either weaken or become easier to interpret.
- Surprises and Tensions:
  - The strongest review point was not “ridge is weak” but “ridge is learning the wrong object in the wrong coordinates,” which feels more actionable than I expected.
  - The prompt-matched stability fix is satisfying precisely because it is not flashy; it just removes a blind spot.
- Confidence:
  - high that the current no-regret infrastructure fixes are correct
  - medium that the next redesign should start with target parameterization rather than a larger prompt set
- Interesting facts:
  - The saved config had already declared `r_squared` primary and `mean_js_divergence` secondary; the problem was that the runner was ignoring that declaration.
  - Aggregate alpha stability and prompt-matched alpha stability are different objects, and the predictiveness question really needs the latter.
