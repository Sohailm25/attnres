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

## [2026-03-17T09:29:48-0500] Build The Big Run So It Does Not Become Two Big Runs
- Stage: planning
- Feel of the Experiment: The main emotional shift is that the repo finally feels ready to treat the next oracle-alpha scale-up like real local research infrastructure rather than a larger notebook-ish rerun. That feels less flashy, but it is probably what keeps the next week from dissolving into repeated near-duplicate runs.
- Working Hypotheses:
  - The current positive `32 / 16` result is real enough that the next bottleneck is operational discipline more than predictor redesign.
- Hunches and Guesses:
  - The most annoying future reruns would come from forgetting to save the per-prompt oracle outputs, not from forgetting one more metric in the summary table.
- Predictions:
  - Once the larger prompt surface is frozen, the next useful disagreements will be about prompt composition and confirm-split size, not about whether the runner can survive interruption.
- Surprises and Tensions:
  - The old held-out script was scientifically decent but operationally flimsy; that mismatch only really became obvious once the result was finally good enough to deserve scaling.
- Confidence:
  - medium-high that checkpointed prompt-level artifacts are the right no-regret build-out
  - medium that the next saved prompt surface should be materially larger in both pilot and confirm, not just pilot again
- Interesting facts:
  - The current runner already had most of the scientific logic we needed; the missing piece was durable persistence of oracle results, feature vectors, and the full regularization sweep.

## [2026-03-17T10:28:19-0500] Freezing The Bigger Surface Feels Like The Last Pure Setup Step
- Stage: planning
- Feel of the Experiment: `registry_v4` feels like the first prompt freeze that is genuinely large enough to justify a real local campaign rather than another confidence-building rehearsal. The main tension now is operational discipline, not experimental indecision.
- Working Hypotheses:
  - The next failure mode is more likely to be runtime hygiene or checkpoint sloppiness than another sample-size miss on the method itself.
- Hunches and Guesses:
  - If the larger run still comes back positive on held-out routed loss, the next design question will probably shift from “is the signal real?” to “what is the cheapest credible route to the prereg gate and the null comparisons?”
- Predictions:
  - The first useful surprise from the campaign will come from per-prompt heterogeneity, not from the top-line mean alone, because the run now has enough confirm prompts to show shape rather than only direction.
- Surprises and Tensions:
  - Expanding the prompt surface was mechanically easy compared with the campaign build-out, which is a reminder that the expensive part of local research is often not writing the prompt text but making sure the next run only has to happen once.
- Confidence:
  - medium-high that freezing `96 / 128` before launch is the right tradeoff
  - medium that this is large enough to avoid another near-term registry revision
- Interesting facts:
  - The new oracle-alpha ids now span `oa-pilot-001..096` and `oa-confirm-001..128`.
  - The tool-breakage collection staying fixed is useful: the larger prompt freeze isolates the oracle-alpha scale question without quietly changing another lane.

## [2026-03-17T11:05:55-0500] The Big Split Cleared The Oracle Gate And Reopened The Target Question
- Stage: analysis
- Feel of the Experiment: This is the first oracle artifact that feels genuinely difficult to dismiss on the core loss metric. At the same time, it is not emotionally tidy, because the bigger split did not reward the cleaner logit-target story I half-expected it to.
- Working Hypotheses:
  - The development-model oracle optimization is now firmly real enough to move beyond feasibility arguments.
  - The right interpretation target may still be closer to routed-loss recovery than to high-fidelity alpha-shape recovery, at least for this sequence-level predictor family.
- Hunches and Guesses:
  - The raw-simplex target winning again with `lambda=100.0` does not mean the earlier geometry critique was wrong; it may mean that bigger pilot sets favor an aggressively shrunk predictor that preserves only the loss-relevant part of the signal.
- Predictions:
  - The next useful oracle work is not another scale-up. It is pattern analysis on the saved `registry_v4` artifact and a clearer story about what the held-out predictor is supposed to recover.
- Surprises and Tensions:
  - I expected the larger split to either keep the logit target selected or collapse the routed-loss gain. Instead it kept the routed-loss gain and reverted to the raw-simplex target.
  - `R^2` staying negative while predicted routed loss is solidly positive on `128` confirm prompts is the cleanest evidence yet that “alpha-shape similarity” and “loss-relevant recovery” are not the same object here.
- Confidence:
  - high that the development-model oracle gate is truly cleared
  - medium that the current positive predictiveness result is the right object for stronger claims without more interpretive work
- Interesting facts:
  - The confirm oracle run improved over uniform by `+1.2993` nats with `128 / 128` prompts positive.
  - The held-out predictor still improved over uniform by `+0.1162` nats with `95 / 128` prompts positive, despite confirm `R^2 = -0.0884` and mean JS `= 0.2427`.

## [2026-03-17T09:08:00-0500] The Logit Path Survived A Bigger Split
- Stage: implementation
- Feel of the Experiment: This is the first result that feels like a genuine scaling signal instead of a fragile local win. The larger split did not just preserve positivity; it kept the same target family alive on a broader surface.
- Working Hypotheses:
  - `oracle_alpha_logit_vector` is now the best current predictiveness target for scaling.
  - The main remaining risk is not “wrong target family” but “still not enough data to reach prereg-grade confidence.”
- Hunches and Guesses:
  - The next useful failure mode, if it happens, will come from much larger scale rather than from another small methodological mismatch.
- Predictions:
  - If this same path remains positive on the next prereg-sized scale-up, the repo should stop treating held-out predictiveness as the central blocker for Phase 1 scaling.
- Surprises and Tensions:
  - The selected regularization snapped back to `100.0` even while held-out routed loss improved, which suggests shrinkage preference is not by itself a sign that the path is bad.
  - Mean JS worsened slightly while routed-loss recovery improved again, which keeps reinforcing that the routed-loss metric is the thing to trust most here.
- Confidence:
  - high that further scaling on the logit path is the right next move
  - medium that descriptive alpha metrics may remain weak even if the loss story continues to improve
- Interesting facts:
  - The `32 / 16` run selected `oracle_alpha_logit_vector` with pilot mean predicted improvement `+0.1012` nats and held-out confirm improvement `+0.1277` nats.
  - `12 / 16` confirm prompts improved over uniform, while confirm `R^2` improved to `-0.1954`.

## [2026-03-17T08:52:22-0500] Pilot Size Mattered More Than The Last Few Tuning Tweaks
- Stage: implementation
- Feel of the Experiment: This is the first oracle-alpha predictiveness result in a while that actually changes the shape of the story instead of just sharpening a failure. Doubling the pilot surface did not solve everything, but it clearly changed what the runner trusted.
- Working Hypotheses:
  - The earlier raw-simplex selection was at least partly a small-sample artifact of the `8`-prompt pilot surface.
  - `oracle_alpha_logit_vector` still looks like the most promising target family once pilot selection has enough room to stabilize.
- Hunches and Guesses:
  - The main next risk is that `16 / 8` is still too small, so this positive held-out result could remain fragile until the saved prompt surface gets larger again.
- Predictions:
  - If the next prompt-surface expansion preserves positive confirm routed loss on the logit target, pilot size will look like a major part of the bottleneck rather than a minor nuisance.
- Surprises and Tensions:
  - The confirm mean JS barely moved while the selected target and held-out routed loss changed materially, which is another reminder that descriptive alpha metrics and the real objective are only partially aligned.
  - The compressed target did not benefit from the larger pilot split the way I half-expected; it fell back to `lambda=100.0`.
- Confidence:
  - medium-high that pilot size is a real lever for this lane
  - medium that the logit target is now the best current path to scale
- Interesting facts:
  - The enlarged pilot split selected `oracle_alpha_logit_vector` with `lambda=0.01` and pilot mean predicted improvement `+0.1092` nats.
  - The held-out confirm routed-loss metric improved to `+0.0857` nats over uniform with `5 / 8` prompts positive, while confirm `R^2` stayed negative at `-0.2616`.

## [2026-03-17T08:28:00-0500] Loss-Aware Tuning Still Didn’t Rescue Predictiveness
- Stage: implementation
- Feel of the Experiment: This is a useful failure. The metric-governance fix landed cleanly, but it did not change the scientific answer. That makes the blocker feel less like a runner bug and more like a mismatch between the supervised object and the amount of data we are giving it.
- Working Hypotheses:
  - The current `8 / 8` pilot-confirm split is too small to select among sequence-level alpha targets reliably, even when the tuning metric is routed loss rather than alpha similarity.
  - The stronger problem may still be target coarseness: one full alpha vector per sequence may not be the right object to predict from these summaries.
- Hunches and Guesses:
  - The raw-simplex target winning the pilot loss-aware comparison but failing again on confirm makes me suspect pilot selection noise more than subtle regularization nuance.
  - The full-logit target may still be the most interesting path empirically, but the repo now has evidence that the current pilot surface cannot pick it consistently.
- Predictions:
  - A larger saved pilot surface or a token/span-level target will move the held-out story more than another target-selection tweak.
- Surprises and Tensions:
  - I expected loss-aware pilot tuning to at least keep the full-logit target competitive enough to win; instead the raw-simplex path edged it out and reproduced the old negative confirm result almost exactly.
  - That makes the project feel more bottlenecked by experimental scale and target object than by metric governance.
- Confidence:
  - high that `resattn-xaa` should be treated as a real negative result for the “selection metric alone fixes it” hypothesis
  - medium that the next correct move is a deeper predictiveness redesign rather than another bounded tuning pass
- Interesting facts:
  - All three target candidates chose `lambda=100.0` under the new loss-aware pilot rule, so the selection change by itself did not escape the strong-shrinkage regime.
  - The selected confirm result was effectively identical to the earlier raw-simplex token-aware baseline: `R^2 = -0.2154`, mean JS `= 0.2377`, predicted mean improvement `= -0.0011` nats.

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

## [2026-03-17T12:15:00-0500] The First Pattern Signal Is Real But Smaller And Stranger Than A Clean Block Story
- Stage: analysis
- Feel of the Experiment: This is a good place to be skeptical. There is enough structure in the prereg-scale `final_alpha` distributions to beat a matched random control slightly, but the best clustering is mostly a two-prompt outlier carve-out rather than a broad partition of the confirm set.
- Working Hypotheses:
  - The development-model oracle lane really does expose some prompt-dependent routing structure at sequence level.
  - The current raw-source clustering surface is too diffuse and outlier-sensitive to support the `~8`-cluster hypothesis on its own.
- Hunches and Guesses:
  - Grouping sources by a thesis-relevant structure such as depth bands and source type may give a cleaner pattern read than clustering the full `98`-source raw simplex directly.
  - The very frequent `pos_embed` and early-attention top-1 winners suggest part of the current geometry is still dominated by broad scaffold-like routing mass rather than task-specific routing regimes.
- Predictions:
  - If the pattern story is real, grouped-source views or prompt-resampled stability checks should preserve a modest above-random gap without collapsing entirely.
  - If the story is mostly an artifact of a few outliers, that gap will wash out quickly once the clustering view changes.
- Surprises and Tensions:
  - The mean effective source count is still about `49.45`, which is much more diffuse than the “clean router picks a small block” intuition might have suggested.
  - The best silhouette being `0.1428` versus a matched-random `0.1093` is enough to be interesting but not enough to feel safe.
- Confidence:
  - medium that the current artifact is worth keeping as a real Phase 2 entry point
  - high that it does not justify any clean block-structure claim yet
- Interesting facts:
  - `k = 2` gives cluster sizes `126 / 2`; `k = 8` still gives one huge cluster of `114`.
  - `pos_embed` is the top-1 source on `36 / 128` confirm prompts, and attention mass still dominates MLP mass on average (`0.5560` vs `0.4090`).

## [2026-03-17T12:34:00-0500] The Decision Lane Is Less Ambiguous Than It Looked
- Stage: planning
- Feel of the Experiment: This one looked like a fork on paper, but the repo’s evidence is mostly one-sided. The secondary-model tuned-lens path reads like a convenience escape hatch, not like the strongest scientific design.
- Working Hypotheses:
  - Tool-breakage only really means what we want if the tuned-lens-aware comparison is on the same Gemma-2 model as the routed-versus-original claim.
  - Strong Figure 8 language will keep sounding slippery until there is a small trained depth-routing model in the repo, not just a figure-matching story.
- Hunches and Guesses:
  - Training a small custom Gemma-2 tuned lens will be annoying operationally but conceptually clean.
  - The small local AttnRes reproduction is likely the right proxy even if an open depth-mixing alternative might be easier, because it addresses the exact co-adaptation objection instead of only approximating it.
- Predictions:
  - Once these choices are recorded, the next blockers will feel more like concrete implementation work and less like “we still haven’t decided what evidence would count.”
- Surprises and Tensions:
  - The repo had already mostly made this decision in prose; what was missing was the final step of deleting the ambiguous branch language from the prereg.
- Confidence:
  - high that custom Gemma-2 tuned lens is the right primary path
  - medium-to-high that small local AttnRes reproduction is the right Figure 8 default proxy

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

## [2026-03-17T12:56:00-0500] The Gemma Tuned-Lens Path Is Real, But Its Strength Is Not The Obvious One
- Stage: analysis
- Feel of the Experiment: The good news is that the annoying part worked. The bad news is that the easy story is not the right one. The custom Gemma tuned lens clearly beats the raw lens, but the cleanest gains are in KL and broad token-distribution recovery, not in crisp final factual-recall answer recovery.
- Working Hypotheses:
  - The same-model tuned-lens baseline is now viable enough that we should stop discussing whether Gemma is the right host and move on to what metric later routed-versus-original claims should center.
  - If we want stronger answer-token-facing tool-breakage claims later, we may need either a more targeted tuned-lens objective or a more explicit metric hierarchy instead of pretending the current pilot already solved that.
- Hunches and Guesses:
  - The later tool-breakage lane will probably be most honest if it treats KL-like distributional differences as the primary tuned-lens baseline and keeps answer-token top-1 as a secondary stress metric.
  - Reusing the saved prompt caches will matter more than it seems now; they turn a potentially annoying rerun tax into a mostly solved operational problem.
- Predictions:
  - A routed-versus-original comparison built on this baseline should already be informative about lens-shape disruptions even if final answer-token swings remain smaller.
- Surprises and Tensions:
  - Mean top-1 over all held-out positions jumped from `0.1863` to `0.5119`, but final-position top-1 only moved from `0.1010` to `0.1250`.
  - The viability pilot feels like a pass and a warning at the same time: good enough to proceed, not good enough to get lazy about what “better tuned lens” means.
- Confidence:
  - high that keeping the tuned-lens-aware comparison on Gemma-2 was the right call
  - medium that later tool-breakage interpretation will need careful metric discipline to avoid overselling the final-position story

## [2026-03-17T13:05:00-0500] KL Should Drive The Baseline, But Not The Rhetoric
- Stage: analysis
- Feel of the Experiment: This one is clearer after looking at the layerwise counts. The lens is doing something real almost everywhere, just not the thing a casual reader would jump to first.
- Working Hypotheses:
  - Held-out KL is the right primary tuned-lens baseline metric for the Gemma factual-recall lane because it tracks the strongest same-model fidelity signal in the saved pilot.
  - Final-position answer-token behavior should constrain later write-up language, not block the whole routed lane from starting.
- Hunches and Guesses:
  - If we force final-position top-1 to be primary now, we will spend time rebuilding the lens objective before even finding out whether the routed-versus-original comparison is interesting.
  - If we ignore final-position metrics, we will talk ourselves into claims the artifact does not support.
- Predictions:
  - The later routed-versus-original tool-breakage plots will probably look most convincing when framed as "routing changes lens faithfulness and stability relative to the original-model baseline" rather than as "routing hides the correct answer token" unless the final-position story gets stronger.
- Surprises and Tensions:
  - `25 / 26` layers improved on held-out KL and `25 / 26` improved on final-position KL, but only `4 / 26` improved on final-position top-1.
  - That spread is too lopsided to treat all these metrics as if they were saying the same thing.
- Confidence:
  - high that KL-primary is the right control decision
  - medium-to-high that answer-token-facing claims should stay gated until a sharper lens objective exists or later routed results make the final-position story much stronger

## [2026-03-17T13:45:00-0500] The Tool-Breakage Runner Worked, But The Legacy Boolean Broke First
- Stage: analysis
- Feel of the Experiment: This is the right kind of frustrating. The implementation did what it needed to do, and the pilot is clearly not noise. But the first clean read is also that the old “non-monotonic on >50%” shorthand is too blunt for Gemma factual recall because the original baseline already maxes it out.
- Working Hypotheses:
  - The routed-versus-original Gemma story is real at the KL-primary level, not yet sharp at the legacy boolean level.
  - The next honest metric surface is something explicitly relative, probably target-rank degradation or another routed-versus-original instability summary, not another absolute non-monotonicity count.
- Hunches and Guesses:
  - If we jump straight to confirm with the current boolean, we will spend compute to rediscover that the threshold is uninformative.
  - The saved prompt-level checkpoints are now the important asset. They let us redesign the summary metric without paying another Gemma rerun tax first.
- Predictions:
  - A better relative metric will keep the broad KL degradation signal and split the prompts more meaningfully than `8 / 8` versus `8 / 8`.
- Surprises and Tensions:
  - The tuned-lens degradation under routing is broader than I expected on the pilot: mean KL worsened on all `8 / 8` prompts and final-position KL also worsened on all `8 / 8`.
  - Final-layer target-rank degradation is only `4 / 8`, which is exactly the kind of mixed read that says “use a stronger metric,” not “declare victory.”
- Confidence:
  - high that `resattn-28b` is a real baseline pass
  - high that `resattn-ypj` is the right next blocker before any confirmatory factual-recall run

## [2026-03-17T14:05:00-0500] Rank Spread Is The Cleanest Rescue Path
- Stage: analysis
- Feel of the Experiment: This looks better after forcing the metric comparison onto the saved traces. The probability-shape alternatives were tempting, but they mostly compress the same saturation problem into a noisier number. The rank-based view is the first one that feels both honest and useful.
- Working Hypotheses:
  - The right near-term tool-breakage surface is “does routing broaden or worsen target-rank behavior relative to the original baseline?” not “is the curve technically non-monotonic?”
  - The confirm run is now worth doing because the next metric is at least discriminative on the pilot artifact.
- Hunches and Guesses:
  - The tuned-lens confirm run will probably keep the same pattern as pilot: little to no non-monotonicity increase by the old boolean, but broad increases in rank spread and a mixed best-rank suppression story.
- Predictions:
  - If the confirm split reproduces the `7 / 8` tuned rank-range increase pattern at a similar level, the tool-breakage lane will have a much cleaner story about routed trace instability even before the later dynamic-routing counterfactual.
- Surprises and Tensions:
  - The rank-range metric is much stronger on the tuned lens (`7 / 8`) than the best-rank metric (`4 / 8`), which says routing often makes the trace less stable even when it does not always suppress the best answer visibility outright.
  - That split is actually useful: it separates instability from outright suppression instead of forcing them into one overloaded number.
- Confidence:
  - medium-to-high that rank spread should now drive the confirmatory tool-breakage read
  - medium that best-rank worsening will remain an important but weaker companion metric rather than the sole headline

## [2026-03-17T13:52:00-0500] The Confirm Split Kept The New Story And Killed The Old One
- Stage: analysis
- Feel of the Experiment: This is a good confirm result because it is not trying to flatter us. The old boolean stayed dead, which is exactly what should have happened if the pilot diagnosis was real. The rank-based story, though, survived on unseen prompts and actually looks cleaner now.
- Working Hypotheses:
  - The Gemma tool-breakage lane should now be framed around routed-versus-original instability and rank degradation, not newly induced non-monotonicity.
  - The next scientifically necessary step is the controlled dynamic-routing counterfactual, not another baseline rerun.
- Hunches and Guesses:
  - The tuned rank-range metric may end up being the best eventual headline because it captures instability even when the best token rank is not always suppressed.
- Predictions:
  - If the controlled dynamic-routing counterfactual materially weakens these confirm metrics, the final claim will look much stronger and cleaner than anything built on the old boolean ever would have.
- Surprises and Tensions:
  - The confirm split is even harsher on the old non-monotonicity story than the pilot: routed-versus-original increase stayed exactly `0 / 8` again.
  - At the same time, tuned rank-range increase strengthened to `7 / 8`, which is a much more stable-looking signal than I expected from only eight confirm prompts.
- Confidence:
  - high that the confirmatory baseline stage is now complete
  - medium-to-high that the counterfactual is the real next test, not more metric churn

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

## [2026-03-17T14:19:00-0500] The Gemma Counterfactual Narrowed The Claim More Than The Story
- Stage: implementation
- Feel of the Experiment: This is exactly the kind of result that is annoying operationally and valuable scientifically. The counterfactual did not kill the lane, but it did kill the easy version of the claim.
- Working Hypotheses:
  - The same-model Gemma tool-breakage result is real as a routed-versus-original degradation and as a stronger-than-static-control effect.
  - What failed is the more specific story that prompt-matched input-dependent routing is uniquely responsible for the damage, at least under the current dynamic control.
- Hunches and Guesses:
  - The prompt-permuted control may be so damaging because dynamic but misaligned mixtures are enough to scramble the same intermediate lens traces, which pushes the interpretation toward "non-uniform routed mixtures are brittle" rather than "the learned prompt-specific route is the uniquely bad actor."
- Predictions:
  - The next highest-value lane should probably shift to the Figure 8 proxy rather than immediately trying to rescue the Gemma claim with another control.
- Surprises and Tensions:
  - I expected the prompt-permuted control to stay harmful, but I did not expect it to beat the prompt-matched route on the tuned mean KL metric.
  - The fixed pilot-mean alpha control being weaker is still useful; it says the control did not fail across the board.
- Confidence:
  - high that the strong same-model Gemma claim is still blocked
  - medium that a finer dynamic control would change the qualitative conclusion enough to justify immediate follow-up
- Interesting facts:
  - `pilot_mean_alpha` stayed clearly weaker than the prompt-matched route on tuned KL (`routed minus control = +0.6091`), while `prompt_permuted_alpha` was more damaging (`routed minus control = -0.3082`).
  - The tuned routed-versus-prompt-permuted rank metrics were also weak: final-rank worsening `4 / 8`, best-rank worsening `3 / 8`, and rank-range increase only `2 / 8`.

## [2026-03-17T15:05:00-0500] The Local AttnRes Proxy Is Real, But It Is Not Yet The Figure
- Stage: implementation
- Feel of the Experiment: This lane stopped feeling hypothetical. The tiny `8`-block proxy actually trains, checkpoints, resumes, and even beats the matched baseline once scaled a bit. But the pattern surface is stubbornly not the paper yet, which is exactly the kind of tension I want recorded.
- Working Hypotheses:
  - `resattn-7hb` succeeded as a build-and-viability issue.
  - The next blocker is no longer "can we train a local proxy at all?" but "is this char-level Wikitext regime capable of expressing the Figure 8 signatures we care about?"
- Hunches and Guesses:
  - The loss advantage surviving the larger run makes me think the proxy architecture is not the wrong bet.
  - The still-inverted entropy ordering makes me suspect the data/tokenization regime is at least part of the problem, not just undertraining.
- Predictions:
  - A more realistic tokenizer or corpus will matter more than another tiny architectural tweak.
  - If we keep the exact current regime and just train longer, we may get a cleaner baseline comparison but still not a convincing Figure 8-style pattern surface.
- Surprises and Tensions:
  - I expected the larger run to improve the paper-facing metrics at least a little once the loss gap opened up more, but deep embedding persistence actually dropped from `0.1824` to `0.1049`.
  - The proxy already prefers fairly late blocks at `final_output` even while the broad qualitative pattern is still mixed.
- Confidence:
  - high that the local reproducible-proxy infrastructure now exists
  - medium that the next right move is a proxy-regime decision rather than another same-regime rerun
- Interesting facts:
  - The scaled run beat the matched baseline by `0.0327` eval-loss points while still showing `mean_pre_attn_entropy < mean_pre_mlp_entropy`.
  - The first larger launch failed for a boring but useful reason: `vocab_size=256` no longer covered the larger Wikitext slice (`269` characters), and widening to `512` fixed it immediately.

## [2026-03-17T15:40:00-0500] Tokenizer Realism Helped The Figure More Than The Model
- Stage: implementation
- Feel of the Experiment: This is a good regime-decision result. It’s not the satisfying pass I wanted, but it cleanly changes what I distrust. The char-level regime now looks like a real proxy mismatch, not just a harmless simplification.
- Working Hypotheses:
  - Compact GPT-2 subword remapping is the correct default proxy regime going forward.
  - The remaining blocker is no longer tokenization realism; it is probably capacity or training horizon within the more realistic regime.
- Hunches and Guesses:
  - If I scale the compact-subword run modestly in capacity, the loss comparison may recover while preserving the modest Figure 8 metric improvements.
  - I do not expect another char-level rerun to teach us anything new now.
- Predictions:
  - The next useful run should keep compact subword fixed and scale one axis only: either `d_model`/`d_ff` or training horizon.
  - If the entropy ordering still refuses to flip after that, the next thing to question is corpus rather than tokenizer.
- Surprises and Tensions:
  - The subword smoke looked encouraging immediately, then the longer run lost the baseline comparison even while the Figure-facing metrics improved.
  - That split is actually informative: "better figure surface, worse training outcome" is exactly what a regime change plus underpowered model can look like.
- Confidence:
  - high that tokenization was one real mismatch in the char-level proxy
  - medium that capacity is the next bottleneck rather than corpus
- Interesting facts:
  - Relative to the char-level scaled run, compact subword improved deep embedding persistence from `0.1049` to `0.1364`.
  - The entropy gap improved from `-0.0590` to `-0.0349`, but it still stayed on the wrong side of zero.

## [2026-03-17T16:40:00-0500] Width Helped The Figure A Little, And The Baseline More
- Stage: implementation
- Feel of the Experiment: This is the kind of mixed result I trust. It did not hand us the clean “just make it bigger” story, but it did sharpen the next move without reopening old questions.
- Working Hypotheses:
  - Compact subword is still the right proxy regime.
  - Width alone is not enough at the fixed `1500`-step horizon; the widened proxy likely needs more optimization budget if it is going to compete with the matched baseline.
- Hunches and Guesses:
  - The deeper embedding persistence gain feels real enough that I do not want to abandon the widened regime.
  - The baseline absorbing more of the added capacity than the routed proxy makes horizon the next cleaner lever than another width bump.
- Predictions:
  - A horizon-only follow-up on the widened compact-subword regime is more informative than another width or tokenization change.
  - If the entropy ordering still refuses to move after the longer-horizon run, the next thing to question is the corpus or proxy objective, not width.
- Surprises and Tensions:
  - I expected width to at least recover the baseline comparison if capacity were the main blocker, but the widened baseline improved more than the widened proxy.
  - Deep embedding persistence improved again even while the entropy gap regressed, which keeps the Figure 8 read frustratingly split.
- Confidence:
  - high that `resattn-111` should close as a real bottleneck-isolation result
  - medium that the next Figure 8 follow-up should be optimization horizon rather than a bigger architectural change
- Interesting facts:
  - The widened run improved deep embedding persistence from `0.1364` to `0.1515`.
  - The widened loss delta worsened to `+0.0598`, because the matched baseline improved from `6.6463` to `6.5899` while the AttnRes proxy improved only to `6.6496`.

## [2026-03-17T17:15:00-0500] More Steps Changed Almost Nothing Important
- Stage: implementation
- Feel of the Experiment: This is a useful dead end. I do not like negative results operationally, but this one is clean enough that it saves us from wasting another evening nudging the same regime.
- Working Hypotheses:
  - The widened compact-subword proxy is not primarily blocked on optimization horizon.
  - The next Figure 8 move needs to change the regime more substantively than “same setup, more steps.”
- Hunches and Guesses:
  - The local proxy may need a different corpus or a more faithful local objective before it starts looking like the paper rather than just a workable tiny routed LM.
  - The fact that deep embedding persistence drifted back down a bit makes me less optimistic that the current Wikitext setup will eventually self-correct into the Figure with enough steps alone.
- Predictions:
  - The next honest Figure 8 issue is a redesign decision, not another rerun.
  - The overall repo priority should probably swing back to `resattn-3f1` until we have a better Figure 8 proxy hypothesis than “train the same tiny regime longer.”
- Surprises and Tensions:
  - I expected at least a small improvement in best routed loss from `1500` to `4500` steps, but it stayed exactly flat.
  - The training losses kept falling hard while the best eval losses stayed frozen, which makes the overfitting / wrong-regime interpretation harder to ignore.
- Confidence:
  - high that `resattn-jci` is a real negative result
  - medium-high that the next Figure 8 task should be a bounded redesign decision rather than more scaling
- Interesting facts:
  - The widened baseline and widened AttnRes proxy kept the exact same best eval losses from `1500` through `4500` total steps.
  - The final eval losses were much worse than the best saved losses for both models, even though final train loss kept falling, which is another sign that longer horizon on this regime is not giving useful generalization.

## [2026-03-17T17:47:00-0500] The Safety Lane Needed An Aligned Model Before It Needed Better Features
- Stage: implementation
- Feel of the Experiment: This felt like the right kind of correction. The most important work was not a clever feature method. It was refusing to build a refusal workflow on a model that did not actually refuse. Once that was fixed, the first aligned-Gemma pass was much cleaner than I expected.
- Working Hypotheses:
  - `google/gemma-2-2b-it` is the correct Gemma spine for the safety lane, while the base `google/gemma-2-2b` remains the right spine for the main frozen-model and tool-breakage lanes.
  - The current safety result is strong enough to clear the workflow blocker and weak enough to keep causal humility intact.
- Hunches and Guesses:
  - The near-zero cosine between the refusal and harmfulness directions is probably the most useful signal from this run. It suggests the separation story is real enough to build on, even though the prompt set is still small and templated.
  - The non-zero cross-direction accuracies are a healthy warning sign, not a failure. They are the repo reminding us not to tell a “single perfectly clean safety vector” story too early.
- Predictions:
  - The next safety result that matters will be a causal intervention check, not another prompt-surface expansion.
  - If the causal mediator check is weak, the right interpretation will be “workflow validated, mediator still uncertain,” not “the workflow was a mistake.”
- Surprises and Tensions:
  - The aligned model looked cleaner than expected on the held-out prompt behavior checks: `36 / 36` matched the intended refusal/non-refusal labels.
  - The harmfulness direction still had cross accuracy `0.8333` on the confirm pairs when evaluated on refusal-vs-benign-adjacent structure, which keeps the separation story honest.
- Confidence:
  - high that `resattn-3f1` can close as a workflow-validation success
  - medium-high that the next safety blocker should be a causal mediator issue rather than more discovery plumbing
- Interesting facts:
  - The base `google/gemma-2-2b` preflight complied with a harmful request directly, which would have made any refusal-feature result invalid by construction.
  - The aligned run localized refusal at assistant-prefill layer `22` and harmfulness at instruction-final layer `25`, which is exactly the sort of positional split the literature made me expect without guaranteeing it.

## [2026-03-17T18:05:00-0500] The Figure 8 Lane Finally Stopped Pretending The Next Lever Was Obvious
- Stage: analysis
- Feel of the Experiment: This is a relief more than anything else. The width and horizon follow-ups were useful, but they left the lane in that annoying “maybe one more tweak” state. The redesign decision is the first moment the repo has a clean reason to stop doing that.
- Working Hypotheses:
  - The strongest remaining bottleneck is corpus size, not another dose of width or another round of training on the same small dataset.
  - The proxy architecture is still worth keeping fixed for one more discriminating run because the metric surface did move in the right direction under compact subword and width, even while the baseline comparison stayed weak.
- Hunches and Guesses:
  - `wikitext-2` is now doing two bad things at once: making overfitting easy and making it too tempting to read every negative result as an architecture failure.
  - If the widened compact-subword proxy still looks bad on `wikitext-103`, then the next honest move really will be objective or architecture redesign rather than “more realistic data.”
- Predictions:
  - The next Figure 8 artifact should be far more informative than the last two because it will answer the overfitting hypothesis directly instead of circling it.
  - If the entropy ordering still refuses to move on the larger same-family corpus, I will stop defending the current local objective as the likely fix.
- Surprises and Tensions:
  - The char-level scaled run still matters here because it showed the local routed proxy can beat the baseline at all. That keeps me from blaming the whole architecture too early.
  - The single cleanest piece of evidence came from the horizon run, not the width run: best eval froze while train loss kept dropping. That is much more “data problem” than “capacity problem.”
- Confidence:
  - high that `resattn-du2` should choose corpus first
  - medium-high that `wikitext-103` is the right bounded next step rather than jumping straight to a totally different pretraining distribution
- Interesting facts:
  - Compact subword improved deep embedding persistence twice in a row, first over char-level and then again with width.
  - The best widened eval losses stayed exactly unchanged from `1500` through `4500` steps, which is about as direct a warning against more same-regime optimization as I could ask for.

## [2026-03-17T18:48:00-0500] `wikitext-103` Helped, But It Mostly Changed The Optimization Question
- Stage: implementation
- Feel of the Experiment: This is the kind of mixed result that changes what I distrust. The larger corpus did help, but not in the neat “problem solved” way I hoped for. The important shift is that the lane now looks less like pure tiny-corpus overfitting and more like a checkpoint-selection / optimization-shape problem inside a still-imperfect proxy regime.
- Working Hypotheses:
  - The widened compact-subword proxy benefits from the larger same-family corpus, but the full `1500`-step training path still drifts away from the early competitive regime.
  - The next Figure 8 follow-up should capture checkpoint-level eval trajectory and best-model state before another training redesign.
- Hunches and Guesses:
  - The `200`-step positive calibration matters more than it first looks. It suggests the widened routed proxy can briefly occupy a healthier regime on `wikitext-103` before the later optimization path gives the baseline back the edge.
  - If the best checkpoint on `wikitext-103` still has a weak Figure 8 surface, then I will stop defending the current objective and schedule as the likely fix.
- Predictions:
  - A best-checkpoint / trajectory follow-up will be more informative than another horizon-only rerun.
  - If the best-checkpoint read stays mixed, the next redesign should move to objective or optimization regularization rather than more corpus arguments.
- Surprises and Tensions:
  - I expected the larger corpus to move the entropy ordering at least a little, but the gap stayed essentially unchanged and negative.
  - The routed gap narrowed from `+0.0598` to `+0.0386`, which is real progress, but it is still not the kind of directional win that justifies staying in the lane without better checkpoint visibility.
- Confidence:
  - high that `resattn-7y4` is a useful mixed result rather than a failed idea
  - medium-high that `resattn-fby` is the right Figure 8-specific follow-up
  - medium that `resattn-73l` is the stronger overall next repo step once `7y4` is landed
- Interesting facts:
  - On the exact widened `wikitext-103` regime, the `200`-step calibration beat the matched baseline by `0.0534` eval-loss points.
  - At the full `1500`-step horizon, deep embedding persistence improved to `0.1615`, but the entropy ordering remained inverted (`1.4318 < 1.4893`).

## [2026-03-17T18:13:35-0500] The Safety Mediator Finally Did Something Causal Without Turning Into A Jailbreak Story
- Stage: implementation
- Feel of the Experiment: This was the right kind of safety result. The first binary readout looked disappointingly flat, but that turned out to be a measurement problem more than an intervention failure. Once I added the matched continuation-preference metric, the refusal direction started behaving like a real mediator instead of just a separability artifact.
- Working Hypotheses:
  - The aligned-Gemma refusal direction at assistant-prefill layer `22` has causal bite on refusal-style execution, not just descriptive separation.
  - The matched harmfulness direction at instruction-final layer `25` does not explain the same effects on this frozen prompt set.
- Hunches and Guesses:
  - The benign-prompt shift is the interesting discomfort in this artifact. It says the refusal direction is not a neatly selective harmful-only switch; it is more like a refusal-style execution axis with some spillover.
  - That actually makes the next safety step cleaner: routing analysis should be mediator-conditioned, but it should not talk like the mediator is perfectly disentangled.
- Predictions:
  - The next safety result that matters is whether routing differences survive or sharpen when conditioned on this mediator.
  - If future routing analysis only tracks the saturated greedy refusal marker and ignores the continuation-preference surface, it will miss real movement again.
- Surprises and Tensions:
  - The first `1 / 1` smoke looked like a null on the binary marker, but the continuation-preference metric immediately separated refusal-direction effects from the harmfulness controls.
  - On the full confirm split, refusal injection only flipped `1 / 6` harmful-context prompts outright, even though the refusal preference margin shifted consistently in the expected direction.
- Confidence:
  - high that `resattn-73l` clears the bounded causal mediator blocker
  - medium-high that the next safety lane should use this mediator directly rather than reopening discovery
- Interesting facts:
  - Refusal suppression on refusal prompts reduced the refusal-versus-context preference margin by `0.1891`, while harmfulness suppression stayed exactly flat.
  - Refusal injection on harmful-context prompts increased the same preference margin by `0.1741` and produced the only greedy refusal flip (`1 / 6`).

## [2026-03-17T18:46:49-0500] Best Checkpoint Helped The Proxy, But Not Enough To Let Me Blame Everything On Late Drift
- Stage: implementation
- Feel of the Experiment: This is the kind of result that narrows the story without making it pleasant. The runner was definitely hiding useful signal by only preserving the final checkpoint, but the moment I gave the proxy its best checkpoint, the deeper problem stayed visible.
- Working Hypotheses:
  - The widened compact-subword `wikitext-103` regime has a real earlier operating region that is better than the final checkpoint suggests.
  - But the main Figure 8 blocker is no longer “we are judging the wrong checkpoint.” It is something about the training setup itself: objective, regularization, or a related optimization design choice.
- Hunches and Guesses:
  - If best-checkpoint export had flipped the entropy ordering or at least erased the baseline loss gap, I would still defend optimization-shape as the next main lever. It did neither.
  - The modest bump in deep embedding persistence feels important because it says the local proxy is not completely missing the paper surface; it is just not sustaining it strongly enough.
- Predictions:
  - The next useful Figure 8 step should be a bounded redesign question, not another same-regime rerun.
  - If the next redesign only improves loss without fixing the entropy ordering, the claim boundary will stay descriptive no matter how operationally clean the proxy becomes.
- Surprises and Tensions:
  - The final-checkpoint loss gap (`+0.1272`) was much harsher than the best-checkpoint gap (`+0.0386`). That is a non-trivial measurement correction.
  - Even after that correction, the best-checkpoint entropy ordering still ran the wrong way. That is the uncomfortable part I do not get to explain away anymore.
- Confidence:
  - high that `resattn-fby` was worth doing before any redesign
  - medium-high that checkpoint drift is now a secondary issue rather than the primary Figure 8 bottleneck
- Interesting facts:
  - The matched baseline peaked at step `900`, while the AttnRes proxy peaked slightly earlier at `850`.
  - Best-checkpoint deep embedding persistence improved from `0.1615` to `0.1689`, but the entropy gap only moved from `-0.0574` to `-0.0549`.

## [2026-03-17T19:00:00-0500] The Pattern Story Got Better Once I Stopped Asking Raw Sources To Do All The Work
- Stage: implementation
- Feel of the Experiment: This one clarified more than it pleased. The raw block-structure story is still weak, but the lane is not empty noise. Once I compress the routing object to grouped source types, a real coarse regime signal shows up and survives resampling.
- Working Hypotheses:
  - Raw-source sequence-level routing on `gpt2-xl` really does contain structure above random, but it is mostly too diffuse and too outlier-driven to support the prereg block hypothesis.
  - The strongest stable pattern is coarse balance between embedding / attention / MLP usage, not a clean many-cluster decomposition of the raw `98`-source object.
- Hunches and Guesses:
  - The `source_type` view is probably closer to the right descriptive scale for this development-model artifact than the raw-source view, but it is also too compressed to carry the stronger thesis by itself.
  - The `depth_thirds_by_type` result is the uncomfortable middle ground: more expressive than `source_type`, still above random, and still mostly collapsing to `k = 2`.
- Predictions:
  - If we revisit the pattern lane later, the interesting next question will be “what do the coarse grouped regimes mean?” not “can we force raw-source clustering into looking cleaner?”
  - Without richer source surfaces or primary-model replication, the block-structure gate will stay closed.
- Surprises and Tensions:
  - The raw-source resampling result is stronger than I expected in one sense: it is consistently weak rather than flaky.
  - The grouped-source views clearly help, but they also lift the random-control silhouette a lot. That is exactly why this cannot be sold as a simple victory.
- Confidence:
  - high that `resattn-ojq` is enough to stop hand-waving about the pattern lane
  - medium-high that the next repo step should move to another lane rather than another immediate clustering follow-up
- Interesting facts:
  - Raw-source best `k` stayed `2` on all `128 / 128` resamples.
  - The `source_type` view produced a much more balanced full-sample `k = 2` split (`87 / 41`) than either the raw-source view (`126 / 2`) or the `depth_thirds_by_type` view (`117 / 11`).

## [2026-03-17T19:26:00-0500] The Safety Mediator Still Collapses To Refusal Labels, But It Finally Moves Whole Trajectories
- Stage: implementation
- Feel of the Experiment: This was the right place to push harder. The first version of `h1p` looked too much like a relabeled refusal partition. After adding intervention-conditioned trajectories, the artifact became scientifically worth keeping even though the prompt split is still too clean.
- Working Hypotheses:
  - The aligned-Gemma refusal mediator changes full-depth refusal-style trajectories, not just the local selected-layer coefficient.
  - The current frozen prompt collection is too role-aligned to expose a finer mediator-active subset inside non-refusal prompts.
- Hunches and Guesses:
  - The important number here is the final-layer shift after intervening at refusal layer `22`, not the selected-layer shift itself. The selected-layer change is partly built into the intervention design; the downstream persistence is the real signal.
  - Safety routing now looks more bottlenecked by prompt-surface diversity than by analysis plumbing.
- Predictions:
  - A broader aligned-Gemma prompt surface with softer refusal structure would be more informative than another rewrite of this same `6 / 6` collection.
  - If a later prompt surface still role-collapses under the mediator threshold, the safety lane may simply be too templated for the routing question we want.
- Surprises and Tensions:
  - The mediator partition stayed exactly refusal versus non-refusal, which is weaker than I wanted.
  - But the refusal-direction interventions still moved final-layer refusal trajectories by about `±360` to `±370`, which is much harder to dismiss as a role-label summary.
- Confidence:
  - medium-high that `resattn-h1p` is worth closing
  - medium that stronger safety-routing claims now depend on prompt-surface redesign rather than more analysis refactoring
- Interesting facts:
  - Refusal suppression on refusal prompts changed the final-layer refusal-direction trajectory by `-372.8501`.
  - Refusal injection on harmful-context prompts changed the same final-layer trajectory by `+369.9403`.

## [2026-03-17T19:45:00-0500] The Figure 8 Proxy Has Earned One Last Faithful Stabilization Attempt Before I Let Objective Drift In
- Stage: planning
- Feel of the Experiment: `fby` changed my mind about what the live problem is. The proxy is not just failing cleanly under the standard objective; it is reaching a healthier region and then failing to hold it. That matters because it makes objective changes look premature rather than bold.
- Working Hypotheses:
  - The widened `wikitext-103` proxy is still mostly a stabilization problem.
  - If matched regularization cannot rescue the best-checkpoint gap or the entropy ordering, then the proxy objective becomes the honest next question.
- Hunches and Guesses:
  - Objective changes right now would feel like proxy-shopping. They might improve the paper-facing metrics, but they would also weaken the story that this is still a believable small AttnRes reproduction.
  - A small matched sweep over dropout and weight decay is the right “last faithful chance” before opening that door.
- Predictions:
  - One of the regularization arms will probably narrow the best-checkpoint loss gap further.
  - I am less confident it will actually flip the entropy ordering. That is the stricter test.
- Surprises and Tensions:
  - The best-checkpoint correction was big enough that I no longer believe “just change the objective” is the next disciplined move.
  - At the same time, the entropy gap still being `-0.0549` means the current proxy cannot be defended as secretly paper-like.
- Confidence:
  - high that `resattn-8xu` should close in favor of a regularization-first follow-up
  - medium that `resattn-bux` will improve stability enough to preserve the standard objective
- Interesting facts:
  - The same run that looked strongly negative at the final checkpoint was only `+0.0386` behind the baseline at its best checkpoint.
  - Deep embedding persistence has improved monotonically across several redesigns even while the entropy ordering stayed stubbornly inverted.

## [2026-03-17T19:48:30-0500] The Regularization Sweep Was The Right Last Check, And It Still Said No
- Stage: implementation
- Feel of the Experiment: This is the kind of negative result I trust. The three arms were close enough to the control that they rule out the comforting story without creating a lot of interpretive noise.
- Working Hypotheses:
  - The widened `wikitext-103` proxy is no longer mainly blocked on simple matched regularization.
  - The next honest Figure 8 move has to change the objective rather than keep massaging the same training surface.
- Hunches and Guesses:
  - The dropout arms improving deep embedding persistence without touching the loss gap is exactly the awkward pattern that makes this feel like a proxy-objective mismatch rather than pure overfitting.
  - If an objective-level redesign also fails, the repo should start taking the “small local AttnRes proxy is qualitatively different from the paper regime” possibility much more seriously.
- Predictions:
  - `resattn-9fo` should end up choosing one minimal objective change, not a broad search.
  - I do not expect another small regularization tweak to change the lane qualitatively.
- Surprises and Tensions:
  - I expected the combined dropout-plus-weight-decay arm to either clearly help or clearly hurt. Instead it barely moved the loss gap at all.
  - The best entropy gap improvement (`-0.0533`) is technically better than control, but not remotely enough to matter.
- Confidence:
  - high that `resattn-bux` should close as a useful negative result
  - medium-high that objective redesign is now the right next Figure 8 question
- Interesting facts:
  - The best loss delta across the sweep was only `+0.0368`, still negative.
  - The strongest persistence came from dropout (`0.1855`) rather than stronger weight decay.

## [2026-03-17T19:58:00-0500] The Gemma Control Story Is Done Unless Something New Breaks
- Stage: planning
- Feel of the Experiment: This one is cleaner than it feels emotionally. The lane already told us the hard thing: prompt-matched routing is not uniquely worst under the current counterfactual. Running a fancier control right now would mostly be a way of arguing with an answer I do not like.
- Working Hypotheses:
  - The current Gemma claim boundary is the right one to freeze.
  - A finer control on the same tiny confirm surface would add more ambiguity than truth.
- Hunches and Guesses:
  - If this lane ever reopens, it should be because the confirm surface gets meaningfully larger or because we find a genuine methodological defect in the current permutation control, not because the current result is inconvenient.
- Predictions:
  - `resattn-5d9` should close without a follow-up issue.
  - Future reviewer pressure, if it comes, will probably ask for a larger confirm set before it asks for a more baroque dynamic control.
- Surprises and Tensions:
  - The easiest way to waste time from here would be to convince myself that a more “semantic” donor control is obviously more valid. Right now the repo has not earned that assumption.
- Confidence:
  - high that `resattn-5d9` should close as a no-follow-up freeze decision
  - medium that the Gemma lane should stay deprioritized behind the Figure 8 redesign question
- Interesting facts:
  - The prompt-permuted dynamic control was not just similar to routed; it was more damaging on tuned mean KL by `0.3082`.
  - The routed-versus-prompt-permuted tuned rank-range increase was only `2 / 8`.

## [2026-03-17T20:05:00-0500] The Honest Figure 8 Decision Was To Refuse The Tempting Objective Hack
- Stage: planning
- Feel of the Experiment: This is exactly the kind of decision that feels unsatisfying and is probably right. The local proxy gave us enough rope to invent a clever loss, but not enough justification to trust one.
- Working Hypotheses:
  - A custom objective on the current tiny proxy would weaken the scientific object more than it would strengthen the result.
  - If the Figure 8 strong-claim lane comes back, it should come back through a more faithful proxy, not a more helpful loss.
- Hunches and Guesses:
  - The easiest bad move from here would be to regularize or reward the very patterns we want and then call the resulting proxy “aligned.”
  - The current negative result is more valuable than that kind of win.
- Predictions:
  - `resattn-9fo` should close without a new objective run.
  - `resattn-1lk` should become the real strategic Figure 8 question.
- Surprises and Tensions:
  - The repo now has a fully operational local proxy and still cannot justify the obvious next experiment. That is uncomfortable, but it is also the right signal that the object itself may be the problem.
- Confidence:
  - high that `resattn-9fo` should close as a refusal to add a custom objective
  - medium-high that the next honest Figure 8 move has to change the proxy, not the loss
- Interesting facts:
  - The full faithful sweep path ended with the best regularized loss delta still negative at `+0.0368`.
  - The strongest persistence improvement came from dropout, which still did not make the entropy story credible.

## [2026-03-17T20:18:00-0500] The Biggest Remaining Oracle Risk Finally Got Narrower
- Stage: implementation
- Feel of the Experiment: This was the right place to push after the external review. The interesting part is not that Gemma worked; it is that the repo had quietly allowed "backend readiness" and "primary-model evidence" to blur together, and this smoke cleanly separates them again.
- Working Hypotheses:
  - The next oracle bottleneck is now actual primary-model optimization behavior, not activation plumbing.
  - `resattn-1lk` still matters, but it should no longer sit above the primary Gemma oracle slice in the repo's mental priority stack.
- Hunches and Guesses:
  - If the bounded Gemma oracle slice is also clean, the whole project will feel less like “strong method on dev model, mixed extensions elsewhere” and more like an actual primary-spine paper in progress.
- Predictions:
  - The first Gemma oracle slice will probably surface runtime or caching pressure before it surfaces a deep conceptual blocker.
- Surprises and Tensions:
  - I expected at least one Gemma-specific hook or normalization quirk to show up. Instead the exact same reconstruction smoke path that worked on `gpt2-xl` came back exact on `gemma-2-2b`.
  - That makes the remaining “dev-model-only” risk feel less excusable than it did an hour ago.
- Confidence:
  - high that primary-spine reconstruction readiness is now solved
  - medium that the next bounded Gemma oracle slice will be straightforward
- Interesting facts:
  - Gemma-2 exposed `53` residual sources on this path: embedding plus `26` attention outputs and `26` MLP outputs.
  - The primary-spine smoke was exact on local MPS with `final_residual_max_abs_error = 0.0` and `uniform_logits_max_abs_error = 0.0`.

## [2026-03-17T20:24:00-0500] The Primary Spine Finally Has A Real Oracle Artifact
- Stage: implementation
- Feel of the Experiment: This is the kind of quick win that actually matters. It does not solve the primary-model lane, but it removes the most annoying excuse for not touching it.
- Working Hypotheses:
  - The next Gemma oracle bottleneck is now scale and stability, not whether the runner can execute at all.
  - The development-model result is still the strongest positive one, but it no longer stands alone as the only real oracle artifact in the repo.
- Hunches and Guesses:
  - The first full Gemma pilot suite is more likely to expose runtime and stability texture than to collapse the basic loss-improvement story.
- Predictions:
  - `resattn-a7j` should be a real scientific step, not just more backend hardening.
- Surprises and Tensions:
  - I expected the first Gemma oracle run to need at least one small runner patch. It did not.
  - That makes the remaining dev-model-only risk feel like a scaling question now, not a capability question.
- Confidence:
  - high that the primary-model oracle lane is now genuinely live
  - medium that the saved pilot stability suite will stay clean on Gemma
- Interesting facts:
  - The bounded `2`-prompt Gemma slice improved over uniform by `2.1634` nats.
  - The same tiny slice was already well separated from the prereg nulls.
