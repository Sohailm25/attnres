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

## [2026-03-18T07:05:00-0500] The Next Gemma Oracle Win Should Come From Breadth, Not Cleverness
- Stage: planning
- Feel of the Experiment: The repo finally feels disciplined enough to cash in on its strongest positive lane instead of orbiting it. The main temptation now is overdesign: inventing a fancier oracle method when the obvious underexplored lever is simply a broader primary-model prompt surface.
- Working Hypotheses:
  - The next useful Gemma oracle result will come from stratification across prompt families rather than another target or feature redesign.
- Hunches and Guesses:
  - If the v5 campaign changes the story materially, it will do so by exposing stratum-conditioned routing differences, not by overturning the already-positive held-out loss result.
- Predictions:
  - A larger stratified surface will keep the softmax competition story alive and likely sharpen coarse source-type structure before it sharpens raw block structure.
- Surprises and Tensions:
  - The cleanest implementation path is not glamorous. It is a deterministic prompt generator, round-robin ordering, and a calibration slice. That feels more like tedious infrastructure than a new experiment, but it is probably the right scientific move.
- Confidence:
  - high that the primary-model oracle lane is the right place to invest next
  - medium-high that prompt-surface expansion is a better next lever than another method tweak

## [2026-03-18T07:28:00-0500] The Generator Bug Was Exactly The Kind Worth Catching Early
- Stage: implementation
- Feel of the Experiment: This was a good calibration because it did not just say "the code runs." It flushed out a real prompt-bank flaw before the repo burned a much larger run on it.
- Working Hypotheses:
  - The first larger `registry_v5` campaign should answer a real scale question now that the surface is repaired.
- Hunches and Guesses:
  - The right efficiency move is to stop sweeping feature and target families for the first v5 pass and simply press on the current primary-model winner.
- Predictions:
  - The first larger `registry_v5` run will stay positive on routed-loss recovery and will probably surface stratum-conditioned variation before it resolves raw-source block structure.
- Surprises and Tensions:
  - The bug was not subtle. Seeing `a lemons` in a saved calibration prompt was enough to know the generator needed one more pass before the scale-up became trustworthy.
- Confidence:
  - high that the repaired calibration now justifies the larger v5 campaign

## [2026-03-18T08:26:00-0500] The Big Stratified Gemma Run Finally Feels Like A Paper Spine, Not Just A Promising Lane
- Stage: analysis
- Feel of the Experiment: This is the first primary-model oracle artifact that feels both broad and hard to dismiss. The main emotional shift is that I no longer feel pressure to prove the lane exists; the more interesting pressure is understanding what structure the lane is actually carrying.
- Working Hypotheses:
  - The underexplored upside is now grouped and stratum-conditioned structure, not another larger oracle rerun.
- Hunches and Guesses:
  - Factual recall is likely to look the most coherent under grouped-source views and may become the cleanest bridge between the primary-model oracle lane and the bounded Gemma tool-breakage story.
  - General text probably carries the messiest alpha geometry because the prompts are more stylistically varied than semantically tight.
- Predictions:
  - A stratum-aware pattern analysis on `registry_v5` will strengthen the coarse grouped-routing story before it strengthens the raw-source block-structure story.
- Surprises and Tensions:
  - The improvement was stronger than I expected on both axes at once: broader confirm coverage and materially better alpha-shape recovery.
  - The selected ridge penalty staying at `100.0` is less worrying now that `R^2` finally turned positive on the larger surface.
- Confidence:
  - high that the next oracle move should be analysis on the saved `registry_v5` artifact rather than another immediate rerun
  - medium-high that factual recall is the most underexploited promising stratum right now
- Interesting facts:
  - The full run improved over uniform by `+1.6299` nats on `1024` confirm prompts, and the held-out predictor still recovered `+0.8292` nats.
  - Exact-command reuse on the completed output directory finished in `319.63` seconds versus `2544.10` seconds for the original launch, which means the operational side of this lane is finally behaving like durable local infrastructure.

## [2026-03-18T09:09:00-0500] The Aggregate Gemma Clustering Story Was Too Blunt
- Stage: analysis
- Feel of the Experiment: This was exactly the kind of follow-up worth doing. The full mixed artifact by itself was headed toward another muddy “grouped yes, raw maybe” summary, but the stratum split made the signal legible.
- Working Hypotheses:
  - The raw-source structure story on the primary model is real, but it lives inside tighter prompt families rather than in the full mixed prompt pool.
- Hunches and Guesses:
  - Factual recall is now the most promising bridge between the oracle lane and the bounded Gemma tool-breakage story.
  - Reasoning/math may support a second raw-source structure story, but it feels less immediately paper-shaping than factual recall.
- Predictions:
  - A factual-recall-focused follow-up will produce a cleaner raw-source cluster interpretation than any new mixed-surface aggregate rerun.
- Surprises and Tensions:
  - Full-sample raw-source clustering still looks basically binary and outlier-driven even though factual recall alone shows a much cleaner `k = 12` structure.
  - The grouped `source_type` and `depth_thirds_by_type` views are almost too strong at the full-sample level, which makes them useful but also easy to overread.
- Confidence:
  - high that the next oracle follow-up should be factual-recall-focused
  - medium that the repo will eventually tell a two-level story: broad grouped structure across many prompt families and cleaner raw-source structure only inside narrower families
- Interesting facts:
  - Factual recall raw-source silhouette reached `0.4709` with balanced cluster sizes relative to the rest of the repo's raw-source analyses.
  - Code/procedural and general-text raw-source views stayed weak, which makes the mixed full-sample raw artifact easier to explain.

## [2026-03-18T09:29:00-0500] The Factual-Recall Raw Clusters Are Actually Real
- Stage: analysis
- Feel of the Experiment: This is the first time the block-structure lane has felt concretely alive on the primary model instead of being a mostly negative gate. The nice part is that the signal is interpretable without needing a heroic story.
- Working Hypotheses:
  - The strongest bridge from the oracle lane into the bounded Gemma tool-breakage lane now runs through factual-recall routing families.
- Hunches and Guesses:
  - The author-fact family may be the richest place to look next because it fragments into the most distinct routing modes (`5` clusters) rather than staying nearly one-mode-per-subcategory.
  - Moon facts are useful as a control because they are structurally clean but do not overlap the existing factual tool-breakage set.
- Predictions:
  - A tool-breakage bridge pass will probably show that capitals, elements, and authors sit inside different factual routing families rather than one universal “factual recall” route.
- Surprises and Tensions:
  - The factual clusters are more semantically pure than I expected. I thought there would be more template contamination inside the best-`k` split.
  - The best result is still not a clean `~8`-cluster story; it is a `k = 12` family-conditioned split.
- Confidence:
  - high that the next oracle-adjacent step should connect these factual routing families to tool-breakage rather than reopen aggregate clustering
  - medium that author facts will end up being the most interesting micro-family
- Interesting facts:
  - Capital, element, and moon facts each split into a small number of pure routing clusters, while author facts fragment more.
  - One moon-fact cluster is MLP-heavier and another is more attention-heavy, which suggests the split is not just lexical or answer-format noise.

## [2026-03-18T09:19:38-0500] The Bridge Worked, But It Exposed The Wrong Tool-Breakage Surface
- Stage: analysis
- Feel of the Experiment: This result is satisfying in a stern way. The bridge did not collapse; it got cleaner. But it also refused to flatter the current tool-breakage prompt set.
- Working Hypotheses:
  - The current bounded Gemma tool-breakage surface is only a partial overlap with the strongest factual routing families, so it is not maximizing the new primary-model oracle lead.
- Hunches and Guesses:
  - A matched-family expansion around `capital`, `element`, and `author` prompts is more likely to produce a cleaner cross-lane story than another pass on the existing mixed factual set.
  - The non-overlap outliers may still matter, but they probably belong to a different factual-instability story than the one the new block-structure result surfaced.
- Predictions:
  - If the next tool-breakage surface is rebuilt around the matched families, it will show much cleaner same-family routing assignments even if the raw degradation effect is smaller than the current mixed confirm set.
- Surprises and Tensions:
  - The overlap prompts matched perfectly (`6 / 6`), which is cleaner than I expected.
  - The biggest confirm breakage is mostly on out-of-family prompts, which means the current confirm result is not aligned with the strongest new oracle structure.
- Confidence:
  - high that the next tool-breakage move should be prompt-surface expansion, not dynamic-control polishing
  - medium that matched-family breakage will end up cleaner but not necessarily larger in raw effect size
- Interesting facts:
  - Overlap prompts are much closer to their nearest factual family (`mean JS = 0.2273`) than non-overlap prompts (`0.3498`).
  - The current worst confirm outlier is anatomy, not one of the structured oracle factual families.

## [2026-03-18T09:36:38-0500] The Better-Aligned Tool-Breakage Surface Still Bites
- Stage: experiment
- Feel of the Experiment: This was the right bet. The aligned surface did not collapse into a gentle no-op once the mixed factual noise came out.
- Working Hypotheses:
  - The next confirm run on the matched-family surface is worth the time because the pilot kept a real same-model degradation signal while being much better aligned to the oracle structure.
- Hunches and Guesses:
  - Moon facts may end up being the sleeper family on the confirm split, not just a completeness add-on.
  - Authors might stay weaker in raw effect size but cleaner in family alignment.
- Predictions:
  - The confirm run should stay positive on tuned KL and probably keep the tuned rank-range metric above `50%`, even if family-level heterogeneity remains.
- Surprises and Tensions:
  - Capitals were stronger on pilot KL than I expected.
  - Element prompts showed positive KL but negative range deltas, which means the matched-family surface is not going to collapse into one monolithic instability signature.
- Confidence:
  - high that the repo should run the matched-family confirm baseline next
  - medium that the confirm result will beat the old mixed surface on alignment clarity more than on raw effect size
- Interesting facts:
  - The rerun durability check was clean: prompt checkpoints stayed at their original timestamps and only `summary.json` moved.
  - Authors are the weakest matched family on this pilot even though they were among the cleanest in the oracle-family bridge.

## [2026-03-18T09:45:06-0500] The Aligned Confirm Surface Is Good Enough To Replace V1
- Stage: confirm
- Feel of the Experiment: This is the first time the tool-breakage lane feels like it has a baseline surface that actually matches the main oracle story instead of merely coexisting with it.
- Working Hypotheses:
  - The dynamic counterfactual on `v2` is now the right bottleneck. Baseline reshuffling is done.
- Hunches and Guesses:
  - Capitals and moons may end up being the most persuasive families in the counterfactual step.
  - Elements will stay mechanistically odd because their KL degradation and rank-range story point in different directions.
- Predictions:
  - The `v2` dynamic counterfactual should show routed traces beating the fixed-alpha control more clearly than the old mixed surface did.
- Surprises and Tensions:
  - The aligned surface did not just survive confirm; it beat `v1` on the KL-primary metrics while using twice as many prompts.
  - Author facts recovered a lot from the pilot weakness and are no longer the obvious soft spot.
- Confidence:
  - high that `v2` should replace `v1` as the main bounded baseline
  - medium that the dynamic counterfactual will now be cleaner than it was on the mixed surface
- Interesting facts:
  - `v2` confirm final-position tuned KL delta (`+3.4477`) is meaningfully above `v1` (`+2.9096`).
  - Both confirm relative rank metrics now clear `11 / 16`, which is a cleaner prompt-count base than the old `8`-prompt surface.

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

## [2026-03-17T20:39:00-0500] Gemma Now Feels Like An Actual Oracle Lane, Not A Placeholder
- Stage: implementation
- Feel of the Experiment: This is the first point where the primary model stops feeling aspirational. The lane now has the same basic structure the dev model had before the harder interpretive questions started.
- Working Hypotheses:
  - The next meaningful discriminator is held-out predictiveness on the primary model, not more pilot-side feasibility work.
- Hunches and Guesses:
  - The primary-model holdout might be noisier or harsher than `gpt2-xl`, but if it is, that will at least be a scientific answer instead of a missing-lane problem.
- Predictions:
  - The first Gemma held-out check is more likely to be mixed than outright negative, because the pilot suite already looks clean and stable.
- Surprises and Tensions:
  - Restart stability was even cleaner than I expected: aggregate top-1 source agreement stayed `1.0`.
  - Paraphrases still move prompt-matched alphas a lot (`JS = 0.1434`), which is good because it means the lane is not merely learning a brittle seed artifact.
- Confidence:
  - high that the primary-model oracle lane is now real enough to deserve held-out scrutiny
  - medium that the first held-out Gemma check will stay encouraging
- Interesting facts:
  - The full saved `8`-prompt Gemma pilot suite kept mean improvement over uniform at `1.7613` nats.
  - The qualitative stability pattern matches the dev model: restarts are tiny, perturbations matter.

## [2026-03-18T00:55:00-0500] The Primary Spine Finally Cleared The Held-Out Oracle Test That Mattered
- Stage: implementation
- Feel of the Experiment: This is the first result that really changes the paper shape. The important part is not just that Gemma stayed positive; it is that the positive held-out story is now strong enough that the development-model-only objection is no longer the center of gravity.
- Working Hypotheses:
  - The next oracle question is now structure on the primary model rather than more existence checks.
  - The slightly negative `R^2` is a warning about target geometry and metric interpretation, not a reason to dismiss the loss result.
- Hunches and Guesses:
  - The hybrid source winning on Gemma probably means lexical or prompt-level information matters more on the primary spine than it did on the earlier dev-model sweeps.
  - If Gemma pattern analysis is also cleaner than `gpt2-xl`, the repo will feel much more like a primary-spine paper than a development-model proof-of-concept with interesting extensions.
- Predictions:
  - `resattn-2sb` is now the right oracle follow-up, not another tiny predictiveness redesign.
  - `resattn-b4q` should turn out to be an implementation win rather than a scientific argument.
- Surprises and Tensions:
  - The run stayed on `lambda = 100.0` and still produced a strong held-out loss result. That is a real reminder that Euclidean alpha-fit metrics and routed-loss recovery are not telling the same story.
  - The bounded check took about an hour because the ridge path is numerically wasteful in the `n << d` regime. That is annoying, but it is also a clean engineering problem rather than a conceptual collapse.
- Confidence:
  - high that the biggest remaining oracle-paper weakness just got much smaller
  - medium-high that primary-model pattern analysis is now the highest-value scientific move
- Interesting facts:
  - The confirm predicted mean improvement over uniform was `+1.0428` nats with `123 / 128` prompts positive.
  - The confirm oracle mean improvement over uniform was `+2.5525` nats with `128 / 128` prompts positive.

## [2026-03-18T01:19:00-0500] Gemma's Pattern Story Got Stronger And Narrower At The Same Time
- Stage: implementation
- Feel of the Experiment: This is a satisfying kind of mixed result. The primary model looks more structured than `gpt2-xl` in the grouped views, but it does not let us cheat the raw block-structure claim boundary.
- Working Hypotheses:
  - The primary model may really have cleaner coarse source-type organization than the development model even if raw source-level blocks remain weak.
  - The next central oracle move should be the regime comparison, because the pattern story is now informative enough that more clustering variants are lower value than testing routing constraints directly.
- Hunches and Guesses:
  - The stronger grouped views probably matter more for the final paper than another attempt to squeeze a raw `k ≈ 8` story out of the same `53`-source simplex.
  - The very low embedding mass on Gemma (`0.0113`) feels notable; this primary spine may be routing mostly within later compute rather than preserving a large embedding anchor.
- Predictions:
  - `resattn-5eo` should now be the next core-lane issue.
  - If the regime comparison is also clean on Gemma, the oracle lane will feel substantially more complete even without a raw block-structure pass.
- Surprises and Tensions:
  - Raw-source clustering did worse than the matched random silhouette. That is a stronger “do not overclaim” signal than I expected after the positive held-out predictiveness result.
  - At the same time, `source_type` was stronger than I expected and far more stable under resampling.
- Confidence:
  - high that the grouped coarse-structure story is real on the primary model
  - high that the prereg raw block-structure gate is still unpassed
- Interesting facts:
  - `source_type` reached oracle best silhouette `0.6731` versus random `0.5569`.
  - `depth_thirds_by_type` beat random on `56 / 64` resamples, while raw-source beat random on only `23 / 64`.

## [2026-03-18T02:08:00-0500] The Regime Comparison Finally Looks Like A Real Competition Story
- Stage: implementation
- Feel of the Experiment: This is the kind of mandatory lane that can either turn into paperwork or genuinely change the thesis. On Gemma it changed the thesis shape.
- Working Hypotheses:
  - The primary-model oracle lane is now strong enough that the next high-value move is reducing claim-boundary ambiguity, not squeezing out another small oracle rerun.
  - `resattn-1lk` is probably the right next step because the Figure 8 lane is now the biggest remaining interpretive ambiguity, not the oracle lane.
- Hunches and Guesses:
  - The large softmax-over-unconstrained gap on every confirm prompt makes the competition story feel more robust than I expected after the earlier “positive but mixed” predictiveness artifacts.
  - The top-k family is useful mostly as a shape-of-failure result: sparse routing can recover a lot, but it gives up too much to match dense competition on this surface.
- Predictions:
  - If `resattn-1lk` stays conservative and freezes the strong Figure 8 lane on the current proxy, the repo will read as more disciplined rather than less ambitious.
  - `resattn-mo5` is more likely than `resattn-b4q` to change the paper qualitatively after that.
- Surprises and Tensions:
  - The first top-k run was optimization-invalid because hard zero-init support search was crippled. That was a good catch, and it would have badly understated the sparse family if left alone.
  - Even after the repair, `k = 26` stayed more than a nat behind softmax. That is stronger separation than I expected.
- Confidence:
  - high that the primary-model competition story is now real
  - medium-high that the next honest move is strategic cleanup of mixed lanes rather than more oracle machinery
- Interesting facts:
  - Softmax beat unconstrained and every top-k setting on all `128` confirm prompts.
  - The mean softmax advantage over unconstrained was `0.4888` nats, and over top-k `k = 26` it was still `1.0861` nats.

## [2026-03-18T02:26:00-0500] Freezing Figure 8 Feels Cleaner Than Pretending Another Tiny Proxy Pass Will Save It
- Stage: implementation
- Feel of the Experiment: This is a satisfying decision because it cuts off a very tempting but low-honesty loop. The current proxy is real enough to describe and not strong enough to sell.
- Working Hypotheses:
  - The repo is stronger if it freezes the strong Figure 8 lane now and shifts effort to live mixed lanes rather than building a speculative new proxy without a concrete faithful path.
  - `resattn-mo5` is now the highest-value experimental extension because it can still move a bounded positive lane rather than just narrowing a known negative one.
- Hunches and Guesses:
  - If we had reopened the Figure 8 lane immediately, it would have turned into another round of proxy plausibility arguments rather than clean evidence.
  - The freeze will read better to a reviewer now that the oracle lane is stronger on the primary model.
- Predictions:
  - The repo will feel more coherent after this because the next issue list will stop implying that Figure 8 is still an active near-term build.
- Surprises and Tensions:
  - It is mildly uncomfortable to freeze a lane after putting so much real work into it, but that discomfort is exactly why the freeze is probably the honest move.
  - The better the Gemma oracle lane gets, the less excuse there is for keeping the Figure 8 lane alive on weak proxy momentum.
- Confidence:
  - high that freezing the strong Figure 8 lane is the right current decision
  - medium-high that `mo5` is now the best experimental next step
- Interesting facts:
  - The repo already had the reopen conditions written down in `resattn-9fo`; `1lk` mostly had to admit that none of them is concrete yet.
  - The current local proxy remains useful descriptively even though it is now frozen for strong-claim purposes.

## [2026-03-18T02:31:00-0500] The Broadened Safety Surface Was Useful Precisely Because It Still Failed
- Stage: implementation
- Feel of the Experiment: This is the good kind of negative result. The surface was broader enough to matter, and it still did not break the role collapse.
- Working Hypotheses:
  - The current aligned-Gemma refusal mediator is closer to an overt-refusal detector than to a broader safety-manifold partition on small prompt sets.
  - Future safety-surface work will need better behavior semantics, not just more clever prompts.
- Hunches and Guesses:
  - The non-refusal prompts that explicitly asked for refusal-style language were enough to stress the old validator, but not enough to move the mediator threshold into a genuinely mixed subset.
  - That makes `resattn-ac2` feel more valuable than another immediate prompt rewrite.
- Predictions:
  - If we make the validator tag-aware later, the broadened-surface result will still read as negative on the core partition question.
- Surprises and Tensions:
  - I expected at least one or two non-refusal prompts to cross the mediator threshold. None did.
  - The intervention-conditioned trajectory story stayed strong even while the partition story stayed collapsed.
- Confidence:
  - high that `mo5` answered its core question cleanly
  - medium-high that the best next step is back on the oracle/infrastructure lane
- Interesting facts:
  - The broadened mediator threshold was `135.4067`, and all `6` active confirm prompts were still outright refusals.
  - Refusal injection still moved the final-layer refusal trajectory by `+284.3705` on harmful-context prompts and `+315.3803` on benign prompts.

## [2026-03-18T03:07:00-0500] The Ridge Bottleneck Was More Mechanical Than I Hoped
- Stage: implementation
- Feel of the Experiment: This one felt gratifyingly boring. The slow path really was just the wrong linear algebra for the shape of the data.
- Working Hypotheses:
  - The next annoyance on the primary-model oracle lane is going to be summary-stage visibility, not the ridge solve itself.
- Hunches and Guesses:
  - Once the summary stage is instrumented, prereg-scale reruns on the primary model should feel operationally normal instead of slightly fragile.
- Predictions:
  - `resattn-9co` will matter more for user trust during long runs than for the final paper.
- Surprises and Tensions:
  - The speedup was larger than I expected even on the full fit, not just on the leave-one-out-shaped fold.
- Confidence:
  - high that `b4q` is a real implementation win rather than a scientific fork
- Interesting facts:
  - The new helper stayed within `2.5e-12` of the legacy primal predictions on the Gemma-shaped synthetic benchmark.
  - The leave-one-out-shaped case sped up by `92.10x`, which is enough to stop thinking about the primal solve as the right default here.

## [2026-03-18T03:31:00-0500] Visibility Was The Last Annoying Part Of The Campaign
- Stage: implementation
- Feel of the Experiment: This was the right kind of cleanup. Nothing scientific changed, but the campaign path finally feels less opaque while it works.
- Working Hypotheses:
  - The next truly annoying oracle-lane bottleneck will now come from a new scientific requirement, not from not knowing whether the runner is alive.
- Hunches and Guesses:
  - A simple progress JSON plus short log lines is probably the sweet spot here; anything fancier would have been premature.
- Predictions:
  - Future long runs will still be slow in places, but they should stop feeling half-broken once the oracle stage is done.
- Surprises and Tensions:
  - The implementation path was cleaner than expected once the callback seam existed in the runner.
- Confidence:
  - high that `9co` is enough for the current campaign UX problem
- Interesting facts:
  - The smoke run produced exactly the artifact surface we needed: manifest, progress, final summary, and two progress log lines during tuning.

## [2026-03-18T04:15:00-0500] The Safety Semantics Fix Worked, But It Did Not Magically Make The Surface Clean
- Stage: implementation
- Feel of the Experiment: This was the right kind of cleanup too. The validator stopped blaming the wrong thing, and the remaining mess is now much more informative.
- Working Hypotheses:
  - The next safety-surface semantics issue is narrower: policy-style compliant responses are not captured well by a first-person refusal marker.
- Hunches and Guesses:
  - That residual problem is worth tracking, but not worth pretending `ac2` failed.
- Predictions:
  - If safety-surface work resumes, `resattn-7km` will improve readability more than it will change the mechanistic conclusion.
- Surprises and Tensions:
  - The confirm rate improved exactly where the old broadened artifact looked most artificially harsh, which is a good sign that the semantics fix hit the intended target.
- Confidence:
  - high that `ac2` closes cleanly
- Interesting facts:
  - The new tag-aware artifact moved confirm non-refusal pass rate to `0.8333` with no change to the layer-localization or direction-separation story.

## [2026-03-18T03:36:00-0500] Policy-Style Semantics Were Worth Adding, But They Exposed A Different Residual Problem
- Stage: implementation
- Feel of the Experiment: This was another good cleanup pass. The behavior classifier is less wrong now, and the leftover failures are more informative than before.
- Working Hypotheses:
  - The remaining policy-note misses are about completion completeness under the current token budget, not about missing another behavior label.
- Hunches and Guesses:
  - The two header-only policy notes should probably stay failures unless a later budget check shows they were truncated rather than intrinsically weak prompts.
- Predictions:
  - If `resattn-9us` runs later with a modestly larger generation budget, the confirm metric might move a bit, but the core mediator-collapse story will still stay negative.
- Surprises and Tensions:
  - The first rerun on the same output directory silently reused stale checkpoints, which was a good reminder that semantics changes need fresh output dirs rather than wishful reruns.
  - The real fixed case was not an institutional policy note at all; it was a direct refusal-like safe answer on a policy-style prompt.
- Confidence:
  - high that `7km` closes honestly as a semantics improvement
  - medium-high that the next safety follow-up, if any, is about prompt completeness rather than classification
- Interesting facts:
  - The clean rerun improved pilot non-refusal pass rate to `0.9167` while keeping confirm at `0.8333`.
  - The two remaining benign policy-note misses were literally skeletal headers with no substantive body.

## [2026-03-18T04:02:00-0500] The Budget Sweep Helped, But It Did Not Let Me Blame Everything On Truncation
- Stage: experiment
- Feel of the Experiment: This was a useful falsification pass. The easy story was “just give the policy notes more room,” and the sweep only half agreed.
- Working Hypotheses:
  - One residual miss was genuinely truncation-sensitive.
  - The remaining confirm miss is now a matcher-definition problem around prohibition-style institutional language.
- Hunches and Guesses:
  - If `1wr` lands, the confirm surface should probably move to `0.9167`, but the mediator-collapse story will still stay negative.
- Predictions:
  - More token budget beyond `96` is unlikely to be the right default next move.
- Surprises and Tensions:
  - The `64`-token rerun already made the notes substantive, yet one still failed because the language was “strictly prohibited” rather than “cannot.”
- Confidence:
  - high that `9us` closes as a mixed but decisive run
  - medium-high that the next safety follow-up should be matcher wording, not more budget
- Interesting facts:
  - `96` tokens moved pilot non-refusal pass rate to `1.0` while confirm stayed `0.8333`.
  - The confirm counterfeiting note became long and clearly institutional, but it still did not match the current policy-style rule.

## [2026-03-18T04:14:00-0500] The Prohibition-Style Fix Was The Last One Worth Doing Here
- Stage: implementation
- Feel of the Experiment: This was the satisfying end of a cleanup ladder. The last confirm benign miss flipped for the right reason, and the lane finally stopped looking semantically underdefined.
- Working Hypotheses:
  - The current broadened safety surface is now good enough for bounded interpretation, and further work on this family should only happen for new scientific reasons, not for validator hygiene.
- Hunches and Guesses:
  - The next high-value move is back on the primary-model oracle lane, where the repo has the strongest positive signal and the biggest unmaximized upside.
- Predictions:
  - If we reopen safety again soon, it should be because we introduce a genuinely broader prompt family, not because the current matcher still feels slightly imperfect.
- Surprises and Tensions:
  - The confirm counterfeiting note really was one regex family away from passing cleanly once the budget sweep made it substantive.
- Confidence:
  - high that `1wr` is the last worthwhile semantics follow-up on this surface
  - medium-high that the next repo step should pivot back to primary-model oracle scale-up planning
- Interesting facts:
  - The final rerun moved confirm non-refusal pass rate to `0.9167`.
  - The only remaining confirm miss is now a genuinely refusal-like harmful-context output, not a policy-note marker failure.
## [2026-03-18T09:56:00-0500] The Aligned Dynamic Counterfactual Got Sharper, Not Cleaner
- Stage: experiment
- Feel of the Experiment: This was the right control to run next. The result is still mixed, but it is much more informative than the old `v1` counterfactual because the aligned surface and family balance removed a lot of easy excuses.
- Working Hypotheses:
  - The fixed-alpha objection is now substantially weaker on the aligned `v2` surface.
  - The remaining dynamic-control ambiguity is probably family-conditioned rather than purely aggregate.
- Hunches and Guesses:
  - The current cyclic prompt-permuted arm is already a mostly within-family control because the confirm ids are grouped by family, so the next honest split is explicit within-family versus cross-family donors rather than another generic rerun.
  - Element prompts look like the strongest candidate family for a true prompt-specific routing story; authors look weakest.
- Predictions:
  - If an explicit within-family donor arm still nearly matches routed on mean tuned KL, the strong same-model claim should stay blocked cleanly.
  - If routed clearly beats the within-family arm while only tying or losing to the cross-family arm, the lane gets much more interesting again.
- Surprises and Tensions:
  - The prompt-permuted control did not simply flatten the story. It nearly tied routed on mean tuned KL but lost on final-position tuned KL, which is a narrower and more uncomfortable result than the old broad negative.
  - The donor mapping turned out to be within-family on `12 / 16` prompts, which makes the mixed result harder to wave away as a coarse across-family mismatch artifact.
- Confidence:
  - high that `qww` should close as mixed
  - medium-high that `o3n` is the right next tool-breakage move if this lane continues
- Interesting facts:
  - Routed minus `pilot_mean_alpha` mean tuned KL is `+1.0917`.
  - Routed minus `prompt_permuted_alpha` mean tuned KL is only `-0.0123`, but routed is still worse on final-position tuned KL by `+0.3533`.
  - Family breakdown is not uniform: elements favor routed-worse-than-permuted, while authors skew the other way on mean tuned KL.
## [2026-03-18T10:06:00-0500] The Donor-Arm Split Helped, But It Did Not Buy A Clean Strong Claim
- Stage: experiment
- Feel of the Experiment: This was worth doing. The old “prompt_permuted” ambiguity is gone, and the result moved in the hopeful direction, but not by enough to feel finished.
- Working Hypotheses:
  - The same-model dynamic-control story is now slightly positive on the tuned primary metric.
  - The main remaining uncertainty is prompt-surface breadth, not control geometry.
- Hunches and Guesses:
  - A larger balanced matched-family surface is the right next stress test.
  - If the larger surface keeps the within-family donor gap positive, the tool-breakage lane becomes much easier to defend.
  - If the larger surface washes it out, the element family was carrying too much of the current story.
- Predictions:
  - The next expansion will probably keep the fixed-alpha gap strong.
  - The real knife-edge question is whether the within-family donor gap stays positive once the current `16`-prompt confirm set stops dominating the aggregate.
- Surprises and Tensions:
  - Routed did beat the explicit within-family donor arm on mean tuned KL, which is better than the `qww` read, but the margin is only `+0.0380`.
  - The cross-family donor arm did not become the main villain; it is almost a tie too (`+0.0089`), which means the lane is not just “cross-family donors are unrealistic.”
- Confidence:
  - high that `o3n` should close as mixed rather than pass
  - medium-high that `0mu` is now the right next move
- Interesting facts:
  - Within-family donor arm: routed is worse on mean tuned KL on `10 / 16` prompts.
  - The within-family aggregate is carried mainly by the element family; capitals are slightly negative on mean KL and authors are nearly flat.
## [2026-03-18T10:15:00-0500] The Larger Matched-Family Pilot Held Up
- Stage: experiment
- Feel of the Experiment: This was the right kind of scale-up. The broader surface did not collapse the effect, which means the donor-arm result is worth stress-testing rather than shelving.
- Working Hypotheses:
  - The aligned same-model breakage signal survives a larger balanced prompt surface.
  - The next real question is whether the `v3` confirm baseline stays this positive before the larger donor-arm rerun.
- Hunches and Guesses:
  - If the `32`-prompt confirm baseline stays near the `v3` pilot magnitude, the larger donor-arm rerun is worth spending the final run on.
  - If the confirm baseline softens sharply, the `v2` donor-arm margin was probably too prompt-surface-specific to chase further.
- Predictions:
  - The fixed-alpha objection should stay clearly weaker on the larger surface.
  - The larger confirm read should be at least as informative as the old `v2` confirm even if the average delta comes down slightly.
- Surprises and Tensions:
  - The mean tuned KL delta barely moved at all when I doubled the pilot surface.
  - The final-position tuned KL delta actually improved slightly, which makes the next confirm step feel justified rather than speculative.
- Confidence:
  - high that `0mu` should close cleanly on the pilot result
  - medium-high that `cky` is the right next follow-up
- Interesting facts:
  - `v3` pilot mean tuned KL delta `= +2.6019` versus `v2` pilot `+2.5849`
  - `v3` pilot final-position tuned KL delta `= +3.0324` versus `v2` pilot `+2.9118`
## [2026-03-18T10:19:30-0500] The Larger Confirm Baseline Also Held Up
- Stage: experiment
- Feel of the Experiment: This was the confirmation I wanted before spending the last run. The broader locked surface did not dilute the story at all.
- Working Hypotheses:
  - The larger donor-arm counterfactual is now worth the final run in this batch.
  - If the donor-arm advantage survives on `v3`, the same-model tool-breakage lane will look materially stronger than it did on `v2`.
- Hunches and Guesses:
  - The tuned primary metric should stay the clearest read on `v3`; the range metric will probably remain supportive but somewhat noisy by family.
- Predictions:
  - The fixed-alpha control should stay clearly weaker than routed on `v3`.
  - The harder question is whether the explicit within-family donor arm still comes out slightly weaker than routed once the confirm surface doubles.
- Surprises and Tensions:
  - The mean tuned KL delta improved slightly on the larger confirm surface instead of shrinking.
  - Final-position tuned KL stayed almost unchanged relative to `v2`, which is a good sign that the pilot was not flattering us.
- Confidence:
  - high that `cky` should close cleanly
  - medium-high that `4g2` is the right last run in this batch
- Interesting facts:
  - `v3` confirm mean tuned KL delta `= +2.9127` versus `v2` confirm `+2.7644`
  - `v3` confirm tuned best-target-rank worsening `= 0.65625`
## [2026-03-18T10:24:30-0500] The Broader Donor-Arm Test Did Not Broaden The Claim
- Stage: experiment
- Feel of the Experiment: This was the right place to stop. The broader control answered the question cleanly enough that another immediate rerun would mostly be goalpost movement.
- Working Hypotheses:
  - The same-model tool-breakage lane has a solid bounded baseline story and a cleared fixed-alpha objection.
  - The stronger prompt-specific donor-arm story is not broad enough yet on the expanded surface.
- Hunches and Guesses:
  - The author family is the real spoiler on the broader within-family aggregate.
  - If this lane reopens, it should reopen through family-conditioned analysis, not a bigger pooled run.
- Predictions:
  - A write-up that emphasizes “bounded same-model breakage plus mixed donor-arm control” will hold up better than trying to force a stronger dynamic-routing story.
- Surprises and Tensions:
  - Routed still won on `18 / 32` prompts against the within-family donor arm by mean tuned KL, but lost on the aggregate because the author-family negatives were larger.
  - Final-position tuned KL stayed positive versus the within-family donor arm even while the mean tuned KL aggregate turned negative.
- Confidence:
  - high that `4g2` should close as mixed
  - high that the same-model tool-breakage claim should now stay bounded at this surface
- Interesting facts:
  - Routed minus `within_family_permuted_alpha` mean tuned KL `= -0.0768`
  - Routed minus `cross_family_permuted_alpha` mean tuned KL `= +0.0560`
  - `subcategory_author_fact` mean routed-minus-within-family KL `= -0.3678`
## [2026-03-18T12:10:30-0500] The Pooled Tool-Breakage Story Was Too Smooth
- Stage: analysis
- Feel of the Experiment: This sharpened the lane more than I expected. The family-conditioned profile did not rescue the broader same-model claim, but it did expose where the real structure is and where the current prompt surface is probably confounded.
- Working Hypotheses:
  - The element family is the cleanest surviving prompt-specific same-model signal on the broader matched-family surface.
  - The author-family reversal is more about answer format and first-token evaluation than about unusually similar within-family routing weights.
- Hunches and Guesses:
  - If this lane ever reopens, the right move is a target-format-aware redesign, not another pooled donor-arm rerun.
  - A single-token or otherwise answer-format-controlled follow-up could materially strengthen the interpretability of the same-model tool-breakage result even if the aggregate effect size shrinks.
- Predictions:
  - A target-format-aware follow-up will look more like the element-family story than the current pooled `v3` mix.
  - If it does not, the prompt-specific same-model claim is probably near its ceiling on this lane.
- Surprises and Tensions:
  - The pooled fixed-alpha gap does not survive uniformly at the family level; authors are slightly negative even versus `pilot_mean_alpha`.
  - Author within-family donor alphas are not especially close, so alpha similarity alone does not explain why that family reverses.
- Confidence:
  - high that `mxf` is worth keeping as a real artifact, not a footnote
  - high that `4xn` is the only sensible preserved follow-up for this lane
- Interesting facts:
  - author within-family alpha JS `= 0.3211`
  - capital within-family alpha JS `= 0.2458`
  - element within-family alpha JS `= 0.2085`
  - author multiword-target fraction `= 1.0`
## [2026-03-18T12:28:30-0500] The Next Tool-Breakage Reopening Should Be Surface-First
- Stage: design
- Feel of the Experiment: This was a relieving fork to settle. The cleanest next discriminator is simpler than I feared: fix the target format and keep everything else fixed.
- Working Hypotheses:
  - A one-token matched-family surface is the smallest honest way to test whether the author-family reversal is mostly answer-format confounding.
  - If authors stay negative even there, the family-conditioned mixed boundary is probably real.
- Hunches and Guesses:
  - A one-token author-surname surface will be much more informative than inventing a new continuation metric on the current mixed prompt set.
  - The element-family signal will probably survive almost unchanged on the one-token surface.
- Predictions:
  - `t0p` will either materially reduce the author-family gap or confirm that the stronger prompt-specific same-model claim is near its ceiling.
  - If it reduces the gap, the lane becomes worth reopening; if it does not, the current freeze will look prescient rather than conservative.
- Surprises and Tensions:
  - Gemma tokenization makes this cleaner than expected: enough one-token capitals, surnames, and moon names exist that we do not need a contrived workaround.
  - That makes a metric-first redesign look even less justified.
- Confidence:
  - high that `4xn` should close on a prompt-surface decision
  - medium-high that `t0p` is the right next experimental slice if this lane resumes
- Interesting facts:
  - `Canberra`, `Cairo`, `Bangkok`, `Rome`, and `Madrid` are single Gemma tokens
  - `Lee`, `Morrison`, `Shelley`, `Tolstoy`, `Kafka`, and `Austen` are single Gemma tokens
  - `Moon`, `Titan`, `Triton`, `Europa`, `Io`, `Rhea`, `Hyperion`, `Miranda`, and `Ariel` are single Gemma tokens
## [2026-03-18T12:56:30-0500] The One-Token Pilot Looks Like a Real Reopening Signal
- Stage: experiment
- Feel of the Experiment: This was the outcome I wanted from the redesign test. The lane did not just stay alive overall; it also stopped making authors look uniquely broken at the baseline stage.
- Working Hypotheses:
  - The answer-format confound was materially suppressing the old matched-family read.
  - The one-token surface is worth a real confirm baseline before any donor-arm rerun.
- Hunches and Guesses:
  - If the locked `v4` confirm stays broadly positive by family, the one-token surface will be the right place to revisit donor-arm controls.
  - If authors stay positive on confirm, the `v3` author-family drag was at least partly format-driven.
- Predictions:
  - `czd` should stay clearly positive overall and keep authors out of the unique-baseline-liability role.
  - The confirm read may still compress somewhat for moons because some of the new prompts are more stylized than the old “largest moon of X” surface.
- Surprises and Tensions:
  - Authors are not the weakest family by a dramatic margin anymore; they are simply the smallest positive family.
  - The overall tuned-KL magnitude stayed close to the old `v3` pilot despite the prompt rewrite.
- Confidence:
  - high that `t0p` should close as a positive reopening signal
  - medium-high that `czd` is now the right next tool-breakage move
- Interesting facts:
  - author pilot mean tuned KL delta under routing `= +2.2556`
  - every family is `4 / 4` positive on mean tuned KL under routing
## [2026-03-18T13:19:30-0500] The One-Token Surface Held Up On Confirm
- Stage: experiment
- Feel of the Experiment: This is the cleanest tool-breakage result in a while. The redesign did not just rescue pilot optics; it stayed positive everywhere on the locked split.
- Working Hypotheses:
  - The one-token redesign genuinely removed an important source of distortion from the old matched-family baseline.
  - The donor-arm rerun on `v4` is now the real reopening test for the stronger same-model claim.
- Hunches and Guesses:
  - The `v4` donor-arm read should be materially healthier than `v3`, especially on authors.
  - If authors still fail on donor-arm controls after this confirm baseline, the remaining blocker is more likely to be true prompt-specific control weakness than answer format.
- Predictions:
  - `apy` should narrow or eliminate the old author-family reversal against the within-family donor arm.
  - The redesigned surface may still leave moons slightly noisier because the prompt style is less uniform than capitals/elements.
- Surprises and Tensions:
  - All four families stayed `8 / 8` positive on mean tuned KL under routing; that is cleaner than I expected on a first confirm pass.
  - Authors are still the smallest positive family, so the redesign helped without making the family completely ordinary.
- Confidence:
  - high that `czd` should close as a confirmatory pass
  - medium-high that `apy` is now the right next move
- Interesting facts:
  - author confirm mean tuned KL delta under routing `= +2.0532`
  - capital confirm mean tuned KL delta under routing `= +2.9547`
  - element confirm mean tuned KL delta under routing `= +2.9847`
  - moon confirm mean tuned KL delta under routing `= +2.8556`
## [2026-03-18T13:41:30-0500] The One-Token Donor-Arm Rerun Helped, But Did Not Fully Reopen The Claim
- Stage: experiment
- Feel of the Experiment: This is a real improvement, not a wash. The one-token redesign fixed enough of the old donor-arm problem that the old `v3` freeze is too pessimistic now. But it also exposed a different family-specific failure instead of making the pooled story clean.
- Working Hypotheses:
  - The answer-format confound was real, because the within-family donor arm flipped from aggregate-negative to slightly positive.
  - The remaining blocker is now mostly about donor pairing and prompt style in the moon family, not just the old author-family continuation issue.
- Hunches and Guesses:
  - A family-conditioned `v4` profile will show that the moon cross-family mapping is the sharpest mismatch worth understanding before any more surface redesign.
  - Capitals and elements are now the cleanest prompt-specific same-model signal on the one-token surface.
- Predictions:
  - `8h7` will probably recommend either remapping the cross-family donor pairing for moons or freezing the claim at a new mixed boundary rather than doing another pooled rerun.
  - If a future moon-specific cleanup works, the one-token surface could still support a noticeably stronger same-model story than `v3`.
- Surprises and Tensions:
  - My first family scan was too strong because it used final-layer prompt deltas instead of the primary mean-over-layers KL metric.
  - Under the correct metric, elements are the only clean all-arm positive family, while capitals and authors are near ties and moons are the actual blocker.
- Confidence:
  - high that `apy` should close as mixed improvement rather than pass/fail
  - high that `8h7` is the next honest follow-up if this lane continues
- Interesting facts:
  - routed minus `within_family_permuted_alpha` mean tuned KL `= +0.0165`
  - routed minus `prompt_permuted_alpha` mean tuned KL `= -0.0139`
  - moon routed minus `cross_family_permuted_alpha` mean tuned KL `= -2.2528`
## [2026-03-18T13:49:30-0500] The One-Token Family Profile Says Freeze The Pooled Claim
- Stage: analysis
- Feel of the Experiment: This is the kind of clarification worth doing. It did not magically rescue the whole lane, but it turned a mushy pooled mixed result into a concrete family-conditioned boundary.
- Working Hypotheses:
  - The one-token redesign genuinely fixed most of the old author-format problem.
  - The moon family is the only part of the one-token surface that still clearly resists the stronger same-model story.
- Hunches and Guesses:
  - The next honest follow-up is a moon-specific prompt-style audit, not another pooled rerun.
  - If that audit fails to find a concrete prompt-style explanation, the family-conditioned boundary is probably the right final tool-breakage wording on this surface.
- Predictions:
  - A moon-family audit will either point to a small rewrite/remap or confirm that moons are just a null family for this claim.
  - Elements will remain the cleanest prompt-specific same-model family on any nearby surface.
- Surprises and Tensions:
  - Authors are not the main blocker anymore; they are slightly negative on some arms but positive on others.
  - Moons are slightly positive within-family yet negative versus prompt-permuted, cross-family, and even pilot-mean controls, which is a much stranger pattern than the old author-format story.
- Confidence:
  - high that `8h7` should close as a freeze-at-boundary decision
  - medium-high that `a1w` is the only follow-up worth preserving immediately
- Interesting facts:
  - element routed minus `cross_family_permuted_alpha` mean tuned KL `= +0.8831`
  - author routed minus `cross_family_permuted_alpha` mean tuned KL `= +0.2805`
  - moon routed minus `pilot_mean_alpha` mean tuned KL `= -1.1565`
## [2026-03-18T12:56:00-0500] The Project Center Of Gravity Has Moved Back To Gemma Oracle
- Stage: synthesis
- Feel of the Experiment: The repo is healthier than the stale task ordering made it look. The strongest finding is no longer a development-model feasibility win or a bounded tool-breakage curiosity. It is a broad primary-model oracle result with meaningful held-out predictiveness and regime separation.
- Working Hypotheses:
  - The most novel truthful claim is now about a primary-model effective depth mixture on Gemma, not about global raw block structure or strong same-model tool-breakage.
  - The right next discovery move is to explain where that Gemma signal lives: grouped structure, stratum-conditioned structure, and especially factual recall.
- Hunches and Guesses:
  - The factual-recall stratum is the cleanest bridge between the main oracle result and the bounded tool-breakage lane.
  - The softmax regime result is underweighted in the current narrative and should move into the core story.
- Predictions:
  - A careful `registry_v5` synthesis pass will tighten the paper story more than another small extension run.
  - If that synthesis holds up, later extension lanes will look like corroborating mechanisms rather than rescue attempts.
- Surprises and Tensions:
  - The docs still had the project mentally centered on tool-breakage even after the strongest positive result moved to the primary Gemma oracle lane.
  - The raw `~8`-cluster dream is weaker than the grouped and stratum-conditioned structure that actually showed up.
- Confidence:
  - high that the main lane should now be `resattn-pjd`
  - high that tool-breakage should stay bounded unless the moon-family audit reveals something unexpectedly clean
- Interesting facts:
  - Gemma `registry_v5` oracle mean improvement `= +1.6299`
  - Gemma `registry_v5` predicted mean improvement `= +0.8292`
  - Gemma `registry_v5` confirm `R^2 = 0.2392`
## [2026-03-18T13:03:00-0500] The Next Oracle Move Should Explain Factual Modes, Not Chase More Breadth
- Stage: synthesis
- Feel of the Experiment: The repo does not need another broad Gemma oracle rerun right now. It needs to understand the best structure result it already has.
- Working Hypotheses:
  - Factual recall is the highest-value raw-source discovery surface on the primary model.
  - The right next gain will come from characterizing its within-family routing modes, not from adding another larger mixed pool.
- Hunches and Guesses:
  - Capitals, elements, and authors are still the cleanest bridge families into the extension lanes, even though factual recall as a whole includes moon facts.
  - Reasoning/math is a real secondary structure lane, but it is not the best next discriminator.
- Predictions:
  - A route-mode characterization pass will tighten the main paper story more than another scaling run.
  - It will also make any later bridge into tool-breakage or safety less arbitrary.
- Surprises and Tensions:
  - The repo already had enough evidence to make this pivot without another experiment; the missing piece was synthesis discipline.
  - The strongest raw-source result peaks at `k = 12`, which is useful and meaningful, but cuts directly against any temptation to force the old `~8` narrative.
- Confidence:
  - high that `2mx` is the right main next issue
  - medium-high that `a1w` should remain sidecar only
- Interesting facts:
  - factual-recall raw-source silhouette `= 0.4709`
  - factual-recall best `k = 12`
  - reasoning/math raw-source silhouette `= 0.2456`
## [2026-03-18T13:05:56-0500] The Bridge Question Changed From Families To Modes
- Stage: analysis
- Feel of the Experiment: This was the right kind of narrowing pass. It did not just restate the factual-family result; it exposed that the families themselves are internally structured enough to change how the next bridge should be designed.
- Working Hypotheses:
  - The next extension should stay on factual recall, not move to reasoning/math.
  - The one-token tool-breakage lane is now under-specified at the mode level even when it looks acceptable at the family level.
- Hunches and Guesses:
  - Elements are the cleanest bridge family because they combine high within-family separation with a positive same-model tool-breakage read.
  - Capitals and authors should stay in the main bridge, while moons should stay bounded until their prompt-style ambiguity is resolved.
- Predictions:
  - A mode-aware audit of the one-token `v4` surface will find missing or under-covered factual modes even when family labels overlap.
  - That audit will probably matter more than a fresh reasoning/math extension right now.
- Surprises and Tensions:
  - The family story was already real, but the element family turned out to be more internally separated than capitals or authors.
  - Moons are structurally clean on the oracle side even though they remain the messiest extension family.
- Confidence:
  - high that `resattn-unp` is the right next bridge issue
  - medium-high that `a1w` should remain sidecar only
- Interesting facts:
  - author mode count `= 5`
  - capital mode count `= 3`
  - element mean within-family centroid JS `= 0.2096`
## [2026-03-18T13:05:56-0500] The One-Token Surface Is Family-Aligned But Mode-Underspecified
- Stage: bridge analysis
- Feel of the Experiment: This was a good reality check. The current `v4` lane is not wrong, but it is much thinner than the family labels made it look.
- Working Hypotheses:
  - The next bridge should stay on capitals, elements, and authors.
  - The current `v4` surface needs a route-mode-aware redesign before another strong interpretive step is worth taking.
- Hunches and Guesses:
  - A redesigned surface that deliberately targets the uncovered capital and element modes will matter more than any additional moon-specific cleanup.
  - Authors may need prompt rewrites that stop collapsing into the dominant capital mode.
- Predictions:
  - `resattn-q38` will produce a more useful surface than just extending `v4` by a few more prompts within the same templates.
  - `a1w` may still improve the moon sidecar, but it should not drive the main bridge after this result.
- Surprises and Tensions:
  - Capitals and elements looked like clean bridge families at the family level, yet both currently cover only one of their three saved modes.
  - The route-mode count is `13`, not `12`, because the saved factual artifact faithfully preserves one author outlier inside a moon-dominant cluster.
- Confidence:
  - high that `resattn-q38` is the right next issue
  - high that `resattn-a1w` should stay bounded
- Interesting facts:
  - matching-family mode coverage `= 5 / 13`
  - all `12` capital prompts nearest-match capital mode `cluster 3`
  - all `12` element prompts nearest-match element mode `cluster 1`
## [2026-03-18T16:08:00-0500] The Honest Redesign Is Narrower, Not Broader
- Stage: bridge design
- Feel of the Experiment: The saved evidence finally made the design choice crisp. A truthful next surface is not “v4 plus a few more prompts.” It is a route-mode-aware collection that narrows to the robust capital, element, and author modes and refuses to pretend that moons or the singleton author outlier are already understood.
- Working Hypotheses:
  - The next informative tool-breakage run should be on a route-mode-aware surface, not on another pooled `v4` rerun.
  - Keeping moons out of the main collection will sharpen the next result rather than weaken it.
- Hunches and Guesses:
  - The capital and element missing modes are clean enough that the next pilot should show a more informative family-conditioned picture quickly.
  - The author story will still be the trickiest because the mode split is real but the phrasing families are partly overlapping.
- Predictions:
  - `tool_breakage_factual_recall_v5` will be a better discriminator than `v4` even before any larger confirm run.
  - If `xfg` still looks mixed, the next bottleneck will be author-mode heterogeneity rather than “we forgot moons.”
- Surprises and Tensions:
  - The singleton author outlier is scientifically real enough to preserve in the notes, but not reliable enough to drive the main collection.
  - Narrowing the main surface actually feels more ambitious here because it forces the next run to answer the route-mode question directly.
- Confidence:
  - high that closing `q38` with a route-mode-aware `v5` surface is the right move
  - medium-high that `xfg` should come before `a1w`
- Interesting facts:
  - `v5` main collection prompt counts `= 10` pilot / `20` confirm
  - robust targeted modes `= 10`
  - moon modes in main collection `= 0`
## [2026-03-18T15:03:00-0500] The Narrowed Route-Mode Bridge Has Real Life
- Stage: tool-breakage pilot
- Feel of the Experiment: This is the kind of positive result I trust more than a flashy pooled improvement. The narrowed `v5` surface did not just stay positive overall; it exposed a sharper route-mode split inside the same healthy pilot.
- Working Hypotheses:
  - `v5` should now be treated as the active factual tool-breakage bridge surface.
  - The next useful question is confirm survival on `v5`, not whether to reopen `v4`.
- Hunches and Guesses:
  - The main confirm risk is not broad failure. It is that a few author and capital modes may dominate the rank-instability diagnostics.
  - Moon prompts are now more likely to stay a sidecar than to re-enter the main bridge soon.
- Predictions:
  - A locked `v5` confirm run will probably stay positive on mean tuned KL while keeping the mixed signal concentrated in the same subset of modes.
  - If confirm fails, the failure will be mode-specific and therefore more informative than any old pooled `v4` rerun.
- Surprises and Tensions:
  - Authors are the strongest family on the narrowed pilot even though author-mode heterogeneity is still the most jagged.
  - All targeted modes are positive on tuned mean KL, but only half avoid final target-rank worsening; that is exactly the kind of mixed-but-structured read the old family summaries could not isolate.
- Confidence:
  - high that `xfg` should close positive
  - medium-high that `138` is the right next step over `a1w`
- Interesting facts:
  - overall `v5` pilot mean tuned KL delta `= +2.8378`
  - authors `= +3.2970`
  - route modes with final-rank worsening `= 5 / 10`
## [2026-03-18T15:13:00-0500] The Confirm Split Strengthened The Narrowed Bridge
- Stage: tool-breakage confirm
- Feel of the Experiment: This is a stronger result than the pilot because it closes the obvious “nice pilot, weak confirm” escape hatch. The narrowed bridge held, and the confirm read is still structured rather than mushy.
- Working Hypotheses:
  - The active factual tool-breakage question should now shift to donor-arm controls on `v5`.
  - The moon sidecar is even less justified as the main next move now that the narrowed bridge has a confirm pass.
- Hunches and Guesses:
  - The donor-arm rerun on `v5` will probably stay healthier than `v4` because the surface is no longer mode-blind.
  - The most fragile confirm mode is author `cluster 12`, not the whole author family.
- Predictions:
  - `hth` will be the real discriminator for whether the narrowed bridge supports a stronger same-model statement.
  - If `hth` turns mixed again, the mixture will be interpretable at the mode level instead of collapsing back into pooled family confusion.
- Surprises and Tensions:
  - Author `cluster 7` became the strongest confirm mode while still carrying full final-rank worsening; that is a useful reminder that KL and rank metrics are not the same object.
  - The confirm split reduced the overall final-rank-worsening fraction to `0.35` even while the range-increase fraction stayed high at `0.85`.
- Confidence:
  - high that `138` should close positive
  - high that `hth` is the next honest move over `a1w`
- Interesting facts:
  - overall `v5` confirm mean tuned KL delta `= +2.9065`
  - strongest confirm mode `= route_mode_author_cluster_7` at `+4.9136`
  - weakest confirm mode `= route_mode_author_cluster_12` at `+1.9931`
## [2026-03-18T15:35:00-0500] The Narrowed Bridge Survives, But The Donor Story Is Still Jagged
- Stage: tool-breakage donor-arm controls
- Feel of the Experiment: This is a useful mixed result, not a disappointment. The fixed-alpha objection weakened again, which means the narrowed bridge is not fake. But the donor-arm read still refuses to flatten into a clean overall yes, and the route-mode split explains why.
- Working Hypotheses:
  - Tool-breakage should not get another pooled rerun next.
  - The next real value in this lane is understanding donor geometry, especially why author modes still dominate the negative within-family result.
- Hunches and Guesses:
  - The partial collapse between `prompt_permuted_alpha` and `within_family_permuted_alpha` is now a real design limitation of the `v5` confirm ordering, not just a reporting quirk.
  - Elements are the healthiest family if this lane ever reopens.
- Predictions:
  - A donor-assignment audit will say “freeze unless you can make the controls more independent,” not “just rerun bigger.”
  - The main paper story should gain more from oracle synthesis than from another factual tool-breakage iteration right now.
- Surprises and Tensions:
  - Author `cluster 7` is cleanly positive on every arm while author `cluster 12` is negative on every dynamic arm. That is stronger heterogeneity than the family-level split suggests.
  - Capitals are almost exactly a tie within-family, which is more interesting than either a clean fail or a clean pass.
- Confidence:
  - high that `hth` should close as mixed
  - high that `qxz` is now more important than any immediate tool-breakage rerun
- Interesting facts:
  - routed minus `pilot_mean_alpha` mean tuned KL `= +1.0898`
  - routed minus `within_family_permuted_alpha` mean tuned KL `= -0.1021`
  - route modes with identical `prompt_permuted_alpha` and `within_family_permuted_alpha` means `= 7 / 10`
## [2026-03-18T15:55:00-0500] The Gemma Story Is Finally Coherent
- Stage: oracle synthesis
- Feel of the Experiment: The repo is finally in a cleaner state than the story around it. The strongest evidence is no longer a scattered pile of artifacts. It points in one direction: effective depth mixture is real on the primary model, competition matters, and the most meaningful raw structure lives inside factual recall rather than in one grand global cluster claim.
- Working Hypotheses:
  - The next high-value move is a factual bridge analysis, not another mixed-surface oracle rerun.
  - The paper should now lead with the primary-model oracle result rather than treating it as one lane among many.
- Hunches and Guesses:
  - The best next cross-lane result is to show that the narrowed `v5` tool-breakage route modes overlap the strongest factual raw-source clusters only partially, not perfectly.
  - Reasoning/math is real but should stay second-tier until it gets an equally strong bridge.
- Predictions:
  - `lnu` will clarify whether the current factual tool-breakage bridge already targets the strongest factual routing modes or only a subset.
  - If we keep the narrative disciplined here, reviewer pushback shifts from “is the signal real?” to “how far can you extend it?”
- Surprises and Tensions:
  - The regime-comparison result is stronger and cleaner than some of the flashier extension lanes, which means we should stop treating it like a side artifact.
  - The global raw-source story is weak enough that forcing it would be dishonest, but the factual-recall raw-source story is strong enough that ignoring it would also be dishonest.
- Confidence:
  - high that `qxz` should close on a narrative correction rather than another run
  - high that `lnu` is the right next bridge issue
- Interesting facts:
  - full-surface raw-source silhouette `= 0.1664`
  - factual-recall raw-source silhouette `= 0.4709`
  - softmax beats every tested alternative on `128 / 128` confirm prompts
## [2026-03-18T16:05:00-0500] The Bridge Coverage Problem Is Basically Solved
- Stage: factual bridge analysis
- Feel of the Experiment: This is cleaner than I expected. The narrowed `v5` bridge is not just directionally better. It already covers all targeted capital and element modes and every non-outlier author mode. That means the remaining tool-breakage mess is about donor geometry, not about missing the strongest factual oracle modes.
- Working Hypotheses:
  - We should stop treating bridge redesign as the main factual tool-breakage lever.
  - If tool-breakage continues, donor-assignment geometry is the right next microscope.
- Hunches and Guesses:
  - The moon sidecar is now even less relevant to the main bridge story.
  - The singleton author outlier is not worth promoting into the main surface unless another lane independently makes it important.
- Predictions:
  - `oi7` will say the remaining mixed donor-arm result is mostly about prompt pairing and control independence, not bridge undercoverage.
  - Another factual bridge redesign before that audit would be wasted motion.
- Surprises and Tensions:
  - The bridge is now structurally aligned even though the donor-arm result is still mixed. That is a useful narrowing result.
  - Author `cluster 7` stays both well covered and very breakage-heavy, which makes it a better diagnostic mode than I initially expected.
- Confidence:
  - high that `lnu` should close positive
  - high that `oi7` is now the only meaningful next tool-breakage follow-up
- Interesting facts:
  - matching-family mode coverage rose from `5 / 13` on `v4` to `10 / 13` on `v5`
  - `v5` covers all capital and element modes
  - the only uncovered author mode is singleton `cluster 10`
## [2026-03-18T16:45:00-0500] The Donor Collapse Was Mostly Structural, But The Author Problem Survived It
- Stage: donor-geometry audit
- Feel of the Experiment: This was the right audit to do because it killed the lazy explanation cleanly. Most of the `prompt_permuted` versus `within_family` collapse really is just the family-grouped confirm ordering. But once that is isolated, author modes `6` and `11` are still negative with identical donors, so we do not get to explain the whole mixed result away.
- Working Hypotheses:
  - The pooled negative within-family read is mostly driven by three family-endpoint prompts.
  - The residual real limitation is author-family donor pairing, not general bridge misalignment.
- Hunches and Guesses:
  - A donor-remap rerun would mostly just move the pooled number closer to zero and tempt over-interpretation.
  - Freezing the lane is more truthful than trying to engineer one more same-model donor win.
- Predictions:
  - If we ever revisit donor remapping, the main movement would come from `tb5-confirm-020` and the other family-endpoint prompts rather than from a broad shift across modes.
  - Author modes `6` and `11` would still be the honest blockers even after a cleaner remap.
- Surprises and Tensions:
  - The differing prompts explain `82.9%` of the total negative within-family sum, which is larger than I expected.
  - Even so, author `cluster 12` is not the main within-family blocker. `6` and `11` are.
- Confidence:
  - high that `oi7` should close with a freeze recommendation
  - high that donor-remap is not the right default next move
- Interesting facts:
  - identical donor source prompt on `17 / 20` prompts
  - identical route-mode mean tuned-KL delta on `7 / 10` modes
  - author family contribution sum `= -2.7318`
  - `route_mode_author_cluster_6 = -1.9465`, `route_mode_author_cluster_11 = -0.6423`
## [2026-03-18T17:20:00-0500] The Moon Miss Was Mostly Bad Surface Design
- Stage: moon-family audit
- Feel of the Experiment: This one got cleaner the deeper I looked. The moon family was not failing because “moons just don’t work.” It was failing because the sidecar quietly changed styles between pilot and confirm, and the worst donor-arm read was hanging on a single boundary wrap into a capital prompt.
- Working Hypotheses:
  - The moon sidecar is underdesigned rather than intrinsically null.
  - If it ever matters again, prompt rewrite is the right first move.
- Hunches and Guesses:
  - The mythological `named after` style is just a worse surface for this family than the direct descriptor prompts.
  - A donor-remap rerun on the existing `v4` moon prompts would mostly produce cleaner-looking numbers without answering the real design flaw.
- Predictions:
  - A moon-only rewrite that keeps pilot and confirm on the same descriptor style would look materially healthier than the saved `v4` moon family.
  - Cross-family moon donors would still need separate thought, because the all-capital donor map is too blunt.
- Surprises and Tensions:
  - The whole family-level `prompt_permuted` gap came from one prompt, `tb4-confirm-032`.
  - The within-family moon read is not dead; it is just too dependent on a messy confirm surface to support a stronger claim.
- Confidence:
  - high that `a1w` should close with a freeze recommendation
  - medium-high that a rewrite would help if the sidecar ever mattered again
- Interesting facts:
  - moon pilot descriptor prompts `= 4 / 4`
  - moon confirm `named after` prompts `= 6 / 8`
  - all `8 / 8` moon cross-family donors are capitals
## [2026-03-18T18:05:00-0500] Phase 6 Is Scientifically Ready But Operationally Under-Cached
- Stage: router-distillation readiness
- Feel of the Experiment: This is exactly the kind of blocker I want to find before writing training code. The saved Gemma campaign is strong enough that Phase 6 is worth doing now, but the current checkpoint format is still built for sequence-level predictiveness, not token-level distillation.
- Working Hypotheses:
  - The next real bottleneck is data export, not model design.
  - A pilot-only per-token export slice is enough to unblock the prereg router lane without another broad oracle rerun.
- Hunches and Guesses:
  - Once the per-token export exists, the first router pilot will move quickly because the oracle lane already has strong signal.
  - The aggregation choice from per-token router outputs back to sequence-level alpha will be the next real design fork, not checkpointing itself.
- Predictions:
  - The smallest honest next implementation will be adding token ids plus `h_1[t]` and `h_4[t]` to pilot checkpoints, not touching training yet.
  - The lack of saved per-token supervision would otherwise create a fake “training problem” that is really just a data problem.
- Surprises and Tensions:
  - The runner already computes the token-level states we need, but the campaign throws them away.
  - The repo is closer to Phase 6 scientifically than operationally.
- Confidence:
  - high that `5qd` should close as a readiness audit
  - high that per-token export is the next honest slice
- Interesting facts:
  - oracle checkpoints already persist sequence-level `final_alpha`
  - feature checkpoints persist only one aggregated vector per prompt
  - no token ids or per-token hidden states are saved in the current campaign
## [2026-03-18T16:10:00-0500] Phase 6 Is Finally Past The Data-Readiness Excuse
- Stage: router-distillation export
- Feel of the Experiment: This is the kind of infrastructure result that matters because it removes a fake blocker. The repo no longer gets to say “router training is next once we cache the right things.” The right things are now cached on the real Gemma pilot surface.
- Working Hypotheses:
  - The next genuine uncertainty is model-side: whether a small 2-layer router on `h_1[t]` or `h_4[t]` can recover enough of the saved oracle signal on pilot.
  - The aggregation choice from per-token router outputs back to sequence-level alpha will become the next real design fork only after the first pilot fit exists.
- Hunches and Guesses:
  - `h_1[t]` is still the right prereg favorite, but the saved `h_4[t]` export makes the comparison cheap enough that there is no reason to guess.
  - Balanced `64 x 4` pilot coverage will matter more than squeezing another export field into the dataset.
- Predictions:
  - The first router pilot will expose the real bottleneck quickly: either token-level input choice, sequence aggregation, or model capacity.
  - If Phase 6 stalls now, it will be for modeling reasons rather than missing supervision.
- Surprises and Tensions:
  - The export path itself was not the hard part. The subtle bug was schema drift: it was easy to make the manifest more complete than the saved prompt checkpoints until the invalidation guard was explicit.
  - Once that was fixed, the completed rerun was very fast. Resume reuse on the full output is only `10.07s`.
- Confidence:
  - high that `1ot` should close as a pass
  - high that the next honest move is router fitting, not more export plumbing
- Interesting facts:
  - full pilot export size: `256` prompts, `4013` total tokens
  - prompt checkpoint sample hash stayed unchanged after exact-command rerun: `6dd29ee0a297d5c085c4e895aa4a64a624aa1d41`
  - every exported prompt still carries `prompt_paraphrase` metadata, which will matter for future pilot-side robustness checks
## [2026-03-18T16:32:00-0500] The First Router Pilot Failed In A Useful Way
- Stage: router-distillation pilot
- Feel of the Experiment: This is disappointing in the right way. The router lane is no longer blocked by missing data, and the first actual fit immediately exposed a modeling problem instead of giving a fake maybe-positive. That is valuable.
- Working Hypotheses:
  - The main blocker is not `h_1[t]` versus `h_4[t]` by itself.
  - The current raw-alpha MSE target plus `mean_token_logits_then_softmax` aggregation is too collapse-prone on these diffuse `53`-source targets.
- Hunches and Guesses:
  - A target-space redesign is the right next lever before an aggregation sweep.
  - The tiny `h_4[t]` win is probably real only in the narrow sense that early context is not worse than `h_1[t]`, but it is nowhere near strong enough to lock.
- Predictions:
  - A logit-space or compressed-target pilot will move more than another blind `h_1[t]` versus `h_4[t]` rerun.
  - If the next pilot still predicts almost-uniform mixtures, the real blocker becomes aggregation or router output structure, not export or hidden-state choice.
- Surprises and Tensions:
  - The router barely trains at all in the useful sense: best epoch is `1` or `2`, and both held-out predictions stay near the uniform entropy ceiling.
  - `h_4[t]` wins, but only by a trivial amount, so the run answers the input question only negatively.
- Confidence:
  - high that `m6r` should close as a negative pilot
  - medium-high that `4hj` is the right next Phase 6 step
- Interesting facts:
  - `h_1[t]` held-out `R^2 = -0.0599`
  - `h_4[t]` held-out `R^2 = -0.0524`
  - mean predicted entropy `≈ 3.969` for both inputs versus mean oracle entropy `= 3.514`
  - rerunning after compacting the saved summary artifact changed the exact values slightly on MPS but did not change the qualitative result
## [2026-03-18T16:54:40-0500] Target Geometry Was Real, But It Was Not The Whole Story
- Stage: router-distillation target comparison
- Feel of the Experiment: This is the kind of partial rescue I trust. The old failure was not fake, but it was also not the whole truth. Changing only the target geometry unlocked a large jump without magically solving the lane.
- Working Hypotheses:
  - `oracle_alpha_logit_vector` should stay frozen as the target baseline for the next pilot slice.
  - The next real bottleneck is sequence aggregation, not another pass at `h_1[t]` versus `h_4[t]`.
- Hunches and Guesses:
  - Mean-token aggregation is probably washing out exactly the token-local structure the lane is supposed to exploit.
  - A capacity sweep before aggregation would blur the diagnosis because the target change already did the heavy conceptual work.
- Predictions:
  - An aggregation comparison will move more than a width-only sweep on this fixed pilot surface.
  - `h_4[t]` may stay slightly better than `h_1[t]`, but I still do not expect a decisive input lock until aggregation changes.
- Surprises and Tensions:
  - The gain from target geometry is bigger than I expected: held-out `R^2` moved from negative to `0.3028`.
  - The exact-command rerun changed the summary hash on MPS, so the decimal-level values are not stable enough to fetishize even though the ordering is.
- Confidence:
  - high that `4hj` should close as a partial pass
  - high that `914` is the next honest Phase 6 step
- Interesting facts:
  - raw-alpha `h_4[t]`: `R^2 = -0.0486`, mean JS `= 0.1565`
  - alpha-logit `h_4[t]`: `R^2 = 0.3028`, mean JS `= 0.0835`
  - input ranking stayed unchanged across targets
## [2026-03-18T19:31:01-0500] The Ready Queue Is Not The Same Thing As The Best Scientific Next Step
- Stage: repo-wide priority review
- Feel of the Experiment: The repo has enough positive signal now that sequencing matters more than hustle. If I only follow the ready implementation issue, I risk underusing the strongest result we already have.
- Working Hypotheses:
  - The main scientific next step is factual-recall oracle synthesis on saved Gemma artifacts.
  - The main implementation next step is still router aggregation on the saved pilot export.
- Hunches and Guesses:
  - The factual-recall route-mode story is closer to a meaningful interpretability finding than another immediate router pilot tweak.
  - `resattn-914` is still worth doing soon, but it should not crowd out the strongest existing structure result.
- Predictions:
  - A careful factual-recall synthesis pass will sharpen the paper more than another pooled rerun in any secondary lane.
  - Figure 8 and broader tool-breakage reruns would be a distraction right now unless a materially stronger proxy or bridge surface appears.
- Surprises and Tensions:
  - The repo already knows this in pieces, but the ready queue hid it: the strongest underexploited positive result is not the newest implementation lane.
  - Safety is methodologically healthier than I expected, but still not the right main focus because the mediator-conditioned result remains role-collapsed.
- Confidence:
  - high that factual-recall synthesis should be the next scientific step
  - high that `914` should stay next on the implementation side
- Interesting facts:
  - full `registry_v5` Gemma oracle improvement `= +1.6299` nats on `1024` confirm prompts
  - factual-recall raw-source silhouette `= 0.4709`
  - router pilot after target fix `R^2 = 0.3028`, still below the Phase 6 readiness gate
## [2026-03-18T19:43:56-0500] The Factual Story Is Sharper Than “Semantic Family” But Narrower Than “Concept Circuit”
- Stage: factual-recall synthesis
- Feel of the Experiment: This is the kind of refinement I want. The strongest result got more interesting and more constrained at the same time.
- Working Hypotheses:
  - The best current claim is family-plus-prompt-frame-conditioned route modes inside factual recall.
  - The next saved-artifact question is whether that frame-conditioned read survives a more explicit audit.
- Hunches and Guesses:
  - Capitals and elements are almost template-identified by route mode.
  - Authors are a little messier, but the same family-plus-frame pattern is still there.
- Predictions:
  - A dedicated prompt-frame audit will sharpen the claim boundary further without needing another expensive oracle rerun.
  - If the frame-conditioned story holds, it will make the current tool-breakage bridge easier to interpret rather than weaker.
- Surprises and Tensions:
  - The route-mode bridge is better aligned than I expected; the mixed donor-arm result is harder to dismiss as simple coverage failure now.
  - The more I read the saved examples, the less I believe the strongest factual route modes are “just semantics” in a narrow answer-family sense.
- Confidence:
  - high that `chz` should close as a meaningful synthesis pass
  - medium-high that `i6i` is the right next scientific follow-up
- Interesting facts:
  - factual bridge coverage now spans all capital modes, all element modes, and all non-outlier author modes
  - strongest covered mode is author `cluster 7` with mean tuned final-position KL delta `= +12.7356`
  - element family still has the largest within-family centroid JS separation `= 0.2096`

## [2026-03-18T20:02:29-0500] The Frame Question Mostly Collapsed Cleanly
- Stage: factual route-mode frame audit
- Feel of the Experiment: This is a satisfying kind of narrowing. The result got more specific without getting weaker. The repo no longer has to gesture at prompt framing from a few examples; the capital and element families are almost embarrassingly template-organized, and authors are only slightly messier.
- Working Hypotheses:
  - The strongest current factual claim should now explicitly say family-plus-prompt-frame-conditioned route modes.
  - Any future factual sidecar should focus only on the residual author `novel_title` split.
- Hunches and Guesses:
  - The two `novel_title` spillovers into the direct-author cluster feel more like title-shape or lexical-surface effects than like a new semantic family.
  - The frame-conditioned read actually makes the bounded tool-breakage bridge easier to interpret, not less interesting.
- Predictions:
  - `resattn-914` is now the right active next move because the biggest remaining factual-structure ambiguity is small and local.
  - If we come back to factual route modes later, it should be for a tiny author-title audit rather than another broad family/frame rerun.
- Surprises and Tensions:
  - Capitals and elements are cleaner than I expected: they are not merely frame-biased; they are deterministic frame-group splits across all `16` entities.
  - Authors are messier in exactly one place, and even there the mess is structured rather than diffuse.
- Confidence:
  - high that `i6i` should close as a pass
  - high that the strongest current factual claim is family-plus-frame-conditioned, not family-only
- Interesting facts:
  - capital modes partition exactly into `direct+travel`, `map`, and `quiz`
  - element modes partition exactly into `notes+lab`, `direct`, and `periodic table`
  - covered author modes have frame-majority share `= 61 / 63 = 0.9683`

## [2026-03-18T20:18:34-0500] The Obvious Aggregation Rescue Failed
- Stage: router-distillation aggregation comparison
- Feel of the Experiment: This is the right kind of negative result. It killed a plausible next excuse cleanly. The lane no longer gets to say “maybe the whole problem is just mean pooling” without stronger evidence.
- Working Hypotheses:
  - The next honest Phase 6 blocker is model capacity or router family, not another obvious aggregation variant on this fixed export.
  - `h_4[t]` is still provisionally best, but the lane is not bottlenecked on the input choice right now.
- Hunches and Guesses:
  - Last-token-only is too narrow for this prompt-only export, but its failure still matters because it was the most defensible token-specific rescue to try first.
  - A width comparison is now more likely to teach something than another hand-picked aggregation tweak.
- Predictions:
  - `resattn-3ak` should be the next main implementation step.
  - If a larger width still fails badly, the next blocker will likely be router family rather than another scalar hyperparameter.
- Surprises and Tensions:
  - The negative result is fairly clean: both `h_1[t]` and `h_4[t]` get worse under `last_token`.
  - MPS still nudges the decimals and hashes around, which is annoying, but the ordering held on rerun.
- Confidence:
  - high that `914` should close as a negative aggregation comparison
  - medium-high that capacity is the next honest lever
- Interesting facts:
  - best retained pilot combination: `oracle_alpha_logit_vector + mean_token_logits_then_softmax + h_4[t]`
  - rerun-stable ordering: `mean` stayed above `last` for both inputs

## [2026-03-18T20:40:28-0500] Width Was Real Signal, But Not the Rescue
- Stage: router-distillation capacity comparison
- Feel of the Experiment: This is the right kind of mixed result. The run moved just enough to say width is not totally irrelevant, but not enough to justify another width-by-inertia sweep. The more important win is that the seed bug is now fixed and the rerun finally stayed bit-stable.
- Working Hypotheses:
  - The next honest Phase 6 blocker is router family, not another width bump.
  - `h_4[t]` is still the best tested baseline input, but the lane is not bottlenecked on input choice right now.
- Hunches and Guesses:
  - A linear-versus-MLP family comparison is more likely to teach something than `512` versus `768`.
  - The tiny `h_4[t]` gain from width may just mean the current family can exploit a little more smooth capacity, not that it is the right family for the target.
- Predictions:
  - `resattn-zic` should be the next main Phase 6 step.
  - If a router-family comparison also stays bounded, the next blocker will likely be target object or supervision granularity rather than another architecture scalar.
- Surprises and Tensions:
  - The most useful result in this pass may be operational rather than scientific: the fitter was not seeding Torch initialization, which was a real bug.
  - After the seeding fix, the rerun staying on the exact same hash is unusually satisfying on MPS.
- Confidence:
  - high that `3ak` should close as a mixed width-only result
  - medium-high that router family is now the next honest lever
- Interesting facts:
  - best retained pilot combination after the width sweep: `oracle_alpha_logit_vector + mean_token_logits_then_softmax + h_4[t] + hidden_dim=512`
  - stable rerun hash after the seeding fix: `0cd7c7bee688d503d44f06b54ea9200d634c8b57`

## [2026-03-18T21:00:00-0500] Nonlinearity Didn’t Save It Either
- Stage: router-distillation family comparison
- Feel of the Experiment: This is the right kind of narrowing again. The MLP story just got weaker in a useful way. I expected either a clean MLP win or a noisy tie; instead the linear head actually won on the primary metric and lost only slightly on JS.
- Working Hypotheses:
  - The next honest Phase 6 blocker is supervision granularity, not another architecture scalar.
  - The least-bad current baseline is now the linear head on `h_4[t]`, but it is still nowhere near a readiness pass.
- Hunches and Guesses:
  - The saved prompt-level target is probably too coarse for the token-level object we eventually want a router to learn.
  - Another family tweak without changing the supervision story would mostly be motion, not discovery.
- Predictions:
  - `resattn-but` should be the next main Phase 6 step.
  - If the supervision-granularity audit points cleanly to coarse targets, the next real implementation move should be richer token/span supervision rather than another head redesign.
- Surprises and Tensions:
  - The linear win on `R^2` is real but only modest, and the JS metric still prefers the MLP slightly.
  - That mixed metric read is annoying, but it is also exactly why the fixed selection rule matters.
- Confidence:
  - high that `zic` should close as a negative result for nonlinear-family rescue
  - medium-high that supervision granularity is now the right next Phase 6 question
- Interesting facts:
  - `linear`: `R^2 = 0.3445`, mean JS `= 0.0853`
  - `mlp`: `R^2 = 0.3270`, mean JS `= 0.0837`
  - stable rerun hash: `89f3b2682375aa7956f969879af7479e85852fb9`

## [2026-03-18T21:20:00-0500] The Residual Author Split Looks Lexical, Not Like a New Mode
- Stage: factual route-mode cleanup
- Feel of the Experiment: This is the sort of sidecar I want more of. It narrows a lingering ambiguity without pretending the repo discovered a whole new result.
- Working Hypotheses:
  - The residual `novel_title` split is mostly lexical/title-shape noise on top of the stronger family-plus-frame structure.
  - It is not evidence for a distinct answer-entity routing mode.
- Hunches and Guesses:
  - `Moby-Dick`, `Frankenstein`, and `Things Fall Apart` are exceptional because their title surfaces are unusual in different ways, not because Melville, Shelley, and Achebe form a coherent author cluster.
- Predictions:
  - `0kc` should close the factual sidecar cleanly rather than spawning another broad audit.
  - The next high-value work should stay on `resattn-but`, not on more author-title prompt archaeology.
- Surprises and Tensions:
  - The lexical read is plausible, but it is still only `n = 3` exceptional titles, so it should stay bounded.
  - The singleton `Things Fall Apart` outlier is interesting, but not enough to justify reopening the broader factual clustering story.
- Confidence:
  - medium-high that the residual split is better read as lexical/title-shape than answer-entity
  - high that this is cleanup, not a new core discovery
- Interesting facts:
  - spillovers: `Moby-Dick`, `Frankenstein`
  - singleton outlier: `Things Fall Apart`

## [2026-03-18T21:39:10-0500] The Router Errors Finally Look Like the Supervision Object, Not the Head
- Stage: router-distillation supervision-granularity audit
- Feel of the Experiment: This is the first Phase 6 result in a while that feels like a real design pivot instead of another negative knob turn. The useful surprise is that the worst held-out stratum is `general_text`, not code, and the scalar “difficulty” surrogates are too weak to explain the split away.
- Working Hypotheses:
  - The next honest Phase 6 move is richer token/span supervision on the same saved pilot split.
  - Another width or family tweak before changing the supervision object would mostly be ritual.
- Hunches and Guesses:
  - The current sequence-level alpha target is probably adequate for factual prompts and visibly too coarse for more open-ended prompt frames.
  - General-text prompts may be exposing a token-local routing signal that the current single-vector target washes out.
- Predictions:
  - `resattn-tqn` should be the next main Phase 6 step.
  - If a minimally richer supervision target helps, the gain should appear first on `general_text` and `code_procedural`, not on factual prompts.
- Surprises and Tensions:
  - `general_text` landing above `code_procedural` was not my prior.
  - The strongest scalar correlation being only `|r| = 0.1654` is exactly the kind of weak explanation I wanted to see before blaming the supervision object.
- Confidence:
  - high that `resattn-but` should close as a real diagnosis
  - medium-high that token/span supervision is the next honest lever
- Interesting facts:
  - frozen-baseline aggregate matched the saved family artifact exactly on the same split
  - rerun-stable audit hash: `e55d9afc17c33a0b3d4ad3961a3d443cea7895dc`

## [2026-03-18T22:09:39-0500] Denser All-Token Supervision Actually Pays Off
- Stage: router-distillation supervision-objective comparison
- Feel of the Experiment: This is the right kind of positive result. It is not flashy, but it changes the design landscape. The important surprise is that `all_tokens` helps while `last_third` hurts, which means the missing signal is not just “closer to the answer token.”
- Working Hypotheses:
  - The new least-bad Phase 6 baseline should be `all_tokens_target_mse`, not the old sequence-level objective.
  - The next honest architecture question is whether the old linear-vs-MLP result survives under this better supervision surface.
- Hunches and Guesses:
  - The all-token win looks more like distributed local regularization than span localization.
  - If another gain is available without a new export, it is more likely to come from the family comparison under `all_tokens` than from another handcrafted span mask.
- Predictions:
  - `resattn-d36` should be the next main Phase 6 step.
  - If `d36` stays linear-favored even under `all_tokens`, the next move should probably be more faithful tokenwise targets rather than more architecture churn.
- Surprises and Tensions:
  - `general_text` stayed the worst stratum even after the all-token improvement, which is consistent with the earlier audit rather than contradicting it.
  - `last_third` hurting factual recall this badly is a cleaner negative than I expected.
- Confidence:
  - high that `tqn` should close as a real positive result
  - medium-high that the next bounded follow-up should be family-under-all-tokens, not an immediate export redesign
- Interesting facts:
  - `all_tokens_target_mse`: `R^2 = 0.4076`, mean JS `= 0.0797`
  - rerun-stable comparison hash: `13cbb0d9d69f05cdb5d97e0ed69b66d0f72ea350`

## [2026-03-18T22:50:00-0500] The MLP Came Back, But Only a Little
- Stage: router-distillation family comparison under all-token supervision
- Feel of the Experiment: This is a useful correction, not a breakthrough. The old linear win really was partly about the weaker supervision surface. But the new MLP win is still small enough that I would not trust any story that says “the remaining problem was just head family.”
- Working Hypotheses:
  - The retained Phase 6 baseline should now be `mlp + h_4[t] + oracle_alpha_logit_vector + mean_token_logits_then_softmax + all_tokens_target_mse`.
  - The next honest blocker is target fidelity, not another blind architecture sweep.
- Hunches and Guesses:
  - Repeating one sequence-level teacher target at every token is now the most suspicious simplification left on the saved export.
  - If a bigger jump is still available on this pilot surface, it is more likely to come from bounded tokenwise teacher targets than from `mlp` versus `deeper_mlp`.
- Predictions:
  - `resattn-d36` should close as a bounded pass.
  - `resattn-afu` should be the next main Phase 6 step.
- Surprises and Tensions:
  - I expected either a tie or a noisier rerun. Getting a stable hash with the MLP now winning both metrics is cleaner than that.
  - The gain is real but annoyingly small, which is exactly the kind of result that can tempt over-iteration if we do not freeze the architecture surface on purpose.
- Confidence:
  - high that the Phase 6 baseline should move back to the MLP under `all_tokens_target_mse`
  - high that more faithful tokenwise teacher targets are now the right next question
- Interesting facts:
  - `linear`: `R^2 = 0.4076`, mean JS `= 0.0797`
  - `mlp`: `R^2 = 0.4211`, mean JS `= 0.0785`
  - rerun-stable family-comparison hash: `0ceba4be9932c94ff312c7724cf8528fb856c15c`

## [2026-03-18T23:59:20-0500] The Approximate Tokenwise Teacher Was the Wrong Idea
- Stage: router-distillation bounded tokenwise teacher comparison
- Feel of the Experiment: This is a good negative result. The bounded teacher did not merely fail to help; it failed hard enough to kill the tempting story that “anything more token-local must be better.” The matched next-token mask control being almost identical to the baseline makes the read cleaner.
- Working Hypotheses:
  - The retained Phase 6 baseline should stay `mlp + h_4[t] + oracle_alpha_logit_vector + mean_token_logits_then_softmax + all_tokens_target_mse`.
  - The next honest target-fidelity test is a small exact tokenwise oracle slice, not a full export redesign and not another architecture tweak.
- Hunches and Guesses:
  - The shared-final-norm next-token contribution teacher is misaligned with the sequence-level alpha object we eventually decode and score.
  - If true tokenwise oracle targets help, they will need to be genuinely exact on a bounded subset, because this approximation is too wrong to extrapolate from.
- Predictions:
  - `resattn-afu` should close as a stable negative result for the bounded approximation.
  - `resattn-7xo` is now the right next step.
- Surprises and Tensions:
  - The first real launch catching the `RMSPre` assumption bug and the missing `_fixed_residual_sources` import was annoying, but also exactly why the first real run mattered.
  - The matched-mask control staying almost tied means “next-token positions only” is not the missing lever here.
- Confidence:
  - high that the bounded contribution teacher should not be used again on this lane
  - medium-high that the next honest question is exact tokenwise oracle fidelity on a smaller subset
- Interesting facts:
  - `all_tokens_target_mse`: `R^2 = 0.4211`, mean JS `= 0.0785`
  - `next_token_positions_sequence_target_mse`: `R^2 = 0.4201`, mean JS `= 0.0792`
  - `next_token_positions_oracle_alpha_target_logit_contribution_mse`: `R^2 = -11.2639`, mean JS `= 0.4320`
  - rerun-stable comparison hash: `97872d32c0f5fa3d6bf5c75bd7f06c7acab5874a`

## [2026-03-19T00:38:11-0500] Exact Tokenwise Teachers Still Didn’t Save It
- Stage: exact tokenwise-oracle subset comparison
- Feel of the Experiment: This is the kind of negative result I trust. We removed the approximation excuse, paid the runtime cost, and the exact teacher still lost. That narrows the space a lot. The only wrinkle is that the support-only next-token mask got a small win on the bounded subset, which is interesting but not big enough to rewrite the larger full-split story.
- Working Hypotheses:
  - The broader Phase 6 default should stay the retained all-token sequence target, because the larger `192 / 64` split still outweighs the subset-local support-only gain.
  - The next honest question is diagnosis, not redesign: is the exact teacher failing because tokenwise routes vary too much within prompts for mean-token decoding, or because the sequence-level alpha target is simply the wrong thing to compare tokenwise supervision against?
- Hunches and Guesses:
  - The exact tokenwise teacher is probably too high-variance within prompt for the current sequence aggregation to absorb cleanly.
  - The small positive shift for `next_token_positions_sequence_target_mse` may just be a support/alignment effect on the bounded slice rather than a new default objective.
- Predictions:
  - `resattn-7xo` should close as a stable negative result for exact teacher rescue.
  - `resattn-b4h` is the right next step.
- Surprises and Tensions:
  - The exact teacher being worse than both controls is cleaner than I expected after paying the “exactness” tax.
  - The support-only control beating `all_tokens` on the bounded slice but not on the larger full split is exactly the sort of scale-sensitive tension that should slow down any baseline switch.
- Confidence:
  - high that a full tokenwise export redesign is not justified by current evidence
  - medium-high that the next value is in diagnosing mismatch rather than launching another training run
- Interesting facts:
  - `all_tokens_target_mse`: `R^2 = 0.1567`, mean JS `= 0.1014`
  - `next_token_positions_sequence_target_mse`: `R^2 = 0.1803`, mean JS `= 0.0998`
  - `next_token_positions_exact_oracle_alpha_logit_mse`: `R^2 = -0.1126`, mean JS `= 0.1486`
  - rerun-stable subset hash: `ff6c608adfbb05dcf5895a11ce8740553304f9b2`

## [2026-03-19T10:25:00-0500] The Queue Was Drifting Toward Phase 6 By Recency, Not By Strength
- Stage: cross-session synthesis and reprioritization
- Feel of the Review: The project is in better shape than the ready queue made it look. The strongest results are not the latest router-training negatives; they are the saved primary-model Gemma oracle artifacts plus the factual route-mode synthesis. The risk was not lack of progress. It was letting recency bias quietly narrow the whole project to Phase 6.
- Working Hypotheses:
  - `resattn-b4h` is still the next honest implementation step because it tells us whether the tokenwise-teacher path is misaligned or just too high-variance.
  - the main underexplored science result is now reasoning/math, not another factual cleanup and not another broad oracle rerun.
- Hunches and Guesses:
  - reasoning/math may be the first place where the raw-source structure looks more task-like and less template-dominated than factual recall.
  - if that stratum turns out to be mostly frame-driven too, that is still valuable because it tightens the overall interpretation boundary rather than just adding another positive story.
- Predictions:
  - `resattn-v7h` is worth doing even though it is saved-artifact-only, because it could materially sharpen what kind of interpretability signal the primary Gemma oracle is actually exposing.
  - if `b4h` points mostly to aggregation mismatch, the right follow-up will be a bounded decoder/objective diagnosis, not a larger tokenwise export.
- Surprises and Tensions:
  - the queue having only `resattn-b4h` ready understated how much meaningful science is already sitting in saved artifacts.
  - the project is now strongest when it says “primary-model effective depth mixture plus grouped and stratum-conditioned structure,” not when it chases the next training tweak.
- Confidence:
  - high that Figure 8 and tool-breakage should stay frozen at their current honest boundaries
  - high that the next queue should expose both a scientific saved-artifact step and an implementation step
  - medium-high that reasoning/math is the right underexplored science follow-up

## [2026-03-19T11:10:00-0500] Reasoning/Math Was More Task-Like Than Factual, But Not Cleaner
- Stage: saved-artifact reasoning/math route-mode audit
- Feel of the Result: This is a genuinely useful secondary result. The good surprise is that reasoning/math is more operation-dominated than the factual stratum at the top cluster level. The limiting surprise is that the within-operation splits are still heavily frame-conditioned, so this is not the clean “content-only reasoning route” story either.
- Working Hypotheses:
  - Factual recall should remain the main structured-interpretability center because it is both stronger and already bridgeable.
  - Reasoning/math should be kept as the strongest supporting contrast: it shows that the primary Gemma oracle signal is not only factual-family routing, but it still does not justify a new bridge lane ahead of Phase 6.
- Hunches and Guesses:
  - The mixed arithmetic/sequence and arithmetic/magnitude clusters may be tracking abstract surface forms like continuation or comparison statements more than semantic subcategory labels.
  - If reasoning/math comes back later, the honest move is to inspect those cross-operation joins directly rather than rerunning the whole oracle surface.
- Predictions:
  - `resattn-v7h` should close cleanly.
  - `resattn-b4h` is again the highest-value next active step.
- Surprises and Tensions:
  - Weighted dominant-subcategory majority (`0.8125`) came out much higher than weighted dominant-frame majority (`0.5938`), which is more task-like than the factual surface.
  - Arithmetic still ended up almost perfectly frame-organized (`0.9844`), which keeps me from wanting to overstate the result.
- Confidence:
  - high that reasoning/math is a real secondary positive
  - high that it should not displace factual recall as the main bridge lane
  - medium that the mixed cross-operation joins would repay a later targeted audit

## [2026-03-19T08:40:03-0500] The AttnRes Memoir Is More Useful As Design Taste Than As Evidence
- Stage: planning / synthesis
- Feel of the Review: This was a good correction. The memoir is relevant, but not in the lazy way. It does not license stronger claims for our frozen-model results. It does sharpen what the next honest Phase 6 comparison should look like.
- Working Hypotheses:
  - If Phase 6 improves again, a more likely win is a compressed target that preserves residual-style equal mixing as a special case than a sparser target that cannot represent the baseline.
  - Embedding being isolated as its own block is specific enough to test directly on the saved Gemma pilot surface.
- Hunches and Guesses:
  - The exact-teacher failure may be telling us that the trainable object wants a coarser but still competition-preserving target, not a more exact but noisier tokenwise teacher.
  - A block-compressed target with embedding singled out could end up more trainable without needing to pretend the raw-source `~8`-cluster claim is solved.
- Predictions:
  - `resattn-b4h` should still run first because it tells us whether the failure is teacher variance or aggregation mismatch.
  - If `b4h` does not kill target-design work entirely, the next honest comparison is embedding-isolated compression, not another width/family tweak.
- Surprises and Tensions:
  - The memoir's strongest overlap with our repo is not Figure 8. It is the compression-versus-sparsity argument and the embedding singleton choice.
  - That creates a temptation to overread their `~8` block choice into our raw-cluster lane, which would be sloppy.
- Confidence:
  - high that the memoir should influence queue discipline
  - high that it should not move any claim boundary by itself
- Interesting facts:
  - Their story explicitly says sliding-window style sparsity underperformed because it could not recover the residual baseline.
  - Their block design isolates embedding and compresses the rest into a small number of blocks rather than dropping history outright.
- Sidecar research:
  - This is one of the cleaner external arguments for testing compressed Phase 6 teacher targets before any more open-ended architecture search.

## [2026-03-19T09:27:54-0500] The Exact-Teacher Story Finally Collapsed The Right Way
- Stage: analysis / diagnosis
- Feel of the Result: This is the clean negative I wanted. The wrong easy story would have been “maybe we just aggregate tokenwise teachers badly.” The audit makes that much harder to believe. Mean aggregation is already the less-bad sequence summary; the deeper problem is that the tokenwise teachers fight each other within prompts and average into something blurrier than the sequence oracle.
- Working Hypotheses:
  - The current Phase 6 failure is dominated by contradictory within-prompt tokenwise supervision, not by the current sequence aggregation rule.
  - The next useful target-design move is coarser compression that preserves competition, not more exact teacher fidelity.
- Hunches and Guesses:
  - The exact-tokenwise objective is probably too local relative to the sequence-level routing object we actually care about.
  - Embedding-isolated block compression is now the cleanest next comparison because it moves toward a coarser object without abandoning the competitive routing geometry.
- Predictions:
  - `resattn-b4h` should close cleanly.
  - `resattn-y4m` is now the right next Phase 6 step.
- Surprises and Tensions:
  - The last-position exact teacher is much worse than the prompt-mean exact teacher, which is stronger evidence against a simple aggregation bug than I expected.
  - The entropy gap is large enough that even the averaged exact teachers feel like the wrong object, not just a noisy one.
- Confidence:
  - high that exact-tokenwise teacher work should stay frozen on this pilot surface
  - medium-high that a coarser compressed target is the right next design branch
- Interesting facts:
  - mean within-prompt JS to the prompt-mean exact teacher: `0.1292`
  - mean prompt-mean exact-teacher JS to sequence oracle: `0.0832`
  - mean last-position exact-teacher JS to sequence oracle: `0.1910`
  - mean exact-teacher entropy minus oracle entropy: `+0.3727`
- Sidecar research:
  - This is the strongest internal argument yet for compression-over-sparsity in Phase 6: not because AttnRes used blocks, but because our exact tokenwise teachers are too internally inconsistent to be the trainable object we want.

## [2026-03-20T11:45:00-0500] High-Leverage Plan Feels Cleaner Than the Old “One More Tweak” Loop
- Stage: pre-paper prioritization and Phase 6 re-entry
- Feel of the Review: The repo is in a strong place scientifically, but recent queue shape risked spending more cycles on local maxima. The new gameplan feels cleaner: one decisive Phase 6 bundle, one decisive tool-breakage control pass, one safety surface redesign, one external anchor package.
- Working Hypotheses:
  - If Phase 6 cannot clear `R^2 >= 0.5` on the scaled pilot-to-confirm bundle, additional geometry tweaks are unlikely to change the paper story this cycle.
  - Tool-breakage value is now in route-mode-aware dynamic control adjudication, not another baseline rerun.
  - Safety value is now in breaking role collapse by prompt design, not another v2 mediator replay.
- Surprises and Tensions:
  - The split export run was slower than the earlier pilot-only intuition but still very manageable on MPS.
  - The biggest remaining risk feels methodological (confirm leakage and control alignment), not missing signal.
- Confidence:
  - high that the new high-leverage sequence is better than continuing ad hoc sweeps
  - medium-high that `resattn-r7s` now has the right first slice in place (split-safe pilot+confirm exports)

## [2026-03-20T11:50:32-0500] The Decisive Phase 6 Bundle Reduced Uncertainty, Not Blockers
- Stage: Phase 6 adjudication
- Feel of the Result: This was worth doing. The run bundle did exactly what it should: removed the “maybe we just need one bigger clean run” ambiguity. It did not rescue readiness.
- Working Hypotheses:
  - Router-distillation readiness is still blocked on this cycle for the retained objective family.
  - The next value is in moving to tool-breakage/safety adjudication, not reopening Phase 6 geometry churn.
- Surprises and Tensions:
  - `h_4[t]` + `512` did improve over `256`, but by a modest amount.
  - `h_1[t]` still lags and one confirm stratum goes negative, which is stronger evidence against an `h_1[t]`-centered readiness narrative.
- Confidence:
  - high that Phase 6 should remain mixed/blocked in paper language this cycle
  - high that `resattn-88i` is now the right next high-leverage move

## [2026-03-20T11:59:21-0500] Seeded Donors Removed the Easy Excuse, but Not the Mixed Read
- Stage: tool-breakage donor-control adjudication
- Feel of the Result: This is the result I wanted epistemically. We removed the obvious ordering artifact and the lane still did not become broadly positive against dynamic donors.
- Working Hypotheses:
  - The donor-arm limitation is now genuinely route-mode/family-heterogeneous rather than a bookkeeping bug.
  - Safety surface redesign (`cv9`) is likely higher value than another donor rerun.
- Surprises and Tensions:
  - `prompt_permuted_alpha` moved close to parity, but `within_family` stayed materially negative.
  - Fixed-alpha control remains strongly weaker, so the lane is not a null; it is specifically a mixed dynamic-control result.
- Confidence:
  - high that the tool-breakage donor boundary should stay mixed for this cycle
  - high that next queue value is safety surface redesign, not another broad donor-arm pass

## [2026-03-20T13:27:59-0500] OIH Is Real Now, But Static Baseline Calibration Is the Next Honesty Test
- Stage: OIH execution (`cv9` closeout + `oih` full anchor)
- Feel of the Result: Good progress, mixed confidence. `cv9` closed cleanly as a negative, which is scientifically useful. `oih` produced a real full-scale artifact quickly, but the primary pruned static baseline is so weak that it cannot carry the whole dynamic-vs-static claim by itself.
- Working Hypotheses:
  - The dynamic signal on the OIH surface is real (`predicted` and `oracle` both positive on confirm).
  - The current ShortGPT-style pruning policy is likely over-harsh for this routed-mixture objective.
  - Supplementary pilot-mean static policy is the better immediate comparator and should be explicit in interpretation.
- Surprises and Tensions:
  - Even with milder pruning (`0.10`), the pruned static baseline stayed strongly negative on all confirm prompts.
  - Pilot-mean static baseline was much healthier and close enough to dynamic predicted to be informative.
- Confidence:
  - high that `cv9` should stay bounded negative
  - medium-high that `oih` now meaningfully advances the external-anchor lane
  - medium that the current pruned static definition should remain the primary reviewer-facing static baseline without one calibration follow-up

## [2026-03-20T13:35:00-0500] Skeleton Lock Reduced Narrative Drift Risk
- Stage: writing transition (`resattn-mfp`)
- Feel of the Result: This was the right time to freeze structure. The project now has enough mixed edges that writing without an explicit skeleton would almost certainly drift into overstatement.
- Working Hypotheses:
  - Most remaining risk is wording calibration, not missing core evidence.
  - `resattn-73r` is the key final leverage point before stronger OIH language.
- Confidence:
  - high that moving into skeleton-first drafting is correct now
  - medium-high that static baseline calibration is the next technical bottleneck for external-anchor claims

## [2026-03-20T13:55:14-0500] The OIH Calibration Pass Was Boring, but It Closed a Real Credibility Gap
- Stage: OIH follow-up (`resattn-73r`)
- Feel of the Result: This was operationally tedious but scientifically important. The full artifact now says what we wanted it to say without hand-wavy caveats: calibrated static is explicit, and MIB metadata matches the control plan.
- Working Hypotheses:
  - The OIH lane is now solid enough for manuscript integration without another immediate rerun.
  - The real remaining work is synthesis and truthful claim calibration, not more static-baseline hacking.
- Surprises and Tensions:
  - The full rerun took close to 10 minutes even with cache reuse, so “just rerun quickly” is less true than it looked at first.
  - The dynamic-over-calibrated margin is positive but small (`+0.0478`), which is useful but should keep language conservative.
- Confidence:
  - high that `resattn-73r` is complete as scoped
  - medium-high that OIH now supports a bounded, reviewer-respectable dynamic-vs-static statement

## [2026-03-20T14:20:11-0500] First Prose Pass Reduced “Paper Anxiety” More Than Another Experiment Would
- Stage: manuscript drafting (`resattn-lmp`)
- Feel of the Result: This was the right next step. The draft now exists as actual prose with claim boundaries baked in, which is more useful than another incremental run at this point.
- Working Hypotheses:
  - The main remaining risk is wording calibration and figure/table clarity, not missing core evidence.
  - A bounded mixed-lane narrative is now viable without over-selling tool-breakage or safety.
- Surprises and Tensions:
  - The biggest practical drift risk was older claim tables referencing pre-v3 safety/tool-breakage artifacts; syncing those references was necessary for draft integrity.
  - OIH reads much cleaner once calibrated-static deltas are used directly instead of arguing from the weak pruned baseline.
- Confidence:
  - high that moving from skeleton to prose was the highest-leverage next move
  - medium-high that the current draft supports external review without immediate new runs

## [2026-03-20T14:20:11-0500] The Next Good Move Is Communication Fidelity, Not More Compute
- Stage: post-draft queue shaping
- Feel of the Decision: Opening a writing-refinement issue right away felt cleaner than letting the queue go empty and drifting back into run-first behavior.
- Working Hypotheses:
  - Figure/table insertion and wording calibration will increase reviewer trust more than another medium-scale run right now.
  - Any new run should now have a specific defect-trigger, not just “more signal” intuition.
- Confidence:
  - high that `resattn-ca1` is the right immediate next issue

## [2026-03-20T14:20:11-0500] Draft v2 Finally Feels Reviewer-Consumable
- Stage: manuscript refinement (`resattn-ca1`)
- Feel of the Result: The v2 pass is materially better than v1 for external readers because the key claims now sit in table-ready form with exact numbers, not just prose paragraphs.
- Working Hypotheses:
  - The next bottleneck is visual packaging and compression, not evidence discovery.
  - If reviewers push back now, it is more likely on interpretation boundaries than missing metrics.
- Confidence:
  - high that table/figure placeholder integration was worth doing immediately
  - medium-high that the narrative is now stable enough for external feedback

## [2026-03-20T14:20:11-0500] Queueing Reviewer-Packet Work Keeps Us Honest
- Stage: post-refinement planning
- Feel of the Decision: Good discipline move. Instead of letting the queue drift, we now have a concrete polish/export task.
- Working Hypotheses:
  - A compact reviewer packet will expose any remaining overclaim risk quickly.
  - If there is a hidden weakness left, it will likely be in wording consistency, not missing experiments.

## [2026-03-20T14:20:11-0500] Reviewer Packet Export Feels Like the Right Stopping Point Before Fresh Feedback
- Stage: writing packaging (`resattn-57z`)
- Feel of the Result: This pass turned the writing bundle into something a reviewer can actually consume quickly, which should produce better criticism than sending raw evolving draft files.
- Working Hypotheses:
  - The next major gains now depend on reviewer critique quality, not additional solo polishing.
  - If objections appear, they will likely target mixed-lane interpretation sharpness.

## [2026-03-20T14:20:11-0500] Waiting on External Critique Is Now the Most Rational Constraint
- Stage: queue gating
- Feel of the Decision: opening a feedback-integration issue and stopping there feels disciplined; anything else now would be guesswork.
- Working Hypotheses:
  - External review will likely reveal wording-level improvements we cannot simulate internally.
