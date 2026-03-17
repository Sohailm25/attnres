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
