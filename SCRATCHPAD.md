# Scratchpad

Use this file for execution checkpoints and transient notes. Every substantial local run should have a pre-run and post-run entry.

## Pre-Run Template

```text
## [TIMESTAMP] PRE-RUN: [run name]
- Command: `.venv/bin/python ...`
- Device: `mps` / `cpu`
- Model: [exact model id]
- Data slice: [dataset / stratum / number of sequences]
- Output path: `results/...`
- What I'm testing: [one sentence]
- Expected outcome: [one sentence]
- Implementation verified: YES/NO - [what local check was run]
- Status: LAUNCHING
```

## Post-Run Template

```text
## [TIMESTAMP] POST-RUN: [run name]
- Command: `.venv/bin/python ...`
- Outcome: SUCCESS / FAILURE / PARTIAL
- Key metric: [main number]
- Artifacts saved: `results/...`
- Anomalies: [unexpected behavior or `none`]
- Next step: [single next action]
```

## Bootstrap Notes

- No experimental runs have been executed from this scaffold yet.
- Local execution only; no Modal app IDs, remote volumes, or cloud runner assumptions belong here.

## [2026-03-16T21:15:04-0500] PRE-RUN: development-model reconstruction smoke
- tmux session: N/A
- Script: `scripts/phase1_model_backed_reconstruction_smoke.py`
- Command: `.venv/bin/python scripts/phase1_model_backed_reconstruction_smoke.py --output results/infrastructure/20260316-gpt2xl-reconstruction-smoke.json`
- Config: `model=gpt2-xl`, `prompt="The capital of France is"`, `device=mps`, `fallback=cpu`
- What I'm testing: the configured development model can reconstruct its original logits from cached embedding and sublayer writes after applying the model's own final normalization and unembedding.
- Expected outcome: max absolute error stays near floating-point tolerance and per-layer residual identities hold.
- Expected duration: ~5-15 minutes including first download
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/infrastructure/20260316-gpt2xl-reconstruction-smoke.json`
- Resume command: rerun the command above
- Main confound to watch: first-run download or MPS execution quirks could mask a genuine reconstruction bug.
- Implementation verified: YES - tiny-model TransformerLens-backed unit tests are green before this run.
- Status: LAUNCHING

## [2026-03-16T21:18:40-0500] POST-RUN: development-model reconstruction smoke
- Command: `.venv/bin/python scripts/phase1_model_backed_reconstruction_smoke.py --output results/infrastructure/20260316-gpt2xl-reconstruction-smoke.json`
- Outcome: SUCCESS
- Key metric: `final_residual_max_abs_error=0.0`, `uniform_logits_max_abs_error=0.0`
- Artifacts saved: `results/infrastructure/20260316-gpt2xl-reconstruction-smoke.json`
- Anomalies: an initial MPS residual mismatch disappeared after changing reconstruction accumulation from bulk reduction to forward-order summation
- Next step: treat model-backed reconstruction as green and move the remaining claim-bearing blockers back to the pilot/confirmatory split plus identifiability controls

## [2026-03-16T23:05:00-0500] PRE-RUN: development-model oracle-alpha slice
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_development_slice.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_development_slice.py --output results/oracle_alpha/20260316-gpt2xl-development-slice.json --max-sequences 2 --optimization-steps 20 --learning-rate 0.1 --seed 11`
- Config: `model=gpt2-xl`, `collection=oracle_alpha_phase1_v1`, `split=pilot`, `exploratory=true`, `device=mps fallback cpu`
- What I'm testing: the first development-model oracle-alpha runner can consume the saved prompt and control registries, optimize a per-sequence alpha vector on fixed residual sources, and emit a real JSON artifact.
- Expected outcome: the run completes on a tiny pilot slice, optimized loss is never worse than uniform, and the artifact records sequence-level improvements plus preregistered null summaries.
- Expected duration: ~5-15 minutes
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/oracle_alpha/20260316-gpt2xl-development-slice.json`
- Resume command: rerun the command above
- Main confound to watch: MPS execution or model-loading quirks could dominate the first runner slice before the actual alpha optimization logic is exercised.
- Implementation verified: YES - `tests/test_oracle_alpha_runner.py` passes on `tiny-stories-1M` before this run.
- Status: LAUNCHING

## [2026-03-16T23:08:00-0500] POST-RUN: development-model oracle-alpha slice
- Command: `.venv/bin/python scripts/run_oracle_alpha_development_slice.py --output results/oracle_alpha/20260316-gpt2xl-development-slice.json --max-sequences 2 --optimization-steps 20 --learning-rate 0.1 --seed 11`
- Outcome: SUCCESS
- Key metric: `sequence_mean_improvement=1.2141` nats over uniform on `2` pilot prompts
- Artifacts saved: `results/oracle_alpha/20260316-gpt2xl-development-slice.json`
- Anomalies: none
- Next step: treat the runner as live and promote the next follow-up to scaling the pilot slice plus stability perturbations beyond this two-prompt smoke

## [2026-03-16T23:20:00-0500] PRE-RUN: development-model oracle-alpha pilot stability suite
- Command: `.venv/bin/python scripts/run_oracle_alpha_pilot_stability_suite.py --output results/oracle_alpha/20260316-gpt2xl-pilot-stability-suite.json --max-sequences 8 --optimization-steps 20 --learning-rate 0.1 --resample-count 3 --resample-size 6`
- Device: `mps`
- Model: `gpt2-xl`
- Data slice: `oracle_alpha_phase1_v1` pilot split, `8` prompts, saved paraphrases, restart seeds from `configs/oracle_alpha_controls_v1.yaml`
- Output path: `results/oracle_alpha/20260316-gpt2xl-pilot-stability-suite.json`
- What I'm testing: the scaled development-model runner can survive the preregistered restart and prompt-perturbation checks on the full pilot prompt set.
- Expected outcome: the run completes, mean pilot improvement stays positive, and restart/paraphrase/resample stability metrics are inspectable from one JSON artifact.
- Implementation verified: YES - unit tests cover prompt perturbations and the stability-suite runner on `tiny-stories-1M`.
- Status: LAUNCHING

## [2026-03-16T23:29:00-0500] PRE-RUN: development-model oracle-alpha pilot stability suite rerun
- Command: `.venv/bin/python scripts/run_oracle_alpha_pilot_stability_suite.py --output results/oracle_alpha/20260316-gpt2xl-pilot-stability-suite.json --max-sequences 8 --optimization-steps 20 --learning-rate 0.1 --resample-count 3 --resample-size 6`
- Device: `mps`
- Model: `gpt2-xl`
- Data slice: `oracle_alpha_phase1_v1` pilot split, `8` prompts, saved paraphrases, restart seeds from `configs/oracle_alpha_controls_v1.yaml`
- Output path: `results/oracle_alpha/20260316-gpt2xl-pilot-stability-suite.json`
- What I'm testing: rerun the scaled pilot suite after fixing a deterministic restart bug so restart seeds actually perturb the optimization path.
- Expected outcome: restart metrics are no longer trivially identical across seeds, while mean pilot improvement remains positive.
- Implementation verified: YES - `tests/test_oracle_alpha_runner.py` now checks that different restart seeds change final alpha outputs.
- Status: LAUNCHING

## [2026-03-16T23:45:00-0500] PRE-RUN: development-model oracle-alpha pilot stability suite paraphrase-fix rerun
- Command: `.venv/bin/python scripts/run_oracle_alpha_pilot_stability_suite.py --output results/oracle_alpha/20260316-gpt2xl-pilot-stability-suite.json --max-sequences 8 --optimization-steps 20 --learning-rate 0.1 --resample-count 3 --resample-size 6`
- Device: `mps`
- Model: `gpt2-xl`
- Data slice: `oracle_alpha_phase1_v1` pilot split, `8` prompts, saved paraphrases, restart seeds from `configs/oracle_alpha_controls_v1.yaml`
- Output path: `results/oracle_alpha/20260316-gpt2xl-pilot-stability-suite.json`
- What I'm testing: rerun the stability suite after fixing a malformed saved paraphrase so the prompt-perturbation check is a genuine paraphrase comparison rather than a broken prompt control.
- Expected outcome: the pilot improvement remains positive and the paraphrase metrics reflect a real wording perturbation rather than prompt corruption.
- Implementation verified: YES - `tests/test_prompt_registry.py` enforces saved paraphrases for the Phase 1 pilot collection and the stability-suite tests are green before this rerun.
- Status: LAUNCHING

## [2026-03-16T23:52:00-0500] POST-RUN: development-model oracle-alpha pilot stability suite paraphrase-fix rerun
- Command: `.venv/bin/python scripts/run_oracle_alpha_pilot_stability_suite.py --output results/oracle_alpha/20260316-gpt2xl-pilot-stability-suite.json --max-sequences 8 --optimization-steps 20 --learning-rate 0.1 --resample-count 3 --resample-size 6`
- Outcome: SUCCESS
- Key metric: `sequence_mean_improvement=1.2432` nats; `restart_js=2.87e-07`; `paraphrase_js=0.0305`; `resample_js=0.0191`
- Artifacts saved: `results/oracle_alpha/20260316-gpt2xl-pilot-stability-suite.json`
- Latest checkpoint: none
- Anomalies: the earlier restart artifact was invalid because zero-logit initialization made restarts deterministic, and one saved paraphrase had to be fixed before the perturbation comparison was interpretable
- Next step: execute the held-out alpha predictiveness check on the confirm split in `resattn-53q`

## [2026-03-17T00:20:00-0500] PRE-RUN: development-model oracle-alpha held-out predictiveness check
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_heldout_predictiveness_check.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260316-gpt2xl-heldout-predictiveness-check.json --optimization-steps 20 --learning-rate 0.1 --seed 11`
- Config: `model=gpt2-xl`, `collection=oracle_alpha_phase1_v1`, `train_split=pilot`, `eval_split=confirm`, `feature_source=mean_pooled_h_1[t]`, `model_family=ridge_regression`, `device=mps fallback cpu`
- What I'm testing: whether sequence-level oracle-alpha vectors on the confirm split are predictable from mean-pooled early hidden-state summaries learned on the pilot split, and whether the predicted alpha vectors retain any confirm-split loss advantage over uniform.
- Expected outcome: the run completes on the saved pilot/confirm split, writes a held-out predictiveness artifact, and makes the current runner-stage MIB omission explicit rather than silent.
- Expected duration: ~10-20 minutes
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/oracle_alpha/20260316-gpt2xl-heldout-predictiveness-check.json`
- Resume command: rerun the command above
- Main confound to watch: with only `8` pilot and `8` confirm prompts, the ridge fit may look unstable or weak even if the underlying signal is real.
- Implementation verified: YES - targeted unit tests for the ridge helper and predictiveness runner pass on `tiny-stories-1M`.
- Status: LAUNCHING

## [2026-03-17T00:35:00-0500] POST-RUN: development-model oracle-alpha held-out predictiveness check
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260316-gpt2xl-heldout-predictiveness-check.json --optimization-steps 20 --learning-rate 0.1 --seed 11`
- Outcome: SUCCESS
- Key metric: `confirm_r_squared=-0.2456`; `confirm_mean_js=0.2434`; `predicted_mean_improvement=-0.0348` nats versus `oracle_mean_improvement=1.3409`
- Artifacts saved: `results/oracle_alpha/20260316-gpt2xl-heldout-predictiveness-check.json`
- Latest checkpoint: none
- Anomalies: the selected ridge strength saturated at the largest tested value (`100.0`), which is another sign that the current mean-pooled `h_1[t]` feature summary is too weak or too coarse for held-out alpha recovery at this prompt scale
- Next step: treat the current predictiveness check as failed for this feature spec, create a follow-up to test richer feature summaries on the pilot split, and keep strong oracle-alpha interpretation blocked until the confirm check improves

## [2026-03-17T00:52:00-0500] PRE-RUN: development-model oracle-alpha feature-summary comparison
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_heldout_predictiveness_check.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-feature-comparison.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources mean_pooled_h_1[t]_resid_post_layer_0 mean_pooled_h_4[t]_resid_post_layer_3 mean_pooled_h_1[t]_plus_h_4[t]_concat final_token_h_1[t]_plus_h_4[t]_concat`
- Config: `model=gpt2-xl`, `collection=oracle_alpha_phase1_v1`, `train_split=pilot`, `eval_split=confirm`, `selection_metric=pilot_leave_one_out_mean_js`, `device=mps fallback cpu`
- What I'm testing: whether a small richer set of sequence-level early-state summaries improves the held-out alpha predictiveness check without changing the saved prompt registry or touching the confirm split during feature selection.
- Expected outcome: one candidate feature summary beats the current mean-pooled `h_1[t]` baseline on pilot LOO and either improves or at least clarifies the confirm-split predictiveness failure.
- Expected duration: ~10-20 minutes
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-feature-comparison.json`
- Resume command: rerun the command above
- Main confound to watch: the pilot set is only `8` prompts, so even pilot-only feature selection can still be unstable enough to overfit weakly.
- Implementation verified: YES - targeted unit tests for candidate feature-source selection pass on `tiny-stories-1M`.
- Status: LAUNCHING

## [2026-03-17T01:08:00-0500] POST-RUN: development-model oracle-alpha feature-summary comparison
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-feature-comparison.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources 'mean_pooled_h_1[t]_resid_post_layer_0' 'mean_pooled_h_4[t]_resid_post_layer_3' 'mean_pooled_h_1[t]_plus_h_4[t]_concat' 'final_token_h_1[t]_plus_h_4[t]_concat'`
- Outcome: PARTIAL
- Key metric: selected feature source `mean_pooled_h_4[t]_resid_post_layer_3`; `confirm_r_squared=-0.2314`; `confirm_mean_js=0.2409`; `predicted_mean_improvement=-0.0010` nats
- Artifacts saved: `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-feature-comparison.json`
- Latest checkpoint: none
- Anomalies: the first launch failed before execution because unquoted feature-source names hit shell globbing; the rerun fixed that and all candidate feature summaries still selected the strongest ridge penalty (`100.0`)
- Next step: treat `h_4[t]` mean pooling as the best tested internal-state summary so far, but keep the blocker open and move to token-aware or prompt-level feature surfaces in `resattn-7ve`

## [2026-03-16T22:57:18-0500] PRE-RUN: development-model oracle-alpha token-aware feature comparison
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_heldout_predictiveness_check.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-token-aware-comparison.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources 'mean_pooled_h_1[t]_resid_post_layer_0' 'mean_pooled_h_4[t]_resid_post_layer_3' 'position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat' 'start_mid_end_h_4[t]_resid_post_layer_3_concat' 'mean_pooled_h_1[t]_plus_h_4[t]_concat' 'final_token_h_1[t]_plus_h_4[t]_concat'`
- Config: `model=gpt2-xl`, `collection=oracle_alpha_phase1_v1`, `train_split=pilot`, `eval_split=confirm`, `selection_metric=pilot_leave_one_out_mean_js`, `device=mps fallback cpu`
- What I'm testing: whether minimal token-aware `h_4[t]` summaries beat the current mean-pooled baseline on pilot-only feature selection and improve the held-out confirm predictiveness check without touching the prompt registry.
- Expected outcome: one token-aware candidate wins pilot LOO tuning and moves the confirm-split predictor meaningfully above the near-neutral `h_4[t]` mean-pooled result.
- Expected duration: ~10-20 minutes
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-token-aware-comparison.json`
- Resume command: rerun the command above
- Main confound to watch: the token-aware summaries increase feature dimension on only `8` pilot prompts, so they could improve pilot JS while still washing out on confirm under strong ridge shrinkage.
- Implementation verified: YES - `tests/test_oracle_alpha_runner.py` passes with the new token-aware feature sources before this run.
- Status: LAUNCHING

## [2026-03-16T22:59:08-0500] POST-RUN: development-model oracle-alpha token-aware feature comparison
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-token-aware-comparison.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources 'mean_pooled_h_1[t]_resid_post_layer_0' 'mean_pooled_h_4[t]_resid_post_layer_3' 'position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat' 'start_mid_end_h_4[t]_resid_post_layer_3_concat' 'mean_pooled_h_1[t]_plus_h_4[t]_concat' 'final_token_h_1[t]_plus_h_4[t]_concat'`
- Outcome: PARTIAL
- Key metric: selected feature source `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`; `confirm_r_squared=-0.2154`; `confirm_mean_js=0.2377`; `predicted_mean_improvement=-0.0011` nats
- Artifacts saved: `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-token-aware-comparison.json`
- Latest checkpoint: none
- Anomalies: token-aware features improved alpha-shape metrics but not held-out routed-loss improvement, and the selected ridge penalty again saturated at `100.0`
- Next step: keep the blocker open and move to prompt-level or hybrid feature surfaces in `resattn-27f`
