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

## [2026-03-16T23:10:43-0500] PRE-RUN: development-model oracle-alpha prompt-level and hybrid feature comparison
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_heldout_predictiveness_check.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-prompt-hybrid-comparison.json --optimization-steps 20 --learning-rate 0.1 --seed 11`
- Config: `model=gpt2-xl`, `collection=oracle_alpha_phase1_v1`, `train_split=pilot`, `eval_split=confirm`, `selection_metric=pilot_leave_one_out_mean_js`, `candidate_set=position_thirds_h4 + prompt baselines/hybrids`, `device=mps fallback cpu`
- What I'm testing: whether prompt-level baselines or hybrids that append prompt-shape or mean token-embedding features to the best token-aware `h_4[t]` summary can finally turn the confirm-split predicted routed-loss metric positive.
- Expected outcome: either a hybrid feature surface wins pilot selection and moves predicted mean improvement above uniform, or the run makes it clear that the current sequence-level target is still not recoverable with this simple feature family.
- Expected duration: ~10-20 minutes
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-prompt-hybrid-comparison.json`
- Resume command: rerun the command above
- Main confound to watch: the hybrid candidates raise feature dimension again on only `8` pilot prompts, so they may improve alpha similarity while remaining too over-regularized to recover routed-loss advantage.
- Implementation verified: YES - `tests/test_oracle_alpha_runner.py` passes with the new prompt-level and hybrid feature sources before this run.
- Status: LAUNCHING

## [2026-03-16T23:13:05-0500] POST-RUN: development-model oracle-alpha prompt-level and hybrid feature comparison
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-prompt-hybrid-comparison.json --optimization-steps 20 --learning-rate 0.1 --seed 11`
- Outcome: FAILURE

## [2026-03-17T14:50:52-0500] PRE-RUN: small local Block AttnRes viability smoke
- tmux session: `attnres-proxy-smoke-v1`
- Script: `scripts/run_attnres_proxy_viability.py`
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 256 --max-eval-texts 64 --vocab-size 256 --d-model 96 --n-heads 4 --n-layers 8 --d-ff 384 --max-seq-len 64 --batch-size 8 --num-steps 200 --checkpoint-every-steps 25 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-viability-smoke-v1`
- Config: `dataset=wikitext-2-raw-v1`, `tokenizer=character`, `proxy=8-block`, `baseline=matched standard residual`, `device=mps fallback cpu via script`
- What I'm testing: the smallest honest local AttnRes proxy can train stably on local hardware, emit resumable checkpoints, and export routing summaries rich enough to score later Figure 8 metrics.
- Expected outcome: both tiny models finish training without NaNs, the AttnRes proxy is not catastrophically worse than the matched baseline, and `summary.json` contains per-target locality / entropy / embedding / skip summaries.
- Expected duration: ~15-30 minutes
- Checkpoint path: `results/figure8_validation/20260317-attnres-proxy-viability-smoke-v1/checkpoints/`
- Checkpoint cadence: every `25` steps
- Log path: `results/figure8_validation/20260317-attnres-proxy-viability-smoke-v1/run.log`
- Resume command: rerun the command above inside tmux
- Main confound to watch: a char-level tiny LM can train stably yet still be too weak to make routing summaries informative, so loss stability matters but is not the only success condition.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_attnres_reproduction`
- Status: LAUNCHING

## [2026-03-17T14:53:49-0500] POST-RUN: small local Block AttnRes viability smoke
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 256 --max-eval-texts 64 --vocab-size 256 --d-model 96 --n-heads 4 --n-layers 8 --d-ff 384 --max-seq-len 64 --batch-size 8 --num-steps 200 --checkpoint-every-steps 25 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-viability-smoke-v1`
- Outcome: SUCCESS
- Key metric: `attnres_best_eval_loss=2.7130` versus `baseline_best_eval_loss=2.7338` (`delta=-0.0207`)
- Artifacts saved: `results/figure8_validation/20260317-attnres-proxy-viability-smoke-v1/`
- Latest checkpoint: `results/figure8_validation/20260317-attnres-proxy-viability-smoke-v1/checkpoints/attnres_training_state.pt`
- Anomalies: the initial Figure 8 proxy read is mixed rather than paper-like: `deep_embedding_persistence=0.1824` and `mean_pre_attn_entropy < mean_pre_mlp_entropy`, so this run validates the pipeline more than the final pattern claim.
- Next step: scale the exact same 8-block setup on more text and more optimization steps before judging whether the local proxy has real Figure 8-like structure.

## [2026-03-17T14:53:49-0500] PRE-RUN: scaled local Block AttnRes viability run
- tmux session: `attnres-proxy-scale-v1`
- Script: `scripts/run_attnres_proxy_viability.py`
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 256 --vocab-size 256 --d-model 96 --n-heads 4 --n-layers 8 --d-ff 384 --max-seq-len 64 --batch-size 16 --num-steps 1500 --checkpoint-every-steps 100 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-viability-scale-v1`
- Config: `dataset=wikitext-2-raw-v1`, `tokenizer=character`, `proxy=8-block`, `architecture frozen from smoke`, `scale-up=data+steps only`
- What I'm testing: whether the same 8-block local proxy develops clearer routing structure and a more stable loss read when trained longer on a larger text surface.
- Expected outcome: the AttnRes proxy remains at least competitive with the matched baseline and the saved Figure 8 summaries become less obviously undertrained than the smoke artifact.
- Expected duration: ~20-60 minutes
- Checkpoint path: `results/figure8_validation/20260317-attnres-proxy-viability-scale-v1/checkpoints/`
- Checkpoint cadence: every `100` steps
- Log path: `results/figure8_validation/20260317-attnres-proxy-viability-scale-v1/run.log`
- Resume command: rerun the command above inside tmux
- Main confound to watch: if the char-level proxy still shows weak or inverted Figure 8 metrics after a materially longer run, the issue may be the proxy/data choice rather than simple undertraining.
- Implementation verified: YES - smoke run completed, wrote checkpoints, and the exact resume command reused them successfully in `7.642` seconds.
- Status: LAUNCHING

## [2026-03-17T14:54:57-0500] POST-RUN: scaled local Block AttnRes viability run
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 256 --vocab-size 256 --d-model 96 --n-heads 4 --n-layers 8 --d-ff 384 --max-seq-len 64 --batch-size 16 --num-steps 1500 --checkpoint-every-steps 100 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-viability-scale-v1`
- Outcome: FAILURE
- Key metric: `character_vocabulary_size=269 > vocab_size=256`
- Artifacts saved: `results/figure8_validation/20260317-attnres-proxy-viability-scale-v1/run.log`
- Latest checkpoint: none
- Anomalies: scaling the Wikitext slice increased the observed character vocabulary beyond the smoke-safe default, so the larger run failed before training rather than revealing a model issue.
- Next step: relaunch with `vocab_size=512` and keep the rest of the scale-up fixed so the only substantive change remains data+optimization scale.

## [2026-03-17T14:54:57-0500] PRE-RUN: scaled local Block AttnRes viability run relaunch
- tmux session: `attnres-proxy-scale-v1`
- Script: `scripts/run_attnres_proxy_viability.py`
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 256 --vocab-size 512 --d-model 96 --n-heads 4 --n-layers 8 --d-ff 384 --max-seq-len 64 --batch-size 16 --num-steps 1500 --checkpoint-every-steps 100 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-viability-scale-v1`
- Config: `dataset=wikitext-2-raw-v1`, `tokenizer=character`, `proxy=8-block`, `architecture frozen from smoke`, `scale-up=data+steps only`, `vocab_size widened to fit corpus`
- What I'm testing: the same larger run as above, now with enough character vocabulary capacity to cover the real corpus slice.
- Expected outcome: the run reaches checkpoints and completes so the pattern read is about training, not an avoidable tokenizer-capacity failure.
- Expected duration: ~20-60 minutes
- Checkpoint path: `results/figure8_validation/20260317-attnres-proxy-viability-scale-v1/checkpoints/`
- Checkpoint cadence: every `100` steps
- Log path: `results/figure8_validation/20260317-attnres-proxy-viability-scale-v1/run.log`
- Resume command: rerun the command above inside tmux
- Main confound to watch: if the pattern surface stays weak after this relaunch, the bottleneck is likely the proxy/data regime rather than a simple configuration mistake.
- Implementation verified: YES - the failure was pre-training and isolated to vocabulary coverage; no model code changes were needed beyond a safer default.
- Status: LAUNCHING

## [2026-03-17T15:04:32-0500] POST-RUN: scaled local Block AttnRes viability run relaunch
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 256 --vocab-size 512 --d-model 96 --n-heads 4 --n-layers 8 --d-ff 384 --max-seq-len 64 --batch-size 16 --num-steps 1500 --checkpoint-every-steps 100 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-viability-scale-v1`
- Outcome: SUCCESS
- Key metric: `attnres_best_eval_loss=2.1130` versus `baseline_best_eval_loss=2.1458` (`delta=-0.0327`)
- Artifacts saved: `results/figure8_validation/20260317-attnres-proxy-viability-scale-v1/`, `results/figure8_validation/20260317-attnres-proxy-viability-scale-v1.md`
- Latest checkpoint: `results/figure8_validation/20260317-attnres-proxy-viability-scale-v1/checkpoints/attnres_training_state.pt`
- Anomalies: the routed pattern surface is still mixed rather than paper-like even after the stronger run; `deep_embedding_persistence` fell to `0.1049` and `mean_pre_attn_entropy` remained below `mean_pre_mlp_entropy`
- Next step: treat the local AttnRes proxy as operationally viable, close the build issue, and move to a bounded proxy-regime decision (`resattn-3l6`) before any Figure 8 alignment claim

## [2026-03-17T15:14:08-0500] PRE-RUN: compact-subword local Block AttnRes viability smoke
- tmux session: `attnres-proxy-subword-v1`
- Script: `scripts/run_attnres_proxy_viability.py`
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 256 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 96 --n-heads 4 --n-layers 8 --d-ff 384 --max-seq-len 64 --batch-size 16 --num-steps 200 --checkpoint-every-steps 50 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-v1`
- Config: `dataset=wikitext-2-raw-v1`, `tokenizer=compact GPT-2 subword remap`, `proxy=8-block`, `baseline=matched standard residual`, `device=mps fallback cpu via script`
- What I'm testing: whether replacing the char-level regime with a compact GPT-2 subword regime on the same Wikitext slice gives a cleaner Figure 8 proxy signal without changing the tiny 8-block architecture.
- Expected outcome: both models train stably, checkpoints are written, and the first 200-step summary is good enough to justify resuming the same run to a longer horizon.
- Expected duration: ~10-25 minutes
- Checkpoint path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-v1/checkpoints/`
- Checkpoint cadence: every `50` steps
- Log path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-v1/run.log`
- Resume command: rerun the command above with `--num-steps 1500`
- Main confound to watch: if the compact vocabulary materially slows training but leaves the pattern metrics unchanged, the next bottleneck is likely corpus or capacity rather than tokenization.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_attnres_reproduction`
- Status: LAUNCHING

## [2026-03-17T15:15:27-0500] POST-RUN: compact-subword local Block AttnRes viability smoke
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 256 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 96 --n-heads 4 --n-layers 8 --d-ff 384 --max-seq-len 64 --batch-size 16 --num-steps 200 --checkpoint-every-steps 50 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-v1`
- Outcome: SUCCESS
- Key metric: `attnres_best_eval_loss=7.0459` versus `baseline_best_eval_loss=7.0926` (`delta=-0.0467`)
- Artifacts saved: `results/figure8_validation/20260317-attnres-proxy-compact-subword-v1/summary.json`, `results/figure8_validation/20260317-attnres-proxy-compact-subword-v1/tokenizer_manifest.json`
- Latest checkpoint: `results/figure8_validation/20260317-attnres-proxy-compact-subword-v1/checkpoints/attnres_training_state.pt`
- Anomalies: the entropy ordering is still inverted, but deep embedding persistence improved relative to the char-level scaled run (`0.1378` versus `0.1049`)
- Next step: resume the same output directory to `1500` steps and see whether the compact-subword regime keeps its stronger loss gap and improves the proxy metrics further

## [2026-03-17T15:15:27-0500] PRE-RUN: compact-subword local Block AttnRes scale continuation
- tmux session: `attnres-proxy-subword-v1`
- Script: `scripts/run_attnres_proxy_viability.py`
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 256 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 96 --n-heads 4 --n-layers 8 --d-ff 384 --max-seq-len 64 --batch-size 16 --num-steps 1500 --checkpoint-every-steps 100 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-v1`
- Config: `same corpus`, `same tokenizer`, `same architecture`, `resume from 200-step checkpoints`
- What I'm testing: whether the compact-subword regime keeps improving and produces a more credible Figure 8 proxy read when allowed to train to the same horizon as the earlier char-level scale run.
- Expected outcome: the AttnRes proxy remains at least competitive with the baseline and the saved Figure 8 proxy metrics improve enough to clarify whether tokenizer realism was the main missing ingredient.
- Expected duration: ~20-60 minutes
- Checkpoint path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-v1/checkpoints/`
- Checkpoint cadence: every `100` steps
- Log path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-v1/run.log`
- Resume command: rerun the command above
- Main confound to watch: if deeper training keeps the loss gap but the pattern surface remains mixed, the next bottleneck is likely corpus or capacity rather than tokenization.
- Implementation verified: YES - the 200-step compact-subword run completed and wrote resumable checkpoints plus the expected manifest/summary files
- Status: LAUNCHING

## [2026-03-17T15:40:32-0500] POST-RUN: compact-subword local Block AttnRes scale continuation
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 256 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 96 --n-heads 4 --n-layers 8 --d-ff 384 --max-seq-len 64 --batch-size 16 --num-steps 1500 --checkpoint-every-steps 100 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-v1`
- Outcome: SUCCESS
- Key metric: `attnres_best_eval_loss=6.6764` versus `baseline_best_eval_loss=6.6463` (`delta=+0.0300`)
- Artifacts saved: `results/figure8_validation/20260317-attnres-proxy-compact-subword-v1/`, `results/figure8_validation/20260317-attnres-proxy-compact-subword-v1.md`
- Latest checkpoint: `results/figure8_validation/20260317-attnres-proxy-compact-subword-v1/checkpoints/attnres_training_state.pt`
- Anomalies: tokenization realism helped the Figure-facing metrics but not the baseline comparison; `deep_embedding_persistence` improved to `0.1364` and the entropy gap improved to `-0.0349`, but the Block AttnRes proxy no longer beat the matched baseline at the longer horizon
- Next step: keep compact subword as the preferred proxy regime, close the tokenizer-vs-char decision, and move the next Figure 8 follow-up to capacity or optimization scaling (`resattn-111`)

## [2026-03-17T12:08:00-0500] PRE-RUN: prereg-scale oracle-alpha pattern analysis
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_pattern_analysis.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_pattern_analysis.py --run-path results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/oracle_eval_run.json --output results/pattern_analysis/20260317-gpt2xl-prereg-scale-pattern-analysis-v1.json --random-seed 11 --max-clusters 12`
- Config: `model=gpt2-xl`, `artifact=registry_v4 confirm oracle_eval_run`, `distance=Jensen-Shannon`, `linkage=average`, `random_control=symmetric_dirichlet_matched_shape`
- What I'm testing: whether the saved prereg-scale confirm alphas show sequence-level routing structure above a matched random control without overstating the development-model result.
- Expected outcome: weak-but-nonzero sequence-level structure, descriptive source-type mass summaries, and a reusable compact artifact for the next pattern-analysis decision.
- Expected duration: ~1 minute
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/pattern_analysis/20260317-gpt2xl-prereg-scale-pattern-analysis-v1.json`
- Resume command: rerun the command above
- Main confound to watch: final-alpha sequence summaries can show structure even when stronger block-structure or Figure 8 claims are still unwarranted.
- Implementation verified: YES - `tests.test_pattern_analysis` and `tests.test_scaffold` are green before launch.
- Status: LAUNCHING

## [2026-03-17T12:09:00-0500] POST-RUN: prereg-scale oracle-alpha pattern analysis
- Command: `.venv/bin/python scripts/run_oracle_alpha_pattern_analysis.py --run-path results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/oracle_eval_run.json --output results/pattern_analysis/20260317-gpt2xl-prereg-scale-pattern-analysis-v1.json --random-seed 11 --max-clusters 12`
- Outcome: PARTIAL

## [2026-03-17T12:45:32-0500] PRE-RUN: Gemma-2 tuned-lens viability calibration
- tmux session: N/A
- Script: `scripts/run_tuned_lens_viability_pilot.py`
- Command: `.venv/bin/python scripts/run_tuned_lens_viability_pilot.py --model-name google/gemma-2-2b --device mps --output-dir results/tool_breakage/20260317-gemma2-tuned-lens-viability-calibration --max-train-prompts 2 --max-eval-prompts 1 --translator-rank 16 --num-steps 50 --checkpoint-every-steps 10 --learning-rate 0.01 --seed 11`
- Config: `train_collection=oracle_alpha_phase1_v1`, `train_split=pilot`, `eval_collection=tool_breakage_factual_recall_v1`, `eval_split=pilot`, `checkpoint_root=results/tool_breakage/20260317-gemma2-tuned-lens-viability-calibration/checkpoints`
- What I'm testing: the real Gemma-2 viability path loads locally, writes prompt caches and intermediate training checkpoints, and can complete a tiny original-model-only tuned-lens slice without code-path surprises.
- Expected outcome: the calibration saves train/eval prompt caches, writes `training_state.pt`, and emits a held-out summary artifact for a tiny prompt slice.
- Expected duration: ~10-20 minutes including model load
- Checkpoint path: `results/tool_breakage/20260317-gemma2-tuned-lens-viability-calibration/checkpoints/training_state.pt`
- Checkpoint cadence: every `10` optimization steps plus final save
- Log path: `results/tool_breakage/20260317-gemma2-tuned-lens-viability-calibration`
- Resume command: `.venv/bin/python scripts/run_tuned_lens_viability_pilot.py --model-name google/gemma-2-2b --device mps --output-dir results/tool_breakage/20260317-gemma2-tuned-lens-viability-calibration --max-train-prompts 2 --max-eval-prompts 1 --translator-rank 16 --num-steps 50 --checkpoint-every-steps 10 --learning-rate 0.01 --seed 11`
- Main confound to watch: Gemma-2 weight availability or MPS execution quirks could fail before the tuned-lens logic is actually exercised.
- Implementation verified: YES - `.venv/bin/python -m unittest discover -s tests -p 'test*.py'` is green after adding cache reuse and training resume coverage.
- Status: LAUNCHING

## [2026-03-17T12:48:12-0500] POST-RUN: Gemma-2 tuned-lens viability calibration
- Command: `.venv/bin/python scripts/run_tuned_lens_viability_pilot.py --model-name google/gemma-2-2b --device mps --output-dir results/tool_breakage/20260317-gemma2-tuned-lens-viability-calibration --max-train-prompts 2 --max-eval-prompts 1 --translator-rank 16 --num-steps 50 --checkpoint-every-steps 10 --learning-rate 0.01 --seed 11`
- Outcome: SUCCESS
- Key metric: `mean_raw_kl_to_final=9.8754 -> mean_tuned_kl_to_final=1.9276`; `final_position_mean_raw_kl_to_final=5.8291 -> 1.7539`; `mean_top1=0.1923 -> 0.5449`
- Artifacts saved: `results/tool_breakage/20260317-gemma2-tuned-lens-viability-calibration`
- Latest checkpoint: `results/tool_breakage/20260317-gemma2-tuned-lens-viability-calibration/checkpoints/training_state.pt`
- Anomalies: the first calibration attempt failed in evaluation because CPU residuals were passed into an MPS unembed path; fixing `_apply_final_norm_and_unembed` to move residuals onto the model device made the rerun resume cleanly from the saved prompt caches and training checkpoint
- Next step: launch the full original-model-only Gemma-2 viability pilot in tmux on the complete `96`-prompt train slice and `8`-prompt factual-recall eval slice

## [2026-03-17T12:49:17-0500] PRE-RUN: Gemma-2 tuned-lens viability pilot v1
- tmux session: `gemma2-tuned-lens-v1`
- Script: `scripts/run_tuned_lens_viability_pilot.py`
- Command: `.venv/bin/python scripts/run_tuned_lens_viability_pilot.py --model-name google/gemma-2-2b --device mps --output-dir results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1 --translator-rank 16 --num-steps 200 --checkpoint-every-steps 25 --learning-rate 0.01 --seed 11`
- Config: `train_collection=oracle_alpha_phase1_v1`, `train_split=pilot`, `eval_collection=tool_breakage_factual_recall_v1`, `eval_split=pilot`, `checkpoint_root=results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoints`
- What I'm testing: the smallest full-surface Gemma-2 tuned-lens viability artifact can beat raw logit lens on held-out factual-recall pilot prompts while saving reusable original-model caches and a resumable training state.
- Expected outcome: prompt caches and training checkpoints are written during the run, the held-out tuned-lens metrics beat the raw baseline on the `8` factual-recall pilot prompts, and the output is strong enough to justify the later routed-vs-original tool-breakage lane.
- Expected duration: ~30-60 minutes
- Checkpoint path: `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoints/training_state.pt`
- Checkpoint cadence: every `25` optimization steps plus final save
- Log path: `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/run.log`
- Resume command: `.venv/bin/python scripts/run_tuned_lens_viability_pilot.py --model-name google/gemma-2-2b --device mps --output-dir results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1 --translator-rank 16 --num-steps 200 --checkpoint-every-steps 25 --learning-rate 0.01 --seed 11`
- Main confound to watch: residual-MSE training could look good on held-out KL while still failing to deliver a meaningful tuned-vs-raw gap on the factual-recall final-position metrics that matter for later tool-breakage use.
- Implementation verified: YES - the full test suite is green and the Gemma calibration already verified prompt-cache writes, checkpoint resume, and held-out metric improvement on a tiny real-model slice.
- Status: LAUNCHING

## [2026-03-17T12:52:30-0500] POST-RUN: Gemma-2 tuned-lens viability pilot v1
- Command: `.venv/bin/python scripts/run_tuned_lens_viability_pilot.py --model-name google/gemma-2-2b --device mps --output-dir results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1 --translator-rank 16 --num-steps 200 --checkpoint-every-steps 25 --learning-rate 0.01 --seed 11`
- Outcome: SUCCESS
- Key metric: `mean_raw_kl_to_final=11.3881 -> mean_tuned_kl_to_final=3.4580`; `final_position_mean_raw_kl_to_final=14.8685 -> 7.5721`
- Artifacts saved: `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1`
- Latest checkpoint: `results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoints/training_state.pt`
- Anomalies: final-position top-1 only improved from `0.1010` to `0.1250`, so the cleanest gain is distributional rather than answer-token recovery
- Next step: register the pilot as a viability pass, then open the follow-up that decides whether later routed-versus-original work should keep KL as the primary tuned-lens baseline metric or sharpen the lens objective for stronger final-position recovery

## [2026-03-17T13:17:57-0500] PRE-RUN: Gemma-2 tool-breakage baseline calibration
- tmux session: N/A
- Script: `scripts/run_tool_breakage_factual_recall_baseline.py`
- Command: `.venv/bin/python scripts/run_tool_breakage_factual_recall_baseline.py --model-name google/gemma-2-2b --device mps --output-dir results/tool_breakage/20260317-gemma2-tool-breakage-baseline-calibration --split pilot --exploratory --max-prompts 1 --optimization-steps 10 --learning-rate 0.1 --seed 11`
- Config: `collection=tool_breakage_factual_recall_v1`, `tuned_lens_checkpoint=results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`, `checkpoint_root=results/tool_breakage/20260317-gemma2-tool-breakage-baseline-calibration/checkpoints`
- What I'm testing: the first same-model tool-breakage runner can load the saved Gemma tuned lens, derive target-token traces from prompt metadata, optimize oracle alpha for one factual-recall prompt, and write a resumable prompt-result artifact.
- Expected outcome: the run saves `summary.json` plus one prompt-result checkpoint and reports both original and routed raw/tuned trace metrics without code-path failures.
- Expected duration: ~5-15 minutes
- Checkpoint path: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-calibration/checkpoints/prompt_results`
- Checkpoint cadence: once per completed prompt
- Log path: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-calibration`
- Resume command: `.venv/bin/python scripts/run_tool_breakage_factual_recall_baseline.py --model-name google/gemma-2-2b --device mps --output-dir results/tool_breakage/20260317-gemma2-tool-breakage-baseline-calibration --split pilot --exploratory --max-prompts 1 --optimization-steps 10 --learning-rate 0.1 --seed 11`
- Main confound to watch: target-token derivation and routed-prefix trace construction could be internally consistent enough to run while still misaligning the target token or layer-depth mapping.
- Implementation verified: YES - `.venv/bin/python -m unittest discover -s tests -p 'test*.py'` is green after adding prompt metadata and tool-breakage trace tests.
- Status: LAUNCHING
- Key metric: oracle best silhouette `= 0.1428` at `k = 2` versus matched-random `0.1093`; mean attention mass `= 0.5560`, mean MLP mass `= 0.4090`
- Artifacts saved: `results/pattern_analysis/20260317-gpt2xl-prereg-scale-pattern-analysis-v1.json`, `results/pattern_analysis/20260317-gpt2xl-prereg-scale-pattern-analysis-v1.md`
- Anomalies: the positive clustering gap is dominated by a `126 / 2` outlier split rather than a broad multi-cluster partition
- Next step: keep block-structure claims blocked and use `resattn-ojq` to test grouped-source and prompt-resampled robustness
- Key metric: selected feature source stayed `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`; `confirm_r_squared=-0.2154`; `confirm_mean_js=0.2377`; `predicted_mean_improvement=-0.0011` nats
- Artifacts saved: `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-prompt-hybrid-comparison.json`
- Latest checkpoint: none
- Anomalies: none of the prompt-level or hybrid candidates beat the existing token-aware baseline on pilot leave-one-out JS, so the confirm artifact stayed unchanged; every candidate again selected ridge `100.0`
- Next step: stop adding small feature families and move to the design-review follow-up in `resattn-23p`

## [2026-03-17T16:20:00-0500] PRE-RUN: development-model oracle-alpha logit-target geometry check
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_heldout_predictiveness_check.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-logit-target-check.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources 'position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat'`
- Config: `model=gpt2-xl`, `collection=oracle_alpha_phase1_v1`, `train_split=pilot`, `eval_split=confirm`, `target=oracle_alpha_logit_vector`, `feature_source=position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`, `device=mps fallback cpu`
- What I'm testing: whether moving the predictiveness target from raw simplex coordinates to alpha-logit coordinates improves held-out confirm recovery without reopening feature-family selection.
- Expected outcome: the run completes on the saved split, writes a geometry-isolated artifact, and shows whether constrained target coordinates can recover a positive routed-loss advantage or at least materially improve confirm metrics.
- Expected duration: ~10-20 minutes
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-logit-target-check.json`
- Resume command: rerun the command above
- Main confound to watch: with only `8` pilot and `8` confirm prompts, any gain could still reflect better geometry under the same sample bottleneck rather than a genuinely adequate predictor surface.
- Implementation verified: YES - `tests/test_oracle_alpha_controls.py` and `tests/test_oracle_alpha_runner.py` pass with the logit-target transform path before launch.
- Status: LAUNCHING

## [2026-03-17T16:26:00-0500] POST-RUN: development-model oracle-alpha logit-target geometry check
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-logit-target-check.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources 'position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat'`
- Outcome: PARTIAL
- Key metric: `confirm_r_squared=-0.3291`; `confirm_mean_js=0.2457`; `predicted_mean_improvement=+0.0693` nats versus `oracle_mean_improvement=+1.3409`
- Artifacts saved: `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-logit-target-check.json`
- Latest checkpoint: none
- Anomalies: the constrained target finally made held-out predicted loss positive on average, but descriptive alpha-recovery metrics regressed relative to the raw-simplex token-aware baseline and the selected ridge penalty stayed at `100.0`
- Next step: follow `resattn-3ns` with a compressed or lower-dimensional predictiveness target rather than more raw logit-target iteration

## [2026-03-17T17:05:00-0500] PRE-RUN: development-model oracle-alpha compressed depth-type-band target check
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_heldout_predictiveness_check.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-depth-type-band-logit-target-check.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --target-name oracle_alpha_depth_type_band_logit_vector --candidate-feature-sources 'position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat'`
- Config: `model=gpt2-xl`, `collection=oracle_alpha_phase1_v1`, `train_split=pilot`, `eval_split=confirm`, `target=oracle_alpha_depth_type_band_logit_vector`, `feature_source=position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`, `device=mps fallback cpu`
- What I'm testing: whether compressing the alpha target into deterministic depth-band-by-source-type group logits improves held-out predictiveness relative to the full-source logit target while keeping the same feature surface.
- Expected outcome: the run completes on the saved split, writes a compressed-target artifact, and shows whether lower target dimensionality improves routed-loss recovery without making alpha-shape diagnostics unusably worse.
- Expected duration: ~10-20 minutes
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-depth-type-band-logit-target-check.json`
- Resume command: rerun the command above
- Main confound to watch: if routed loss improves only because the lift-back template is too coarse, the result may trade away too much descriptive alpha fidelity to support a stronger interpretation.
- Implementation verified: YES - `tests/test_oracle_alpha_controls.py` and `tests/test_oracle_alpha_runner.py` pass with the compressed target path and runner override before launch.
- Status: LAUNCHING

## [2026-03-17T17:10:00-0500] POST-RUN: development-model oracle-alpha compressed depth-type-band target check
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-depth-type-band-logit-target-check.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --target-name oracle_alpha_depth_type_band_logit_vector --candidate-feature-sources 'position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat'`
- Outcome: PARTIAL
- Key metric: `confirm_r_squared=-0.2241`; `confirm_mean_js=0.2380`; `predicted_mean_improvement=-0.0031` nats versus `oracle_mean_improvement=+1.3409`
- Artifacts saved: `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-depth-type-band-logit-target-check.json`
- Latest checkpoint: none
- Anomalies: the compressed target improved descriptive alpha metrics and broke the earlier `lambda=100.0` saturation by selecting `0.0001`, but it lost the positive held-out routed-loss gain of the full-source logit target
- Next step: follow `resattn-xaa` with loss-aware target and regularization selection rather than another blind target swap

## [2026-03-17T08:26:23-0500] PRE-RUN: development-model oracle-alpha loss-aware target comparison
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_heldout_predictiveness_check.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-loss-aware-target-comparison.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources 'position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat' --candidate-target-names oracle_alpha_vector oracle_alpha_logit_vector oracle_alpha_depth_type_band_logit_vector`
- Config: `model=gpt2-xl`, `collection=oracle_alpha_phase1_v1`, `train_split=pilot`, `eval_split=confirm`, `selection_metric=pilot_leave_one_out_mean_predicted_improvement_over_uniform`, `secondary_metric=mean_js_divergence`, `feature_source=position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`, `device=mps fallback cpu`
- What I'm testing: whether loss-aware pilot tuning over the current raw-simplex, full-logit, and compressed-logit targets picks a held-out predictiveness path that preserves routed-loss recovery better than the earlier descriptive-metric-first comparisons.
- Expected outcome: the run completes on the saved split, writes one comparison artifact, and clarifies whether the target tradeoff was mainly a metric-selection problem or a deeper predictor limitation.
- Expected duration: ~10-20 minutes
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-loss-aware-target-comparison.json`
- Resume command: rerun the command above
- Main confound to watch: with only `8` pilot prompts, loss-aware leave-one-out tuning may still be noisy enough to over-select a target that looks good on pilot but does not hold up on confirm.
- Implementation verified: YES - `tests/test_oracle_alpha_controls.py` and `tests/test_oracle_alpha_runner.py` pass with the loss-aware tuning path and multi-target comparison before launch.
- Status: LAUNCHING

## [2026-03-17T08:28:00-0500] POST-RUN: development-model oracle-alpha loss-aware target comparison
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-loss-aware-target-comparison.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources 'position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat' --candidate-target-names oracle_alpha_vector oracle_alpha_logit_vector oracle_alpha_depth_type_band_logit_vector`
- Outcome: FAILURE
- Key metric: selected target `oracle_alpha_vector`; `confirm_r_squared=-0.2154`; `confirm_mean_js=0.2377`; `predicted_mean_improvement=-0.0011` nats versus `oracle_mean_improvement=+1.3409`
- Artifacts saved: `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-loss-aware-target-comparison.json`
- Latest checkpoint: none
- Anomalies: loss-aware pilot tuning still preferred the raw-simplex target (`pilot mean improvement=+0.0990` nats) over the full-logit and compressed-logit alternatives, so the confirm result reverted to the same slightly negative raw-simplex baseline rather than recovering the earlier positive full-logit confirm result
- Next step: stop treating target/regularization selection as the main blocker and move to `resattn-7mb` for a more fundamental predictiveness redesign

## [2026-03-17T08:48:17-0500] PRE-RUN: development-model oracle-alpha pilot expansion check
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_heldout_predictiveness_check.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-pilot-expansion-check.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources 'position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat' --candidate-target-names oracle_alpha_vector oracle_alpha_logit_vector oracle_alpha_depth_type_band_logit_vector`
- Config: `model=gpt2-xl`, `prompt_registry=prompts/registry_v2.yaml`, `collection=oracle_alpha_phase1_v1`, `train_split=pilot (16 prompts)`, `eval_split=confirm (8 prompts)`, `selection_metric=pilot_leave_one_out_mean_predicted_improvement_over_uniform`, `secondary_metric=mean_js_divergence`, `feature_source=position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`, `device=mps fallback cpu`
- What I'm testing: whether doubling the saved oracle-alpha pilot surface is enough for the loss-aware target comparison to select a better-generalizing predictiveness path without changing the confirm set or predictor family.
- Expected outcome: the run completes on the expanded pilot split, writes a directly comparable artifact, and either stabilizes target selection away from the raw-simplex negative baseline or shows that pilot size alone is not the main blocker.
- Expected duration: ~15-25 minutes
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-pilot-expansion-check.json`
- Resume command: rerun the command above
- Main confound to watch: the new pilot prompts are still inline prompts rather than a benchmark dataset, so the larger split could help by smoothing idiosyncratic prompt selection rather than by fixing the target object itself.
- Implementation verified: YES - `tests/test_prompt_registry.py` and `tests/test_scaffold.py` pass with `prompts/registry_v2.yaml` before launch.
- Status: LAUNCHING

## [2026-03-17T08:52:22-0500] POST-RUN: development-model oracle-alpha pilot expansion check
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-pilot-expansion-check.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources 'position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat' --candidate-target-names oracle_alpha_vector oracle_alpha_logit_vector oracle_alpha_depth_type_band_logit_vector`
- Outcome: PARTIAL
- Key metric: selected target `oracle_alpha_logit_vector`; `confirm_r_squared=-0.2616`; `confirm_mean_js=0.2377`; `predicted_mean_improvement=+0.0857` nats versus `oracle_mean_improvement=+1.3409`
- Artifacts saved: `results/infrastructure/20260317-pilot-confirm-registry-v2.md`, `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-pilot-expansion-check.json`, `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-pilot-expansion-check.md`
- Latest checkpoint: none
- Anomalies: none
- Next step: follow `resattn-0vx` to scale the `registry_v2` logit-target path beyond the current `16 / 8` slice

## [2026-03-17T09:02:44-0500] PRE-RUN: development-model oracle-alpha registry_v3 scale check
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_heldout_predictiveness_check.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-registry-v3-scale-check.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources 'position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat' --candidate-target-names oracle_alpha_vector oracle_alpha_logit_vector oracle_alpha_depth_type_band_logit_vector`
- Config: `model=gpt2-xl`, `prompt_registry=prompts/registry_v3.yaml`, `collection=oracle_alpha_phase1_v1`, `train_split=pilot (32 prompts)`, `eval_split=confirm (16 prompts)`, `selection_metric=pilot_leave_one_out_mean_predicted_improvement_over_uniform`, `secondary_metric=mean_js_divergence`, `feature_source=position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`, `device=mps fallback cpu`
- What I'm testing: whether the positive `registry_v2` logit-target result survives a materially larger saved pilot and confirm surface without changing the feature source, target candidates, or tuning rule.
- Expected outcome: the run completes on the `32 / 16` split, writes a directly comparable artifact, and either keeps the logit target selected with positive confirm routed loss or shows that the `16 / 8` result was still too fragile.
- Expected duration: ~15-30 minutes
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-registry-v3-scale-check.json`
- Resume command: rerun the command above
- Main confound to watch: the larger saved surface is still an inline prompt set, so stronger results here would show scale sensitivity inside this local regime, not benchmark-level robustness by themselves.
- Implementation verified: YES - `tests/test_prompt_registry.py` and `tests/test_scaffold.py` pass with `prompts/registry_v3.yaml` before launch.
- Status: LAUNCHING

## [2026-03-17T09:08:00-0500] POST-RUN: development-model oracle-alpha registry_v3 scale check
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --output results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-registry-v3-scale-check.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources 'position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat' --candidate-target-names oracle_alpha_vector oracle_alpha_logit_vector oracle_alpha_depth_type_band_logit_vector`
- Outcome: PARTIAL
- Key metric: selected target `oracle_alpha_logit_vector`; `confirm_r_squared=-0.1954`; `confirm_mean_js=0.2421`; `predicted_mean_improvement=+0.1277` nats versus `oracle_mean_improvement=+1.2726`
- Artifacts saved: `results/infrastructure/20260317-pilot-confirm-registry-v3.md`, `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-registry-v3-scale-check.json`, `results/oracle_alpha/20260317-gpt2xl-heldout-predictiveness-registry-v3-scale-check.md`
- Latest checkpoint: none
- Anomalies: none
- Next step: follow `resattn-9jq` to scale the same logit-target path toward the prereg-sized Phase 1 gate

## [2026-03-17T10:32:57-0500] PRE-RUN: prereg-scale oracle-alpha campaign calibration
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_predictiveness_campaign.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_predictiveness_campaign.py --output-dir results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4 --max-train-sequences 2 --max-eval-sequences 2 --optimization-steps 20 --learning-rate 0.1 --seed 11 --regularization-grid 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100`
- Config: `model=gpt2-xl`, `device=mps fallback cpu`, `collection=oracle_alpha_phase1_v1`, `train_split=pilot`, `eval_split=confirm`, `feature_source=position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`, `targets=oracle_alpha_vector + oracle_alpha_logit_vector + oracle_alpha_depth_type_band_logit_vector`
- What I'm testing: the checkpointed campaign runner writes prompt-level oracle checkpoints and feature caches into the final prereg-scale output directory and can be resumed cleanly before the long tmux launch.
- Expected outcome: `2` pilot checkpoints, `2` confirm checkpoints, and matching feature-cache files appear on disk, and rerunning the same command reuses them instead of recomputing them.
- Expected duration: ~10-20 minutes including first model load
- Checkpoint path: `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/checkpoints`
- Checkpoint cadence: after each prompt-level oracle result and each prompt-level feature-vector materialization
- Log path: `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/calibration.log`
- Resume command: `.venv/bin/python scripts/run_oracle_alpha_predictiveness_campaign.py --output-dir results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4 --max-train-sequences 2 --max-eval-sequences 2 --optimization-steps 20 --learning-rate 0.1 --seed 11 --regularization-grid 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100`
- Main confound to watch: the first `gpt2-xl` load on MPS could dominate runtime and make resume behavior harder to notice if I only look at wall-clock time.
- Implementation verified: YES - full unit suite and `pre-commit` are green on `wip/resattn-scaffold`, and the cached-result path was inspected in `validation/oracle_alpha_runner.py` plus `validation/oracle_alpha_campaign.py` before launch.
- Status: LAUNCHING

## [2026-03-17T10:36:41-0500] POST-RUN: prereg-scale oracle-alpha campaign calibration
- Command: `.venv/bin/python scripts/run_oracle_alpha_predictiveness_campaign.py --output-dir results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4 --max-train-sequences 2 --max-eval-sequences 2 --optimization-steps 20 --learning-rate 0.1 --seed 11 --regularization-grid 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100`
- Outcome: SUCCESS
- Key metric: `2` pilot oracle checkpoints, `2` confirm oracle checkpoints, and matching feature-cache files were written; rerunning the exact same command left checkpoint mtimes unchanged, confirming resume reuse on the cached prompt subset
- Artifacts saved: `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/`
- Latest checkpoint: `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/checkpoints/`
- Anomalies: the calibration summary itself is not scientifically meaningful on `2 / 2`; it is only an operational checkpoint and resume verification
- Next step: launch the full prereg-scale `96 / 128` campaign in `tmux` against the same output directory

## [2026-03-17T10:36:41-0500] PRE-RUN: prereg-scale oracle-alpha campaign
- tmux session: `oa-v4-prereg-scale`
- Script: `scripts/run_oracle_alpha_predictiveness_campaign.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_predictiveness_campaign.py --model-name gpt2-xl --device mps --output-dir results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4 --optimization-steps 20 --learning-rate 0.1 --seed 11 --regularization-grid 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100`
- Config: `model=gpt2-xl`, `device=mps fallback cpu`, `collection=oracle_alpha_phase1_v1`, `train_split=pilot (96 prompts)`, `eval_split=confirm (128 prompts)`, `feature_source=position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`, `targets=oracle_alpha_vector + oracle_alpha_logit_vector + oracle_alpha_depth_type_band_logit_vector`
- What I'm testing: whether the current loss-aware predictiveness path stays positive and stable on the first prereg-scale saved prompt surface while persisting reusable per-prompt oracle and feature artifacts.
- Expected outcome: the run completes on `registry_v4`, writes the full campaign manifest plus split summaries, keeps checkpoint counts growing beyond the calibration subset, and produces the largest held-out predictiveness artifact in the repo.
- Expected duration: ~45-90 minutes
- Checkpoint path: `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/checkpoints`
- Checkpoint cadence: after each prompt-level oracle result and each prompt-level feature-vector materialization
- Log path: `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/campaign.log`
- Resume command: `.venv/bin/python scripts/run_oracle_alpha_predictiveness_campaign.py --model-name gpt2-xl --device mps --output-dir results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4 --optimization-steps 20 --learning-rate 0.1 --seed 11 --regularization-grid 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100`
- Main confound to watch: the run could stay positive on mean predicted routed loss while still concentrating that gain unevenly across confirm prompts, so the per-prompt outputs matter as much as the top-line average.
- Implementation verified: YES - the calibration run wrote prompt-level checkpoints and the exact resume command reused them without rewriting cached files.
- Status: LAUNCHING

## [2026-03-17T11:05:55-0500] POST-RUN: prereg-scale oracle-alpha campaign
- Command: `.venv/bin/python scripts/run_oracle_alpha_predictiveness_campaign.py --model-name gpt2-xl --device mps --output-dir results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4 --optimization-steps 20 --learning-rate 0.1 --seed 11 --regularization-grid 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100`
- Outcome: SUCCESS
- Key metric: oracle confirm mean improvement over uniform `= +1.2993` nats on `128` confirm prompts; predicted mean improvement over uniform `= +0.1162` nats with `95 / 128` prompts positive
- Artifacts saved: `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/`, `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4.md`
- Latest checkpoint: `results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/checkpoints/`
- Anomalies: the summary-stage sweep took much longer than the oracle checkpointing stage and left `predictiveness_summary.json` and `campaign_manifest.json` stale until the final write; follow-up `resattn-9co` tracks that observability gap
- Next step: register the prereg-scale artifact as the first development-model oracle gate clear, close `resattn-9jq`, and move the next oracle task to prereg-scale pattern analysis on the saved `registry_v4` artifact

## [2026-03-17T13:17:57-0500] PRE-RUN: Gemma-2 tool-breakage baseline calibration
- tmux session: N/A
- Script: `scripts/run_tool_breakage_factual_recall_baseline.py`
- Command: `.venv/bin/python scripts/run_tool_breakage_factual_recall_baseline.py --model-name google/gemma-2-2b --device mps --output-dir results/tool_breakage/20260317-gemma2-tool-breakage-baseline-calibration --split pilot --exploratory --max-prompts 1 --optimization-steps 10 --learning-rate 0.1 --seed 11`
- Config: `collection=tool_breakage_factual_recall_v1`, `split=pilot (1 prompt calibration)`, `tuned_lens_checkpoint=results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`, `primary_metric=held-out KL to final distribution`, `device=mps fallback cpu`
- What I'm testing: whether the first same-model Gemma tool-breakage runner produces sane routed-versus-original traces, writes prompt-level checkpoints, and keeps the tuned-lens baseline aligned with the saved metric hierarchy.
- Expected outcome: one prompt checkpoint and one `summary.json` appear under the output directory, and routed traces differ enough from original traces to catch obvious layer-order or target-token bugs.
- Expected duration: ~5-10 minutes
- Checkpoint path: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-calibration/checkpoints/prompt_results`
- Checkpoint cadence: after each prompt result
- Log path: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-calibration/calibration.log`
- Resume command: `.venv/bin/python scripts/run_tool_breakage_factual_recall_baseline.py --model-name google/gemma-2-2b --device mps --output-dir results/tool_breakage/20260317-gemma2-tool-breakage-baseline-calibration --split pilot --exploratory --max-prompts 1 --optimization-steps 10 --learning-rate 0.1 --seed 11`
- Main confound to watch: the first runner could look broken either because the tuned lens is misapplied layerwise or because the factual-recall prompt does not expose a clear routed-versus-original gap on one example.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_prompt_registry tests.test_tool_breakage tests.test_scaffold` passed before launch.
- Status: LAUNCHING

## [2026-03-17T13:29:41-0500] POST-RUN: Gemma-2 tool-breakage baseline calibration
- Outcome: SUCCESS
- Key metric: routing increased tuned-lens mean KL to the final distribution from `1.6851` to `3.5009` and final-position tuned KL from `2.0873` to `3.7710` on the first factual-recall pilot prompt
- Artifacts saved: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-calibration/summary.json`, `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-calibration/checkpoints/prompt_results/`
- Latest checkpoint: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-calibration/checkpoints/prompt_results/tb-pilot-001-c72233c57c.json`
- Anomalies: raw and tuned traces were already non-monotonic on the original-model baseline for this first prompt, so the calibration only establishes runner sanity, not a strong breakage delta
- Next step: launch the full `8`-prompt pilot baseline in `tmux` with the checkpointed runner and inspect whether routing increases KL or non-monotonicity relative to the original baseline across the saved factual-recall pilot set

## [2026-03-17T13:36:00-0500] PRE-RUN: Gemma-2 tool-breakage baseline pilot v1
- tmux session: `gemma-tb-pilot-v1`
- Script: `scripts/run_tool_breakage_factual_recall_baseline.py`
- Command: `.venv/bin/python scripts/run_tool_breakage_factual_recall_baseline.py --model-name google/gemma-2-2b --device mps --output-dir results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1 --split pilot --exploratory --optimization-steps 20 --learning-rate 0.1 --seed 11`
- Config: `collection=tool_breakage_factual_recall_v1`, `split=pilot (8 prompts)`, `tuned_lens_checkpoint=results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`, `primary_metric=held-out KL to final distribution`, `secondary_metrics=mean top1 + final-position KL/top1`, `device=mps fallback cpu`
- What I'm testing: whether same-model oracle-routed traces on the full factual-recall pilot set show larger KL degradation and more non-monotonic correct-token traces than the original-model baseline under both raw and tuned lens.
- Expected outcome: prompt-level checkpoint JSONs appear after each pilot prompt, the run writes a reusable `summary.json`, and the pilot artifact shows whether routed traces materially destabilize the tuned-lens baseline instead of just the raw logit lens.
- Expected duration: ~25-45 minutes
- Checkpoint path: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1/checkpoints/prompt_results`
- Checkpoint cadence: after each prompt result
- Log path: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1/pilot.log`
- Resume command: `.venv/bin/python scripts/run_tool_breakage_factual_recall_baseline.py --model-name google/gemma-2-2b --device mps --output-dir results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1 --split pilot --exploratory --optimization-steps 20 --learning-rate 0.1 --seed 11`
- Main confound to watch: the first-token factual-recall target may understate instability on multi-token answers, so any answer-token-facing read must stay subordinate to the KL-primary metric hierarchy.
- Implementation verified: YES - calibration wrote a prompt checkpoint and produced the expected routed-versus-original KL separation on the first pilot prompt.
- Status: LAUNCHING

## [2026-03-17T13:44:57-0500] POST-RUN: Gemma-2 tool-breakage baseline pilot v1
- Outcome: SUCCESS
- Key metric: routing worsened tuned-lens mean KL from `3.3544` to `5.8709` and tuned final-position KL from `7.5721` to `11.4121` on the full `8`-prompt factual-recall pilot split
- Artifacts saved: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1/summary.json`, `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1.md`, `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1/checkpoints/prompt_results/`
- Latest checkpoint: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1/checkpoints/prompt_results/`
- Anomalies: the boolean non-monotonicity metric saturated on the original-model baseline (`8 / 8` raw and tuned), so the pilot validates runner behavior and broad KL degradation under routing but does not yet provide an informative relative strong-claim threshold
- Next step: use the saved pilot traces to codify stronger routed-versus-original success metrics in `resattn-ypj` before any confirmatory factual-recall run

## [2026-03-17T14:18:00-0500] PRE-RUN: Gemma-2 tool-breakage baseline confirm v1
- tmux session: `gemma-tb-confirm-v1`
- Script: `scripts/run_tool_breakage_factual_recall_baseline.py`
- Command: `.venv/bin/python scripts/run_tool_breakage_factual_recall_baseline.py --model-name google/gemma-2-2b --device mps --output-dir results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1 --split confirm --optimization-steps 20 --learning-rate 0.1 --seed 11`
- Config: `collection=tool_breakage_factual_recall_v1`, `split=confirm (8 prompts, locked)`, `exploratory=false`, `tuned_lens_checkpoint=results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`, `primary_metric=held-out KL to final distribution`, `relative_metrics=non-monotonicity increase + final-rank worsening + best-rank worsening + rank-range increase`, `device=mps fallback cpu`
- What I'm testing: whether the locked confirm split reproduces broad same-model routed-versus-original degradation under the KL-primary metric hierarchy and whether the new relative rank metrics stay informative on unseen factual-recall prompts.
- Expected outcome: prompt-level confirm checkpoints appear after each prompt, the run writes a reusable `summary.json`, and the confirm artifact clarifies whether tuned routed traces consistently broaden target-rank instability relative to the original baseline.
- Expected duration: ~25-45 minutes
- Checkpoint path: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1/checkpoints/prompt_results`
- Checkpoint cadence: after each prompt result
- Log path: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1/confirm.log`
- Resume command: `.venv/bin/python scripts/run_tool_breakage_factual_recall_baseline.py --model-name google/gemma-2-2b --device mps --output-dir results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1 --split confirm --optimization-steps 20 --learning-rate 0.1 --seed 11`
- Main confound to watch: the first-token factual-recall target may still understate answer-string instability on multi-token targets, so the confirm write-up must preserve the answer-token caveat even if the rank-based metrics replicate cleanly.
- Implementation verified: YES - the pilot run completed with prompt-level resume checkpoints and the saved metric-hardening step regenerated the summary from cached prompt results successfully.
- Status: LAUNCHING

## [2026-03-17T13:51:26-0500] POST-RUN: Gemma-2 tool-breakage baseline confirm v1
- Outcome: SUCCESS
- Key metric: routing worsened tuned-lens mean KL from `3.3834` to `5.9061` and tuned final-position KL from `5.9851` to `8.8947` on the locked `8`-prompt confirm split
- Artifacts saved: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1/summary.json`, `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1.md`, `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1/checkpoints/prompt_results/`
- Latest checkpoint: `results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1/checkpoints/prompt_results/`
- Anomalies: the non-monotonicity-increase metric stayed `0 / 8` under both raw and tuned lens, so the confirm read still depends on the codified rank-instability surface rather than the legacy boolean
- Next step: use `resattn-g09` to add the controlled dynamic-routing counterfactual before making the strong Gemma tool-breakage claim

## [2026-03-17T14:14:28-0500] PRE-RUN: Gemma-2 tool-breakage counterfactual smoke
- tmux session: N/A
- Script: `scripts/run_tool_breakage_dynamic_counterfactual.py`
- Command: `.venv/bin/python scripts/run_tool_breakage_dynamic_counterfactual.py --model-name google/gemma-2-2b --device mps --baseline-summary results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1/summary.json --fixed-alpha-summary results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1/summary.json --max-prompts 2 --output-dir results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-smoke`
- Config: `collection=tool_breakage_factual_recall_v1`, `split=confirm source of truth from resattn-6te`, `controls=prompt_permuted_alpha + pilot_mean_alpha`, `tuned_lens_checkpoint=results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`, `device=mps fallback cpu`
- What I'm testing: whether the new counterfactual runner reuses the saved baseline prompt results, writes checkpointed control-arm artifacts, and produces sane routed-versus-control deltas on a tiny confirm subset.
- Expected outcome: `summary.json` plus per-prompt checkpoint files appear under the output directory, and routed traces are measurably worse than at least one control on tuned KL / tuned rank metrics.
- Expected duration: ~5-10 minutes
- Checkpoint path: `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-smoke/checkpoints/prompt_results`
- Checkpoint cadence: after each prompt result
- Log path: `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-smoke/smoke.log`
- Resume command: `.venv/bin/python scripts/run_tool_breakage_dynamic_counterfactual.py --model-name google/gemma-2-2b --device mps --baseline-summary results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1/summary.json --fixed-alpha-summary results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1/summary.json --max-prompts 2 --output-dir results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-smoke`
- Main confound to watch: with only two prompts, the prompt-permuted control is a single swap, so a weird donor pairing could exaggerate or suppress the routed-versus-control gap.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_tool_breakage` passed before launch.
- Status: LAUNCHING

## [2026-03-17T14:15:35-0500] POST-RUN: Gemma-2 tool-breakage counterfactual smoke
- Outcome: SUCCESS
- Key metric: on the `2`-prompt smoke subset, routed tuned KL stayed worse than the fixed `pilot_mean_alpha` control (`routed minus control = +1.2240`) but was better than the prompt-permuted swap control (`routed minus control = -0.7417`)
- Artifacts saved: `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-smoke/summary.json`, `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-smoke/checkpoints/prompt_results/`
- Latest checkpoint: `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-smoke/checkpoints/prompt_results/`
- Anomalies: the `2`-prompt prompt-permuted control is just a single donor swap, so it is too pairing-sensitive to interpret as anything except runner smoke.
- Next step: launch the full locked confirm counterfactual artifact in `tmux` and judge the prompt-permuted control only on the full `8`-prompt confirm set.

## [2026-03-17T14:16:06-0500] PRE-RUN: Gemma-2 tool-breakage counterfactual confirm v1
- tmux session: `gemma-tb-counterfactual-v1`
- Script: `scripts/run_tool_breakage_dynamic_counterfactual.py`
- Command: `.venv/bin/python scripts/run_tool_breakage_dynamic_counterfactual.py --model-name google/gemma-2-2b --device mps --baseline-summary results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1/summary.json --fixed-alpha-summary results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1/summary.json --output-dir results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-confirm-v1`
- Config: `collection=tool_breakage_factual_recall_v1`, `split=confirm (8 prompts, locked)`, `source_of_truth=resattn-6te baseline summary`, `controls=prompt_permuted_alpha + pilot_mean_alpha`, `tuned_lens_checkpoint=results/tool_breakage/20260317-gemma2-tuned-lens-viability-pilot-v1/checkpoint.pt`, `device=mps fallback cpu`
- What I'm testing: whether the saved Gemma routed traces remain more damaging than both a prompt-permuted dynamic control and a fixed pilot-mean alpha control on the locked factual-recall confirm split.
- Expected outcome: prompt-level counterfactual checkpoints appear after each prompt, the run writes a reusable `summary.json`, and the tuned routed-vs-control deltas stay positive on KL and mostly positive on the relative rank-instability metrics.
- Expected duration: ~20-35 minutes
- Checkpoint path: `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-confirm-v1/checkpoints/prompt_results`
- Checkpoint cadence: after each prompt result
- Log path: `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-confirm-v1/confirm.log`
- Resume command: `.venv/bin/python scripts/run_tool_breakage_dynamic_counterfactual.py --model-name google/gemma-2-2b --device mps --baseline-summary results/tool_breakage/20260317-gemma2-tool-breakage-baseline-confirm-v1/summary.json --fixed-alpha-summary results/tool_breakage/20260317-gemma2-tool-breakage-baseline-pilot-v1/summary.json --output-dir results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-confirm-v1`
- Main confound to watch: if prompt-permuted alphas are almost as damaging as the prompt-matched route, the strong Gemma claim weakens from “input-dependent routing specifically breaks the lens traces” to “non-uniform routed mixtures often break them.”
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_tool_breakage` passed and the `2`-prompt smoke wrote reusable control-arm checkpoints before launch.
- Status: LAUNCHING

## [2026-03-17T14:18:38-0500] POST-RUN: Gemma-2 tool-breakage counterfactual confirm v1
- Outcome: SUCCESS
- Key metric: routed tuned KL remained worse than the fixed `pilot_mean_alpha` control (`routed minus control = +0.6091`), but the prompt-permuted dynamic control was more damaging than the prompt-matched routed trace on average (`routed minus control = -0.3082`)
- Artifacts saved: `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-confirm-v1/summary.json`, `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-confirm-v1/checkpoints/prompt_results/`
- Latest checkpoint: `results/tool_breakage/20260317-gemma2-tool-breakage-counterfactual-confirm-v1/checkpoints/prompt_results/`
- Anomalies: the dynamic control answers the prereg question cleanly but weakens the strong same-model claim boundary, because prompt-misaligned dynamic routing is at least as damaging as the prompt-matched route on the primary tuned-KL metric.
- Next step: register `resattn-g09` as a mixed / negative control result, keep the strong Gemma tool-breakage claim blocked, and decide whether the next highest-value work is the Figure 8 proxy lane or a finer-grained follow-up control.
