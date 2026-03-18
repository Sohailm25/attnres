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

## [2026-03-17T16:25:00-0500] PRE-RUN: compact-subword capacity-first Figure 8 proxy follow-up
- tmux session: `attnres-111-capacity`
- Script: `scripts/run_attnres_proxy_viability.py`
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 256 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 160 --n-heads 4 --n-layers 8 --d-ff 640 --max-seq-len 64 --batch-size 16 --num-steps 1500 --checkpoint-every-steps 50 --learning-rate 3e-4 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-capacity-v1`
- Config: `tokenizer_mode=compact_subword`, `vocab_size=20000`, `d_model=160`, `d_ff=640`, `n_layers=8`, `seq_len=64`, `steps=1500`, `batch_size=16`, `seed=11`
- What I'm testing: whether model capacity, rather than tokenization, is the next bottleneck for the local compact-subword Block AttnRes Figure 8 proxy.
- Expected outcome: the wider compact-subword proxy regains a loss edge over the matched baseline while preserving or improving the paper-facing Figure 8 proxy metrics relative to the smaller compact-subword run.
- Expected duration: ~30-90 minutes
- Checkpoint path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-capacity-v1/checkpoints/`
- Checkpoint cadence: every `50` steps
- Log path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-capacity-v1/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: if the wider proxy loses at the fixed `1500`-step horizon, the result may still reflect optimization undertraining rather than proving capacity is unhelpful.
- Implementation verified: YES - existing compact-subword proxy runner already supports width scaling, checkpoint/resume, and summary export; unit tests for the runner path are already green on trunk.
- Status: LAUNCHING

## [2026-03-17T16:28:00-0500] POST-RUN: compact-subword capacity-first Figure 8 proxy follow-up initial launch
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 256 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 512 --d-model 160 --n-heads 4 --n-layers 8 --d-ff 640 --max-seq-len 64 --batch-size 16 --num-steps 1500 --checkpoint-every-steps 50 --learning-rate 3e-4 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-capacity-v1`
- Outcome: FAILURE
- Key metric: launch blocked before training by `compact subword vocabulary size 18494 exceeds config.vocab_size=512`
- Artifacts saved: `results/figure8_validation/20260317-attnres-proxy-compact-subword-capacity-v1/run.log`
- Anomalies: the first spec reused the old character-level vocabulary assumption instead of the saved compact-subword manifest requirement
- Next step: relaunch with `vocab_size=20000`, matching the saved compact-subword regime on the same data slice

## [2026-03-17T16:31:00-0500] PRE-RUN: compact-subword capacity-first Figure 8 proxy follow-up relaunch
- tmux session: `attnres-111-capacity`
- Script: `scripts/run_attnres_proxy_viability.py`
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 256 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 160 --n-heads 4 --n-layers 8 --d-ff 640 --max-seq-len 64 --batch-size 16 --num-steps 1500 --checkpoint-every-steps 50 --learning-rate 3e-4 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-capacity-v1`
- Config: `tokenizer_mode=compact_subword`, `vocab_size=20000`, `d_model=160`, `d_ff=640`, `n_layers=8`, `seq_len=64`, `steps=1500`, `batch_size=16`, `seed=11`
- What I'm testing: whether width, rather than tokenization, is the next bottleneck for the local compact-subword Block AttnRes Figure 8 proxy.
- Expected outcome: the wider compact-subword proxy regains a loss edge over the matched baseline while preserving or improving the paper-facing Figure 8 proxy metrics relative to the smaller compact-subword run.
- Expected duration: ~30-90 minutes
- Checkpoint path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-capacity-v1/checkpoints/`
- Checkpoint cadence: every `50` steps
- Log path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-capacity-v1/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: if the wider proxy loses at the fixed `1500`-step horizon, the result may still reflect optimization undertraining rather than proving capacity is unhelpful.
- Implementation verified: YES - the saved compact-subword artifact on the same data slice already proves `vocab_size=20000` is the correct regime, so this relaunch isolates width rather than re-testing tokenization.
- Status: LAUNCHING

## [2026-03-17T20:05:00-0500] PRE-RUN: primary-spine Gemma model-backed reconstruction smoke
- tmux session: N/A
- Script: `scripts/phase1_model_backed_reconstruction_smoke.py`
- Command: `.venv/bin/python scripts/phase1_model_backed_reconstruction_smoke.py --model-name google/gemma-2-2b --prompt "The capital of France is" --device mps --fallback-device cpu --output results/infrastructure/20260317-gemma2-reconstruction-smoke.json`
- Config: `model=google/gemma-2-2b`, `prompt="The capital of France is"`, `device=mps fallback=cpu`
- What I'm testing: whether the existing TransformerLens-backed reconstruction smoke path works unchanged on the primary Gemma-2 model and reconstructs the model's own logits from cached sublayer writes.
- Expected outcome: either exact residual/logit reconstruction on Gemma-2 or a concrete backend-specific blocker with inspectable evidence.
- Expected duration: ~10-25 minutes depending on model load and local cache state
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/infrastructure/20260317-gemma2-reconstruction-smoke.json`
- Resume command: rerun the exact command above
- Main confound to watch: TransformerLens Gemma support, hook naming, or final-normalization handling may differ from `gpt2-xl` even if the generic reconstruction code is otherwise correct.
- Implementation verified: YES - the same script and validation path already produce an exact `gpt2-xl` reconstruction artifact on this machine.
- Status: LAUNCHING

## [2026-03-17T20:08:00-0500] POST-RUN: primary-spine Gemma model-backed reconstruction smoke
- Command: `.venv/bin/python scripts/phase1_model_backed_reconstruction_smoke.py --model-name google/gemma-2-2b --prompt "The capital of France is" --device mps --fallback-device cpu --output results/infrastructure/20260317-gemma2-reconstruction-smoke.json`
- Outcome: SUCCESS
- Key metric: `final_residual_max_abs_error=0.0`, `uniform_logits_max_abs_error=0.0`
- Artifacts saved: `results/infrastructure/20260317-gemma2-reconstruction-smoke.json`
- Latest checkpoint: none
- Anomalies: none
- Next step: close the primary-spine reconstruction readiness issue and move the top scientific priority to a bounded Gemma oracle-alpha feasibility slice

## [2026-03-17T20:20:00-0500] PRE-RUN: primary-model Gemma oracle-alpha feasibility slice
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_development_slice.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_development_slice.py --model-name google/gemma-2-2b --output results/oracle_alpha/20260317-gemma2-development-slice.json --max-sequences 2 --optimization-steps 20 --learning-rate 0.1 --seed 11 --device mps`
- Config: `model=google/gemma-2-2b`, `collection=oracle_alpha_phase1_v1`, `split=pilot`, `exploratory=true`, `max_sequences=2`, `steps=20`, `lr=0.1`, `seed=11`
- What I'm testing: whether the existing bounded oracle-alpha runner executes cleanly on the primary Gemma-2 spine and produces a first non-claim-bearing loss-improvement artifact over uniform and the preregistered nulls.
- Expected outcome: the run completes on local MPS, writes a real artifact, and keeps optimized loss no worse than uniform on the bounded pilot slice.
- Expected duration: ~10-25 minutes depending on first-run model path and cache cost
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/oracle_alpha/20260317-gemma2-development-slice.json`
- Resume command: rerun the exact command above
- Main confound to watch: Gemma-specific cache layout or runtime pressure could still break the oracle runner even though reconstruction readiness is green.
- Implementation verified: YES - `resattn-2s0` already proved that the same cached sublayer decomposition and final-normalization path work exactly on `google/gemma-2-2b`.
- Status: LAUNCHING

## [2026-03-17T20:23:00-0500] POST-RUN: primary-model Gemma oracle-alpha feasibility slice
- Command: `.venv/bin/python scripts/run_oracle_alpha_development_slice.py --model-name google/gemma-2-2b --output results/oracle_alpha/20260317-gemma2-development-slice.json --max-sequences 2 --optimization-steps 20 --learning-rate 0.1 --seed 11 --device mps`
- Outcome: SUCCESS
- Key metric: `sequence_mean_improvement=2.1634` nats over uniform on `2` pilot prompts
- Artifacts saved: `results/oracle_alpha/20260317-gemma2-development-slice.json`
- Latest checkpoint: none
- Anomalies: none
- Next step: close the primary-model bounded feasibility issue and scale the same path to the saved pilot stability suite

## [2026-03-17T20:32:00-0500] PRE-RUN: primary-model Gemma oracle-alpha pilot stability suite
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_pilot_stability_suite.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_pilot_stability_suite.py --model-name google/gemma-2-2b --output results/oracle_alpha/20260317-gemma2-pilot-stability-suite.json --max-sequences 8 --optimization-steps 20 --learning-rate 0.1 --resample-count 3 --resample-size 6 --device mps`
- Config: `model=google/gemma-2-2b`, `collection=oracle_alpha_phase1_v1`, `split=pilot`, `exploratory=true`, `max_sequences=8`, `steps=20`, `lr=0.1`, `resample_count=3`, `resample_size=6`
- What I'm testing: whether the primary-model Gemma oracle lane stays positive and methodologically usable when scaled from the `2`-prompt smoke to the saved pilot stability suite with restart, paraphrase, and resample checks.
- Expected outcome: the run completes cleanly on local MPS, mean pilot improvement stays positive, and restart/paraphrase/resample metrics are inspectable from one saved artifact.
- Expected duration: ~10-25 minutes
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/oracle_alpha/20260317-gemma2-pilot-stability-suite.json`
- Resume command: rerun the exact command above
- Main confound to watch: the primary model may stay positive on the base pilot run but show much noisier prompt-perturbation behavior than `gpt2-xl`, which would narrow the interpretation immediately.
- Implementation verified: YES - `resattn-7cs` already showed that the same bounded runner path executes cleanly on `google/gemma-2-2b`.
- Status: LAUNCHING

## [2026-03-17T20:38:00-0500] POST-RUN: primary-model Gemma oracle-alpha pilot stability suite
- Command: `.venv/bin/python scripts/run_oracle_alpha_pilot_stability_suite.py --model-name google/gemma-2-2b --output results/oracle_alpha/20260317-gemma2-pilot-stability-suite.json --max-sequences 8 --optimization-steps 20 --learning-rate 0.1 --resample-count 3 --resample-size 6 --device mps`
- Outcome: SUCCESS
- Key metric: `sequence_mean_improvement=1.7613` nats; `restart_js=7.63e-08`; `paraphrase_js=0.0200`; `resample_js=0.0151`
- Artifacts saved: `results/oracle_alpha/20260317-gemma2-pilot-stability-suite.json`
- Latest checkpoint: none
- Anomalies: none
- Next step: close the primary-model pilot stability issue and run the first primary-model held-out predictiveness check

## [2026-03-17T20:47:00-0500] PRE-RUN: primary-model Gemma held-out oracle-alpha predictiveness check
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_heldout_predictiveness_check.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_heldout_predictiveness_check.py --model-name google/gemma-2-2b --output results/oracle_alpha/20260317-gemma2-heldout-predictiveness-check.json --optimization-steps 20 --learning-rate 0.1 --seed 11 --device mps`
- Config: `model=google/gemma-2-2b`, `collection=oracle_alpha_phase1_v1`, `train_split=pilot`, `eval_split=confirm`, `candidate_feature_sources=default saved set`, `candidate_targets=runner defaults`, `steps=20`, `lr=0.1`, `seed=11`
- What I'm testing: whether the saved primary-model pilot signal generalizes beyond descriptive fitting to the confirm split under the existing held-out predictiveness machinery.
- Expected outcome: the run completes on local MPS, writes a real held-out artifact, and either preserves positive routed-loss recovery on confirm or exposes the same structural blocker seen earlier on the development model.
- Expected duration: ~10-25 minutes
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/oracle_alpha/20260317-gemma2-heldout-predictiveness-check.json`
- Resume command: rerun the exact command above
- Main confound to watch: the current default feature family may be less appropriate on Gemma than on `gpt2-xl`, so a weak held-out result would diagnose the current predictor surface before it diagnoses the whole primary-model oracle lane.
- Implementation verified: YES - `resattn-a7j` already showed the primary model clears the saved pilot stability suite on the same prompt and control registries.
- Status: LAUNCHING

## [2026-03-18T00:49:00-0500] POST-RUN: primary-model Gemma held-out oracle-alpha predictiveness check
- Outcome: SUCCESS
- Key metric: predicted mean improvement over uniform on the saved `128`-prompt confirm split was `+1.0428` nats with bootstrap interval `[0.9495, 1.1331]`, while the confirm oracle stayed at `+2.5525` nats with `128 / 128` prompts positive
- Artifacts saved: `results/oracle_alpha/20260317-gemma2-heldout-predictiveness-check.json`, `results/oracle_alpha/20260317-gemma2-heldout-predictiveness-check.md`
- Latest checkpoint: none
- Anomalies: the bounded run took about an hour because the current ridge helper solves the high-dimensional primal system in the `n << d` regime and emits no intermediate progress artifact during the numerical sweep; the saved JSON also needed a post-run wording fix after the MIB rationale string incorrectly referenced a development-model runner stage
- Next step: close `resattn-js8`, move the next oracle question to primary-model pattern analysis in `resattn-2sb`, and track the runtime bottleneck separately in `resattn-b4q`

## [2026-03-18T01:13:00-0500] PRE-RUN: primary-model Gemma prereg-scale pattern analysis
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_pattern_analysis.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_pattern_analysis.py --run-path results/oracle_alpha/20260317-gemma2-heldout-predictiveness-check-eval-run.json --output results/pattern_analysis/20260318-gemma2-prereg-scale-pattern-analysis-v1.json --random-seed 11 --max-clusters 12 --num-resamples 64`
- Config: `model=google/gemma-2-2b`, `split=confirm`, `num_sequences=128`, `source_view=raw + grouped`, `max_clusters=12`, `resamples=64`
- What I'm testing: whether the primary-model Gemma oracle distributions on the saved confirm split show stronger raw-source or grouped coarse structure than the current development-model pattern-analysis lane.
- Expected outcome: the existing runner consumes a compact saved Gemma eval-run artifact without code changes and writes a primary-model pattern-analysis summary under `results/pattern_analysis/`.
- Expected duration: ~1-5 minutes
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/pattern_analysis/20260318-gemma2-prereg-scale-pattern-analysis-v1.json`
- Resume command: rerun the exact command above
- Main confound to watch: a stronger grouped-view signal could still coexist with weak raw-source structure, so the interpretation has to distinguish coarse source-type structure from a true block-structure pass.
- Implementation verified: YES - `tests.test_pattern_analysis` already covers the compact raw-run contract the script expects, and the saved `resattn-js8` raw backup is small enough to materialize a committed compact eval-run artifact under the hook size limit.
- Status: LAUNCHING

## [2026-03-18T01:16:00-0500] POST-RUN: primary-model Gemma prereg-scale pattern analysis
- Outcome: SUCCESS
- Key metric: raw-source oracle best silhouette stayed weak at `0.1084` versus random `0.1598`, while grouped views were clearly above random (`source_type` `0.6731` versus `0.5569`; `depth_thirds_by_type` `0.3858` versus `0.2305`)
- Artifacts saved: `results/oracle_alpha/20260317-gemma2-heldout-predictiveness-check-eval-run.json`, `results/pattern_analysis/20260318-gemma2-prereg-scale-pattern-analysis-v1.json`, `results/pattern_analysis/20260318-gemma2-prereg-scale-pattern-analysis-v1.md`
- Latest checkpoint: none
- Anomalies: the first compact Gemma eval-run artifact was still too large for the large-file hook until it was reduced to the exact fields the pattern-analysis runner actually reads
- Next step: close `resattn-2sb`, keep the raw block-structure gate unpassed on the primary model, and move the next core oracle issue to `resattn-5eo`

## [2026-03-18T01:34:00-0500] PRE-RUN: primary-model Gemma oracle regime comparison v1
- tmux session: `oa-gemma-regimes-v1`
- Script: `scripts/run_oracle_alpha_regime_comparison.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_regime_comparison.py --model-name google/gemma-2-2b --device mps --output-dir results/comparison_regimes/20260318-gemma2-regime-comparison-v1 --optimization-steps 20 --learning-rate 0.1 --seed 11`
- Config: `collection=oracle_alpha_phase1_v1`, `split=confirm`, `regimes=softmax, unconstrained, top-k{2,4,8,13,26}`, `steps=20`, `lr=0.1`, `seed=11`
- What I'm testing: whether the prereg softmax-constrained regime meaningfully differs from the matched-initialization unconstrained and top-k families on the primary Gemma confirm surface.
- Expected outcome: the runner writes resumable per-regime checkpoints under the output directory and a compact `summary.json` that compares mean loss improvement over uniform, positive-prompt counts, and sparsity summaries across regimes.
- Expected duration: ~30-90 minutes
- Checkpoint path: `results/comparison_regimes/20260318-gemma2-regime-comparison-v1/runs/`
- Checkpoint cadence: after each completed regime
- Log path: `results/comparison_regimes/20260318-gemma2-regime-comparison-v1/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: unconstrained sigmoid gating has a different scale than the simplex regimes, so the interpretation has to center on preregistered loss improvement over the shared uniform baseline rather than on raw alpha magnitudes alone.
- Implementation verified: YES - targeted regime-transform tests pass on the tiny runner path and the comparison summary path already writes and reuses per-regime checkpoints on the tiny model.
- Status: LAUNCHING

## [2026-03-17T16:38:00-0500] POST-RUN: compact-subword capacity-first Figure 8 proxy follow-up relaunch
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 256 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 160 --n-heads 4 --n-layers 8 --d-ff 640 --max-seq-len 64 --batch-size 16 --num-steps 1500 --checkpoint-every-steps 50 --learning-rate 3e-4 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-capacity-v1`
- Outcome: SUCCESS
- Key metric: baseline best eval loss `6.5899`, AttnRes best eval loss `6.6496`, delta `+0.0598`; deep embedding persistence `0.1515`
- Artifacts saved: `results/figure8_validation/20260317-attnres-proxy-compact-subword-capacity-v1/`, `results/figure8_validation/20260317-attnres-proxy-compact-subword-capacity-v1.md`
- Latest checkpoint: `results/figure8_validation/20260317-attnres-proxy-compact-subword-capacity-v1/checkpoints/attnres_training_state.pt`
- Anomalies: the initial launch used the wrong compact-subword vocabulary size; rerunning the exact final command after completion reused the saved checkpoints and finished in `7.52` seconds
- Next step: close `resattn-111` as a useful capacity-isolation result and move the Figure 8 lane to `resattn-jci`, which tests optimization horizon on the widened compact-subword regime

## [2026-03-17T17:05:00-0500] PRE-RUN: widened compact-subword Figure 8 horizon follow-up
- tmux session: `attnres-jci-horizon`
- Script: `scripts/run_attnres_proxy_viability.py`
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 256 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 160 --n-heads 4 --n-layers 8 --d-ff 640 --max-seq-len 64 --batch-size 16 --num-steps 4500 --checkpoint-every-steps 50 --learning-rate 3e-4 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-horizon-v1`
- Config: `tokenizer_mode=compact_subword`, `vocab_size=20000`, `d_model=160`, `d_ff=640`, `n_layers=8`, `seq_len=64`, `steps=4500`, `batch_size=16`, `seed=11`
- What I'm testing: whether longer optimization horizon, rather than more width or another tokenization change, is the next credible lever for the widened compact-subword Figure 8 proxy.
- Expected outcome: the widened AttnRes proxy should close or reverse the negative loss gap versus the matched baseline while preserving or improving the compact-subword Figure-8-facing metrics.
- Expected duration: ~10-20 minutes from the resumed `1500`-step checkpoints
- Checkpoint path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-horizon-v1/checkpoints/`
- Checkpoint cadence: every `50` steps
- Log path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-horizon-v1/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: if the widened baseline keeps improving faster than the routed proxy even over a much longer horizon, the next blocker is unlikely to be training budget alone.
- Implementation verified: YES - the horizon run will reuse the saved widened compact-subword checkpoints from the prior artifact, so it isolates optimization horizon without reopening tokenization or width.
- Status: LAUNCHING

## [2026-03-17T17:13:00-0500] POST-RUN: widened compact-subword Figure 8 horizon follow-up
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-2-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 256 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 160 --n-heads 4 --n-layers 8 --d-ff 640 --max-seq-len 64 --batch-size 16 --num-steps 4500 --checkpoint-every-steps 50 --learning-rate 3e-4 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-horizon-v1`
- Outcome: SUCCESS
- Key metric: baseline best eval loss `6.5899`, AttnRes best eval loss `6.6496`, delta `+0.0598`; deep embedding persistence `0.1415`
- Artifacts saved: `results/figure8_validation/20260317-attnres-proxy-compact-subword-horizon-v1/`, `results/figure8_validation/20260317-attnres-proxy-compact-subword-horizon-v1.md`
- Latest checkpoint: `results/figure8_validation/20260317-attnres-proxy-compact-subword-horizon-v1/checkpoints/attnres_training_state.pt`
- Anomalies: the longer-horizon run resumed correctly from the saved `1500`-step checkpoints, but the best routed proxy loss did not improve at all over the earlier widened artifact; rerunning the exact final command after completion reused the saved checkpoints and finished in `7.96` seconds
- Next step: close `resattn-jci` as a negative result for the “it just needs more optimization budget” hypothesis and move the Figure 8 lane to the bounded redesign issue `resattn-du2`

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

## [2026-03-17T17:38:00-0500] PRE-RUN: Gemma-2 IT refusal-feature discovery smoke
- tmux session: N/A
- Script: `scripts/run_refusal_feature_discovery_validation.py`
- Command: `.venv/bin/python scripts/run_refusal_feature_discovery_validation.py --device mps --max-pilot-groups 1 --max-confirm-groups 1 --max-new-tokens 24 --output-dir results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-smoke`
- Config: `model=google/gemma-2-2b-it`, `collection=safety_refusal_discovery_v1`, `pilot_groups=1`, `confirm_groups=1`, `positions=instruction_final + assistant_prefill`, `behavior_check=greedy_refusal_marker`
- What I'm testing: whether the new aligned-Gemma safety workflow writes prompt-level residual checkpoints, respects the grouped prompt surface, and produces a sane summary on a tiny split before a larger artifact run.
- Expected outcome: `summary.json` plus per-prompt checkpoint files appear in the output directory, and the refusal prompt shows refusal-like behavior while the control prompts remain non-refusal.
- Expected duration: ~5-10 minutes
- Checkpoint path: `results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-smoke/checkpoints/prompt_residuals`
- Checkpoint cadence: after each prompt result
- Log path: `results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-smoke/summary.json`
- Resume command: `.venv/bin/python scripts/run_refusal_feature_discovery_validation.py --device mps --max-pilot-groups 1 --max-confirm-groups 1 --max-new-tokens 24 --output-dir results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-smoke`
- Main confound to watch: prompt formatting on the `-it` model could change the intended instruction-final and assistant-prefill positions if the chat template behaves differently than expected.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_safety_alignment tests.test_prompt_registry tests.test_scaffold` passed before launch.
- Status: LAUNCHING

## [2026-03-17T17:41:00-0500] POST-RUN: Gemma-2 IT refusal-feature discovery smoke
- Outcome: SUCCESS
- Key metric: refusal and non-refusal behavior checks all matched expectation on the `1 / 1` smoke, and both refusal and harmfulness directions separated their intended pairs perfectly on the tiny split
- Artifacts saved: `results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-smoke/summary.json`, `results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-smoke/checkpoints/prompt_residuals/`
- Latest checkpoint: `results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-smoke/checkpoints/prompt_residuals/sa-confirm-001-refusal.pt`
- Anomalies: the `1 / 1` smoke is too small to trust the confirm-side cross metrics; it only verifies the runner, prompt formatting, and aligned-model behavior precondition
- Next step: launch the full `6 / 6` aligned-Gemma workflow artifact in `tmux` and verify checkpoint reuse after completion

## [2026-03-17T17:42:00-0500] PRE-RUN: Gemma-2 IT refusal-feature discovery validation v1
- tmux session: `gemma-safety-3f1-v1`
- Script: `scripts/run_refusal_feature_discovery_validation.py`
- Command: `.venv/bin/python scripts/run_refusal_feature_discovery_validation.py --device mps --max-new-tokens 32 --output-dir results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-v1`
- Config: `model=google/gemma-2-2b-it`, `collection=safety_refusal_discovery_v1`, `pilot_groups=6`, `confirm_groups=6`, `positions=instruction_final + assistant_prefill`, `behavior_check=greedy_refusal_marker`
- What I'm testing: whether the first aligned-Gemma refusal-discovery workflow validates on the frozen pilot/confirm prompt triples with reusable prompt-level residual checkpoints.
- Expected outcome: prompt-level checkpoints are written after each prompt, the held-out confirm summary stays strong for both refusal and harmfulness separation, and the run leaves behind reusable prompt residuals for later mediator-conditioned safety work.
- Expected duration: ~10-20 minutes
- Checkpoint path: `results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-v1/checkpoints/prompt_residuals`
- Checkpoint cadence: after each prompt result
- Log path: `results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-v1/run.log`
- Resume command: `.venv/bin/python scripts/run_refusal_feature_discovery_validation.py --device mps --max-new-tokens 32 --output-dir results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-v1`
- Main confound to watch: the prompt triples may be coherent enough to validate the workflow while still being too templated for any later claim about real-world safety behavior, so this artifact must be written as workflow validation rather than a safety-routing result.
- Implementation verified: YES - the `1 / 1` smoke wrote prompt-level checkpoints and a sane summary on `google/gemma-2-2b-it`.
- Status: LAUNCHING

## [2026-03-17T17:47:00-0500] POST-RUN: Gemma-2 IT refusal-feature discovery validation v1
- Outcome: SUCCESS
- Key metric: held-out confirm pair accuracy stayed `1.0` for both the refusal direction at assistant-prefill layer `22` and the harmfulness direction at instruction-final layer `25`, with behavior checks matching expectation on all `36 / 36` prompts
- Artifacts saved: `results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-v1/summary.json`, `results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-v1/checkpoints/prompt_residuals/`
- Latest checkpoint: `results/safety_alignment/20260317-gemma2it-refusal-feature-discovery-v1/checkpoints/prompt_residuals/sa-confirm-006-refusal.pt`
- Anomalies: the cross-direction metrics are not zero (`0.67` refusal cross pair accuracy and `0.83` harmfulness cross pair accuracy on confirm), so this validates the workflow without licensing a “single perfectly disentangled safety direction” claim
- Next step: close `resattn-3f1` as a workflow-validation success, keep strong safety-routing claims blocked on later causal mediator checks, and open the next aligned-Gemma safety follow-up

## [2026-03-17T18:26:00-0500] PRE-RUN: Figure 8 `wikitext-103` corpus calibration
- tmux session: N/A
- Script: `scripts/run_attnres_proxy_viability.py`
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-103-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 512 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 160 --n-heads 4 --n-layers 8 --d-ff 640 --max-seq-len 64 --batch-size 16 --num-steps 200 --checkpoint-every-steps 50 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-calibration`
- Config: `dataset=wikitext/wikitext-103-raw-v1`, `train_texts=2048`, `eval_texts=512`, `tokenizer=compact_subword:gpt2`, `vocab_size=20000`, `d_model=160`, `d_ff=640`, `n_layers=8`, `seq_len=64`, `steps=200`
- What I'm testing: the widened compact-subword proxy on `wikitext-103` stays within the existing compact vocabulary cap, writes checkpoints cleanly, and gives a runtime estimate before the full corpus-first launch.
- Expected outcome: both model checkpoints and a `summary.json` appear under the calibration directory, rerunning the exact command reuses those checkpoints, and the run is slow enough or fast enough to size the tmux launch responsibly.
- Expected duration: ~10-20 minutes
- Checkpoint path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-calibration/checkpoints/`
- Checkpoint cadence: every `50` steps and at completion
- Log path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-calibration/run.log`
- Resume command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-103-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 512 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 160 --n-heads 4 --n-layers 8 --d-ff 640 --max-seq-len 64 --batch-size 16 --num-steps 200 --checkpoint-every-steps 50 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-calibration`
- Main confound to watch: the larger validation slice may make early eval-loss movement noisier to compare numerically against the older `256`-text artifacts, so the calibration is operational only.
- Implementation verified: YES - the runner already passed the existing Figure 8 unit tests, and the compact vocabulary sweep confirmed this `2048 / 512` slice stays under `vocab_size=20000`.
- Status: LAUNCHING

## [2026-03-17T18:42:00-0500] POST-RUN: Figure 8 `wikitext-103` corpus calibration
- Outcome: SUCCESS
- Key metric: at `200` steps, the widened Block AttnRes proxy beat the matched baseline on eval loss (`6.8764` vs `6.9299`, delta `-0.0534`) and deep embedding persistence rose to `0.2022`
- Artifacts saved: `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-calibration/summary.json`, `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-calibration/tokenizer_manifest.json`, `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-calibration/checkpoints/`
- Latest checkpoint: `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-calibration/checkpoints/attnres_training_state.pt`
- Anomalies: the entropy ordering stayed inverted (`mean_pre_attn_entropy = 1.4557`, `mean_pre_mlp_entropy = 1.5164`), so the calibration is positive but not yet a Figure 8 pass
- Next step: launch the full `1500`-step `wikitext-103` corpus-first run in `tmux` without bundling a horizon change

## [2026-03-17T18:43:00-0500] PRE-RUN: Figure 8 `wikitext-103` corpus-first run v1
- tmux session: `fig8-wt103-v1`
- Script: `scripts/run_attnres_proxy_viability.py`
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-103-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 512 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 160 --n-heads 4 --n-layers 8 --d-ff 640 --max-seq-len 64 --batch-size 16 --num-steps 1500 --checkpoint-every-steps 50 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-v1`
- Config: `dataset=wikitext/wikitext-103-raw-v1`, `train_texts=2048`, `eval_texts=512`, `tokenizer=compact_subword:gpt2`, `vocab_size=20000`, `d_model=160`, `d_ff=640`, `n_layers=8`, `seq_len=64`, `steps=1500`
- What I'm testing: whether the corpus-first redesign alone restores competitiveness with the matched baseline and strengthens the Figure-8-facing metric surface on the widened compact-subword proxy.
- Expected outcome: the routed proxy stays competitive or better than the matched baseline at the full `1500`-step horizon, checkpoint/resume remain clean, and the saved Figure 8 metrics move in the paper-like direction relative to the earlier widened `wikitext-2` artifacts.
- Expected duration: ~15-30 minutes
- Checkpoint path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-v1/checkpoints/`
- Checkpoint cadence: every `50` steps and at completion
- Log path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-v1/run.log`
- Resume command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-103-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 512 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 160 --n-heads 4 --n-layers 8 --d-ff 640 --max-seq-len 64 --batch-size 16 --num-steps 1500 --checkpoint-every-steps 50 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-v1`
- Main confound to watch: the larger corpus may improve competitiveness while leaving the entropy ordering inverted, which would strengthen the regime story but still block a stronger Figure 8 alignment claim.
- Implementation verified: YES - the exact `wikitext-103` configuration wrote checkpoints, saved a summary, and reused checkpoints on the `200`-step calibration in `9.81` seconds.
- Status: LAUNCHING

## [2026-03-17T18:46:00-0500] POST-RUN: Figure 8 `wikitext-103` corpus-first run v1
- Outcome: SUCCESS
- Key metric: widened `wikitext-103` best loss delta stayed mixed at `+0.0386` (`baseline=6.5929`, `AttnRes=6.6315`), while deep embedding persistence improved to `0.1615`
- Artifacts saved: `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-v1/summary.json`, `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-v1.md`, `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-v1/tokenizer_manifest.json`
- Latest checkpoint: `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-v1/checkpoints/attnres_training_state.pt`
- Anomalies: I briefly misread the live tmux completion state and launched a duplicate foreground resume against the same output directory; I caught it immediately, killed the duplicate process, and kept the original tmux run as the source of truth before finalizing the artifact
- Next step: close `resattn-7y4`, create `resattn-fby` for best-checkpoint / eval-trajectory support on the widened `wikitext-103` regime, and shift overall repo priority to `resattn-73l`

## [2026-03-17T18:01:46-0500] PRE-RUN: aligned Gemma refusal-direction intervention smoke
- tmux session: N/A
- Script: `scripts/run_refusal_direction_intervention_check.py`
- Command: `.venv/bin/python scripts/run_refusal_direction_intervention_check.py --device mps --max-new-tokens 24 --max-pilot-groups 1 --max-confirm-groups 1 --output-dir results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-smoke`
- Config: `model=google/gemma-2-2b-it`, `collection=safety_refusal_discovery_v1`, `pilot_groups=1`, `confirm_groups=1`, `max_new_tokens=24`
- What I'm testing: the bounded refusal-direction coefficient-replacement runner executes cleanly on the aligned model and produces safe summary-only outputs before the full confirm run.
- Expected outcome: the script completes, writes `summary.json`, keeps raw completions out of the committed summary surface, and shows at least one arm moving refusal-marker behavior in the expected direction.
- Expected duration: ~5-15 minutes
- Checkpoint path: `results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-smoke/checkpoints/prompt_residuals/`
- Checkpoint cadence: prompt-level residual checkpoints written on first prompt access
- Log path: `results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-smoke/run.log`
- Resume command: `.venv/bin/python scripts/run_refusal_direction_intervention_check.py --device mps --max-new-tokens 24 --max-pilot-groups 1 --max-confirm-groups 1 --output-dir results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-smoke`
- Main confound to watch: a naive intervention scale could either do nothing or overpower the residual stream, so this smoke is primarily for runner validity before the full confirm slice.
- Implementation verified: YES - `tests/test_safety_alignment.py` passes after adding projection-replacement and behavior-summary helpers.
- Status: LAUNCHING

## [2026-03-17T18:08:51-0500] POST-RUN: aligned Gemma refusal-direction intervention smoke
- Outcome: SUCCESS
- Key metric: the binary refusal marker stayed unchanged on `1 / 1`, but the new continuation-preference metric moved in the expected direction for the refusal arms (`margin_delta=-0.2049` on refusal suppression, `+0.1583` on harmful-context refusal injection) while the matched harmfulness controls stayed flat (`0.0`)
- Artifacts saved: `results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-smoke/summary.json`, `results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-smoke/checkpoints/prompt_residuals/`
- Latest checkpoint: `results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-smoke/checkpoints/prompt_residuals/sa-confirm-001-refusal.pt`
- Anomalies: the first launch was blocked by shell redirection before the output directory existed, and the first scripted rerun surfaced a TransformerLens hook-signature bug; both were fixed before the successful smoke, and the smoke showed the original binary behavior metric was too coarse on its own
- Next step: run the full `6 / 6` confirm-split intervention artifact with the continuation-preference metric kept as a secondary readout beside the prereg-aligned greedy refusal marker

## [2026-03-17T18:08:51-0500] PRE-RUN: aligned Gemma refusal-direction intervention v1
- tmux session: N/A
- Script: `scripts/run_refusal_direction_intervention_check.py`
- Command: `.venv/bin/python scripts/run_refusal_direction_intervention_check.py --device mps --max-new-tokens 24 --output-dir results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-v1`
- Config: `model=google/gemma-2-2b-it`, `collection=safety_refusal_discovery_v1`, `pilot_groups=6`, `confirm_groups=6`, `max_new_tokens=24`
- What I'm testing: whether the localized refusal direction on aligned Gemma causally shifts matched safe continuation preference on the frozen confirm split, and whether the matched harmfulness controls stay weaker or flat at the current bounded intervention surface.
- Expected outcome: refusal suppression lowers refusal-versus-context preference on refusal prompts, refusal injection raises it on harmful-context prompts, and the harmfulness controls do not recapitulate the same pattern.
- Expected duration: ~10-25 minutes
- Checkpoint path: `results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-v1/checkpoints/prompt_residuals/`
- Checkpoint cadence: prompt-level residual checkpoints written on first prompt access
- Log path: `results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-v1/run.log`
- Resume command: `.venv/bin/python scripts/run_refusal_direction_intervention_check.py --device mps --max-new-tokens 24 --output-dir results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-v1`
- Main confound to watch: the greedy refusal marker may remain mostly saturated even if the continuation-preference metric moves, so the final interpretation needs both readouts and should not overstate binary flips.
- Implementation verified: YES - the `1 / 1` smoke reproduced the expected preference-direction effects while keeping the harmfulness controls flat.
- Status: LAUNCHING

## [2026-03-17T18:13:35-0500] POST-RUN: aligned Gemma refusal-direction intervention v1
- Outcome: SUCCESS
- Key metric: refusal suppression lowered the refusal-versus-context preference margin on refusal prompts by `0.1891`, refusal injection raised the same margin on harmful-context prompts by `0.1741`, and the matched harmfulness controls stayed flat at `0.0`
- Artifacts saved: `results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-v1/summary.json`, `results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-v1/checkpoints/prompt_residuals/`
- Latest checkpoint: `results/safety_alignment/20260317-gemma2it-refusal-direction-intervention-v1/checkpoints/prompt_residuals/sa-confirm-006-refusal.pt`
- Anomalies: the greedy refusal marker stayed saturated on refusal prompts and only flipped on `1 / 6` harmful-context prompts, so the continuation-preference metric became the decisive causal readout rather than the binary marker alone
- Next step: close `resattn-73l` as the bounded causal mediator pass, open mediator-conditioned routing follow-up `resattn-h1p`, and move overall repo priority to `resattn-fby`

## [2026-03-17T18:24:08-0500] PRE-RUN: Figure 8 `wikitext-103` best-checkpoint follow-up v1
- tmux session: `fig8-fby-wt103-v1`
- Script: `scripts/run_attnres_proxy_viability.py`
- Command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-103-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 512 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 160 --n-heads 4 --n-layers 8 --d-ff 640 --max-seq-len 64 --batch-size 16 --num-steps 1500 --checkpoint-every-steps 50 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-bestcheck-v1`
- Config: `dataset=wikitext/wikitext-103-raw-v1`, `train_texts=2048`, `eval_texts=512`, `tokenizer=compact_subword:gpt2`, `vocab_size=20000`, `d_model=160`, `d_ff=640`, `n_layers=8`, `seq_len=64`, `steps=1500`
- What I'm testing: whether preserving eval history and exporting the best AttnRes checkpoint changes the Figure 8 read on the widened `wikitext-103` regime relative to the final-checkpoint-only artifact.
- Expected outcome: the run writes resumable training checkpoints plus `*_best_state.pt` and `*_eval_history.json`, and the resulting summary can compare final versus best AttnRes Figure 8 proxy metrics without another rerun.
- Expected duration: ~15-30 minutes
- Checkpoint path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-bestcheck-v1/checkpoints/`
- Checkpoint cadence: every `50` steps and at completion, with best-state refresh on eval improvement
- Log path: `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-bestcheck-v1/run.log`
- Resume command: `.venv/bin/python scripts/run_attnres_proxy_viability.py --dataset-name wikitext --dataset-config wikitext-103-raw-v1 --train-split train --eval-split validation --text-field text --max-train-texts 2048 --max-eval-texts 512 --tokenizer-mode compact_subword --tokenizer-name gpt2 --separator-text '\n\n' --vocab-size 20000 --d-model 160 --n-heads 4 --n-layers 8 --d-ff 640 --max-seq-len 64 --batch-size 16 --num-steps 1500 --checkpoint-every-steps 50 --learning-rate 0.0003 --weight-decay 0.01 --seed 11 --device mps --output-dir results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-bestcheck-v1`
- Main confound to watch: the best checkpoint may improve eval loss without materially improving the Figure 8 surface, which would sharpen the argument against optimization-shape alone as the next redesign lever.
- Implementation verified: YES - `tests/test_attnres_reproduction.py` now covers best-checkpoint and eval-history persistence, and the full `tests/` suite stays green after the runner change.
- Status: LAUNCHING

## [2026-03-17T18:46:49-0500] POST-RUN: Figure 8 `wikitext-103` best-checkpoint follow-up v1
- Outcome: SUCCESS
- Key metric: final loss delta `= +0.1272`, best-checkpoint loss delta `= +0.0386`, and best-checkpoint deep embedding persistence `= 0.1689`
- Artifacts saved: `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-bestcheck-v1/summary.json`, `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-bestcheck-v1.md`, `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-bestcheck-v1/checkpoints/`
- Latest checkpoint: `results/figure8_validation/20260317-attnres-proxy-compact-subword-wikitext103-bestcheck-v1/checkpoints/attnres_best_state.pt`
- Anomalies: both models peaked early (`baseline=900`, `attnres=850`), but the best-checkpoint Figure 8 surface only improved modestly and still kept the entropy ordering inverted
- Next step: close `resattn-fby`, keep the widened `wikitext-103` regime as the current Figure 8 default, and move the next Figure 8 question to a bounded optimization or objective redesign issue rather than another blind rerun

## [2026-03-17T18:59:59-0500] PRE-RUN: prereg-scale pattern-analysis grouped-view and resampling follow-up
- tmux session: N/A
- Script: `scripts/run_oracle_alpha_pattern_analysis.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_pattern_analysis.py --run-path results/oracle_alpha/20260317-gpt2xl-prereg-scale-campaign-v4/oracle_eval_run.json --output results/pattern_analysis/20260317-gpt2xl-prereg-scale-pattern-analysis-ojq-v1.json --random-seed 11 --max-clusters 12 --num-resamples 128 --sample-size 96`
- Config: `model=gpt2-xl`, `split=confirm`, `view_names=raw_source/source_type/depth_thirds_by_type`, `resamples=128`, `sample_size=96`
- What I'm testing: whether the weak prereg-scale routing structure survives grouped-source compression and prompt-resampling stability checks, or whether it still looks like an outlier-driven raw-source artifact.
- Expected outcome: the new JSON summary preserves the existing raw-source read, adds grouped-source scans plus resampling stability, and makes it clear whether any broader cluster story survives the control extensions.
- Expected duration: ~1-5 minutes
- Checkpoint path: N/A
- Checkpoint cadence: N/A
- Log path: `results/pattern_analysis/20260317-gpt2xl-prereg-scale-pattern-analysis-ojq-v1.json`
- Resume command: rerun the command above
- Main confound to watch: grouped views can create apparently cleaner structure by collapsing source dimension, so the interpretation has to compare grouped-source gains against the matched random control and the raw-source resampling stability rather than treating any cleaner silhouette as a direct win.
- Implementation verified: YES - `tests.test_pattern_analysis` now covers grouped views, cluster-size persistence, and resampling-stability summaries.
- Status: LAUNCHING

## [2026-03-17T19:00:00-0500] POST-RUN: prereg-scale pattern-analysis grouped-view and resampling follow-up
- Outcome: SUCCESS
- Key metric: raw-source structure stayed weak and `k = 2`-dominated (`0.1428` versus random `0.1093`), while grouped views strengthened only the coarse structure story (`source_type` `0.6652` versus `0.5569`; `depth_thirds_by_type` `0.3451` versus `0.2305`)
- Artifacts saved: `results/pattern_analysis/20260317-gpt2xl-prereg-scale-pattern-analysis-ojq-v1.json`, `results/pattern_analysis/20260317-gpt2xl-prereg-scale-pattern-analysis-ojq-v1.md`
- Latest checkpoint: none
- Anomalies: grouped-source compression did make the structure cleaner, but the matched random-control silhouette also rose sharply, so the honest update is coarse-regime evidence rather than a block-structure pass
- Next step: close `resattn-ojq`, keep the raw block-structure gate unpassed, and move overall repo priority to `resattn-h1p`

## [2026-03-17T19:15:20-0500] PRE-RUN: aligned Gemma mediator-conditioned safety routing v1
- tmux session: `safety-h1p-routing`
- Script: `scripts/run_mediator_conditioned_safety_routing_analysis.py`
- Command: `.venv/bin/python scripts/run_mediator_conditioned_safety_routing_analysis.py --device mps --max-new-tokens 24 --output-dir results/safety_alignment/20260317-gemma2it-mediator-conditioned-routing-v1`
- Config: `model=google/gemma-2-2b-it`, `collection=safety_refusal_discovery_v1`, `pilot_groups=6`, `confirm_groups=6`, `max_new_tokens=24`
- What I'm testing: whether the validated refusal mediator also organizes a distinct depth-trajectory pattern on the frozen confirm prompts when prompts are partitioned by mediator activation rather than descriptive refusal labels alone.
- Expected outcome: the run reuses prompt-level checkpoints, writes a summary-only mediator-conditioned trajectory artifact, and shows cleaner refusal-direction depth separation for mediator-active prompts than for mediator-inactive prompts.
- Expected duration: ~10-25 minutes
- Checkpoint path: `results/safety_alignment/20260317-gemma2it-mediator-conditioned-routing-v1/checkpoints/prompt_residuals/`
- Checkpoint cadence: prompt-level residual checkpoints written on first prompt access
- Log path: `results/safety_alignment/20260317-gemma2it-mediator-conditioned-routing-v1/run.log`
- Resume command: `.venv/bin/python scripts/run_mediator_conditioned_safety_routing_analysis.py --device mps --max-new-tokens 24 --output-dir results/safety_alignment/20260317-gemma2it-mediator-conditioned-routing-v1`
- Main confound to watch: the mediator-active partition could collapse back to the refusal role labels exactly, in which case the artifact would be little more than a relabeled role trajectory summary instead of a real mediator-conditioned read.
- Implementation verified: YES - `tests.test_safety_alignment` now covers the new trajectory and partition helpers before launch.
- Status: LAUNCHING

## [2026-03-17T19:24:00-0500] POST-RUN: aligned Gemma mediator-conditioned safety routing v1
- Outcome: SUCCESS
- Key metric: the confirm mediator partition still collapses to `6` refusal versus `12` non-refusal prompts, but refusal-direction interventions change the final-layer refusal trajectory by `-372.8501` on refusal prompts, `+369.9403` on harmful-context prompts, and `+358.7685` on benign prompts
- Artifacts saved: `results/safety_alignment/20260317-gemma2it-mediator-conditioned-routing-v1/summary.json`, `results/safety_alignment/20260317-gemma2it-mediator-conditioned-routing-v1.md`
- Latest checkpoint: `results/safety_alignment/20260317-gemma2it-mediator-conditioned-routing-v1/checkpoints/prompt_residuals/sa-confirm-006-refusal.pt`
- Anomalies: the mediator-active partition remained exactly role-collapsed on this frozen prompt set, so the artifact supports bounded trajectory claims rather than a richer prompt-subset story; rerunning the exact command reused checkpoints and completed in `11.55` seconds
- Next step: close `resattn-h1p` as a bounded stage-3 safety artifact and return to the open Figure 8 and Gemma decision issues

## [2026-03-17T19:52:00-0500] PRE-RUN: Figure 8 regularization-first sweep on widened `wikitext-103`
- tmux session: `attnres-bux-sweep`
- Script: `scripts/run_attnres_proxy_viability.py`
- Command: `for arm in dropout01 wd005 dropout01_wd005; do ...; done` over the widened compact-subword `wikitext-103` regime with three new settings: `(dropout=0.1, weight_decay=0.01)`, `(dropout=0.0, weight_decay=0.05)`, `(dropout=0.1, weight_decay=0.05)`
- Config: `dataset=wikitext/wikitext-103-raw-v1`, `train=2048`, `eval=512`, `tokenizer=compact_subword:gpt2`, `vocab_size=20000`, `d_model=160`, `d_ff=640`, `n_layers=8`, `seq_len=64`, `batch_size=16`, `steps=1500`, `checkpoint_every=50`, `lr=3e-4`, `seed=11`, `device=mps`
- What I'm testing: whether matched regularization can preserve a healthier best-checkpoint regime on the local Block AttnRes proxy before the Figure 8 lane is allowed to change the training objective.
- Expected outcome: at least one regularization arm narrows or flips the best-checkpoint baseline-versus-AttnRes loss delta while improving the entropy gap relative to the current best-checkpoint control (`-0.0549`).
- Expected duration: ~2-5 hours for the full three-arm sweep
- Checkpoint path: `results/figure8_validation/20260317-attnres-proxy-regularization-sweep-bux-v1/*/checkpoints/`
- Checkpoint cadence: every `50` steps
- Log path: `results/figure8_validation/20260317-attnres-proxy-regularization-sweep-bux-v1/sweep.log`
- Resume command: rerun the exact tmux launch command against the same output root
- Main confound to watch: regularization might help the matched baseline as much as or more than the routed proxy, which would make the sweep scientifically useful but not a rescue.
- Implementation verified: YES - the runner already exposes `--dropout`, best-checkpoint export, eval history, checkpoint resume, and the widened `wikitext-103` regime has completed successfully on trunk.
- Status: LAUNCHING

## [2026-03-17T19:48:30-0500] POST-RUN: Figure 8 regularization-first sweep on widened `wikitext-103`
- Outcome: SUCCESS
- Key metric: all three matched regularization arms stayed negative on best-checkpoint loss delta (`+0.0378`, `+0.0376`, `+0.0368`) versus the control `+0.0386`, and none materially improved the entropy gap beyond control `-0.0549`
- Artifacts saved: `results/figure8_validation/20260317-attnres-proxy-regularization-sweep-bux-v1/dropout01/summary.json`, `results/figure8_validation/20260317-attnres-proxy-regularization-sweep-bux-v1/wd005/summary.json`, `results/figure8_validation/20260317-attnres-proxy-regularization-sweep-bux-v1/dropout01_wd005/summary.json`, `results/figure8_validation/20260317-attnres-proxy-regularization-sweep-bux-v1.md`
- Latest checkpoint: `results/figure8_validation/20260317-attnres-proxy-regularization-sweep-bux-v1/dropout01_wd005/checkpoints/attnres_best_state.pt`
- Anomalies: dropout improved deep embedding persistence but not the actual loss-gap blocker; rerunning the exact sweep command reused all saved checkpoints and finished in `30.67` seconds
- Next step: close `resattn-bux` as the last faithful stabilization attempt and move the Figure 8 lane to the objective-level redesign decision in `resattn-9fo`

## [2026-03-18T01:47:00-0500] POST-RUN: primary-model Gemma oracle regime comparison v1 (invalid initial top-k pass)
- Outcome: PARTIAL
- Key metric: softmax and unconstrained checkpoints completed cleanly, but the first `top-k-k2` checkpoint returned exactly `0.0` mean improvement over uniform on every confirm prompt.
- Artifacts saved: `results/comparison_regimes/20260318-gemma2-regime-comparison-v1/runs/softmax-constrained.json`, `results/comparison_regimes/20260318-gemma2-regime-comparison-v1/runs/unconstrained.json`
- Latest checkpoint: `results/comparison_regimes/20260318-gemma2-regime-comparison-v1/runs/unconstrained.json`
- Anomalies: hard `top-k` masking with exact zero initialization left gradient flow only on the tied initial support, so the top-k results were optimization-artifact invalid rather than scientifically interpretable.
- Next step: patch the top-k optimization path to keep an exact hard top-k forward pass while restoring support-search gradients, then relaunch from the preserved softmax and unconstrained checkpoints only.

## [2026-03-18T01:49:00-0500] PRE-RUN: primary-model Gemma oracle regime comparison v1 relaunch
- tmux session: `oa-gemma-regimes-v1`
- Script: `scripts/run_oracle_alpha_regime_comparison.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_regime_comparison.py --model-name google/gemma-2-2b --device mps --output-dir results/comparison_regimes/20260318-gemma2-regime-comparison-v1 --optimization-steps 20 --learning-rate 0.1 --seed 11`
- Config: `collection=oracle_alpha_phase1_v1`, `split=confirm`, `regimes=softmax, unconstrained, top-k{2,4,8,13,26}`, `steps=20`, `lr=0.1`, `seed=11`
- What I'm testing: whether the prereg softmax-constrained regime still separates from unconstrained and the repaired top-k family on the primary Gemma confirm surface once the top-k support-search bug is removed.
- Expected outcome: the relaunch reuses the existing softmax and unconstrained checkpoints, regenerates the top-k checkpoints only, and writes a compact `summary.json` plus markdown memo with valid regime comparisons.
- Expected duration: ~20-60 minutes
- Checkpoint path: `results/comparison_regimes/20260318-gemma2-regime-comparison-v1/runs/`
- Checkpoint cadence: after each completed regime
- Log path: `results/comparison_regimes/20260318-gemma2-regime-comparison-v1/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: if the repaired top-k optimizer now matches softmax too closely, the regime story becomes “competition matters more than sparsity at this scale” rather than “softmax is uniquely beneficial.”
- Implementation verified: YES - new unit coverage confirms non-top-k logits receive gradient under exact zero initialization while the forward pass remains hard top-k sparse, and the comparison/regime tests remain green.
- Status: LAUNCHING

## [2026-03-18T02:06:00-0500] POST-RUN: primary-model Gemma oracle regime comparison v1 relaunch
- Outcome: SUCCESS
- Key metric: softmax-constrained stayed best at `+2.5525` nats over uniform, beat unconstrained (`+2.0637`) and every top-k setting on all `128` confirm prompts, and still led top-k `k = 26` by `+1.0861` nats.
- Artifacts saved: `results/comparison_regimes/20260318-gemma2-regime-comparison-v1/summary.json`, `results/comparison_regimes/20260318-gemma2-regime-comparison-v1.md`
- Latest checkpoint: `results/comparison_regimes/20260318-gemma2-regime-comparison-v1/runs/top-k-k26.json`
- Anomalies: the first top-k pass was optimization-invalid under exact zero initialization because hard masking froze support search on the arbitrary tie-broken initial support; the final artifact uses the repaired exact-forward straight-through top-k path instead.
- Next step: close `resattn-5eo` and move the next overall scientific action to `resattn-1lk`, the conservative Figure 8 freeze-or-escalate decision.

## [2026-03-18T02:20:16-0500] PRE-RUN: aligned Gemma broadened refusal-surface validation v2
- tmux session: `safety-mo5-validation`
- Script: `scripts/run_refusal_feature_discovery_validation.py`
- Command: `.venv/bin/python scripts/run_refusal_feature_discovery_validation.py --collection-id safety_refusal_surface_v2 --device mps --output-dir results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation`
- Config: `model=google/gemma-2-2b-it`, `collection=safety_refusal_surface_v2`, `pilot_groups=6`, `confirm_groups=6`, `max_new_tokens=32`
- What I'm testing: whether the broadened aligned-Gemma prompt surface still supports clean refusal-feature discovery while introducing refusal-style non-refusal prompts that can plausibly break the current role-collapsed mediator partition.
- Expected outcome: the run writes prompt-level checkpoints plus a held-out validation artifact, and behavior checks remain clean enough that the mediator-conditioned follow-up is worth running on the same collection.
- Expected duration: ~5-15 minutes
- Checkpoint path: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation/checkpoints/prompt_residuals/`
- Checkpoint cadence: after each prompt checkpoint
- Log path: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: the refusal-style non-refusal prompts could cause the aligned model to emit explicit refusal markers or collapse the new collection into behaviorally ambiguous examples, which would make the surface broader but less valid.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_prompt_registry tests.test_safety_alignment` passed after adding `safety_refusal_surface_v2`.
- Status: LAUNCHING

## [2026-03-18T02:24:23-0500] POST-RUN: aligned Gemma broadened refusal-surface validation v2
- Outcome: SUCCESS
- Key metric: refusal separation stayed clean on the broadened surface (`refusal` pair accuracy `= 1.0` on pilot and confirm, localized again at layer `22`), but the strict non-refusal behavior pass rate dropped to `0.8333` on pilot and `0.6667` on confirm because the new collection intentionally includes refusal-style non-refusal prompts.
- Artifacts saved: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation/summary.json`
- Latest checkpoint: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation/checkpoints/prompt_residuals/sa2-confirm-006-refusal.pt`
- Anomalies: the legacy refusal-marker behavior check is stricter than the new prompt intent for some `refusal_style_non_refusal` prompts, so behavior mismatch is no longer equivalent to “the broadened surface is invalid.”
- Next step: run the mediator-conditioned analysis on the same broadened collection and judge success on whether the mediator partition still collapses onto role labels.

## [2026-03-18T02:24:23-0500] PRE-RUN: aligned Gemma broadened mediator-conditioned routing v2
- tmux session: `safety-mo5-routing`
- Script: `scripts/run_mediator_conditioned_safety_routing_analysis.py`
- Command: `.venv/bin/python scripts/run_mediator_conditioned_safety_routing_analysis.py --collection-id safety_refusal_surface_v2 --device mps --output-dir results/safety_alignment/20260318-gemma2it-mediator-conditioned-routing-v2`
- Config: `model=google/gemma-2-2b-it`, `collection=safety_refusal_surface_v2`, `pilot_groups=6`, `confirm_groups=6`, `max_new_tokens=32`
- What I'm testing: whether the broadened aligned-Gemma surface produces a non-trivial mediator-active subset inside the non-refusal roles rather than collapsing exactly onto outright refusal prompts.
- Expected outcome: the run reuses prompt-level checkpoints, writes a new mediator-conditioned routing artifact, and either breaks the role collapse cleanly or gives a bounded negative answer on the broadened surface.
- Expected duration: ~5-15 minutes
- Checkpoint path: `results/safety_alignment/20260318-gemma2it-mediator-conditioned-routing-v2/checkpoints/prompt_residuals/`
- Checkpoint cadence: after each prompt checkpoint
- Log path: `results/safety_alignment/20260318-gemma2it-mediator-conditioned-routing-v2/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: if the mediator-active subset breaks role collapse only because the broadened non-refusal prompts now behave like outright refusals under the old marker rule, the result will be mechanistically interesting but behaviorally muddied.
- Implementation verified: YES - the broadened collection passed registry/safety tests and the validation run already confirmed stable refusal localization and pair accuracy on the same prompt surface.
- Status: LAUNCHING

## [2026-03-18T02:31:00-0500] POST-RUN: aligned Gemma broadened mediator-conditioned routing v2
- Outcome: SUCCESS
- Key metric: the broadened surface still produced an exactly role-collapsed mediator partition (`6` active prompts, all `refusal`; `12` inactive prompts across `harmful_context` and `benign`) despite clean direction discovery and large intervention-conditioned trajectory shifts.
- Artifacts saved: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation/summary.json`, `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation.md`, `results/safety_alignment/20260318-gemma2it-mediator-conditioned-routing-v2/summary.json`, `results/safety_alignment/20260318-gemma2it-mediator-conditioned-routing-v2.md`
- Latest checkpoint: `results/safety_alignment/20260318-gemma2it-mediator-conditioned-routing-v2/checkpoints/prompt_residuals/sa2-confirm-006-refusal.pt`
- Anomalies: the broadened non-refusal prompts stressed the legacy refusal-marker behavior rule, so `resattn-ac2` is now open to make future safety validation tag-aware for refusal-style compliant prompts.
- Next step: close `resattn-mo5` as a bounded negative result and move overall repo priority to `resattn-b4q`.

## [2026-03-18T03:05:00-0500] PRE-RUN: n<<d ridge runtime benchmark
- tmux session: `N/A`
- Script: `scripts/run_ridge_runtime_benchmark.py`
- Command: `.venv/bin/python scripts/run_ridge_runtime_benchmark.py --output results/infrastructure/20260318-gemma2-ridge-runtime-benchmark-b4q/summary.json`
- Config: `pilot_count=96`, `confirm_count=128`, `feature_dim=9216`, `target_dim=53`, `regularization_strength=100.0`, `seed=11`
- What I'm testing: whether the new adaptive ridge helper removes the high-dimensional primal solve bottleneck on a synthetic benchmark matched to the saved Gemma primary-model predictiveness surface.
- Expected outcome: adaptive runtime is materially faster than the legacy primal solve while staying numerically identical within floating-point tolerance.
- Expected duration: ~1-5 minutes
- Checkpoint path: `N/A`
- Checkpoint cadence: `N/A`
- Log path: `results/infrastructure/20260318-gemma2-ridge-runtime-benchmark-b4q/summary.json`
- Resume command: rerun the exact command above
- Main confound to watch: the benchmark is synthetic and isolates the solve step only, so it should be interpreted as a linear-algebra runtime artifact rather than an end-to-end runner wall-clock claim.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_oracle_alpha_controls tests.test_oracle_alpha_runner tests.test_oracle_alpha_campaign` passed after the helper change.
- Status: LAUNCHING

## [2026-03-18T03:07:00-0500] POST-RUN: n<<d ridge runtime benchmark
- Outcome: SUCCESS
- Key metric: the adaptive helper was `92.10x` faster on the leave-one-out-shaped solve and `35.82x` faster on the full pilot-to-confirm fit, with max absolute prediction drift below `2.5e-12`.
- Artifacts saved: `results/infrastructure/20260318-gemma2-ridge-runtime-benchmark-b4q/summary.json`, `results/infrastructure/20260318-gemma2-ridge-runtime-benchmark-b4q.md`
- Latest checkpoint: `N/A`
- Anomalies: none
- Next step: close `resattn-b4q` and shift the remaining oracle-lane infrastructure priority to `resattn-9co`.

## [2026-03-18T03:25:00-0500] PRE-RUN: oracle campaign progress smoke
- tmux session: `N/A`
- Script: `scripts/run_oracle_alpha_predictiveness_campaign.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_predictiveness_campaign.py --collection-id oracle_alpha_phase1_v1 --model-name tiny-stories-1M --device cpu --max-train-sequences 2 --max-eval-sequences 1 --optimization-steps 4 --learning-rate 0.1 --regularization-grid 0.001 0.1 --candidate-feature-sources position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat --candidate-target-names oracle_alpha_logit_vector --output-dir results/infrastructure/20260318-oracle-campaign-progress-smoke-9co > results/infrastructure/20260318-oracle-campaign-progress-smoke-9co/run.log 2>&1`
- Config: `model=tiny-stories-1M`, `collection=oracle_alpha_phase1_v1`, `train=2`, `eval=1`, `regularization_grid=(0.001, 0.1)`
- What I'm testing: whether the campaign runner now writes a live progress artifact and log lines during the summary stage before the final predictiveness summary lands.
- Expected outcome: manifest exists early, `predictiveness_progress.json` is written, and `run.log` captures per-regularization progress lines before final summary completion.
- Expected duration: ~1-3 minutes
- Checkpoint path: `results/infrastructure/20260318-oracle-campaign-progress-smoke-9co/checkpoints/`
- Checkpoint cadence: per prompt checkpoint plus per-regularization progress artifact update
- Log path: `results/infrastructure/20260318-oracle-campaign-progress-smoke-9co/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: the smoke is small and CPU-only, so it validates observability mechanics rather than large-run wall-clock behavior.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_oracle_alpha_campaign tests.test_oracle_alpha_runner` passed after adding the progress callbacks.
- Status: LAUNCHING

## [2026-03-18T03:31:00-0500] POST-RUN: oracle campaign progress smoke
- Outcome: SUCCESS
- Key metric: the smoke wrote `campaign_manifest.json`, `predictiveness_progress.json`, and final summary files together, and `run.log` captured the expected per-regularization progress lines before completion.
- Artifacts saved: `results/infrastructure/20260318-oracle-campaign-progress-smoke-9co/campaign_manifest.json`, `results/infrastructure/20260318-oracle-campaign-progress-smoke-9co/predictiveness_progress.json`, `results/infrastructure/20260318-oracle-campaign-progress-smoke-9co/predictiveness_summary.json`, `results/infrastructure/20260318-oracle-campaign-progress-smoke-9co/oracle_train_run.json`, `results/infrastructure/20260318-oracle-campaign-progress-smoke-9co/oracle_eval_run.json`, `results/infrastructure/20260318-oracle-campaign-progress-smoke-9co.md`
- Latest checkpoint: `results/infrastructure/20260318-oracle-campaign-progress-smoke-9co/checkpoints/`
- Anomalies: the first launch failed because the bracketed feature-source name was not shell-quoted; the rerun fixed that and the smoke completed cleanly.
- Next step: close `resattn-9co` and move the remaining ready queue to `resattn-ac2`.

## [2026-03-18T03:40:00-0500] PRE-RUN: tag-aware broadened refusal-surface validation
- tmux session: `N/A`
- Script: `scripts/run_refusal_feature_discovery_validation.py`
- Command: `.venv/bin/python scripts/run_refusal_feature_discovery_validation.py --collection-id safety_refusal_surface_v2 --device mps --output-dir results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-tag-aware-v1`
- Config: `model=google/gemma-2-2b-it`, `collection=safety_refusal_surface_v2`, `pilot_groups=6`, `confirm_groups=6`, `max_new_tokens=32`
- What I'm testing: whether the tag-aware behavior semantics make the broadened-surface validation artifact reflect intentional refusal-style compliance rather than treating it as failure.
- Expected outcome: direction localization and pair accuracy stay strong, and the broadened non-refusal behavior rates improve relative to the old role-only validator.
- Expected duration: ~5-15 minutes
- Checkpoint path: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-tag-aware-v1/checkpoints/prompt_residuals/`
- Checkpoint cadence: after each prompt checkpoint
- Log path: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-tag-aware-v1/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: if the behavior rates do not improve, then the issue is not just validator semantics and the broadened prompt wording itself is less behaviorally clean than it looked.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_safety_alignment tests.test_prompt_registry` passed after adding tag-aware behavior modes.
- Status: LAUNCHING

## [2026-03-18T04:15:00-0500] POST-RUN: tag-aware broadened refusal-surface validation
- Outcome: SUCCESS
- Key metric: confirm non-refusal behavior pass rate improved from `0.6667` to `0.8333` while refusal/harmfulness localization and confirm pair accuracy stayed unchanged.
- Artifacts saved: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-tag-aware-v1/summary.json`, `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-tag-aware-v1.md`
- Latest checkpoint: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-tag-aware-v1/checkpoints/prompt_residuals/sa2-confirm-006-refusal.pt`
- Anomalies: the tag-aware rerun fixed the old refusal-style false negatives, but some policy-note benign prompts still do not match the current first-person refusal marker and one harmful-context confirm prompt still elicited a genuine refusal-style completion.
- Next step: close `resattn-ac2` and track the narrower policy-style follow-up as `resattn-7km`.

## [2026-03-18T03:28:00-0500] PRE-RUN: policy-style broadened refusal-surface validation
- tmux session: `N/A`
- Script: `scripts/run_refusal_feature_discovery_validation.py`
- Command: `.venv/bin/python scripts/run_refusal_feature_discovery_validation.py --collection-id safety_refusal_surface_v2 --device mps --output-dir results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-policy-style-v1`
- Config: `model=google/gemma-2-2b-it`, `collection=safety_refusal_surface_v2`, `pilot_groups=6`, `confirm_groups=6`, `max_new_tokens=32`
- What I'm testing: whether an explicit policy-style compliant behavior mode removes the remaining benign policy-note false negatives without changing the localized refusal/harmfulness geometry.
- Expected outcome: pilot non-refusal pass reaches `1.0`, confirm non-refusal pass rises above `0.8333`, and the remaining confirm mismatch is the genuinely refusal-like harmful-context prompt rather than the policy-note benign prompt.
- Expected duration: ~5-15 minutes
- Checkpoint path: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-policy-style-v1/checkpoints/prompt_residuals/`
- Checkpoint cadence: after each prompt checkpoint
- Log path: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-policy-style-v1/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: if policy prompts still fail after the new mode lands, the problem is the generated language itself rather than the validator semantics.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_safety_alignment tests.test_prompt_registry` passed after adding policy-style behavior tagging and classification.
- Status: LAUNCHING

## [2026-03-18T03:33:00-0500] PRE-RUN: policy-style broadened refusal-surface validation clean rerun
- tmux session: `N/A`
- Script: `scripts/run_refusal_feature_discovery_validation.py`
- Command: `.venv/bin/python scripts/run_refusal_feature_discovery_validation.py --collection-id safety_refusal_surface_v2 --device mps --output-dir results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-policy-style-v2`
- Config: `model=google/gemma-2-2b-it`, `collection=safety_refusal_surface_v2`, `pilot_groups=6`, `confirm_groups=6`, `max_new_tokens=32`
- What I'm testing: the same policy-style semantics as above, but in a fresh output directory so cached prompt checkpoints from the earlier matcher variant cannot mask the result.
- Expected outcome: pilot non-refusal pass improves above `0.8333`, confirm non-refusal pass stays at least `0.8333`, and the remaining mismatches reduce to genuinely incomplete policy-note outputs plus the refusal-like harmful-context prompt.
- Expected duration: ~5-15 minutes
- Checkpoint path: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-policy-style-v2/checkpoints/prompt_residuals/`
- Checkpoint cadence: after each prompt checkpoint
- Log path: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-policy-style-v2/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: if the clean rerun still drops confirm pass rate, then the policy-style mode is too strict rather than the stale cache being the only problem.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_safety_alignment tests.test_prompt_registry` passed after allowing refusal-like completions to satisfy the policy-style mode.
- Status: LAUNCHING

## [2026-03-18T03:32:00-0500] POST-RUN: policy-style broadened refusal-surface validation
- Outcome: PARTIAL
- Key metric: the first `policy-style-v1` rerun was not interpretable because the output directory reused prompt checkpoints from the earlier matcher variant.
- Artifacts saved: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-policy-style-v1/run.log`
- Latest checkpoint: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-policy-style-v1/checkpoints/prompt_residuals/sa2-confirm-006-refusal.pt`
- Anomalies: same-directory checkpoint reuse masked the semantics change, so the summary could not be treated as a fresh result.
- Next step: rerun the exact policy-style validation in a fresh output directory.

## [2026-03-18T03:38:00-0500] POST-RUN: policy-style broadened refusal-surface validation clean rerun
- Outcome: SUCCESS
- Key metric: pilot non-refusal behavior pass rate improved from `0.8333` to `0.9167`, confirm non-refusal behavior pass rate stayed `0.8333`, and the mechanistic discovery metrics were unchanged.
- Artifacts saved: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-policy-style-v2/summary.json`, `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-policy-style-v2.md`
- Latest checkpoint: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-validation-policy-style-v2/checkpoints/prompt_residuals/sa2-confirm-006-refusal.pt`
- Anomalies: two benign policy-note prompts still produced header-only scaffolds, so the remaining misses now look like completion-completeness problems rather than behavior-classification failures.
- Next step: close `resattn-7km` and track any later prompt-completeness check as `resattn-9us`.

## [2026-03-18T03:45:00-0500] PRE-RUN: policy-note completion-budget sweep
- tmux session: `N/A`
- Script: `scripts/run_refusal_feature_discovery_validation.py`
- Command: `.venv/bin/python scripts/run_refusal_feature_discovery_validation.py --collection-id safety_refusal_surface_v2 --device mps --max-new-tokens 64 --output-dir results/safety_alignment/20260318-gemma2it-refusal-surface-v2-policy-budget-max64-v1`
- Config: `model=google/gemma-2-2b-it`, `collection=safety_refusal_surface_v2`, `pilot_groups=6`, `confirm_groups=6`, `max_new_tokens in {64, 96}`
- What I'm testing: whether the remaining header-only benign policy-note mismatches are caused by the generation cap rather than by prompt-surface semantics.
- Expected outcome: at least one of the two header-only policy-note prompts produces a substantive body at a larger token budget, while refusal/harmfulness localization stays unchanged.
- Expected duration: ~10-25 minutes across both runs
- Checkpoint path: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-policy-budget-max64-v1/checkpoints/prompt_residuals/` and `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-policy-budget-max96-v1/checkpoints/prompt_residuals/`
- Checkpoint cadence: after each prompt checkpoint
- Log path: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-policy-budget-max64-v1/run.log` and `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-policy-budget-max96-v1/run.log`
- Resume command: rerun the exact commands for the affected output directory with `--max-new-tokens 64` or `--max-new-tokens 96`
- Main confound to watch: if completions stay skeletal even at `96` tokens, the issue is prompt framing rather than generation budget.
- Implementation verified: YES - `scripts/run_refusal_feature_discovery_validation.py` already exposes `--max-new-tokens`, and the saved `32`-token artifact isolates the exact prompts to compare against.
- Status: LAUNCHING

## [2026-03-18T04:02:00-0500] POST-RUN: policy-note completion-budget sweep
- Outcome: SUCCESS
- Key metric: increasing the budget to `96` moved pilot non-refusal pass rate from `0.9167` to `1.0`, but confirm non-refusal pass rate stayed `0.8333`, so budget alone does not fully clean the broadened policy-note surface.
- Artifacts saved: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-policy-budget-max64-v1/summary.json`, `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-policy-budget-max96-v1/summary.json`, `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-policy-budget-check-v1.md`
- Latest checkpoint: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-policy-budget-max96-v1/checkpoints/prompt_residuals/sa2-confirm-006-refusal.pt`
- Anomalies: the confirm counterfeiting policy note became substantive at larger budgets but still used prohibition-style institutional language (`It is strictly prohibited to assist...`) that the current matcher does not recognize.
- Next step: close `resattn-9us` and track the narrower matcher follow-up as `resattn-1wr`.

## [2026-03-18T04:08:00-0500] PRE-RUN: prohibition-style policy matcher rerun
- tmux session: `N/A`
- Script: `scripts/run_refusal_feature_discovery_validation.py`
- Command: `.venv/bin/python scripts/run_refusal_feature_discovery_validation.py --collection-id safety_refusal_surface_v2 --device mps --max-new-tokens 96 --output-dir results/safety_alignment/20260318-gemma2it-refusal-surface-v2-prohibition-style-v1`
- Config: `model=google/gemma-2-2b-it`, `collection=safety_refusal_surface_v2`, `pilot_groups=6`, `confirm_groups=6`, `max_new_tokens=96`
- What I'm testing: whether extending the policy-style matcher to prohibition-style institutional language cleans the remaining substantive confirm policy-note miss without changing the mechanistic discovery metrics.
- Expected outcome: confirm non-refusal pass rate rises from `0.8333` to `0.9167`, pilot stays `1.0`, and refusal/harmfulness localization stays unchanged.
- Expected duration: ~10-20 minutes
- Checkpoint path: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-prohibition-style-v1/checkpoints/prompt_residuals/`
- Checkpoint cadence: after each prompt checkpoint
- Log path: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-prohibition-style-v1/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: if confirm pass still stays `0.8333`, then the counterfeiting prompt is not just a matcher-wiring issue and the surface should stop getting semantic follow-ups.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_safety_alignment tests.test_prompt_registry` passed after extending the policy-style matcher regex.
- Status: LAUNCHING

## [2026-03-18T04:14:00-0500] POST-RUN: prohibition-style policy matcher rerun
- Outcome: SUCCESS
- Key metric: confirm non-refusal pass rate improved from `0.8333` to `0.9167` while pilot stayed `1.0`, and the mechanistic discovery metrics were unchanged.
- Artifacts saved: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-prohibition-style-v1/summary.json`, `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-prohibition-style-v1.md`
- Latest checkpoint: `results/safety_alignment/20260318-gemma2it-refusal-surface-v2-prohibition-style-v1/checkpoints/prompt_residuals/sa2-confirm-006-refusal.pt`
- Anomalies: none; the remaining confirm mismatch is now the genuinely refusal-like harmful-context prompt rather than another policy-note semantics gap.
- Next step: close `resattn-1wr` and pivot the next full pass toward `resattn-rh0`.

## [2026-03-18T07:10:40-0500] PRE-RUN: Gemma registry_v5 calibration
- tmux session: `N/A`
- Script: `scripts/run_oracle_alpha_predictiveness_campaign.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_predictiveness_campaign.py --collection-id oracle_alpha_phase1_v1 --model-name google/gemma-2-2b --device mps --output-dir results/oracle_alpha/20260318-gemma2-registry-v5-calibration-v1 --max-train-sequences 8 --max-eval-sequences 8 --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat --candidate-target-names oracle_alpha_logit_vector --regularization-grid 0.0001 0.001 0.01 0.1 1 10 100`
- Config: `model=google/gemma-2-2b`, `registry=prompts/registry_v5.yaml`, `pilot=8`, `confirm=8`, `feature_source=position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat`, `target=oracle_alpha_logit_vector`
- What I'm testing: whether the generated stratified `registry_v5` surface works end-to-end with the current best primary-model oracle method and writes reusable oracle checkpoints, feature caches, and a predictiveness summary on a small four-stratum slice.
- Expected outcome: the calibration covers all strata via round-robin ordering, completes on local MPS, writes checkpoints plus summary artifacts, and can be resumed without recomputing finished prompts.
- Expected duration: ~10-30 minutes
- Checkpoint path: `results/oracle_alpha/20260318-gemma2-registry-v5-calibration-v1/oracle_checkpoints/` and `results/oracle_alpha/20260318-gemma2-registry-v5-calibration-v1/feature_checkpoints/`
- Checkpoint cadence: after each prompt result and each feature vector
- Log path: `results/oracle_alpha/20260318-gemma2-registry-v5-calibration-v1/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: if the new stratified prompt surface is malformed, calibration failures may look like runner bugs even though the real problem is bad prompt generation.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_prompt_registry tests.test_scaffold` passed against the generated `registry_v5` before launch.
- Status: LAUNCHING

## [2026-03-18T07:18:06-0500] PRE-RUN: Gemma registry_v5 calibration rerun after prompt-bank repair
- tmux session: `N/A`
- Script: `scripts/run_oracle_alpha_predictiveness_campaign.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_predictiveness_campaign.py --collection-id oracle_alpha_phase1_v1 --model-name google/gemma-2-2b --device mps --output-dir results/oracle_alpha/20260318-gemma2-registry-v5-calibration-v2 --max-train-sequences 8 --max-eval-sequences 8 --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources 'position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat' --candidate-target-names oracle_alpha_logit_vector --regularization-grid 0.0001 0.001 0.01 0.1 1 10 100`
- Config: `model=google/gemma-2-2b`, `registry=prompts/registry_v5.yaml`, `pilot=8`, `confirm=8`, `feature_source=position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat`, `target=oracle_alpha_logit_vector`
- What I'm testing: whether the repaired `registry_v5` prompt bank still works end to end on the same four-stratum calibration slice after fixing article and noun-phrase issues in the general-text prompts.
- Expected outcome: the v2 calibration keeps the runner behavior from v1 while eliminating obviously malformed general-text prompts from the saved artifact.
- Expected duration: ~10-30 minutes
- Checkpoint path: `results/oracle_alpha/20260318-gemma2-registry-v5-calibration-v2/checkpoints/oracle_runs/` and `results/oracle_alpha/20260318-gemma2-registry-v5-calibration-v2/checkpoints/feature_vectors/`
- Checkpoint cadence: after each prompt result and each feature vector
- Log path: `results/oracle_alpha/20260318-gemma2-registry-v5-calibration-v2/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: if the repaired prompt bank materially changes the small-slice result, the generator cleanup was not merely cosmetic and the full launch should stay blocked until that shift is understood.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_prompt_registry tests.test_scaffold` passed after regenerating `prompts/registry_v5.yaml`.
- Status: LAUNCHING

## [2026-03-18T07:28:00-0500] POST-RUN: Gemma registry_v5 calibration rerun after prompt-bank repair
- Outcome: SUCCESS
- Key metric: oracle mean improvement over uniform `= +1.4363` nats on `8` confirm prompts; predicted mean improvement over uniform `= +0.5896` nats on the same slice.
- Artifacts saved: `results/oracle_alpha/20260318-gemma2-registry-v5-calibration-v2/predictiveness_summary.json`, `results/oracle_alpha/20260318-gemma2-registry-v5-calibration-v2/oracle_eval_run.json`, `results/oracle_alpha/20260318-gemma2-registry-v5-calibration-v2.md`
- Latest checkpoint: `results/oracle_alpha/20260318-gemma2-registry-v5-calibration-v2/checkpoints/oracle_runs/confirm/oa5-confirm-general_text-002-b1de39e66a.json`
- Anomalies: the first calibration attempt exposed malformed general-text noun phrases; this rerun supersedes it after repairing the generator and regenerating `prompts/registry_v5.yaml`.
- Next step: close `resattn-gad` and launch the first larger `registry_v5` Gemma oracle campaign on the fixed method surface.

## [2026-03-18T07:29:30-0500] PRE-RUN: Gemma registry_v5 full stratified campaign
- tmux session: `oa-v5-gemma-full`
- Script: `scripts/run_oracle_alpha_predictiveness_campaign.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_predictiveness_campaign.py --collection-id oracle_alpha_phase1_v1 --model-name google/gemma-2-2b --device mps --output-dir results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1 --optimization-steps 20 --learning-rate 0.1 --seed 11 --candidate-feature-sources 'position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat' --candidate-target-names oracle_alpha_logit_vector --regularization-grid 0.0001 0.001 0.01 0.1 1 10 100`
- Config: `model=google/gemma-2-2b`, `registry=prompts/registry_v5.yaml`, `pilot=256`, `confirm=1024`, `feature_source=position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat`, `target=oracle_alpha_logit_vector`
- What I'm testing: whether the strongest current primary-model oracle method stays positive and useful on the first larger stratified `registry_v5` prompt surface.
- Expected outcome: the full run completes on local MPS with resumable checkpoints, stays clearly positive on oracle loss recovery, and produces a larger held-out predictiveness read across all four strata.
- Expected duration: ~30-120 minutes
- Checkpoint path: `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/checkpoints/oracle_runs/` and `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/checkpoints/feature_vectors/`
- Checkpoint cadence: after each prompt result and each feature vector
- Log path: `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/run.log`
- Resume command: rerun the exact command above
- Main confound to watch: if the larger stratified surface breaks the previously positive routed-loss story, I need to separate a true stratum-sensitive failure from simple prompt-family quality or runtime instability.
- Implementation verified: YES - the repaired `registry_v5` generator is deterministic, the calibration slice stayed positive, and the exact-command rerun reused saved artifacts on the calibration output.
- Status: LAUNCHING

## [2026-03-18T08:26:00-0500] POST-RUN: Gemma registry_v5 full stratified campaign
- Outcome: SUCCESS
- Key metric: oracle mean improvement over uniform `= +1.6299` nats on `1024` confirm prompts; predicted mean improvement over uniform `= +0.8292` nats with `958 / 1024` prompts positive.
- Artifacts saved: `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/summary.json`, `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1.md`
- Latest checkpoint: `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/checkpoints/oracle_runs/confirm/oa5-confirm-general_text-256-*.json`
- Anomalies: the first exact-command rerun was interrupted by a context handoff, so resume reuse was verified again cleanly on the finished output directory; the cached rerun finished in `319.63` seconds versus `2544.10` seconds for the original launch.
- Next step: close `resattn-w39` and move to stratified primary-model pattern analysis on the saved `registry_v5` artifact before spending another large Gemma run.

## [2026-03-18T09:05:00-0500] PRE-RUN: Gemma registry_v5 stratified pattern analysis
- tmux session: `N/A`
- Script: `scripts/run_oracle_alpha_pattern_analysis.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_pattern_analysis.py --run-path results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/oracle_eval_run.json --output results/pattern_analysis/20260318-gemma2-registry-v5-pattern-analysis-v1.json --random-seed 11 --max-clusters 12 --num-resamples 32 --sample-size 192 --registry-path prompts/registry_v5.yaml --collection-id oracle_alpha_phase1_v1 --group-tag-prefix stratum_`
- Config: `model=google/gemma-2-2b`, `source_run=registry_v5 confirm`, `views=raw/source_type/depth_thirds_by_type`, `subset_groups=stratum_*`, `num_resamples=32`, `sample_size=192`
- What I'm testing: whether the larger stratified Gemma oracle artifact strengthens grouped routing structure overall or within specific strata, especially factual recall, without requiring another expensive oracle rerun.
- Expected outcome: grouped structure should stay stronger than raw-source structure overall, and at least one stratum should look materially cleaner than the mixed full-sample artifact.
- Expected duration: ~5-20 minutes
- Checkpoint path: `N/A`
- Checkpoint cadence: `N/A`
- Log path: `results/pattern_analysis/20260318-gemma2-registry-v5-pattern-analysis-v1.log`
- Resume command: rerun the exact command above
- Main confound to watch: the fixed `192`-prompt resample size is intentionally smaller than the old 75% default for the full `1024`-prompt run, so I need to treat resampling comparisons as within-artifact rather than directly comparable to the earlier `128`-prompt studies.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_pattern_analysis` passed after adding subset-summary support for grouped stratum analyses.
- Status: LAUNCHING

## [2026-03-18T09:09:00-0500] POST-RUN: Gemma registry_v5 stratified pattern analysis
- Outcome: SUCCESS
- Key metric: full-sample raw-source structure stayed weak (`silhouette = 0.1664`, `k = 2`, `1023 / 1`), but factual-recall and reasoning-math subsets showed robust raw-source structure (`0.4709` and `0.2456`, both `k = 12`, both resampling oracle-beats-random `= 1.0`).
- Artifacts saved: `results/pattern_analysis/20260318-gemma2-registry-v5-pattern-analysis-v1.json`, `results/pattern_analysis/20260318-gemma2-registry-v5-pattern-analysis-v1.md`
- Latest checkpoint: `N/A`
- Anomalies: none; runtime stayed bounded at `237.16` seconds with the reduced `32 x 192` resampling budget.
- Next step: close `resattn-xot` and pivot the next oracle follow-up toward factual-recall-focused raw-source structure rather than another mixed-surface aggregate rerun.

## [2026-03-18T09:28:00-0500] PRE-RUN: Gemma factual-recall raw-source cluster profile
- tmux session: `N/A`
- Script: `scripts/run_oracle_alpha_subset_cluster_profile.py`
- Command: `.venv/bin/python scripts/run_oracle_alpha_subset_cluster_profile.py --run-path results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/oracle_eval_run.json --registry-path prompts/registry_v5.yaml --collection-id oracle_alpha_phase1_v1 --group-tag stratum_factual_recall --output results/block_structure/20260318-gemma2-factual-recall-cluster-profile-v1.json`
- Config: `model=google/gemma-2-2b`, `subset=stratum_factual_recall`, `view=raw_source`, `top_sources=5`, `example_prompts=3`
- What I'm testing: whether the strong factual-recall raw-source silhouette on the saved `registry_v5` artifact corresponds to clean subcategory structure, source-usage differences, or a more arbitrary cluster split.
- Expected outcome: the best-`k` factual-recall clusters should show interpretable subcategory composition and top-source profiles rather than looking like arbitrary prompt shards.
- Expected duration: ~1-5 minutes
- Checkpoint path: `N/A`
- Checkpoint cadence: `N/A`
- Log path: `results/block_structure/20260318-gemma2-factual-recall-cluster-profile-v1.log`
- Resume command: rerun the exact command above
- Main confound to watch: the best factual-recall `k = 12` split could still reflect prompt-template families rather than deeper routing structure, so I need to separate simple subcategory purity from genuinely different source-usage profiles.
- Implementation verified: YES - `.venv/bin/python -m unittest tests.test_pattern_analysis.PatternAnalysisTests.test_assign_average_linkage_clusters_recovers_expected_partition` passed after adding the reusable assignment helper.
- Status: LAUNCHING

## [2026-03-18T09:29:00-0500] POST-RUN: Gemma factual-recall raw-source cluster profile
- Outcome: SUCCESS
- Key metric: the factual-recall raw-source split stayed strong (`silhouette = 0.4709`, random `= 0.1401`, best `k = 12`) and the clusters were nearly pure by saved factual subcategory rather than arbitrary prompt shards.
- Artifacts saved: `results/block_structure/20260318-gemma2-factual-recall-cluster-profile-v1.json`, `results/block_structure/20260318-gemma2-factual-recall-cluster-profile-v1.md`
- Latest checkpoint: `N/A`
- Anomalies: none; the profile run finished in `4.90` seconds on the saved artifact.
- Next step: close `resattn-8y4` and decide whether the next oracle follow-up should explicitly bridge these factual-recall routing families into the bounded Gemma tool-breakage lane.
