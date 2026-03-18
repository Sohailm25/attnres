ABOUTME: Registers the first pilot-only per-token export artifact for Gemma router distillation.
ABOUTME: Documents the exact exported supervision surface and the resume check that makes the Phase 6 dataset durable.

# Gemma-2 Router-Distillation Pilot Export v1

## Motivation

`resattn-5qd` showed that the saved Gemma `registry_v5` oracle campaign was
scientifically ready for Phase 6 but still operationally blocked: the repo had
sequence-level oracle targets and aggregated prompt features, but no saved
per-token router inputs.

`resattn-1ot` answers the smallest honest follow-up:

- export pilot-only token ids and per-token hidden states from the saved Gemma
  oracle campaign
- join them to the saved sequence-level `final_alpha` targets and prompt
  metadata
- verify that the export is resumable on the completed output directory

## Methods

- Source oracle campaign:
  `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/`
- Export module:
  `validation/router_training_export.py`
- Export script:
  `scripts/export_router_distillation_pilot_dataset.py`
- Model:
  `google/gemma-2-2b`
- Device:
  `mps`
- Collection:
  `oracle_alpha_phase1_v1`, pilot split only
- Exported fields per prompt checkpoint:
  - `prompt_id`
  - `prompt`
  - `split`
  - `target_text`
  - `tags`
  - `perturbations`
  - `token_ids`
  - `h_1[t]`
  - `h_4[t]`
  - `source_labels`
  - `final_alpha`

Machine-readable artifacts:

- `results/router_training/20260318-gemma2-router-distillation-pilot-export-v1/summary.json`
- `results/router_training/20260318-gemma2-router-distillation-pilot-export-v1/dataset_manifest.json`

## Results

The export succeeded on the full saved pilot surface without rerunning oracle
optimization.

Dataset surface:

- prompts exported: `256`
- total tokens exported: `4013`
- mean tokens per prompt: `15.6758`
- token range: `5` to `34`
- model width: `2304`
- routing sources per prompt: `53`
- strata:
  - factual recall: `64`
  - reasoning/math: `64`
  - code/procedural: `64`
  - general text: `64`
- saved perturbation names:
  - `prompt_paraphrase` on all `256` prompts

Operational checks:

- the export writes ignored per-prompt `.pt` checkpoints under
  `results/router_training/20260318-gemma2-router-distillation-pilot-export-v1/checkpoints/prompt_exports/`
- the committed manifest and summary fully describe the saved training fields
- after adding `tags` and `perturbations` to the payload schema, rerunning the
  exact export command recomputed stale prompt checkpoints in place
- an exact-command rerun against the completed output directory reused the
  refreshed checkpoints with an unchanged sample hash
  (`6dd29ee0a297d5c085c4e895aa4a64a624aa1d41`) and completed in `10.07s`

Interpretation:

- Phase 6 is no longer blocked by missing per-token supervision on the saved
  Gemma pilot surface
- the next honest move is pilot router distillation, not more export plumbing

## Limitations

- This is still pilot-only. No confirm-split export exists yet.
- The target remains sequence-level `final_alpha`; this artifact does not
  invent a token-level oracle target.
- No router was trained here.
- No `w_l`-analog geometry claim is enabled by export alone.

## Next Steps

- Close `resattn-1ot`.
- Start `resattn-m6r`: fit the prereg pilot router on the exported dataset and
  compare `h_1[t]` against `h_4[t]` before any confirmatory router training.
