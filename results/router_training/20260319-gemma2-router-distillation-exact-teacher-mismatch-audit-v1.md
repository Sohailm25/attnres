# ABOUTME: Diagnoses why exact tokenwise Gemma teachers fail on the bounded saved subset.
# ABOUTME: Reuses the 7xo subset and compares within-prompt teacher variance against sequence-level mismatch without retraining the router.

# Gemma-2 Router-Distillation Exact-Teacher Mismatch Audit v1

## Motivation

`resattn-7xo` answered the first fair tokenwise-teacher question:

- exact tokenwise oracle teachers did **not** beat the retained repeated
  sequence-level target on the bounded `32 / 16` subset

But that still left the mechanism unclear. There were two main possibilities:

- the exact tokenwise teachers vary too much within each prompt for a
  sequence-level decoder to absorb cleanly
- or the exact tokenwise teachers are simply being aggregated the wrong way when
  we score a sequence-level alpha object

`resattn-b4h` isolates that mechanism on the same frozen subset rather than
  launching another expensive teacher run blindly.

## Methods

- Input dataset:
  `results/router_training/20260318-gemma2-router-distillation-pilot-export-v1/`
- Frozen subset source:
  `results/router_training/20260319-gemma2-router-distillation-exact-teacher-subset-v1/summary.json`
- Prompt surface:
  - train: `32` prompts
  - eval: `16` prompts
  - total audited prompts: `48`
  - balanced across:
    - `stratum_factual_recall`
    - `stratum_reasoning_math`
    - `stratum_code_procedural`
    - `stratum_general_text`
- Target:
  - `oracle_alpha_logit_vector`
- Exact-teacher reconstruction:
  - per-position routed-loss optimization under Gemma-2's own final norm and
    logits softcap
  - optimization steps per token: `30`
  - learning rate: `0.1`

Per-prompt diagnostics:

- tokenwise exact-teacher alpha distributions on valid next-token positions
- mean-aggregated exact-teacher alpha
- last-position exact-teacher alpha
- repeated prompt-level sequence target
- final saved sequence-level oracle alpha

Key summary metrics:

- `within_prompt_js_to_mean_teacher`:
  average tokenwise JS to the prompt's own mean exact teacher
- `mean_teacher_js_to_oracle`:
  JS between mean-aggregated exact teacher and saved sequence-level oracle alpha
- `last_teacher_js_to_oracle`:
  JS between last-position exact teacher and saved sequence-level oracle alpha
- entropy gap:
  mean exact-teacher entropy minus sequence-level oracle entropy

Machine-readable artifact:

- `results/router_training/20260319-gemma2-router-distillation-exact-teacher-mismatch-audit-v1/summary.json`

Implementation checks:

- focused unit tests cover the new mismatch-summary classification and a
  synthetic end-to-end audit case
- the exact-command rerun preserved the `summary.json` hash on MPS:
  `58819a5bf2eb7b34a3d90fe8959fbbc2bf91e3a4`

## Results

Overall audited subset summary:

- dominant failure mechanism:
  `within_prompt_teacher_variance`
- recommended next step:
  `do_not_expand_exact_tokenwise_teacher_work`
- mean within-prompt tokenwise JS to prompt-mean teacher:
  `0.1292`
- mean prompt-mean-teacher JS to sequence oracle:
  `0.0832`
- mean last-position-teacher JS to sequence oracle:
  `0.1910`
- mean exact-teacher entropy minus oracle entropy:
  `+0.3727`
- mean tokenwise top-1 agreement with sequence-oracle top-1 source:
  `0.0446`
- mean tokenwise top-1 agreement with prompt-mean-teacher top-1 source:
  `0.0805`

By stratum:

| Stratum | Prompt count | Mean within-prompt JS to mean teacher | Mean mean-teacher JS to oracle | Mean last-teacher JS to oracle | Mean entropy gap |
|---|---:|---:|---:|---:|---:|
| `stratum_code_procedural` | `12` | `0.0915` | `0.0782` | `0.1295` | `0.3704` |
| `stratum_factual_recall` | `12` | `0.1336` | `0.0846` | `0.2038` | `0.4074` |
| `stratum_general_text` | `12` | `0.1438` | `0.0776` | `0.2084` | `0.3468` |
| `stratum_reasoning_math` | `12` | `0.1479` | `0.0923` | `0.2223` | `0.3661` |

Selected worst prompts by prompt-mean-teacher JS to oracle were mostly
reasoning/math and factual prompts, but the dominant pattern was not one
isolated stratum. It was broad tokenwise disagreement inside prompts.

Runtime:

- first audit run wall-clock: `640.11s`
- exact-command rerun wall-clock: `585.81s`

## Interpretation

This is the cleanest diagnosis we have had on the tokenwise-teacher lane.

The main failure is **not** that we are obviously aggregating tokenwise teachers
the wrong way at sequence level. If that were true, the last-position teacher
should have looked materially better than the mean teacher. It does not. It is
much worse.

The stronger explanation is:

- tokenwise exact teachers disagree substantially within prompts
- averaging those teachers still gets you closer to the scored sequence-level
  oracle than any single last-position teacher
- but the averaged exact teachers are still noticeably more diffuse than the
  scored sequence-level oracle

So the truthful read is:

- the exact-teacher path is asking the router to fit contradictory token-level
  labels inside each prompt
- mean-token sequence decoding is already the less-bad sequence-level summary on
  this surface
- additional exact-teacher work is not justified from current evidence

This sharpens the next Phase 6 move. The repo should prefer **coarser,
competition-preserving compression targets** over more tokenwise teacher
fidelity. That is exactly why the queued follow-up is now the embedding-isolated
block-compressed target comparison, not another tokenwise export redesign.

## Limitations

- This remains a bounded saved subset, not the full `192 / 64` pilot split.
- The audit reconstructs exact teachers again from the frozen subset config, so
  it inherits the same routed-loss teacher definition as `resattn-7xo`.
- This does not prove that every conceivable tokenwise teacher would fail. It
  shows that the exact next-token teacher on this saved surface is not the right
  next investment.

## Next Steps

- Close `resattn-b4h` as a successful diagnosis pass.
- Keep the retained broader Phase 6 baseline unchanged:
  `mlp + h_4[t] + oracle_alpha_logit_vector + mean_token_logits_then_softmax + all_tokens_target_mse`
- Do **not** expand exact-tokenwise-teacher work.
- Unblock `resattn-y4m` and move next to the embedding-isolated
  block-compressed target comparison.
