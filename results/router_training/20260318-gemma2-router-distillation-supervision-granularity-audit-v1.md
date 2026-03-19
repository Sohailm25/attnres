ABOUTME: Audits held-out router-distillation errors on the frozen Gemma pilot baseline.
ABOUTME: Uses prompt-level diagnostics to decide whether the next honest Phase 6 move is richer supervision granularity or simply more data.

# Gemma-2 Router-Distillation Supervision-Granularity Audit v1

## Motivation

`resattn-zic` closed the obvious architecture escape hatches on the saved Gemma
pilot export:

- the current best baseline is still
  `linear + h_4[t] + oracle_alpha_logit_vector + mean_token_logits_then_softmax`
- held-out `R^2 = 0.3445`
- held-out mean JS `= 0.0853`

That result matters, but it leaves the Phase 6 lane far below the prereg
readiness gate (`R^2 > 0.5`).

`resattn-but` asks a narrower question than another width or family sweep:

- do the remaining held-out errors mostly track coarse prompt strata?
- or do they mostly track simple scalar difficulty surrogates like prompt
  length or oracle entropy?

If the errors are stratum-structured and weakly explained by those scalars, the
next honest move is richer token/span supervision rather than another blind
architecture tweak or a generic “just add more prompts” story.

## Methods

- Input dataset:
  `results/router_training/20260318-gemma2-router-distillation-pilot-export-v1/`
- Frozen baseline selection source:
  `results/router_training/20260318-gemma2-router-distillation-family-comparison-v1/summary.json`
- Frozen baseline:
  - router family: `linear`
  - input: `h_4[t]`
  - target: `oracle_alpha_logit_vector`
  - aggregation: `mean_token_logits_then_softmax`
- Frozen split:
  - exact `192 / 64` train/eval prompt IDs from the saved family-comparison
    artifact
- Optimization:
  - Adam
  - learning rate `0.001`
  - weight decay `0.0001`
  - batch size `16`
  - max epochs `300`
  - patience `40`
  - seed `11`
- Device:
  - `mps`
- Prompt-level diagnostics saved for each held-out prompt:
  - mean JS divergence to oracle alpha
  - target-space MSE
  - prompt token count
  - oracle entropy
  - oracle top-1 mass
- Grouping / summary:
  - grouped by saved `stratum_*` tag
  - scalar correlations computed against held-out prompt-level mean JS

Machine-readable artifact:

- `results/router_training/20260318-gemma2-router-distillation-supervision-granularity-audit-v1/summary.json`

Implementation checks:

- the refit baseline exactly matched the saved family-comparison aggregate on
  the same split:
  - held-out `R^2 = 0.3445`
  - held-out mean JS `= 0.0853`
- train/eval prompt IDs matched the frozen family artifact exactly
- the exact-command rerun preserved the `summary.json` hash on MPS:
  `e55d9afc17c33a0b3d4ad3961a3d443cea7895dc`

## Results

Held-out baseline check:

- eval loss `= 0.5808`
- held-out `R^2 = 0.3445`
- held-out mean JS `= 0.0853`

Stratum-level held-out mean JS:

| Stratum | Prompts | Mean JS | Mean target MSE | Mean tokens |
|---|---:|---:|---:|---:|
| `stratum_factual_recall` | `16` | `0.0409` | `0.4022` | `10.06` |
| `stratum_reasoning_math` | `16` | `0.0576` | `0.5598` | `22.44` |
| `stratum_code_procedural` | `16` | `0.0627` | `0.6215` | `16.69` |
| `stratum_general_text` | `16` | `0.0754` | `0.7396` | `13.12` |

Key summary values:

- worst stratum: `stratum_general_text`
- stratum mean-JS range: `0.0345`
- strongest scalar correlation with held-out JS:
  - attribute: `oracle_top1_mass`
  - absolute Pearson `= 0.1654`

Other scalar correlations:

- `num_tokens`: `0.0169`
- `oracle_entropy`: `-0.1237`

High-error prompts were spread across multiple non-factual regimes rather than
collapsing to one trivial length or entropy band. The worst prompts included:

- code / python snippet:
  - `oa5-pilot-code_procedural-008`, JS `= 0.1441`
  - `oa5-pilot-code_procedural-012`, JS `= 0.1209`
- general text / scene prompts:
  - `oa5-pilot-general_text-011`, JS `= 0.1206`
  - `oa5-pilot-general_text-052`, JS `= 0.1188`

The factual stratum remained the easiest held-out regime, while the largest
residual errors sat in general-text and code prompts.

## Interpretation

This audit supports the supervision-granularity hypothesis more than the
“another architecture scalar” or “just add more prompts” hypotheses.

The truthful read is:

- held-out error is materially stratified on the frozen baseline
- that stratification is not well explained by prompt length
- it is also not well explained by oracle entropy or oracle top-1 mass
- factual prompts are already relatively easy for the current sequence-level
  supervision object
- the remaining misses concentrate in prompt types where a single
  sequence-level alpha target is most plausibly too coarse

So the next honest Phase 6 move should be:

- richer token/span supervision on the same pilot surface

It should **not** be:

- another width sweep
- another family sweep
- a generic larger train/eval surface without changing the supervision object

This audit does **not** yet prove which richer supervision object is right. It
only clears a narrower question: the remaining error pattern now looks more
like a supervision-granularity problem than a simple capacity/family problem.

## Limitations

- This is still pilot-only and still uses a held-out split inside the pilot
  export, not the confirmatory surface.
- The current export only contains sequence-level oracle targets, so this audit
  can diagnose the likely blocker but cannot itself compare token-level oracle
  supervision objects yet.
- Scalar correlations were intentionally narrow and only used saved coarse
  attributes; they do not exhaust every possible prompt descriptor.

## Next Steps

- Close `resattn-but` as a real supervision-granularity diagnosis on the frozen
  pilot baseline.
- Take a new Phase 6 issue to augment the pilot supervision surface with a
  richer token/span target and compare it directly against the retained
  sequence-level baseline on the same saved prompt split.
