# ABOUTME: Summarizes the first primary-model Gemma oracle routing-regime comparison.
# ABOUTME: Records the prereg softmax versus unconstrained versus top-k result after fixing the initial top-k support-search artifact.

## Motivation

`resattn-5eo` asked for the prereg-required routing-regime comparison on the
primary frozen-model spine. The question is not just whether non-uniform
routing helps on `google/gemma-2-2b`, but whether the softmax-constrained
AttnRes-style competition regime actually outperforms or meaningfully separates
from the matched-initialization unconstrained and top-k families.

## Methods

- Model: `google/gemma-2-2b`
- Collection: `oracle_alpha_phase1_v1`
- Split: locked `confirm`
- Size: `128` prompts
- Sources: `53`
- Optimization: `20` Adam steps, `lr = 0.1`, `seed = 11`
- Initialization:
  - matched `z = 0` start for all regimes
  - softmax-constrained starts from the uniform simplex point
  - unconstrained starts from `0.5` per source under sigmoid gating
- Regimes:
  - softmax-constrained
  - unconstrained sigmoid gating
  - top-k with `k ∈ {2, 4, 8, 13, 26}`

### Implementation note

The first top-k pass exposed a real optimization artifact: hard top-k masking
with exact zero initialization fixed the support to the arbitrary tie-broken
initial indices, so the early sparse results were not scientifically usable.

The final artifact keeps the prereg forward definition exact
(`α = softmax(top_k(z, k))`) but uses a dense-softmax straight-through gradient
path so masked-out logits can still receive support-search gradients from the
matched `z = 0` start. The saved results below are all from the repaired run.

## Results

### Main comparison

| Regime | Mean improvement over uniform | 95% bootstrap CI | Positive prompts | Mean effective sources | Mean top-1 mass | Mean Gini |
|---|---:|---|---:|---:|---:|---:|
| softmax-constrained | `2.5525` | `[2.4603, 2.6442]` | `128 / 128` | `34.0719` | `0.0810` | `0.5111` |
| unconstrained | `2.0637` | `[1.9858, 2.1402]` | `128 / 128` | `45.5897` | `0.0347` | `0.3034` |
| top-k `k = 2` | `0.0115` | `[0.0012, 0.0260]` | `6 / 128` | `50.6094` | `0.0417` | `0.0451` |
| top-k `k = 4` | `0.2032` | `[0.1461, 0.2693]` | `47 / 128` | `35.0066` | `0.1081` | `0.3399` |
| top-k `k = 8` | `0.6765` | `[0.5798, 0.7834]` | `110 / 128` | `14.2356` | `0.1418` | `0.7386` |
| top-k `k = 13` | `1.0891` | `[0.9914, 1.1922]` | `124 / 128` | `13.5528` | `0.1401` | `0.7725` |
| top-k `k = 26` | `1.4664` | `[1.3722, 1.5689]` | `128 / 128` | `22.9511` | `0.1020` | `0.6416` |

### Softmax advantage over the alternatives

| Comparison | Mean softmax advantage over the alternative | Prompts where softmax won |
|---|---:|---:|
| softmax vs unconstrained | `0.4888` | `128 / 128` |
| softmax vs top-k `k = 2` | `2.5410` | `128 / 128` |
| softmax vs top-k `k = 4` | `2.3493` | `128 / 128` |
| softmax vs top-k `k = 8` | `1.8760` | `128 / 128` |
| softmax vs top-k `k = 13` | `1.4634` | `128 / 128` |
| softmax vs top-k `k = 26` | `1.0861` | `128 / 128` |

## Interpretation

- The primary prereg comparison is now clear on the primary model:
  softmax-constrained routing beats the matched-initialization unconstrained
  regime and every tested top-k regime on all `128` confirm prompts.
- This is the strongest primary-model evidence so far that zero-sum competition
  over depth is not just compatible with the oracle signal but materially better
  than independent gating on the locked Gemma surface.
- The top-k family shows a monotonic sparsity-performance tradeoff:
  wider support recovers more loss, but even half-depth support (`k = 26`)
  remains well below the dense softmax regime.
- Unconstrained routing is competitive but meaningfully weaker. It recovers
  large loss improvements while staying much denser (`45.5897` effective
  sources versus `34.0719` for softmax), which fits the interpretation that the
  Gemma oracle signal benefits from competition rather than mere per-layer gain
  adjustment.
- The `k = 2` and `k = 4` sparsity summaries need careful reading: those regime
  summaries use `best_alpha`, and on many prompts the best loss never beat the
  uniform null, so the saved best state remains the uniform fallback rather than
  a sparse routed solution. The key metric there is the very low positive-prompt
  count, not the apparent density of the fallback state.

## Limitations

- This is still the current oracle development slice over final-output residual
  sources, not the later full claim-bearing multi-layer analysis surface.
- The top-k path now uses a straight-through gradient estimator for support
  search from the matched `z = 0` start. The forward regime is exact top-k, but
  the optimizer is not a pure projected-gradient method.
- The result is sequence-level and confirm-split clean, but it is still only one
  model/task surface. It does not by itself establish a universal regime law.

## Next Steps

- Close `resattn-5eo` as the primary-model prereg regime-comparison pass.
- Move to `resattn-1lk` next and decide conservatively whether the strong Figure
  8 lane stays frozen or is revisited only through a materially more faithful
  proxy path.
- Keep `resattn-mo5` as the next experimental extension lane after that.
