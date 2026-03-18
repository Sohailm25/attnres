ABOUTME: Synthesizes the strongest saved primary-model Gemma oracle artifacts into one truthful claim surface.
ABOUTME: Re-centers the repo narrative on effective depth mixture, competition, and stratum-conditioned structure rather than a forced global raw-cluster story.

# Gemma-2 `registry_v5` Oracle Synthesis v1

## Motivation

`resattn-qxz` exists because the repo now has enough saved primary-model
Gemma results that the main risk is no longer “lack of signal.” The main risk
is narrative drift: over-centering weaker side lanes, aggregate raw clustering,
or older development-model artifacts when the strongest evidence is already on
the primary spine.

This synthesis uses only saved primary-model artifacts:

- `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1.md`
- `results/pattern_analysis/20260318-gemma2-registry-v5-pattern-analysis-v1.md`
- `results/block_structure/20260318-gemma2-factual-recall-cluster-profile-v1.md`
- `results/comparison_regimes/20260318-gemma2-regime-comparison-v1.md`

## Core Results

### 1. The primary Gemma oracle result is now strong, broad, and held-out

On `google/gemma-2-2b` over the saved `registry_v5` confirm split
(`1024` prompts), oracle-alpha is clearly better than uniform and every
preregistered null:

- mean oracle improvement over uniform: `+1.6299` nats
- bootstrap interval: `[1.5865, 1.6758]`
- positive prompts: `1024 / 1024`
- mean losses:
  - `uniform = 4.7341`
  - `random_dirichlet = 9.4581`
  - `magnitude_proportional = 6.9846`
  - `last_layer_only = 20.9150`
  - `optimized = 3.1042`

Held-out predictiveness is also now materially positive on the primary spine:

- predicted mean improvement over uniform: `+0.8292` nats
- predicted bootstrap interval: `[0.7805, 0.8758]`
- predicted positive prompts: `958 / 1024`
- confirm `R^2 = 0.2392`
- mean JS to oracle alpha: `0.0985`

This is the strongest direct support in the repo for the core frozen-model
claim: standard Gemma representations expose a recoverable effective depth
mixture that generalizes out of sample on a large confirm surface.

### 2. The primary structure story is two-level, not one global raw-cluster story

The full mixed-surface artifact is strongest on grouped structure:

- full-sample `source_type` silhouette: `0.7558` vs random `0.4681`
- full-sample `depth_thirds_by_type` silhouette: `0.5168` vs random `0.1433`
- both grouped views beat the matched random control on `32 / 32` resamples

But the full mixed-surface raw-source view is still weak and aggregate-heavy:

- raw-source oracle best silhouette: `0.1664`
- raw-source random best silhouette: `0.1417`
- best `k = 2`
- cluster sizes: `1023 / 1`
- raw-source oracle-beats-random resampling fraction: `0.0938`

So the truthful raw-source claim is not “Gemma shows a clean global `~8`-cluster
depth-routing decomposition.” That gate is still unpassed.

### 3. Raw-source structure is concentrated inside semantic strata

The stratified `registry_v5` analysis shows a clear split:

| Stratum | Oracle improvement | Predicted improvement | Mean JS | Raw-source silhouette | Raw-source interpretation |
|---|---:|---:|---:|---:|---|
| factual recall | `+2.2488` | `+1.6297` | `0.0580` | `0.4709` | strongest and cleanest raw-source structure |
| reasoning/math | `+0.8613` | `+0.4848` | `0.0803` | `0.2456` | real but weaker raw-source structure |
| code/procedural | `+1.4595` | `+0.5376` | `0.1181` | `0.1339` | weak raw-source structure |
| general text | `+1.9501` | `+0.6645` | `0.1374` | `0.1301` | weak raw-source structure |

All four strata retain strong grouped coarse structure, but only factual recall
and reasoning/math show strong raw-source structure above the matched random
control.

### 4. Factual recall is the clearest bridge between oracle structure and downstream interpretation

The factual-recall subset is the strongest raw-source result on the primary
model:

- raw-source silhouette: `0.4709` vs random `0.1401`
- best `k = 12`
- cluster sizes:
  `33 / 32 / 32 / 32 / 18 / 16 / 16 / 16 / 16 / 16 / 16 / 13`
- largest cluster fraction: `0.1289`

The resulting raw clusters are semantically coherent:

- capitals occupy `3` clusters
- elements occupy `3` clusters
- authors occupy `5` clusters
- moons occupy `2` clusters
- all clusters are pure except one `33`-prompt moon cluster with a single
  author outlier

This does not rescue the global raw block-structure prereg gate. It does
establish something narrower and still meaningful: the primary-model raw-source
signal is semantically organized and internally differentiated inside factual
recall.

### 5. Competition is now a core result, not a side result

The prereg regime comparison on the primary spine is clean:

- softmax-constrained mean improvement over uniform: `+2.5525`
- unconstrained mean improvement: `+2.0637`
- top-k mean improvements:
  - `k = 2`: `+0.0115`
  - `k = 4`: `+0.2032`
  - `k = 8`: `+0.6765`
  - `k = 13`: `+1.0891`
  - `k = 26`: `+1.4664`

Softmax beats every alternative on all `128 / 128` confirm prompts in the saved
regime artifact. That means the repo already has strong primary-model evidence
that zero-sum competition over depth is meaningfully better than independent
gating or sparse top-k truncation on this surface.

## Supported Claims

The repo can now support these claims on the primary frozen-model spine:

1. `google/gemma-2-2b` exposes a recoverable input-dependent effective depth
   mixture that improves next-token loss over uniform routing and all saved nulls
   on a large held-out prompt surface.
2. Held-out predictiveness is real on the primary model. The current feature and
   target path does not merely fit descriptively on the training split.
3. Competitive softmax-constrained routing is materially better than the matched
   unconstrained and tested top-k alternatives on the primary model.
4. Routing structure is broad at the grouped/coarse level across the mixed
   prompt surface.
5. Raw-source structure on the primary model is strongest inside tighter
   semantic strata, especially factual recall and secondarily reasoning/math.
6. Factual recall contains semantically coherent and internally differentiated
   raw routing modes on the primary model.

## Claims Still Blocked

The repo still cannot honestly claim:

1. that frozen-model oracle-alpha reveals a literal trained router variable
2. that the primary model shows a clean global raw `~8`-cluster block structure
3. that frozen-model routing is already aligned with trained AttnRes routing
   beyond descriptive comparison
4. that Figure 8 alignment is strongly established on a faithful trained-routing
   proxy
5. that the strongest same-model Gemma tool-breakage claim is broadly
   donor-arm-positive
6. that safety-routing differences generalize beyond the current aligned-Gemma
   refusal workflow boundary
7. that learned `w_l` analog geometry or trained router distillation is already
   characterized

## Interpretation

The original thesis path has tightened.

The main story is now:

- effective depth mixture on the primary Gemma spine is real
- competition over depth matters
- grouped structure is broad
- raw-source structure is not broad in the same way; it is strongest inside
  factual recall and other tighter semantic strata

That is a better and more truthful story than trying to force a single global
raw-cluster narrative. It is also more novel, because it says the interpretably
structured routing signal on the primary model lives most clearly in specific
semantic slices rather than as one uniform mixed-surface phenomenon.

## Next Steps

- Close `resattn-qxz` on this synthesis pass.
- Keep the main lane centered on the saved primary-model oracle artifacts.
- If the oracle lane moves next, prefer stratum-conditioned analysis or bridge
  work over another broad rerun.
- Keep tool-breakage, safety, and Figure 8 as bounded extension lanes rather
  than the narrative center of the project.
