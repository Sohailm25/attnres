ABOUTME: Synthesizes the strongest saved Gemma oracle artifacts into the next truthful primary-model claim.
ABOUTME: Narrows the next oracle follow-up to structure understanding rather than another large rerun.

# Primary-Model Oracle Synthesis After `registry_v5`

## Motivation

`resattn-pjd` exists because the repo's center of gravity moved while the task
ordering lagged behind. The strongest positive result is no longer a
development-model feasibility pass or a bounded side-lane effect. It is the
primary-model Gemma `registry_v5` oracle artifact, together with the saved
pattern-analysis and regime-comparison follow-ups.

The goal of this memo is to say clearly:

- what the strongest truthful claim now is
- which stronger claims remain blocked
- what next oracle analysis would actually sharpen discovery rather than only
  add more volume

## Main Evidence Reviewed

- `results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1.md`
- `results/pattern_analysis/20260318-gemma2-registry-v5-pattern-analysis-v1.md`
- `results/block_structure/20260318-gemma2-factual-recall-cluster-profile-v1.md`
- `results/comparison_regimes/20260318-gemma2-regime-comparison-v1.md`
- `results/tool_breakage/20260318-gemma2-factual-routing-tool-breakage-bridge-v1.md`

## Strongest Supported Claim

The strongest supported claim is now:

> On the primary `google/gemma-2-2b` spine, frozen-model oracle-alpha recovers a
> robust, input-dependent effective depth mixture that materially improves loss
> over uniform and the preregistered nulls, generalizes out of sample on the
> main routed-loss metric, and exhibits non-random structure that is strongest
> at grouped coarse views and inside specific semantic strata.

The key supporting points are:

- primary-model prereg-scale breadth is real:
  - oracle mean improvement over uniform on `registry_v5` confirm:
    `+1.6299` nats
  - predicted mean improvement over uniform:
    `+0.8292` nats
  - predicted positives: `958 / 1024`
  - confirm `R^2 = 0.2392`
  - mean JS to oracle alpha: `0.0985`
- competition matters on the primary spine:
  - softmax-constrained routing beat unconstrained and every tested top-k regime
    on all `128` confirm prompts
- the structure story is real, but narrower than the original raw block
  hypothesis:
  - grouped coarse structure is strong and robust on the full mixed surface
  - full-surface raw-source structure remains weak and aggregate-heavy
  - factual recall and reasoning/math contain real raw-source structure
  - factual recall is the clearest raw-source result in the repo so far

## Stronger Claims Still Blocked

The following claims remain blocked and should stay blocked:

- a global raw-source `~8`-cluster story on the full mixed primary-model surface
- a broad claim that the primary model contains a well-recovered literal router
  variable
- a strong Figure 8 / trained-routing alignment claim on the current local proxy
- a broad prompt-specific same-model Gemma tool-breakage claim
- a broad safety-routing claim beyond the current bounded aligned-Gemma results

## What The Structure Story Actually Is

The current structure story is two-level rather than one-level.

### 1. Full-surface grouped structure is the broad result

Across the mixed `registry_v5` confirm surface:

- `source_type` structure is strong and robust
- `depth_thirds_by_type` structure is also strong and robust
- both grouped views beat the matched random control on every resample

This supports a meaningful coarse-routing claim on the primary model.

### 2. Raw-source structure is stratum-conditioned, not global

The raw-source story is strongest inside semantic strata:

- factual recall:
  - best silhouette `= 0.4709`
  - best `k = 12`
  - nearly pure subcategory clusters
- reasoning and math:
  - best silhouette `= 0.2456`
  - real above-random raw-source structure
- code/procedural and general text:
  - weak raw-source structure

So the right interpretation is not “Gemma has a clean global raw routing
decomposition.” It is “Gemma has robust coarse routing structure broadly, and
sharper raw-source structure inside certain semantic regimes.”

## Why Factual Recall Is The Best Next Oracle Focus

Factual recall is now the best next oracle focus because it satisfies all three
criteria at once:

1. It is the strongest raw-source structure result on the primary model.
2. It already bridges naturally to the bounded Gemma tool-breakage lane.
3. Its clusters appear semantically meaningful rather than merely random or
   template-sharded.

The existing bridge artifact already showed that the old factual tool-breakage
surface only partially overlapped the strongest factual routing families. The
more recent one-token tool-breakage work improved that lane's internal
interpretability, but it did not change the basic strategic point: the oracle
lane is now ahead of the tool-breakage lane in factual-family discovery.

## What Not To Do Next

The next oracle step should **not** be:

- another larger mixed-surface oracle campaign
- another attempt to force a full-surface raw `~8`-cluster story
- a return to Figure 8 rescue work
- a return to pooled tool-breakage reruns

Those would either add less signal than the saved artifacts already contain, or
would tempt the repo into overclaiming.

## Best Next Oracle Follow-Up

The best next oracle follow-up is:

> characterize factual-recall routing modes on the saved `registry_v5` artifact
> more deeply before launching another expensive run.

Concretely, that means a saved-artifact analysis that:

- treats factual recall as the main raw-source discovery surface
- characterizes its internal routing modes in a way that is more mechanistic
  than “subcategory purity”
- explicitly distinguishes:
  - family identity (`capital`, `element`, `author`, `moon`)
  - within-family route modes
  - their source-type and top-source signatures
- decides whether the next cross-lane bridge should target:
  - capitals/elements/authors as the cleanest bridge families
  - or reasoning/math as the next independent oracle discovery lane

## Decision

`resattn-pjd` should be read as a synthesis and narrowing step, not as a call
for another broad rerun.

The main lane from here should be:

1. primary-model oracle synthesis
2. factual-recall route-mode characterization
3. only then selective extension into tool-breakage or safety where the saved
   oracle structure actually points

That ordering is the strongest alignment with both the original thesis and the
current evidence.
