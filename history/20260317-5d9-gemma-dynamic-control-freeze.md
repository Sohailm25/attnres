# ABOUTME: Records the post-g09 decision on whether the Gemma tool-breakage lane needs a finer dynamic control.
# ABOUTME: Freezes the current same-model claim boundary unless a new control-design reason appears later.

## Question

After the mixed `resattn-g09` counterfactual result, should the repo spend more
time on a finer same-model Gemma dynamic control now?

## Evidence

The current counterfactual already separates the two relevant objections:

- fixed non-uniform control is weaker than the routed trace
  - routed minus `pilot_mean_alpha` tuned mean KL `= +0.6091`
- prompt-misaligned dynamic control is at least as damaging as the routed trace
  - routed minus `prompt_permuted_alpha` tuned mean KL `= -0.3082`
  - routed minus `prompt_permuted_alpha` tuned final-position KL `= -0.6541`

The rank-based confirm metrics tell the same story:

- versus `prompt_permuted_alpha`, routed worsens final target rank on only
  `4 / 8` prompts
- versus `prompt_permuted_alpha`, routed increases target-rank range on only
  `2 / 8` prompts

That is already enough to block the strong same-model claim that prompt-matched
input-dependent routing is uniquely responsible for the damage.

## Decision

Do **not** run a finer same-model Gemma dynamic control now.

Freeze the current Gemma claim boundary at:

- routing degrades the original-model baseline on the locked confirm split
- the degradation is stronger than a fixed non-uniform control
- the current prompt-misaligned dynamic control is at least as damaging as the
  prompt-matched routed trace on the tuned primary metric

## Why No Follow-Up Now

1. The prereg requirement is already satisfied.
   `resattn-g09` implemented the required controlled dynamic-routing
   counterfactual.

2. A finer control would currently look like goalpost movement.
   The prompt-permuted control did not merely produce an ambiguous tie; it
   actively came out as more damaging on the tuned primary KL metric.

3. The current confirm split is small.
   With only `8` prompts, a more intricate donor-matching or resampling control
   would be easy to over-interpret and hard to defend as the new decisive test.

## Reopen Condition

Only revisit this lane if a new control is motivated by a real methodological
defect in the current counterfactual, for example:

- evidence that cyclic prompt permutation is invalid for reasons beyond being
  strong
- a prereg or reviewer requirement for semantic donor matching that changes the
  scientific target explicitly
- a larger confirm surface that can support a stronger same-model claim without
  looking like post-hoc rescue

Absent one of those conditions, the Gemma lane should stay frozen at the current
claim boundary and repo effort should move elsewhere.
