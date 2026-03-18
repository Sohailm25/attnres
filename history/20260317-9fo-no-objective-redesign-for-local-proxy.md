# ABOUTME: Records the post-bux decision on whether the local Figure 8 proxy should take an objective-level redesign.
# ABOUTME: Declines custom objective changes on the current tiny proxy in favor of an explicit claim-boundary freeze.

## Question

After the negative regularization sweep, what is the smallest honest
objective-level Figure 8 redesign for the current local Block AttnRes proxy?

## Decision

Do **not** introduce an objective-level redesign on the current tiny local
proxy.

Instead:

- freeze the strong Figure 8 / trained-routing alignment lane at the current
  descriptive boundary on this proxy
- treat the next Figure 8 strategic question as whether to revisit the lane with
  a more faithful proxy, not with a custom loss on the current one

The follow-up issue for that strategic choice is `resattn-1lk`.

## Why No Objective Change Is Honest Right Now

1. The current proxy is already fidelity-limited.
   It is a tiny local reproduction meant to act as a believable small AttnRes
   proxy. The obvious objective tweaks now available are not paper-faithful
   training details; they are custom interventions chosen because the current
   proxy does not show the desired pattern.

2. The repo has already exhausted the faithful stabilization path.
   Tokenization, width, horizon, corpus, best-checkpoint export, and matched
   regularization have all been tested.

3. A custom objective would make a positive result harder to trust.
   The most natural next losses would be things like routing-entropy shaping or
   direct Figure-8-facing auxiliary terms. Those would move the paper-facing
   metrics by construction and would weaken the claim that the proxy
   independently recovered them.

## Consequence

The current local proxy supports:

- an operationally real small trained-routing reproduction
- descriptive comparison against the published Figure 8 pattern surface

It does **not** currently support:

- a strong claim that the local trained-routing proxy independently recovers the
  published Figure 8 signatures

## Reopen Condition

Only revisit the strong Figure 8 lane through one of these more faithful paths:

- a larger or otherwise more faithful local AttnRes reproduction
- another open depth-mixing comparison with released weights, if explicitly
  logged as fallback-only
- some new external evidence that a specific objective change is part of the
  real AttnRes training recipe rather than a post-hoc proxy rescue

Absent one of those conditions, the repo should keep the current local proxy at
the descriptive boundary and stop adding custom objective tweaks.
