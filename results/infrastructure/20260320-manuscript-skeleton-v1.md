ABOUTME: Provides the first manuscript skeleton with section order, claim placement, and caveat placement.
ABOUTME: Uses the locked claim matrix and latest cv9/oih artifacts as source-of-truth boundaries.

# Manuscript Skeleton v1

## Working Title

Latent Depth-Routing Spectroscopy in Standard Transformers: Primary-Model Oracle Evidence, Structured Regimes, and Boundaries

## One-Paragraph Abstract Skeleton

We analyze frozen `google/gemma-2-2b` with oracle-alpha to test whether
standard transformers expose an input-dependent effective depth mixture.
On held-out prompt surfaces, optimized routing improves loss versus uniform and
preregistered nulls (C1), and routing structure is non-random with strongest
raw-source organization in factual recall while grouped structure is broad (C2).
Softmax-constrained competitive routing outperforms tested unconstrained and
top-k alternatives on the confirm surface (C3). Extension lanes are bounded:
tool-breakage is mixed under dynamic controls (C4), safety mediator activity
remains refusal-collapsed on the broadened quartet surface (C5), and router
distillation/Figure-8-strong-claim lanes remain blocked. We frame oracle-alpha
as an effective-depth-mixture analysis on fixed representations, not a trained
router equivalence claim.

## Section Plan (Claims + Caveats)

### 1. Introduction

- Place: thesis motivation + main contribution framing.
- Claim placement:
  - C1 high-level statement (primary-model oracle signal exists).
  - C2 high-level statement (structured, task-conditioned routing).
- Required caveats:
  - frozen-model oracle is post-hoc, no co-adaptation equivalence.
  - no trained-AttnRes equivalence claim.

### 2. Method and Controls

- Place: oracle-alpha setup, nulls, split discipline, stability/predictiveness framing.
- Claim placement:
  - C1 methodological foundation (uniform/null comparisons).
  - C3 setup (softmax vs unconstrained/top-k regime design).
- Required caveats:
  - sequence-level statistical unit default.
  - shared-final-norm decomposition and reconstruction constraints.
  - MIB-compatible anchor handled as explicit extension lane (`oih`), not main spine.

### 3. Primary Results: Oracle Signal and Regimes

- Place: main quantitative section.
- Claim placement:
  - C1 with primary quantitative anchors from `registry_v5`.
  - C3 with regime-comparison outcomes.
- Required caveats:
  - bound to tested model/prompt surface.
  - no generalization to arbitrary architectures/tasks.

### 4. Structure Results: Grouped and Stratum-Conditioned Patterns

- Place: pattern analysis and block-structure interpretation.
- Claim placement:
  - C2 full details (grouped robustness + factual-recall strongest raw-source signal).
- Required caveats:
  - do not claim global clean `~8` raw clusters.
  - report stratum heterogeneity explicitly.

### 5. Extension Lanes (Bounded Evidence)

- Place: tool-breakage, safety, OIH anchor.
- Claim placement:
  - C4 (tool-breakage mixed under dynamic controls).
  - C5 (safety signals present, mediator role expansion negative on v3).
  - OIH extension: dynamic held-out signal is positive; calibrated static
    comparison stays positive but modest.
- Required caveats:
  - tool-breakage strongest same-model claim remains bounded.
  - safety mediator partition remains refusal-only on current v3 surface.
  - OIH primary pruned-static baseline is weak; use calibrated static deltas.

### 6. Limitations and Non-Claims

- Place: explicit claim boundary section.
- Must include:
  - no frozen-model == trained-router equivalence.
  - no strong trained-routing Figure 8 alignment claim.
  - router-distillation readiness gate unpassed.
  - training-dynamics not claim-bearing in this cycle.

### 7. Discussion and Next Experiments

- Place: concrete, bounded follow-ups.
- Include:
  - calibrated-static OIH interpretation discipline (now landed in `resattn-73r`).
  - optional safety mediator-definition redesign only with new mechanistic rationale.
  - no reopening frozen Figure 8 or ad hoc Phase 6 geometry sweeps in this cycle.

## Figure/Table Skeleton

1. Main table: C1/C3 quantitative anchors (oracle vs nulls vs regimes).
2. Structure figure: grouped + stratum-conditioned pattern summary (C2).
3. Extension table: C4/C5/OIH bounded outcomes with explicit status labels (`supported/mixed/blocked`).
4. Non-claims box: explicit exclusions to prevent overstatement.

## Artifact Mapping Used

- Claim matrix:
  - `results/infrastructure/20260320-paper-readiness-claim-matrix-v1.md`
- Claim-to-artifact table:
  - `results/infrastructure/20260320-manuscript-claim-to-artifact-table-v1.md`
- New extension artifacts included in this skeleton:
  - `results/safety_alignment/20260320-gemma2it-refusal-surface-v3-validation-v3.md`
  - `results/safety_alignment/20260320-gemma2it-mediator-conditioned-routing-v3.md`
  - `results/oracle_alpha/20260320-resattn-oih-full-v1.md`

## Ready-to-Write Checklist

- [x] Section order locked
- [x] Claim IDs mapped to sections
- [x] Caveat placement defined per section
- [x] Non-claims section specified
- [x] Convert this skeleton into manuscript prose draft (`resattn-lmp`)
