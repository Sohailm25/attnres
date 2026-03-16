# AGENTS.md - Codex Agent Directives for Depth-Routing Spectroscopy

## Scope and Precedence

This file governs all work within `resattn/` and supersedes any AGENTS.md found in parent directories for this workspace.

Parent-workspace assumptions that do not apply here:

- no Modal
- no persona-circuits phase names
- no remote volume or app-ID tracking
- no claim that oracle-alpha on frozen models is equivalent to trained Attention Residuals

This experiment is grounded in local source documents, not in inherited assumptions from `~/braindstorms`.

## Mission

Execute the depth-routing research defined by:

- `research/master-research-document.docx`
- `research/decision-matrix.md`
- `research/artifact2.md`
- `research/artifact3.md`

The goal is to show, as rigorously as possible, whether standard transformers expose latent depth-routing structure that becomes visible under oracle-alpha analysis and whether that structure supports the interpretability thesis motivated by Attention Residuals.

## Thesis Locks

These are not optional. Every implementation and write-up must preserve them.

1. `oracle-alpha` on frozen models is a post-hoc analysis, not a simulation of co-adapted AttnRes training.
2. Treat oracle-alpha as an upper bound on the routing signal available in standard architectures under fixed representations.
3. The experiment must explicitly test Figure 8 predictions rather than merely reference them.
4. The tool-breakage demonstration is a first-class lane, not a future nice-to-have.
5. The comparison between softmax, unconstrained, and top-k routing is mandatory.
6. The learned router query vectors are the closest available `w_l` analogs and must be analyzed as first-class objects.
7. The block-structure hypothesis requires an explicit test for whether roughly 8 clusters emerge.
8. The safety angle is in scope: depth-routing differences tied to refusal or honesty-related features must be checked.

## Runtime Assumptions

- Primary machine: MacBook Pro
- Python environment: `.venv`
- Primary backend: PyTorch + MPS
- Fallback tooling: CPU where MPS coverage is incomplete
- no Modal

## Directory Map

```text
resattn/
├── AGENTS.md
├── CURRENT_STATE.md
├── DECISIONS.md
├── README.md
├── SCRATCHPAD.md
├── THOUGHT_LOG.md
├── background-work/
│   ├── REFERENCES.md
│   ├── MECH_INTERP_GUIDANCE.md
│   └── GAPS_SYNTHESIS.md
├── configs/
│   └── experiment.yaml
├── history/
│   ├── PREREG.md
│   └── 20260316-thesis-alignment-and-gap-closure.md
├── journal/
│   ├── current_state.md
│   └── logs/
├── knowledge/
├── notebooks/
├── prompts/
├── research/
├── results/
│   ├── infrastructure/
│   ├── oracle_alpha/
│   ├── pattern_analysis/
│   ├── figure8_validation/
│   ├── block_structure/
│   ├── comparison_regimes/
│   ├── tool_breakage/
│   ├── training_dynamics/
│   ├── router_training/
│   ├── safety_alignment/
│   └── figures/
├── scratch/
├── scripts/
├── sessions/
│   └── SESSION_TEMPLATE.md
└── tests/
```

## Read Order

For a fresh session or after context loss, read in this order:

1. `CURRENT_STATE.md`
2. `journal/current_state.md`
3. `SCRATCHPAD.md`
4. `DECISIONS.md`
5. `history/PREREG.md`
6. `history/20260316-thesis-alignment-and-gap-closure.md`
7. Relevant sections of `research/master-research-document.docx`
8. Relevant sections of `research/decision-matrix.md`

## Operating Rules

### 1. Scientific Framing

- Never imply that frozen-model oracle-alpha proves how a trained Attention Residuals model would behave after co-adaptation.
- Never treat visual similarity to Figure 8 as evidence by itself; operationalize each prediction before looking at results.
- Never skip the null models from `research/decision-matrix.md`.

### 2. Required Experiment Lanes

The following lanes must remain visible in `CURRENT_STATE.md`, `history/PREREG.md`, and `results/RESULTS_INDEX.md`:

- oracle-alpha loss improvement against preregistered nulls
- pattern analysis over routing distributions
- Figure 8 validation
- block-structure clustering around the ~8-cluster hypothesis
- softmax vs unconstrained vs top-k comparison
- tool-breakage demonstration
- training-dynamics extension
- router training plus `w_l` analog geometry
- safety analysis focused on routing differences around refusal or honesty features

### 3. Run Logging

Before any substantial local run, write a pre-run checkpoint to `SCRATCHPAD.md`.
Immediately after completion or failure, write a post-run checkpoint.

### 4. Decision Logging

Any non-obvious methodological decision or pivot goes to `DECISIONS.md` before proceeding.

### 5. Results Registration

Every saved artifact belongs in `results/RESULTS_INDEX.md`.
Do not delete old entries; mark them superseded.
