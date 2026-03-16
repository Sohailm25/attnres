# AGENTS.md - Codex Agent Directives for Depth-Routing Spectroscopy

## Scope and Precedence

This file governs all work within `resattn/` and supersedes any AGENTS.md found in parent directories for this workspace.

Parent-workspace assumptions that do not apply here:

- no Modal
- no persona-circuits phase names
- no remote volume or app-ID tracking
- no claim that oracle-alpha on frozen models is equivalent to trained Attention Residuals
- no dependency on `persona-circuits-proposal.md`

Everything below this section is the authoritative directive set for this experiment.

This experiment is grounded in local source documents, not in inherited assumptions from `~/braindstorms`.

## Mission

Execute the depth-routing research defined by:

- `research/master-research-document.docx`
- `research/decision-matrix.md`
- `research/artifact2.md`
- `research/artifact3.md`

The goal is to show, as rigorously as possible, whether standard transformers expose latent depth-routing structure that becomes visible under oracle-alpha analysis and whether that structure supports the interpretability thesis motivated by Attention Residuals.

The strongest default frozen-model framing is recovery of an `effective depth mixture` or latent routing signal, not proof that the model contains a literal trained router variable.

## Thesis Locks

These are not optional. Every implementation and write-up must preserve them.

1. `oracle-alpha` on frozen models is a post-hoc analysis, not a simulation of co-adapted AttnRes training.
2. Treat oracle-alpha as an upper bound on the routing signal recoverable from fixed standard-model representations and a lower bound on the benefit of depth routing once routing and computation can co-adapt.
3. The experiment must explicitly test Figure 8 predictions rather than merely reference them.
4. The tool-breakage demonstration is a first-class lane, not a future nice-to-have.
5. The comparison between softmax, unconstrained, and top-k routing is mandatory.
6. The learned router query vectors are the closest available `w_l` analogs and must be analyzed as first-class objects.
7. The block-structure hypothesis requires an explicit test for whether roughly 8 clusters emerge.
8. The safety angle is in scope: depth-routing differences tied to refusal or honesty-related features must be checked.
9. Figure 8 refers to Figure 8 in `research/Attention_Residuals.pdf`; strong alignment claims against trained routing require a reproducible proxy if direct trained-routing comparisons are unavailable locally.

## Runtime Assumptions

- Primary machine: MacBook Pro
- Python environment: `.venv`
- Primary backend: PyTorch + MPS
- Fallback tooling: CPU where MPS coverage is incomplete
- no Modal
- `research/artifact3.md` discusses conda, but this workspace standard is `.venv` and that supersedes the document for local execution here.

## Epistemic Standards

You are doing scientific work. Act like it.

1. Assumption quarantine. Any unverified statement is a hypothesis, not a fact. Label claims with `known`, `observed`, `inferred`, or `unknown`. Never upgrade `inferred` to `known` without evidence.
2. Evidence-first reasoning. Base conclusions on artifacts you can inspect: logs, metrics, cached activations, saved plots, test output, and local result files. A claim without an inspectable artifact is weak evidence.
3. No forced logic. Reject narrative chains that skip causal steps. If A implies B and B implies C, verify A to C directly before writing as if it were established.
4. Claim-evidence proportionality. Never write `validated`, `confirmed`, or `significant` without the metric, the comparison, and the threshold.
5. Adversarial self-questioning is mandatory before claim-bearing runs:
   - What is the most likely design flaw?
   - What is the simplest confound that could explain a positive result?
   - What would failure look like, and is the run designed to detect it?
   - If the expected result appears immediately, what is the probability the implementation is wrong?
6. Pre-register before running. The local prereg lives in `history/PREREG.md`. The public prereg target is LessWrong before claim-bearing execution.
7. Skepticism toward clean results. A result that looks too neat on the first pass deserves extra scrutiny, not celebration.
8. Implementation skepticism is critical. A script that runs and emits plausible numbers has not been validated. Validation requires an independent check.

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
│   ├── GAPS_SYNTHESIS.md
│   ├── RESEARCH_POSITIONING.md
│   ├── PROPOSAL_REVIEW.md
│   └── papers/
│       ├── DOWNLOAD_MANIFEST.md
│       └── *.pdf / *.html
├── configs/
│   └── experiment.yaml
├── history/
│   ├── PREREG.md
│   ├── 20260316-thesis-alignment-and-gap-closure.md
│   ├── 20260316-second-review-readiness.md
│   └── 20260316-methodology-gap-audit.md
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
│   └── download_reference_papers.py
├── sessions/
│   └── SESSION_TEMPLATE.md
└── tests/
```

## Research Navigation Guide

The source material is split across a long master document plus three supporting artifacts. Do not re-read everything blindly every session.

### Always read first in a fresh session

1. `CURRENT_STATE.md`
2. `journal/current_state.md`
3. `SCRATCHPAD.md`
4. `DECISIONS.md`
5. `history/PREREG.md`
6. `history/20260316-methodology-gap-audit.md`
7. `history/20260316-secondary-red-team-review.md`
8. `history/20260316-deepresearch-review-and-actions.md`

### Read by question

- Hypothesis, nulls, and phase gates:
  - `research/decision-matrix.md`
- Thesis motivation and why the experiment matters:
  - `research/artifact2.md`
- Local hardware and runtime feasibility:
  - `research/artifact3.md`
- Full context and detailed planning language:
  - `research/master-research-document.docx`

### Read only when needed

- `background-work/REFERENCES.md` when you need a paper, tool, or official URL
- `background-work/papers/DOWNLOAD_MANIFEST.md` when you need the local paper cache index
- `background-work/papers/*` when you need to read a paper directly without going back to the web
- `background-work/MECH_INTERP_GUIDANCE.md` when results are unexpected or implementation details feel shaky
- `background-work/GAPS_SYNTHESIS.md` when you need the short list of thesis-level non-negotiables
- `background-work/SAFETY_PUBLICATION_POLICY.md` before writing about refusal, jailbreaks, or other dual-use safety findings

## Operating Rules

### 1. Document Discipline

Before starting a non-trivial work session:

1. Read `CURRENT_STATE.md`
2. Create or update a session log in `sessions/`
3. Confirm the next task against the active phase and the prereg

During work:

- Update `SCRATCHPAD.md` before and after any substantial local run.
- Log non-obvious pivots in `DECISIONS.md` before proceeding.
- Update `CURRENT_STATE.md` whenever the actual project state changes.
- Register durable outputs in `results/RESULTS_INDEX.md`.

Long-running process rules:

- Any run that is expensive enough to care about surviving laptop movement, terminal closure, Wi-Fi changes, or disconnects must run inside `tmux`.
- Long-running runs must save resumable checkpoints to disk on a defined cadence. If the code cannot checkpoint, add checkpointing before launch rather than hoping the run survives.
- Before launch, record the `tmux` session name, checkpoint path, checkpoint cadence, log path, and resume command in `SCRATCHPAD.md`.
- After launch, verify that checkpoints are actually being written and that the resume command works against the latest checkpoint.
- Prefer durable checkpoint locations under the relevant `results/` lane rather than ephemeral temp directories.

Pre-run checkpoint format for `SCRATCHPAD.md`:

```text
## [TIMESTAMP] PRE-RUN: [run name]
- tmux session: [session name or N/A]
- Script: scripts/[filename].py
- Command: [exact command]
- Config: [key hyperparameters]
- What I'm testing: [one-sentence hypothesis]
- Expected outcome: [what success looks like]
- Expected duration: ~X minutes
- Checkpoint path: [path or N/A]
- Checkpoint cadence: [every N steps / minutes / epochs]
- Log path: [path]
- Resume command: [exact command]
- Main confound to watch: [one sentence]
- Implementation verified: YES/NO - [what independent check was run]
- Status: LAUNCHING
```

Post-run checkpoint format for `SCRATCHPAD.md`:

```text
## [TIMESTAMP] POST-RUN: [run name]
- Outcome: SUCCESS / FAILURE / PARTIAL
- Key metric: [the number that matters]
- Artifacts saved: [paths]
- Latest checkpoint: [path or none]
- Anomalies: [anything unexpected, or none]
- Next step: [what follows from this result]
```

### 2. Session Check-In Protocol

If context is thin or the session resumed after compaction:

1. Read `CURRENT_STATE.md`
2. Read the latest entries in `SCRATCHPAD.md`
3. Read the latest entries in `DECISIONS.md`
4. Read the latest session log in `sessions/`
5. Check `results/RESULTS_INDEX.md`
6. Only then return to the research documents

Do not re-explore the whole repo if the state docs already answer the question.

### 3. The Research Documents Are the Spec

- `research/decision-matrix.md` defines the critical decisions, null models, and gates.
- `history/PREREG.md` defines what is pre-registered locally.
- `history/20260316-methodology-gap-audit.md` defines the current known implementation hazards.
- If these documents conflict, resolve the conflict explicitly in `DECISIONS.md` before coding.

### 4. Execution Order

Use this as the default phase flow:

1. Phase 0: scaffold, prereg, methodology hardening, and paper archive
2. Phase 1: oracle-alpha infrastructure, reconstruction sanity checks, and stability gates
3. Phase 2: pattern analysis, Figure 8 validation, and regime comparisons
4. Phase 3: tool-breakage and safety routing analysis
5. Phase 4: training-dynamics extension and router training with `w_l` analog geometry
6. Phase 5: synthesis, writing, and artifact cleanup

Do not skip ahead to result interpretation until the reconstruction and null-model checks are green.

### 5. Experiment Design Defaults

- When a question is materially underspecified or multiple experimental approaches seem plausible, draft a short plan collaboratively with Sohail before execution.
- That plan should include the motivation, the concrete comparison or measurement, and a mock-up of the main plot or table using fake numbers if needed.
- Approved non-trivial new sprints should create or update a bd issue before execution so the work is visible to future sessions and parallel copies.
- Start with the smallest experiment that can genuinely falsify or support the idea. Do not scale up before the tiny version shows signs of life.
- Prefer tight feedback loops. A five-minute run is excellent, an hour is acceptable, and anything longer than a day requires explicit justification in `DECISIONS.md`.
- Treat most early-stage work as exploratory: the goal is often to gain surface area, expose unknown unknowns, and sharpen the ontology before expensive runs.
- Use MIB as a benchmark anchor or sanity control when the task-model pair fits it. If a lane cannot use MIB, record the reason in `DECISIONS.md`.
- Before making a high-claim interpretation, clear a stability suite and an out-of-sample predictiveness check on the confirmatory split rather than trusting a descriptive fit.
- Treat a controlled dynamic-routing counterfactual as the preferred confirmatory control for the tool-breakage lane.
- Treat strong AttnRes-alignment language as gated on a reproducible proxy, such as a small local reproduction or another open depth-mixing model, if direct trained-routing comparisons are not locally available.

### 6. Run Design Guardrails

- Never overclaim beyond an effective depth mixture unless stronger causal validation justifies richer router language.
- Never imply that frozen-model oracle-alpha proves how a trained Attention Residuals model would behave after co-adaptation.
- Never treat visual similarity to Figure 8 as evidence by itself; operationalize each prediction before looking at aggregate heatmaps.
- Never skip the preregistered null models.
- Never use `resid_post`-only caches for claim-bearing Figure 8 or layer-type specialization results; those require sublayer outputs.
- Never use Ward linkage directly on Jensen-Shannon distances; if Ward is required, move to an explicit Euclidean embedding first.
- Never validate the uniform-routing reconstruction against `logits / L`; the check is agreement with the model's original logits after the model's own final normalization.
- Never compute per-source logit contributions by applying LayerNorm or RMSNorm to each source independently; use the shared final normalization factor from the full routed mixture.
- Never report a paired significance test over token-level points as if they were independent examples; the default unit for claim-bearing significance is the sequence-level aggregate unless a stronger dependence-aware method is documented.
- Never assume raw logit lens is a clean monotonic baseline; tool-breakage claims must compare routed behavior against the original model and a tuned-lens-aware baseline.
- Never present a strong tool-breakage claim without a controlled dynamic-routing counterfactual or an equally explicit failure metric.
- Never run claim-bearing analysis on the same prompts used to tune the method. Use a pilot/confirmatory split for thresholds, prompt curation, and design choices.
- Never describe the router as if it consumes a single global `h_1`; the intended object is a per-token early hidden state such as `h_1[t]`, unless a different design is explicitly logged.
- Never assume refusal-feature labels are already available in GemmaScope. The safety lane requires a discovery and validation phase before causal claims.
- Never collapse harmfulness and refusal into one safety signal. Localize candidate safety layers first and separate harmfulness-encoding from refusal-execution before causal routing claims.

### 7. Required Experiment Lanes

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

### 8. Results Registration

Every saved artifact belongs in `results/RESULTS_INDEX.md`.
Do not delete old entries; mark them superseded.

### 9. Experiment Write-Ups

Every non-trivial experiment should end with a concise technical write-up stored near the relevant artifacts. The default structure is:

- Motivation / Methods / Results / Limitations / Next Steps

The main figure or table should be easy to identify from a quick scan of the directory.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
