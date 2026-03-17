# Depth-Routing Spectroscopy

This workspace mirrors the operating scaffold used in `~/braindstorms`, but it is adapted to the depth-routing experiment defined by the local `research/` materials rather than the persona-circuits project.

## Source Documents

- `research/master-research-document.docx`
- `research/decision-matrix.md`
- `research/artifact1.md`
- `research/artifact2.md`
- `research/artifact3.md`

## Local Runtime Assumptions

- Execution happens on this MacBook Pro, locally.
- Python runs through `.venv`.
- The expected accelerator is PyTorch MPS with CPU fallback when needed.
- No Modal, no remote container assumptions, and no inherited persona-circuits runtime shortcuts.
- `research/artifact3.md` discusses a conda setup, but this workspace intentionally standardizes on `.venv` to match the local operating rules.

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/python -m unittest discover -s tests -p 'test*.py'
```

Optional setup:

```bash
.venv/bin/pre-commit install
```

## Operating Scaffold

- `AGENTS.md` is the local operating contract.
- `CURRENT_STATE.md` is the current single-source status file.
- `DECISIONS.md` records non-trivial decisions and pivots.
- `SCRATCHPAD.md` is the pre-run and post-run execution log.
- `THOUGHT_LOG.md` captures research reflections: hunches, predictions, surprises, confidence shifts, and the evolving feel of the experiment.
- `history/PREREG.md` is the preregistered claim and gate document.
- `history/20260316-thesis-alignment-and-gap-closure.md` records how this scaffold was adapted from `braindstorms`.
- `history/20260316-methodology-gap-audit.md` records the implementation hazards that were found after the second review.
- `background-work/papers/DOWNLOAD_MANIFEST.md` indexes the local paper cache used throughout the experiment.
- `requirements.lock.txt` freezes the full transitive Phase 1 environment used in `.venv`.
- `validation/` holds the normalization-aware reconstruction and cache-validity checks that gate oracle-alpha work.
- `scripts/download_reference_papers.py` refreshes the local paper cache.
