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

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/python -m unittest tests.test_scaffold
```

Optional setup:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/pre-commit install
```

## Operating Scaffold

- `AGENTS.md` is the local operating contract.
- `CURRENT_STATE.md` is the current single-source status file.
- `DECISIONS.md` records non-trivial decisions and pivots.
- `SCRATCHPAD.md` is the pre-run and post-run execution log.
- `THOUGHT_LOG.md` captures open questions, risks, and new ideas.
- `history/PREREG.md` is the preregistered claim and gate document.
- `history/20260316-thesis-alignment-and-gap-closure.md` records how this scaffold was adapted from `braindstorms`.
