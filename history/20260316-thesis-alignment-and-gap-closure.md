# Thesis Alignment And Gap Closure

## Purpose

This memo records how the `braindstorms` scaffold was adapted for the depth-routing experiment so the local workspace inherits the same rigor without importing the wrong assumptions.

## What Was Reused

- Root control documents: state, decisions, scratchpad, thought log
- Session logging
- Results indexing
- Stage-oriented directory structure
- Preregistration-first workflow

## What Was Explicitly Removed

- Modal execution assumptions
- persona-circuits terminology
- remote app and volume tracking
- any implication that the prior experiment's phase sequence applies here

## Gap Closure Mapping

### Gap 1: Oracle-alpha misses co-adaptation

- Fixed by locking the framing across `AGENTS.md`, `CURRENT_STATE.md`, `configs/experiment.yaml`, and `history/PREREG.md`.
- The workspace now treats oracle-alpha as an upper bound on the routing signal available in standard architectures under fixed representations.

### Gap 2: Tool-breakage analysis vanished

- Fixed by adding `results/tool_breakage/` and making the tool-breakage lane mandatory in `AGENTS.md` and `history/PREREG.md`.

### Gap 3: `w_l` geometry disappeared

- Fixed by making learned router query vectors a required w_l analog analysis lane.
- The geometry lane is now called out in `CURRENT_STATE.md`, `configs/experiment.yaml`, and `history/PREREG.md`.

### Gap 4: Block structure was absent

- Fixed by adding `results/block_structure/` and preregistering the explicit `8 clusters` hypothesis.

### Gap 5: Softmax competition was untested

- Fixed by adding `results/comparison_regimes/` and preregistering softmax-constrained, unconstrained, and top-k regimes.

### Gap 6: Figure 8 was not used as a prediction surface

- Fixed by adding `results/figure8_validation/` and locking the Figure 8 signatures into preregistered tests.

### Gap 7: Safety and alignment angle disappeared

- Fixed by adding `results/safety_alignment/` and preregistering refusal or honesty-focused routing analysis.

## Structural Adaptation Summary

The resulting scaffold is intentionally stricter than a simple port:

- same discipline level as `braindstorms`
- local MacBook Pro runtime instead of Modal
- experiment lanes aligned to depth-routing claims rather than persona steering claims
