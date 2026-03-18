ABOUTME: Freezes the next stratified primary-model Gemma oracle-alpha campaign plan beyond registry_v4.
ABOUTME: Records the intended prompt surface, method freeze, MIB decision, and calibration-before-launch sequence.

# Motivation

`resattn-rh0` exists because the strongest underexploited positive lane in the
repo is now the primary-model Gemma oracle path. The repo already has:

- positive held-out routed-loss recovery on `google/gemma-2-2b`
- grouped pattern structure above matched random controls
- a decisive softmax versus unconstrained versus top-k regime result

What it does not have is a larger stratified prompt surface that can test
whether those gains survive across materially different input families rather
than on the saved `registry_v4` surface alone.

# Decision

The next scale-up should be a stratified `registry_v5` campaign on the primary
Gemma spine. Keep the current method surface fixed and change the prompt surface
only.

This means:

- keep the current primary-model runner and campaign infrastructure
- keep the current candidate target family comparison
- keep the current held-out predictiveness control plan
- do not reopen feature-family search, target redesign, or new oracle
  parameterizations inside the scale-up

# Planned Prompt Surface

Use four explicit strata:

1. factual recall
2. reasoning and math
3. code and procedural text
4. general narrative and expository text

Target counts:

- `64` pilot prompts per stratum
- `256` confirm prompts per stratum
- total `256` pilot prompts
- total `1024` confirm prompts

Each pilot prompt must include one saved paraphrase so the existing stability
suite remains usable without new prompt tooling.

Each prompt should carry explicit stratum tags so later pattern analysis can ask
stratum-conditioned questions directly from the saved registry rather than from
an external mapping file.

# Implementation Shape

Generate `prompts/registry_v5.yaml` rather than hand-editing a thousand-line
oracle collection.

Design constraints for the generated registry:

- preserve the existing `oracle_alpha_phase1_v1` collection id so the saved
  control registry remains valid
- keep the existing non-oracle collections unchanged
- emit prompts in round-robin stratum order within each split so small
  calibration slices cover all strata without needing a special calibration
  collection
- keep prompt ids stable and stratum-encoded

The generator should be deterministic and re-runnable from the repo so future
sessions can rebuild the committed registry artifact instead of editing it by
hand.

# MIB Decision

Omit MIB for this mixed stratified campaign and record the omission explicitly.

Rationale:

- the proposed `registry_v5` surface mixes custom prompt families across several
  task types
- that is not a benchmark-compatible single task-model pair
- forcing MIB into this campaign would turn the control into a mismatched side
  benchmark rather than a meaningful within-lane anchor

This is an omission for this campaign, not a repo-wide rejection of MIB.

# Calibration Before Launch

Before any tmux-backed full campaign, run one calibration slice on the final
`registry_v5` surface.

Minimum calibration:

- use the real `registry_v5` collection
- cap the run to a small number of pilot and confirm prompts
- rely on the round-robin registry ordering so the slice spans all four strata
- verify:
  - prompt loading works on the new registry
  - oracle checkpoints are written
  - feature caches are written
  - predictiveness summary completes
  - resume reuse works against saved checkpoints

The calibration is not a scientific gate. It is an operational gate for the
full tmux launch.

# Acceptance Criteria For The Full Scale-Up

The full campaign should only launch after all of the following are true:

- `registry_v5` is generated deterministically and committed
- the prompt registry tests pass on the new counts
- the calibration run completes on local MPS
- checkpoint and resume behavior is verified on the new output path
- `SCRATCHPAD.md` records the tmux session, checkpoint cadence, log path, and
  resume command before launch

# Immediate Follow-Up

Implement `resattn-gad` next:

- generate `registry_v5`
- add the smallest code or test support needed for deterministic generation
- run the calibration slice
- only then decide whether to launch the full tmux-backed campaign immediately
