# Motivation

`resattn-syn` blocks all claim-bearing work because the repo needed a reproducible local environment and a first correctness surface for reconstruction and cache-validity checks.

# Methods

- Pinned the direct Phase 1 dependency set in `requirements.txt`.
- Generated a fully resolved local lockfile in `requirements.lock.txt` from the pinned direct requirements.
- Installed the pinned stack into `.venv` and ran a core import smoke test for `torch`, `transformer-lens`, `sae-lens`, `nnsight`, and `pyvene`.
- Added backend-agnostic validation helpers in `validation/reconstruction.py` and tests in `tests/test_phase1_validation.py`.

# Results

- The local Phase 1 stack now imports successfully in `.venv`.
- `torch.backends.mps.is_available()` returned `True`.
- The first validation suite now checks:
  - cache reconstruction from embedding plus sublayer outputs
  - uniform-routing agreement with the original logits after final normalization
  - exact shared-final-norm routed-logit decomposition
  - failure of the per-source-normalization shortcut

# Limitations

- The current validation module is linear-algebra level only; it does not yet exercise a real model backend or cached activations.
- The lockfile is a local-machine freeze, not a cross-platform environment guarantee.

# Next Steps

- Extend the validation helpers into model-backed reconstruction checks on the development model.
- Keep broader oracle-alpha work gated behind those checks plus the pilot/confirmatory split and the identifiability controls.
