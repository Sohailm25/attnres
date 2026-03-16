# Methodology Gap Audit

**Date:** 2026-03-16
**Status:** active control document
**Purpose:** capture the implementation hazards still capable of invalidating the experiment even after the thesis-level scaffold was corrected.

## Scope

This audit was produced after re-reading the local research corpus, comparing `resattn/` against `~/braindstorms`, and checking the cited methodology against external primary sources where needed.

The research documents remain the source material, but the control documents in this repo supersede any method detail that is technically invalid or underspecified for implementation.

## Findings

### 1. Ward linkage on Jensen-Shannon distances is invalid

- Local source locations:
  - `research/decision-matrix.md`
  - `research/master-research-document.docx` paragraphs 242-243
- Problem:
  - the plan names Jensen-Shannon divergence as the primary routing distance and then pairs it with Ward linkage
  - scikit-learn's `AgglomerativeClustering` documentation states that if linkage is Ward, only "euclidean" is accepted
- Why this matters:
  - Ward optimizes within-cluster variance in a Euclidean feature space
  - using it directly on a Jensen-Shannon distance matrix is not just suboptimal; it is methodologically wrong
- Required correction:
  - if clustering is run directly on Jensen-Shannon distances, use average or complete linkage
  - if Ward is desired, first embed the routing vectors into a Euclidean space and document that embedding explicitly

### 2. The uniform-routing reconstruction check should match the original logits, not logits / L

- Local source location:
  - `research/decision-matrix.md`
- Problem:
  - the current validation text says the uniform reconstruction should equal the original model logits divided by `L`
- Why this is wrong:
  - the routed reconstruction still passes through the model's own final RMSNorm or LayerNorm
  - for positive scalar rescaling, the final normalization largely cancels the global scale factor, so the comparison target is the model's original logits after the model's own final normalization
  - the correct bug signal is failure to match the original logits, not failure to match `logits / L`
- Required correction:
  - the uniform-routing validation target is agreement with the original logits
  - write down any residual mismatch only if it survives ordinary floating-point tolerance and known normalization epsilon effects

### 3. The current alpha-spectroscopy write-up overstates exactness unless the shared final normalization factor is used

- Local source location:
  - `research/artifact2.md`
- Problem:
  - the current text defines per-source contributions by applying per-source LayerNorm or per-source RMSNorm to `v_i`
  - specifically, it describes `logit_i = alpha_i * (W_U * LayerNorm(v_i))`
- Why this is wrong:
  - the final normalization is applied to the full routed mixture, not to each source separately
  - per-source LayerNorm or per-source RMSNorm changes the denominator and therefore breaks exactness
- What is still salvageable:
  - exact decomposition is still possible if the shared final normalization factor from the full routed mixture is computed first and then distributed across sources
  - that is the correct exact decomposition because the unembedding is linear once the shared final normalization factor has been fixed
- Required correction:
  - any claim of exact routed-logit decomposition must use the shared final normalization factor
  - otherwise the method must be labeled approximate, not exact

### 4. Claim-bearing significance should use a sequence-level unit to avoid pseudoreplication

- Local source locations:
  - `research/decision-matrix.md`
  - `research/master-research-document.docx` paragraph 224
- Problem:
  - the preregistered gate references a paired t-test but does not lock the statistical unit strongly enough
- Why this matters:
  - token losses inside a sequence are correlated
  - treating token-level deltas as independent points will inflate significance through pseudoreplication
- Required correction:
  - default claim-bearing significance tests to paired tests on per-sequence mean loss deltas
  - if a token-level analysis is added later, it must use an explicit dependence-aware method and be justified before interpretation

### 5. `resid_post`-only caching is insufficient for final Figure 8 claims

- Local source locations:
  - `research/artifact3.md`
  - `research/decision-matrix.md`
  - `research/master-research-document.docx` paragraphs 234 and 368-370
- Problem:
  - `artifact3` highlights `resid_post`-only caching as the minimal-memory path
  - the decision matrix and master doc require sublayer outputs to test layer-type specialization
- Why this matters:
  - Figure 8's attention-versus-MLP specialization cannot be tested from one `resid_post` state per block
  - `resid_post` remains acceptable for smoke tests and fast iteration, but not for claim-bearing Figure 8 analysis
- Required correction:
  - final analysis that touches Figure 8 or layer-type specialization must use sublayer outputs
  - memory planning should assume this richer cache for the claim-bearing runs

## Remaining Non-Structural Preconditions

These are not methodology bugs, but they still block claim-bearing execution:

1. dependency freeze before scientific runs

## Applied Repo-Level Corrections

The following documents now encode these corrections:

- `AGENTS.md`
- `CURRENT_STATE.md`
- `history/PREREG.md`
- `configs/experiment.yaml`
- `background-work/MECH_INTERP_GUIDANCE.md`
- `background-work/GAPS_SYNTHESIS.md`

The original research documents under `research/` were intentionally left untouched as source materials.
