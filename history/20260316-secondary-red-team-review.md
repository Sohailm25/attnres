# Secondary Red-Team Review

**Date:** 2026-03-16
**Purpose:** surface the remaining weak points after the scaffold and methodology hardening pass, with emphasis on publishability rather than mere executability.

## Verdict

The repo is now structurally strong, but four remaining issues would materially weaken a blog post or paper if left implicit.

## Finding 1: The current tool-breakage lane over-trusts raw logit lens

- Why this matters:
  - raw logit lens is often brittle even in ordinary transformers
  - the Tuned Lens paper exists partly because intermediate states are not in a stable prediction basis under the raw lens
- Risk:
  - if the baseline already shows irregular raw-lens behavior, a routed-model plot that looks messy does not prove that routing broke the tool
- Required fix:
  - compare original-model and routed-model behavior under both raw logit lens and tuned lens
  - define the strong claim as additional instability under routing relative to the original-model baseline, not as non-monotonicity alone

## Finding 2: A pilot/confirmatory split is missing

- Why this matters:
  - without a pilot/confirmatory split, the same prompt pool can quietly be used to tune thresholds, select examples, and support the final claim
  - that is acceptable for exploration but weak for a paper-quality result
- Required fix:
  - create and save a pilot/confirmatory split before method tuning
  - use the pilot for prompt curation, architecture choice, and visualization decisions
  - reserve the confirmatory tranche for claim-bearing statistics

## Finding 3: Router-input wording is ambiguous enough to invite the wrong implementation

- Why this matters:
  - `2-layer MLP on h_1` can be misread as a single sequence-level vector
  - the intended object is a per-token early hidden state such as `h_1[t]`
- Risk:
  - an agent could accidentally build a weak global router, conclude the lane failed, and attribute the failure to the thesis rather than the misread spec
- Required fix:
  - keep the wording explicit as `h_1[t]`
  - run a pilot comparison against an early contextual state if needed

## Finding 4: The safety lane assumes refusal-feature discovery is already solved

- Why this matters:
  - there is strong literature support for refusal structure, including *Refusal in Language Models Is Mediated by a Single Direction*
  - but that does not mean the exact refusal-feature set needed for this GemmaScope workflow is already identified locally
- Risk:
  - the safety lane could stall late because the project treated refusal-feature discovery as already done
- Required fix:
  - add an explicit refusal-feature discovery and validation sub-phase before causal routing comparisons

## Additional Research Added

The local paper cache should include the following because they directly shore up the weak points above:

- *Refusal in Language Models Is Mediated by a Single Direction*
- *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*
- *In-context Learning and Induction Heads*
- *Measuring Faithfulness in Chain-of-Thought Reasoning*

## Bottom Line

If these four fixes hold, the project is much better positioned to produce a post or paper that interpretability researchers take seriously rather than reading as an exploratory notebook with strong conclusions.
