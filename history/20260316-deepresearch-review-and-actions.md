# Deep Research Review and Actions

**Date:** 2026-03-16
**Purpose:** record what changed after reading `research/deepresearch1.md` and `research/deepresearch2.md` end to end and comparing them against the current repo controls.

## Findings Worth Adopting

1. The strongest honest frozen-model claim is an effective depth mixture, not proof of a literal internal router.
2. Figure 8 ambiguity has been resolved locally. `research/Attention_Residuals.pdf` does contain Figure 8, and it explicitly names diagonal dominance, embedding persistence, layer specialization, learned skip connections, and Block AttnRes with `N = 8` preserving the structure.
3. Oracle-alpha needs an identifiability gate before high-claim interpretation:
   - stability across optimization restarts and prompt resamples or paraphrases
   - out-of-sample predictiveness on the confirmatory split
   - MIB as a benchmark anchor or sanity control when the task-model pair fits it
4. Tool-breakage needs a stronger confirmatory control than raw-versus-tuned lens plots alone. The adopted control is a controlled dynamic-routing counterfactual plus an explicit failure metric.
5. The safety lane needs stronger structure:
   - layer localization first
   - feature discovery and validation second
   - mediator-conditioned routing analysis third
   - harmfulness must be separated from refusal before causal interpretation
6. Training-dynamics claims need a scope lock: checkpoint work supports within-run evolution claims, not counterfactual schedule claims.

## High-Signal Reference Additions

The local reference archive should include, at minimum:

- MIB
- Safety Layers in Aligned LLMs
- LLMs Encode Harmfulness and Refusal Separately
- Weight-sparse transformers have interpretable circuits
- OLMo 2
- EvoLM

These are useful because they directly change evaluation expectations, safety methodology, or the interpretation of checkpoint-based claims.

## What We Are Not Adopting Blindly

- MIB is not the new project spine. It is a benchmark/control anchor.
- The softmax versus unconstrained versus top-k lane remains mandatory. It is not downgraded to supplementary.
- We are not blocking implementation on Shapley-style interaction analysis.
- We are not pivoting the project around public AttnRes-scale checkpoints on this MacBook Pro.
- We are not adding a backend-abstraction layer before the first implementation slice proves the need.

## Follow-Up Implications

Remaining follow-up after this review:

- decide the tuned-lens path for Gemma-2 tool-breakage
- define a reproducible proxy for strong Figure 8 alignment claims
- keep the public prereg, dependency freeze, pilot/confirmatory split, and refusal-feature workflow as blockers before claim-bearing runs
