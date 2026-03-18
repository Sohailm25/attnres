# ABOUTME: Records the Figure 8 redesign decision after the mixed widened `wikitext-103` best-checkpoint artifact.
# ABOUTME: Chooses a bounded regularization-first follow-up before any objective-level proxy changes.

## Question

After the widened compact-subword `wikitext-103` best-checkpoint artifact in
`resattn-fby`, should the next Figure 8 redesign change optimization
regularization or change the training objective itself?

## Evidence

### What the current proxy already does

- The local `8`-block Block AttnRes proxy is operationally real.
- Best-checkpoint export corrected an important measurement bug:
  - final loss delta `= +0.1272`
  - best-checkpoint loss delta `= +0.0386`
- The paper-facing metrics have moved gradually in the right direction over the
  redesign sequence:
  - deep embedding persistence:
    `0.1049 -> 0.1364 -> 0.1515 -> 0.1615 -> 0.1689`

This says the proxy is not obviously broken and that the standard next-token
objective can already reach a healthier operating region than the final
checkpoint suggested.

### What is still failing

- The matched baseline still wins at the best checkpoint on widened
  `wikitext-103`.
- The Figure 8 entropy ordering is still inverted at the best checkpoint:
  - mean pre-attn entropy `= 1.4454`
  - mean pre-MLP entropy `= 1.5003`
  - entropy gap `= -0.0549`
- The earlier positive `200`-step calibration did not survive training, and the
  best checkpoints for both models arrived well before the final step.

This is now more consistent with a stability / regularization problem than with
the standard next-token objective being absent altogether.

## Decision

The next bounded Figure 8 redesign should be **regularization-first, not
objective-first**.

Specifically:

- keep compact remapped GPT-2 subword tokenization
- keep the widened `d_model=160`, `d_ff=640`, `8`-block local proxy
- keep `wikitext/wikitext-103-raw-v1`
- keep the standard next-token language-model objective
- keep best-checkpoint export as mandatory
- run a matched regularization sweep before any objective-level change

The concrete follow-up is `resattn-bux`.

## Why objective changes are deferred

1. They would weaken proxy fidelity.
   The local reproduction is supposed to remain a believable small AttnRes
   proxy. Adding auxiliary routing losses or a custom Figure-8-shaped objective
   before exhausting simple stabilization levers would make any improvement much
   harder to interpret.

2. The current objective already finds a healthier region.
   The early positive calibration and the narrower best-checkpoint gap show the
   proxy can reach a better state under the standard objective. The live problem
   is sustaining that state, not proving it can exist at all.

3. Regularization is the smaller intervention.
   Dropout and weight decay can be varied on the existing runner without
   changing the model interface, routing export, or evaluation surface.

## Bounded Follow-Up

`resattn-bux` should compare the existing control against three new matched
regularization settings:

1. `dropout=0.1`, `weight_decay=0.01`
2. `dropout=0.0`, `weight_decay=0.05`
3. `dropout=0.1`, `weight_decay=0.05`

Use the existing best-checkpoint-enabled widened `wikitext-103` regime and
judge each arm on:

- best-checkpoint baseline-versus-AttnRes loss delta
- best-checkpoint Figure 8 entropy gap

## Exit Condition

- If a regularization arm restores a routed win or materially narrows the loss
  gap while improving the entropy gap relative to `-0.0549`, keep the standard
  objective and continue from that arm.
- If all three arms remain negative or equally mixed, objective-level redesign
  becomes the next live Figure 8 question.
