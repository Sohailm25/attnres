ABOUTME: Provides a single reviewer-facing packet with the current manuscript draft surface and claim boundaries.
ABOUTME: Focuses reviewer attention on validity of bounded claims rather than exploratory backlog.

# External Reviewer Packet v1

## Packet Goal

Provide a fast, self-contained review surface for the current manuscript state:
supported core claims (C1-C3), mixed extension lanes (C4-C6), and explicit
non-claims.

## Primary Draft Artifacts

- Compressed draft (current): `results/infrastructure/20260320-manuscript-draft-pass-v3.md`
- Expanded draft: `results/infrastructure/20260320-manuscript-draft-pass-v2.md`
- First prose pass: `results/infrastructure/20260320-manuscript-draft-pass-v1.md`

## Claim Boundary Artifacts

- Claim table: `results/infrastructure/20260320-manuscript-claim-to-artifact-table-v1.md`
- Claim matrix: `results/infrastructure/20260320-paper-readiness-claim-matrix-v1.md`
- Current state: `CURRENT_STATE.md`
- Preregistered framing/guardrails: `history/PREREG.md`

## Extension-Lane Anchors

- Tool-breakage (mixed): `results/tool_breakage/20260320-gemma2-tool-breakage-route-mode-dynamic-seeded-v1.md`
- Safety v3 validation: `results/safety_alignment/20260320-gemma2it-refusal-surface-v3-validation-v3.md`
- Safety v3 mediator: `results/safety_alignment/20260320-gemma2it-mediator-conditioned-routing-v3.md`
- OIH calibrated anchor: `results/oracle_alpha/20260320-resattn-oih-full-v1.md`

## Reviewer Questions (Requested)

1. Are C1-C3 claim wordings appropriately scoped to frozen-model, tested-surface evidence?
2. Are C4-C6 caveats sufficiently strong, especially for tool-breakage and safety mediator collapse?
3. Is OIH phrasing appropriately calibrated around the modest dynamic-over-calibrated-static margin (`+0.0478`)?
4. Are any phrases still implying trained-routing equivalence, strong Figure 8 trained alignment, or Phase 6 readiness?
5. Which single revision would most improve external clarity without adding new experiments?

## Known Non-Claims To Enforce

- No frozen-model oracle-alpha equals trained routing claim.
- No strong trained-routing Figure 8 alignment claim.
- No router-distillation readiness claim (`R^2 > 0.5` not reached).
- No claim-bearing training-dynamics result in this cycle.
