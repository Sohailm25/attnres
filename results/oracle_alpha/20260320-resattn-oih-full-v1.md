ABOUTME: Summarizes the first full OIH anchor run pairing oracle-alpha with a static block-influence baseline.
ABOUTME: Records dynamic-vs-static outcomes on the new `resattn_oih_mcqa_v1` MCQA surface.

# Resattn OIH MCQA Anchor Full v1

## Motivation

`resattn-oih` asks whether the primary Gemma oracle signal remains meaningfully
strong when compared against static alternatives on a MIB-compatible
multiple-choice surface, rather than only against uniform/null policies.

## Methods

- Script: `scripts/run_resattn_oih.py`
- Model: `google/gemma-2-2b`
- Device: `mps`
- Collection: `resattn_oih_mcqa_v1`
- Split: `110` pilot / `50` confirm
- Dynamic path:
  - checkpointed oracle campaign + held-out predictiveness
  - feature source: `position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat`
  - target: `block_compressed_alpha_logit_vector`
  - regularization grid: `1e-4 ... 100`
- Static path (primary comparator):
  - pilot-only block-influence scoring by drop-one-block loss impact
  - prune fraction: `0.10`
  - static alpha fixed across confirm prompts
- Supplementary static check:
  - pilot-mean-alpha policy evaluated on the same `50` confirm prompts

## Results

### Dynamic oracle path

- predicted mean improvement over uniform: `+0.2290` nats
- oracle mean improvement over uniform: `+0.5709` nats
- confirm `R^2`: `0.0906`
- confirm mean JS: `0.0758`
- selected regularization: `100.0`

### Static comparators

- primary ShortGPT-style pruned static baseline:
  - mean improvement over uniform: `-3.5916` nats
  - positive prompts: `0 / 50`
- pilot-mean-alpha static baseline:
  - mean improvement over uniform: `+0.1813` nats
  - positive prompts: `49 / 50`
- calibrated static comparator (selected best static policy from the two above):
  - selected policy: `pilot_mean_alpha`
  - mean improvement over uniform: `+0.1813` nats

### Dynamic vs static deltas

- predicted minus pruned-static: `+3.8207` nats
- oracle minus pruned-static: `+4.1625` nats
- predicted minus calibrated-static: `+0.0478` nats
- oracle minus calibrated-static: `+0.3896` nats

## Interpretation

- The lane now has a full primary-model OIH artifact with positive held-out
  dynamic predictiveness and a clear dynamic-over-static gap.
- The primary pruned static baseline appears too weak on this surface.
- The calibrated static comparator now defaults to pilot-mean-alpha, which is
  materially stronger than pruned static and still remains below dynamic
  predicted performance.
- `mib_status` is now control-plan aligned (`planned`) in the refreshed summary
  output for this same full artifact path.

## Limitations

- The calibrated static policy currently chooses between two fixed candidates
  (pruned static and pilot-mean static), not a broader static-policy family.
- The dynamic margin over calibrated static is positive but modest (`+0.0478`
  nats), so stronger claim language should still be conservative.

## Next Steps

1. Keep this artifact as the first full OIH anchor result with calibrated
   static comparison enabled.
2. Carry both dynamic-vs-pruned and dynamic-vs-calibrated deltas in manuscript
   language to avoid over-reliance on the weak pruned baseline.
3. Prioritize primary-model synthesis and writing integration over additional
   OIH baseline redesign unless a concrete methodological defect appears.
