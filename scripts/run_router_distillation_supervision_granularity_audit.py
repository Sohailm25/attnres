# ABOUTME: Runs the saved-split supervision-granularity audit for the Gemma Phase 6 pilot baseline.
# ABOUTME: Reuses the frozen family-comparison selection and emits prompt-level held-out diagnostics plus a compact recommendation.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the remaining Gemma router-distillation pilot blocker "
            "looks like supervision granularity on the frozen family baseline."
        )
    )
    parser.add_argument(
        "--export-dir",
        default=(
            "results/router_training/"
            "20260318-gemma2-router-distillation-pilot-export-v1"
        ),
        help="Saved per-token pilot export directory",
    )
    parser.add_argument(
        "--family-summary-path",
        default=(
            "results/router_training/"
            "20260318-gemma2-router-distillation-family-comparison-v1/"
            "summary.json"
        ),
        help="Saved family-comparison summary that freezes the baseline split",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the supervision-granularity audit summary JSON",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Torch device",
    )
    return parser.parse_args()


def _selected_family_input_summary(
    family_summary: dict[str, object],
) -> dict[str, object]:
    selected_router_family = str(family_summary["selected_router_family"])
    fixed_input_field = str(family_summary["fixed_input_field"])
    fixed_target_name = str(family_summary["fixed_target_name"])
    fixed_aggregation = str(family_summary["fixed_aggregation"])
    for input_summary in family_summary["input_summaries"]:
        if (
            str(input_summary["router_family"]) == selected_router_family
            and str(input_summary["input_field"]) == fixed_input_field
            and str(input_summary["target_name"]) == fixed_target_name
            and str(input_summary["aggregation"]) == fixed_aggregation
        ):
            return dict(input_summary)
    raise ValueError("selected family summary is missing from input_summaries")


def main() -> None:
    from validation.router_distillation import (
        audit_router_supervision_granularity,
        compact_router_distillation_supervision_granularity_payload,
        load_router_distillation_pilot_dataset,
    )

    args = parse_args()
    family_summary_path = Path(args.family_summary_path)
    family_summary = json.loads(family_summary_path.read_text())
    baseline_summary = _selected_family_input_summary(family_summary)

    dataset = load_router_distillation_pilot_dataset(Path(args.export_dir))
    audit = audit_router_supervision_granularity(
        examples=dataset.examples,
        fixed_input_field=str(family_summary["fixed_input_field"]),
        fixed_target_name=str(family_summary["fixed_target_name"]),
        fixed_aggregation=str(family_summary["fixed_aggregation"]),
        fixed_router_family=str(family_summary["selected_router_family"]),
        hidden_dim=int(family_summary["hidden_dim"]),
        train_prompt_ids=tuple(family_summary["train_prompt_ids"]),
        eval_prompt_ids=tuple(family_summary["eval_prompt_ids"]),
        learning_rate=float(baseline_summary["learning_rate"]),
        weight_decay=float(baseline_summary["weight_decay"]),
        batch_size=int(baseline_summary["batch_size"]),
        max_epochs=int(baseline_summary["max_epochs"]),
        patience=int(baseline_summary["patience"]),
        device=args.device,
        collection_id=dataset.collection_id,
        model_name=dataset.model_name,
    )

    payload = compact_router_distillation_supervision_granularity_payload(audit)
    payload["frozen_family_summary_path"] = str(family_summary_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
