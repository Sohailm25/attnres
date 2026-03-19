# ABOUTME: Runs the pilot-only capacity comparison on the saved Gemma router-distillation export.
# ABOUTME: Keeps the alpha-logit target and mean-token aggregation fixed while comparing hidden widths before any richer router redesign.

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
            "Run the pilot-only capacity comparison on the saved Gemma "
            "router-distillation export."
        )
    )
    parser.add_argument(
        "--export-dir",
        default=(
            "results/router_training/"
            "20260318-gemma2-router-distillation-pilot-export-v1"
        ),
        help="Saved per-token export directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the capacity comparison summary JSON",
    )
    parser.add_argument(
        "--candidate-input-fields",
        nargs="+",
        default=("h_1[t]", "h_4[t]"),
        help="Router input fields to compare on the pilot split",
    )
    parser.add_argument(
        "--fixed-target-name",
        default="oracle_alpha_logit_vector",
        help="Frozen target parameterization for the capacity comparison",
    )
    parser.add_argument(
        "--fixed-aggregation",
        default="mean_token_logits_then_softmax",
        help="Frozen token-to-sequence aggregation rule",
    )
    parser.add_argument(
        "--candidate-hidden-dims",
        nargs="+",
        type=int,
        default=(256, 512),
        help="Hidden widths to compare",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.25,
        help="Held-out pilot fraction for capacity comparison",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Adam weight decay",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Prompt batch size",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=300,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=40,
        help="Early-stopping patience on held-out pilot loss",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Split and training seed",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Torch device",
    )
    return parser.parse_args()


def main() -> None:
    from validation.router_distillation import (
        compare_router_capacities,
        compact_router_distillation_capacity_summary_payload,
        load_router_distillation_pilot_dataset,
    )

    args = parse_args()
    dataset = load_router_distillation_pilot_dataset(Path(args.export_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = compare_router_capacities(
        examples=dataset.examples,
        candidate_input_fields=tuple(args.candidate_input_fields),
        fixed_target_name=args.fixed_target_name,
        fixed_aggregation=args.fixed_aggregation,
        candidate_hidden_dims=tuple(args.candidate_hidden_dims),
        eval_fraction=args.eval_fraction,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        collection_id=dataset.collection_id,
        model_name=dataset.model_name,
    )
    payload = compact_router_distillation_capacity_summary_payload(summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
