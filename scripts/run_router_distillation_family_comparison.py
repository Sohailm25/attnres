# ABOUTME: Runs the pilot-only router-family comparison on the saved Gemma router-distillation export.
# ABOUTME: Keeps the baseline input, target, and aggregation fixed while comparing a linear head against the current MLP family.

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
            "Run the pilot-only router-family comparison on the saved Gemma "
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
        help="Output directory for the family comparison summary JSON",
    )
    parser.add_argument(
        "--fixed-input-field",
        default="h_4[t]",
        help="Frozen router input field",
    )
    parser.add_argument(
        "--fixed-target-name",
        default="oracle_alpha_logit_vector",
        help="Frozen target parameterization",
    )
    parser.add_argument(
        "--fixed-aggregation",
        default="mean_token_logits_then_softmax",
        help="Frozen token-to-sequence aggregation rule",
    )
    parser.add_argument(
        "--candidate-router-families",
        nargs="+",
        default=("linear", "mlp"),
        help="Router families to compare",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden width for the MLP baseline",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.25,
        help="Held-out pilot fraction for family comparison",
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
        compare_router_families,
        compact_router_distillation_family_summary_payload,
        load_router_distillation_pilot_dataset,
    )

    args = parse_args()
    dataset = load_router_distillation_pilot_dataset(Path(args.export_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = compare_router_families(
        examples=dataset.examples,
        fixed_input_field=args.fixed_input_field,
        fixed_target_name=args.fixed_target_name,
        fixed_aggregation=args.fixed_aggregation,
        candidate_router_families=tuple(args.candidate_router_families),
        hidden_dim=args.hidden_dim,
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
    payload = compact_router_distillation_family_summary_payload(summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
