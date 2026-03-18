# ABOUTME: Runs the first pilot-only router-distillation comparison on the saved Gemma export.
# ABOUTME: Compares h_1[t] and h_4[t] with one explicit token-to-sequence aggregation rule before confirmatory training.

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
            "Run the pilot-only router-distillation comparison on the saved "
            "Gemma per-token export."
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
        help="Output directory for the pilot comparison summary JSON",
    )
    parser.add_argument(
        "--candidate-input-fields",
        nargs="+",
        default=("h_1[t]", "h_4[t]"),
        help="Router input fields to compare on the pilot split",
    )
    parser.add_argument(
        "--aggregation",
        default="mean_token_logits_then_softmax",
        help="Token-to-sequence aggregation rule",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.25,
        help="Held-out pilot fraction for input-source comparison",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden width for the 2-layer router MLP",
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
        compact_router_distillation_summary_payload,
        run_router_distillation_pilot_comparison,
    )

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_router_distillation_pilot_comparison(
        export_dir=Path(args.export_dir),
        candidate_input_fields=tuple(args.candidate_input_fields),
        aggregation=args.aggregation,
        eval_fraction=args.eval_fraction,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    payload = compact_router_distillation_summary_payload(summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
