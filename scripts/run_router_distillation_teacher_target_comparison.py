# ABOUTME: Compares the retained all-token Gemma router baseline against bounded tokenwise teacher targets.
# ABOUTME: Reuses the frozen all-token family-summary split so only target fidelity changes.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformer_lens import HookedTransformer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare bounded tokenwise teacher targets against the retained "
            "all-token Gemma router baseline."
        )
    )
    parser.add_argument(
        "--model-name",
        default="google/gemma-2-2b",
        help="TransformerLens model name for tokenwise teacher construction",
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
            "20260318-gemma2-router-distillation-family-comparison-all-tokens-v1/"
            "summary.json"
        ),
        help="Saved family-comparison summary that freezes the split and baseline",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the teacher-target comparison summary",
    )
    parser.add_argument(
        "--candidate-supervision-objectives",
        nargs="+",
        default=(
            "all_tokens_target_mse",
            "next_token_positions_sequence_target_mse",
            "next_token_positions_oracle_alpha_target_logit_contribution_mse",
        ),
        help="Teacher-target paths to compare on the frozen baseline",
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
    fixed_supervision_objective = str(family_summary["fixed_supervision_objective"])
    for input_summary in family_summary["input_summaries"]:
        if (
            str(input_summary["router_family"]) == selected_router_family
            and str(input_summary["input_field"]) == fixed_input_field
            and str(input_summary["target_name"]) == fixed_target_name
            and str(input_summary["aggregation"]) == fixed_aggregation
            and str(input_summary["supervision_objective"])
            == fixed_supervision_objective
        ):
            return dict(input_summary)
    raise ValueError("selected family summary is missing from input_summaries")


def main() -> None:
    from validation.router_distillation import (
        build_oracle_alpha_tokenwise_teacher_lookup,
        compare_router_supervision_objectives,
        compact_router_distillation_supervision_comparison_payload,
        load_router_distillation_pilot_dataset,
    )

    args = parse_args()
    family_summary_path = Path(args.family_summary_path)
    family_summary = json.loads(family_summary_path.read_text())
    baseline_summary = _selected_family_input_summary(family_summary)

    dataset = load_router_distillation_pilot_dataset(Path(args.export_dir))
    prompt_ids = tuple(
        list(family_summary["train_prompt_ids"])
        + list(family_summary["eval_prompt_ids"])
    )
    model = HookedTransformer.from_pretrained(
        args.model_name,
        device=args.device,
        dtype=torch.float32,
    )
    tokenwise_teacher_lookup = build_oracle_alpha_tokenwise_teacher_lookup(
        model=model,
        examples=dataset.examples,
        prompt_ids=prompt_ids,
    )
    summary = compare_router_supervision_objectives(
        examples=dataset.examples,
        fixed_input_field=str(family_summary["fixed_input_field"]),
        fixed_target_name=str(family_summary["fixed_target_name"]),
        fixed_aggregation=str(family_summary["fixed_aggregation"]),
        fixed_router_family=str(family_summary["selected_router_family"]),
        hidden_dim=int(family_summary["hidden_dim"]),
        train_prompt_ids=tuple(family_summary["train_prompt_ids"]),
        eval_prompt_ids=tuple(family_summary["eval_prompt_ids"]),
        candidate_supervision_objectives=tuple(args.candidate_supervision_objectives),
        learning_rate=float(baseline_summary["learning_rate"]),
        weight_decay=float(baseline_summary["weight_decay"]),
        batch_size=int(baseline_summary["batch_size"]),
        max_epochs=int(baseline_summary["max_epochs"]),
        patience=int(baseline_summary["patience"]),
        tokenwise_teacher_lookup=tokenwise_teacher_lookup,
        device=args.device,
        collection_id=dataset.collection_id,
        model_name=dataset.model_name,
    )
    payload = compact_router_distillation_supervision_comparison_payload(summary)
    payload["frozen_family_summary_path"] = str(family_summary_path)
    payload["teacher_target_definition"] = {
        "baseline_path": "repeated_prompt_level_oracle_alpha_logit_vector_on_all_tokens",
        "matched_mask_control": (
            "repeated_prompt_level_oracle_alpha_logit_vector_on_next_token_positions_only"
        ),
        "bounded_tokenwise_path": (
            "oracle-conditioned per-token source preferences from shared-final-norm "
            "pre-softcap contributions to the ground-truth next-token logit"
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
