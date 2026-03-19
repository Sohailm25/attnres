# ABOUTME: Audits why exact tokenwise Gemma teachers fail on the saved bounded subset.
# ABOUTME: Reuses the frozen 7xo subset ids and exact-teacher config to compare teacher variance against sequence-level mismatch.

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
            "Audit whether exact tokenwise oracle teachers fail mainly because "
            "they vary too much within prompts or because they are misaligned "
            "with the scored sequence-level oracle target."
        )
    )
    parser.add_argument(
        "--model-name",
        default="google/gemma-2-2b",
        help="TransformerLens model name for exact tokenwise teacher reconstruction",
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
        "--exact-subset-summary-path",
        default=(
            "results/router_training/"
            "20260319-gemma2-router-distillation-exact-teacher-subset-v1/"
            "summary.json"
        ),
        help="Saved exact-teacher subset summary that freezes the prompt ids/config",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the exact-teacher mismatch audit summary JSON",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Torch device",
    )
    return parser.parse_args()


def main() -> None:
    from validation.router_distillation import (
        audit_exact_tokenwise_teacher_mismatch,
        build_exact_tokenwise_oracle_teacher_lookup,
        compact_router_distillation_teacher_mismatch_payload,
        load_router_distillation_pilot_dataset,
    )

    args = parse_args()
    exact_subset_summary_path = Path(args.exact_subset_summary_path)
    exact_subset_summary = json.loads(exact_subset_summary_path.read_text())
    dataset = load_router_distillation_pilot_dataset(Path(args.export_dir))
    prompt_ids = tuple(exact_subset_summary["train_prompt_ids"]) + tuple(
        exact_subset_summary["eval_prompt_ids"]
    )

    model = HookedTransformer.from_pretrained(
        args.model_name,
        device=args.device,
        dtype=torch.float32,
    )
    tokenwise_teacher_lookup = build_exact_tokenwise_oracle_teacher_lookup(
        model=model,
        examples=dataset.examples,
        prompt_ids=prompt_ids,
        optimization_steps=int(
            exact_subset_summary["exact_teacher_config"]["optimization_steps"]
        ),
        learning_rate=float(
            exact_subset_summary["exact_teacher_config"]["learning_rate"]
        ),
    )
    audit = audit_exact_tokenwise_teacher_mismatch(
        examples=dataset.examples,
        train_prompt_ids=tuple(exact_subset_summary["train_prompt_ids"]),
        eval_prompt_ids=tuple(exact_subset_summary["eval_prompt_ids"]),
        target_name=str(exact_subset_summary["fixed_target_name"]),
        tokenwise_teacher_lookup=tokenwise_teacher_lookup,
        collection_id=dataset.collection_id,
        model_name=dataset.model_name,
    )

    payload = compact_router_distillation_teacher_mismatch_payload(audit)
    payload["exact_subset_summary_path"] = str(exact_subset_summary_path)
    payload["exact_teacher_config"] = dict(exact_subset_summary["exact_teacher_config"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
