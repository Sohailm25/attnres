# ABOUTME: Runs the OIH anchor lane by pairing oracle-alpha predictiveness with a static block-influence baseline.
# ABOUTME: Produces one summary artifact for dynamic-versus-static comparison on the same MCQA prompt split.

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import io
from huggingface_hub import logging as huggingface_logging
import json
import logging
from pathlib import Path
import sys
import warnings

import torch
from transformer_lens import HookedTransformer
from transformers.utils import logging as transformers_logging
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


warnings.filterwarnings(
    "ignore",
    message=r"`torch_dtype` is deprecated! Use `dtype` instead!",
)
huggingface_logging.set_verbosity_error()
logging.getLogger("huggingface_hub.file_download").setLevel(logging.CRITICAL)
transformers_logging.set_verbosity_error()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-id", default="resattn_oih_mcqa_v1")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-eval-sequences", type=int, default=None)
    parser.add_argument("--optimization-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--regularization-grid",
        type=float,
        nargs="+",
        default=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    )
    parser.add_argument(
        "--feature-source",
        default="position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat",
    )
    parser.add_argument(
        "--target-name",
        default="block_compressed_alpha_logit_vector",
    )
    parser.add_argument("--prune-fraction", type=float, default=0.25)
    return parser.parse_args()


def _load_repo_config() -> dict[str, object]:
    return yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())


def _resolve_device(requested_device: str | None) -> str:
    if requested_device:
        if requested_device == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return requested_device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    from prompts import resolve_prompt_entries
    from validation.oracle_alpha_campaign import (
        run_oracle_alpha_predictiveness_campaign,
    )
    from validation.oracle_alpha_controls import load_oracle_alpha_control_registry
    from validation.shortgpt_baseline import (
        evaluate_static_block_influence_baseline,
        fit_static_block_influence_baseline,
        mean_alpha_policy_from_sequences,
        select_best_static_evaluation,
        StaticBlockInfluenceFitSummary,
    )

    args = parse_args()
    config = _load_repo_config()
    model_name = args.model_name or config["models"]["primary"]["name"]
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            model = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                dtype=torch.float32,
            )

    predictiveness_summary = run_oracle_alpha_predictiveness_campaign(
        model=model,
        collection_id=args.collection_id,
        output_dir=output_dir,
        max_train_sequences=args.max_train_sequences,
        max_eval_sequences=args.max_eval_sequences,
        optimization_steps=args.optimization_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        regularization_grid=tuple(args.regularization_grid),
        candidate_feature_sources=(args.feature_source,),
        candidate_target_names=(args.target_name,),
    )

    control_registry = load_oracle_alpha_control_registry()
    predictiveness_plan = control_registry.plans[args.collection_id].predictiveness
    train_entries = list(
        resolve_prompt_entries(
            collection_id=predictiveness_plan.train_collection_id,
            split=predictiveness_plan.train_split,
            exploratory=True,
        )
    )
    eval_entries = list(
        resolve_prompt_entries(
            collection_id=predictiveness_plan.eval_collection_id,
            split=predictiveness_plan.eval_split,
            exploratory=False,
        )
    )
    if args.max_train_sequences is not None:
        train_entries = train_entries[: args.max_train_sequences]
    if args.max_eval_sequences is not None:
        eval_entries = eval_entries[: args.max_eval_sequences]

    static_fit = fit_static_block_influence_baseline(
        model=model,
        prompt_entries=train_entries,
        prune_fraction=args.prune_fraction,
    )
    static_eval = evaluate_static_block_influence_baseline(
        model=model,
        prompt_entries=eval_entries,
        fit_summary=static_fit,
    )
    pilot_mean_alpha = mean_alpha_policy_from_sequences(
        tuple(
            sequence_result.final_alpha
            for sequence_result in predictiveness_summary.train_run.sequence_results
        )
    )
    pilot_mean_fit = StaticBlockInfluenceFitSummary(
        source_labels=static_fit.source_labels,
        group_names=tuple(),
        mean_group_influences=tuple(),
        kept_group_indices=tuple(),
        pruned_group_indices=tuple(),
        static_alpha=pilot_mean_alpha,
    )
    pilot_mean_eval = evaluate_static_block_influence_baseline(
        model=model,
        prompt_entries=eval_entries,
        fit_summary=pilot_mean_fit,
    )
    calibrated_label, calibrated_eval = select_best_static_evaluation(
        {
            "pruned_block_influence": static_eval,
            "pilot_mean_alpha": pilot_mean_eval,
        }
    )

    summary_payload = {
        "collection_id": args.collection_id,
        "model_name": model.cfg.model_name,
        "feature_source": args.feature_source,
        "target_name": args.target_name,
        "prune_fraction": args.prune_fraction,
        "predictiveness_summary": {
            "predicted_mean_improvement_over_uniform": (
                predictiveness_summary.predicted_mean_improvement_over_uniform
            ),
            "oracle_mean_improvement_over_uniform": (
                predictiveness_summary.oracle_mean_improvement_over_uniform
            ),
            "r_squared": predictiveness_summary.predictiveness_summary.r_squared,
            "mean_js_divergence": (
                predictiveness_summary.predictiveness_summary.mean_js_divergence
            ),
            "selected_regularization_strength": (
                predictiveness_summary.selected_regularization_strength
            ),
            "mib_status": predictiveness_summary.mib_status,
            "mib_rationale": predictiveness_summary.mib_rationale,
        },
        "static_block_influence_fit": asdict(static_fit),
        "static_block_influence_eval": asdict(static_eval),
        "pilot_mean_alpha_static_eval": asdict(pilot_mean_eval),
        "calibrated_static_baseline": {
            "selected_policy": calibrated_label,
            "summary": asdict(calibrated_eval),
        },
        "comparison": {
            "predicted_minus_static_improvement": (
                predictiveness_summary.predicted_mean_improvement_over_uniform
                - static_eval.mean_improvement_over_uniform
            ),
            "oracle_minus_static_improvement": (
                predictiveness_summary.oracle_mean_improvement_over_uniform
                - static_eval.mean_improvement_over_uniform
            ),
            "predicted_minus_calibrated_static_improvement": (
                predictiveness_summary.predicted_mean_improvement_over_uniform
                - calibrated_eval.mean_improvement_over_uniform
            ),
            "oracle_minus_calibrated_static_improvement": (
                predictiveness_summary.oracle_mean_improvement_over_uniform
                - calibrated_eval.mean_improvement_over_uniform
            ),
        },
    }

    summary_path = output_dir / "oih_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
