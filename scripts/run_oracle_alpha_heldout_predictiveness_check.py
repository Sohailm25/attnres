# ABOUTME: Runs the held-out oracle-alpha predictiveness check from the pilot split to the confirm split.
# ABOUTME: Writes a JSON artifact that records the tuned ridge summary, held-out alpha metrics, and runner-stage MIB decision.

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from huggingface_hub import logging as huggingface_logging
import io
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
    parser.add_argument("--collection-id", default="oracle_alpha_phase1_v1")
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-eval-sequences", type=int, default=None)
    parser.add_argument("--optimization-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--target-name", default=None)
    parser.add_argument("--candidate-target-names", nargs="+", default=None)
    parser.add_argument(
        "--regularization-grid",
        type=float,
        nargs="+",
        default=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    )
    parser.add_argument(
        "--candidate-feature-sources",
        nargs="+",
        default=(
            "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat",
            "prompt_shape_scalar_features_v1",
            "mean_pooled_token_embedding",
            "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_prompt_shape_scalar_features_v1_concat",
            "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat",
        ),
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
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
    from validation.oracle_alpha_runner import run_oracle_alpha_predictiveness_check

    args = parse_args()
    config = _load_repo_config()
    model_name = args.model_name or config["models"]["development"]["name"]
    device = _resolve_device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            model = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                dtype="float32",
            )

    summary = run_oracle_alpha_predictiveness_check(
        model=model,
        collection_id=args.collection_id,
        max_train_sequences=args.max_train_sequences,
        max_eval_sequences=args.max_eval_sequences,
        optimization_steps=args.optimization_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        regularization_grid=tuple(args.regularization_grid),
        candidate_feature_sources=tuple(args.candidate_feature_sources),
        candidate_target_names=(
            tuple(args.candidate_target_names)
            if args.candidate_target_names is not None
            else None
        ),
        target_name_override=args.target_name,
    )
    output_path.write_text(json.dumps(asdict(summary), separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
