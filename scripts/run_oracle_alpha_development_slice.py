# ABOUTME: Runs the first development-model oracle-alpha slice against the saved registries.
# ABOUTME: Produces a small JSON artifact so future work starts from real runner output, not pure config.

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import io
import json
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
transformers_logging.set_verbosity_error()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection-id",
        default="oracle_alpha_phase1_v1",
    )
    parser.add_argument(
        "--split",
        choices=("pilot", "confirm"),
        default="pilot",
    )
    parser.add_argument(
        "--exploratory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-sequences", type=int, default=2)
    parser.add_argument("--optimization-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        required=True,
    )
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
    from validation.oracle_alpha_runner import run_oracle_alpha_collection

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

    summary = run_oracle_alpha_collection(
        model=model,
        collection_id=args.collection_id,
        split=args.split,
        exploratory=args.exploratory,
        max_sequences=args.max_sequences,
        optimization_steps=args.optimization_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    output_path.write_text(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
