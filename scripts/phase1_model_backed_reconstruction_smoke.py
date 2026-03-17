# ABOUTME: Runs the first model-backed Phase 1 reconstruction smoke check and saves the metrics.
# ABOUTME: Uses the configured development model to verify cached sublayer writes reconstruct logits.

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
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings(
    "ignore",
    message=r"`torch_dtype` is deprecated! Use `dtype` instead!",
)
transformers_logging.set_verbosity_error()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def resolve_device(requested_device: str, fallback_device: str) -> str:
    if requested_device == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested_device == "cpu":
        return "cpu"
    if fallback_device == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    from validation import model_backed_reconstruction_metrics

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override the configured development model name.",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Prompt used for the reconstruction smoke check.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cpu"],
        help="Preferred execution device.",
    )
    parser.add_argument(
        "--fallback-device",
        default="cpu",
        choices=["mps", "cpu"],
        help="Fallback device if the preferred device is unavailable.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the JSON metrics artifact.",
    )
    args = parser.parse_args()

    config = load_config()
    model_name = args.model_name or config["models"]["development"]["name"]
    device = resolve_device(args.device, args.fallback_device)

    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            model = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                dtype="float32",
            )
    metrics = model_backed_reconstruction_metrics(
        model=model,
        prompt=args.prompt,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(metrics), indent=2))

    print(json.dumps(asdict(metrics), indent=2))


if __name__ == "__main__":
    main()
