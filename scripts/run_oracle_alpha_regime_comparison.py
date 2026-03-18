# ABOUTME: Runs the preregistered softmax versus unconstrained versus top-k oracle comparison on a saved prompt surface.
# ABOUTME: Writes per-regime checkpoints plus a compact summary under a dedicated comparison-regimes output directory.

from __future__ import annotations

import argparse
import contextlib
from huggingface_hub import logging as huggingface_logging
import io
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
    parser.add_argument("--split", default="confirm")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--optimization-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", required=True)
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
    from validation.comparison_regimes import write_regime_comparison_summary

    args = parse_args()
    config = _load_repo_config()
    model_name = args.model_name or config["models"]["primary"]["name"]
    device = _resolve_device(args.device)

    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            model = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                dtype="float32",
            )

    write_regime_comparison_summary(
        model=model,
        collection_id=args.collection_id,
        split=args.split,
        exploratory=args.exploratory,
        output_dir=Path(args.output_dir),
        optimization_steps=args.optimization_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
