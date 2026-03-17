# ABOUTME: Runs the bounded original-model tuned-lens viability pilot for the Gemma-2 tool-breakage lane.
# ABOUTME: Writes a checkpoint plus held-out raw-vs-tuned summary before any routed-vs-original analysis begins.

from __future__ import annotations

import argparse
import contextlib
import io
from huggingface_hub import logging as huggingface_logging
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
    parser.add_argument("--train-collection-id", default="oracle_alpha_phase1_v1")
    parser.add_argument("--train-split", default="pilot")
    parser.add_argument(
        "--eval-collection-id",
        default="tool_breakage_factual_recall_v1",
    )
    parser.add_argument("--eval-split", default="pilot")
    parser.add_argument("--max-train-prompts", type=int, default=None)
    parser.add_argument("--max-eval-prompts", type=int, default=None)
    parser.add_argument("--translator-rank", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
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
    from validation.tuned_lens import run_tuned_lens_viability_pilot

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
                dtype="float32",
            )

    run_tuned_lens_viability_pilot(
        model=model,
        train_collection_id=args.train_collection_id,
        train_split=args.train_split,
        eval_collection_id=args.eval_collection_id,
        eval_split=args.eval_split,
        output_dir=output_dir,
        translator_rank=args.translator_rank,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_train_prompts=args.max_train_prompts,
        max_eval_prompts=args.max_eval_prompts,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
