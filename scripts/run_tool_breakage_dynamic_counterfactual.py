# ABOUTME: Runs the controlled dynamic-routing counterfactual for the Gemma factual-recall lane.
# ABOUTME: Reuses the saved baseline oracle alphas and compares routed traces against prompt-permuted and fixed-alpha controls.

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import sys
import warnings

from huggingface_hub import logging as huggingface_logging
import logging
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


DEFAULT_TUNED_LENS_CHECKPOINT = (
    ROOT
    / "results"
    / "tool_breakage"
    / "20260317-gemma2-tuned-lens-viability-pilot-v1"
    / "checkpoint.pt"
)
DEFAULT_BASELINE_SUMMARY = (
    ROOT
    / "results"
    / "tool_breakage"
    / "20260317-gemma2-tool-breakage-baseline-confirm-v1"
    / "summary.json"
)
DEFAULT_FIXED_ALPHA_SUMMARY = (
    ROOT
    / "results"
    / "tool_breakage"
    / "20260317-gemma2-tool-breakage-baseline-pilot-v1"
    / "summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--tuned-lens-checkpoint",
        default=str(DEFAULT_TUNED_LENS_CHECKPOINT),
    )
    parser.add_argument(
        "--baseline-summary",
        default=str(DEFAULT_BASELINE_SUMMARY),
    )
    parser.add_argument(
        "--fixed-alpha-summary",
        default=str(DEFAULT_FIXED_ALPHA_SUMMARY),
    )
    parser.add_argument("--max-prompts", type=int, default=None)
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
    from validation.tool_breakage import run_tool_breakage_dynamic_counterfactual

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

    run_tool_breakage_dynamic_counterfactual(
        model=model,
        tuned_lens_checkpoint_path=Path(args.tuned_lens_checkpoint),
        baseline_summary_path=Path(args.baseline_summary),
        fixed_alpha_summary_path=Path(args.fixed_alpha_summary),
        output_dir=output_dir,
        max_prompts=args.max_prompts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
