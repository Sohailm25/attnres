# ABOUTME: Runs the bounded causal refusal-direction intervention check on aligned Gemma.
# ABOUTME: Saves summary-only intervention metrics before any mediator-conditioned safety-routing claim.

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-id", default="safety_refusal_discovery_v1")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-pilot-groups", type=int, default=None)
    parser.add_argument("--max-confirm-groups", type=int, default=None)
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
    from validation.safety_alignment import run_refusal_direction_intervention_check

    args = parse_args()
    config = _load_repo_config()
    model_name = (
        args.model_name
        or config["safety_alignment"]["model_name"]
        or config["models"]["safety_alignment"]["name"]
    )
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

    run_refusal_direction_intervention_check(
        model=model,
        collection_id=args.collection_id,
        output_dir=output_dir,
        max_new_tokens=args.max_new_tokens,
        max_pilot_groups=args.max_pilot_groups,
        max_confirm_groups=args.max_confirm_groups,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
