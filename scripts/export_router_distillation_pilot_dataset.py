# ABOUTME: Exports pilot-only per-token hidden states joined to saved oracle targets for Gemma router distillation.
# ABOUTME: Reuses saved oracle checkpoints and resumable prompt exports instead of rerunning oracle optimization.

from __future__ import annotations

import argparse
from dataclasses import asdict
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
            "Export pilot-only per-token hidden states joined to saved oracle "
            "targets for Phase 6 router distillation."
        )
    )
    parser.add_argument(
        "--model-name",
        default="google/gemma-2-2b",
        help="TransformerLens model name",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Torch device for state export",
    )
    parser.add_argument(
        "--collection-id",
        default="oracle_alpha_phase1_v1",
        help="Prompt collection id to export from the pilot split",
    )
    parser.add_argument(
        "--campaign-dir",
        required=True,
        help="Saved oracle campaign directory containing checkpoints/oracle_runs/pilot",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the export summary, manifest, and prompt checkpoints",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional cap for a calibration slice",
    )
    return parser.parse_args()


def main() -> None:
    from validation.router_training_export import run_router_distillation_pilot_export

    args = parse_args()
    campaign_dir = Path(args.campaign_dir)
    oracle_checkpoint_dir = campaign_dir / "checkpoints" / "oracle_runs" / "pilot"
    output_dir = Path(args.output_dir)

    model = HookedTransformer.from_pretrained(
        args.model_name,
        device=args.device,
        dtype=torch.float32,
    )
    summary = run_router_distillation_pilot_export(
        model=model,
        collection_id=args.collection_id,
        oracle_checkpoint_dir=oracle_checkpoint_dir,
        output_dir=output_dir,
        max_sequences=args.max_sequences,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
