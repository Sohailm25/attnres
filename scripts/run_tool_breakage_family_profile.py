# ABOUTME: Builds a saved-artifact family-conditioned donor-arm profile for Gemma tool-breakage.
# ABOUTME: Reuses the existing counterfactual prompt checkpoints instead of launching another model run.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-path", default="prompts/registry_v5.yaml")
    parser.add_argument(
        "--collection-id",
        default="tool_breakage_factual_recall_v3",
    )
    parser.add_argument("--split", default="confirm")
    parser.add_argument(
        "--counterfactual-summary-path",
        default=(
            "results/tool_breakage/"
            "20260318-gemma2-tool-breakage-matched-family-donor-arms-v3/summary.json"
        ),
    )
    parser.add_argument(
        "--prompt-checkpoint-dir",
        default=(
            "results/tool_breakage/"
            "20260318-gemma2-tool-breakage-matched-family-donor-arms-v3/"
            "checkpoints/prompt_results"
        ),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    from prompts.registry import load_prompt_registry
    from validation.tool_breakage_family_profile import (
        build_tool_breakage_family_profile_summary,
    )

    args = parse_args()
    registry = load_prompt_registry(Path(args.registry_path))
    prompt_entries = registry.collections[args.collection_id].prompt_entries
    prompt_tags_by_id = {entry.prompt_id: entry.tags for entry in prompt_entries}

    counterfactual_summary = json.loads(
        Path(args.counterfactual_summary_path).read_text()
    )
    prompt_checkpoint_dir = Path(args.prompt_checkpoint_dir)
    counterfactual_prompt_results = [
        json.loads(path.read_text())
        for path in sorted(prompt_checkpoint_dir.glob("*.json"))
    ]
    summary = build_tool_breakage_family_profile_summary(
        counterfactual_prompt_results=counterfactual_prompt_results,
        prompt_tags_by_id=prompt_tags_by_id,
    )
    summary.update(
        {
            "model_name": counterfactual_summary["model_name"],
            "collection_id": counterfactual_summary["collection_id"],
            "split": counterfactual_summary["split"],
            "baseline_summary_path": counterfactual_summary["baseline_summary_path"],
            "fixed_alpha_summary_path": counterfactual_summary[
                "fixed_alpha_summary_path"
            ],
            "counterfactual_summary_path": str(Path(args.counterfactual_summary_path)),
            "prompt_checkpoint_dir": str(prompt_checkpoint_dir),
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
