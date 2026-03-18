# ABOUTME: Builds a saved-artifact bridge between the Gemma factual-recall oracle clusters and tool-breakage prompts.
# ABOUTME: Reuses the registry_v5 oracle run plus existing tool-breakage summaries instead of launching another model run.

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
    parser.add_argument(
        "--factual-run-path",
        default=(
            "results/oracle_alpha/"
            "20260318-gemma2-registry-v5-campaign-v1/oracle_eval_run.json"
        ),
    )
    parser.add_argument(
        "--factual-cluster-profile-path",
        default=(
            "results/block_structure/"
            "20260318-gemma2-factual-recall-cluster-profile-v1.json"
        ),
    )
    parser.add_argument("--registry-path", default="prompts/registry_v5.yaml")
    parser.add_argument(
        "--factual-collection-id",
        default="oracle_alpha_phase1_v1",
    )
    parser.add_argument(
        "--tool-breakage-collection-id",
        default="tool_breakage_factual_recall_v1",
    )
    parser.add_argument("--factual-group-tag", default="stratum_factual_recall")
    parser.add_argument(
        "--tool-breakage-pilot-summary-path",
        default=(
            "results/tool_breakage/"
            "20260317-gemma2-tool-breakage-baseline-pilot-v1/summary.json"
        ),
    )
    parser.add_argument(
        "--tool-breakage-confirm-summary-path",
        default=(
            "results/tool_breakage/"
            "20260317-gemma2-tool-breakage-baseline-confirm-v1/summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    from prompts.registry import load_prompt_registry
    from validation.tool_breakage_bridge import (
        build_tool_breakage_factual_bridge_summary,
    )

    args = parse_args()
    registry = load_prompt_registry(Path(args.registry_path))

    factual_run_payload = json.loads(Path(args.factual_run_path).read_text())
    cluster_profile_payload = json.loads(
        Path(args.factual_cluster_profile_path).read_text()
    )
    pilot_summary = json.loads(Path(args.tool_breakage_pilot_summary_path).read_text())
    confirm_summary = json.loads(
        Path(args.tool_breakage_confirm_summary_path).read_text()
    )

    factual_prompt_tags_by_id = {
        entry.prompt_id: entry.tags
        for entry in registry.collections[args.factual_collection_id].prompt_entries
    }
    factual_prompt_results = [
        result
        for result in factual_run_payload["sequence_results"]
        if args.factual_group_tag in factual_prompt_tags_by_id[result["prompt_id"]]
    ]
    tool_prompt_tags_by_id = {
        entry.prompt_id: entry.tags
        for entry in registry.collections[
            args.tool_breakage_collection_id
        ].prompt_entries
    }
    tool_prompt_results = tuple(pilot_summary["prompt_results"]) + tuple(
        confirm_summary["prompt_results"]
    )

    summary = build_tool_breakage_factual_bridge_summary(
        factual_prompt_results=factual_prompt_results,
        factual_tags_by_prompt_id=factual_prompt_tags_by_id,
        tool_prompt_results=tool_prompt_results,
        tool_tags_by_prompt_id=tool_prompt_tags_by_id,
        cluster_count=int(cluster_profile_payload["raw_source_summary"]["best_k"]),
    )
    summary.update(
        {
            "model_name": factual_run_payload["model_name"],
            "factual_group_tag": args.factual_group_tag,
            "factual_run_path": str(Path(args.factual_run_path)),
            "factual_cluster_profile_path": str(
                Path(args.factual_cluster_profile_path)
            ),
            "tool_breakage_pilot_summary_path": str(
                Path(args.tool_breakage_pilot_summary_path)
            ),
            "tool_breakage_confirm_summary_path": str(
                Path(args.tool_breakage_confirm_summary_path)
            ),
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
