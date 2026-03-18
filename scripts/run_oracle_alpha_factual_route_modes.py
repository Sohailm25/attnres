# ABOUTME: Reuses the saved Gemma factual oracle artifact to characterize within-family route modes.
# ABOUTME: Keeps the next oracle step reproducible without relaunching the large registry_v5 campaign.

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
        "--run-path",
        default=(
            "results/oracle_alpha/20260318-gemma2-registry-v5-campaign-v1/"
            "oracle_eval_run.json"
        ),
    )
    parser.add_argument(
        "--cluster-profile-path",
        default=(
            "results/block_structure/"
            "20260318-gemma2-factual-recall-cluster-profile-v1.json"
        ),
    )
    parser.add_argument(
        "--registry-path",
        default="prompts/registry_v5.yaml",
    )
    parser.add_argument(
        "--collection-id",
        default="oracle_alpha_phase1_v1",
    )
    parser.add_argument(
        "--group-tag",
        default="stratum_factual_recall",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-sources", type=int, default=5)
    parser.add_argument("--example-prompts", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    from prompts.registry import load_prompt_registry
    from validation.factual_route_modes import build_factual_route_mode_summary

    args = parse_args()
    run_payload = json.loads(Path(args.run_path).read_text())
    cluster_profile = json.loads(Path(args.cluster_profile_path).read_text())
    registry = load_prompt_registry(Path(args.registry_path))
    collection = registry.collections[args.collection_id]

    prompt_entries = {entry.prompt_id: entry for entry in collection.prompt_entries}
    subset_results = [
        result
        for result in run_payload["sequence_results"]
        if args.group_tag in prompt_entries[str(result["prompt_id"])].tags
    ]
    tags_by_prompt_id = {
        entry.prompt_id: entry.tags for entry in collection.prompt_entries
    }

    summary = build_factual_route_mode_summary(
        prompt_results=subset_results,
        tags_by_prompt_id=tags_by_prompt_id,
        cluster_count=int(cluster_profile["raw_source_summary"]["best_k"]),
        top_sources=args.top_sources,
        example_prompts=args.example_prompts,
    )

    payload = {
        "model_name": run_payload["model_name"],
        "collection_id": run_payload["collection_id"],
        "split": run_payload["split"],
        "group_tag": args.group_tag,
        "run_path": str(Path(args.run_path)),
        "cluster_profile_path": str(Path(args.cluster_profile_path)),
        **summary,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
