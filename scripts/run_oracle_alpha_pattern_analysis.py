# ABOUTME: Runs the prereg-scale sequence-level oracle-alpha pattern-analysis slice on a saved raw run artifact.
# ABOUTME: Writes a compact JSON summary with source-type mass and average-linkage JSD clustering against a random control.

from __future__ import annotations

import argparse
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
            "results/oracle_alpha/"
            "20260317-gpt2xl-prereg-scale-campaign-v4/oracle_eval_run.json"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument("--max-clusters", type=int, default=12)
    parser.add_argument("--num-resamples", type=int, default=64)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--registry-path")
    parser.add_argument("--collection-id")
    parser.add_argument("--group-tag-prefix")
    return parser.parse_args()


def main() -> int:
    from prompts.registry import load_prompt_registry
    from validation.pattern_analysis import write_pattern_analysis_summary

    args = parse_args()
    subset_prompt_ids_by_name = None
    if args.group_tag_prefix:
        if not args.registry_path or not args.collection_id:
            raise ValueError(
                "--group-tag-prefix requires both --registry-path and --collection-id"
            )
        registry = load_prompt_registry(Path(args.registry_path))
        collection = registry.collections[args.collection_id]
        subset_prompt_ids: dict[str, list[str]] = {}
        for entry in collection.prompt_entries:
            for tag in entry.tags:
                if tag.startswith(args.group_tag_prefix):
                    subset_prompt_ids.setdefault(tag, []).append(entry.prompt_id)
        subset_prompt_ids_by_name = subset_prompt_ids
    write_pattern_analysis_summary(
        run_path=Path(args.run_path),
        output_path=Path(args.output),
        random_seed=args.random_seed,
        max_clusters=args.max_clusters,
        num_resamples=args.num_resamples,
        sample_size=args.sample_size,
        subset_prompt_ids_by_name=subset_prompt_ids_by_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
