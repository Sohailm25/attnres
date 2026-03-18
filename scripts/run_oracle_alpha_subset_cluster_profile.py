# ABOUTME: Profiles raw-source cluster assignments for one tagged subset of a saved oracle-alpha run.
# ABOUTME: Keeps the next factual-recall-style follow-up reproducible without relaunching a large oracle campaign.

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-path",
        required=True,
    )
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--group-tag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument("--max-clusters", type=int, default=12)
    parser.add_argument("--composition-tag-prefix", default="subcategory_")
    parser.add_argument("--top-sources", type=int, default=5)
    parser.add_argument("--example-prompts", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    from prompts.registry import load_prompt_registry
    from validation.pattern_analysis import (
        assign_average_linkage_clusters,
        build_sequence_level_pattern_summary,
        summarize_source_type_mass,
    )

    args = parse_args()
    run_payload = json.loads(Path(args.run_path).read_text())
    registry = load_prompt_registry(Path(args.registry_path))
    collection = registry.collections[args.collection_id]
    prompt_entries = {entry.prompt_id: entry for entry in collection.prompt_entries}

    subset_results = [
        result
        for result in run_payload["sequence_results"]
        if args.group_tag in prompt_entries[result["prompt_id"]].tags
    ]
    if len(subset_results) < 2:
        raise ValueError("subset must contain at least two matching prompts")

    summary = build_sequence_level_pattern_summary(
        subset_results,
        random_seed=args.random_seed,
        max_clusters=args.max_clusters,
    )
    best_k = summary.oracle_cluster_scan.best_k
    assignments = assign_average_linkage_clusters(
        [result["final_alpha"] for result in subset_results],
        cluster_count=best_k,
    )
    source_labels = tuple(subset_results[0]["source_labels"])

    cluster_profiles = []
    subcategory_to_clusters: defaultdict[str, set[int]] = defaultdict(set)
    for cluster_label in sorted(
        set(assignments), key=lambda label: assignments.count(label), reverse=True
    ):
        cluster_results = [
            result
            for result, assignment in zip(subset_results, assignments, strict=True)
            if assignment == cluster_label
        ]
        subcategory_counts = Counter(
            next(
                tag
                for tag in prompt_entries[result["prompt_id"]].tags
                if tag.startswith(args.composition_tag_prefix)
            )
            for result in cluster_results
        )
        for subcategory in subcategory_counts:
            subcategory_to_clusters[subcategory].add(cluster_label)
        mean_alpha = np.mean(
            np.asarray(
                [result["final_alpha"] for result in cluster_results], dtype=float
            ),
            axis=0,
        )
        top_sources = sorted(
            (
                {
                    "source_label": source_label,
                    "mean_mass": float(mean_mass),
                }
                for source_label, mean_mass in zip(
                    source_labels,
                    mean_alpha.tolist(),
                    strict=True,
                )
            ),
            key=lambda item: item["mean_mass"],
            reverse=True,
        )[: args.top_sources]
        source_type_mass = summarize_source_type_mass(
            [result["final_alpha"] for result in cluster_results],
            source_labels,
        )
        dominant_subcategory, dominant_count = subcategory_counts.most_common(1)[0]
        cluster_profiles.append(
            {
                "cluster_label": cluster_label,
                "size": len(cluster_results),
                "dominant_subcategory": dominant_subcategory,
                "dominant_subcategory_fraction": dominant_count / len(cluster_results),
                "subcategory_counts": dict(sorted(subcategory_counts.items())),
                "source_type_mass": {
                    "mean_embedding_mass": source_type_mass.mean_embedding_mass,
                    "mean_attention_mass": source_type_mass.mean_attention_mass,
                    "mean_mlp_mass": source_type_mass.mean_mlp_mass,
                },
                "top_mean_sources": top_sources,
                "example_prompt_ids": [
                    result["prompt_id"]
                    for result in cluster_results[: args.example_prompts]
                ],
                "example_prompts": [
                    result["prompt"]
                    for result in cluster_results[: args.example_prompts]
                ],
            }
        )

    payload = {
        "model_name": run_payload["model_name"],
        "collection_id": run_payload["collection_id"],
        "split": run_payload["split"],
        "group_tag": args.group_tag,
        "num_sequences": len(subset_results),
        "num_sources": summary.num_sources,
        "raw_source_summary": {
            "best_k": best_k,
            "best_silhouette": summary.oracle_cluster_scan.best_silhouette,
            "random_best_silhouette": summary.random_control_cluster_scan.best_silhouette,
            "cluster_sizes": summary.oracle_cluster_scan.cluster_sizes_by_k[best_k],
            "largest_cluster_fraction": summary.oracle_cluster_scan.largest_cluster_fraction_by_k[
                best_k
            ],
        },
        "mean_entropy": summary.mean_entropy,
        "mean_effective_sources": summary.mean_effective_sources,
        "mean_top1_mass": summary.mean_top1_mass,
        "subcategory_cluster_support": {
            subcategory: len(cluster_labels)
            for subcategory, cluster_labels in sorted(subcategory_to_clusters.items())
        },
        "cluster_profiles": cluster_profiles,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
