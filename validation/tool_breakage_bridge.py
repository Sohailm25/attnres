# ABOUTME: Bridges saved factual-recall oracle cluster structure to the bounded Gemma tool-breakage prompts.
# ABOUTME: Keeps this lane on reusable artifact analysis rather than another model run.

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.distance import jensenshannon

from .pattern_analysis import assign_average_linkage_clusters


def _require_single_subcategory(
    *,
    prompt_id: str,
    tags: Sequence[str],
    composition_tag_prefix: str,
) -> str:
    subcategory_tags = [tag for tag in tags if tag.startswith(composition_tag_prefix)]
    if len(subcategory_tags) != 1:
        raise ValueError(
            f"prompt {prompt_id!r} must define exactly one {composition_tag_prefix} tag"
        )
    return subcategory_tags[0]


def _validated_source_labels(
    prompt_results: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    if not prompt_results:
        raise ValueError("prompt_results must not be empty")

    source_labels = tuple(str(label) for label in prompt_results[0]["source_labels"])
    for result in prompt_results[1:]:
        candidate = tuple(str(label) for label in result["source_labels"])
        if candidate != source_labels:
            raise ValueError("all prompt results must share the same source labels")
    return source_labels


def _js_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return float(
        jensenshannon(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    )


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _prompt_rank_deltas(
    layer_traces: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    if not layer_traces:
        raise ValueError("tool-breakage prompt results must include layer traces")

    original_ranks = [
        int(trace["tuned_original_final_position_target_rank"])
        for trace in layer_traces
    ]
    routed_ranks = [
        int(trace["tuned_routed_final_position_target_rank"]) for trace in layer_traces
    ]
    final_trace = layer_traces[-1]
    return {
        "tuned_final_target_rank_delta_under_routing": float(
            routed_ranks[-1] - original_ranks[-1]
        ),
        "tuned_best_target_rank_delta_under_routing": float(
            min(routed_ranks) - min(original_ranks)
        ),
        "tuned_target_rank_range_delta_under_routing": float(
            (max(routed_ranks) - min(routed_ranks))
            - (max(original_ranks) - min(original_ranks))
        ),
        "tuned_final_position_kl_delta_under_routing": float(
            float(final_trace["tuned_routed_final_position_kl_to_final"])
            - float(final_trace["tuned_original_final_position_kl_to_final"])
        ),
    }


def build_tool_breakage_factual_bridge_summary(
    *,
    factual_prompt_results: Sequence[Mapping[str, object]],
    factual_tags_by_prompt_id: Mapping[str, Sequence[str]],
    tool_prompt_results: Sequence[Mapping[str, object]],
    tool_tags_by_prompt_id: Mapping[str, Sequence[str]],
    cluster_count: int,
    composition_tag_prefix: str = "subcategory_",
) -> dict[str, object]:
    if cluster_count < 2:
        raise ValueError("cluster_count must be at least 2")

    factual_source_labels = _validated_source_labels(factual_prompt_results)
    tool_source_labels = _validated_source_labels(tool_prompt_results)
    if tool_source_labels != factual_source_labels:
        raise ValueError("factual and tool-breakage source labels must match exactly")

    factual_alphas = [
        tuple(float(value) for value in result["final_alpha"])
        for result in factual_prompt_results
    ]
    assignments = assign_average_linkage_clusters(
        factual_alphas,
        cluster_count=cluster_count,
    )
    factual_subcategories = {
        _require_single_subcategory(
            prompt_id=str(result["prompt_id"]),
            tags=factual_tags_by_prompt_id[str(result["prompt_id"])],
            composition_tag_prefix=composition_tag_prefix,
        )
        for result in factual_prompt_results
    }

    cluster_payloads: dict[int, dict[str, object]] = {}
    for cluster_label in sorted(set(assignments)):
        member_results = [
            result
            for result, assigned_label in zip(
                factual_prompt_results,
                assignments,
                strict=True,
            )
            if assigned_label == cluster_label
        ]
        member_subcategories = [
            _require_single_subcategory(
                prompt_id=str(result["prompt_id"]),
                tags=factual_tags_by_prompt_id[str(result["prompt_id"])],
                composition_tag_prefix=composition_tag_prefix,
            )
            for result in member_results
        ]
        dominant_subcategory, dominant_count = Counter(
            member_subcategories
        ).most_common(1)[0]
        cluster_payloads[cluster_label] = {
            "cluster_label": cluster_label,
            "size": len(member_results),
            "dominant_subcategory": dominant_subcategory,
            "dominant_subcategory_fraction": dominant_count / len(member_results),
            "member_prompt_ids": [
                str(result["prompt_id"]) for result in member_results
            ],
            "mean_alpha": np.mean(
                np.asarray(
                    [result["final_alpha"] for result in member_results],
                    dtype=float,
                ),
                axis=0,
            ).tolist(),
        }

    prompt_assignments: list[dict[str, object]] = []
    for result in tool_prompt_results:
        prompt_id = str(result["prompt_id"])
        tool_subcategory = _require_single_subcategory(
            prompt_id=prompt_id,
            tags=tool_tags_by_prompt_id[prompt_id],
            composition_tag_prefix=composition_tag_prefix,
        )
        nearest_cluster = min(
            cluster_payloads.values(),
            key=lambda cluster: _js_distance(
                result["oracle_alpha"], cluster["mean_alpha"]
            ),
        )
        prompt_assignment = {
            "prompt_id": prompt_id,
            "prompt": str(result["prompt"]),
            "split": str(result["split"]),
            "tool_subcategory": tool_subcategory,
            "overlaps_factual_cluster_families": tool_subcategory
            in factual_subcategories,
            "nearest_cluster_label": int(nearest_cluster["cluster_label"]),
            "nearest_cluster_dominant_subcategory": str(
                nearest_cluster["dominant_subcategory"]
            ),
            "nearest_cluster_js_distance": _js_distance(
                result["oracle_alpha"],
                nearest_cluster["mean_alpha"],
            ),
            "matches_nearest_cluster_subcategory": (
                tool_subcategory == nearest_cluster["dominant_subcategory"]
            ),
        }
        prompt_assignment.update(_prompt_rank_deltas(result["layer_traces"]))
        prompt_assignments.append(prompt_assignment)

    def summarize_group(
        group_assignments: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        return {
            "prompt_count": len(group_assignments),
            "confirm_prompt_count": sum(
                1 for item in group_assignments if item["split"] == "confirm"
            ),
            "matching_subcategory_fraction": float(
                mean(
                    1.0 if item["matches_nearest_cluster_subcategory"] else 0.0
                    for item in group_assignments
                )
            ),
            "mean_js_distance_to_nearest_cluster": float(
                mean(item["nearest_cluster_js_distance"] for item in group_assignments)
            ),
            "mean_tuned_final_position_kl_delta_under_routing": float(
                mean(
                    item["tuned_final_position_kl_delta_under_routing"]
                    for item in group_assignments
                )
            ),
            "mean_tuned_final_target_rank_delta_under_routing": float(
                mean(
                    item["tuned_final_target_rank_delta_under_routing"]
                    for item in group_assignments
                )
            ),
            "mean_tuned_best_target_rank_delta_under_routing": float(
                mean(
                    item["tuned_best_target_rank_delta_under_routing"]
                    for item in group_assignments
                )
            ),
            "mean_tuned_target_rank_range_delta_under_routing": float(
                mean(
                    item["tuned_target_rank_range_delta_under_routing"]
                    for item in group_assignments
                )
            ),
        }

    by_tool_subcategory = {
        subcategory: summarize_group(
            [
                item
                for item in prompt_assignments
                if item["tool_subcategory"] == subcategory
            ]
        )
        for subcategory in sorted(
            {item["tool_subcategory"] for item in prompt_assignments}
        )
    }
    by_nearest_cluster_subcategory = {
        subcategory: summarize_group(
            [
                item
                for item in prompt_assignments
                if item["nearest_cluster_dominant_subcategory"] == subcategory
            ]
        )
        for subcategory in sorted(
            {
                item["nearest_cluster_dominant_subcategory"]
                for item in prompt_assignments
            }
        )
    }

    overlap_assignments = [
        item for item in prompt_assignments if item["overlaps_factual_cluster_families"]
    ]
    return {
        "num_factual_sequences": len(factual_prompt_results),
        "num_factual_clusters": len(cluster_payloads),
        "num_tool_prompts": len(tool_prompt_results),
        "factual_subcategories": sorted(factual_subcategories),
        "cluster_summaries": [
            {
                key: value
                for key, value in cluster_payload.items()
                if key != "mean_alpha"
            }
            for cluster_payload in sorted(
                cluster_payloads.values(),
                key=lambda item: int(item["cluster_label"]),
            )
        ],
        "overlap_prompt_fraction": float(
            mean(
                1.0 if item["overlaps_factual_cluster_families"] else 0.0
                for item in prompt_assignments
            )
        ),
        "matching_subcategory_fraction": float(
            mean(
                1.0 if item["matches_nearest_cluster_subcategory"] else 0.0
                for item in prompt_assignments
            )
        ),
        "matching_subcategory_fraction_on_overlap_prompts": _mean_or_none(
            [
                1.0 if item["matches_nearest_cluster_subcategory"] else 0.0
                for item in overlap_assignments
            ]
        ),
        "prompt_assignments": prompt_assignments,
        "by_tool_subcategory": by_tool_subcategory,
        "by_nearest_cluster_subcategory": by_nearest_cluster_subcategory,
    }
