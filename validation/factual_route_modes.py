# ABOUTME: Characterizes within-family factual routing modes from saved oracle-alpha artifacts.
# ABOUTME: Keeps the next oracle step on reusable summaries instead of another model run.

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Mapping, Sequence

import numpy as np

from .oracle_alpha_controls import jensen_shannon_divergence
from .pattern_analysis import (
    assign_average_linkage_clusters,
    summarize_source_type_mass,
)


def _require_single_subcategory_tag(
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


def _top_mass_entries(
    values: Sequence[float],
    source_labels: Sequence[str],
    *,
    limit: int,
    key_name: str,
) -> list[dict[str, float | str]]:
    ranked = sorted(
        (
            {
                "source_label": str(source_label),
                key_name: float(value),
            }
            for source_label, value in zip(source_labels, values, strict=True)
        ),
        key=lambda item: float(item[key_name]),
        reverse=True,
    )
    return ranked[:limit]


def _pairwise_js_summary(mode_means: Sequence[np.ndarray]) -> dict[str, float]:
    if len(mode_means) < 2:
        return {
            "min": 0.0,
            "mean": 0.0,
            "max": 0.0,
        }

    distances = []
    for left_index, left_mode in enumerate(mode_means):
        for right_mode in mode_means[left_index + 1 :]:
            distances.append(
                jensen_shannon_divergence(
                    left_mode.tolist(),
                    right_mode.tolist(),
                )
            )

    return {
        "min": float(min(distances)),
        "mean": float(mean(distances)),
        "max": float(max(distances)),
    }


def _source_type_mass_for_alphas(
    alpha_vectors: Sequence[Sequence[float]],
    source_labels: Sequence[str],
) -> dict[str, float]:
    if len(alpha_vectors) >= 2:
        summary = summarize_source_type_mass(alpha_vectors, source_labels)
        return {
            "mean_embedding_mass": summary.mean_embedding_mass,
            "mean_attention_mass": summary.mean_attention_mass,
            "mean_mlp_mass": summary.mean_mlp_mass,
        }

    alpha = np.asarray(alpha_vectors[0], dtype=float)
    if alpha.ndim != 1:
        raise ValueError("alpha_vectors must contain flat distributions")
    if len(alpha) != len(source_labels):
        raise ValueError("source_labels must match alpha dimension")
    if np.any(alpha < 0.0):
        raise ValueError("alpha values must be non-negative")
    if float(alpha.sum()) <= 0.0:
        raise ValueError("alpha values must sum to a positive value")
    alpha = alpha / float(alpha.sum())

    embedding_mass = 0.0
    attention_mass = 0.0
    mlp_mass = 0.0
    for label, value in zip(source_labels, alpha.tolist(), strict=True):
        if "embed" in label:
            embedding_mass += value
        elif label.endswith("_attn_out") or "_attn_" in label:
            attention_mass += value
        elif label.endswith("_mlp_out") or "mlp" in label:
            mlp_mass += value
        else:
            raise ValueError(f"unsupported source label: {label}")

    return {
        "mean_embedding_mass": float(embedding_mass),
        "mean_attention_mass": float(attention_mass),
        "mean_mlp_mass": float(mlp_mass),
    }


def build_factual_route_mode_summary(
    *,
    prompt_results: Sequence[Mapping[str, object]],
    tags_by_prompt_id: Mapping[str, Sequence[str]],
    cluster_count: int,
    composition_tag_prefix: str = "subcategory_",
    top_sources: int = 5,
    example_prompts: int = 3,
) -> dict[str, object]:
    if cluster_count < 2:
        raise ValueError("cluster_count must be at least 2")
    if top_sources < 1:
        raise ValueError("top_sources must be positive")
    if example_prompts < 1:
        raise ValueError("example_prompts must be positive")

    source_labels = _validated_source_labels(prompt_results)
    assignments = assign_average_linkage_clusters(
        [result["final_alpha"] for result in prompt_results],
        cluster_count=cluster_count,
    )

    family_results: dict[str, list[tuple[Mapping[str, object], int]]] = defaultdict(
        list
    )
    for result, cluster_label in zip(prompt_results, assignments, strict=True):
        prompt_id = str(result["prompt_id"])
        family_tag = _require_single_subcategory_tag(
            prompt_id=prompt_id,
            tags=tags_by_prompt_id[prompt_id],
            composition_tag_prefix=composition_tag_prefix,
        )
        family_results[family_tag].append((result, cluster_label))

    family_summaries = []
    for family_tag in sorted(family_results):
        members = family_results[family_tag]
        family_alphas = np.asarray(
            [member["final_alpha"] for member, _ in members],
            dtype=float,
        )
        family_mean = np.mean(family_alphas, axis=0)
        family_source_type_mass = _source_type_mass_for_alphas(
            family_alphas.tolist(),
            source_labels,
        )
        by_cluster: dict[int, list[Mapping[str, object]]] = defaultdict(list)
        for result, cluster_label in members:
            by_cluster[int(cluster_label)].append(result)

        mode_summaries = []
        mode_means: list[np.ndarray] = []
        for cluster_label, cluster_results in sorted(
            by_cluster.items(),
            key=lambda item: (-len(item[1]), item[0]),
        ):
            mode_alphas = np.asarray(
                [result["final_alpha"] for result in cluster_results],
                dtype=float,
            )
            mode_mean = np.mean(mode_alphas, axis=0)
            mode_means.append(mode_mean)
            source_type_mass = _source_type_mass_for_alphas(
                mode_alphas.tolist(),
                source_labels,
            )
            mode_delta = mode_mean - family_mean
            mode_summaries.append(
                {
                    "cluster_label": int(cluster_label),
                    "size": len(cluster_results),
                    "family_fraction": len(cluster_results) / len(members),
                    "source_type_mass": {
                        "mean_embedding_mass": source_type_mass["mean_embedding_mass"],
                        "mean_attention_mass": source_type_mass["mean_attention_mass"],
                        "mean_mlp_mass": source_type_mass["mean_mlp_mass"],
                    },
                    "source_type_mass_delta_vs_family": {
                        "embedding": float(
                            source_type_mass["mean_embedding_mass"]
                            - family_source_type_mass["mean_embedding_mass"]
                        ),
                        "attention": float(
                            source_type_mass["mean_attention_mass"]
                            - family_source_type_mass["mean_attention_mass"]
                        ),
                        "mlp": float(
                            source_type_mass["mean_mlp_mass"]
                            - family_source_type_mass["mean_mlp_mass"]
                        ),
                    },
                    "top_mean_sources": _top_mass_entries(
                        mode_mean.tolist(),
                        source_labels,
                        limit=top_sources,
                        key_name="mean_mass",
                    ),
                    "top_enriched_sources_vs_family": _top_mass_entries(
                        mode_delta.tolist(),
                        source_labels,
                        limit=top_sources,
                        key_name="delta_mass",
                    ),
                    "example_prompt_ids": [
                        str(result["prompt_id"])
                        for result in cluster_results[:example_prompts]
                    ],
                    "example_prompts": [
                        str(result["prompt"])
                        for result in cluster_results[:example_prompts]
                    ],
                }
            )

        family_summaries.append(
            {
                "family_tag": family_tag,
                "prompt_count": len(members),
                "mode_count": len(by_cluster),
                "mode_sizes": sorted(
                    len(cluster_results) for cluster_results in by_cluster.values()
                ),
                "cluster_labels": sorted(int(label) for label in by_cluster),
                "mode_centroid_js": _pairwise_js_summary(mode_means),
                "family_source_type_mass": {
                    "mean_embedding_mass": family_source_type_mass[
                        "mean_embedding_mass"
                    ],
                    "mean_attention_mass": family_source_type_mass[
                        "mean_attention_mass"
                    ],
                    "mean_mlp_mass": family_source_type_mass["mean_mlp_mass"],
                },
                "family_top_mean_sources": _top_mass_entries(
                    family_mean.tolist(),
                    source_labels,
                    limit=top_sources,
                    key_name="mean_mass",
                ),
                "modes": mode_summaries,
            }
        )

    return {
        "num_sequences": len(prompt_results),
        "num_sources": len(source_labels),
        "cluster_count": cluster_count,
        "num_families": len(family_summaries),
        "family_summaries": family_summaries,
    }
