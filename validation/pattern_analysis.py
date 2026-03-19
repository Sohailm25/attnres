# ABOUTME: Analyzes sequence-level oracle-alpha routing patterns from saved run artifacts.
# ABOUTME: Keeps prereg-scale clustering and source-type summaries reusable and aligned with repo guardrails.

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .oracle_alpha_controls import (
    bootstrap_mean_confidence_interval,
    jensen_shannon_divergence,
)


@dataclass(frozen=True)
class SourceTypeMassSummary:
    mean_embedding_mass: float
    mean_attention_mass: float
    mean_mlp_mass: float


@dataclass(frozen=True)
class ClusterScanResult:
    linkage_method: str
    best_k: int
    best_silhouette: float
    silhouette_by_k: dict[int, float]
    cluster_sizes_by_k: dict[int, tuple[int, ...]] = field(default_factory=dict)
    largest_cluster_fraction_by_k: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupedViewPatternSummary:
    view_name: str
    grouped_source_labels: tuple[str, ...]
    summary: SequenceLevelPatternSummary


@dataclass(frozen=True)
class PromptResamplingStabilitySummary:
    view_name: str
    num_resamples: int
    sample_size: int
    oracle_best_silhouette_mean: float
    oracle_best_silhouette_std: float
    random_best_silhouette_mean: float
    random_best_silhouette_std: float
    oracle_beats_random_fraction: float
    oracle_best_k_counts: dict[int, int]
    oracle_largest_cluster_fraction_mean_by_k: dict[int, float]
    random_largest_cluster_fraction_mean_by_k: dict[int, float]


@dataclass(frozen=True)
class SequenceLevelPatternSummary:
    num_sequences: int
    num_sources: int
    mean_entropy: float
    mean_effective_sources: float
    mean_top1_mass: float
    top1_source_counts: dict[str, int]
    source_type_mass: SourceTypeMassSummary
    oracle_cluster_scan: ClusterScanResult
    random_control_cluster_scan: ClusterScanResult


def _validated_distribution_matrix(
    distributions: Sequence[Sequence[float]],
) -> np.ndarray:
    matrix = np.asarray(distributions, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("distributions must be rank-2")
    if matrix.shape[0] < 2:
        raise ValueError("at least two distributions are required")
    if matrix.shape[1] < 1:
        raise ValueError("distributions must contain at least one source")
    if np.any(matrix < 0.0):
        raise ValueError("distribution values must be non-negative")
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("each distribution must sum to a positive value")
    return matrix / row_sums


def _validated_source_labels(
    source_labels: Sequence[str],
    *,
    expected_num_sources: int,
) -> tuple[str, ...]:
    labels = tuple(source_labels)
    if not labels:
        raise ValueError("source_labels must not be empty")
    if len(labels) != expected_num_sources:
        raise ValueError("source_labels must match the source dimension")
    return labels


def _js_distance_matrix(distributions: np.ndarray) -> np.ndarray:
    num_sequences = distributions.shape[0]
    distances = np.zeros((num_sequences, num_sequences), dtype=float)
    for left_index in range(num_sequences):
        for right_index in range(left_index + 1, num_sequences):
            distance = jensen_shannon_divergence(
                distributions[left_index].tolist(),
                distributions[right_index].tolist(),
            )
            distances[left_index, right_index] = distance
            distances[right_index, left_index] = distance
    return distances


def _cluster_sizes(assignments: np.ndarray) -> tuple[int, ...]:
    counts = Counter(int(label) for label in assignments.tolist())
    return tuple(sorted(counts.values(), reverse=True))


def _largest_cluster_fraction(assignments: np.ndarray) -> float:
    sizes = _cluster_sizes(assignments)
    if not sizes:
        return 0.0
    return float(sizes[0] / assignments.shape[0])


def _silhouette_score_from_distance_matrix(
    distances: np.ndarray,
    assignments: np.ndarray,
) -> float:
    if distances.shape[0] != distances.shape[1]:
        raise ValueError("distance matrix must be square")
    if distances.shape[0] != assignments.shape[0]:
        raise ValueError("distance matrix and assignments must align")

    unique_labels = np.unique(assignments)
    if unique_labels.size < 2:
        return 0.0

    silhouette_values: list[float] = []
    for item_index, cluster_label in enumerate(assignments.tolist()):
        same_cluster = np.flatnonzero(assignments == cluster_label)
        if same_cluster.size <= 1:
            silhouette_values.append(0.0)
            continue

        same_cluster = same_cluster[same_cluster != item_index]
        intra_cluster_distance = float(distances[item_index, same_cluster].mean())

        nearest_other_cluster_distance = math.inf
        for other_label in unique_labels.tolist():
            if other_label == cluster_label:
                continue
            other_cluster = np.flatnonzero(assignments == other_label)
            if other_cluster.size == 0:
                continue
            candidate_distance = float(distances[item_index, other_cluster].mean())
            nearest_other_cluster_distance = min(
                nearest_other_cluster_distance,
                candidate_distance,
            )

        if not math.isfinite(nearest_other_cluster_distance):
            silhouette_values.append(0.0)
            continue

        denominator = max(intra_cluster_distance, nearest_other_cluster_distance)
        if denominator <= 0.0:
            silhouette_values.append(0.0)
            continue

        silhouette_values.append(
            (nearest_other_cluster_distance - intra_cluster_distance) / denominator
        )

    return float(sum(silhouette_values) / len(silhouette_values))


def summarize_source_type_mass(
    alpha_vectors: Sequence[Sequence[float]],
    source_labels: Sequence[str],
) -> SourceTypeMassSummary:
    if len(alpha_vectors) == 0:
        raise ValueError("alpha_vectors must not be empty")

    if len(alpha_vectors) == 1:
        distribution = np.asarray(alpha_vectors[0], dtype=float)
        if distribution.ndim != 1:
            raise ValueError("single alpha vector must be rank-1")
        if np.any(distribution < 0.0):
            raise ValueError("distribution values must be non-negative")
        total_mass = float(distribution.sum())
        if total_mass <= 0.0:
            raise ValueError("distribution values must sum to a positive value")
        distributions = (distribution / total_mass).reshape(1, -1)
    else:
        distributions = _validated_distribution_matrix(alpha_vectors)

    labels = _validated_source_labels(
        source_labels,
        expected_num_sources=distributions.shape[1],
    )

    embedding_indices = [
        index
        for index, label in enumerate(labels)
        if label in {"embed", "pos_embed", "embedding"} or "embed" in label
    ]
    attention_indices = [
        index
        for index, label in enumerate(labels)
        if label.endswith("_attn_out") or "attention" in label or "_attn_" in label
    ]
    mlp_indices = [
        index
        for index, label in enumerate(labels)
        if label.endswith("_mlp_out") or "mlp" in label
    ]

    if not embedding_indices:
        raise ValueError("source_labels must include embedding sources")
    if not attention_indices:
        raise ValueError("source_labels must include attention sources")
    if not mlp_indices:
        raise ValueError("source_labels must include mlp sources")

    return SourceTypeMassSummary(
        mean_embedding_mass=float(
            distributions[:, embedding_indices].sum(axis=1).mean()
        ),
        mean_attention_mass=float(
            distributions[:, attention_indices].sum(axis=1).mean()
        ),
        mean_mlp_mass=float(distributions[:, mlp_indices].sum(axis=1).mean()),
    )


def scan_average_linkage_clusters(
    distributions: Sequence[Sequence[float]],
    *,
    max_clusters: int,
) -> ClusterScanResult:
    if max_clusters < 2:
        raise ValueError("max_clusters must be at least 2")

    distribution_matrix = _validated_distribution_matrix(distributions)
    distance_matrix = _js_distance_matrix(distribution_matrix)
    condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed, method="average")

    upper_k = min(max_clusters, distribution_matrix.shape[0])
    silhouette_by_k: dict[int, float] = {}
    cluster_sizes_by_k: dict[int, tuple[int, ...]] = {}
    largest_cluster_fraction_by_k: dict[int, float] = {}
    best_k = 2
    best_silhouette = float("-inf")

    for cluster_count in range(2, upper_k + 1):
        if cluster_count == distribution_matrix.shape[0]:
            silhouette = 0.0
            assignments = np.arange(1, distribution_matrix.shape[0] + 1)
        else:
            assignments = fcluster(
                linkage_matrix,
                t=cluster_count,
                criterion="maxclust",
            )
            silhouette = _silhouette_score_from_distance_matrix(
                distance_matrix,
                assignments,
            )
        silhouette_by_k[cluster_count] = float(silhouette)
        cluster_sizes_by_k[cluster_count] = _cluster_sizes(assignments)
        largest_cluster_fraction_by_k[cluster_count] = _largest_cluster_fraction(
            assignments
        )
        if silhouette > best_silhouette:
            best_k = cluster_count
            best_silhouette = float(silhouette)

    return ClusterScanResult(
        linkage_method="average",
        best_k=best_k,
        best_silhouette=best_silhouette,
        silhouette_by_k=silhouette_by_k,
        cluster_sizes_by_k=cluster_sizes_by_k,
        largest_cluster_fraction_by_k=largest_cluster_fraction_by_k,
    )


def assign_average_linkage_clusters(
    distributions: Sequence[Sequence[float]],
    *,
    cluster_count: int,
) -> tuple[int, ...]:
    if cluster_count < 2:
        raise ValueError("cluster_count must be at least 2")

    distribution_matrix = _validated_distribution_matrix(distributions)
    if cluster_count > distribution_matrix.shape[0]:
        raise ValueError("cluster_count must not exceed the sequence count")
    if cluster_count == distribution_matrix.shape[0]:
        return tuple(range(1, distribution_matrix.shape[0] + 1))

    distance_matrix = _js_distance_matrix(distribution_matrix)
    condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    assignments = fcluster(
        linkage_matrix,
        t=cluster_count,
        criterion="maxclust",
    )
    return tuple(int(value) for value in assignments.tolist())


def _source_type_for_label(label: str) -> str:
    if label in {"embed", "pos_embed", "embedding"} or "embed" in label:
        return "embedding"
    if label.endswith("_attn_out") or "attention" in label or "_attn_" in label:
        return "attention"
    if label.endswith("_mlp_out") or "mlp" in label:
        return "mlp"
    raise ValueError(f"unsupported source label: {label}")


def _layer_index_for_label(label: str) -> int | None:
    if _source_type_for_label(label) == "embedding":
        return None
    prefix, _, _ = label.partition("_")
    if not prefix.isdigit():
        raise ValueError(f"could not parse layer index from source label: {label}")
    return int(prefix)


def _depth_band_name(layer_index: int, *, num_layers: int) -> str:
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    band_index = min(2, int(layer_index * 3 / num_layers))
    return ("early", "middle", "late")[band_index]


def _group_distributions_for_view(
    distributions: np.ndarray,
    source_labels: Sequence[str],
    *,
    view_name: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if view_name == "raw_source":
        return distributions, _validated_source_labels(
            source_labels,
            expected_num_sources=distributions.shape[1],
        )

    labels = _validated_source_labels(
        source_labels,
        expected_num_sources=distributions.shape[1],
    )
    if view_name == "source_type":
        grouped_labels = ("embedding", "attention", "mlp")
        label_to_group = {name: index for index, name in enumerate(grouped_labels)}
        grouped = np.zeros((distributions.shape[0], len(grouped_labels)), dtype=float)
        for source_index, label in enumerate(labels):
            group_name = _source_type_for_label(label)
            grouped[:, label_to_group[group_name]] += distributions[:, source_index]
        return grouped, grouped_labels

    if view_name == "depth_thirds_by_type":
        layer_indices = [
            layer_index
            for label in labels
            if (layer_index := _layer_index_for_label(label)) is not None
        ]
        if not layer_indices:
            raise ValueError("depth_thirds_by_type requires non-embedding sources")
        num_layers = max(layer_indices) + 1
        grouped_labels = (
            "embedding",
            "early_attention",
            "early_mlp",
            "middle_attention",
            "middle_mlp",
            "late_attention",
            "late_mlp",
        )
        label_to_group = {name: index for index, name in enumerate(grouped_labels)}
        grouped = np.zeros((distributions.shape[0], len(grouped_labels)), dtype=float)
        for source_index, label in enumerate(labels):
            source_type = _source_type_for_label(label)
            if source_type == "embedding":
                grouped[:, label_to_group["embedding"]] += distributions[
                    :, source_index
                ]
                continue
            layer_index = _layer_index_for_label(label)
            if layer_index is None:
                raise ValueError("non-embedding grouped view requires layer index")
            band_name = _depth_band_name(layer_index, num_layers=num_layers)
            group_name = f"{band_name}_{source_type}"
            grouped[:, label_to_group[group_name]] += distributions[:, source_index]
        return grouped, grouped_labels

    raise ValueError(f"unsupported view_name: {view_name}")


def _sequence_level_pattern_summary_from_distributions(
    distributions: np.ndarray,
    source_labels: Sequence[str],
    *,
    random_seed: int,
    max_clusters: int,
) -> SequenceLevelPatternSummary:
    validated = _validated_distribution_matrix(distributions)
    labels = _validated_source_labels(
        source_labels,
        expected_num_sources=validated.shape[1],
    )
    entropies = [
        float(-np.sum(row * np.log(np.clip(row, a_min=1e-12, a_max=None))))
        for row in validated
    ]
    top1_indices = np.argmax(validated, axis=1)
    top1_counts = Counter(labels[index] for index in top1_indices.tolist())

    random_generator = np.random.default_rng(random_seed)
    random_control = random_generator.dirichlet(
        np.ones(validated.shape[1], dtype=float),
        size=validated.shape[0],
    )

    return SequenceLevelPatternSummary(
        num_sequences=validated.shape[0],
        num_sources=validated.shape[1],
        mean_entropy=float(sum(entropies) / len(entropies)),
        mean_effective_sources=float(
            sum(math.exp(entropy) for entropy in entropies) / len(entropies)
        ),
        mean_top1_mass=float(validated.max(axis=1).mean()),
        top1_source_counts=dict(sorted(top1_counts.items())),
        source_type_mass=summarize_source_type_mass(validated, labels),
        oracle_cluster_scan=scan_average_linkage_clusters(
            validated,
            max_clusters=max_clusters,
        ),
        random_control_cluster_scan=scan_average_linkage_clusters(
            random_control,
            max_clusters=max_clusters,
        ),
    )


def build_grouped_view_pattern_summary(
    sequence_results: Sequence[Mapping[str, object]],
    *,
    view_name: str,
    random_seed: int,
    max_clusters: int,
) -> GroupedViewPatternSummary:
    if len(sequence_results) < 2:
        raise ValueError("at least two sequence results are required")
    source_labels = _validated_source_labels(
        sequence_results[0]["source_labels"],
        expected_num_sources=len(sequence_results[0]["final_alpha"]),
    )
    alpha_vectors = []
    for result in sequence_results:
        result_labels = _validated_source_labels(
            result["source_labels"],
            expected_num_sources=len(result["final_alpha"]),
        )
        if result_labels != source_labels:
            raise ValueError("all sequence results must share the same source labels")
        alpha_vectors.append(tuple(float(value) for value in result["final_alpha"]))

    grouped_distributions, grouped_labels = _group_distributions_for_view(
        _validated_distribution_matrix(alpha_vectors),
        source_labels,
        view_name=view_name,
    )
    return GroupedViewPatternSummary(
        view_name=view_name,
        grouped_source_labels=grouped_labels,
        summary=_sequence_level_pattern_summary_from_distributions(
            grouped_distributions,
            grouped_labels,
            random_seed=random_seed,
            max_clusters=max_clusters,
        ),
    )


def build_prompt_resampling_stability_summary(
    sequence_results: Sequence[Mapping[str, object]],
    *,
    view_name: str,
    random_seed: int,
    max_clusters: int,
    num_resamples: int,
    sample_size: int,
) -> PromptResamplingStabilitySummary:
    if len(sequence_results) < 2:
        raise ValueError("at least two sequence results are required")
    if num_resamples < 1:
        raise ValueError("num_resamples must be positive")
    if sample_size < 2 or sample_size > len(sequence_results):
        raise ValueError("sample_size must be between 2 and the sequence count")

    grouped_view = build_grouped_view_pattern_summary(
        sequence_results,
        view_name=view_name,
        random_seed=random_seed,
        max_clusters=max_clusters,
    )
    source_labels = grouped_view.grouped_source_labels
    grouped_distributions, _ = _group_distributions_for_view(
        _validated_distribution_matrix(
            [result["final_alpha"] for result in sequence_results]
        ),
        sequence_results[0]["source_labels"],
        view_name=view_name,
    )

    rng = np.random.default_rng(random_seed)
    oracle_best_silhouettes: list[float] = []
    random_best_silhouettes: list[float] = []
    oracle_best_k_counts: Counter[int] = Counter()
    oracle_largest_cluster_fractions: dict[int, list[float]] = {}
    random_largest_cluster_fractions: dict[int, list[float]] = {}

    for resample_index in range(num_resamples):
        selection = rng.choice(
            grouped_distributions.shape[0],
            size=sample_size,
            replace=False,
        )
        sample_distributions = grouped_distributions[selection]
        sample_summary = _sequence_level_pattern_summary_from_distributions(
            sample_distributions,
            source_labels,
            random_seed=random_seed + resample_index + 1,
            max_clusters=max_clusters,
        )
        oracle_scan = sample_summary.oracle_cluster_scan
        random_scan = sample_summary.random_control_cluster_scan
        oracle_best_silhouettes.append(oracle_scan.best_silhouette)
        random_best_silhouettes.append(random_scan.best_silhouette)
        oracle_best_k_counts.update([oracle_scan.best_k])
        for (
            cluster_count,
            fraction,
        ) in oracle_scan.largest_cluster_fraction_by_k.items():
            oracle_largest_cluster_fractions.setdefault(cluster_count, []).append(
                fraction
            )
        for (
            cluster_count,
            fraction,
        ) in random_scan.largest_cluster_fraction_by_k.items():
            random_largest_cluster_fractions.setdefault(cluster_count, []).append(
                fraction
            )

    return PromptResamplingStabilitySummary(
        view_name=view_name,
        num_resamples=num_resamples,
        sample_size=sample_size,
        oracle_best_silhouette_mean=float(np.mean(oracle_best_silhouettes)),
        oracle_best_silhouette_std=float(np.std(oracle_best_silhouettes)),
        random_best_silhouette_mean=float(np.mean(random_best_silhouettes)),
        random_best_silhouette_std=float(np.std(random_best_silhouettes)),
        oracle_beats_random_fraction=float(
            np.mean(
                [
                    oracle_value > random_value
                    for oracle_value, random_value in zip(
                        oracle_best_silhouettes,
                        random_best_silhouettes,
                        strict=True,
                    )
                ]
            )
        ),
        oracle_best_k_counts=dict(sorted(oracle_best_k_counts.items())),
        oracle_largest_cluster_fraction_mean_by_k={
            cluster_count: float(np.mean(values))
            for cluster_count, values in sorted(
                oracle_largest_cluster_fractions.items()
            )
        },
        random_largest_cluster_fraction_mean_by_k={
            cluster_count: float(np.mean(values))
            for cluster_count, values in sorted(
                random_largest_cluster_fractions.items()
            )
        },
    )


def build_sequence_level_pattern_summary(
    sequence_results: Sequence[Mapping[str, object]],
    *,
    random_seed: int,
    max_clusters: int,
) -> SequenceLevelPatternSummary:
    if len(sequence_results) < 2:
        raise ValueError("at least two sequence results are required")

    source_labels = _validated_source_labels(
        sequence_results[0]["source_labels"],
        expected_num_sources=len(sequence_results[0]["final_alpha"]),
    )
    alpha_vectors = []
    for result in sequence_results:
        result_labels = _validated_source_labels(
            result["source_labels"],
            expected_num_sources=len(result["final_alpha"]),
        )
        if result_labels != source_labels:
            raise ValueError("all sequence results must share the same source labels")
        alpha_vectors.append(tuple(float(value) for value in result["final_alpha"]))

    return _sequence_level_pattern_summary_from_distributions(
        _validated_distribution_matrix(alpha_vectors),
        source_labels,
        random_seed=random_seed,
        max_clusters=max_clusters,
    )


def _build_pattern_payload_from_sequence_results(
    sequence_results: Sequence[Mapping[str, object]],
    *,
    random_seed: int,
    max_clusters: int,
    num_resamples: int,
    sample_size: int | None,
    cluster_view_names: Sequence[str],
) -> dict[str, object]:
    if len(sequence_results) < 2:
        raise ValueError("at least two sequence results are required")

    resolved_sample_size = sample_size
    if resolved_sample_size is None:
        resolved_sample_size = max(2, int(round(len(sequence_results) * 0.75)))
    resolved_sample_size = min(resolved_sample_size, len(sequence_results))

    improvements = [
        float(result["uniform_loss"]) - float(result["optimized_loss"])
        for result in sequence_results
    ]
    null_model_names = sorted(
        {
            null_name
            for result in sequence_results
            for null_name in result["null_losses"].keys()
        }
    )
    null_model_mean_losses = {
        null_name: float(
            np.mean(
                [float(result["null_losses"][null_name]) for result in sequence_results]
            )
        )
        for null_name in null_model_names
    }

    pattern_summary = build_sequence_level_pattern_summary(
        sequence_results,
        random_seed=random_seed,
        max_clusters=max_clusters,
    )
    cluster_views = [
        asdict(
            build_grouped_view_pattern_summary(
                sequence_results,
                view_name=view_name,
                random_seed=random_seed,
                max_clusters=max_clusters,
            )
        )
        for view_name in cluster_view_names
    ]
    resampling_stability = [
        asdict(
            build_prompt_resampling_stability_summary(
                sequence_results,
                view_name=view_name,
                random_seed=random_seed,
                max_clusters=max_clusters,
                num_resamples=num_resamples,
                sample_size=resolved_sample_size,
            )
        )
        for view_name in cluster_view_names
    ]

    return {
        "num_sequences": len(sequence_results),
        "sequence_mean_improvement": float(np.mean(improvements)),
        "bootstrap_interval": asdict(
            bootstrap_mean_confidence_interval(
                improvements,
                num_resamples=1000,
                confidence_level=0.95,
                seed=random_seed,
            )
        ),
        "null_model_mean_losses": null_model_mean_losses,
        "summary": asdict(pattern_summary),
        "cluster_views": cluster_views,
        "resampling_stability": resampling_stability,
    }


def write_pattern_analysis_summary(
    *,
    run_path: Path,
    output_path: Path,
    random_seed: int,
    max_clusters: int,
    num_resamples: int = 64,
    sample_size: int | None = None,
    subset_prompt_ids_by_name: Mapping[str, Sequence[str]] | None = None,
    cluster_view_names: Sequence[str] = (
        "raw_source",
        "source_type",
        "depth_thirds_by_type",
    ),
) -> None:
    run_payload = json.loads(run_path.read_text())
    if "sequence_results" not in run_payload:
        raise ValueError("run payload must include sequence_results")
    if sample_size is None:
        sample_size = max(2, int(round(len(run_payload["sequence_results"]) * 0.75)))

    pattern_summary = build_sequence_level_pattern_summary(
        run_payload["sequence_results"],
        random_seed=random_seed,
        max_clusters=max_clusters,
    )
    cluster_views = [
        asdict(
            build_grouped_view_pattern_summary(
                run_payload["sequence_results"],
                view_name=view_name,
                random_seed=random_seed,
                max_clusters=max_clusters,
            )
        )
        for view_name in cluster_view_names
    ]
    resampling_stability = [
        asdict(
            build_prompt_resampling_stability_summary(
                run_payload["sequence_results"],
                view_name=view_name,
                random_seed=random_seed,
                max_clusters=max_clusters,
                num_resamples=num_resamples,
                sample_size=sample_size,
            )
        )
        for view_name in cluster_view_names
    ]

    payload = {
        "model_name": run_payload["model_name"],
        "collection_id": run_payload["collection_id"],
        "split": run_payload["split"],
        "num_sequences": run_payload["num_sequences"],
        "sequence_mean_improvement": run_payload["sequence_mean_improvement"],
        "bootstrap_interval": run_payload["bootstrap_interval"],
        "null_model_mean_losses": run_payload["null_model_mean_losses"],
        **asdict(pattern_summary),
        "cluster_views": cluster_views,
        "resampling_stability": resampling_stability,
    }
    if subset_prompt_ids_by_name:
        prompt_id_to_result = {
            result["prompt_id"]: result for result in run_payload["sequence_results"]
        }
        payload["subset_summaries"] = {
            subset_name: _build_pattern_payload_from_sequence_results(
                [
                    prompt_id_to_result[prompt_id]
                    for prompt_id in prompt_ids
                    if prompt_id in prompt_id_to_result
                ],
                random_seed=random_seed,
                max_clusters=max_clusters,
                num_resamples=num_resamples,
                sample_size=sample_size,
                cluster_view_names=cluster_view_names,
            )
            for subset_name, prompt_ids in subset_prompt_ids_by_name.items()
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
