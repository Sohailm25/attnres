# ABOUTME: Analyzes sequence-level oracle-alpha routing patterns from saved run artifacts.
# ABOUTME: Keeps prereg-scale clustering and source-type summaries reusable and aligned with repo guardrails.

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .oracle_alpha_controls import jensen_shannon_divergence


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
    distributions = _validated_distribution_matrix(alpha_vectors)
    labels = _validated_source_labels(
        source_labels,
        expected_num_sources=distributions.shape[1],
    )

    embedding_indices = [
        index for index, label in enumerate(labels) if label in {"embed", "pos_embed"}
    ]
    attention_indices = [
        index for index, label in enumerate(labels) if label.endswith("_attn_out")
    ]
    mlp_indices = [
        index for index, label in enumerate(labels) if label.endswith("_mlp_out")
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
    best_k = 2
    best_silhouette = float("-inf")

    for cluster_count in range(2, upper_k + 1):
        if cluster_count == distribution_matrix.shape[0]:
            silhouette = 0.0
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
        if silhouette > best_silhouette:
            best_k = cluster_count
            best_silhouette = float(silhouette)

    return ClusterScanResult(
        linkage_method="average",
        best_k=best_k,
        best_silhouette=best_silhouette,
        silhouette_by_k=silhouette_by_k,
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

    distributions = _validated_distribution_matrix(alpha_vectors)
    entropies = [
        float(-np.sum(row * np.log(np.clip(row, a_min=1e-12, a_max=None))))
        for row in distributions
    ]
    top1_indices = np.argmax(distributions, axis=1)
    top1_counts = Counter(source_labels[index] for index in top1_indices.tolist())

    random_generator = np.random.default_rng(random_seed)
    random_control = random_generator.dirichlet(
        np.ones(distributions.shape[1], dtype=float),
        size=distributions.shape[0],
    )

    return SequenceLevelPatternSummary(
        num_sequences=distributions.shape[0],
        num_sources=distributions.shape[1],
        mean_entropy=float(sum(entropies) / len(entropies)),
        mean_effective_sources=float(
            sum(math.exp(entropy) for entropy in entropies) / len(entropies)
        ),
        mean_top1_mass=float(distributions.max(axis=1).mean()),
        top1_source_counts=dict(sorted(top1_counts.items())),
        source_type_mass=summarize_source_type_mass(distributions, source_labels),
        oracle_cluster_scan=scan_average_linkage_clusters(
            distributions,
            max_clusters=max_clusters,
        ),
        random_control_cluster_scan=scan_average_linkage_clusters(
            random_control,
            max_clusters=max_clusters,
        ),
    )


def write_pattern_analysis_summary(
    *,
    run_path: Path,
    output_path: Path,
    random_seed: int,
    max_clusters: int,
) -> None:
    run_payload = json.loads(run_path.read_text())
    if "sequence_results" not in run_payload:
        raise ValueError("run payload must include sequence_results")

    pattern_summary = build_sequence_level_pattern_summary(
        run_payload["sequence_results"],
        random_seed=random_seed,
        max_clusters=max_clusters,
    )

    payload = {
        "model_name": run_payload["model_name"],
        "collection_id": run_payload["collection_id"],
        "split": run_payload["split"],
        "num_sequences": run_payload["num_sequences"],
        "sequence_mean_improvement": run_payload["sequence_mean_improvement"],
        "bootstrap_interval": run_payload["bootstrap_interval"],
        "null_model_mean_losses": run_payload["null_model_mean_losses"],
        **asdict(pattern_summary),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
