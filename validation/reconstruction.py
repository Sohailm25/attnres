# ABOUTME: Implements normalization-aware reconstruction helpers for the first Phase 1 sanity suite.
# ABOUTME: Uses the shared final normalization factor so exact routed-logit decompositions stay honest.

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


@dataclass(frozen=True)
class CacheReconstructionMetrics:
    num_sources: int
    max_abs_error: float
    l2_error: float


def rms_norm(vector: Vector, eps: float) -> list[float]:
    mean_square = sum(value * value for value in vector) / len(vector)
    scale = math.sqrt(mean_square + eps)
    return [value / scale for value in vector]


def weighted_sum(sources: Sequence[Vector], alpha: Sequence[float]) -> list[float]:
    if not sources:
        raise ValueError("sources must not be empty")
    if len(sources) != len(alpha):
        raise ValueError("sources and alpha must have the same length")

    dimension = len(sources[0])
    combined = [0.0] * dimension
    for source, weight in zip(sources, alpha, strict=True):
        if len(source) != dimension:
            raise ValueError("all sources must share the same dimension")
        for index, value in enumerate(source):
            combined[index] += weight * value
    return combined


def matvec(matrix: Matrix, vector: Vector) -> list[float]:
    return [
        sum(weight * value for weight, value in zip(row, vector, strict=True))
        for row in matrix
    ]


def max_abs_difference(left: Vector, right: Vector) -> float:
    return max(abs(lhs - rhs) for lhs, rhs in zip(left, right, strict=True))


def l2_distance(left: Vector, right: Vector) -> float:
    return math.sqrt(
        sum((lhs - rhs) * (lhs - rhs) for lhs, rhs in zip(left, right, strict=True))
    )


def cache_reconstruction_metrics(
    *,
    embedding: Vector,
    sublayer_outputs: Sequence[Vector],
    final_hidden: Vector,
) -> CacheReconstructionMetrics:
    sources = [embedding, *sublayer_outputs]
    reconstructed = weighted_sum(sources, [1.0] * len(sources))
    return CacheReconstructionMetrics(
        num_sources=len(sources),
        max_abs_error=max_abs_difference(reconstructed, final_hidden),
        l2_error=l2_distance(reconstructed, final_hidden),
    )


def routed_logits(
    *,
    sources: Sequence[Vector],
    alpha: Sequence[float],
    unembed: Matrix,
    eps: float,
) -> list[float]:
    mixture = weighted_sum(sources, alpha)
    normalized = rms_norm(mixture, eps=eps)
    return matvec(unembed, normalized)


def shared_final_norm_logit_contributions(
    *,
    sources: Sequence[Vector],
    alpha: Sequence[float],
    unembed: Matrix,
    eps: float,
) -> list[list[float]]:
    mixture = weighted_sum(sources, alpha)
    mean_square = sum(value * value for value in mixture) / len(mixture)
    shared_scale = math.sqrt(mean_square + eps)

    contributions: list[list[float]] = []
    for source, weight in zip(sources, alpha, strict=True):
        scaled_source = [(weight * value) / shared_scale for value in source]
        contributions.append(matvec(unembed, scaled_source))
    return contributions
