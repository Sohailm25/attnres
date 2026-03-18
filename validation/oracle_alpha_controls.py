# ABOUTME: Encodes the Phase 1 claim-bearing control plan and its reusable statistics.
# ABOUTME: Keeps stability, predictiveness, and MIB handling explicit before oracle-alpha runs exist.

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
import yaml

from prompts import load_prompt_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROL_REGISTRY_PATH = ROOT / "configs" / "oracle_alpha_controls_v1.yaml"
ALLOWED_SPLITS = {"pilot", "confirm"}
ALLOWED_MIB_STATUSES = {"planned", "omitted"}
ALLOWED_PREDICTIVENESS_METRICS = {
    "r_squared",
    "mean_js_divergence",
    "mean_predicted_improvement_over_uniform",
}
ALLOWED_PREDICTIVENESS_TARGETS = {
    "oracle_alpha_vector",
    "oracle_alpha_logit_vector",
    "oracle_alpha_depth_type_band_logit_vector",
}
ALPHA_TARGET_EPSILON = 1e-12
DEPTH_BAND_NAMES = ("early", "mid", "late")


@dataclass(frozen=True)
class BootstrapMeanInterval:
    mean: float
    lower: float
    upper: float
    confidence_level: float
    num_resamples: int


@dataclass(frozen=True)
class StabilitySuitePlan:
    restart_seeds: tuple[int, ...]
    prompt_perturbations: tuple[str, ...]
    report_metrics: tuple[str, ...]


@dataclass(frozen=True)
class PredictivenessPlan:
    train_collection_id: str
    train_split: str
    eval_collection_id: str
    eval_split: str
    model_family: str
    target: str
    features: str
    primary_metric: str
    secondary_metric: str
    failure_action: str


@dataclass(frozen=True)
class MIBPlan:
    mib_status: str
    benchmark: str | None
    sanity_task: str | None
    rationale: str
    revisit_trigger: str


@dataclass(frozen=True)
class OracleAlphaControlPlan:
    plan_id: str
    collection_id: str
    aggregation_unit: str
    bootstrap_resamples: int
    null_models: tuple[str, ...]
    stability_suite: StabilitySuitePlan
    predictiveness: PredictivenessPlan
    mib_anchor: MIBPlan


@dataclass(frozen=True)
class OracleAlphaControlRegistry:
    version: int
    registry_id: str
    plans: dict[str, OracleAlphaControlPlan]


@dataclass(frozen=True)
class PredictivenessSummary:
    r_squared: float
    mean_js_divergence: float
    num_train_examples: int
    num_eval_examples: int


def _validated_distribution_matrix(
    distributions: Sequence[Sequence[float]],
) -> np.ndarray:
    matrix = np.asarray(distributions, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("distributions must be rank-2")
    if matrix.shape[1] < 1:
        raise ValueError("distributions must contain at least one source")
    if np.any(matrix < 0.0):
        raise ValueError("distribution values must be non-negative")
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("each distribution must sum to a positive value")
    return matrix / row_sums


def _validated_source_labels(
    source_labels: Sequence[str] | None,
    *,
    expected_num_sources: int,
) -> tuple[str, ...]:
    if source_labels is None:
        raise ValueError("source_labels are required for this predictiveness target")
    labels = tuple(source_labels)
    if len(labels) != expected_num_sources:
        raise ValueError("source_labels must match the target source dimension")
    if not labels:
        raise ValueError("source_labels must not be empty")
    return labels


def _depth_type_band_group_structure(
    source_labels: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray]:
    parsed_layers = []
    parsed_kinds = []
    for label in source_labels:
        if label in {"embed", "pos_embed"}:
            parsed_layers.append(None)
            parsed_kinds.append(label)
            continue
        layer_token, separator, suffix = label.partition("_")
        if not separator or not layer_token.isdigit():
            raise ValueError(f"unsupported source label {label!r}")
        if suffix not in {"attn_out", "mlp_out"}:
            raise ValueError(f"unsupported source label {label!r}")
        parsed_layers.append(int(layer_token))
        parsed_kinds.append(suffix)

    unique_layers = sorted({layer for layer in parsed_layers if layer is not None})
    band_lookup: dict[int, str] = {}
    for band_name, layer_chunk in zip(
        DEPTH_BAND_NAMES,
        np.array_split(np.asarray(unique_layers, dtype=int), len(DEPTH_BAND_NAMES)),
        strict=True,
    ):
        for layer in layer_chunk.tolist():
            band_lookup[int(layer)] = band_name

    ordered_group_names: list[str] = []
    group_indices = []
    group_name_to_index: dict[str, int] = {}
    for label, layer, kind in zip(
        source_labels,
        parsed_layers,
        parsed_kinds,
        strict=True,
    ):
        if label in {"embed", "pos_embed"}:
            group_name = label
        else:
            band_name = band_lookup[layer]
            source_kind = "attn" if kind == "attn_out" else "mlp"
            group_name = f"{band_name}_{source_kind}"
        if group_name not in group_name_to_index:
            group_name_to_index[group_name] = len(ordered_group_names)
            ordered_group_names.append(group_name)
        group_indices.append(group_name_to_index[group_name])

    return tuple(ordered_group_names), np.asarray(group_indices, dtype=int)


def _compressed_distribution_matrix(
    *,
    distributions: np.ndarray,
    group_indices: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    compressed = np.zeros((distributions.shape[0], num_groups), dtype=float)
    for source_index, group_index in enumerate(group_indices.tolist()):
        compressed[:, group_index] += distributions[:, source_index]
    return compressed


def _group_template_matrix(
    *,
    train_distributions: np.ndarray,
    group_indices: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    templates = np.zeros((num_groups, train_distributions.shape[1]), dtype=float)
    for group_index in range(num_groups):
        mask = group_indices == group_index
        group_width = int(mask.sum())
        if group_width < 1:
            raise ValueError("each compressed target group must contain a source")
        group_slice = train_distributions[:, mask]
        group_mass = group_slice.sum(axis=1, keepdims=True)
        normalized = np.zeros_like(group_slice)
        nonzero_rows = group_mass[:, 0] > 0.0
        if np.any(nonzero_rows):
            normalized[nonzero_rows] = (
                group_slice[nonzero_rows] / group_mass[nonzero_rows]
            )
        if np.any(~nonzero_rows):
            normalized[~nonzero_rows] = 1.0 / group_width
        template = normalized.mean(axis=0)
        template /= template.sum()
        templates[group_index, mask] = template
    return templates


def alpha_target_matrix(
    *,
    target_name: str,
    distributions: Sequence[Sequence[float]],
    source_labels: Sequence[str] | None = None,
) -> np.ndarray:
    matrix = _validated_distribution_matrix(distributions)
    if target_name == "oracle_alpha_vector":
        return matrix
    if target_name == "oracle_alpha_logit_vector":
        stabilized = np.clip(matrix, a_min=ALPHA_TARGET_EPSILON, a_max=None)
        logits = np.log(stabilized)
        return logits - logits.mean(axis=1, keepdims=True)
    if target_name == "oracle_alpha_depth_type_band_logit_vector":
        labels = _validated_source_labels(
            source_labels,
            expected_num_sources=matrix.shape[1],
        )
        group_names, group_indices = _depth_type_band_group_structure(labels)
        compressed = _compressed_distribution_matrix(
            distributions=matrix,
            group_indices=group_indices,
            num_groups=len(group_names),
        )
        stabilized = np.clip(compressed, a_min=ALPHA_TARGET_EPSILON, a_max=None)
        logits = np.log(stabilized)
        return logits - logits.mean(axis=1, keepdims=True)
    raise ValueError(f"unsupported predictiveness target {target_name!r}")


def alpha_target_predictions_to_distributions(
    *,
    target_name: str,
    predictions: Sequence[Sequence[float]],
    source_labels: Sequence[str] | None = None,
    train_distributions: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    matrix = np.asarray(predictions, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("predictions must be rank-2")
    if matrix.shape[1] < 1:
        raise ValueError("predictions must contain at least one source")

    if target_name == "oracle_alpha_vector":
        clipped = np.clip(matrix, a_min=0.0, a_max=None)
        row_sums = clipped.sum(axis=1, keepdims=True)
        zero_rows = row_sums.squeeze(axis=1) <= 0.0
        if np.any(zero_rows):
            clipped[zero_rows] = 1.0 / clipped.shape[1]
            row_sums = clipped.sum(axis=1, keepdims=True)
        return clipped / row_sums
    if target_name == "oracle_alpha_logit_vector":
        shifted = matrix - matrix.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)
    if target_name == "oracle_alpha_depth_type_band_logit_vector":
        if train_distributions is None:
            raise ValueError(
                "train_distributions are required to lift compressed target predictions"
            )
        labels = _validated_source_labels(
            source_labels,
            expected_num_sources=len(source_labels) if source_labels is not None else 0,
        )
        train_matrix = _validated_distribution_matrix(train_distributions)
        if train_matrix.shape[1] != len(labels):
            raise ValueError(
                "train_distributions must share the same source dimension as source_labels"
            )
        group_names, group_indices = _depth_type_band_group_structure(labels)
        if matrix.shape[1] != len(group_names):
            raise ValueError(
                "predictions must match the compressed target dimension for source_labels"
            )
        shifted = matrix - matrix.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        grouped = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        templates = _group_template_matrix(
            train_distributions=train_matrix,
            group_indices=group_indices,
            num_groups=len(group_names),
        )
        lifted = grouped @ templates
        return lifted / lifted.sum(axis=1, keepdims=True)
    raise ValueError(f"unsupported predictiveness target {target_name!r}")


def predictiveness_metric_value(
    *,
    summary: PredictivenessSummary,
    metric_name: str,
    metric_overrides: Mapping[str, float] | None = None,
) -> float:
    if metric_name not in ALLOWED_PREDICTIVENESS_METRICS:
        raise ValueError(f"unsupported predictiveness metric {metric_name!r}")
    if metric_overrides is not None and metric_name in metric_overrides:
        return float(metric_overrides[metric_name])
    if metric_name == "mean_predicted_improvement_over_uniform":
        raise ValueError(
            "mean_predicted_improvement_over_uniform requires a metric override"
        )
    return float(getattr(summary, metric_name))


def compare_predictiveness_metric_values(
    *,
    metric_name: str,
    left: float,
    right: float,
) -> float:
    if metric_name == "r_squared":
        return left - right
    if metric_name == "mean_js_divergence":
        return right - left
    if metric_name == "mean_predicted_improvement_over_uniform":
        return left - right
    raise ValueError(f"unsupported predictiveness metric {metric_name!r}")


def _validated_predictiveness_arrays(
    *,
    train_features: Sequence[Sequence[float]],
    train_targets: Sequence[Sequence[float]],
    eval_features: Sequence[Sequence[float]],
    eval_targets: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.asarray(train_features, dtype=float)
    train_y = np.asarray(train_targets, dtype=float)
    eval_x = np.asarray(eval_features, dtype=float)
    eval_y = np.asarray(eval_targets, dtype=float)

    if train_x.ndim != 2 or eval_x.ndim != 2:
        raise ValueError("feature arrays must be rank-2")
    if train_y.ndim != 2 or eval_y.ndim != 2:
        raise ValueError("target arrays must be rank-2")
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("train feature and target counts must match")
    if eval_x.shape[0] != eval_y.shape[0]:
        raise ValueError("eval feature and target counts must match")
    if train_y.shape[1] != eval_y.shape[1]:
        raise ValueError("target dimensions must match")
    if train_x.shape[1] != eval_x.shape[1]:
        raise ValueError("feature dimensions must match")

    return train_x, train_y, eval_x, eval_y


def _predictiveness_summary_from_predictions(
    *,
    train_y: np.ndarray,
    eval_y: np.ndarray,
    predictions: np.ndarray,
) -> PredictivenessSummary:
    residual_sum = float(np.square(eval_y - predictions).sum())
    centered = eval_y - eval_y.mean(axis=0, keepdims=True)
    total_sum = float(np.square(centered).sum())
    if total_sum == 0.0:
        r_squared = 1.0 if residual_sum == 0.0 else 0.0
    else:
        r_squared = 1.0 - (residual_sum / total_sum)

    predicted_distributions = np.clip(predictions, a_min=0.0, a_max=None)
    actual_distributions = np.clip(eval_y, a_min=0.0, a_max=None)
    mean_js = _mean(
        [
            jensen_shannon_divergence(predicted_row, actual_row)
            for predicted_row, actual_row in zip(
                predicted_distributions.tolist(),
                actual_distributions.tolist(),
                strict=True,
            )
        ]
    )

    return PredictivenessSummary(
        r_squared=r_squared,
        mean_js_divergence=mean_js,
        num_train_examples=train_y.shape[0],
        num_eval_examples=eval_y.shape[0],
    )


def predictiveness_summary_from_predictions(
    *,
    train_targets: Sequence[Sequence[float]],
    eval_targets: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
) -> PredictivenessSummary:
    train_y = np.asarray(train_targets, dtype=float)
    eval_y = np.asarray(eval_targets, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    if train_y.ndim != 2:
        raise ValueError("train_targets must be rank-2")
    if eval_y.ndim != 2 or predicted.ndim != 2:
        raise ValueError("eval_targets and predictions must be rank-2")
    if eval_y.shape != predicted.shape:
        raise ValueError("eval_targets and predictions must share the same shape")
    return _predictiveness_summary_from_predictions(
        train_y=train_y,
        eval_y=eval_y,
        predictions=predicted,
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def bootstrap_mean_confidence_interval(
    values: Sequence[float],
    *,
    num_resamples: int,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapMeanInterval:
    if not values:
        raise ValueError("values must not be empty")
    if num_resamples < 1:
        raise ValueError("num_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = random.Random(seed)
    sample_means = []
    for _ in range(num_resamples):
        resample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        sample_means.append(_mean(resample))
    sample_means.sort()

    alpha = (1.0 - confidence_level) / 2.0
    lower_index = max(0, math.floor(alpha * (num_resamples - 1)))
    upper_index = min(num_resamples - 1, math.ceil((1.0 - alpha) * (num_resamples - 1)))
    return BootstrapMeanInterval(
        mean=_mean(values),
        lower=sample_means[lower_index],
        upper=sample_means[upper_index],
        confidence_level=confidence_level,
        num_resamples=num_resamples,
    )


def _normalize_distribution(distribution: Sequence[float]) -> list[float]:
    if not distribution:
        raise ValueError("distribution must not be empty")
    if any(value < 0.0 for value in distribution):
        raise ValueError("distribution values must be non-negative")
    total = sum(distribution)
    if total <= 0.0:
        raise ValueError("distribution must sum to a positive value")
    return [value / total for value in distribution]


def jensen_shannon_divergence(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("left and right must have the same length")

    normalized_left = _normalize_distribution(left)
    normalized_right = _normalize_distribution(right)
    midpoint = [
        (lhs + rhs) / 2.0
        for lhs, rhs in zip(normalized_left, normalized_right, strict=True)
    ]

    def kl_divergence(source: Sequence[float], target: Sequence[float]) -> float:
        value = 0.0
        for src, dst in zip(source, target, strict=True):
            if src == 0.0:
                continue
            value += src * math.log2(src / dst)
        return value

    return 0.5 * (
        kl_divergence(normalized_left, midpoint)
        + kl_divergence(normalized_right, midpoint)
    )


def mean_pairwise_js_divergence(
    distributions: Sequence[Sequence[float]],
) -> float:
    if len(distributions) < 2:
        raise ValueError("at least two distributions are required")
    distances = [
        jensen_shannon_divergence(left, right)
        for left, right in itertools.combinations(distributions, 2)
    ]
    return _mean(distances)


def mean_topk_jaccard_similarity(
    distributions: Sequence[Sequence[float]],
    *,
    k: int,
) -> float:
    if len(distributions) < 2:
        raise ValueError("at least two distributions are required")
    if k < 1:
        raise ValueError("k must be positive")

    def topk_indices(distribution: Sequence[float]) -> set[int]:
        if not distribution:
            raise ValueError("distribution must not be empty")
        count = min(k, len(distribution))
        ranked = sorted(
            range(len(distribution)),
            key=lambda index: (distribution[index], -index),
            reverse=True,
        )
        return set(ranked[:count])

    similarities = []
    for left, right in itertools.combinations(distributions, 2):
        left_indices = topk_indices(left)
        right_indices = topk_indices(right)
        intersection = len(left_indices & right_indices)
        union = len(left_indices | right_indices)
        similarities.append(intersection / union)
    return _mean(similarities)


def mean_top1_source_agreement(
    distributions: Sequence[Sequence[float]],
) -> float:
    if len(distributions) < 2:
        raise ValueError("at least two distributions are required")

    def top1_index(distribution: Sequence[float]) -> int:
        if not distribution:
            raise ValueError("distribution must not be empty")
        return max(range(len(distribution)), key=lambda index: distribution[index])

    agreements = []
    for left, right in itertools.combinations(distributions, 2):
        agreements.append(1.0 if top1_index(left) == top1_index(right) else 0.0)
    return _mean(agreements)


def linear_alpha_predictiveness_summary(
    *,
    train_features: Sequence[Sequence[float]],
    train_targets: Sequence[Sequence[float]],
    eval_features: Sequence[Sequence[float]],
    eval_targets: Sequence[Sequence[float]],
) -> PredictivenessSummary:
    train_x, train_y, eval_x, eval_y = _validated_predictiveness_arrays(
        train_features=train_features,
        train_targets=train_targets,
        eval_features=eval_features,
        eval_targets=eval_targets,
    )

    train_design = np.concatenate(
        [np.ones((train_x.shape[0], 1)), train_x],
        axis=1,
    )
    eval_design = np.concatenate(
        [np.ones((eval_x.shape[0], 1)), eval_x],
        axis=1,
    )
    coefficients, *_ = np.linalg.lstsq(train_design, train_y, rcond=None)
    predictions = eval_design @ coefficients
    return _predictiveness_summary_from_predictions(
        train_y=train_y,
        eval_y=eval_y,
        predictions=predictions,
    )


def _ridge_regression_solver_mode(*, num_examples: int, num_features: int) -> str:
    if num_examples < 1:
        raise ValueError("num_examples must be positive")
    if num_features < 1:
        raise ValueError("num_features must be positive")
    return "dual" if num_examples < num_features else "primal"


def ridge_regression_predictions(
    *,
    train_features: Sequence[Sequence[float]],
    train_targets: Sequence[Sequence[float]],
    eval_features: Sequence[Sequence[float]],
    regularization_strength: float,
) -> np.ndarray:
    if regularization_strength < 0.0:
        raise ValueError("regularization_strength must be non-negative")

    train_x = np.asarray(train_features, dtype=float)
    train_y = np.asarray(train_targets, dtype=float)
    eval_x = np.asarray(eval_features, dtype=float)

    if train_x.ndim != 2 or eval_x.ndim != 2:
        raise ValueError("feature arrays must be rank-2")
    if train_y.ndim != 2:
        raise ValueError("target arrays must be rank-2")
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("train feature and target counts must match")
    if train_x.shape[1] != eval_x.shape[1]:
        raise ValueError("feature dimensions must match")

    feature_mean = train_x.mean(axis=0, keepdims=True)
    feature_scale = train_x.std(axis=0, keepdims=True)
    feature_scale[feature_scale == 0.0] = 1.0

    train_standardized = (train_x - feature_mean) / feature_scale
    eval_standardized = (eval_x - feature_mean) / feature_scale
    target_mean = train_y.mean(axis=0, keepdims=True)
    centered_train_y = train_y - target_mean

    solver_mode = _ridge_regression_solver_mode(
        num_examples=train_standardized.shape[0],
        num_features=train_standardized.shape[1],
    )
    if solver_mode == "dual":
        gram_matrix = train_standardized @ train_standardized.T
        dual_coefficients = np.linalg.solve(
            gram_matrix
            + (
                regularization_strength
                * np.eye(train_standardized.shape[0], dtype=float)
            ),
            centered_train_y,
        )
        return target_mean + (
            eval_standardized @ train_standardized.T @ dual_coefficients
        )

    feature_coefficients = np.linalg.solve(
        train_standardized.T @ train_standardized
        + (regularization_strength * np.eye(train_standardized.shape[1], dtype=float)),
        train_standardized.T @ centered_train_y,
    )
    return target_mean + (eval_standardized @ feature_coefficients)


def ridge_alpha_predictiveness_summary(
    *,
    train_features: Sequence[Sequence[float]],
    train_targets: Sequence[Sequence[float]],
    eval_features: Sequence[Sequence[float]],
    eval_targets: Sequence[Sequence[float]],
    regularization_strength: float,
) -> PredictivenessSummary:
    train_x, train_y, eval_x, eval_y = _validated_predictiveness_arrays(
        train_features=train_features,
        train_targets=train_targets,
        eval_features=eval_features,
        eval_targets=eval_targets,
    )
    predictions = ridge_regression_predictions(
        train_features=train_x.tolist(),
        train_targets=train_y.tolist(),
        eval_features=eval_x.tolist(),
        regularization_strength=regularization_strength,
    )
    return _predictiveness_summary_from_predictions(
        train_y=train_y,
        eval_y=eval_y,
        predictions=predictions,
    )


def _collection_has_split(collection, split: str) -> bool:
    return any(entry.split == split for entry in collection.prompt_entries)


def _validate_predictiveness_plan(
    plan: PredictivenessPlan,
    *,
    prompt_registry,
) -> None:
    if plan.train_split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported train split {plan.train_split!r}")
    if plan.eval_split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported eval split {plan.eval_split!r}")
    if plan.train_collection_id not in prompt_registry.collections:
        raise ValueError(f"unknown train collection {plan.train_collection_id!r}")
    if plan.eval_collection_id not in prompt_registry.collections:
        raise ValueError(f"unknown eval collection {plan.eval_collection_id!r}")
    if plan.target not in ALLOWED_PREDICTIVENESS_TARGETS:
        raise ValueError(f"unsupported predictiveness target {plan.target!r}")
    if plan.primary_metric not in ALLOWED_PREDICTIVENESS_METRICS:
        raise ValueError(f"unsupported primary metric {plan.primary_metric!r}")
    if plan.secondary_metric not in ALLOWED_PREDICTIVENESS_METRICS:
        raise ValueError(f"unsupported secondary metric {plan.secondary_metric!r}")

    train_collection = prompt_registry.collections[plan.train_collection_id]
    eval_collection = prompt_registry.collections[plan.eval_collection_id]
    if not _collection_has_split(train_collection, plan.train_split):
        raise ValueError("train split must exist in the prompt registry")
    if not _collection_has_split(eval_collection, plan.eval_split):
        raise ValueError("eval split must exist in the prompt registry")


def _validate_mib_plan(plan: MIBPlan) -> None:
    if plan.mib_status not in ALLOWED_MIB_STATUSES:
        raise ValueError(f"unsupported MIB status {plan.mib_status!r}")
    if not plan.revisit_trigger:
        raise ValueError("MIB plan must define a revisit trigger")

    if plan.mib_status == "planned":
        if not plan.benchmark:
            raise ValueError("planned MIB usage must name a benchmark")
        if not plan.sanity_task:
            raise ValueError("planned MIB usage must define a sanity task")
        if not plan.rationale:
            raise ValueError("planned MIB usage must include a rationale")
        return

    if not plan.rationale:
        raise ValueError("omitted MIB usage must include a rationale")


def _validate_control_plan(
    plan: OracleAlphaControlPlan,
    *,
    prompt_registry,
) -> None:
    if plan.collection_id not in prompt_registry.collections:
        raise ValueError(f"unknown collection {plan.collection_id!r}")
    if plan.aggregation_unit != "sequence":
        raise ValueError("claim-bearing oracle-alpha plans must aggregate by sequence")
    if plan.bootstrap_resamples < 1000:
        raise ValueError("claim-bearing bootstrap must use at least 1000 resamples")
    if not plan.null_models:
        raise ValueError("control plans must define null models")
    if not plan.stability_suite.restart_seeds:
        raise ValueError("stability suite must define restart seeds")
    if not plan.stability_suite.prompt_perturbations:
        raise ValueError("stability suite must define prompt perturbations")
    if not plan.stability_suite.report_metrics:
        raise ValueError("stability suite must define report metrics")

    _validate_predictiveness_plan(plan.predictiveness, prompt_registry=prompt_registry)
    _validate_mib_plan(plan.mib_anchor)


def load_oracle_alpha_control_registry(
    *,
    path: Path | None = None,
) -> OracleAlphaControlRegistry:
    registry_path = path or DEFAULT_CONTROL_REGISTRY_PATH
    raw = yaml.safe_load(registry_path.read_text())
    prompt_registry = load_prompt_registry()

    plans: dict[str, OracleAlphaControlPlan] = {}
    for plan_id, plan_raw in raw["plans"].items():
        plan = OracleAlphaControlPlan(
            plan_id=plan_id,
            collection_id=plan_raw["collection_id"],
            aggregation_unit=plan_raw["aggregation_unit"],
            bootstrap_resamples=int(plan_raw["bootstrap_resamples"]),
            null_models=tuple(plan_raw["null_models"]),
            stability_suite=StabilitySuitePlan(
                restart_seeds=tuple(
                    int(seed) for seed in plan_raw["stability_suite"]["restart_seeds"]
                ),
                prompt_perturbations=tuple(
                    plan_raw["stability_suite"]["prompt_perturbations"]
                ),
                report_metrics=tuple(plan_raw["stability_suite"]["report_metrics"]),
            ),
            predictiveness=PredictivenessPlan(
                train_collection_id=plan_raw["predictiveness"]["train_collection_id"],
                train_split=plan_raw["predictiveness"]["train_split"],
                eval_collection_id=plan_raw["predictiveness"]["eval_collection_id"],
                eval_split=plan_raw["predictiveness"]["eval_split"],
                model_family=plan_raw["predictiveness"]["model_family"],
                target=plan_raw["predictiveness"]["target"],
                features=plan_raw["predictiveness"]["features"],
                primary_metric=plan_raw["predictiveness"]["primary_metric"],
                secondary_metric=plan_raw["predictiveness"]["secondary_metric"],
                failure_action=plan_raw["predictiveness"]["failure_action"],
            ),
            mib_anchor=MIBPlan(
                mib_status=plan_raw["mib_anchor"]["mib_status"],
                benchmark=plan_raw["mib_anchor"].get("benchmark"),
                sanity_task=plan_raw["mib_anchor"].get("sanity_task"),
                rationale=plan_raw["mib_anchor"].get("rationale", ""),
                revisit_trigger=plan_raw["mib_anchor"].get("revisit_trigger", ""),
            ),
        )
        _validate_control_plan(plan, prompt_registry=prompt_registry)
        plans[plan_id] = plan

    return OracleAlphaControlRegistry(
        version=int(raw["version"]),
        registry_id=raw["registry_id"],
        plans=plans,
    )
