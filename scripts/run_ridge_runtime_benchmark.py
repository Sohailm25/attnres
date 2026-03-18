# ABOUTME: Benchmarks the held-out predictiveness ridge solve on a Gemma-shaped n<<d surface.
# ABOUTME: Compares the legacy primal solve against the adaptive helper without changing runner semantics.

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class RidgeRuntimeCase:
    case_name: str
    train_examples: int
    eval_examples: int
    feature_dim: int
    target_dim: int
    regularization_strength: float
    primal_seconds: float
    adaptive_seconds: float
    speedup_ratio: float
    max_abs_prediction_difference: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--pilot-count", type=int, default=96)
    parser.add_argument("--confirm-count", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=9216)
    parser.add_argument("--target-dim", type=int, default=53)
    parser.add_argument("--regularization-strength", type=float, default=100.0)
    parser.add_argument(
        "--feature-source",
        default=(
            "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_"
            "plus_mean_pooled_token_embedding_concat"
        ),
    )
    parser.add_argument("--target-name", default="oracle_alpha_logit_vector")
    return parser.parse_args()


def _synthetic_problem(
    *,
    rng: np.random.Generator,
    train_examples: int,
    eval_examples: int,
    feature_dim: int,
    target_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_features = rng.normal(size=(train_examples, feature_dim))
    eval_features = rng.normal(size=(eval_examples, feature_dim))
    latent_weights = rng.normal(size=(feature_dim, target_dim))
    train_targets = (train_features @ latent_weights) + (
        0.05 * rng.normal(size=(train_examples, target_dim))
    )
    return train_features, train_targets, eval_features


def _legacy_primal_ridge_predictions(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    eval_features: np.ndarray,
    regularization_strength: float,
) -> np.ndarray:
    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_scale = train_features.std(axis=0, keepdims=True)
    feature_scale[feature_scale == 0.0] = 1.0

    train_standardized = (train_features - feature_mean) / feature_scale
    eval_standardized = (eval_features - feature_mean) / feature_scale
    train_design = np.concatenate(
        [np.ones((train_standardized.shape[0], 1)), train_standardized],
        axis=1,
    )
    eval_design = np.concatenate(
        [np.ones((eval_standardized.shape[0], 1)), eval_standardized],
        axis=1,
    )
    regularizer = np.eye(train_design.shape[1], dtype=float)
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        train_design.T @ train_design + (regularization_strength * regularizer),
        train_design.T @ train_targets,
    )
    return eval_design @ coefficients


def _adaptive_ridge_predictions(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    eval_features: np.ndarray,
    regularization_strength: float,
) -> np.ndarray:
    from validation.oracle_alpha_controls import ridge_regression_predictions

    return ridge_regression_predictions(
        train_features=train_features.tolist(),
        train_targets=train_targets.tolist(),
        eval_features=eval_features.tolist(),
        regularization_strength=regularization_strength,
    )


def _benchmark_case(
    *,
    case_name: str,
    rng: np.random.Generator,
    train_examples: int,
    eval_examples: int,
    feature_dim: int,
    target_dim: int,
    regularization_strength: float,
) -> RidgeRuntimeCase:
    train_features, train_targets, eval_features = _synthetic_problem(
        rng=rng,
        train_examples=train_examples,
        eval_examples=eval_examples,
        feature_dim=feature_dim,
        target_dim=target_dim,
    )

    primal_start = time.perf_counter()
    legacy_predictions = _legacy_primal_ridge_predictions(
        train_features=train_features,
        train_targets=train_targets,
        eval_features=eval_features,
        regularization_strength=regularization_strength,
    )
    primal_seconds = time.perf_counter() - primal_start

    adaptive_start = time.perf_counter()
    adaptive_predictions = _adaptive_ridge_predictions(
        train_features=train_features,
        train_targets=train_targets,
        eval_features=eval_features,
        regularization_strength=regularization_strength,
    )
    adaptive_seconds = time.perf_counter() - adaptive_start

    return RidgeRuntimeCase(
        case_name=case_name,
        train_examples=train_examples,
        eval_examples=eval_examples,
        feature_dim=feature_dim,
        target_dim=target_dim,
        regularization_strength=regularization_strength,
        primal_seconds=primal_seconds,
        adaptive_seconds=adaptive_seconds,
        speedup_ratio=primal_seconds / adaptive_seconds,
        max_abs_prediction_difference=float(
            np.max(np.abs(legacy_predictions - adaptive_predictions))
        ),
    )


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cases = [
        _benchmark_case(
            case_name="pilot_leave_one_out_fold",
            rng=rng,
            train_examples=args.pilot_count - 1,
            eval_examples=1,
            feature_dim=args.feature_dim,
            target_dim=args.target_dim,
            regularization_strength=args.regularization_strength,
        ),
        _benchmark_case(
            case_name="pilot_to_confirm_fit",
            rng=rng,
            train_examples=args.pilot_count,
            eval_examples=args.confirm_count,
            feature_dim=args.feature_dim,
            target_dim=args.target_dim,
            regularization_strength=args.regularization_strength,
        ),
    ]

    summary = {
        "surface_name": "gemma2_registry_v4_predictiveness_shape",
        "pilot_count": args.pilot_count,
        "confirm_count": args.confirm_count,
        "feature_source": args.feature_source,
        "target_name": args.target_name,
        "feature_dim": args.feature_dim,
        "target_dim": args.target_dim,
        "regularization_strength": args.regularization_strength,
        "cases": [asdict(case) for case in cases],
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
