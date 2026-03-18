# ABOUTME: Runs and summarizes the preregistered oracle-alpha routing-regime comparison.
# ABOUTME: Keeps regime scheduling, checkpoint reuse, and compact summary writing separate from the core runner.

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence

from transformer_lens import HookedTransformer

from prompts import resolve_prompt_entries
from .oracle_alpha_runner import OracleAlphaRunSummary, run_oracle_alpha_collection


@dataclass(frozen=True)
class OracleAlphaRegimeSpec:
    regime: str
    top_k: int | None = None

    @property
    def slug(self) -> str:
        if self.top_k is None:
            return self.regime
        return f"{self.regime}-k{self.top_k}"


@dataclass(frozen=True)
class OracleAlphaRegimeResult:
    regime: str
    top_k: int | None
    num_sequences: int
    sequence_mean_improvement: float
    bootstrap_interval: dict[str, float]
    null_model_mean_losses: dict[str, float]
    positive_prompt_count: int
    mean_effective_sources: float
    mean_gini: float
    mean_top1_mass: float


@dataclass(frozen=True)
class OracleAlphaRegimeComparisonSummary:
    model_name: str
    collection_id: str
    split: str
    regime_results: tuple[OracleAlphaRegimeResult, ...]


def build_regime_specs(num_sources: int) -> tuple[OracleAlphaRegimeSpec, ...]:
    if num_sources < 1:
        raise ValueError("num_sources must be positive")
    top_k_values = sorted(
        {
            min(num_sources, value)
            for value in (2, 4, 8, max(1, num_sources // 4), max(1, num_sources // 2))
            if value <= num_sources
        }
    )
    specs = [
        OracleAlphaRegimeSpec(regime="softmax-constrained"),
        OracleAlphaRegimeSpec(regime="unconstrained"),
    ]
    specs.extend(
        OracleAlphaRegimeSpec(regime="top-k", top_k=value) for value in top_k_values
    )
    return tuple(specs)


def _normalized_distribution(alpha: Sequence[float]) -> list[float]:
    if not alpha:
        raise ValueError("alpha must not be empty")
    positive = [max(0.0, float(value)) for value in alpha]
    total = sum(positive)
    if total <= 0.0:
        return [1.0 / len(positive)] * len(positive)
    return [value / total for value in positive]


def _effective_sources(alpha: Sequence[float]) -> float:
    distribution = _normalized_distribution(alpha)
    entropy = -sum(value * math.log(value) for value in distribution if value > 0.0)
    return math.exp(entropy)


def _gini(alpha: Sequence[float]) -> float:
    distribution = sorted(_normalized_distribution(alpha))
    n = len(distribution)
    if n == 0:
        raise ValueError("alpha must not be empty")
    weighted = sum((index + 1) * value for index, value in enumerate(distribution))
    return (2.0 * weighted / n) - ((n + 1.0) / n)


def summarize_regime_run(
    run_summary: OracleAlphaRunSummary | dict[str, object],
    *,
    regime: str,
    top_k: int | None = None,
) -> OracleAlphaRegimeResult:
    if isinstance(run_summary, OracleAlphaRunSummary):
        payload = asdict(run_summary)
    else:
        payload = run_summary
    sequence_results = payload["sequence_results"]
    best_alphas = [result["best_alpha"] for result in sequence_results]
    return OracleAlphaRegimeResult(
        regime=regime,
        top_k=top_k,
        num_sequences=payload["num_sequences"],
        sequence_mean_improvement=payload["sequence_mean_improvement"],
        bootstrap_interval=dict(payload["bootstrap_interval"]),
        null_model_mean_losses=dict(payload["null_model_mean_losses"]),
        positive_prompt_count=sum(
            1
            for result in sequence_results
            if result["uniform_loss"] - result["optimized_loss"] > 0.0
        ),
        mean_effective_sources=sum(_effective_sources(alpha) for alpha in best_alphas)
        / len(best_alphas),
        mean_gini=sum(_gini(alpha) for alpha in best_alphas) / len(best_alphas),
        mean_top1_mass=sum(
            max(_normalized_distribution(alpha)) for alpha in best_alphas
        )
        / len(best_alphas),
    )


def _regime_run_path(output_dir: Path, spec: OracleAlphaRegimeSpec) -> Path:
    return output_dir / "runs" / f"{spec.slug}.json"


def _load_or_run_regime(
    *,
    model: HookedTransformer,
    collection_id: str,
    split: str,
    exploratory: bool,
    prompt_entries,
    optimization_steps: int,
    learning_rate: float,
    seed: int,
    prepend_bos: bool | None,
    spec: OracleAlphaRegimeSpec,
    output_dir: Path,
) -> dict[str, object]:
    run_path = _regime_run_path(output_dir, spec)
    if run_path.is_file():
        return json.loads(run_path.read_text())

    run_summary = run_oracle_alpha_collection(
        model=model,
        collection_id=collection_id,
        split=split,
        exploratory=exploratory,
        prompt_entries=prompt_entries,
        optimization_steps=optimization_steps,
        learning_rate=learning_rate,
        seed=seed,
        prepend_bos=prepend_bos,
        regime=spec.regime,
        top_k=spec.top_k,
        matched_zero_init=True,
    )
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(asdict(run_summary), indent=2) + "\n")
    return asdict(run_summary)


def write_regime_comparison_summary(
    *,
    model: HookedTransformer,
    collection_id: str,
    split: str,
    exploratory: bool,
    output_dir: Path,
    optimization_steps: int,
    learning_rate: float,
    seed: int,
    prepend_bos: bool | None = None,
) -> Path:
    prompt_entries = list(
        resolve_prompt_entries(
            collection_id=collection_id,
            split=split,
            exploratory=exploratory,
        )
    )
    if not prompt_entries:
        raise ValueError("at least one prompt entry is required")

    softmax_spec = OracleAlphaRegimeSpec(regime="softmax-constrained")
    softmax_payload = _load_or_run_regime(
        model=model,
        collection_id=collection_id,
        split=split,
        exploratory=exploratory,
        prompt_entries=prompt_entries,
        optimization_steps=optimization_steps,
        learning_rate=learning_rate,
        seed=seed,
        prepend_bos=prepend_bos,
        spec=softmax_spec,
        output_dir=output_dir,
    )

    num_sources = softmax_payload["sequence_results"][0]["num_sources"]
    regime_specs = build_regime_specs(num_sources)
    regime_payloads = [softmax_payload]
    for spec in regime_specs[1:]:
        regime_payloads.append(
            _load_or_run_regime(
                model=model,
                collection_id=collection_id,
                split=split,
                exploratory=exploratory,
                prompt_entries=prompt_entries,
                optimization_steps=optimization_steps,
                learning_rate=learning_rate,
                seed=seed,
                prepend_bos=prepend_bos,
                spec=spec,
                output_dir=output_dir,
            )
        )

    summary = OracleAlphaRegimeComparisonSummary(
        model_name=model.cfg.model_name,
        collection_id=collection_id,
        split=split,
        regime_results=tuple(
            summarize_regime_run(
                payload,
                regime=spec.regime,
                top_k=spec.top_k,
            )
            for spec, payload in zip(regime_specs, regime_payloads, strict=True)
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2) + "\n")
    return summary_path
