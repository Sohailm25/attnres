# ABOUTME: Runs the first development-model oracle-alpha slice on fixed cached residual sources.
# ABOUTME: Uses the saved prompt and control registries so runner code cannot drift from the repo contract.

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from prompts import PromptEntry, perturb_prompt_entry, resolve_prompt_entries
from .model_backed import cache_name_filter
from .oracle_alpha_controls import (
    BootstrapMeanInterval,
    PredictivenessSummary,
    bootstrap_mean_confidence_interval,
    jensen_shannon_divergence,
    load_oracle_alpha_control_registry,
    mean_pairwise_js_divergence,
    mean_top1_source_agreement,
    mean_topk_jaccard_similarity,
    ridge_alpha_predictiveness_summary,
    ridge_regression_predictions,
)


@dataclass(frozen=True)
class OracleAlphaSequenceResult:
    prompt_id: str
    prompt: str
    split: str
    num_sources: int
    source_labels: tuple[str, ...]
    uniform_loss: float
    optimized_loss: float
    null_losses: dict[str, float]
    best_alpha_entropy: float
    best_alpha: tuple[float, ...]
    final_alpha: tuple[float, ...]


@dataclass(frozen=True)
class OracleAlphaRunSummary:
    model_name: str
    collection_id: str
    split: str
    exploratory: bool
    control_plan_id: str
    control_registry_id: str
    num_sequences: int
    sequence_mean_improvement: float
    bootstrap_interval: BootstrapMeanInterval
    null_model_mean_losses: dict[str, float]
    sequence_results: tuple[OracleAlphaSequenceResult, ...]


@dataclass(frozen=True)
class OracleAlphaStabilityMetrics:
    num_runs: int
    mean_pairwise_js_divergence: float
    mean_topk_jaccard_at_4: float
    mean_top1_source_agreement: float
    mean_sequence_improvement: float


@dataclass(frozen=True)
class OracleAlphaStabilitySummary:
    model_name: str
    collection_id: str
    split: str
    exploratory: bool
    control_plan_id: str
    control_registry_id: str
    base_run: OracleAlphaRunSummary
    restart_runs: tuple[OracleAlphaRunSummary, ...]
    restart_metrics: OracleAlphaStabilityMetrics
    paraphrase_run: OracleAlphaRunSummary | None
    paraphrase_metrics: OracleAlphaStabilityMetrics | None
    resample_runs: tuple[OracleAlphaRunSummary, ...]
    resample_metrics: OracleAlphaStabilityMetrics | None


@dataclass(frozen=True)
class OracleAlphaPredictivenessSequenceResult:
    prompt_id: str
    prompt: str
    split: str
    num_sources: int
    predicted_alpha: tuple[float, ...]
    oracle_alpha: tuple[float, ...]
    predicted_loss: float
    oracle_loss: float
    uniform_loss: float
    js_divergence_to_oracle: float


@dataclass(frozen=True)
class OracleAlphaPredictivenessSummary:
    model_name: str
    collection_id: str
    train_split: str
    eval_split: str
    control_plan_id: str
    control_registry_id: str
    feature_source: str
    selected_regularization_strength: float
    tuning_mean_js_divergence: float
    candidate_feature_summaries: tuple["OracleAlphaPredictivenessFeatureCandidate", ...]
    predictiveness_summary: PredictivenessSummary
    train_run: OracleAlphaRunSummary
    eval_run: OracleAlphaRunSummary
    eval_predictions: tuple[OracleAlphaPredictivenessSequenceResult, ...]
    mib_status: str
    mib_rationale: str


@dataclass(frozen=True)
class OracleAlphaPredictivenessFeatureCandidate:
    feature_source: str
    selected_regularization_strength: float
    tuning_mean_js_divergence: float


def _apply_final_norm_and_unembed(
    model: HookedTransformer,
    residual: torch.Tensor,
) -> torch.Tensor:
    if residual.ndim == 2:
        residual = residual.unsqueeze(0)
    if model.cfg.normalization_type is not None:
        residual = model.ln_final(residual)
    logits = model.unembed(residual)
    if model.cfg.output_logits_soft_cap > 0.0:
        logits = model.cfg.output_logits_soft_cap * torch.tanh(
            logits / model.cfg.output_logits_soft_cap
        )
    return logits.squeeze(0)


def _sequence_mean_cross_entropy(
    logits: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [pos, vocab]")
    if tokens.ndim != 1:
        raise ValueError("tokens must have shape [pos]")
    if logits.shape[0] != tokens.shape[0]:
        raise ValueError("logits and tokens must share the same sequence length")

    return F.cross_entropy(
        logits[:-1],
        tokens[1:],
        reduction="mean",
    )


def _mixture_from_alpha(
    residual_stack: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("s,spd->pd", alpha, residual_stack)


def _loss_for_alpha(
    *,
    model: HookedTransformer,
    residual_stack: torch.Tensor,
    tokens: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    mixture = _mixture_from_alpha(residual_stack, alpha)
    logits = _apply_final_norm_and_unembed(model, mixture)
    return _sequence_mean_cross_entropy(logits, tokens)


def _magnitude_proportional_alpha(residual_stack: torch.Tensor) -> torch.Tensor:
    magnitudes = residual_stack.norm(dim=-1).mean(dim=-1)
    total = magnitudes.sum()
    if total <= 0:
        return torch.full_like(magnitudes, 1.0 / len(magnitudes))
    return magnitudes / total


def _last_layer_only_alpha(num_sources: int, *, device: torch.device) -> torch.Tensor:
    alpha = torch.zeros(num_sources, device=device)
    alpha[-1] = 1.0
    return alpha


def _random_dirichlet_alpha(
    num_sources: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    sampled = rng.dirichlet(np.ones(num_sources, dtype=float))
    return torch.tensor(sampled, dtype=torch.float32, device=device)


def _entropy(alpha: torch.Tensor) -> float:
    safe = torch.clamp(alpha, min=1e-12)
    return float((-(safe * safe.log()).sum()).detach().cpu().item())


def _initial_alpha_logits(
    *,
    num_sources: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = (
        torch.randn(
            num_sources,
            generator=generator,
            dtype=torch.float32,
        )
        * 1e-3
    )
    return noise.to(device=device, dtype=dtype)


def _fixed_residual_sources(
    *,
    model: HookedTransformer,
    prompt: str,
    prepend_bos: bool | None,
) -> tuple[torch.Tensor, tuple[str, ...], torch.Tensor]:
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            return_type="logits",
            names_filter=cache_name_filter,
        )
        residual_stack, labels = cache.decompose_resid(
            layer=model.cfg.n_layers,
            mode="all",
            incl_embeds=True,
            return_labels=True,
        )
    return residual_stack[:, 0].detach(), tuple(labels), tokens[0].detach()


def _resid_post_states(
    *,
    model: HookedTransformer,
    prompt: str,
    prepend_bos: bool | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            return_type="logits",
            names_filter=lambda name: name.endswith("hook_resid_post"),
        )
    h_1 = cache[("resid_post", 0)][0]
    h_4 = cache[("resid_post", min(3, model.cfg.n_layers - 1))][0]
    return h_1, h_4


def _feature_vector_for_source(
    *,
    model: HookedTransformer,
    prompt: str,
    prepend_bos: bool | None,
    feature_source: str,
) -> list[float]:
    h_1, h_4 = _resid_post_states(
        model=model,
        prompt=prompt,
        prepend_bos=prepend_bos,
    )
    if feature_source == "mean_pooled_h_1[t]_resid_post_layer_0":
        feature = h_1.mean(dim=0)
    elif feature_source == "mean_pooled_h_4[t]_resid_post_layer_3":
        feature = h_4.mean(dim=0)
    elif feature_source == "mean_pooled_h_1[t]_plus_h_4[t]_concat":
        feature = torch.cat((h_1.mean(dim=0), h_4.mean(dim=0)))
    elif feature_source == "final_token_h_1[t]_plus_h_4[t]_concat":
        feature = torch.cat((h_1[-1], h_4[-1]))
    else:
        raise ValueError(f"unsupported feature source {feature_source!r}")
    return [float(value) for value in feature.detach().cpu().tolist()]


def _normalize_predicted_alpha(
    predicted_values: Sequence[float],
) -> list[float]:
    clipped = [max(0.0, float(value)) for value in predicted_values]
    total = sum(clipped)
    if total <= 0.0:
        return [1.0 / len(clipped)] * len(clipped)
    return [value / total for value in clipped]


def _loss_for_predicted_alpha(
    *,
    model: HookedTransformer,
    entry: PromptEntry,
    predicted_alpha: Sequence[float],
    prepend_bos: bool | None,
) -> float:
    device = torch.device(str(model.cfg.device))
    residual_stack, _, tokens = _fixed_residual_sources(
        model=model,
        prompt=entry.text,
        prepend_bos=prepend_bos,
    )
    alpha = torch.tensor(
        predicted_alpha,
        dtype=residual_stack.dtype,
        device=device,
    )
    residual_stack = residual_stack.to(device)
    tokens = tokens.to(device)
    with torch.no_grad():
        return float(
            _loss_for_alpha(
                model=model,
                residual_stack=residual_stack,
                tokens=tokens,
                alpha=alpha,
            )
            .detach()
            .cpu()
            .item()
        )


def _optimize_sequence(
    *,
    model: HookedTransformer,
    entry: PromptEntry,
    optimization_steps: int,
    learning_rate: float,
    seed: int,
    prepend_bos: bool | None,
) -> OracleAlphaSequenceResult:
    device = torch.device(str(model.cfg.device))
    residual_stack, labels, tokens = _fixed_residual_sources(
        model=model,
        prompt=entry.text,
        prepend_bos=prepend_bos,
    )
    residual_stack = residual_stack.to(device)
    tokens = tokens.to(device)

    num_sources = residual_stack.shape[0]
    uniform_alpha = torch.full(
        (num_sources,),
        1.0 / num_sources,
        dtype=residual_stack.dtype,
        device=device,
    )
    with torch.no_grad():
        uniform_loss = float(
            _loss_for_alpha(
                model=model,
                residual_stack=residual_stack,
                tokens=tokens,
                alpha=uniform_alpha,
            )
            .detach()
            .cpu()
            .item()
        )

    null_losses = {
        "uniform": uniform_loss,
        "random_dirichlet": float(
            _loss_for_alpha(
                model=model,
                residual_stack=residual_stack,
                tokens=tokens,
                alpha=_random_dirichlet_alpha(
                    num_sources,
                    seed=seed + 1000,
                    device=device,
                ),
            )
            .detach()
            .cpu()
            .item()
        ),
        "magnitude_proportional": float(
            _loss_for_alpha(
                model=model,
                residual_stack=residual_stack,
                tokens=tokens,
                alpha=_magnitude_proportional_alpha(residual_stack),
            )
            .detach()
            .cpu()
            .item()
        ),
        "last_layer_only": float(
            _loss_for_alpha(
                model=model,
                residual_stack=residual_stack,
                tokens=tokens,
                alpha=_last_layer_only_alpha(num_sources, device=device),
            )
            .detach()
            .cpu()
            .item()
        ),
    }

    z = _initial_alpha_logits(
        num_sources=num_sources,
        seed=seed,
        device=device,
        dtype=residual_stack.dtype,
    ).requires_grad_()
    optimizer = torch.optim.Adam([z], lr=learning_rate)
    best_loss = uniform_loss
    best_alpha = uniform_alpha.detach().clone()

    for _ in range(optimization_steps):
        optimizer.zero_grad()
        alpha = torch.softmax(z, dim=0)
        loss = _loss_for_alpha(
            model=model,
            residual_stack=residual_stack,
            tokens=tokens,
            alpha=alpha,
        )
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_alpha = alpha.detach().clone()

    final_alpha = torch.softmax(z.detach(), dim=0)

    return OracleAlphaSequenceResult(
        prompt_id=entry.prompt_id,
        prompt=entry.text,
        split=entry.split,
        num_sources=num_sources,
        source_labels=labels,
        uniform_loss=uniform_loss,
        optimized_loss=best_loss,
        null_losses=null_losses,
        best_alpha_entropy=_entropy(best_alpha),
        best_alpha=tuple(float(value) for value in best_alpha.detach().cpu().tolist()),
        final_alpha=tuple(float(value) for value in final_alpha.cpu().tolist()),
    )


def run_oracle_alpha_collection(
    *,
    model: HookedTransformer,
    collection_id: str,
    split: str,
    exploratory: bool,
    prompt_entries: Sequence[PromptEntry] | None = None,
    max_sequences: int | None = None,
    optimization_steps: int = 20,
    learning_rate: float = 0.1,
    seed: int = 0,
    prepend_bos: bool | None = None,
) -> OracleAlphaRunSummary:
    control_registry = load_oracle_alpha_control_registry()
    control_plan = control_registry.plans[collection_id]
    if prompt_entries is None:
        entries = list(
            resolve_prompt_entries(
                collection_id=collection_id,
                split=split,
                exploratory=exploratory,
            )
        )
    else:
        entries = list(prompt_entries)
    if max_sequences is not None:
        entries = entries[:max_sequences]
    if not entries:
        raise ValueError("at least one prompt entry is required")

    sequence_results = tuple(
        _optimize_sequence(
            model=model,
            entry=entry,
            optimization_steps=optimization_steps,
            learning_rate=learning_rate,
            seed=seed + index,
            prepend_bos=prepend_bos,
        )
        for index, entry in enumerate(entries)
    )
    improvements = [
        result.uniform_loss - result.optimized_loss for result in sequence_results
    ]
    bootstrap_interval = bootstrap_mean_confidence_interval(
        improvements,
        num_resamples=control_plan.bootstrap_resamples,
        confidence_level=0.95,
        seed=seed,
    )

    null_model_names = tuple(control_plan.null_models)
    null_model_mean_losses = {
        name: sum(result.null_losses[name] for result in sequence_results)
        / len(sequence_results)
        for name in null_model_names
    }

    return OracleAlphaRunSummary(
        model_name=model.cfg.model_name,
        collection_id=collection_id,
        split=split,
        exploratory=exploratory,
        control_plan_id=control_plan.plan_id,
        control_registry_id=control_registry.registry_id,
        num_sequences=len(sequence_results),
        sequence_mean_improvement=sum(improvements) / len(improvements),
        bootstrap_interval=bootstrap_interval,
        null_model_mean_losses=null_model_mean_losses,
        sequence_results=sequence_results,
    )


def _aggregate_alpha_distribution(run: OracleAlphaRunSummary) -> list[float]:
    if not run.sequence_results:
        raise ValueError("run must contain sequence results")
    num_sources = run.sequence_results[0].num_sources
    aggregate = [0.0] * num_sources
    for result in run.sequence_results:
        if len(result.final_alpha) != num_sources:
            raise ValueError("all alpha vectors must share the same source count")
        for index, value in enumerate(result.final_alpha):
            aggregate[index] += value
    return [value / len(run.sequence_results) for value in aggregate]


def _stability_metrics_for_runs(
    runs: Sequence[OracleAlphaRunSummary],
) -> OracleAlphaStabilityMetrics:
    if not runs:
        raise ValueError("at least one run is required")
    aggregated = [_aggregate_alpha_distribution(run) for run in runs]
    num_sources = len(aggregated[0])
    if len(runs) == 1:
        js_divergence = 0.0
        topk_jaccard = 1.0
        top1_agreement = 1.0
    else:
        js_divergence = mean_pairwise_js_divergence(aggregated)
        topk_jaccard = mean_topk_jaccard_similarity(
            aggregated,
            k=min(4, num_sources),
        )
        top1_agreement = mean_top1_source_agreement(aggregated)

    return OracleAlphaStabilityMetrics(
        num_runs=len(runs),
        mean_pairwise_js_divergence=js_divergence,
        mean_topk_jaccard_at_4=topk_jaccard,
        mean_top1_source_agreement=top1_agreement,
        mean_sequence_improvement=sum(run.sequence_mean_improvement for run in runs)
        / len(runs),
    )


def run_oracle_alpha_stability_suite(
    *,
    model: HookedTransformer,
    collection_id: str,
    split: str,
    exploratory: bool,
    max_sequences: int | None = None,
    optimization_steps: int = 20,
    learning_rate: float = 0.1,
    restart_seeds: Sequence[int] | None = None,
    resample_count: int = 3,
    resample_size: int | None = None,
    prepend_bos: bool | None = None,
) -> OracleAlphaStabilitySummary:
    control_registry = load_oracle_alpha_control_registry()
    control_plan = control_registry.plans[collection_id]
    base_entries = list(
        resolve_prompt_entries(
            collection_id=collection_id,
            split=split,
            exploratory=exploratory,
        )
    )
    if max_sequences is not None:
        base_entries = base_entries[:max_sequences]
    if not base_entries:
        raise ValueError("at least one prompt entry is required")

    seeds = tuple(restart_seeds or control_plan.stability_suite.restart_seeds)
    if not seeds:
        raise ValueError("at least one restart seed is required")

    restart_runs = tuple(
        run_oracle_alpha_collection(
            model=model,
            collection_id=collection_id,
            split=split,
            exploratory=exploratory,
            prompt_entries=base_entries,
            optimization_steps=optimization_steps,
            learning_rate=learning_rate,
            seed=seed,
            prepend_bos=prepend_bos,
        )
        for seed in seeds
    )
    base_run = restart_runs[0]
    restart_metrics = _stability_metrics_for_runs(restart_runs)

    paraphrase_run: OracleAlphaRunSummary | None = None
    paraphrase_metrics: OracleAlphaStabilityMetrics | None = None
    if "prompt_paraphrase" in control_plan.stability_suite.prompt_perturbations:
        paraphrase_entries = [
            perturb_prompt_entry(entry, perturbation_name="prompt_paraphrase")
            for entry in base_entries
        ]
        paraphrase_run = run_oracle_alpha_collection(
            model=model,
            collection_id=collection_id,
            split=split,
            exploratory=exploratory,
            prompt_entries=paraphrase_entries,
            optimization_steps=optimization_steps,
            learning_rate=learning_rate,
            seed=seeds[0],
            prepend_bos=prepend_bos,
        )
        paraphrase_metrics = _stability_metrics_for_runs((base_run, paraphrase_run))

    resample_runs: tuple[OracleAlphaRunSummary, ...] = ()
    resample_metrics: OracleAlphaStabilityMetrics | None = None
    if "prompt_resample" in control_plan.stability_suite.prompt_perturbations:
        if len(base_entries) < 2:
            raise ValueError("prompt resampling requires at least two base entries")
        sample_size = resample_size or max(2, len(base_entries) - 1)
        sample_size = min(sample_size, len(base_entries))
        rng = np.random.default_rng(seeds[0] + 4096)
        resample_runs = tuple(
            run_oracle_alpha_collection(
                model=model,
                collection_id=collection_id,
                split=split,
                exploratory=exploratory,
                prompt_entries=[
                    base_entries[index]
                    for index in sorted(
                        rng.choice(
                            len(base_entries),
                            size=sample_size,
                            replace=False,
                        ).tolist()
                    )
                ],
                optimization_steps=optimization_steps,
                learning_rate=learning_rate,
                seed=seeds[0] + 200 + resample_index,
                prepend_bos=prepend_bos,
            )
            for resample_index in range(resample_count)
        )
        resample_metrics = _stability_metrics_for_runs((base_run, *resample_runs))

    return OracleAlphaStabilitySummary(
        model_name=model.cfg.model_name,
        collection_id=collection_id,
        split=split,
        exploratory=exploratory,
        control_plan_id=control_plan.plan_id,
        control_registry_id=control_registry.registry_id,
        base_run=base_run,
        restart_runs=restart_runs,
        restart_metrics=restart_metrics,
        paraphrase_run=paraphrase_run,
        paraphrase_metrics=paraphrase_metrics,
        resample_runs=resample_runs,
        resample_metrics=resample_metrics,
    )


def _tuned_ridge_regularization(
    *,
    train_features: Sequence[Sequence[float]],
    train_targets: Sequence[Sequence[float]],
    regularization_grid: Sequence[float],
) -> tuple[float, float]:
    if len(train_features) < 2:
        raise ValueError("at least two training examples are required")
    if not regularization_grid:
        raise ValueError("regularization_grid must not be empty")

    best_regularization: float | None = None
    best_mean_js: float | None = None
    num_examples = len(train_features)
    for regularization_strength in regularization_grid:
        holdout_js_values = []
        for holdout_index in range(num_examples):
            fold_train_features = [
                feature
                for index, feature in enumerate(train_features)
                if index != holdout_index
            ]
            fold_train_targets = [
                target
                for index, target in enumerate(train_targets)
                if index != holdout_index
            ]
            predicted = ridge_regression_predictions(
                train_features=fold_train_features,
                train_targets=fold_train_targets,
                eval_features=[train_features[holdout_index]],
                regularization_strength=regularization_strength,
            )[0].tolist()
            holdout_js_values.append(
                jensen_shannon_divergence(
                    _normalize_predicted_alpha(predicted),
                    train_targets[holdout_index],
                )
            )

        mean_js = sum(holdout_js_values) / len(holdout_js_values)
        if best_mean_js is None or mean_js < best_mean_js:
            best_regularization = float(regularization_strength)
            best_mean_js = mean_js

    return best_regularization, best_mean_js


def run_oracle_alpha_predictiveness_check(
    *,
    model: HookedTransformer,
    collection_id: str,
    max_train_sequences: int | None = None,
    max_eval_sequences: int | None = None,
    optimization_steps: int = 20,
    learning_rate: float = 0.1,
    seed: int = 0,
    regularization_grid: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    candidate_feature_sources: Sequence[str] | None = None,
    prepend_bos: bool | None = None,
) -> OracleAlphaPredictivenessSummary:
    control_registry = load_oracle_alpha_control_registry()
    control_plan = control_registry.plans[collection_id]
    predictiveness_plan = control_plan.predictiveness
    if predictiveness_plan.model_family != "ridge_regression":
        raise ValueError("predictiveness plan must use ridge_regression")

    train_entries = list(
        resolve_prompt_entries(
            collection_id=predictiveness_plan.train_collection_id,
            split=predictiveness_plan.train_split,
            exploratory=True,
        )
    )
    eval_entries = list(
        resolve_prompt_entries(
            collection_id=predictiveness_plan.eval_collection_id,
            split=predictiveness_plan.eval_split,
            exploratory=False,
        )
    )
    if max_train_sequences is not None:
        train_entries = train_entries[:max_train_sequences]
    if max_eval_sequences is not None:
        eval_entries = eval_entries[:max_eval_sequences]
    if len(train_entries) < 2:
        raise ValueError("predictiveness training requires at least two pilot prompts")
    if not eval_entries:
        raise ValueError("predictiveness evaluation requires confirm prompts")
    feature_sources = tuple(
        candidate_feature_sources or ("mean_pooled_h_1[t]_resid_post_layer_0",)
    )
    if not feature_sources:
        raise ValueError("candidate_feature_sources must not be empty")

    train_run = run_oracle_alpha_collection(
        model=model,
        collection_id=collection_id,
        split=predictiveness_plan.train_split,
        exploratory=True,
        prompt_entries=train_entries,
        optimization_steps=optimization_steps,
        learning_rate=learning_rate,
        seed=seed,
        prepend_bos=prepend_bos,
    )
    eval_run = run_oracle_alpha_collection(
        model=model,
        collection_id=collection_id,
        split=predictiveness_plan.eval_split,
        exploratory=False,
        prompt_entries=eval_entries,
        optimization_steps=optimization_steps,
        learning_rate=learning_rate,
        seed=seed + 1000,
        prepend_bos=prepend_bos,
    )

    train_targets = [list(result.final_alpha) for result in train_run.sequence_results]
    eval_targets = [list(result.final_alpha) for result in eval_run.sequence_results]
    candidate_feature_summaries = []
    selected_feature_source = None
    selected_regularization_strength = None
    tuning_mean_js_divergence = None
    selected_train_features = None
    for feature_source in feature_sources:
        train_features = [
            _feature_vector_for_source(
                model=model,
                prompt=entry.text,
                prepend_bos=prepend_bos,
                feature_source=feature_source,
            )
            for entry in train_entries
        ]
        (
            candidate_regularization_strength,
            candidate_tuning_mean_js,
        ) = _tuned_ridge_regularization(
            train_features=train_features,
            train_targets=train_targets,
            regularization_grid=regularization_grid,
        )
        candidate_feature_summaries.append(
            OracleAlphaPredictivenessFeatureCandidate(
                feature_source=feature_source,
                selected_regularization_strength=candidate_regularization_strength,
                tuning_mean_js_divergence=candidate_tuning_mean_js,
            )
        )
        if (
            tuning_mean_js_divergence is None
            or candidate_tuning_mean_js < tuning_mean_js_divergence
        ):
            selected_feature_source = feature_source
            selected_regularization_strength = candidate_regularization_strength
            tuning_mean_js_divergence = candidate_tuning_mean_js
            selected_train_features = train_features

    eval_features = [
        _feature_vector_for_source(
            model=model,
            prompt=entry.text,
            prepend_bos=prepend_bos,
            feature_source=selected_feature_source,
        )
        for entry in eval_entries
    ]
    predictiveness_summary = ridge_alpha_predictiveness_summary(
        train_features=selected_train_features,
        train_targets=train_targets,
        eval_features=eval_features,
        eval_targets=eval_targets,
        regularization_strength=selected_regularization_strength,
    )
    predicted_eval_alphas = ridge_regression_predictions(
        train_features=selected_train_features,
        train_targets=train_targets,
        eval_features=eval_features,
        regularization_strength=selected_regularization_strength,
    )
    eval_predictions = []
    for index, result in enumerate(eval_run.sequence_results):
        normalized_predicted_alpha = _normalize_predicted_alpha(
            predicted_eval_alphas[index].tolist()
        )
        eval_predictions.append(
            OracleAlphaPredictivenessSequenceResult(
                prompt_id=result.prompt_id,
                prompt=result.prompt,
                split=result.split,
                num_sources=result.num_sources,
                predicted_alpha=tuple(normalized_predicted_alpha),
                oracle_alpha=result.final_alpha,
                predicted_loss=_loss_for_predicted_alpha(
                    model=model,
                    entry=eval_entries[index],
                    predicted_alpha=normalized_predicted_alpha,
                    prepend_bos=prepend_bos,
                ),
                oracle_loss=result.optimized_loss,
                uniform_loss=result.uniform_loss,
                js_divergence_to_oracle=jensen_shannon_divergence(
                    normalized_predicted_alpha,
                    result.final_alpha,
                ),
            )
        )

    mib_rationale = (
        "omitted for the current development-model runner stage because the "
        "pilot/confirm prompt registry is a custom local prompt slice rather than "
        "a benchmark-compatible task surface, so a MIB-style sanity task would be "
        "artificial here; keep the global MIB anchor planned for a later compatible lane"
    )

    return OracleAlphaPredictivenessSummary(
        model_name=model.cfg.model_name,
        collection_id=collection_id,
        train_split=predictiveness_plan.train_split,
        eval_split=predictiveness_plan.eval_split,
        control_plan_id=control_plan.plan_id,
        control_registry_id=control_registry.registry_id,
        feature_source=selected_feature_source,
        selected_regularization_strength=selected_regularization_strength,
        tuning_mean_js_divergence=tuning_mean_js_divergence,
        candidate_feature_summaries=tuple(candidate_feature_summaries),
        predictiveness_summary=predictiveness_summary,
        train_run=train_run,
        eval_run=eval_run,
        eval_predictions=tuple(eval_predictions),
        mib_status="omitted",
        mib_rationale=mib_rationale,
    )
