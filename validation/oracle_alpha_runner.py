# ABOUTME: Runs the first development-model oracle-alpha slice on fixed cached residual sources.
# ABOUTME: Uses the saved prompt and control registries so runner code cannot drift from the repo contract.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from prompts import PromptEntry, perturb_prompt_entry, resolve_prompt_entries
from .model_backed import cache_name_filter
from .oracle_alpha_controls import (
    BootstrapMeanInterval,
    PredictivenessSummary,
    alpha_target_matrix,
    alpha_target_predictions_to_distributions,
    bootstrap_mean_confidence_interval,
    compare_predictiveness_metric_values,
    jensen_shannon_divergence,
    load_oracle_alpha_control_registry,
    mean_pairwise_js_divergence,
    mean_top1_source_agreement,
    mean_topk_jaccard_similarity,
    predictiveness_metric_value,
    predictiveness_summary_from_predictions,
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
class OracleAlphaPromptMatchedStabilityMetrics:
    matched_prompt_count: int
    mean_pairwise_js_divergence: float
    mean_topk_jaccard_at_4: float
    mean_top1_source_agreement: float


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
    restart_per_sequence_metrics: OracleAlphaPromptMatchedStabilityMetrics
    paraphrase_run: OracleAlphaRunSummary | None
    paraphrase_metrics: OracleAlphaStabilityMetrics | None
    paraphrase_per_sequence_metrics: OracleAlphaPromptMatchedStabilityMetrics | None
    resample_runs: tuple[OracleAlphaRunSummary, ...]
    resample_metrics: OracleAlphaStabilityMetrics | None
    resample_per_sequence_metrics: OracleAlphaPromptMatchedStabilityMetrics | None


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
    target: str
    control_plan_id: str
    control_registry_id: str
    feature_source: str
    tuning_primary_metric: str
    tuning_secondary_metric: str
    tuning_primary_metric_value: float
    tuning_secondary_metric_value: float
    selected_regularization_strength: float
    tuning_mean_js_divergence: float
    candidate_feature_summaries: tuple["OracleAlphaPredictivenessFeatureCandidate", ...]
    predictiveness_summary: PredictivenessSummary
    train_run: OracleAlphaRunSummary
    eval_run: OracleAlphaRunSummary
    eval_predictions: tuple[OracleAlphaPredictivenessSequenceResult, ...]
    predicted_mean_improvement_over_uniform: float
    oracle_mean_improvement_over_uniform: float
    mib_status: str
    mib_rationale: str


@dataclass(frozen=True)
class OracleAlphaPredictivenessRegularizationCandidate:
    regularization_strength: float
    tuning_primary_metric: str
    tuning_primary_metric_value: float
    tuning_secondary_metric: str
    tuning_secondary_metric_value: float
    tuning_r_squared: float
    tuning_mean_js_divergence: float


@dataclass(frozen=True)
class OracleAlphaPredictivenessFeatureCandidate:
    target: str
    feature_source: str
    selected_regularization_strength: float
    tuning_primary_metric: str
    tuning_primary_metric_value: float
    tuning_secondary_metric: str
    tuning_secondary_metric_value: float
    tuning_r_squared: float
    tuning_mean_js_divergence: float
    regularization_summaries: tuple[
        OracleAlphaPredictivenessRegularizationCandidate,
        ...,
    ]


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


def _alpha_from_regime_logits(
    logits: torch.Tensor,
    *,
    regime: str,
    top_k: int | None = None,
) -> torch.Tensor:
    if regime == "softmax-constrained":
        return torch.softmax(logits, dim=0)
    if regime == "unconstrained":
        return torch.sigmoid(logits)
    if regime == "top-k":
        if top_k is None:
            raise ValueError("top_k must be provided for the top-k regime")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if top_k >= logits.shape[0]:
            return torch.softmax(logits, dim=0)
        topk_indices = torch.topk(logits, k=top_k).indices
        dense_softmax = torch.softmax(logits, dim=0)
        hard_alpha = torch.zeros_like(logits)
        hard_alpha[topk_indices] = dense_softmax[topk_indices]
        hard_alpha = hard_alpha / hard_alpha.sum()
        return hard_alpha.detach() + (dense_softmax - dense_softmax.detach())
    raise ValueError(f"unsupported oracle-alpha regime {regime!r}")


def _initial_alpha_logits(
    *,
    num_sources: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    matched_zero_init: bool = False,
) -> torch.Tensor:
    if matched_zero_init:
        return torch.zeros(num_sources, device=device, dtype=dtype)
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


def _mean_pooled_token_embedding(
    *,
    model: HookedTransformer,
    prompt: str,
) -> torch.Tensor:
    tokens = model.to_tokens(prompt, prepend_bos=False)
    with torch.no_grad():
        embeddings = model.embed(tokens)[0]
    return embeddings.mean(dim=0)


def _prompt_shape_scalar_features(
    *,
    model: HookedTransformer,
    prompt: str,
) -> torch.Tensor:
    stripped_prompt = prompt.strip()
    words = stripped_prompt.split()
    token_count = int(model.to_tokens(prompt, prepend_bos=False).shape[-1])
    word_count = len(words)
    char_count = len(prompt)
    mean_word_length = (
        sum(len(word.strip(".,;:!?")) for word in words) / word_count
        if word_count
        else 0.0
    )
    features = (
        float(token_count),
        float(word_count),
        float(char_count),
        float(char_count / max(token_count, 1)),
        float(mean_word_length),
        float(prompt.count(",")),
        float(prompt.count("'")),
        float(prompt.endswith(" of")),
    )
    return torch.tensor(features, dtype=torch.float32)


def _position_thirds_mean_pooled(states: torch.Tensor) -> torch.Tensor:
    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError("states must have shape [pos, d_model] with pos > 0")

    summaries = []
    last_non_empty_summary = states[0]
    for chunk in torch.tensor_split(states, 3, dim=0):
        if chunk.shape[0] == 0:
            summaries.append(last_non_empty_summary)
            continue
        last_non_empty_summary = chunk.mean(dim=0)
        summaries.append(last_non_empty_summary)
    return torch.cat(tuple(summaries))


def _start_mid_end_summary(states: torch.Tensor) -> torch.Tensor:
    if states.ndim != 2 or states.shape[0] == 0:
        raise ValueError("states must have shape [pos, d_model] with pos > 0")

    final_index = states.shape[0] - 1
    positions = (0, final_index // 2, final_index)
    return torch.cat(tuple(states[index] for index in positions))


def _feature_vector_for_source(
    *,
    model: HookedTransformer,
    prompt: str,
    prepend_bos: bool | None,
    feature_source: str,
) -> list[float]:
    h_1: torch.Tensor | None = None
    h_4: torch.Tensor | None = None

    def resid_states() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal h_1, h_4
        if h_1 is None or h_4 is None:
            h_1, h_4 = _resid_post_states(
                model=model,
                prompt=prompt,
                prepend_bos=prepend_bos,
            )
        return h_1, h_4

    if feature_source == "mean_pooled_h_1[t]_resid_post_layer_0":
        h_1, _ = resid_states()
        feature = h_1.mean(dim=0)
    elif feature_source == "mean_pooled_h_4[t]_resid_post_layer_3":
        _, h_4 = resid_states()
        feature = h_4.mean(dim=0)
    elif (
        feature_source == "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat"
    ):
        _, h_4 = resid_states()
        feature = _position_thirds_mean_pooled(h_4)
    elif feature_source == "start_mid_end_h_4[t]_resid_post_layer_3_concat":
        _, h_4 = resid_states()
        feature = _start_mid_end_summary(h_4)
    elif feature_source == "prompt_shape_scalar_features_v1":
        feature = _prompt_shape_scalar_features(
            model=model,
            prompt=prompt,
        )
    elif feature_source == "mean_pooled_token_embedding":
        feature = _mean_pooled_token_embedding(
            model=model,
            prompt=prompt,
        )
    elif feature_source == "mean_pooled_h_1[t]_plus_h_4[t]_concat":
        h_1, h_4 = resid_states()
        feature = torch.cat((h_1.mean(dim=0), h_4.mean(dim=0)))
    elif feature_source == "final_token_h_1[t]_plus_h_4[t]_concat":
        h_1, h_4 = resid_states()
        feature = torch.cat((h_1[-1], h_4[-1]))
    elif (
        feature_source
        == "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_prompt_shape_scalar_features_v1_concat"
    ):
        _, h_4 = resid_states()
        feature = torch.cat(
            (
                _position_thirds_mean_pooled(h_4),
                _prompt_shape_scalar_features(
                    model=model,
                    prompt=prompt,
                ).to(dtype=h_4.dtype, device=h_4.device),
            )
        )
    elif (
        feature_source
        == "position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_plus_mean_pooled_token_embedding_concat"
    ):
        _, h_4 = resid_states()
        feature = torch.cat(
            (
                _position_thirds_mean_pooled(h_4),
                _mean_pooled_token_embedding(
                    model=model,
                    prompt=prompt,
                ).to(dtype=h_4.dtype, device=h_4.device),
            )
        )
    else:
        raise ValueError(f"unsupported feature source {feature_source!r}")
    return [float(value) for value in feature.detach().cpu().tolist()]


def collect_feature_vectors(
    *,
    model: HookedTransformer,
    prompt_entries: Sequence[PromptEntry],
    prepend_bos: bool | None,
    feature_source: str,
    existing_feature_vectors: Mapping[str, Sequence[float]] | None = None,
    on_feature_vector: Callable[[PromptEntry, list[float]], None] | None = None,
) -> list[list[float]]:
    cached_vectors = existing_feature_vectors or {}
    collected_vectors = []
    for entry in prompt_entries:
        cached = cached_vectors.get(entry.prompt_id)
        if cached is None:
            vector = _feature_vector_for_source(
                model=model,
                prompt=entry.text,
                prepend_bos=prepend_bos,
                feature_source=feature_source,
            )
            if on_feature_vector is not None:
                on_feature_vector(entry, vector)
        else:
            vector = [float(value) for value in cached]
        collected_vectors.append(vector)
    return collected_vectors


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
    regime: str = "softmax-constrained",
    top_k: int | None = None,
    matched_zero_init: bool = False,
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
        matched_zero_init=matched_zero_init,
    ).requires_grad_()
    optimizer = torch.optim.Adam([z], lr=learning_rate)
    best_loss = uniform_loss
    best_alpha = uniform_alpha.detach().clone()

    for _ in range(optimization_steps):
        optimizer.zero_grad()
        alpha = _alpha_from_regime_logits(z, regime=regime, top_k=top_k)
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

    final_alpha = _alpha_from_regime_logits(
        z.detach(),
        regime=regime,
        top_k=top_k,
    )

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
    regime: str = "softmax-constrained",
    top_k: int | None = None,
    matched_zero_init: bool = False,
    existing_sequence_results: Mapping[str, OracleAlphaSequenceResult] | None = None,
    on_sequence_result: Callable[[OracleAlphaSequenceResult], None] | None = None,
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

    cached_results = existing_sequence_results or {}
    sequence_results = []
    for index, entry in enumerate(entries):
        cached = cached_results.get(entry.prompt_id)
        if cached is None:
            result = _optimize_sequence(
                model=model,
                entry=entry,
                optimization_steps=optimization_steps,
                learning_rate=learning_rate,
                seed=seed + index,
                prepend_bos=prepend_bos,
                regime=regime,
                top_k=top_k,
                matched_zero_init=matched_zero_init,
            )
            if on_sequence_result is not None:
                on_sequence_result(result)
        else:
            if cached.prompt != entry.text or cached.split != entry.split:
                raise ValueError(
                    f"cached result for {entry.prompt_id!r} does not match prompt entry"
                )
            result = cached
        sequence_results.append(result)
    sequence_results = tuple(sequence_results)
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


def _base_prompt_id(prompt_id: str) -> str:
    return prompt_id.split(":", 1)[0]


def _prompt_matched_stability_metrics_for_runs(
    runs: Sequence[OracleAlphaRunSummary],
) -> OracleAlphaPromptMatchedStabilityMetrics:
    if not runs:
        raise ValueError("at least one run is required")

    per_run_distributions = [
        {
            _base_prompt_id(result.prompt_id): list(result.final_alpha)
            for result in run.sequence_results
        }
        for run in runs
    ]
    if len(per_run_distributions) == 1:
        return OracleAlphaPromptMatchedStabilityMetrics(
            matched_prompt_count=len(per_run_distributions[0]),
            mean_pairwise_js_divergence=0.0,
            mean_topk_jaccard_at_4=1.0,
            mean_top1_source_agreement=1.0,
        )

    js_values = []
    topk_values = []
    top1_values = []
    matched_prompt_ids: set[str] = set()
    for left_index in range(len(per_run_distributions)):
        for right_index in range(left_index + 1, len(per_run_distributions)):
            left_run = per_run_distributions[left_index]
            right_run = per_run_distributions[right_index]
            overlap = sorted(set(left_run) & set(right_run))
            matched_prompt_ids.update(overlap)
            for prompt_id in overlap:
                left_distribution = left_run[prompt_id]
                right_distribution = right_run[prompt_id]
                js_values.append(
                    jensen_shannon_divergence(left_distribution, right_distribution)
                )
                topk_values.append(
                    mean_topk_jaccard_similarity(
                        (left_distribution, right_distribution),
                        k=min(4, len(left_distribution)),
                    )
                )
                top1_values.append(
                    mean_top1_source_agreement(
                        (left_distribution, right_distribution),
                    )
                )

    if not matched_prompt_ids:
        raise ValueError("runs must share at least one prompt id")

    return OracleAlphaPromptMatchedStabilityMetrics(
        matched_prompt_count=len(matched_prompt_ids),
        mean_pairwise_js_divergence=0.0
        if not js_values
        else sum(js_values) / len(js_values),
        mean_topk_jaccard_at_4=1.0
        if not topk_values
        else sum(topk_values) / len(topk_values),
        mean_top1_source_agreement=1.0
        if not top1_values
        else sum(top1_values) / len(top1_values),
    )


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


def _shared_source_labels_for_runs(
    runs: Sequence[OracleAlphaRunSummary],
) -> tuple[str, ...]:
    if not runs:
        raise ValueError("at least one run is required")
    if not runs[0].sequence_results:
        raise ValueError("runs must contain at least one sequence result")
    reference = runs[0].sequence_results[0].source_labels
    for run in runs:
        for result in run.sequence_results:
            if result.source_labels != reference:
                raise ValueError(
                    "predictiveness runs must share the same source labels"
                )
    return reference


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
    restart_per_sequence_metrics = _prompt_matched_stability_metrics_for_runs(
        restart_runs
    )

    paraphrase_run: OracleAlphaRunSummary | None = None
    paraphrase_metrics: OracleAlphaStabilityMetrics | None = None
    paraphrase_per_sequence_metrics: OracleAlphaPromptMatchedStabilityMetrics | None = (
        None
    )
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
        paraphrase_per_sequence_metrics = _prompt_matched_stability_metrics_for_runs(
            (base_run, paraphrase_run)
        )

    resample_runs: tuple[OracleAlphaRunSummary, ...] = ()
    resample_metrics: OracleAlphaStabilityMetrics | None = None
    resample_per_sequence_metrics: OracleAlphaPromptMatchedStabilityMetrics | None = (
        None
    )
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
        resample_per_sequence_metrics = _prompt_matched_stability_metrics_for_runs(
            (base_run, *resample_runs)
        )

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
        restart_per_sequence_metrics=restart_per_sequence_metrics,
        paraphrase_run=paraphrase_run,
        paraphrase_metrics=paraphrase_metrics,
        paraphrase_per_sequence_metrics=paraphrase_per_sequence_metrics,
        resample_runs=resample_runs,
        resample_metrics=resample_metrics,
        resample_per_sequence_metrics=resample_per_sequence_metrics,
    )


def _tuned_ridge_regularization(
    *,
    model: HookedTransformer,
    train_entries: Sequence[PromptEntry],
    train_features: Sequence[Sequence[float]],
    train_targets: Sequence[Sequence[float]],
    train_uniform_losses: Sequence[float],
    target_name: str,
    source_labels: Sequence[str] | None,
    regularization_grid: Sequence[float],
    primary_metric: str,
    secondary_metric: str,
    prepend_bos: bool | None,
    on_regularization_summary: Callable[
        [OracleAlphaPredictivenessRegularizationCandidate],
        None,
    ]
    | None = None,
) -> tuple[
    float,
    PredictivenessSummary,
    dict[str, float],
    tuple[OracleAlphaPredictivenessRegularizationCandidate, ...],
]:
    if len(train_features) < 2:
        raise ValueError("at least two training examples are required")
    if not regularization_grid:
        raise ValueError("regularization_grid must not be empty")
    if len(train_entries) != len(train_features):
        raise ValueError("train_entries and train_features must match in length")
    if len(train_uniform_losses) != len(train_features):
        raise ValueError("train_uniform_losses and train_features must match in length")

    best_regularization: float | None = None
    best_summary: PredictivenessSummary | None = None
    best_metric_overrides: dict[str, float] | None = None
    regularization_summaries = []
    num_examples = len(train_features)
    for regularization_strength in regularization_grid:
        fold_predictions = []
        fold_targets = []
        fold_improvements = []
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
            fold_train_target_matrix = alpha_target_matrix(
                target_name=target_name,
                distributions=fold_train_targets,
                source_labels=source_labels,
            )
            predicted_target = ridge_regression_predictions(
                train_features=fold_train_features,
                train_targets=fold_train_target_matrix.tolist(),
                eval_features=[train_features[holdout_index]],
                regularization_strength=regularization_strength,
            )
            predicted_distribution = alpha_target_predictions_to_distributions(
                target_name=target_name,
                predictions=predicted_target.tolist(),
                source_labels=source_labels,
                train_distributions=fold_train_targets,
            )
            predicted_loss = _loss_for_predicted_alpha(
                model=model,
                entry=train_entries[holdout_index],
                predicted_alpha=predicted_distribution[0].tolist(),
                prepend_bos=prepend_bos,
            )
            fold_predictions.append(predicted_distribution[0].tolist())
            fold_targets.append(train_targets[holdout_index])
            fold_improvements.append(
                float(train_uniform_losses[holdout_index]) - predicted_loss
            )

        candidate_summary = predictiveness_summary_from_predictions(
            train_targets=train_targets,
            eval_targets=fold_targets,
            predictions=fold_predictions,
        )
        candidate_metric_overrides = {
            "mean_predicted_improvement_over_uniform": (
                sum(fold_improvements) / len(fold_improvements)
            )
        }
        regularization_summary = OracleAlphaPredictivenessRegularizationCandidate(
            regularization_strength=float(regularization_strength),
            tuning_primary_metric=primary_metric,
            tuning_primary_metric_value=predictiveness_metric_value(
                summary=candidate_summary,
                metric_name=primary_metric,
                metric_overrides=candidate_metric_overrides,
            ),
            tuning_secondary_metric=secondary_metric,
            tuning_secondary_metric_value=predictiveness_metric_value(
                summary=candidate_summary,
                metric_name=secondary_metric,
                metric_overrides=candidate_metric_overrides,
            ),
            tuning_r_squared=candidate_summary.r_squared,
            tuning_mean_js_divergence=candidate_summary.mean_js_divergence,
        )
        regularization_summaries.append(regularization_summary)
        if on_regularization_summary is not None:
            on_regularization_summary(regularization_summary)
        if best_summary is None:
            best_regularization = float(regularization_strength)
            best_summary = candidate_summary
            best_metric_overrides = candidate_metric_overrides
            continue

        primary_delta = compare_predictiveness_metric_values(
            metric_name=primary_metric,
            left=predictiveness_metric_value(
                summary=candidate_summary,
                metric_name=primary_metric,
                metric_overrides=candidate_metric_overrides,
            ),
            right=predictiveness_metric_value(
                summary=best_summary,
                metric_name=primary_metric,
                metric_overrides=best_metric_overrides,
            ),
        )
        secondary_delta = compare_predictiveness_metric_values(
            metric_name=secondary_metric,
            left=predictiveness_metric_value(
                summary=candidate_summary,
                metric_name=secondary_metric,
                metric_overrides=candidate_metric_overrides,
            ),
            right=predictiveness_metric_value(
                summary=best_summary,
                metric_name=secondary_metric,
                metric_overrides=best_metric_overrides,
            ),
        )
        if primary_delta > 1e-12 or (
            abs(primary_delta) <= 1e-12 and secondary_delta > 1e-12
        ):
            best_regularization = float(regularization_strength)
            best_summary = candidate_summary
            best_metric_overrides = candidate_metric_overrides

    return (
        best_regularization,
        best_summary,
        best_metric_overrides,
        tuple(regularization_summaries),
    )


def build_oracle_alpha_predictiveness_summary(
    *,
    model: HookedTransformer,
    collection_id: str,
    train_entries: Sequence[PromptEntry],
    eval_entries: Sequence[PromptEntry],
    train_run: OracleAlphaRunSummary,
    eval_run: OracleAlphaRunSummary,
    regularization_grid: Sequence[float],
    candidate_feature_sources: Sequence[str],
    candidate_target_names: Sequence[str],
    prepend_bos: bool | None = None,
    train_feature_vectors_by_source: Mapping[str, Sequence[Sequence[float]]]
    | None = None,
    eval_feature_vectors_by_source: Mapping[str, Sequence[Sequence[float]]]
    | None = None,
    on_candidate_regularization_evaluated: Callable[
        [
            str,
            str,
            OracleAlphaPredictivenessRegularizationCandidate,
        ],
        None,
    ]
    | None = None,
    on_candidate_completed: Callable[
        [OracleAlphaPredictivenessFeatureCandidate],
        None,
    ]
    | None = None,
) -> OracleAlphaPredictivenessSummary:
    control_registry = load_oracle_alpha_control_registry()
    control_plan = control_registry.plans[collection_id]
    predictiveness_plan = control_plan.predictiveness

    if not candidate_feature_sources:
        raise ValueError("candidate_feature_sources must not be empty")
    if not candidate_target_names:
        raise ValueError("candidate_target_names must not be empty")

    train_targets = [list(result.final_alpha) for result in train_run.sequence_results]
    eval_targets = [list(result.final_alpha) for result in eval_run.sequence_results]
    train_uniform_losses = [
        result.uniform_loss for result in train_run.sequence_results
    ]
    source_labels = _shared_source_labels_for_runs((train_run, eval_run))
    candidate_feature_summaries = []
    selected_feature_source = None
    selected_target_name = None
    selected_regularization_strength = None
    tuning_mean_js_divergence = None
    tuning_primary_metric_value = None
    tuning_secondary_metric_value = None
    selected_train_features = None
    train_feature_cache = train_feature_vectors_by_source or {}
    eval_feature_cache = eval_feature_vectors_by_source or {}

    for feature_source in candidate_feature_sources:
        cached_train_vectors = train_feature_cache.get(feature_source)
        if cached_train_vectors is None:
            cached_train_vectors = collect_feature_vectors(
                model=model,
                prompt_entries=train_entries,
                prepend_bos=prepend_bos,
                feature_source=feature_source,
            )
        train_features = [
            [float(value) for value in vector] for vector in cached_train_vectors
        ]
        for candidate_target_name in candidate_target_names:
            (
                candidate_regularization_strength,
                candidate_summary,
                candidate_metric_overrides,
                regularization_summaries,
            ) = _tuned_ridge_regularization(
                model=model,
                train_entries=train_entries,
                train_features=train_features,
                train_targets=train_targets,
                train_uniform_losses=train_uniform_losses,
                target_name=candidate_target_name,
                source_labels=source_labels,
                regularization_grid=regularization_grid,
                primary_metric=predictiveness_plan.primary_metric,
                secondary_metric=predictiveness_plan.secondary_metric,
                prepend_bos=prepend_bos,
                on_regularization_summary=(
                    None
                    if on_candidate_regularization_evaluated is None
                    else (
                        lambda regularization_summary,
                        feature_source=feature_source,
                        candidate_target_name=candidate_target_name: on_candidate_regularization_evaluated(
                            feature_source,
                            candidate_target_name,
                            regularization_summary,
                        )
                    )
                ),
            )
            candidate_tuning_primary_metric_value = predictiveness_metric_value(
                summary=candidate_summary,
                metric_name=predictiveness_plan.primary_metric,
                metric_overrides=candidate_metric_overrides,
            )
            candidate_tuning_secondary_metric_value = predictiveness_metric_value(
                summary=candidate_summary,
                metric_name=predictiveness_plan.secondary_metric,
                metric_overrides=candidate_metric_overrides,
            )
            candidate_feature_summary = OracleAlphaPredictivenessFeatureCandidate(
                target=candidate_target_name,
                feature_source=feature_source,
                selected_regularization_strength=candidate_regularization_strength,
                tuning_primary_metric=predictiveness_plan.primary_metric,
                tuning_primary_metric_value=candidate_tuning_primary_metric_value,
                tuning_secondary_metric=predictiveness_plan.secondary_metric,
                tuning_secondary_metric_value=candidate_tuning_secondary_metric_value,
                tuning_r_squared=candidate_summary.r_squared,
                tuning_mean_js_divergence=candidate_summary.mean_js_divergence,
                regularization_summaries=regularization_summaries,
            )
            candidate_feature_summaries.append(candidate_feature_summary)
            if on_candidate_completed is not None:
                on_candidate_completed(candidate_feature_summary)
            if tuning_primary_metric_value is None:
                selected_feature_source = feature_source
                selected_target_name = candidate_target_name
                selected_regularization_strength = candidate_regularization_strength
                tuning_primary_metric_value = candidate_tuning_primary_metric_value
                tuning_secondary_metric_value = candidate_tuning_secondary_metric_value
                tuning_mean_js_divergence = candidate_summary.mean_js_divergence
                selected_train_features = train_features
                continue

            primary_delta = compare_predictiveness_metric_values(
                metric_name=predictiveness_plan.primary_metric,
                left=candidate_tuning_primary_metric_value,
                right=tuning_primary_metric_value,
            )
            secondary_delta = compare_predictiveness_metric_values(
                metric_name=predictiveness_plan.secondary_metric,
                left=candidate_tuning_secondary_metric_value,
                right=tuning_secondary_metric_value,
            )
            if primary_delta > 1e-12 or (
                abs(primary_delta) <= 1e-12 and secondary_delta > 1e-12
            ):
                selected_feature_source = feature_source
                selected_target_name = candidate_target_name
                selected_regularization_strength = candidate_regularization_strength
                tuning_primary_metric_value = candidate_tuning_primary_metric_value
                tuning_secondary_metric_value = candidate_tuning_secondary_metric_value
                tuning_mean_js_divergence = candidate_summary.mean_js_divergence
                selected_train_features = train_features

    cached_eval_vectors = eval_feature_cache.get(selected_feature_source)
    if cached_eval_vectors is None:
        cached_eval_vectors = collect_feature_vectors(
            model=model,
            prompt_entries=eval_entries,
            prepend_bos=prepend_bos,
            feature_source=selected_feature_source,
        )
    eval_features = [
        [float(value) for value in vector] for vector in cached_eval_vectors
    ]
    selected_train_target_matrix = alpha_target_matrix(
        target_name=selected_target_name,
        distributions=train_targets,
        source_labels=source_labels,
    )
    predicted_eval_targets = ridge_regression_predictions(
        train_features=selected_train_features,
        train_targets=selected_train_target_matrix.tolist(),
        eval_features=eval_features,
        regularization_strength=selected_regularization_strength,
    )
    predicted_eval_alphas = alpha_target_predictions_to_distributions(
        target_name=selected_target_name,
        predictions=predicted_eval_targets.tolist(),
        source_labels=source_labels,
        train_distributions=train_targets,
    )
    predictiveness_summary = predictiveness_summary_from_predictions(
        train_targets=train_targets,
        eval_targets=eval_targets,
        predictions=predicted_eval_alphas.tolist(),
    )
    eval_predictions = []
    for index, result in enumerate(eval_run.sequence_results):
        normalized_predicted_alpha = predicted_eval_alphas[index].tolist()
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
    predicted_mean_improvement_over_uniform = sum(
        prediction.uniform_loss - prediction.predicted_loss
        for prediction in eval_predictions
    ) / len(eval_predictions)
    oracle_mean_improvement_over_uniform = sum(
        prediction.uniform_loss - prediction.oracle_loss
        for prediction in eval_predictions
    ) / len(eval_predictions)

    return OracleAlphaPredictivenessSummary(
        model_name=model.cfg.model_name,
        collection_id=collection_id,
        train_split=predictiveness_plan.train_split,
        eval_split=predictiveness_plan.eval_split,
        target=selected_target_name,
        control_plan_id=control_plan.plan_id,
        control_registry_id=control_registry.registry_id,
        feature_source=selected_feature_source,
        tuning_primary_metric=predictiveness_plan.primary_metric,
        tuning_secondary_metric=predictiveness_plan.secondary_metric,
        tuning_primary_metric_value=tuning_primary_metric_value,
        tuning_secondary_metric_value=tuning_secondary_metric_value,
        selected_regularization_strength=selected_regularization_strength,
        tuning_mean_js_divergence=tuning_mean_js_divergence,
        candidate_feature_summaries=tuple(candidate_feature_summaries),
        predictiveness_summary=predictiveness_summary,
        train_run=train_run,
        eval_run=eval_run,
        eval_predictions=tuple(eval_predictions),
        predicted_mean_improvement_over_uniform=predicted_mean_improvement_over_uniform,
        oracle_mean_improvement_over_uniform=oracle_mean_improvement_over_uniform,
        mib_status=control_plan.mib_anchor.mib_status,
        mib_rationale=control_plan.mib_anchor.rationale,
    )


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
    candidate_target_names: Sequence[str] | None = None,
    target_name_override: str | None = None,
    prepend_bos: bool | None = None,
) -> OracleAlphaPredictivenessSummary:
    control_registry = load_oracle_alpha_control_registry()
    control_plan = control_registry.plans[collection_id]
    predictiveness_plan = control_plan.predictiveness
    target_name = target_name_override or predictiveness_plan.target
    if predictiveness_plan.model_family != "ridge_regression":
        raise ValueError("predictiveness plan must use ridge_regression")
    if target_name_override is not None and candidate_target_names is not None:
        raise ValueError(
            "target_name_override and candidate_target_names are mutually exclusive"
        )

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
    target_names = tuple(candidate_target_names or (target_name,))
    if not target_names:
        raise ValueError("candidate_target_names must not be empty")

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
    return build_oracle_alpha_predictiveness_summary(
        model=model,
        collection_id=collection_id,
        train_entries=train_entries,
        eval_entries=eval_entries,
        train_run=train_run,
        eval_run=eval_run,
        regularization_grid=regularization_grid,
        candidate_feature_sources=feature_sources,
        candidate_target_names=target_names,
        prepend_bos=prepend_bos,
    )
