# ABOUTME: Implements a static block-influence baseline for OIH dynamic-versus-static comparisons.
# ABOUTME: Fits a pilot-only pruned block policy and evaluates it on held-out prompts with the oracle loss surface.

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from transformer_lens import HookedTransformer

from prompts import PromptEntry
from .oracle_alpha_controls import block_group_structure
from .oracle_alpha_runner import _fixed_residual_sources, _loss_for_alpha


@dataclass(frozen=True)
class StaticBlockInfluenceFitSummary:
    source_labels: tuple[str, ...]
    group_names: tuple[str, ...]
    mean_group_influences: tuple[float, ...]
    kept_group_indices: tuple[int, ...]
    pruned_group_indices: tuple[int, ...]
    static_alpha: tuple[float, ...]


@dataclass(frozen=True)
class StaticBlockInfluenceEvalSummary:
    num_prompts: int
    mean_uniform_loss: float
    mean_static_loss: float
    mean_improvement_over_uniform: float
    positive_improvement_count: int


def mean_alpha_policy_from_sequences(
    sequence_alphas: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    if not sequence_alphas:
        raise ValueError("sequence_alphas must not be empty")
    expected_dim = len(sequence_alphas[0])
    if expected_dim < 1:
        raise ValueError("alpha vectors must contain at least one source")

    accumulator = torch.zeros(expected_dim, dtype=torch.float64)
    for alpha in sequence_alphas:
        if len(alpha) != expected_dim:
            raise ValueError("all alpha vectors must share the same dimension")
        vector = torch.tensor(alpha, dtype=torch.float64)
        if torch.any(vector < 0):
            raise ValueError("alpha values must be non-negative")
        total = float(vector.sum().item())
        if total <= 0.0:
            raise ValueError("alpha vectors must sum to a positive value")
        accumulator += vector / total

    mean_policy = accumulator / float(len(sequence_alphas))
    mean_policy = mean_policy / float(mean_policy.sum().item())
    return tuple(float(value.item()) for value in mean_policy.to(dtype=torch.float32))


def select_best_static_evaluation(
    evaluations: Mapping[str, StaticBlockInfluenceEvalSummary],
) -> tuple[str, StaticBlockInfluenceEvalSummary]:
    if not evaluations:
        raise ValueError("evaluations must not be empty")
    return max(
        evaluations.items(),
        key=lambda item: (
            item[1].mean_improvement_over_uniform,
            item[1].positive_improvement_count,
            -item[1].mean_static_loss,
        ),
    )


def select_kept_groups_from_influences(
    *,
    mean_group_influences: Sequence[float],
    prune_fraction: float,
) -> tuple[int, ...]:
    if not 0.0 <= prune_fraction <= 1.0:
        raise ValueError("prune_fraction must be in [0, 1]")
    num_groups = len(mean_group_influences)
    if num_groups < 2:
        raise ValueError("at least two groups are required for pruning")

    num_to_prune = int(math.floor(prune_fraction * num_groups))
    if prune_fraction > 0.0:
        num_to_prune = max(1, num_to_prune)
    if num_to_prune >= num_groups:
        raise ValueError("pruning removed all groups")

    ranked = sorted(
        range(num_groups),
        key=lambda group_index: (
            float(mean_group_influences[group_index]),
            group_index,
        ),
    )
    pruned = set(ranked[:num_to_prune])
    kept = tuple(
        group_index for group_index in range(num_groups) if group_index not in pruned
    )
    if not kept:
        raise ValueError("pruning removed all groups")
    return kept


def build_static_alpha_from_group_influences(
    *,
    group_indices: torch.Tensor,
    mean_group_influences: Sequence[float],
    prune_fraction: float,
) -> torch.Tensor:
    if group_indices.ndim != 1:
        raise ValueError("group_indices must have shape [num_sources]")
    if group_indices.numel() == 0:
        raise ValueError("group_indices must not be empty")

    num_groups = int(group_indices.max().item()) + 1
    if len(mean_group_influences) != num_groups:
        raise ValueError("mean_group_influences must match number of groups")
    kept_group_indices = select_kept_groups_from_influences(
        mean_group_influences=mean_group_influences,
        prune_fraction=prune_fraction,
    )

    alpha = torch.zeros(
        group_indices.shape[0],
        dtype=torch.float32,
        device=group_indices.device,
    )
    group_mass = 1.0 / len(kept_group_indices)
    for group_index in kept_group_indices:
        source_positions = torch.nonzero(
            group_indices == group_index, as_tuple=False
        ).squeeze(-1)
        if source_positions.numel() == 0:
            raise ValueError(f"group {group_index} has no source positions")
        alpha[source_positions] = group_mass / float(source_positions.numel())
    return alpha


def fit_static_block_influence_baseline(
    *,
    model: HookedTransformer,
    prompt_entries: Sequence[PromptEntry],
    prune_fraction: float = 0.25,
    prepend_bos: bool | None = None,
) -> StaticBlockInfluenceFitSummary:
    if not prompt_entries:
        raise ValueError("prompt_entries must not be empty")

    expected_source_labels: tuple[str, ...] | None = None
    group_names: tuple[str, ...] | None = None
    group_indices_tensor: torch.Tensor | None = None
    total_group_influence: torch.Tensor | None = None

    for entry in prompt_entries:
        residual_stack, source_labels, tokens = _fixed_residual_sources(
            model=model,
            prompt=entry.text,
            prepend_bos=prepend_bos,
        )
        if expected_source_labels is None:
            expected_source_labels = source_labels
            group_names, group_indices = block_group_structure(source_labels)
            group_indices_tensor = torch.tensor(
                group_indices.tolist(),
                dtype=torch.long,
                device=residual_stack.device,
            )
            total_group_influence = torch.zeros(
                len(group_names),
                dtype=torch.float32,
                device=residual_stack.device,
            )
        elif source_labels != expected_source_labels:
            raise ValueError("all prompts must share identical source labels")

        assert group_indices_tensor is not None
        assert total_group_influence is not None
        num_sources = residual_stack.shape[0]
        uniform_alpha = torch.full(
            (num_sources,),
            1.0 / float(num_sources),
            dtype=torch.float32,
            device=residual_stack.device,
        )
        base_loss = _loss_for_alpha(
            model=model,
            residual_stack=residual_stack,
            tokens=tokens,
            alpha=uniform_alpha,
        )
        for group_index in range(total_group_influence.shape[0]):
            kept_mask = group_indices_tensor != group_index
            kept_count = int(kept_mask.sum().item())
            if kept_count <= 0:
                raise ValueError(
                    "each prompt must retain at least one source after pruning"
                )
            pruned_alpha = kept_mask.to(dtype=torch.float32) / float(kept_count)
            pruned_loss = _loss_for_alpha(
                model=model,
                residual_stack=residual_stack,
                tokens=tokens,
                alpha=pruned_alpha,
            )
            total_group_influence[group_index] += float(
                (pruned_loss - base_loss).item()
            )

    assert expected_source_labels is not None
    assert group_names is not None
    assert group_indices_tensor is not None
    assert total_group_influence is not None
    mean_group_influence = total_group_influence / float(len(prompt_entries))
    static_alpha = build_static_alpha_from_group_influences(
        group_indices=group_indices_tensor,
        mean_group_influences=tuple(
            float(value.item()) for value in mean_group_influence
        ),
        prune_fraction=prune_fraction,
    )
    kept_group_indices = select_kept_groups_from_influences(
        mean_group_influences=tuple(
            float(value.item()) for value in mean_group_influence
        ),
        prune_fraction=prune_fraction,
    )
    pruned_group_indices = tuple(
        index for index in range(len(group_names)) if index not in kept_group_indices
    )
    return StaticBlockInfluenceFitSummary(
        source_labels=expected_source_labels,
        group_names=group_names,
        mean_group_influences=tuple(
            float(value.item()) for value in mean_group_influence
        ),
        kept_group_indices=kept_group_indices,
        pruned_group_indices=pruned_group_indices,
        static_alpha=tuple(
            float(value.item()) for value in static_alpha.detach().cpu()
        ),
    )


def evaluate_static_block_influence_baseline(
    *,
    model: HookedTransformer,
    prompt_entries: Sequence[PromptEntry],
    fit_summary: StaticBlockInfluenceFitSummary,
    prepend_bos: bool | None = None,
) -> StaticBlockInfluenceEvalSummary:
    if not prompt_entries:
        raise ValueError("prompt_entries must not be empty")
    static_alpha_tensor = torch.tensor(fit_summary.static_alpha, dtype=torch.float32)
    uniform_losses: list[float] = []
    static_losses: list[float] = []

    for entry in prompt_entries:
        residual_stack, source_labels, tokens = _fixed_residual_sources(
            model=model,
            prompt=entry.text,
            prepend_bos=prepend_bos,
        )
        if source_labels != fit_summary.source_labels:
            raise ValueError("fit/eval source labels do not match")
        if residual_stack.shape[0] != static_alpha_tensor.shape[0]:
            raise ValueError("static alpha dimension does not match source count")

        static_alpha = static_alpha_tensor.to(device=residual_stack.device)
        num_sources = residual_stack.shape[0]
        uniform_alpha = torch.full(
            (num_sources,),
            1.0 / float(num_sources),
            dtype=torch.float32,
            device=residual_stack.device,
        )
        uniform_loss = _loss_for_alpha(
            model=model,
            residual_stack=residual_stack,
            tokens=tokens,
            alpha=uniform_alpha,
        )
        static_loss = _loss_for_alpha(
            model=model,
            residual_stack=residual_stack,
            tokens=tokens,
            alpha=static_alpha,
        )
        uniform_losses.append(float(uniform_loss.item()))
        static_losses.append(float(static_loss.item()))

    improvements = [
        uniform - static
        for uniform, static in zip(uniform_losses, static_losses, strict=True)
    ]
    mean_uniform_loss = sum(uniform_losses) / len(uniform_losses)
    mean_static_loss = sum(static_losses) / len(static_losses)
    return StaticBlockInfluenceEvalSummary(
        num_prompts=len(prompt_entries),
        mean_uniform_loss=mean_uniform_loss,
        mean_static_loss=mean_static_loss,
        mean_improvement_over_uniform=sum(improvements) / len(improvements),
        positive_improvement_count=sum(
            1 for improvement in improvements if improvement > 0.0
        ),
    )
