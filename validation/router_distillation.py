# ABOUTME: Fits the smallest honest Phase 6 router-distillation pilot on saved per-token exports.
# ABOUTME: Compares router inputs and target parameterizations on one held-out pilot split using an explicit sequence-aggregation rule.

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .oracle_alpha_controls import (
    PredictivenessSummary,
    alpha_target_matrix,
    alpha_target_predictions_to_distributions,
    compare_predictiveness_metric_values,
    predictiveness_summary_from_predictions,
)
from .oracle_alpha_runner import _fixed_residual_sources


ROOT = Path(__file__).resolve().parents[1]
READINESS_TARGET_R_SQUARED = 0.5
ALLOWED_INPUT_FIELDS = {"h_1[t]", "h_4[t]"}
ALLOWED_AGGREGATIONS = {
    "mean_token_logits_then_softmax",
    "last_token_logits_then_softmax",
}
ALLOWED_TARGET_NAMES = {"oracle_alpha_vector", "oracle_alpha_logit_vector"}
ALLOWED_ROUTER_FAMILIES = {"linear", "mlp"}
ALLOWED_SUPERVISION_OBJECTIVES = {
    "sequence_target_mse",
    "all_tokens_target_mse",
    "last_third_tokens_target_mse",
    "next_token_positions_sequence_target_mse",
    "next_token_positions_oracle_alpha_target_logit_contribution_mse",
}


@dataclass(frozen=True)
class RouterDistillationExample:
    prompt_id: str
    prompt: str
    split: str
    target_text: str | None
    tags: tuple[str, ...]
    perturbation_names: tuple[str, ...]
    token_ids: torch.Tensor
    h_1: torch.Tensor
    h_4: torch.Tensor
    source_labels: tuple[str, ...]
    final_alpha: torch.Tensor


@dataclass(frozen=True)
class RouterDistillationPilotDataset:
    collection_id: str
    model_name: str
    split: str
    exported_fields: tuple[str, ...]
    examples: tuple[RouterDistillationExample, ...]


@dataclass(frozen=True)
class RouterDistillationInputSummary:
    input_field: str
    target_name: str
    aggregation: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    best_epoch: int
    train_prompt_count: int
    eval_prompt_count: int
    train_loss: float
    eval_loss: float
    mean_predicted_entropy: float
    mean_oracle_entropy: float
    eval_summary: PredictivenessSummary
    router_family: str = "mlp"
    hidden_dim: int | None = None
    train_prompt_ids: tuple[str, ...] = ()
    eval_prompt_ids: tuple[str, ...] = ()
    supervision_objective: str = "sequence_target_mse"


@dataclass(frozen=True)
class RouterDistillationComparisonSummary:
    collection_id: str
    model_name: str
    split: str
    aggregation: str
    candidate_target_names: tuple[str, ...]
    selection_primary_metric: str
    selection_secondary_metric: str
    readiness_target_r_squared: float
    train_prompt_count: int
    eval_prompt_count: int
    selected_input_field: str
    selected_target_name: str
    target_changed_input_ranking: bool
    selected_meets_readiness_target: bool
    train_prompt_ids: tuple[str, ...]
    eval_prompt_ids: tuple[str, ...]
    input_summaries: tuple[RouterDistillationInputSummary, ...]

    @property
    def candidate_summaries(self) -> tuple[RouterDistillationInputSummary, ...]:
        return self.input_summaries


@dataclass(frozen=True)
class RouterDistillationAggregationComparisonSummary:
    collection_id: str
    model_name: str
    split: str
    fixed_target_name: str
    candidate_aggregations: tuple[str, ...]
    selection_primary_metric: str
    selection_secondary_metric: str
    readiness_target_r_squared: float
    train_prompt_count: int
    eval_prompt_count: int
    selected_input_field: str
    selected_aggregation: str
    aggregation_changed_input_ranking: bool
    selected_meets_readiness_target: bool
    train_prompt_ids: tuple[str, ...]
    eval_prompt_ids: tuple[str, ...]
    input_summaries: tuple[RouterDistillationInputSummary, ...]

    @property
    def candidate_summaries(self) -> tuple[RouterDistillationInputSummary, ...]:
        return self.input_summaries


@dataclass(frozen=True)
class RouterDistillationCapacityComparisonSummary:
    collection_id: str
    model_name: str
    split: str
    fixed_target_name: str
    fixed_aggregation: str
    candidate_hidden_dims: tuple[int, ...]
    selection_primary_metric: str
    selection_secondary_metric: str
    readiness_target_r_squared: float
    train_prompt_count: int
    eval_prompt_count: int
    selected_input_field: str
    selected_hidden_dim: int
    capacity_changed_input_ranking: bool
    selected_meets_readiness_target: bool
    train_prompt_ids: tuple[str, ...]
    eval_prompt_ids: tuple[str, ...]
    input_summaries: tuple[RouterDistillationInputSummary, ...]

    @property
    def candidate_summaries(self) -> tuple[RouterDistillationInputSummary, ...]:
        return self.input_summaries


@dataclass(frozen=True)
class RouterDistillationFamilyComparisonSummary:
    collection_id: str
    model_name: str
    split: str
    fixed_input_field: str
    fixed_target_name: str
    fixed_aggregation: str
    fixed_supervision_objective: str
    candidate_router_families: tuple[str, ...]
    hidden_dim: int
    selection_primary_metric: str
    selection_secondary_metric: str
    readiness_target_r_squared: float
    train_prompt_count: int
    eval_prompt_count: int
    selected_router_family: str
    selected_meets_readiness_target: bool
    train_prompt_ids: tuple[str, ...]
    eval_prompt_ids: tuple[str, ...]
    input_summaries: tuple[RouterDistillationInputSummary, ...]

    @property
    def candidate_summaries(self) -> tuple[RouterDistillationInputSummary, ...]:
        return self.input_summaries


@dataclass(frozen=True)
class RouterDistillationSupervisionComparisonSummary:
    collection_id: str
    model_name: str
    split: str
    fixed_input_field: str
    fixed_target_name: str
    fixed_aggregation: str
    fixed_router_family: str
    hidden_dim: int
    candidate_supervision_objectives: tuple[str, ...]
    selection_primary_metric: str
    selection_secondary_metric: str
    readiness_target_r_squared: float
    train_prompt_count: int
    eval_prompt_count: int
    selected_supervision_objective: str
    selected_meets_readiness_target: bool
    train_prompt_ids: tuple[str, ...]
    eval_prompt_ids: tuple[str, ...]
    input_summaries: tuple[RouterDistillationInputSummary, ...]

    @property
    def candidate_summaries(self) -> tuple[RouterDistillationInputSummary, ...]:
        return self.input_summaries


@dataclass(frozen=True)
class RouterDistillationPromptDiagnostic:
    prompt_id: str
    prompt: str
    tags: tuple[str, ...]
    num_tokens: int
    oracle_entropy: float
    oracle_top1_mass: float
    mean_js_divergence: float
    target_mse: float
    predicted_entropy: float = 0.0
    predicted_top1_mass: float = 0.0


@dataclass(frozen=True)
class RouterDistillationDiagnosticGroupSummary:
    group_key: str
    prompt_count: int
    mean_js_divergence: float
    mean_target_mse: float
    mean_num_tokens: float
    mean_oracle_entropy: float
    mean_oracle_top1_mass: float


@dataclass(frozen=True)
class RouterDistillationAttributeCorrelation:
    attribute_name: str
    pearson_r: float


@dataclass(frozen=True)
class RouterDistillationSupervisionGranularitySummary:
    prompt_count: int
    worst_stratum_tag: str
    stratum_mean_js_range: float
    strongest_attribute_name: str
    strongest_attribute_abs_correlation: float
    recommended_next_step: str
    rationale: str
    stratum_summaries: tuple[RouterDistillationDiagnosticGroupSummary, ...]
    attribute_correlations: tuple[RouterDistillationAttributeCorrelation, ...]


@dataclass(frozen=True)
class RouterDistillationSupervisionGranularityAudit:
    collection_id: str
    model_name: str
    split: str
    fixed_input_field: str
    fixed_target_name: str
    fixed_aggregation: str
    fixed_router_family: str
    hidden_dim: int | None
    train_prompt_count: int
    eval_prompt_count: int
    train_prompt_ids: tuple[str, ...]
    eval_prompt_ids: tuple[str, ...]
    baseline_eval_loss: float
    baseline_eval_summary: PredictivenessSummary
    prompt_diagnostics: tuple[RouterDistillationPromptDiagnostic, ...]
    supervision_summary: RouterDistillationSupervisionGranularitySummary


@dataclass
class _FittedRouterArtifacts:
    model: torch.nn.Module
    mean: torch.Tensor
    std: torch.Tensor
    train_target_lookup: dict[str, torch.Tensor]
    eval_target_lookup: dict[str, torch.Tensor]
    best_epoch: int


class _SequenceRouterMLP(torch.nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, num_sources: int) -> None:
        super().__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, num_sources)

    def forward_token_logits(self, token_states: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.input_layer(token_states))
        return self.output_layer(hidden)


class _SequenceRouterLinear(torch.nn.Module):
    def __init__(self, *, input_dim: int, num_sources: int) -> None:
        super().__init__()
        self.output_layer = torch.nn.Linear(input_dim, num_sources)

    def forward_token_logits(self, token_states: torch.Tensor) -> torch.Tensor:
        return self.output_layer(token_states)


def _resolve_checkpoint_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def _require_fields(payload: dict[str, object], *, checkpoint_path: Path) -> None:
    required_fields = {
        "prompt_id",
        "prompt",
        "split",
        "target_text",
        "tags",
        "perturbations",
        "token_ids",
        "h_1[t]",
        "h_4[t]",
        "source_labels",
        "final_alpha",
    }
    missing = sorted(required_fields.difference(payload.keys()))
    if missing:
        raise ValueError(
            f"router distillation checkpoint {checkpoint_path} is missing {missing}"
        )


def load_router_distillation_pilot_dataset(
    export_dir: Path,
) -> RouterDistillationPilotDataset:
    manifest_path = Path(export_dir) / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing dataset manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    examples: list[RouterDistillationExample] = []
    for entry in manifest["entries"]:
        checkpoint_path = _resolve_checkpoint_path(entry["checkpoint_path"])
        payload = torch.load(checkpoint_path, map_location="cpu")
        _require_fields(payload, checkpoint_path=checkpoint_path)
        examples.append(
            RouterDistillationExample(
                prompt_id=str(payload["prompt_id"]),
                prompt=str(payload["prompt"]),
                split=str(payload["split"]),
                target_text=payload["target_text"],
                tags=tuple(str(tag) for tag in payload["tags"]),
                perturbation_names=tuple(
                    sorted(str(name) for name in payload["perturbations"].keys())
                ),
                token_ids=payload["token_ids"].detach().cpu().to(dtype=torch.long),
                h_1=payload["h_1[t]"].detach().cpu().to(dtype=torch.float32),
                h_4=payload["h_4[t]"].detach().cpu().to(dtype=torch.float32),
                source_labels=tuple(str(label) for label in payload["source_labels"]),
                final_alpha=payload["final_alpha"]
                .detach()
                .cpu()
                .to(dtype=torch.float32),
            )
        )
    if not examples:
        raise ValueError(
            "router distillation dataset must contain at least one example"
        )
    return RouterDistillationPilotDataset(
        collection_id=str(manifest["collection_id"]),
        model_name=str(manifest["model_name"]),
        split=str(manifest["split"]),
        exported_fields=tuple(str(field) for field in manifest["dataset_fields"]),
        examples=tuple(examples),
    )


def aggregate_token_logits_to_sequence_alpha(
    *,
    token_logits: torch.Tensor,
    aggregation: str,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    sequence_logits = _aggregate_token_logits_to_sequence_logits(
        token_logits=token_logits,
        aggregation=aggregation,
        token_mask=token_mask,
    )
    return torch.softmax(sequence_logits, dim=-1)


def _aggregate_token_logits_to_sequence_logits(
    *,
    token_logits: torch.Tensor,
    aggregation: str,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if aggregation not in ALLOWED_AGGREGATIONS:
        raise ValueError(f"unsupported aggregation {aggregation!r}")
    if token_logits.ndim not in {2, 3}:
        raise ValueError("token_logits must be rank-2 or rank-3")

    squeeze_batch = token_logits.ndim == 2
    if squeeze_batch:
        token_logits = token_logits.unsqueeze(0)
    if token_mask is None:
        token_mask = torch.ones(
            token_logits.shape[:2],
            dtype=torch.bool,
            device=token_logits.device,
        )
    elif token_mask.ndim == 1:
        token_mask = token_mask.unsqueeze(0)
    if token_mask.shape != token_logits.shape[:2]:
        raise ValueError("token_mask must match token_logits over batch and sequence")

    mask = token_mask.to(dtype=token_logits.dtype).unsqueeze(-1)
    if aggregation == "mean_token_logits_then_softmax":
        counts = mask.sum(dim=1).clamp_min(1.0)
        sequence_logits = (token_logits * mask).sum(dim=1) / counts
    elif aggregation == "last_token_logits_then_softmax":
        valid_counts = token_mask.sum(dim=1)
        if torch.any(valid_counts <= 0):
            raise ValueError("each sequence must include at least one valid token")
        last_indices = (valid_counts - 1).to(device=token_logits.device)
        batch_indices = torch.arange(
            token_logits.shape[0],
            device=token_logits.device,
        )
        sequence_logits = token_logits[batch_indices, last_indices]
    else:
        raise ValueError(f"unsupported aggregation {aggregation!r}")
    if squeeze_batch:
        return sequence_logits.squeeze(0)
    return sequence_logits


def target_matrix_for_router_distillation(
    *,
    examples: Sequence[RouterDistillationExample],
    target_name: str,
) -> torch.Tensor:
    if target_name not in ALLOWED_TARGET_NAMES:
        raise ValueError(f"unsupported target name {target_name!r}")
    if not examples:
        raise ValueError("examples must not be empty")
    source_labels = tuple(examples[0].source_labels)
    target_matrix = alpha_target_matrix(
        target_name=target_name,
        distributions=[example.final_alpha.tolist() for example in examples],
        source_labels=source_labels,
    )
    return torch.tensor(target_matrix, dtype=torch.float32)


def _prediction_matrix_to_alpha(
    *,
    prediction_matrix: torch.Tensor,
    target_name: str,
    train_examples: Sequence[RouterDistillationExample],
    source_labels: Sequence[str],
) -> torch.Tensor:
    distributions = alpha_target_predictions_to_distributions(
        target_name=target_name,
        predictions=prediction_matrix.detach().cpu().tolist(),
        source_labels=source_labels,
        train_distributions=[
            example.final_alpha.tolist() for example in train_examples
        ],
    )
    return torch.tensor(
        distributions, dtype=torch.float32, device=prediction_matrix.device
    )


def _prediction_matrix_for_target(
    *,
    token_logits: torch.Tensor,
    aggregation: str,
    token_mask: torch.Tensor,
    target_name: str,
) -> torch.Tensor:
    sequence_logits = _aggregate_token_logits_to_sequence_logits(
        token_logits=token_logits,
        aggregation=aggregation,
        token_mask=token_mask,
    )
    if target_name == "oracle_alpha_vector":
        return torch.softmax(sequence_logits, dim=-1)
    if target_name == "oracle_alpha_logit_vector":
        return sequence_logits - sequence_logits.mean(dim=-1, keepdim=True)
    raise ValueError(f"unsupported target name {target_name!r}")


def _token_prediction_matrix_for_target(
    *,
    token_logits: torch.Tensor,
    target_name: str,
) -> torch.Tensor:
    if target_name == "oracle_alpha_vector":
        return torch.softmax(token_logits, dim=-1)
    if target_name == "oracle_alpha_logit_vector":
        return token_logits - token_logits.mean(dim=-1, keepdim=True)
    raise ValueError(f"unsupported target name {target_name!r}")


def tokenwise_oracle_alpha_target_logit_contribution_targets(
    *,
    residual_stack: torch.Tensor,
    final_alpha: torch.Tensor,
    token_ids: torch.Tensor,
    final_norm_weight: torch.Tensor,
    unembed: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual_stack.ndim != 3:
        raise ValueError("residual_stack must have shape [sources, pos, d_model]")
    if final_alpha.ndim != 1:
        raise ValueError("final_alpha must have shape [sources]")
    if token_ids.ndim != 1:
        raise ValueError("token_ids must have shape [pos]")
    if final_norm_weight.ndim != 1:
        raise ValueError("final_norm_weight must have shape [d_model]")
    if unembed.ndim != 2:
        raise ValueError("unembed must have shape [d_model, vocab]")
    if residual_stack.shape[0] != final_alpha.shape[0]:
        raise ValueError("residual_stack and final_alpha must share the source axis")
    if residual_stack.shape[1] != token_ids.shape[0]:
        raise ValueError("residual_stack and token_ids must share the position axis")
    if residual_stack.shape[2] != final_norm_weight.shape[0]:
        raise ValueError(
            "residual_stack and final_norm_weight must share the model dimension"
        )
    if unembed.shape[0] != final_norm_weight.shape[0]:
        raise ValueError("unembed and final_norm_weight must share the model dimension")

    source_weights = final_alpha[:, None, None]
    weighted_sources = residual_stack * source_weights
    mixture = weighted_sources.sum(dim=0)
    shared_scale = mixture.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    normalized_weighted_sources = (
        weighted_sources / shared_scale.unsqueeze(0)
    ) * final_norm_weight[None, None, :]
    next_token_ids = token_ids[1:]
    per_source_target_logits = torch.stack(
        [
            normalized_weighted_sources[:, position, :]
            @ unembed[:, int(target_token_id.item())]
            for position, target_token_id in enumerate(next_token_ids)
        ],
        dim=0,
    )
    centered_logits = per_source_target_logits - per_source_target_logits.mean(
        dim=-1,
        keepdim=True,
    )
    teacher_targets = torch.zeros(
        (token_ids.shape[0], residual_stack.shape[0]),
        dtype=residual_stack.dtype,
    )
    teacher_targets[:-1] = centered_logits.to(dtype=residual_stack.dtype)
    teacher_mask = torch.zeros(token_ids.shape[0], dtype=torch.bool)
    teacher_mask[:-1] = True
    return teacher_targets, teacher_mask


def _supervision_token_mask(
    *,
    token_mask: torch.Tensor,
    supervision_objective: str,
) -> torch.Tensor:
    if supervision_objective == "all_tokens_target_mse":
        return token_mask
    if supervision_objective in {
        "next_token_positions_sequence_target_mse",
        "next_token_positions_oracle_alpha_target_logit_contribution_mse",
    }:
        supervision_mask = token_mask.clone()
        valid_counts = token_mask.sum(dim=1)
        for batch_index, valid_count in enumerate(valid_counts.tolist()):
            if valid_count <= 0:
                continue
            supervision_mask[batch_index, int(valid_count) - 1] = False
        return supervision_mask
    if supervision_objective == "last_third_tokens_target_mse":
        supervision_mask = torch.zeros_like(token_mask)
        valid_counts = token_mask.sum(dim=1)
        for batch_index, valid_count in enumerate(valid_counts.tolist()):
            if valid_count <= 0:
                continue
            start_index = max(0, (2 * int(valid_count)) // 3)
            supervision_mask[batch_index, start_index : int(valid_count)] = True
        return supervision_mask
    raise ValueError(f"unsupported supervision objective {supervision_objective!r}")


def router_supervision_loss(
    *,
    token_logits: torch.Tensor,
    token_mask: torch.Tensor,
    targets: torch.Tensor,
    aggregation: str,
    target_name: str,
    supervision_objective: str,
    tokenwise_teacher_targets: torch.Tensor | None = None,
    tokenwise_teacher_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if supervision_objective == "sequence_target_mse":
        predicted_target = _prediction_matrix_for_target(
            token_logits=token_logits,
            token_mask=token_mask,
            aggregation=aggregation,
            target_name=target_name,
        )
        return F.mse_loss(predicted_target, targets)
    if supervision_objective not in ALLOWED_SUPERVISION_OBJECTIVES:
        raise ValueError(f"unsupported supervision objective {supervision_objective!r}")
    supervision_mask = _supervision_token_mask(
        token_mask=token_mask,
        supervision_objective=supervision_objective,
    )
    token_predictions = _token_prediction_matrix_for_target(
        token_logits=token_logits,
        target_name=target_name,
    )
    if (
        supervision_objective
        == "next_token_positions_oracle_alpha_target_logit_contribution_mse"
    ):
        if tokenwise_teacher_targets is None or tokenwise_teacher_mask is None:
            raise ValueError(
                "tokenwise teacher targets and mask are required for oracle-alpha "
                "tokenwise contribution supervision"
            )
        if tokenwise_teacher_targets.shape != token_predictions.shape:
            raise ValueError(
                "tokenwise_teacher_targets must match token_logits over batch, "
                "sequence, and source dimensions"
            )
        if tokenwise_teacher_mask.shape != token_mask.shape:
            raise ValueError("tokenwise_teacher_mask must match token_mask")
        broadcast_targets = tokenwise_teacher_targets
        supervision_mask = supervision_mask & tokenwise_teacher_mask
    else:
        broadcast_targets = targets.unsqueeze(1).expand_as(token_predictions)
    mask = supervision_mask.unsqueeze(-1).to(dtype=token_predictions.dtype)
    denominator = mask.sum() * token_predictions.shape[-1]
    if float(denominator.item()) <= 0.0:
        raise ValueError("supervision objective must include at least one token")
    return ((token_predictions - broadcast_targets) ** 2 * mask).sum() / denominator


def build_oracle_alpha_tokenwise_teacher_lookup(
    *,
    model,
    examples: Sequence[RouterDistillationExample],
    prompt_ids: Sequence[str],
    prepend_bos: bool | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    normalization_type = str(model.cfg.normalization_type).upper()
    if normalization_type not in {"RMS", "RMSPRE"}:
        raise ValueError(
            "oracle-alpha tokenwise contribution teachers currently require "
            "RMS-style final normalization"
        )
    if getattr(model.ln_final, "b", None) is not None:
        raise ValueError(
            "oracle-alpha tokenwise contribution teachers do not support a "
            "final-norm bias term"
        )
    selected_examples = _examples_for_prompt_ids(
        examples=examples,
        prompt_ids=prompt_ids,
    )
    unembed = model.unembed.W_U.detach().cpu().to(dtype=torch.float32)
    final_norm_weight_param = getattr(model.ln_final, "w", None)
    if final_norm_weight_param is None:
        final_norm_weight = torch.ones(unembed.shape[0], dtype=torch.float32)
    else:
        final_norm_weight = (
            final_norm_weight_param.detach().cpu().to(dtype=torch.float32)
        )
    lookup: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for example in selected_examples:
        residual_stack, source_labels, token_ids = _fixed_residual_sources(
            model=model,
            prompt=example.prompt,
            prepend_bos=prepend_bos,
        )
        residual_stack = residual_stack.detach().cpu().to(dtype=torch.float32)
        token_ids = token_ids.detach().cpu().to(dtype=torch.long)
        if tuple(source_labels) != tuple(example.source_labels):
            raise ValueError(
                f"source labels for {example.prompt_id!r} do not match the saved export"
            )
        if not torch.equal(token_ids, example.token_ids):
            raise ValueError(
                f"token ids for {example.prompt_id!r} do not match the saved export"
            )
        lookup[example.prompt_id] = (
            tokenwise_oracle_alpha_target_logit_contribution_targets(
                residual_stack=residual_stack,
                final_alpha=example.final_alpha,
                token_ids=token_ids,
                final_norm_weight=final_norm_weight,
                unembed=unembed,
                eps=float(model.cfg.eps),
            )
        )
    return lookup


def _collate_tokenwise_teacher_targets(
    *,
    examples: Sequence[RouterDistillationExample],
    tokenwise_teacher_lookup: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    max_tokens: int,
    target_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_targets = torch.zeros(
        (len(examples), max_tokens, target_dim),
        dtype=torch.float32,
        device=device,
    )
    teacher_mask = torch.zeros(
        (len(examples), max_tokens),
        dtype=torch.bool,
        device=device,
    )
    for batch_index, example in enumerate(examples):
        try:
            prompt_teacher_targets, prompt_teacher_mask = tokenwise_teacher_lookup[
                example.prompt_id
            ]
        except KeyError as error:
            raise ValueError(
                f"missing tokenwise teacher targets for prompt {example.prompt_id!r}"
            ) from error
        if prompt_teacher_targets.ndim != 2:
            raise ValueError("prompt_teacher_targets must have shape [pos, sources]")
        if prompt_teacher_mask.ndim != 1:
            raise ValueError("prompt_teacher_mask must have shape [pos]")
        if prompt_teacher_targets.shape[0] != example.token_ids.shape[0]:
            raise ValueError(
                f"tokenwise teacher targets for {example.prompt_id!r} must match "
                "the prompt token count"
            )
        if prompt_teacher_mask.shape[0] != example.token_ids.shape[0]:
            raise ValueError(
                f"tokenwise teacher mask for {example.prompt_id!r} must match the "
                "prompt token count"
            )
        if prompt_teacher_targets.shape[1] != target_dim:
            raise ValueError(
                f"tokenwise teacher targets for {example.prompt_id!r} must match "
                "the router target dimension"
            )
        num_tokens = int(prompt_teacher_targets.shape[0])
        teacher_targets[batch_index, :num_tokens] = prompt_teacher_targets.to(
            device=device,
            dtype=torch.float32,
        )
        teacher_mask[batch_index, :num_tokens] = prompt_teacher_mask.to(
            device=device,
            dtype=torch.bool,
        )
    return teacher_targets, teacher_mask


def _input_tensor_for_field(
    example: RouterDistillationExample,
    *,
    input_field: str,
) -> torch.Tensor:
    if input_field == "h_1[t]":
        return example.h_1
    if input_field == "h_4[t]":
        return example.h_4
    raise ValueError(f"unsupported input field {input_field!r}")


def _split_group_key(example: RouterDistillationExample) -> str:
    for tag in example.tags:
        if tag.startswith("subcategory_"):
            return tag
    for tag in example.tags:
        if tag.startswith("stratum_"):
            return tag
    return "all"


def _tag_with_prefix(tags: Sequence[str], *, prefix: str) -> str:
    for tag in tags:
        if tag.startswith(prefix):
            return tag
    return "missing"


def _distribution_entropy(distribution: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in distribution if value > 0.0)


def _js_divergence(
    left_distribution: Sequence[float],
    right_distribution: Sequence[float],
) -> float:
    if len(left_distribution) != len(right_distribution):
        raise ValueError("distributions must share the same dimension")
    midpoint = [
        0.5 * (left_value + right_value)
        for left_value, right_value in zip(
            left_distribution,
            right_distribution,
            strict=True,
        )
    ]

    def _kl_divergence(left: Sequence[float], right: Sequence[float]) -> float:
        value = 0.0
        for left_entry, right_entry in zip(left, right, strict=True):
            if left_entry <= 0.0:
                continue
            value += left_entry * math.log(left_entry / max(right_entry, 1e-12))
        return value

    return 0.5 * (
        _kl_divergence(left_distribution, midpoint)
        + _kl_divergence(right_distribution, midpoint)
    )


def _pearson_r(left_values: Sequence[float], right_values: Sequence[float]) -> float:
    if len(left_values) != len(right_values):
        raise ValueError("correlation inputs must share the same length")
    if len(left_values) < 2:
        return 0.0
    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)
    centered_left = [value - left_mean for value in left_values]
    centered_right = [value - right_mean for value in right_values]
    numerator = sum(
        left * right for left, right in zip(centered_left, centered_right, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in centered_left))
    right_norm = math.sqrt(sum(value * value for value in centered_right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return numerator / (left_norm * right_norm)


def summarize_router_supervision_granularity(
    *,
    prompt_diagnostics: Sequence[RouterDistillationPromptDiagnostic],
) -> RouterDistillationSupervisionGranularitySummary:
    if not prompt_diagnostics:
        raise ValueError("prompt_diagnostics must not be empty")

    grouped_by_stratum: dict[str, list[RouterDistillationPromptDiagnostic]] = {}
    for diagnostic in prompt_diagnostics:
        grouped_by_stratum.setdefault(
            _tag_with_prefix(diagnostic.tags, prefix="stratum_"),
            [],
        ).append(diagnostic)

    stratum_summaries = []
    for stratum_tag in sorted(grouped_by_stratum):
        diagnostics = grouped_by_stratum[stratum_tag]
        prompt_count = len(diagnostics)
        stratum_summaries.append(
            RouterDistillationDiagnosticGroupSummary(
                group_key=stratum_tag,
                prompt_count=prompt_count,
                mean_js_divergence=sum(item.mean_js_divergence for item in diagnostics)
                / prompt_count,
                mean_target_mse=sum(item.target_mse for item in diagnostics)
                / prompt_count,
                mean_num_tokens=sum(item.num_tokens for item in diagnostics)
                / prompt_count,
                mean_oracle_entropy=sum(item.oracle_entropy for item in diagnostics)
                / prompt_count,
                mean_oracle_top1_mass=sum(item.oracle_top1_mass for item in diagnostics)
                / prompt_count,
            )
        )
    worst_stratum_summary = max(
        stratum_summaries,
        key=lambda summary: summary.mean_js_divergence,
    )
    mean_js_values = [summary.mean_js_divergence for summary in stratum_summaries]
    stratum_mean_js_range = max(mean_js_values) - min(mean_js_values)

    per_prompt_js = [item.mean_js_divergence for item in prompt_diagnostics]
    attribute_correlations = tuple(
        RouterDistillationAttributeCorrelation(
            attribute_name=attribute_name,
            pearson_r=_pearson_r(attribute_values, per_prompt_js),
        )
        for attribute_name, attribute_values in (
            ("num_tokens", [float(item.num_tokens) for item in prompt_diagnostics]),
            (
                "oracle_entropy",
                [item.oracle_entropy for item in prompt_diagnostics],
            ),
            (
                "oracle_top1_mass",
                [item.oracle_top1_mass for item in prompt_diagnostics],
            ),
        )
    )
    strongest_attribute = max(
        attribute_correlations,
        key=lambda item: abs(item.pearson_r),
    )
    if stratum_mean_js_range >= 0.02:
        recommended_next_step = "richer_token_or_span_supervision"
        rationale = (
            "Held-out error varies materially across prompt strata on the frozen "
            "baseline, so supervision granularity is a cleaner next hypothesis "
            "than another width or family tweak."
        )
    else:
        recommended_next_step = "larger_training_eval_surface"
        rationale = (
            "Held-out error does not separate cleanly by stratum on the frozen "
            "baseline, so the next honest move is a larger training/eval surface "
            "before adding finer supervision."
        )

    return RouterDistillationSupervisionGranularitySummary(
        prompt_count=len(prompt_diagnostics),
        worst_stratum_tag=worst_stratum_summary.group_key,
        stratum_mean_js_range=stratum_mean_js_range,
        strongest_attribute_name=strongest_attribute.attribute_name,
        strongest_attribute_abs_correlation=abs(strongest_attribute.pearson_r),
        recommended_next_step=recommended_next_step,
        rationale=rationale,
        stratum_summaries=tuple(stratum_summaries),
        attribute_correlations=attribute_correlations,
    )


def stratified_router_train_eval_split(
    examples: Sequence[RouterDistillationExample],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[
    tuple[RouterDistillationExample, ...], tuple[RouterDistillationExample, ...]
]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must lie strictly between 0 and 1")
    if len(examples) < 2:
        raise ValueError("at least two examples are required")

    grouped: dict[str, list[RouterDistillationExample]] = {}
    for example in examples:
        grouped.setdefault(_split_group_key(example), []).append(example)

    rng = random.Random(seed)
    train_examples: list[RouterDistillationExample] = []
    eval_examples: list[RouterDistillationExample] = []
    for group_key in sorted(grouped):
        group_examples = list(grouped[group_key])
        rng.shuffle(group_examples)
        if len(group_examples) == 1:
            num_eval = 0
        else:
            num_eval = max(1, int(round(len(group_examples) * eval_fraction)))
            num_eval = min(num_eval, len(group_examples) - 1)
        eval_examples.extend(group_examples[:num_eval])
        train_examples.extend(group_examples[num_eval:])

    if not train_examples or not eval_examples:
        raise ValueError("split must produce non-empty train and eval sets")
    return tuple(train_examples), tuple(eval_examples)


def _standardization_stats(
    examples: Sequence[RouterDistillationExample],
    *,
    input_field: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.cat(
        [
            _input_tensor_for_field(example, input_field=input_field)
            for example in examples
        ],
        dim=0,
    )
    mean = states.mean(dim=0)
    std = states.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def _batched(
    examples: Sequence[RouterDistillationExample],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[list[RouterDistillationExample]]:
    shuffled = list(examples)
    random.Random(seed + epoch).shuffle(shuffled)
    return [
        shuffled[start : start + batch_size]
        for start in range(0, len(shuffled), batch_size)
    ]


def _collate_examples(
    examples: Sequence[RouterDistillationExample],
    *,
    input_field: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_lookup: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("batch must not be empty")
    max_tokens = max(int(example.token_ids.shape[0]) for example in examples)
    input_dim = int(mean.shape[0])
    target_dim = int(target_lookup[examples[0].prompt_id].shape[0])
    num_sources = int(examples[0].final_alpha.shape[0])
    states = torch.zeros(
        (len(examples), max_tokens, input_dim),
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros((len(examples), max_tokens), dtype=torch.bool, device=device)
    targets = torch.zeros(
        (len(examples), target_dim),
        dtype=torch.float32,
        device=device,
    )
    alpha_targets = torch.zeros(
        (len(examples), num_sources),
        dtype=torch.float32,
        device=device,
    )
    mean = mean.to(device=device)
    std = std.to(device=device)
    for batch_index, example in enumerate(examples):
        raw_states = _input_tensor_for_field(example, input_field=input_field)
        normalized_states = (raw_states.to(device=device) - mean) / std
        num_tokens = int(normalized_states.shape[0])
        states[batch_index, :num_tokens] = normalized_states
        mask[batch_index, :num_tokens] = True
        targets[batch_index] = target_lookup[example.prompt_id].to(device=device)
        alpha_targets[batch_index] = example.final_alpha.to(device=device)
    return states, mask, targets, alpha_targets


def _evaluate_router(
    *,
    model: _SequenceRouterMLP,
    examples: Sequence[RouterDistillationExample],
    train_examples: Sequence[RouterDistillationExample],
    input_field: str,
    target_name: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_lookup: dict[str, torch.Tensor],
    aggregation: str,
    batch_size: int,
    device: torch.device,
) -> tuple[float, PredictivenessSummary, float, float]:
    model.eval()
    losses: list[float] = []
    predictions: list[list[float]] = []
    targets: list[list[float]] = []
    predicted_entropies: list[float] = []
    oracle_entropies: list[float] = []
    source_labels = tuple(examples[0].source_labels)
    with torch.no_grad():
        for batch_start in range(0, len(examples), batch_size):
            batch_examples = examples[batch_start : batch_start + batch_size]
            states, mask, batch_targets, batch_alpha_targets = _collate_examples(
                batch_examples,
                input_field=input_field,
                mean=mean,
                std=std,
                target_lookup=target_lookup,
                device=device,
            )
            token_logits = model.forward_token_logits(states)
            batch_predictions = _prediction_matrix_for_target(
                token_logits=token_logits,
                token_mask=mask,
                aggregation=aggregation,
                target_name=target_name,
            )
            losses.append(float(F.mse_loss(batch_predictions, batch_targets).item()))
            batch_alpha_predictions = _prediction_matrix_to_alpha(
                prediction_matrix=batch_predictions,
                target_name=target_name,
                train_examples=train_examples,
                source_labels=source_labels,
            )
            predictions.extend(batch_alpha_predictions.cpu().tolist())
            targets.extend(batch_alpha_targets.cpu().tolist())
            for predicted_alpha, target_alpha in zip(
                batch_alpha_predictions.cpu().tolist(),
                batch_alpha_targets.cpu().tolist(),
                strict=True,
            ):
                predicted_entropies.append(
                    -sum(
                        value * math.log(value)
                        for value in predicted_alpha
                        if value > 0.0
                    )
                )
                oracle_entropies.append(
                    -sum(
                        value * math.log(value) for value in target_alpha if value > 0.0
                    )
                )

    return (
        sum(losses) / len(losses),
        predictiveness_summary_from_predictions(
            train_targets=targets,
            eval_targets=targets,
            predictions=predictions,
        ),
        sum(predicted_entropies) / len(predicted_entropies),
        sum(oracle_entropies) / len(oracle_entropies),
    )


def _fit_router_model(
    *,
    train_examples: Sequence[RouterDistillationExample],
    eval_examples: Sequence[RouterDistillationExample],
    input_field: str,
    target_name: str,
    aggregation: str,
    router_family: str,
    supervision_objective: str,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    tokenwise_teacher_lookup: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    | None = None,
    seed: int,
    device: torch.device,
) -> _FittedRouterArtifacts:
    mean, std = _standardization_stats(train_examples, input_field=input_field)
    train_target_matrix = target_matrix_for_router_distillation(
        examples=train_examples,
        target_name=target_name,
    )
    eval_target_matrix = target_matrix_for_router_distillation(
        examples=eval_examples,
        target_name=target_name,
    )
    train_target_lookup = {
        example.prompt_id: train_target_matrix[index]
        for index, example in enumerate(train_examples)
    }
    eval_target_lookup = {
        example.prompt_id: eval_target_matrix[index]
        for index, example in enumerate(eval_examples)
    }
    if router_family not in ALLOWED_ROUTER_FAMILIES:
        raise ValueError(f"unsupported router family {router_family!r}")
    if supervision_objective not in ALLOWED_SUPERVISION_OBJECTIVES:
        raise ValueError(f"unsupported supervision objective {supervision_objective!r}")
    torch.manual_seed(seed)
    if router_family == "mlp":
        model = _SequenceRouterMLP(
            input_dim=int(mean.shape[0]),
            hidden_dim=hidden_dim,
            num_sources=int(train_examples[0].final_alpha.shape[0]),
        ).to(device=device)
    elif router_family == "linear":
        model = _SequenceRouterLinear(
            input_dim=int(mean.shape[0]),
            num_sources=int(train_examples[0].final_alpha.shape[0]),
        ).to(device=device)
    else:
        raise ValueError(f"unsupported router family {router_family!r}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_epoch = 0
    best_eval_loss = float("inf")
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_examples in _batched(
            train_examples,
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
        ):
            states, mask, targets, _ = _collate_examples(
                batch_examples,
                input_field=input_field,
                mean=mean,
                std=std,
                target_lookup=train_target_lookup,
                device=device,
            )
            tokenwise_teacher_targets = None
            tokenwise_teacher_mask = None
            if tokenwise_teacher_lookup is not None:
                tokenwise_teacher_targets, tokenwise_teacher_mask = (
                    _collate_tokenwise_teacher_targets(
                        examples=batch_examples,
                        tokenwise_teacher_lookup=tokenwise_teacher_lookup,
                        max_tokens=int(states.shape[1]),
                        target_dim=int(targets.shape[1]),
                        device=device,
                    )
                )
            optimizer.zero_grad(set_to_none=True)
            token_logits = model.forward_token_logits(states)
            loss = router_supervision_loss(
                token_logits=token_logits,
                token_mask=mask,
                targets=targets,
                aggregation=aggregation,
                target_name=target_name,
                supervision_objective=supervision_objective,
                tokenwise_teacher_targets=tokenwise_teacher_targets,
                tokenwise_teacher_mask=tokenwise_teacher_mask,
            )
            loss.backward()
            optimizer.step()

        eval_loss, _, _, _ = _evaluate_router(
            model=model,
            examples=eval_examples,
            train_examples=train_examples,
            input_field=input_field,
            target_name=target_name,
            mean=mean,
            std=std,
            target_lookup=eval_target_lookup,
            aggregation=aggregation,
            batch_size=batch_size,
            device=device,
        )
        if eval_loss < best_eval_loss - 1e-8:
            best_eval_loss = eval_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return _FittedRouterArtifacts(
        model=model,
        mean=mean,
        std=std,
        train_target_lookup=train_target_lookup,
        eval_target_lookup=eval_target_lookup,
        best_epoch=best_epoch,
    )


def _collect_prompt_diagnostics(
    *,
    model: torch.nn.Module,
    examples: Sequence[RouterDistillationExample],
    train_examples: Sequence[RouterDistillationExample],
    input_field: str,
    target_name: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_lookup: dict[str, torch.Tensor],
    aggregation: str,
    device: torch.device,
) -> tuple[RouterDistillationPromptDiagnostic, ...]:
    if not examples:
        raise ValueError("examples must not be empty")
    source_labels = tuple(examples[0].source_labels)
    diagnostics = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            states, mask, targets, alpha_targets = _collate_examples(
                (example,),
                input_field=input_field,
                mean=mean,
                std=std,
                target_lookup=target_lookup,
                device=device,
            )
            token_logits = model.forward_token_logits(states)
            prediction_matrix = _prediction_matrix_for_target(
                token_logits=token_logits,
                token_mask=mask,
                aggregation=aggregation,
                target_name=target_name,
            )
            alpha_predictions = _prediction_matrix_to_alpha(
                prediction_matrix=prediction_matrix,
                target_name=target_name,
                train_examples=train_examples,
                source_labels=source_labels,
            )
            predicted_alpha = alpha_predictions[0].detach().cpu().tolist()
            oracle_alpha = alpha_targets[0].detach().cpu().tolist()
            diagnostics.append(
                RouterDistillationPromptDiagnostic(
                    prompt_id=example.prompt_id,
                    prompt=example.prompt,
                    tags=example.tags,
                    num_tokens=int(example.token_ids.shape[0]),
                    oracle_entropy=_distribution_entropy(oracle_alpha),
                    oracle_top1_mass=max(oracle_alpha),
                    mean_js_divergence=_js_divergence(predicted_alpha, oracle_alpha),
                    target_mse=float(F.mse_loss(prediction_matrix, targets).item()),
                    predicted_entropy=_distribution_entropy(predicted_alpha),
                    predicted_top1_mass=max(predicted_alpha),
                )
            )
    return tuple(diagnostics)


def _fit_router_for_input(
    *,
    train_examples: Sequence[RouterDistillationExample],
    eval_examples: Sequence[RouterDistillationExample],
    input_field: str,
    target_name: str,
    aggregation: str,
    router_family: str = "mlp",
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    supervision_objective: str = "sequence_target_mse",
    tokenwise_teacher_lookup: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    | None = None,
    seed: int,
    device: torch.device,
) -> RouterDistillationInputSummary:
    fitted = _fit_router_model(
        train_examples=train_examples,
        eval_examples=eval_examples,
        input_field=input_field,
        target_name=target_name,
        aggregation=aggregation,
        router_family=router_family,
        supervision_objective=supervision_objective,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        tokenwise_teacher_lookup=tokenwise_teacher_lookup,
        seed=seed,
        device=device,
    )
    train_loss, _, _, _ = _evaluate_router(
        model=fitted.model,
        examples=train_examples,
        train_examples=train_examples,
        input_field=input_field,
        target_name=target_name,
        mean=fitted.mean,
        std=fitted.std,
        target_lookup=fitted.train_target_lookup,
        aggregation=aggregation,
        batch_size=batch_size,
        device=device,
    )
    (
        eval_loss,
        eval_summary,
        mean_predicted_entropy,
        mean_oracle_entropy,
    ) = _evaluate_router(
        model=fitted.model,
        examples=eval_examples,
        train_examples=train_examples,
        input_field=input_field,
        target_name=target_name,
        mean=fitted.mean,
        std=fitted.std,
        target_lookup=fitted.eval_target_lookup,
        aggregation=aggregation,
        batch_size=batch_size,
        device=device,
    )
    return RouterDistillationInputSummary(
        input_field=input_field,
        target_name=target_name,
        aggregation=aggregation,
        supervision_objective=supervision_objective,
        router_family=router_family,
        hidden_dim=hidden_dim if router_family == "mlp" else None,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        best_epoch=fitted.best_epoch,
        train_prompt_count=len(train_examples),
        eval_prompt_count=len(eval_examples),
        train_loss=train_loss,
        eval_loss=eval_loss,
        mean_predicted_entropy=mean_predicted_entropy,
        mean_oracle_entropy=mean_oracle_entropy,
        eval_summary=eval_summary,
        train_prompt_ids=tuple(example.prompt_id for example in train_examples),
        eval_prompt_ids=tuple(example.prompt_id for example in eval_examples),
    )


def _examples_for_prompt_ids(
    *,
    examples: Sequence[RouterDistillationExample],
    prompt_ids: Sequence[str],
) -> tuple[RouterDistillationExample, ...]:
    example_lookup = {example.prompt_id: example for example in examples}
    missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in example_lookup]
    if missing:
        raise ValueError(f"missing prompt ids in dataset: {missing}")
    return tuple(example_lookup[prompt_id] for prompt_id in prompt_ids)


def audit_router_supervision_granularity(
    *,
    examples: Sequence[RouterDistillationExample],
    fixed_input_field: str,
    fixed_target_name: str,
    fixed_aggregation: str,
    fixed_router_family: str,
    hidden_dim: int,
    train_prompt_ids: Sequence[str],
    eval_prompt_ids: Sequence[str],
    learning_rate: float,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 40,
    supervision_objective: str = "sequence_target_mse",
    seed: int = 11,
    device: str = "cpu",
    collection_id: str = "unknown",
    model_name: str = "unknown",
) -> RouterDistillationSupervisionGranularityAudit:
    if fixed_input_field not in ALLOWED_INPUT_FIELDS:
        raise ValueError(f"unsupported input field {fixed_input_field!r}")
    if fixed_target_name not in ALLOWED_TARGET_NAMES:
        raise ValueError(f"unsupported target name {fixed_target_name!r}")
    if fixed_aggregation not in ALLOWED_AGGREGATIONS:
        raise ValueError(f"unsupported aggregation {fixed_aggregation!r}")
    if fixed_router_family not in ALLOWED_ROUTER_FAMILIES:
        raise ValueError(f"unsupported router family {fixed_router_family!r}")
    if supervision_objective not in ALLOWED_SUPERVISION_OBJECTIVES:
        raise ValueError(f"unsupported supervision objective {supervision_objective!r}")

    train_examples = _examples_for_prompt_ids(
        examples=examples,
        prompt_ids=train_prompt_ids,
    )
    eval_examples = _examples_for_prompt_ids(
        examples=examples,
        prompt_ids=eval_prompt_ids,
    )
    torch_device = torch.device(device)
    fitted = _fit_router_model(
        train_examples=train_examples,
        eval_examples=eval_examples,
        input_field=fixed_input_field,
        target_name=fixed_target_name,
        aggregation=fixed_aggregation,
        router_family=fixed_router_family,
        supervision_objective=supervision_objective,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        device=torch_device,
    )
    eval_loss, eval_summary, _, _ = _evaluate_router(
        model=fitted.model,
        examples=eval_examples,
        train_examples=train_examples,
        input_field=fixed_input_field,
        target_name=fixed_target_name,
        mean=fitted.mean,
        std=fitted.std,
        target_lookup=fitted.eval_target_lookup,
        aggregation=fixed_aggregation,
        batch_size=batch_size,
        device=torch_device,
    )
    prompt_diagnostics = _collect_prompt_diagnostics(
        model=fitted.model,
        examples=eval_examples,
        train_examples=train_examples,
        input_field=fixed_input_field,
        target_name=fixed_target_name,
        mean=fitted.mean,
        std=fitted.std,
        target_lookup=fitted.eval_target_lookup,
        aggregation=fixed_aggregation,
        device=torch_device,
    )
    supervision_summary = summarize_router_supervision_granularity(
        prompt_diagnostics=prompt_diagnostics
    )
    return RouterDistillationSupervisionGranularityAudit(
        collection_id=collection_id,
        model_name=model_name,
        split="pilot",
        fixed_input_field=fixed_input_field,
        fixed_target_name=fixed_target_name,
        fixed_aggregation=fixed_aggregation,
        fixed_router_family=fixed_router_family,
        hidden_dim=hidden_dim if fixed_router_family == "mlp" else None,
        train_prompt_count=len(train_examples),
        eval_prompt_count=len(eval_examples),
        train_prompt_ids=tuple(train_prompt_ids),
        eval_prompt_ids=tuple(eval_prompt_ids),
        baseline_eval_loss=eval_loss,
        baseline_eval_summary=eval_summary,
        prompt_diagnostics=prompt_diagnostics,
        supervision_summary=supervision_summary,
    )


def compare_router_supervision_objectives(
    *,
    examples: Sequence[RouterDistillationExample],
    fixed_input_field: str,
    fixed_target_name: str,
    fixed_aggregation: str,
    fixed_router_family: str,
    hidden_dim: int,
    train_prompt_ids: Sequence[str],
    eval_prompt_ids: Sequence[str],
    candidate_supervision_objectives: Sequence[str],
    learning_rate: float,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 40,
    seed: int = 11,
    device: str = "cpu",
    tokenwise_teacher_lookup: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    | None = None,
    collection_id: str = "unknown",
    model_name: str = "unknown",
) -> RouterDistillationSupervisionComparisonSummary:
    if fixed_input_field not in ALLOWED_INPUT_FIELDS:
        raise ValueError(f"unsupported input field {fixed_input_field!r}")
    if fixed_target_name not in ALLOWED_TARGET_NAMES:
        raise ValueError(f"unsupported target name {fixed_target_name!r}")
    if fixed_aggregation not in ALLOWED_AGGREGATIONS:
        raise ValueError(f"unsupported aggregation {fixed_aggregation!r}")
    if fixed_router_family not in ALLOWED_ROUTER_FAMILIES:
        raise ValueError(f"unsupported router family {fixed_router_family!r}")
    if not candidate_supervision_objectives:
        raise ValueError("candidate_supervision_objectives must not be empty")
    for supervision_objective in candidate_supervision_objectives:
        if supervision_objective not in ALLOWED_SUPERVISION_OBJECTIVES:
            raise ValueError(
                f"unsupported supervision objective {supervision_objective!r}"
            )

    train_examples = _examples_for_prompt_ids(
        examples=examples,
        prompt_ids=train_prompt_ids,
    )
    eval_examples = _examples_for_prompt_ids(
        examples=examples,
        prompt_ids=eval_prompt_ids,
    )
    torch_device = torch.device(device)

    input_summaries: list[RouterDistillationInputSummary] = []
    selected_summary: RouterDistillationInputSummary | None = None
    for supervision_objective in candidate_supervision_objectives:
        summary = _fit_router_for_input(
            train_examples=train_examples,
            eval_examples=eval_examples,
            input_field=fixed_input_field,
            target_name=fixed_target_name,
            aggregation=fixed_aggregation,
            router_family=fixed_router_family,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            supervision_objective=supervision_objective,
            tokenwise_teacher_lookup=tokenwise_teacher_lookup,
            seed=seed,
            device=torch_device,
        )
        input_summaries.append(summary)
        if selected_summary is None:
            selected_summary = summary
            continue
        primary_delta = compare_predictiveness_metric_values(
            metric_name="r_squared",
            left=summary.eval_summary.r_squared,
            right=selected_summary.eval_summary.r_squared,
        )
        if primary_delta > 1e-12:
            selected_summary = summary
            continue
        if abs(primary_delta) <= 1e-12:
            secondary_delta = compare_predictiveness_metric_values(
                metric_name="mean_js_divergence",
                left=summary.eval_summary.mean_js_divergence,
                right=selected_summary.eval_summary.mean_js_divergence,
            )
            if secondary_delta > 1e-12:
                selected_summary = summary

    assert selected_summary is not None
    return RouterDistillationSupervisionComparisonSummary(
        collection_id=collection_id,
        model_name=model_name,
        split="pilot",
        fixed_input_field=fixed_input_field,
        fixed_target_name=fixed_target_name,
        fixed_aggregation=fixed_aggregation,
        fixed_router_family=fixed_router_family,
        hidden_dim=hidden_dim,
        candidate_supervision_objectives=tuple(candidate_supervision_objectives),
        selection_primary_metric="r_squared",
        selection_secondary_metric="mean_js_divergence",
        readiness_target_r_squared=READINESS_TARGET_R_SQUARED,
        train_prompt_count=len(train_examples),
        eval_prompt_count=len(eval_examples),
        selected_supervision_objective=selected_summary.supervision_objective,
        selected_meets_readiness_target=(
            selected_summary.eval_summary.r_squared >= READINESS_TARGET_R_SQUARED
        ),
        train_prompt_ids=tuple(train_prompt_ids),
        eval_prompt_ids=tuple(eval_prompt_ids),
        input_summaries=tuple(input_summaries),
    )


def compare_router_input_sources(
    *,
    examples: Sequence[RouterDistillationExample],
    candidate_input_fields: Sequence[str],
    candidate_target_names: Sequence[str] = ("oracle_alpha_vector",),
    aggregation: str,
    eval_fraction: float,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 40,
    seed: int = 11,
    device: str = "cpu",
    collection_id: str = "unknown",
    model_name: str = "unknown",
) -> RouterDistillationComparisonSummary:
    return compare_router_target_parameterizations(
        examples=examples,
        candidate_input_fields=candidate_input_fields,
        candidate_target_names=candidate_target_names,
        aggregation=aggregation,
        eval_fraction=eval_fraction,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        device=device,
        collection_id=collection_id,
        model_name=model_name,
    )


def compare_router_target_parameterizations(
    *,
    examples: Sequence[RouterDistillationExample],
    candidate_input_fields: Sequence[str],
    candidate_target_names: Sequence[str],
    aggregation: str,
    eval_fraction: float,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 40,
    seed: int = 11,
    device: str = "cpu",
    collection_id: str = "unknown",
    model_name: str = "unknown",
) -> RouterDistillationComparisonSummary:
    if aggregation not in ALLOWED_AGGREGATIONS:
        raise ValueError(f"unsupported aggregation {aggregation!r}")
    if not candidate_input_fields:
        raise ValueError("candidate_input_fields must not be empty")
    if not candidate_target_names:
        raise ValueError("candidate_target_names must not be empty")
    for input_field in candidate_input_fields:
        if input_field not in ALLOWED_INPUT_FIELDS:
            raise ValueError(f"unsupported input field {input_field!r}")
    for target_name in candidate_target_names:
        if target_name not in ALLOWED_TARGET_NAMES:
            raise ValueError(f"unsupported target name {target_name!r}")

    train_examples, eval_examples = stratified_router_train_eval_split(
        examples,
        eval_fraction=eval_fraction,
        seed=seed,
    )
    torch_device = torch.device(device)

    input_summaries = []
    selected_summary: RouterDistillationInputSummary | None = None
    best_input_by_target: dict[str, str] = {}
    for target_name in candidate_target_names:
        best_for_target: RouterDistillationInputSummary | None = None
        for input_field in candidate_input_fields:
            summary = _fit_router_for_input(
                train_examples=train_examples,
                eval_examples=eval_examples,
                input_field=input_field,
                target_name=target_name,
                aggregation=aggregation,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed,
                device=torch_device,
            )
            input_summaries.append(summary)
            if selected_summary is None:
                selected_summary = summary
            else:
                primary_delta = compare_predictiveness_metric_values(
                    metric_name="r_squared",
                    left=summary.eval_summary.r_squared,
                    right=selected_summary.eval_summary.r_squared,
                )
                if primary_delta > 1e-12:
                    selected_summary = summary
                elif abs(primary_delta) <= 1e-12:
                    secondary_delta = compare_predictiveness_metric_values(
                        metric_name="mean_js_divergence",
                        left=summary.eval_summary.mean_js_divergence,
                        right=selected_summary.eval_summary.mean_js_divergence,
                    )
                    if secondary_delta > 1e-12:
                        selected_summary = summary

            if best_for_target is None:
                best_for_target = summary
                continue
            primary_delta = compare_predictiveness_metric_values(
                metric_name="r_squared",
                left=summary.eval_summary.r_squared,
                right=best_for_target.eval_summary.r_squared,
            )
            if primary_delta > 1e-12:
                best_for_target = summary
                continue
            if abs(primary_delta) <= 1e-12:
                secondary_delta = compare_predictiveness_metric_values(
                    metric_name="mean_js_divergence",
                    left=summary.eval_summary.mean_js_divergence,
                    right=best_for_target.eval_summary.mean_js_divergence,
                )
                if secondary_delta > 1e-12:
                    best_for_target = summary
        assert best_for_target is not None
        best_input_by_target[target_name] = best_for_target.input_field

    assert selected_summary is not None
    return RouterDistillationComparisonSummary(
        collection_id=collection_id,
        model_name=model_name,
        split="pilot",
        aggregation=aggregation,
        candidate_target_names=tuple(candidate_target_names),
        selection_primary_metric="r_squared",
        selection_secondary_metric="mean_js_divergence",
        readiness_target_r_squared=READINESS_TARGET_R_SQUARED,
        train_prompt_count=len(train_examples),
        eval_prompt_count=len(eval_examples),
        selected_input_field=selected_summary.input_field,
        selected_target_name=selected_summary.target_name,
        target_changed_input_ranking=len(set(best_input_by_target.values())) > 1,
        selected_meets_readiness_target=(
            selected_summary.eval_summary.r_squared >= READINESS_TARGET_R_SQUARED
        ),
        train_prompt_ids=tuple(example.prompt_id for example in train_examples),
        eval_prompt_ids=tuple(example.prompt_id for example in eval_examples),
        input_summaries=tuple(input_summaries),
    )


def compare_router_aggregations(
    *,
    examples: Sequence[RouterDistillationExample],
    candidate_input_fields: Sequence[str],
    fixed_target_name: str,
    candidate_aggregations: Sequence[str],
    eval_fraction: float,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 40,
    seed: int = 11,
    device: str = "cpu",
    collection_id: str = "unknown",
    model_name: str = "unknown",
) -> RouterDistillationAggregationComparisonSummary:
    if fixed_target_name not in ALLOWED_TARGET_NAMES:
        raise ValueError(f"unsupported target name {fixed_target_name!r}")
    if not candidate_input_fields:
        raise ValueError("candidate_input_fields must not be empty")
    if not candidate_aggregations:
        raise ValueError("candidate_aggregations must not be empty")
    for input_field in candidate_input_fields:
        if input_field not in ALLOWED_INPUT_FIELDS:
            raise ValueError(f"unsupported input field {input_field!r}")
    for aggregation in candidate_aggregations:
        if aggregation not in ALLOWED_AGGREGATIONS:
            raise ValueError(f"unsupported aggregation {aggregation!r}")

    train_examples, eval_examples = stratified_router_train_eval_split(
        examples,
        eval_fraction=eval_fraction,
        seed=seed,
    )
    torch_device = torch.device(device)

    input_summaries: list[RouterDistillationInputSummary] = []
    selected_summary: RouterDistillationInputSummary | None = None
    best_input_by_aggregation: dict[str, str] = {}
    for aggregation in candidate_aggregations:
        best_for_aggregation: RouterDistillationInputSummary | None = None
        for input_field in candidate_input_fields:
            summary = _fit_router_for_input(
                train_examples=train_examples,
                eval_examples=eval_examples,
                input_field=input_field,
                target_name=fixed_target_name,
                aggregation=aggregation,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed,
                device=torch_device,
            )
            input_summaries.append(summary)
            if selected_summary is None:
                selected_summary = summary
            else:
                primary_delta = compare_predictiveness_metric_values(
                    metric_name="r_squared",
                    left=summary.eval_summary.r_squared,
                    right=selected_summary.eval_summary.r_squared,
                )
                if primary_delta > 1e-12:
                    selected_summary = summary
                elif abs(primary_delta) <= 1e-12:
                    secondary_delta = compare_predictiveness_metric_values(
                        metric_name="mean_js_divergence",
                        left=summary.eval_summary.mean_js_divergence,
                        right=selected_summary.eval_summary.mean_js_divergence,
                    )
                    if secondary_delta > 1e-12:
                        selected_summary = summary

            if best_for_aggregation is None:
                best_for_aggregation = summary
                continue
            primary_delta = compare_predictiveness_metric_values(
                metric_name="r_squared",
                left=summary.eval_summary.r_squared,
                right=best_for_aggregation.eval_summary.r_squared,
            )
            if primary_delta > 1e-12:
                best_for_aggregation = summary
                continue
            if abs(primary_delta) <= 1e-12:
                secondary_delta = compare_predictiveness_metric_values(
                    metric_name="mean_js_divergence",
                    left=summary.eval_summary.mean_js_divergence,
                    right=best_for_aggregation.eval_summary.mean_js_divergence,
                )
                if secondary_delta > 1e-12:
                    best_for_aggregation = summary
        assert best_for_aggregation is not None
        best_input_by_aggregation[aggregation] = best_for_aggregation.input_field

    assert selected_summary is not None
    return RouterDistillationAggregationComparisonSummary(
        collection_id=collection_id,
        model_name=model_name,
        split="pilot",
        fixed_target_name=fixed_target_name,
        candidate_aggregations=tuple(candidate_aggregations),
        selection_primary_metric="r_squared",
        selection_secondary_metric="mean_js_divergence",
        readiness_target_r_squared=READINESS_TARGET_R_SQUARED,
        train_prompt_count=len(train_examples),
        eval_prompt_count=len(eval_examples),
        selected_input_field=selected_summary.input_field,
        selected_aggregation=selected_summary.aggregation,
        aggregation_changed_input_ranking=len(set(best_input_by_aggregation.values()))
        > 1,
        selected_meets_readiness_target=(
            selected_summary.eval_summary.r_squared >= READINESS_TARGET_R_SQUARED
        ),
        train_prompt_ids=tuple(example.prompt_id for example in train_examples),
        eval_prompt_ids=tuple(example.prompt_id for example in eval_examples),
        input_summaries=tuple(input_summaries),
    )


def compare_router_capacities(
    *,
    examples: Sequence[RouterDistillationExample],
    candidate_input_fields: Sequence[str],
    fixed_target_name: str,
    fixed_aggregation: str,
    candidate_hidden_dims: Sequence[int],
    eval_fraction: float,
    learning_rate: float,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 40,
    seed: int = 11,
    device: str = "cpu",
    collection_id: str = "unknown",
    model_name: str = "unknown",
) -> RouterDistillationCapacityComparisonSummary:
    if fixed_target_name not in ALLOWED_TARGET_NAMES:
        raise ValueError(f"unsupported target name {fixed_target_name!r}")
    if fixed_aggregation not in ALLOWED_AGGREGATIONS:
        raise ValueError(f"unsupported aggregation {fixed_aggregation!r}")
    if not candidate_input_fields:
        raise ValueError("candidate_input_fields must not be empty")
    if not candidate_hidden_dims:
        raise ValueError("candidate_hidden_dims must not be empty")
    for input_field in candidate_input_fields:
        if input_field not in ALLOWED_INPUT_FIELDS:
            raise ValueError(f"unsupported input field {input_field!r}")
    for hidden_dim in candidate_hidden_dims:
        if hidden_dim < 1:
            raise ValueError("candidate_hidden_dims must be positive")

    train_examples, eval_examples = stratified_router_train_eval_split(
        examples,
        eval_fraction=eval_fraction,
        seed=seed,
    )
    torch_device = torch.device(device)

    input_summaries: list[RouterDistillationInputSummary] = []
    selected_summary: RouterDistillationInputSummary | None = None
    best_input_by_hidden_dim: dict[int, str] = {}
    for hidden_dim in candidate_hidden_dims:
        best_for_hidden_dim: RouterDistillationInputSummary | None = None
        for input_field in candidate_input_fields:
            summary = _fit_router_for_input(
                train_examples=train_examples,
                eval_examples=eval_examples,
                input_field=input_field,
                target_name=fixed_target_name,
                aggregation=fixed_aggregation,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed,
                device=torch_device,
            )
            input_summaries.append(summary)
            if selected_summary is None:
                selected_summary = summary
            else:
                primary_delta = compare_predictiveness_metric_values(
                    metric_name="r_squared",
                    left=summary.eval_summary.r_squared,
                    right=selected_summary.eval_summary.r_squared,
                )
                if primary_delta > 1e-12:
                    selected_summary = summary
                elif abs(primary_delta) <= 1e-12:
                    secondary_delta = compare_predictiveness_metric_values(
                        metric_name="mean_js_divergence",
                        left=summary.eval_summary.mean_js_divergence,
                        right=selected_summary.eval_summary.mean_js_divergence,
                    )
                    if secondary_delta > 1e-12:
                        selected_summary = summary

            if best_for_hidden_dim is None:
                best_for_hidden_dim = summary
                continue
            primary_delta = compare_predictiveness_metric_values(
                metric_name="r_squared",
                left=summary.eval_summary.r_squared,
                right=best_for_hidden_dim.eval_summary.r_squared,
            )
            if primary_delta > 1e-12:
                best_for_hidden_dim = summary
                continue
            if abs(primary_delta) <= 1e-12:
                secondary_delta = compare_predictiveness_metric_values(
                    metric_name="mean_js_divergence",
                    left=summary.eval_summary.mean_js_divergence,
                    right=best_for_hidden_dim.eval_summary.mean_js_divergence,
                )
                if secondary_delta > 1e-12:
                    best_for_hidden_dim = summary
        assert best_for_hidden_dim is not None
        best_input_by_hidden_dim[int(hidden_dim)] = best_for_hidden_dim.input_field

    assert selected_summary is not None
    return RouterDistillationCapacityComparisonSummary(
        collection_id=collection_id,
        model_name=model_name,
        split="pilot",
        fixed_target_name=fixed_target_name,
        fixed_aggregation=fixed_aggregation,
        candidate_hidden_dims=tuple(int(value) for value in candidate_hidden_dims),
        selection_primary_metric="r_squared",
        selection_secondary_metric="mean_js_divergence",
        readiness_target_r_squared=READINESS_TARGET_R_SQUARED,
        train_prompt_count=len(train_examples),
        eval_prompt_count=len(eval_examples),
        selected_input_field=selected_summary.input_field,
        selected_hidden_dim=selected_summary.hidden_dim,
        capacity_changed_input_ranking=len(set(best_input_by_hidden_dim.values())) > 1,
        selected_meets_readiness_target=(
            selected_summary.eval_summary.r_squared >= READINESS_TARGET_R_SQUARED
        ),
        train_prompt_ids=tuple(example.prompt_id for example in train_examples),
        eval_prompt_ids=tuple(example.prompt_id for example in eval_examples),
        input_summaries=tuple(input_summaries),
    )


def compare_router_families(
    *,
    examples: Sequence[RouterDistillationExample],
    fixed_input_field: str,
    fixed_target_name: str,
    fixed_aggregation: str,
    candidate_router_families: Sequence[str],
    hidden_dim: int,
    eval_fraction: float,
    train_prompt_ids: Sequence[str] | None = None,
    eval_prompt_ids: Sequence[str] | None = None,
    learning_rate: float,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 40,
    supervision_objective: str = "sequence_target_mse",
    seed: int = 11,
    device: str = "cpu",
    collection_id: str = "unknown",
    model_name: str = "unknown",
) -> RouterDistillationFamilyComparisonSummary:
    if fixed_input_field not in ALLOWED_INPUT_FIELDS:
        raise ValueError(f"unsupported input field {fixed_input_field!r}")
    if fixed_target_name not in ALLOWED_TARGET_NAMES:
        raise ValueError(f"unsupported target name {fixed_target_name!r}")
    if fixed_aggregation not in ALLOWED_AGGREGATIONS:
        raise ValueError(f"unsupported aggregation {fixed_aggregation!r}")
    if not candidate_router_families:
        raise ValueError("candidate_router_families must not be empty")
    for router_family in candidate_router_families:
        if router_family not in ALLOWED_ROUTER_FAMILIES:
            raise ValueError(f"unsupported router family {router_family!r}")
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be positive")
    if supervision_objective not in ALLOWED_SUPERVISION_OBJECTIVES:
        raise ValueError(f"unsupported supervision objective {supervision_objective!r}")
    if (train_prompt_ids is None) != (eval_prompt_ids is None):
        raise ValueError(
            "train_prompt_ids and eval_prompt_ids must either both be provided "
            "or both be omitted"
        )
    if train_prompt_ids is None:
        train_examples, eval_examples = stratified_router_train_eval_split(
            examples,
            eval_fraction=eval_fraction,
            seed=seed,
        )
    else:
        train_examples = _examples_for_prompt_ids(
            examples=examples,
            prompt_ids=train_prompt_ids,
        )
        eval_examples = _examples_for_prompt_ids(
            examples=examples,
            prompt_ids=eval_prompt_ids or (),
        )
    torch_device = torch.device(device)

    input_summaries: list[RouterDistillationInputSummary] = []
    selected_summary: RouterDistillationInputSummary | None = None
    for router_family in candidate_router_families:
        summary = _fit_router_for_input(
            train_examples=train_examples,
            eval_examples=eval_examples,
            input_field=fixed_input_field,
            target_name=fixed_target_name,
            aggregation=fixed_aggregation,
            router_family=router_family,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            supervision_objective=supervision_objective,
            seed=seed,
            device=torch_device,
        )
        input_summaries.append(summary)
        if selected_summary is None:
            selected_summary = summary
            continue
        primary_delta = compare_predictiveness_metric_values(
            metric_name="r_squared",
            left=summary.eval_summary.r_squared,
            right=selected_summary.eval_summary.r_squared,
        )
        if primary_delta > 1e-12:
            selected_summary = summary
            continue
        if abs(primary_delta) <= 1e-12:
            secondary_delta = compare_predictiveness_metric_values(
                metric_name="mean_js_divergence",
                left=summary.eval_summary.mean_js_divergence,
                right=selected_summary.eval_summary.mean_js_divergence,
            )
            if secondary_delta > 1e-12:
                selected_summary = summary

    assert selected_summary is not None
    return RouterDistillationFamilyComparisonSummary(
        collection_id=collection_id,
        model_name=model_name,
        split="pilot",
        fixed_input_field=fixed_input_field,
        fixed_target_name=fixed_target_name,
        fixed_aggregation=fixed_aggregation,
        fixed_supervision_objective=supervision_objective,
        candidate_router_families=tuple(candidate_router_families),
        hidden_dim=hidden_dim,
        selection_primary_metric="r_squared",
        selection_secondary_metric="mean_js_divergence",
        readiness_target_r_squared=READINESS_TARGET_R_SQUARED,
        train_prompt_count=len(train_examples),
        eval_prompt_count=len(eval_examples),
        selected_router_family=selected_summary.router_family,
        selected_meets_readiness_target=(
            selected_summary.eval_summary.r_squared >= READINESS_TARGET_R_SQUARED
        ),
        train_prompt_ids=tuple(example.prompt_id for example in train_examples),
        eval_prompt_ids=tuple(example.prompt_id for example in eval_examples),
        input_summaries=tuple(input_summaries),
    )


def run_router_distillation_pilot_comparison(
    *,
    export_dir: Path,
    candidate_input_fields: Sequence[str],
    candidate_target_names: Sequence[str] = ("oracle_alpha_vector",),
    aggregation: str,
    eval_fraction: float,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 40,
    seed: int = 11,
    device: str = "cpu",
) -> RouterDistillationComparisonSummary:
    dataset = load_router_distillation_pilot_dataset(export_dir)
    return compare_router_input_sources(
        examples=dataset.examples,
        candidate_input_fields=candidate_input_fields,
        candidate_target_names=candidate_target_names,
        aggregation=aggregation,
        eval_fraction=eval_fraction,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        device=device,
        collection_id=dataset.collection_id,
        model_name=dataset.model_name,
    )


def compact_router_distillation_summary_payload(
    summary: RouterDistillationComparisonSummary,
) -> dict[str, object]:
    payload = asdict(summary)
    payload["input_summaries"] = [
        {
            "input_field": input_summary.input_field,
            "target_name": input_summary.target_name,
            "aggregation": input_summary.aggregation,
            "supervision_objective": input_summary.supervision_objective,
            "router_family": input_summary.router_family,
            "hidden_dim": input_summary.hidden_dim,
            "learning_rate": input_summary.learning_rate,
            "weight_decay": input_summary.weight_decay,
            "batch_size": input_summary.batch_size,
            "max_epochs": input_summary.max_epochs,
            "patience": input_summary.patience,
            "best_epoch": input_summary.best_epoch,
            "train_prompt_count": input_summary.train_prompt_count,
            "eval_prompt_count": input_summary.eval_prompt_count,
            "train_loss": input_summary.train_loss,
            "eval_loss": input_summary.eval_loss,
            "mean_predicted_entropy": input_summary.mean_predicted_entropy,
            "mean_oracle_entropy": input_summary.mean_oracle_entropy,
            "eval_summary": asdict(input_summary.eval_summary),
        }
        for input_summary in summary.input_summaries
    ]
    return payload


def compact_router_distillation_aggregation_summary_payload(
    summary: RouterDistillationAggregationComparisonSummary,
) -> dict[str, object]:
    payload = asdict(summary)
    payload["input_summaries"] = [
        {
            "input_field": input_summary.input_field,
            "target_name": input_summary.target_name,
            "aggregation": input_summary.aggregation,
            "supervision_objective": input_summary.supervision_objective,
            "router_family": input_summary.router_family,
            "hidden_dim": input_summary.hidden_dim,
            "learning_rate": input_summary.learning_rate,
            "weight_decay": input_summary.weight_decay,
            "batch_size": input_summary.batch_size,
            "max_epochs": input_summary.max_epochs,
            "patience": input_summary.patience,
            "best_epoch": input_summary.best_epoch,
            "train_prompt_count": input_summary.train_prompt_count,
            "eval_prompt_count": input_summary.eval_prompt_count,
            "train_loss": input_summary.train_loss,
            "eval_loss": input_summary.eval_loss,
            "mean_predicted_entropy": input_summary.mean_predicted_entropy,
            "mean_oracle_entropy": input_summary.mean_oracle_entropy,
            "eval_summary": asdict(input_summary.eval_summary),
        }
        for input_summary in summary.input_summaries
    ]
    return payload


def compact_router_distillation_capacity_summary_payload(
    summary: RouterDistillationCapacityComparisonSummary,
) -> dict[str, object]:
    payload = asdict(summary)
    payload["input_summaries"] = [
        {
            "input_field": input_summary.input_field,
            "target_name": input_summary.target_name,
            "aggregation": input_summary.aggregation,
            "supervision_objective": input_summary.supervision_objective,
            "router_family": input_summary.router_family,
            "hidden_dim": input_summary.hidden_dim,
            "learning_rate": input_summary.learning_rate,
            "weight_decay": input_summary.weight_decay,
            "batch_size": input_summary.batch_size,
            "max_epochs": input_summary.max_epochs,
            "patience": input_summary.patience,
            "best_epoch": input_summary.best_epoch,
            "train_prompt_count": input_summary.train_prompt_count,
            "eval_prompt_count": input_summary.eval_prompt_count,
            "train_loss": input_summary.train_loss,
            "eval_loss": input_summary.eval_loss,
            "mean_predicted_entropy": input_summary.mean_predicted_entropy,
            "mean_oracle_entropy": input_summary.mean_oracle_entropy,
            "eval_summary": asdict(input_summary.eval_summary),
        }
        for input_summary in summary.input_summaries
    ]
    return payload


def compact_router_distillation_family_summary_payload(
    summary: RouterDistillationFamilyComparisonSummary,
) -> dict[str, object]:
    payload = asdict(summary)
    payload["input_summaries"] = [
        {
            "input_field": input_summary.input_field,
            "target_name": input_summary.target_name,
            "aggregation": input_summary.aggregation,
            "supervision_objective": input_summary.supervision_objective,
            "router_family": input_summary.router_family,
            "hidden_dim": input_summary.hidden_dim,
            "learning_rate": input_summary.learning_rate,
            "weight_decay": input_summary.weight_decay,
            "batch_size": input_summary.batch_size,
            "max_epochs": input_summary.max_epochs,
            "patience": input_summary.patience,
            "best_epoch": input_summary.best_epoch,
            "train_prompt_count": input_summary.train_prompt_count,
            "eval_prompt_count": input_summary.eval_prompt_count,
            "train_loss": input_summary.train_loss,
            "eval_loss": input_summary.eval_loss,
            "mean_predicted_entropy": input_summary.mean_predicted_entropy,
            "mean_oracle_entropy": input_summary.mean_oracle_entropy,
            "eval_summary": asdict(input_summary.eval_summary),
        }
        for input_summary in summary.input_summaries
    ]
    return payload


def compact_router_distillation_supervision_granularity_payload(
    audit: RouterDistillationSupervisionGranularityAudit,
) -> dict[str, object]:
    payload = asdict(audit)
    payload["baseline_eval_summary"] = asdict(audit.baseline_eval_summary)
    payload["prompt_diagnostics"] = [
        asdict(prompt_diagnostic) for prompt_diagnostic in audit.prompt_diagnostics
    ]
    payload["supervision_summary"] = {
        "prompt_count": audit.supervision_summary.prompt_count,
        "worst_stratum_tag": audit.supervision_summary.worst_stratum_tag,
        "stratum_mean_js_range": audit.supervision_summary.stratum_mean_js_range,
        "strongest_attribute_name": audit.supervision_summary.strongest_attribute_name,
        "strongest_attribute_abs_correlation": (
            audit.supervision_summary.strongest_attribute_abs_correlation
        ),
        "recommended_next_step": audit.supervision_summary.recommended_next_step,
        "rationale": audit.supervision_summary.rationale,
        "stratum_summaries": [
            asdict(group_summary)
            for group_summary in audit.supervision_summary.stratum_summaries
        ],
        "attribute_correlations": [
            asdict(correlation)
            for correlation in audit.supervision_summary.attribute_correlations
        ],
    }
    return payload


def compact_router_distillation_supervision_comparison_payload(
    summary: RouterDistillationSupervisionComparisonSummary,
) -> dict[str, object]:
    payload = asdict(summary)
    payload["input_summaries"] = [
        {
            "input_field": input_summary.input_field,
            "target_name": input_summary.target_name,
            "aggregation": input_summary.aggregation,
            "supervision_objective": input_summary.supervision_objective,
            "router_family": input_summary.router_family,
            "hidden_dim": input_summary.hidden_dim,
            "learning_rate": input_summary.learning_rate,
            "weight_decay": input_summary.weight_decay,
            "batch_size": input_summary.batch_size,
            "max_epochs": input_summary.max_epochs,
            "patience": input_summary.patience,
            "best_epoch": input_summary.best_epoch,
            "train_prompt_count": input_summary.train_prompt_count,
            "eval_prompt_count": input_summary.eval_prompt_count,
            "train_loss": input_summary.train_loss,
            "eval_loss": input_summary.eval_loss,
            "mean_predicted_entropy": input_summary.mean_predicted_entropy,
            "mean_oracle_entropy": input_summary.mean_oracle_entropy,
            "eval_summary": asdict(input_summary.eval_summary),
        }
        for input_summary in summary.input_summaries
    ]
    return payload
