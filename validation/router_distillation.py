# ABOUTME: Fits the smallest honest Phase 6 router-distillation pilot on saved per-token exports.
# ABOUTME: Compares router inputs and target parameterizations on one held-out pilot split using an explicit sequence-aggregation rule.

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Sequence

import torch
import torch.nn.functional as F

from .oracle_alpha_controls import (
    PredictivenessSummary,
    alpha_target_matrix,
    alpha_target_predictions_to_distributions,
    compare_predictiveness_metric_values,
    predictiveness_summary_from_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
READINESS_TARGET_R_SQUARED = 0.5
ALLOWED_INPUT_FIELDS = {"h_1[t]", "h_4[t]"}
ALLOWED_AGGREGATIONS = {
    "mean_token_logits_then_softmax",
    "last_token_logits_then_softmax",
}
ALLOWED_TARGET_NAMES = {"oracle_alpha_vector", "oracle_alpha_logit_vector"}
ALLOWED_ROUTER_FAMILIES = {"linear", "mlp"}


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
    seed: int,
    device: torch.device,
) -> RouterDistillationInputSummary:
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
        epoch_losses: list[float] = []
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
            optimizer.zero_grad(set_to_none=True)
            token_logits = model.forward_token_logits(states)
            predicted_target = _prediction_matrix_for_target(
                token_logits=token_logits,
                token_mask=mask,
                aggregation=aggregation,
                target_name=target_name,
            )
            loss = F.mse_loss(predicted_target, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

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
    train_loss, _, _, _ = _evaluate_router(
        model=model,
        examples=train_examples,
        train_examples=train_examples,
        input_field=input_field,
        target_name=target_name,
        mean=mean,
        std=std,
        target_lookup=train_target_lookup,
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
    return RouterDistillationInputSummary(
        input_field=input_field,
        target_name=target_name,
        aggregation=aggregation,
        router_family=router_family,
        hidden_dim=hidden_dim if router_family == "mlp" else None,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        best_epoch=best_epoch,
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
    learning_rate: float,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 40,
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

    train_examples, eval_examples = stratified_router_train_eval_split(
        examples,
        eval_fraction=eval_fraction,
        seed=seed,
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
