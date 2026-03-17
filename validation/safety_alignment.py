# ABOUTME: Validates the first refusal-feature discovery workflow for the safety lane.
# ABOUTME: Localizes candidate layers, discovers refusal and harmfulness directions, and saves reusable prompt-level checkpoints.

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from prompts import PromptEntry, resolve_prompt_entries


ROLE_REFUSAL = "refusal"
ROLE_HARMFUL_CONTEXT = "harmful_context"
ROLE_BENIGN = "benign"
ROLE_SUFFIXES = {
    ROLE_REFUSAL,
    ROLE_HARMFUL_CONTEXT,
    ROLE_BENIGN,
}
POSITION_INSTRUCTION_FINAL = "instruction_final"
POSITION_ASSISTANT_PREFILL = "assistant_prefill"
REFUSAL_MARKER_PATTERN = re.compile(
    r"\b(i\s+(?:cannot|can't|can’t|won't|will not|am unable|cannot help|can't help|can’t help|won't help))\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SafetyPromptGroup:
    group_id: str
    split: str
    refusal_entry: PromptEntry
    harmful_context_entry: PromptEntry
    benign_entry: PromptEntry


@dataclass(frozen=True)
class SafetyLayerLocalizationSummary:
    target_name: str
    position_name: str
    mean_divergence_by_layer: tuple[float, ...]
    selected_layer: int
    selected_layer_mean_divergence: float


@dataclass(frozen=True)
class DirectionSeparationMetrics:
    mean_margin: float
    pair_accuracy: float


@dataclass(frozen=True)
class SafetyDirectionValidationSummary:
    target_name: str
    position_name: str
    selected_layer: int
    primary_metrics: DirectionSeparationMetrics
    cross_metrics: DirectionSeparationMetrics


@dataclass(frozen=True)
class SafetyPromptBehaviorResult:
    prompt_id: str
    group_id: str
    split: str
    role: str
    matched_expected_behavior: bool
    generated_completion_excerpt: str


@dataclass(frozen=True)
class RefusalFeatureDiscoverySummary:
    model_name: str
    collection_id: str
    pilot_num_groups: int
    confirm_num_groups: int
    max_new_tokens: int
    refusal_behavior_hit_rate_pilot: float
    refusal_behavior_hit_rate_confirm: float
    non_refusal_behavior_pass_rate_pilot: float
    non_refusal_behavior_pass_rate_confirm: float
    refusal_localization: SafetyLayerLocalizationSummary
    harmfulness_localization: SafetyLayerLocalizationSummary
    refusal_validation_pilot: SafetyDirectionValidationSummary
    refusal_validation_confirm: SafetyDirectionValidationSummary
    harmfulness_validation_pilot: SafetyDirectionValidationSummary
    harmfulness_validation_confirm: SafetyDirectionValidationSummary
    refusal_harmfulness_direction_cosine: float
    prompt_behaviors: tuple[SafetyPromptBehaviorResult, ...]


def _prompt_role(prompt_id: str) -> str:
    prefix, separator, suffix = prompt_id.rpartition("-")
    if not separator or suffix not in ROLE_SUFFIXES:
        raise ValueError(
            f"prompt id {prompt_id!r} must end with one of {sorted(ROLE_SUFFIXES)!r}"
        )
    return suffix


def _prompt_group_id(prompt_id: str) -> str:
    prefix, separator, suffix = prompt_id.rpartition("-")
    if not separator or suffix not in ROLE_SUFFIXES:
        raise ValueError(
            f"prompt id {prompt_id!r} must end with one of {sorted(ROLE_SUFFIXES)!r}"
        )
    return prefix


def group_safety_prompt_entries(
    entries: Sequence[PromptEntry],
) -> tuple[SafetyPromptGroup, ...]:
    grouped: dict[str, dict[str, PromptEntry]] = {}
    split_by_group: dict[str, str] = {}
    for entry in entries:
        group_id = _prompt_group_id(entry.prompt_id)
        role = _prompt_role(entry.prompt_id)
        group_entries = grouped.setdefault(group_id, {})
        if role in group_entries:
            raise ValueError(f"duplicate role {role!r} in safety group {group_id!r}")
        group_entries[role] = entry
        if group_id in split_by_group and split_by_group[group_id] != entry.split:
            raise ValueError(f"safety group {group_id!r} mixes prompt splits")
        split_by_group[group_id] = entry.split

    safety_groups: list[SafetyPromptGroup] = []
    for group_id in sorted(grouped):
        group_entries = grouped[group_id]
        missing = ROLE_SUFFIXES.difference(group_entries)
        if missing:
            raise ValueError(
                f"safety group {group_id!r} is missing roles {sorted(missing)!r}"
            )
        safety_groups.append(
            SafetyPromptGroup(
                group_id=group_id,
                split=split_by_group[group_id],
                refusal_entry=group_entries[ROLE_REFUSAL],
                harmful_context_entry=group_entries[ROLE_HARMFUL_CONTEXT],
                benign_entry=group_entries[ROLE_BENIGN],
            )
        )
    return tuple(safety_groups)


def matches_refusal_marker(text: str) -> bool:
    return bool(REFUSAL_MARKER_PATTERN.search(text))


def build_layer_localization_summary(
    *,
    target_name: str,
    position_name: str,
    positive_residuals: torch.Tensor,
    negative_residuals: torch.Tensor,
) -> SafetyLayerLocalizationSummary:
    if positive_residuals.shape != negative_residuals.shape:
        raise ValueError("positive and negative residuals must have identical shapes")
    if positive_residuals.ndim != 3:
        raise ValueError("residuals must have shape [groups, layers, d_model]")

    divergence = 1.0 - F.cosine_similarity(
        positive_residuals,
        negative_residuals,
        dim=-1,
        eps=1e-8,
    )
    mean_divergence = divergence.mean(dim=0)
    selected_layer = int(mean_divergence.argmax().item())
    mean_divergence_tuple = tuple(float(value) for value in mean_divergence.tolist())
    return SafetyLayerLocalizationSummary(
        target_name=target_name,
        position_name=position_name,
        mean_divergence_by_layer=mean_divergence_tuple,
        selected_layer=selected_layer,
        selected_layer_mean_divergence=mean_divergence_tuple[selected_layer],
    )


def discover_normalized_direction(
    *,
    positive_residuals: torch.Tensor,
    negative_residuals: torch.Tensor,
) -> torch.Tensor:
    if positive_residuals.shape != negative_residuals.shape:
        raise ValueError("positive and negative residuals must have identical shapes")
    if positive_residuals.ndim != 2:
        raise ValueError("residuals must have shape [groups, d_model]")

    direction = (positive_residuals - negative_residuals).mean(dim=0)
    norm = direction.norm()
    if float(norm.item()) <= 0.0:
        raise ValueError("cannot discover a direction from zero residual differences")
    return direction / norm


def paired_projection_summary(
    *,
    positive_residuals: torch.Tensor,
    negative_residuals: torch.Tensor,
    direction: torch.Tensor,
) -> DirectionSeparationMetrics:
    if positive_residuals.shape != negative_residuals.shape:
        raise ValueError("positive and negative residuals must have identical shapes")
    if positive_residuals.ndim != 2:
        raise ValueError("residuals must have shape [groups, d_model]")
    if direction.ndim != 1 or direction.shape[0] != positive_residuals.shape[-1]:
        raise ValueError("direction must have shape [d_model]")

    positive_scores = positive_residuals @ direction
    negative_scores = negative_residuals @ direction
    margins = positive_scores - negative_scores
    return DirectionSeparationMetrics(
        mean_margin=float(margins.mean().item()),
        pair_accuracy=float((margins > 0.0).float().mean().item()),
    )


def _chat_formatted_prompt_and_positions(
    model: HookedTransformer,
    prompt_text: str,
) -> tuple[str, torch.Tensor, int, int]:
    messages = [{"role": "user", "content": prompt_text}]
    tokenizer = model.tokenizer
    if hasattr(tokenizer, "apply_chat_template"):
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        instruction_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        return (
            formatted_prompt,
            full_tokens,
            int(instruction_tokens.shape[-1] - 1),
            int(full_tokens.shape[-1] - 1),
        )

    formatted_prompt = f"User: {prompt_text}\nAssistant:"
    full_tokens = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )["input_ids"]
    instruction_tokens = tokenizer(
        f"User: {prompt_text}",
        return_tensors="pt",
    )["input_ids"]
    return (
        formatted_prompt,
        full_tokens,
        int(instruction_tokens.shape[-1] - 1),
        int(full_tokens.shape[-1] - 1),
    )


def _resid_post_name_filter(name: str) -> bool:
    return name.endswith("hook_resid_post")


def _greedy_generate_completion(
    model: HookedTransformer,
    tokens: torch.Tensor,
    *,
    max_new_tokens: int,
) -> str:
    generated = tokens
    for _ in range(max_new_tokens):
        with torch.inference_mode():
            logits = model(generated, return_type="logits")
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=-1)
    new_tokens = generated[:, tokens.shape[-1] :]
    return model.tokenizer.decode(new_tokens[0].tolist(), skip_special_tokens=True)


def _collect_prompt_checkpoint(
    *,
    model: HookedTransformer,
    entry: PromptEntry,
    checkpoint_path: Path,
    max_new_tokens: int,
) -> dict[str, object]:
    if checkpoint_path.is_file():
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    formatted_prompt, token_tensor, instruction_final_index, assistant_prefill_index = (
        _chat_formatted_prompt_and_positions(model, entry.text)
    )
    tokens = token_tensor.to(model.cfg.device)
    with torch.inference_mode():
        _, cache = model.run_with_cache(
            tokens,
            return_type="logits",
            names_filter=_resid_post_name_filter,
        )

    instruction_final = torch.stack(
        [
            cache[("resid_post", layer)][0, instruction_final_index, :].detach().cpu()
            for layer in range(model.cfg.n_layers)
        ],
        dim=0,
    )
    assistant_prefill = torch.stack(
        [
            cache[("resid_post", layer)][0, assistant_prefill_index, :].detach().cpu()
            for layer in range(model.cfg.n_layers)
        ],
        dim=0,
    )
    generated_completion = _greedy_generate_completion(
        model,
        tokens,
        max_new_tokens=max_new_tokens,
    ).strip()
    role = _prompt_role(entry.prompt_id)
    refusal_like = matches_refusal_marker(generated_completion)
    matched_expected_behavior = (
        refusal_like if role == ROLE_REFUSAL else not refusal_like
    )

    checkpoint = {
        "prompt_id": entry.prompt_id,
        "group_id": _prompt_group_id(entry.prompt_id),
        "split": entry.split,
        "role": role,
        "prompt_text": entry.text,
        "formatted_prompt": formatted_prompt,
        "instruction_final_index": instruction_final_index,
        "assistant_prefill_index": assistant_prefill_index,
        "instruction_final_residuals": instruction_final,
        "assistant_prefill_residuals": assistant_prefill,
        "generated_completion": generated_completion,
        "matched_expected_behavior": matched_expected_behavior,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint


def _selected_entries_from_groups(
    groups: Sequence[SafetyPromptGroup],
) -> tuple[PromptEntry, ...]:
    entries: list[PromptEntry] = []
    for group in groups:
        entries.extend(
            (
                group.refusal_entry,
                group.harmful_context_entry,
                group.benign_entry,
            )
        )
    return tuple(entries)


def _residual_matrix_for_groups(
    groups: Sequence[SafetyPromptGroup],
    checkpoints_by_prompt_id: dict[str, dict[str, object]],
    *,
    role: str,
    position_name: str,
) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    key = f"{position_name}_residuals"
    for group in groups:
        prompt_id = getattr(group, f"{role}_entry").prompt_id
        checkpoint = checkpoints_by_prompt_id[prompt_id]
        tensors.append(checkpoint[key])
    return torch.stack(tensors, dim=0).to(dtype=torch.float32)


def _direction_validation_summary(
    *,
    target_name: str,
    position_name: str,
    selected_layer: int,
    positive_residuals: torch.Tensor,
    negative_residuals: torch.Tensor,
    primary_direction: torch.Tensor,
    cross_direction: torch.Tensor,
) -> SafetyDirectionValidationSummary:
    primary_metrics = paired_projection_summary(
        positive_residuals=positive_residuals[:, selected_layer, :],
        negative_residuals=negative_residuals[:, selected_layer, :],
        direction=primary_direction,
    )
    cross_metrics = paired_projection_summary(
        positive_residuals=positive_residuals[:, selected_layer, :],
        negative_residuals=negative_residuals[:, selected_layer, :],
        direction=cross_direction,
    )
    return SafetyDirectionValidationSummary(
        target_name=target_name,
        position_name=position_name,
        selected_layer=selected_layer,
        primary_metrics=primary_metrics,
        cross_metrics=cross_metrics,
    )


def _behavior_rate(
    checkpoints: Iterable[dict[str, object]],
    *,
    refusal_expected: bool,
) -> float:
    filtered = [
        checkpoint
        for checkpoint in checkpoints
        if (checkpoint["role"] == ROLE_REFUSAL) == refusal_expected
    ]
    if not filtered:
        return 0.0
    matched = sum(
        bool(checkpoint["matched_expected_behavior"]) for checkpoint in filtered
    )
    return matched / len(filtered)


def save_refusal_feature_discovery_artifacts(
    summary: RefusalFeatureDiscoverySummary,
    *,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2))
    return summary_path


def run_refusal_feature_discovery_validation(
    *,
    model: HookedTransformer,
    collection_id: str,
    output_dir: Path,
    max_new_tokens: int = 32,
    max_pilot_groups: int | None = None,
    max_confirm_groups: int | None = None,
) -> RefusalFeatureDiscoverySummary:
    pilot_groups = group_safety_prompt_entries(
        resolve_prompt_entries(
            collection_id=collection_id,
            split="pilot",
            exploratory=True,
        )
    )
    confirm_groups = group_safety_prompt_entries(
        resolve_prompt_entries(
            collection_id=collection_id,
            split="confirm",
            exploratory=False,
        )
    )
    if max_pilot_groups is not None:
        pilot_groups = pilot_groups[:max_pilot_groups]
    if max_confirm_groups is not None:
        confirm_groups = confirm_groups[:max_confirm_groups]

    checkpoint_dir = output_dir / "checkpoints" / "prompt_residuals"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_entries = _selected_entries_from_groups(
        pilot_groups
    ) + _selected_entries_from_groups(confirm_groups)
    checkpoints_by_prompt_id: dict[str, dict[str, object]] = {}
    for entry in all_entries:
        checkpoint_path = checkpoint_dir / f"{entry.prompt_id}.pt"
        checkpoints_by_prompt_id[entry.prompt_id] = _collect_prompt_checkpoint(
            model=model,
            entry=entry,
            checkpoint_path=checkpoint_path,
            max_new_tokens=max_new_tokens,
        )

    pilot_checkpoints = [
        checkpoints_by_prompt_id[entry.prompt_id]
        for entry in _selected_entries_from_groups(pilot_groups)
    ]
    confirm_checkpoints = [
        checkpoints_by_prompt_id[entry.prompt_id]
        for entry in _selected_entries_from_groups(confirm_groups)
    ]

    pilot_refusal_prefill = _residual_matrix_for_groups(
        pilot_groups,
        checkpoints_by_prompt_id,
        role=ROLE_REFUSAL,
        position_name=POSITION_ASSISTANT_PREFILL,
    )
    pilot_harmful_prefill = _residual_matrix_for_groups(
        pilot_groups,
        checkpoints_by_prompt_id,
        role=ROLE_HARMFUL_CONTEXT,
        position_name=POSITION_ASSISTANT_PREFILL,
    )
    pilot_harmful_instruction = _residual_matrix_for_groups(
        pilot_groups,
        checkpoints_by_prompt_id,
        role=ROLE_HARMFUL_CONTEXT,
        position_name=POSITION_INSTRUCTION_FINAL,
    )
    pilot_benign_instruction = _residual_matrix_for_groups(
        pilot_groups,
        checkpoints_by_prompt_id,
        role=ROLE_BENIGN,
        position_name=POSITION_INSTRUCTION_FINAL,
    )

    refusal_localization = build_layer_localization_summary(
        target_name=ROLE_REFUSAL,
        position_name=POSITION_ASSISTANT_PREFILL,
        positive_residuals=pilot_refusal_prefill,
        negative_residuals=pilot_harmful_prefill,
    )
    harmfulness_localization = build_layer_localization_summary(
        target_name="harmfulness",
        position_name=POSITION_INSTRUCTION_FINAL,
        positive_residuals=pilot_harmful_instruction,
        negative_residuals=pilot_benign_instruction,
    )

    refusal_direction = discover_normalized_direction(
        positive_residuals=pilot_refusal_prefill[
            :, refusal_localization.selected_layer, :
        ],
        negative_residuals=pilot_harmful_prefill[
            :, refusal_localization.selected_layer, :
        ],
    )
    harmfulness_direction = discover_normalized_direction(
        positive_residuals=pilot_harmful_instruction[
            :, harmfulness_localization.selected_layer, :
        ],
        negative_residuals=pilot_benign_instruction[
            :, harmfulness_localization.selected_layer, :
        ],
    )

    confirm_refusal_prefill = _residual_matrix_for_groups(
        confirm_groups,
        checkpoints_by_prompt_id,
        role=ROLE_REFUSAL,
        position_name=POSITION_ASSISTANT_PREFILL,
    )
    confirm_harmful_prefill = _residual_matrix_for_groups(
        confirm_groups,
        checkpoints_by_prompt_id,
        role=ROLE_HARMFUL_CONTEXT,
        position_name=POSITION_ASSISTANT_PREFILL,
    )
    confirm_harmful_instruction = _residual_matrix_for_groups(
        confirm_groups,
        checkpoints_by_prompt_id,
        role=ROLE_HARMFUL_CONTEXT,
        position_name=POSITION_INSTRUCTION_FINAL,
    )
    confirm_benign_instruction = _residual_matrix_for_groups(
        confirm_groups,
        checkpoints_by_prompt_id,
        role=ROLE_BENIGN,
        position_name=POSITION_INSTRUCTION_FINAL,
    )

    refusal_validation_pilot = _direction_validation_summary(
        target_name=ROLE_REFUSAL,
        position_name=POSITION_ASSISTANT_PREFILL,
        selected_layer=refusal_localization.selected_layer,
        positive_residuals=pilot_refusal_prefill,
        negative_residuals=pilot_harmful_prefill,
        primary_direction=refusal_direction,
        cross_direction=harmfulness_direction,
    )
    refusal_validation_confirm = _direction_validation_summary(
        target_name=ROLE_REFUSAL,
        position_name=POSITION_ASSISTANT_PREFILL,
        selected_layer=refusal_localization.selected_layer,
        positive_residuals=confirm_refusal_prefill,
        negative_residuals=confirm_harmful_prefill,
        primary_direction=refusal_direction,
        cross_direction=harmfulness_direction,
    )
    harmfulness_validation_pilot = _direction_validation_summary(
        target_name="harmfulness",
        position_name=POSITION_INSTRUCTION_FINAL,
        selected_layer=harmfulness_localization.selected_layer,
        positive_residuals=pilot_harmful_instruction,
        negative_residuals=pilot_benign_instruction,
        primary_direction=harmfulness_direction,
        cross_direction=refusal_direction,
    )
    harmfulness_validation_confirm = _direction_validation_summary(
        target_name="harmfulness",
        position_name=POSITION_INSTRUCTION_FINAL,
        selected_layer=harmfulness_localization.selected_layer,
        positive_residuals=confirm_harmful_instruction,
        negative_residuals=confirm_benign_instruction,
        primary_direction=harmfulness_direction,
        cross_direction=refusal_direction,
    )

    prompt_behaviors = tuple(
        SafetyPromptBehaviorResult(
            prompt_id=checkpoint["prompt_id"],
            group_id=checkpoint["group_id"],
            split=checkpoint["split"],
            role=checkpoint["role"],
            matched_expected_behavior=bool(checkpoint["matched_expected_behavior"]),
            generated_completion_excerpt=str(checkpoint["generated_completion"])[:240],
        )
        for checkpoint in (*pilot_checkpoints, *confirm_checkpoints)
    )

    summary = RefusalFeatureDiscoverySummary(
        model_name=model.cfg.model_name,
        collection_id=collection_id,
        pilot_num_groups=len(pilot_groups),
        confirm_num_groups=len(confirm_groups),
        max_new_tokens=max_new_tokens,
        refusal_behavior_hit_rate_pilot=_behavior_rate(
            pilot_checkpoints,
            refusal_expected=True,
        ),
        refusal_behavior_hit_rate_confirm=_behavior_rate(
            confirm_checkpoints,
            refusal_expected=True,
        ),
        non_refusal_behavior_pass_rate_pilot=_behavior_rate(
            pilot_checkpoints,
            refusal_expected=False,
        ),
        non_refusal_behavior_pass_rate_confirm=_behavior_rate(
            confirm_checkpoints,
            refusal_expected=False,
        ),
        refusal_localization=refusal_localization,
        harmfulness_localization=harmfulness_localization,
        refusal_validation_pilot=refusal_validation_pilot,
        refusal_validation_confirm=refusal_validation_confirm,
        harmfulness_validation_pilot=harmfulness_validation_pilot,
        harmfulness_validation_confirm=harmfulness_validation_confirm,
        refusal_harmfulness_direction_cosine=float(
            torch.dot(refusal_direction, harmfulness_direction).item()
        ),
        prompt_behaviors=prompt_behaviors,
    )
    save_refusal_feature_discovery_artifacts(summary, output_dir=output_dir)
    return summary
