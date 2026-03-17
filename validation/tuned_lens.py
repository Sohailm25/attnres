# ABOUTME: Trains and evaluates the smallest custom tuned-lens path needed for the Gemma-2 tool-breakage baseline.
# ABOUTME: Keeps the first viability pilot focused on original-model residual translation before routed-vs-original comparisons begin.

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from prompts import PromptEntry, resolve_prompt_entries


@dataclass(frozen=True)
class ResidualTranslationExamples:
    model_name: str
    prompt_ids: tuple[str, ...]
    num_prompts: int
    num_layers: int
    num_positions: int
    d_model: int
    layer_residuals: torch.Tensor
    final_residuals: torch.Tensor
    final_position_mask: torch.Tensor


@dataclass(frozen=True)
class TunedLensLayerMetric:
    layer: int
    raw_kl_to_final: float
    tuned_kl_to_final: float
    raw_top1_agreement: float
    tuned_top1_agreement: float
    final_position_raw_kl_to_final: float
    final_position_tuned_kl_to_final: float
    final_position_raw_top1_agreement: float
    final_position_tuned_top1_agreement: float


@dataclass(frozen=True)
class TunedLensViabilitySummary:
    model_name: str
    train_collection_id: str
    train_split: str
    eval_collection_id: str
    eval_split: str
    translator_rank: int
    training_objective: str
    num_train_prompts: int
    num_eval_prompts: int
    num_train_positions: int
    num_eval_positions: int
    mean_raw_kl_to_final: float
    mean_tuned_kl_to_final: float
    mean_raw_top1_agreement: float
    mean_tuned_top1_agreement: float
    final_position_mean_raw_kl_to_final: float
    final_position_mean_tuned_kl_to_final: float
    final_position_mean_raw_top1_agreement: float
    final_position_mean_tuned_top1_agreement: float
    layer_metrics: tuple[TunedLensLayerMetric, ...]
    checkpoint_path: str | None


def tuned_lens_cache_name_filter(name: str) -> bool:
    return name.endswith("hook_resid_post")


def _safe_key(raw: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    if not normalized:
        normalized = "artifact"
    return f"{normalized}-{digest}"


def _move_tree_to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _move_tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tree_to_cpu(item) for item in value)
    return value


class LowRankAffineResidualLens(torch.nn.Module):
    def __init__(self, *, num_layers: int, d_model: int, rank: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if rank < 1:
            raise ValueError("rank must be positive")

        self.num_layers = num_layers
        self.d_model = d_model
        self.rank = rank
        self.left_factors = torch.nn.Parameter(
            torch.empty(num_layers, d_model, rank, dtype=torch.float32)
        )
        self.right_factors = torch.nn.Parameter(
            torch.empty(num_layers, rank, d_model, dtype=torch.float32)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_layers, d_model, dtype=torch.float32)
        )
        torch.nn.init.normal_(self.left_factors, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.right_factors, mean=0.0, std=0.01)

    def forward_all(self, layer_residuals: torch.Tensor) -> torch.Tensor:
        if layer_residuals.ndim != 3:
            raise ValueError(
                "layer_residuals must have shape [layers, positions, d_model]"
            )
        if layer_residuals.shape[0] != self.num_layers:
            raise ValueError("layer count does not match the fitted lens")
        if layer_residuals.shape[2] != self.d_model:
            raise ValueError("d_model does not match the fitted lens")

        projected = torch.einsum("lpd,ldr->lpr", layer_residuals, self.left_factors)
        correction = torch.einsum("lpr,lrd->lpd", projected, self.right_factors)
        return layer_residuals + correction + self.bias.unsqueeze(1)

    def forward_layer(self, layer: int, residuals: torch.Tensor) -> torch.Tensor:
        if residuals.ndim != 2:
            raise ValueError("residuals must have shape [positions, d_model]")
        if not 0 <= layer < self.num_layers:
            raise ValueError("layer index out of range")
        if residuals.shape[1] != self.d_model:
            raise ValueError("d_model does not match the fitted lens")

        projected = residuals @ self.left_factors[layer]
        correction = projected @ self.right_factors[layer]
        return residuals + correction + self.bias[layer]


def _apply_final_norm_and_unembed(
    model: HookedTransformer,
    residuals: torch.Tensor,
) -> torch.Tensor:
    flat = residuals.reshape(-1, residuals.shape[-1]).to(model.W_U.device)
    if model.cfg.normalization_type is not None:
        flat = model.ln_final(flat)
    logits = model.unembed(flat)
    if model.cfg.output_logits_soft_cap > 0.0:
        logits = model.cfg.output_logits_soft_cap * torch.tanh(
            logits / model.cfg.output_logits_soft_cap
        )
    return logits.reshape(*residuals.shape[:-1], logits.shape[-1])


def collect_residual_translation_examples(
    *,
    model: HookedTransformer,
    prompt_entries: Sequence[PromptEntry],
    names_filter: Callable[[str], bool] = tuned_lens_cache_name_filter,
    checkpoint_dir: Path | None = None,
) -> ResidualTranslationExamples:
    if not prompt_entries:
        raise ValueError("prompt_entries must not be empty")

    layer_residuals_by_prompt: list[torch.Tensor] = []
    final_residuals_by_prompt: list[torch.Tensor] = []
    final_position_masks: list[torch.Tensor] = []
    prompt_ids: list[str] = []

    for entry in prompt_entries:
        checkpoint_path = None
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{_safe_key(entry.prompt_id)}.pt"

        if checkpoint_path is not None and checkpoint_path.is_file():
            cached = torch.load(checkpoint_path, map_location="cpu")
            prompt_layer_residuals = cached["layer_residuals"].to(torch.float32)
            prompt_final_residuals = cached["final_residuals"].to(torch.float32)
            final_position_mask = cached["final_position_mask"].to(torch.bool)
        else:
            tokens = model.to_tokens(entry.text)
            with torch.inference_mode():
                _, cache = model.run_with_cache(
                    tokens,
                    return_type="logits",
                    names_filter=names_filter,
                )

            prompt_layer_residuals = torch.stack(
                [
                    cache[("resid_post", layer)]
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    for layer in range(model.cfg.n_layers)
                ]
            )
            prompt_final_residuals = (
                cache[("resid_post", model.cfg.n_layers - 1)]
                .squeeze(0)
                .detach()
                .cpu()
                .to(torch.float32)
            )
            final_position_mask = torch.zeros(
                prompt_final_residuals.shape[0],
                dtype=torch.bool,
            )
            final_position_mask[-1] = True
            if checkpoint_path is not None:
                torch.save(
                    {
                        "prompt_id": entry.prompt_id,
                        "layer_residuals": prompt_layer_residuals,
                        "final_residuals": prompt_final_residuals,
                        "final_position_mask": final_position_mask,
                    },
                    checkpoint_path,
                )

        layer_residuals_by_prompt.append(prompt_layer_residuals)
        final_residuals_by_prompt.append(prompt_final_residuals)
        final_position_masks.append(final_position_mask)
        prompt_ids.append(entry.prompt_id)

    layer_residuals = torch.cat(layer_residuals_by_prompt, dim=1)
    final_residuals = torch.cat(final_residuals_by_prompt, dim=0)
    final_position_mask = torch.cat(final_position_masks, dim=0)

    return ResidualTranslationExamples(
        model_name=model.cfg.model_name,
        prompt_ids=tuple(prompt_ids),
        num_prompts=len(prompt_entries),
        num_layers=model.cfg.n_layers,
        num_positions=final_residuals.shape[0],
        d_model=model.cfg.d_model,
        layer_residuals=layer_residuals,
        final_residuals=final_residuals,
        final_position_mask=final_position_mask,
    )


def fit_low_rank_residual_lens(
    *,
    examples: ResidualTranslationExamples,
    rank: int,
    num_steps: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
    readout_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every_steps: int | None = None,
) -> LowRankAffineResidualLens:
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if checkpoint_every_steps is not None and checkpoint_every_steps < 1:
        raise ValueError("checkpoint_every_steps must be positive when provided")
    del readout_fn

    torch.manual_seed(seed)
    lens = LowRankAffineResidualLens(
        num_layers=examples.num_layers,
        d_model=examples.d_model,
        rank=rank,
    ).to(device)
    optimizer = torch.optim.Adam(
        lens.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    start_step = 0
    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        lens.load_state_dict(checkpoint["lens_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"])
    layer_residuals = examples.layer_residuals.to(device)
    final_residuals = examples.final_residuals.to(device)
    batch_size = min(256, examples.num_positions)

    for step in range(start_step, num_steps):
        if batch_size == examples.num_positions:
            batch_indices = torch.arange(examples.num_positions, device=device)
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + step)
            sampled = torch.randperm(examples.num_positions, generator=generator)[
                :batch_size
            ]
            batch_indices = sampled.to(device)

        translated = lens.forward_all(layer_residuals[:, batch_indices, :])
        targets = final_residuals[batch_indices].unsqueeze(0).expand_as(translated)
        loss = F.mse_loss(translated, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if checkpoint_path is not None and (
            (step + 1) == num_steps
            or (
                checkpoint_every_steps is not None
                and (step + 1) % checkpoint_every_steps == 0
            )
        ):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step + 1,
                    "seed": seed,
                    "num_layers": examples.num_layers,
                    "d_model": examples.d_model,
                    "rank": rank,
                    "lens_state_dict": _move_tree_to_cpu(lens.state_dict()),
                    "optimizer_state_dict": _move_tree_to_cpu(optimizer.state_dict()),
                },
                checkpoint_path,
            )

    return lens.cpu()


def _mean_kl_to_final(
    *,
    target_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> float:
    target_log_probs = F.log_softmax(target_logits, dim=-1)
    target_probs = target_log_probs.exp()
    candidate_log_probs = F.log_softmax(candidate_logits, dim=-1)
    return float(
        F.kl_div(
            candidate_log_probs,
            target_probs,
            reduction="batchmean",
            log_target=False,
        )
        .detach()
        .cpu()
        .item()
    )


def _top1_agreement(
    *,
    target_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> float:
    target_top1 = torch.argmax(target_logits, dim=-1)
    candidate_top1 = torch.argmax(candidate_logits, dim=-1)
    return float((target_top1 == candidate_top1).to(torch.float32).mean().item())


def evaluate_low_rank_residual_lens(
    *,
    lens: LowRankAffineResidualLens,
    examples: ResidualTranslationExamples,
    readout_fn: Callable[[torch.Tensor], torch.Tensor],
    train_collection_id: str,
    train_split: str,
    eval_collection_id: str,
    eval_split: str,
) -> TunedLensViabilitySummary:
    target_logits = readout_fn(examples.final_residuals)
    final_position_mask = examples.final_position_mask

    layer_metrics: list[TunedLensLayerMetric] = []
    for layer in range(examples.num_layers):
        raw_residuals = examples.layer_residuals[layer]
        tuned_residuals = lens.forward_layer(layer, raw_residuals)

        raw_logits = readout_fn(raw_residuals)
        tuned_logits = readout_fn(tuned_residuals)

        raw_kl = _mean_kl_to_final(
            target_logits=target_logits,
            candidate_logits=raw_logits,
        )
        tuned_kl = _mean_kl_to_final(
            target_logits=target_logits,
            candidate_logits=tuned_logits,
        )
        raw_top1 = _top1_agreement(
            target_logits=target_logits,
            candidate_logits=raw_logits,
        )
        tuned_top1 = _top1_agreement(
            target_logits=target_logits,
            candidate_logits=tuned_logits,
        )

        raw_final_position_logits = raw_logits[final_position_mask]
        tuned_final_position_logits = tuned_logits[final_position_mask]
        target_final_position_logits = target_logits[final_position_mask]
        raw_final_kl = _mean_kl_to_final(
            target_logits=target_final_position_logits,
            candidate_logits=raw_final_position_logits,
        )
        tuned_final_kl = _mean_kl_to_final(
            target_logits=target_final_position_logits,
            candidate_logits=tuned_final_position_logits,
        )
        raw_final_top1 = _top1_agreement(
            target_logits=target_final_position_logits,
            candidate_logits=raw_final_position_logits,
        )
        tuned_final_top1 = _top1_agreement(
            target_logits=target_final_position_logits,
            candidate_logits=tuned_final_position_logits,
        )

        layer_metrics.append(
            TunedLensLayerMetric(
                layer=layer,
                raw_kl_to_final=raw_kl,
                tuned_kl_to_final=tuned_kl,
                raw_top1_agreement=raw_top1,
                tuned_top1_agreement=tuned_top1,
                final_position_raw_kl_to_final=raw_final_kl,
                final_position_tuned_kl_to_final=tuned_final_kl,
                final_position_raw_top1_agreement=raw_final_top1,
                final_position_tuned_top1_agreement=tuned_final_top1,
            )
        )

    def mean(values: Sequence[float]) -> float:
        return float(sum(values) / len(values))

    return TunedLensViabilitySummary(
        model_name=examples.model_name,
        train_collection_id=train_collection_id,
        train_split=train_split,
        eval_collection_id=eval_collection_id,
        eval_split=eval_split,
        translator_rank=lens.rank,
        training_objective="residual_mse_then_logit_eval",
        num_train_prompts=0,
        num_eval_prompts=examples.num_prompts,
        num_train_positions=0,
        num_eval_positions=examples.num_positions,
        mean_raw_kl_to_final=mean([metric.raw_kl_to_final for metric in layer_metrics]),
        mean_tuned_kl_to_final=mean(
            [metric.tuned_kl_to_final for metric in layer_metrics]
        ),
        mean_raw_top1_agreement=mean(
            [metric.raw_top1_agreement for metric in layer_metrics]
        ),
        mean_tuned_top1_agreement=mean(
            [metric.tuned_top1_agreement for metric in layer_metrics]
        ),
        final_position_mean_raw_kl_to_final=mean(
            [metric.final_position_raw_kl_to_final for metric in layer_metrics]
        ),
        final_position_mean_tuned_kl_to_final=mean(
            [metric.final_position_tuned_kl_to_final for metric in layer_metrics]
        ),
        final_position_mean_raw_top1_agreement=mean(
            [metric.final_position_raw_top1_agreement for metric in layer_metrics]
        ),
        final_position_mean_tuned_top1_agreement=mean(
            [metric.final_position_tuned_top1_agreement for metric in layer_metrics]
        ),
        layer_metrics=tuple(layer_metrics),
        checkpoint_path=None,
    )


def save_tuned_lens_artifacts(
    *,
    output_dir: Path,
    lens: LowRankAffineResidualLens,
    summary: TunedLensViabilitySummary,
) -> TunedLensViabilitySummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    summary_path = output_dir / "summary.json"

    torch.save(
        {
            "num_layers": lens.num_layers,
            "d_model": lens.d_model,
            "rank": lens.rank,
            "state_dict": lens.state_dict(),
        },
        checkpoint_path,
    )
    saved_summary = replace(summary, checkpoint_path=str(checkpoint_path))
    summary_path.write_text(json.dumps(asdict(saved_summary), indent=2) + "\n")
    return saved_summary


def run_tuned_lens_viability_pilot(
    *,
    model: HookedTransformer,
    train_collection_id: str,
    train_split: str,
    eval_collection_id: str,
    eval_split: str,
    output_dir: Path,
    translator_rank: int,
    num_steps: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    max_train_prompts: int | None = None,
    max_eval_prompts: int | None = None,
    checkpoint_every_steps: int = 25,
) -> TunedLensViabilitySummary:
    train_entries = resolve_prompt_entries(
        collection_id=train_collection_id,
        split=train_split,
        exploratory=False,
    )
    eval_entries = resolve_prompt_entries(
        collection_id=eval_collection_id,
        split=eval_split,
        exploratory=False,
    )
    if max_train_prompts is not None:
        train_entries = train_entries[:max_train_prompts]
    if max_eval_prompts is not None:
        eval_entries = eval_entries[:max_eval_prompts]

    checkpoint_root = output_dir / "checkpoints"
    train_examples = collect_residual_translation_examples(
        model=model,
        prompt_entries=train_entries,
        checkpoint_dir=checkpoint_root / "train_examples",
    )
    eval_examples = collect_residual_translation_examples(
        model=model,
        prompt_entries=eval_entries,
        checkpoint_dir=checkpoint_root / "eval_examples",
    )
    lens = fit_low_rank_residual_lens(
        examples=train_examples,
        rank=translator_rank,
        num_steps=num_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=str(model.cfg.device),
        checkpoint_path=checkpoint_root / "training_state.pt",
        checkpoint_every_steps=checkpoint_every_steps,
    )
    summary = evaluate_low_rank_residual_lens(
        lens=lens,
        examples=eval_examples,
        readout_fn=lambda residuals: _apply_final_norm_and_unembed(model, residuals),
        train_collection_id=train_collection_id,
        train_split=train_split,
        eval_collection_id=eval_collection_id,
        eval_split=eval_split,
    )
    summary = replace(
        summary,
        num_train_prompts=train_examples.num_prompts,
        num_train_positions=train_examples.num_positions,
        num_eval_prompts=eval_examples.num_prompts,
        num_eval_positions=eval_examples.num_positions,
    )
    return save_tuned_lens_artifacts(
        output_dir=output_dir,
        lens=lens,
        summary=summary,
    )
