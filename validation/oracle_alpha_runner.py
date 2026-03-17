# ABOUTME: Runs the first development-model oracle-alpha slice on fixed cached residual sources.
# ABOUTME: Uses the saved prompt and control registries so runner code cannot drift from the repo contract.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from prompts import PromptEntry, resolve_prompt_entries
from .model_backed import cache_name_filter
from .oracle_alpha_controls import (
    BootstrapMeanInterval,
    bootstrap_mean_confidence_interval,
    load_oracle_alpha_control_registry,
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

    z = torch.zeros(
        num_sources,
        dtype=residual_stack.dtype,
        device=device,
        requires_grad=True,
    )
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
    )


def run_oracle_alpha_collection(
    *,
    model: HookedTransformer,
    collection_id: str,
    split: str,
    exploratory: bool,
    max_sequences: int | None = None,
    optimization_steps: int = 20,
    learning_rate: float = 0.1,
    seed: int = 0,
    prepend_bos: bool | None = None,
) -> OracleAlphaRunSummary:
    control_registry = load_oracle_alpha_control_registry()
    control_plan = control_registry.plans[collection_id]
    entries = list(
        resolve_prompt_entries(
            collection_id=collection_id,
            split=split,
            exploratory=exploratory,
        )
    )
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
