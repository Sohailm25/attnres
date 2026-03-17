# ABOUTME: Runs TransformerLens-backed reconstruction checks on cached residual components.
# ABOUTME: Verifies real cache identities before broader oracle-alpha work touches model activations.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from transformer_lens import ActivationCache, HookedTransformer


@dataclass(frozen=True)
class LayerResidualMetrics:
    layer: int
    resid_mid_max_abs_error: float | None
    resid_post_max_abs_error: float


@dataclass(frozen=True)
class ModelBackedReconstructionMetrics:
    model_name: str
    prompt: str
    device: str
    num_sources: int
    source_labels: tuple[str, ...]
    final_residual_max_abs_error: float
    uniform_logits_max_abs_error: float
    layer_metrics: tuple[LayerResidualMetrics, ...]


def cache_name_filter(name: str) -> bool:
    if name in {"hook_embed", "hook_pos_embed", "ln_final.hook_scale"}:
        return True
    suffixes = (
        "hook_attn_out",
        "hook_mlp_out",
        "hook_resid_pre",
        "hook_resid_mid",
        "hook_resid_post",
    )
    return name.endswith(suffixes)


def _apply_final_norm_and_unembed(
    model: HookedTransformer,
    residual: torch.Tensor,
) -> torch.Tensor:
    if model.cfg.normalization_type is not None:
        residual = model.ln_final(residual)
    logits = model.unembed(residual)
    if model.cfg.output_logits_soft_cap > 0.0:
        logits = model.cfg.output_logits_soft_cap * torch.tanh(
            logits / model.cfg.output_logits_soft_cap
        )
    return logits


def _final_residual_from_cache(
    cache: ActivationCache, model: HookedTransformer
) -> torch.Tensor:
    if model.cfg.n_layers == 0:
        raise ValueError("model must have at least one layer")
    return cache[("resid_post", model.cfg.n_layers - 1)]


def _sum_residual_components(
    cache: ActivationCache,
    model: HookedTransformer,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    residual_stack, labels = cache.decompose_resid(
        layer=model.cfg.n_layers,
        mode="all",
        incl_embeds=True,
        return_labels=True,
    )
    reconstructed = residual_stack[0].clone()
    for component in residual_stack[1:]:
        reconstructed = reconstructed + component
    return reconstructed, tuple(labels)


def _collect_layer_metrics(
    cache: ActivationCache,
    model: HookedTransformer,
) -> tuple[LayerResidualMetrics, ...]:
    metrics: list[LayerResidualMetrics] = []
    for layer in range(model.cfg.n_layers):
        resid_pre = cache[("resid_pre", layer)]
        attn_out = cache[("attn_out", layer)]
        resid_mid_error: float | None = None

        if model.cfg.attn_only:
            expected_resid_post = resid_pre + attn_out
        elif model.cfg.parallel_attn_mlp:
            mlp_out = cache[("mlp_out", layer)]
            expected_resid_post = resid_pre + attn_out + mlp_out
        else:
            resid_mid = cache[("resid_mid", layer)]
            expected_resid_mid = resid_pre + attn_out
            resid_mid_error = float(
                (resid_mid - expected_resid_mid).abs().max().detach().cpu().item()
            )
            mlp_out = cache[("mlp_out", layer)]
            expected_resid_post = resid_mid + mlp_out

        resid_post = cache[("resid_post", layer)]
        resid_post_error = float(
            (resid_post - expected_resid_post).abs().max().detach().cpu().item()
        )
        metrics.append(
            LayerResidualMetrics(
                layer=layer,
                resid_mid_max_abs_error=resid_mid_error,
                resid_post_max_abs_error=resid_post_error,
            )
        )
    return tuple(metrics)


def model_backed_reconstruction_metrics(
    *,
    model: HookedTransformer,
    prompt: str,
    prepend_bos: bool | None = None,
    names_filter: Callable[[str], bool] = cache_name_filter,
) -> ModelBackedReconstructionMetrics:
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    with torch.inference_mode():
        logits, cache = model.run_with_cache(
            tokens,
            return_type="logits",
            names_filter=names_filter,
        )

    reconstructed_residual, labels = _sum_residual_components(cache, model)
    final_residual = _final_residual_from_cache(cache, model)
    reconstructed_logits = _apply_final_norm_and_unembed(model, reconstructed_residual)
    layer_metrics = _collect_layer_metrics(cache, model)

    return ModelBackedReconstructionMetrics(
        model_name=model.cfg.model_name,
        prompt=prompt,
        device=str(model.cfg.device),
        num_sources=len(labels),
        source_labels=labels,
        final_residual_max_abs_error=float(
            (reconstructed_residual - final_residual).abs().max().detach().cpu().item()
        ),
        uniform_logits_max_abs_error=float(
            (reconstructed_logits - logits).abs().max().detach().cpu().item()
        ),
        layer_metrics=layer_metrics,
    )
