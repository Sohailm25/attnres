# ABOUTME: Builds the first same-model factual-recall tool-breakage baseline on Gemma-2.
# ABOUTME: Compares original-versus-routed traces under raw and tuned lens while keeping KL-primary metric discipline explicit.

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from prompts import PromptEntry, resolve_prompt_entries
from .model_backed import cache_name_filter
from .tuned_lens import LowRankAffineResidualLens


@dataclass(frozen=True)
class ToolBreakageLayerTrace:
    layer: int
    raw_original_mean_kl_to_final: float
    raw_routed_mean_kl_to_final: float
    tuned_original_mean_kl_to_final: float
    tuned_routed_mean_kl_to_final: float
    raw_original_mean_top1_agreement: float
    raw_routed_mean_top1_agreement: float
    tuned_original_mean_top1_agreement: float
    tuned_routed_mean_top1_agreement: float
    raw_original_final_position_kl_to_final: float
    raw_routed_final_position_kl_to_final: float
    tuned_original_final_position_kl_to_final: float
    tuned_routed_final_position_kl_to_final: float
    raw_original_final_position_top1_agreement: float
    raw_routed_final_position_top1_agreement: float
    tuned_original_final_position_top1_agreement: float
    tuned_routed_final_position_top1_agreement: float
    raw_original_final_position_target_probability: float
    raw_routed_final_position_target_probability: float
    tuned_original_final_position_target_probability: float
    tuned_routed_final_position_target_probability: float
    raw_original_final_position_target_rank: int
    raw_routed_final_position_target_rank: int
    tuned_original_final_position_target_rank: int
    tuned_routed_final_position_target_rank: int


@dataclass(frozen=True)
class ToolBreakagePromptResult:
    prompt_id: str
    prompt: str
    split: str
    target_text: str
    target_token_id: int
    target_token_text: str
    source_labels: tuple[str, ...]
    oracle_alpha: tuple[float, ...]
    raw_original_non_monotonic: bool
    raw_routed_non_monotonic: bool
    tuned_original_non_monotonic: bool
    tuned_routed_non_monotonic: bool
    layer_traces: tuple[ToolBreakageLayerTrace, ...]


@dataclass(frozen=True)
class ToolBreakageRunSummary:
    model_name: str
    collection_id: str
    split: str
    exploratory: bool
    tuned_lens_checkpoint_path: str
    num_prompts: int
    mean_raw_original_kl_to_final: float
    mean_raw_routed_kl_to_final: float
    mean_tuned_original_kl_to_final: float
    mean_tuned_routed_kl_to_final: float
    mean_tuned_kl_increase_under_routing: float
    final_position_mean_raw_original_kl_to_final: float
    final_position_mean_raw_routed_kl_to_final: float
    final_position_mean_tuned_original_kl_to_final: float
    final_position_mean_tuned_routed_kl_to_final: float
    final_position_mean_tuned_kl_increase_under_routing: float
    mean_raw_original_top1_agreement: float
    mean_raw_routed_top1_agreement: float
    mean_tuned_original_top1_agreement: float
    mean_tuned_routed_top1_agreement: float
    final_position_mean_raw_original_top1_agreement: float
    final_position_mean_raw_routed_top1_agreement: float
    final_position_mean_tuned_original_top1_agreement: float
    final_position_mean_tuned_routed_top1_agreement: float
    fraction_raw_original_non_monotonic_prompts: float
    fraction_raw_routed_non_monotonic_prompts: float
    fraction_tuned_original_non_monotonic_prompts: float
    fraction_tuned_routed_non_monotonic_prompts: float
    prompt_results: tuple[ToolBreakagePromptResult, ...]


def _safe_key(raw: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    if not normalized:
        normalized = "artifact"
    return f"{normalized}-{digest}"


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


def _sequence_mean_cross_entropy(
    logits: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
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


def _optimize_oracle_alpha(
    *,
    model: HookedTransformer,
    residual_stack: torch.Tensor,
    tokens: torch.Tensor,
    optimization_steps: int,
    learning_rate: float,
    seed: int,
) -> torch.Tensor:
    num_sources = residual_stack.shape[0]
    device = residual_stack.device
    z = _initial_alpha_logits(
        num_sources=num_sources,
        seed=seed,
        device=device,
        dtype=residual_stack.dtype,
    ).requires_grad_()
    optimizer = torch.optim.Adam([z], lr=learning_rate)
    uniform_alpha = torch.full(
        (num_sources,),
        1.0 / num_sources,
        dtype=residual_stack.dtype,
        device=device,
    )
    best_alpha = uniform_alpha.detach().clone()
    with torch.no_grad():
        best_loss = float(
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
    return best_alpha


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


def _target_probability(logits: torch.Tensor, *, target_token_id: int) -> float:
    probabilities = torch.softmax(logits, dim=-1)
    return float(probabilities[..., target_token_id].detach().cpu().item())


def _target_rank(logits: torch.Tensor, *, target_token_id: int) -> int:
    final_position_logits = logits.reshape(-1)
    target_logit = final_position_logits[target_token_id]
    rank = int((final_position_logits > target_logit).sum().detach().cpu().item()) + 1
    return rank


def _is_non_monotonic(values: Sequence[float]) -> bool:
    return any(right < left for left, right in zip(values, values[1:]))


def _source_layer(label: str) -> int:
    if label in {"embed", "pos_embed"}:
        return -1
    match = re.match(r"^(\d+)_", label)
    if match is None:
        raise ValueError(f"unsupported source label {label!r}")
    return int(match.group(1))


def build_routed_residual_traces(
    *,
    residual_stack: torch.Tensor,
    source_labels: Sequence[str],
    alpha: Sequence[float] | torch.Tensor,
    num_layers: int,
) -> torch.Tensor:
    if residual_stack.ndim != 3:
        raise ValueError("residual_stack must have shape [sources, pos, d_model]")
    if residual_stack.shape[0] != len(source_labels):
        raise ValueError("source_labels must align with residual_stack")

    alpha_tensor = torch.as_tensor(
        alpha,
        dtype=residual_stack.dtype,
        device=residual_stack.device,
    )
    if alpha_tensor.ndim != 1 or alpha_tensor.shape[0] != residual_stack.shape[0]:
        raise ValueError("alpha must have one weight per source")

    traces = []
    for layer in range(num_layers):
        mask_values = [_source_layer(label) <= layer for label in source_labels]
        mask = torch.tensor(mask_values, dtype=torch.bool, device=residual_stack.device)
        prefix_sources = residual_stack[mask]
        prefix_alpha = alpha_tensor[mask]
        if prefix_sources.shape[0] == 0:
            raise ValueError(f"layer {layer} has no prefix sources")
        prefix_total = float(prefix_alpha.sum().detach().cpu().item())
        if prefix_total <= 0.0:
            prefix_alpha = torch.full_like(prefix_alpha, 1.0 / prefix_alpha.numel())
        else:
            prefix_alpha = prefix_alpha / prefix_alpha.sum()
        traces.append(torch.einsum("s,spd->pd", prefix_alpha, prefix_sources))
    return torch.stack(traces)


def target_token_for_entry(
    *,
    model: HookedTransformer,
    entry: PromptEntry,
    prepend_bos: bool | None,
) -> tuple[int, str]:
    if not entry.target_text:
        raise ValueError(f"prompt {entry.prompt_id!r} is missing target_text")

    separator = "" if entry.text.endswith(" ") else " "
    prompt_tokens = model.to_tokens(entry.text, prepend_bos=prepend_bos)
    full_tokens = model.to_tokens(
        f"{entry.text}{separator}{entry.target_text}",
        prepend_bos=prepend_bos,
    )
    prompt_length = prompt_tokens.shape[-1]
    if full_tokens.shape[-1] <= prompt_length:
        raise ValueError(
            f"target text for prompt {entry.prompt_id!r} did not add a next token"
        )
    target_token_id = int(full_tokens[0, prompt_length].detach().cpu().item())
    target_token_text = model.to_string(torch.tensor([target_token_id]))
    return target_token_id, target_token_text


def summarize_tool_breakage_prompt(
    *,
    model_name: str,
    prompt_id: str,
    prompt: str,
    split: str,
    target_text: str,
    target_token_id: int,
    target_token_text: str,
    oracle_alpha: Sequence[float],
    source_labels: Sequence[str],
    final_logits: torch.Tensor,
    raw_original_logits: torch.Tensor,
    raw_routed_logits: torch.Tensor,
    tuned_original_logits: torch.Tensor,
    tuned_routed_logits: torch.Tensor,
) -> ToolBreakagePromptResult:
    if raw_original_logits.ndim != 3:
        raise ValueError("original logits must have shape [layers, pos, vocab]")
    if raw_routed_logits.shape != raw_original_logits.shape:
        raise ValueError("routed raw logits must match original raw logits shape")
    if tuned_original_logits.shape != raw_original_logits.shape:
        raise ValueError("tuned original logits must match original raw logits shape")
    if tuned_routed_logits.shape != raw_original_logits.shape:
        raise ValueError("tuned routed logits must match original raw logits shape")
    if final_logits.ndim != 2:
        raise ValueError("final_logits must have shape [pos, vocab]")

    final_position_index = final_logits.shape[0] - 1
    layer_traces: list[ToolBreakageLayerTrace] = []
    for layer in range(raw_original_logits.shape[0]):
        raw_original = raw_original_logits[layer]
        raw_routed = raw_routed_logits[layer]
        tuned_original = tuned_original_logits[layer]
        tuned_routed = tuned_routed_logits[layer]

        raw_original_final_position = raw_original[final_position_index]
        raw_routed_final_position = raw_routed[final_position_index]
        tuned_original_final_position = tuned_original[final_position_index]
        tuned_routed_final_position = tuned_routed[final_position_index]
        target_final_position = final_logits[final_position_index]

        layer_traces.append(
            ToolBreakageLayerTrace(
                layer=layer,
                raw_original_mean_kl_to_final=_mean_kl_to_final(
                    target_logits=final_logits,
                    candidate_logits=raw_original,
                ),
                raw_routed_mean_kl_to_final=_mean_kl_to_final(
                    target_logits=final_logits,
                    candidate_logits=raw_routed,
                ),
                tuned_original_mean_kl_to_final=_mean_kl_to_final(
                    target_logits=final_logits,
                    candidate_logits=tuned_original,
                ),
                tuned_routed_mean_kl_to_final=_mean_kl_to_final(
                    target_logits=final_logits,
                    candidate_logits=tuned_routed,
                ),
                raw_original_mean_top1_agreement=_top1_agreement(
                    target_logits=final_logits,
                    candidate_logits=raw_original,
                ),
                raw_routed_mean_top1_agreement=_top1_agreement(
                    target_logits=final_logits,
                    candidate_logits=raw_routed,
                ),
                tuned_original_mean_top1_agreement=_top1_agreement(
                    target_logits=final_logits,
                    candidate_logits=tuned_original,
                ),
                tuned_routed_mean_top1_agreement=_top1_agreement(
                    target_logits=final_logits,
                    candidate_logits=tuned_routed,
                ),
                raw_original_final_position_kl_to_final=_mean_kl_to_final(
                    target_logits=target_final_position.unsqueeze(0),
                    candidate_logits=raw_original_final_position.unsqueeze(0),
                ),
                raw_routed_final_position_kl_to_final=_mean_kl_to_final(
                    target_logits=target_final_position.unsqueeze(0),
                    candidate_logits=raw_routed_final_position.unsqueeze(0),
                ),
                tuned_original_final_position_kl_to_final=_mean_kl_to_final(
                    target_logits=target_final_position.unsqueeze(0),
                    candidate_logits=tuned_original_final_position.unsqueeze(0),
                ),
                tuned_routed_final_position_kl_to_final=_mean_kl_to_final(
                    target_logits=target_final_position.unsqueeze(0),
                    candidate_logits=tuned_routed_final_position.unsqueeze(0),
                ),
                raw_original_final_position_top1_agreement=_top1_agreement(
                    target_logits=target_final_position.unsqueeze(0),
                    candidate_logits=raw_original_final_position.unsqueeze(0),
                ),
                raw_routed_final_position_top1_agreement=_top1_agreement(
                    target_logits=target_final_position.unsqueeze(0),
                    candidate_logits=raw_routed_final_position.unsqueeze(0),
                ),
                tuned_original_final_position_top1_agreement=_top1_agreement(
                    target_logits=target_final_position.unsqueeze(0),
                    candidate_logits=tuned_original_final_position.unsqueeze(0),
                ),
                tuned_routed_final_position_top1_agreement=_top1_agreement(
                    target_logits=target_final_position.unsqueeze(0),
                    candidate_logits=tuned_routed_final_position.unsqueeze(0),
                ),
                raw_original_final_position_target_probability=_target_probability(
                    raw_original_final_position,
                    target_token_id=target_token_id,
                ),
                raw_routed_final_position_target_probability=_target_probability(
                    raw_routed_final_position,
                    target_token_id=target_token_id,
                ),
                tuned_original_final_position_target_probability=_target_probability(
                    tuned_original_final_position,
                    target_token_id=target_token_id,
                ),
                tuned_routed_final_position_target_probability=_target_probability(
                    tuned_routed_final_position,
                    target_token_id=target_token_id,
                ),
                raw_original_final_position_target_rank=_target_rank(
                    raw_original_final_position,
                    target_token_id=target_token_id,
                ),
                raw_routed_final_position_target_rank=_target_rank(
                    raw_routed_final_position,
                    target_token_id=target_token_id,
                ),
                tuned_original_final_position_target_rank=_target_rank(
                    tuned_original_final_position,
                    target_token_id=target_token_id,
                ),
                tuned_routed_final_position_target_rank=_target_rank(
                    tuned_routed_final_position,
                    target_token_id=target_token_id,
                ),
            )
        )

    raw_original_target_probabilities = [
        trace.raw_original_final_position_target_probability for trace in layer_traces
    ]
    raw_routed_target_probabilities = [
        trace.raw_routed_final_position_target_probability for trace in layer_traces
    ]
    tuned_original_target_probabilities = [
        trace.tuned_original_final_position_target_probability for trace in layer_traces
    ]
    tuned_routed_target_probabilities = [
        trace.tuned_routed_final_position_target_probability for trace in layer_traces
    ]

    return ToolBreakagePromptResult(
        prompt_id=prompt_id,
        prompt=prompt,
        split=split,
        target_text=target_text,
        target_token_id=target_token_id,
        target_token_text=target_token_text,
        source_labels=tuple(source_labels),
        oracle_alpha=tuple(float(value) for value in oracle_alpha),
        raw_original_non_monotonic=_is_non_monotonic(raw_original_target_probabilities),
        raw_routed_non_monotonic=_is_non_monotonic(raw_routed_target_probabilities),
        tuned_original_non_monotonic=_is_non_monotonic(
            tuned_original_target_probabilities
        ),
        tuned_routed_non_monotonic=_is_non_monotonic(tuned_routed_target_probabilities),
        layer_traces=tuple(layer_traces),
    )


def summarize_tool_breakage_run(
    *,
    model_name: str,
    collection_id: str,
    split: str,
    exploratory: bool = False,
    tuned_lens_checkpoint_path: str,
    prompt_results: Sequence[ToolBreakagePromptResult],
) -> ToolBreakageRunSummary:
    if not prompt_results:
        raise ValueError("prompt_results must not be empty")

    traces = [trace for result in prompt_results for trace in result.layer_traces]

    def mean(values: Sequence[float]) -> float:
        return float(sum(values) / len(values))

    return ToolBreakageRunSummary(
        model_name=model_name,
        collection_id=collection_id,
        split=split,
        exploratory=exploratory,
        tuned_lens_checkpoint_path=tuned_lens_checkpoint_path,
        num_prompts=len(prompt_results),
        mean_raw_original_kl_to_final=mean(
            [trace.raw_original_mean_kl_to_final for trace in traces]
        ),
        mean_raw_routed_kl_to_final=mean(
            [trace.raw_routed_mean_kl_to_final for trace in traces]
        ),
        mean_tuned_original_kl_to_final=mean(
            [trace.tuned_original_mean_kl_to_final for trace in traces]
        ),
        mean_tuned_routed_kl_to_final=mean(
            [trace.tuned_routed_mean_kl_to_final for trace in traces]
        ),
        mean_tuned_kl_increase_under_routing=mean(
            [
                trace.tuned_routed_mean_kl_to_final
                - trace.tuned_original_mean_kl_to_final
                for trace in traces
            ]
        ),
        final_position_mean_raw_original_kl_to_final=mean(
            [trace.raw_original_final_position_kl_to_final for trace in traces]
        ),
        final_position_mean_raw_routed_kl_to_final=mean(
            [trace.raw_routed_final_position_kl_to_final for trace in traces]
        ),
        final_position_mean_tuned_original_kl_to_final=mean(
            [trace.tuned_original_final_position_kl_to_final for trace in traces]
        ),
        final_position_mean_tuned_routed_kl_to_final=mean(
            [trace.tuned_routed_final_position_kl_to_final for trace in traces]
        ),
        final_position_mean_tuned_kl_increase_under_routing=mean(
            [
                trace.tuned_routed_final_position_kl_to_final
                - trace.tuned_original_final_position_kl_to_final
                for trace in traces
            ]
        ),
        mean_raw_original_top1_agreement=mean(
            [trace.raw_original_mean_top1_agreement for trace in traces]
        ),
        mean_raw_routed_top1_agreement=mean(
            [trace.raw_routed_mean_top1_agreement for trace in traces]
        ),
        mean_tuned_original_top1_agreement=mean(
            [trace.tuned_original_mean_top1_agreement for trace in traces]
        ),
        mean_tuned_routed_top1_agreement=mean(
            [trace.tuned_routed_mean_top1_agreement for trace in traces]
        ),
        final_position_mean_raw_original_top1_agreement=mean(
            [trace.raw_original_final_position_top1_agreement for trace in traces]
        ),
        final_position_mean_raw_routed_top1_agreement=mean(
            [trace.raw_routed_final_position_top1_agreement for trace in traces]
        ),
        final_position_mean_tuned_original_top1_agreement=mean(
            [trace.tuned_original_final_position_top1_agreement for trace in traces]
        ),
        final_position_mean_tuned_routed_top1_agreement=mean(
            [trace.tuned_routed_final_position_top1_agreement for trace in traces]
        ),
        fraction_raw_original_non_monotonic_prompts=mean(
            [
                1.0 if result.raw_original_non_monotonic else 0.0
                for result in prompt_results
            ]
        ),
        fraction_raw_routed_non_monotonic_prompts=mean(
            [
                1.0 if result.raw_routed_non_monotonic else 0.0
                for result in prompt_results
            ]
        ),
        fraction_tuned_original_non_monotonic_prompts=mean(
            [
                1.0 if result.tuned_original_non_monotonic else 0.0
                for result in prompt_results
            ]
        ),
        fraction_tuned_routed_non_monotonic_prompts=mean(
            [
                1.0 if result.tuned_routed_non_monotonic else 0.0
                for result in prompt_results
            ]
        ),
        prompt_results=tuple(prompt_results),
    )


def _load_tuned_lens_checkpoint(
    *,
    checkpoint_path: Path,
    device: str,
) -> LowRankAffineResidualLens:
    raw = torch.load(checkpoint_path, map_location=device)
    lens = LowRankAffineResidualLens(
        num_layers=int(raw["num_layers"]),
        d_model=int(raw["d_model"]),
        rank=int(raw["rank"]),
    ).to(device)
    lens.load_state_dict(raw["state_dict"])
    lens.eval()
    return lens


def _tool_breakage_cache(
    *,
    model: HookedTransformer,
    prompt: str,
    prepend_bos: bool | None,
) -> tuple[torch.Tensor, tuple[str, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    with torch.no_grad():
        logits, cache = model.run_with_cache(
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
        original_resid_post = torch.stack(
            [cache[("resid_post", layer)][0] for layer in range(model.cfg.n_layers)]
        )
    return (
        residual_stack[:, 0],
        tuple(labels),
        original_resid_post,
        logits[0],
        tokens[0],
    )


def _prompt_result_from_cache(
    raw: Mapping[str, object],
) -> ToolBreakagePromptResult:
    return ToolBreakagePromptResult(
        prompt_id=str(raw["prompt_id"]),
        prompt=str(raw["prompt"]),
        split=str(raw["split"]),
        target_text=str(raw["target_text"]),
        target_token_id=int(raw["target_token_id"]),
        target_token_text=str(raw["target_token_text"]),
        source_labels=tuple(str(value) for value in raw["source_labels"]),
        oracle_alpha=tuple(float(value) for value in raw["oracle_alpha"]),
        raw_original_non_monotonic=bool(raw["raw_original_non_monotonic"]),
        raw_routed_non_monotonic=bool(raw["raw_routed_non_monotonic"]),
        tuned_original_non_monotonic=bool(raw["tuned_original_non_monotonic"]),
        tuned_routed_non_monotonic=bool(raw["tuned_routed_non_monotonic"]),
        layer_traces=tuple(
            ToolBreakageLayerTrace(
                layer=int(trace["layer"]),
                raw_original_mean_kl_to_final=float(
                    trace["raw_original_mean_kl_to_final"]
                ),
                raw_routed_mean_kl_to_final=float(trace["raw_routed_mean_kl_to_final"]),
                tuned_original_mean_kl_to_final=float(
                    trace["tuned_original_mean_kl_to_final"]
                ),
                tuned_routed_mean_kl_to_final=float(
                    trace["tuned_routed_mean_kl_to_final"]
                ),
                raw_original_mean_top1_agreement=float(
                    trace["raw_original_mean_top1_agreement"]
                ),
                raw_routed_mean_top1_agreement=float(
                    trace["raw_routed_mean_top1_agreement"]
                ),
                tuned_original_mean_top1_agreement=float(
                    trace["tuned_original_mean_top1_agreement"]
                ),
                tuned_routed_mean_top1_agreement=float(
                    trace["tuned_routed_mean_top1_agreement"]
                ),
                raw_original_final_position_kl_to_final=float(
                    trace["raw_original_final_position_kl_to_final"]
                ),
                raw_routed_final_position_kl_to_final=float(
                    trace["raw_routed_final_position_kl_to_final"]
                ),
                tuned_original_final_position_kl_to_final=float(
                    trace["tuned_original_final_position_kl_to_final"]
                ),
                tuned_routed_final_position_kl_to_final=float(
                    trace["tuned_routed_final_position_kl_to_final"]
                ),
                raw_original_final_position_top1_agreement=float(
                    trace["raw_original_final_position_top1_agreement"]
                ),
                raw_routed_final_position_top1_agreement=float(
                    trace["raw_routed_final_position_top1_agreement"]
                ),
                tuned_original_final_position_top1_agreement=float(
                    trace["tuned_original_final_position_top1_agreement"]
                ),
                tuned_routed_final_position_top1_agreement=float(
                    trace["tuned_routed_final_position_top1_agreement"]
                ),
                raw_original_final_position_target_probability=float(
                    trace["raw_original_final_position_target_probability"]
                ),
                raw_routed_final_position_target_probability=float(
                    trace["raw_routed_final_position_target_probability"]
                ),
                tuned_original_final_position_target_probability=float(
                    trace["tuned_original_final_position_target_probability"]
                ),
                tuned_routed_final_position_target_probability=float(
                    trace["tuned_routed_final_position_target_probability"]
                ),
                raw_original_final_position_target_rank=int(
                    trace["raw_original_final_position_target_rank"]
                ),
                raw_routed_final_position_target_rank=int(
                    trace["raw_routed_final_position_target_rank"]
                ),
                tuned_original_final_position_target_rank=int(
                    trace["tuned_original_final_position_target_rank"]
                ),
                tuned_routed_final_position_target_rank=int(
                    trace["tuned_routed_final_position_target_rank"]
                ),
            )
            for trace in raw["layer_traces"]
        ),
    )


def _load_prompt_result_cache(
    checkpoint_dir: Path,
) -> dict[str, ToolBreakagePromptResult]:
    if not checkpoint_dir.is_dir():
        return {}

    loaded = {}
    for path in sorted(checkpoint_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        loaded[str(raw["prompt_id"])] = _prompt_result_from_cache(raw)
    return loaded


def _save_prompt_result(
    checkpoint_dir: Path,
    result: ToolBreakagePromptResult,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{_safe_key(result.prompt_id)}.json"
    path.write_text(json.dumps(asdict(result), indent=2) + "\n")


def save_tool_breakage_artifacts(
    *,
    output_dir: Path,
    summary: ToolBreakageRunSummary,
) -> ToolBreakageRunSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2) + "\n"
    )
    return summary


def run_tool_breakage_factual_recall_baseline(
    *,
    model: HookedTransformer,
    tuned_lens_checkpoint_path: Path,
    collection_id: str,
    split: str,
    exploratory: bool,
    output_dir: Path,
    optimization_steps: int,
    learning_rate: float,
    seed: int,
    prepend_bos: bool | None = None,
    max_prompts: int | None = None,
) -> ToolBreakageRunSummary:
    prompt_entries = list(
        resolve_prompt_entries(
            collection_id=collection_id,
            split=split,
            exploratory=exploratory,
        )
    )
    if max_prompts is not None:
        prompt_entries = prompt_entries[:max_prompts]
    if not prompt_entries:
        raise ValueError("at least one prompt entry is required")

    lens = _load_tuned_lens_checkpoint(
        checkpoint_path=tuned_lens_checkpoint_path,
        device=str(model.cfg.device),
    )
    checkpoint_dir = output_dir / "checkpoints" / "prompt_results"
    cached_results = _load_prompt_result_cache(checkpoint_dir)

    prompt_results = []
    for index, entry in enumerate(prompt_entries):
        cached = cached_results.get(entry.prompt_id)
        if cached is not None:
            prompt_results.append(cached)
            continue

        target_token_id, target_token_text = target_token_for_entry(
            model=model,
            entry=entry,
            prepend_bos=prepend_bos,
        )
        residual_stack, source_labels, original_resid_post, final_logits, tokens = (
            _tool_breakage_cache(
                model=model,
                prompt=entry.text,
                prepend_bos=prepend_bos,
            )
        )
        original_resid_post = original_resid_post.to(str(model.cfg.device))
        residual_stack = residual_stack.to(str(model.cfg.device))
        final_logits = final_logits.to(str(model.cfg.device))
        tokens = tokens.to(str(model.cfg.device))
        oracle_alpha = _optimize_oracle_alpha(
            model=model,
            residual_stack=residual_stack,
            tokens=tokens,
            optimization_steps=optimization_steps,
            learning_rate=learning_rate,
            seed=seed + index,
        )
        routed_resid_post = build_routed_residual_traces(
            residual_stack=residual_stack,
            source_labels=source_labels,
            alpha=oracle_alpha,
            num_layers=model.cfg.n_layers,
        )

        with torch.no_grad():
            tuned_original_resid_post = torch.stack(
                [
                    lens.forward_layer(layer, original_resid_post[layer])
                    for layer in range(model.cfg.n_layers)
                ]
            )
            tuned_routed_resid_post = torch.stack(
                [
                    lens.forward_layer(layer, routed_resid_post[layer])
                    for layer in range(model.cfg.n_layers)
                ]
            )
            prompt_result = summarize_tool_breakage_prompt(
                model_name=model.cfg.model_name,
                prompt_id=entry.prompt_id,
                prompt=entry.text,
                split=entry.split,
                target_text=entry.target_text or "",
                target_token_id=target_token_id,
                target_token_text=target_token_text,
                oracle_alpha=tuple(
                    float(value) for value in oracle_alpha.cpu().tolist()
                ),
                source_labels=source_labels,
                final_logits=final_logits,
                raw_original_logits=_apply_final_norm_and_unembed(
                    model,
                    original_resid_post,
                ),
                raw_routed_logits=_apply_final_norm_and_unembed(
                    model,
                    routed_resid_post,
                ),
                tuned_original_logits=_apply_final_norm_and_unembed(
                    model,
                    tuned_original_resid_post,
                ),
                tuned_routed_logits=_apply_final_norm_and_unembed(
                    model,
                    tuned_routed_resid_post,
                ),
            )
        _save_prompt_result(checkpoint_dir, prompt_result)
        prompt_results.append(prompt_result)

    summary = summarize_tool_breakage_run(
        model_name=model.cfg.model_name,
        collection_id=collection_id,
        split=split,
        exploratory=exploratory,
        tuned_lens_checkpoint_path=str(tuned_lens_checkpoint_path),
        prompt_results=tuple(prompt_results),
    )
    return save_tool_breakage_artifacts(output_dir=output_dir, summary=summary)
