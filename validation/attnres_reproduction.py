# ABOUTME: Implements the smallest local Block AttnRes proxy and matched baseline for Figure 8 validation.
# ABOUTME: Keeps the first reproduction slice focused on stable training scaffolding and explicit routing export.

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AttnResProxyConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    dropout: float = 0.0
    num_blocks: int | None = None

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if self.num_blocks is None:
            object.__setattr__(self, "num_blocks", self.n_layers)
        if self.num_blocks != self.n_layers:
            raise ValueError(
                "the first reproduction slice requires one routing block per transformer layer"
            )


@dataclass(frozen=True)
class RoutingSnapshot:
    target_label: str
    source_labels: tuple[str, ...]
    alpha: torch.Tensor


@dataclass(frozen=True)
class TinyLMTrainingSummary:
    model_label: str
    final_train_loss: float
    final_eval_loss: float
    best_eval_loss: float
    num_steps: int
    batch_size: int
    learning_rate: float
    checkpoint_path: str | None


@dataclass(frozen=True)
class Figure8TargetSummary:
    target_label: str
    immediate_predecessor_source: str
    locality_score: float
    embedding_weight: float
    entropy: float
    mean_alpha_by_source: tuple[tuple[str, float], ...]
    skip_connection_peak_fractions: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Figure8ProxyMetrics:
    target_summaries: tuple[Figure8TargetSummary, ...]
    mean_pre_attn_entropy: float
    mean_pre_mlp_entropy: float
    deep_embedding_persistence: float


@dataclass(frozen=True)
class AttnResProxyViabilitySummary:
    dataset_name: str
    tokenizer_mode: str
    proxy_config: AttnResProxyConfig
    sequence_length: int
    num_train_examples: int
    num_eval_examples: int
    baseline_summary: TinyLMTrainingSummary
    attnres_summary: TinyLMTrainingSummary
    attnres_routing_entropy_by_target: tuple[tuple[str, float], ...]
    figure8_proxy_metrics: Figure8ProxyMetrics


class _CausalSelfAttention(torch.nn.Module):
    def __init__(self, config: AttnResProxyConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = torch.nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = torch.nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = hidden_states.shape
        qkv = self.qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(
            1, 2
        )
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(
            1, 2
        )
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        return self.out_proj(attn_output)


class _FeedForward(torch.nn.Module):
    def __init__(self, config: AttnResProxyConfig) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = torch.nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.up_proj(hidden_states))
        hidden_states = self.down_proj(hidden_states)
        return self.dropout(hidden_states)


class _StandardResidualLayer(torch.nn.Module):
    def __init__(self, config: AttnResProxyConfig) -> None:
        super().__init__()
        self.attn_norm = torch.nn.RMSNorm(config.d_model)
        self.attn = _CausalSelfAttention(config)
        self.mlp_norm = torch.nn.RMSNorm(config.d_model)
        self.mlp = _FeedForward(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class _BlockAttnResLayer(torch.nn.Module):
    def __init__(self, config: AttnResProxyConfig) -> None:
        super().__init__()
        self.attn_norm = torch.nn.RMSNorm(config.d_model)
        self.attn = _CausalSelfAttention(config)
        self.mlp_norm = torch.nn.RMSNorm(config.d_model)
        self.mlp = _FeedForward(config)
        self.attn_res_norm = torch.nn.RMSNorm(config.d_model)
        self.mlp_res_norm = torch.nn.RMSNorm(config.d_model)
        self.attn_query = torch.nn.Parameter(torch.zeros(config.d_model))
        self.mlp_query = torch.nn.Parameter(torch.zeros(config.d_model))
        torch.nn.init.normal_(self.attn_query, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.mlp_query, mean=0.0, std=0.02)


def _depth_route(
    *,
    sources: Sequence[torch.Tensor],
    source_labels: Sequence[str],
    query: torch.Tensor,
    norm: torch.nn.RMSNorm,
    target_label: str,
    return_routing: bool,
) -> tuple[torch.Tensor, RoutingSnapshot | None]:
    stacked_sources = torch.stack(tuple(sources), dim=0)
    normalized_keys = norm(stacked_sources)
    logits = torch.einsum("d,sbtd->sbt", query, normalized_keys)
    alpha = torch.softmax(logits, dim=0)
    hidden_states = torch.einsum("sbt,sbtd->btd", alpha, stacked_sources)
    if not return_routing:
        return hidden_states, None
    return hidden_states, RoutingSnapshot(
        target_label=target_label,
        source_labels=tuple(source_labels),
        alpha=alpha.detach().cpu(),
    )


class BlockAttnResTinyLM(torch.nn.Module):
    def __init__(self, config: AttnResProxyConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = torch.nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = torch.nn.ModuleList(
            [_BlockAttnResLayer(config) for _ in range(config.n_layers)]
        )
        self.final_res_norm = torch.nn.RMSNorm(config.d_model)
        self.final_query = torch.nn.Parameter(torch.zeros(config.d_model))
        self.final_norm = torch.nn.RMSNorm(config.d_model)
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
        torch.nn.init.normal_(self.final_query, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_routing: bool = False,
    ) -> tuple[torch.Tensor, tuple[RoutingSnapshot, ...]]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, seq]")
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence length exceeds config.max_seq_len")

        positions = torch.arange(tokens.shape[1], device=tokens.device)
        embed = self.token_embedding(tokens) + self.position_embedding(positions)
        completed_blocks: list[torch.Tensor] = []
        routing_steps: list[RoutingSnapshot] = []

        for layer_index, layer in enumerate(self.layers):
            pre_attn_sources = [embed] + completed_blocks
            pre_attn_labels = ["embed"] + [
                f"block_{index}" for index in range(len(completed_blocks))
            ]
            hidden_states, routing = _depth_route(
                sources=pre_attn_sources,
                source_labels=pre_attn_labels,
                query=layer.attn_query,
                norm=layer.attn_res_norm,
                target_label=f"{layer_index}_pre_attn",
                return_routing=return_routing,
            )
            if routing is not None:
                routing_steps.append(routing)

            attn_out = layer.attn(layer.attn_norm(hidden_states))

            pre_mlp_sources = pre_attn_sources + [attn_out]
            pre_mlp_labels = pre_attn_labels + [f"block_{layer_index}_attn"]
            hidden_states, routing = _depth_route(
                sources=pre_mlp_sources,
                source_labels=pre_mlp_labels,
                query=layer.mlp_query,
                norm=layer.mlp_res_norm,
                target_label=f"{layer_index}_pre_mlp",
                return_routing=return_routing,
            )
            if routing is not None:
                routing_steps.append(routing)

            mlp_out = layer.mlp(layer.mlp_norm(hidden_states))
            completed_blocks.append(attn_out + mlp_out)

        final_sources = [embed] + completed_blocks
        final_labels = ["embed"] + [
            f"block_{index}" for index in range(len(completed_blocks))
        ]
        hidden_states, routing = _depth_route(
            sources=final_sources,
            source_labels=final_labels,
            query=self.final_query,
            norm=self.final_res_norm,
            target_label="final_output",
            return_routing=return_routing,
        )
        if routing is not None:
            routing_steps.append(routing)
        logits = self.lm_head(self.final_norm(hidden_states))
        return logits, tuple(routing_steps)


class StandardResidualTinyLM(torch.nn.Module):
    def __init__(self, config: AttnResProxyConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = torch.nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = torch.nn.ModuleList(
            [_StandardResidualLayer(config) for _ in range(config.n_layers)]
        )
        self.final_norm = torch.nn.RMSNorm(config.d_model)
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_routing: bool = False,
    ) -> tuple[torch.Tensor, tuple[RoutingSnapshot, ...]]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, seq]")
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence length exceeds config.max_seq_len")

        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden_states = self.token_embedding(tokens) + self.position_embedding(
            positions
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        logits = self.lm_head(self.final_norm(hidden_states))
        if not return_routing:
            return logits, ()
        return logits, ()


def build_matched_language_models(
    *,
    config: AttnResProxyConfig,
    seed: int,
) -> tuple[StandardResidualTinyLM, BlockAttnResTinyLM]:
    torch.manual_seed(seed)
    baseline = StandardResidualTinyLM(config)
    torch.manual_seed(seed)
    attnres = BlockAttnResTinyLM(config)
    return baseline, attnres


def build_next_token_examples(
    *,
    token_ids: Sequence[int],
    sequence_length: int,
) -> torch.Tensor:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    window = sequence_length + 1
    examples = []
    for start in range(0, len(token_ids) - window + 1, sequence_length):
        examples.append(token_ids[start : start + window])
    if not examples:
        raise ValueError("token_ids do not contain a full next-token example window")
    return torch.tensor(examples, dtype=torch.long)


def next_token_loss(
    *,
    logits: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq, vocab]")
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, seq_plus_one]")
    if logits.shape[:2] != tokens[:, :-1].shape:
        raise ValueError("logits must align with tokens[:, :-1]")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        tokens[:, 1:].reshape(-1),
    )


def evaluate_language_model(
    *,
    model: torch.nn.Module,
    examples: torch.Tensor,
    batch_size: int,
    device: str,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, examples.shape[0], batch_size):
            batch = examples[start : start + batch_size].to(device)
            logits, _ = model(batch[:, :-1], return_routing=False)
            losses.append(float(next_token_loss(logits=logits, tokens=batch).item()))
    return float(sum(losses) / len(losses))


def train_language_model(
    *,
    model: torch.nn.Module,
    model_label: str,
    train_examples: torch.Tensor,
    eval_examples: torch.Tensor,
    batch_size: int,
    num_steps: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 50,
) -> TinyLMTrainingSummary:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    start_step = 0
    best_eval_loss = float("inf")
    final_train_loss = float("nan")
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = int(checkpoint["step"])
            best_eval_loss = float(checkpoint["best_eval_loss"])

    model.train()
    for step in range(start_step, num_steps):
        indices = torch.randint(
            train_examples.shape[0],
            (batch_size,),
            generator=generator,
        )
        batch = train_examples[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch[:, :-1], return_routing=False)
        loss = next_token_loss(logits=logits, tokens=batch)
        loss.backward()
        optimizer.step()
        final_train_loss = float(loss.detach().cpu().item())

        if checkpoint_path is not None and (
            (step + 1) % checkpoint_interval == 0 or step + 1 == num_steps
        ):
            current_eval_loss = evaluate_language_model(
                model=model,
                examples=eval_examples,
                batch_size=batch_size,
                device=device,
            )
            best_eval_loss = min(best_eval_loss, current_eval_loss)
            torch.save(
                {
                    "step": step + 1,
                    "best_eval_loss": best_eval_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            model.train()

    final_eval_loss = evaluate_language_model(
        model=model,
        examples=eval_examples,
        batch_size=batch_size,
        device=device,
    )
    best_eval_loss = min(best_eval_loss, final_eval_loss)
    if checkpoint_path is not None:
        torch.save(
            {
                "step": num_steps,
                "best_eval_loss": best_eval_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )

    return TinyLMTrainingSummary(
        model_label=model_label,
        final_train_loss=final_train_loss,
        final_eval_loss=final_eval_loss,
        best_eval_loss=best_eval_loss,
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
    )


def summarize_routing_entropy(
    *,
    model: BlockAttnResTinyLM,
    examples: torch.Tensor,
    batch_size: int,
    device: str,
) -> tuple[tuple[str, float], ...]:
    model = model.to(device)
    model.eval()
    batch = examples[:batch_size].to(device)
    with torch.no_grad():
        _, routing_steps = model(batch[:, :-1], return_routing=True)

    summaries = []
    for step in routing_steps:
        alpha = step.alpha.clamp_min(1e-12)
        entropy = -(alpha * alpha.log()).sum(dim=0).mean()
        summaries.append((step.target_label, float(entropy.item())))
    return tuple(summaries)


def save_attnres_proxy_summary(
    *,
    output_dir: Path,
    summary: AttnResProxyViabilitySummary,
) -> AttnResProxyViabilitySummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2) + "\n"
    )
    return summary


def _sorted_character_vocab(texts: Sequence[str]) -> dict[str, int]:
    joined_text = "\n\n".join(texts)
    vocabulary = sorted(set(joined_text))
    if not vocabulary:
        raise ValueError("texts must contain at least one character")
    return {character: index for index, character in enumerate(vocabulary)}


def _encode_character_texts(
    *,
    texts: Sequence[str],
    vocabulary: dict[str, int],
) -> list[int]:
    if not texts:
        raise ValueError("texts must contain at least one element")
    joined_text = "\n\n".join(texts)
    return [vocabulary[character] for character in joined_text]


def _immediate_predecessor_source(
    target_label: str, source_labels: Sequence[str]
) -> str:
    if target_label == "final_output":
        return source_labels[-1]
    layer_prefix, target_kind = target_label.split("_", maxsplit=1)
    layer_index = int(layer_prefix)
    if target_kind == "pre_attn":
        if layer_index == 0:
            return "embed"
        return f"block_{layer_index - 1}"
    if target_kind == "pre_mlp":
        return f"block_{layer_index}_attn"
    raise ValueError(f"unsupported target label: {target_label}")


def summarize_figure8_proxy_metrics(
    *,
    routing_steps: Sequence[RoutingSnapshot],
) -> Figure8ProxyMetrics:
    if not routing_steps:
        raise ValueError("routing_steps must not be empty")

    target_summaries = []
    pre_attn_entropies = []
    pre_mlp_entropies = []
    deep_embedding_weights = []
    layer_indices = [
        int(step.target_label.split("_", maxsplit=1)[0])
        for step in routing_steps
        if step.target_label != "final_output"
    ]
    num_layers = max(layer_indices) + 1 if layer_indices else 0
    deep_layer_start = max(0, num_layers // 2)

    for step in routing_steps:
        alpha = step.alpha.clamp_min(1e-12)
        mean_alpha = alpha.mean(dim=(1, 2))
        entropy = float((-(alpha * alpha.log()).sum(dim=0).mean()).item())
        immediate_predecessor = _immediate_predecessor_source(
            step.target_label,
            step.source_labels,
        )
        locality_index = step.source_labels.index(immediate_predecessor)
        locality_score = float(mean_alpha[locality_index].item())
        embedding_weight = 0.0
        if "embed" in step.source_labels:
            embedding_weight = float(
                mean_alpha[step.source_labels.index("embed")].item()
            )
        threshold = 2.0 / len(step.source_labels)
        skip_connection_peak_fractions = []
        for source_index, source_label in enumerate(step.source_labels):
            if source_label == immediate_predecessor:
                continue
            peak_fraction = float(
                (alpha[source_index] > threshold).float().mean().item()
            )
            if peak_fraction > 0.0:
                skip_connection_peak_fractions.append((source_label, peak_fraction))

        if step.target_label.endswith("pre_attn") and len(step.source_labels) > 1:
            pre_attn_entropies.append(entropy)
        if step.target_label.endswith("pre_mlp") and len(step.source_labels) > 1:
            pre_mlp_entropies.append(entropy)
        if step.target_label == "final_output":
            deep_embedding_weights.append(embedding_weight)
        elif int(step.target_label.split("_", maxsplit=1)[0]) >= deep_layer_start:
            deep_embedding_weights.append(embedding_weight)

        target_summaries.append(
            Figure8TargetSummary(
                target_label=step.target_label,
                immediate_predecessor_source=immediate_predecessor,
                locality_score=locality_score,
                embedding_weight=embedding_weight,
                entropy=entropy,
                mean_alpha_by_source=tuple(
                    (source_label, float(mean_alpha[source_index].item()))
                    for source_index, source_label in enumerate(step.source_labels)
                ),
                skip_connection_peak_fractions=tuple(skip_connection_peak_fractions),
            )
        )

    return Figure8ProxyMetrics(
        target_summaries=tuple(target_summaries),
        mean_pre_attn_entropy=float(sum(pre_attn_entropies) / len(pre_attn_entropies)),
        mean_pre_mlp_entropy=float(sum(pre_mlp_entropies) / len(pre_mlp_entropies)),
        deep_embedding_persistence=float(
            sum(deep_embedding_weights) / len(deep_embedding_weights)
        ),
    )


def run_attnres_proxy_viability_from_texts(
    *,
    config: AttnResProxyConfig,
    dataset_name: str,
    train_texts: Sequence[str],
    eval_texts: Sequence[str],
    output_dir: Path,
    batch_size: int,
    num_steps: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
    checkpoint_interval: int = 50,
) -> AttnResProxyViabilitySummary:
    all_texts = tuple(train_texts) + tuple(eval_texts)
    vocabulary = _sorted_character_vocab(all_texts)
    if len(vocabulary) > config.vocab_size:
        raise ValueError(
            f"character vocabulary size {len(vocabulary)} exceeds config.vocab_size={config.vocab_size}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "character_vocabulary.json").write_text(
        json.dumps(vocabulary, indent=2, sort_keys=True) + "\n"
    )

    train_examples = build_next_token_examples(
        token_ids=_encode_character_texts(texts=train_texts, vocabulary=vocabulary),
        sequence_length=config.max_seq_len,
    )
    eval_examples = build_next_token_examples(
        token_ids=_encode_character_texts(texts=eval_texts, vocabulary=vocabulary),
        sequence_length=config.max_seq_len,
    )

    baseline, attnres = build_matched_language_models(config=config, seed=seed)
    baseline_summary = train_language_model(
        model=baseline,
        model_label="standard_residual_baseline",
        train_examples=train_examples,
        eval_examples=eval_examples,
        batch_size=batch_size,
        num_steps=num_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
        checkpoint_path=checkpoints_dir / "baseline_training_state.pt",
        checkpoint_interval=checkpoint_interval,
    )
    attnres_summary = train_language_model(
        model=attnres,
        model_label="block_attnres_proxy",
        train_examples=train_examples,
        eval_examples=eval_examples,
        batch_size=batch_size,
        num_steps=num_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
        checkpoint_path=checkpoints_dir / "attnres_training_state.pt",
        checkpoint_interval=checkpoint_interval,
    )

    attnres = attnres.to(device)
    attnres.eval()
    routing_by_target: dict[str, list[torch.Tensor]] = {}
    source_labels_by_target: dict[str, tuple[str, ...]] = {}
    with torch.no_grad():
        for start in range(0, eval_examples.shape[0], batch_size):
            batch = eval_examples[start : start + batch_size].to(device)
            _, routing_steps = attnres(batch[:, :-1], return_routing=True)
            for step in routing_steps:
                routing_by_target.setdefault(step.target_label, []).append(step.alpha)
                source_labels_by_target.setdefault(
                    step.target_label, step.source_labels
                )

    ordered_routing_steps = []
    for target_label in list(routing_by_target.keys()):
        ordered_routing_steps.append(
            RoutingSnapshot(
                target_label=target_label,
                source_labels=source_labels_by_target[target_label],
                alpha=torch.cat(routing_by_target[target_label], dim=1),
            )
        )
    figure8_proxy_metrics = summarize_figure8_proxy_metrics(
        routing_steps=tuple(ordered_routing_steps)
    )

    summary = AttnResProxyViabilitySummary(
        dataset_name=dataset_name,
        tokenizer_mode="character",
        proxy_config=config,
        sequence_length=config.max_seq_len,
        num_train_examples=int(train_examples.shape[0]),
        num_eval_examples=int(eval_examples.shape[0]),
        baseline_summary=baseline_summary,
        attnres_summary=attnres_summary,
        attnres_routing_entropy_by_target=tuple(
            (target_summary.target_label, target_summary.entropy)
            for target_summary in figure8_proxy_metrics.target_summaries
        ),
        figure8_proxy_metrics=figure8_proxy_metrics,
    )
    return save_attnres_proxy_summary(output_dir=output_dir, summary=summary)
