# ABOUTME: Exports pilot-only per-token hidden states joined to saved oracle targets for Phase 6 router distillation.
# ABOUTME: Uses resumable per-prompt checkpoints so Gemma pilot export can be rerun without recomputing finished prompts.

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

import torch
from transformer_lens import HookedTransformer

from prompts import PromptEntry, resolve_prompt_entries


@dataclass(frozen=True)
class RouterDistillationPromptExportSummary:
    prompt_id: str
    split: str
    prompt: str
    target_text: str | None
    tags: tuple[str, ...]
    perturbation_names: tuple[str, ...]
    checkpoint_path: str
    num_tokens: int
    d_model: int
    num_sources: int


@dataclass(frozen=True)
class RouterDistillationPilotExportSummary:
    model_name: str
    collection_id: str
    split: str
    num_prompts: int
    exported_fields: tuple[str, ...]
    checkpoint_dir: str
    manifest_path: str
    prompt_summaries: tuple[RouterDistillationPromptExportSummary, ...]


@dataclass(frozen=True)
class _OracleSequenceCheckpoint:
    prompt_id: str
    prompt: str
    split: str
    num_sources: int
    source_labels: tuple[str, ...]
    final_alpha: tuple[float, ...]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _safe_key(raw: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    if not normalized:
        normalized = "artifact"
    return f"{normalized}-{digest}"


def _load_oracle_sequence_checkpoints(
    checkpoint_dir: Path,
) -> dict[str, _OracleSequenceCheckpoint]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"oracle checkpoint directory does not exist: {checkpoint_dir}"
        )

    loaded = {}
    for path in sorted(checkpoint_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        loaded[raw["prompt_id"]] = _OracleSequenceCheckpoint(
            prompt_id=raw["prompt_id"],
            prompt=raw["prompt"],
            split=raw["split"],
            num_sources=int(raw["num_sources"]),
            source_labels=tuple(raw["source_labels"]),
            final_alpha=tuple(float(value) for value in raw["final_alpha"]),
        )
    if not loaded:
        raise ValueError(f"no oracle checkpoints found under {checkpoint_dir}")
    return loaded


def _token_level_resid_post_states(
    *,
    model: HookedTransformer,
    prompt: str,
    prepend_bos: bool | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            return_type="logits",
            names_filter=lambda name: name.endswith("hook_resid_post"),
        )
    token_ids = tokens[0].detach().cpu().to(dtype=torch.long)
    h_1 = cache[("resid_post", 0)][0].detach().cpu().to(dtype=torch.float32)
    h_4 = (
        cache[("resid_post", min(3, model.cfg.n_layers - 1))][0]
        .detach()
        .cpu()
        .to(dtype=torch.float32)
    )
    return token_ids, h_1, h_4


def _compute_prompt_export_payload(
    *,
    model: HookedTransformer,
    entry: PromptEntry,
    oracle_checkpoint: _OracleSequenceCheckpoint,
    prepend_bos: bool | None,
) -> dict[str, Any]:
    token_ids, h_1, h_4 = _token_level_resid_post_states(
        model=model,
        prompt=entry.text,
        prepend_bos=prepend_bos,
    )
    final_alpha = torch.tensor(oracle_checkpoint.final_alpha, dtype=torch.float32)
    if final_alpha.shape[0] != oracle_checkpoint.num_sources:
        raise ValueError(
            f"oracle checkpoint for {entry.prompt_id!r} has mismatched source count"
        )
    return {
        "prompt_id": entry.prompt_id,
        "prompt": entry.text,
        "split": entry.split,
        "target_text": entry.target_text,
        "tags": tuple(entry.tags),
        "perturbations": dict(entry.perturbations),
        "token_ids": token_ids,
        "h_1[t]": h_1,
        "h_4[t]": h_4,
        "source_labels": tuple(oracle_checkpoint.source_labels),
        "final_alpha": final_alpha,
    }


def _load_prompt_export_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _save_prompt_export_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _summary_from_payload(
    *,
    payload: dict[str, Any],
    checkpoint_path: Path,
) -> RouterDistillationPromptExportSummary:
    h_1 = payload["h_1[t]"]
    final_alpha = payload["final_alpha"]
    return RouterDistillationPromptExportSummary(
        prompt_id=payload["prompt_id"],
        split=payload["split"],
        prompt=payload["prompt"],
        target_text=payload["target_text"],
        tags=tuple(payload["tags"]),
        perturbation_names=tuple(sorted(payload["perturbations"].keys())),
        checkpoint_path=str(checkpoint_path),
        num_tokens=int(payload["token_ids"].shape[0]),
        d_model=int(h_1.shape[-1]),
        num_sources=int(final_alpha.shape[0]),
    )


def _payload_has_required_fields(payload: dict[str, Any]) -> bool:
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
    return required_fields.issubset(payload.keys())


def _manifest_payload(
    *,
    collection_id: str,
    model_name: str,
    output_dir: Path,
    checkpoint_dir: Path,
    prompt_summaries: Sequence[RouterDistillationPromptExportSummary],
) -> dict[str, Any]:
    return {
        "collection_id": collection_id,
        "model_name": model_name,
        "split": "pilot",
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "dataset_fields": [
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
        ],
        "entries": [asdict(summary) for summary in prompt_summaries],
    }


def run_router_distillation_pilot_export(
    *,
    model: HookedTransformer,
    collection_id: str,
    oracle_checkpoint_dir: Path,
    output_dir: Path,
    max_sequences: int | None = None,
    prepend_bos: bool | None = None,
) -> RouterDistillationPilotExportSummary:
    prompt_entries = list(
        resolve_prompt_entries(
            collection_id=collection_id,
            split="pilot",
            exploratory=True,
        )
    )
    if max_sequences is not None:
        prompt_entries = prompt_entries[:max_sequences]
    if not prompt_entries:
        raise ValueError("pilot export requires at least one prompt entry")

    oracle_by_prompt_id = _load_oracle_sequence_checkpoints(oracle_checkpoint_dir)
    checkpoint_dir = output_dir / "checkpoints" / "prompt_exports"
    prompt_summaries = []
    for entry in prompt_entries:
        try:
            oracle_checkpoint = oracle_by_prompt_id[entry.prompt_id]
        except KeyError as error:
            raise ValueError(
                f"missing oracle checkpoint for prompt {entry.prompt_id!r}"
            ) from error

        checkpoint_path = checkpoint_dir / f"{_safe_key(entry.prompt_id)}.pt"
        if checkpoint_path.is_file():
            payload = _load_prompt_export_checkpoint(checkpoint_path)
            if not _payload_has_required_fields(payload):
                payload = _compute_prompt_export_payload(
                    model=model,
                    entry=entry,
                    oracle_checkpoint=oracle_checkpoint,
                    prepend_bos=prepend_bos,
                )
                _save_prompt_export_checkpoint(checkpoint_path, payload)
        else:
            payload = _compute_prompt_export_payload(
                model=model,
                entry=entry,
                oracle_checkpoint=oracle_checkpoint,
                prepend_bos=prepend_bos,
            )
            _save_prompt_export_checkpoint(checkpoint_path, payload)
        prompt_summaries.append(
            _summary_from_payload(payload=payload, checkpoint_path=checkpoint_path)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset_manifest.json"
    summary_path = output_dir / "summary.json"
    manifest = _manifest_payload(
        collection_id=collection_id,
        model_name=model.cfg.model_name,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        prompt_summaries=prompt_summaries,
    )
    _write_json(manifest_path, manifest)
    _write_json(
        summary_path,
        {
            "model_name": model.cfg.model_name,
            "collection_id": collection_id,
            "split": "pilot",
            "num_prompts": len(prompt_summaries),
            "exported_fields": manifest["dataset_fields"],
            "checkpoint_dir": str(checkpoint_dir),
            "manifest_path": str(manifest_path),
        },
    )
    return RouterDistillationPilotExportSummary(
        model_name=model.cfg.model_name,
        collection_id=collection_id,
        split="pilot",
        num_prompts=len(prompt_summaries),
        exported_fields=tuple(manifest["dataset_fields"]),
        checkpoint_dir=str(checkpoint_dir),
        manifest_path=str(manifest_path),
        prompt_summaries=tuple(prompt_summaries),
    )
