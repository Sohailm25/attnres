# ABOUTME: Materializes resumable oracle-alpha campaign artifacts for prereg-scale predictiveness runs.
# ABOUTME: Persists prompt-level oracle checkpoints and feature caches so larger campaigns can resume cleanly.

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

from transformer_lens import HookedTransformer

from prompts import PromptEntry, resolve_prompt_entries
from .oracle_alpha_controls import load_oracle_alpha_control_registry
from .oracle_alpha_runner import (
    OracleAlphaPredictivenessSummary,
    OracleAlphaSequenceResult,
    build_oracle_alpha_predictiveness_summary,
    collect_feature_vectors,
    run_oracle_alpha_collection,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _safe_key(raw: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    if not normalized:
        normalized = "artifact"
    return f"{normalized}-{digest}"


def _load_sequence_result_cache(
    checkpoint_dir: Path,
) -> dict[str, OracleAlphaSequenceResult]:
    if not checkpoint_dir.is_dir():
        return {}

    loaded = {}
    for path in sorted(checkpoint_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        loaded[raw["prompt_id"]] = OracleAlphaSequenceResult(
            prompt_id=raw["prompt_id"],
            prompt=raw["prompt"],
            split=raw["split"],
            num_sources=int(raw["num_sources"]),
            source_labels=tuple(raw["source_labels"]),
            uniform_loss=float(raw["uniform_loss"]),
            optimized_loss=float(raw["optimized_loss"]),
            null_losses={
                key: float(value) for key, value in raw["null_losses"].items()
            },
            best_alpha_entropy=float(raw["best_alpha_entropy"]),
            best_alpha=tuple(float(value) for value in raw["best_alpha"]),
            final_alpha=tuple(float(value) for value in raw["final_alpha"]),
        )
    return loaded


def _save_sequence_result(
    checkpoint_dir: Path,
    result: OracleAlphaSequenceResult,
) -> None:
    path = checkpoint_dir / f"{_safe_key(result.prompt_id)}.json"
    _write_json(path, asdict(result))


def _load_feature_vector_cache(
    checkpoint_dir: Path,
) -> dict[str, list[float]]:
    if not checkpoint_dir.is_dir():
        return {}

    loaded = {}
    for path in sorted(checkpoint_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        loaded[raw["prompt_id"]] = [float(value) for value in raw["values"]]
    return loaded


def _save_feature_vector(
    checkpoint_dir: Path,
    entry: PromptEntry,
    values: Sequence[float],
) -> None:
    path = checkpoint_dir / f"{_safe_key(entry.prompt_id)}.json"
    _write_json(
        path,
        {
            "prompt_id": entry.prompt_id,
            "split": entry.split,
            "values": [float(value) for value in values],
        },
    )


def _materialize_oracle_run(
    *,
    model: HookedTransformer,
    collection_id: str,
    split: str,
    exploratory: bool,
    prompt_entries: Sequence[PromptEntry],
    checkpoint_dir: Path,
    optimization_steps: int,
    learning_rate: float,
    seed: int,
    prepend_bos: bool | None,
):
    existing_results = _load_sequence_result_cache(checkpoint_dir)
    return run_oracle_alpha_collection(
        model=model,
        collection_id=collection_id,
        split=split,
        exploratory=exploratory,
        prompt_entries=prompt_entries,
        optimization_steps=optimization_steps,
        learning_rate=learning_rate,
        seed=seed,
        prepend_bos=prepend_bos,
        existing_sequence_results=existing_results,
        on_sequence_result=lambda result: _save_sequence_result(checkpoint_dir, result),
    )


def _materialize_feature_vectors_by_source(
    *,
    model: HookedTransformer,
    prompt_entries: Sequence[PromptEntry],
    split: str,
    feature_sources: Sequence[str],
    checkpoint_root: Path,
    prepend_bos: bool | None,
) -> dict[str, list[list[float]]]:
    materialized = {}
    for feature_source in feature_sources:
        feature_dir = checkpoint_root / split / _safe_key(feature_source)
        existing_vectors = _load_feature_vector_cache(feature_dir)
        materialized[feature_source] = collect_feature_vectors(
            model=model,
            prompt_entries=prompt_entries,
            prepend_bos=prepend_bos,
            feature_source=feature_source,
            existing_feature_vectors=existing_vectors,
            on_feature_vector=lambda entry,
            vector,
            feature_dir=feature_dir: _save_feature_vector(
                feature_dir,
                entry,
                vector,
            ),
        )
    return materialized


def run_oracle_alpha_predictiveness_campaign(
    *,
    model: HookedTransformer,
    collection_id: str,
    output_dir: Path,
    max_train_sequences: int | None = None,
    max_eval_sequences: int | None = None,
    optimization_steps: int = 20,
    learning_rate: float = 0.1,
    seed: int = 0,
    regularization_grid: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    candidate_feature_sources: Sequence[str] | None = None,
    candidate_target_names: Sequence[str] | None = None,
    target_name_override: str | None = None,
    prepend_bos: bool | None = None,
) -> OracleAlphaPredictivenessSummary:
    control_registry = load_oracle_alpha_control_registry()
    control_plan = control_registry.plans[collection_id]
    predictiveness_plan = control_plan.predictiveness
    if target_name_override is not None and candidate_target_names is not None:
        raise ValueError(
            "target_name_override and candidate_target_names are mutually exclusive"
        )

    feature_sources = tuple(
        candidate_feature_sources
        or ("position_thirds_mean_pooled_h_4[t]_resid_post_layer_3_concat",)
    )
    target_names = tuple(
        candidate_target_names
        or (
            (target_name_override,)
            if target_name_override is not None
            else (predictiveness_plan.target,)
        )
    )
    if not feature_sources:
        raise ValueError("candidate_feature_sources must not be empty")
    if not target_names:
        raise ValueError("candidate_target_names must not be empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_entries = list(
        resolve_prompt_entries(
            collection_id=predictiveness_plan.train_collection_id,
            split=predictiveness_plan.train_split,
            exploratory=True,
        )
    )
    eval_entries = list(
        resolve_prompt_entries(
            collection_id=predictiveness_plan.eval_collection_id,
            split=predictiveness_plan.eval_split,
            exploratory=False,
        )
    )
    if max_train_sequences is not None:
        train_entries = train_entries[:max_train_sequences]
    if max_eval_sequences is not None:
        eval_entries = eval_entries[:max_eval_sequences]
    if len(train_entries) < 2:
        raise ValueError("predictiveness training requires at least two pilot prompts")
    if not eval_entries:
        raise ValueError("predictiveness evaluation requires confirm prompts")

    checkpoint_root = output_dir / "checkpoints"
    oracle_root = checkpoint_root / "oracle_runs"
    feature_root = checkpoint_root / "feature_vectors"
    train_run = _materialize_oracle_run(
        model=model,
        collection_id=collection_id,
        split=predictiveness_plan.train_split,
        exploratory=True,
        prompt_entries=train_entries,
        checkpoint_dir=oracle_root / predictiveness_plan.train_split,
        optimization_steps=optimization_steps,
        learning_rate=learning_rate,
        seed=seed,
        prepend_bos=prepend_bos,
    )
    _write_json(output_dir / "oracle_train_run.json", asdict(train_run))
    eval_run = _materialize_oracle_run(
        model=model,
        collection_id=collection_id,
        split=predictiveness_plan.eval_split,
        exploratory=False,
        prompt_entries=eval_entries,
        checkpoint_dir=oracle_root / predictiveness_plan.eval_split,
        optimization_steps=optimization_steps,
        learning_rate=learning_rate,
        seed=seed + 1000,
        prepend_bos=prepend_bos,
    )
    _write_json(output_dir / "oracle_eval_run.json", asdict(eval_run))

    train_feature_vectors_by_source = _materialize_feature_vectors_by_source(
        model=model,
        prompt_entries=train_entries,
        split=predictiveness_plan.train_split,
        feature_sources=feature_sources,
        checkpoint_root=feature_root,
        prepend_bos=prepend_bos,
    )
    eval_feature_vectors_by_source = _materialize_feature_vectors_by_source(
        model=model,
        prompt_entries=eval_entries,
        split=predictiveness_plan.eval_split,
        feature_sources=feature_sources,
        checkpoint_root=feature_root,
        prepend_bos=prepend_bos,
    )

    summary = build_oracle_alpha_predictiveness_summary(
        model=model,
        collection_id=collection_id,
        train_entries=train_entries,
        eval_entries=eval_entries,
        train_run=train_run,
        eval_run=eval_run,
        regularization_grid=regularization_grid,
        candidate_feature_sources=feature_sources,
        candidate_target_names=target_names,
        prepend_bos=prepend_bos,
        train_feature_vectors_by_source=train_feature_vectors_by_source,
        eval_feature_vectors_by_source=eval_feature_vectors_by_source,
    )
    _write_json(output_dir / "predictiveness_summary.json", asdict(summary))
    _write_json(
        output_dir / "campaign_manifest.json",
        {
            "collection_id": collection_id,
            "control_plan_id": control_plan.plan_id,
            "control_registry_id": control_registry.registry_id,
            "model_name": model.cfg.model_name,
            "output_dir": str(output_dir),
            "candidate_feature_sources": list(feature_sources),
            "candidate_target_names": list(target_names),
            "regularization_grid": [float(value) for value in regularization_grid],
            "optimization_steps": optimization_steps,
            "learning_rate": learning_rate,
            "seed": seed,
            "train_split": predictiveness_plan.train_split,
            "eval_split": predictiveness_plan.eval_split,
            "num_train_sequences": len(train_entries),
            "num_eval_sequences": len(eval_entries),
            "artifacts": {
                "oracle_train_run": "oracle_train_run.json",
                "oracle_eval_run": "oracle_eval_run.json",
                "predictiveness_summary": "predictiveness_summary.json",
                "oracle_checkpoints": str(oracle_root),
                "feature_checkpoints": str(feature_root),
            },
        },
    )
    return summary
