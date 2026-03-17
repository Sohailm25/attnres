# ABOUTME: Loads the versioned prompt registry and enforces pilot/confirmatory access rules.
# ABOUTME: Keeps prompt metadata, nulls, and split discipline in one reusable source of truth.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = ROOT / "prompts" / "registry_v1.yaml"
ALLOWED_SPLITS = {"pilot", "confirm"}


class ConfirmatoryAccessError(RuntimeError):
    """Raised when confirmatory prompts are requested in exploratory mode."""


@dataclass(frozen=True)
class PromptEntry:
    prompt_id: str
    text: str
    split: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class PromptCollection:
    collection_id: str
    lane: str
    description: str
    dataset_source: str
    objective_families: tuple[str, ...]
    aggregation_unit: str
    null_models: tuple[str, ...]
    statistical_tests: tuple[str, ...]
    prompt_entries: tuple[PromptEntry, ...]


@dataclass(frozen=True)
class PromptRegistry:
    version: int
    registry_id: str
    created_on: str
    collections: dict[str, PromptCollection]


def _validate_collection(collection: PromptCollection) -> None:
    prompt_ids: set[str] = set()
    for entry in collection.prompt_entries:
        if entry.prompt_id in prompt_ids:
            raise ValueError(
                f"duplicate prompt id {entry.prompt_id!r} in {collection.collection_id}"
            )
        if entry.split not in ALLOWED_SPLITS:
            raise ValueError(
                f"unsupported split {entry.split!r} in {collection.collection_id}"
            )
        prompt_ids.add(entry.prompt_id)

    if not collection.objective_families:
        raise ValueError(f"{collection.collection_id} must define objective families")
    if collection.aggregation_unit != "sequence":
        raise ValueError(
            f"{collection.collection_id} must use sequence-level aggregation by default"
        )
    if not collection.null_models:
        raise ValueError(f"{collection.collection_id} must define null models")
    if not collection.statistical_tests:
        raise ValueError(f"{collection.collection_id} must define statistical tests")


def load_prompt_registry(path: Path | None = None) -> PromptRegistry:
    registry_path = path or DEFAULT_REGISTRY_PATH
    raw = yaml.safe_load(registry_path.read_text())

    defaults = raw["defaults"]
    collections: dict[str, PromptCollection] = {}
    for collection_id, collection_raw in raw["collections"].items():
        prompt_entries = tuple(
            PromptEntry(
                prompt_id=entry["id"],
                text=entry["text"],
                split=entry["split"],
                tags=tuple(entry.get("tags", [])),
            )
            for entry in collection_raw["prompts"]
        )
        collection = PromptCollection(
            collection_id=collection_id,
            lane=collection_raw["lane"],
            description=collection_raw["description"],
            dataset_source=collection_raw["dataset_source"],
            objective_families=tuple(collection_raw["objective_families"]),
            aggregation_unit=collection_raw.get(
                "aggregation_unit",
                defaults["aggregation_unit"],
            ),
            null_models=tuple(
                collection_raw.get("null_models", defaults["null_models"])
            ),
            statistical_tests=tuple(
                collection_raw.get(
                    "statistical_tests",
                    defaults["statistical_tests"],
                )
            ),
            prompt_entries=prompt_entries,
        )
        _validate_collection(collection)
        collections[collection_id] = collection

    return PromptRegistry(
        version=int(raw["version"]),
        registry_id=raw["registry_id"],
        created_on=raw["created_on"],
        collections=collections,
    )


def resolve_prompt_entries(
    *,
    collection_id: str,
    split: str,
    exploratory: bool,
    registry: PromptRegistry | None = None,
) -> tuple[PromptEntry, ...]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported split {split!r}")
    if split == "confirm" and exploratory:
        raise ConfirmatoryAccessError(
            "confirmatory prompts are locked for non-exploratory access only"
        )

    loaded_registry = registry or load_prompt_registry()
    try:
        collection = loaded_registry.collections[collection_id]
    except KeyError as error:
        raise KeyError(f"unknown prompt collection {collection_id!r}") from error

    entries = tuple(
        entry for entry in collection.prompt_entries if entry.split == split
    )
    if not entries:
        raise ValueError(
            f"prompt collection {collection_id!r} has no entries for split {split!r}"
        )
    return entries
