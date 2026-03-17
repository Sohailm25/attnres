# ABOUTME: Exposes the versioned prompt registry and pilot/confirm access helpers.
# ABOUTME: Centralizes prompt-split enforcement so confirmatory prompts stay non-exploratory.

from .registry import (
    ConfirmatoryAccessError,
    PromptCollection,
    PromptEntry,
    PromptRegistry,
    load_prompt_registry,
    perturb_prompt_entry,
    resolve_prompt_entries,
)

__all__ = [
    "ConfirmatoryAccessError",
    "PromptCollection",
    "PromptEntry",
    "PromptRegistry",
    "load_prompt_registry",
    "perturb_prompt_entry",
    "resolve_prompt_entries",
]
