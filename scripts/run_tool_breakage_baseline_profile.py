# ABOUTME: Builds a saved-artifact baseline profile for Gemma tool-breakage runs grouped by prompt tags.
# ABOUTME: Reuses the baseline summary plus registry metadata so route-mode and family reads stay reproducible.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-path", default="prompts/registry_v5.yaml")
    parser.add_argument("--collection-id", default=None)
    parser.add_argument(
        "--summary-path",
        required=True,
    )
    parser.add_argument(
        "--group-tag-prefix",
        action="append",
        default=None,
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    from prompts.registry import load_prompt_registry
    from validation.tool_breakage_profile import (
        build_tool_breakage_baseline_tag_profile_summary,
    )

    args = parse_args()
    summary_payload = json.loads(Path(args.summary_path).read_text())
    collection_id = args.collection_id or summary_payload["collection_id"]
    tag_prefixes = tuple(args.group_tag_prefix or ["subcategory_", "route_mode_"])

    registry = load_prompt_registry(Path(args.registry_path))
    prompt_entries = registry.collections[collection_id].prompt_entries
    prompt_tags_by_id = {entry.prompt_id: entry.tags for entry in prompt_entries}

    profile_summary = build_tool_breakage_baseline_tag_profile_summary(
        prompt_results=summary_payload["prompt_results"],
        prompt_tags_by_id=prompt_tags_by_id,
        group_tag_prefixes=tag_prefixes,
    )
    payload = {
        "model_name": summary_payload["model_name"],
        "collection_id": collection_id,
        "split": summary_payload["split"],
        "exploratory": summary_payload["exploratory"],
        "summary_path": str(Path(args.summary_path)),
        "profile_summary": profile_summary,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
