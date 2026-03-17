# ABOUTME: Exports prompt text from the saved registry for a requested lane and split.
# ABOUTME: Refuses confirmatory access when exploratory mode is enabled.

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
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--split", choices=("pilot", "confirm"), required=True)
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="Mark the request as exploratory. Confirmatory prompts reject this.",
    )
    return parser.parse_args()


def _entry_to_dict(entry: object) -> dict[str, object]:
    return {
        "prompt_id": entry.prompt_id,
        "split": entry.split,
        "text": entry.text,
        "tags": list(entry.tags),
    }


def main() -> int:
    from prompts import resolve_prompt_entries

    args = parse_args()
    entries = resolve_prompt_entries(
        collection_id=args.collection_id,
        split=args.split,
        exploratory=args.exploratory,
    )
    print(json.dumps([_entry_to_dict(entry) for entry in entries], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
