"""Load and validate DocVLM-Fingerprint annotations."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from schema import AnnotationValidationError, validate_annotations


ROOT_DIR = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = ROOT_DIR / "data" / "annotations.jsonl"


def load_annotations(path: Path = ANNOTATIONS_PATH) -> list[dict[str, Any]]:
    """Load newline-delimited JSON annotations."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnnotationValidationError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise AnnotationValidationError(f"line {line_number}: annotation must be a JSON object")
            records.append(record)
    return records


def main() -> int:
    """Validate the default annotation file and print counts by domain."""
    try:
        records = load_annotations()
        validate_annotations(records, ROOT_DIR)
    except (OSError, AnnotationValidationError) as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        return 1

    counts = Counter(record["domain"] for record in records)
    print(f"total_examples: {len(records)}")
    for domain in sorted(counts):
        print(f"{domain}: {counts[domain]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
