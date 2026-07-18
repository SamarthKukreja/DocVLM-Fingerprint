"""Dataset schema and validation helpers for DocVLM-Fingerprint."""

from __future__ import annotations

from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "domain",
    "image_path",
    "question",
    "answer",
    "evidence",
    "question_type",
}

ALLOWED_DOMAINS = {"chart", "ocr_doc", "scientific_figure"}


class AnnotationValidationError(ValueError):
    """Raised when an annotation record does not match the MVP schema."""


def validate_annotation(record: dict[str, Any], root_dir: Path) -> None:
    """Validate one annotation record.

    Args:
        record: Parsed JSON object for one dataset example.
        root_dir: Repository root used to resolve relative image paths.
    """
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        raise AnnotationValidationError(
            f"{record.get('id', '<missing id>')}: missing required field(s): {', '.join(missing)}"
        )

    record_id = record["id"]
    for field in REQUIRED_FIELDS - {"evidence"}:
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise AnnotationValidationError(f"{record_id}: {field} must be a non-empty string")

    if record["domain"] not in ALLOWED_DOMAINS:
        allowed = ", ".join(sorted(ALLOWED_DOMAINS))
        raise AnnotationValidationError(f"{record_id}: domain must be one of: {allowed}")

    evidence = record["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise AnnotationValidationError(f"{record_id}: evidence must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in evidence):
        raise AnnotationValidationError(f"{record_id}: evidence entries must be non-empty strings")

    image_path = root_dir / record["image_path"]
    if not image_path.exists():
        raise AnnotationValidationError(f"{record_id}: image_path does not exist: {record['image_path']}")


def validate_annotations(records: list[dict[str, Any]], root_dir: Path) -> None:
    """Validate a complete annotation list and fail on duplicate IDs."""
    seen_ids: set[str] = set()
    for record in records:
        record_id = record.get("id")
        if record_id in seen_ids:
            raise AnnotationValidationError(f"{record_id}: duplicate id")
        seen_ids.add(record_id)
        validate_annotation(record, root_dir)
